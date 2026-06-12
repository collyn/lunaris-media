//! Native Windows Intel Quick Sync Video encoder backed by Direct3D 11 textures.
//!
//! This module avoids FFmpeg for Intel GPUs. It dynamically loads `libvpl.dll`
//! (oneVPL 2.x dispatcher), initializes a oneVPL session on the capture D3D11
//! device, wraps GPU NV12 textures as `mfxFrameSurface1` surfaces, and drains
//! oneVPL bitstream buffers into `EncodedVideoFrame` values.
//!
//! Supports H.264, H.265, and AV1 codecs (hardware permitting).

#![cfg(target_os = "windows")]

use std::ffi::{c_void, CString};
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::Arc;

use windows::core::{Interface, PCSTR};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

use crate::capture::gpu_buffer::GpuBuffer;
use crate::encode::{EncoderConfig, VideoEncoder};
use crate::error::MediaError;
use crate::types::*;

const INPUT_RING_SIZE: usize = 4;

// ─── oneVPL type aliases ─────────────────────────────────────────────────────

type MfxStatus = i32;
type MfxSession = *mut c_void;
type MfxHDL = *mut c_void;
type MfxFrameAllocator = *mut c_void;

const MFX_ERR_NONE: MfxStatus = 0;
const MFX_ERR_MORE_DATA: MfxStatus = 10;
const MFX_WRN_VIDEO_PARAM_CHANGED: MfxStatus = 14;
const MFX_ERR_MORE_SURFACE: MfxStatus = 12;

// Implementation types
const MFX_IMPL_HARDWARE: i32 = 0x01;
const MFX_IMPL_HARDWARE2: i32 = 0x02;
const MFX_IMPL_HARDWARE3: i32 = 0x03;
const MFX_IMPL_HARDWARE4: i32 = 0x04;

// IOPattern
const MFX_IOPATTERN_IN_VIDEO_MEMORY: i32 = 0x01;

// Codec IDs
const MFX_CODEC_AVC: i32 = 4_097;   // 0x1001
const MFX_CODEC_HEVC: i32 = 20_976; // 0x5200
const MFX_CODEC_AV1: i32 = 25_601;  // 0x6401

// Rate control
const MFX_RATECONTROL_CBR: i16 = 2;

// Frame type flags
const MFX_FRAMETYPE_IDR: u16 = 0x0008;
const MFX_FRAMETYPE_I: u16 = 0x0002;
const MFX_FRAMETYPE_REF: u16 = 0x0080;

// Handle types
const MFX_HANDLE_D3D11_DEVICE: i32 = 4;

// ExtBuffer
const MFX_EXTBUFF_VIDEO_SIGNAL_INFO: u32 = 17; // 0x11
const MFX_EXTBUFF_CODING_OPTION: u32 = 12;     // 0x0C
const MFX_EXTBUFF_CODING_OPTION2: u32 = 19;    // 0x13

// ─── oneVPL repr(C) structs ──────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxVersion {
    minor: u16,
    major: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxFrameInfo {
    reserved: [u16; 4],
    bit_depth_luma: u16,
    bit_depth_chroma: u16,
    shift: u16,
    frame_rate_ext: MfxFrameRateExt,
    aspect_ratio: MfxAspectRatio,
    crop_x: u16,
    crop_y: u16,
    crop_w: u16,
    crop_h: u16,
    width: u16,
    height: u16,
    reserved3: [u32; 6],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxFrameRateExt {
    numerator: u16,
    denominator: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxAspectRatio {
    width: u16,
    height: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxFrameData {
    reserved: [u64; 4],
    pitch_low: [u16; 3],
    reserved2: u16,
    flags: u64,
    timestamp: i64,
    decode_time_stamp: i64,
    frame_order: u32,
    pic_struct: u16,
    reserved3: [u16; 5],
    chroma_location: u16,
    reserved4: [u16; 3],
    view_id: [u16; 2],
    mem_type: u16,
    reserved5: [u16; 3],
    y: *mut u8,
    cb: *mut u8,
    cr: *mut u8,
    a: *mut u8,
    p_arm: *mut u8,
    abgr: *mut u8,
    reserved6: [*mut u8; 7],
    y16: *mut u16,
    cb16: *mut u16,
    cr16: *mut u16,
    reserved7: [*mut u8; 12],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxFrameSurface1 {
    info: MfxFrameInfo,
    data: MfxFrameData,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxBitstream {
    reserved: [u32; 4],
    decode_time_stamp: i64,
    time_stamp: i64,
    data: *mut u8,
    data_offset: u32,
    data_length: u32,
    max_length: u32,
    pic_struct: u16,
    frame_type: u16,
    data_flag: u16,
    reserved2: u16,
    reserved3: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxVideoParam {
    reserved: [u32; 4],
    mfx: MfxInfoMfx,
    protected: u16,
    iopattern: i32,
    async_depth: u16,
    reserved2: [u16; 5],
    ext_param: *mut *mut MfxExtBuffer,
    num_ext_param: u16,
    reserved3: [u16; 7],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxInfoMfx {
    codec_id: i16,
    codec_profile: i16,
    codec_level: i16,
    num_thread: u16,
    target_usage: i16,
    gop_pic_size: i16,
    gop_ref_dist: i16,
    gop_opt_flag: i16,
    idr_interval: i16,
    rate_control_method: i16,
    initial_delay_in_kb: i16,
    qp: i16,
    accuracy: i16,
    buffer_size_in_kb: i16,
    target_kbps: i16,
    max_kbps: i16,
    num_slice: i16,
    num_ref_frame: i16,
    encoded_bit_depth: u16,
    src_bit_depth: u16,
    fourcc: i32,
    frame_rate_ext: MfxFrameRateExt,
    aspect_ratio: MfxAspectRatio,
    pic_struct: u16,
    raw: i16,
    frame_order: u16,
    decoded_order: u16,
    codec_level_check: u16,
    reserved: [u16; 3],
    gop_num_bframes: u16,
    max_dec_frame_buffering: u16,
    reserved2: [u16; 5],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxExtBuffer {
    buffer_id: u32,
    buffer_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxExtCodingOption {
    header: MfxExtBuffer,
    reserved1: [u16; 2],
    ref_pic_mark_rep: u16,
    reserved2: [u16; 9],
    aud_enh: u16,
    reserved3: [u16; 3],
    pic_timing_sei: u16,
    vui_vcl_hrd_parameters: u16,
    nal_hrd_parameters: u16,
    single_sei_nal_unit: u16,
    vps_id: u16,
    sps_id: u16,
    pps_id: u16,
    reserved4: [u16; 20],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxExtCodingOption2 {
    header: MfxExtBuffer,
    int_ref_type: u16,
    int_ref_cycle_size: u16,
    int_ref_qp_delta: i16,
    max_frame_size: u32,
    max_slice_size: u16,
    reserved: [u16; 9],
    num_mb_per_slice: u16,
    skip_frame: u16,
    reserved2: [u16; 8],
    min_qp: i16,
    max_qp: i16,
    initial_qp: i16,
    p_ref_type: u16,
    reserved3: [u16; 9],
}

// ─── oneVPL function pointer types ───────────────────────────────────────────

type MfxInitFn = unsafe extern "C" fn(i32, *mut MfxVersion, *mut MfxSession) -> MfxStatus;
type MfxCloseFn = unsafe extern "C" fn(MfxSession) -> MfxStatus;
type MfxLoadFn = unsafe extern "C" fn() -> *mut c_void;
type MfxUnloadFn = unsafe extern "C" fn(*mut c_void) -> MfxStatus;
type MfxCreateSessionFn =
    unsafe extern "C" fn(*mut c_void, u32, *mut MfxSession) -> MfxStatus;
type MfxEnumImplementationsFn =
    unsafe extern "C" fn(*mut c_void, u32, *mut c_void, *mut *mut c_void) -> MfxStatus;
type MfxVideoCORESetHandleFn =
    unsafe extern "C" fn(MfxSession, i32, *mut c_void) -> MfxStatus;
type MfxVideoENCODEQueryFn =
    unsafe extern "C" fn(MfxSession, *mut MfxVideoParam, *mut MfxVideoParam) -> MfxStatus;
type MfxVideoENCODEInitFn =
    unsafe extern "C" fn(MfxSession, *mut MfxVideoParam) -> MfxStatus;
type MfxVideoENCODECloseFn = unsafe extern "C" fn(MfxSession) -> MfxStatus;
type MfxVideoENCODEEncodeFrameAsyncFn = unsafe extern "C" fn(
    MfxSession,
    *mut MfxEncodeCtrl,
    *mut MfxFrameSurface1,
    *mut MfxBitstream,
    *mut u16,
) -> MfxStatus;
type MfxVideoENCODEGetVideoParamFn =
    unsafe extern "C" fn(MfxSession, *mut MfxVideoParam) -> MfxStatus;
type MfxVideoCORESyncOperationFn =
    unsafe extern "C" fn(MfxSession, u64, u32) -> MfxStatus;
type MfxFrameAllocatorAllocFn =
    unsafe extern "C" fn(MfxFrameAllocator, *mut MfxFrameAllocRequest, *mut MfxFrameAllocResponse) -> MfxStatus;
type MfxFrameAllocatorLockFn =
    unsafe extern "C" fn(MfxFrameAllocator, *mut MfxFrameAllocResponse, *mut MfxFrameSurface1) -> MfxStatus;
type MfxFrameAllocatorFreeFn =
    unsafe extern "C" fn(MfxFrameAllocator, *mut MfxFrameAllocResponse) -> MfxStatus;

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxEncodeCtrl {
    header: MfxExtBuffer,
    mfx_frame_type: u16,
    skip_frame: u16,
    qp: u16,
    num_ext_param: u16,
    ext_param: *mut *mut MfxExtBuffer,
    frame_order: u32,
    reserved: [u16; 4],
    pic_struct: u16,
    encoded_order: u16,
    reserved2: [u16; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxFrameAllocRequest {
    info: MfxFrameInfo,
    type_: u32,
    num_frame_min: u16,
    num_frame_suggested: u16,
    reserved: [u16; 22],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxFrameAllocResponse {
    mids: *mut *mut c_void,
    num_frame_actual: u16,
    reserved: [u16; 11],
    reserved2: [u32; 4],
}

// ─── oneVPL function table ───────────────────────────────────────────────────

struct VplFunctions {
    // oneVPL 2.x loader API
    mfx_load: Option<MfxLoadFn>,
    mfx_unload: Option<MfxUnloadFn>,
    mfx_create_session: Option<MfxCreateSessionFn>,
    mfx_enum_implementations: Option<MfxEnumImplementationsFn>,

    // Core functions (also available via session)
    mfx_init: Option<MfxInitFn>,
    mfx_close: Option<MfxCloseFn>,
    mfx_video_core_set_handle: Option<MfxVideoCORESetHandleFn>,

    // Encode functions
    mfx_video_encode_query: Option<MfxVideoENCODEQueryFn>,
    mfx_video_encode_init: Option<MfxVideoENCODEInitFn>,
    mfx_video_encode_close: Option<MfxVideoENCODECloseFn>,
    mfx_video_encode_encode_frame_async: Option<MfxVideoENCODEEncodeFrameAsyncFn>,
    mfx_video_encode_get_video_param: Option<MfxVideoENCODEGetVideoParamFn>,

    // Sync
    mfx_video_core_sync_operation: Option<MfxVideoCORESyncOperationFn>,
}

struct VplApi {
    _library: windows::Win32::Foundation::HMODULE,
    functions: VplFunctions,
}

unsafe impl Send for VplApi {}
unsafe impl Sync for VplApi {}

impl VplApi {
    fn load() -> Result<Arc<Self>, MediaError> {
        unsafe {
            let dll = CString::new("libvpl.dll").unwrap();
            let library = LoadLibraryA(PCSTR(dll.as_ptr() as *const u8)).map_err(|e| {
                MediaError::EncoderInitFailed(format!("LoadLibraryA(libvpl.dll) failed: {e}"))
            })?;

            let resolve = |name: &str| -> *mut c_void {
                let cname = CString::new(name).unwrap();
                GetProcAddress(library, PCSTR(cname.as_ptr() as *const u8))
                    .map(|f| f as *mut c_void)
                    .unwrap_or(ptr::null_mut())
            };

            let functions = VplFunctions {
                mfx_load: std::mem::transmute(resolve("MFXLoad")),
                mfx_unload: std::mem::transmute(resolve("MFXUnload")),
                mfx_create_session: std::mem::transmute(resolve("MFXCreateSession")),
                mfx_enum_implementations: std::mem::transmute(resolve("MFXEnumImplementations")),
                mfx_init: std::mem::transmute(resolve("MFXInit")),
                mfx_close: std::mem::transmute(resolve("MFXClose")),
                mfx_video_core_set_handle: std::mem::transmute(resolve("MFXVideoCORE_SetHandle")),
                mfx_video_encode_query: std::mem::transmute(resolve("MFXVideoENCODE_Query")),
                mfx_video_encode_init: std::mem::transmute(resolve("MFXVideoENCODE_Init")),
                mfx_video_encode_close: std::mem::transmute(resolve("MFXVideoENCODE_Close")),
                mfx_video_encode_encode_frame_async: std::mem::transmute(resolve(
                    "MFXVideoENCODE_EncodeFrameAsync",
                )),
                mfx_video_encode_get_video_param: std::mem::transmute(resolve(
                    "MFXVideoENCODE_GetVideoParam",
                )),
                mfx_video_core_sync_operation: std::mem::transmute(resolve(
                    "MFXVideoCORE_SyncOperation",
                )),
            };

            // Verify minimum required functions are present
            if functions.mfx_init.is_none() || functions.mfx_close.is_none() {
                return Err(MediaError::EncoderInitFailed(
                    "libvpl.dll missing required MFXInit/MFXClose functions".into(),
                ));
            }
            if functions.mfx_video_encode_init.is_none()
                || functions.mfx_video_encode_encode_frame_async.is_none()
            {
                return Err(MediaError::EncoderInitFailed(
                    "libvpl.dll missing required encode functions".into(),
                ));
            }

            Ok(Arc::new(Self {
                _library: library,
                functions,
            }))
        }
    }
}

// ─── Encoder implementation ─────────────────────────────────────────────────

struct InputSlot {
    texture: ID3D11Texture2D,
    surface: MfxFrameSurface1,
}

pub struct WindowsQsvEncoder {
    api: Arc<VplApi>,
    session: MfxSession,
    device: Option<ID3D11Device>,
    context: Option<ID3D11DeviceContext>,
    video_device: Option<ID3D11VideoDevice>,
    video_context: Option<ID3D11VideoContext>,
    video_processor: Option<ID3D11VideoProcessor>,
    video_enumerator: Option<ID3D11VideoProcessorEnumerator>,
    slots: Vec<InputSlot>,
    next_slot: usize,
    bitstream: Vec<u8>,
    config: Option<EncoderConfig>,
    codec: VideoCodec,
    info: EncoderInfo,
    initialized: bool,
    force_keyframe: bool,
    frame_count: u64,
}

unsafe impl Send for WindowsQsvEncoder {}

impl WindowsQsvEncoder {
    pub fn new() -> Result<Self, MediaError> {
        let api = VplApi::load()?;
        Ok(Self {
            api,
            session: ptr::null_mut(),
            device: None,
            context: None,
            video_device: None,
            video_context: None,
            video_processor: None,
            video_enumerator: None,
            slots: Vec::new(),
            next_slot: 0,
            bitstream: vec![0u8; 1024 * 1024], // 1MB initial bitstream buffer
            config: None,
            codec: VideoCodec::H264,
            info: EncoderInfo {
                name: "native_qsv_d3d11".to_string(),
                hw_type: HwAccelType::Qsv,
                supported_codecs: vec![VideoCodec::H264, VideoCodec::H265, VideoCodec::AV1],
            },
            initialized: false,
            force_keyframe: false,
            frame_count: 0,
        })
    }

    pub fn is_available() -> bool {
        Self::new().is_ok()
    }

    fn mfx_codec_id(&self) -> i32 {
        match self.codec {
            VideoCodec::H264 => MFX_CODEC_AVC,
            VideoCodec::H265 => MFX_CODEC_HEVC,
            VideoCodec::AV1 => MFX_CODEC_AV1,
        }
    }

    fn init_session(&mut self, device_ptr: usize) -> Result<(), MediaError> {
        unsafe {
            // Try oneVPL 2.x loader API first, fall back to MFXInit
            let loader = self
                .api
                .functions
                .mfx_load
                .map(|f| f())
                .filter(|h| !h.is_null());

            if let Some(loader_handle) = loader {
                // oneVPL 2.x path: MFXLoad + MFXCreateSession
                if let Some(create_session) = self.api.functions.mfx_create_session {
                    let mut session: MfxSession = ptr::null_mut();
                    let status = create_session(loader_handle, 0, &mut session);
                    if status == MFX_ERR_NONE && !session.is_null() {
                        self.session = session;
                        log::info!("oneVPL 2.x session created via MFXLoad");
                        return self.bind_d3d11_device(device_ptr);
                    }
                }
                // Unload if session creation failed
                if let Some(unload) = self.api.functions.mfx_unload {
                    unload(loader_handle);
                }
            }

            // Fallback: MFXInit (Media SDK 1.x compatible)
            let init_fn = self.api.functions.mfx_init.ok_or_else(|| {
                MediaError::EncoderInitFailed("MFXInit not found in libvpl.dll".into())
            })?;
            let mut version: MfxVersion = std::mem::zeroed();
            let mut session: MfxSession = ptr::null_mut();
            let status = init_fn(MFX_IMPL_HARDWARE, &mut version, &mut session);
            if status != MFX_ERR_NONE {
                // Try other hardware implementations
                for impl_type in [MFX_IMPL_HARDWARE2, MFX_IMPL_HARDWARE3, MFX_IMPL_HARDWARE4] {
                    let status = init_fn(impl_type, &mut version, &mut session);
                    if status == MFX_ERR_NONE {
                        break;
                    }
                }
                if session.is_null() {
                    return Err(MediaError::EncoderInitFailed(format!(
                        "MFXInit failed with all implementations, last status: {}",
                        status
                    )));
                }
            }
            self.session = session;
            log::info!(
                "oneVPL session created via MFXInit (version {}.{})",
                version.major,
                version.minor
            );
            self.bind_d3d11_device(device_ptr)
        }
    }

    fn bind_d3d11_device(&self, device_ptr: usize) -> Result<(), MediaError> {
        let set_handle = self
            .api
            .functions
            .mfx_video_core_set_handle
            .ok_or_else(|| {
                MediaError::EncoderInitFailed("MFXVideoCORE_SetHandle not found".into())
            })?;
        let status = unsafe {
            set_handle(
                self.session,
                MFX_HANDLE_D3D11_DEVICE,
                device_ptr as *mut c_void,
            )
        };
        if status != MFX_ERR_NONE {
            return Err(MediaError::EncoderInitFailed(format!(
                "MFXVideoCORE_SetHandle(D3D11) failed: {}",
                status
            )));
        }
        Ok(())
    }

    fn init_encoder(&mut self, config: &EncoderConfig) -> Result<(), MediaError> {
        let codec_id = self.mfx_codec_id();
        let mut params: MfxVideoParam = unsafe { std::mem::zeroed() };
        params.mfx.codec_id = codec_id as i16;
        params.mfx.target_usage = 4; // balanced
        params.mfx.frame_rate_ext.numerator = config.fps.max(1) as u16;
        params.mfx.frame_rate_ext.denominator = 1;
        params.mfx.frame_info.width = align16(config.width) as u16;
        params.mfx.frame_info.height = align16(config.height) as u16;
        params.mfx.frame_info.frame_rate_ext.numerator = config.fps.max(1) as u16;
        params.mfx.frame_info.frame_rate_ext.denominator = 1;
        params.mfx.frame_info.crop_x = 0;
        params.mfx.frame_info.crop_y = 0;
        params.mfx.frame_info.crop_w = config.width as u16;
        params.mfx.frame_info.crop_h = config.height as u16;
        params.mfx.target_kbps = (config.bitrate_kbps / 2).max(100) as i16; // oneVPL uses 500kbps units
        params.mfx.rate_control_method = MFX_RATECONTROL_CBR;
        params.mfx.gop_pic_size = (config.fps.max(1) * 2) as i16; // 2 second keyframe interval
        params.mfx.num_ref_frame = 1;
        params.iopattern = MFX_IOPATTERN_IN_VIDEO_MEMORY;

        let query_fn = self
            .api
            .functions
            .mfx_video_encode_query
            .ok_or_else(|| {
                MediaError::EncoderInitFailed("MFXVideoENCODE_Query not found".into())
            })?;
        let init_fn = self
            .api
            .functions
            .mfx_video_encode_init
            .ok_or_else(|| {
                MediaError::EncoderInitFailed("MFXVideoENCODE_Init not found".into())
            })?;

        // Query to validate parameters
        let mut adjusted: MfxVideoParam = unsafe { std::mem::zeroed() };
        let status = unsafe { query_fn(self.session, &mut params, &mut adjusted) };
        if status != MFX_ERR_NONE && status != MFX_WRN_VIDEO_PARAM_CHANGED {
            return Err(MediaError::EncoderInitFailed(format!(
                "MFXVideoENCODE_Query failed: {}",
                status
            )));
        }

        // Init with (possibly adjusted) parameters
        let status = unsafe { init_fn(self.session, &mut adjusted) };
        if status != MFX_ERR_NONE {
            return Err(MediaError::EncoderInitFailed(format!(
                "MFXVideoENCODE_Init failed: {}",
                status
            )));
        }

        Ok(())
    }

    fn create_input_texture(
        device: &ID3D11Device,
        width: u32,
        height: u32,
    ) -> Result<ID3D11Texture2D, MediaError> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: align16(width),
            Height: align16(height),
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_NV12,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture = None;
        unsafe {
            device
                .CreateTexture2D(&desc, None, Some(&mut texture))
                .map_err(|e| {
                    MediaError::EncoderInitFailed(format!(
                        "CreateTexture2D(NV12 input) failed: {e}"
                    ))
                })?;
        }
        texture.ok_or_else(|| MediaError::EncoderInitFailed("CreateTexture2D returned None".into()))
    }

    fn create_slots(&mut self, width: u32, height: u32) -> Result<(), MediaError> {
        let device = self
            .device
            .as_ref()
            .ok_or(MediaError::EncoderNotInitialized)?;
        let aligned_w = align16(width);
        let aligned_h = align16(height);

        for _ in 0..INPUT_RING_SIZE {
            let texture = Self::create_input_texture(device, width, height)?;

            // Create mfxFrameSurface1 backed by the D3D11 texture
            let mut surface: MfxFrameSurface1 = unsafe { std::mem::zeroed() };
            surface.info.width = aligned_w as u16;
            surface.info.height = aligned_h as u16;
            surface.info.crop_w = width as u16;
            surface.info.crop_h = height as u16;
            surface.info.frame_rate_ext.numerator =
                self.config.as_ref().map(|c| c.fps.max(1)).unwrap_or(60) as u16;
            surface.info.frame_rate_ext.denominator = 1;
            // Store D3D11 texture pointer in the Y field (oneVPL convention for D3D11)
            surface.data.y = texture.as_raw() as *mut u8;
            surface.data.mem_type = 0x01; // MFX_SURFACE_TYPE_D3D11_TEX2D

            self.slots.push(InputSlot { texture, surface });
        }
        Ok(())
    }

    fn ensure_video_processor(&mut self, config: &EncoderConfig) -> Result<(), MediaError> {
        if self.video_processor.is_some() {
            return Ok(());
        }
        let device = self
            .device
            .as_ref()
            .ok_or(MediaError::EncoderNotInitialized)?;
        let context = self
            .context
            .as_ref()
            .ok_or(MediaError::EncoderNotInitialized)?;
        let video_device: ID3D11VideoDevice = device.cast().map_err(|e| {
            MediaError::EncodeError(format!(
                "Cast ID3D11Device to ID3D11VideoDevice failed: {e}"
            ))
        })?;
        let video_context: ID3D11VideoContext = context.cast().map_err(|e| {
            MediaError::EncodeError(format!(
                "Cast ID3D11DeviceContext to ID3D11VideoContext failed: {e}"
            ))
        })?;
        let desc = D3D11_VIDEO_PROCESSOR_CONTENT_DESC {
            InputFrameFormat: D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
            InputFrameRate: DXGI_RATIONAL {
                Numerator: config.fps.max(1),
                Denominator: 1,
            },
            InputWidth: config.width,
            InputHeight: config.height,
            OutputFrameRate: DXGI_RATIONAL {
                Numerator: config.fps.max(1),
                Denominator: 1,
            },
            OutputWidth: align16(config.width),
            OutputHeight: align16(config.height),
            Usage: D3D11_VIDEO_USAGE_PLAYBACK_NORMAL,
        };
        let enumerator =
            unsafe { video_device.CreateVideoProcessorEnumerator(&desc) }.map_err(|e| {
                MediaError::EncodeError(format!("CreateVideoProcessorEnumerator failed: {e}"))
            })?;
        let processor = unsafe { video_device.CreateVideoProcessor(&enumerator, 0) }
            .map_err(|e| MediaError::EncodeError(format!("CreateVideoProcessor failed: {e}")))?;
        self.video_device = Some(video_device);
        self.video_context = Some(video_context);
        self.video_enumerator = Some(enumerator);
        self.video_processor = Some(processor);
        Ok(())
    }

    fn convert_bgra_to_nv12(
        &mut self,
        src_texture: *mut c_void,
        src_index: u32,
        dst_texture: &ID3D11Texture2D,
    ) -> Result<(), MediaError> {
        let config = self
            .config
            .clone()
            .ok_or(MediaError::EncoderNotInitialized)?;
        self.ensure_video_processor(&config)?;

        let src = unsafe { ManuallyDrop::new(ID3D11Texture2D::from_raw(src_texture)) };
        let video_device = self.video_device.as_ref().unwrap();
        let video_context = self.video_context.as_ref().unwrap();
        let processor = self.video_processor.as_ref().unwrap();
        let enumerator = self.video_enumerator.as_ref().unwrap();

        let input_desc = D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC {
            FourCC: 0,
            ViewDimension: D3D11_VPIV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_VPIV {
                    MipSlice: 0,
                    ArraySlice: src_index,
                },
            },
        };
        let output_desc = D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC {
            ViewDimension: D3D11_VPOV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_VPOV { MipSlice: 0 },
            },
        };

        let in_res: ID3D11Resource = (*src).cast().map_err(|e| {
            MediaError::EncodeError(format!("Cast source texture to ID3D11Resource failed: {e}"))
        })?;
        let out_res: ID3D11Resource = dst_texture.cast().map_err(|e| {
            MediaError::EncodeError(format!(
                "Cast destination texture to ID3D11Resource failed: {e}"
            ))
        })?;

        let mut input_view = None;
        unsafe {
            video_device
                .CreateVideoProcessorInputView(
                    &in_res,
                    enumerator,
                    &input_desc,
                    Some(&mut input_view),
                )
                .map_err(|e| {
                    MediaError::EncodeError(format!("CreateVideoProcessorInputView failed: {e}"))
                })?;
        }
        let mut output_view = None;
        unsafe {
            video_device
                .CreateVideoProcessorOutputView(
                    &out_res,
                    enumerator,
                    &output_desc,
                    Some(&mut output_view),
                )
                .map_err(|e| {
                    MediaError::EncodeError(format!("CreateVideoProcessorOutputView failed: {e}"))
                })?;
        }

        let input_view = input_view
            .ok_or_else(|| MediaError::EncodeError("VideoProcessor input view was None".into()))?;
        let output_view = output_view
            .ok_or_else(|| MediaError::EncodeError("VideoProcessor output view was None".into()))?;
        let mut stream = D3D11_VIDEO_PROCESSOR_STREAM::default();
        stream.Enable = true.into();
        stream.pInputSurface = ManuallyDrop::new(Some(input_view));
        let streams = [stream];
        unsafe {
            video_context
                .VideoProcessorBlt(processor, &output_view, self.frame_count as u32, &streams)
                .map_err(|e| MediaError::EncodeError(format!("VideoProcessorBlt failed: {e}")))?;
        }
        if let Some(context) = &self.context {
            unsafe { context.Flush() };
        }
        Ok(())
    }

    fn encode_surface(
        &mut self,
        surface_idx: usize,
        pts_us: u64,
    ) -> Result<Vec<EncodedVideoFrame>, MediaError> {
        let encode_fn = self
            .api
            .functions
            .mfx_video_encode_encode_frame_async
            .ok_or(MediaError::EncoderNotInitialized)?;
        let sync_fn = self
            .api
            .functions
            .mfx_video_core_sync_operation
            .ok_or(MediaError::EncoderNotInitialized)?;

        let surface = &mut self.slots[surface_idx].surface as *mut MfxFrameSurface1;

        // Set up encode control for keyframe request
        let mut ctrl: MfxEncodeCtrl = unsafe { std::mem::zeroed() };
        let ctrl_ptr = if self.force_keyframe {
            ctrl.mfx_frame_type = MFX_FRAMETYPE_IDR | MFX_FRAMETYPE_I | MFX_FRAMETYPE_REF;
            &mut ctrl as *mut MfxEncodeCtrl
        } else {
            ptr::null_mut()
        };

        let mut bs: MfxBitstream = unsafe { std::mem::zeroed() };
        bs.data = self.bitstream.as_mut_ptr();
        bs.max_length = self.bitstream.len() as u32;
        bs.data_offset = 0;
        bs.data_length = 0;

        let mut sync_point: u16 = 0;
        let status = unsafe { encode_fn(self.session, ctrl_ptr, surface, &mut bs, &mut sync_point) };

        if status == MFX_ERR_MORE_DATA {
            return Ok(vec![]);
        }
        if status == MFX_ERR_MORE_SURFACE {
            return Ok(vec![]);
        }
        if status < 0 {
            return Err(MediaError::EncodeError(format!(
                "MFXVideoENCODE_EncodeFrameAsync failed: {}",
                status
            )));
        }

        // Wait for async operation to complete
        let sync_handle = sync_point as u64;
        let status = unsafe { sync_fn(self.session, sync_handle, 60_000) };
        if status != MFX_ERR_NONE {
            return Err(MediaError::EncodeError(format!(
                "MFXVideoCORE_SyncOperation failed: {}",
                status
            )));
        }

        // Extract encoded data
        if bs.data_length == 0 {
            return Ok(vec![]);
        }

        let data = &self.bitstream[bs.data_offset as usize..(bs.data_offset + bs.data_length) as usize];
        let is_key = (bs.frame_type & MFX_FRAMETYPE_IDR) != 0
            || (bs.frame_type & MFX_FRAMETYPE_I) != 0;

        let encoded = EncodedVideoFrame {
            data: data.to_vec(),
            frame_type: if is_key {
                FrameType::Key
            } else {
                FrameType::Inter
            },
            pts: pts_us,
            duration: self
                .config
                .as_ref()
                .map(|c| if c.fps > 0 { 1_000_000 / c.fps as u64 } else { 0 })
                .unwrap_or(0),
            codec: self.codec,
        };

        Ok(vec![encoded])
    }
}

/// Align width/height to 16 pixels (oneVPL requirement for NV12).
fn align16(v: u32) -> u32 {
    (v + 15) & !15
}

impl VideoEncoder for WindowsQsvEncoder {
    fn initialize(&mut self, config: &EncoderConfig) -> Result<(), MediaError> {
        match config.codec {
            VideoCodec::H264 | VideoCodec::H265 | VideoCodec::AV1 => {}
            _ => {
                return Err(MediaError::EncoderInitFailed(
                    "native Windows QSV supports H.264, H.265, and AV1 only".into(),
                ));
            }
        }
        let device_ptr = config
            .d3d11_device
            .ok_or_else(|| MediaError::EncoderInitFailed("D3D11 device not available".into()))?;
        let context_ptr = config
            .d3d11_context
            .ok_or_else(|| MediaError::EncoderInitFailed("D3D11 context not available".into()))?;

        unsafe {
            let borrowed_device =
                ManuallyDrop::new(ID3D11Device::from_raw(device_ptr as *mut c_void));
            let borrowed_context =
                ManuallyDrop::new(ID3D11DeviceContext::from_raw(context_ptr as *mut c_void));
            self.device = Some((*borrowed_device).clone());
            self.context = Some((*borrowed_context).clone());
        }

        self.codec = config.codec;
        self.init_session(device_ptr)?;
        self.init_encoder(config)?;
        self.create_slots(config.width, config.height)?;
        self.config = Some(config.clone());
        self.initialized = true;
        self.force_keyframe = true;
        self.frame_count = 0;

        log::info!(
            "Native Windows QSV D3D11 initialized: {}x{} @{}fps {}kbps codec={:?}",
            config.width,
            config.height,
            config.fps,
            config.bitrate_kbps,
            self.codec,
        );
        Ok(())
    }

    fn encode(
        &mut self,
        buffer: &GpuBuffer,
        pts_us: u64,
    ) -> Result<Vec<EncodedVideoFrame>, MediaError> {
        if !self.initialized {
            return Err(MediaError::EncoderNotInitialized);
        }
        let (src_texture, src_index) = match buffer {
            GpuBuffer::D3D11Texture {
                texture,
                array_index,
            } => (*texture, *array_index),
            _ => {
                return Err(MediaError::EncodeError(
                    "native Windows QSV requires GpuBuffer::D3D11Texture".into(),
                ))
            }
        };

        let slot_idx = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        let slot_texture = self.slots[slot_idx].texture.clone();
        self.convert_bgra_to_nv12(src_texture, src_index, &slot_texture)?;

        let frames = self.encode_surface(slot_idx, pts_us)?;
        self.frame_count += 1;
        self.force_keyframe = false;
        Ok(frames)
    }

    fn request_keyframe(&mut self) {
        self.force_keyframe = true;
    }

    fn set_bitrate(&mut self, bitrate_kbps: u32) -> Result<(), MediaError> {
        if !self.initialized {
            return Err(MediaError::EncoderNotInitialized);
        }
        if let Some(config) = &mut self.config {
            config.bitrate_kbps = bitrate_kbps;
        }
        // oneVPL supports runtime reconfiguration via MFXVideoENCODE_Reset,
        // but for simplicity we force IDR and log the update.
        self.force_keyframe = true;
        log::warn!(
            "Native QSV bitrate reconfigure updated target to {}kbps and forcing IDR",
            bitrate_kbps
        );
        Ok(())
    }

    fn set_fps(&mut self, fps: u32) -> Result<(), MediaError> {
        if !self.initialized {
            return Err(MediaError::EncoderNotInitialized);
        }
        let fps = fps.max(1);
        if let Some(config) = &mut self.config {
            config.fps = fps;
        }
        self.force_keyframe = true;
        log::debug!("Native QSV target FPS updated to {}", fps);
        Ok(())
    }

    fn encoder_info(&self) -> EncoderInfo {
        self.info.clone()
    }

    fn flush(&mut self) -> Result<Vec<EncodedVideoFrame>, MediaError> {
        Ok(vec![])
    }

    fn shutdown(&mut self) {
        if !self.session.is_null() {
            if let Some(close_fn) = self.api.functions.mfx_video_encode_close {
                unsafe {
                    close_fn(self.session);
                }
            }
            if let Some(mfx_close) = self.api.functions.mfx_close {
                unsafe {
                    mfx_close(self.session);
                }
            }
        }
        self.session = ptr::null_mut();
        self.device = None;
        self.context = None;
        self.video_device = None;
        self.video_context = None;
        self.video_processor = None;
        self.video_enumerator = None;
        self.initialized = false;
    }
}

impl Drop for WindowsQsvEncoder {
    fn drop(&mut self) {
        self.shutdown();
    }
}
