//! Native Windows NVENC encoder backed by Direct3D 11 textures.
//!
//! This path follows Sunshine's Windows/NVIDIA approach: capture stays in D3D11,
//! color conversion happens on the GPU, and NVENC receives registered DirectX
//! resources without routing frames through FFmpeg AVFrames.

#![cfg(target_os = "windows")]

use std::ffi::{c_void, CString};
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::Arc;

use windows::core::{Interface, GUID, PCSTR};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

use crate::capture::gpu_buffer::GpuBuffer;
use crate::encode::{EncoderConfig, VideoEncoder};
use crate::error::MediaError;
use crate::types::*;

const INPUT_RING_SIZE: usize = 4;
const NVENCAPI_MAJOR_VERSION: u32 = 12;
const NVENCAPI_MINOR_VERSION: u32 = 0;
const NVENCAPI_VERSION: u32 = NVENCAPI_MAJOR_VERSION | (NVENCAPI_MINOR_VERSION << 24);
const NV_ENC_SUCCESS: NvencStatus = 0;
const NV_ENC_ERR_NEED_MORE_INPUT: NvencStatus = 10;
const NV_ENC_DEVICE_TYPE_DIRECTX: u32 = 0;
const NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX: u32 = 0;
const NV_ENC_BUFFER_FORMAT_NV12: u32 = 1;
const NV_ENC_PIC_STRUCT_FRAME: u32 = 1;
const NV_ENC_PIC_TYPE_AUTOSELECT: u32 = 0;
const NV_ENC_PIC_TYPE_IDR: u32 = 3;
const NV_ENC_PIC_FLAG_FORCEIDR: u32 = 0x2;
const NV_ENC_PIC_FLAG_OUTPUT_SPSPPS: u32 = 0x4;

const NV_ENC_CODEC_H264_GUID: GUID = GUID::from_values(
    0x6bc82762,
    0x4e63,
    0x4ca4,
    [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf],
);

// NVENC accepts any supported preset GUID. We enumerate presets at runtime and
// use the first one rather than pinning to a newer SDK-only GUID.
const ZERO_GUID: GUID = GUID::from_values(0, 0, 0, [0; 8]);

type NvencStatus = u32;
type NvencHandle = *mut c_void;
type NvEncInputPtr = *mut c_void;
type NvEncOutputPtr = *mut c_void;
type NvEncRegisteredPtr = *mut c_void;

type NvEncodeApiCreateInstance =
    unsafe extern "system" fn(*mut NvEncodeApiFunctionList) -> NvencStatus;
type NvEncodeApiGetMaxSupportedVersion = unsafe extern "system" fn(*mut u32) -> NvencStatus;
type NvencGenericFn = *mut c_void;

fn nvenc_struct_version(version: u32) -> u32 {
    NVENCAPI_VERSION | (version << 16) | (0x7 << 28)
}

fn nvenc_extended_struct_version(version: u32) -> u32 {
    nvenc_struct_version(version) | (1 << 31)
}

macro_rules! nvenc_raw {
    ($api:expr, $field:ident($($arg:expr),*), $op:expr) => {{
        let f = $api.functions.$field.ok_or_else(|| {
            MediaError::EncodeError(format!("{} unavailable in NVENC function table", $op))
        })?;
        unsafe { f($($arg),*) }
    }};
}

fn nvenc_status(status: NvencStatus, op: &str) -> Result<(), MediaError> {
    if status == NV_ENC_SUCCESS {
        Ok(())
    } else {
        Err(MediaError::EncodeError(format!(
            "{} failed with NVENC status {}",
            op, status
        )))
    }
}

fn is_idr_annexb(data: &[u8]) -> bool {
    let mut i = 0usize;
    while i + 5 <= data.len() {
        let start_len = if data[i..].starts_with(&[0, 0, 0, 1]) {
            4
        } else if data[i..].starts_with(&[0, 0, 1]) {
            3
        } else {
            i += 1;
            continue;
        };
        if i + start_len < data.len() && (data[i + start_len] & 0x1f) == 5 {
            return true;
        }
        i += start_len;
    }
    false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NvEncodeApiFunctionList {
    version: u32,
    reserved: u32,
    nv_enc_open_encode_session: NvencGenericFn,
    nv_enc_get_encode_guid_count:
        Option<unsafe extern "system" fn(NvencHandle, *mut u32) -> NvencStatus>,
    nv_enc_get_encode_profile_guid_count: NvencGenericFn,
    nv_enc_get_encode_profile_guids: NvencGenericFn,
    nv_enc_get_encode_guids:
        Option<unsafe extern "system" fn(NvencHandle, *mut GUID, u32, *mut u32) -> NvencStatus>,
    nv_enc_get_input_format_count:
        Option<unsafe extern "system" fn(NvencHandle, GUID, *mut u32) -> NvencStatus>,
    nv_enc_get_input_formats: Option<
        unsafe extern "system" fn(NvencHandle, GUID, *mut u32, u32, *mut u32) -> NvencStatus,
    >,
    nv_enc_get_encode_caps: NvencGenericFn,
    nv_enc_get_encode_preset_count:
        Option<unsafe extern "system" fn(NvencHandle, GUID, *mut u32) -> NvencStatus>,
    nv_enc_get_encode_preset_guids: Option<
        unsafe extern "system" fn(NvencHandle, GUID, *mut GUID, u32, *mut u32) -> NvencStatus,
    >,
    nv_enc_get_encode_preset_config: NvencGenericFn,
    nv_enc_initialize_encoder:
        Option<unsafe extern "system" fn(NvencHandle, *mut NvEncInitializeParams) -> NvencStatus>,
    nv_enc_create_input_buffer: NvencGenericFn,
    nv_enc_destroy_input_buffer: NvencGenericFn,
    nv_enc_create_bitstream_buffer: Option<
        unsafe extern "system" fn(NvencHandle, *mut NvEncCreateBitstreamBuffer) -> NvencStatus,
    >,
    nv_enc_destroy_bitstream_buffer:
        Option<unsafe extern "system" fn(NvencHandle, NvEncOutputPtr) -> NvencStatus>,
    nv_enc_encode_picture:
        Option<unsafe extern "system" fn(NvencHandle, *mut NvEncPicParams) -> NvencStatus>,
    nv_enc_lock_bitstream:
        Option<unsafe extern "system" fn(NvencHandle, *mut NvEncLockBitstream) -> NvencStatus>,
    nv_enc_unlock_bitstream:
        Option<unsafe extern "system" fn(NvencHandle, NvEncOutputPtr) -> NvencStatus>,
    nv_enc_lock_input_buffer: NvencGenericFn,
    nv_enc_unlock_input_buffer: NvencGenericFn,
    nv_enc_get_encode_stats: NvencGenericFn,
    nv_enc_get_sequence_params: Option<
        unsafe extern "system" fn(NvencHandle, *mut NvEncSequenceParamPayload) -> NvencStatus,
    >,
    nv_enc_register_async_event: NvencGenericFn,
    nv_enc_unregister_async_event: NvencGenericFn,
    nv_enc_map_input_resource:
        Option<unsafe extern "system" fn(NvencHandle, *mut NvEncMapInputResource) -> NvencStatus>,
    nv_enc_unmap_input_resource:
        Option<unsafe extern "system" fn(NvencHandle, NvEncInputPtr) -> NvencStatus>,
    nv_enc_destroy_encoder: Option<unsafe extern "system" fn(NvencHandle) -> NvencStatus>,
    nv_enc_invalidate_ref_frames: NvencGenericFn,
    nv_enc_open_encode_session_ex: Option<
        unsafe extern "system" fn(
            *mut NvEncOpenEncodeSessionExParams,
            *mut NvencHandle,
        ) -> NvencStatus,
    >,
    nv_enc_register_resource:
        Option<unsafe extern "system" fn(NvencHandle, *mut NvEncRegisterResource) -> NvencStatus>,
    nv_enc_unregister_resource:
        Option<unsafe extern "system" fn(NvencHandle, NvEncRegisteredPtr) -> NvencStatus>,
    nv_enc_reconfigure_encoder: NvencGenericFn,
    reserved1: NvencGenericFn,
    nv_enc_create_mv_buffer: NvencGenericFn,
    nv_enc_destroy_mv_buffer: NvencGenericFn,
    nv_enc_run_motion_estimation_only: NvencGenericFn,
    reserved2: [NvencGenericFn; 281],
}

#[repr(C)]
struct NvEncOpenEncodeSessionExParams {
    version: u32,
    device_type: u32,
    device: *mut c_void,
    reserved: *mut c_void,
    api_version: u32,
    reserved1: [u32; 253],
    reserved2: [*mut c_void; 64],
}

#[repr(C)]
struct NvEncInitializeParams {
    version: u32,
    encode_guid: GUID,
    preset_guid: GUID,
    encode_width: u32,
    encode_height: u32,
    dar_width: u32,
    dar_height: u32,
    frame_rate_num: u32,
    frame_rate_den: u32,
    flags: u32,
    enable_output_in_vidmem: u32,
    priv_data_size: u32,
    priv_data: *mut c_void,
    encode_config: *mut c_void,
    max_encode_width: u32,
    max_encode_height: u32,
    max_me_hint_counts_per_block: [u32; 2],
    reserved: [u32; 289],
    reserved2: [*mut c_void; 64],
}

#[repr(C)]
struct NvEncCreateBitstreamBuffer {
    version: u32,
    size: u32,
    memory_heap: u32,
    reserved: u32,
    bitstream_buffer: NvEncOutputPtr,
    bitstream_buffer_ptr: *mut c_void,
    reserved1: [u32; 58],
    reserved2: [*mut c_void; 64],
}

#[repr(C)]
struct NvEncRegisterResource {
    version: u32,
    resource_type: u32,
    width: u32,
    height: u32,
    pitch: u32,
    sub_resource_index: u32,
    resource_to_register: *mut c_void,
    registered_resource: NvEncRegisteredPtr,
    buffer_format: u32,
    reserved1: [u32; 248],
    reserved2: [*mut c_void; 62],
}

#[repr(C)]
struct NvEncMapInputResource {
    version: u32,
    sub_resource_index: u32,
    input_resource: *mut c_void,
    registered_resource: NvEncRegisteredPtr,
    mapped_resource: NvEncInputPtr,
    mapped_buffer_fmt: u32,
    reserved1: [u32; 251],
    reserved2: [*mut c_void; 63],
}

#[repr(C)]
struct NvEncCodecPicParams {
    raw: [u8; 256],
}

#[repr(C)]
struct NvencExternalMeHintCountsPerBlockType {
    raw: [u32; 2],
}

#[repr(C)]
struct NvEncPicParams {
    version: u32,
    input_width: u32,
    input_height: u32,
    input_pitch: u32,
    encode_pic_flags: u32,
    frame_idx: u32,
    input_time_stamp: u64,
    input_duration: u64,
    input_buffer: NvEncInputPtr,
    output_bitstream: NvEncOutputPtr,
    completion_event: *mut c_void,
    buffer_fmt: u32,
    picture_struct: u32,
    picture_type: u32,
    codec_pic_params: NvEncCodecPicParams,
    me_hint_counts_per_block: [NvencExternalMeHintCountsPerBlockType; 2],
    me_external_hints: *mut c_void,
    reserved1: [u32; 6],
    reserved2: [*mut c_void; 2],
    qp_delta_map: *mut i8,
    qp_delta_map_size: u32,
    reserved_bit_fields: u32,
    me_hint_ref_pic_dist: [u16; 2],
    reserved3: [u32; 286],
    reserved4: [*mut c_void; 60],
}

#[repr(C)]
struct NvEncLockBitstream {
    version: u32,
    flags: u32,
    output_bitstream: NvEncOutputPtr,
    slice_offsets: *mut u32,
    frame_idx: u32,
    hw_encode_status: u32,
    num_slices: u32,
    bitstream_size_in_bytes: u32,
    output_time_stamp: u64,
    output_duration: u64,
    bitstream_buffer_ptr: *mut c_void,
    picture_type: u32,
    picture_struct: u32,
    frame_avg_qp: u32,
    frame_satd: u32,
    ltr_frame_idx: u32,
    ltr_frame_bitmap: u32,
    reserved: [u32; 236],
    reserved2: [*mut c_void; 64],
}

#[repr(C)]
struct NvEncSequenceParamPayload {
    version: u32,
    in_buffer_size: u32,
    sps_id: u32,
    pps_id: u32,
    spspps_buffer: *mut c_void,
    out_spspps_payload_size: *mut u32,
    reserved: [u32; 250],
    reserved2: [*mut c_void; 64],
}

struct NvencApi {
    _library: windows::Win32::Foundation::HMODULE,
    functions: NvEncodeApiFunctionList,
}

unsafe impl Send for NvencApi {}
unsafe impl Sync for NvencApi {}

impl NvencApi {
    fn load() -> Result<Arc<Self>, MediaError> {
        unsafe {
            let dll = CString::new("nvEncodeAPI64.dll").unwrap();
            let library = LoadLibraryA(PCSTR(dll.as_ptr() as *const u8)).map_err(|e| {
                MediaError::EncoderInitFailed(format!(
                    "LoadLibraryA(nvEncodeAPI64.dll) failed: {e}"
                ))
            })?;

            let get_max_name = CString::new("NvEncodeAPIGetMaxSupportedVersion").unwrap();
            let create_name = CString::new("NvEncodeAPICreateInstance").unwrap();
            let get_max = GetProcAddress(library, PCSTR(get_max_name.as_ptr() as *const u8))
                .ok_or_else(|| {
                    MediaError::EncoderInitFailed(
                        "NvEncodeAPIGetMaxSupportedVersion not found".into(),
                    )
                })?;
            let create = GetProcAddress(library, PCSTR(create_name.as_ptr() as *const u8))
                .ok_or_else(|| {
                    MediaError::EncoderInitFailed("NvEncodeAPICreateInstance not found".into())
                })?;

            let get_max: NvEncodeApiGetMaxSupportedVersion = std::mem::transmute(get_max);
            let create: NvEncodeApiCreateInstance = std::mem::transmute(create);

            let mut max_supported = 0u32;
            nvenc_status(
                get_max(&mut max_supported),
                "NvEncodeAPIGetMaxSupportedVersion",
            )?;
            let max_major = max_supported >> 4;
            if max_major < NVENCAPI_MAJOR_VERSION {
                return Err(MediaError::EncoderInitFailed(format!(
                    "NVIDIA driver NVENC API {} is older than required {}",
                    max_major, NVENCAPI_MAJOR_VERSION
                )));
            }

            let mut functions: NvEncodeApiFunctionList = std::mem::zeroed();
            functions.version = nvenc_struct_version(2);
            nvenc_status(create(&mut functions), "NvEncodeAPICreateInstance")?;

            Ok(Arc::new(Self {
                _library: library,
                functions,
            }))
        }
    }
}

struct InputSlot {
    texture: ID3D11Texture2D,
    registered: NvEncRegisteredPtr,
    bitstream: NvEncOutputPtr,
}

pub struct WindowsNvencEncoder {
    api: Arc<NvencApi>,
    encoder: NvencHandle,
    device: Option<ID3D11Device>,
    context: Option<ID3D11DeviceContext>,
    video_device: Option<ID3D11VideoDevice>,
    video_context: Option<ID3D11VideoContext>,
    video_processor: Option<ID3D11VideoProcessor>,
    video_enumerator: Option<ID3D11VideoProcessorEnumerator>,
    slots: Vec<InputSlot>,
    next_slot: usize,
    config: Option<EncoderConfig>,
    info: EncoderInfo,
    initialized: bool,
    force_keyframe: bool,
    frame_count: u64,
    sps_pps: Vec<u8>,
}

unsafe impl Send for WindowsNvencEncoder {}

impl WindowsNvencEncoder {
    pub fn new() -> Result<Self, MediaError> {
        let api = NvencApi::load()?;
        Ok(Self {
            api,
            encoder: ptr::null_mut(),
            device: None,
            context: None,
            video_device: None,
            video_context: None,
            video_processor: None,
            video_enumerator: None,
            slots: Vec::new(),
            next_slot: 0,
            config: None,
            info: EncoderInfo {
                name: "native_nvenc_d3d11".to_string(),
                hw_type: HwAccelType::Nvenc,
                supported_codecs: vec![VideoCodec::H264],
            },
            initialized: false,
            force_keyframe: false,
            frame_count: 0,
            sps_pps: Vec::new(),
        })
    }

    pub fn is_available() -> bool {
        Self::new().is_ok()
    }

    fn open_session(&mut self, device_ptr: usize) -> Result<(), MediaError> {
        unsafe {
            let mut params: NvEncOpenEncodeSessionExParams = std::mem::zeroed();
            params.version = nvenc_struct_version(1);
            params.device_type = NV_ENC_DEVICE_TYPE_DIRECTX;
            params.device = device_ptr as *mut c_void;
            params.api_version = NVENCAPI_VERSION;

            let mut encoder = ptr::null_mut();
            nvenc_status(
                nvenc_raw!(
                    self.api,
                    nv_enc_open_encode_session_ex(&mut params, &mut encoder),
                    "NvEncOpenEncodeSessionEx"
                ),
                "NvEncOpenEncodeSessionEx",
            )?;
            self.encoder = encoder;
            Ok(())
        }
    }

    fn check_h264_nv12_support(&self) -> Result<(), MediaError> {
        unsafe {
            let mut guid_count = 0u32;
            nvenc_status(
                nvenc_raw!(
                    self.api,
                    nv_enc_get_encode_guid_count(self.encoder, &mut guid_count),
                    "NvEncGetEncodeGUIDCount"
                ),
                "NvEncGetEncodeGUIDCount",
            )?;
            let mut guids = vec![ZERO_GUID; guid_count as usize];
            let mut written = 0u32;
            nvenc_status(
                nvenc_raw!(
                    self.api,
                    nv_enc_get_encode_guids(
                        self.encoder,
                        guids.as_mut_ptr(),
                        guid_count,
                        &mut written
                    ),
                    "NvEncGetEncodeGUIDs"
                ),
                "NvEncGetEncodeGUIDs",
            )?;
            if !guids
                .iter()
                .take(written as usize)
                .any(|g| *g == NV_ENC_CODEC_H264_GUID)
            {
                return Err(MediaError::EncoderInitFailed(
                    "NVENC H.264 GUID not supported".into(),
                ));
            }

            let mut fmt_count = 0u32;
            nvenc_status(
                nvenc_raw!(
                    self.api,
                    nv_enc_get_input_format_count(
                        self.encoder,
                        NV_ENC_CODEC_H264_GUID,
                        &mut fmt_count
                    ),
                    "NvEncGetInputFormatCount"
                ),
                "NvEncGetInputFormatCount",
            )?;
            let mut formats = vec![0u32; fmt_count as usize];
            let mut fmt_written = 0u32;
            nvenc_status(
                nvenc_raw!(
                    self.api,
                    nv_enc_get_input_formats(
                        self.encoder,
                        NV_ENC_CODEC_H264_GUID,
                        formats.as_mut_ptr(),
                        fmt_count,
                        &mut fmt_written
                    ),
                    "NvEncGetInputFormats"
                ),
                "NvEncGetInputFormats",
            )?;
            if !formats
                .iter()
                .take(fmt_written as usize)
                .any(|f| *f == NV_ENC_BUFFER_FORMAT_NV12)
            {
                return Err(MediaError::EncoderInitFailed(
                    "NVENC NV12 input not supported".into(),
                ));
            }
            Ok(())
        }
    }

    fn choose_preset(&self) -> Result<GUID, MediaError> {
        unsafe {
            let mut count = 0u32;
            nvenc_status(
                nvenc_raw!(
                    self.api,
                    nv_enc_get_encode_preset_count(
                        self.encoder,
                        NV_ENC_CODEC_H264_GUID,
                        &mut count
                    ),
                    "NvEncGetEncodePresetCount"
                ),
                "NvEncGetEncodePresetCount",
            )?;
            if count == 0 {
                return Err(MediaError::EncoderInitFailed(
                    "NVENC returned no H.264 presets".into(),
                ));
            }
            let mut presets = vec![ZERO_GUID; count as usize];
            let mut written = 0u32;
            nvenc_status(
                nvenc_raw!(
                    self.api,
                    nv_enc_get_encode_preset_guids(
                        self.encoder,
                        NV_ENC_CODEC_H264_GUID,
                        presets.as_mut_ptr(),
                        count,
                        &mut written
                    ),
                    "NvEncGetEncodePresetGUIDs"
                ),
                "NvEncGetEncodePresetGUIDs",
            )?;
            presets
                .into_iter()
                .next()
                .ok_or_else(|| MediaError::EncoderInitFailed("NVENC preset list was empty".into()))
        }
    }

    fn initialize_encoder(&mut self, config: &EncoderConfig) -> Result<(), MediaError> {
        let preset = self.choose_preset()?;
        unsafe {
            let mut params: NvEncInitializeParams = std::mem::zeroed();
            params.version = nvenc_extended_struct_version(7);
            params.encode_guid = NV_ENC_CODEC_H264_GUID;
            params.preset_guid = preset;
            params.encode_width = config.width;
            params.encode_height = config.height;
            params.dar_width = config.width;
            params.dar_height = config.height;
            params.frame_rate_num = config.fps.max(1);
            params.frame_rate_den = 1;
            params.flags = 1 << 1; // enablePTD: let NVENC choose picture types.
            params.max_encode_width = config.width;
            params.max_encode_height = config.height;

            nvenc_status(
                nvenc_raw!(
                    self.api,
                    nv_enc_initialize_encoder(self.encoder, &mut params),
                    "NvEncInitializeEncoder"
                ),
                "NvEncInitializeEncoder",
            )?;
            Ok(())
        }
    }

    fn create_input_texture(
        device: &ID3D11Device,
        width: u32,
        height: u32,
    ) -> Result<ID3D11Texture2D, MediaError> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
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
        for _ in 0..INPUT_RING_SIZE {
            let texture = Self::create_input_texture(device, width, height)?;
            let mut reg: NvEncRegisterResource = unsafe { std::mem::zeroed() };
            reg.version = nvenc_struct_version(3);
            reg.resource_type = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
            reg.width = width;
            reg.height = height;
            reg.pitch = 0;
            reg.sub_resource_index = 0;
            reg.resource_to_register = texture.as_raw() as *mut c_void;
            reg.buffer_format = NV_ENC_BUFFER_FORMAT_NV12;

            unsafe {
                nvenc_status(
                    nvenc_raw!(
                        self.api,
                        nv_enc_register_resource(self.encoder, &mut reg),
                        "NvEncRegisterResource"
                    ),
                    "NvEncRegisterResource",
                )?;
            }

            let mut bitstream: NvEncCreateBitstreamBuffer = unsafe { std::mem::zeroed() };
            bitstream.version = nvenc_struct_version(1);
            unsafe {
                nvenc_status(
                    nvenc_raw!(
                        self.api,
                        nv_enc_create_bitstream_buffer(self.encoder, &mut bitstream),
                        "NvEncCreateBitstreamBuffer"
                    ),
                    "NvEncCreateBitstreamBuffer",
                )?;
            }

            self.slots.push(InputSlot {
                texture,
                registered: reg.registered_resource,
                bitstream: bitstream.bitstream_buffer,
            });
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
            OutputWidth: config.width,
            OutputHeight: config.height,
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

    fn query_sps_pps(&mut self) -> Result<(), MediaError> {
        let mut buffer = vec![0u8; 512];
        let mut out_size = 0u32;
        let mut payload: NvEncSequenceParamPayload = unsafe { std::mem::zeroed() };
        payload.version = nvenc_struct_version(1);
        payload.in_buffer_size = buffer.len() as u32;
        payload.spspps_buffer = buffer.as_mut_ptr() as *mut c_void;
        payload.out_spspps_payload_size = &mut out_size;
        let status = nvenc_raw!(
            self.api,
            nv_enc_get_sequence_params(self.encoder, &mut payload),
            "NvEncGetSequenceParams"
        );
        if status == NV_ENC_SUCCESS && out_size > 0 && out_size as usize <= buffer.len() {
            buffer.truncate(out_size as usize);
            self.sps_pps = buffer;
            log::info!("Cached native NVENC SPS/PPS: {} bytes", self.sps_pps.len());
        } else {
            log::warn!(
                "NvEncGetSequenceParams failed or returned no payload: {}",
                status
            );
        }
        Ok(())
    }

    fn drain_slot(
        &mut self,
        slot_idx: usize,
        pts_us: u64,
    ) -> Result<Vec<EncodedVideoFrame>, MediaError> {
        let config = self
            .config
            .as_ref()
            .ok_or(MediaError::EncoderNotInitialized)?;
        let slot = &self.slots[slot_idx];
        let mut lock: NvEncLockBitstream = unsafe { std::mem::zeroed() };
        lock.version = nvenc_struct_version(1);
        lock.output_bitstream = slot.bitstream;

        unsafe {
            nvenc_status(
                nvenc_raw!(
                    self.api,
                    nv_enc_lock_bitstream(self.encoder, &mut lock),
                    "NvEncLockBitstream"
                ),
                "NvEncLockBitstream",
            )?;
        }

        let mut data = if !lock.bitstream_buffer_ptr.is_null() && lock.bitstream_size_in_bytes > 0 {
            unsafe {
                std::slice::from_raw_parts(
                    lock.bitstream_buffer_ptr as *const u8,
                    lock.bitstream_size_in_bytes as usize,
                )
                .to_vec()
            }
        } else {
            Vec::new()
        };

        unsafe {
            let _ = self
                .api
                .functions
                .nv_enc_unlock_bitstream
                .map(|f| f(self.encoder, slot.bitstream))
                .unwrap_or(NV_ENC_SUCCESS);
        }

        if data.is_empty() {
            return Ok(vec![]);
        }

        let is_key = lock.picture_type == NV_ENC_PIC_TYPE_IDR || is_idr_annexb(&data);
        if is_key && !self.sps_pps.is_empty() && !data.starts_with(&self.sps_pps) {
            let mut with_headers = Vec::with_capacity(self.sps_pps.len() + data.len());
            with_headers.extend_from_slice(&self.sps_pps);
            with_headers.extend_from_slice(&data);
            data = with_headers;
        }

        Ok(vec![EncodedVideoFrame {
            data,
            frame_type: if is_key {
                FrameType::Key
            } else {
                FrameType::Inter
            },
            pts: pts_us,
            duration: if config.fps > 0 {
                1_000_000 / config.fps as u64
            } else {
                0
            },
            codec: VideoCodec::H264,
        }])
    }
}

impl VideoEncoder for WindowsNvencEncoder {
    fn initialize(&mut self, config: &EncoderConfig) -> Result<(), MediaError> {
        if config.codec != VideoCodec::H264 {
            return Err(MediaError::EncoderInitFailed(
                "native Windows NVENC v1 supports H.264 only".into(),
            ));
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

        self.open_session(device_ptr)?;
        self.check_h264_nv12_support()?;
        self.initialize_encoder(config)?;
        self.create_slots(config.width, config.height)?;
        self.config = Some(config.clone());
        if let Err(err) = self.query_sps_pps() {
            log::warn!("Failed to query native NVENC SPS/PPS: {err}");
        }
        self.initialized = true;
        self.force_keyframe = true;
        self.frame_count = 0;

        log::info!(
            "Native Windows NVENC D3D11 initialized: {}x{} @{}fps {}kbps",
            config.width,
            config.height,
            config.fps,
            config.bitrate_kbps,
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
                    "native Windows NVENC requires GpuBuffer::D3D11Texture".into(),
                ))
            }
        };

        let slot_idx = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        let slot_texture = self.slots[slot_idx].texture.clone();
        self.convert_bgra_to_nv12(src_texture, src_index, &slot_texture)?;

        let slot = &self.slots[slot_idx];
        let mut map: NvEncMapInputResource = unsafe { std::mem::zeroed() };
        map.version = nvenc_struct_version(4);
        map.registered_resource = slot.registered;
        unsafe {
            nvenc_status(
                nvenc_raw!(
                    self.api,
                    nv_enc_map_input_resource(self.encoder, &mut map),
                    "NvEncMapInputResource"
                ),
                "NvEncMapInputResource",
            )?;
        }

        let mut pic: NvEncPicParams = unsafe { std::mem::zeroed() };
        let config = self.config.as_ref().unwrap();
        pic.version = nvenc_extended_struct_version(4);
        pic.input_width = config.width;
        pic.input_height = config.height;
        pic.input_pitch = 0;
        pic.frame_idx = self.frame_count as u32;
        pic.input_time_stamp = pts_us;
        pic.input_duration = if config.fps > 0 {
            1_000_000 / config.fps as u64
        } else {
            0
        };
        pic.input_buffer = map.mapped_resource;
        pic.output_bitstream = slot.bitstream;
        pic.buffer_fmt = map.mapped_buffer_fmt;
        pic.picture_struct = NV_ENC_PIC_STRUCT_FRAME;
        pic.picture_type = NV_ENC_PIC_TYPE_AUTOSELECT;
        if self.force_keyframe {
            pic.encode_pic_flags = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
        }

        let status = unsafe {
            nvenc_raw!(
                self.api,
                nv_enc_encode_picture(self.encoder, &mut pic),
                "NvEncEncodePicture"
            )
        };
        unsafe {
            let _ = self
                .api
                .functions
                .nv_enc_unmap_input_resource
                .map(|f| f(self.encoder, map.mapped_resource))
                .unwrap_or(NV_ENC_SUCCESS);
        }
        if status == NV_ENC_ERR_NEED_MORE_INPUT {
            self.frame_count += 1;
            self.force_keyframe = false;
            return Ok(vec![]);
        }
        nvenc_status(status, "NvEncEncodePicture")?;

        let frames = self.drain_slot(slot_idx, pts_us)?;
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
        log::warn!(
            "Native NVENC bitrate reconfigure is not wired yet; updated target to {}kbps and forcing IDR",
            bitrate_kbps
        );
        self.force_keyframe = true;
        Ok(())
    }

    fn encoder_info(&self) -> EncoderInfo {
        self.info.clone()
    }

    fn flush(&mut self) -> Result<Vec<EncodedVideoFrame>, MediaError> {
        Ok(vec![])
    }

    fn shutdown(&mut self) {
        if !self.encoder.is_null() {
            unsafe {
                for slot in self.slots.drain(..) {
                    if !slot.registered.is_null() {
                        let _ = self
                            .api
                            .functions
                            .nv_enc_unregister_resource
                            .map(|f| f(self.encoder, slot.registered))
                            .unwrap_or(NV_ENC_SUCCESS);
                    }
                    if !slot.bitstream.is_null() {
                        let _ = self
                            .api
                            .functions
                            .nv_enc_destroy_bitstream_buffer
                            .map(|f| f(self.encoder, slot.bitstream))
                            .unwrap_or(NV_ENC_SUCCESS);
                    }
                }
                let _ = self
                    .api
                    .functions
                    .nv_enc_destroy_encoder
                    .map(|f| f(self.encoder))
                    .unwrap_or(NV_ENC_SUCCESS);
            }
        }
        self.encoder = ptr::null_mut();
        self.device = None;
        self.context = None;
        self.video_device = None;
        self.video_context = None;
        self.video_processor = None;
        self.video_enumerator = None;
        self.initialized = false;
    }
}

impl Drop for WindowsNvencEncoder {
    fn drop(&mut self) {
        self.shutdown();
    }
}
