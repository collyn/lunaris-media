//! Native Windows AMD AMF encoder backed by Direct3D 11 textures.
//!
//! This module avoids FFmpeg for AMD GPUs. It dynamically loads `amfrt64.dll`,
//! initializes AMF on the capture D3D11 device, wraps GPU NV12 textures as AMF
//! surfaces, and drains AMF buffers into `EncodedVideoFrame` values.

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
const AMF_VERSION_MAJOR: u64 = 1;
const AMF_VERSION_MINOR: u64 = 4;
const AMF_VERSION_RELEASE: u64 = 4;
const AMF_VERSION_BUILD_NUM: u64 = 0;
const AMF_FULL_VERSION: u64 = (AMF_VERSION_MAJOR << 48)
    | (AMF_VERSION_MINOR << 32)
    | (AMF_VERSION_RELEASE << 16)
    | AMF_VERSION_BUILD_NUM;

const AMF_OK: AmfResult = 0;
const AMF_ALREADY_INITIALIZED: AmfResult = 12;
const AMF_EOF: AmfResult = 23;
const AMF_REPEAT: AmfResult = 24;
const AMF_INPUT_FULL: AmfResult = 25;
const AMF_DX11_1: i32 = 111;
const AMF_SURFACE_NV12: i32 = 1;

const AMF_VIDEO_ENCODER_VCE_AVC: &[u16] = &[
    'A' as u16, 'M' as u16, 'F' as u16, 'V' as u16, 'i' as u16, 'd' as u16, 'e' as u16, 'o' as u16,
    'E' as u16, 'n' as u16, 'c' as u16, 'o' as u16, 'd' as u16, 'e' as u16, 'r' as u16, 'V' as u16,
    'C' as u16, 'E' as u16, '_' as u16, 'A' as u16, 'V' as u16, 'C' as u16, 0,
];

type AmfResult = i32;
type AmfBool = u8;
type AmfLong = i32;
type AmfSize = usize;
type AmfPts = i64;
type AmfInitFn = unsafe extern "C" fn(u64, *mut *mut AmfFactory) -> AmfResult;
type AmfQueryVersionFn = unsafe extern "C" fn(*mut u64) -> AmfResult;

#[repr(C)]
#[derive(Clone, Copy)]
struct AmfGuid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

const IID_AMF_BUFFER: AmfGuid = AmfGuid {
    data1: 0xb04b7248,
    data2: 0xb6f0,
    data3: 0x4321,
    data4: [0xb6, 0x91, 0xba, 0xa4, 0x74, 0x0f, 0x9f, 0xcb],
};

#[repr(C)]
#[derive(Clone, Copy)]
struct AmfSizeValue {
    width: i32,
    height: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AmfRateValue {
    num: u32,
    den: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
union AmfVariantValue {
    bool_value: AmfBool,
    int64_value: i64,
    size_value: AmfSizeValue,
    rate_value: AmfRateValue,
    raw: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AmfVariantStruct {
    variant_type: i32,
    value: AmfVariantValue,
}

const AMF_VARIANT_BOOL: i32 = 1;
const AMF_VARIANT_INT64: i32 = 2;
const AMF_VARIANT_SIZE: i32 = 5;
const AMF_VARIANT_RATE: i32 = 7;
const AMF_VIDEO_ENCODER_USAGE_TRANSCONDING: i64 = 0;
const AMF_VIDEO_ENCODER_USAGE_ULTRA_LOW_LATENCY: i64 = 1;
const AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR: i64 = 1;
const AMF_VIDEO_ENCODER_QUALITY_PRESET_BALANCED: i64 = 0;
const AMF_VIDEO_ENCODER_QUALITY_PRESET_SPEED: i64 = 1;
const AMF_VIDEO_ENCODER_OUTPUT_MODE_FRAME: i64 = 0;
const AMF_VIDEO_ENCODER_PICTURE_TYPE_IDR: i64 = 2;

fn align_to(value: u32, alignment: u32) -> u32 {
    ((value + alignment - 1) / alignment) * alignment
}

#[repr(C)]
struct AmfFactory {
    vtbl: *const AmfFactoryVtbl,
}

#[repr(C)]
struct AmfContext {
    vtbl: *const AmfContextVtbl,
}

#[repr(C)]
struct AmfComponent {
    vtbl: *const AmfComponentVtbl,
}

#[repr(C)]
struct AmfData {
    vtbl: *const AmfDataVtbl,
}

#[repr(C)]
struct AmfSurface {
    vtbl: *const AmfSurfaceVtbl,
}

#[repr(C)]
struct AmfBuffer {
    vtbl: *const AmfBufferVtbl,
}

#[repr(C)]
struct AmfFactoryVtbl {
    create_context: unsafe extern "system" fn(*mut AmfFactory, *mut *mut AmfContext) -> AmfResult,
    create_component: unsafe extern "system" fn(
        *mut AmfFactory,
        *mut AmfContext,
        *const u16,
        *mut *mut AmfComponent,
    ) -> AmfResult,
    set_cache_folder: *mut c_void,
    get_cache_folder: *mut c_void,
    get_debug: *mut c_void,
    get_trace: *mut c_void,
    get_programs: *mut c_void,
}

#[repr(C)]
struct AmfContextVtbl {
    acquire: unsafe extern "system" fn(*mut AmfContext) -> AmfLong,
    release: unsafe extern "system" fn(*mut AmfContext) -> AmfLong,
    query_interface:
        unsafe extern "system" fn(*mut AmfContext, *const AmfGuid, *mut *mut c_void) -> AmfResult,
    set_property:
        unsafe extern "system" fn(*mut AmfContext, *const u16, AmfVariantStruct) -> AmfResult,
    get_property:
        unsafe extern "system" fn(*mut AmfContext, *const u16, *mut AmfVariantStruct) -> AmfResult,
    has_property: unsafe extern "system" fn(*mut AmfContext, *const u16) -> AmfBool,
    get_property_count: unsafe extern "system" fn(*mut AmfContext) -> AmfSize,
    get_property_at: *mut c_void,
    clear: *mut c_void,
    add_to: *mut c_void,
    copy_to: *mut c_void,
    add_observer: *mut c_void,
    remove_observer: *mut c_void,
    terminate: unsafe extern "system" fn(*mut AmfContext) -> AmfResult,
    init_dx9: *mut c_void,
    get_dx9_device: *mut c_void,
    lock_dx9: *mut c_void,
    unlock_dx9: *mut c_void,
    init_dx11: unsafe extern "system" fn(*mut AmfContext, *mut c_void, i32) -> AmfResult,
    get_dx11_device: *mut c_void,
    lock_dx11: *mut c_void,
    unlock_dx11: *mut c_void,
    init_opencl: *mut c_void,
    get_opencl_context: *mut c_void,
    get_opencl_command_queue: *mut c_void,
    get_opencl_device_id: *mut c_void,
    get_opencl_factory: *mut c_void,
    init_opencl_ex: *mut c_void,
    lock_opencl: *mut c_void,
    unlock_opencl: *mut c_void,
    init_opengl: *mut c_void,
    get_opengl_context: *mut c_void,
    get_opengl_drawable: *mut c_void,
    lock_opengl: *mut c_void,
    unlock_opengl: *mut c_void,
    init_xv: *mut c_void,
    get_xv_device: *mut c_void,
    lock_xv: *mut c_void,
    unlock_xv: *mut c_void,
    init_gralloc: *mut c_void,
    get_gralloc_device: *mut c_void,
    lock_gralloc: *mut c_void,
    unlock_gralloc: *mut c_void,
    alloc_buffer: *mut c_void,
    alloc_surface: *mut c_void,
    alloc_audio_buffer: *mut c_void,
    create_buffer_from_host_native: *mut c_void,
    create_surface_from_host_native: *mut c_void,
    create_surface_from_dx9_native: *mut c_void,
    create_surface_from_dx11_native: unsafe extern "system" fn(
        *mut AmfContext,
        *mut c_void,
        *mut *mut AmfSurface,
        *mut c_void,
    ) -> AmfResult,
    create_surface_from_opengl_native: *mut c_void,
    create_surface_from_gralloc_native: *mut c_void,
    create_surface_from_opencl_native: *mut c_void,
    create_buffer_from_opencl_native: *mut c_void,
    get_compute: *mut c_void,
}

#[repr(C)]
struct AmfComponentVtbl {
    acquire: unsafe extern "system" fn(*mut AmfComponent) -> AmfLong,
    release: unsafe extern "system" fn(*mut AmfComponent) -> AmfLong,
    query_interface:
        unsafe extern "system" fn(*mut AmfComponent, *const AmfGuid, *mut *mut c_void) -> AmfResult,
    set_property:
        unsafe extern "system" fn(*mut AmfComponent, *const u16, AmfVariantStruct) -> AmfResult,
    get_property: unsafe extern "system" fn(
        *mut AmfComponent,
        *const u16,
        *mut AmfVariantStruct,
    ) -> AmfResult,
    has_property: unsafe extern "system" fn(*mut AmfComponent, *const u16) -> AmfBool,
    get_property_count: unsafe extern "system" fn(*mut AmfComponent) -> AmfSize,
    get_property_at: *mut c_void,
    clear: *mut c_void,
    add_to: *mut c_void,
    copy_to: *mut c_void,
    add_observer: *mut c_void,
    remove_observer: *mut c_void,
    get_properties_info_count: *mut c_void,
    get_property_info_at: *mut c_void,
    get_property_info: *mut c_void,
    validate_property: *mut c_void,
    init: unsafe extern "system" fn(*mut AmfComponent, i32, i32, i32) -> AmfResult,
    reinit: unsafe extern "system" fn(*mut AmfComponent, i32, i32) -> AmfResult,
    terminate: unsafe extern "system" fn(*mut AmfComponent) -> AmfResult,
    drain: unsafe extern "system" fn(*mut AmfComponent) -> AmfResult,
    flush: unsafe extern "system" fn(*mut AmfComponent) -> AmfResult,
    submit_input: unsafe extern "system" fn(*mut AmfComponent, *mut AmfData) -> AmfResult,
    query_output: unsafe extern "system" fn(*mut AmfComponent, *mut *mut AmfData) -> AmfResult,
    get_context: *mut c_void,
    set_output_data_allocator_cb: *mut c_void,
    get_caps: *mut c_void,
    optimize: *mut c_void,
}

#[repr(C)]
struct AmfDataVtbl {
    acquire: unsafe extern "system" fn(*mut AmfData) -> AmfLong,
    release: unsafe extern "system" fn(*mut AmfData) -> AmfLong,
    query_interface:
        unsafe extern "system" fn(*mut AmfData, *const AmfGuid, *mut *mut c_void) -> AmfResult,
    set_property:
        unsafe extern "system" fn(*mut AmfData, *const u16, AmfVariantStruct) -> AmfResult,
    get_property: *mut c_void,
    has_property: *mut c_void,
    get_property_count: *mut c_void,
    get_property_at: *mut c_void,
    clear: *mut c_void,
    add_to: *mut c_void,
    copy_to: *mut c_void,
    add_observer: *mut c_void,
    remove_observer: *mut c_void,
    get_memory_type: *mut c_void,
    duplicate: *mut c_void,
    convert: *mut c_void,
    interop: *mut c_void,
    get_data_type: *mut c_void,
    is_reusable: *mut c_void,
    set_pts: unsafe extern "system" fn(*mut AmfData, AmfPts),
    get_pts: unsafe extern "system" fn(*mut AmfData) -> AmfPts,
    set_duration: unsafe extern "system" fn(*mut AmfData, AmfPts),
    get_duration: unsafe extern "system" fn(*mut AmfData) -> AmfPts,
}

#[repr(C)]
struct AmfSurfaceVtbl {
    acquire: unsafe extern "system" fn(*mut AmfSurface) -> AmfLong,
    release: unsafe extern "system" fn(*mut AmfSurface) -> AmfLong,
    query_interface:
        unsafe extern "system" fn(*mut AmfSurface, *const AmfGuid, *mut *mut c_void) -> AmfResult,
    set_property:
        unsafe extern "system" fn(*mut AmfSurface, *const u16, AmfVariantStruct) -> AmfResult,
    get_property: *mut c_void,
    has_property: *mut c_void,
    get_property_count: *mut c_void,
    get_property_at: *mut c_void,
    clear: *mut c_void,
    add_to: *mut c_void,
    copy_to: *mut c_void,
    add_observer: *mut c_void,
    remove_observer: *mut c_void,
    get_memory_type: *mut c_void,
    duplicate: *mut c_void,
    convert: *mut c_void,
    interop: *mut c_void,
    get_data_type: *mut c_void,
    is_reusable: *mut c_void,
    set_pts: unsafe extern "system" fn(*mut AmfSurface, AmfPts),
    get_pts: unsafe extern "system" fn(*mut AmfSurface) -> AmfPts,
    set_duration: unsafe extern "system" fn(*mut AmfSurface, AmfPts),
    get_duration: unsafe extern "system" fn(*mut AmfSurface) -> AmfPts,
    get_format: *mut c_void,
    get_planes_count: *mut c_void,
    get_plane_at: *mut c_void,
    get_plane: *mut c_void,
    get_frame_type: *mut c_void,
    set_frame_type: *mut c_void,
    set_crop: unsafe extern "system" fn(*mut AmfSurface, i32, i32, i32, i32) -> AmfResult,
    copy_surface_region: *mut c_void,
    add_observer_surface: *mut c_void,
    remove_observer_surface: *mut c_void,
}

#[repr(C)]
struct AmfBufferVtbl {
    acquire: unsafe extern "system" fn(*mut AmfBuffer) -> AmfLong,
    release: unsafe extern "system" fn(*mut AmfBuffer) -> AmfLong,
    query_interface:
        unsafe extern "system" fn(*mut AmfBuffer, *const AmfGuid, *mut *mut c_void) -> AmfResult,
    set_property:
        unsafe extern "system" fn(*mut AmfBuffer, *const u16, AmfVariantStruct) -> AmfResult,
    get_property: *mut c_void,
    has_property: *mut c_void,
    get_property_count: *mut c_void,
    get_property_at: *mut c_void,
    clear: *mut c_void,
    add_to: *mut c_void,
    copy_to: *mut c_void,
    add_observer: *mut c_void,
    remove_observer: *mut c_void,
    get_memory_type: *mut c_void,
    duplicate: *mut c_void,
    convert: *mut c_void,
    interop: *mut c_void,
    get_data_type: *mut c_void,
    is_reusable: *mut c_void,
    set_pts: unsafe extern "system" fn(*mut AmfBuffer, AmfPts),
    get_pts: unsafe extern "system" fn(*mut AmfBuffer) -> AmfPts,
    set_duration: unsafe extern "system" fn(*mut AmfBuffer, AmfPts),
    get_duration: unsafe extern "system" fn(*mut AmfBuffer) -> AmfPts,
    set_size: *mut c_void,
    get_size: unsafe extern "system" fn(*mut AmfBuffer) -> AmfSize,
    get_native: unsafe extern "system" fn(*mut AmfBuffer) -> *mut c_void,
    add_observer_buffer: *mut c_void,
    remove_observer_buffer: *mut c_void,
}

fn amf_ok(status: AmfResult, op: &str) -> Result<(), MediaError> {
    if status == AMF_OK || status == AMF_ALREADY_INITIALIZED {
        Ok(())
    } else {
        Err(MediaError::EncodeError(format!(
            "{} failed with AMF status {}",
            op, status
        )))
    }
}

fn amf_wide(name: &str) -> Vec<u16> {
    name.encode_utf16().chain(std::iter::once(0)).collect()
}

fn amf_variant_bool(value: bool) -> AmfVariantStruct {
    AmfVariantStruct {
        variant_type: AMF_VARIANT_BOOL,
        value: AmfVariantValue {
            bool_value: if value { 1 } else { 0 },
        },
    }
}

fn amf_variant_int64(value: i64) -> AmfVariantStruct {
    AmfVariantStruct {
        variant_type: AMF_VARIANT_INT64,
        value: AmfVariantValue { int64_value: value },
    }
}

fn amf_variant_size(width: u32, height: u32) -> AmfVariantStruct {
    AmfVariantStruct {
        variant_type: AMF_VARIANT_SIZE,
        value: AmfVariantValue {
            size_value: AmfSizeValue {
                width: width as i32,
                height: height as i32,
            },
        },
    }
}

fn amf_variant_rate(num: u32, den: u32) -> AmfVariantStruct {
    AmfVariantStruct {
        variant_type: AMF_VARIANT_RATE,
        value: AmfVariantValue {
            rate_value: AmfRateValue {
                num,
                den: den.max(1),
            },
        },
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

struct AmfApi {
    _library: windows::Win32::Foundation::HMODULE,
    factory: *mut AmfFactory,
    runtime_version: u64,
}

unsafe impl Send for AmfApi {}
unsafe impl Sync for AmfApi {}

impl AmfApi {
    fn load() -> Result<Arc<Self>, MediaError> {
        unsafe {
            let dll = CString::new("amfrt64.dll").unwrap();
            let library = LoadLibraryA(PCSTR(dll.as_ptr() as *const u8)).map_err(|e| {
                MediaError::EncoderInitFailed(format!("LoadLibraryA(amfrt64.dll) failed: {e}"))
            })?;
            let init_name = CString::new("AMFInit").unwrap();
            let query_name = CString::new("AMFQueryVersion").unwrap();
            let init = GetProcAddress(library, PCSTR(init_name.as_ptr() as *const u8))
                .ok_or_else(|| MediaError::EncoderInitFailed("AMFInit not found".into()))?;
            let query = GetProcAddress(library, PCSTR(query_name.as_ptr() as *const u8))
                .ok_or_else(|| MediaError::EncoderInitFailed("AMFQueryVersion not found".into()))?;
            let init: AmfInitFn = std::mem::transmute(init);
            let query: AmfQueryVersionFn = std::mem::transmute(query);
            let mut runtime_version = 0u64;
            amf_ok(query(&mut runtime_version), "AMFQueryVersion")?;
            let mut factory = ptr::null_mut();
            amf_ok(init(AMF_FULL_VERSION, &mut factory), "AMFInit")?;
            if factory.is_null() {
                return Err(MediaError::EncoderInitFailed(
                    "AMFInit returned null factory".into(),
                ));
            }
            Ok(Arc::new(Self {
                _library: library,
                factory,
                runtime_version,
            }))
        }
    }
}

struct InputSlot {
    texture: ID3D11Texture2D,
    output_view: Option<ID3D11VideoProcessorOutputView>,
}

pub struct WindowsAmfEncoder {
    api: Arc<AmfApi>,
    context: *mut AmfContext,
    component: *mut AmfComponent,
    device: Option<ID3D11Device>,
    d3d_context: Option<ID3D11DeviceContext>,
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
    cached_input_texture: usize,
    cached_input_index: u32,
    cached_input_view: Option<ID3D11VideoProcessorInputView>,
    flush_each_frame: bool,
}

unsafe impl Send for WindowsAmfEncoder {}

impl WindowsAmfEncoder {
    pub fn new() -> Result<Self, MediaError> {
        let api = AmfApi::load()?;
        Ok(Self {
            api,
            context: ptr::null_mut(),
            component: ptr::null_mut(),
            device: None,
            d3d_context: None,
            video_device: None,
            video_context: None,
            video_processor: None,
            video_enumerator: None,
            slots: Vec::new(),
            next_slot: 0,
            config: None,
            info: EncoderInfo {
                name: "native_amf_d3d11".to_string(),
                hw_type: HwAccelType::Amf,
                supported_codecs: vec![VideoCodec::H264],
            },
            initialized: false,
            force_keyframe: false,
            frame_count: 0,
            cached_input_texture: 0,
            cached_input_index: 0,
            cached_input_view: None,
            flush_each_frame: std::env::var("LUNARIS_AMF_FLUSH_EACH_FRAME")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false),
        })
    }

    pub fn is_available() -> bool {
        Self::new().is_ok()
    }

    fn create_context(&mut self, device_ptr: usize) -> Result<(), MediaError> {
        unsafe {
            let factory = &*self.api.factory;
            let mut context = ptr::null_mut();
            amf_ok(
                ((*factory.vtbl).create_context)(self.api.factory, &mut context),
                "AMFFactory::CreateContext",
            )?;
            if context.is_null() {
                return Err(MediaError::EncoderInitFailed(
                    "AMF CreateContext returned null".into(),
                ));
            }
            amf_ok(
                ((*(*context).vtbl).init_dx11)(context, device_ptr as *mut c_void, AMF_DX11_1),
                "AMFContext::InitDX11",
            )?;
            self.context = context;
            Ok(())
        }
    }

    fn create_component(&mut self, config: &EncoderConfig) -> Result<(), MediaError> {
        unsafe {
            let factory = &*self.api.factory;
            let mut component = ptr::null_mut();
            amf_ok(
                ((*factory.vtbl).create_component)(
                    self.api.factory,
                    self.context,
                    AMF_VIDEO_ENCODER_VCE_AVC.as_ptr(),
                    &mut component,
                ),
                "AMFFactory::CreateComponent(AMFVideoEncoderVCE_AVC)",
            )?;
            if component.is_null() {
                return Err(MediaError::EncoderInitFailed(
                    "AMF CreateComponent returned null".into(),
                ));
            }
            self.configure_component(component, config)?;
            amf_ok(
                ((*(*component).vtbl).init)(
                    component,
                    AMF_SURFACE_NV12,
                    config.width as i32,
                    config.height as i32,
                ),
                "AMFComponent::Init",
            )?;
            self.component = component;
            Ok(())
        }
    }

    fn set_component_property(
        &self,
        component: *mut AmfComponent,
        name: &str,
        value: AmfVariantStruct,
    ) -> Result<(), MediaError> {
        let wide = amf_wide(name);
        unsafe {
            amf_ok(
                ((*(*component).vtbl).set_property)(component, wide.as_ptr(), value),
                &format!("AMFComponent::SetProperty({name})"),
            )
        }
    }

    fn try_set_component_property(
        &self,
        component: *mut AmfComponent,
        name: &str,
        value: AmfVariantStruct,
    ) {
        if let Err(err) = self.set_component_property(component, name, value) {
            log::warn!("Native AMF optional property {name} was not accepted: {err}");
        }
    }

    fn configure_component(
        &self,
        component: *mut AmfComponent,
        config: &EncoderConfig,
    ) -> Result<(), MediaError> {
        let usage = if config.low_latency {
            AMF_VIDEO_ENCODER_USAGE_ULTRA_LOW_LATENCY
        } else {
            AMF_VIDEO_ENCODER_USAGE_TRANSCONDING
        };
        let idr_period = if config.keyframe_interval > 0 {
            config.keyframe_interval
        } else {
            config.fps.max(1) * 2
        };
        let bitrate = config.bitrate_kbps as i64 * 1000;
        log::info!(
            "Native AMF: configuring component (usage={}, size={}x{}, fps={}, bitrate={}kbps)",
            usage,
            config.width,
            config.height,
            config.fps.max(1),
            config.bitrate_kbps
        );
        self.set_component_property(component, "Usage", amf_variant_int64(usage))?;
        self.set_component_property(
            component,
            "FrameSize",
            amf_variant_size(config.width, config.height),
        )?;
        self.set_component_property(
            component,
            "FrameRate",
            amf_variant_rate(config.fps.max(1), 1),
        )?;
        self.try_set_component_property(component, "TargetBitrate", amf_variant_int64(bitrate));
        self.try_set_component_property(component, "PeakBitrate", amf_variant_int64(bitrate));
        self.try_set_component_property(
            component,
            "RateControlMethod",
            amf_variant_int64(AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR),
        );
        self.try_set_component_property(
            component,
            "QualityPreset",
            amf_variant_int64(if config.low_latency {
                AMF_VIDEO_ENCODER_QUALITY_PRESET_SPEED
            } else {
                AMF_VIDEO_ENCODER_QUALITY_PRESET_BALANCED
            }),
        );
        self.try_set_component_property(component, "BPicturesPattern", amf_variant_int64(0));
        self.try_set_component_property(
            component,
            "LowLatencyInternal",
            amf_variant_bool(config.low_latency),
        );
        self.try_set_component_property(
            component,
            "IDRPeriod",
            amf_variant_int64(idr_period as i64),
        );
        self.try_set_component_property(
            component,
            "HeaderInsertionSpacing",
            amf_variant_int64(idr_period as i64),
        );
        self.try_set_component_property(
            component,
            "OutputMode",
            amf_variant_int64(AMF_VIDEO_ENCODER_OUTPUT_MODE_FRAME),
        );
        self.try_set_component_property(component, "QueryTimeout", amf_variant_int64(0));
        Ok(())
    }

    fn create_input_texture(
        device: &ID3D11Device,
        width: u32,
        height: u32,
    ) -> Result<ID3D11Texture2D, MediaError> {
        let aligned_height = align_to(height, 16);
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: aligned_height,
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
                        "CreateTexture2D(NV12 AMF input) failed: {e}"
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
            self.slots.push(InputSlot {
                texture: Self::create_input_texture(device, width, height)?,
                output_view: None,
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
            .d3d_context
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
        slot_idx: usize,
    ) -> Result<ID3D11Texture2D, MediaError> {
        let config = self
            .config
            .clone()
            .ok_or(MediaError::EncoderNotInitialized)?;
        self.ensure_video_processor(&config)?;
        if slot_idx >= self.slots.len() {
            return Err(MediaError::EncoderNotInitialized);
        }

        let src = unsafe { ManuallyDrop::new(ID3D11Texture2D::from_raw(src_texture)) };
        let video_device = self.video_device.as_ref().unwrap().clone();
        let video_context = self.video_context.as_ref().unwrap().clone();
        let processor = self.video_processor.as_ref().unwrap().clone();
        let enumerator = self.video_enumerator.as_ref().unwrap().clone();
        let src_key = src_texture as usize;

        if self.cached_input_view.is_none()
            || self.cached_input_texture != src_key
            || self.cached_input_index != src_index
        {
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
            let in_res: ID3D11Resource = (*src).cast().map_err(|e| {
                MediaError::EncodeError(format!(
                    "Cast source texture to ID3D11Resource failed: {e}"
                ))
            })?;
            let mut input_view = None;
            unsafe {
                video_device
                    .CreateVideoProcessorInputView(
                        &in_res,
                        &enumerator,
                        &input_desc,
                        Some(&mut input_view),
                    )
                    .map_err(|e| {
                        MediaError::EncodeError(format!(
                            "CreateVideoProcessorInputView failed: {e}"
                        ))
                    })?;
            }
            self.cached_input_texture = src_key;
            self.cached_input_index = src_index;
            self.cached_input_view = Some(input_view.ok_or_else(|| {
                MediaError::EncodeError("VideoProcessor input view was None".into())
            })?);
        }

        let dst_texture = self.slots[slot_idx].texture.clone();
        if self.slots[slot_idx].output_view.is_none() {
            let output_desc = D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC {
                ViewDimension: D3D11_VPOV_DIMENSION_TEXTURE2D,
                Anonymous: D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_VPOV { MipSlice: 0 },
                },
            };
            let out_res: ID3D11Resource = dst_texture.cast().map_err(|e| {
                MediaError::EncodeError(format!(
                    "Cast destination texture to ID3D11Resource failed: {e}"
                ))
            })?;
            let mut output_view = None;
            unsafe {
                video_device
                    .CreateVideoProcessorOutputView(
                        &out_res,
                        &enumerator,
                        &output_desc,
                        Some(&mut output_view),
                    )
                    .map_err(|e| {
                        MediaError::EncodeError(format!(
                            "CreateVideoProcessorOutputView failed: {e}"
                        ))
                    })?;
            }
            self.slots[slot_idx].output_view = Some(output_view.ok_or_else(|| {
                MediaError::EncodeError("VideoProcessor output view was None".into())
            })?);
        }

        let input_view = self.cached_input_view.as_ref().unwrap().clone();
        let output_view = self.slots[slot_idx].output_view.as_ref().unwrap().clone();
        let mut stream = D3D11_VIDEO_PROCESSOR_STREAM::default();
        stream.Enable = true.into();
        stream.pInputSurface = ManuallyDrop::new(Some(input_view));
        let streams = [stream];
        unsafe {
            video_context
                .VideoProcessorBlt(&processor, &output_view, self.frame_count as u32, &streams)
                .map_err(|e| MediaError::EncodeError(format!("VideoProcessorBlt failed: {e}")))?;
        }
        if self.flush_each_frame {
            if let Some(context) = &self.d3d_context {
                unsafe { context.Flush() };
            }
        }
        Ok(dst_texture)
    }

    fn wrap_surface(
        &self,
        texture: &ID3D11Texture2D,
        pts_us: u64,
    ) -> Result<*mut AmfSurface, MediaError> {
        unsafe {
            let mut surface = ptr::null_mut();
            amf_ok(
                ((*(*self.context).vtbl).create_surface_from_dx11_native)(
                    self.context,
                    texture.as_raw() as *mut c_void,
                    &mut surface,
                    ptr::null_mut(),
                ),
                "AMFContext::CreateSurfaceFromDX11Native",
            )?;
            if surface.is_null() {
                return Err(MediaError::EncodeError("AMF returned null surface".into()));
            }
            amf_ok(
                ((*(*surface).vtbl).set_crop)(
                    surface,
                    0,
                    0,
                    self.config.as_ref().unwrap().width as i32,
                    self.config.as_ref().unwrap().height as i32,
                ),
                "AMFSurface::SetCrop",
            )?;
            if self.force_keyframe || self.frame_count == 0 {
                let force_type = amf_wide("ForcePictureType");
                let insert_sps = amf_wide("InsertSPS");
                let insert_pps = amf_wide("InsertPPS");
                amf_ok(
                    ((*(*surface).vtbl).set_property)(
                        surface,
                        force_type.as_ptr(),
                        amf_variant_int64(AMF_VIDEO_ENCODER_PICTURE_TYPE_IDR),
                    ),
                    "AMFSurface::SetProperty(ForcePictureType)",
                )?;
                amf_ok(
                    ((*(*surface).vtbl).set_property)(
                        surface,
                        insert_sps.as_ptr(),
                        amf_variant_bool(true),
                    ),
                    "AMFSurface::SetProperty(InsertSPS)",
                )?;
                amf_ok(
                    ((*(*surface).vtbl).set_property)(
                        surface,
                        insert_pps.as_ptr(),
                        amf_variant_bool(true),
                    ),
                    "AMFSurface::SetProperty(InsertPPS)",
                )?;
            }
            ((*(*surface).vtbl).set_pts)(surface, (pts_us * 10) as AmfPts);
            ((*(*surface).vtbl).set_duration)(
                surface,
                if self.config.as_ref().unwrap().fps > 0 {
                    (10_000_000 / self.config.as_ref().unwrap().fps as u64) as AmfPts
                } else {
                    0
                },
            );
            Ok(surface)
        }
    }

    fn drain_output(&mut self, pts_us: u64) -> Result<Vec<EncodedVideoFrame>, MediaError> {
        let config = self
            .config
            .as_ref()
            .ok_or(MediaError::EncoderNotInitialized)?;
        let mut frames = Vec::new();
        loop {
            let mut data = ptr::null_mut();
            let res =
                unsafe { ((*(*self.component).vtbl).query_output)(self.component, &mut data) };
            if res == AMF_REPEAT || data.is_null() {
                break;
            }
            if res == AMF_EOF {
                break;
            }
            amf_ok(res, "AMFComponent::QueryOutput")?;
            let mut buffer_ptr: *mut c_void = ptr::null_mut();
            let qi = unsafe {
                ((*(*data).vtbl).query_interface)(data, &IID_AMF_BUFFER, &mut buffer_ptr)
            };
            if qi != AMF_OK || buffer_ptr.is_null() {
                unsafe { ((*(*data).vtbl).release)(data) };
                return Err(MediaError::EncodeError(format!(
                    "AMF QueryInterface(AMFBuffer) failed: {qi}"
                )));
            }
            let buffer = buffer_ptr as *mut AmfBuffer;
            let size = unsafe { ((*(*buffer).vtbl).get_size)(buffer) };
            let native = unsafe { ((*(*buffer).vtbl).get_native)(buffer) } as *const u8;
            let data_vec = if !native.is_null() && size > 0 {
                unsafe { std::slice::from_raw_parts(native, size).to_vec() }
            } else {
                Vec::new()
            };
            unsafe {
                ((*(*buffer).vtbl).release)(buffer);
                ((*(*data).vtbl).release)(data);
            }
            if !data_vec.is_empty() {
                let is_key =
                    is_idr_annexb(&data_vec) || self.force_keyframe || self.frame_count == 0;
                frames.push(EncodedVideoFrame {
                    data: data_vec,
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
                });
            }
        }
        Ok(frames)
    }
}

impl VideoEncoder for WindowsAmfEncoder {
    fn initialize(&mut self, config: &EncoderConfig) -> Result<(), MediaError> {
        if config.codec != VideoCodec::H264 {
            return Err(MediaError::EncoderInitFailed(
                "native Windows AMF v1 supports H.264 only".into(),
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
            self.d3d_context = Some((*borrowed_context).clone());
        }
        log::info!("Native AMF: creating D3D11 context");
        self.create_context(device_ptr)?;
        log::info!("Native AMF: creating AVC component");
        self.create_component(config)?;
        log::info!(
            "Native AMF: creating {} aligned NV12 input textures",
            INPUT_RING_SIZE
        );
        self.create_slots(config.width, config.height)?;
        self.config = Some(config.clone());
        self.initialized = true;
        self.force_keyframe = true;
        self.frame_count = 0;
        self.cached_input_texture = 0;
        self.cached_input_index = 0;
        self.cached_input_view = None;
        if self.flush_each_frame {
            log::warn!("Native AMF: flushing D3D11 context after every VideoProcessorBlt");
        }
        log::info!(
            "Native Windows AMF D3D11 initialized: {}x{} @{}fps {}kbps (runtime={:#x})",
            config.width,
            config.height,
            config.fps,
            config.bitrate_kbps,
            self.api.runtime_version,
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
                    "native Windows AMF requires GpuBuffer::D3D11Texture".into(),
                ))
            }
        };
        let slot_idx = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        let slot_texture = self.convert_bgra_to_nv12(src_texture, src_index, slot_idx)?;
        let surface = self.wrap_surface(&slot_texture, pts_us)?;
        let submit = unsafe {
            ((*(*self.component).vtbl).submit_input)(self.component, surface as *mut AmfData)
        };
        unsafe { ((*(*surface).vtbl).release)(surface) };
        if submit == AMF_INPUT_FULL {
            let mut frames = self.drain_output(pts_us)?;
            let surface = self.wrap_surface(&slot_texture, pts_us)?;
            let retry = unsafe {
                ((*(*self.component).vtbl).submit_input)(self.component, surface as *mut AmfData)
            };
            unsafe { ((*(*surface).vtbl).release)(surface) };
            amf_ok(retry, "AMFComponent::SubmitInput retry")?;
            frames.extend(self.drain_output(pts_us)?);
            self.frame_count += 1;
            self.force_keyframe = false;
            return Ok(frames);
        }
        amf_ok(submit, "AMFComponent::SubmitInput")?;
        let frames = self.drain_output(pts_us)?;
        self.frame_count += 1;
        self.force_keyframe = false;
        Ok(frames)
    }

    fn request_keyframe(&mut self) {
        self.force_keyframe = true;
        log::debug!("Native AMF IDR requested for the next submitted surface");
    }

    fn set_bitrate(&mut self, bitrate_kbps: u32) -> Result<(), MediaError> {
        if !self.initialized {
            return Err(MediaError::EncoderNotInitialized);
        }
        if let Some(config) = &mut self.config {
            config.bitrate_kbps = bitrate_kbps;
        }
        if !self.component.is_null() {
            let bitrate = bitrate_kbps as i64 * 1000;
            self.set_component_property(
                self.component,
                "TargetBitrate",
                amf_variant_int64(bitrate),
            )?;
            self.try_set_component_property(
                self.component,
                "PeakBitrate",
                amf_variant_int64(bitrate),
            );
        }
        self.force_keyframe = true;
        log::info!("Native AMF target bitrate updated to {}kbps", bitrate_kbps);
        Ok(())
    }

    fn encoder_info(&self) -> EncoderInfo {
        self.info.clone()
    }

    fn flush(&mut self) -> Result<Vec<EncodedVideoFrame>, MediaError> {
        if self.component.is_null() {
            return Ok(vec![]);
        }
        let mut frames = Vec::new();
        let mut drain = unsafe { ((*(*self.component).vtbl).drain)(self.component) };
        while drain == AMF_INPUT_FULL {
            frames.extend(self.drain_output(0)?);
            drain = unsafe { ((*(*self.component).vtbl).drain)(self.component) };
        }
        if drain != AMF_EOF {
            amf_ok(drain, "AMFComponent::Drain")?;
        }
        frames.extend(self.drain_output(0)?);
        Ok(frames)
    }

    fn shutdown(&mut self) {
        unsafe {
            if !self.component.is_null() {
                let _ = ((*(*self.component).vtbl).terminate)(self.component);
                let _ = ((*(*self.component).vtbl).release)(self.component);
            }
            if !self.context.is_null() {
                let _ = ((*(*self.context).vtbl).terminate)(self.context);
                let _ = ((*(*self.context).vtbl).release)(self.context);
            }
        }
        self.component = ptr::null_mut();
        self.context = ptr::null_mut();
        self.device = None;
        self.d3d_context = None;
        self.video_device = None;
        self.video_context = None;
        self.video_processor = None;
        self.video_enumerator = None;
        self.cached_input_texture = 0;
        self.cached_input_index = 0;
        self.cached_input_view = None;
        self.slots.clear();
        self.initialized = false;
    }
}

impl Drop for WindowsAmfEncoder {
    fn drop(&mut self) {
        self.shutdown();
    }
}
