//! FFmpeg hardware-accelerated video encoder implementation.
//!
//! This module provides [`FfmpegEncoder`] which implements the [`VideoEncoder`]
//! trait using FFmpeg's hardware encoding APIs (VAAPI, NVENC, QSV) for
//! zero-copy GPU encoding.
//!
//! # Architecture
//!
//! ```text
//! GpuBuffer (DmaBuf/CpuBuffer)
//!   │
//!   ▼
//! AVFrame (DRM_PRIME or upload to HW surface)
//!   │
//!   ▼
//! avcodec_send_frame → avcodec_receive_packet
//!   │
//!   ▼
//! EncodedVideoFrame (Annex-B H.264 bitstream)
//! ```
//!
//! # Encoder Selection Priority (Linux)
//!
//! 1. `h264_nvenc` — NVIDIA GPU (via CUDA)
//! 2. `h264_vaapi` — Intel/AMD GPU (via VAAPI)
//! 3. `h264_qsv` — Intel Quick Sync Video
//! 4. `libx264` — Software fallback

#[allow(non_upper_case_globals)]
use std::ffi::CString;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

use ffmpeg_next::sys as ffi;

#[cfg(target_os = "windows")]
const SWS_FAST_BILINEAR: libc::c_int = ffi::SwsFlags::SWS_FAST_BILINEAR as libc::c_int;

#[cfg(target_os = "windows")]
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, ID3D11Resource,
    ID3D11VideoDevice, ID3D11VideoContext, ID3D11VideoProcessor,
    ID3D11VideoProcessorEnumerator, ID3D11VideoProcessorInputView,
    ID3D11VideoProcessorOutputView, D3D11_VIDEO_PROCESSOR_CONTENT_DESC,
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC, D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC,
    D3D11_VIDEO_PROCESSOR_STREAM, D3D11_VIDEO_FRAME_FORMAT,
    D3D11_VIDEO_USAGE, D3D11_VPIV_DIMENSION, D3D11_VPOV_DIMENSION,
    D3D11_TEX2D_VPIV, D3D11_TEX2D_VPOV, D3D11_TEX2D_ARRAY_VPOV,
    D3D11_VIDEO_USAGE_PLAYBACK_NORMAL, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
    D3D11_VPIV_DIMENSION_TEXTURE2D, D3D11_VPOV_DIMENSION_TEXTURE2D,
    D3D11_VPOV_DIMENSION_TEXTURE2DARRAY,
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0, D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0,
};
#[cfg(target_os = "windows")]
use windows::Win32::Graphics::Dxgi::Common::DXGI_RATIONAL;
#[cfg(target_os = "windows")]
use windows::core::Interface;

#[cfg(not(target_os = "windows"))]
const SWS_FAST_BILINEAR: libc::c_int = ffi::SWS_FAST_BILINEAR as libc::c_int;

use crate::capture::gpu_buffer::GpuBuffer;
use crate::encode::{EncoderConfig, VideoEncoder};
use crate::error::MediaError;
use crate::types::*;

#[cfg(target_os = "linux")]
struct CudaCopier {
    cu_memcpy_dtod: unsafe extern "C" fn(u64, u64, usize) -> i32,
    cu_ctx_set_current: unsafe extern "C" fn(*mut libc::c_void) -> i32,
    cu_ctx_get_current: unsafe extern "C" fn(*mut *mut libc::c_void) -> i32,
    _lib: *mut libc::c_void,
}

#[cfg(target_os = "linux")]
unsafe impl Send for CudaCopier {}
#[cfg(target_os = "linux")]
unsafe impl Sync for CudaCopier {}
#[cfg(target_os = "windows")]
#[repr(C)]
struct AVD3D11VADeviceContext {
    device: *mut std::ffi::c_void,
    device_context: *mut std::ffi::c_void,
    video_device: *mut std::ffi::c_void,
    video_context: *mut std::ffi::c_void,
    lock: Option<unsafe extern "C" fn(lock_ctx: *mut std::ffi::c_void)>,
    unlock: Option<unsafe extern "C" fn(lock_ctx: *mut std::ffi::c_void)>,
    lock_ctx: *mut std::ffi::c_void,
}

#[cfg(target_os = "windows")]
#[repr(C)]
struct AVD3D11VAFramesContext {
    texture: *mut std::ffi::c_void,
    textures: *mut *mut std::ffi::c_void,
    nb_textures: std::ffi::c_int,
    bind_flags: std::ffi::c_uint,
    misc_flags: std::ffi::c_uint,
}

#[cfg(target_os = "linux")]
static CUDA_COPIER: std::sync::OnceLock<Option<CudaCopier>> = std::sync::OnceLock::new();

#[cfg(target_os = "linux")]
fn get_cuda_copier() -> Option<&'static CudaCopier> {
    CUDA_COPIER
        .get_or_init(|| unsafe {
            let mut lib = libc::dlopen(
                b"libcuda.so.1\0".as_ptr() as *const libc::c_char,
                libc::RTLD_LAZY,
            );
            if lib.is_null() {
                lib = libc::dlopen(
                    b"libcuda.so\0".as_ptr() as *const libc::c_char,
                    libc::RTLD_LAZY,
                );
            }
            if lib.is_null() {
                return None;
            }
            let sym_memcpy = libc::dlsym(lib, b"cuMemcpyDtoD_v2\0".as_ptr() as *const libc::c_char);
            let sym_set_cur =
                libc::dlsym(lib, b"cuCtxSetCurrent\0".as_ptr() as *const libc::c_char);
            let sym_get_cur =
                libc::dlsym(lib, b"cuCtxGetCurrent\0".as_ptr() as *const libc::c_char);
            if sym_memcpy.is_null() || sym_set_cur.is_null() || sym_get_cur.is_null() {
                libc::dlclose(lib);
                return None;
            }
            Some(CudaCopier {
                cu_memcpy_dtod: std::mem::transmute(sym_memcpy),
                cu_ctx_set_current: std::mem::transmute(sym_set_cur),
                cu_ctx_get_current: std::mem::transmute(sym_get_cur),
                _lib: lib,
            })
        })
        .as_ref()
}

// ---------------------------------------------------------------------------
// FFmpeg global init (idempotent)
// ---------------------------------------------------------------------------

static FFMPEG_INIT: Once = Once::new();
static FFMPEG_INIT_OK: AtomicBool = AtomicBool::new(false);

fn ensure_ffmpeg_init() {
    FFMPEG_INIT.call_once(|| {
        ffmpeg_next::init().expect("Failed to initialize FFmpeg");
        FFMPEG_INIT_OK.store(true, Ordering::SeqCst);
        log::info!("FFmpeg initialized successfully");
    });
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

/// Convert an FFmpeg negative error code into a [`MediaError::FfmpegError`].
fn ff_err(code: i32) -> MediaError {
    let mut buf = [0u8; 256];
    // SAFETY: `av_strerror` writes a NUL-terminated error string into the
    // supplied buffer. The buffer is large enough and the pointer is valid.
    unsafe {
        ffi::av_strerror(code, buf.as_mut_ptr() as *mut libc::c_char, buf.len());
    }
    let msg = String::from_utf8_lossy(&buf)
        .trim_end_matches('\0')
        .to_string();
    MediaError::FfmpegError { code, message: msg }
}

/// Return the hardware pixel format corresponding to a [`HwAccelType`].
fn hw_pix_fmt(hw_type: HwAccelType) -> ffi::AVPixelFormat {
    match hw_type {
        HwAccelType::Vaapi => ffi::AVPixelFormat::AV_PIX_FMT_VAAPI,
        HwAccelType::Nvenc => ffi::AVPixelFormat::AV_PIX_FMT_CUDA,
        HwAccelType::Amf => ffi::AVPixelFormat::AV_PIX_FMT_D3D11,
        HwAccelType::Qsv => ffi::AVPixelFormat::AV_PIX_FMT_QSV,
        _ => ffi::AVPixelFormat::AV_PIX_FMT_YUV420P,
    }
}

/// Check whether a given [`HwAccelType`] uses a hardware pixel format or is
/// software-only.
fn is_hw_encoder(hw_type: HwAccelType) -> bool {
    !matches!(hw_type, HwAccelType::Software)
}

// ---------------------------------------------------------------------------
// FfmpegEncoder
// ---------------------------------------------------------------------------

/// FFmpeg-based hardware-accelerated H.264 video encoder.
///
/// Uses FFmpeg's codec API to encode frames. Supports hardware-accelerated
/// encoding via VAAPI, NVENC, QSV on Linux, with software fallback to libx264.
///
/// # Thread Safety
///
/// This type is `Send` but **not** `Sync`. The FFmpeg codec context must only
/// be accessed from a single thread at a time.
pub struct FfmpegEncoder {
    /// FFmpeg codec context — owns the encoder state.
    codec_ctx: *mut ffi::AVCodecContext,
    /// Hardware device context (e.g. VAAPI display, CUDA context).
    hw_device_ctx: *mut ffi::AVBufferRef,
    /// Hardware frames context — pool of GPU surfaces.
    hw_frames_ctx: *mut ffi::AVBufferRef,
    /// Reusable AVFrame for the hardware surface (or SW frame if software).
    frame: *mut ffi::AVFrame,
    /// Reusable AVFrame used as the CPU-side staging surface for upload.
    sw_frame: *mut ffi::AVFrame,
    /// Reusable AVPacket for receiving encoded data.
    packet: *mut ffi::AVPacket,
    /// SIMD-optimized color converter (BGRA → NV12) via FFmpeg swscale.
    sws_ctx: *mut ffi::SwsContext,
    /// When `true`, the next encoded frame will be forced as a keyframe.
    force_keyframe: bool,
    /// Monotonically increasing frame counter.
    frame_count: u64,
    /// The active encoder configuration.
    config: Option<EncoderConfig>,
    /// Metadata about the selected encoder.
    info: Option<EncoderInfo>,
    /// Whether the encoder has been initialized.
    initialized: bool,
    /// Cached SPS/PPS extradata in Annex-B format, extracted from codec_ctx after init.
    /// Prepended to every IDR keyframe to guarantee the browser has codec parameters,
    /// regardless of whether the encoder supports the `repeat-headers` option.
    sps_pps_cache: Option<Vec<u8>>,
    /// Cached Direct3D11 Video Device for zero-copy format conversion.
    #[cfg(target_os = "windows")]
    video_device: Option<ID3D11VideoDevice>,
    /// Cached Direct3D11 Video Context for zero-copy format conversion.
    #[cfg(target_os = "windows")]
    video_context: Option<ID3D11VideoContext>,
    /// Cached Direct3D11 Video Processor for zero-copy format conversion.
    #[cfg(target_os = "windows")]
    video_processor: Option<ID3D11VideoProcessor>,
    /// Cached Direct3D11 Video Processor Enumerator.
    #[cfg(target_os = "windows")]
    video_enumerator: Option<ID3D11VideoProcessorEnumerator>,
    /// Cached Input View for the video processor.
    #[cfg(target_os = "windows")]
    video_input_view: Option<ID3D11VideoProcessorInputView>,
    /// Cached Output View for the video processor.
    #[cfg(target_os = "windows")]
    video_output_view: Option<ID3D11VideoProcessorOutputView>,
    /// The source texture pointer used to create the cached input view.
    #[cfg(target_os = "windows")]
    cached_src_tex: usize,
    /// The destination texture pointer used to create the cached output view.
    #[cfg(target_os = "windows")]
    cached_dst_tex: usize,
    /// The array index used to create the cached output view.
    #[cfg(target_os = "windows")]
    cached_dst_idx: u32,
    /// Cached Direct3D11 Staging Texture for software fallback.
    #[cfg(target_os = "windows")]
    staging_texture: Option<ID3D11Texture2D>,
}

// SAFETY: The FFmpeg codec context is only accessed through `&mut self` methods,
// so sending the encoder to another thread is safe as long as it is not used
// concurrently (which Rust's ownership system enforces).
unsafe impl Send for FfmpegEncoder {}

impl FfmpegEncoder {
    /// Create a new uninitialized FFmpeg encoder.
    ///
    /// Call [`initialize`](VideoEncoder::initialize) before encoding frames.
    pub fn new() -> Result<Self, MediaError> {
        ensure_ffmpeg_init();

        Ok(Self {
            codec_ctx: ptr::null_mut(),
            hw_device_ctx: ptr::null_mut(),
            hw_frames_ctx: ptr::null_mut(),
            frame: ptr::null_mut(),
            sw_frame: ptr::null_mut(),
            packet: ptr::null_mut(),
            sws_ctx: ptr::null_mut(),
            force_keyframe: false,
            frame_count: 0,
            config: None,
            info: None,
            initialized: false,
            sps_pps_cache: None,
            #[cfg(target_os = "windows")]
            video_device: None,
            #[cfg(target_os = "windows")]
            video_context: None,
            #[cfg(target_os = "windows")]
            video_processor: None,
            #[cfg(target_os = "windows")]
            video_enumerator: None,
            #[cfg(target_os = "windows")]
            video_input_view: None,
            #[cfg(target_os = "windows")]
            video_output_view: None,
            #[cfg(target_os = "windows")]
            cached_src_tex: 0,
            #[cfg(target_os = "windows")]
            cached_dst_tex: 0,
            #[cfg(target_os = "windows")]
            cached_dst_idx: 0,
            #[cfg(target_os = "windows")]
            staging_texture: None,
        })
    }

    /// Probe which video encoders are available on this system.
    ///
    /// Performs a lightweight check by attempting to find each encoder in the
    /// FFmpeg build.
    pub fn list_available() -> Vec<EncoderInfo> {
        ensure_ffmpeg_init();

        let mut available = Vec::new();

        // Encoder candidates in priority order
        let candidates: &[(&str, HwAccelType, VideoCodec)] = &[
            // H.264 Candidates
            #[cfg(target_os = "linux")]
            ("h264_nvenc", HwAccelType::Nvenc, VideoCodec::H264),
            #[cfg(target_os = "linux")]
            ("h264_vaapi", HwAccelType::Vaapi, VideoCodec::H264),
            #[cfg(target_os = "linux")]
            ("h264_qsv", HwAccelType::Qsv, VideoCodec::H264),
            #[cfg(target_os = "windows")]
            ("h264_nvenc", HwAccelType::Nvenc, VideoCodec::H264),
            #[cfg(target_os = "windows")]
            ("h264_amf", HwAccelType::Amf, VideoCodec::H264),
            #[cfg(target_os = "windows")]
            ("h264_qsv", HwAccelType::Qsv, VideoCodec::H264),
            #[cfg(target_os = "macos")]
            (
                "h264_videotoolbox",
                HwAccelType::VideoToolbox,
                VideoCodec::H264,
            ),
            ("libx264", HwAccelType::Software, VideoCodec::H264),
            // H.265 / HEVC Candidates
            #[cfg(target_os = "linux")]
            ("hevc_nvenc", HwAccelType::Nvenc, VideoCodec::H265),
            #[cfg(target_os = "linux")]
            ("hevc_vaapi", HwAccelType::Vaapi, VideoCodec::H265),
            #[cfg(target_os = "linux")]
            ("hevc_qsv", HwAccelType::Qsv, VideoCodec::H265),
            #[cfg(target_os = "windows")]
            ("hevc_nvenc", HwAccelType::Nvenc, VideoCodec::H265),
            #[cfg(target_os = "windows")]
            ("hevc_amf", HwAccelType::Amf, VideoCodec::H265),
            #[cfg(target_os = "windows")]
            ("hevc_qsv", HwAccelType::Qsv, VideoCodec::H265),
            #[cfg(target_os = "macos")]
            (
                "hevc_videotoolbox",
                HwAccelType::VideoToolbox,
                VideoCodec::H265,
            ),
            ("libx265", HwAccelType::Software, VideoCodec::H265),
            // AV1 Candidates
            #[cfg(target_os = "linux")]
            ("av1_nvenc", HwAccelType::Nvenc, VideoCodec::AV1),
            #[cfg(target_os = "linux")]
            ("av1_vaapi", HwAccelType::Vaapi, VideoCodec::AV1),
            #[cfg(target_os = "linux")]
            ("av1_qsv", HwAccelType::Qsv, VideoCodec::AV1),
            #[cfg(target_os = "windows")]
            ("av1_nvenc", HwAccelType::Nvenc, VideoCodec::AV1),
            #[cfg(target_os = "windows")]
            ("av1_amf", HwAccelType::Amf, VideoCodec::AV1),
            #[cfg(target_os = "windows")]
            ("av1_qsv", HwAccelType::Qsv, VideoCodec::AV1),
            #[cfg(target_os = "macos")]
            (
                "av1_videotoolbox",
                HwAccelType::VideoToolbox,
                VideoCodec::AV1,
            ),
            ("libsvtav1", HwAccelType::Software, VideoCodec::AV1),
            ("libaom-av1", HwAccelType::Software, VideoCodec::AV1),
        ];

        for &(name, hw_type, codec) in candidates {
            if ffmpeg_next::encoder::find_by_name(name).is_some() {
                log::info!(
                    "Found encoder: {} ({:?}) for codec {}",
                    name,
                    hw_type,
                    codec
                );
                available.push(EncoderInfo {
                    name: name.to_string(),
                    hw_type,
                    supported_codecs: vec![codec],
                });
            } else {
                log::debug!("Encoder '{}' not found in FFmpeg build", name);
            }
        }

        if available.is_empty() {
            log::warn!("No video encoders found! Encoding will not be possible.");
        }

        available
    }

    /// Select the best encoder for the given codec and optional hardware
    /// preference. Returns `(encoder_name, hw_accel_type)`.
    #[allow(dead_code)]
    fn select_encoder(
        codec: VideoCodec,
        preferred_hw: Option<HwAccelType>,
    ) -> Result<(&'static str, HwAccelType), MediaError> {
        let candidates: &[(&str, HwAccelType)] = match codec {
            VideoCodec::H264 => &[
                #[cfg(target_os = "linux")]
                ("h264_nvenc", HwAccelType::Nvenc),
                #[cfg(target_os = "linux")]
                ("h264_vaapi", HwAccelType::Vaapi),
                #[cfg(target_os = "linux")]
                ("h264_qsv", HwAccelType::Qsv),
                #[cfg(target_os = "windows")]
                ("h264_nvenc", HwAccelType::Nvenc),
                #[cfg(target_os = "windows")]
                ("h264_amf", HwAccelType::Amf),
                #[cfg(target_os = "windows")]
                ("h264_qsv", HwAccelType::Qsv),
                #[cfg(target_os = "macos")]
                ("h264_videotoolbox", HwAccelType::VideoToolbox),
                ("libx264", HwAccelType::Software),
            ],
            VideoCodec::H265 => &[
                #[cfg(target_os = "linux")]
                ("hevc_nvenc", HwAccelType::Nvenc),
                #[cfg(target_os = "linux")]
                ("hevc_vaapi", HwAccelType::Vaapi),
                #[cfg(target_os = "linux")]
                ("hevc_qsv", HwAccelType::Qsv),
                #[cfg(target_os = "windows")]
                ("hevc_nvenc", HwAccelType::Nvenc),
                #[cfg(target_os = "windows")]
                ("hevc_amf", HwAccelType::Amf),
                #[cfg(target_os = "windows")]
                ("hevc_qsv", HwAccelType::Qsv),
                #[cfg(target_os = "macos")]
                ("hevc_videotoolbox", HwAccelType::VideoToolbox),
                ("libx265", HwAccelType::Software),
            ],
            VideoCodec::AV1 => &[
                #[cfg(target_os = "linux")]
                ("av1_nvenc", HwAccelType::Nvenc),
                #[cfg(target_os = "linux")]
                ("av1_vaapi", HwAccelType::Vaapi),
                #[cfg(target_os = "linux")]
                ("av1_qsv", HwAccelType::Qsv),
                #[cfg(target_os = "windows")]
                ("av1_nvenc", HwAccelType::Nvenc),
                #[cfg(target_os = "windows")]
                ("av1_amf", HwAccelType::Amf),
                #[cfg(target_os = "windows")]
                ("av1_qsv", HwAccelType::Qsv),
                #[cfg(target_os = "macos")]
                ("av1_videotoolbox", HwAccelType::VideoToolbox),
                ("libsvtav1", HwAccelType::Software),
                ("libaom-av1", HwAccelType::Software),
            ],
        };

        // If the caller expressed a preference, try that first.
        if let Some(preferred) = preferred_hw {
            for &(name, hw_type) in candidates {
                if hw_type == preferred {
                    let cname = CString::new(name).unwrap();
                    // SAFETY: FFmpeg is initialized; the C string is valid.
                    let codec_ptr = unsafe { ffi::avcodec_find_encoder_by_name(cname.as_ptr()) };
                    if !codec_ptr.is_null() {
                        log::info!("Selected preferred encoder: {} ({})", name, hw_type);
                        return Ok((name, hw_type));
                    }
                }
            }
            log::warn!(
                "Preferred HW type {:?} not available, falling back to auto-select",
                preferred
            );
        }

        // Auto-select: first available from the priority list.
        for &(name, hw_type) in candidates {
            let cname = CString::new(name).unwrap();
            // SAFETY: FFmpeg is initialized; the C string is valid.
            let codec_ptr = unsafe { ffi::avcodec_find_encoder_by_name(cname.as_ptr()) };
            if !codec_ptr.is_null() {
                log::info!("Auto-selected encoder: {} ({})", name, hw_type);
                return Ok((name, hw_type));
            }
        }

        Err(MediaError::NoEncoderAvailable)
    }

    /// Returns all available encoder candidates in priority order.
    /// Used by `initialize()` to try each one until one succeeds at `avcodec_open2`.
    pub fn select_encoder_candidates(
        codec: VideoCodec,
        preferred_hw: Option<HwAccelType>,
    ) -> Vec<(String, HwAccelType)> {
        let candidates: &[(&str, HwAccelType)] = match codec {
            VideoCodec::H264 => &[
                #[cfg(target_os = "linux")]
                ("h264_nvenc", HwAccelType::Nvenc),
                #[cfg(target_os = "linux")]
                ("h264_vaapi", HwAccelType::Vaapi),
                #[cfg(target_os = "linux")]
                ("h264_qsv", HwAccelType::Qsv),
                #[cfg(target_os = "windows")]
                ("h264_nvenc", HwAccelType::Nvenc),
                #[cfg(target_os = "windows")]
                ("h264_amf", HwAccelType::Amf),
                #[cfg(target_os = "windows")]
                ("h264_qsv", HwAccelType::Qsv),
                #[cfg(target_os = "macos")]
                ("h264_videotoolbox", HwAccelType::VideoToolbox),
                ("libx264", HwAccelType::Software),
            ],
            VideoCodec::H265 => &[
                #[cfg(target_os = "linux")]
                ("hevc_nvenc", HwAccelType::Nvenc),
                #[cfg(target_os = "linux")]
                ("hevc_vaapi", HwAccelType::Vaapi),
                #[cfg(target_os = "linux")]
                ("hevc_qsv", HwAccelType::Qsv),
                #[cfg(target_os = "windows")]
                ("hevc_nvenc", HwAccelType::Nvenc),
                #[cfg(target_os = "windows")]
                ("hevc_amf", HwAccelType::Amf),
                #[cfg(target_os = "windows")]
                ("hevc_qsv", HwAccelType::Qsv),
                #[cfg(target_os = "macos")]
                ("hevc_videotoolbox", HwAccelType::VideoToolbox),
                ("libx265", HwAccelType::Software),
            ],
            VideoCodec::AV1 => &[
                #[cfg(target_os = "linux")]
                ("av1_nvenc", HwAccelType::Nvenc),
                #[cfg(target_os = "linux")]
                ("av1_vaapi", HwAccelType::Vaapi),
                #[cfg(target_os = "linux")]
                ("av1_qsv", HwAccelType::Qsv),
                #[cfg(target_os = "windows")]
                ("av1_nvenc", HwAccelType::Nvenc),
                #[cfg(target_os = "windows")]
                ("av1_amf", HwAccelType::Amf),
                #[cfg(target_os = "windows")]
                ("av1_qsv", HwAccelType::Qsv),
                #[cfg(target_os = "macos")]
                ("av1_videotoolbox", HwAccelType::VideoToolbox),
                ("libsvtav1", HwAccelType::Software),
                ("libaom-av1", HwAccelType::Software),
            ],
        };

        let mut result = Vec::new();

        // If preferred HW specified, put those first
        if let Some(preferred) = preferred_hw {
            for &(name, hw_type) in candidates {
                if hw_type == preferred {
                    let cname = CString::new(name).unwrap();
                    let codec_ptr = unsafe { ffi::avcodec_find_encoder_by_name(cname.as_ptr()) };
                    if !codec_ptr.is_null() {
                        result.push((name.to_string(), hw_type));
                    }
                }
            }
        }

        // Then add all remaining available candidates
        for &(name, hw_type) in candidates {
            let cname = CString::new(name).unwrap();
            let codec_ptr = unsafe { ffi::avcodec_find_encoder_by_name(cname.as_ptr()) };
            if !codec_ptr.is_null() {
                let entry = (name.to_string(), hw_type);
                if !result.contains(&entry) {
                    result.push(entry);
                }
            }
        }

        log::info!(
            "Encoder candidates: {:?}",
            result
                .iter()
                .map(|(n, h)| format!("{} ({:?})", n, h))
                .collect::<Vec<_>>()
        );
        result
    }

    /// Set encoder-specific private options via `av_opt_set`.
    ///
    /// This applies tuning parameters that differ per encoder backend.
    fn set_encoder_options(
        codec_ctx: *mut ffi::AVCodecContext,
        hw_type: HwAccelType,
        codec: VideoCodec,
        low_latency: bool,
    ) {
        // Helper to call av_opt_set on the codec's private data.
        let opt_set = |key: &str, value: &str| {
            let ckey = CString::new(key).unwrap();
            let cval = CString::new(value).unwrap();
            // SAFETY: codec_ctx is valid and opened; av_opt_set on priv_data
            // is the standard way to pass encoder-specific options.
            let ret =
                unsafe { ffi::av_opt_set((*codec_ctx).priv_data, ckey.as_ptr(), cval.as_ptr(), 0) };
            if ret < 0 {
                log::warn!("av_opt_set({}, {}) failed: {} (non-fatal)", key, value, ret);
            } else {
                log::debug!("av_opt_set({}, {})", key, value);
            }
        };

        if codec == VideoCodec::H264 || codec == VideoCodec::H265 {
            opt_set("repeat-headers", "1");
        }
        if codec == VideoCodec::H265 {
            opt_set("annexb", "1");
        }

        match hw_type {
            HwAccelType::Nvenc => {
                opt_set("preset", "p3"); // Changed from p1 to p3 for better compression quality
                opt_set("tune", "ull");
                opt_set("rc", "cbr");
                opt_set("zerolatency", "1");
                opt_set("forced-idr", "1");
                if codec == VideoCodec::H264 {
                    // Use baseline profile (CAVLC) to match SDP profile-level-id=42e033 (CBP).
                    // High Profile uses CABAC which is incompatible with CBP decoders.
                    opt_set("profile", "baseline");
                } else if codec == VideoCodec::H265 {
                    opt_set("profile", "main");
                    opt_set("repeat_vps_sps_pps", "1");
                }
                log::info!(
                    "Applied NVENC low-latency options (forced-idr=1 preset=p3 repeat headers)"
                );
            }
            HwAccelType::Vaapi => {
                opt_set("rc_mode", "CBR");
                if codec == VideoCodec::H264 {
                    opt_set("profile", "baseline"); // Match SDP 42e033 (CBP)
                } else if codec == VideoCodec::H265 {
                    opt_set("profile", "main");
                }
                if low_latency {
                    opt_set("async_depth", "1");
                }
                log::info!("Applied VAAPI encoder options");
            }
            HwAccelType::Qsv => {
                opt_set("preset", "veryfast");
                opt_set("forced_idr", "1");
                opt_set("repeat_pps", "1");
                if codec == VideoCodec::H264 {
                    opt_set("profile", "baseline"); // Match SDP 42e033 (CBP)
                } else if codec == VideoCodec::H265 {
                    opt_set("profile", "main");
                }
                if low_latency {
                    opt_set("async_depth", "1");
                }
                log::info!("Applied QSV encoder options");
            }
            HwAccelType::Amf => {
                opt_set("usage", "ultralowlatency");
                opt_set("rc", "cbr");
                opt_set("forced_idr", "1");
                opt_set("header_insertion_mode", "idr");
                if codec == VideoCodec::H264 {
                    // constrained_baseline matches SDP profile-level-id=42e033
                    // AMF encoder: CAVLC entropy coding for CBP compatibility
                    opt_set("profile", "constrained_baseline");
                    opt_set("level", "3.1");
                } else if codec == VideoCodec::H265 {
                    opt_set("profile", "main");
                }
                log::info!("Applied AMF encoder options (forced_idr=1)");
            }
            HwAccelType::VideoToolbox => {
                opt_set("realtime", "1");
                if codec == VideoCodec::H264 {
                    opt_set("profile", "baseline"); // Match SDP 42e033 (CBP)
                } else if codec == VideoCodec::H265 {
                    opt_set("profile", "main");
                }
                log::info!("Applied VideoToolbox encoder options");
            }
            HwAccelType::Software => {
                opt_set("preset", "ultrafast");
                if low_latency {
                    opt_set("tune", "zerolatency");
                }
                if codec == VideoCodec::H264 {
                    // Use baseline to match SDP 42e033 and enable CAVLC (constrained_baseline is invalid for libx264)
                    opt_set("profile", "baseline");
                } else if codec == VideoCodec::H265 {
                    opt_set("profile", "main");
                }
                log::info!("Applied software encoder options");
            }
        }
    }

    /// Receive all available packets from the encoder and convert them to
    /// [`EncodedVideoFrame`]s. If `sps_pps_cache` is provided, it will be
    /// prepended to every IDR keyframe to ensure the browser always has the
    /// codec parameters it needs to decode, even if the encoder doesn't
    /// support the `repeat-headers` option (e.g. AMF on Windows).
    fn drain_packets(
        &mut self,
        codec: VideoCodec,
        fps: u32,
        prepend_headers: bool,
    ) -> Result<Vec<EncodedVideoFrame>, MediaError> {
        let mut frames = Vec::new();
        let duration_us = if fps > 0 {
            1_000_000u64 / fps as u64
        } else {
            0
        };

        loop {
            // SAFETY: codec_ctx and packet are valid, non-null pointers
            // allocated by FFmpeg. `avcodec_receive_packet` may return
            // AVERROR(EAGAIN) when no more output is available.
            let ret = unsafe { ffi::avcodec_receive_packet(self.codec_ctx, self.packet) };

            if ret == ffi::AVERROR(ffi::EAGAIN) || ret == ffi::AVERROR_EOF {
                // No more packets available right now.
                break;
            }
            if ret < 0 {
                return Err(ff_err(ret));
            }

            // SAFETY: If avcodec_receive_packet returned 0, the packet data
            // pointer and size are valid. We copy the data out immediately.
            let data = unsafe {
                let pkt = &*self.packet;
                if pkt.data.is_null() || pkt.size <= 0 {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(pkt.data, pkt.size as usize).to_vec()
                }
            };

            let is_key = unsafe { (*self.packet).flags & ffi::AV_PKT_FLAG_KEY != 0 };
            let frame_type = if is_key {
                FrameType::Key
            } else {
                FrameType::Inter
            };

            // Prepend SPS/PPS (extradata) to every IDR keyframe.
            // This is required for WebRTC because:
            // 1. The `repeat-headers` AVOption fails silently for some encoders (AMF on Windows).
            // 2. Without inline SPS/PPS, the browser's H264 decoder fails on every IDR
            //    and continuously sends PLI requests, preventing video from displaying.
            let final_data = if is_key && prepend_headers {
                // Diagnostic: log first 16 bytes of raw encoder output to confirm Annex-B vs AVCC.
                // Annex-B: starts with 00 00 00 01 or 00 00 01 (start code)
                // AVCC: starts with 4-byte big-endian NAL length
                let raw_prefix: Vec<String> = data.iter().take(16).map(|b| format!("{:02x}", b)).collect();
                log::info!(
                    "IDR frame from encoder: {} bytes, first 16 bytes: [{}] ({})",
                    data.len(),
                    raw_prefix.join(" "),
                    if data.len() >= 4 && data[0] == 0 && data[1] == 0 && ((data[2] == 0 && data[3] == 1) || data[2] == 1) {
                        "Annex-B ✓"
                    } else {
                        "NOT Annex-B! (AVCC?)"
                    }
                );

                let slice_offset = find_annexb_headers(&data, codec).unwrap_or(0);
                let already_has_headers = has_sps_pps(&data[..slice_offset], codec);

                // Dynamically cache SPS/PPS/VPS headers from first keyframe if not already cached.
                if self.sps_pps_cache.is_none() && already_has_headers {
                    let extracted = extract_sps_pps(&data[..slice_offset], codec);
                    if !extracted.is_empty() {
                        log::info!("Dynamically cached SPS/PPS/VPS headers from first keyframe: {} bytes", extracted.len());
                        self.sps_pps_cache = Some(extracted);
                    }
                }

                if !already_has_headers {
                    if let Some(headers) = &self.sps_pps_cache {
                        let mut combined = Vec::with_capacity(data.len() + headers.len());
                        combined.extend_from_slice(&data[..slice_offset]);
                        combined.extend_from_slice(headers);
                        combined.extend_from_slice(&data[slice_offset..]);
                        log::info!("IDR combined with SPS/PPS at offset {}: {} bytes total", slice_offset, combined.len());
                        combined
                    } else {
                        data
                    }
                } else {
                    log::info!("IDR frame already has SPS/PPS prepended, using as-is");
                    data
                }
            } else {
                data
            };

            let pts = unsafe { (*self.packet).pts as u64 };

            frames.push(EncodedVideoFrame {
                data: final_data,
                frame_type,
                pts,
                duration: duration_us,
                codec,
            });

            log::trace!(
                "Received encoded packet: size={} key={} pts={}",
                frames.last().map(|f| f.data.len()).unwrap_or(0),
                is_key,
                pts,
            );

            // SAFETY: After extracting data, we must unref the packet to
            // release its internal buffers before the next receive call.
            unsafe {
                ffi::av_packet_unref(self.packet);
            }
        }

        Ok(frames)
    }

    /// Try to initialize a specific encoder. Returns Ok(()) on success,
    /// or an error if this encoder can't be used (e.g. NVENC session limit).
    fn try_initialize_encoder(
        &mut self,
        config: &EncoderConfig,
        encoder_name: &str,
        hw_type: HwAccelType,
    ) -> Result<(), MediaError> {
        // Step 2: Find the FFmpeg encoder by name.
        let cname = CString::new(encoder_name).unwrap();
        let codec = unsafe { ffi::avcodec_find_encoder_by_name(cname.as_ptr()) };
        if codec.is_null() {
            return Err(MediaError::EncoderInitFailed(format!(
                "Encoder '{}' not found",
                encoder_name
            )));
        }

        let codec_ctx = unsafe { ffi::avcodec_alloc_context3(codec) };
        if codec_ctx.is_null() {
            return Err(MediaError::EncoderInitFailed(
                "Failed to allocate AVCodecContext".into(),
            ));
        }

        unsafe {
            let bitrate = (config.bitrate_kbps as i64) * 1000;
            (*codec_ctx).bit_rate = bitrate;
            (*codec_ctx).rc_max_rate = bitrate;
            (*codec_ctx).rc_buffer_size = ((bitrate * 2) / config.fps as i64) as libc::c_int;
            (*codec_ctx).width = config.width as libc::c_int;
            (*codec_ctx).height = config.height as libc::c_int;
            (*codec_ctx).colorspace = ffi::AVColorSpace::AVCOL_SPC_BT709;
            (*codec_ctx).color_range = ffi::AVColorRange::AVCOL_RANGE_JPEG;
            (*codec_ctx).color_primaries = ffi::AVColorPrimaries::AVCOL_PRI_BT709;
            (*codec_ctx).color_trc = ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT709;
            (*codec_ctx).time_base = ffi::AVRational {
                num: 1,
                den: config.fps as libc::c_int,
            };
            (*codec_ctx).framerate = ffi::AVRational {
                num: config.fps as libc::c_int,
                den: 1,
            };
            (*codec_ctx).gop_size = if config.keyframe_interval > 0 {
                config.keyframe_interval as libc::c_int
            } else {
                (config.fps * 5) as libc::c_int // 5 seconds GOP (allows dynamic keyframe insertion on HW encoders)
            };
            (*codec_ctx).max_b_frames = 0;
            (*codec_ctx).thread_count = 1;

            if config.codec == VideoCodec::H264 {
                (*codec_ctx).level = 31; // Match SDP profile-level-id=42001f (Level 3.1)
            }

            let is_d3d11 = if cfg!(target_os = "windows") && config.d3d11_device.is_some() && config.d3d11_context.is_some() && matches!(hw_type, HwAccelType::Amf | HwAccelType::Nvenc) {
                true
            } else {
                false
            };

            let has_hw_frames = matches!(hw_type, HwAccelType::Vaapi | HwAccelType::Nvenc) || is_d3d11;

            if is_hw_encoder(hw_type) {
                if is_d3d11 {
                    (*codec_ctx).pix_fmt = ffi::AVPixelFormat::AV_PIX_FMT_D3D11;
                } else if has_hw_frames {
                    (*codec_ctx).pix_fmt = hw_pix_fmt(hw_type);
                } else {
                    (*codec_ctx).pix_fmt = ffi::AVPixelFormat::AV_PIX_FMT_NV12;
                }
            } else {
                (*codec_ctx).pix_fmt = ffi::AVPixelFormat::AV_PIX_FMT_YUV420P;
            }

            if config.low_latency {
                (*codec_ctx).flags |= ffi::AV_CODEC_FLAG_LOW_DELAY as libc::c_int;
            }
        }

        let mut hw_device_ctx: *mut ffi::AVBufferRef = ptr::null_mut();
        let mut hw_frames_ctx: *mut ffi::AVBufferRef = ptr::null_mut();

        let is_d3d11 = if cfg!(target_os = "windows") && config.d3d11_device.is_some() && config.d3d11_context.is_some() && matches!(hw_type, HwAccelType::Amf | HwAccelType::Nvenc) {
            true
        } else {
            false
        };

        let has_hw_frames = matches!(hw_type, HwAccelType::Vaapi | HwAccelType::Nvenc) || is_d3d11;

        if is_hw_encoder(hw_type) && has_hw_frames {
            let ret = if is_d3d11 {
                #[cfg(target_os = "windows")]
                {
                    hw_device_ctx = unsafe { ffi::av_hwdevice_ctx_alloc(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA) };
                    if hw_device_ctx.is_null() {
                        -1
                    } else {
                        unsafe {
                            let device_raw = config.d3d11_device.unwrap() as *mut std::ffi::c_void;
                            let context_raw = config.d3d11_context.unwrap() as *mut std::ffi::c_void;

                            // Increment refcount so FFmpeg's internal release doesn't destroy our shared context
                            let device_owned = std::mem::ManuallyDrop::new(ID3D11Device::from_raw(device_raw));
                            let context_owned = std::mem::ManuallyDrop::new(ID3D11DeviceContext::from_raw(context_raw));

                            let device_clone = (*device_owned).clone();
                            let context_clone = (*context_owned).clone();

                            let device_ctx = (*hw_device_ctx).data as *mut ffi::AVHWDeviceContext;
                            let d3d11_ctx = (*device_ctx).hwctx as *mut AVD3D11VADeviceContext;
                            (*d3d11_ctx).device = device_clone.as_raw() as *mut std::ffi::c_void;
                            (*d3d11_ctx).device_context = context_clone.as_raw() as *mut std::ffi::c_void;

                            std::mem::forget(device_clone);
                            std::mem::forget(context_clone);

                            ffi::av_hwdevice_ctx_init(hw_device_ctx)
                        }
                    }
                }
                #[cfg(not(target_os = "windows"))]
                {
                    -1
                }
            } else {
                let (device_type, device_path): (ffi::AVHWDeviceType, Option<CString>) = match hw_type {
                    HwAccelType::Vaapi => (
                        ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                        Some(CString::new("/dev/dri/renderD128").unwrap()),
                    ),
                    HwAccelType::Nvenc => (ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA, None),
                    _ => (ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE, None),
                };

                let device_ptr = device_path
                    .as_ref()
                    .map(|p| p.as_ptr())
                    .unwrap_or(ptr::null());

                unsafe {
                    ffi::av_hwdevice_ctx_create(
                        &mut hw_device_ctx,
                        device_type,
                        device_ptr,
                        ptr::null_mut(),
                        0,
                    )
                }
            };

            if ret < 0 || hw_device_ctx.is_null() {
                unsafe {
                    if !hw_device_ctx.is_null() {
                        ffi::av_buffer_unref(&mut hw_device_ctx);
                    }
                    ffi::avcodec_free_context(&mut (codec_ctx as *mut _));
                }
                log::error!(
                    "Failed to create HW device context for {:?}: {}",
                    hw_type,
                    ret
                );
                return Err(ff_err(ret));
            }

            log::info!(
                "Created HW device context for {:?} (shared_d3d11={})",
                hw_type,
                is_d3d11
            );

            hw_frames_ctx = unsafe { ffi::av_hwframe_ctx_alloc(hw_device_ctx) };
            if hw_frames_ctx.is_null() {
                unsafe {
                    ffi::av_buffer_unref(&mut hw_device_ctx);
                    ffi::avcodec_free_context(&mut (codec_ctx as *mut _));
                }
                return Err(MediaError::EncoderInitFailed(
                    "Failed to allocate HW frames context".into(),
                ));
            }

            unsafe {
                let frames_ctx = (*hw_frames_ctx).data as *mut ffi::AVHWFramesContext;
                let sw_format = match hw_type {
                    HwAccelType::Nvenc => ffi::AVPixelFormat::AV_PIX_FMT_BGRA,
                    _ => ffi::AVPixelFormat::AV_PIX_FMT_NV12,
                };
                if is_d3d11 {
                    (*frames_ctx).format = ffi::AVPixelFormat::AV_PIX_FMT_D3D11;
                    #[cfg(target_os = "windows")]
                    {
                        let d3d11_frames = (*frames_ctx).hwctx as *mut AVD3D11VAFramesContext;
                        if sw_format == ffi::AVPixelFormat::AV_PIX_FMT_BGRA {
                            (*d3d11_frames).bind_flags = 40; // D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET
                        } else {
                            (*d3d11_frames).bind_flags = 8;  // D3D11_BIND_SHADER_RESOURCE
                        }
                        (*d3d11_frames).misc_flags = 0;
                    }
                } else {
                    (*frames_ctx).format = hw_pix_fmt(hw_type);
                }
                (*frames_ctx).sw_format = sw_format;
                (*frames_ctx).width = config.width as libc::c_int;
                (*frames_ctx).height = config.height as libc::c_int;
                (*frames_ctx).initial_pool_size = 4;
            }

            let ret = unsafe { ffi::av_hwframe_ctx_init(hw_frames_ctx) };
            if ret < 0 {
                unsafe {
                    ffi::av_buffer_unref(&mut hw_frames_ctx);
                    ffi::av_buffer_unref(&mut hw_device_ctx);
                    ffi::avcodec_free_context(&mut (codec_ctx as *mut _));
                }
                log::error!("Failed to init HW frames context: {}", ret);
                return Err(ff_err(ret));
            }

            unsafe {
                (*codec_ctx).hw_frames_ctx = ffi::av_buffer_ref(hw_frames_ctx);
                if (*codec_ctx).hw_frames_ctx.is_null() {
                    ffi::av_buffer_unref(&mut hw_frames_ctx);
                    ffi::av_buffer_unref(&mut hw_device_ctx);
                    ffi::avcodec_free_context(&mut (codec_ctx as *mut _));
                    return Err(MediaError::EncoderInitFailed(
                        "av_buffer_ref for hw_frames_ctx failed".into(),
                    ));
                }
                let frames_ctx = (*hw_frames_ctx).data as *mut ffi::AVHWFramesContext;
                log::info!(
                    "HW frames context initialized: pool_size={} format={:?} sw_format={:?}",
                    (*frames_ctx).initial_pool_size,
                    (*frames_ctx).format,
                    (*frames_ctx).sw_format
                );
            }
        }

        Self::set_encoder_options(codec_ctx, hw_type, config.codec, config.low_latency);

        let ret = unsafe { ffi::avcodec_open2(codec_ctx, codec, ptr::null_mut()) };
        if ret < 0 {
            log::error!(
                "avcodec_open2 failed for '{}': {} (e.g. NVENC session limit)",
                encoder_name,
                ret
            );
            unsafe {
                if !hw_frames_ctx.is_null() {
                    ffi::av_buffer_unref(&mut hw_frames_ctx);
                }
                if !hw_device_ctx.is_null() {
                    ffi::av_buffer_unref(&mut hw_device_ctx);
                }
                ffi::avcodec_free_context(&mut (codec_ctx as *mut _));
            }
            return Err(ff_err(ret));
        }

        log::info!(
            "Encoder '{}' opened successfully (hw_type={:?})",
            encoder_name,
            hw_type
        );

        let frame = unsafe { ffi::av_frame_alloc() };
        let sw_frame = unsafe { ffi::av_frame_alloc() };
        let packet = unsafe { ffi::av_packet_alloc() };

        if frame.is_null() || sw_frame.is_null() || packet.is_null() {
            unsafe {
                if !frame.is_null() {
                    ffi::av_frame_free(&mut (frame as *mut _));
                }
                if !sw_frame.is_null() {
                    ffi::av_frame_free(&mut (sw_frame as *mut _));
                }
                if !packet.is_null() {
                    ffi::av_packet_free(&mut (packet as *mut _));
                }
                if !hw_frames_ctx.is_null() {
                    ffi::av_buffer_unref(&mut hw_frames_ctx);
                }
                if !hw_device_ctx.is_null() {
                    ffi::av_buffer_unref(&mut hw_device_ctx);
                }
                ffi::avcodec_free_context(&mut (codec_ctx as *mut _));
            }
            return Err(MediaError::EncoderInitFailed(
                "Failed to allocate AVFrame/AVPacket".into(),
            ));
        }

        self.codec_ctx = codec_ctx;
        self.hw_device_ctx = hw_device_ctx;
        self.hw_frames_ctx = hw_frames_ctx;
        self.frame = frame;
        self.sw_frame = sw_frame;
        self.packet = packet;
        self.force_keyframe = false;
        self.frame_count = 0;
        self.config = Some(config.clone());
        self.info = Some(EncoderInfo {
            name: encoder_name.to_string(),
            hw_type,
            supported_codecs: vec![config.codec],
        });
        self.initialized = true;

        unsafe {
            (*self.sw_frame).width = config.width as libc::c_int;
            (*self.sw_frame).height = config.height as libc::c_int;
            let sw_format = match hw_type {
                HwAccelType::Nvenc => ffi::AVPixelFormat::AV_PIX_FMT_BGRA,
                HwAccelType::Software => ffi::AVPixelFormat::AV_PIX_FMT_YUV420P,
                _ => ffi::AVPixelFormat::AV_PIX_FMT_NV12,
            };
            (*self.sw_frame).format = sw_format as libc::c_int;
            let ret = ffi::av_frame_get_buffer(self.sw_frame, 32);
            if ret < 0 {
                log::warn!("Pre-allocation of sw_frame buffer failed: {}", ret);
            }
        }

        // Extract SPS/PPS extradata from the codec context and cache it in Annex-B format.
        // Prepended to every IDR keyframe so the browser always has codec parameters,
        // regardless of whether the encoder supports `repeat-headers` (AMF does not).
        self.sps_pps_cache = unsafe {
            let ctx = &*codec_ctx;
            if !ctx.extradata.is_null() && ctx.extradata_size > 0 {
                let extra = std::slice::from_raw_parts(ctx.extradata, ctx.extradata_size as usize);
                let annexb = avcc_extradata_to_annexb(extra);
                if !annexb.is_empty() {
                    // Log the actual H264 profile from the SPS for verification.
                    // SPS Annex-B layout: [00 00 00 01] [NAL_header=0x67] [profile_idc] [constraint_flags] [level_idc]
                    // Bytes 4,5,6,7 (0-indexed) = start_code(4) + NAL_header(1) + profile_idc + constraints + level
                    let profile_desc = if annexb.len() >= 8 {
                        let profile_idc = annexb[5];
                        let constraints = annexb[6];
                        let level_idc = annexb[7];
                        let profile_name = match profile_idc {
                            66 if constraints & 0xE0 == 0xE0 => "Constrained Baseline (CBP)",
                            66 => "Baseline",
                            77 => "Main",
                            88 => "Extended",
                            100 if constraints & 0x0C == 0x0C => "Constrained High",
                            100 => "High",
                            110 => "High 10",
                            122 => "High 4:2:2",
                            244 => "High 4:4:4",
                            _ => "Unknown",
                        };
                        format!(
                            "{} (profile_idc={}, constraints=0x{:02x}, level={}.{})",
                            profile_name,
                            profile_idc,
                            constraints,
                            level_idc / 10,
                            level_idc % 10,
                        )
                    } else {
                        "Unknown (SPS too short)".to_string()
                    };
                    log::info!(
                        "Cached SPS/PPS extradata: {} bytes (Annex-B). Actual H264 profile: {}",
                        annexb.len(),
                        profile_desc,
                    );
                    Some(annexb)
                } else {
                    log::warn!("extradata present but AVCC→Annex-B conversion produced empty output");
                    None
                }
            } else {
                log::warn!("No extradata in codec context after init; IDR frames may lack SPS/PPS");
                None
            }
        };

        log::info!(
            "FFmpeg encoder fully initialized: {} {}x{} @{}fps {}kbps",
            encoder_name,
            config.width,
            config.height,
            config.fps,
            config.bitrate_kbps,
        );

        Ok(())
    }
}

/// Normalize encoder extradata to Annex-B format for prepending to IDR keyframes.
///
/// Handles two common extradata formats from FFmpeg encoders:
///
/// **Annex-B format** (AMF, some software encoders): starts with `00 00 00 01` or `00 00 01`.
///   Used directly as-is — already the correct format.
///
/// **AVCC format** (NVENC, most encoders): starts with configurationVersion byte `0x01`,
///   followed by length-prefixed SPS/PPS NAL units. Converted to Annex-B by replacing
///   length prefixes with `00 00 00 01` start codes.
fn avcc_extradata_to_annexb(extra: &[u8]) -> Vec<u8> {
    if extra.is_empty() {
        return Vec::new();
    }

    // Detect Annex-B format: starts with 4-byte start code 00 00 00 01
    if extra.len() >= 4 && extra[0] == 0 && extra[1] == 0 && extra[2] == 0 && extra[3] == 1 {
        log::debug!("Extradata is already in Annex-B format (4-byte start code), using directly");
        return extra.to_vec();
    }

    // Detect Annex-B with 3-byte start code 00 00 01
    if extra.len() >= 3 && extra[0] == 0 && extra[1] == 0 && extra[2] == 1 {
        log::debug!("Extradata is already in Annex-B format (3-byte start code), using directly");
        return extra.to_vec();
    }

    // Try AVCC format: configurationVersion must be 1, minimum 7 bytes
    if extra.len() < 7 || extra[0] != 1 {
        log::warn!(
            "Extradata is neither Annex-B nor valid AVCC (first bytes: {:02x?}); cannot extract SPS/PPS",
            &extra[..extra.len().min(8)]
        );
        return Vec::new();
    }

    // Parse AVCC and convert each NAL unit to Annex-B
    let mut out = Vec::new();
    let annexb_start_code = &[0u8, 0, 0, 1];
    let mut pos = 5; // skip version, profile, compat, level, lengthSizeMinusOne

    // SPS NAL units
    if pos >= extra.len() { return out; }
    let num_sps = (extra[pos] & 0x1F) as usize;
    pos += 1;
    for _ in 0..num_sps {
        if pos + 2 > extra.len() { break; }
        let len = u16::from_be_bytes([extra[pos], extra[pos + 1]]) as usize;
        pos += 2;
        if pos + len > extra.len() { break; }
        out.extend_from_slice(annexb_start_code);
        out.extend_from_slice(&extra[pos..pos + len]);
        pos += len;
    }

    // PPS NAL units
    if pos >= extra.len() { return out; }
    let num_pps = (extra[pos] & 0xFF) as usize;
    pos += 1;
    for _ in 0..num_pps {
        if pos + 2 > extra.len() { break; }
        let len = u16::from_be_bytes([extra[pos], extra[pos + 1]]) as usize;
        pos += 2;
        if pos + len > extra.len() { break; }
        out.extend_from_slice(annexb_start_code);
        out.extend_from_slice(&extra[pos..pos + len]);
        pos += len;
    }

    out
}

/// Find the byte offset in an Annex-B stream where the first actual coded slice NAL unit starts.
/// Returns Some(offset) if found, where everything before that offset constitutes the stream headers (VPS/SPS/PPS/SEI).
fn find_annexb_headers(data: &[u8], codec: VideoCodec) -> Option<usize> {
    if data.len() < 3 {
        return None;
    }
    let mut i = 0;
    while i + 3 <= data.len() {
        let mut start_code_len = 0;
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            start_code_len = 3;
        } else if i + 4 <= data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1 {
            start_code_len = 4;
        }

        if start_code_len > 0 {
            let nalu_start = i + start_code_len;
            if nalu_start < data.len() {
                match codec {
                    VideoCodec::H264 => {
                        let nalu_type = data[nalu_start] & 0x1F;
                        // H.264 slice NAL units: 1 to 5
                        if nalu_type >= 1 && nalu_type <= 5 {
                            return Some(i);
                        }
                    }
                    VideoCodec::H265 => {
                        let nalu_type = (data[nalu_start] >> 1) & 0x3F;
                        // H.265 coded slice NAL units: 0 to 21
                        if nalu_type <= 21 {
                            return Some(i);
                        }
                    }
                    _ => {}
                }
            }
            i += start_code_len;
        } else {
            i += 1;
        }
    }
    None
}

/// Check if the provided header bytes contain both SPS and PPS NAL units.
fn has_sps_pps(data: &[u8], codec: VideoCodec) -> bool {
    if data.len() < 3 {
        return false;
    }
    let mut has_sps = false;
    let mut has_pps = false;
    let mut i = 0;
    while i + 3 <= data.len() {
        let mut start_code_len = 0;
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            start_code_len = 3;
        } else if i + 4 <= data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1 {
            start_code_len = 4;
        }

        if start_code_len > 0 {
            let nalu_start = i + start_code_len;
            if nalu_start < data.len() {
                match codec {
                    VideoCodec::H264 => {
                        let nalu_type = data[nalu_start] & 0x1F;
                        if nalu_type == 7 {
                            has_sps = true;
                        } else if nalu_type == 8 {
                            has_pps = true;
                        }
                    }
                    VideoCodec::H265 => {
                        let nalu_type = (data[nalu_start] >> 1) & 0x3F;
                        if nalu_type == 33 {
                            has_sps = true;
                        } else if nalu_type == 34 {
                            has_pps = true;
                        }
                    }
                    _ => {}
                }
            }
            i += start_code_len;
        } else {
            i += 1;
        }
    }
    has_sps && has_pps
}

/// Extract only the SPS, PPS (and VPS for H.265) NAL units from the header data.
fn extract_sps_pps(data: &[u8], codec: VideoCodec) -> Vec<u8> {
    let mut extracted = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        let mut start_code_len = 0;
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            start_code_len = 3;
        } else if i + 4 <= data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1 {
            start_code_len = 4;
        }

        if start_code_len > 0 {
            let nalu_start = i + start_code_len;
            let mut nalu_end = data.len();
            let mut next_i = nalu_start;
            while next_i + 3 <= data.len() {
                let is_next_start = data[next_i] == 0 && data[next_i + 1] == 0 && (data[next_i + 2] == 1 || (next_i + 4 <= data.len() && data[next_i + 2] == 0 && data[next_i + 3] == 1));
                if is_next_start {
                    nalu_end = next_i;
                    break;
                }
                next_i += 1;
            }

            if nalu_start < data.len() {
                let is_header = match codec {
                    VideoCodec::H264 => {
                        let nalu_type = data[nalu_start] & 0x1F;
                        nalu_type == 7 || nalu_type == 8
                    }
                    VideoCodec::H265 => {
                        let nalu_type = (data[nalu_start] >> 1) & 0x3F;
                        nalu_type == 32 || nalu_type == 33 || nalu_type == 34
                    }
                    _ => false,
                };

                if is_header {
                    extracted.extend_from_slice(&data[i..nalu_end]);
                }
            }
            i = nalu_end;
        } else {
            i += 1;
        }
    }
    extracted
}

impl VideoEncoder for FfmpegEncoder {
    fn initialize(&mut self, config: &EncoderConfig) -> Result<(), MediaError> {
        if self.initialized {
            return Err(MediaError::EncoderInitFailed(
                "Encoder already initialized".into(),
            ));
        }

        log::info!(
            "Initializing FFmpeg encoder: {}x{} {}fps {}kbps codec={} low_latency={}",
            config.width,
            config.height,
            config.fps,
            config.bitrate_kbps,
            config.codec,
            config.low_latency,
        );

        // Get the full list of encoder candidates to try.
        let candidates = Self::select_encoder_candidates(config.codec, config.preferred_hw);

        let mut last_err = MediaError::EncoderInitFailed("No encoders available".into());

        for (encoder_name, hw_type) in &candidates {
            match self.try_initialize_encoder(config, encoder_name, *hw_type) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    log::warn!(
                        "Encoder '{}' ({:?}) failed to initialize: {}. Trying next candidate...",
                        encoder_name,
                        hw_type,
                        e
                    );
                    last_err = e;
                }
            }
        }

        Err(last_err)
    }

    fn encode(
        &mut self,
        buffer: &GpuBuffer,
        pts_us: u64,
    ) -> Result<Vec<EncodedVideoFrame>, MediaError> {
        if !self.initialized {
            return Err(MediaError::EncoderNotInitialized);
        }

        let config = self.config.as_ref().unwrap();
        let info = self.info.as_ref().unwrap();
        let hw_type = info.hw_type;
        let use_hw = is_hw_encoder(hw_type);

        let send_frame: *mut ffi::AVFrame = match buffer {
            #[cfg(target_os = "windows")]
            GpuBuffer::D3D11Texture { texture, array_index } => {
                if self.hw_frames_ctx.is_null() {
                    #[cfg(target_os = "windows")]
                    unsafe {
                        use windows::Win32::Graphics::Direct3D11::{
                            ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, ID3D11Resource,
                            D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ,
                            D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ,
                        };
                        use windows::core::Interface;

                        let device_ptr = config.d3d11_device.ok_or_else(|| {
                            MediaError::EncodeError("D3D11 device not available in config".into())
                        })? as *mut std::ffi::c_void;
                        let context_ptr = config.d3d11_context.ok_or_else(|| {
                            MediaError::EncodeError("D3D11 context not available in config".into())
                        })? as *mut std::ffi::c_void;

                        let device = std::mem::ManuallyDrop::new(ID3D11Device::from_raw(device_ptr));
                        let context = std::mem::ManuallyDrop::new(ID3D11DeviceContext::from_raw(context_ptr));
                        let src_tex = std::mem::ManuallyDrop::new(ID3D11Texture2D::from_raw(*texture));

                        let mut src_desc = D3D11_TEXTURE2D_DESC::default();
                        (*src_tex).GetDesc(&mut src_desc);

                        let w = src_desc.Width as libc::c_int;
                        let h = src_desc.Height as libc::c_int;

                        let staging = if let Some(stg) = &self.staging_texture {
                            let mut stg_desc = D3D11_TEXTURE2D_DESC::default();
                            stg.GetDesc(&mut stg_desc);
                            if stg_desc.Width != src_desc.Width || stg_desc.Height != src_desc.Height {
                                let stg_desc_new = D3D11_TEXTURE2D_DESC {
                                    Width: src_desc.Width,
                                    Height: src_desc.Height,
                                    MipLevels: 1,
                                    ArraySize: 1,
                                    Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                                        Count: 1,
                                        Quality: 0,
                                    },
                                    Usage: D3D11_USAGE_STAGING,
                                    BindFlags: 0,
                                    CPUAccessFlags: windows::Win32::Graphics::Direct3D11::D3D11_CPU_ACCESS_READ.0 as u32,
                                    MiscFlags: 0,
                                };
                                let mut stg_new = None;
                                (*device).CreateTexture2D(&stg_desc_new, None, Some(&mut stg_new)).map_err(|e| {
                                    MediaError::EncodeError(format!("CreateTexture2D staging failed: {e}"))
                                })?;
                                let stg_new = stg_new.unwrap();
                                self.staging_texture = Some(stg_new.clone());
                                stg_new
                            } else {
                                stg.clone()
                            }
                        } else {
                            let stg_desc_new = D3D11_TEXTURE2D_DESC {
                                Width: src_desc.Width,
                                Height: src_desc.Height,
                                MipLevels: 1,
                                ArraySize: 1,
                                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                                    Count: 1,
                                    Quality: 0,
                                },
                                Usage: D3D11_USAGE_STAGING,
                                BindFlags: 0,
                                CPUAccessFlags: windows::Win32::Graphics::Direct3D11::D3D11_CPU_ACCESS_READ.0 as u32,
                                MiscFlags: 0,
                            };
                            let mut stg_new = None;
                            (*device).CreateTexture2D(&stg_desc_new, None, Some(&mut stg_new)).map_err(|e| {
                                MediaError::EncodeError(format!("CreateTexture2D staging failed: {e}"))
                            })?;
                            let stg_new = stg_new.unwrap();
                            self.staging_texture = Some(stg_new.clone());
                            stg_new
                        };

                        let src_res = (*src_tex).cast::<ID3D11Resource>().map_err(|e| {
                            MediaError::EncodeError(format!("Cast src_tex to ID3D11Resource failed: {e}"))
                        })?;
                        let dst_res = staging.cast::<ID3D11Resource>().map_err(|e| {
                            MediaError::EncodeError(format!("Cast staging to ID3D11Resource failed: {e}"))
                        })?;

                        (*context).CopySubresourceRegion(
                            &dst_res,
                            0,
                            0, 0, 0,
                            &src_res,
                            *array_index,
                            None,
                        );
                        (*context).Flush();

                        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
                        (*context).Map(&dst_res, 0, D3D11_MAP_READ, 0, Some(&mut mapped)).map_err(|e| {
                            MediaError::EncodeError(format!("Map staging failed: {e}"))
                        })?;

                        let src_stride = mapped.RowPitch as usize;
                        let src_ptr = mapped.pData as *const u8;

                        let sw_format = match hw_type {
                            HwAccelType::Nvenc => ffi::AVPixelFormat::AV_PIX_FMT_BGRA,
                            HwAccelType::Software => ffi::AVPixelFormat::AV_PIX_FMT_YUV420P,
                            _ => ffi::AVPixelFormat::AV_PIX_FMT_NV12,
                        };

                        let ret = ffi::av_frame_make_writable(self.sw_frame);
                        if ret < 0 {
                            ffi::av_frame_unref(self.sw_frame);
                            (*self.sw_frame).width = w;
                            (*self.sw_frame).height = h;
                            (*self.sw_frame).format = sw_format as libc::c_int;
                            let ret = ffi::av_frame_get_buffer(self.sw_frame, 32);
                            if ret < 0 {
                                (*context).Unmap(&dst_res, 0);
                                return Err(ff_err(ret));
                            }
                        }

                        if sw_format == ffi::AVPixelFormat::AV_PIX_FMT_BGRA {
                            let dst = (*self.sw_frame).data[0];
                            let dst_stride = (*self.sw_frame).linesize[0] as usize;
                            if src_stride == dst_stride {
                                ptr::copy_nonoverlapping(src_ptr, dst, src_stride * h as usize);
                            } else {
                                let w_bytes = w as usize * 4;
                                for row in 0..h as usize {
                                    ptr::copy_nonoverlapping(
                                        src_ptr.add(row * src_stride),
                                        dst.add(row * dst_stride),
                                        w_bytes,
                                    );
                                }
                            }
                        } else {
                            if self.sws_ctx.is_null() {
                                self.sws_ctx = ffi::sws_getContext(
                                    w,
                                    h,
                                    ffi::AVPixelFormat::AV_PIX_FMT_BGRA,
                                    w,
                                    h,
                                    sw_format,
                                    SWS_FAST_BILINEAR,
                                    ptr::null_mut(),
                                    ptr::null_mut(),
                                    ptr::null(),
                                );
                                if self.sws_ctx.is_null() {
                                    (*context).Unmap(&dst_res, 0);
                                    return Err(MediaError::EncodeError("Failed to create SwsContext for BGRA→YUV/NV12".into()));
                                }
                                let inv_table = ffi::sws_getCoefficients(ffi::SWS_CS_DEFAULT);
                                let table = ffi::sws_getCoefficients(ffi::SWS_CS_ITU709);
                                ffi::sws_setColorspaceDetails(
                                    self.sws_ctx,
                                    inv_table,
                                    1, // srcRange = Full
                                    table,
                                    1,       // dstRange = Full
                                    0,       // brightness
                                    1 << 16, // contrast
                                    1 << 16, // saturation
                                );
                            }

                            let src_data: [*const u8; 4] = [src_ptr, ptr::null(), ptr::null(), ptr::null()];
                            let src_linesize: [libc::c_int; 4] = [src_stride as libc::c_int, 0, 0, 0];

                            ffi::sws_scale(
                                self.sws_ctx,
                                src_data.as_ptr(),
                                src_linesize.as_ptr(),
                                0,
                                h,
                                (*self.sw_frame).data.as_ptr() as *const *mut u8,
                                (*self.sw_frame).linesize.as_ptr(),
                            );
                        }

                        (*context).Unmap(&dst_res, 0);

                        (*self.sw_frame).width = w;
                        (*self.sw_frame).height = h;
                        (*self.sw_frame).pts = pts_us as i64;
                    }

                    #[cfg(not(target_os = "windows"))]
                    {
                        return Err(MediaError::EncodeError("D3D11 zero-copy not supported on non-Windows".into()));
                    }

                    self.sw_frame
                } else { unsafe {
                    ffi::av_frame_unref(self.frame);
                    let ret = ffi::av_hwframe_get_buffer(self.hw_frames_ctx, self.frame, 0);
                    if ret < 0 {
                        log::error!("av_hwframe_get_buffer failed: {}", ret);
                        return Err(ff_err(ret));
                    }

                    let dst_tex_ptr = (*self.frame).data[0] as *mut std::ffi::c_void;
                    let dst_idx = (*self.frame).data[1] as u32;

                    use windows::Win32::Graphics::Direct3D11::{
                        ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, ID3D11Resource,
                        ID3D11VideoDevice, ID3D11VideoContext, ID3D11VideoProcessor,
                        ID3D11VideoProcessorEnumerator, ID3D11VideoProcessorInputView,
                        ID3D11VideoProcessorOutputView, D3D11_VIDEO_PROCESSOR_CONTENT_DESC,
                        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC, D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC,
                        D3D11_VIDEO_PROCESSOR_STREAM, D3D11_VIDEO_FRAME_FORMAT,
                        D3D11_VIDEO_USAGE, D3D11_VPIV_DIMENSION, D3D11_VPOV_DIMENSION,
                        D3D11_TEX2D_VPIV, D3D11_TEX2D_VPOV, D3D11_TEX2D_ARRAY_VPOV,
                        D3D11_VIDEO_USAGE_PLAYBACK_NORMAL, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
                        D3D11_VPIV_DIMENSION_TEXTURE2D, D3D11_VPOV_DIMENSION_TEXTURE2D,
                        D3D11_VPOV_DIMENSION_TEXTURE2DARRAY,
                        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0, D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0,
                    };
                    use windows::Win32::Graphics::Dxgi::Common::DXGI_RATIONAL;
                    use windows::core::Interface;

                    let context_ptr = config.d3d11_context.ok_or_else(|| {
                        MediaError::EncodeError("D3D11 context not available in config".into())
                    })? as *mut std::ffi::c_void;

                    let context = std::mem::ManuallyDrop::new(ID3D11DeviceContext::from_raw(context_ptr));
                    let src_tex = std::mem::ManuallyDrop::new(ID3D11Texture2D::from_raw(*texture));
                    let dst_tex = std::mem::ManuallyDrop::new(ID3D11Texture2D::from_raw(dst_tex_ptr));

                    let res = (|| -> Result<(), MediaError> {
                        let mut src_desc = windows::Win32::Graphics::Direct3D11::D3D11_TEXTURE2D_DESC::default();
                        let mut dst_desc = windows::Win32::Graphics::Direct3D11::D3D11_TEXTURE2D_DESC::default();
                        (*src_tex).GetDesc(&mut src_desc);
                        (*dst_tex).GetDesc(&mut dst_desc);

                        static mut COPY_LOG_COUNT: u32 = 0;
                        COPY_LOG_COUNT += 1;
                        if COPY_LOG_COUNT == 1 || COPY_LOG_COUNT % 120 == 0 {
                            log::info!(
                                "D3D11 zero-copy textures (copy #{}): src={:?}x{:?} format={} ArraySize={} | dst={:?}x{:?} format={} ArraySize={} dst_idx={}",
                                COPY_LOG_COUNT,
                                src_desc.Width, src_desc.Height, src_desc.Format.0, src_desc.ArraySize,
                                dst_desc.Width, dst_desc.Height, dst_desc.Format.0, dst_desc.ArraySize,
                                dst_idx
                            );
                        }

                        if src_desc.Format == dst_desc.Format {
                            if let (Ok(src_res), Ok(dst_res)) = ((*src_tex).cast::<ID3D11Resource>(), (*dst_tex).cast::<ID3D11Resource>()) {
                                (*context).CopySubresourceRegion(
                                    &dst_res,
                                    dst_idx,
                                    0, 0, 0,
                                    &src_res,
                                    *array_index,
                                    None,
                                );
                                (*context).Flush();
                            } else {
                                return Err(MediaError::EncodeError("Failed to cast textures to ID3D11Resource".into()));
                            }
                        } else {
                            // Video Processor path
                            let device_ptr = config.d3d11_device.ok_or_else(|| {
                                MediaError::EncodeError("D3D11 device not available in config".into())
                            })? as *mut std::ffi::c_void;

                            if self.video_device.is_none() {
                                let d3d_device = std::mem::ManuallyDrop::new(ID3D11Device::from_raw(device_ptr));
                                let v_device = (*d3d_device).cast::<ID3D11VideoDevice>().map_err(|e| {
                                    MediaError::EncodeError(format!("Cast ID3D11Device to ID3D11VideoDevice failed: {e}"))
                                })?;
                                self.video_device = Some(v_device);
                            }
                            let video_device = self.video_device.as_ref().unwrap();

                            if self.video_context.is_none() {
                                let v_context = (*context).cast::<ID3D11VideoContext>().map_err(|e| {
                                    MediaError::EncodeError(format!("Cast ID3D11DeviceContext to ID3D11VideoContext failed: {e}"))
                                })?;
                                self.video_context = Some(v_context);
                            }
                            let video_context = self.video_context.as_ref().unwrap();

                            if self.video_processor.is_none() {
                                let desc = D3D11_VIDEO_PROCESSOR_CONTENT_DESC {
                                    InputFrameFormat: D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
                                    InputFrameRate: DXGI_RATIONAL { Numerator: config.fps, Denominator: 1 },
                                    InputWidth: config.width,
                                    InputHeight: config.height,
                                    OutputFrameRate: DXGI_RATIONAL { Numerator: config.fps, Denominator: 1 },
                                    OutputWidth: config.width,
                                    OutputHeight: config.height,
                                    Usage: D3D11_VIDEO_USAGE_PLAYBACK_NORMAL,
                                };
                                let enumerator = video_device.CreateVideoProcessorEnumerator(&desc).map_err(|e| {
                                    MediaError::EncodeError(format!("CreateVideoProcessorEnumerator failed: {e}"))
                                })?;
                                let processor = video_device.CreateVideoProcessor(&enumerator, 0).map_err(|e| {
                                    MediaError::EncodeError(format!("CreateVideoProcessor failed: {e}"))
                                })?;
                                self.video_enumerator = Some(enumerator);
                                self.video_processor = Some(processor);
                            }
                            let video_processor = self.video_processor.as_ref().unwrap();
                            let video_enumerator = self.video_enumerator.as_ref().unwrap();

                            let mut recreate_views = false;
                            if self.video_input_view.is_none() || self.cached_src_tex != *texture as usize {
                                recreate_views = true;
                            }
                            if self.video_output_view.is_none() || self.cached_dst_tex != dst_tex_ptr as usize || self.cached_dst_idx != dst_idx {
                                recreate_views = true;
                            }

                            if recreate_views {
                                let input_desc = D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC {
                                    FourCC: 0,
                                    ViewDimension: D3D11_VPIV_DIMENSION_TEXTURE2D,
                                    Anonymous: D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0 {
                                        Texture2D: D3D11_TEX2D_VPIV {
                                            MipSlice: 0,
                                            ArraySlice: *array_index,
                                        },
                                    },
                                };
                                let in_res = (*src_tex).cast::<ID3D11Resource>().map_err(|e| {
                                    MediaError::EncodeError(format!("Cast src_tex to ID3D11Resource failed: {e}"))
                                })?;
                                let mut in_view = None;
                                video_device.CreateVideoProcessorInputView(&in_res, video_enumerator, &input_desc, Some(&mut in_view)).map_err(|e| {
                                    MediaError::EncodeError(format!("CreateVideoProcessorInputView failed: {e}"))
                                })?;
                                self.video_input_view = in_view;
                                self.cached_src_tex = *texture as usize;

                                let output_desc = if dst_desc.ArraySize > 1 {
                                    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC {
                                        ViewDimension: D3D11_VPOV_DIMENSION_TEXTURE2DARRAY,
                                        Anonymous: D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0 {
                                            Texture2DArray: D3D11_TEX2D_ARRAY_VPOV {
                                                MipSlice: 0,
                                                FirstArraySlice: dst_idx,
                                                ArraySize: 1,
                                            },
                                        },
                                    }
                                } else {
                                    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC {
                                        ViewDimension: D3D11_VPOV_DIMENSION_TEXTURE2D,
                                        Anonymous: D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0 {
                                            Texture2D: D3D11_TEX2D_VPOV {
                                                MipSlice: 0,
                                            },
                                        },
                                    }
                                };
                                let out_res = (*dst_tex).cast::<ID3D11Resource>().map_err(|e| {
                                    MediaError::EncodeError(format!("Cast dst_tex to ID3D11Resource failed: {e}"))
                                })?;
                                let mut out_view = None;
                                video_device.CreateVideoProcessorOutputView(&out_res, video_enumerator, &output_desc, Some(&mut out_view)).map_err(|e| {
                                    MediaError::EncodeError(format!("CreateVideoProcessorOutputView failed: {e}"))
                                })?;
                                self.video_output_view = out_view;
                                self.cached_dst_tex = dst_tex_ptr as usize;
                                self.cached_dst_idx = dst_idx;
                            }

                            let input_view = self.video_input_view.as_ref().unwrap();
                            let output_view = self.video_output_view.as_ref().unwrap();

                            let mut stream = D3D11_VIDEO_PROCESSOR_STREAM::default();
                            stream.Enable = true.into();
                            stream.pInputSurface = std::mem::ManuallyDrop::new(Some(input_view.clone()));
                            let streams = [stream];

                            video_context.VideoProcessorBlt(video_processor, output_view, 0, &streams).map_err(|e| {
                                MediaError::EncodeError(format!("VideoProcessorBlt failed: {e}"))
                            })?;
                            (*context).Flush();
                        }
                        Ok(())
                    })();

                    res?;

                    (*self.frame).width = config.width as libc::c_int;
                    (*self.frame).height = config.height as libc::c_int;
                    self.frame
                } }
            }

            #[cfg(target_os = "linux")]
            GpuBuffer::DmaBuf { .. } => {
                log::warn!(
                    "DMA-BUF zero-copy import not yet implemented; \
                     please use CpuBuffer path for now"
                );
                return Err(MediaError::EncodeError(
                    "DMA-BUF zero-copy import not yet implemented".into(),
                ));
            }

            #[cfg(target_os = "linux")]
            GpuBuffer::CudaPointer {
                ptr,
                width,
                height,
                stride,
                format,
                ..
            } => {
                let copier = get_cuda_copier().ok_or_else(|| {
                    MediaError::EncodeError("CUDA driver library not loaded".into())
                })?;

                let src_y = *ptr as u64;
                let src_stride = *stride as usize;
                let h = *height as usize;

                unsafe {
                    // Extract FFmpeg's CUDA context
                    if self.hw_device_ctx.is_null() {
                        return Err(MediaError::EncodeError("hw_device_ctx is null".into()));
                    }
                    let buffer_ref = self.hw_device_ctx as *mut ffi::AVBufferRef;
                    if buffer_ref.is_null() || (*buffer_ref).data.is_null() {
                        return Err(MediaError::EncodeError("hw_device_ctx data is null".into()));
                    }
                    let device_ctx = (*buffer_ref).data as *mut ffi::AVHWDeviceContext;
                    if device_ctx.is_null() || (*device_ctx).hwctx.is_null() {
                        return Err(MediaError::EncodeError("hwctx is null".into()));
                    }
                    let ffmpeg_ctx = *((*device_ctx).hwctx as *mut *mut std::ffi::c_void);
                    if ffmpeg_ctx.is_null() {
                        return Err(MediaError::EncodeError("ffmpeg_ctx is null".into()));
                    }

                    // Save current context, and set FFmpeg's context as current
                    let mut old_ctx: *mut std::ffi::c_void = std::ptr::null_mut();
                    (copier.cu_ctx_get_current)(&mut old_ctx);
                    let res = (copier.cu_ctx_set_current)(ffmpeg_ctx);
                    if res != 0 {
                        return Err(MediaError::EncodeError(format!(
                            "cuCtxSetCurrent failed: {}",
                            res
                        )));
                    }

                    // Use RAII to restore context on scope exit
                    struct ContextRestorer<'a> {
                        copier: &'a CudaCopier,
                        old_ctx: *mut std::ffi::c_void,
                    }
                    impl<'a> Drop for ContextRestorer<'a> {
                        fn drop(&mut self) {
                            unsafe {
                                (self.copier.cu_ctx_set_current)(self.old_ctx);
                            }
                        }
                    }
                    let _restorer = ContextRestorer { copier, old_ctx };

                    ffi::av_frame_unref(self.frame);
                    let ret = ffi::av_hwframe_get_buffer(self.hw_frames_ctx, self.frame, 0);
                    if ret < 0 {
                        log::error!("av_hwframe_get_buffer failed: {}", ret);
                        return Err(ff_err(ret));
                    }

                    let dst_y = (*self.frame).data[0] as u64;
                    let dst_stride_y = (*self.frame).linesize[0] as usize;

                    if *format == PixelFormat::BGRA {
                        if src_stride == dst_stride_y {
                            let res = (copier.cu_memcpy_dtod)(dst_y, src_y, src_stride * h);
                            if res != 0 {
                                return Err(MediaError::EncodeError(format!(
                                    "cuMemcpyDtoD BGRA plane failed: {}",
                                    res
                                )));
                            }
                        } else {
                            let w_bytes = *width as usize * 4;
                            for row in 0..h {
                                let src_row = src_y + (row * src_stride) as u64;
                                let dst_row = dst_y + (row * dst_stride_y) as u64;
                                let res = (copier.cu_memcpy_dtod)(dst_row, src_row, w_bytes);
                                if res != 0 {
                                    return Err(MediaError::EncodeError(format!(
                                        "cuMemcpyDtoD BGRA row {} failed: {}",
                                        row, res
                                    )));
                                }
                            }
                        }
                    } else {
                        let src_uv = src_y + (src_stride as u64 * h as u64);
                        let dst_uv = (*self.frame).data[1] as u64;
                        let dst_stride_uv = (*self.frame).linesize[1] as usize;
                        let w = *width as usize;

                        if src_stride == dst_stride_y && src_stride == dst_stride_uv {
                            let res = (copier.cu_memcpy_dtod)(dst_y, src_y, src_stride * h);
                            if res != 0 {
                                return Err(MediaError::EncodeError(format!(
                                    "cuMemcpyDtoD Y plane failed: {}",
                                    res
                                )));
                            }
                            let res = (copier.cu_memcpy_dtod)(dst_uv, src_uv, src_stride * (h / 2));
                            if res != 0 {
                                return Err(MediaError::EncodeError(format!(
                                    "cuMemcpyDtoD UV plane failed: {}",
                                    res
                                )));
                            }
                        } else {
                            for row in 0..h {
                                let src_row = src_y + (row * src_stride) as u64;
                                let dst_row = dst_y + (row * dst_stride_y) as u64;
                                let res = (copier.cu_memcpy_dtod)(dst_row, src_row, w);
                                if res != 0 {
                                    return Err(MediaError::EncodeError(format!(
                                        "cuMemcpyDtoD Y row {} failed: {}",
                                        row, res
                                    )));
                                }
                            }
                            for row in 0..(h / 2) {
                                let src_row = src_uv + (row * src_stride) as u64;
                                let dst_row = dst_uv + (row * dst_stride_uv) as u64;
                                let res = (copier.cu_memcpy_dtod)(dst_row, src_row, w);
                                if res != 0 {
                                    return Err(MediaError::EncodeError(format!(
                                        "cuMemcpyDtoD UV row {} failed: {}",
                                        row, res
                                    )));
                                }
                            }
                        }
                    }
                }
                self.frame
            }

            GpuBuffer::CpuBuffer {
                data,
                stride,
                format,
                width,
                height,
            } => {
                let w = *width as libc::c_int;
                let h = *height as libc::c_int;
                let src_stride = *stride as usize;

                let sw_format = match hw_type {
                    HwAccelType::Nvenc => ffi::AVPixelFormat::AV_PIX_FMT_BGRA,
                    HwAccelType::Software => ffi::AVPixelFormat::AV_PIX_FMT_YUV420P,
                    _ => ffi::AVPixelFormat::AV_PIX_FMT_NV12,
                };
                unsafe {
                    let ret = ffi::av_frame_make_writable(self.sw_frame);
                    if ret < 0 {
                        ffi::av_frame_unref(self.sw_frame);
                        (*self.sw_frame).width = w;
                        (*self.sw_frame).height = h;
                        (*self.sw_frame).format = sw_format as libc::c_int;
                        let ret = ffi::av_frame_get_buffer(self.sw_frame, 32);
                        if ret < 0 {
                            return Err(ff_err(ret));
                        }
                    }
                }

                match format {
                    PixelFormat::NV12 => unsafe {
                        let y_dst = (*self.sw_frame).data[0];
                        let y_dst_stride = (*self.sw_frame).linesize[0] as usize;

                        for row in 0..h as usize {
                            let src_offset = row * src_stride;
                            let dst_offset = row * y_dst_stride;
                            let copy_len = w as usize;
                            if src_offset + copy_len <= data.len() {
                                ptr::copy_nonoverlapping(
                                    data.as_ptr().add(src_offset),
                                    y_dst.add(dst_offset),
                                    copy_len,
                                );
                            }
                        }

                        let uv_dst = (*self.sw_frame).data[1];
                        let uv_dst_stride = (*self.sw_frame).linesize[1] as usize;
                        let uv_h = (h / 2) as usize;
                        let y_plane_size = h as usize * src_stride;

                        for row in 0..uv_h {
                            let src_offset = y_plane_size + row * src_stride;
                            let dst_offset = row * uv_dst_stride;
                            let copy_len = w as usize;
                            if src_offset + copy_len <= data.len() {
                                ptr::copy_nonoverlapping(
                                    data.as_ptr().add(src_offset),
                                    uv_dst.add(dst_offset),
                                    copy_len,
                                );
                            }
                        }
                    },
                    PixelFormat::BGRA => {
                        if sw_format == ffi::AVPixelFormat::AV_PIX_FMT_BGRA {
                            unsafe {
                                let dst = (*self.sw_frame).data[0];
                                let dst_stride = (*self.sw_frame).linesize[0] as usize;
                                if src_stride == dst_stride {
                                    ptr::copy_nonoverlapping(
                                        data.as_ptr(),
                                        dst,
                                        src_stride * h as usize,
                                    );
                                } else {
                                    let w_bytes = w as usize * 4;
                                    for row in 0..h as usize {
                                        ptr::copy_nonoverlapping(
                                            data.as_ptr().add(row * src_stride),
                                            dst.add(row * dst_stride),
                                            w_bytes,
                                        );
                                    }
                                }
                            }
                        } else {
                            if self.sws_ctx.is_null() {
                                self.sws_ctx = unsafe {
                                    ffi::sws_getContext(
                                        w,
                                        h,
                                        ffi::AVPixelFormat::AV_PIX_FMT_BGRA,
                                        w,
                                        h,
                                        sw_format,
                                        SWS_FAST_BILINEAR,
                                        ptr::null_mut(),
                                        ptr::null_mut(),
                                        ptr::null(),
                                    )
                                };
                                if self.sws_ctx.is_null() {
                                    return Err(MediaError::EncodeError(
                                        "Failed to create SwsContext for BGRA→NV12".into(),
                                    ));
                                }
                                unsafe {
                                    let inv_table = ffi::sws_getCoefficients(ffi::SWS_CS_DEFAULT);
                                    let table = ffi::sws_getCoefficients(ffi::SWS_CS_ITU709);
                                    ffi::sws_setColorspaceDetails(
                                        self.sws_ctx,
                                        inv_table,
                                        1, // srcRange = Full
                                        table,
                                        1,       // dstRange = Full
                                        0,       // brightness
                                        1 << 16, // contrast
                                        1 << 16, // saturation
                                    );
                                }
                                log::info!("Created SIMD SwsContext for BGRA→{:?} {}x{} with BT.709 colorspace", sw_format, w, h);
                            }

                            let src_data: [*const u8; 4] =
                                [data.as_ptr(), ptr::null(), ptr::null(), ptr::null()];
                            let src_linesize: [libc::c_int; 4] =
                                [(src_stride) as libc::c_int, 0, 0, 0];

                            unsafe {
                                ffi::sws_scale(
                                    self.sws_ctx,
                                    src_data.as_ptr(),
                                    src_linesize.as_ptr(),
                                    0,
                                    h,
                                    (*self.sw_frame).data.as_ptr() as *const *mut u8,
                                    (*self.sw_frame).linesize.as_ptr(),
                                );
                            }
                        }
                    }
                    PixelFormat::P010 => {
                        log::warn!("P010 pixel format not yet supported for encoding");
                        return Err(MediaError::EncodeError(
                            "P010 pixel format not supported".into(),
                        ));
                    }
                }

                if use_hw && !self.hw_frames_ctx.is_null() {
                    unsafe {
                        ffi::av_frame_unref(self.frame);

                        let ret = ffi::av_hwframe_get_buffer(self.hw_frames_ctx, self.frame, 0);
                        if ret < 0 {
                            log::error!("av_hwframe_get_buffer failed: {}", ret);
                            return Err(ff_err(ret));
                        }

                        let ret = ffi::av_hwframe_transfer_data(self.frame, self.sw_frame, 0);
                        if ret < 0 {
                            log::error!("av_hwframe_transfer_data failed: {}", ret);
                            return Err(ff_err(ret));
                        }

                        (*self.frame).width = w;
                        (*self.frame).height = h;
                    }
                    self.frame
                } else {
                    self.sw_frame
                }
            }

            #[allow(unreachable_patterns)]
            _ => {
                log::error!("Unsupported GpuBuffer variant for encoding");
                return Err(MediaError::EncodeError(
                    "Unsupported buffer type for this platform".into(),
                ));
            }
        };

        // Set PTS and colorspace characteristics.
        unsafe {
            (*send_frame).pts = self.frame_count as i64;
            (*send_frame).colorspace = ffi::AVColorSpace::AVCOL_SPC_BT709;
            (*send_frame).color_range = ffi::AVColorRange::AVCOL_RANGE_JPEG;
            (*send_frame).color_primaries = ffi::AVColorPrimaries::AVCOL_PRI_BT709;
            (*send_frame).color_trc = ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT709;
        }

        // Force keyframe if requested.
        if self.force_keyframe {
            unsafe {
                (*send_frame).pict_type = ffi::AVPictureType::AV_PICTURE_TYPE_I;
                (*send_frame).flags |= ffi::AV_FRAME_FLAG_KEY as libc::c_int;
            }
            log::info!("Forcing keyframe at frame #{}", self.frame_count);
        } else {
            unsafe {
                (*send_frame).pict_type = ffi::AVPictureType::AV_PICTURE_TYPE_NONE;
                (*send_frame).flags &= !(ffi::AV_FRAME_FLAG_KEY as libc::c_int);
            }
        }

        // Send the frame to the encoder.
        let ret = unsafe { ffi::avcodec_send_frame(self.codec_ctx, send_frame) };
        if ret < 0 {
            log::error!(
                "avcodec_send_frame failed at frame #{}: {}",
                self.frame_count,
                ret
            );
            unsafe {
                ffi::av_frame_unref(self.frame);
                ffi::av_frame_unref(self.sw_frame);
            }
            return Err(ff_err(ret));
        }

        // Drain output packets, prepending SPS/PPS to any IDR keyframes.
        let result = self.drain_packets(config.codec, config.fps, true);

        // Clean up frame state for this iteration.
        unsafe {
            ffi::av_frame_unref(self.frame);
            ffi::av_frame_unref(self.sw_frame);
        }

        self.frame_count += 1;
        self.force_keyframe = false;

        log::trace!(
            "Encoded frame #{} pts={}us -> {} packets",
            self.frame_count - 1,
            pts_us,
            result.as_ref().map(|v| v.len()).unwrap_or(0),
        );

        result
    }

    fn request_keyframe(&mut self) {
        log::debug!("Keyframe requested for next frame");
        self.force_keyframe = true;
    }

    fn set_bitrate(&mut self, bitrate_kbps: u32) -> Result<(), MediaError> {
        if !self.initialized {
            return Err(MediaError::EncoderNotInitialized);
        }

        log::info!("Setting bitrate to {} kbps", bitrate_kbps);

        unsafe {
            let bitrate = (bitrate_kbps as i64) * 1000;
            (*self.codec_ctx).bit_rate = bitrate;
            (*self.codec_ctx).rc_max_rate = bitrate;
            // Use 1 second of video data as buffer size (same as initial encoder setup).
            // The old formula `(bitrate * 2) / fps` was far too small and caused
            // NVENC to produce corrupted output after bitrate changes.
            (*self.codec_ctx).rc_buffer_size = bitrate as libc::c_int;
        }

        if let Some(config) = &mut self.config {
            config.bitrate_kbps = bitrate_kbps;
        }

        // Force a keyframe after bitrate change so the decoder can recover cleanly
        self.force_keyframe = true;

        Ok(())
    }

    fn encoder_info(&self) -> EncoderInfo {
        self.info.clone().unwrap_or(EncoderInfo {
            name: "uninitialized".to_string(),
            hw_type: HwAccelType::Software,
            supported_codecs: vec![],
        })
    }

    fn flush(&mut self) -> Result<Vec<EncodedVideoFrame>, MediaError> {
        if !self.initialized {
            return Ok(vec![]);
        }

        log::debug!("Flushing encoder (sending NULL frame)");

        let config = self.config.as_ref().unwrap();

        // SAFETY: Sending a null frame signals end-of-stream to the encoder.
        // The codec context is valid because we checked `initialized`.
        let ret = unsafe { ffi::avcodec_send_frame(self.codec_ctx, ptr::null()) };
        if ret < 0 && ret != ffi::AVERROR_EOF {
            log::warn!("avcodec_send_frame(NULL) returned {}", ret);
            // Some encoders return AVERROR_EOF if already flushed; not fatal.
        }

        // Drain all remaining packets (flush does not need SPS/PPS prepend).
        let result = self.drain_packets(config.codec, config.fps, false);

        log::debug!(
            "Flush complete: {} packets drained",
            result.as_ref().map(|v| v.len()).unwrap_or(0),
        );

        result
    }

    fn shutdown(&mut self) {
        if !self.initialized {
            return;
        }

        log::info!("Shutting down FFmpeg encoder");

        // SAFETY: All pointers below were set during initialize() and are
        // either valid or null. FFmpeg's free functions are null-safe
        // (they check internally), and we null them out after freeing to
        // prevent double-free.
        unsafe {
            if !self.codec_ctx.is_null() {
                ffi::avcodec_free_context(&mut self.codec_ctx);
                self.codec_ctx = ptr::null_mut();
            }

            if !self.hw_frames_ctx.is_null() {
                ffi::av_buffer_unref(&mut self.hw_frames_ctx);
                self.hw_frames_ctx = ptr::null_mut();
            }

            if !self.hw_device_ctx.is_null() {
                ffi::av_buffer_unref(&mut self.hw_device_ctx);
                self.hw_device_ctx = ptr::null_mut();
            }

            if !self.frame.is_null() {
                ffi::av_frame_free(&mut self.frame);
                self.frame = ptr::null_mut();
            }

            if !self.sw_frame.is_null() {
                ffi::av_frame_free(&mut self.sw_frame);
                self.sw_frame = ptr::null_mut();
            }

            if !self.packet.is_null() {
                ffi::av_packet_free(&mut self.packet);
                self.packet = ptr::null_mut();
            }

            if !self.sws_ctx.is_null() {
                ffi::sws_freeContext(self.sws_ctx);
                self.sws_ctx = ptr::null_mut();
            }
        }

        self.initialized = false;
        self.config = None;
        self.info = None;
        self.frame_count = 0;
        self.force_keyframe = false;
        #[cfg(target_os = "windows")]
        {
            self.video_device = None;
            self.video_context = None;
            self.video_processor = None;
            self.video_enumerator = None;
            self.video_input_view = None;
            self.video_output_view = None;
            self.cached_src_tex = 0;
            self.cached_dst_tex = 0;
            self.cached_dst_idx = 0;
        }

        log::info!("FFmpeg encoder shut down successfully");
    }
}

impl Drop for FfmpegEncoder {
    fn drop(&mut self) {
        self.shutdown();
    }
}
