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
                    opt_set("profile", "high"); // Changed from baseline to high (enables CABAC)
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
                    opt_set("profile", "high");
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
                    opt_set("profile", "high");
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
                opt_set("header_insertion_mode", "idr");
                if codec == VideoCodec::H264 {
                    opt_set("profile", "high");
                } else if codec == VideoCodec::H265 {
                    opt_set("profile", "main");
                }
                log::info!("Applied AMF encoder options");
            }
            HwAccelType::VideoToolbox => {
                opt_set("realtime", "1");
                if codec == VideoCodec::H264 {
                    opt_set("profile", "high");
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
                    opt_set("profile", "high"); // Changed from baseline to high (enables CABAC)
                } else if codec == VideoCodec::H265 {
                    opt_set("profile", "main");
                }
                log::info!("Applied software encoder options");
            }
        }
    }

    /// Receive all available packets from the encoder and convert them to
    /// [`EncodedVideoFrame`]s.
    fn drain_packets(
        codec_ctx: *mut ffi::AVCodecContext,
        packet: *mut ffi::AVPacket,
        codec: VideoCodec,
        fps: u32,
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
            let ret = unsafe { ffi::avcodec_receive_packet(codec_ctx, packet) };

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
                let pkt = &*packet;
                if pkt.data.is_null() || pkt.size <= 0 {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(pkt.data, pkt.size as usize).to_vec()
                }
            };

            let is_key = unsafe { (*packet).flags & ffi::AV_PKT_FLAG_KEY != 0 };
            let frame_type = if is_key {
                FrameType::Key
            } else {
                FrameType::Inter
            };

            let pts = unsafe { (*packet).pts as u64 };

            frames.push(EncodedVideoFrame {
                data,
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
                ffi::av_packet_unref(packet);
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
                100_000_000
            };
            (*codec_ctx).max_b_frames = 0;
            (*codec_ctx).thread_count = 1;

            if is_hw_encoder(hw_type) {
                (*codec_ctx).pix_fmt = hw_pix_fmt(hw_type);
            } else {
                (*codec_ctx).pix_fmt = ffi::AVPixelFormat::AV_PIX_FMT_YUV420P;
            }

            if config.low_latency {
                (*codec_ctx).flags |= ffi::AV_CODEC_FLAG_LOW_DELAY as libc::c_int;
            }
        }

        let mut hw_device_ctx: *mut ffi::AVBufferRef = ptr::null_mut();
        let mut hw_frames_ctx: *mut ffi::AVBufferRef = ptr::null_mut();

        if is_hw_encoder(hw_type) {
            let (device_type, device_path): (ffi::AVHWDeviceType, Option<CString>) = match hw_type {
                HwAccelType::Vaapi => (
                    ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                    Some(CString::new("/dev/dri/renderD128").unwrap()),
                ),
                HwAccelType::Nvenc => (ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA, None),
                HwAccelType::Qsv => (ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_QSV, None),
                HwAccelType::Amf => (ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA, None),
                _ => (ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE, None),
            };

            let device_ptr = device_path
                .as_ref()
                .map(|p| p.as_ptr())
                .unwrap_or(ptr::null());

            let ret = unsafe {
                ffi::av_hwdevice_ctx_create(
                    &mut hw_device_ctx,
                    device_type,
                    device_ptr,
                    ptr::null_mut(),
                    0,
                )
            };

            if ret < 0 || hw_device_ctx.is_null() {
                unsafe {
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
                "Created HW device context for {:?} (device={:?})",
                hw_type,
                device_path.as_ref().map(|p| p.to_str().unwrap_or("?"))
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
                (*frames_ctx).format = hw_pix_fmt(hw_type);
                let sw_format = match hw_type {
                    HwAccelType::Nvenc => ffi::AVPixelFormat::AV_PIX_FMT_BGRA,
                    _ => ffi::AVPixelFormat::AV_PIX_FMT_NV12,
                };
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
            }

            log::info!("HW frames context initialized: pool_size=4 sw_format=NV12");
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

                if use_hw {
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
            log::debug!("Forcing keyframe at frame #{}", self.frame_count);
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

        // Drain output packets.
        let result = Self::drain_packets(self.codec_ctx, self.packet, config.codec, config.fps);

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

        // Drain all remaining packets.
        let result = Self::drain_packets(self.codec_ctx, self.packet, config.codec, config.fps);

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

        log::info!("FFmpeg encoder shut down successfully");
    }
}

impl Drop for FfmpegEncoder {
    fn drop(&mut self) {
        self.shutdown();
    }
}
