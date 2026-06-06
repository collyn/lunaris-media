//! Video encoding subsystem.
//!
//! This module defines the [`VideoEncoder`] trait for encoding captured GPU
//! frames into a compressed bitstream (e.g., Annex-B H.264). The primary
//! implementation wraps FFmpeg's hardware-accelerated encoders (VAAPI, NVENC,
//! QSV, AMF, VideoToolbox).
//!
//! Use [`create_encoder`] to obtain the best available encoder for the current
//! platform, or [`list_available_encoders`] to enumerate all detected hardware
//! encoders.

#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
pub mod ffmpeg;

use crate::capture::gpu_buffer::GpuBuffer;
use crate::error::MediaError;
use crate::types::*;

/// Configuration for initializing a [`VideoEncoder`].
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Video codec to encode.
    pub codec: VideoCodec,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Target frame rate.
    pub fps: u32,
    /// Target bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Enable low-latency tuning (disables look-ahead, reduces buffering).
    pub low_latency: bool,
    /// Keyframe interval in frames. `0` means automatic (`2 × fps`).
    pub keyframe_interval: u32,
    /// Preferred hardware acceleration type, or `None` for auto-detection.
    pub preferred_hw: Option<HwAccelType>,
    /// Optional Direct3D11 device pointer (cast to usize) for Windows zero-copy GPU encoding.
    pub d3d11_device: Option<usize>,
    /// Optional Direct3D11 device context pointer (cast to usize) for Windows zero-copy GPU encoding.
    pub d3d11_context: Option<usize>,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            codec: VideoCodec::H264,
            width: 1920,
            height: 1080,
            fps: 60,
            bitrate_kbps: 10_000,
            low_latency: true,
            keyframe_interval: 0,
            preferred_hw: None,
            d3d11_device: None,
            d3d11_context: None,
        }
    }
}

/// Trait for hardware-accelerated video encoders.
///
/// Implementations consume GPU-resident buffers from the capture subsystem and
/// produce encoded video frames. The encoder is designed to be driven from a
/// single thread/task; `Send` is required so the owning task can migrate between
/// executor threads.
pub trait VideoEncoder: Send {
    /// Initialize the encoder with the given configuration.
    ///
    /// Must be called before [`encode`](Self::encode). May probe the system for
    /// available hardware and select the best encoder automatically.
    fn initialize(&mut self, config: &EncoderConfig) -> Result<(), MediaError>;

    /// Encode a single GPU frame.
    ///
    /// Returns zero or more encoded frames (encoders may buffer internally).
    fn encode(&mut self, buffer: &GpuBuffer, pts_us: u64) -> Result<Vec<EncodedVideoFrame>, MediaError>;

    /// Request that the next encoded frame be an IDR/keyframe.
    fn request_keyframe(&mut self);

    /// Dynamically change the target bitrate.
    fn set_bitrate(&mut self, bitrate_kbps: u32) -> Result<(), MediaError>;

    /// Return metadata about this encoder instance.
    fn encoder_info(&self) -> EncoderInfo;

    /// Flush any internally buffered frames.
    fn flush(&mut self) -> Result<Vec<EncodedVideoFrame>, MediaError>;

    /// Shut down the encoder and release all resources.
    fn shutdown(&mut self);
}

/// Create the best available [`VideoEncoder`] for the current platform.
pub fn create_encoder() -> Result<Box<dyn VideoEncoder>, MediaError> {
    #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
    {
        return Ok(Box::new(ffmpeg::FfmpegEncoder::new()?));
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    Err(MediaError::PlatformNotSupported(
        "No encoder available on this platform".into(),
    ))
}

/// List all hardware-accelerated encoders detected on this system.
pub fn list_available_encoders() -> Vec<EncoderInfo> {
    #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
    {
        return ffmpeg::FfmpegEncoder::list_available();
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    Vec::new()
}
