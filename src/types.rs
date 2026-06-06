//! Shared data types used across the lunaris-media library.
//!
//! This module defines the core value types for video/audio frames, display
//! metadata, encoder information, and configuration structures. All types are
//! designed to be `Send`-safe and cheaply cloneable where possible.

use std::fmt;

// ---------------------------------------------------------------------------
// Video codec
// ---------------------------------------------------------------------------

/// Supported video codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    /// H.264 / AVC — widely supported, Phase 1 default.
    H264,
    /// H.265 / HEVC — better compression, Phase 2.
    H265,
    /// AV1 — next-gen royalty-free codec, Phase 3.
    AV1,
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoCodec::H264 => write!(f, "H.264"),
            VideoCodec::H265 => write!(f, "H.265"),
            VideoCodec::AV1 => write!(f, "AV1"),
        }
    }
}

// ---------------------------------------------------------------------------
// Pixel format
// ---------------------------------------------------------------------------

/// Pixel formats used for captured frames and encoder input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 4:2:0 planar with interleaved UV — standard for hardware encoders.
    NV12,
    /// 8-bit BGRA — common raw desktop capture format.
    BGRA,
    /// 10-bit 4:2:0 — used for HDR content.
    P010,
}

// ---------------------------------------------------------------------------
// Stream configuration
// ---------------------------------------------------------------------------

/// Configuration for a capture + encode stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamConfig {
    /// Capture / encode width in pixels.
    pub width: u32,
    /// Capture / encode height in pixels.
    pub height: u32,
    /// Target frame rate.
    pub fps: u32,
    /// Video codec to use.
    pub codec: VideoCodec,
    /// Target bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Pixel format for the encoder input.
    pub pixel_format: PixelFormat,
    /// Optional encoder preference (e.g. "nvenc", "vaapi", "software").
    pub preferred_encoder: Option<String>,
    /// Create and capture a temporary virtual display when supported.
    pub virtual_display: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fps: 60,
            codec: VideoCodec::H264,
            bitrate_kbps: 10_000,
            pixel_format: PixelFormat::NV12,
            preferred_encoder: None,
            virtual_display: false,
        }
    }
}

impl StreamConfig {
    fn normalize_encoder_preference(s: &str) -> String {
        s.trim().to_ascii_lowercase().replace('-', "_")
    }

    /// Parse an encoder preference string into an HwAccelType.
    pub fn parse_encoder_preference(s: &str) -> Option<HwAccelType> {
        match Self::normalize_encoder_preference(s).as_str() {
            "nvenc" | "native_nvenc" | "native_nvenc_d3d11" | "ffmpeg_nvenc" | "h264_nvenc"
            | "hevc_nvenc" | "av1_nvenc" => Some(HwAccelType::Nvenc),
            "vaapi" | "ffmpeg_vaapi" | "h264_vaapi" | "hevc_vaapi" | "av1_vaapi" => {
                Some(HwAccelType::Vaapi)
            }
            "qsv" | "ffmpeg_qsv" | "h264_qsv" | "hevc_qsv" | "av1_qsv" => Some(HwAccelType::Qsv),
            "amf" | "native_amf" | "native_amf_d3d11" | "ffmpeg_amf" | "h264_amf" | "hevc_amf"
            | "av1_amf" => Some(HwAccelType::Amf),
            "videotoolbox" | "ffmpeg_videotoolbox" | "h264_videotoolbox" | "hevc_videotoolbox" => {
                Some(HwAccelType::VideoToolbox)
            }
            "software" | "ffmpeg_software" | "libx264" | "libx265" | "libsvtav1" | "libaom_av1" => {
                Some(HwAccelType::Software)
            }
            "auto" | "gpu" | "native" | "native_gpu" | "ffmpeg" | "ffmpeg_gpu" | "" => None,
            _ => None,
        }
    }

    /// Whether the encoder preference explicitly requests the FFmpeg backend.
    pub fn encoder_prefers_ffmpeg(s: &str) -> bool {
        let normalized = Self::normalize_encoder_preference(s);
        normalized == "ffmpeg"
            || normalized == "ffmpeg_gpu"
            || normalized.starts_with("ffmpeg_")
            || matches!(
                normalized.as_str(),
                "h264_nvenc"
                    | "hevc_nvenc"
                    | "av1_nvenc"
                    | "h264_amf"
                    | "hevc_amf"
                    | "av1_amf"
                    | "h264_qsv"
                    | "hevc_qsv"
                    | "av1_qsv"
                    | "h264_vaapi"
                    | "hevc_vaapi"
                    | "av1_vaapi"
                    | "h264_videotoolbox"
                    | "hevc_videotoolbox"
                    | "libx264"
                    | "libx265"
                    | "libsvtav1"
                    | "libaom_av1"
            )
    }

    /// Whether the encoder preference explicitly requests native OS/GPU backend.
    pub fn encoder_prefers_native(s: &str) -> bool {
        let normalized = Self::normalize_encoder_preference(s);
        normalized == "gpu"
            || normalized == "native"
            || normalized == "native_gpu"
            || normalized.starts_with("native_")
    }
}

// ---------------------------------------------------------------------------
// Frame type
// ---------------------------------------------------------------------------

/// Indicates whether an encoded frame is a keyframe or an inter-frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Intra-coded / IDR frame — can be decoded independently.
    Key,
    /// Inter-coded frame — depends on previous frames.
    Inter,
}

// ---------------------------------------------------------------------------
// Encoded video frame
// ---------------------------------------------------------------------------

/// A single encoded video frame (Annex-B H.264 or equivalent bitstream).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedVideoFrame {
    /// Raw encoded bitstream data.
    pub data: Vec<u8>,
    /// Whether this is a keyframe or inter-frame.
    pub frame_type: FrameType,
    /// Presentation timestamp in microseconds.
    pub pts: u64,
    /// Frame duration in microseconds.
    pub duration: u64,
    /// Codec used to produce this frame.
    pub codec: VideoCodec,
}

// ---------------------------------------------------------------------------
// Encoded audio frame
// ---------------------------------------------------------------------------

/// A single encoded audio frame (typically Opus).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedAudioFrame {
    /// Raw encoded audio data.
    pub data: Vec<u8>,
    /// Presentation timestamp in microseconds.
    pub pts: u64,
    /// Frame duration in microseconds.
    pub duration: u64,
    /// Sample rate in Hz (e.g., 48000).
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u16,
}

// ---------------------------------------------------------------------------
// Cursor state
// ---------------------------------------------------------------------------

/// Current state of the mouse cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorState {
    /// Cursor X position in screen coordinates.
    pub x: i32,
    /// Cursor Y position in screen coordinates.
    pub y: i32,
    /// Whether the cursor is currently visible.
    pub visible: bool,
    /// Optional cursor image (provided when the shape changes).
    pub image: Option<CursorImage>,
}

/// RGBA image data for a custom cursor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Hotspot X offset within the image.
    pub hotspot_x: u32,
    /// Hotspot Y offset within the image.
    pub hotspot_y: u32,
    /// Raw RGBA pixel data (width × height × 4 bytes).
    pub rgba_data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Display info
// ---------------------------------------------------------------------------

/// Metadata about a display/monitor.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayInfo {
    /// Platform-specific display identifier.
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Display width in pixels.
    pub width: u32,
    /// Display height in pixels.
    pub height: u32,
    /// Refresh rate in Hz (e.g., 60.0, 144.0).
    pub refresh_rate: f64,
    /// Whether this is the primary display.
    pub is_primary: bool,
}

// ---------------------------------------------------------------------------
// Hardware acceleration
// ---------------------------------------------------------------------------

/// Hardware acceleration type for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HwAccelType {
    /// NVIDIA NVENC.
    Nvenc,
    /// Intel / AMD VAAPI (Linux).
    Vaapi,
    /// Intel Quick Sync Video.
    Qsv,
    /// AMD Advanced Media Framework.
    Amf,
    /// Apple VideoToolbox.
    VideoToolbox,
    /// Software fallback (libx264 / libx265).
    Software,
}

impl fmt::Display for HwAccelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HwAccelType::Nvenc => write!(f, "NVENC"),
            HwAccelType::Vaapi => write!(f, "VAAPI"),
            HwAccelType::Qsv => write!(f, "QSV"),
            HwAccelType::Amf => write!(f, "AMF"),
            HwAccelType::VideoToolbox => write!(f, "VideoToolbox"),
            HwAccelType::Software => write!(f, "Software"),
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder info
// ---------------------------------------------------------------------------

/// Information about an available encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderInfo {
    /// Encoder name (e.g., "h264_nvenc", "h264_vaapi").
    pub name: String,
    /// Hardware acceleration type used by this encoder.
    pub hw_type: HwAccelType,
    /// Set of codecs this encoder supports.
    pub supported_codecs: Vec<VideoCodec>,
}

// ---------------------------------------------------------------------------
// Audio capture configuration
// ---------------------------------------------------------------------------

/// Configuration for audio capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioCaptureConfig {
    /// Sample rate in Hz (e.g., 48000).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Audio frame size in milliseconds (e.g., 20).
    pub frame_size_ms: u32,
}

impl Default for AudioCaptureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
            frame_size_ms: 20,
        }
    }
}
