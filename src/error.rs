//! Error types for the lunaris-media library.
//!
//! All errors across the capture, encode, audio, cursor, and pipeline subsystems
//! are unified under [`MediaError`], enabling ergonomic `?` propagation throughout
//! the crate.

use thiserror::Error;

/// Unified error type for all media operations.
///
/// Each variant covers a specific failure domain, making it straightforward to
/// match on and handle errors from different subsystems independently.
#[derive(Error, Debug)]
pub enum MediaError {
    /// No display/monitor was detected on the system.
    #[error("No display found")]
    NoDisplayFound,

    /// Attempted to read a frame before starting capture.
    #[error("Capture not started")]
    CaptureNotStarted,

    /// Attempted to start capture when it is already running.
    #[error("Capture already started")]
    CaptureAlreadyStarted,

    /// No hardware encoder (VAAPI, NVENC, QSV, etc.) could be found.
    #[error("No hardware encoder available")]
    NoEncoderAvailable,

    /// The encoder has not been initialized yet.
    #[error("Encoder not initialized")]
    EncoderNotInitialized,

    /// Encoder initialization failed with the given reason.
    #[error("Encoder initialization failed: {0}")]
    EncoderInitFailed(String),

    /// An error occurred during screen capture.
    #[error("Capture error: {0}")]
    CaptureError(String),

    /// An error occurred during video encoding.
    #[error("Encode error: {0}")]
    EncodeError(String),

    /// An error occurred during audio capture or encoding.
    #[error("Audio error: {0}")]
    AudioError(String),

    /// An error occurred during cursor capture.
    #[error("Cursor error: {0}")]
    CursorError(String),

    /// An error occurred in the media pipeline orchestration.
    #[error("Pipeline error: {0}")]
    PipelineError(String),

    /// FFmpeg returned an error code.
    #[error("FFmpeg error: {code} - {message}")]
    FfmpegError {
        /// FFmpeg/libav error code (negative AVERROR value).
        code: i32,
        /// Human-readable description of the error.
        message: String,
    },

    /// The current platform does not support the requested operation.
    #[error("Platform not supported: {0}")]
    PlatformNotSupported(String),

    /// The user or system denied a required permission (e.g., screen recording).
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// An operation exceeded its deadline.
    #[error("Timeout")]
    Timeout,

    /// A captured frame was dropped (e.g., encoder back-pressure).
    #[error("Frame dropped")]
    FrameDropped,

    /// A standard I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
