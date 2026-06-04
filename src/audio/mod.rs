//! Audio capture subsystem.
//!
//! This module defines the [`AudioCapture`] trait for capturing system/desktop
//! audio and encoding it into Opus frames. The factory function
//! [`create_audio_capture`] selects the appropriate backend for the current
//! platform.
//!
//! ## Platform Backends
//!
//! | Platform | Backend                |
//! |----------|------------------------|
//! | Linux    | PipeWire / PulseAudio  |
//! | Windows  | WASAPI loopback        |
//! | macOS    | ScreenCaptureKit audio |

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "windows")]
pub mod windows;
#[cfg(target_os = "macos")]
pub mod macos;

use crate::error::MediaError;
use crate::types::*;

/// Trait for system audio capture backends.
///
/// Implementations capture desktop/system audio, encode it into Opus frames,
/// and return them via [`next_frame`](Self::next_frame).
pub trait AudioCapture: Send {
    /// Start capturing audio with the given configuration.
    fn start(&mut self, config: &AudioCaptureConfig) -> Result<(), MediaError>;

    /// Wait for and return the next encoded audio frame.
    fn next_frame(&mut self) -> Result<EncodedAudioFrame, MediaError>;

    /// Stop the audio capture session and release resources.
    fn stop(&mut self) -> Result<(), MediaError>;

    /// Returns `true` if audio capture is currently active.
    fn is_capturing(&self) -> bool;
}

/// Create an [`AudioCapture`] backend appropriate for the current platform.
pub fn create_audio_capture() -> Result<Box<dyn AudioCapture>, MediaError> {
    #[cfg(target_os = "linux")]
    return Ok(Box::new(linux::LinuxAudioCapture::new()?));

    #[cfg(target_os = "windows")]
    return Ok(Box::new(windows::WindowsAudioCapture::new()?));

    #[cfg(target_os = "macos")]
    return Ok(Box::new(macos::MacOsAudioCapture::new()?));

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    Err(MediaError::PlatformNotSupported("Unsupported OS".into()))
}
