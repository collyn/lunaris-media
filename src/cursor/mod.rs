//! Cursor capture subsystem.
//!
//! This module defines the [`CursorCapture`] trait for tracking mouse cursor
//! position and shape. The factory function [`create_cursor_capture`] selects
//! the appropriate backend for the current platform.

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "windows")]
pub mod windows;
#[cfg(target_os = "macos")]
pub mod macos;

use crate::error::MediaError;
use crate::types::*;

/// Trait for cursor state tracking backends.
///
/// Implementations poll or listen for cursor position and shape changes,
/// returning the current state via [`get_cursor_state`](Self::get_cursor_state).
pub trait CursorCapture: Send {
    /// Start tracking cursor state.
    fn start(&mut self) -> Result<(), MediaError>;

    /// Get the current cursor position and image.
    fn get_cursor_state(&mut self) -> Result<CursorState, MediaError>;

    /// Stop tracking cursor state and release resources.
    fn stop(&mut self) -> Result<(), MediaError>;
}

/// Create a [`CursorCapture`] backend appropriate for the current platform.
pub fn create_cursor_capture() -> Result<Box<dyn CursorCapture>, MediaError> {
    #[cfg(target_os = "linux")]
    return Ok(Box::new(linux::LinuxCursorCapture::new()?));

    #[cfg(target_os = "windows")]
    return Ok(Box::new(windows::WindowsCursorCapture::new()?));

    #[cfg(target_os = "macos")]
    return Ok(Box::new(macos::MacOsCursorCapture::new()?));

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    Err(MediaError::PlatformNotSupported("Unsupported OS".into()))
}
