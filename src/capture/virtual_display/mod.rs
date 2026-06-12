//! Cross-platform virtual display management.
//!
//! This module provides a [`DisplayManager`] trait for creating and destroying
//! virtual displays across platforms. On Linux it uses XRandR virtual outputs;
//! on Windows it uses an IddCx-based virtual display driver.

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "windows")]
pub mod windows;

use crate::error::MediaError;

/// Trait for platform-specific virtual display lifecycle management.
///
/// A virtual display appears as a real monitor to the OS, allowing screen
/// capture backends to target it. The display is automatically destroyed
/// when the manager is dropped.
pub trait DisplayManager {
    /// Create a virtual display with the given resolution and refresh rate.
    ///
    /// Returns a handle that, when dropped, tears down the display.
    fn create(width: u32, height: u32, fps: u32) -> Result<Self, MediaError>
    where
        Self: Sized;

    /// Return the platform-specific display identifier.
    ///
    /// On Linux this is the XRandR output name (e.g. `VIRTUAL1`).
    /// On Windows this is the display device name (e.g. `\\.\DISPLAY2`).
    fn display_id(&self) -> &str;

    /// Explicitly destroy the virtual display before the handle is dropped.
    fn destroy(&mut self) -> Result<(), MediaError>;
}

/// Create a platform-appropriate virtual display manager.
pub fn create_virtual_display(
    width: u32,
    height: u32,
    fps: u32,
) -> Result<Box<dyn VirtualDisplayHandle>, MediaError> {
    #[cfg(target_os = "linux")]
    {
        let vd = linux::VirtualDisplay::create(width, height, fps)?;
        return Ok(Box::new(vd));
    }

    #[cfg(target_os = "windows")]
    {
        let vd = windows::WindowsVirtualDisplay::create(width, height, fps)?;
        return Ok(Box::new(vd));
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    Err(MediaError::PlatformNotSupported(
        "Virtual display not supported on this platform".into(),
    ))
}

/// Object-safe wrapper so the pipeline can hold a boxed virtual display
/// without knowing the concrete platform type.
pub trait VirtualDisplayHandle: Send {
    fn display_id(&self) -> &str;
    fn destroy(&mut self) -> Result<(), MediaError>;
}

#[cfg(target_os = "linux")]
impl VirtualDisplayHandle for linux::VirtualDisplay {
    fn display_id(&self) -> &str {
        self.output_name()
    }

    fn destroy(&mut self) -> Result<(), MediaError> {
        linux::VirtualDisplay::destroy(self)
    }
}

#[cfg(target_os = "windows")]
impl VirtualDisplayHandle for windows::WindowsVirtualDisplay {
    fn display_id(&self) -> &str {
        windows::WindowsVirtualDisplay::display_id(self)
    }

    fn destroy(&mut self) -> Result<(), MediaError> {
        windows::WindowsVirtualDisplay::destroy(self)
    }
}
