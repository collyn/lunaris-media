//! Screen capture subsystem.
//!
//! This module defines the [`ScreenCapture`] trait — the primary abstraction for
//! capturing screen content into GPU-resident buffers — and provides a factory
//! function [`create_screen_capture`] that selects the best backend for the
//! current platform.
//!
//! ## Platform Backends
//!
//! | Platform | Primary              | Fallback 1          | Fallback 2 |
//! |----------|----------------------|---------------------|------------|
//! | Linux    | DRM/KMS (zero-copy)  | PipeWire (Wayland)  | X11        |
//! | Windows  | DXGI Desktop Dup     | —                   | —          |
//! | macOS    | ScreenCaptureKit     | —                   | —          |

pub mod gpu_buffer;

#[cfg(target_os = "linux")]
pub mod linux_drm;
#[cfg(target_os = "linux")]
pub mod linux_nvfbc;
#[cfg(target_os = "linux")]
pub mod linux_wayland;
#[cfg(target_os = "linux")]
pub mod linux_x11;
#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub mod virtual_display;
#[cfg(target_os = "windows")]
pub mod windows;

use crate::error::MediaError;
use crate::types::*;
use gpu_buffer::GpuBuffer;

#[cfg(target_os = "linux")]
pub(crate) fn should_embed_host_cursor() -> bool {
    if std::env::var("LUNARIS_HIDE_HOST_CURSOR").is_ok() {
        return false;
    }

    std::env::var("LUNARIS_EMBED_HOST_CURSOR")
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// A frame captured from the screen, containing a GPU-resident buffer.
///
/// The [`GpuBuffer`] inside this struct can be handed directly to a hardware
/// encoder, avoiding any GPU→CPU→GPU copy.
#[derive(Debug)]
pub struct CapturedFrame {
    /// GPU-resident (or CPU-fallback) buffer holding the raw frame pixels.
    pub buffer: GpuBuffer,
    /// Capture timestamp in microseconds (monotonic clock).
    pub timestamp_us: u64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format of the captured data.
    pub format: PixelFormat,
    /// Whether this frame is a new frame (e.g. from change detection).
    pub is_new_frame: bool,
    /// Optional cursor metadata embedded by the capture backend (e.g. PipeWire).
    /// When present, the pipeline can skip separate cursor polling.
    pub cursor: Option<FrameCursorMeta>,
}

/// Cursor metadata embedded in a captured frame (e.g. from PipeWire spa_meta_cursor).
#[derive(Debug, Clone)]
pub struct FrameCursorMeta {
    /// Cursor X position in screen coordinates.
    pub x: i32,
    /// Cursor Y position in screen coordinates.
    pub y: i32,
    /// Hotspot X offset within the cursor bitmap.
    pub hotspot_x: i32,
    /// Hotspot Y offset within the cursor bitmap.
    pub hotspot_y: i32,
    /// Whether the cursor is visible (spa_meta_cursor.id != 0).
    pub visible: bool,
    /// Optional cursor bitmap. Present when the cursor shape changes.
    pub image: Option<crate::types::CursorImage>,
}

/// Trait for screen capture backends.
///
/// Implementations are expected to be used from a single async task. The trait
/// is `Send` so that the owning task can be moved between executor threads.
#[async_trait::async_trait]
pub trait ScreenCapture: Send {
    /// Enumerate all available displays/monitors.
    async fn list_displays(&self) -> Result<Vec<DisplayInfo>, MediaError>;

    /// Start capturing the given display at the requested configuration.
    ///
    /// Returns [`MediaError::CaptureAlreadyStarted`] if capture is already running.
    async fn start(&mut self, display_id: &str, config: &StreamConfig) -> Result<(), MediaError>;

    /// Wait for and return the next captured frame.
    ///
    /// This method blocks (asynchronously) until a new frame is available or an
    /// error occurs. Returns [`MediaError::CaptureNotStarted`] if capture has
    /// not been started.
    async fn next_frame(&mut self) -> Result<CapturedFrame, MediaError>;

    /// Stop the running capture session and release resources.
    async fn stop(&mut self) -> Result<(), MediaError>;

    /// Update the target capture frame rate while capture is running.
    ///
    /// Some backends bake their capture pacing into the session at creation
    /// time (e.g. NvFBC's `dwSamplingRateMs = 1000 / fps`). For those, simply
    /// changing the pipeline's frame ticker is not enough — the backend keeps
    /// producing frames at the original rate and the stream stays capped. Such
    /// backends override this to reconfigure themselves.
    ///
    /// The default implementation is a no-op, appropriate for backends that are
    /// paced purely by the pipeline's frame ticker (X11, DRM, DXGI, …).
    async fn set_fps(&mut self, _fps: u32) -> Result<(), MediaError> {
        Ok(())
    }

    /// Returns `true` if capture is currently active.
    fn is_capturing(&self) -> bool;

    /// Optional: Get the Direct3D11 device and context pointers (cast to usize) for Windows.
    fn get_d3d11_device(&self) -> Option<(usize, usize)> {
        None
    }
}

/// Create a [`ScreenCapture`] backend appropriate for the current platform.
///
/// On Linux this tries PipeWire first (for Wayland compositors), falling back
/// to X11 if PipeWire is unavailable. On other platforms the native API is used
/// directly.
pub fn create_screen_capture() -> Result<Box<dyn ScreenCapture>, MediaError> {
    #[cfg(target_os = "linux")]
    {
        // Priority: NvFBC (NVIDIA GPU direct) > DRM/KMS (zero-copy GPU) > PipeWire (Wayland) > X11
        match linux_nvfbc::NvfbcCapture::new() {
            Ok(nvfbc) => {
                log::info!("Using NVIDIA NvFBC screen capture (GPU accelerated)");
                return Ok(Box::new(nvfbc));
            }
            Err(e) => {
                log::warn!("NvFBC capture not available: {}, trying DRM/KMS", e);
            }
        }
        match linux_drm::DrmCapture::new() {
            Ok(drm) => {
                log::info!("Using DRM/KMS screen capture (zero-copy GPU)");
                return Ok(Box::new(drm));
            }
            Err(e) => {
                log::warn!("DRM capture not available: {}, trying PipeWire", e);
            }
        }
        match linux_wayland::PipeWireCapture::new() {
            Ok(pw) => {
                log::info!("Using PipeWire screen capture (Wayland)");
                return Ok(Box::new(pw));
            }
            Err(e) => {
                log::warn!("PipeWire not available: {}, falling back to X11", e);
            }
        }
        let x11 = linux_x11::X11Capture::new()?;
        log::info!("Using X11 screen capture");
        return Ok(Box::new(x11));
    }

    #[cfg(target_os = "windows")]
    {
        return Ok(Box::new(windows::WindowsScreenCapture::new()?));
    }

    #[cfg(target_os = "macos")]
    {
        return Ok(Box::new(macos::ScreenCaptureKitCapture::new()?));
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    Err(MediaError::PlatformNotSupported("Unsupported OS".into()))
}
