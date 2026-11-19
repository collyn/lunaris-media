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
//! | Linux    | Capture service / DRM/KMS (zero-copy) | PipeWire (Wayland) | X11 |
//! | Windows  | DXGI Desktop Dup     | —                   | —          |
//! | macOS    | ScreenCaptureKit     | —                   | —          |

pub mod gpu_buffer;

#[cfg(target_os = "linux")]
pub mod dmabuf_caps;
/// Unattended capture through the root capture service (no portal prompt).
#[cfg(target_os = "linux")]
pub mod drm_service;
#[cfg(target_os = "linux")]
pub mod linux_drm;
#[cfg(target_os = "linux")]
pub mod linux_nvfbc;
/// NvFBC bindings, loaded with dlopen so the agent runs without the NVIDIA driver.
#[cfg(target_os = "linux")]
mod nvfbc;
#[cfg(target_os = "linux")]
pub mod linux_wayland;
#[cfg(target_os = "linux")]
pub mod linux_x11;
#[cfg(target_os = "linux")]
pub mod egl_import;
#[cfg(target_os = "macos")]
pub mod macos;
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

    /// Whether frames are delivered as DMA-BUFs that the VAAPI encoder on the
    /// display GPU can consume zero-copy. The pipeline then prefers VAAPI over
    /// encoders that would need a GPU→CPU readback.
    fn delivers_dmabuf(&self) -> bool {
        false
    }

    /// `true` when [`next_frame`](Self::next_frame) itself waits for the next
    /// compositor frame (or a padding deadline of one frame interval). The
    /// pipeline then encodes each frame as soon as it arrives instead of
    /// sampling on its own fixed-rate ticker, which would drop/duplicate
    /// frames whenever the compositor cadence (e.g. a 75 Hz panel) does not
    /// line up with the stream FPS.
    fn paces_frames(&self) -> bool {
        false
    }

    /// Called once the encoder is known: `true` when it consumes DMA-BUFs
    /// zero-copy (VAAPI on the display GPU). Backends that can read frames
    /// back themselves switch to CPU frames when `false`, which is cheaper
    /// than letting the encoder read back every (re-)encoded frame.
    fn set_dmabuf_passthrough(&mut self, _enabled: bool) {}

    /// Optional: Get the Direct3D11 device and context pointers (cast to usize) for Windows.
    fn get_d3d11_device(&self) -> Option<(usize, usize)> {
        None
    }
}

/// `true` when this process may export KMS framebuffers itself
/// (`CAP_SYS_ADMIN` effective or permitted, e.g. `setcap cap_sys_admin+p`).
#[cfg(target_os = "linux")]
fn has_cap_sys_admin() -> bool {
    const CAP_SYS_ADMIN: u32 = 21;
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else { return false };
    status
        .lines()
        .filter(|l| l.starts_with("CapEff:") || l.starts_with("CapPrm:"))
        .filter_map(|l| u64::from_str_radix(l.split_whitespace().nth(1)?, 16).ok())
        .any(|caps| caps & (1 << CAP_SYS_ADMIN) != 0)
}

/// How screen capture on a Wayland session may be done (a user setting of
/// the agent, "Wayland screen capture"). Other sessions ignore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaylandCapturePreference {
    /// The capture service, then direct KMS (with `CAP_SYS_ADMIN`), then the
    /// portal, then X11: no prompt when the service is installed.
    #[default]
    Auto,
    /// Always through the XDG portal (PipeWire), then X11: the person at the
    /// computer approves and the desktop shows that the screen is shared.
    /// The capture service and KMS are never used.
    Portal,
}

impl WaylandCapturePreference {
    /// `"auto"` or `"portal"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Portal => "portal",
        }
    }

    /// `"auto"` / `"portal"` (any case, surrounding spaces ignored); `None`
    /// for anything else.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "portal" => Some(Self::Portal),
            _ => None,
        }
    }
}

/// 0 = Auto, 1 = Portal.
static WAYLAND_CAPTURE_PREFERENCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Sets the [`WaylandCapturePreference`] for this process; capture backends
/// created afterwards follow it. `LUNARIS_CAPTURE` (a developer override)
/// still wins over it.
pub fn set_wayland_capture_preference(preference: WaylandCapturePreference) {
    let value = match preference {
        WaylandCapturePreference::Auto => 0,
        WaylandCapturePreference::Portal => 1,
    };
    let old = WAYLAND_CAPTURE_PREFERENCE.swap(value, std::sync::atomic::Ordering::Relaxed);
    if old != value {
        log::info!("Wayland screen capture preference: {}", preference.as_str());
    }
}

/// The preference last set with [`set_wayland_capture_preference`]
/// (`Auto` when never set).
pub fn wayland_capture_preference() -> WaylandCapturePreference {
    match WAYLAND_CAPTURE_PREFERENCE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => WaylandCapturePreference::Portal,
        _ => WaylandCapturePreference::Auto,
    }
}

/// A Wayland capture backend.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaylandBackend {
    Service,
    Kms,
    Portal,
}

/// The Wayland backends to try, in order, worked out from `LUNARIS_CAPTURE`
/// and the preference.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[derive(Debug, PartialEq, Eq)]
struct WaylandPlan {
    order: Vec<WaylandBackend>,
    /// The backend `LUNARIS_CAPTURE` names: tried first, and even when it
    /// looks unavailable (no socket, no `CAP_SYS_ADMIN`).
    forced: Option<WaylandBackend>,
    /// `LUNARIS_CAPTURE` was set to something it does not know (ignored).
    unknown_env: Option<String>,
}

/// `env`: the value of `LUNARIS_CAPTURE` (empty when unset).
///
/// A known `LUNARIS_CAPTURE` (`service`, `kms`/`drm`, `portal`/`pipewire`)
/// wins over the preference, as it always has: that backend first, then the
/// others in the usual order. Otherwise `Auto` is service, KMS, portal and
/// `Portal` is the portal alone. X11 comes after any of these.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn wayland_plan(env: &str, preference: WaylandCapturePreference) -> WaylandPlan {
    let env = env.trim().to_ascii_lowercase();
    let forced = match env.as_str() {
        "service" => Some(WaylandBackend::Service),
        "kms" | "drm" => Some(WaylandBackend::Kms),
        "portal" | "pipewire" => Some(WaylandBackend::Portal),
        _ => None,
    };
    let unknown_env = (forced.is_none() && !env.is_empty()).then_some(env);
    let mut order = match (forced, preference) {
        (None, WaylandCapturePreference::Portal) => vec![WaylandBackend::Portal],
        _ => vec![WaylandBackend::Service, WaylandBackend::Kms, WaylandBackend::Portal],
    };
    if let Some(f) = forced {
        order.retain(|b| *b != f);
        order.insert(0, f);
    }
    WaylandPlan { order, forced, unknown_env }
}

/// Wayland backends in order; `None` falls through to X11.
///
/// 1. The Lunaris capture service (root helper owning DRM): no prompt.
/// 2. Direct KMS when this process itself has `CAP_SYS_ADMIN`.
/// 3. PipeWire through the XDG Desktop Portal (asks the person once; the
///    restore token remembers the answer).
///
/// With [`WaylandCapturePreference::Portal`] only the portal is tried.
/// `LUNARIS_CAPTURE=service|kms|portal` tries that one first (falling back
/// to the others with a warning if it is unavailable), whatever the
/// preference (see [`wayland_plan`]).
#[cfg(target_os = "linux")]
fn wayland_capture() -> Option<Box<dyn ScreenCapture>> {
    let env = std::env::var("LUNARIS_CAPTURE").unwrap_or_default();
    let preference = wayland_capture_preference();
    let plan = wayland_plan(&env, preference);
    if let Some(unknown) = &plan.unknown_env {
        log::warn!("LUNARIS_CAPTURE={} is not one of service, kms, portal; ignoring it", unknown);
    }
    match plan.forced {
        Some(f) => log::info!(
            "Wayland capture: LUNARIS_CAPTURE={:?} overrides the preference \"{}\"",
            f,
            preference.as_str()
        ),
        None => log::info!("Wayland capture preference \"{}\": trying {:?}", preference.as_str(), plan.order),
    }
    for backend in plan.order {
        let is_forced = Some(backend) == plan.forced;
        match backend {
            WaylandBackend::Service => {
                let path = drm_service::socket_path();
                if !is_forced && !path.exists() {
                    log::info!("Capture service not installed ({} missing)", path.display());
                    continue;
                }
                match drm_service::DrmServiceCapture::connect(&path) {
                    Ok(c) => {
                        log::info!(
                            "Using the Lunaris capture service on {} (Wayland, no portal prompt, {} display(s))",
                            path.display(),
                            c.wire_displays().len()
                        );
                        return Some(Box::new(c));
                    }
                    Err(e) if is_forced => log::warn!("LUNARIS_CAPTURE=service but {}; trying others", e),
                    Err(e) => log::info!("Capture service not usable: {}", e),
                }
            }
            WaylandBackend::Kms => {
                if !is_forced && !has_cap_sys_admin() {
                    continue;
                }
                match linux_drm::DrmCapture::new() {
                    Ok(drm) => {
                        log::info!("Using direct DRM/KMS capture on Wayland (this process has CAP_SYS_ADMIN)");
                        return Some(Box::new(drm));
                    }
                    Err(e) => log::warn!("KMS capture unavailable: {}", e),
                }
            }
            WaylandBackend::Portal => {
                log::info!("Using the XDG portal (PipeWire) on Wayland: the first session asks on this screen");
                match linux_wayland::PipeWireCapture::new() {
                    Ok(pw) => return Some(Box::new(pw)),
                    Err(e) => log::warn!("PipeWire not available on Wayland: {}, trying X11 fallback", e),
                }
            }
        }
    }
    None
}

/// Create a [`ScreenCapture`] backend appropriate for the current platform.
///
/// On Linux/Wayland this tries the capture service, direct KMS (with
/// `CAP_SYS_ADMIN`) and the portal in that order (see `wayland_capture`),
/// falling back to X11. On other platforms the native API is used
/// directly.
pub fn create_screen_capture() -> Result<Box<dyn ScreenCapture>, MediaError> {
    #[cfg(target_os = "linux")]
    {
        let is_wayland = std::env::var("XDG_SESSION_TYPE")
            .unwrap_or_default()
            == "wayland";

        if is_wayland {
            if let Some(capture) = wayland_capture() {
                return Ok(capture);
            }
        } else {
            // X11 session: NvFBC > DRM > PipeWire > X11
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

/// Enumerate displays for capability queries as cheaply as possible.
///
/// Unlike [`create_screen_capture`] followed by [`ScreenCapture::list_displays`],
/// this avoids spinning up a full GPU capture backend — and, on NVIDIA, a
/// ~28 MiB CUDA context plus a slow `cuInit` — just to read the display list.
/// That matters because capability queries fire every time a client opens its
/// settings; the heavyweight path made each query allocate GPU memory and (before
/// the drop fix) leak it.
///
/// On Linux it first tries an NvFBC status-only query that skips CUDA entirely;
/// if NvFBC is unavailable it falls back to the regular capture backend. On other
/// platforms it uses the regular backend, whose enumeration is already light.
/// The returned display ids are identical to the capture path, so display
/// selection is unaffected.
pub async fn list_displays_for_capabilities() -> Result<Vec<DisplayInfo>, MediaError> {
    #[cfg(target_os = "linux")]
    {
        let is_wayland = std::env::var("XDG_SESSION_TYPE")
            .unwrap_or_default()
            == "wayland";

        // NvFBC lightweight enumeration is X11-only — skip on Wayland
        if !is_wayland {
            match tokio::task::spawn_blocking(linux_nvfbc::list_displays_lightweight).await {
                Ok(Ok(displays)) if !displays.is_empty() => return Ok(displays),
                Ok(Ok(_)) => {
                    log::info!(
                        "NvFBC lightweight enumeration returned no outputs; using full backend"
                    );
                }
                Ok(Err(e)) => {
                    log::info!(
                        "NvFBC lightweight enumeration unavailable ({}); using full backend",
                        e
                    );
                }
                Err(e) => log::warn!("NvFBC lightweight enumeration task join failed: {}", e),
            }
        }
    }

    // Fallback: full capture backend. Its teardown is leak-free after the drop
    // fix, but on Linux `NvfbcCapture::drop()` still joins a background thread, so
    // drop it off the async runtime rather than blocking a runtime worker.
    let cap = create_screen_capture()?;
    let displays = cap.list_displays().await;
    tokio::task::spawn_blocking(move || drop(cap));
    displays
}

#[cfg(test)]
mod tests {
    use super::WaylandBackend::{Kms, Portal, Service};
    use super::WaylandCapturePreference as Pref;
    use super::*;

    #[test]
    fn auto_without_override_is_service_kms_portal() {
        let plan = wayland_plan("", Pref::Auto);
        assert_eq!(plan, WaylandPlan { order: vec![Service, Kms, Portal], forced: None, unknown_env: None });
    }

    #[test]
    fn portal_preference_never_tries_service_or_kms() {
        let plan = wayland_plan("", Pref::Portal);
        assert_eq!(plan.order, vec![Portal]);
        assert_eq!(plan.forced, None);
        // An unknown override is ignored and does not widen the order.
        let plan = wayland_plan("nvfbc", Pref::Portal);
        assert_eq!(plan.order, vec![Portal]);
        assert_eq!(plan.unknown_env.as_deref(), Some("nvfbc"));
    }

    #[test]
    fn env_override_wins_over_the_preference() {
        for pref in [Pref::Auto, Pref::Portal] {
            let plan = wayland_plan("service", pref);
            assert_eq!(plan.order, vec![Service, Kms, Portal]);
            assert_eq!(plan.forced, Some(Service));
            let plan = wayland_plan("KMS", pref);
            assert_eq!(plan.order, vec![Kms, Service, Portal]);
            assert_eq!(plan.forced, Some(Kms));
            let plan = wayland_plan("drm", pref);
            assert_eq!((plan.order[0], plan.forced), (Kms, Some(Kms)));
            let plan = wayland_plan(" portal ", pref);
            assert_eq!(plan.order, vec![Portal, Service, Kms]);
            let plan = wayland_plan("pipewire", pref);
            assert_eq!(plan.forced, Some(Portal));
        }
    }

    #[test]
    fn unknown_override_is_reported_and_ignored() {
        let plan = wayland_plan("Bogus", Pref::Auto);
        assert_eq!(plan.order, vec![Service, Kms, Portal]);
        assert_eq!(plan.forced, None);
        assert_eq!(plan.unknown_env.as_deref(), Some("bogus"));
    }

    #[test]
    fn preference_parses_and_is_stored() {
        assert_eq!(Pref::parse("auto"), Some(Pref::Auto));
        assert_eq!(Pref::parse(" Portal "), Some(Pref::Portal));
        assert_eq!(Pref::parse(""), None);
        assert_eq!(Pref::parse("service"), None);
        assert_eq!(Pref::default(), Pref::Auto);
        for p in [Pref::Auto, Pref::Portal] {
            assert_eq!(Pref::parse(p.as_str()), Some(p));
        }
        // The only test that touches the process-wide value.
        set_wayland_capture_preference(Pref::Portal);
        assert_eq!(wayland_capture_preference(), Pref::Portal);
        set_wayland_capture_preference(Pref::Auto);
        assert_eq!(wayland_capture_preference(), Pref::Auto);
    }
}
