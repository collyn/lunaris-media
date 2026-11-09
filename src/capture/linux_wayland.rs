//! PipeWire-based screen capture for Wayland compositors (Linux).
//!
//! Uses the XDG Desktop Portal (via `ashpd`) to request screen capture permission
//! and PipeWire to receive frames. Format negotiation follows Sunshine: every
//! format is offered with the DRM modifiers the display GPU can import (queried
//! through EGL/GBM, see [`super::dmabuf_caps`]) so the compositor shares its
//! GPU buffers directly. Those DMA-BUFs are forwarded untouched to the encoder
//! (`GpuBuffer::DmaBuf`), which converts/scales them on the GPU (VAAPI VPP) —
//! no pixel ever crosses the CPU. Shared-memory buffers remain as fallback, and
//! `LUNARIS_PIPEWIRE_ZERO_COPY=0` restores the EGL `glReadPixels` readback.

use std::cell::RefCell;
use std::rc::Rc;
use tokio::sync::mpsc;

use crate::capture::gpu_buffer::GpuBuffer;
use crate::capture::{CapturedFrame, FrameCursorMeta, ScreenCapture};
use crate::error::MediaError;
use crate::types::*;

/// Latest cursor reported by the compositor through `SPA_META_Cursor`.
///
/// Wayland has no global pointer query (XFixes only sees XWayland windows), so
/// the cursor backend reads this instead. Coordinates are relative to the
/// streamed monitor, in encoded-video pixels.
#[derive(Clone, Debug, Default)]
pub struct PipewireCursor {
    pub x: i32,
    pub y: i32,
    pub visible: bool,
    /// Most recent cursor bitmap.
    pub image: Option<CursorImage>,
    /// Incremented whenever `image` changes.
    pub image_serial: u64,
}

static PIPEWIRE_CURSOR: std::sync::Mutex<Option<PipewireCursor>> = std::sync::Mutex::new(None);

/// Snapshot of the PipeWire cursor, if a PipeWire capture is running.
pub fn pipewire_cursor() -> Option<PipewireCursor> {
    PIPEWIRE_CURSOR.lock().ok().and_then(|c| c.clone())
}

fn publish_cursor(meta: &FrameCursorMeta) {
    if let Ok(mut slot) = PIPEWIRE_CURSOR.lock() {
        let cur = slot.get_or_insert_with(PipewireCursor::default);
        cur.x = meta.x;
        cur.y = meta.y;
        cur.visible = meta.visible;
        if let Some(image) = &meta.image {
            cur.image = Some(image.clone());
            cur.image_serial = cur.image_serial.wrapping_add(1);
        }
    }
}

/// Rescale a coordinate from `from` pixels to `to` pixels (identity when
/// either size is unknown).
fn scale_coord(v: i32, from: u32, to: u32) -> i32 {
    if from == 0 || to == 0 || from == to {
        return v;
    }
    ((v as i64 * to as i64) / from as i64) as i32
}

fn clear_cursor() {
    if let Ok(mut slot) = PIPEWIRE_CURSOR.lock() {
        *slot = None;
    }
}

/// Capacity of the internal frame channel.
const FRAME_CHANNEL_CAPACITY: usize = 2;

/// PipeWire-based screen capture backend for Wayland sessions.
pub struct PipeWireCapture {
    frame_rx: Option<mpsc::Receiver<CapturedFrame>>,
    pw_thread: Option<std::thread::JoinHandle<()>>,
    shutdown_tx: Option<pipewire::channel::Sender<()>>,
    capturing: bool,
    /// Target frame rate; bounds how long `next_frame` waits for damage.
    fps: u32,
    /// Forward DMA-BUFs untouched (zero-copy) instead of reading them back.
    passthrough: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Fixed emission schedule: when `next_frame` may return next (advances
    /// by exactly one interval per frame, so timer rounding never drifts).
    next_due: Option<tokio::time::Instant>,
}

impl PipeWireCapture {
    /// Creates a new PipeWire capture instance.
    ///
    /// Fails if the current session is not Wayland.
    pub fn new() -> Result<Self, MediaError> {
        let session_type = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();
        let wayland_display = std::env::var("WAYLAND_DISPLAY").ok();

        if session_type != "wayland" && wayland_display.is_none() {
            return Err(MediaError::PlatformNotSupported(
                "PipeWire screen capture requires a Wayland session.".into(),
            ));
        }

        pipewire::init();
        log::info!("PipeWire capture backend initialized");

        Ok(Self {
            frame_rx: None,
            pw_thread: None,
            shutdown_tx: None,
            capturing: false,
            fps: 60,
            passthrough: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                pipewire_zero_copy_enabled(),
            )),
            next_due: None,
        })
    }
}

#[async_trait::async_trait]
impl ScreenCapture for PipeWireCapture {
    async fn list_displays(&self) -> Result<Vec<DisplayInfo>, MediaError> {
        log::info!("PipeWire: returning default display (portal will show picker)");
        Ok(vec![DisplayInfo {
            id: "default".to_string(),
            name: "Default Display (portal will show picker)".to_string(),
            width: 1920,
            height: 1080,
            refresh_rate: 60.0,
            is_primary: true,
        }])
    }

    async fn start(&mut self, display_id: &str, config: &StreamConfig) -> Result<(), MediaError> {
        if self.capturing {
            return Err(MediaError::CaptureAlreadyStarted);
        }

        log::info!(
            "PipeWireCapture: starting capture on '{}' at {}x{} {}fps",
            display_id,
            config.width,
            config.height,
            config.fps
        );

        // All portal (D-Bus) traffic runs on one long-lived runtime. ashpd
        // caches a process-wide zbus connection whose helper tasks are spawned
        // on whatever runtime first touches it; using short-lived runtimes
        // (or aborted caller tasks) could leave that connection wedged and
        // make every later session hang at "Connecting to Screencast portal".
        let opened = portal_runtime()
            .spawn(open_portal_session())
            .await
            .map_err(|e| MediaError::CaptureError(format!("Portal task failed: {}", e)))??;
        let PortalSession { session, node_id, position, size } = opened;

        log::info!(
            "Screencast portal started. Node ID: {} position={:?} size={:?}",
            node_id,
            position,
            size
        );
        // Map absolute pointer input onto the monitor the user picked.
        if let (Some((x, y)), Some((w, h))) = (position, size) {
            crate::input::set_capture_region(crate::input::ScreenRegion {
                x,
                y,
                width: w,
                height: h,
            });
        }

        let (frame_tx, frame_rx) = mpsc::channel(FRAME_CHANNEL_CAPACITY);
        let (pw_sender, pw_receiver) = pipewire::channel::channel::<()>();

        let config = config.clone();
        self.fps = config.fps.max(1);
        let passthrough = self.passthrough.clone();

        let pw_thread = std::thread::Builder::new()
            .name("lunaris-pw".into())
            .spawn(move || {
                if let Err(e) = run_pipewire_loop(node_id, frame_tx, pw_receiver, config, passthrough) {
                    log::error!("PipeWire loop error: {:?}", e);
                }
                // Close the portal session after PipeWire stops (removes the
                // compositor's screencast indicator). Runs on the portal
                // runtime; this thread has no runtime of its own.
                let closed = portal_runtime().block_on(async move {
                    tokio::time::timeout(PORTAL_CALL_TIMEOUT, session.close()).await
                });
                match closed {
                    Ok(Ok(())) => log::info!("Screencast portal session closed"),
                    Ok(Err(e)) => log::warn!("Failed to close screencast portal session: {}", e),
                    Err(_) => log::warn!("Timed out closing screencast portal session"),
                }
            })
            .map_err(|e| {
                MediaError::CaptureError(format!("Failed to spawn PipeWire thread: {}", e))
            })?;

        self.pw_thread = Some(pw_thread);
        self.shutdown_tx = Some(pw_sender);
        self.frame_rx = Some(frame_rx);
        self.capturing = true;
        self.next_due = None;

        log::info!("PipeWire screen capture started successfully");
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<CapturedFrame, MediaError> {
        if !self.capturing {
            return Err(MediaError::CaptureNotStarted);
        }

        let rx = self
            .frame_rx
            .as_mut()
            .ok_or(MediaError::CaptureNotStarted)?;

        // Frame-driven pacing (see `paces_frames`), capped at the stream FPS:
        // - never emit earlier than one interval after the previous frame
        //   (KWin ignores maxFramerate and pushes at the panel rate, e.g. 75 Hz);
        // - at that point take the newest frame; if none is queued yet, wait a
        //   short grace (¼ interval) so a frame that is just about to arrive
        //   is not missed (this also phase-locks to the compositor);
        // - otherwise return an empty buffer so the pipeline re-encodes the
        //   last picture (constant FPS while the screen is idle).
        // Sampling on a separate fixed ticker instead lost frames (~45-54 FPS
        // on a 60 FPS stream).
        let interval = std::time::Duration::from_nanos(1_000_000_000 / self.fps.max(1) as u64);
        let grace = interval / 4;
        let first = self.next_due.is_none();
        let due = self.next_due.unwrap_or_else(tokio::time::Instant::now);
        tokio::time::sleep_until(due).await;
        let received = match rx.try_recv() {
            Ok(frame) => Ok(Some(frame)),
            Err(mpsc::error::TryRecvError::Disconnected) => Ok(None),
            Err(mpsc::error::TryRecvError::Empty) => {
                // First call: allow a full interval for the first frame.
                tokio::time::timeout(if first { interval } else { grace }, rx.recv()).await
            }
        };
        // Advance the schedule by exactly one interval; if we fell more than
        // an interval behind (slow encode, system hiccup), resynchronise
        // instead of bursting to catch up.
        let now = tokio::time::Instant::now();
        let next = due + interval;
        self.next_due = Some(if next + interval < now { now } else { next });
        let mut frame = match received {
            Ok(Some(frame)) => frame,
            Ok(None) => return Err(MediaError::CaptureError("Frame channel closed".into())),
            Err(_) => {
                return Ok(CapturedFrame {
                    buffer: GpuBuffer::CpuBuffer {
                        data: Vec::new(),
                        stride: 0,
                        format: PixelFormat::BGRA,
                        width: 0,
                        height: 0,
                    },
                    timestamp_us: 0,
                    width: 0,
                    height: 0,
                    format: PixelFormat::BGRA,
                    is_new_frame: false,
                    cursor: None,
                })
            }
        };
        // Always encode the newest picture available.
        while let Ok(newer) = rx.try_recv() {
            frame = newer;
        }
        Ok(frame)
    }

    async fn set_fps(&mut self, fps: u32) -> Result<(), MediaError> {
        self.fps = fps.max(1);
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), MediaError> {
        if !self.capturing {
            return Ok(());
        }

        log::info!("Stopping PipeWire capture");
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        if let Some(handle) = self.pw_thread.take() {
            let _ = handle.join();
        }

        self.frame_rx = None;
        self.capturing = false;
        clear_cursor();

        log::info!("PipeWire capture stopped");
        Ok(())
    }

    fn is_capturing(&self) -> bool {
        self.capturing
    }

    fn delivers_dmabuf(&self) -> bool {
        self.passthrough.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn paces_frames(&self) -> bool {
        true
    }

    fn set_dmabuf_passthrough(&mut self, enabled: bool) {
        let enabled = enabled && pipewire_zero_copy_enabled();
        log::info!(
            "PipeWire: DMA-BUF passthrough {}",
            if enabled { "ON (zero-copy)" } else { "OFF (EGL readback in capture thread)" }
        );
        self.passthrough
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for PipeWireCapture {
    fn drop(&mut self) {
        log::info!("PipeWireCapture dropped");
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.pw_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Timeout for portal calls that never involve the user.
const PORTAL_CALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);
/// Timeout for `Start`, which may show the source picker to the user.
const PORTAL_START_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Dedicated, never-dropped runtime for all XDG portal traffic.
fn portal_runtime() -> &'static tokio::runtime::Runtime {
    static RT: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("lunaris-portal")
            .enable_all()
            .build()
            .expect("failed to build portal runtime")
    })
}

type ScreencastSession = ashpd::desktop::Session<'static, ashpd::desktop::screencast::Screencast<'static>>;

struct PortalSession {
    session: ScreencastSession,
    node_id: u32,
    position: Option<(i32, i32)>,
    size: Option<(i32, i32)>,
}

async fn portal_step<T, E: std::fmt::Display>(
    what: &str,
    timeout: std::time::Duration,
    fut: impl std::future::Future<Output = Result<T, E>>,
) -> Result<T, MediaError> {
    match tokio::time::timeout(timeout, fut).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(MediaError::CaptureError(format!("{}: {}", what, e))),
        Err(_) => Err(MediaError::CaptureError(format!(
            "{}: timed out after {}s (is xdg-desktop-portal running?)",
            what,
            timeout.as_secs()
        ))),
    }
}

/// Open a ScreenCast portal session for one monitor, reusing the saved
/// restore token so reconnects do not prompt the user.
async fn open_portal_session() -> Result<PortalSession, MediaError> {
    use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType};
    use ashpd::desktop::PersistMode;
    use ashpd::WindowIdentifier;
    use enumflags2::BitFlags;

    log::info!("Connecting to Screencast portal...");
    let screencast = portal_step("Failed to create Screencast portal", PORTAL_CALL_TIMEOUT, Screencast::new()).await?;
    let session = portal_step("Failed to create portal session", PORTAL_CALL_TIMEOUT, screencast.create_session()).await?;

    let config_dir = std::env::var("XDG_CONFIG_HOME")
        .unwrap_or_else(|_| format!("{}/.config", std::env::var("HOME").unwrap_or_default()));
    let restore_token_path = format!("{}/lunaris/screencast-token", config_dir);
    let saved_token = std::fs::read_to_string(&restore_token_path)
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty());

    log::info!("Requesting source selection...");
    let cursor_mode = if crate::capture::should_embed_host_cursor() {
        log::info!("PipeWireCapture: embedding host cursor in video stream");
        CursorMode::Embedded
    } else {
        // Cursor as metadata (SPA_META_Cursor): not drawn into the video,
        // forwarded to the client overlay instead. Fall back to embedding
        // when the portal cannot provide metadata.
        let available = tokio::time::timeout(PORTAL_CALL_TIMEOUT, screencast.available_cursor_modes())
            .await
            .ok()
            .and_then(|r| r.ok())
            .unwrap_or_default();
        if available.contains(CursorMode::Metadata) {
            log::info!("PipeWireCapture: cursor as metadata; browser overlay will render it");
            CursorMode::Metadata
        } else {
            log::info!("PipeWireCapture: portal lacks cursor metadata; embedding cursor in video");
            CursorMode::Embedded
        }
    };
    portal_step(
        "Failed to select sources",
        PORTAL_CALL_TIMEOUT,
        screencast.select_sources(
            &session,
            cursor_mode,
            BitFlags::from(SourceType::Monitor),
            false,
            saved_token.as_deref(),
            PersistMode::ExplicitlyRevoked,
        ),
    )
    .await?;

    log::info!("Starting screencast session (user prompt)...");
    let request = portal_step(
        "Failed to start portal",
        PORTAL_START_TIMEOUT,
        screencast.start(&session, &WindowIdentifier::default()),
    )
    .await?;
    let start_response = request
        .response()
        .map_err(|e| MediaError::CaptureError(format!("Portal response error: {:?}", e)))?;

    if let Some(token) = start_response.restore_token().filter(|t| !t.is_empty()) {
        if let Some(parent) = std::path::Path::new(&restore_token_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::write(&restore_token_path, token) {
            Ok(()) => log::info!("Screencast restore token saved for future sessions"),
            Err(e) => log::warn!("Failed to save screencast restore token: {}", e),
        }
    }

    let stream = start_response
        .streams()
        .first()
        .ok_or_else(|| MediaError::CaptureError("No stream returned from portal".into()))?;
    Ok(PortalSession {
        node_id: stream.pipe_wire_node_id(),
        position: stream.position(),
        size: stream.size(),
        session,
    })
}

/// Format fixed by the compositor in `param_changed`, read by `process`.
#[derive(Clone, Copy, Debug, Default)]
struct NegotiatedFormat {
    /// DRM fourcc equivalent of the negotiated SPA format.
    drm_format: u32,
    /// DRM modifier (only meaningful when `dmabuf` is true).
    modifier: u64,
    /// Actual frame size chosen by the compositor (may differ from the
    /// encoder size; the encoder scales on the GPU).
    width: u32,
    height: u32,
    /// The format carried a modifier, i.e. buffers will be DMA-BUFs.
    dmabuf: bool,
    ready: bool,
}

/// Whether DMA-BUF frames are forwarded untouched to the encoder (zero-copy).
///
/// Set `LUNARIS_PIPEWIRE_ZERO_COPY=0` to force the legacy EGL `glReadPixels`
/// readback inside the capture thread.
fn pipewire_zero_copy_enabled() -> bool {
    std::env::var("LUNARIS_PIPEWIRE_ZERO_COPY")
        .map(|v| !matches!(v.to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"))
        .unwrap_or(true)
}

/// Size of `SPA_META_Cursor` holding a `side`×`side` RGBA bitmap.
fn cursor_meta_size(side: i32) -> i32 {
    (std::mem::size_of::<pipewire::spa::sys::spa_meta_cursor>()
        + std::mem::size_of::<pipewire::spa::sys::spa_meta_bitmap>()) as i32
        + side * side * 4
}

/// SPA POD builders for the stream parameters (modelled on Sunshine's
/// `pipewire.cpp` / OBS' `pipewire.c`).
mod pods {
    use pipewire::spa::pod::builder::Builder;
    use pipewire::spa::sys as spa;
    use pipewire::spa::utils::{Fraction, Id, Rectangle};
    use std::mem::MaybeUninit;

    fn err<E: std::fmt::Debug>(what: &'static str) -> impl FnOnce(E) -> String {
        move |e| format!("{}: {:?}", what, e)
    }

    /// One `EnumFormat` entry: a single video format, a size range defaulting
    /// to the requested size, variable framerate capped at `fps`, and — when
    /// `modifiers` is non-empty — a mandatory, non-fixated modifier list that
    /// lets the compositor pick a GPU layout we can import.
    pub fn enum_format(
        spa_format: u32,
        width: u32,
        height: u32,
        fps: u32,
        modifiers: &[u64],
    ) -> Result<Vec<u8>, String> {
        let mut data = Vec::with_capacity(512);
        unsafe {
            let mut b = Builder::new(&mut data);
            let mut obj = MaybeUninit::uninit();
            b.push_object(&mut obj, spa::SPA_TYPE_OBJECT_Format, spa::SPA_PARAM_EnumFormat)
                .map_err(err("push Format"))?;

            b.add_prop(spa::SPA_FORMAT_mediaType, 0).map_err(err("mediaType"))?;
            b.add_id(Id(spa::SPA_MEDIA_TYPE_video)).map_err(err("mediaType"))?;
            b.add_prop(spa::SPA_FORMAT_mediaSubtype, 0).map_err(err("mediaSubtype"))?;
            b.add_id(Id(spa::SPA_MEDIA_SUBTYPE_raw)).map_err(err("mediaSubtype"))?;
            b.add_prop(spa::SPA_FORMAT_VIDEO_format, 0).map_err(err("format"))?;
            b.add_id(Id(spa_format)).map_err(err("format"))?;

            b.add_prop(spa::SPA_FORMAT_VIDEO_size, 0).map_err(err("size"))?;
            let mut choice = MaybeUninit::uninit();
            b.push_choice(&mut choice, spa::SPA_CHOICE_Range, 0).map_err(err("size choice"))?;
            b.add_rectangle(Rectangle { width: width.max(1), height: height.max(1) })
                .map_err(err("size"))?;
            b.add_rectangle(Rectangle { width: 1, height: 1 }).map_err(err("size"))?;
            b.add_rectangle(Rectangle { width: 8192, height: 8192 }).map_err(err("size"))?;
            b.pop(&mut *choice.as_mut_ptr());

            // Variable framerate (0/1) bounded by maxFramerate: the compositor
            // then pushes a frame whenever the screen changes, up to `fps`.
            b.add_prop(spa::SPA_FORMAT_VIDEO_framerate, 0).map_err(err("framerate"))?;
            b.add_fraction(Fraction { num: 0, denom: 1 }).map_err(err("framerate"))?;
            b.add_prop(spa::SPA_FORMAT_VIDEO_maxFramerate, 0).map_err(err("maxFramerate"))?;
            let mut choice = MaybeUninit::uninit();
            b.push_choice(&mut choice, spa::SPA_CHOICE_Range, 0)
                .map_err(err("maxFramerate choice"))?;
            b.add_fraction(Fraction { num: fps.max(1), denom: 1 }).map_err(err("maxFramerate"))?;
            b.add_fraction(Fraction { num: 0, denom: 1 }).map_err(err("maxFramerate"))?;
            b.add_fraction(Fraction { num: 1000, denom: 1 }).map_err(err("maxFramerate"))?;
            b.pop(&mut *choice.as_mut_ptr());

            if !modifiers.is_empty() {
                b.add_prop(
                    spa::SPA_FORMAT_VIDEO_modifier,
                    spa::SPA_POD_PROP_FLAG_MANDATORY | spa::SPA_POD_PROP_FLAG_DONT_FIXATE,
                )
                .map_err(err("modifier"))?;
                let mut choice = MaybeUninit::uninit();
                b.push_choice(&mut choice, spa::SPA_CHOICE_Enum, 0)
                    .map_err(err("modifier choice"))?;
                // First value of an Enum choice is the default.
                b.add_long(modifiers[0] as i64).map_err(err("modifier"))?;
                for &m in modifiers {
                    b.add_long(m as i64).map_err(err("modifier"))?;
                }
                b.pop(&mut *choice.as_mut_ptr());
            }

            b.pop(&mut *obj.as_mut_ptr());
        }
        Ok(data)
    }

    /// `ParamBuffers` restricting the buffer memory types.
    pub fn buffers(data_types: i32) -> Result<Vec<u8>, String> {
        let mut data = Vec::with_capacity(128);
        unsafe {
            let mut b = Builder::new(&mut data);
            let mut obj = MaybeUninit::uninit();
            b.push_object(&mut obj, spa::SPA_TYPE_OBJECT_ParamBuffers, spa::SPA_PARAM_Buffers)
                .map_err(err("push Buffers"))?;
            b.add_prop(spa::SPA_PARAM_BUFFERS_dataType, 0).map_err(err("dataType"))?;
            b.add_int(data_types).map_err(err("dataType"))?;
            b.pop(&mut *obj.as_mut_ptr());
        }
        Ok(data)
    }

    /// `ParamMeta` requesting `SPA_META_Cursor` with room for a bitmap.
    pub fn cursor_meta() -> Result<Vec<u8>, String> {
        let mut data = Vec::with_capacity(128);
        unsafe {
            let mut b = Builder::new(&mut data);
            let mut obj = MaybeUninit::uninit();
            b.push_object(&mut obj, spa::SPA_TYPE_OBJECT_ParamMeta, spa::SPA_PARAM_Meta)
                .map_err(err("push Meta"))?;
            b.add_prop(spa::SPA_PARAM_META_type, 0).map_err(err("meta type"))?;
            b.add_id(Id(spa::SPA_META_Cursor)).map_err(err("meta type"))?;
            b.add_prop(spa::SPA_PARAM_META_size, 0).map_err(err("meta size"))?;
            let mut choice = MaybeUninit::uninit();
            b.push_choice(&mut choice, spa::SPA_CHOICE_Range, 0).map_err(err("meta choice"))?;
            b.add_int(super::cursor_meta_size(64)).map_err(err("meta size"))?;
            b.add_int(super::cursor_meta_size(1)).map_err(err("meta size"))?;
            b.add_int(super::cursor_meta_size(256)).map_err(err("meta size"))?;
            b.pop(&mut *choice.as_mut_ptr());
            b.pop(&mut *obj.as_mut_ptr());
        }
        Ok(data)
    }

    /// View a built POD buffer as a `Pod`. The returned reference borrows `data`.
    pub fn as_pod(data: &[u8]) -> &pipewire::spa::pod::Pod {
        unsafe { pipewire::spa::pod::Pod::from_raw(data.as_ptr() as *const spa::spa_pod) }
    }
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

fn run_pipewire_loop(
    node_id: u32,
    frame_tx: mpsc::Sender<CapturedFrame>,
    pw_receiver: pipewire::channel::Receiver<()>,
    config: StreamConfig,
    passthrough: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<(), MediaError> {
    use pipewire::main_loop::MainLoop;
    use pipewire::spa::sys as spa;

    let mainloop = Rc::new(
        MainLoop::new(None)
            .map_err(|e| MediaError::CaptureError(format!("Failed to create MainLoop: {}", e)))?,
    );
    let context = pipewire::context::Context::new(mainloop.as_ref())
        .map_err(|e| MediaError::CaptureError(format!("Failed to create Context: {}", e)))?;
    let core = context
        .connect(None)
        .map_err(|e| MediaError::CaptureError(format!("Failed to connect Context: {}", e)))?;

    let props = pipewire::properties::properties! {
        *pipewire::keys::MEDIA_TYPE => "Video",
        *pipewire::keys::MEDIA_CATEGORY => "Capture",
        *pipewire::keys::MEDIA_ROLE => "Screen",
    };

    let stream = pipewire::stream::Stream::new(&core, "lunaris-capture", props)
        .map_err(|e| MediaError::CaptureError(format!("Failed to create Stream: {}", e)))?;

    let mainloop_clone = mainloop.clone();
    let _receiver = pw_receiver.attach(mainloop.loop_(), move |()| {
        mainloop_clone.quit();
    });

    // --- Capability discovery (which GPU, which modifiers) -----------------
    let zero_copy = passthrough.load(std::sync::atomic::Ordering::Relaxed);
    let render_node = super::dmabuf_caps::display_render_node();
    let dmabuf_formats = render_node
        .as_deref()
        .map(super::dmabuf_caps::query_dmabuf_formats)
        .unwrap_or_default();
    log::info!(
        "PipeWire: zero-copy={} render_node={:?} dmabuf formats with modifiers={}",
        zero_copy,
        render_node,
        dmabuf_formats.iter().filter(|f| !f.modifiers.is_empty()).count()
    );

    let negotiated = Rc::new(RefCell::new(NegotiatedFormat::default()));
    // The most recent DMA-BUF buffer handed to the encoder. It stays dequeued
    // (so the compositor cannot draw into it) until a newer frame replaces it.
    let held_buffer: Rc<RefCell<*mut pipewire::sys::pw_buffer>> =
        Rc::new(RefCell::new(std::ptr::null_mut()));
    // Legacy readback path (zero-copy disabled), created lazily.
    let egl_importer: Rc<RefCell<Option<super::egl_import::EglImporter>>> =
        Rc::new(RefCell::new(None));

    let negotiated_pc = negotiated.clone();
    let negotiated_proc = negotiated.clone();
    let held_proc = held_buffer.clone();
    let held_state = held_buffer.clone();
    let egl_proc = egl_importer.clone();

    let _listener = stream
        .add_local_listener_with_user_data(frame_tx)
        .state_changed(move |stream, _, old, new| {
            log::info!("PipeWire stream state: {:?} -> {:?}", old, new);
            // Return a held buffer when the stream stops streaming.
            if !matches!(new, pipewire::stream::StreamState::Streaming) {
                let prev = std::mem::replace(&mut *held_state.borrow_mut(), std::ptr::null_mut());
                if !prev.is_null() {
                    unsafe { stream.queue_raw_buffer(prev) };
                }
            }
        })
        .param_changed(move |stream, _user_data, id, pod_opt| {
            if id != spa::SPA_PARAM_Format {
                return;
            }
            let Some(pod) = pod_opt else { return };

            let Ok((media_type, media_subtype)) =
                pipewire::spa::param::format_utils::parse_format(pod)
            else {
                return;
            };
            if media_type != pipewire::spa::param::format::MediaType::Video
                || media_subtype != pipewire::spa::param::format::MediaSubtype::Raw
            {
                return;
            }

            let mut info = pipewire::spa::param::video::VideoInfoRaw::new();
            if let Err(e) = info.parse(pod) {
                log::warn!("PipeWire: failed to parse video format: {:?}", e);
                return;
            }
            let has_modifier = unsafe {
                !spa::spa_pod_find_prop(
                    pod.as_raw_ptr(),
                    std::ptr::null(),
                    spa::SPA_FORMAT_VIDEO_modifier,
                )
                .is_null()
            };
            let spa_format = info.format().as_raw();
            let size = info.size();
            let drm_format = super::dmabuf_caps::spa_to_drm_format(spa_format).unwrap_or(0);

            let neg = NegotiatedFormat {
                drm_format,
                modifier: info.modifier(),
                width: size.width,
                height: size.height,
                dmabuf: has_modifier && drm_format != 0,
                ready: true,
            };
            log::info!(
                "PipeWire negotiated: {}x{} spa_format={} drm=0x{:08X} modifier=0x{:016X} dmabuf={} max_fps={}/{}",
                neg.width,
                neg.height,
                spa_format,
                drm_format,
                neg.modifier,
                neg.dmabuf,
                info.max_framerate().num,
                info.max_framerate().denom,
            );
            *negotiated_pc.borrow_mut() = neg;

            // Tell the producer which memory we accept for this format and
            // ask for cursor metadata (with bitmap room).
            let data_types: i32 = if neg.dmabuf {
                1 << spa::SPA_DATA_DmaBuf
            } else {
                (1 << spa::SPA_DATA_MemFd) | (1 << spa::SPA_DATA_MemPtr)
            };
            match (pods::buffers(data_types), pods::cursor_meta()) {
                (Ok(buffers), Ok(meta)) => {
                    let mut params = [pods::as_pod(&buffers), pods::as_pod(&meta)];
                    if let Err(e) = stream.update_params(&mut params) {
                        log::warn!("PipeWire: update_params failed: {}", e);
                    }
                }
                (Err(e), _) | (_, Err(e)) => log::warn!("PipeWire: building params: {}", e),
            }
        })
        .process(move |stream, tx| {
            let raw_buf = unsafe { stream.dequeue_raw_buffer() };
            if raw_buf.is_null() {
                return;
            }
            // Requeue `raw_buf` on every early exit below.
            let requeue = |buf: *mut pipewire::sys::pw_buffer| unsafe {
                stream.queue_raw_buffer(buf)
            };

            let spa_buf = unsafe { (*raw_buf).buffer };
            if spa_buf.is_null() {
                requeue(raw_buf);
                return;
            }
            let spa_ref = unsafe { &*spa_buf };
            if spa_ref.n_datas == 0 || spa_ref.datas.is_null() {
                requeue(raw_buf);
                return;
            }
            let spa_data = unsafe { &*spa_ref.datas };
            if spa_data.chunk.is_null() {
                requeue(raw_buf);
                return;
            }
            let chunk = unsafe { &*spa_data.chunk };
            let cursor_meta = extract_cursor_meta(spa_buf);
            let neg = *negotiated_proc.borrow();
            if let Some(meta) = &cursor_meta {
                // Clients place the overlay in *encoded video* pixels; the
                // compositor reports frame pixels, which differ when the
                // monitor resolution is not the stream resolution.
                let (sx, sy) = (
                    scale_coord(meta.x, neg.width, config.width),
                    scale_coord(meta.y, neg.height, config.height),
                );
                publish_cursor(&FrameCursorMeta { x: sx, y: sy, ..meta.clone() });
            }

            // Corrupted buffers and cursor-only updates (size 0) carry no new
            // picture; the pipeline keeps re-encoding the previous frame.
            if !neg.ready
                || (chunk.flags & spa::SPA_CHUNK_FLAG_CORRUPTED as i32) != 0
                || chunk.size == 0
            {
                requeue(raw_buf);
                return;
            }

            let width = neg.width;
            let height = neg.height;

            if spa_data.type_ == spa::SPA_DATA_DmaBuf {
                if spa_ref.n_datas > 1 {
                    // Multi-plane layouts are filtered out during negotiation;
                    // if one slips through we cannot describe it.
                    log::warn!(
                        "PipeWire: multi-plane DMA-BUF ({} planes) unsupported; dropping frame",
                        spa_ref.n_datas
                    );
                    requeue(raw_buf);
                    return;
                }

                if passthrough.load(std::sync::atomic::Ordering::Relaxed) {
                    let fd = unsafe { libc::fcntl(spa_data.fd as i32, libc::F_DUPFD_CLOEXEC, 0) };
                    if fd < 0 {
                        log::warn!("PipeWire: dup DMA-BUF fd failed: {}", std::io::Error::last_os_error());
                        requeue(raw_buf);
                        return;
                    }
                    let frame = CapturedFrame {
                        buffer: GpuBuffer::DmaBuf {
                            fd,
                            offset: chunk.offset,
                            stride: chunk.stride as u32,
                            modifier: neg.modifier,
                            size: spa_data.maxsize as usize,
                            width,
                            height,
                            fourcc: neg.drm_format,
                        },
                        timestamp_us: now_us(),
                        width,
                        height,
                        format: PixelFormat::BGRA,
                        is_new_frame: true,
                        cursor: cursor_meta,
                    };
                    match tx.try_send(frame) {
                        Ok(()) => {
                            // Keep this buffer away from the compositor while
                            // the encoder reads it; release the previous one.
                            let prev = std::mem::replace(&mut *held_proc.borrow_mut(), raw_buf);
                            if !prev.is_null() {
                                requeue(prev);
                            }
                        }
                        Err(_) => {
                            // Encoder busy: frame (and its dup'd fd) dropped.
                            requeue(raw_buf);
                        }
                    }
                    return;
                }

                // Legacy: EGL import + glReadPixels into a CPU buffer.
                let mut egl_ref = egl_proc.borrow_mut();
                if egl_ref.is_none() {
                    match super::egl_import::EglImporter::new() {
                        Ok(importer) => *egl_ref = Some(importer),
                        Err(e) => log::warn!("EGL importer unavailable: {}", e),
                    }
                }
                if let Some(importer) = egl_ref.as_mut() {
                    match importer.import_dmabuf(
                        spa_data.fd as i32,
                        width,
                        height,
                        chunk.stride as u32,
                        neg.drm_format,
                        chunk.offset,
                        neg.modifier,
                    ) {
                        Ok(bgra) => {
                            let _ = tx.try_send(CapturedFrame {
                                buffer: GpuBuffer::CpuBuffer {
                                    data: bgra,
                                    stride: width * 4,
                                    format: PixelFormat::BGRA,
                                    width,
                                    height,
                                },
                                timestamp_us: now_us(),
                                width,
                                height,
                                format: PixelFormat::BGRA,
                                is_new_frame: true,
                                cursor: cursor_meta,
                            });
                        }
                        Err(e) => log::warn!("EGL DMA-BUF import failed: {}", e),
                    }
                }
                requeue(raw_buf);
                return;
            }

            // Shared-memory buffers (only BGRx/BGRA are offered for these).
            let offset = chunk.offset as usize;
            let size = chunk.size as usize;
            let stride = chunk.stride as u32;
            let data: Option<Vec<u8>> = if !spa_data.data.is_null()
                && offset.saturating_add(size) <= spa_data.maxsize as usize
            {
                // MAP_BUFFERS maps MemFd too, so `data` is usually valid.
                Some(unsafe {
                    std::slice::from_raw_parts((spa_data.data as *const u8).add(offset), size)
                        .to_vec()
                })
            } else if spa_data.type_ == spa::SPA_DATA_MemFd && spa_data.fd >= 0 {
                let map_size = offset.saturating_add(size);
                let mapped = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        map_size,
                        libc::PROT_READ,
                        libc::MAP_SHARED,
                        spa_data.fd as i32,
                        0,
                    )
                };
                if mapped == libc::MAP_FAILED {
                    log::warn!("PipeWire: MemFd mmap failed: {}", std::io::Error::last_os_error());
                    None
                } else {
                    let v = unsafe {
                        std::slice::from_raw_parts((mapped as *const u8).add(offset), size).to_vec()
                    };
                    unsafe { libc::munmap(mapped, map_size) };
                    Some(v)
                }
            } else {
                None
            };

            if let Some(data) = data {
                let _ = tx.try_send(CapturedFrame {
                    buffer: GpuBuffer::CpuBuffer {
                        data,
                        stride,
                        format: PixelFormat::BGRA,
                        width,
                        height,
                    },
                    timestamp_us: now_us(),
                    width,
                    height,
                    format: PixelFormat::BGRA,
                    is_new_frame: true,
                    cursor: cursor_meta,
                });
            }
            requeue(raw_buf);
        })
        .register()
        .map_err(|e| MediaError::CaptureError(format!("Failed to register stream listener: {}", e)))?;

    // --- EnumFormat offers: DMA-BUF (with modifiers) first, SHM fallback ---
    let mut format_pods: Vec<Vec<u8>> = Vec::new();
    for f in &dmabuf_formats {
        if f.modifiers.is_empty() {
            continue;
        }
        let Some(spa_format) = super::dmabuf_caps::drm_to_spa_format(f.drm_format) else {
            continue;
        };
        match pods::enum_format(spa_format, config.width, config.height, config.fps, &f.modifiers) {
            Ok(p) => format_pods.push(p),
            Err(e) => log::warn!("PipeWire: building DMA-BUF format pod: {}", e),
        }
    }
    for spa_format in [spa::SPA_VIDEO_FORMAT_BGRx, spa::SPA_VIDEO_FORMAT_BGRA] {
        match pods::enum_format(spa_format, config.width, config.height, config.fps, &[]) {
            Ok(p) => format_pods.push(p),
            Err(e) => log::warn!("PipeWire: building SHM format pod: {}", e),
        }
    }
    let mut params: Vec<&pipewire::spa::pod::Pod> =
        format_pods.iter().map(|p| pods::as_pod(p)).collect();

    stream
        .connect(
            pipewire::spa::utils::Direction::Input,
            Some(node_id),
            pipewire::stream::StreamFlags::AUTOCONNECT | pipewire::stream::StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .map_err(|e| MediaError::CaptureError(format!("Failed to connect Stream: {}", e)))?;

    log::info!(
        "Running PipeWire MainLoop ({} format offers)...",
        format_pods.len()
    );
    mainloop.run();

    // Release a still-held buffer before the stream is torn down.
    let prev = std::mem::replace(&mut *held_buffer.borrow_mut(), std::ptr::null_mut());
    if !prev.is_null() {
        unsafe { stream.queue_raw_buffer(prev) };
    }
    log::info!("PipeWire MainLoop exited");
    Ok(())
}

/// Extract cursor metadata from a PipeWire spa_buffer, if present.
///
/// Returns `Some(FrameCursorMeta)` when the compositor provided `SPA_META_Cursor`
/// data in the buffer. Returns `None` if the metadata is absent or invalid.
fn extract_cursor_meta(
    spa_buf: *mut pipewire::spa::sys::spa_buffer,
) -> Option<FrameCursorMeta> {
    if spa_buf.is_null() {
        return None;
    }
    unsafe {
        let meta_ptr = pipewire::spa::sys::spa_buffer_find_meta(
            spa_buf,
            pipewire::spa::sys::SPA_META_Cursor,
        );
        if meta_ptr.is_null() {
            return None;
        }
        let meta = &*meta_ptr;
        if meta.data.is_null() || meta.size == 0 {
            return None;
        }
        let cursor = &*(meta.data as *const pipewire::spa::sys::spa_meta_cursor);
        // id == 0 means no new cursor data this frame
        if cursor.id == 0 {
            return None;
        }

        let x = cursor.position.x;
        let y = cursor.position.y;
        let hotspot_x = cursor.hotspot.x;
        let hotspot_y = cursor.hotspot.y;

        // Check if bitmap data is available
        let cursor_meta_size = std::mem::size_of::<pipewire::spa::sys::spa_meta_cursor>() as u32;
        let image = if cursor.bitmap_offset >= cursor_meta_size {
            let bitmap_ptr = (meta.data as *const u8).add(cursor.bitmap_offset as usize)
                as *const pipewire::spa::sys::spa_meta_bitmap;
            let bitmap = &*bitmap_ptr;
            let bitmap_meta_size =
                std::mem::size_of::<pipewire::spa::sys::spa_meta_bitmap>() as u32;

            if bitmap.offset >= bitmap_meta_size
                && bitmap.size.width > 0
                && bitmap.size.height > 0
            {
                let pixel_data =
                    (bitmap_ptr as *const u8).add(bitmap.offset as usize);
                let width = bitmap.size.width;
                let height = bitmap.size.height;
                let stride = bitmap.stride as u32;
                let expected_len = (stride * height) as usize;

                // Convert pixel data to RGBA based on the spa_video_format
                let rgba = convert_cursor_bitmap_to_rgba(
                    pixel_data,
                    expected_len,
                    bitmap.format,
                    width,
                    height,
                    stride,
                );

                rgba.map(|data| CursorImage {
                    width,
                    height,
                    hotspot_x: hotspot_x.max(0) as u32,
                    hotspot_y: hotspot_y.max(0) as u32,
                    rgba_data: data,
                })
            } else {
                None
            }
        } else {
            None
        };

        Some(FrameCursorMeta {
            x,
            y,
            hotspot_x,
            hotspot_y,
            visible: true,
            image,
        })
    }
}

/// Convert cursor bitmap pixel data from a spa_video_format to RGBA.
fn convert_cursor_bitmap_to_rgba(
    raw: *const u8,
    raw_len: usize,
    format: u32,
    width: u32,
    height: u32,
    stride: u32,
) -> Option<Vec<u8>> {
    use pipewire::spa::sys::*;
    let expected = (stride * height) as usize;
    if raw_len < expected {
        return None;
    }
    let src = unsafe { std::slice::from_raw_parts(raw, expected) };
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);

    for row in 0..height {
        let row_offset = (row * stride) as usize;
        for col in 0..width {
            let px_offset = row_offset + (col * 4) as usize;
            if px_offset + 4 > src.len() {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            let (r, g, b, a) = match format {
                SPA_VIDEO_FORMAT_RGBA => (
                    src[px_offset],
                    src[px_offset + 1],
                    src[px_offset + 2],
                    src[px_offset + 3],
                ),
                SPA_VIDEO_FORMAT_BGRA => (
                    src[px_offset + 2],
                    src[px_offset + 1],
                    src[px_offset],
                    src[px_offset + 3],
                ),
                SPA_VIDEO_FORMAT_ARGB => (
                    src[px_offset + 1],
                    src[px_offset + 2],
                    src[px_offset + 3],
                    src[px_offset],
                ),
                #[allow(non_upper_case_globals)]
                SPA_VIDEO_FORMAT_BGRx => (
                    src[px_offset + 2],
                    src[px_offset + 1],
                    src[px_offset],
                    0xFF,
                ),
                #[allow(non_upper_case_globals)]
                SPA_VIDEO_FORMAT_RGBx => (
                    src[px_offset],
                    src[px_offset + 1],
                    src[px_offset + 2],
                    0xFF,
                ),
                // Default: treat as RGBA
                _ => (
                    src[px_offset],
                    src[px_offset + 1],
                    src[px_offset + 2],
                    src[px_offset + 3],
                ),
            };
            rgba.extend_from_slice(&[r, g, b, a]);
        }
    }

    Some(rgba)
}

#[cfg(test)]
mod tests {
    use super::scale_coord;

    #[test]
    fn cursor_coordinates_follow_encoder_size() {
        assert_eq!(scale_coord(1280, 2560, 1920), 960);
        assert_eq!(scale_coord(720, 1440, 1080), 540);
        assert_eq!(scale_coord(500, 1920, 1920), 500);
        assert_eq!(scale_coord(500, 0, 1920), 500);
    }
}
