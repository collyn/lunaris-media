//! The consumer side: [`DrmServiceCapture`], a [`ScreenCapture`] backend that
//! receives DMA-BUF frames from the capture service.
//!
//! Frames arrive as read-only DMA-BUFs of the GPU that scans the display
//! out; the backend points the VAAPI encoder at that GPU's render node
//! (sysfs card → renderD mapping) so the import stays zero-copy. The hardware
//! cursor is published through the same slot the PipeWire backend fills, so
//! the cursor pipeline needs no change.

use std::os::fd::{IntoRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;

use super::kms;
use super::protocol::{ErrorCode, Message, StatusCode, WireCursor, WireDisplay, WireFrame, PROTOCOL_VERSION};
use super::transport::{Conn, TransportError};
use crate::capture::gpu_buffer::{DmaBufAuxPlanes, GpuBuffer};
use crate::capture::{CapturedFrame, FrameCursorMeta, ScreenCapture};
use crate::error::MediaError;
use crate::types::*;

/// Frames the service may have outstanding for us.
const MAX_IN_FLIGHT: u32 = 2;
/// Channel between the reader thread and `next_frame`.
const CHANNEL_CAPACITY: usize = 4;

/// Why connecting to the service failed.
#[derive(Debug)]
pub enum ClientError {
    /// No service (socket missing, refused, timed out).
    Unavailable(String),
    /// The service refused this process.
    Refused(String),
    Protocol(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::Unavailable(m) => write!(f, "capture service unavailable: {}", m),
            ClientError::Refused(m) => write!(f, "capture service refused this process: {}", m),
            ClientError::Protocol(m) => write!(f, "capture service protocol error: {}", m),
        }
    }
}

impl From<ClientError> for MediaError {
    fn from(e: ClientError) -> Self {
        match e {
            ClientError::Refused(m) => MediaError::PermissionDenied(format!("capture service: {}", m)),
            other => MediaError::CaptureError(other.to_string()),
        }
    }
}

/// A handshaken connection to the service.
pub struct ServiceClient {
    pub conn: Arc<Conn>,
    pub displays: Vec<WireDisplay>,
}

impl ServiceClient {
    /// Connect, check that the service runs as root (or as ourselves, for
    /// tests) and exchange `Hello` + `DisplayList`.
    pub fn connect(path: &Path, timeout: Duration) -> Result<ServiceClient, ClientError> {
        let conn = Conn::connect(path).map_err(|e| ClientError::Unavailable(format!("{}: {}", path.display(), e)))?;
        Self::handshake(conn, timeout)
    }

    /// Handshake over an already connected socket.
    pub fn handshake(conn: Conn, timeout: Duration) -> Result<ServiceClient, ClientError> {
        let cred = conn.peer_cred().map_err(|e| ClientError::Unavailable(e.to_string()))?;
        // SAFETY: getuid has no preconditions.
        let me = unsafe { libc::getuid() };
        if cred.uid != 0 && cred.uid != me {
            return Err(ClientError::Refused(format!(
                "the socket is served by uid {}, not root",
                cred.uid
            )));
        }
        let _ = conn.set_send_timeout(Duration::from_secs(2));
        // A refusing service may already have answered and closed; read its
        // answer even if this send fails.
        let sent = conn.send(&Message::Hello { version: PROTOCOL_VERSION }, &[]);
        let mut displays = None;
        let mut hello = false;
        while displays.is_none() {
            match conn.recv(Some(timeout)) {
                Ok(Some((Message::Hello { version }, _))) if version == PROTOCOL_VERSION => hello = true,
                Ok(Some((Message::Hello { version }, _))) => {
                    return Err(ClientError::Protocol(format!("service speaks version {}", version)))
                }
                Ok(Some((Message::DisplayList(list), _))) if hello => displays = Some(list),
                Ok(Some((Message::Error { code: ErrorCode::Unauthorized, message }, _))) => {
                    return Err(ClientError::Refused(message))
                }
                Ok(Some((Message::Error { message, .. }, _))) => return Err(ClientError::Protocol(message)),
                Ok(Some((other, _))) => {
                    return Err(ClientError::Protocol(format!("unexpected {:?} during handshake", other.kind())))
                }
                Ok(None) => return Err(ClientError::Unavailable("no answer".into())),
                Err(TransportError::Closed) => {
                    let why = sent.err().map(|e| e.to_string()).unwrap_or_else(|| "connection closed".into());
                    return Err(ClientError::Unavailable(why));
                }
                Err(e) => return Err(ClientError::Protocol(e.to_string())),
            }
        }
        Ok(ServiceClient { conn: Arc::new(conn), displays: displays.unwrap_or_default() })
    }
}

/// Socket path in use (`LUNARIS_CAPTURE_SOCKET` overrides the default).
pub fn socket_path() -> PathBuf {
    std::env::var_os("LUNARIS_CAPTURE_SOCKET")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(super::DEFAULT_SOCKET_PATH))
}

/// `true` when a capture service answers on the socket and accepts this
/// process (lets callers offer every monitor on Wayland).
pub fn service_available() -> bool {
    let path = socket_path();
    if !path.exists() {
        return false;
    }
    ServiceClient::connect(&path, Duration::from_millis(1500)).is_ok()
}

/// The id a display gets in [`DisplayInfo::id`].
pub fn display_info_id(d: &WireDisplay) -> String {
    format!("kms:{}:{}", d.card, d.name)
}

fn pixel_format_of(fourcc: u32) -> PixelFormat {
    use drm_fourcc::DrmFourcc;
    match DrmFourcc::try_from(fourcc) {
        Ok(DrmFourcc::Nv12) => PixelFormat::NV12,
        Ok(DrmFourcc::P010) => PixelFormat::P010,
        _ => PixelFormat::BGRA,
    }
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// Why a frame could not be turned into a [`CapturedFrame`].
#[derive(Debug, PartialEq, Eq)]
pub enum FrameRejection {
    MultiPlane(usize),
    MissingFd,
}

/// Turn a received frame into a [`CapturedFrame`] owning the descriptor.
/// All planes must live in one buffer object (one descriptor), as plain and
/// compressed (DCC/CCS) scan-out buffers do; further planes become
/// `aux_planes`. A crop of a linear framebuffer is applied through the
/// offset, other crops show the whole framebuffer.
pub fn captured_frame(frame: &WireFrame, mut fds: Vec<OwnedFd>) -> Result<CapturedFrame, FrameRejection> {
    if frame.planes.is_empty() || fds.len() != 1 || frame.planes.iter().any(|p| p.fd_index != 0) {
        return Err(FrameRejection::MultiPlane(frame.planes.len()));
    }
    let aux: Vec<(u32, u32)> = frame.planes[1..].iter().map(|p| (p.offset, p.stride)).collect();
    let aux_planes = DmaBufAuxPlanes::from_slice(&aux).ok_or(FrameRejection::MultiPlane(frame.planes.len()))?;
    let fd = fds.pop().ok_or(FrameRejection::MissingFd)?;
    let plane = frame.planes[0];
    let full = frame.crop_x == 0 && frame.crop_y == 0 && frame.crop_w == frame.width && frame.crop_h == frame.height;
    let (mut offset, mut width, mut height) = (plane.offset, frame.width, frame.height);
    if !full && frame.modifier == kms::MOD_LINEAR && aux_planes.count == 0 {
        offset = offset
            .saturating_add(frame.crop_y.saturating_mul(plane.stride))
            .saturating_add(frame.crop_x.saturating_mul(4));
        width = frame.crop_w;
        height = frame.crop_h;
    }
    // A DMA-BUF reports its size through lseek(SEEK_END).
    let raw = fd.into_raw_fd();
    // SAFETY: lseek on a descriptor we own; the result is only a size hint.
    let end = unsafe { libc::lseek(raw, 0, libc::SEEK_END) };
    // SAFETY: rewinding the same descriptor.
    unsafe { libc::lseek(raw, 0, libc::SEEK_SET) };
    let size = if end > 0 { end as usize } else { plane.stride as usize * frame.height as usize };
    Ok(CapturedFrame {
        buffer: GpuBuffer::DmaBuf {
            fd: raw,
            offset,
            stride: plane.stride,
            modifier: frame.modifier,
            size,
            width,
            height,
            fourcc: frame.fourcc,
            aux_planes,
        },
        timestamp_us: now_us(),
        width,
        height,
        format: pixel_format_of(frame.fourcc),
        is_new_frame: true,
        cursor: None,
    })
}

/// Cursor metadata in the shape the PipeWire backend publishes: position of
/// the pointer (image origin + hotspot) in encoded-video pixels.
pub fn cursor_meta(c: &WireCursor, display: (u32, u32), encoded: (u32, u32)) -> FrameCursorMeta {
    cursor_meta_with(c, (c.hot_x, c.hot_y), c.image.as_ref(), display, encoded)
}

/// `cursor_meta` with the hotspot `hot`, and with `image` (the cursor's
/// picture, sent again when its hotspot changed).
fn cursor_meta_with(
    c: &WireCursor,
    hot: (i32, i32),
    image: Option<&Vec<u8>>,
    display: (u32, u32),
    encoded: (u32, u32),
) -> FrameCursorMeta {
    use crate::capture::linux_wayland::scale_coord;
    let image = image.map(|rgba| CursorImage {
        width: c.width,
        height: c.height,
        hotspot_x: hot.0.max(0) as u32,
        hotspot_y: hot.1.max(0) as u32,
        rgba_data: rgba.clone(),
    });
    FrameCursorMeta {
        x: scale_coord(c.x + hot.0, display.0, encoded.0),
        y: scale_coord(c.y + hot.1, display.1, encoded.1),
        hotspot_x: hot.0,
        hotspot_y: hot.1,
        visible: c.visible,
        image,
    }
}

/// The cursor's hotspot where KMS has none. A cursor plane says where the
/// image's corner is; only virtual GPUs have the HOTSPOT_X/Y properties, so
/// on real ones the hotspot comes as 0,0 and the pointer would be shown (and
/// said to be) a few pixels up and left of where it is: the arrow's tip
/// drawn by the viewer lands below and right of where a click goes.
///
/// Each new picture gets a guess (its first opaque pixel: near an arrow's
/// tip), then the real hotspot is learned once the viewer has put the
/// pointer somewhere (an absolute move, input.rs) and it stays there: the
/// pointer minus the image's corner as reported after that move. What was
/// learned is kept by picture.
#[derive(Default)]
struct Hotspots {
    width: u32,
    height: u32,
    image: Option<Vec<u8>>,
    key: u64,
    hot: (i32, i32),
    learned: std::collections::HashMap<u64, (i32, i32)>,
    /// The last report (without its picture) and when it came.
    last: Option<(WireCursor, std::time::Instant)>,
    /// The absolute move last looked at (each is looked at once).
    checked: Option<std::time::Instant>,
}

impl Hotspots {
    /// How long the pointer has to stay where the viewer put it.
    const SETTLED: Duration = Duration::from_millis(40);
    /// A move older than this says nothing any more.
    const STALE: Duration = Duration::from_millis(1500);

    /// The meta for `c`: its own hotspot when KMS has one, else this one.
    fn meta(&mut self, c: &WireCursor, display: (u32, u32), encoded: (u32, u32)) -> FrameCursorMeta {
        if c.hot_x != 0 || c.hot_y != 0 {
            self.last = None;
            return cursor_meta(c, display, encoded);
        }
        if let Some(rgba) = &c.image {
            self.width = c.width;
            self.height = c.height;
            self.key = picture_key(c.width, c.height, rgba);
            self.hot = self.learned.get(&self.key).copied().unwrap_or_else(|| first_opaque(c.width, c.height, rgba));
            self.image = Some(rgba.clone());
        }
        self.last = Some((WireCursor { image: None, ..c.clone() }, std::time::Instant::now()));
        cursor_meta_with(c, self.hot, c.image.as_ref(), display, encoded)
    }

    /// Learns the hotspot when the viewer's last absolute move has settled
    /// and the cursor was reported after it; the meta to publish again
    /// (with the picture, whose hotspot changed) when it is a new one.
    fn settle(&mut self, display: (u32, u32), encoded: (u32, u32)) -> Option<FrameCursorMeta> {
        let (c, seen_at) = self.last.clone()?;
        self.image.as_ref()?;
        let (fx, fy, at) = crate::input::last_absolute_pointer()?;
        if self.checked == Some(at) || at.elapsed() < Self::SETTLED {
            return None;
        }
        self.checked = Some(at);
        if at.elapsed() > Self::STALE || seen_at < at || !c.visible || display.0 == 0 || display.1 == 0 {
            return None;
        }
        let seen = hotspot_from(fx, fy, display, (c.x, c.y), (self.width, self.height))?;
        if seen == self.hot {
            return None;
        }
        log::debug!("capture service: cursor hotspot {:?} (was {:?})", seen, self.hot);
        self.hot = seen;
        self.learned.insert(self.key, seen);
        Some(cursor_meta_with(&c, seen, self.image.as_ref(), display, encoded))
    }
}

/// Pointer (`fx`, `fy` of the display) minus the image's corner, when that
/// is inside the image.
fn hotspot_from(fx: f64, fy: f64, display: (u32, u32), corner: (i32, i32), size: (u32, u32)) -> Option<(i32, i32)> {
    let px = (fx * display.0 as f64).floor() as i32;
    let py = (fy * display.1 as f64).floor() as i32;
    let hot = (px - corner.0, py - corner.1);
    (hot.0 >= 0 && hot.1 >= 0 && (hot.0 as u32) < size.0 && (hot.1 as u32) < size.1).then_some(hot)
}

/// The first pixel that is not transparent, row by row: the tip of an arrow.
fn first_opaque(width: u32, height: u32, rgba: &[u8]) -> (i32, i32) {
    let w = width as usize;
    for y in 0..height as usize {
        for x in 0..w {
            if rgba.get((y * w + x) * 4 + 3).is_some_and(|a| *a >= 128) {
                return (x as i32, y as i32);
            }
        }
    }
    (0, 0)
}

fn picture_key(width: u32, height: u32, rgba: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (width, height).hash(&mut h);
    rgba.hash(&mut h);
    h.finish()
}

struct Incoming {
    frame: CapturedFrame,
    seq: u64,
}

/// State shared by the backend and its reader thread.
struct Shared {
    conn: Mutex<Option<Arc<Conn>>>,
    shutdown: AtomicBool,
}

impl Shared {
    fn ack(&self, seq: u64) {
        let conn = self.conn.lock().ok().and_then(|c| c.clone());
        if let Some(conn) = conn {
            let _ = conn.send(&Message::FrameAck { seq }, &[]);
        }
    }
}

/// Screen capture through the root capture service (no portal prompt).
pub struct DrmServiceCapture {
    socket: PathBuf,
    client: Option<ServiceClient>,
    displays: Vec<WireDisplay>,
    shared: Arc<Shared>,
    frame_rx: Option<mpsc::Receiver<Incoming>>,
    reader: Option<std::thread::JoinHandle<()>>,
    capturing: bool,
    fps: u32,
    streaming: Option<WireDisplay>,
    first_frame: bool,
    /// When `next_frame` last returned (a new picture or a repeat). An
    /// unchanged screen is repeated one frame interval after it, so the
    /// stream keeps the requested rate.
    last_out: std::time::Instant,
}

impl DrmServiceCapture {
    /// Connect to the service at the default (or `LUNARIS_CAPTURE_SOCKET`)
    /// path. Fails when there is none or it refuses this process.
    pub fn new() -> Result<Self, MediaError> {
        Self::connect(&socket_path())
    }

    pub fn connect(socket: &Path) -> Result<Self, MediaError> {
        let client = ServiceClient::connect(socket, Duration::from_millis(1500))?;
        Ok(Self::from_client(socket.to_path_buf(), client))
    }

    /// Wrap an already handshaken client (tests use a socketpair).
    pub fn from_client(socket: PathBuf, client: ServiceClient) -> Self {
        DrmServiceCapture {
            socket,
            displays: client.displays.clone(),
            client: Some(client),
            shared: Arc::new(Shared { conn: Mutex::new(None), shutdown: AtomicBool::new(false) }),
            frame_rx: None,
            reader: None,
            capturing: false,
            fps: 60,
            streaming: None,
            first_frame: true,
            last_out: std::time::Instant::now(),
        }
    }

    /// The displays the service reported.
    pub fn wire_displays(&self) -> &[WireDisplay] {
        &self.displays
    }

    fn desktop_region(&self, d: &WireDisplay) -> crate::input::ScreenRegion {
        if let Some(r) = crate::input::desktop_output_named(&d.name) {
            return r;
        }
        // Unknown layout (no kscreen-doctor/xrandr names): assume the
        // displays sit left to right in the service's order.
        let x: u32 = self.displays.iter().take_while(|o| o.id != d.id).map(|o| o.width).sum();
        crate::input::ScreenRegion { x: x as i32, y: 0, width: d.width as i32, height: d.height as i32 }
    }

    fn pick(&self, display_id: &str) -> Option<WireDisplay> {
        self.displays
            .iter()
            .find(|d| display_info_id(d) == display_id || d.name == display_id)
            .or_else(|| {
                self.displays
                    .iter()
                    .find(|d| crate::input::desktop_output_named(&d.name).is_some_and(|r| r.x == 0 && r.y == 0))
            })
            .or_else(|| self.displays.first())
            .cloned()
    }

    fn halt(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        let conn = self.shared.conn.lock().ok().and_then(|mut c| c.take());
        if let Some(conn) = conn {
            let _ = conn.send(&Message::Stop, &[]);
            conn.shutdown();
        }
        if let Some(h) = self.reader.take() {
            let _ = h.join();
        }
        self.frame_rx = None;
        if self.capturing {
            crate::capture::linux_wayland::clear_cursor();
            crate::capture::dmabuf_caps::set_capture_render_node(None);
        }
        self.capturing = false;
        self.streaming = None;
    }
}

/// What the reader thread knows.
struct ReaderCtx {
    socket: PathBuf,
    shared: Arc<Shared>,
    tx: mpsc::Sender<Incoming>,
    display: WireDisplay,
    fps: u32,
    encoded: (u32, u32),
}

fn start_msg(display: &WireDisplay, fps: u32) -> Message {
    Message::Start {
        display_id: display.id,
        max_fps: fps.clamp(1, super::protocol::MAX_FPS),
        max_in_flight: MAX_IN_FLIGHT,
    }
}

fn reader_loop(mut ctx: ReaderCtx, mut conn: Arc<Conn>) {
    let mut warned_planes = false;
    let mut hotspots = Hotspots::default();
    'outer: loop {
        if ctx.shared.shutdown.load(Ordering::SeqCst) {
            return;
        }
        let received = conn.recv(Some(Duration::from_millis(250)));
        // The hotspot, once the pointer the viewer moved has stopped (also
        // when nothing came: the last report is the pointer at rest).
        if let Some(meta) = hotspots.settle((ctx.display.width, ctx.display.height), ctx.encoded) {
            crate::capture::linux_wayland::publish_cursor(&meta);
        }
        match received {
            Ok(None) => {}
            Ok(Some((Message::Frame(f), fds))) => {
                if f.display_id != ctx.display.id {
                    ctx.shared.ack(f.seq);
                    continue;
                }
                match captured_frame(&f, fds) {
                    Ok(frame) => match ctx.tx.try_send(Incoming { frame, seq: f.seq }) {
                        Ok(()) => {}
                        Err(mpsc::error::TrySendError::Full(dropped)) => ctx.shared.ack(dropped.seq),
                        Err(mpsc::error::TrySendError::Closed(_)) => return,
                    },
                    Err(why) => {
                        if !warned_planes {
                            warned_planes = true;
                            log::warn!(
                                "capture service: cannot use frames of {} ({:?}, modifier {:#x}); \
                                 try LUNARIS_CAPTURE=portal",
                                ctx.display.name,
                                why,
                                f.modifier
                            );
                        }
                        ctx.shared.ack(f.seq);
                    }
                }
            }
            Ok(Some((Message::Cursor(c), _))) => {
                if c.display_id == ctx.display.id {
                    let meta = hotspots.meta(&c, (ctx.display.width, ctx.display.height), ctx.encoded);
                    crate::capture::linux_wayland::publish_cursor(&meta);
                }
            }
            Ok(Some((Message::Status { display_id, code }, _))) => {
                if display_id == ctx.display.id || display_id == 0 {
                    match code {
                        StatusCode::NoFramebuffer => log::info!("capture service: {} is off/blanked", ctx.display.name),
                        StatusCode::Paused => log::info!("capture service: paused (this session is not active)"),
                        StatusCode::Active => log::info!("capture service: {} streams again", ctx.display.name),
                        StatusCode::Gone => log::warn!("capture service: {} is gone", ctx.display.name),
                    }
                }
            }
            Ok(Some((Message::DisplaysChanged(list), _))) | Ok(Some((Message::DisplayList(list), _))) => {
                if !list.iter().any(|d| d.id == ctx.display.id) {
                    // Same output under a new id (e.g. re-plugged): follow it.
                    if let Some(d) = list.iter().find(|d| d.name == ctx.display.name && d.card == ctx.display.card) {
                        ctx.display = d.clone();
                        let _ = conn.send(&start_msg(&ctx.display, ctx.fps), &[]);
                    }
                } else if let Some(d) = list.iter().find(|d| d.id == ctx.display.id) {
                    ctx.display = d.clone();
                }
            }
            Ok(Some((Message::Error { code, message }, _))) => {
                log::warn!("capture service: {:?}: {}", code, message);
                if code == ErrorCode::Unauthorized {
                    return;
                }
            }
            Ok(Some(_)) => {}
            Err(e) => {
                if ctx.shared.shutdown.load(Ordering::SeqCst) {
                    return;
                }
                log::warn!("capture service: connection lost ({}); reconnecting", e);
                // The service restarts in well under a few seconds.
                for attempt in 0..20u32 {
                    std::thread::sleep(Duration::from_millis(250 + 50 * attempt as u64));
                    if ctx.shared.shutdown.load(Ordering::SeqCst) {
                        return;
                    }
                    let Ok(client) = ServiceClient::connect(&ctx.socket, Duration::from_millis(1500)) else {
                        continue;
                    };
                    let Some(d) = client
                        .displays
                        .iter()
                        .find(|d| d.id == ctx.display.id)
                        .or_else(|| client.displays.iter().find(|d| d.name == ctx.display.name))
                        .cloned()
                    else {
                        log::warn!("capture service: {} no longer exists", ctx.display.name);
                        return;
                    };
                    if client.conn.send(&start_msg(&d, ctx.fps), &[]).is_err() {
                        continue;
                    }
                    ctx.display = d;
                    conn = client.conn.clone();
                    if let Ok(mut slot) = ctx.shared.conn.lock() {
                        *slot = Some(conn.clone());
                    }
                    log::info!("capture service: reconnected");
                    continue 'outer;
                }
                log::error!("capture service: could not reconnect");
                return;
            }
        }
    }
}

#[async_trait::async_trait]
impl ScreenCapture for DrmServiceCapture {
    async fn list_displays(&self) -> Result<Vec<DisplayInfo>, MediaError> {
        let primary = self.pick("default").map(|d| d.id);
        Ok(self
            .displays
            .iter()
            .map(|d| DisplayInfo {
                id: display_info_id(d),
                name: d.name.clone(),
                width: d.width,
                height: d.height,
                refresh_rate: d.refresh_mhz as f64 / 1000.0,
                is_primary: Some(d.id) == primary,
            })
            .collect())
    }

    async fn start(&mut self, display_id: &str, config: &StreamConfig) -> Result<(), MediaError> {
        if self.capturing {
            return Err(MediaError::CaptureAlreadyStarted);
        }
        let client = match self.client.take() {
            Some(c) => c,
            None => ServiceClient::connect(&self.socket, Duration::from_millis(1500))?,
        };
        self.displays = client.displays.clone();
        let display = self.pick(display_id).ok_or(MediaError::NoDisplayFound)?;
        self.fps = config.fps.max(1);

        match kms::render_node_for_card(Path::new("/sys"), &display.card) {
            Some(node) => {
                log::info!("capture service: {} is on {} ({})", display.name, display.card, node);
                crate::capture::dmabuf_caps::set_capture_render_node(Some(node));
            }
            None => log::warn!("capture service: no render node for {}", display.card),
        }
        let region = self.desktop_region(&display);
        crate::input::set_capture_region(region);

        client
            .conn
            .send(&start_msg(&display, self.fps), &[])
            .map_err(|e| MediaError::CaptureError(format!("capture service: {}", e)))?;
        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        self.shared = Arc::new(Shared { conn: Mutex::new(Some(client.conn.clone())), shutdown: AtomicBool::new(false) });
        let ctx = ReaderCtx {
            socket: self.socket.clone(),
            shared: self.shared.clone(),
            tx,
            display: display.clone(),
            fps: self.fps,
            encoded: (config.width, config.height),
        };
        let conn = client.conn.clone();
        let reader = std::thread::Builder::new()
            .name("lunaris-capsvc-rx".into())
            .spawn(move || reader_loop(ctx, conn))
            .map_err(|e| MediaError::CaptureError(e.to_string()))?;
        self.reader = Some(reader);
        self.frame_rx = Some(rx);
        self.capturing = true;
        self.first_frame = true;
        log::info!(
            "Using the Lunaris capture service: {} ({}x{} @{:.0}Hz) at {} fps, region {:?}",
            display.name,
            display.width,
            display.height,
            display.refresh_mhz as f64 / 1000.0,
            self.fps,
            region
        );
        self.streaming = Some(display);
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<CapturedFrame, MediaError> {
        if !self.capturing {
            return Err(MediaError::CaptureNotStarted);
        }
        let interval = Duration::from_nanos(1_000_000_000 / self.fps.max(1) as u64);
        // A still screen sends nothing, so the pipeline repeats the last
        // picture when the next one is due: one interval after the previous
        // frame went out. (Waiting 1.5 intervals from each call capped a
        // still or slowly changing screen at 2/3 of the rate: 40 of 60 fps.)
        let wait = if self.first_frame {
            Duration::from_secs(2)
        } else {
            (self.last_out + interval).saturating_duration_since(std::time::Instant::now())
        };
        let rx = self.frame_rx.as_mut().ok_or(MediaError::CaptureNotStarted)?;
        let received = tokio::time::timeout(wait, rx.recv()).await;
        let mut incoming = match received {
            Ok(Some(i)) => i,
            Ok(None) => return Err(MediaError::CaptureError("capture service connection ended".into())),
            Err(_) => {
                // Unchanged, off or paused: let the pipeline repeat the last
                // picture.
                self.last_out = std::time::Instant::now();
                return Ok(CapturedFrame {
                    buffer: GpuBuffer::CpuBuffer { data: Vec::new(), stride: 0, format: PixelFormat::BGRA, width: 0, height: 0 },
                    timestamp_us: 0,
                    width: 0,
                    height: 0,
                    format: PixelFormat::BGRA,
                    is_new_frame: false,
                    cursor: None,
                });
            }
        };
        // Newest picture wins; older ones are returned to the service.
        while let Ok(newer) = rx.try_recv() {
            self.shared.ack(incoming.seq);
            incoming = newer;
        }
        self.shared.ack(incoming.seq);
        self.first_frame = false;
        self.last_out = std::time::Instant::now();
        Ok(incoming.frame)
    }

    async fn set_fps(&mut self, fps: u32) -> Result<(), MediaError> {
        self.fps = fps.max(1);
        if let Some(d) = &self.streaming {
            let conn = self.shared.conn.lock().ok().and_then(|c| c.clone());
            if let Some(conn) = conn {
                let _ = conn.send(&start_msg(d, self.fps), &[]);
            }
        }
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), MediaError> {
        self.halt();
        log::info!("capture service capture stopped");
        Ok(())
    }

    fn is_capturing(&self) -> bool {
        self.capturing
    }

    fn delivers_dmabuf(&self) -> bool {
        true
    }

    fn paces_frames(&self) -> bool {
        true
    }
}

impl Drop for DrmServiceCapture {
    fn drop(&mut self) {
        self.halt();
    }
}

#[cfg(test)]
mod hotspot_tests {
    use super::*;

    /// The tests that set the last absolute move (one for the process).
    static POINTER: Mutex<()> = Mutex::new(());

    /// A 32x32 picture with an arrow-like tip at (4, 3).
    fn arrow() -> Vec<u8> {
        let mut rgba = vec![0u8; 32 * 32 * 4];
        for (x, y) in [(4, 3), (4, 4), (5, 4), (4, 5), (5, 5), (6, 5)] {
            rgba[(y * 32 + x) * 4 + 3] = 255;
        }
        rgba
    }

    fn report(x: i32, y: i32, image: Option<Vec<u8>>) -> WireCursor {
        WireCursor { display_id: 1, visible: true, x, y, hot_x: 0, hot_y: 0, width: 32, height: 32, generation: 1, image }
    }

    #[test]
    fn a_new_picture_is_aimed_at_its_first_opaque_pixel() {
        assert_eq!(first_opaque(32, 32, &arrow()), (4, 3));
        assert_eq!(first_opaque(2, 2, &[0; 16]), (0, 0));
        let mut h = Hotspots::default();
        let meta = h.meta(&report(100, 200, Some(arrow())), (1920, 1080), (1920, 1080));
        assert_eq!((meta.x, meta.y, meta.hotspot_x, meta.hotspot_y), (104, 203, 4, 3));
        assert_eq!(meta.image.unwrap().hotspot_x, 4);
        // A KMS that knows the hotspot (a virtual GPU) is believed.
        let known = WireCursor { hot_x: 7, hot_y: 9, ..report(100, 200, Some(arrow())) };
        let meta = h.meta(&known, (1920, 1080), (1920, 1080));
        assert_eq!((meta.x, meta.y), (107, 209));
    }

    #[test]
    fn the_pointer_inside_the_picture_is_its_hotspot() {
        // Pointer at pixel (960, 540) of 1920x1080, image corner (952, 532).
        let f = |p: f64, size: f64| (p + 0.5) / size;
        assert_eq!(hotspot_from(f(960.0, 1920.0), f(540.0, 1080.0), (1920, 1080), (952, 532), (32, 32)), Some((8, 8)));
        // Outside the picture: something else moved the pointer.
        assert_eq!(hotspot_from(f(960.0, 1920.0), f(540.0, 1080.0), (1920, 1080), (900, 532), (32, 32)), None);
        assert_eq!(hotspot_from(f(960.0, 1920.0), f(540.0, 1080.0), (1920, 1080), (961, 532), (32, 32)), None);
    }

    #[test]
    fn the_hotspot_is_learned_from_an_absolute_move() {
        let _one = POINTER.lock().unwrap_or_else(|e| e.into_inner());
        let mut h = Hotspots::default();
        let display = (1920, 1080);
        h.meta(&report(480, 480, Some(arrow())), display, display);
        // The viewer puts the pointer at (512, 510); the plane moves after it
        // (its corner at (500, 500)).
        crate::input::note_absolute_pointer_for_tests(Some(((512.0 + 0.5) / 1920.0, (510.0 + 0.5) / 1080.0)));
        std::thread::sleep(Duration::from_millis(2));
        let moved = h.meta(&report(500, 500, None), display, display);
        assert_eq!((moved.hotspot_x, moved.hotspot_y), (4, 3));
        // Not settled yet: nothing learned.
        assert!(h.settle(display, display).is_none());
        std::thread::sleep(Hotspots::SETTLED + Duration::from_millis(10));
        let learned = h.settle(display, display).expect("learned");
        assert_eq!((learned.x, learned.y, learned.hotspot_x, learned.hotspot_y), (512, 510, 12, 10));
        // The picture goes out again, with it.
        assert_eq!(learned.image.as_ref().map(|i| (i.hotspot_x, i.hotspot_y)), Some((12, 10)));
        // Each move is looked at once.
        assert!(h.settle(display, display).is_none());
        // The same picture later: what was learned.
        let later = h.meta(&report(10, 10, Some(arrow())), display, display);
        assert_eq!((later.hotspot_x, later.hotspot_y), (12, 10));
        crate::input::note_absolute_pointer_for_tests(None);
    }

    #[test]
    fn a_report_from_before_the_move_teaches_nothing() {
        let _one = POINTER.lock().unwrap_or_else(|e| e.into_inner());
        let mut h = Hotspots::default();
        h.meta(&report(100, 100, Some(arrow())), (1920, 1080), (1920, 1080));
        std::thread::sleep(Duration::from_millis(2));
        crate::input::note_absolute_pointer_for_tests(Some(((105.0 + 0.5) / 1920.0, (105.0 + 0.5) / 1080.0)));
        std::thread::sleep(Hotspots::SETTLED + Duration::from_millis(10));
        assert!(h.settle((1920, 1080), (1920, 1080)).is_none());
        crate::input::note_absolute_pointer_for_tests(None);
    }
}
