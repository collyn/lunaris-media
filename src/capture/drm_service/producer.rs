//! The capture service: owns DRM, serves frames to authorised clients.
//!
//! It never encodes, never touches the network and never writes to the
//! screen: per client it samples the framebuffer a CRTC scans out, exports it
//! as a read-only DMA-BUF and passes the descriptor over the Unix socket
//! (closing its own copy right after the send).
//!
//! One thread per client (at most [`ServiceConfig::max_clients`]); each one
//! paces itself with `poll` on its socket, so control messages, frames and
//! cursor updates need no locking.

use std::collections::VecDeque;
use std::io;
use std::os::fd::{AsRawFd, RawFd};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::auth::{self, Deny, PeerFacts, RefusalLog};
use super::kms::{self, Card, CursorPicture, CursorPlaneState, Grab};
use super::protocol::{
    ErrorCode, Message, StatusCode, WireCursor, WireDisplay, WireFrame, MAX_DISPLAYS, PROTOCOL_VERSION,
};
use super::transport::{Conn, Listener, TransportError};
use crate::error::MediaError;

/// Where frames come from (the real one is [`KmsSource`]; tests use fakes).
pub trait FrameSource {
    /// Currently lit displays.
    fn displays(&mut self) -> Vec<WireDisplay>;
    /// Sample `display` now.
    fn grab(&mut self, display: &WireDisplay) -> io::Result<Grab>;
    /// Hardware cursor on `display`, if one is shown there.
    fn cursor_state(&mut self, display: &WireDisplay) -> Option<CursorPlaneState>;
    /// Picture of cursor framebuffer `fb_id`.
    fn cursor_picture(&mut self, display: &WireDisplay, fb_id: u32) -> Option<CursorPicture>;
}

/// Decides who may connect and keep streaming.
pub trait Authorizer: Send + Sync {
    fn admit(&self, conn: &Conn) -> Result<PeerFacts, Deny>;
    fn recheck(&self, peer: &mut PeerFacts) -> Result<(), Deny>;
}

/// Production authorizer (see [`super::auth`]).
pub struct SystemAuthorizer {
    pub allow_exes: Vec<PathBuf>,
    pub allow_appimages: Vec<PathBuf>,
    refusals: Mutex<RefusalLog>,
}

impl SystemAuthorizer {
    pub fn new(allow_exes: Vec<PathBuf>, allow_appimages: Vec<PathBuf>) -> Self {
        SystemAuthorizer { allow_exes, allow_appimages, refusals: Mutex::new(RefusalLog::new()) }
    }
}

const ACTIVE_UID_TTL: Duration = Duration::from_millis(500);

impl Authorizer for SystemAuthorizer {
    fn admit(&self, conn: &Conn) -> Result<PeerFacts, Deny> {
        let cred = conn.peer_cred().map_err(|_| Deny::ProcessGone)?;
        let pidfd = conn.peer_pidfd();
        let facts = match auth::gather_checked(cred.pid, cred.uid, pidfd.as_ref()) {
            Ok(f) => f,
            Err(d) => {
                let f = PeerFacts { pid: cred.pid, uid: cred.uid, ..Default::default() };
                self.refusals.lock().unwrap_or_else(|e| e.into_inner()).log(&f, &d);
                return Err(d);
            }
        };
        // Resolved per connection: installing/removing a package is seen
        // without restarting the service.
        let policy = auth::Policy::resolve(&self.allow_exes, &self.allow_appimages, true);
        match auth::authorize(&facts, auth::active_uid_cached(ACTIVE_UID_TTL), &policy) {
            Ok(()) => {
                log::info!(
                    "capture service: accepted pid {} uid {} ({})",
                    facts.pid,
                    facts.uid,
                    facts.exe_path.as_deref().unwrap_or("?")
                );
                Ok(facts)
            }
            Err(d) => {
                self.refusals.lock().unwrap_or_else(|e| e.into_inner()).log(&facts, &d);
                Err(d)
            }
        }
    }

    fn recheck(&self, peer: &mut PeerFacts) -> Result<(), Deny> {
        auth::refresh_traced(peer);
        auth::still_allowed(peer, auth::active_uid_cached(ACTIVE_UID_TTL))
    }
}

/// Real frames from every `/dev/dri/card*`.
pub struct KmsSource {
    dev_dir: PathBuf,
    cards: Vec<Card>,
    warned_export: bool,
}

impl KmsSource {
    pub fn new() -> Self {
        Self::with_dev_dir(PathBuf::from("/dev/dri"))
    }

    pub fn with_dev_dir(dev_dir: PathBuf) -> Self {
        let mut s = KmsSource { dev_dir, cards: Vec::new(), warned_export: false };
        s.refresh_cards();
        s
    }

    fn refresh_cards(&mut self) {
        let paths = kms::card_paths(&self.dev_dir);
        let current: Vec<PathBuf> = self.cards.iter().map(|c| c.path.clone()).collect();
        if paths == current {
            return;
        }
        self.cards = paths
            .iter()
            .filter_map(|p| match Card::open(p) {
                Ok(c) => Some(c),
                Err(e) => {
                    log::debug!("capture service: cannot open {}: {}", p.display(), e);
                    None
                }
            })
            .collect();
    }

    fn card_for(&self, display: &WireDisplay) -> Option<&Card> {
        self.cards.iter().find(|c| c.name == display.card)
    }

    pub fn cards(&self) -> &[Card] {
        &self.cards
    }
}

impl Default for KmsSource {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameSource for KmsSource {
    fn displays(&mut self) -> Vec<WireDisplay> {
        self.refresh_cards();
        let mut all = Vec::new();
        for card in &self.cards {
            match kms::card_displays(card) {
                Ok(list) => all.extend(list),
                Err(e) => log::debug!("capture service: {}: {}", card.name, e),
            }
        }
        all.truncate(MAX_DISPLAYS);
        all
    }

    fn grab(&mut self, display: &WireDisplay) -> io::Result<Grab> {
        let Some(card) = self.card_for(display) else {
            return Ok(Grab::Gone);
        };
        let res = kms::grab(card, display.crtc_id);
        if let Err(e) = &res {
            if e.kind() == io::ErrorKind::PermissionDenied && !self.warned_export {
                self.warned_export = true;
                log::error!("capture service: {} (is the service running as root?)", e);
            }
        }
        res
    }

    fn cursor_state(&mut self, display: &WireDisplay) -> Option<CursorPlaneState> {
        self.card_for(display)?.cursor_on(display.crtc_id)
    }

    fn cursor_picture(&mut self, display: &WireDisplay, fb_id: u32) -> Option<CursorPicture> {
        self.card_for(display)?.read_cursor(fb_id)
    }
}

/// Timing knobs (tests shorten them).
#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub hello_timeout: Duration,
    pub send_timeout: Duration,
    pub reauth_interval: Duration,
    pub rescan_interval: Duration,
    pub cursor_reread: Duration,
    /// Frames not acknowledged within this time stop counting against the
    /// client's credit (recovers from a client that lost track).
    pub ack_timeout: Duration,
}

impl Default for SessionOptions {
    fn default() -> Self {
        SessionOptions {
            hello_timeout: Duration::from_secs(5),
            send_timeout: Duration::from_secs(2),
            reauth_interval: Duration::from_secs(1),
            rescan_interval: Duration::from_secs(2),
            cursor_reread: Duration::from_millis(500),
            ack_timeout: Duration::from_secs(3),
        }
    }
}

fn monotonic_us() -> u64 {
    // SAFETY: timespec is plain old data; clock_gettime fills it.
    let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
    // SAFETY: valid clock id and out-parameter.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000 + ts.tv_nsec as u64 / 1000
}

#[derive(Default)]
struct CursorTrack {
    fb_id: u32,
    last_read: Option<Instant>,
    picture: Option<CursorPicture>,
    generation: u32,
    image_pending: bool,
    last_sent: Option<(bool, i32, i32, i32, i32, u32)>,
}

struct Stream {
    display: WireDisplay,
    interval: Duration,
    next_due: Instant,
    max_in_flight: usize,
    in_flight: VecDeque<(u64, Instant)>,
    seq: u64,
    status: Option<StatusCode>,
    cursor: CursorTrack,
}

/// Why a session ended (tests look at it).
#[derive(Debug, PartialEq, Eq)]
pub enum SessionEnd {
    Refused,
    ClientClosed,
    ProtocolError,
    Stopped,
    SendFailed,
}

fn send(conn: &Conn, msg: &Message, fds: &[RawFd]) -> Result<(), SessionEnd> {
    conn.send(msg, fds).map_err(|e| {
        if !matches!(e, TransportError::Closed) {
            log::debug!("capture service: send failed: {}", e);
        }
        SessionEnd::SendFailed
    })
}

/// Serve one connected client until it leaves, misbehaves or `stop` is set.
pub fn run_session(
    conn: &Conn,
    source: &mut dyn FrameSource,
    authorizer: &dyn Authorizer,
    opts: &SessionOptions,
    stop: &AtomicBool,
) -> SessionEnd {
    let mut peer = match authorizer.admit(conn) {
        Ok(p) => p,
        Err(deny) => {
            let _ = conn.send(
                &Message::Error { code: ErrorCode::Unauthorized, message: deny.to_string() },
                &[],
            );
            conn.drain(Duration::from_millis(300));
            return SessionEnd::Refused;
        }
    };
    let _ = conn.set_send_timeout(opts.send_timeout);
    match conn.recv(Some(opts.hello_timeout)) {
        Ok(Some((Message::Hello { version }, _))) if version == PROTOCOL_VERSION => {}
        Ok(Some((Message::Hello { version }, _))) => {
            let _ = conn.send(
                &Message::Error {
                    code: ErrorCode::BadVersion,
                    message: format!("service speaks version {}, client {}", PROTOCOL_VERSION, version),
                },
                &[],
            );
            conn.drain(Duration::from_millis(100));
            return SessionEnd::ProtocolError;
        }
        _ => return SessionEnd::ProtocolError,
    }
    if let Err(e) = session_loop(conn, source, authorizer, opts, stop, &mut peer) {
        return e;
    }
    SessionEnd::Stopped
}

fn session_loop(
    conn: &Conn,
    source: &mut dyn FrameSource,
    authorizer: &dyn Authorizer,
    opts: &SessionOptions,
    stop: &AtomicBool,
    peer: &mut PeerFacts,
) -> Result<(), SessionEnd> {
    send(conn, &Message::Hello { version: PROTOCOL_VERSION }, &[])?;
    let mut displays = source.displays();
    send(conn, &Message::DisplayList(displays.clone()), &[])?;

    let mut stream: Option<Stream> = None;
    let mut paused = false;
    let mut next_reauth = Instant::now() + opts.reauth_interval;
    let mut next_rescan = Instant::now() + opts.rescan_interval;
    let mut grab_errors = 0u32;

    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let now = Instant::now();
        let mut wake = next_rescan.min(next_reauth);
        if let Some(s) = &stream {
            wake = wake.min(s.next_due);
        }
        // Never sleep long, so `stop` is seen promptly.
        let timeout = wake.saturating_duration_since(now).min(Duration::from_millis(250));
        match conn.recv(Some(timeout)) {
            Ok(Some((msg, _fds))) => match msg {
                Message::ListDisplays => {
                    displays = source.displays();
                    send(conn, &Message::DisplayList(displays.clone()), &[])?;
                }
                Message::Start { display_id, max_fps, max_in_flight } => {
                    displays = source.displays();
                    match displays.iter().find(|d| d.id == display_id) {
                        Some(d) => {
                            stream = Some(Stream {
                                display: d.clone(),
                                interval: Duration::from_nanos(1_000_000_000 / max_fps as u64),
                                next_due: Instant::now(),
                                max_in_flight: max_in_flight as usize,
                                in_flight: VecDeque::new(),
                                seq: 0,
                                status: None,
                                cursor: CursorTrack::default(),
                            });
                            log::info!(
                                "capture service: pid {} streams {} ({}) at up to {} fps",
                                peer.pid,
                                d.name,
                                d.card,
                                max_fps
                            );
                        }
                        None => {
                            send(
                                conn,
                                &Message::Error {
                                    code: ErrorCode::NoSuchDisplay,
                                    message: format!("no display {:#x}", display_id),
                                },
                                &[],
                            )?;
                        }
                    }
                }
                Message::Stop => stream = None,
                Message::FrameAck { seq } => {
                    if let Some(s) = &mut stream {
                        s.in_flight.retain(|(q, _)| *q != seq);
                    }
                }
                _ => {
                    let _ = conn.send(
                        &Message::Error { code: ErrorCode::BadMessage, message: "unexpected message".into() },
                        &[],
                    );
                    return Err(SessionEnd::ProtocolError);
                }
            },
            Ok(None) => {}
            Err(TransportError::Closed) => return Err(SessionEnd::ClientClosed),
            Err(TransportError::Protocol(e)) => {
                log::debug!("capture service: pid {}: {}", peer.pid, e);
                let _ = conn.send(&Message::Error { code: ErrorCode::BadMessage, message: e.to_string() }, &[]);
                return Err(SessionEnd::ProtocolError);
            }
            Err(_) => return Err(SessionEnd::ClientClosed),
        }

        let now = Instant::now();
        if now >= next_reauth {
            next_reauth = now + opts.reauth_interval;
            match authorizer.recheck(peer) {
                Ok(()) if paused => {
                    paused = false;
                    log::info!("capture service: pid {} may stream again", peer.pid);
                    if let Some(s) = &mut stream {
                        s.status = None;
                        s.cursor.last_sent = None;
                        s.cursor.image_pending = s.cursor.picture.is_some();
                        send(conn, &Message::Status { display_id: s.display.id, code: StatusCode::Active }, &[])?;
                    }
                }
                Ok(()) => {}
                Err(deny) if !paused => {
                    paused = true;
                    log::info!("capture service: pausing pid {}: {}", peer.pid, deny);
                    let id = stream.as_ref().map(|s| s.display.id).unwrap_or(0);
                    send(conn, &Message::Status { display_id: id, code: StatusCode::Paused }, &[])?;
                }
                Err(_) => {}
            }
        }

        if now >= next_rescan {
            next_rescan = now + opts.rescan_interval;
            let fresh = source.displays();
            if fresh != displays {
                displays = fresh;
                send(conn, &Message::DisplaysChanged(displays.clone()), &[])?;
                if let Some(s) = &mut stream {
                    match displays.iter().find(|d| d.id == s.display.id) {
                        Some(d) => s.display = d.clone(),
                        None => {
                            send(conn, &Message::Status { display_id: s.display.id, code: StatusCode::Gone }, &[])?;
                            stream = None;
                        }
                    }
                }
            }
        }

        let Some(s) = &mut stream else { continue };
        if now < s.next_due {
            continue;
        }
        s.next_due += s.interval;
        if s.next_due < now {
            // Fell behind: resynchronise instead of bursting.
            s.next_due = now + s.interval;
        }
        if paused {
            continue;
        }
        // Credits: frames the client has not acknowledged yet.
        s.in_flight.retain(|(_, at)| now.duration_since(*at) < opts.ack_timeout);
        if s.in_flight.len() < s.max_in_flight {
            match source.grab(&s.display) {
                Ok(Grab::Frame { fb, crop }) => {
                    grab_errors = 0;
                    s.seq += 1;
                    let frame = WireFrame {
                        seq: s.seq,
                        display_id: s.display.id,
                        timestamp_us: monotonic_us(),
                        width: fb.layout.width,
                        height: fb.layout.height,
                        fourcc: fb.layout.fourcc,
                        modifier: fb.layout.modifier,
                        buffer_id: fb.buffer_id,
                        crop_x: crop.0,
                        crop_y: crop.1,
                        crop_w: crop.2,
                        crop_h: crop.3,
                        planes: fb.planes.clone(),
                    };
                    let raw: Vec<RawFd> = fb.fds.iter().map(|f| f.as_raw_fd()).collect();
                    if s.status.is_some_and(|st| st != StatusCode::Active) {
                        send(conn, &Message::Status { display_id: s.display.id, code: StatusCode::Active }, &[])?;
                    }
                    s.status = Some(StatusCode::Active);
                    let sent = send(conn, &Message::Frame(frame), &raw);
                    // Our copies of the descriptors close here, sent or not.
                    drop(fb);
                    sent?;
                    s.in_flight.push_back((s.seq, now));
                }
                Ok(Grab::NoFramebuffer) => {
                    if s.status != Some(StatusCode::NoFramebuffer) {
                        s.status = Some(StatusCode::NoFramebuffer);
                        send(conn, &Message::Status { display_id: s.display.id, code: StatusCode::NoFramebuffer }, &[])?;
                    }
                }
                Ok(Grab::Gone) => {
                    send(conn, &Message::Status { display_id: s.display.id, code: StatusCode::Gone }, &[])?;
                    stream = None;
                    next_rescan = now;
                    continue;
                }
                Err(e) => {
                    grab_errors += 1;
                    if grab_errors == 1 || grab_errors.is_multiple_of(600) {
                        log::warn!("capture service: grab {} failed ({}x): {}", s.display.name, grab_errors, e);
                    }
                }
            }
        }
        let display = s.display.clone();
        update_cursor(conn, source, &display, &mut s.cursor, opts, now)?;
    }
}

fn update_cursor(
    conn: &Conn,
    source: &mut dyn FrameSource,
    display: &WireDisplay,
    track: &mut CursorTrack,
    opts: &SessionOptions,
    now: Instant,
) -> Result<(), SessionEnd> {
    let state = source.cursor_state(display);
    if let Some(st) = &state {
        let stale = track.last_read.is_none_or(|t| now.duration_since(t) >= opts.cursor_reread);
        if st.fb_id != track.fb_id || stale {
            let picture = source.cursor_picture(display, st.fb_id);
            track.fb_id = st.fb_id;
            track.last_read = Some(now);
            if picture != track.picture {
                track.picture = picture;
                track.generation = track.generation.wrapping_add(1);
                track.image_pending = true;
            }
        }
    }
    let empty = track.picture.as_ref().is_some_and(|p| p.width == 0);
    let visible = state.is_some() && !empty;
    let (x, y, hx, hy) = state.as_ref().map_or((0, 0, 0, 0), |s| (s.x, s.y, s.hot_x, s.hot_y));
    let key = (visible, x, y, hx, hy, track.generation);
    if track.last_sent == Some(key) {
        return Ok(());
    }
    let (w, h, image) = match (&track.picture, track.image_pending && visible) {
        (Some(p), true) if p.width > 0 => (p.width, p.height, Some(p.rgba.clone())),
        (Some(p), _) => (p.width, p.height, None),
        (None, _) => (0, 0, None),
    };
    let msg = Message::Cursor(WireCursor {
        display_id: display.id,
        visible,
        x,
        y,
        hot_x: hx,
        hot_y: hy,
        width: w,
        height: h,
        generation: track.generation,
        image: image.clone(),
    });
    send(conn, &msg, &[])?;
    if image.is_some() {
        track.image_pending = false;
    }
    track.last_sent = Some(key);
    Ok(())
}

/// Service configuration.
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub socket_path: PathBuf,
    pub allow_exes: Vec<PathBuf>,
    pub allow_appimages: Vec<PathBuf>,
    pub max_clients: usize,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        ServiceConfig {
            socket_path: PathBuf::from(super::DEFAULT_SOCKET_PATH),
            allow_exes: Vec::new(),
            allow_appimages: Vec::new(),
            max_clients: 4,
        }
    }
}

/// Accept clients on `listener` until `stop` is set.
pub fn serve_with<S, F>(
    listener: &Listener,
    make_source: F,
    authorizer: Arc<dyn Authorizer>,
    opts: SessionOptions,
    max_clients: usize,
    stop: Arc<AtomicBool>,
) -> io::Result<()>
where
    S: FrameSource + 'static,
    F: Fn() -> S + Send + Sync + 'static,
{
    let make_source = Arc::new(make_source);
    let active = Arc::new(AtomicUsize::new(0));
    let mut workers: Vec<std::thread::JoinHandle<()>> = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        workers.retain(|h| !h.is_finished());
        let Some(conn) = listener.accept(Some(Duration::from_millis(250)))? else { continue };
        if active.load(Ordering::SeqCst) >= max_clients {
            let _ = conn.set_send_timeout(Duration::from_millis(200));
            let _ = conn.send(&Message::Error { code: ErrorCode::Busy, message: "too many clients".into() }, &[]);
            conn.drain(Duration::from_millis(100));
            continue;
        }
        active.fetch_add(1, Ordering::SeqCst);
        let (make_source, authorizer, opts, stop, active) =
            (make_source.clone(), authorizer.clone(), opts.clone(), stop.clone(), active.clone());
        let spawned = std::thread::Builder::new().name("lunaris-capsvc".into()).spawn(move || {
            let mut source = make_source();
            let end = run_session(&conn, &mut source, authorizer.as_ref(), &opts, &stop);
            log::debug!("capture service: session ended: {:?}", end);
            active.fetch_sub(1, Ordering::SeqCst);
        });
        match spawned {
            Ok(h) => workers.push(h),
            Err(e) => log::warn!("capture service: cannot start a client thread: {}", e),
        }
    }
    for h in workers {
        let _ = h.join();
    }
    Ok(())
}

static SIGNALLED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    SIGNALLED.store(true, Ordering::SeqCst);
}

/// Run the service in this process until SIGTERM/SIGINT (what
/// `lunaris-agent --capture-service` does).
pub fn serve(cfg: ServiceConfig) -> Result<(), MediaError> {
    // SAFETY: geteuid has no preconditions.
    if unsafe { libc::geteuid() } != 0 {
        log::warn!("capture service: not running as root; framebuffers cannot be exported");
    }
    if let Some(dir) = cfg.socket_path.parent() {
        if !dir.exists() {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new()
                .mode(0o755)
                .recursive(true)
                .create(dir)
                .map_err(|e| MediaError::CaptureError(format!("{}: {}", dir.display(), e)))?;
        }
    }
    let listener = Listener::bind(&cfg.socket_path, 0o666)
        .map_err(|e| MediaError::CaptureError(format!("{}: {}", cfg.socket_path.display(), e)))?;

    // SAFETY: installing an async-signal-safe handler (an atomic store).
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = on_signal as extern "C" fn(libc::c_int) as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(libc::SIGTERM, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
    }
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        std::thread::Builder::new()
            .name("lunaris-capsvc-sig".into())
            .spawn(move || loop {
                if SIGNALLED.load(Ordering::SeqCst) {
                    stop.store(true, Ordering::SeqCst);
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            })
            .map_err(|e| MediaError::CaptureError(e.to_string()))?;
    }

    let mut probe = KmsSource::new();
    let displays = probe.displays();
    log::info!(
        "capture service: listening on {} ({} card(s), {} display(s))",
        cfg.socket_path.display(),
        probe.cards().len(),
        displays.len()
    );
    for d in &displays {
        log::info!(
            "capture service:   {} on {} crtc {} {}x{} @{:.2}Hz{}",
            d.name,
            d.card,
            d.crtc_id,
            d.width,
            d.height,
            d.refresh_mhz as f64 / 1000.0,
            if d.flags & super::protocol::DISPLAY_ACTIVE != 0 { "" } else { " (off)" }
        );
    }
    drop(probe);

    let authorizer: Arc<dyn Authorizer> =
        Arc::new(SystemAuthorizer::new(cfg.allow_exes.clone(), cfg.allow_appimages.clone()));
    serve_with(&listener, KmsSource::new, authorizer, SessionOptions::default(), cfg.max_clients.max(1), stop)
        .map_err(|e| MediaError::CaptureError(e.to_string()))?;
    log::info!("capture service: stopped");
    Ok(())
}
