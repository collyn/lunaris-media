//! End-to-end tests of the capture service without DRM or root: a fake
//! frame source hands out memfds as stand-in DMA-BUFs, a fake authorizer
//! plays logind, and the real session loop, transport and
//! [`DrmServiceCapture`] run on top.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::auth::{Deny, PeerFacts};
use super::client::{ClientError, DrmServiceCapture, ServiceClient};
use super::kms::{CursorPicture, CursorPlaneState, ExportedFb, FbLayout, Grab};
use super::producer::{run_session, serve_with, Authorizer, FrameSource, SessionEnd, SessionOptions};
use super::protocol::{ErrorCode, Message, StatusCode, WireDisplay, WirePlane, DISPLAY_ACTIVE, PROTOCOL_VERSION};
use super::transport::{open_fds_matching, Conn, Listener};
use crate::capture::gpu_buffer::GpuBuffer;
use crate::capture::ScreenCapture;
use crate::types::StreamConfig;

const XR24: u32 = u32::from_le_bytes(*b"XR24");

fn display(id: u64, name: &str) -> WireDisplay {
    WireDisplay {
        id,
        card: "card7".into(),
        connector_id: id as u32,
        crtc_id: 80 + id as u32,
        name: name.into(),
        x: 0,
        y: 0,
        width: 64,
        height: 32,
        refresh_mhz: 60_000,
        flags: DISPLAY_ACTIVE,
    }
}

#[derive(Clone)]
enum Screen {
    On,
    Off,
    Gone,
}

struct FakeState {
    displays: Vec<WireDisplay>,
    screen: Screen,
    cursor: Option<CursorPlaneState>,
    cursor_reads: u32,
    /// How long one cursor picture read takes (VRAM reads are slow).
    cursor_read_delay: Duration,
    /// Vertical blank period (`None`: no vblank, the session paces by clock).
    vblank: Option<Duration>,
    /// Framebuffer id every grab reports (`None`: 0, a new buffer each grab).
    fixed_fb: Option<u32>,
    tag: String,
}

#[derive(Clone)]
struct FakeSource(Arc<Mutex<FakeState>>);

impl FakeSource {
    fn new(tag: &str) -> Self {
        FakeSource(Arc::new(Mutex::new(FakeState {
            displays: vec![display(1, "DP-1"), display(2, "HDMI-A-1")],
            screen: Screen::On,
            cursor: Some(CursorPlaneState { plane_id: 9, fb_id: 500, x: 10, y: 5, hot_x: 0, hot_y: 0 }),
            cursor_reads: 0,
            cursor_read_delay: Duration::ZERO,
            vblank: None,
            fixed_fb: None,
            tag: tag.into(),
        })))
    }
    fn set<F: FnOnce(&mut FakeState)>(&self, f: F) {
        f(&mut self.0.lock().unwrap());
    }
}

fn memfd(name: &str, len: usize) -> OwnedFd {
    let cname = std::ffi::CString::new(name).unwrap();
    // SAFETY: memfd_create with a valid name.
    let fd = unsafe { libc::memfd_create(cname.as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0);
    // SAFETY: fresh descriptor.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    // SAFETY: sizing our memfd.
    unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) };
    fd
}

impl FrameSource for FakeSource {
    fn displays(&mut self) -> Vec<WireDisplay> {
        self.0.lock().unwrap().displays.clone()
    }
    fn grab(&mut self, d: &WireDisplay) -> std::io::Result<Grab> {
        let st = self.0.lock().unwrap();
        Ok(match st.screen {
            Screen::Off => Grab::NoFramebuffer,
            Screen::Gone => Grab::Gone,
            Screen::On => {
                let stride = d.width * 4;
                let fd = memfd(&format!("{}-fb", st.tag), (stride * d.height) as usize);
                Grab::Frame {
                    fb: ExportedFb {
                        layout: FbLayout {
                            width: d.width,
                            height: d.height,
                            fourcc: XR24,
                            modifier: 0,
                            planes: vec![(0, stride)],
                            exportable: true,
                        },
                        buffer_id: super::kms::fd_inode(&fd),
                        fb_id: st.fixed_fb.unwrap_or(0),
                        fds: vec![fd],
                        planes: vec![WirePlane { fd_index: 0, offset: 0, stride }],
                    },
                    crop: (0, 0, d.width, d.height),
                }
            }
        })
    }
    fn wait_vblank(&mut self, _d: &WireDisplay) -> std::io::Result<()> {
        let period = self.0.lock().unwrap().vblank;
        match period {
            Some(p) => {
                std::thread::sleep(p);
                Ok(())
            }
            None => Err(std::io::Error::from(std::io::ErrorKind::Unsupported)),
        }
    }
    fn cursor_state(&mut self, _d: &WireDisplay) -> Option<CursorPlaneState> {
        self.0.lock().unwrap().cursor.clone()
    }
    fn cursor_reader(&mut self, _d: &WireDisplay, fb_id: u32) -> super::producer::CursorJob {
        let delay = {
            let mut st = self.0.lock().unwrap();
            st.cursor_reads += 1;
            st.cursor_read_delay
        };
        // Picture depends on the framebuffer id only.
        Box::new(move || {
            std::thread::sleep(delay);
            Some(CursorPicture { width: 2, height: 1, rgba: vec![fb_id as u8, 0, 0, 255, 0, 0, 0, 128] })
        })
    }
}

struct FakeAuth {
    admit: AtomicBool,
    active: AtomicBool,
}

impl FakeAuth {
    fn new() -> Arc<Self> {
        Arc::new(FakeAuth { admit: AtomicBool::new(true), active: AtomicBool::new(true) })
    }
}

impl Authorizer for FakeAuth {
    fn admit(&self, conn: &Conn) -> Result<PeerFacts, Deny> {
        let cred = conn.peer_cred().map_err(|_| Deny::ProcessGone)?;
        if self.admit.load(Ordering::SeqCst) {
            Ok(PeerFacts { pid: cred.pid, uid: cred.uid, ..Default::default() })
        } else {
            Err(Deny::ExeNotAllowed)
        }
    }
    fn recheck(&self, _peer: &mut PeerFacts) -> Result<(), Deny> {
        if self.active.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err(Deny::NotActiveUser { uid: 1000, active: 1001 })
        }
    }
}

fn fast_opts() -> SessionOptions {
    SessionOptions {
        hello_timeout: Duration::from_secs(2),
        send_timeout: Duration::from_secs(1),
        reauth_interval: Duration::from_millis(50),
        rescan_interval: Duration::from_millis(50),
        cursor_reread: Duration::from_secs(60),
        ack_timeout: Duration::from_secs(60),
    }
}

/// A session on one end of a socketpair, running in a thread.
struct Harness {
    client: Conn,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<SessionEnd>>,
}

impl Harness {
    fn new(source: FakeSource, auth: Arc<FakeAuth>) -> Harness {
        let (client, server) = Conn::pair().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let thread = std::thread::spawn(move || {
            let mut source = source;
            run_session(&server, &mut source, auth.as_ref(), &fast_opts(), &stop2)
        });
        Harness { client, stop, thread: Some(thread) }
    }

    fn recv(&self) -> (Message, Vec<OwnedFd>) {
        self.client.recv(Some(Duration::from_secs(2))).unwrap().expect("message expected")
    }

    /// Next message matching `pred`, skipping (and closing) others.
    fn wait_for<F: Fn(&Message) -> bool>(&self, pred: F) -> (Message, Vec<OwnedFd>) {
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if let Some((m, fds)) = self.client.recv(Some(Duration::from_millis(200))).unwrap() {
                if pred(&m) {
                    return (m, fds);
                }
                if let Message::Frame(f) = &m {
                    self.client.send(&Message::FrameAck { seq: f.seq }, &[]).unwrap();
                }
            }
        }
        panic!("message not received in time");
    }

    fn handshake(&self) -> Vec<WireDisplay> {
        self.client.send(&Message::Hello { version: PROTOCOL_VERSION }, &[]).unwrap();
        assert_eq!(self.recv().0, Message::Hello { version: PROTOCOL_VERSION });
        match self.recv().0 {
            Message::DisplayList(l) => l,
            other => panic!("{:?}", other),
        }
    }

    fn finish(mut self) -> SessionEnd {
        self.stop.store(true, Ordering::SeqCst);
        self.thread.take().unwrap().join().unwrap()
    }
}

#[test]
fn handshake_frames_and_fd_hygiene() {
    let source = FakeSource::new("e2e-hyg");
    let h = Harness::new(source, FakeAuth::new());
    let list = h.handshake();
    assert_eq!(list.len(), 2);
    h.client.send(&Message::Start { display_id: 2, max_fps: 120, max_in_flight: 2 }, &[]).unwrap();
    let mut frames = 0;
    let mut last_seq = 0;
    while frames < 20 {
        let (m, fds) = h.wait_for(|m| matches!(m, Message::Frame(_)));
        let Message::Frame(f) = m else { unreachable!() };
        assert_eq!(f.display_id, 2);
        assert_eq!(fds.len(), 1);
        assert!(f.seq > last_seq);
        last_seq = f.seq;
        // The memfd behind the frame is the right size.
        // SAFETY: fstat on a received descriptor.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        unsafe { libc::fstat(fds[0].as_raw_fd(), &mut st) };
        assert_eq!(st.st_size as u32, 64 * 4 * 32);
        drop(fds);
        h.client.send(&Message::FrameAck { seq: f.seq }, &[]).unwrap();
        frames += 1;
    }
    h.client.send(&Message::Stop, &[]).unwrap();
    assert_eq!(h.finish(), SessionEnd::Stopped);
    // Neither side keeps a frame descriptor once done.
    assert_eq!(open_fds_matching("e2e-hyg-fb"), 0, "descriptor leak");
}

#[test]
fn credits_limit_frames_in_flight() {
    let h = Harness::new(FakeSource::new("e2e-credit"), FakeAuth::new());
    h.handshake();
    h.client.send(&Message::Start { display_id: 1, max_fps: 200, max_in_flight: 2 }, &[]).unwrap();
    let mut held = Vec::new();
    let deadline = Instant::now() + Duration::from_millis(400);
    while Instant::now() < deadline {
        if let Some((Message::Frame(f), fds)) = h.client.recv(Some(Duration::from_millis(50))).unwrap() {
            held.push((f.seq, fds));
        }
    }
    // Without acknowledgements the service stops after two frames.
    assert_eq!(held.len(), 2, "frames in flight: {}", held.len());
    // Returning one credit lets exactly one more through.
    h.client.send(&Message::FrameAck { seq: held[0].0 }, &[]).unwrap();
    let (m, _fds) = h.wait_for(|m| matches!(m, Message::Frame(_)));
    let Message::Frame(f) = m else { unreachable!() };
    assert_eq!(f.seq, held[1].0 + 1);
    assert!(h.client.recv(Some(Duration::from_millis(150))).unwrap().map_or(true, |(m, _)| !matches!(m, Message::Frame(_))));
    drop(held);
    h.finish();
}

#[test]
fn unauthorized_peer_gets_an_error_and_no_frames() {
    let auth = FakeAuth::new();
    auth.admit.store(false, Ordering::SeqCst);
    let h = Harness::new(FakeSource::new("e2e-unauth"), auth);
    match h.recv().0 {
        Message::Error { code: ErrorCode::Unauthorized, message } => assert!(message.contains("not an allowed")),
        other => panic!("{:?}", other),
    }
    assert_eq!(h.finish(), SessionEnd::Refused);

    // The client side reports it as a refusal.
    let auth = FakeAuth::new();
    auth.admit.store(false, Ordering::SeqCst);
    let (client, server) = Conn::pair().unwrap();
    let t = std::thread::spawn(move || {
        let mut s = FakeSource::new("e2e-unauth2");
        run_session(&server, &mut s, auth.as_ref(), &fast_opts(), &AtomicBool::new(false))
    });
    assert!(matches!(ServiceClient::handshake(client, Duration::from_secs(2)), Err(ClientError::Refused(_))));
    t.join().unwrap();
}

#[test]
fn wrong_version_and_garbage_are_refused() {
    let h = Harness::new(FakeSource::new("e2e-ver"), FakeAuth::new());
    h.client.send(&Message::Hello { version: PROTOCOL_VERSION + 1 }, &[]).unwrap();
    assert!(matches!(h.recv().0, Message::Error { code: ErrorCode::BadVersion, .. }));
    assert_eq!(h.finish(), SessionEnd::ProtocolError);

    let h = Harness::new(FakeSource::new("e2e-garbage"), FakeAuth::new());
    h.handshake();
    h.client.send_raw(b"\x10\x00\x00\x00\x04\x00\x00\x00garbage!", &[]).unwrap();
    assert!(matches!(h.recv().0, Message::Error { code: ErrorCode::BadMessage, .. }));
    assert_eq!(h.finish(), SessionEnd::ProtocolError);

    // A client must not send service messages.
    let h = Harness::new(FakeSource::new("e2e-role"), FakeAuth::new());
    h.handshake();
    h.client.send(&Message::DisplayList(vec![]), &[]).unwrap();
    assert!(matches!(h.recv().0, Message::Error { code: ErrorCode::BadMessage, .. }));
    assert_eq!(h.finish(), SessionEnd::ProtocolError);
}

#[test]
fn unknown_display_is_an_error_not_a_disconnect() {
    let h = Harness::new(FakeSource::new("e2e-nodisp"), FakeAuth::new());
    h.handshake();
    h.client.send(&Message::Start { display_id: 77, max_fps: 30, max_in_flight: 1 }, &[]).unwrap();
    assert!(matches!(h.recv().0, Message::Error { code: ErrorCode::NoSuchDisplay, .. }));
    h.client.send(&Message::ListDisplays, &[]).unwrap();
    assert!(matches!(h.recv().0, Message::DisplayList(l) if l.len() == 2));
    h.finish();
}

#[test]
fn session_switch_pauses_and_resumes() {
    let auth = FakeAuth::new();
    let h = Harness::new(FakeSource::new("e2e-pause"), auth.clone());
    h.handshake();
    h.client.send(&Message::Start { display_id: 1, max_fps: 100, max_in_flight: 4 }, &[]).unwrap();
    h.wait_for(|m| matches!(m, Message::Frame(_)));
    auth.active.store(false, Ordering::SeqCst);
    h.wait_for(|m| matches!(m, Message::Status { code: StatusCode::Paused, .. }));
    // Paused: nothing but possibly frames already queued before the switch.
    let deadline = Instant::now() + Duration::from_millis(300);
    while Instant::now() < deadline {
        if let Some((m, _)) = h.client.recv(Some(Duration::from_millis(50))).unwrap() {
            assert!(!matches!(m, Message::Frame(_) | Message::Cursor(_)), "sent while paused: {:?}", m.kind());
        }
    }
    auth.active.store(true, Ordering::SeqCst);
    h.wait_for(|m| matches!(m, Message::Status { code: StatusCode::Active, .. }));
    h.wait_for(|m| matches!(m, Message::Frame(_)));
    h.finish();
}

#[test]
fn screen_off_and_display_changes_are_reported() {
    let source = FakeSource::new("e2e-off");
    let h = Harness::new(source.clone(), FakeAuth::new());
    h.handshake();
    h.client.send(&Message::Start { display_id: 1, max_fps: 100, max_in_flight: 4 }, &[]).unwrap();
    h.wait_for(|m| matches!(m, Message::Frame(_)));
    source.set(|s| s.screen = Screen::Off);
    h.wait_for(|m| matches!(m, Message::Status { code: StatusCode::NoFramebuffer, display_id: 1 }));
    source.set(|s| s.screen = Screen::On);
    h.wait_for(|m| matches!(m, Message::Status { code: StatusCode::Active, display_id: 1 }));
    h.wait_for(|m| matches!(m, Message::Frame(_)));

    // Hotplug: a third display appears.
    source.set(|s| s.displays.push(display(3, "DP-3")));
    h.wait_for(|m| matches!(m, Message::DisplaysChanged(l) if l.len() == 3));
    // The streamed display disappears.
    source.set(|s| s.displays.retain(|d| d.id != 1));
    h.wait_for(|m| matches!(m, Message::Status { code: StatusCode::Gone, display_id: 1 }));
    h.finish();
}

#[test]
fn cursor_image_only_when_it_changes() {
    let source = FakeSource::new("e2e-cursor");
    let h = Harness::new(source.clone(), FakeAuth::new());
    h.handshake();
    h.client.send(&Message::Start { display_id: 1, max_fps: 100, max_in_flight: 4 }, &[]).unwrap();
    let (m, _) = h.wait_for(|m| matches!(m, Message::Cursor(_)));
    let Message::Cursor(c) = m else { unreachable!() };
    assert!(c.visible);
    assert_eq!((c.x, c.y, c.width, c.height), (10, 5, 2, 1));
    let g1 = c.generation;
    assert_eq!(c.image.as_deref().map(|i| i[0]), Some(500u32 as u8));

    // Moving: position updates without a picture.
    source.set(|s| s.cursor.as_mut().unwrap().x = 20);
    let (m, _) = h.wait_for(|m| matches!(m, Message::Cursor(_)));
    let Message::Cursor(c) = m else { unreachable!() };
    assert_eq!((c.x, c.generation, c.image.is_none()), (20, g1, true));

    // New shape (new framebuffer): new generation with the picture.
    source.set(|s| s.cursor.as_mut().unwrap().fb_id = 501);
    let (m, _) = h.wait_for(|m| matches!(m, Message::Cursor(_)));
    let Message::Cursor(c) = m else { unreachable!() };
    assert_ne!(c.generation, g1);
    assert_eq!(c.image.as_deref().map(|i| i[0]), Some(501u32 as u8));

    // Cursor leaves the display (plane moves to another CRTC).
    source.set(|s| s.cursor = None);
    let (m, _) = h.wait_for(|m| matches!(m, Message::Cursor(_)));
    let Message::Cursor(c) = m else { unreachable!() };
    assert!(!c.visible);
    // The picture is read once per framebuffer, not per tick.
    assert!(source.0.lock().unwrap().cursor_reads <= 3);
    h.finish();
}

fn temp_socket(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lunaris-capsvc-{}-{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("capture.sock")
}

struct Server {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Server {
    fn start(path: &std::path::Path, source: FakeSource, max_clients: usize) -> Server {
        let listener = Listener::bind(path, 0o666).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let auth: Arc<dyn Authorizer> = FakeAuth::new();
        let thread = std::thread::spawn(move || {
            serve_with(&listener, move || source.clone(), auth, fast_opts(), max_clients, stop2).unwrap();
        });
        Server { stop, thread: Some(thread) }
    }
    fn shutdown(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.thread.take().unwrap().join().unwrap();
    }
}

fn config() -> StreamConfig {
    StreamConfig { width: 128, height: 64, fps: 60, ..StreamConfig::default() }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drm_service_capture_end_to_end_with_reconnect() {
    let path = temp_socket("e2e");
    let source = FakeSource::new("e2e-cap");
    let server = Server::start(&path, source.clone(), 4);

    let mut cap = DrmServiceCapture::connect(&path).unwrap();
    let displays = cap.list_displays().await.unwrap();
    assert_eq!(displays.len(), 2);
    assert_eq!(displays[1].id, "kms:card7:HDMI-A-1");
    assert_eq!(displays.iter().filter(|d| d.is_primary).count(), 1);
    assert!(cap.delivers_dmabuf() && cap.paces_frames());

    cap.start("kms:card7:HDMI-A-1", &config()).await.unwrap();
    let mut got = 0;
    for _ in 0..200 {
        let f = cap.next_frame().await.unwrap();
        if !f.is_new_frame {
            continue;
        }
        match &f.buffer {
            GpuBuffer::DmaBuf { fd, width, height, stride, fourcc, size, .. } => {
                assert!(*fd >= 0);
                assert_eq!((*width, *height, *stride, *fourcc), (64, 32, 256, XR24));
                assert_eq!(*size, 64 * 32 * 4);
            }
            other => panic!("{:?}", other),
        }
        got += 1;
        if got == 10 {
            break;
        }
    }
    assert_eq!(got, 10);

    // The cursor reaches the slot the cursor pipeline reads, scaled from
    // display pixels (64x32) to encoded pixels (128x64).
    let deadline = Instant::now() + Duration::from_secs(2);
    let cursor = loop {
        if let Some(c) = crate::capture::linux_wayland::pipewire_cursor() {
            break c;
        }
        assert!(Instant::now() < deadline, "no cursor published");
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!((cursor.x, cursor.y, cursor.visible), (20, 10, true));
    assert!(cursor.image.is_some());

    // The service restarts: the backend reconnects and frames continue.
    server.shutdown();
    let server = Server::start(&path, source.clone(), 4);
    let mut after = 0;
    let deadline = Instant::now() + Duration::from_secs(10);
    while after < 5 {
        assert!(Instant::now() < deadline, "no frames after the service restarted");
        let f = cap.next_frame().await.unwrap();
        if f.is_new_frame {
            after += 1;
        }
    }

    // Screen off: next_frame keeps returning "no new frame" quickly.
    source.set(|s| s.screen = Screen::Off);
    tokio::time::sleep(Duration::from_millis(100)).await;
    while cap.next_frame().await.unwrap().is_new_frame {}
    let t = Instant::now();
    assert!(!cap.next_frame().await.unwrap().is_new_frame);
    assert!(t.elapsed() < Duration::from_millis(500));

    // Nothing new: the last picture is repeated at the requested rate (60 fps
    // here), not at 2/3 of it.
    let t = Instant::now();
    let mut repeats = 0;
    while t.elapsed() < Duration::from_millis(600) {
        assert!(!cap.next_frame().await.unwrap().is_new_frame);
        repeats += 1;
    }
    assert!((32..=38).contains(&repeats), "{} repeats in 600 ms at 60 fps", repeats);

    cap.stop().await.unwrap();
    assert!(!cap.is_capturing());
    drop(cap);
    server.shutdown();
    // Every frame descriptor was closed (memfd names are per test).
    assert_eq!(open_fds_matching("e2e-cap-fb"), 0, "descriptor leak");
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn too_many_clients_are_turned_away() {
    let path = temp_socket("busy");
    let server = Server::start(&path, FakeSource::new("e2e-busy"), 1);
    let first = ServiceClient::connect(&path, Duration::from_secs(2)).unwrap();
    match ServiceClient::connect(&path, Duration::from_secs(2)) {
        Err(ClientError::Protocol(m)) => assert!(m.contains("too many")),
        other => panic!("{:?}", other.map(|_| ())),
    }
    drop(first);
    // The slot frees up once the first client leaves.
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if ServiceClient::connect(&path, Duration::from_secs(2)).is_ok() {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(50));
    }
    server.shutdown();
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn missing_service_is_reported_as_unavailable() {
    let path = temp_socket("missing");
    assert!(matches!(
        ServiceClient::connect(&path, Duration::from_millis(200)),
        Err(ClientError::Unavailable(_))
    ));
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// Counts how many sessions exist at once (sanity of the thread-per-client
/// model under a burst of connects and disconnects).
#[test]
fn connect_burst_does_not_leak() {
    let path = temp_socket("burst");
    let server = Server::start(&path, FakeSource::new("e2e-burst"), 4);
    let ok = AtomicU32::new(0);
    for _ in 0..20 {
        if let Ok(c) = ServiceClient::connect(&path, Duration::from_secs(2)) {
            c.conn.send(&Message::Start { display_id: 1, max_fps: 60, max_in_flight: 1 }, &[]).unwrap();
            let _ = c.conn.recv(Some(Duration::from_millis(200)));
            ok.fetch_add(1, Ordering::SeqCst);
        }
    }
    assert!(ok.load(Ordering::SeqCst) >= 18);
    server.shutdown();
    assert_eq!(open_fds_matching("e2e-burst-fb"), 0, "descriptor leak");
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn slow_cursor_reads_do_not_stall_frames() {
    // Each cursor read takes 150 ms (a 33-2000 ms read was measured on VRAM)
    // and the shape changes all the time: frames must keep their pace.
    let source = FakeSource::new("e2e-slowcur");
    source.set(|s| s.cursor_read_delay = Duration::from_millis(150));
    let h = Harness::new(source.clone(), FakeAuth::new());
    h.handshake();
    h.client.send(&Message::Start { display_id: 1, max_fps: 100, max_in_flight: 4 }, &[]).unwrap();
    let t0 = Instant::now();
    let (mut frames, mut max_gap, mut last) = (0u32, Duration::ZERO, Instant::now());
    let mut fb = 500u32;
    let mut images = Vec::new();
    while t0.elapsed() < Duration::from_millis(1000) {
        let Some((m, fds)) = h.client.recv(Some(Duration::from_millis(200))).unwrap() else { continue };
        match m {
            Message::Frame(f) => {
                drop(fds);
                h.client.send(&Message::FrameAck { seq: f.seq }, &[]).unwrap();
                if frames > 0 {
                    max_gap = max_gap.max(last.elapsed());
                }
                last = Instant::now();
                frames += 1;
                if frames % 10 == 0 {
                    fb += 1;
                    source.set(|s| s.cursor.as_mut().unwrap().fb_id = fb);
                }
            }
            Message::Cursor(c) => {
                if let Some(img) = c.image {
                    images.push(img[0]);
                }
            }
            _ => {}
        }
    }
    h.client.send(&Message::Stop, &[]).unwrap();
    h.finish();
    assert!(frames >= 60, "only {} frames in 1 s at 100 fps", frames);
    assert!(max_gap < Duration::from_millis(100), "a frame waited {:?} for a cursor read", max_gap);
    // Pictures still arrive, each matching the framebuffer it was read from.
    assert!(!images.is_empty() && images[0] == 500u32 as u8, "pictures {:?}", images);
}

/// Frames (acknowledged at once) and cursor messages received for `dur`.
fn count_frames(h: &Harness, dur: Duration, mut each: impl FnMut(&Message)) -> u32 {
    let t0 = Instant::now();
    let mut frames = 0;
    while t0.elapsed() < dur {
        let Some((m, fds)) = h.client.recv(Some(Duration::from_millis(50))).unwrap() else { continue };
        drop(fds);
        if let Message::Frame(f) = &m {
            h.client.send(&Message::FrameAck { seq: f.seq }, &[]).unwrap();
            frames += 1;
        }
        each(&m);
    }
    frames
}

#[test]
fn vblank_paces_frames_to_the_refresh_rate() {
    // 100 Hz display, client allows 240: one frame per flip, not 240.
    let source = FakeSource::new("e2e-vbl");
    source.set(|s| s.vblank = Some(Duration::from_millis(10)));
    let h = Harness::new(source, FakeAuth::new());
    h.handshake();
    h.client.send(&Message::Start { display_id: 1, max_fps: 240, max_in_flight: 4 }, &[]).unwrap();
    let frames = count_frames(&h, Duration::from_secs(1), |_| {});
    h.client.send(&Message::Stop, &[]).unwrap();
    h.finish();
    assert!((80..=105).contains(&frames), "{} frames at 100 Hz", frames);
}

#[test]
fn frame_rate_cap_holds_with_vblank() {
    // 200 Hz display, client asks for 50.
    let source = FakeSource::new("e2e-cap");
    source.set(|s| s.vblank = Some(Duration::from_millis(5)));
    let h = Harness::new(source, FakeAuth::new());
    h.handshake();
    h.client.send(&Message::Start { display_id: 1, max_fps: 50, max_in_flight: 4 }, &[]).unwrap();
    let frames = count_frames(&h, Duration::from_secs(1), |_| {});
    h.client.send(&Message::Stop, &[]).unwrap();
    h.finish();
    assert!((42..=55).contains(&frames), "{} frames for a cap of 50", frames);
}

#[test]
fn unchanged_picture_is_not_sent_again_but_the_cursor_is() {
    let source = FakeSource::new("e2e-static");
    source.set(|s| {
        s.vblank = Some(Duration::from_millis(10));
        s.fixed_fb = Some(77);
    });
    let h = Harness::new(source.clone(), FakeAuth::new());
    h.handshake();
    h.client.send(&Message::Start { display_id: 1, max_fps: 100, max_in_flight: 4 }, &[]).unwrap();
    let mut cursor_x = Vec::new();
    let mut moved = false;
    let t0 = Instant::now();
    let frames = count_frames(&h, Duration::from_millis(1500), |m| {
        if let Message::Cursor(c) = m {
            cursor_x.push(c.x);
        }
        if !moved && t0.elapsed() > Duration::from_millis(300) {
            moved = true;
            source.set(|s| s.cursor.as_mut().unwrap().x = 40);
        }
    });
    // The first picture, then one refresh after a second.
    assert!((1..=3).contains(&frames), "{} frames of a static screen", frames);
    assert!(cursor_x.contains(&40), "cursor moves still reported: {:?}", cursor_x);
    // A new picture (the compositor flips) goes out at the next vblank.
    source.set(|s| s.fixed_fb = Some(78));
    let frames = count_frames(&h, Duration::from_millis(200), |_| {});
    assert!(frames >= 1, "new picture not sent");
    h.client.send(&Message::Stop, &[]).unwrap();
    h.finish();
}

#[test]
fn picture_comes_back_at_once_after_the_screen_was_off() {
    let source = FakeSource::new("e2e-wake");
    source.set(|s| {
        s.vblank = Some(Duration::from_millis(10));
        s.fixed_fb = Some(5);
    });
    let h = Harness::new(source.clone(), FakeAuth::new());
    h.handshake();
    h.client.send(&Message::Start { display_id: 1, max_fps: 60, max_in_flight: 4 }, &[]).unwrap();
    h.wait_for(|m| matches!(m, Message::Frame(_)));
    // Screen off: no vblank and no framebuffer; then on with the same id.
    source.set(|s| {
        s.vblank = None;
        s.screen = Screen::Off;
    });
    h.wait_for(|m| matches!(m, Message::Status { code: StatusCode::NoFramebuffer, .. }));
    source.set(|s| {
        s.vblank = Some(Duration::from_millis(10));
        s.screen = Screen::On;
    });
    let t = Instant::now();
    h.wait_for(|m| matches!(m, Message::Frame(_)));
    assert!(t.elapsed() < Duration::from_millis(500), "picture took {:?} to come back", t.elapsed());
    h.client.send(&Message::Stop, &[]).unwrap();
    h.finish();
}
