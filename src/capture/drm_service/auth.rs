//! Who may receive frames from the capture service.
//!
//! The service runs as root and can read every screen, so each connection is
//! checked (fail closed):
//!
//! 1. `SO_PEERCRED`: the peer's uid must be root or the uid of the **active
//!    graphical session on seat0** (logind; read through libsystemd /
//!    libelogind like `sd_seat_get_active`, with a parser of logind's runtime
//!    files as fallback).
//! 2. The peer's executable (`/proc/<pid>/exe`, compared by device+inode, so
//!    renames and symlinks do not matter) must be the service's own binary or
//!    an allow-listed path; every allow-listed file and all its parent
//!    directories must be owned by root and not writable by group/others.
//!    An allow-listed AppImage is accepted through the `APPIMAGE` variable
//!    its runtime sets (its code runs from a FUSE mount the user owns, so
//!    there is no root-owned file to compare against; see the security doc).
//! 3. No `LD_PRELOAD`/`LD_AUDIT` (and no `LD_LIBRARY_PATH` outside AppImages)
//!    in the peer's environment, and no tracer attached (`TracerPid`).
//! 4. The pid cannot have been recycled between `SO_PEERCRED` and the checks
//!    (`SO_PEERPIDFD` liveness, else the process start time).
//!
//! The decision itself is the pure function [`authorize`]; everything that
//! reads the system only gathers facts for it.

use std::collections::HashMap;
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Identity of an executable file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExeId {
    pub dev: u64,
    pub ino: u64,
}

/// What the service knows about a connected process.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PeerFacts {
    pub pid: i32,
    pub uid: u32,
    /// `None` when `/proc/<pid>/exe` could not be read.
    pub exe: Option<ExeId>,
    /// For logs only.
    pub exe_path: Option<String>,
    /// `APPIMAGE` from the peer's environment.
    pub appimage: Option<String>,
    /// `LD_PRELOAD` or `LD_AUDIT` set.
    pub preload: bool,
    /// `LD_LIBRARY_PATH` set.
    pub library_path: bool,
    /// `/proc/<pid>/environ` could not be read (refused: the checks above
    /// cannot be made).
    pub environ_unreadable: bool,
    /// A debugger is attached.
    pub traced: bool,
}

/// What the service accepts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Policy {
    pub allowed_exes: Vec<ExeId>,
    /// Canonical paths of allow-listed AppImages.
    pub allowed_appimages: Vec<String>,
}

/// Why a peer was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Deny {
    NoActiveSession,
    NotActiveUser { uid: u32, active: u32 },
    Traced,
    Preload,
    EnvironmentUnreadable,
    UnknownExe,
    ExeNotAllowed,
    ProcessGone,
}

impl std::fmt::Display for Deny {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Deny::NoActiveSession => write!(f, "no active graphical session on seat0"),
            Deny::NotActiveUser { uid, active } => {
                write!(f, "uid {} is not the active session's user (uid {})", uid, active)
            }
            Deny::Traced => write!(f, "the process is being traced"),
            Deny::Preload => write!(f, "the process runs with LD_PRELOAD/LD_AUDIT/LD_LIBRARY_PATH"),
            Deny::EnvironmentUnreadable => write!(
                f,
                "the process's environment cannot be read (does the service lack CAP_DAC_READ_SEARCH?)"
            ),
            Deny::UnknownExe => write!(f, "the process's executable cannot be read"),
            Deny::ExeNotAllowed => write!(f, "the process's executable is not an allowed Lunaris binary"),
            Deny::ProcessGone => write!(f, "the process exited during the check"),
        }
    }
}

/// The decision (pure). `active_uid` is the uid of the active graphical
/// session on seat0, if any.
pub fn authorize(peer: &PeerFacts, active_uid: Option<u32>, policy: &Policy) -> Result<(), Deny> {
    if peer.uid == 0 {
        // Root can open the DRM devices itself; nothing to protect.
        return Ok(());
    }
    match active_uid {
        None => return Err(Deny::NoActiveSession),
        Some(active) if active != peer.uid => {
            return Err(Deny::NotActiveUser { uid: peer.uid, active })
        }
        Some(_) => {}
    }
    if peer.traced {
        return Err(Deny::Traced);
    }
    if peer.environ_unreadable {
        return Err(Deny::EnvironmentUnreadable);
    }
    if peer.preload {
        return Err(Deny::Preload);
    }
    if let Some(exe) = peer.exe {
        if policy.allowed_exes.contains(&exe) {
            if peer.library_path {
                return Err(Deny::Preload);
            }
            return Ok(());
        }
    }
    // An AppImage runs from a FUSE mount that root may not even stat (no
    // allow_other), so it is recognised by the APPIMAGE variable its runtime
    // sets. AppRun sets LD_LIBRARY_PATH itself, so that is tolerated.
    if let Some(appimage) = &peer.appimage {
        if policy.allowed_appimages.iter().any(|a| a == appimage) {
            return Ok(());
        }
    }
    if peer.exe.is_none() {
        return Err(Deny::UnknownExe);
    }
    Err(Deny::ExeNotAllowed)
}

/// Re-check of a connected peer while streaming: only what can change.
pub fn still_allowed(peer: &PeerFacts, active_uid: Option<u32>) -> Result<(), Deny> {
    if peer.uid == 0 {
        return Ok(());
    }
    match active_uid {
        None => Err(Deny::NoActiveSession),
        Some(active) if active != peer.uid => Err(Deny::NotActiveUser { uid: peer.uid, active }),
        Some(_) if peer.traced => Err(Deny::Traced),
        Some(_) => Ok(()),
    }
}

// --- Trusted paths -------------------------------------------------------------

/// `path` and each parent directory must be owned by root and not writable
/// by group or others (sticky world-writable dirs like /tmp included: they
/// are refused). Returns the file's identity.
pub fn trusted_file(path: &Path) -> Result<ExeId, String> {
    use std::os::unix::fs::MetadataExt;
    let canonical = std::fs::canonicalize(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let meta = std::fs::metadata(&canonical).map_err(|e| format!("{}: {}", canonical.display(), e))?;
    if !meta.is_file() {
        return Err(format!("{} is not a regular file", canonical.display()));
    }
    let mut cur: Option<&Path> = Some(&canonical);
    while let Some(p) = cur {
        let m = std::fs::metadata(p).map_err(|e| format!("{}: {}", p.display(), e))?;
        if m.uid() != 0 {
            return Err(format!("{} is not owned by root", p.display()));
        }
        if m.mode() & 0o022 != 0 {
            return Err(format!("{} is writable by group or others", p.display()));
        }
        cur = p.parent();
    }
    Ok(ExeId { dev: meta.dev(), ino: meta.ino() })
}

impl Policy {
    /// Build the policy from the service's own executable plus allow-lists.
    /// Entries that are missing or not trusted are skipped (and logged), so
    /// the service keeps working when an optional package is absent.
    pub fn resolve(allow_exes: &[PathBuf], allow_appimages: &[PathBuf], include_self: bool) -> Policy {
        let mut policy = Policy::default();
        let mut candidates: Vec<PathBuf> = Vec::new();
        if include_self {
            if let Ok(own) = std::fs::read_link("/proc/self/exe") {
                candidates.push(own);
            }
        }
        candidates.extend(allow_exes.iter().cloned());
        for path in candidates {
            match trusted_file(&path) {
                Ok(id) => {
                    if !policy.allowed_exes.contains(&id) {
                        policy.allowed_exes.push(id);
                    }
                }
                Err(e) => log::debug!("capture service: not allowing {}: {}", path.display(), e),
            }
        }
        for path in allow_appimages {
            match trusted_file(path) {
                Ok(_) => {
                    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.clone());
                    policy.allowed_appimages.push(canonical.to_string_lossy().into_owned());
                }
                Err(e) => log::debug!("capture service: not allowing AppImage {}: {}", path.display(), e),
            }
        }
        policy
    }
}

// --- Facts about a peer ------------------------------------------------------

/// Values of the variables that matter, from a `/proc/<pid>/environ` blob.
pub fn parse_environ(blob: &[u8]) -> (Option<String>, bool, bool) {
    let mut appimage = None;
    let mut preload = false;
    let mut library_path = false;
    for entry in blob.split(|b| *b == 0) {
        let Some(eq) = entry.iter().position(|b| *b == b'=') else { continue };
        let (key, value) = (&entry[..eq], &entry[eq + 1..]);
        match key {
            b"APPIMAGE" => appimage = Some(String::from_utf8_lossy(value).into_owned()),
            b"LD_PRELOAD" | b"LD_AUDIT" if !value.is_empty() => preload = true,
            b"LD_LIBRARY_PATH" if !value.is_empty() => library_path = true,
            _ => {}
        }
    }
    (appimage, preload, library_path)
}

/// `TracerPid` from `/proc/<pid>/status` text.
pub fn parse_tracer_pid(status: &str) -> Option<i32> {
    status
        .lines()
        .find_map(|l| l.strip_prefix("TracerPid:"))
        .and_then(|v| v.trim().parse().ok())
}

/// Start time (field 22) from `/proc/<pid>/stat` text.
pub fn parse_start_time(stat: &str) -> Option<u64> {
    // The command name (field 2) may contain spaces and parentheses; fields
    // after the last ')' are well-formed.
    let rest = &stat[stat.rfind(')')? + 1..];
    rest.split_whitespace().nth(19)?.parse().ok()
}

fn read_start_time(pid: i32) -> Option<u64> {
    parse_start_time(&std::fs::read_to_string(format!("/proc/{}/stat", pid)).ok()?)
}

/// Gather the facts about process `pid` (running as `uid`).
pub fn gather_peer_facts(pid: i32, uid: u32) -> PeerFacts {
    use std::os::unix::fs::MetadataExt;
    let proc_dir = format!("/proc/{}", pid);
    let exe = std::fs::metadata(format!("{}/exe", proc_dir))
        .ok()
        .map(|m| ExeId { dev: m.dev(), ino: m.ino() });
    let exe_path = std::fs::read_link(format!("{}/exe", proc_dir))
        .ok()
        .map(|p| p.to_string_lossy().into_owned());
    let environ = std::fs::read(format!("{}/environ", proc_dir));
    let environ_unreadable = environ.is_err();
    let (appimage, preload, library_path) = match environ {
        Ok(blob) => parse_environ(&blob),
        // Unreadable environment: assume the worst (and say why).
        Err(_) => (None, true, true),
    };
    let traced = std::fs::read_to_string(format!("{}/status", proc_dir))
        .ok()
        .and_then(|s| parse_tracer_pid(&s))
        .is_none_or(|t| t != 0);
    PeerFacts { pid, uid, exe, exe_path, appimage, preload, library_path, environ_unreadable, traced }
}

/// Refresh only the facts that can change while connected.
pub fn refresh_traced(facts: &mut PeerFacts) {
    facts.traced = std::fs::read_to_string(format!("/proc/{}/status", facts.pid))
        .ok()
        .and_then(|s| parse_tracer_pid(&s))
        .is_none_or(|t| t != 0);
}

/// `true` while the process behind `pidfd` has not exited.
///
/// Signal 0 checks existence only. EPERM ("may not signal it") still means
/// the process exists: the service runs without CAP_KILL (systemd unit), so
/// that is the answer for every agent of another user. Only ESRCH means gone.
pub fn pidfd_alive(pidfd: &OwnedFd) -> bool {
    // SAFETY: pidfd_send_signal with signal 0 only checks for existence.
    let ret = unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            pidfd.as_raw_fd(),
            0,
            std::ptr::null::<libc::siginfo_t>(),
            0,
        )
    };
    ret == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Gather facts for a peer and make sure they describe the process that
/// connected (not a later process that got the same pid).
pub fn gather_checked(pid: i32, uid: u32, pidfd: Option<&OwnedFd>) -> Result<PeerFacts, Deny> {
    if pid <= 0 {
        return Err(Deny::ProcessGone);
    }
    let start_before = read_start_time(pid);
    let facts = gather_peer_facts(pid, uid);
    let same = match pidfd {
        Some(fd) => pidfd_alive(fd),
        None => start_before.is_some() && start_before == read_start_time(pid),
    };
    if !same {
        return Err(Deny::ProcessGone);
    }
    Ok(facts)
}

// --- Active session ------------------------------------------------------------

/// What logind says about a session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionInfo {
    pub uid: Option<u32>,
    pub kind: String,
    pub class: String,
    pub state: String,
}

/// The uid that may capture: the active session's user when the session is
/// a graphical user session that is active.
pub fn graphical_uid(session: &SessionInfo) -> Option<u32> {
    let graphical = matches!(session.kind.as_str(), "wayland" | "x11" | "mir");
    let user = session.class == "user" || session.class.is_empty();
    let active = session.state == "active" || session.state.is_empty();
    if graphical && user && active {
        session.uid
    } else {
        None
    }
}

fn kv(text: &str, key: &str) -> Option<String> {
    text.lines()
        .filter_map(|l| l.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v.trim().to_string())
}

/// `ACTIVE` session id from logind's seat file.
pub fn parse_seat_active(text: &str) -> Option<String> {
    kv(text, "ACTIVE").filter(|s| !s.is_empty())
}

/// Session facts from logind's session file.
pub fn parse_session_file(text: &str) -> SessionInfo {
    SessionInfo {
        uid: kv(text, "UID").and_then(|v| v.parse().ok()),
        kind: kv(text, "TYPE").unwrap_or_default(),
        class: kv(text, "CLASS").unwrap_or_default(),
        state: kv(text, "STATE").unwrap_or_default(),
    }
}

type SdSeatGetActive = unsafe extern "C" fn(*const libc::c_char, *mut *mut libc::c_char, *mut libc::uid_t) -> libc::c_int;
type SdSessionGetStr = unsafe extern "C" fn(*const libc::c_char, *mut *mut libc::c_char) -> libc::c_int;

struct SdLogin {
    _lib: libloading::Library,
    seat_get_active: SdSeatGetActive,
    session_get_type: SdSessionGetStr,
    session_get_class: SdSessionGetStr,
    session_get_state: SdSessionGetStr,
}

impl SdLogin {
    fn load() -> Option<SdLogin> {
        for name in ["libsystemd.so.0", "libelogind.so.0"] {
            // SAFETY: loading the system's login library; its initialisers
            // are trusted like any linked system library.
            let Ok(lib) = (unsafe { libloading::Library::new(name) }) else { continue };
            // SAFETY: the symbols have the C signatures declared above
            // (sd-login.h).
            let loaded = unsafe {
                (|| -> Result<SdLogin, libloading::Error> {
                    let seat_get_active = *lib.get::<SdSeatGetActive>(b"sd_seat_get_active\0")?;
                    let session_get_type = *lib.get::<SdSessionGetStr>(b"sd_session_get_type\0")?;
                    let session_get_class = *lib.get::<SdSessionGetStr>(b"sd_session_get_class\0")?;
                    let session_get_state = *lib.get::<SdSessionGetStr>(b"sd_session_get_state\0")?;
                    Ok(SdLogin {
                        seat_get_active,
                        session_get_type,
                        session_get_class,
                        session_get_state,
                        _lib: lib,
                    })
                })()
            };
            if let Ok(sd) = loaded {
                return Some(sd);
            }
        }
        None
    }

    fn string(&self, f: SdSessionGetStr, session: &std::ffi::CStr) -> String {
        let mut out: *mut libc::c_char = std::ptr::null_mut();
        // SAFETY: session is a valid C string; out receives a malloc'd
        // string we free below.
        let ret = unsafe { f(session.as_ptr(), &mut out) };
        if ret < 0 || out.is_null() {
            return String::new();
        }
        // SAFETY: out is a valid NUL-terminated string from libsystemd.
        let s = unsafe { std::ffi::CStr::from_ptr(out) }.to_string_lossy().into_owned();
        // SAFETY: libsystemd allocates with malloc.
        unsafe { libc::free(out as *mut libc::c_void) };
        s
    }

    fn active(&self, seat: &str) -> Option<SessionInfo> {
        let seat = std::ffi::CString::new(seat).ok()?;
        let mut session: *mut libc::c_char = std::ptr::null_mut();
        let mut uid: libc::uid_t = 0;
        // SAFETY: valid seat string and out-parameters.
        let ret = unsafe { (self.seat_get_active)(seat.as_ptr(), &mut session, &mut uid) };
        if ret < 0 || session.is_null() {
            return None;
        }
        // SAFETY: session is a malloc'd NUL-terminated string.
        let id = unsafe { std::ffi::CStr::from_ptr(session) }.to_owned();
        // SAFETY: freeing the string libsystemd allocated.
        unsafe { libc::free(session as *mut libc::c_void) };
        Some(SessionInfo {
            uid: Some(uid),
            kind: self.string(self.session_get_type, &id),
            class: self.string(self.session_get_class, &id),
            state: self.string(self.session_get_state, &id),
        })
    }
}

fn sd_login() -> Option<&'static SdLogin> {
    static SD: std::sync::OnceLock<Option<SdLogin>> = std::sync::OnceLock::new();
    SD.get_or_init(SdLogin::load).as_ref()
}

fn active_session_from_files() -> Option<SessionInfo> {
    let seat = std::fs::read_to_string("/run/systemd/seats/seat0").ok()?;
    let id = parse_seat_active(&seat)?;
    if id.contains('/') || id.contains("..") {
        return None;
    }
    let text = std::fs::read_to_string(format!("/run/systemd/sessions/{}", id)).ok()?;
    Some(parse_session_file(&text))
}

/// The active graphical session's user on seat0, uncached.
pub fn query_active_uid() -> Option<u32> {
    let session = match sd_login() {
        Some(sd) => sd.active("seat0"),
        None => active_session_from_files(),
    }?;
    graphical_uid(&session)
}

/// [`query_active_uid`] cached for `ttl` (several checks per second
/// otherwise each reread logind's state).
pub fn active_uid_cached(ttl: Duration) -> Option<u32> {
    static CACHE: Mutex<Option<(Instant, Option<u32>)>> = Mutex::new(None);
    let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, uid)) = *guard {
        if at.elapsed() < ttl {
            return uid;
        }
    }
    let uid = query_active_uid();
    *guard = Some((Instant::now(), uid));
    uid
}

/// Rate-limits refusal logs per (uid, reason) so a misbehaving client
/// cannot flood the journal.
pub struct RefusalLog {
    last: HashMap<(u32, String), Instant>,
}

impl RefusalLog {
    pub fn new() -> Self {
        RefusalLog { last: HashMap::new() }
    }

    pub fn log(&mut self, peer: &PeerFacts, deny: &Deny) {
        let key = (peer.uid, format!("{:?}", deny));
        let now = Instant::now();
        if self.last.get(&key).is_some_and(|t| now.duration_since(*t) < Duration::from_secs(60)) {
            return;
        }
        if self.last.len() > 256 {
            self.last.clear();
        }
        self.last.insert(key, now);
        log::warn!(
            "capture service: refused pid {} uid {} exe {}: {}",
            peer.pid,
            peer.uid,
            peer.exe_path.as_deref().unwrap_or("?"),
            deny
        );
    }
}

impl Default for RefusalLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AGENT: ExeId = ExeId { dev: 1, ino: 100 };
    const OTHER: ExeId = ExeId { dev: 1, ino: 200 };

    fn policy() -> Policy {
        Policy {
            allowed_exes: vec![AGENT],
            allowed_appimages: vec!["/opt/lunaris-agent/lunaris-agent.AppImage".into()],
        }
    }

    fn peer(uid: u32, exe: ExeId) -> PeerFacts {
        PeerFacts { pid: 1234, uid, exe: Some(exe), exe_path: Some("/usr/bin/x".into()), ..Default::default() }
    }

    #[test]
    fn active_user_with_the_agent_binary_is_allowed() {
        assert_eq!(authorize(&peer(1000, AGENT), Some(1000), &policy()), Ok(()));
    }

    #[test]
    fn root_is_allowed_without_further_checks() {
        let p = PeerFacts { uid: 0, traced: true, exe: None, ..Default::default() };
        assert_eq!(authorize(&p, None, &policy()), Ok(()));
        assert_eq!(still_allowed(&p, None), Ok(()));
    }

    #[test]
    fn other_users_and_no_session_are_refused() {
        assert_eq!(
            authorize(&peer(1001, AGENT), Some(1000), &policy()),
            Err(Deny::NotActiveUser { uid: 1001, active: 1000 })
        );
        assert_eq!(authorize(&peer(1000, AGENT), None, &policy()), Err(Deny::NoActiveSession));
    }

    #[test]
    fn foreign_binaries_are_refused() {
        assert_eq!(authorize(&peer(1000, OTHER), Some(1000), &policy()), Err(Deny::ExeNotAllowed));
        let mut p = peer(1000, OTHER);
        p.exe = None;
        assert_eq!(authorize(&p, Some(1000), &policy()), Err(Deny::UnknownExe));
    }

    #[test]
    fn preload_and_tracing_are_refused() {
        let mut p = peer(1000, AGENT);
        p.preload = true;
        assert_eq!(authorize(&p, Some(1000), &policy()), Err(Deny::Preload));
        let mut p = peer(1000, AGENT);
        p.library_path = true;
        assert_eq!(authorize(&p, Some(1000), &policy()), Err(Deny::Preload));
        let mut p = peer(1000, AGENT);
        p.traced = true;
        assert_eq!(authorize(&p, Some(1000), &policy()), Err(Deny::Traced));
        assert_eq!(still_allowed(&p, Some(1000)), Err(Deny::Traced));
        // An environment the service cannot read is refused, and says so.
        let mut p = peer(1000, AGENT);
        (p.environ_unreadable, p.preload, p.library_path) = (true, true, true);
        assert_eq!(authorize(&p, Some(1000), &policy()), Err(Deny::EnvironmentUnreadable));
        assert!(Deny::EnvironmentUnreadable.to_string().contains("CAP_DAC_READ_SEARCH"));
    }

    #[test]
    fn appimage_is_matched_by_its_path() {
        let mut p = peer(1000, OTHER);
        p.appimage = Some("/opt/lunaris-agent/lunaris-agent.AppImage".into());
        p.library_path = true; // AppRun sets it
        assert_eq!(authorize(&p, Some(1000), &policy()), Ok(()));
        p.appimage = Some("/home/u/evil.AppImage".into());
        assert_eq!(authorize(&p, Some(1000), &policy()), Err(Deny::ExeNotAllowed));
        // LD_PRELOAD is refused for AppImages too.
        p.appimage = Some("/opt/lunaris-agent/lunaris-agent.AppImage".into());
        p.preload = true;
        assert_eq!(authorize(&p, Some(1000), &policy()), Err(Deny::Preload));
        // Root cannot stat a FUSE-mounted executable: still matched.
        let mut p = peer(1000, OTHER);
        p.exe = None;
        p.appimage = Some("/opt/lunaris-agent/lunaris-agent.AppImage".into());
        assert_eq!(authorize(&p, Some(1000), &policy()), Ok(()));
    }

    #[test]
    fn session_switch_stops_a_connected_peer() {
        let p = peer(1000, AGENT);
        assert_eq!(still_allowed(&p, Some(1000)), Ok(()));
        assert!(still_allowed(&p, Some(1001)).is_err());
        assert!(still_allowed(&p, None).is_err());
    }

    #[test]
    fn environ_parsing() {
        let blob = b"HOME=/home/u\0APPIMAGE=/opt/a.AppImage\0LD_LIBRARY_PATH=/tmp/x\0LD_PRELOAD=\0";
        assert_eq!(parse_environ(blob), (Some("/opt/a.AppImage".into()), false, true));
        let blob = b"LD_AUDIT=/x.so\0";
        assert_eq!(parse_environ(blob), (None, true, false));
        assert_eq!(parse_environ(b""), (None, false, false));
    }

    #[test]
    fn proc_parsing() {
        assert_eq!(parse_tracer_pid("Name:\tx\nTracerPid:\t0\n"), Some(0));
        assert_eq!(parse_tracer_pid("TracerPid:\t812\n"), Some(812));
        let stat = "1234 (we ird) name)) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 424242 20";
        assert_eq!(parse_start_time(stat), Some(424242));
        let own = std::fs::read_to_string("/proc/self/stat").unwrap();
        assert!(parse_start_time(&own).is_some());
    }

    #[test]
    fn logind_files_and_graphical_rule() {
        let seat = "# This is private data. Do not parse.\nIS_SEAT0=1\nACTIVE=148\nACTIVE_UID=1000\n";
        assert_eq!(parse_seat_active(seat), Some("148".into()));
        assert_eq!(parse_seat_active("ACTIVE=\n"), None);
        let s = parse_session_file("UID=1000\nTYPE=wayland\nCLASS=user\nSTATE=active\n");
        assert_eq!(graphical_uid(&s), Some(1000));
        let greeter = parse_session_file("UID=120\nTYPE=wayland\nCLASS=greeter\nSTATE=active\n");
        assert_eq!(graphical_uid(&greeter), None);
        let tty = parse_session_file("UID=1000\nTYPE=tty\nCLASS=user\nSTATE=active\n");
        assert_eq!(graphical_uid(&tty), None);
        let online = parse_session_file("UID=1000\nTYPE=x11\nCLASS=user\nSTATE=online\n");
        assert_eq!(graphical_uid(&online), None);
    }

    #[test]
    fn trusted_file_rules() {
        // A system binary owned by root.
        let ls = ["/usr/bin/ls", "/bin/ls"].iter().map(Path::new).find(|p| p.exists()).unwrap();
        assert!(trusted_file(ls).is_ok());
        // A file in a user-owned directory is not trusted.
        let dir = std::env::temp_dir().join(format!("lunaris-auth-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("agent");
        std::fs::write(&f, b"x").unwrap();
        // SAFETY: getuid has no preconditions.
        if unsafe { libc::getuid() } != 0 {
            assert!(trusted_file(&f).is_err());
        }
        assert!(trusted_file(&dir.join("missing")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn pidfd_of(pid: i32) -> Option<OwnedFd> {
        use std::os::fd::FromRawFd;
        // SAFETY: pidfd_open with flags 0; the result is checked.
        let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as i32;
        // SAFETY: a fresh descriptor owned here.
        (fd >= 0).then(|| unsafe { OwnedFd::from_raw_fd(fd) })
    }

    #[test]
    fn a_process_we_may_not_signal_is_still_alive() {
        // The service runs without CAP_KILL (systemd unit), so probing another
        // user's process with signal 0 gets EPERM: that process exists.
        // Reproduced unprivileged with pid 1 (root's), which we may not signal.
        // SAFETY: getuid has no preconditions.
        if unsafe { libc::getuid() } == 0 {
            return;
        }
        let Some(init) = pidfd_of(1) else { return };
        assert!(pidfd_alive(&init), "EPERM was taken for an exited process");
        // Our own process: alive. An exited child: not.
        // SAFETY: getpid has no preconditions.
        assert!(pidfd_alive(&pidfd_of(unsafe { libc::getpid() }).unwrap()));
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let fd = pidfd_of(child.id() as i32).unwrap();
        child.wait().unwrap();
        assert!(!pidfd_alive(&fd), "a reaped child counts as alive");
    }

    #[test]
    fn own_process_facts() {
        // SAFETY: getpid/getuid have no preconditions.
        let (pid, uid) = unsafe { (libc::getpid(), libc::getuid()) };
        let facts = gather_checked(pid, uid, None).unwrap();
        assert!(facts.exe.is_some());
        assert!(!facts.traced || std::env::var("RUST_TEST_UNDER_DEBUGGER").is_ok());
        assert_eq!(gather_checked(0, uid, None), Err(Deny::ProcessGone));
    }
}
