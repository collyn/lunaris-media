//! `AF_UNIX`/`SOCK_SEQPACKET` transport with descriptor passing.
//!
//! One message is one packet: boundaries are kept by the kernel, and the
//! `SCM_RIGHTS` descriptors of a packet arrive with exactly that packet.
//! Received descriptors are wrapped in [`OwnedFd`] before anything else is
//! looked at, so every error path closes them.

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::protocol::{self, Message, ProtoError, MAX_MESSAGE, MAX_PLANES};

/// `SO_PEERPIDFD` (Linux 6.5+): a pidfd of the peer at connect time.
const SO_PEERPIDFD: libc::c_int = 77;

/// Transport failures.
#[derive(Debug)]
pub enum TransportError {
    /// The peer closed the connection.
    Closed,
    Io(io::Error),
    Protocol(ProtoError),
    /// Packet or its control data did not fit (and was cut by the kernel).
    Truncated,
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::Closed => write!(f, "connection closed"),
            TransportError::Io(e) => write!(f, "{}", e),
            TransportError::Protocol(e) => write!(f, "protocol: {}", e),
            TransportError::Truncated => write!(f, "packet truncated"),
        }
    }
}

impl std::error::Error for TransportError {}

impl From<io::Error> for TransportError {
    fn from(e: io::Error) -> Self {
        TransportError::Io(e)
    }
}

impl From<ProtoError> for TransportError {
    fn from(e: ProtoError) -> Self {
        TransportError::Protocol(e)
    }
}

fn cvt(ret: libc::c_int) -> io::Result<libc::c_int> {
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(ret)
    }
}

fn sockaddr_for(path: &Path) -> io::Result<(libc::sockaddr_un, libc::socklen_t)> {
    use std::os::unix::ffi::OsStrExt;
    // SAFETY: sockaddr_un is plain old data; all-zero is a valid value.
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let bytes = path.as_os_str().as_bytes();
    if bytes.is_empty() || bytes.len() >= addr.sun_path.len() || bytes.contains(&0) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "bad socket path"));
    }
    for (dst, src) in addr.sun_path.iter_mut().zip(bytes) {
        *dst = *src as libc::c_char;
    }
    let len = std::mem::size_of::<libc::sa_family_t>() + bytes.len() + 1;
    Ok((addr, len as libc::socklen_t))
}

fn new_socket() -> io::Result<OwnedFd> {
    // SAFETY: plain socket(2) call; the result is checked and owned at once.
    let fd = cvt(unsafe {
        libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0)
    })?;
    // SAFETY: fd was just returned by socket(2) and is owned by nobody else.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Wait until `fd` is readable. `Ok(false)` on timeout.
pub fn wait_readable(fd: RawFd, timeout: Option<Duration>) -> io::Result<bool> {
    let ms = match timeout {
        None => -1,
        Some(t) => t.as_millis().min(i32::MAX as u128) as libc::c_int,
    };
    let mut pfd = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
    loop {
        // SAFETY: pfd is a valid pollfd for the duration of the call.
        let ret = unsafe { libc::poll(&mut pfd, 1, ms) };
        if ret < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        return Ok(ret > 0);
    }
}

/// Credentials of the peer process at connect time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerCred {
    pub pid: i32,
    pub uid: u32,
    pub gid: u32,
}

/// One connected packet socket.
#[derive(Debug)]
pub struct Conn {
    fd: OwnedFd,
}

impl Conn {
    /// Connect to the service at `path`.
    pub fn connect(path: &Path) -> io::Result<Conn> {
        let fd = new_socket()?;
        let (addr, len) = sockaddr_for(path)?;
        loop {
            // SAFETY: addr/len describe a valid sockaddr_un.
            let ret = unsafe {
                libc::connect(fd.as_raw_fd(), &addr as *const _ as *const libc::sockaddr, len)
            };
            if ret == 0 {
                break;
            }
            let e = io::Error::last_os_error();
            if e.kind() != io::ErrorKind::Interrupted {
                return Err(e);
            }
        }
        Ok(Conn { fd })
    }

    /// A connected pair (tests, in-process use).
    pub fn pair() -> io::Result<(Conn, Conn)> {
        let mut fds = [0 as libc::c_int; 2];
        // SAFETY: fds has room for the two descriptors socketpair(2) returns.
        cvt(unsafe {
            libc::socketpair(
                libc::AF_UNIX,
                libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC,
                0,
                fds.as_mut_ptr(),
            )
        })?;
        // SAFETY: both descriptors were just created and are owned here only.
        Ok(unsafe {
            (
                Conn { fd: OwnedFd::from_raw_fd(fds[0]) },
                Conn { fd: OwnedFd::from_raw_fd(fds[1]) },
            )
        })
    }

    pub fn raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Bound how long a send may block (a client that stops reading must not
    /// stall the service).
    pub fn set_send_timeout(&self, timeout: Duration) -> io::Result<()> {
        let tv = libc::timeval {
            tv_sec: timeout.as_secs() as libc::time_t,
            tv_usec: timeout.subsec_micros() as libc::suseconds_t,
        };
        // SAFETY: tv is a valid timeval for SO_SNDTIMEO.
        cvt(unsafe {
            libc::setsockopt(
                self.raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_SNDTIMEO,
                &tv as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::timeval>() as libc::socklen_t,
            )
        })?;
        Ok(())
    }

    /// `SO_PEERCRED`.
    pub fn peer_cred(&self) -> io::Result<PeerCred> {
        // SAFETY: ucred is plain old data.
        let mut cred: libc::ucred = unsafe { std::mem::zeroed() };
        let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        // SAFETY: cred/len are valid out-parameters for SO_PEERCRED.
        cvt(unsafe {
            libc::getsockopt(
                self.raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                &mut cred as *mut _ as *mut libc::c_void,
                &mut len,
            )
        })?;
        Ok(PeerCred { pid: cred.pid, uid: cred.uid, gid: cred.gid })
    }

    /// `SO_PEERPIDFD`, where the kernel has it.
    pub fn peer_pidfd(&self) -> Option<OwnedFd> {
        let mut fd: libc::c_int = -1;
        let mut len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
        // SAFETY: fd/len are valid out-parameters for an int-sized option.
        let ret = unsafe {
            libc::getsockopt(
                self.raw_fd(),
                libc::SOL_SOCKET,
                SO_PEERPIDFD,
                &mut fd as *mut _ as *mut libc::c_void,
                &mut len,
            )
        };
        if ret < 0 || fd < 0 {
            return None;
        }
        // SAFETY: the kernel just handed us this new descriptor.
        Some(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// Before closing after a refusal: read (and discard) what the peer
    /// already sent, for up to `wait`. Closing a socket with unread data
    /// makes the kernel reset the connection, and the peer would lose the
    /// error message it was about to read.
    pub fn drain(&self, wait: Duration) {
        let deadline = std::time::Instant::now() + wait;
        loop {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            match wait_readable(self.raw_fd(), Some(left)) {
                Ok(true) => {
                    if self.recv_raw().is_err() {
                        return;
                    }
                }
                _ => return,
            }
            if left.is_zero() {
                return;
            }
        }
    }

    /// Wake up and fail any blocked `recv`/`send` (other threads included).
    pub fn shutdown(&self) {
        // SAFETY: shutdown(2) on a descriptor we own.
        unsafe { libc::shutdown(self.raw_fd(), libc::SHUT_RDWR) };
    }

    /// Send `msg` with `fds` attached (the count must match the message).
    pub fn send(&self, msg: &Message, fds: &[RawFd]) -> Result<(), TransportError> {
        if fds.len() != msg.fd_count() || fds.len() > MAX_PLANES {
            return Err(TransportError::Protocol(ProtoError::FdCount {
                expected: msg.fd_count(),
                got: fds.len(),
            }));
        }
        let bytes = protocol::encode(msg)?;
        self.send_raw(&bytes, fds)
    }

    /// Send pre-encoded bytes (tests use it to inject malformed packets).
    pub fn send_raw(&self, bytes: &[u8], fds: &[RawFd]) -> Result<(), TransportError> {
        let mut iov = libc::iovec { iov_base: bytes.as_ptr() as *mut libc::c_void, iov_len: bytes.len() };
        let fd_bytes = std::mem::size_of_val(fds) as u32;
        // u64 storage keeps the control buffer aligned for cmsghdr.
        let mut cmsg_buf = [0u64; 8];
        // SAFETY: msghdr is plain old data; fields are set below.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        if !fds.is_empty() {
            // SAFETY: CMSG_SPACE is a pure size computation.
            let space = unsafe { libc::CMSG_SPACE(fd_bytes) } as usize;
            assert!(space <= std::mem::size_of_val(&cmsg_buf));
            msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
            msg.msg_controllen = space as _;
            // SAFETY: msg_control points at `space` zeroed, aligned bytes, so
            // CMSG_FIRSTHDR yields a valid header with room for the fds.
            unsafe {
                let cmsg = libc::CMSG_FIRSTHDR(&msg);
                (*cmsg).cmsg_level = libc::SOL_SOCKET;
                (*cmsg).cmsg_type = libc::SCM_RIGHTS;
                (*cmsg).cmsg_len = libc::CMSG_LEN(fd_bytes) as _;
                std::ptr::copy_nonoverlapping(
                    fds.as_ptr() as *const u8,
                    libc::CMSG_DATA(cmsg),
                    fd_bytes as usize,
                );
            }
        }
        loop {
            // SAFETY: msg points at valid iov/control buffers for the call.
            let ret = unsafe { libc::sendmsg(self.raw_fd(), &msg, libc::MSG_NOSIGNAL) };
            if ret >= 0 {
                return Ok(());
            }
            let e = io::Error::last_os_error();
            match e.raw_os_error() {
                Some(libc::EINTR) => continue,
                Some(libc::EPIPE) | Some(libc::ECONNRESET) => return Err(TransportError::Closed),
                _ => return Err(TransportError::Io(e)),
            }
        }
    }

    /// Receive one message. `Ok(None)` when `timeout` passed without one.
    pub fn recv(&self, timeout: Option<Duration>) -> Result<Option<(Message, Vec<OwnedFd>)>, TransportError> {
        if timeout.is_some() && !wait_readable(self.raw_fd(), timeout)? {
            return Ok(None);
        }
        let (bytes, fds) = self.recv_raw()?;
        let msg = protocol::decode(&bytes, fds.len())?;
        Ok(Some((msg, fds)))
    }

    /// Receive one packet and its descriptors (already owned).
    pub fn recv_raw(&self) -> Result<(Vec<u8>, Vec<OwnedFd>), TransportError> {
        // One byte more than the largest valid packet, to see oversized ones.
        let mut buf = vec![0u8; MAX_MESSAGE + 1];
        let mut iov = libc::iovec { iov_base: buf.as_mut_ptr() as *mut libc::c_void, iov_len: buf.len() };
        // Room for MAX_PLANES descriptors; more makes the kernel set
        // MSG_CTRUNC (and close the ones that did not fit).
        let mut cmsg_buf = [0u64; 8];
        // SAFETY: CMSG_SPACE is a pure size computation.
        let space = unsafe { libc::CMSG_SPACE((MAX_PLANES * std::mem::size_of::<RawFd>()) as u32) } as usize;
        assert!(space <= std::mem::size_of_val(&cmsg_buf));
        // SAFETY: msghdr is plain old data; fields are set below.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
        msg.msg_controllen = space as _;
        let n = loop {
            // SAFETY: msg points at valid buffers for the duration of the call.
            let ret = unsafe { libc::recvmsg(self.raw_fd(), &mut msg, libc::MSG_CMSG_CLOEXEC) };
            if ret >= 0 {
                break ret as usize;
            }
            let e = io::Error::last_os_error();
            match e.raw_os_error() {
                Some(libc::EINTR) => continue,
                Some(libc::ECONNRESET) => return Err(TransportError::Closed),
                _ => return Err(TransportError::Io(e)),
            }
        };
        // Take ownership of every received descriptor first.
        let mut fds = Vec::new();
        // SAFETY: walking the control buffer the kernel filled, bounded by
        // msg_controllen; each SCM_RIGHTS payload is an array of ints.
        unsafe {
            let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
            while !cmsg.is_null() {
                if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                    let data = libc::CMSG_DATA(cmsg);
                    let payload = (*cmsg).cmsg_len as usize - (data as usize - cmsg as usize);
                    let count = payload / std::mem::size_of::<RawFd>();
                    for i in 0..count {
                        let raw = std::ptr::read_unaligned((data as *const RawFd).add(i));
                        if raw >= 0 {
                            fds.push(OwnedFd::from_raw_fd(raw));
                        }
                    }
                }
                cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
            }
        }
        if msg.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC) != 0 || n > MAX_MESSAGE {
            return Err(TransportError::Truncated);
        }
        if n == 0 {
            // SEQPACKET: a zero-length read is end of stream.
            return Err(TransportError::Closed);
        }
        buf.truncate(n);
        Ok((buf, fds))
    }
}

/// Listening socket bound to a path.
#[derive(Debug)]
pub struct Listener {
    fd: OwnedFd,
    path: PathBuf,
}

impl Listener {
    /// Bind `path` (replacing a stale socket, never another kind of file)
    /// and make it connectable by everyone (`mode`, e.g. 0o666): who may
    /// actually use it is decided per connection.
    pub fn bind(path: &Path, mode: u32) -> io::Result<Listener> {
        use std::os::unix::fs::FileTypeExt;
        match std::fs::symlink_metadata(path) {
            Ok(meta) if meta.file_type().is_socket() => std::fs::remove_file(path)?,
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("{} exists and is not a socket", path.display()),
                ))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        let fd = new_socket()?;
        let (addr, len) = sockaddr_for(path)?;
        // Created with no permissions for others, then opened up explicitly,
        // so there is no window with an unintended mode.
        // SAFETY: umask(2) only changes this process's file mode mask.
        let old = unsafe { libc::umask(0o177) };
        // SAFETY: addr/len describe a valid sockaddr_un.
        let ret = unsafe { libc::bind(fd.as_raw_fd(), &addr as *const _ as *const libc::sockaddr, len) };
        // SAFETY: restores the previous mask.
        unsafe { libc::umask(old) };
        cvt(ret)?;
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
        // SAFETY: listen(2) on our bound socket.
        cvt(unsafe { libc::listen(fd.as_raw_fd(), 16) })?;
        Ok(Listener { fd, path: path.to_path_buf() })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Accept one connection; `Ok(None)` when `timeout` passed.
    pub fn accept(&self, timeout: Option<Duration>) -> io::Result<Option<Conn>> {
        if !wait_readable(self.raw_fd(), timeout)? {
            return Ok(None);
        }
        loop {
            // SAFETY: accept4(2) without an address buffer.
            let ret = unsafe {
                libc::accept4(self.raw_fd(), std::ptr::null_mut(), std::ptr::null_mut(), libc::SOCK_CLOEXEC)
            };
            if ret >= 0 {
                // SAFETY: a fresh descriptor from accept4(2).
                return Ok(Some(Conn { fd: unsafe { OwnedFd::from_raw_fd(ret) } }));
            }
            let e = io::Error::last_os_error();
            match e.raw_os_error() {
                Some(libc::EINTR) => continue,
                Some(libc::EAGAIN) | Some(libc::ECONNABORTED) => return Ok(None),
                _ => return Err(e),
            }
        }
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Number of open descriptors of this process (leak checks).
pub fn open_fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd").map(|d| d.count()).unwrap_or(0)
}

/// Number of open descriptors whose target contains `needle` (e.g. a memfd
/// name): leak checks that other threads' descriptors do not disturb.
pub fn open_fds_matching(needle: &str) -> usize {
    std::fs::read_dir("/proc/self/fd")
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter_map(|e| std::fs::read_link(e.path()).ok())
                .filter(|t| t.to_string_lossy().contains(needle))
                .count()
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::drm_service::protocol::{WireFrame, WirePlane};

    fn memfd(name: &str, len: usize) -> OwnedFd {
        let cname = std::ffi::CString::new(name).unwrap();
        // SAFETY: memfd_create with a valid C string.
        let fd = unsafe { libc::memfd_create(cname.as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0);
        // SAFETY: fresh descriptor.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        // SAFETY: ftruncate on our memfd.
        assert_eq!(unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) }, 0);
        fd
    }

    fn frame_msg(planes: u32) -> Message {
        Message::Frame(WireFrame {
            seq: 1,
            display_id: 1,
            timestamp_us: 1,
            width: 4,
            height: 4,
            fourcc: 0,
            modifier: 0,
            buffer_id: 0,
            crop_x: 0,
            crop_y: 0,
            crop_w: 4,
            crop_h: 4,
            planes: (0..planes).map(|i| WirePlane { fd_index: i, offset: 0, stride: 16 }).collect(),
        })
    }

    #[test]
    fn descriptors_travel_and_are_closed() {
        let (a, b) = Conn::pair().unwrap();
        for _ in 0..50 {
            let m1 = memfd("tp-travel-0", 4096);
            let m2 = memfd("tp-travel-1", 4096);
            // Mark the memfd so the receiver can tell it got the right file.
            // SAFETY: write to our memfd.
            unsafe { libc::pwrite(m1.as_raw_fd(), b"LUNA".as_ptr() as *const _, 4, 0) };
            a.send(&frame_msg(2), &[m1.as_raw_fd(), m2.as_raw_fd()]).unwrap();
            drop((m1, m2));
            let (msg, fds) = b.recv(Some(Duration::from_secs(1))).unwrap().unwrap();
            assert!(matches!(msg, Message::Frame(_)));
            assert_eq!(fds.len(), 2);
            let mut word = [0u8; 4];
            // SAFETY: read from the received descriptor.
            unsafe { libc::pread(fds[0].as_raw_fd(), word.as_mut_ptr() as *mut _, 4, 0) };
            assert_eq!(&word, b"LUNA");
            // Received descriptors are close-on-exec.
            // SAFETY: fcntl on a valid descriptor.
            let flags = unsafe { libc::fcntl(fds[0].as_raw_fd(), libc::F_GETFD) };
            assert!(flags & libc::FD_CLOEXEC != 0);
        }
        assert_eq!(open_fds_matching("tp-travel-"), 0, "descriptor leak");
    }

    #[test]
    fn fd_count_mismatch_closes_received_fds() {
        let (a, b) = Conn::pair().unwrap();
        // A Stop packet carrying a descriptor is refused, and the descriptor
        // does not stay open in the receiver.
        let m = memfd("tp-stray", 16);
        let bytes = protocol::encode(&Message::Stop).unwrap();
        a.send_raw(&bytes, &[m.as_raw_fd()]).unwrap();
        drop(m);
        let err = b.recv(Some(Duration::from_secs(1))).unwrap_err();
        assert!(matches!(err, TransportError::Protocol(ProtoError::FdCount { .. })));
        // Too many descriptors: the kernel truncates the control data.
        let many: Vec<OwnedFd> = (0..6).map(|i| memfd(&format!("tp-stray-{}", i), 16)).collect();
        let raw: Vec<RawFd> = many.iter().map(|f| f.as_raw_fd()).collect();
        a.send_raw(&bytes, &raw).unwrap();
        drop(many);
        assert!(b.recv(Some(Duration::from_secs(1))).is_err());
        assert_eq!(open_fds_matching("tp-stray"), 0, "descriptor leak");
    }

    #[test]
    fn send_refuses_wrong_fd_count() {
        let (a, _b) = Conn::pair().unwrap();
        let m = memfd("x", 16);
        assert!(a.send(&frame_msg(2), &[m.as_raw_fd()]).is_err());
        assert!(a.send(&Message::Stop, &[m.as_raw_fd()]).is_err());
    }

    #[test]
    fn timeout_and_close() {
        let (a, b) = Conn::pair().unwrap();
        assert!(b.recv(Some(Duration::from_millis(20))).unwrap().is_none());
        drop(a);
        assert!(matches!(b.recv(Some(Duration::from_millis(200))), Err(TransportError::Closed)));
    }

    #[test]
    fn oversized_packet_is_refused() {
        let (a, b) = Conn::pair().unwrap();
        // Raise the send buffer so the kernel accepts the big packet.
        let size: libc::c_int = 1 << 20;
        // SAFETY: setsockopt with a valid int.
        unsafe {
            libc::setsockopt(
                a.raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_SNDBUF,
                &size as *const _ as *const libc::c_void,
                4,
            )
        };
        let big = vec![0u8; MAX_MESSAGE + 100];
        if a.send_raw(&big, &[]).is_ok() {
            assert!(matches!(b.recv(Some(Duration::from_secs(1))), Err(TransportError::Truncated)));
        }
    }

    #[test]
    fn listener_accepts_and_reports_peer() {
        let dir = std::env::temp_dir().join(format!("lunaris-cs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.sock");
        let l = Listener::bind(&path, 0o666).unwrap();
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o666);
        let c = Conn::connect(&path).unwrap();
        let s = l.accept(Some(Duration::from_secs(1))).unwrap().unwrap();
        let cred = s.peer_cred().unwrap();
        assert_eq!(cred.pid as u32, std::process::id());
        // SAFETY: getuid has no preconditions.
        assert_eq!(cred.uid, unsafe { libc::getuid() });
        c.send(&Message::Stop, &[]).unwrap();
        assert!(matches!(s.recv(Some(Duration::from_secs(1))).unwrap(), Some((Message::Stop, _))));
        // Rebinding replaces the stale socket; a regular file is never removed.
        drop(l);
        assert!(!path.exists());
        std::fs::write(&path, b"x").unwrap();
        assert!(Listener::bind(&path, 0o666).is_err());
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
