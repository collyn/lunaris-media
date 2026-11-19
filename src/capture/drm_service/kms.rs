//! Read-only KMS access for the capture service.
//!
//! Only "get" ioctls are used (resources, connectors, encoders, CRTCs,
//! planes, properties, `GETFB2`) plus `PRIME_HANDLE_TO_FD`/`GEM_CLOSE` on
//! handles the service itself received. Nothing here sets a mode, commits
//! an atomic state or becomes DRM master; `SET_CLIENT_CAP` only changes what
//! this file descriptor is shown (universal planes / atomic properties).
//!
//! Connector queries never ask the kernel to re-probe the outputs (a
//! non-zero mode count is passed, like libdrm's `drmModeGetConnectorCurrent`).
//!
//! Enumeration works without privileges for any user with access to the card
//! node; exporting buffers (`GETFB2` handles) needs `CAP_SYS_ADMIN`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::{Path, PathBuf};

use super::protocol::{WireDisplay, WirePlane, DISPLAY_ACTIVE, MAX_CURSOR_SIDE, MAX_NAME};

// --- ioctl numbers -------------------------------------------------------------

const fn ioc(dir: u32, ty: u32, nr: u32, size: usize) -> libc::c_ulong {
    ((dir << 30) | ((size as u32) << 16) | (ty << 8) | nr) as libc::c_ulong
}
const IOW: u32 = 1;
const IOWR: u32 = 3;
const D: u32 = b'd' as u32;

const DRM_IOCTL_GEM_CLOSE: libc::c_ulong = ioc(IOW, D, 0x09, std::mem::size_of::<GemClose>());
const DRM_IOCTL_SET_CLIENT_CAP: libc::c_ulong = ioc(IOW, D, 0x0D, std::mem::size_of::<SetClientCap>());
const DRM_IOCTL_PRIME_HANDLE_TO_FD: libc::c_ulong = ioc(IOWR, D, 0x2D, std::mem::size_of::<PrimeHandle>());
const DRM_IOCTL_MODE_GETRESOURCES: libc::c_ulong = ioc(IOWR, D, 0xA0, std::mem::size_of::<CardRes>());
const DRM_IOCTL_MODE_GETCRTC: libc::c_ulong = ioc(IOWR, D, 0xA1, std::mem::size_of::<ModeCrtc>());
const DRM_IOCTL_MODE_GETENCODER: libc::c_ulong = ioc(IOWR, D, 0xA6, std::mem::size_of::<GetEncoder>());
const DRM_IOCTL_MODE_GETCONNECTOR: libc::c_ulong = ioc(IOWR, D, 0xA7, std::mem::size_of::<GetConnector>());
const DRM_IOCTL_MODE_GETPROPERTY: libc::c_ulong = ioc(IOWR, D, 0xAA, std::mem::size_of::<GetProperty>());
const DRM_IOCTL_MODE_GETPLANERESOURCES: libc::c_ulong = ioc(IOWR, D, 0xB5, std::mem::size_of::<GetPlaneRes>());
const DRM_IOCTL_MODE_GETPLANE: libc::c_ulong = ioc(IOWR, D, 0xB6, std::mem::size_of::<GetPlane>());
const DRM_IOCTL_MODE_OBJ_GETPROPERTIES: libc::c_ulong = ioc(IOWR, D, 0xB9, std::mem::size_of::<ObjGetProps>());
const DRM_IOCTL_MODE_GETFB2: libc::c_ulong = ioc(IOWR, D, 0xCE, std::mem::size_of::<FbCmd2>());
const DRM_IOCTL_WAIT_VBLANK: libc::c_ulong = ioc(IOWR, D, 0x3A, std::mem::size_of::<WaitVblank>());
const DMA_BUF_IOCTL_SYNC: libc::c_ulong = ioc(IOW, b'b' as u32, 0, std::mem::size_of::<DmaBufSync>());

const DRM_CLIENT_CAP_UNIVERSAL_PLANES: u64 = 2;
const DRM_CLIENT_CAP_ATOMIC: u64 = 3;
const DRM_CLOEXEC: u32 = libc::O_CLOEXEC as u32;
const DRM_MODE_FB_MODIFIERS: u32 = 2;
const DRM_MODE_OBJECT_CRTC: u32 = 0xcccc_cccc;
const DRM_MODE_OBJECT_PLANE: u32 = 0xeeee_eeee;
const DRM_MODE_CONNECTED: u32 = 1;
const DRM_MODE_CONNECTOR_WRITEBACK: u32 = 18;
const DRM_PLANE_TYPE_CURSOR: u64 = 2;
const DRM_VBLANK_RELATIVE: u32 = 0x1;
const DRM_VBLANK_HIGH_CRTC_SHIFT: u32 = 1;
const DRM_VBLANK_HIGH_CRTC_MASK: u32 = 0x3e;
const DMA_BUF_SYNC_READ: u64 = 1;
const DMA_BUF_SYNC_START: u64 = 0;
const DMA_BUF_SYNC_END: u64 = 4;

/// `DRM_FORMAT_MOD_LINEAR`.
pub const MOD_LINEAR: u64 = 0;
/// `DRM_FORMAT_MOD_INVALID` (implicit, driver-private layout).
pub const MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

const FOURCC_ARGB8888: u32 = u32::from_le_bytes(*b"AR24");
const FOURCC_XRGB8888: u32 = u32::from_le_bytes(*b"XR24");
const FOURCC_ABGR8888: u32 = u32::from_le_bytes(*b"AB24");
const FOURCC_XBGR8888: u32 = u32::from_le_bytes(*b"XB24");

// --- Kernel structs (include/uapi/drm/drm.h, drm_mode.h) --------------------

#[repr(C)]
#[derive(Default)]
struct SetClientCap {
    capability: u64,
    value: u64,
}

#[repr(C)]
#[derive(Default)]
struct GemClose {
    handle: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct PrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}

#[repr(C)]
#[derive(Default)]
struct CardRes {
    fb_id_ptr: u64,
    crtc_id_ptr: u64,
    connector_id_ptr: u64,
    encoder_id_ptr: u64,
    count_fbs: u32,
    count_crtcs: u32,
    count_connectors: u32,
    count_encoders: u32,
    min_width: u32,
    max_width: u32,
    min_height: u32,
    max_height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ModeInfo {
    clock: u32,
    hdisplay: u16,
    hsync_start: u16,
    hsync_end: u16,
    htotal: u16,
    hskew: u16,
    vdisplay: u16,
    vsync_start: u16,
    vsync_end: u16,
    vtotal: u16,
    vscan: u16,
    vrefresh: u32,
    flags: u32,
    type_: u32,
    name: [u8; 32],
}

#[repr(C)]
#[derive(Default)]
struct ModeCrtc {
    set_connectors_ptr: u64,
    count_connectors: u32,
    crtc_id: u32,
    fb_id: u32,
    x: u32,
    y: u32,
    gamma_size: u32,
    mode_valid: u32,
    mode: ModeInfo,
}

#[repr(C)]
#[derive(Default)]
struct GetEncoder {
    encoder_id: u32,
    encoder_type: u32,
    crtc_id: u32,
    possible_crtcs: u32,
    possible_clones: u32,
}

#[repr(C)]
#[derive(Default)]
struct GetConnector {
    encoders_ptr: u64,
    modes_ptr: u64,
    props_ptr: u64,
    prop_values_ptr: u64,
    count_modes: u32,
    count_props: u32,
    count_encoders: u32,
    encoder_id: u32,
    connector_id: u32,
    connector_type: u32,
    connector_type_id: u32,
    connection: u32,
    mm_width: u32,
    mm_height: u32,
    subpixel: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct GetProperty {
    values_ptr: u64,
    enum_blob_ptr: u64,
    prop_id: u32,
    flags: u32,
    name: [u8; 32],
    count_values: u32,
    count_enum_blobs: u32,
}

#[repr(C)]
#[derive(Default)]
struct GetPlaneRes {
    plane_id_ptr: u64,
    count_planes: u32,
}

#[repr(C)]
#[derive(Default)]
struct GetPlane {
    plane_id: u32,
    crtc_id: u32,
    fb_id: u32,
    possible_crtcs: u32,
    gamma_size: u32,
    count_format_types: u32,
    format_type_ptr: u64,
}

#[repr(C)]
#[derive(Default)]
struct ObjGetProps {
    props_ptr: u64,
    prop_values_ptr: u64,
    count_props: u32,
    obj_id: u32,
    obj_type: u32,
}

#[repr(C)]
#[derive(Default)]
struct FbCmd2 {
    fb_id: u32,
    width: u32,
    height: u32,
    pixel_format: u32,
    flags: u32,
    handles: [u32; 4],
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: [u64; 4],
}

/// `union drm_wait_vblank` (request and reply share the 24 bytes).
#[repr(C)]
#[derive(Default)]
struct WaitVblank {
    kind: u32,
    sequence: u32,
    /// Request: `signal`; reply: `tval_sec`.
    a: u64,
    /// Reply: `tval_usec`.
    b: u64,
}

#[repr(C)]
#[derive(Default)]
struct DmaBufSync {
    flags: u64,
}

fn ioctl<T>(fd: RawFd, request: libc::c_ulong, arg: &mut T) -> io::Result<()> {
    loop {
        // SAFETY: `arg` matches the struct size encoded in `request` (the
        // constants above are built from these very types); any pointers it
        // contains point at buffers that outlive the call.
        let ret = unsafe { libc::ioctl(fd, request, arg as *mut T) };
        if ret == 0 {
            return Ok(());
        }
        let e = io::Error::last_os_error();
        match e.raw_os_error() {
            Some(libc::EINTR) | Some(libc::EAGAIN) => continue,
            _ => return Err(e),
        }
    }
}

/// Kernel connector type names (`drm_connector_enum_list`), as compositors
/// name outputs ("DP-2", "HDMI-A-1", "eDP-1").
pub fn connector_type_name(kind: u32) -> &'static str {
    const NAMES: [&str; 21] = [
        "Unknown", "VGA", "DVI-I", "DVI-D", "DVI-A", "Composite", "SVIDEO", "LVDS", "Component",
        "DIN", "DP", "HDMI-A", "HDMI-B", "TV", "eDP", "Virtual", "DSI", "DPI", "Writeback", "SPI",
        "USB",
    ];
    NAMES.get(kind as usize).copied().unwrap_or("Unknown")
}

/// Printable fourcc ("XR24").
pub fn fourcc_str(fourcc: u32) -> String {
    fourcc
        .to_le_bytes()
        .iter()
        .map(|b| if b.is_ascii_graphic() { *b as char } else { '?' })
        .collect()
}

// --- Info structs ------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Resources {
    pub crtcs: Vec<u32>,
    pub connectors: Vec<u32>,
    pub encoders: Vec<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConnectorInfo {
    pub id: u32,
    pub kind: u32,
    pub kind_id: u32,
    pub connected: bool,
    pub encoder_id: u32,
}

impl ConnectorInfo {
    pub fn name(&self) -> String {
        format!("{}-{}", connector_type_name(self.kind), self.kind_id)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrtcInfo {
    pub id: u32,
    pub fb_id: u32,
    pub x: u32,
    pub y: u32,
    pub mode_valid: bool,
    pub width: u32,
    pub height: u32,
    pub refresh_mhz: u32,
}

/// Framebuffer layout from `GETFB2` (handles already closed).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FbLayout {
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub modifier: u64,
    /// (offset, pitch) per plane.
    pub planes: Vec<(u32, u32)>,
    /// The caller may export the buffer (privileged).
    pub exportable: bool,
}

/// A framebuffer exported as DMA-BUF(s).
#[derive(Debug)]
pub struct ExportedFb {
    pub layout: FbLayout,
    pub fds: Vec<OwnedFd>,
    pub planes: Vec<WirePlane>,
    /// Inode of the first DMA-BUF: equal for the same buffer.
    pub buffer_id: u64,
    /// KMS framebuffer id (0 when not from KMS): the compositor shows new
    /// content by flipping to another framebuffer, so an unchanged id means
    /// an unchanged picture.
    pub fb_id: u32,
}

/// State of a cursor plane.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CursorPlaneState {
    pub plane_id: u32,
    pub fb_id: u32,
    pub x: i32,
    pub y: i32,
    pub hot_x: i32,
    pub hot_y: i32,
}

/// RGBA (straight alpha) cursor picture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorPicture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// One open DRM card.
#[derive(Debug)]
pub struct Card {
    fd: OwnedFd,
    pub path: PathBuf,
    /// `card2`.
    pub name: String,
    pub minor: u32,
    pub atomic: bool,
    prop_names: RefCell<HashMap<u32, String>>,
    cursor_planes: RefCell<Option<Vec<u32>>>,
    crtcs: RefCell<Option<Vec<u32>>>,
}

impl Card {
    /// Open a card node read/write (DRM ioctls want it) and ask for
    /// universal planes and atomic properties (visibility only).
    pub fn open(path: &Path) -> io::Result<Card> {
        use std::os::unix::ffi::OsStrExt;
        let cpath = std::ffi::CString::new(path.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path"))?;
        // SAFETY: valid C string; result checked.
        let raw = unsafe { libc::open(cpath.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if raw < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: fresh descriptor owned here.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let minor = name.trim_start_matches("card").parse().unwrap_or(0);
        let mut cap = SetClientCap { capability: DRM_CLIENT_CAP_UNIVERSAL_PLANES, value: 1 };
        let _ = ioctl(fd.as_raw_fd(), DRM_IOCTL_SET_CLIENT_CAP, &mut cap);
        let mut cap = SetClientCap { capability: DRM_CLIENT_CAP_ATOMIC, value: 1 };
        let atomic = ioctl(fd.as_raw_fd(), DRM_IOCTL_SET_CLIENT_CAP, &mut cap).is_ok();
        Ok(Card {
            fd,
            path: path.to_path_buf(),
            name,
            minor,
            atomic,
            prop_names: RefCell::new(HashMap::new()),
            cursor_planes: RefCell::new(None),
            crtcs: RefCell::new(None),
        })
    }

    pub fn raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    pub fn resources(&self) -> io::Result<Resources> {
        for _ in 0..5 {
            let mut res = CardRes::default();
            ioctl(self.raw_fd(), DRM_IOCTL_MODE_GETRESOURCES, &mut res)?;
            let (nf, nc, nk, ne) = (res.count_fbs, res.count_crtcs, res.count_connectors, res.count_encoders);
            let mut fbs = vec![0u32; nf as usize];
            let mut crtcs = vec![0u32; nc as usize];
            let mut conns = vec![0u32; nk as usize];
            let mut encs = vec![0u32; ne as usize];
            res.fb_id_ptr = fbs.as_mut_ptr() as u64;
            res.crtc_id_ptr = crtcs.as_mut_ptr() as u64;
            res.connector_id_ptr = conns.as_mut_ptr() as u64;
            res.encoder_id_ptr = encs.as_mut_ptr() as u64;
            ioctl(self.raw_fd(), DRM_IOCTL_MODE_GETRESOURCES, &mut res)?;
            // Hotplug between the calls: the kernel did not fill the arrays.
            if res.count_fbs > nf || res.count_crtcs > nc || res.count_connectors > nk || res.count_encoders > ne {
                continue;
            }
            crtcs.truncate(res.count_crtcs as usize);
            conns.truncate(res.count_connectors as usize);
            encs.truncate(res.count_encoders as usize);
            return Ok(Resources { crtcs, connectors: conns, encoders: encs });
        }
        Err(io::Error::other("DRM resources kept changing"))
    }

    /// Connector state without triggering a probe.
    pub fn connector(&self, id: u32) -> io::Result<ConnectorInfo> {
        let mut mode = ModeInfo::default();
        let mut c = GetConnector {
            connector_id: id,
            // Non-zero: "current state", never a forced re-probe.
            count_modes: 1,
            modes_ptr: &mut mode as *mut ModeInfo as u64,
            ..Default::default()
        };
        ioctl(self.raw_fd(), DRM_IOCTL_MODE_GETCONNECTOR, &mut c)?;
        Ok(ConnectorInfo {
            id,
            kind: c.connector_type,
            kind_id: c.connector_type_id,
            connected: c.connection == DRM_MODE_CONNECTED,
            encoder_id: c.encoder_id,
        })
    }

    pub fn encoder_crtc(&self, encoder_id: u32) -> io::Result<u32> {
        let mut e = GetEncoder { encoder_id, ..Default::default() };
        ioctl(self.raw_fd(), DRM_IOCTL_MODE_GETENCODER, &mut e)?;
        Ok(e.crtc_id)
    }

    pub fn crtc(&self, id: u32) -> io::Result<CrtcInfo> {
        let mut c = ModeCrtc { crtc_id: id, ..Default::default() };
        ioctl(self.raw_fd(), DRM_IOCTL_MODE_GETCRTC, &mut c)?;
        let m = c.mode;
        let refresh_mhz = if m.htotal > 0 && m.vtotal > 0 && m.clock > 0 {
            ((m.clock as u64 * 1_000_000) / (m.htotal as u64 * m.vtotal as u64)) as u32
        } else {
            m.vrefresh * 1000
        };
        Ok(CrtcInfo {
            id,
            fb_id: c.fb_id,
            x: c.x,
            y: c.y,
            mode_valid: c.mode_valid != 0,
            width: m.hdisplay as u32,
            height: m.vdisplay as u32,
            refresh_mhz,
        })
    }

    /// (property id, value) pairs of a KMS object.
    pub fn properties(&self, obj_id: u32, obj_type: u32) -> io::Result<Vec<(u32, u64)>> {
        for _ in 0..3 {
            let mut q = ObjGetProps { obj_id, obj_type, ..Default::default() };
            ioctl(self.raw_fd(), DRM_IOCTL_MODE_OBJ_GETPROPERTIES, &mut q)?;
            let n = q.count_props;
            let mut ids = vec![0u32; n as usize];
            let mut values = vec![0u64; n as usize];
            q.props_ptr = ids.as_mut_ptr() as u64;
            q.prop_values_ptr = values.as_mut_ptr() as u64;
            ioctl(self.raw_fd(), DRM_IOCTL_MODE_OBJ_GETPROPERTIES, &mut q)?;
            if q.count_props > n {
                continue;
            }
            ids.truncate(q.count_props as usize);
            values.truncate(q.count_props as usize);
            return Ok(ids.into_iter().zip(values).collect());
        }
        Err(io::Error::other("properties kept changing"))
    }

    pub fn property_name(&self, prop_id: u32) -> String {
        if let Some(n) = self.prop_names.borrow().get(&prop_id) {
            return n.clone();
        }
        let mut p = GetProperty { prop_id, ..Default::default() };
        let name = match ioctl(self.raw_fd(), DRM_IOCTL_MODE_GETPROPERTY, &mut p) {
            Ok(()) => {
                let end = p.name.iter().position(|b| *b == 0).unwrap_or(p.name.len());
                String::from_utf8_lossy(&p.name[..end]).into_owned()
            }
            Err(_) => String::new(),
        };
        self.prop_names.borrow_mut().insert(prop_id, name.clone());
        name
    }

    /// Properties by name.
    pub fn named_properties(&self, obj_id: u32, obj_type: u32) -> io::Result<HashMap<String, u64>> {
        Ok(self
            .properties(obj_id, obj_type)?
            .into_iter()
            .map(|(id, v)| (self.property_name(id), v))
            .collect())
    }

    /// CRTC `ACTIVE` (atomic drivers): `None` when unknown.
    pub fn crtc_active(&self, crtc_id: u32) -> Option<bool> {
        if !self.atomic {
            return None;
        }
        self.named_properties(crtc_id, DRM_MODE_OBJECT_CRTC)
            .ok()?
            .get("ACTIVE")
            .map(|v| *v != 0)
    }

    pub fn planes(&self) -> io::Result<Vec<u32>> {
        for _ in 0..3 {
            let mut r = GetPlaneRes::default();
            ioctl(self.raw_fd(), DRM_IOCTL_MODE_GETPLANERESOURCES, &mut r)?;
            let n = r.count_planes;
            let mut ids = vec![0u32; n as usize];
            r.plane_id_ptr = ids.as_mut_ptr() as u64;
            ioctl(self.raw_fd(), DRM_IOCTL_MODE_GETPLANERESOURCES, &mut r)?;
            if r.count_planes > n {
                continue;
            }
            ids.truncate(r.count_planes as usize);
            return Ok(ids);
        }
        Err(io::Error::other("planes kept changing"))
    }

    /// (crtc id, fb id) of a plane.
    pub fn plane(&self, plane_id: u32) -> io::Result<(u32, u32)> {
        let mut p = GetPlane { plane_id, ..Default::default() };
        ioctl(self.raw_fd(), DRM_IOCTL_MODE_GETPLANE, &mut p)?;
        Ok((p.crtc_id, p.fb_id))
    }

    fn cursor_plane_ids(&self) -> Vec<u32> {
        if let Some(ids) = self.cursor_planes.borrow().as_ref() {
            return ids.clone();
        }
        let ids: Vec<u32> = self
            .planes()
            .unwrap_or_default()
            .into_iter()
            .filter(|id| {
                self.named_properties(*id, DRM_MODE_OBJECT_PLANE)
                    .ok()
                    .and_then(|p| p.get("type").copied())
                    == Some(DRM_PLANE_TYPE_CURSOR)
            })
            .collect();
        *self.cursor_planes.borrow_mut() = Some(ids.clone());
        ids
    }

    /// The cursor plane shown on `crtc_id`, if any (needs atomic properties
    /// for the position).
    pub fn cursor_on(&self, crtc_id: u32) -> Option<CursorPlaneState> {
        if !self.atomic {
            return None;
        }
        for plane_id in self.cursor_plane_ids() {
            let Ok(p) = self.named_properties(plane_id, DRM_MODE_OBJECT_PLANE) else { continue };
            if p.get("CRTC_ID").copied() != Some(crtc_id as u64) {
                continue;
            }
            let fb_id = p.get("FB_ID").copied().unwrap_or(0) as u32;
            if fb_id == 0 {
                continue;
            }
            // CRTC_X/Y are signed 32-bit values carried in the u64.
            let signed = |name: &str| p.get(name).map(|v| *v as i64 as i32).unwrap_or(0);
            return Some(CursorPlaneState {
                plane_id,
                fb_id,
                x: signed("CRTC_X"),
                y: signed("CRTC_Y"),
                hot_x: signed("HOTSPOT_X"),
                hot_y: signed("HOTSPOT_Y"),
            });
        }
        None
    }

    /// `GETFB2` and the GEM handles it returned (caller closes them).
    fn getfb2(&self, fb_id: u32) -> io::Result<FbCmd2> {
        let mut fb = FbCmd2 { fb_id, ..Default::default() };
        ioctl(self.raw_fd(), DRM_IOCTL_MODE_GETFB2, &mut fb)?;
        Ok(fb)
    }

    fn close_handles(&self, fb: &FbCmd2) {
        let mut closed: Vec<u32> = Vec::new();
        for &h in &fb.handles {
            if h != 0 && !closed.contains(&h) {
                let mut c = GemClose { handle: h, pad: 0 };
                if let Err(e) = ioctl(self.raw_fd(), DRM_IOCTL_GEM_CLOSE, &mut c) {
                    log::warn!("GEM_CLOSE({}) failed: {}", h, e);
                }
                closed.push(h);
            }
        }
    }

    fn layout_of(fb: &FbCmd2) -> FbLayout {
        let n = (0..4).take_while(|&i| fb.pitches[i] != 0 || fb.handles[i] != 0).count().max(1);
        FbLayout {
            width: fb.width,
            height: fb.height,
            fourcc: fb.pixel_format,
            modifier: if fb.flags & DRM_MODE_FB_MODIFIERS != 0 { fb.modifier[0] } else { MOD_INVALID },
            planes: (0..n).map(|i| (fb.offsets[i], fb.pitches[i])).collect(),
            exportable: fb.handles[0] != 0,
        }
    }

    /// Index of `crtc_id` among the card's CRTCs (the "pipe" vblank waits use).
    pub fn crtc_index(&self, crtc_id: u32) -> Option<u32> {
        if self.crtcs.borrow().is_none() {
            *self.crtcs.borrow_mut() = Some(self.resources().ok()?.crtcs);
        }
        self.crtcs.borrow().as_ref()?.iter().position(|c| *c == crtc_id).map(|i| i as u32)
    }

    /// Block until the next vertical blank of `crtc_id`: at most one refresh
    /// period while the display is on. Fails at once when it is off (no
    /// vblank interrupts) or the driver has none; callers then pace by clock.
    /// Read-only: needs no DRM master.
    pub fn wait_vblank(&self, crtc_id: u32) -> io::Result<()> {
        let pipe = self
            .crtc_index(crtc_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no such CRTC"))?;
        let high = (pipe << DRM_VBLANK_HIGH_CRTC_SHIFT) & DRM_VBLANK_HIGH_CRTC_MASK;
        if (high >> DRM_VBLANK_HIGH_CRTC_SHIFT) != pipe {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "CRTC index too large"));
        }
        let mut w = WaitVblank { kind: DRM_VBLANK_RELATIVE | high, sequence: 1, ..Default::default() };
        ioctl(self.raw_fd(), DRM_IOCTL_WAIT_VBLANK, &mut w)
    }

    /// Layout of a framebuffer (works unprivileged; handles are dropped).
    pub fn fb_layout(&self, fb_id: u32) -> io::Result<FbLayout> {
        let fb = self.getfb2(fb_id)?;
        self.close_handles(&fb);
        Ok(Self::layout_of(&fb))
    }

    /// Export a framebuffer as read-only DMA-BUF(s). Needs `CAP_SYS_ADMIN`.
    pub fn export_fb(&self, fb_id: u32) -> io::Result<ExportedFb> {
        let fb = self.getfb2(fb_id)?;
        let layout = Self::layout_of(&fb);
        if !layout.exportable {
            self.close_handles(&fb);
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "GETFB2 returned no buffer handle (CAP_SYS_ADMIN needed)",
            ));
        }
        let mut handle_order: Vec<u32> = Vec::new();
        let mut fds: Vec<OwnedFd> = Vec::new();
        let mut planes = Vec::with_capacity(layout.planes.len());
        let mut result = Ok(());
        for (i, &(offset, pitch)) in layout.planes.iter().enumerate() {
            let h = if fb.handles[i] != 0 { fb.handles[i] } else { fb.handles[0] };
            let index = match handle_order.iter().position(|x| *x == h) {
                Some(idx) => idx,
                None => {
                    // Read-only: the consumer has no business writing into
                    // the compositor's buffers.
                    let mut prime = PrimeHandle { handle: h, flags: DRM_CLOEXEC, fd: -1 };
                    if let Err(e) = ioctl(self.raw_fd(), DRM_IOCTL_PRIME_HANDLE_TO_FD, &mut prime) {
                        result = Err(e);
                        break;
                    }
                    // SAFETY: the kernel handed us this new descriptor.
                    fds.push(unsafe { OwnedFd::from_raw_fd(prime.fd) });
                    handle_order.push(h);
                    handle_order.len() - 1
                }
            };
            planes.push(WirePlane { fd_index: index as u32, offset, stride: pitch });
        }
        // The DMA-BUFs keep the buffers alive; the handles are not needed.
        self.close_handles(&fb);
        result?;
        let buffer_id = fds.first().map(fd_inode).unwrap_or(0);
        Ok(ExportedFb { layout, fds, planes, buffer_id, fb_id })
    }

    /// Read a (linear, 32-bit) cursor framebuffer into straight-alpha RGBA,
    /// trimmed of transparent right/bottom margins. `None` when the buffer
    /// cannot be mapped or has an unknown layout.
    pub fn read_cursor(&self, fb_id: u32) -> Option<CursorPicture> {
        read_cursor_buffer(&self.export_fb(fb_id).ok()?)
    }
}

/// [`Card::read_cursor`] on an already exported framebuffer. Needs no card,
/// so it can run on another thread (the read takes tens of milliseconds when
/// the buffer sits in VRAM).
pub fn read_cursor_buffer(exported: &ExportedFb) -> Option<CursorPicture> {
    {
        let layout = &exported.layout;
        if exported.fds.len() != 1 || layout.planes.len() != 1 {
            return None;
        }
        if layout.modifier != MOD_LINEAR && layout.modifier != MOD_INVALID {
            return None;
        }
        let (offset, pitch) = layout.planes[0];
        let (w, h) = (layout.width, layout.height);
        if w == 0 || h == 0 || w > 512 || h > 512 || (pitch as u64) < w as u64 * 4 {
            return None;
        }
        let len = offset as usize + pitch as usize * h as usize;
        let fd = exported.fds[0].as_raw_fd();
        // SAFETY: read-only shared mapping of a DMA-BUF we own, `len` bytes
        // (bounded above); unmapped below on every path.
        let map = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, fd, 0) };
        if map == libc::MAP_FAILED {
            return None;
        }
        let mut sync = DmaBufSync { flags: DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ };
        let _ = ioctl(fd, DMA_BUF_IOCTL_SYNC, &mut sync);
        // One bulk copy into ordinary memory first: the buffer usually lives
        // in uncached VRAM, where the per-byte reads of `convert_cursor`
        // each cross the bus (measured ~2 s per cursor on an RX 6600, which
        // stalled the frame loop; a bulk copy uses wide loads).
        let mut bytes = vec![0u8; len];
        // SAFETY: the mapping is `len` bytes long and stays mapped until the
        // munmap below; `bytes` is a distinct allocation of the same length.
        unsafe { std::ptr::copy_nonoverlapping(map as *const u8, bytes.as_mut_ptr(), len) };
        let mut sync = DmaBufSync { flags: DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ };
        let _ = ioctl(fd, DMA_BUF_IOCTL_SYNC, &mut sync);
        // SAFETY: unmapping the mapping created above.
        unsafe { libc::munmap(map, len) };
        convert_cursor(&bytes[offset as usize..], w, h, pitch, layout.fourcc)
    }
}

/// Inode of a descriptor (DMA-BUF identity).
pub fn fd_inode(fd: &OwnedFd) -> u64 {
    // SAFETY: stat is plain old data, fstat fills it.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: valid descriptor and out-parameter.
    if unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } == 0 {
        st.st_ino
    } else {
        0
    }
}

/// Convert a 32-bit cursor (KMS planes use premultiplied alpha) into
/// straight-alpha RGBA, trimming fully transparent right/bottom margins (the
/// image origin stays where the plane is). Pure; `src` starts at plane 0.
pub fn convert_cursor(src: &[u8], width: u32, height: u32, pitch: u32, fourcc: u32) -> Option<CursorPicture> {
    let (has_alpha, bgr) = match fourcc {
        FOURCC_ARGB8888 => (true, true),
        FOURCC_XRGB8888 => (false, true),
        FOURCC_ABGR8888 => (true, false),
        FOURCC_XBGR8888 => (false, false),
        _ => return None,
    };
    let (w, h, pitch) = (width as usize, height as usize, pitch as usize);
    if src.len() < pitch * (h - 1) + w * 4 {
        return None;
    }
    let mut rgba = vec![0u8; w * h * 4];
    let (mut max_x, mut max_y) = (0usize, 0usize);
    let mut any = false;
    for y in 0..h {
        for x in 0..w {
            let s = &src[y * pitch + x * 4..y * pitch + x * 4 + 4];
            let (r, g, b) = if bgr { (s[2], s[1], s[0]) } else { (s[0], s[1], s[2]) };
            let a = if has_alpha { s[3] } else { 255 };
            let unpremul = |c: u8| -> u8 {
                if a == 0 || a == 255 {
                    c
                } else {
                    ((c as u32 * 255 + a as u32 / 2) / a as u32).min(255) as u8
                }
            };
            let d = &mut rgba[(y * w + x) * 4..(y * w + x) * 4 + 4];
            if a == 0 {
                d.copy_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            d.copy_from_slice(&[unpremul(r), unpremul(g), unpremul(b), a]);
            any = true;
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !any {
        return Some(CursorPicture { width: 0, height: 0, rgba: Vec::new() });
    }
    let (tw, th) = (max_x + 1, max_y + 1);
    let mut out = Vec::with_capacity(tw * th * 4);
    for y in 0..th {
        out.extend_from_slice(&rgba[y * w * 4..y * w * 4 + tw * 4]);
    }
    if tw as u32 > MAX_CURSOR_SIDE || th as u32 > MAX_CURSOR_SIDE {
        return None;
    }
    Some(CursorPicture { width: tw as u32, height: th as u32, rgba: out })
}

// --- Enumeration -------------------------------------------------------------

/// `/dev/dri/card*` nodes, sorted.
pub fn card_paths(dev_dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dev_dir)
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter(|e| {
                    let n = e.file_name();
                    let n = n.to_string_lossy();
                    n.starts_with("card") && n[4..].chars().all(|c| c.is_ascii_digit()) && n.len() > 4
                })
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default();
    paths.sort_by_key(|p| {
        p.file_name()
            .and_then(|n| n.to_string_lossy().trim_start_matches("card").parse::<u32>().ok())
            .unwrap_or(u32::MAX)
    });
    paths
}

/// Render node (`/dev/dri/renderD129`) of the GPU behind card `card`
/// (`card2`), from sysfs under `sysfs_root` (normally `/sys`).
pub fn render_node_for_card(sysfs_root: &Path, card: &str) -> Option<String> {
    if card.is_empty() || card.contains('/') || card.contains("..") {
        return None;
    }
    let dir = sysfs_root.join("class/drm").join(card).join("device/drm");
    let mut nodes: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("renderD") && n[7..].chars().all(|c| c.is_ascii_digit()))
        .collect();
    nodes.sort();
    nodes.first().map(|n| format!("/dev/dri/{}", n))
}

/// Display id: `(card minor << 32) | connector id`.
pub fn display_id(card_minor: u32, connector_id: u32) -> u64 {
    ((card_minor as u64) << 32) | connector_id as u64
}

fn truncate_name(mut s: String) -> String {
    if s.len() > MAX_NAME {
        let mut end = MAX_NAME;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        s.truncate(end);
    }
    s
}

/// Lit displays of one card: connected connectors that drive a CRTC with a
/// valid mode. A blanked CRTC (no framebuffer / inactive) is listed without
/// `DISPLAY_ACTIVE`.
pub fn card_displays(card: &Card) -> io::Result<Vec<WireDisplay>> {
    let res = card.resources()?;
    let mut out = Vec::new();
    for &conn_id in &res.connectors {
        let Ok(conn) = card.connector(conn_id) else { continue };
        if !conn.connected || conn.kind == DRM_MODE_CONNECTOR_WRITEBACK {
            continue;
        }
        let mut crtc_id = if conn.encoder_id != 0 { card.encoder_crtc(conn.encoder_id).unwrap_or(0) } else { 0 };
        if crtc_id == 0 && card.atomic {
            crtc_id = card
                .named_properties(conn_id, 0xc0c0_c0c0)
                .ok()
                .and_then(|p| p.get("CRTC_ID").copied())
                .unwrap_or(0) as u32;
        }
        if crtc_id == 0 {
            continue;
        }
        let Ok(crtc) = card.crtc(crtc_id) else { continue };
        if !crtc.mode_valid || crtc.width == 0 || crtc.height == 0 {
            continue;
        }
        let active = crtc.fb_id != 0 && card.crtc_active(crtc_id).unwrap_or(true);
        out.push(WireDisplay {
            id: display_id(card.minor, conn_id),
            card: truncate_name(card.name.clone()),
            connector_id: conn_id,
            crtc_id,
            name: truncate_name(conn.name()),
            x: 0,
            y: 0,
            width: crtc.width,
            height: crtc.height,
            refresh_mhz: crtc.refresh_mhz,
            flags: if active { DISPLAY_ACTIVE } else { 0 },
        });
    }
    out.sort_by_key(|d| (d.crtc_id, d.connector_id));
    Ok(out)
}

/// Result of sampling one display.
#[derive(Debug)]
pub enum Grab {
    Frame { fb: ExportedFb, crop: (u32, u32, u32, u32) },
    /// CRTC lit but without a framebuffer, or inactive (screen off).
    NoFramebuffer,
    /// CRTC gone or no longer has a mode.
    Gone,
}

/// Sample the framebuffer `crtc_id` scans out right now.
pub fn grab(card: &Card, crtc_id: u32) -> io::Result<Grab> {
    let crtc = match card.crtc(crtc_id) {
        Ok(c) => c,
        Err(e) if e.raw_os_error() == Some(libc::ENOENT) => return Ok(Grab::Gone),
        Err(e) => return Err(e),
    };
    if !crtc.mode_valid {
        return Ok(Grab::Gone);
    }
    if crtc.fb_id == 0 || card.crtc_active(crtc_id) == Some(false) {
        return Ok(Grab::NoFramebuffer);
    }
    let fb = card.export_fb(crtc.fb_id)?;
    let (fw, fh) = (fb.layout.width, fb.layout.height);
    let x = crtc.x.min(fw.saturating_sub(1));
    let y = crtc.y.min(fh.saturating_sub(1));
    let w = crtc.width.min(fw - x).max(1);
    let h = crtc.height.min(fh - y).max(1);
    Ok(Grab::Frame { fb, crop: (x, y, w, h) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes_match_the_kernel() {
        assert_eq!(std::mem::size_of::<CardRes>(), 64);
        assert_eq!(std::mem::size_of::<ModeInfo>(), 68);
        assert_eq!(std::mem::size_of::<ModeCrtc>(), 104);
        assert_eq!(std::mem::size_of::<GetEncoder>(), 20);
        assert_eq!(std::mem::size_of::<GetConnector>(), 80);
        assert_eq!(std::mem::size_of::<GetProperty>(), 64);
        assert_eq!(std::mem::size_of::<GetPlaneRes>(), 16);
        assert_eq!(std::mem::size_of::<GetPlane>(), 32);
        assert_eq!(std::mem::size_of::<ObjGetProps>(), 32);
        assert_eq!(std::mem::size_of::<FbCmd2>(), 104);
        assert_eq!(std::mem::size_of::<PrimeHandle>(), 12);
        assert_eq!(std::mem::size_of::<GemClose>(), 8);
        assert_eq!(std::mem::size_of::<WaitVblank>(), 24);
        assert_eq!(DRM_IOCTL_WAIT_VBLANK, 0xC018643A);
        // A couple of well-known request numbers (libdrm's values).
        assert_eq!(DRM_IOCTL_MODE_GETRESOURCES, 0xC04064A0);
        assert_eq!(DRM_IOCTL_MODE_GETCRTC, 0xC06864A1);
        assert_eq!(DRM_IOCTL_MODE_GETCONNECTOR, 0xC05064A7);
        assert_eq!(DRM_IOCTL_MODE_GETFB2, 0xC06864CE);
        assert_eq!(DRM_IOCTL_PRIME_HANDLE_TO_FD, 0xC00C642D);
        assert_eq!(DRM_IOCTL_SET_CLIENT_CAP, 0x4010640D);
        assert_eq!(DMA_BUF_IOCTL_SYNC, 0x40086200);
    }

    #[test]
    fn connector_names() {
        assert_eq!(connector_type_name(10), "DP");
        assert_eq!(connector_type_name(11), "HDMI-A");
        assert_eq!(connector_type_name(14), "eDP");
        assert_eq!(connector_type_name(999), "Unknown");
        let c = ConnectorInfo { kind: 11, kind_id: 2, ..Default::default() };
        assert_eq!(c.name(), "HDMI-A-2");
        assert_eq!(fourcc_str(FOURCC_XRGB8888), "XR24");
    }

    fn fake_sysfs(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("lunaris-sysfs-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (card, nodes) in [("card0", vec!["card0", "renderD130"]), ("card2", vec!["card2", "renderD129"]), ("card1", vec!["card1"])] {
            for n in nodes {
                std::fs::create_dir_all(root.join("class/drm").join(card).join("device/drm").join(n)).unwrap();
            }
        }
        root
    }

    #[test]
    fn render_node_mapping_from_sysfs() {
        let root = fake_sysfs("map");
        assert_eq!(render_node_for_card(&root, "card2").as_deref(), Some("/dev/dri/renderD129"));
        assert_eq!(render_node_for_card(&root, "card0").as_deref(), Some("/dev/dri/renderD130"));
        assert_eq!(render_node_for_card(&root, "card1"), None);
        assert_eq!(render_node_for_card(&root, "card9"), None);
        assert_eq!(render_node_for_card(&root, "../card2"), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn card_paths_are_sorted_numerically() {
        let dir = std::env::temp_dir().join(format!("lunaris-dri-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for n in ["card10", "card2", "renderD128", "card0", "cardX", "by-path"] {
            std::fs::write(dir.join(n), b"").unwrap();
        }
        let names: Vec<String> = card_paths(&dir)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["card0", "card2", "card10"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cursor_conversion_unpremultiplies_and_trims() {
        // 4x3 ARGB8888 (memory B,G,R,A), pitch 20 (padding), opaque red at
        // (0,0), half-transparent premultiplied white at (1,1), rest clear.
        let (w, h, pitch) = (4u32, 3u32, 20u32);
        let mut src = vec![0u8; (pitch * h) as usize];
        src[0..4].copy_from_slice(&[0, 0, 255, 255]);
        let o = (pitch + 4) as usize;
        src[o..o + 4].copy_from_slice(&[128, 128, 128, 128]);
        let pic = convert_cursor(&src, w, h, pitch, FOURCC_ARGB8888).unwrap();
        assert_eq!((pic.width, pic.height), (2, 2));
        assert_eq!(&pic.rgba[0..4], &[255, 0, 0, 255]);
        assert_eq!(&pic.rgba[12..16], &[255, 255, 255, 128]);
        // XRGB: opaque everywhere, nothing trimmed.
        let pic = convert_cursor(&src, w, h, pitch, FOURCC_XRGB8888).unwrap();
        assert_eq!((pic.width, pic.height), (4, 3));
        // Fully transparent: empty picture.
        let clear = vec![0u8; (pitch * h) as usize];
        let pic = convert_cursor(&clear, w, h, pitch, FOURCC_ARGB8888).unwrap();
        assert_eq!((pic.width, pic.height), (0, 0));
        // Unknown format / short buffer.
        assert!(convert_cursor(&src, w, h, pitch, u32::from_le_bytes(*b"NV12")).is_none());
        assert!(convert_cursor(&src[..10], w, h, pitch, FOURCC_ARGB8888).is_none());
    }

    #[test]
    fn display_ids() {
        assert_eq!(display_id(2, 448), (2 << 32) | 448);
    }
}
