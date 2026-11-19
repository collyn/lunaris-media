//! Wire protocol between the capture service (root, owns DRM) and the
//! unprivileged agent in the user's session.
//!
//! Transport: an `AF_UNIX` `SOCK_SEQPACKET` socket (see [`super::transport`]),
//! so every message is one packet and the file descriptors of a frame travel
//! with exactly that packet (`SCM_RIGHTS`).
//!
//! Every message is little-endian and starts with an 8-byte header:
//!
//! ```text
//! u32 total_len   (header included; must equal the packet length)
//! u16 type        (MessageType)
//! u16 flags       (0; anything else is refused)
//! ```
//!
//! Every variable-length field is bounded (see the `MAX_*` constants) and the
//! decoder refuses truncated input, trailing bytes and fd counts that do not
//! match the message, so a malformed packet can never make either side
//! allocate without bound or keep an unexpected descriptor.

use std::fmt;

/// Protocol version spoken by this build. Bump on any incompatible change.
pub const PROTOCOL_VERSION: u16 = 1;
/// `Hello` magic: "LNCS" (Lunaris capture service).
pub const MAGIC: u32 = u32::from_le_bytes(*b"LNCS");
/// Size of the common header.
pub const HEADER_LEN: usize = 8;
/// Largest packet either side sends or accepts.
pub const MAX_MESSAGE: usize = 192 * 1024;
/// Most displays one list may carry.
pub const MAX_DISPLAYS: usize = 16;
/// Longest connector name ("HDMI-A-2") / card name ("card2").
pub const MAX_NAME: usize = 32;
/// Most planes (and descriptors) one frame may carry (DRM's own limit).
pub const MAX_PLANES: usize = 4;
/// Longest error text.
pub const MAX_ERROR_MSG: usize = 256;
/// Largest cursor bitmap side; bigger cursors are sent without a picture.
pub const MAX_CURSOR_SIDE: u32 = 256;
/// Largest frame rate a client may ask for.
pub const MAX_FPS: u32 = 240;
/// Most frames a client may let the service have in flight.
pub const MAX_IN_FLIGHT: u32 = 4;

/// `WireDisplay::flags`: the CRTC currently scans out a framebuffer.
pub const DISPLAY_ACTIVE: u32 = 1;
/// `WireDisplay::flags`: `x`/`y` are the desktop position. KMS does not know
/// the compositor's layout, so the service never sets it; the consumer looks
/// the position up by connector name in the session.
pub const DISPLAY_POSITION_KNOWN: u32 = 2;

/// Message type codes.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Hello = 1,
    DisplayList = 2,
    Start = 3,
    Frame = 4,
    FrameAck = 5,
    Cursor = 6,
    DisplaysChanged = 7,
    Status = 8,
    Error = 9,
    Stop = 10,
    ListDisplays = 11,
}

impl MessageType {
    fn from_u16(v: u16) -> Option<Self> {
        Some(match v {
            1 => Self::Hello,
            2 => Self::DisplayList,
            3 => Self::Start,
            4 => Self::Frame,
            5 => Self::FrameAck,
            6 => Self::Cursor,
            7 => Self::DisplaysChanged,
            8 => Self::Status,
            9 => Self::Error,
            10 => Self::Stop,
            11 => Self::ListDisplays,
            _ => return None,
        })
    }
}

/// Display state reported with [`Message::Status`].
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusCode {
    /// Frames flow (sent after a pause/blank ends).
    Active = 1,
    /// The CRTC has no framebuffer or is inactive: the screen is off/blanked
    /// (DPMS, compositor idle). No frames until it comes back.
    NoFramebuffer = 2,
    /// The display disappeared (unplugged, mode set off). Streaming stopped.
    Gone = 3,
    /// The peer's user is no longer the active session on seat0 (user
    /// switch, greeter). No frames until it is active again.
    Paused = 4,
}

impl StatusCode {
    fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            1 => Self::Active,
            2 => Self::NoFramebuffer,
            3 => Self::Gone,
            4 => Self::Paused,
            _ => return None,
        })
    }
}

/// Error codes of [`Message::Error`].
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    Unauthorized = 1,
    BadVersion = 2,
    BadMessage = 3,
    NoSuchDisplay = 4,
    Busy = 5,
    Internal = 6,
}

impl ErrorCode {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            1 => Self::Unauthorized,
            2 => Self::BadVersion,
            3 => Self::BadMessage,
            4 => Self::NoSuchDisplay,
            5 => Self::Busy,
            6 => Self::Internal,
            _ => return None,
        })
    }
}

/// One display (one active CRTC + its connector).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireDisplay {
    /// `(card minor << 32) | connector id`: stable while the card stays.
    pub id: u64,
    /// DRM card node name, e.g. `card2`.
    pub card: String,
    pub connector_id: u32,
    pub crtc_id: u32,
    /// Connector name, e.g. `DP-2` (what compositors call the output).
    pub name: String,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    /// Refresh rate in millihertz.
    pub refresh_mhz: u32,
    /// `DISPLAY_*` bits.
    pub flags: u32,
}

/// One plane of a frame; `fd_index` names a descriptor of the packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WirePlane {
    pub fd_index: u32,
    pub offset: u32,
    pub stride: u32,
}

/// Frame description; the DMA-BUF descriptors ride along in the packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireFrame {
    pub seq: u64,
    pub display_id: u64,
    /// `CLOCK_MONOTONIC` microseconds when the framebuffer was sampled.
    pub timestamp_us: u64,
    /// Framebuffer size.
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub modifier: u64,
    /// Identity of the underlying buffer (DMA-BUF inode): equal ids are the
    /// same buffer, so consumers may cache imports.
    pub buffer_id: u64,
    /// Part of the framebuffer the CRTC shows.
    pub crop_x: u32,
    pub crop_y: u32,
    pub crop_w: u32,
    pub crop_h: u32,
    pub planes: Vec<WirePlane>,
}

impl WireFrame {
    /// Number of descriptors the packet must carry.
    pub fn fd_count(&self) -> usize {
        self.planes.iter().map(|p| p.fd_index as usize + 1).max().unwrap_or(0)
    }
}

/// Hardware cursor of one display, in that display's pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireCursor {
    pub display_id: u64,
    pub visible: bool,
    /// Top-left of the cursor image relative to the display.
    pub x: i32,
    pub y: i32,
    /// Hotspot inside the image (KMS does not know it: 0,0).
    pub hot_x: i32,
    pub hot_y: i32,
    pub width: u32,
    pub height: u32,
    /// Changes whenever the picture changes.
    pub generation: u32,
    /// RGBA, `width*height*4` bytes; only sent when `generation` changed.
    pub image: Option<Vec<u8>>,
}

/// All messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    /// Both directions: the client opens with it, the service answers.
    Hello { version: u16 },
    /// Service → client after `Hello`, and in answer to `ListDisplays`.
    DisplayList(Vec<WireDisplay>),
    /// Service → client when the set or modes of displays changed.
    DisplaysChanged(Vec<WireDisplay>),
    /// Client → service.
    ListDisplays,
    /// Client → service: stream `display_id` at up to `max_fps`, with at most
    /// `max_in_flight` frames not yet acknowledged.
    Start { display_id: u64, max_fps: u32, max_in_flight: u32 },
    /// Client → service: stop streaming (the connection stays).
    Stop,
    /// Service → client, with the plane descriptors.
    Frame(WireFrame),
    /// Client → service: the frame `seq` is done with (credit returns).
    FrameAck { seq: u64 },
    /// Service → client.
    Cursor(WireCursor),
    /// Service → client.
    Status { display_id: u64, code: StatusCode },
    /// Service → client; after `Unauthorized`/`BadVersion` the service closes.
    Error { code: ErrorCode, message: String },
}

impl Message {
    pub fn kind(&self) -> MessageType {
        match self {
            Message::Hello { .. } => MessageType::Hello,
            Message::DisplayList(_) => MessageType::DisplayList,
            Message::DisplaysChanged(_) => MessageType::DisplaysChanged,
            Message::ListDisplays => MessageType::ListDisplays,
            Message::Start { .. } => MessageType::Start,
            Message::Stop => MessageType::Stop,
            Message::Frame(_) => MessageType::Frame,
            Message::FrameAck { .. } => MessageType::FrameAck,
            Message::Cursor(_) => MessageType::Cursor,
            Message::Status { .. } => MessageType::Status,
            Message::Error { .. } => MessageType::Error,
        }
    }

    /// Descriptors that must travel with this message.
    pub fn fd_count(&self) -> usize {
        match self {
            Message::Frame(f) => f.fd_count(),
            _ => 0,
        }
    }
}

/// Why a packet was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtoError {
    Truncated,
    TooLong,
    LengthMismatch { header: usize, actual: usize },
    UnknownType(u16),
    BadFlags(u16),
    TrailingBytes(usize),
    Limit(&'static str),
    BadValue(&'static str),
    FdCount { expected: usize, got: usize },
}

impl fmt::Display for ProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtoError::Truncated => write!(f, "truncated message"),
            ProtoError::TooLong => write!(f, "message too long"),
            ProtoError::LengthMismatch { header, actual } => {
                write!(f, "length {} in header, {} received", header, actual)
            }
            ProtoError::UnknownType(t) => write!(f, "unknown message type {}", t),
            ProtoError::BadFlags(v) => write!(f, "unsupported flags 0x{:04x}", v),
            ProtoError::TrailingBytes(n) => write!(f, "{} trailing bytes", n),
            ProtoError::Limit(what) => write!(f, "limit exceeded: {}", what),
            ProtoError::BadValue(what) => write!(f, "invalid value: {}", what),
            ProtoError::FdCount { expected, got } => {
                write!(f, "expected {} descriptors, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for ProtoError {}

// --- Encoding ----------------------------------------------------------------

struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str, max: usize, what: &'static str) -> Result<(), ProtoError> {
        if s.len() > max {
            return Err(ProtoError::Limit(what));
        }
        if max > u8::MAX as usize {
            self.u16(s.len() as u16);
        } else {
            self.u8(s.len() as u8);
        }
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

fn encode_displays(w: &mut Writer, list: &[WireDisplay]) -> Result<(), ProtoError> {
    if list.len() > MAX_DISPLAYS {
        return Err(ProtoError::Limit("displays"));
    }
    w.u32(list.len() as u32);
    for d in list {
        w.u64(d.id);
        w.str(&d.card, MAX_NAME, "card name")?;
        w.u32(d.connector_id);
        w.u32(d.crtc_id);
        w.str(&d.name, MAX_NAME, "display name")?;
        w.i32(d.x);
        w.i32(d.y);
        w.u32(d.width);
        w.u32(d.height);
        w.u32(d.refresh_mhz);
        w.u32(d.flags);
    }
    Ok(())
}

/// Serialize `msg` into one packet.
pub fn encode(msg: &Message) -> Result<Vec<u8>, ProtoError> {
    let mut w = Writer(Vec::with_capacity(64));
    w.u32(0); // length, patched below
    w.u16(msg.kind() as u16);
    w.u16(0);
    match msg {
        Message::Hello { version } => {
            w.u32(MAGIC);
            w.u16(*version);
            w.u16(0);
        }
        Message::DisplayList(list) | Message::DisplaysChanged(list) => encode_displays(&mut w, list)?,
        Message::ListDisplays | Message::Stop => {}
        Message::Start { display_id, max_fps, max_in_flight } => {
            w.u64(*display_id);
            w.u32(*max_fps);
            w.u32(*max_in_flight);
        }
        Message::Frame(f) => {
            if f.planes.is_empty() || f.planes.len() > MAX_PLANES {
                return Err(ProtoError::Limit("planes"));
            }
            if f.fd_count() > MAX_PLANES {
                return Err(ProtoError::Limit("descriptors"));
            }
            w.u64(f.seq);
            w.u64(f.display_id);
            w.u64(f.timestamp_us);
            w.u32(f.width);
            w.u32(f.height);
            w.u32(f.fourcc);
            w.u64(f.modifier);
            w.u64(f.buffer_id);
            w.u32(f.crop_x);
            w.u32(f.crop_y);
            w.u32(f.crop_w);
            w.u32(f.crop_h);
            w.u32(f.planes.len() as u32);
            for p in &f.planes {
                w.u32(p.fd_index);
                w.u32(p.offset);
                w.u32(p.stride);
            }
        }
        Message::FrameAck { seq } => w.u64(*seq),
        Message::Cursor(c) => {
            w.u64(c.display_id);
            w.u8(c.visible as u8);
            w.u8(c.image.is_some() as u8);
            w.u16(0);
            w.i32(c.x);
            w.i32(c.y);
            w.i32(c.hot_x);
            w.i32(c.hot_y);
            w.u32(c.width);
            w.u32(c.height);
            w.u32(c.generation);
            if let Some(img) = &c.image {
                if c.width > MAX_CURSOR_SIDE || c.height > MAX_CURSOR_SIDE {
                    return Err(ProtoError::Limit("cursor size"));
                }
                if img.len() != (c.width * c.height * 4) as usize {
                    return Err(ProtoError::BadValue("cursor image length"));
                }
                w.0.extend_from_slice(img);
            }
        }
        Message::Status { display_id, code } => {
            w.u64(*display_id);
            w.u32(*code as u32);
        }
        Message::Error { code, message } => {
            w.u32(*code as u32);
            // Truncate on a char boundary rather than refusing to report.
            let mut end = message.len().min(MAX_ERROR_MSG);
            while !message.is_char_boundary(end) {
                end -= 1;
            }
            w.str(&message[..end], MAX_ERROR_MSG, "error message")?;
        }
    }
    if w.0.len() > MAX_MESSAGE {
        return Err(ProtoError::TooLong);
    }
    let len = w.0.len() as u32;
    w.0[..4].copy_from_slice(&len.to_le_bytes());
    Ok(w.0)
}

// --- Decoding ----------------------------------------------------------------

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ProtoError> {
        let end = self.pos.checked_add(n).ok_or(ProtoError::Truncated)?;
        if end > self.buf.len() {
            return Err(ProtoError::Truncated);
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, ProtoError> {
        Ok(self.take(1)?[0])
    }
    fn bool(&mut self, what: &'static str) -> Result<bool, ProtoError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(ProtoError::BadValue(what)),
        }
    }
    fn u16(&mut self) -> Result<u16, ProtoError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, ProtoError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, ProtoError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, ProtoError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self, max: usize, what: &'static str) -> Result<String, ProtoError> {
        let len = if max > u8::MAX as usize { self.u16()? as usize } else { self.u8()? as usize };
        if len > max {
            return Err(ProtoError::Limit(what));
        }
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| ProtoError::BadValue(what))
    }
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
}

fn decode_displays(r: &mut Reader) -> Result<Vec<WireDisplay>, ProtoError> {
    let count = r.u32()? as usize;
    if count > MAX_DISPLAYS {
        return Err(ProtoError::Limit("displays"));
    }
    let mut list = Vec::with_capacity(count);
    for _ in 0..count {
        list.push(WireDisplay {
            id: r.u64()?,
            card: r.str(MAX_NAME, "card name")?,
            connector_id: r.u32()?,
            crtc_id: r.u32()?,
            name: r.str(MAX_NAME, "display name")?,
            x: r.i32()?,
            y: r.i32()?,
            width: r.u32()?,
            height: r.u32()?,
            refresh_mhz: r.u32()?,
            flags: r.u32()?,
        });
    }
    Ok(list)
}

/// Parse one packet that arrived with `fds` descriptors.
pub fn decode(buf: &[u8], fds: usize) -> Result<Message, ProtoError> {
    if buf.len() > MAX_MESSAGE {
        return Err(ProtoError::TooLong);
    }
    let mut r = Reader { buf, pos: 0 };
    let len = r.u32()? as usize;
    let ty = r.u16()?;
    let flags = r.u16()?;
    if len != buf.len() {
        return Err(ProtoError::LengthMismatch { header: len, actual: buf.len() });
    }
    if flags != 0 {
        return Err(ProtoError::BadFlags(flags));
    }
    let ty = MessageType::from_u16(ty).ok_or(ProtoError::UnknownType(ty))?;
    let msg = match ty {
        MessageType::Hello => {
            if r.u32()? != MAGIC {
                return Err(ProtoError::BadValue("magic"));
            }
            let version = r.u16()?;
            if r.u16()? != 0 {
                return Err(ProtoError::BadValue("reserved"));
            }
            Message::Hello { version }
        }
        MessageType::DisplayList => Message::DisplayList(decode_displays(&mut r)?),
        MessageType::DisplaysChanged => Message::DisplaysChanged(decode_displays(&mut r)?),
        MessageType::ListDisplays => Message::ListDisplays,
        MessageType::Stop => Message::Stop,
        MessageType::Start => {
            let display_id = r.u64()?;
            let max_fps = r.u32()?;
            let max_in_flight = r.u32()?;
            if max_fps == 0 || max_fps > MAX_FPS {
                return Err(ProtoError::BadValue("max_fps"));
            }
            if max_in_flight == 0 || max_in_flight > MAX_IN_FLIGHT {
                return Err(ProtoError::BadValue("max_in_flight"));
            }
            Message::Start { display_id, max_fps, max_in_flight }
        }
        MessageType::Frame => {
            let seq = r.u64()?;
            let display_id = r.u64()?;
            let timestamp_us = r.u64()?;
            let width = r.u32()?;
            let height = r.u32()?;
            let fourcc = r.u32()?;
            let modifier = r.u64()?;
            let buffer_id = r.u64()?;
            let crop_x = r.u32()?;
            let crop_y = r.u32()?;
            let crop_w = r.u32()?;
            let crop_h = r.u32()?;
            let n = r.u32()? as usize;
            if n == 0 || n > MAX_PLANES {
                return Err(ProtoError::Limit("planes"));
            }
            let mut planes = Vec::with_capacity(n);
            for _ in 0..n {
                let p = WirePlane { fd_index: r.u32()?, offset: r.u32()?, stride: r.u32()? };
                if p.fd_index as usize >= MAX_PLANES {
                    return Err(ProtoError::Limit("descriptor index"));
                }
                planes.push(p);
            }
            if width == 0 || height == 0 || width > 16384 || height > 16384 {
                return Err(ProtoError::BadValue("frame size"));
            }
            if crop_w == 0
                || crop_h == 0
                || crop_x.checked_add(crop_w).is_none_or(|e| e > width)
                || crop_y.checked_add(crop_h).is_none_or(|e| e > height)
            {
                return Err(ProtoError::BadValue("crop"));
            }
            Message::Frame(WireFrame {
                seq,
                display_id,
                timestamp_us,
                width,
                height,
                fourcc,
                modifier,
                buffer_id,
                crop_x,
                crop_y,
                crop_w,
                crop_h,
                planes,
            })
        }
        MessageType::FrameAck => Message::FrameAck { seq: r.u64()? },
        MessageType::Cursor => {
            let display_id = r.u64()?;
            let visible = r.bool("visible")?;
            let has_image = r.bool("has_image")?;
            if r.u16()? != 0 {
                return Err(ProtoError::BadValue("reserved"));
            }
            let x = r.i32()?;
            let y = r.i32()?;
            let hot_x = r.i32()?;
            let hot_y = r.i32()?;
            let width = r.u32()?;
            let height = r.u32()?;
            let generation = r.u32()?;
            let image = if has_image {
                if width == 0 || height == 0 || width > MAX_CURSOR_SIDE || height > MAX_CURSOR_SIDE {
                    return Err(ProtoError::Limit("cursor size"));
                }
                Some(r.take((width * height * 4) as usize)?.to_vec())
            } else {
                None
            };
            Message::Cursor(WireCursor {
                display_id,
                visible,
                x,
                y,
                hot_x,
                hot_y,
                width,
                height,
                generation,
                image,
            })
        }
        MessageType::Status => {
            let display_id = r.u64()?;
            let code = StatusCode::from_u32(r.u32()?).ok_or(ProtoError::BadValue("status"))?;
            Message::Status { display_id, code }
        }
        MessageType::Error => {
            let code = ErrorCode::from_u32(r.u32()?).ok_or(ProtoError::BadValue("error code"))?;
            let message = r.str(MAX_ERROR_MSG, "error message")?;
            Message::Error { code, message }
        }
    };
    if r.remaining() != 0 {
        return Err(ProtoError::TrailingBytes(r.remaining()));
    }
    let expected = msg.fd_count();
    if expected != fds {
        return Err(ProtoError::FdCount { expected, got: fds });
    }
    Ok(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn display(i: u32) -> WireDisplay {
        WireDisplay {
            id: (2u64 << 32) | (448 + i) as u64,
            card: "card2".into(),
            connector_id: 448 + i,
            crtc_id: 80 + i,
            name: format!("DP-{}", i),
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
            refresh_mhz: 60_000,
            flags: DISPLAY_ACTIVE,
        }
    }

    fn frame(planes: usize) -> WireFrame {
        WireFrame {
            seq: 7,
            display_id: 42,
            timestamp_us: 123_456,
            width: 1920,
            height: 1080,
            fourcc: u32::from_le_bytes(*b"XR24"),
            modifier: 0x0200_0000_1234_5678,
            buffer_id: 99,
            crop_x: 0,
            crop_y: 0,
            crop_w: 1920,
            crop_h: 1080,
            planes: (0..planes)
                .map(|i| WirePlane { fd_index: i as u32, offset: i as u32 * 4096, stride: 7680 })
                .collect(),
        }
    }

    fn all_messages() -> Vec<Message> {
        vec![
            Message::Hello { version: PROTOCOL_VERSION },
            Message::DisplayList(vec![display(1), display(2)]),
            Message::DisplaysChanged(vec![]),
            Message::ListDisplays,
            Message::Start { display_id: 5, max_fps: 60, max_in_flight: 2 },
            Message::Stop,
            Message::Frame(frame(1)),
            Message::Frame(frame(3)),
            Message::FrameAck { seq: u64::MAX },
            Message::Cursor(WireCursor {
                display_id: 1,
                visible: true,
                x: -5,
                y: 10,
                hot_x: 0,
                hot_y: 0,
                width: 2,
                height: 2,
                generation: 3,
                image: Some(vec![0xAB; 16]),
            }),
            Message::Cursor(WireCursor {
                display_id: 1,
                visible: false,
                x: 0,
                y: 0,
                hot_x: 0,
                hot_y: 0,
                width: 0,
                height: 0,
                generation: 3,
                image: None,
            }),
            Message::Status { display_id: 9, code: StatusCode::NoFramebuffer },
            Message::Error { code: ErrorCode::Unauthorized, message: "not the active session".into() },
        ]
    }

    #[test]
    fn round_trip_every_message() {
        for msg in all_messages() {
            let bytes = encode(&msg).unwrap();
            assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize, bytes.len());
            let back = decode(&bytes, msg.fd_count()).unwrap();
            assert_eq!(back, msg);
        }
    }

    #[test]
    fn every_truncation_is_refused() {
        for msg in all_messages() {
            let bytes = encode(&msg).unwrap();
            for cut in 0..bytes.len() {
                // Keep the header's length honest for the shorter packet so
                // the body parser itself has to notice the truncation.
                let mut short = bytes[..cut].to_vec();
                if cut >= 4 {
                    short[..4].copy_from_slice(&(cut as u32).to_le_bytes());
                }
                assert!(decode(&short, msg.fd_count()).is_err(), "{:?} cut at {}", msg.kind(), cut);
            }
        }
    }

    #[test]
    fn trailing_bytes_and_length_mismatch_are_refused() {
        let mut bytes = encode(&Message::Stop).unwrap();
        bytes.push(0);
        assert!(matches!(decode(&bytes, 0), Err(ProtoError::LengthMismatch { .. })));
        let len = bytes.len() as u32;
        bytes[..4].copy_from_slice(&len.to_le_bytes());
        assert_eq!(decode(&bytes, 0), Err(ProtoError::TrailingBytes(1)));
    }

    #[test]
    fn fd_count_must_match() {
        let msg = Message::Frame(frame(2));
        let bytes = encode(&msg).unwrap();
        assert!(decode(&bytes, 2).is_ok());
        assert_eq!(decode(&bytes, 1), Err(ProtoError::FdCount { expected: 2, got: 1 }));
        assert_eq!(decode(&bytes, 3), Err(ProtoError::FdCount { expected: 2, got: 3 }));
        // Descriptors on a message that carries none are refused too.
        let stop = encode(&Message::Stop).unwrap();
        assert_eq!(decode(&stop, 1), Err(ProtoError::FdCount { expected: 0, got: 1 }));
    }

    #[test]
    fn limits_are_enforced() {
        let many: Vec<_> = (0..MAX_DISPLAYS as u32 + 1).map(display).collect();
        assert_eq!(encode(&Message::DisplayList(many)), Err(ProtoError::Limit("displays")));
        let mut long = display(1);
        long.name = "x".repeat(MAX_NAME + 1);
        assert!(encode(&Message::DisplayList(vec![long])).is_err());
        assert!(encode(&Message::Frame(frame(MAX_PLANES + 1))).is_err());
        assert!(encode(&Message::Frame(frame(0))).is_err());

        // A forged count beyond the limit is refused before allocating.
        let mut bytes = encode(&Message::DisplayList(vec![])).unwrap();
        let n = bytes.len() - 4;
        bytes[n..].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode(&bytes, 0), Err(ProtoError::Limit("displays")));

        // Oversized cursors.
        let big = Message::Cursor(WireCursor {
            display_id: 1,
            visible: true,
            x: 0,
            y: 0,
            hot_x: 0,
            hot_y: 0,
            width: MAX_CURSOR_SIDE + 1,
            height: 1,
            generation: 1,
            image: Some(vec![0; (MAX_CURSOR_SIDE as usize + 1) * 4]),
        });
        assert!(encode(&big).is_err());

        // Start values out of range.
        for (fps, inflight) in [(0, 1), (MAX_FPS + 1, 1), (60, 0), (60, MAX_IN_FLIGHT + 1)] {
            let mut bytes = encode(&Message::Start { display_id: 1, max_fps: 60, max_in_flight: 1 }).unwrap();
            bytes[16..20].copy_from_slice(&fps.to_le_bytes());
            bytes[20..24].copy_from_slice(&(inflight as u32).to_le_bytes());
            assert!(decode(&bytes, 0).is_err());
        }

        let huge = vec![0u8; MAX_MESSAGE + 1];
        assert_eq!(decode(&huge, 0), Err(ProtoError::TooLong));
    }

    #[test]
    fn bad_header_fields_are_refused() {
        let mut bytes = encode(&Message::Stop).unwrap();
        bytes[4..6].copy_from_slice(&999u16.to_le_bytes());
        assert_eq!(decode(&bytes, 0), Err(ProtoError::UnknownType(999)));
        let mut bytes = encode(&Message::Stop).unwrap();
        bytes[6..8].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(decode(&bytes, 0), Err(ProtoError::BadFlags(1)));
        let mut bytes = encode(&Message::Hello { version: 1 }).unwrap();
        bytes[8] ^= 0xFF;
        assert_eq!(decode(&bytes, 0), Err(ProtoError::BadValue("magic")));
    }

    #[test]
    fn crop_outside_the_frame_is_refused() {
        let mut f = frame(1);
        f.crop_x = 1;
        let bytes = encode(&Message::Frame(f)).unwrap();
        assert_eq!(decode(&bytes, 1), Err(ProtoError::BadValue("crop")));
    }

    #[test]
    fn long_error_text_is_truncated_not_refused() {
        let msg = Message::Error { code: ErrorCode::Internal, message: "é".repeat(400) };
        let bytes = encode(&msg).unwrap();
        match decode(&bytes, 0).unwrap() {
            Message::Error { message, .. } => assert!(message.len() <= MAX_ERROR_MSG),
            other => panic!("{:?}", other),
        }
    }

    /// Cheap deterministic fuzz: random mutations of valid packets must never
    /// panic, and anything accepted must re-encode to the same bytes.
    #[test]
    fn mutations_never_panic() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for msg in all_messages() {
            let base = encode(&msg).unwrap();
            for _ in 0..2000 {
                let mut b = base.clone();
                let flips = 1 + (next() % 4) as usize;
                for _ in 0..flips {
                    let i = (next() as usize) % b.len();
                    b[i] = next() as u8;
                }
                let fds = (next() % 5) as usize;
                if let Ok(decoded) = decode(&b, fds) {
                    assert_eq!(encode(&decoded).unwrap(), b);
                }
            }
        }
    }
}
