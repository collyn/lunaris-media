//! DRM/KMS screen capture backend — zero-copy GPU capture.
//!
//! Captures the active display framebuffer via DRM (Direct Rendering Manager)
//! and exports it as a DMA-BUF file descriptor. This avoids any GPU→CPU copies
//! because the framebuffer stays in GPU VRAM.
//!
//! # Pipeline
//! ```text
//! /dev/dri/card1
//!   → drmModeGetResources → list CRTCs
//!   → drmModeGetCrtc → active framebuffer ID
//!   → DRM_IOCTL_MODE_GETFB2 → GEM handles + format + modifier
//!   → DRM_IOCTL_PRIME_HANDLE_TO_FD → DMA-BUF fd
//!   → GpuBuffer::DmaBuf { fd, stride, modifier, fourcc, ... }
//! ```
//!
//! # Requirements
//! - `CAP_SYS_ADMIN` or root (for `DRM_IOCTL_MODE_GETFB2`)
//! - DRM-capable GPU driver (Intel i915, AMD amdgpu, NVIDIA with DRM support)

use std::os::unix::io::RawFd;
use std::time::Instant;

use crate::capture::gpu_buffer::GpuBuffer;
use crate::capture::{CapturedFrame, ScreenCapture};
use crate::error::MediaError;
use crate::types::*;

// ---------------------------------------------------------------------------
// DRM ioctl number calculation
// ---------------------------------------------------------------------------
//
// Linux ioctl encoding: direction(2) | size(14) | type(8) | nr(8)
//
// _IOW(type, nr, size)  = (1 << 30) | (size << 16) | (type << 8) | nr
// _IOR(type, nr, size)  = (2 << 30) | (size << 16) | (type << 8) | nr
// _IOWR(type, nr, size) = (3 << 30) | (size << 16) | (type << 8) | nr

const fn iowr(ty: u32, nr: u32, size: u32) -> libc::c_ulong {
    ((3u32 << 30) | (size << 16) | (ty << 8) | nr) as libc::c_ulong
}

const fn iow(ty: u32, nr: u32, size: u32) -> libc::c_ulong {
    ((1u32 << 30) | (size << 16) | (ty << 8) | nr) as libc::c_ulong
}

const DRM_IOCTL_TYPE: u32 = b'd' as u32;

// drm_mode_card_res: 8*4 (u64 ptrs) + 8*4 (u32 counts) + 4*4 (min/max) = 32 + 32 + 16 = 80
const DRM_IOCTL_MODE_GETRESOURCES: libc::c_ulong = iowr(
    DRM_IOCTL_TYPE,
    0xA0,
    std::mem::size_of::<DrmModeCardRes>() as u32,
);

// drm_mode_crtc: 8 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + sizeof(drm_mode_modeinfo) = 32 + 68 = 100
// Actually we compute from struct definition below.
const DRM_IOCTL_MODE_GETCRTC: libc::c_ulong = iowr(
    DRM_IOCTL_TYPE,
    0xA1,
    std::mem::size_of::<DrmModeCrtc>() as u32,
);

// drm_mode_fb_cmd2: 4+4+4+4+4 + 4*4 + 4*4 + 4*4 + 8*4 = 20 + 16 + 16 + 16 + 32 = 100
// But the actual kernel struct size may differ; we compute from our definition.
const DRM_IOCTL_MODE_GETFB2: libc::c_ulong = iowr(
    DRM_IOCTL_TYPE,
    0xCE,
    std::mem::size_of::<DrmModeFbCmd2>() as u32,
);

// drm_prime_handle: 4 + 4 + 4 = 12
const DRM_IOCTL_PRIME_HANDLE_TO_FD: libc::c_ulong = iowr(
    DRM_IOCTL_TYPE,
    0x2D,
    std::mem::size_of::<DrmPrimeHandle>() as u32,
);

// drm_gem_close: 4 + 4 = 8
const DRM_IOCTL_GEM_CLOSE: libc::c_ulong = iow(
    DRM_IOCTL_TYPE,
    0x09,
    std::mem::size_of::<DrmGemClose>() as u32,
);

/// DRM_CLOEXEC flag for prime handle export.
const DRM_CLOEXEC: u32 = 0x02;
/// DRM_RDWR flag for prime handle export.
const DRM_RDWR: u32 = 0x02;

// ---------------------------------------------------------------------------
// DRM kernel structs (matching linux/drm.h / drm_mode.h layout exactly)
// ---------------------------------------------------------------------------

/// `struct drm_mode_card_res` — returned by `DRM_IOCTL_MODE_GETRESOURCES`.
#[repr(C)]
#[derive(Debug, Default)]
struct DrmModeCardRes {
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

/// `struct drm_mode_modeinfo` — embedded in CRTC and connector structs.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
struct DrmModeModeinfo {
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
    name: [u8; 32], // DRM_DISPLAY_MODE_LEN
}

/// `struct drm_mode_crtc` — returned by `DRM_IOCTL_MODE_GETCRTC`.
#[repr(C)]
#[derive(Debug, Default)]
struct DrmModeCrtc {
    set_connectors_ptr: u64,
    count_connectors: u32,
    crtc_id: u32,
    fb_id: u32,
    x: u32,
    y: u32,
    gamma_size: u32,
    mode_valid: u32,
    mode: DrmModeModeinfo,
}

/// `struct drm_mode_fb_cmd2` — returned by `DRM_IOCTL_MODE_GETFB2`.
#[repr(C)]
#[derive(Debug, Default)]
struct DrmModeFbCmd2 {
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

/// `struct drm_prime_handle` — used by `DRM_IOCTL_PRIME_HANDLE_TO_FD`.
#[repr(C)]
#[derive(Debug, Default)]
struct DrmPrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}

/// `struct drm_gem_close` — used by `DRM_IOCTL_GEM_CLOSE`.
#[repr(C)]
#[derive(Debug, Default)]
struct DrmGemClose {
    handle: u32,
    pad: u32,
}

// ---------------------------------------------------------------------------
// DrmCapture
// ---------------------------------------------------------------------------

/// DRM/KMS screen capture backend.
///
/// Opens the DRM device, locates the active CRTC, and captures framebuffers
/// as DMA-BUF file descriptors for true zero-copy GPU capture.
pub struct DrmCapture {
    /// File descriptor for the opened `/dev/dri/cardN` device.
    drm_fd: RawFd,
    /// ID of the active CRTC (the one driving a display).
    crtc_id: u32,
    /// Display width from the active CRTC mode.
    width: u32,
    /// Display height from the active CRTC mode.
    height: u32,
    /// Refresh rate from the active CRTC mode.
    refresh_rate: u32,
    /// Configured capture frame rate.
    fps: u32,
    /// Whether capture is currently running.
    capturing: bool,
    /// Timestamp of the last captured frame (for frame pacing).
    last_frame_time: Instant,
}

// Safety: DRM file descriptors are process-global resources. The capture
// pipeline guarantees single-threaded access to the DrmCapture instance,
// and file descriptors are safe to send between threads.
unsafe impl Send for DrmCapture {}
unsafe impl Sync for DrmCapture {}

impl DrmCapture {
    /// Device paths to try, in order of preference.
    const DRM_DEVICES: &'static [&'static str] = &["/dev/dri/card1", "/dev/dri/card0"];

    /// Creates a new DRM capture backend.
    ///
    /// Opens the DRM device, enumerates CRTCs, finds the active display, and
    /// verifies that `DRM_IOCTL_MODE_GETFB2` works (requires CAP_SYS_ADMIN).
    pub fn new() -> Result<Self, MediaError> {
        let drm_fd = Self::open_drm_device()?;

        // Find the active CRTC (the one actually driving a display).
        let (crtc_id, width, height, refresh_rate) = match Self::find_active_crtc(drm_fd) {
            Ok(info) => info,
            Err(e) => {
                // SAFETY: drm_fd is a valid file descriptor we just opened.
                unsafe {
                    libc::close(drm_fd);
                }
                return Err(e);
            }
        };

        // Verify GETFB2 works (this is the ioctl that requires CAP_SYS_ADMIN).
        if let Err(e) = Self::verify_getfb2(drm_fd, crtc_id) {
            // SAFETY: drm_fd is a valid file descriptor we just opened.
            unsafe {
                libc::close(drm_fd);
            }
            return Err(e);
        }

        log::info!(
            "DRM capture initialized: CRTC {} active at {}x{} @{}Hz",
            crtc_id,
            width,
            height,
            refresh_rate
        );

        Ok(Self {
            drm_fd,
            crtc_id,
            width,
            height,
            refresh_rate,
            fps: 60,
            capturing: false,
            last_frame_time: Instant::now(),
        })
    }

    /// Try to open a DRM device node, attempting card1 first then card0.
    fn open_drm_device() -> Result<RawFd, MediaError> {
        for device_path in Self::DRM_DEVICES {
            log::info!("Trying DRM device: {}", device_path);

            let c_path = std::ffi::CString::new(*device_path)
                .map_err(|_| MediaError::CaptureError("Invalid device path".into()))?;

            // SAFETY: c_path is a valid null-terminated C string. O_RDWR | O_CLOEXEC
            // are standard flags for opening DRM device nodes.
            let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };

            if fd >= 0 {
                log::info!("Opened DRM device: {} (fd={})", device_path, fd);
                return Ok(fd);
            }

            let err = std::io::Error::last_os_error();
            log::warn!("Failed to open {}: {}", device_path, err);
        }

        Err(MediaError::CaptureError(
            "No DRM device available. Check /dev/dri/card* permissions or run as root.".into(),
        ))
    }

    /// Enumerate CRTCs via `DRM_IOCTL_MODE_GETRESOURCES` and find the active one.
    ///
    /// Returns `(crtc_id, width, height, refresh_rate)`.
    fn find_active_crtc(drm_fd: RawFd) -> Result<(u32, u32, u32, u32), MediaError> {
        // --- First call: get counts ---
        let mut res = DrmModeCardRes::default();

        // SAFETY: drm_fd is a valid DRM file descriptor. res is a properly
        // initialized struct matching the kernel's expected layout. The ioctl
        // populates count fields when pointer fields are zero/null.
        let ret = unsafe { libc::ioctl(drm_fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            return Err(MediaError::CaptureError(format!(
                "DRM_IOCTL_MODE_GETRESOURCES (count) failed: {}",
                err
            )));
        }

        if res.count_crtcs == 0 {
            return Err(MediaError::NoDisplayFound);
        }

        log::info!(
            "DRM resources: {} CRTCs, {} connectors, {} encoders, {} FBs",
            res.count_crtcs,
            res.count_connectors,
            res.count_encoders,
            res.count_fbs
        );

        // --- Second call: fetch CRTC IDs ---
        let mut crtc_ids = vec![0u32; res.count_crtcs as usize];
        res.crtc_id_ptr = crtc_ids.as_mut_ptr() as u64;

        // We also need to provide buffers for the other resource types to avoid
        // the kernel overwriting count fields. Allocate minimal buffers.
        let mut fb_ids = vec![0u32; res.count_fbs as usize];
        let mut connector_ids = vec![0u32; res.count_connectors as usize];
        let mut encoder_ids = vec![0u32; res.count_encoders as usize];
        res.fb_id_ptr = fb_ids.as_mut_ptr() as u64;
        res.connector_id_ptr = connector_ids.as_mut_ptr() as u64;
        res.encoder_id_ptr = encoder_ids.as_mut_ptr() as u64;

        // SAFETY: All buffers are allocated with sufficient size matching the
        // counts from the first ioctl call. Pointers are valid for the
        // duration of this call.
        let ret = unsafe { libc::ioctl(drm_fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            return Err(MediaError::CaptureError(format!(
                "DRM_IOCTL_MODE_GETRESOURCES (fetch) failed: {}",
                err
            )));
        }

        // Keep buffers alive until after the ioctl returns.
        drop(fb_ids);
        drop(connector_ids);
        drop(encoder_ids);

        // --- Walk CRTCs to find the active one ---
        for &crtc_id in &crtc_ids {
            let mut crtc = DrmModeCrtc::default();
            crtc.crtc_id = crtc_id;

            // SAFETY: crtc is properly initialized with crtc_id set. The ioctl
            // fills in the remaining fields including fb_id and mode.
            let ret = unsafe { libc::ioctl(drm_fd, DRM_IOCTL_MODE_GETCRTC, &mut crtc) };
            if ret < 0 {
                let err = std::io::Error::last_os_error();
                log::warn!(
                    "DRM_IOCTL_MODE_GETCRTC for CRTC {} failed: {}",
                    crtc_id,
                    err
                );
                continue;
            }

            log::info!(
                "CRTC {}: fb_id={} mode_valid={} {}x{} @{}Hz",
                crtc_id,
                crtc.fb_id,
                crtc.mode_valid,
                crtc.mode.hdisplay,
                crtc.mode.vdisplay,
                crtc.mode.vrefresh
            );

            if crtc.fb_id != 0 && crtc.mode_valid != 0 {
                let width = crtc.mode.hdisplay as u32;
                let height = crtc.mode.vdisplay as u32;
                let vrefresh = if crtc.mode.vrefresh > 0 {
                    crtc.mode.vrefresh
                } else {
                    60 // Sensible default if driver doesn't report refresh
                };
                return Ok((crtc_id, width, height, vrefresh));
            }
        }

        Err(MediaError::NoDisplayFound)
    }

    /// Verify that `DRM_IOCTL_MODE_GETFB2` works for the active CRTC's framebuffer.
    ///
    /// This ioctl requires `CAP_SYS_ADMIN` on most kernels, so we probe it early
    /// to give a clear error message.
    fn verify_getfb2(drm_fd: RawFd, crtc_id: u32) -> Result<(), MediaError> {
        // Get the current fb_id from the CRTC.
        let mut crtc = DrmModeCrtc::default();
        crtc.crtc_id = crtc_id;

        // SAFETY: crtc is properly initialized. drm_fd is valid.
        let ret = unsafe { libc::ioctl(drm_fd, DRM_IOCTL_MODE_GETCRTC, &mut crtc) };
        if ret < 0 || crtc.fb_id == 0 {
            return Err(MediaError::CaptureError(
                "Cannot read active framebuffer from CRTC".into(),
            ));
        }

        // Try GETFB2 to verify permissions.
        let mut fb2 = DrmModeFbCmd2::default();
        fb2.fb_id = crtc.fb_id;

        // SAFETY: fb2 is properly initialized with a valid fb_id. drm_fd is valid.
        let ret = unsafe { libc::ioctl(drm_fd, DRM_IOCTL_MODE_GETFB2, &mut fb2) };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EACCES) {
                return Err(MediaError::PermissionDenied(
                    "DRM_IOCTL_MODE_GETFB2 requires CAP_SYS_ADMIN. \
                     Run as root or set: sudo setcap cap_sys_admin+ep <binary>"
                        .into(),
                ));
            }
            return Err(MediaError::CaptureError(format!(
                "DRM_IOCTL_MODE_GETFB2 failed: {}",
                err
            )));
        }

        log::info!(
            "GETFB2 verified: fb_id={} {}x{} fourcc=0x{:08X} modifier=0x{:016X}",
            fb2.fb_id,
            fb2.width,
            fb2.height,
            fb2.pixel_format,
            fb2.modifier[0]
        );

        // Close the GEM handle from the verification probe — we don't need it.
        if fb2.handles[0] != 0 {
            let mut close = DrmGemClose {
                handle: fb2.handles[0],
                pad: 0,
            };
            // SAFETY: Closing a GEM handle we just received. drm_fd is valid.
            unsafe {
                libc::ioctl(drm_fd, DRM_IOCTL_GEM_CLOSE, &mut close);
            }
        }

        Ok(())
    }

    /// Capture a single frame as a DMA-BUF file descriptor.
    ///
    /// This is the hot path — it should be as fast as possible.
    fn capture_dmabuf(&self) -> Result<CapturedFrame, MediaError> {
        // Step 1: Get current framebuffer ID from the CRTC.
        let mut crtc = DrmModeCrtc::default();
        crtc.crtc_id = self.crtc_id;

        // SAFETY: crtc is properly initialized. self.drm_fd is valid.
        let ret = unsafe { libc::ioctl(self.drm_fd, DRM_IOCTL_MODE_GETCRTC, &mut crtc) };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            return Err(MediaError::CaptureError(format!(
                "DRM_IOCTL_MODE_GETCRTC failed during capture: {}",
                err
            )));
        }

        if crtc.fb_id == 0 {
            return Err(MediaError::CaptureError(
                "CRTC has no active framebuffer".into(),
            ));
        }

        // Step 2: Get framebuffer details via GETFB2.
        let mut fb2 = DrmModeFbCmd2::default();
        fb2.fb_id = crtc.fb_id;

        // SAFETY: fb2 is properly initialized with valid fb_id. self.drm_fd is valid.
        let ret = unsafe { libc::ioctl(self.drm_fd, DRM_IOCTL_MODE_GETFB2, &mut fb2) };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            return Err(MediaError::CaptureError(format!(
                "DRM_IOCTL_MODE_GETFB2 failed: {}",
                err
            )));
        }

        if fb2.handles[0] == 0 {
            return Err(MediaError::CaptureError(
                "GETFB2 returned no GEM handle".into(),
            ));
        }

        // Step 3: Export GEM handle → DMA-BUF fd.
        let mut prime = DrmPrimeHandle {
            handle: fb2.handles[0],
            flags: DRM_CLOEXEC | DRM_RDWR,
            fd: -1,
        };

        // SAFETY: prime is properly initialized with a valid GEM handle.
        // self.drm_fd is valid. The kernel will set prime.fd on success.
        let ret = unsafe { libc::ioctl(self.drm_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &mut prime) };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            // Clean up the GEM handle even on failure.
            Self::close_gem_handle(self.drm_fd, fb2.handles[0]);
            return Err(MediaError::CaptureError(format!(
                "DRM_IOCTL_PRIME_HANDLE_TO_FD failed: {}",
                err
            )));
        }

        // Step 4: Close the GEM handle — we have the DMA-BUF fd now.
        // The DMA-BUF fd keeps the underlying buffer alive independently.
        Self::close_gem_handle(self.drm_fd, fb2.handles[0]);

        // Also close GEM handles for additional planes if they differ from plane 0.
        for i in 1..4 {
            if fb2.handles[i] != 0 && fb2.handles[i] != fb2.handles[0] {
                Self::close_gem_handle(self.drm_fd, fb2.handles[i]);
            }
        }

        let timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Estimate buffer size from stride × height (sufficient for single-plane).
        let buffer_size = (fb2.pitches[0] as usize) * (fb2.height as usize);

        Ok(CapturedFrame {
            buffer: GpuBuffer::DmaBuf {
                fd: prime.fd,
                offset: fb2.offsets[0],
                stride: fb2.pitches[0],
                modifier: fb2.modifier[0],
                size: buffer_size,
                width: fb2.width,
                height: fb2.height,
                fourcc: fb2.pixel_format,
            },
            timestamp_us,
            width: fb2.width,
            height: fb2.height,
            format: fourcc_to_pixel_format(fb2.pixel_format),
            is_new_frame: true,
        })
    }

    /// Close a GEM handle to avoid leaking kernel resources.
    fn close_gem_handle(drm_fd: RawFd, handle: u32) {
        let mut close = DrmGemClose { handle, pad: 0 };
        // SAFETY: Closing a GEM handle we received from a prior ioctl.
        // drm_fd is valid. Failure is non-fatal (logged but not propagated).
        let ret = unsafe { libc::ioctl(drm_fd, DRM_IOCTL_GEM_CLOSE, &mut close) };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            log::warn!("DRM_IOCTL_GEM_CLOSE(handle={}) failed: {}", handle, err);
        }
    }
}

impl Drop for DrmCapture {
    fn drop(&mut self) {
        if self.drm_fd >= 0 {
            // SAFETY: self.drm_fd is a valid file descriptor opened by us.
            // Closing it releases all DRM resources associated with this fd.
            unsafe {
                libc::close(self.drm_fd);
            }
            log::info!("Closed DRM device fd={}", self.drm_fd);
        }
    }
}

#[async_trait::async_trait]
impl ScreenCapture for DrmCapture {
    /// List displays based on the active CRTC mode.
    async fn list_displays(&self) -> Result<Vec<DisplayInfo>, MediaError> {
        Ok(vec![DisplayInfo {
            id: format!("drm-crtc-{}", self.crtc_id),
            name: format!("DRM Display (CRTC {})", self.crtc_id),
            width: self.width,
            height: self.height,
            refresh_rate: self.refresh_rate as f64,
            is_primary: true,
        }])
    }

    /// Start DRM capture at the configured frame rate.
    async fn start(&mut self, _display_id: &str, config: &StreamConfig) -> Result<(), MediaError> {
        if self.capturing {
            return Err(MediaError::CaptureAlreadyStarted);
        }

        self.fps = config.fps;
        self.last_frame_time = Instant::now();
        self.capturing = true;

        log::info!(
            "Started DRM capture: CRTC {} at {}x{} @{}fps (display @{}Hz)",
            self.crtc_id,
            self.width,
            self.height,
            self.fps,
            self.refresh_rate
        );

        Ok(())
    }

    /// Capture the next frame. Frame pacing is handled by the pipeline timer.
    async fn next_frame(&mut self) -> Result<CapturedFrame, MediaError> {
        if !self.capturing {
            return Err(MediaError::CaptureNotStarted);
        }

        tokio::task::yield_now().await;
        self.last_frame_time = Instant::now();

        self.capture_dmabuf()
    }

    /// Stop DRM capture.
    async fn stop(&mut self) -> Result<(), MediaError> {
        self.capturing = false;
        log::info!("Stopped DRM capture");
        Ok(())
    }

    /// Returns whether capture is active.
    fn is_capturing(&self) -> bool {
        self.capturing
    }
}

/// Convert a DRM fourcc pixel format to our internal [`PixelFormat`].
///
/// Common framebuffer formats from the kernel:
/// - `XR24` (0x34325258) = XRGB8888 → we report as BGRA (byte order is BGRX on little-endian)
/// - `AR24` (0x34325241) = ARGB8888 → BGRA
/// - `XB24` (0x34324258) = XBGR8888 → BGRA (close enough for encoder input)
/// - `NV12` (0x3231564E) = NV12
fn fourcc_to_pixel_format(fourcc: u32) -> PixelFormat {
    // DRM fourcc codes are stored as little-endian ASCII characters.
    // drm_fourcc::DrmFourcc provides named constants.
    use drm_fourcc::DrmFourcc;

    match DrmFourcc::try_from(fourcc) {
        Ok(DrmFourcc::Xrgb8888) | Ok(DrmFourcc::Argb8888) => PixelFormat::BGRA,
        Ok(DrmFourcc::Xbgr8888) | Ok(DrmFourcc::Abgr8888) => PixelFormat::BGRA,
        Ok(DrmFourcc::Nv12) => PixelFormat::NV12,
        Ok(DrmFourcc::P010) => PixelFormat::P010,
        _ => {
            log::warn!("Unknown DRM fourcc 0x{:08X}, assuming BGRA layout", fourcc);
            PixelFormat::BGRA
        }
    }
}
