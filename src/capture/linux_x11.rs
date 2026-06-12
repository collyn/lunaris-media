//! # X11 Screen Capture via XShm (X Shared Memory Extension)
//!
//! Uses shared memory to capture the screen without X server round-trip
//! overhead. The shared memory segment is allocated once during `start()`
//! and reused for every frame, eliminating per-frame allocation.
//!
//! ## Performance vs XGetImage
//!
//! | Method     | Per-frame alloc | X server round-trip | CPU copy   |
//! |------------|-----------------|---------------------|------------|
//! | XGetImage  | 8MB XImage      | Yes (synchronous)   | 8MB + 8MB  |
//! | XShmGetImage | None (reuse)  | Minimal             | 0 (shared) |

use std::ptr;
use x11::xlib;
use x11::xshm;

use crate::capture::gpu_buffer::GpuBuffer;
use crate::capture::{CapturedFrame, ScreenCapture};
use crate::error::MediaError;
use crate::types::*;

/// X11-based screen capture backend using XShm for zero-copy shared memory.
pub struct X11Capture {
    display: *mut xlib::Display,
    root: xlib::Window,
    width: u32,
    height: u32,
    fps: u32,
    capture_x: i32,
    capture_y: i32,
    capturing: bool,
    last_frame_time: std::time::Instant,
    /// XShm shared memory segment info (persistent across frames).
    shm_info: xshm::XShmSegmentInfo,
    /// XImage backed by shared memory (persistent across frames).
    shm_image: *mut xlib::XImage,
    /// Whether XShm was successfully initialized.
    shm_active: bool,
    /// Fallback: reusable BGRA buffer for non-XShm path.
    bgra_buffer: Vec<u8>,
}

// Safety: X11 Display pointer is safe to send between threads when
// synchronized via X11_MUTEX. Shared memory pointers are process-global.
unsafe impl Send for X11Capture {}
unsafe impl Sync for X11Capture {}

impl X11Capture {
    /// Attempts to create a new X11 capture instance.
    pub fn new() -> Result<Self, MediaError> {
        unsafe {
            xlib::XInitThreads();
        }
        let display = {
            let _lock = crate::X11_MUTEX.lock().unwrap();
            unsafe { xlib::XOpenDisplay(ptr::null()) }
        };
        if display.is_null() {
            return Err(MediaError::CaptureError(
                "Failed to open X11 display. Check your DISPLAY environment variable.".into(),
            ));
        }
        let screen = unsafe { xlib::XDefaultScreen(display) };
        let root = unsafe { xlib::XRootWindow(display, screen) };

        let width = unsafe { xlib::XDisplayWidth(display, screen) } as u32;
        let height = unsafe { xlib::XDisplayHeight(display, screen) } as u32;

        log::info!(
            "Initialized X11 capture on root window ({}x{})",
            width,
            height
        );

        Ok(Self {
            display,
            root,
            width,
            height,
            fps: 60,
            capture_x: 0,
            capture_y: 0,
            capturing: false,
            last_frame_time: std::time::Instant::now(),
            shm_info: unsafe { std::mem::zeroed() },
            shm_image: ptr::null_mut(),
            shm_active: false,
            bgra_buffer: Vec::new(),
        })
    }

    /// Initialize XShm shared memory segment and XImage.
    fn init_shm(&mut self) -> bool {
        let _lock = crate::X11_MUTEX.lock().unwrap();

        // Check if XShm extension is available
        let shm_available = unsafe { xshm::XShmQueryExtension(self.display) };
        if shm_available == 0 {
            log::warn!("XShm extension not available, falling back to XGetImage");
            return false;
        }

        let screen = unsafe { xlib::XDefaultScreen(self.display) };
        let visual = unsafe { xlib::XDefaultVisual(self.display, screen) };
        let depth = unsafe { xlib::XDefaultDepth(self.display, screen) } as u32;

        // Create XImage backed by shared memory
        let image = unsafe {
            xshm::XShmCreateImage(
                self.display,
                visual,
                depth,
                xlib::ZPixmap,
                ptr::null_mut(),
                &mut self.shm_info,
                self.width,
                self.height,
            )
        };

        if image.is_null() {
            log::warn!("XShmCreateImage failed, falling back to XGetImage");
            return false;
        }

        // Calculate shared memory size
        let image_size = unsafe { (*image).bytes_per_line * (*image).height };

        // Create shared memory segment
        let shmid = unsafe {
            libc::shmget(
                libc::IPC_PRIVATE,
                image_size as usize,
                libc::IPC_CREAT | 0o777,
            )
        };

        if shmid < 0 {
            log::warn!("shmget failed: {}", std::io::Error::last_os_error());
            unsafe {
                xlib::XDestroyImage(image);
            }
            return false;
        }

        // Attach shared memory
        let shmaddr = unsafe { libc::shmat(shmid, ptr::null(), 0) };
        if shmaddr == (-1isize) as *mut libc::c_void {
            log::warn!("shmat failed: {}", std::io::Error::last_os_error());
            unsafe {
                libc::shmctl(shmid, libc::IPC_RMID, ptr::null_mut());
                xlib::XDestroyImage(image);
            }
            return false;
        }

        self.shm_info.shmid = shmid;
        self.shm_info.shmaddr = shmaddr as *mut libc::c_char;
        self.shm_info.readOnly = xlib::False;

        // Point XImage data to shared memory
        unsafe {
            (*image).data = self.shm_info.shmaddr;
        }

        // Attach to X server
        let attached = unsafe { xshm::XShmAttach(self.display, &mut self.shm_info) };
        if attached == 0 {
            log::warn!("XShmAttach failed, falling back to XGetImage");
            unsafe {
                libc::shmdt(shmaddr);
                libc::shmctl(shmid, libc::IPC_RMID, ptr::null_mut());
                xlib::XDestroyImage(image);
            }
            return false;
        }

        // Mark segment for deletion when all processes detach
        unsafe {
            libc::shmctl(shmid, libc::IPC_RMID, ptr::null_mut());
        }

        self.shm_image = image;
        self.shm_active = true;

        log::info!(
            "XShm initialized: {}x{} depth={} shm_size={}MB",
            self.width,
            self.height,
            depth,
            image_size / (1024 * 1024)
        );

        true
    }

    /// Cleanup XShm resources.
    fn cleanup_shm(&mut self) {
        if !self.shm_active {
            return;
        }

        let _lock = crate::X11_MUTEX.lock().unwrap();

        unsafe {
            if !self.shm_image.is_null() {
                xshm::XShmDetach(self.display, &mut self.shm_info);
                (*self.shm_image).data = ptr::null_mut();
                xlib::XDestroyImage(self.shm_image);
                self.shm_image = ptr::null_mut();
            }

            if !self.shm_info.shmaddr.is_null() {
                libc::shmdt(self.shm_info.shmaddr as *const libc::c_void);
                self.shm_info.shmaddr = ptr::null_mut();
            }
        }

        self.shm_active = false;
        log::info!("XShm resources cleaned up");
    }

    /// Capture using XShm (fast path — no per-frame allocation, no X round-trip).
    fn capture_shm(&mut self) -> Result<CapturedFrame, MediaError> {
        let data_len = (self.width * self.height * 4) as usize;

        {
            let _lock = crate::X11_MUTEX.lock().unwrap();

            // XShmGetImage captures directly into shared memory — no allocation
            let result = unsafe {
                xshm::XShmGetImage(
                    self.display,
                    self.root,
                    self.shm_image,
                    self.capture_x,
                    self.capture_y,
                    xlib::XAllPlanes() as u32,
                )
            };

            if result == 0 {
                return Err(MediaError::CaptureError("XShmGetImage failed".into()));
            }

            // Draw cursor directly onto the shared memory buffer
            let bgra_slice = unsafe {
                std::slice::from_raw_parts_mut(self.shm_info.shmaddr as *mut u8, data_len)
            };
            if crate::capture::should_embed_host_cursor() {
                unsafe {
                    draw_cursor(
                        self.display,
                        self.root,
                        bgra_slice,
                        self.width,
                        self.height,
                        self.capture_x,
                        self.capture_y,
                    );
                }
            }
        }
        // Lock released — input injection can proceed immediately

        // Copy from shared memory into reusable buffer (single memcpy, no allocation).
        // We must copy because the shm buffer will be overwritten next frame.
        self.bgra_buffer.resize(data_len, 0);
        unsafe {
            ptr::copy_nonoverlapping(
                self.shm_info.shmaddr as *const u8,
                self.bgra_buffer.as_mut_ptr(),
                data_len,
            );
        }

        // Swap buffer out to avoid clone (zero-copy handoff to encoder)
        let frame_data = std::mem::take(&mut self.bgra_buffer);

        let timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        Ok(CapturedFrame {
            buffer: GpuBuffer::CpuBuffer {
                data: frame_data,
                stride: self.width * 4,
                format: PixelFormat::BGRA,
                width: self.width,
                height: self.height,
            },
            timestamp_us,
            width: self.width,
            height: self.height,
            format: PixelFormat::BGRA,
            is_new_frame: true,
            cursor: None,
        })
    }

    /// Capture using XGetImage (slow fallback).
    fn capture_xgetimage(&mut self) -> Result<CapturedFrame, MediaError> {
        let image = {
            let _lock = crate::X11_MUTEX.lock().unwrap();
            let img = unsafe {
                xlib::XGetImage(
                    self.display,
                    self.root,
                    self.capture_x,
                    self.capture_y,
                    self.width,
                    self.height,
                    xlib::XAllPlanes(),
                    xlib::ZPixmap,
                )
            };

            if !img.is_null() {
                let data_len = (self.width * self.height * 4) as usize;
                let bgra_slice =
                    unsafe { std::slice::from_raw_parts_mut((*img).data as *mut u8, data_len) };
                if crate::capture::should_embed_host_cursor() {
                    unsafe {
                        draw_cursor(
                            self.display,
                            self.root,
                            bgra_slice,
                            self.width,
                            self.height,
                            self.capture_x,
                            self.capture_y,
                        );
                    }
                }
            }
            img
        };

        if image.is_null() {
            return Err(MediaError::CaptureError("XGetImage returned NULL".into()));
        }

        let data_len = (self.width * self.height * 4) as usize;
        self.bgra_buffer.resize(data_len, 0);
        unsafe {
            let src = (*image).data as *const u8;
            ptr::copy_nonoverlapping(src, self.bgra_buffer.as_mut_ptr(), data_len);
        }

        unsafe {
            xlib::XDestroyImage(image);
        }

        let frame_data = std::mem::take(&mut self.bgra_buffer);

        let timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        Ok(CapturedFrame {
            buffer: GpuBuffer::CpuBuffer {
                data: frame_data,
                stride: self.width * 4,
                format: PixelFormat::BGRA,
                width: self.width,
                height: self.height,
            },
            timestamp_us,
            width: self.width,
            height: self.height,
            format: PixelFormat::BGRA,
            is_new_frame: true,
            cursor: None,
        })
    }

    /// Parses `xrandr --query` output to enumerate connected displays.
    #[allow(dead_code)]
    fn parse_xrandr_output(output: &str) -> Vec<DisplayInfo> {
        Self::parse_xrandr_output_with_offsets(output)
            .into_iter()
            .map(|(display, _, _)| display)
            .collect()
    }

    fn parse_xrandr_output_with_offsets(output: &str) -> Vec<(DisplayInfo, i32, i32)> {
        let mut displays = Vec::new();

        for line in output.lines() {
            if !line.starts_with(' ') && line.contains(" connected") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 2 {
                    continue;
                }

                let id = parts[0].to_string();
                let is_primary = parts.contains(&"primary");
                let (width, height, x, y) =
                    Self::extract_xrandr_geometry(&parts).unwrap_or((0, 0, 0, 0));

                displays.push((
                    DisplayInfo {
                        id: id.clone(),
                        name: id,
                        width,
                        height,
                        refresh_rate: 0.0,
                        is_primary,
                    },
                    x,
                    y,
                ));
            }

            if line.starts_with("   ") && !displays.is_empty() {
                let trimmed = line.trim();
                if trimmed.contains('*') {
                    if let Some(hz) = Self::extract_active_refresh_rate(trimmed) {
                        if let Some((last, _, _)) = displays.last_mut() {
                            if last.refresh_rate == 0.0 {
                                last.refresh_rate = hz;
                            }
                        }
                    }
                }
            }
        }

        displays
    }

    fn extract_xrandr_geometry(parts: &[&str]) -> Option<(u32, u32, i32, i32)> {
        for part in parts {
            if !part.contains('x') || !part.contains('+') {
                continue;
            }

            let mut geometry = part.split('+');
            let resolution = geometry.next()?;
            let x = geometry.next()?.parse::<i32>().ok()?;
            let y = geometry.next()?.parse::<i32>().ok()?;
            let mut dims = resolution.split('x');
            let width = dims.next()?.parse::<u32>().ok()?;
            let height = dims.next()?.parse::<u32>().ok()?;
            return Some((width, height, x, y));
        }
        None
    }

    fn query_xrandr_displays() -> Result<Vec<(DisplayInfo, i32, i32)>, MediaError> {
        let output = std::process::Command::new("xrandr")
            .arg("--query")
            .output()
            .map_err(|e| MediaError::CaptureError(format!("xrandr failed: {}", e)))?;

        if !output.status.success() {
            return Err(MediaError::CaptureError(format!(
                "xrandr --query failed with status {}",
                output.status
            )));
        }

        let text = String::from_utf8_lossy(&output.stdout);
        Ok(Self::parse_xrandr_output_with_offsets(&text))
    }

    /// Extracts the active refresh rate from an xrandr mode line.
    #[allow(dead_code)]
    fn extract_active_refresh_rate(mode_line: &str) -> Option<f64> {
        for part in mode_line.split_whitespace().skip(1) {
            if part.contains('*') {
                let rate_str = part.trim_end_matches(|c| c == '*' || c == '+');
                if let Ok(rate) = rate_str.parse::<f64>() {
                    return Some(rate);
                }
            }
        }
        None
    }
}

impl Drop for X11Capture {
    fn drop(&mut self) {
        self.cleanup_shm();

        if !self.display.is_null() {
            let _lock = crate::X11_MUTEX.lock().unwrap();
            unsafe {
                xlib::XCloseDisplay(self.display);
            }
            log::info!("Closed X11 display connection");
        }
    }
}

#[async_trait::async_trait]
impl ScreenCapture for X11Capture {
    /// Lists available displays.
    async fn list_displays(&self) -> Result<Vec<DisplayInfo>, MediaError> {
        match Self::query_xrandr_displays() {
            Ok(displays) if !displays.is_empty() => Ok(displays
                .into_iter()
                .map(|(display, _, _)| display)
                .collect()),
            Ok(_) | Err(_) => Ok(vec![DisplayInfo {
                id: "default".to_string(),
                name: "Default X11 Display".to_string(),
                width: self.width,
                height: self.height,
                refresh_rate: 60.0,
                is_primary: true,
            }]),
        }
    }

    /// Starts screen capture, initializing XShm if available.
    async fn start(&mut self, display_id: &str, config: &StreamConfig) -> Result<(), MediaError> {
        self.width = config.width;
        self.height = config.height;
        self.fps = config.fps;
        self.capture_x = 0;
        self.capture_y = 0;
        self.last_frame_time = std::time::Instant::now();

        if crate::capture::should_embed_host_cursor() {
            log::info!("X11 capture: embedding host cursor in video frames");
        } else {
            log::info!(
                "X11 capture: hiding host cursor from video frames; browser overlay will render it"
            );
        }

        if let Ok(displays) = Self::query_xrandr_displays() {
            if let Some((display, x, y)) = displays
                .iter()
                .find(|(display, _, _)| display.id == display_id)
                .or_else(|| displays.iter().find(|(display, _, _)| display.is_primary))
            {
                self.capture_x = *x;
                self.capture_y = *y;
                log::info!(
                    "X11 capture selected display '{}' at {}x{}+{}+{}",
                    display.id,
                    display.width,
                    display.height,
                    self.capture_x,
                    self.capture_y
                );
            }
        }

        // Try to initialize XShm for fast capture
        if !self.shm_active {
            if !self.init_shm() {
                log::warn!("XShm init failed; will use XGetImage fallback (slower)");
            }
        }

        self.capturing = true;
        let mode = if self.shm_active {
            "XShm"
        } else {
            "XGetImage (fallback)"
        };
        log::info!(
            "Started X11 capture at {}x{} {}fps via {}",
            self.width,
            self.height,
            self.fps,
            mode
        );
        Ok(())
    }

    /// Receives the next captured frame.
    async fn next_frame(&mut self) -> Result<CapturedFrame, MediaError> {
        if !self.capturing {
            return Err(MediaError::CaptureNotStarted);
        }

        self.last_frame_time = std::time::Instant::now();
        tokio::task::yield_now().await;

        // Use XShm fast path if available, otherwise fall back to XGetImage
        if self.shm_active {
            self.capture_shm()
        } else {
            self.capture_xgetimage()
        }
    }

    /// Stops screen capture.
    async fn stop(&mut self) -> Result<(), MediaError> {
        self.capturing = false;
        self.cleanup_shm();
        log::info!("Stopped X11 capture");
        Ok(())
    }

    /// Returns capturing status.
    fn is_capturing(&self) -> bool {
        self.capturing
    }
}

unsafe fn draw_cursor(
    display: *mut xlib::Display,
    _root: xlib::Window,
    bgra_slice: &mut [u8],
    width: u32,
    height: u32,
    capture_x: i32,
    capture_y: i32,
) {
    let cursor_img = x11::xfixes::XFixesGetCursorImage(display);
    if !cursor_img.is_null() {
        let img = &*cursor_img;
        let x_start = img.x as i32 - capture_x;
        let y_start = img.y as i32 - capture_y;
        let c_width = img.width as i32;
        let c_height = img.height as i32;

        let pixel_count = (c_width * c_height) as usize;
        let pixels = std::slice::from_raw_parts(img.pixels, pixel_count);

        for cy in 0..c_height {
            let sy = y_start + cy;
            if sy < 0 || sy >= height as i32 {
                continue;
            }

            for cx in 0..c_width {
                let sx = x_start + cx;
                if sx < 0 || sx >= width as i32 {
                    continue;
                }

                let pixel = pixels[(cy * c_width + cx) as usize] as u32;
                let alpha = (pixel >> 24) & 0xFF;
                if alpha == 0 {
                    continue;
                }

                let red = (pixel >> 16) & 0xFF;
                let green = (pixel >> 8) & 0xFF;
                let blue = pixel & 0xFF;

                let bgra_idx = ((sy * width as i32 + sx) * 4) as usize;
                if bgra_idx + 3 < bgra_slice.len() {
                    if alpha == 255 {
                        bgra_slice[bgra_idx] = blue as u8;
                        bgra_slice[bgra_idx + 1] = green as u8;
                        bgra_slice[bgra_idx + 2] = red as u8;
                    } else {
                        let dst_b = bgra_slice[bgra_idx] as u32;
                        let dst_g = bgra_slice[bgra_idx + 1] as u32;
                        let dst_r = bgra_slice[bgra_idx + 2] as u32;

                        bgra_slice[bgra_idx] = ((blue * alpha + dst_b * (255 - alpha)) / 255) as u8;
                        bgra_slice[bgra_idx + 1] =
                            ((green * alpha + dst_g * (255 - alpha)) / 255) as u8;
                        bgra_slice[bgra_idx + 2] =
                            ((red * alpha + dst_r * (255 - alpha)) / 255) as u8;
                    }
                }
            }
        }
        xlib::XFree(cursor_img as *mut _);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_xrandr_output() {
        let xrandr_output = r#"Screen 0: minimum 320 x 200, current 5760 x 2160, maximum 16384 x 16384
DP-1 connected primary 3840x2160+0+0 (normal left inverted right x axis y axis) 597mm x 336mm
   3840x2160     60.00*+  30.00  
   2560x1440     59.95  
HDMI-1 disconnected (normal left inverted right x axis y axis)
DP-2 connected 1920x1080+3840+0 (normal left inverted right x axis y axis) 530mm x 300mm
   1920x1080     60.00*+  50.00    59.94  
"#;

        let displays = X11Capture::parse_xrandr_output(xrandr_output);
        assert_eq!(displays.len(), 2, "Should find 2 connected displays");

        assert_eq!(displays[0].id, "DP-1");
        assert_eq!(displays[0].width, 3840);
        assert_eq!(displays[0].height, 2160);
        assert!((displays[0].refresh_rate - 60.0).abs() < 0.01);
        assert!(displays[0].is_primary);

        assert_eq!(displays[1].id, "DP-2");
        assert_eq!(displays[1].width, 1920);
        assert_eq!(displays[1].height, 1080);
        assert!((displays[1].refresh_rate - 60.0).abs() < 0.01);
        assert!(!displays[1].is_primary);
    }
}
