//! Linux cursor capture backend.
//!
//! Tracks mouse cursor position and shape on Linux desktops. The ideal
//! implementation depends on the display server:
//!
//! - **Wayland**: Cursor metadata is typically embedded in the PipeWire screen
//!   capture stream. This module provides a separate tracking path for overlay
//!   rendering or when PipeWire cursor metadata is unavailable.
//!
//! - **X11**: Cursor position can be queried via `XQueryPointer` and shape
//!   changes tracked with the XFixes extension.
//!
//! ## Phase 1 Status
//!
//! This is a basic stub implementation that returns a default cursor state.
//! Proper cursor tracking via X11/XFixes or PipeWire metadata will be added
//! in a future phase.

use std::ptr;

use crate::cursor::CursorCapture;
use crate::error::MediaError;
use crate::types::*;

use x11::{xfixes, xlib};

/// Linux cursor capture backend.
///
/// # Phase 1 Limitations
///
/// The current implementation returns a static default cursor position. Future
/// phases will add:
///
/// - **X11**: `XQueryPointer` for position + XFixes for shape changes
/// - **Wayland/PipeWire**: Cursor metadata from the PipeWire capture stream
/// - **`/dev/input/`**: Raw input events as a fallback for headless setups
pub struct LinuxCursorCapture {
    /// Last known cursor state (used for change detection in the pipeline).
    last_state: CursorState,
    /// Open X11 display for pointer and cursor image queries.
    x11_display: Option<*mut xlib::Display>,
    /// Last XFixes cursor serial sent to consumers.
    last_cursor_serial: Option<u64>,
    /// Whether cursor tracking is currently active.
    active: bool,
}

unsafe impl Send for LinuxCursorCapture {}

impl LinuxCursorCapture {
    /// Create a new Linux cursor capture instance.
    ///
    /// Always succeeds in Phase 1 since the stub doesn't require any system
    /// resources.
    pub fn new() -> Result<Self, MediaError> {
        Ok(Self {
            last_state: CursorState {
                x: 0,
                y: 0,
                visible: true,
                kind: CursorKind::Arrow,
                image: None,
            },
            x11_display: None,
            last_cursor_serial: None,
            active: false,
        })
    }

    /// Initialize X11 cursor tracking when available.
    fn init_x11(&mut self) {
        if self.x11_display.is_some() || std::env::var_os("DISPLAY").is_none() {
            return;
        }

        let display = unsafe { xlib::XOpenDisplay(ptr::null()) };
        if display.is_null() {
            return;
        }

        let mut event_base = 0;
        let mut error_base = 0;
        let has_xfixes =
            unsafe { xfixes::XFixesQueryExtension(display, &mut event_base, &mut error_base) != 0 };
        if !has_xfixes {
            unsafe {
                xlib::XCloseDisplay(display);
            }
            return;
        }

        self.x11_display = Some(display);
        log::info!("Linux cursor capture: using X11 XFixes cursor image backend");
    }

    /// Try to read the cursor position from the X11 display server.
    fn query_x11_cursor(&self) -> Option<(i32, i32)> {
        let display = self.x11_display?;
        let root = unsafe { xlib::XDefaultRootWindow(display) };
        let mut root_return = 0;
        let mut child_return = 0;
        let mut root_x = 0;
        let mut root_y = 0;
        let mut win_x = 0;
        let mut win_y = 0;
        let mut mask = 0;
        let ok = unsafe {
            xlib::XQueryPointer(
                display,
                root,
                &mut root_return,
                &mut child_return,
                &mut root_x,
                &mut root_y,
                &mut win_x,
                &mut win_y,
                &mut mask,
            )
        };
        (ok != 0).then_some((root_x, root_y))
    }

    /// Try to read the current cursor image from XFixes.
    fn query_x11_cursor_image(&mut self) -> Option<CursorImage> {
        let display = self.x11_display?;
        let image_ptr = unsafe { xfixes::XFixesGetCursorImage(display) };
        if image_ptr.is_null() {
            return None;
        }

        let image = unsafe { *image_ptr };
        self.last_state.x = image.x as i32;
        self.last_state.y = image.y as i32;
        let serial = image.cursor_serial as u64;
        let width = image.width as u32;
        let height = image.height as u32;
        let pixels_len = width.checked_mul(height)? as usize;

        let cursor_image = if width > 0
            && height > 0
            && !image.pixels.is_null()
            && self.last_cursor_serial != Some(serial)
        {
            let pixels = unsafe { std::slice::from_raw_parts(image.pixels, pixels_len) };
            let mut rgba = Vec::with_capacity(pixels_len * 4);
            for &pixel in pixels {
                let argb = pixel as u32;
                rgba.extend_from_slice(&[
                    ((argb >> 16) & 0xff) as u8,
                    ((argb >> 8) & 0xff) as u8,
                    (argb & 0xff) as u8,
                    ((argb >> 24) & 0xff) as u8,
                ]);
            }
            self.last_cursor_serial = Some(serial);
            Some(CursorImage {
                width,
                height,
                hotspot_x: image.xhot as u32,
                hotspot_y: image.yhot as u32,
                rgba_data: rgba,
            })
        } else {
            None
        };

        unsafe {
            xlib::XFree(image_ptr.cast());
        }

        cursor_image
    }

    /// Try to read cursor metadata from the PipeWire capture stream.
    ///
    /// On Wayland, cursor position and shape are part of the screen capture
    /// metadata rather than a separately queryable resource.
    ///
    /// TODO(phase-2): Extract cursor data from PipeWire stream metadata.
    #[allow(dead_code)]
    fn query_pipewire_cursor(&self) -> Option<CursorState> {
        // TODO: When the PipeWire capture backend is integrated, cursor
        // metadata (position, visibility, RGBA bitmap) can be extracted
        // from the `SPA_META_Cursor` metadata attached to each buffer.
        None
    }
}

impl CursorCapture for LinuxCursorCapture {
    /// Start tracking cursor state.
    ///
    /// In Phase 1 this is a no-op. Future phases will initialize X11/PipeWire
    /// cursor tracking resources.
    fn start(&mut self) -> Result<(), MediaError> {
        log::info!("Starting Linux cursor capture");
        self.init_x11();
        if self.x11_display.is_none() {
            log::info!("Linux cursor capture: X11/XFixes unavailable, using fallback cursor state");
        }

        self.active = true;
        Ok(())
    }

    /// Get the current cursor position and image.
    ///
    /// In Phase 1 this returns the last known state (defaults to position 0,0
    /// with the cursor visible and no custom image).
    ///
    /// TODO(phase-2): Query real cursor position from X11 or PipeWire.
    fn get_cursor_state(&mut self) -> Result<CursorState, MediaError> {
        if !self.active {
            return Err(MediaError::CursorError("Cursor capture not started".into()));
        }

        if let Some((x, y)) = self.query_x11_cursor() {
            self.last_state.x = x;
            self.last_state.y = y;
            self.last_state.visible = true;
        }
        self.last_state.image = self.query_x11_cursor_image();

        Ok(self.last_state.clone())
    }

    /// Stop tracking cursor state and release resources.
    fn stop(&mut self) -> Result<(), MediaError> {
        self.active = false;
        if let Some(display) = self.x11_display.take() {
            unsafe {
                xlib::XCloseDisplay(display);
            }
        }
        log::info!("Linux cursor capture stopped");
        Ok(())
    }
}
