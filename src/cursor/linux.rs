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

use crate::cursor::CursorCapture;
use crate::error::MediaError;
use crate::types::*;

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
    /// Whether cursor tracking is currently active.
    active: bool,
}

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
                image: None,
            },
            active: false,
        })
    }

    /// Try to read the cursor position from the X11 display server.
    ///
    /// Returns `None` if X11 is not available or the query fails.
    ///
    /// TODO(phase-2): Implement using `x11rb` or raw Xlib FFI.
    #[allow(dead_code)]
    fn query_x11_cursor(&self) -> Option<(i32, i32)> {
        // TODO: Use XQueryPointer via x11rb:
        //
        // let (conn, screen_num) = x11rb::connect(None)?;
        // let screen = &conn.setup().roots[screen_num];
        // let reply = conn.query_pointer(screen.root)?.reply()?;
        // Some((reply.root_x as i32, reply.root_y as i32))
        None
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
        log::info!("Starting Linux cursor capture (stub — Phase 1)");

        // TODO(phase-2): Detect display server (Wayland vs X11) and
        // initialize the appropriate cursor tracking backend:
        //
        // if std::env::var("WAYLAND_DISPLAY").is_ok() {
        //     // PipeWire cursor metadata path
        // } else if std::env::var("DISPLAY").is_ok() {
        //     // X11 XFixes path
        // } else {
        //     // /dev/input fallback
        // }

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
            return Err(MediaError::CursorError(
                "Cursor capture not started".into(),
            ));
        }

        // TODO(phase-2): Replace with actual cursor queries:
        //
        // if let Some((x, y)) = self.query_x11_cursor() {
        //     self.last_state.x = x;
        //     self.last_state.y = y;
        // } else if let Some(state) = self.query_pipewire_cursor() {
        //     self.last_state = state;
        // }

        Ok(self.last_state.clone())
    }

    /// Stop tracking cursor state and release resources.
    fn stop(&mut self) -> Result<(), MediaError> {
        self.active = false;
        log::info!("Linux cursor capture stopped");
        Ok(())
    }
}
