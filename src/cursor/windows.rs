//! Windows cursor capture backend using Win32 APIs.
//!
//! Uses `GetCursorPos` for position and `GetCursorInfo` for visibility.
//! Cursor image extraction (shape changes) is deferred to Phase 2.

use crate::cursor::CursorCapture;
use crate::error::MediaError;
use crate::types::*;

use windows::Win32::Foundation::POINT;
use windows::Win32::UI::WindowsAndMessaging::{
    GetCursorInfo, GetCursorPos, CURSORINFO, CURSOR_SHOWING,
};

pub struct WindowsCursorCapture {
    active: bool,
}

impl WindowsCursorCapture {
    pub fn new() -> Result<Self, MediaError> {
        Ok(Self { active: false })
    }
}

impl CursorCapture for WindowsCursorCapture {
    fn start(&mut self) -> Result<(), MediaError> {
        log::info!("Starting Windows cursor capture");
        self.active = true;
        Ok(())
    }

    fn get_cursor_state(&mut self) -> Result<CursorState, MediaError> {
        if !self.active {
            return Err(MediaError::CursorError(
                "Cursor capture not started".into(),
            ));
        }

        let mut point = POINT::default();
        let mut cursor_info = CURSORINFO {
            cbSize: std::mem::size_of::<CURSORINFO>() as u32,
            ..Default::default()
        };

        unsafe {
            let _ = GetCursorPos(&mut point);
            let _ = GetCursorInfo(&mut cursor_info);
        }

        let visible = (cursor_info.flags.0 & CURSOR_SHOWING.0) != 0;

        Ok(CursorState {
            x: point.x,
            y: point.y,
            visible,
            image: None, // Phase 2: extract cursor image via GetIconInfo + GetBitmapBits
        })
    }

    fn stop(&mut self) -> Result<(), MediaError> {
        self.active = false;
        log::info!("Windows cursor capture stopped");
        Ok(())
    }
}
