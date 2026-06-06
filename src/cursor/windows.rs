//! Windows cursor capture backend using Win32 APIs.
//!
//! Uses `GetCursorPos` for position and `GetCursorInfo` for visibility.
//! Cursor image extraction (shape changes) is deferred to Phase 2.

use crate::cursor::CursorCapture;
use crate::error::MediaError;
use crate::types::*;

use windows::Win32::Foundation::POINT;
use windows::Win32::UI::WindowsAndMessaging::{
    GetCursorInfo, GetCursorPos, LoadCursorW, CURSORINFO, CURSOR_SHOWING, HCURSOR, IDC_ARROW,
    IDC_CROSS, IDC_HAND, IDC_IBEAM, IDC_NO, IDC_SIZEALL, IDC_SIZENESW, IDC_SIZENS, IDC_SIZENWSE,
    IDC_SIZEWE,
};

pub struct WindowsCursorCapture {
    active: bool,
}

impl WindowsCursorCapture {
    pub fn new() -> Result<Self, MediaError> {
        Ok(Self { active: false })
    }
}

fn cursor_kind_from_handle(handle: HCURSOR) -> CursorKind {
    fn matches_system_cursor(handle: HCURSOR, cursor_name: windows::core::PCWSTR) -> bool {
        unsafe { LoadCursorW(None, cursor_name).is_ok_and(|system| system == handle) }
    }

    if matches_system_cursor(handle, IDC_IBEAM) {
        CursorKind::IBeam
    } else if matches_system_cursor(handle, IDC_HAND) {
        CursorKind::Hand
    } else if matches_system_cursor(handle, IDC_CROSS) {
        CursorKind::Cross
    } else if matches_system_cursor(handle, IDC_SIZEALL) {
        CursorKind::Move
    } else if matches_system_cursor(handle, IDC_SIZENS) {
        CursorKind::ResizeNs
    } else if matches_system_cursor(handle, IDC_SIZEWE) {
        CursorKind::ResizeEw
    } else if matches_system_cursor(handle, IDC_SIZENESW) {
        CursorKind::ResizeNesw
    } else if matches_system_cursor(handle, IDC_SIZENWSE) {
        CursorKind::ResizeNwse
    } else if matches_system_cursor(handle, IDC_NO) {
        CursorKind::Unavailable
    } else if matches_system_cursor(handle, IDC_ARROW) {
        CursorKind::Arrow
    } else {
        CursorKind::Unknown
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
            return Err(MediaError::CursorError("Cursor capture not started".into()));
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
        let kind = cursor_kind_from_handle(cursor_info.hCursor);

        Ok(CursorState {
            x: point.x,
            y: point.y,
            visible,
            kind,
            image: None, // Phase 2: extract cursor image via GetIconInfo + GetBitmapBits
        })
    }

    fn stop(&mut self) -> Result<(), MediaError> {
        self.active = false;
        log::info!("Windows cursor capture stopped");
        Ok(())
    }
}
