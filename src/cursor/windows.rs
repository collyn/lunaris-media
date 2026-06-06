//! Windows cursor capture backend using Win32 APIs.
//!
//! Uses `GetCursorPos` for position and `GetCursorInfo` for visibility.
//! Extracts cursor image pixels on shape changes for browser-side overlay rendering.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::cursor::CursorCapture;
use crate::error::MediaError;
use crate::types::*;

use windows::Win32::Foundation::POINT;
use windows::Win32::Graphics::Gdi::{
    DeleteObject, GetDC, GetDIBits, GetObjectW, ReleaseDC, BITMAP, BITMAPINFO, BITMAPINFOHEADER,
    BI_RGB, DIB_RGB_COLORS,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CopyIcon, DestroyIcon, GetCursorInfo, GetCursorPos, GetIconInfo, LoadCursorW, CURSORINFO,
    CURSOR_SHOWING, HCURSOR, HICON, ICONINFO, IDC_ARROW, IDC_CROSS, IDC_HAND, IDC_IBEAM, IDC_NO,
    IDC_SIZEALL, IDC_SIZENESW, IDC_SIZENS, IDC_SIZENWSE, IDC_SIZEWE,
};

pub struct WindowsCursorCapture {
    active: bool,
    last_image_hash: Option<u64>,
}

impl WindowsCursorCapture {
    pub fn new() -> Result<Self, MediaError> {
        Ok(Self {
            active: false,
            last_image_hash: None,
        })
    }
}

fn cursor_image_hash(image: &CursorImage) -> u64 {
    let mut hasher = DefaultHasher::new();
    image.width.hash(&mut hasher);
    image.height.hash(&mut hasher);
    image.hotspot_x.hash(&mut hasher);
    image.hotspot_y.hash(&mut hasher);
    image.rgba_data.hash(&mut hasher);
    hasher.finish()
}

fn read_bitmap_bgra(bitmap: windows::Win32::Graphics::Gdi::HBITMAP) -> Option<(u32, u32, Vec<u8>)> {
    if bitmap.is_invalid() {
        return None;
    }

    let mut bitmap_info_raw = BITMAP::default();
    let object_size = std::mem::size_of::<BITMAP>() as i32;
    let read = unsafe {
        GetObjectW(
            bitmap,
            object_size,
            Some((&mut bitmap_info_raw as *mut BITMAP).cast()),
        )
    };
    if read != object_size || bitmap_info_raw.bmWidth <= 0 || bitmap_info_raw.bmHeight == 0 {
        return None;
    }

    let width = bitmap_info_raw.bmWidth as u32;
    let height = bitmap_info_raw.bmHeight.unsigned_abs();
    let mut info = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: width as i32,
            biHeight: -(height as i32),
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut bgra = vec![0u8; (width * height * 4) as usize];
    let hdc = unsafe { GetDC(None) };
    if hdc.is_invalid() {
        return None;
    }
    let rows = unsafe {
        GetDIBits(
            hdc,
            bitmap,
            0,
            height,
            Some(bgra.as_mut_ptr().cast()),
            &mut info,
            DIB_RGB_COLORS,
        )
    };
    unsafe {
        ReleaseDC(None, hdc);
    }
    if rows == 0 {
        return None;
    }

    Some((width, height, bgra))
}

fn bitmap_bgra_to_rgba(bgra: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(bgra.len());
    for px in bgra.chunks_exact(4) {
        rgba.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
    }
    rgba
}

fn mask_bitmap_to_rgba(mask_bgra: &[u8], width: u32, height: u32) -> Vec<u8> {
    let cursor_height = height / 2;
    let row_len = (width * 4) as usize;
    let mut rgba = vec![0u8; (width * cursor_height * 4) as usize];

    for y in 0..cursor_height as usize {
        let and_row = &mask_bgra[y * row_len..(y + 1) * row_len];
        let xor_row_start = (y + cursor_height as usize) * row_len;
        let xor_row = &mask_bgra[xor_row_start..xor_row_start + row_len];
        for x in 0..width as usize {
            let and_value = and_row[x * 4];
            let xor_value = xor_row[x * 4];
            let out = (y * width as usize + x) * 4;
            if and_value > 0 && xor_value == 0 {
                rgba[out + 3] = 0;
            } else if xor_value > 0 {
                rgba[out..out + 4].copy_from_slice(&[255, 255, 255, 255]);
            } else {
                rgba[out..out + 4].copy_from_slice(&[0, 0, 0, 255]);
            }
        }
    }

    rgba
}

fn cursor_image_from_handle(handle: HCURSOR) -> Option<CursorImage> {
    if handle.is_invalid() {
        return None;
    }

    let icon = unsafe { CopyIcon(HICON::from(handle)).ok()? };
    let mut icon_info = ICONINFO::default();
    let result = unsafe { GetIconInfo(icon, &mut icon_info) };
    unsafe {
        let _ = DestroyIcon(icon);
    }
    if result.is_err() {
        return None;
    }

    let image = if !icon_info.hbmColor.is_invalid() {
        read_bitmap_bgra(icon_info.hbmColor).map(|(width, height, bgra)| CursorImage {
            width,
            height,
            hotspot_x: icon_info.xHotspot,
            hotspot_y: icon_info.yHotspot,
            rgba_data: bitmap_bgra_to_rgba(&bgra),
        })
    } else {
        read_bitmap_bgra(icon_info.hbmMask).and_then(|(width, height, bgra)| {
            if height < 2 {
                return None;
            }
            Some(CursorImage {
                width,
                height: height / 2,
                hotspot_x: icon_info.xHotspot,
                hotspot_y: icon_info.yHotspot,
                rgba_data: mask_bitmap_to_rgba(&bgra, width, height),
            })
        })
    };

    unsafe {
        if !icon_info.hbmColor.is_invalid() {
            let _ = DeleteObject(icon_info.hbmColor);
        }
        if !icon_info.hbmMask.is_invalid() {
            let _ = DeleteObject(icon_info.hbmMask);
        }
    }

    image.filter(|img| {
        img.width > 0
            && img.height > 0
            && img.rgba_data.len() == (img.width * img.height * 4) as usize
    })
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
        let mut image = None;
        if visible {
            if let Some(cursor_image) = cursor_image_from_handle(cursor_info.hCursor) {
                let hash = cursor_image_hash(&cursor_image);
                if self.last_image_hash != Some(hash) {
                    self.last_image_hash = Some(hash);
                    image = Some(cursor_image);
                }
            }
        }

        Ok(CursorState {
            x: point.x,
            y: point.y,
            visible,
            kind,
            image,
        })
    }

    fn stop(&mut self) -> Result<(), MediaError> {
        self.active = false;
        log::info!("Windows cursor capture stopped");
        Ok(())
    }
}
