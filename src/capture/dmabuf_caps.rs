//! DMA-BUF capability discovery for zero-copy capture on Linux.
//!
//! Two questions have to be answered before a PipeWire screencast can hand us
//! GPU buffers that the encoder can consume without a copy:
//!
//! 1. **Which GPU?** The compositor renders on the GPU that drives the
//!    displays. A DMA-BUF exported from that GPU can only be imported without
//!    a copy by the same device, so the VAAPI encoder must open the matching
//!    render node (on multi-GPU systems `/dev/dri/renderD128` is frequently an
//!    unrelated iGPU).
//! 2. **Which modifiers?** PipeWire only negotiates DMA-BUF buffers when the
//!    consumer advertises the DRM format modifiers it can import (this mirrors
//!    Sunshine's `pipewire.cpp`, which queries them through EGL on a GBM
//!    display of the capture GPU).
//!
//! Everything is loaded at runtime through `libloading` so the crate keeps
//! building on hosts without the development headers.

use std::path::Path;

/// DRM fourcc codes (little-endian `fourcc_code(a, b, c, d)`).
pub const DRM_FORMAT_XRGB8888: u32 = 0x3432_5258; // 'XR24' — memory order B,G,R,X
pub const DRM_FORMAT_ARGB8888: u32 = 0x3432_5241; // 'AR24' — memory order B,G,R,A
pub const DRM_FORMAT_XBGR8888: u32 = 0x3432_4258; // 'XB24' — memory order R,G,B,X
pub const DRM_FORMAT_ABGR8888: u32 = 0x3432_4241; // 'AB24' — memory order R,G,B,A

/// `DRM_FORMAT_MOD_LINEAR`.
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;
/// `DRM_FORMAT_MOD_INVALID` — "implicit modifier" (driver-private layout).
pub const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

/// Formats offered to the compositor, in order of preference.
///
/// The X variants come first: compositors normally scan out opaque buffers and
/// the alpha channel is irrelevant for streaming.
pub const PREFERRED_DRM_FORMATS: [u32; 4] = [
    DRM_FORMAT_XRGB8888,
    DRM_FORMAT_ARGB8888,
    DRM_FORMAT_XBGR8888,
    DRM_FORMAT_ABGR8888,
];

/// Resolve the render node (`/dev/dri/renderD*`) of the GPU that drives the
/// connected displays.
///
/// Order of precedence:
/// 1. `LUNARIS_RENDER_NODE` (explicit override, e.g. `/dev/dri/renderD129`).
/// 2. The render node belonging to the first DRM card that has a connected
///    connector (that is the GPU the compositor scans out from).
/// 3. The first render node found.
pub fn display_render_node() -> Option<String> {
    if let Ok(node) = std::env::var("LUNARIS_RENDER_NODE") {
        let node = node.trim().to_string();
        if !node.is_empty() && Path::new(&node).exists() {
            return Some(node);
        }
        log::warn!(
            "LUNARIS_RENDER_NODE='{}' does not exist; auto-detecting",
            node
        );
    }

    for card in cards_with_connected_display() {
        if let Some(node) = render_node_of_card(&card) {
            log::info!("DMA-BUF: display GPU {} → {}", card, node);
            return Some(node);
        }
    }

    // Fallback: first render node on the system.
    let mut nodes: Vec<String> = std::fs::read_dir("/dev/dri")
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("renderD"))
        .collect();
    nodes.sort();
    nodes.first().map(|n| format!("/dev/dri/{}", n))
}

/// DRM primary nodes (`/dev/dri/cardN`), cards driving a connected display
/// first. Used by KMS capture, which must open the scan-out GPU.
pub fn card_nodes_display_first() -> Vec<String> {
    let mut cards = cards_with_connected_display();
    if let Ok(entries) = std::fs::read_dir("/dev/dri") {
        let mut rest: Vec<String> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("card") && !cards.contains(n))
            .collect();
        rest.sort();
        cards.extend(rest);
    }
    cards
        .into_iter()
        .map(|c| format!("/dev/dri/{}", c))
        .collect()
}

/// Names (`cardN`) of DRM cards with at least one connected connector.
fn cards_with_connected_display() -> Vec<String> {
    let mut cards_with_display: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        let mut names: Vec<String> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for name in names {
            // Connector entries look like "card2-DP-2".
            let Some((card, _connector)) = name.split_once('-') else {
                continue;
            };
            if !card.starts_with("card") {
                continue;
            }
            let status = std::fs::read_to_string(format!("/sys/class/drm/{}/status", name))
                .unwrap_or_default();
            if status.trim() == "connected" && !cards_with_display.iter().any(|c| c == card) {
                cards_with_display.push(card.to_string());
            }
        }
    }

    cards_with_display
}

fn render_node_of_card(card: &str) -> Option<String> {
    let dir = format!("/sys/class/drm/{}/device/drm", card);
    let mut nodes: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("renderD"))
        .collect();
    nodes.sort();
    nodes.first().map(|n| format!("/dev/dri/{}", n))
}

/// A DRM format together with the modifiers the capture GPU can import.
#[derive(Debug, Clone)]
pub struct DmabufFormat {
    pub drm_format: u32,
    pub modifiers: Vec<u64>,
}

// --- EGL / GBM FFI (subset) -------------------------------------------------

type EGLDisplay = *mut libc::c_void;
type EGLint = i32;
type EGLBoolean = u32;

const EGL_PLATFORM_GBM_KHR: u32 = 0x31D7;
const EGL_EXTENSIONS: EGLint = 0x3055;

type EglGetProcAddressFn = unsafe extern "C" fn(*const libc::c_char) -> *mut libc::c_void;
type EglGetPlatformDisplayExtFn =
    unsafe extern "C" fn(u32, *mut libc::c_void, *const EGLint) -> EGLDisplay;
type EglInitializeFn = unsafe extern "C" fn(EGLDisplay, *mut EGLint, *mut EGLint) -> EGLBoolean;
type EglTerminateFn = unsafe extern "C" fn(EGLDisplay) -> EGLBoolean;
type EglQueryStringFn = unsafe extern "C" fn(EGLDisplay, EGLint) -> *const libc::c_char;
type EglQueryDmaBufModifiersFn = unsafe extern "C" fn(
    EGLDisplay,
    EGLint,
    EGLint,
    *mut u64,
    *mut EGLBoolean,
    *mut EGLint,
) -> EGLBoolean;
type GbmCreateDeviceFn = unsafe extern "C" fn(libc::c_int) -> *mut libc::c_void;
type GbmDeviceDestroyFn = unsafe extern "C" fn(*mut libc::c_void);
type GbmPlaneCountFn = unsafe extern "C" fn(*mut libc::c_void, u32, u64) -> libc::c_int;

/// Query the DMA-BUF modifiers importable on `render_node` for every format in
/// [`PREFERRED_DRM_FORMATS`].
///
/// Formats for which the query fails are returned with an empty modifier list
/// (callers then offer them only through the shared-memory fallback).
pub fn query_dmabuf_formats(render_node: &str) -> Vec<DmabufFormat> {
    match query_dmabuf_formats_inner(render_node) {
        Ok(formats) => formats,
        Err(e) => {
            log::warn!("DMA-BUF modifier query on {} failed: {}", render_node, e);
            Vec::new()
        }
    }
}

fn query_dmabuf_formats_inner(render_node: &str) -> Result<Vec<DmabufFormat>, String> {
    let gbm = unsafe { libloading::Library::new("libgbm.so.1") }
        .map_err(|e| format!("libgbm.so.1: {}", e))?;
    let egl = unsafe { libloading::Library::new("libEGL.so.1") }
        .map_err(|e| format!("libEGL.so.1: {}", e))?;

    unsafe {
        let gbm_create_device: libloading::Symbol<GbmCreateDeviceFn> =
            gbm.get(b"gbm_create_device\0").map_err(|e| e.to_string())?;
        let gbm_device_destroy: libloading::Symbol<GbmDeviceDestroyFn> = gbm
            .get(b"gbm_device_destroy\0")
            .map_err(|e| e.to_string())?;
        // Optional (Mesa >= 21): lets us drop multi-plane modifiers (e.g. AMD
        // DCC with separate metadata planes) that the single-plane import
        // path cannot describe.
        let plane_count: Option<libloading::Symbol<GbmPlaneCountFn>> = gbm
            .get(b"gbm_device_get_format_modifier_plane_count\0")
            .ok();
        let egl_get_proc: libloading::Symbol<EglGetProcAddressFn> =
            egl.get(b"eglGetProcAddress\0").map_err(|e| e.to_string())?;
        let egl_initialize: libloading::Symbol<EglInitializeFn> =
            egl.get(b"eglInitialize\0").map_err(|e| e.to_string())?;
        let egl_terminate: libloading::Symbol<EglTerminateFn> =
            egl.get(b"eglTerminate\0").map_err(|e| e.to_string())?;
        let egl_query_string: libloading::Symbol<EglQueryStringFn> =
            egl.get(b"eglQueryString\0").map_err(|e| e.to_string())?;

        let get_platform_display = egl_get_proc(b"eglGetPlatformDisplayEXT\0".as_ptr() as _);
        if get_platform_display.is_null() {
            return Err("eglGetPlatformDisplayEXT unavailable".into());
        }
        let get_platform_display: EglGetPlatformDisplayExtFn =
            std::mem::transmute(get_platform_display);

        let path = std::ffi::CString::new(render_node).map_err(|e| e.to_string())?;
        let fd = libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC);
        if fd < 0 {
            return Err(format!("open: {}", std::io::Error::last_os_error()));
        }
        let gbm_dev = gbm_create_device(fd);
        if gbm_dev.is_null() {
            libc::close(fd);
            return Err("gbm_create_device failed".into());
        }

        let cleanup = |display: EGLDisplay| {
            if !display.is_null() {
                egl_terminate(display);
            }
            gbm_device_destroy(gbm_dev);
            libc::close(fd);
        };

        let display = get_platform_display(EGL_PLATFORM_GBM_KHR, gbm_dev, std::ptr::null());
        if display.is_null() {
            cleanup(display);
            return Err("eglGetPlatformDisplayEXT(GBM) returned no display".into());
        }
        let (mut major, mut minor) = (0, 0);
        if egl_initialize(display, &mut major, &mut minor) == 0 {
            cleanup(std::ptr::null_mut());
            return Err("eglInitialize failed".into());
        }

        let ext_ptr = egl_query_string(display, EGL_EXTENSIONS);
        let extensions = if ext_ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ext_ptr)
                .to_string_lossy()
                .into_owned()
        };
        if !extensions
            .split_whitespace()
            .any(|e| e == "EGL_EXT_image_dma_buf_import_modifiers")
        {
            cleanup(display);
            return Err("EGL_EXT_image_dma_buf_import_modifiers unsupported".into());
        }
        let query_ptr = egl_get_proc(b"eglQueryDmaBufModifiersEXT\0".as_ptr() as _);
        if query_ptr.is_null() {
            cleanup(display);
            return Err("eglQueryDmaBufModifiersEXT unavailable".into());
        }
        let query: EglQueryDmaBufModifiersFn = std::mem::transmute(query_ptr);

        let mut result = Vec::new();
        for &format in PREFERRED_DRM_FORMATS.iter() {
            let mut count: EGLint = 0;
            if query(
                display,
                format as EGLint,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut count,
            ) == 0
                || count <= 0
            {
                result.push(DmabufFormat {
                    drm_format: format,
                    modifiers: Vec::new(),
                });
                continue;
            }
            let mut modifiers = vec![0u64; count as usize];
            let mut external_only = vec![0u32; count as usize];
            if query(
                display,
                format as EGLint,
                count,
                modifiers.as_mut_ptr(),
                external_only.as_mut_ptr(),
                &mut count,
            ) == 0
            {
                result.push(DmabufFormat {
                    drm_format: format,
                    modifiers: Vec::new(),
                });
                continue;
            }
            modifiers.truncate(count.max(0) as usize);
            external_only.truncate(count.max(0) as usize);
            let mut usable: Vec<u64> = modifiers
                .into_iter()
                .zip(external_only)
                .filter(|(_, ext)| *ext == 0)
                .map(|(m, _)| m)
                .filter(|&m| match &plane_count {
                    Some(count) => count(gbm_dev, format, m) == 1,
                    // Without the query only trust layouts known to be single-plane.
                    None => m == DRM_FORMAT_MOD_LINEAR,
                })
                .collect();
            // Like Sunshine, also allow the implicit modifier so compositors
            // that allocate without explicit modifiers can still share buffers.
            if !usable.is_empty() && !usable.contains(&DRM_FORMAT_MOD_INVALID) {
                usable.push(DRM_FORMAT_MOD_INVALID);
            }
            result.push(DmabufFormat {
                drm_format: format,
                modifiers: usable,
            });
        }

        cleanup(display);
        log::info!(
            "DMA-BUF formats on {} (EGL {}.{}): {}",
            render_node,
            major,
            minor,
            result
                .iter()
                .map(|f| format!("{:08X}×{}", f.drm_format, f.modifiers.len()))
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(result)
    }
}

/// Map a DRM fourcc to the equivalent SPA video format.
///
/// SPA names describe byte order in memory, DRM names describe a
/// little-endian packed word, so `DRM_FORMAT_XRGB8888` is `SPA_VIDEO_FORMAT_BGRx`.
pub fn drm_to_spa_format(drm: u32) -> Option<u32> {
    use pipewire::spa::sys::*;
    Some(match drm {
        DRM_FORMAT_XRGB8888 => SPA_VIDEO_FORMAT_BGRx,
        DRM_FORMAT_ARGB8888 => SPA_VIDEO_FORMAT_BGRA,
        DRM_FORMAT_XBGR8888 => SPA_VIDEO_FORMAT_RGBx,
        DRM_FORMAT_ABGR8888 => SPA_VIDEO_FORMAT_RGBA,
        _ => return None,
    })
}

/// Map a SPA video format to the equivalent DRM fourcc.
#[allow(non_upper_case_globals)]
pub fn spa_to_drm_format(spa: u32) -> Option<u32> {
    use pipewire::spa::sys::*;
    Some(match spa {
        SPA_VIDEO_FORMAT_BGRx => DRM_FORMAT_XRGB8888,
        SPA_VIDEO_FORMAT_BGRA => DRM_FORMAT_ARGB8888,
        SPA_VIDEO_FORMAT_RGBx => DRM_FORMAT_XBGR8888,
        SPA_VIDEO_FORMAT_RGBA => DRM_FORMAT_ABGR8888,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drm_spa_round_trip() {
        for &f in PREFERRED_DRM_FORMATS.iter() {
            let spa = drm_to_spa_format(f).unwrap();
            assert_eq!(spa_to_drm_format(spa), Some(f));
        }
    }

    #[test]
    fn fourcc_values_match_drm_fourcc_h() {
        let fourcc = |s: &[u8; 4]| u32::from_le_bytes(*s);
        assert_eq!(DRM_FORMAT_XRGB8888, fourcc(b"XR24"));
        assert_eq!(DRM_FORMAT_ARGB8888, fourcc(b"AR24"));
        assert_eq!(DRM_FORMAT_XBGR8888, fourcc(b"XB24"));
        assert_eq!(DRM_FORMAT_ABGR8888, fourcc(b"AB24"));
    }
}
