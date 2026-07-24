//! EGL DMA-BUF import for GPU-tiled memory (NVIDIA Wayland).
//!
//! On NVIDIA/Wayland setups, DMA-BUF buffers use block-linear tiling (e.g.,
//! `DRM_FORMAT_MOD_NVIDIA_16BX2_BLOCK`). CPU mmap cannot read tiled GPU memory
//! — every pixel format produces noise. DMA_BUF_IOCTL_SYNC doesn't help.
//!
//! This module imports DMA-BUFs through EGL (`EGL_LINUX_DMA_BUF_EXT`), which
//! goes through the GPU memory controller and handles tiling transparently.
//! The result is read back via `glReadPixels` as linear BGRA data.
//!
//! ## Architecture
//!
//! ```text
//! DMA-BUF fd → eglCreateImage → GL texture → FBO → glReadPixels → Vec<u8>
//! ```
//!
//! The EGL context is created once per [`EglImporter`] instance and must be
//! used from a single thread (the PipeWire callback thread).

use std::os::unix::io::RawFd;
use std::sync::OnceLock;

use crate::error::MediaError;

// ---------------------------------------------------------------------------
// EGL / GL type aliases
// ---------------------------------------------------------------------------

type EGLBoolean = libc::c_uint;
type EGLint = libc::c_int;
type EGLLabel = *const libc::c_char; // KHR_debug label
type EGLAttrib = libc::intptr_t;
type EGLDisplay = *mut std::ffi::c_void;
type EGLConfig = *mut std::ffi::c_void;
type EGLContext = *mut std::ffi::c_void;
type EGLImage = *mut std::ffi::c_void;
type EGLDeviceEXT = *mut std::ffi::c_void;
type EGLSurface = *mut std::ffi::c_void;

type GLuint = libc::c_uint;
type GLint = libc::c_int;
type GLenum = libc::c_uint;
type GLsizei = libc::c_int;
type GLboolean = libc::c_uchar;
type GLbitfield = libc::c_uint;

// ---------------------------------------------------------------------------
// EGL constants
// ---------------------------------------------------------------------------

const EGL_TRUE: EGLBoolean = 1;
const EGL_FALSE: EGLBoolean = 0;
const EGL_NO_CONTEXT: EGLContext = std::ptr::null_mut();
const EGL_NO_DISPLAY: EGLDisplay = std::ptr::null_mut();
const EGL_NO_SURFACE: EGLSurface = std::ptr::null_mut();
const EGL_NO_IMAGE: EGLImage = std::ptr::null_mut();
const EGL_DEFAULT_DISPLAY: EGLDisplay = 0 as EGLDisplay;
const EGL_NONE: EGLint = 0x3038;
const EGL_SUCCESS: EGLint = 0x3000;

const EGL_ALPHA_SIZE: EGLint = 0x3021;
const EGL_BLUE_SIZE: EGLint = 0x3022;
const EGL_GREEN_SIZE: EGLint = 0x3023;
const EGL_RED_SIZE: EGLint = 0x3024;
const EGL_RENDERABLE_TYPE: EGLint = 0x3040;
const EGL_SURFACE_TYPE: EGLint = 0x3033;
const EGL_OPENGL_BIT: EGLint = 0x0008;
const EGL_PBUFFER_BIT: EGLint = 0x0001;
const EGL_HEIGHT: EGLint = 0x3056;
const EGL_WIDTH: EGLint = 0x3057;
const EGL_CONTEXT_CLIENT_VERSION: EGLint = 0x3098;
const EGL_OPENGL_API: EGLint = 0x30A2;
const EGL_CONTEXT_MAJOR_VERSION: EGLint = 0x3098;
const EGL_CONTEXT_MINOR_VERSION: EGLint = 0x30FB;
const EGL_CONTEXT_OPENGL_PROFILE_MASK: EGLint = 0x30FD;
const EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT: EGLint = 0x00000001;

const EGL_LINUX_DMA_BUF_EXT: EGLint = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: EGLint = 0x3271;
const EGL_DMA_BUF_PLANE0_FD_EXT: EGLint = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EGLint = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: EGLint = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EGLint = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EGLint = 0x3444;

// GL constants
const GL_TEXTURE_2D: GLenum = 0x0DE1;
const GL_TEXTURE_MIN_FILTER: GLenum = 0x2801;
const GL_TEXTURE_MAG_FILTER: GLenum = 0x2800;
const GL_LINEAR: GLint = 0x2601;
const GL_RGBA: GLenum = 0x1908;
const GL_BGRA: GLenum = 0x80E1;
const GL_UNSIGNED_BYTE: GLenum = 0x1401;
const GL_FRAMEBUFFER: GLenum = 0x8D40;
const GL_READ_FRAMEBUFFER: GLenum = 0x8CA8;
const GL_DRAW_FRAMEBUFFER: GLenum = 0x8CA9;
const GL_COLOR_ATTACHMENT0: GLenum = 0x8CE0;
const GL_FRAMEBUFFER_COMPLETE: GLenum = 0x8CD5;
const GL_NO_ERROR: GLenum = 0;
const GL_COLOR_BUFFER_BIT: u32 = 0x00004000;
const GL_NEAREST: GLint = 0x2600;
const GL_PROJECTION: GLenum = 0x1701;
const GL_MODELVIEW: GLenum = 0x1700;
const GL_QUADS: GLenum = 0x0007;

const EGL_PLATFORM_DEVICE_EXT: EGLint = 0x313F;

// ---------------------------------------------------------------------------
// EGL function pointer types
// ---------------------------------------------------------------------------

type EglGetErrorFn = unsafe extern "C" fn() -> EGLint;
type EglBindAPIFn = unsafe extern "C" fn(api: EGLint) -> EGLBoolean;
type EglGetDisplayFn = unsafe extern "C" fn(display_id: EGLDisplay) -> EGLDisplay;
type EglGetPlatformDisplayEXTFn =
    unsafe extern "C" fn(platform: EGLint, native_display: *mut std::ffi::c_void, attrib_list: *const EGLint) -> EGLDisplay;
type EglInitializeFn =
    unsafe extern "C" fn(dpy: EGLDisplay, major: *mut EGLint, minor: *mut EGLint) -> EGLBoolean;
type EglChooseConfigFn =
    unsafe extern "C" fn(dpy: EGLDisplay, attrib_list: *const EGLint, configs: *mut EGLConfig, config_size: EGLint, num_config: *mut EGLint) -> EGLBoolean;
type EglCreateContextFn =
    unsafe extern "C" fn(dpy: EGLDisplay, config: EGLConfig, share_context: EGLContext, attrib_list: *const EGLint) -> EGLContext;
type EglMakeCurrentFn =
    unsafe extern "C" fn(dpy: EGLDisplay, draw: EGLSurface, read: EGLSurface, ctx: EGLContext) -> EGLBoolean;
type EglDestroyContextFn =
    unsafe extern "C" fn(dpy: EGLDisplay, ctx: EGLContext) -> EGLBoolean;
type EglTerminateFn =
    unsafe extern "C" fn(dpy: EGLDisplay) -> EGLBoolean;
type EglQueryStringFn =
    unsafe extern "C" fn(dpy: EGLDisplay, name: EGLint) -> *const libc::c_char;
type EglCreateImageFn =
    unsafe extern "C" fn(dpy: EGLDisplay, ctx: EGLContext, target: EGLint, buffer: *mut std::ffi::c_void, attrib_list: *const EGLAttrib) -> EGLImage;
/// eglCreateImageKHR uses EGLint (32-bit) attributes — the correct calling
/// convention for EGL_KHR_image_base which most NVIDIA drivers implement.
type EglCreateImageKHRFn =
    unsafe extern "C" fn(dpy: EGLDisplay, ctx: EGLContext, target: EGLint, buffer: *mut std::ffi::c_void, attrib_list: *const EGLint) -> EGLImage;
type EglDestroyImageFn =
    unsafe extern "C" fn(dpy: EGLDisplay, image: EGLImage) -> EGLBoolean;
type EglQueryDevicesEXTFn =
    unsafe extern "C" fn(max_devices: EGLint, devices: *mut EGLDeviceEXT, num_devices: *mut EGLint) -> EGLBoolean;
type EglGetProcAddressFn =
    unsafe extern "C" fn(procname: *const libc::c_char) -> *mut std::ffi::c_void;

// ---------------------------------------------------------------------------
// GL function pointer types
// ---------------------------------------------------------------------------

type GlGenTexturesFn = unsafe extern "C" fn(n: GLsizei, textures: *mut GLuint);
type GlDeleteTexturesFn = unsafe extern "C" fn(n: GLsizei, textures: *const GLuint);
type GlBindTextureFn = unsafe extern "C" fn(target: GLenum, texture: GLuint);
type GlTexParameteriFn = unsafe extern "C" fn(target: GLenum, pname: GLenum, param: GLint);
type GlGenFramebuffersFn = unsafe extern "C" fn(n: GLsizei, framebuffers: *mut GLuint);
type GlDeleteFramebuffersFn = unsafe extern "C" fn(n: GLsizei, framebuffers: *const GLuint);
type GlBindFramebufferFn = unsafe extern "C" fn(target: GLenum, framebuffer: GLuint);
type GlFramebufferTexture2DFn =
    unsafe extern "C" fn(target: GLenum, attachment: GLenum, textarget: GLenum, texture: GLuint, level: GLint);
type GlCheckFramebufferStatusFn = unsafe extern "C" fn(target: GLenum) -> GLenum;
type GlReadPixelsFn = unsafe extern "C" fn(
    x: GLint, y: GLint, width: GLsizei, height: GLsizei,
    format: GLenum, type_: GLenum, pixels: *mut std::ffi::c_void,
);
type GlGetErrorFn = unsafe extern "C" fn() -> GLenum;
type GlEGLImageTargetTexture2DOESFn =
    unsafe extern "C" fn(target: GLenum, image: EGLImage);
type GlTexImage2DFn = unsafe extern "C" fn(
    target: GLenum, level: GLint, internalformat: GLint,
    width: GLsizei, height: GLsizei, border: GLint,
    format: GLenum, type_: GLenum, pixels: *const std::ffi::c_void,
);
type GlBlitFramebufferFn = unsafe extern "C" fn(
    srcX0: GLint, srcY0: GLint, srcX1: GLint, srcY1: GLint,
    dstX0: GLint, dstY0: GLint, dstX1: GLint, dstY1: GLint,
    mask: u32, filter: GLenum,
);
type GlEnableFn = unsafe extern "C" fn(cap: GLenum);
type GlDisableFn = unsafe extern "C" fn(cap: GLenum);
type GlViewportFn = unsafe extern "C" fn(x: GLint, y: GLint, width: GLsizei, height: GLsizei);
type GlMatrixModeFn = unsafe extern "C" fn(mode: GLenum);
type GlLoadIdentityFn = unsafe extern "C" fn();
type GlOrthoFn = unsafe extern "C" fn(left: f64, right: f64, bottom: f64, top: f64, near_val: f64, far_val: f64);
type GlBeginFn = unsafe extern "C" fn(mode: GLenum);
type GlEndFn = unsafe extern "C" fn();
type GlTexCoord2fFn = unsafe extern "C" fn(s: f32, t: f32);
type GlVertex2fFn = unsafe extern "C" fn(x: f32, y: f32);

// ---------------------------------------------------------------------------
// EglImporter
// ---------------------------------------------------------------------------

/// Manages an EGL context and provides DMA-BUF → CPU conversion.
///
/// The EGL context is headless (no window surface) and only used for importing
/// DMA-BUFs as GL textures for readback. Thread-safe to create, but the GL
/// context must only be used from one thread at a time (it is made current on
/// the thread that calls [`import_dmabuf`]).
pub struct EglImporter {
    egllib: libloading::Library,
    gleslib: libloading::Library,
    display: EGLDisplay,
    context: EGLContext,
    config: EGLConfig,
    surface: EGLSurface,
    /// Cached DRM modifier found during first-frame probing.
    /// Once a working modifier is found, it's reused for all subsequent imports.
    cached_modifier: Option<u64>,

    // EGL function pointers
    eglGetError: EglGetErrorFn,
    eglGetProcAddress: EglGetProcAddressFn,
    eglInitialize: EglInitializeFn,
    eglCreateContext: EglCreateContextFn,
    eglMakeCurrent: EglMakeCurrentFn,
    eglDestroyContext: EglDestroyContextFn,
    eglTerminate: EglTerminateFn,
    eglCreateImage: EglCreateImageFn,
    eglCreateImageKHR: Option<EglCreateImageKHRFn>,
    eglDestroyImage: EglDestroyImageFn,

    // GL function pointers
    glGenTextures: GlGenTexturesFn,
    glDeleteTextures: GlDeleteTexturesFn,
    glBindTexture: GlBindTextureFn,
    glTexParameteri: GlTexParameteriFn,
    glGenFramebuffers: GlGenFramebuffersFn,
    glDeleteFramebuffers: GlDeleteFramebuffersFn,
    glBindFramebuffer: GlBindFramebufferFn,
    glFramebufferTexture2D: GlFramebufferTexture2DFn,
    glCheckFramebufferStatus: GlCheckFramebufferStatusFn,
    glReadPixels: GlReadPixelsFn,
    glGetError: GlGetErrorFn,
    glEGLImageTargetTexture2DOES: GlEGLImageTargetTexture2DOESFn,
    glTexImage2D: GlTexImage2DFn,
    glBlitFramebuffer: Option<GlBlitFramebufferFn>,
    glEnable: Option<GlEnableFn>,
    glDisable: Option<GlDisableFn>,
    glViewport: Option<GlViewportFn>,
    glMatrixMode: Option<GlMatrixModeFn>,
    glLoadIdentity: Option<GlLoadIdentityFn>,
    glOrtho: Option<GlOrthoFn>,
    glBegin: Option<GlBeginFn>,
    glEnd: Option<GlEndFn>,
    glTexCoord2f: Option<GlTexCoord2fFn>,
    glVertex2f: Option<GlVertex2fFn>,
}

/// Safely load a symbol from a library, returning a descriptive error on failure.
macro_rules! load_sym {
    ($lib:expr, $name:literal) => {
        unsafe {
            $lib.get::<*const u8>($name.as_bytes())
                .map_err(|e| MediaError::CaptureError(format!(
                    "Failed to load {}: {}", $name, e
                )))?
                .into_raw()
                .cast::<std::ffi::c_void>()
        }
    };
}

impl EglImporter {
    /// Create a new EGL importer with a headless OpenGL context.
    ///
    /// Dynamically loads `libEGL.so.1` and `libGL.so.1` at runtime so the
    /// same binary can run on systems without these libraries (falling back to
    /// mmap-based DMA-BUF read).
    ///
    /// Uses full **OpenGL** (not GLES2) because NVIDIA's EGL implementation
    /// only supports DMA-BUF import via `glEGLImageTargetTexture2DOES` into
    /// `GL_TEXTURE_2D` on full OpenGL contexts. GLES contexts return
    /// `GL_INVALID_OPERATION (0x0502)` — this is the approach Sunshine uses.
    pub fn new() -> Result<Self, MediaError> {
        // --- Load libraries ---
        let (egllib, gleslib, vendor) = Self::load_egl_libs()?;
        log::info!("EGL library vendor preference: {}", vendor);

        // --- Load EGL function pointers ---
        let eglGetError: EglGetErrorFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglGetError")) };
        let eglBindAPI: EglBindAPIFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglBindAPI")) };
        let eglGetDisplay: EglGetDisplayFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglGetDisplay")) };
        let eglQueryString: EglQueryStringFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglQueryString")) };
        let eglInitialize: EglInitializeFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglInitialize")) };
        let eglChooseConfig: EglChooseConfigFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglChooseConfig")) };
        let eglCreateContext: EglCreateContextFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglCreateContext")) };
        let eglMakeCurrent: EglMakeCurrentFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglMakeCurrent")) };
        let eglDestroyContext: EglDestroyContextFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglDestroyContext")) };
        let eglTerminate: EglTerminateFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglTerminate")) };
        let eglCreateImage: EglCreateImageFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglCreateImage")) };
        let eglDestroyImage: EglDestroyImageFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglDestroyImage")) };
        // eglCreateImageKHR will be loaded via eglGetProcAddress after context init (deferred)
        let eglGetProcAddress: EglGetProcAddressFn = unsafe { std::mem::transmute(load_sym!(egllib, "eglGetProcAddress")) };

        // --- Load GL function pointers (desktop OpenGL, from libGL) ---
        let glGenTextures: GlGenTexturesFn = unsafe { std::mem::transmute(load_sym!(gleslib, "glGenTextures")) };
        let glDeleteTextures: GlDeleteTexturesFn = unsafe { std::mem::transmute(load_sym!(gleslib, "glDeleteTextures")) };
        let glBindTexture: GlBindTextureFn = unsafe { std::mem::transmute(load_sym!(gleslib, "glBindTexture")) };
        let glTexParameteri: GlTexParameteriFn = unsafe { std::mem::transmute(load_sym!(gleslib, "glTexParameteri")) };
        let glGenFramebuffers: GlGenFramebuffersFn = unsafe { std::mem::transmute(load_sym!(gleslib, "glGenFramebuffers")) };
        let glDeleteFramebuffers: GlDeleteFramebuffersFn = unsafe { std::mem::transmute(load_sym!(gleslib, "glDeleteFramebuffers")) };
        let glBindFramebuffer: GlBindFramebufferFn = unsafe { std::mem::transmute(load_sym!(gleslib, "glBindFramebuffer")) };
        let glFramebufferTexture2D: GlFramebufferTexture2DFn = unsafe { std::mem::transmute(load_sym!(gleslib, "glFramebufferTexture2D")) };
        let glCheckFramebufferStatus: GlCheckFramebufferStatusFn = unsafe { std::mem::transmute(load_sym!(gleslib, "glCheckFramebufferStatus")) };
        let glReadPixels: GlReadPixelsFn = unsafe { std::mem::transmute(load_sym!(gleslib, "glReadPixels")) };
        let glGetError: GlGetErrorFn = unsafe { std::mem::transmute(load_sym!(gleslib, "glGetError")) };

        // glEGLImageTargetTexture2DOES is an extension that must be loaded
        // via eglGetProcAddress — it is NOT exported from libGLESv2.so.2 directly.
        // Defer loading until after EGL is initialized.

        // --- Create EGL display ---
        // Strategy: try EGL_EXT_platform_device with the first available EGL device,
        // fall back to eglGetDisplay(EGL_DEFAULT_DISPLAY) which usually works with
        // NVIDIA's EGL implementation even without a display server connection.
        let display = Self::create_display(
            &egllib,
            &eglGetDisplay,
            &eglGetError,
            &eglQueryString,
            &eglGetProcAddress,
        )?;

        // --- Initialize EGL ---
        let mut major: EGLint = 0;
        let mut minor: EGLint = 0;
        let ok = unsafe { eglInitialize(display, &mut major, &mut minor) };
        if ok == EGL_FALSE {
            let err = unsafe { eglGetError() };
            return Err(MediaError::CaptureError(format!(
                "eglInitialize failed: 0x{:04X}", err
            )));
        }
        log::info!(
            "EGL initialized: vendor=\"{}\" version={}.{}",
            unsafe {
                let s = eglQueryString(display, 0x3053); // EGL_VENDOR
                if s.is_null() {
                    "(null)"
                } else {
                    std::ffi::CStr::from_ptr(s).to_str().unwrap_or("(invalid utf8)")
                }
            },
            major,
            minor,
        );

        // --- Bind OpenGL API ---
        // CRITICAL: Must call eglBindAPI(EGL_OPENGL_API) before eglChooseConfig
        // and eglCreateContext. Without this, EGL defaults to EGL_OPENGL_ES_API,
        // and NVIDIA GLES contexts cannot import DMA-BUFs via
        // glEGLImageTargetTexture2DOES into GL_TEXTURE_2D.
        let ok = unsafe { eglBindAPI(EGL_OPENGL_API) };
        if ok == EGL_FALSE {
            let err = unsafe { eglGetError() };
            return Err(MediaError::CaptureError(format!(
                "eglBindAPI(EGL_OPENGL_API) failed: 0x{:04X}", err
            )));
        }

        // --- Choose config (full OpenGL, not GLES) ---
        let config_attribs: [EGLint; 13] = [
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_NONE,
        ];
        let mut config: EGLConfig = std::ptr::null_mut();
        let mut num_configs: EGLint = 0;
        let ok = unsafe {
            eglChooseConfig(display, config_attribs.as_ptr(), &mut config, 1, &mut num_configs)
        };
        if ok == EGL_FALSE || num_configs == 0 {
            let err = unsafe { eglGetError() };
            return Err(MediaError::CaptureError(format!(
                "eglChooseConfig failed: 0x{:04X} (num_configs={})", err, num_configs
            )));
        }

        // --- Create OpenGL context (compatibility profile) ---
        // Sunshine uses EGL_CONTEXT_CLIENT_VERSION=3 without a profile mask,
        // which defaults to the compatibility profile. Core profile strips
        // GL_OES_EGL_image which is needed for glEGLImageTargetTexture2DOES.
        let ctx_attribs: [EGLint; 3] = [
            EGL_CONTEXT_MAJOR_VERSION, 3,
            EGL_NONE,
        ];
        let context = unsafe { eglCreateContext(display, config, EGL_NO_CONTEXT, ctx_attribs.as_ptr()) };
        if context.is_null() {
            let err = unsafe { eglGetError() };
            return Err(MediaError::CaptureError(format!(
                "eglCreateContext failed: 0x{:04X}", err
            )));
        }

        // --- Create a tiny PBuffer surface for context binding ---
        // Desktop OpenGL may not support surfaceless contexts
        // (EGL_KHR_surfaceless_context) as well as GLES. Create a
        // 1x1 PBuffer to ensure proper context binding.
        type EglCreatePbufferSurfaceFn =
            unsafe extern "C" fn(dpy: EGLDisplay, config: EGLConfig, attrib_list: *const EGLint) -> EGLSurface;
        let eglCreatePbufferSurface: EglCreatePbufferSurfaceFn =
            unsafe { std::mem::transmute(load_sym!(egllib, "eglCreatePbufferSurface")) };

        let pbuf_attribs: [EGLint; 5] = [
            EGL_WIDTH, 1,
            EGL_HEIGHT, 1,
            EGL_NONE,
        ];
        let pbuf_surface = unsafe { eglCreatePbufferSurface(display, config, pbuf_attribs.as_ptr()) };
        let (read_surf, draw_surf) = if pbuf_surface.is_null() {
            log::warn!("eglCreatePbufferSurface failed, falling back to surfaceless");
            (EGL_NO_SURFACE, EGL_NO_SURFACE)
        } else {
            log::info!("Created 1x1 PBuffer surface for context binding");
            (pbuf_surface, pbuf_surface)
        };

        // --- Make context current ---
        let ok = unsafe { eglMakeCurrent(display, draw_surf, read_surf, context) };
        if ok == EGL_FALSE {
            let err = unsafe { eglGetError() };
            unsafe { eglDestroyContext(display, context); }
            return Err(MediaError::CaptureError(format!(
                "eglMakeCurrent failed: 0x{:04X}", err
            )));
        }

        // --- Load GL extensions ---
        // glEGLImageTargetTexture2DOES is exported by GLVND's libGL.so.1.
        // Loading from libGL ensures proper GLVND GL dispatch to the current
        // context's vendor driver. eglGetProcAddress may return an EGL-side
        // stub that doesn't go through GLVND GL dispatch correctly.
        let glEGLImageTargetTexture2DOES: GlEGLImageTargetTexture2DOESFn = unsafe {
            // Try libGL.so.1 first (GLVND GL dispatch)
            let from_gl = gleslib.get::<*const u8>(b"glEGLImageTargetTexture2DOES");
            if let Ok(ptr) = from_gl {
                log::info!("glEGLImageTargetTexture2DOES loaded from libGL (GLVND dispatch)");
                std::mem::transmute(ptr.into_raw().cast::<std::ffi::c_void>())
            } else {
                // Fallback: eglGetProcAddress
                let name = b"glEGLImageTargetTexture2DOES\0".as_ptr() as *const libc::c_char;
                let ptr = eglGetProcAddress(name);
                if ptr.is_null() {
                    eglDestroyContext(display, context);
                    return Err(MediaError::CaptureError(
                        "glEGLImageTargetTexture2DOES not available from libGL or eglGetProcAddress".into(),
                    ));
                }
                log::info!("glEGLImageTargetTexture2DOES loaded via eglGetProcAddress (fallback)");
                std::mem::transmute(ptr)
            }
        };

        // Load eglCreateImageKHR via eglGetProcAddress (must be done after
        // EGL is initialized). This is the KHR extension variant that uses
        // EGLint (32-bit) attributes — most drivers implement this.
        let eglCreateImageKHR: Option<EglCreateImageKHRFn> = unsafe {
            let name = b"eglCreateImageKHR\0".as_ptr() as *const libc::c_char;
            let ptr = eglGetProcAddress(name);
            if ptr.is_null() {
                log::info!("eglCreateImageKHR not available, will use eglCreateImage (EGL 1.5)");
                None
            } else {
                log::info!("eglCreateImageKHR loaded via eglGetProcAddress");
                Some(std::mem::transmute(ptr))
            }
        };

        // Query supported DMA-BUF formats for diagnostic
        unsafe {
            type QueryDmaBufFormatsFn = unsafe extern "C" fn(
                dpy: EGLDisplay, max_formats: EGLint, formats: *mut EGLint, num_formats: *mut EGLint
            ) -> EGLBoolean;
            let name = b"eglQueryDmaBufFormatsEXT\0".as_ptr() as *const libc::c_char;
            let ptr = eglGetProcAddress(name);
            if !ptr.is_null() {
                let query_fn: QueryDmaBufFormatsFn = std::mem::transmute(ptr);
                let mut num_formats: EGLint = 0;
                if query_fn(display, 0, std::ptr::null_mut(), &mut num_formats) == EGL_TRUE && num_formats > 0 {
                    let mut formats = vec![0i32; num_formats as usize];
                    if query_fn(display, num_formats, formats.as_mut_ptr(), &mut num_formats) == EGL_TRUE {
                        // Log formats of interest
                        let xrgb = formats.contains(&(0x34325258u32 as i32));
                        let argb = formats.contains(&(0x34325241u32 as i32));
                        let xbgr = formats.contains(&(0x34324258u32 as i32));
                        let abgr = formats.contains(&(0x34324241u32 as i32));
                        log::info!(
                            "EGL DMA-BUF formats: {} total, XRGB={}, ARGB={}, XBGR={}, ABGR={}",
                            num_formats, xrgb, argb, xbgr, abgr
                        );
                    }
                } else {
                    log::info!("eglQueryDmaBufFormatsEXT: {} formats (or query failed)", num_formats);
                }
            } else {
                log::info!("eglQueryDmaBufFormatsEXT not available");
            }
        }

        let glTexImage2D: GlTexImage2DFn = unsafe { std::mem::transmute(load_sym!(gleslib, "glTexImage2D")) };
        let glBlitFramebuffer: Option<GlBlitFramebufferFn> = unsafe {
            let ptr = (eglGetProcAddress)(b"glBlitFramebuffer\0".as_ptr() as *const libc::c_char);
            if !ptr.is_null() {
                Some(std::mem::transmute(ptr))
            } else {
                gleslib.get::<*const u8>(b"glBlitFramebuffer\0")
                    .ok()
                    .map(|sym| std::mem::transmute(sym.into_raw().cast::<std::ffi::c_void>()))
            }
        };

        log::info!("EGL OpenGL context created successfully (headless, compat profile), glBlitFramebuffer available: {}", glBlitFramebuffer.is_some());

        type GlGetStringFn = unsafe extern "C" fn(name: GLenum) -> *const libc::c_char;
        const GL_RENDERER: GLenum = 0x1F01;
        const GL_EXTENSIONS: GLenum = 0x1F03;
        if let Ok(fn_ptr) = unsafe { gleslib.get::<GlGetStringFn>(b"glGetString\0") } {
            let gl_get_string: GlGetStringFn = *fn_ptr;
            let renderer = unsafe {
                let s = gl_get_string(GL_RENDERER);
                if s.is_null() { "(null)" } else {
                    std::ffi::CStr::from_ptr(s).to_str().unwrap_or("(invalid)")
                }
            };
            let extensions = unsafe {
                let s = gl_get_string(GL_EXTENSIONS);
                if s.is_null() { "(null)" } else {
                    std::ffi::CStr::from_ptr(s).to_str().unwrap_or("(invalid)")
                }
            };
            log::info!("GL Renderer: {}, extensions count: {}", renderer, extensions.split_whitespace().count());
        }

        macro_rules! load_gl_opt {
            ($name:literal) => {
                unsafe {
                    gleslib.get::<*const u8>($name.as_bytes())
                        .ok()
                        .map(|sym| std::mem::transmute(sym.into_raw().cast::<std::ffi::c_void>()))
                }
            };
        }

        let glEnable: Option<GlEnableFn> = load_gl_opt!("glEnable\0");
        let glDisable: Option<GlDisableFn> = load_gl_opt!("glDisable\0");
        let glViewport: Option<GlViewportFn> = load_gl_opt!("glViewport\0");
        let glMatrixMode: Option<GlMatrixModeFn> = load_gl_opt!("glMatrixMode\0");
        let glLoadIdentity: Option<GlLoadIdentityFn> = load_gl_opt!("glLoadIdentity\0");
        let glOrtho: Option<GlOrthoFn> = load_gl_opt!("glOrtho\0");
        let glBegin: Option<GlBeginFn> = load_gl_opt!("glBegin\0");
        let glEnd: Option<GlEndFn> = load_gl_opt!("glEnd\0");
        let glTexCoord2f: Option<GlTexCoord2fFn> = load_gl_opt!("glTexCoord2f\0");
        let glVertex2f: Option<GlVertex2fFn> = load_gl_opt!("glVertex2f\0");

        Ok(Self {
            egllib,
            gleslib,
            display,
            context,
            config,
            surface: draw_surf,
            cached_modifier: None,
            eglGetError,
            eglGetProcAddress,
            eglInitialize,
            eglCreateContext,
            eglMakeCurrent,
            eglDestroyContext,
            eglTerminate,
            eglCreateImage,
            eglCreateImageKHR,
            eglDestroyImage,
            glGenTextures,
            glDeleteTextures,
            glBindTexture,
            glTexParameteri,
            glGenFramebuffers,
            glDeleteFramebuffers,
            glBindFramebuffer,
            glFramebufferTexture2D,
            glCheckFramebufferStatus,
            glReadPixels,
            glGetError,
            glEGLImageTargetTexture2DOES,
            glTexImage2D,
            glBlitFramebuffer,
            glEnable,
            glDisable,
            glViewport,
            glMatrixMode,
            glLoadIdentity,
            glOrtho,
            glBegin,
            glEnd,
            glTexCoord2f,
            glVertex2f,
        })
    }

    /// Load EGL and GLES libraries for DMA-BUF import.
    ///
    /// Prefers GLVND dispatch libraries (`libEGL.so.1` / `libGLESv2.so.2`)
    /// which export standard EGL/GLES symbols and internally route to the
    /// correct vendor driver (NVIDIA, Mesa, etc.). The NVIDIA vendor
    /// libraries (`libEGL_nvidia.so.0`) are GLVND *plugins* that only
    /// export `__egl_Main` — loading them directly fails on the first
    /// `eglGetError` lookup. They are only tried as a last resort for
    /// headless setups where GLVND is not installed.
    ///
    /// Returns `(egllib, gllib, vendor_name)`.
    fn load_egl_libs() -> Result<(libloading::Library, libloading::Library, &'static str), MediaError> {
        // 1. GLVND dispatch — the standard path on modern Linux desktops.
        //    libEGL.so.1 exports all standard EGL symbols and dispatches to
        //    the correct vendor driver (NVIDIA, Mesa, etc.) automatically.
        //    libGL.so.1 exports desktop OpenGL symbols (needed for DMA-BUF import).
        if let Ok(egl) = unsafe { libloading::Library::new("libEGL.so.1") } {
            if let Ok(gl) = unsafe { libloading::Library::new("libGL.so.1") } {
                // Verify that the critical symbol is actually exported
                let has_egl_get_error = unsafe {
                    egl.get::<*const u8>(b"eglGetError").is_ok()
                };
                if has_egl_get_error {
                    return Ok((egl, gl, "GLVND"));
                }
                log::warn!("libEGL.so.1 loaded but missing eglGetError, trying NVIDIA vendor libs");
            }
        }

        // 2. NVIDIA vendor libraries directly (headless/DRM setups without GLVND)
        if let Ok(lib) = unsafe { libloading::Library::new("libEGL_nvidia.so.0") } {
            if let Ok(gl) = unsafe { libloading::Library::new("libGL_nvidia.so.0") } {
                return Ok((lib, gl, "NVIDIA (direct)"));
            }
            // Try GLVND GL with NVIDIA EGL
            if let Ok(gl) = unsafe { libloading::Library::new("libGL.so.1") } {
                return Ok((lib, gl, "NVIDIA EGL + GLVND GL"));
            }
        }

        Err(MediaError::CaptureError(
            "Failed to load EGL/GL libraries: neither libEGL.so.1+libGL.so.1 (GLVND) nor NVIDIA vendor libs available".into(),
        ))
    }

    /// Try to create an EGL display using the best available method.
    ///
    /// On NVIDIA+GLVND setups, DMA-BUF import requires the EGL context to be
    /// tied to the same DRM render device that produced the DMA-BUF. We try
    /// GBM platform first (renders on the actual GPU), then EGL device, then
    /// the default display.
    fn create_display(
        _egllib: &libloading::Library,
        eglGetDisplay: &EglGetDisplayFn,
        eglGetError: &EglGetErrorFn,
        _eglQueryString: &EglQueryStringFn,
        eglGetProcAddress: &EglGetProcAddressFn,
    ) -> Result<EGLDisplay, MediaError> {
        // Method 0: EGL GBM platform — ties EGL context to the DRM render node
        // so that DMA-BUF import works correctly with the same GPU that
        // produced the buffers (compositor and EGL on the same device).
        let get_platform_display_ptr = unsafe {
            eglGetProcAddress(b"eglGetPlatformDisplayEXT\0".as_ptr() as *const libc::c_char)
        };
        if !get_platform_display_ptr.is_null() {
            let eglGetPlatformDisplayEXT: EglGetPlatformDisplayEXTFn =
                unsafe { std::mem::transmute(get_platform_display_ptr) };

            // Try GBM platform with render node
            const EGL_PLATFORM_GBM_KHR: EGLint = 0x31D7;
            if let Ok(gbmlib) = unsafe { libloading::Library::new("libgbm.so.1") } {
                type GbmCreateDeviceFn = unsafe extern "C" fn(fd: libc::c_int) -> *mut std::ffi::c_void;
                if let Ok(gbm_create_device) = unsafe { gbmlib.get::<GbmCreateDeviceFn>(b"gbm_create_device") } {
                    let gbm_create_device = *gbm_create_device;
                    // Try common render nodes
                    for render_node in &["/dev/dri/renderD128", "/dev/dri/renderD129"] {
                        let c_path = std::ffi::CString::new(*render_node).unwrap();
                        let drm_fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
                        if drm_fd < 0 {
                            continue;
                        }
                        let gbm_dev = unsafe { gbm_create_device(drm_fd) };
                        if gbm_dev.is_null() {
                            unsafe { libc::close(drm_fd); }
                            continue;
                        }
                        let attrs: [EGLint; 1] = [EGL_NONE];
                        let display = unsafe {
                            eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_KHR, gbm_dev, attrs.as_ptr())
                        };
                        if !display.is_null() {
                            log::info!("EGL display created via GBM platform ({}) — DMA-BUF import should work", render_node);
                            return Ok(display);
                        }
                        // GBM display failed — clean up and try next node
                        // Note: we don't have gbm_device_destroy loaded but that's ok
                        // since we're about to try the next approach
                        unsafe { libc::close(drm_fd); }
                    }
                }
                log::info!("GBM platform: could not create display on any render node");
            } else {
                log::info!("libgbm.so.1 not available, skipping GBM platform");
            }
        }

        // Method 1: EGL_EXT_platform_device — query EGL devices and use the first GPU.
        // These are EGL extension functions that must be loaded via
        // eglGetProcAddress, not dlsym — GLVND doesn't export them directly.
        let query_devices_ptr = unsafe {
            eglGetProcAddress(b"eglQueryDevicesEXT\0".as_ptr() as *const libc::c_char)
        };
        if !get_platform_display_ptr.is_null() && !query_devices_ptr.is_null() {
            let eglGetPlatformDisplayEXT: EglGetPlatformDisplayEXTFn =
                unsafe { std::mem::transmute(get_platform_display_ptr) };
            let eglQueryDevicesEXT: EglQueryDevicesEXTFn = unsafe { std::mem::transmute(query_devices_ptr) };
            let mut devices: [EGLDeviceEXT; 4] = [std::ptr::null_mut(); 4];
            let mut num_devices: EGLint = 0;
            let ok = unsafe { eglQueryDevicesEXT(4, devices.as_mut_ptr(), &mut num_devices) };
            if ok != EGL_FALSE && num_devices > 0 {
                log::info!("EGL_EXT_platform_device: found {} device(s)", num_devices);
                let attrs: [EGLint; 1] = [EGL_NONE];
                let display = unsafe {
                    eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[0] as *mut std::ffi::c_void, attrs.as_ptr())
                };
                if !display.is_null() {
                    log::info!("EGL display created via EGL_EXT_platform_device (device 0)");
                    return Ok(display);
                }
                log::warn!("eglGetPlatformDisplayEXT returned null for device 0, falling back");
            } else {
                log::info!("eglQueryDevicesEXT returned {} devices (ok={}), falling back", num_devices, ok);
            }
        }

        // Method 2: eglGetDisplay(EGL_DEFAULT_DISPLAY) — may select Mesa on GLVND
        let display = unsafe { eglGetDisplay(EGL_DEFAULT_DISPLAY) };
        if !display.is_null() {
            log::info!("EGL display created via eglGetDisplay(EGL_DEFAULT_DISPLAY)");
            return Ok(display);
        }

        let err = unsafe { eglGetError() };
        Err(MediaError::CaptureError(format!(
            "Failed to create EGL display: 0x{:04X}", err
        )))
    }

    /// Import a DMA-BUF fd and read back linear pixel data via GL.
    ///
    /// The EGL context must be current on the calling thread (it is made
    /// current in [`new`] and must not have been changed since).
    ///
    /// Returns `Vec<u8>` containing linear BGRA pixel data.
    pub fn import_dmabuf(
        &mut self,
        fd: RawFd,
        width: u32,
        height: u32,
        stride: u32,
        fourcc: u32,
        offset: u32,
        modifier: u64,
    ) -> Result<Vec<u8>, MediaError> {
        // Ensure EGL context is current on the calling thread.
        // The PipeWire process callback may run on a different thread from
        // where EglImporter::new() was called. EGL contexts are per-thread.
        let ok = unsafe {
            (self.eglMakeCurrent)(self.display, self.surface, self.surface, self.context)
        };
        if ok == EGL_FALSE {
            let err = unsafe { (self.eglGetError)() };
            return Err(MediaError::CaptureError(format!(
                "eglMakeCurrent failed in import_dmabuf: 0x{:04X}", err
            )));
        }

        // --- 1. Create EGLImage from DMA-BUF ---
        // Prefer eglCreateImageKHR (EGLint attrs) over eglCreateImage (EGLAttrib attrs).
        // On NVIDIA GLVND, the KHR extension is natively implemented by the driver,
        // while the EGL 1.5 core function may produce images incompatible with GL textures.
        const DRM_FORMAT_MOD_INVALID: u64 = 0x00ffffffffffffff;
        let use_modifier = modifier != 0 && modifier != DRM_FORMAT_MOD_INVALID;

        let image = if use_modifier {
            // Modifier known — create image directly
            self.create_egl_image(fd, width, height, stride, fourcc, offset, modifier, true)?
        } else {
            // Modifier unknown — try each supported modifier, validate the
            // result via glReadPixels, and cache the working modifier.
            // On NVIDIA, DMA-BUF buffers use block-linear tiling and the
            // modifier MUST match the buffer's actual layout.
            if let Some(cached) = self.cached_modifier {
                // Use cached modifier from previous successful import
                self.create_egl_image(fd, width, height, stride, fourcc, offset, cached, true)?
            } else {
                // First frame — probe all supported modifiers and pick the
                // one with the best pixel quality (lowest adjacent pixel diff).
                let gbm_mod = Self::query_dmabuf_modifier(fd, width, height, stride, fourcc);
                log::info!("query_dmabuf_modifier returned: 0x{:016X}", gbm_mod);
                let modifiers = self.query_supported_modifiers(fourcc);
                log::info!(
                    "Probing {} supported modifiers for fourcc 0x{:08X}",
                    modifiers.len(), fourcc
                );

                let mut best_image: Option<EGLImage> = None;
                let mut best_modifier: Option<u64> = None;
                let mut best_score: u64 = u64::MAX;

                for &m in &modifiers {
                    match self.create_egl_image(fd, width, height, stride, fourcc, offset, m, true) {
                        Ok(img) => {
                            let score = self.image_quality_score(img, width, height);
                            log::info!("Modifier 0x{:016X}: quality_score={}", m, score);
                            if score != u64::MAX {
                                log::info!("Selected modifier 0x{:016X} (score={})", m, score);
                                best_image = Some(img);
                                best_modifier = Some(m);
                                break;
                            } else {
                                unsafe { (self.eglDestroyImage)(self.display, img); }
                            }
                        }
                        Err(_) => {}
                    }
                }

                // If no supported modifier worked, try without modifier (Mesa/Intel/AMD)
                if best_image.is_none() {
                    if let Ok(img) = self.create_egl_image(fd, width, height, stride, fourcc, offset, 0, false) {
                        let score = self.image_quality_score(img, width, height);
                        log::debug!("No modifier: quality_score={}", score);
                        if score != u64::MAX {
                            best_score = score;
                            best_image = Some(img);
                            best_modifier = None;
                        } else {
                            unsafe { (self.eglDestroyImage)(self.display, img); }
                        }
                    }
                }

                if let Some(m) = best_modifier {
                    log::info!("Best modifier: 0x{:016X} (score={})", m, best_score);
                    self.cached_modifier = Some(m);
                } else if best_image.is_some() {
                    log::info!("Best result: no modifier (score={})", best_score);
                }

                best_image.ok_or_else(|| {
                    MediaError::CaptureError(format!(
                        "DMA-BUF import failed: no working modifier found for fourcc 0x{:08X}", fourcc
                    ))
                })?
            }
        };

        log::debug!(
            "EGLImage created OK: image={:?}, fd={}, {}x{}, stride={}, fourcc=0x{:08X}, offset={}, modifier=0x{:016X}",
            image, fd, width, height, stride, fourcc, offset, modifier
        );

        // --- 3. Create GL texture from EGLImage ---
        let mut tex: GLuint = 0;
        unsafe { (self.glGenTextures)(1, &mut tex); }
        if tex == 0 {
            unsafe { (self.eglDestroyImage)(self.display, image); }
            return Err(MediaError::CaptureError("glGenTextures failed".into()));
        }

        // --- 3 & 4. Bind EGLImage and read back linear pixel data via GPU texture sampler ---
        let data_size = (width * height * 4) as usize;
        let mut data: Vec<u8> = vec![0u8; data_size];
        let mut src_tex: GLuint = 0;
        let mut src_fbo: GLuint = 0;
        let mut dst_tex: GLuint = 0;
        let mut dst_fbo: GLuint = 0;

        unsafe {
            (self.glGenTextures)(1, &mut src_tex);
            (self.glBindTexture)(GL_TEXTURE_2D, src_tex);
            (self.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            (self.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            (self.glEGLImageTargetTexture2DOES)(GL_TEXTURE_2D, image);

            (self.glGenFramebuffers)(1, &mut src_fbo);
            (self.glBindFramebuffer)(GL_READ_FRAMEBUFFER, src_fbo);
            (self.glFramebufferTexture2D)(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src_tex, 0);

            (self.glGenTextures)(1, &mut dst_tex);
            (self.glBindTexture)(GL_TEXTURE_2D, dst_tex);
            (self.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            (self.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            (self.glTexImage2D)(
                GL_TEXTURE_2D, 0, GL_RGBA as GLint,
                width as GLsizei, height as GLsizei, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, std::ptr::null(),
            );

            (self.glGenFramebuffers)(1, &mut dst_fbo);
            (self.glBindFramebuffer)(GL_FRAMEBUFFER, dst_fbo);
            (self.glFramebufferTexture2D)(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst_tex, 0);
        }

        let rendered = self.draw_textured_quad(src_tex, dst_fbo, width, height);

        let result = if !rendered {
            Err(MediaError::CaptureError("draw_textured_quad failed".into()))
        } else {
            unsafe {
                (self.glReadPixels)(
                    0, 0,
                    width as GLsizei, height as GLsizei,
                    GL_RGBA, GL_UNSIGNED_BYTE,
                    data.as_mut_ptr() as *mut std::ffi::c_void,
                );
                // Convert RGBA -> BGRA in place
                for pixel in data.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }
            }

            let gl_err = unsafe { (self.glGetError)() };
            if gl_err != GL_NO_ERROR {
                Err(MediaError::CaptureError(format!(
                    "glReadPixels failed: GL error 0x{:04X}", gl_err
                )))
            } else {
                // Log first frame info
                {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static LOGGED: AtomicBool = AtomicBool::new(false);
                    if !LOGGED.swap(true, Ordering::Relaxed) {
                        let non_zero = data.iter().filter(|&&b| b != 0).count();
                        let row0 = &data[..data.len().min(32)];
                        let row100_off = (100 * width as usize * 4).min(data.len().saturating_sub(32));
                        let row100 = &data[row100_off..(row100_off + 32).min(data.len())];
                        log::info!(
                            "EGL DMA-BUF import: {}x{} stride={} fourcc=0x{:08X} -> RGBA bytes={} non_zero={}",
                            width, height, stride, fourcc, data.len(), non_zero,
                        );
                    }
                }
                Ok(data)
            }
        };

        // Cleanup
        unsafe {
            (self.glBindFramebuffer)(GL_FRAMEBUFFER, 0);
            (self.glDeleteFramebuffers)(1, &dst_fbo);
            (self.glDeleteTextures)(1, &dst_tex);
            (self.glDeleteFramebuffers)(1, &src_fbo);
            (self.glDeleteTextures)(1, &src_tex);
            (self.eglDestroyImage)(self.display, image);
        }

        result
    }

    /// Render a textured quad from src_tex into dst_fbo using OpenGL texture sampling.
    /// This forces GPU hardware texture sampling which de-tiles block-linear DMA-BUF memory.
    fn draw_textured_quad(&self, src_tex: GLuint, dst_fbo: GLuint, width: u32, height: u32) -> bool {
        let (enable, viewport, matrix_mode, load_identity, ortho, begin, end, tex_coord, vertex) = match (
            self.glEnable, self.glViewport, self.glMatrixMode, self.glLoadIdentity,
            self.glOrtho, self.glBegin, self.glEnd, self.glTexCoord2f, self.glVertex2f
        ) {
            (Some(e), Some(vp), Some(mm), Some(li), Some(o), Some(b), Some(end), Some(tc), Some(v)) =>
                (e, vp, mm, li, o, b, end, tc, v),
            _ => {
                log::warn!(
                    "draw_textured_quad failed: enable={}, vp={}, mm={}, li={}, ortho={}, begin={}, end={}, tc={}, v={}",
                    self.glEnable.is_some(), self.glViewport.is_some(), self.glMatrixMode.is_some(),
                    self.glLoadIdentity.is_some(), self.glOrtho.is_some(), self.glBegin.is_some(),
                    self.glEnd.is_some(), self.glTexCoord2f.is_some(), self.glVertex2f.is_some()
                );
                return false;
            }
        };

        unsafe {
            (self.glBindFramebuffer)(GL_FRAMEBUFFER, dst_fbo);
            viewport(0, 0, width as GLsizei, height as GLsizei);

            matrix_mode(GL_PROJECTION);
            load_identity();
            ortho(0.0, width as f64, 0.0, height as f64, -1.0, 1.0);

            matrix_mode(GL_MODELVIEW);
            load_identity();

            enable(GL_TEXTURE_2D);
            (self.glBindTexture)(GL_TEXTURE_2D, src_tex);

            begin(GL_QUADS);
            tex_coord(0.0, 0.0); vertex(0.0, 0.0);
            tex_coord(1.0, 0.0); vertex(width as f32, 0.0);
            tex_coord(1.0, 1.0); vertex(width as f32, height as f32);
            tex_coord(0.0, 1.0); vertex(0.0, height as f32);
            end();

            if let Some(disable) = self.glDisable {
                disable(GL_TEXTURE_2D);
            }
        }
        true
    }
    /// Score the quality of an EGLImage by rendering a textured quad and measuring
    /// block boundary discontinuities (block_spike). Returns a score where lower = better.
    /// Correct modifiers have low block spikes (< 4.0), while mis-tiled images have high spikes (> 14.0).
    fn image_quality_score(&self, image: EGLImage, width: u32, height: u32) -> u64 {
        let mut src_tex: GLuint = 0;
        let mut src_fbo: GLuint = 0;
        let mut dst_tex: GLuint = 0;
        let mut dst_fbo: GLuint = 0;

        // Clear prior errors
        while unsafe { (self.glGetError)() } != GL_NO_ERROR {}

        unsafe {
            (self.glGenTextures)(1, &mut src_tex);
            (self.glBindTexture)(GL_TEXTURE_2D, src_tex);
            (self.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            (self.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            (self.glEGLImageTargetTexture2DOES)(GL_TEXTURE_2D, image);

            (self.glGenFramebuffers)(1, &mut src_fbo);
            (self.glBindFramebuffer)(GL_READ_FRAMEBUFFER, src_fbo);
            (self.glFramebufferTexture2D)(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src_tex, 0);

            (self.glGenTextures)(1, &mut dst_tex);
            (self.glBindTexture)(GL_TEXTURE_2D, dst_tex);
            (self.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            (self.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            (self.glTexImage2D)(
                GL_TEXTURE_2D, 0, GL_RGBA as GLint,
                width as GLsizei, height as GLsizei, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, std::ptr::null(),
            );

            (self.glGenFramebuffers)(1, &mut dst_fbo);
            (self.glBindFramebuffer)(GL_FRAMEBUFFER, dst_fbo);
            (self.glFramebufferTexture2D)(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst_tex, 0);
        }

        let rendered = self.draw_textured_quad(src_tex, dst_fbo, width, height);
        if !rendered {
            unsafe {
                (self.glDeleteFramebuffers)(1, &dst_fbo);
                (self.glDeleteTextures)(1, &dst_tex);
                (self.glDeleteFramebuffers)(1, &src_fbo);
                (self.glDeleteTextures)(1, &src_tex);
            }
            return u64::MAX;
        }

        let sample_rows = 64u32.min(height);
        let sample_y = (height / 2).saturating_sub(sample_rows / 2);
        let sample_size = (width * sample_rows * 4) as usize;
        let mut pixels = vec![0u8; sample_size];

        unsafe {
            (self.glReadPixels)(
                0, sample_y as GLint,
                width as GLsizei, sample_rows as GLsizei,
                GL_RGBA, GL_UNSIGNED_BYTE,
                pixels.as_mut_ptr() as *mut std::ffi::c_void,
            );
            (self.glBindFramebuffer)(GL_FRAMEBUFFER, 0);
            (self.glDeleteFramebuffers)(1, &dst_fbo);
            (self.glDeleteTextures)(1, &dst_tex);
            (self.glDeleteFramebuffers)(1, &src_fbo);
            (self.glDeleteTextures)(1, &src_tex);
        }

        // Ignore blank (all zero) images
        let non_zero = pixels.iter().filter(|&&b| b != 0).count();
        if non_zero < 100 {
            return u64::MAX;
        }

        // Measure block boundary diff at 16-row steps (block boundaries)
        let row_stride = (width * 4) as usize;
        let block_step = 16usize;
        let mut block_diff_sum: u64 = 0;
        let mut block_count: u64 = 0;

        let sample_rows_usize = sample_rows as usize;
        for r in (block_step - 1..sample_rows_usize - 1).step_by(block_step) {
            let r0_start = r * row_stride;
            let r1_start = (r + 1) * row_stride;
            for x in 0..width as usize {
                let idx0 = r0_start + x * 4;
                let idx1 = r1_start + x * 4;
                if idx0 + 3 < pixels.len() && idx1 + 3 < pixels.len() {
                    for c in 0..3 {
                        let diff = (pixels[idx0 + c] as i32 - pixels[idx1 + c] as i32).unsigned_abs();
                        block_diff_sum += diff as u64;
                    }
                    block_count += 1;
                }
            }
        }

        if block_count > 0 {
            let avg = block_diff_sum / block_count;
            if avg == 0 {
                999999 // Penalty for zero variance / blank image
            } else {
                avg
            }
        } else {
            u64::MAX
        }
    }

    /// Query the actual DRM format modifier of a DMA-BUF fd by importing
    /// it via GBM. Returns 0 if GBM is unavailable or import fails.
    fn query_dmabuf_modifier(fd: RawFd, width: u32, height: u32, stride: u32, fourcc: u32) -> u64 {
        // GBM function types
        type GbmCreateDeviceFn = unsafe extern "C" fn(fd: libc::c_int) -> *mut std::ffi::c_void;
        type GbmBoImportFn = unsafe extern "C" fn(
            gbm: *mut std::ffi::c_void, type_: u32,
            buffer: *mut std::ffi::c_void, flags: u32,
        ) -> *mut std::ffi::c_void;
        type GbmBoGetModifierFn = unsafe extern "C" fn(bo: *mut std::ffi::c_void) -> u64;
        type GbmBoDestroyFn = unsafe extern "C" fn(bo: *mut std::ffi::c_void);
        type GbmDeviceDestroyFn = unsafe extern "C" fn(gbm: *mut std::ffi::c_void);

        // GBM_BO_IMPORT_FD_MODIFIER = 2 (import with fd + modifier)
        // GBM_BO_IMPORT_FD = 1 (import with fd, no modifier)
        const GBM_BO_IMPORT_FD: u32 = 1;

        #[repr(C)]
        struct GbmImportFdData {
            fd: libc::c_int,
            width: u32,
            height: u32,
            stride: u32,
            format: u32,
        }

        let gbmlib = match unsafe { libloading::Library::new("libgbm.so.1") } {
            Ok(lib) => lib,
            Err(_) => return 0,
        };

        let gbm_create_device: GbmCreateDeviceFn = match unsafe { gbmlib.get(b"gbm_create_device") } {
            Ok(f) => *f,
            Err(_) => return 0,
        };
        let gbm_bo_import: GbmBoImportFn = match unsafe { gbmlib.get(b"gbm_bo_import") } {
            Ok(f) => *f,
            Err(_) => return 0,
        };
        let gbm_bo_get_modifier: GbmBoGetModifierFn = match unsafe { gbmlib.get(b"gbm_bo_get_modifier") } {
            Ok(f) => *f,
            Err(_) => return 0,
        };
        let gbm_bo_destroy: GbmBoDestroyFn = match unsafe { gbmlib.get(b"gbm_bo_destroy") } {
            Ok(f) => *f,
            Err(_) => return 0,
        };
        let gbm_device_destroy: GbmDeviceDestroyFn = match unsafe { gbmlib.get(b"gbm_device_destroy") } {
            Ok(f) => *f,
            Err(_) => return 0,
        };

        // Open render node
        let render_path = std::ffi::CString::new("/dev/dri/renderD128").unwrap();
        let drm_fd = unsafe { libc::open(render_path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if drm_fd < 0 {
            return 0;
        }

        let gbm_dev = unsafe { gbm_create_device(drm_fd) };
        if gbm_dev.is_null() {
            unsafe { libc::close(drm_fd); }
            return 0;
        }

        // Dup the fd because gbm_bo_import takes ownership
        let dup_fd = unsafe { libc::dup(fd) };
        if dup_fd < 0 {
            unsafe { gbm_device_destroy(gbm_dev); libc::close(drm_fd); }
            return 0;
        }

        let mut import_data = GbmImportFdData {
            fd: dup_fd,
            width,
            height,
            stride,
            format: fourcc,
        };

        let bo = unsafe {
            gbm_bo_import(
                gbm_dev, GBM_BO_IMPORT_FD,
                &mut import_data as *mut GbmImportFdData as *mut std::ffi::c_void,
                0, // usage flags
            )
        };

        let modifier = if !bo.is_null() {
            let m = unsafe { gbm_bo_get_modifier(bo) };
            unsafe { gbm_bo_destroy(bo); }
            m
        } else {
            log::debug!("gbm_bo_import failed for fd={}", fd);
            0
        };

        unsafe {
            gbm_device_destroy(gbm_dev);
            libc::close(drm_fd);
        }

        modifier
    }

    /// Query supported DRM format modifiers for a given fourcc via
    /// `eglQueryDmaBufModifiersEXT`. Returns an empty vec if the
    /// extension is unavailable or the query fails.
    fn query_supported_modifiers(&self, fourcc: u32) -> Vec<u64> {
        type QueryModifiersFn = unsafe extern "C" fn(
            dpy: EGLDisplay, format: EGLint, max_modifiers: EGLint,
            modifiers: *mut u64, external_only: *mut EGLBoolean,
            num_modifiers: *mut EGLint,
        ) -> EGLBoolean;

        let ptr = unsafe {
            (self.eglGetProcAddress)(b"eglQueryDmaBufModifiersEXT\0".as_ptr() as *const libc::c_char)
        };
        if ptr.is_null() {
            log::debug!("eglQueryDmaBufModifiersEXT not available");
            return vec![];
        }
        let query_fn: QueryModifiersFn = unsafe { std::mem::transmute(ptr) };

        // First query: how many modifiers?
        let mut num_modifiers: EGLint = 0;
        let ok = unsafe {
            query_fn(self.display, fourcc as EGLint, 0, std::ptr::null_mut(), std::ptr::null_mut(), &mut num_modifiers)
        };
        if ok != EGL_TRUE || num_modifiers <= 0 {
            log::debug!("eglQueryDmaBufModifiersEXT: no modifiers for fourcc 0x{:08X}", fourcc);
            return vec![];
        }

        // Second query: get the actual modifiers
        let mut modifiers = vec![0u64; num_modifiers as usize];
        let mut external_only = vec![0u32; num_modifiers as usize];
        let ok = unsafe {
            query_fn(
                self.display, fourcc as EGLint, num_modifiers,
                modifiers.as_mut_ptr(), external_only.as_mut_ptr(), &mut num_modifiers,
            )
        };
        if ok != EGL_TRUE {
            log::debug!("eglQueryDmaBufModifiersEXT second query failed");
            return vec![];
        }
        modifiers.truncate(num_modifiers as usize);

        // Filter out external-only modifiers (they can't be used with GL_TEXTURE_2D)
        let mut filtered: Vec<u64> = modifiers.iter().zip(external_only.iter())
            .filter(|(_, &ext)| ext == 0u32)
            .map(|(&m, _)| m)
            .collect();

        // Sort candidate modifiers by preference:
        // On NVIDIA, memory kind 0xE080 with block height 4 (16 GOBs = 0x...E08014)
        // is the standard layout for 1080p desktop surfaces on Turing/Ampere/Ada GPUs.
        filtered.sort_by_key(|&m| {
            let vendor = (m >> 56) & 0xFF;
            if vendor == 3 {
                let kind = (m >> 8) & 0xFFFF; // 0xE080 vs 0x6060
                let block_height = m & 0xF;
                let kind_score = if kind == 0xE080 { 0 } else { 1 };
                let height_score = match block_height {
                    4 => 0, // 16 GOBs (0x...14) -> top priority for 1080p
                    3 => 1, // 8 GOBs (0x...13)
                    2 => 2, // 4 GOBs
                    1 => 3, // 2 GOBs
                    0 => 4, // 1 GOB
                    _ => 5, // 32 GOBs (0x...15)
                };
                kind_score * 10 + height_score
            } else {
                100
            }
        });

        log::info!(
            "Supported modifiers for fourcc 0x{:08X}: {} total, {} non-external: {:016X?}",
            fourcc, modifiers.len(), filtered.len(), &filtered
        );

        filtered
    }

    /// Create an EGLImage from a DMA-BUF, trying eglCreateImageKHR first.
    ///
    /// eglCreateImageKHR (EGLint attrs, 32-bit) is the KHR extension most
    /// drivers natively implement. The EGL 1.5 core eglCreateImage (EGLAttrib
    /// attrs, pointer-size) may produce images incompatible with GL textures
    /// on some NVIDIA GLVND setups.
    fn create_egl_image(
        &self,
        fd: RawFd,
        width: u32,
        height: u32,
        stride: u32,
        fourcc: u32,
        offset: u32,
        modifier: u64,
        use_modifier: bool,
    ) -> Result<EGLImage, MediaError> {
        // Check fd validity
        let fd_valid = unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0;
        log::debug!(
            "create_egl_image: fd={} valid={}, fourcc=0x{:08X}, modifier=0x{:016X}, use_modifier={}",
            fd, fd_valid, fourcc, modifier, use_modifier
        );

        // Clear EGL errors
        unsafe { (self.eglGetError)(); }

        // Try eglCreateImageKHR first (EGLint attributes — correct for NVIDIA)
        if let Some(create_khr) = self.eglCreateImageKHR {
            let image = if use_modifier {
                let modifier_lo = (modifier & 0xFFFFFFFF) as EGLint;
                let modifier_hi = ((modifier >> 32) & 0xFFFFFFFF) as EGLint;
                let attrs: [EGLint; 17] = [
                    EGL_WIDTH as EGLint, width as EGLint,
                    EGL_HEIGHT as EGLint, height as EGLint,
                    EGL_LINUX_DRM_FOURCC_EXT, fourcc as EGLint,
                    EGL_DMA_BUF_PLANE0_FD_EXT, fd as EGLint,
                    EGL_DMA_BUF_PLANE0_OFFSET_EXT, offset as EGLint,
                    EGL_DMA_BUF_PLANE0_PITCH_EXT, stride as EGLint,
                    EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, modifier_lo,
                    EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, modifier_hi,
                    EGL_NONE,
                ];
                let img = unsafe {
                    create_khr(self.display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT,
                        std::ptr::null_mut(), attrs.as_ptr())
                };
                img
            } else {
                let attrs: [EGLint; 13] = [
                    EGL_WIDTH as EGLint, width as EGLint,
                    EGL_HEIGHT as EGLint, height as EGLint,
                    EGL_LINUX_DRM_FOURCC_EXT, fourcc as EGLint,
                    EGL_DMA_BUF_PLANE0_FD_EXT, fd as EGLint,
                    EGL_DMA_BUF_PLANE0_OFFSET_EXT, offset as EGLint,
                    EGL_DMA_BUF_PLANE0_PITCH_EXT, stride as EGLint,
                    EGL_NONE,
                ];
                unsafe {
                    create_khr(self.display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT,
                        std::ptr::null_mut(), attrs.as_ptr())
                }
            };
            if image != EGL_NO_IMAGE {
                return Ok(image);
            }
            let err = unsafe { (self.eglGetError)() };
            log::warn!("eglCreateImageKHR failed: 0x{:04X}, trying eglCreateImage", err);
        }

        // Fallback: eglCreateImage (EGL 1.5 core, EGLAttrib attributes)
        let attrs: [EGLAttrib; 13] = [
            EGL_WIDTH as EGLAttrib, width as EGLAttrib,
            EGL_HEIGHT as EGLAttrib, height as EGLAttrib,
            EGL_LINUX_DRM_FOURCC_EXT as EGLAttrib, fourcc as EGLAttrib,
            EGL_DMA_BUF_PLANE0_FD_EXT as EGLAttrib, fd as EGLAttrib,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT as EGLAttrib, offset as EGLAttrib,
            EGL_DMA_BUF_PLANE0_PITCH_EXT as EGLAttrib, stride as EGLAttrib,
            EGL_NONE as EGLAttrib,
        ];
        let image = unsafe {
            (self.eglCreateImage)(
                self.display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(), attrs.as_ptr(),
            )
        };
        if image != EGL_NO_IMAGE {
            return Ok(image);
        }
        let err = unsafe { (self.eglGetError)() };
        Err(MediaError::CaptureError(format!(
            "eglCreateImage(DMA-BUF) failed: 0x{:04X} (fd={}, {}x{}, stride={}, fourcc=0x{:08X}, modifier=0x{:016X})",
            err, fd, width, height, stride, fourcc, modifier
        )))
    }
}

impl Drop for EglImporter {
    fn drop(&mut self) {
        log::info!("EGL importer: destroying context and display");
        unsafe {
            (self.eglMakeCurrent)(self.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            (self.eglDestroyContext)(self.display, self.context);
            (self.eglTerminate)(self.display);
        }
        // Libraries are dropped automatically
    }
}

// SAFETY: EglImporter holds raw EGL/GL state (display, context, function pointers).
// The EGL context must be used from a single thread. The caller is responsible for
// ensuring import_dmabuf() is called from the thread that created the importer.
unsafe impl Send for EglImporter {}
