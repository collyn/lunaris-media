//! EGL DMA-BUF Importer
//!
//! On NVIDIA Wayland, PipeWire delivers DMA-BUF file descriptors with block-linear
//! tiled memory (modifier like `0x0500000008`). CPU mmap cannot read tiled memory correctly,
//! which produces noise or garbled data.
//!
//! This module uses EGL to import the DMA-BUF through the GPU memory controller
//! (which handles tiling transparently) and reads back linear pixels via `glReadPixels`.
//!
//! Architecture:
//! DMA-BUF fd -> eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT) -> GL texture -> FBO -> glReadPixels -> Vec<u8> BGRA

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(unused)]

use std::ffi::{c_void, CStr};
use std::ptr;
use libloading::Library;
use crate::error::MediaError;

// Constants
type EGLBoolean = libc::c_uint;
type EGLint = libc::c_int;
type EGLAttrib = libc::intptr_t;
type EGLDisplay = *mut c_void;
type EGLConfig = *mut c_void;
type EGLContext = *mut c_void;
type EGLSurface = *mut c_void;
type EGLImageKHR = *mut c_void;
type EGLClientBuffer = *mut c_void;
type EGLenum = libc::c_uint;
type GLenum = libc::c_uint;
type GLint = libc::c_int;
type GLuint = libc::c_uint;
type GLsizei = libc::c_int;

const EGL_TRUE: EGLBoolean = 1;
const EGL_FALSE: EGLBoolean = 0;
const EGL_NO_CONTEXT: EGLContext = ptr::null_mut();
const EGL_NO_DISPLAY: EGLDisplay = ptr::null_mut();
const EGL_NO_SURFACE: EGLSurface = ptr::null_mut();
const EGL_NO_IMAGE_KHR: EGLImageKHR = ptr::null_mut();
const EGL_DEFAULT_DISPLAY: EGLDisplay = 0 as *mut c_void;
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
const EGL_WIDTH: EGLint = 0x3057;
const EGL_HEIGHT: EGLint = 0x3056;
const EGL_OPENGL_API: EGLenum = 0x30A2;
const EGL_CONTEXT_MAJOR_VERSION: EGLint = 0x3098;
const EGL_CONTEXT_MINOR_VERSION: EGLint = 0x30FB;
const EGL_CONTEXT_OPENGL_PROFILE_MASK: EGLint = 0x30FD;
const EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT: EGLint = 0x00000002;
const EGL_LINUX_DMA_BUF_EXT: EGLenum = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: EGLint = 0x3271;
const EGL_DMA_BUF_PLANE0_FD_EXT: EGLint = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EGLint = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: EGLint = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EGLint = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EGLint = 0x3444;
const EGL_PLATFORM_DEVICE_EXT: EGLenum = 0x313F;
const EGL_PLATFORM_GBM_KHR: EGLenum = 0x31D7;

const GL_TEXTURE_2D: GLenum = 0x0DE1;
const GL_BGRA: GLenum = 0x80E1;
const GL_UNSIGNED_BYTE: GLenum = 0x1401;
const GL_FRAMEBUFFER: GLenum = 0x8D40;
const GL_COLOR_ATTACHMENT0: GLenum = 0x8CE0;
const GL_FRAMEBUFFER_COMPLETE: GLenum = 0x8CD5;
const GL_TEXTURE_MIN_FILTER: GLenum = 0x2801;
const GL_TEXTURE_MAG_FILTER: GLenum = 0x2800;
const GL_NEAREST: GLint = 0x2600;

// Function pointer types
type EglGetErrorFn = unsafe extern "C" fn() -> EGLint;
type EglInitializeFn = unsafe extern "C" fn(EGLDisplay, *mut EGLint, *mut EGLint) -> EGLBoolean;
type EglBindAPIFn = unsafe extern "C" fn(EGLenum) -> EGLBoolean;
type EglChooseConfigFn = unsafe extern "C" fn(EGLDisplay, *const EGLint, *mut EGLConfig, EGLint, *mut EGLint) -> EGLBoolean;
type EglCreatePbufferSurfaceFn = unsafe extern "C" fn(EGLDisplay, EGLConfig, *const EGLint) -> EGLSurface;
type EglCreateContextFn = unsafe extern "C" fn(EGLDisplay, EGLConfig, EGLContext, *const EGLint) -> EGLContext;
type EglMakeCurrentFn = unsafe extern "C" fn(EGLDisplay, EGLSurface, EGLSurface, EGLContext) -> EGLBoolean;
type EglGetProcAddressFn = unsafe extern "C" fn(*const libc::c_char) -> *mut c_void;
type EglDestroyContextFn = unsafe extern "C" fn(EGLDisplay, EGLContext) -> EGLBoolean;
type EglDestroySurfaceFn = unsafe extern "C" fn(EGLDisplay, EGLSurface) -> EGLBoolean;
type EglTerminateFn = unsafe extern "C" fn(EGLDisplay) -> EGLBoolean;
type EglQueryStringFn = unsafe extern "C" fn(EGLDisplay, EGLint) -> *const libc::c_char;
type EglQueryDevicesEXTFn = unsafe extern "C" fn(EGLint, *mut *mut c_void, *mut EGLint) -> EGLBoolean;
type EglGetPlatformDisplayEXTFn = unsafe extern "C" fn(EGLenum, *mut c_void, *const EGLint) -> EGLDisplay;
type EglGetDisplayFn = unsafe extern "C" fn(*mut c_void) -> EGLDisplay;
type EglCreateImageKHRFn = unsafe extern "C" fn(EGLDisplay, EGLContext, EGLenum, EGLClientBuffer, *const EGLint) -> EGLImageKHR;
type EglDestroyImageKHRFn = unsafe extern "C" fn(EGLDisplay, EGLImageKHR) -> EGLBoolean;

type GlGetStringFn = unsafe extern "C" fn(GLenum) -> *const libc::c_char;
type GlGenTexturesFn = unsafe extern "C" fn(GLsizei, *mut GLuint);
type GlBindTextureFn = unsafe extern "C" fn(GLenum, GLuint);
type GlTexParameteriFn = unsafe extern "C" fn(GLenum, GLenum, GLint);
type GlGenFramebuffersFn = unsafe extern "C" fn(GLsizei, *mut GLuint);
type GlBindFramebufferFn = unsafe extern "C" fn(GLenum, GLuint);
type GlFramebufferTexture2DFn = unsafe extern "C" fn(GLenum, GLenum, GLenum, GLuint, GLint);
type GlCheckFramebufferStatusFn = unsafe extern "C" fn(GLenum) -> GLenum;
type GlReadPixelsFn = unsafe extern "C" fn(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, *mut c_void);
type GlDeleteTexturesFn = unsafe extern "C" fn(GLsizei, *const GLuint);
type GlDeleteFramebuffersFn = unsafe extern "C" fn(GLsizei, *const GLuint);
type GlEGLImageTargetTexture2DOESFn = unsafe extern "C" fn(GLenum, EGLImageKHR);

macro_rules! load_sym {
    ($lib:expr, $name:literal) => {
        unsafe {
            $lib.get::<*const u8>($name.as_bytes())
                .map_err(|e| MediaError::CaptureError(format!("Failed to load {}: {}", $name, e)))?
                .into_raw()
                .cast::<std::ffi::c_void>()
        }
    };
}

pub struct EglImporter {
    egllib: Library,
    gleslib: Library,
    _gbmlib: Option<Library>,
    _gbm_dev: *mut std::ffi::c_void,
    _drm_fd: libc::c_int,
    display: EGLDisplay,
    context: EGLContext,
    surface: EGLSurface,
    pub cached_modifier: Option<u64>,

    // EGL Fns
    eglGetError: EglGetErrorFn,
    eglQueryString: EglQueryStringFn,
    eglBindAPI: EglBindAPIFn,
    eglChooseConfig: EglChooseConfigFn,
    eglCreatePbufferSurface: EglCreatePbufferSurfaceFn,
    eglCreateContext: EglCreateContextFn,
    eglMakeCurrent: EglMakeCurrentFn,
    eglGetProcAddress: EglGetProcAddressFn,
    eglDestroyContext: EglDestroyContextFn,
    eglDestroySurface: EglDestroySurfaceFn,
    eglTerminate: EglTerminateFn,
    eglCreateImageKHR: EglCreateImageKHRFn,
    eglDestroyImageKHR: EglDestroyImageKHRFn,

    // GL Fns
    glGenTextures: GlGenTexturesFn,
    glBindTexture: GlBindTextureFn,
    glTexParameteri: GlTexParameteriFn,
    glGenFramebuffers: GlGenFramebuffersFn,
    glBindFramebuffer: GlBindFramebufferFn,
    glFramebufferTexture2D: GlFramebufferTexture2DFn,
    glCheckFramebufferStatus: GlCheckFramebufferStatusFn,
    glReadPixels: GlReadPixelsFn,
    glDeleteTextures: GlDeleteTexturesFn,
    glDeleteFramebuffers: GlDeleteFramebuffersFn,
    glEGLImageTargetTexture2DOES: GlEGLImageTargetTexture2DOESFn,
}

impl EglImporter {
    pub fn new() -> Result<Self, MediaError> {
        let egllib = unsafe { Library::new("libEGL.so.1").map_err(|e| MediaError::CaptureError(format!("EGL missing: {}", e)))? };
        let gleslib = unsafe { Library::new("libGL.so.1").map_err(|e| MediaError::CaptureError(format!("GL missing: {}", e)))? };

        unsafe {
            let eglGetProcAddress_ptr = load_sym!(egllib, "eglGetProcAddress\0");
            let eglGetProcAddress: EglGetProcAddressFn = std::mem::transmute(eglGetProcAddress_ptr);

            let eglGetDisplay_ptr = load_sym!(egllib, "eglGetDisplay\0");
            let eglGetDisplay: EglGetDisplayFn = std::mem::transmute(eglGetDisplay_ptr);

            let (display, gbmlib, gbm_dev, drm_fd) = Self::create_display(&egllib, eglGetProcAddress, eglGetDisplay)?;

            let eglInitialize: EglInitializeFn = std::mem::transmute(load_sym!(egllib, "eglInitialize\0"));
            let mut major = 0;
            let mut minor = 0;
            if eglInitialize(display, &mut major, &mut minor) != EGL_TRUE as u32 {
                let eglGetError: EglGetErrorFn = std::mem::transmute(load_sym!(egllib, "eglGetError\0"));
                return Err(MediaError::CaptureError(format!("eglInitialize failed: 0x{:04X}", eglGetError())));
            }

            let eglBindAPI: EglBindAPIFn = std::mem::transmute(load_sym!(egllib, "eglBindAPI\0"));
            if eglBindAPI(EGL_OPENGL_API) != EGL_TRUE as u32 {
                return Err(MediaError::CaptureError("eglBindAPI(EGL_OPENGL_API) failed".into()));
            }

            let eglChooseConfig: EglChooseConfigFn = std::mem::transmute(load_sym!(egllib, "eglChooseConfig\0"));
            let config_attribs = [
                EGL_RED_SIZE, 8,
                EGL_GREEN_SIZE, 8,
                EGL_BLUE_SIZE, 8,
                EGL_ALPHA_SIZE, 8,
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_NONE,
            ];
            let mut config: EGLConfig = ptr::null_mut();
            let mut num_config = 0;
            if eglChooseConfig(display, config_attribs.as_ptr(), &mut config, 1, &mut num_config) != EGL_TRUE as u32 || num_config == 0 {
                return Err(MediaError::CaptureError("eglChooseConfig failed".into()));
            }

            let eglCreatePbufferSurface: EglCreatePbufferSurfaceFn = std::mem::transmute(load_sym!(egllib, "eglCreatePbufferSurface\0"));
            let pbuffer_attribs = [
                EGL_WIDTH, 1,
                EGL_HEIGHT, 1,
                EGL_NONE,
            ];
            let surface = eglCreatePbufferSurface(display, config, pbuffer_attribs.as_ptr());
            if surface == EGL_NO_SURFACE {
                return Err(MediaError::CaptureError("eglCreatePbufferSurface failed".into()));
            }

            let eglCreateContext: EglCreateContextFn = std::mem::transmute(load_sym!(egllib, "eglCreateContext\0"));
            let context_attribs = [
                EGL_CONTEXT_MAJOR_VERSION, 2,
                EGL_CONTEXT_MINOR_VERSION, 0,
                EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
                EGL_NONE,
            ];
            let context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs.as_ptr());
            if context == EGL_NO_CONTEXT {
                return Err(MediaError::CaptureError("eglCreateContext failed".into()));
            }

            let eglMakeCurrent: EglMakeCurrentFn = std::mem::transmute(load_sym!(egllib, "eglMakeCurrent\0"));
            if eglMakeCurrent(display, surface, surface, context) != EGL_TRUE as u32 {
                return Err(MediaError::CaptureError("eglMakeCurrent failed".into()));
            }

            let create_img_ptr = eglGetProcAddress(b"eglCreateImageKHR\0".as_ptr() as *const _);
            if create_img_ptr.is_null() {
                return Err(MediaError::CaptureError("eglCreateImageKHR not found".into()));
            }
            let eglCreateImageKHR: EglCreateImageKHRFn = std::mem::transmute(create_img_ptr);

            let eglDestroyImageKHR_ptr = eglGetProcAddress(b"eglDestroyImageKHR\0".as_ptr() as *const _);
            let eglDestroyImageKHR: EglDestroyImageKHRFn = std::mem::transmute(eglDestroyImageKHR_ptr);

            let target_tex_ptr = eglGetProcAddress(b"glEGLImageTargetTexture2DOES\0".as_ptr() as *const _);
            if target_tex_ptr.is_null() {
                return Err(MediaError::CaptureError("glEGLImageTargetTexture2DOES not found".into()));
            }
            let glEGLImageTargetTexture2DOES: GlEGLImageTargetTexture2DOESFn = std::mem::transmute(target_tex_ptr);

            let glGetString: GlGetStringFn = std::mem::transmute(load_sym!(gleslib, "glGetString\0"));
            let eglQueryString: EglQueryStringFn = std::mem::transmute(load_sym!(egllib, "eglQueryString\0"));
            let vendor = eglQueryString(display, 0x3053); // EGL_VENDOR
            if !vendor.is_null() {
                if let Ok(s) = CStr::from_ptr(vendor).to_str() {
                    log::info!("EGL initialized: vendor=\"{}\" version={}.{}", s, major, minor);
                }
            }
            let renderer = glGetString(0x1F01); // GL_RENDERER
            if !renderer.is_null() {
                if let Ok(s) = CStr::from_ptr(renderer).to_str() {
                    log::info!("EGL GL renderer: {}", s);
                }
            }

            Ok(EglImporter {
                eglGetError: std::mem::transmute(load_sym!(egllib, "eglGetError\0")),
                eglQueryString,
                eglBindAPI,
                eglChooseConfig,
                eglCreatePbufferSurface,
                eglCreateContext,
                eglMakeCurrent,
                eglGetProcAddress,
                eglDestroyContext: std::mem::transmute(load_sym!(egllib, "eglDestroyContext\0")),
                eglDestroySurface: std::mem::transmute(load_sym!(egllib, "eglDestroySurface\0")),
                eglTerminate: std::mem::transmute(load_sym!(egllib, "eglTerminate\0")),
                eglCreateImageKHR,
                eglDestroyImageKHR,
                glGenTextures: std::mem::transmute(load_sym!(gleslib, "glGenTextures\0")),
                glBindTexture: std::mem::transmute(load_sym!(gleslib, "glBindTexture\0")),
                glTexParameteri: std::mem::transmute(load_sym!(gleslib, "glTexParameteri\0")),
                glGenFramebuffers: std::mem::transmute(load_sym!(gleslib, "glGenFramebuffers\0")),
                glBindFramebuffer: std::mem::transmute(load_sym!(gleslib, "glBindFramebuffer\0")),
                glFramebufferTexture2D: std::mem::transmute(load_sym!(gleslib, "glFramebufferTexture2D\0")),
                glCheckFramebufferStatus: std::mem::transmute(load_sym!(gleslib, "glCheckFramebufferStatus\0")),
                glReadPixels: std::mem::transmute(load_sym!(gleslib, "glReadPixels\0")),
                glDeleteTextures: std::mem::transmute(load_sym!(gleslib, "glDeleteTextures\0")),
                glDeleteFramebuffers: std::mem::transmute(load_sym!(gleslib, "glDeleteFramebuffers\0")),
                glEGLImageTargetTexture2DOES,
                egllib,
                gleslib,
                _gbmlib: gbmlib,
                _gbm_dev: gbm_dev,
                _drm_fd: drm_fd,
                display,
                context,
                surface,
                cached_modifier: None,
            })
        }
    }

    unsafe fn create_display(
        _egllib: &Library,
        eglGetProcAddress: EglGetProcAddressFn,
        eglGetDisplay: EglGetDisplayFn,
    ) -> Result<(EGLDisplay, Option<Library>, *mut c_void, libc::c_int), MediaError> {
        // Try Method 0: GBM platform
        if let Ok(gbmlib) = Library::new("libgbm.so.1") {
            if let Ok(sym) = gbmlib.get::<unsafe extern "C" fn(libc::c_int) -> *mut c_void>(b"gbm_create_device\0") {
                let drm_fd = libc::open(b"/dev/dri/renderD128\0".as_ptr() as *const _, libc::O_RDWR | libc::O_CLOEXEC);
                if drm_fd >= 0 {
                    let gbm_dev = sym(drm_fd);
                    if !gbm_dev.is_null() {
                        let eglGetPlatformDisplayEXT_ptr = eglGetProcAddress(b"eglGetPlatformDisplayEXT\0".as_ptr() as *const _);
                        if !eglGetPlatformDisplayEXT_ptr.is_null() {
                            let eglGetPlatformDisplayEXT: EglGetPlatformDisplayEXTFn = std::mem::transmute(eglGetPlatformDisplayEXT_ptr);
                            let display = eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_KHR, gbm_dev, ptr::null());
                            if !display.is_null() && display != EGL_NO_DISPLAY {
                                log::info!("EGL display created via GBM platform (/dev/dri/renderD128)");
                                return Ok((display, Some(gbmlib), gbm_dev, drm_fd));
                            }
                        }
                    }
                    libc::close(drm_fd);
                }
            }
        }

        // Try Method 1: EGL device platform
        let eglQueryDevicesEXT_ptr = eglGetProcAddress(b"eglQueryDevicesEXT\0".as_ptr() as *const _);
        let eglGetPlatformDisplayEXT_ptr = eglGetProcAddress(b"eglGetPlatformDisplayEXT\0".as_ptr() as *const _);

        if !eglQueryDevicesEXT_ptr.is_null() && !eglGetPlatformDisplayEXT_ptr.is_null() {
            let eglQueryDevicesEXT: EglQueryDevicesEXTFn = std::mem::transmute(eglQueryDevicesEXT_ptr);
            let eglGetPlatformDisplayEXT: EglGetPlatformDisplayEXTFn = std::mem::transmute(eglGetPlatformDisplayEXT_ptr);

            let mut num_devices = 0;
            if eglQueryDevicesEXT(1, ptr::null_mut(), &mut num_devices) == EGL_TRUE as u32 && num_devices > 0 {
                let mut device = ptr::null_mut();
                if eglQueryDevicesEXT(1, &mut device, &mut num_devices) == EGL_TRUE as u32 {
                    let display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device, ptr::null());
                    if display != EGL_NO_DISPLAY {
                        return Ok((display, None, ptr::null_mut(), -1));
                    }
                }
            }
        }

        // Try Method 2: Default display
        let display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if display != EGL_NO_DISPLAY {
            return Ok((display, None, ptr::null_mut(), -1));
        }

        Err(MediaError::CaptureError("Failed to create EGL display".into()))
    }

    pub fn import_dmabuf(
        &mut self,
        fd: i32,
        width: u32,
        height: u32,
        stride: u32,
        fourcc: u32,
        offset: u32,
        modifier: u64,
    ) -> Result<Vec<u8>, MediaError> {
        unsafe {
            if (self.eglMakeCurrent)(self.display, self.surface, self.surface, self.context) != EGL_TRUE as u32 {
                return Err(MediaError::CaptureError("eglMakeCurrent failed during import".into()));
            }

            let w = width as EGLint;
            let h = height as EGLint;

            // DRM_FORMAT_XRGB8888 as fallback (no alpha, common on GNOME Wayland)
            const DRM_FORMAT_XRGB8888: u32 = 0x34325258;

            // Build candidate attribute lists. On NVIDIA Wayland, the DMA-BUF
            // uses block-linear tiling but PipeWire may not report the modifier
            // correctly (modifier=0 means LINEAR, which is wrong). Strategy:
            //
            // 1. Try WITHOUT modifier attrs — EGL auto-detects the layout.
            //    This works on NVIDIA 535+ with EGL_EXT_image_dma_buf_import.
            // 2. Try WITH explicit modifier (if non-zero and explicitly found).
            // 3. Try alternate fourcc (XRGB8888 instead of ARGB8888) without modifier.
            let mut attempts: Vec<(Vec<EGLint>, &str)> = Vec::new();

            // Attempt 1: no modifier, original fourcc
            attempts.push((vec![
                EGL_WIDTH, w,
                EGL_HEIGHT, h,
                EGL_LINUX_DRM_FOURCC_EXT, fourcc as EGLint,
                EGL_DMA_BUF_PLANE0_FD_EXT, fd as EGLint,
                EGL_DMA_BUF_PLANE0_OFFSET_EXT, offset as EGLint,
                EGL_DMA_BUF_PLANE0_PITCH_EXT, stride as EGLint,
                EGL_NONE,
            ], "no-modifier"));

            // Attempt 2: with explicit modifier (if non-zero)
            if modifier != 0 {
                attempts.push((vec![
                    EGL_WIDTH, w,
                    EGL_HEIGHT, h,
                    EGL_LINUX_DRM_FOURCC_EXT, fourcc as EGLint,
                    EGL_DMA_BUF_PLANE0_FD_EXT, fd as EGLint,
                    EGL_DMA_BUF_PLANE0_OFFSET_EXT, offset as EGLint,
                    EGL_DMA_BUF_PLANE0_PITCH_EXT, stride as EGLint,
                    EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (modifier & 0xFFFFFFFF) as EGLint,
                    EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (modifier >> 32) as EGLint,
                    EGL_NONE,
                ], "with-modifier"));
            }

            // Attempt 3: alternate fourcc (XRGB8888) without modifier
            if fourcc != DRM_FORMAT_XRGB8888 {
                attempts.push((vec![
                    EGL_WIDTH, w,
                    EGL_HEIGHT, h,
                    EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_XRGB8888 as EGLint,
                    EGL_DMA_BUF_PLANE0_FD_EXT, fd as EGLint,
                    EGL_DMA_BUF_PLANE0_OFFSET_EXT, offset as EGLint,
                    EGL_DMA_BUF_PLANE0_PITCH_EXT, stride as EGLint,
                    EGL_NONE,
                ], "XRGB8888-no-modifier"));
            }

            let mut last_error = String::new();
            for (attrs, label) in &attempts {
                let image = (self.eglCreateImageKHR)(
                    self.display,
                    EGL_NO_CONTEXT,
                    EGL_LINUX_DMA_BUF_EXT,
                    ptr::null_mut(),
                    attrs.as_ptr(),
                );
                if image == EGL_NO_IMAGE_KHR {
                    last_error = format!("eglCreateImageKHR({}) failed: 0x{:X}", label, (self.eglGetError)());
                    continue;
                }

                let mut tex = 0;
                (self.glGenTextures)(1, &mut tex);
                (self.glBindTexture)(GL_TEXTURE_2D, tex);
                (self.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                (self.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                (self.glEGLImageTargetTexture2DOES)(GL_TEXTURE_2D, image);

                let mut fbo = 0;
                (self.glGenFramebuffers)(1, &mut fbo);
                (self.glBindFramebuffer)(GL_FRAMEBUFFER, fbo);
                (self.glFramebufferTexture2D)(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

                let fb_status = (self.glCheckFramebufferStatus)(GL_FRAMEBUFFER);
                if fb_status != GL_FRAMEBUFFER_COMPLETE {
                    (self.glBindFramebuffer)(GL_FRAMEBUFFER, 0);
                    (self.glDeleteFramebuffers)(1, &fbo);
                    (self.glDeleteTextures)(1, &tex);
                    (self.eglDestroyImageKHR)(self.display, image);
                    last_error = format!("GL framebuffer incomplete ({}, status=0x{:X})", label, fb_status);
                    continue;
                }

                // Success! Read pixels.
                let mut buf = vec![0u8; (width * height * 4) as usize];
                (self.glReadPixels)(0, 0, width as GLsizei, height as GLsizei, GL_BGRA, GL_UNSIGNED_BYTE, buf.as_mut_ptr() as *mut c_void);

                // Cleanup
                (self.glBindFramebuffer)(GL_FRAMEBUFFER, 0);
                (self.glDeleteFramebuffers)(1, &fbo);
                (self.glDeleteTextures)(1, &tex);
                (self.eglDestroyImageKHR)(self.display, image);

                // Log which strategy worked (only on first success)
                if self.cached_modifier.is_none() {
                    log::info!("EGL DMA-BUF import succeeded with strategy '{}' (fourcc=0x{:08X}, modifier=0x{:016X})", label, fourcc, modifier);
                }
                self.cached_modifier = Some(modifier);

                return Ok(buf);
            }

            Err(MediaError::CaptureError(format!("All EGL import attempts failed. Last: {}", last_error)))
        }
    }
}

impl Drop for EglImporter {
    fn drop(&mut self) {
        unsafe {
            if self.context != EGL_NO_CONTEXT {
                (self.eglDestroyContext)(self.display, self.context);
            }
            if self.surface != EGL_NO_SURFACE {
                (self.eglDestroySurface)(self.display, self.surface);
            }
            if self.display != EGL_NO_DISPLAY {
                (self.eglTerminate)(self.display);
            }
            if self._drm_fd >= 0 {
                libc::close(self._drm_fd);
            }
        }
    }
}
