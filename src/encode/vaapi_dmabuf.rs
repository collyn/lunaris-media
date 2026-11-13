//! Zero-copy DMA-BUF → VAAPI surface conversion (Linux).
//!
//! The capture side (PipeWire or KMS) hands us an RGB DMA-BUF that lives on
//! the display GPU. Like Sunshine's VAAPI path, we never touch the pixels on
//! the CPU:
//!
//! ```text
//! DMA-BUF (BGRX, tiled) ──vaCreateSurfaces(DRM_PRIME_2)──▶ VASurface (RGB)
//!        ──VAAPI VPP (colour convert + scale)──▶ NV12 surface from the
//!        encoder's hw_frames pool ──▶ h264/hevc/av1_vaapi
//! ```
//!
//! Imported RGB surfaces are cached per DMA-BUF inode: compositors and KMS
//! cycle through a handful of buffers, so after warm-up there is no per-frame
//! import cost at all.
//!
//! libva is loaded at runtime (it is already loaded by FFmpeg's VAAPI
//! hwcontext, so this only resolves symbols). Struct layouts below match the
//! libva 2.x ABI (`VAProcPipelineParameterBuffer` is padded to a fixed size).

use std::collections::HashMap;
use std::os::unix::io::RawFd;

use crate::colorspace::{codec_colorspace, Colorspace, StreamColorspace};
use crate::error::MediaError;

type VADisplay = *mut libc::c_void;
type VAStatus = libc::c_int;
type VASurfaceID = u32;
type VAConfigID = u32;
type VAContextID = u32;
type VABufferID = u32;

const VA_STATUS_SUCCESS: VAStatus = 0;
const VA_INVALID_ID: u32 = 0xffff_ffff;
const VA_PROFILE_NONE: libc::c_int = -1;
const VA_ENTRYPOINT_VIDEO_PROC: libc::c_int = 10;
const VA_PROGRESSIVE: libc::c_int = 0x1;
const VA_PROC_PIPELINE_PARAMETER_BUFFER_TYPE: libc::c_int = 41;
const VA_RT_FORMAT_RGB32: u32 = 0x0002_0000;
const VA_SURFACE_ATTRIB_MEMORY_TYPE: u32 = 6;
const VA_SURFACE_ATTRIB_EXTERNAL_BUFFER_DESCRIPTOR: u32 = 7;
const VA_SURFACE_ATTRIB_SETTABLE: u32 = 0x2;
const VA_GENERIC_VALUE_TYPE_INTEGER: u32 = 1;
const VA_GENERIC_VALUE_TYPE_POINTER: u32 = 3;
const VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2: u32 = 0x4000_0000;

const VA_FOURCC_BGRX: u32 = 0x5852_4742;
const VA_FOURCC_BGRA: u32 = 0x4152_4742;
const VA_FOURCC_RGBX: u32 = 0x5842_4752;
const VA_FOURCC_RGBA: u32 = 0x4142_4752;

/// `VAProcColorStandardBT601` (va_vpp.h:319, enum `_VAProcColorStandardType`).
const VA_PROC_COLOR_STANDARD_BT601: u32 = 1;
/// `VAProcColorStandardBT709` (va_vpp.h:327).
const VA_PROC_COLOR_STANDARD_BT709: u32 = 2;
/// `VAProcColorStandardBT2020` (va_vpp.h:404).
const VA_PROC_COLOR_STANDARD_BT2020: u32 = 12;
/// `VA_SOURCE_RANGE_REDUCED` (va_vpp.h:560).
const VA_SOURCE_RANGE_REDUCED: u8 = 1;
/// `VA_SOURCE_RANGE_FULL` (va_vpp.h:561).
const VA_SOURCE_RANGE_FULL: u8 = 2;
/// Matrix coefficients 0 = identity (the data is RGB, not YCbCr); ISO/IEC 23001-8.
const CICP_IDENTITY: u8 = 0;

/// Size of `VAProcPipelineParameterBuffer` (fixed by `VA_PADDING_LARGE`).
const PIPELINE_PARAM_SIZE: usize = 224;
const PP_SURFACE: usize = 0;
const PP_SURFACE_REGION: usize = 8;
const PP_SURFACE_COLOR_STANDARD: usize = 16;
const PP_OUTPUT_REGION: usize = 24;
const PP_OUTPUT_BACKGROUND_COLOR: usize = 32;
const PP_OUTPUT_COLOR_STANDARD: usize = 36;
const PP_INPUT_COLOR_PROPERTIES: usize = 132;
const PP_OUTPUT_COLOR_PROPERTIES: usize = 140;
// Offsets inside VAProcColorProperties.
const CP_COLOR_RANGE: usize = 1;
const CP_PRIMARIES: usize = 2;
const CP_TRANSFER: usize = 3;
const CP_MATRIX: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VARectangle {
    x: i16,
    y: i16,
    width: u16,
    height: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DrmPrimeObject {
    fd: libc::c_int,
    size: u32,
    drm_format_modifier: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DrmPrimeLayer {
    drm_format: u32,
    num_planes: u32,
    object_index: [u32; 4],
    offset: [u32; 4],
    pitch: [u32; 4],
}

/// `VADRMPRIMESurfaceDescriptor`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VADRMPRIMESurfaceDescriptor {
    fourcc: u32,
    width: u32,
    height: u32,
    num_objects: u32,
    objects: [DrmPrimeObject; 4],
    num_layers: u32,
    layers: [DrmPrimeLayer; 4],
}

/// `VAGenericValue` (type + 8-byte union).
#[repr(C)]
#[derive(Clone, Copy)]
struct VAGenericValue {
    type_: u32,
    value: u64,
}

/// `VASurfaceAttrib`.
#[repr(C)]
#[derive(Clone, Copy)]
struct VASurfaceAttrib {
    type_: u32,
    flags: u32,
    value: VAGenericValue,
}

const _: () = assert!(std::mem::size_of::<VADRMPRIMESurfaceDescriptor>() == 312);
const _: () = assert!(std::mem::size_of::<VASurfaceAttrib>() == 24);
const _: () = assert!(std::mem::size_of::<VARectangle>() == 8);

type VaCreateConfigFn = unsafe extern "C" fn(
    VADisplay,
    libc::c_int,
    libc::c_int,
    *mut libc::c_void,
    libc::c_int,
    *mut VAConfigID,
) -> VAStatus;
type VaDestroyConfigFn = unsafe extern "C" fn(VADisplay, VAConfigID) -> VAStatus;
type VaCreateContextFn = unsafe extern "C" fn(
    VADisplay,
    VAConfigID,
    libc::c_int,
    libc::c_int,
    libc::c_int,
    *mut VASurfaceID,
    libc::c_int,
    *mut VAContextID,
) -> VAStatus;
type VaDestroyContextFn = unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus;
type VaCreateSurfacesFn = unsafe extern "C" fn(
    VADisplay,
    libc::c_uint,
    libc::c_uint,
    libc::c_uint,
    *mut VASurfaceID,
    libc::c_uint,
    *mut VASurfaceAttrib,
    libc::c_uint,
) -> VAStatus;
type VaDestroySurfacesFn =
    unsafe extern "C" fn(VADisplay, *mut VASurfaceID, libc::c_int) -> VAStatus;
type VaCreateBufferFn = unsafe extern "C" fn(
    VADisplay,
    VAContextID,
    libc::c_int,
    libc::c_uint,
    libc::c_uint,
    *mut libc::c_void,
    *mut VABufferID,
) -> VAStatus;
type VaDestroyBufferFn = unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus;
type VaBeginPictureFn = unsafe extern "C" fn(VADisplay, VAContextID, VASurfaceID) -> VAStatus;
type VaRenderPictureFn =
    unsafe extern "C" fn(VADisplay, VAContextID, *mut VABufferID, libc::c_int) -> VAStatus;
type VaEndPictureFn = unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus;
type VaSyncSurfaceFn = unsafe extern "C" fn(VADisplay, VASurfaceID) -> VAStatus;
type VaErrorStrFn = unsafe extern "C" fn(VAStatus) -> *const libc::c_char;

struct VaApi {
    _lib: libloading::Library,
    create_config: VaCreateConfigFn,
    destroy_config: VaDestroyConfigFn,
    create_context: VaCreateContextFn,
    destroy_context: VaDestroyContextFn,
    create_surfaces: VaCreateSurfacesFn,
    destroy_surfaces: VaDestroySurfacesFn,
    create_buffer: VaCreateBufferFn,
    destroy_buffer: VaDestroyBufferFn,
    begin_picture: VaBeginPictureFn,
    render_picture: VaRenderPictureFn,
    end_picture: VaEndPictureFn,
    sync_surface: VaSyncSurfaceFn,
    error_str: VaErrorStrFn,
}

impl VaApi {
    fn load() -> Result<Self, MediaError> {
        let lib = unsafe { libloading::Library::new("libva.so.2") }
            .map_err(|e| MediaError::EncodeError(format!("libva.so.2: {}", e)))?;
        macro_rules! sym {
            ($name:literal) => {
                unsafe {
                    *lib.get(concat!($name, "\0").as_bytes()).map_err(|e| {
                        MediaError::EncodeError(format!("libva symbol {}: {}", $name, e))
                    })?
                }
            };
        }
        Ok(Self {
            create_config: sym!("vaCreateConfig"),
            destroy_config: sym!("vaDestroyConfig"),
            create_context: sym!("vaCreateContext"),
            destroy_context: sym!("vaDestroyContext"),
            create_surfaces: sym!("vaCreateSurfaces"),
            destroy_surfaces: sym!("vaDestroySurfaces"),
            create_buffer: sym!("vaCreateBuffer"),
            destroy_buffer: sym!("vaDestroyBuffer"),
            begin_picture: sym!("vaBeginPicture"),
            render_picture: sym!("vaRenderPicture"),
            end_picture: sym!("vaEndPicture"),
            sync_surface: sym!("vaSyncSurface"),
            error_str: sym!("vaErrorStr"),
            _lib: lib,
        })
    }

    fn check(&self, what: &str, status: VAStatus) -> Result<(), MediaError> {
        if status == VA_STATUS_SUCCESS {
            return Ok(());
        }
        let msg = unsafe {
            let p = (self.error_str)(status);
            if p.is_null() {
                String::from("unknown")
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Err(MediaError::EncodeError(format!(
            "{} failed: {} ({})",
            what, msg, status
        )))
    }
}

/// Description of one single-plane RGB DMA-BUF frame.
#[derive(Clone, Copy, Debug)]
pub struct DmabufFrame {
    pub fd: RawFd,
    pub offset: u32,
    pub stride: u32,
    pub modifier: u64,
    pub size: usize,
    pub width: u32,
    pub height: u32,
    pub drm_format: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct SurfaceKey {
    inode: u64,
    offset: u32,
    stride: u32,
    modifier: u64,
    width: u32,
    height: u32,
    drm_format: u32,
}

/// Maximum number of cached imported surfaces before the cache is flushed
/// (protects against compositors that keep reallocating buffers).
const MAX_CACHED_SURFACES: usize = 16;

/// Imports RGB DMA-BUFs and converts them into NV12 VAAPI surfaces.
pub struct VaapiDmabufConverter {
    va: VaApi,
    display: VADisplay,
    config: VAConfigID,
    context: VAContextID,
    out_width: u32,
    out_height: u32,
    /// Colourspace and range the NV12 output is converted to.
    colorspace: StreamColorspace,
    surfaces: HashMap<SurfaceKey, VASurfaceID>,
}

// The VADisplay is owned by FFmpeg's hw device context, which outlives this
// converter (it is dropped before the device context is unreferenced).
unsafe impl Send for VaapiDmabufConverter {}

fn drm_to_va_fourcc(drm: u32) -> Option<u32> {
    use crate::capture::dmabuf_caps::*;
    Some(match drm {
        DRM_FORMAT_XRGB8888 => VA_FOURCC_BGRX,
        DRM_FORMAT_ARGB8888 => VA_FOURCC_BGRA,
        DRM_FORMAT_XBGR8888 => VA_FOURCC_RGBX,
        DRM_FORMAT_ABGR8888 => VA_FOURCC_RGBA,
        _ => return None,
    })
}

impl VaapiDmabufConverter {
    /// Create a VPP context on `display` producing `out_width`×`out_height`.
    ///
    /// # Safety
    /// `display` must be a valid, initialised `VADisplay` that outlives the
    /// returned converter.
    pub unsafe fn new(
        display: VADisplay,
        out_width: u32,
        out_height: u32,
        colorspace: StreamColorspace,
    ) -> Result<Self, MediaError> {
        if display.is_null() {
            return Err(MediaError::EncodeError("VADisplay is null".into()));
        }
        let va = VaApi::load()?;
        let mut config: VAConfigID = VA_INVALID_ID;
        va.check(
            "vaCreateConfig(VideoProc)",
            (va.create_config)(
                display,
                VA_PROFILE_NONE,
                VA_ENTRYPOINT_VIDEO_PROC,
                std::ptr::null_mut(),
                0,
                &mut config,
            ),
        )?;
        let mut context: VAContextID = VA_INVALID_ID;
        if let Err(e) = va.check(
            "vaCreateContext(VideoProc)",
            (va.create_context)(
                display,
                config,
                out_width as libc::c_int,
                out_height as libc::c_int,
                VA_PROGRESSIVE,
                std::ptr::null_mut(),
                0,
                &mut context,
            ),
        ) {
            (va.destroy_config)(display, config);
            return Err(e);
        }
        log::info!(
            "VAAPI zero-copy: VPP context ready ({}x{} NV12 output)",
            out_width,
            out_height
        );
        Ok(Self {
            va,
            display,
            config,
            context,
            out_width,
            out_height,
            colorspace,
            surfaces: HashMap::new(),
        })
    }

    fn import(&mut self, frame: &DmabufFrame) -> Result<VASurfaceID, MediaError> {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(frame.fd, &mut st) } != 0 {
            return Err(MediaError::EncodeError(format!(
                "fstat(dmabuf) failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        let key = SurfaceKey {
            inode: st.st_ino as u64,
            offset: frame.offset,
            stride: frame.stride,
            modifier: frame.modifier,
            width: frame.width,
            height: frame.height,
            drm_format: frame.drm_format,
        };
        if let Some(&surface) = self.surfaces.get(&key) {
            return Ok(surface);
        }
        if self.surfaces.len() >= MAX_CACHED_SURFACES {
            self.flush_cache();
        }

        let va_fourcc = drm_to_va_fourcc(frame.drm_format).ok_or_else(|| {
            MediaError::EncodeError(format!("unsupported DRM format 0x{:08X}", frame.drm_format))
        })?;
        let size = if frame.size > 0 {
            frame.size as u32
        } else {
            frame.offset + frame.stride * frame.height
        };
        let mut desc = VADRMPRIMESurfaceDescriptor {
            fourcc: va_fourcc,
            width: frame.width,
            height: frame.height,
            num_objects: 1,
            num_layers: 1,
            ..Default::default()
        };
        desc.objects[0] = DrmPrimeObject {
            fd: frame.fd,
            size,
            drm_format_modifier: frame.modifier,
        };
        desc.layers[0].drm_format = frame.drm_format;
        desc.layers[0].num_planes = 1;
        desc.layers[0].object_index[0] = 0;
        desc.layers[0].offset[0] = frame.offset;
        desc.layers[0].pitch[0] = frame.stride;

        let mut attribs = [
            VASurfaceAttrib {
                type_: VA_SURFACE_ATTRIB_MEMORY_TYPE,
                flags: VA_SURFACE_ATTRIB_SETTABLE,
                value: VAGenericValue {
                    type_: VA_GENERIC_VALUE_TYPE_INTEGER,
                    value: VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2 as u64,
                },
            },
            VASurfaceAttrib {
                type_: VA_SURFACE_ATTRIB_EXTERNAL_BUFFER_DESCRIPTOR,
                flags: VA_SURFACE_ATTRIB_SETTABLE,
                value: VAGenericValue {
                    type_: VA_GENERIC_VALUE_TYPE_POINTER,
                    value: &mut desc as *mut _ as usize as u64,
                },
            },
        ];
        let mut surface: VASurfaceID = VA_INVALID_ID;
        let status = unsafe {
            (self.va.create_surfaces)(
                self.display,
                VA_RT_FORMAT_RGB32,
                frame.width,
                frame.height,
                &mut surface,
                1,
                attribs.as_mut_ptr(),
                attribs.len() as libc::c_uint,
            )
        };
        self.va.check("vaCreateSurfaces(DRM_PRIME_2)", status)?;
        log::debug!(
            "VAAPI zero-copy: imported DMA-BUF ino={} {}x{} fmt=0x{:08X} mod=0x{:016X} → surface {}",
            key.inode,
            frame.width,
            frame.height,
            frame.drm_format,
            frame.modifier,
            surface
        );
        self.surfaces.insert(key, surface);
        Ok(surface)
    }

    /// Colour-convert (and scale) `frame` into the NV12 surface `dst`.
    ///
    /// Blocks until the GPU finished reading the source, so the capture side
    /// may recycle its buffer as soon as this returns.
    pub fn convert(&mut self, frame: &DmabufFrame, dst: u32) -> Result<(), MediaError> {
        let src = self.import(frame)?;

        let src_rect = VARectangle {
            x: 0,
            y: 0,
            width: frame.width.min(u16::MAX as u32) as u16,
            height: frame.height.min(u16::MAX as u32) as u16,
        };
        let dst_rect = VARectangle {
            x: 0,
            y: 0,
            width: self.out_width.min(u16::MAX as u32) as u16,
            height: self.out_height.min(u16::MAX as u32) as u16,
        };

        let mut params = [0u8; PIPELINE_PARAM_SIZE];
        let put_u32 = |buf: &mut [u8; PIPELINE_PARAM_SIZE], off: usize, v: u32| {
            buf[off..off + 4].copy_from_slice(&v.to_ne_bytes());
        };
        let put_ptr = |buf: &mut [u8; PIPELINE_PARAM_SIZE], off: usize, p: *const VARectangle| {
            buf[off..off + 8].copy_from_slice(&(p as usize as u64).to_ne_bytes());
        };
        put_u32(&mut params, PP_SURFACE, src);
        put_ptr(&mut params, PP_SURFACE_REGION, &src_rect);
        // Desktop pixels must keep their code values: only the RGB→YCbCr
        // matrix may change. Declaring the input as sRGB made the VPP convert
        // the transfer curve (sRGB → BT.709), lifting mid-tones (grey 128 was
        // encoded as ~136). So the input uses the same primaries/transfer as
        // the output, with the identity (RGB) matrix — what Sunshine does too.
        let color_standard = match self.colorspace.colorspace {
            Colorspace::Rec601 => VA_PROC_COLOR_STANDARD_BT601,
            Colorspace::Rec709 => VA_PROC_COLOR_STANDARD_BT709,
            Colorspace::Bt2020Sdr | Colorspace::Bt2020 => VA_PROC_COLOR_STANDARD_BT2020,
        };
        let codec = codec_colorspace(&self.colorspace);
        let range = if codec.full_range {
            VA_SOURCE_RANGE_FULL
        } else {
            VA_SOURCE_RANGE_REDUCED
        };
        put_u32(&mut params, PP_SURFACE_COLOR_STANDARD, color_standard);
        put_ptr(&mut params, PP_OUTPUT_REGION, &dst_rect);
        put_u32(&mut params, PP_OUTPUT_BACKGROUND_COLOR, 0xff00_0000);
        put_u32(&mut params, PP_OUTPUT_COLOR_STANDARD, color_standard);
        // The input is full-range RGB; the output range and code points are
        // the colour description the encoder signals (codec_colorspace).
        params[PP_INPUT_COLOR_PROPERTIES + CP_COLOR_RANGE] = VA_SOURCE_RANGE_FULL;
        params[PP_INPUT_COLOR_PROPERTIES + CP_PRIMARIES] = codec.primaries;
        params[PP_INPUT_COLOR_PROPERTIES + CP_TRANSFER] = codec.transfer;
        params[PP_INPUT_COLOR_PROPERTIES + CP_MATRIX] = CICP_IDENTITY;
        params[PP_OUTPUT_COLOR_PROPERTIES + CP_COLOR_RANGE] = range;
        params[PP_OUTPUT_COLOR_PROPERTIES + CP_PRIMARIES] = codec.primaries;
        params[PP_OUTPUT_COLOR_PROPERTIES + CP_TRANSFER] = codec.transfer;
        params[PP_OUTPUT_COLOR_PROPERTIES + CP_MATRIX] = codec.matrix;

        unsafe {
            let mut buffer: VABufferID = VA_INVALID_ID;
            self.va.check(
                "vaCreateBuffer(ProcPipeline)",
                (self.va.create_buffer)(
                    self.display,
                    self.context,
                    VA_PROC_PIPELINE_PARAMETER_BUFFER_TYPE,
                    PIPELINE_PARAM_SIZE as libc::c_uint,
                    1,
                    params.as_mut_ptr() as *mut libc::c_void,
                    &mut buffer,
                ),
            )?;

            let result = (|| {
                self.va.check(
                    "vaBeginPicture",
                    (self.va.begin_picture)(self.display, self.context, dst),
                )?;
                let render = (self.va.render_picture)(self.display, self.context, &mut buffer, 1);
                // vaEndPicture must follow vaBeginPicture even on error.
                let end = (self.va.end_picture)(self.display, self.context);
                self.va.check("vaRenderPicture", render)?;
                self.va.check("vaEndPicture", end)?;
                self.va
                    .check("vaSyncSurface", (self.va.sync_surface)(self.display, dst))
            })();

            (self.va.destroy_buffer)(self.display, buffer);
            result
        }
    }

    fn flush_cache(&mut self) {
        let mut ids: Vec<VASurfaceID> = self.surfaces.drain().map(|(_, s)| s).collect();
        if !ids.is_empty() {
            unsafe {
                (self.va.destroy_surfaces)(
                    self.display,
                    ids.as_mut_ptr(),
                    ids.len() as libc::c_int,
                );
            }
        }
    }
}

impl Drop for VaapiDmabufConverter {
    fn drop(&mut self) {
        self.flush_cache();
        unsafe {
            if self.context != VA_INVALID_ID {
                (self.va.destroy_context)(self.display, self.context);
            }
            if self.config != VA_INVALID_ID {
                (self.va.destroy_config)(self.display, self.config);
            }
        }
    }
}
