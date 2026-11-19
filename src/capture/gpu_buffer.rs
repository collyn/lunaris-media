//! Cross-platform GPU buffer abstraction.
//!
//! [`GpuBuffer`] represents a frame residing in GPU memory. Each variant wraps
//! a platform-specific handle (DMA-BUF fd on Linux, D3D11 texture on Windows,
//! CVPixelBuffer on macOS) so that the encoder can consume the frame without
//! any GPU→CPU→GPU round-trip.

use crate::types::PixelFormat;

/// Planes 1..=3 of a multi-plane DMA-BUF whose planes all live in the same
/// buffer object (offset, pitch), as KMS scan-out buffers with compression
/// metadata do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DmaBufAuxPlanes {
    /// Number of valid entries in `planes`.
    pub count: u8,
    /// (offset, pitch) of each further plane.
    pub planes: [(u32, u32); 3],
}

impl DmaBufAuxPlanes {
    /// No further planes.
    pub const NONE: DmaBufAuxPlanes = DmaBufAuxPlanes { count: 0, planes: [(0, 0); 3] };

    /// Build from (offset, pitch) pairs; `None` if there are more than 3.
    pub fn from_slice(planes: &[(u32, u32)]) -> Option<Self> {
        if planes.len() > 3 {
            return None;
        }
        let mut aux = Self::NONE;
        for (i, p) in planes.iter().enumerate() {
            aux.planes[i] = *p;
        }
        aux.count = planes.len() as u8;
        Some(aux)
    }

    /// The valid (offset, pitch) pairs.
    pub fn as_slice(&self) -> &[(u32, u32)] {
        &self.planes[..self.count.min(3) as usize]
    }
}

/// Represents a GPU-resident buffer that can be passed directly to a hardware
/// encoder without copying through CPU memory.
pub enum GpuBuffer {
    /// Linux: DMA-BUF file descriptor pointing to GPU memory.
    ///
    /// The file descriptor is borrowed from the PipeWire/DRM subsystem and must
    /// remain valid for the lifetime of this buffer.
    #[cfg(target_os = "linux")]
    DmaBuf {
        /// DMA-BUF file descriptor.
        fd: std::os::unix::io::RawFd,
        /// Byte offset into the DMA-BUF where pixel data begins.
        offset: u32,
        /// Row stride in bytes.
        stride: u32,
        /// DRM format modifier (DRM_FORMAT_MOD_*).
        modifier: u64,
        /// Total buffer size in bytes.
        size: usize,
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
        /// DRM fourcc pixel format code.
        fourcc: u32,
        /// Further planes of the same buffer object (same `fd`), e.g. the
        /// compression metadata of AMD DCC or Intel CCS scan-out buffers.
        /// Empty for ordinary single-plane buffers.
        aux_planes: DmaBufAuxPlanes,
    },

    /// Linux/NVIDIA: CUDA device memory pointer.
    #[cfg(target_os = "linux")]
    CudaPointer {
        /// Address of the CUDA buffer.
        ptr: usize,
        /// CUDA context that owns this pointer, if known.
        cuda_context: usize,
        /// Total buffer size in bytes.
        size: usize,
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
        /// Row stride/pitch in bytes.
        stride: u32,
        /// Pixel format.
        format: PixelFormat,
    },

    /// Windows: D3D11 Texture residing in GPU VRAM.
    ///
    /// The texture pointer is an `ID3D11Texture2D*` cast to `*mut c_void`.
    #[cfg(target_os = "windows")]
    D3D11Texture {
        /// Pointer to the ID3D11Texture2D interface.
        texture: *mut std::ffi::c_void,
        /// Array slice index within the texture.
        array_index: u32,
    },

    /// macOS: CVPixelBuffer backed by IOSurface in GPU memory.
    #[cfg(target_os = "macos")]
    CVPixelBuffer {
        /// Pointer to the CVPixelBufferRef.
        pixel_buffer: *mut std::ffi::c_void,
    },

    /// Fallback: CPU-accessible memory (used when zero-copy is not available).
    CpuBuffer {
        /// Raw pixel data.
        data: Vec<u8>,
        /// Row stride in bytes.
        stride: u32,
        /// Pixel format of the data.
        format: PixelFormat,
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(target_os = "linux")]
            GpuBuffer::DmaBuf {
                fd, width, height, ..
            } => {
                write!(f, "DmaBuf(fd={}, {}x{})", fd, width, height)
            }
            #[cfg(target_os = "linux")]
            GpuBuffer::CudaPointer {
                ptr, width, height, ..
            } => {
                write!(f, "CudaPointer(ptr=0x{:X}, {}x{})", ptr, width, height)
            }
            #[cfg(target_os = "windows")]
            GpuBuffer::D3D11Texture { array_index, .. } => {
                write!(f, "D3D11Texture(index={})", array_index)
            }
            #[cfg(target_os = "macos")]
            GpuBuffer::CVPixelBuffer { .. } => write!(f, "CVPixelBuffer"),
            GpuBuffer::CpuBuffer {
                width,
                height,
                format,
                ..
            } => {
                write!(f, "CpuBuffer({}x{}, {:?})", width, height, format)
            }
        }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if let GpuBuffer::DmaBuf { fd, .. } = self {
            if *fd >= 0 {
                // SAFETY: fd is an owned file descriptor exported via DRM or PipeWire dup.
                // Closing it releases the kernel reference to the DMA-BUF.
                unsafe {
                    libc::close(*fd);
                }
                log::debug!("Closed GpuBuffer DmaBuf fd: {}", fd);
            }
        }
    }
}

// Safety: GPU buffer handles (file descriptors, pointers) are safe to send
// between threads when the capture/encode pipeline ensures proper
// synchronization. The pipeline is responsible for guaranteeing that no two
// threads access the same underlying GPU resource concurrently.
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}
