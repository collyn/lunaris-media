//! Video encoding subsystem.
//!
//! This module defines the [`VideoEncoder`] trait for encoding captured GPU
//! frames into a compressed bitstream (e.g., Annex-B H.264). The primary
//! implementation wraps FFmpeg's hardware-accelerated encoders (VAAPI, NVENC,
//! QSV, AMF, VideoToolbox).
//!
//! Use [`create_encoder`] to obtain the best available encoder for the current
//! platform, or [`list_available_encoders`] to enumerate all detected hardware
//! encoders.

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod ffmpeg;
#[cfg(target_os = "windows")]
pub mod windows_amf;
#[cfg(target_os = "windows")]
pub mod windows_nvenc;
#[cfg(target_os = "windows")]
pub mod windows_qsv;

use crate::capture::gpu_buffer::GpuBuffer;
use crate::error::MediaError;
use crate::types::*;

/// Configuration for initializing a [`VideoEncoder`].
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Video codec to encode.
    pub codec: VideoCodec,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Target frame rate.
    pub fps: u32,
    /// Target bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Enable low-latency tuning (disables look-ahead, reduces buffering).
    pub low_latency: bool,
    /// Keyframe interval in frames. `0` means automatic (`2 × fps`).
    pub keyframe_interval: u32,
    /// Preferred hardware acceleration type, or `None` for auto-detection.
    pub preferred_hw: Option<HwAccelType>,
    /// Force the FFmpeg backend even when a native Windows backend is available.
    pub force_ffmpeg: bool,
    /// Optional Direct3D11 device pointer (cast to usize) for Windows zero-copy GPU encoding.
    pub d3d11_device: Option<usize>,
    /// Optional Direct3D11 device context pointer (cast to usize) for Windows zero-copy GPU encoding.
    pub d3d11_context: Option<usize>,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            codec: VideoCodec::H264,
            width: 1920,
            height: 1080,
            fps: 60,
            bitrate_kbps: 10_000,
            low_latency: true,
            keyframe_interval: 0,
            preferred_hw: None,
            force_ffmpeg: false,
            d3d11_device: None,
            d3d11_context: None,
        }
    }
}

/// Trait for hardware-accelerated video encoders.
///
/// Implementations consume GPU-resident buffers from the capture subsystem and
/// produce encoded video frames. The encoder is designed to be driven from a
/// single thread/task; `Send` is required so the owning task can migrate between
/// executor threads.
pub trait VideoEncoder: Send {
    /// Initialize the encoder with the given configuration.
    ///
    /// Must be called before [`encode`](Self::encode). May probe the system for
    /// available hardware and select the best encoder automatically.
    fn initialize(&mut self, config: &EncoderConfig) -> Result<(), MediaError>;

    /// Encode a single GPU frame.
    ///
    /// Returns zero or more encoded frames (encoders may buffer internally).
    fn encode(
        &mut self,
        buffer: &GpuBuffer,
        pts_us: u64,
    ) -> Result<Vec<EncodedVideoFrame>, MediaError>;

    /// Request that the next encoded frame be an IDR/keyframe.
    fn request_keyframe(&mut self);

    /// Dynamically change the target bitrate.
    fn set_bitrate(&mut self, bitrate_kbps: u32) -> Result<(), MediaError>;

    /// Dynamically change the target frame rate.
    fn set_fps(&mut self, fps: u32) -> Result<(), MediaError> {
        let _ = fps;
        Ok(())
    }

    /// Return metadata about this encoder instance.
    fn encoder_info(&self) -> EncoderInfo;

    /// Flush any internally buffered frames.
    fn flush(&mut self) -> Result<Vec<EncodedVideoFrame>, MediaError>;

    /// Shut down the encoder and release all resources.
    fn shutdown(&mut self);
}

#[cfg(target_os = "windows")]
enum WindowsEncoderBackend {
    NativeNvenc(windows_nvenc::WindowsNvencEncoder),
    NativeAmf(windows_amf::WindowsAmfEncoder),
    NativeQsv(windows_qsv::WindowsQsvEncoder),
}

#[cfg(target_os = "windows")]
struct WindowsAutoEncoder {
    backend: Option<WindowsEncoderBackend>,
}

#[cfg(target_os = "windows")]
impl WindowsAutoEncoder {
    fn new() -> Self {
        Self { backend: None }
    }

    fn detected_d3d11_hw(config: &EncoderConfig) -> Option<HwAccelType> {
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D11::ID3D11Device;
        use windows::Win32::Graphics::Dxgi::IDXGIDevice;

        let device_ptr = config.d3d11_device? as *mut std::ffi::c_void;
        unsafe {
            let device = std::mem::ManuallyDrop::new(ID3D11Device::from_raw(device_ptr));
            let dxgi_device: IDXGIDevice = (*device).cast().ok()?;
            let adapter = dxgi_device.GetAdapter().ok()?;
            let desc = adapter.GetDesc().ok()?;
            match desc.VendorId {
                0x10DE => Some(HwAccelType::Nvenc),
                0x1002 => Some(HwAccelType::Amf),
                0x8086 => Some(HwAccelType::Qsv),
                _ => None,
            }
        }
    }

    fn has_d3d11_pair(config: &EncoderConfig) -> bool {
        config.d3d11_device.is_some() && config.d3d11_context.is_some()
    }

    fn supports_native_d3d11_h264(config: &EncoderConfig) -> bool {
        config.codec == VideoCodec::H264 && Self::has_d3d11_pair(config)
    }

    fn supports_native_d3d11_amf(config: &EncoderConfig) -> bool {
        matches!(config.codec, VideoCodec::H264 | VideoCodec::H265 | VideoCodec::AV1) && Self::has_d3d11_pair(config)
    }

    fn supports_native_d3d11_qsv(config: &EncoderConfig) -> bool {
        matches!(config.codec, VideoCodec::H264 | VideoCodec::H265 | VideoCodec::AV1) && Self::has_d3d11_pair(config)
    }

    fn should_try_native_nvenc(config: &EncoderConfig) -> bool {
        if config.force_ffmpeg || !Self::supports_native_d3d11_h264(config) {
            return false;
        }

        match config.preferred_hw {
            Some(HwAccelType::Nvenc) => true,
            Some(_) => false,
            None => Self::detected_d3d11_hw(config).map_or(false, |hw| hw == HwAccelType::Nvenc),
        }
    }

    fn should_try_native_amf(config: &EncoderConfig) -> bool {
        if config.force_ffmpeg || !Self::supports_native_d3d11_amf(config) {
            return false;
        }

        match config.preferred_hw {
            Some(HwAccelType::Amf) => true,
            Some(_) => false,
            None => Self::detected_d3d11_hw(config).map_or(false, |hw| hw == HwAccelType::Amf),
        }
    }

    fn should_try_native_qsv(config: &EncoderConfig) -> bool {
        if config.force_ffmpeg || !Self::supports_native_d3d11_qsv(config) {
            return false;
        }

        match config.preferred_hw {
            Some(HwAccelType::Qsv) => true,
            Some(_) => false,
            None => Self::detected_d3d11_hw(config).map_or(false, |hw| hw == HwAccelType::Qsv),
        }
    }
}

#[cfg(target_os = "windows")]
impl VideoEncoder for WindowsAutoEncoder {
    fn initialize(&mut self, config: &EncoderConfig) -> Result<(), MediaError> {
        if config.force_ffmpeg {
            return Err(MediaError::EncoderInitFailed(
                "FFmpeg backend is not compiled on Windows; select native_nvenc_d3d11, native_amf_d3d11, gpu, or auto".into(),
            ));
        }

        if Self::should_try_native_nvenc(config) {
            match windows_nvenc::WindowsNvencEncoder::new().and_then(|mut encoder| {
                encoder.initialize(config)?;
                Ok(encoder)
            }) {
                Ok(encoder) => {
                    log::info!("Selected native Windows NVENC D3D11 encoder");
                    self.backend = Some(WindowsEncoderBackend::NativeNvenc(encoder));
                    return Ok(());
                }
                Err(err) => {
                    log::warn!("Native Windows NVENC initialization failed: {}", err);
                }
            }
        }

        if Self::should_try_native_amf(config) {
            match windows_amf::WindowsAmfEncoder::new().and_then(|mut encoder| {
                encoder.initialize(config)?;
                Ok(encoder)
            }) {
                Ok(encoder) => {
                    log::info!("Selected native Windows AMF D3D11 encoder");
                    self.backend = Some(WindowsEncoderBackend::NativeAmf(encoder));
                    return Ok(());
                }
                Err(err) => {
                    log::warn!("Native Windows AMF initialization failed: {}", err);
                }
            }
        }

        if Self::should_try_native_qsv(config) {
            match windows_qsv::WindowsQsvEncoder::new().and_then(|mut encoder| {
                encoder.initialize(config)?;
                Ok(encoder)
            }) {
                Ok(encoder) => {
                    log::info!("Selected native Windows QSV D3D11 encoder");
                    self.backend = Some(WindowsEncoderBackend::NativeQsv(encoder));
                    return Ok(());
                }
                Err(err) => {
                    log::warn!("Native Windows QSV initialization failed: {}", err);
                }
            }
        }

        let adapter = Self::detected_d3d11_hw(config);
        let reason = match (config.preferred_hw, adapter) {
            (Some(HwAccelType::Qsv), _) | (_, Some(HwAccelType::Qsv)) => {
                "Native Intel/QSV Windows encoding requires libvpl.dll and successful initialization"
            }
            (Some(HwAccelType::Software), _) => {
                "Software/FFmpeg encoding is not compiled on Windows"
            }
            (Some(HwAccelType::Nvenc), _) | (_, Some(HwAccelType::Nvenc)) => {
                "Native NVENC requires a D3D11 device/context and successful NVENC initialization"
            }
            (Some(HwAccelType::Amf), _) | (_, Some(HwAccelType::Amf)) => {
                "Native AMF requires a D3D11 device/context and successful AMF initialization"
            }
            _ => "No supported native Windows encoder was available",
        };
        Err(MediaError::EncoderInitFailed(reason.into()))
    }

    fn encode(
        &mut self,
        buffer: &GpuBuffer,
        pts_us: u64,
    ) -> Result<Vec<EncodedVideoFrame>, MediaError> {
        match self.backend.as_mut() {
            Some(WindowsEncoderBackend::NativeNvenc(encoder)) => encoder.encode(buffer, pts_us),
            Some(WindowsEncoderBackend::NativeAmf(encoder)) => encoder.encode(buffer, pts_us),
            Some(WindowsEncoderBackend::NativeQsv(encoder)) => encoder.encode(buffer, pts_us),
            None => Err(MediaError::EncoderNotInitialized),
        }
    }

    fn request_keyframe(&mut self) {
        match self.backend.as_mut() {
            Some(WindowsEncoderBackend::NativeNvenc(encoder)) => encoder.request_keyframe(),
            Some(WindowsEncoderBackend::NativeAmf(encoder)) => encoder.request_keyframe(),
            Some(WindowsEncoderBackend::NativeQsv(encoder)) => encoder.request_keyframe(),
            None => {}
        }
    }

    fn set_bitrate(&mut self, bitrate_kbps: u32) -> Result<(), MediaError> {
        match self.backend.as_mut() {
            Some(WindowsEncoderBackend::NativeNvenc(encoder)) => encoder.set_bitrate(bitrate_kbps),
            Some(WindowsEncoderBackend::NativeAmf(encoder)) => encoder.set_bitrate(bitrate_kbps),
            Some(WindowsEncoderBackend::NativeQsv(encoder)) => encoder.set_bitrate(bitrate_kbps),
            None => Err(MediaError::EncoderNotInitialized),
        }
    }

    fn set_fps(&mut self, fps: u32) -> Result<(), MediaError> {
        match self.backend.as_mut() {
            Some(WindowsEncoderBackend::NativeNvenc(encoder)) => encoder.set_fps(fps),
            Some(WindowsEncoderBackend::NativeAmf(encoder)) => encoder.set_fps(fps),
            Some(WindowsEncoderBackend::NativeQsv(encoder)) => encoder.set_fps(fps),
            None => Err(MediaError::EncoderNotInitialized),
        }
    }

    fn encoder_info(&self) -> EncoderInfo {
        match self.backend.as_ref() {
            Some(WindowsEncoderBackend::NativeNvenc(encoder)) => encoder.encoder_info(),
            Some(WindowsEncoderBackend::NativeAmf(encoder)) => encoder.encoder_info(),
            Some(WindowsEncoderBackend::NativeQsv(encoder)) => encoder.encoder_info(),
            None => EncoderInfo {
                name: "uninitialized".to_string(),
                hw_type: HwAccelType::Software,
                supported_codecs: vec![],
            },
        }
    }

    fn flush(&mut self) -> Result<Vec<EncodedVideoFrame>, MediaError> {
        match self.backend.as_mut() {
            Some(WindowsEncoderBackend::NativeNvenc(encoder)) => encoder.flush(),
            Some(WindowsEncoderBackend::NativeAmf(encoder)) => encoder.flush(),
            Some(WindowsEncoderBackend::NativeQsv(encoder)) => encoder.flush(),
            None => Ok(vec![]),
        }
    }

    fn shutdown(&mut self) {
        if let Some(backend) = self.backend.as_mut() {
            match backend {
                WindowsEncoderBackend::NativeNvenc(encoder) => encoder.shutdown(),
                WindowsEncoderBackend::NativeAmf(encoder) => encoder.shutdown(),
                WindowsEncoderBackend::NativeQsv(encoder) => encoder.shutdown(),
            }
        }
        self.backend = None;
    }
}

/// Return a human-readable D3D11 adapter description for a borrowed device pointer.
#[cfg(target_os = "windows")]
pub fn describe_d3d11_device(device_ptr: Option<usize>) -> Option<String> {
    use std::mem::ManuallyDrop;
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D11::ID3D11Device;
    use windows::Win32::Graphics::Dxgi::IDXGIDevice;

    let device_ptr = device_ptr? as *mut std::ffi::c_void;
    unsafe {
        let device = ManuallyDrop::new(ID3D11Device::from_raw(device_ptr));
        let dxgi_device: IDXGIDevice = (*device).cast().ok()?;
        let adapter = dxgi_device.GetAdapter().ok()?;
        let desc = adapter.GetDesc().ok()?;
        let name = String::from_utf16_lossy(&desc.Description)
            .trim_end_matches(' ')
            .trim()
            .to_string();
        if name.is_empty() {
            Some(format!(
                "Unknown GPU (vendor={:#06x}, device={:#06x})",
                desc.VendorId, desc.DeviceId
            ))
        } else {
            Some(format!(
                "{} (vendor={:#06x}, device={:#06x})",
                name, desc.VendorId, desc.DeviceId
            ))
        }
    }
}

/// Return a human-readable GPU description when the platform exposes one.
#[cfg(not(target_os = "windows"))]
pub fn describe_d3d11_device(_device_ptr: Option<usize>) -> Option<String> {
    None
}

/// Return a human-readable GPU description for the active host.
pub fn describe_host_gpu(d3d11_device: Option<usize>) -> Option<String> {
    #[cfg(not(target_os = "windows"))]
    let _ = d3d11_device;

    #[cfg(target_os = "windows")]
    {
        return describe_d3d11_device(d3d11_device);
    }

    #[cfg(target_os = "linux")]
    {
        return describe_linux_gpu();
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        let _ = d3d11_device;
        None
    }
}

#[cfg(target_os = "linux")]
fn describe_linux_gpu() -> Option<String> {
    describe_nvidia_proc_gpu().or_else(describe_drm_gpu)
}

#[cfg(target_os = "linux")]
fn describe_nvidia_proc_gpu() -> Option<String> {
    let entries = std::fs::read_dir("/proc/driver/nvidia/gpus").ok()?;
    for entry in entries.flatten() {
        let info = std::fs::read_to_string(entry.path().join("information")).ok()?;
        if let Some(model) = info
            .lines()
            .find_map(|line| line.strip_prefix("Model:").map(str::trim))
        {
            if !model.is_empty() {
                return Some(format!("NVIDIA {model}"));
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn describe_drm_gpu() -> Option<String> {
    let entries = std::fs::read_dir("/sys/class/drm").ok()?;
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if !file_name.starts_with("card") || file_name.contains('-') {
            continue;
        }

        let device_dir = entry.path().join("device");
        let vendor = read_trimmed(device_dir.join("vendor"))?;
        let device =
            read_trimmed(device_dir.join("device")).unwrap_or_else(|| "unknown".to_string());
        let vendor_name = match vendor.as_str() {
            "0x10de" => "NVIDIA",
            "0x1002" | "0x1022" => "AMD",
            "0x8086" => "Intel",
            _ => "GPU",
        };

        if let Some(product_name) = read_trimmed(device_dir.join("product_name"))
            .or_else(|| read_trimmed(device_dir.join("product_info")))
            .filter(|name| !name.is_empty())
        {
            return Some(format!(
                "{vendor_name} {product_name} (vendor={vendor}, device={device})"
            ));
        }

        let driver = read_trimmed(device_dir.join("uevent"))
            .and_then(|uevent| {
                uevent
                    .lines()
                    .find_map(|line| line.strip_prefix("DRIVER=").map(str::to_string))
            })
            .filter(|driver| !driver.is_empty());

        return Some(match driver {
            Some(driver) => {
                format!("{vendor_name} GPU ({driver}, vendor={vendor}, device={device})")
            }
            None => format!("{vendor_name} GPU (vendor={vendor}, device={device})"),
        });
    }
    None
}

#[cfg(target_os = "linux")]
fn read_trimmed(path: impl AsRef<std::path::Path>) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
}

/// Create the best available [`VideoEncoder`] for the current platform.
pub fn create_encoder() -> Result<Box<dyn VideoEncoder>, MediaError> {
    #[cfg(target_os = "windows")]
    {
        return Ok(Box::new(WindowsAutoEncoder::new()));
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        return Ok(Box::new(ffmpeg::FfmpegEncoder::new()?));
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    Err(MediaError::PlatformNotSupported(
        "No encoder available on this platform".into(),
    ))
}

/// List all hardware-accelerated encoders detected on this system.
pub fn list_available_encoders() -> Vec<EncoderInfo> {
    #[cfg(target_os = "windows")]
    {
        let mut encoders = Vec::new();
        if windows_nvenc::WindowsNvencEncoder::is_available() {
            encoders.push(EncoderInfo {
                name: "native_nvenc_d3d11".to_string(),
                hw_type: HwAccelType::Nvenc,
                supported_codecs: vec![VideoCodec::H264, VideoCodec::H265],
            });
        }
        if windows_amf::WindowsAmfEncoder::is_available() {
            encoders.push(EncoderInfo {
                name: "native_amf_d3d11".to_string(),
                hw_type: HwAccelType::Amf,
                supported_codecs: vec![VideoCodec::H264, VideoCodec::H265, VideoCodec::AV1],
            });
        }
        if windows_qsv::WindowsQsvEncoder::is_available() {
            encoders.push(EncoderInfo {
                name: "native_qsv_d3d11".to_string(),
                hw_type: HwAccelType::Qsv,
                supported_codecs: vec![VideoCodec::H264, VideoCodec::H265, VideoCodec::AV1],
            });
        }
        return encoders;
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        return ffmpeg::FfmpegEncoder::list_available();
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    Vec::new()
}
