//! Windows screen capture using DXGI Desktop Duplication API.
//!
//! Uses Direct3D 11 and DXGI 1.2+ to capture the desktop with minimal CPU overhead.
//! Frames are captured as BGRA textures and copied to a staging buffer for CPU readback.
//!
//! ## Capture Flow
//!
//! ```text
//!   DXGI OutputDuplication → AcquireNextFrame
//!     → ID3D11Texture2D (GPU) → CopyResource to staging
//!     → Map staging → BGRA bytes → CpuBuffer
//! ```

use std::time::Instant;

use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D11::*;
use windows::core::Interface;

use crate::capture::{CapturedFrame, ScreenCapture};
use crate::capture::gpu_buffer::GpuBuffer;
use crate::error::MediaError;
use crate::types::*;

pub struct DxgiCapture {
    device: Option<ID3D11Device>,
    context: Option<ID3D11DeviceContext>,
    duplication: Option<IDXGIOutputDuplication>,
    staging_texture: Option<ID3D11Texture2D>,
    width: u32,
    height: u32,
    capturing: bool,
    start_time: Option<Instant>,
    frame_count: u64,
}

impl DxgiCapture {
    pub fn new() -> Result<Self, MediaError> {
        Ok(Self {
            device: None,
            context: None,
            duplication: None,
            staging_texture: None,
            width: 0,
            height: 0,
            capturing: false,
            start_time: None,
            frame_count: 0,
        })
    }

    fn create_d3d11_device() -> Result<(ID3D11Device, ID3D11DeviceContext), MediaError> {
        let mut device = None;
        let mut context = None;

        let feature_levels = [D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1];

        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                None,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&feature_levels),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
            .map_err(|e| MediaError::CaptureError(format!("D3D11CreateDevice failed: {e}")))?;
        }

        Ok((
            device.ok_or_else(|| MediaError::CaptureError("D3D11 device is None".into()))?,
            context.ok_or_else(|| MediaError::CaptureError("D3D11 context is None".into()))?,
        ))
    }

    fn enumerate_outputs(
        device: &ID3D11Device,
    ) -> Result<Vec<(IDXGIOutput1, DXGI_OUTPUT_DESC)>, MediaError> {
        let dxgi_device: IDXGIDevice = device
            .cast()
            .map_err(|e| MediaError::CaptureError(format!("Cast to IDXGIDevice failed: {e}")))?;

        let adapter: IDXGIAdapter = unsafe { dxgi_device.GetAdapter() }
            .map_err(|e| MediaError::CaptureError(format!("GetAdapter failed: {e}")))?;

        let mut outputs = Vec::new();
        let mut i = 0u32;
        loop {
            match unsafe { adapter.EnumOutputs(i) } {
                Ok(output) => {
                    let desc = unsafe { output.GetDesc() }
                        .map_err(|e| MediaError::CaptureError(format!("GetDesc failed: {e}")))?;

                    if let Ok(output1) = output.cast::<IDXGIOutput1>() {
                        outputs.push((output1, desc));
                    }
                    i += 1;
                }
                Err(_) => break,
            }
        }

        if outputs.is_empty() {
            return Err(MediaError::CaptureError("No DXGI outputs found".into()));
        }

        Ok(outputs)
    }

    fn create_staging_texture(
        device: &ID3D11Device,
        width: u32,
        height: u32,
    ) -> Result<ID3D11Texture2D, MediaError> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_STAGING,
            BindFlags: D3D11_BIND_FLAG(0),
            CPUAccessFlags: D3D11_CPU_ACCESS_READ,
            MiscFlags: D3D11_RESOURCE_MISC_FLAG(0),
        };

        let mut texture = None;
        unsafe {
            device
                .CreateTexture2D(&desc, None, Some(&mut texture))
                .map_err(|e| {
                    MediaError::CaptureError(format!("CreateTexture2D staging failed: {e}"))
                })?;
        }

        texture.ok_or_else(|| MediaError::CaptureError("Staging texture is None".into()))
    }
}

#[async_trait::async_trait]
impl ScreenCapture for DxgiCapture {
    async fn list_displays(&self) -> Result<Vec<DisplayInfo>, MediaError> {
        let (device, _context) = Self::create_d3d11_device()?;
        let outputs = Self::enumerate_outputs(&device)?;

        let mut displays = Vec::new();
        for (i, (_output, desc)) in outputs.iter().enumerate() {
            let rect = desc.DesktopCoordinates;
            let width = (rect.right - rect.left) as u32;
            let height = (rect.bottom - rect.top) as u32;

            let name_chars: Vec<u16> = desc.DeviceName.iter().take_while(|&&c| c != 0).copied().collect();
            let name = String::from_utf16_lossy(&name_chars);

            displays.push(DisplayInfo {
                id: format!("{}", i),
                name,
                width,
                height,
                refresh_rate: 60.0,
                is_primary: i == 0,
            });
        }

        Ok(displays)
    }

    async fn start(&mut self, display_id: &str, config: &StreamConfig) -> Result<(), MediaError> {
        if self.capturing {
            return Err(MediaError::CaptureAlreadyStarted);
        }

        let (device, context) = Self::create_d3d11_device()?;
        let outputs = Self::enumerate_outputs(&device)?;

        let output_index: usize = display_id.parse().unwrap_or(0);
        let (output, desc) = outputs
            .into_iter()
            .nth(output_index)
            .ok_or_else(|| MediaError::CaptureError(format!("Display {} not found", display_id)))?;

        let rect = desc.DesktopCoordinates;
        self.width = (rect.right - rect.left) as u32;
        self.height = (rect.bottom - rect.top) as u32;

        log::info!(
            "DXGI: Starting capture on output {} ({}x{})",
            output_index,
            self.width,
            self.height
        );

        let duplication = unsafe { output.DuplicateOutput(&device) }.map_err(|e| {
            MediaError::CaptureError(format!(
                "DuplicateOutput failed: {e}. Ensure no other app is using Desktop Duplication."
            ))
        })?;

        let staging = Self::create_staging_texture(&device, self.width, self.height)?;

        self.device = Some(device);
        self.context = Some(context);
        self.duplication = Some(duplication);
        self.staging_texture = Some(staging);
        self.capturing = true;
        self.start_time = Some(Instant::now());
        self.frame_count = 0;

        log::info!("DXGI Desktop Duplication capture started");
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<CapturedFrame, MediaError> {
        let duplication = self
            .duplication
            .as_ref()
            .ok_or_else(|| MediaError::CaptureError("Not capturing".into()))?;
        let context = self.context.as_ref().unwrap();
        let staging = self.staging_texture.as_ref().unwrap();

        let mut frame_info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut desktop_resource = None;

        let result = unsafe {
            duplication.AcquireNextFrame(100, &mut frame_info, &mut desktop_resource)
        };

        match result {
            Ok(()) => {}
            Err(e) if e.code() == DXGI_ERROR_WAIT_TIMEOUT => {
                let timestamp =
                    self.start_time.as_ref().map_or(0, |t| t.elapsed().as_micros() as u64);
                return Ok(CapturedFrame {
                    buffer: GpuBuffer::CpuBuffer {
                        data: Vec::new(),
                        stride: 0,
                        format: PixelFormat::BGRA,
                        width: self.width,
                        height: self.height,
                    },
                    timestamp_us: timestamp,
                    width: self.width,
                    height: self.height,
                    format: PixelFormat::BGRA,
                    is_new_frame: false,
                });
            }
            Err(e) => {
                return Err(MediaError::CaptureError(format!(
                    "AcquireNextFrame failed: {e}"
                )));
            }
        }

        let resource = desktop_resource
            .ok_or_else(|| MediaError::CaptureError("Desktop resource is None".into()))?;

        let texture: ID3D11Texture2D = resource
            .cast()
            .map_err(|e| MediaError::CaptureError(format!("Cast to Texture2D failed: {e}")))?;

        unsafe {
            context.CopyResource(staging, &texture);
        }

        unsafe {
            let _ = duplication.ReleaseFrame();
        }

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            context
                .Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(|e| MediaError::CaptureError(format!("Map failed: {e}")))?;
        }

        let stride = mapped.RowPitch;
        let data_size = (stride * self.height) as usize;
        let data = unsafe {
            std::slice::from_raw_parts(mapped.pData as *const u8, data_size).to_vec()
        };

        unsafe {
            context.Unmap(staging, 0);
        }

        let timestamp = self
            .start_time
            .as_ref()
            .map_or(0, |t| t.elapsed().as_micros() as u64);
        self.frame_count += 1;

        Ok(CapturedFrame {
            buffer: GpuBuffer::CpuBuffer {
                data,
                stride,
                format: PixelFormat::BGRA,
                width: self.width,
                height: self.height,
            },
            timestamp_us: timestamp,
            width: self.width,
            height: self.height,
            format: PixelFormat::BGRA,
            is_new_frame: true,
        })
    }

    async fn stop(&mut self) -> Result<(), MediaError> {
        self.duplication = None;
        self.staging_texture = None;
        self.context = None;
        self.device = None;
        self.capturing = false;
        log::info!(
            "DXGI Desktop Duplication capture stopped ({} frames captured)",
            self.frame_count
        );
        Ok(())
    }

    fn is_capturing(&self) -> bool {
        self.capturing
    }
}
