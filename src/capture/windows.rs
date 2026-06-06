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
    x_offset: i32,
    y_offset: i32,
    capturing: bool,
    start_time: Option<Instant>,
    frame_count: u64,
    last_frame_data: Option<(Vec<u8>, u32)>, // (data, stride) — cached for timeout reuse
    last_returned_time: Option<Instant>,
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
            x_offset: 0,
            y_offset: 0,
            capturing: false,
            start_time: None,
            frame_count: 0,
            last_frame_data: None,
            last_returned_time: None,
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
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
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
        self.x_offset = rect.left;
        self.y_offset = rect.top;
        self.capturing = true;
        self.start_time = Some(Instant::now());
        self.frame_count = 0;
        self.last_returned_time = None;

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
            duplication.AcquireNextFrame(5, &mut frame_info, &mut desktop_resource)
        };

        match result {
            Ok(()) => {}
            Err(e) if e.code() == DXGI_ERROR_WAIT_TIMEOUT => {
                let timestamp =
                    self.start_time.as_ref().map_or(0, |t| t.elapsed().as_micros() as u64);
                
                // Return a keepalive frame at most once every 500ms when screen is static
                let need_keepalive = self.last_returned_time.map_or(true, |t| t.elapsed() >= std::time::Duration::from_millis(500));

                let (data, stride) = if need_keepalive {
                    if let Some((d, s)) = &self.last_frame_data {
                        self.last_returned_time = Some(std::time::Instant::now());
                        (d.clone(), *s)
                    } else {
                        (Vec::new(), 0)
                    }
                } else {
                    (Vec::new(), 0)
                };

                return Ok(CapturedFrame {
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
        let mut data = unsafe {
            std::slice::from_raw_parts(mapped.pData as *const u8, data_size).to_vec()
        };

        unsafe {
            context.Unmap(staging, 0);
        }

        // Overlay Windows cursor directly on the captured staging texture
        if std::env::var("LUNARIS_HIDE_HOST_CURSOR").is_err() {
            unsafe {
                draw_cursor_onto_buffer(
                    &mut data,
                    self.width,
                    self.height,
                    stride,
                    self.x_offset,
                    self.y_offset,
                );
            }
        }

        let timestamp = self
            .start_time
            .as_ref()
            .map_or(0, |t| t.elapsed().as_micros() as u64);
        self.frame_count += 1;

        // Cache this frame for reuse during DXGI timeouts
        self.last_frame_data = Some((data.clone(), stride));
        self.last_returned_time = Some(std::time::Instant::now());

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

struct GdiMonitorInfo {
    name: String,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    is_primary: bool,
}

fn enumerate_gdi_monitors_with_coords() -> Result<Vec<GdiMonitorInfo>, MediaError> {
    use windows::Win32::Graphics::Gdi::{EnumDisplayMonitors, GetMonitorInfoW, MONITORINFOEXW};
    use windows::Win32::Foundation::{RECT, LPARAM, BOOL};

    unsafe extern "system" fn monitor_enum_proc(
        hmonitor: windows::Win32::Graphics::Gdi::HMONITOR,
        _hdc: windows::Win32::Graphics::Gdi::HDC,
        _rect: *mut RECT,
        lparam: LPARAM,
    ) -> BOOL {
        let monitors = &mut *(lparam.0 as *mut Vec<GdiMonitorInfo>);
        let mut info = MONITORINFOEXW::default();
        info.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
        if GetMonitorInfoW(hmonitor, &mut info as *mut MONITORINFOEXW as *mut _).is_ok() {
            let rect = info.monitorInfo.rcMonitor;
            let x = rect.left;
            let y = rect.top;
            let width = (rect.right - rect.left) as u32;
            let height = (rect.bottom - rect.top) as u32;
            
            let name_chars: Vec<u16> = info.szDevice.iter().take_while(|&&c| c != 0).copied().collect();
            let name = String::from_utf16_lossy(&name_chars);
            
            let is_primary = (info.monitorInfo.dwFlags & 1) != 0; // MONITORINFOF_PRIMARY = 1
            
            monitors.push(GdiMonitorInfo {
                name,
                x,
                y,
                width,
                height,
                is_primary,
            });
        }
        true.into()
    }

    let mut monitors = Vec::new();
    unsafe {
        let lparam = LPARAM(&mut monitors as *mut Vec<GdiMonitorInfo> as isize);
        if !EnumDisplayMonitors(None, None, Some(monitor_enum_proc), lparam).as_bool() {
            return Err(MediaError::CaptureError("EnumDisplayMonitors failed".into()));
        }
    }
    Ok(monitors)
}

pub fn list_gdi_displays() -> Result<Vec<DisplayInfo>, MediaError> {
    let monitors = enumerate_gdi_monitors_with_coords()?;
    let mut displays = Vec::new();
    for (i, m) in monitors.into_iter().enumerate() {
        displays.push(DisplayInfo {
            id: format!("{}", i),
            name: m.name,
            width: m.width,
            height: m.height,
            refresh_rate: 60.0,
            is_primary: m.is_primary,
        });
    }
    Ok(displays)
}

fn capture_gdi(
    x: i32,
    y: i32,
    width: u32,
    height: u32,
) -> Result<(Vec<u8>, u32), MediaError> {
    use windows::Win32::Graphics::Gdi::*;

    unsafe {
        let hdc_screen = GetDC(None);
        if hdc_screen.is_invalid() {
            return Err(MediaError::CaptureError("GetDC(None) failed".into()));
        }

        let hdc_mem = CreateCompatibleDC(hdc_screen);
        if hdc_mem.is_invalid() {
            let _ = ReleaseDC(None, hdc_screen);
            return Err(MediaError::CaptureError("CreateCompatibleDC failed".into()));
        }

        let h_bitmap = CreateCompatibleBitmap(hdc_screen, width as i32, height as i32);
        if h_bitmap.is_invalid() {
            let _ = DeleteDC(hdc_mem);
            let _ = ReleaseDC(None, hdc_screen);
            return Err(MediaError::CaptureError("CreateCompatibleBitmap failed".into()));
        }

        let h_old = SelectObject(hdc_mem, h_bitmap);

        let bitblt_res = BitBlt(
            hdc_mem,
            0,
            0,
            width as i32,
            height as i32,
            hdc_screen,
            x,
            y,
            SRCCOPY,
        );

        if !bitblt_res.as_bool() {
            let _ = SelectObject(hdc_mem, h_old);
            let _ = DeleteObject(h_bitmap);
            let _ = DeleteDC(hdc_mem);
            let _ = ReleaseDC(None, hdc_screen);
            return Err(MediaError::CaptureError("BitBlt failed".into()));
        }

        let mut bmi = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: width as i32,
                biHeight: -(height as i32), // Negative height for top-down DIB
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let stride = width * 4;
        let mut buffer = vec![0u8; (stride * height) as usize];

        let get_di_res = GetDIBits(
            hdc_screen,
            h_bitmap,
            0,
            height,
            Some(buffer.as_mut_ptr() as *mut _),
            &mut bmi,
            DIB_RGB_COLORS,
        );

        // Cleanup
        let _ = SelectObject(hdc_mem, h_old);
        let _ = DeleteObject(h_bitmap);
        let _ = DeleteDC(hdc_mem);
        let _ = ReleaseDC(None, hdc_screen);

        if get_di_res == 0 {
            return Err(MediaError::CaptureError("GetDIBits failed".into()));
        }

        Ok((buffer, stride))
    }
}

pub struct GdiCapture {
    width: u32,
    height: u32,
    x_offset: i32,
    y_offset: i32,
    capturing: bool,
    start_time: Option<Instant>,
    frame_count: u64,
    last_frame_data: Option<(Vec<u8>, u32)>,
    last_returned_time: Option<Instant>,
}

impl GdiCapture {
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            x_offset: 0,
            y_offset: 0,
            capturing: false,
            start_time: None,
            frame_count: 0,
            last_frame_data: None,
            last_returned_time: None,
        }
    }

    pub fn is_capturing(&self) -> bool {
        self.capturing
    }
}

#[async_trait::async_trait]
impl ScreenCapture for GdiCapture {
    async fn list_displays(&self) -> Result<Vec<DisplayInfo>, MediaError> {
        list_gdi_displays()
    }

    async fn start(&mut self, display_id: &str, _config: &StreamConfig) -> Result<(), MediaError> {
        if self.capturing {
            return Err(MediaError::CaptureAlreadyStarted);
        }

        let displays = enumerate_gdi_monitors_with_coords()?;
        let display_index: usize = display_id.parse().unwrap_or(0);
        let display = displays
            .into_iter()
            .nth(display_index)
            .ok_or_else(|| MediaError::CaptureError(format!("Display {} not found", display_id)))?;

        self.width = display.width;
        self.height = display.height;
        self.x_offset = display.x;
        self.y_offset = display.y;
        self.capturing = true;
        self.start_time = Some(Instant::now());
        self.frame_count = 0;
        self.last_returned_time = None;
        self.last_frame_data = None;

        log::info!(
            "GDI screen capture started on display {} ({}x{} at {},{})",
            display_id,
            self.width,
            self.height,
            self.x_offset,
            self.y_offset
        );
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<CapturedFrame, MediaError> {
        if !self.capturing {
            return Err(MediaError::CaptureError("Not capturing".into()));
        }

        let timestamp = self
            .start_time
            .as_ref()
            .map_or(0, |t| t.elapsed().as_micros() as u64);

        let (mut data, stride) = capture_gdi(
            self.x_offset,
            self.y_offset,
            self.width,
            self.height,
        )?;

        // Overlay Windows cursor directly on the captured buffer
        if std::env::var("LUNARIS_HIDE_HOST_CURSOR").is_err() {
            unsafe {
                draw_cursor_onto_buffer(
                    &mut data,
                    self.width,
                    self.height,
                    stride,
                    self.x_offset,
                    self.y_offset,
                );
            }
        }

        self.frame_count += 1;
        self.last_frame_data = Some((data.clone(), stride));
        self.last_returned_time = Some(Instant::now());

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
        self.capturing = false;
        self.last_frame_data = None;
        log::info!(
            "GDI screen capture stopped ({} frames captured)",
            self.frame_count
        );
        Ok(())
    }

    fn is_capturing(&self) -> bool {
        self.capturing
    }
}

fn is_buffer_black(data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }
    for i in (0..data.len()).step_by(256) {
        if data[i] != 0 {
            return false;
        }
    }
    true
}

pub struct WindowsScreenCapture {
    dxgi: Option<DxgiCapture>,
    gdi: Option<GdiCapture>,
    use_gdi_only: bool,
    active_is_gdi: bool,
    display_id: String,
    config: Option<StreamConfig>,
    black_frame_count: u32,
}

impl WindowsScreenCapture {
    pub fn new() -> Result<Self, MediaError> {
        let use_gdi_only = std::env::var("LUNARIS_USE_GDI")
            .map(|val| val == "1" || val.to_lowercase() == "true")
            .unwrap_or(false);

        Ok(Self {
            dxgi: Some(DxgiCapture::new()?),
            gdi: Some(GdiCapture::new()),
            use_gdi_only,
            active_is_gdi: use_gdi_only,
            display_id: String::new(),
            config: None,
            black_frame_count: 0,
        })
    }
}

#[async_trait::async_trait]
impl ScreenCapture for WindowsScreenCapture {
    async fn list_displays(&self) -> Result<Vec<DisplayInfo>, MediaError> {
        if let Some(dxgi) = &self.dxgi {
            if let Ok(displays) = dxgi.list_displays().await {
                return Ok(displays);
            }
        }
        list_gdi_displays()
    }

    async fn start(&mut self, display_id: &str, config: &StreamConfig) -> Result<(), MediaError> {
        self.display_id = display_id.to_string();
        self.config = Some(config.clone());
        self.black_frame_count = 0;

        if self.use_gdi_only {
            log::info!("LUNARIS_USE_GDI is set. Starting directly with GDI capture.");
            self.active_is_gdi = true;
            if let Some(gdi) = &mut self.gdi {
                gdi.start(display_id, config).await?;
            }
            return Ok(());
        }

        log::info!("Starting screen capture with DXGI backend...");
        if let Some(dxgi) = &mut self.dxgi {
            match dxgi.start(display_id, config).await {
                Ok(()) => {
                    self.active_is_gdi = false;
                    return Ok(());
                }
                Err(e) => {
                    log::warn!("Failed to start DXGI capture: {}. Falling back to GDI capture.", e);
                }
            }
        }

        self.active_is_gdi = true;
        if let Some(gdi) = &mut self.gdi {
            gdi.start(display_id, config).await?;
        }
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<CapturedFrame, MediaError> {
        if self.active_is_gdi {
            if let Some(gdi) = &mut self.gdi {
                return gdi.next_frame().await;
            }
            return Err(MediaError::CaptureError("GDI capture backend not initialized".into()));
        }

        let dxgi = self
            .dxgi
            .as_mut()
            .ok_or_else(|| MediaError::CaptureError("DXGI capture backend not initialized".into()))?;

        let frame = dxgi.next_frame().await?;

        if frame.is_new_frame {
            if let GpuBuffer::CpuBuffer { data, .. } = &frame.buffer {
                if is_buffer_black(data) {
                    self.black_frame_count += 1;
                    if self.black_frame_count >= 3 {
                        log::warn!(
                            "DXGI capture returned black frames for 3 consecutive frames. \
                             Possibly in an RDP session, VM, or hybrid GPU environment. \
                             Switching capture backend to GDI."
                        );
                        let _ = dxgi.stop().await;

                        if let Some(gdi) = &mut self.gdi {
                            if let Some(config) = &self.config {
                                match gdi.start(&self.display_id, config).await {
                                    Ok(()) => {
                                        self.active_is_gdi = true;
                                        return gdi.next_frame().await;
                                    }
                                    Err(e) => {
                                        log::error!("Failed to start GDI capture fallback: {:?}", e);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    self.black_frame_count = 0;
                }
            }
        }

        Ok(frame)
    }

    async fn stop(&mut self) -> Result<(), MediaError> {
        if self.active_is_gdi {
            if let Some(gdi) = &mut self.gdi {
                gdi.stop().await?;
            }
        } else {
            if let Some(dxgi) = &mut self.dxgi {
                dxgi.stop().await?;
            }
        }
        Ok(())
    }

    fn is_capturing(&self) -> bool {
        if self.active_is_gdi {
            self.gdi.as_ref().map_or(false, |gdi| gdi.is_capturing())
        } else {
            self.dxgi.as_ref().map_or(false, |dxgi| dxgi.is_capturing())
        }
    }
}

unsafe fn draw_cursor_onto_buffer(
    frame_buffer: &mut [u8],
    frame_width: u32,
    frame_height: u32,
    frame_stride: u32,
    x_offset: i32,
    y_offset: i32,
) {
    use windows::Win32::UI::WindowsAndMessaging::{
        GetCursorInfo, GetIconInfo, CURSORINFO, CURSOR_SHOWING, ICONINFO,
    };
    use windows::Win32::Graphics::Gdi::{
        GetDIBits, GetDC, ReleaseDC, DeleteObject, BITMAPINFO, BITMAPINFOHEADER,
        DIB_RGB_COLORS, HGDIOBJ, BITMAP, GetObjectW, BI_RGB,
    };

    let mut cursor_info = CURSORINFO {
        cbSize: std::mem::size_of::<CURSORINFO>() as u32,
        ..Default::default()
    };

    if GetCursorInfo(&mut cursor_info).is_ok() {
        if (cursor_info.flags.0 & CURSOR_SHOWING.0) != 0 {
            let h_cursor = cursor_info.hCursor;
            let mut icon_info = ICONINFO::default();
            if GetIconInfo(h_cursor, &mut icon_info).is_ok() {
                let x_hotspot = icon_info.xHotspot as i32;
                let y_hotspot = icon_info.yHotspot as i32;
                let mut has_color = false;

                let h_bmp = if !icon_info.hbmColor.is_invalid() {
                    has_color = true;
                    icon_info.hbmColor
                } else {
                    icon_info.hbmMask
                };

                let mut bmp = BITMAP::default();
                let hgdiobj = HGDIOBJ(h_bmp.0);
                let get_obj_res = GetObjectW(
                    hgdiobj,
                    std::mem::size_of::<BITMAP>() as i32,
                    Some(&mut bmp as *mut BITMAP as *mut _),
                );

                if get_obj_res > 0 {
                    let width = bmp.bmWidth;
                    let height = if has_color { bmp.bmHeight } else { bmp.bmHeight / 2 };

                    let mut color_buffer = vec![0u8; (width * height * 4) as usize];
                    let mut mask_buffer = vec![0u8; (width * height * 2 * 4) as usize];

                    let hdc = GetDC(None);
                    let mut bmi = BITMAPINFO {
                        bmiHeader: BITMAPINFOHEADER {
                            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                            biWidth: width,
                            biHeight: if has_color { -height } else { -height * 2 },
                            biPlanes: 1,
                            biBitCount: 32,
                            biCompression: BI_RGB.0,
                            ..Default::default()
                        },
                        ..Default::default()
                    };

                    if has_color {
                        let _ = GetDIBits(
                            hdc,
                            icon_info.hbmColor,
                            0,
                            height as u32,
                            Some(color_buffer.as_mut_ptr() as *mut _),
                            &mut bmi,
                            DIB_RGB_COLORS,
                        );

                        let mut has_alpha = false;
                        for i in 0..color_buffer.len() / 4 {
                            if color_buffer[i * 4 + 3] != 0 {
                                has_alpha = true;
                                break;
                            }
                        }

                        if !has_alpha && !icon_info.hbmMask.is_invalid() {
                            let mut bmi_mask = BITMAPINFO {
                                bmiHeader: BITMAPINFOHEADER {
                                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                                    biWidth: width,
                                    biHeight: -height * 2,
                                    biPlanes: 1,
                                    biBitCount: 32,
                                    biCompression: BI_RGB.0,
                                    ..Default::default()
                                },
                                ..Default::default()
                            };
                            let _ = GetDIBits(
                                hdc,
                                icon_info.hbmMask,
                                0,
                                (height * 2) as u32,
                                Some(mask_buffer.as_mut_ptr() as *mut _),
                                &mut bmi_mask,
                                DIB_RGB_COLORS,
                            );

                            for i in 0..(width * height) as usize {
                                let mask_val = mask_buffer[i * 4];
                                color_buffer[i * 4 + 3] = if mask_val == 0 { 255 } else { 0 };
                            }
                        }
                    } else {
                        let _ = GetDIBits(
                            hdc,
                            icon_info.hbmMask,
                            0,
                            (height * 2) as u32,
                            Some(mask_buffer.as_mut_ptr() as *mut _),
                            &mut bmi,
                            DIB_RGB_COLORS,
                        );

                        for i in 0..(width * height) as usize {
                            let and_val = mask_buffer[i * 4];
                            let xor_val = mask_buffer[(i + (width * height) as usize) * 4];

                            if and_val == 0 {
                                color_buffer[i * 4] = xor_val;
                                color_buffer[i * 4 + 1] = xor_val;
                                color_buffer[i * 4 + 2] = xor_val;
                                color_buffer[i * 4 + 3] = 255;
                            } else {
                                color_buffer[i * 4 + 3] = 0;
                            }
                        }
                    }

                    let _ = ReleaseDC(None, hdc);

                    // Draw onto frame_buffer
                    let cursor_x = cursor_info.ptScreenPos.x - x_offset;
                    let cursor_y = cursor_info.ptScreenPos.y - y_offset;

                    let x_start = cursor_x - x_hotspot;
                    let y_start = cursor_y - y_hotspot;

                    let cursor_stride = width as usize * 4;
                    for cy in 0..height as usize {
                        let sy = y_start + cy as i32;
                        if sy < 0 || sy >= frame_height as i32 {
                            continue;
                        }

                        for cx in 0..width as usize {
                            let sx = x_start + cx as i32;
                            if sx < 0 || sx >= frame_width as i32 {
                                continue;
                            }

                            let c_idx = cy * cursor_stride + cx * 4;
                            let alpha = color_buffer[c_idx + 3];
                            if alpha == 0 {
                                continue;
                            }

                            let blue = color_buffer[c_idx];
                            let green = color_buffer[c_idx + 1];
                            let red = color_buffer[c_idx + 2];

                            let bgra_idx = sy as usize * frame_stride as usize + sx as usize * 4;
                            if bgra_idx + 3 < frame_buffer.len() {
                                if alpha == 255 {
                                    frame_buffer[bgra_idx] = blue;
                                    frame_buffer[bgra_idx + 1] = green;
                                    frame_buffer[bgra_idx + 2] = red;
                                } else {
                                    let dst_b = frame_buffer[bgra_idx] as u32;
                                    let dst_g = frame_buffer[bgra_idx + 1] as u32;
                                    let dst_r = frame_buffer[bgra_idx + 2] as u32;

                                    frame_buffer[bgra_idx] = ((blue as u32 * alpha as u32 + dst_b * (255 - alpha as u32)) / 255) as u8;
                                    frame_buffer[bgra_idx + 1] = ((green as u32 * alpha as u32 + dst_g * (255 - alpha as u32)) / 255) as u8;
                                    frame_buffer[bgra_idx + 2] = ((red as u32 * alpha as u32 + dst_r * (255 - alpha as u32)) / 255) as u8;
                                }
                            }
                        }
                    }
                }

                if !icon_info.hbmMask.is_invalid() {
                    let _ = DeleteObject(icon_info.hbmMask);
                }
                if !icon_info.hbmColor.is_invalid() {
                    let _ = DeleteObject(icon_info.hbmColor);
                }
            }
        }
    }
}
