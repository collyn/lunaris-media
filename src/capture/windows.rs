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

use std::time::{Duration, Instant};

use windows::Win32::Foundation::{GENERIC_ALL, WAIT_ABANDONED};
use windows::Win32::System::Power::{SetThreadExecutionState, ES_CONTINUOUS, ES_DISPLAY_REQUIRED};
use windows::Win32::System::StationsAndDesktops::{
    CloseDesktop, OpenInputDesktop, SetThreadDesktop, DESKTOP_ACCESS_FLAGS,
    DF_ALLOWOTHERACCOUNTHOOK,
};
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D11::*;
use windows::core::Interface;

use crate::capture::{CapturedFrame, ScreenCapture};
use crate::capture::gpu_buffer::GpuBuffer;
use crate::error::MediaError;
use crate::types::*;

/// Port of Sunshine `syncThreadDesktop()` (src/platform/windows/misc.cpp:247-264):
/// attach the CALLING thread to the desktop that currently receives input, so
/// that `DuplicateOutput` is not refused (E_ACCESSDENIED) after a desktop
/// switch (lock screen, UAC prompt, sleeping monitor).
///
/// Adaptation: it acts on the calling thread only. Lunaris runs on tokio worker
/// threads and a task may move to another thread at any `.await`, so every
/// caller must invoke this in the same synchronous stretch of code as the
/// `DuplicateOutput` it protects, with no `.await` in between.
fn sync_thread_desktop() {
    unsafe {
        let desk = match OpenInputDesktop(
            DF_ALLOWOTHERACCOUNTHOOK,
            false,
            DESKTOP_ACCESS_FLAGS(GENERIC_ALL.0),
        ) {
            Ok(d) => d,
            Err(e) => {
                log::error!("Failed to Open Input Desktop [{e}]");
                return;
            }
        };
        if let Err(e) = SetThreadDesktop(desk) {
            log::error!("Failed to sync desktop to thread [{e}]");
        }
        let _ = CloseDesktop(desk);
    }
}

/// Keeps the display awake while capturing.
///
/// Port of Sunshine `display_base_t::capture` (display_base.cpp:254-261):
/// `SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED)` for the whole
/// capture, `SetThreadExecutionState(ES_CONTINUOUS)` on exit.
///
/// Adaptation: the execution state belongs to the calling thread. Sunshine's
/// capture runs on one dedicated thread, Lunaris's `next_frame` on whatever
/// tokio worker polls it, so the state is held by this small dedicated thread
/// instead; it lives until the object is dropped (stop / Drop).
struct KeepAwake {
    stop_tx: Option<std::sync::mpsc::Sender<()>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl KeepAwake {
    fn start() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let handle = std::thread::Builder::new()
            .name("lunaris-keep-display-awake".into())
            .spawn(move || {
                unsafe {
                    SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED);
                }
                // Blocks until the sender is dropped or a message is sent.
                let _ = rx.recv();
                unsafe {
                    SetThreadExecutionState(ES_CONTINUOUS);
                }
            })
            .ok();
        if handle.is_none() {
            log::warn!("Could not start the keep-display-awake thread");
        }
        Self { stop_tx: Some(tx), handle }
    }
}

impl Drop for KeepAwake {
    fn drop(&mut self) {
        self.stop_tx.take(); // wakes the thread
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Result of `AcquireNextFrame`, mapped as Sunshine's `duplication_t::next_frame`
/// (display_base.cpp:158-182) maps it to `capture_e`.
enum AcquireStatus {
    Ok,
    Timeout,
    Reinit,
    Error,
}

fn classify_acquire(code: windows::core::HRESULT) -> AcquireStatus {
    if code.is_ok() {
        AcquireStatus::Ok
    } else if code == DXGI_ERROR_WAIT_TIMEOUT {
        AcquireStatus::Timeout
    } else if code.0 == WAIT_ABANDONED.0 as i32
        || code == DXGI_ERROR_ACCESS_LOST
        || code == DXGI_ERROR_ACCESS_DENIED
    {
        AcquireStatus::Reinit
    } else {
        AcquireStatus::Error
    }
}

/// Port of `duplication_t::release_frame` (display_base.cpp:195-218).
/// Ok for S_OK and DXGI_ERROR_INVALID_CALL ("already released"), Reinit for
/// DXGI_ERROR_ACCESS_LOST, Error otherwise.
fn release_frame(duplication: &IDXGIOutputDuplication) -> AcquireStatus {
    match unsafe { duplication.ReleaseFrame() } {
        Ok(()) => AcquireStatus::Ok,
        Err(e) if e.code() == DXGI_ERROR_INVALID_CALL => {
            log::warn!("Duplication frame already released");
            AcquireStatus::Ok
        }
        Err(e) if e.code() == DXGI_ERROR_ACCESS_LOST => AcquireStatus::Reinit,
        Err(e) => {
            log::error!("Error while releasing duplication frame [{e}]");
            AcquireStatus::Error
        }
    }
}

/// A duplication together with what was found while creating it.
struct Duplicated {
    factory: IDXGIFactory1,
    duplication: IDXGIOutputDuplication,
    /// `DXGI_OUTPUT_DESC::DesktopCoordinates` (the whole desc holds an HMONITOR
    /// raw pointer, which would make the async futures non-`Send`).
    rect: windows::Win32::Foundation::RECT,
    index: usize,
    /// `IDXGIOutputDuplication::GetDesc().ModeDesc` size = size of the textures
    /// `AcquireNextFrame` delivers.
    tex_width: u32,
    tex_height: u32,
}

pub struct DxgiCapture {
    keep_awake: Option<KeepAwake>,
    factory: Option<IDXGIFactory1>,
    tex_width: u32,
    tex_height: u32,
    device: Option<ID3D11Device>,
    context: Option<ID3D11DeviceContext>,
    duplication: Option<IDXGIOutputDuplication>,
    staging_texture: Option<ID3D11Texture2D>,
    gpu_texture: Option<ID3D11Texture2D>,
    width: u32,
    height: u32,
    x_offset: i32,
    y_offset: i32,
    capturing: bool,
    start_time: Option<Instant>,
    frame_count: u64,
    last_frame_data: Option<(Vec<u8>, u32)>, // (data, stride) — cached for timeout reuse
    last_returned_time: Option<Instant>,
    use_zero_copy: bool,
}

impl DxgiCapture {
    pub fn new() -> Result<Self, MediaError> {
        Ok(Self {
            keep_awake: None,
            factory: None,
            tex_width: 0,
            tex_height: 0,
            device: None,
            context: None,
            duplication: None,
            staging_texture: None,
            gpu_texture: None,
            width: 0,
            height: 0,
            x_offset: 0,
            y_offset: 0,
            capturing: false,
            start_time: None,
            frame_count: 0,
            last_frame_data: None,
            last_returned_time: None,
            use_zero_copy: false,
        })
    }

    pub(crate) fn create_d3d11_device() -> Result<(ID3D11Device, ID3D11DeviceContext), MediaError> {
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

    fn create_gpu_texture(
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
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };

        let mut texture = None;
        unsafe {
            device
                .CreateTexture2D(&desc, None, Some(&mut texture))
                .map_err(|e| {
                    MediaError::CaptureError(format!("CreateTexture2D gpu failed: {e}"))
                })?;
        }

        texture.ok_or_else(|| MediaError::CaptureError("GPU texture is None".into()))
    }

    /// One pass of Sunshine's output search (display_base.cpp:~520-557): pick
    /// the wanted output, require `AttachedToDesktop`, and make sure Desktop
    /// Duplication works on it.
    ///
    /// Sunshine runs `test_dxgi_duplication` (display_base.cpp:~395-445: two
    /// `DuplicateOutput` tries, `syncThreadDesktop()` before each, 200 ms
    /// sleep after each failure) as part of the search condition and then
    /// duplicates again in `duplication_t::init` (display_base.cpp:111-122,
    /// same two tries). Adaptation: Lunaris has one device and no separate
    /// test device, so the two tries (also `syncThreadDesktop()` before each
    /// and 200 ms after each failure) produce the duplication that is kept.
    /// A failed duplication makes the pass fail, as in Sunshine, where a failed
    /// test means the output is not found.
    ///
    /// Each `DuplicateOutput` is immediately preceded by `sync_thread_desktop()`
    /// with no `.await` in between (the sleep comes after a failed try, and the
    /// loop calls `sync_thread_desktop()` again afterwards).
    async fn output_pass(
        device: &ID3D11Device,
        id: &str,
        factory: &IDXGIFactory1,
        last_err: &mut String,
    ) -> Option<Duplicated> {
        // Synchronous helper: DXGI_OUTPUT_DESC holds an HMONITOR raw pointer and
        // must not live across an `.await` (the futures must stay `Send`).
        let (output, rect, index) = Self::select_output(device, id, last_err)?;

        for _ in 0..2 {
            // Ensure we can duplicate the current display (display_base.cpp:115)
            sync_thread_desktop();
            match unsafe { output.DuplicateOutput(device) } {
                Ok(duplication) => {
                    let dup_desc = unsafe { duplication.GetDesc() };
                    return Some(Duplicated {
                        factory: factory.clone(),
                        duplication,
                        rect,
                        index,
                        tex_width: dup_desc.ModeDesc.Width,
                        tex_height: dup_desc.ModeDesc.Height,
                    });
                }
                Err(e) => {
                    log::warn!("DuplicateOutput failed [{e}]");
                    *last_err = format!(
                        "DuplicateOutput failed: {e}. Ensure no other app is using Desktop Duplication."
                    );
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }
        None
    }

    /// Output choice of one search pass (see `output_pass`).
    fn select_output(
        device: &ID3D11Device,
        id: &str,
        last_err: &mut String,
    ) -> Option<(IDXGIOutput1, windows::Win32::Foundation::RECT, usize)> {
        let outputs = match Self::enumerate_outputs(device) {
            Ok(o) => o,
            Err(e) => {
                *last_err = e.to_string();
                return None;
            }
        };

        // "default" (no explicit selection) must map to the PRIMARY output, not
        // output index 0 — DXGI EnumOutputs order is not guaranteed to be
        // primary-first, which left the default stream capturing a blank/wrong
        // monitor. The primary output's desktop rect is always at the origin.
        let index: usize = if id == "default" || id.is_empty() {
            outputs
                .iter()
                .position(|(_, d)| d.DesktopCoordinates.left == 0 && d.DesktopCoordinates.top == 0)
                .unwrap_or(0)
        } else {
            id.parse().unwrap_or(0)
        };

        let (output, desc) = match outputs.into_iter().nth(index) {
            Some(o) => o,
            None => {
                *last_err = format!("Display {index} not found");
                return None;
            }
        };

        // display_base.cpp:541 `desc.AttachedToDesktop && test_dxgi_duplication(...)`
        if !desc.AttachedToDesktop.as_bool() {
            *last_err = format!("Display {index} is not attached to the desktop");
            return None;
        }

        let rect = desc.DesktopCoordinates;
        Some((output, rect, index))
    }

    /// Create the DXGI factory and find + duplicate the wanted output.
    ///
    /// Port of `display_base_t::init` (display_base.cpp:~505-567): the factory
    /// is created first (kept for `IsCurrent`, display_base.cpp:267), the
    /// output search runs at most twice, and when the first pass finds nothing
    /// `SetThreadExecutionState(ES_DISPLAY_REQUIRED); Sleep(500);` tries to
    /// power the display on before the second pass (lines 563-567).
    /// Adaptation: `tokio::time::sleep` instead of `Sleep` (this is async code).
    async fn find_and_duplicate(
        device: &ID3D11Device,
        id: &str,
    ) -> Result<Duplicated, MediaError> {
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }
            .map_err(|e| MediaError::CaptureError(format!("Failed to create DXGIFactory1: {e}")))?;

        let mut last_err = String::from("no output found");
        for tries in 0..2 {
            if let Some(found) = Self::output_pass(device, id, &factory, &mut last_err).await {
                return Ok(found);
            }
            // If we made it here without finding an output, try to power on the
            // display and retry.
            if tries == 0 {
                unsafe {
                    SetThreadExecutionState(ES_DISPLAY_REQUIRED);
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
        Err(MediaError::CaptureError(format!(
            "Failed to locate an output device: {last_err}"
        )))
    }

    /// What the caller already understands as "no new frame yet": the frame
    /// `DXGI_ERROR_WAIT_TIMEOUT` produces (`is_new_frame: false`).
    fn timeout_frame(&mut self) -> CapturedFrame {
        let timestamp = self.start_time.as_ref().map_or(0, |t| t.elapsed().as_micros() as u64);

        // Return a keepalive frame at most once every 500ms when screen is static
        let need_keepalive = self
            .last_returned_time
            .map_or(true, |t| t.elapsed() >= std::time::Duration::from_millis(500));

        if self.use_zero_copy {
            if need_keepalive && self.gpu_texture.is_some() {
                self.last_returned_time = Some(std::time::Instant::now());
                return CapturedFrame {
                    buffer: GpuBuffer::D3D11Texture {
                        texture: self.gpu_texture.as_ref().unwrap().as_raw(),
                        array_index: 0,
                    },
                    timestamp_us: timestamp,
                    width: self.width,
                    height: self.height,
                    format: PixelFormat::BGRA,
                    is_new_frame: false,
                    cursor: None,
                };
            } else {
                return CapturedFrame {
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
                    cursor: None,
                };
            }
        }

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

        CapturedFrame {
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
            cursor: None,
        }
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
                is_primary: rect.left == 0 && rect.top == 0,
            });
        }

        Ok(displays)
    }

    async fn start(&mut self, display_id: &str, _config: &StreamConfig) -> Result<(), MediaError> {
        if self.capturing {
            return Err(MediaError::CaptureAlreadyStarted);
        }

        // Started BEFORE the duplication attempts so that a sleeping monitor is
        // being woken while they run. Dropped (execution state released) if
        // start fails. See `KeepAwake` (display_base.cpp:254-261).
        let keep_awake = KeepAwake::start();

        let (device, context) = Self::create_d3d11_device()?;

        // Output search, `DuplicateOutput` x2 with `syncThreadDesktop()`, and the
        // wake-and-retry pass: see `find_and_duplicate`.
        let found = Self::find_and_duplicate(&device, display_id).await?;

        let output_index = found.index;
        let rect = found.rect;
        self.width = (rect.right - rect.left) as u32;
        self.height = (rect.bottom - rect.top) as u32;

        log::info!(
            "DXGI: Starting capture on output {} ({}x{})",
            output_index,
            self.width,
            self.height
        );

        let duplication = found.duplication;

        let staging = Self::create_staging_texture(&device, self.width, self.height)?;

        let use_zero_copy = std::env::var("LUNARIS_ZERO_COPY")
            .map(|val| val == "1" || val.to_lowercase() == "true")
            .unwrap_or(false);

        let gpu_texture = if use_zero_copy {
            log::info!("DXGI: Creating GPU-only texture for Zero-Copy mode");
            Some(Self::create_gpu_texture(&device, self.width, self.height)?)
        } else {
            None
        };

        self.keep_awake = Some(keep_awake);
        self.factory = Some(found.factory);
        self.tex_width = found.tex_width;
        self.tex_height = found.tex_height;
        self.device = Some(device);
        self.context = Some(context);
        self.duplication = Some(duplication);
        self.staging_texture = Some(staging);
        self.gpu_texture = gpu_texture;
        self.use_zero_copy = use_zero_copy;
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
        if !self.capturing {
            return Err(MediaError::CaptureError("Not capturing".into()));
        }

        // display_base.cpp:264-271: "This will return false if the HDR state
        // changes or for any number of other display or GPU changes. We should
        // reinit to examine the updated state of the display subsystem."
        if let Some(factory) = &self.factory {
            if !unsafe { factory.IsCurrent() }.as_bool() {
                log::info!("DXGI factory is no longer current: capture has to be re-initialised");
                return Err(MediaError::CaptureReinit);
            }
        }

        // Cloned (cheap COM AddRef) so that `self` stays free for `timeout_frame()`
        // below.
        let duplication = self
            .duplication
            .clone()
            .ok_or_else(|| MediaError::CaptureError("Not capturing".into()))?;
        let context = self.context.clone().unwrap();
        let staging = self.staging_texture.clone().unwrap();

        let mut frame_info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut desktop_resource = None;

        let result = unsafe {
            duplication.AcquireNextFrame(5, &mut frame_info, &mut desktop_resource)
        };

        // Mapping of duplication_t::next_frame (display_base.cpp:158-182):
        //   S_OK -> ok; DXGI_ERROR_WAIT_TIMEOUT -> timeout;
        //   WAIT_ABANDONED / DXGI_ERROR_ACCESS_LOST / DXGI_ERROR_ACCESS_DENIED -> reinit
        //   (`MediaError::CaptureReinit`, handled by the pipeline);
        //   anything else -> capture_error.
        let code = match &result {
            Ok(()) => windows::core::HRESULT(0),
            Err(e) => e.code(),
        };
        match classify_acquire(code) {
            AcquireStatus::Ok => {}
            AcquireStatus::Timeout => return Ok(self.timeout_frame()),
            AcquireStatus::Reinit => {
                log::warn!(
                    "AcquireNextFrame returned {code:?}: capture has to be re-initialised (display_base.cpp:175)"
                );
                return Err(MediaError::CaptureReinit);
            }
            AcquireStatus::Error => {
                // Sunshine: "Couldn't acquire next frame" -> capture_e::capture_error
                return Err(MediaError::CaptureError(format!(
                    "AcquireNextFrame failed: {code:?}"
                )));
            }
        }

        let resource = desktop_resource
            .ok_or_else(|| MediaError::CaptureError("Desktop resource is None".into()))?;

        let texture: ID3D11Texture2D = resource
            .cast()
            .map_err(|e| MediaError::CaptureError(format!("Cast to Texture2D failed: {e}")))?;

        // display_vram.cpp:1430-1438: "It's possible for our display enumeration
        // to race with mode changes and result in mismatched image pool and
        // desktop texture sizes. If this happens, just reinit again."
        // (capture_e::reinit -> MediaError::CaptureReinit)
        // Adaptation: the expected size is the one `IDXGIOutputDuplication::GetDesc`
        // reported (`tex_width/tex_height`), not Sunshine's width_before_rotation:
        // it is the true pixel size of the delivered texture whatever the
        // rotation or the process's DPI awareness (DesktopCoordinates can be
        // DPI-virtualised, which would make this check fire on every frame).
        let mut tex_desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { texture.GetDesc(&mut tex_desc) };
        if tex_desc.Width != self.tex_width || tex_desc.Height != self.tex_height {
            log::info!(
                "Capture size changed [{}x{} -> {}x{}]",
                self.tex_width,
                self.tex_height,
                tex_desc.Width,
                tex_desc.Height
            );
            drop(texture);
            drop(resource);
            let _ = release_frame(&duplication);
            return Err(MediaError::CaptureReinit);
        }

        let timestamp = self
            .start_time
            .as_ref()
            .map_or(0, |t| t.elapsed().as_micros() as u64);
        self.frame_count += 1;

        if self.use_zero_copy {
            let gpu_tex = self.gpu_texture.as_ref().unwrap();
            unsafe {
                context.CopyResource(gpu_tex, &texture);
                context.Flush();
            }

            let is_black = if self.frame_count <= 3 {
                unsafe {
                    context.CopyResource(&staging, &texture);
                    context.Flush();
                    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
                    if context.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped)).is_ok() {
                        let stride = mapped.RowPitch;
                        let data_size = (stride * self.height) as usize;
                        let data_slice = std::slice::from_raw_parts(mapped.pData as *const u8, data_size);
                        let black = is_buffer_black(data_slice);
                        context.Unmap(&staging, 0);
                        black
                    } else {
                        false
                    }
                }
            } else {
                false
            };

            // display_base.cpp:~350 release_snapshot(): the frame is delivered,
            // then ACCESS_LOST from ReleaseFrame means reinit. The frame is dropped
            // (the encoder session is rebuilt right after anyway).
            match release_frame(&duplication) {
                AcquireStatus::Reinit => return Err(MediaError::CaptureReinit),
                AcquireStatus::Error => {
                    return Err(MediaError::CaptureError("ReleaseFrame failed".into()));
                }
                _ => {}
            }

            if is_black {
                let stride = self.width * 4;
                let data_size = (stride * self.height) as usize;
                return Ok(CapturedFrame {
                    buffer: GpuBuffer::CpuBuffer {
                        data: vec![0; data_size],
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
                    cursor: None,
                });
            }

            self.last_returned_time = Some(std::time::Instant::now());
            return Ok(CapturedFrame {
                buffer: GpuBuffer::D3D11Texture {
                    texture: gpu_tex.as_raw(),
                    array_index: 0,
                },
                timestamp_us: timestamp,
                width: self.width,
                height: self.height,
                format: PixelFormat::BGRA,
                is_new_frame: true,
                cursor: None,
            });
        }

        unsafe {
            context.CopyResource(&staging, &texture);
        }

        match release_frame(&duplication) {
            AcquireStatus::Reinit => return Err(MediaError::CaptureReinit),
            AcquireStatus::Error => {
                return Err(MediaError::CaptureError("ReleaseFrame failed".into()));
            }
            _ => {}
        }

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(|e| MediaError::CaptureError(format!("Map failed: {e}")))?;
        }

        let stride = mapped.RowPitch;
        let data_size = (stride * self.height) as usize;
        let mut data = unsafe {
            std::slice::from_raw_parts(mapped.pData as *const u8, data_size).to_vec()
        };

        unsafe {
            context.Unmap(&staging, 0);
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
            cursor: None,
        })
    }

    async fn stop(&mut self) -> Result<(), MediaError> {
        self.keep_awake = None; // SetThreadExecutionState(ES_CONTINUOUS), display_base.cpp:259
        self.factory = None;
        self.duplication = None;
        self.staging_texture = None;
        self.gpu_texture = None;
        self.last_frame_data = None;
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

    fn get_d3d11_device(&self) -> Option<(usize, usize)> {
        match (&self.device, &self.context) {
            (Some(d), Some(c)) => Some((d.as_raw() as usize, c.as_raw() as usize)),
            _ => None,
        }
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
        if GetMonitorInfoW(hmonitor, &mut info as *mut MONITORINFOEXW as *mut _).as_bool() {
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

        if let Err(e) = BitBlt(
            hdc_mem,
            0,
            0,
            width as i32,
            height as i32,
            hdc_screen,
            x,
            y,
            SRCCOPY,
        ) {
            let _ = SelectObject(hdc_mem, h_old);
            let _ = DeleteObject(h_bitmap);
            let _ = DeleteDC(hdc_mem);
            let _ = ReleaseDC(None, hdc_screen);
            return Err(MediaError::CaptureError(format!("BitBlt failed: {e}")));
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
        // "default" → the primary monitor (positioned at the desktop origin),
        // not GDI monitor index 0.
        let display_index: usize = if display_id == "default" || display_id.is_empty() {
            displays
                .iter()
                .position(|d| d.x == 0 && d.y == 0)
                .unwrap_or(0)
        } else {
            display_id.parse().unwrap_or(0)
        };
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
            cursor: None,
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
                    log::warn!(
                        "Failed to start DXGI capture: {}. Falling back to GDI capture \
                         (last resort: GDI frames are CPU buffers that the hardware encoders cannot encode).",
                        e
                    );
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

    fn get_d3d11_device(&self) -> Option<(usize, usize)> {
        if self.active_is_gdi {
            None
        } else {
            self.dxgi.as_ref().and_then(|dxgi| dxgi.get_d3d11_device())
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
