//! NvFBC screen capture backend (Linux/NVIDIA).
//!
//! Uses NVIDIA Frame Buffer Capture (NvFBC) to capture the screen directly on the GPU.
//! The GPU performs the NV12 color conversion, avoiding any expensive CPU-side
//! conversion.
//!
//! All NvFBC calls are executed on a dedicated background thread to guarantee that
//! the underlying OpenGL/CUDA context remains bound to the same OS thread.
//!
//! Capture pacing uses NvFBC's native blocking frame-grab mode, synchronizing screen
//! capture directly with rendering events on the GPU for minimal mouse cursor delay.

use nvfbc::cuda::CudaCapturer;
use nvfbc::{BufferFormat, SystemCapturer};
use nvfbc_sys;
use std::time::Instant;

use crate::capture::gpu_buffer::GpuBuffer;
use crate::capture::{CapturedFrame, ScreenCapture};
use crate::error::MediaError;
use crate::types::*;

enum ActiveCapturer {
    System(SystemCapturer),
    Cuda(CudaCapturer),
}

impl ActiveCapturer {
    fn status(&self) -> Result<nvfbc::Status, nvfbc::Error> {
        match self {
            ActiveCapturer::System(c) => c.status(),
            ActiveCapturer::Cuda(c) => c.status(),
        }
    }

    #[allow(dead_code)]
    fn start(&mut self, format: BufferFormat, fps: u32) -> Result<(), nvfbc::Error> {
        match self {
            ActiveCapturer::System(c) => c.start(format, fps),
            ActiveCapturer::Cuda(c) => c.start(format, fps),
        }
    }

    fn stop(&mut self) -> Result<(), nvfbc::Error> {
        match self {
            ActiveCapturer::System(c) => c.stop(),
            ActiveCapturer::Cuda(c) => c.stop(),
        }
    }
}

struct CudaContextGuard {
    context: *mut std::ffi::c_void,
    cu_ctx_destroy: unsafe extern "C" fn(*mut std::ffi::c_void) -> i32,
    lib: *mut std::ffi::c_void,
}

impl Drop for CudaContextGuard {
    fn drop(&mut self) {
        unsafe {
            if !self.context.is_null() {
                (self.cu_ctx_destroy)(self.context);
                log::info!("CUDA context destroyed");
            }
            if !self.lib.is_null() {
                libc::dlclose(self.lib);
                log::info!("libcuda closed");
            }
        }
    }
}

unsafe fn try_init_cuda() -> Result<CudaContextGuard, String> {
    let mut lib = libc::dlopen(
        b"libcuda.so.1\0".as_ptr() as *const libc::c_char,
        libc::RTLD_LAZY,
    );
    if lib.is_null() {
        lib = libc::dlopen(
            b"libcuda.so\0".as_ptr() as *const libc::c_char,
            libc::RTLD_LAZY,
        );
    }
    if lib.is_null() {
        return Err("Failed to load libcuda.so.1 or libcuda.so".into());
    }

    let cu_init_ptr = libc::dlsym(lib, b"cuInit\0".as_ptr() as *const libc::c_char);
    if cu_init_ptr.is_null() {
        libc::dlclose(lib);
        return Err("cuInit symbol not found".into());
    }
    let cu_init: unsafe extern "C" fn(u32) -> i32 = std::mem::transmute(cu_init_ptr);

    let cu_device_get_ptr = libc::dlsym(lib, b"cuDeviceGet\0".as_ptr() as *const libc::c_char);
    if cu_device_get_ptr.is_null() {
        libc::dlclose(lib);
        return Err("cuDeviceGet symbol not found".into());
    }
    let cu_device_get: unsafe extern "C" fn(*mut i32, i32) -> i32 =
        std::mem::transmute(cu_device_get_ptr);

    let cu_ctx_create_ptr = libc::dlsym(lib, b"cuCtxCreate_v2\0".as_ptr() as *const libc::c_char);
    if cu_ctx_create_ptr.is_null() {
        libc::dlclose(lib);
        return Err("cuCtxCreate_v2 symbol not found".into());
    }
    let cu_ctx_create: unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32, i32) -> i32 =
        std::mem::transmute(cu_ctx_create_ptr);

    let cu_ctx_destroy_ptr = libc::dlsym(lib, b"cuCtxDestroy_v2\0".as_ptr() as *const libc::c_char);
    if cu_ctx_destroy_ptr.is_null() {
        libc::dlclose(lib);
        return Err("cuCtxDestroy_v2 symbol not found".into());
    }
    let cu_ctx_destroy: unsafe extern "C" fn(*mut std::ffi::c_void) -> i32 =
        std::mem::transmute(cu_ctx_destroy_ptr);

    // 1. Initialize CUDA
    let res = cu_init(0);
    if res != 0 {
        libc::dlclose(lib);
        return Err(format!("cuInit failed with code: {}", res));
    }

    // 2. Get device 0
    let mut device: i32 = 0;
    let res = cu_device_get(&mut device, 0);
    if res != 0 {
        libc::dlclose(lib);
        return Err(format!("cuDeviceGet failed with code: {}", res));
    }

    // 3. Create context (flags = 0x08 for CU_CTX_MAP_HOST | CU_CTX_SCHED_AUTO)
    let mut context: *mut std::ffi::c_void = std::ptr::null_mut();
    let res = cu_ctx_create(&mut context, 0x08, device);
    if res != 0 {
        libc::dlclose(lib);
        return Err(format!("cuCtxCreate_v2 failed with code: {}", res));
    }

    Ok(CudaContextGuard {
        context,
        cu_ctx_destroy,
        lib,
    })
}

unsafe fn custom_start_nvfbc(
    handle: nvfbc_sys::NVFBC_SESSION_HANDLE,
    capture_type: nvfbc_sys::_NVFBC_CAPTURE_TYPE,
    format: nvfbc::BufferFormat,
    fps: u32,
    with_cursor: bool,
    display_id: &str,
) -> Result<(), String> {
    let mut params: nvfbc_sys::_NVFBC_CREATE_CAPTURE_SESSION_PARAMS = std::mem::zeroed();
    params.dwVersion = nvfbc_sys::NVFBC_CREATE_CAPTURE_SESSION_PARAMS_VER;
    params.eCaptureType = capture_type;
    params.bWithCursor = if with_cursor {
        nvfbc_sys::_NVFBC_BOOL_NVFBC_TRUE
    } else {
        nvfbc_sys::_NVFBC_BOOL_NVFBC_FALSE
    };
    params.frameSize = nvfbc_sys::NVFBC_SIZE { w: 0, h: 0 };
    params.eTrackingType = nvfbc_sys::NVFBC_TRACKING_TYPE_NVFBC_TRACKING_DEFAULT;
    if display_id != "default" {
        params.eTrackingType = nvfbc_sys::NVFBC_TRACKING_TYPE_NVFBC_TRACKING_OUTPUT;
        params.dwOutputId = display_id
            .parse::<u32>()
            .map_err(|_| format!("Invalid NvFBC output id '{}'", display_id))?;
    }
    params.dwSamplingRateMs = (1000 / fps.max(1)) as u32;
    params.bPushModel = nvfbc_sys::_NVFBC_BOOL_NVFBC_TRUE;
    params.bAllowDirectCapture = nvfbc_sys::_NVFBC_BOOL_NVFBC_TRUE;

    let ret = nvfbc_sys::NvFBCCreateCaptureSession(handle, &mut params);
    if ret != nvfbc_sys::_NVFBCSTATUS_NVFBC_SUCCESS {
        return Err(format!("NvFBCCreateCaptureSession failed: {}", ret));
    }

    match capture_type {
        nvfbc_sys::_NVFBC_CAPTURE_TYPE_NVFBC_CAPTURE_TO_SYS => {
            let mut setup_params: nvfbc_sys::NVFBC_TOSYS_SETUP_PARAMS = std::mem::zeroed();
            setup_params.dwVersion = nvfbc_sys::NVFBC_TOSYS_SETUP_PARAMS_VER;
            setup_params.eBufferFormat = format as u32;
            let ret = nvfbc_sys::NvFBCToSysSetUp(handle, &mut setup_params);
            if ret != nvfbc_sys::_NVFBCSTATUS_NVFBC_SUCCESS {
                return Err(format!("NvFBCToSysSetUp failed: {}", ret));
            }
        }
        nvfbc_sys::_NVFBC_CAPTURE_TYPE_NVFBC_CAPTURE_SHARED_CUDA => {
            let mut setup_params: nvfbc_sys::NVFBC_TOCUDA_SETUP_PARAMS = std::mem::zeroed();
            setup_params.dwVersion = nvfbc_sys::NVFBC_TOCUDA_SETUP_PARAMS_VER;
            setup_params.eBufferFormat = format as u32;
            let ret = nvfbc_sys::NvFBCToCudaSetUp(handle, &mut setup_params);
            if ret != nvfbc_sys::_NVFBCSTATUS_NVFBC_SUCCESS {
                return Err(format!("NvFBCToCudaSetUp failed: {}", ret));
            }
        }
        _ => return Err("Unsupported capture type".into()),
    }

    Ok(())
}

// --- X11 error handler to prevent GLX errors from killing the process ---
// NvFBC/CUDA initialization can trigger X11 GLX errors (e.g., BadValue on
// X_GLXCreateNewContext) which by default call exit(). We install a custom
// handler to suppress these and allow graceful fallback.

use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

static X11_ERROR_OCCURRED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" fn custom_x11_error_handler(
    _display: *mut libc::c_void,
    event: *mut libc::c_void,
) -> libc::c_int {
    // XErrorEvent structure: first field after Display* is the error_code
    // We just log and flag it instead of calling exit()
    X11_ERROR_OCCURRED.store(true, AtomicOrdering::SeqCst);

    // Try to extract useful info from the XErrorEvent
    #[repr(C)]
    struct XErrorEvent {
        type_: libc::c_int,
        display: *mut libc::c_void,
        resourceid: libc::c_ulong,
        serial: libc::c_ulong,
        error_code: u8,
        request_code: u8,
        minor_code: u8,
    }

    if !event.is_null() {
        let err = &*(event as *const XErrorEvent);
        log::warn!(
            "Suppressed X11 error during NvFBC init: error_code={}, request_code={}, minor_code={}",
            err.error_code,
            err.request_code,
            err.minor_code
        );
    }

    0 // Return 0 to continue (don't exit)
}

/// Initialize an NvFBC capturer under a temporary X11 error-handler guard.
///
/// When `prefer_cuda` is `true`, a CUDA context + NvFBC CUDA capturer is tried
/// first (zero-copy GPU capture path). When `false`, CUDA is skipped entirely
/// and only the System capturer is created — much cheaper (no ~28 MiB CUDA
/// context, no `cuInit`), which is all that display enumeration / status queries
/// need.
fn init_capturer(prefer_cuda: bool) -> Result<(ActiveCapturer, Option<CudaContextGuard>), String> {
    // Install custom X11 error handler to prevent GLX errors from being fatal.
    // NvFBC internally creates GLX contexts which can trigger X errors that
    // would kill the process. We suppress them and check after each step.
    X11_ERROR_OCCURRED.store(false, AtomicOrdering::SeqCst);

    // Open X display for XSync calls (to flush errors synchronously)
    let (xlib_handle, x_display, set_handler_fn) = unsafe {
        let xlib = libc::dlopen(
            b"libX11.so.6\0".as_ptr() as *const libc::c_char,
            libc::RTLD_LAZY,
        );
        if xlib.is_null() {
            (std::ptr::null_mut(), std::ptr::null_mut(), None)
        } else {
            let x_open = libc::dlsym(xlib, b"XOpenDisplay\0".as_ptr() as *const libc::c_char);
            let display = if !x_open.is_null() {
                let open_fn: unsafe extern "C" fn(*const libc::c_char) -> *mut libc::c_void =
                    std::mem::transmute(x_open);
                open_fn(std::ptr::null())
            } else {
                std::ptr::null_mut()
            };
            let set_err = libc::dlsym(xlib, b"XSetErrorHandler\0".as_ptr() as *const libc::c_char);
            let handler_fn = if !set_err.is_null() {
                type XSetErrorHandlerFn = unsafe extern "C" fn(
                    Option<
                        unsafe extern "C" fn(*mut libc::c_void, *mut libc::c_void) -> libc::c_int,
                    >,
                ) -> Option<
                    unsafe extern "C" fn(*mut libc::c_void, *mut libc::c_void) -> libc::c_int,
                >;
                Some(std::mem::transmute::<_, XSetErrorHandlerFn>(set_err))
            } else {
                None
            };
            (xlib, display, handler_fn)
        }
    };

    // Install our error handler
    let prev_handler: Option<
        unsafe extern "C" fn(*mut libc::c_void, *mut libc::c_void) -> libc::c_int,
    > = unsafe {
        if let Some(set_handler) = set_handler_fn {
            set_handler(Some(custom_x11_error_handler))
        } else {
            None
        }
    };

    // Helper to flush X11 events and check for errors
    let x_sync = |display: *mut libc::c_void, xlib: *mut libc::c_void| {
        if !display.is_null() && !xlib.is_null() {
            unsafe {
                let x_sync_sym = libc::dlsym(xlib, b"XSync\0".as_ptr() as *const libc::c_char);
                if !x_sync_sym.is_null() {
                    let sync_fn: unsafe extern "C" fn(
                        *mut libc::c_void,
                        libc::c_int,
                    ) -> libc::c_int = std::mem::transmute(x_sync_sym);
                    sync_fn(display, 0); // discard=False
                }
            }
        }
    };

    let result = init_capturer_inner(x_display, xlib_handle, &x_sync, prefer_cuda);

    // Restore previous X11 error handler
    unsafe {
        if let Some(set_handler) = set_handler_fn {
            set_handler(prev_handler);
        }
        // Close our X display
        if !x_display.is_null() && !xlib_handle.is_null() {
            let x_close = libc::dlsym(
                xlib_handle,
                b"XCloseDisplay\0".as_ptr() as *const libc::c_char,
            );
            if !x_close.is_null() {
                let close_fn: unsafe extern "C" fn(*mut libc::c_void) -> libc::c_int =
                    std::mem::transmute(x_close);
                close_fn(x_display);
            }
        }
        if !xlib_handle.is_null() {
            libc::dlclose(xlib_handle);
        }
    }

    result
}

fn init_capturer_inner(
    x_display: *mut libc::c_void,
    xlib_handle: *mut libc::c_void,
    x_sync: &dyn Fn(*mut libc::c_void, *mut libc::c_void),
    prefer_cuda: bool,
) -> Result<(ActiveCapturer, Option<CudaContextGuard>), String> {
    // Step 1: Try CUDA + NvFBC CUDA Capturer (unless the caller only needs a
    // lightweight status query, in which case CUDA is skipped entirely).
    if prefer_cuda {
        match unsafe { try_init_cuda() } {
        Ok(guard) => {
            // XSync to flush any X errors from CUDA init
            x_sync(x_display, xlib_handle);
            if X11_ERROR_OCCURRED.load(AtomicOrdering::SeqCst) {
                log::warn!("X11 error occurred during CUDA init — dropping CUDA context and skipping NvFBC");
                drop(guard); // Explicitly destroy CUDA context
                X11_ERROR_OCCURRED.store(false, AtomicOrdering::SeqCst);
                return Err("X11 error during CUDA initialization".into());
            }

            log::info!(
                "CUDA initialized successfully. Attempting to initialize NvFBC CUDA Capturer..."
            );
            match CudaCapturer::new() {
                Ok(c) => {
                    // XSync to flush any X errors from CudaCapturer::new()
                    x_sync(x_display, xlib_handle);
                    if X11_ERROR_OCCURRED.load(AtomicOrdering::SeqCst) {
                        log::warn!(
                            "X11/GLX error occurred during NvFBC CUDA Capturer init — cleaning up"
                        );
                        drop(c); // Drop the capturer first
                        drop(guard); // Then destroy CUDA context
                        X11_ERROR_OCCURRED.store(false, AtomicOrdering::SeqCst);
                        return Err("X11/GLX error during NvFBC CUDA capturer init (e.g., GLX context creation failed)".into());
                    }
                    log::info!("Successfully initialized NvFBC CUDA Capturer");
                    return Ok((ActiveCapturer::Cuda(c), Some(guard)));
                }
                Err(e) => {
                    log::warn!("Failed to initialize NvFBC CUDA Capturer: {:?}. Falling back to System Capturer", e);
                    // XSync after failed attempt too
                    x_sync(x_display, xlib_handle);
                    if X11_ERROR_OCCURRED.load(AtomicOrdering::SeqCst) {
                        log::warn!("X11 error detected after NvFBC CUDA failure — cleaning up CUDA context");
                        drop(guard);
                        X11_ERROR_OCCURRED.store(false, AtomicOrdering::SeqCst);
                        return Err("X11 error during NvFBC initialization".into());
                    }
                    drop(guard); // Drop CUDA guard since we're not using CUDA capturer
                }
            }
        }
        Err(e) => {
            log::warn!(
                "CUDA context initialization failed: {}. Falling back to NvFBC System Capturer",
                e
            );
        }
        }
    } // end `if prefer_cuda`

    // Step 2: Try NvFBC System Capturer (no CUDA needed)
    match SystemCapturer::new() {
        Ok(c) => {
            x_sync(x_display, xlib_handle);
            if X11_ERROR_OCCURRED.load(AtomicOrdering::SeqCst) {
                log::warn!("X11 error occurred during NvFBC System Capturer init — skipping");
                drop(c);
                return Err("X11 error during NvFBC System capturer init".into());
            }
            Ok((ActiveCapturer::System(c), None))
        }
        Err(e) => Err(format!(
            "Failed to initialize NvFBC System Capturer: {:?}",
            e
        )),
    }
}

/// Build [`DisplayInfo`]s from an NvFBC status. The display id is the **numeric
/// NvFBC output id**, which is what [`custom_start_nvfbc`] parses back into
/// `dwOutputId` — so ids produced here round-trip correctly to a capture start.
fn displays_from_status(status: &nvfbc::Status) -> Vec<DisplayInfo> {
    status
        .outputs
        .iter()
        .enumerate()
        .map(|(index, output)| DisplayInfo {
            id: output.id.to_string(),
            name: output.name.clone(),
            width: output.tracked_box.w,
            height: output.tracked_box.h,
            refresh_rate: 60.0,
            is_primary: index == 0,
        })
        .collect()
}

/// Enumerate displays for capability queries **without** creating a CUDA context.
///
/// Creates only the NvFBC *System* capturer (no CUDA context, no capture
/// session) purely to read output status, then tears it down. This avoids the
/// ~28 MiB CUDA context and the slow `cuInit`/`cuCtxCreate` that a full capture
/// backend pays — which matters because capability queries fire every time a
/// client opens its settings. Returns the same numeric NvFBC output ids as the
/// capture path, so display selection is unaffected.
///
/// Returns `Err` when NvFBC is unavailable (non-NVIDIA / Wayland), letting the
/// caller fall back to the full capture backend. Must be called from a blocking
/// context: it performs synchronous NvFBC/GLX init on the calling thread and
/// drops the capturer on that same thread (GLX context thread-affinity).
pub fn list_displays_lightweight() -> Result<Vec<DisplayInfo>, MediaError> {
    let (capturer, _guard) = init_capturer(false)
        .map_err(|e| MediaError::CaptureError(format!("NvFBC status init failed: {}", e)))?;
    let status_res = capturer.status();
    // Drop on THIS thread (same thread that created the GLX context).
    drop(capturer);
    let status = status_res
        .map_err(|e| MediaError::CaptureError(format!("NvFBC status query failed: {:?}", e)))?;
    Ok(displays_from_status(&status))
}

enum NvfbcCommand {
    GetStatus {
        reply: std::sync::mpsc::Sender<Result<nvfbc::Status, String>>,
    },
    Start {
        display_id: String,
        config: StreamConfig,
        frame_tx: tokio::sync::mpsc::Sender<Result<CapturedFrame, String>>,
        reply: std::sync::mpsc::Sender<Result<(), String>>,
    },
    Stop {
        reply: std::sync::mpsc::Sender<Result<(), String>>,
    },
    /// Terminate the background thread entirely (used by `Drop`). Unlike `Stop`
    /// — which only ends the current capture session but keeps the thread alive
    /// for reuse (e.g. `set_fps` does stop+start) — `Shutdown` breaks the thread
    /// loop so the thread returns, dropping the capturer and CUDA context. Without
    /// a dedicated shutdown signal, `Drop::join()` deadlocks: `Stop` leaves the
    /// thread blocked on `cmd_rx.recv()`, which never returns while `cmd_tx` is
    /// still held by the struct being dropped.
    Shutdown {
        reply: std::sync::mpsc::Sender<Result<(), String>>,
    },
}

/// NvFBC-based screen capture backend.
pub struct NvfbcCapture {
    cmd_tx: std::sync::mpsc::Sender<NvfbcCommand>,
    thread_handle: Option<std::thread::JoinHandle<()>>,
    width: u32,
    height: u32,
    fps: u32,
    capturing: bool,
    last_frame_time: Instant,
    frame_rx: Option<tokio::sync::mpsc::Receiver<Result<CapturedFrame, String>>>,
    /// Display id passed to the last `start` call, needed to recreate the
    /// session when the capture frame rate changes at runtime.
    display_id: String,
    /// Config from the last `start` call, reused when reconfiguring the session
    /// (e.g. on `set_fps`).
    last_config: Option<StreamConfig>,
}

impl NvfbcCapture {
    /// Creates a new NvFBC capture instance.
    ///
    /// Spawns a dedicated background thread for all NvFBC API interactions.
    pub fn new() -> Result<Self, MediaError> {
        // NvFBC is only supported on X11
        let session_type = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();
        if session_type == "wayland" {
            return Err(MediaError::PlatformNotSupported(
                "NvFBC capture is only supported on X11 sessions.".into(),
            ));
        }

        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<NvfbcCommand>();
        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();

        let thread_handle = std::thread::Builder::new()
            .name("lunaris-nvfbc".into())
            .spawn(move || {
                let (mut capturer, cuda_guard) = match init_capturer(true) {
                    Ok(res) => {
                        let _ = init_tx.send(Ok(()));
                        res
                    }
                    Err(e) => {
                        let _ = init_tx.send(Err(e));
                        return;
                    }
                };

                let mut active_frame_tx: Option<tokio::sync::mpsc::Sender<Result<CapturedFrame, String>>> = None;

                'thread_loop: loop {
                    // 1. Process all pending commands first (non-blocking check)
                    while let Ok(cmd) = cmd_rx.try_recv() {
                        match cmd {
                            NvfbcCommand::GetStatus { reply } => {
                                let res = capturer.status().map_err(|e| format!("{:?}", e));
                                let _ = reply.send(res);
                            }
                            NvfbcCommand::Start { display_id, config, frame_tx, reply } => {
                                let handle: nvfbc_sys::NVFBC_SESSION_HANDLE = match &capturer {
                                    ActiveCapturer::System(c) => unsafe { *(c as *const SystemCapturer as *const nvfbc_sys::NVFBC_SESSION_HANDLE) },
                                    ActiveCapturer::Cuda(c) => unsafe { *(c as *const CudaCapturer as *const nvfbc_sys::NVFBC_SESSION_HANDLE) },
                                };
                                let capture_type = match &capturer {
                                    ActiveCapturer::System(_) => nvfbc_sys::_NVFBC_CAPTURE_TYPE_NVFBC_CAPTURE_TO_SYS,
                                    ActiveCapturer::Cuda(_) => nvfbc_sys::_NVFBC_CAPTURE_TYPE_NVFBC_CAPTURE_SHARED_CUDA,
                                };

                                let with_cursor = crate::capture::should_embed_host_cursor();
                                log::info!(
                                    "NvFBC capture: {} host cursor in video stream",
                                    if with_cursor { "embedding" } else { "hiding" }
                                );

                                let res = unsafe {
                                    custom_start_nvfbc(
                                        handle,
                                        capture_type,
                                        BufferFormat::Bgra,
                                        config.fps,
                                        with_cursor,
                                        &display_id,
                                    )
                                };
                                if res.is_ok() {
                                    active_frame_tx = Some(frame_tx);
                                }
                                let _ = reply.send(res);
                            }
                            NvfbcCommand::Stop { reply } => {
                                active_frame_tx = None;
                                let res = capturer.stop().map_err(|e| format!("{:?}", e));
                                let _ = reply.send(res);
                            }
                            NvfbcCommand::Shutdown { reply } => {
                                // End any active session, then break the loop so the
                                // thread returns and its capturer + CUDA context drop.
                                let _ = capturer.stop();
                                let _ = reply.send(Ok(()));
                                break 'thread_loop;
                            }
                        }
                    }

                    // 2. Grab a frame if capturing
                    if let Some(ref frame_tx) = active_frame_tx {
                        let timestamp_us = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_micros() as u64;

                        let grab_res = match &mut capturer {
                            ActiveCapturer::System(c) => {
                                c.next_frame(
                                    nvfbc::system::CaptureMethod::Blocking,
                                    Some(std::time::Duration::from_millis(50)),
                                ).map(|frame_info| CapturedFrame {
                                    buffer: GpuBuffer::CpuBuffer {
                                        data: frame_info.buffer.to_vec(),
                                        stride: frame_info.width * 4,
                                        format: PixelFormat::BGRA,
                                        width: frame_info.width,
                                        height: frame_info.height,
                                    },
                                    timestamp_us,
                                    width: frame_info.width,
                                    height: frame_info.height,
                                    format: PixelFormat::BGRA,
                                    is_new_frame: frame_info.is_new_frame,
                                    cursor: None,
                                })
                            }
                            ActiveCapturer::Cuda(c) => {
                                c.next_frame(
                                    nvfbc::cuda::CaptureMethod::Blocking,
                                    Some(std::time::Duration::from_millis(50)),
                                ).map(|frame_info| CapturedFrame {
                                    buffer: GpuBuffer::CudaPointer {
                                        ptr: frame_info.device_buffer,
                                        cuda_context: cuda_guard.as_ref().map(|g| g.context as usize).unwrap_or(0),
                                        size: frame_info.device_buffer_len as usize,
                                        width: frame_info.width,
                                        height: frame_info.height,
                                        stride: frame_info.width * 4,
                                        format: PixelFormat::BGRA,
                                    },
                                    timestamp_us,
                                    width: frame_info.width,
                                    height: frame_info.height,
                                    format: PixelFormat::BGRA,
                                    is_new_frame: frame_info.is_new_frame,
                                    cursor: None,
                                })
                            }
                        };

                        let res = grab_res.map_err(|e| format!("{:?}", e));
                        if let Err(e) = frame_tx.try_send(res) {
                            if matches!(e, tokio::sync::mpsc::error::TrySendError::Closed(_)) {
                                active_frame_tx = None;
                            }
                        }
                    } else {
                        // Not capturing, block on cmd_rx
                        match cmd_rx.recv() {
                            Ok(cmd) => {
                                match cmd {
                                    NvfbcCommand::GetStatus { reply } => {
                                        let res = capturer.status().map_err(|e| format!("{:?}", e));
                                        let _ = reply.send(res);
                                    }
                                    NvfbcCommand::Start { display_id, config, frame_tx, reply } => {
                                        let handle: nvfbc_sys::NVFBC_SESSION_HANDLE = match &capturer {
                                            ActiveCapturer::System(c) => unsafe { *(c as *const SystemCapturer as *const nvfbc_sys::NVFBC_SESSION_HANDLE) },
                                            ActiveCapturer::Cuda(c) => unsafe { *(c as *const CudaCapturer as *const nvfbc_sys::NVFBC_SESSION_HANDLE) },
                                        };
                                        let capture_type = match &capturer {
                                            ActiveCapturer::System(_) => nvfbc_sys::_NVFBC_CAPTURE_TYPE_NVFBC_CAPTURE_TO_SYS,
                                            ActiveCapturer::Cuda(_) => nvfbc_sys::_NVFBC_CAPTURE_TYPE_NVFBC_CAPTURE_SHARED_CUDA,
                                        };
                                        let with_cursor = crate::capture::should_embed_host_cursor();
                                        log::info!(
                                            "NvFBC capture: {} host cursor in video stream",
                                            if with_cursor { "embedding" } else { "hiding" }
                                        );
                                        let res = unsafe {
                                            custom_start_nvfbc(
                                                handle,
                                                capture_type,
                                                BufferFormat::Bgra,
                                                config.fps,
                                                with_cursor,
                                        &display_id,
                                            )
                                        };
                                        if res.is_ok() {
                                            active_frame_tx = Some(frame_tx);
                                        }
                                        let _ = reply.send(res);
                                    }
                                    NvfbcCommand::Stop { reply } => {
                                        let _ = reply.send(Ok(()));
                                    }
                                    NvfbcCommand::Shutdown { reply } => {
                                        // Not capturing here, so just acknowledge and
                                        // break out so the thread returns and drops its
                                        // capturer + CUDA context.
                                        let _ = reply.send(Ok(()));
                                        break 'thread_loop;
                                    }
                                }
                            }
                            Err(_) => break 'thread_loop, // cmd_tx dropped, exit thread
                        }
                    }
                }
            })
            .map_err(|e| MediaError::CaptureError(format!("Failed to spawn NvFBC thread: {}", e)))?;

        // Wait for thread initialization status
        init_rx
            .recv()
            .map_err(|_| MediaError::CaptureError("NvFBC thread panicked during setup".into()))?
            .map_err(|e| MediaError::CaptureError(e))?;

        // Query status to get the current screen resolution
        let (status_tx, status_rx) = std::sync::mpsc::channel();
        cmd_tx
            .send(NvfbcCommand::GetStatus { reply: status_tx })
            .map_err(|_| {
                MediaError::CaptureError("Failed to communicate with NvFBC thread".into())
            })?;

        let status = status_rx
            .recv()
            .map_err(|_| MediaError::CaptureError("NvFBC thread query failed".into()))?
            .map_err(|e_str| {
                MediaError::CaptureError(format!("Failed to query status: {}", e_str))
            })?;

        if !status.is_capture_possible {
            return Err(MediaError::CaptureError(
                "NvFBC capture is not supported by the graphics driver or hardware.".into(),
            ));
        }

        Ok(Self {
            cmd_tx,
            thread_handle: Some(thread_handle),
            width: status.screen_size.w,
            height: status.screen_size.h,
            fps: 60,
            capturing: false,
            last_frame_time: Instant::now(),
            frame_rx: None,
            display_id: "default".to_string(),
            last_config: None,
        })
    }

    /// Fallback display enumeration using xrandr.
    fn list_displays_xrandr() -> Result<Vec<DisplayInfo>, MediaError> {
        let output = std::process::Command::new("xrandr")
            .arg("--query")
            .output()
            .map_err(|e| MediaError::CaptureError(format!("xrandr failed: {}", e)))?;

        let text = String::from_utf8_lossy(&output.stdout);
        let mut displays = Vec::new();

        for line in text.lines() {
            // Match lines like: "DP-1 connected primary 1920x1080+0+0"
            // or: "HDMI-0 connected 1920x1080+1920+0"
            if !line.contains(" connected") {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            let name = parts[0].to_string();
            let is_primary = parts.contains(&"primary");

            // Extract resolution and offset from geometry string (e.g. "1920x1080+0+0")
            let (width, height) = parts
                .iter()
                .find(|p| p.contains('x') && p.contains('+'))
                .and_then(|geo| {
                    let res_part = geo.split('+').next()?;
                    let mut wh = res_part.split('x');
                    let w = wh.next()?.parse::<u32>().ok()?;
                    let h = wh.next()?.parse::<u32>().ok()?;
                    Some((w, h))
                })
                .unwrap_or((0, 0));

            if width > 0 && height > 0 {
                displays.push(DisplayInfo {
                    id: name.clone(),
                    name,
                    width,
                    height,
                    refresh_rate: 60.0, // xrandr doesn't easily give refresh rate per output here
                    is_primary,
                });
            }
        }

        Ok(displays)
    }
}

impl Drop for NvfbcCapture {
    fn drop(&mut self) {
        // Send a dedicated Shutdown (NOT Stop): Stop only ends the capture
        // session but leaves the thread blocked on `cmd_rx.recv()`, so the
        // `join()` below would deadlock (the thread never exits while `cmd_tx`
        // is still held here). Shutdown breaks the thread loop so `join()`
        // returns and the CUDA context / NvFBC session are released.
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        if self
            .cmd_tx
            .send(NvfbcCommand::Shutdown { reply: reply_tx })
            .is_ok()
        {
            let _ = reply_rx.recv();
        }

        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

#[async_trait::async_trait]
impl ScreenCapture for NvfbcCapture {
    async fn list_displays(&self) -> Result<Vec<DisplayInfo>, MediaError> {
        let (status_tx, status_rx) = std::sync::mpsc::channel();
        self.cmd_tx
            .send(NvfbcCommand::GetStatus { reply: status_tx })
            .map_err(|_| {
                MediaError::CaptureError("Failed to communicate with NvFBC thread".into())
            })?;

        let status = status_rx
            .recv()
            .map_err(|_| MediaError::CaptureError("NvFBC thread query failed".into()))?
            .map_err(|e| {
                MediaError::CaptureError(format!("Failed to query NvFBC status: {}", e))
            })?;

        log::info!("NvFBC status: {} outputs", status.outputs.len());
        if !status.outputs.is_empty() {
            for (index, output) in status.outputs.iter().enumerate() {
                log::info!(
                    "NvFBC output[{}]: id={} name={} {}x{}",
                    index,
                    output.id,
                    output.name,
                    output.tracked_box.w,
                    output.tracked_box.h
                );
            }
            return Ok(displays_from_status(&status));
        }

        // Fallback: use xrandr to enumerate displays
        log::info!("NvFBC returned no outputs, falling back to xrandr for display enumeration");
        match Self::list_displays_xrandr() {
            Ok(displays) if !displays.is_empty() => return Ok(displays),
            Ok(_) => log::warn!("xrandr returned no displays"),
            Err(e) => log::warn!("xrandr fallback failed: {}", e),
        }

        Ok(vec![DisplayInfo {
            id: "default".to_string(),
            name: "NVIDIA NvFBC Display".to_string(),
            width: self.width,
            height: self.height,
            refresh_rate: 60.0,
            is_primary: true,
        }])
    }

    async fn start(&mut self, display_id: &str, config: &StreamConfig) -> Result<(), MediaError> {
        if self.capturing {
            return Err(MediaError::CaptureAlreadyStarted);
        }

        self.width = config.width;
        self.height = config.height;
        self.fps = config.fps;
        self.display_id = display_id.to_string();
        self.last_config = Some(config.clone());

        // Create the mpsc channel for streaming frames.
        // Capacity of 2 is optimal for double buffering.
        let (frame_tx, frame_rx) = tokio::sync::mpsc::channel(2);
        self.frame_rx = Some(frame_rx);

        // Send start capture command
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.cmd_tx
            .send(NvfbcCommand::Start {
                display_id: display_id.to_string(),
                config: config.clone(),
                frame_tx,
                reply: reply_tx,
            })
            .map_err(|_| {
                MediaError::CaptureError("Failed to communicate with NvFBC thread".into())
            })?;

        reply_rx
            .recv()
            .map_err(|_| MediaError::CaptureError("NvFBC thread communication failed".into()))?
            .map_err(|e| {
                MediaError::CaptureError(format!("Failed to start NvFBC session: {}", e))
            })?;

        self.capturing = true;
        self.last_frame_time = Instant::now();

        log::info!(
            "Started NvFBC capture: {}x{} @{}fps (Format: NV12)",
            self.width,
            self.height,
            self.fps
        );

        Ok(())
    }

    async fn next_frame(&mut self) -> Result<CapturedFrame, MediaError> {
        if !self.capturing {
            return Err(MediaError::CaptureNotStarted);
        }

        if let Some(ref mut rx) = self.frame_rx {
            match rx.recv().await {
                Some(Ok(frame)) => Ok(frame),
                Some(Err(e)) => Err(MediaError::CaptureError(e)),
                None => Err(MediaError::CaptureError("Frame channel closed".into())),
            }
        } else {
            Err(MediaError::CaptureNotStarted)
        }
    }

    async fn stop(&mut self) -> Result<(), MediaError> {
        if !self.capturing {
            return Ok(());
        }

        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        if self
            .cmd_tx
            .send(NvfbcCommand::Stop { reply: reply_tx })
            .is_ok()
        {
            let _ = reply_rx.recv();
        }

        self.frame_rx = None;
        self.capturing = false;
        log::info!("Stopped NvFBC capture");
        Ok(())
    }

    async fn set_fps(&mut self, fps: u32) -> Result<(), MediaError> {
        let fps = fps.max(1);
        if fps == self.fps {
            return Ok(());
        }

        // If not currently capturing, just remember the new rate; it will be
        // applied by the next `start`.
        if !self.capturing {
            self.fps = fps;
            return Ok(());
        }

        // NvFBC bakes the sampling rate (dwSamplingRateMs = 1000 / fps) into the
        // capture session at creation time, so the only way to change the
        // capture rate at runtime is to tear the session down and recreate it.
        let config = match &self.last_config {
            Some(cfg) => {
                let mut cfg = cfg.clone();
                cfg.fps = fps;
                cfg
            }
            None => {
                // No prior config to rebuild from — record the rate and bail out.
                self.fps = fps;
                return Ok(());
            }
        };
        let display_id = self.display_id.clone();

        log::info!(
            "NvFBC set_fps: recreating capture session at {} fps (was {})",
            fps,
            self.fps
        );
        self.stop().await?;
        self.start(&display_id, &config).await
    }

    fn is_capturing(&self) -> bool {
        self.capturing
    }
}
