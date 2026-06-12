//! PipeWire-based screen capture for Wayland compositors (Linux).
//!
//! Uses the XDG Desktop Portal (via `ashpd`) to request screen capture permission
//! and PipeWire to receive DMA-BUF frames with zero GPU-CPU copies.

use std::rc::Rc;
use tokio::sync::mpsc;

use crate::capture::gpu_buffer::GpuBuffer;
use crate::capture::{CapturedFrame, FrameCursorMeta, ScreenCapture};
use crate::error::MediaError;
use crate::types::*;

/// Capacity of the internal frame channel.
const FRAME_CHANNEL_CAPACITY: usize = 2;

/// PipeWire-based screen capture backend for Wayland sessions.
pub struct PipeWireCapture {
    frame_rx: Option<mpsc::Receiver<CapturedFrame>>,
    pw_thread: Option<std::thread::JoinHandle<()>>,
    shutdown_tx: Option<pipewire::channel::Sender<()>>,
    capturing: bool,
}

impl PipeWireCapture {
    /// Creates a new PipeWire capture instance.
    ///
    /// Fails if the current session is not Wayland.
    pub fn new() -> Result<Self, MediaError> {
        let session_type = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();
        let wayland_display = std::env::var("WAYLAND_DISPLAY").ok();

        if session_type != "wayland" && wayland_display.is_none() {
            return Err(MediaError::PlatformNotSupported(
                "PipeWire screen capture requires a Wayland session.".into(),
            ));
        }

        pipewire::init();
        log::info!("PipeWire capture backend initialized");

        Ok(Self {
            frame_rx: None,
            pw_thread: None,
            shutdown_tx: None,
            capturing: false,
        })
    }
}

#[async_trait::async_trait]
impl ScreenCapture for PipeWireCapture {
    async fn list_displays(&self) -> Result<Vec<DisplayInfo>, MediaError> {
        log::info!("PipeWire: returning default display (portal will show picker)");
        Ok(vec![DisplayInfo {
            id: "default".to_string(),
            name: "Default Display (portal will show picker)".to_string(),
            width: 1920,
            height: 1080,
            refresh_rate: 60.0,
            is_primary: true,
        }])
    }

    async fn start(&mut self, display_id: &str, config: &StreamConfig) -> Result<(), MediaError> {
        if self.capturing {
            return Err(MediaError::CaptureAlreadyStarted);
        }

        log::info!(
            "PipeWireCapture: starting capture on '{}' at {}x{} {}fps",
            display_id,
            config.width,
            config.height,
            config.fps
        );

        // Open portal session via ashpd
        use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType};
        use ashpd::desktop::PersistMode;
        use ashpd::WindowIdentifier;
        use enumflags2::BitFlags;

        log::info!("Connecting to Screencast portal...");
        let screencast = Screencast::new().await.map_err(|e| {
            MediaError::CaptureError(format!("Failed to create Screencast portal: {}", e))
        })?;

        let session = screencast.create_session().await.map_err(|e| {
            MediaError::CaptureError(format!("Failed to create portal session: {}", e))
        })?;

        log::info!("Requesting source selection...");
        let cursor_mode = if crate::capture::should_embed_host_cursor() {
            log::info!("PipeWireCapture: embedding host cursor in video stream");
            CursorMode::Embedded
        } else {
            log::info!("PipeWireCapture: hiding host cursor from video stream; browser overlay will render it");
            CursorMode::Hidden
        };
        let types = BitFlags::from(SourceType::Monitor);
        screencast
            .select_sources(
                &session,
                cursor_mode,
                types,
                false,
                None,
                PersistMode::DoNot,
            )
            .await
            .map_err(|e| MediaError::CaptureError(format!("Failed to select sources: {}", e)))?;

        log::info!("Starting screencast session (user prompt)...");
        let start_response = screencast
            .start(&session, &WindowIdentifier::default())
            .await
            .map_err(|e| MediaError::CaptureError(format!("Failed to start portal: {}", e)))?
            .response()
            .map_err(|e| MediaError::CaptureError(format!("Portal response error: {:?}", e)))?;

        let streams = start_response.streams();
        let stream_info = streams
            .first()
            .ok_or_else(|| MediaError::CaptureError("No stream returned from portal".into()))?;

        let node_id = stream_info.pipe_wire_node_id();
        log::info!("Screencast portal started. Node ID: {}", node_id);

        let (frame_tx, frame_rx) = mpsc::channel(FRAME_CHANNEL_CAPACITY);
        let (pw_sender, pw_receiver) = pipewire::channel::channel::<()>();

        let config = config.clone();

        let pw_thread = std::thread::Builder::new()
            .name("lunaris-pw".into())
            .spawn(move || {
                if let Err(e) = run_pipewire_loop(node_id, frame_tx, pw_receiver, config) {
                    log::error!("PipeWire loop error: {:?}", e);
                }
            })
            .map_err(|e| {
                MediaError::CaptureError(format!("Failed to spawn PipeWire thread: {}", e))
            })?;

        self.pw_thread = Some(pw_thread);
        self.shutdown_tx = Some(pw_sender);
        self.frame_rx = Some(frame_rx);
        self.capturing = true;

        log::info!("PipeWire screen capture started successfully");
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<CapturedFrame, MediaError> {
        if !self.capturing {
            return Err(MediaError::CaptureNotStarted);
        }

        let rx = self
            .frame_rx
            .as_mut()
            .ok_or(MediaError::CaptureNotStarted)?;

        rx.recv()
            .await
            .ok_or_else(|| MediaError::CaptureError("Frame channel closed".into()))
    }

    async fn stop(&mut self) -> Result<(), MediaError> {
        if !self.capturing {
            return Ok(());
        }

        log::info!("Stopping PipeWire capture");
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        if let Some(handle) = self.pw_thread.take() {
            let _ = handle.join();
        }

        self.frame_rx = None;
        self.capturing = false;

        log::info!("PipeWire capture stopped");
        Ok(())
    }

    fn is_capturing(&self) -> bool {
        self.capturing
    }
}

impl Drop for PipeWireCapture {
    fn drop(&mut self) {
        log::info!("PipeWireCapture dropped");
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.pw_thread.take() {
            let _ = handle.join();
        }
    }
}

fn run_pipewire_loop(
    node_id: u32,
    frame_tx: mpsc::Sender<CapturedFrame>,
    pw_receiver: pipewire::channel::Receiver<()>,
    config: StreamConfig,
) -> Result<(), MediaError> {
    use pipewire::main_loop::MainLoop;
    use pipewire::spa::pod::builder::Builder;
    use pipewire::spa::pod::Pod;

    let mainloop = Rc::new(
        MainLoop::new(None)
            .map_err(|e| MediaError::CaptureError(format!("Failed to create MainLoop: {}", e)))?,
    );
    let context = pipewire::context::Context::new(mainloop.as_ref())
        .map_err(|e| MediaError::CaptureError(format!("Failed to create Context: {}", e)))?;
    let core = context
        .connect(None)
        .map_err(|e| MediaError::CaptureError(format!("Failed to connect Context: {}", e)))?;

    let props = pipewire::properties::properties! {
        *pipewire::keys::MEDIA_TYPE => "Video",
        *pipewire::keys::MEDIA_CATEGORY => "Capture",
        *pipewire::keys::MEDIA_ROLE => "Screen",
    };

    let stream = pipewire::stream::Stream::new(&core, "lunaris-capture", props)
        .map_err(|e| MediaError::CaptureError(format!("Failed to create Stream: {}", e)))?;

    // Attach the shutdown receiver
    let mainloop_clone = mainloop.clone();
    let _receiver = pw_receiver.attach(&mainloop.loop_(), move |()| {
        mainloop_clone.quit();
    });

    let _listener = stream
        .add_local_listener_with_user_data(frame_tx)
        .process(move |stream, tx| {
            // Use raw buffer to access both video data and spa_buffer metas (cursor)
            let raw_buf = unsafe { stream.dequeue_raw_buffer() };
            if raw_buf.is_null() {
                return;
            }
            let pw_buf = unsafe { &*raw_buf };
            let spa_buf = pw_buf.buffer;
            if spa_buf.is_null() {
                return;
            }
            let spa = unsafe { &*spa_buf };

            // Extract cursor metadata from spa_buffer if available
            let cursor_meta = extract_cursor_meta(spa_buf);

            // Access the first spa_data for video frame data
            if spa.n_datas == 0 || spa.datas.is_null() {
                return;
            }
            let spa_data = unsafe { &mut *spa.datas };
            let memory_type = spa_data.type_;

            if memory_type == pipewire::spa::sys::SPA_DATA_DmaBuf {
                let fd = spa_data.fd as i32;
                let dup_fd = nix::unistd::dup(fd).unwrap_or(-1);
                if dup_fd >= 0 {
                    let chunk = unsafe { &*spa_data.chunk };
                    let offset = chunk.offset;
                    let stride = chunk.stride as u32;
                    let size = spa_data.maxsize as usize;

                    let gpu_buf = GpuBuffer::DmaBuf {
                        fd: dup_fd,
                        offset,
                        stride,
                        modifier: 0,
                        size,
                        width: config.width,
                        height: config.height,
                        fourcc: 0x34325241, // DRM_FORMAT_ARGB8888
                    };

                    let frame = CapturedFrame {
                        buffer: gpu_buf,
                        timestamp_us: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_micros()
                            as u64,
                        width: config.width,
                        height: config.height,
                        format: PixelFormat::NV12,
                        is_new_frame: true,
                        cursor: cursor_meta,
                    };

                    let _ = tx.try_send(frame);
                }
            } else if memory_type == pipewire::spa::sys::SPA_DATA_MemPtr {
                let chunk = unsafe { &*spa_data.chunk };
                let stride = chunk.stride as u32;

                if !spa_data.data.is_null() && spa_data.maxsize > 0 {
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            spa_data.data as *const u8,
                            spa_data.maxsize as usize,
                        )
                    };
                    let gpu_buf = GpuBuffer::CpuBuffer {
                        data: slice.to_vec(),
                        stride,
                        format: PixelFormat::NV12,
                        width: config.width,
                        height: config.height,
                    };

                    let frame = CapturedFrame {
                        buffer: gpu_buf,
                        timestamp_us: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_micros()
                            as u64,
                        width: config.width,
                        height: config.height,
                        format: PixelFormat::NV12,
                        is_new_frame: true,
                        cursor: cursor_meta,
                    };

                    let _ = tx.try_send(frame);
                }
            }
        })
        .register()
        .map_err(|e| {
            MediaError::CaptureError(format!("Failed to register process callback: {}", e))
        })?;

    // Build ParamMeta pod to request SPA_META_Cursor from the compositor
    let mut meta_pod_data = Vec::with_capacity(256);
    let cursor_meta_size = std::mem::size_of::<pipewire::spa::sys::spa_meta_cursor>() as i32;
    unsafe {
        let mut builder = Builder::new(&mut meta_pod_data);
        let mut frame = std::mem::MaybeUninit::<pipewire::spa::sys::spa_pod_frame>::uninit();
        builder
            .push_object(
                &mut frame,
                pipewire::spa::sys::SPA_TYPE_OBJECT_ParamMeta,
                0,
            )
            .map_err(|e| {
                MediaError::CaptureError(format!("Failed to push ParamMeta object: {}", e))
            })?;
        // SPA_PARAM_META_type = 1
        builder
            .add_prop(pipewire::spa::sys::SPA_PARAM_META_type, 0)
            .map_err(|e| {
                MediaError::CaptureError(format!("Failed to add meta type prop: {}", e))
            })?;
        builder
            .add_id(pipewire::spa::utils::Id(
                pipewire::spa::sys::SPA_META_Cursor,
            ))
            .map_err(|e| {
                MediaError::CaptureError(format!("Failed to add SPA_META_Cursor id: {}", e))
            })?;
        // SPA_PARAM_META_size = 2
        builder
            .add_prop(pipewire::spa::sys::SPA_PARAM_META_size, 0)
            .map_err(|e| {
                MediaError::CaptureError(format!("Failed to add meta size prop: {}", e))
            })?;
        builder.add_int(cursor_meta_size).map_err(|e| {
            MediaError::CaptureError(format!("Failed to add cursor meta size: {}", e))
        })?;
        let mut frame_val = frame.assume_init();
        builder.pop(&mut frame_val);
    }
    let meta_pod = unsafe {
        Pod::from_raw(meta_pod_data.as_ptr() as *const pipewire::spa::sys::spa_pod)
    };
    let mut params = vec![meta_pod];

    // Connect stream with cursor metadata request
    stream
        .connect(
            pipewire::spa::utils::Direction::Input,
            Some(node_id),
            pipewire::stream::StreamFlags::AUTOCONNECT | pipewire::stream::StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .map_err(|e| MediaError::CaptureError(format!("Failed to connect Stream: {}", e)))?;

    log::info!("Running PipeWire MainLoop (with cursor metadata)...");
    mainloop.run();
    log::info!("PipeWire MainLoop exited");

    Ok(())
}

/// Extract cursor metadata from a PipeWire spa_buffer, if present.
///
/// Returns `Some(FrameCursorMeta)` when the compositor provided `SPA_META_Cursor`
/// data in the buffer. Returns `None` if the metadata is absent or invalid.
fn extract_cursor_meta(
    spa_buf: *mut pipewire::spa::sys::spa_buffer,
) -> Option<FrameCursorMeta> {
    if spa_buf.is_null() {
        return None;
    }
    unsafe {
        let meta_ptr = pipewire::spa::sys::spa_buffer_find_meta(
            spa_buf,
            pipewire::spa::sys::SPA_META_Cursor,
        );
        if meta_ptr.is_null() {
            return None;
        }
        let meta = &*meta_ptr;
        if meta.data.is_null() || meta.size == 0 {
            return None;
        }
        let cursor = &*(meta.data as *const pipewire::spa::sys::spa_meta_cursor);
        // id == 0 means no new cursor data this frame
        if cursor.id == 0 {
            return None;
        }

        let x = cursor.position.x;
        let y = cursor.position.y;
        let hotspot_x = cursor.hotspot.x;
        let hotspot_y = cursor.hotspot.y;

        // Check if bitmap data is available
        let cursor_meta_size = std::mem::size_of::<pipewire::spa::sys::spa_meta_cursor>() as u32;
        let image = if cursor.bitmap_offset >= cursor_meta_size {
            let bitmap_ptr = (meta.data as *const u8).add(cursor.bitmap_offset as usize)
                as *const pipewire::spa::sys::spa_meta_bitmap;
            let bitmap = &*bitmap_ptr;
            let bitmap_meta_size =
                std::mem::size_of::<pipewire::spa::sys::spa_meta_bitmap>() as u32;

            if bitmap.offset >= bitmap_meta_size
                && bitmap.size.width > 0
                && bitmap.size.height > 0
            {
                let pixel_data =
                    (bitmap_ptr as *const u8).add(bitmap.offset as usize);
                let width = bitmap.size.width;
                let height = bitmap.size.height;
                let stride = bitmap.stride as u32;
                let expected_len = (stride * height) as usize;

                // Convert pixel data to RGBA based on the spa_video_format
                let rgba = convert_cursor_bitmap_to_rgba(
                    pixel_data,
                    expected_len,
                    bitmap.format,
                    width,
                    height,
                    stride,
                );

                rgba.map(|data| CursorImage {
                    width,
                    height,
                    hotspot_x: hotspot_x.max(0) as u32,
                    hotspot_y: hotspot_y.max(0) as u32,
                    rgba_data: data,
                })
            } else {
                None
            }
        } else {
            None
        };

        Some(FrameCursorMeta {
            x,
            y,
            hotspot_x,
            hotspot_y,
            visible: true,
            image,
        })
    }
}

/// Convert cursor bitmap pixel data from a spa_video_format to RGBA.
fn convert_cursor_bitmap_to_rgba(
    raw: *const u8,
    raw_len: usize,
    format: u32,
    width: u32,
    height: u32,
    stride: u32,
) -> Option<Vec<u8>> {
    use pipewire::spa::sys::*;
    let expected = (stride * height) as usize;
    if raw_len < expected {
        return None;
    }
    let src = unsafe { std::slice::from_raw_parts(raw, expected) };
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);

    for row in 0..height {
        let row_offset = (row * stride) as usize;
        for col in 0..width {
            let px_offset = row_offset + (col * 4) as usize;
            if px_offset + 4 > src.len() {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            let (r, g, b, a) = match format {
                SPA_VIDEO_FORMAT_RGBA => (
                    src[px_offset],
                    src[px_offset + 1],
                    src[px_offset + 2],
                    src[px_offset + 3],
                ),
                SPA_VIDEO_FORMAT_BGRA => (
                    src[px_offset + 2],
                    src[px_offset + 1],
                    src[px_offset],
                    src[px_offset + 3],
                ),
                SPA_VIDEO_FORMAT_ARGB => (
                    src[px_offset + 1],
                    src[px_offset + 2],
                    src[px_offset + 3],
                    src[px_offset],
                ),
                #[allow(non_upper_case_globals)]
                SPA_VIDEO_FORMAT_BGRx => (
                    src[px_offset + 2],
                    src[px_offset + 1],
                    src[px_offset],
                    0xFF,
                ),
                #[allow(non_upper_case_globals)]
                SPA_VIDEO_FORMAT_RGBx => (
                    src[px_offset],
                    src[px_offset + 1],
                    src[px_offset + 2],
                    0xFF,
                ),
                // Default: treat as RGBA
                _ => (
                    src[px_offset],
                    src[px_offset + 1],
                    src[px_offset + 2],
                    src[px_offset + 3],
                ),
            };
            rgba.extend_from_slice(&[r, g, b, a]);
        }
    }

    Some(rgba)
}
