//! PipeWire-based screen capture for Wayland compositors (Linux).
//!
//! Uses the XDG Desktop Portal (via `ashpd`) to request screen capture permission
//! and PipeWire to receive DMA-BUF frames with zero GPU-CPU copies.

use std::rc::Rc;
use tokio::sync::mpsc;

use crate::capture::{CapturedFrame, ScreenCapture};
use crate::capture::gpu_buffer::GpuBuffer;
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
        let screencast = Screencast::new().await
            .map_err(|e| MediaError::CaptureError(format!("Failed to create Screencast portal: {}", e)))?;
            
        let session = screencast.create_session().await
            .map_err(|e| MediaError::CaptureError(format!("Failed to create portal session: {}", e)))?;
        
        log::info!("Requesting source selection...");
        let types = BitFlags::from(SourceType::Monitor);
        screencast.select_sources(
            &session,
            CursorMode::Embedded,
            types,
            false,
            None,
            PersistMode::DoNot,
        ).await
        .map_err(|e| MediaError::CaptureError(format!("Failed to select sources: {}", e)))?;
            
        log::info!("Starting screencast session (user prompt)...");
        let start_response = screencast.start(&session, &WindowIdentifier::default()).await
            .map_err(|e| MediaError::CaptureError(format!("Failed to start portal: {}", e)))?
            .response()
            .map_err(|e| MediaError::CaptureError(format!("Portal response error: {:?}", e)))?;
            
        let streams = start_response.streams();
        let stream_info = streams.first()
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
            .map_err(|e| MediaError::CaptureError(format!("Failed to spawn PipeWire thread: {}", e)))?;

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
    
    let mainloop = Rc::new(
        MainLoop::new(None)
            .map_err(|e| MediaError::CaptureError(format!("Failed to create MainLoop: {}", e)))?
    );
    let context = pipewire::context::Context::new(mainloop.as_ref())
        .map_err(|e| MediaError::CaptureError(format!("Failed to create Context: {}", e)))?;
    let core = context.connect(None)
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
            if let Some(mut buffer) = stream.dequeue_buffer() {
                let datas = buffer.datas_mut();
                if !datas.is_empty() {
                    let data = &mut datas[0];
                    let memory_type = data.type_();
                    
                    if memory_type == pipewire::spa::buffer::DataType::DmaBuf {
                        let fd = data.as_raw().fd as i32;
                        let dup_fd = nix::unistd::dup(fd).unwrap_or(-1);
                        if dup_fd >= 0 {
                            let chunk = data.chunk();
                            let offset = chunk.offset();
                            let stride = chunk.stride() as u32;
                            let size = data.as_raw().maxsize as usize;
                            
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
                                    .as_micros() as u64,
                                width: config.width,
                                height: config.height,
                                format: PixelFormat::NV12,
                                is_new_frame: true,
                            };
                            
                            let _ = tx.try_send(frame);
                        }
                    } else if memory_type == pipewire::spa::buffer::DataType::MemPtr {
                        let chunk = data.chunk();
                        let stride = chunk.stride() as u32;
                        
                        if let Some(slice) = data.data() {
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
                                    .as_micros() as u64,
                                width: config.width,
                                height: config.height,
                                format: PixelFormat::NV12,
                                is_new_frame: true,
                            };
                            
                            let _ = tx.try_send(frame);
                        }
                    }
                }
            }
        })
        .register()
        .map_err(|e| MediaError::CaptureError(format!("Failed to register process callback: {}", e)))?;
        
    // Connect stream
    let mut params = Vec::new();
    stream.connect(
        pipewire::spa::utils::Direction::Input,
        Some(node_id),
        pipewire::stream::StreamFlags::AUTOCONNECT | pipewire::stream::StreamFlags::MAP_BUFFERS,
        &mut params,
    ).map_err(|e| MediaError::CaptureError(format!("Failed to connect Stream: {}", e)))?;
    
    log::info!("Running PipeWire MainLoop...");
    mainloop.run();
    log::info!("PipeWire MainLoop exited");
    
    Ok(())
}
