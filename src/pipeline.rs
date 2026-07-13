//! Media pipeline orchestration.
//!
//! [`MediaPipeline`] ties together the capture, encode, audio, and cursor
//! subsystems into a single async event loop. Consumers receive
//! [`MediaEvent`]s via a channel and can send [`PipelineCommand`]s to control
//! the pipeline at runtime.
//!
//! # Design
//!
//! ```text
//!  ┌────────────────────────────────────────────────────────┐
//!  │                   MediaPipeline::run()                 │
//!  │                                                        │
//!  │  ┌─ main select! loop ──────────────────────────────┐  │
//!  │  │  capture.next_frame() ──▶ encoder.encode()       │  │
//!  │  │                          ──▶ send VideoFrame     │  │
//!  │  │  commands.recv()    ──▶ handle command            │  │
//!  │  └──────────────────────────────────────────────────┘  │
//!  │                                                        │
//!  │  ┌─ audio task ─────────────────────────────────────┐  │
//!  │  │  audio.next_frame() ──▶ send AudioFrame          │  │
//!  │  └──────────────────────────────────────────────────┘  │
//!  │                                                        │
//!  │  ┌─ cursor task ────────────────────────────────────┐  │
//!  │  │  cursor.get_cursor_state() at 60Hz               │  │
//!  │  │  ──▶ send CursorUpdate (on change)               │  │
//!  │  └──────────────────────────────────────────────────┘  │
//!  └────────────────────────────────────────────────────────┘
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::audio;
use crate::capture;
use crate::cursor;
#[cfg(any(target_os = "linux", target_os = "windows"))]
use crate::capture::virtual_display::VirtualDisplayHandle;
use crate::encode;
use crate::encode::EncoderConfig;
use crate::error::MediaError;
use crate::types::*;

/// Events emitted by the running media pipeline.
#[derive(Debug, Clone)]
pub enum MediaEvent {
    /// The video encoder backend has been initialized.
    EncoderStarted {
        encoder: EncoderInfo,
        gpu_name: Option<String>,
        requested_encoder: Option<String>,
    },
    /// A new encoded video frame is available.
    VideoFrame(EncodedVideoFrame),
    /// A new encoded audio frame is available.
    AudioFrame(EncodedAudioFrame),
    /// The cursor state has changed.
    CursorUpdate(CursorState),
    /// The pipeline has started successfully.
    Started,
    /// The pipeline has stopped.
    Stopped,
    /// A non-fatal error occurred; the pipeline continues running.
    Error(String),
}

/// Commands that can be sent to a running pipeline.
#[derive(Debug, Clone)]
pub enum PipelineCommand {
    /// Request an immediate keyframe from the video encoder.
    RequestKeyframe,
    /// Change the target video bitrate.
    SetBitrate(u32),
    /// Change the target video FPS.
    SetFps(u32),
    /// Stop the pipeline gracefully.
    Stop,
}

/// Orchestrates capture, encoding, audio, and cursor subsystems.
///
/// Create a pipeline with [`MediaPipeline::new`], which returns the pipeline
/// instance, a receiver for [`MediaEvent`]s, and a sender for
/// [`PipelineCommand`]s.
///
/// # Usage
///
/// ```no_run
/// use lunaris_media::pipeline::{MediaPipeline, MediaEvent, PipelineCommand};
/// use lunaris_media::types::StreamConfig;
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let config = StreamConfig::default();
/// let (pipeline, mut events, commands) = MediaPipeline::new(config);
///
/// // Spawn the pipeline
/// tokio::spawn(async move {
///     pipeline.run("default").await.ok();
/// });
///
/// // Process events
/// while let Some(event) = events.recv().await {
///     match event {
///         MediaEvent::VideoFrame(f) => log::info!("video: {} bytes", f.data.len()),
///         MediaEvent::Stopped => break,
///         _ => {}
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct MediaPipeline {
    config: StreamConfig,
    command_rx: mpsc::Receiver<PipelineCommand>,
    event_tx: mpsc::Sender<MediaEvent>,
    audio_shutdown: Arc<AtomicBool>,
}

impl MediaPipeline {
    /// Create a new media pipeline.
    ///
    /// Returns `(pipeline, event_receiver, command_sender)`. The caller should
    /// spawn [`run`](Self::run) on a tokio task and consume events from the
    /// receiver.
    pub fn new(
        config: StreamConfig,
    ) -> (
        Self,
        mpsc::Receiver<MediaEvent>,
        mpsc::Sender<PipelineCommand>,
    ) {
        let (event_tx, event_rx) = mpsc::channel(64);
        let (command_tx, command_rx) = mpsc::channel(16);
        let audio_shutdown = Arc::new(AtomicBool::new(false));

        let pipeline = Self {
            config,
            command_rx,
            event_tx,
            audio_shutdown,
        };

        (pipeline, event_rx, command_tx)
    }

    /// Run the pipeline, capturing from `display_id` until stopped.
    ///
    /// This method takes ownership and runs until a [`PipelineCommand::Stop`]
    /// is received or an unrecoverable error occurs.
    ///
    /// # Pipeline lifecycle
    ///
    /// 1. Create and initialize all subsystem components (capture, encoder,
    ///    audio, cursor).
    /// 2. Start screen capture on the requested display.
    /// 3. Spawn a blocking task for audio capture (cpal streams are `!Send`).
    /// 4. Spawn an async task for cursor tracking at ~60 Hz.
    /// 5. Enter the main `select!` loop:
    ///    - Capture a video frame → encode → emit [`MediaEvent::VideoFrame`].
    ///    - Receive and handle [`PipelineCommand`]s.
    /// 6. On shutdown: flush encoder, stop capture, abort background tasks,
    ///    and log statistics.
    pub async fn run(mut self, display_id: &str) -> Result<(), MediaError> {
        log::info!(
            "Starting media pipeline for display '{}' at {}x{} {}fps",
            display_id,
            self.config.width,
            self.config.height,
            self.config.fps,
        );

        // 1. Create components
        let mut capture = capture::create_screen_capture()?;
        let mut encoder = encode::create_encoder()?;
        let mut audio_capture = audio::create_audio_capture()?;
        let mut cursor_capture = cursor::create_cursor_capture()?;

        let requested_encoder = self.config.preferred_encoder.clone();
        let preferred_encoder = requested_encoder.as_deref().unwrap_or("auto");
        let preferred_hw = StreamConfig::parse_encoder_preference(preferred_encoder);
        let force_ffmpeg = StreamConfig::encoder_prefers_ffmpeg(preferred_encoder);

        let use_gdi_only = std::env::var("LUNARIS_USE_GDI")
            .map(|val| val == "1" || val.to_lowercase() == "true")
            .unwrap_or(false);

        let is_hw_preferred = preferred_hw.map_or(true, |hw| hw != HwAccelType::Software);

        if cfg!(target_os = "windows") && !use_gdi_only && is_hw_preferred {
            log::info!("Zero-copy GPU pipeline requested. Setting LUNARIS_ZERO_COPY=1");
            std::env::set_var("LUNARIS_ZERO_COPY", "1");
        }

        // 2. Start capture
        #[allow(unused_mut)]
        let mut capture_display_id = display_id.to_string();

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        let mut _virtual_display: Option<Box<dyn VirtualDisplayHandle>> = None;
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        let mut _virtual_display: Option<()> = None;
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if self.config.virtual_display {
            match crate::capture::virtual_display::create_virtual_display(
                self.config.width,
                self.config.height,
                self.config.fps,
            ) {
                Ok(vd) => {
                    capture_display_id = vd.display_id().to_string();
                    log::info!("Created virtual display: {}", capture_display_id);
                    _virtual_display = Some(vd);
                }
                Err(e) => {
                    log::warn!("Failed to create virtual display: {}", e);
                }
            }
        }

        #[cfg(target_os = "linux")]
        if self.config.fps > 60 {
            if let Ok(displays) = capture.list_displays().await {
                if let Some(display) = displays
                    .iter()
                    .find(|display| display.id == capture_display_id)
                    .or_else(|| displays.iter().find(|display| display.is_primary))
                {
                    if (display.refresh_rate as u32) < self.config.fps {
                        log::info!(
                            "Target FPS {} > display refresh rate {}, attempting to change display '{}'",
                            self.config.fps,
                            display.refresh_rate,
                            display.id
                        );
                        Self::try_set_refresh_rate(&display.id, self.config.fps);
                    }
                }
            }
        }

        capture.start(&capture_display_id, &self.config).await?;
        log::info!("Screen capture started on '{}'", capture_display_id);

        let (d3d11_device, d3d11_context) = if cfg!(target_os = "windows") {
            if let Some((device, context)) = capture.get_d3d11_device() {
                log::info!(
                    "Retrieved shared D3D11 device from capture backend (device={:#x}, context={:#x})",
                    device,
                    context
                );
                (Some(device), Some(context))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // 3. Initialize encoder
        encoder.initialize(&EncoderConfig {
            codec: self.config.codec,
            width: self.config.width,
            height: self.config.height,
            fps: self.config.fps,
            bitrate_kbps: self.config.bitrate_kbps,
            low_latency: true,
            // IDR every 4 s at 60 fps (240 frames). Was 0 (=fps=60=1 s),
            // which caused a bandwidth spike every second from the large IDR
            // frame, producing visible micro-stutter. PLI from the browser
            // still triggers immediate IDR on demand.
            keyframe_interval: 240,
            preferred_hw,
            force_ffmpeg,
            d3d11_device,
            d3d11_context,
        })?;
        let encoder_info = encoder.encoder_info();
        let gpu_name = encode::describe_host_gpu(d3d11_device);
        log::info!(
            "Encoder initialized: {} ({}) on {}",
            encoder_info.name,
            encoder_info.hw_type,
            gpu_name.as_deref().unwrap_or("unknown GPU")
        );
        let _ = self
            .event_tx
            .send(MediaEvent::EncoderStarted {
                encoder: encoder_info,
                gpu_name,
                requested_encoder,
            })
            .await;

        // 4. Start audio on a separate blocking task (cpal is !Send for Stream)
        let mut audio_config = AudioCaptureConfig::default();
        audio_config.frame_size_ms = 10;
        let audio_event_tx = self.event_tx.clone();
        let audio_shutdown = self.audio_shutdown.clone();
        let audio_shutdown_clone = audio_shutdown.clone();
        let audio_handle = tokio::task::spawn_blocking(move || {
            if let Err(e) = audio_capture.start(&audio_config) {
                log::warn!("Audio capture failed to start: {}", e);
                return;
            }
            let mut audio_drop_count: u64 = 0;
            loop {
                // Check shutdown flag before blocking on next_frame
                if audio_shutdown_clone.load(Ordering::Relaxed) {
                    log::info!("Audio task received shutdown signal");
                    audio_capture.stop().ok();
                    break;
                }
                match audio_capture.next_frame() {
                    Ok(frame) => {
                        if let Err(e) = audio_event_tx.try_send(MediaEvent::AudioFrame(frame)) {
                            if matches!(e, mpsc::error::TrySendError::Closed(_)) {
                                break; // channel closed
                            } else {
                                audio_drop_count += 1;
                                // Rate-limit warning to prevent log spam from freezing the agent
                                if audio_drop_count == 1 || audio_drop_count % 100 == 0 {
                                    log::warn!(
                                        "Audio event channel full, dropped {} audio frames total",
                                        audio_drop_count
                                    );
                                }
                            }
                        } else {
                            // Reset counter on successful send
                            if audio_drop_count > 0 {
                                log::info!(
                                    "Audio channel recovered after dropping {} frames",
                                    audio_drop_count
                                );
                                audio_drop_count = 0;
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Audio capture error: {}", e);
                        break;
                    }
                }
            }
        });

        // 5. Start cursor tracking (~60 Hz polling)
        cursor_capture.start().ok(); // Non-fatal if cursor capture fails
        let cursor_event_tx = self.event_tx.clone();
        let cursor_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(16)); // ~60Hz
            let mut last_state: Option<CursorState> = None;
            loop {
                interval.tick().await;
                if let Ok(state) = cursor_capture.get_cursor_state() {
                    if last_state.as_ref() != Some(&state) {
                        last_state = Some(state.clone());
                        if let Err(e) = cursor_event_tx.try_send(MediaEvent::CursorUpdate(state)) {
                            if matches!(e, mpsc::error::TrySendError::Closed(_)) {
                                break; // channel closed
                            } else {
                                log::warn!("Cursor event channel full, dropping cursor update");
                            }
                        }
                    }
                }
            }
        });

        // Notify started
        let _ = self.event_tx.send(MediaEvent::Started).await;

        // 6. Main loop — video capture + encode + command handling
        let mut total_frames: u64 = 0;
        let mut total_bytes: u64 = 0;
        let mut skipped_frames: u64 = 0;
        let mut dropped_frames: u64 = 0;
        let start_time = Instant::now();
        let event_capacity = 64usize; // must match mpsc::channel capacity above
        let mut target_interval = Duration::from_nanos(1_000_000_000 / self.config.fps as u64);
        let mut frame_ticker = tokio::time::interval(target_interval);
        frame_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Constant-FPS output pads the stream with duplicate frames so it always
        // reaches the target FPS, instead of only emitting frames when the screen
        // changes. Default ON so the stream honours the client's requested rate;
        // set LUNARIS_CONSTANT_FPS=0 to fall back to change-driven encoding (lower
        // bitrate/CPU on static screens).
        let constant_fps = std::env::var("LUNARIS_CONSTANT_FPS")
            .map(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(true);
        log::info!("Pipeline constant-FPS output: {}", constant_fps);

        let mut last_sent_time = Instant::now() - Duration::from_secs(5);
        let mut keyframe_requested = true; // Force first frame immediately
        let mut last_frame: Option<crate::capture::CapturedFrame> = None;
        let mut metrics_started = Instant::now();
        let mut metrics_ticks: u64 = 0;
        let mut metrics_new_captures: u64 = 0;
        let mut metrics_encode_attempts: u64 = 0;
        let mut metrics_encoded_frames: u64 = 0;
        let mut metrics_sent_frames: u64 = 0;
        let mut metrics_dropped_frames: u64 = 0;
        let mut metrics_bytes: u64 = 0;
        let mut metrics_encode_time = Duration::ZERO;

        loop {
            tokio::select! {
                // Video: capture → encode → send
                _ = frame_ticker.tick() => {
                    metrics_ticks += 1;
                    let frame_result = capture.next_frame().await;
                    // Backpressure: skip encode if downstream is congested.
                    // This prevents unbounded queue growth and keeps latency low.
                    let queue_len = event_capacity - self.event_tx.capacity();
                    if queue_len >= 56 {
                        // Downstream can't keep up — skip this frame.
                        // Threshold at ~87% of 64 (was 75%=48) gives the encoder
                        // more breathing room before we start dropping frames,
                        // reducing visible stutter during high-motion scenes.
                        skipped_frames += 1;
                        if skipped_frames % 60 == 1 {
                            log::warn!("Pipeline backpressure: skipped {} frames (queue_len={})",
                                skipped_frames, queue_len);
                        }
                        continue;
                    }

                    match frame_result {
                        Ok(captured) => {
                            let is_empty = matches!(&captured.buffer, crate::capture::gpu_buffer::GpuBuffer::CpuBuffer { data, .. } if data.is_empty());
                            let is_new_frame = !is_empty && captured.is_new_frame;
                            if is_new_frame {
                                metrics_new_captures += 1;
                            }

                            if !is_empty {
                                last_frame = Some(captured);
                            }

                            let frame = match &last_frame {
                                Some(f) => f,
                                None => continue, // No frame captured yet
                            };

                            let should_encode = constant_fps
                                || is_new_frame
                                || keyframe_requested
                                || last_sent_time.elapsed() >= Duration::from_millis(500);

                            if !should_encode {
                                continue;
                            }

                            keyframe_requested = false;
                            last_sent_time = Instant::now();

                            let pts = start_time.elapsed().as_micros() as u64;
                            metrics_encode_attempts += 1;
                            let encode_started = Instant::now();
                            match encoder.encode(&frame.buffer, pts) {
                                Ok(encoded_frames) => {
                                    metrics_encode_time += encode_started.elapsed();
                                    for ef in encoded_frames {
                                        total_frames += 1;
                                        total_bytes += ef.data.len() as u64;
                                        metrics_encoded_frames += 1;
                                        metrics_bytes += ef.data.len() as u64;
                                        if self.event_tx.try_send(MediaEvent::VideoFrame(ef)).is_err() {
                                            dropped_frames += 1;
                                            metrics_dropped_frames += 1;
                                            log::warn!("Video event channel full, dropping frame");
                                        } else {
                                            metrics_sent_frames += 1;
                                        }
                                    }
                                }
                                Err(e) => {
                                    metrics_encode_time += encode_started.elapsed();
                                    log::error!("Encode error: {}", e);
                                    let _ = self.event_tx.try_send(MediaEvent::Error(e.to_string()));
                                }
                            }
                        }
                        Err(e) => {
                            log::error!("Capture error: {}", e);
                            let _ = self.event_tx.try_send(MediaEvent::Error(e.to_string()));
                        }
                    }
                    let metrics_elapsed = metrics_started.elapsed();
                    if metrics_elapsed >= Duration::from_secs(1) {
                        let secs = metrics_elapsed.as_secs_f64();
                        let avg_encode_ms = if metrics_encode_attempts > 0 {
                            metrics_encode_time.as_secs_f64() * 1000.0 / metrics_encode_attempts as f64
                        } else {
                            0.0
                        };
                        let queue_len = event_capacity - self.event_tx.capacity();
                        log::debug!(
                            "Pipeline metrics: ticks={:.1}/s capture_new={:.1}/s encode_attempts={:.1}/s encoded={:.1}/s sent={:.1}/s dropped={} bitrate={:.2}Mbps queue={} avg_encode={:.2}ms constant_fps={}",
                            metrics_ticks as f64 / secs,
                            metrics_new_captures as f64 / secs,
                            metrics_encode_attempts as f64 / secs,
                            metrics_encoded_frames as f64 / secs,
                            metrics_sent_frames as f64 / secs,
                            metrics_dropped_frames,
                            (metrics_bytes as f64 * 8.0 / secs) / 1_000_000.0,
                            queue_len,
                            avg_encode_ms,
                            constant_fps
                        );
                        metrics_started = Instant::now();
                        metrics_ticks = 0;
                        metrics_new_captures = 0;
                        metrics_encode_attempts = 0;
                        metrics_encoded_frames = 0;
                        metrics_sent_frames = 0;
                        metrics_dropped_frames = 0;
                        metrics_bytes = 0;
                        metrics_encode_time = Duration::ZERO;
                    }
                    tokio::task::yield_now().await;
                }

                // Commands
                Some(cmd) = self.command_rx.recv() => {
                    match cmd {
                        PipelineCommand::RequestKeyframe => {
                            log::debug!("Keyframe requested");
                            encoder.request_keyframe();
                            keyframe_requested = true;
                        }
                        PipelineCommand::SetBitrate(kbps) => {
                            log::debug!("Setting bitrate to {} kbps", kbps);
                            if let Err(e) = encoder.set_bitrate(kbps) {
                                log::warn!("Failed to set bitrate: {}", e);
                            }
                            keyframe_requested = true;
                        }
                        PipelineCommand::SetFps(new_fps) => {
                            let new_fps = new_fps.clamp(1, 240);
                            log::info!("Changing target FPS from {} to {}", self.config.fps, new_fps);
                            self.config.fps = new_fps;
                            if let Err(e) = encoder.set_fps(new_fps) {
                                log::warn!("Failed to update encoder FPS: {}", e);
                            }
                            // On Linux, capturing above the display's refresh rate is
                            // impossible unless we raise it first — the compositor
                            // simply won't produce new frames any faster.
                            #[cfg(target_os = "linux")]
                            if new_fps > 60 {
                                if let Ok(displays) = capture.list_displays().await {
                                    if let Some(display) = displays
                                        .iter()
                                        .find(|display| display.is_primary)
                                        .or_else(|| displays.first())
                                    {
                                        if (display.refresh_rate as u32) < new_fps {
                                            log::info!(
                                                "Target FPS {} > display refresh rate {}, attempting to change display '{}'",
                                                new_fps,
                                                display.refresh_rate,
                                                display.id
                                            );
                                            Self::try_set_refresh_rate(&display.id, new_fps);
                                        }
                                    }
                                }
                            }
                            // Reconfigure the capture backend so it actually captures
                            // at the new rate. Some backends (e.g. NvFBC) bake the
                            // capture pacing into the session and must recreate it;
                            // without this the stream stays stuck at the initial FPS.
                            if let Err(e) = capture.set_fps(new_fps).await {
                                log::warn!("Failed to update capture FPS: {}", e);
                            }
                            target_interval = Duration::from_nanos(1_000_000_000 / new_fps as u64);
                            frame_ticker = tokio::time::interval(target_interval);
                            frame_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                            keyframe_requested = true;
                        }
                        PipelineCommand::Stop => {
                            log::info!("Stop command received");
                            break;
                        }
                    }
                }
            }
        }

        // 7. Cleanup
        log::info!("Pipeline shutting down...");

        // Flush encoder
        if let Ok(remaining) = encoder.flush() {
            for ef in remaining {
                let _ = self.event_tx.try_send(MediaEvent::VideoFrame(ef));
            }
        }
        encoder.shutdown();

        // Stop capture
        capture.stop().await.ok();

        // Signal audio shutdown and wait for the task to finish.
        // NOTE: `abort()` does NOT stop `spawn_blocking` tasks — the thread keeps running.
        // We must signal via the flag and wait for the task to self-terminate.
        audio_shutdown.store(true, Ordering::Relaxed);
        // Give the audio task up to 500ms to notice the flag and exit cleanly.
        match tokio::time::timeout(Duration::from_millis(500), audio_handle).await {
            Ok(_) => log::info!("Audio task stopped cleanly"),
            Err(_) => {
                log::warn!("Audio task did not stop within 500ms (will be cleaned up on drop)")
            }
        }
        cursor_handle.abort();

        // Stats
        let elapsed = start_time.elapsed();
        log::info!(
            "Pipeline stopped. {} frames ({} skipped, {} dropped), {:.2} MB, {:.1} FPS, {:.1}s",
            total_frames,
            skipped_frames,
            dropped_frames,
            total_bytes as f64 / 1_048_576.0,
            total_frames as f64 / elapsed.as_secs_f64(),
            elapsed.as_secs_f64(),
        );

        let _ = self.event_tx.send(MediaEvent::Stopped).await;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn try_set_refresh_rate(display_id: &str, target_fps: u32) {
        // xrandr / the NVIDIA driver will happily accept `--rate N` and return a
        // success exit code while silently clamping to the panel's real maximum.
        // So: only attempt the change when the output actually advertises a mode
        // at ~target_fps, and verify the rate that was truly applied afterwards.
        if !Self::output_advertises_rate(display_id, target_fps) {
            log::warn!(
                "Display '{}' does not advertise a ~{}Hz mode; leaving its refresh \
                 rate unchanged. Capture stays capped at the panel's real refresh \
                 rate — use a genuine high-refresh output (or a forced-EDID / \
                 virtual display) to exceed it.",
                display_id,
                target_fps
            );
            return;
        }

        let target = target_fps.to_string();
        if let Err(e) = std::process::Command::new("xrandr")
            .args(["--output", display_id, "--rate", target.as_str()])
            .output()
        {
            log::warn!(
                "Failed to run xrandr for {}Hz refresh-rate change on display {}: {}",
                target_fps,
                display_id,
                e
            );
            return;
        }

        // Read back the rate the driver actually applied instead of trusting the
        // exit code.
        match Self::active_refresh_rate(display_id) {
            Some(actual) if (actual - target_fps as f64).abs() <= 1.5 => {
                log::info!(
                    "Changed display {} refresh rate to {}Hz",
                    display_id,
                    target_fps
                );
            }
            Some(actual) => {
                log::warn!(
                    "Requested {}Hz on display {} but the driver applied {:.2}Hz — \
                     capture stays capped at ~{:.0}fps.",
                    target_fps,
                    display_id,
                    actual,
                    actual
                );
            }
            None => {
                log::warn!(
                    "Set {}Hz on display {} but could not verify the applied rate.",
                    target_fps,
                    display_id
                );
            }
        }
    }

    /// Returns `true` if `xrandr` lists any mode for `display_id` whose refresh
    /// rate is within ~1.5 Hz of `target_fps`.
    #[cfg(target_os = "linux")]
    fn output_advertises_rate(display_id: &str, target_fps: u32) -> bool {
        Self::xrandr_output_rates(display_id)
            .iter()
            .any(|rate| (rate - target_fps as f64).abs() <= 1.5)
    }

    /// Returns the currently-active refresh rate (the one marked with `*`) for
    /// `display_id`, parsed from `xrandr --query`.
    #[cfg(target_os = "linux")]
    fn active_refresh_rate(display_id: &str) -> Option<f64> {
        Self::xrandr_rates(display_id, true).into_iter().next()
    }

    /// Returns all refresh rates advertised for `display_id`.
    #[cfg(target_os = "linux")]
    fn xrandr_output_rates(display_id: &str) -> Vec<f64> {
        Self::xrandr_rates(display_id, false)
    }

    /// Parse refresh rates for a single output from `xrandr --query`. When
    /// `active_only` is set, only the rate marked with `*` is returned.
    #[cfg(target_os = "linux")]
    fn xrandr_rates(display_id: &str, active_only: bool) -> Vec<f64> {
        let mut rates = Vec::new();
        let output = match std::process::Command::new("xrandr").arg("--query").output() {
            Ok(o) => o,
            Err(_) => return rates,
        };
        let text = String::from_utf8_lossy(&output.stdout);
        let mut in_block = false;
        for line in text.lines() {
            // Output header lines start at column 0 (e.g. "DP-1 connected ...").
            let is_header = !line.is_empty() && !line.starts_with(char::is_whitespace);
            if is_header {
                in_block = line.split_whitespace().next() == Some(display_id);
                continue;
            }
            if !in_block {
                continue;
            }
            // Mode lines are indented: "   1920x1080  60.00*+  59.94  50.00".
            for tok in line.split_whitespace() {
                if active_only && !tok.contains('*') {
                    continue;
                }
                let cleaned = tok.trim_end_matches(|c| c == '*' || c == '+');
                if let Ok(rate) = cleaned.parse::<f64>() {
                    rates.push(rate);
                }
            }
        }
        rates
    }
}

impl Drop for MediaPipeline {
    fn drop(&mut self) {
        log::info!("MediaPipeline dropped — signaling audio shutdown");
        self.audio_shutdown.store(true, Ordering::Relaxed);
    }
}
