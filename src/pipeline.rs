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
use crate::encode;
use crate::encode::EncoderConfig;
use crate::error::MediaError;
use crate::types::*;

/// Events emitted by the running media pipeline.
#[derive(Debug, Clone)]
pub enum MediaEvent {
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
    ) -> (Self, mpsc::Receiver<MediaEvent>, mpsc::Sender<PipelineCommand>) {
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

        // 2. Initialize encoder
        encoder.initialize(&EncoderConfig {
            codec: self.config.codec,
            width: self.config.width,
            height: self.config.height,
            fps: self.config.fps,
            bitrate_kbps: self.config.bitrate_kbps,
            low_latency: true,
            keyframe_interval: 0,
            preferred_hw: StreamConfig::parse_encoder_preference(
                self.config.preferred_encoder.as_deref().unwrap_or("auto")
            ),
        })?;
        log::info!("Encoder initialized: {}", encoder.encoder_info().name);

        // 3. Start capture
        capture.start(display_id, &self.config).await?;
        log::info!("Screen capture started");

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
                                    log::warn!("Audio event channel full, dropped {} audio frames total", audio_drop_count);
                                }
                            }
                        } else {
                            // Reset counter on successful send
                            if audio_drop_count > 0 {
                                log::info!("Audio channel recovered after dropping {} frames", audio_drop_count);
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
            let mut last_pos = (0i32, 0i32);
            loop {
                interval.tick().await;
                if let Ok(state) = cursor_capture.get_cursor_state() {
                    if state.x != last_pos.0 || state.y != last_pos.1 || state.image.is_some() {
                        last_pos = (state.x, state.y);
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
        let start_time = Instant::now();
        let event_capacity = 64usize; // must match mpsc::channel capacity above
        let mut last_sent_time = Instant::now();
        let mut target_interval = Duration::from_nanos(1_000_000_000 / self.config.fps as u64);
        let mut force_next_frame = true;

        loop {
            tokio::select! {
                // Video: capture → encode → send
                frame_result = capture.next_frame() => {
                    // Backpressure: skip encode if downstream is congested.
                    // This prevents unbounded queue growth and keeps latency low.
                    let queue_len = event_capacity - self.event_tx.capacity();
                    if queue_len >= 48 {
                        // Downstream can't keep up — skip this frame
                        skipped_frames += 1;
                        if skipped_frames % 60 == 1 {
                            log::warn!("Pipeline backpressure: skipped {} frames (queue_len={})",
                                skipped_frames, queue_len);
                        }
                        continue;
                    }

                    match frame_result {
                        Ok(captured) => {
                            let now = Instant::now();
                            let elapsed = last_sent_time.elapsed();

                            // Duplicate frame discarding (bypassed if keyframe is forced)
                            if !force_next_frame {
                                if !captured.is_new_frame {
                                    if elapsed < Duration::from_millis(33) {
                                        // 33ms = 30fps minimum floor for static/duplicate frames
                                        continue;
                                    }
                                } else {
                                    // Frame rate pacing (e.g. game running faster than target FPS)
                                    // Allow 50% margin to prevent timing jitters from causing dropped frames.
                                    if elapsed < (target_interval * 5 / 10) {
                                        continue;
                                    }
                                }
                            }

                            force_next_frame = false;
                            last_sent_time = now;
                            let pts = captured.timestamp_us;
                            match encoder.encode(&captured.buffer, pts) {
                                Ok(encoded_frames) => {
                                    for ef in encoded_frames {
                                        total_frames += 1;
                                        total_bytes += ef.data.len() as u64;
                                        if self.event_tx.try_send(MediaEvent::VideoFrame(ef)).is_err() {
                                            log::warn!("Video event channel full, dropping frame");
                                        }
                                    }
                                }
                                Err(e) => {
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
                    tokio::task::yield_now().await;
                }

                // Commands
                Some(cmd) = self.command_rx.recv() => {
                    match cmd {
                        PipelineCommand::RequestKeyframe => {
                            log::debug!("Keyframe requested");
                            encoder.request_keyframe();
                            force_next_frame = true;
                        }
                        PipelineCommand::SetBitrate(kbps) => {
                            log::debug!("Setting bitrate to {} kbps", kbps);
                            if let Err(e) = encoder.set_bitrate(kbps) {
                                log::warn!("Failed to set bitrate: {}", e);
                            }
                        }
                        PipelineCommand::SetFps(new_fps) => {
                            let new_fps = new_fps.clamp(1, 240);
                            log::info!("Changing target FPS from {} to {}", self.config.fps, new_fps);
                            self.config.fps = new_fps;
                            target_interval = Duration::from_nanos(1_000_000_000 / new_fps as u64);
                            force_next_frame = true;
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
            Err(_) => log::warn!("Audio task did not stop within 500ms (will be cleaned up on drop)"),
        }
        cursor_handle.abort();

        // Stats
        let elapsed = start_time.elapsed();
        log::info!(
            "Pipeline stopped. {} frames ({} skipped), {:.2} MB, {:.1} FPS, {:.1}s",
            total_frames,
            skipped_frames,
            total_bytes as f64 / 1_048_576.0,
            total_frames as f64 / elapsed.as_secs_f64(),
            elapsed.as_secs_f64(),
        );

        let _ = self.event_tx.send(MediaEvent::Stopped).await;
        Ok(())
    }
}

impl Drop for MediaPipeline {
    fn drop(&mut self) {
        log::info!("MediaPipeline dropped — signaling audio shutdown");
        self.audio_shutdown.store(true, Ordering::Relaxed);
    }
}
