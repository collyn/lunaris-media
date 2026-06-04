//! Linux audio capture using cpal loopback and Opus encoding.
//!
//! On PulseAudio systems, loopback capture works via "monitor" devices — every
//! output sink automatically gets a corresponding `.monitor` source. On
//! PipeWire the same mechanism is exposed through the PulseAudio compatibility
//! layer.
//!
//! ## Capture Flow
//!
//! ```text
//!   cpal input stream (monitor device)  ← runs on dedicated thread
//!       → accumulate f32 PCM into frame-sized chunks
//!       → Opus encode each chunk
//!       → send EncodedAudioFrame through internal channel
//! ```
//!
//! ## Threading
//!
//! `cpal::Stream` is not `Send`, so the stream lives on a dedicated OS thread.
//! A shutdown flag and frame channel bridge between the stream thread and the
//! async pipeline.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use crate::audio::AudioCapture;
use crate::error::MediaError;
use crate::types::*;

/// Linux system audio capture backend.
///
/// Uses [`cpal`] to capture from the default output device's monitor/loopback
/// source and [`opus`] to encode captured PCM samples into Opus frames.
///
/// Because `cpal::Stream` is `!Send`, the stream runs on a dedicated OS thread
/// and communicates via a `std::sync::mpsc` channel.
pub struct LinuxAudioCapture {
    /// Receiver for encoded audio frames produced by the stream thread.
    frame_rx: Option<mpsc::Receiver<EncodedAudioFrame>>,
    /// Handle to the dedicated thread running the cpal stream.
    stream_thread: Option<std::thread::JoinHandle<()>>,
    /// Shared flag to signal the stream thread to shut down.
    shutdown: Arc<AtomicBool>,
    /// Whether the capture session is currently active.
    capturing: bool,
}

impl LinuxAudioCapture {
    /// Create a new Linux audio capture instance.
    ///
    /// Validates that a suitable audio output device is available but does
    /// **not** start capturing. Call [`start`](AudioCapture::start) to begin.
    pub fn new() -> Result<Self, MediaError> {
        // Validate that cpal can find an output device (we'll use its monitor).
        use cpal::traits::HostTrait;
        let host = cpal::default_host();
        let _device = host.default_output_device().ok_or_else(|| {
            MediaError::AudioError(
                "No default output device found. On Linux, ensure PulseAudio or \
                 PipeWire (with PulseAudio compat) is running and has a default \
                 output sink configured."
                    .into(),
            )
        })?;

        Ok(Self {
            frame_rx: None,
            stream_thread: None,
            shutdown: Arc::new(AtomicBool::new(false)),
            capturing: false,
        })
    }
}

impl AudioCapture for LinuxAudioCapture {
    /// Start capturing audio from the system loopback device.
    ///
    /// Spawns a dedicated OS thread that:
    /// 1. Locates the monitor/loopback input device.
    /// 2. Creates an Opus encoder configured for low-latency audio.
    /// 3. Opens a cpal input stream on the monitor device.
    /// 4. In the stream callback, accumulates f32 samples into frame-sized
    ///    chunks, encodes each chunk with Opus, and sends the result through
    ///    an internal channel.
    fn start(&mut self, config: &AudioCaptureConfig) -> Result<(), MediaError> {
        if self.capturing {
            return Err(MediaError::AudioError(
                "Audio capture already started".into(),
            ));
        }

        let sample_rate = config.sample_rate;
        let channels = config.channels;
        let frame_size_ms = config.frame_size_ms;

        let (tx, rx) = mpsc::sync_channel::<EncodedAudioFrame>(64);
        let shutdown = Arc::clone(&self.shutdown);
        self.shutdown.store(false, Ordering::Relaxed);

        // Spawn a dedicated thread for cpal (Stream is !Send).
        let thread = std::thread::Builder::new()
            .name("lunaris-audio-capture".into())
            .spawn(move || {
                if let Err(e) =
                    run_audio_stream(tx, shutdown, sample_rate, channels, frame_size_ms)
                {
                    log::error!("Audio stream thread error: {e}");
                }
            })
            .map_err(|e| {
                MediaError::AudioError(format!("Failed to spawn audio thread: {e}"))
            })?;

        self.stream_thread = Some(thread);
        self.frame_rx = Some(rx);
        self.capturing = true;

        log::info!(
            "Audio capture started: {}Hz, {} ch, {}ms frames",
            sample_rate,
            channels,
            frame_size_ms,
        );

        Ok(())
    }

    /// Wait for and return the next encoded audio frame.
    ///
    /// Blocks the current thread until a frame is available. This is designed
    /// to be called from a blocking Tokio task (see
    /// [`MediaPipeline::audio_task`](crate::pipeline::MediaPipeline)).
    fn next_frame(&mut self) -> Result<EncodedAudioFrame, MediaError> {
        let rx = self.frame_rx.as_ref().ok_or_else(|| {
            MediaError::AudioError("Audio capture not started".into())
        })?;

        rx.recv().map_err(|_| {
            MediaError::AudioError("Audio frame channel closed".into())
        })
    }

    /// Stop the audio capture session and release all resources.
    fn stop(&mut self) -> Result<(), MediaError> {
        self.shutdown.store(true, Ordering::Relaxed);

        if let Some(handle) = self.stream_thread.take() {
            let _ = handle.join();
        }

        self.frame_rx.take();
        self.capturing = false;
        log::info!("Audio capture stopped");
        Ok(())
    }

    /// Returns `true` if audio capture is currently active.
    fn is_capturing(&self) -> bool {
        self.capturing
    }
}

impl Drop for LinuxAudioCapture {
    fn drop(&mut self) {
        if self.capturing {
            log::info!("LinuxAudioCapture dropped while capturing — stopping audio thread");
            self.shutdown.store(true, Ordering::Relaxed);
            if let Some(handle) = self.stream_thread.take() {
                let _ = handle.join();
            }
            self.frame_rx.take();
            self.capturing = false;
            log::info!("Audio capture stopped via Drop");
        }
    }
}

// ---------------------------------------------------------------------------
// Stream thread entry point
// ---------------------------------------------------------------------------

/// Runs on a dedicated OS thread. Creates the cpal stream, plays it, and
/// blocks until the shutdown flag is set.
fn run_audio_stream(
    tx: mpsc::SyncSender<EncodedAudioFrame>,
    shutdown: Arc<AtomicBool>,
    sample_rate: u32,
    channels: u16,
    frame_size_ms: u32,
) -> Result<(), MediaError> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();

    // ── Find monitor device ─────────────────────────────────────────
    let device = host
        .default_input_device()
        .or_else(|| {
            host.input_devices().ok().and_then(|mut devs| {
                devs.find(|d| {
                    d.name()
                        .map(|n| n.to_lowercase().contains("monitor"))
                        .unwrap_or(false)
                })
            })
        })
        .ok_or_else(|| {
            MediaError::AudioError(
                "No loopback/monitor audio device found. On PulseAudio, ensure \
                 the default output sink has a .monitor source. On PipeWire, \
                 ensure the PulseAudio compatibility layer is active."
                    .into(),
            )
        })?;

    let device_name = device.name().unwrap_or_else(|_| "<unknown>".into());
    log::info!("Audio thread: using device '{device_name}'");

    // ── Opus encoder ────────────────────────────────────────────────
    let opus_channels = match channels {
        1 => opus::Channels::Mono,
        _ => opus::Channels::Stereo,
    };

    let mut opus_encoder =
        opus::Encoder::new(sample_rate, opus_channels, opus::Application::LowDelay)
            .map_err(|e| {
                MediaError::AudioError(format!("Opus encoder init failed: {e}"))
            })?;

    opus_encoder
        .set_bitrate(opus::Bitrate::Bits(128_000))
        .map_err(|e| {
            MediaError::AudioError(format!("Opus set_bitrate failed: {e}"))
        })?;

    // ── Prepare accumulator ─────────────────────────────────────────
    let frame_size_samples = (sample_rate * frame_size_ms / 1000) as usize;
    let samples_per_frame = frame_size_samples * channels as usize;
    let mut pcm_buffer: Vec<f32> = Vec::with_capacity(samples_per_frame);
    let start_time = Instant::now();
    let channel_count = channels;

    let stream_config = cpal::StreamConfig {
        channels: cpal::ChannelCount::from(channels),
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Fixed(frame_size_samples as u32),
    };

    // ── Build and play the stream ───────────────────────────────────
    let stream = device
        .build_input_stream(
            &stream_config,
            move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                pcm_buffer.extend_from_slice(data);

                while pcm_buffer.len() >= samples_per_frame {
                    let frame_pcm: Vec<f32> =
                        pcm_buffer.drain(..samples_per_frame).collect();

                    let mut opus_out = vec![0u8; 4000];
                    match opus_encoder.encode_float(&frame_pcm, &mut opus_out) {
                        Ok(len) => {
                            opus_out.truncate(len);

                            let pts = start_time.elapsed().as_micros() as u64;
                            let duration_us = u64::from(frame_size_samples as u32)
                                * 1_000_000
                                / u64::from(sample_rate);

                            let frame = EncodedAudioFrame {
                                data: opus_out,
                                pts,
                                duration: duration_us,
                                sample_rate,
                                channels: channel_count,
                            };

                            let _ = tx.try_send(frame);
                        }
                        Err(e) => {
                            log::warn!("Opus encode error: {e}");
                        }
                    }
                }
            },
            |err| log::error!("cpal stream error: {err}"),
            None,
        )
        .map_err(|e| {
            MediaError::AudioError(format!("Failed to build cpal input stream: {e}"))
        })?;

    stream.play().map_err(|e| {
        MediaError::AudioError(format!("Failed to start cpal stream: {e}"))
    })?;

    log::info!("Audio stream playing");

    // Block until shutdown is requested.
    while !shutdown.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // Stream is dropped here, stopping capture.
    log::info!("Audio stream thread shutting down");
    Ok(())
}
