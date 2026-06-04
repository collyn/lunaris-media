//! Windows audio capture using cpal (WASAPI loopback) and Opus encoding.
//!
//! On Windows, WASAPI provides loopback capture from the default output device.
//! The capture flow is identical to the Linux backend:
//!
//! ```text
//!   cpal input stream (WASAPI loopback)  ← runs on dedicated thread
//!       → accumulate f32 PCM into frame-sized chunks
//!       → Opus encode each chunk
//!       → send EncodedAudioFrame through internal channel
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use crate::audio::AudioCapture;
use crate::error::MediaError;
use crate::types::*;

pub struct WindowsAudioCapture {
    frame_rx: Option<mpsc::Receiver<EncodedAudioFrame>>,
    stream_thread: Option<std::thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    capturing: bool,
}

impl WindowsAudioCapture {
    pub fn new() -> Result<Self, MediaError> {
        use cpal::traits::HostTrait;
        let host = cpal::default_host();
        let _device = host.default_output_device().ok_or_else(|| {
            MediaError::AudioError(
                "No default output device found for WASAPI loopback capture.".into(),
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

impl AudioCapture for WindowsAudioCapture {
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

        let thread = std::thread::Builder::new()
            .name("lunaris-audio-capture".into())
            .spawn(move || {
                if let Err(e) =
                    run_audio_stream(tx, shutdown, sample_rate, channels, frame_size_ms)
                {
                    log::error!("Audio stream thread error: {e}");
                }
            })
            .map_err(|e| MediaError::AudioError(format!("Failed to spawn audio thread: {e}")))?;

        self.stream_thread = Some(thread);
        self.frame_rx = Some(rx);
        self.capturing = true;

        log::info!(
            "Windows audio capture started: {}Hz, {} ch, {}ms frames",
            sample_rate,
            channels,
            frame_size_ms
        );
        Ok(())
    }

    fn next_frame(&mut self) -> Result<EncodedAudioFrame, MediaError> {
        let rx = self
            .frame_rx
            .as_ref()
            .ok_or_else(|| MediaError::AudioError("Audio capture not started".into()))?;
        rx.recv()
            .map_err(|_| MediaError::AudioError("Audio frame channel closed".into()))
    }

    fn stop(&mut self) -> Result<(), MediaError> {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.stream_thread.take() {
            let _ = handle.join();
        }
        self.frame_rx.take();
        self.capturing = false;
        log::info!("Windows audio capture stopped");
        Ok(())
    }

    fn is_capturing(&self) -> bool {
        self.capturing
    }
}

impl Drop for WindowsAudioCapture {
    fn drop(&mut self) {
        if self.capturing {
            log::info!("WindowsAudioCapture dropped while capturing — stopping audio thread");
            self.shutdown.store(true, Ordering::Relaxed);
            if let Some(handle) = self.stream_thread.take() {
                let _ = handle.join();
            }
            self.frame_rx.take();
            self.capturing = false;
        }
    }
}

fn run_audio_stream(
    tx: mpsc::SyncSender<EncodedAudioFrame>,
    shutdown: Arc<AtomicBool>,
    sample_rate: u32,
    channels: u16,
    frame_size_ms: u32,
) -> Result<(), MediaError> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();

    // On Windows, use the default output device for WASAPI loopback.
    // cpal's WASAPI host can open an input stream on an output device for loopback.
    let device = host
        .default_output_device()
        .ok_or_else(|| MediaError::AudioError("No default output device for loopback".into()))?;

    let device_name = device.name().unwrap_or_else(|_| "<unknown>".into());
    log::info!(
        "Audio thread: using device '{}' for WASAPI loopback",
        device_name
    );

    let opus_channels = match channels {
        1 => opus::Channels::Mono,
        _ => opus::Channels::Stereo,
    };

    let mut opus_encoder =
        opus::Encoder::new(sample_rate, opus_channels, opus::Application::LowDelay)
            .map_err(|e| MediaError::AudioError(format!("Opus encoder init failed: {e}")))?;

    opus_encoder
        .set_bitrate(opus::Bitrate::Bits(128_000))
        .map_err(|e| MediaError::AudioError(format!("Opus set_bitrate failed: {e}")))?;

    let frame_size_samples = (sample_rate * frame_size_ms / 1000) as usize;
    let samples_per_frame = frame_size_samples * channels as usize;
    let mut pcm_buffer: Vec<f32> = Vec::with_capacity(samples_per_frame);
    let start_time = Instant::now();
    let channel_count = channels;

    let stream_config = cpal::StreamConfig {
        channels: cpal::ChannelCount::from(channels),
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let stream = device
        .build_input_stream(
            &stream_config,
            move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                pcm_buffer.extend_from_slice(data);

                while pcm_buffer.len() >= samples_per_frame {
                    let frame_pcm: Vec<f32> = pcm_buffer.drain(..samples_per_frame).collect();

                    let mut opus_out = vec![0u8; 4000];
                    match opus_encoder.encode_float(&frame_pcm, &mut opus_out) {
                        Ok(len) => {
                            opus_out.truncate(len);
                            let pts = start_time.elapsed().as_micros() as u64;
                            let duration_us = u64::from(frame_size_samples as u32) * 1_000_000
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
        .map_err(|e| MediaError::AudioError(format!("Failed to build cpal input stream: {e}")))?;

    stream
        .play()
        .map_err(|e| MediaError::AudioError(format!("Failed to start cpal stream: {e}")))?;

    log::info!("Audio stream playing (WASAPI loopback)");

    while !shutdown.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    log::info!("Audio stream thread shutting down");
    Ok(())
}
