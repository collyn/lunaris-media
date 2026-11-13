//! Linux audio capture (what the computer plays, not the microphone) and
//! Opus encoding: PulseAudio/PipeWire's default-output monitor, else cpal.
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
        let microphone = config.microphone;

        let (tx, rx) = mpsc::sync_channel::<EncodedAudioFrame>(64);
        let shutdown = Arc::clone(&self.shutdown);
        self.shutdown.store(false, Ordering::Relaxed);

        // Spawn a dedicated thread for cpal (Stream is !Send).
        let thread = std::thread::Builder::new()
            .name("lunaris-audio-capture".into())
            .spawn(move || {
                if let Err(e) =
                    run_audio_stream(tx, shutdown, sample_rate, channels, frame_size_ms, microphone)
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

/// Opus-encodes PCM in frame-sized chunks and sends the frames on.
struct FrameEncoder {
    tx: mpsc::SyncSender<EncodedAudioFrame>,
    encoder: opus::Encoder,
    pcm: Vec<f32>,
    samples_per_frame: usize,
    frame_size_samples: usize,
    sample_rate: u32,
    channels: u16,
    start_time: Instant,
}

impl FrameEncoder {
    fn new(
        tx: mpsc::SyncSender<EncodedAudioFrame>,
        sample_rate: u32,
        channels: u16,
        frame_size_ms: u32,
    ) -> Result<Self, MediaError> {
        let opus_channels = match channels {
            1 => opus::Channels::Mono,
            _ => opus::Channels::Stereo,
        };
        let mut encoder =
            opus::Encoder::new(sample_rate, opus_channels, opus::Application::LowDelay)
                .map_err(|e| MediaError::AudioError(format!("Opus encoder init failed: {e}")))?;
        encoder
            .set_bitrate(opus::Bitrate::Bits(128_000))
            .map_err(|e| MediaError::AudioError(format!("Opus set_bitrate failed: {e}")))?;
        let frame_size_samples = (sample_rate * frame_size_ms / 1000) as usize;
        let samples_per_frame = frame_size_samples * channels as usize;
        Ok(Self {
            tx,
            encoder,
            pcm: Vec::with_capacity(samples_per_frame * 2),
            samples_per_frame,
            frame_size_samples,
            sample_rate,
            channels,
            start_time: Instant::now(),
        })
    }

    fn push(&mut self, data: &[f32]) {
        self.pcm.extend_from_slice(data);
        while self.pcm.len() >= self.samples_per_frame {
            let mut opus_out = vec![0u8; 4000];
            let encoded = self
                .encoder
                .encode_float(&self.pcm[..self.samples_per_frame], &mut opus_out);
            self.pcm.drain(..self.samples_per_frame);
            match encoded {
                Ok(len) => {
                    opus_out.truncate(len);
                    let frame = EncodedAudioFrame {
                        data: opus_out,
                        pts: self.start_time.elapsed().as_micros() as u64,
                        duration: self.frame_size_samples as u64 * 1_000_000
                            / u64::from(self.sample_rate),
                        sample_rate: self.sample_rate,
                        channels: self.channels,
                    };
                    let _ = self.tx.try_send(frame);
                }
                Err(e) => log::warn!("Opus encode error: {e}"),
            }
        }
    }
}

/// Runs on a dedicated OS thread: captures what the computer plays (not the
/// microphone) until the shutdown flag is set.
fn run_audio_stream(
    tx: mpsc::SyncSender<EncodedAudioFrame>,
    shutdown: Arc<AtomicBool>,
    sample_rate: u32,
    channels: u16,
    frame_size_ms: u32,
    microphone: bool,
) -> Result<(), MediaError> {
    let mut encoder = FrameEncoder::new(tx, sample_rate, channels, frame_size_ms)?;

    // PulseAudio / PipeWire: the default output's monitor, which follows the
    // output the user picks.
    match pulse::Recorder::open(pulse::DEFAULT_MONITOR, sample_rate, channels, frame_size_ms) {
        Ok(recorder) => {
            log::info!("Audio thread: recording the default output (@DEFAULT_MONITOR@) through PulseAudio");
            // The microphone, mixed in when asked for.
            let mic = if microphone {
                match pulse::Recorder::open(pulse::DEFAULT_SOURCE, sample_rate, channels, frame_size_ms) {
                    Ok(m) => {
                        log::info!("Audio thread: mixing in the default microphone (@DEFAULT_SOURCE@)");
                        Some(m)
                    }
                    Err(e) => {
                        log::warn!("Microphone not recorded: {e}");
                        None
                    }
                }
            } else {
                None
            };
            let mut buf = vec![0f32; encoder.samples_per_frame];
            let mut mic_buf = vec![0f32; encoder.samples_per_frame];
            while !shutdown.load(Ordering::Relaxed) {
                if let Err(e) = recorder.read(&mut buf) {
                    log::error!("PulseAudio read failed: {e}");
                    break;
                }
                if let Some(mic) = &mic {
                    if mic.read(&mut mic_buf).is_ok() {
                        mix_into(&mut buf, &mic_buf);
                    }
                }
                encoder.push(&buf);
            }
            log::info!("Audio stream thread shutting down");
            return Ok(());
        }
        Err(e) => log::warn!("PulseAudio monitor capture unavailable ({e}); trying ALSA"),
    }

    if microphone {
        log::warn!("Microphone not recorded: it needs PulseAudio or PipeWire");
    }
    run_cpal_stream(encoder, shutdown, sample_rate, channels)
}

/// Adds `other` to `out`, kept within [-1, 1].
fn mix_into(out: &mut [f32], other: &[f32]) {
    for (o, m) in out.iter_mut().zip(other) {
        *o = (*o + *m).clamp(-1.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn mixing_adds_and_clamps() {
        let mut out = [0.25, 0.9, -0.9];
        super::mix_into(&mut out, &[0.25, 0.5, -0.5]);
        assert_eq!(out, [0.5, 1.0, -1.0]);
    }
}

/// Capture through cpal (ALSA): a monitor device when one is listed. The
/// default input device is the microphone, so it is only the last resort.
fn run_cpal_stream(
    mut encoder: FrameEncoder,
    shutdown: Arc<AtomicBool>,
    sample_rate: u32,
    channels: u16,
) -> Result<(), MediaError> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let monitor = host.input_devices().ok().and_then(|mut devs| {
        devs.find(|d| {
            // cpal 0.15's `name()` was the ALSA PCM name; in 0.18 that is
            // the device id (`description().name()` is now a human label).
            d.id()
                .map(|id| id.id().to_lowercase().contains("monitor"))
                .unwrap_or(false)
        })
    });
    let device = match monitor {
        Some(d) => d,
        None => {
            log::warn!(
                "No monitor device found: capturing the default input, which is usually a microphone"
            );
            host.default_input_device().ok_or_else(|| {
                MediaError::AudioError(
                    "No loopback/monitor audio device found. On PulseAudio, ensure \
                     the default output sink has a .monitor source. On PipeWire, \
                     ensure the PulseAudio compatibility layer is active."
                        .into(),
                )
            })?
        }
    };

    let device_name = device
        .id()
        .map(|id| id.id().to_string())
        .unwrap_or_else(|_| "<unknown>".into());
    log::info!("Audio thread: using device '{device_name}'");

    let stream_config = cpal::StreamConfig {
        channels: cpal::ChannelCount::from(channels),
        sample_rate,
        // cpal 0.15 set the ALSA ring buffer to this many frames (one Opus
        // frame) and the period to a quarter of it. 0.18 reads `Fixed(x)` as
        // the period and sizes the ring at two periods, so half a frame keeps
        // the ring at one frame, as before.
        buffer_size: cpal::BufferSize::Fixed((encoder.frame_size_samples / 2) as u32),
    };

    let stream = device
        .build_input_stream(
            stream_config,
            move |data: &[f32], _info: &cpal::InputCallbackInfo| encoder.push(data),
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

/// Recording through libpulse-simple, loaded at run time (no build-time
/// dependency; PipeWire provides it through pipewire-pulse).
mod pulse {
    use std::ffi::{c_char, c_int, c_void, CStr};

    #[repr(C)]
    struct SampleSpec {
        format: c_int,
        rate: u32,
        channels: u8,
    }

    #[repr(C)]
    struct BufferAttr {
        maxlength: u32,
        tlength: u32,
        prebuf: u32,
        minreq: u32,
        fragsize: u32,
    }

    const PA_SAMPLE_FLOAT32LE: c_int = 5;
    const PA_STREAM_RECORD: c_int = 2;

    type NewFn = unsafe extern "C" fn(
        *const c_char,
        *const c_char,
        c_int,
        *const c_char,
        *const c_char,
        *const SampleSpec,
        *const c_void,
        *const BufferAttr,
        *mut c_int,
    ) -> *mut c_void;
    type ReadFn = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, *mut c_int) -> c_int;
    type FreeFn = unsafe extern "C" fn(*mut c_void);
    type StrErrorFn = unsafe extern "C" fn(c_int) -> *const c_char;

    /// The default output's monitor: what the computer plays.
    pub const DEFAULT_MONITOR: &CStr = c"@DEFAULT_MONITOR@";
    /// The default input: the microphone.
    pub const DEFAULT_SOURCE: &CStr = c"@DEFAULT_SOURCE@";

    pub struct Recorder {
        handle: *mut c_void,
        read: ReadFn,
        free: FreeFn,
        strerror: Option<StrErrorFn>,
        _lib: libloading::Library,
    }

    impl Recorder {
        pub fn open(device: &CStr, sample_rate: u32, channels: u16, frame_size_ms: u32) -> Result<Self, String> {
            unsafe {
                let lib = libloading::Library::new("libpulse-simple.so.0")
                    .map_err(|e| format!("cannot load libpulse-simple: {e}"))?;
                let new: NewFn = *lib.get::<NewFn>(b"pa_simple_new\0").map_err(|e| e.to_string())?;
                let read: ReadFn = *lib.get::<ReadFn>(b"pa_simple_read\0").map_err(|e| e.to_string())?;
                let free: FreeFn = *lib.get::<FreeFn>(b"pa_simple_free\0").map_err(|e| e.to_string())?;
                let strerror = lib.get::<StrErrorFn>(b"pa_strerror\0").ok().map(|f| *f);

                let spec = SampleSpec {
                    format: PA_SAMPLE_FLOAT32LE,
                    rate: sample_rate,
                    channels: channels as u8,
                };
                // Deliver one frame at a time so the delay stays at one frame.
                let frame_bytes = (sample_rate * frame_size_ms / 1000) * u32::from(channels) * 4;
                // At most a few frames queued: a device whose clock runs a
                // little ahead then drops old sound instead of adding delay.
                let attr = BufferAttr {
                    maxlength: frame_bytes * 4,
                    tlength: u32::MAX,
                    prebuf: u32::MAX,
                    minreq: u32::MAX,
                    fragsize: frame_bytes,
                };
                let mut err: c_int = 0;
                let handle = new(
                    std::ptr::null(),
                    c"Lunaris".as_ptr(),
                    PA_STREAM_RECORD,
                    device.as_ptr(),
                    c"Remote desktop sound".as_ptr(),
                    &spec,
                    std::ptr::null(),
                    &attr,
                    &mut err,
                );
                if handle.is_null() {
                    let msg = strerror
                        .map(|f| CStr::from_ptr(f(err)).to_string_lossy().into_owned())
                        .unwrap_or_else(|| format!("error {err}"));
                    return Err(format!("cannot record {}: {msg}", device.to_string_lossy()));
                }
                Ok(Self { handle, read, free, strerror, _lib: lib })
            }
        }

        /// Blocks until `buf` is filled.
        pub fn read(&self, buf: &mut [f32]) -> Result<(), String> {
            let mut err: c_int = 0;
            let ret = unsafe {
                (self.read)(
                    self.handle,
                    buf.as_mut_ptr() as *mut c_void,
                    std::mem::size_of_val(buf),
                    &mut err,
                )
            };
            if ret < 0 {
                let msg = self
                    .strerror
                    .map(|f| unsafe { CStr::from_ptr(f(err)).to_string_lossy().into_owned() })
                    .unwrap_or_else(|| format!("error {err}"));
                return Err(msg);
            }
            Ok(())
        }
    }

    impl Drop for Recorder {
        fn drop(&mut self) {
            unsafe { (self.free)(self.handle) };
        }
    }

    // Used only from the capture thread that opened it.
    unsafe impl Send for Recorder {}
}
