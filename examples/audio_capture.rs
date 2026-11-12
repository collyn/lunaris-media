//! Records what this computer plays for a few seconds and reports the Opus
//! frames: `cargo run --release --example audio_capture -- [seconds] [--microphone]`.
//! The log says which source is used (it must be the output's monitor, not
//! the microphone).

#[cfg(target_os = "linux")]
fn main() {
    use lunaris_media::audio::linux::LinuxAudioCapture;
    use lunaris_media::audio::AudioCapture;
    use lunaris_media::types::AudioCaptureConfig;

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let seconds: f64 = std::env::args().nth(1).filter(|a| !a.starts_with('-')).and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let microphone = std::env::args().any(|a| a == "--microphone");
    let config = AudioCaptureConfig { sample_rate: 48_000, channels: 2, frame_size_ms: 10, microphone };
    let mut capture = LinuxAudioCapture::new().expect("audio device");
    capture.start(&config).expect("start");
    let start = std::time::Instant::now();
    let (mut frames, mut bytes, mut worst_gap_ms) = (0u32, 0usize, 0f64);
    let mut last = start;
    while start.elapsed().as_secs_f64() < seconds {
        let frame = capture.next_frame().expect("frame");
        let now = std::time::Instant::now();
        if frames > 0 {
            worst_gap_ms = f64::max(worst_gap_ms, (now - last).as_secs_f64() * 1000.0);
        }
        last = now;
        frames += 1;
        bytes += frame.data.len();
    }
    capture.stop().ok();
    println!(
        "{} frames in {:.1}s ({:.0}/s, expected 100/s), {} bytes, longest gap {:.1} ms",
        frames,
        seconds,
        f64::from(frames) / seconds,
        bytes,
        worst_gap_ms
    );
}

#[cfg(not(target_os = "linux"))]
fn main() {}
