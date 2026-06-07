//! Basic capture + encode example.
//!
//! Creates a [`MediaPipeline`], captures 5 seconds of video, and prints
//! statistics about the captured frames.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example capture_encode
//! ```
//!
//! To write a raw bitstream to a file and choose a codec/backend:
//!
//! ```bash
//! cargo run --example capture_encode -- --codec h265 --encoder hevc_vaapi --output capture.h265
//! cargo run --example capture_encode -- --codec av1 --output capture.obu
//! ```

use std::io::Write;
use std::time::{Duration, Instant};

use lunaris_media::pipeline::{MediaEvent, MediaPipeline, PipelineCommand};
use lunaris_media::types::{FrameType, PixelFormat, StreamConfig, VideoCodec};

const DEFAULT_CAPTURE_SECONDS: u64 = 5;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // ── Parse simple CLI args ───────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let codec = arg_value(&args, "--codec")
        .map(|value| parse_codec(&value))
        .transpose()?
        .unwrap_or(VideoCodec::H264);
    let fps = parse_u32_arg(&args, "--fps", 60)?;
    let bitrate_kbps = parse_u32_arg(&args, "--bitrate", 10_000)?;
    let duration =
        Duration::from_secs(parse_u64_arg(&args, "--duration", DEFAULT_CAPTURE_SECONDS)?);
    let output_path = arg_value(&args, "--output");
    let preferred_encoder = arg_value(&args, "--encoder");

    // ── Configure the pipeline ──────────────────────────────────────
    let config = StreamConfig {
        width: 1920,
        height: 1080,
        fps,
        codec,
        bitrate_kbps,
        pixel_format: PixelFormat::NV12,
        preferred_encoder,
        virtual_display: false,
    };

    println!("╔══════════════════════════════════════════════════╗");
    println!("║          lunaris-media capture example           ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!("║  Resolution:  {}x{:<26}║", config.width, config.height);
    println!("║  FPS:         {:<36}║", config.fps);
    println!("║  Codec:       {:<36}║", config.codec);
    println!("║  Bitrate:     {} kbps{:<25}║", config.bitrate_kbps, "");
    println!("║  Duration:    {} seconds{:<24}║", duration.as_secs(), "");
    if let Some(ref path) = output_path {
        println!("║  Output:      {:<36}║", path);
    }
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    // ── Create and start the pipeline ───────────────────────────────
    let (pipeline, mut events, commands) = MediaPipeline::new(config);

    let pipeline_handle = tokio::spawn(async move {
        if let Err(e) = pipeline.run("default").await {
            log::error!("Pipeline error: {e}");
        }
    });

    // ── Collect statistics ──────────────────────────────────────────
    let mut video_frame_count: u64 = 0;
    let mut audio_frame_count: u64 = 0;
    let mut cursor_update_count: u64 = 0;
    let mut total_video_bytes: u64 = 0;
    let mut total_audio_bytes: u64 = 0;
    let mut keyframe_count: u64 = 0;
    let mut error_count: u64 = 0;

    let mut output_file = output_path.as_ref().map(|p| {
        std::fs::File::create(p)
            .unwrap_or_else(|e| panic!("Failed to create output file '{p}': {e}"))
    });

    let start = Instant::now();

    // ── Schedule stop after duration ────────────────────────
    let stop_commands = commands.clone();
    tokio::spawn(async move {
        tokio::time::sleep(duration).await;
        log::info!("Capture duration elapsed — sending Stop command");
        let _ = stop_commands.send(PipelineCommand::Stop).await;
    });

    // ── Request a keyframe after 2 seconds to demonstrate commands ──
    let kf_commands = commands.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(2)).await;
        log::info!("Requesting keyframe");
        let _ = kf_commands.send(PipelineCommand::RequestKeyframe).await;
    });

    // ── Event loop ──────────────────────────────────────────────────
    while let Some(event) = events.recv().await {
        match event {
            MediaEvent::VideoFrame(frame) => {
                video_frame_count += 1;
                total_video_bytes += frame.data.len() as u64;

                if frame.frame_type == FrameType::Key {
                    keyframe_count += 1;
                }

                // Write to output file if requested.
                if let Some(ref mut f) = output_file {
                    f.write_all(&frame.data)?;
                }

                // Periodic progress logging.
                if video_frame_count % 60 == 0 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let fps = video_frame_count as f64 / elapsed;
                    let mbps = (total_video_bytes as f64 * 8.0) / (elapsed * 1_000_000.0);
                    log::info!(
                        "Progress: {} frames, {:.1} fps, {:.2} Mbps",
                        video_frame_count,
                        fps,
                        mbps,
                    );
                }
            }

            MediaEvent::AudioFrame(frame) => {
                audio_frame_count += 1;
                total_audio_bytes += frame.data.len() as u64;
            }

            MediaEvent::CursorUpdate(_state) => {
                cursor_update_count += 1;
            }

            MediaEvent::Error(e) => {
                error_count += 1;
                log::warn!("Pipeline error: {e}");
            }

            MediaEvent::EncoderStarted {
                encoder,
                gpu_name,
                requested_encoder,
            } => {
                log::info!(
                    "Encoder active: {} ({}) on {} requested={}",
                    encoder.name,
                    encoder.hw_type,
                    gpu_name.as_deref().unwrap_or("unknown GPU"),
                    requested_encoder.as_deref().unwrap_or("auto"),
                );
            }

            MediaEvent::Started => {
                println!(
                    "✓ Pipeline started — capturing for {} seconds...",
                    duration.as_secs()
                );
            }

            MediaEvent::Stopped => {
                println!("✓ Pipeline stopped");
                break;
            }
        }
    }

    // ── Wait for pipeline task to finish ────────────────────────────
    let _ = pipeline_handle.await;

    // ── Print results ───────────────────────────────────────────────
    let elapsed = start.elapsed();
    let avg_fps = if elapsed.as_secs_f64() > 0.0 {
        video_frame_count as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    let avg_video_frame_size = if video_frame_count > 0 {
        total_video_bytes / video_frame_count
    } else {
        0
    };
    let video_bitrate_mbps = if elapsed.as_secs_f64() > 0.0 {
        (total_video_bytes as f64 * 8.0) / (elapsed.as_secs_f64() * 1_000_000.0)
    } else {
        0.0
    };

    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║                 Capture Results                  ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!(
        "║  Duration:        {:.2}s{:<28}║",
        elapsed.as_secs_f64(),
        ""
    );
    println!("║  Video frames:    {:<32}║", video_frame_count);
    println!("║  Keyframes:       {:<32}║", keyframe_count);
    println!("║  Average FPS:     {:.1}{:<30}║", avg_fps, "");
    println!(
        "║  Video bytes:     {:<32}║",
        format_bytes(total_video_bytes)
    );
    println!(
        "║  Avg frame size:  {} bytes{:<20}║",
        avg_video_frame_size, ""
    );
    println!(
        "║  Video bitrate:   {:.2} Mbps{:<22}║",
        video_bitrate_mbps, ""
    );
    println!("║  Audio frames:    {:<32}║", audio_frame_count);
    println!(
        "║  Audio bytes:     {:<32}║",
        format_bytes(total_audio_bytes)
    );
    println!("║  Cursor updates:  {:<32}║", cursor_update_count);
    println!("║  Errors:          {:<32}║", error_count);
    if let Some(ref path) = output_path {
        println!("║  Output file:     {:<32}║", path);
    }
    println!("╚══════════════════════════════════════════════════╝");

    Ok(())
}

/// Format a byte count in a human-readable way.
fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn parse_codec(value: &str) -> Result<VideoCodec, Box<dyn std::error::Error>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "h264" | "h.264" | "avc" => Ok(VideoCodec::H264),
        "h265" | "h.265" | "hevc" => Ok(VideoCodec::H265),
        "av1" => Ok(VideoCodec::AV1),
        other => Err(format!("unsupported codec '{other}' (expected h264, h265, or av1)").into()),
    }
}

fn parse_u32_arg(
    args: &[String],
    name: &str,
    default: u32,
) -> Result<u32, Box<dyn std::error::Error>> {
    arg_value(args, name)
        .map(|value| {
            value
                .parse::<u32>()
                .map_err(|e| format!("invalid value for {name}: {value} ({e})").into())
        })
        .unwrap_or(Ok(default))
}

fn parse_u64_arg(
    args: &[String],
    name: &str,
    default: u64,
) -> Result<u64, Box<dyn std::error::Error>> {
    arg_value(args, name)
        .map(|value| {
            value
                .parse::<u64>()
                .map_err(|e| format!("invalid value for {name}: {value} ({e})").into())
        })
        .unwrap_or(Ok(default))
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}
