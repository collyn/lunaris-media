//! Dynamic screen capture frame debug tool.
//! Captures a single frame from the default display and saves it as a PNG file `frame.png`.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example capture_frame
//! ```

use std::time::Duration;
use lunaris_media::capture::create_screen_capture;
use lunaris_media::types::{PixelFormat, StreamConfig, VideoCodec};
use lunaris_media::capture::gpu_buffer::GpuBuffer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Creating screen capture backend...");
    let mut capture = create_screen_capture()?;

    println!("Listing displays...");
    let displays = capture.list_displays().await?;
    for d in &displays {
        println!("  Display ID: {}, Name: {}, {}x{} @{}Hz, primary: {}", d.id, d.name, d.width, d.height, d.refresh_rate, d.is_primary);
    }

    if displays.is_empty() {
        println!("No displays found!");
        return Ok(());
    }

    let primary_display = displays.iter().find(|d| d.is_primary).unwrap_or(&displays[0]);
    println!("Selected display: {} (ID: {})", primary_display.name, primary_display.id);

    let config = StreamConfig {
        width: primary_display.width,
        height: primary_display.height,
        fps: 60,
        codec: VideoCodec::H264,
        bitrate_kbps: 10_000,
        pixel_format: PixelFormat::NV12,
        preferred_encoder: None,
        virtual_display: false,
    };

    println!("Starting capture...");
    capture.start(&primary_display.id, &config).await?;

    println!("Waiting for the first new frame...");
    let mut frame = None;
    for _ in 0..10 {
        match capture.next_frame().await {
            Ok(f) => {
                if f.is_new_frame {
                    println!("Successfully captured a new frame! Size: {}x{}", f.width, f.height);
                    frame = Some(f);
                    break;
                } else {
                    println!("Captured a keepalive/timeout frame, waiting...");
                }
            }
            Err(e) => {
                println!("Error capturing frame: {:?}", e);
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    capture.stop().await?;

    if let Some(f) = frame {
        match &f.buffer {
            GpuBuffer::CpuBuffer { data, stride, format, width, height } => {
                println!("Frame buffer type: CpuBuffer, format: {:?}", format);
                if *format == PixelFormat::BGRA {
                    let out_path = "frame.png";
                    println!("Saving frame as PNG to {}...", out_path);
                    
                    // If stride is different from width * 4, we need to copy row by row
                    let mut clean_data = if *stride as usize != *width as usize * 4 {
                        let mut temp = Vec::with_capacity((*width * *height * 4) as usize);
                        for r in 0..*height as usize {
                            let start = r * *stride as usize;
                            let end = start + (*width * 4) as usize;
                            temp.extend_from_slice(&data[start..end]);
                        }
                        temp
                    } else {
                        data.clone()
                    };

                    // Convert BGRA to RGBA by swapping R and B channels
                    for chunk in clean_data.chunks_exact_mut(4) {
                        chunk.swap(0, 2);
                    }

                    image::save_buffer(
                        out_path,
                        &clean_data,
                        *width,
                        *height,
                        image::ColorType::Rgba8,
                    )?;
                    println!("Frame successfully saved to {}!", out_path);
                } else {
                    println!("Unsupported frame format for saving: {:?}", format);
                }
            }
            _ => {
                println!("Captured frame buffer is resident in GPU, cannot read directly in CPU fallback example.");
            }
        }
    } else {
        println!("Failed to capture a frame within timeout!");
    }

    Ok(())
}
