use lunaris_media::capture::{self, ScreenCapture, CapturedFrame};
use lunaris_media::capture::gpu_buffer::GpuBuffer;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,lunaris_media=debug"),
    )
    .init();

    println!("=== Lunaris PipeWire Capture Test ===");
    println!("Creating screen capture...");

    let mut capture = capture::create_screen_capture()?;
    let displays = capture.list_displays().await?;
    println!("Found {} display(s):", displays.len());
    for d in &displays {
        println!("  {} ({}) {}x{}", d.id, d.name, d.width, d.height);
    }

    let display_id = displays.first().map(|d| d.id.as_str()).unwrap_or("default");
    println!("\nStarting capture on '{}'...", display_id);
    let mut config = lunaris_media::types::StreamConfig::default();
    config.width = 1920;
    config.height = 1080;
    config.fps = 30;
    capture.start(display_id, &config).await?;

    println!("Capturing 5 frames...");
    for i in 0..5 {
        match capture.next_frame().await {
            Ok(frame) => {
                let (data, w, h, fmt) = match &frame.buffer {
                    GpuBuffer::CpuBuffer { data, width, height, format, .. } => {
                        (data, *width, *height, format)
                    }
                    other => {
                        println!("  Frame {}: non-CPU buffer type: {:?}", i + 1, std::mem::discriminant(other));
                        continue;
                    }
                };

                let non_zero = data.iter().filter(|&&b| b != 0).count();
                let total = data.len();
                println!(
                    "  Frame {}: {}x{} format={:?} bytes={} non_zero={} ({:.1}%)",
                    i + 1, w, h, fmt, total, non_zero,
                    non_zero as f64 / total as f64 * 100.0
                );

                // Save first frame as PPM (portable pixmap)
                if i == 0 && !data.is_empty() {
                    let ppm_path = "/tmp/lunaris_test_frame.ppm";
                    let mut file = std::fs::File::create(ppm_path)?;
                    // PPM format: P6 header, RGB data
                    write!(file, "P6\n{} {}\n255\n", w, h)?;
                    // Convert BGRA to RGB
                    for pixel in data.chunks(4) {
                        if pixel.len() >= 3 {
                            file.write_all(&[pixel[2], pixel[1], pixel[0]])?; // BGR → RGB
                        }
                    }
                    println!("  → Saved first frame to {}", ppm_path);
                    println!("    View with: feh {} or display {}", ppm_path, ppm_path);

                    // Also save as raw BGRA for debugging
                    let raw_path = "/tmp/lunaris_test_frame.bgra";
                    std::fs::write(raw_path, data)?;
                    println!("  → Saved raw BGRA to {}", raw_path);
                    println!("    View with: ffplay -f rawvideo -pixel_format bgra -video_size {}x{} {}", w, h, raw_path);
                }
            }
            Err(e) => {
                println!("  Frame {}: ERROR: {}", i + 1, e);
            }
        }
    }

    println!("\nStopping capture...");
    capture.stop().await?;
    println!("Done!");
    Ok(())
}
