//! Colour round-trip helper: encodes 8 colour bars with a chosen encoderCscMode.
//!
//! Feeds 30 CPU BGRA frames made of 8 vertical bars (white, yellow, cyan,
//! green, magenta, red, blue, grey 128) to the encoder and writes the Annex B
//! bitstream to a file, so the decoded colours can be compared with the input.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example color_roundtrip -- <csc 0..=3> [encoder] <out.h264>
//! # e.g. csc 2 = Rec. 709 limited, 3 = Rec. 709 full, 0 = Rec. 601 limited
//! cargo run --release --example color_roundtrip -- 2 software /tmp/bars.h264
//! ```
//!
//! With two arguments only, the second one is the output path and the
//! encoder defaults to "software".

use std::io::Write;

use lunaris_media::capture::gpu_buffer::GpuBuffer;
use lunaris_media::colorspace::from_client_config;
use lunaris_media::encode::{self, EncoderConfig};
use lunaris_media::types::{PixelFormat, StreamConfig, VideoCodec};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FPS: u32 = 30;
const FRAMES: u64 = 30;

/// Bar colours in BGRA byte order (B, G, R, A).
const BARS: [[u8; 4]; 8] = [
    [255, 255, 255, 255], // white
    [0, 255, 255, 255],   // yellow
    [255, 255, 0, 255],   // cyan
    [0, 255, 0, 255],     // green
    [255, 0, 255, 255],   // magenta
    [0, 0, 255, 255],     // red
    [255, 0, 0, 255],     // blue
    [128, 128, 128, 255], // grey
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: color_roundtrip <csc 0..=3> [encoder] <output.h264>");
        std::process::exit(2);
    }
    let csc: i32 = args[1].parse()?;
    if !(0..=3).contains(&csc) {
        return Err("csc must be 0..=3".into());
    }
    let (encoder_pref, output_path) = if args.len() >= 4 {
        (args[2].clone(), args[3].clone())
    } else {
        ("software".to_string(), args[2].clone())
    };

    let config = StreamConfig {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        codec: VideoCodec::H264,
        pixel_format: PixelFormat::BGRA,
        preferred_encoder: Some(encoder_pref.clone()),
        colorspace: from_client_config(csc, 0, false),
        ..Default::default()
    };
    println!("colourspace: {:?}", config.colorspace);

    let mut encoder = encode::create_encoder()?;
    encoder.initialize(&EncoderConfig {
        codec: config.codec,
        width: config.width,
        height: config.height,
        fps: config.fps,
        bitrate_kbps: config.bitrate_kbps,
        preferred_hw: StreamConfig::parse_encoder_preference(&encoder_pref),
        force_ffmpeg: StreamConfig::encoder_prefers_ffmpeg(&encoder_pref),
        colorspace: config.colorspace,
        ..EncoderConfig::default()
    })?;
    println!("encoder: {:?}", encoder.encoder_info());

    let mut data = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for _ in 0..HEIGHT {
        for x in 0..WIDTH {
            data.extend_from_slice(&BARS[(x * BARS.len() as u32 / WIDTH) as usize]);
        }
    }

    let mut out = std::fs::File::create(&output_path)?;
    let mut packets = 0usize;
    let mut bytes = 0usize;
    for i in 0..FRAMES {
        let buffer = GpuBuffer::CpuBuffer {
            data: data.clone(),
            stride: WIDTH * 4,
            format: PixelFormat::BGRA,
            width: WIDTH,
            height: HEIGHT,
        };
        let pts_us = i * 1_000_000 / FPS as u64;
        for frame in encoder.encode(&buffer, pts_us)? {
            out.write_all(&frame.data)?;
            packets += 1;
            bytes += frame.data.len();
        }
    }
    for frame in encoder.flush()? {
        out.write_all(&frame.data)?;
        packets += 1;
        bytes += frame.data.len();
    }
    println!("wrote {packets} packets, {bytes} bytes to {output_path}");
    Ok(())
}
