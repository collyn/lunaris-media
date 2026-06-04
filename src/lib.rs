//! # lunaris-media
//!
//! Zero-copy GPU screen capture and hardware-accelerated video encoding library
//! for Rust.
//!
//! This crate provides a unified API for capturing screen content directly into
//! GPU-resident buffers and encoding it with hardware-accelerated codecs (VAAPI,
//! NVENC, etc.), producing Annex-B H.264 (and later H.265 / AV1) bitstream data
//! ready for transport over WebRTC or other protocols.
//!
//! ## Architecture
//!
//! ```text
//!   PipeWire / DXGI / SCK           FFmpeg hwaccel
//!       ┌──────────┐               ┌──────────┐
//!       │ Capture   │──GPU Buffer──▶│ Encoder  │──Bitstream──▶ Consumer
//!       └──────────┘               └──────────┘
//!                        ┌──────────┐
//!                        │ Audio    │──Opus frames──▶ Consumer
//!                        └──────────┘
//!                        ┌──────────┐
//!                        │ Cursor   │──State──▶ Consumer
//!                        └──────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```no_run
//! use lunaris_media::pipeline::{MediaPipeline, MediaEvent};
//! use lunaris_media::types::StreamConfig;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = StreamConfig::default();
//! let (pipeline, mut events, commands) = MediaPipeline::new(config);
//!
//! // Run the pipeline on a background task
//! tokio::spawn(async move {
//!     if let Err(e) = pipeline.run("default").await {
//!         eprintln!("Pipeline error: {e}");
//!     }
//! });
//!
//! // Consume media events
//! while let Some(event) = events.recv().await {
//!     match event {
//!         MediaEvent::VideoFrame(frame) => { /* send over WebRTC */ }
//!         MediaEvent::AudioFrame(frame) => { /* send over WebRTC */ }
//!         MediaEvent::CursorUpdate(state) => { /* overlay or send */ }
//!         _ => {}
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Platform Support
//!
//! | Platform | Capture Backend       | Encoders                 |
//! |----------|-----------------------|--------------------------|
//! | Linux    | PipeWire / X11        | VAAPI, NVENC, Software   |
//! | Windows  | DXGI Desktop Dup      | NVENC, AMF, QSV          |
//! | macOS    | ScreenCaptureKit      | VideoToolbox             |

pub mod capture;
pub mod encode;
pub mod audio;
pub mod cursor;
pub mod error;
pub mod pipeline;
pub mod types;
pub mod input;

// Re-export primary public API for convenience.
pub use error::MediaError;
pub use pipeline::{MediaEvent, MediaPipeline, PipelineCommand};
pub use types::*;
pub use input::InputInjector;

#[cfg(target_os = "linux")]
pub static X11_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
