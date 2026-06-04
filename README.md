# lunaris-media

Zero-copy-oriented screen capture, audio capture, input injection, and hardware-accelerated video encoding library for Rust.

`lunaris-media` is currently a Linux-first prototype. The crate exposes a `MediaPipeline` that combines screen capture, FFmpeg video encoding, cpal/Opus audio capture, and cursor events into Tokio channels. It is intended to feed a separate transport layer such as WebRTC; this crate does not implement networking or packetization.

## Current Status

Updated: 2026-06-04

| Area | Status | Notes |
| --- | --- | --- |
| Linux screen capture | Prototype implemented | Factory order is NvFBC, DRM/KMS, PipeWire portal, then X11. Runtime permissions and hardware support vary by backend. |
| NVIDIA NvFBC capture | Prototype implemented | Uses NvFBC with CUDA/system capture paths. Requires supported NVIDIA driver/runtime and a compatible session. |
| DRM/KMS capture | Prototype implemented | Exports active framebuffers as DMA-BUF. Usually requires root or `CAP_SYS_ADMIN` for `DRM_IOCTL_MODE_GETFB2`. |
| PipeWire/Wayland capture | Prototype implemented | Uses XDG Desktop Portal via `ashpd` and PipeWire. Requests user permission at runtime. |
| X11 capture | Prototype implemented | Uses XShm when available, with XGetImage fallback. Produces CPU-backed BGRA buffers. |
| FFmpeg video encoding | Prototype implemented | Probes H.264/H.265/AV1 encoders from the local FFmpeg build and falls back through hardware/software candidates. H.264 is the primary validated target. |
| Linux audio capture | Prototype implemented | Uses cpal input/monitor capture and encodes Opus frames. Requires a working PulseAudio/PipeWire monitor-style device. |
| Cursor tracking | Stub | `LinuxCursorCapture` currently returns a static default cursor state. Real X11/XFixes or PipeWire cursor metadata is still pending. |
| Linux input injection | Prototype implemented | `InputInjector` uses X11 XTest when `DISPLAY` is available, otherwise tries `/dev/uinput`. Non-Linux is a stub. |
| Windows/macOS capture | Not implemented in this repo | Dependencies and cfg declarations exist, but platform backend source files are not present yet. |
| Integration tests/benchmarks | Pending | `cargo check` passes for the library; hardware/runtime validation is still needed. |

## Architecture

```text
MediaPipeline
  |
  |-- Screen capture
  |     Linux: NvFBC -> DRM/KMS -> PipeWire -> X11
  |     Output: GpuBuffer::CudaPointer, GpuBuffer::DmaBuf, or GpuBuffer::CpuBuffer
  |
  |-- Video encoding
  |     FFmpeg encoder candidates: NVENC, VAAPI, QSV, VideoToolbox/AMF cfg paths, software fallback
  |     Output: EncodedVideoFrame
  |
  |-- Audio capture
  |     Linux cpal stream -> Opus encoder
  |     Output: EncodedAudioFrame
  |
  |-- Cursor capture
  |     Current Linux implementation is a stub
  |
  |-- Commands
        RequestKeyframe, SetBitrate, SetFps, Stop
```

Consumers receive `MediaEvent` values from a Tokio channel:

- `VideoFrame(EncodedVideoFrame)`
- `AudioFrame(EncodedAudioFrame)`
- `CursorUpdate(CursorState)`
- `Started`
- `Stopped`
- `Error(String)`

## Quick Start

```rust
use lunaris_media::{MediaEvent, MediaPipeline, StreamConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = StreamConfig {
        width: 1920,
        height: 1080,
        fps: 60,
        bitrate_kbps: 10_000,
        preferred_encoder: Some("nvenc".to_string()), // use "auto", "vaapi", "qsv", or "software" as needed
        ..Default::default()
    };

    let (pipeline, mut events, commands) = MediaPipeline::new(config);

    tokio::spawn(async move {
        if let Err(error) = pipeline.run("default").await {
            eprintln!("pipeline error: {error}");
        }
    });

    commands.send(lunaris_media::PipelineCommand::RequestKeyframe).await?;

    while let Some(event) = events.recv().await {
        match event {
            MediaEvent::VideoFrame(frame) => {
                println!("video: {} bytes, {:?}", frame.data.len(), frame.frame_type);
            }
            MediaEvent::AudioFrame(frame) => {
                println!("audio: {} bytes", frame.data.len());
            }
            MediaEvent::CursorUpdate(state) => {
                println!("cursor: {}, {}", state.x, state.y);
            }
            MediaEvent::Started => println!("pipeline started"),
            MediaEvent::Stopped => break,
            MediaEvent::Error(error) => eprintln!("pipeline event error: {error}"),
        }
    }

    Ok(())
}
```

`preferred_encoder` accepts `nvenc`, `vaapi`, `qsv`, `amf`, `videotoolbox`, `software`, `auto`, or an empty string. Unknown values are treated like auto-selection.

## Build

### Linux Prerequisites

Ubuntu/Debian packages commonly needed by this crate:

```bash
sudo apt install \
    clang pkg-config \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    libpipewire-0.3-dev libopus-dev libasound2-dev \
    libx11-dev libxext-dev libxtst-dev libxfixes-dev libdrm-dev
```

Fedora:

```bash
sudo dnf install \
    clang pkg-config \
    ffmpeg-devel pipewire-devel opus-devel alsa-lib-devel \
    libX11-devel libXext-devel libXtst-devel libXfixes-devel libdrm-devel
```

Arch Linux:

```bash
sudo pacman -S \
    clang pkgconf ffmpeg pipewire opus alsa-lib \
    libx11 libxext libxtst libxfixes libdrm
```

NVIDIA NvFBC/NVENC paths also require a compatible NVIDIA driver and CUDA/NvFBC runtime availability.

### Commands

```bash
cargo check
cargo check --examples
cargo test
```

Run the main capture example:

```bash
cargo run --example capture_encode
```

Write raw encoded video to a file:

```bash
cargo run --example capture_encode -- --output capture.h264
ffplay capture.h264
```

Additional hardware/debug examples:

```bash
cargo run --example test_nvfbc_raw
cargo run --example test_xtest
```

## Runtime Notes

- Wayland/PipeWire capture opens an XDG Desktop Portal screen-cast permission prompt.
- DRM/KMS capture may fail without elevated privileges because exporting the active framebuffer requires privileged DRM ioctls.
- X11 capture requires `DISPLAY` and can use XShm for faster capture when available.
- `/dev/uinput` input injection usually requires permissions through udev rules or root.
- Audio capture depends on the system exposing a monitor or loopback input source.
- Hardware encoder availability depends on the FFmpeg build and installed GPU drivers. `list_available_encoders()` reports encoders found by FFmpeg.

## Project Structure

```text
lunaris-media/
|-- Cargo.toml
|-- README.md
|-- src/
|   |-- lib.rs
|   |-- types.rs
|   |-- error.rs
|   |-- pipeline.rs
|   |-- input.rs
|   |-- capture/
|   |   |-- mod.rs
|   |   |-- gpu_buffer.rs
|   |   |-- linux_nvfbc.rs
|   |   |-- linux_drm.rs
|   |   |-- linux_wayland.rs
|   |   `-- linux_x11.rs
|   |-- encode/
|   |   |-- mod.rs
|   |   `-- ffmpeg.rs
|   |-- audio/
|   |   |-- mod.rs
|   |   `-- linux.rs
|   `-- cursor/
|       |-- mod.rs
|       `-- linux.rs
`-- examples/
    |-- capture_encode.rs
    |-- test_nvfbc_raw.rs
    `-- test_xtest.rs
```

## Public API Highlights

- `MediaPipeline::new(config)` returns `(pipeline, event_rx, command_tx)`.
- `MediaPipeline::run(display_id)` starts capture, encode, audio, and cursor tasks.
- `PipelineCommand` supports keyframe requests, bitrate changes, FPS changes, and graceful stop.
- `GpuBuffer` supports Linux DMA-BUF, Linux CUDA pointer, platform cfg placeholders, and CPU fallback buffers.
- `InputInjector` provides Linux mouse/keyboard injection helpers through XTest or uinput.

## Roadmap

### Next Milestones

- Replace Linux cursor stub with real X11/XFixes and PipeWire metadata support.
- Add integration tests for the Linux backend fallback chain on real hardware.
- Add benchmarks for capture latency, encode latency, FPS stability, and CPU/GPU copy behavior.
- Harden FFmpeg hardware-frame handling for each encoder/backend combination.
- Document required udev/capability setup for DRM capture and uinput injection.

### Future Work

- Implement Windows backend source files for DXGI capture and WASAPI audio.
- Implement macOS backend source files for ScreenCaptureKit and VideoToolbox.
- Validate H.265 and AV1 paths across NVENC, VAAPI, QSV, and software encoders.
- Add dynamic resolution handling, adaptive bitrate control, and multi-monitor/region capture.
- Add HDR/10-bit capture and encode support.

## Dependency Snapshot

| Dependency | Version in `Cargo.toml` |
| --- | --- |
| `ffmpeg-next` | 8 |
| `opus` | 0.3 |
| `cpal` | 0.15 |
| `pipewire` | 0.8 |
| `ashpd` | 0.9 |
| `tokio` | 1 |

## License

MIT, as declared in `Cargo.toml`.
