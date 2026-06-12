# lunaris-media

Zero-copy-oriented screen capture, audio capture, input injection, and hardware-accelerated video encoding library for Rust.

`lunaris-media` exposes a `MediaPipeline` that combines screen capture, hardware-accelerated video encoding, system audio capture, and cursor events into Tokio channels. It is designed to feed a separate transport layer such as WebRTC; this crate does not implement networking or packetization.

## Current Status

Updated: 2026-06-12

### Linux

| Area | Status | Notes |
| --- | --- | --- |
| Screen capture | Implemented | Factory order: NvFBC → DRM/KMS → PipeWire portal → X11. Runtime fallback chain. |
| NVIDIA NvFBC capture | Implemented | CUDA/system capture paths. Requires NVIDIA driver and CUDA/NvFBC runtime. |
| DRM/KMS capture | Implemented | Zero-copy DMA-BUF export. Requires `CAP_SYS_ADMIN` or root for `DRM_IOCTL_MODE_GETFB2`. |
| PipeWire/Wayland capture | Implemented | XDG Desktop Portal via `ashpd`. Supports embedded or hidden cursor modes. |
| X11 capture | Implemented | XShm with XGetImage fallback. Produces CPU-backed BGRA buffers. |
| FFmpeg video encoding | Implemented | Probes NVENC, VAAPI, QSV, AMF, VideoToolbox, software fallback. H.264/H.265/AV1. |
| Audio capture | Implemented | cpal + Opus (PulseAudio/PipeWire monitor capture). |
| Cursor tracking | Implemented | X11/XFixes cursor position + shape. PipeWire `SPA_META_Cursor` metadata per-frame. |
| Input injection | Implemented | XTest when `DISPLAY` is available, `/dev/uinput` otherwise. |
| Virtual display | Implemented | XRandR VIRTUAL output management. |

### Windows

| Area | Status | Notes |
| --- | --- | --- |
| Screen capture | Implemented | DXGI Desktop Duplication with GDI fallback (for RDP/VM/hybrid GPU). |
| NVENC encoding | Implemented | Native D3D11 zero-copy. H.264 + H.265. Loads `nvEncodeAPI64.dll` at runtime. |
| AMF encoding | Implemented | Native D3D11 zero-copy. H.264 + H.265 + AV1. Loads `amfrt64.dll` at runtime. |
| QSV encoding | Implemented | Native D3D11 zero-copy. H.264 + H.265 + AV1. Loads `libvpl.dll` (oneVPL 2.x) at runtime. |
| Audio capture | Implemented | WASAPI loopback capture. |
| Cursor tracking | Implemented | Native Windows cursor position + shape. |
| Input injection | Implemented | `SendInput` for mouse/keyboard. |
| Virtual display | Implemented | IddCx driver communication (requires `usbmmidd` or compatible driver installed). |

### macOS

| Area | Status | Notes |
| --- | --- | --- |
| All subsystems | Stub only | Dependencies declared in `Cargo.toml` but no source files exist. |

## Architecture

```text
MediaPipeline
  |
  |-- Screen capture
  |     Linux:  NvFBC → DRM/KMS → PipeWire → X11
  |     Windows: DXGI Desktop Duplication (GDI fallback)
  |     Output: GpuBuffer::DmaBuf, GpuBuffer::CudaPointer, GpuBuffer::D3D11Texture, or GpuBuffer::CpuBuffer
  |
  |-- Video encoding
  |     Linux:   FFmpeg (NVENC → VAAPI → QSV → AMF → software)
  |     Windows: Native NVENC / Native AMF / Native QSV (D3D11 zero-copy, no FFmpeg)
  |     Codecs:  H.264, H.265/HEVC, AV1
  |     Output:  EncodedVideoFrame
  |
  |-- Audio capture
  |     Linux:   cpal + Opus (PulseAudio/PipeWire)
  |     Windows: WASAPI loopback
  |     Output:  EncodedAudioFrame
  |
  |-- Cursor capture
  |     Linux:   X11/XFixes (position + shape), PipeWire SPA_META_Cursor (per-frame metadata)
  |     Windows: Native cursor tracking
  |     Output:  CursorState (position, shape, image)
  |
  |-- Input injection
  |     Linux:   XTest + uinput
  |     Windows: SendInput
  |
  |-- Virtual display
  |     Linux:   XRandR VIRTUAL output
  |     Windows: IddCx driver (usbmmidd)
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
        preferred_encoder: Some("nvenc".to_string()), // use "auto", "vaapi", "qsv", "amf", or "software"
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

Ubuntu/Debian:

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

### Windows Prerequisites

- Rust toolchain with MSVC target (`x86_64-pc-windows-msvc`)
- The `windows` crate (v0.58) handles Win32 API bindings automatically
- Native encoders load DLLs at runtime:
  - **NVENC**: `nvEncodeAPI64.dll` (NVIDIA driver)
  - **AMF**: `amfrt64.dll` (AMD driver)
  - **QSV**: `libvpl.dll` (Intel oneVPL runtime)
- Virtual display requires an IddCx driver installed (e.g. [usbmmidd](https://github.com/ge9/IddSampleDriver))

### Build Commands

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
cargo run --example capture_encode -- --codec h265 --output capture.h265
cargo run --example capture_encode -- --codec av1 --output capture.av1
ffplay capture.h264
```

Additional hardware/debug examples:

```bash
cargo run --example test_nvfbc_raw
cargo run --example test_xtest
```

## Runtime Notes

- **Wayland/PipeWire**: Opens an XDG Desktop Portal screen-cast permission prompt. Cursor metadata is extracted from `SPA_META_Cursor` per-frame when available.
- **DRM/KMS**: May fail without elevated privileges because exporting the active framebuffer requires privileged DRM ioctls.
- **X11**: Requires `DISPLAY`. Uses XShm for faster capture when available. Cursor tracking via XFixes.
- **`/dev/uinput`**: Input injection usually requires permissions through udev rules or root.
- **Audio**: Depends on the system exposing a monitor or loopback input source.
- **Windows encoders**: Native NVENC/AMF/QSV encoders use D3D11 zero-copy — frames stay on the GPU. No FFmpeg dependency on Windows.
- **Windows virtual display**: Requires an IddCx driver (e.g. `usbmmidd`) installed and running.

## Project Structure

```text
lunaris-media/
├── Cargo.toml
├── README.md
├── .github/workflows/ci.yml
├── src/
│   ├── lib.rs
│   ├── types.rs
│   ├── error.rs
│   ├── pipeline.rs
│   ├── input.rs
│   ├── capture/
│   │   ├── mod.rs
│   │   ├── gpu_buffer.rs
│   │   ├── linux_nvfbc.rs
│   │   ├── linux_drm.rs
│   │   ├── linux_wayland.rs
│   │   ├── linux_x11.rs
│   │   ├── windows.rs
│   │   └── virtual_display/
│   │       ├── mod.rs
│   │       ├── linux.rs
│   │       └── windows.rs
│   ├── encode/
│   │   ├── mod.rs
│   │   ├── ffmpeg.rs
│   │   ├── windows_nvenc.rs
│   │   ├── windows_amf.rs
│   │   └── windows_qsv.rs
│   ├── audio/
│   │   ├── mod.rs
│   │   ├── linux.rs
│   │   └── windows.rs
│   └── cursor/
│       ├── mod.rs
│       ├── linux.rs
│       └── windows.rs
└── examples/
    ├── capture_encode.rs
    ├── capture_frame.rs
    ├── test_nvfbc_raw.rs
    └── test_xtest.rs
```

## Public API Highlights

- `MediaPipeline::new(config)` returns `(pipeline, event_rx, command_tx)`.
- `MediaPipeline::run(display_id)` starts capture, encode, audio, and cursor tasks.
- `PipelineCommand` supports keyframe requests, bitrate changes, FPS changes, and graceful stop.
- `GpuBuffer` supports DMA-BUF, CUDA pointer, D3D11 texture, and CPU fallback buffers.
- `InputInjector` provides mouse/keyboard injection (XTest/uinput on Linux, SendInput on Windows).
- `list_available_encoders()` reports native hardware encoders detected on the system.
- `FrameCursorMeta` carries per-frame cursor metadata from PipeWire capture.

## Encoder Support Matrix

| Encoder | H.264 | H.265 | AV1 | Platform | Backend |
| --- | --- | --- | --- | --- | --- |
| NVENC (FFmpeg) | ✅ | ✅ | ✅ | Linux | FFmpeg → nvenc |
| VAAPI (FFmpeg) | ✅ | ✅ | ✅ | Linux | FFmpeg → vaapi |
| QSV (FFmpeg) | ✅ | ✅ | ✅ | Linux | FFmpeg → qsv |
| Software (FFmpeg) | ✅ | ✅ | ✅ | Linux | libx264/libx265/libsvtav1 |
| NVENC (native) | ✅ | ✅ | ❌ | Windows | `nvEncodeAPI64.dll` D3D11 zero-copy |
| AMF (native) | ✅ | ✅ | ✅ | Windows | `amfrt64.dll` D3D11 zero-copy |
| QSV (native) | ✅ | ✅ | ✅ | Windows | `libvpl.dll` D3D11 zero-copy |

## CI/CD

GitHub Actions runs on every push and PR:

- **Tier 1** (every commit): `cargo fmt`, `cargo clippy`, `cargo check`, `cargo test` on Linux + Windows
- **Tier 2** (PR only): Integration tests with software H.264/H.265/AV1 encoding on Linux with Xvfb

## Dependency Snapshot

| Dependency | Version | Platform |
| --- | --- | --- |
| `ffmpeg-next` | 8 | Linux |
| `pipewire` | 0.8 | Linux |
| `ashpd` | 0.9 | Linux |
| `x11` | 2.21 | Linux |
| `nix` | 0.29 | Linux |
| `drm-fourcc` | 2.2 | Linux |
| `nvfbc` | 0.2 | Linux |
| `evdev` | 0.11 | Linux |
| `windows` | 0.58 | Windows |
| `opus` | 0.3 | Cross-platform |
| `cpal` | 0.15 | Cross-platform |
| `tokio` | 1 | Cross-platform |

## License

MIT, as declared in `Cargo.toml`.
