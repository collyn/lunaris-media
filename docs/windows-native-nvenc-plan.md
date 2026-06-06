# Windows Native D3D11 Zero-Copy Encoder Plan

## Summary

- Add Windows native encoders that consume D3D11 textures directly.
- Use standalone native NVENC for NVIDIA and native AMF for AMD.
- Keep FFmpeg as fallback for non-Windows platforms, software encoding, Intel QuickSync, and native NVENC initialization failures.
- For AMD, do not fall back to FFmpeg once AMF is explicitly requested or an AMD D3D11 adapter is detected.
- Preserve the current public pipeline API and `VideoEncoder` contract.

## Key Changes

- Add Windows-only `windows_nvenc` and `windows_amf` encoder modules implementing `VideoEncoder`.
- On Windows, choose native NVENC first when `preferred_encoder` is `nvenc`; for `auto`, only try native NVENC on an NVIDIA D3D11 adapter.
- Encode flow:
  - dynamically load `nvEncodeAPI64.dll`;
  - create an NVENC DirectX session with the shared `ID3D11Device`;
  - maintain a ring of D3D11 NV12 input textures registered as NVENC DirectX resources;
  - convert/copy capture `BGRA` textures to NV12 GPU textures with D3D11 VideoProcessor;
  - map the registered input resource, call `NvEncEncodePicture`, lock/copy bitstream output, and unmap/unlock.
- Keep `request_keyframe()` mapped to native NVENC IDR flags; bitrate changes update the target and force IDR until full `NvEncReconfigureEncoder` support is added.
- Native AMF flow mirrors the D3D11 zero-copy shape: dynamically load `amfrt64.dll`, initialize AMF on the shared `ID3D11Device`, convert capture `BGRA` textures to reusable GPU NV12 textures with D3D11 VideoProcessor, wrap those textures as AMF DX11 surfaces, submit to AMF, and copy the returned bitstream buffer.

## Public Interface / Compatibility

- Do not change `StreamConfig`, `EncoderConfig`, `VideoEncoder`, `MediaPipeline`, or event APIs.
- `preferred_encoder: Some("nvenc")` selects native NVENC on Windows first, then FFmpeg fallback.
- `preferred_encoder: Some("amf")` selects native AMF on Windows and fails clearly if H.264/D3D11/AMF runtime requirements are not met.
- `preferred_encoder: None` or `"auto"` tries native NVENC for NVIDIA D3D11 adapters and native AMF for AMD D3D11 adapters.
- AMD D3D11 adapters no longer use FFmpeg AMF fallback in this path.
- Intel D3D11 adapters still go through the FFmpeg QuickSync path.
- `list_available_encoders()` includes `native_nvenc_d3d11` and `native_amf_d3d11` when their vendor runtimes can be loaded.
- FFmpeg remains a dependency for fallback and other platforms.

## Test Plan

- Run `cargo check` on the current host to verify cross-platform code still builds.
- On Windows with NVIDIA:
  - run capture+encode with `preferred_encoder=nvenc`;
  - confirm logs select `native_nvenc_d3d11`;
  - confirm no FFmpeg `avcodec_send_frame` path is used;
  - confirm the hot path avoids D3D11 staging `Map`;
  - verify H.264 decode, startup SPS/PPS, keyframe request, and bitrate reconfigure.
- On Windows with AMD:
  - run capture+encode with `preferred_encoder=amf` or auto on an AMD D3D11 adapter;
  - confirm logs select `native_amf_d3d11`;
  - confirm no FFmpeg AMF path is used;
  - verify H.264 decode, startup SPS/PPS, and keyframe behavior.
- Failure checks:
  - missing `nvEncodeAPI64.dll` falls back to FFmpeg for NVIDIA;
  - missing `amfrt64.dll` fails clearly for AMD without FFmpeg fallback;
  - NVENC session limit falls back to FFmpeg;
  - AMD non-H.264 or missing D3D11 device/context fails clearly without FFmpeg fallback;
  - non-D3D11 buffers fail clearly or use fallback for non-AMD, without crashing.

## Assumptions

- V1 implements native NVENC and native AMF for H.264/D3D11.
- Intel QuickSync follows Sunshine's FFmpeg-backed path in this phase; a native oneVPL/Media SDK wrapper remains a later phase if FFmpeg must also be removed for Intel.
- Native AMF currently has conservative keyframe and bitrate handling; full AMF property-based IDR/reconfigure should be validated on Windows AMD hardware.
- D3D11 VideoProcessor is acceptable for initial GPU-side BGRA-to-NV12 conversion.
- The target bitstream remains Annex-B H.264 for v1 native NVENC.
