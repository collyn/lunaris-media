# Windows Native NVENC Zero-Copy Plan

## Summary

- Add a Windows native NVENC encoder that consumes D3D11 textures directly, following Sunshine's architecture.
- Keep FFmpeg as fallback for non-Windows platforms, software encoding, AMD/Intel, or native NVENC initialization failures.
- Match Sunshine's actual vendor split: standalone native NVENC for NVIDIA; FFmpeg encode sessions for AMD AMF and Intel QuickSync.
- Preserve the current public pipeline API and `VideoEncoder` contract.

## Key Changes

- Add a Windows-only `windows_nvenc` encoder module implementing `VideoEncoder`.
- On Windows, choose native NVENC first when `preferred_encoder` is `nvenc`; for `auto`, only try native NVENC on an NVIDIA D3D11 adapter.
- Encode flow:
  - dynamically load `nvEncodeAPI64.dll`;
  - create an NVENC DirectX session with the shared `ID3D11Device`;
  - maintain a ring of D3D11 NV12 input textures registered as NVENC DirectX resources;
  - convert/copy capture `BGRA` textures to NV12 GPU textures with D3D11 VideoProcessor;
  - map the registered input resource, call `NvEncEncodePicture`, lock/copy bitstream output, and unmap/unlock.
- Keep `request_keyframe()` mapped to native NVENC IDR flags; bitrate changes update the target and force IDR until full `NvEncReconfigureEncoder` support is added.

## Public Interface / Compatibility

- Do not change `StreamConfig`, `EncoderConfig`, `VideoEncoder`, `MediaPipeline`, or event APIs.
- `preferred_encoder: Some("nvenc")` selects native NVENC on Windows first, then FFmpeg fallback.
- `preferred_encoder: None` or `"auto"` tries native NVENC first only for an NVIDIA D3D11 adapter.
- AMD D3D11 adapters go through the FFmpeg AMF path; Intel D3D11 adapters go through the FFmpeg QuickSync path.
- `list_available_encoders()` includes `native_nvenc_d3d11` when NVENC can be loaded.
- FFmpeg remains a dependency for fallback and other platforms.

## Test Plan

- Run `cargo check` on the current host to verify cross-platform code still builds.
- On Windows with NVIDIA:
  - run capture+encode with `preferred_encoder=nvenc`;
  - confirm logs select `native_nvenc_d3d11`;
  - confirm no FFmpeg `avcodec_send_frame` path is used;
  - confirm the hot path avoids D3D11 staging `Map`;
  - verify H.264 decode, startup SPS/PPS, keyframe request, and bitrate reconfigure.
- Failure checks:
  - missing `nvEncodeAPI64.dll` falls back to FFmpeg;
  - NVENC session limit falls back to FFmpeg;
  - non-D3D11 buffers fail clearly or use fallback, without crashing.

## Assumptions

- V1 implements native NVENC only.
- AMD AMF and Intel QuickSync follow Sunshine's FFmpeg-backed path in this phase; fully native AMF/oneVPL wrappers remain later phases if FFmpeg must be removed for those vendors too.
- D3D11 VideoProcessor is acceptable for initial GPU-side BGRA-to-NV12 conversion.
- The target bitstream remains Annex-B H.264 for v1 native NVENC.
