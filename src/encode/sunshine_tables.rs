// Wired in by the next step (Phase 3c-2).
#![allow(dead_code)]
//! Sunshine's FFmpeg encoder tables and the small pure functions around them,
//! transcribed as plain data (Phase 3c-1).
//!
//! Reference: Sunshine commit f186813, `/mnt/nvme/projects/_ref/Sunshine`.
//! Every value carries the Sunshine source line it was copied from
//! (`video.cpp:NNN`, `video.h:NNN`, `config.cpp:NNN`, `nvenc_config.h:NNN`).
//! Nothing here is "improved": where Sunshine does something odd, so do we.
//!
//! The module is pure data plus small functions with no dependency beyond
//! `std`, so it compiles and is unit-tested on every platform. `#ifdef`
//! conditions of the C++ are kept as [`Platform`] data on each [`EncoderDef`]
//! rather than being compiled out.
//!
//! Not ported (see the report of the transcription for the reasons):
//! the native Windows `nvenc` definition (`video.cpp:716-760`, an
//! `encoder_platform_formats_nvenc`, not an avcodec encoder), the colour
//! space fields of the codec context (`video.cpp:2115-2120`),
//! `framerate_to_rational` (`video.cpp:2057-2059`), the HDR metadata and the
//! SPS/VPS `inject` computation (`video.cpp:2317-2377`).

// ---------------------------------------------------------------------------
// Platforms
// ---------------------------------------------------------------------------

/// Platform an encoder definition is compiled for in Sunshine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// `_WIN32`.
    Windows,
    /// `__linux__ || linux || __linux`.
    Linux,
    /// `__APPLE__`.
    MacOs,
    /// `__FreeBSD__`.
    FreeBsd,
}

/// All four platforms (`software` is compiled unconditionally, `video.cpp:1161`).
const ALL_PLATFORMS: &[Platform] = &[
    Platform::Windows,
    Platform::Linux,
    Platform::MacOs,
    Platform::FreeBsd,
];

// ---------------------------------------------------------------------------
// FFmpeg-ish enums (names only; no FFmpeg types in this module)
// ---------------------------------------------------------------------------

/// `AVHWDeviceType` values used by the tables (`AV_HWDEVICE_TYPE_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwDeviceType {
    /// `AV_HWDEVICE_TYPE_NONE`.
    None,
    /// `AV_HWDEVICE_TYPE_D3D11VA`.
    D3d11va,
    /// `AV_HWDEVICE_TYPE_QSV`.
    Qsv,
    /// `AV_HWDEVICE_TYPE_CUDA`.
    Cuda,
    /// `AV_HWDEVICE_TYPE_VAAPI`.
    Vaapi,
    /// `AV_HWDEVICE_TYPE_VULKAN`.
    Vulkan,
    /// `AV_HWDEVICE_TYPE_VIDEOTOOLBOX`.
    VideoToolbox,
}

/// `AVPixelFormat` values used by the tables (`AV_PIX_FMT_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixFmt {
    /// `AV_PIX_FMT_NONE`.
    None,
    /// `AV_PIX_FMT_D3D11`.
    D3d11,
    /// `AV_PIX_FMT_QSV`.
    Qsv,
    /// `AV_PIX_FMT_CUDA`.
    Cuda,
    /// `AV_PIX_FMT_VAAPI`.
    Vaapi,
    /// `AV_PIX_FMT_VULKAN`.
    Vulkan,
    /// `AV_PIX_FMT_VIDEOTOOLBOX`.
    VideoToolbox,
    /// `AV_PIX_FMT_NV12`.
    Nv12,
    /// `AV_PIX_FMT_P010`.
    P010,
    /// `AV_PIX_FMT_YUV444P`.
    Yuv444p,
    /// `AV_PIX_FMT_YUV444P16`.
    Yuv444p16,
    /// `AV_PIX_FMT_VUYX`.
    Vuyx,
    /// `AV_PIX_FMT_XV30`.
    Xv30,
    /// `AV_PIX_FMT_YUV420P`.
    Yuv420p,
    /// `AV_PIX_FMT_YUV420P10` (native-endian alias).
    Yuv420p10,
    /// `AV_PIX_FMT_YUV444P10` (native-endian alias).
    Yuv444p10,
}

/// The `init_avcodec_hardware_input_buffer` callback an encoder passes as the
/// last constructor argument of `encoder_platform_formats_avcodec`
/// (`video.h:267-275`). Named only; the functions live in Sunshine's platform
/// code (declared at `video.cpp:196-223`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwInputInit {
    /// `nullptr` (software encoder).
    None,
    /// `dxgi_init_avcodec_hardware_input_buffer` (`video.cpp:196`).
    Dxgi,
    /// `vaapi_init_avcodec_hardware_input_buffer` (`video.cpp:203`).
    Vaapi,
    /// `cuda_init_avcodec_hardware_input_buffer` (`video.cpp:210`).
    Cuda,
    /// `vt_init_avcodec_hardware_input_buffer` (`video.cpp:217`).
    VideoToolbox,
    /// `vulkan_init_avcodec_hardware_input_buffer` (`video.cpp:223`).
    Vulkan,
}

// ---------------------------------------------------------------------------
// Encoder flags (video.cpp:412-426) and codec capability flags (video.h:340-348)
// ---------------------------------------------------------------------------

/// `enum flag_e : uint32_t` of `video.cpp:412-426`: the per-encoder flags that
/// go into `encoder_t::flags`.
pub mod flag {
    /// Default flags (`video.cpp:413`).
    pub const DEFAULT: u32 = 0;
    /// Capture and encoding can run concurrently on separate threads (`video.cpp:414`).
    pub const PARALLEL_ENCODING: u32 = 1 << 1;
    /// When HEVC is too heavy (`video.cpp:415`).
    pub const H264_ONLY: u32 = 1 << 2;
    /// Some encoders don't like it when you have an infinite GOP_SIZE. e.g. VAAPI (`video.cpp:416`).
    pub const LIMITED_GOP_SIZE: u32 = 1 << 3;
    /// Never use multiple slices. Older intel iGPU's ruin it for everyone else (`video.cpp:417`).
    pub const SINGLE_SLICE_ONLY: u32 = 1 << 4;
    /// Use a VBR rate control mode to simulate CBR (`video.cpp:418`).
    pub const CBR_WITH_VBR: u32 = 1 << 5;
    /// Use FF_COMPLIANCE_UNOFFICIAL compliance mode (`video.cpp:419`).
    pub const RELAXED_COMPLIANCE: u32 = 1 << 6;
    /// Don't set rc_buffer_size (`video.cpp:420`).
    pub const NO_RC_BUF_LIMIT: u32 = 1 << 7;
    /// Support reference frames invalidation (`video.cpp:421`).
    pub const REF_FRAMES_INVALIDATION: u32 = 1 << 8;
    /// This is an encoder of last resort and we want to aggressively probe for a better one (`video.cpp:422`).
    pub const ALWAYS_REPROBE: u32 = 1 << 9;
    /// Encoder may support 4:4:4 chroma sampling depending on hardware (`video.cpp:423`).
    pub const YUV444_SUPPORT: u32 = 1 << 10;
    /// Encoder supports async teardown on a different thread (`video.cpp:424`).
    pub const ASYNC_TEARDOWN: u32 = 1 << 11;
    /// Use fixed small GOP size (encoder doesn't support on-demand IDR frames) (`video.cpp:425`).
    pub const FIXED_GOP_SIZE: u32 = 1 << 12;
}

/// `encoder_t::flag_e` of `video.h:340-348`: the per-codec probe capability
/// bits (`codec_t::capabilities`). Every table below initialises them empty
/// (`{},  // capabilities`); they are set by Sunshine's runtime probe, which is
/// not ported. Listed so the numbering is not lost.
pub mod codec_flag {
    /// Indicates the encoder is supported (`video.h:341`).
    pub const PASSED: usize = 0;
    /// Set maximum reference frames (`video.h:342`).
    pub const REF_FRAMES_RESTRICT: usize = 1;
    /// HDR support (`video.h:343`).
    pub const DYNAMIC_RANGE: usize = 2;
    /// YUV 4:4:4 support (`video.h:344`).
    pub const YUV444: usize = 3;
    /// YUV 4:4:4 HDR support (`video.h:345`).
    pub const DYNAMIC_RANGE_YUV444: usize = 4;
    /// AMD encoder with VAAPI doesn't add VUI parameters to SPS (`video.h:346`).
    pub const VUI_PARAMETERS: usize = 5;
    /// Maximum number of flags (`video.h:347`).
    pub const MAX_FLAGS: usize = 6;
}

#[inline]
fn has(flags: u32, flag: u32) -> bool {
    flags & flag != 0
}

// ---------------------------------------------------------------------------
// Option values
// ---------------------------------------------------------------------------

/// Where an option's value comes from: Sunshine's
/// `std::variant<int, int *, std::optional<int> *, std::function<int()>,
/// std::string, std::string *, std::function<const std::string(const config_t &)>>`
/// (`video.h:389`), consumed by `handle_option` (`video.cpp:2196-2227`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionValue {
    /// `int` literal (`av_dict_set_int`, `video.cpp:2199-2201`). C++ `bool`
    /// literals (`false`) are stored as `Int(0)`; `std::numeric_limits<int>::max()`
    /// as `Int(i32::MAX)`.
    Int(i64),
    /// `std::string` literal (`av_dict_set`, `video.cpp:2213-2215`).
    Str(&'static str),
    /// `int *`, `std::optional<int> *` or `std::string *` into `config::video`
    /// (`video.cpp:2202-2220`). Resolved with [`config_default`]; `None`
    /// (empty optional / empty string) is skipped, as `handle_option` does.
    Config(ConfigKey),
    /// `std::function<int()>` or `std::function<const std::string(const config_t &)>`
    /// lambda (`video.cpp:2210-2212`, `2221-2223`). Resolved with [`compute`].
    Computed(Computed),
}

/// One variant per distinct `config::video.*` member the tables point at.
///
/// The doc of each variant gives the member, the default's citation in
/// `config.cpp` (`video_t video {...}`, `config.cpp:737-821`) and the numeric
/// value the default resolves to. See [`config_default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigKey {
    // --- nvenc (`config::video.nv_legacy`, config.h:92-98) ---
    /// `nv_legacy.preset` (`std::string`): `config.cpp:1651` = `ffmpeg_preset_from_quality(nv.quality_preset)`
    /// (`config.cpp:68-70`, `"p{}"`) with `nvenc_config.h:32` `quality_preset = 1` => `"p1"`.
    NvLegacyPreset,
    /// `nv_legacy.multipass` (`int`): `config.cpp:1652-1654` with `nvenc_config.h:35`
    /// `two_pass = quarter_resolution` => `NV_ENC_TWO_PASS_QUARTER_RESOLUTION` = 1.
    NvLegacyMultipass,
    /// `nv_legacy.spatial_aq` (`int`): `config.cpp:1656` = `nv.adaptive_quantization`,
    /// `nvenc_config.h:44` `false` => 0.
    NvLegacySpatialAq,
    /// `nv_legacy.h264_coder` (`int`): `config.cpp:1655` with `nvenc_config.h:59`
    /// `h264_cavlc = false` => `NV_ENC_H264_ENTROPY_CODING_MODE_CABAC` = 1.
    NvLegacyH264Coder,
    // --- quicksync (`config::video.qsv`, config.h:100-104) ---
    /// `qsv.qsv_preset` (`optional<int>`): `config.cpp:757` `qsv::medium` => `config.cpp:360` = 4.
    QsvPreset,
    /// `qsv.qsv_cavlc` (`optional<int>`): `config.cpp:758` `qsv::_auto` => `config.cpp:370` `_auto = false` = 0.
    QsvCavlc,
    // --- amdvce (`config::video.amd`, config.h:106-121) ---
    /// `amd.amd_preanalysis` (`optional<int>`): `config.cpp:773` = 0.
    AmdPreanalysis,
    /// `amd.amd_quality_av1` (`optional<int>`): `config.cpp:772` `quality_av1_e::balanced`
    /// => `config.cpp:166` `AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_BALANCED` => `config.cpp:121` = 70.
    AmdQualityAv1,
    /// `amd.amd_quality_hevc` (`optional<int>`): `config.cpp:771` `quality_hevc_e::balanced`
    /// => `config.cpp:175` => `config.cpp:124` `AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_BALANCED` = 5.
    AmdQualityHevc,
    /// `amd.amd_quality_h264` (`optional<int>`): `config.cpp:770` `quality_h264_e::balanced`
    /// => `config.cpp:184` => `config.cpp:127` `AMF_VIDEO_ENCODER_QUALITY_PRESET_BALANCED` = 0.
    AmdQualityH264,
    /// `amd.amd_rc_av1` (`optional<int>`): `config.cpp:768` `rc_av1_e::vbr_latency`
    /// => `config.cpp:193` => `config.cpp:131` `..._AV1_RATE_CONTROL_METHOD_LATENCY_CONSTRAINED_VBR` = 1.
    AmdRcAv1,
    /// `amd.amd_rc_hevc` (`optional<int>`): `config.cpp:767` `rc_hevc_e::vbr_latency`
    /// => `config.cpp:203` => `config.cpp:135` `..._HEVC_RATE_CONTROL_METHOD_LATENCY_CONSTRAINED_VBR` = 1.
    AmdRcHevc,
    /// `amd.amd_rc_h264` (`optional<int>`): `config.cpp:766` `rc_h264_e::vbr_latency`
    /// => `config.cpp:213` => `config.cpp:139` `AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_LATENCY_CONSTRAINED_VBR` = 3.
    AmdRcH264,
    /// `amd.amd_usage_av1` (`optional<int>`): `config.cpp:765` `usage_av1_e::ultralowlatency`
    /// => `config.cpp:225` => `config.cpp:142` `AMF_VIDEO_ENCODER_AV1_USAGE_ULTRA_LOW_LATENCY` = 2.
    AmdUsageAv1,
    /// `amd.amd_usage_hevc` (`optional<int>`): `config.cpp:764` `usage_hevc_e::ultralowlatency`
    /// => `config.cpp:236` => `config.cpp:146` `AMF_VIDEO_ENCODER_HEVC_USAGE_ULTRA_LOW_LATENCY` = 1.
    AmdUsageHevc,
    /// `amd.amd_usage_h264` (`optional<int>`): `config.cpp:763` `usage_h264_e::ultralowlatency`
    /// => `config.cpp:247` => `config.cpp:151` `AMF_VIDEO_ENCODER_USAGE_ULTRA_LOW_LATENCY` = 1.
    AmdUsageH264,
    /// `amd.amd_vbaq` (`optional<int>`): `config.cpp:774` = 1.
    AmdVbaq,
    /// `amd.amd_enforce_hrd` (`optional<int>`): `config.cpp:769` = 0.
    AmdEnforceHrd,
    /// `amd.amd_max_au_size` (`optional<int>`): `config.cpp:775` `{}` ("disabled by default") => unset => `None`.
    AmdMaxAuSize,
    /// `amd.amd_coder` (`int`): `config.cpp:776` `amd::coder_e::_auto` => `config.cpp:254`
    /// `AMF_VIDEO_ENCODER_UNDEFINED` => `config.cpp:155` = 0.
    AmdCoder,
    // --- vaapi (`config::video.vaapi`, config.h:130-136) ---
    /// `vaapi.blbrc` (`optional<int>`): `config.cpp:787` = 0.
    VaapiBlbrc,
    // --- vulkan (`config::video.vk`, config.h:138-142) --- not used by Lunaris yet
    /// `vk.tune` (`int`): `config.cpp:795` = 2.
    VkTune,
    /// `vk.rc_mode` (`int`): `config.cpp:796` = 2.
    VkRcMode,
    // --- videotoolbox (`config::video.vt`, config.h:123-128) ---
    /// `vt.vt_allow_sw` (`int`): `config.cpp:780` = 0.
    VtAllowSw,
    /// `vt.vt_require_sw` (`int`): `config.cpp:781` = 0.
    VtRequireSw,
    /// `vt.vt_realtime` (`int`): `config.cpp:782` = 1.
    VtRealtime,
    // --- software (`config::video.sw`, config.h:81-85) ---
    /// `sw.svtav1_preset` (`optional<int>`): `config.cpp:747` = 11 ("superfast").
    SwSvtav1Preset,
    /// `sw.sw_preset` (`std::string`): `config.cpp:745` = `"superfast"`.
    SwPreset,
    /// `sw.sw_tune` (`std::string`): `config.cpp:746` = `"zerolatency"`.
    SwTune,
}

/// A resolved option value (`int` or string), what `handle_option` hands to
/// `av_dict_set_int` / `av_dict_set`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigDefault {
    /// Integer value.
    Int(i64),
    /// String value.
    Str(&'static str),
}

/// One variant per lambda in the tables. See [`compute`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Computed {
    /// `log_to_dbg` of `amdvce` AV1/HEVC/H.264 (`video.cpp:1003-1005`,
    /// `1028-1030`, `1068-1070`): `config::sunshine.min_log_level < 2 ? 1 : 0`.
    AmdLogToDbg,
    /// `level` of `amdvce` HEVC (`video.cpp:1040-1050`): `"5.1"`, `"5.2"` or `"auto"`.
    AmdHevcLevel,
    /// `low_power` fallback of `quicksync` HEVC (`video.cpp:939-941`):
    /// `config::video.qsv.qsv_slow_hevc ? 0 : 1`.
    QsvHevcLowPower,
}

// ---------------------------------------------------------------------------
// Sunshine defaults referenced by the lambdas and helpers
// ---------------------------------------------------------------------------

/// `config::sunshine.min_log_level` default (`config.cpp:898`, `sunshine_t sunshine {...}`).
pub const SUNSHINE_DEFAULT_MIN_LOG_LEVEL: i32 = 2;
/// `config::video.qsv.qsv_slow_hevc` default (`config.cpp:759`).
pub const DEFAULT_QSV_SLOW_HEVC: bool = false;
/// `config::video.max_bitrate` default (`config.cpp:819`); `0` = no cap.
pub const DEFAULT_MAX_BITRATE: i32 = 0;
/// `config::video.min_threads` default (`config.cpp:743`).
pub const DEFAULT_MIN_THREADS: i32 = 2;
/// `config::video.nv_legacy.vbv_percentage_increase` default: `config.cpp:1657`
/// = `nv.vbv_percentage_increase`, `nvenc_config.h:38` = 0.
pub const DEFAULT_NV_LEGACY_VBV_PERCENTAGE_INCREASE: i32 = 0;

// Constants from `ffnvcodec/nvEncodeAPI.h`. That header is NOT part of the
// Sunshine checkout (`third-party/build-deps` is an empty submodule), so these
// numeric values are the NVENC SDK's, not read from the reference tree. The
// symbolic names are the ones Sunshine uses at the cited lines.
/// `NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY` (used at `video.cpp:793`, `815`, `844`).
const NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY: i64 = 3;
/// `NV_ENC_PARAMS_RC_CBR` (used at `video.cpp:794`, `816`, `845`).
const NV_ENC_PARAMS_RC_CBR: i64 = 2;
/// `NV_ENC_TWO_PASS_QUARTER_RESOLUTION` (used at `config.cpp:1652`).
const NV_ENC_TWO_PASS_QUARTER_RESOLUTION: i64 = 1;
/// `NV_ENC_H264_ENTROPY_CODING_MODE_CABAC` (used at `config.cpp:1655`).
const NV_ENC_H264_ENTROPY_CODING_MODE_CABAC: i64 = 1;

/// `std::numeric_limits<int>::max()` (`video.cpp:1255`, `1271`, `1287`, `1317`, ...).
const INT_MAX: i64 = i32::MAX as i64;

/// `nv::profile_h264_e::high` (`video.cpp:108`).
const NV_PROFILE_H264_HIGH: i64 = 2;
/// `nv::profile_hevc_e::main` (`video.cpp:116`).
const NV_PROFILE_HEVC_MAIN: i64 = 0;
/// `nv::profile_hevc_e::main_10` (`video.cpp:117`).
const NV_PROFILE_HEVC_MAIN_10: i64 = 1;
/// `qsv::profile_h264_e::high` (`video.cpp:129`).
const QSV_PROFILE_H264_HIGH: i64 = 100;
/// `qsv::profile_h264_e::high_444p` (`video.cpp:130`).
const QSV_PROFILE_H264_HIGH_444P: i64 = 244;
/// `qsv::profile_hevc_e::main` (`video.cpp:137`).
const QSV_PROFILE_HEVC_MAIN: i64 = 1;
/// `qsv::profile_hevc_e::main_10` (`video.cpp:138`).
const QSV_PROFILE_HEVC_MAIN_10: i64 = 2;
/// `qsv::profile_hevc_e::rext` (`video.cpp:139`).
const QSV_PROFILE_HEVC_REXT: i64 = 4;
/// `qsv::profile_av1_e::main` (`video.cpp:146`).
const QSV_PROFILE_AV1_MAIN: i64 = 1;
/// `qsv::profile_av1_e::high` (`video.cpp:147`).
const QSV_PROFILE_AV1_HIGH: i64 = 2;

/// `amf::coder_e::cavlc` (`video.h:68`).
pub const AMF_CODER_CAVLC: i32 = 2;

/// Default value of a [`ConfigKey`] in Sunshine (`video_t video {...}`,
/// `config.cpp:737-821`, plus the `nv_legacy` derivation at `config.cpp:1651-1657`
/// applied to the `nvenc::nvenc_config` defaults, `nvenc_config.h:32-59`).
///
/// `None` means Sunshine leaves the option unset (empty `std::optional` /
/// empty `std::string`); `handle_option` then skips it (`video.cpp:2205-2209`,
/// `2216-2220`).
///
/// The `nv_legacy` members are value-initialised to ""/0 at `config.cpp:754`
/// and only become the derived values when `apply_config` runs
/// (`config.cpp:1651`); the derived values are what is returned, since
/// `apply_config` runs at every Sunshine start-up.
pub fn config_default(key: ConfigKey) -> Option<ConfigDefault> {
    use ConfigDefault::{Int, Str};
    match key {
        ConfigKey::NvLegacyPreset => Some(Str("p1")),
        ConfigKey::NvLegacyMultipass => Some(Int(NV_ENC_TWO_PASS_QUARTER_RESOLUTION)),
        ConfigKey::NvLegacySpatialAq => Some(Int(0)),
        ConfigKey::NvLegacyH264Coder => Some(Int(NV_ENC_H264_ENTROPY_CODING_MODE_CABAC)),
        ConfigKey::QsvPreset => Some(Int(4)),
        ConfigKey::QsvCavlc => Some(Int(0)),
        ConfigKey::AmdPreanalysis => Some(Int(0)),
        ConfigKey::AmdQualityAv1 => Some(Int(70)),
        ConfigKey::AmdQualityHevc => Some(Int(5)),
        ConfigKey::AmdQualityH264 => Some(Int(0)),
        ConfigKey::AmdRcAv1 => Some(Int(1)),
        ConfigKey::AmdRcHevc => Some(Int(1)),
        ConfigKey::AmdRcH264 => Some(Int(3)),
        ConfigKey::AmdUsageAv1 => Some(Int(2)),
        ConfigKey::AmdUsageHevc => Some(Int(1)),
        ConfigKey::AmdUsageH264 => Some(Int(1)),
        ConfigKey::AmdVbaq => Some(Int(1)),
        ConfigKey::AmdEnforceHrd => Some(Int(0)),
        ConfigKey::AmdMaxAuSize => None,
        ConfigKey::AmdCoder => Some(Int(0)),
        ConfigKey::VaapiBlbrc => Some(Int(0)),
        ConfigKey::VkTune => Some(Int(2)),
        ConfigKey::VkRcMode => Some(Int(2)),
        ConfigKey::VtAllowSw => Some(Int(0)),
        ConfigKey::VtRequireSw => Some(Int(0)),
        ConfigKey::VtRealtime => Some(Int(1)),
        ConfigKey::SwSvtav1Preset => Some(Int(11)),
        ConfigKey::SwPreset => Some(Str("superfast")),
        ConfigKey::SwTune => Some(Str("zerolatency")),
    }
}

// ---------------------------------------------------------------------------
// Session parameters (the subset of `config_t`, video.h:36-49, that is read)
// ---------------------------------------------------------------------------

/// The fields of Sunshine's `config_t` (`video.h:36-49`) read by the lambdas
/// and helpers of this module. All `int` in C++.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionParams {
    /// `config_t::width`.
    pub width: i32,
    /// `config_t::height`.
    pub height: i32,
    /// `config_t::framerate`.
    pub framerate: i32,
    /// `config_t::bitrate`, in kbps.
    pub bitrate: i32,
    /// `config_t::slicesPerFrame`.
    pub slices_per_frame: i32,
    /// `config_t::numRefFrames`.
    pub num_ref_frames: i32,
    /// `config_t::videoFormat`: 0 = H.264, 1 = HEVC, 2 = AV1.
    pub video_format: i32,
    /// `config_t::dynamicRange`: 0 = SDR, non-zero = HDR.
    pub dynamic_range: i32,
    /// `config_t::chromaSamplingType`: 0 = 4:2:0, 1 = 4:4:4.
    pub chroma_sampling_type: i32,
}

/// Port of every lambda in the tables (see [`Computed`]). Always returns `Some`
/// (the lambdas always produce a value); the `Option` mirrors [`config_default`].
///
/// `config::sunshine.min_log_level` and `config::video.qsv.qsv_slow_hevc` use
/// Sunshine's defaults, [`SUNSHINE_DEFAULT_MIN_LOG_LEVEL`] and
/// [`DEFAULT_QSV_SLOW_HEVC`].
pub fn compute(c: Computed, cfg: &SessionParams) -> Option<ConfigDefault> {
    match c {
        // video.cpp:1003-1005 / 1028-1030 / 1068-1070
        Computed::AmdLogToDbg => Some(ConfigDefault::Int(
            if SUNSHINE_DEFAULT_MIN_LOG_LEVEL < 2 { 1 } else { 0 },
        )),
        // video.cpp:1040-1050 (C++ `int` arithmetic; done in i64 here, which only
        // differs where the C++ would overflow `int`, i.e. undefined behaviour)
        Computed::AmdHevcLevel => {
            // For 4K and below, try to use level 5.1 or 5.2 if possible
            let size = cfg.width as i64 * cfg.height as i64;
            if size <= 8912896 {
                if size * cfg.framerate as i64 <= 534773760 {
                    return Some(ConfigDefault::Str("5.1"));
                } else if size * cfg.framerate as i64 <= 1069547520 {
                    return Some(ConfigDefault::Str("5.2"));
                }
            }
            Some(ConfigDefault::Str("auto"))
        }
        // video.cpp:939-941
        Computed::QsvHevcLowPower => Some(ConfigDefault::Int(if DEFAULT_QSV_SLOW_HEVC { 0 } else { 1 })),
    }
}

// ---------------------------------------------------------------------------
// Table types
// ---------------------------------------------------------------------------

/// `encoder_t::option_t` (`video.h:381-401`): an FFmpeg private option name and
/// where its value comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderOption {
    /// Option name passed to `av_dict_set*`.
    pub name: &'static str,
    /// Value source.
    pub value: OptionValue,
}

const fn int_opt(name: &'static str, v: i64) -> EncoderOption {
    EncoderOption {
        name,
        value: OptionValue::Int(v),
    }
}
const fn str_opt(name: &'static str, v: &'static str) -> EncoderOption {
    EncoderOption {
        name,
        value: OptionValue::Str(v),
    }
}
const fn cfg_opt(name: &'static str, k: ConfigKey) -> EncoderOption {
    EncoderOption {
        name,
        value: OptionValue::Config(k),
    }
}
const fn computed_opt(name: &'static str, c: Computed) -> EncoderOption {
    EncoderOption {
        name,
        value: OptionValue::Computed(c),
    }
}

/// `encoder_t::codec_t` (`video.h:408-438`) without the probe capability
/// bitset: the six option lists and the FFmpeg encoder name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodecFormat {
    /// `common_options`: applied to every mode.
    pub common: &'static [EncoderOption],
    /// `sdr_options`.
    pub sdr: &'static [EncoderOption],
    /// `hdr_options`.
    pub hdr: &'static [EncoderOption],
    /// `sdr444_options`.
    pub sdr444: &'static [EncoderOption],
    /// `hdr444_options`.
    pub hdr444: &'static [EncoderOption],
    /// `fallback_options`: applied on the retry only.
    pub fallback: &'static [EncoderOption],
    /// FFmpeg encoder name (`codec_t::name`). Empty string = `{}` in the table
    /// (no encoder for this codec).
    pub name: &'static str,
}

/// `encoder_t` (`video.h:334-465`) for an FFmpeg (`encoder_platform_formats_avcodec`)
/// encoder, with the platform condition kept as data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderDef {
    /// `encoder_t::name`.
    pub name: &'static str,
    /// Platforms Sunshine compiles this definition for.
    pub platforms: &'static [Platform],
    /// `avcodec_base_dev_type` (`video.h:292`).
    pub base_dev_type: HwDeviceType,
    /// `avcodec_derived_dev_type` (`video.h:293`).
    pub derived_dev_type: HwDeviceType,
    /// `avcodec_dev_pix_fmt` (`video.h:294`).
    pub dev_pix_fmt: PixFmt,
    /// `avcodec_pix_fmt_8bit` (`video.h:295`).
    pub pix_fmt_8bit: PixFmt,
    /// `avcodec_pix_fmt_10bit` (`video.h:296`).
    pub pix_fmt_10bit: PixFmt,
    /// `avcodec_pix_fmt_yuv444_8bit` (`video.h:297`).
    pub pix_fmt_yuv444_8bit: PixFmt,
    /// `avcodec_pix_fmt_yuv444_10bit` (`video.h:298`).
    pub pix_fmt_yuv444_10bit: PixFmt,
    /// `init_avcodec_hardware_input_buffer` (`video.h:300`).
    pub init_hw_input: HwInputInit,
    /// `encoder_t::av1`.
    pub av1: CodecFormat,
    /// `encoder_t::hevc`.
    pub hevc: CodecFormat,
    /// `encoder_t::h264`.
    pub h264: CodecFormat,
    /// `encoder_t::flags`: a combination of the [`flag`] constants.
    pub flags: u32,
}

impl EncoderDef {
    /// `hardware = avcodec_base_dev_type != AV_HWDEVICE_TYPE_NONE` (`video.cpp:2007`).
    pub fn is_hardware(&self) -> bool {
        self.base_dev_type != HwDeviceType::None
    }

    /// `encoder_t::codec_from_config` (`video.h:450-461`): 0 (and anything
    /// unknown, which Sunshine logs) = H.264, 1 = HEVC, 2 = AV1.
    pub fn codec_from_video_format(&self, video_format: i32) -> &CodecFormat {
        match video_format {
            1 => &self.hevc,
            2 => &self.av1,
            _ => &self.h264,
        }
    }
}

const NO_OPTIONS: &[EncoderOption] = &[];

// ---------------------------------------------------------------------------
// Encoder definitions (video.cpp:716-1452)
// ---------------------------------------------------------------------------

/// `nvenc` as an avcodec encoder: `video.cpp:761-863` (`#elif !defined(__APPLE__)`
/// after `#ifdef _WIN32`, so non-Windows, non-Apple).
///
/// The native Windows `nvenc` (`video.cpp:716-760`, `encoder_platform_formats_nvenc`)
/// is NOT ported. The `#ifdef _WIN32` branches inside this definition
/// (`video.cpp:765-772`, `778-782`: D3D11VA / `AV_PIX_FMT_D3D11` /
/// `dxgi_init_avcodec_hardware_input_buffer`) sit inside the `#elif` that
/// excludes Windows, so they apply to no platform and are not represented;
/// only the `#else` (CUDA) values are.
pub static NVENC: EncoderDef = EncoderDef {
    name: "nvenc", // 763
    platforms: &[Platform::Linux, Platform::FreeBsd], // 761-762, 863
    base_dev_type: HwDeviceType::Cuda, // 770
    derived_dev_type: HwDeviceType::None, // 771
    dev_pix_fmt: PixFmt::Cuda, // 772
    pix_fmt_8bit: PixFmt::Nv12, // 774
    pix_fmt_10bit: PixFmt::P010, // 775
    pix_fmt_yuv444_8bit: PixFmt::Yuv444p, // 776
    pix_fmt_yuv444_10bit: PixFmt::Yuv444p16, // 777
    init_hw_input: HwInputInit::Cuda, // 781
    // 784-805
    av1: CodecFormat {
        common: &[
            int_opt("delay", 0), // 787
            int_opt("forced-idr", 1), // 788
            int_opt("zerolatency", 1), // 789
            int_opt("surfaces", 1), // 790
            int_opt("cbr_padding", 0), // 791 (`false`)
            cfg_opt("preset", ConfigKey::NvLegacyPreset), // 792
            int_opt("tune", NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY), // 793
            int_opt("rc", NV_ENC_PARAMS_RC_CBR), // 794
            cfg_opt("multipass", ConfigKey::NvLegacyMultipass), // 795
            cfg_opt("spatial-aq", ConfigKey::NvLegacySpatialAq), // 796
        ],
        sdr: NO_OPTIONS, // 798
        hdr: NO_OPTIONS, // 799
        sdr444: NO_OPTIONS, // 800
        hdr444: NO_OPTIONS, // 801
        fallback: NO_OPTIONS, // 802
        name: "av1_nvenc", // 803
    },
    // 806-835
    hevc: CodecFormat {
        common: &[
            int_opt("delay", 0), // 809
            int_opt("forced-idr", 1), // 810
            int_opt("zerolatency", 1), // 811
            int_opt("surfaces", 1), // 812
            int_opt("cbr_padding", 0), // 813 (`false`)
            cfg_opt("preset", ConfigKey::NvLegacyPreset), // 814
            int_opt("tune", NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY), // 815
            int_opt("rc", NV_ENC_PARAMS_RC_CBR), // 816
            cfg_opt("multipass", ConfigKey::NvLegacyMultipass), // 817
            cfg_opt("spatial-aq", ConfigKey::NvLegacySpatialAq), // 818
        ],
        sdr: &[int_opt("profile", NV_PROFILE_HEVC_MAIN)], // 822
        hdr: &[int_opt("profile", NV_PROFILE_HEVC_MAIN_10)], // 826
        // 828-829: libavcodec uses the rext profile for YUV444 by itself
        sdr444: NO_OPTIONS, // 830
        hdr444: NO_OPTIONS, // 831
        fallback: NO_OPTIONS, // 832
        name: "hevc_nvenc", // 833
    },
    // 836-860
    h264: CodecFormat {
        common: &[
            int_opt("delay", 0), // 838
            int_opt("forced-idr", 1), // 839
            int_opt("zerolatency", 1), // 840
            int_opt("surfaces", 1), // 841
            int_opt("cbr_padding", 0), // 842 (`false`)
            cfg_opt("preset", ConfigKey::NvLegacyPreset), // 843
            int_opt("tune", NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY), // 844
            int_opt("rc", NV_ENC_PARAMS_RC_CBR), // 845
            cfg_opt("coder", ConfigKey::NvLegacyH264Coder), // 846
            cfg_opt("multipass", ConfigKey::NvLegacyMultipass), // 847
            cfg_opt("spatial-aq", ConfigKey::NvLegacySpatialAq), // 848
        ],
        sdr: &[int_opt("profile", NV_PROFILE_H264_HIGH)], // 852
        hdr: NO_OPTIONS, // 854
        sdr444: NO_OPTIONS, // 855
        hdr444: NO_OPTIONS, // 856
        fallback: NO_OPTIONS, // 857
        name: "h264_nvenc", // 858
    },
    flags: flag::PARALLEL_ENCODING | flag::YUV444_SUPPORT, // 861
};

/// `quicksync`: `video.cpp:865-978` (`#ifdef _WIN32`).
pub static QUICKSYNC: EncoderDef = EncoderDef {
    name: "quicksync", // 870
    platforms: &[Platform::Windows], // 865, 1156
    base_dev_type: HwDeviceType::D3d11va, // 872
    derived_dev_type: HwDeviceType::Qsv, // 873
    dev_pix_fmt: PixFmt::Qsv, // 874
    pix_fmt_8bit: PixFmt::Nv12, // 875
    pix_fmt_10bit: PixFmt::P010, // 876
    pix_fmt_yuv444_8bit: PixFmt::Vuyx, // 877
    pix_fmt_yuv444_10bit: PixFmt::Xv30, // 878
    init_hw_input: HwInputInit::Dxgi, // 879
    // 881-909
    av1: CodecFormat {
        common: &[
            cfg_opt("preset", ConfigKey::QsvPreset), // 884
            int_opt("forced_idr", 1), // 885
            int_opt("async_depth", 1), // 886
            int_opt("low_delay_brc", 1), // 887
            int_opt("low_power", 1), // 888
        ],
        sdr: &[int_opt("profile", QSV_PROFILE_AV1_MAIN)], // 892
        hdr: &[int_opt("profile", QSV_PROFILE_AV1_MAIN)], // 896
        sdr444: &[int_opt("profile", QSV_PROFILE_AV1_HIGH)], // 900
        hdr444: &[int_opt("profile", QSV_PROFILE_AV1_HIGH)], // 904
        fallback: NO_OPTIONS, // 906
        name: "av1_qsv", // 907
    },
    // 910-945
    hevc: CodecFormat {
        common: &[
            cfg_opt("preset", ConfigKey::QsvPreset), // 913
            int_opt("forced_idr", 1), // 914
            int_opt("async_depth", 1), // 915
            int_opt("low_delay_brc", 1), // 916
            int_opt("low_power", 1), // 917
            int_opt("recovery_point_sei", 0), // 918
            int_opt("pic_timing_sei", 0), // 919
        ],
        sdr: &[int_opt("profile", QSV_PROFILE_HEVC_MAIN)], // 923
        hdr: &[int_opt("profile", QSV_PROFILE_HEVC_MAIN_10)], // 927
        sdr444: &[int_opt("profile", QSV_PROFILE_HEVC_REXT)], // 931
        hdr444: &[int_opt("profile", QSV_PROFILE_HEVC_REXT)], // 935
        fallback: &[computed_opt("low_power", Computed::QsvHevcLowPower)], // 939-941
        name: "hevc_qsv", // 943
    },
    // 946-976
    h264: CodecFormat {
        common: &[
            cfg_opt("preset", ConfigKey::QsvPreset), // 949
            cfg_opt("cavlc", ConfigKey::QsvCavlc), // 950
            int_opt("forced_idr", 1), // 951
            int_opt("async_depth", 1), // 952
            int_opt("low_delay_brc", 1), // 953
            int_opt("low_power", 1), // 954
            int_opt("recovery_point_sei", 0), // 955
            int_opt("vcm", 1), // 956
            int_opt("pic_timing_sei", 0), // 957
            int_opt("max_dec_frame_buffering", 1), // 958
        ],
        sdr: &[int_opt("profile", QSV_PROFILE_H264_HIGH)], // 962
        hdr: NO_OPTIONS, // 964
        sdr444: &[int_opt("profile", QSV_PROFILE_H264_HIGH_444P)], // 967
        hdr444: NO_OPTIONS, // 969
        // 972: some old/low-end Intel GPUs don't support low power encoding
        fallback: &[int_opt("low_power", 0)],
        name: "h264_qsv", // 974
    },
    flags: flag::PARALLEL_ENCODING
        | flag::CBR_WITH_VBR
        | flag::RELAXED_COMPLIANCE
        | flag::NO_RC_BUF_LIMIT
        | flag::YUV444_SUPPORT, // 977
};

/// `amdvce`: `video.cpp:980-1092` (`#ifdef _WIN32`).
pub static AMDVCE: EncoderDef = EncoderDef {
    name: "amdvce", // 984
    platforms: &[Platform::Windows], // 865, 1156
    base_dev_type: HwDeviceType::D3d11va, // 986
    derived_dev_type: HwDeviceType::None, // 987
    dev_pix_fmt: PixFmt::D3d11, // 988
    pix_fmt_8bit: PixFmt::Nv12, // 989
    pix_fmt_10bit: PixFmt::P010, // 990
    pix_fmt_yuv444_8bit: PixFmt::None, // 991
    pix_fmt_yuv444_10bit: PixFmt::None, // 992
    init_hw_input: HwInputInit::Dxgi, // 993
    // 995-1019
    av1: CodecFormat {
        common: &[
            int_opt("filler_data", 0), // 998 (`false`)
            int_opt("forced_idr", 1), // 999
            str_opt("latency", "lowest_latency"), // 1000
            int_opt("async_depth", 1), // 1001
            int_opt("skip_frame", 0), // 1002
            computed_opt("log_to_dbg", Computed::AmdLogToDbg), // 1003-1005
            cfg_opt("preencode", ConfigKey::AmdPreanalysis), // 1006
            cfg_opt("quality", ConfigKey::AmdQualityAv1), // 1007
            cfg_opt("rc", ConfigKey::AmdRcAv1), // 1008
            cfg_opt("usage", ConfigKey::AmdUsageAv1), // 1009
            cfg_opt("enforce_hrd", ConfigKey::AmdEnforceHrd), // 1010
        ],
        sdr: NO_OPTIONS, // 1012
        hdr: NO_OPTIONS, // 1013
        sdr444: NO_OPTIONS, // 1014
        hdr444: NO_OPTIONS, // 1015
        fallback: NO_OPTIONS, // 1016
        name: "av1_amf", // 1017
    },
    // 1020-1059
    hevc: CodecFormat {
        common: &[
            int_opt("filler_data", 0), // 1023 (`false`)
            int_opt("forced_idr", 1), // 1024
            int_opt("latency", 1), // 1025
            int_opt("async_depth", 1), // 1026
            int_opt("skip_frame", 0), // 1027
            computed_opt("log_to_dbg", Computed::AmdLogToDbg), // 1028-1030
            int_opt("gops_per_idr", 1), // 1031
            str_opt("header_insertion_mode", "idr"), // 1032
            cfg_opt("preencode", ConfigKey::AmdPreanalysis), // 1033
            cfg_opt("quality", ConfigKey::AmdQualityHevc), // 1034
            cfg_opt("rc", ConfigKey::AmdRcHevc), // 1035
            cfg_opt("usage", ConfigKey::AmdUsageHevc), // 1036
            cfg_opt("vbaq", ConfigKey::AmdVbaq), // 1037
            cfg_opt("enforce_hrd", ConfigKey::AmdEnforceHrd), // 1038
            cfg_opt("max_au_size", ConfigKey::AmdMaxAuSize), // 1039
            computed_opt("level", Computed::AmdHevcLevel), // 1040-1050
        ],
        sdr: NO_OPTIONS, // 1052
        hdr: NO_OPTIONS, // 1053
        sdr444: NO_OPTIONS, // 1054
        hdr444: NO_OPTIONS, // 1055
        fallback: NO_OPTIONS, // 1056
        name: "hevc_amf", // 1057
    },
    // 1060-1090
    h264: CodecFormat {
        common: &[
            int_opt("filler_data", 0), // 1063 (`false`)
            int_opt("forced_idr", 1), // 1064
            int_opt("latency", 1), // 1065
            int_opt("async_depth", 1), // 1066
            int_opt("frame_skipping", 0), // 1067
            computed_opt("log_to_dbg", Computed::AmdLogToDbg), // 1068-1070
            cfg_opt("preencode", ConfigKey::AmdPreanalysis), // 1071
            cfg_opt("quality", ConfigKey::AmdQualityH264), // 1072
            cfg_opt("rc", ConfigKey::AmdRcH264), // 1073
            cfg_opt("usage", ConfigKey::AmdUsageH264), // 1074
            cfg_opt("vbaq", ConfigKey::AmdVbaq), // 1075
            cfg_opt("coder", ConfigKey::AmdCoder), // 1076
            cfg_opt("enforce_hrd", ConfigKey::AmdEnforceHrd), // 1077
            cfg_opt("max_au_size", ConfigKey::AmdMaxAuSize), // 1078
        ],
        sdr: NO_OPTIONS, // 1080
        hdr: NO_OPTIONS, // 1081
        sdr444: NO_OPTIONS, // 1082
        hdr444: NO_OPTIONS, // 1083
        // 1086: AMF_VIDEO_ENCODER_USAGE_LOW_LATENCY; workaround for
        // https://github.com/GPUOpen-LibrariesAndSDKs/AMF/issues/410
        fallback: &[int_opt("usage", 2)],
        name: "h264_amf", // 1088
    },
    flags: flag::PARALLEL_ENCODING, // 1091
};

/// `mediafoundation`: `video.cpp:1094-1155` (`#ifdef _WIN32`).
pub static MEDIAFOUNDATION: EncoderDef = EncoderDef {
    name: "mediafoundation", // 1098
    platforms: &[Platform::Windows], // 865, 1156
    base_dev_type: HwDeviceType::D3d11va, // 1100
    derived_dev_type: HwDeviceType::None, // 1101
    dev_pix_fmt: PixFmt::D3d11, // 1102
    pix_fmt_8bit: PixFmt::Nv12, // 1103 (SDR 4:2:0 8-bit, only format Qualcomm supports)
    pix_fmt_10bit: PixFmt::None, // 1104 (no HDR)
    pix_fmt_yuv444_8bit: PixFmt::None, // 1105 (no YUV444 SDR)
    pix_fmt_yuv444_10bit: PixFmt::None, // 1106 (no YUV444 HDR)
    init_hw_input: HwInputInit::Dxgi, // 1107
    // 1109-1123 (Qualcomm MF encoder)
    av1: CodecFormat {
        common: &[
            int_opt("hw_encoding", 1), // 1112
            str_opt("rate_control", "cbr"), // 1113
            str_opt("scenario", "display_remoting"), // 1114
        ],
        sdr: NO_OPTIONS, // 1116
        hdr: NO_OPTIONS, // 1117
        sdr444: NO_OPTIONS, // 1118
        hdr444: NO_OPTIONS, // 1119
        fallback: NO_OPTIONS, // 1120
        name: "av1_mf", // 1121
    },
    // 1124-1138
    hevc: CodecFormat {
        common: &[
            int_opt("hw_encoding", 1), // 1127
            str_opt("rate_control", "cbr"), // 1128
            str_opt("scenario", "display_remoting"), // 1129
        ],
        sdr: NO_OPTIONS, // 1131
        hdr: NO_OPTIONS, // 1132
        sdr444: NO_OPTIONS, // 1133
        hdr444: NO_OPTIONS, // 1134
        fallback: NO_OPTIONS, // 1135
        name: "hevc_mf", // 1136
    },
    // 1139-1153
    h264: CodecFormat {
        common: &[
            int_opt("hw_encoding", 1), // 1142
            str_opt("rate_control", "cbr"), // 1143
            str_opt("scenario", "display_remoting"), // 1144
        ],
        sdr: NO_OPTIONS, // 1146
        hdr: NO_OPTIONS, // 1147
        sdr444: NO_OPTIONS, // 1148
        hdr444: NO_OPTIONS, // 1149
        fallback: NO_OPTIONS, // 1150
        name: "h264_mf", // 1151
    },
    // 1154: MF encoder doesn't support on-demand IDR frames
    flags: flag::PARALLEL_ENCODING | flag::FIXED_GOP_SIZE,
};

/// FFmpeg encoder name of the software AV1 codec when Sunshine is built with
/// `ENABLE_BROKEN_AV1_ENCODER` (`video.cpp:1188-1192`). That macro is not
/// defined anywhere in the Sunshine build files (grep of `cmake/` and
/// `CMakeLists.txt`), so the default table has an empty name (`video.cpp:1194`).
pub const SOFTWARE_AV1_BROKEN_ENCODER_NAME: &str = "libsvtav1";

/// `software`: `video.cpp:1158-1232` (no platform condition).
pub static SOFTWARE: EncoderDef = EncoderDef {
    name: "software", // 1162
    platforms: ALL_PLATFORMS, // 1161 (unconditional)
    base_dev_type: HwDeviceType::None, // 1164
    derived_dev_type: HwDeviceType::None, // 1165
    dev_pix_fmt: PixFmt::None, // 1166
    pix_fmt_8bit: PixFmt::Yuv420p, // 1167
    pix_fmt_10bit: PixFmt::Yuv420p10, // 1168
    pix_fmt_yuv444_8bit: PixFmt::Yuv444p, // 1169
    pix_fmt_yuv444_10bit: PixFmt::Yuv444p10, // 1170
    init_hw_input: HwInputInit::None, // 1171 (`nullptr`)
    // 1173-1197. libsvtav1 takes different presets than libx264/libx265.
    // We set an infinite GOP length, use a low delay prediction structure,
    // force I frames to be key frames, and set max bitrate to default to work
    // around a FFmpeg bug with CBR mode.
    av1: CodecFormat {
        common: &[
            str_opt("svtav1-params", "keyint=-1:pred-struct=1:force-key-frames=1:mbr=0"), // 1179
            cfg_opt("preset", ConfigKey::SwSvtav1Preset), // 1180
        ],
        sdr: NO_OPTIONS, // 1182
        hdr: NO_OPTIONS, // 1183
        sdr444: NO_OPTIONS, // 1184
        hdr444: NO_OPTIONS, // 1185
        fallback: NO_OPTIONS, // 1186
        name: "", // 1194 (`{}`; `"libsvtav1"` only with ENABLE_BROKEN_AV1_ENCODER, 1188-1192)
    },
    // 1198-1216. x265's Info SEI is so long that it causes the IDR picture data
    // to be kicked to the 2nd packet in the frame, breaking Moonlight's parsing
    // logic. It also looks like gop_size isn't passed on to x265, so we have to
    // set 'keyint=-1' in the parameters ourselves.
    hevc: CodecFormat {
        common: &[
            int_opt("forced-idr", 1), // 1204
            str_opt("x265-params", "info=0:keyint=-1"), // 1205
            cfg_opt("preset", ConfigKey::SwPreset), // 1206
            cfg_opt("tune", ConfigKey::SwTune), // 1207
        ],
        sdr: NO_OPTIONS, // 1209
        hdr: NO_OPTIONS, // 1210
        sdr444: NO_OPTIONS, // 1211
        hdr444: NO_OPTIONS, // 1212
        fallback: NO_OPTIONS, // 1213
        name: "libx265", // 1214
    },
    // 1217-1230
    h264: CodecFormat {
        common: &[
            cfg_opt("preset", ConfigKey::SwPreset), // 1220
            cfg_opt("tune", ConfigKey::SwTune), // 1221
        ],
        sdr: NO_OPTIONS, // 1223
        hdr: NO_OPTIONS, // 1224
        sdr444: NO_OPTIONS, // 1225
        hdr444: NO_OPTIONS, // 1226
        fallback: NO_OPTIONS, // 1227
        name: "libx264", // 1228
    },
    flags: flag::H264_ONLY | flag::PARALLEL_ENCODING | flag::ALWAYS_REPROBE | flag::YUV444_SUPPORT, // 1231
};

/// `vaapi`: `video.cpp:1234-1299` (`__linux__ || __FreeBSD__`).
pub static VAAPI: EncoderDef = EncoderDef {
    name: "vaapi", // 1239
    platforms: &[Platform::Linux, Platform::FreeBsd], // 1234
    base_dev_type: HwDeviceType::Vaapi, // 1241
    derived_dev_type: HwDeviceType::None, // 1242
    dev_pix_fmt: PixFmt::Vaapi, // 1243
    pix_fmt_8bit: PixFmt::Nv12, // 1244
    pix_fmt_10bit: PixFmt::P010, // 1245
    pix_fmt_yuv444_8bit: PixFmt::None, // 1246
    pix_fmt_yuv444_10bit: PixFmt::None, // 1247
    init_hw_input: HwInputInit::Vaapi, // 1248
    // 1250-1264
    av1: CodecFormat {
        common: &[
            int_opt("async_depth", 1), // 1253
            cfg_opt("blbrc", ConfigKey::VaapiBlbrc), // 1254
            int_opt("idr_interval", INT_MAX), // 1255
        ],
        sdr: NO_OPTIONS, // 1257
        hdr: NO_OPTIONS, // 1258
        sdr444: NO_OPTIONS, // 1259
        hdr444: NO_OPTIONS, // 1260
        fallback: NO_OPTIONS, // 1261
        name: "av1_vaapi", // 1262
    },
    // 1265-1280
    hevc: CodecFormat {
        common: &[
            int_opt("async_depth", 1), // 1268
            cfg_opt("blbrc", ConfigKey::VaapiBlbrc), // 1269
            int_opt("sei", 0), // 1270
            int_opt("idr_interval", INT_MAX), // 1271
        ],
        sdr: NO_OPTIONS, // 1273
        hdr: NO_OPTIONS, // 1274
        sdr444: NO_OPTIONS, // 1275
        hdr444: NO_OPTIONS, // 1276
        fallback: NO_OPTIONS, // 1277
        name: "hevc_vaapi", // 1278
    },
    // 1281-1296
    h264: CodecFormat {
        common: &[
            int_opt("async_depth", 1), // 1284
            cfg_opt("blbrc", ConfigKey::VaapiBlbrc), // 1285
            int_opt("sei", 0), // 1286
            int_opt("idr_interval", INT_MAX), // 1287
        ],
        sdr: NO_OPTIONS, // 1289
        hdr: NO_OPTIONS, // 1290
        sdr444: NO_OPTIONS, // 1291
        hdr444: NO_OPTIONS, // 1292
        fallback: NO_OPTIONS, // 1293
        name: "h264_vaapi", // 1294
    },
    // 1297: RC buffer size will be set in platform code if supported
    flags: flag::LIMITED_GOP_SIZE | flag::PARALLEL_ENCODING | flag::NO_RC_BUF_LIMIT, // 1298
};

/// `vulkan`: `video.cpp:1301-1372` (`__linux__ || __FreeBSD__` and
/// `SUNSHINE_BUILD_VULKAN`, which `cmake/compile_definitions/linux.cmake:165`
/// defines for Linux builds).
///
/// not used by Lunaris yet
pub static VULKAN: EncoderDef = EncoderDef {
    name: "vulkan", // 1303
    platforms: &[Platform::Linux, Platform::FreeBsd], // 1234, 1301
    base_dev_type: HwDeviceType::Vulkan, // 1305
    derived_dev_type: HwDeviceType::None, // 1306
    dev_pix_fmt: PixFmt::Vulkan, // 1307
    pix_fmt_8bit: PixFmt::Nv12, // 1308
    pix_fmt_10bit: PixFmt::P010, // 1309
    pix_fmt_yuv444_8bit: PixFmt::None, // 1310
    pix_fmt_yuv444_10bit: PixFmt::None, // 1311
    init_hw_input: HwInputInit::Vulkan, // 1312
    // 1314-1332 (AV1)
    av1: CodecFormat {
        common: &[
            int_opt("idr_interval", INT_MAX), // 1317
            cfg_opt("tune", ConfigKey::VkTune), // 1318
            cfg_opt("rc_mode", ConfigKey::VkRcMode), // 1319
            int_opt("units", 0), // 1320
            str_opt("usage", "stream"), // 1321
            str_opt("content", "rendered"), // 1322
            int_opt("async_depth", 1), // 1323
        ],
        sdr: NO_OPTIONS, // 1325
        hdr: NO_OPTIONS, // 1326
        sdr444: NO_OPTIONS, // 1327
        hdr444: NO_OPTIONS, // 1328
        fallback: NO_OPTIONS, // 1329
        name: "av1_vulkan", // 1330
    },
    // 1333-1351 (HEVC)
    hevc: CodecFormat {
        common: &[
            int_opt("idr_interval", INT_MAX), // 1336
            cfg_opt("tune", ConfigKey::VkTune), // 1337
            cfg_opt("rc_mode", ConfigKey::VkRcMode), // 1338
            int_opt("units", 0), // 1339
            str_opt("usage", "stream"), // 1340
            str_opt("content", "rendered"), // 1341
            int_opt("async_depth", 1), // 1342
        ],
        sdr: NO_OPTIONS, // 1344
        hdr: NO_OPTIONS, // 1345
        sdr444: NO_OPTIONS, // 1346
        hdr444: NO_OPTIONS, // 1347
        fallback: NO_OPTIONS, // 1348
        name: "hevc_vulkan", // 1349
    },
    // 1352-1370 (H.264)
    h264: CodecFormat {
        common: &[
            int_opt("idr_interval", INT_MAX), // 1355
            cfg_opt("tune", ConfigKey::VkTune), // 1356
            cfg_opt("rc_mode", ConfigKey::VkRcMode), // 1357
            int_opt("units", 0), // 1358
            str_opt("usage", "stream"), // 1359
            str_opt("content", "rendered"), // 1360
            int_opt("async_depth", 1), // 1361
        ],
        sdr: NO_OPTIONS, // 1363
        hdr: NO_OPTIONS, // 1364
        sdr444: NO_OPTIONS, // 1365
        hdr444: NO_OPTIONS, // 1366
        fallback: NO_OPTIONS, // 1367
        name: "h264_vulkan", // 1368
    },
    flags: flag::LIMITED_GOP_SIZE | flag::PARALLEL_ENCODING, // 1371
};

/// `videotoolbox`: `video.cpp:1376-1451` (`#ifdef __APPLE__`).
pub static VIDEOTOOLBOX: EncoderDef = EncoderDef {
    name: "videotoolbox", // 1381
    platforms: &[Platform::MacOs], // 1376
    base_dev_type: HwDeviceType::VideoToolbox, // 1383
    derived_dev_type: HwDeviceType::None, // 1384
    dev_pix_fmt: PixFmt::VideoToolbox, // 1385
    pix_fmt_8bit: PixFmt::Nv12, // 1386
    pix_fmt_10bit: PixFmt::P010, // 1387
    pix_fmt_yuv444_8bit: PixFmt::None, // 1388
    pix_fmt_yuv444_10bit: PixFmt::None, // 1389
    init_hw_input: HwInputInit::VideoToolbox, // 1390
    // 1392-1408
    av1: CodecFormat {
        common: &[
            cfg_opt("allow_sw", ConfigKey::VtAllowSw), // 1395
            cfg_opt("require_sw", ConfigKey::VtRequireSw), // 1396
            cfg_opt("realtime", ConfigKey::VtRealtime), // 1397
            int_opt("prio_speed", 1), // 1398
            int_opt("max_ref_frames", 1), // 1399
        ],
        sdr: NO_OPTIONS, // 1401
        hdr: NO_OPTIONS, // 1402
        sdr444: NO_OPTIONS, // 1403
        hdr444: NO_OPTIONS, // 1404
        fallback: NO_OPTIONS, // 1405
        name: "av1_videotoolbox", // 1406
    },
    // 1409-1425
    hevc: CodecFormat {
        common: &[
            cfg_opt("allow_sw", ConfigKey::VtAllowSw), // 1412
            cfg_opt("require_sw", ConfigKey::VtRequireSw), // 1413
            cfg_opt("realtime", ConfigKey::VtRealtime), // 1414
            int_opt("prio_speed", 1), // 1415
            int_opt("max_ref_frames", 1), // 1416
        ],
        sdr: NO_OPTIONS, // 1418
        hdr: NO_OPTIONS, // 1419
        sdr444: NO_OPTIONS, // 1420
        hdr444: NO_OPTIONS, // 1421
        fallback: NO_OPTIONS, // 1422
        name: "hevc_videotoolbox", // 1423
    },
    // 1426-1449. max_ref_frames is intentionally omitted for H.264 because
    // VideoToolbox on Apple Silicon produces all-IDR output when
    // ReferenceBufferCount=1 is set for H.264, causing massive bandwidth
    // inflation (~3x) and frame drops (1428-1432, LizardByte/Sunshine#5013).
    h264: CodecFormat {
        common: &[
            cfg_opt("allow_sw", ConfigKey::VtAllowSw), // 1434
            cfg_opt("require_sw", ConfigKey::VtRequireSw), // 1435
            cfg_opt("realtime", ConfigKey::VtRealtime), // 1436
            int_opt("prio_speed", 1), // 1437
        ],
        sdr: NO_OPTIONS, // 1439
        hdr: NO_OPTIONS, // 1440
        sdr444: NO_OPTIONS, // 1441
        hdr444: NO_OPTIONS, // 1442
        fallback: &[str_opt("flags", "-low_delay")], // 1445
        name: "h264_videotoolbox", // 1447
    },
    flags: flag::PARALLEL_ENCODING, // 1450
};

/// Sunshine's `encoders` list (`video.cpp:1454-1473`), in order, as definitions
/// (the platform of each is on [`EncoderDef::platforms`]). The native Windows
/// `nvenc` (which Sunshine lists first on Windows, `video.cpp:1456`) is not
/// ported; the [`NVENC`] entry here is the non-Windows, avcodec one.
pub static ENCODERS: &[&EncoderDef] = &[
    &NVENC,           // 1456 (`#ifndef __APPLE__`)
    &QUICKSYNC,       // 1459 (`_WIN32`)
    &AMDVCE,          // 1460
    &MEDIAFOUNDATION, // 1461
    &VULKAN,          // 1465 (`SUNSHINE_BUILD_VULKAN`, linux)
    &VAAPI,           // 1467 (linux)
    &VIDEOTOOLBOX,    // 1470 (`__APPLE__`)
    &SOFTWARE,        // 1472
];

/// The entries of [`ENCODERS`] compiled for `platform`, in Sunshine's order.
/// For [`Platform::Windows`] the native `nvenc` is missing (not ported) and
/// [`NVENC`] is not returned (its platforms are Linux/FreeBSD only).
pub fn encoders_for(platform: Platform) -> Vec<&'static EncoderDef> {
    ENCODERS
        .iter()
        .copied()
        .filter(|e| e.platforms.contains(&platform))
        .collect()
}

// ---------------------------------------------------------------------------
// handle_option order of application (video.cpp:2195-2245)
// ---------------------------------------------------------------------------

fn resolve(value: &OptionValue, params: &SessionParams) -> Option<ConfigDefault> {
    match *value {
        OptionValue::Int(v) => Some(ConfigDefault::Int(v)),
        OptionValue::Str(s) => Some(ConfigDefault::Str(s)),
        OptionValue::Config(k) => config_default(k),
        OptionValue::Computed(c) => compute(c, params),
    }
}

/// The option list `make_avcodec_encode_session` hands to FFmpeg for one
/// attempt (`video.cpp:2229-2245`), in order:
///
/// 1. `common` (2230-2232);
/// 2. `hdr` if `dynamicRange` else `sdr` (2233-2235);
/// 3. only if `chromaSamplingType == 1`: `hdr444` if `dynamicRange` else
///    `sdr444` (2236-2240);
/// 4. only if `retry` (`retries > 0`): `fallback` (2241-2245).
///
/// `Config` values resolve through [`config_default`] and `Computed` through
/// [`compute`]; unset ones are skipped (`handle_option`, `video.cpp:2205-2220`).
/// The result keeps duplicates in application order: `av_dict_set*` with flags
/// 0 replaces an existing key, so a later entry (e.g. a fallback `low_power`)
/// overrides an earlier one.
pub fn options_for(
    format: &CodecFormat,
    params: &SessionParams,
    retry: bool,
) -> Vec<(&'static str, ConfigDefault)> {
    let mut out = Vec::new();
    let mut apply = |list: &[EncoderOption]| {
        for o in list {
            if let Some(v) = resolve(&o.value, params) {
                out.push((o.name, v));
            }
        }
    };
    apply(format.common);
    apply(if params.dynamic_range != 0 { format.hdr } else { format.sdr });
    if params.chroma_sampling_type == 1 {
        apply(if params.dynamic_range != 0 { format.hdr444 } else { format.sdr444 });
    }
    if retry {
        apply(format.fallback);
    }
    out
}

// ---------------------------------------------------------------------------
// Profiles (video.cpp:152-162, 2061-2082)
// ---------------------------------------------------------------------------

/// Name of the FFmpeg `AV_PROFILE_*` constant Sunshine assigns to `ctx->profile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// `AV_PROFILE_H264_HIGH_444_PREDICTIVE` (`video.cpp:154`).
    H264High444Predictive,
    /// `AV_PROFILE_H264_CONSTRAINED_BASELINE` (`video.cpp:158`).
    H264ConstrainedBaseline,
    /// `AV_PROFILE_H264_HIGH` (`video.cpp:161`).
    H264High,
    /// `AV_PROFILE_HEVC_REXT` (`video.cpp:2071`).
    HevcRext,
    /// `AV_PROFILE_HEVC_MAIN_10` (`video.cpp:2073`).
    HevcMain10,
    /// `AV_PROFILE_HEVC_MAIN` (`video.cpp:2073`).
    HevcMain,
    /// `AV_PROFILE_AV1_HIGH` (`video.cpp:2080`).
    Av1High,
    /// `AV_PROFILE_AV1_MAIN` (`video.cpp:2080`).
    Av1Main,
    /// `config.videoFormat` matched no `case` of the switch at `video.cpp:2061`:
    /// `ctx->profile` is left at the `avcodec_alloc_context3` default.
    Unset,
}

/// Port of `select_h264_profile` (`video.cpp:152-162`).
pub fn select_h264_profile(encoder_name: &str, chroma_sampling_type: i32, amd_coder: i32) -> Profile {
    if chroma_sampling_type == 1 {
        return Profile::H264High444Predictive;
    }
    if encoder_name == "h264_amf" && amd_coder == AMF_CODER_CAVLC {
        return Profile::H264ConstrainedBaseline;
    }
    Profile::H264High
}

/// The `ctx->profile` switch of `make_avcodec_encode_session` (`video.cpp:2061-2082`).
///
/// `encoder_name` is the FFmpeg codec name (`video_format.name`) and
/// `amd_coder` is `config::video.amd.amd_coder` (default
/// [`config_default`]`(AmdCoder)` = 0), both used only by case 0.
pub fn profile(
    video_format: i32,
    dynamic_range: i32,
    chroma_sampling_type: i32,
    encoder_name: &str,
    amd_coder: i32,
) -> Profile {
    match video_format {
        // 10-bit h264 encoding is not supported by our streaming protocol
        // (`assert(!config.dynamicRange)`, video.cpp:2064)
        0 => select_h264_profile(encoder_name, chroma_sampling_type, amd_coder),
        1 => {
            if chroma_sampling_type == 1 {
                // HEVC uses the same RExt profile for both 8 and 10 bit YUV 4:4:4 encoding
                Profile::HevcRext
            } else if dynamic_range != 0 {
                Profile::HevcMain10
            } else {
                Profile::HevcMain
            }
        }
        // AV1 supports both 8 and 10 bit encoding with the same Main profile
        // but YUV 4:4:4 sampling requires High profile
        2 => {
            if chroma_sampling_type == 1 {
                Profile::Av1High
            } else {
                Profile::Av1Main
            }
        }
        _ => Profile::Unset,
    }
}

// ---------------------------------------------------------------------------
// Pixel format selection (video.cpp:2041-2045)
// ---------------------------------------------------------------------------

/// `sw_fmt` of `make_avcodec_encode_session` (`video.cpp:2041-2045`), from the
/// colourspace bit depth and `chromaSamplingType`.
pub fn sw_pix_fmt(def: &EncoderDef, bit_depth: i32, chroma_sampling_type: i32) -> PixFmt {
    match (bit_depth, chroma_sampling_type) {
        (8, 0) => def.pix_fmt_8bit,
        (8, 1) => def.pix_fmt_yuv444_8bit,
        (10, 0) => def.pix_fmt_10bit,
        (10, 1) => def.pix_fmt_yuv444_10bit,
        _ => PixFmt::None,
    }
}

// ---------------------------------------------------------------------------
// GOP (video.cpp:2084-2098)
// ---------------------------------------------------------------------------

/// `ctx->max_b_frames` (`video.cpp:2085`): B-frames delay decoder output, so never use them.
pub const MAX_B_FRAMES: i32 = 0;

/// `ctx->gop_size` and `ctx->keyint_min` (`video.cpp:2087-2098`).
///
/// `FIXED_GOP_SIZE`: `(120, 120)` (2089-2092). Otherwise `gop_size` is
/// `i16::MAX` with `LIMITED_GOP_SIZE` and `i32::MAX` without, and
/// `keyint_min` is `i32::MAX` (2094-2097).
pub fn gop(flags: u32) -> (i32, i32) {
    if has(flags, flag::FIXED_GOP_SIZE) {
        // ~2 seconds at 60 FPS - larger to reduce oversized IDR frame frequency
        (120, 120)
    } else {
        let gop_size = if has(flags, flag::LIMITED_GOP_SIZE) {
            i16::MAX as i32
        } else {
            i32::MAX
        };
        (gop_size, i32::MAX)
    }
}

/// `ctx->refs` (`video.cpp:2100-2107`): `Some(numRefFrames)` when the client
/// asked for a limit and the codec has the `REF_FRAMES_RESTRICT` capability
/// ([`codec_flag::REF_FRAMES_RESTRICT`], set by the runtime probe); `None`
/// otherwise (Sunshine logs a warning when the client asked but the encoder
/// cannot).
pub fn refs(num_ref_frames: i32, ref_frames_restrict: bool) -> Option<i32> {
    if num_ref_frames != 0 && ref_frames_restrict {
        Some(num_ref_frames)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Slices and threads (video.cpp:2178-2193)
// ---------------------------------------------------------------------------

/// `ctx->slices`, `ctx->thread_type` and `ctx->thread_count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slices {
    /// `ctx->slices`.
    pub slices: i32,
    /// `ctx->thread_count` (= `slices`, `video.cpp:2193`).
    pub thread_count: i32,
    /// `ctx->thread_type = FF_THREAD_SLICE` (`video.cpp:2192`); always true.
    pub thread_type_slice: bool,
}

/// Port of `video.cpp:2178-2193`: hardware uses `slicesPerFrame` (2178);
/// software uses `max(slicesPerFrame, min_threads)` (2185); `SINGLE_SLICE_ONLY`
/// forces 1 (2188-2190). `min_threads` is `config::video.min_threads`
/// ([`DEFAULT_MIN_THREADS`] by default).
pub fn slices(flags: u32, hardware: bool, slices_per_frame: i32, min_threads: i32) -> Slices {
    let mut slices = if hardware {
        slices_per_frame
    } else {
        // Clients will request for the fewest slices per frame to get the most
        // efficient encode, but we may want to provide more slices than
        // requested to ensure we have enough parallelism for good performance.
        slices_per_frame.max(min_threads)
    };
    if has(flags, flag::SINGLE_SLICE_ONLY) {
        slices = 1;
    }
    Slices {
        slices,
        thread_count: slices,
        thread_type_slice: true,
    }
}

// ---------------------------------------------------------------------------
// Bitrate and VBV (video.cpp:2247-2279)
// ---------------------------------------------------------------------------

/// Rate-control fields of the codec context set at `video.cpp:2247-2279`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateControl {
    /// `bitrate`, in bit/s (`video.cpp:2247`).
    pub bitrate: i64,
    /// `ctx->rc_max_rate` (2249).
    pub rc_max_rate: i64,
    /// `ctx->bit_rate` (2250; decremented with `CBR_WITH_VBR`, 2254).
    pub bit_rate: i64,
    /// `ctx->rc_min_rate`: `Some(bitrate)` (2256) unless `CBR_WITH_VBR`, in
    /// which case it is not set (`None`, left at the context default).
    pub rc_min_rate: Option<i64>,
    /// `ctx->strict_std_compliance = FF_COMPLIANCE_UNOFFICIAL` (2259-2261),
    /// from `RELAXED_COMPLIANCE`.
    pub compliance_unofficial: bool,
    /// `ctx->rc_buffer_size` (2263-2279): `None` with `NO_RC_BUF_LIMIT` (not set).
    pub rc_buffer_size: Option<i64>,
}

/// Port of `video.cpp:2247-2279`.
///
/// * `encoder_name`: `encoder.name` (the `"nvenc"` rule, 2274).
/// * `hardware`: [`EncoderDef::is_hardware`].
/// * `slices`: the final `ctx->slices` from [`slices`].
/// * `max_bitrate`: `config::video.max_bitrate` ([`DEFAULT_MAX_BITRATE`]).
/// * `nv_vbv_percentage_increase`: `config::video.nv_legacy.vbv_percentage_increase`
///   ([`DEFAULT_NV_LEGACY_VBV_PERCENTAGE_INCREASE`]); the `#ifndef __APPLE__`
///   guard (2273) is moot because `nvenc` is not defined on Apple.
///
/// Arithmetic is C++ `int` division/multiplication done in `i64`; a zero
/// divisor (undefined behaviour in C++) yields a 0 buffer instead of a panic.
pub fn rate_control(
    flags: u32,
    encoder_name: &str,
    hardware: bool,
    slices: i32,
    params: &SessionParams,
    max_bitrate: i32,
    nv_vbv_percentage_increase: i32,
) -> RateControl {
    let requested = if max_bitrate > 0 {
        params.bitrate.min(max_bitrate)
    } else {
        params.bitrate
    };
    let bitrate = requested as i64 * 1000; // 2247
    let rc_max_rate = bitrate; // 2249
    let mut bit_rate = bitrate; // 2250

    let rc_min_rate = if has(flags, flag::CBR_WITH_VBR) {
        // Ensure rc_max_bitrate != bit_rate to force VBR mode
        bit_rate -= 1; // 2254
        None
    } else {
        Some(bitrate) // 2256
    };

    let compliance_unofficial = has(flags, flag::RELAXED_COMPLIANCE); // 2259-2261

    let framerate = params.framerate as i64;
    let rc_buffer_size = if !has(flags, flag::NO_RC_BUF_LIMIT) {
        // 2263
        if !hardware && (slices > 1 || params.video_format == 1) {
            // Use a larger rc_buffer_size for software encoding when slices are
            // enabled, because libx264 can severely degrade quality if the
            // buffer is too small. libx265 encounters this issue more
            // frequently, so always scale the buffer by 1.5x for software HEVC
            // encoding.
            let divisor = (framerate * 10) / 15; // 2269
            Some(bitrate.checked_div(divisor).unwrap_or(0))
        } else {
            let mut size = bitrate.checked_div(framerate).unwrap_or(0); // 2271
            if encoder_name == "nvenc" && nv_vbv_percentage_increase > 0 {
                // 2274-2275
                size += size * nv_vbv_percentage_increase as i64 / 100;
            }
            Some(size)
        }
    } else {
        None
    };

    RateControl {
        bitrate,
        rc_max_rate,
        bit_rate,
        rc_min_rate,
        compliance_unofficial,
        rc_buffer_size,
    }
}

// ---------------------------------------------------------------------------
// Codec context flags (video.cpp:2109-2113)
// ---------------------------------------------------------------------------

/// The `ctx->flags` / `ctx->flags2` bits set before `avcodec_open2`
/// (`video.cpp:2109-2113`). `ctx->flags` is first reset to 0 ("to avoid clash
/// on reuse of AVCodecContext", 2109-2110).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextFlags {
    /// `AV_CODEC_FLAG_CLOSED_GOP` (2111).
    pub closed_gop: bool,
    /// `AV_CODEC_FLAG_LOW_DELAY` (2111).
    pub low_delay: bool,
    /// `AV_CODEC_FLAG2_FAST` (2113).
    pub flag2_fast: bool,
}

/// `ctx->flags = 0; ctx->flags |= CLOSED_GOP | LOW_DELAY; ctx->flags2 |= FAST`
/// (`video.cpp:2110-2113`). Unconditional.
pub const CONTEXT_FLAGS: ContextFlags = ContextFlags {
    closed_gop: true,
    low_delay: true,
    flag2_fast: true,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::ConfigDefault::{Int, Str};
    use super::*;

    fn params(width: i32, height: i32, framerate: i32) -> SessionParams {
        SessionParams {
            width,
            height,
            framerate,
            bitrate: 20000,
            slices_per_frame: 1,
            num_ref_frames: 0,
            video_format: 0,
            dynamic_range: 0,
            chroma_sampling_type: 0,
        }
    }

    // Second, independent transcription of video.cpp:983-1100 with the
    // defaults of config.cpp:737-821 / config.cpp:119-157 worked out by hand.

    #[test]
    fn amdvce_h264_common_with_defaults() {
        // video.cpp:1060-1079; max_au_size (1078) is unset by default => skipped.
        let expected: Vec<(&str, ConfigDefault)> = vec![
            ("filler_data", Int(0)),
            ("forced_idr", Int(1)),
            ("latency", Int(1)),
            ("async_depth", Int(1)),
            ("frame_skipping", Int(0)),
            ("log_to_dbg", Int(0)), // min_log_level 2 < 2 is false
            ("preencode", Int(0)),
            ("quality", Int(0)), // h264 balanced = 0 (config.cpp:127)
            ("rc", Int(3)),      // h264 vbr_latency = 3 (config.cpp:139)
            ("usage", Int(1)),   // h264 ultralowlatency = 1 (config.cpp:151)
            ("vbaq", Int(1)),
            ("coder", Int(0)),
            ("enforce_hrd", Int(0)),
        ];
        let got = options_for(&AMDVCE.h264, &params(1920, 1080, 60), false);
        assert_eq!(got, expected);
    }

    #[test]
    fn amdvce_hevc_common_with_defaults() {
        // video.cpp:1020-1051, 1920x1080@60 => level 5.1.
        let expected: Vec<(&str, ConfigDefault)> = vec![
            ("filler_data", Int(0)),
            ("forced_idr", Int(1)),
            ("latency", Int(1)),
            ("async_depth", Int(1)),
            ("skip_frame", Int(0)),
            ("log_to_dbg", Int(0)),
            ("gops_per_idr", Int(1)),
            ("header_insertion_mode", Str("idr")),
            ("preencode", Int(0)),
            ("quality", Int(5)), // hevc balanced = 5 (config.cpp:124)
            ("rc", Int(1)),      // hevc vbr_latency = 1 (config.cpp:135)
            ("usage", Int(1)),   // hevc ultralowlatency = 1 (config.cpp:146)
            ("vbaq", Int(1)),
            ("enforce_hrd", Int(0)),
            ("level", Str("5.1")),
        ];
        let got = options_for(&AMDVCE.hevc, &params(1920, 1080, 60), false);
        assert_eq!(got, expected);
    }

    #[test]
    fn amdvce_av1_common_with_defaults() {
        // video.cpp:995-1011.
        let expected: Vec<(&str, ConfigDefault)> = vec![
            ("filler_data", Int(0)),
            ("forced_idr", Int(1)),
            ("latency", Str("lowest_latency")),
            ("async_depth", Int(1)),
            ("skip_frame", Int(0)),
            ("log_to_dbg", Int(0)),
            ("preencode", Int(0)),
            ("quality", Int(70)), // av1 balanced = 70 (config.cpp:121)
            ("rc", Int(1)),       // av1 vbr_latency = 1 (config.cpp:131)
            ("usage", Int(2)),    // av1 ultralowlatency = 2 (config.cpp:142)
            ("enforce_hrd", Int(0)),
        ];
        let got = options_for(&AMDVCE.av1, &params(1920, 1080, 60), false);
        assert_eq!(got, expected);
    }

    #[test]
    fn amdvce_h264_fallback_appends_usage() {
        // video.cpp:1084-1087.
        let base = options_for(&AMDVCE.h264, &params(1920, 1080, 60), false);
        let retry = options_for(&AMDVCE.h264, &params(1920, 1080, 60), true);
        assert_eq!(retry.len(), base.len() + 1);
        assert_eq!(retry.last(), Some(&("usage", Int(2))));
    }

    #[test]
    fn quicksync_hevc_retry_overrides_low_power() {
        // video.cpp:917 then fallback 939-941 (qsv_slow_hevc default false => 1).
        let retry = options_for(&QUICKSYNC.hevc, &params(1920, 1080, 60), true);
        assert_eq!(retry.last(), Some(&("low_power", Int(1))));
        // HDR 4:4:4 adds the hdr then hdr444 lists in that order (video.cpp:2233-2240).
        let mut p = params(1920, 1080, 60);
        p.dynamic_range = 1;
        p.chroma_sampling_type = 1;
        let got = options_for(&QUICKSYNC.hevc, &p, false);
        let profiles: Vec<_> = got.iter().filter(|(n, _)| *n == "profile").collect();
        assert_eq!(profiles, vec![&("profile", Int(2)), &("profile", Int(4))]);
    }

    #[test]
    fn nvenc_hevc_options_and_unset_skipping() {
        let got = options_for(&NVENC.hevc, &params(1920, 1080, 60), false);
        assert!(got.contains(&("preset", Str("p1"))));
        assert!(got.contains(&("multipass", Int(1))));
        assert_eq!(got.last(), Some(&("profile", Int(0))));
        // AMD max_au_size is unset by default and must not appear (video.cpp:2205-2209).
        let amd = options_for(&AMDVCE.h264, &params(1920, 1080, 60), false);
        assert!(!amd.iter().any(|(n, _)| *n == "max_au_size"));
    }

    #[test]
    fn amd_hevc_level_lambda() {
        // video.cpp:1040-1050
        let lvl = |w, h, fps| compute(Computed::AmdHevcLevel, &params(w, h, fps));
        assert_eq!(lvl(1280, 720, 60), Some(Str("5.1")));
        assert_eq!(lvl(3840, 2160, 60), Some(Str("5.1"))); // 497,664,000 <= 534,773,760
        assert_eq!(lvl(3840, 2160, 120), Some(Str("5.2"))); // 995,328,000 <= 1,069,547,520
        assert_eq!(lvl(7680, 4320, 60), Some(Str("auto"))); // size 33,177,600 > 8,912,896
    }

    #[test]
    fn log_to_dbg_and_qsv_low_power_lambdas() {
        let p = params(1920, 1080, 60);
        assert_eq!(compute(Computed::AmdLogToDbg, &p), Some(Int(0)));
        assert_eq!(compute(Computed::QsvHevcLowPower, &p), Some(Int(1)));
    }

    #[test]
    fn h264_profile_branches() {
        // video.cpp:152-162
        assert_eq!(select_h264_profile("h264_amf", 1, 2), Profile::H264High444Predictive);
        assert_eq!(select_h264_profile("h264_nvenc", 1, 0), Profile::H264High444Predictive);
        assert_eq!(select_h264_profile("h264_amf", 0, AMF_CODER_CAVLC), Profile::H264ConstrainedBaseline);
        assert_eq!(select_h264_profile("h264_amf", 0, 0), Profile::H264High);
        assert_eq!(select_h264_profile("h264_nvenc", 0, AMF_CODER_CAVLC), Profile::H264High);
    }

    #[test]
    fn hevc_av1_profile_choice() {
        // video.cpp:2061-2082
        assert_eq!(profile(1, 0, 1, "hevc_nvenc", 0), Profile::HevcRext);
        assert_eq!(profile(1, 1, 1, "hevc_nvenc", 0), Profile::HevcRext);
        assert_eq!(profile(1, 1, 0, "hevc_nvenc", 0), Profile::HevcMain10);
        assert_eq!(profile(1, 0, 0, "hevc_nvenc", 0), Profile::HevcMain);
        assert_eq!(profile(2, 1, 0, "av1_nvenc", 0), Profile::Av1Main);
        assert_eq!(profile(2, 0, 1, "av1_nvenc", 0), Profile::Av1High);
        assert_eq!(profile(0, 0, 0, "h264_amf", 2), Profile::H264ConstrainedBaseline);
        assert_eq!(profile(7, 0, 0, "x", 0), Profile::Unset);
    }

    #[test]
    fn gop_cases() {
        // video.cpp:2087-2098
        assert_eq!(gop(flag::DEFAULT), (i32::MAX, i32::MAX));
        assert_eq!(gop(flag::PARALLEL_ENCODING), (i32::MAX, i32::MAX));
        assert_eq!(gop(flag::LIMITED_GOP_SIZE), (32767, i32::MAX));
        assert_eq!(gop(flag::FIXED_GOP_SIZE), (120, 120));
        // FIXED_GOP_SIZE wins over LIMITED_GOP_SIZE (checked first, 2089).
        assert_eq!(gop(flag::FIXED_GOP_SIZE | flag::LIMITED_GOP_SIZE), (120, 120));
        assert_eq!(gop(VAAPI.flags), (32767, i32::MAX));
        assert_eq!(gop(MEDIAFOUNDATION.flags), (120, 120));
    }

    #[test]
    fn slices_cases() {
        // video.cpp:2178-2193
        let hw = slices(flag::PARALLEL_ENCODING, true, 1, DEFAULT_MIN_THREADS);
        assert_eq!((hw.slices, hw.thread_count), (1, 1));
        let sw = slices(SOFTWARE.flags, false, 1, DEFAULT_MIN_THREADS);
        assert_eq!((sw.slices, sw.thread_count), (2, 2));
        let sw4 = slices(SOFTWARE.flags, false, 4, DEFAULT_MIN_THREADS);
        assert_eq!(sw4.slices, 4);
        let single = slices(flag::SINGLE_SLICE_ONLY, true, 4, DEFAULT_MIN_THREADS);
        assert_eq!((single.slices, single.thread_count), (1, 1));
    }

    #[test]
    fn rate_control_hardware() {
        // 20000 kbps @ 60 fps, hardware, default flags (video.cpp:2247-2271).
        let p = params(1920, 1080, 60);
        let rc = rate_control(flag::PARALLEL_ENCODING, "amdvce", true, 1, &p, DEFAULT_MAX_BITRATE, 0);
        assert_eq!(rc.rc_max_rate, 20_000_000);
        assert_eq!(rc.bit_rate, 20_000_000);
        assert_eq!(rc.rc_min_rate, Some(20_000_000));
        assert!(!rc.compliance_unofficial);
        assert_eq!(rc.rc_buffer_size, Some(333_333)); // 20_000_000 / 60
    }

    #[test]
    fn rate_control_max_bitrate_cap() {
        let p = params(1920, 1080, 60);
        let rc = rate_control(flag::DEFAULT, "amdvce", true, 1, &p, 10000, 0);
        assert_eq!(rc.rc_max_rate, 10_000_000);
    }

    #[test]
    fn rate_control_software_multi_slice() {
        // Software with slices > 1: buffer scaled by 1.5x (video.cpp:2264-2269).
        let p = params(1920, 1080, 60);
        let rc = rate_control(SOFTWARE.flags, "software", false, 4, &p, DEFAULT_MAX_BITRATE, 0);
        assert_eq!(rc.rc_buffer_size, Some(500_000)); // 20_000_000 / ((60*10)/15 = 40)
        // Software HEVC always scales, even with one slice.
        let mut hevc = p;
        hevc.video_format = 1;
        let rc = rate_control(SOFTWARE.flags, "software", false, 1, &hevc, DEFAULT_MAX_BITRATE, 0);
        assert_eq!(rc.rc_buffer_size, Some(500_000));
        // Software H.264, one slice: plain bitrate / framerate.
        let rc = rate_control(SOFTWARE.flags, "software", false, 1, &p, DEFAULT_MAX_BITRATE, 0);
        assert_eq!(rc.rc_buffer_size, Some(333_333));
    }

    #[test]
    fn rate_control_cbr_with_vbr_and_relaxed_compliance() {
        // quicksync flags (video.cpp:977): CBR_WITH_VBR | RELAXED_COMPLIANCE | NO_RC_BUF_LIMIT
        let p = params(1920, 1080, 60);
        let rc = rate_control(QUICKSYNC.flags, "quicksync", true, 1, &p, DEFAULT_MAX_BITRATE, 0);
        assert_eq!(rc.rc_max_rate, 20_000_000);
        assert_eq!(rc.bit_rate, 19_999_999); // 2254
        assert_eq!(rc.rc_min_rate, None);
        assert!(rc.compliance_unofficial);
        assert_eq!(rc.rc_buffer_size, None); // NO_RC_BUF_LIMIT
    }

    #[test]
    fn rate_control_no_rc_buf_limit_only() {
        let p = params(1920, 1080, 60);
        let rc = rate_control(VAAPI.flags, "vaapi", true, 1, &p, DEFAULT_MAX_BITRATE, 0);
        assert_eq!(rc.rc_buffer_size, None);
        assert_eq!(rc.rc_min_rate, Some(20_000_000));
        assert_eq!(rc.bit_rate, 20_000_000);
    }

    #[test]
    fn rate_control_nvenc_vbv_increase() {
        // video.cpp:2274-2276: only for encoder.name == "nvenc" and percentage > 0.
        let p = params(1920, 1080, 60);
        let base = rate_control(NVENC.flags, "nvenc", true, 1, &p, DEFAULT_MAX_BITRATE, 0);
        assert_eq!(base.rc_buffer_size, Some(333_333));
        let inc = rate_control(NVENC.flags, "nvenc", true, 1, &p, DEFAULT_MAX_BITRATE, 50);
        assert_eq!(inc.rc_buffer_size, Some(333_333 + 333_333 * 50 / 100));
        let other = rate_control(NVENC.flags, "vaapi", true, 1, &p, DEFAULT_MAX_BITRATE, 50);
        assert_eq!(other.rc_buffer_size, Some(333_333));
    }

    #[test]
    fn context_flags_and_pix_fmt() {
        assert!(CONTEXT_FLAGS.closed_gop && CONTEXT_FLAGS.low_delay && CONTEXT_FLAGS.flag2_fast);
        assert_eq!(MAX_B_FRAMES, 0);
        assert_eq!(sw_pix_fmt(&AMDVCE, 8, 0), PixFmt::Nv12);
        assert_eq!(sw_pix_fmt(&AMDVCE, 10, 0), PixFmt::P010);
        assert_eq!(sw_pix_fmt(&AMDVCE, 8, 1), PixFmt::None);
        assert_eq!(sw_pix_fmt(&QUICKSYNC, 10, 1), PixFmt::Xv30);
        assert_eq!(sw_pix_fmt(&SOFTWARE, 8, 1), PixFmt::Yuv444p);
        assert_eq!(sw_pix_fmt(&SOFTWARE, 12, 0), PixFmt::None);
    }

    #[test]
    fn flag_values_and_tables() {
        assert_eq!(flag::PARALLEL_ENCODING, 2);
        assert_eq!(flag::FIXED_GOP_SIZE, 4096);
        assert_eq!(AMDVCE.flags, flag::PARALLEL_ENCODING);
        assert_eq!(SOFTWARE.av1.name, "");
        assert_eq!(AMDVCE.h264.name, "h264_amf");
        assert!(AMDVCE.is_hardware() && !SOFTWARE.is_hardware());
        assert_eq!(AMDVCE.codec_from_video_format(1).name, "hevc_amf");
        assert_eq!(AMDVCE.codec_from_video_format(2).name, "av1_amf");
        assert_eq!(AMDVCE.codec_from_video_format(9).name, "h264_amf");
        let linux: Vec<_> = encoders_for(Platform::Linux).iter().map(|e| e.name).collect();
        assert_eq!(linux, vec!["nvenc", "vulkan", "vaapi", "software"]);
        let win: Vec<_> = encoders_for(Platform::Windows).iter().map(|e| e.name).collect();
        assert_eq!(win, vec!["quicksync", "amdvce", "mediafoundation", "software"]);
        let mac: Vec<_> = encoders_for(Platform::MacOs).iter().map(|e| e.name).collect();
        assert_eq!(mac, vec!["videotoolbox", "software"]);
    }

    #[test]
    fn config_defaults_cover_every_key_used_by_the_tables() {
        // Every Config option in every table resolves, except max_au_size (unset).
        for e in ENCODERS {
            for c in [&e.av1, &e.hevc, &e.h264] {
                for list in [c.common, c.sdr, c.hdr, c.sdr444, c.hdr444, c.fallback] {
                    for o in list {
                        if let OptionValue::Config(k) = o.value {
                            if k != ConfigKey::AmdMaxAuSize {
                                assert!(config_default(k).is_some(), "{} {:?}", o.name, k);
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(config_default(ConfigKey::AmdMaxAuSize), None);
    }
}
