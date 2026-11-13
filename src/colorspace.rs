//! Colorspace selection and RGB->YUV conversion vectors.
//!
//! Port of Sunshine `src/video_colorspace.h` and `src/video_colorspace.cpp`
//! (reference commit f186813). Line references in comments point to
//! `src/video_colorspace.cpp` unless stated otherwise.
//!
//! This module has no dependencies outside `std` and `log`, and no FFmpeg types.
//! The FFmpeg enumerator values used by Sunshine are mirrored as ITU-T H.273
//! code points (see the `COLOR_PRIMARIES_*`, `TRANSFER_*` and `MATRIX_*` constants).

// ---------------------------------------------------------------------------
// H.273 code points
//
// Values taken from /usr/include/x86_64-linux-gnu/libavutil/pixfmt.h
// (FFmpeg enumerators named in each comment).
// ---------------------------------------------------------------------------

/// `AVCOL_PRI_BT709` (pixfmt.h:638).
pub const COLOR_PRIMARIES_BT709: u8 = 1;
/// `AVCOL_PRI_SMPTE170M` (pixfmt.h:644).
pub const COLOR_PRIMARIES_SMPTE170M: u8 = 6;
/// `AVCOL_PRI_BT2020` (pixfmt.h:647).
pub const COLOR_PRIMARIES_BT2020: u8 = 9;

/// `AVCOL_TRC_BT709` (pixfmt.h:663).
pub const TRANSFER_BT709: u8 = 1;
/// `AVCOL_TRC_SMPTE170M` (pixfmt.h:668).
pub const TRANSFER_SMPTE170M: u8 = 6;
/// `AVCOL_TRC_BT2020_10` (pixfmt.h:676).
pub const TRANSFER_BT2020_10: u8 = 14;
/// `AVCOL_TRC_SMPTE2084` (pixfmt.h:678).
pub const TRANSFER_SMPTE2084: u8 = 16;

/// `AVCOL_SPC_BT709` (pixfmt.h:692).
pub const MATRIX_BT709: u8 = 1;
/// `AVCOL_SPC_SMPTE170M` (pixfmt.h:697).
pub const MATRIX_SMPTE170M: u8 = 6;
/// `AVCOL_SPC_BT2020_NCL` (pixfmt.h:701).
pub const MATRIX_BT2020_NCL: u8 = 9;

/// Enumerates supported colorspace options.
/// Port of `video::colorspace_e` (video_colorspace.h:18-23).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colorspace {
    /// Rec. 601 (`rec601`).
    Rec601,
    /// Rec. 709 (`rec709`).
    Rec709,
    /// Rec. 2020 SDR (`bt2020sdr`).
    Bt2020Sdr,
    /// Rec. 2020 HDR (`bt2020`).
    Bt2020,
}

impl Colorspace {
    /// Describe a colorspace for encoder diagnostic messages.
    /// Port of `colorspace_to_string` (video_colorspace.cpp:23-37).
    /// The reference "unknown" default is unreachable for a closed enum.
    pub fn label(self) -> &'static str {
        match self {
            Colorspace::Rec601 => "SDR (Rec. 601)",
            Colorspace::Rec709 => "SDR (Rec. 709)",
            Colorspace::Bt2020Sdr => "SDR (Rec. 2020)",
            Colorspace::Bt2020 => "HDR (Rec. 2020 + SMPTE 2084 PQ)",
        }
    }
}

/// Sunshine colorimetry values derived from stream configuration.
/// Port of `video::sunshine_colorspace_t` (video_colorspace.h:36-40).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamColorspace {
    /// Colorspace.
    pub colorspace: Colorspace,
    /// Whether the video range is full-range instead of limited.
    pub full_range: bool,
    /// Bit depth (8 or 10 for values produced by `from_client_config`).
    pub bit_depth: u32,
}

impl StreamColorspace {
    /// Check whether this colorspace represents HDR video.
    /// Port of `colorspace_is_hdr` (video_colorspace.cpp:42-44).
    pub fn is_hdr(&self) -> bool {
        self.colorspace == Colorspace::Bt2020
    }
}

/// Derive Sunshine colorspace metadata from the client stream configuration.
///
/// Port of `colorspace_from_client_config` (video_colorspace.cpp:49-105).
/// Parameters map to the reference `config_t` fields:
/// - `encoder_csc_mode` = `config.encoderCscMode` (bit 0 = full range, bits 1.. = colorspace)
/// - `dynamic_range` = `config.dynamicRange` (0 = 8-bit, 1 = 10-bit)
///
/// Field comments in the reference (src/video.h:44, 46):
/// `encoderCscMode`: "Requested color range and SDR colorspace; HDR always uses BT.2020 and ST2084."
/// `dynamicRange`: "Encoding color depth: 0 = 8-bit, 1 = 10-bit."
pub fn from_client_config(encoder_csc_mode: i32, dynamic_range: i32, hdr_display: bool) -> StreamColorspace {
    // Reference lines 55-80: HDR is selected only when dynamicRange > 0 and the display is HDR.
    let mut colorspace = if dynamic_range > 0 && hdr_display {
        // Rec. 2020 with ST 2084 perceptual quantizer (line 57)
        Colorspace::Bt2020
    } else {
        // Reference line 59: `switch (config.encoderCscMode >> 1)`
        match encoder_csc_mode >> 1 {
            0 => Colorspace::Rec601,
            1 => Colorspace::Rec709,
            2 => Colorspace::Bt2020Sdr,
            _ => {
                // Reference line 76
                log::error!("Unknown video colorspace in csc, falling back to Rec. 709");
                Colorspace::Rec709
            }
        }
    };

    // Reference line 82: `(config.encoderCscMode & 0x1)` converted to bool.
    let full_range = (encoder_csc_mode & 0x1) != 0;

    // Reference lines 84-97
    let bit_depth: u32 = match dynamic_range {
        0 => 8,
        1 => 10,
        _ => {
            // Reference line 94
            log::error!("Unknown dynamicRange value, falling back to 10-bit color depth");
            10
        }
    };

    // Reference lines 99-102
    if colorspace == Colorspace::Bt2020Sdr && bit_depth != 10 {
        // Reference line 100
        log::error!("BT.2020 SDR colorspace expects 10-bit color depth, falling back to Rec. 709");
        colorspace = Colorspace::Rec709;
    }

    StreamColorspace {
        colorspace,
        full_range,
        bit_depth,
    }
}

/// FFmpeg colorimetry values used by AVCodec encoders, expressed as H.273 code points.
///
/// Port of `video::avcodec_colorspace_t` (video_colorspace.h:65-71) without
/// `software_format`: that field is the swscale constant `SWS_CS_*` (reference
/// lines 119, 127, 136, 145). The crate root has no swscale type, so it is
/// exposed separately as [`SoftwareFormat`] / [`software_format`].
///
/// `full_range` corresponds to the reference `range` field: `AVCOL_RANGE_JPEG`
/// when true, `AVCOL_RANGE_MPEG` when false (reference line 149).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodecColorspace {
    /// H.273 colour primaries code point (`AVCOL_PRI_*`).
    pub primaries: u8,
    /// H.273 transfer characteristics code point (`AVCOL_TRC_*`).
    pub transfer: u8,
    /// H.273 matrix coefficients code point (`AVCOL_SPC_*`).
    pub matrix: u8,
    /// True for full range (`AVCOL_RANGE_JPEG`), false for limited (`AVCOL_RANGE_MPEG`).
    pub full_range: bool,
}

/// Convert Sunshine colorspace metadata to codec colorimetry code points.
///
/// Port of `avcodec_colorspace_from_sunshine_colorspace` (video_colorspace.cpp:110-152).
///
/// The reference `assert(bit_depth == 10)` (lines 133 and 142) is a `debug_assert!` here.
pub fn codec_colorspace(sunshine_colorspace: &StreamColorspace) -> CodecColorspace {
    let (primaries, transfer, matrix) = match sunshine_colorspace.colorspace {
        // Rec. 601 (lines 114-120)
        Colorspace::Rec601 => (
            COLOR_PRIMARIES_SMPTE170M,
            TRANSFER_SMPTE170M,
            MATRIX_SMPTE170M,
        ),
        // Rec. 709 (lines 122-128)
        Colorspace::Rec709 => (COLOR_PRIMARIES_BT709, TRANSFER_BT709, MATRIX_BT709),
        // Rec. 2020 (lines 130-137)
        Colorspace::Bt2020Sdr => {
            debug_assert!(sunshine_colorspace.bit_depth == 10);
            (
                COLOR_PRIMARIES_BT2020,
                TRANSFER_BT2020_10,
                MATRIX_BT2020_NCL,
            )
        }
        // Rec. 2020 with ST 2084 PQ (lines 139-146)
        Colorspace::Bt2020 => {
            debug_assert!(sunshine_colorspace.bit_depth == 10);
            (
                COLOR_PRIMARIES_BT2020,
                TRANSFER_SMPTE2084,
                MATRIX_BT2020_NCL,
            )
        }
    };

    CodecColorspace {
        primaries,
        transfer,
        matrix,
        full_range: sunshine_colorspace.full_range,
    }
}

/// swscale coefficient set (`avcodec_colorspace_t::software_format`,
/// video_colorspace.cpp:119, 127, 136, 145).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftwareFormat {
    /// `SWS_CS_SMPTE170M` (line 119).
    Smpte170m,
    /// `SWS_CS_ITU709` (line 127).
    Itu709,
    /// `SWS_CS_BT2020` (lines 136, 145).
    Bt2020,
}

/// swscale coefficient set for a colourspace (`software_format`,
/// video_colorspace.cpp:119, 127, 136, 145).
pub fn software_format(cs: &StreamColorspace) -> SoftwareFormat {
    match cs.colorspace {
        Colorspace::Rec601 => SoftwareFormat::Smpte170m,
        Colorspace::Rec709 => SoftwareFormat::Itu709,
        Colorspace::Bt2020Sdr | Colorspace::Bt2020 => SoftwareFormat::Bt2020,
    }
}

/// Pair of Sunshine and AVCodec colorimetry descriptions.
///
/// Port of `video::color_t` (video_colorspace.h:84-90), which is declared
/// `struct alignas(16)` with `std::array<float, N>` fields. This struct uses
/// `#[repr(C, align(16))]` to match: `repr(C)` keeps the field order, and
/// `align(16)` matches `alignas(16)`. Each `[f32; 4]` / `[f32; 2]` maps to
/// `std::array<float, 4>` / `std::array<float, 2>`. The struct is intended for
/// GPU constant buffers, so the layout is kept identical to the reference.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorVectors {
    /// Color vec y (`color_vec_y`).
    pub color_vec_y: [f32; 4],
    /// Color vec u (`color_vec_u`).
    pub color_vec_u: [f32; 4],
    /// Color vec v (`color_vec_v`).
    pub color_vec_v: [f32; 4],
    /// Range y (`range_y`). Unused by the computation (reference line 229).
    pub range_y: [f32; 2],
    /// Range uv (`range_uv`). Unused by the computation (reference line 229).
    pub range_uv: [f32; 2],
}

/// Return the RGB->YUV conversion vectors for the requested colorspace and range.
///
/// Port of `color_vectors_from_colorspace` (video_colorspace.cpp:158-293). The reference
/// builds a static table of 24 entries with `generate_color_vectors`, then picks one entry.
/// This function computes the same value on demand, using the same selection rules:
///
/// - Matrix constants come from the colorspace (lines 165-179): Rec. 601 uses
///   Kr=0.299, Kb=0.114; Bt2020 and Bt2020Sdr both use Kr=0.2627, Kb=0.0593; Rec. 709 uses
///   Kr=0.2126, Kb=0.0722.
/// - The table has only 8-bit and 10-bit entries. Line 282 selects the 10-bit entry only
///   when `bit_depth == 10`, so any other bit depth (for example 12 or 0) uses the 8-bit
///   entry. The arithmetic below therefore uses an effective bit depth of 10 or 8.
/// - `full_range` selects the full-range entry (line 285).
/// - `unorm_output` selects the normalised (divided by 2^bd - 1) entries (line 288). Without
///   it, the rounding offset 0.5 is added to y_add and uv_add (lines 207-209).
///
/// The reference computes in `double` and stores into `float` (lines 212-227). This
/// function does the same: f64 arithmetic in the same operation order, then `as f32`.
pub fn color_vectors(colorspace: &StreamColorspace, unorm_output: bool) -> ColorVectors {
    // Effective bit depth used by the table (lines 282-284): only 10 selects the 10-bit entries.
    let bit_depth: u32 = if colorspace.bit_depth == 10 { 10 } else { 8 };

    // Lines 165-179
    let (kr, kb): (f64, f64) = match colorspace.colorspace {
        Colorspace::Rec601 => (0.299, 0.114),
        Colorspace::Bt2020 | Colorspace::Bt2020Sdr => (0.2627, 0.0593),
        Colorspace::Rec709 => (0.2126, 0.0722),
    };
    // Line 180
    let kg = 1.0 - kr - kb;

    // Lines 182-198 ("8.3 Matrix coefficients" of ITU-T H.273)
    let mut y_mult: f64;
    let mut y_add: f64;
    let mut uv_mult: f64;
    let mut uv_add: f64;
    if colorspace.full_range {
        y_mult = ((1u32 << bit_depth) - 1) as f64;
        y_add = 0.0;
        uv_mult = ((1u32 << bit_depth) - 1) as f64;
        uv_add = (1u32 << (bit_depth - 1)) as f64;
    } else {
        y_mult = ((1u32 << (bit_depth - 8)) * 219) as f64;
        y_add = ((1u32 << (bit_depth - 8)) * 16) as f64;
        uv_mult = ((1u32 << (bit_depth - 8)) * 224) as f64;
        uv_add = ((1u32 << (bit_depth - 8)) * 128) as f64;
    }

    // Lines 200-210
    if unorm_output {
        let unorm_range = ((1u32 << bit_depth) - 1) as f64;
        y_mult /= unorm_range;
        y_add /= unorm_range;
        uv_mult /= unorm_range;
        uv_add /= unorm_range;
    } else {
        // For rounding (reference uses the float literal 0.5f; exact in f64)
        y_add += 0.5;
        uv_add += 0.5;
    }

    // Lines 212-233
    let mut color_vectors = ColorVectors {
        color_vec_y: [0.0; 4],
        color_vec_u: [0.0; 4],
        color_vec_v: [0.0; 4],
        range_y: [0.0; 2],
        range_uv: [0.0; 2],
    };

    color_vectors.color_vec_y[0] = (kr * y_mult) as f32;
    color_vectors.color_vec_y[1] = (kg * y_mult) as f32;
    color_vectors.color_vec_y[2] = (kb * y_mult) as f32;
    color_vectors.color_vec_y[3] = y_add as f32;

    color_vectors.color_vec_u[0] = (-0.5 * kr / (1.0 - kb) * uv_mult) as f32;
    color_vectors.color_vec_u[1] = (-0.5 * kg / (1.0 - kb) * uv_mult) as f32;
    color_vectors.color_vec_u[2] = (0.5 * uv_mult) as f32;
    color_vectors.color_vec_u[3] = uv_add as f32;

    color_vectors.color_vec_v[0] = (0.5 * uv_mult) as f32;
    color_vectors.color_vec_v[1] = (-0.5 * kg / (1.0 - kr) * uv_mult) as f32;
    color_vectors.color_vec_v[2] = (-0.5 * kb / (1.0 - kr) * uv_mult) as f32;
    color_vectors.color_vec_v[3] = uv_add as f32;

    // Unused (lines 229-233)
    color_vectors.range_y = [1.0, 0.0];
    color_vectors.range_uv = [1.0, 0.0];

    color_vectors
}

#[cfg(test)]
mod tests {
    use super::*;

    // The vectors are stored as f32. At magnitude 65 one f32 ulp is about 7.6e-6, so an absolute
    // 1e-6 tolerance cannot hold there. The check is 1e-6 relative to max(1, |expected|).
    const TOL: f64 = 1e-6;

    fn assert_close(actual: f32, expected: f64, what: &str) {
        let diff = (actual as f64 - expected).abs();
        let bound = TOL * expected.abs().max(1.0);
        assert!(
            diff <= bound,
            "{what}: actual {actual}, expected {expected}, diff {diff}, bound {bound}"
        );
    }

    fn assert_vectors(cv: &ColorVectors, y: [f64; 4], u: [f64; 4], v: [f64; 4]) {
        for i in 0..4 {
            assert_close(cv.color_vec_y[i], y[i], &format!("color_vec_y[{i}]"));
            assert_close(cv.color_vec_u[i], u[i], &format!("color_vec_u[{i}]"));
            assert_close(cv.color_vec_v[i], v[i], &format!("color_vec_v[{i}]"));
        }
    }

    fn sc(colorspace: Colorspace, full_range: bool, bit_depth: u32) -> StreamColorspace {
        StreamColorspace {
            colorspace,
            full_range,
            bit_depth,
        }
    }

    // ---- from_client_config --------------------------------------------------

    #[test]
    fn from_client_config_sdr_and_hdr_table() {
        // (encoderCscMode, dynamicRange, hdr_display) -> (colorspace, full_range, bit_depth)
        // csc >> 1: 0 = Rec601, 1 = Rec709, 2 = Bt2020Sdr; csc & 1 = full range.
        // HDR (dynamicRange > 0 && hdr_display) always gives Bt2020 with 10-bit.
        use Colorspace::*;
        let cases: &[(i32, i32, bool, Colorspace, bool, u32)] = &[
            // csc 0 (Rec601 limited), 1 (Rec601 full)
            (0, 0, false, Rec601, false, 8),
            (0, 1, false, Rec601, false, 10),
            (0, 0, true, Rec601, false, 8),
            (0, 1, true, Bt2020, false, 10),
            (1, 0, false, Rec601, true, 8),
            (1, 1, false, Rec601, true, 10),
            (1, 0, true, Rec601, true, 8),
            (1, 1, true, Bt2020, true, 10),
            // csc 2 (Rec709 limited), 3 (Rec709 full)
            (2, 0, false, Rec709, false, 8),
            (2, 1, false, Rec709, false, 10),
            (2, 0, true, Rec709, false, 8),
            (2, 1, true, Bt2020, false, 10),
            (3, 0, false, Rec709, true, 8),
            (3, 1, false, Rec709, true, 10),
            (3, 0, true, Rec709, true, 8),
            (3, 1, true, Bt2020, true, 10),
            // csc 4 (Bt2020Sdr limited), 5 (Bt2020Sdr full)
            (4, 0, false, Rec709, false, 8), // BT.2020 SDR with 8-bit -> Rec. 709 fallback
            (4, 1, false, Bt2020Sdr, false, 10),
            (4, 0, true, Rec709, false, 8),
            (4, 1, true, Bt2020, false, 10),
            (5, 0, false, Rec709, true, 8), // BT.2020 SDR with 8-bit -> Rec. 709 fallback
            (5, 1, false, Bt2020Sdr, true, 10),
            (5, 0, true, Rec709, true, 8),
            (5, 1, true, Bt2020, true, 10),
        ];
        for &(csc, dr, hdr, cs, full, bd) in cases {
            let got = from_client_config(csc, dr, hdr);
            assert_eq!(
                got,
                sc(cs, full, bd),
                "encoderCscMode={csc} dynamicRange={dr} hdr_display={hdr}"
            );
        }
    }

    #[test]
    fn from_client_config_unknown_csc_falls_back_to_rec709() {
        // csc >> 1 == 3 (csc 6, 7) and negative values hit the default branch (line 75-78).
        assert_eq!(from_client_config(6, 1, false).colorspace, Colorspace::Rec709);
        assert_eq!(from_client_config(7, 0, false).colorspace, Colorspace::Rec709);
        assert_eq!(from_client_config(-2, 1, false).colorspace, Colorspace::Rec709);
        // Full range bit of an unknown csc is still taken from bit 0.
        assert!(from_client_config(7, 1, false).full_range);
    }

    #[test]
    fn from_client_config_unknown_dynamic_range_falls_back_to_10bit() {
        // Lines 93-96: any dynamicRange other than 0 or 1 gives 10-bit.
        assert_eq!(from_client_config(0, 2, false), sc(Colorspace::Rec601, false, 10));
        // encoderCscMode 4 gives csc >> 1 == 2, which is Bt2020Sdr (2 would give Rec. 709).
        assert_eq!(from_client_config(4, -1, false), sc(Colorspace::Bt2020Sdr, false, 10));
        // dynamicRange 2 with HDR display is still > 0, so it selects Bt2020 (line 55).
        assert_eq!(from_client_config(0, 2, true), sc(Colorspace::Bt2020, false, 10));
    }

    #[test]
    fn from_client_config_bt2020sdr_8bit_falls_back_to_rec709() {
        // Lines 99-102
        assert_eq!(from_client_config(4, 0, false), sc(Colorspace::Rec709, false, 8));
        // Also: full range with an 8-bit BT.2020 SDR request.
        assert_eq!(from_client_config(5, 0, false), sc(Colorspace::Rec709, true, 8));
    }

    #[test]
    fn is_hdr_and_label() {
        assert!(sc(Colorspace::Bt2020, false, 10).is_hdr());
        assert!(!sc(Colorspace::Bt2020Sdr, false, 10).is_hdr());
        assert!(!sc(Colorspace::Rec709, false, 8).is_hdr());
        assert_eq!(Colorspace::Rec601.label(), "SDR (Rec. 601)");
        assert_eq!(Colorspace::Rec709.label(), "SDR (Rec. 709)");
        assert_eq!(Colorspace::Bt2020Sdr.label(), "SDR (Rec. 2020)");
        assert_eq!(Colorspace::Bt2020.label(), "HDR (Rec. 2020 + SMPTE 2084 PQ)");
    }

    // ---- codec_colorspace ----------------------------------------------------

    #[test]
    fn codec_colorspace_all_colorspaces_and_ranges() {
        // Rec. 601: SMPTE170M for all three (lines 114-118)
        let c = codec_colorspace(&sc(Colorspace::Rec601, false, 8));
        assert_eq!(c, CodecColorspace { primaries: 6, transfer: 6, matrix: 6, full_range: false });
        let c = codec_colorspace(&sc(Colorspace::Rec601, true, 8));
        assert_eq!(c, CodecColorspace { primaries: 6, transfer: 6, matrix: 6, full_range: true });

        // Rec. 709: BT709 for all three (lines 124-126)
        let c = codec_colorspace(&sc(Colorspace::Rec709, false, 8));
        assert_eq!(c, CodecColorspace { primaries: 1, transfer: 1, matrix: 1, full_range: false });
        let c = codec_colorspace(&sc(Colorspace::Rec709, true, 8));
        assert_eq!(c, CodecColorspace { primaries: 1, transfer: 1, matrix: 1, full_range: true });

        // Rec. 2020 SDR: BT2020 primaries, BT2020_10 transfer, BT2020_NCL matrix (lines 132-135)
        let c = codec_colorspace(&sc(Colorspace::Bt2020Sdr, false, 10));
        assert_eq!(c, CodecColorspace { primaries: 9, transfer: 14, matrix: 9, full_range: false });
        let c = codec_colorspace(&sc(Colorspace::Bt2020Sdr, true, 10));
        assert_eq!(c, CodecColorspace { primaries: 9, transfer: 14, matrix: 9, full_range: true });

        // HDR: BT2020 primaries, SMPTE2084 transfer, BT2020_NCL matrix (lines 141-144)
        let c = codec_colorspace(&sc(Colorspace::Bt2020, false, 10));
        assert_eq!(c, CodecColorspace { primaries: 9, transfer: 16, matrix: 9, full_range: false });
        let c = codec_colorspace(&sc(Colorspace::Bt2020, true, 10));
        assert_eq!(c, CodecColorspace { primaries: 9, transfer: 16, matrix: 9, full_range: true });
    }

    #[test]
    fn software_format_mappings() {
        assert_eq!(software_format(&sc(Colorspace::Rec601, false, 8)), SoftwareFormat::Smpte170m);
        assert_eq!(software_format(&sc(Colorspace::Rec709, true, 8)), SoftwareFormat::Itu709);
        assert_eq!(software_format(&sc(Colorspace::Bt2020Sdr, false, 10)), SoftwareFormat::Bt2020);
        assert_eq!(software_format(&sc(Colorspace::Bt2020, true, 10)), SoftwareFormat::Bt2020);
    }

    // ---- color_vectors -------------------------------------------------------

    #[test]
    fn color_vectors_rec601_limited_8bit_not_unorm() {
        // Kr = 0.299, Kb = 0.114, Kg = 1 - Kr - Kb.
        // Limited 8-bit: y_mult = 219, y_add = 16, uv_mult = 224, uv_add = 128.
        // Not unorm: y_add += 0.5 and uv_add += 0.5, giving 16.5 and 128.5.
        let kr = 0.299;
        let kb = 0.114;
        let kg = 1.0 - kr - kb;
        let cv = color_vectors(&sc(Colorspace::Rec601, false, 8), false);
        assert_vectors(
            &cv,
            [kr * 219.0, kg * 219.0, kb * 219.0, 16.5],
            [-0.5 * kr / (1.0 - kb) * 224.0, -0.5 * kg / (1.0 - kb) * 224.0, 0.5 * 224.0, 128.5],
            [0.5 * 224.0, -0.5 * kg / (1.0 - kr) * 224.0, -0.5 * kb / (1.0 - kr) * 224.0, 128.5],
        );
        assert_eq!(cv.range_y, [1.0, 0.0]);
        assert_eq!(cv.range_uv, [1.0, 0.0]);
    }

    #[test]
    fn color_vectors_rec709_limited_8bit_unorm() {
        // Kr = 0.2126, Kb = 0.0722, Kg = 1 - Kr - Kb.
        // Limited 8-bit unorm: all four values divided by 255.
        // y_mult = 219/255, y_add = 16/255, uv_mult = 224/255, uv_add = 128/255.
        let kr = 0.2126;
        let kb = 0.0722;
        let kg = 1.0 - kr - kb;
        let cv = color_vectors(&sc(Colorspace::Rec709, false, 8), true);
        let ym = 219.0 / 255.0;
        let um = 224.0 / 255.0;
        assert_vectors(
            &cv,
            [kr * ym, kg * ym, kb * ym, 16.0 / 255.0],
            [-0.5 * kr / (1.0 - kb) * um, -0.5 * kg / (1.0 - kb) * um, 0.5 * um, 128.0 / 255.0],
            [0.5 * um, -0.5 * kg / (1.0 - kr) * um, -0.5 * kb / (1.0 - kr) * um, 128.0 / 255.0],
        );
    }

    #[test]
    fn color_vectors_rec709_full_8bit_unorm() {
        // Full 8-bit unorm: y_mult = 255/255 = 1, y_add = 0,
        // uv_mult = 255/255 = 1, uv_add = 128/255.
        let kr = 0.2126;
        let kb = 0.0722;
        let kg = 1.0 - kr - kb;
        let cv = color_vectors(&sc(Colorspace::Rec709, true, 8), true);
        assert_vectors(
            &cv,
            [kr, kg, kb, 0.0],
            [-0.5 * kr / (1.0 - kb), -0.5 * kg / (1.0 - kb), 0.5, 128.0 / 255.0],
            [0.5, -0.5 * kg / (1.0 - kr), -0.5 * kb / (1.0 - kr), 128.0 / 255.0],
        );
    }

    #[test]
    fn color_vectors_bt2020_limited_10bit_unorm() {
        // Kr = 0.2627, Kb = 0.0593, Kg = 1 - Kr - Kb.
        // Limited 10-bit unorm, divided by 1023:
        // y_mult = 4 * 219 = 876, y_add = 4 * 16 = 64,
        // uv_mult = 4 * 224 = 896, uv_add = 4 * 128 = 512.
        let kr = 0.2627;
        let kb = 0.0593;
        let kg = 1.0 - kr - kb;
        let cv = color_vectors(&sc(Colorspace::Bt2020, false, 10), true);
        let ym = 876.0 / 1023.0;
        let um = 896.0 / 1023.0;
        assert_vectors(
            &cv,
            [kr * ym, kg * ym, kb * ym, 64.0 / 1023.0],
            [-0.5 * kr / (1.0 - kb) * um, -0.5 * kg / (1.0 - kb) * um, 0.5 * um, 512.0 / 1023.0],
            [0.5 * um, -0.5 * kg / (1.0 - kr) * um, -0.5 * kb / (1.0 - kr) * um, 512.0 / 1023.0],
        );
    }

    #[test]
    fn color_vectors_round_trip_rec709_limited_8bit_unorm() {
        let kr = 0.2126;
        let kb = 0.0722;
        let cv = color_vectors(&sc(Colorspace::Rec709, false, 8), true);

        // Helpers: Y = Y_row . rgb + Y_add; U = U_row . rgb + U_add; V = V_row . rgb + V_add
        let apply = |row: &[f32; 4], rgb: [f64; 3]| -> f64 {
            row[0] as f64 * rgb[0] + row[1] as f64 * rgb[1] + row[2] as f64 * rgb[2] + row[3] as f64
        };

        // White (1,1,1): Y = 235/255, U = V = 128/255
        let white = [1.0, 1.0, 1.0];
        assert_close(
            apply(&cv.color_vec_y, white) as f32,
            235.0 / 255.0,
            "white Y",
        );
        assert_close(
            apply(&cv.color_vec_u, white) as f32,
            128.0 / 255.0,
            "white U",
        );
        assert_close(
            apply(&cv.color_vec_v, white) as f32,
            128.0 / 255.0,
            "white V",
        );

        // Black (0,0,0): Y = 16/255, U = V = 128/255
        let black = [0.0, 0.0, 0.0];
        assert_close(apply(&cv.color_vec_y, black) as f32, 16.0 / 255.0, "black Y");
        assert_close(apply(&cv.color_vec_u, black) as f32, 128.0 / 255.0, "black U");
        assert_close(apply(&cv.color_vec_v, black) as f32, 128.0 / 255.0, "black V");

        // Pure red (1,0,0): Y = (Kr*219 + 16)/255, U = (-0.5*Kr/(1-Kb)*224 + 128)/255,
        // V = (0.5*224 + 128)/255 = 240/255
        let red = [1.0, 0.0, 0.0];
        assert_close(
            apply(&cv.color_vec_y, red) as f32,
            (kr * 219.0 + 16.0) / 255.0,
            "red Y",
        );
        assert_close(
            apply(&cv.color_vec_u, red) as f32,
            (-0.5 * kr / (1.0 - kb) * 224.0 + 128.0) / 255.0,
            "red U",
        );
        assert_close(apply(&cv.color_vec_v, red) as f32, 240.0 / 255.0, "red V");
    }

    #[test]
    fn color_vectors_selection_rules() {
        // Bit depths other than 10 select the 8-bit entry (line 282).
        let a = color_vectors(&sc(Colorspace::Rec709, false, 12), true);
        let b = color_vectors(&sc(Colorspace::Rec709, false, 8), true);
        assert_eq!(a, b);
        // Bt2020Sdr uses the BT.2020 matrix constants (line 170-171).
        let a = color_vectors(&sc(Colorspace::Bt2020Sdr, false, 10), false);
        let b = color_vectors(&sc(Colorspace::Bt2020, false, 10), false);
        assert_eq!(a, b);
    }
}
