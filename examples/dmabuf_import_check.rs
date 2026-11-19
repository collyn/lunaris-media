//! Checks, without root, that the encoders can consume the scan-out buffers
//! the capture service will hand them on this machine.
//!
//! It reads (unprivileged, read-only) which format and modifier the
//! compositor scans out, allocates an own buffer with exactly that layout on
//! the same GPU's render node (GBM), fills it with a known picture, and then
//!   1. imports it with EGL and reads it back (the readback path), and
//!   2. encodes it with the real VAAPI encoder through the zero-copy import
//!      (or, if VA-API refuses the buffer, the GPU copy into a VA surface)
//!      (`GpuBuffer::DmaBuf` with its auxiliary planes), decodes the result
//!      and checks the colours.
//!
//! Touches nothing on screen: render node only, its own buffers only.
//!
//! ```text
//! cargo run --example dmabuf_import_check [-- --format AB30 --modifier 0x... --card card2]
//! ```

use std::ffi::c_void;
use std::path::Path;
use std::sync::Mutex;

use lunaris_media::capture::drm_service::kms::{self, Card};
use lunaris_media::capture::gpu_buffer::{DmaBufAuxPlanes, GpuBuffer};
use lunaris_media::encode::EncoderConfig;
use lunaris_media::types::{HwAccelType, VideoCodec};

static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Recorder;
impl log::Log for Recorder {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Debug && m.target().starts_with("lunaris_media")
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            let line = format!("{} {}", r.level(), r.args());
            if r.level() <= log::Level::Info {
                eprintln!("  [log] {}", line);
            }
            LOG.lock().unwrap().push(line);
        }
    }
    fn flush(&self) {}
}

type GbmCreateDevice = unsafe extern "C" fn(i32) -> *mut c_void;
type GbmDeviceDestroy = unsafe extern "C" fn(*mut c_void);
type GbmBoCreateWithModifiers =
    unsafe extern "C" fn(*mut c_void, u32, u32, u32, *const u64, u32) -> *mut c_void;
type GbmBoGetU32 = unsafe extern "C" fn(*mut c_void) -> u32;
type GbmBoGetI32 = unsafe extern "C" fn(*mut c_void) -> i32;
type GbmBoGetU64 = unsafe extern "C" fn(*mut c_void) -> u64;
type GbmBoPlaneU32 = unsafe extern "C" fn(*mut c_void, i32) -> u32;
type GbmBoMap = unsafe extern "C" fn(*mut c_void, u32, u32, u32, u32, u32, *mut u32, *mut *mut c_void) -> *mut c_void;
type GbmBoUnmap = unsafe extern "C" fn(*mut c_void, *mut c_void);
type GbmBoDestroy = unsafe extern "C" fn(*mut c_void);

const GBM_BO_TRANSFER_WRITE: u32 = 2;

fn fourcc(s: &str) -> u32 {
    let b = s.as_bytes();
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// One pixel of the test picture in `format` (top half red, bottom half grey).
fn pixel(format: u32, red: bool) -> u32 {
    let (r, g, b) = if red { (1.0, 0.0, 0.0) } else { (0.5, 0.5, 0.5) };
    match &format.to_le_bytes() {
        b"AB30" | b"XB30" => {
            let q = |v: f64| (v * 1023.0).round() as u32;
            q(r) | (q(g) << 10) | (q(b) << 20) | (3 << 30)
        }
        b"AR30" | b"XR30" => {
            let q = |v: f64| (v * 1023.0).round() as u32;
            q(b) | (q(g) << 10) | (q(r) << 20) | (3 << 30)
        }
        b"AB24" | b"XB24" => {
            let q = |v: f64| (v * 255.0).round() as u32;
            q(r) | (q(g) << 8) | (q(b) << 16) | (255 << 24)
        }
        _ => {
            let q = |v: f64| (v * 255.0).round() as u32;
            q(b) | (q(g) << 8) | (q(r) << 16) | (255 << 24)
        }
    }
}

fn scanout_layout(card_filter: Option<&str>) -> Option<(String, u32, u64)> {
    for path in kms::card_paths(Path::new("/dev/dri")) {
        let Ok(card) = Card::open(&path) else { continue };
        if card_filter.is_some_and(|c| c != card.name) {
            continue;
        }
        for d in kms::card_displays(&card).unwrap_or_default() {
            let crtc = card.crtc(d.crtc_id).ok()?;
            if let Ok(l) = card.fb_layout(crtc.fb_id) {
                println!(
                    "scan-out of {} on {}: {} modifier {:#018x} {} plane(s)",
                    d.name,
                    card.name,
                    kms::fourcc_str(l.fourcc),
                    l.modifier,
                    l.planes.len()
                );
                return Some((card.name.clone(), l.fourcc, l.modifier));
            }
        }
    }
    None
}

fn main() {
    log::set_logger(&Recorder).unwrap();
    log::set_max_level(log::LevelFilter::Debug);
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());

    let detected = scanout_layout(arg("--card").as_deref());
    let card = arg("--card").or_else(|| detected.as_ref().map(|d| d.0.clone())).unwrap_or_else(|| "card0".into());
    let format = arg("--format").map(|f| fourcc(&f)).or(detected.as_ref().map(|d| d.1)).unwrap_or(fourcc("XR24"));
    let modifier = arg("--modifier")
        .and_then(|m| u64::from_str_radix(m.trim_start_matches("0x"), 16).ok())
        .or(detected.as_ref().map(|d| d.2))
        .unwrap_or(0);
    let node = kms::render_node_for_card(Path::new("/sys"), &card).expect("render node");
    println!("testing {} modifier {:#018x} on {} ({})", kms::fourcc_str(format), modifier, card, node);
    let (w, h) = (1920u32, 1080u32);

    // --- Allocate and fill a buffer with that layout (GBM) ---
    let gbm = unsafe { libloading::Library::new("libgbm.so.1") }.expect("libgbm.so.1");
    let node_c = std::ffi::CString::new(node.clone()).unwrap();
    let node_fd = unsafe { libc::open(node_c.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
    assert!(node_fd >= 0, "open {}", node);
    let mut ok = true;
    unsafe {
        let create_device: libloading::Symbol<GbmCreateDevice> = gbm.get(b"gbm_create_device\0").unwrap();
        let device_destroy: libloading::Symbol<GbmDeviceDestroy> = gbm.get(b"gbm_device_destroy\0").unwrap();
        let bo_create: libloading::Symbol<GbmBoCreateWithModifiers> =
            gbm.get(b"gbm_bo_create_with_modifiers\0").unwrap();
        let plane_count: libloading::Symbol<GbmBoGetI32> = gbm.get(b"gbm_bo_get_plane_count\0").unwrap();
        let get_fd: libloading::Symbol<GbmBoGetI32> = gbm.get(b"gbm_bo_get_fd\0").unwrap();
        let get_modifier: libloading::Symbol<GbmBoGetU64> = gbm.get(b"gbm_bo_get_modifier\0").unwrap();
        let get_format: libloading::Symbol<GbmBoGetU32> = gbm.get(b"gbm_bo_get_format\0").unwrap();
        let get_offset: libloading::Symbol<GbmBoPlaneU32> = gbm.get(b"gbm_bo_get_offset\0").unwrap();
        let get_stride: libloading::Symbol<GbmBoPlaneU32> = gbm.get(b"gbm_bo_get_stride_for_plane\0").unwrap();
        let bo_map: libloading::Symbol<GbmBoMap> = gbm.get(b"gbm_bo_map\0").unwrap();
        let bo_unmap: libloading::Symbol<GbmBoUnmap> = gbm.get(b"gbm_bo_unmap\0").unwrap();
        let bo_destroy: libloading::Symbol<GbmBoDestroy> = gbm.get(b"gbm_bo_destroy\0").unwrap();

        let dev = create_device(node_fd);
        assert!(!dev.is_null(), "gbm_create_device");
        let mods = [modifier];
        let bo = bo_create(dev, w, h, format, mods.as_ptr(), 1);
        if bo.is_null() {
            println!("FAIL: GBM cannot allocate {} with modifier {:#x}", kms::fourcc_str(format), modifier);
            std::process::exit(1);
        }
        let planes = plane_count(bo);
        let layout: Vec<(u32, u32)> = (0..planes).map(|i| (get_offset(bo, i), get_stride(bo, i))).collect();
        println!(
            "allocated {} modifier {:#018x}, planes {:?}",
            kms::fourcc_str(get_format(bo)),
            get_modifier(bo),
            layout
        );

        let mut stride = 0u32;
        let mut map_data: *mut c_void = std::ptr::null_mut();
        let ptr = bo_map(bo, 0, 0, w, h, GBM_BO_TRANSFER_WRITE, &mut stride, &mut map_data);
        assert!(!ptr.is_null(), "gbm_bo_map");
        for y in 0..h {
            let row = (ptr as *mut u8).add((y * stride) as usize) as *mut u32;
            let px = pixel(format, y < h / 2);
            for x in 0..w {
                *row.add(x as usize) = px;
            }
        }
        bo_unmap(bo, map_data);
        let fd = get_fd(bo);
        assert!(fd >= 0, "gbm_bo_get_fd");
        let aux = DmaBufAuxPlanes::from_slice(&layout[1..]).expect("at most 4 planes");

        // --- 1. EGL import + readback ---
        lunaris_media::capture::dmabuf_caps::set_capture_render_node(Some(node.clone()));
        match lunaris_media::capture::egl_import::EglImporter::new() {
            Ok(mut egl) => match egl.import_dmabuf_planes(fd, w, h, layout[0].1, format, layout[0].0, get_modifier(bo), aux) {
                Ok(bgra) => {
                    let at = |x: u32, y: u32| {
                        let i = ((y * w + x) * 4) as usize;
                        (bgra[i + 2], bgra[i + 1], bgra[i])
                    };
                    let (top, bottom) = (at(100, 100), at(100, h - 100));
                    let good = top.0 > 240 && top.1 < 16 && top.2 < 16 && (120..=136).contains(&bottom.0);
                    println!("EGL readback: top RGB {:?} bottom RGB {:?} -> {}", top, bottom, if good { "OK" } else { "WRONG" });
                    ok &= good;
                }
                Err(e) => {
                    println!("FAIL EGL import: {}", e);
                    ok = false;
                }
            },
            Err(e) => {
                println!("FAIL EGL init: {}", e);
                ok = false;
            }
        }

        // --- 2. VAAPI zero-copy encode, decode, compare ---
        let mut encoder = lunaris_media::encode::create_encoder().expect("encoder");
        let config = EncoderConfig {
            codec: VideoCodec::H264,
            width: w,
            height: h,
            fps: 60,
            bitrate_kbps: 20_000,
            preferred_hw: Some(HwAccelType::Vaapi),
            hardware_only: true,
            ..EncoderConfig::default()
        };
        match encoder.initialize(&config) {
            Err(e) => {
                println!("FAIL VAAPI encoder: {}", e);
                ok = false;
            }
            Ok(()) => {
                println!("encoder: {} ({:?})", encoder.encoder_info().name, encoder.encoder_info().hw_type);
                let mut bitstream = Vec::new();
                let mut timed = std::time::Duration::ZERO;
                const FRAMES: u64 = 60;
                for i in 0..FRAMES {
                    let dup = libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0);
                    let buffer = GpuBuffer::DmaBuf {
                        fd: dup,
                        offset: layout[0].0,
                        stride: layout[0].1,
                        modifier: get_modifier(bo),
                        size: 0,
                        width: w,
                        height: h,
                        fourcc: format,
                        aux_planes: aux,
                    };
                    if i == 0 {
                        encoder.request_keyframe();
                    }
                    let started = std::time::Instant::now();
                    let encoded = encoder.encode(&buffer, i * 16_666);
                    if i > 0 {
                        timed += started.elapsed();
                    }
                    match encoded {
                        Ok(frames) => frames.into_iter().for_each(|f| bitstream.extend_from_slice(&f.data)),
                        Err(e) => {
                            println!("FAIL encode: {}", e);
                            ok = false;
                            break;
                        }
                    }
                }
                println!(
                    "encode time per frame (after the first): {:.2} ms",
                    timed.as_secs_f64() * 1000.0 / (FRAMES - 1) as f64
                );
                let log = LOG.lock().unwrap().clone();
                let zero_copy = log.iter().any(|l| l.contains("VAAPI zero-copy: imported DMA-BUF"));
                let refused = log.iter().any(|l| {
                    l.contains("VAAPI cannot import the captured buffer") || l.contains("wrongly; copying captured buffers")
                });
                let gpu_copy = log.iter().any(|l| l.contains("VAAPI: staging surface"));
                let readback = log.iter().any(|l| l.contains("falling back to EGL readback"));
                let path = if readback {
                    "FAILED (CPU readback)"
                } else if zero_copy && !refused {
                    "OK (direct import)"
                } else if refused && gpu_copy {
                    "OK (direct import refused, GPU copy into a VA surface)"
                } else {
                    "not seen"
                };
                println!("VAAPI path: {}", path);
                ok &= !readback && (zero_copy || gpu_copy);
                match decode_check(&bitstream, w, h) {
                    Ok(msg) => println!("decoded picture: {}", msg),
                    Err(msg) => {
                        println!("FAIL decoded picture: {}", msg);
                        ok = false;
                    }
                }
            }
        }
        libc::close(fd);
        bo_destroy(bo);
        device_destroy(dev);
        libc::close(node_fd);
    }
    println!("{}", if ok { "RESULT: OK" } else { "RESULT: FAILED" });
    std::process::exit(if ok { 0 } else { 1 });
}

/// Decode the H.264 stream and compare a top (red) and bottom (grey) sample.
fn decode_check(bitstream: &[u8], w: u32, h: u32) -> Result<String, String> {
    use ffmpeg_next as ff;
    ff::init().map_err(|e| e.to_string())?;
    let codec = ff::decoder::find(ff::codec::Id::H264).ok_or("no h264 decoder")?;
    let mut dec = ff::codec::context::Context::new_with_codec(codec)
        .decoder()
        .video()
        .map_err(|e| e.to_string())?;
    dec.send_packet(&ff::Packet::copy(bitstream)).map_err(|e| e.to_string())?;
    let _ = dec.send_eof();
    let mut frame = ff::frame::Video::empty();
    let mut last = None;
    while dec.receive_frame(&mut frame).is_ok() {
        last = Some(frame.clone());
    }
    let f = last.ok_or("no frame decoded")?;
    if f.width() != w || f.height() != h {
        return Err(format!("decoded {}x{}", f.width(), f.height()));
    }
    let y = |x: u32, yy: u32| f.data(0)[(yy as usize) * f.stride(0) + x as usize];
    let c = |p: usize, x: u32, yy: u32| f.data(p)[(yy as usize / 2) * f.stride(p) + x as usize / 2];
    let (ty, tu, tv) = (y(100, 100), c(1, 100, 100), c(2, 100, 100));
    let (by, bu, bv) = (y(100, h - 100), c(1, 100, h - 100), c(2, 100, h - 100));
    let msg = format!("top Y/Cb/Cr {}/{}/{} (red), bottom {}/{}/{} (grey)", ty, tu, tv, by, bu, bv);
    let red_ok = tv > 200 && tu < 128 && (50..=100).contains(&ty);
    let grey_ok = (110..=140).contains(&by) && (120..=136).contains(&bu) && (120..=136).contains(&bv);
    if red_ok && grey_ok {
        Ok(format!("{} -> OK", msg))
    } else {
        Err(msg)
    }
}
