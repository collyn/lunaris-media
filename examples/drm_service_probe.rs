//! Probe for the Lunaris capture service (unattended KMS capture).
//!
//! ```text
//! drm_service_probe enumerate
//!     Read-only, no root needed: cards, render nodes, lit displays, the
//!     framebuffer layout each CRTC scans out, cursor planes, logind's
//!     active session. Uses only "get" ioctls; sets no mode.
//!
//! drm_service_probe serve --socket PATH [--allow-exe PATH]... [--allow-appimage PATH]...
//!     Run the capture service in the foreground (needs root to export
//!     framebuffers). Same code as `lunaris-agent --capture-service`.
//!
//! drm_service_probe client --socket PATH [--frames N] [--fps F] [--out DIR]
//!     Connect like the agent does, list displays, grab N frames of each,
//!     report fps/latency/cursor/descriptor counts, save one PNG per display
//!     (EGL readback on the display's GPU), then run the real
//!     DrmServiceCapture backend for a few frames.
//! ```
//!
//! drm_service_probe soak --socket PATH [--display NAME] [--seconds S] [--fps F]
//!     The agent's path for S seconds: DrmServiceCapture, then VAAPI H.264
//!     encoding of every new frame (zero-copy). Every 5 s: frame rate, frame
//!     gaps, encode time, bitrate, cursor updates, descriptors and memory.
//!     Survives a service restart (reconnects) and reports the gap.
//!
//! Reads the screen; never injects input and never changes display state.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use lunaris_media::capture::drm_service::client::{captured_frame, ServiceClient};
use lunaris_media::capture::drm_service::kms::{self, Card};
use lunaris_media::capture::drm_service::protocol::{Message, StatusCode, DISPLAY_ACTIVE};
use lunaris_media::capture::drm_service::transport::open_fd_count;
use lunaris_media::capture::drm_service::{self as svc, auth};
use lunaris_media::capture::gpu_buffer::GpuBuffer;
use lunaris_media::capture::ScreenCapture;

fn arg_values(args: &[String], name: &str) -> Vec<String> {
    args.windows(2).filter(|w| w[0] == name).map(|w| w[1].clone()).collect()
}

fn arg_value(args: &[String], name: &str) -> Option<String> {
    arg_values(args, name).into_iter().next()
}

fn monotonic_us() -> u64 {
    let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000 + ts.tv_nsec as u64 / 1000
}

fn enumerate() {
    println!("euid {} ; logind active graphical uid on seat0: {:?}", unsafe { libc::geteuid() }, auth::query_active_uid());
    let own = std::fs::read_link("/proc/self/exe").unwrap_or_default();
    println!("this binary {} trusted for the allow-list: {:?}", own.display(), auth::trusted_file(&own).map(|_| "yes"));
    for path in kms::card_paths(Path::new("/dev/dri")) {
        let card = match Card::open(&path) {
            Ok(c) => c,
            Err(e) => {
                println!("{}: cannot open: {}", path.display(), e);
                continue;
            }
        };
        println!(
            "{} (render node {:?}, atomic properties: {})",
            card.name,
            kms::render_node_for_card(Path::new("/sys"), &card.name),
            card.atomic
        );
        let displays = match kms::card_displays(&card) {
            Ok(d) => d,
            Err(e) => {
                println!("  resources: {}", e);
                continue;
            }
        };
        if displays.is_empty() {
            println!("  no lit display");
        }
        for d in displays {
            println!(
                "  display {:#x}: {} connector {} crtc {} {}x{} @{:.3}Hz {}",
                d.id,
                d.name,
                d.connector_id,
                d.crtc_id,
                d.width,
                d.height,
                d.refresh_mhz as f64 / 1000.0,
                if d.flags & DISPLAY_ACTIVE != 0 { "active" } else { "OFF/blanked" }
            );
            if let Ok(crtc) = card.crtc(d.crtc_id) {
                match card.fb_layout(crtc.fb_id) {
                    Ok(l) => println!(
                        "    fb {} {}x{} {} modifier {:#018x} planes {:?} exportable here: {}",
                        crtc.fb_id,
                        l.width,
                        l.height,
                        kms::fourcc_str(l.fourcc),
                        l.modifier,
                        l.planes,
                        l.exportable
                    ),
                    Err(e) => println!("    fb {}: GETFB2: {}", crtc.fb_id, e),
                }
                println!("    crtc offset {},{} ACTIVE={:?}", crtc.x, crtc.y, card.crtc_active(d.crtc_id));
            }
            match card.cursor_on(d.crtc_id) {
                Some(c) => {
                    let layout = card.fb_layout(c.fb_id).ok();
                    println!(
                        "    cursor plane {} at {},{} hotspot {},{} fb {} {}",
                        c.plane_id,
                        c.x,
                        c.y,
                        c.hot_x,
                        c.hot_y,
                        c.fb_id,
                        layout
                            .map(|l| format!("{}x{} {} modifier {:#x}", l.width, l.height, kms::fourcc_str(l.fourcc), l.modifier))
                            .unwrap_or_default()
                    );
                }
                None => println!("    no hardware cursor on this CRTC (software cursor or pointer elsewhere)"),
            }
        }
    }
}

fn serve(args: &[String]) {
    let cfg = svc::ServiceConfig {
        socket_path: PathBuf::from(arg_value(args, "--socket").unwrap_or_else(|| svc::DEFAULT_SOCKET_PATH.into())),
        allow_exes: arg_values(args, "--allow-exe").into_iter().map(PathBuf::from).collect(),
        allow_appimages: arg_values(args, "--allow-appimage").into_iter().map(PathBuf::from).collect(),
        max_clients: 4,
    };
    if let Err(e) = svc::serve(cfg) {
        eprintln!("serve: {}", e);
        std::process::exit(1);
    }
}

fn save_png(frame: &lunaris_media::capture::CapturedFrame, card: &str, out: &Path, name: &str) -> Result<PathBuf, String> {
    let GpuBuffer::DmaBuf { fd, offset, stride, modifier, width, height, fourcc, aux_planes, .. } = &frame.buffer else {
        return Err("not a DMA-BUF".into());
    };
    let node = kms::render_node_for_card(Path::new("/sys"), card).ok_or("no render node")?;
    lunaris_media::capture::dmabuf_caps::set_capture_render_node(Some(node));
    let mut egl = lunaris_media::capture::egl_import::EglImporter::new().map_err(|e| e.to_string())?;
    let bgra = egl
        .import_dmabuf_planes(*fd, *width, *height, *stride, *fourcc, *offset, *modifier, *aux_planes)
        .map_err(|e| e.to_string())?;
    let mut rgba = bgra;
    for px in rgba.chunks_exact_mut(4) {
        px.swap(0, 2);
        px[3] = 255;
    }
    let img = image::RgbaImage::from_raw(*width, *height, rgba).ok_or("bad size")?;
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let path = out.join(format!("{}.png", name));
    img.save(&path).map_err(|e| e.to_string())?;
    Ok(path)
}

fn client(args: &[String]) {
    let socket = PathBuf::from(arg_value(args, "--socket").unwrap_or_else(|| svc::DEFAULT_SOCKET_PATH.into()));
    let frames: usize = arg_value(args, "--frames").and_then(|v| v.parse().ok()).unwrap_or(120);
    let fps: u32 = arg_value(args, "--fps").and_then(|v| v.parse().ok()).unwrap_or(60);
    let out = PathBuf::from(arg_value(args, "--out").unwrap_or_else(|| ".".into()));
    let fds_start = open_fd_count();
    println!("open descriptors at start: {}", fds_start);

    let client = match ServiceClient::connect(&socket, Duration::from_secs(3)) {
        Ok(c) => c,
        Err(e) => {
            println!("FAIL connect: {}", e);
            std::process::exit(2);
        }
    };
    println!("connected; {} display(s)", client.displays.len());
    let conn = client.conn.clone();
    for d in &client.displays {
        println!(
            "== {} ({}, crtc {}) {}x{} @{:.2}Hz flags {}",
            d.name,
            d.card,
            d.crtc_id,
            d.width,
            d.height,
            d.refresh_mhz as f64 / 1000.0,
            d.flags
        );
        conn.send(&Message::Start { display_id: d.id, max_fps: fps, max_in_flight: 2 }, &[]).unwrap();
        let t0 = Instant::now();
        let (mut got, mut latency_sum, mut latency_max) = (0usize, 0u64, 0u64);
        let mut saved = false;
        let mut cursor_msgs = 0;
        let mut buffers = std::collections::HashSet::new();
        let deadline = Instant::now() + Duration::from_secs(5 + frames as u64 / fps.max(1) as u64 * 2);
        while got < frames && Instant::now() < deadline {
            let Ok(Some((msg, fds))) = conn.recv(Some(Duration::from_millis(500))) else { continue };
            match msg {
                Message::Frame(f) => {
                    let lat = monotonic_us().saturating_sub(f.timestamp_us);
                    latency_sum += lat;
                    latency_max = latency_max.max(lat);
                    buffers.insert(f.buffer_id);
                    if got == 0 {
                        println!(
                            "   first frame: {}x{} {} modifier {:#018x} planes {:?} crop {},{} {}x{} fds {}",
                            f.width,
                            f.height,
                            kms::fourcc_str(f.fourcc),
                            f.modifier,
                            f.planes,
                            f.crop_x,
                            f.crop_y,
                            f.crop_w,
                            f.crop_h,
                            fds.len()
                        );
                    }
                    let seq = f.seq;
                    if !saved && got >= 2 {
                        saved = true;
                        match captured_frame(&f, fds) {
                            Ok(frame) => match save_png(&frame, &d.card, &out, &format!("display-{}", d.name)) {
                                Ok(p) => println!("   saved {}", p.display()),
                                Err(e) => println!("   PNG readback failed: {}", e),
                            },
                            Err(e) => println!("   frame not importable by the encoders: {:?}", e),
                        }
                    } else {
                        drop(fds);
                    }
                    conn.send(&Message::FrameAck { seq }, &[]).unwrap();
                    got += 1;
                }
                Message::Cursor(c) => {
                    cursor_msgs += 1;
                    if cursor_msgs <= 3 || c.image.is_some() {
                        println!(
                            "   cursor: visible {} at {},{} hotspot {},{} {}x{} gen {} image {}",
                            c.visible,
                            c.x,
                            c.y,
                            c.hot_x,
                            c.hot_y,
                            c.width,
                            c.height,
                            c.generation,
                            c.image.is_some()
                        );
                    }
                }
                Message::Status { code, .. } => {
                    println!("   status: {:?}", code);
                    if code == StatusCode::NoFramebuffer {
                        println!("   (screen off/blanked: no frames until it is on again)");
                    }
                }
                other => println!("   {:?}", other.kind()),
            }
        }
        conn.send(&Message::Stop, &[]).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "   {} frames in {:.2}s = {:.1} fps; service→client latency avg {} us max {} us; {} distinct buffers; {} cursor messages",
            got,
            secs,
            got as f64 / secs,
            if got > 0 { latency_sum / got as u64 } else { 0 },
            latency_max,
            buffers.len(),
            cursor_msgs
        );
        // Let in-flight packets settle, then drain them (closing their fds).
        while let Ok(Some(_)) = conn.recv(Some(Duration::from_millis(200))) {}
    }
    drop(conn);
    drop(client);
    println!("open descriptors after the raw protocol test: {} (start {})", open_fd_count(), fds_start);

    // The real backend, as the agent uses it.
    std::env::set_var("LUNARIS_CAPTURE_SOCKET", &socket);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut cap = match svc::DrmServiceCapture::connect(&socket) {
            Ok(c) => c,
            Err(e) => {
                println!("FAIL DrmServiceCapture: {}", e);
                return;
            }
        };
        let displays = cap.list_displays().await.unwrap_or_default();
        for d in &displays {
            println!("backend display {} ({}) {}x{} primary {}", d.id, d.name, d.width, d.height, d.is_primary);
        }
        let config = lunaris_media::types::StreamConfig { width: 1920, height: 1080, fps, ..Default::default() };
        if let Err(e) = cap.start("default", &config).await {
            println!("FAIL start: {}", e);
            return;
        }
        let (mut new, t0) = (0, Instant::now());
        while new < 30 && t0.elapsed() < Duration::from_secs(5) {
            match cap.next_frame().await {
                Ok(f) if f.is_new_frame => new += 1,
                Ok(_) => {}
                Err(e) => {
                    println!("FAIL next_frame: {}", e);
                    break;
                }
            }
        }
        println!("backend: {} new frames in {:.2}s", new, t0.elapsed().as_secs_f64());
        if let Some(c) = lunaris_media::capture::linux_wayland::pipewire_cursor() {
            println!("backend cursor slot: {},{} visible {} image {}", c.x, c.y, c.visible, c.image.is_some());
        }
        let _ = cap.stop().await;
    });
    drop(rt);
    println!("open descriptors at the end: {} (start {})", open_fd_count(), fds_start);
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("enumerate") => enumerate(),
        Some("serve") => serve(&args),
        Some("client") => client(&args),
        Some("soak") => soak(&args),
        _ => {
            eprintln!("usage: drm_service_probe enumerate | serve --socket PATH [--allow-exe P]... | client --socket PATH [--frames N] [--fps F] [--out DIR] | soak --socket PATH [--display NAME] [--seconds S] [--fps F]");
            std::process::exit(64);
        }
    }
}

fn vm_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)?.parse().ok()))
        .unwrap_or(0)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn soak(args: &[String]) {
    use lunaris_media::encode::EncoderConfig;
    use lunaris_media::types::{HwAccelType, VideoCodec};

    let socket = PathBuf::from(arg_value(args, "--socket").expect("--socket"));
    let seconds: u64 = arg_value(args, "--seconds").and_then(|v| v.parse().ok()).unwrap_or(30);
    let fps: u32 = arg_value(args, "--fps").and_then(|v| v.parse().ok()).unwrap_or(60);
    let want = arg_value(args, "--display");
    std::env::set_var("LUNARIS_CAPTURE_SOCKET", &socket);
    let fds_start = open_fd_count();
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut cap = match svc::DrmServiceCapture::connect(&socket) {
            Ok(c) => c,
            Err(e) => {
                println!("FAIL connect: {}", e);
                std::process::exit(1);
            }
        };
        let displays = cap.list_displays().await.unwrap_or_default();
        let Some(d) = displays
            .iter()
            .find(|d| want.as_deref().is_none_or(|w| d.name == w))
            .or_else(|| displays.iter().find(|d| d.is_primary))
            .cloned()
        else {
            println!("FAIL no display {:?}", want);
            std::process::exit(1);
        };
        let config = lunaris_media::types::StreamConfig { width: d.width, height: d.height, fps, ..Default::default() };
        if let Err(e) = cap.start(&d.id, &config).await {
            println!("FAIL start: {}", e);
            std::process::exit(1);
        }
        let mut encoder = lunaris_media::encode::create_encoder().expect("encoder");
        let enc_config = EncoderConfig {
            codec: VideoCodec::H264,
            width: d.width,
            height: d.height,
            fps,
            bitrate_kbps: 20_000,
            low_latency: true,
            preferred_hw: Some(HwAccelType::Vaapi),
            hardware_only: true,
            ..EncoderConfig::default()
        };
        if let Err(e) = encoder.initialize(&enc_config) {
            println!("FAIL VAAPI encoder: {}", e);
            std::process::exit(1);
        }
        println!(
            "soak {} {}x{} at {} fps for {} s, encoder {} ({:?})",
            d.name,
            d.width,
            d.height,
            fps,
            seconds,
            encoder.encoder_info().name,
            encoder.encoder_info().hw_type
        );
        let rss_start = vm_rss_kb();
        let t0 = Instant::now();
        let mut window_start = Instant::now();
        let (mut w_new, mut w_repeat, mut w_bytes, mut w_cursor) = (0u32, 0u32, 0usize, 0u32);
        let (mut w_gaps, mut w_enc): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let (mut all_gaps, mut total_new, mut total_bytes, mut enc_errors) = (Vec::new(), 0u32, 0usize, 0u32);
        let mut last_new: Option<Instant> = None;
        let mut last_cursor: Option<(i32, i32, bool, bool)> = None;
        let mut pts = 0u64;
        encoder.request_keyframe();
        while t0.elapsed() < Duration::from_secs(seconds) {
            let frame = match cap.next_frame().await {
                Ok(f) => f,
                Err(e) => {
                    println!("FAIL next_frame: {}", e);
                    break;
                }
            };
            if let Some(c) = lunaris_media::capture::linux_wayland::pipewire_cursor() {
                let key = (c.x, c.y, c.visible, c.image.is_some());
                if last_cursor != Some(key) {
                    w_cursor += 1;
                    last_cursor = Some(key);
                }
            }
            if !frame.is_new_frame {
                w_repeat += 1;
            } else {
                let now = Instant::now();
                if let Some(prev) = last_new {
                    let gap = now.duration_since(prev).as_secs_f64() * 1000.0;
                    w_gaps.push(gap);
                    all_gaps.push(gap);
                }
                last_new = Some(now);
                pts += 1_000_000 / fps as u64;
                let e0 = Instant::now();
                match encoder.encode(&frame.buffer, pts) {
                    Ok(out) => {
                        let b: usize = out.iter().map(|f| f.data.len()).sum();
                        w_bytes += b;
                        total_bytes += b;
                    }
                    Err(e) => {
                        enc_errors += 1;
                        if enc_errors <= 3 {
                            println!("   encode error: {}", e);
                        }
                    }
                }
                w_enc.push(e0.elapsed().as_secs_f64() * 1000.0);
                w_new += 1;
                total_new += 1;
            }
            drop(frame);
            if window_start.elapsed() >= Duration::from_secs(5) {
                let secs = window_start.elapsed().as_secs_f64();
                w_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
                w_enc.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!(
                    "   t={:>3.0}s  {:>5.1} fps  repeats {:>3}  gap p50 {:>5.1} p99 {:>6.1} max {:>7.1} ms  encode avg {:>5.2} max {:>6.2} ms  {:>6.0} kbit/s  cursor updates {:>3}  fds {}  rss {} MB",
                    t0.elapsed().as_secs_f64(),
                    w_new as f64 / secs,
                    w_repeat,
                    percentile(&w_gaps, 0.5),
                    percentile(&w_gaps, 0.99),
                    w_gaps.last().copied().unwrap_or(0.0),
                    if w_enc.is_empty() { 0.0 } else { w_enc.iter().sum::<f64>() / w_enc.len() as f64 },
                    w_enc.last().copied().unwrap_or(0.0),
                    w_bytes as f64 * 8.0 / 1000.0 / secs,
                    w_cursor,
                    open_fd_count(),
                    vm_rss_kb() / 1024
                );
                window_start = Instant::now();
                (w_new, w_repeat, w_bytes, w_cursor) = (0, 0, 0, 0);
                w_gaps.clear();
                w_enc.clear();
            }
        }
        let _ = cap.stop().await;
        drop(encoder);
        drop(cap);
        all_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "SOAK {}: {} new frames in {:.1} s = {:.1} fps; gap p50 {:.1} p99 {:.1} max {:.1} ms; gaps >50 ms: {}, >250 ms: {}; {:.1} Mbit/s; encode errors {}",
            d.name,
            total_new,
            secs,
            total_new as f64 / secs,
            percentile(&all_gaps, 0.5),
            percentile(&all_gaps, 0.99),
            all_gaps.last().copied().unwrap_or(0.0),
            all_gaps.iter().filter(|g| **g > 50.0).count(),
            all_gaps.iter().filter(|g| **g > 250.0).count(),
            total_bytes as f64 * 8.0 / 1e6 / secs,
            enc_errors
        );
        println!("SOAK memory {} -> {} MB; descriptors {} -> {}", rss_start / 1024, vm_rss_kb() / 1024, fds_start, open_fd_count());
    });
}
