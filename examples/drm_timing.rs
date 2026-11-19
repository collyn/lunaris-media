//! Times each KMS call the capture service makes per frame, per display.
//!
//! ```text
//! drm_timing [SECONDS]
//! ```
//!
//! Read-only: "get" ioctls only (and, as root, framebuffer export plus a
//! read-only map of the cursor buffer). Sets no mode, injects nothing.
//! Without root the export/cursor-read rows are skipped.

use std::path::Path;
use std::time::{Duration, Instant};

use lunaris_media::capture::drm_service::kms::{self, Card};

#[derive(Default)]
struct Stat {
    n: u32,
    sum: Duration,
    max: Duration,
    errors: u32,
}

impl Stat {
    fn time<T, E>(&mut self, f: impl FnOnce() -> Result<T, E>) -> Option<T> {
        let t = Instant::now();
        let r = f();
        let d = t.elapsed();
        self.n += 1;
        self.sum += d;
        self.max = self.max.max(d);
        match r {
            Ok(v) => Some(v),
            Err(_) => {
                self.errors += 1;
                None
            }
        }
    }

    fn line(&self, name: &str) {
        if self.n == 0 {
            println!("   {:<14} not run", name);
            return;
        }
        println!(
            "   {:<14} {:>5} calls  avg {:>8.3} ms  max {:>8.3} ms  errors {}",
            name,
            self.n,
            self.sum.as_secs_f64() * 1000.0 / self.n as f64,
            self.max.as_secs_f64() * 1000.0,
            self.errors
        );
    }
}

fn main() {
    let secs: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let root = unsafe { libc::geteuid() } == 0;
    println!("root: {}", root);
    for path in kms::card_paths(Path::new("/dev/dri")) {
        let Ok(card) = Card::open(&path) else { continue };
        let Ok(displays) = kms::card_displays(&card) else { continue };
        for d in displays {
            println!("== {} {} crtc {} ({} s)", d.card, d.name, d.crtc_id, secs);
            let (mut crtc, mut active, mut cursor, mut export, mut read_cur, mut grab) =
                (Stat::default(), Stat::default(), Stat::default(), Stat::default(), Stat::default(), Stat::default());
            let mut fbs = std::collections::HashSet::new();
            let mut cursor_seen = 0u32;
            let end = Instant::now() + Duration::from_secs(secs);
            while Instant::now() < end {
                if let Some(c) = crtc.time(|| card.crtc(d.crtc_id)) {
                    fbs.insert(c.fb_id);
                    if root && c.fb_id != 0 {
                        export.time(|| card.export_fb(c.fb_id));
                    }
                }
                active.time(|| card.crtc_active(d.crtc_id).ok_or(()));
                if let Some(st) = cursor.time(|| card.cursor_on(d.crtc_id).ok_or(())) {
                    cursor_seen += 1;
                    if root {
                        read_cur.time(|| card.read_cursor(st.fb_id).ok_or(()));
                    }
                }
                if root {
                    grab.time(|| kms::grab(&card, d.crtc_id));
                }
            }
            crtc.line("GETCRTC");
            active.line("crtc ACTIVE");
            cursor.line("cursor_on");
            export.line("export_fb");
            read_cur.line("read_cursor");
            grab.line("grab()");
            println!("   distinct fb ids {}; cursor on this display in {} samples", fbs.len(), cursor_seen);
        }
    }
}
