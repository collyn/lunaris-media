//! Input injection module for Linux (uinput + X11 XTest) and Windows (SendInput).
//! Stubs are provided for other platforms to prevent compile errors.

#[cfg(target_os = "linux")]
use x11::xlib;
#[cfg(target_os = "linux")]
use x11::xtest;

#[cfg(target_os = "linux")]
use evdev::uinput::{VirtualDevice, VirtualDeviceBuilder};
#[cfg(target_os = "linux")]
use evdev::{AttributeSet, InputEvent, Key, RelativeAxisType, AbsoluteAxisType, UinputAbsSetup, AbsInfo};

#[cfg(target_os = "windows")]
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_TYPE, KEYBDINPUT, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE,
    MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_MOVE,
    MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_WHEEL,
    MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, MOUSEINPUT, MOUSE_EVENT_FLAGS,
    VIRTUAL_KEY, MapVirtualKeyW, MAPVK_VK_TO_VSC, KEYEVENTF_UNICODE,
};
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::GetLastError;
#[cfg(target_os = "windows")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(target_os = "windows")]
/// Direct FFI binding to `IsUserAnAdmin` in shell32.dll. Avoids pulling in the
/// `Win32_UI_Shell` windows-rs feature just for one bool function.
extern "system" {
    fn IsUserAnAdmin() -> i32;
}

/// Absolute axis range of the virtual absolute mouse (same as inputtino,
/// the input library used by Sunshine/Wolf). The compositor maps the full
/// range onto the whole desktop; resolution must be non-zero for libinput.
#[cfg(target_os = "linux")]
const ABS_MAX_WIDTH: i32 = 19200;
#[cfg(target_os = "linux")]
const ABS_MAX_HEIGHT: i32 = 12000;
#[cfg(target_os = "linux")]
const ABS_RESOLUTION: i32 = 28;

/// Mouse buttons exposed by both virtual mice.
#[cfg(target_os = "linux")]
const MOUSE_BUTTONS: [Key; 8] = [
    Key::BTN_LEFT,
    Key::BTN_RIGHT,
    Key::BTN_MIDDLE,
    Key::BTN_SIDE,
    Key::BTN_EXTRA,
    Key::BTN_FORWARD,
    Key::BTN_BACK,
    Key::BTN_TASK,
];

/// Virtual input devices created through `/dev/uinput`, modelled on
/// inputtino (Sunshine): a keyboard, a relative mouse with high-resolution
/// wheels, and an absolute mouse (ABS_X/ABS_Y + buttons — classified by udev
/// as a pointer, not a tablet). Works on every Wayland compositor because the
/// events enter through libinput like a physical device.
#[cfg(target_os = "linux")]
struct UinputInjector {
    mouse_rel: std::sync::Mutex<VirtualDevice>,
    mouse_abs: std::sync::Mutex<VirtualDevice>,
    keyboard: std::sync::Mutex<VirtualDevice>,
    /// Whether the last pointer motion was absolute. Buttons are emitted on
    /// the device that moved last so the compositor clicks where the pointer
    /// actually is (same rule as inputtino).
    last_absolute: std::sync::atomic::AtomicBool,
}

#[cfg(target_os = "linux")]
impl UinputInjector {
    fn new() -> Option<Self> {
        let log_err = |what: &str, e: std::io::Error| {
            log::warn!("uinput: {} failed: {} (is /dev/uinput writable? add the user to the 'input' group or install a udev rule)", what, e);
        };

        // 1. Keyboard device
        let mut kbd_keys = AttributeSet::new();
        for code in 1..=255 {
            kbd_keys.insert(Key::new(code));
        }
        let keyboard = VirtualDeviceBuilder::new()
            .map_err(|e| log_err("open", e))
            .ok()?
            .name("Lunaris Keyboard passthrough")
            .with_keys(&kbd_keys)
            .ok()?
            .build()
            .map_err(|e| log_err("create keyboard", e))
            .ok()?;

        let mut buttons = AttributeSet::new();
        for b in MOUSE_BUTTONS {
            buttons.insert(b);
        }

        // 2. Relative mouse with high-resolution wheels
        let mut rel_axes = AttributeSet::new();
        rel_axes.insert(RelativeAxisType::REL_X);
        rel_axes.insert(RelativeAxisType::REL_Y);
        rel_axes.insert(RelativeAxisType::REL_WHEEL);
        rel_axes.insert(RelativeAxisType::REL_HWHEEL);
        rel_axes.insert(RelativeAxisType::REL_WHEEL_HI_RES);
        rel_axes.insert(RelativeAxisType::REL_HWHEEL_HI_RES);
        let mouse_rel = VirtualDeviceBuilder::new()
            .ok()?
            .name("Lunaris Mouse passthrough")
            .with_keys(&buttons)
            .ok()?
            .with_relative_axes(&rel_axes)
            .ok()?
            .build()
            .map_err(|e| log_err("create relative mouse", e))
            .ok()?;

        // 3. Absolute mouse: buttons + ABS_X/ABS_Y with a real resolution.
        //    No BTN_TOOL_PEN/BTN_TOUCH, otherwise libinput treats it as a
        //    tablet (and rejects it without proper tablet capabilities).
        let abs_x = UinputAbsSetup::new(
            AbsoluteAxisType::ABS_X,
            AbsInfo::new(0, 0, ABS_MAX_WIDTH, 1, 0, ABS_RESOLUTION),
        );
        let abs_y = UinputAbsSetup::new(
            AbsoluteAxisType::ABS_Y,
            AbsInfo::new(0, 0, ABS_MAX_HEIGHT, 1, 0, ABS_RESOLUTION),
        );
        let mouse_abs = VirtualDeviceBuilder::new()
            .ok()?
            .name("Lunaris Mouse passthrough (absolute)")
            .with_keys(&buttons)
            .ok()?
            .with_absolute_axis(&abs_x)
            .ok()?
            .with_absolute_axis(&abs_y)
            .ok()?
            .build()
            .map_err(|e| log_err("create absolute mouse", e))
            .ok()?;

        log::info!("uinput: created virtual keyboard, relative mouse and absolute mouse");
        Some(Self {
            keyboard: std::sync::Mutex::new(keyboard),
            mouse_rel: std::sync::Mutex::new(mouse_rel),
            mouse_abs: std::sync::Mutex::new(mouse_abs),
            last_absolute: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn emit(dev: &std::sync::Mutex<VirtualDevice>, events: &[InputEvent]) {
        if let Ok(mut dev) = dev.lock() {
            if let Err(e) = dev.emit(events) {
                log::debug!("uinput emit failed: {}", e);
            }
        }
    }

    /// Device that should receive button events.
    fn pointer_device(&self) -> &std::sync::Mutex<VirtualDevice> {
        if self.last_absolute.load(std::sync::atomic::Ordering::Relaxed) {
            &self.mouse_abs
        } else {
            &self.mouse_rel
        }
    }

    fn move_absolute(&self, abs_x: i32, abs_y: i32) {
        self.last_absolute
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Self::emit(
            &self.mouse_abs,
            &[
                InputEvent::new(evdev::EventType::ABSOLUTE, AbsoluteAxisType::ABS_X.0, abs_x),
                InputEvent::new(evdev::EventType::ABSOLUTE, AbsoluteAxisType::ABS_Y.0, abs_y),
                InputEvent::new(evdev::EventType::SYNCHRONIZATION, 0, 0),
            ],
        );
    }

    /// Scroll by `notches` detents (positive = up/right), emitting both the
    /// high-resolution (120 per detent) and legacy wheel events.
    fn scroll(&self, horizontal: bool, notches: i32) {
        let (lo, hi) = if horizontal {
            (RelativeAxisType::REL_HWHEEL, RelativeAxisType::REL_HWHEEL_HI_RES)
        } else {
            (RelativeAxisType::REL_WHEEL, RelativeAxisType::REL_WHEEL_HI_RES)
        };
        Self::emit(
            &self.mouse_rel,
            &[
                InputEvent::new(evdev::EventType::RELATIVE, hi.0, notches * 120),
                InputEvent::new(evdev::EventType::RELATIVE, lo.0, notches),
                InputEvent::new(evdev::EventType::SYNCHRONIZATION, 0, 0),
            ],
        );
    }
}

/// A rectangle in desktop (compositor logical) coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenRegion {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl ScreenRegion {
    fn union(self, other: ScreenRegion) -> ScreenRegion {
        let x0 = self.x.min(other.x);
        let y0 = self.y.min(other.y);
        let x1 = (self.x + self.width).max(other.x + other.width);
        let y1 = (self.y + self.height).max(other.y + other.height);
        ScreenRegion { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }
    }
}

/// Region of the desktop being streamed plus the desktop bounding box, used
/// to map client coordinates onto the absolute mouse (which spans the whole
/// desktop). Set by the capture backend once the streamed monitor is known.
#[cfg(target_os = "linux")]
static CAPTURE_MAPPING: std::sync::Mutex<Option<(ScreenRegion, ScreenRegion)>> =
    std::sync::Mutex::new(None);

/// Tell the input layer which part of the desktop is being streamed (logical
/// coordinates, e.g. the portal stream position/size). Absolute pointer
/// events are then mapped onto that monitor, like Sunshine's touch port.
#[cfg(target_os = "linux")]
pub fn set_capture_region(target: ScreenRegion) {
    let desktop = desktop_layout()
        .into_iter()
        .fold(target, |acc, r| acc.union(r));
    log::info!("Input mapping: streamed region {:?} within desktop {:?}", target, desktop);
    if let Ok(mut m) = CAPTURE_MAPPING.lock() {
        *m = Some((target, desktop));
    }
}

/// Logical geometry of all enabled outputs. KDE's `kscreen-doctor` reports the
/// compositor's logical layout (scale-aware); `xrandr --listmonitors` (via
/// XWayland or X11) is the generic fallback.
#[cfg(target_os = "linux")]
fn desktop_layout() -> Vec<ScreenRegion> {
    if let Some(regions) = kscreen_layout() {
        if !regions.is_empty() {
            return regions;
        }
    }
    xrandr_monitors()
}

#[cfg(target_os = "linux")]
fn kscreen_layout() -> Option<Vec<ScreenRegion>> {
    let out = std::process::Command::new("kscreen-doctor").arg("-j").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // Minimal extraction without a JSON dependency: walk each output object.
    let mut regions = Vec::new();
    let outputs = text.split("\"outputs\"").nth(1)?;
    // Tiny scanner over the top-level `{...}` objects of the outputs array.
    let bytes = outputs.as_bytes();
    let mut depth = 0i32;
    let mut start = None;
    for (i, &c) in bytes.iter().enumerate() {
        match c {
            b'{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s0) = start.take() {
                        if let Some(r) = parse_kscreen_output(&outputs[s0..=i]) {
                            regions.push(r);
                        }
                    }
                }
                if depth < 0 {
                    break;
                }
            }
            _ => {}
        }
    }
    Some(regions)
}

/// Extract `"<key>": <number>` from a JSON fragment.
#[cfg(target_os = "linux")]
fn json_number(fragment: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{}\"", key);
    let idx = fragment.find(&pat)? + pat.len();
    let rest = fragment[idx..].trim_start().strip_prefix(':')?.trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

#[cfg(target_os = "linux")]
fn json_object<'a>(fragment: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{}\"", key);
    let idx = fragment.find(&pat)? + pat.len();
    let rest = &fragment[idx..];
    let open = rest.find('{')?;
    let close = rest[open..].find('}')? + open;
    Some(&rest[open..=close])
}

#[cfg(target_os = "linux")]
fn parse_kscreen_output(obj: &str) -> Option<ScreenRegion> {
    if !obj.contains("\"enabled\": true") && !obj.contains("\"enabled\":true") {
        return None;
    }
    // A replicating (mirrored) output shows its source's content; KDE keeps a
    // stale `pos` for it, which would wrongly widen the desktop (e.g. two
    // mirrored 1920x1080 screens reported as 3840x1080).
    if json_number(obj, "replicationSource").is_some_and(|src| src != 0.0) {
        return None;
    }
    let pos = json_object(obj, "pos")?;
    let size = json_object(obj, "size")?;
    let scale = json_number(obj, "scale").filter(|s| *s > 0.0).unwrap_or(1.0);
    let rotation = json_number(obj, "rotation").unwrap_or(1.0) as i32;
    let (mut w, mut h) = (json_number(size, "width")?, json_number(size, "height")?);
    if rotation == 2 || rotation == 8 {
        std::mem::swap(&mut w, &mut h);
    }
    Some(ScreenRegion {
        x: json_number(pos, "x")? as i32,
        y: json_number(pos, "y")? as i32,
        width: (w / scale).round() as i32,
        height: (h / scale).round() as i32,
    })
}

#[cfg(target_os = "linux")]
fn xrandr_monitors() -> Vec<ScreenRegion> {
    let Ok(out) = std::process::Command::new("xrandr").arg("--listmonitors").output() else {
        return Vec::new();
    };
    // Lines look like: " 0: +*DP-2 1920/527x1080/296+0+0  DP-2"
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .skip(1)
        .filter_map(|line| {
            let geom = line.split_whitespace().nth(2)?;
            let (w, rest) = geom.split_once('/')?;
            let (_, rest) = rest.split_once('x')?;
            let (h, rest) = rest.split_once('/')?;
            let mut parts = rest.split('+').skip(1);
            Some(ScreenRegion {
                x: parts.next()?.parse().ok()?,
                y: parts.next()?.parse().ok()?,
                width: w.parse().ok()?,
                height: h.parse().ok()?,
            })
        })
        .collect()
}

/// Map client coordinates (`x`/`y` within `ref_w`×`ref_h`) to the absolute
/// mouse axis range, honouring the streamed region inside the desktop.
#[cfg(target_os = "linux")]
fn map_to_abs_axes(x: i32, y: i32, ref_w: i32, ref_h: i32) -> (i32, i32) {
    let mapping = CAPTURE_MAPPING.lock().ok().and_then(|m| *m);
    // Aim at the pixel centre so the compositor's floor() lands on (x, y).
    let fx = ((x as f64 + 0.5) / ref_w as f64).clamp(0.0, 1.0);
    let fy = ((y as f64 + 0.5) / ref_h as f64).clamp(0.0, 1.0);
    let (nx, ny) = match mapping {
        Some((target, desktop)) if desktop.width > 0 && desktop.height > 0 => (
            (target.x - desktop.x) as f64 / desktop.width as f64
                + fx * target.width as f64 / desktop.width as f64,
            (target.y - desktop.y) as f64 / desktop.height as f64
                + fy * target.height as f64 / desktop.height as f64,
        ),
        _ => (fx, fy),
    };
    (
        (nx * ABS_MAX_WIDTH as f64).round() as i32,
        (ny * ABS_MAX_HEIGHT as f64).round() as i32,
    )
}

#[cfg(target_os = "linux")]
struct MonitorGeometry {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

#[cfg(target_os = "linux")]
fn get_primary_monitor_geometry() -> Option<MonitorGeometry> {
    let output = std::process::Command::new("xrandr")
        .arg("--query")
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    let mut first_connected = None;
    
    for line in stdout.lines() {
        if !line.starts_with(' ') && line.contains(" connected") {
            let is_primary = line.contains(" primary");
            let parts: Vec<&str> = line.split_whitespace().collect();
            for part in parts {
                if part.contains('x') && part.contains('+') {
                    let subparts: Vec<&str> = part.split('+').collect();
                    if subparts.len() >= 3 {
                        let mut res_parts = subparts[0].split('x');
                        let w: i32 = res_parts.next()?.parse().ok()?;
                        let h: i32 = res_parts.next()?.parse().ok()?;
                        let x: i32 = subparts[1].parse().ok()?;
                        let y: i32 = subparts[2].parse().ok()?;
                        
                        let geom = MonitorGeometry { x, y, width: w, height: h };
                        if is_primary {
                            return Some(geom);
                        }
                        if first_connected.is_none() {
                            first_connected = Some(geom);
                        }
                    }
                }
            }
        }
    }
    first_connected
}

#[cfg(target_os = "linux")]
struct X11Backend {
    display: *mut xlib::Display,
    screen: libc::c_int,
    monitor_x: i32,
    monitor_y: i32,
    monitor_width: i32,
    monitor_height: i32,
    /// Keycodes without keysyms, borrowed to type characters the keyboard
    /// layout doesn't have (é, ệ, emoji…), like xdotool does.
    spare_keys: std::sync::Mutex<SpareKeys>,
}

#[cfg(target_os = "linux")]
#[derive(Default)]
struct SpareKeys {
    /// None until looked up.
    codes: Option<Vec<u8>>,
    next: usize,
}

#[cfg(target_os = "linux")]
enum Backend {
    Uinput(UinputInjector),
    X11(X11Backend),
}

/// A cross-platform input injector helper.
pub struct InputInjector {
    #[cfg(target_os = "linux")]
    backend: Backend,
    #[cfg(target_os = "windows")]
    _private: (), // SendInput is stateless, no per-instance state needed
}

#[cfg(target_os = "windows")]
/// Timestamp of last UIPI-access-denied log line (milliseconds since epoch),
/// used to rate-limit the warning to once every ~5 seconds instead of
/// thousands of times per second while an elevated window is in focus.
static LAST_UIPI_WARN_MS: AtomicU64 = AtomicU64::new(0);

#[cfg(target_os = "windows")]
// Desktop handle value that the last `sync_thread_desktop()` of this thread
// returned. Port of `thread_local HDESK _lastKnownInputDesktop`
// (Sunshine-v0.23.1/input.cpp:33). Each of the input threads has its own, as
// the desktop is attached per thread.
thread_local! {
    static LAST_KNOWN_INPUT_DESKTOP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(target_os = "windows")]
/// Wrap `SendInput` with error checking and input desktop switching. Returns
/// the same value as `SendInput` (count of successfully inserted events).
///
/// Port of `send_input` (Sunshine-v0.23.1/input.cpp:468-483): when `SendInput`
/// inserts fewer events than given (it does so when the desktop that receives
/// input is not the one of the calling thread: UAC prompt, lock screen), the
/// thread is attached to the input desktop (`sync_thread_desktop`, :476) and,
/// when that desktop is not the one of the last attempt (:477-478), the call is
/// repeated (`goto retry`, :473 and :479). Adaptation: Sunshine sends one `INPUT` per
/// call, here a slice is sent and "failed" means fewer events than given.
///
/// When it still fails, logs a rate-limited diagnostic (Sunshine: "Couldn't
/// send input", :481): with no desktop change to blame it is nearly always UIPI,
/// when a privileged window (Task Manager) has focus and blocks input from a
/// non-admin process.
unsafe fn send_input_checked(inputs: &[INPUT], cb_size: i32) -> u32 {
    let mut ret;
    loop {
        ret = SendInput(inputs, cb_size);
        if ret as usize == inputs.len() {
            break;
        }
        let desk = crate::win_desktop::sync_thread_desktop();
        if LAST_KNOWN_INPUT_DESKTOP.with(|d| d.get()) != desk {
            LAST_KNOWN_INPUT_DESKTOP.with(|d| d.set(desk));
            continue;
        }
        break;
    }
    if ret == 0 {
        let err = GetLastError();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let last = LAST_UIPI_WARN_MS.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last) > 5000 {
            LAST_UIPI_WARN_MS.store(now_ms, Ordering::Relaxed);
            log::warn!(
                "SendInput blocked (Win32 error {}). This usually means an \
                 elevated window (e.g. Task Manager, UAC) is in focus and \
                 UIPI is preventing non-admin input injection. To fix, \
                 restart the Lunaris agent as Administrator.",
                err.0
            );
        }
    }
    ret
}

// SAFETY: input injection backends are safe to send between threads when wrapped
// and synchronized (e.g. by using a Mutex in the calling/owning task).
unsafe impl Send for InputInjector {}
unsafe impl Sync for InputInjector {}

impl InputInjector {
    /// Creates a new InputInjector.
    pub fn new() -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            let is_wayland = std::env::var("XDG_SESSION_TYPE")
                .unwrap_or_default()
                == "wayland";

            // On Wayland, X11 XTest events go through XWayland and cannot
            // reach native Wayland windows — they silently fail. Prefer
            // uinput (kernel-level evdev injection) which works everywhere.
            // On X11, prefer XTest for lower latency and better compositor
            // integration (uinput devices may be ignored by some WMs).
            if is_wayland {
                if let Some(uinput) = UinputInjector::new() {
                    log::info!("InputInjector: initialized uinput backend (Wayland)");
                    return Some(Self {
                        backend: Backend::Uinput(uinput),
                    });
                }
                log::warn!("InputInjector: uinput unavailable on Wayland, falling back to X11 XTest (may not reach native Wayland windows)");
            }

            // X11 path (primary for X11 sessions, fallback for Wayland)
            if std::env::var("DISPLAY").is_ok() {
                unsafe {
                    xlib::XInitThreads();
                }
                let display = {
                    let _lock = crate::X11_MUTEX.lock().unwrap();
                    unsafe { xlib::XOpenDisplay(std::ptr::null()) }
                };
                if !display.is_null() {
                    log::info!("InputInjector: initialized X11 display connection (X11 XTest backend)");
                    let (screen, screen_width, screen_height) = unsafe {
                        let s = xlib::XDefaultScreen(display);
                        let w = xlib::XDisplayWidth(display, s) as i32;
                        let h = xlib::XDisplayHeight(display, s) as i32;
                        (s, w, h)
                    };

                    let (monitor_x, monitor_y, monitor_width, monitor_height) = if let Some(geom) = get_primary_monitor_geometry() {
                        (geom.x, geom.y, geom.width, geom.height)
                    } else {
                        (0, 0, screen_width, screen_height)
                    };

                    return Some(Self {
                        backend: Backend::X11(X11Backend {
                            display,
                            screen,
                            monitor_x,
                            monitor_y,
                            monitor_width,
                            monitor_height,
                            spare_keys: std::sync::Mutex::new(SpareKeys::default()),
                        }),
                    });
                }
            }

            // uinput fallback for X11 (or headless without DISPLAY)
            if !is_wayland {
                if let Some(uinput) = UinputInjector::new() {
                    log::info!("InputInjector: initialized uinput backend");
                    return Some(Self {
                        backend: Backend::Uinput(uinput),
                    });
                }
            }

            log::error!("InputInjector: failed to initialize both X11 XTest and uinput");
            None
        }
        #[cfg(target_os = "windows")]
        {
            // Warn early if the agent isn't running as Administrator. Without
            // admin elevation, Windows UIPI blocks `SendInput` to any elevated
            // window — most notably Task Manager (Ctrl+Shift+Esc). When that
            // happens all mouse/keyboard injection silently fails and remote
            // input appears frozen.
            unsafe {
                if IsUserAnAdmin() == 0 {
                    log::warn!(
                        "Windows UIPI: agent is NOT running as Administrator. \
                         If the remote mouse/keyboard stops responding when an \
                         elevated window (Task Manager, UAC prompt, …) is in \
                         focus, restart the agent as Administrator."
                    );
                }
            }
            log::info!("InputInjector: initialized Windows SendInput backend");
            Some(Self { _private: () })
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            log::info!("InputInjector: stub initialized for this platform");
            Some(Self {})
        }
    }

    /// Moves the mouse pointer relatively by `dx` and `dy`.
    pub fn move_mouse_relative(&self, dx: i32, dy: i32) {
        #[cfg(target_os = "linux")]
        match &self.backend {
            Backend::Uinput(u) => {
                u.last_absolute.store(false, std::sync::atomic::Ordering::Relaxed);
                let events = [
                    InputEvent::new(evdev::EventType::RELATIVE, RelativeAxisType::REL_X.0, dx),
                    InputEvent::new(evdev::EventType::RELATIVE, RelativeAxisType::REL_Y.0, dy),
                    InputEvent::new(evdev::EventType::SYNCHRONIZATION, 0, 0),
                ];
                if let Ok(mut dev) = u.mouse_rel.lock() {
                    let _ = dev.emit(&events);
                }
            }
            Backend::X11(x11) => {
                let _lock = crate::X11_MUTEX.lock().unwrap();
                unsafe {
                    let mut root_return = 0;
                    let mut child_return = 0;
                    let mut root_x = 0;
                    let mut root_y = 0;
                    let mut win_x = 0;
                    let mut win_y = 0;
                    let mut mask = 0;
                    let root = xlib::XDefaultRootWindow(x11.display);
                    let ok = xlib::XQueryPointer(
                        x11.display,
                        root,
                        &mut root_return,
                        &mut child_return,
                        &mut root_x,
                        &mut root_y,
                        &mut win_x,
                        &mut win_y,
                        &mut mask,
                    );
                    if ok != 0 {
                        let target_x = (root_x + dx).clamp(x11.monitor_x, x11.monitor_x + x11.monitor_width - 1);
                        let target_y = (root_y + dy).clamp(x11.monitor_y, x11.monitor_y + x11.monitor_height - 1);
                        xtest::XTestFakeMotionEvent(x11.display, x11.screen, target_x, target_y, 0);
                    } else {
                        // Fallback: if querying pointer fails, use relative event injection
                        let wrong_func: unsafe extern "C" fn(
                            *mut x11::xlib::Display,
                            std::os::raw::c_int,
                            std::os::raw::c_int,
                            std::os::raw::c_int,
                            std::os::raw::c_ulong,
                        ) -> std::os::raw::c_int = xtest::XTestFakeRelativeMotionEvent;
                        let correct_func = std::mem::transmute::<
                            _,
                            unsafe extern "C" fn(
                                *mut x11::xlib::Display,
                                std::os::raw::c_int,
                                std::os::raw::c_int,
                                std::os::raw::c_ulong,
                            ) -> std::os::raw::c_int,
                        >(wrong_func);
                        correct_func(x11.display, dx, dy, 0);
                    }
                }
            }
        }
        #[cfg(target_os = "windows")]
        unsafe {
            let input = INPUT {
                r#type: INPUT_TYPE(0), // INPUT_MOUSE
                Anonymous: INPUT_0 {
                    mi: MOUSEINPUT {
                        dx: dx,
                        dy: dy,
                        mouseData: 0,
                        dwFlags: MOUSEEVENTF_MOVE,
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            };
            send_input_checked(&[input], std::mem::size_of::<INPUT>() as i32);
        }
    }

    /// Moves the mouse pointer absolutely to scaled coordinates.
    pub fn move_mouse_absolute(&self, x: i32, y: i32, reference_width: i32, reference_height: i32) {
        #[cfg(target_os = "linux")]
        match &self.backend {
            Backend::Uinput(u) => {
                if reference_width > 0 && reference_height > 0 {
                    let (abs_x, abs_y) = map_to_abs_axes(x, y, reference_width, reference_height);
                    u.move_absolute(abs_x, abs_y);
                }
            }
            Backend::X11(x11) => {
                let _lock = crate::X11_MUTEX.lock().unwrap();
                unsafe {
                    if reference_width > 0 && reference_height > 0 {
                        let abs_x = x11.monitor_x + (x * x11.monitor_width) / reference_width;
                        let abs_y = x11.monitor_y + (y * x11.monitor_height) / reference_height;
                        xtest::XTestFakeMotionEvent(x11.display, x11.screen, abs_x, abs_y, 0);
                    }
                }
            }
        }
        #[cfg(target_os = "windows")]
        {
            if reference_width > 0 && reference_height > 0 {
                let abs_x = (x as i64 * 65535) / reference_width as i64;
                let abs_y = (y as i64 * 65535) / reference_height as i64;
                unsafe {
                    let input = INPUT {
                        r#type: INPUT_TYPE(0), // INPUT_MOUSE
                        Anonymous: INPUT_0 {
                            mi: MOUSEINPUT {
                                dx: abs_x as i32,
                                dy: abs_y as i32,
                                mouseData: 0,
                                dwFlags: MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE,
                                time: 0,
                                dwExtraInfo: 0,
                            },
                        },
                    };
                    send_input_checked(&[input], std::mem::size_of::<INPUT>() as i32);
                }
            }
        }
    }

    /// Injects a mouse button press or release.
    pub fn mouse_button(&self, button: u32, is_press: bool) {
        #[cfg(target_os = "linux")]
        match &self.backend {
            Backend::Uinput(u) => {
                match button {
                    4 | 5 => {
                        // Scroll Wheel: 4 = Up, 5 = Down
                        if is_press {
                            u.scroll(false, if button == 4 { 1 } else { -1 });
                        }
                    }
                    6 | 7 => {
                        // Horizontal Scroll Wheel: 6 = Left, 7 = Right
                        if is_press {
                            u.scroll(true, if button == 7 { 1 } else { -1 });
                        }
                    }
                    _ => {
                        if let Some(linux_btn) = mouse_button_to_linux_keycode(button) {
                            let events = [
                                InputEvent::new(
                                    evdev::EventType::KEY,
                                    linux_btn.0,
                                    if is_press { 1 } else { 0 }
                                ),
                                InputEvent::new(evdev::EventType::SYNCHRONIZATION, 0, 0),
                            ];
                            UinputInjector::emit(u.pointer_device(), &events);
                        }
                    }
                }
            }
            Backend::X11(x11) => {
                let _lock = crate::X11_MUTEX.lock().unwrap();
                unsafe {
                    let press_val = if is_press { 1 } else { 0 };
                    xtest::XTestFakeButtonEvent(x11.display, button, press_val, 0);
                    xlib::XFlush(x11.display);
                }
            }
        }
        #[cfg(target_os = "windows")]
        unsafe {
            let (flags, mouse_data) = match button {
                1 => (if is_press { MOUSEEVENTF_LEFTDOWN } else { MOUSEEVENTF_LEFTUP }, 0i32),
                2 => (if is_press { MOUSEEVENTF_MIDDLEDOWN } else { MOUSEEVENTF_MIDDLEUP }, 0i32),
                3 => (if is_press { MOUSEEVENTF_RIGHTDOWN } else { MOUSEEVENTF_RIGHTUP }, 0i32),
                4 => if is_press { (MOUSEEVENTF_WHEEL, 120i32) } else { return },
                5 => if is_press { (MOUSEEVENTF_WHEEL, -120i32) } else { return },
                8 => (if is_press { MOUSEEVENTF_XDOWN } else { MOUSEEVENTF_XUP }, 1i32), // XBUTTON1
                9 => (if is_press { MOUSEEVENTF_XDOWN } else { MOUSEEVENTF_XUP }, 2i32), // XBUTTON2
                _ => return,
            };
            let input = INPUT {
                r#type: INPUT_TYPE(0), // INPUT_MOUSE
                Anonymous: INPUT_0 {
                    mi: MOUSEINPUT {
                        dx: 0,
                        dy: 0,
                        mouseData: mouse_data as u32,
                        dwFlags: flags,
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            };
            send_input_checked(&[input], std::mem::size_of::<INPUT>() as i32);
        }
    }

    /// Injects a keyboard key press or release.
    pub fn keyboard_key(&self, vk_code: u16, is_press: bool) {
        #[cfg(target_os = "linux")]
        match &self.backend {
            Backend::Uinput(u) => {
                if let Some(linux_key) = vk_to_linux_keycode(vk_code) {
                    let events = [
                        InputEvent::new(
                            evdev::EventType::KEY,
                            linux_key.0,
                            if is_press { 1 } else { 0 }
                        ),
                        InputEvent::new(evdev::EventType::SYNCHRONIZATION, 0, 0),
                    ];
                    if let Ok(mut dev) = u.keyboard.lock() {
                        let _ = dev.emit(&events);
                    }
                }
            }
            Backend::X11(x11) => {
                let _lock = crate::X11_MUTEX.lock().unwrap();
                unsafe {
                    if let Some(keysym) = vk_to_keysym(vk_code) {
                        let keycode = xlib::XKeysymToKeycode(x11.display, keysym as xlib::KeySym);
                        if keycode > 0 {
                            let press_val = if is_press { 1 } else { 0 };
                            xtest::XTestFakeKeyEvent(x11.display, keycode as u32, press_val, 0);
                            xlib::XFlush(x11.display);
                        }
                    }
                }
            }
        }
        #[cfg(target_os = "windows")]
        unsafe {
            let vk = VIRTUAL_KEY(vk_code);
            let scan = MapVirtualKeyW(vk_code as u32, MAPVK_VK_TO_VSC) as u16;
            let mut flags = KEYEVENTF_SCANCODE;
            if !is_press {
                flags |= KEYEVENTF_KEYUP;
            }
            let input = INPUT {
                r#type: INPUT_TYPE(1), // INPUT_KEYBOARD
                Anonymous: INPUT_0 {
                    ki: KEYBDINPUT {
                        wVk: vk,
                        wScan: scan,
                        dwFlags: flags,
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            };
            send_input_checked(&[input], std::mem::size_of::<INPUT>() as i32);
        }
    }

    /// Types text as the user would (any Unicode character, independent of
    /// the keyboard layout where the platform allows it). Used for text from
    /// phone keyboards, which send characters rather than keys.
    pub fn type_text(&self, text: &str) {
        #[cfg(target_os = "linux")]
        match &self.backend {
            Backend::X11(x11) => {
                let _lock = crate::X11_MUTEX.lock().unwrap();
                for ch in text.chars() {
                    unsafe { x11_type_char(x11, ch) };
                }
                unsafe { xlib::XFlush(x11.display) };
            }
            Backend::Uinput(u) => {
                for ch in text.chars() {
                    uinput_type_char(u, ch);
                }
            }
        }
        #[cfg(target_os = "windows")]
        unsafe {
            let mut inputs: Vec<INPUT> = Vec::new();
            for ch in text.chars() {
                if ch == '\n' || ch == '\r' {
                    // Enter as a key: some apps ignore a Unicode newline.
                    for up in [false, true] {
                        inputs.push(INPUT {
                            r#type: INPUT_TYPE(1),
                            Anonymous: INPUT_0 { ki: KEYBDINPUT {
                                wVk: VIRTUAL_KEY(0x0D), wScan: 0,
                                dwFlags: if up { KEYEVENTF_KEYUP } else { Default::default() },
                                time: 0, dwExtraInfo: 0,
                            } },
                        });
                    }
                    continue;
                }
                let mut units = [0u16; 2];
                for &unit in ch.encode_utf16(&mut units).iter() {
                    for up in [false, true] {
                        let mut flags = KEYEVENTF_UNICODE;
                        if up { flags |= KEYEVENTF_KEYUP; }
                        inputs.push(INPUT {
                            r#type: INPUT_TYPE(1),
                            Anonymous: INPUT_0 { ki: KEYBDINPUT {
                                wVk: VIRTUAL_KEY(0), wScan: unit, dwFlags: flags, time: 0, dwExtraInfo: 0,
                            } },
                        });
                    }
                }
            }
            if !inputs.is_empty() {
                send_input_checked(&inputs, std::mem::size_of::<INPUT>() as i32);
            }
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        let _ = text;
    }

    /// Flushes any pending input events to the display server.
    pub fn flush(&self) {
        #[cfg(target_os = "linux")]
        match &self.backend {
            Backend::X11(x11) => {
                let _lock = crate::X11_MUTEX.lock().unwrap();
                unsafe {
                    xlib::XFlush(x11.display);
                }
            }
            _ => {}
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for InputInjector {
    fn drop(&mut self) {
        match &self.backend {
            Backend::X11(x11) => {
                if !x11.display.is_null() {
                    unsafe {
                        xlib::XCloseDisplay(x11.display);
                    }
                    log::info!("InputInjector: closed X11 display connection");
                }
            }
            _ => {}
        }
    }
}

#[cfg(target_os = "linux")]
fn mouse_button_to_linux_keycode(button: u32) -> Option<Key> {
    let code = match button {
        1 => Key::BTN_LEFT,
        2 => Key::BTN_MIDDLE,
        3 => Key::BTN_RIGHT,
        8 => Key::BTN_SIDE,
        9 => Key::BTN_EXTRA,
        _ => return None,
    };
    Some(code)
}

/// X keysym of a character: Latin-1 keysyms equal the code point, the rest
/// are 0x0100_0000 + code point.
#[cfg(target_os = "linux")]
fn char_keysym(ch: char) -> xlib::KeySym {
    match ch {
        '\n' | '\r' => 0xff0d, // Return
        '\t' => 0xff09,         // Tab
        '\u{8}' => 0xff08,      // BackSpace
        c if (c as u32) >= 0x20 && (c as u32) <= 0x7e => c as xlib::KeySym,
        c if (c as u32) >= 0xa0 && (c as u32) <= 0xff => c as xlib::KeySym,
        c => 0x0100_0000 + c as xlib::KeySym,
    }
}

#[cfg(target_os = "linux")]
unsafe fn x11_key(x11: &X11Backend, keycode: u8, shift: bool) {
    let d = x11.display;
    let shift_code = if shift { xlib::XKeysymToKeycode(d, 0xffe1 /* Shift_L */) } else { 0 };
    if shift_code > 0 {
        xtest::XTestFakeKeyEvent(d, shift_code as u32, 1, 0);
    }
    xtest::XTestFakeKeyEvent(d, keycode as u32, 1, 0);
    xtest::XTestFakeKeyEvent(d, keycode as u32, 0, 0);
    if shift_code > 0 {
        xtest::XTestFakeKeyEvent(d, shift_code as u32, 0, 0);
    }
}

/// Types one character with XTest: through the key that has it in the
/// current layout, else by giving a spare keycode that keysym for a moment.
#[cfg(target_os = "linux")]
unsafe fn x11_type_char(x11: &X11Backend, ch: char) {
    let d = x11.display;
    let keysym = char_keysym(ch);
    let keycode = xlib::XKeysymToKeycode(d, keysym);
    if keycode > 0 {
        if xlib::XKeycodeToKeysym(d, keycode, 0) == keysym {
            x11_key(x11, keycode, false);
            return;
        }
        if xlib::XKeycodeToKeysym(d, keycode, 1) == keysym {
            x11_key(x11, keycode, true);
            return;
        }
    }
    let Ok(mut spare) = x11.spare_keys.lock() else { return };
    if spare.codes.is_none() {
        spare.codes = Some(x11_spare_keycodes(d));
    }
    let codes = spare.codes.as_ref().unwrap();
    if codes.is_empty() {
        log::warn!("No spare X keycode to type {:?}", ch);
        return;
    }
    // Rotate through the spare keycodes so an app still reading the
    // previous character's mapping isn't given this one's.
    let code = codes[spare.next % codes.len()];
    spare.next = spare.next.wrapping_add(1);
    let mut syms: [libc::c_ulong; 2] = [keysym as libc::c_ulong, keysym as libc::c_ulong];
    xlib::XChangeKeyboardMapping(d, code as i32, 2, syms.as_mut_ptr(), 1);
    xlib::XSync(d, 0);
    x11_key(x11, code, false);
    xlib::XSync(d, 0);
}

/// Keycodes that have no keysym at all (up to 8).
#[cfg(target_os = "linux")]
unsafe fn x11_spare_keycodes(d: *mut xlib::Display) -> Vec<u8> {
    let (mut min, mut max) = (0, 0);
    xlib::XDisplayKeycodes(d, &mut min, &mut max);
    let count = max - min + 1;
    let mut per = 0;
    let map = xlib::XGetKeyboardMapping(d, min as u8, count, &mut per);
    if map.is_null() || per <= 0 {
        return Vec::new();
    }
    let mut spare = Vec::new();
    for i in (0..count).rev() {
        let syms = std::slice::from_raw_parts(map.add((i * per) as usize), per as usize);
        if syms.iter().all(|&s| s == 0) {
            spare.push((min + i) as u8);
            if spare.len() == 8 {
                break;
            }
        }
    }
    xlib::XFree(map as *mut _);
    spare
}

/// A printable ASCII character on a US keyboard: (key, with Shift).
#[cfg(target_os = "linux")]
fn ascii_us_key(ch: char) -> Option<(Key, bool)> {
    const LETTERS: [Key; 26] = [
        Key::KEY_A, Key::KEY_B, Key::KEY_C, Key::KEY_D, Key::KEY_E, Key::KEY_F, Key::KEY_G,
        Key::KEY_H, Key::KEY_I, Key::KEY_J, Key::KEY_K, Key::KEY_L, Key::KEY_M, Key::KEY_N,
        Key::KEY_O, Key::KEY_P, Key::KEY_Q, Key::KEY_R, Key::KEY_S, Key::KEY_T, Key::KEY_U,
        Key::KEY_V, Key::KEY_W, Key::KEY_X, Key::KEY_Y, Key::KEY_Z,
    ];
    const DIGITS: [Key; 10] = [
        Key::KEY_0, Key::KEY_1, Key::KEY_2, Key::KEY_3, Key::KEY_4,
        Key::KEY_5, Key::KEY_6, Key::KEY_7, Key::KEY_8, Key::KEY_9,
    ];
    Some(match ch {
        'a'..='z' => (LETTERS[ch as usize - 'a' as usize], false),
        'A'..='Z' => (LETTERS[ch as usize - 'A' as usize], true),
        '0'..='9' => (DIGITS[ch as usize - '0' as usize], false),
        ' ' => (Key::KEY_SPACE, false),
        '\n' | '\r' => (Key::KEY_ENTER, false),
        '\t' => (Key::KEY_TAB, false),
        '-' => (Key::KEY_MINUS, false), '_' => (Key::KEY_MINUS, true),
        '=' => (Key::KEY_EQUAL, false), '+' => (Key::KEY_EQUAL, true),
        '[' => (Key::KEY_LEFTBRACE, false), '{' => (Key::KEY_LEFTBRACE, true),
        ']' => (Key::KEY_RIGHTBRACE, false), '}' => (Key::KEY_RIGHTBRACE, true),
        '\\' => (Key::KEY_BACKSLASH, false), '|' => (Key::KEY_BACKSLASH, true),
        ';' => (Key::KEY_SEMICOLON, false), ':' => (Key::KEY_SEMICOLON, true),
        '\'' => (Key::KEY_APOSTROPHE, false), '"' => (Key::KEY_APOSTROPHE, true),
        '`' => (Key::KEY_GRAVE, false), '~' => (Key::KEY_GRAVE, true),
        ',' => (Key::KEY_COMMA, false), '<' => (Key::KEY_COMMA, true),
        '.' => (Key::KEY_DOT, false), '>' => (Key::KEY_DOT, true),
        '/' => (Key::KEY_SLASH, false), '?' => (Key::KEY_SLASH, true),
        '!' => (Key::KEY_1, true), '@' => (Key::KEY_2, true), '#' => (Key::KEY_3, true),
        '$' => (Key::KEY_4, true), '%' => (Key::KEY_5, true), '^' => (Key::KEY_6, true),
        '&' => (Key::KEY_7, true), '*' => (Key::KEY_8, true), '(' => (Key::KEY_9, true),
        ')' => (Key::KEY_0, true),
        _ => return None,
    })
}

#[cfg(target_os = "linux")]
fn uinput_tap(u: &UinputInjector, key: Key, shift: bool) {
    let mut events = Vec::new();
    if shift {
        events.push(InputEvent::new(evdev::EventType::KEY, Key::KEY_LEFTSHIFT.0, 1));
    }
    events.push(InputEvent::new(evdev::EventType::KEY, key.0, 1));
    events.push(InputEvent::new(evdev::EventType::SYNCHRONIZATION, 0, 0));
    events.push(InputEvent::new(evdev::EventType::KEY, key.0, 0));
    if shift {
        events.push(InputEvent::new(evdev::EventType::KEY, Key::KEY_LEFTSHIFT.0, 0));
    }
    events.push(InputEvent::new(evdev::EventType::SYNCHRONIZATION, 0, 0));
    if let Ok(mut dev) = u.keyboard.lock() {
        let _ = dev.emit(&events);
    }
}

/// uinput only has keys: ASCII is typed on a US layout; other characters
/// through Ctrl+Shift+U <hex> Space (GTK and IBus/Fcitx apps).
#[cfg(target_os = "linux")]
fn uinput_type_char(u: &UinputInjector, ch: char) {
    if let Some((key, shift)) = ascii_us_key(ch) {
        uinput_tap(u, key, shift);
        return;
    }
    let mut events = vec![
        InputEvent::new(evdev::EventType::KEY, Key::KEY_LEFTCTRL.0, 1),
        InputEvent::new(evdev::EventType::KEY, Key::KEY_LEFTSHIFT.0, 1),
        InputEvent::new(evdev::EventType::KEY, Key::KEY_U.0, 1),
        InputEvent::new(evdev::EventType::SYNCHRONIZATION, 0, 0),
        InputEvent::new(evdev::EventType::KEY, Key::KEY_U.0, 0),
        InputEvent::new(evdev::EventType::KEY, Key::KEY_LEFTSHIFT.0, 0),
        InputEvent::new(evdev::EventType::KEY, Key::KEY_LEFTCTRL.0, 0),
        InputEvent::new(evdev::EventType::SYNCHRONIZATION, 0, 0),
    ];
    if let Ok(mut dev) = u.keyboard.lock() {
        let _ = dev.emit(&events);
    }
    events.clear();
    for digit in format!("{:x}", ch as u32).chars() {
        if let Some((key, shift)) = ascii_us_key(digit) {
            uinput_tap(u, key, shift);
        }
    }
    uinput_tap(u, Key::KEY_SPACE, false);
}

#[cfg(all(test, target_os = "linux"))]
mod type_text_tests {
    use super::*;

    #[test]
    fn keysyms_follow_x11_rules() {
        assert_eq!(char_keysym('a'), 0x61);
        assert_eq!(char_keysym('é'), 0xe9);
        assert_eq!(char_keysym('ệ'), 0x0100_0000 + 0x1ec7);
        assert_eq!(char_keysym('\n'), 0xff0d);
    }

    #[test]
    fn ascii_on_a_us_keyboard() {
        assert_eq!(ascii_us_key('a'), Some((Key::KEY_A, false)));
        assert_eq!(ascii_us_key('Z'), Some((Key::KEY_Z, true)));
        assert_eq!(ascii_us_key('?'), Some((Key::KEY_SLASH, true)));
        assert_eq!(ascii_us_key('@'), Some((Key::KEY_2, true)));
        assert_eq!(ascii_us_key('ệ'), None);
    }
}

#[cfg(target_os = "linux")]
fn vk_to_linux_keycode(vk: u16) -> Option<Key> {
    let code = match vk {
        0x08 => Key::KEY_BACKSPACE,
        0x09 => Key::KEY_TAB,
        0x0D => Key::KEY_ENTER,
        0x1B => Key::KEY_ESC,
        0x20 => Key::KEY_SPACE,
        0x21 => Key::KEY_PAGEUP,
        0x22 => Key::KEY_PAGEDOWN,
        0x23 => Key::KEY_END,
        0x24 => Key::KEY_HOME,
        0x25 => Key::KEY_LEFT,
        0x26 => Key::KEY_UP,
        0x27 => Key::KEY_RIGHT,
        0x28 => Key::KEY_DOWN,
        0x2D => Key::KEY_INSERT,
        0x2E => Key::KEY_DELETE,
        // Numbers 0-9
        0x30 => Key::KEY_0,
        0x31 => Key::KEY_1,
        0x32 => Key::KEY_2,
        0x33 => Key::KEY_3,
        0x34 => Key::KEY_4,
        0x35 => Key::KEY_5,
        0x36 => Key::KEY_6,
        0x37 => Key::KEY_7,
        0x38 => Key::KEY_8,
        0x39 => Key::KEY_9,
        // Letters A-Z
        0x41 => Key::KEY_A,
        0x42 => Key::KEY_B,
        0x43 => Key::KEY_C,
        0x44 => Key::KEY_D,
        0x45 => Key::KEY_E,
        0x46 => Key::KEY_F,
        0x47 => Key::KEY_G,
        0x48 => Key::KEY_H,
        0x49 => Key::KEY_I,
        0x4A => Key::KEY_J,
        0x4B => Key::KEY_K,
        0x4C => Key::KEY_L,
        0x4D => Key::KEY_M,
        0x4E => Key::KEY_N,
        0x4F => Key::KEY_O,
        0x50 => Key::KEY_P,
        0x51 => Key::KEY_Q,
        0x52 => Key::KEY_R,
        0x53 => Key::KEY_S,
        0x54 => Key::KEY_T,
        0x55 => Key::KEY_U,
        0x56 => Key::KEY_V,
        0x57 => Key::KEY_W,
        0x58 => Key::KEY_X,
        0x59 => Key::KEY_Y,
        0x5A => Key::KEY_Z,
        // Win/Super
        0x5B => Key::KEY_LEFTMETA,
        0x5C => Key::KEY_RIGHTMETA,
        // Numpad
        0x60 => Key::KEY_KP0,
        0x61 => Key::KEY_KP1,
        0x62 => Key::KEY_KP2,
        0x63 => Key::KEY_KP3,
        0x64 => Key::KEY_KP4,
        0x65 => Key::KEY_KP5,
        0x66 => Key::KEY_KP6,
        0x67 => Key::KEY_KP7,
        0x68 => Key::KEY_KP8,
        0x69 => Key::KEY_KP9,
        0x6A => Key::KEY_KPASTERISK,
        0x6B => Key::KEY_KPPLUS,
        0x6D => Key::KEY_KPMINUS,
        0x6E => Key::KEY_KPDOT,
        0x6F => Key::KEY_KPSLASH,
        // F1-F12
        0x70 => Key::KEY_F1,
        0x71 => Key::KEY_F2,
        0x72 => Key::KEY_F3,
        0x73 => Key::KEY_F4,
        0x74 => Key::KEY_F5,
        0x75 => Key::KEY_F6,
        0x76 => Key::KEY_F7,
        0x77 => Key::KEY_F8,
        0x78 => Key::KEY_F9,
        0x79 => Key::KEY_F10,
        0x7A => Key::KEY_F11,
        0x7B => Key::KEY_F12,
        0x90 => Key::KEY_NUMLOCK,
        0x91 => Key::KEY_SCROLLLOCK,
        // Modifiers
        0x10 | 0xA0 => Key::KEY_LEFTSHIFT,
        0xA1 => Key::KEY_RIGHTSHIFT,
        0x11 | 0xA2 => Key::KEY_LEFTCTRL,
        0xA3 => Key::KEY_RIGHTCTRL,
        0x12 | 0xA4 => Key::KEY_LEFTALT,
        0xA5 => Key::KEY_RIGHTALT,
        // Punctuation keys
        0xBA => Key::KEY_SEMICOLON,
        0xBB => Key::KEY_EQUAL,
        0xBC => Key::KEY_COMMA,
        0xBD => Key::KEY_MINUS,
        0xBE => Key::KEY_DOT,
        0xBF => Key::KEY_SLASH,
        0xC0 => Key::KEY_GRAVE,
        0xDB => Key::KEY_LEFTBRACE,
        0xDC => Key::KEY_BACKSLASH,
        0xDD => Key::KEY_RIGHTBRACE,
        0xDE => Key::KEY_APOSTROPHE,
        _ => return None,
    };
    Some(code)
}

#[cfg(target_os = "linux")]
fn vk_to_keysym(vk: u16) -> Option<u32> {
    use x11::keysym::*;
    let sym = match vk {
        0x08 => XK_BackSpace,
        0x09 => XK_Tab,
        0x0D => XK_Return,
        0x1B => XK_Escape,
        0x20 => XK_space,
        0x21 => XK_Prior, // Page Up
        0x22 => XK_Next,  // Page Down
        0x23 => XK_End,
        0x24 => XK_Home,
        0x25 => XK_Left,
        0x26 => XK_Up,
        0x27 => XK_Right,
        0x28 => XK_Down,
        0x2D => XK_Insert,
        0x2E => XK_Delete,
        // Numbers 0-9
        0x30..=0x39 => XK_0 + (vk - 0x30) as u32,
        // Letters A-Z
        0x41..=0x5A => XK_a + (vk - 0x41) as u32,
        // Win/Super
        0x5B => XK_Super_L,
        0x5C => XK_Super_R,
        // Numpad
        0x60 => XK_KP_0,
        0x61 => XK_KP_1,
        0x62 => XK_KP_2,
        0x63 => XK_KP_3,
        0x64 => XK_KP_4,
        0x65 => XK_KP_5,
        0x66 => XK_KP_6,
        0x67 => XK_KP_7,
        0x68 => XK_KP_8,
        0x69 => XK_KP_9,
        0x6A => XK_KP_Multiply,
        0x6B => XK_KP_Add,
        0x6D => XK_KP_Subtract,
        0x6E => XK_KP_Decimal,
        0x6F => XK_KP_Divide,
        // F1-F12
        0x70..=0x7B => XK_F1 + (vk - 0x70) as u32,
        0x90 => XK_Num_Lock,
        0x91 => XK_Scroll_Lock,
        // Modifiers
        0xA0 | 0x10 => XK_Shift_L,
        0xA1 => XK_Shift_R,
        0xA2 | 0x11 => XK_Control_L,
        0xA3 => XK_Control_R,
        0xA4 | 0x12 => XK_Alt_L,
        0xA5 => XK_Alt_R,

        // Punctuation keys
        0xBA => XK_semicolon,
        0xBB => XK_equal,
        0xBC => XK_comma,
        0xBD => XK_minus,
        0xBE => XK_period,
        0xBF => XK_slash,
        0xC0 => XK_grave,
        0xDB => XK_bracketleft,
        0xDC => XK_backslash,
        0xDD => XK_bracketright,
        0xDE => XK_apostrophe,

        _ => return None,
    };
    Some(sym)
}

#[cfg(all(test, target_os = "linux"))]
mod layout_tests {
    use super::{parse_kscreen_output, ScreenRegion};

    const DP2: &str = r#"{"enabled": true, "id": 2, "name": "DP-2", "pos": {"x": 0, "y": 0}, "replicationSource": 0, "rotation": 1, "scale": 1, "size": {"height": 1080, "width": 1920}}"#;

    #[test]
    fn mirrored_output_is_ignored() {
        let hdmi = r#"{"enabled": true, "id": 1, "name": "HDMI-A-2", "pos": {"x": 1920, "y": 0}, "replicationSource": 2, "rotation": 1, "scale": 1, "size": {"height": 1080, "width": 1920}}"#;
        assert_eq!(parse_kscreen_output(hdmi), None);
        assert_eq!(
            parse_kscreen_output(DP2),
            Some(ScreenRegion { x: 0, y: 0, width: 1920, height: 1080 })
        );
    }

    #[test]
    fn extended_and_scaled_outputs() {
        let side = r#"{"enabled": true, "pos": {"x": 1920, "y": 0}, "replicationSource": 0, "rotation": 1, "scale": 2, "size": {"height": 2160, "width": 3840}}"#;
        assert_eq!(
            parse_kscreen_output(side),
            Some(ScreenRegion { x: 1920, y: 0, width: 1920, height: 1080 })
        );
        let rotated = r#"{"enabled": true, "pos": {"x": 0, "y": 0}, "rotation": 2, "scale": 1, "size": {"height": 1080, "width": 1920}}"#;
        assert_eq!(
            parse_kscreen_output(rotated),
            Some(ScreenRegion { x: 0, y: 0, width: 1080, height: 1920 })
        );
        let disabled = r#"{"enabled": false, "pos": {"x": 0, "y": 0}, "scale": 1, "size": {"height": 1080, "width": 1920}}"#;
        assert_eq!(parse_kscreen_output(disabled), None);
    }
}
