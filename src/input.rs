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
    VIRTUAL_KEY, MapVirtualKeyW, MAPVK_VK_TO_VSC,
};

#[cfg(target_os = "linux")]
struct UinputInjector {
    mouse_rel: std::sync::Mutex<VirtualDevice>,
    mouse_abs: std::sync::Mutex<VirtualDevice>,
    keyboard: std::sync::Mutex<VirtualDevice>,
}

#[cfg(target_os = "linux")]
impl UinputInjector {
    fn new() -> Option<Self> {
        // 1. Keyboard device
        let mut kbd_keys = AttributeSet::new();
        for code in 1..=255 {
            kbd_keys.insert(Key::new(code));
        }
        let keyboard = VirtualDeviceBuilder::new()
            .ok()?
            .name("Lunaris Keyboard passthrough")
            .with_keys(&kbd_keys).ok()?
            .build()
            .ok()?;

        // 2. Relative Mouse device
        let mut rel_keys = AttributeSet::new();
        rel_keys.insert(Key::BTN_LEFT);
        rel_keys.insert(Key::BTN_RIGHT);
        rel_keys.insert(Key::BTN_MIDDLE);
        rel_keys.insert(Key::BTN_SIDE);
        rel_keys.insert(Key::BTN_EXTRA);

        let mut rel_axes = AttributeSet::new();
        rel_axes.insert(RelativeAxisType::REL_X);
        rel_axes.insert(RelativeAxisType::REL_Y);
        rel_axes.insert(RelativeAxisType::REL_WHEEL);
        rel_axes.insert(RelativeAxisType::REL_HWHEEL);

        let mouse_rel = VirtualDeviceBuilder::new()
            .ok()?
            .name("Lunaris Mouse passthrough")
            .with_keys(&rel_keys).ok()?
            .with_relative_axes(&rel_axes).ok()?
            .build()
            .ok()?;

        // 3. Absolute Mouse/Tablet device
        let mut abs_keys = AttributeSet::new();
        abs_keys.insert(Key::BTN_TOOL_PEN);

        let abs_x_info = AbsInfo::new(0, 0, 32767, 0, 0, 0);
        let abs_y_info = AbsInfo::new(0, 0, 32767, 0, 0, 0);
        let abs_x_setup = UinputAbsSetup::new(AbsoluteAxisType::ABS_X, abs_x_info);
        let abs_y_setup = UinputAbsSetup::new(AbsoluteAxisType::ABS_Y, abs_y_info);

        let mouse_abs = VirtualDeviceBuilder::new()
            .ok()?
            .name("Lunaris Mouse passthrough (absolute)")
            .with_keys(&abs_keys).ok()?
            .with_absolute_axis(&abs_x_setup).ok()?
            .with_absolute_axis(&abs_y_setup).ok()?
            .build()
            .ok()?;

        log::info!("uinput: successfully created virtual input devices (/dev/uinput)");
        Some(Self {
            keyboard: std::sync::Mutex::new(keyboard),
            mouse_rel: std::sync::Mutex::new(mouse_rel),
            mouse_abs: std::sync::Mutex::new(mouse_abs),
        })
    }
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

// SAFETY: input injection backends are safe to send between threads when wrapped
// and synchronized (e.g. by using a Mutex in the calling/owning task).
unsafe impl Send for InputInjector {}
unsafe impl Sync for InputInjector {}

impl InputInjector {
    /// Creates a new InputInjector.
    pub fn new() -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            // 1. Prioritize X11 XTest if running inside an active display session (DISPLAY is set),
            // as uinput devices might be ignored/unplugged by the active X11/display server.
            if std::env::var("DISPLAY").is_ok() {
                unsafe {
                    xlib::XInitThreads();
                }
                let display = {
                    let _lock = crate::X11_MUTEX.lock().unwrap();
                    unsafe { xlib::XOpenDisplay(std::ptr::null()) }
                };
                if !display.is_null() {
                    log::info!("InputInjector: initialized X11 display connection (preferred backend)");
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
                        }),
                    });
                }
            }

            // 2. Try uinput (optimal kernel-level low-latency injection, matching Sunshine, fallback for headless/Wayland)
            if let Some(uinput) = UinputInjector::new() {
                log::info!("InputInjector: initialized uinput backend");
                return Some(Self {
                    backend: Backend::Uinput(uinput),
                });
            }

            log::error!("InputInjector: failed to initialize both X11 XTest and uinput");
            None
        }
        #[cfg(target_os = "windows")]
        {
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
                    xtest::XTestFakeRelativeMotionEvent(x11.display, x11.screen, dx, dy, 0);
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
            SendInput(&[input], std::mem::size_of::<INPUT>() as i32);
        }
    }

    /// Moves the mouse pointer absolutely to scaled coordinates.
    pub fn move_mouse_absolute(&self, x: i32, y: i32, reference_width: i32, reference_height: i32) {
        #[cfg(target_os = "linux")]
        match &self.backend {
            Backend::Uinput(u) => {
                if reference_width > 0 && reference_height > 0 {
                    let abs_x = (x * 32767) / reference_width;
                    let abs_y = (y * 32767) / reference_height;
                    let events = [
                        InputEvent::new(evdev::EventType::KEY, Key::BTN_TOOL_PEN.0, 1),
                        InputEvent::new(evdev::EventType::ABSOLUTE, AbsoluteAxisType::ABS_X.0, abs_x),
                        InputEvent::new(evdev::EventType::ABSOLUTE, AbsoluteAxisType::ABS_Y.0, abs_y),
                        InputEvent::new(evdev::EventType::SYNCHRONIZATION, 0, 0),
                    ];
                    if let Ok(mut dev) = u.mouse_abs.lock() {
                        let _ = dev.emit(&events);
                    }
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
                    SendInput(&[input], std::mem::size_of::<INPUT>() as i32);
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
                            let val = if button == 4 { 1 } else { -1 };
                            let events = [
                                InputEvent::new(evdev::EventType::RELATIVE, RelativeAxisType::REL_WHEEL.0, val),
                                InputEvent::new(evdev::EventType::SYNCHRONIZATION, 0, 0),
                            ];
                            if let Ok(mut dev) = u.mouse_rel.lock() {
                                let _ = dev.emit(&events);
                            }
                        }
                    }
                    6 | 7 => {
                        // Horizontal Scroll Wheel: 6 = Left, 7 = Right
                        if is_press {
                            let val = if button == 7 { 1 } else { -1 };
                            let events = [
                                InputEvent::new(evdev::EventType::RELATIVE, RelativeAxisType::REL_HWHEEL.0, val),
                                InputEvent::new(evdev::EventType::SYNCHRONIZATION, 0, 0),
                            ];
                            if let Ok(mut dev) = u.mouse_rel.lock() {
                                let _ = dev.emit(&events);
                            }
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
                            if let Ok(mut dev) = u.mouse_rel.lock() {
                                let _ = dev.emit(&events);
                            }
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
            SendInput(&[input], std::mem::size_of::<INPUT>() as i32);
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
            SendInput(&[input], std::mem::size_of::<INPUT>() as i32);
        }
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
