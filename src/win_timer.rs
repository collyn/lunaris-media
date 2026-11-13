//! Windows timing helpers ported from Sunshine (`src/platform/windows/misc.cpp`):
//! the high-precision sleep timer, the system timer-resolution request that
//! is held while streaming, and the QPC -> monotonic-clock translation.

use std::time::{Duration, Instant};

use windows::core::{s, w};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Media::{timeBeginPeriod, timeEndPeriod};
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows::Win32::System::Threading::{
    CreateWaitableTimerExW, SetWaitableTimer, WaitForSingleObject, INFINITE,
};

// Values of CREATE_WAITABLE_TIMER_HIGH_RESOLUTION and TIMER_ALL_ACCESS
// (synchapi.h / winnt.h), spelled out so they do not depend on how the
// `windows` crate types them.
const CREATE_WAITABLE_TIMER_HIGH_RESOLUTION: u32 = 0x0000_0002;
const TIMER_ALL_ACCESS: u32 = 0x001F_0003;

/// Port of `win32_high_precision_timer` (misc.cpp:1824-1870, class
/// "Implements the high-precision timer using Win32 waitable timers").
///
/// Constructor (misc.cpp, `win32_high_precision_timer()`): "Use
/// CREATE_WAITABLE_TIMER_HIGH_RESOLUTION if supported (Windows 10 1809+)",
/// falling back to a plain waitable timer (flags 0).
pub(crate) struct HighPrecisionTimer {
    timer: Option<HANDLE>,
}

// A waitable-timer handle is a kernel object handle usable from any thread
// (SetWaitableTimer / WaitForSingleObject are thread-safe).
unsafe impl Send for HighPrecisionTimer {}
unsafe impl Sync for HighPrecisionTimer {}

impl HighPrecisionTimer {
    pub(crate) fn new() -> Self {
        let create = |flags: u32| unsafe {
            CreateWaitableTimerExW(None, windows::core::PCWSTR::null(), flags, TIMER_ALL_ACCESS)
        };
        let timer = match create(CREATE_WAITABLE_TIMER_HIGH_RESOLUTION) {
            Ok(h) => Some(h),
            Err(_) => match create(0) {
                Ok(h) => Some(h),
                Err(e) => {
                    log::error!(
                        "Unable to create high_precision_timer, CreateWaitableTimerEx() failed: {e}"
                    );
                    None
                }
            },
        };
        Self { timer }
    }

    /// Port of `win32_high_precision_timer::sleep_for`. Like Sunshine, an
    /// uninitialised timer and durations > 5 s are logged and NOT slept.
    /// (A negative duration cannot occur: `Duration` is unsigned.)
    pub(crate) fn sleep_for(&self, duration: Duration) {
        let Some(timer) = self.timer else {
            log::error!("Attempting high_precision_timer::sleep_for() with uninitialized timer");
            return;
        };
        if duration > Duration::from_secs(5) {
            log::error!(
                "Attempting high_precision_timer::sleep_for() with unexpectedly large duration (>5s)"
            );
            return;
        }
        // due_time.QuadPart = duration.count() / -100  (negative = relative, 100 ns units)
        let due_time: i64 = duration.as_nanos() as i64 / -100;
        unsafe {
            let _ = SetWaitableTimer(timer, &due_time, 0, None, None, false);
            WaitForSingleObject(timer, INFINITE);
        }
    }
}

impl Drop for HighPrecisionTimer {
    fn drop(&mut self) {
        if let Some(t) = self.timer.take() {
            unsafe {
                let _ = CloseHandle(t);
            }
        }
    }
}

type NtQueryTimerResolutionFn = unsafe extern "system" fn(*mut u32, *mut u32, *mut u32) -> i32;
type NtSetTimerResolutionFn = unsafe extern "system" fn(u32, u8, *mut u32) -> i32;

/// Port of `nt_set_timer_resolution_max` / `nt_set_timer_resolution_min`
/// (misc.cpp:110-125): query min/max, then `NtSetTimerResolution(maximum or
/// minimum, TRUE, &current)`; false when either call fails (NTSTATUS < 0).
///
/// Adaptation: Sunshine declares the two ntdll functions `extern "C"` at link
/// time; here they are resolved with `GetProcAddress` so that no import
/// library is needed.
fn nt_set_timer_resolution(to_max: bool) -> bool {
    unsafe {
        let Ok(ntdll) = GetModuleHandleW(w!("ntdll.dll")) else {
            return false;
        };
        let (Some(query), Some(set)) = (
            GetProcAddress(ntdll, s!("NtQueryTimerResolution")),
            GetProcAddress(ntdll, s!("NtSetTimerResolution")),
        ) else {
            return false;
        };
        let query: NtQueryTimerResolutionFn = std::mem::transmute(query);
        let set: NtSetTimerResolutionFn = std::mem::transmute(set);
        let (mut minimum, mut maximum, mut current) = (0u32, 0u32, 0u32);
        if query(&mut minimum, &mut maximum, &mut current) < 0 {
            return false;
        }
        set(if to_max { maximum } else { minimum }, 1, &mut current) >= 0
    }
}

/// The system timer resolution raised while streaming.
///
/// Port of the timer part of `streaming_will_start()` (misc.cpp:1218-1225,
/// "Reduce timer period to 0.5ms": `nt_set_timer_resolution_max()`, on
/// failure "falling back to timeBeginPeriod()" with `timeBeginPeriod(1)`) and
/// of `streaming_will_stop()` (misc.cpp:1306-1314, "End our 0.5ms timer
/// request": `nt_set_timer_resolution_min()` if the NT call had been used,
/// else `timeEndPeriod(1)`).
///
/// `raise()` is the start part, `Drop` the stop part, so that the resolution
/// is restored on every path (stop, failed start, reinit).
/// Adaptation: Sunshine keeps the "used NtSetTimerResolution" flag in a global
/// atomic (misc.cpp:108); here it lives in the guard.
pub(crate) struct TimerResolution {
    used_nt_set_timer_resolution: bool,
}

impl TimerResolution {
    pub(crate) fn raise() -> Self {
        if nt_set_timer_resolution(true) {
            Self { used_nt_set_timer_resolution: true }
        } else {
            log::error!("NtSetTimerResolution() failed, falling back to timeBeginPeriod()");
            unsafe {
                timeBeginPeriod(1);
            }
            Self { used_nt_set_timer_resolution: false }
        }
    }
}

impl Drop for TimerResolution {
    fn drop(&mut self) {
        if self.used_nt_set_timer_resolution {
            if !nt_set_timer_resolution(false) {
                log::error!(
                    "nt_set_timer_resolution_min() failed even though nt_set_timer_resolution_max() succeeded"
                );
            }
        } else {
            unsafe {
                timeEndPeriod(1);
            }
        }
    }
}

/// Port of `qpc_counter()` (misc.cpp:1789-1794).
pub(crate) fn qpc_counter() -> i64 {
    let mut counter = 0i64;
    if unsafe { QueryPerformanceCounter(&mut counter) }.is_ok() {
        counter
    } else {
        0
    }
}

/// Port of `qpc_time_difference` (misc.cpp:1799-1811): nanoseconds between two
/// QPC readings, 0 when there is no frequency. Signed.
fn qpc_time_difference_ns(c1: i64, c2: i64) -> i64 {
    static FREQUENCY: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
    let frequency = *FREQUENCY.get_or_init(|| {
        let mut f = 0i64;
        let _ = unsafe { QueryPerformanceFrequency(&mut f) };
        f
    });
    if frequency == 0 {
        return 0;
    }
    // Adaptation (flagged for review): the Sunshine copy in _ref computes
    // `(c1 - c2) * frequency / std::nano::den` (misc.cpp:1805), i.e. ticks *
    // ticks-per-second / 1e9, which is not a time (for a 10 MHz QPC it turns
    // 10 ms into 1 us, so the "frame timestamp" would always be ~now). The
    // intent is nanoseconds = ticks * 1e9 / frequency, used here.
    ((c1 - c2) as i128 * 1_000_000_000 / frequency as i128) as i64
}

/// `std::chrono::steady_clock::now() - qpc_time_difference(qpc_counter(),
/// qpc_displayed)` (display_vram.cpp:1386-1390): translate a QPC value into a
/// point on the monotonic clock. `None` if the result is not representable.
pub(crate) fn qpc_to_instant(qpc_displayed: i64) -> Option<Instant> {
    let diff_ns = qpc_time_difference_ns(qpc_counter(), qpc_displayed);
    let now = Instant::now();
    if diff_ns >= 0 {
        now.checked_sub(Duration::from_nanos(diff_ns as u64))
    } else {
        now.checked_add(Duration::from_nanos(diff_ns.unsigned_abs()))
    }
}

/// Port of `sleep_overshoot_logger` (`logging::time_delta_periodic_logger`,
/// platform/common.h:822, logging.h time_delta_periodic_logger +
/// stat_trackers.h:51-62 `collect_and_callback_on_interval`): min/max/avg of
/// how late a sleep returned, logged at debug level every 20 s.
pub(crate) struct SleepOvershootLogger {
    point1: Instant,
    last_callback_time: Instant,
    min: f64,
    max: f64,
    total: f64,
    calls: u32,
}

impl SleepOvershootLogger {
    pub(crate) fn new() -> Self {
        let now = Instant::now();
        Self {
            point1: now,
            last_callback_time: now,
            min: f64::MAX,
            max: f64::MIN,
            total: 0.0,
            calls: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::new();
    }

    pub(crate) fn first_point(&mut self, point: Instant) {
        self.point1 = point;
    }

    pub(crate) fn second_point_now_and_log(&mut self) {
        let now = Instant::now();
        // std::chrono::duration<double, std::milli>(point - point1), signed.
        let ms = if now >= self.point1 {
            (now - self.point1).as_secs_f64() * 1000.0
        } else {
            -((self.point1 - now).as_secs_f64() * 1000.0)
        };
        if self.calls == 0 {
            self.last_callback_time = now;
        } else if now > self.last_callback_time + Duration::from_secs(20) {
            log::debug!(
                "Frame capture sleep overshoot (min/max/avg): {:.2}ms/{:.2}ms/{:.2}ms",
                self.min,
                self.max,
                self.total / self.calls as f64
            );
            self.min = f64::MAX;
            self.max = f64::MIN;
            self.total = 0.0;
            self.calls = 0;
        }
        self.min = self.min.min(ms);
        self.max = self.max.max(ms);
        self.total += ms;
        self.calls += 1;
    }
}
