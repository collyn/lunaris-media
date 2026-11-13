//! Windows input-desktop helper shared by capture and input injection.

use windows::Win32::Foundation::GENERIC_ALL;
use windows::Win32::System::StationsAndDesktops::{
    CloseDesktop, OpenInputDesktop, SetThreadDesktop, DESKTOP_ACCESS_FLAGS,
    DF_ALLOWOTHERACCOUNTHOOK,
};

/// Port of Sunshine `syncThreadDesktop()` (src/platform/windows/misc.cpp:247-264):
/// attach the CALLING thread to the desktop that currently receives input, so
/// that `DuplicateOutput` is not refused (E_ACCESSDENIED) after a desktop
/// switch (lock screen, UAC prompt, sleeping monitor), and `SendInput` reaches
/// the new desktop.
///
/// Returns the desktop handle value as Sunshine's does (misc.cpp:251-262): the
/// value of the handle returned by `OpenInputDesktop`, 0 (`nullptr`) when it
/// failed. As in Sunshine the handle is closed before it is returned, so the
/// value is only good for comparing with an earlier return value (see
/// `send_input_checked` in input.rs, port of `send_input`,
/// Sunshine-v0.23.1/input.cpp:468-483).
///
/// Adaptation: it acts on the calling thread only. Lunaris runs on tokio worker
/// threads and a task may move to another thread at any `.await`, so every
/// caller must invoke this in the same synchronous stretch of code as the
/// call it protects, with no `.await` in between.
pub(crate) fn sync_thread_desktop() -> usize {
    unsafe {
        let desk = match OpenInputDesktop(
            DF_ALLOWOTHERACCOUNTHOOK,
            false,
            DESKTOP_ACCESS_FLAGS(GENERIC_ALL.0),
        ) {
            Ok(d) => d,
            Err(e) => {
                log::error!("Failed to Open Input Desktop [{e}]");
                return 0;
            }
        };
        if let Err(e) = SetThreadDesktop(desk) {
            log::error!("Failed to sync desktop to thread [{e}]");
        }
        let _ = CloseDesktop(desk);
        desk.0 as usize
    }
}
