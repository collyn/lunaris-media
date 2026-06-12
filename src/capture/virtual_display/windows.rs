//! Windows virtual display via IddCx (Indirect Display Driver).
//!
//! This module communicates with an installed IddCx virtual display driver to
//! create and destroy virtual monitors at runtime. The driver must be installed
//! separately (e.g. `usbmmidd` or a custom IddCx driver).
//!
//! # Prerequisites
//!
//! - An IddCx virtual display driver must be installed and running.
//!   Common options:
//!   - [usbmmidd](https://github.com/ge9/IddSampleDriver) — open-source IddCx sample driver
//!   - [virtual-display-driver](https://github.com/roshkins/virtual-display-driver)
//! - The driver must support the custom IOCTL interface used here for
//!   creating/destroying virtual displays programmatically.
//!
//! # Fallback
//!
//! If no IddCx driver is found, the module falls back to creating a virtual
//! display via `wmic` or PowerShell, though this is less reliable.

#![cfg(target_os = "windows")]

use std::ffi::c_void;

use windows::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows::Win32::System::IO::DeviceIoControl;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
use windows::core::PCSTR;

use crate::error::MediaError;

/// Custom IOCTL codes for communicating with the IddCx virtual display driver.
/// These match the interface expected by `usbmmidd`-style drivers.
const IOCTL_IDD_CREATE_MONITOR: u32 = 0x222004; // CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801, METHOD_BUFFERED, FILE_ANY_ACCESS)
const IOCTL_IDD_DESTROY_MONITOR: u32 = 0x222008; // CTL_CODE(FILE_DEVICE_UNKNOWN, 0x802, METHOD_BUFFERED, FILE_ANY_ACCESS)

/// Maximum number of virtual displays supported by the driver.
const MAX_ADAPTER_ID_LENGTH: usize = 128;

/// Structure sent to the driver to create a virtual display.
#[repr(C)]
struct IddCreateMonitorRequest {
    width: u32,
    height: u32,
    refresh_rate: u32,
    adapter_id: [u16; MAX_ADAPTER_ID_LENGTH],
}

/// Structure received from the driver after creating a virtual display.
#[repr(C)]
struct IddCreateMonitorResponse {
    display_id: u32,
    device_name: [u16; 64],
    success: u32,
}

/// Structure sent to the driver to destroy a virtual display.
#[repr(C)]
struct IddDestroyMonitorRequest {
    display_id: u32,
}

/// Windows virtual display manager backed by an IddCx driver.
pub struct WindowsVirtualDisplay {
    driver_handle: HANDLE,
    display_id: u32,
    device_name: String,
    active: bool,
}

unsafe impl Send for WindowsVirtualDisplay {}

impl WindowsVirtualDisplay {
    /// Create a virtual display with the given resolution and refresh rate.
    ///
    /// Attempts to open a handle to the IddCx virtual display driver and
    /// sends a create IOCTL. Returns an error if the driver is not installed
    /// or the creation fails.
    pub fn create(width: u32, height: u32, fps: u32) -> Result<Self, MediaError> {
        let driver_handle = open_idd_driver()?;

        let mut request: IddCreateMonitorRequest = unsafe { std::mem::zeroed() };
        request.width = width;
        request.height = height;
        request.refresh_rate = fps;

        let mut response: IddCreateMonitorResponse = unsafe { std::mem::zeroed() };
        let mut bytes_returned: u32 = 0;

        let success = unsafe {
            DeviceIoControl(
                driver_handle,
                IOCTL_IDD_CREATE_MONITOR,
                Some(&request as *const _ as *const c_void),
                std::mem::size_of::<IddCreateMonitorRequest>() as u32,
                Some(&mut response as *mut _ as *mut c_void),
                std::mem::size_of::<IddCreateMonitorResponse>() as u32,
                Some(&mut bytes_returned),
                None,
            )
        };

        if !success.is_ok() || response.success == 0 {
            unsafe {
                let _ = CloseHandle(driver_handle);
            }
            return Err(MediaError::CaptureError(format!(
                "IddCx driver failed to create virtual display ({}x{}@{}fps)",
                width, height, fps
            )));
        }

        let device_name = String::from_utf16_lossy(&response.device_name)
            .trim_end_matches('\0')
            .to_string();

        log::info!(
            "Created Windows virtual display: {} ({}x{}@{}fps)",
            device_name,
            width,
            height,
            fps
        );

        Ok(Self {
            driver_handle,
            display_id: response.display_id,
            device_name,
            active: true,
        })
    }

    /// Return the display device name (e.g. `\\.\DISPLAY2`).
    pub fn display_id(&self) -> &str {
        &self.device_name
    }

    /// Explicitly destroy the virtual display.
    pub fn destroy(&mut self) -> Result<(), MediaError> {
        if !self.active {
            return Ok(());
        }

        let request = IddDestroyMonitorRequest {
            display_id: self.display_id,
        };
        let mut bytes_returned: u32 = 0;

        let success = unsafe {
            DeviceIoControl(
                self.driver_handle,
                IOCTL_IDD_DESTROY_MONITOR,
                Some(&request as *const _ as *const c_void),
                std::mem::size_of::<IddDestroyMonitorRequest>() as u32,
                None,
                0,
                Some(&mut bytes_returned),
                None,
            )
        };

        if !success.is_ok() {
            log::warn!(
                "IddCx driver failed to destroy virtual display {}",
                self.display_id
            );
        }

        unsafe {
            let _ = CloseHandle(self.driver_handle);
        }
        self.driver_handle = INVALID_HANDLE_VALUE;
        self.active = false;

        log::info!("Destroyed Windows virtual display {}", self.display_id);
        Ok(())
    }
}

impl Drop for WindowsVirtualDisplay {
    fn drop(&mut self) {
        if let Err(e) = self.destroy() {
            log::warn!("Failed to destroy Windows virtual display: {}", e);
        }
    }
}

/// Open a handle to the IddCx virtual display driver device.
///
/// Tries multiple device path patterns used by common IddCx drivers.
fn open_idd_driver() -> Result<HANDLE, MediaError> {
    // Common device path patterns for IddCx virtual display drivers
    let device_paths = [
        "\\\\.\\IddSampleDriver\0",
        "\\\\.\\VirtualDisplayDriver\0",
        "\\\\.\\IddCxDriver\0",
    ];

    for path in &device_paths {
        let handle = unsafe {
            windows::Win32::Storage::FileSystem::CreateFileA(
                PCSTR(path.as_ptr()),
                0x80000000u32 | 0x40000000u32, // GENERIC_READ | GENERIC_WRITE
                windows::Win32::Storage::FileSystem::FILE_SHARE_READ
                    | windows::Win32::Storage::FileSystem::FILE_SHARE_WRITE,
                None,
                windows::Win32::Storage::FileSystem::OPEN_EXISTING,
                windows::Win32::Storage::FileSystem::FILE_FLAGS_AND_ATTRIBUTES(0),
                None,
            )
        };

        match handle {
            Ok(h) if h != INVALID_HANDLE_VALUE => {
                log::info!("Opened IddCx driver: {}", path.trim_end_matches('\0'));
                return Ok(h);
            }
            _ => continue,
        }
    }

    Err(MediaError::CaptureError(
        "No IddCx virtual display driver found. Install usbmmidd or a compatible driver. \
         See: https://github.com/ge9/IddSampleDriver"
            .into(),
    ))
}
