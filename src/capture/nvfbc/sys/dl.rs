// Vendored from the `nvfbc-sys` 0.2.0 crate (BSD-2-Clause, Copyright (c) Hans Gaiser;
// see src/capture/nvfbc/LICENSE). Changed so NvFBC is loaded with dlopen at
// runtime instead of being linked, like Sunshine does.

use std::ffi::CStr;
use std::sync::OnceLock;

use super::generated::*;
use super::NVFBC_VERSION;

const LIBRARY: &str = "libnvidia-fbc.so.1";
static UNAVAILABLE_MSG: &CStr = c"NvFBC library (libnvidia-fbc.so.1) is not available";

struct Api {
	_lib: libloading::Library,
	functions: NVFBC_API_FUNCTION_LIST,
}
// The function table only holds C function pointers into the loaded library.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

static API: OnceLock<Result<Api, String>> = OnceLock::new();

fn load() -> Result<Api, String> {
	unsafe {
		let lib = libloading::Library::new(LIBRARY).map_err(|e| format!("{LIBRARY}: {e}"))?;
		let create: libloading::Symbol<unsafe extern "C" fn(*mut NVFBC_API_FUNCTION_LIST) -> NVFBCSTATUS> =
			lib.get(b"NvFBCCreateInstance\0").map_err(|e| format!("NvFBCCreateInstance: {e}"))?;
		let mut functions: NVFBC_API_FUNCTION_LIST = std::mem::zeroed();
		functions.dwVersion = NVFBC_VERSION;
		let status = create(&mut functions);
		if status != _NVFBCSTATUS_NVFBC_SUCCESS {
			return Err(format!("NvFBCCreateInstance failed with status {status}"));
		}
		Ok(Api { _lib: lib, functions })
	}
}

fn api() -> Result<&'static NVFBC_API_FUNCTION_LIST, &'static str> {
	API.get_or_init(load).as_ref().map(|a| &a.functions).map_err(|e| e.as_str())
}

/// Loads NvFBC on first use. `Err` explains why it is unavailable (no NVIDIA
/// driver, too old a driver, ...); callers then fall back to other capture paths.
pub fn load_error() -> Option<&'static str> {
	api().err()
}

/// Forwards to `NvFBCGetLastErrorStr` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCGetLastErrorStr(sessionHandle: NVFBC_SESSION_HANDLE) -> *const ::std::os::raw::c_char {
	match api().ok().and_then(|fl| fl.nvFBCGetLastErrorStr) {
		Some(f) => f(sessionHandle),
		None => UNAVAILABLE_MSG.as_ptr(),
	}
}

/// Forwards to `NvFBCCreateHandle` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCCreateHandle(pSessionHandle: *mut NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_CREATE_HANDLE_PARAMS,) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCCreateHandle) {
		Some(f) => f(pSessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

/// Forwards to `NvFBCDestroyHandle` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCDestroyHandle(sessionHandle: NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_DESTROY_HANDLE_PARAMS,) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCDestroyHandle) {
		Some(f) => f(sessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

/// Forwards to `NvFBCGetStatus` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCGetStatus(sessionHandle: NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_GET_STATUS_PARAMS) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCGetStatus) {
		Some(f) => f(sessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

/// Forwards to `NvFBCBindContext` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCBindContext(sessionHandle: NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_BIND_CONTEXT_PARAMS,) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCBindContext) {
		Some(f) => f(sessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

/// Forwards to `NvFBCReleaseContext` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCReleaseContext(sessionHandle: NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_RELEASE_CONTEXT_PARAMS,) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCReleaseContext) {
		Some(f) => f(sessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

/// Forwards to `NvFBCCreateCaptureSession` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCCreateCaptureSession(sessionHandle: NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_CREATE_CAPTURE_SESSION_PARAMS,) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCCreateCaptureSession) {
		Some(f) => f(sessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

/// Forwards to `NvFBCDestroyCaptureSession` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCDestroyCaptureSession(sessionHandle: NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_DESTROY_CAPTURE_SESSION_PARAMS,) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCDestroyCaptureSession) {
		Some(f) => f(sessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

/// Forwards to `NvFBCToSysSetUp` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCToSysSetUp(sessionHandle: NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_TOSYS_SETUP_PARAMS) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCToSysSetUp) {
		Some(f) => f(sessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

/// Forwards to `NvFBCToSysGrabFrame` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCToSysGrabFrame(sessionHandle: NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_TOSYS_GRAB_FRAME_PARAMS,) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCToSysGrabFrame) {
		Some(f) => f(sessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

/// Forwards to `NvFBCToCudaSetUp` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCToCudaSetUp(sessionHandle: NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_TOCUDA_SETUP_PARAMS,) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCToCudaSetUp) {
		Some(f) => f(sessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

/// Forwards to `NvFBCToCudaGrabFrame` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCToCudaGrabFrame(sessionHandle: NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_TOCUDA_GRAB_FRAME_PARAMS,) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCToCudaGrabFrame) {
		Some(f) => f(sessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

/// Forwards to `NvFBCToGLSetUp` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCToGLSetUp(sessionHandle: NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_TOGL_SETUP_PARAMS) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCToGLSetUp) {
		Some(f) => f(sessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

/// Forwards to `NvFBCToGLGrabFrame` in the dynamically loaded NvFBC library.
pub unsafe fn NvFBCToGLGrabFrame(sessionHandle: NVFBC_SESSION_HANDLE, pParams: *mut NVFBC_TOGL_GRAB_FRAME_PARAMS,) -> NVFBCSTATUS {
	match api().ok().and_then(|fl| fl.nvFBCToGLGrabFrame) {
		Some(f) => f(sessionHandle, pParams),
		None => _NVFBCSTATUS_NVFBC_ERR_INTERNAL,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn loads_when_driver_present() {
		let driver_present = ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"]
			.iter()
			.any(|d| std::path::Path::new(d).join(LIBRARY).exists());
		match load_error() {
			None => assert!(driver_present, "loaded NvFBC without the library on disk?"),
			Some(e) => assert!(!driver_present, "NvFBC present but failed to load: {e}"),
		}
		// Calls must fail gracefully, never crash, when the library is missing.
		let msg = unsafe { std::ffi::CStr::from_ptr(NvFBCGetLastErrorStr(0)) };
		assert!(!msg.to_bytes().is_empty() || driver_present);
	}
}
