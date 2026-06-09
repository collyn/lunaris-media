use nvfbc::cuda::CaptureMethod;
use nvfbc::{BufferFormat, CudaCapturer};
use std::{error::Error, ptr};

type CuInitFn = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGetFn = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CuCtxCreateFn = unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32, i32) -> i32;
type CuCtxDestroyFn = unsafe extern "C" fn(*mut std::ffi::c_void) -> i32;
type CuMemAllocFn = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemFreeFn = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyDtoDFn = unsafe extern "C" fn(u64, u64, usize) -> i32;

fn main() -> Result<(), Box<dyn Error>> {
    // Dynamically load libcuda.so
    let mut lib = unsafe {
        libc::dlopen(
            b"libcuda.so.1\0".as_ptr() as *const libc::c_char,
            libc::RTLD_LAZY,
        )
    };
    if lib.is_null() {
        lib = unsafe {
            libc::dlopen(
                b"libcuda.so\0".as_ptr() as *const libc::c_char,
                libc::RTLD_LAZY,
            )
        };
    }
    if lib.is_null() {
        panic!("Failed to load libcuda.so.1 or libcuda.so");
    }

    unsafe {
        let cu_init_ptr = libc::dlsym(lib, b"cuInit\0".as_ptr() as *const libc::c_char);
        let cu_init: CuInitFn = std::mem::transmute(cu_init_ptr);

        let cu_device_get_ptr = libc::dlsym(lib, b"cuDeviceGet\0".as_ptr() as *const libc::c_char);
        let cu_device_get: CuDeviceGetFn = std::mem::transmute(cu_device_get_ptr);

        let cu_ctx_create_ptr =
            libc::dlsym(lib, b"cuCtxCreate_v2\0".as_ptr() as *const libc::c_char);
        let cu_ctx_create: CuCtxCreateFn = std::mem::transmute(cu_ctx_create_ptr);

        let cu_ctx_destroy_ptr =
            libc::dlsym(lib, b"cuCtxDestroy_v2\0".as_ptr() as *const libc::c_char);
        let cu_ctx_destroy: CuCtxDestroyFn = std::mem::transmute(cu_ctx_destroy_ptr);

        let cu_mem_alloc_ptr = libc::dlsym(lib, b"cuMemAlloc_v2\0".as_ptr() as *const libc::c_char);
        let cu_mem_alloc: CuMemAllocFn = std::mem::transmute(cu_mem_alloc_ptr);

        let cu_mem_free_ptr = libc::dlsym(lib, b"cuMemFree_v2\0".as_ptr() as *const libc::c_char);
        let cu_mem_free: CuMemFreeFn = std::mem::transmute(cu_mem_free_ptr);

        let cu_memcpy_dtod_ptr =
            libc::dlsym(lib, b"cuMemcpyDtoD_v2\0".as_ptr() as *const libc::c_char);
        let cu_memcpy_dtod: CuMemcpyDtoDFn = std::mem::transmute(cu_memcpy_dtod_ptr);

        // Initialize CUDA
        cu_init(0);

        // Get device 0
        let mut device: i32 = 0;
        cu_device_get(&mut device, 0);

        // Create Context 1 (for NvFBC)
        let mut context1: *mut std::ffi::c_void = ptr::null_mut();
        cu_ctx_create(&mut context1, 0x08, device);
        println!("Context 1 created: {:?}", context1);

        // Initialize and start capturer under Context 1
        let mut capturer = CudaCapturer::new()?;
        capturer.start(BufferFormat::Nv12, 30)?;
        let frame_info = capturer.next_frame(CaptureMethod::NoWait, None)?;
        println!(
            "Frame grabbed under Context 1, ptr = 0x{:X}",
            frame_info.device_buffer
        );

        // Create Context 2 (simulating FFmpeg context)
        let mut context2: *mut std::ffi::c_void = ptr::null_mut();
        cu_ctx_create(&mut context2, 0x08, device);
        println!("Context 2 created: {:?}", context2);

        // Allocate memory in Context 2
        let size = frame_info.device_buffer_len as usize;
        let mut dst_buffer: u64 = 0;
        let res = cu_mem_alloc(&mut dst_buffer, size);
        if res != 0 {
            panic!("cuMemAlloc in Context 2 failed: {}", res);
        }
        println!("Allocated buffer in Context 2: 0x{:X}", dst_buffer);

        // Copy from Context 1 buffer to Context 2 buffer
        // Note: cuCtxCreate automatically pushed Context 2, so Context 2 is now current.
        let res = cu_memcpy_dtod(dst_buffer, frame_info.device_buffer as u64, size);
        if res != 0 {
            println!("cuMemcpyDtoD cross-context failed with code: {}", res);
        } else {
            println!("cuMemcpyDtoD cross-context SUCCESS!");
        }

        // Clean up
        cu_mem_free(dst_buffer);
        capturer.stop()?;
        cu_ctx_destroy(context2);
        cu_ctx_destroy(context1);
    }

    Ok(())
}
