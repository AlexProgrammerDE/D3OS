//! Validate process memory counters and the writable syscall result range.
use alloc::{format, string::String};
use d3os_compute::nvidia::Nvidia;
use d3os_nvidia::compute as abi;
use syscall::{SystemCall, return_vals::Errno, syscall};

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let verify = || -> d3os_compute::Result<()> {
        let before = backend.memory_profile()?;
        let buffer = backend.allocate_words(513)?;
        let allocated = backend.memory_profile()?;
        if allocated.process_live_bytes != before.process_live_bytes + 4096 {
            return Err(d3os_compute::Error::DeviceFault);
        }
        drop(buffer);
        if backend.memory_profile()?.process_live_bytes != before.process_live_bytes {
            return Err(d3os_compute::Error::DeviceFault);
        }
        let mut guarded = [0x6bad_beef_dead_cafeu64; 5];
        let mut request = abi::Request {
            version: abi::VERSION,
            operation: abi::MEMORY_PROFILE,
            elements: 1,
            pointer: guarded[1..].as_mut_ptr() as u64,
            ..abi::Request::default()
        };
        syscall(SystemCall::GpuCompute, &[&request as *const abi::Request as usize]).map_err(|_| d3os_compute::Error::DeviceFault)?;
        if guarded[0] != 0x6bad_beef_dead_cafe || guarded[4] != 0x6bad_beef_dead_cafe || guarded[1] != before.process_live_bytes {
            return Err(d3os_compute::Error::DeviceFault);
        }
        let untouched = guarded;
        request.elements = 2;
        if syscall(SystemCall::GpuCompute, &[&request as *const abi::Request as usize]) != Err(Errno::EINVAL) || guarded != untouched {
            return Err(d3os_compute::Error::DeviceFault);
        }
        request.elements = 1;
        request.pointer = 1;
        if syscall(SystemCall::GpuCompute, &[&request as *const abi::Request as usize]) != Err(Errno::EINVAL) {
            return Err(d3os_compute::Error::DeviceFault);
        }
        Ok(())
    };
    verify().map_err(|e| format!("Burn memory counters: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_MEMORY_OK owner=process allocation=rounded free=exact pointer_guards=ok backend=nvidia"
    ));
    Ok(())
}
