//! Validate process memory counters and the writable syscall result range.
use alloc::{format, string::String};
use d3os_compute::nvidia::Nvidia;
use d3os_nvidia::compute as abi;
use syscall::{SystemCall, return_vals::Errno, syscall};

fn packed_upload(backend: &Nvidia) -> d3os_compute::Result<()> {
    let mut source = [0u8; 69];
    for (index, byte) in source.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(117).wrapping_add(31);
    }
    // Exercise byte pointers that cannot be represented as aligned u32 slices.
    let start = usize::from(source.as_ptr().addr().is_multiple_of(4));
    for length in [1usize, 2, 3, 4, 5, 7, 8, 13, 17, 64] {
        let bytes = &source[start..start + length];
        let buffer = backend.upload_bytes(bytes)?;
        let mut actual = [0u32; 16];
        backend.read_words_into(&buffer, 0, &mut actual[..length.div_ceil(4)])?;
        for (word, chunk) in actual.iter().zip(bytes.chunks(4)) {
            let mut expected = [0; 4];
            expected[..chunk.len()].copy_from_slice(chunk);
            if *word != u32::from_le_bytes(expected) {
                return Err(d3os_compute::Error::DeviceFault);
            }
        }
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let verify = || -> d3os_compute::Result<()> {
        packed_upload(backend)?;
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
