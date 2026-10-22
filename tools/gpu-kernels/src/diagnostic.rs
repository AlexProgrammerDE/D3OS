//! Read actual SM clock progress against the GPU's nanosecond timer.
use core::arch::asm;

#[inline(always)]
fn timers() -> (u64, u64) {
    let clock: u64;
    let time: u64;
    unsafe {
        asm!("mov.u64 {clock}, %clock64;", "mov.u64 {time}, %globaltimer;",
            clock = out(reg64) clock, time = out(reg64) time, options(nomem, nostack));
    }
    (clock, time)
}

/// # Safety
/// Output contains two aligned u64 words. Launch exactly one thread.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn clock_probe(output: u64) {
    if super::index() != 0 {
        return;
    }
    let (start_clock, start_time) = timers();
    let (end_clock, end_time) = loop {
        let current = timers();
        if current.1.wrapping_sub(start_time) >= 10_000_000 {
            break current;
        }
    };
    unsafe {
        (output as *mut u64).write(end_clock.wrapping_sub(start_clock));
        (output as *mut u64).add(1).write(end_time.wrapping_sub(start_time));
    }
}

/// # Safety
/// Output contains count words. This diagnostic deliberately uses a local
/// stack frame to validate the driver's metadata-derived scratch allocation.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn local_stack_probe(output: u64, count: u32) {
    let index = super::index();
    if index >= count {
        return;
    }
    let seed = index.wrapping_mul(0x9e3779b9);
    let mut scratch = [0u32; 16];
    let pointer = scratch.as_mut_ptr();
    for i in 0..16 {
        unsafe {
            pointer.add(i).write_volatile(seed ^ (i as u32).wrapping_mul(0x51a6d305));
        }
    }
    let mut checksum = 0u32;
    for i in 0..16 {
        checksum = checksum.wrapping_add(unsafe { pointer.add(((i * 7) as u32 + (seed & 15)) as usize % 16).read_volatile() });
    }
    unsafe {
        (output as *mut u32).add(index as usize).write(checksum);
    }
}
