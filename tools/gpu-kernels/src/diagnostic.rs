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
