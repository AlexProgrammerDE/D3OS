//! CPU copies while coherent DMA storage is exclusively owned by the CPU.

/// Copy an owned memory range without letting the compiler remove or reuse
/// accesses to storage that an external DMA agent can modify between calls.
/// On x86, REP MOVSB avoids both a scalar volatile loop and SIMD registers.
///
/// # Safety
/// Both ranges must be valid for `bytes` and must not overlap. The destination
/// must be writable. The CPU must own both ranges for the complete copy, with
/// DMA completion and publication barriers supplied by the caller. Neither
/// range may be MMIO. The x86 calling convention requires a clear direction flag.
pub unsafe fn copy_owned(destination: *mut u8, source: *const u8, bytes: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // Omitting nomem/readonly gives this volatile assembly a compiler memory
        // barrier. REP MOVSB changes RCX, RSI, and RDI, but preserves flags.
        core::arch::asm!(
            "rep movsb",
            inout("rdi") destination => _,
            inout("rsi") source => _,
            inout("rcx") bytes => _,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    for index in 0..bytes {
        unsafe {
            destination.add(index).write_volatile(source.add(index).read_volatile());
        }
    }
}
