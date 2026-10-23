//! Circular completion timelines, matching NVIDIA's 64-bit semaphore acquire.

/// Whether `observed` completes `expected`, including counter wraparound.
/// Reports must advance regularly, with less than half the counter space
/// between any outstanding fence and its report.
pub fn reached(observed: u64, expected: u64) -> bool {
    observed.wrapping_sub(expected) as i64 >= 0
}
