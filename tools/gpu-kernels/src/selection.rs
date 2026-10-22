//! Stable vocabulary reduction with bounded host transfer.
use core::arch::asm;

#[inline(always)]
unsafe fn partner<const SHIFT: u32>(index: u32, value: f32, invalid: u32) -> (u32, f32, u32) {
    let (other_index, other_value, other_invalid): (u32, f32, u32);
    unsafe {
        asm!(
            "shfl.sync.bfly.b32 {i}, {a}, {shift}, 31, -1;",
            "shfl.sync.bfly.b32 {v}, {b}, {shift}, 31, -1;",
            "shfl.sync.bfly.b32 {n}, {c}, {shift}, 31, -1;",
            i = out(reg32) other_index, v = out(reg32) other_value, n = out(reg32) other_invalid,
            a = in(reg32) index, b = in(reg32) value, c = in(reg32) invalid,
            shift = const SHIFT, options(nostack)
        );
    }
    let bad = invalid | other_invalid;
    if other_value > value || (other_value == value && other_index < index) {
        (other_index, other_value, bad)
    } else {
        (index, value, bad)
    }
}

/// # Safety
/// Input contains count f32 values. Output contains 32 pairs of u32 index/value
/// bits and cannot overlap input. Launch exactly 1024 threads in full warps.
/// An empty warp writes index MAX; any non-finite input writes index 0 and NaN.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn argmax_partials(input: u64, output: u64, count: u32) {
    let thread = super::index();
    let mut cursor = u64::from(thread);
    let mut best = (u32::MAX, f32::NEG_INFINITY, 0u32);
    while cursor < u64::from(count) {
        let value = unsafe { (input as *const f32).add(cursor as usize).read() };
        if !value.is_finite() {
            best.2 = 1;
        } else if value > best.1 || (value == best.1 && (cursor as u32) < best.0) {
            best.0 = cursor as u32;
            best.1 = value;
        }
        cursor += 1024;
    }
    best = unsafe { partner::<16>(best.0, best.1, best.2) };
    best = unsafe { partner::<8>(best.0, best.1, best.2) };
    best = unsafe { partner::<4>(best.0, best.1, best.2) };
    best = unsafe { partner::<2>(best.0, best.1, best.2) };
    best = unsafe { partner::<1>(best.0, best.1, best.2) };
    if thread.is_multiple_of(32) {
        let (index, value) = if best.2 != 0 { (0, f32::NAN) } else { (best.0, best.1) };
        let pair = (output as *mut u32).wrapping_add((thread / 32 * 2) as usize);
        unsafe {
            pair.write(index);
            pair.add(1).write(value.to_bits());
        }
    }
}
