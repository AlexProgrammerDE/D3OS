//! Warp collectives must execute in every participating lane.
use core::arch::asm;

#[inline(always)]
unsafe fn add_partner<const SHIFT: u32>(value: f32) -> f32 {
    let other: f32;
    // `pure` permits dead-lane elimination; `nomem` rules out synchronization.
    // Neither option describes a warp collective. Immediate shifts also avoid
    // materializing an iterator's shift table in GPU local memory.
    unsafe {
        asm!("shfl.sync.bfly.b32 {d}, {a}, {shift}, 31, -1;",
            d = out(reg32) other, a = in(reg32) value, shift = const SHIFT,
            options(nostack));
    }
    value + other
}

/// # Safety
/// Every lane in the full warp executes this function in the same order.
#[inline(always)]
pub(super) unsafe fn sum(mut value: f32) -> f32 {
    value = unsafe { add_partner::<16>(value) };
    value = unsafe { add_partner::<8>(value) };
    value = unsafe { add_partner::<4>(value) };
    value = unsafe { add_partner::<2>(value) };
    unsafe { add_partner::<1>(value) }
}

/// # Safety
/// Every lane in the full warp executes this function in the same order.
#[inline(always)]
pub(super) unsafe fn broadcast(value: f32) -> f32 {
    let result: f32;
    unsafe {
        asm!("shfl.sync.idx.b32 {d}, {a}, 0, 31, -1;",
            d = out(reg32) result, a = in(reg32) value, options(nostack));
    }
    result
}

#[inline(always)]
unsafe fn max_partner<const SHIFT: u32>(value: f32) -> f32 {
    let other: f32;
    unsafe {
        asm!("shfl.sync.bfly.b32 {d}, {a}, {shift}, 31, -1;",
            d = out(reg32) other, a = in(reg32) value, shift = const SHIFT,
            options(nostack));
    }
    value.max(other)
}

/// # Safety
/// Every lane in the full warp executes this function in the same order.
#[inline(always)]
pub(super) unsafe fn maximum(mut value: f32) -> f32 {
    value = unsafe { max_partner::<16>(value) };
    value = unsafe { max_partner::<8>(value) };
    value = unsafe { max_partner::<4>(value) };
    value = unsafe { max_partner::<2>(value) };
    unsafe { max_partner::<1>(value) }
}
