//! Two sixteen-lane output groups per warp on padded Q3_K decode.
use super::{_block_idx_x, _thread_idx_x, add, dot, mask_index, mul, shuffle, sub, widen};
use core::arch::asm;

#[inline(always)]
unsafe fn bounded_shift(value: u32, shift: u32) -> u32 {
    let out: u32;
    unsafe {
        asm!("shr.u32 {out}, {value}, {shift};", out=out(reg32) out,value=in(reg32) value,shift=in(reg32) shift,options(nomem,nostack,pure));
    }
    out
}
#[inline(always)]
fn signed_q3(low: u32, high: u32, qshift: u32, hshift: u32) -> u32 {
    let low = (unsafe { bounded_shift(low, qshift) }) & 0x03030303;
    let negative = ((unsafe { bounded_shift(high, hshift) }) & 0x01010101) ^ 0x01010101;
    low | negative.wrapping_mul(252)
}

#[inline(always)]
unsafe fn pair(p: *const u32, index: u32) -> (u32, u32) {
    // The padded block base and every call-site index are eight-byte aligned.
    let bits = unsafe { p.add(index as usize).cast::<u64>().read() };
    (bits as u32, (bits >> 32) as u32)
}

// Two outputs per warp, one subgroup per lane. Tail lanes read column zero
// and remain in every collective; they never publish outside the output.
/// # Safety
/// Launch 128 threads, one row, positive inner divisible by 256, and positive
/// columns. Weights contain complete 112-byte Q3 blocks aligned to eight bytes.
/// Activation storage holds three S8 planes and one F32 scale per K32. Output
/// contains cols F32 values and is disjoint from both inputs.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q3_components_halfwarp(
    activation: *const u32, weights: *const u32, output: *mut f32, _rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32,
) {
    if cols == 0 {
        return;
    }
    let thread = unsafe { _thread_idx_x() };
    let lane = unsafe { mask_index(thread, 15) };
    let col = unsafe { _block_idx_x() } * 8 + thread / 16;
    let source_col = if col < cols { col } else { 0 };
    let blocks = inner / 256;
    let offset = unsafe { mask_index(lane * 16, 31) };
    let qshift = 2 * unsafe { mask_index(lane / 2, 3) };
    let hshift = lane / 2;
    let plane = inner / 4;
    let mut sum = 0.0f32;
    let mut correction = 0.0f32;
    for block in 0..blocks {
        let base = (source_col * blocks + block) * 28;
        let qi = base + (32 + (lane / 8) * 32 + offset) / 4;
        let hi = base + offset / 4;
        let (lo0, lo1) = unsafe { pair(weights, qi) };
        let (lo2, lo3) = unsafe { pair(weights, qi + 2) };
        let (hi0, hi1) = unsafe { pair(weights, hi) };
        let (hi2, hi3) = unsafe { pair(weights, hi + 2) };
        let q0 = signed_q3(lo0, hi0, qshift, hshift);
        let q1 = signed_q3(lo1, hi1, qshift, hshift);
        let q2 = signed_q3(lo2, hi2, qshift, hshift);
        let q3 = signed_q3(lo3, hi3, qshift, hshift);
        let low = unsafe { weights.add((base + 24 + mask_index(lane, 7) / 4) as usize).read() };
        let (high, header) = unsafe { pair(weights, base + 26) };
        let sl = low >> (unsafe { mask_index(lane, 3) } * 8 + (lane / 8) * 4);
        let sh = high >> (unsafe { mask_index(lane, 3) } * 8 + (lane / 4) * 2);
        let scale = ((sl & 15) | ((sh & 3) << 4)) as i32 - 32;
        let d = unsafe { widen(header as u16) };
        let ai = block * 64 + lane * 4;
        let a_scale = f32::from_bits(unsafe { activation.add((3 * plane + block * 8 + lane / 2) as usize).read() });
        for (component, divisor) in [(0, 1.0f32), (1, 1.0f32 / 128.0), (2, 1.0f32 / 16384.0)] {
            let a = unsafe { activation.add((component * plane + ai) as usize) };
            let v = unsafe { dot(q0, a.read(), 0) };
            let v = unsafe { dot(q1, a.add(1).read(), v) };
            let v = unsafe { dot(q2, a.add(2).read(), v) };
            let v = unsafe { dot(q3, a.add(3).read(), v) };
            let value = unsafe { mul(mul((v * scale) as f32, mul(a_scale, divisor)), d) };
            let adjusted = unsafe { sub(value, correction) };
            let next = unsafe { add(sum, adjusted) };
            correction = unsafe { sub(sub(next, sum), adjusted) };
            sum = next;
        }
    }
    for shift in [8, 4, 2, 1] {
        sum = unsafe { add(sum, f32::from_bits(shuffle(sum.to_bits(), shift))) };
    }
    if lane == 0 && col < cols {
        unsafe { output.add(col as usize).write(sum) };
    }
}
