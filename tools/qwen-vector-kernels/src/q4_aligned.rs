//! Aligned Q4 loads with the original component arithmetic and reduction.
use super::{add, dot, mask_index, mul, shuffle, sub, widen};
use core::arch::nvptx::{_block_idx_x, _thread_idx_x};

/// # Safety
/// Launch 128 threads for one row, with an inner extent divisible by 256.
/// Weight pointers must be eight-byte aligned.
/// Supply three signed-byte component planes followed by one F32 scale per
/// 32 inputs, complete Q4_K weight blocks, and a disjoint F32 output buffer.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q4_components_wide64(
    activation: *const u32, weights: *const u32, output: *mut f32, _rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32,
) {
    let thread = unsafe { _thread_idx_x() };
    let lane = thread & 31;
    let col = unsafe { _block_idx_x() } * 4 + thread / 32;
    if col >= cols {
        return;
    }
    let blocks = inner / 256;
    let group = lane / 4;
    let subgroup_index = unsafe { mask_index(group, 3) };
    let offset = (lane * 8) & 31;
    let qshift = lane & 4;
    let plane = inner / 4;
    let mut sum = 0.0f32;
    let mut correction = 0.0f32;
    for block in 0..blocks {
        let base = (col * blocks + block) * 36;
        let qindex = base + (16 + (lane / 8) * 32 + offset) / 4;
        let bits = unsafe { weights.add(qindex as usize).cast::<u64>().read() };
        let q0 = (bits as u32 >> qshift) & 0x0f0f0f0f;
        let q1 = ((bits >> 32) as u32 >> qshift) & 0x0f0f0f0f;
        let shift = subgroup_index * 8;
        let a = unsafe { (weights.add((base + 1) as usize).read() >> shift) & 255 };
        let b = unsafe { (weights.add((base + 2) as usize).read() >> shift) & 255 };
        let c = unsafe { (weights.add((base + 3) as usize).read() >> shift) & 255 };
        let scale = if group < 4 { a & 63 } else { (c & 15) | ((a >> 6) << 4) } as i32;
        let minimum = if group < 4 { b & 63 } else { (c >> 4) | ((b >> 6) << 4) } as i32;
        let scales = unsafe { weights.add(base as usize).read() };
        let d = unsafe { widen(scales as u16) };
        let dmin = unsafe { widen((scales >> 16) as u16) };
        let ai = block * 64 + lane * 2;
        let a_scale = f32::from_bits(unsafe { activation.add((3 * plane + block * 8 + lane / 4) as usize).read() });
        for (component, divisor) in [(0, 1.0f32), (1, 1.0f32 / 128.0), (2, 1.0f32 / 16384.0)] {
            let a = unsafe { activation.add((component * plane + ai) as usize) };
            let a0 = unsafe { a.read() };
            let a1 = unsafe { a.add(1).read() };
            let integer = unsafe { dot(q0, a0, 0) };
            let integer = unsafe { dot(q1, a1, integer) };
            let asum = unsafe { dot(0x01010101, a0, 0) };
            let asum = unsafe { dot(0x01010101, a1, asum) };
            let unit = unsafe { mul(a_scale, divisor) };
            let positive = unsafe { mul(mul((integer * scale) as f32, unit), d) };
            let negative = unsafe { mul(mul((asum * minimum) as f32, unit), dmin) };
            let value = unsafe { sub(positive, negative) };
            let adjusted = unsafe { sub(value, correction) };
            let next = unsafe { add(sum, adjusted) };
            correction = unsafe { sub(sub(next, sum), adjusted) };
            sum = next;
        }
    }
    for shift in [16, 8, 4, 2, 1] {
        sum = unsafe { add(sum, f32::from_bits(shuffle(sum.to_bits(), shift))) };
    }
    if lane == 0 {
        unsafe { output.add(col as usize).write(sum) };
    }
}
