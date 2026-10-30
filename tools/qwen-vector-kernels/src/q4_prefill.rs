//! Direct raw Q4_K register-MMA prefill for measured row counts.
use super::{_block_idx_x, _thread_idx_x, add, dot, mask_index, mul, shuffle, sub, widen};
use crate::q3_padded::mma;

#[inline(always)]
unsafe fn scale_min(weights: *const u32, base: u32, step: u32) -> (i32, i32) {
    let shift = (step & 3) * 8;
    let a = (unsafe { weights.add((base + 1) as usize).read() } >> shift) & 255;
    let b = (unsafe { weights.add((base + 2) as usize).read() } >> shift) & 255;
    let c = (unsafe { weights.add((base + 3) as usize).read() } >> shift) & 255;
    let scale = if step < 4 { a & 63 } else { (c & 15) | ((a >> 6) << 4) };
    let minimum = if step < 4 { b & 63 } else { (c >> 4) | ((b >> 6) << 4) };
    (scale as i32, minimum as i32)
}

#[inline(always)]
unsafe fn fragment(weights: *const u32, base: u32, step: u32, thread: u32) -> [u32; 2] {
    let at = base + 4 + (step / 2) * 8 + thread;
    let shift = (step & 1) * 4;
    [
        (unsafe { weights.add(at as usize).read() } >> shift) & 0x0f0f_0f0f,
        (unsafe { weights.add((at + 4) as usize).read() } >> shift) & 0x0f0f_0f0f,
    ]
}

/// Four independent M16/N8 warps, covering an M16/N32 output tile.
///
/// # Safety
/// Launch 128 threads. Inner is positive and divisible by 256. Activation
/// storage holds three row-major S8 planes followed by F32 scales per K32.
/// Weights hold cols * inner/256 complete 144-byte Q4_K blocks, aligned to
/// four bytes. Output holds rows * cols F32 values, disjoint from both inputs.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q4_warp_m16(
    activation: *const u32, weights: *const u32, output: *mut f32, rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32,
) {
    let thread = unsafe { _thread_idx_x() };
    let lane = thread & 31;
    let group = unsafe { mask_index(thread >> 2, 7) };
    let in_group = lane & 3;
    let tiles = cols.div_ceil(32);
    if tiles == 0 {
        return;
    }
    let block_index = unsafe { _block_idx_x() };
    let row = (block_index / tiles) * 16 + group;
    let column = (block_index % tiles) * 32 + (thread / 32) * 8;
    let b_column = column + group;
    let out_column = column + 2 * in_group;
    let row_words = inner / 4;
    let groups = inner / 32;
    let plane = rows * row_words;
    let blocks = inner / 256;
    let mut total = [0.0f32; 4];
    let mut correction = [0.0f32; 4];
    for block in 0..blocks {
        let bbase = (b_column * blocks + block) * 36;
        let base0 = (out_column * blocks + block) * 36;
        let base1 = ((out_column + 1) * blocks + block) * 36;
        let header0 = if out_column < cols {
            unsafe { weights.add(base0 as usize).read() }
        } else {
            0
        };
        let header1 = if out_column + 1 < cols {
            unsafe { weights.add(base1 as usize).read() }
        } else {
            0
        };
        let d0 = unsafe { widen(header0 as u16) };
        let d1 = unsafe { widen(header1 as u16) };
        let dm0 = unsafe { widen((header0 >> 16) as u16) };
        let dm1 = unsafe { widen((header1 >> 16) as u16) };
        for step in 0..8 {
            let b = if b_column < cols {
                unsafe { fragment(weights, bbase, step, in_group) }
            } else {
                [0; 2]
            };
            let (qs0, min0) = if out_column < cols {
                unsafe { scale_min(weights, base0, step) }
            } else {
                (0, 0)
            };
            let (qs1, min1) = if out_column + 1 < cols {
                unsafe { scale_min(weights, base1, step) }
            } else {
                (0, 0)
            };
            let k_group = block * 8 + step;
            let scale0 = if row < rows {
                f32::from_bits(unsafe { activation.add((3 * plane + row * groups + k_group) as usize).read() })
            } else {
                0.0
            };
            let scale1 = if row + 8 < rows {
                f32::from_bits(unsafe { activation.add((3 * plane + (row + 8) * groups + k_group) as usize).read() })
            } else {
                0.0
            };
            for (component, divisor) in [(0, 1.0f32), (1, 1.0f32 / 128.0), (2, 1.0f32 / 16384.0)] {
                let base = component * plane + row * row_words + k_group * 8 + in_group;
                let a0 = if row < rows { unsafe { activation.add(base as usize).read() } } else { 0 };
                let a1 = if row + 8 < rows {
                    unsafe { activation.add((base + 8 * row_words) as usize).read() }
                } else {
                    0
                };
                let a2 = if row < rows {
                    unsafe { activation.add((base + 4) as usize).read() }
                } else {
                    0
                };
                let a3 = if row + 8 < rows {
                    unsafe { activation.add((base + 8 * row_words + 4) as usize).read() }
                } else {
                    0
                };
                let products = unsafe { mma([a0, a1, a2, a3], b) };
                let mut sum0 = unsafe { dot(a0, 0x0101_0101, dot(a2, 0x0101_0101, 0)) };
                let mut sum1 = unsafe { dot(a1, 0x0101_0101, dot(a3, 0x0101_0101, 0)) };
                for shift in [1, 2] {
                    sum0 += unsafe { shuffle(sum0 as u32, shift) } as i32;
                    sum1 += unsafe { shuffle(sum1 as u32, shift) } as i32;
                }
                // 32 * 15 * 128 * 63 bounds every scaled dot by 3,870,720.
                // The minimum correction is bounded by 258,048. Both fit I32.
                let positives = [products[0] * qs0, products[1] * qs1, products[2] * qs0, products[3] * qs1];
                let negatives = [sum0 * min0, sum0 * min1, sum1 * min0, sum1 * min1];
                let unit0 = unsafe { mul(scale0, divisor) };
                let unit1 = unsafe { mul(scale1, divisor) };
                for index in 0..4 {
                    let unit = if index < 2 { unit0 } else { unit1 };
                    let (d, dm) = if index & 1 == 0 { (d0, dm0) } else { (d1, dm1) };
                    let positive = unsafe { mul(mul(positives[index] as f32, unit), d) };
                    let negative = unsafe { mul(mul(negatives[index] as f32, unit), dm) };
                    let value = unsafe { sub(positive, negative) };
                    let adjusted = unsafe { sub(value, correction[index]) };
                    let next = unsafe { add(total[index], adjusted) };
                    correction[index] = unsafe { sub(sub(next, total[index]), adjusted) };
                    total[index] = next;
                }
            }
        }
    }
    for index in 0..4 {
        let r = row + if index >= 2 { 8 } else { 0 };
        let c = out_column + (index as u32 & 1);
        if r < rows && c < cols {
            unsafe { output.add((r * cols + c) as usize).write(total[index]) };
        }
    }
}
