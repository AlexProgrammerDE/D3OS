//! Q3_K prefill with three signed activation residuals and warp-register MMA.
use super::{add, byte, dot, mask_index, mul, shuffle, sub, widen, word};
use core::arch::{
    asm,
    nvptx::{_block_idx_x, _thread_idx_x},
};

#[inline(always)]
fn scaled_q3_word(code: u32, scale: i32) -> u32 {
    // Each input byte is q3 + 4 in 0..=7; scale is in -32..=31.
    // Both packed operations remain carry-free for the entire legal domain.
    let magnitude = scale.unsigned_abs();
    let (unsigned, offset) = if scale < 0 {
        (0x0707_0707 - code, 127 - 3 * magnitude)
    } else {
        (code, 127 - 4 * magnitude)
    };
    (unsigned * magnitude + offset * 0x0101_0101) ^ 0x8080_8080
}

// A's registers cover (row g, row g+8) at K offsets 0 then 16.
// B's registers cover column g at K offsets 0 and 16. Every warp lane
// executes the collective even when its fragment or result is outside a tail.
#[inline(always)]
unsafe fn mma(a: [u32; 4], b: [u32; 2]) -> [i32; 4] {
    let d0: i32;
    let d1: i32;
    let d2: i32;
    let d3: i32;
    unsafe {
        asm!(
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{0, 0, 0, 0}};",
            d0=out(reg32) d0, d1=out(reg32) d1, d2=out(reg32) d2, d3=out(reg32) d3,
            a0=in(reg32) a[0], a1=in(reg32) a[1], a2=in(reg32) a[2], a3=in(reg32) a[3],
            b0=in(reg32) b[0], b1=in(reg32) b[1], options(nomem, nostack)
        );
    }
    [d0, d1, d2, d3]
}

#[inline(always)]
unsafe fn weight_fragment(p: *const u16, step: u32, thread: u32) -> [u32; 2] {
    let qindex = (32 + (step / 4) * 32 + 4 * thread) / 2;
    let hindex = 2 * thread;
    let mut result = [0; 2];
    for half in 0..2 {
        let low = unsafe { word(p, qindex + half * 8) };
        let high = unsafe { word(p, hindex + half * 8) };
        let code = ((low >> (2 * (step % 4))) & 0x0303_0303) | (((high >> step) & 0x0101_0101) << 2);
        let subgroup = 2 * step + half;
        let sl = unsafe { byte(p, 96 + subgroup % 8) } >> ((subgroup / 8) * 4);
        let sh = unsafe { byte(p, 104 + subgroup % 4) } >> ((subgroup / 4) * 2);
        let scale = ((sl & 15) | ((sh & 3) << 4)) as i32 - 32;
        result[half as usize] = scaled_q3_word(code, scale);
    }
    result
}

/// Four independent M16/N8 warps per CTA, covering an M16/N32 output tile.
///
/// # Safety
/// Launch 128 threads. `inner` is positive and divisible by 256. Activation
/// storage contains three row-major S8 planes followed by F32 scales per K32.
/// Weights contain `cols * inner / 256` complete Q3_K blocks and are aligned
/// to two bytes. Output has `rows * cols` F32 elements and overlaps no input.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q3_warp_m16(
    activation: *const u32, weights: *const u16, output: *mut f32, rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32,
) {
    let thread = unsafe { _thread_idx_x() };
    let group = unsafe { mask_index(thread >> 2, 7) };
    let lane = thread & 31;
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
    let mut sum = [0.0f32; 4];
    let mut correction = [0.0f32; 4];
    for block in 0..blocks {
        let bbase = (b_column * blocks + block) * 55;
        let d0 = if out_column < cols {
            unsafe { widen(weights.add(((out_column * blocks + block) * 55 + 54) as usize).read()) }
        } else {
            0.0
        };
        let d1 = if out_column + 1 < cols {
            unsafe { widen(weights.add((((out_column + 1) * blocks + block) * 55 + 54) as usize).read()) }
        } else {
            0.0
        };
        for step in 0..8 {
            let b = if b_column < cols {
                unsafe { weight_fragment(weights.add(bbase as usize), step, in_group) }
            } else {
                [0; 2]
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
                let mut activation_sum0 = unsafe { dot(a0, 0x0101_0101, dot(a2, 0x0101_0101, 0)) };
                let mut activation_sum1 = unsafe { dot(a1, 0x0101_0101, dot(a3, 0x0101_0101, 0)) };
                for shift in [1, 2] {
                    activation_sum0 += unsafe { shuffle(activation_sum0 as u32, shift) } as i32;
                    activation_sum1 += unsafe { shuffle(activation_sum1 as u32, shift) } as i32;
                }
                let values = [
                    unsafe { mul(mul((products[0] + activation_sum0) as f32, mul(scale0, divisor)), d0) },
                    unsafe { mul(mul((products[1] + activation_sum0) as f32, mul(scale0, divisor)), d1) },
                    unsafe { mul(mul((products[2] + activation_sum1) as f32, mul(scale1, divisor)), d0) },
                    unsafe { mul(mul((products[3] + activation_sum1) as f32, mul(scale1, divisor)), d1) },
                ];
                for index in 0..4 {
                    let adjusted = unsafe { sub(values[index], correction[index]) };
                    let next = unsafe { add(sum[index], adjusted) };
                    correction[index] = unsafe { sub(sub(next, sum[index]), adjusted) };
                    sum[index] = next;
                }
            }
        }
    }
    for index in 0..4 {
        let r = row + if index >= 2 { 8 } else { 0 };
        let c = out_column + (index as u32 & 1);
        if r < rows && c < cols {
            unsafe { output.add((r * cols + c) as usize).write(sum[index]) };
        }
    }
}
