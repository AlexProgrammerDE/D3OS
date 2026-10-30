//! Q3_K consumers sharing one 112-byte aligned block layout.
use super::{add, dot, mask_index, mul, shuffle, signed_q3, sub, widen};
use core::arch::{
    asm,
    nvptx::{_block_idx_x, _thread_idx_x},
};

#[inline(always)]
unsafe fn pair(p: *const u32, index: u32) -> (u32, u32) {
    // The padded block base and every call-site index are eight-byte aligned.
    let bits = unsafe { p.add(index as usize).cast::<u64>().read() };
    (bits as u32, (bits >> 32) as u32)
}
// One warp per output, four warps per CTA. Rows must equal one. Each lane
// handles eight adjacent values, sharing one subgroup scale and A scale.
// Subgroup scale follows the integer dot, avoiding S8 scale-product overflow.
/// # Safety
/// Launch 128 threads with one row and a nonzero `inner` divisible by 256.
/// Activation storage must contain three byte planes and one F32 scale per 32
/// inputs. Weights must be eight-byte aligned and contain `cols * (inner / 256)`
/// Q3_K blocks padded from 110 to 112 bytes. Padding has no numerical meaning. Output must
/// hold `cols` F32 values without overlapping either input allocation.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q3_components_padded112(
    activation: *const u32, weights: *const u32, output: *mut f32, _rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32,
) {
    let thread = unsafe { _thread_idx_x() };
    let lane = thread & 31;
    let col = unsafe { _block_idx_x() } * 4 + thread / 32;
    if col >= cols {
        return;
    }
    let blocks = inner / 256;
    let sg = lane / 2;
    let offset = (lane * 8) & 31;
    let qshift = 2 * ((lane / 4) & 3);
    let hshift = lane / 4;
    let plane = inner / 4;
    let mut sum = 0.0f32;
    let mut correction = 0.0f32;
    for block in 0..blocks {
        let base = (col * blocks + block) * 28;
        let qindex = base + (32 + (lane / 16) * 32 + offset) / 4;
        let hindex = base + offset / 4;
        let (low0, low1) = unsafe { pair(weights, qindex) };
        let (high0, high1) = unsafe { pair(weights, hindex) };
        let q0 = signed_q3(low0, high0, qshift, hshift);
        let q1 = signed_q3(low1, high1, qshift, hshift);
        let low_word = unsafe { weights.add((base + 24 + mask_index(sg, 7) / 4) as usize).read() };
        let high_word = unsafe { weights.add((base + 26) as usize).read() };
        let sl = low_word >> (unsafe { mask_index(sg, 3) } * 8 + (sg / 8) * 4);
        let sh = high_word >> (unsafe { mask_index(sg, 3) } * 8 + (sg / 4) * 2);
        let scale = ((sl & 15) | ((sh & 3) << 4)) as i32 - 32;
        let d = unsafe { widen(weights.add((base + 27) as usize).read() as u16) };
        let ai = block * 64 + lane * 2;
        let a_scale = f32::from_bits(unsafe { activation.add((3 * plane + block * 8 + lane / 4) as usize).read() });
        for (component, divisor) in [(0, 1.0f32), (1, 1.0f32 / 128.0), (2, 1.0f32 / 16384.0)] {
            let a = unsafe { activation.add((component * plane + ai) as usize) };
            let integer = unsafe { dot(q0, a.read(), 0) };
            let integer = unsafe { dot(q1, a.add(1).read(), integer) };
            let value = unsafe { mul(mul((integer * scale) as f32, mul(a_scale, divisor)), d) };
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

#[inline(always)]
unsafe fn padded_word(p: *const u32, index: u32) -> u32 {
    unsafe { p.add(index as usize).read() }
}
#[inline(always)]
unsafe fn padded_byte(p: *const u32, index: u32) -> u32 {
    let word = unsafe { padded_word(p, index / 4) };
    unsafe { mask_index(word >> (mask_index(index, 3) * 8), 255) }
}
#[inline(always)]
unsafe fn fma(a: f32, b: f32, c: f32) -> f32 {
    let out: f32;
    unsafe {
        asm!("fma.rn.f32 {out}, {a}, {b}, {c};", out=out(reg32) out, a=in(reg32) a, b=in(reg32) b, c=in(reg32) c, options(nomem, nostack, pure));
    }
    out
}
#[inline(always)]
unsafe fn bounded_shift(value: u32, shift: u32) -> u32 {
    // Every caller passes the Q3 group index in0..=7.
    let out: u32;
    unsafe {
        asm!("shr.u32 {out}, {value}, {shift};", out=out(reg32) out, value=in(reg32) value, shift=in(reg32) shift, options(nomem, nostack, pure));
    }
    out
}
#[inline(always)]
unsafe fn padded_value(weights: *const u32, block: u32, offset: u32) -> f32 {
    let p = unsafe { weights.add((block * 28) as usize) };
    let low = unsafe { padded_byte(p, 32 + (offset / 128) * 32 + mask_index(offset, 31)) };
    let high = unsafe { padded_byte(p, mask_index(offset, 31)) };
    let group = offset / 32;
    let code = unsafe { mask_index(low >> (2 * mask_index(group, 3)), 3) };
    let negative = (unsafe { mask_index(bounded_shift(high, group), 1) } ^ 1) * 4;
    let q = code as i32 - negative as i32;
    let sg = offset / 16;
    let sl = unsafe { padded_byte(p, 96 + mask_index(sg, 7)) } >> ((sg / 8) * 4);
    let sh = unsafe { padded_byte(p, 104 + mask_index(sg, 3)) } >> ((sg / 4) * 2);
    let scale = ((sl & 15) | ((sh & 3) << 4)) as i32 - 32;
    let d = unsafe { widen(p.add(27).read() as u16) };
    unsafe { mul(mul(d, scale as f32), q as f32) }
}

// Accurate scalar-vector consumer: the original four accumulators, eight
// FMA positions per block and full warp reduction remain explicit.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q3_default_padded112(
    input: *const f32, weights: *const u32, output: *mut f32, _rows: u32, inner: u32, cols: u32, _format: u32, _layout: u32, _components: u32,
) {
    let thread = unsafe { _thread_idx_x() };
    let lane = thread & 31;
    let col = unsafe { _block_idx_x() } * 4 + thread / 32;
    if col >= cols {
        return;
    }
    let blocks = inner / 256;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    for block in 0..blocks {
        let weight_block = col * blocks + block;
        let start = block * 256 + lane;
        s0 = unsafe { fma(input.add((start + 0) as usize).read(), padded_value(weights, weight_block, lane + 0), s0) };
        s1 = unsafe { fma(input.add((start + 32) as usize).read(), padded_value(weights, weight_block, lane + 32), s1) };
        s2 = unsafe { fma(input.add((start + 64) as usize).read(), padded_value(weights, weight_block, lane + 64), s2) };
        s3 = unsafe { fma(input.add((start + 96) as usize).read(), padded_value(weights, weight_block, lane + 96), s3) };
        s0 = unsafe { fma(input.add((start + 128) as usize).read(), padded_value(weights, weight_block, lane + 128), s0) };
        s1 = unsafe { fma(input.add((start + 160) as usize).read(), padded_value(weights, weight_block, lane + 160), s1) };
        s2 = unsafe { fma(input.add((start + 192) as usize).read(), padded_value(weights, weight_block, lane + 192), s2) };
        s3 = unsafe { fma(input.add((start + 224) as usize).read(), padded_value(weights, weight_block, lane + 224), s3) };
    }
    let mut sum = unsafe { add(add(s0, s1), add(s2, s3)) };
    // cuTile transposes four output warps before reducing each eight-lane
    // subgroup, then merges those groups. Match its logical lane tree.
    for shift in [4, 2, 1, 16, 8] {
        sum = unsafe { add(sum, f32::from_bits(shuffle(sum.to_bits(), shift))) };
    }
    if lane == 0 {
        unsafe { output.add(col as usize).write(sum) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn embedding_q3_padded112(
    weights: *const u32, tokens: *const u32, output: *mut f32, vocab: u32, width: u32, rows: u32, _format: u32, _layout: u32,
) {
    if width == 0 {
        return;
    }
    let index = unsafe { _block_idx_x() * 128 + _thread_idx_x() };
    if index >= rows * width {
        return;
    }
    let row = index / width;
    let col = index % width;
    let token = unsafe { tokens.add(row as usize).read() };
    let value = if token < vocab {
        unsafe { padded_value(weights, token * (width / 256) + col / 256, mask_index(col, 255)) }
    } else {
        0.0
    };
    unsafe { output.add(index as usize).write(value) };
}

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
unsafe fn weight_fragment(p: *const u32, step: u32, thread: u32) -> [u32; 2] {
    let qindex = (32 + (step / 4) * 32 + 4 * thread) / 4;
    let hindex = thread;
    let mut result = [0; 2];
    for half in 0..2 {
        let low = unsafe { padded_word(p, qindex + half * 4) };
        let high = unsafe { padded_word(p, hindex + half * 4) };
        let code = ((low >> (2 * (step % 4))) & 0x0303_0303) | (((high >> step) & 0x0101_0101) << 2);
        let subgroup = 2 * step + half;
        let sl = unsafe { padded_byte(p, 96 + subgroup % 8) } >> ((subgroup / 8) * 4);
        let sh = unsafe { padded_byte(p, 104 + subgroup % 4) } >> ((subgroup / 4) * 2);
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
/// Weights contain `cols * inner / 256` complete 112-byte padded Q3_K blocks and are aligned
/// to four bytes. Output has `rows * cols` F32 elements and overlaps no input.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q3_warp_padded112(
    activation: *const u32, weights: *const u32, output: *mut f32, rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32,
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
        let bbase = (b_column * blocks + block) * 28;
        let d0 = if out_column < cols {
            unsafe { widen(weights.add(((out_column * blocks + block) * 28 + 27) as usize).read() as u16) }
        } else {
            0.0
        };
        let d1 = if out_column + 1 < cols {
            unsafe { widen(weights.add((((out_column + 1) * blocks + block) * 28 + 27) as usize).read() as u16) }
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
