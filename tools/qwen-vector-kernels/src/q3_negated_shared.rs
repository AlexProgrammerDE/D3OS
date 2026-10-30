//! Shared Q3 prefill with exact negated S8 weights and no activation-sum correction.
use super::{_block_idx_x, _thread_idx_x, add, mask_index, mul, sub, widen};
use crate::q3_padded::mma;
use core::arch::asm;

#[inline(always)]
unsafe fn shared_address() -> u32 {
    let address: u32;
    unsafe {
        asm!(".shared .align 16 .b8 q3_tile[3584]; mov.u32 {address}, q3_tile;", address=out(reg32) address, options(nostack));
    }
    address
}

#[inline(always)]
unsafe fn store_shared(address: u32, value: u32) {
    unsafe {
        asm!("st.shared.u32 [{address}], {value};", address=in(reg32) address, value=in(reg32) value, options(nostack));
    }
}

#[inline(always)]
unsafe fn load_shared(address: u32) -> u32 {
    let value: u32;
    unsafe {
        asm!("ld.shared.u32 {value}, [{address}];", address=in(reg32) address, value=out(reg32) value, options(nostack, readonly));
    }
    value
}

#[inline(always)]
unsafe fn barrier() {
    // All 128 threads reach both barriers in every uniform K256 iteration.
    // The default memory clobber also prevents compiler motion across them.
    unsafe {
        asm!("bar.sync 0;", options(nostack));
    }
}

#[inline(always)]
unsafe fn word(shared: u32, at: u32) -> u32 {
    unsafe { load_shared(shared + at * 4) }
}

#[path = "q3_negated.rs"]
mod encode;
use encode::negated_q3;

#[inline(always)]
unsafe fn bounded_shift(value: u32, shift: u32) -> u32 {
    let output: u32;
    unsafe {
        asm!("shr.u32 {output}, {value}, {shift};", output=out(reg32) output, value=in(reg32) value, shift=in(reg32) shift, options(nomem, nostack, pure));
    }
    output
}

#[inline(always)]
unsafe fn fragment(shared: u32, base: u32, step: u32, thread: u32) -> [u32; 2] {
    let mut result = [0; 2];
    for half in 0..2 {
        let low = unsafe { word(shared, base + 8 + (step / 4) * 8 + thread + half * 4) };
        let high = unsafe { word(shared, base + thread + half * 4) };
        let code = ((unsafe { bounded_shift(low, 2 * mask_index(step, 3)) }) & 0x0303_0303) | (((unsafe { bounded_shift(high, step) }) & 0x0101_0101) << 2);
        let sg = 2 * step + half;
        let sl = unsafe { bounded_shift(word(shared, base + 24 + mask_index(step, 2) / 2), mask_index(sg, 3) * 8 + (sg / 8) * 4) };
        let sh = unsafe { bounded_shift(word(shared, base + 26), mask_index(sg, 3) * 8 + (sg / 4) * 2) };
        let scale = ((sl & 15) | ((sh & 3) << 4)) as i32 - 32;
        result[half as usize] = negated_q3(code, scale);
    }
    result
}

#[inline(always)]
unsafe fn project<const TILES: usize>(activation: *const u32, weights: *const u32, output: *mut f32, rows: u32, inner: u32, cols: u32) {
    let thread = unsafe { _thread_idx_x() };
    let lane = thread & 31;
    let warp = thread / 32;
    let group = unsafe { mask_index(thread >> 2, 7) };
    let in_group = lane & 3;
    let columns = cols.div_ceil(32);
    if columns == 0 {
        return;
    }
    let index = unsafe { _block_idx_x() };
    let tile_col = (index % columns) * 32;
    let row = (index / columns) * (16 * TILES as u32) + if TILES == 1 { 0 } else { (warp / 2) * 16 } + group;
    let warp_col = if TILES == 1 { warp * 8 } else { (warp & 1) * 16 };
    let row_words = inner / 4;
    let groups = inner / 32;
    let plane = rows * row_words;
    let blocks = inner / 256;
    let shared = unsafe { shared_address() };
    let mut total = [0.0f32; 8];
    let mut correction = [0.0f32; 8];
    for block in 0..blocks {
        // Seven coalesced words per thread cover every raw word in N32/K256.
        // Invalid tail columns are initialized, not left as stale shared data.
        for round in 0..7 {
            let at = thread + round * 128;
            let column = tile_col + at / 28;
            let value = if column < cols {
                unsafe { weights.add(((column * blocks + block) * 28 + at % 28) as usize).read() }
            } else {
                0
            };
            unsafe { store_shared(shared + at * 4, value) };
        }
        unsafe { barrier() };
        let mut headers = [[0.0f32; 2]; 2];
        for tile in 0..TILES {
            let c = warp_col + 2 * in_group + tile as u32 * 8;
            let h0 = unsafe { word(shared, c * 28 + 27) };
            let h1 = unsafe { word(shared, (c + 1) * 28 + 27) };
            headers[tile] = [unsafe { widen(h0 as u16) }, unsafe { widen(h1 as u16) }];
        }
        for step in 0..8 {
            let mut fragments = [[0u32; 2]; 2];
            for tile in 0..TILES {
                let b = warp_col + group + tile as u32 * 8;
                fragments[tile] = unsafe { fragment(shared, b * 28, step, in_group) };
            }
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
                let unit0 = unsafe { mul(scale0, divisor) };
                let unit1 = unsafe { mul(scale1, divisor) };
                for tile in 0..TILES {
                    let products = unsafe { mma([a0, a1, a2, a3], fragments[tile]) };
                    let [d0, d1] = headers[tile];
                    let values = [
                        unsafe { mul(mul((-products[0]) as f32, unit0), d0) },
                        unsafe { mul(mul((-products[1]) as f32, unit0), d1) },
                        unsafe { mul(mul((-products[2]) as f32, unit1), d0) },
                        unsafe { mul(mul((-products[3]) as f32, unit1), d1) },
                    ];
                    for index in 0..4 {
                        let value = values[index];
                        let at = tile * 4 + index;
                        let adjusted = unsafe { sub(value, correction[at]) };
                        let next = unsafe { add(total[at], adjusted) };
                        correction[at] = unsafe { sub(sub(next, total[at]), adjusted) };
                        total[at] = next;
                    }
                }
            }
        }
        // Finish every read before any warp overwrites the tile for the next block.
        unsafe { barrier() };
    }
    for tile in 0..TILES {
        for index in 0..4 {
            let r = row + if index >= 2 { 8 } else { 0 };
            let c = tile_col + warp_col + tile as u32 * 8 + 2 * in_group + (index as u32 & 1);
            if r < rows && c < cols {
                unsafe { output.add((r * cols + c) as usize).write(total[tile * 4 + index]) };
            }
        }
    }
}

/// # Safety
/// Launch 128 threads per CTA over M16/N32 tiles. Input/output contracts match
/// the accepted q3_warp_padded112 kernel. Each CTA owns 3,584 bytes of shared memory.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q3_negated_shared_m16(
    activation: *const u32, weights: *const u32, output: *mut f32, rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32,
) {
    unsafe { project::<1>(activation, weights, output, rows, inner, cols) };
}

/// # Safety
/// Launch 128 threads per CTA over M32/N32 tiles. Input/output contracts match
/// the accepted q3_warp_padded112 kernel. Each CTA owns 3,584 bytes of shared memory.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q3_negated_shared_m32(
    activation: *const u32, weights: *const u32, output: *mut f32, rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32,
) {
    unsafe { project::<2>(activation, weights, output, rows, inner, cols) };
}
