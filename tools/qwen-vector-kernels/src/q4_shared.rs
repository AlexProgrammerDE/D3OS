//! Shared raw Q4_K tiles for the measured prefill cells.
use super::{_block_idx_x, _thread_idx_x, add, dot, mask_index, mul, shuffle, sub, widen};
use crate::q3_padded::mma;
use core::arch::asm;

#[inline(always)]
unsafe fn shared_address() -> u32 {
    let address: u32;
    unsafe {
        asm!(".shared .align 16 .b8 q4_tile[4608]; mov.u32 {address}, q4_tile;", address=out(reg32) address, options(nostack));
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

#[inline(always)]
unsafe fn scale_min(shared: u32, base: u32, step: u32) -> (i32, i32) {
    let shift = (step & 3) * 8;
    let a = (unsafe { word(shared, base + 1) } >> shift) & 255;
    let b = (unsafe { word(shared, base + 2) } >> shift) & 255;
    let c = (unsafe { word(shared, base + 3) } >> shift) & 255;
    let scale = if step < 4 { a & 63 } else { (c & 15) | ((a >> 6) << 4) };
    let minimum = if step < 4 { b & 63 } else { (c >> 4) | ((b >> 6) << 4) };
    (scale as i32, minimum as i32)
}

#[inline(always)]
unsafe fn fragment(shared: u32, base: u32, step: u32, thread: u32) -> [u32; 2] {
    let at = base + 4 + (step / 2) * 8 + thread;
    let shift = (step & 1) * 4;
    [
        (unsafe { word(shared, at) } >> shift) & 0x0f0f_0f0f,
        (unsafe { word(shared, at + 4) } >> shift) & 0x0f0f_0f0f,
    ]
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
        // Nine coalesced words per thread cover every raw word in N32/K256.
        // Invalid tail columns are initialized, not left as stale shared data.
        for round in 0..9 {
            let at = thread + round * 128;
            let column = tile_col + at / 36;
            let value = if column < cols {
                unsafe { weights.add(((column * blocks + block) * 36 + at % 36) as usize).read() }
            } else {
                0
            };
            unsafe { store_shared(shared + at * 4, value) };
        }
        unsafe { barrier() };
        let mut headers = [[0.0f32; 4]; 2];
        for tile in 0..TILES {
            let c = warp_col + 2 * in_group + tile as u32 * 8;
            let h0 = unsafe { word(shared, c * 36) };
            let h1 = unsafe { word(shared, (c + 1) * 36) };
            headers[tile] = [
                unsafe { widen(h0 as u16) },
                unsafe { widen(h1 as u16) },
                unsafe { widen((h0 >> 16) as u16) },
                unsafe { widen((h1 >> 16) as u16) },
            ];
        }
        for step in 0..8 {
            let mut fragments = [[0u32; 2]; 2];
            let mut metadata = [[0i32; 4]; 2];
            for tile in 0..TILES {
                let b = warp_col + group + tile as u32 * 8;
                let c = warp_col + 2 * in_group + tile as u32 * 8;
                fragments[tile] = unsafe { fragment(shared, b * 36, step, in_group) };
                let (s0, m0) = unsafe { scale_min(shared, c * 36, step) };
                let (s1, m1) = unsafe { scale_min(shared, (c + 1) * 36, step) };
                metadata[tile] = [s0, s1, m0, m1];
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
                let mut sum0 = unsafe { dot(a0, 0x0101_0101, dot(a2, 0x0101_0101, 0)) };
                let mut sum1 = unsafe { dot(a1, 0x0101_0101, dot(a3, 0x0101_0101, 0)) };
                for shift in [1, 2] {
                    sum0 += unsafe { shuffle(sum0 as u32, shift) } as i32;
                    sum1 += unsafe { shuffle(sum1 as u32, shift) } as i32;
                }
                let unit0 = unsafe { mul(scale0, divisor) };
                let unit1 = unsafe { mul(scale1, divisor) };
                for tile in 0..TILES {
                    let products = unsafe { mma([a0, a1, a2, a3], fragments[tile]) };
                    let [qs0, qs1, min0, min1] = metadata[tile];
                    let [d0, d1, dm0, dm1] = headers[tile];
                    let positives = [products[0] * qs0, products[1] * qs1, products[2] * qs0, products[3] * qs1];
                    let negatives = [sum0 * min0, sum0 * min1, sum1 * min0, sum1 * min1];
                    for index in 0..4 {
                        let unit = if index < 2 { unit0 } else { unit1 };
                        let (d, dm) = if index & 1 == 0 { (d0, dm0) } else { (d1, dm1) };
                        let positive = unsafe { mul(mul(positives[index] as f32, unit), d) };
                        let negative = unsafe { mul(mul(negatives[index] as f32, unit), dm) };
                        let value = unsafe { sub(positive, negative) };
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
/// the accepted q4_warp_m16 kernel. Each CTA owns 4,608 bytes of shared memory.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q4_shared_m16(
    activation: *const u32, weights: *const u32, output: *mut f32, rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32,
) {
    unsafe { project::<1>(activation, weights, output, rows, inner, cols) };
}

/// # Safety
/// Launch 128 threads per CTA over M32/N32 tiles. Input/output contracts match
/// the accepted q4_warp_m16 kernel. Each CTA owns 4,608 bytes of shared memory.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q4_shared_m32(
    activation: *const u32, weights: *const u32, output: *mut f32, rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32,
) {
    unsafe { project::<2>(activation, weights, output, rows, inner, cols) };
}
