//! Dense m16n8k16 tensor-core GEMM, with FP32 input/output and accumulation.
//! Inputs round to FP16 or BF16 before MMA. One full warp owns each 16x8 tile.
//! Fragment coordinates follow NVIDIA's PTX ISA, section mma.m16n8k16:
//! https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float

#[inline(always)]
pub const fn a_coordinate(lane: u32, element: u32) -> (u32, u32) {
    (lane / 4 + ((element / 2) % 2) * 8, (lane % 4) * 2 + element % 2 + (element / 4) * 8)
}

#[inline(always)]
pub const fn b_coordinate(lane: u32, element: u32) -> (u32, u32) {
    ((lane % 4) * 2 + element % 2 + (element / 2) * 8, lane / 4)
}

#[inline(always)]
pub const fn output_coordinate(lane: u32, element: u32) -> (u32, u32) {
    (lane / 4 + (element / 2) * 8, (lane % 4) * 2 + element % 2)
}

#[cfg(target_arch = "nvptx64")]
mod device {
    use super::{a_coordinate, b_coordinate, output_coordinate};
    use core::arch::asm;

    #[inline(always)]
    unsafe fn pack<const BF16: bool>(low: f32, high: f32) -> u32 {
        let packed;
        // cvt.*x2 puts its first source in the high half of the result.
        unsafe {
            if BF16 {
                asm!("cvt.rn.bf16x2.f32 {d}, {hi}, {lo};", d = out(reg32) packed,
                    hi = in(reg32) high, lo = in(reg32) low, options(pure, nomem, nostack));
            } else {
                asm!("cvt.rn.f16x2.f32 {d}, {hi}, {lo};", d = out(reg32) packed,
                    hi = in(reg32) high, lo = in(reg32) low, options(pure, nomem, nostack));
            }
        }
        packed
    }

    #[inline(always)]
    unsafe fn load(base: *const f32, row: u32, col: u32, rows: u32, cols: u32) -> f32 {
        if row < rows && col < cols {
            unsafe { base.add(row as usize * cols as usize + col as usize).read() }
        } else {
            0.0
        }
    }

    #[inline(always)]
    unsafe fn multiply<const BF16: bool>(a: u64, b: u64, out: u64, rows: u32, inner: u32, cols: u32) {
        if rows == 0 || cols == 0 {
            return;
        }
        let index = super::super::index();
        let lane = index % 32;
        let tile = index / 32;
        let tile_cols = cols.div_ceil(8);
        let row_base = (tile / tile_cols) * 16;
        let col_base = (tile % tile_cols) * 8;
        if row_base >= rows {
            return; // Uniform across the entire warp, including excess launch warps.
        }
        let mut c = [0.0f32; 4];
        for step in 0..inner.div_ceil(16) {
            let mut ar = [0u32; 4];
            let mut br = [0u32; 2];
            for pair in 0..4 {
                let (row, col) = a_coordinate(lane, pair * 2);
                ar[pair as usize] = unsafe {
                    pack::<BF16>(
                        load(a as *const f32, row_base + row, step * 16 + col, rows, inner),
                        load(a as *const f32, row_base + row, step * 16 + col + 1, rows, inner),
                    )
                };
            }
            for pair in 0..2 {
                let (row, col) = b_coordinate(lane, pair * 2);
                br[pair as usize] = unsafe {
                    pack::<BF16>(
                        load(b as *const f32, step * 16 + row, col_base + col, inner, cols),
                        load(b as *const f32, step * 16 + row + 1, col_base + col, inner, cols),
                    )
                };
            }
            unsafe {
                if BF16 {
                    asm!(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \
                        {{{c0}, {c1}, {c2}, {c3}}}, {{{a0}, {a1}, {a2}, {a3}}}, \
                        {{{b0}, {b1}}}, {{{c0}, {c1}, {c2}, {c3}}};",
                        c0 = inout(reg32) c[0], c1 = inout(reg32) c[1],
                        c2 = inout(reg32) c[2], c3 = inout(reg32) c[3],
                        a0 = in(reg32) ar[0], a1 = in(reg32) ar[1],
                        a2 = in(reg32) ar[2], a3 = in(reg32) ar[3],
                        b0 = in(reg32) br[0], b1 = in(reg32) br[1],
                        options(nostack)
                    );
                } else {
                    asm!(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                        {{{c0}, {c1}, {c2}, {c3}}}, {{{a0}, {a1}, {a2}, {a3}}}, \
                        {{{b0}, {b1}}}, {{{c0}, {c1}, {c2}, {c3}}};",
                        c0 = inout(reg32) c[0], c1 = inout(reg32) c[1],
                        c2 = inout(reg32) c[2], c3 = inout(reg32) c[3],
                        a0 = in(reg32) ar[0], a1 = in(reg32) ar[1],
                        a2 = in(reg32) ar[2], a3 = in(reg32) ar[3],
                        b0 = in(reg32) br[0], b1 = in(reg32) br[1],
                        options(nostack)
                    );
                }
            }
        }
        for element in 0..4 {
            let (row, col) = output_coordinate(lane, element);
            let (row, col) = (row_base + row, col_base + col);
            if row < rows && col < cols {
                unsafe { (out as *mut f32).add(row as usize * cols as usize + col as usize).write(c[element as usize]) };
            }
        }
    }

    /// # Safety
    /// Buffers hold rows*inner, inner*cols, and rows*cols FP32 values without aliasing.
    /// Launch full 32-thread warps in a 1D grid, at least ceil(rows/16)*ceil(cols/8).
    /// The total thread count and padded row/column coordinates must fit u32.
    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn tensor_matmul_f16(a: u64, b: u64, out: u64, rows: u32, inner: u32, cols: u32) {
        unsafe { multiply::<false>(a, b, out, rows, inner, cols) };
    }

    /// # Safety
    /// Same buffer and launch requirements as `tensor_matmul_f16`.
    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn tensor_matmul_bf16(a: u64, b: u64, out: u64, rows: u32, inner: u32, cols: u32) {
        unsafe { multiply::<true>(a, b, out, rows, inner, cols) };
    }
}
