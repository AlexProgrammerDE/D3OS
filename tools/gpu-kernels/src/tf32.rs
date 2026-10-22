//! FP32-range GEMM with three TF32 tensor products per K tile.
//! Splitting each operand into a rounded high part and a residual reduces
//! truncation error without changing the inference API's FP32 buffer format.
use core::arch::asm;

#[inline(always)]
unsafe fn split(value: f32) -> (u32, u32) {
    let high: u32;
    let low: u32;
    unsafe {
        asm!("cvt.rna.tf32.f32 {d}, {s};", d = out(reg32) high, s = in(reg32) value, options(pure, nomem, nostack));
        let residual = value - f32::from_bits(high);
        asm!("cvt.rna.tf32.f32 {d}, {s};", d = out(reg32) low, s = in(reg32) residual, options(pure, nomem, nostack));
    }
    (high, low)
}

#[inline(always)]
unsafe fn mma(a: [u32; 4], b: [u32; 2], mut c: [f32; 4]) -> [f32; 4] {
    unsafe {
        asm!(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 \
            {{{c0}, {c1}, {c2}, {c3}}}, {{{a0}, {a1}, {a2}, {a3}}}, \
            {{{b0}, {b1}}}, {{{c0}, {c1}, {c2}, {c3}}};",
            c0 = inout(reg32) c[0], c1 = inout(reg32) c[1],
            c2 = inout(reg32) c[2], c3 = inout(reg32) c[3],
            a0 = in(reg32) a[0], a1 = in(reg32) a[1],
            a2 = in(reg32) a[2], a3 = in(reg32) a[3],
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], options(nomem, nostack)
        );
    }
    c
}

#[inline(always)]
unsafe fn multiply<const TRANSPOSE_B: bool>(a: u64, b: u64, out: u64, rows: u32, inner: u32, cols: u32) {
    let index = super::index();
    let lane = index % 32;
    let tile = index / 32;
    let row_base = tile / cols.div_ceil(8) * 16;
    let col_base = tile % cols.div_ceil(8) * 8;
    if row_base >= rows {
        return;
    }
    let mut c = [0.0f32; 4];
    let mut correction = [0.0f32; 4];
    for step in 0..inner.div_ceil(8) {
        let mut ah = [0u32; 4];
        let mut al = [0u32; 4];
        let mut bh = [0u32; 2];
        let mut bl = [0u32; 2];
        for element in 0..4 {
            let row = row_base + lane / 4 + (element % 2) * 8;
            let k = step * 8 + lane % 4 + (element / 2) * 4;
            let value = if row < rows && k < inner {
                unsafe { (a as *const f32).add((row * inner + k) as usize).read() }
            } else {
                0.0
            };
            (ah[element as usize], al[element as usize]) = unsafe { split(value) };
        }
        for element in 0..2 {
            let k = step * 8 + lane % 4 + element * 4;
            let col = col_base + lane / 4;
            let offset = if TRANSPOSE_B { col * inner + k } else { k * cols + col };
            let value = if k < inner && col < cols {
                unsafe { (b as *const f32).add(offset as usize).read() }
            } else {
                0.0
            };
            (bh[element as usize], bl[element as usize]) = unsafe { split(value) };
        }
        // Keep the MMA's short reductions separate from the long K reduction.
        // Compensated FP32 accumulation preserves residual products that would
        // disappear when repeatedly added to a large running accumulator.
        let low = unsafe { mma(ah, bl, mma(al, bh, [0.; 4])) };
        let high = unsafe { mma(ah, bh, [0.; 4]) };
        for term in [high, low] {
            for i in 0..4 {
                let adjusted = term[i] - correction[i];
                let sum = c[i] + adjusted;
                correction[i] = (sum - c[i]) - adjusted;
                c[i] = sum;
            }
        }
    }
    for element in 0..4 {
        let row = row_base + lane / 4 + (element / 2) * 8;
        let col = col_base + (lane % 4) * 2 + element % 2;
        if row < rows && col < cols {
            unsafe { (out as *mut f32).add((row * cols + col) as usize).write(c[element as usize]) };
        }
    }
}

/// # Safety
/// A[M,K], B[K,N], output[M,N] are disjoint valid FP32 buffers. Dimensions and
/// products fit u32. Launch full warps, at least ceil(M/16)*ceil(N/8) warps.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn tensor_matmul_tf32x3(a: u64, b: u64, out: u64, rows: u32, inner: u32, cols: u32) {
    unsafe { multiply::<false>(a, b, out, rows, inner, cols) };
}

/// # Safety
/// Same requirements as GEMM, except B is stored as B[N,K].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn tensor_project_tf32x3(a: u64, b: u64, out: u64, rows: u32, inner: u32, cols: u32) {
    unsafe { multiply::<true>(a, b, out, rows, inner, cols) };
}
