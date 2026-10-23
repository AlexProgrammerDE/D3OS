//! FP32-range GEMM with three TF32 tensor products per K tile.
//! Splitting each operand into a rounded high part and a residual reduces
//! truncation error without changing the inference API's FP32 buffer format.
//! BF16 weights have no TF32 residual and therefore need only two products.
use core::arch::asm;

#[inline(always)]
unsafe fn weight<const BF16: bool>(address: u64, index: u32) -> f32 {
    if BF16 {
        let bits = unsafe { (address as *const u16).add(index as usize).read() };
        f32::from_bits(u32::from(bits) << 16)
    } else {
        unsafe { (address as *const f32).add(index as usize).read() }
    }
}

#[inline(always)]
unsafe fn projection_vector<const BF16: bool, const GATED: bool, const OUTPUTS: usize>(a: u64, b: u64, up: u64, out: u64, inner: u32, cols: u32) {
    if BF16 && inner.is_multiple_of(64) && b.is_multiple_of(4) && (!GATED || up.is_multiple_of(4)) {
        unsafe { projection_vector_inner::<BF16, GATED, OUTPUTS, true>(a, b, up, out, inner, cols) };
    } else {
        unsafe { projection_vector_inner::<BF16, GATED, OUTPUTS, false>(a, b, up, out, inner, cols) };
    }
}

#[inline(always)]
unsafe fn vector_weights<const BF16: bool, const PAIR: bool>(address: u64, index: u32) -> [f32; 2] {
    if PAIR {
        // The caller establishes aligned rows and complete 64-element tiles.
        // One 32-bit load expands two BF16 values without further rounding.
        let packed = unsafe { (address as *const u32).add((index / 2) as usize).read() };
        [f32::from_bits(packed << 16), f32::from_bits(packed & 0xffff0000)]
    } else {
        [unsafe { weight::<BF16>(address, index) }, 0.]
    }
}

#[inline(always)]
unsafe fn vector_fma(input: f32, weight: f32, mut sum: f32) -> f32 {
    unsafe {
        asm!("fma.rn.f32 {d}, {a}, {b}, {d};", d = inout(reg32) sum,
            a = in(reg32) input, b = in(reg32) weight, options(pure, nomem, nostack));
    }
    sum
}

#[inline(always)]
unsafe fn projection_vector_inner<const BF16: bool, const GATED: bool, const OUTPUTS: usize, const PAIR: bool>(
    a: u64, b: u64, up: u64, out: u64, inner: u32, cols: u32,
) {
    let index = super::index();
    let lane = index % 32;
    let column = (index / 32) * OUTPUTS as u32;
    if column >= cols {
        return;
    }
    // Wide matrices reuse each input across several outputs. Narrow matrices
    // use one output per warp to occupy more SMs. Both coalesce weight reads.
    let mut sums = [0.0f32; OUTPUTS];
    let mut up_sums = [0.0f32; OUTPUTS];
    let values = if PAIR { 2 } else { 1 };
    let mut k = lane * values;
    while k < inner {
        let input = [
            unsafe { (a as *const f32).add(k as usize).read() },
            if PAIR {
                unsafe { (a as *const f32).add((k + 1) as usize).read() }
            } else {
                0.
            },
        ];
        for j in 0..OUTPUTS as u32 {
            if column + j < cols {
                let factor = unsafe { vector_weights::<BF16, PAIR>(b, (column + j) * inner + k) };
                for part in 0..values as usize {
                    sums[j as usize] = unsafe { vector_fma(input[part], factor[part], sums[j as usize]) };
                }
                if GATED {
                    let weight = unsafe { vector_weights::<BF16, PAIR>(up, (column + j) * inner + k) };
                    for part in 0..values as usize {
                        up_sums[j as usize] = unsafe { vector_fma(input[part], weight[part], up_sums[j as usize]) };
                    }
                }
            }
        }
        k += 32 * values;
    }
    for sum in &mut sums {
        *sum = unsafe { super::warp::sum(*sum) };
    }
    if GATED {
        for sum in &mut up_sums {
            *sum = unsafe { super::warp::sum(*sum) };
        }
    }
    if lane == 0 {
        for j in 0..OUTPUTS as u32 {
            if column + j < cols {
                unsafe {
                    let value = if GATED {
                        super::transformer::silu(sums[j as usize]) * up_sums[j as usize]
                    } else {
                        sums[j as usize]
                    };
                    (out as *mut f32).add((column + j) as usize).write(value);
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn split(value: f32) -> (u32, u32) {
    let high = unsafe { rounded(value) };
    let low: u32;
    unsafe {
        let residual = value - f32::from_bits(high);
        asm!("cvt.rna.tf32.f32 {d}, {s};", d = out(reg32) low, s = in(reg32) residual, options(pure, nomem, nostack));
    }
    (high, low)
}

#[inline(always)]
unsafe fn rounded(value: f32) -> u32 {
    let result;
    unsafe {
        asm!("cvt.rna.tf32.f32 {d}, {s};", d = out(reg32) result, s = in(reg32) value, options(pure, nomem, nostack));
    }
    result
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
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], options(nostack)
        );
    }
    c
}

#[inline(always)]
unsafe fn multiply<const TRANSPOSE_B: bool, const BF16: bool, const SPLIT: bool>(a: u64, b: u64, out: u64, rows: u32, inner: u32, cols: u32, splits: u32) {
    if TRANSPOSE_B && rows == 1 && !SPLIT {
        if cols <= 2048 {
            unsafe { projection_vector::<BF16, false, 1>(a, b, 0, out, inner, cols) };
        } else if BF16 && cols <= 8192 {
            unsafe { projection_vector::<BF16, false, 4>(a, b, 0, out, inner, cols) };
        } else {
            unsafe { projection_vector::<BF16, false, 8>(a, b, 0, out, inner, cols) };
        }
        return;
    }
    let index = super::index();
    let lane = index % 32;
    let mut tile = index / 32;
    let tiles = rows.div_ceil(16) * cols.div_ceil(8);
    let part = if SPLIT { tile / tiles } else { 0 };
    if SPLIT && part >= splits {
        return;
    }
    if SPLIT {
        tile %= tiles;
    }
    let row_base = tile / cols.div_ceil(8) * 16;
    let col_base = tile % cols.div_ceil(8) * 8;
    if row_base >= rows {
        return;
    }
    let mut c = [0.0f32; 4];
    let mut correction = [0.0f32; 4];
    let k_tiles = inner.div_ceil(8);
    let chunk = k_tiles.div_ceil(splits);
    let first = part * chunk;
    for step in first..(first + chunk).min(k_tiles) {
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
            let value = if k < inner && col < cols { unsafe { weight::<BF16>(b, offset) } } else { 0.0 };
            if BF16 {
                // Seven BF16 fraction bits fit exactly in TF32. Keep the PTX
                // conversion because TF32's register encoding is target-defined.
                bh[element as usize] = unsafe { rounded(value) };
            } else {
                (bh[element as usize], bl[element as usize]) = unsafe { split(value) };
            }
        }
        // Keep the MMA's short reductions separate from the long K reduction.
        // Compensated FP32 accumulation preserves residual products that would
        // disappear when repeatedly added to a large running accumulator.
        let low = if BF16 {
            unsafe { mma(al, bh, [0.; 4]) }
        } else {
            unsafe { mma(ah, bl, mma(al, bh, [0.; 4])) }
        };
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
            unsafe { (out as *mut f32).add(((part * rows + row) * cols + col) as usize).write(c[element as usize]) };
        }
    }
}

/// # Safety
/// A[M,K], B[K,N], output[M,N] are disjoint valid FP32 buffers. Dimensions and
/// products fit u32. Launch full warps, at least ceil(M/16)*ceil(N/8) warps.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn tensor_matmul_tf32x3(a: u64, b: u64, out: u64, rows: u32, inner: u32, cols: u32) {
    unsafe { multiply::<false, false, false>(a, b, out, rows, inner, cols, 1) };
}

/// # Safety
/// Same bounds as GEMM, except B is stored as B[N,K]. Launch full warps:
/// ceil(M/16)*ceil(N/8) warps, or N warps when M=1 and N<=2048.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn tensor_project_tf32x3(a: u64, b: u64, out: u64, rows: u32, inner: u32, cols: u32) {
    unsafe { multiply::<true, false, false>(a, b, out, rows, inner, cols, 1) };
}

/// # Safety
/// A[M,K] and output[M,N] contain FP32 values. B[N,K] contains packed BF16.
/// Buffers are disjoint, products fit u32, and the launch uses full warps.
/// Cover ceil(M/16)*ceil(N/8) tiles when M>1. For M=1, each warp owns one
/// output through N=2048, four through N=8192, or eight for wider matrices.
/// Accumulation remains FP32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn tensor_project_bf16_weights(a: u64, b: u64, out: u64, rows: u32, inner: u32, cols: u32) {
    unsafe { multiply::<true, true, false>(a, b, out, rows, inner, cols, 1) };
}

/// # Safety
/// A[rows,inner] is FP32, B[cols,inner] packed BF16, and output[splits,rows,cols]
/// FP32. Buffers are disjoint; products fit u32. Launch full warps covering
/// splits*ceil(rows/16)*ceil(cols/8) tiles. Empty K partitions write zeros.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn project_bf16_split(a: u64, b: u64, out: u64, rows: u32, inner: u32, cols: u32, splits: u32) {
    unsafe { multiply::<true, true, true>(a, b, out, rows, inner, cols, splits) };
}

/// # Safety
/// Input[splits,rows,cols] and output[rows,cols] are disjoint FP32 buffers.
/// Products fit u32. Launch at least rows*cols threads.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn sum_projection_partials(input: u64, output: u64, rows: u32, cols: u32, splits: u32) {
    let i = super::index();
    let count = rows * cols;
    if i >= count {
        return;
    }
    let mut sum = 0.;
    let mut correction = 0.;
    for part in 0..splits {
        let value = unsafe { (input as *const f32).add((part * count + i) as usize).read() };
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    unsafe { (output as *mut f32).add(i as usize).write(sum) };
}

/// # Safety
/// A[K] and output[N] are FP32; gate[N,K] and up[N,K] are packed BF16.
/// Output cannot overlap any input, and products fit u32. Launch full warps:
/// N warps when N<=2048, otherwise ceil(N/4) warps. Accumulation is FP32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gated_project_bf16_weights(a: u64, gate: u64, up: u64, out: u64, inner: u32, cols: u32) {
    if cols <= 2048 {
        unsafe { projection_vector::<true, true, 1>(a, gate, up, out, inner, cols) };
    } else {
        // Four gated outputs have the same number of accumulators as eight
        // ordinary outputs, with more warps available to cover memory latency.
        unsafe { projection_vector::<true, true, 4>(a, gate, up, out, inner, cols) };
    }
}
