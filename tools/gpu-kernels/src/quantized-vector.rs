// Each lane reads two aligned Q8 bytes and keeps both products in FP32.
// Narrow projections use 128 logical lanes for one row; wider ones use 32 per row.
fn vector_q8_byte<const SIGNED: bool>(value: Tile<u32, { [128] }>) -> Tile<f32, { [128] }> {
    if SIGNED {
        let small: Tile<u8, { [128] }> = trunci(value, overflow::NoUnsignedWrap);
        let signed: Tile<i8, { [128] }> = bitcast(small);
        itof(signed, rounding::NearestEven)
    } else {
        quant_signed_byte_128(value)
    }
}

unsafe fn vector_q8_pairs<const GROUP: i32, const NARROW: bool, const SIGNED: bool>(
    a: *const f32, b: *const f32, col: Tile<u32, { [128] }>, lane: Tile<u32, { [128] }>, block: u32, blocks: u32, cols: u32, input_offset: u32,
    even: Tile<f32, { [128] }>, odd: Tile<f32, { [128] }>,
) -> (Tile<f32, { [128] }>, Tile<f32, { [128] }>) {
    let group_blocks = if NARROW { 8u32 } else { 2u32 };
    let qblock = lane / broadcast_scalar(16u32, shape![128]) + broadcast_scalar(block + unsigned(GROUP) * group_blocks, shape![128]);
    let pair = lane % broadcast_scalar(16u32, shape![128]);
    let mask = lt_tile(col, broadcast_scalar(cols, shape![128])) & lt_tile(qblock, broadcast_scalar(blocks, shape![128]));
    let base = (col * broadcast_scalar(blocks, shape![128]) + qblock) * broadcast_scalar(17u32, shape![128]);
    let scale = unsafe { quant_half_128(b, base * broadcast_scalar(2u32, shape![128]), mask) };
    let packed = unsafe { read_tile(bf16_pointer(b), base + broadcast_scalar(1u32, shape![128]) + pair, mask, 0u16) };
    let packed: Tile<u32, { [128] }> = exti(packed);
    let low = packed & broadcast_scalar(255u32, shape![128]);
    let high = packed >> broadcast_scalar(8u32, shape![128]);
    let low = scale * vector_q8_byte::<SIGNED>(low);
    let high = scale * vector_q8_byte::<SIGNED>(high);
    let index = broadcast_scalar(input_offset, shape![128]) + qblock * broadcast_scalar(32u32, shape![128]) + pair * broadcast_scalar(2u32, shape![128]);
    let x0 = unsafe { read_tile(a, index, mask, 0f32) };
    let x1 = unsafe { read_tile(a, index + broadcast_scalar(1u32, shape![128]), mask, 0f32) };
    (
        fma(x0, low, even, rounding::NearestEven, ftz::Disabled),
        fma(x1, high, odd, rounding::NearestEven, ftz::Disabled),
    )
}

unsafe fn vector_project_q8<const NARROW: bool, const SIGNED: bool>(a: *const f32, b: *const f32, output: *mut f32, inner: u32, cols: u32) {
    let first = block_id() * if NARROW { 1u32 } else { 4u32 };
    unsafe { vector_project_q8_at::<NARROW, SIGNED>(a, b, output, inner, cols, first, 0u32, 0u32) }
}

unsafe fn vector_project_q8_at<const NARROW: bool, const SIGNED: bool>(
    a: *const f32, b: *const f32, output: *mut f32, inner: u32, cols: u32, first: u32, output_offset: u32, input_offset: u32,
) {
    let lane = if NARROW {
        lanes::<128>()
    } else {
        lanes::<128>() % broadcast_scalar(32u32, shape![128])
    };
    let col = if NARROW {
        broadcast_scalar(first, shape![128])
    } else {
        lanes::<128>() / broadcast_scalar(32u32, shape![128]) + broadcast_scalar(first, shape![128])
    };
    let blocks = inner / 32u32;
    let mut e0 = broadcast_scalar(0f32, shape![128]);
    let mut o0 = e0;
    let mut e1 = e0;
    let mut o1 = e0;
    let mut e2 = e0;
    let mut o2 = e0;
    let mut e3 = e0;
    let mut o3 = e0;
    let mut block = 0u32;
    while block < blocks {
        let (even, odd) = unsafe { vector_q8_pairs::<0, NARROW, SIGNED>(a, b, col, lane, block, blocks, cols, input_offset, e0, o0) };
        e0 = even;
        o0 = odd;
        let (even, odd) = unsafe { vector_q8_pairs::<1, NARROW, SIGNED>(a, b, col, lane, block, blocks, cols, input_offset, e1, o1) };
        e1 = even;
        o1 = odd;
        let (even, odd) = unsafe { vector_q8_pairs::<2, NARROW, SIGNED>(a, b, col, lane, block, blocks, cols, input_offset, e2, o2) };
        e2 = even;
        o2 = odd;
        let (even, odd) = unsafe { vector_q8_pairs::<3, NARROW, SIGNED>(a, b, col, lane, block, blocks, cols, input_offset, e3, o3) };
        e3 = even;
        o3 = odd;
        block = block + if NARROW { 32u32 } else { 8u32 };
    }
    let sum = ((e0 + o0) + (e1 + o1)) + ((e2 + o2) + (e3 + o3));
    if NARROW {
        // A single output uses all four warps, avoiding a second merge launch.
        let sum: Tile<f32, { [1, 128] }> = sum.reshape(shape![1, 128]);
        let value: Tile<f32, { [1] }> = reduce_sum(sum, 1i32);
        let col = broadcast_scalar(first, shape![1]);
        unsafe {
            write_tile(
                output,
                col + broadcast_scalar(output_offset, shape![1]),
                value,
                lt_tile(col, broadcast_scalar(cols, shape![1])),
            )
        };
    } else {
        let sum: Tile<f32, { [4, 32] }> = sum.reshape(shape![4, 32]);
        let value: Tile<f32, { [4] }> = reduce_sum(sum, 1i32);
        let col = lanes::<4>() + broadcast_scalar(first, shape![4]);
        unsafe {
            write_tile(
                output,
                col + broadcast_scalar(output_offset, shape![4]),
                value,
                lt_tile(col, broadcast_scalar(cols, shape![4])),
            )
        };
    }
}

// Parallelize narrow batched projections across rows and output columns.
#[cutile::entry()]
unsafe fn q8_batched_narrow_signed(input: *const f32, weights: *const f32, output: *mut f32, _rows: u32, inner: u32, cols: u32, _format: u32) {
    let row = block_id() / cols;
    let column = block_id() % cols;
    unsafe { vector_project_q8_at::<true, true>(input, weights, output, inner, cols, column, row * cols, row * inner) };
}

// Schedule adjacent rows together to reuse each packed weight column.
#[cutile::entry()]
unsafe fn q8_batched_wide_column(input: *const f32, weights: *const f32, output: *mut f32, rows: u32, inner: u32, cols: u32, _format: u32) {
    let row = block_id() % rows;
    let column = (block_id() / rows) * 4u32;
    unsafe { vector_project_q8_at::<false, true>(input, weights, output, inner, cols, column, row * cols, row * inner) };
}

// Group wide qkv/gate and narrow beta/alpha vectors without repacking weights.
unsafe fn group_project_q8<const TILED: bool>(
    input: *const f32, qkv: *const f32, gate: *const f32, beta: *const f32, alpha: *const f32, output: *mut f32, inner: u32, qkv_cols: u32, gate_cols: u32,
    heads: u32, _format: u32,
) {
    let block = block_id();
    let qkv_blocks = (qkv_cols + 3u32) / 4u32;
    let gate_end = qkv_blocks + (gate_cols + 3u32) / 4u32;
    let beta_end = gate_end + heads;
    let weights = if block < qkv_blocks {
        qkv
    } else if block < gate_end {
        gate
    } else if block < beta_end {
        beta
    } else {
        alpha
    };
    let first_block = if block < qkv_blocks {
        0u32
    } else if block < gate_end {
        qkv_blocks
    } else if block < beta_end {
        gate_end
    } else {
        beta_end
    };
    let output_offset = if block < qkv_blocks {
        0u32
    } else if block < gate_end {
        qkv_cols
    } else if block < beta_end {
        qkv_cols + gate_cols
    } else {
        qkv_cols + gate_cols + heads
    };
    let cols = if block < qkv_blocks {
        qkv_cols
    } else if block < gate_end {
        gate_cols
    } else {
        heads
    };
    if block < gate_end {
        if TILED {
            if inner == 4096u32 && cols == 8192u32 {
                unsafe { blocked_vector_at::<1, false, true, 4>(input, weights, output, inner, cols, (block - first_block) * 4u32, output_offset, 0u32) };
            } else {
                unsafe { blocked_vector_at::<1, false, true, 0>(input, weights, output, inner, cols, (block - first_block) * 4u32, output_offset, 0u32) };
            }
        } else {
            unsafe { vector_project_q8_at::<false, true>(input, weights, output, inner, cols, (block - first_block) * 4u32, output_offset, 0u32) };
        }
    } else {
        unsafe { vector_project_q8_at::<true, false>(input, weights, output, inner, cols, block - first_block, output_offset, 0u32) };
    }
}

// Decode each 256-weight block with 32 logical lanes per output row. A CTA
// projects four rows, reusing packed bytes across eight 32-value groups.
// Keep FP32 activations and the generic decoder's scale multiplication order.
fn vector_q3_group<const G: i32>(
    q: Tile<u32, { [128] }>, high: Tile<u32, { [128] }>, low_scale: Tile<u32, { [128] }>, high_scale: Tile<u32, { [128] }>, d: Tile<f32, { [128] }>,
) -> Tile<f32, { [128] }> {
    let low = (low_scale >> broadcast_scalar(4u32 * (unsigned(G) / 4u32), shape![128])) & broadcast_scalar(15u32, shape![128]);
    let high_scale = (high_scale >> broadcast_scalar(2u32 * (unsigned(G) / 2u32), shape![128])) & broadcast_scalar(3u32, shape![128]);
    let scale: Tile<f32, { [128] }> = itof(low | (high_scale << broadcast_scalar(4u32, shape![128])), rounding::NearestEven);
    let scale = scale - broadcast_scalar(32f32, shape![128]);
    let q = (q >> broadcast_scalar(2u32 * (unsigned(G) % 4u32), shape![128])) & broadcast_scalar(3u32, shape![128]);
    let q: Tile<f32, { [128] }> = itof(q, rounding::NearestEven);
    let q = q - select(
        eq_tile(
            (high >> broadcast_scalar(unsigned(G), shape![128])) & broadcast_scalar(1u32, shape![128]),
            broadcast_scalar(0u32, shape![128]),
        ),
        broadcast_scalar(4f32, shape![128]),
        broadcast_scalar(0f32, shape![128]),
    );
    (d * scale) * q
}
fn vector_q4_group<const FORMAT: u32, const G: i32>(
    q: Tile<u32, { [128] }>, high: Tile<u32, { [128] }>, s0: Tile<u32, { [128] }>, s1: Tile<u32, { [128] }>, s2: Tile<u32, { [128] }>, d: Tile<f32, { [128] }>,
    dmin: Tile<f32, { [128] }>,
) -> Tile<f32, { [128] }> {
    let a = (s0 >> broadcast_scalar(8u32 * (unsigned(G) % 4u32), shape![128])) & broadcast_scalar(255u32, shape![128]);
    let b = (s1 >> broadcast_scalar(8u32 * (unsigned(G) % 4u32), shape![128])) & broadcast_scalar(255u32, shape![128]);
    let c = (s2 >> broadcast_scalar(8u32 * (unsigned(G) % 4u32), shape![128])) & broadcast_scalar(255u32, shape![128]);
    let scale = if unsigned(G) < 4u32 {
        a & broadcast_scalar(63u32, shape![128])
    } else {
        (c & broadcast_scalar(15u32, shape![128])) | ((a >> broadcast_scalar(6u32, shape![128])) << broadcast_scalar(4u32, shape![128]))
    };
    let minimum = if unsigned(G) < 4u32 {
        b & broadcast_scalar(63u32, shape![128])
    } else {
        (c >> broadcast_scalar(4u32, shape![128])) | ((b >> broadcast_scalar(6u32, shape![128])) << broadcast_scalar(4u32, shape![128]))
    };
    let q = (q >> broadcast_scalar(4u32 * (unsigned(G) % 2u32), shape![128])) & broadcast_scalar(15u32, shape![128]);
    let q = if FORMAT == 13u32 {
        q | (((high >> broadcast_scalar(unsigned(G), shape![128])) & broadcast_scalar(1u32, shape![128])) << broadcast_scalar(4u32, shape![128]))
    } else {
        q
    };
    let scale: Tile<f32, { [128] }> = itof(scale, rounding::NearestEven);
    let minimum: Tile<f32, { [128] }> = itof(minimum, rounding::NearestEven);
    let q: Tile<f32, { [128] }> = itof(q, rounding::NearestEven);
    (d * scale) * q - dmin * minimum
}
fn vector_q6_group<const G: i32>(
    low: Tile<u32, { [128] }>, high: Tile<u32, { [128] }>, scale: Tile<u32, { [128] }>, d: Tile<f32, { [128] }>,
) -> Tile<f32, { [128] }> {
    let low = (low >> broadcast_scalar(4u32 * ((unsigned(G) % 4u32) / 2u32), shape![128])) & broadcast_scalar(15u32, shape![128]);
    let high = (high >> broadcast_scalar(2u32 * (unsigned(G) % 4u32), shape![128])) & broadcast_scalar(3u32, shape![128]);
    let q: Tile<f32, { [128] }> = itof(low | (high << broadcast_scalar(4u32, shape![128])), rounding::NearestEven);
    let q = q - broadcast_scalar(32f32, shape![128]);
    (d * quant_signed_byte_128(scale)) * q
}
unsafe fn vector_quantized_fma<const G: i32>(
    a: *const f32, start: u32, lane: Tile<u32, { [128] }>, w: Tile<f32, { [128] }>, sum: Tile<f32, { [128] }>,
) -> Tile<f32, { [128] }> {
    let x = unsafe {
        read_tile(
            a,
            lane + broadcast_scalar(start + unsigned(G) * 32u32, shape![128]),
            broadcast_scalar(true, shape![128]),
            0f32,
        )
    };
    fma(x, w, sum, rounding::NearestEven, ftz::Disabled)
}
#[cutile::entry()]
unsafe fn group_project_q8_0(
    input: *const f32, qkv: *const f32, gate: *const f32, beta: *const f32, alpha: *const f32, output: *mut f32, inner: u32, qkv_cols: u32, gate_cols: u32,
    heads: u32, _format: u32,
) {
    unsafe { group_project_q8::<false>(input, qkv, gate, beta, alpha, output, inner, qkv_cols, gate_cols, heads, _format) }
}

#[cutile::entry()]
unsafe fn group_project_q8_columns32(
    input: *const f32, qkv: *const f32, gate: *const f32, beta: *const f32, alpha: *const f32, output: *mut f32, inner: u32, qkv_cols: u32, gate_cols: u32,
    heads: u32, _format: u32, _layout: u32,
) {
    unsafe { group_project_q8::<true>(input, qkv, gate, beta, alpha, output, inner, qkv_cols, gate_cols, heads, _format) }
}

unsafe fn vector_project_k<const FORMAT: u32>(a: *const f32, b: *const f32, output: *mut f32, inner: u32, cols: u32) {
    let first = block_id() * 4u32;
    let lane = lanes::<128>() % broadcast_scalar(32u32, shape![128]);
    let col = lanes::<128>() / broadcast_scalar(32u32, shape![128]) + broadcast_scalar(first, shape![128]);
    // Inactive output lanes read a valid weight row; their stores stay masked.
    // Unconditional packed loads avoid padding work throughout the reduction.
    let col = select(
        lt_tile(col, broadcast_scalar(cols, shape![128])),
        col,
        broadcast_scalar(cols - 1u32, shape![128]),
    );
    let mask = broadcast_scalar(true, shape![128]);
    let half = lane / broadcast_scalar(16u32, shape![128]);
    let blocks = inner / 256u32;
    let bytes = if FORMAT == 11u32 {
        110u32
    } else if FORMAT == 12u32 {
        144u32
    } else if FORMAT == 13u32 {
        176u32
    } else {
        210u32
    };
    let mut sum0 = broadcast_scalar(0f32, shape![128]);
    let mut sum1 = broadcast_scalar(0f32, shape![128]);
    let mut sum2 = broadcast_scalar(0f32, shape![128]);
    let mut sum3 = broadcast_scalar(0f32, shape![128]);
    let mut block = 0u32;
    while block < blocks {
        let base = (col * broadcast_scalar(blocks, shape![128]) + broadcast_scalar(block, shape![128])) * broadcast_scalar(bytes, shape![128]);
        let start = block * 256u32;
        if FORMAT == 11u32 {
            let d = unsafe { quant_half_128(b, base + broadcast_scalar(108u32, shape![128]), mask) };
            let q0 = unsafe { quant_byte_128(b, base + broadcast_scalar(32u32, shape![128]) + lane, mask) };
            let q1 = unsafe { quant_byte_128(b, base + broadcast_scalar(64u32, shape![128]) + lane, mask) };
            let h = unsafe { quant_byte_128(b, base + broadcast_scalar(0u32, shape![128]) + lane, mask) };
            let s0 = unsafe { quant_byte_128(b, base + broadcast_scalar(96u32, shape![128]) + half, mask) };
            let s1 = unsafe { quant_byte_128(b, base + broadcast_scalar(98u32, shape![128]) + half, mask) };
            let s2 = unsafe { quant_byte_128(b, base + broadcast_scalar(100u32, shape![128]) + half, mask) };
            let s3 = unsafe { quant_byte_128(b, base + broadcast_scalar(102u32, shape![128]) + half, mask) };
            let s4 = unsafe { quant_byte_128(b, base + broadcast_scalar(104u32, shape![128]) + half, mask) };
            let s5 = unsafe { quant_byte_128(b, base + broadcast_scalar(106u32, shape![128]) + half, mask) };
            let w = vector_q3_group::<0>(q0, h, s0, s4, d);
            sum0 = unsafe { vector_quantized_fma::<0>(a, start, lane, w, sum0) };
            let w = vector_q3_group::<1>(q0, h, s1, s5, d);
            sum1 = unsafe { vector_quantized_fma::<1>(a, start, lane, w, sum1) };
            let w = vector_q3_group::<2>(q0, h, s2, s4, d);
            sum2 = unsafe { vector_quantized_fma::<2>(a, start, lane, w, sum2) };
            let w = vector_q3_group::<3>(q0, h, s3, s5, d);
            sum3 = unsafe { vector_quantized_fma::<3>(a, start, lane, w, sum3) };
            let w = vector_q3_group::<4>(q1, h, s0, s4, d);
            sum0 = unsafe { vector_quantized_fma::<4>(a, start, lane, w, sum0) };
            let w = vector_q3_group::<5>(q1, h, s1, s5, d);
            sum1 = unsafe { vector_quantized_fma::<5>(a, start, lane, w, sum1) };
            let w = vector_q3_group::<6>(q1, h, s2, s4, d);
            sum2 = unsafe { vector_quantized_fma::<6>(a, start, lane, w, sum2) };
            let w = vector_q3_group::<7>(q1, h, s3, s5, d);
            sum3 = unsafe { vector_quantized_fma::<7>(a, start, lane, w, sum3) };
        } else if FORMAT == 12u32 || FORMAT == 13u32 {
            let d = unsafe { quant_half_128(b, base, mask) };
            let dmin = unsafe { quant_half_128(b, base + broadcast_scalar(2u32, shape![128]), mask) };
            let s0 = unsafe {
                read_tile(
                    word_pointer(b),
                    (base + broadcast_scalar(4u32, shape![128])) / broadcast_scalar(4u32, shape![128]),
                    mask,
                    0u32,
                )
            };
            let s1 = unsafe {
                read_tile(
                    word_pointer(b),
                    (base + broadcast_scalar(8u32, shape![128])) / broadcast_scalar(4u32, shape![128]),
                    mask,
                    0u32,
                )
            };
            let s2 = unsafe {
                read_tile(
                    word_pointer(b),
                    (base + broadcast_scalar(12u32, shape![128])) / broadcast_scalar(4u32, shape![128]),
                    mask,
                    0u32,
                )
            };
            let offset = if FORMAT == 13u32 { 48u32 } else { 16u32 };
            let q0 = unsafe { quant_byte_128(b, base + broadcast_scalar(offset + 0u32, shape![128]) + lane, mask) };
            let q1 = unsafe { quant_byte_128(b, base + broadcast_scalar(offset + 32u32, shape![128]) + lane, mask) };
            let q2 = unsafe { quant_byte_128(b, base + broadcast_scalar(offset + 64u32, shape![128]) + lane, mask) };
            let q3 = unsafe { quant_byte_128(b, base + broadcast_scalar(offset + 96u32, shape![128]) + lane, mask) };
            let h = if FORMAT == 13u32 {
                unsafe { quant_byte_128(b, base + broadcast_scalar(16u32, shape![128]) + lane, mask) }
            } else {
                broadcast_scalar(0u32, shape![128])
            };
            let w = vector_q4_group::<FORMAT, 0>(q0, h, s0, s1, s2, d, dmin);
            sum0 = unsafe { vector_quantized_fma::<0>(a, start, lane, w, sum0) };
            let w = vector_q4_group::<FORMAT, 1>(q0, h, s0, s1, s2, d, dmin);
            sum1 = unsafe { vector_quantized_fma::<1>(a, start, lane, w, sum1) };
            let w = vector_q4_group::<FORMAT, 2>(q1, h, s0, s1, s2, d, dmin);
            sum2 = unsafe { vector_quantized_fma::<2>(a, start, lane, w, sum2) };
            let w = vector_q4_group::<FORMAT, 3>(q1, h, s0, s1, s2, d, dmin);
            sum3 = unsafe { vector_quantized_fma::<3>(a, start, lane, w, sum3) };
            let w = vector_q4_group::<FORMAT, 4>(q2, h, s0, s1, s2, d, dmin);
            sum0 = unsafe { vector_quantized_fma::<4>(a, start, lane, w, sum0) };
            let w = vector_q4_group::<FORMAT, 5>(q2, h, s0, s1, s2, d, dmin);
            sum1 = unsafe { vector_quantized_fma::<5>(a, start, lane, w, sum1) };
            let w = vector_q4_group::<FORMAT, 6>(q3, h, s0, s1, s2, d, dmin);
            sum2 = unsafe { vector_quantized_fma::<6>(a, start, lane, w, sum2) };
            let w = vector_q4_group::<FORMAT, 7>(q3, h, s0, s1, s2, d, dmin);
            sum3 = unsafe { vector_quantized_fma::<7>(a, start, lane, w, sum3) };
        } else {
            let d = unsafe { quant_half_128(b, base + broadcast_scalar(208u32, shape![128]), mask) };
            let q0 = unsafe { quant_byte_128(b, base + broadcast_scalar(0u32, shape![128]) + lane, mask) };
            let q1 = unsafe { quant_byte_128(b, base + broadcast_scalar(32u32, shape![128]) + lane, mask) };
            let q2 = unsafe { quant_byte_128(b, base + broadcast_scalar(64u32, shape![128]) + lane, mask) };
            let q3 = unsafe { quant_byte_128(b, base + broadcast_scalar(96u32, shape![128]) + lane, mask) };
            let h0 = unsafe { quant_byte_128(b, base + broadcast_scalar(128u32, shape![128]) + lane, mask) };
            let h1 = unsafe { quant_byte_128(b, base + broadcast_scalar(160u32, shape![128]) + lane, mask) };
            let scale = unsafe { quant_byte_128(b, base + broadcast_scalar(192u32, shape![128]) + half, mask) };
            let w = vector_q6_group::<0>(q0, h0, scale, d);
            sum0 = unsafe { vector_quantized_fma::<0>(a, start, lane, w, sum0) };
            let scale = unsafe { quant_byte_128(b, base + broadcast_scalar(194u32, shape![128]) + half, mask) };
            let w = vector_q6_group::<1>(q1, h0, scale, d);
            sum1 = unsafe { vector_quantized_fma::<1>(a, start, lane, w, sum1) };
            let scale = unsafe { quant_byte_128(b, base + broadcast_scalar(196u32, shape![128]) + half, mask) };
            let w = vector_q6_group::<2>(q0, h0, scale, d);
            sum2 = unsafe { vector_quantized_fma::<2>(a, start, lane, w, sum2) };
            let scale = unsafe { quant_byte_128(b, base + broadcast_scalar(198u32, shape![128]) + half, mask) };
            let w = vector_q6_group::<3>(q1, h0, scale, d);
            sum3 = unsafe { vector_quantized_fma::<3>(a, start, lane, w, sum3) };
            let scale = unsafe { quant_byte_128(b, base + broadcast_scalar(200u32, shape![128]) + half, mask) };
            let w = vector_q6_group::<4>(q2, h1, scale, d);
            sum0 = unsafe { vector_quantized_fma::<4>(a, start, lane, w, sum0) };
            let scale = unsafe { quant_byte_128(b, base + broadcast_scalar(202u32, shape![128]) + half, mask) };
            let w = vector_q6_group::<5>(q3, h1, scale, d);
            sum1 = unsafe { vector_quantized_fma::<5>(a, start, lane, w, sum1) };
            let scale = unsafe { quant_byte_128(b, base + broadcast_scalar(204u32, shape![128]) + half, mask) };
            let w = vector_q6_group::<6>(q2, h1, scale, d);
            sum2 = unsafe { vector_quantized_fma::<6>(a, start, lane, w, sum2) };
            let scale = unsafe { quant_byte_128(b, base + broadcast_scalar(206u32, shape![128]) + half, mask) };
            let w = vector_q6_group::<7>(q3, h1, scale, d);
            sum3 = unsafe { vector_quantized_fma::<7>(a, start, lane, w, sum3) };
        }
        block = block + 1u32;
    }
    let sum = (sum0 + sum1) + (sum2 + sum3);
    let sum: Tile<f32, { [4, 32] }> = sum.reshape(shape![4, 32]);
    let value: Tile<f32, { [4] }> = reduce_sum(sum, 1i32);
    let col = lanes::<4>() + broadcast_scalar(first, shape![4]);
    unsafe { write_tile(output, col, value, lt_tile(col, broadcast_scalar(cols, shape![4]))) };
}
