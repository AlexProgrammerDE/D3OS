// Tensor Core matrix products retain FP32 operands through TF32 decomposition.
unsafe fn compensated_matrix<const TRANSPOSE: bool, const BF16: bool, const MATRIX_LOAD: bool>(
    a: *const f32, b: *const f32, output: *mut f32, rows: u32, inner: u32, cols: u32, splits: u32,
) {
    let block: u32 = block_id();
    let tile_cols = (cols + 31u32) / 32u32;
    let tile_rows = (rows + 15u32) / 16u32;
    let tiles = tile_rows * tile_cols;
    let part = block / tiles;
    let tile = block % tiles;
    let r: Tile<u32, { [16] }> = lanes::<16>() + broadcast_scalar(tile / tile_cols * 16u32, shape![16]);
    let r: Tile<u32, { [16, 1] }> = r.reshape(shape![16, 1]);
    let r: Tile<u32, { [16, 8] }> = r.broadcast(shape![16, 8]);
    let c: Tile<u32, { [32] }> = lanes::<32>() + broadcast_scalar(tile % tile_cols * 32u32, shape![32]);
    let c: Tile<u32, { [1, 32] }> = c.reshape(shape![1, 32]);
    let c: Tile<u32, { [8, 32] }> = c.broadcast(shape![8, 32]);
    let ka: Tile<u32, { [8] }> = lanes::<8>();
    let ka: Tile<u32, { [1, 8] }> = ka.reshape(shape![1, 8]);
    let ka: Tile<u32, { [16, 8] }> = ka.broadcast(shape![16, 8]);
    let kb: Tile<u32, { [8] }> = lanes::<8>();
    let kb: Tile<u32, { [8, 1] }> = kb.reshape(shape![8, 1]);
    let kb: Tile<u32, { [8, 32] }> = kb.broadcast(shape![8, 32]);
    let ktiles = (inner + 8u32 - 1u32) / (8u32);
    let chunk = (ktiles + splits - 1u32) / splits;
    let first = part * chunk;
    let end = min(first + chunk, ktiles);
    let mut sum: Tile<f32, { [16, 32] }> = broadcast_scalar(0f32, shape![16, 32]);
    let mut correction: Tile<f32, { [16, 32] }> = broadcast_scalar(0f32, shape![16, 32]);
    for step in first..end {
        let ak = ka + broadcast_scalar(step * 8u32, shape![16, 8]);
        let bk = kb + broadcast_scalar(step * 8u32, shape![8, 32]);
        let am = lt_tile(r, broadcast_scalar(rows, shape![16, 8])) & lt_tile(ak, broadcast_scalar(inner, shape![16, 8]));
        let bm = lt_tile(c, broadcast_scalar(cols, shape![8, 32])) & lt_tile(bk, broadcast_scalar(inner, shape![8, 32]));
        let ai = r * broadcast_scalar(inner, shape![16, 8]) + ak;
        let av: Tile<f32, { [16, 8] }> = if MATRIX_LOAD {
            unsafe { read_matrix_tile(a, ai, am, 0f32) }
        } else {
            unsafe { read_matrix(a, ai, am, 0f32, shape![128]) }
        };
        let bi = if TRANSPOSE {
            c * broadcast_scalar(inner, shape![8, 32]) + bk
        } else {
            bk * broadcast_scalar(cols, shape![8, 32]) + c
        };
        let bv: Tile<f32, { [8, 32] }> = unsafe { read_weights::<BF16, MATRIX_LOAD, 8, 32, 256>(b, bi, bm, shape![256]) };
        let ah: Tile<tf32, { [16, 8] }> = ftof(av, rounding::NearestAway);
        let bh: Tile<tf32, { [8, 32] }> = ftof(bv, rounding::NearestAway);
        let ar: Tile<f32, { [16, 8] }> = ftof(ah, rounding::NearestEven);
        let av: Tile<f32, { [16, 8] }> = av.reshape(shape![16, 8]);
        let al: Tile<tf32, { [16, 8] }> = ftof(av - ar, rounding::NearestAway);
        let zero: Tile<f32, { [16, 32] }> = broadcast_scalar(0f32, shape![16, 32]);
        let high: Tile<f32, { [16, 32] }> = mmaf(ah, bh, zero);
        let mut low: Tile<f32, { [16, 32] }> = mmaf(al, bh, zero);
        if BF16 == false {
            let br: Tile<f32, { [8, 32] }> = ftof(bh, rounding::NearestEven);
            let bv: Tile<f32, { [8, 32] }> = bv.reshape(shape![8, 32]);
            let bl: Tile<tf32, { [8, 32] }> = ftof(bv - br, rounding::NearestAway);
            low = mmaf(ah, bl, low);
        }
        let adjusted = high - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
        let adjusted = low - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    let rr: Tile<u32, { [16] }> = lanes::<16>() + broadcast_scalar(tile / tile_cols * 16u32, shape![16]);
    let rr: Tile<u32, { [16, 1] }> = rr.reshape(shape![16, 1]);
    let rr: Tile<u32, { [16, 32] }> = rr.broadcast(shape![16, 32]);
    let cc: Tile<u32, { [32] }> = lanes::<32>() + broadcast_scalar(tile % tile_cols * 32u32, shape![32]);
    let cc: Tile<u32, { [1, 32] }> = cc.reshape(shape![1, 32]);
    let cc: Tile<u32, { [16, 32] }> = cc.broadcast(shape![16, 32]);
    let mask = lt_tile(rr, broadcast_scalar(rows, shape![16, 32])) & lt_tile(cc, broadcast_scalar(cols, shape![16, 32]));
    let index = (rr + broadcast_scalar(part * rows, shape![16, 32])) * broadcast_scalar(cols, shape![16, 32]) + cc;
    unsafe { write_matrix(output, index, sum, mask, shape![512]) };
}
#[cutile::entry()]
unsafe fn tensor_matmul_tf32x3(a: *const f32, b: *const f32, out: *mut f32, m: u32, k: u32, n: u32) {
    unsafe { compensated_matrix::<false, false, false>(a, b, out, m, k, n, 1u32) }
}
#[cutile::entry()]
unsafe fn tensor_project_tf32x3(a: *const f32, b: *const f32, out: *mut f32, m: u32, k: u32, n: u32) {
    unsafe { compensated_matrix::<true, false, false>(a, b, out, m, k, n, 1u32) }
}
#[cutile::entry()]
unsafe fn tensor_project_bf16_weights(a: *const f32, b: *const f32, out: *mut f32, m: u32, k: u32, n: u32) {
    unsafe { compensated_matrix::<true, true, true>(a, b, out, m, k, n, 1u32) }
}
// Separate entries keep the matrix tile's shared-memory reservation out of
// every single-token vector launch.
#[cutile::entry()]
unsafe fn vector_project_tf32x3(a: *const f32, b: *const f32, out: *mut f32, _m: u32, k: u32, n: u32) {
    unsafe { vector_project_4::<false, false>(a, b, b, out, k, n) }
}
#[cutile::entry()]
unsafe fn vector_project_bf16_weights(a: *const f32, b: *const f32, out: *mut f32, _m: u32, k: u32, n: u32) {
    unsafe { vector_project_4::<true, false>(a, b, b, out, k, n) }
}
#[cutile::entry()]
unsafe fn project_bf16_split(a: *const f32, b: *const f32, out: *mut f32, m: u32, k: u32, n: u32, splits: u32) {
    unsafe { compensated_matrix::<true, true, true>(a, b, out, m, k, n, splits) }
}
#[cutile::entry()]
unsafe fn sum_projection_partials(input: *const f32, output: *mut f32, rows: u32, cols: u32, splits: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(rows * cols, shape![128]));
    let mut sum: Tile<f32, { [128] }> = broadcast_scalar(0f32, shape![128]);
    let mut correction: Tile<f32, { [128] }> = broadcast_scalar(0f32, shape![128]);
    for part in 0u32..splits {
        let value = unsafe { read_tile(input, i + broadcast_scalar(part * rows * cols, shape![128]), mask, 0f32) };
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    unsafe { write_tile(output, i, sum, mask) };
}
#[cutile::entry()]
unsafe fn gated_project_bf16_weights(a: *const f32, gate: *const f32, up: *const f32, output: *mut f32, inner: u32, cols: u32) {
    unsafe { vector_project_4::<true, true>(a, gate, up, output, inner, cols) }
}

#[cutile::entry()]
unsafe fn vector_project_wide_tf32x3(a: *const f32, b: *const f32, out: *mut f32, _m: u32, k: u32, n: u32) {
    unsafe { vector_project_32::<false, false>(a, b, b, out, k, n) }
}
#[cutile::entry()]
unsafe fn vector_project_wide_bf16_weights(a: *const f32, b: *const f32, out: *mut f32, _m: u32, k: u32, n: u32) {
    unsafe { paired_project_32::<false>(a, b, b, out, k, n) }
}

#[cutile::entry()]
unsafe fn vector_project_unpaired_wide_bf16_weights(a: *const f32, b: *const f32, out: *mut f32, _m: u32, k: u32, n: u32) {
    unsafe { vector_project_32::<true, false>(a, b, b, out, k, n) }
}
#[cutile::entry()]
unsafe fn gated_project_paired_bf16_weights(a: *const f32, gate: *const f32, up: *const f32, output: *mut f32, inner: u32, cols: u32) {
    unsafe { paired_project_4::<true>(a, gate, up, output, inner, cols) }
}
