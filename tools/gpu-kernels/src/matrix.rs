#[cutile::entry()]
unsafe fn tensor_matmul_tf32x3(a: *const f32, b: *const f32, out: *mut f32, m: u32, k: u32, n: u32) {
    unsafe { compensated_matrix_8::<false, false, false>(a, b, out, m, k, n, 1u32) }
}
#[cutile::entry()]
unsafe fn tensor_project_tf32x3(a: *const f32, b: *const f32, out: *mut f32, m: u32, k: u32, n: u32) {
    unsafe { compensated_matrix_8::<true, false, false>(a, b, out, m, k, n, 1u32) }
}
#[cutile::entry()]
unsafe fn tensor_project_bf16_weights(a: *const f32, b: *const f32, out: *mut f32, m: u32, k: u32, n: u32) {
    // Longer reductions amortize the compensated sum in regular projections.
    unsafe { compensated_matrix_16::<true, true, true>(a, b, out, m, k, n, 1u32) }
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
    // Short split reductions measured faster with the smaller tile.
    unsafe { compensated_matrix_8::<true, true, true>(a, b, out, m, k, n, splits) }
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
