// Dedicated low-precision Tensor Core checks use FP32 accumulators.
unsafe fn rounded_matrix<const BF16: bool>(a: *const f32, b: *const f32, out: *mut f32, rows: u32, inner: u32, cols: u32) {
    let block = block_id();
    let tile_cols = (cols + 31u32) / 32u32;
    let r: Tile<u32, { [16] }> = lanes::<16>() + broadcast_scalar(block / tile_cols * 16u32, shape![16]);
    let r: Tile<u32, { [16, 1] }> = r.reshape(shape![16, 1]);
    let r: Tile<u32, { [16, 32] }> = r.broadcast(shape![16, 32]);
    let c: Tile<u32, { [32] }> = lanes::<32>() + broadcast_scalar(block % tile_cols * 32u32, shape![32]);
    let c: Tile<u32, { [1, 32] }> = c.reshape(shape![1, 32]);
    let c: Tile<u32, { [32, 32] }> = c.broadcast(shape![32, 32]);
    let k: Tile<u32, { [32] }> = lanes::<32>();
    let ak: Tile<u32, { [1, 32] }> = k.reshape(shape![1, 32]);
    let ak: Tile<u32, { [16, 32] }> = ak.broadcast(shape![16, 32]);
    let bk: Tile<u32, { [32, 1] }> = k.reshape(shape![32, 1]);
    let bk: Tile<u32, { [32, 32] }> = bk.broadcast(shape![32, 32]);
    let mut sum: Tile<f32, { [16, 32] }> = broadcast_scalar(0f32, shape![16, 32]);
    let mut start = 0u32;
    while start < inner {
        let ka = ak + broadcast_scalar(start, shape![16, 32]);
        let kb = bk + broadcast_scalar(start, shape![32, 32]);
        let am = lt_tile(r, broadcast_scalar(rows, shape![16, 32])) & lt_tile(ka, broadcast_scalar(inner, shape![16, 32]));
        let bm = lt_tile(c, broadcast_scalar(cols, shape![32, 32])) & lt_tile(kb, broadcast_scalar(inner, shape![32, 32]));
        let av = unsafe { read_matrix(a, r * broadcast_scalar(inner, shape![16, 32]) + ka, am, 0f32, shape![512]) };
        let bv = unsafe { read_matrix(b, kb * broadcast_scalar(cols, shape![32, 32]) + c, bm, 0f32, shape![1024]) };
        if BF16 {
            let av: Tile<bf16, { [16, 32] }> = ftof(av, rounding::NearestEven);
            let bv: Tile<bf16, { [32, 32] }> = ftof(bv, rounding::NearestEven);
            sum = mmaf(av, bv, sum);
        } else {
            let av: Tile<f16, { [16, 32] }> = ftof(av, rounding::NearestEven);
            let bv: Tile<f16, { [32, 32] }> = ftof(bv, rounding::NearestEven);
            sum = mmaf(av, bv, sum);
        }
        start = start + 32u32;
    }
    let cc: Tile<u32, { [32] }> = lanes::<32>() + broadcast_scalar(block % tile_cols * 32u32, shape![32]);
    let cc: Tile<u32, { [1, 32] }> = cc.reshape(shape![1, 32]);
    let cc: Tile<u32, { [16, 32] }> = cc.broadcast(shape![16, 32]);
    let mask = lt_tile(r, broadcast_scalar(rows, shape![16, 32])) & lt_tile(cc, broadcast_scalar(cols, shape![16, 32]));
    unsafe { write_matrix(out, r * broadcast_scalar(cols, shape![16, 32]) + cc, sum, mask, shape![512]) };
}
#[cutile::entry()]
unsafe fn tensor_matmul_f16(a: *const f32, b: *const f32, out: *mut f32, rows: u32, inner: u32, cols: u32) {
    unsafe { rounded_matrix::<false>(a, b, out, rows, inner, cols) }
}
#[cutile::entry()]
unsafe fn tensor_matmul_bf16(a: *const f32, b: *const f32, out: *mut f32, rows: u32, inner: u32, cols: u32) {
    unsafe { rounded_matrix::<true>(a, b, out, rows, inner, cols) }
}
