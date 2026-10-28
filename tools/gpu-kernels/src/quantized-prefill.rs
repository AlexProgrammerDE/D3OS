// Three signed-byte components retain the FP32 input residuals.
// Storage is three row-major byte planes followed by one FP32 scale per 32 inputs.
fn quantized_word_pointer(p: *const f32) -> *const u32 {
    let p: PointerTile<*const f32, { [] }> = pointer_to_tile(p);
    let p: PointerTile<*mut f32, { [] }> = cast_tile_mut(p);
    let p: PointerTile<*mut u32, { [] }> = ptr_to_ptr(p);
    let p: PointerTile<*const u32, { [] }> = cast_tile_const(p);
    tile_to_pointer(p)
}

#[cutile::entry()]
unsafe fn q8_components_pack(input: *const f32, storage: *mut f32, rows: u32, inner: u32, _cols: u32, _format: u32, _phase: u32, _components: u32) {
    let pointer: PointerTile<*mut f32, { [] }> = pointer_to_tile(storage);
    let bytes: PointerTile<*mut i8, { [] }> = ptr_to_ptr(pointer);
    let output = tile_to_pointer(bytes);
    let scale_offset = 3u32 * rows * inner / 4u32;
    let id = block_id();
    let blocks = inner / 32u32;
    let row = lanes::<4>() + broadcast_scalar(id / blocks * 4u32, shape![4]);
    let r: Tile<u32, { [4, 1] }> = row.reshape(shape![4, 1]);
    let r: Tile<u32, { [4, 32] }> = r.broadcast(shape![4, 32]);
    let k = lanes::<32>() + broadcast_scalar((id % blocks) * 32u32, shape![32]);
    let k: Tile<u32, { [1, 32] }> = k.reshape(shape![1, 32]);
    let k: Tile<u32, { [4, 32] }> = k.broadcast(shape![4, 32]);
    let mask = lt_tile(r, broadcast_scalar(rows, shape![4, 32]));
    let index = r * broadcast_scalar(inner, shape![4, 32]) + k;
    let mut residual = unsafe { read_matrix_tile(input, index, mask, 0f32) };
    let maxima: Tile<f32, { [4] }> = reduce_max(absf(residual), 1i32);
    let scale = select(
        eq_tile(maxima, broadcast_scalar(0f32, shape![4])),
        broadcast_scalar(1f32, shape![4]),
        divf(maxima, broadcast_scalar(127f32, shape![4]), rounding::NearestEven, ftz::Disabled),
    );
    unsafe {
        write_tile(
            storage,
            row * broadcast_scalar(blocks, shape![4]) + broadcast_scalar(id % blocks + scale_offset, shape![4]),
            scale,
            lt_tile(row, broadcast_scalar(rows, shape![4])),
        )
    };
    let scale: Tile<f32, { [4, 1] }> = scale.reshape(shape![4, 1]);
    let scale: Tile<f32, { [4, 32] }> = scale.broadcast(shape![4, 32]);
    let s0 = scale;
    let scaled0 = divf(residual, s0, rounding::NearestEven, ftz::Disabled);
    let bias0 = select(
        lt_tile(scaled0, broadcast_scalar(0f32, shape![4, 32])),
        broadcast_scalar(-0.5f32, shape![4, 32]),
        broadcast_scalar(0.5f32, shape![4, 32]),
    );
    let q0: Tile<i32, { [4, 32] }> = ftoi(scaled0 + bias0, rounding::NearestIntToZero);
    let qf0: Tile<f32, { [4, 32] }> = itof(q0, rounding::NearestEven);
    let qi0: Tile<i8, { [4, 32] }> = trunci(q0, overflow::NoSignedWrap);
    unsafe { write_matrix(output, index + broadcast_scalar(0u32, shape![4, 32]), qi0, mask, shape![128]) };
    residual = fma(qf0, broadcast_scalar(0f32, shape![4, 32]) - s0, residual, rounding::NearestEven, ftz::Disabled);
    let s1 = scale / broadcast_scalar(128f32, shape![4, 32]);
    let scaled1 = divf(residual, s1, rounding::NearestEven, ftz::Disabled);
    let bias1 = select(
        lt_tile(scaled1, broadcast_scalar(0f32, shape![4, 32])),
        broadcast_scalar(-0.5f32, shape![4, 32]),
        broadcast_scalar(0.5f32, shape![4, 32]),
    );
    let q1: Tile<i32, { [4, 32] }> = ftoi(scaled1 + bias1, rounding::NearestIntToZero);
    let qf1: Tile<f32, { [4, 32] }> = itof(q1, rounding::NearestEven);
    let qi1: Tile<i8, { [4, 32] }> = trunci(q1, overflow::NoSignedWrap);
    unsafe { write_matrix(output, index + broadcast_scalar(rows * inner, shape![4, 32]), qi1, mask, shape![128]) };
    residual = fma(qf1, broadcast_scalar(0f32, shape![4, 32]) - s1, residual, rounding::NearestEven, ftz::Disabled);
    let s2 = scale / broadcast_scalar(16384f32, shape![4, 32]);
    let scaled2 = divf(residual, s2, rounding::NearestEven, ftz::Disabled);
    let bias2 = select(
        lt_tile(scaled2, broadcast_scalar(0f32, shape![4, 32])),
        broadcast_scalar(-0.5f32, shape![4, 32]),
        broadcast_scalar(0.5f32, shape![4, 32]),
    );
    let q2: Tile<i32, { [4, 32] }> = ftoi(scaled2 + bias2, rounding::NearestIntToZero);
    let qi2: Tile<i8, { [4, 32] }> = trunci(q2, overflow::NoSignedWrap);
    unsafe { write_matrix(output, index + broadcast_scalar(2u32 * rows * inner, shape![4, 32]), qi2, mask, shape![128]) };
}

#[cutile::entry()]
unsafe fn q8_components_project(
    storage: *const f32, b: *const f32, output: *mut f32, rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32,
) {
    let pointer: PointerTile<*const f32, { [] }> = pointer_to_tile(storage);
    let mutable: PointerTile<*mut f32, { [] }> = cast_tile_mut(pointer);
    let bytes: PointerTile<*mut i8, { [] }> = ptr_to_ptr(mutable);
    let bytes: PointerTile<*const i8, { [] }> = cast_tile_const(bytes);
    let a = tile_to_pointer(bytes);
    let scale_offset = 3u32 * rows * inner / 4u32;
    let id = block_id();
    let ntiles = (cols + 31u32) / 32u32;
    let blocks = inner / 32u32;
    let rr = lanes::<32>() + broadcast_scalar(id / ntiles * 32u32, shape![32]);
    let cc = lanes::<32>() + broadcast_scalar(id % ntiles * 32u32, shape![32]);
    let r: Tile<u32, { [32, 1] }> = rr.reshape(shape![32, 1]);
    let r: Tile<u32, { [32, 32] }> = r.broadcast(shape![32, 32]);
    let c: Tile<u32, { [32, 1] }> = cc.reshape(shape![32, 1]);
    let c: Tile<u32, { [32, 32] }> = c.broadcast(shape![32, 32]);
    let kk = lanes::<32>();
    let k: Tile<u32, { [1, 32] }> = kk.reshape(shape![1, 32]);
    let k: Tile<u32, { [32, 32] }> = k.broadcast(shape![32, 32]);
    let am = lt_tile(r, broadcast_scalar(rows, shape![32, 32]));
    let bm = lt_tile(c, broadcast_scalar(cols, shape![32, 32]));
    let mut sum = broadcast_scalar(0f32, shape![32, 32]);
    let mut correction = sum;
    for step in 0u32..blocks {
        let ai = r * broadcast_scalar(inner, shape![32, 32]) + k + broadcast_scalar(step * 32u32, shape![32, 32]);
        let qb = c * broadcast_scalar(blocks, shape![32, 32]) + broadcast_scalar(step, shape![32, 32]);
        let bytes = qb * broadcast_scalar(34u32, shape![32, 32]) + broadcast_scalar(2u32, shape![32, 32]) + k;
        let scale_address = (cc * broadcast_scalar(blocks, shape![32]) + broadcast_scalar(step, shape![32])) * broadcast_scalar(17u32, shape![32]);
        let raw = unsafe { read_matrix_tile(byte_pointer(quantized_word_pointer(b)), bytes, bm, 0u8) };
        let raw: Tile<i8, { [32, 32] }> = bitcast(raw);
        let bq = raw.transpose();
        let sa = unsafe {
            read_tile(
                storage,
                rr * broadcast_scalar(blocks, shape![32]) + broadcast_scalar(step + scale_offset, shape![32]),
                lt_tile(rr, broadcast_scalar(rows, shape![32])),
                0f32,
            )
        };
        let sa: Tile<f32, { [32, 1] }> = sa.reshape(shape![32, 1]);
        let sa: Tile<f32, { [32, 32] }> = sa.broadcast(shape![32, 32]);
        let sb = unsafe { read_tile(bf16_pointer(b), scale_address, lt_tile(cc, broadcast_scalar(cols, shape![32])), 0u16) };
        let sb: Tile<f16, { [32] }> = bitcast(sb);
        let sb: Tile<f32, { [32] }> = ftof(sb, rounding::NearestEven);
        let sb: Tile<f32, { [1, 32] }> = sb.reshape(shape![1, 32]);
        let sb: Tile<f32, { [32, 32] }> = sb.broadcast(shape![32, 32]);
        let zero = broadcast_scalar(0i32, shape![32, 32]);
        let a0 = unsafe { read_matrix_tile(a, ai + broadcast_scalar(0u32, shape![32, 32]), am, 0i8) };
        let d0 = mmai(a0, bq, zero, signedness::Signed, signedness::Signed);
        let v0: Tile<f32, { [32, 32] }> = itof(d0, rounding::NearestEven);
        let unit0 = sa / broadcast_scalar(1f32, shape![32, 32]);
        let value0 = (v0 * unit0) * sb;
        let adjusted = value0 - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
        let a1 = unsafe { read_matrix_tile(a, ai + broadcast_scalar(rows * inner, shape![32, 32]), am, 0i8) };
        let d1 = mmai(a1, bq, zero, signedness::Signed, signedness::Signed);
        let v1: Tile<f32, { [32, 32] }> = itof(d1, rounding::NearestEven);
        let unit1 = sa / broadcast_scalar(128f32, shape![32, 32]);
        let value1 = (v1 * unit1) * sb;
        let adjusted = value1 - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
        let a2 = unsafe { read_matrix_tile(a, ai + broadcast_scalar(2u32 * rows * inner, shape![32, 32]), am, 0i8) };
        let d2 = mmai(a2, bq, zero, signedness::Signed, signedness::Signed);
        let v2: Tile<f32, { [32, 32] }> = itof(d2, rounding::NearestEven);
        let unit2 = sa / broadcast_scalar(16384f32, shape![32, 32]);
        let value2 = (v2 * unit2) * sb;
        let adjusted = value2 - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    let rr: Tile<u32, { [32, 1] }> = rr.reshape(shape![32, 1]);
    let rr: Tile<u32, { [32, 32] }> = rr.broadcast(shape![32, 32]);
    let cc: Tile<u32, { [1, 32] }> = cc.reshape(shape![1, 32]);
    let cc: Tile<u32, { [32, 32] }> = cc.broadcast(shape![32, 32]);
    let mask = lt_tile(rr, broadcast_scalar(rows, shape![32, 32])) & lt_tile(cc, broadcast_scalar(cols, shape![32, 32]));
    unsafe { write_matrix(output, rr * broadcast_scalar(cols, shape![32, 32]) + cc, sum, mask, shape![1024]) };
}
