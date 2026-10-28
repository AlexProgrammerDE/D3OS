// Q8_1 activations use llama.cpp's transposed 144-byte MMQ D4 blocks.
#[cutile::entry()]
unsafe fn q8_d4_pack(input: *const f32, storage: *mut f32, rows: u32, inner: u32, _cols: u32, _format: u32, _phase: u32, _components: u32) {
    let pointer: PointerTile<*mut f32, { [] }> = pointer_to_tile(storage);
    let bytes: PointerTile<*mut i8, { [] }> = ptr_to_ptr(pointer);
    let output = tile_to_pointer(bytes);
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
    let value = unsafe { read_matrix_tile(input, index, mask, 0f32) };
    let maxima: Tile<f32, { [4] }> = reduce_max(absf(value), 1i32);
    let inverse = select(
        eq_tile(maxima, broadcast_scalar(0f32, shape![4])),
        broadcast_scalar(0f32, shape![4]),
        divf(broadcast_scalar(127f32, shape![4]), maxima, rounding::NearestEven, ftz::Disabled),
    );
    let scale = select(
        eq_tile(maxima, broadcast_scalar(0f32, shape![4])),
        broadcast_scalar(0f32, shape![4]),
        divf(broadcast_scalar(1f32, shape![4]), inverse, rounding::NearestEven, ftz::Disabled),
    );
    let block = id % blocks;
    let scale_index = (row + broadcast_scalar(block / 4u32 * rows, shape![4])) * broadcast_scalar(36u32, shape![4]) + broadcast_scalar(block % 4u32, shape![4]);
    unsafe { write_tile(storage, scale_index, scale, lt_tile(row, broadcast_scalar(rows, shape![4]))) };
    let inverse: Tile<f32, { [4, 1] }> = inverse.reshape(shape![4, 1]);
    let inverse: Tile<f32, { [4, 32] }> = inverse.broadcast(shape![4, 32]);
    let scaled = fma(value, inverse, broadcast_scalar(0f32, shape![4, 32]), rounding::NearestEven, ftz::Disabled);
    let integer: Tile<i32, { [4, 32] }> = ftoi(scaled, rounding::NearestIntToZero);
    let whole: Tile<f32, { [4, 32] }> = itof(integer, rounding::NearestEven);
    let step = select(
        lt_tile(scaled, broadcast_scalar(0f32, shape![4, 32])),
        broadcast_scalar(-1i32, shape![4, 32]),
        broadcast_scalar(1i32, shape![4, 32]),
    );
    let rounded = integer
        + select(
            ge_tile(absf(scaled - whole), broadcast_scalar(0.5f32, shape![4, 32])),
            step,
            broadcast_scalar(0i32, shape![4, 32]),
        );
    let quantized: Tile<i8, { [4, 32] }> = trunci(rounded, overflow::NoSignedWrap);
    let lane = k - broadcast_scalar(block * 32u32, shape![4, 32]);
    let address = (r + broadcast_scalar(block / 4u32 * rows, shape![4, 32])) * broadcast_scalar(144u32, shape![4, 32])
        + broadcast_scalar(16u32 + block % 4u32 * 32u32, shape![4, 32])
        + lane;
    unsafe { write_matrix(output, address, quantized, mask, shape![128]) };
}

#[cutile::entry()]
unsafe fn q8_columns32_d4(storage: *const f32, b: *const f32, output: *mut f32, rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32) {
    let pointer: PointerTile<*const f32, { [] }> = pointer_to_tile(storage);
    let mutable: PointerTile<*mut f32, { [] }> = cast_tile_mut(pointer);
    let bytes: PointerTile<*mut i8, { [] }> = ptr_to_ptr(mutable);
    let bytes: PointerTile<*const i8, { [] }> = cast_tile_const(bytes);
    let a = tile_to_pointer(bytes);
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
    let local_columns: Tile<u32, { [32, 1] }> = lanes::<32>().reshape(shape![32, 1]);
    let local_columns: Tile<u32, { [32, 32] }> = local_columns.broadcast(shape![32, 32]);
    let block_tile = (id % ntiles) * blocks * 32u32;
    for step in 0u32..blocks {
        let ai = (r + broadcast_scalar(step / 4u32 * rows, shape![32, 32])) * broadcast_scalar(144u32, shape![32, 32])
            + k
            + broadcast_scalar(16u32 + step % 4u32 * 32u32, shape![32, 32]);
        let qb = local_columns + broadcast_scalar(block_tile + step * 32u32, shape![32, 32]);
        let bytes = qb * broadcast_scalar(34u32, shape![32, 32]) + broadcast_scalar(2u32, shape![32, 32]) + k;
        let scale_address = (lanes::<32>() + broadcast_scalar(block_tile + step * 32u32, shape![32])) * broadcast_scalar(17u32, shape![32]);
        let raw = unsafe { read_matrix_tile(byte_pointer(quantized_word_pointer(b)), bytes, bm, 0u8) };
        let raw: Tile<i8, { [32, 32] }> = bitcast(raw);
        let bq = raw.transpose();
        let sa = unsafe {
            read_tile(
                storage,
                (rr + broadcast_scalar(step / 4u32 * rows, shape![32])) * broadcast_scalar(36u32, shape![32]) + broadcast_scalar(step % 4u32, shape![32]),
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
        sum = fma(v0 * sb, sa, sum, rounding::NearestEven, ftz::Disabled);
    }
    let rr: Tile<u32, { [32, 1] }> = rr.reshape(shape![32, 1]);
    let rr: Tile<u32, { [32, 32] }> = rr.broadcast(shape![32, 32]);
    let cc: Tile<u32, { [1, 32] }> = cc.reshape(shape![1, 32]);
    let cc: Tile<u32, { [32, 32] }> = cc.broadcast(shape![32, 32]);
    let mask = lt_tile(rr, broadcast_scalar(rows, shape![32, 32])) & lt_tile(cc, broadcast_scalar(cols, shape![32, 32]));
    unsafe { write_matrix(output, rr * broadcast_scalar(cols, shape![32, 32]) + cc, sum, mask, shape![1024]) };
}

#[cutile::entry()]
unsafe fn q8_rowmajor_d4(storage: *const f32, b: *const f32, output: *mut f32, rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32) {
    let pointer: PointerTile<*const f32, { [] }> = pointer_to_tile(storage);
    let mutable: PointerTile<*mut f32, { [] }> = cast_tile_mut(pointer);
    let bytes: PointerTile<*mut i8, { [] }> = ptr_to_ptr(mutable);
    let bytes: PointerTile<*const i8, { [] }> = cast_tile_const(bytes);
    let a = tile_to_pointer(bytes);
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
    for step in 0u32..blocks {
        let ai = (r + broadcast_scalar(step / 4u32 * rows, shape![32, 32])) * broadcast_scalar(144u32, shape![32, 32])
            + k
            + broadcast_scalar(16u32 + step % 4u32 * 32u32, shape![32, 32]);
        let qb = c * broadcast_scalar(blocks, shape![32, 32]) + broadcast_scalar(step, shape![32, 32]);
        let bytes = qb * broadcast_scalar(34u32, shape![32, 32]) + broadcast_scalar(2u32, shape![32, 32]) + k;
        let scale_address = (cc * broadcast_scalar(blocks, shape![32]) + broadcast_scalar(step, shape![32])) * broadcast_scalar(17u32, shape![32]);
        let raw = unsafe { read_matrix_tile(byte_pointer(quantized_word_pointer(b)), bytes, bm, 0u8) };
        let raw: Tile<i8, { [32, 32] }> = bitcast(raw);
        let bq = raw.transpose();
        let sa = unsafe {
            read_tile(
                storage,
                (rr + broadcast_scalar(step / 4u32 * rows, shape![32])) * broadcast_scalar(36u32, shape![32]) + broadcast_scalar(step % 4u32, shape![32]),
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
        sum = fma(v0 * sb, sa, sum, rounding::NearestEven, ftz::Disabled);
    }
    let rr: Tile<u32, { [32, 1] }> = rr.reshape(shape![32, 1]);
    let rr: Tile<u32, { [32, 32] }> = rr.broadcast(shape![32, 32]);
    let cc: Tile<u32, { [1, 32] }> = cc.reshape(shape![1, 32]);
    let cc: Tile<u32, { [32, 32] }> = cc.broadcast(shape![32, 32]);
    let mask = lt_tile(rr, broadcast_scalar(rows, shape![32, 32])) & lt_tile(cc, broadcast_scalar(cols, shape![32, 32]));
    unsafe { write_matrix(output, rr * broadcast_scalar(cols, shape![32, 32]) + cc, sum, mask, shape![1024]) };
}
