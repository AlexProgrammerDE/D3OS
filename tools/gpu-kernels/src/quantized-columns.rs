// Lossless 32-column Q8 block layout. The three-plane arithmetic is unchanged.
#[cutile::entry()]
unsafe fn q8_columns32_components(
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
    let local_columns: Tile<u32, { [32, 1] }> = lanes::<32>().reshape(shape![32, 1]);
    let local_columns: Tile<u32, { [32, 32] }> = local_columns.broadcast(shape![32, 32]);
    let block_tile = (id % ntiles) * blocks * 32u32;
    for step in 0u32..blocks {
        let ai = r * broadcast_scalar(inner, shape![32, 32]) + k + broadcast_scalar(step * 32u32, shape![32, 32]);
        let qb = local_columns + broadcast_scalar(block_tile + step * 32u32, shape![32, 32]);
        let bytes = qb * broadcast_scalar(34u32, shape![32, 32]) + broadcast_scalar(2u32, shape![32, 32]) + k;
        let scale_address = (lanes::<32>() + broadcast_scalar(block_tile + step * 32u32, shape![32])) * broadcast_scalar(17u32, shape![32]);
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

// Scheduling tier for Q8 weight traffic; zero keeps the compiler default.
unsafe fn read_q8_weight<const LATENCY: i32>(p: *const f32, index: Tile<u32, { [128] }>, mask: Tile<bool, { [128] }>) -> Tile<u16, { [128] }> {
    let base: PointerTile<*const u16, { [] }> = pointer_to_tile(bf16_pointer(p));
    let base: PointerTile<*const u16, { [1] }> = base.reshape(shape![1]);
    let base: PointerTile<*const u16, { [128] }> = base.broadcast(shape![128]);
    let pointer = base.offset_tile(index);
    let (value, _): (Tile<u16, { [128] }>, Token) = unsafe {
        load_ptr_tko(
            pointer,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0u16),
            None,
            Latency::<LATENCY>,
        )
    };
    value
}

unsafe fn blocked_vector_pairs<const BLOCK_GROUP: i32, const GROUP: i32, const NARROW: bool, const SIGNED: bool, const LATENCY: i32>(
    a: *const f32, b: *const f32, col: Tile<u32, { [128] }>, lane: Tile<u32, { [128] }>, block: u32, blocks: u32, cols: u32, input_offset: u32,
    even: Tile<f32, { [128] }>, odd: Tile<f32, { [128] }>,
) -> (Tile<f32, { [128] }>, Tile<f32, { [128] }>) {
    let group_blocks = if NARROW { 8u32 } else { 2u32 };
    let qblock = lane / broadcast_scalar(16u32, shape![128]) + broadcast_scalar(block + unsigned(GROUP) * group_blocks, shape![128]);
    let pair = lane % broadcast_scalar(16u32, shape![128]);
    let mask = lt_tile(col, broadcast_scalar(cols, shape![128])) & lt_tile(qblock, broadcast_scalar(blocks, shape![128]));
    let groups = (blocks + unsigned(BLOCK_GROUP) - 1u32) / unsigned(BLOCK_GROUP);
    let base = (((col / broadcast_scalar(32u32, shape![128])) * broadcast_scalar(groups, shape![128])
        + qblock / broadcast_scalar(unsigned(BLOCK_GROUP), shape![128]))
        * broadcast_scalar(32u32 * unsigned(BLOCK_GROUP), shape![128])
        + (col % broadcast_scalar(32u32, shape![128])) * broadcast_scalar(unsigned(BLOCK_GROUP), shape![128])
        + qblock % broadcast_scalar(unsigned(BLOCK_GROUP), shape![128]))
        * broadcast_scalar(17u32, shape![128]);
    let scale = if LATENCY == 0 {
        unsafe { quant_half_128(b, base * broadcast_scalar(2u32, shape![128]), mask) }
    } else {
        let value = unsafe { read_q8_weight::<LATENCY>(b, base, mask) };
        let value: Tile<f16, { [128] }> = bitcast(value);
        let value: Tile<f32, { [128] }> = ftof(value, rounding::NearestEven);
        value
    };
    let address = base + broadcast_scalar(1u32, shape![128]) + pair;
    let packed = if LATENCY == 0 {
        unsafe { read_tile(bf16_pointer(b), address, mask, 0u16) }
    } else {
        unsafe { read_q8_weight::<LATENCY>(b, address, mask) }
    };
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

unsafe fn blocked_vector_at<const BLOCK_GROUP: i32, const NARROW: bool, const SIGNED: bool, const LATENCY: i32>(
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
        let (even, odd) =
            unsafe { blocked_vector_pairs::<BLOCK_GROUP, 0, NARROW, SIGNED, LATENCY>(a, b, col, lane, block, blocks, cols, input_offset, e0, o0) };
        e0 = even;
        o0 = odd;
        let (even, odd) =
            unsafe { blocked_vector_pairs::<BLOCK_GROUP, 1, NARROW, SIGNED, LATENCY>(a, b, col, lane, block, blocks, cols, input_offset, e1, o1) };
        e1 = even;
        o1 = odd;
        let (even, odd) =
            unsafe { blocked_vector_pairs::<BLOCK_GROUP, 2, NARROW, SIGNED, LATENCY>(a, b, col, lane, block, blocks, cols, input_offset, e2, o2) };
        e2 = even;
        o2 = odd;
        let (even, odd) =
            unsafe { blocked_vector_pairs::<BLOCK_GROUP, 3, NARROW, SIGNED, LATENCY>(a, b, col, lane, block, blocks, cols, input_offset, e3, o3) };
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

#[cutile::entry()]
unsafe fn q8_columns32_vector(a: *const f32, b: *const f32, out: *mut f32, m: u32, k: u32, n: u32, _f: u32) {
    let row = block_id() % m;
    let first = (block_id() / m) * 4u32;
    unsafe { blocked_vector_at::<1, false, true, 0>(a, b, out, k, n, first, row * n, row * k) }
}

#[cutile::entry()]
unsafe fn q8_columns32_vector_streaming(a: *const f32, b: *const f32, out: *mut f32, m: u32, k: u32, n: u32, _f: u32) {
    let row = block_id() % m;
    let first = (block_id() / m) * 4u32;
    unsafe { blocked_vector_at::<1, false, true, 4>(a, b, out, k, n, first, row * n, row * k) }
}

// Translate logical embedding elements to unchanged Q8 blocks in tiled storage.
#[cutile::entry()]
unsafe fn q8_columns32_embedding(input: *const f32, ids: *const u32, output: *mut f32, vocab: u32, width: u32, rows: u32, _format: u32, _layout: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(rows * width, shape![128]));
    let token = unsafe { read_tile(ids, i / broadcast_scalar(width, shape![128]), mask, 0u32) };
    let valid = mask & lt_tile(token, broadcast_scalar(vocab, shape![128]));
    let inner = i % broadcast_scalar(width, shape![128]);
    let block = (token / broadcast_scalar(32u32, shape![128])) * broadcast_scalar(width, shape![128])
        + (inner / broadcast_scalar(32u32, shape![128])) * broadcast_scalar(32u32, shape![128])
        + token % broadcast_scalar(32u32, shape![128]);
    let source = block * broadcast_scalar(32u32, shape![128]) + inner % broadcast_scalar(32u32, shape![128]);
    let value = unsafe { read_quantized_128::<8>(input, source, valid) };
    unsafe { write_tile(output, i, select(valid, value, broadcast_scalar(0f32, shape![128])), mask) };
}

// Share dequantized Q8 weights across two FP32 rows without changing each row's accumulation order.
unsafe fn paired_q8_accumulate<const GROUP: i32>(
    a: *const f32, b: *const f32, col: Tile<u32, { [128] }>, lane: Tile<u32, { [128] }>, block: u32, blocks: u32, cols: u32, row: u32, rows: u32, inner: u32,
    e0: Tile<f32, { [128] }>, o0: Tile<f32, { [128] }>, e1: Tile<f32, { [128] }>, o1: Tile<f32, { [128] }>,
) -> (Tile<f32, { [128] }>, Tile<f32, { [128] }>, Tile<f32, { [128] }>, Tile<f32, { [128] }>) {
    let qblock = lane / broadcast_scalar(16u32, shape![128]) + broadcast_scalar(block + unsigned(GROUP) * 2u32, shape![128]);
    let pair = lane % broadcast_scalar(16u32, shape![128]);
    let mask = lt_tile(col, broadcast_scalar(cols, shape![128])) & lt_tile(qblock, broadcast_scalar(blocks, shape![128]));
    let base = (((col / broadcast_scalar(32u32, shape![128])) * broadcast_scalar(blocks, shape![128]) + qblock) * broadcast_scalar(32u32, shape![128])
        + col % broadcast_scalar(32u32, shape![128]))
        * broadcast_scalar(17u32, shape![128]);
    let scale = unsafe { quant_half_128(b, base * broadcast_scalar(2u32, shape![128]), mask) };
    let packed = unsafe { read_tile(bf16_pointer(b), base + broadcast_scalar(1u32, shape![128]) + pair, mask, 0u16) };
    let packed: Tile<u32, { [128] }> = exti(packed);
    let low = scale * vector_q8_byte::<true>(packed & broadcast_scalar(255u32, shape![128]));
    let high = scale * vector_q8_byte::<true>(packed >> broadcast_scalar(8u32, shape![128]));
    let index = broadcast_scalar(row * inner, shape![128]) + qblock * broadcast_scalar(32u32, shape![128]) + pair * broadcast_scalar(2u32, shape![128]);
    let second = index + broadcast_scalar(inner, shape![128]);
    let second_mask = mask & lt_tile(broadcast_scalar(row + 1u32, shape![128]), broadcast_scalar(rows, shape![128]));
    let x00 = unsafe { read_tile(a, index, mask, 0f32) };
    let x01 = unsafe { read_tile(a, index + broadcast_scalar(1u32, shape![128]), mask, 0f32) };
    let x10 = unsafe { read_tile(a, second, second_mask, 0f32) };
    let x11 = unsafe { read_tile(a, second + broadcast_scalar(1u32, shape![128]), second_mask, 0f32) };
    (
        fma(x00, low, e0, rounding::NearestEven, ftz::Disabled),
        fma(x01, high, o0, rounding::NearestEven, ftz::Disabled),
        fma(x10, low, e1, rounding::NearestEven, ftz::Disabled),
        fma(x11, high, o1, rounding::NearestEven, ftz::Disabled),
    )
}

#[cutile::entry()]
unsafe fn q8_columns32_pair(a: *const f32, b: *const f32, output: *mut f32, rows: u32, inner: u32, cols: u32, _format: u32) {
    let groups = (rows + 1u32) / 2u32;
    let row = (block_id() % groups) * 2u32;
    let first = (block_id() / groups) * 4u32;
    let lane = lanes::<128>() % broadcast_scalar(32u32, shape![128]);
    let col = lanes::<128>() / broadcast_scalar(32u32, shape![128]) + broadcast_scalar(first, shape![128]);
    let blocks = inner / 32u32;
    let mut e00 = broadcast_scalar(0f32, shape![128]);
    let mut o00 = broadcast_scalar(0f32, shape![128]);
    let mut e01 = broadcast_scalar(0f32, shape![128]);
    let mut o01 = broadcast_scalar(0f32, shape![128]);
    let mut e02 = broadcast_scalar(0f32, shape![128]);
    let mut o02 = broadcast_scalar(0f32, shape![128]);
    let mut e03 = broadcast_scalar(0f32, shape![128]);
    let mut o03 = broadcast_scalar(0f32, shape![128]);
    let mut e10 = broadcast_scalar(0f32, shape![128]);
    let mut o10 = broadcast_scalar(0f32, shape![128]);
    let mut e11 = broadcast_scalar(0f32, shape![128]);
    let mut o11 = broadcast_scalar(0f32, shape![128]);
    let mut e12 = broadcast_scalar(0f32, shape![128]);
    let mut o12 = broadcast_scalar(0f32, shape![128]);
    let mut e13 = broadcast_scalar(0f32, shape![128]);
    let mut o13 = broadcast_scalar(0f32, shape![128]);
    let mut block = 0u32;
    while block < blocks {
        let (a0, b0, a1, b1) = unsafe { paired_q8_accumulate::<0>(a, b, col, lane, block, blocks, cols, row, rows, inner, e00, o00, e10, o10) };
        e00 = a0;
        o00 = b0;
        e10 = a1;
        o10 = b1;
        let (a0, b0, a1, b1) = unsafe { paired_q8_accumulate::<1>(a, b, col, lane, block, blocks, cols, row, rows, inner, e01, o01, e11, o11) };
        e01 = a0;
        o01 = b0;
        e11 = a1;
        o11 = b1;
        let (a0, b0, a1, b1) = unsafe { paired_q8_accumulate::<2>(a, b, col, lane, block, blocks, cols, row, rows, inner, e02, o02, e12, o12) };
        e02 = a0;
        o02 = b0;
        e12 = a1;
        o12 = b1;
        let (a0, b0, a1, b1) = unsafe { paired_q8_accumulate::<3>(a, b, col, lane, block, blocks, cols, row, rows, inner, e03, o03, e13, o13) };
        e03 = a0;
        o03 = b0;
        e13 = a1;
        o13 = b1;
        block = block + 8u32;
    }
    let sum = ((e00 + o00) + (e01 + o01)) + ((e02 + o02) + (e03 + o03));
    let sum: Tile<f32, { [4, 32] }> = sum.reshape(shape![4, 32]);
    let value: Tile<f32, { [4] }> = reduce_sum(sum, 1i32);
    let column = lanes::<4>() + broadcast_scalar(first, shape![4]);
    let mask = lt_tile(column, broadcast_scalar(cols, shape![4])) & lt_tile(broadcast_scalar(row + 0u32, shape![4]), broadcast_scalar(rows, shape![4]));
    unsafe { write_tile(output, column + broadcast_scalar((row + 0u32) * cols, shape![4]), value, mask) };
    let sum = ((e10 + o10) + (e11 + o11)) + ((e12 + o12) + (e13 + o13));
    let sum: Tile<f32, { [4, 32] }> = sum.reshape(shape![4, 32]);
    let value: Tile<f32, { [4] }> = reduce_sum(sum, 1i32);
    let column = lanes::<4>() + broadcast_scalar(first, shape![4]);
    let mask = lt_tile(column, broadcast_scalar(cols, shape![4])) & lt_tile(broadcast_scalar(row + 1u32, shape![4]), broadcast_scalar(rows, shape![4]));
    unsafe { write_tile(output, column + broadcast_scalar((row + 1u32) * cols, shape![4]), value, mask) };
}
