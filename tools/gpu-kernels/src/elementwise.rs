// Elementwise, pooling, normalization, selection, and embedding tile programs.
#[cutile::entry()]
unsafe fn vector_add(a: *const f32, b: *const f32, output: *mut f32, count: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(count, shape![128]));
    let a = unsafe { read_tile(a, i, mask, 0f32) };
    let b = unsafe { read_tile(b, i, mask, 0f32) };
    unsafe { write_tile(output, i, a + b, mask) };
}
#[cutile::entry()]
unsafe fn buffer_zero(output: *mut u32, count: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(count, shape![128]));
    unsafe { write_tile(output, i, broadcast_scalar(0u32, shape![128]), mask) };
}
#[cutile::entry()]
unsafe fn tensor_copy(input: *const u32, output: *mut u32, count: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(count, shape![128]));
    let value = unsafe { read_tile(input, i, mask, 0u32) };
    unsafe { write_tile(output, i, value, mask) };
}
#[cutile::entry()]
unsafe fn inference_relu(input: *const u32, output: *mut u32, count: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(count, shape![128]));
    let bits = unsafe { read_tile(input, i, mask, 0u32) };
    // Floating maximum canonicalizes NaNs. Select integer bits so both NaN
    // signs and payloads survive, while either signed zero becomes +0.
    let magnitude = bits & broadcast_scalar(0x7fff_ffffu32, shape![128]);
    let zero = broadcast_scalar(0u32, shape![128]);
    let negative = gt_tile(bits, broadcast_scalar(0x7fff_ffffu32, shape![128]));
    let nonpositive = eq_tile(magnitude, zero) | (negative & le_tile(magnitude, broadcast_scalar(0x7f80_0000u32, shape![128])));
    unsafe { write_tile(output, i, select(nonpositive, zero, bits), mask) };
}
#[cutile::entry()]
unsafe fn tensor_add(input: *const f32, other: *const f32, output: *mut f32, count: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(count, shape![128]));
    let a = unsafe { read_tile(input, i, mask, 0f32) };
    let b = unsafe { read_tile(other, i, mask, 0f32) };
    unsafe { write_tile(output, i, a + b, mask) };
}
#[cutile::entry()]
unsafe fn silu_mul(input: *const f32, other: *const f32, output: *mut f32, count: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(count, shape![128]));
    let a = unsafe { read_tile(input, i, mask, 0f32) };
    let b = unsafe { read_tile(other, i, mask, 0f32) };
    unsafe { write_tile(output, i, silu_tile(a) * b, mask) };
}
#[cutile::entry()]
unsafe fn bias_row(input: *const f32, bias: *const f32, output: *mut f32, count: u32, cols: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(count, shape![128]));
    let value = unsafe { read_tile(input, i, mask, 0f32) };
    let bias = unsafe { read_tile(bias, i % broadcast_scalar(cols, shape![128]), mask, 0f32) };
    unsafe { write_tile(output, i, value + bias, mask) };
}
#[cutile::entry()]
unsafe fn bias_channel(input: *const f32, bias: *const f32, output: *mut f32, count: u32, spatial: u32, channels: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(count, shape![128]));
    let value = unsafe { read_tile(input, i, mask, 0f32) };
    let c = i / broadcast_scalar(spatial, shape![128]) % broadcast_scalar(channels, shape![128]);
    let bias = unsafe { read_tile(bias, c, mask, 0f32) };
    unsafe { write_tile(output, i, value + bias, mask) };
}
#[cutile::entry()]
unsafe fn dequantize_bf16(input: *const u16, output: *mut f32, count: u32, half: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(count, shape![128]));
    let packed = unsafe { read_tile(input, i + broadcast_scalar(half, shape![128]), mask, 0u16) };
    let bits: Tile<u32, { [128] }> = exti(packed);
    let value: Tile<f32, { [128] }> = bitcast(bits << broadcast_scalar(16u32, shape![128]));
    unsafe { write_tile(output, i, value, mask) };
}
#[cutile::entry()]
unsafe fn embedding_gather(table: *const u32, tokens: *const u32, output: *mut f32, vocab: u32, width: u32, rows: u32, packed: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(rows * width, shape![128]));
    let token = unsafe { read_tile(tokens, i / broadcast_scalar(width, shape![128]), mask, 0u32) };
    let valid = mask & lt_tile(token, broadcast_scalar(vocab, shape![128]));
    let offset = token * broadcast_scalar(width, shape![128]) + i % broadcast_scalar(width, shape![128]);
    let value: Tile<f32, { [128] }> = if packed == 1u32 {
        let p: PointerTile<*const u32, { [] }> = pointer_to_tile(table);
        let p: PointerTile<*mut u32, { [] }> = cast_tile_mut(p);
        let p: PointerTile<*mut u16, { [] }> = ptr_to_ptr(p);
        let p: PointerTile<*const u16, { [] }> = cast_tile_const(p);
        let p: *const u16 = tile_to_pointer(p);
        let weights = unsafe { read_tile(p, offset, valid, 0u16) };
        let bits: Tile<u32, { [128] }> = exti(weights);
        let value: Tile<f32, { [128] }> = bitcast(bits << broadcast_scalar(16u32, shape![128]));
        value
    } else {
        let bits = unsafe { read_tile(table, offset, valid, 0u32) };
        let value: Tile<f32, { [128] }> = bitcast(bits);
        value
    };
    unsafe { write_tile(output, i, value, mask) };
}
#[cutile::entry()]
unsafe fn im2col(
    input: *const f32, output: *mut f32, channels: u32, height: u32, width: u32, kh: u32, kw: u32, oh: u32, ow: u32, sh: u32, sw: u32, ph: u32, pw: u32,
) {
    let i = indices();
    let spatial = broadcast_scalar(oh * ow, shape![128]);
    let column = i / spatial;
    let y = (i % spatial / broadcast_scalar(ow, shape![128])) * broadcast_scalar(sh, shape![128])
        + column / broadcast_scalar(kw, shape![128]) % broadcast_scalar(kh, shape![128]);
    let x = (i % broadcast_scalar(ow, shape![128])) * broadcast_scalar(sw, shape![128]) + column % broadcast_scalar(kw, shape![128]);
    let mask = lt_tile(i, broadcast_scalar(channels * kh * kw * oh * ow, shape![128]));
    let valid = mask
        & ge_tile(y, broadcast_scalar(ph, shape![128]))
        & ge_tile(x, broadcast_scalar(pw, shape![128]))
        & lt_tile(y, broadcast_scalar(ph + height, shape![128]))
        & lt_tile(x, broadcast_scalar(pw + width, shape![128]));
    let address = (column / broadcast_scalar(kh * kw, shape![128]) * broadcast_scalar(height, shape![128]) + y - broadcast_scalar(ph, shape![128]))
        * broadcast_scalar(width, shape![128])
        + x
        - broadcast_scalar(pw, shape![128]);
    let value = unsafe { read_tile(input, address, valid, 0f32) };
    unsafe { write_tile(output, i, value, mask) };
}
#[cutile::entry()]
unsafe fn max_pool(input: *const f32, output: *mut f32, planes: u32, height: u32, width: u32, kh: u32, kw: u32, sh: u32, sw: u32, oh: u32, ow: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(planes * oh * ow, shape![128]));
    let plane = i / broadcast_scalar(oh * ow, shape![128]);
    let y = i / broadcast_scalar(ow, shape![128]) % broadcast_scalar(oh, shape![128]) * broadcast_scalar(sh, shape![128]);
    let x = i % broadcast_scalar(ow, shape![128]) * broadcast_scalar(sw, shape![128]);
    let mut result: Tile<f32, { [128] }> = broadcast_scalar(bits_f32(unsigned(-8388608i32)), shape![128]);
    for ky in 0u32..kh {
        for kx in 0u32..kw {
            let index = (plane * broadcast_scalar(height, shape![128]) + y + broadcast_scalar(ky, shape![128])) * broadcast_scalar(width, shape![128])
                + x
                + broadcast_scalar(kx, shape![128]);
            let value = unsafe { read_tile(input, index, mask, bits_f32(unsigned(-8388608i32))) };
            result = select(ne_tile(value, value) | gt_tile(value, result), value, result);
        }
    }
    unsafe { write_tile(output, i, result, mask) };
}

#[cutile::entry()]
unsafe fn residual_norm(input: *mut f32, residual: *const f32, weights: *const f32, output: *mut f32, _rows: u32, width: u32, epsilon: u32) {
    let col = lanes::<1024>();
    let mask = lt_tile(col, broadcast_scalar(width, shape![1024]));
    let index = col + broadcast_scalar(block_id() * width, shape![1024]);
    let pointer: PointerTile<*mut f32, { [] }> = pointer_to_tile(input);
    let pointer: PointerTile<*const f32, { [] }> = cast_tile_const(pointer);
    let source: *const f32 = tile_to_pointer(pointer);
    let a = unsafe { read_tile(source, index, mask, 0f32) };
    let b = unsafe { read_tile(residual, index, mask, 0f32) };
    let sum = a + b;
    let square: Tile<f32, { [] }> = reduce_sum(sum * sum, 0i32);
    let square: f32 = tile_to_scalar(square);
    let factor = 1f32 / root(square / float_u32(width) + bits_f32(epsilon));
    let w = unsafe { read_tile(weights, col, mask, 0f32) };
    let normalized = sum * broadcast_scalar(factor, shape![1024]) * (w + broadcast_scalar(1f32, shape![1024]));
    unsafe {
        write_tile(input, index, sum, mask);
        write_tile(output, index, normalized, mask);
    }
}
#[cutile::entry()]
unsafe fn rms_norm(input: *const f32, weights: *const f32, output: *mut f32, _rows: u32, width: u32, epsilon: u32, offset: u32) {
    let row: u32 = block_id();
    let lane = lanes::<128>();
    let mut square: Tile<f32, { [128] }> = broadcast_scalar(0f32, shape![128]);
    let mut start = 0u32;
    while start < width {
        let col = lane + broadcast_scalar(start, shape![128]);
        let mask = lt_tile(col, broadcast_scalar(width, shape![128]));
        let x = unsafe { read_tile(input, col + broadcast_scalar(row * width, shape![128]), mask, 0f32) };
        square = square + x * x;

        start = start + 128u32;
    }
    let sum: Tile<f32, { [] }> = reduce_sum(square, 0i32);
    let sum: f32 = tile_to_scalar(sum);
    let factor = 1f32 / root(sum / (float_u32(width)) + bits_f32(epsilon));
    let mut start = 0u32;
    while start < width {
        let col = lane + broadcast_scalar(start, shape![128]);
        let mask = lt_tile(col, broadcast_scalar(width, shape![128]));
        let index = col + broadcast_scalar(row * width, shape![128]);
        let x = unsafe { read_tile(input, index, mask, 0f32) };
        let w = unsafe { read_tile(weights, col, mask, 0f32) };
        let value = x * broadcast_scalar(factor, shape![128]) * (w + broadcast_scalar(bits_f32(offset), shape![128]));
        unsafe { write_tile(output, index, value, mask) };

        start = start + 128u32;
    }
}
#[cutile::entry()]
unsafe fn inference_softmax(input: *const f32, output: *mut f32, _rows: u32, cols: u32) {
    let row: u32 = block_id();
    let lane = lanes::<128>();
    let mut largest: Tile<f32, { [128] }> = broadcast_scalar(bits_f32(unsigned(-8388608i32)), shape![128]);
    let mut bad: Tile<u32, { [128] }> = broadcast_scalar(0u32, shape![128]);
    let mut start = 0u32;
    while start < cols {
        let col = lane + broadcast_scalar(start, shape![128]);
        let mask = lt_tile(col, broadcast_scalar(cols, shape![128]));
        let x = unsafe { read_tile(input, col + broadcast_scalar(row * cols, shape![128]), mask, 0f32) };
        let finite = le_tile(absf(x), broadcast_scalar(3.4028235e38f32, shape![128]));
        bad = bad
            | select(
                mask & eq_tile(finite, broadcast_scalar(false, shape![128])),
                broadcast_scalar(1u32, shape![128]),
                broadcast_scalar(0u32, shape![128]),
            );
        largest = max_tile(largest, select(mask, x, broadcast_scalar(bits_f32(unsigned(-8388608i32)), shape![128])));

        start = start + 128u32;
    }
    let largest: Tile<f32, { [] }> = reduce_max(largest, 0i32);
    let bad: Tile<u32, { [] }> = reduce_max(bad, 0i32);
    let maximum: f32 = tile_to_scalar(largest);
    let invalid: u32 = tile_to_scalar(bad);
    let maximum = if invalid == 0u32 { maximum } else { bits_f32(0x7fc00000u32) };
    let mut mass: Tile<f32, { [128] }> = broadcast_scalar(0f32, shape![128]);
    let mut start = 0u32;
    while start < cols {
        let col = lane + broadcast_scalar(start, shape![128]);
        let mask = lt_tile(col, broadcast_scalar(cols, shape![128]));
        let x = unsafe { read_tile(input, col + broadcast_scalar(row * cols, shape![128]), mask, bits_f32(unsigned(-8388608i32))) };
        mass = mass + exp(x - broadcast_scalar(maximum, shape![128]));

        start = start + 128u32;
    }
    let mass: Tile<f32, { [] }> = reduce_sum(mass, 0i32);
    let total: f32 = tile_to_scalar(mass);
    let mut start = 0u32;
    while start < cols {
        let col = lane + broadcast_scalar(start, shape![128]);
        let mask = lt_tile(col, broadcast_scalar(cols, shape![128]));
        let index = col + broadcast_scalar(row * cols, shape![128]);
        let x = unsafe { read_tile(input, index, mask, 0f32) };
        let value = exp(x - broadcast_scalar(maximum, shape![128])) / broadcast_scalar(total, shape![128]);
        unsafe { write_tile(output, index, value, mask) };

        start = start + 128u32;
    }
}
#[cutile::entry()]
unsafe fn argmax_partials(input: *const f32, output: *mut u32, count: u32) {
    let part: u32 = block_id();
    let lane = lanes::<128>();
    let mut best: Tile<f32, { [128] }> = broadcast_scalar(bits_f32(unsigned(-8388608i32)), shape![128]);
    let mut winner: Tile<u32, { [128] }> = broadcast_scalar(unsigned(-1i32), shape![128]);
    let mut bad: Tile<u32, { [128] }> = broadcast_scalar(0u32, shape![128]);
    let mut start = part * 128u32;
    while start < count {
        let index = lane + broadcast_scalar(start, shape![128]);
        let mask = lt_tile(index, broadcast_scalar(count, shape![128]));
        let value = unsafe { read_tile(input, index, mask, 0f32) };
        let finite = le_tile(absf(value), broadcast_scalar(3.4028235e38f32, shape![128]));
        bad = bad
            | select(
                mask & eq_tile(finite, broadcast_scalar(false, shape![128])),
                broadcast_scalar(1u32, shape![128]),
                broadcast_scalar(0u32, shape![128]),
            );
        let take = mask & finite & (gt_tile(value, best) | (eq_tile(value, best) & lt_tile(index, winner)));
        best = select(take, value, best);
        winner = select(take, index, winner);

        start = if count - start <= 4096u32 { count } else { start + 4096u32 };
    }
    let maximum: Tile<f32, { [] }> = reduce_max(best, 0i32);
    let matching = eq_tile(best, broadcast_scalar(tile_to_scalar::<f32, f32>(maximum), shape![128]));
    let winners = select(matching, winner, broadcast_scalar(unsigned(-1i32), shape![128]));
    let winner: Tile<u32, { [] }> = reduce_min(winners, 0i32);
    let bad: Tile<u32, { [] }> = reduce_max(bad, 0i32);
    let invalid: u32 = tile_to_scalar(bad);
    let index: u32 = tile_to_scalar(winner);
    let address: Tile<u32, { [1] }> = broadcast_scalar(index, shape![1]);
    let valid: Tile<bool, { [1] }> = broadcast_scalar(index < count, shape![1]);
    let actual = unsafe { read_tile(input, address, valid, bits_f32(unsigned(-8388608i32))) };
    let bits: Tile<u32, { [1] }> = bitcast(actual);
    let bits: Tile<u32, { [] }> = bits.reshape(shape![]);
    let bits: u32 = tile_to_scalar(bits);
    let result_index = if invalid == 0u32 { index } else { 0u32 };
    let result_value = if invalid == 0u32 { bits } else { 0x7fc00000u32 };
    let index: Tile<u32, { [1] }> = broadcast_scalar(part * 2u32, shape![1]);
    let valid: Tile<bool, { [1] }> = broadcast_scalar(true, shape![1]);
    let value: Tile<u32, { [1] }> = broadcast_scalar(result_index, shape![1]);
    unsafe { write_tile(output, index, value, valid) };
    let index = index + broadcast_scalar(1u32, shape![1]);
    let value: Tile<u32, { [1] }> = broadcast_scalar(result_value, shape![1]);
    unsafe { write_tile(output, index, value, valid) };
}
