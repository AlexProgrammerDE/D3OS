// Recurrent delta operators. The fused path keeps state and history in tiles.
#[cutile::entry()]
unsafe fn delta_convolution(input: *const f32, weights: *const f32, history: *mut f32, mixed: *mut f32, channels: u32, kernel: u32) {
    let c = indices();
    let mask = lt_tile(c, broadcast_scalar(channels, shape![128]));
    let mut sum: Tile<f32, { [128] }> = broadcast_scalar(0f32, shape![128]);
    let mut token = new_token_unordered();
    for i in 0u32..kernel {
        let value: Tile<f32, { [128] }> = if i + 1u32 < kernel {
            let (value, next): (Tile<f32, { [128] }>, Token) = unsafe {
                ordered_read_f32_128(
                    history,
                    c * broadcast_scalar(kernel, shape![128]) + broadcast_scalar(i + 1u32, shape![128]),
                    mask,
                    0f32,
                    token,
                )
            };
            token = next;
            value
        } else {
            unsafe { read_tile(input, c, mask, 0f32) }
        };
        let weight = unsafe {
            read_tile(
                weights,
                c * broadcast_scalar(kernel, shape![128]) + broadcast_scalar(i, shape![128]),
                mask,
                0f32,
            )
        };
        sum = sum + value * weight;
        token = unsafe {
            ordered_write(
                history,
                c * broadcast_scalar(kernel, shape![128]) + broadcast_scalar(i, shape![128]),
                value,
                mask,
                token,
            )
        };
    }
    unsafe { write_tile(mixed, c, silu_tile(sum), mask) };
}
#[cutile::entry()]
unsafe fn delta_prepare(
    mixed: *const f32, beta: *const f32, alpha: *const f32, log_decay: *const f32, time_bias: *const f32, scales: *mut f32, heads: u32, key_dim: u32,
    _value_dim: u32,
) {
    let head = block_id();
    let lane = lanes::<128>();
    let mut qs: Tile<f32, { [128] }> = broadcast_scalar(0f32, shape![128]);
    let mut ks = qs;
    let mut start = 0u32;
    while start < key_dim {
        let col = lane + broadcast_scalar(start, shape![128]);
        let mask = lt_tile(col, broadcast_scalar(key_dim, shape![128]));
        let q = unsafe { read_tile(mixed, col + broadcast_scalar(head * key_dim, shape![128]), mask, 0f32) };
        let k = unsafe { read_tile(mixed, col + broadcast_scalar((heads + head) * key_dim, shape![128]), mask, 0f32) };
        qs = qs + q * q;
        ks = ks + k * k;
        start = start + 128u32;
    }
    let qs: Tile<f32, { [] }> = reduce_sum(qs, 0i32);
    let ks: Tile<f32, { [] }> = reduce_sum(ks, 0i32);
    let qs: f32 = tile_to_scalar(qs);
    let ks: f32 = tile_to_scalar(ks);
    let a = unsafe { read_scalar(alpha, head) + read_scalar(time_bias, head) };
    let b = unsafe { read_scalar(beta, head) };
    let decay = unsafe { read_scalar(log_decay, head) };
    unsafe { write_scalar(scales, head * 4u32, 1f32 / root(qs + 1e-6f32) / root(float_u32(key_dim))) };
    unsafe { write_scalar(scales, head * 4u32 + 1u32, 1f32 / root(ks + 1e-6f32)) };
    unsafe { write_scalar(scales, head * 4u32 + 2u32, sigmoid_scalar(b)) };
    unsafe { write_scalar(scales, head * 4u32 + 3u32, exponential(0f32 - exponential(decay) * softplus(a))) };
}
#[cutile::entry()]
unsafe fn delta_update(mixed: *const f32, scales: *const f32, state: *mut f32, output: *mut f32, heads: u32, key_dim: u32, value_dim: u32) {
    let blocks = (value_dim + 127u32) / 128u32;
    let head = block_id() / blocks;
    let col = lanes::<128>() + broadcast_scalar(block_id() % blocks * 128u32, shape![128]);
    let mask = lt_tile(col, broadcast_scalar(value_dim, shape![128]));
    let qs = unsafe { read_scalar(scales, head * 4u32) };
    let ks = unsafe { read_scalar(scales, head * 4u32 + 1u32) };
    let beta = unsafe { read_scalar(scales, head * 4u32 + 2u32) };
    let decay = unsafe { read_scalar(scales, head * 4u32 + 3u32) };
    let mut prediction: Tile<f32, { [128] }> = broadcast_scalar(0f32, shape![128]);
    for k in 0u32..key_dim {
        let old = unsafe { read_tile(state, col + broadcast_scalar((head * key_dim + k) * value_dim, shape![128]), mask, 0f32) };
        let key = unsafe { read_scalar(mixed, (heads + head) * key_dim + k) };
        prediction = prediction + old * broadcast_scalar(decay * (key * ks), shape![128]);
    }
    let v = unsafe {
        read_tile(
            mixed,
            col + broadcast_scalar(2u32 * heads * key_dim + head * value_dim, shape![128]),
            mask,
            0f32,
        )
    };
    let delta = (v - prediction) * broadcast_scalar(beta, shape![128]);
    let mut result: Tile<f32, { [128] }> = broadcast_scalar(0f32, shape![128]);
    for k in 0u32..key_dim {
        let index = col + broadcast_scalar((head * key_dim + k) * value_dim, shape![128]);
        let old = unsafe { read_tile(state, index, mask, 0f32) };
        let key = unsafe { read_scalar(mixed, (heads + head) * key_dim + k) };
        let q = unsafe { read_scalar(mixed, head * key_dim + k) };
        let value = old * broadcast_scalar(decay, shape![128]) + delta * broadcast_scalar(key * ks, shape![128]);
        unsafe { write_tile(state, index, value, mask) };
        result = result + value * broadcast_scalar(q * qs, shape![128]);
    }
    unsafe { write_tile(output, col + broadcast_scalar(head * value_dim, shape![128]), result, mask) };
}
#[cutile::entry()]
unsafe fn delta_output(raw: *const f32, weights: *const f32, gate: *const f32, output: *mut f32, _heads: u32, value_dim: u32, epsilon: u32) {
    let head = block_id();
    let lane = lanes::<128>();
    let mut square: Tile<f32, { [128] }> = broadcast_scalar(0f32, shape![128]);
    let mut start = 0u32;
    while start < value_dim {
        let col = lane + broadcast_scalar(start, shape![128]);
        let mask = lt_tile(col, broadcast_scalar(value_dim, shape![128]));
        let value = unsafe { read_tile(raw, col + broadcast_scalar(head * value_dim, shape![128]), mask, 0f32) };
        square = square + value * value;
        start = start + 128u32;
    }
    let square: Tile<f32, { [] }> = reduce_sum(square, 0i32);
    let square: f32 = tile_to_scalar(square);
    let scale = 1f32 / root(square / float_u32(value_dim) + bits_f32(epsilon));
    let mut start = 0u32;
    while start < value_dim {
        let col = lane + broadcast_scalar(start, shape![128]);
        let mask = lt_tile(col, broadcast_scalar(value_dim, shape![128]));
        let index = col + broadcast_scalar(head * value_dim, shape![128]);
        let value = unsafe { read_tile(raw, index, mask, 0f32) };
        let w = unsafe { read_tile(weights, col, mask, 0f32) };
        let g = unsafe { read_tile(gate, index, mask, 0f32) };
        unsafe { write_tile(output, index, value * broadcast_scalar(scale, shape![128]) * w * silu_tile(g), mask) };
        start = start + 128u32;
    }
}

#[cutile::entry()]
unsafe fn delta_fused(
    qkv: *const f32, gate: *const f32, beta: *const f32, alpha: *const f32, convolution: *const f32, log_decay: *const f32, time_bias: *const f32,
    norm: *const f32, history: *mut f32, state: *mut f32, output: *mut f32, heads: u32, key_dim: u32, value_dim: u32, kernel: u32, epsilon: u32,
) {
    unsafe {
        delta_sequence(
            qkv,
            gate,
            beta,
            alpha,
            convolution,
            log_decay,
            time_bias,
            norm,
            history,
            state,
            output,
            heads,
            key_dim,
            value_dim,
            kernel,
            epsilon,
            1u32,
        )
    }
}
#[cutile::entry()]
unsafe fn delta_fused_batch(
    qkv: *const f32, gate: *const f32, beta: *const f32, alpha: *const f32, convolution: *const f32, log_decay: *const f32, time_bias: *const f32,
    norm: *const f32, history: *mut f32, state: *mut f32, output: *mut f32, heads: u32, key_dim: u32, value_dim: u32, kernel: u32, epsilon: u32, rows: u32,
) {
    unsafe {
        delta_sequence(
            qkv,
            gate,
            beta,
            alpha,
            convolution,
            log_decay,
            time_bias,
            norm,
            history,
            state,
            output,
            heads,
            key_dim,
            value_dim,
            kernel,
            epsilon,
            rows,
        )
    }
}

#[cutile::entry()]
unsafe fn delta_fused_four_taps(
    qkv: *const f32, gate: *const f32, beta: *const f32, alpha: *const f32, convolution: *const f32, log_decay: *const f32, time_bias: *const f32,
    norm: *const f32, history: *mut f32, state: *mut f32, output: *mut f32, heads: u32, key_dim: u32, value_dim: u32, kernel: u32, epsilon: u32,
) {
    unsafe {
        delta_sequence_four_taps(
            qkv,
            gate,
            beta,
            alpha,
            convolution,
            log_decay,
            time_bias,
            norm,
            history,
            state,
            output,
            heads,
            key_dim,
            value_dim,
            kernel,
            epsilon,
            1u32,
        )
    }
}
