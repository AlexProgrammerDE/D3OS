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

unsafe fn delta_sequence(
    qkv: *const f32, gate: *const f32, beta: *const f32, alpha: *const f32, convolution: *const f32, log_decay: *const f32, time_bias: *const f32,
    norm: *const f32, history: *mut f32, state: *mut f32, output: *mut f32, heads: u32, key_dim: u32, value_dim: u32, kernel: u32, epsilon: u32, rows: u32,
) {
    let head = block_id();
    let k = lanes::<128>();
    let v = lanes::<128>();
    let kk: Tile<u32, { [128, 1] }> = k.reshape(shape![128, 1]);
    let kk: Tile<u32, { [128, 128] }> = kk.broadcast(shape![128, 128]);
    let vv: Tile<u32, { [1, 128] }> = v.reshape(shape![1, 128]);
    let vv: Tile<u32, { [128, 128] }> = vv.broadcast(shape![128, 128]);
    let sm = lt_tile(kk, broadcast_scalar(key_dim, shape![128, 128])) & lt_tile(vv, broadcast_scalar(value_dim, shape![128, 128]));
    let si = (kk + broadcast_scalar(head * key_dim, shape![128, 128])) * broadcast_scalar(value_dim, shape![128, 128]) + vv;
    let mut resident = unsafe { read_matrix(state, si, sm, 0f32, shape![16384]) };
    let c = lanes::<512>();
    let qm = lt_tile(c, broadcast_scalar(key_dim, shape![512]));
    let km = ge_tile(c, broadcast_scalar(128u32, shape![512])) & lt_tile(c, broadcast_scalar(128u32 + key_dim, shape![512]));
    let vm = ge_tile(c, broadcast_scalar(256u32, shape![512])) & lt_tile(c, broadcast_scalar(256u32 + value_dim, shape![512]));
    let cm = qm | km | vm;
    let global = select(
        qm,
        c + broadcast_scalar(head * key_dim, shape![512]),
        select(
            km,
            c + broadcast_scalar((heads + head) * key_dim - 128u32, shape![512]),
            c + broadcast_scalar(2u32 * heads * key_dim + head * value_dim - 256u32, shape![512]),
        ),
    );
    let global2: Tile<u32, { [512, 1] }> = global.reshape(shape![512, 1]);
    let global2: Tile<u32, { [512, 8] }> = global2.broadcast(shape![512, 8]);
    let cm2: Tile<bool, { [512, 1] }> = cm.reshape(shape![512, 1]);
    let cm2: Tile<bool, { [512, 8] }> = cm2.broadcast(shape![512, 8]);
    let h = lanes::<8>();
    let h: Tile<u32, { [1, 8] }> = h.reshape(shape![1, 8]);
    let h: Tile<u32, { [512, 8] }> = h.broadcast(shape![512, 8]);
    let hm = cm2 & lt_tile(h, broadcast_scalar(kernel, shape![512, 8]));
    let hi = global2 * broadcast_scalar(kernel, shape![512, 8]) + h;
    let mut past = unsafe { read_matrix(history, hi, hm, 0f32, shape![4096]) };
    let weights = unsafe { read_matrix(convolution, hi, hm, 0f32, shape![4096]) };
    let alog = unsafe { read_scalar(log_decay, head) };
    let dt = unsafe { read_scalar(time_bias, head) };
    let vmask = lt_tile(v, broadcast_scalar(value_dim, shape![128]));
    let norm = unsafe { read_tile(norm, v, vmask, 0f32) };
    for row in 0u32..rows {
        let incoming = unsafe {
            read_tile(
                qkv,
                global + broadcast_scalar(row * heads * (2u32 * key_dim + value_dim), shape![512]),
                cm,
                0f32,
            )
        };
        let mut next: Tile<f32, { [512, 8] }> = broadcast_scalar(0f32, shape![512, 8]);
        for i in 0u32..kernel {
            let value: Tile<f32, { [512, 1] }> = if i + 1u32 < kernel {
                extract(past, [scalar_to_tile(0i32), scalar_to_tile(signed(i + 1u32))])
            } else {
                incoming.reshape(shape![512, 1])
            };
            next = unsafe { insert(value, next, [scalar_to_tile(0i32), scalar_to_tile(signed(i))]) };
        }
        past = next;
        let product = past * weights;
        let mixed: Tile<f32, { [512] }> = reduce_sum(product, 1i32);
        let mixed = silu_tile(mixed);
        let q: Tile<f32, { [128] }> = extract(mixed, [scalar_to_tile(0i32)]);
        let key: Tile<f32, { [128] }> = extract(mixed, [scalar_to_tile(1i32)]);
        let value: Tile<f32, { [128] }> = extract(mixed, [scalar_to_tile(2i32)]);
        let qs: Tile<f32, { [] }> = reduce_sum(q * q, 0i32);
        let ks: Tile<f32, { [] }> = reduce_sum(key * key, 0i32);
        let qs: f32 = tile_to_scalar(qs);
        let ks: f32 = tile_to_scalar(ks);
        let q = q * broadcast_scalar(1f32 / root(qs + 1e-6f32) / root(float_u32(key_dim)), shape![128]);
        let key = key * broadcast_scalar(1f32 / root(ks + 1e-6f32), shape![128]);
        let a = unsafe { read_scalar(alpha, row * heads + head) };
        let b = unsafe { read_scalar(beta, row * heads + head) };
        let decay = exponential(0f32 - exponential(alog) * softplus(a + dt));
        resident = resident * broadcast_scalar(decay, shape![128, 128]);
        let key2: Tile<f32, { [128, 1] }> = key.reshape(shape![128, 1]);
        let key2: Tile<f32, { [128, 128] }> = key2.broadcast(shape![128, 128]);
        let prediction: Tile<f32, { [128] }> = reduce_sum(resident * key2, 0i32);
        let delta = (value - prediction) * broadcast_scalar(sigmoid_scalar(b), shape![128]);
        let delta: Tile<f32, { [1, 128] }> = delta.reshape(shape![1, 128]);
        let delta: Tile<f32, { [128, 128] }> = delta.broadcast(shape![128, 128]);
        resident = resident + key2 * delta;
        let q: Tile<f32, { [128, 1] }> = q.reshape(shape![128, 1]);
        let q: Tile<f32, { [128, 128] }> = q.broadcast(shape![128, 128]);
        let raw: Tile<f32, { [128] }> = reduce_sum(resident * q, 0i32);
        let square: Tile<f32, { [] }> = reduce_sum(raw * raw, 0i32);
        let square: f32 = tile_to_scalar(square);
        let scale = 1f32 / root(square / float_u32(value_dim) + bits_f32(epsilon));
        let gi = v + broadcast_scalar((row * heads + head) * value_dim, shape![128]);
        let g = unsafe { read_tile(gate, gi, vmask, 0f32) };
        unsafe { write_tile(output, gi, raw * broadcast_scalar(scale, shape![128]) * norm * silu_tile(g), vmask) };
    }
    unsafe { write_matrix(history, hi, past, hm, shape![4096]) };
    unsafe { write_matrix(state, si, resident, sm, shape![16384]) };
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
