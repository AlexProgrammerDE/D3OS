// Keep convolution history separate from the recurrent matrix so neither tile
// competes with the other's live values for the SM register file.
#[cutile::entry()]
unsafe fn delta_convolution_batch(input: *const f32, weights: *const f32, history: *mut f32, mixed: *mut f32, channels: u32, kernel: u32, rows: u32) {
    let c = indices();
    let cm = lt_tile(c, broadcast_scalar(channels, shape![128]));
    let cc: Tile<u32, { [128, 1] }> = c.reshape(shape![128, 1]);
    let cc: Tile<u32, { [128, 8] }> = cc.broadcast(shape![128, 8]);
    let h = lanes::<8>();
    let h: Tile<u32, { [1, 8] }> = h.reshape(shape![1, 8]);
    let h: Tile<u32, { [128, 8] }> = h.broadcast(shape![128, 8]);
    let mask = lt_tile(cc, broadcast_scalar(channels, shape![128, 8])) & lt_tile(h, broadcast_scalar(kernel, shape![128, 8]));
    let index = cc * broadcast_scalar(kernel, shape![128, 8]) + h;
    let mut past = unsafe { read_matrix(history, index, mask, 0f32, shape![1024]) };
    let weights = unsafe { read_matrix(weights, index, mask, 0f32, shape![1024]) };
    for row in 0u32..rows {
        let incoming = unsafe { read_tile(input, c + broadcast_scalar(row * channels, shape![128]), cm, 0f32) };
        let mut next: Tile<f32, { [128, 8] }> = broadcast_scalar(0f32, shape![128, 8]);
        for i in 0u32..kernel {
            let value: Tile<f32, { [128, 1] }> = if i + 1u32 < kernel {
                extract(past, [scalar_to_tile(0i32), scalar_to_tile(signed(i + 1u32))])
            } else {
                incoming.reshape(shape![128, 1])
            };
            next = unsafe { insert(value, next, [scalar_to_tile(0i32), scalar_to_tile(signed(i))]) };
        }
        past = next;
        let sum: Tile<f32, { [128] }> = reduce_sum(past * weights, 1i32);
        unsafe { write_tile(mixed, c + broadcast_scalar(row * channels, shape![128]), silu_tile(sum), cm) };
    }
    unsafe { write_matrix(history, index, past, mask, shape![1024]) };
}

#[cutile::entry()]
unsafe fn delta_recurrent_batch(
    mixed: *const f32, beta: *const f32, alpha: *const f32, log_decay: *const f32, time_bias: *const f32, state: *mut f32, output: *mut f32, heads: u32,
    key_dim: u32, value_dim: u32, rows: u32,
) {
    let groups = (value_dim + 31u32) / 32u32;
    let head = block_id() / groups;
    let k = lanes::<128>();
    let v = lanes::<32>() + broadcast_scalar(block_id() % groups * 32u32, shape![32]);
    let km = lt_tile(k, broadcast_scalar(key_dim, shape![128]));
    let vm = lt_tile(v, broadcast_scalar(value_dim, shape![32]));
    let kk: Tile<u32, { [128, 1] }> = k.reshape(shape![128, 1]);
    let kk: Tile<u32, { [128, 32] }> = kk.broadcast(shape![128, 32]);
    let vv: Tile<u32, { [1, 32] }> = v.reshape(shape![1, 32]);
    let vv: Tile<u32, { [128, 32] }> = vv.broadcast(shape![128, 32]);
    let mask = lt_tile(kk, broadcast_scalar(key_dim, shape![128, 32])) & lt_tile(vv, broadcast_scalar(value_dim, shape![128, 32]));
    let index = (kk + broadcast_scalar(head * key_dim, shape![128, 32])) * broadcast_scalar(value_dim, shape![128, 32]) + vv;
    let mut resident = unsafe { read_matrix(state, index, mask, 0f32, shape![4096]) };
    let alog = unsafe { read_scalar(log_decay, head) };
    let dt = unsafe { read_scalar(time_bias, head) };
    for row in 0u32..rows {
        let base = row * heads * (2u32 * key_dim + value_dim);
        let q = unsafe { read_tile(mixed, k + broadcast_scalar(base + head * key_dim, shape![128]), km, 0f32) };
        let key = unsafe { read_tile(mixed, k + broadcast_scalar(base + (heads + head) * key_dim, shape![128]), km, 0f32) };
        let value = unsafe {
            read_tile(
                mixed,
                v + broadcast_scalar(base + 2u32 * heads * key_dim + head * value_dim, shape![32]),
                vm,
                0f32,
            )
        };
        let qs: Tile<f32, { [] }> = reduce_sum(q * q, 0i32);
        let ks: Tile<f32, { [] }> = reduce_sum(key * key, 0i32);
        let qs: f32 = tile_to_scalar(qs);
        let ks: f32 = tile_to_scalar(ks);
        let q = q * broadcast_scalar(1f32 / root(qs + 1e-6f32) / root(float_u32(key_dim)), shape![128]);
        let key = key * broadcast_scalar(1f32 / root(ks + 1e-6f32), shape![128]);
        let a = unsafe { read_scalar(alpha, row * heads + head) };
        let b = unsafe { read_scalar(beta, row * heads + head) };
        let decay = exponential(0f32 - exponential(alog) * softplus(a + dt));
        resident = resident * broadcast_scalar(decay, shape![128, 32]);
        let key: Tile<f32, { [128, 1] }> = key.reshape(shape![128, 1]);
        let key: Tile<f32, { [128, 32] }> = key.broadcast(shape![128, 32]);
        let prediction: Tile<f32, { [32] }> = reduce_sum(resident * key, 0i32);
        let delta = (value - prediction) * broadcast_scalar(sigmoid_scalar(b), shape![32]);
        let delta: Tile<f32, { [1, 32] }> = delta.reshape(shape![1, 32]);
        let delta: Tile<f32, { [128, 32] }> = delta.broadcast(shape![128, 32]);
        resident = resident + key * delta;
        let q: Tile<f32, { [128, 1] }> = q.reshape(shape![128, 1]);
        let q: Tile<f32, { [128, 32] }> = q.broadcast(shape![128, 32]);
        let raw: Tile<f32, { [32] }> = reduce_sum(resident * q, 0i32);
        unsafe { write_tile(output, v + broadcast_scalar((row * heads + head) * value_dim, shape![32]), raw, vm) };
    }
    unsafe { write_matrix(state, index, resident, mask, shape![4096]) };
}
