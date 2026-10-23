// Cooperative normalization, rotary embedding, and causal attention.
unsafe fn prepare_attention(
    input: *const f32, weights: *const f32, output: *mut f32, heads: u32, dim: u32, rotary: u32, position: u32, theta: u32, epsilon: u32, stride: u32,
) {
    let head = block_id();
    let lane = lanes::<128>();
    let base = head * dim * stride;
    let dest = head * dim;
    let mut square: Tile<f32, { [128] }> = broadcast_scalar(0f32, shape![128]);
    let mut start = 0u32;
    while start < dim {
        let col = lane + broadcast_scalar(start, shape![128]);
        let mask = lt_tile(col, broadcast_scalar(dim, shape![128]));
        let x = unsafe { read_tile(input, col + broadcast_scalar(base, shape![128]), mask, 0f32) };
        square = square + x * x;
        start = start + 128u32;
    }
    let sum: Tile<f32, { [] }> = reduce_sum(square, 0i32);
    let sum: f32 = tile_to_scalar(sum);
    let scale = 1f32 / root(sum / float_u32(dim) + bits_f32(epsilon));
    let half = rotary / 2u32;
    let theta: Tile<f32, { [] }> = scalar_to_tile(bits_f32(theta));
    let logarithm = log(theta);
    let logarithm: f32 = tile_to_scalar(logarithm);
    let mut start = 0u32;
    while start < dim {
        let col = lane + broadcast_scalar(start, shape![128]);
        let mask = lt_tile(col, broadcast_scalar(dim, shape![128]));
        let first = lt_tile(col, broadcast_scalar(half, shape![128]));
        let rotated = lt_tile(col, broadcast_scalar(rotary, shape![128]));
        let partner = select(first, col + broadcast_scalar(half, shape![128]), col - broadcast_scalar(half, shape![128]));
        let a = unsafe { read_tile(input, col + broadcast_scalar(base, shape![128]), mask, 0f32) };
        let aw = unsafe { read_tile(weights, col, mask, 0f32) };
        let b = unsafe { read_tile(input, partner + broadcast_scalar(base, shape![128]), mask & rotated, 0f32) };
        let bw = unsafe { read_tile(weights, partner, mask & rotated, 0f32) };
        let a = a * broadcast_scalar(scale, shape![128]) * (aw + broadcast_scalar(1f32, shape![128]));
        let b = b * broadcast_scalar(scale, shape![128]) * (bw + broadcast_scalar(1f32, shape![128]));
        let frequency: Tile<f32, { [128] }> = itof(col % broadcast_scalar(half, shape![128]), rounding::NearestEven);
        let angle = broadcast_scalar(float_u32(position + head / heads), shape![128])
            / exp(frequency * broadcast_scalar(2f32 * logarithm / float_u32(rotary), shape![128]));
        let angle = angle - floor(angle / broadcast_scalar(6.2831855f32, shape![128])) * broadcast_scalar(6.2831855f32, shape![128]);
        let cosine = cos(angle);
        let sine = sin(angle);
        let value = select(rotated, select(first, a * cosine - b * sine, a * cosine + b * sine), a);
        unsafe { write_tile(output, col + broadcast_scalar(dest, shape![128]), value, mask) };
        start = start + 128u32;
    }
}
#[cutile::entry()]
unsafe fn attention_prepare(
    input: *const f32, weights: *const f32, output: *mut f32, heads: u32, dim: u32, rotary: u32, position: u32, theta: u32, epsilon: u32, stride: u32,
) {
    unsafe { prepare_attention(input, weights, output, heads, dim, rotary, position, theta, epsilon, stride) }
}
#[cutile::entry()]
unsafe fn attention_prepare_batch(
    input: *const f32, weights: *const f32, output: *mut f32, heads: u32, dim: u32, rotary: u32, position: u32, theta: u32, epsilon: u32, stride: u32,
    _rows: u32,
) {
    unsafe { prepare_attention(input, weights, output, heads, dim, rotary, position, theta, epsilon, stride) }
}
#[cutile::entry()]
unsafe fn attention_scores(query: *const f32, keys: *const f32, scores: *mut f32, heads: u32, kv_heads: u32, dim: u32, positions: u32) {
    let blocks = (positions + 31u32) / 32u32;
    let head = block_id() / blocks;
    let kv = head / (heads / kv_heads);
    let p = lanes::<32>() + broadcast_scalar(block_id() % blocks * 32u32, shape![32]);
    let p: Tile<u32, { [32, 1] }> = p.reshape(shape![32, 1]);
    let p: Tile<u32, { [32, 128] }> = p.broadcast(shape![32, 128]);
    let d = lanes::<128>();
    let d: Tile<u32, { [1, 128] }> = d.reshape(shape![1, 128]);
    let d: Tile<u32, { [32, 128] }> = d.broadcast(shape![32, 128]);
    let mut sum: Tile<f32, { [32, 128] }> = broadcast_scalar(0f32, shape![32, 128]);
    let mut start = 0u32;
    while start < dim {
        let col = d + broadcast_scalar(start, shape![32, 128]);
        let mask = lt_tile(col, broadcast_scalar(dim, shape![32, 128])) & lt_tile(p, broadcast_scalar(positions, shape![32, 128]));
        let q = unsafe { read_matrix(query, col + broadcast_scalar(head * dim, shape![32, 128]), mask, 0f32, shape![4096]) };
        let k = unsafe {
            read_matrix(
                keys,
                (p * broadcast_scalar(kv_heads, shape![32, 128]) + broadcast_scalar(kv, shape![32, 128])) * broadcast_scalar(dim, shape![32, 128]) + col,
                mask,
                0f32,
                shape![4096],
            )
        };
        sum = sum + q * k;
        start = start + 128u32;
    }
    let sum: Tile<f32, { [32] }> = reduce_sum(sum, 1i32);
    let value = sum / broadcast_scalar(root(float_u32(dim)), shape![32]);
    let p = lanes::<32>() + broadcast_scalar(block_id() % blocks * 32u32, shape![32]);
    let mask = lt_tile(p, broadcast_scalar(positions, shape![32]));
    unsafe { write_tile(scores, p + broadcast_scalar(head * positions, shape![32]), value, mask) };
}
#[cutile::entry()]
unsafe fn attention_output(
    scores: *const f32, values: *const f32, query_gate: *const f32, output: *mut f32, heads: u32, kv_heads: u32, dim: u32, positions: u32,
) {
    let blocks = (dim + 127u32) / 128u32;
    let head = block_id() / blocks;
    let kv = head / (heads / kv_heads);
    let col = lanes::<128>() + broadcast_scalar(block_id() % blocks * 128u32, shape![128]);
    let mask = lt_tile(col, broadcast_scalar(dim, shape![128]));
    let mut sum: Tile<f32, { [128] }> = broadcast_scalar(0f32, shape![128]);
    for position in 0u32..positions {
        let score = unsafe { read_tile(scores, broadcast_scalar(head * positions + position, shape![128]), mask, 0f32) };
        let value = unsafe { read_tile(values, col + broadcast_scalar((position * kv_heads + kv) * dim, shape![128]), mask, 0f32) };
        sum = sum + score * value;
    }
    let gate = unsafe { read_tile(query_gate, col + broadcast_scalar(head * 2u32 * dim + dim, shape![128]), mask, 0f32) };
    unsafe { write_tile(output, col + broadcast_scalar(head * dim, shape![128]), sum * sigmoid_tile(gate), mask) };
}
unsafe fn attention_sequence<const SPLIT: bool>(
    query: *const f32, keys: *const f32, values: *const f32, query_gate: *const f32, output: *mut f32, heads: u32, kv_heads: u32, dim: u32,
    position: u32, splits: u32,
) {
    let head = if SPLIT { block_id() / splits } else { block_id() };
    let kv = head % heads / (heads / kv_heads);
    let visible = position + head / heads + 1u32;
    let d = lanes::<256>();
    let dm = lt_tile(d, broadcast_scalar(dim, shape![256]));
    let q = unsafe { read_tile(query, d + broadcast_scalar(head * dim, shape![256]), dm, 0f32) };
    let q: Tile<f32, { [1, 256] }> = q.reshape(shape![1, 256]);
    let q: Tile<f32, { [32, 256] }> = q.broadcast(shape![32, 256]);
    let dd: Tile<u32, { [1, 256] }> = d.reshape(shape![1, 256]);
    let dd: Tile<u32, { [32, 256] }> = dd.broadcast(shape![32, 256]);
    let mut maximum = bits_f32(unsigned(-8388608i32));
    let mut mass = 0f32;
    let mut accumulated: Tile<f32, { [256] }> = broadcast_scalar(0f32, shape![256]);
    let chunk = if SPLIT { (visible + splits * 32u32 - 1u32) / (splits * 32u32) * 32u32 } else { 0u32 };
    let mut start = if SPLIT { block_id() % splits * chunk } else { 0u32 };
    let end = if SPLIT { min(start + chunk, visible) } else { visible };
    while start < end {
        let p = lanes::<32>() + broadcast_scalar(start, shape![32]);
        let pm = lt_tile(p, broadcast_scalar(visible, shape![32]));
        let pp: Tile<u32, { [32, 1] }> = p.reshape(shape![32, 1]);
        let pp: Tile<u32, { [32, 256] }> = pp.broadcast(shape![32, 256]);
        let mask = lt_tile(pp, broadcast_scalar(visible, shape![32, 256])) & lt_tile(dd, broadcast_scalar(dim, shape![32, 256]));
        let index = (pp * broadcast_scalar(kv_heads, shape![32, 256]) + broadcast_scalar(kv, shape![32, 256])) * broadcast_scalar(dim, shape![32, 256]) + dd;
        let k = unsafe { read_matrix(keys, index, mask, 0f32, shape![8192]) };
        let product = k * q;
        let scores: Tile<f32, { [32] }> = reduce_sum(product, 1i32);
        let scores = scores / broadcast_scalar(root(float_u32(dim)), shape![32]);
        let scores = select(pm, scores, broadcast_scalar(bits_f32(unsigned(-8388608i32)), shape![32]));
        let tilemax: Tile<f32, { [] }> = reduce_max(scores, 0i32);
        let tilemax: f32 = tile_to_scalar(tilemax);
        let next = max(maximum, tilemax);
        let previous: Tile<f32, { [] }> = scalar_to_tile(maximum - next);
        let rescale = exp(previous);
        let rescale: f32 = tile_to_scalar(rescale);
        let weights = exp(scores - broadcast_scalar(next, shape![32]));
        let tilemass: Tile<f32, { [] }> = reduce_sum(weights, 0i32);
        let tilemass: f32 = tile_to_scalar(tilemass);
        let old = mass * rescale;
        mass = old + tilemass;
        maximum = next;
        let weights = weights / broadcast_scalar(mass, shape![32]);
        let weights: Tile<f32, { [32, 1] }> = weights.reshape(shape![32, 1]);
        let weights: Tile<f32, { [32, 256] }> = weights.broadcast(shape![32, 256]);
        let v = unsafe { read_matrix(values, index, mask, 0f32, shape![8192]) };
        let weighted = v * weights;
        let total: Tile<f32, { [256] }> = reduce_sum(weighted, 0i32);
        accumulated = accumulated * broadcast_scalar(old / mass, shape![256]) + total;
        start = start + 32u32;
    }
    if SPLIT {
        let base = block_id() * (dim + 2u32);
        unsafe { write_tile(output, d + broadcast_scalar(base, shape![256]), accumulated, dm) };
        unsafe { write_scalar(output, base + dim, maximum) };
        unsafe { write_scalar(output, base + dim + 1u32, mass) };
    } else {
        let gate = unsafe { read_tile(query_gate, d + broadcast_scalar(head * 2u32 * dim + dim, shape![256]), dm, 0f32) };
        unsafe { write_tile(output, d + broadcast_scalar(head * dim, shape![256]), accumulated * sigmoid_tile(gate), dm) };
    }
}

#[cutile::entry()]
unsafe fn attention_fused_batch(
    query: *const f32, keys: *const f32, values: *const f32, query_gate: *const f32, output: *mut f32, _rows: u32, heads: u32, kv_heads: u32, dim: u32,
    position: u32,
) {
    unsafe { attention_sequence::<false>(query, keys, values, query_gate, output, heads, kv_heads, dim, position, 1u32) }
}

#[cutile::entry()]
unsafe fn attention_split_batch(
    query: *const f32, keys: *const f32, values: *const f32, query_gate: *const f32, output: *mut f32, _rows: u32, heads: u32, kv_heads: u32, dim: u32,
    position: u32, splits: u32,
) {
    unsafe { attention_sequence::<true>(query, keys, values, query_gate, output, heads, kv_heads, dim, position, splits) }
}

#[cutile::entry()]
unsafe fn attention_merge_splits(partials: *const f32, query_gate: *const f32, output: *mut f32, _heads: u32, dim: u32, splits: u32) {
    let head = block_id();
    let p = lanes::<32>();
    let pm = lt_tile(p, broadcast_scalar(splits, shape![32]));
    let index = (p + broadcast_scalar(head * splits, shape![32])) * broadcast_scalar(dim + 2u32, shape![32]);
    let maxima = unsafe { read_tile(partials, index + broadcast_scalar(dim, shape![32]), pm, bits_f32(unsigned(-8388608i32))) };
    let masses = unsafe { read_tile(partials, index + broadcast_scalar(dim + 1u32, shape![32]), pm, 0f32) };
    let maximum: Tile<f32, { [] }> = reduce_max(maxima, 0i32);
    let maximum: f32 = tile_to_scalar(maximum);
    let weights = masses * exp(maxima - broadcast_scalar(maximum, shape![32]));
    let mass: Tile<f32, { [] }> = reduce_sum(weights, 0i32);
    let mass: f32 = tile_to_scalar(mass);
    let weights = weights / broadcast_scalar(mass, shape![32]);
    let weights: Tile<f32, { [32, 1] }> = weights.reshape(shape![32, 1]);
    let weights: Tile<f32, { [32, 256] }> = weights.broadcast(shape![32, 256]);
    let index: Tile<u32, { [32, 1] }> = index.reshape(shape![32, 1]);
    let index: Tile<u32, { [32, 256] }> = index.broadcast(shape![32, 256]);
    let pm: Tile<bool, { [32, 1] }> = pm.reshape(shape![32, 1]);
    let pm: Tile<bool, { [32, 256] }> = pm.broadcast(shape![32, 256]);
    let d = lanes::<256>();
    let dm = lt_tile(d, broadcast_scalar(dim, shape![256]));
    let dd: Tile<u32, { [1, 256] }> = d.reshape(shape![1, 256]);
    let dd: Tile<u32, { [32, 256] }> = dd.broadcast(shape![32, 256]);
    let mask = pm & lt_tile(dd, broadcast_scalar(dim, shape![32, 256]));
    let values = unsafe { read_matrix(partials, index + dd, mask, 0f32, shape![8192]) };
    let sum: Tile<f32, { [256] }> = reduce_sum(values * weights, 0i32);
    let gate = unsafe { read_tile(query_gate, d + broadcast_scalar(head * 2u32 * dim + dim, shape![256]), dm, 0f32) };
    unsafe { write_tile(output, d + broadcast_scalar(head * dim, shape![256]), sum * sigmoid_tile(gate), dm) };
}
