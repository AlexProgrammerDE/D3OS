//! Qwen transformer operators. Mutable caches never leave GPU memory.
use core::arch::asm;
use core::arch::nvptx::{_block_idx_x, _thread_idx_x};

mod delta;

#[inline(always)]
unsafe fn shared_read(base: u32, index: u32) -> f32 {
    let value;
    unsafe {
        asm!("ld.shared.f32 {value}, [{address}];", value = out(reg32) value, address = in(reg32) (base + index * 4), options(nostack));
    }
    value
}

#[inline(always)]
unsafe fn shared_write(base: u32, index: u32, value: f32) {
    unsafe {
        asm!("st.shared.f32 [{address}], {value};", address = in(reg32) (base + index * 4), value = in(reg32) value, options(nostack));
    }
}

#[inline(always)]
unsafe fn barrier() {
    unsafe {
        asm!("bar.sync 0;", options(nostack));
    }
}

#[inline(always)]
fn exp(value: f32) -> f32 {
    let result;
    unsafe {
        asm!("ex2.approx.f32 {d}, {s};", d = out(reg32) result, s = in(reg32) (value * core::f32::consts::LOG2_E), options(pure, nomem, nostack));
    }
    result
}

#[inline(always)]
fn log(value: f32) -> f32 {
    let result: f32;
    unsafe {
        asm!("lg2.approx.f32 {d}, {s};", d = out(reg32) result, s = in(reg32) value, options(pure, nomem, nostack));
    }
    result * core::f32::consts::LN_2
}

#[inline(always)]
fn log1p(value: f32) -> f32 {
    let sum = 1. + value;
    if sum == 1. { value } else { log(sum) * value / (sum - 1.) }
}

#[inline(always)]
fn sqrt(value: f32) -> f32 {
    let result;
    unsafe {
        asm!("sqrt.rn.f32 {d}, {s};", d = out(reg32) result, s = in(reg32) value, options(pure, nomem, nostack));
    }
    result
}

#[inline(always)]
fn sigmoid(value: f32) -> f32 {
    if value >= 0. {
        1. / (1. + exp(-value))
    } else {
        let e = exp(value);
        e / (1. + e)
    }
}

#[inline(always)]
fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

#[inline(always)]
unsafe fn read(address: u64, index: u32) -> f32 {
    unsafe { (address as *const f32).add(index as usize).read() }
}

#[inline(always)]
unsafe fn write(address: u64, index: u32, value: f32) {
    unsafe { (address as *mut f32).add(index as usize).write(value) }
}

#[inline(always)]
unsafe fn dot(a: u64, a_offset: u32, b: u64, b_offset: u32, width: u32) -> f32 {
    let mut sums = [0.; 8];
    for block in 0..width / 8 {
        for lane in 0..8 {
            let k = block * 8 + lane;
            sums[lane as usize] += unsafe { read(a, a_offset + k) * read(b, b_offset + k) };
        }
    }
    let mut tail = 0.;
    for k in width / 8 * 8..width {
        tail += unsafe { read(a, a_offset + k) * read(b, b_offset + k) };
    }
    sums.into_iter().sum::<f32>() + tail
}

#[inline(always)]
unsafe fn warp_dot(a: u64, a_offset: u32, b: u64, b_offset: u32, width: u32) -> f32 {
    let lane = super::index() % 32;
    let mut sum = 0.;
    let mut k = lane;
    while k < width {
        sum += unsafe { read(a, a_offset + k) * read(b, b_offset + k) };
        k += 32;
    }
    // Broadcast lane zero so every row element uses one reduction order.
    unsafe { super::warp::broadcast(super::warp::sum(sum)) }
}

/// # Safety
/// Input and output contain count elements. Exact aliasing is allowed.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn tensor_copy(input: u64, output: u64, count: u32) {
    let i = super::index();
    if i < count {
        unsafe {
            write(output, i, read(input, i));
        }
    }
}

/// # Safety
/// All buffers contain count elements. Output can exactly alias input only.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn tensor_add(input: u64, other: u64, output: u64, count: u32) {
    let i = super::index();
    if i < count {
        unsafe {
            write(output, i, read(input, i) + read(other, i));
        }
    }
}

/// # Safety
/// Same bounds and alias requirements as tensor_add.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn silu_mul(input: u64, other: u64, output: u64, count: u32) {
    let i = super::index();
    if i < count {
        unsafe {
            write(output, i, silu(read(input, i)) * read(other, i));
        }
    }
}

/// # Safety
/// Input/output contain rows*width elements, weights width elements, with no
/// output overlap. Epsilon is finite and positive, offset is zero or one.
/// Launch 128 threads per block, one block per row.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn rms_norm(input: u64, weights: u64, output: u64, rows: u32, width: u32, epsilon: u32, offset: u32) {
    let (row, thread) = unsafe { (_block_idx_x(), _thread_idx_x()) };
    if row >= rows {
        return;
    }
    let base = row * width;
    let shared: u32;
    unsafe {
        asm!("{{ .shared .align 16 .b8 rms_shared[16]; mov.u32 {base}, rms_shared; }}", base = out(reg32) shared, options(nostack));
    }
    let mut square = 0.;
    let mut i = thread;
    while i < width {
        let value = unsafe { read(input, base + i) };
        square += value * value;
        i += 128;
    }
    square = unsafe { super::warp::sum(square) };
    if thread % 32 == 0 {
        unsafe { shared_write(shared, thread / 32, square) };
    }
    unsafe { barrier() };
    if thread < 32 {
        let partial = if thread < 4 { unsafe { shared_read(shared, thread) } } else { 0. };
        let square = unsafe { super::warp::sum(partial) };
        if thread == 0 {
            unsafe { shared_write(shared, 0, 1. / sqrt(square / width as f32 + f32::from_bits(epsilon))) };
        }
    }
    unsafe { barrier() };
    let scale = unsafe { shared_read(shared, 0) };
    let mut i = thread;
    while i < width {
        unsafe {
            write(output, base + i, read(input, base + i) * scale * (f32::from_bits(offset) + read(weights, i)));
        }
        i += 128;
    }
}

/// # Safety
/// Input has heads*dim*stride elements, weights dim, and disjoint output
/// heads*dim. Stride is 1 (key) or 2 (query/gate). Rotary is even and <= dim.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prepare(
    input: u64, weights: u64, output: u64, heads: u32, dim: u32, rotary: u32, position: u32, theta: u32, epsilon: u32, stride: u32,
) {
    let head = super::index();
    if head >= heads {
        return;
    }
    let base = head * dim * stride;
    let dest = head * dim;
    let scale = 1. / sqrt(unsafe { dot(input, base, input, base, dim) } / dim as f32 + f32::from_bits(epsilon));
    for i in 0..dim {
        unsafe {
            write(output, dest + i, read(input, base + i) * scale * (1. + read(weights, i)));
        }
    }
    for i in 0..rotary / 2 {
        let angle = position as f32 / exp(log(f32::from_bits(theta)) * (2 * i) as f32 / rotary as f32);
        let turns: i32;
        unsafe {
            asm!("cvt.rmi.s32.f32 {d}, {s};", d = out(reg32) turns, s = in(reg32) (angle / core::f32::consts::TAU), options(pure, nomem, nostack));
        }
        let angle = angle - turns as f32 * core::f32::consts::TAU;
        let sin: f32;
        let cos: f32;
        unsafe {
            asm!("sin.approx.f32 {d}, {s};", d = out(reg32) sin, s = in(reg32) angle, options(pure, nomem, nostack));
            asm!("cos.approx.f32 {d}, {s};", d = out(reg32) cos, s = in(reg32) angle, options(pure, nomem, nostack));
            let a = read(output, dest + i);
            let b = read(output, dest + i + rotary / 2);
            write(output, dest + i, a * cos - b * sin);
            write(output, dest + i + rotary / 2, b * cos + a * sin);
        }
    }
}

/// # Safety
/// Query[heads,dim], keys[positions,kv_heads,dim], scores[heads,positions].
/// Buffers are disjoint. Heads is divisible by kv_heads. Products fit u32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_scores(query: u64, keys: u64, scores: u64, heads: u32, kv_heads: u32, dim: u32, positions: u32) {
    let i = super::index();
    if i >= heads * positions {
        return;
    }
    let head = i / positions;
    let position = i % positions;
    let key_base = (position * kv_heads + head / (heads / kv_heads)) * dim;
    unsafe {
        write(scores, i, dot(query, head * dim, keys, key_base, dim) / sqrt(dim as f32));
    }
}

/// # Safety
/// Scores[heads,positions], values[positions,kv_heads,dim], query_gate[heads,2*dim]
/// and disjoint output[heads,dim]. Heads is divisible by kv_heads.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_output(
    scores: u64, values: u64, query_gate: u64, output: u64, heads: u32, kv_heads: u32, dim: u32, positions: u32,
) {
    let i = super::index();
    if i >= heads * dim {
        return;
    }
    let head = i / dim;
    let col = i % dim;
    let kv = head / (heads / kv_heads);
    let mut sum = 0.;
    for position in 0..positions {
        sum += unsafe { read(scores, head * positions + position) * read(values, (position * kv_heads + kv) * dim + col) };
    }
    unsafe {
        write(output, i, sum * sigmoid(read(query_gate, head * 2 * dim + dim + col)));
    }
}

/// # Safety
/// Input[channels], weights/history[channels,kernel], mixed[channels]. All
/// ranges are disjoint. Only history and mixed are modified.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn delta_convolution(input: u64, weights: u64, history: u64, mixed: u64, channels: u32, kernel: u32) {
    let channel = super::index();
    if channel >= channels {
        return;
    }
    let base = channel * kernel;
    for i in 0..kernel - 1 {
        unsafe {
            write(history, base + i, read(history, base + i + 1));
        }
    }
    unsafe {
        write(history, base + kernel - 1, read(input, channel));
        write(mixed, channel, silu(dot(history, base, weights, base, kernel)));
    }
}

/// # Safety
/// Mixed contains heads*(2*key_dim+value_dim), beta/alpha/log_decay/time_bias
/// contain heads, and scales contains heads*4 elements. Output is disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn delta_prepare(
    mixed: u64, beta: u64, alpha: u64, log_decay: u64, time_bias: u64, scales: u64, heads: u32, key_dim: u32, _value_dim: u32,
) {
    let head = super::index();
    if head >= heads {
        return;
    }
    let q = head * key_dim;
    let k = heads * key_dim + q;
    let q_scale = 1. / sqrt(unsafe { dot(mixed, q, mixed, q, key_dim) } + 1e-6) / sqrt(key_dim as f32);
    let k_scale = 1. / sqrt(unsafe { dot(mixed, k, mixed, k, key_dim) } + 1e-6);
    unsafe {
        let alpha = read(alpha, head) + read(time_bias, head);
        let softplus = alpha.max(0.) + log1p(exp(-alpha.abs()));
        write(scales, head * 4, q_scale);
        write(scales, head * 4 + 1, k_scale);
        write(scales, head * 4 + 2, sigmoid(read(beta, head)));
        write(scales, head * 4 + 3, exp(-exp(read(log_decay, head)) * softplus));
    }
}

/// # Safety
/// Mixed[heads*(2*key_dim+value_dim)], scales[heads,4], state[heads,key_dim,value_dim],
/// output[heads,value_dim] are disjoint. Only state and output are modified.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn delta_update(mixed: u64, scales: u64, state: u64, output: u64, heads: u32, key_dim: u32, value_dim: u32) {
    let i = super::index();
    if i >= heads * value_dim {
        return;
    }
    let head = i / value_dim;
    let col = i % value_dim;
    unsafe {
        let q_scale = read(scales, head * 4);
        let k_scale = read(scales, head * 4 + 1);
        let beta = read(scales, head * 4 + 2);
        let decay = read(scales, head * 4 + 3);
        let mut prediction = 0.;
        for k in 0..key_dim {
            let index = (head * key_dim + k) * value_dim + col;
            let value = read(state, index) * decay;
            write(state, index, value);
            prediction += value * (read(mixed, heads * key_dim + head * key_dim + k) * k_scale);
        }
        let delta = (read(mixed, 2 * heads * key_dim + i) - prediction) * beta;
        let mut result = 0.;
        for k in 0..key_dim {
            let index = (head * key_dim + k) * value_dim + col;
            let value = read(state, index) + read(mixed, heads * key_dim + head * key_dim + k) * k_scale * delta;
            write(state, index, value);
            result += value * (read(mixed, head * key_dim + k) * q_scale);
        }
        write(output, i, result);
    }
}

/// # Safety
/// Raw/gate/output[heads,value_dim], weights[value_dim], disjoint output.
/// Epsilon is finite and positive. Launch full warps, one warp per head.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn delta_output(raw: u64, weights: u64, gate: u64, output: u64, heads: u32, value_dim: u32, epsilon: u32) {
    let head = super::index() / 32;
    let lane = super::index() % 32;
    if head >= heads {
        return;
    }
    let base = head * value_dim;
    let scale = 1. / sqrt(unsafe { warp_dot(raw, base, raw, base, value_dim) } / value_dim as f32 + f32::from_bits(epsilon));
    let mut i = lane;
    while i < value_dim {
        unsafe {
            write(output, base + i, read(raw, base + i) * scale * read(weights, i) * silu(read(gate, base + i)));
        }
        i += 32;
    }
}

/// # Safety
/// Input contains ceil((count+half)/2) packed BF16 words, output count FP32
/// elements. Buffers are disjoint, half is 0 or 1, and count is nonzero.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dequantize_bf16(input: u64, output: u64, count: u32, half: u32) {
    let i = super::index();
    if i < count {
        let bits = unsafe { (input as *const u16).add((u64::from(i) + u64::from(half)) as usize).read() };
        unsafe {
            write(output, i, f32::from_bits(u32::from(bits) << 16));
        }
    }
}
