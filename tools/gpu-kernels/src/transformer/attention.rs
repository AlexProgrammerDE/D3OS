//! Cooperative normalization and rotary preparation for consecutive token rows.
use super::{barrier, exp, log, read, shared_read, shared_write, sigmoid, sqrt, write};
use crate::warp;
use core::arch::asm;
use core::arch::nvptx::{_block_idx_x, _thread_idx_x};

/// # Safety
/// Input[rows,heads,dim*stride], weights[dim], and output[rows,heads,dim]
/// are disjoint FP32 buffers. Products and position+rows fit u32. Stride is
/// 1 or 2, rotary is positive and even, and rotary<=dim. Launch full warps,
/// at least rows*heads warps; every active warp executes every collective.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prepare_batch(
    input: u64, weights: u64, output: u64, heads: u32, dim: u32, rotary: u32, position: u32, theta: u32, epsilon: u32, stride: u32, rows: u32,
) {
    let index = crate::index();
    let head = index / 32;
    let lane = index % 32;
    if head >= rows * heads {
        return;
    }
    let base = head * dim * stride;
    let dest = head * dim;
    let mut square = 0.;
    for block in 0..dim.div_ceil(32) {
        let k = block * 32 + lane;
        if k < dim {
            let value = unsafe { read(input, base + k) };
            square += value * value;
        }
    }
    square = unsafe { warp::broadcast(warp::sum(square)) };
    let scale = 1. / sqrt(square / dim as f32 + f32::from_bits(epsilon));
    let position = position + head / heads;
    let half = rotary / 2;
    for block in 0..half.div_ceil(32) {
        let i = block * 32 + lane;
        if i < half {
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
                let a = read(input, base + i) * scale * (1. + read(weights, i));
                let b = read(input, base + i + half) * scale * (1. + read(weights, i + half));
                write(output, dest + i, a * cos - b * sin);
                write(output, dest + i + half, b * cos + a * sin);
            }
        }
    }
    for block in 0..(dim - rotary).div_ceil(32) {
        let relative = block * 32 + lane;
        if relative < dim - rotary {
            let i = rotary + relative;
            unsafe {
                write(output, dest + i, read(input, base + i) * scale * (1. + read(weights, i)));
            }
        }
    }
}

/// # Safety
/// Query[rows,heads,dim], keys/values[position+rows,kv_heads,dim],
/// query_gate[rows,heads,2*dim], and output[rows,heads,dim] are disjoint FP32
/// arrays. Heads is divisible by kv_heads, 1<=dim<=256, and all products fit
/// u32. Launch one 128-thread CTA per row and head. All threads reach barriers.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_fused_batch(
    query: u64, keys: u64, values: u64, query_gate: u64, output: u64, rows: u32, heads: u32, kv_heads: u32, dim: u32, position: u32,
) {
    let (block, thread) = unsafe { (_block_idx_x(), _thread_idx_x()) };
    if block >= rows * heads {
        return;
    }
    let lane = thread % 32;
    let warp_index = thread / 32;
    let row = block / heads;
    let kv = block % heads / (heads / kv_heads);
    let visible = position + row + 1;
    let base: u32;
    unsafe {
        // 128 scores, four warp partials, maximum, mass, and rescale factor.
        asm!("{{ .shared .align 16 .b8 attention_shared[540]; mov.u32 {base}, attention_shared; }}", base = out(reg32) base, options(nostack));
    }
    if thread == 0 {
        unsafe {
            shared_write(base, 132, f32::NEG_INFINITY);
            shared_write(base, 133, 0.);
        }
    }
    let mut accumulated = [0.; 2];
    let mut start = 0;
    while start < visible {
        let count = (visible - start).min(128);
        let mut key = warp_index;
        while key < count {
            let mut score = 0.;
            for part in 0..dim.div_ceil(32) {
                let col = part * 32 + lane;
                if col < dim {
                    score += unsafe { read(query, block * dim + col) * read(keys, ((start + key) * kv_heads + kv) * dim + col) };
                }
            }
            score = unsafe { warp::sum(score) };
            if lane == 0 {
                unsafe { shared_write(base, key, score / sqrt(dim as f32)) };
            }
            key += 4;
        }
        unsafe { barrier() };
        let score = if thread < count {
            unsafe { shared_read(base, thread) }
        } else {
            f32::NEG_INFINITY
        };
        let maximum = unsafe { warp::maximum(score) };
        if lane == 0 {
            unsafe { shared_write(base, 128 + warp_index, maximum) };
        }
        unsafe { barrier() };
        if thread == 0 {
            let old = unsafe { shared_read(base, 132) };
            let mut maximum = old;
            for warp in 0..4 {
                maximum = maximum.max(unsafe { shared_read(base, 128 + warp) });
            }
            unsafe {
                shared_write(base, 132, maximum);
                shared_write(base, 134, exp(old - maximum));
            }
        }
        unsafe { barrier() };
        let weight = if thread < count { exp(score - unsafe { shared_read(base, 132) }) } else { 0. };
        unsafe { shared_write(base, thread, weight) };
        let mass = unsafe { warp::sum(weight) };
        if lane == 0 {
            unsafe { shared_write(base, 128 + warp_index, mass) };
        }
        unsafe { barrier() };
        if thread == 0 {
            let old_mass = unsafe { shared_read(base, 133) * shared_read(base, 134) };
            let mut mass = old_mass;
            for warp in 0..4 {
                mass += unsafe { shared_read(base, 128 + warp) };
            }
            unsafe {
                shared_write(base, 133, mass);
                shared_write(base, 134, old_mass / mass);
            }
        }
        unsafe { barrier() };
        // Normalize before multiplying values so a large finite value does
        // not overflow merely because several equal scores share a tile.
        unsafe { shared_write(base, thread, weight / shared_read(base, 133)) };
        unsafe { barrier() };
        let factor = unsafe { shared_read(base, 134) };
        for (slot, sum) in accumulated.iter_mut().enumerate() {
            let col = thread + slot as u32 * 128;
            if col < dim {
                *sum *= factor;
                for key in 0..count {
                    *sum += unsafe { shared_read(base, key) * read(values, ((start + key) * kv_heads + kv) * dim + col) };
                }
            }
        }
        // Other warps still consume these weights before the next tile.
        unsafe { barrier() };
        start += count;
    }
    for (slot, sum) in accumulated.into_iter().enumerate() {
        let col = thread + slot as u32 * 128;
        if col < dim {
            let gate = unsafe { read(query_gate, block * 2 * dim + dim + col) };
            unsafe { write(output, block * dim + col, sum * sigmoid(gate)) };
        }
    }
}
