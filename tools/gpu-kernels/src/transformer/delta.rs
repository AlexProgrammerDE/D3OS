//! One CTA per head fuses the recurrent delta step without global scratch.
use super::{barrier, exp, log1p, read, shared_read, shared_write, sigmoid, silu, sqrt, write};
use crate::warp;
use core::arch::{
    asm,
    nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x},
};

#[inline(always)]
unsafe fn add_key_partner<const SHIFT: u32>(value: f32) -> f32 {
    let other: f32;
    unsafe {
        asm!("shfl.sync.bfly.b32 {d}, {a}, {shift}, 31, -1;", d = out(reg32) other, a = in(reg32) value, shift = const SHIFT, options(nostack));
    }
    value + other
}

#[inline(always)]
unsafe fn sum_keys(value: f32) -> f32 {
    let value = unsafe { add_key_partner::<16>(value) };
    let value = unsafe { add_key_partner::<8>(value) };
    unsafe { add_key_partner::<4>(value) }
}

/// # Safety
/// All buffers are disjoint FP32 arrays with extents given by DeltaFused::work.
/// Key/value dimensions are 1..=128, convolution width 1..=8. Launch one CTA per
/// head with ceil(value_dim/4)*32 threads. Every thread reaches every barrier.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn delta_fused(
    qkv: u64, gate: u64, beta: u64, alpha: u64, convolution: u64, log_decay: u64, time_bias: u64, norm: u64, history: u64, state: u64, output: u64, heads: u32,
    key_dim: u32, value_dim: u32, kernel: u32, epsilon: u32,
) {
    let (head, thread, threads) = unsafe { (_block_idx_x(), _thread_idx_x(), _block_dim_x()) };
    let base: u32;
    unsafe {
        asm!("{{ .shared .align 16 .b8 delta_shared[2064]; mov.u32 {base}, delta_shared; }}", base = out(reg32) base, options(nostack));
    }
    let mut channel = thread;
    while channel < 2 * key_dim + value_dim {
        let (global, local) = if channel < key_dim {
            (head * key_dim + channel, channel)
        } else if channel < 2 * key_dim {
            (heads * key_dim + head * key_dim + channel - key_dim, 128 + channel - key_dim)
        } else {
            (2 * heads * key_dim + head * value_dim + channel - 2 * key_dim, 256 + channel - 2 * key_dim)
        };
        let mut sum = 0.;
        for k in 0..kernel {
            let value = unsafe {
                if k + 1 < kernel {
                    read(history, global * kernel + k + 1)
                } else {
                    read(qkv, global)
                }
            };
            unsafe {
                write(history, global * kernel + k, value);
                sum += value * read(convolution, global * kernel + k);
            }
        }
        unsafe {
            shared_write(base, local, silu(sum));
        }
        channel += threads;
    }
    unsafe {
        barrier();
    }

    if thread < 32 {
        let mut q_sum = 0.;
        let mut k_sum = 0.;
        let mut k = thread;
        while k < key_dim {
            let q = unsafe { shared_read(base, k) };
            let key = unsafe { shared_read(base, 128 + k) };
            q_sum += q * q;
            k_sum += key * key;
            k += 32;
        }
        q_sum = unsafe { warp::sum(q_sum) };
        k_sum = unsafe { warp::sum(k_sum) };
        if thread == 0 {
            let alpha = unsafe { read(alpha, head) + read(time_bias, head) };
            let softplus = alpha.max(0.) + log1p(exp(-alpha.abs()));
            unsafe {
                shared_write(base, 512, 1. / sqrt(q_sum + 1e-6) / sqrt(key_dim as f32));
                shared_write(base, 513, 1. / sqrt(k_sum + 1e-6));
                shared_write(base, 514, sigmoid(read(beta, head)));
                shared_write(base, 515, exp(-exp(read(log_decay, head)) * softplus));
            }
        }
    }
    unsafe {
        barrier();
    }

    let column = thread / 32 * 4 + thread % 4;
    let key_lane = thread % 32 / 4;
    let valid = column < value_dim;
    let q_scale = unsafe { shared_read(base, 512) };
    let k_scale = unsafe { shared_read(base, 513) };
    let beta = unsafe { shared_read(base, 514) };
    let decay = unsafe { shared_read(base, 515) };
    let mut values = [0.; 16];
    let mut prediction = 0.;
    for (slot, value) in values.iter_mut().enumerate() {
        let k = slot as u32 * 8 + key_lane;
        if valid && k < key_dim {
            *value = unsafe { read(state, (head * key_dim + k) * value_dim + column) } * decay;
            prediction += *value * (unsafe { shared_read(base, 128 + k) } * k_scale);
        }
    }
    prediction = unsafe { sum_keys(prediction) };
    let delta = if valid {
        (unsafe { shared_read(base, 256 + column) } - prediction) * beta
    } else {
        0.
    };
    let mut result = 0.;
    for (slot, &decayed) in values.iter().enumerate() {
        let k = slot as u32 * 8 + key_lane;
        if valid && k < key_dim {
            let value = decayed + unsafe { shared_read(base, 128 + k) } * k_scale * delta;
            unsafe {
                write(state, (head * key_dim + k) * value_dim + column, value);
            }
            result += value * (unsafe { shared_read(base, k) } * q_scale);
        }
    }
    result = unsafe { sum_keys(result) };
    if valid && key_lane == 0 {
        unsafe {
            shared_write(base, 384 + column, result);
        }
    }
    unsafe {
        barrier();
    }

    if thread < 32 {
        let mut square = 0.;
        let mut column = thread;
        while column < value_dim {
            let value = unsafe { shared_read(base, 384 + column) };
            square += value * value;
            column += 32;
        }
        square = unsafe { warp::sum(square) };
        if thread == 0 {
            unsafe {
                shared_write(base, 512, 1. / sqrt(square / value_dim as f32 + f32::from_bits(epsilon)));
            }
        }
    }
    unsafe {
        barrier();
    }
    if thread < value_dim {
        let index = head * value_dim + thread;
        unsafe {
            let result = shared_read(base, 384 + thread) * shared_read(base, 512) * read(norm, thread) * silu(read(gate, index));
            write(output, index, result);
        }
    }
}
