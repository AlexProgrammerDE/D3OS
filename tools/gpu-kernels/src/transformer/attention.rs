//! Cooperative normalization and rotary preparation for consecutive token rows.
use super::{exp, log, read, sqrt, write};
use crate::warp;
use core::arch::asm;

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
