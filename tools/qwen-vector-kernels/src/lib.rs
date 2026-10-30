#![no_std]
#![feature(abi_ptx, stdarch_nvptx, asm_experimental_arch)]
use core::arch::{
    asm,
    nvptx::{_block_idx_x, _thread_idx_x},
};

#[inline(always)]
unsafe fn shuffle(value: u32, mask: u32) -> u32 {
    let out: u32;
    unsafe {
        asm!("shfl.sync.bfly.b32 {out}, {value}, {mask}, 31, -1;", out=out(reg32) out, value=in(reg32) value, mask=in(reg32) mask, options(nomem, nostack));
    }
    out
}
#[inline(always)]
unsafe fn dot(a: u32, b: u32, sum: i32) -> i32 {
    let out: i32;
    unsafe {
        asm!("dp4a.s32.s32 {out}, {a}, {b}, {sum};", out=out(reg32) out, a=in(reg32) a, b=in(reg32) b, sum=in(reg32) sum, options(nomem, nostack, pure));
    }
    out
}
#[inline(always)]
unsafe fn widen(value: u16) -> f32 {
    let out: f32;
    unsafe {
        asm!("cvt.f32.f16 {out}, {value};", out=out(reg32) out, value=in(reg16) value, options(nomem, nostack, pure));
    }
    out
}
#[inline(always)]
unsafe fn mul(a: f32, b: f32) -> f32 {
    let out: f32;
    unsafe {
        asm!("mul.rn.f32 {out}, {a}, {b};", out=out(reg32) out, a=in(reg32) a, b=in(reg32) b, options(nomem, nostack, pure));
    }
    out
}
#[inline(always)]
unsafe fn add(a: f32, b: f32) -> f32 {
    let out: f32;
    unsafe {
        asm!("add.rn.f32 {out}, {a}, {b};", out=out(reg32) out, a=in(reg32) a, b=in(reg32) b, options(nomem, nostack, pure));
    }
    out
}
#[inline(always)]
unsafe fn sub(a: f32, b: f32) -> f32 {
    let out: f32;
    unsafe {
        asm!("sub.rn.f32 {out}, {a}, {b};", out=out(reg32) out, a=in(reg32) a, b=in(reg32) b, options(nomem, nostack, pure));
    }
    out
}
#[inline(always)]
unsafe fn mask_index(value: u32, mask: u32) -> u32 {
    let out: u32;
    unsafe {
        asm!("lop3.b32 {out}, {value}, {mask}, 0, 0xc0;", out=out(reg32) out, value=in(reg32) value, mask=in(reg32) mask, options(nomem, nostack, pure));
    }
    out
}
#[inline(always)]
unsafe fn word(p: *const u16, index: u32) -> u32 {
    unsafe { p.add(index as usize).read() as u32 | ((p.add(index as usize + 1).read() as u32) << 16) }
}
#[inline(always)]
unsafe fn byte(p: *const u16, index: u32) -> u32 {
    unsafe { (p.add((index / 2) as usize).read() as u32 >> ((index & 1) * 8)) & 255 }
}
#[inline(always)]
fn signed_q3(low: u32, high: u32, qshift: u32, hshift: u32) -> u32 {
    let low = (low >> qshift) & 0x03030303;
    let negative = ((high >> hshift) & 0x01010101) ^ 0x01010101;
    low | negative.wrapping_mul(252)
}

// One warp per output, four warps per CTA. Rows must equal one. Each lane
// handles eight adjacent values, sharing one subgroup scale and A scale.
// Subgroup scale follows the integer dot, avoiding S8 scale-product overflow.
/// # Safety
/// Launch 128 threads with one row and a nonzero `inner` divisible by 256.
/// Activation storage must contain three byte planes and one F32 scale per 32
/// inputs. Weights must contain `cols * (inner / 256)` Q3_K blocks. Output must
/// hold `cols` F32 values without overlapping either input allocation.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q3_components_dp4a(
    activation: *const u32, weights: *const u16, output: *mut f32, _rows: u32, inner: u32, cols: u32, _format: u32, _phase: u32, _components: u32,
) {
    let thread = unsafe { _thread_idx_x() };
    let lane = thread & 31;
    let col = unsafe { _block_idx_x() } * 4 + thread / 32;
    if col >= cols {
        return;
    }
    let blocks = inner / 256;
    let sg = lane / 2;
    let offset = (lane * 8) & 31;
    let qshift = 2 * ((lane / 4) & 3);
    let hshift = lane / 4;
    let plane = inner / 4;
    let mut sum = 0.0f32;
    let mut correction = 0.0f32;
    for block in 0..blocks {
        let base = (col * blocks + block) * 55;
        let qindex = base + (32 + (lane / 16) * 32 + offset) / 2;
        let hindex = base + offset / 2;
        let q0 = signed_q3(unsafe { word(weights, qindex) }, unsafe { word(weights, hindex) }, qshift, hshift);
        let q1 = signed_q3(unsafe { word(weights, qindex + 2) }, unsafe { word(weights, hindex + 2) }, qshift, hshift);
        let sl = unsafe { byte(weights.add(base as usize), 96 + mask_index(sg, 7)) } >> ((sg / 8) * 4);
        let sh = unsafe { byte(weights.add(base as usize), 104 + mask_index(sg, 3)) } >> ((sg / 4) * 2);
        let scale = ((sl & 15) | ((sh & 3) << 4)) as i32 - 32;
        let d = unsafe { widen(weights.add((base + 54) as usize).read()) };
        let ai = block * 64 + lane * 2;
        let a_scale = f32::from_bits(unsafe { activation.add((3 * plane + block * 8 + lane / 4) as usize).read() });
        for (component, divisor) in [(0, 1.0f32), (1, 1.0f32 / 128.0), (2, 1.0f32 / 16384.0)] {
            let a = unsafe { activation.add((component * plane + ai) as usize) };
            let integer = unsafe { dot(q0, a.read(), 0) };
            let integer = unsafe { dot(q1, a.add(1).read(), integer) };
            let value = unsafe { mul(mul((integer * scale) as f32, mul(a_scale, divisor)), d) };
            let adjusted = unsafe { sub(value, correction) };
            let next = unsafe { add(sum, adjusted) };
            correction = unsafe { sub(sub(next, sum), adjusted) };
            sum = next;
        }
    }
    for shift in [16, 8, 4, 2, 1] {
        sum = unsafe { add(sum, f32::from_bits(shuffle(sum.to_bits(), shift))) };
    }
    if lane == 0 {
        unsafe { output.add(col as usize).write(sum) };
    }
}

mod q3_padded;
mod q3_prefill;
mod q4;
