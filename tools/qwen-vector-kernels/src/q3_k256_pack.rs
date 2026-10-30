use super::{_block_idx_x, _thread_idx_x, add, mul, shuffle};
use core::arch::asm;

#[inline(always)]
unsafe fn divide(a: f32, b: f32) -> f32 {
    let out: f32;
    unsafe {
        asm!("div.rn.f32 {out}, {a}, {b};", out=out(reg32) out,a=in(reg32) a,b=in(reg32) b,options(nomem,nostack,pure));
    }
    out
}

#[inline(always)]
unsafe fn residual(value: f32, q: i32, unit: f32) -> f32 {
    let out: f32;
    let q = q as f32;
    let unit = -unit;
    unsafe {
        asm!("fma.rn.f32 {out}, {q}, {unit}, {value};", out=out(reg32) out,q=in(reg32) q,unit=in(reg32) unit,value=in(reg32) value,options(nomem,nostack,pure));
    }
    out
}

#[inline(always)]
unsafe fn quantize(value: f32, unit: f32) -> i32 {
    let scaled = unsafe { divide(value, unit) };
    let biased = unsafe { add(scaled, if scaled < 0.0 { -0.5 } else { 0.5 }) };
    let out: i32;
    unsafe {
        asm!("cvt.rzi.s32.f32 {out}, {biased};",out=out(reg32) out,biased=in(reg32) biased,options(nomem,nostack,pure));
    }
    out
}

#[inline(always)]
unsafe fn shared_address() -> u32 {
    let out: u32;
    unsafe {
        asm!(".shared .align 16 .b8 pack_maximum[16]; mov.u32 {out}, pack_maximum;",out=out(reg32) out,options(nostack));
    }
    out
}

/// # Safety
/// Launch 128 threads per row/K256 block. The input holds rows*inner F32
/// values; inner is positive and divisible by 256. Storage holds three byte
/// planes followed by rows*(inner/256) F32 scales, disjoint from input.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn q3_k256_components_pack(
    input: *const f32, storage: *mut u32, rows: u32, inner: u32, _cols: u32, _format: u32, _phase: u32, _components: u32,
) {
    let blocks = inner / 256;
    if blocks == 0 {
        return;
    }
    let index = unsafe { _block_idx_x() };
    let row = index / blocks;
    if row >= rows {
        return;
    }
    let block = index % blocks;
    let thread = unsafe { _thread_idx_x() };
    let at = row * inner + block * 256 + thread;
    let mut x0 = unsafe { input.add(at as usize).read() };
    let mut x1 = unsafe { input.add((at + 128) as usize).read() };
    let a0 = f32::from_bits(x0.to_bits() & 0x7fffffff);
    let a1 = f32::from_bits(x1.to_bits() & 0x7fffffff);
    let mut maximum = if a0 > a1 { a0 } else { a1 };
    for shift in [16, 8, 4, 2, 1] {
        let other = f32::from_bits(unsafe { shuffle(maximum.to_bits(), shift) });
        maximum = if maximum > other { maximum } else { other };
    }
    let shared = unsafe { shared_address() };
    if thread % 32 == 0 {
        let address = shared + (thread / 32) * 4;
        unsafe {
            asm!("st.shared.f32 [{address}], {maximum};",address=in(reg32) address,maximum=in(reg32) maximum,options(nostack));
        }
    }
    unsafe {
        asm!("bar.sync 0;", options(nostack));
    }
    for warp in 0..4 {
        let address = shared + warp * 4;
        let other: f32;
        unsafe {
            asm!("ld.shared.f32 {other}, [{address}];",other=out(reg32) other,address=in(reg32) address,options(nostack,readonly));
        }
        maximum = if maximum > other { maximum } else { other };
    }
    let scale = if maximum == 0.0 { 1.0 } else { unsafe { divide(maximum, 127.0) } };
    let plane = rows * inner;
    if thread == 0 {
        unsafe { storage.add((3 * plane / 4 + row * blocks + block) as usize).write(scale.to_bits()) };
    }
    let bytes = storage.cast::<i8>();
    for (component, divisor) in [(0, 1.0f32), (1, 1.0f32 / 128.0), (2, 1.0f32 / 16384.0)] {
        let unit = unsafe { mul(scale, divisor) };
        let q0 = unsafe { quantize(x0, unit) };
        let q1 = unsafe { quantize(x1, unit) };
        unsafe {
            bytes.add((component * plane + at) as usize).write(q0 as i8);
            bytes.add((component * plane + at + 128) as usize).write(q1 as i8);
        }
        if component < 2 {
            x0 = unsafe { residual(x0, q0, unit) };
            x1 = unsafe { residual(x1, q1, unit) };
        }
    }
}
