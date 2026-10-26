// Pure cuTile SIMD log1p, ported from libm 0.2.16 math/log1pf.rs and math/log1p.rs.
// origin: FreeBSD /usr/src/lib/msun/src/s_log1pf.c and s_log1p.c
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
// The pinned frontend parses integer literals through i32; wider constants use
// unsigned 16-bit chunks and fixed, in-range shifts.
// Each arithmetic primitive rounds separately to nearest, ties to even, with
// gradual underflow. Explicit rounding prevents multiply/add contraction.
// Domain and tiny lanes use zero during reduction; final bit selection restores
// signed zeros/subnormals, infinities and NaNs. Floating exceptions are not exposed.

fn log1p_constant_f32(bits: u32) -> Tile<f32, { [128] }> {
    bitcast(broadcast_scalar(bits, shape![128]))
}
fn log1p_add_f32(a: Tile<f32, { [128] }>, b: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
    addf(a, b, rounding::NearestEven, ftz::Disabled)
}
fn log1p_sub_f32(a: Tile<f32, { [128] }>, b: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
    subf(a, b, rounding::NearestEven, ftz::Disabled)
}
fn log1p_mul_f32(a: Tile<f32, { [128] }>, b: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
    mulf(a, b, rounding::NearestEven, ftz::Disabled)
}
fn log1p_div_f32(a: Tile<f32, { [128] }>, b: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
    divf(a, b, rounding::NearestEven, ftz::Disabled)
}

fn float_math_log1p_f32(bits: Tile<u32, { [128] }>) -> Tile<u32, { [128] }> {
    let magnitude = bits & broadcast_scalar(0x7fffffffu32, shape![128]);
    let negative = ne_tile(
        bits & broadcast_scalar(0x8000u32 << 16u32, shape![128]),
        broadcast_scalar(0x00000000u32, shape![128]),
    );
    let finite = lt_tile(magnitude, broadcast_scalar(0x7f800000u32, shape![128]));
    let domain = negative & ge_tile(magnitude, broadcast_scalar(0x3f800000u32, shape![128]));
    let tiny = lt_tile(magnitude, broadcast_scalar(0x33800000u32, shape![128]));
    let active = finite & eq_tile(domain, broadcast_scalar(false, shape![128])) & eq_tile(tiny, broadcast_scalar(false, shape![128]));
    let x: Tile<f32, { [128] }> = bitcast(select(active, bits, broadcast_scalar(0x00000000u32, shape![128])));
    let one = log1p_constant_f32(0x3f800000u32);
    let zero = log1p_constant_f32(0x00000000u32);
    let safe_bits = select(active, bits, broadcast_scalar(0x00000000u32, shape![128]));
    let direct = (lt_tile(safe_bits, broadcast_scalar(0x3ed413d0u32, shape![128]))
        | ne_tile(safe_bits >> broadcast_scalar(31u32, shape![128]), broadcast_scalar(0u32, shape![128])))
        & le_tile(safe_bits, broadcast_scalar((0xbe95u32 << 16u32) | 0xf619u32, shape![128]));
    let u = log1p_add_f32(one, x);
    let u_bits: Tile<u32, { [128] }> = bitcast(u);
    let adjusted = u_bits + broadcast_scalar(0x004afb0du32, shape![128]);
    let exponent: Tile<i32, { [128] }> = bitcast(adjusted >> broadcast_scalar(23u32, shape![128]));
    let reduced_bits = (adjusted & broadcast_scalar(0x007fffffu32, shape![128])) + broadcast_scalar(0x3f3504f3u32, shape![128]);
    let k = select(direct, broadcast_scalar(0i32, shape![128]), exponent - broadcast_scalar(127i32, shape![128]));
    let correction_active = eq_tile(direct, broadcast_scalar(false, shape![128])) & lt_tile(k, broadcast_scalar(25i32, shape![128]));
    let correction_numerator = select(
        ge_tile(k, broadcast_scalar(2i32, shape![128])),
        log1p_sub_f32(one, log1p_sub_f32(u, x)),
        log1p_sub_f32(x, log1p_sub_f32(u, one)),
    );
    // Avoid evaluating an inactive tiny correction divided by a huge u.
    let c = log1p_div_f32(select(correction_active, correction_numerator, zero), select(correction_active, u, one));
    let reduced: Tile<f32, { [128] }> = bitcast(reduced_bits);
    let f = select(direct, x, log1p_sub_f32(reduced, one));
    let hfsq = log1p_mul_f32(log1p_mul_f32(log1p_constant_f32(0x3f000000u32), f), f);
    let s = log1p_div_f32(f, log1p_add_f32(log1p_constant_f32(0x40000000u32), f));
    let z = log1p_mul_f32(s, s);
    let w = log1p_mul_f32(z, z);
    let even = log1p_add_f32(log1p_constant_f32(0x3eccce13u32), log1p_mul_f32(w, log1p_constant_f32(0x3e789e26u32)));
    let odd = log1p_add_f32(log1p_constant_f32(0x3f2aaaaau32), log1p_mul_f32(w, log1p_constant_f32(0x3e91e9eeu32)));
    let t1 = log1p_mul_f32(w, even);
    let t2 = log1p_mul_f32(z, odd);
    let r = log1p_add_f32(t2, t1);
    let dk: Tile<f32, { [128] }> = itof(k, rounding::NearestEven);
    let polynomial = log1p_mul_f32(s, log1p_add_f32(hfsq, r));
    let correction = log1p_add_f32(log1p_mul_f32(dk, log1p_constant_f32(0x3717f7d1u32)), c);
    let low = log1p_sub_f32(log1p_add_f32(polynomial, correction), hfsq);
    let value = log1p_add_f32(log1p_add_f32(low, f), log1p_mul_f32(dk, log1p_constant_f32(0x3f317180u32)));
    let result: Tile<u32, { [128] }> = bitcast(value);
    let result = select(tiny, bits, result);
    let result = select(domain, broadcast_scalar(0x7fc00000u32, shape![128]), result);
    let result = select(
        eq_tile(bits, broadcast_scalar(0xbf80u32 << 16u32, shape![128])),
        broadcast_scalar(0xff80u32 << 16u32, shape![128]),
        result,
    );
    let result = select(eq_tile(bits, broadcast_scalar(0x7f800000u32, shape![128])), bits, result);
    select(
        gt_tile(magnitude, broadcast_scalar(0x7f800000u32, shape![128])),
        bits | broadcast_scalar(0x00400000u32, shape![128]),
        result,
    )
}

fn log1p_constant_f64(bits: u64) -> Tile<f64, { [128] }> {
    bitcast(broadcast_scalar(bits, shape![128]))
}
fn log1p_add_f64(a: Tile<f64, { [128] }>, b: Tile<f64, { [128] }>) -> Tile<f64, { [128] }> {
    addf(a, b, rounding::NearestEven, ftz::Disabled)
}
fn log1p_sub_f64(a: Tile<f64, { [128] }>, b: Tile<f64, { [128] }>) -> Tile<f64, { [128] }> {
    subf(a, b, rounding::NearestEven, ftz::Disabled)
}
fn log1p_mul_f64(a: Tile<f64, { [128] }>, b: Tile<f64, { [128] }>) -> Tile<f64, { [128] }> {
    mulf(a, b, rounding::NearestEven, ftz::Disabled)
}
fn log1p_div_f64(a: Tile<f64, { [128] }>, b: Tile<f64, { [128] }>) -> Tile<f64, { [128] }> {
    divf(a, b, rounding::NearestEven, ftz::Disabled)
}

fn float_math_log1p_f64(bits: Tile<u64, { [128] }>) -> Tile<u64, { [128] }> {
    let magnitude = bits & broadcast_scalar((0x7fffu64 << 48u64) | (0xffffu64 << 32u64) | (0xffffu64 << 16u64) | 0xffffu64, shape![128]);
    let negative = ne_tile(
        bits & broadcast_scalar(0x8000u64 << 48u64, shape![128]),
        broadcast_scalar(0x0000000000000000u64, shape![128]),
    );
    let finite = lt_tile(magnitude, broadcast_scalar(0x7ff0u64 << 48u64, shape![128]));
    let domain = negative & ge_tile(magnitude, broadcast_scalar(0x3ff0u64 << 48u64, shape![128]));
    let tiny = lt_tile(magnitude, broadcast_scalar(0x3ca0u64 << 48u64, shape![128]));
    let active = finite & eq_tile(domain, broadcast_scalar(false, shape![128])) & eq_tile(tiny, broadcast_scalar(false, shape![128]));
    let x: Tile<f64, { [128] }> = bitcast(select(active, bits, broadcast_scalar(0x0000000000000000u64, shape![128])));
    let one = log1p_constant_f64(0x3ff0u64 << 48u64);
    let zero = log1p_constant_f64(0x0000000000000000u64);
    let safe_bits = select(active, bits, broadcast_scalar(0x0000000000000000u64, shape![128]));
    let hx: Tile<u32, { [128] }> = trunci(safe_bits >> broadcast_scalar(32u64, shape![128]), overflow::NoUnsignedWrap);
    let direct = (lt_tile(hx, broadcast_scalar(0x3fda827au32, shape![128]))
        | ne_tile(hx >> broadcast_scalar(31u32, shape![128]), broadcast_scalar(0u32, shape![128])))
        & le_tile(hx, broadcast_scalar((0xbfd2u32 << 16u32) | 0xbec4u32, shape![128]));
    let u = log1p_add_f64(one, x);
    let u_bits: Tile<u64, { [128] }> = bitcast(u);
    let hu: Tile<u32, { [128] }> = trunci(u_bits >> broadcast_scalar(32u64, shape![128]), overflow::NoUnsignedWrap);
    let adjusted = hu + broadcast_scalar(0x00095f62u32, shape![128]);
    let exponent: Tile<i32, { [128] }> = bitcast(adjusted >> broadcast_scalar(20u32, shape![128]));
    let reduced_high = (adjusted & broadcast_scalar(0x000fffffu32, shape![128])) + broadcast_scalar(0x3fe6a09eu32, shape![128]);
    let reduced_bits =
        (exti(reduced_high) << broadcast_scalar(32u64, shape![128])) | (u_bits & broadcast_scalar((0xffffu64 << 16u64) | 0xffffu64, shape![128]));
    let k = select(direct, broadcast_scalar(0i32, shape![128]), exponent - broadcast_scalar(1023i32, shape![128]));
    let correction_active = eq_tile(direct, broadcast_scalar(false, shape![128])) & lt_tile(k, broadcast_scalar(54i32, shape![128]));
    let correction_numerator = select(
        ge_tile(k, broadcast_scalar(2i32, shape![128])),
        log1p_sub_f64(one, log1p_sub_f64(u, x)),
        log1p_sub_f64(x, log1p_sub_f64(u, one)),
    );
    // Avoid evaluating an inactive tiny correction divided by a huge u.
    let c = log1p_div_f64(select(correction_active, correction_numerator, zero), select(correction_active, u, one));
    let reduced: Tile<f64, { [128] }> = bitcast(reduced_bits);
    let f = select(direct, x, log1p_sub_f64(reduced, one));
    let hfsq = log1p_mul_f64(log1p_mul_f64(log1p_constant_f64(0x3fe0u64 << 48u64), f), f);
    let s = log1p_div_f64(f, log1p_add_f64(log1p_constant_f64(0x4000u64 << 48u64), f));
    let z = log1p_mul_f64(s, s);
    let w = log1p_mul_f64(z, z);
    let even_tail = log1p_add_f64(
        log1p_constant_f64((0x3fccu64 << 48u64) | (0x71c5u64 << 32u64) | (0x1d8eu64 << 16u64) | 0x78afu64),
        log1p_mul_f64(
            w,
            log1p_constant_f64((0x3fc3u64 << 48u64) | (0x9a09u64 << 32u64) | (0xd078u64 << 16u64) | 0xc69fu64),
        ),
    );
    let even = log1p_add_f64(
        log1p_constant_f64((0x3fd9u64 << 48u64) | (0x9999u64 << 32u64) | (0x9997u64 << 16u64) | 0xfa04u64),
        log1p_mul_f64(w, even_tail),
    );
    let odd_tail = log1p_add_f64(
        log1p_constant_f64((0x3fc7u64 << 48u64) | (0x4664u64 << 32u64) | (0x96cbu64 << 16u64) | 0x03deu64),
        log1p_mul_f64(
            w,
            log1p_constant_f64((0x3fc2u64 << 48u64) | (0xf112u64 << 32u64) | (0xdf3eu64 << 16u64) | 0x5244u64),
        ),
    );
    let odd_middle = log1p_add_f64(
        log1p_constant_f64((0x3fd2u64 << 48u64) | (0x4924u64 << 32u64) | (0x9422u64 << 16u64) | 0x9359u64),
        log1p_mul_f64(w, odd_tail),
    );
    let odd = log1p_add_f64(
        log1p_constant_f64((0x3fe5u64 << 48u64) | (0x5555u64 << 32u64) | (0x5555u64 << 16u64) | 0x5593u64),
        log1p_mul_f64(w, odd_middle),
    );
    let t1 = log1p_mul_f64(w, even);
    let t2 = log1p_mul_f64(z, odd);
    let r = log1p_add_f64(t2, t1);
    let dk: Tile<f64, { [128] }> = itof(k, rounding::NearestEven);
    let polynomial = log1p_mul_f64(s, log1p_add_f64(hfsq, r));
    let correction = log1p_add_f64(
        log1p_mul_f64(
            dk,
            log1p_constant_f64((0x3deau64 << 48u64) | (0x39efu64 << 32u64) | (0x3579u64 << 16u64) | 0x3c76u64),
        ),
        c,
    );
    let low = log1p_sub_f64(log1p_add_f64(polynomial, correction), hfsq);
    let value = log1p_add_f64(
        log1p_add_f64(low, f),
        log1p_mul_f64(dk, log1p_constant_f64((0x3fe6u64 << 48u64) | (0x2e42u64 << 32u64) | 0xfee0u64 << 16u64)),
    );
    let result: Tile<u64, { [128] }> = bitcast(value);
    let result = select(tiny, bits, result);
    let result = select(domain, broadcast_scalar(0x7ff8u64 << 48u64, shape![128]), result);
    let result = select(
        eq_tile(bits, broadcast_scalar(0xbff0u64 << 48u64, shape![128])),
        broadcast_scalar(0xfff0u64 << 48u64, shape![128]),
        result,
    );
    let result = select(eq_tile(bits, broadcast_scalar(0x7ff0u64 << 48u64, shape![128])), bits, result);
    select(
        gt_tile(magnitude, broadcast_scalar(0x7ff0u64 << 48u64, shape![128])),
        bits | broadcast_scalar(0x0008u64 << 48u64, shape![128]),
        result,
    )
}
