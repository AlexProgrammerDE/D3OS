// FP32 erf adapted from FreeBSD msun and Rust libm 0.2.16.
// Preserve the original permission notice below.
/* origin: FreeBSD /usr/src/lib/msun/src/s_erff.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */
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

fn burn_erf_bits(bits: Tile<u32, { [128] }>) -> Tile<u32, { [128] }> {
    let magnitude_bits = bits & broadcast_scalar(0x7fffffffu32, shape![128]);
    let sign = bits & broadcast_scalar(unsigned(-1i32) << 31u32, shape![128]);
    let magnitude: Tile<f32, { [128] }> = bitcast(magnitude_bits);
    let one = broadcast_scalar(1f32, shape![128]);
    let zero = broadcast_scalar(0f32, shape![128]);
    // Keep every eagerly computed branch in its finite approximation domain.
    let small = select(lt_tile(magnitude_bits, broadcast_scalar(0x3f580000u32, shape![128])), magnitude, zero);
    let z = small * small;
    let mut r = broadcast_scalar(bits_f32(unsigned(-1211672911i32)), shape![128]);
    r = broadcast_scalar(bits_f32(unsigned(-1145236343i32)), shape![128]) + z * r;
    r = broadcast_scalar(bits_f32(unsigned(-1125559665i32)), shape![128]) + z * r;
    r = broadcast_scalar(bits_f32(unsigned(-1096389653i32)), shape![128]) + z * r;
    r = broadcast_scalar(bits_f32(1040414164u32), shape![128]) + z * r;
    let mut q = broadcast_scalar(bits_f32(unsigned(-1232805350i32)), shape![128]);
    q = broadcast_scalar(bits_f32(957017673u32), shape![128]) + z * q;
    q = broadcast_scalar(bits_f32(1000767766u32), shape![128]) + z * q;
    q = broadcast_scalar(bits_f32(1032137315u32), shape![128]) + z * q;
    q = broadcast_scalar(bits_f32(1053539278u32), shape![128]) + z * q;
    let q = one + z * q;
    let small_value = small + small * divf(r, q, rounding::NearestEven, ftz::Disabled);
    let tiny = mulf(broadcast_scalar(8f32, shape![128]), small, rounding::NearestEven, ftz::Disabled);
    let extra = mulf(
        broadcast_scalar(bits_f32(1065579988u32), shape![128]),
        small,
        rounding::NearestEven,
        ftz::Disabled,
    );
    let tiny = mulf(
        broadcast_scalar(0.125f32, shape![128]),
        addf(tiny, extra, rounding::NearestEven, ftz::Disabled),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let small_value = select(lt_tile(magnitude_bits, broadcast_scalar(0x31800000u32, shape![128])), tiny, small_value);
    let middle_mask =
        ge_tile(magnitude_bits, broadcast_scalar(0x3f580000u32, shape![128])) & lt_tile(magnitude_bits, broadcast_scalar(0x3fa00000u32, shape![128]));
    let middle = select(middle_mask, magnitude, one) - one;
    let mut r = broadcast_scalar(bits_f32(unsigned(-1156712000i32)), shape![128]);
    r = broadcast_scalar(bits_f32(1024545203u32), shape![128]) + middle * r;
    r = broadcast_scalar(bits_f32(unsigned(-1109189438i32)), shape![128]) + middle * r;
    r = broadcast_scalar(bits_f32(1050869332u32), shape![128]) + middle * r;
    r = broadcast_scalar(bits_f32(unsigned(-1094807032i32)), shape![128]) + middle * r;
    r = broadcast_scalar(bits_f32(1054107653u32), shape![128]) + middle * r;
    r = broadcast_scalar(bits_f32(unsigned(-1155871290i32)), shape![128]) + middle * r;
    let mut q = broadcast_scalar(bits_f32(1011112611u32), shape![128]);
    q = broadcast_scalar(bits_f32(1012887059u32), shape![128]) + middle * q;
    q = broadcast_scalar(bits_f32(1040265991u32), shape![128]) + middle * q;
    q = broadcast_scalar(bits_f32(1033050855u32), shape![128]) + middle * q;
    q = broadcast_scalar(bits_f32(1057642373u32), shape![128]) + middle * q;
    q = broadcast_scalar(bits_f32(1037693745u32), shape![128]) + middle * q;
    let q = one + middle * q;
    let middle_value = broadcast_scalar(bits_f32(1062753803u32), shape![128]) + divf(r, q, rounding::NearestEven, ftz::Disabled);
    let tail_mask =
        ge_tile(magnitude_bits, broadcast_scalar(0x3fa00000u32, shape![128])) & lt_tile(magnitude_bits, broadcast_scalar(0x40c00000u32, shape![128]));
    let tail = select(tail_mask, magnitude, broadcast_scalar(1.25f32, shape![128]));
    let z = divf(one, tail * tail, rounding::NearestEven, ftz::Disabled);
    let mut ra = broadcast_scalar(bits_f32(unsigned(-1055062146i32)), shape![128]);
    ra = broadcast_scalar(bits_f32(unsigned(-1029532885i32)), shape![128]) + z * ra;
    ra = broadcast_scalar(bits_f32(unsigned(-1019700505i32)), shape![128]) + z * ra;
    ra = broadcast_scalar(bits_f32(unsigned(-1021155956i32)), shape![128]) + z * ra;
    ra = broadcast_scalar(bits_f32(unsigned(-1032224681i32)), shape![128]) + z * ra;
    ra = broadcast_scalar(bits_f32(unsigned(-1054281694i32)), shape![128]) + z * ra;
    ra = broadcast_scalar(bits_f32(unsigned(-1087266633i32)), shape![128]) + z * ra;
    ra = broadcast_scalar(bits_f32(unsigned(-1138646893i32)), shape![128]) + z * ra;
    let mut qa = broadcast_scalar(bits_f32(unsigned(-1116242025i32)), shape![128]);
    qa = broadcast_scalar(bits_f32(1087520636u32), shape![128]) + z * qa;
    qa = broadcast_scalar(bits_f32(1121535263u32), shape![128]) + z * qa;
    qa = broadcast_scalar(bits_f32(1138131211u32), shape![128]) + z * qa;
    qa = broadcast_scalar(bits_f32(1143036105u32), shape![128]) + z * qa;
    qa = broadcast_scalar(bits_f32(1138313327u32), shape![128]) + z * qa;
    qa = broadcast_scalar(bits_f32(1124706403u32), shape![128]) + z * qa;
    qa = broadcast_scalar(bits_f32(1100821966u32), shape![128]) + z * qa;
    let qa = one + z * qa;
    let mut rb = broadcast_scalar(bits_f32(unsigned(-1007566219i32)), shape![128]);
    rb = broadcast_scalar(bits_f32(unsigned(-998235381i32)), shape![128]) + z * rb;
    rb = broadcast_scalar(bits_f32(unsigned(-1004575679i32)), shape![128]) + z * rb;
    rb = broadcast_scalar(bits_f32(unsigned(-1021271318i32)), shape![128]) + z * rb;
    rb = broadcast_scalar(bits_f32(unsigned(-1047654325i32)), shape![128]) + z * rb;
    rb = broadcast_scalar(bits_f32(unsigned(-1085497900i32)), shape![128]) + z * rb;
    rb = broadcast_scalar(bits_f32(unsigned(-1138646894i32)), shape![128]) + z * rb;
    let mut qb = broadcast_scalar(bits_f32(unsigned(-1045199086i32)), shape![128]);
    qb = broadcast_scalar(bits_f32(1139622823u32), shape![128]) + z * qb;
    qb = broadcast_scalar(bits_f32(1159696590u32), shape![128]) + z * qb;
    qb = broadcast_scalar(bits_f32(1162345915u32), shape![128]) + z * qb;
    qb = broadcast_scalar(bits_f32(1153439577u32), shape![128]) + z * qb;
    qb = broadcast_scalar(bits_f32(1134749041u32), shape![128]) + z * qb;
    qb = broadcast_scalar(bits_f32(1106424921u32), shape![128]) + z * qb;
    let qb = one + z * qb;
    let near = lt_tile(magnitude_bits, broadcast_scalar(0x4036db6du32, shape![128]));
    let ratio = divf(select(near, ra, rb), select(near, qa, qb), rounding::NearestEven, ftz::Disabled);
    let tail_bits: Tile<u32, { [128] }> = bitcast(tail);
    let truncated: Tile<f32, { [128] }> = bitcast(tail_bits & broadcast_scalar(unsigned(-8192i32), shape![128]));
    let first = unsafe { exp_with_rounding(zero - truncated * truncated - broadcast_scalar(0.5625f32, shape![128]), rounding::Full) };
    let second = unsafe { exp_with_rounding((truncated - tail) * (truncated + tail) + ratio, rounding::Full) };
    let tail_value = one - divf(first * second, tail, rounding::NearestEven, ftz::Disabled);
    let value = select(
        lt_tile(magnitude_bits, broadcast_scalar(0x3f580000u32, shape![128])),
        small_value,
        select(middle_mask, middle_value, select(tail_mask, tail_value, one)),
    );
    let value: Tile<u32, { [128] }> = bitcast(value);
    let nan = bits | broadcast_scalar(0x00400000u32, shape![128]);
    select(gt_tile(magnitude_bits, broadcast_scalar(0x7f800000u32, shape![128])), nan, value | sign)
}
