// Typed Erf helpers for the new forward conformance target.
// Coefficients and interval reconstruction derive from Rust libm 0.2.16,
// FreeBSD msun s_erf.c/s_erff.c, and the Sun fdlibm algorithm.
// Pinned upstream: https://github.com/rust-lang/compiler-builtins/tree/dfd2203a4d6110820ad7bb65cafe1bf331a03a3d/libm
// F32 conversion by Ian Lance Taylor, Cygnus Support.
// Keep the original permission notice below.
/* origin: FreeBSD /usr/src/lib/msun/src/s_erf.c */
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

/*
Rust libm copyright (c) 2018 Jorge Aparicio

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Each polynomial node uses explicit nearest-even FMA with FTZ disabled.
// The two tail exponentials use Full precision. Device execution accuracy
// requires qualification of the complete registered entry.
// Half storage must widen exactly to F32 and use one final RNE conversion.
// This file does not register an operation or replace the existing F32 path.

fn erf_f32bits(bits: Tile<u32, { [128] }>) -> Tile<u32, { [128] }> {
    let magnitude_bits = bits & broadcast_scalar(0x7fffffffu32, shape![128]);
    let sign = bits ^ magnitude_bits;
    let magnitude: Tile<f32, { [128] }> = bitcast(magnitude_bits);
    let region = magnitude_bits;
    let one = broadcast_scalar(1f32, shape![128]);
    let zero = broadcast_scalar(0f32, shape![128]);
    // Keep every eager approximation in a finite input domain.
    let small = select(lt_tile(region, broadcast_scalar(0x3f580000u32, shape![128])), magnitude, zero);
    let z = mulf(small, small, rounding::NearestEven, ftz::Disabled);
    let mut p = broadcast_scalar(-2.3763017452e-05f32, shape![128]);
    p = fma(z, p, broadcast_scalar(-5.7702702470e-03f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    p = fma(z, p, broadcast_scalar(-2.8481749818e-02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    p = fma(z, p, broadcast_scalar(-3.2504209876e-01f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    p = fma(z, p, broadcast_scalar(1.2837916613e-01f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    let mut q = broadcast_scalar(-3.9602282413e-06f32, shape![128]);
    q = fma(z, q, broadcast_scalar(1.3249473704e-04f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    q = fma(z, q, broadcast_scalar(5.0813062117e-03f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    q = fma(z, q, broadcast_scalar(6.5022252500e-02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    q = fma(z, q, broadcast_scalar(3.9791721106e-01f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    let q = fma(z, q, one, rounding::NearestEven, ftz::Disabled);
    let small_value = fma(
        small,
        divf(p, q, rounding::NearestEven, ftz::Disabled),
        small,
        rounding::NearestEven,
        ftz::Disabled,
    );
    // Scaling by eight is exact here, including subnormal inputs.
    let scaled = mulf(broadcast_scalar(8f32, shape![128]), small, rounding::NearestEven, ftz::Disabled);
    let tiny_sum = fma(
        broadcast_scalar(1.0270333290e+00f32, shape![128]),
        small,
        scaled,
        rounding::NearestEven,
        ftz::Disabled,
    );
    let tiny = mulf(broadcast_scalar(0.125f32, shape![128]), tiny_sum, rounding::NearestEven, ftz::Disabled);
    let small_value = select(lt_tile(region, broadcast_scalar(0x31800000u32, shape![128])), tiny, small_value);
    let middle_mask = ge_tile(region, broadcast_scalar(0x3f580000u32, shape![128])) & lt_tile(region, broadcast_scalar(0x3fa00000u32, shape![128]));
    let middle = subf(select(middle_mask, magnitude, one), one, rounding::NearestEven, ftz::Disabled);
    let mut p = broadcast_scalar(-2.1663755178e-03f32, shape![128]);
    p = fma(
        middle,
        p,
        broadcast_scalar(3.5478305072e-02f32, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        middle,
        p,
        broadcast_scalar(-1.1089469492e-01f32, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        middle,
        p,
        broadcast_scalar(3.1834661961e-01f32, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        middle,
        p,
        broadcast_scalar(-3.7220788002e-01f32, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        middle,
        p,
        broadcast_scalar(4.1485610604e-01f32, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        middle,
        p,
        broadcast_scalar(-2.3621185683e-03f32, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let mut q = broadcast_scalar(1.1984500103e-02f32, shape![128]);
    q = fma(
        middle,
        q,
        broadcast_scalar(1.3637083583e-02f32, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    q = fma(
        middle,
        q,
        broadcast_scalar(1.2617121637e-01f32, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    q = fma(
        middle,
        q,
        broadcast_scalar(7.1828655899e-02f32, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    q = fma(
        middle,
        q,
        broadcast_scalar(5.4039794207e-01f32, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    q = fma(
        middle,
        q,
        broadcast_scalar(1.0642088205e-01f32, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let q = fma(middle, q, one, rounding::NearestEven, ftz::Disabled);
    let middle_value = addf(
        broadcast_scalar(8.4506291151e-01f32, shape![128]),
        divf(p, q, rounding::NearestEven, ftz::Disabled),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let tail_mask = ge_tile(region, broadcast_scalar(0x3fa00000u32, shape![128])) & lt_tile(region, broadcast_scalar(0x40c00000u32, shape![128]));
    let tail = select(tail_mask, magnitude, broadcast_scalar(1.25f32, shape![128]));
    let s = divf(
        one,
        mulf(tail, tail, rounding::NearestEven, ftz::Disabled),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let mut ra = broadcast_scalar(-9.8143291473e+00f32, shape![128]);
    ra = fma(s, ra, broadcast_scalar(-8.1287437439e+01f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    ra = fma(s, ra, broadcast_scalar(-1.8460508728e+02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    ra = fma(s, ra, broadcast_scalar(-1.6239666748e+02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    ra = fma(s, ra, broadcast_scalar(-6.2375331879e+01f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    ra = fma(s, ra, broadcast_scalar(-1.0558626175e+01f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    ra = fma(s, ra, broadcast_scalar(-6.9385856390e-01f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    ra = fma(s, ra, broadcast_scalar(-9.8649440333e-03f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    let mut qa = broadcast_scalar(-6.0424413532e-02f32, shape![128]);
    qa = fma(s, qa, broadcast_scalar(6.5702495575e+00f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    qa = fma(s, qa, broadcast_scalar(1.0863500214e+02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    qa = fma(s, qa, broadcast_scalar(4.2900814819e+02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    qa = fma(s, qa, broadcast_scalar(6.4538726807e+02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    qa = fma(s, qa, broadcast_scalar(4.3456588745e+02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    qa = fma(s, qa, broadcast_scalar(1.3765776062e+02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    qa = fma(s, qa, broadcast_scalar(1.9651271820e+01f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    let qa = fma(s, qa, one, rounding::NearestEven, ftz::Disabled);
    let mut rb = broadcast_scalar(-4.8351919556e+02f32, shape![128]);
    rb = fma(s, rb, broadcast_scalar(-1.0250950928e+03f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    rb = fma(s, rb, broadcast_scalar(-6.3756646729e+02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    rb = fma(s, rb, broadcast_scalar(-1.6063638306e+02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    rb = fma(s, rb, broadcast_scalar(-1.7757955551e+01f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    rb = fma(s, rb, broadcast_scalar(-7.9928326607e-01f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    rb = fma(s, rb, broadcast_scalar(-9.8649431020e-03f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    let mut qb = broadcast_scalar(-2.2440952301e+01f32, shape![128]);
    qb = fma(s, qb, broadcast_scalar(4.7452853394e+02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    qb = fma(s, qb, broadcast_scalar(2.5530502930e+03f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    qb = fma(s, qb, broadcast_scalar(3.1998581543e+03f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    qb = fma(s, qb, broadcast_scalar(1.5367296143e+03f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    qb = fma(s, qb, broadcast_scalar(3.2579251099e+02f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    qb = fma(s, qb, broadcast_scalar(3.0338060379e+01f32, shape![128]), rounding::NearestEven, ftz::Disabled);
    let qb = fma(s, qb, one, rounding::NearestEven, ftz::Disabled);
    let near = lt_tile(region, broadcast_scalar(0x4036db6du32, shape![128]));
    let ratio = divf(select(near, ra, rb), select(near, qa, qb), rounding::NearestEven, ftz::Disabled);
    let tail_bits: Tile<u32, { [128] }> = bitcast(tail);
    let truncated: Tile<f32, { [128] }> = bitcast(tail_bits & broadcast_scalar(unsigned(-8192i32), shape![128]));
    let first_arg = fma(
        subf(zero, truncated, rounding::NearestEven, ftz::Disabled),
        truncated,
        broadcast_scalar(-0.5625f32, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let second_arg = fma(
        subf(truncated, tail, rounding::NearestEven, ftz::Disabled),
        addf(truncated, tail, rounding::NearestEven, ftz::Disabled),
        ratio,
        rounding::NearestEven,
        ftz::Disabled,
    );
    let first = unsafe { exp_with_rounding(first_arg, rounding::Full) };
    let second = unsafe { exp_with_rounding(second_arg, rounding::Full) };
    let tail_value = subf(
        one,
        divf(
            mulf(first, second, rounding::NearestEven, ftz::Disabled),
            tail,
            rounding::NearestEven,
            ftz::Disabled,
        ),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let value = select(
        lt_tile(region, broadcast_scalar(0x3f580000u32, shape![128])),
        small_value,
        select(middle_mask, middle_value, select(tail_mask, tail_value, one)),
    );
    let value: Tile<u32, { [128] }> = bitcast(value);
    let nan = bits | broadcast_scalar(0x00400000u32, shape![128]);
    select(gt_tile(magnitude_bits, broadcast_scalar(0x7f800000u32, shape![128])), nan, value | sign)
}

fn erf_f64bits(bits: Tile<u64, { [128] }>) -> Tile<u64, { [128] }> {
    let sign_bit = broadcast_scalar(1u64, shape![128]) << broadcast_scalar(63u64, shape![128]);
    let magnitude_bits = bits & (sign_bit - broadcast_scalar(1u64, shape![128]));
    let sign = bits ^ magnitude_bits;
    let magnitude: Tile<f64, { [128] }> = bitcast(magnitude_bits);
    let region = magnitude_bits >> broadcast_scalar(32u64, shape![128]);
    let one = broadcast_scalar(1f64, shape![128]);
    let zero = broadcast_scalar(0f64, shape![128]);
    // Keep every eager approximation in a finite input domain.
    let small = select(lt_tile(region, broadcast_scalar(0x3feb0000u64, shape![128])), magnitude, zero);
    let z = mulf(small, small, rounding::NearestEven, ftz::Disabled);
    let mut p = broadcast_scalar(-2.37630166566501626084e-05f64, shape![128]);
    p = fma(
        z,
        p,
        broadcast_scalar(-5.77027029648944159157e-03f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        z,
        p,
        broadcast_scalar(-2.84817495755985104766e-02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        z,
        p,
        broadcast_scalar(-3.25042107247001499370e-01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        z,
        p,
        broadcast_scalar(1.28379167095512558561e-01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let mut q = broadcast_scalar(-3.96022827877536812320e-06f64, shape![128]);
    q = fma(
        z,
        q,
        broadcast_scalar(1.32494738004321644526e-04f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    q = fma(
        z,
        q,
        broadcast_scalar(5.08130628187576562776e-03f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    q = fma(
        z,
        q,
        broadcast_scalar(6.50222499887672944485e-02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    q = fma(
        z,
        q,
        broadcast_scalar(3.97917223959155352819e-01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let q = fma(z, q, one, rounding::NearestEven, ftz::Disabled);
    let small_value = fma(
        small,
        divf(p, q, rounding::NearestEven, ftz::Disabled),
        small,
        rounding::NearestEven,
        ftz::Disabled,
    );
    // Scaling by eight is exact here, including subnormal inputs.
    let scaled = mulf(broadcast_scalar(8f64, shape![128]), small, rounding::NearestEven, ftz::Disabled);
    let tiny_sum = fma(
        broadcast_scalar(1.02703333676410069053e+00f64, shape![128]),
        small,
        scaled,
        rounding::NearestEven,
        ftz::Disabled,
    );
    let tiny = mulf(broadcast_scalar(0.125f64, shape![128]), tiny_sum, rounding::NearestEven, ftz::Disabled);
    let small_value = select(lt_tile(region, broadcast_scalar(0x3e300000u64, shape![128])), tiny, small_value);
    let middle_mask = ge_tile(region, broadcast_scalar(0x3feb0000u64, shape![128])) & lt_tile(region, broadcast_scalar(0x3ff40000u64, shape![128]));
    let middle = subf(select(middle_mask, magnitude, one), one, rounding::NearestEven, ftz::Disabled);
    let mut p = broadcast_scalar(-2.16637559486879084300e-03f64, shape![128]);
    p = fma(
        middle,
        p,
        broadcast_scalar(3.54783043256182359371e-02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        middle,
        p,
        broadcast_scalar(-1.10894694282396677476e-01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        middle,
        p,
        broadcast_scalar(3.18346619901161753674e-01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        middle,
        p,
        broadcast_scalar(-3.72207876035701323847e-01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        middle,
        p,
        broadcast_scalar(4.14856118683748331666e-01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    p = fma(
        middle,
        p,
        broadcast_scalar(-2.36211856075265944077e-03f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let mut q = broadcast_scalar(1.19844998467991074170e-02f64, shape![128]);
    q = fma(
        middle,
        q,
        broadcast_scalar(1.36370839120290507362e-02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    q = fma(
        middle,
        q,
        broadcast_scalar(1.26171219808761642112e-01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    q = fma(
        middle,
        q,
        broadcast_scalar(7.18286544141962662868e-02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    q = fma(
        middle,
        q,
        broadcast_scalar(5.40397917702171048937e-01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    q = fma(
        middle,
        q,
        broadcast_scalar(1.06420880400844228286e-01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let q = fma(middle, q, one, rounding::NearestEven, ftz::Disabled);
    let middle_value = subf(
        one,
        subf(
            subf(
                one,
                broadcast_scalar(8.45062911510467529297e-01f64, shape![128]),
                rounding::NearestEven,
                ftz::Disabled,
            ),
            divf(p, q, rounding::NearestEven, ftz::Disabled),
            rounding::NearestEven,
            ftz::Disabled,
        ),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let tail_mask = ge_tile(region, broadcast_scalar(0x3ff40000u64, shape![128])) & lt_tile(region, broadcast_scalar(0x40180000u64, shape![128]));
    let tail = select(tail_mask, magnitude, broadcast_scalar(1.25f64, shape![128]));
    let s = divf(
        one,
        mulf(tail, tail, rounding::NearestEven, ftz::Disabled),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let mut ra = broadcast_scalar(-9.81432934416914548592e+00f64, shape![128]);
    ra = fma(
        s,
        ra,
        broadcast_scalar(-8.12874355063065934246e+01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    ra = fma(
        s,
        ra,
        broadcast_scalar(-1.84605092906711035994e+02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    ra = fma(
        s,
        ra,
        broadcast_scalar(-1.62396669462573470355e+02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    ra = fma(
        s,
        ra,
        broadcast_scalar(-6.23753324503260060396e+01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    ra = fma(
        s,
        ra,
        broadcast_scalar(-1.05586262253232909814e+01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    ra = fma(
        s,
        ra,
        broadcast_scalar(-6.93858572707181764372e-01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    ra = fma(
        s,
        ra,
        broadcast_scalar(-9.86494403484714822705e-03f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let mut qa = broadcast_scalar(-6.04244152148580987438e-02f64, shape![128]);
    qa = fma(
        s,
        qa,
        broadcast_scalar(6.57024977031928170135e+00f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    qa = fma(
        s,
        qa,
        broadcast_scalar(1.08635005541779435134e+02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    qa = fma(
        s,
        qa,
        broadcast_scalar(4.29008140027567833386e+02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    qa = fma(
        s,
        qa,
        broadcast_scalar(6.45387271733267880336e+02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    qa = fma(
        s,
        qa,
        broadcast_scalar(4.34565877475229228821e+02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    qa = fma(
        s,
        qa,
        broadcast_scalar(1.37657754143519042600e+02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    qa = fma(
        s,
        qa,
        broadcast_scalar(1.96512716674392571292e+01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let qa = fma(s, qa, one, rounding::NearestEven, ftz::Disabled);
    let mut rb = broadcast_scalar(-4.83519191608651397019e+02f64, shape![128]);
    rb = fma(
        s,
        rb,
        broadcast_scalar(-1.02509513161107724954e+03f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    rb = fma(
        s,
        rb,
        broadcast_scalar(-6.37566443368389627722e+02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    rb = fma(
        s,
        rb,
        broadcast_scalar(-1.60636384855821916062e+02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    rb = fma(
        s,
        rb,
        broadcast_scalar(-1.77579549177547519889e+01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    rb = fma(
        s,
        rb,
        broadcast_scalar(-7.99283237680523006574e-01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    rb = fma(
        s,
        rb,
        broadcast_scalar(-9.86494292470009928597e-03f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let mut qb = broadcast_scalar(-2.24409524465858183362e+01f64, shape![128]);
    qb = fma(
        s,
        qb,
        broadcast_scalar(4.74528541206955367215e+02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    qb = fma(
        s,
        qb,
        broadcast_scalar(2.55305040643316442583e+03f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    qb = fma(
        s,
        qb,
        broadcast_scalar(3.19985821950859553908e+03f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    qb = fma(
        s,
        qb,
        broadcast_scalar(1.53672958608443695994e+03f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    qb = fma(
        s,
        qb,
        broadcast_scalar(3.25792512996573918826e+02f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    qb = fma(
        s,
        qb,
        broadcast_scalar(3.03380607434824582924e+01f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let qb = fma(s, qb, one, rounding::NearestEven, ftz::Disabled);
    let near = lt_tile(region, broadcast_scalar(0x4006db6du64, shape![128]));
    let ratio = divf(select(near, ra, rb), select(near, qa, qb), rounding::NearestEven, ftz::Disabled);
    let tail_bits: Tile<u64, { [128] }> = bitcast(tail);
    let truncated: Tile<f64, { [128] }> = bitcast((tail_bits >> broadcast_scalar(32u64, shape![128])) << broadcast_scalar(32u64, shape![128]));
    let first_arg = fma(
        subf(zero, truncated, rounding::NearestEven, ftz::Disabled),
        truncated,
        broadcast_scalar(-0.5625f64, shape![128]),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let second_arg = fma(
        subf(truncated, tail, rounding::NearestEven, ftz::Disabled),
        addf(truncated, tail, rounding::NearestEven, ftz::Disabled),
        ratio,
        rounding::NearestEven,
        ftz::Disabled,
    );
    let first = unsafe { exp_with_rounding(first_arg, rounding::Full) };
    let second = unsafe { exp_with_rounding(second_arg, rounding::Full) };
    let tail_value = subf(
        one,
        divf(
            mulf(first, second, rounding::NearestEven, ftz::Disabled),
            tail,
            rounding::NearestEven,
            ftz::Disabled,
        ),
        rounding::NearestEven,
        ftz::Disabled,
    );
    let value = select(
        lt_tile(region, broadcast_scalar(0x3feb0000u64, shape![128])),
        small_value,
        select(middle_mask, middle_value, select(tail_mask, tail_value, one)),
    );
    let value: Tile<u64, { [128] }> = bitcast(value);
    // Build wide masks with TileIR shifts; the pinned frontend parses scalar integers as 32-bit.
    let quiet_bit = broadcast_scalar(1u64, shape![128]) << broadcast_scalar(51u64, shape![128]);
    let infinity_bits = broadcast_scalar(0x7ff00000u64, shape![128]) << broadcast_scalar(32u64, shape![128]);
    let nan = bits | quiet_bit;
    select(gt_tile(magnitude_bits, infinity_bits), nan, value | sign)
}
