// Hyperbolic range reductions follow Rust libm 0.2.16 (MIT).
// Circular inverses use Tile IR atan2 and a factored radicand near +/-1.
/*
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
fn burn_asin_bits(bits: Tile<u32, { [128] }>) -> Tile<u32, { [128] }> {
    let x: Tile<f32, { [128] }> = bitcast(bits);
    let one = broadcast_scalar(1f32, shape![128]);
    let a = subf(one, x, rounding::NearestEven, ftz::Disabled);
    let b = addf(one, x, rounding::NearestEven, ftz::Disabled);
    let root = sqrt(mulf(a, b, rounding::NearestEven, ftz::Disabled), rounding::NearestEven, ftz::Disabled);
    bitcast(atan2(x, root))
}
fn burn_acos_bits(bits: Tile<u32, { [128] }>) -> Tile<u32, { [128] }> {
    let x: Tile<f32, { [128] }> = bitcast(bits);
    let one = broadcast_scalar(1f32, shape![128]);
    let a = subf(one, x, rounding::NearestEven, ftz::Disabled);
    let b = addf(one, x, rounding::NearestEven, ftz::Disabled);
    let root = sqrt(mulf(a, b, rounding::NearestEven, ftz::Disabled), rounding::NearestEven, ftz::Disabled);
    bitcast(atan2(root, x))
}
fn burn_atan_bits(bits: Tile<u32, { [128] }>) -> Tile<u32, { [128] }> {
    let x: Tile<f32, { [128] }> = bitcast(bits);
    bitcast(atan2(x, broadcast_scalar(1f32, shape![128])))
}
fn burn_acosh_bits(bits: Tile<u32, { [128] }>) -> Tile<u32, { [128] }> {
    let x: Tile<f32, { [128] }> = bitcast(bits);
    let one = broadcast_scalar(1f32, shape![128]);
    let two = broadcast_scalar(2f32, shape![128]);
    let small = select(ge_tile(x, one) & lt_tile(x, two), x, one);
    let t = small - one;
    let z = t + sqrt(t * t + two * t, rounding::NearestEven, ftz::Disabled);
    let low: Tile<f32, { [128] }> = bitcast(burn_log1p_bits(bitcast(z)));
    let medium = select(ge_tile(x, two) & lt_tile(x, broadcast_scalar(4096f32, shape![128])), x, two);
    let root = sqrt(medium * medium - one, rounding::NearestEven, ftz::Disabled);
    let mid = log(two * medium - divf(one, medium + root, rounding::NearestEven, ftz::Disabled));
    let high = log(x) + broadcast_scalar(bits_f32(0x3f317218u32), shape![128]);
    let value = select(lt_tile(x, two), low, select(lt_tile(x, broadcast_scalar(4096f32, shape![128])), mid, high));
    let invalid: Tile<f32, { [128] }> = bitcast(broadcast_scalar(0x7fc00000u32, shape![128]));
    bitcast(select(ge_tile(x, one), value, invalid))
}
fn burn_asinh_bits(bits: Tile<u32, { [128] }>) -> Tile<u32, { [128] }> {
    let magnitude = bits & broadcast_scalar(0x7fffffffu32, shape![128]);
    let x: Tile<f32, { [128] }> = bitcast(magnitude);
    let zero = broadcast_scalar(0f32, shape![128]);
    let one = broadcast_scalar(1f32, shape![128]);
    let two = broadcast_scalar(2f32, shape![128]);
    let small = select(lt_tile(x, two), x, zero);
    let z = small
        + divf(
            small * small,
            sqrt(small * small + one, rounding::NearestEven, ftz::Disabled) + one,
            rounding::NearestEven,
            ftz::Disabled,
        );
    let low: Tile<f32, { [128] }> = bitcast(burn_log1p_bits(bitcast(z)));
    let medium = select(ge_tile(x, two) & lt_tile(x, broadcast_scalar(4096f32, shape![128])), x, two);
    let mid = log(two * medium
        + divf(
            one,
            sqrt(medium * medium + one, rounding::NearestEven, ftz::Disabled) + medium,
            rounding::NearestEven,
            ftz::Disabled,
        ));
    let high = log(x) + broadcast_scalar(bits_f32(0x3f317218u32), shape![128]);
    let result = select(lt_tile(x, two), low, select(lt_tile(x, broadcast_scalar(4096f32, shape![128])), mid, high));
    let encoded: Tile<u32, { [128] }> = bitcast(result);
    let signed = encoded | (bits & broadcast_scalar(unsigned(-1i32) << 31u32, shape![128]));
    select(lt_tile(magnitude, broadcast_scalar(0x39800000u32, shape![128])), bits, signed)
}
fn burn_atanh_bits(bits: Tile<u32, { [128] }>) -> Tile<u32, { [128] }> {
    let magnitude = bits & broadcast_scalar(0x7fffffffu32, shape![128]);
    let x: Tile<f32, { [128] }> = bitcast(magnitude);
    let zero = broadcast_scalar(0f32, shape![128]);
    let one = broadcast_scalar(1f32, shape![128]);
    let two = broadcast_scalar(2f32, shape![128]);
    let half = broadcast_scalar(0.5f32, shape![128]);
    let small = select(lt_tile(x, half), x, zero);
    let z = two * small + divf(two * small * small, one - small, rounding::NearestEven, ftz::Disabled);
    let low: Tile<f32, { [128] }> = bitcast(burn_log1p_bits(bitcast(z)));
    let big = select(ge_tile(x, half) & le_tile(x, one), x, half);
    let z = two * divf(big, one - big, rounding::NearestEven, ftz::Disabled);
    let high: Tile<f32, { [128] }> = bitcast(burn_log1p_bits(bitcast(z)));
    let value = half * select(lt_tile(x, half), low, high);
    let encoded: Tile<u32, { [128] }> = bitcast(value);
    let signed = encoded | (bits & broadcast_scalar(unsigned(-1i32) << 31u32, shape![128]));
    let result = select(
        gt_tile(magnitude, broadcast_scalar(0x3f800000u32, shape![128])),
        broadcast_scalar(0x7fc00000u32, shape![128]),
        signed,
    );
    select(lt_tile(magnitude, broadcast_scalar(0x39800000u32, shape![128])), bits, result)
}
