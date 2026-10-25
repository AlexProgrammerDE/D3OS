// GPU scales retain raw storage codes, one per word. This makes strided
// scale views lossless, including NaN payloads imported from a model record.
fn burn_scale_decode(code: Tile<u32, { [128] }>, dtype: u32) -> Tile<f32, { [128] }> {
    if dtype == 0u32 {
        bitcast(code)
    } else if dtype == 1u32 {
        let magnitude = code & broadcast_scalar(0x7fffu32, shape![128]);
        let exponent = magnitude >> broadcast_scalar(10u32, shape![128]);
        let mantissa = magnitude & broadcast_scalar(1023u32, shape![128]);
        let normal = (magnitude << broadcast_scalar(13u32, shape![128])) + broadcast_scalar(112u32 << 23u32, shape![128]);
        let fraction: Tile<f32, { [128] }> = itof(mantissa, rounding::NearestEven);
        let subnormal: Tile<u32, { [128] }> = bitcast(fraction * broadcast_scalar(0.000000059604644775390625f32, shape![128]));
        let special = broadcast_scalar(0x7f800000u32, shape![128]) | (mantissa << broadcast_scalar(13u32, shape![128]));
        let value = select(eq_tile(exponent, broadcast_scalar(31u32, shape![128])), special,
            select(eq_tile(exponent, broadcast_scalar(0u32, shape![128])), subnormal, normal));
        bitcast(value | ((code & broadcast_scalar(0x8000u32, shape![128])) << broadcast_scalar(16u32, shape![128])))
    } else if dtype == 2u32 {
        bitcast(code << broadcast_scalar(16u32, shape![128]))
    } else if dtype == 3u32 {
        let exponent = code << broadcast_scalar(23u32, shape![128]);
        let value = select(eq_tile(code, broadcast_scalar(0u32, shape![128])), broadcast_scalar(0x00400000u32, shape![128]), exponent);
        bitcast(select(eq_tile(code, broadcast_scalar(255u32, shape![128])), broadcast_scalar(0x7fc00000u32, shape![128]), value))
    } else {
        burn_minifloat_decode(code, 2u32)
    }
}

fn burn_scale_code(value: Tile<f32, { [128] }>, dtype: u32) -> Tile<u32, { [128] }> {
    let bits: Tile<u32, { [128] }> = bitcast(value);
    if dtype == 0u32 {
        bits
    } else if dtype == 3u32 {
        let code = (bits + broadcast_scalar(0x007fffffu32, shape![128])) >> broadcast_scalar(23u32, shape![128]);
        let minimum: Tile<f32, { [128] }> = bitcast(broadcast_scalar(0x00400000u32, shape![128]));
        let maximum: Tile<f32, { [128] }> = bitcast(broadcast_scalar(0x7f000000u32, shape![128]));
        let code = select(le_tile(value, minimum), broadcast_scalar(0u32, shape![128]), code);
        let code = select(ge_tile(value, maximum), broadcast_scalar(254u32, shape![128]), code);
        select(eq_tile(value, value), code, broadcast_scalar(255u32, shape![128]))
    } else if dtype == 2u32 {
        let code = (bits + broadcast_scalar(65535u32, shape![128])) >> broadcast_scalar(16u32, shape![128]);
        let maximum: Tile<f32, { [128] }> = bitcast(broadcast_scalar(0x7f7f0000u32, shape![128]));
        select(ge_tile(value, maximum), broadcast_scalar(0x7f7fu32, shape![128]), code)
    } else {
        let shift = if dtype == 1u32 { 13u32 } else { 20u32 };
        let bias = if dtype == 1u32 { 112u32 << 10u32 } else { 120u32 << 3u32 };
        let min_normal = if dtype == 1u32 { 0.00006103515625f32 } else { 0.015625f32 };
        let step_inverse = if dtype == 1u32 { 16777216f32 } else { 512f32 };
        let small = select(lt_tile(value, broadcast_scalar(min_normal, shape![128])), value, broadcast_scalar(0f32, shape![128]));
        let steps: Tile<u32, { [128] }> = ftoi(ceil(small * broadcast_scalar(step_inverse, shape![128])), rounding::NearestIntToZero);
        let normal = ((bits + broadcast_scalar((1u32 << shift) - 1u32, shape![128])) >> broadcast_scalar(shift, shape![128])) - broadcast_scalar(bias, shape![128]);
        let code = select(lt_tile(value, broadcast_scalar(min_normal, shape![128])), steps, normal);
        let maximum = if dtype == 1u32 { 65504f32 } else { 448f32 };
        let max_code = if dtype == 1u32 { 0x7bffu32 } else { 126u32 };
        let code = select(ge_tile(value, broadcast_scalar(maximum, shape![128])), broadcast_scalar(max_code, shape![128]), code);
        let nan_code = if dtype == 1u32 { 0x7e00u32 } else { 127u32 };
        select(eq_tile(value, value), code, broadcast_scalar(nan_code, shape![128]))
    }
}

unsafe fn burn_quantization_scale(
    scales: *const u32, global: *const u32, address: Tile<u32, { [128] }>, mask: Tile<bool, { [128] }>, count: u32, flags: u32,
) -> Tile<f32, { [128] }> {
    let code = unsafe { read_tile(scales, address, mask & lt_tile(address, broadcast_scalar(count, shape![128])), 0u32) };
    let inner = burn_scale_decode(code, flags & 7u32);
    if flags & 256u32 != 0u32 {
        let word = unsafe { read_tile(global, broadcast_scalar(0u32, shape![128]), mask, 0u32) };
        let factor: Tile<f32, { [128] }> = bitcast(word);
        mulf(inner, factor, rounding::NearestEven, ftz::Disabled)
    } else {
        inner
    }
}
