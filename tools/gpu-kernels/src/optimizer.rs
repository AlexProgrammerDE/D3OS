// Explicit rounding preserves the intermediate FP32 results of Burn's
// separate tensor operations, including subnormals and the epsilon placement.
fn adam_mul(a: Tile<f32, { [128] }>, b: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
    mulf(a, b, rounding::NearestEven, ftz::Disabled)
}
fn adam_add(a: Tile<f32, { [128] }>, b: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
    addf(a, b, rounding::NearestEven, ftz::Disabled)
}

#[cutile::entry()]
unsafe fn burn_adam(
    parameter: *const f32, gradient: *const f32, moment1: *const f32, moment2: *const f32, maximum: *const f32, output: *mut f32,
    count: u32, flags: u32, beta1: u32, factor1: u32, beta2: u32, factor2: u32, combined: u32, epsilon: u32, rate: u32,
) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(count, shape![128]));
    let grad = unsafe { read_tile(gradient, i, mask, 0f32) };
    let value = unsafe { read_tile(parameter, i, mask, 0f32) };
    let mut first = adam_mul(grad, broadcast_scalar(bits_f32(factor1), shape![128]));
    let mut second = adam_mul(adam_mul(grad, grad), broadcast_scalar(bits_f32(factor2), shape![128]));
    if flags & 1u32 != 0u32 {
        let old1 = unsafe { read_tile(moment1, i, mask, 0f32) };
        let old2 = unsafe { read_tile(moment2, i, mask, 0f32) };
        first = adam_add(adam_mul(old1, broadcast_scalar(bits_f32(beta1), shape![128])), first);
        second = adam_add(adam_mul(old2, broadcast_scalar(bits_f32(beta2), shape![128])), second);
    }
    let variance = if flags & 4u32 != 0u32 {
        let old_maximum = unsafe { read_tile(maximum, i, mask, 0f32) };
        select(lt_tile(old_maximum, second), second, old_maximum)
    } else {
        second
    };
    let denominator = adam_add(sqrt(variance, rounding::NearestEven, ftz::Disabled), broadcast_scalar(bits_f32(epsilon), shape![128]));
    let numerator = adam_mul(first, broadcast_scalar(bits_f32(combined), shape![128]));
    let adjusted = divf(numerator, denominator, rounding::NearestEven, ftz::Disabled);
    let delta = adam_mul(adjusted, broadcast_scalar(bits_f32(rate), shape![128]));
    let next = subf(value, delta, rounding::NearestEven, ftz::Disabled);
    unsafe {
        write_tile(output, i, next, mask);
        write_tile(output, i + broadcast_scalar(count, shape![128]), first, mask);
        write_tile(output, i + broadcast_scalar(2u32 * count, shape![128]), second, mask);
        if flags & 2u32 != 0u32 {
            write_tile(output, i + broadcast_scalar(3u32 * count, shape![128]), variance, mask);
        }
    }
}
