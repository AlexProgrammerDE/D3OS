// Exact rational coordinates avoid losing a nearest-neighbor boundary in FP32.
fn interp_coordinate(index: Tile<u32, { [128] }>, scale: u32, bias: u32, divisor: u32) -> (Tile<i32, { [128] }>, Tile<f32, { [128] }>) {
    let j: Tile<i64, { [128] }> = exti(index);
    let a: Tile<i64, { [128] }> = exti(broadcast_scalar(scale, shape![128]));
    let b: Tile<i64, { [128] }> = exti(broadcast_scalar(signed(bias), shape![128]));
    let d: Tile<i64, { [128] }> = exti(broadcast_scalar(divisor, shape![128]));
    let numerator = j * a + b;
    let q = numerator / d;
    let r = numerator - q * d;
    let negative = lt_tile(r, broadcast_scalar(0i64, shape![128]));
    let whole = q - select(negative, broadcast_scalar(1i64, shape![128]), broadcast_scalar(0i64, shape![128]));
    let remainder = r + select(negative, d, broadcast_scalar(0i64, shape![128]));
    let whole: Tile<i32, { [128] }> = trunci(whole, overflow::NoSignedWrap);
    let numerator: Tile<f32, { [128] }> = itof(remainder, rounding::NearestEven);
    let den: Tile<f32, { [128] }> = itof(d, rounding::NearestEven);
    (whole, divf(numerator, den, rounding::NearestEven, ftz::Disabled))
}
fn interp_weight(mode: u32, distance: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
    let x = absf(distance);
    let zero = broadcast_scalar(0f32, shape![128]);
    let one = broadcast_scalar(1f32, shape![128]);
    if mode < 2u32 {
        one
    } else if mode == 2u32 {
        max_tile(one - x, zero)
    } else if mode == 3u32 {
        let first = fma(
            broadcast_scalar(1.25f32, shape![128]) * x - broadcast_scalar(2.25f32, shape![128]),
            x * x,
            one,
            rounding::NearestEven,
            ftz::Disabled,
        );
        let second = fma(
            fma(
                broadcast_scalar(-0.75f32, shape![128]),
                x,
                broadcast_scalar(3.75f32, shape![128]),
                rounding::NearestEven,
                ftz::Disabled,
            ),
            x,
            broadcast_scalar(-6f32, shape![128]),
            rounding::NearestEven,
            ftz::Disabled,
        );
        let second = fma(second, x, broadcast_scalar(3f32, shape![128]), rounding::NearestEven, ftz::Disabled);
        select(le_tile(x, one), first, select(lt_tile(x, broadcast_scalar(2f32, shape![128])), second, zero))
    } else {
        let u = broadcast_scalar(1.0471975511965976f32, shape![128]) * distance;
        // Here |u| <= pi. Reflect into [0, pi/2] before a degree-13 sine
        // series. This avoids the general sine's large-angle scratch path.
        let angle = absf(u);
        let z = max_tile(min_tile(angle, broadcast_scalar(3.141592653589793f32, shape![128]) - angle), zero);
        let z2 = z * z;
        let mut polynomial = broadcast_scalar(1.6059043836821613e-10f32, shape![128]);
        polynomial = fma(
            polynomial,
            z2,
            broadcast_scalar(-2.505210838544172e-8f32, shape![128]),
            rounding::NearestEven,
            ftz::Disabled,
        );
        polynomial = fma(
            polynomial,
            z2,
            broadcast_scalar(2.7557319223985893e-6f32, shape![128]),
            rounding::NearestEven,
            ftz::Disabled,
        );
        polynomial = fma(
            polynomial,
            z2,
            broadcast_scalar(-0.0001984126984126984f32, shape![128]),
            rounding::NearestEven,
            ftz::Disabled,
        );
        polynomial = fma(
            polynomial,
            z2,
            broadcast_scalar(0.008333333333333333f32, shape![128]),
            rounding::NearestEven,
            ftz::Disabled,
        );
        polynomial = fma(
            polynomial,
            z2,
            broadcast_scalar(-0.16666666666666666f32, shape![128]),
            rounding::NearestEven,
            ftz::Disabled,
        );
        let sine = fma(z * z2, polynomial, z, rounding::NearestEven, ftz::Disabled);
        let square = sine * sine;
        let numerator = square
            * fma(
                broadcast_scalar(-4f32, shape![128]),
                square,
                broadcast_scalar(3f32, shape![128]),
                rounding::NearestEven,
                ftz::Disabled,
            );
        let tiny = lt_tile(x, broadcast_scalar(1e-7f32, shape![128]));
        let den = select(tiny, one, broadcast_scalar(3f32, shape![128]) * u * u);
        select(
            tiny,
            one,
            select(
                lt_tile(x, broadcast_scalar(3f32, shape![128])),
                divf(numerator, den, rounding::NearestEven, ftz::Disabled),
                zero,
            ),
        )
    }
}
fn interp_candidate(base: Tile<i32, { [128] }>, tap: u32, radius: u32, limit: u32) -> (Tile<i32, { [128] }>, Tile<bool, { [128] }>) {
    let base: Tile<i64, { [128] }> = exti(base);
    let step: Tile<i64, { [128] }> = exti(broadcast_scalar(signed(tap) - signed(radius), shape![128]));
    let raw = base + step;
    let high: Tile<i64, { [128] }> = exti(broadcast_scalar(limit, shape![128]));
    let valid = ge_tile(raw, broadcast_scalar(0i64, shape![128])) & lt_tile(raw, high);
    let clamped = min_tile(max_tile(raw, broadcast_scalar(0i64, shape![128])), high - broadcast_scalar(1i64, shape![128]));
    (trunci(clamped, overflow::NoSignedWrap), valid)
}
fn interp_distance(phase: Tile<f32, { [128] }>, tap: u32, radius: u32) -> Tile<f32, { [128] }> {
    let k: Tile<f32, { [128] }> = itof(broadcast_scalar(signed(tap) - signed(radius), shape![128]), rounding::NearestEven);
    k - phase
}
fn interp_bound(index: Tile<u32, { [128] }>, extra: i32, scale: u32, bias: u32, divisor: u32, outputs: u32, limit: u32) -> Tile<u32, { [128] }> {
    let k: Tile<i64, { [128] }> = exti(index);
    let shift: Tile<i64, { [128] }> = exti(broadcast_scalar(extra, shape![128]));
    let a: Tile<i64, { [128] }> = exti(broadcast_scalar(if scale == 0u32 { 1u32 } else { scale }, shape![128]));
    let b: Tile<i64, { [128] }> = exti(broadcast_scalar(signed(bias), shape![128]));
    let d: Tile<i64, { [128] }> = exti(broadcast_scalar(divisor, shape![128]));
    let input: Tile<i64, { [128] }> = exti(broadcast_scalar(limit, shape![128]));
    let n = min_tile(k + shift, input) * d - b;
    let q = n / a;
    let r = n - q * a;
    let bound = q + select(
        gt_tile(r, broadcast_scalar(0i64, shape![128])),
        broadcast_scalar(1i64, shape![128]),
        broadcast_scalar(0i64, shape![128]),
    );
    let high: Tile<i64, { [128] }> = exti(broadcast_scalar(outputs, shape![128]));
    let clamped = min_tile(max_tile(bound, broadcast_scalar(0i64, shape![128])), high);
    trunci(clamped, overflow::NoUnsignedWrap)
}
fn interp_coefficient(
    index: Tile<u32, { [128] }>, output: Tile<u32, { [128] }>, limit: u32, mode: u32, taps: u32, radius: u32, scale: u32, bias: u32, divisor: u32,
) -> Tile<f32, { [128] }> {
    let (base, phase) = interp_coordinate(output, scale, bias, divisor);
    let k: Tile<i32, { [128] }> = bitcast(index);
    let mut weight = broadcast_scalar(0f32, shape![128]);
    let mut normalizer = broadcast_scalar(0f32, shape![128]);
    for tap in 0..taps {
        let (at, valid) = interp_candidate(base, tap, radius, limit);
        let w = interp_weight(mode, interp_distance(phase, tap, radius));
        let w = if mode == 4u32 {
            select(valid, w, broadcast_scalar(0f32, shape![128]))
        } else {
            w
        };
        weight = addf(
            weight,
            select(eq_tile(at, k), w, broadcast_scalar(0f32, shape![128])),
            rounding::NearestEven,
            ftz::Disabled,
        );
        normalizer = addf(normalizer, w, rounding::NearestEven, ftz::Disabled);
    }
    if mode == 4u32 {
        select(
            gt_tile(absf(normalizer), broadcast_scalar(1e-7f32, shape![128])),
            divf(weight, normalizer, rounding::NearestEven, ftz::Disabled),
            broadcast_scalar(0f32, shape![128]),
        )
    } else {
        weight
    }
}
