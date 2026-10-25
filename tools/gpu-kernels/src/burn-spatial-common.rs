fn spatial_coordinate(
    position: Tile<u32, { [128] }>, tap: Tile<u32, { [128] }>, stride: u32, dilation: u32, padding: u32, limit: u32,
) -> (Tile<i32, { [128] }>, Tile<bool, { [128] }>) {
    let position: Tile<i64, { [128] }> = exti(position);
    let tap: Tile<i64, { [128] }> = exti(tap);
    let s: Tile<i64, { [128] }> = exti(broadcast_scalar(stride, shape![128]));
    let d: Tile<i64, { [128] }> = exti(broadcast_scalar(dilation, shape![128]));
    let p: Tile<i64, { [128] }> = exti(broadcast_scalar(padding, shape![128]));
    let high: Tile<i64, { [128] }> = exti(broadcast_scalar(limit, shape![128]));
    let coordinate = position * s + tap * d - p;
    let valid = ge_tile(coordinate, broadcast_scalar(0i64, shape![128])) & lt_tile(coordinate, high);
    let clamped = min_tile(
        max_tile(coordinate, broadcast_scalar(0i64, shape![128])),
        high - broadcast_scalar(1i64, shape![128]),
    );
    (trunci(clamped, overflow::NoSignedWrap), valid)
}
fn spatial_inverse(
    position: Tile<u32, { [128] }>, tap: u32, stride: u32, dilation: u32, padding: u32, limit: u32,
) -> (Tile<u32, { [128] }>, Tile<bool, { [128] }>) {
    let position: Tile<i64, { [128] }> = exti(position);
    let k: Tile<i64, { [128] }> = exti(broadcast_scalar(tap, shape![128]));
    let s: Tile<i64, { [128] }> = exti(broadcast_scalar(stride, shape![128]));
    let d: Tile<i64, { [128] }> = exti(broadcast_scalar(dilation, shape![128]));
    let p: Tile<i64, { [128] }> = exti(broadcast_scalar(padding, shape![128]));
    let high: Tile<i64, { [128] }> = exti(broadcast_scalar(limit, shape![128]));
    let numerator = position + p - k * d;
    let coordinate = numerator / s;
    let valid = ge_tile(numerator, broadcast_scalar(0i64, shape![128]))
        & eq_tile(numerator - coordinate * s, broadcast_scalar(0i64, shape![128]))
        & lt_tile(coordinate, high);
    let clamped = min_tile(
        max_tile(coordinate, broadcast_scalar(0i64, shape![128])),
        high - broadcast_scalar(1i64, shape![128]),
    );
    (trunci(clamped, overflow::NoUnsignedWrap), valid)
}
