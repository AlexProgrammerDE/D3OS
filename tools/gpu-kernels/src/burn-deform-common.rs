// Coordinates are clamped before conversion/addressing, including for NaNs.
fn deform_axis(
    position: Tile<u32, { [128] }>, tap: Tile<u32, { [128] }>, stride: u32, dilation: u32, padding: u32, offset: Tile<f32, { [128] }>,
) -> Tile<f32, { [128] }> {
    let position: Tile<i64, { [128] }> = exti(position);
    let tap: Tile<i64, { [128] }> = exti(tap);
    let s: Tile<i64, { [128] }> = exti(broadcast_scalar(stride, shape![128]));
    let d: Tile<i64, { [128] }> = exti(broadcast_scalar(dilation, shape![128]));
    let p: Tile<i64, { [128] }> = exti(broadcast_scalar(padding, shape![128]));
    let base: Tile<f32, { [128] }> = itof(position * s + tap * d - p, rounding::NearestEven);
    addf(base, offset, rounding::NearestEven, ftz::Disabled)
}
fn deform_location(
    y: Tile<f32, { [128] }>, x: Tile<f32, { [128] }>, height: u32, width: u32,
) -> (
    Tile<i32, { [128] }>,
    Tile<i32, { [128] }>,
    Tile<f32, { [128] }>,
    Tile<f32, { [128] }>,
    Tile<bool, { [128] }>,
) {
    let valid = ge_tile(y, broadcast_scalar(-1f32, shape![128]))
        & lt_tile(y, broadcast_scalar(float_u32(height), shape![128]))
        & ge_tile(x, broadcast_scalar(-1f32, shape![128]))
        & lt_tile(x, broadcast_scalar(float_u32(width), shape![128]));
    let y = select(valid, y, broadcast_scalar(0f32, shape![128]));
    let x = select(valid, x, broadcast_scalar(0f32, shape![128]));
    let fy = floor(y);
    let fx = floor(x);
    let iy: Tile<i32, { [128] }> = ftoi(fy, rounding::NearestIntToZero);
    let ix: Tile<i32, { [128] }> = ftoi(fx, rounding::NearestIntToZero);
    (iy, ix, y - fy, x - fx, valid)
}
unsafe fn deform_pixel(
    input: *const f32, n: Tile<u32, { [128] }>, c: Tile<u32, { [128] }>, y: Tile<i32, { [128] }>, x: Tile<i32, { [128] }>, channels: u32, height: u32,
    width: u32, active: Tile<bool, { [128] }>,
) -> Tile<f32, { [128] }> {
    let valid = active
        & ge_tile(y, broadcast_scalar(0i32, shape![128]))
        & lt_tile(y, broadcast_scalar(signed(height), shape![128]))
        & ge_tile(x, broadcast_scalar(0i32, shape![128]))
        & lt_tile(x, broadcast_scalar(signed(width), shape![128]));
    let y = min_tile(
        max_tile(y, broadcast_scalar(0i32, shape![128])),
        broadcast_scalar(signed(height) - 1, shape![128]),
    );
    let x = min_tile(
        max_tile(x, broadcast_scalar(0i32, shape![128])),
        broadcast_scalar(signed(width) - 1, shape![128]),
    );
    let y: Tile<u32, { [128] }> = bitcast(y);
    let x: Tile<u32, { [128] }> = bitcast(x);
    let address = ((n * broadcast_scalar(channels, shape![128]) + c) * broadcast_scalar(height, shape![128]) + y) * broadcast_scalar(width, shape![128]) + x;
    unsafe { read_tile(input, address, valid, 0f32) }
}
unsafe fn deform_sample(
    input: *const f32, n: Tile<u32, { [128] }>, c: Tile<u32, { [128] }>, y: Tile<f32, { [128] }>, x: Tile<f32, { [128] }>, channels: u32, height: u32,
    width: u32, active: Tile<bool, { [128] }>,
) -> (Tile<f32, { [128] }>, Tile<f32, { [128] }>, Tile<f32, { [128] }>) {
    let (iy, ix, ly, lx, valid) = deform_location(y, x, height, width);
    let one = broadcast_scalar(1i32, shape![128]);
    let active = active & valid;
    let v00 = unsafe { deform_pixel(input, n, c, iy, ix, channels, height, width, active) };
    let v01 = unsafe { deform_pixel(input, n, c, iy, ix + one, channels, height, width, active) };
    let v10 = unsafe { deform_pixel(input, n, c, iy + one, ix, channels, height, width, active) };
    let v11 = unsafe { deform_pixel(input, n, c, iy + one, ix + one, channels, height, width, active) };
    let hy = broadcast_scalar(1f32, shape![128]) - ly;
    let hx = broadcast_scalar(1f32, shape![128]) - lx;
    let value = hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11;
    let dy = hx * (v10 - v00) + lx * (v11 - v01);
    let dx = hy * (v01 - v00) + ly * (v11 - v10);
    (value, dy, dx)
}
