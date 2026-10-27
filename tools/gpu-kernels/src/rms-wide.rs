// A wide tile keeps each 4096-value row in registers through normalization.
#[cutile::entry()]
unsafe fn rms_norm_4096(input: *const f32, weights: *const f32, output: *mut f32, _rows: u32, width: u32, epsilon: u32, offset: u32) {
    let col = lanes::<4096>();
    let mask = lt_tile(col, broadcast_scalar(width, shape![4096]));
    let index = col + broadcast_scalar(block_id() * width, shape![4096]);
    let x = unsafe { read_tile(input, index, mask, 0f32) };
    let square: Tile<f32, { [] }> = reduce_sum(x * x, 0i32);
    let sum: f32 = tile_to_scalar(square);
    let factor = 1f32 / root(sum / float_u32(width) + bits_f32(epsilon));
    let w = unsafe { read_tile(weights, col, mask, 0f32) };
    let value = x * broadcast_scalar(factor, shape![4096]) * (w + broadcast_scalar(bits_f32(offset), shape![4096]));
    unsafe { write_tile(output, index, value, mask) };
}
