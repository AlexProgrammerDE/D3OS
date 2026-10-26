// Round FP32 to the nearest integer, with ties to even. Work in the IEEE
// representation to preserve zeros, large integral values, and NaN payloads.
fn burn_round_even(bits: Tile<u32, { [128] }>) -> Tile<u32, { [128] }> {
    let one = broadcast_scalar(1u32, shape![128]);
    let sign = bits & broadcast_scalar(unsigned(-1i32) << 31u32, shape![128]);
    let magnitude = bits & broadcast_scalar(0x7fffffffu32, shape![128]);
    let exponent = magnitude >> broadcast_scalar(23u32, shape![128]);
    let bounded = min_tile(max_tile(exponent, broadcast_scalar(127u32, shape![128])), broadcast_scalar(149u32, shape![128]));
    let shift = broadcast_scalar(150u32, shape![128]) - bounded;
    let increment = one << shift;
    let mask = increment - one;
    let half = increment >> one;
    let fraction = magnitude & mask;
    let whole = magnitude & (mask ^ broadcast_scalar(unsigned(-1i32), shape![128]));
    let odd = eq_tile((whole >> shift) & one, one);
    let up = gt_tile(fraction, half) | (eq_tile(fraction, half) & odd);
    let rounded = whole + select(up, increment, broadcast_scalar(0u32, shape![128]));
    let small = select(
        gt_tile(magnitude, broadcast_scalar(0x3f000000u32, shape![128])),
        broadcast_scalar(0x3f800000u32, shape![128]),
        broadcast_scalar(0u32, shape![128]),
    );
    let rounded = select(lt_tile(magnitude, broadcast_scalar(0x3f800000u32, shape![128])), small, rounded) | sign;
    select(lt_tile(magnitude, broadcast_scalar(0x4b000000u32, shape![128])), rounded, bits)
}

#[cutile::entry()]
unsafe fn burn_fill(output: *mut u32, count: u32, value: u32) {
    let index = indices();
    unsafe {
        write_tile(
            output,
            index,
            broadcast_scalar(value, shape![128]),
            lt_tile(index, broadcast_scalar(count, shape![128])),
        )
    };
}
