// Tile IR remf supplies the truncated residue. Burn uses floor modulo, so
// adjust only a nonzero residue whose sign differs from the divisor.
fn burn_remainder_bits(left: Tile<u32, { [128] }>, right: Tile<u32, { [128] }>) -> Tile<u32, { [128] }> {
    let x: Tile<f32, { [128] }> = bitcast(left);
    let y: Tile<f32, { [128] }> = bitcast(right);
    let remainder = remf(x, y);
    let bits: Tile<u32, { [128] }> = bitcast(remainder);
    let nonzero = ne_tile(bits & broadcast_scalar(0x7fffffffu32, shape![128]), broadcast_scalar(0u32, shape![128]));
    let different_sign = ne_tile(
        (bits ^ right) & broadcast_scalar(unsigned(-1i32) << 31u32, shape![128]),
        broadcast_scalar(0u32, shape![128]),
    );
    let corrected = addf(remainder, y, rounding::NearestEven, ftz::Disabled);
    select(nonzero & different_sign, bitcast(corrected), bits)
}
