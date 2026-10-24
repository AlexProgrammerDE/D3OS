// Philox4x32-10 uses a 64-bit key and a 128-bit counter. The low 64 bits
// select a reserved four-word block; the high bits distinguish rejection retries.
fn burn_philox(low: Tile<u32, { [128] }>, high: Tile<u32, { [128] }>, attempt: Tile<u64, { [128] }>, seed_low: u32, seed_high: u32)
    -> (Tile<u32, { [128] }>, Tile<u32, { [128] }>, Tile<u32, { [128] }>, Tile<u32, { [128] }>) {
    let mut a = low;
    let mut b = high;
    let word_mask = (broadcast_scalar(1u64, shape![128]) << broadcast_scalar(32u64, shape![128])) - broadcast_scalar(1u64, shape![128]);
    let mut c: Tile<u32, { [128] }> = trunci(attempt & word_mask, overflow::NoUnsignedWrap);
    let mut d: Tile<u32, { [128] }> = trunci(attempt >> broadcast_scalar(32u64, shape![128]), overflow::NoUnsignedWrap);
    let mut key_a = broadcast_scalar(seed_low, shape![128]);
    let mut key_b = broadcast_scalar(seed_high, shape![128]);
    let m0 = broadcast_scalar(unsigned(-766435501i32), shape![128]);
    let m1 = broadcast_scalar(unsigned(-845247145i32), shape![128]);
    let w0 = broadcast_scalar(unsigned(-1640531527i32), shape![128]);
    let w1 = broadcast_scalar(unsigned(-1150833019i32), shape![128]);
    let mut round = 0u32;
    while round < 10u32 {
        let next_a = mulhii(m1, c) ^ b ^ key_a;
        let next_b = m1 * c;
        let next_c = mulhii(m0, a) ^ d ^ key_b;
        let next_d = m0 * a;
        a = next_a; b = next_b; c = next_c; d = next_d;
        key_a = key_a + w0; key_b = key_b + w1;
        round = round + 1u32;
    }
    (a, b, c, d)
}

fn burn_random_lane(a: Tile<u32, { [128] }>, b: Tile<u32, { [128] }>, c: Tile<u32, { [128] }>, d: Tile<u32, { [128] }>, lane: Tile<u32, { [128] }>) -> Tile<u32, { [128] }> {
    select(eq_tile(lane, broadcast_scalar(0u32, shape![128])), a,
        select(eq_tile(lane, broadcast_scalar(1u32, shape![128])), b,
            select(eq_tile(lane, broadcast_scalar(2u32, shape![128])), c, d)))
}

fn burn_random_unit(word: Tile<u32, { [128] }>, open: bool) -> Tile<f32, { [128] }> {
    let bits = (word >> broadcast_scalar(8u32, shape![128])) + broadcast_scalar(if open { 1u32 } else { 0u32 }, shape![128]);
    let value: Tile<f32, { [128] }> = itof(bits, rounding::NearestEven);
    mulf(value, broadcast_scalar(0.000000059604644775390625f32, shape![128]), rounding::NearestEven, ftz::Disabled)
}
