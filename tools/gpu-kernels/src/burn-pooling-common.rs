// Adaptive window ratios use 64-bit products so valid large dimensions do not
// overflow the native 32-bit element address space during intermediate math.
fn pool_scaled(index: Tile<u32, { [128] }>, numerator: u32, denominator: u32, ceil: bool) -> Tile<u32, { [128] }> {
    let index: Tile<u64, { [128] }> = exti(index);
    let n: Tile<u32, { [128] }> = broadcast_scalar(numerator, shape![128]);
    let n: Tile<u64, { [128] }> = exti(n);
    let d: Tile<u32, { [128] }> = broadcast_scalar(denominator, shape![128]);
    let d: Tile<u64, { [128] }> = exti(d);
    let add = if ceil { d - broadcast_scalar(1u64, shape![128]) } else { broadcast_scalar(0u64, shape![128]) };
    let result = (index * n + add) / d;
    trunci(result, overflow::NoUnsignedWrap)
}

fn pool_width(index: Tile<u32, { [128] }>, input: u32, kernel: u32, stride: u32, before: u32, after: u32, include: bool) -> Tile<u32, { [128] }> {
    let start = index * broadcast_scalar(stride, shape![128]);
    let start: Tile<i32, { [128] }> = bitcast(start);
    let start = start - broadcast_scalar(signed(before), shape![128]);
    let end = start + broadcast_scalar(signed(kernel), shape![128]);
    let low = max_tile(start, broadcast_scalar(if include { 0i32 - signed(before) } else { 0i32 }, shape![128]));
    let high = min_tile(end, broadcast_scalar(signed(if include { input + after } else { input }), shape![128]));
    let length = max_tile(high - low, broadcast_scalar(0i32, shape![128]));
    bitcast(length)
}

fn pool_span(begin: Tile<u32, { [128] }>, end: Tile<u32, { [128] }>) -> u32 {
    let length = end - begin;
    let length: Tile<u32, { [] }> = reduce_max(length, 0i32);
    tile_to_scalar(length)
}

fn pool_inverse_begin(index: Tile<u32, { [128] }>, output: u32, kernel: u32, stride: u32, padding: u32) -> Tile<u32, { [128] }> {
    let shifted = index + broadcast_scalar(padding, shape![128]);
    let begin = (shifted - broadcast_scalar(kernel, shape![128])) / broadcast_scalar(stride, shape![128]) + broadcast_scalar(1u32, shape![128]);
    let begin = select(ge_tile(shifted, broadcast_scalar(kernel, shape![128])), begin, broadcast_scalar(0u32, shape![128]));
    min_tile(begin, broadcast_scalar(output, shape![128]))
}

fn pool_inverse_end(index: Tile<u32, { [128] }>, output: u32, stride: u32, padding: u32) -> Tile<u32, { [128] }> {
    min_tile((index + broadcast_scalar(padding, shape![128])) / broadcast_scalar(stride, shape![128]) + broadcast_scalar(1u32, shape![128]), broadcast_scalar(output, shape![128]))
}
