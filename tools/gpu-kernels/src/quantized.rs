unsafe fn read_quantized_matrix<const FORMAT: u32, const R: i32, const C: i32, const F: i32>(
    p: *const f32, index: Tile<u32, { [R, C] }>, mask: Tile<bool, { [R, C] }>, flat: Shape<{ [F] }>,
) -> Tile<f32, { [R, C] }> {
    // Packed weights are row-major [output, inner], while MMA consumes
    // [inner, output]. Decode adjacent weights before transposing the tile.
    let index = index.transpose().reshape(flat).reshape(shape![512]);
    let mask = mask.transpose().reshape(flat).reshape(shape![512]);
    let result = unsafe { read_quantized_512::<FORMAT>(p, index, mask) };
    result.reshape(shape![C, R]).transpose()
}
unsafe fn unpack_quantized<const FORMAT: u32>(input: *const f32, output: *mut f32, n: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(n, shape![128]));
    let value = unsafe { read_quantized_128::<FORMAT>(input, i, mask) };
    unsafe { write_tile(output, i, value, mask) };
}
unsafe fn gather_quantized<const FORMAT: u32>(input: *const f32, ids: *const u32, output: *mut f32, vocab: u32, width: u32, rows: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(rows * width, shape![128]));
    let token = unsafe { read_tile(ids, i / broadcast_scalar(width, shape![128]), mask, 0u32) };
    let valid = mask & lt_tile(token, broadcast_scalar(vocab, shape![128]));
    let source = token * broadcast_scalar(width, shape![128]) + i % broadcast_scalar(width, shape![128]);
    let value = unsafe { read_quantized_128::<FORMAT>(input, source, valid) };
    let value = select(valid, value, broadcast_scalar(0f32, shape![128]));
    unsafe { write_tile(output, i, value, mask) };
}
