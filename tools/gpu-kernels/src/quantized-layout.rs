// Preserve every Q8_0 block byte while moving uploaded weights into 32-column tiles.
#[cutile::entry()]
unsafe fn q8_columns32_pack(input: *const f32, output: *mut f32, cols: u32, inner: u32, _format: u32, _phase: u32) {
    let i = indices();
    let blocks = inner / 32u32;
    let words = cols * blocks * 17u32 / 2u32;
    let mask = lt_tile(i, broadcast_scalar(words, shape![128]));
    let value = unsafe { read_tile(quantized_word_pointer(input), i, mask, 0u32) };
    let low = value & broadcast_scalar(65535u32, shape![128]);
    let high = value >> broadcast_scalar(16u32, shape![128]);
    let low: Tile<u16, { [128] }> = trunci(low, overflow::NoUnsignedWrap);
    let high: Tile<u16, { [128] }> = trunci(high, overflow::NoUnsignedWrap);
    let h0 = i * broadcast_scalar(2u32, shape![128]);
    let h1 = h0 + broadcast_scalar(1u32, shape![128]);
    let q0 = h0 / broadcast_scalar(17u32, shape![128]);
    let q1 = h1 / broadcast_scalar(17u32, shape![128]);
    let col0 = q0 / broadcast_scalar(blocks, shape![128]);
    let col1 = q1 / broadcast_scalar(blocks, shape![128]);
    let target0 = ((col0 / broadcast_scalar(32u32, shape![128]) * broadcast_scalar(blocks, shape![128]) + q0 % broadcast_scalar(blocks, shape![128]))
        * broadcast_scalar(32u32, shape![128])
        + col0 % broadcast_scalar(32u32, shape![128]))
        * broadcast_scalar(17u32, shape![128])
        + h0 % broadcast_scalar(17u32, shape![128]);
    let target1 = ((col1 / broadcast_scalar(32u32, shape![128]) * broadcast_scalar(blocks, shape![128]) + q1 % broadcast_scalar(blocks, shape![128]))
        * broadcast_scalar(32u32, shape![128])
        + col1 % broadcast_scalar(32u32, shape![128]))
        * broadcast_scalar(17u32, shape![128])
        + h1 % broadcast_scalar(17u32, shape![128]);
    let pointer: PointerTile<*mut f32, { [] }> = pointer_to_tile(output);
    let pointer: PointerTile<*mut u16, { [] }> = ptr_to_ptr(pointer);
    unsafe { write_tile(tile_to_pointer(pointer), target0, low, mask) };
    unsafe { write_tile(tile_to_pointer(pointer), target1, high, mask) };
}
