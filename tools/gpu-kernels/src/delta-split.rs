// Keep convolution history separate from the recurrent matrix so neither tile
// competes with the other's live values for the SM register file.
#[cutile::entry()]
unsafe fn delta_convolution_batch(input: *const f32, weights: *const f32, history: *mut f32, mixed: *mut f32, channels: u32, kernel: u32, rows: u32) {
    let c = indices();
    let cm = lt_tile(c, broadcast_scalar(channels, shape![128]));
    let cc: Tile<u32, { [128, 1] }> = c.reshape(shape![128, 1]);
    let cc: Tile<u32, { [128, 8] }> = cc.broadcast(shape![128, 8]);
    let h = lanes::<8>();
    let h: Tile<u32, { [1, 8] }> = h.reshape(shape![1, 8]);
    let h: Tile<u32, { [128, 8] }> = h.broadcast(shape![128, 8]);
    let mask = lt_tile(cc, broadcast_scalar(channels, shape![128, 8])) & lt_tile(h, broadcast_scalar(kernel, shape![128, 8]));
    let index = cc * broadcast_scalar(kernel, shape![128, 8]) + h;
    let mut past = unsafe { read_matrix(history, index, mask, 0f32, shape![1024]) };
    let weights = unsafe { read_matrix(weights, index, mask, 0f32, shape![1024]) };
    for row in 0u32..rows {
        let incoming = unsafe { read_tile(input, c + broadcast_scalar(row * channels, shape![128]), cm, 0f32) };
        let mut next: Tile<f32, { [128, 8] }> = broadcast_scalar(0f32, shape![128, 8]);
        for i in 0u32..kernel {
            let value: Tile<f32, { [128, 1] }> = if i + 1u32 < kernel {
                extract(past, [scalar_to_tile(0i32), scalar_to_tile(signed(i + 1u32))])
            } else {
                incoming.reshape(shape![128, 1])
            };
            next = unsafe { insert(value, next, [scalar_to_tile(0i32), scalar_to_tile(signed(i))]) };
        }
        past = next;
        let sum: Tile<f32, { [128] }> = reduce_sum(past * weights, 1i32);
        unsafe { write_tile(mixed, c + broadcast_scalar(row * channels, shape![128]), silu_tile(sum), cm) };
    }
    unsafe { write_matrix(history, index, past, mask, shape![1024]) };
}
