// Full-range I32 products use modulo-2^32 arithmetic throughout. The FP32
// Tensor Core product cannot represent arbitrary I32 operands exactly.
#[cutile::entry()]
unsafe fn burn_i32_matmul(a: *const u32, b: *const u32, output: *mut u32, rows: u32, inner: u32, cols: u32) {
    let columns = (cols + 127u32) / 128u32;
    let row = block_id() / columns;
    let col = lanes::<128>() + broadcast_scalar(block_id() % columns * 128u32, shape![128]);
    let mask = lt_tile(col, broadcast_scalar(cols, shape![128])) & broadcast_scalar(row < rows, shape![128]);
    let mut sum = broadcast_scalar(0u32, shape![128]);
    for k in 0u32..inner {
        let left = unsafe { read_tile(a, broadcast_scalar(row * inner + k, shape![128]), mask, 0u32) };
        let right = unsafe { read_tile(b, broadcast_scalar(k * cols, shape![128]) + col, mask, 0u32) };
        sum = sum + left * right;
    }
    unsafe { write_tile(output, broadcast_scalar(row * cols, shape![128]) + col, sum, mask) };
}
