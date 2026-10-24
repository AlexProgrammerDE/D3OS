#[cutile::entry()]
unsafe fn burn_i32_range(output: *mut u32, count: u32, first: u32, step: u32) {
    let index = lanes::<128>() + broadcast_scalar(block_id() * 128u32, shape![128]);
    let value = broadcast_scalar(first, shape![128]) + index * broadcast_scalar(step, shape![128]);
    unsafe { write_tile(output, index, value, lt_tile(index, broadcast_scalar(count, shape![128]))) };
}
