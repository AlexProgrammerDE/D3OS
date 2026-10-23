// Native driver diagnostics are tile programs as well.
#[cutile::entry()]
unsafe fn timestamp_probe(output: *mut f32, iterations: u32) {
    let lane = lanes::<128>();
    let mask = broadcast_scalar(true, shape![128]);
    let mut x = unsafe { read_tile(output, lane, mask, 0f32) };
    for _i in 0u32..iterations {
        x = fma(
            x,
            broadcast_scalar(0.9999990463256836f32, shape![128]),
            broadcast_scalar(0.00000095367431640625f32, shape![128]),
            rounding::NearestEven,
            ftz::Disabled,
        );
    }
    unsafe { write_tile(output, lane, x, mask) };
}
// A runtime loop keeps more values live than the register file can hold.
// The assembler audit requires real LDL/STL instructions in this entry.
#[cutile::entry()]
unsafe fn local_stack_probe(output: *mut u32, count: u32) {
    let seed = block_id() * unsigned(-1640531527i32);
    let mut value = lanes::<65536>() ^ broadcast_scalar(seed, shape![65536]);
    for _i in 0u32..(count % 3u32 + 1u32) {
        value = (value ^ (value << broadcast_scalar(13u32, shape![65536]))) * broadcast_scalar(0x51a6d305u32, shape![65536]);
    }
    let checksum: Tile<u32, { [] }> = reduce_sum(value, 0i32);
    let checksum: u32 = tile_to_scalar(checksum);
    unsafe {
        write_tile(
            output,
            broadcast_scalar(block_id(), shape![1]),
            broadcast_scalar(checksum, shape![1]),
            broadcast_scalar(true, shape![1]),
        )
    };
}
