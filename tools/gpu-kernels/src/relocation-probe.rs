// Raw F64 probe for the vendor Sin/Cos table-pointer relocations.
// The caller supplies count input words and 2 * count output words.
#[cutile::entry()]
unsafe fn relocation_probe(input: *const u64, output: *mut u64, count: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(count, shape![128]));
    let bits = unsafe { read_tile(input, i, mask, 0u64) };
    let value: Tile<f64, { [128] }> = bitcast(bits);
    let sine: Tile<u64, { [128] }> = bitcast(sin(value));
    let cosine: Tile<u64, { [128] }> = bitcast(cos(value));
    let address = i * broadcast_scalar(2u32, shape![128]);
    unsafe {
        write_tile(output, address, sine, mask);
        write_tile(output, address + broadcast_scalar(1u32, shape![128]), cosine, mask);
    }
}
