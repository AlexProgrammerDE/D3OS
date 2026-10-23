fn float_u32(value: u32) -> f32 {
    let source: Tile<u32, { [] }> = scalar_to_tile(value);
    let result: Tile<f32, { [] }> = itof(source, rounding::NearestEven);
    tile_to_scalar(result)
}
fn unsigned(value: i32) -> u32 {
    let source: Tile<i32, { [] }> = scalar_to_tile(value);
    let bits: Tile<u32, { [] }> = bitcast(source);
    tile_to_scalar(bits)
}
fn block_id() -> u32 {
    let id: i32 = get_tile_block_id().0;
    let source: Tile<i32, { [] }> = scalar_to_tile(id);
    let bits: Tile<u32, { [] }> = bitcast(source);
    tile_to_scalar(bits)
}
// Masked device memory and common FP32 tile arithmetic.
fn lanes<const B: i32>() -> Tile<u32, { [B] }> {
    iota(shape![B])
}
fn indices() -> Tile<u32, { [128] }> {
    lanes::<128>() + broadcast_scalar(block_id() * 128u32, shape![128])
}
unsafe fn read_tile<E: ElementType, const B: i32>(p: *const E, index: Tile<u32, { [B] }>, mask: Tile<bool, { [B] }>, padding: E) -> Tile<E, { [B] }> {
    let base: PointerTile<*const E, { [] }> = pointer_to_tile(p);
    let base: PointerTile<*const E, { [1] }> = base.reshape(shape![1]);
    let base: PointerTile<*const E, { [B] }> = base.broadcast(shape![B]);
    let pointer = base.offset_tile(index);
    let (value, _): (Tile<E, { [B] }>, Token) =
        unsafe { load_ptr_tko(pointer, ordering::Weak, None::<scope::TileBlock>, Some(mask), Some(padding), None, Latency::<0>) };
    value
}
unsafe fn write_tile<E: ElementType, const B: i32>(p: *mut E, index: Tile<u32, { [B] }>, value: Tile<E, { [B] }>, mask: Tile<bool, { [B] }>) {
    let base: PointerTile<*mut E, { [] }> = pointer_to_tile(p);
    let base: PointerTile<*mut E, { [1] }> = base.reshape(shape![1]);
    let base: PointerTile<*mut E, { [B] }> = base.broadcast(shape![B]);
    let pointer = base.offset_tile(index);
    unsafe { store_ptr_tko(pointer, value, ordering::Weak, None::<scope::TileBlock>, Some(mask), None, Latency::<0>) };
}
fn bits_f32(value: u32) -> f32 {
    let bits: Tile<u32, { [] }> = scalar_to_tile(value);
    let float: Tile<f32, { [] }> = bitcast(bits);
    tile_to_scalar(float)
}
fn sigmoid_tile<const B: i32>(x: Tile<f32, { [B] }>) -> Tile<f32, { [B] }> {
    let one: Tile<f32, { [B] }> = broadcast_scalar(1f32, shape![B]);
    let zero: Tile<f32, { [B] }> = broadcast_scalar(0f32, shape![B]);
    let e = exp(zero - absf(x));
    select(ge_tile(x, zero), one / (one + e), e / (one + e))
}
fn silu_tile<const B: i32>(x: Tile<f32, { [B] }>) -> Tile<f32, { [B] }> {
    let s: Tile<f32, { [B] }> = sigmoid_tile(x);
    let s: Tile<f32, { [B] }> = s.reshape(shape![B]);
    x * s
}
fn root(x: f32) -> f32 {
    let tile: Tile<f32, { [] }> = scalar_to_tile(x);
    let result = sqrt(tile, rounding::NearestEven, ftz::Disabled);
    tile_to_scalar(result)
}

unsafe fn read_matrix<E: ElementType, const R: i32, const C: i32, const F: i32>(
    p: *const E, index: Tile<u32, { [R, C] }>, mask: Tile<bool, { [R, C] }>, padding: E, flat: Shape<{ [F] }>,
) -> Tile<E, { [R, C] }> {
    let index: Tile<u32, { [F] }> = index.reshape(flat);
    let mask: Tile<bool, { [F] }> = mask.reshape(flat);
    let value: Tile<E, { [F] }> = unsafe { read_tile(p, index, mask, padding) };
    value.reshape(shape![R, C])
}
unsafe fn write_matrix<E: ElementType, const R: i32, const C: i32, const F: i32>(
    p: *mut E, index: Tile<u32, { [R, C] }>, value: Tile<E, { [R, C] }>, mask: Tile<bool, { [R, C] }>, flat: Shape<{ [F] }>,
) {
    let index: Tile<u32, { [F] }> = index.reshape(flat);
    let mask: Tile<bool, { [F] }> = mask.reshape(flat);
    let value: Tile<E, { [F] }> = value.reshape(flat);
    unsafe { write_tile(p, index, value, mask) };
}
fn bf16_pointer(p: *const f32) -> *const u16 {
    let pointer: PointerTile<*const f32, { [] }> = pointer_to_tile(p);
    let pointer: PointerTile<*mut f32, { [] }> = cast_tile_mut(pointer);
    let pointer: PointerTile<*mut u16, { [] }> = ptr_to_ptr(pointer);
    let pointer: PointerTile<*const u16, { [] }> = cast_tile_const(pointer);
    tile_to_pointer(pointer)
}
unsafe fn read_weights<const BF16: bool, const R: i32, const C: i32, const F: i32>(
    p: *const f32, index: Tile<u32, { [R, C] }>, mask: Tile<bool, { [R, C] }>, flat: Shape<{ [F] }>,
) -> Tile<f32, { [R, C] }> {
    if BF16 {
        let bits = unsafe { read_matrix(bf16_pointer(p), index, mask, 0u16, flat) };
        let bits: Tile<u32, { [R, C] }> = exti(bits);
        let shifted = bits << broadcast_scalar(16u32, shape![R, C]);
        let value: Tile<f32, { [R, C] }> = bitcast(shifted);
        value
    } else {
        unsafe { read_matrix(p, index, mask, 0f32, flat) }
    }
}
fn signed(value: u32) -> i32 {
    let bits: Tile<u32, { [] }> = scalar_to_tile(value);
    let result: Tile<i32, { [] }> = bitcast(bits);
    tile_to_scalar(result)
}
unsafe fn read_scalar(p: *const f32, index: u32) -> f32 {
    let value = unsafe { read_tile(p, broadcast_scalar(index, shape![1]), broadcast_scalar(true, shape![1]), 0f32) };
    let value: Tile<f32, { [] }> = value.reshape(shape![]);
    tile_to_scalar(value)
}
unsafe fn write_scalar(p: *mut f32, index: u32, value: f32) {
    unsafe {
        write_tile(
            p,
            broadcast_scalar(index, shape![1]),
            broadcast_scalar(value, shape![1]),
            broadcast_scalar(true, shape![1]),
        )
    };
}
fn exponential(x: f32) -> f32 {
    let x: Tile<f32, { [] }> = scalar_to_tile(x);
    let y = exp(x);
    tile_to_scalar(y)
}
fn logarithm(x: f32) -> f32 {
    let x: Tile<f32, { [] }> = scalar_to_tile(x);
    let y = log(x);
    tile_to_scalar(y)
}
fn softplus(x: f32) -> f32 {
    let e = exponential(0f32 - absolute(x));
    let u = 1f32 + e;
    let correction = if u == 1f32 { e } else { logarithm(u) * e / (u - 1f32) };
    max(x, 0f32) + correction
}
fn sigmoid_scalar(x: f32) -> f32 {
    let e = exponential(0f32 - absolute(x));
    if x >= 0f32 { 1f32 / (1f32 + e) } else { e / (1f32 + e) }
}

fn absolute(x: f32) -> f32 {
    let x: Tile<f32, { [] }> = scalar_to_tile(x);
    let y = absf(x);
    tile_to_scalar(y)
}
fn byte_pointer(p: *const u32) -> *const u8 {
    let p: PointerTile<*const u32, { [] }> = pointer_to_tile(p);
    let p: PointerTile<*mut u32, { [] }> = cast_tile_mut(p);
    let p: PointerTile<*mut u8, { [] }> = ptr_to_ptr(p);
    let p: PointerTile<*const u8, { [] }> = cast_tile_const(p);
    tile_to_pointer(p)
}
fn byte_output(p: *mut u32) -> *mut u8 {
    let p: PointerTile<*mut u32, { [] }> = pointer_to_tile(p);
    let p: PointerTile<*mut u8, { [] }> = ptr_to_ptr(p);
    tile_to_pointer(p)
}
unsafe fn ordered_write<E: ElementType, const B: i32>(
    p: *mut E, index: Tile<u32, { [B] }>, value: Tile<E, { [B] }>, mask: Tile<bool, { [B] }>, token: Token,
) -> Token {
    let p: PointerTile<*mut E, { [] }> = pointer_to_tile(p);
    let p: PointerTile<*mut E, { [1] }> = p.reshape(shape![1]);
    let p: PointerTile<*mut E, { [B] }> = p.broadcast(shape![B]);
    let p = p.offset_tile(index);
    unsafe { store_ptr_tko(p, value, ordering::Weak, None::<scope::TileBlock>, Some(mask), Some(token), Latency::<0>) }
}
unsafe fn read_word(p: *const u32, index: u32) -> u32 {
    let x = unsafe { read_tile(p, broadcast_scalar(index, shape![1]), broadcast_scalar(true, shape![1]), 0u32) };
    let x: Tile<u32, { [] }> = x.reshape(shape![]);
    tile_to_scalar(x)
}
fn word_pointer(p: *const f32) -> *const u32 {
    let p: PointerTile<*const f32, { [] }> = pointer_to_tile(p);
    let p: PointerTile<*mut f32, { [] }> = cast_tile_mut(p);
    let p: PointerTile<*mut u32, { [] }> = ptr_to_ptr(p);
    let p: PointerTile<*const u32, { [] }> = cast_tile_const(p);
    tile_to_pointer(p)
}
