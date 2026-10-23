unsafe fn read_quad<E: ElementType>(p: *const E, index: Tile<u32, { [2, 4, 32] }>, mask: Tile<bool, { [2, 4, 32] }>, padding: E) -> Tile<E, { [2, 4, 32] }> {
    let base: PointerTile<*const E, { [] }> = pointer_to_tile(p);
    let base: PointerTile<*const E, { [1, 1, 1] }> = base.reshape(shape![1, 1, 1]);
    let base: PointerTile<*const E, { [2, 4, 32] }> = base.broadcast(shape![2, 4, 32]);
    let pointer = base.offset_tile(index);
    let (value, _): (Tile<E, { [2, 4, 32] }>, Token) =
        unsafe { load_ptr_tko(pointer, ordering::Weak, None::<scope::TileBlock>, Some(mask), None, None, Latency::<0>) };
    select(mask, value, broadcast_scalar(padding, shape![2, 4, 32]))
}

unsafe fn quad_project_4<const GATED: bool, const RESIDUAL: bool>(
    a: *const f32, b: *const f32, up: *const f32, residual: *const f32, output: *mut f32, inner: u32, cols: u32,
) {
    let first = block_id() * 4u32;
    unsafe { quad_project_4_at::<GATED, RESIDUAL>(a, b, up, residual, output, inner, cols, first, first) }
}

unsafe fn quad_project_4_at<const GATED: bool, const RESIDUAL: bool>(
    a: *const f32, b: *const f32, up: *const f32, residual: *const f32, output: *mut f32, inner: u32, cols: u32, first: u32, dest_first: u32,
) {
    let col = lanes::<4>() + broadcast_scalar(first, shape![4]);
    let col: Tile<u32, { [1, 4, 1] }> = col.reshape(shape![1, 4, 1]);
    let col: Tile<u32, { [2, 4, 32] }> = col.broadcast(shape![2, 4, 32]);
    let lane = lanes::<32>();
    let lane: Tile<u32, { [1, 1, 32] }> = lane.reshape(shape![1, 1, 32]);
    let lane: Tile<u32, { [2, 4, 32] }> = lane.broadcast(shape![2, 4, 32]);
    let component = lanes::<2>();
    let component: Tile<u32, { [2, 1, 1] }> = component.reshape(shape![2, 1, 1]);
    let component: Tile<u32, { [2, 4, 32] }> = component.broadcast(shape![2, 4, 32]);
    let mut even: Tile<f32, { [2, 4, 32] }> = broadcast_scalar(0f32, shape![2, 4, 32]);
    let mut odd = even;
    let mut up_even = even;
    let mut up_odd = even;
    let mut start = 0u32;
    while start < inner / 2u32 {
        let pair = lane * broadcast_scalar(2u32, shape![2, 4, 32]) + component + broadcast_scalar(start, shape![2, 4, 32]);
        let mask = lt_tile(pair, broadcast_scalar(inner / 2u32, shape![2, 4, 32])) & lt_tile(col, broadcast_scalar(cols, shape![2, 4, 32]));
        let xe = unsafe { read_quad(a, pair * broadcast_scalar(2u32, shape![2, 4, 32]), mask, 0f32) };
        let xo = unsafe {
            read_quad(
                a,
                pair * broadcast_scalar(2u32, shape![2, 4, 32]) + broadcast_scalar(1u32, shape![2, 4, 32]),
                mask,
                0f32,
            )
        };
        let index = col * broadcast_scalar(inner / 2u32, shape![2, 4, 32]) + pair;
        let words = unsafe { read_quad(word_pointer(b), index, mask, 0u32) };
        let lo: Tile<f32, { [2, 4, 32] }> = bitcast(words << broadcast_scalar(16u32, shape![2, 4, 32]));
        let hi: Tile<f32, { [2, 4, 32] }> = bitcast(words & broadcast_scalar(unsigned(-65536i32), shape![2, 4, 32]));
        even = fma(xe, lo, even, rounding::NearestEven, ftz::Disabled);
        odd = fma(xo, hi, odd, rounding::NearestEven, ftz::Disabled);
        if GATED {
            let words = unsafe { read_quad(word_pointer(up), index, mask, 0u32) };
            let lo: Tile<f32, { [2, 4, 32] }> = bitcast(words << broadcast_scalar(16u32, shape![2, 4, 32]));
            let hi: Tile<f32, { [2, 4, 32] }> = bitcast(words & broadcast_scalar(unsigned(-65536i32), shape![2, 4, 32]));
            up_even = fma(xe, lo, up_even, rounding::NearestEven, ftz::Disabled);
            up_odd = fma(xo, hi, up_odd, rounding::NearestEven, ftz::Disabled);
        }
        start = start + 64u32;
    }
    let sum: Tile<f32, { [4, 32] }> = reduce_sum(even + odd, 0i32);
    let sum: Tile<f32, { [4] }> = reduce_sum(sum, 1i32);
    let value = if GATED {
        let up: Tile<f32, { [4, 32] }> = reduce_sum(up_even + up_odd, 0i32);
        let up: Tile<f32, { [4] }> = reduce_sum(up, 1i32);
        silu_tile(sum) * up
    } else {
        sum
    };
    let col = lanes::<4>() + broadcast_scalar(first, shape![4]);
    let mask = lt_tile(col, broadcast_scalar(cols, shape![4]));
    let value = if RESIDUAL {
        let previous = unsafe { read_tile(residual, col, mask, 0f32) };
        previous + value
    } else {
        value
    };
    let destination = lanes::<4>() + broadcast_scalar(dest_first, shape![4]);
    unsafe { write_tile(output, destination, value, mask) };
}
