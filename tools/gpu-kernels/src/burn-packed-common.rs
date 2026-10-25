// Raw packed values retain every source bit, including NaN payloads.
unsafe fn packed_read_width(input: *const u32, index: Tile<u32, {[128]}>, width: u32, valid: Tile<bool, {[128]}>) -> Tile<u64, {[128]}> {
    if width == 3u32 {
        let low = unsafe { read_tile(input, index * broadcast_scalar(2u32, shape![128]), valid, 0u32) };
        let high = unsafe { read_tile(input, index * broadcast_scalar(2u32, shape![128]) + broadcast_scalar(1u32, shape![128]), valid, 0u32) };
        exti(low) | (exti(high) << broadcast_scalar(32u64, shape![128]))
    } else {
        let per_word = 4u32 >> width;
        let word = unsafe { read_tile(input, index / broadcast_scalar(per_word, shape![128]), valid, 0u32) };
        let shift = index % broadcast_scalar(per_word, shape![128]) * broadcast_scalar(8u32 << width, shape![128]);
        exti((word >> shift) & broadcast_scalar(unsigned(-1i32) >> (32u32 - (8u32 << width)), shape![128]))
    }
}
fn packed_low(bits: Tile<u64, {[128]}>) -> Tile<u32, {[128]}> {
    trunci(bits & ((broadcast_scalar(1u64, shape![128]) << broadcast_scalar(32u64, shape![128])) - broadcast_scalar(1u64, shape![128])), overflow::NoUnsignedWrap)
}
unsafe fn packed_set(output: *mut u32, index: Tile<u32, {[128]}>, bits: Tile<u64, {[128]}>, width: u32, valid: Tile<bool, {[128]}>) {
    let low = packed_low(bits);
    if width == 3u32 {
        unsafe { write_tile(output, index * broadcast_scalar(2u32, shape![128]), low, valid) };
        let high = packed_low(bits >> broadcast_scalar(32u64, shape![128]));
        unsafe { write_tile(output, index * broadcast_scalar(2u32, shape![128]) + broadcast_scalar(1u32, shape![128]), high, valid) };
    } else if width == 2u32 {
        unsafe { write_tile(output, index, low, valid) };
    } else {
        // Disjoint logical elements can share a word. A masked raw-word CAS
        // preserves neighboring fields and tail bytes in initialized storage.
        let per_word = 4u32 >> width;
        let word = index / broadcast_scalar(per_word, shape![128]);
        let shift = index % broadcast_scalar(per_word, shape![128]) * broadcast_scalar(8u32 << width, shape![128]);
        let field = broadcast_scalar(unsigned(-1i32) >> (32u32 - (8u32 << width)), shape![128]);
        let field_mask = field << shift;
        let value = (low & field) << shift;
        let pointer: PointerTile<*mut u32, {[]}> = pointer_to_tile(output);
        let pointer: PointerTile<*mut u32, {[1]}> = pointer.reshape(shape![1]);
        let pointer: PointerTile<*mut u32, {[128]}> = pointer.broadcast(shape![128]);
        let pointer = pointer.offset_tile(word);
        let (initial, initial_token): (Tile<u32, {[128]}>, Token) = unsafe { atomic_rmw_tko(pointer, broadcast_scalar(0u32, shape![128]), atomic::Or, ordering::Relaxed, scope::Device, Some(valid), None) };
        let mut expected = initial;
        let mut token = initial_token;
        let mut active = valid;
        let mut remaining = 1u32;
        while remaining != 0u32 {
            let replacement = (expected & (field_mask ^ broadcast_scalar(unsigned(-1i32), shape![128]))) | value;
            let (observed, next_token): (Tile<u32, {[128]}>, Token) = unsafe { atomic_cas_tko(pointer, expected, replacement, ordering::Relaxed, scope::Device, Some(active), Some(token)) };
            active = active & ne_tile(observed, expected);
            expected = observed;
            token = next_token;
            let pending = select(active, broadcast_scalar(1u32, shape![128]), broadcast_scalar(0u32, shape![128]));
            let pending: Tile<u32, {[]}> = reduce_sum(pending, 0i32);
            remaining = tile_to_scalar(pending);
        }
    }
}
