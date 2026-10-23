// Indexed drawing preserves command order; texture arithmetic is checked in u64.
unsafe fn read_indexed<const BYTES: bool, const B: i32>(p: *const u32, index: Tile<u32, { [B] }>, mask: Tile<bool, { [B] }>) -> Tile<u32, { [B] }> {
    if BYTES {
        let x = unsafe { read_tile(byte_pointer(p), index, mask, 0u8) };
        let x: Tile<u32, { [B] }> = exti(x);
        x
    } else {
        unsafe { read_tile(p, index, mask, 0u32) }
    }
}
unsafe fn write_indexed<const BYTES: bool, const B: i32>(p: *mut u32, index: Tile<u32, { [B] }>, value: Tile<u32, { [B] }>, mask: Tile<bool, { [B] }>) {
    if BYTES {
        let x: Tile<u8, { [B] }> = trunci(value & broadcast_scalar(255u32, shape![B]), overflow::NoUnsignedWrap);
        unsafe { write_tile(byte_output(p), index, x, mask) }
    } else {
        unsafe { write_tile(p, index, value, mask) }
    }
}
unsafe fn draw_tile<const BYTES: bool>(
    color: Tile<u32, { [32] }>, x: Tile<u32, { [32] }>, y: Tile<u32, { [32] }>, commands: *const u32, command: Tile<u32, { [32] }>, textures: *const u32,
    words: u32, active: Tile<bool, { [32] }>,
) -> Tile<u32, { [32] }> {
    let base = command * broadcast_scalar(12u32, shape![32]);
    let kind = unsafe { read_tile(commands, base, active, 0u32) };
    let x0 = unsafe { read_tile(commands, base + broadcast_scalar(1u32, shape![32]), active, 0u32) };
    let y0 = unsafe { read_tile(commands, base + broadcast_scalar(2u32, shape![32]), active, 0u32) };
    let end = unsafe { read_tile(commands, base + broadcast_scalar(3u32, shape![32]), active, 0u32) };
    let flags = unsafe { read_tile(commands, base + broadcast_scalar(11u32, shape![32]), active, 0u32) };
    let double = select(
        ne_tile(flags & broadcast_scalar(1u32, shape![32]), broadcast_scalar(0u32, shape![32])),
        broadcast_scalar(2u32, shape![32]),
        broadcast_scalar(1u32, shape![32]),
    );
    let vertical = eq_tile(kind, broadcast_scalar(0u32, shape![32])) & ge_tile(x, x0) & lt_tile(x - x0, double) & ge_tile(y, y0) & le_tile(y, end);
    let horizontal = eq_tile(kind, broadcast_scalar(1u32, shape![32])) & eq_tile(y, y0) & ge_tile(x, x0) & le_tile(x, end);
    let active = active & (vertical | horizontal);
    let delta = select(vertical, y - y0, (x - x0) / double);
    let start = unsafe { read_tile(commands, base + broadcast_scalar(4u32, shape![32]), active, 0u32) };
    let step = unsafe { read_tile(commands, base + broadcast_scalar(5u32, shape![32]), active, 0u32) };
    let position = start + delta * step;
    let translated = ne_tile(flags & broadcast_scalar(2u32, shape![32]), broadcast_scalar(0u32, shape![32]));
    let signed_position: Tile<i32, { [32] }> = bitcast(position);
    let arithmetic = signed_position >> broadcast_scalar(16i32, shape![32]);
    let arithmetic: Tile<u32, { [32] }> = bitcast(arithmetic);
    let index = select(
        vertical,
        select(
            translated,
            arithmetic,
            (position >> broadcast_scalar(16u32, shape![32])) & broadcast_scalar(127u32, shape![32]),
        ),
        ((position >> broadcast_scalar(4u32, shape![32])) & broadcast_scalar(0xfc0u32, shape![32])) | (position >> broadcast_scalar(26u32, shape![32])),
    );
    let source = unsafe { read_tile(commands, base + broadcast_scalar(6u32, shape![32]), active, 0u32) };
    let length = unsafe { read_tile(commands, base + broadcast_scalar(7u32, shape![32]), active, 0u32) };
    let (value, valid): (Tile<u32, { [32] }>, Tile<bool, { [32] }>) =
        unsafe { texture_tile_32::<BYTES>(textures, source, index, words, active & lt_tile(index, length)) };
    let translation = unsafe { read_tile(commands, base + broadcast_scalar(9u32, shape![32]), valid & translated, 0u32) };
    let (mapped, mapped_valid): (Tile<u32, { [32] }>, Tile<bool, { [32] }>) =
        unsafe { texture_tile_32::<BYTES>(textures, translation, value, words, valid & translated) };
    let value = select(translated, mapped, value);
    let valid = valid & (eq_tile(translated, broadcast_scalar(false, shape![32])) | mapped_valid);
    let palette = unsafe { read_tile(commands, base + broadcast_scalar(8u32, shape![32]), valid, 0u32) };
    let (result, valid): (Tile<u32, { [32] }>, Tile<bool, { [32] }>) = unsafe { texture_tile_32::<BYTES>(textures, palette, value, words, valid) };
    select(valid, result, color)
}
#[cutile::entry()]
unsafe fn raster_commands(base: *const u32, commands: *const u32, textures: *const u32, output: *mut u32, width: u32, height: u32, count: u32, words: u32) {
    let i = lanes::<32>() + broadcast_scalar(block_id() * 32u32, shape![32]);
    let mask = lt_tile(i, broadcast_scalar(width * height, shape![32]));
    let x = i % broadcast_scalar(width, shape![32]);
    let y = i / broadcast_scalar(width, shape![32]);
    let mut color = unsafe { read_tile(base, i, mask, 0u32) };
    for command in 0u32..count {
        color = unsafe { draw_tile::<false>(color, x, y, commands, broadcast_scalar(command, shape![32]), textures, words, mask) }
    }
    unsafe { write_tile(output, i, color, mask) };
}
unsafe fn binned<const BYTES: bool>(
    base: *const u32, commands: *const u32, textures: *const u32, output: *mut u32, bins: *const u32, width: u32, height: u32, count: u32, words: u32,
    bin_words: u32,
) {
    let i = lanes::<32>() + broadcast_scalar(block_id() * 32u32, shape![32]);
    let mask = lt_tile(i, broadcast_scalar(width * height, shape![32]));
    let x = i % broadcast_scalar(width, shape![32]);
    let y = i / broadcast_scalar(width, shape![32]);
    let mut column = unsafe { read_tile(bins, x, mask, 0u32) };
    let column_end = unsafe { read_tile(bins, x + broadcast_scalar(1u32, shape![32]), mask, 0u32) };
    let mut row = unsafe { read_tile(bins, y + broadcast_scalar(width, shape![32]), mask, 0u32) };
    let row_end = unsafe { read_tile(bins, y + broadcast_scalar(width + 1u32, shape![32]), mask, 0u32) };
    column = min_tile(column, broadcast_scalar(bin_words, shape![32]));
    row = min_tile(row, broadcast_scalar(bin_words, shape![32]));
    let column_end = min_tile(column_end, broadcast_scalar(bin_words, shape![32]));
    let row_end = min_tile(row_end, broadcast_scalar(bin_words, shape![32]));
    let mut color = unsafe { read_indexed::<BYTES, 32>(base, i, mask) };
    let mut running = true;
    while running {
        let cm = mask & lt_tile(column, column_end);
        let rm = mask & lt_tile(row, row_end);
        let c = unsafe { read_tile(bins, column, cm, unsigned(-1i32)) };
        let r = unsafe { read_tile(bins, row, rm, unsigned(-1i32)) };
        let take_column = cm & (eq_tile(rm, broadcast_scalar(false, shape![32])) | le_tile(c, r));
        let command = select(take_column, c, r);
        let active = cm | rm;
        column = column + select(take_column, broadcast_scalar(1u32, shape![32]), broadcast_scalar(0u32, shape![32]));
        row = row
            + select(
                active & eq_tile(take_column, broadcast_scalar(false, shape![32])),
                broadcast_scalar(1u32, shape![32]),
                broadcast_scalar(0u32, shape![32]),
            );
        color = unsafe {
            draw_tile::<BYTES>(
                color,
                x,
                y,
                commands,
                command,
                textures,
                words,
                active & lt_tile(command, broadcast_scalar(count, shape![32])),
            )
        };
        let remains = mask & (lt_tile(column, column_end) | lt_tile(row, row_end));
        let remains = select(remains, broadcast_scalar(1u32, shape![32]), broadcast_scalar(0u32, shape![32]));
        let any: Tile<u32, { [] }> = reduce_max(remains, 0i32);
        let any: u32 = tile_to_scalar(any);
        running = any != 0u32;
    }
    unsafe { write_indexed::<BYTES, 32>(output, i, color, mask) };
}
#[cutile::entry()]
unsafe fn raster_binned(
    base: *const u32, commands: *const u32, textures: *const u32, output: *mut u32, bins: *const u32, width: u32, height: u32, count: u32, words: u32,
    bin_words: u32,
) {
    unsafe { binned::<false>(base, commands, textures, output, bins, width, height, count, words, bin_words) }
}
#[cutile::entry()]
unsafe fn raster_binned_bytes(
    base: *const u32, commands: *const u32, textures: *const u32, output: *mut u32, bins: *const u32, width: u32, height: u32, count: u32, words: u32,
    bin_words: u32,
) {
    unsafe { binned::<true>(base, commands, textures, output, bins, width, height, count, words, bin_words) }
}
unsafe fn palette<const BYTES: bool>(input: *const u32, colors: *const u32, output: *mut u32, width: u32, height: u32, scale: u32) {
    let i = indices();
    let mask = lt_tile(i, broadcast_scalar(width * height * scale * scale, shape![128]));
    let row = i / broadcast_scalar(width * scale * scale, shape![128]);
    let col = i % broadcast_scalar(width * scale, shape![128]) / broadcast_scalar(scale, shape![128]);
    let index = unsafe { read_indexed::<BYTES, 128>(input, row * broadcast_scalar(width, shape![128]) + col, mask) };
    let color = unsafe { read_tile(colors, index & broadcast_scalar(255u32, shape![128]), mask, 0u32) };
    unsafe { write_tile(output, i, color, mask) };
}
#[cutile::entry()]
unsafe fn palette_scale(input: *const u32, colors: *const u32, output: *mut u32, width: u32, height: u32, scale: u32) {
    unsafe { palette::<false>(input, colors, output, width, height, scale) }
}
#[cutile::entry()]
unsafe fn palette_scale_bytes(input: *const u32, colors: *const u32, output: *mut u32, width: u32, height: u32, scale: u32) {
    unsafe { palette::<true>(input, colors, output, width, height, scale) }
}
unsafe fn texture_scalar<const BYTES: bool>(p: *const u32, offset: u32, relative: u32, words: u32) -> (u32, u32) {
    let (value, valid): (Tile<u32, { [1] }>, Tile<bool, { [1] }>) = unsafe {
        texture_tile_1::<BYTES>(
            p,
            broadcast_scalar(offset, shape![1]),
            broadcast_scalar(relative, shape![1]),
            words,
            broadcast_scalar(true, shape![1]),
        )
    };
    let value: Tile<u32, { [] }> = value.reshape(shape![]);
    let valid: Tile<bool, { [] }> = valid.reshape(shape![]);
    (tile_to_scalar(value), if tile_to_scalar::<bool, bool>(valid) { 1u32 } else { 0u32 })
}
#[allow(unused_assignments, reason = "cuTile requires initialized loop and branch bindings")]
fn neighbor_index(pixel: u32, offset: u32, total: u32) -> (u32, bool) {
    let mut neighbor = 0u32;
    let mut valid = false;
    if signed(offset) < 0i32 {
        let magnitude = 0u32 - offset;
        neighbor = pixel - magnitude;
        valid = pixel >= magnitude;
    } else {
        neighbor = pixel + offset;
        valid = offset < total && pixel < total - offset;
    }
    (neighbor, valid)
}
#[allow(unused_assignments, reason = "cuTile requires initialized loop and branch bindings")]
unsafe fn fuzz<const BYTES: bool>(frame: *mut u32, commands: *const u32, textures: *const u32, width: u32, height: u32, count: u32, words: u32) {
    let column = block_id();
    let mut token = new_token_unordered();
    for command in 0u32..count {
        let x = unsafe { read_word(commands, command * 12u32 + 1u32) };
        let y0 = unsafe { read_word(commands, command * 12u32 + 2u32) };
        let end = unsafe { read_word(commands, command * 12u32 + 3u32) };
        let end = min(end, height - 1u32);
        let flags = unsafe { read_word(commands, command * 12u32 + 11u32) };
        let columns = if flags & 1u32 != 0u32 { 2u32 } else { 1u32 };
        let owns = if BYTES { column >= x && column - x < columns } else { x < width };
        if owns && y0 < height {
            let initial = unsafe { read_word(commands, command * 12u32 + 4u32) } % 50u32;
            let table = unsafe { read_word(commands, command * 12u32 + 6u32) };
            let palette = unsafe { read_word(commands, command * 12u32 + 8u32) };
            let mut y = y0;
            while y <= end {
                let entry = (initial + y - y0) % 50u32;
                let mut delta = 0u32;
                let mut valid = 0u32;
                if BYTES {
                    let (a, va) = unsafe { texture_scalar::<true>(textures, table, entry * 4u32, words) };
                    let (b, vb) = unsafe { texture_scalar::<true>(textures, table, entry * 4u32 + 1u32, words) };
                    let (c, vc) = unsafe { texture_scalar::<true>(textures, table, entry * 4u32 + 2u32, words) };
                    let (d, vd) = unsafe { texture_scalar::<true>(textures, table, entry * 4u32 + 3u32, words) };
                    delta = a | (b << 8u32) | (c << 16u32) | (d << 24u32);
                    valid = va & vb & vc & vd;
                } else {
                    let (d, v) = unsafe { texture_scalar::<false>(textures, table, entry, words) };
                    delta = d;
                    valid = v;
                }
                let vertical = if signed(delta) < 0i32 {
                    (0u32 - delta) % width == 0u32
                } else {
                    delta % width == 0u32
                };
                if valid != 0u32 && (BYTES == false || vertical) {
                    let first = if BYTES { column } else { x };
                    let span = if BYTES { 1u32 } else { columns };
                    for part in 0u32..span {
                        let col = first + part;
                        if col < width {
                            let pixel = y * width + col;
                            let (neighbor, valid) = neighbor_index(pixel, delta, width * height);
                            if valid {
                                let ni: Tile<u32, { [1] }> = broadcast_scalar(neighbor, shape![1]);
                                let all: Tile<bool, { [1] }> = broadcast_scalar(true, shape![1]);
                                let mut index: Tile<u32, { [1] }> = broadcast_scalar(0u32, shape![1]);
                                if BYTES {
                                    let (x, next): (Tile<u8, { [1] }>, Token) = unsafe { ordered_read_u8_1(byte_output(frame), ni, all, 0u8, token) };
                                    index = exti(x);
                                    token = next;
                                } else {
                                    let (x, next): (Tile<u32, { [1] }>, Token) = unsafe { ordered_read_u32_1(frame, ni, all, 0u32, token) };
                                    index = x;
                                    token = next;
                                }
                                let index = index & broadcast_scalar(255u32, shape![1]);
                                let index: Tile<u32, { [] }> = index.reshape(shape![]);
                                let index: u32 = tile_to_scalar(index);
                                let (value, valid) = unsafe { texture_scalar::<BYTES>(textures, palette, index, words) };
                                if valid != 0u32 {
                                    let pi: Tile<u32, { [1] }> = broadcast_scalar(pixel, shape![1]);
                                    let value: Tile<u32, { [1] }> = broadcast_scalar(value, shape![1]);
                                    if BYTES {
                                        let value: Tile<u8, { [1] }> = trunci(value, overflow::NoUnsignedWrap);
                                        token = unsafe { ordered_write(byte_output(frame), pi, value, all, token) }
                                    } else {
                                        token = unsafe { ordered_write(frame, pi, value, all, token) }
                                    }
                                }
                            }
                        }
                    }
                }
                y = y + 1u32;
            }
        }
    }
}
#[cutile::entry()]
unsafe fn raster_fuzz(frame: *mut u32, commands: *const u32, textures: *const u32, width: u32, height: u32, count: u32, words: u32) {
    unsafe { fuzz::<false>(frame, commands, textures, width, height, count, words) }
}
#[cutile::entry()]
unsafe fn raster_fuzz_bytes(frame: *mut u32, commands: *const u32, textures: *const u32, width: u32, height: u32, count: u32, words: u32) {
    unsafe { fuzz::<true>(frame, commands, textures, width, height, count, words) }
}
