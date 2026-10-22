//! Ordered indexed drawing and palette presentation for native D3OS games.
#[inline(always)]
unsafe fn texture<const BYTES: bool>(address: u64, words: u32, offset: u64) -> Option<u32> {
    if offset < u64::from(words) {
        Some(if BYTES {
            u32::from(unsafe { (address as *const u8).add(offset as usize).read() })
        } else {
            unsafe { (address as *const u32).add(offset as usize).read() }
        })
    } else {
        None
    }
}

#[inline(always)]
unsafe fn draw<const BYTES: bool>(mut color: u32, x: u32, y: u32, p: *const u32, textures: u64, words: u32) -> u32 {
    let kind = unsafe { p.read() };
    let x0 = unsafe { p.add(1).read() };
    let y0 = unsafe { p.add(2).read() };
    let end = unsafe { p.add(3).read() };
    let flags = unsafe { p.add(11).read() };
    let double = if flags & 1 != 0 { 2 } else { 1 };
    let delta = if kind == 0 && x >= x0 && x - x0 < double && y >= y0 && y <= end {
        y - y0
    } else if kind == 1 && y == y0 && x >= x0 && x <= end {
        (x - x0) / double
    } else {
        return color;
    };
    let start = unsafe { p.add(4).read() };
    let step = unsafe { p.add(5).read() };
    let position = start.wrapping_add(delta.wrapping_mul(step));
    let index = if kind == 0 {
        if flags & 2 != 0 {
            ((position as i32) >> 16) as u32
        } else {
            (position >> 16) & 127
        }
    } else {
        ((position >> 4) & 0xfc0) | (position >> 26)
    };
    let source = unsafe { p.add(6).read() };
    let length = unsafe { p.add(7).read() };
    if index >= length {
        return color;
    }
    let Some(mut value) = (unsafe { texture::<BYTES>(textures, words, u64::from(source) + u64::from(index)) }) else {
        return color;
    };
    if flags & 2 != 0 {
        let translation = unsafe { p.add(9).read() };
        let Some(translated) = (unsafe { texture::<BYTES>(textures, words, u64::from(translation) + u64::from(value)) }) else {
            return color;
        };
        value = translated;
    }
    let palette = unsafe { p.add(8).read() };
    if let Some(value) = unsafe { texture::<BYTES>(textures, words, u64::from(palette) + u64::from(value)) } {
        color = value;
    }
    color
}

/// # Safety
/// Base/output contain width*height words, commands contains count*12 words,
/// textures contains words words. Output is disjoint. All products fit u32.
/// Texture offsets are untrusted and checked on the GPU before every access.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn raster_commands(base: u64, commands: u64, textures: u64, output: u64, width: u32, height: u32, count: u32, words: u32) {
    let pixel = super::index();
    if pixel >= width * height {
        return;
    }
    let x = pixel % width;
    let y = pixel / width;
    let mut color = unsafe { (base as *const u32).add(pixel as usize).read() };
    for command in 0..count {
        let p = unsafe { (commands as *const u32).add((command * 12) as usize) };
        color = unsafe { draw::<false>(color, x, y, p, textures, words) };
    }
    unsafe {
        (output as *mut u32).add(pixel as usize).write(color);
    }
}

/// # Safety
/// The first four buffers have the same sizes and alias rules as raster_commands.
/// Bins contains bin_words words and at least width+height+1 header offsets.
/// Every list offset and command ID is bounds-checked before use.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn raster_binned(
    base: u64, commands: u64, textures: u64, output: u64, bins: u64, width: u32, height: u32, count: u32, words: u32, bin_words: u32,
) {
    unsafe { binned::<false>([base, commands, textures, output, bins], [width, height, count, words, bin_words]) };
}

/// # Safety
/// Base/output contain width*height bytes; textures contains texture_bytes bytes.
/// Commands and bins retain their word layout and bounds rules from raster_binned.
/// Buffers are disjoint. Pixel and texture storage can end in a partial word.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn raster_binned_bytes(
    base: u64, commands: u64, textures: u64, output: u64, bins: u64, width: u32, height: u32, count: u32, texture_bytes: u32, bin_words: u32,
) {
    unsafe { binned::<true>([base, commands, textures, output, bins], [width, height, count, texture_bytes, bin_words]) };
}

#[inline(always)]
unsafe fn binned<const BYTES: bool>(pointers: [u64; 5], parameters: [u32; 5]) {
    let [base, commands, textures, output, bins] = pointers;
    let [width, height, count, words, bin_words] = parameters;
    let pixel = super::index();
    if pixel >= width * height {
        return;
    }
    let x = pixel % width;
    let y = pixel / width;
    let bins = bins as *const u32;
    let range = |i: u32| unsafe { (bins.add(i as usize).read().min(bin_words), bins.add(i as usize + 1).read().min(bin_words)) };
    let (mut column, column_end) = range(x);
    let (mut row, row_end) = range(width + y);
    let mut color = unsafe { texture::<BYTES>(base, width * height, u64::from(pixel)).unwrap_unchecked() };
    while column < column_end || row < row_end {
        let c = if column < column_end {
            unsafe { bins.add(column as usize).read() }
        } else {
            u32::MAX
        };
        let r = if row < row_end { unsafe { bins.add(row as usize).read() } } else { u32::MAX };
        let command = if column < column_end && (row >= row_end || c <= r) {
            column += 1;
            c
        } else {
            row += 1;
            r
        };
        if command < count {
            let p = unsafe { (commands as *const u32).add((command * 12) as usize) };
            color = unsafe { draw::<BYTES>(color, x, y, p, textures, words) };
        }
    }
    unsafe {
        if BYTES {
            (output as *mut u8).add(pixel as usize).write(color as u8);
        } else {
            (output as *mut u32).add(pixel as usize).write(color);
        }
    }
}

/// # Safety
/// Input contains width*height indices, palette contains 256 colors, disjoint
/// output contains width*height*scale*scale pixels. Scale is 1 through 4.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn palette_scale(input: u64, colors: u64, output: u64, width: u32, height: u32, scale: u32) {
    unsafe { palette::<false>(input, colors, output, width, height, scale) };
}

/// # Safety
/// Input contains width*height bytes, palette 256 words, output
/// width*height*scale*scale words. Ranges are disjoint and scale is 1 through 4.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn palette_scale_bytes(input: u64, colors: u64, output: u64, width: u32, height: u32, scale: u32) {
    unsafe { palette::<true>(input, colors, output, width, height, scale) };
}

#[inline(always)]
unsafe fn palette<const BYTES: bool>(input: u64, palette: u64, output: u64, width: u32, height: u32, scale: u32) {
    let i = super::index();
    if i >= width * height * scale * scale {
        return;
    }
    let row = i / (width * scale) / scale;
    let col = (i % (width * scale)) / scale;
    unsafe {
        let index = texture::<BYTES>(input, width * height, u64::from(row * width + col)).unwrap_unchecked() & 255;
        (output as *mut u32).add(i as usize).write((palette as *const u32).add(index as usize).read());
    }
}

/// # Safety
/// Frame contains width*height words, commands count*12, textures words.
/// Ranges are disjoint. One GPU thread preserves ordered neighbor-dependent
/// fuzz writes, including double-width columns and overlapping commands.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn raster_fuzz(frame: u64, commands: u64, textures: u64, width: u32, height: u32, count: u32, words: u32) {
    if super::index() != 0 {
        return;
    }
    for command in 0..count {
        let p = unsafe { (commands as *const u32).add((command * 12) as usize) };
        let x = unsafe { p.add(1).read() };
        let y0 = unsafe { p.add(2).read() };
        let end = unsafe { p.add(3).read() }.min(height - 1);
        if x >= width || y0 >= height {
            continue;
        }
        let initial = unsafe { p.add(4).read() } % 50;
        let table = unsafe { p.add(6).read() };
        let palette = unsafe { p.add(8).read() };
        let columns = if unsafe { p.add(11).read() } & 1 != 0 { 2 } else { 1 };
        for y in y0..=end {
            let Some(offset) = (unsafe { texture::<false>(textures, words, u64::from(table) + u64::from(initial.wrapping_add(y - y0) % 50)) }) else {
                continue;
            };
            for col in 0..columns {
                if x + col >= width {
                    continue;
                }
                let pixel = y * width + x + col;
                let neighbor = i64::from(pixel) + i64::from(offset as i32);
                if neighbor < 0 || neighbor >= i64::from(width) * i64::from(height) {
                    continue;
                }
                let index = unsafe { (frame as *const u32).add(neighbor as usize).read() } & 255;
                if let Some(value) = unsafe { texture::<false>(textures, words, u64::from(palette) + u64::from(index)) } {
                    unsafe {
                        (frame as *mut u32).add(pixel as usize).write(value);
                    }
                }
            }
        }
    }
}

/// # Safety
/// Frame has width*height bytes, commands count*12 words, textures texture_bytes
/// bytes, disjoint. A fuzz table contains 50 little-endian i32 byte offsets.
/// Only vertical offsets are supported. Nonvertical or invalid entries are ignored.
/// One active lane per warp owns a column and preserves command/pixel feedback order.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn raster_fuzz_bytes(frame: u64, commands: u64, textures: u64, width: u32, height: u32, count: u32, texture_bytes: u32) {
    let thread = super::index();
    if !thread.is_multiple_of(32) {
        return;
    }
    // Give columns separate warps: assigning adjacent columns to lanes would
    // serialize their distinct command branches within each warp.
    let column = thread / 32;
    if column >= width {
        return;
    }
    for command in 0..count {
        let p = unsafe { (commands as *const u32).add((command * 12) as usize) };
        let x = unsafe { p.add(1).read() };
        let y0 = unsafe { p.add(2).read() };
        let end = unsafe { p.add(3).read() }.min(height - 1);
        let columns = if unsafe { p.add(11).read() } & 1 != 0 { 2 } else { 1 };
        if column < x || column - x >= columns || y0 >= height {
            continue;
        }
        let initial = unsafe { p.add(4).read() } % 50;
        let table = u64::from(unsafe { p.add(6).read() });
        let palette = u64::from(unsafe { p.add(8).read() });
        for y in y0..=end {
            let offset = table + u64::from(initial.wrapping_add(y - y0) % 50) * 4;
            if offset + 4 > u64::from(texture_bytes) {
                continue;
            }
            let delta = unsafe { (textures as *const i32).cast::<u8>().add(offset as usize).cast::<i32>().read_unaligned() };
            if i64::from(delta) % i64::from(width) != 0 {
                continue;
            }
            let pixel = y * width + column;
            let neighbor = i64::from(pixel) + i64::from(delta);
            if neighbor < 0 || neighbor >= i64::from(width) * i64::from(height) {
                continue;
            }
            let index = unsafe { (frame as *const u8).add(neighbor as usize).read() };
            if let Some(value) = unsafe { texture::<true>(textures, texture_bytes, palette + u64::from(index)) } {
                unsafe { (frame as *mut u8).add(pixel as usize).write(value as u8) };
            }
        }
    }
}
