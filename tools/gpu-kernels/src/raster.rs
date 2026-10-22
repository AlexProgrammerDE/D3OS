//! Ordered indexed drawing and palette presentation for native D3OS games.
#[inline(always)]
unsafe fn texture(address: u64, words: u32, offset: u64) -> Option<u32> {
    if offset < u64::from(words) {
        Some(unsafe { (address as *const u32).add(offset as usize).read() })
    } else {
        None
    }
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
            continue;
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
            continue;
        }
        let Some(mut value) = (unsafe { texture(textures, words, u64::from(source) + u64::from(index)) }) else {
            continue;
        };
        if flags & 2 != 0 {
            let translation = unsafe { p.add(9).read() };
            let Some(translated) = (unsafe { texture(textures, words, u64::from(translation) + u64::from(value)) }) else {
                continue;
            };
            value = translated;
        }
        let palette = unsafe { p.add(8).read() };
        if let Some(value) = unsafe { texture(textures, words, u64::from(palette) + u64::from(value)) } {
            color = value;
        }
    }
    unsafe {
        (output as *mut u32).add(pixel as usize).write(color);
    }
}

/// # Safety
/// Input contains width*height indices, palette contains 256 colors, disjoint
/// output contains width*height*scale*scale pixels. Scale is 1 through 4.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn palette_scale(input: u64, palette: u64, output: u64, width: u32, height: u32, scale: u32) {
    let i = super::index();
    if i >= width * height * scale * scale {
        return;
    }
    let row = i / (width * scale) / scale;
    let col = (i % (width * scale)) / scale;
    unsafe {
        let index = (input as *const u32).add((row * width + col) as usize).read() & 255;
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
            let Some(offset) = (unsafe { texture(textures, words, u64::from(table) + u64::from(initial.wrapping_add(y - y0) % 50)) }) else {
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
                if let Some(value) = unsafe { texture(textures, words, u64::from(palette) + u64::from(index)) } {
                    unsafe {
                        (frame as *mut u32).add(pixel as usize).write(value);
                    }
                }
            }
        }
    }
}
