//! Byte-addressed drawing is checked against the word kernels and guarded tails.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_compute::{Buffer, Error, Result, nvidia::Nvidia, raster::command_bins};
use d3os_nvidia::compute::Kernel;

const WIDTH: usize = 13;
const HEIGHT: usize = 17;
const PIXELS: usize = WIDTH * HEIGHT;
const GUARD: u32 = 0xd305abcd;

pub(super) fn run(backend: &Nvidia) -> core::result::Result<(), String> {
    incremental_upload(backend).map_err(|error| format!("byte upload: {error}"))?;
    drawing(backend).map_err(|error| format!("byte raster: {error}"))?;
    fuzz(backend).map_err(|error| format!("byte fuzz: {error}"))?;
    super::super::report(format_args!(
        "NVIDIA_BYTE_RASTER_OK pixels=221 guards=words_and_bytes fuzz=ordered_columns palette_scales=4 backend=nvidia"
    ));
    Ok(())
}

fn incremental_upload(backend: &Nvidia) -> Result<()> {
    let buffer = backend.upload_words(&[GUARD; 6])?;
    for length in [1usize, 3, 4, 5, 9, 2] {
        backend.write_words(&buffer, 0, &[GUARD; 6])?;
        let values: Vec<_> = (0..length).map(|i| (i * 71 + 17) as u8).collect();
        backend.write_bytes(&buffer, 1, &values)?;
        let mut actual = [0; 6];
        backend.read_words_into(&buffer, 0, &mut actual)?;
        let mut expected: Vec<_> = [GUARD; 6].into_iter().flat_map(u32::to_le_bytes).collect();
        expected[4..4 + length.div_ceil(4) * 4].fill(0);
        expected[4..4 + length].copy_from_slice(&values);
        if actual.into_iter().flat_map(u32::to_le_bytes).ne(expected) {
            return Err(Error::DeviceFault);
        }
    }
    let mut before = [0; 6];
    backend.read_words_into(&buffer, 0, &mut before)?;
    backend.write_bytes(&buffer, buffer.len(), &[])?;
    for (offset, values) in [(buffer.len(), &[1][..]), (usize::MAX, &[][..]), (5, &[1; 5][..])] {
        if !matches!(backend.write_bytes(&buffer, offset, values), Err(Error::ShapeMismatch)) {
            return Err(Error::DeviceFault);
        }
    }
    let mut after = [0; 6];
    backend.read_words_into(&buffer, 0, &mut after)?;
    if after != before {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn drawing(backend: &Nvidia) -> Result<()> {
    let initial: Vec<u8> = (0..PIXELS).map(|i| (i * 11 % 256) as u8).collect();
    let mut textures: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    textures.extend((0..256).map(|i| (255 - i) as u8));
    textures.extend((0..256).map(|i| ((i + 7) % 256) as u8));
    // Leave a partial final resource word to check logical byte bounds.
    textures.push(251);
    let commands = [
        [0, 3, 1, 15, 120 << 16, 3 << 16, 0, 128, 4096, 0, 0, 0],
        [1, 1, 8, 12, 4 << 26 | 8 << 10, 1 << 26, 0, 4096, 4096, 0, 0, 0],
        [0, 9, 2, 12, 1 << 16, 1 << 16, 0, 128, 4096, 4352, 0, 3],
        [0, 12, 0, 16, 0, 1 << 16, 0, 128, 4096, 0, 0, 1],
        [0, 0, 0, 16, 0, 0, u32::MAX, 128, u32::MAX, 0, 0, 0],
        [0, 4, 0, 16, 17 << 16, 0, 0, 128, textures.len() as u32 - 1, 0, 0, 0],
    ];
    let bins = command_bins(commands.as_flattened(), WIDTH, HEIGHT)?;
    let bin_count = bins.len();
    let bins = backend.upload_words(&bins)?;
    let commands_gpu = backend.upload_words(commands.as_flattened())?;
    let base_words = backend.upload_words(&initial.iter().map(|&v| u32::from(v)).collect::<Vec<_>>())?;
    let textures_words = backend.upload_words(&textures.iter().map(|&v| u32::from(v)).collect::<Vec<_>>())?;
    let reference = backend.allocate_words(PIXELS)?;
    backend.dispatch_words(
        Kernel::Raster,
        &[(&base_words, 0), (&commands_gpu, 0), (&textures_words, 0), (&reference, 0)],
        &[WIDTH, HEIGHT, commands.len(), textures.len()],
    )?;
    let mut expected = vec![0; PIXELS];
    backend.read_words_into(&reference, 0, &mut expected)?;

    let base = backend.upload_bytes(&initial)?;
    let textures_bytes = backend.upload_bytes(&textures)?;
    let mut texture_roundtrip = vec![0; textures.len().div_ceil(4)];
    backend.read_words_into(&textures_bytes, 0, &mut texture_roundtrip)?;
    let packed: Vec<_> = texture_roundtrip.iter().flat_map(|v| v.to_le_bytes()).collect();
    if packed[..textures.len()] != textures || packed[textures.len()..].iter().any(|&v| v != 0) {
        return Err(Error::DeviceFault);
    }
    let output = backend.upload_words(&vec![GUARD; PIXELS.div_ceil(4) + 2])?;
    backend.dispatch_words(
        Kernel::RasterBinnedBytes,
        &[(&base, 0), (&commands_gpu, 0), (&textures_bytes, 0), (&output, 1), (&bins, 0)],
        &[WIDTH, HEIGHT, commands.len(), textures.len(), bin_count],
    )?;
    check_guarded(backend, &output, &expected.iter().map(|&v| v as u8).collect::<Vec<_>>(), GUARD)?;

    let colors: Vec<u32> = (0..256).map(|i| i * 0x71351 + 0x151709).collect();
    let palette = backend.upload_words(&colors)?;
    for scale in 1..=4 {
        let expanded = backend.allocate_words(PIXELS * scale * scale + 2)?;
        backend.write_words(&expanded, 0, &[GUARD])?;
        backend.write_words(&expanded, expanded.len() - 1, &[GUARD])?;
        backend.dispatch_words(Kernel::PaletteBytes, &[(&output, 1), (&palette, 0), (&expanded, 1)], &[WIDTH, HEIGHT, scale])?;
        let mut actual = vec![0; expanded.len()];
        backend.read_words_into(&expanded, 0, &mut actual)?;
        if actual[0] != GUARD || actual[actual.len() - 1] != GUARD {
            return Err(Error::DeviceFault);
        }
        for (i, &color) in actual[1..actual.len() - 1].iter().enumerate() {
            let pixel = (i / (WIDTH * scale) / scale) * WIDTH + (i % (WIDTH * scale)) / scale;
            if color != colors[expected[pixel] as usize] {
                return Err(Error::DeviceFault);
            }
        }
    }
    Ok(())
}

fn check_guarded(backend: &Nvidia, output: &d3os_compute::nvidia::GpuBuffer<u32>, expected: &[u8], padding: u32) -> Result<()> {
    let mut words = vec![0; output.len()];
    backend.read_words_into(output, 0, &mut words)?;
    if words[0] != GUARD || words[words.len() - 1] != GUARD {
        return Err(Error::DeviceFault);
    }
    let bytes: Vec<_> = words[1..words.len() - 1].iter().flat_map(|word| word.to_le_bytes()).collect();
    let tail = padding.to_le_bytes();
    if bytes[..expected.len()] != *expected
        || bytes[expected.len()..]
            .iter()
            .enumerate()
            .any(|(i, &value)| value != tail[(expected.len() + i) % 4])
    {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn fuzz(backend: &Nvidia) -> Result<()> {
    let initial: Vec<u8> = (0..PIXELS).map(|i| (i * 11 % 256) as u8).collect();
    let pattern: Vec<i32> = (0..50).map(|i| [WIDTH as i32, -(WIDTH as i32), 0, 2 * WIDTH as i32][i % 4]).collect();
    let mut textures = vec![0x37u8]; // Exercise unaligned i32 table loads.
    textures.extend(pattern.iter().flat_map(|offset| offset.to_le_bytes()));
    let palette_offset = textures.len();
    textures.extend((0..256).map(|i| ((i * 17 + 31) % 256) as u8));
    let commands = [
        [2, 3, 1, 15, 49, 0, 1, 50, palette_offset as u32, 0, 0, 0],
        [2, 2, 2, 14, 17, 0, 1, 50, palette_offset as u32, 0, 0, 1],
        [2, 12, 0, 16, 7, 0, 1, 50, palette_offset as u32, 0, 0, 1],
        [2, 1, 0, 16, 0, 0, u32::MAX, 50, palette_offset as u32, 0, 0, 0],
    ];
    let mut expected = initial.clone();
    for command in &commands[..3] {
        for y in command[2]..=command[3] {
            let delta = pattern[((command[4] + y - command[2]) % 50) as usize];
            for x in command[1]..=command[1] + (command[11] & 1) {
                if x >= WIDTH as u32 {
                    continue;
                }
                let pixel = y as usize * WIDTH + x as usize;
                let neighbor = pixel as i64 + i64::from(delta);
                if (0..PIXELS as i64).contains(&neighbor) {
                    expected[pixel] = textures[palette_offset + expected[neighbor as usize] as usize];
                }
            }
        }
    }
    let mut packed = vec![GUARD; PIXELS.div_ceil(4) + 2];
    for (i, &value) in initial.iter().enumerate() {
        let shift = i % 4 * 8;
        packed[1 + i / 4] = (packed[1 + i / 4] & !(255 << shift)) | (u32::from(value) << shift);
    }
    let frame = backend.upload_words(&packed)?;
    let command_buffer = backend.upload_words(commands.as_flattened())?;
    let textures_gpu = backend.upload_bytes(&textures)?;
    backend.dispatch_words(
        Kernel::FuzzBytes,
        &[(&frame, 1), (&command_buffer, 0), (&textures_gpu, 0)],
        &[WIDTH, HEIGHT, commands.len(), textures.len()],
    )?;
    check_guarded(backend, &frame, &expected, GUARD)?;

    for invalid in [1i32, -1, i32::MAX, i32::MIN] {
        textures[1..201].copy_from_slice(&invalid.to_le_bytes().repeat(50));
        let textures_gpu = backend.upload_bytes(&textures)?;
        backend.write_words(&frame, 0, &packed)?;
        backend.dispatch_words(
            Kernel::FuzzBytes,
            &[(&frame, 1), (&command_buffer, 0), (&textures_gpu, 0)],
            &[WIDTH, HEIGHT, commands.len(), textures.len()],
        )?;
        check_guarded(backend, &frame, &initial, GUARD)?;
    }
    Ok(())
}
