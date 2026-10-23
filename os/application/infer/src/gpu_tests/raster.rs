use alloc::{format, string::String, vec, vec::Vec};
use d3os_compute::nvidia::Nvidia;
use d3os_nvidia::compute::Kernel;

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    check(backend).map_err(|error| format!("raster: {error}"))?;
    fuzz(backend).map_err(|error| format!("fuzz: {error}"))?;
    super::super::report(format_args!(
        "NVIDIA_RASTER_OK columns=2 spans=1 fuzz=2 palette_scale=2 guards=ok backend=nvidia"
    ));
    Ok(())
}

fn fuzz(backend: &Nvidia) -> d3os_compute::Result<()> {
    let (width, height) = (13usize, 17usize);
    let initial: Vec<u32> = (0..width * height).map(|i| (i * 11 % 256) as u32).collect();
    let mut expected = initial.clone();
    let mut textures: Vec<u32> = (0..50)
        .map(|i| match i % 3 {
            0 => width as u32,
            1 => (-(width as i32)) as u32,
            _ => 0,
        })
        .collect();
    textures.extend((0..256).map(|i| (i + 31) % 256));
    // Overlapping columns require sequential feedback from earlier pixels.
    let commands = [
        [2, 3, 1, 15, 49, 0, 0, 50, 50, 0, 0, 0],
        [2, 2, 2, 14, 17, 0, 0, 50, 50, 0, 0, 1],
        [2, 1, 0, 16, 0, 0, u32::MAX, 50, 50, 0, 0, 0],
    ];
    for command in &commands[..2] {
        for y in command[2]..=command[3] {
            let delta = textures[((command[4] + y - command[2]) % 50) as usize] as i32;
            for x in command[1]..=command[1] + (command[11] & 1) {
                let pixel = y as usize * width + x as usize;
                let neighbor = (pixel as i32 + delta) as usize;
                expected[pixel] = (expected[neighbor] + 31) % 256;
            }
        }
    }
    let frame = backend.allocate_words(initial.len() + 2)?;
    let guard = 0xd305abcd;
    backend.write_words(&frame, 0, &[guard])?;
    backend.write_words(&frame, 1, &initial)?;
    backend.write_words(&frame, initial.len() + 1, &[guard])?;
    let command_buffer = backend.upload_words(commands.as_flattened())?;
    let texture_buffer = backend.upload_words(&textures)?;
    backend.dispatch_words(
        Kernel::Fuzz,
        &[(&frame, 1), (&command_buffer, 0), (&texture_buffer, 0)],
        &[width, height, commands.len(), textures.len()],
    )?;
    let mut actual = vec![0; initial.len() + 2];
    backend.read_words_into(&frame, 0, &mut actual)?;
    if actual[0] != guard || actual[initial.len() + 1] != guard || actual[1..actual.len() - 1] != expected {
        return Err(d3os_compute::Error::DeviceFault);
    }
    Ok(())
}

fn check(backend: &Nvidia) -> d3os_compute::Result<()> {
    let (width, height) = (32usize, 17usize);
    let mut textures: Vec<u32> = (0..4096).map(|i| (i % 256) as u32).collect();
    textures.extend((0..256).map(|i| 255 - i));
    textures.extend((0..256).map(|i| (i + 7) % 256));
    let mut expected = vec![3u32; width * height];
    let mut commands = Vec::new();
    commands.extend_from_slice(&[0, 5, 1, 15, 120 << 16, 3 << 16, 0, 128, 4096, 0, 0, 0]);
    for y in 1..=15 {
        expected[y * width + 5] = 255 - ((120 + (y - 1) * 3) % 128) as u32;
    }
    commands.extend_from_slice(&[1, 1, 8, 25, 4 << 26 | 8 << 10, 1 << 26, 0, 4096, 4096, 0, 0, 0]);
    let mut position = 4u32 << 26 | 8 << 10;
    for x in 1..=25 {
        let texel = (position >> 26) | ((position >> 4) & 0xfc0);
        expected[8 * width + x] = 255 - texel % 256;
        position = position.wrapping_add(1 << 26);
    }
    commands.extend_from_slice(&[0, 9, 2, 12, 1 << 16, 1 << 16, 0, 128, 4096, 4352, 0, 3]);
    for y in 2..=12 {
        for x in 9..=10 {
            expected[y * width + x] = 255 - ((1 + (y - 2) + 7) % 256) as u32;
        }
    }
    // Invalid resource offsets must never read outside the owned word buffer.
    commands.extend_from_slice(&[0, 0, 0, 16, 0, 0, u32::MAX, 128, u32::MAX, 0, 0, 0]);
    let bins = d3os_compute::raster::command_bins(&commands, width, height)?;
    let bin_count = bins.len();
    let bins = backend.upload_words(&bins)?;
    let base = backend.upload_words(&vec![3; width * height])?;
    let commands = backend.upload_words(&commands)?;
    let textures = backend.upload_words(&textures)?;
    let output = backend.allocate_words(width * height + 2)?;
    let guard = 0xd305abcd;
    backend.write_words(&output, 0, &[guard])?;
    backend.write_words(&output, width * height + 1, &[guard])?;
    backend.dispatch_words(
        Kernel::Raster,
        &[(&base, 0), (&commands, 0), (&textures, 0), (&output, 1)],
        &[width, height, 4, 4608],
    )?;
    let mut actual = vec![0; width * height + 2];
    backend.read_words_into(&output, 0, &mut actual)?;
    if actual[0] != guard || actual[width * height + 1] != guard || actual[1..actual.len() - 1] != expected {
        let mismatch = actual[1..actual.len() - 1].iter().zip(&expected).position(|(a, b)| a != b);
        if let Some(i) = mismatch {
            super::super::report(format_args!(
                "NVIDIA_RASTER_MISMATCH stage=commands index={i} actual={} expected={}",
                actual[i + 1],
                expected[i]
            ));
        }
        return Err(d3os_compute::Error::DeviceFault);
    }
    backend.dispatch_words(
        Kernel::RasterBinned,
        &[(&base, 0), (&commands, 0), (&textures, 0), (&output, 1), (&bins, 0)],
        &[width, height, 4, 4608, bin_count],
    )?;
    backend.read_words_into(&output, 0, &mut actual)?;
    if actual[0] != guard || actual[width * height + 1] != guard || actual[1..actual.len() - 1] != expected {
        let mismatch = actual[1..actual.len() - 1].iter().zip(&expected).position(|(a, b)| a != b);
        if let Some(i) = mismatch {
            super::super::report(format_args!(
                "NVIDIA_RASTER_MISMATCH stage=binned index={i} actual={} expected={}",
                actual[i + 1],
                expected[i]
            ));
        }
        return Err(d3os_compute::Error::DeviceFault);
    }
    let palette: Vec<_> = (0..256).map(|i| 0xff000000 | (i << 16) | ((255 - i) << 8) | i).collect();
    let palette_gpu = backend.upload_words(&palette)?;
    let rgba = backend.allocate_words(width * height * 4)?;
    backend.dispatch_words(Kernel::Palette, &[(&output, 1), (&palette_gpu, 0), (&rgba, 0)], &[width, height, 2])?;
    let mut actual = vec![0; width * height * 4];
    backend.read_words_into(&rgba, 0, &mut actual)?;
    for y in 0..height * 2 {
        for x in 0..width * 2 {
            if actual[y * width * 2 + x] != palette[expected[(y / 2) * width + x / 2] as usize] {
                return Err(d3os_compute::Error::DeviceFault);
            }
        }
    }
    Ok(())
}
