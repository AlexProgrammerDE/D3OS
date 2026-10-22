use d3os_nvidia::{Error, firmware};

fn put(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn signed_image() -> Vec<u8> {
    let mut data = vec![0; 0x400];
    for (offset, value) in [
        (0, 0x10de),
        (4, 1),
        (8, 0x500),
        (12, 24),
        (16, 0x200),
        (20, 0x200),
        (24, 0x80),
        (28, 32),
        (32, 0x90),
        (36, 0x94),
        (48, 0x98),
        (52, 0xa0),
        (0x90, 4),
        (0x94, 0),
        (0x98, 1),
        (0xa8, 0x100),
        (0xac, 0x80),
        (0xb0, 1),
        (0xb4, 0x20),
        (0xb8, 0x40),
    ] {
        put(&mut data, offset, value);
    }
    data[0x80..0xa0].fill(0x5a);
    put(&mut data, 0x90, 4);
    put(&mut data, 0x94, 0);
    put(&mut data, 0x98, 1);
    data
}

#[test]
fn signed_booter_patches_the_selected_signature_inside_the_image() {
    let bytes = signed_image();
    let parsed = firmware::signed_booter(&bytes).unwrap();
    assert_eq!(&parsed.image[4..36], &bytes[0x80..0xa0]);
    assert_eq!(parsed.image.len(), 512);
    assert_eq!((parsed.code_offset, parsed.code_size), (0x20, 0x40));
    assert_eq!((parsed.data_offset, parsed.data_size), (0x100, 0x80));
}

#[test]
fn firmware_rejects_invalid_signature_tables_and_image_ranges() {
    let original = signed_image();
    for (offset, value) in [
        (0, 0),
        (4, 2),
        (16, u32::MAX),
        (20, u32::MAX),
        (0x90, 500),
        (0x98, 0),
        (0x98, 3),
        (0xb0, 2),
        (0xb4, 0x400),
        (0xb8, 0x400),
    ] {
        let mut bad = original.clone();
        put(&mut bad, offset, value);
        assert!(firmware::signed_booter(&bad).is_err());
    }
    for length in [0, 23, 60, 0x200, 0x3ff] {
        assert!(firmware::signed_booter(&original[..length]).is_err());
    }
    assert!(matches!(firmware::fwsec(&[0; 1024], 0), Err(Error::InvalidFirmware)));
}

fn put16(data: &mut [u8], offset: usize, value: u16) {
    data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn vbios() -> Vec<u8> {
    let mut data = vec![0; 0xc00];
    for (offset, signature, block_size, kind) in [(0, 0xaa55, 1, 0), (0x200, 0xaa55, 1, 3), (0x400, 0x4e56, 4, 0xe0)] {
        put16(&mut data, offset, signature);
        put16(&mut data, offset + 24, 0x40);
        data[offset + 0x40..offset + 0x44].copy_from_slice(if kind == 0xe0 { b"NPDS" } else { b"PCIR" });
        put16(&mut data, offset + 0x50, block_size);
        data[offset + 0x54] = kind;
    }
    data[0x100..0x106].copy_from_slice(b"\xff\xb8BIT\0");
    data[0x108..0x10b].copy_from_slice(&[12, 6, 1]);
    data[0x10c..0x10e].copy_from_slice(&[0x70, 2]);
    put16(&mut data, 0x10e, 4);
    put16(&mut data, 0x110, 0x180);
    put(&mut data, 0x180, 0x300);
    data[0x500..0x506].copy_from_slice(&[1, 6, 6, 1, 1, 44]);
    data[0x506] = 0x85;
    put(&mut data, 0x508, 0x400);
    for (offset, value) in [(0, (428 << 16) | 3), (4, 1024), (8, 256), (20, 256), (32, 768)] {
        put(&mut data, 0x600 + offset, value);
    }
    data[0x62c..0x7ac].fill(0xa7);
    let image = 0x7ac;
    data[image + 256..image + 260].copy_from_slice(&[1, 4, 8, 1]);
    put(&mut data, image + 260, 4);
    put(&mut data, image + 264, 16);
    put(&mut data, image + 272 + 8, 96);
    put(&mut data, image + 272 + 12, 44);
    data
}

#[test]
fn fwsec_supports_nv_extension_rom_and_six_byte_bit_tokens() {
    let original = vbios();
    let fwsec = firmware::fwsec(&original, 0x1234000).unwrap();
    assert_eq!(fwsec.image.len(), 1024);
    assert!(fwsec.image[512..896].iter().all(|byte| *byte == 0xa7));
    assert_eq!(&fwsec.image[272 + 44..272 + 48], &0x15u32.to_le_bytes());
    assert_eq!(&fwsec.image[352 + 32..352 + 36], &0x1234u32.to_le_bytes());
    assert!(original[0x7ac + 512..0x7ac + 896].iter().all(|byte| *byte == 0));
    for length in [0, 512, 0x600, 0x7ac, 0xbab] {
        assert!(firmware::fwsec(&original[..length], 0).is_err());
    }
    let mut corrupt = original;
    corrupt[0x110..0x112].fill(0xff);
    assert!(firmware::fwsec(&corrupt, 0).is_err());
}
