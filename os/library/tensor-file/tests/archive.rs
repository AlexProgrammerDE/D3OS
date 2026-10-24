use d3os_tensor_file::Archive;

fn archive(records: &[(&str, &[u64], &[f32])]) -> Vec<u8> {
    let mut data = b"D3TENSR\0".to_vec();
    data.extend(1u32.to_le_bytes());
    data.extend((records.len() as u32).to_le_bytes());
    for &(name, shape, values) in records {
        data.extend((name.len() as u16).to_le_bytes());
        data.extend((shape.len() as u16).to_le_bytes());
        data.extend((values.len() as u64).to_le_bytes());
        data.extend(name.bytes());
        for dim in shape {
            data.extend(dim.to_le_bytes());
        }
        for value in values {
            data.extend(value.to_le_bytes());
        }
    }
    data
}

#[test]
fn archive_decodes_endianness_and_rejects_every_truncated_prefix() {
    let data = archive(&[("input", &[1, 3], &[1.25, -2.5, 0.125])]);
    for length in 0..data.len() {
        assert!(Archive::parse(&data[..length]).is_err());
    }
    let mut parsed = Archive::parse(&data).unwrap();
    let input = parsed.take("input").unwrap();
    parsed.finish().unwrap();
    assert_eq!(input.shape(), [1, 3]);
    assert_eq!(input.to_vec().unwrap(), [1.25, -2.5, 0.125]);
}

#[test]
fn archive_rejects_duplicate_names_trailing_bytes_and_mismatched_shapes() {
    assert!(Archive::parse(&archive(&[("x", &[1], &[1.]), ("x", &[1], &[2.])])).is_err());
    assert!(Archive::parse(&archive(&[("x", &[2], &[1.])])).is_err());
    assert!(Archive::parse(&archive(&[("x", &[u64::MAX, 2], &[1.])])).is_err());
    let mut data = archive(&[("x", &[1], &[1.])]);
    data.push(0);
    assert!(Archive::parse(&data).is_err());
    let data = archive(&[("x", &[1], &[f32::NAN])]);
    assert!(Archive::parse(&data).unwrap().take("x").unwrap().to_vec().is_err());
}

#[test]
fn typed_archive_decodes_bfloat16_and_rejects_truncation_and_nonfinite_weights() {
    let mut bytes = b"D3TENSR\0".to_vec();
    bytes.extend(2u32.to_le_bytes());
    bytes.extend(1u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(3u64.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.push(b'x');
    bytes.extend(3u64.to_le_bytes());
    for value in [0x3fc0u16, 0xc020, 0x3e00] {
        bytes.extend(value.to_le_bytes());
    }
    for length in 0..bytes.len() {
        assert!(Archive::parse(&bytes[..length]).is_err());
    }
    assert_eq!(Archive::parse(&bytes).unwrap().take("x").unwrap().to_vec().unwrap(), [1.5, -2.5, 0.125]);
    let len = bytes.len();
    bytes[len - 2..].copy_from_slice(&0x7fc0u16.to_le_bytes());
    assert!(Archive::parse(&bytes).unwrap().take("x").unwrap().to_vec().is_err());
    bytes[28..30].copy_from_slice(&2u16.to_le_bytes());
    assert!(Archive::parse(&bytes).is_err());
}
