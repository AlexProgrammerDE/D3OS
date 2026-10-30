use d3os_tensor_file::{
    Error,
    matrix::{MatrixFormat, MatrixLayout},
    quantized::Quantization,
};

#[test]
fn tiled_q8_preserves_each_complete_block_and_round_trips_without_padding() {
    let format = MatrixFormat::Quantized(Quantization::Q8_0);
    for shape in [[32, 32], [64, 96], [96, 1056]] {
        let blocks = shape[1] / 32;
        let mut source = vec![0u8; format.byte_len(shape).unwrap()];
        for (index, block) in source.chunks_exact_mut(34).enumerate() {
            let scale = [0x0000u16, 0x8000, 0x3c00, 0xbc00, 0x0001][index % 5];
            block[..2].copy_from_slice(&scale.to_le_bytes());
            for (lane, value) in block[2..].iter_mut().enumerate() {
                *value = (index * 37 + lane * 13) as u8;
            }
        }
        let tiled = MatrixLayout::Q8Columns32.pack(shape, format, &source).unwrap();
        assert_eq!(tiled.len(), source.len());
        let mut seen = vec![false; source.len() / 34];
        for row in 0..shape[0] {
            for block in 0..blocks {
                let physical = (row / 32) * (32 * blocks) + block * 32 + row % 32;
                assert!(!seen[physical]);
                seen[physical] = true;
                assert_eq!(
                    &tiled[physical * 34..(physical + 1) * 34],
                    &source[(row * blocks + block) * 34..(row * blocks + block + 1) * 34]
                );
            }
        }
        assert!(seen.into_iter().all(|value| value));
        let restored = MatrixLayout::Q8Columns32.row_major_bytes(shape, format, &tiled).unwrap();
        assert_eq!(&*restored, source);
        let expected = format.decode(shape, &source).unwrap();
        let actual = format.decode(shape, &restored).unwrap();
        assert!(actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

#[test]
fn tiled_storage_rejects_wrong_formats_geometry_and_incomplete_encodings() {
    let layout = MatrixLayout::Q8Columns32;
    let q8 = MatrixFormat::Quantized(Quantization::Q8_0);
    for (shape, format) in [
        ([32, 32], MatrixFormat::Bf16),
        ([32, 32], MatrixFormat::Quantized(Quantization::Q4K)),
        ([33, 32], q8),
        ([32, 33], q8),
        ([0, 32], q8),
    ] {
        assert!(layout.byte_len(shape, format).is_err());
    }
    let bytes = vec![0u8; q8.byte_len([32, 32]).unwrap()];
    assert_eq!(layout.pack([32, 32], q8, &bytes[..bytes.len() - 1]).unwrap_err(), Error::ShapeMismatch);
    assert_eq!(
        layout.row_major_bytes([32, 32], q8, &bytes[..bytes.len() - 1]).unwrap_err(),
        Error::ShapeMismatch
    );
    let mut nonfinite = bytes;
    nonfinite[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert_eq!(layout.pack([32, 32], q8, &nonfinite).unwrap_err(), Error::NonFinite);
}

#[test]
fn padded_q3_preserves_file_bits_and_ignores_padding_on_restore() {
    let layout = MatrixLayout::Q3Padded112;
    let format = MatrixFormat::Quantized(Quantization::Q3K);
    for shape in [[1, 256], [3, 768], [32, 5120]] {
        let mut source = vec![0u8; format.byte_len(shape).unwrap()];
        for (index, block) in source.chunks_exact_mut(110).enumerate() {
            for (byte, value) in block[..108].iter_mut().enumerate() {
                *value = (index * 37 + byte * 13) as u8;
            }
            let scale = [0x0000u16, 0x8000, 0x3c00, 0xbc00, 0x0001][index % 5];
            block[108..].copy_from_slice(&scale.to_le_bytes());
        }
        let mut packed = layout.pack(shape, format, &source).unwrap();
        assert_eq!(packed.len(), source.len() / 110 * 112);
        for (raw, padded) in source.chunks_exact(110).zip(packed.chunks_exact(112)) {
            assert_eq!(&padded[..110], raw);
            assert_eq!(&padded[110..], &[0, 0]);
        }
        for pattern in [0xa5, 0x5a] {
            for block in packed.chunks_exact_mut(112) {
                block[110..].fill(pattern);
            }
            let restored = layout.row_major_bytes(shape, format, &packed).unwrap();
            assert_eq!(&*restored, &source);
            assert!(
                format
                    .decode(shape, &restored)
                    .unwrap()
                    .iter()
                    .zip(format.decode(shape, &source).unwrap())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
}

#[test]
fn padded_q3_rejects_invalid_geometry_lengths_and_scales() {
    let layout = MatrixLayout::Q3Padded112;
    let format = MatrixFormat::Quantized(Quantization::Q3K);
    for (shape, format) in [
        ([1, 256], MatrixFormat::F32),
        ([1, 256], MatrixFormat::Quantized(Quantization::Q4K)),
        ([0, 256], format),
        ([1, 255], format),
    ] {
        assert!(layout.byte_len(shape, format).is_err());
    }
    assert_eq!(layout.byte_len([usize::MAX, 256], format).unwrap_err(), Error::SizeOverflow);
    assert!(layout.pack([1, 256], format, &[0; 109]).is_err());
    assert!(layout.row_major_bytes([1, 256], format, &[0; 110]).is_err());
    let mut nonfinite = [0; 110];
    nonfinite[108..].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert_eq!(layout.pack([1, 256], format, &nonfinite).unwrap_err(), Error::NonFinite);
}
