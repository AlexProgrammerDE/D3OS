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
