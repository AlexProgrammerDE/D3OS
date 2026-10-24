use d3os_compute::{Backend, Tensor, cpu::Cpu};
use d3os_tensor_file::{Error, quantized::Quantization};

#[test]
fn checkpoint_blocks_match_independent_gguf_reference_bits() {
    let mut data = &include_bytes!("fixtures/quantized.bin")[8..];
    for _ in 0..5 {
        let format = Quantization::from_raw(u32::from_le_bytes(data[..4].try_into().unwrap())).unwrap();
        let elements = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let bytes = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let raw = &data[12..12 + bytes];
        let expected = &data[12 + bytes..12 + bytes + elements * 4];
        let tensor = Tensor::from_buffer(&[elements], Cpu.upload_quantized(format, elements, raw).unwrap()).unwrap();
        for (actual, expected) in Cpu.values(&tensor).iter().zip(expected.chunks_exact(4)) {
            assert_eq!(actual.to_bits(), u32::from_le_bytes(expected.try_into().unwrap()), "{format:?}");
        }
        data = &data[12 + bytes + elements * 4..];
    }
    assert!(data.is_empty());
}

#[test]
fn packed_shapes_and_nonfinite_scales_are_rejected() {
    for format in [Quantization::Q8_0, Quantization::Q3K, Quantization::Q4K, Quantization::Q5K, Quantization::Q6K] {
        let n = format.block_elements();
        let mut bytes = vec![0; format.block_bytes()];
        assert!(format.validate(n, &bytes).is_ok());
        assert!(format.validate(n, &bytes[..bytes.len() - 1]).is_err());
        assert!(format.byte_len(n - 1).is_err());
        assert!(format.byte_len(usize::MAX).is_err());
        let scale = match format {
            Quantization::Q3K => 108,
            Quantization::Q6K => 208,
            _ => 0,
        };
        for bits in [0x7c00u16, 0xfc00, 0x7e01, 0xfe01] {
            bytes[scale..scale + 2].copy_from_slice(&bits.to_le_bytes());
            assert_eq!(format.validate(n, &bytes), Err(Error::NonFinite));
        }
        if matches!(format, Quantization::Q4K | Quantization::Q5K) {
            bytes[scale..scale + 2].fill(0);
            bytes[2..4].copy_from_slice(&0x7c00u16.to_le_bytes());
            assert_eq!(format.validate(n, &bytes), Err(Error::NonFinite));
        }
    }
}

#[test]
fn q8_half_scales_preserve_subnormal_sign_and_extremes() {
    for (scale, bits) in [
        (0u16, 0u32),
        (0x8000, 0x8000_0000),
        (1, 0x3380_0000),
        (0x3ff, 0x387f_c000),
        (0x400, 0x3880_0000),
        (0x7bff, 0x477f_e000),
    ] {
        let mut bytes = vec![1; 34];
        bytes[..2].copy_from_slice(&scale.to_le_bytes());
        let values = Quantization::Q8_0.decode(32, &bytes).unwrap();
        assert!(values.iter().all(|value| value.to_bits() == bits));
    }
}
