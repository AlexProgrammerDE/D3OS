use d3os_compute::{Backend, Error, Tensor, cpu::Cpu, transformer::TransformerBackend};

#[test]
fn borrowed_little_endian_weights_preserve_bits_and_reject_malformed_data() {
    let bits = [0u16, 0x8000, 1, 0x807f, 0x3f80, 0xbf80, 0x7f7f, 0x0080, 0x3fc0];
    for offset in 0..4 {
        let mut bytes = vec![0xd3; offset];
        bytes.extend(bits.into_iter().flat_map(u16::to_le_bytes));
        bytes.push(0xd3);
        let buffer = Cpu.upload_bf16_le(&bytes[offset..bytes.len() - 1]).unwrap();
        let tensor = Tensor::from_buffer(&[bits.len()], buffer).unwrap();
        assert_eq!(
            Cpu.download(&tensor).unwrap().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            bits.map(|value| u32::from(value) << 16)
        );
    }
    for bytes in [&[][..], &[0][..], &[0, 1, 2][..]] {
        assert!(Cpu.upload_bf16_le(bytes).is_err());
    }
    for invalid in [0x7f80u16, 0xff80, 0x7fc1] {
        let bytes: Vec<_> = [0x3f80, invalid, 0xbf80].into_iter().flat_map(u16::to_le_bytes).collect();
        assert!(matches!(Cpu.upload_bf16_le(&bytes), Err(Error::NonFinite)));
        #[cfg(feature = "nvidia")]
        assert!(matches!(d3os_compute::nvidia::Nvidia.upload_bf16_le(&bytes), Err(Error::NonFinite)));
    }
}

#[test]
fn fallback_preserves_bf16_bits_and_odd_embedding_rows() {
    let bits = [0, 0x8000, 1, 0x807f, 0x3f80, 0xbf80, 0x7f7f, 0x0080, 0x3fc0];
    let expected: Vec<_> = bits.iter().map(|&value| (value as u32) << 16).collect();
    let table = Tensor::from_buffer(&[3, 3], Cpu.upload_bf16(bits.into_iter()).unwrap()).unwrap();
    assert_eq!(Cpu.download(&table).unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected);
    for token in 0..3 {
        let row = Cpu.embedding(&table, token).unwrap();
        assert_eq!(
            Cpu.download(&row).unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected[token * 3..(token + 1) * 3]
        );
    }
    for invalid in [0x7f80, 0xff80, 0x7fc1] {
        assert!(matches!(Cpu.upload_bf16([invalid].into_iter()), Err(Error::NonFinite)));
    }
}
