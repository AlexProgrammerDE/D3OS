use d3os_compute::{Backend, Error, Tensor, cpu::Cpu, transformer::TransformerBackend};

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
