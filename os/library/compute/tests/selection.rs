use d3os_compute::{Backend, Error, Tensor, cpu::Cpu};

#[test]
fn argmax_uses_the_first_equal_value_and_rejects_nonfinite_losers() {
    for (values, expected) in [
        (vec![-9., -2., -2., -5.], 1),
        (vec![-0., 0., -0.], 0),
        (vec![f32::MIN, f32::MIN], 0),
        (vec![0., f32::from_bits(1), f32::from_bits(2)], 2),
    ] {
        let input = Tensor::from_vec(&Cpu, &[values.len()], values).unwrap();
        assert_eq!(Cpu.argmax(&input).unwrap(), expected);
    }
    for invalid in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
        let input = Tensor::from_vec(&Cpu, &[3], vec![1., invalid, 2.]).unwrap();
        assert!(matches!(Cpu.argmax(&input), Err(Error::NonFinite)));
    }
}
