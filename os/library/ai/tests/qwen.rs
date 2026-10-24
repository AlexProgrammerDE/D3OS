#![cfg(all(feature = "cpu", feature = "training"))]
use d3os_ai::{
    burn::tensor::{Device, Tensor},
    qwen::{MatrixFormat, PackedMatrix},
};

#[test]
fn packed_qwen_matches_scalar_and_independent_gguf_references_on_burn_cpu() {
    d3os_ai::validation::qwen::run(&Device::flex()).unwrap();
}

#[test]
fn packed_weights_reject_corruption_and_autodiff_without_creating_a_model() {
    let device = Device::flex();
    for (shape, format, data) in [
        ([1, 1], MatrixFormat::F32, vec![0, 0, 128, 127]),
        ([1, 1], MatrixFormat::Bf16, vec![128, 127]),
        ([1, 2], MatrixFormat::Bf16, vec![0, 0, 0]),
        ([0, 2], MatrixFormat::F32, vec![]),
        ([2, 16], MatrixFormat::Quantized(d3os_ai::qwen::Quantization::Q8_0), vec![0; 34]),
    ] {
        assert!(PackedMatrix::from_bytes(shape, format, &data, &device).is_err());
    }
    assert!(PackedMatrix::from_bytes([1, 1], MatrixFormat::F32, &1f32.to_le_bytes(), &device.clone().autodiff()).is_err());
    let matrix = PackedMatrix::from_bytes([1, 1], MatrixFormat::F32, &1f32.to_le_bytes(), &device).unwrap();
    assert!(matrix.project(Tensor::<2>::ones([1, 1], &device.clone().autodiff())).is_err());
}
