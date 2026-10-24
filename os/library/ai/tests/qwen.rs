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

#[test]
fn owned_qwen_state_matches_independent_binary64_sequences_and_preserves_snapshots() {
    d3os_ai::validation::qwen_state::run(&Device::flex()).unwrap();
}

#[test]
fn cpu_packed_float_views_honor_transformed_parameter_layouts() {
    use d3os_ai::burn::{
        module::{Module, ModuleMapper, Param},
        tensor::Int,
    };
    struct Reverse;
    impl ModuleMapper for Reverse {
        fn map_int<const D: usize>(&mut self, param: Param<Tensor<D, Int>>) -> Param<Tensor<D, Int>> {
            param.map(|tensor| tensor.flip([0]))
        }
    }
    let device = Device::flex();
    let bytes: Vec<_> = [1f32, 2., 3., 4., 5., 6.].into_iter().flat_map(f32::to_le_bytes).collect();
    let original = PackedMatrix::from_bytes([2, 3], MatrixFormat::F32, &bytes, &device).unwrap();
    let reversed = original.clone().map(&mut Reverse);
    let input = Tensor::<2>::from_floats([[1., 2., 3.]], &device);
    let actual = reversed.project(input.clone()).unwrap().try_into_vec_as::<f32>().unwrap();
    assert_eq!(actual, [28., 10.]);
    assert_eq!(original.project(input).unwrap().try_into_vec_as::<f32>().unwrap(), [14., 32.]);
    assert_eq!(reversed.embedding(&[1, 0]).unwrap().try_into_vec_as::<f32>().unwrap(), [3., 2., 1., 6., 5., 4.]);
}

#[test]
fn qwen_state_rejects_invalid_geometry_and_autodiff_devices() {
    use d3os_ai::qwen::{AttentionConfig, AttentionState, DeltaConfig, DeltaState};
    let device = Device::flex();
    let attention = AttentionConfig {
        heads: 2,
        kv_heads: 1,
        head_dim: 4,
        rotary_dim: 2,
        rope_theta: 10000.,
        epsilon: 1e-6,
    };
    let delta = DeltaConfig {
        key_heads: 1,
        heads: 2,
        key_dim: 4,
        value_dim: 3,
        kernel: 3,
        epsilon: 1e-6,
    };
    for invalid in [
        AttentionConfig { heads: 0, ..attention },
        AttentionConfig { kv_heads: 0, ..attention },
        AttentionConfig { rotary_dim: 3, ..attention },
        AttentionConfig {
            epsilon: f32::NAN,
            ..attention
        },
    ] {
        assert!(AttentionState::new(invalid, 4, &device).is_err());
    }
    for invalid in [
        DeltaConfig { key_heads: 0, ..delta },
        DeltaConfig { heads: 0, ..delta },
        DeltaConfig { key_heads: 3, ..delta },
        DeltaConfig { kernel: 0, ..delta },
        DeltaConfig { epsilon: -1., ..delta },
    ] {
        assert!(DeltaState::new(invalid, &device).is_err());
    }
    assert!(AttentionState::from_tensors(Tensor::zeros([4, 1, 4], &device), Tensor::zeros([4, 2, 4], &device), 0, attention).is_err());
    assert!(AttentionState::from_tensors(Tensor::zeros([4, 1, 4], &device), Tensor::zeros([4, 1, 4], &device), 5, attention).is_err());
    assert!(DeltaState::from_tensors(Tensor::zeros([14, 3], &device), Tensor::zeros([2, 4, 4], &device), delta).is_err());
    let autodiff = device.autodiff();
    assert!(AttentionState::new(attention, 4, &autodiff).is_err());
    assert!(DeltaState::new(delta, &autodiff).is_err());
}
