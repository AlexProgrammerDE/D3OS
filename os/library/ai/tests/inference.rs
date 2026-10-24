#![cfg(feature = "cpu")]
use d3os_ai::burn::tensor::{DType, Device, Tensor, TensorData};
use d3os_ai::{
    qwen::{Error, greedy, verify},
    self_test, top_k,
};

#[test]
fn native_self_test_exercises_the_complete_small_network() {
    self_test(&Device::flex()).unwrap();
}

#[test]
fn verification_rejects_nonfinite_values_wrong_lengths_and_excess_error() {
    assert_eq!(verify(&[f32::NAN], &[0.], 1e-3, 1e-3), Err(Error::VerificationFailed));
    assert!(verify(&[1.], &[1., 2.], 0., 0.).is_err());
    assert!(verify(&[1.01], &[1.], 0.001, 0.001).is_err());
    assert!(verify(&[1.0001, 0.00001], &[1., 0.], 0.0001, 0.0001).is_ok());
    assert!(verify(&[1.], &[1.], f32::NAN, 0.).is_err());
}

#[test]
fn class_ranking_is_stable_and_limits_output() {
    assert_eq!(top_k(&[0.2, 0.4, 0.4], 2).unwrap(), [(1, 0.4), (2, 0.4)]);
    assert!(top_k(&[f32::NAN], 1).is_err());
}

#[test]
fn greedy_selection_rejects_nonfinite_logits_and_keeps_the_first_equal_maximum() {
    let device = Device::flex();
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(greedy(&Tensor::<2>::from_floats([[1., invalid, 2.]], &device)), Err(Error::NonFinite)));
    }
    assert_eq!(greedy(&Tensor::<2>::from_floats([[0.2, 0.4, 0.4]], &device)).unwrap(), 1);
}

#[test]
fn greedy_selection_preserves_cpu_precision_and_logical_view_order() {
    let device = Device::flex();
    let values = Tensor::<2>::from_data(
        TensorData::new(vec![1f64, 1.0000000000000002, 1.0000000000000002], [1, 3]),
        (&device, DType::F64),
    );
    assert_eq!(greedy(&values).unwrap(), 1);
    assert_eq!(greedy(&values.clone().flip([1])).unwrap(), 0);
    assert_eq!(greedy(&values).unwrap(), 1);
    #[cfg(feature = "training")]
    {
        let device = device.autodiff();
        let input = Tensor::<2>::from_floats([[1., 3., 3.]], &device).require_grad();
        assert_eq!(greedy(&input).unwrap(), 1);
        let gradients = input.clone().sum().backward();
        assert_eq!(input.grad(&gradients).unwrap().into_data().as_slice::<f32>().unwrap(), &[1.; 3]);
    }
}
