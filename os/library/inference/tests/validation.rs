use d3os_compute::cpu::Cpu;
use d3os_inference::{Error, self_test, top_k, verify};

#[test]
fn native_self_test_exercises_the_complete_small_network() {
    self_test(&Cpu).unwrap();
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
