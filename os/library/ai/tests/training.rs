#![cfg(all(feature = "cpu", feature = "training"))]

use d3os_ai::{burn::tensor::Device, validation};

#[test]
fn analytic_gradients_and_adam_resume() {
    let device = Device::flex().autodiff();
    device.seed(42);
    validation::training(&device).unwrap();
}
