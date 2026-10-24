//! Portable inference checks and prediction ranking over Burn tensors.
use alloc::{format, string::String, vec::Vec};
use burn::backend::ops::{ConvOptions, MaxPoolOptions};
use burn::tensor::{
    Device, Tensor,
    activation::{relu, softmax},
    module::{conv2d, max_pool2d},
};

pub fn top_k(probabilities: &[f32], count: usize) -> Result<Vec<(usize, f32)>, String> {
    if probabilities.iter().any(|value| !value.is_finite()) {
        return Err(String::from("prediction contains a nonfinite value"));
    }
    let mut ranked: Vec<_> = probabilities.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(count);
    Ok(ranked)
}

/// Run a small convolutional network with independently calculated logits.
pub fn self_test(device: &Device) -> Result<(), String> {
    let input = Tensor::<4>::from_floats([[[[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]], device);
    let weights = Tensor::<4>::from_floats([[[[1., 0.], [0., -1.]]]], device);
    let bias = Tensor::<1>::from_floats([6.], device);
    let hidden = relu(conv2d(input, weights, Some(bias), ConvOptions::new([1; 2], [0; 2], [1; 2], 1)));
    let pooled = max_pool2d(hidden, MaxPoolOptions::new([2; 2]).with_stride([1; 2])).reshape([1, 4]);
    let weights = Tensor::<2>::from_floats([[1., -1., 0.5], [2., -2., 0.5], [3., -3., 0.5], [4., -4., 0.5]], device);
    let logits = pooled.matmul(weights) + Tensor::<1>::from_floats([0.5, -0.5, 1.], device).unsqueeze::<2>();
    let actual = logits.clone().try_into_vec_as::<f32>().map_err(|e| format!("Burn self-test logits: {e:?}"))?;
    crate::qwen::verify(&actual, &[10.5, -10.5, 3.], 1e-6, 1e-6)?;
    let probabilities = softmax(logits, 1)
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("Burn self-test probabilities: {e:?}"))?;
    if top_k(&probabilities, 1)?.first().map(|(index, _)| *index) != Some(0) {
        return Err(String::from("Burn self-test selected the wrong class"));
    }
    Ok(())
}
