//! Model execution shared by the native D3OS application and host tools.
#![no_std]

extern crate alloc;

pub mod alexnet;
pub mod archive;
pub mod qwen;
pub mod tokenizer;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use d3os_compute::{Backend, Conv2d, Pool2d, Tensor};

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Compute(d3os_compute::Error),
    InvalidArchive,
    MissingTensor(String),
    UnexpectedTensor,
    InvalidModelShape,
    VerificationFailed,
    InvalidConfig,
    InvalidToken,
    InvalidTokenizer,
    ContextFull,
    SessionFailed,
    VerificationAt { position: usize, layer: usize },
}

impl From<d3os_compute::Error> for Error {
    fn from(value: d3os_compute::Error) -> Self {
        Self::Compute(value)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compute(error) => error.fmt(f),
            Self::InvalidArchive => f.write_str("invalid or unsupported tensor archive"),
            Self::MissingTensor(name) => write!(f, "model is missing tensor {name}"),
            Self::UnexpectedTensor => f.write_str("model contains unexpected tensors"),
            Self::InvalidModelShape => f.write_str("model tensor has an invalid shape"),
            Self::VerificationAt { position, layer } => write!(f, "Qwen reference mismatch at position {position}, layer {layer}"),
            Self::InvalidConfig => f.write_str("invalid Qwen model configuration"),
            Self::InvalidToken => f.write_str("invalid token ID or generation length"),
            Self::InvalidTokenizer => f.write_str("invalid or unsupported tokenizer"),
            Self::ContextFull => f.write_str("sequence exceeds the session capacity"),
            Self::SessionFailed => f.write_str("session failed; create a new session before continuing"),
            Self::VerificationFailed => f.write_str("inference output differs from the reference"),
        }
    }
}

impl core::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

/// Return class indices and probabilities, ordered by descending probability.
pub fn top_k(probabilities: &[f32], count: usize) -> Result<Vec<(usize, f32)>> {
    if probabilities.iter().any(|value| !value.is_finite()) {
        return Err(d3os_compute::Error::NonFinite.into());
    }
    let mut ranked: Vec<_> = probabilities.iter().copied().enumerate().collect();
    // Stable sorting preserves the class order for equal probabilities.
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(count);
    Ok(ranked)
}

/// A small convolutional network with independently calculated exact logits.
/// This runs the same backend interface that AlexNet uses, without model files.
pub fn self_test<B: Backend>(backend: &B) -> Result<()> {
    let input = Tensor::from_vec(
        backend,
        &[1, 1, 4, 4],
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.],
    )?;
    let weights = Tensor::from_vec(backend, &[1, 1, 2, 2], vec![1., 0., 0., -1.])?;
    let bias = Tensor::from_vec(backend, &[1], vec![6.])?;
    // Each 2x2 window has top-left minus bottom-right = -5. Bias makes it 1.
    let hidden = backend.relu(backend.conv2d(&input, &weights, &bias, Conv2d::default())?)?;
    let pooled = backend
        .max_pool2d(
            &hidden,
            Pool2d {
                kernel: [2, 2],
                stride: [1, 1],
            },
        )?
        .reshape(&[1, 4])?;
    let weights = Tensor::from_vec(backend, &[3, 4], vec![1., 2., 3., 4., -1., -2., -3., -4., 0.5, 0.5, 0.5, 0.5])?;
    let bias = Tensor::from_vec(backend, &[3], vec![0.5, -0.5, 1.])?;
    let logits = backend.linear(&pooled, &weights, &bias)?;
    verify(&backend.download(&logits)?, &[10.5, -10.5, 3.], 1e-6, 1e-6)?;
    let probabilities = backend.download(&backend.softmax(&logits)?)?;
    if top_k(&probabilities, 1)?.first().map(|(class, _)| *class) != Some(0) {
        return Err(Error::VerificationFailed);
    }
    Ok(())
}

/// Execute pretrained AlexNet and compare all eight observed layers with a fixture.
pub fn verify_alexnet<B: Backend>(backend: &B, model: &[u8], fixture: &[u8]) -> Result<Vec<(usize, f32)>> {
    let model = alexnet::AlexNet::load(backend, model)?;
    let mut reference = archive::Archive::parse(fixture)?;
    let input = reference.take("input")?.load(backend)?;
    let logits = model.forward(backend, input, |name, tensor| {
        let record = reference.take(name)?;
        if record.shape() != tensor.shape() {
            return Err(Error::InvalidModelShape);
        }
        verify(&backend.download(tensor)?, &record.to_vec()?, 2e-4, 2e-4)
    })?;
    reference.finish()?;
    top_k(&backend.download(&backend.softmax(&logits)?)?, 5)
}

/// Check every value, including low-magnitude logits. NaNs never pass.
pub fn verify(actual: &[f32], expected: &[f32], absolute: f32, relative: f32) -> Result<()> {
    if actual.len() != expected.len() || actual.is_empty() || !absolute.is_finite() || !relative.is_finite() || absolute < 0. || relative < 0. {
        return Err(Error::VerificationFailed);
    }
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() || (actual - expected).abs() > absolute + relative * expected.abs() {
            return Err(Error::VerificationFailed);
        }
    }
    Ok(())
}
