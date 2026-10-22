//! Torchvision AlexNet in evaluation mode for one 224x224 RGB image.
use alloc::vec::Vec;
use d3os_compute::{Backend, Conv2d, Pool2d, Tensor};

use crate::{Error, Result, archive::Archive};

// Canonical torchvision parameter names. Dropout has no effect in evaluation mode.
pub const PARAMETERS: [(&str, &[usize]); 16] = [
    ("features.0.weight", &[64, 3, 11, 11]),
    ("features.0.bias", &[64]),
    ("features.3.weight", &[192, 64, 5, 5]),
    ("features.3.bias", &[192]),
    ("features.6.weight", &[384, 192, 3, 3]),
    ("features.6.bias", &[384]),
    ("features.8.weight", &[256, 384, 3, 3]),
    ("features.8.bias", &[256]),
    ("features.10.weight", &[256, 256, 3, 3]),
    ("features.10.bias", &[256]),
    ("classifier.1.weight", &[4096, 9216]),
    ("classifier.1.bias", &[4096]),
    ("classifier.4.weight", &[4096, 4096]),
    ("classifier.4.bias", &[4096]),
    ("classifier.6.weight", &[1000, 4096]),
    ("classifier.6.bias", &[1000]),
];

struct Layer<B: Backend> {
    weights: Tensor<B>,
    bias: Tensor<B>,
}

pub struct AlexNet<B: Backend> {
    convolutions: Vec<Layer<B>>,
    classifier: Vec<Layer<B>>,
}

impl<B: Backend> AlexNet<B> {
    pub fn load(backend: &B, bytes: &[u8]) -> Result<Self> {
        let mut archive = Archive::parse(bytes)?;
        let mut convolutions = Vec::with_capacity(5);
        let mut classifier = Vec::with_capacity(3);
        // Validate every shape before allocating the large weight buffers.
        let mut records = Vec::with_capacity(PARAMETERS.len());
        for (name, shape) in PARAMETERS {
            let record = archive.take(name)?;
            if record.shape() != shape {
                return Err(Error::InvalidModelShape);
            }
            records.push(record);
        }
        archive.finish()?;
        let mut records = records.into_iter();
        for index in 0..8 {
            let layer = Layer {
                weights: records.next().ok_or(Error::InvalidArchive)?.load(backend)?,
                bias: records.next().ok_or(Error::InvalidArchive)?.load(backend)?,
            };
            if index < 5 {
                convolutions.push(layer);
            } else {
                classifier.push(layer);
            }
        }
        Ok(Self { convolutions, classifier })
    }

    /// The observer receives intermediate tensors for layer-by-layer verification.
    pub fn forward(&self, backend: &B, mut input: Tensor<B>, mut observe: impl FnMut(&'static str, &Tensor<B>) -> Result<()>) -> Result<Tensor<B>> {
        if input.shape() != [1, 3, 224, 224] {
            return Err(Error::InvalidModelShape);
        }
        const NAMES: [&str; 5] = ["features.0", "features.3", "features.6", "features.8", "features.10"];
        for (index, layer) in self.convolutions.iter().enumerate() {
            let config = match index {
                0 => Conv2d {
                    stride: [4, 4],
                    padding: [2, 2],
                },
                1 => Conv2d {
                    padding: [2, 2],
                    ..Conv2d::default()
                },
                _ => Conv2d {
                    padding: [1, 1],
                    ..Conv2d::default()
                },
            };
            input = backend.relu(backend.conv2d(&input, &layer.weights, &layer.bias, config)?)?;
            observe(NAMES[index], &input)?;
            if matches!(index, 0 | 1 | 4) {
                input = backend.max_pool2d(
                    &input,
                    Pool2d {
                        kernel: [3, 3],
                        stride: [2, 2],
                    },
                )?;
            }
        }
        // At 224x224, the last pool already produces 6x6, so adaptive pooling is an identity.
        input = input.reshape(&[1, 9216])?;
        for (index, layer) in self.classifier.iter().enumerate() {
            input = backend.linear(&input, &layer.weights, &layer.bias)?;
            if index < 2 {
                input = backend.relu(input)?;
            }
            observe(["classifier.1", "classifier.4", "logits"][index], &input)?;
        }
        Ok(input)
    }
}
