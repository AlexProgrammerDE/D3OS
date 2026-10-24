//! A convolutional classifier for normalized NCHW MNIST images.
pub mod artifact;
pub mod data;
#[cfg(all(feature = "training", feature = "native"))]
pub mod training;

use burn::{
    module::Module,
    nn::{
        Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    tensor::{Device, Tensor, activation::relu},
};

/// Two convolutions and two dense layers, shared by training and inference.
#[derive(Module, Debug)]
pub struct Mnist {
    convolution1: Conv2d,
    convolution2: Conv2d,
    pool: MaxPool2d,
    hidden: Linear,
    output: Linear,
}

impl Mnist {
    pub fn new(device: &Device) -> Self {
        Self {
            convolution1: Conv2dConfig::new([1, 16], [5, 5]).init(device),
            convolution2: Conv2dConfig::new([16, 32], [5, 5]).init(device),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            hidden: LinearConfig::new(32 * 4 * 4, 64).init(device),
            output: LinearConfig::new(64, 10).init(device),
        }
    }

    /// Input shape is [batch, 1, 28, 28]. Output contains ten logits per image.
    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        let images = self.pool.forward(relu(self.convolution1.forward(images)));
        let images = self.pool.forward(relu(self.convolution2.forward(images)));
        let images = images.flatten(1, 3);
        self.output.forward(relu(self.hidden.forward(images)))
    }
}

/// The same normalization applies during dataset loading and inference.
pub fn normalize(pixel: u8) -> f32 {
    (f32::from(pixel) / 255.0 - 0.1307) / 0.3081
}
