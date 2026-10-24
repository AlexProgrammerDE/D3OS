//! Torchvision AlexNet expressed with Burn modules and public tensors.
use alloc::{format, string::String, vec::Vec};
use burn::{
    module::{Module, Param},
    nn::{
        Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    tensor::{
        Device, Tensor, TensorData,
        activation::{relu, softmax},
    },
};
use d3os_tensor_file::{Archive, Record};

type Result<T> = core::result::Result<T, String>;

/// Canonical checkpoint names and shapes, checked before device allocation.
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

/// Observe resident activations without rank erasure or automatic readback.
pub trait Observer {
    fn observe<const D: usize>(&mut self, name: &'static str, tensor: &Tensor<D>) -> Result<()>;
}
struct Unobserved;
impl Observer for Unobserved {
    fn observe<const D: usize>(&mut self, _: &'static str, _: &Tensor<D>) -> Result<()> {
        Ok(())
    }
}

#[derive(Module, Debug)]
pub struct AlexNet {
    convolutions: Vec<Conv2d>,
    classifier: Vec<Linear>,
    pool: MaxPool2d,
}

fn tensor<const D: usize>(record: &Record<'_>, device: &Device) -> Result<Tensor<D>> {
    let values = record.to_vec().map_err(|e| format!("AlexNet tensor data: {e}"))?;
    Ok(Tensor::from_data(TensorData::new(values, record.shape()), device))
}

impl AlexNet {
    pub fn load(device: &Device, bytes: &[u8]) -> Result<Self> {
        let mut archive = Archive::parse(bytes).map_err(|e| format!("AlexNet model: {e}"))?;
        let mut records = Vec::with_capacity(PARAMETERS.len());
        for (name, shape) in PARAMETERS {
            let record = archive.take(name).map_err(|e| format!("AlexNet model: {e}"))?;
            if record.shape() != shape {
                return Err(format!("AlexNet parameter {name} has shape {:?}, expected {shape:?}", record.shape()));
            }
            records.push(record);
        }
        archive.finish().map_err(|e| format!("AlexNet model: {e}"))?;
        let mut convolutions = Vec::with_capacity(5);
        let mut classifier = Vec::with_capacity(3);
        for (index, pair) in records.chunks_exact(2).enumerate() {
            let shape = pair[0].shape();
            let bias = Some(Param::from_tensor(tensor::<1>(&pair[1], device)?));
            if index < 5 {
                let padding = if index < 2 { 2 } else { 1 };
                let mut layer = Conv2dConfig::new([shape[1], shape[0]], [shape[2], shape[3]])
                    .with_stride(if index == 0 { [4, 4] } else { [1, 1] })
                    .with_padding(PaddingConfig2d::Explicit(padding, padding, padding, padding))
                    .init(device);
                layer.weight = Param::from_tensor(tensor::<4>(&pair[0], device)?);
                layer.bias = bias;
                convolutions.push(layer);
            } else {
                let mut layer = LinearConfig::new(shape[1], shape[0]).init(device);
                // Torch stores [out, in]. Materialize Burn's [in, out] layout
                // once during loading; forwards reuse the resident matrix.
                let weight = tensor::<2>(&pair[0], device)?.transpose().reshape([shape[1], shape[0]]);
                layer.weight = Param::from_tensor(weight);
                layer.bias = bias;
                classifier.push(layer);
            }
        }
        device.sync().map_err(|e| format!("AlexNet model upload: {e:?}"))?;
        Ok(Self {
            convolutions,
            classifier,
            pool: MaxPool2dConfig::new([3, 3]).with_strides([2, 2]).init(),
        }
        .valid())
    }

    pub fn forward(&self, input: Tensor<4>) -> Result<Tensor<2>> {
        self.forward_observed(input, &mut Unobserved)
    }

    pub fn forward_observed(&self, mut input: Tensor<4>, observer: &mut impl Observer) -> Result<Tensor<2>> {
        let [batch, channels, height, width] = input.dims();
        if batch == 0 || channels != 3 || height != 224 || width != 224 {
            return Err(String::from("AlexNet input must have shape [batch, 3, 224, 224]"));
        }
        for (index, layer) in self.convolutions.iter().enumerate() {
            input = relu(layer.forward(input));
            observer.observe(["features.0", "features.3", "features.6", "features.8", "features.10"][index], &input)?;
            if matches!(index, 0 | 1 | 4) {
                input = self.pool.forward(input);
            }
        }
        // The final 224x224 pool output is already 6x6.
        let mut input = input.reshape([batch, 9216]);
        for (index, layer) in self.classifier.iter().enumerate() {
            input = layer.forward(input);
            if index < 2 {
                input = relu(input);
            }
            observer.observe(["classifier.1", "classifier.4", "logits"][index], &input)?;
        }
        Ok(input)
    }
}

struct Reference<'a> {
    archive: Archive<'a>,
}
impl Observer for Reference<'_> {
    fn observe<const D: usize>(&mut self, name: &'static str, tensor: &Tensor<D>) -> Result<()> {
        let record = self.archive.take(name).map_err(|e| format!("AlexNet fixture: {e}"))?;
        if record.shape() != tensor.dims() {
            return Err(format!("AlexNet reference shape differs at {name}"));
        }
        let actual = tensor.clone().try_into_vec_as::<f32>().map_err(|e| format!("AlexNet layer {name}: {e:?}"))?;
        let expected = record.to_vec().map_err(|e| format!("AlexNet reference values: {e}"))?;
        if actual.len() != expected.len() {
            return Err(format!("AlexNet reference length differs at {name}"));
        }
        for (index, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > 2e-4 + 2e-4 * b.abs() {
                return Err(format!("AlexNet reference mismatch at {name}[{index}]: actual={a}, expected={b}"));
            }
        }
        Ok(())
    }
}

/// Compare every value of all eight original layer references at their original tolerance.
pub fn verify(device: &Device, weights: &[u8], fixture: &[u8]) -> Result<Vec<(usize, f32)>> {
    let model = AlexNet::load(device, weights)?;
    verify_model(&model, device, fixture)
}

pub fn verify_model(model: &AlexNet, device: &Device, fixture: &[u8]) -> Result<Vec<(usize, f32)>> {
    let mut reference = Reference {
        archive: Archive::parse(fixture).map_err(|e| format!("AlexNet fixture: {e}"))?,
    };
    let record = reference.archive.take("input").map_err(|e| format!("AlexNet fixture input: {e}"))?;
    if record.shape() != [1, 3, 224, 224] {
        return Err(String::from("AlexNet fixture input shape differs"));
    }
    let input = tensor::<4>(&record, device)?;
    let logits = model.forward_observed(input, &mut reference)?;
    reference.archive.finish().map_err(|e| format!("AlexNet fixture: {e}"))?;
    let probabilities = softmax(logits, 1)
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("AlexNet probabilities: {e:?}"))?;
    if probabilities.iter().any(|v| !v.is_finite()) {
        return Err(String::from("AlexNet probabilities are not finite"));
    }
    let mut ranked: Vec<_> = probabilities.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(5);
    Ok(ranked)
}
