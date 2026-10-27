//! Independent gradient and checkpoint checks that also run inside D3OS.
pub mod capabilities;
pub mod compaction;
pub mod cross;
pub mod deform;
pub mod erf;
pub mod float_arithmetic;
pub mod float_extrema;
pub mod float_reduction;
pub mod float_rounding;
pub mod indexed_accumulation;
pub mod integer;
pub mod integer_arithmetic;
pub mod integer_matrix;
pub mod integer_reduction;
pub mod interpolate;
pub mod inverse;
pub mod mixed_matmul;
pub mod movement;
pub mod nd_indexing;
pub mod predicate;
pub mod quantization;
pub mod quantized_empty;
pub mod qwen;
pub mod qwen_state;
pub mod relu;
pub mod remainder;
#[cfg(target_os = "none")]
pub mod running_state;
pub mod scan;
pub mod scatter_updates;
pub mod sort;
pub mod spatial_convolution;
pub mod storage;
pub mod typed_erf;
pub mod typed_exp_log;
pub mod typed_power;
pub mod typed_remainder;
pub mod typed_trig;
pub mod typed_unary;
use alloc::{format, string::String};
use burn::{
    module::{Module, Param},
    optim::{AdamConfig, GradientsParams, ModuleOptimizer, SgdConfig},
    store::ModuleRecord,
    tensor::{Device, Tensor},
};

#[derive(Module, Debug)]
struct Regression {
    weight: Param<Tensor<2>>,
    bias: Param<Tensor<1>>,
}

impl Regression {
    fn new(device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::from_floats([[0.0], [0.0]], device)),
            bias: Param::from_tensor(Tensor::from_floats([0.0], device)),
        }
    }

    fn predict(&self, device: &Device) -> Tensor<2> {
        Tensor::from_floats([[1.0, 2.0], [-1.0, 1.0], [2.0, -1.0]], device).matmul(self.weight.val()) + self.bias.val().unsqueeze::<2>()
    }

    fn loss(&self, device: &Device) -> Tensor<1> {
        let error = self.predict(device) - Tensor::from_floats([[1.0], [-1.0], [2.0]], device);
        (error.clone() * error).mean()
    }
}

fn check<const D: usize>(actual: Tensor<D>, expected: &[f32]) -> Result<(), String> {
    let values = actual.try_into_vec_as::<f32>().map_err(|e| format!("Burn tensor read: {e:?}"))?;
    if values.len() != expected.len() || values.iter().zip(expected).any(|(a, b)| !a.is_finite() || (a - b).abs() > 2e-6) {
        return Err(format!("Burn numerical reference: {values:?} != {expected:?}"));
    }
    Ok(())
}

fn step(model: Regression, optimizer: &mut ModuleOptimizer, device: &Device) -> Regression {
    let grads = GradientsParams::from_grads(model.loss(device).backward(), &model);
    optimizer.step(0.05, model, grads)
}

/// Check analytic derivatives, SGD, tensor aliasing, and stateful Adam resume.
/// The caller chooses the execution device explicitly.
pub fn training(device: &Device) -> Result<(), String> {
    #[cfg(target_os = "none")]
    crate::platform::validate_timers()?;
    if !device.is_autodiff() {
        return Err(String::from("Burn training validation requires autodiff"));
    }
    let model = Regression::new(device);
    let retained_weight = model.weight.val();
    check(model.loss(device), &[2.0])?;
    let grads = model.loss(device).backward();
    check(model.weight.val().grad(&grads).ok_or("missing weight gradient")?, &[-4.0, 2.0 / 3.0])?;
    check(model.bias.val().grad(&grads).ok_or("missing bias gradient")?, &[-4.0 / 3.0])?;
    let mut sgd = SgdConfig::new().init();
    let updated = sgd.step(0.05, model.clone(), GradientsParams::from_grads(grads, &model));
    check(updated.weight.val(), &[0.2, -1.0 / 30.0])?;
    check(updated.bias.val(), &[1.0 / 15.0])?;
    check(retained_weight, &[0.0, 0.0])?;

    let mut optimizer = AdamConfig::new().init();
    let mut uninterrupted = model;
    for _ in 0..5 {
        uninterrupted = step(uninterrupted, &mut optimizer, device);
    }
    let weights = uninterrupted
        .clone()
        .into_record()
        .into_bytes()
        .map_err(|e| format!("Burn checkpoint weights: {e:?}"))?;
    let state = optimizer.into_bytes().map_err(|e| format!("Burn checkpoint optimizer: {e:?}"))?;
    let record = ModuleRecord::from_bytes(weights).map_err(|e| format!("Burn checkpoint decode: {e:?}"))?;
    let mut resumed = Regression::new(device)
        .try_load_record(record)
        .map_err(|e| format!("Burn checkpoint load: {e:?}"))?;
    let mut resumed_optimizer = AdamConfig::new().init().from_bytes(state).map_err(|e| format!("Burn optimizer load: {e:?}"))?;
    for _ in 0..7 {
        uninterrupted = step(uninterrupted, &mut optimizer, device);
        resumed = step(resumed, &mut resumed_optimizer, device);
        let expected = uninterrupted
            .predict(device)
            .try_into_vec_as::<f32>()
            .map_err(|e| format!("Burn reference read: {e:?}"))?;
        check(resumed.predict(device), &expected)?;
    }
    Ok(())
}

/// Validate complete convolutional Adam checkpoints across a partial batch and
/// epoch boundary, including device random state and inference artifacts.
#[cfg(feature = "native")]
pub fn mnist_checkpoint(device: &Device) -> Result<(), String> {
    use crate::{
        mnist::{
            artifact,
            data::Dataset,
            training::{Config, Trainer},
        },
        native::NativeDevice,
    };
    use alloc::vec::Vec;
    let mut images = Vec::new();
    for word in [2051u32, 5, 28, 28] {
        images.extend_from_slice(&word.to_be_bytes());
    }
    images.extend((0..5 * 784).map(|i| (i * 37 + i / 784 * 79) as u8));
    let mut labels = Vec::new();
    for word in [2049u32, 5] {
        labels.extend_from_slice(&word.to_be_bytes());
    }
    labels.extend_from_slice(&[0, 1, 2, 3, 4]);
    let data = Dataset::parse(&images, &labels)?;
    let mut original = Trainer::new(
        &data,
        device,
        Config {
            batch_size: 2,
            ..Config::default()
        },
    )?;
    for _ in 0..2 {
        original = original.step(&data)?.0;
    }
    let checkpoint = original.checkpoint()?;
    let random = NativeDevice.random_state();
    NativeDevice.seed(0x0bad_5eed);
    let mut resumed = Trainer::restore(&checkpoint, &data, device)?;
    if NativeDevice.random_state() != random || resumed.progress() != original.progress() {
        return Err(String::from("MNIST checkpoint random state or progress changed"));
    }
    for _ in 0..3 {
        let (next, expected) = original.step(&data)?;
        original = next;
        let (next, actual) = resumed.step(&data)?;
        resumed = next;
        if original.progress() != resumed.progress()
            || expected.samples != actual.samples
            || expected.completed_epoch != actual.completed_epoch
            || (expected.loss - actual.loss).abs() > 2e-6
        {
            return Err(format!(
                "MNIST Adam resume differs: expected_loss={} actual_loss={}",
                expected.loss, actual.loss
            ));
        }
    }
    if (resumed.progress().epoch, resumed.progress().offset, resumed.progress().steps) != (1, 4, 5) {
        return Err(String::from("MNIST resumed epoch or partial batch changed"));
    }
    let plain = device.clone().without_autodiff();
    let (input, _) = data.batch(&[0, 1, 2, 3, 4], &plain)?;
    let expected = original
        .model()
        .valid()
        .forward(input.clone())
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("MNIST checkpoint reference: {e:?}"))?;
    check(resumed.model().valid().forward(input.clone()), &expected)?;
    let bytes = artifact::export(resumed.model())?;
    let reloaded = artifact::load(&bytes, &plain)?;
    check(reloaded.forward(input), &expected)?;
    Ok(())
}

pub mod conversions;

pub mod indexed;
