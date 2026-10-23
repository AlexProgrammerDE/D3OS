//! Independent gradient and checkpoint checks that also run inside D3OS.
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
