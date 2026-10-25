//! Shared Burn training and deterministic progress, with native Adam checkpoints.
use super::{
    Mnist, artifact,
    data::{Dataset, SampleOrder},
};
#[cfg(feature = "native")]
use crate::native::{NativeDevice, random::RandomState};
use alloc::{format, string::String, vec::Vec};
use burn::{
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, ModuleOptimizer},
    tensor::Device,
};
#[cfg(feature = "native")]
use burn::tensor::Bytes;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub seed: u64,
    pub batch_size: usize,
    pub learning_rate: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 42,
            batch_size: 64,
            learning_rate: 0.001,
        }
    }
}
impl Config {
    fn validate(self) -> Result<(), String> {
        if self.batch_size == 0 || self.batch_size > 4096 || !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(String::from("invalid MNIST batch size or learning rate"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Progress {
    pub epoch: u32,
    pub offset: usize,
    pub steps: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub loss: f32,
    pub samples: usize,
    pub completed_epoch: Option<u32>,
}

pub struct Trainer {
    model: Mnist,
    optimizer: ModuleOptimizer,
    device: Device,
    config: Config,
    progress: Progress,
    order: SampleOrder,
    samples: usize,
    fingerprint: u32,
}

// Artifact version 1 fixes these Adam parameters independently of upstream defaults.
fn adam() -> ModuleOptimizer {
    AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-5)
        .with_amsgrad(false)
        .init()
}

impl Trainer {
    pub fn new(dataset: &Dataset<'_>, device: &Device, config: Config) -> Result<Self, String> {
        Self::initialize(dataset, device, config, || Ok(Mnist::new(device)))
    }

    /// Start with the artifact's weights, a fresh Adam state, and a fresh sample order.
    pub fn from_artifact(bytes: &[u8], dataset: &Dataset<'_>, device: &Device, config: Config) -> Result<Self, String> {
        Self::initialize(dataset, device, config, || artifact::load_trainable(bytes, device))
    }

    fn initialize(
        dataset: &Dataset<'_>,
        device: &Device,
        config: Config,
        model: impl FnOnce() -> Result<Mnist, String>,
    ) -> Result<Self, String> {
        config.validate()?;
        if !device.is_autodiff() {
            return Err(String::from("MNIST training requires an autodiff device"));
        }
        device.seed(config.seed);
        Ok(Self {
            model: model()?,
            optimizer: adam(),
            device: device.clone(),
            config,
            progress: Progress::default(),
            order: SampleOrder::new(dataset.len(), config.seed, 0),
            samples: dataset.len(),
            fingerprint: dataset.fingerprint(),
        })
    }
    pub fn model(&self) -> &Mnist {
        &self.model
    }
    pub fn config(&self) -> Config {
        self.config
    }
    /// Change only the learning rate, preserving moments, progress, and sample order.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<(), String> {
        let config = Config { learning_rate, ..self.config };
        config.validate()?;
        self.config = config;
        Ok(())
    }

    pub fn progress(&self) -> Progress {
        self.progress
    }

    pub fn step(mut self, dataset: &Dataset<'_>) -> Result<(Self, Step), String> {
        if dataset.len() != self.samples || dataset.fingerprint() != self.fingerprint {
            return Err(String::from("MNIST training dataset changed"));
        }
        let indices = self.order.batch(self.progress.offset, self.config.batch_size)?;
        let samples = indices.len();
        let (images, labels) = dataset.batch(indices, &self.device)?;
        let loss = CrossEntropyLossConfig::new().init(&self.device).forward(self.model.forward(images), labels);
        let gradients = GradientsParams::from_grads(loss.clone().backward(), &self.model);
        if gradients.len() != 8 {
            return Err(String::from("MNIST backward did not produce all eight parameter gradients"));
        }
        self.model = self.optimizer.step(self.config.learning_rate, self.model, gradients);
        self.device.sync().map_err(|e| format!("MNIST training execution: {e:?}"))?;
        let loss = loss.try_into_scalar::<f32>().map_err(|e| format!("MNIST loss read: {e:?}"))?;
        if !loss.is_finite() {
            return Err(String::from("MNIST training loss is not finite"));
        }
        self.progress.steps = self.progress.steps.checked_add(1).ok_or("MNIST step count exhausted")?;
        self.progress.offset += samples;
        let completed_epoch = if self.progress.offset == self.samples {
            self.progress.epoch = self.progress.epoch.checked_add(1).ok_or("MNIST epoch count exhausted")?;
            self.progress.offset = 0;
            self.order = SampleOrder::new(self.samples, self.config.seed, self.progress.epoch);
            Some(self.progress.epoch)
        } else {
            None
        };
        Ok((
            self,
            Step {
                loss,
                samples,
                completed_epoch,
            },
        ))
    }

    /// Save only at a completed step boundary, with no concurrent random users.
    #[cfg(feature = "native")]
    pub fn checkpoint(&self) -> Result<Vec<u8>, String> {
        self.device.sync().map_err(|e| format!("MNIST checkpoint boundary: {e:?}"))?;
        let weights = self
            .model
            .clone()
            .into_record()
            .into_bytes()
            .map_err(|e| format!("MNIST checkpoint weights: {e}"))?;
        let optimizer = self.optimizer.into_bytes().map_err(|e| format!("MNIST checkpoint Adam: {e}"))?;
        let mut metadata = Vec::new();
        for word in [
            self.config.seed,
            self.config.batch_size as u64,
            self.config.learning_rate.to_bits(),
            u64::from(self.progress.epoch),
            self.progress.offset as u64,
            self.progress.steps,
            self.samples as u64,
            u64::from(self.fingerprint),
            self.order.state(),
        ] {
            metadata.extend_from_slice(&word.to_le_bytes());
        }
        metadata.extend_from_slice(&NativeDevice.random_state().to_bytes());
        artifact::encode(artifact::CHECKPOINT, &metadata, &weights, &optimizer)
    }

    #[cfg(feature = "native")]
    pub fn restore(bytes: &[u8], dataset: &Dataset<'_>, device: &Device) -> Result<Self, String> {
        if !device.is_autodiff() {
            return Err(String::from("MNIST resume requires an autodiff device"));
        }
        let parts = artifact::decode(bytes, artifact::CHECKPOINT)?;
        if parts.metadata.len() != 96 || parts.optimizer.is_empty() {
            return Err(String::from("invalid MNIST training metadata"));
        }
        let mut words = [0u64; 9];
        for (value, raw) in words.iter_mut().zip(parts.metadata[..72].chunks_exact(8)) {
            *value = u64::from_le_bytes(raw.try_into().map_err(|_| "MNIST training word")?);
        }
        let config = Config {
            seed: words[0],
            batch_size: usize::try_from(words[1]).map_err(|_| "MNIST batch size overflow")?,
            learning_rate: f64::from_bits(words[2]),
        };
        config.validate()?;
        let progress = Progress {
            epoch: u32::try_from(words[3]).map_err(|_| "MNIST epoch overflow")?,
            offset: usize::try_from(words[4]).map_err(|_| "MNIST offset overflow")?,
            steps: words[5],
        };
        if words[6] != dataset.len() as u64
            || words[7] != u64::from(dataset.fingerprint())
            || progress.offset >= dataset.len()
            || !progress.offset.is_multiple_of(config.batch_size)
            || progress.steps
                != u64::from(progress.epoch)
                    .checked_mul((dataset.len().div_ceil(config.batch_size)) as u64)
                    .and_then(|n| n.checked_add((progress.offset / config.batch_size) as u64))
                    .ok_or("MNIST progress overflow")?
        {
            return Err(String::from("MNIST checkpoint dataset or progress mismatch"));
        }
        let random =
            RandomState::from_bytes(parts.metadata[72..96].try_into().map_err(|_| "MNIST random state")?).map_err(|e| format!("MNIST random state: {e}"))?;
        let order = SampleOrder::new(dataset.len(), config.seed, progress.epoch);
        if order.state() != words[8] {
            return Err(String::from("MNIST shuffle state mismatch"));
        }
        let model = artifact::load_weights(parts.weights, device)?;
        let optimizer = adam()
            .from_bytes_on_device(Bytes::from_bytes_vec(parts.optimizer.to_vec()), device)
            .map_err(|e| format!("MNIST Adam restore: {e}"))?;
        device.sync().map_err(|e| format!("MNIST checkpoint upload: {e:?}"))?;
        NativeDevice.restore_random_state(random);
        Ok(Self {
            model,
            optimizer,
            device: device.clone(),
            config,
            progress,
            order,
            samples: dataset.len(),
            fingerprint: dataset.fingerprint(),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Evaluation {
    pub samples: usize,
    pub correct: usize,
    pub predictions_crc32: u32,
}
impl Evaluation {
    pub fn accuracy(&self) -> f64 {
        self.correct as f64 / self.samples as f64
    }
}

/// Evaluation uses every item in the supplied split, including the final partial batch.
pub fn evaluate(model: &Mnist, dataset: &Dataset<'_>, batch_size: usize) -> Result<Evaluation, String> {
    if batch_size == 0 {
        return Err(String::from("MNIST evaluation batch size must be positive"));
    }
    let model = model.valid();
    let device = model.devices().into_iter().next().ok_or("MNIST model has no device")?;
    let mut correct = 0usize;
    let mut hasher = crc32fast::Hasher::new();
    for offset in (0..dataset.len()).step_by(batch_size) {
        let indices: Vec<_> = (offset..offset.saturating_add(batch_size).min(dataset.len())).collect();
        let (images, labels) = dataset.batch(&indices, &device)?;
        let predictions = model.forward(images).argmax(1).reshape([indices.len()]);
        let count = predictions
            .clone()
            .equal(labels)
            .int()
            .sum()
            .try_into_scalar::<i32>()
            .map_err(|e| format!("MNIST accuracy read: {e:?}"))?;
        let values = predictions.try_into_vec_as::<i32>().map_err(|e| format!("MNIST predictions read: {e:?}"))?;
        let expected = values
            .iter()
            .zip(&indices)
            .filter(|(value, index)| dataset.label(**index).is_some_and(|label| i32::from(label) == **value))
            .count();
        if usize::try_from(count).ok() != Some(expected) {
            return Err(String::from("MNIST GPU accuracy differs from the label reference"));
        }
        for value in values {
            hasher.update(&value.to_le_bytes());
        }
        correct += expected;
    }
    Ok(Evaluation {
        samples: dataset.len(),
        correct,
        predictions_crc32: hasher.finalize(),
    })
}
