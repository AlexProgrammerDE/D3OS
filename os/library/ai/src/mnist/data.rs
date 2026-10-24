//! Validated IDX data and deterministic training sample order.
use alloc::{format, string::String, vec::Vec};
use burn::tensor::{Device, Int, Tensor, TensorData};

use super::normalize;

pub const PIXELS: usize = 28 * 28;

/// Borrowed MNIST images and labels, validated before tensor creation.
#[derive(Clone, Debug)]
pub struct Dataset<'a> {
    images: &'a [u8],
    labels: &'a [u8],
    fingerprint: u32,
}

impl<'a> Dataset<'a> {
    pub fn parse(images: &'a [u8], labels: &'a [u8]) -> Result<Self, String> {
        let word = |data: &[u8], offset| -> Result<usize, String> {
            let bytes = data.get(offset..offset + 4).ok_or("truncated IDX header")?;
            Ok(u32::from_be_bytes(bytes.try_into().map_err(|_| "invalid IDX header")?) as usize)
        };
        let count = word(images, 4)?;
        if word(images, 0)? != 2051
            || word(labels, 0)? != 2049
            || count == 0
            || word(labels, 4)? != count
            || word(images, 8)? != 28
            || word(images, 12)? != 28
            || count.checked_mul(PIXELS).and_then(|n| n.checked_add(16)) != Some(images.len())
            || count.checked_add(8) != Some(labels.len())
            || labels[8..].iter().any(|&label| label >= 10)
        {
            return Err(String::from("invalid MNIST dimensions, labels, or IDX length"));
        }
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(images);
        hasher.update(labels);
        Ok(Self {
            images: &images[16..],
            labels: &labels[8..],
            fingerprint: hasher.finalize(),
        })
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
    pub fn label(&self, index: usize) -> Option<u8> {
        self.labels.get(index).copied()
    }

    pub fn fingerprint(&self) -> u32 {
        self.fingerprint
    }

    /// Decode and normalize input data. All model operations run on the chosen device.
    pub fn batch(&self, indices: &[usize], device: &Device) -> Result<(Tensor<4>, Tensor<1, Int>), String> {
        if indices.is_empty() || indices.iter().any(|&i| i >= self.len()) {
            return Err(String::from("MNIST batch indices are empty or out of bounds"));
        }
        let count = indices.len().checked_mul(PIXELS).ok_or("MNIST batch size overflow")?;
        let mut pixels = Vec::new();
        pixels.try_reserve_exact(count).map_err(|_| "cannot allocate MNIST input batch")?;
        let mut labels = Vec::with_capacity(indices.len());
        for &index in indices {
            pixels.extend(self.images[index * PIXELS..(index + 1) * PIXELS].iter().copied().map(normalize));
            labels.push(i32::from(self.labels[index]));
        }
        Ok((
            Tensor::from_data(TensorData::new(pixels, [indices.len(), 1, 28, 28]), device),
            Tensor::from_data(TensorData::new(labels, [indices.len()]), device),
        ))
    }
}

/// Reconstructible Fisher-Yates order for one epoch. No sample order is implicit.
#[derive(Clone, Debug)]
pub struct SampleOrder {
    indices: Vec<usize>,
    state: u64,
}

impl SampleOrder {
    pub fn new(count: usize, seed: u64, epoch: u32) -> Self {
        let mut indices: Vec<_> = (0..count).collect();
        let mut state = seed.wrapping_add(u64::from(epoch).wrapping_mul(0xd134_2543_de82_ef95));
        for i in (1..count).rev() {
            let width = i as u64 + 1;
            let threshold = width.wrapping_neg() % width;
            let chosen = loop {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut value = state;
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                value ^= value >> 31;
                if value >= threshold {
                    break (value % width) as usize;
                }
            };
            indices.swap(i, chosen);
        }
        Self { indices, state }
    }
    pub fn state(&self) -> u64 {
        self.state
    }
    pub fn batch(&self, offset: usize, batch_size: usize) -> Result<&[usize], String> {
        if batch_size == 0 || offset >= self.indices.len() {
            return Err(format!("invalid MNIST batch at {offset}"));
        }
        Ok(&self.indices[offset..offset.saturating_add(batch_size).min(self.indices.len())])
    }
}
