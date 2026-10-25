//! Versioned MNIST artifacts containing named Burn weights and preprocessing.
use super::Mnist;
use alloc::{format, string::String, vec::Vec};
use burn::{
    module::Module,
    store::ModuleRecord,
    tensor::{Bytes, Device},
};

const TAG: &[u8; 8] = b"D3MNIST1";
const HEADER: usize = 32;
const MAX_BYTES: usize = 16 << 20;
const CONFIG: [u32; 9] = [1, 28, 28, 16, 32, 64, 10, 0.1307f32.to_bits(), 0.3081f32.to_bits()];
#[cfg(all(feature = "training", feature = "native"))]
pub(super) const CHECKPOINT: u32 = 1;
const INFERENCE: u32 = 2;

pub(super) struct Parts<'a> {
    pub metadata: &'a [u8],
    pub weights: &'a [u8],
    pub optimizer: &'a [u8],
}

pub(super) fn encode(kind: u32, metadata: &[u8], weights: &[u8], optimizer: &[u8]) -> Result<Vec<u8>, String> {
    let metadata_len = CONFIG.len() * 4 + metadata.len();
    let length = HEADER
        .checked_add(metadata_len)
        .and_then(|n| n.checked_add(weights.len()))
        .and_then(|n| n.checked_add(optimizer.len()))
        .and_then(|n| n.checked_add(4))
        .filter(|&n| n <= MAX_BYTES)
        .ok_or("MNIST artifact exceeds its size limit")?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|_| "cannot allocate MNIST artifact")?;
    bytes.extend_from_slice(TAG);
    bytes.extend_from_slice(&kind.to_le_bytes());
    bytes.extend_from_slice(&(metadata_len as u32).to_le_bytes());
    bytes.extend_from_slice(&(weights.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(optimizer.len() as u64).to_le_bytes());
    for field in CONFIG {
        bytes.extend_from_slice(&field.to_le_bytes());
    }
    bytes.extend_from_slice(metadata);
    bytes.extend_from_slice(weights);
    bytes.extend_from_slice(optimizer);
    bytes.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
    Ok(bytes)
}

pub(super) fn decode(bytes: &[u8], kind: u32) -> Result<Parts<'_>, String> {
    if bytes.len() < HEADER + CONFIG.len() * 4 + 4
        || bytes.len() > MAX_BYTES
        || &bytes[..8] != TAG
        || u32::from_le_bytes(bytes[8..12].try_into().map_err(|_| "MNIST kind")?) != kind
    {
        return Err(String::from("invalid MNIST artifact header or kind"));
    }
    let metadata = u32::from_le_bytes(bytes[12..16].try_into().map_err(|_| "MNIST metadata length")?) as usize;
    let length = |start| -> Result<usize, String> {
        usize::try_from(u64::from_le_bytes(bytes[start..start + 8].try_into().map_err(|_| "MNIST record length")?))
            .map_err(|_| String::from("MNIST record size overflow"))
    };
    let weights = length(16)?;
    let optimizer = length(24)?;
    if metadata < CONFIG.len() * 4
        || HEADER
            .checked_add(metadata)
            .and_then(|n| n.checked_add(weights))
            .and_then(|n| n.checked_add(optimizer))
            .and_then(|n| n.checked_add(4))
            != Some(bytes.len())
        || crc32fast::hash(&bytes[..bytes.len() - 4]) != u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().map_err(|_| "MNIST checksum")?)
    {
        return Err(String::from("invalid MNIST artifact length or checksum"));
    }
    for (field, expected) in CONFIG.into_iter().enumerate() {
        let offset = HEADER + field * 4;
        if u32::from_le_bytes(bytes[offset..offset + 4].try_into().map_err(|_| "MNIST configuration")?) != expected {
            return Err(String::from("unsupported MNIST architecture or normalization"));
        }
    }
    Ok(Parts {
        metadata: &bytes[HEADER + CONFIG.len() * 4..HEADER + metadata],
        weights: &bytes[HEADER + metadata..HEADER + metadata + weights],
        optimizer: &bytes[HEADER + metadata + weights..bytes.len() - 4],
    })
}

pub(super) fn load_weights(weights: &[u8], device: &Device) -> Result<Mnist, String> {
    let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(weights.to_vec())).map_err(|e| format!("MNIST weight record: {e}"))?;
    if record.len() != 8 {
        return Err(String::from("MNIST requires all eight parameter tensors"));
    }
    Mnist::new(device).try_load_record(record).map_err(|e| format!("MNIST weight shapes: {e}"))
}

/// Export the trained model for the same Burn inference API used in the guest.
pub fn export(model: &Mnist) -> Result<Vec<u8>, String> {
    let weights = model.clone().into_record().into_bytes().map_err(|e| format!("MNIST weights: {e}"))?;
    encode(INFERENCE, &[], &weights, &[])
}

fn inference_parts(bytes: &[u8]) -> Result<Parts<'_>, String> {
    let parts = decode(bytes, INFERENCE)?;
    if !parts.metadata.is_empty() || !parts.optimizer.is_empty() {
        return Err(String::from("inference artifact contains training state"));
    }
    Ok(parts)
}

pub fn load(bytes: &[u8], device: &Device) -> Result<Mnist, String> {
    load_weights(inference_parts(bytes)?.weights, &device.clone().without_autodiff()).map(|model| model.valid())
}

#[cfg(feature = "training")]
pub(super) fn load_trainable(bytes: &[u8], device: &Device) -> Result<Mnist, String> {
    load_weights(inference_parts(bytes)?.weights, device)
}
