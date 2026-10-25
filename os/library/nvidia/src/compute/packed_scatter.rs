//! Ordered indexed assignment preserves packed payloads and neighboring fields.
use super::{indexing, storage::Width};
use crate::{Error, Result};

pub const PARAMETERS: usize = indexing::PARAMETERS;

/// Gather geometry with rank in bits 0..8 and width in bits 8..10.
/// Values are dense from logical offset zero. Destination geometry must be
/// dense, so each lane exclusively owns one row excluding the indexed axis.
pub fn validate(p: &[u32]) -> Result<[u64; 4]> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let width = Width::from_raw(p[0] >> 8)?;
    let mut geometry = *p;
    geometry[0] = (p[0] & 0xff) | ((indexing::Scatter::FloatAssign as u32) << 16);
    let [_, indices, status, _] = indexing::validate_scatter(&geometry)?;
    Ok([width.words(p[2]), indices, status, width.words(p[3])])
}

pub fn work_items(p: &[u32]) -> Result<u32> {
    validate(p)?;
    Ok(p[2] / p[8 + p[1] as usize])
}
