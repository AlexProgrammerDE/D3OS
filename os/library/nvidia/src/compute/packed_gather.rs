//! Bit-preserving gather with logical element geometry and physical word bounds.
use super::{indexing, storage::Width};
use crate::{Error, Result};

pub const PARAMETERS: usize = indexing::PARAMETERS;

fn width(p: &[u32; PARAMETERS]) -> Result<Width> {
    // Decoding the entire upper portion also rejects every reserved flag.
    Width::from_raw(p[0] >> 8)
}

/// Gather's 32-word ABI, with rank in bits 0..8 and storage width in bits 8..10.
/// Source strides and extents count logical values; indices remain I32 words.
/// Empty outputs bypass dispatch, as they do for ordinary gather.
pub fn validate(p: &[u32]) -> Result<[u64; 4]> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let width = width(p)?;
    let mut geometry = *p;
    geometry[0] &= 0xff;
    let [_, indices, status, _] = indexing::validate(&geometry)?;
    Ok([width.words(p[3]), indices, status, width.words(p[2])])
}

/// Narrow lanes own whole output words. A 64-bit lane owns both of its words.
pub fn work_items(p: &[u32]) -> Result<u32> {
    let extents = validate(p)?;
    if p[0] >> 8 == Width::Eight as u32 {
        Ok(p[2])
    } else {
        u32::try_from(extents[3]).map_err(|_| Error::InvalidField)
    }
}
