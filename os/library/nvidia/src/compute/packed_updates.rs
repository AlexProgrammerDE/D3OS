//! Packed stored-value updates with word bounds and disjoint logical writes.
use super::{elementwise, indexing, storage::Width, updates};
use crate::{Error, Result};
pub const PARAMETERS: usize = 32;
pub const MASK_U8: u32 = 1 << 8;
pub const SCALAR: u32 = 1 << 9;
pub const REMAINDER_SHIFT: u32 = 10;
fn width(raw: u32) -> Result<Width> {
    Width::from_raw(raw)
}
/// Assignment retains the ordinary two-view geometry, with width in word two.
pub fn validate_assign(p: &[u32]) -> Result<[u64; 2]> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let width = width(p[2])?;
    let mut base = *p;
    base[2] = 0;
    let sizes = updates::validate_assign(&base)?;
    Ok(sizes.map(|n| width.words(n as u32)))
}
/// Mask geometry retains two data views. Scalars use p3/p5 as their low/high
/// words and canonical unused right geometry. A dense mask pointer can start
/// at a byte offset inside its first word, encoded in flag bits 10..11.
pub fn validate_mask(p: &[u32]) -> Result<[u64; 4]> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let width = width(p[2] & 3)?;
    let scalar = p[2] & SCALAR != 0;
    let packed_mask = p[2] & MASK_U8 != 0;
    let remainder = p[2] >> REMAINDER_SHIFT & 3;
    if p[2] & !0xf03 != 0 || (!packed_mask && remainder != 0) || (!scalar && p[3] != 0) {
        return Err(Error::InvalidField);
    }
    let mut base = *p;
    base[2] = 0;
    base[3] = 0;
    if scalar {
        let bits = u64::from(p[3]) | u64::from(p[5]) << 32;
        if p[7] != 0 || p[24..32].iter().any(|&v| v != 0) || (width != Width::Eight && bits >= 1u64 << (8 * width.bytes())) {
            return Err(Error::InvalidField);
        }
        base[5] = 0;
        base[7] = 1;
    }
    let [a, b, count] = elementwise::validate_layout(&base)?;
    let mask_width = if packed_mask { Width::One } else { Width::Four };
    Ok([
        width.words(a as u32),
        if scalar { 1 } else { width.words(b as u32) },
        mask_width.words((count as u32).checked_add(remainder).ok_or(Error::InvalidField)?),
        width.words(count as u32),
    ])
}
pub fn mask_work_items(p: &[u32]) -> Result<u32> {
    let sizes = validate_mask(p)?;
    if width(p[2] & 3)? == Width::Eight {
        Ok(p[0])
    } else {
        u32::try_from(sizes[3]).map_err(|_| Error::InvalidField)
    }
}
/// Indexed boolean OR uses gather geometry. Rank occupies bits 0..7, storage
/// width bits 8..9, and the dense values' initial byte offset bits 10..11.
pub fn validate_bool_scatter(p: &[u32]) -> Result<[u64; 4]> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let width = width(p[0] >> 8 & 3)?;
    let remainder = p[0] >> REMAINDER_SHIFT & 3;
    if p[0] & !0xfff != 0 || !matches!(width, Width::One | Width::Four) || (width == Width::Four && remainder != 0) {
        return Err(Error::InvalidField);
    }
    let mut base = *p;
    base[0] &= 0xff;
    let [destination, indices, status, values] = indexing::validate(&base)?;
    Ok([
        width.words((values as u32).checked_add(remainder).ok_or(Error::InvalidField)?),
        indices,
        status,
        width.words(destination as u32),
    ])
}
