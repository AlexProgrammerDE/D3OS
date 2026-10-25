//! Checked strided indexing metadata. Device index values need a runtime bounds check.
use crate::{Error, Result};

pub const PARAMETERS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Scatter {
    FloatAdd = 0,
    IntAdd = 1,
    BoolOr = 2,
    FloatAssign = 3,
    IntAssign = 4,
    FloatMultiply = 3 | (1 << 8),
    IntMultiply = 4 | (1 << 8),
    FloatMin = 3 | (2 << 8),
    IntMin = 4 | (2 << 8),
    FloatMax = 3 | (3 << 8),
    IntMax = 4 | (3 << 8),
}

impl Scatter {
    pub const ALL: [Self; 11] = [
        Self::FloatAdd,
        Self::IntAdd,
        Self::BoolOr,
        Self::FloatAssign,
        Self::IntAssign,
        Self::FloatMultiply,
        Self::IntMultiply,
        Self::FloatMin,
        Self::IntMin,
        Self::FloatMax,
        Self::IntMax,
    ];
    pub const ENTRIES: [Self; 5] = [Self::FloatAdd, Self::IntAdd, Self::BoolOr, Self::FloatAssign, Self::IntAssign];

    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.iter().find(|&&op| op as u32 == raw).copied().ok_or(Error::InvalidField)
    }

    pub fn ordered(self) -> bool {
        self as u32 & 0xff >= 3
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::FloatAdd => "burn_scatter_add_f32",
            Self::IntAdd => "burn_scatter_add_i32",
            Self::BoolOr => "burn_scatter_or_bool",
            Self::FloatAssign | Self::FloatMultiply | Self::FloatMin | Self::FloatMax => "burn_scatter_ordered_f32",
            Self::IntAssign | Self::IntMultiply | Self::IntMin | Self::IntMax => "burn_scatter_ordered_i32",
        }
    }
}

/// Scatter shares gather geometry, with the operation in the rank word's high
/// half. Values are dense and indices can have signed strides. Atomic modes
/// accept signed destination strides; ordered modes require a dense destination.
pub fn validate_scatter(p: &[u32]) -> Result<[u64; 4]> {
    let mut p: [u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let operation = Scatter::from_raw(p[0] >> 16)?;
    p[0] &= 0xffff;
    let [destination, indices, status, values] = validate(&p)?;
    if operation.ordered() {
        // One lane owns each output row. Dense destination strides prove that
        // separate rows cannot race, including when indices are duplicated.
        if p[7] == 0 {
            return Err(Error::InvalidField);
        }
        let mut stride = 1u32;
        for axis in (0..p[0] as usize).rev() {
            if p[16 + axis] != stride {
                return Err(Error::InvalidField);
            }
            stride = stride
                .checked_mul(if axis == p[1] as usize { p[7] } else { p[8 + axis] })
                .ok_or(Error::InvalidField)?;
        }
    }
    Ok([values, indices, status, destination])
}

/// rank, axis, count, source extent/offset, index extent/offset, axis length,
/// eight index dimensions, eight source strides, and eight index strides.
pub fn validate(p: &[u32]) -> Result<[u64; 4]> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let rank = p[0] as usize;
    let axis = p[1] as usize;
    if !(1..=8).contains(&rank)
        || axis >= rank
        || p[2] == 0
        || p[2] > i32::MAX as u32
        || p[7] > i32::MAX as u32
        || p[8..16].iter().any(|&d| d == 0 || d > i32::MAX as u32)
        || p[8 + rank..16].iter().any(|&d| d != 1)
        || p[16 + rank..24].iter().chain(&p[24 + rank..32]).any(|&s| s != 0)
    {
        return Err(Error::InvalidField);
    }
    let count = p[8..16].iter().try_fold(1u32, |n, &d| n.checked_mul(d)).ok_or(Error::InvalidField)?;
    if count != p[2] {
        return Err(Error::InvalidField);
    }
    for (input, extent, offset, strides) in [(0, p[3], p[4], &p[16..24]), (1, p[5], p[6], &p[24..32])] {
        if extent == 0 || extent > i32::MAX as u32 || offset > extent {
            return Err(Error::InvalidField);
        }
        if input == 0 && p[7] == 0 {
            continue;
        }
        let mut low = i64::from(offset);
        let mut high = low;
        for (a, (&d, &s)) in p[8..16].iter().zip(strides).enumerate() {
            let dim = if input == 0 && a == axis { p[7] } else { d };
            let span = i64::from(dim - 1) * i64::from(s as i32);
            low = low.checked_add(span.min(0)).ok_or(Error::InvalidField)?;
            high = high.checked_add(span.max(0)).ok_or(Error::InvalidField)?;
        }
        if low < 0 || high >= i64::from(extent) {
            return Err(Error::InvalidField);
        }
    }
    Ok([u64::from(p[3]), u64::from(p[5]), 1, u64::from(p[2])])
}
