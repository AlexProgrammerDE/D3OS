//! Typed indexed folds over initialized, private dense output storage.
use super::{cast::Format, storage::Width};
use crate::{Error, Result};

pub const PARAMETERS: usize = 32;
pub const THREADS: u32 = 128;
pub const SYMBOL: &str = "burn_indexed_accumulation";
pub const FORMATS: [Format; 13] = [
    Format::F64,
    Format::F32,
    Format::Flex32,
    Format::F16,
    Format::BF16,
    Format::I64,
    Format::I32,
    Format::I16,
    Format::I8,
    Format::U64,
    Format::U32,
    Format::U16,
    Format::U8,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Operation {
    Add = 0,
    Mul = 1,
    Min = 2,
    Max = 3,
}

impl Operation {
    pub const ALL: [Self; 4] = [Self::Add, Self::Mul, Self::Min, Self::Max];

    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.get(raw as usize).copied().ok_or(Error::InvalidField)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub values_words: u64,
    pub indices_words: u64,
    pub status_words: u64,
    pub output_words: u64,
    pub owners: u32,
}

/// Rank occupies bits 0..7, format bits 8..15, and operation bits 16..23.
pub fn control(rank: u32, format: Format, operation: Operation) -> Result<u32> {
    if !(1..=8).contains(&rank) || !FORMATS.contains(&format) {
        return Err(Error::InvalidField);
    }
    Ok(rank | ((format as u32) << 8) | ((operation as u32) << 16))
}

/// Validate the complete nonempty packet before computing physical ownership.
///
/// The header contains control, axis, update count, destination extent/offset,
/// index extent/offset, and destination axis length. The remaining words hold
/// eight update dimensions, dense destination strides, and signed index strides.
/// Values and destination have logical offset zero. Status occupies one word.
pub fn geometry(parameters: &[u32]) -> Result<Geometry> {
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    let rank = p[0] & 0xff;
    let format = Format::from_raw((p[0] >> 8) & 0xff)?;
    let operation = Operation::from_raw((p[0] >> 16) & 0xff)?;
    if control(rank, format, operation)? != p[0] {
        return Err(Error::InvalidField);
    }
    let rank = rank as usize;
    let axis = p[1] as usize;
    if axis >= rank
        || [p[2], p[3], p[5], p[7]].iter().any(|&n| n == 0 || n > i32::MAX as u32)
        || p[4] != 0
        || p[8..16].iter().any(|&n| n == 0 || n > i32::MAX as u32)
        || p[8 + rank..16].iter().any(|&n| n != 1)
        || p[16 + rank..24].iter().chain(&p[24 + rank..32]).any(|&stride| stride != 0)
    {
        return Err(Error::InvalidField);
    }
    let count = p[8..16].iter().try_fold(1u32, |n, &dim| n.checked_mul(dim)).ok_or(Error::InvalidField)?;
    if count != p[2] {
        return Err(Error::InvalidField);
    }
    let mut dense_stride = 1u32;
    for dimension in (0..rank).rev() {
        if p[16 + dimension] != dense_stride {
            return Err(Error::InvalidField);
        }
        let length = if dimension == axis { p[7] } else { p[8 + dimension] };
        dense_stride = dense_stride.checked_mul(length).ok_or(Error::InvalidField)?;
    }
    if dense_stride != p[3] {
        return Err(Error::InvalidField);
    }

    // Eight signed-I32 spans fit I128 even for malformed metadata. Neither a
    // negative stride nor an overflowing I64 intermediate can hide a bad span.
    let mut low = i128::from(p[6]);
    let mut high = low;
    for (&dim, &stride) in p[8..16].iter().zip(&p[24..32]) {
        let span = i128::from(dim - 1) * i128::from(stride as i32);
        low += span.min(0);
        high += span.max(0);
    }
    if low < 0 || high >= i128::from(p[5]) {
        return Err(Error::InvalidField);
    }

    let words = |elements: u32| -> Result<u64> {
        let bytes = u64::from(elements).checked_mul(u64::from(format.width().bytes())).ok_or(Error::InvalidField)?;
        let words = bytes.div_ceil(4);
        u32::try_from(words).map_err(|_| Error::InvalidField)?;
        Ok(words)
    };
    let values_words = words(p[2])?;
    let output_words = words(p[3])?;
    let owners = if format.width() == Width::Eight {
        p[3]
    } else {
        u32::try_from(output_words).map_err(|_| Error::InvalidField)?
    };
    Ok(Geometry {
        values_words,
        indices_words: u64::from(p[5]),
        status_words: 1,
        output_words,
        owners,
    })
}

/// Required physical word extents in values, indices, status, output order.
pub fn validate(parameters: &[u32]) -> Result<[u64; 4]> {
    let g = geometry(parameters)?;
    Ok([g.values_words, g.indices_words, g.status_words, g.output_words])
}

/// One owner per output word, or one owner for both words of a wide value.
pub fn work_items(parameters: &[u32]) -> Result<u32> {
    Ok(geometry(parameters)?.owners)
}
