//! Serial nearest-even floating arithmetic reductions over packed signed views.
use super::{cast::Format, reduction::Reduction, storage::Width};
use crate::{Error, Result};

pub const PARAMETERS: usize = 24;
pub const THREADS: u32 = 128;
pub const SERIAL_POLICY: u32 = 1;
pub const FORMATS: [Format; 4] = [Format::F64, Format::Flex32, Format::F16, Format::BF16];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Operation {
    Sum = 0,
    Mean = 1,
    Product = 2,
}

impl Operation {
    pub const ALL: [Self; 3] = [Self::Sum, Self::Mean, Self::Product];

    pub fn from_raw(raw: u32) -> Result<Self> {
        match raw {
            0 => Ok(Self::Sum),
            1 => Ok(Self::Mean),
            2 => Ok(Self::Product),
            _ => Err(Error::InvalidField),
        }
    }

    pub fn from_reduction(reduction: Reduction) -> Result<Self> {
        if reduction.is_integer() {
            return Err(Error::InvalidField);
        }
        Self::from_raw(reduction as u32)
    }
}

/// Source and destination share their dtype. Half formats accumulate in F32.
/// F32 input retains the existing cooperative reduction contract instead.
pub fn control(operation: Operation, input: Format) -> Result<u32> {
    if !FORMATS.contains(&input) {
        return Err(Error::InvalidField);
    }
    let accumulator = if input == Format::F64 { Format::F64 } else { Format::F32 };
    Ok(operation as u32 | (input as u32) << 8 | (input as u32) << 16 | (accumulator as u32) << 24)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub input_words: u64,
    pub output_words: u64,
    pub output_count: u32,
    pub reduction_length: u32,
    pub owners: u32,
}

fn product(mut dimensions: impl Iterator<Item = u32> + Clone) -> Result<u32> {
    if dimensions.clone().any(|dimension| dimension == 0) {
        return Ok(0);
    }
    dimensions
        .try_fold(1u32, |count, dimension| count.checked_mul(dimension).filter(|&count| count <= i32::MAX as u32))
        .ok_or(Error::InvalidField)
}

/// Validate axis-only metadata, including no-read and planner-only empty cases.
/// Words 0..8 hold N, rank, control, axis, offset, storage extent, L, and policy.
/// Dimensions and signed element strides occupy the remaining sixteen words.
pub fn geometry(parameters: &[u32]) -> Result<Geometry> {
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    let rank = p[1] as usize;
    let operation = Operation::from_raw(p[2] & 0xff)?;
    let input = Format::from_raw((p[2] >> 8) & 0xff)?;
    if p[2] != control(operation, input)?
        || !(1..=8).contains(&rank)
        || p[3] >= p[1]
        || p[7] != SERIAL_POLICY
        || p[4] > p[5]
        || p[5] > i32::MAX as u32
        || p[8..16].iter().any(|&dimension| dimension > i32::MAX as u32)
        || p[8 + rank..16].iter().any(|&dimension| dimension != 1)
        || p[16 + rank..24].iter().any(|&stride| stride != 0)
    {
        return Err(Error::InvalidField);
    }
    let count = product(p[8..8 + rank].iter().copied())?;
    let output_count = product(
        p[8..8 + rank]
            .iter()
            .enumerate()
            .filter_map(|(axis, &dimension)| (axis != p[3] as usize).then_some(dimension)),
    )?;
    let reduction_length = p[8 + p[3] as usize];
    if p[0] != output_count || p[6] != reduction_length {
        return Err(Error::InvalidField);
    }
    if count != 0 {
        let mut lower = i64::from(p[4]);
        let mut upper = lower;
        for (&dimension, &stride) in p[8..8 + rank].iter().zip(&p[16..16 + rank]) {
            let span = i64::from(dimension - 1).checked_mul(i64::from(stride as i32)).ok_or(Error::InvalidField)?;
            lower = lower.checked_add(span.min(0)).ok_or(Error::InvalidField)?;
            upper = upper.checked_add(span.max(0)).ok_or(Error::InvalidField)?;
        }
        if lower < 0 || upper >= i64::from(p[5]) {
            return Err(Error::InvalidField);
        }
    }
    let input_words = if count == 0 { 1 } else { input.width().words(p[5]) };
    let output_words = input.width().words(output_count).max(1);
    u32::try_from(input_words).map_err(|_| Error::InvalidField)?;
    u32::try_from(output_words).map_err(|_| Error::InvalidField)?;
    let owners = if input.width() == Width::Eight {
        output_count
    } else {
        u32::try_from(input.width().words(output_count)).map_err(|_| Error::InvalidField)?
    };
    Ok(Geometry { input_words, output_words, output_count, reduction_length, owners })
}

fn launch_geometry(parameters: &[u32]) -> Result<Geometry> {
    let geometry = geometry(parameters)?;
    if geometry.output_count == 0 {
        return Err(Error::InvalidField);
    }
    Ok(geometry)
}

pub fn validate(parameters: &[u32]) -> Result<[u64; 2]> {
    let geometry = launch_geometry(parameters)?;
    Ok([geometry.input_words, geometry.output_words])
}

pub fn work_items(parameters: &[u32]) -> Result<u32> {
    Ok(launch_geometry(parameters)?.owners)
}
