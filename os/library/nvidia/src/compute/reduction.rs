//! Strided FP32 and exact I32 reduction contract shared by the driver and native tensor backend.
use crate::{Error, Result};

pub const PARAMETERS: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Reduction {
    Sum = 0,
    Mean = 1,
    Product = 2,
    Min = 3,
    Max = 4,
    ArgMin = 5,
    ArgMax = 6,
    IntSum = 7,
    IntMean = 8,
    IntProduct = 9,
    IntMin = 10,
    IntMax = 11,
    IntArgMin = 12,
    IntArgMax = 13,
}

impl Reduction {
    pub const ALL: [Self; 7] = [Self::Sum, Self::Mean, Self::Product, Self::Min, Self::Max, Self::ArgMin, Self::ArgMax];

    pub const INTEGERS: [Self; 7] = [
        Self::IntSum,
        Self::IntMean,
        Self::IntProduct,
        Self::IntMin,
        Self::IntMax,
        Self::IntArgMin,
        Self::IntArgMax,
    ];

    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL
            .iter()
            .chain(Self::INTEGERS.iter())
            .find(|&&op| op as u32 == raw)
            .copied()
            .ok_or(Error::InvalidField)
    }

    pub fn integer(self) -> Self {
        if (self as u32) < 7 { Self::INTEGERS[self as usize] } else { self }
    }

    pub fn is_integer(self) -> bool {
        (self as u32) >= 7
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Sum => "burn_sum",
            Self::Mean => "burn_mean",
            Self::Product => "burn_product",
            Self::Min => "burn_min",
            Self::Max => "burn_max",
            Self::ArgMin => "burn_arg_min",
            Self::ArgMax => "burn_arg_max",
            Self::IntSum => "burn_i32_sum",
            Self::IntMean => "burn_i32_mean",
            Self::IntProduct => "burn_i32_product",
            Self::IntMin => "burn_i32_min",
            Self::IntMax => "burn_i32_max",
            Self::IntArgMin => "burn_i32_arg_min",
            Self::IntArgMax => "burn_i32_arg_max",
        }
    }

    pub fn index(self) -> bool {
        matches!(self, Self::ArgMin | Self::ArgMax | Self::IntArgMin | Self::IntArgMax)
    }

    pub fn allows_empty(self) -> bool {
        matches!(self, Self::Sum | Self::Mean | Self::Product | Self::IntSum | Self::IntProduct)
    }
}

/// Parameters: output count, rank, operation, axis, base offset, storage extent,
/// reduction length, reserved zero, eight dimensions, and eight signed strides.
/// The reduced dimension stays present with size one in the dense output.
pub fn validate(parameters: &[u32]) -> Result<[u64; 2]> {
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    let rank = p[1] as usize;
    let axis = p[3] as usize;
    let op = Reduction::from_raw(p[2])?;
    if !(1..=8).contains(&rank)
        || axis >= rank
        || p[0] == 0
        || p[0] > i32::MAX as u32
        || p[5] == 0
        || p[5] > i32::MAX as u32
        || p[7] != 0
        || p[8 + axis] != p[6]
        || p[8 + rank..16].iter().any(|&d| d != 1)
        || p[16 + rank..24].iter().any(|&s| s != 0)
        || p[8..16].iter().any(|&d| d > i32::MAX as u32)
        || (p[6] == 0 && !op.allows_empty())
    {
        return Err(Error::InvalidField);
    }
    let outputs = p[8..16]
        .iter()
        .enumerate()
        .try_fold(1u32, |n, (a, &d)| n.checked_mul(if a == axis { 1 } else { d }))
        .ok_or(Error::InvalidField)?;
    if outputs != p[0] || outputs.checked_mul(p[6]).is_none_or(|n| n > i32::MAX as u32) {
        return Err(Error::InvalidField);
    }
    if p[6] == 0 {
        // No input addresses exist, but the declared base must still lie within
        // its allocation. The kernel masks every load for an empty reduction.
        if p[4] > p[5] {
            return Err(Error::InvalidField);
        }
    } else {
        let mut lowest = i64::from(p[4]);
        let mut highest = lowest;
        for (&dim, &stride) in p[8..16].iter().zip(&p[16..24]) {
            let span = i64::from(dim - 1) * i64::from(stride as i32);
            lowest = lowest.checked_add(span.min(0)).ok_or(Error::InvalidField)?;
            highest = highest.checked_add(span.max(0)).ok_or(Error::InvalidField)?;
        }
        if lowest < 0 || highest >= i64::from(p[5]) {
            return Err(Error::InvalidField);
        }
    }
    Ok([u64::from(p[5]), u64::from(outputs)])
}
