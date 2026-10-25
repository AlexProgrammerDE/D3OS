//! Inclusive scans along one checked, strided tensor axis.
use super::reduction::{self, Reduction};
use crate::{Error, Result};

pub const PARAMETERS: usize = reduction::PARAMETERS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Scan {
    Sum = 0,
    Product = 1,
    Min = 2,
    Max = 3,
    IntSum = 4,
    IntProduct = 5,
    IntMin = 6,
    IntMax = 7,
}

impl Scan {
    pub const ALL: [Self; 8] = [
        Self::Sum,
        Self::Product,
        Self::Min,
        Self::Max,
        Self::IntSum,
        Self::IntProduct,
        Self::IntMin,
        Self::IntMax,
    ];

    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.get(raw as usize).copied().ok_or(Error::InvalidField)
    }

    pub fn integer(self) -> Self {
        Self::ALL[self as usize % 4 + 4]
    }

    pub fn is_integer(self) -> bool {
        self as u32 >= 4
    }

    pub fn reduction(self) -> Reduction {
        match self {
            Self::Sum => Reduction::Sum,
            Self::Product => Reduction::Product,
            Self::Min => Reduction::Min,
            Self::Max => Reduction::Max,
            Self::IntSum => Reduction::IntSum,
            Self::IntProduct => Reduction::IntProduct,
            Self::IntMin => Reduction::IntMin,
            Self::IntMax => Reduction::IntMax,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Sum => "burn_cumsum",
            Self::Product => "burn_cumprod",
            Self::Min => "burn_cummin",
            Self::Max => "burn_cummax",
            Self::IntSum => "burn_i32_cumsum",
            Self::IntProduct => "burn_i32_cumprod",
            Self::IntMin => "burn_i32_cummin",
            Self::IntMax => "burn_i32_cummax",
        }
    }
}

/// Reduction layout, except field 2 selects the scan and field 7 gives the
/// dense output stride of its axis. Each row writes every inclusive prefix.
pub fn validate(parameters: &[u32]) -> Result<[u64; 2]> {
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    let op = Scan::from_raw(p[2])?;
    if p[6] == 0 {
        return Err(Error::InvalidField);
    }
    let mut reduced = *p;
    reduced[2] = op.reduction() as u32;
    reduced[7] = 0;
    let [input, rows] = reduction::validate(&reduced)?;
    // The reduction validator proves the rank, axis, full element count, and
    // every source address before these axis-dependent slices are accessed.
    let suffix = p[8 + p[3] as usize + 1..8 + p[1] as usize]
        .iter()
        .try_fold(1u32, |n, &d| n.checked_mul(d))
        .ok_or(Error::InvalidField)?;
    if p[7] != suffix {
        return Err(Error::InvalidField);
    }
    Ok([input, rows * u64::from(p[6])])
}
