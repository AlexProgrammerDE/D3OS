//! Strided elementwise launch contract shared by the driver and Burn backend.
use crate::{Error, Result};

pub const PARAMETERS: usize = 32;
pub const SCALAR_RHS: u32 = 1 << 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Elementwise {
    Copy = 0,
    Add = 1,
    Subtract = 2,
    Multiply = 3,
    Divide = 4,
    Equal = 5,
    Greater = 6,
    GreaterEqual = 7,
    Less = 8,
    LessEqual = 9,
    MaskFill = 10,
}

impl Elementwise {
    pub const ALL: [Self; 11] = [
        Self::Copy,
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Divide,
        Self::Equal,
        Self::Greater,
        Self::GreaterEqual,
        Self::Less,
        Self::LessEqual,
        Self::MaskFill,
    ];

    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.get(raw as usize).copied().ok_or(Error::InvalidField)
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Copy => "burn_strided_copy",
            Self::Add => "burn_add",
            Self::Subtract => "burn_subtract",
            Self::Multiply => "burn_multiply",
            Self::Divide => "burn_divide",
            Self::Equal => "burn_equal",
            Self::Greater => "burn_greater",
            Self::GreaterEqual => "burn_greater_equal",
            Self::Less => "burn_less",
            Self::LessEqual => "burn_less_equal",
            Self::MaskFill => "burn_mask_fill",
        }
    }

    pub fn comparison(self) -> bool {
        matches!(self, Self::Equal | Self::Greater | Self::GreaterEqual | Self::Less | Self::LessEqual)
    }
}

/// Parameters: count, rank, operation/flags, scalar bits, two base offsets,
/// two storage extents, eight dimensions, and two sets of eight signed strides.
/// Every reachable address must lie within its declared storage extent.
pub fn validate(parameters: &[u32]) -> Result<[u64; 3]> {
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    let rank = p[1] as usize;
    if !(1..=8).contains(&rank) || p[2] & !(SCALAR_RHS | 0xff) != 0 || p[0] == 0 || p[0] > i32::MAX as u32 {
        return Err(Error::InvalidField);
    }
    let op = Elementwise::from_raw(p[2] & 0xff)?;
    if (op == Elementwise::MaskFill && p[2] & SCALAR_RHS != 0) || (p[2] & SCALAR_RHS == 0 && op != Elementwise::MaskFill && p[3] != 0) {
        return Err(Error::InvalidField);
    }
    let count = p[8..16]
        .iter()
        .try_fold(1u32, |n, &d| if d == 0 { None } else { n.checked_mul(d) })
        .ok_or(Error::InvalidField)?;
    if count != p[0] || p[8 + rank..16].iter().any(|&d| d != 1) || p[16 + rank..24].iter().chain(&p[24 + rank..32]).any(|&s| s != 0) {
        return Err(Error::InvalidField);
    }
    for input in 0..2 {
        let extent = p[6 + input];
        if extent == 0 || extent > i32::MAX as u32 {
            return Err(Error::InvalidField);
        }
        let mut lowest = i64::from(p[4 + input]);
        let mut highest = lowest;
        for (&dim, &stride) in p[8..16].iter().zip(&p[16 + input * 8..24 + input * 8]) {
            let span = i64::from(dim - 1) * i64::from(stride as i32);
            lowest = lowest.checked_add(span.min(0)).ok_or(Error::InvalidField)?;
            highest = highest.checked_add(span.max(0)).ok_or(Error::InvalidField)?;
        }
        if lowest < 0 || highest >= i64::from(extent) {
            return Err(Error::InvalidField);
        }
    }
    Ok([u64::from(p[6]), u64::from(p[7]), u64::from(count)])
}
