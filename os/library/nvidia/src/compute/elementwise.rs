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
    Negate = 11,
    Absolute = 12,
    Reciprocal = 13,
    Sqrt = 14,
    Exp = 15,
    Log = 16,
    Log1p = 17,
    Tanh = 18,
    Sin = 19,
    Cos = 20,
    Floor = 21,
    Ceil = 22,
    Round = 23,
    Trunc = 24,
    Sign = 25,
    IsNan = 26,
    IsInfinite = 27,
    IsFinite = 28,
    Power = 29,
    Sinh = 30,
    Cosh = 31,
    Tan = 32,
    Erf = 33,
    Remainder = 34,
}

impl Elementwise {
    pub const ALL: [Self; 35] = [
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
        Self::Negate,
        Self::Absolute,
        Self::Reciprocal,
        Self::Sqrt,
        Self::Exp,
        Self::Log,
        Self::Log1p,
        Self::Tanh,
        Self::Sin,
        Self::Cos,
        Self::Floor,
        Self::Ceil,
        Self::Round,
        Self::Trunc,
        Self::Sign,
        Self::IsNan,
        Self::IsInfinite,
        Self::IsFinite,
        Self::Power,
        Self::Sinh,
        Self::Cosh,
        Self::Tan,
        Self::Erf,
        Self::Remainder,
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
            Self::Negate => "burn_negate",
            Self::Absolute => "burn_absolute",
            Self::Reciprocal => "burn_reciprocal",
            Self::Sqrt => "burn_sqrt",
            Self::Exp => "burn_exp",
            Self::Log => "burn_log",
            Self::Log1p => "burn_log1p",
            Self::Tanh => "burn_tanh",
            Self::Sin => "burn_sin",
            Self::Cos => "burn_cos",
            Self::Floor => "burn_floor",
            Self::Ceil => "burn_ceil",
            Self::Round => "burn_round",
            Self::Trunc => "burn_trunc",
            Self::Sign => "burn_sign",
            Self::IsNan => "burn_is_nan",
            Self::IsInfinite => "burn_is_infinite",
            Self::IsFinite => "burn_is_finite",
            Self::Power => "burn_power",
            Self::Sinh => "burn_sinh",
            Self::Cosh => "burn_cosh",
            Self::Tan => "burn_tan",
            Self::Erf => "burn_erf",
            Self::Remainder => "burn_remainder",
        }
    }

    pub fn comparison(self) -> bool {
        matches!(self, Self::Equal | Self::Greater | Self::GreaterEqual | Self::Less | Self::LessEqual)
    }

    pub fn binary(self) -> bool {
        self.comparison() || matches!(self, Self::Add | Self::Subtract | Self::Multiply | Self::Divide | Self::Power | Self::Remainder)
    }

    pub fn unary(self) -> bool {
        !self.binary() && !matches!(self, Self::Copy | Self::MaskFill)
    }

    pub fn boolean(self) -> bool {
        self.comparison() || matches!(self, Self::IsNan | Self::IsInfinite | Self::IsFinite)
    }
}

/// Parameters: count, rank, operation/flags, scalar bits, two base offsets,
/// two storage extents, eight dimensions, and two sets of eight signed strides.
/// Every reachable address must lie within its declared storage extent.
pub fn validate(parameters: &[u32]) -> Result<[u64; 3]> {
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    if p[2] & !(SCALAR_RHS | 0xff) != 0 {
        return Err(Error::InvalidField);
    }
    let op = Elementwise::from_raw(p[2] & 0xff)?;
    if ((op == Elementwise::MaskFill || op.unary()) && p[2] & SCALAR_RHS != 0) || (p[2] & SCALAR_RHS == 0 && op != Elementwise::MaskFill && p[3] != 0) {
        return Err(Error::InvalidField);
    }
    validate_layout(p)
}

pub(crate) fn validate_layout(p: &[u32; PARAMETERS]) -> Result<[u64; 3]> {
    let rank = p[1] as usize;
    if !(1..=8).contains(&rank) || p[0] == 0 || p[0] > i32::MAX as u32 {
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
