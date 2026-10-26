//! Typed floating arithmetic over packed storage and signed logical views.
use super::{
    cast::Format,
    elementwise::{self, Elementwise},
    storage::Width,
};
use crate::{Error, Result};

pub const PARAMETERS: usize = 32;
pub const SCALAR_RHS: u32 = 1 << 24;
pub const THREADS: u32 = 128;
pub const SYMBOL: &str = "burn_float_arithmetic";
pub const FORMATS: [Format; 5] = [Format::F64, Format::F32, Format::Flex32, Format::F16, Format::BF16];

/// Discriminants match the corresponding `Elementwise` operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Operation {
    Add = 1,
    Subtract = 2,
    Multiply = 3,
    Divide = 4,
    Negate = 11,
    Absolute = 12,
    Reciprocal = 13,
    Sqrt = 14,
    Exp = 15,
    Log = 16,
    Tanh = 18,
    Sin = 19,
    Cos = 20,
    Floor = 21,
    Ceil = 22,
    Round = 23,
    Trunc = 24,
    Sign = 25,
    Remainder = 34,
}

impl Operation {
    pub const ALL: [Self; 19] = [
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Divide,
        Self::Negate,
        Self::Absolute,
        Self::Reciprocal,
        Self::Sqrt,
        Self::Exp,
        Self::Log,
        Self::Tanh,
        Self::Sin,
        Self::Cos,
        Self::Floor,
        Self::Ceil,
        Self::Round,
        Self::Trunc,
        Self::Sign,
        Self::Remainder,
    ];

    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.into_iter().find(|op| *op as u32 == raw).ok_or(Error::InvalidField)
    }

    pub fn from_elementwise(op: Elementwise) -> Result<Self> {
        Self::from_raw(op as u32)
    }

    pub fn unary(self) -> bool {
        matches!(
            self,
            Self::Negate
                | Self::Absolute
                | Self::Reciprocal
                | Self::Sqrt
                | Self::Exp
                | Self::Log
                | Self::Tanh
                | Self::Sin
                | Self::Cos
                | Self::Floor
                | Self::Ceil
                | Self::Round
                | Self::Trunc
                | Self::Sign
        )
    }

    pub fn binary(self) -> bool {
        !self.unary()
    }

    pub fn supports(self, format: Format) -> bool {
        FORMATS.contains(&format)
    }
}

/// count, physical rank (1..=8), op | format<<8 | scalar flag,
/// scalar low, lhs offset, rhs offset or scalar high, lhs extent, rhs extent,
/// eight dimensions, eight signed lhs strides, eight signed rhs strides.
///
/// Tensor operands use the same exact format, checked by Native before encoding.
/// Unary/scalar RHS extent and strides are zero; unary scalar words are zero.
/// Scalar bits already have destination precision, with unused upper bits zero.
/// Empty results bypass dispatch. Output is contiguous and newly owned.
pub fn validate(p: &[u32]) -> Result<[u64; 3]> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let op = Operation::from_raw(p[2] & 0xff)?;
    let format = Format::from_raw((p[2] >> 8) & 0xff)?;
    let scalar = p[2] & SCALAR_RHS != 0;
    let unused_rhs = scalar || op.unary();
    if p[2] & !(0xffff | SCALAR_RHS) != 0
        || !op.supports(format)
        || (scalar && op.unary())
        || (unused_rhs && (p[7] != 0 || p[24..32].iter().any(|&stride| stride != 0)))
        || (!scalar && p[3] != 0)
        || (op.unary() && p[5] != 0)
    {
        return Err(Error::InvalidField);
    }
    if scalar {
        let bits = u64::from(p[3]) | (u64::from(p[5]) << 32);
        if format.width() != Width::Eight && bits >> (format.width().bytes() * 8) != 0 {
            return Err(Error::InvalidField);
        }
    }
    let mut geometry = *p;
    if unused_rhs {
        geometry[5] = 0;
        geometry[7] = 1;
    }
    elementwise::validate_layout(&geometry)?;
    let extents = [
        format.width().words(p[6]),
        if unused_rhs { 1 } else { format.width().words(p[7]) },
        format.width().words(p[0]),
    ];
    // Logical addresses remain signed I32. Physical packed offsets use U32,
    // including the second word of every F64 value.
    for words in extents {
        u32::try_from(words).map_err(|_| Error::InvalidField)?;
    }
    Ok(extents)
}

/// One owner writes a half pair, one 32-bit value, or both words of an F64.
pub fn work_items(p: &[u32]) -> Result<u32> {
    let extents = validate(p)?;
    let format = Format::from_raw((p[2] >> 8) & 0xff)?;
    if format.width() == Width::Eight {
        Ok(p[0])
    } else {
        u32::try_from(extents[2]).map_err(|_| Error::InvalidField)
    }
}
