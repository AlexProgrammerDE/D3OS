//! Exact integer arithmetic over packed storage and signed logical views.
use super::{cast::Format, elementwise, storage::Width, words::WordOp};
use crate::{Error, Result};

pub const PARAMETERS: usize = 32;
pub const SCALAR_RHS: u32 = 1 << 24;
pub const THREADS: u32 = 128;
pub const SYMBOL: &str = "burn_integer_arithmetic";
pub const FORMATS: [Format; 8] = [
    Format::I64,
    Format::I32,
    Format::I16,
    Format::I8,
    Format::U64,
    Format::U32,
    Format::U16,
    Format::U8,
];

/// The discriminants intentionally match the corresponding `WordOp` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Operation {
    Add = 0,
    Subtract = 1,
    Multiply = 2,
    Negate = 8,
    Absolute = 9,
    Sign = 10,
    BitAnd = 11,
    BitOr = 12,
    BitXor = 13,
    BitNot = 14,
    ShiftLeft = 15,
    ShiftRight = 16,
    Divide = 29,
    Remainder = 30,
    Power = 31,
}

impl Operation {
    pub const ALL: [Self; 15] = [
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Negate,
        Self::Absolute,
        Self::Sign,
        Self::BitAnd,
        Self::BitOr,
        Self::BitXor,
        Self::BitNot,
        Self::ShiftLeft,
        Self::ShiftRight,
        Self::Divide,
        Self::Remainder,
        Self::Power,
    ];

    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.into_iter().find(|op| *op as u32 == raw).ok_or(Error::InvalidField)
    }

    pub fn from_word(op: WordOp) -> Result<Self> {
        Self::from_raw(op as u32)
    }

    pub fn unary(self) -> bool {
        matches!(self, Self::Negate | Self::Absolute | Self::Sign | Self::BitNot)
    }

    pub fn binary(self) -> bool {
        !self.unary()
    }

    /// Power alone accepts an exponent tensor with another integer format.
    pub fn supports(self, lhs: Format, rhs: Format) -> bool {
        FORMATS.contains(&lhs) && FORMATS.contains(&rhs) && (lhs == rhs || self == Self::Power)
    }
}

/// count, rank, op | lhs format<<8 | rhs format<<16 | scalar flag,
/// scalar low, lhs offset, rhs offset or scalar high, lhs extent, rhs extent,
/// eight dimensions, eight signed lhs strides, eight signed rhs strides.
///
/// Unary and scalar RHS geometry is zero. Scalar bits are normalized to the
/// lhs format before encoding, with all unused bits zero. Empty Native outputs
/// bypass dispatch; this launch contract requires a nonzero count.
pub fn validate(p: &[u32]) -> Result<[u64; 3]> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let op = Operation::from_raw(p[2] & 0xff)?;
    let lhs = Format::from_raw((p[2] >> 8) & 0xff)?;
    let rhs = Format::from_raw((p[2] >> 16) & 0xff)?;
    let scalar = p[2] & SCALAR_RHS != 0;
    let unused_rhs = scalar || op.unary();
    if p[2] & !(0x00ff_ffff | SCALAR_RHS) != 0
        || !op.supports(lhs, rhs)
        || (scalar && op.unary())
        || (unused_rhs && (rhs != lhs || p[7] != 0 || p[24..32].iter().any(|&stride| stride != 0)))
        || (!scalar && p[3] != 0)
        || (op.unary() && p[5] != 0)
    {
        return Err(Error::InvalidField);
    }
    if scalar {
        let bits = u64::from(p[3]) | (u64::from(p[5]) << 32);
        if lhs.width() != Width::Eight && bits >> (lhs.width().bytes() * 8) != 0 {
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
        lhs.width().words(p[6]),
        if unused_rhs { 1 } else { rhs.width().words(p[7]) },
        lhs.width().words(p[0]),
    ];
    // Logical addresses are signed I32, but packed pointer offsets are unsigned
    // U32. In particular, both words of an I64/U64 value must remain reachable.
    for words in extents {
        u32::try_from(words).map_err(|_| Error::InvalidField)?;
    }
    Ok(extents)
}

/// One owner per packed output word, or per pair of words for 64-bit values.
pub fn work_items(p: &[u32]) -> Result<u32> {
    let extents = validate(p)?;
    let lhs = Format::from_raw((p[2] >> 8) & 0xff)?;
    if lhs.width() == Width::Eight {
        Ok(p[0])
    } else {
        u32::try_from(extents[2]).map_err(|_| Error::InvalidField)
    }
}
