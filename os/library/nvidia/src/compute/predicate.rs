//! Packed boolean predicates with exact ordinary input storage.
use super::{cast::Format, elementwise, storage::Width};
use crate::{Error, Result};
pub const PARAMETERS: usize = 32;
pub const PACKED_BOOL: u32 = 1 << 16;
pub const SCALAR_RHS: u32 = 1 << 17;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Predicate {
    Equal = 0,
    Greater = 1,
    GreaterEqual = 2,
    Less = 3,
    LessEqual = 4,
    And = 5,
    Or = 6,
    Xor = 7,
    Not = 8,
    IsNan = 9,
    IsInfinite = 10,
    IsFinite = 11,
}
impl Predicate {
    pub const ALL: [Self; 12] = [
        Self::Equal,
        Self::Greater,
        Self::GreaterEqual,
        Self::Less,
        Self::LessEqual,
        Self::And,
        Self::Or,
        Self::Xor,
        Self::Not,
        Self::IsNan,
        Self::IsInfinite,
        Self::IsFinite,
    ];
    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.get(raw as usize).copied().ok_or(Error::InvalidField)
    }
    pub fn binary(self) -> bool {
        (self as u32) < Self::Not as u32
    }
    pub fn supports(self, format: Format) -> bool {
        match self {
            Self::Equal => true,
            Self::Greater | Self::GreaterEqual | Self::Less | Self::LessEqual => (format as u32) < Format::Bool8 as u32,
            Self::And | Self::Or | Self::Xor | Self::Not => matches!(format, Format::Bool8 | Format::Bool32),
            _ => (format as u32) <= Format::BF16 as u32,
        }
    }
}
/// count, rank, op|format<<8|flags, scalar low, left offset,
/// right offset or scalar high, left extent, right extent, dimensions,
/// left strides, right strides. Unary/scalar right geometry is canonical zero.
pub fn validate(p: &[u32]) -> Result<[u64; 3]> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let op = Predicate::from_raw(p[2] & 0xff)?;
    let format = Format::from_raw((p[2] >> 8) & 0xff)?;
    let scalar = p[2] & SCALAR_RHS != 0;
    let unused = scalar || !op.binary();
    let scalar_bits = u64::from(p[3]) | (u64::from(p[5]) << 32);
    let scalar_valid = match format {
        Format::Bool8 | Format::Bool32 => scalar_bits <= 1,
        _ => format.width() == Width::Eight || scalar_bits < (1u64 << (8 * format.width().bytes())),
    };
    if p[2] & !(0xffff | PACKED_BOOL | SCALAR_RHS) != 0
        || !op.supports(format)
        || (scalar && !op.binary())
        || (scalar && !scalar_valid)
        || (!scalar && p[3] != 0)
        || (unused && (p[7] != 0 || p[24..32].iter().any(|&v| v != 0)))
        || (!scalar && unused && p[5] != 0)
    {
        return Err(Error::InvalidField);
    }
    let mut geometry = *p;
    if unused {
        geometry[5] = 0;
        geometry[7] = 1;
    }
    elementwise::validate_layout(&geometry)?;
    let width = if p[2] & PACKED_BOOL != 0 { Width::One } else { Width::Four };
    Ok([
        format.width().words(p[6]),
        if unused { 1 } else { format.width().words(p[7]) },
        width.words(p[0]),
    ])
}
