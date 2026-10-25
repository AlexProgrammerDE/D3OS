//! Ordinary numeric conversions with logical geometry and packed buffer bounds.
use super::storage::{self, Width};
use crate::{Error, Result};

pub const PARAMETERS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Format {
    F64 = 0,
    F32 = 1,
    Flex32 = 2,
    F16 = 3,
    BF16 = 4,
    I64 = 5,
    I32 = 6,
    I16 = 7,
    I8 = 8,
    U64 = 9,
    U32 = 10,
    U16 = 11,
    U8 = 12,
    Bool8 = 13,
    Bool32 = 14,
}
impl Format {
    pub const ALL: [Self; 15] = [
        Self::F64,
        Self::F32,
        Self::Flex32,
        Self::F16,
        Self::BF16,
        Self::I64,
        Self::I32,
        Self::I16,
        Self::I8,
        Self::U64,
        Self::U32,
        Self::U16,
        Self::U8,
        Self::Bool8,
        Self::Bool32,
    ];
    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.get(raw as usize).copied().ok_or(Error::InvalidField)
    }
    pub fn width(self) -> Width {
        match self {
            Self::F64 | Self::I64 | Self::U64 => Width::Eight,
            Self::F32 | Self::Flex32 | Self::I32 | Self::U32 | Self::Bool32 => Width::Four,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => Width::Two,
            Self::I8 | Self::U8 | Self::Bool8 => Width::One,
        }
    }
}

/// Same geometry as storage copy, with source | destination<<8 in word two.
pub fn validate(p: &[u32]) -> Result<(u64, u64)> {
    let mut geometry: [u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let source = Format::from_raw(p[2] & 0xff)?;
    let destination = Format::from_raw(p[2] >> 8)?;
    geometry[2] = source.width() as u32;
    let (_, _, input_words, _) = storage::validate(&geometry)?;
    Ok((input_words, destination.width().words(p[0])))
}
