//! Packed ordinary storage. Geometry uses elements; pointer extents use words.
use super::elementwise;
use crate::{Error, Result};

pub const PARAMETERS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Width {
    One = 0,
    Two = 1,
    Four = 2,
    Eight = 3,
}

impl Width {
    pub const ALL: [Self; 4] = [Self::One, Self::Two, Self::Four, Self::Eight];

    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.get(raw as usize).copied().ok_or(Error::InvalidField)
    }

    pub fn bytes(self) -> u32 {
        1 << self as u32
    }

    pub fn words(self, elements: u32) -> u64 {
        (u64::from(elements) * u64::from(self.bytes())).div_ceil(4)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Mode {
    Copy = 0,
    Fill = 1,
}

impl Mode {
    pub const ALL: [Self; 2] = [Self::Copy, Self::Fill];
    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.get(raw as usize).copied().ok_or(Error::InvalidField)
    }
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Copy => "burn_storage_copy",
            Self::Fill => "burn_storage_fill",
        }
    }
}

/// count, rank, width | mode<<8, scalar low, offset, scalar high, source extent,
/// zero, eight dimensions, eight signed source strides, and eight zero words.
pub fn validate(p: &[u32]) -> Result<(Mode, Width, u64, u64)> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let width = Width::from_raw(p[2] & 0xff)?;
    let mode = Mode::from_raw(p[2] >> 8)?;
    if p[7] != 0 || p[24..].iter().any(|&v| v != 0) {
        return Err(Error::InvalidField);
    }
    let mut layout = *p;
    layout[2] = 0;
    layout[3] = 0;
    layout[5] = p[4];
    layout[7] = p[6];
    layout[24..32].copy_from_slice(&p[16..24]);
    elementwise::validate_layout(&layout)?;
    match mode {
        Mode::Copy if p[3] != 0 || p[5] != 0 => return Err(Error::InvalidField),
        Mode::Fill => {
            let bits = u64::from(p[3]) | (u64::from(p[5]) << 32);
            if p[4] != 0 || p[6] != p[0] || (width != Width::Eight && bits >> (width.bytes() * 8) != 0) {
                return Err(Error::InvalidField);
            }
            let mut stride = 1u32;
            for axis in (0..p[1] as usize).rev() {
                if p[16 + axis] != stride {
                    return Err(Error::InvalidField);
                }
                stride = stride.checked_mul(p[8 + axis]).ok_or(Error::InvalidField)?;
            }
        }
        _ => {}
    }
    Ok((mode, width, width.words(p[6]), width.words(p[0])))
}
