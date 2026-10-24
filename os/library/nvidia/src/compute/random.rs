//! Counter-based random tensor generation. Counters count four-word blocks.
use crate::{Error, Result};

pub const PARAMETERS: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Mode {
    Words = 0,
    Uniform = 1,
    Normal = 2,
    Bernoulli = 3,
    IntegerUniform = 4,
}

impl Mode {
    pub const ALL: [Self; 5] = [Self::Words, Self::Uniform, Self::Normal, Self::Bernoulli, Self::IntegerUniform];

    pub fn from_raw(value: u32) -> Result<Self> {
        Self::ALL.get(value as usize).copied().ok_or(Error::InvalidField)
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Words => "burn_random_words",
            Self::Uniform => "burn_random_uniform",
            Self::Normal => "burn_random_normal",
            Self::Bernoulli => "burn_random_bernoulli",
            Self::IntegerUniform => "burn_random_integer_uniform",
        }
    }
}

/// Largest representable FP32 value strictly less than a finite value.
pub fn predecessor(bits: u32) -> u32 {
    if bits & 0x7fff_ffff == 0 {
        0x8000_0001
    } else if bits >> 31 == 0 {
        bits - 1
    } else {
        bits + 1
    }
}

/// Count, two key words, two block-counter words, mode, and three mode arguments.
pub fn validate(parameters: &[u32]) -> Result<[u64; 1]> {
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    if p[0] == 0 || p[0] > i32::MAX as u32 {
        return Err(Error::InvalidField);
    }
    let counter = u64::from(p[3]) | (u64::from(p[4]) << 32);
    counter.checked_add(u64::from(p[0]).div_ceil(4)).ok_or(Error::InvalidField)?;
    let a = f32::from_bits(p[6]);
    let b = f32::from_bits(p[7]);
    let valid = match Mode::from_raw(p[5])? {
        Mode::Words => p[6..].iter().all(|&v| v == 0),
        Mode::Uniform => a.is_finite() && b.is_finite() && a < b && p[8] == predecessor(p[7]),
        Mode::Normal => a.is_finite() && b.is_finite() && b >= 0.0 && p[8] == 0,
        Mode::Bernoulli => p[8] == 0 && (p[7] == 0 || (p[7] == 1 && p[6] == 0)),
        Mode::IntegerUniform => p[7] != 0 && i64::from(p[6] as i32) + i64::from(p[7]) <= i64::from(i32::MAX) && p[8] == p[7].wrapping_neg() % p[7],
    };
    if !valid {
        return Err(Error::InvalidField);
    }
    Ok([u64::from(p[0])])
}
