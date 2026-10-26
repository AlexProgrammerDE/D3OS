//! Dense mixed-precision products with packed inputs and a full F32 writer.
use super::Kernel;
use crate::{Error, Result};

pub const PARAMETERS: usize = 5;
pub const THREADS: u32 = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    F16,
    BF16,
    Flex32,
}

impl Kind {
    pub const ALL: [Self; 3] = [Self::F16, Self::BF16, Self::Flex32];

    pub const fn kernel(self) -> Kernel {
        match self {
            Self::F16 => Kernel::MatmulF16,
            Self::BF16 => Kernel::MatmulBf16,
            Self::Flex32 => Kernel::MatmulFlex32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub input_words: [u64; 2],
    pub output_words: u64,
    pub blocks: u32,
    pub work_items: u32,
}

fn count(a: u32, b: u32) -> Result<u32> {
    a.checked_mul(b).filter(|&n| n <= i32::MAX as u32).ok_or(Error::InvalidField)
}

/// Parameters are M, K, N, lhs half-slot, and rhs half-slot. Input pointers
/// already address the aligned U32 word. Flex32 requires both slots to be zero.
/// Empty outputs belong to Native; zero K launches write positive F32 zero
/// without reading either one-word placeholder input and require zero slots.
pub fn geometry(kind: Kind, parameters: &[u32]) -> Result<Geometry> {
    let &[m, k, n, lhs_slot, rhs_slot]: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    if m == 0
        || n == 0
        || [m, k, n].iter().any(|&dimension| dimension > i32::MAX as u32)
        || lhs_slot > 1
        || rhs_slot > 1
        || ((kind == Kind::Flex32 || k == 0) && (lhs_slot != 0 || rhs_slot != 0))
    {
        return Err(Error::InvalidField);
    }
    let lhs_count = count(m, k)?;
    let rhs_count = count(k, n)?;
    let output_count = count(m, n)?;
    let tile_rows = m.div_ceil(16);
    let tile_cols = n.div_ceil(32);
    let blocks = tile_rows.checked_mul(tile_cols).ok_or(Error::InvalidField)?;
    let work_items = blocks.checked_mul(THREADS).ok_or(Error::InvalidField)?;
    // Device masks replace invalid coordinates before multiplication. Live
    // addresses plus a half-slot fit U32 under the checked dense counts above.
    let words = |elements: u32, slot: u32| {
        if elements == 0 {
            1
        } else if kind == Kind::Flex32 {
            u64::from(elements)
        } else {
            (u64::from(elements) + u64::from(slot)).div_ceil(2)
        }
    };
    Ok(Geometry {
        input_words: [words(lhs_count, lhs_slot), words(rhs_count, rhs_slot)],
        output_words: u64::from(output_count),
        blocks,
        work_items,
    })
}

pub fn validate(kind: Kind, parameters: &[u32]) -> Result<[u64; 3]> {
    let geometry = geometry(kind, parameters)?;
    Ok([geometry.input_words[0], geometry.input_words[1], geometry.output_words])
}

pub fn work_items(kind: Kind, parameters: &[u32]) -> Result<u32> {
    Ok(geometry(kind, parameters)?.work_items)
}
