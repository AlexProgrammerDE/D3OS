//! Bounds for full-writer, resident merge-sort passes.
use super::scan;
use crate::{Error, Result};

pub const PARAMETERS: usize = 32;
pub const INTEGER: u32 = 1;
pub const DESCENDING: u32 = 2;
pub const INITIAL: u32 = 4;
pub const TOPK: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Mode {
    FloatAscending = 0,
    IntAscending = 1,
    FloatDescending = 2,
    IntDescending = 3,
    FloatAscendingInitial = 4,
    IntAscendingInitial = 5,
    FloatDescendingInitial = 6,
    IntDescendingInitial = 7,
    FloatTopK = 10,
    FloatTopKInitial = 14,
}

impl Mode {
    pub const ALL: [Self; 10] = [
        Self::FloatAscending,
        Self::IntAscending,
        Self::FloatDescending,
        Self::IntDescending,
        Self::FloatAscendingInitial,
        Self::IntAscendingInitial,
        Self::FloatDescendingInitial,
        Self::IntDescendingInitial,
        Self::FloatTopK,
        Self::FloatTopKInitial,
    ];
    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.iter().find(|&&mode| mode as u32 == raw).copied().ok_or(Error::InvalidField)
    }
    pub fn new(integer: bool, descending: bool, initial: bool) -> Self {
        Self::ALL[((u32::from(integer) * INTEGER) | (u32::from(descending) * DESCENDING) | (u32::from(initial) * INITIAL)) as usize]
    }
    pub fn initial(self) -> bool {
        self as u32 & INITIAL != 0
    }
    pub fn for_topk(integer: bool, initial: bool) -> Self {
        if integer {
            Self::new(true, true, initial)
        } else if initial {
            Self::FloatTopKInitial
        } else {
            Self::FloatTopK
        }
    }
    pub fn topk(self) -> bool {
        self as u32 & TOPK != 0
    }
    pub fn integer(self) -> bool {
        self as u32 & INTEGER != 0
    }
    pub fn descending(self) -> bool {
        self as u32 & DESCENDING != 0
    }
    pub fn symbol(self) -> &'static str {
        match self {
            Self::FloatAscending => "burn_sort_f32_ascending",
            Self::IntAscending => "burn_sort_i32_ascending",
            Self::FloatDescending => "burn_sort_f32_descending",
            Self::IntDescending => "burn_sort_i32_descending",
            Self::FloatAscendingInitial => "burn_sort_f32_ascending_initial",
            Self::IntAscendingInitial => "burn_sort_i32_ascending_initial",
            Self::FloatDescendingInitial => "burn_sort_f32_descending_initial",
            Self::IntDescendingInitial => "burn_sort_i32_descending_initial",
            Self::FloatTopK => "burn_sort_f32_topk",
            Self::FloatTopKInitial => "burn_sort_f32_topk_initial",
        }
    }
}

/// Header: element count, rank, mode, axis, source offset, source extent,
/// axis length, dense axis stride. Dimensions and signed strides follow.
/// Fields 24 and 25 hold the sorted-run width and binary-search iterations.
/// The output packs values followed by I32 indices in one owned allocation.
pub fn validate(parameters: &[u32]) -> Result<[u64; 3]> {
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    let mode = Mode::from_raw(p[2])?;
    if p[6] == 0
        || p[6] > i32::MAX as u32
        || p[24] == 0
        || !p[24].is_power_of_two()
        || p[24] > p[6].next_power_of_two()
        || p[25] != p[24].ilog2() + 1
        || p[26..].iter().any(|&v| v != 0)
        || (mode.initial() && p[24] != 1)
    {
        return Err(Error::InvalidField);
    }
    let mut view: [u32; 24] = p[..24].try_into().unwrap();
    view[0] = p[0] / p[6];
    view[2] = scan::Scan::Sum as u32;
    let [input, elements] = scan::validate(&view)?;
    if elements != u64::from(p[0]) || elements * 2 > i32::MAX as u64 {
        return Err(Error::InvalidField);
    }
    if !mode.initial() {
        let mut stride = 1u32;
        for axis in (0..p[1] as usize).rev() {
            if p[16 + axis] != stride {
                return Err(Error::InvalidField);
            }
            stride = stride.checked_mul(p[8 + axis]).ok_or(Error::InvalidField)?;
        }
        if p[4] != 0 || p[5] != p[0] {
            return Err(Error::InvalidField);
        }
    }
    Ok([input, if mode.initial() { 0 } else { elements }, elements * 2])
}
