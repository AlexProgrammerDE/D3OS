//! Exact floating extrema over signed views and packed word storage.
use super::{cast::Format, reduction::Reduction, storage::Width};
use crate::{Error, Result};

pub const PARAMETERS: usize = 24;
pub const THREADS: u32 = 128;
pub const ALL_AXES: u32 = u32::MAX;
pub const FORMATS: [Format; 5] = [Format::F64, Format::F32, Format::Flex32, Format::F16, Format::BF16];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Operation {
    Min = 3,
    Max = 4,
    ArgMin = 5,
    ArgMax = 6,
}

impl Operation {
    pub const ALL: [Self; 4] = [Self::Min, Self::Max, Self::ArgMin, Self::ArgMax];

    pub fn from_raw(raw: u32) -> Result<Self> {
        match raw {
            3 => Ok(Self::Min),
            4 => Ok(Self::Max),
            5 => Ok(Self::ArgMin),
            6 => Ok(Self::ArgMax),
            _ => Err(Error::InvalidField),
        }
    }

    pub fn from_reduction(op: Reduction) -> Result<Self> {
        if op.is_integer() {
            return Err(Error::InvalidField);
        }
        Self::from_raw(op as u32)
    }

    pub fn index(self) -> bool {
        matches!(self, Self::ArgMin | Self::ArgMax)
    }
}

/// Encode exact source-bit selection, with integer projection only for indices.
pub fn control(op: Operation, input: Format, output: Format) -> Result<u32> {
    let valid_output = if op.index() {
        matches!(
            output,
            Format::I64 | Format::I32 | Format::I16 | Format::I8 | Format::U64 | Format::U32 | Format::U16 | Format::U8
        )
    } else {
        output == input
    };
    if !FORMATS.contains(&input) || !valid_output {
        return Err(Error::InvalidField);
    }
    Ok(op as u32 | (input as u32) << 8 | (output as u32) << 16 | (input as u32) << 24)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Short,
    Tree,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub input_words: u64,
    pub output_words: u64,
    pub output_count: u32,
    pub reduction_length: u32,
    pub owners: u32,
    pub schedule: Schedule,
}

fn product(mut dimensions: impl Iterator<Item = u32> + Clone) -> Result<u32> {
    if dimensions.clone().any(|dim| dim == 0) {
        return Ok(0);
    }
    dimensions
        .try_fold(1u32, |n, dim| n.checked_mul(dim).filter(|&n| n <= i32::MAX as u32))
        .ok_or(Error::InvalidField)
}

/// Validate all metadata even when the host will return an empty output.
/// No-read inputs need only Native's single four-byte dummy word, including F64.
///
/// Words 0..8 encode output count, rank, control, axis, element offset,
/// declared storage elements, reduction length, and policy zero.
/// Words 8..16 contain dimensions; words 16..24 contain signed I32 strides.
/// Inactive dimensions equal one and inactive strides equal zero.
pub fn geometry(parameters: &[u32]) -> Result<Geometry> {
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    let rank = p[1] as usize;
    let op = Operation::from_raw(p[2] & 0xff)?;
    let input = Format::from_raw((p[2] >> 8) & 0xff)?;
    let output = Format::from_raw((p[2] >> 16) & 0xff)?;
    let all = p[3] == ALL_AXES;
    if p[2] != control(op, input, output)?
        || !(1..=8).contains(&rank)
        || (!all && p[3] >= p[1])
        || (all && op.index())
        || p[7] != 0
        || p[4] > p[5]
        || p[5] > i32::MAX as u32
        || p[8..16].iter().any(|&d| d > i32::MAX as u32)
        || p[8 + rank..16].iter().any(|&d| d != 1)
        || p[16 + rank..24].iter().any(|&s| s != 0)
    {
        return Err(Error::InvalidField);
    }
    let count = product(p[8..8 + rank].iter().copied())?;
    let outputs = if all {
        1
    } else {
        product(
            p[8..8 + rank]
                .iter()
                .enumerate()
                .filter_map(|(axis, &dim)| (axis != p[3] as usize).then_some(dim)),
        )?
    };
    let length = if all { count } else { p[8 + p[3] as usize] };
    if p[0] != outputs || p[6] != length || length == 0 {
        return Err(Error::InvalidField);
    }
    if count != 0 {
        let mut lower = i64::from(p[4]);
        let mut upper = lower;
        for (&dim, &stride) in p[8..8 + rank].iter().zip(&p[16..16 + rank]) {
            let span = i64::from(dim - 1).checked_mul(i64::from(stride as i32)).ok_or(Error::InvalidField)?;
            lower = lower.checked_add(span.min(0)).ok_or(Error::InvalidField)?;
            upper = upper.checked_add(span.max(0)).ok_or(Error::InvalidField)?;
        }
        if lower < 0 || upper >= i64::from(p[5]) {
            return Err(Error::InvalidField);
        }
    }
    let input_words = if count == 0 { 1 } else { input.width().words(p[5]) };
    let output_words = output.width().words(outputs).max(1);
    u32::try_from(input_words).map_err(|_| Error::InvalidField)?;
    u32::try_from(output_words).map_err(|_| Error::InvalidField)?;
    let owners = if output.width() == Width::Eight {
        outputs
    } else {
        u32::try_from(output.width().words(outputs)).map_err(|_| Error::InvalidField)?
    };
    Ok(Geometry {
        input_words,
        output_words,
        output_count: outputs,
        reduction_length: length,
        owners,
        schedule: if length >= THREADS { Schedule::Tree } else { Schedule::Short },
    })
}

fn launch_geometry(p: &[u32]) -> Result<Geometry> {
    let g = geometry(p)?;
    if g.output_count == 0 {
        return Err(Error::InvalidField);
    }
    Ok(g)
}

pub fn validate(p: &[u32]) -> Result<[u64; 2]> {
    let g = launch_geometry(p)?;
    Ok([g.input_words, g.output_words])
}

pub fn work_items(p: &[u32]) -> Result<u32> {
    let g = launch_geometry(p)?;
    match g.schedule {
        Schedule::Short => Ok(g.owners),
        // owners <= N and 128 <= L, so this is bounded by N*L <= I32_MAX.
        Schedule::Tree => g.owners.checked_mul(THREADS).ok_or(Error::InvalidField),
    }
}

pub fn symbol(p: &[u32]) -> Result<&'static str> {
    Ok(match launch_geometry(p)?.schedule {
        Schedule::Short => "burn_float_extrema_short",
        Schedule::Tree => "burn_float_extrema_tree",
    })
}
