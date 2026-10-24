//! Checked one-, two-, and three-dimensional pooling over normalized NCDHW views.
use crate::{Error, Result};

pub const GEOMETRY: usize = 24;
pub const PARAMETERS: usize = 32;
pub const CEIL: u32 = 1;
pub const INCLUDE_PAD: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Mode {
    Max = 0,
    Average = 1,
    Adaptive = 2,
}

impl Mode {
    pub const ALL: [Self; 3] = [Self::Max, Self::Average, Self::Adaptive];
    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.get(raw as usize).copied().ok_or(Error::InvalidField)
    }
    pub fn symbol(self, backward: bool) -> &'static str {
        match (self, backward) {
            (Self::Max, _) => "burn_max_pool",
            (Self::Average, false) => "burn_avg_pool",
            (Self::Adaptive, false) => "burn_adaptive_avg_pool",
            (Self::Average, true) => "burn_avg_pool_backward",
            (Self::Adaptive, true) => "burn_adaptive_avg_pool_backward",
        }
    }
}

fn count(values: &[u32]) -> Result<u32> {
    values
        .iter()
        .try_fold(1u32, |n, &v| n.checked_mul(v).filter(|&v| v <= i32::MAX as u32).ok_or(Error::InvalidField))
}

pub fn output_extent(input: u32, kernel: u32, stride: u32, before: u32, after: u32, dilation: u32, ceil: bool) -> Result<u32> {
    if [input, kernel, stride, dilation].contains(&0) || [input, kernel, stride, before, after, dilation].iter().any(|&v| v > i32::MAX as u32) {
        return Err(Error::InvalidField);
    }
    let padded = input.checked_add(before).and_then(|v| v.checked_add(after)).ok_or(Error::InvalidField)?;
    let effective = (kernel - 1).checked_mul(dilation).and_then(|v| v.checked_add(1)).ok_or(Error::InvalidField)?;
    let available = padded.checked_add(if ceil { stride - 1 } else { 0 }).ok_or(Error::InvalidField)?;
    let mut output = available.checked_sub(effective).map(|v| v / stride + 1).ok_or(Error::InvalidField)?;
    if ceil && (output - 1) * stride >= input + before {
        output -= 1;
    }
    if output == 0
        || (output - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(effective))
            .is_none_or(|v| v > i32::MAX as u32)
        || padded > i32::MAX as u32
    {
        return Err(Error::InvalidField);
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub images: u32,
    pub outputs: u32,
    pub storage: u32,
}

/// N,C,D,H,W, OD,OH,OW, KD,KH,KW, SD,SH,SW, begin[3], end[3], dilation[3], flags.
pub fn geometry(p: &[u32], mode: Mode) -> Result<Geometry> {
    let p: &[u32; GEOMETRY] = p.try_into().map_err(|_| Error::InvalidField)?;
    if p[..23].iter().any(|&v| v > i32::MAX as u32)
        || p[2..14].contains(&0)
        || p[20..23].contains(&0)
        || p[23] & !(CEIL | INCLUDE_PAD) != 0
        || (mode != Mode::Average && p[23] & INCLUDE_PAD != 0)
    {
        return Err(Error::InvalidField);
    }
    count(&p[2..5])?;
    count(&p[8..11])?;
    for axis in 0..3 {
        if mode == Mode::Adaptive {
            if p[8 + axis] != 1 || p[11 + axis] != 1 || p[14 + axis] != 0 || p[17 + axis] != 0 || p[20 + axis] != 1 || p[23] != 0 {
                return Err(Error::InvalidField);
            }
        } else {
            if mode == Mode::Average && p[20 + axis] != 1 {
                return Err(Error::InvalidField);
            }
            if output_extent(
                p[2 + axis],
                p[8 + axis],
                p[11 + axis],
                p[14 + axis],
                p[17 + axis],
                p[20 + axis],
                p[23] & CEIL != 0,
            )? != p[5 + axis]
            {
                return Err(Error::InvalidField);
            }
        }
    }
    let outputs = count(&[p[0], p[1], p[5], p[6], p[7]])?;
    let storage = count(&[outputs, if mode == Mode::Max { 2 } else { 1 }])?;
    Ok(Geometry {
        images: count(&p[..5])?,
        outputs,
        storage,
    })
}

pub fn validate(p: &[u32], backward: bool) -> Result<(Geometry, [u64; 2])> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let mode = Mode::from_raw(p[31])?;
    if backward && mode == Mode::Max {
        return Err(Error::InvalidField);
    }
    let geometry = geometry(&p[..GEOMETRY], mode)?;
    let shape = if backward {
        [p[0], p[1], p[5], p[6], p[7]]
    } else {
        [p[0], p[1], p[2], p[3], p[4]]
    };
    if p[24] == 0 || p[24] > i32::MAX as u32 || p[25] > p[24] {
        return Err(Error::InvalidField);
    }
    if !shape.contains(&0) {
        let mut low = i64::from(p[25]);
        let mut high = low;
        for (&dim, &stride) in shape.iter().zip(&p[26..31]) {
            let span = i64::from(dim - 1) * i64::from(stride as i32);
            low = low.checked_add(span.min(0)).ok_or(Error::InvalidField)?;
            high = high.checked_add(span.max(0)).ok_or(Error::InvalidField)?;
        }
        if low < 0 || high >= i64::from(p[24]) {
            return Err(Error::InvalidField);
        }
    }
    Ok((
        geometry,
        [u64::from(p[24]), u64::from(if backward { geometry.images } else { geometry.storage })],
    ))
}
