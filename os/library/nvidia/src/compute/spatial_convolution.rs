//! Checked canonical NCDHW window extraction and its transpose.
use crate::{Error, Result};
pub const GEOMETRY: usize = 25;
pub const PARAMETERS: usize = 32;
pub const TRANSPOSED: u32 = 1 << 31;
fn count(values: &[u32]) -> Result<u32> {
    values
        .iter()
        .try_fold(1u32, |n, &v| n.checked_mul(v).filter(|&n| n <= i32::MAX as u32).ok_or(Error::InvalidField))
}
pub fn output_extent(input: u32, kernel: u32, stride: u32, before: u32, after: u32, dilation: u32) -> Result<u32> {
    if [input, kernel, stride, dilation].contains(&0) || [input, kernel, stride, before, after, dilation].iter().any(|&v| v > i32::MAX as u32) {
        return Err(Error::InvalidField);
    }
    let n = i64::from(input) + i64::from(before) + i64::from(after) - 1 - i64::from(kernel - 1) * i64::from(dilation);
    if n < 0 {
        return Err(Error::InvalidField);
    }
    u32::try_from(n / i64::from(stride) + 1)
        .ok()
        .filter(|&n| n <= i32::MAX as u32)
        .ok_or(Error::InvalidField)
}
pub fn transpose_extent(input: u32, kernel: u32, stride: u32, padding: u32, extra: u32, dilation: u32) -> Result<u32> {
    if [input, kernel, stride, dilation].contains(&0) || [input, kernel, stride, padding, extra, dilation].iter().any(|&v| v > i32::MAX as u32) {
        return Err(Error::InvalidField);
    }
    let n = i64::from(input - 1) * i64::from(stride) + i64::from(kernel - 1) * i64::from(dilation) + 1 + i64::from(extra) - 2 * i64::from(padding);
    u32::try_from(n).ok().filter(|&n| n > 0 && n <= i32::MAX as u32).ok_or(Error::InvalidField)
}
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub images: u32,
    pub weights: u32,
    pub grid: u32,
    pub columns: u32,
    pub groups: u32,
    pub inner: u32,
    pub spatial: u32,
}
/// N,C,D,H,W,O,KD,KH,KW,GD,GH,GW,SD,SH,SW,PD,PH,PW,DD,DH,DW,groups|transpose,after/extra[3].
pub fn geometry(p: &[u32]) -> Result<Geometry> {
    let p: &[u32; GEOMETRY] = p.try_into().map_err(|_| Error::InvalidField)?;
    let groups = p[21] & !TRANSPOSED;
    if p[..21].iter().chain(&p[22..]).any(|&n| n > i32::MAX as u32)
        || p[1..15].contains(&0)
        || p[18..21].contains(&0)
        || groups == 0
        || !p[1].is_multiple_of(groups)
        || !p[5].is_multiple_of(groups)
    {
        return Err(Error::InvalidField);
    }
    for axis in 0..3 {
        let expected = if p[21] & TRANSPOSED != 0 {
            transpose_extent(p[9 + axis], p[6 + axis], p[12 + axis], p[15 + axis], p[22 + axis], p[18 + axis])?
        } else {
            output_extent(p[2 + axis], p[6 + axis], p[12 + axis], p[15 + axis], p[22 + axis], p[18 + axis])?
        };
        if expected != p[if p[21] & TRANSPOSED != 0 { 2 + axis } else { 9 + axis }] {
            return Err(Error::InvalidField);
        }
    }
    let inner = count(&[p[1] / groups, p[6], p[7], p[8]])?;
    let spatial = count(&p[9..12])?;
    count(&[p[0], spatial])?;
    Ok(Geometry {
        images: count(&p[..5])?,
        weights: count(&[p[5], inner])?,
        grid: count(&[p[0], p[5], spatial])?,
        columns: count(&[p[0], groups, inner, spatial])?,
        groups,
        inner,
        spatial,
    })
}
pub fn validate_windows(p: &[u32]) -> Result<(Geometry, [u64; 2])> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let g = geometry(&p[..GEOMETRY])?;
    if p[25] == 0 || p[25] > i32::MAX as u32 || p[26] > p[25] {
        return Err(Error::InvalidField);
    }
    if p[0] != 0 {
        let mut low = i64::from(p[26]);
        let mut high = low;
        for (&dim, &stride) in p[..5].iter().zip(&p[27..]) {
            let span = i64::from(dim - 1) * i64::from(stride as i32);
            low = low.checked_add(span.min(0)).ok_or(Error::InvalidField)?;
            high = high.checked_add(span.max(0)).ok_or(Error::InvalidField)?;
        }
        if low < 0 || high >= i64::from(p[25]) {
            return Err(Error::InvalidField);
        }
    }
    Ok((g, [u64::from(p[25]), u64::from(g.columns)]))
}
