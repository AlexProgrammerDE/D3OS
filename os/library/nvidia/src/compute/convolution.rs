//! Checked NCHW convolution geometry and signed input/weight views.
use crate::{Error, Result};

pub const GEOMETRY: usize = 18;
pub const WINDOW_PARAMETERS: usize = GEOMETRY + 6;
pub const PARAMETERS: usize = GEOMETRY + 13;

fn count(values: &[u32]) -> Result<u32> {
    values
        .iter()
        .try_fold(1u32, |n, &v| n.checked_mul(v).filter(|&n| n <= i32::MAX as u32).ok_or(Error::InvalidField))
}

pub fn output_extent(input: u32, kernel: u32, stride: u32, before: u32, after: u32, dilation: u32) -> Result<u32> {
    if [input, kernel, stride, dilation].contains(&0) || [input, kernel, stride, before, after, dilation].iter().any(|&v| v > i32::MAX as u32) {
        return Err(Error::InvalidField);
    }
    let padded = input
        .checked_add(before)
        .and_then(|v| v.checked_add(after))
        .filter(|&v| v <= i32::MAX as u32)
        .ok_or(Error::InvalidField)?;
    let kernel = (kernel - 1).checked_mul(dilation).and_then(|v| v.checked_add(1)).ok_or(Error::InvalidField)?;
    padded.checked_sub(kernel).map(|v| v / stride + 1).ok_or(Error::InvalidField)
}

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub images: u32,
    pub weights: u32,
    pub outputs: u32,
    pub columns: u32,
    pub blocks: u32,
}

/// N,C,H,W,KH,KW,OH,OW,SH,SW,PH,PW,DH,DW,groups,endH,endW,O.
pub fn geometry(p: &[u32]) -> Result<Geometry> {
    let p: &[u32; GEOMETRY] = p.try_into().map_err(|_| Error::InvalidField)?;
    if p.iter().any(|&v| v > i32::MAX as u32)
        || [p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[12], p[13], p[14], p[17]].contains(&0)
        || !p[1].is_multiple_of(p[14])
        || !p[17].is_multiple_of(p[14])
        || output_extent(p[2], p[4], p[8], p[10], p[15], p[12])? != p[6]
        || output_extent(p[3], p[5], p[9], p[11], p[16], p[13])? != p[7]
    {
        return Err(Error::InvalidField);
    }
    let spatial = count(&[p[6], p[7]])?;
    let columns_per_group = count(&[p[1] / p[14], p[4], p[5]])?;
    let batch_spatial = count(&[p[0], spatial])?;
    let blocks = count(&[p[14], (p[17] / p[14]).div_ceil(16), batch_spatial.div_ceil(32)])?;
    Ok(Geometry {
        images: count(&p[..4])?,
        weights: count(&[p[17], columns_per_group])?,
        outputs: count(&[p[0], p[17], spatial])?,
        columns: count(&[p[0], p[14], columns_per_group, spatial])?,
        blocks,
    })
}

/// extent, offset, and four signed strides. Empty batches have no reachable address.
pub fn validate_view(shape: [u32; 4], p: &[u32]) -> Result<u64> {
    let p: &[u32; 6] = p.try_into().map_err(|_| Error::InvalidField)?;
    if p[0] == 0 || p[0] > i32::MAX as u32 || p[1] > p[0] {
        return Err(Error::InvalidField);
    }
    if shape.contains(&0) {
        return Ok(u64::from(p[0]));
    }
    let mut low = i64::from(p[1]);
    let mut high = low;
    for (&dim, &stride) in shape.iter().zip(&p[2..]) {
        let span = i64::from(dim - 1) * i64::from(stride as i32);
        low = low.checked_add(span.min(0)).ok_or(Error::InvalidField)?;
        high = high.checked_add(span.max(0)).ok_or(Error::InvalidField)?;
    }
    if low < 0 || high >= i64::from(p[0]) {
        return Err(Error::InvalidField);
    }
    Ok(u64::from(p[0]))
}

pub fn validate(p: &[u32]) -> Result<(Geometry, [u64; 4])> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let geometry = geometry(&p[..GEOMETRY])?;
    if p[30] > 1 {
        return Err(Error::InvalidField);
    }
    let input = validate_view([p[0], p[1], p[2], p[3]], &p[18..24])?;
    let weights = validate_view([p[17], p[1] / p[14], p[4], p[5]], &p[24..30])?;
    Ok((
        geometry,
        [input, weights, if p[30] == 1 { u64::from(p[17]) } else { 1 }, u64::from(geometry.outputs)],
    ))
}

pub fn validate_windows(p: &[u32]) -> Result<(Geometry, [u64; 2])> {
    let p: &[u32; WINDOW_PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let geometry = geometry(&p[..GEOMETRY])?;
    let input = validate_view([p[0], p[1], p[2], p[3]], &p[18..24])?;
    Ok((geometry, [input, u64::from(geometry.columns)]))
}
