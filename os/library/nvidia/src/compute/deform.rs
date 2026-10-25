//! Dense deformable NCHW convolution geometry and buffer contracts.
use crate::{Error, Result};
pub const PARAMETERS: usize = 18;
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub images: u32,
    pub weights: u32,
    pub outputs: u32,
    pub columns: u32,
    pub masks: u32,
    pub offsets: u32,
    pub coordinates: u32,
}
fn count(values: &[u32]) -> Result<u32> {
    values
        .iter()
        .try_fold(1u32, |n, &v| n.checked_mul(v).filter(|&n| n <= i32::MAX as u32).ok_or(Error::InvalidField))
}
/// N,C,H,W,O,KH,KW,OH,OW,SH,SW,PH,PW,DH,DW,weight_groups,offset_groups,masked.
pub fn geometry(p: &[u32]) -> Result<Geometry> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    if p.iter().any(|&v| v > i32::MAX as u32)
        || p[1..11].contains(&0)
        || p[13..17].contains(&0)
        || p[17] > 1
        || !p[1].is_multiple_of(p[15])
        || !p[4].is_multiple_of(p[15])
        || !p[1].is_multiple_of(p[16])
        || super::spatial_convolution::output_extent(p[2], p[5], p[9], p[11], p[11], p[13])? != p[7]
        || super::spatial_convolution::output_extent(p[3], p[6], p[10], p[12], p[12], p[14])? != p[8]
    {
        return Err(Error::InvalidField);
    }
    count(&[p[1], p[5], p[6]])?;
    count(&[2, p[16], p[5], p[6]])?;
    count(&[p[0], p[7], p[8]])?;
    let masks = count(&[p[0], p[16], p[5], p[6], p[7], p[8]])?;
    Ok(Geometry {
        images: count(&p[..4])?,
        weights: count(&[p[4], p[1] / p[15], p[5], p[6]])?,
        outputs: count(&[p[0], p[4], p[7], p[8]])?,
        columns: count(&[p[1], p[5], p[6], p[0], p[7], p[8]])?,
        masks,
        offsets: count(&[2, masks])?,
        coordinates: count(&[2 + p[17], masks])?,
    })
}
