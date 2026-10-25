//! One separable interpolation axis and its transpose over signed NCHW views.
use crate::{Error, Result};
pub const PARAMETERS: usize = 32;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Mode {
    Nearest = 0,
    NearestExact = 1,
    Bilinear = 2,
    Bicubic = 3,
    Lanczos3 = 4,
}
impl Mode {
    pub const ALL: [Self; 5] = [Self::Nearest, Self::NearestExact, Self::Bilinear, Self::Bicubic, Self::Lanczos3];
    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.get(raw as usize).copied().ok_or(Error::InvalidField)
    }
    pub fn symbol(self, backward: bool) -> &'static str {
        match (self, backward) {
            (Self::Nearest, false) => "burn_interpolate_nearest",
            (Self::Nearest, true) => "burn_interpolate_nearest_backward",
            (Self::NearestExact, false) => "burn_interpolate_exact",
            (Self::NearestExact, true) => "burn_interpolate_exact_backward",
            (Self::Bilinear, false) => "burn_interpolate_linear",
            (Self::Bilinear, true) => "burn_interpolate_linear_backward",
            (Self::Bicubic, false) => "burn_interpolate_cubic",
            (Self::Bicubic, true) => "burn_interpolate_cubic_backward",
            (Self::Lanczos3, false) => "burn_interpolate_lanczos3",
            (Self::Lanczos3, true) => "burn_interpolate_lanczos3_backward",
        }
    }
}
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}
/// Reduced exact source coordinate `(output * scale + offset) / divisor`.
pub fn transform(input: u32, output: u32, mode: Mode, align: bool) -> Result<[u32; 3]> {
    if input == 0 || output == 0 || input > i32::MAX as u32 || output > i32::MAX as u32 {
        return Err(Error::InvalidField);
    }
    let (a, b, d) = match mode {
        Mode::Nearest => (input, 0, output),
        Mode::NearestExact => (2 * input, i64::from(input), 2 * output),
        _ if align => (input - 1, 0, (output - 1).max(1)),
        _ => (2 * input, i64::from(input) - i64::from(output), 2 * output),
    };
    let common = gcd(gcd(a, d), b.unsigned_abs() as u32).max(1);
    Ok([a / common, ((b / i64::from(common)) as i32) as u32, d / common])
}
fn count(shape: &[u32]) -> Result<u32> {
    shape
        .iter()
        .try_fold(1u32, |n, &d| n.checked_mul(d).filter(|&v| v <= i32::MAX as u32).ok_or(Error::InvalidField))
}
/// N,C,H,W,target,axis,mode,align,extent,offset,strides[4],backward,transform[3],reserved[14].
pub fn validate(p: &[u32]) -> Result<([u64; 2], u32)> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let mode = Mode::from_raw(p[6])?;
    if !(2..=3).contains(&p[5])
        || p[7] > 1
        || p[14] > 1
        || p[18..].iter().any(|&x| x != 0)
        || p[..5].iter().any(|&x| x > i32::MAX as u32)
        || p[2..5].contains(&0)
        || p[8] == 0
        || p[8] > i32::MAX as u32
        || p[9] > p[8]
    {
        return Err(Error::InvalidField);
    }
    let axis = p[5] as usize;
    if p[15..18] != transform(p[axis], p[4], mode, p[7] != 0)? {
        return Err(Error::InvalidField);
    }
    count(&p[2..4])?;
    let mut original = [p[0], p[1], p[2], p[3]];
    let mut resized = original;
    resized[axis] = p[4];
    count(&resized[2..])?;
    let input = count(&original)?;
    let output = count(&resized)?;
    let destination = if p[14] != 0 {
        original = resized;
        input
    } else {
        output
    };
    if !original.contains(&0) {
        let mut low = i64::from(p[9]);
        let mut high = low;
        for (&dim, &stride) in original.iter().zip(&p[10..14]) {
            let span = i64::from(dim - 1) * i64::from(stride as i32);
            low = low.checked_add(span.min(0)).ok_or(Error::InvalidField)?;
            high = high.checked_add(span.max(0)).ok_or(Error::InvalidField)?;
        }
        if low < 0 || high >= i64::from(p[8]) {
            return Err(Error::InvalidField);
        }
    }
    Ok(([u64::from(p[8]), u64::from(destination)], destination))
}
