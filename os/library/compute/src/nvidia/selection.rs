//! Parallel GPU selection with one small host read for exact candidate pairs.
use super::{GpuBuffer, Kernel, Nvidia, Storage};
use crate::{Error, Result};

fn combine(partials: &[u32; 64], length: usize) -> Result<usize> {
    let mut best = (usize::MAX, f32::NEG_INFINITY);
    for pair in partials.chunks_exact(2) {
        let index = pair[0] as usize;
        if index == u32::MAX as usize {
            continue;
        }
        let value = f32::from_bits(pair[1]);
        if !value.is_finite() {
            return Err(Error::NonFinite);
        }
        if index >= length {
            return Err(Error::DeviceFault);
        }
        if value > best.1 || (value == best.1 && index < best.0) {
            best = (index, value);
        }
    }
    if best.0 == usize::MAX { Err(Error::DeviceFault) } else { Ok(best.0) }
}

impl Nvidia {
    /// Select the first maximum among FP32 bit patterns in resident words.
    /// Thirty-two GPU blocks return exact index/value pairs in one 256-byte read.
    pub fn argmax_words(&self, input: &GpuBuffer<u32>, offset: usize, length: usize) -> Result<usize> {
        self.argmax_buffer(input, offset, length)
    }

    pub(super) fn argmax_buffer<T>(&self, input: &GpuBuffer<T>, offset: usize, length: usize) -> Result<usize> {
        let output = self.produce_storage(64, Kernel::ArgMax, &[(input, offset)], &[length], Storage::Words, &[])?;
        let mut partials = [0u32; 64];
        self.read_words_into(&output, 0, &mut partials)?;
        combine(&partials, length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> [u32; 64] {
        core::array::from_fn(|i| if i % 2 == 0 { u32::MAX } else { f32::NEG_INFINITY.to_bits() })
    }

    #[test]
    fn candidates_keep_global_tie_order_and_reject_nonfinite_losers() {
        let mut values = empty();
        values[..6].copy_from_slice(&[4096, (-2f32).to_bits(), 128, (-2f32).to_bits(), 256, (-3f32).to_bits()]);
        assert_eq!(combine(&values, 8192), Ok(128));
        values[5] = (-1f32).to_bits();
        assert_eq!(combine(&values, 8192), Ok(256));
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            values[5] = invalid.to_bits();
            assert_eq!(combine(&values, 8192), Err(Error::NonFinite));
        }
    }

    #[test]
    fn candidates_validate_bounds_and_signed_zero_ties() {
        assert_eq!(combine(&empty(), 8192), Err(Error::DeviceFault));
        let mut values = empty();
        values[..4].copy_from_slice(&[4096, (-0f32).to_bits(), 128, 0f32.to_bits()]);
        assert_eq!(combine(&values, 8192), Ok(128));
        assert_eq!(combine(&values, 4096), Err(Error::DeviceFault));
    }
}
