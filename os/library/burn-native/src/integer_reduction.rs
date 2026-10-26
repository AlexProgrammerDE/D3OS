//! Exact integer reduction plans over logical element layouts.
use crate::{NativeTensor, integer_arithmetic::integer_scalar_bits, layout::Layout, storage::format};
use alloc::sync::Arc;
use burn_backend::{DType, Scalar, Shape};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, integer_reduction as abi, reduction::Reduction, words::WordOp};

pub(crate) struct Plan {
    pub parameters: [usize; abi::PARAMETERS],
    pub layout: Layout,
    pub output_words: usize,
}

/// Encode actual logical storage extent, including zero for fresh empties.
/// The shared validator handles signed spans and the one-word no-read binding.
pub(crate) fn plan(layout: &Layout, dtype: DType, axis: Option<usize>, reduction: Reduction) -> Result<Plan> {
    let input = format(dtype)?;
    let operation = abi::Operation::from_reduction(reduction).map_err(|_| Error::InvalidParameter)?;
    let control = abi::control(operation, input, input).map_err(|_| Error::InvalidParameter)?;
    let output = Layout::contiguous(match axis {
        Some(axis) => layout.reduction_shape(axis)?,
        None => Shape::new([1]),
    })?;
    let mut p = [0u32; abi::PARAMETERS];
    p[..8].copy_from_slice(&[
        output.len() as u32,
        layout.shape().num_dims().max(1) as u32,
        control,
        axis.map_or(abi::ALL_AXES, |axis| axis as u32),
        layout.offset() as u32,
        layout.storage_elements() as u32,
        axis.map_or(layout.len(), |axis| layout.shape()[axis]) as u32,
        0,
    ]);
    p[8..16].fill(1);
    for (axis, (&dim, &stride)) in layout.shape().iter().zip(layout.strides()).enumerate() {
        p[8 + axis] = dim as u32;
        p[16 + axis] = i32::try_from(stride).map_err(|_| Error::InvalidParameter)? as u32;
    }
    if layout.shape().num_dims() == 0 {
        p[16] = 1;
    }
    let geometry = abi::geometry(&p).map_err(|_| Error::InvalidParameter)?;
    Ok(Plan {
        parameters: p.map(|value| value as usize),
        layout: output,
        output_words: usize::try_from(geometry.output_words).map_err(|_| Error::SizeOverflow)?,
    })
}

fn mean_count(dtype: DType, count: usize) -> Result<Scalar> {
    if count == 0 {
        return Err(Error::InvalidParameter);
    }
    let count = Scalar::UInt(u64::try_from(count).map_err(|_| Error::SizeOverflow)?);
    integer_scalar_bits(dtype, count)?;
    Ok(count)
}

impl NativeTensor {
    pub(crate) fn integer_reduce(&self, axis: Option<usize>, operation: Reduction) -> Result<Self> {
        let plan = plan(&self.layout, self.dtype, axis, operation)?;
        if plan.layout.is_empty() {
            return Self::empty(plan.layout.shape().clone(), self.dtype);
        }
        let buffer = Nvidia.produce_words(plan.output_words, Kernel::IntegerReduction, &[(&self.buffer, 0)], &plan.parameters)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout: plan.layout,
            dtype: self.dtype,
        })
    }

    pub(crate) fn integer_mean(&self) -> Result<Self> {
        // Burn materializes Sum in the input dtype before scalar division.
        // Check the count before entering a command scope or touching storage.
        let count = mean_count(self.dtype, self.layout.len())?;
        plan(&self.layout, self.dtype, None, Reduction::Sum)?;
        Nvidia.record_commands(|| self.reduce_all(Reduction::Sum)?.integer_scalar(WordOp::Divide, count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use d3os_nvidia::compute::cast::Format;

    const DTYPES: [DType; 8] = [DType::I64, DType::I32, DType::I16, DType::I8, DType::U64, DType::U32, DType::U16, DType::U8];

    #[test]
    fn every_integer_operation_retains_dtype_and_declares_its_accumulator() {
        let layout = Layout::contiguous(Shape::new([3, 129])).unwrap();
        for dtype in DTYPES {
            let input = format(dtype).unwrap();
            for operation in Reduction::ALL.into_iter().chain(Reduction::INTEGERS) {
                let p = plan(&layout, dtype, Some(1), operation).unwrap();
                assert_eq!(p.layout.shape(), &Shape::new([3, 1]));
                assert_eq!((p.parameters[2] >> 8) & 0xff, input as usize);
                assert_eq!((p.parameters[2] >> 16) & 0xff, input as usize);
                let accumulator = if (operation as u32 % 7) >= 3 {
                    input
                } else {
                    match dtype {
                        DType::I64 => Format::I64,
                        DType::I32 | DType::I16 | DType::I8 => Format::I32,
                        DType::U64 => Format::U64,
                        _ => Format::U32,
                    }
                };
                assert_eq!(p.parameters[2] >> 24, accumulator as usize);
                assert_eq!(p.output_words, input.width().words(3) as usize);
            }
        }
    }

    #[test]
    fn empty_wide_identities_need_one_input_word_and_full_output_storage() {
        let layout = Layout::contiguous(Shape::new([2, 0])).unwrap();
        for dtype in [DType::I64, DType::U64] {
            for operation in [Reduction::Sum, Reduction::Product] {
                let p = plan(&layout, dtype, Some(1), operation).unwrap();
                assert_eq!(p.layout.shape(), &Shape::new([2, 1]));
                assert_eq!(p.output_words, 4);
                assert_eq!(p.parameters[5], 0);
                let raw = p.parameters.map(|v| v as u32);
                assert_eq!(abi::validate(&raw).unwrap(), [1, 4]);
            }
        }
    }

    #[test]
    fn empty_outer_dimensions_validate_the_reduced_axis_before_returning() {
        for dtype in DTYPES {
            let positive = Layout::contiguous(Shape::new([0, 3])).unwrap();
            for operation in Reduction::ALL {
                let p = plan(&positive, dtype, Some(1), operation).unwrap();
                assert!(p.layout.is_empty());
                assert_eq!(p.output_words, 1);
                let raw = p.parameters.map(|v| v as u32);
                assert!(abi::geometry(&raw).is_ok());
                assert!(abi::validate(&raw).is_err());
            }
            for shape in [[0, 0], [2, 0]] {
                let zero = Layout::contiguous(Shape::new(shape)).unwrap();
                for operation in [Reduction::Mean, Reduction::Min, Reduction::Max, Reduction::ArgMin, Reduction::ArgMax] {
                    assert!(plan(&zero, dtype, Some(1), operation).is_err());
                }
                assert!(plan(&zero, dtype, Some(2), Reduction::Sum).is_err());
            }
        }
        let empty = Layout::contiguous(Shape::new([0, 3])).unwrap();
        for dtype in [DType::F32, DType::Bool(burn_backend::BoolStore::U32)] {
            assert!(plan(&empty, dtype, Some(1), Reduction::Sum).is_err());
        }
    }

    #[test]
    fn views_preserve_offsets_signed_strides_and_full_declared_storage() {
        let reversed = Layout::from_parts(Shape::new([2, 3]), vec![-3, -1], 5, 6).unwrap();
        let p = plan(&reversed, DType::I64, Some(1), Reduction::Min).unwrap();
        assert_eq!(&p.parameters[4..7], &[5, 6, 3]);
        assert_eq!(&p.parameters[16..18], &[(-3i32) as u32 as usize, (-1i32) as u32 as usize]);
        assert_eq!(abi::validate(&p.parameters.map(|v| v as u32)).unwrap(), [12, 4]);
        let broadcast = Layout::from_parts(Shape::new([2, 3]), vec![0, 0], 0, 1).unwrap();
        let p = plan(&broadcast, DType::U64, Some(1), Reduction::Max).unwrap();
        assert_eq!(abi::validate(&p.parameters.map(|v| v as u32)).unwrap(), [2, 4]);
        let offset = Layout::from_parts(Shape::new([1]), vec![1], 3, 8).unwrap();
        let p = plan(&offset, DType::U64, None, Reduction::Sum).unwrap();
        assert_eq!(abi::validate(&p.parameters.map(|v| v as u32)).unwrap(), [16, 2]);
        let empty = Layout::from_parts(Shape::new([2, 0]), vec![i32::MIN as i64, i32::MAX as i64], 8, 8).unwrap();
        let p = plan(&empty, DType::U64, Some(1), Reduction::Product).unwrap();
        assert_eq!(abi::validate(&p.parameters.map(|v| v as u32)).unwrap(), [1, 4]);
    }

    #[test]
    fn global_scalars_promote_to_one_physical_dimension_and_args_reject() {
        let scalar = Layout::from_parts(Shape::new([]), vec![], 3, 7).unwrap();
        for dtype in DTYPES {
            for operation in [Reduction::Sum, Reduction::Mean, Reduction::Product, Reduction::Min, Reduction::Max] {
                let p = plan(&scalar, dtype, None, operation).unwrap();
                assert_eq!(p.layout.shape(), &Shape::new([1]));
                assert_eq!(&p.parameters[3..7], &[u32::MAX as usize, 3, 7, 1]);
                assert_eq!(&p.parameters[8..16], &[1; 8]);
                assert_eq!(&p.parameters[16..24], &[1, 0, 0, 0, 0, 0, 0, 0]);
                assert!(plan(&scalar, dtype, Some(0), operation).is_err());
            }
            for operation in [Reduction::ArgMin, Reduction::ArgMax] {
                assert!(plan(&scalar, dtype, None, operation).is_err());
            }
        }
    }

    #[test]
    fn zero_aware_shapes_do_not_hide_an_overflowing_output_count() {
        let maximum = i32::MAX as usize;
        let layout = Layout::contiguous(Shape::new([0, maximum, maximum])).unwrap();
        assert!(plan(&layout, DType::I64, Some(1), Reduction::Min).unwrap().layout.is_empty());
        assert!(plan(&layout, DType::I64, Some(0), Reduction::Sum).is_err());
        let p = plan(&layout, DType::U64, None, Reduction::Product).unwrap();
        assert_eq!(abi::validate(&p.parameters.map(|v| v as u32)).unwrap(), [1, 2]);
    }

    #[test]
    fn global_mean_count_uses_checked_input_dtype_conversion() {
        for (dtype, maximum) in [(DType::I8, 127), (DType::U8, 255), (DType::I16, 32767), (DType::U16, 65535)] {
            assert!(mean_count(dtype, maximum).is_ok());
            assert!(matches!(mean_count(dtype, maximum + 1), Err(Error::InvalidParameter)));
            let axis = Layout::contiguous(Shape::new([maximum + 1])).unwrap();
            assert!(plan(&axis, dtype, Some(0), Reduction::Mean).is_ok());
        }
        for dtype in DTYPES {
            assert!(matches!(mean_count(dtype, 0), Err(Error::InvalidParameter)));
            assert!(matches!(mean_count(dtype, 1), Ok(Scalar::UInt(1))));
        }
        for dtype in [DType::I32, DType::U32, DType::I64, DType::U64] {
            assert!(mean_count(dtype, i32::MAX as usize).is_ok());
        }
        assert!(matches!(mean_count(DType::F32, 1), Err(Error::InvalidParameter)));
    }
}
