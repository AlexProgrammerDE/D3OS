//! Exact floating extrema plans and typed storage dispatch.
use crate::{NativeTensor, layout::Layout, storage::format};
use alloc::sync::Arc;
use burn_backend::{DType, Shape};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, float_extrema as abi, reduction::Reduction};

pub(crate) struct Plan {
    pub parameters: [usize; abi::PARAMETERS],
    pub layout: Layout,
    pub output_words: usize,
}

/// Validate both dtypes and the complete signed view before empty handling.
/// Coordinates stay U32 inside the kernel; only its final index store narrows.
pub(crate) fn plan(layout: &Layout, dtype: DType, axis: Option<usize>, reduction: Reduction, output_dtype: DType) -> Result<Plan> {
    if reduction.is_integer() {
        return Err(Error::InvalidParameter);
    }
    let operation = abi::Operation::from_reduction(reduction).map_err(|_| Error::InvalidParameter)?;
    let control = abi::control(operation, format(dtype)?, format(output_dtype)?).map_err(|_| Error::InvalidParameter)?;
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

impl NativeTensor {
    pub(crate) fn float_extrema(&self, axis: Option<usize>, operation: Reduction, output_dtype: DType) -> Result<Self> {
        let plan = plan(&self.layout, self.dtype, axis, operation, output_dtype)?;
        if self.dtype == DType::F32 {
            // F32 keeps its cooperative axis kernels, reverse-axis global
            // staging, and empty aliases, including requested index casts.
            let reduced = match axis {
                Some(axis) => self.reduce(axis, operation)?,
                None => self.reduce_all(operation)?,
            };
            return if operation.index() { reduced.cast(output_dtype) } else { Ok(reduced) };
        }
        if plan.layout.is_empty() {
            return Self::empty(plan.layout.shape().clone(), output_dtype);
        }
        let buffer = Nvidia.produce_words(plan.output_words, Kernel::FloatExtrema, &[(&self.buffer, 0)], &plan.parameters)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout: plan.layout,
            dtype: output_dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use burn_backend::{BoolStore, quantization::QuantScheme};
    use d3os_nvidia::compute::{cast::Format, storage::Width};

    const FLOATS: [DType; 5] = [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16];
    const INDICES: [DType; 8] = [DType::I64, DType::I32, DType::I16, DType::I8, DType::U64, DType::U32, DType::U16, DType::U8];

    #[test]
    fn value_and_index_projections_keep_exact_input_tags_and_packed_output_sizes() {
        let layout = Layout::contiguous(Shape::new([3, 257])).unwrap();
        for dtype in FLOATS {
            let input = format(dtype).unwrap();
            for operation in [Reduction::Min, Reduction::Max, Reduction::ArgMin, Reduction::ArgMax] {
                let outputs = if operation.index() { &INDICES[..] } else { core::slice::from_ref(&dtype) };
                for &output_dtype in outputs {
                    let output = format(output_dtype).unwrap();
                    let p = plan(&layout, dtype, Some(1), operation, output_dtype).unwrap();
                    assert_eq!(p.layout.shape(), &Shape::new([3, 1]));
                    assert_eq!((p.parameters[2] >> 8) & 0xff, input as usize);
                    assert_eq!((p.parameters[2] >> 16) & 0xff, output as usize);
                    assert_eq!(p.parameters[2] >> 24, input as usize);
                    assert_eq!(p.output_words, output.width().words(3) as usize);
                    let raw = p.parameters.map(|v| v as u32);
                    assert_eq!(abi::validate(&raw).unwrap(), [input.width().words(771), output.width().words(3)]);
                    let g = abi::geometry(&raw).unwrap();
                    let owners = if output.width() == Width::Eight { 3 } else { p.output_words as u32 };
                    assert_eq!(g.owners, owners);
                    assert_eq!(abi::work_items(&raw).unwrap(), owners * abi::THREADS);
                }
            }
        }
    }

    #[test]
    fn signed_rank_eight_and_overlapping_views_encode_original_storage() {
        let layout = Layout::from_parts(Shape::new([2, 1, 1, 1, 1, 1, 2, 3]), vec![-3, 0, 0, 0, 0, 0, 0, -1], 8, 12).unwrap();
        for dtype in FLOATS {
            for axis in [Some(7), None] {
                let p = plan(&layout, dtype, axis, Reduction::Min, dtype).unwrap();
                assert_eq!(&p.parameters[4..6], &[8, 12]);
                assert_eq!(&p.parameters[16..24], &[(-3i32) as u32 as usize, 0, 0, 0, 0, 0, 0, (-1i32) as u32 as usize]);
                assert_eq!(
                    abi::validate(&p.parameters.map(|v| v as u32)).unwrap()[0],
                    format(dtype).unwrap().width().words(12)
                );
            }
        }
    }

    #[test]
    fn scalar_globals_promote_metadata_and_reject_raw_global_indices() {
        let scalar = Layout::from_parts(Shape::new([]), vec![], 3, 7).unwrap();
        for dtype in FLOATS {
            for operation in [Reduction::Min, Reduction::Max] {
                let p = plan(&scalar, dtype, None, operation, dtype).unwrap();
                assert_eq!(p.layout.shape(), &Shape::new([1]));
                assert_eq!(&p.parameters[1..2], &[1]);
                assert_eq!(&p.parameters[3..7], &[u32::MAX as usize, 3, 7, 1]);
                assert_eq!(&p.parameters[8..16], &[1; 8]);
                assert_eq!(&p.parameters[16..24], &[1, 0, 0, 0, 0, 0, 0, 0]);
                assert!(plan(&scalar, dtype, Some(0), operation, dtype).is_err());
            }
            for operation in [Reduction::ArgMin, Reduction::ArgMax] {
                for index_dtype in INDICES {
                    assert!(plan(&scalar, dtype, None, operation, index_dtype).is_err());
                }
            }
        }
    }

    #[test]
    fn empty_outputs_require_positive_axes_and_one_dummy_word_for_every_dtype() {
        let empty = Layout::from_parts(Shape::new([0, 3]), vec![i32::MIN as i64, i32::MAX as i64], 8, 8).unwrap();
        for dtype in FLOATS {
            for operation in [Reduction::Min, Reduction::Max, Reduction::ArgMin, Reduction::ArgMax] {
                let outputs = if operation.index() { &INDICES[..] } else { core::slice::from_ref(&dtype) };
                for &output in outputs {
                    let p = plan(&empty, dtype, Some(1), operation, output).unwrap();
                    assert_eq!(p.layout.shape(), &Shape::new([0, 1]));
                    assert_eq!(p.output_words, 1);
                    let raw = p.parameters.map(|v| v as u32);
                    let g = abi::geometry(&raw).unwrap();
                    assert_eq!((g.input_words, g.output_words, g.owners), (1, 1, 0));
                    assert!(abi::validate(&raw).is_err());
                    assert!(abi::work_items(&raw).is_err());
                    assert!(abi::symbol(&raw).is_err());
                    assert!(plan(&empty, dtype, Some(2), operation, output).is_err());
                    for shape in [[0, 0], [2, 0]] {
                        let zero_axis = Layout::contiguous(Shape::new(shape)).unwrap();
                        assert!(plan(&zero_axis, dtype, Some(1), operation, output).is_err());
                    }
                }
            }
            assert!(plan(&empty, dtype, None, Reduction::Min, dtype).is_err());
        }
    }

    #[test]
    fn dtype_and_operation_validation_precedes_empty_output_handling() {
        let empty = Layout::contiguous(Shape::new([0, 3])).unwrap();
        for dtype in FLOATS {
            for operation in [Reduction::Sum, Reduction::Mean, Reduction::Product].into_iter().chain(Reduction::INTEGERS) {
                assert!(plan(&empty, dtype, Some(1), operation, dtype).is_err());
            }
            for output in FLOATS.into_iter().filter(|&output| output != dtype).chain(INDICES) {
                assert!(plan(&empty, dtype, Some(1), Reduction::Min, output).is_err());
            }
            for output in FLOATS.into_iter().chain([
                DType::Bool(BoolStore::Native),
                DType::Bool(BoolStore::U8),
                DType::Bool(BoolStore::U32),
                DType::QFloat(QuantScheme::default()),
            ]) {
                assert!(plan(&empty, dtype, Some(1), Reduction::ArgMax, output).is_err());
            }
        }
        for dtype in INDICES.into_iter().chain([
            DType::Bool(BoolStore::Native),
            DType::Bool(BoolStore::U8),
            DType::Bool(BoolStore::U32),
            DType::QFloat(QuantScheme::default()),
        ]) {
            assert!(plan(&empty, dtype, Some(1), Reduction::Min, dtype).is_err());
        }
    }

    #[test]
    fn zero_aware_products_and_wide_word_limits_remain_checked() {
        let maximum = i32::MAX as usize;
        let empty = Layout::contiguous(Shape::new([0, maximum, maximum])).unwrap();
        assert!(plan(&empty, DType::F64, Some(1), Reduction::Min, DType::F64).unwrap().layout.is_empty());
        assert!(plan(&empty, DType::F64, Some(0), Reduction::Min, DType::F64).is_err());
        let wide = Layout::from_parts(Shape::new([1]), vec![1], (maximum - 1) as i64, maximum).unwrap();
        let p = plan(&wide, DType::F64, None, Reduction::Min, DType::F64).unwrap();
        let raw = p.parameters.map(|v| v as u32);
        assert_eq!(abi::validate(&raw).unwrap(), [2 * maximum as u64, 2]);
        let mut missing_extent = raw;
        missing_extent[5] -= 1;
        assert!(abi::geometry(&missing_extent).is_err());
        let mut mismatched_selection = raw;
        mismatched_selection[2] |= (Format::F32 as u32) << 24;
        assert!(abi::geometry(&mismatched_selection).is_err());
    }

    #[test]
    fn packed_odd_tails_and_wide_owners_use_checked_launch_geometry() {
        for count in [1u32, 3, 127, 129, 257] {
            let layout = Layout::contiguous(Shape::new([count as usize, 3])).unwrap();
            for output in INDICES {
                let p = plan(&layout, DType::F64, Some(1), Reduction::ArgMax, output).unwrap();
                let raw = p.parameters.map(|v| v as u32);
                let g = abi::geometry(&raw).unwrap();
                let width = format(output).unwrap().width();
                let slots = if width == Width::Eight { 1 } else { 4 / width.bytes() };
                assert_eq!(g.owners, count.div_ceil(slots));
                assert_eq!(abi::work_items(&raw).unwrap(), g.owners);
                assert!(g.owners * slots >= count);
                assert!((g.owners - 1) * slots < count);
                assert_eq!(g.owners.div_ceil(abi::THREADS), count.div_ceil(slots * abi::THREADS));
            }
        }
    }
}
