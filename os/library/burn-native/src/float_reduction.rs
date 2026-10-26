//! Typed numeric axis reductions and resident, storage-rounded global stages.
use crate::{NativeTensor, layout::Layout, storage::format};
use alloc::{sync::Arc, vec::Vec};
use burn_backend::{DType, Shape};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, float_reduction as abi, reduction::Reduction};

pub(crate) struct Plan {
    pub parameters: [usize; abi::PARAMETERS],
    pub layout: Layout,
    pub output_words: usize,
}

/// Encode only a real axis. The SDK validates signed spans and independent
/// zero-aware input/output counts before accepting no-read geometry.
pub(crate) fn plan(layout: &Layout, dtype: DType, axis: usize, reduction: Reduction) -> Result<Plan> {
    let operation = abi::Operation::from_reduction(reduction).map_err(|_| Error::InvalidParameter)?;
    let control = abi::control(operation, format(dtype)?).map_err(|_| Error::InvalidParameter)?;
    let output = Layout::contiguous(layout.reduction_shape(axis)?)?;
    let mut p = [0u32; abi::PARAMETERS];
    p[..8].copy_from_slice(&[
        output.len() as u32,
        layout.shape().num_dims() as u32,
        control,
        axis as u32,
        layout.offset() as u32,
        layout.storage_elements() as u32,
        layout.shape()[axis] as u32,
        abi::SERIAL_POLICY,
    ]);
    p[8..16].fill(1);
    for (axis, (&dim, &stride)) in layout.shape().iter().zip(layout.strides()).enumerate() {
        p[8 + axis] = dim as u32;
        p[16 + axis] = i32::try_from(stride).map_err(|_| Error::InvalidParameter)? as u32;
    }
    let geometry = abi::geometry(&p).map_err(|_| Error::InvalidParameter)?;
    Ok(Plan {
        parameters: p.map(|value| value as usize),
        layout: output,
        output_words: usize::try_from(geometry.output_words).map_err(|_| Error::SizeOverflow)?,
    })
}

enum GlobalPlan {
    Relabel(Layout),
    Fill(u64),
    Axes(Vec<usize>),
}

fn global_plan(layout: &Layout, dtype: DType, reduction: Reduction) -> Result<GlobalPlan> {
    let operation = abi::Operation::from_reduction(reduction).map_err(|_| Error::InvalidParameter)?;
    abi::control(operation, format(dtype)?).map_err(|_| Error::InvalidParameter)?;
    // Globals have no raw ALL packet. Validate the original metadata without
    // constructing a hypothetical axis output that may be too large when empty.
    let layout = Layout::from_parts(
        layout.shape().clone(),
        layout.strides().to_vec(),
        layout.offset() as i64,
        layout.storage_elements(),
    )?;
    if layout.shape().num_dims() == 0 {
        return Ok(GlobalPlan::Relabel(layout.reshape(Shape::new([1]))?));
    }
    if layout.is_empty() {
        let bits = match operation {
            abi::Operation::Sum => 0,
            abi::Operation::Product => match dtype {
                DType::F64 => 0x3ff0000000000000,
                DType::Flex32 => 0x3f800000,
                DType::F16 => 0x3c00,
                DType::BF16 => 0x3f80,
                _ => return Err(Error::InvalidParameter),
            },
            abi::Operation::Mean => match dtype {
                DType::F64 => 0x7ff8000000000000,
                DType::Flex32 => 0x7fc00000,
                DType::F16 => 0x7e00,
                DType::BF16 => 0x7fc0,
                _ => return Err(Error::InvalidParameter),
            },
        };
        return Ok(GlobalPlan::Fill(bits));
    }
    let mut axes: Vec<_> = (0..layout.shape().num_dims()).collect();
    axes.sort_by_key(|&axis| layout.shape()[axis]);
    Ok(GlobalPlan::Axes(axes))
}

impl NativeTensor {
    pub(crate) fn float_reduce(&self, axis: usize, operation: Reduction) -> Result<Self> {
        let plan = plan(&self.layout, self.dtype, axis, operation)?;
        if plan.layout.is_empty() {
            return Self::empty(plan.layout.shape().clone(), self.dtype);
        }
        let buffer = Nvidia.produce_words(plan.output_words, Kernel::FloatReduction, &[(&self.buffer, 0)], &plan.parameters)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout: plan.layout,
            dtype: self.dtype,
        })
    }

    pub(crate) fn float_reduce_all(&self, operation: Reduction) -> Result<Self> {
        let plan = global_plan(&self.layout, self.dtype, operation)?;
        match plan {
            GlobalPlan::Relabel(layout) => Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: self.dtype,
            }),
            GlobalPlan::Fill(bits) => Self::fill_bits(Shape::new([1]), self.dtype, bits),
            GlobalPlan::Axes(axes) => Nvidia.record_commands(|| {
                let mut tensor = self.clone();
                for axis in axes {
                    // Every stage writes the declared storage dtype, including
                    // Mean's division before its own storage conversion.
                    tensor = tensor.float_reduce(axis, operation)?;
                }
                tensor.reshape(Shape::new([1]))
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use burn_backend::{BoolStore, f16, quantization::QuantScheme};
    use d3os_nvidia::compute::cast::Format;

    const DTYPES: [DType; 4] = [DType::F64, DType::Flex32, DType::F16, DType::BF16];
    const OPERATIONS: [Reduction; 3] = [Reduction::Sum, Reduction::Mean, Reduction::Product];

    #[test]
    fn axis_plans_preserve_storage_tags_accumulators_and_packed_extents() {
        for count in [1, 3, 129, 257] {
            let layout = Layout::contiguous(Shape::new([count, 31])).unwrap();
            for dtype in DTYPES {
                let input = format(dtype).unwrap();
                for operation in OPERATIONS {
                    let p = plan(&layout, dtype, 1, operation).unwrap();
                    assert_eq!(p.layout.shape(), &Shape::new([count, 1]));
                    assert_eq!(p.parameters[2] & 0xff, operation as usize);
                    assert_eq!((p.parameters[2] >> 8) & 0xff, input as usize);
                    assert_eq!((p.parameters[2] >> 16) & 0xff, input as usize);
                    assert_eq!(p.parameters[2] >> 24, if dtype == DType::F64 { Format::F64 } else { Format::F32 } as usize);
                    assert_eq!(p.parameters[7], abi::SERIAL_POLICY as usize);
                    let raw = p.parameters.map(|v| v as u32);
                    let g = abi::geometry(&raw).unwrap();
                    assert_eq!(p.output_words as u64, input.width().words(count as u32));
                    assert_eq!(abi::validate(&raw).unwrap(), [input.width().words((count * 31) as u32), p.output_words as u64]);
                    assert_eq!(g.owners as usize, if dtype == DType::F64 { count } else { p.output_words });
                }
            }
        }
    }

    #[test]
    fn signed_rank_eight_broadcast_and_overlapping_views_keep_logical_geometry() {
        let layouts = [
            Layout::from_parts(Shape::new([2, 1, 1, 1, 1, 1, 2, 3]), vec![-3, 0, 0, 0, 0, 0, 0, -1], 8, 12).unwrap(),
            Layout::from_parts(Shape::new([2, 3]), vec![0, 0], 2, 4).unwrap(),
            Layout::from_parts(Shape::new([2, 3]), vec![1, 1], 1, 6).unwrap(),
        ];
        for layout in layouts {
            for dtype in DTYPES {
                let axis = layout.shape().num_dims() - 1;
                let p = plan(&layout, dtype, axis, Reduction::Mean).unwrap();
                assert_eq!(&p.parameters[4..7], &[layout.offset(), layout.storage_elements(), 3]);
                for (axis, &stride) in layout.strides().iter().enumerate() {
                    assert_eq!(p.parameters[16 + axis], stride as i32 as u32 as usize);
                }
                assert_eq!(
                    abi::validate(&p.parameters.map(|v| v as u32)).unwrap()[0],
                    format(dtype).unwrap().width().words(layout.storage_elements() as u32)
                );
            }
        }
        let maximum = i32::MAX as usize;
        let wide = Layout::from_parts(Shape::new([1]), vec![1], (maximum - 1) as i64, maximum).unwrap();
        let p = plan(&wide, DType::F64, 0, Reduction::Sum).unwrap();
        assert_eq!(abi::validate(&p.parameters.map(|v| v as u32)).unwrap(), [2 * maximum as u64, 2]);
    }

    #[test]
    fn no_read_axis_geometry_distinguishes_placeholders_from_identity_writers() {
        for shape in [[0, 3], [0, 0], [2, 0]] {
            let layout = Layout::from_parts(Shape::new(shape), vec![i32::MIN as i64, i32::MAX as i64], 8, 8).unwrap();
            for dtype in DTYPES {
                for operation in OPERATIONS {
                    let p = plan(&layout, dtype, 1, operation).unwrap();
                    let raw = p.parameters.map(|v| v as u32);
                    let g = abi::geometry(&raw).unwrap();
                    assert_eq!(g.input_words, 1);
                    if shape[0] == 0 {
                        assert!(p.layout.is_empty());
                        assert_eq!((p.output_words, g.owners), (1, 0));
                        assert!(abi::validate(&raw).is_err());
                    } else {
                        assert_eq!(p.layout.shape(), &Shape::new([2, 1]));
                        assert_eq!(abi::validate(&raw).unwrap(), [1, format(dtype).unwrap().width().words(2)]);
                    }
                    let mut invalid = raw;
                    invalid[4] = 9;
                    assert!(abi::geometry(&invalid).is_err());
                    assert!(plan(&layout, dtype, 2, operation).is_err());
                }
            }
        }
        let maximum = i32::MAX as usize;
        let enormous = Layout::contiguous(Shape::new([0, maximum, maximum])).unwrap();
        for dtype in DTYPES {
            assert!(plan(&enormous, dtype, 0, Reduction::Sum).is_err());
            assert!(plan(&enormous, dtype, 1, Reduction::Mean).unwrap().layout.is_empty());
            for operation in OPERATIONS {
                assert!(matches!(global_plan(&enormous, dtype, operation), Ok(GlobalPlan::Fill(_))));
            }
        }
    }

    #[test]
    fn unsupported_types_and_operations_reject_before_empty_or_scalar_shortcuts() {
        let layouts = [Layout::contiguous(Shape::new([0, 3])).unwrap(), Layout::contiguous(Shape::new([])).unwrap()];
        for layout in &layouts {
            for dtype in [
                DType::F32,
                DType::I32,
                DType::U64,
                DType::Bool(BoolStore::U8),
                DType::QFloat(QuantScheme::default()),
            ] {
                for operation in OPERATIONS {
                    assert!(global_plan(layout, dtype, operation).is_err());
                    assert!(plan(layout, dtype, 0, operation).is_err());
                }
            }
            for dtype in DTYPES {
                for operation in [Reduction::Min, Reduction::Max, Reduction::ArgMin, Reduction::ArgMax]
                    .into_iter()
                    .chain(Reduction::INTEGERS)
                {
                    assert!(global_plan(layout, dtype, operation).is_err());
                    assert!(plan(layout, dtype, 0, operation).is_err());
                }
            }
        }
    }

    #[test]
    fn scalar_globals_relabel_original_extent_and_empty_globals_use_canonical_identities() {
        let scalar = Layout::from_parts(Shape::new([]), vec![], 3, 7).unwrap();
        let empty = Layout::contiguous(Shape::new([0, 2, 0])).unwrap();
        for (dtype, one, nan) in [
            (DType::F64, 0x3ff0000000000000, 0x7ff8000000000000),
            (DType::Flex32, 0x3f800000, 0x7fc00000),
            (DType::F16, 0x3c00, 0x7e00),
            (DType::BF16, 0x3f80, 0x7fc0),
        ] {
            for (operation, identity) in [(Reduction::Sum, 0), (Reduction::Product, one), (Reduction::Mean, nan)] {
                let GlobalPlan::Relabel(layout) = global_plan(&scalar, dtype, operation).unwrap() else {
                    panic!()
                };
                assert_eq!(layout.shape(), &Shape::new([1]));
                assert_eq!((layout.offset(), layout.storage_elements(), layout.len()), (3, 7, 1));
                assert!(plan(&scalar, dtype, 0, operation).is_err());
                let GlobalPlan::Fill(bits) = global_plan(&empty, dtype, operation).unwrap() else {
                    panic!()
                };
                assert_eq!(bits, identity);
            }
        }
    }

    #[test]
    fn globals_stably_sort_original_lengths_and_keep_each_axis_operation() {
        let original = Layout::contiguous(Shape::new([3, 2, 2, 1])).unwrap();
        for dtype in DTYPES {
            for operation in OPERATIONS {
                let GlobalPlan::Axes(axes) = global_plan(&original, dtype, operation).unwrap() else {
                    panic!()
                };
                assert_eq!(axes, [3, 1, 2, 0]);
                let mut layout = original.clone();
                let mut lengths = Vec::new();
                for axis in axes {
                    let p = plan(&layout, dtype, axis, operation).unwrap();
                    assert_eq!(p.parameters[2] & 0xff, operation as usize);
                    lengths.push(p.parameters[6]);
                    layout = p.layout;
                }
                assert_eq!(lengths, [1, 2, 2, 3]);
                assert_eq!(layout.reshape(Shape::new([1])).unwrap().len(), 1);
            }
        }
    }

    #[test]
    fn mean_stage_order_preserves_the_half_storage_rounding_witness() {
        let layout = Layout::contiguous(Shape::new([2, 2])).unwrap();
        let GlobalPlan::Axes(axes) = global_plan(&layout, DType::F16, Reduction::Mean).unwrap() else {
            panic!()
        };
        let fold = |axes: &[usize]| {
            let mut values = vec![2048.0f32, 1.0, -2048.0, 0.0];
            let mut layout = layout.clone();
            for &axis in axes {
                let p = plan(&layout, DType::F16, axis, Reduction::Mean).unwrap();
                let mut next = Vec::new();
                for output in 0..p.layout.len() {
                    let column = output % p.layout.shape()[1];
                    let row = output / p.layout.shape()[1];
                    let mut sum = 0.0f32;
                    for coordinate in 0..layout.shape()[axis] {
                        let index = if axis == 0 {
                            coordinate * layout.shape()[1] + column
                        } else {
                            row * layout.shape()[1] + coordinate
                        };
                        sum += values[index];
                    }
                    next.push(f16::from_f32(sum / layout.shape()[axis] as f32).to_f32());
                }
                values = next;
                layout = p.layout;
            }
            f16::from_f32(values[0]).to_bits()
        };
        assert_eq!(fold(&axes), 0x3400);
        assert_eq!(fold(&[1, 0]), 0);
    }
}
