//! Typed indexed folds use resident indices and a private initialized destination.
use crate::{
    NativeTensor,
    layout::{Layout, nd_shape},
    storage::format,
};
use burn_backend::{DType, Shape, tensor::IndexingUpdateOp};
use d3os_compute::{Buffer, Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{
    Kernel,
    indexed_accumulation::{self as abi, Operation},
};

#[derive(Clone, Copy)]
pub(crate) enum AxisRoute {
    Scatter,
    Select,
}

pub(crate) fn uses_typed_accumulation(dtype: DType, update: IndexingUpdateOp) -> bool {
    matches!(
        dtype,
        DType::F64 | DType::Flex32 | DType::F16 | DType::BF16 | DType::I64 | DType::I16 | DType::I8 | DType::U64 | DType::U32 | DType::U16 | DType::U8
    ) && matches!(
        update,
        IndexingUpdateOp::Add | IndexingUpdateOp::Mul | IndexingUpdateOp::Min | IndexingUpdateOp::Max
    )
}

fn operation(update: IndexingUpdateOp) -> Result<Operation> {
    match update {
        IndexingUpdateOp::Add => Ok(Operation::Add),
        IndexingUpdateOp::Mul => Ok(Operation::Mul),
        IndexingUpdateOp::Min => Ok(Operation::Min),
        IndexingUpdateOp::Max => Ok(Operation::Max),
        _ => Err(Error::InvalidParameter),
    }
}

#[derive(Clone, Copy)]
struct Metadata<'a> {
    layout: &'a Layout,
    dtype: DType,
    words: usize,
}

impl<'a> From<&'a NativeTensor> for Metadata<'a> {
    fn from(tensor: &'a NativeTensor) -> Self {
        Self {
            layout: &tensor.layout,
            dtype: tensor.dtype,
            words: tensor.buffer.len(),
        }
    }
}

impl<'a> Metadata<'a> {
    fn for_layout(layout: &'a Layout, dtype: DType) -> Result<Self> {
        Ok(Self {
            layout,
            dtype,
            words: format(dtype)?.width().words(layout.storage_elements() as u32).max(1) as usize,
        })
    }

    fn validate(self) -> Result<()> {
        // Revalidate signed spans and physical storage before any resident work,
        // including for empty views and views that will later be snapshotted.
        Layout::from_parts(
            self.layout.shape().clone(),
            self.layout.strides().to_vec(),
            self.layout.offset() as i64,
            self.layout.storage_elements(),
        )?;
        let words = format(self.dtype)?.width().words(self.layout.storage_elements() as u32).max(1);
        if (self.words as u64) < words {
            return Err(Error::ShapeMismatch);
        }
        self.words.checked_mul(4).ok_or(Error::SizeOverflow)?;
        Ok(())
    }
}

fn validate_inputs(base: Metadata<'_>, indices: Metadata<'_>, values: Metadata<'_>, update: IndexingUpdateOp) -> Result<()> {
    if !uses_typed_accumulation(base.dtype, update)
        || values.dtype != base.dtype
        || !matches!(
            indices.dtype,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::U8 | DType::U16 | DType::U32 | DType::U64
        )
    {
        return Err(Error::InvalidParameter);
    }
    base.validate()?;
    indices.validate()?;
    values.validate()
}

struct AxisPlan {
    parameters: Option<[usize; abi::PARAMETERS]>,
    snapshot_values: bool,
}

impl AxisPlan {
    fn new(base: Metadata<'_>, axis: usize, indices: Metadata<'_>, values: Metadata<'_>, update: IndexingUpdateOp, route: AxisRoute) -> Result<Self> {
        validate_inputs(base, indices, values, update)?;
        if axis >= base.layout.shape().num_dims() {
            return Err(Error::InvalidParameter);
        }
        // checked_indices retains I32 views; every other integer source is
        // converted to dense offset-zero I32 storage, entirely on the device.
        let normalized = if indices.dtype == DType::I32 {
            indices.layout.clone()
        } else {
            Layout::contiguous(indices.layout.shape().clone())?
        };
        let selected = match route {
            AxisRoute::Scatter => {
                base.layout.gather_shape(&normalized, axis)?;
                normalized
            }
            AxisRoute::Select => {
                if normalized.shape().num_dims() != 1 {
                    return Err(Error::InvalidParameter);
                }
                let mut shape = base.layout.shape().to_vec();
                shape[axis] = normalized.len();
                let mut strides = alloc::vec![0; shape.len()];
                strides[axis] = normalized.strides()[0];
                Layout::from_parts(Shape::from(shape), strides, normalized.offset() as i64, normalized.storage_elements())?
            }
        };
        if values.layout.shape() != selected.shape() {
            return Err(Error::ShapeMismatch);
        }
        let snapshot_values = !values.layout.is_contiguous() || values.layout.offset() != 0;
        if selected.is_empty() {
            return Ok(Self {
                parameters: None,
                snapshot_values,
            });
        }
        if base.layout.is_empty() {
            return Err(Error::InvalidParameter);
        }
        let destination = Layout::contiguous(base.layout.shape().clone())?;
        let mut parameters = [0u32; abi::PARAMETERS];
        parameters[..8].copy_from_slice(&[
            abi::control(selected.shape().num_dims() as u32, format(base.dtype)?, operation(update)?).map_err(|_| Error::InvalidParameter)?,
            axis as u32,
            selected.len() as u32,
            destination.len() as u32,
            0,
            selected.storage_elements() as u32,
            selected.offset() as u32,
            destination.shape()[axis] as u32,
        ]);
        parameters[8..16].fill(1);
        for (dim, (&length, &stride)) in selected.shape().iter().zip(selected.strides()).enumerate() {
            parameters[8 + dim] = length as u32;
            parameters[16 + dim] = destination.strides()[dim] as u32;
            parameters[24 + dim] = stride as i32 as u32;
        }
        abi::geometry(&parameters).map_err(|_| Error::InvalidParameter)?;
        Ok(Self {
            parameters: Some(parameters.map(|word| word as usize)),
            snapshot_values,
        })
    }
}

pub(crate) fn validate_axis_update(
    base: &NativeTensor, axis: usize, indices: &NativeTensor, values: &NativeTensor, update: IndexingUpdateOp, route: AxisRoute,
) -> Result<()> {
    AxisPlan::new(base.into(), axis, indices.into(), values.into(), update, route).map(|_| ())
}

fn product(dimensions: &[usize]) -> Result<usize> {
    if dimensions.contains(&0) {
        return Ok(0);
    }
    dimensions.iter().try_fold(1usize, |n, &dim| {
        n.checked_mul(dim).filter(|&n| n <= i32::MAX as usize).ok_or(Error::SizeOverflow)
    })
}

fn validate_nd(base: Metadata<'_>, indices: Metadata<'_>, values: Metadata<'_>, update: IndexingUpdateOp) -> Result<()> {
    validate_inputs(base, indices, values, update)?;
    let shape = nd_shape(base.layout.shape(), indices.layout.shape())?;
    if values.layout.shape() != &shape {
        return Err(Error::ShapeMismatch);
    }
    let width = *indices.layout.shape().last().ok_or(Error::InvalidShape)?;
    let rows = product(&base.layout.shape()[..width])?;
    let slices = product(&base.layout.shape()[width..])?;
    let tuples = indices.layout.len() / width;
    let data = Layout::contiguous(Shape::new([rows, slices]))?;
    let updates = Layout::contiguous(Shape::new([tuples, slices]))?;
    let linear = Layout::contiguous(Shape::new([tuples]))?;
    // Validate the entire common select lowering before per-coordinate kernels.
    // The resident ND helper produces dense I32 row indices after its masks.
    AxisPlan::new(
        Metadata::for_layout(&data, base.dtype)?,
        0,
        Metadata::for_layout(&linear, DType::I32)?,
        Metadata::for_layout(&updates, values.dtype)?,
        update,
        AxisRoute::Select,
    )?;
    Ok(())
}

pub(crate) fn validate_nd_update(base: &NativeTensor, indices: &NativeTensor, values: &NativeTensor, update: IndexingUpdateOp) -> Result<()> {
    validate_nd(base.into(), indices.into(), values.into(), update)
}

impl NativeTensor {
    pub(crate) fn indexed_accumulation(&self, axis: usize, indices: &Self, values: &Self, update: IndexingUpdateOp) -> Result<Self> {
        let plan = AxisPlan::new(self.into(), axis, indices.into(), values.into(), update, AxisRoute::Scatter)?;
        let Some(parameters) = plan.parameters else {
            return Ok(self.clone());
        };
        Nvidia.record_commands(|| {
            let indices = indices.checked_indices()?;
            let values = if plan.snapshot_values { values.copy()? } else { values.clone() };
            let output = self.copy()?;
            let status = Nvidia.produce_words(1, Kernel::Fill, &[], &[1, 0])?;
            Nvidia.dispatch_words(
                Kernel::IndexedAccumulation,
                &[(&values.buffer, 0), (&indices.buffer, 0), (&status, 0), (&output.buffer, 0)],
                &parameters,
            )?;
            let mut invalid = [0u32];
            Nvidia.read_words_into(&status, 0, &mut invalid)?;
            if invalid[0] != 0 {
                return Err(Error::InvalidParameter);
            }
            Ok(output)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use burn_backend::{BoolStore, quantization::QuantScheme};

    const FORMATS: [DType; 11] = [
        DType::F64,
        DType::Flex32,
        DType::F16,
        DType::BF16,
        DType::I64,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
    ];
    const INDICES: [DType; 8] = [DType::I8, DType::I16, DType::I32, DType::I64, DType::U8, DType::U16, DType::U32, DType::U64];
    const UPDATES: [IndexingUpdateOp; 4] = [IndexingUpdateOp::Add, IndexingUpdateOp::Mul, IndexingUpdateOp::Min, IndexingUpdateOp::Max];

    fn metadata(layout: &Layout, dtype: DType) -> Metadata<'_> {
        Metadata::for_layout(layout, dtype).unwrap()
    }

    #[test]
    fn typed_packets_match_real_sdk_geometry_for_all_formats_operations_and_index_types() {
        let base = Layout::contiguous(Shape::new([2, 5])).unwrap();
        let indices = Layout::from_parts(Shape::new([2, 3]), vec![4, -1], 2, 7).unwrap();
        let values = Layout::contiguous(Shape::new([2, 3])).unwrap();
        for dtype in FORMATS {
            let width = format(dtype).unwrap().width();
            for update in UPDATES {
                for index_dtype in INDICES {
                    let plan = AxisPlan::new(
                        metadata(&base, dtype),
                        1,
                        metadata(&indices, index_dtype),
                        metadata(&values, dtype),
                        update,
                        AxisRoute::Scatter,
                    )
                    .unwrap();
                    let p = plan.parameters.unwrap().map(|v| v as u32);
                    assert_eq!(p[0], abi::control(2, format(dtype).unwrap(), operation(update).unwrap()).unwrap());
                    assert_eq!(&p[1..5], &[1, 6, 10, 0]);
                    assert_eq!(&p[8..16], &[2, 3, 1, 1, 1, 1, 1, 1]);
                    assert_eq!(&p[16..24], &[5, 1, 0, 0, 0, 0, 0, 0]);
                    if index_dtype == DType::I32 {
                        assert_eq!(&p[5..7], &[7, 2]);
                        assert_eq!(&p[24..26], &[4, (-1i32) as u32]);
                    } else {
                        assert_eq!(&p[5..7], &[6, 0]);
                        assert_eq!(&p[24..26], &[3, 1]);
                    }
                    assert_eq!(abi::validate(&p).unwrap(), [width.words(6), u64::from(p[5]), 1, width.words(10)]);
                    assert!(!plan.snapshot_values);
                }
            }
        }
    }

    #[test]
    fn select_preserves_i32_signed_views_and_broadcasts_only_the_selected_axis() {
        let base = Layout::contiguous(Shape::new([1, 1, 1, 1, 1, 4, 2])).unwrap();
        let indices = Layout::from_parts(Shape::new([3]), vec![-2], 5, 7).unwrap();
        let values = Layout::contiguous(Shape::new([1, 1, 1, 1, 1, 3, 2])).unwrap();
        for index_dtype in INDICES {
            let plan = AxisPlan::new(
                metadata(&base, DType::BF16),
                5,
                metadata(&indices, index_dtype),
                metadata(&values, DType::BF16),
                IndexingUpdateOp::Add,
                AxisRoute::Select,
            )
            .unwrap();
            let p = plan.parameters.unwrap().map(|v| v as u32);
            assert_eq!(p[2], 6);
            assert_eq!(&p[24..29], &[0; 5]);
            assert_eq!(&p[30..32], &[0; 2]);
            assert_eq!(p[29], if index_dtype == DType::I32 { (-2i32) as u32 } else { 1 });
            assert_eq!(&p[5..7], if index_dtype == DType::I32 { &[7, 5] } else { &[3, 0] });
            assert!(abi::geometry(&p).is_ok());
        }
    }

    #[test]
    fn snapshots_handle_offsets_reversed_values_and_nonprivate_source_views() {
        let base = Layout::from_parts(Shape::new([2, 5]), vec![-5, 1], 6, 17).unwrap();
        let indices = Layout::contiguous(Shape::new([2, 3])).unwrap();
        let offset = Layout::from_parts(Shape::new([2, 3]), vec![3, 1], 1, 7).unwrap();
        let reversed = offset.flip(&[1]).unwrap();
        for dtype in FORMATS {
            for values in [&offset, &reversed] {
                let plan = AxisPlan::new(
                    metadata(&base, dtype),
                    1,
                    metadata(&indices, DType::I32),
                    metadata(values, dtype),
                    IndexingUpdateOp::Mul,
                    AxisRoute::Scatter,
                )
                .unwrap();
                assert!(plan.snapshot_values);
                let p = plan.parameters.unwrap();
                assert_eq!(&p[3..5], &[10, 0]);
                assert_eq!(&p[16..18], &[5, 1]);
            }
        }
        assert!(offset.is_contiguous());
        assert!(!reversed.is_contiguous());
    }

    #[test]
    fn empty_plans_validate_types_shapes_ranks_axis_and_backing_capacity_first() {
        let base = Layout::contiguous(Shape::new([2, 0])).unwrap();
        let updates = Layout::contiguous(Shape::new([2, 0])).unwrap();
        let indices = Layout::contiguous(Shape::new([0])).unwrap();
        let b = metadata(&base, DType::F16);
        let i = metadata(&indices, DType::U64);
        let v = metadata(&updates, DType::F16);
        assert!(
            AxisPlan::new(b, 1, i, v, IndexingUpdateOp::Add, AxisRoute::Select)
                .unwrap()
                .parameters
                .is_none()
        );
        for dtype in [DType::F32, DType::I32, DType::Bool(BoolStore::U8), DType::QFloat(QuantScheme::default())] {
            assert!(!uses_typed_accumulation(dtype, IndexingUpdateOp::Add));
            assert!(matches!(
                AxisPlan::new(Metadata { dtype, ..b }, 1, i, v, IndexingUpdateOp::Add, AxisRoute::Select),
                Err(Error::InvalidParameter)
            ));
        }
        assert!(matches!(
            AxisPlan::new(b, 1, i, v, IndexingUpdateOp::Assign, AxisRoute::Select),
            Err(Error::InvalidParameter)
        ));
        assert!(matches!(
            AxisPlan::new(b, 2, i, v, IndexingUpdateOp::Add, AxisRoute::Select),
            Err(Error::InvalidParameter)
        ));
        assert!(matches!(
            AxisPlan::new(b, 1, Metadata { dtype: DType::F32, ..i }, v, IndexingUpdateOp::Add, AxisRoute::Select),
            Err(Error::InvalidParameter)
        ));
        assert!(matches!(
            AxisPlan::new(b, 1, i, Metadata { dtype: DType::BF16, ..v }, IndexingUpdateOp::Add, AxisRoute::Select),
            Err(Error::InvalidParameter)
        ));
        assert!(matches!(
            AxisPlan::new(b, 1, i, Metadata { words: 0, ..v }, IndexingUpdateOp::Add, AxisRoute::Select),
            Err(Error::ShapeMismatch)
        ));
        let rank_two_indices = Metadata { dtype: DType::I32, ..v };
        assert!(matches!(
            AxisPlan::new(b, 1, rank_two_indices, v, IndexingUpdateOp::Add, AxisRoute::Select),
            Err(Error::InvalidParameter)
        ));
        let wrong_values = Layout::contiguous(Shape::new([3, 0])).unwrap();
        assert!(matches!(
            AxisPlan::new(b, 1, i, metadata(&wrong_values, DType::F16), IndexingUpdateOp::Add, AxisRoute::Select),
            Err(Error::ShapeMismatch)
        ));
        let live_indices = Layout::contiguous(Shape::new([1])).unwrap();
        let live_values = Layout::contiguous(Shape::new([2, 1])).unwrap();
        assert!(matches!(
            AxisPlan::new(
                b,
                1,
                metadata(&live_indices, DType::I64),
                metadata(&live_values, DType::F16),
                IndexingUpdateOp::Add,
                AxisRoute::Select
            ),
            Err(Error::InvalidParameter)
        ));
        let scalar = Layout::contiguous(Shape::new([])).unwrap();
        assert!(matches!(
            AxisPlan::new(metadata(&scalar, DType::F16), 0, i, v, IndexingUpdateOp::Add, AxisRoute::Scatter),
            Err(Error::InvalidParameter)
        ));
    }

    #[test]
    fn nd_preflight_checks_coordinate_shape_and_lowered_products_before_empty_returns() {
        let base = Layout::contiguous(Shape::new([3, 4, 2])).unwrap();
        let indices = Layout::contiguous(Shape::new([5, 2])).unwrap();
        let values = Layout::contiguous(Shape::new([5, 2])).unwrap();
        for dtype in FORMATS {
            for index_dtype in INDICES {
                assert!(
                    validate_nd(
                        metadata(&base, dtype),
                        metadata(&indices, index_dtype),
                        metadata(&values, dtype),
                        IndexingUpdateOp::Add
                    )
                    .is_ok()
                );
            }
        }
        let empty_base = Layout::contiguous(Shape::new([3, 4, 0])).unwrap();
        let empty_values = Layout::contiguous(Shape::new([5, 0])).unwrap();
        assert!(
            validate_nd(
                metadata(&empty_base, DType::F64),
                metadata(&indices, DType::U64),
                metadata(&empty_values, DType::F64),
                IndexingUpdateOp::Min
            )
            .is_ok()
        );
        for shape in [Shape::new([]), Shape::new([5, 0]), Shape::new([0, 2]), Shape::new([5, 4])] {
            let invalid = Layout::contiguous(shape).unwrap();
            assert!(matches!(
                validate_nd(
                    metadata(&empty_base, DType::F64),
                    metadata(&invalid, DType::I64),
                    metadata(&empty_values, DType::F64),
                    IndexingUpdateOp::Min
                ),
                Err(Error::InvalidShape)
            ));
        }
        let huge = Layout::contiguous(Shape::new([i32::MAX as usize, 2, 0])).unwrap();
        assert!(matches!(
            validate_nd(
                metadata(&huge, DType::F64),
                metadata(&indices, DType::U64),
                metadata(&empty_values, DType::F64),
                IndexingUpdateOp::Add
            ),
            Err(Error::SizeOverflow)
        ));
        let point = Layout::contiguous(Shape::new([3])).unwrap();
        let scalar = Layout::contiguous(Shape::new([])).unwrap();
        assert!(
            validate_nd(
                metadata(&base, DType::F64),
                metadata(&point, DType::I32),
                metadata(&scalar, DType::F64),
                IndexingUpdateOp::Max
            )
            .is_ok()
        );
    }

    #[test]
    fn wide_extents_keep_high_words_and_reject_short_backing_storage() {
        let n = i32::MAX as usize;
        let base = Layout::contiguous(Shape::new([n])).unwrap();
        let indices = Layout::from_parts(Shape::new([n]), vec![-1], n as i64 - 1, n).unwrap();
        let b = metadata(&base, DType::U64);
        let i = metadata(&indices, DType::I32);
        let plan = AxisPlan::new(b, 0, i, b, IndexingUpdateOp::Max, AxisRoute::Scatter).unwrap();
        let p = plan.parameters.unwrap().map(|v| v as u32);
        assert_eq!(abi::validate(&p).unwrap(), [0xffff_fffe, n as u64, 1, 0xffff_fffe]);
        assert!(matches!(
            AxisPlan::new(Metadata { words: b.words - 1, ..b }, 0, i, b, IndexingUpdateOp::Max, AxisRoute::Scatter),
            Err(Error::ShapeMismatch)
        ));
        assert!(matches!(
            AxisPlan::new(b, 0, i, Metadata { words: b.words - 1, ..b }, IndexingUpdateOp::Max, AxisRoute::Scatter),
            Err(Error::ShapeMismatch)
        ));
    }

    #[test]
    fn registered_work_retains_initialized_output_and_isolates_both_writers() {
        let base = Layout::contiguous(Shape::new([2, 5])).unwrap();
        let indices = Layout::contiguous(Shape::new([2, 3])).unwrap();
        for dtype in FORMATS {
            let plan = AxisPlan::new(
                metadata(&base, dtype),
                1,
                metadata(&indices, DType::I32),
                metadata(&indices, dtype),
                IndexingUpdateOp::Add,
                AxisRoute::Scatter,
            )
            .unwrap();
            let p = plan.parameters.unwrap().map(|v| v as u32);
            let work = Kernel::IndexedAccumulation.work(&p).unwrap();
            assert_eq!(&work.extents[..4], &abi::validate(&p).unwrap());
            assert_eq!((work.pointers, work.output, work.mutated), (4, 3, 1 << 2));
            assert!(!work.in_place);
            assert!(!Kernel::IndexedAccumulation.allocates_output());
            assert!(work.validate_aliases(&[0x10000, 0x20000, 0x30000, 0x40000]).is_ok());
            assert!(work.validate_aliases(&[0x10000, 0x10000, 0x30000, 0x40000]).is_ok());
            for addresses in [
                [0x10000, 0x20000, 0x10000, 0x40000],
                [0x10000, 0x20000, 0x20000, 0x40000],
                [0x10000, 0x20000, 0x30000, 0x10000],
                [0x10000, 0x20000, 0x30000, 0x20000],
                [0x10000, 0x20000, 0x30000, 0x30000],
                [0x10000, 0x20000, u64::MAX - 1, 0x40000],
            ] {
                assert!(work.validate_aliases(&addresses).is_err());
            }
        }
    }
}
