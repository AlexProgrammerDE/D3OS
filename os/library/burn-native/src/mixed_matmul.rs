//! Packed-input products accumulate privately in F32, then round the whole result.
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{DType, Shape};
use d3os_compute::{Buffer, Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::mixed_matmul::{self, Kind};

pub(crate) fn supported(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16 | DType::Flex32)
}

fn kind(dtype: DType) -> Result<Kind> {
    match dtype {
        DType::F16 => Ok(Kind::F16),
        DType::BF16 => Ok(Kind::BF16),
        DType::Flex32 => Ok(Kind::Flex32),
        _ => Err(Error::InvalidParameter),
    }
}

fn count(dimensions: &[usize]) -> Result<usize> {
    dimensions
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d).filter(|&n| n <= i32::MAX as usize).ok_or(Error::SizeOverflow))
}

fn matrix_contiguous(layout: &Layout) -> bool {
    let rank = layout.shape().num_dims();
    let rows = layout.shape()[rank - 2];
    let cols = layout.shape()[rank - 1];
    (rows == 1 || layout.strides()[rank - 2] == cols as i64) && (cols == 1 || layout.strides()[rank - 1] == 1)
}

fn check_storage(layout: &Layout, dtype: DType, words: usize) -> Result<()> {
    // Revalidate the complete signed view and its physical backing before a
    // materialization, allocation, recording scope, or empty shortcut.
    Layout::from_parts(
        layout.shape().clone(),
        layout.strides().to_vec(),
        layout.offset() as i64,
        layout.storage_elements(),
    )?;
    let required = crate::storage::width(dtype)?.words(layout.storage_elements() as u32) as usize;
    if words < required.max(1) {
        return Err(Error::ShapeMismatch);
    }
    words.checked_mul(4).ok_or(Error::SizeOverflow)?;
    Ok(())
}

pub(crate) fn validate_tensor(tensor: &NativeTensor) -> Result<()> {
    check_storage(&tensor.layout, tensor.dtype, tensor.buffer.len())
}

struct Operand {
    layout: Layout,
    materialize: bool,
}

impl Operand {
    fn new(source: &Layout, shape: Shape) -> Result<Self> {
        let materialize = !matrix_contiguous(source);
        let layout = if materialize {
            Layout::contiguous(source.shape().clone())?
        } else {
            source.clone()
        };
        Ok(Self {
            layout: layout.expand(shape)?,
            materialize,
        })
    }

    fn pointer(&self, batch: usize, elements: usize, kind: Kind) -> Result<(usize, usize)> {
        let base = self.layout.element_offset(batch.checked_mul(elements).ok_or(Error::SizeOverflow)?)?;
        let (word, slot) = if kind == Kind::Flex32 { (base, 0) } else { (base / 2, base % 2) };
        let end = base.checked_add(elements).ok_or(Error::SizeOverflow)?;
        if end > self.layout.storage_elements() {
            return Err(Error::ShapeMismatch);
        }
        Ok((word, slot))
    }

    fn slots(&self, kind: Kind) -> [usize; 2] {
        let slot = if kind == Kind::Flex32 { 0 } else { self.layout.offset() % 2 };
        let rank = self.layout.shape().num_dims();
        let changes = kind != Kind::Flex32
            && self.layout.shape()[..rank - 2]
                .iter()
                .zip(self.layout.strides())
                .any(|(&n, &stride)| n > 1 && stride % 2 != 0);
        [slot, if changes { 1 - slot } else { slot }]
    }
}

struct Product {
    kind: Kind,
    layout: Layout,
    left: Operand,
    right: Operand,
    rows: usize,
    inner: usize,
    cols: usize,
    batches: usize,
}

impl Product {
    fn new(left: &Layout, left_dtype: DType, left_words: usize, right: &Layout, right_dtype: DType, right_words: usize) -> Result<Self> {
        let kind = kind(left_dtype)?;
        if left_dtype != right_dtype {
            return Err(Error::InvalidParameter);
        }
        let shape = crate::layout::matmul_shape(left.shape(), right.shape())?;
        check_storage(left, left_dtype, left_words)?;
        check_storage(right, right_dtype, right_words)?;
        let rank = shape.num_dims();
        let rows = shape[rank - 2];
        let cols = shape[rank - 1];
        let inner = left.shape()[left.shape().num_dims() - 1];
        // Matrix sizes and batch counts are checked independently even when a
        // zero dimension makes the concatenated logical output empty.
        count(&[rows, inner])?;
        count(&[inner, cols])?;
        count(&[rows, cols])?;
        let batches = count(&shape[..rank - 2])?;
        let layout = Layout::contiguous(shape.clone())?;
        layout.len().checked_mul(4).ok_or(Error::SizeOverflow)?;
        let mut left_shape = shape.to_vec();
        left_shape[rank - 1] = inner;
        let mut right_shape = shape.to_vec();
        right_shape[rank - 2] = inner;
        let left = Operand::new(left, Shape::from(left_shape))?;
        let right = Operand::new(right, Shape::from(right_shape))?;
        if rows != 0 && cols != 0 {
            for lhs_slot in left.slots(kind) {
                for rhs_slot in right.slots(kind) {
                    let parameters = [
                        rows as u32,
                        inner as u32,
                        cols as u32,
                        if inner == 0 { 0 } else { lhs_slot as u32 },
                        if inner == 0 { 0 } else { rhs_slot as u32 },
                    ];
                    mixed_matmul::validate(kind, &parameters).map_err(|_| Error::SizeOverflow)?;
                    mixed_matmul::work_items(kind, &parameters).map_err(|_| Error::SizeOverflow)?;
                }
            }
        }
        Ok(Self {
            kind,
            layout,
            left,
            right,
            rows,
            inner,
            cols,
            batches,
        })
    }
}

pub(crate) fn validate_product(left: &Layout, dtype: DType, left_words: usize, right: &Layout, right_dtype: DType, right_words: usize) -> Result<()> {
    Product::new(left, dtype, left_words, right, right_dtype, right_words).map(|_| ())
}

impl NativeTensor {
    pub(crate) fn mixed_matmul(&self, rhs: &Self) -> Result<Self> {
        let product = Product::new(&self.layout, self.dtype, self.buffer.len(), &rhs.layout, rhs.dtype, rhs.buffer.len())?;
        if product.layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout: product.layout,
                dtype: self.dtype,
            });
        }
        Nvidia.record_commands(|| {
            if product.inner == 0 {
                // The allocator initializes positive-zero bits in the actual
                // destination dtype. No input read, MMA, or cast is needed.
                return Self::empty(product.layout.shape().clone(), self.dtype);
            }
            let left = if product.left.materialize { self.copy()? } else { self.clone() };
            let right = if product.right.materialize { rhs.copy()? } else { rhs.clone() };
            let buffer = Nvidia.allocate_words(product.layout.len())?;
            for batch in 0..product.batches {
                let (left_word, left_slot) = product.left.pointer(batch, product.rows * product.inner, product.kind)?;
                let (right_word, right_slot) = product.right.pointer(batch, product.inner * product.cols, product.kind)?;
                Nvidia.dispatch_words(
                    product.kind.kernel(),
                    &[
                        (&left.buffer, left_word),
                        (&right.buffer, right_word),
                        (&buffer, batch * product.rows * product.cols),
                    ],
                    &[product.rows, product.inner, product.cols, left_slot, right_slot],
                )?;
            }
            let scratch = Self {
                buffer: Arc::new(buffer),
                layout: product.layout,
                dtype: DType::F32,
            };
            // One writer owns every packed word, including odd batch borders.
            // Flex32's cast is only an alias of this private F32 allocation.
            scratch.cast(self.dtype)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn plan(left: &Layout, right: &Layout, dtype: DType) -> Result<Product> {
        let words = |layout: &Layout| (crate::storage::width(dtype).unwrap().words(layout.storage_elements() as u32) as usize).max(1);
        Product::new(left, dtype, words(left), right, dtype, words(right))
    }

    #[test]
    fn rank_eight_signed_batches_retain_storage_and_half_slots() {
        let left = Layout::from_parts(Shape::new([2, 1, 1, 1, 1, 1, 1, 3]), vec![-3, 0, 0, 0, 0, 0, 3, 1], 3, 6).unwrap();
        let right = Layout::contiguous(Shape::new([1, 1, 1, 1, 1, 1, 3, 1])).unwrap();
        for dtype in [DType::F16, DType::BF16, DType::Flex32] {
            let product = plan(&left, &right, dtype).unwrap();
            assert!(!product.left.materialize);
            assert!(!product.right.materialize);
            assert_eq!(product.batches, 2);
            assert_eq!(product.left.layout.strides()[0], -3);
            assert_eq!(product.right.layout.strides()[0], 0);
            let expected = if dtype == DType::Flex32 { [(3, 0), (0, 0)] } else { [(1, 1), (0, 0)] };
            for (batch, expected) in expected.into_iter().enumerate() {
                assert_eq!(product.left.pointer(batch, 3, product.kind).unwrap(), expected);
                assert_eq!(product.right.pointer(batch, 3, product.kind).unwrap(), (0, 0));
            }
        }
    }

    #[test]
    fn odd_dense_batches_and_matrix_materialization_precede_broadcast() {
        let left = Layout::contiguous(Shape::new([2, 1, 3])).unwrap();
        let right = Layout::contiguous(Shape::new([3, 1])).unwrap();
        let product = plan(&left, &right, DType::F16).unwrap();
        assert_eq!(product.left.pointer(0, 3, product.kind).unwrap(), (0, 0));
        assert_eq!(product.left.pointer(1, 3, product.kind).unwrap(), (1, 1));
        let left = Layout::contiguous(Shape::new([1, 3, 2])).unwrap().swap_dims(1, 2).unwrap();
        let right = Layout::contiguous(Shape::new([7, 3, 5])).unwrap();
        let product = plan(&left, &right, DType::BF16).unwrap();
        assert!(product.left.materialize);
        assert_eq!(product.left.layout.storage_elements(), 6);
        assert_eq!(product.left.layout.strides()[0], 0);
        assert_eq!(product.left.pointer(6, 6, product.kind).unwrap(), (0, 0));
    }

    #[test]
    fn empty_outputs_validate_shapes_types_spans_and_matrix_sizes_first() {
        let left = Layout::contiguous(Shape::new([0, 2, 3])).unwrap();
        let right = Layout::contiguous(Shape::new([1, 3, 4])).unwrap();
        assert!(plan(&left, &right, DType::F16).unwrap().layout.is_empty());
        let incompatible = Layout::contiguous(Shape::new([2, 3, 4])).unwrap();
        assert!(matches!(plan(&left, &incompatible, DType::F16), Err(Error::ShapeMismatch)));
        let bad_k = Layout::contiguous(Shape::new([1, 5, 0])).unwrap();
        assert!(matches!(plan(&left, &bad_k, DType::F16), Err(Error::ShapeMismatch)));
        assert!(matches!(
            Product::new(&left, DType::F16, 1, &right, DType::BF16, 6),
            Err(Error::InvalidParameter)
        ));
        assert!(matches!(Product::new(&left, DType::F16, 1, &right, DType::F16, 5), Err(Error::ShapeMismatch)));
        assert!(matches!(Product::new(&left, DType::F16, 0, &right, DType::F16, 6), Err(Error::ShapeMismatch)));
        let huge = Layout::contiguous(Shape::new([0, 2, i32::MAX as usize])).unwrap();
        let rhs = Layout::contiguous(Shape::new([1, i32::MAX as usize, 0])).unwrap();
        assert!(matches!(plan(&huge, &rhs, DType::F16), Err(Error::SizeOverflow)));
        for dims in [vec![], vec![3]] {
            let left = Layout::contiguous(Shape::from(dims)).unwrap();
            assert!(matches!(plan(&left, &right, DType::F16), Err(Error::ShapeMismatch)));
        }
        assert_eq!(kind(DType::F64), Err(Error::InvalidParameter));
    }

    #[test]
    fn zero_k_retains_nonempty_typed_shape_and_huge_live_counts_remain_valid() {
        let left = Layout::contiguous(Shape::new([2, 3, 0])).unwrap();
        let right = Layout::contiguous(Shape::new([1, 0, 5])).unwrap();
        let product = plan(&left, &right, DType::BF16).unwrap();
        assert_eq!((product.inner, product.batches, product.layout.len()), (0, 2, 30));
        let left = Layout::contiguous(Shape::new([1, i32::MAX as usize])).unwrap();
        let right = Layout::contiguous(Shape::new([i32::MAX as usize, 1])).unwrap();
        assert!(plan(&left, &right, DType::F16).is_ok());
    }

    #[test]
    fn planner_pointers_match_registered_work_extents_and_private_writer_aliases() {
        let left = Layout::from_parts(Shape::new([3, 5]), vec![5, 1], 1, 16).unwrap();
        let right = Layout::from_parts(Shape::new([5, 7]), vec![7, 1], 1, 36).unwrap();
        for dtype in [DType::F16, DType::BF16, DType::Flex32] {
            let product = plan(&left, &right, dtype).unwrap();
            let (left_word, left_slot) = product.left.pointer(0, 15, product.kind).unwrap();
            let (right_word, right_slot) = product.right.pointer(0, 35, product.kind).unwrap();
            let parameters = [3, 5, 7, left_slot as u32, right_slot as u32];
            let work = product.kind.kernel().work(&parameters).unwrap();
            let extents = mixed_matmul::validate(product.kind, &parameters).unwrap();
            assert_eq!(work.extents[..3], extents);
            assert_eq!(mixed_matmul::work_items(product.kind, &parameters).unwrap(), 128);
            assert_eq!(work.output, 2);
            assert!(!work.in_place);
            let lhs = 0x10000 + left_word as u64 * 4;
            let rhs = 0x20000 + right_word as u64 * 4;
            assert!(work.validate_aliases(&[lhs, rhs, 0x30000]).is_ok());
            assert!(work.validate_aliases(&[lhs, lhs, 0x30000]).is_ok());
            assert!(work.validate_aliases(&[lhs, rhs, lhs]).is_err());
            assert!(work.validate_aliases(&[lhs, rhs, rhs]).is_err());
        }
    }
}
