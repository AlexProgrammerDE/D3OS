use super::*;
use burn_backend::ops::*;
use burn_backend::tensor::{BoolTensor, Device, FloatTensor, IndexingUpdateOp, IntTensor};
use burn_backend::{BoolDType, Distribution, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData};
impl IntTensorOps<Self> for Native {
    fn int_empty(shape: Shape, _device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        BackendTensor::build("int_empty", shape.clone(), dtype.into(), || NativeTensor::full(shape, dtype.into(), 0))
    }
    async fn int_into_data(tensor: IntTensor<Self>) -> Result<TensorData, ExecutionError> {
        tensor.read()
    }
    fn int_from_data(data: TensorData, _device: &Device<Self>) -> IntTensor<Self> {
        BackendTensor::build("int_from_data", data.shape().clone(), data.dtype(), || {
            if data.dtype() != DType::I32 {
                return Err(Error::InvalidParameter);
            }
            NativeTensor::from_data(data)
        })
    }
    fn int_to_device(tensor: IntTensor<Self>, _device: &Device<Self>) -> IntTensor<Self> {
        tensor
    }
    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        reshape("int_reshape", &tensor, shape)
    }
    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        view("int_slice", &tensor, layout(&tensor).and_then(|l| l.slice(slices)))
    }
    fn int_slice_assign(tensor: IntTensor<Self>, slices: &[Slice], value: IntTensor<Self>) -> IntTensor<Self> {
        BackendTensor::apply("int_slice_assign", tensor.shape(), tensor.dtype(), &[&tensor, &value], |v| {
            v[0].slice_assign(slices, v[1])
        })
    }
    fn int_into_float(tensor: IntTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        cast("int_into_float", &tensor, out_dtype.into())
    }
    fn int_mask_where(tensor: IntTensor<Self>, mask: BoolTensor<Self>, value: IntTensor<Self>) -> IntTensor<Self> {
        BackendTensor::apply("int_mask_where", tensor.shape(), tensor.dtype(), &[&tensor, &mask, &value], |v| {
            v[0].mask_where(v[1], v[2])
        })
    }
    fn int_mask_fill(tensor: IntTensor<Self>, mask: BoolTensor<Self>, value: Scalar) -> IntTensor<Self> {
        BackendTensor::apply("int_mask_fill", tensor.shape(), tensor.dtype(), &[&tensor, &mask], |v| {
            v[0].mask_fill_i32(v[1], value.elem::<i32>())
        })
    }
    fn int_gather(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>) -> IntTensor<Self> {
        BackendTensor::apply("int_gather", indices.shape(), tensor.dtype(), &[&tensor, &indices], |v| v[0].gather(dim, v[1]))
    }
    fn int_scatter(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>, value: IntTensor<Self>, update: IndexingUpdateOp) -> IntTensor<Self> {
        if update != IndexingUpdateOp::Add {
            return BackendTensor::unsupported(
                "int_scatter",
                tensor.shape(),
                tensor.dtype(),
                &[&tensor, &indices, &value],
                "non-additive indexed updates",
            );
        }
        BackendTensor::apply("int_scatter", tensor.shape(), tensor.dtype(), &[&tensor, &indices, &value], |v| {
            v[0].scatter_add(dim, v[1], v[2])
        })
    }
    fn int_select(tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>) -> IntTensor<Self> {
        BackendTensor::apply(
            "int_select",
            {
                let mut s = tensor.shape();
                if let Some(d) = s.get_mut(dim) {
                    *d = indices.shape().num_elements();
                }
                s
            },
            tensor.dtype(),
            &[&tensor, &indices],
            |v| v[0].select(dim, v[1]),
        )
    }
    fn int_select_assign(tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>, value: IntTensor<Self>, update: IndexingUpdateOp) -> IntTensor<Self> {
        if update != IndexingUpdateOp::Add {
            return BackendTensor::unsupported(
                "int_select_assign",
                tensor.shape(),
                tensor.dtype(),
                &[&tensor, &indices, &value],
                "non-additive indexed updates",
            );
        }
        BackendTensor::apply("int_select_assign", tensor.shape(), tensor.dtype(), &[&tensor, &indices, &value], |v| {
            v[0].select_add(dim, v[1], v[2])
        })
    }
    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("int_equal", &lhs, &rhs, out_dtype.into(), |a, b| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.word_binary(b, WordOp::Equal)
        })
    }
    fn int_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("int_equal_elem", &lhs, out_dtype.into(), |a| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.word_scalar(WordOp::Equal, rhs.elem::<i32>() as u32)
        })
    }
    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("int_greater", &lhs, &rhs, out_dtype.into(), |a, b| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.word_binary(b, WordOp::Greater)
        })
    }
    fn int_greater_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("int_greater_elem", &lhs, out_dtype.into(), |a| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.word_scalar(WordOp::Greater, rhs.elem::<i32>() as u32)
        })
    }
    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("int_greater_equal", &lhs, &rhs, out_dtype.into(), |a, b| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.word_binary(b, WordOp::GreaterEqual)
        })
    }
    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("int_greater_equal_elem", &lhs, out_dtype.into(), |a| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.word_scalar(WordOp::GreaterEqual, rhs.elem::<i32>() as u32)
        })
    }
    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("int_lower", &lhs, &rhs, out_dtype.into(), |a, b| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.word_binary(b, WordOp::Less)
        })
    }
    fn int_lower_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("int_lower_elem", &lhs, out_dtype.into(), |a| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.word_scalar(WordOp::Less, rhs.elem::<i32>() as u32)
        })
    }
    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("int_lower_equal", &lhs, &rhs, out_dtype.into(), |a, b| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.word_binary(b, WordOp::LessEqual)
        })
    }
    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("int_lower_equal_elem", &lhs, out_dtype.into(), |a| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.word_scalar(WordOp::LessEqual, rhs.elem::<i32>() as u32)
        })
    }
    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary("int_add", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::Add))
    }
    fn int_add_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        one("int_add_scalar", &lhs, lhs.dtype(), |a| a.word_scalar(WordOp::Add, rhs.elem::<i32>() as u32))
    }
    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary("int_sub", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::Subtract))
    }
    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        one("int_sub_scalar", &lhs, lhs.dtype(), |a| {
            a.word_scalar(WordOp::Subtract, rhs.elem::<i32>() as u32)
        })
    }
    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary("int_mul", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::Multiply))
    }
    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        one("int_mul_scalar", &lhs, lhs.dtype(), |a| {
            a.word_scalar(WordOp::Multiply, rhs.elem::<i32>() as u32)
        })
    }
    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        BackendTensor::unsupported("int_div", lhs.shape(), DType::I32, &[&lhs, &rhs], "int_div")
    }
    fn int_div_scalar(lhs: IntTensor<Self>, _rhs: Scalar) -> IntTensor<Self> {
        BackendTensor::unsupported("int_div_scalar", lhs.shape(), DType::I32, &[&lhs], "int_div_scalar")
    }
    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        BackendTensor::unsupported("int_remainder", lhs.shape(), DType::I32, &[&lhs, &rhs], "int_remainder")
    }
    fn int_remainder_scalar(lhs: IntTensor<Self>, _rhs: Scalar) -> IntTensor<Self> {
        BackendTensor::unsupported("int_remainder_scalar", lhs.shape(), DType::I32, &[&lhs], "int_remainder_scalar")
    }
    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        BackendTensor::unsupported("int_matmul", lhs.shape(), DType::I32, &[&lhs, &rhs], "int_matmul")
    }
    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        BackendTensor::unsupported("int_sum", tensor.shape(), DType::I32, &[&tensor], "int_sum")
    }
    fn int_sum_dim(tensor: IntTensor<Self>, _dim: usize) -> IntTensor<Self> {
        BackendTensor::unsupported("int_sum_dim", tensor.shape(), DType::I32, &[&tensor], "int_sum_dim")
    }
    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        BackendTensor::unsupported("int_prod", tensor.shape(), DType::I32, &[&tensor], "int_prod")
    }
    fn int_prod_dim(tensor: IntTensor<Self>, _dim: usize) -> IntTensor<Self> {
        BackendTensor::unsupported("int_prod_dim", tensor.shape(), DType::I32, &[&tensor], "int_prod_dim")
    }
    fn int_mean_dim(tensor: IntTensor<Self>, _dim: usize) -> IntTensor<Self> {
        BackendTensor::unsupported("int_mean_dim", tensor.shape(), DType::I32, &[&tensor], "int_mean_dim")
    }
    fn int_cumsum(tensor: IntTensor<Self>, _dim: usize) -> IntTensor<Self> {
        BackendTensor::unsupported("int_cumsum", tensor.shape(), DType::I32, &[&tensor], "int_cumsum")
    }
    fn int_cumprod(tensor: IntTensor<Self>, _dim: usize) -> IntTensor<Self> {
        BackendTensor::unsupported("int_cumprod", tensor.shape(), DType::I32, &[&tensor], "int_cumprod")
    }
    fn int_cummin(tensor: IntTensor<Self>, _dim: usize) -> IntTensor<Self> {
        BackendTensor::unsupported("int_cummin", tensor.shape(), DType::I32, &[&tensor], "int_cummin")
    }
    fn int_cummax(tensor: IntTensor<Self>, _dim: usize) -> IntTensor<Self> {
        BackendTensor::unsupported("int_cummax", tensor.shape(), DType::I32, &[&tensor], "int_cummax")
    }
    fn int_argmax(tensor: IntTensor<Self>, _dim: usize) -> IntTensor<Self> {
        BackendTensor::unsupported("int_argmax", tensor.shape(), DType::I32, &[&tensor], "int_argmax")
    }
    fn int_argmin(tensor: IntTensor<Self>, _dim: usize) -> IntTensor<Self> {
        BackendTensor::unsupported("int_argmin", tensor.shape(), DType::I32, &[&tensor], "int_argmin")
    }
    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        one("int_abs", &tensor, tensor.dtype(), |t| t.word_unary(WordOp::Absolute))
    }
    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        view("int_swap_dims", &tensor, layout(&tensor).and_then(|l| l.swap_dims(dim1, dim2)))
    }
    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        view("int_permute", &tensor, layout(&tensor).and_then(|l| l.permute(axes)))
    }
    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        view("int_flip", &tensor, layout(&tensor).and_then(|l| l.flip(axes)))
    }
    fn int_random(shape: Shape, distribution: Distribution, _device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        BackendTensor::build("int_random", shape.clone(), dtype.into(), || {
            NativeTensor::random(shape, dtype.into(), distribution)
        })
    }
    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        view("int_expand", &tensor, layout(&tensor).and_then(|l| l.expand(shape)))
    }
    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary("bitwise_and", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::BitAnd))
    }
    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        one("bitwise_and_scalar", &lhs, lhs.dtype(), |a| {
            a.word_scalar(WordOp::BitAnd, rhs.elem::<i32>() as u32)
        })
    }
    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary("bitwise_or", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::BitOr))
    }
    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        one("bitwise_or_scalar", &lhs, lhs.dtype(), |a| {
            a.word_scalar(WordOp::BitOr, rhs.elem::<i32>() as u32)
        })
    }
    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary("bitwise_xor", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::BitXor))
    }
    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        one("bitwise_xor_scalar", &lhs, lhs.dtype(), |a| {
            a.word_scalar(WordOp::BitXor, rhs.elem::<i32>() as u32)
        })
    }
    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        one("bitwise_not", &tensor, tensor.dtype(), |t| t.word_unary(WordOp::BitNot))
    }
    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary("bitwise_left_shift", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::ShiftLeft))
    }
    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        one("bitwise_left_shift_scalar", &lhs, lhs.dtype(), |a| {
            a.word_scalar(WordOp::ShiftLeft, rhs.elem::<i32>() as u32)
        })
    }
    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary("bitwise_right_shift", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::ShiftRight))
    }
    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        one("bitwise_right_shift_scalar", &lhs, lhs.dtype(), |a| {
            a.word_scalar(WordOp::ShiftRight, rhs.elem::<i32>() as u32)
        })
    }
    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        cast("int_cast", &tensor, dtype.into())
    }
    fn int_unfold(tensor: IntTensor<Self>, dim: usize, size: usize, step: usize) -> IntTensor<Self> {
        view("int_unfold", &tensor, layout(&tensor).and_then(|l| l.unfold(dim, size, step)))
    }
    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        cat("int_cat", tensors, dim)
    }
    fn int_zeros(shape: Shape, _device: &NativeDevice, dtype: IntDType) -> BackendTensor {
        BackendTensor::build("int_zeros", shape.clone(), dtype.into(), || NativeTensor::full(shape, dtype.into(), 0))
    }
    fn int_ones(shape: Shape, _device: &NativeDevice, dtype: IntDType) -> BackendTensor {
        BackendTensor::build("int_ones", shape.clone(), dtype.into(), || NativeTensor::full(shape, dtype.into(), 1))
    }
    fn int_full(shape: Shape, fill_value: Scalar, _device: &NativeDevice, dtype: IntDType) -> BackendTensor {
        BackendTensor::build("int_full", shape.clone(), dtype.into(), || {
            NativeTensor::full(shape, dtype.into(), fill_value.elem::<i32>() as u32)
        })
    }
    fn int_scatter_nd(data: BackendTensor, indices: BackendTensor, values: BackendTensor, _update: IndexingUpdateOp) -> BackendTensor {
        BackendTensor::unsupported(
            "int_scatter_nd",
            data.shape(),
            data.dtype(),
            &[&data, &indices, &values],
            "multidimensional scatter",
        )
    }
    fn int_gather_nd(data: BackendTensor, indices: BackendTensor) -> BackendTensor {
        BackendTensor::unsupported("int_gather_nd", data.shape(), data.dtype(), &[&data, &indices], "multidimensional gather")
    }
    fn int_sort(tensor: BackendTensor, _dim: usize, _descending: bool) -> BackendTensor {
        BackendTensor::unsupported("int_sort", tensor.shape(), tensor.dtype(), &[&tensor], "GPU sorting")
    }
    fn int_sort_with_indices(tensor: BackendTensor, _dim: usize, _descending: bool) -> (BackendTensor, BackendTensor) {
        let output = BackendTensor::unsupported("int_sort_with_indices", tensor.shape(), tensor.dtype(), &[&tensor], "GPU sorting");
        let indices = BackendTensor::unsupported("int_sort_with_indices", tensor.shape(), tensor.dtype(), &[&output], "GPU sorting");
        (output, indices)
    }
    fn int_argsort(tensor: BackendTensor, _dim: usize, _descending: bool) -> BackendTensor {
        BackendTensor::unsupported("int_argsort", tensor.shape(), tensor.dtype(), &[&tensor], "GPU sorting")
    }
    fn int_arange_step(range: core::ops::Range<i64>, step: usize, _device: &NativeDevice, dtype: IntDType) -> BackendTensor {
        let count = if step == 0 {
            0
        } else {
            ((range.end as i128 - range.start as i128).max(0) as u128).div_ceil(step as u128) as usize
        };
        BackendTensor::unsupported("int_arange", Shape::new([count]), dtype.into(), &[], "GPU index generation")
    }
    async fn int_mask_select(tensor: BackendTensor, mask: BackendTensor) -> BackendTensor {
        BackendTensor::unsupported("int_mask_select", Shape::new([0]), tensor.dtype(), &[&tensor, &mask], "GPU compaction")
    }
}
