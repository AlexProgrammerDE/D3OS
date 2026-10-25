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
            if !matches!(
                data.dtype(),
                DType::I64 | DType::I32 | DType::I16 | DType::I8 | DType::U64 | DType::U32 | DType::U16 | DType::U8
            ) {
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
            v[0].mask_fill_scalar(v[1], value)
        })
    }
    fn int_gather(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>) -> IntTensor<Self> {
        BackendTensor::apply("int_gather", indices.shape(), tensor.dtype(), &[&tensor, &indices], |v| v[0].gather(dim, v[1]))
    }
    fn int_scatter(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>, value: IntTensor<Self>, update: IndexingUpdateOp) -> IntTensor<Self> {
        BackendTensor::apply("int_scatter", tensor.shape(), tensor.dtype(), &[&tensor, &indices, &value], |v| {
            v[0].scatter_update(dim, v[1], v[2], update)
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
        BackendTensor::apply("int_select_assign", tensor.shape(), tensor.dtype(), &[&tensor, &indices, &value], |v| {
            v[0].select_update(dim, v[1], v[2], update)
        })
    }
    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("int_equal", &lhs, &rhs, out_dtype.into(), |a, b| {
            a.compare(b, Predicate::Equal, out_dtype.into())
        })
    }
    fn int_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("int_equal_elem", &lhs, out_dtype.into(), |a| {
            a.compare_scalar(rhs, Predicate::Equal, out_dtype.into())
        })
    }
    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("int_greater", &lhs, &rhs, out_dtype.into(), |a, b| {
            a.compare(b, Predicate::Greater, out_dtype.into())
        })
    }
    fn int_greater_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("int_greater_elem", &lhs, out_dtype.into(), |a| {
            a.compare_scalar(rhs, Predicate::Greater, out_dtype.into())
        })
    }
    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("int_greater_equal", &lhs, &rhs, out_dtype.into(), |a, b| {
            a.compare(b, Predicate::GreaterEqual, out_dtype.into())
        })
    }
    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("int_greater_equal_elem", &lhs, out_dtype.into(), |a| {
            a.compare_scalar(rhs, Predicate::GreaterEqual, out_dtype.into())
        })
    }
    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("int_lower", &lhs, &rhs, out_dtype.into(), |a, b| {
            a.compare(b, Predicate::Less, out_dtype.into())
        })
    }
    fn int_lower_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("int_lower_elem", &lhs, out_dtype.into(), |a| {
            a.compare_scalar(rhs, Predicate::Less, out_dtype.into())
        })
    }
    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("int_lower_equal", &lhs, &rhs, out_dtype.into(), |a, b| {
            a.compare(b, Predicate::LessEqual, out_dtype.into())
        })
    }
    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("int_lower_equal_elem", &lhs, out_dtype.into(), |a| {
            a.compare_scalar(rhs, Predicate::LessEqual, out_dtype.into())
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
        binary("int_div", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::Divide))
    }
    fn int_div_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        one("int_div_scalar", &lhs, lhs.dtype(), |a| a.word_scalar(WordOp::Divide, rhs.elem::<i32>() as u32))
    }
    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary("int_remainder", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::Remainder))
    }
    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        one("int_remainder_scalar", &lhs, lhs.dtype(), |a| {
            a.word_scalar(WordOp::Remainder, rhs.elem::<i32>() as u32)
        })
    }
    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        BackendTensor::apply(
            "int_matmul",
            matmul_shape(&lhs.shape(), &rhs.shape()).unwrap_or_else(|_| lhs.shape()),
            DType::I32,
            &[&lhs, &rhs],
            |v| v[0].matmul(v[1]),
        )
    }
    fn int_powi(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary("int_powi", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::Power))
    }
    fn int_powi_scalar_impl(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        one("int_powi_scalar", &lhs, lhs.dtype(), |a| a.word_scalar(WordOp::Power, rhs.elem::<i32>() as u32))
    }
    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        reduce("int_sum", &tensor, None, Reduction::Sum, DType::I32)
    }
    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce("int_sum_dim", &tensor, Some(dim), Reduction::Sum, DType::I32)
    }
    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        reduce("int_prod", &tensor, None, Reduction::Product, DType::I32)
    }
    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce("int_prod_dim", &tensor, Some(dim), Reduction::Product, DType::I32)
    }
    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        BackendTensor::apply("int_mean", Shape::new([1]), DType::I32, &[&tensor], |v| {
            // Integer means must sum once before division. A sequence of
            // per-axis integer means would truncate at each intermediate axis.
            let tensor = if v[0].layout().is_contiguous() { v[0].clone() } else { v[0].copy()? };
            let length = tensor.layout().len();
            tensor.reshape(Shape::new([length]))?.reduce(0, Reduction::Mean)
        })
    }
    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce("int_mean_dim", &tensor, Some(dim), Reduction::Mean, DType::I32)
    }
    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        one("int_cumsum", &tensor, tensor.dtype(), |t| t.scan(dim, Scan::Sum))
    }
    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        one("int_cumprod", &tensor, tensor.dtype(), |t| t.scan(dim, Scan::Product))
    }
    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        one("int_cummin", &tensor, tensor.dtype(), |t| t.scan(dim, Scan::Min))
    }
    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        one("int_cummax", &tensor, tensor.dtype(), |t| t.scan(dim, Scan::Max))
    }
    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce("int_argmax", &tensor, Some(dim), Reduction::ArgMax, DType::I32)
    }
    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce("int_argmin", &tensor, Some(dim), Reduction::ArgMin, DType::I32)
    }
    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        reduce("int_min", &tensor, None, Reduction::Min, DType::I32)
    }
    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        reduce("int_max", &tensor, None, Reduction::Max, DType::I32)
    }
    fn int_min_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce("int_min_dim", &tensor, Some(dim), Reduction::Min, DType::I32)
    }
    fn int_max_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce("int_max_dim", &tensor, Some(dim), Reduction::Max, DType::I32)
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
            NativeTensor::full_scalar(shape, dtype.into(), fill_value)
        })
    }
    fn int_scatter_nd(data: BackendTensor, indices: BackendTensor, values: BackendTensor, update: IndexingUpdateOp) -> BackendTensor {
        BackendTensor::apply("int_scatter_nd", data.shape(), data.dtype(), &[&data, &indices, &values], |v| {
            v[0].scatter_nd(v[1], v[2], update)
        })
    }
    fn int_gather_nd(data: BackendTensor, indices: BackendTensor) -> BackendTensor {
        BackendTensor::apply(
            "int_gather_nd",
            nd_shape(&data.shape(), &indices.shape()).unwrap_or_else(|_| data.shape()),
            data.dtype(),
            &[&data, &indices],
            |v| v[0].gather_nd(v[1]),
        )
    }
    fn int_sort(tensor: BackendTensor, dim: usize, descending: bool) -> BackendTensor {
        one("int_sort", &tensor, tensor.dtype(), |t| t.sort(dim, descending).map(|(values, _)| values))
    }
    fn int_topk(tensor: BackendTensor, dim: usize, k: usize) -> BackendTensor {
        BackendTensor::apply("int_topk", topk_shape(&tensor, dim, k), tensor.dtype(), &[&tensor], |v| {
            v[0].topk(dim, k).map(|(values, _)| values)
        })
    }
    fn int_topk_with_indices(tensor: BackendTensor, dim: usize, k: usize) -> (BackendTensor, BackendTensor) {
        topk_pair("int_topk_with_indices", &tensor, dim, k, burn_backend::IntDType::I32)
    }
    fn int_argtopk(tensor: BackendTensor, dim: usize, k: usize) -> BackendTensor {
        BackendTensor::apply("int_argtopk", topk_shape(&tensor, dim, k), DType::I32, &[&tensor], |v| {
            v[0].topk(dim, k).map(|(_, indices)| indices)
        })
    }
    fn int_sort_with_indices(tensor: BackendTensor, dim: usize, descending: bool) -> (BackendTensor, BackendTensor) {
        sorted_pair("int_sort_with_indices", &tensor, dim, descending, burn_backend::IntDType::I32)
    }
    fn int_argsort(tensor: BackendTensor, dim: usize, descending: bool) -> BackendTensor {
        one("int_argsort", &tensor, DType::I32, |t| t.sort(dim, descending).map(|(_, indices)| indices))
    }
    fn int_arange_step(range: core::ops::Range<i64>, step: usize, _device: &NativeDevice, dtype: IntDType) -> BackendTensor {
        let plan = crate::range::RangePlan::new(range, step);
        let shape = plan.as_ref().map_or_else(|_| Shape::new([0]), |p| Shape::new([p.len()]));
        BackendTensor::build("int_arange", shape, dtype.into(), || {
            if dtype != IntDType::I32 {
                return Err(Error::InvalidParameter);
            }
            NativeTensor::arange(plan?)
        })
    }
}
