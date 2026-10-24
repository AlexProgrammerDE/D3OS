use super::*;
use burn_backend::ops::*;
use burn_backend::tensor::{BoolTensor, Device, FloatTensor, IndexingUpdateOp, IntTensor};
use burn_backend::{BoolDType, Distribution, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData};
impl FloatTensorOps<Self> for Native {
    fn float_from_data(data: TensorData, _device: &Device<Self>) -> FloatTensor<Self> {
        BackendTensor::build("float_from_data", data.shape().clone(), data.dtype(), || {
            if data.dtype() != DType::F32 {
                return Err(Error::InvalidParameter);
            }
            NativeTensor::from_data(data)
        })
    }
    fn float_random(shape: Shape, distribution: Distribution, _device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        BackendTensor::build("float_random", shape.clone(), dtype.into(), || {
            NativeTensor::random(shape, dtype.into(), distribution)
        })
    }
    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        tensor.read()
    }
    fn float_to_device(tensor: FloatTensor<Self>, _device: &Device<Self>) -> FloatTensor<Self> {
        tensor
    }
    fn float_into_int(tensor: FloatTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        cast("float_into_int", &tensor, out_dtype.into())
    }
    fn float_empty(shape: Shape, _device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        BackendTensor::build("float_empty", shape.clone(), dtype.into(), || NativeTensor::full(shape, dtype.into(), 0))
    }
    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary("float_add", &lhs, &rhs, lhs.dtype(), |a, b| a.binary(b, Elementwise::Add))
    }
    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        one("float_add_scalar", &lhs, lhs.dtype(), |a| a.scalar(Elementwise::Add, rhs.elem::<f32>()))
    }
    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary("float_sub", &lhs, &rhs, lhs.dtype(), |a, b| a.binary(b, Elementwise::Subtract))
    }
    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        one("float_sub_scalar", &lhs, lhs.dtype(), |a| a.scalar(Elementwise::Subtract, rhs.elem::<f32>()))
    }
    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary("float_mul", &lhs, &rhs, lhs.dtype(), |a, b| a.binary(b, Elementwise::Multiply))
    }
    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        one("float_mul_scalar", &lhs, lhs.dtype(), |a| a.scalar(Elementwise::Multiply, rhs.elem::<f32>()))
    }
    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary("float_div", &lhs, &rhs, lhs.dtype(), |a, b| a.binary(b, Elementwise::Divide))
    }
    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        one("float_div_scalar", &lhs, lhs.dtype(), |a| a.scalar(Elementwise::Divide, rhs.elem::<f32>()))
    }
    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_remainder", lhs.shape(), DType::F32, &[&lhs, &rhs], "float_remainder")
    }
    fn float_remainder_scalar(lhs: FloatTensor<Self>, _rhs: Scalar) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_remainder_scalar", lhs.shape(), DType::F32, &[&lhs], "float_remainder_scalar")
    }
    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::apply(
            "float_matmul",
            matmul_shape(&lhs.shape(), &rhs.shape()).unwrap_or_else(|_| lhs.shape()),
            DType::F32,
            &[&lhs, &rhs],
            |v| v[0].matmul(v[1]),
        )
    }
    fn float_cross(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, _dim: usize) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_cross", lhs.shape(), DType::F32, &[&lhs, &rhs], "float_cross")
    }
    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_recip", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Reciprocal))
    }
    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        view("float_swap_dims", &tensor, layout(&tensor).and_then(|l| l.swap_dims(dim1, dim2)))
    }
    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        view("float_permute", &tensor, layout(&tensor).and_then(|l| l.permute(axes)))
    }
    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        view("float_flip", &tensor, layout(&tensor).and_then(|l| l.flip(axes)))
    }
    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        reshape("float_reshape", &tensor, shape)
    }
    fn float_gather(dim: usize, tensor: FloatTensor<Self>, indices: IntTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::apply("float_gather", indices.shape(), tensor.dtype(), &[&tensor, &indices], |v| {
            v[0].gather(dim, v[1])
        })
    }
    fn float_scatter(dim: usize, tensor: FloatTensor<Self>, indices: IntTensor<Self>, value: FloatTensor<Self>, update: IndexingUpdateOp) -> FloatTensor<Self> {
        if update != IndexingUpdateOp::Add {
            return BackendTensor::unsupported(
                "float_scatter",
                tensor.shape(),
                tensor.dtype(),
                &[&tensor, &indices, &value],
                "non-additive indexed updates",
            );
        }
        BackendTensor::apply("float_scatter", tensor.shape(), tensor.dtype(), &[&tensor, &indices, &value], |v| {
            v[0].scatter_add(dim, v[1], v[2])
        })
    }
    fn float_select(tensor: FloatTensor<Self>, dim: usize, indices: IntTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::apply(
            "float_select",
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
    fn float_select_assign(
        tensor: FloatTensor<Self>, dim: usize, indices: IntTensor<Self>, value: FloatTensor<Self>, update: IndexingUpdateOp,
    ) -> FloatTensor<Self> {
        if update != IndexingUpdateOp::Add {
            return BackendTensor::unsupported(
                "float_select_assign",
                tensor.shape(),
                tensor.dtype(),
                &[&tensor, &indices, &value],
                "non-additive indexed updates",
            );
        }
        BackendTensor::apply("float_select_assign", tensor.shape(), tensor.dtype(), &[&tensor, &indices, &value], |v| {
            v[0].select_add(dim, v[1], v[2])
        })
    }
    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        view("float_slice", &tensor, layout(&tensor).and_then(|l| l.slice(slices)))
    }
    fn float_slice_assign(tensor: FloatTensor<Self>, slices: &[Slice], value: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::apply("float_slice_assign", tensor.shape(), tensor.dtype(), &[&tensor, &value], |v| {
            v[0].slice_assign(slices, v[1])
        })
    }
    fn float_mask_where(tensor: FloatTensor<Self>, mask: BoolTensor<Self>, value: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::apply("float_mask_where", tensor.shape(), tensor.dtype(), &[&tensor, &mask, &value], |v| {
            v[0].mask_where(v[1], v[2])
        })
    }
    fn float_mask_fill(tensor: FloatTensor<Self>, mask: BoolTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        BackendTensor::apply("float_mask_fill", tensor.shape(), tensor.dtype(), &[&tensor, &mask], |v| {
            v[0].mask_fill(v[1], value.elem::<f32>())
        })
    }
    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("float_equal", &lhs, &rhs, out_dtype.into(), |a, b| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.binary(b, Elementwise::Equal)
        })
    }
    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("float_equal_elem", &lhs, out_dtype.into(), |a| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.scalar(Elementwise::Equal, rhs.elem::<f32>())
        })
    }
    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("float_greater", &lhs, &rhs, out_dtype.into(), |a, b| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.binary(b, Elementwise::Greater)
        })
    }
    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("float_greater_elem", &lhs, out_dtype.into(), |a| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.scalar(Elementwise::Greater, rhs.elem::<f32>())
        })
    }
    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("float_greater_equal", &lhs, &rhs, out_dtype.into(), |a, b| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.binary(b, Elementwise::GreaterEqual)
        })
    }
    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("float_greater_equal_elem", &lhs, out_dtype.into(), |a| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.scalar(Elementwise::GreaterEqual, rhs.elem::<f32>())
        })
    }
    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("float_lower", &lhs, &rhs, out_dtype.into(), |a, b| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.binary(b, Elementwise::Less)
        })
    }
    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("float_lower_elem", &lhs, out_dtype.into(), |a| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.scalar(Elementwise::Less, rhs.elem::<f32>())
        })
    }
    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        binary("float_lower_equal", &lhs, &rhs, out_dtype.into(), |a, b| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.binary(b, Elementwise::LessEqual)
        })
    }
    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        one("float_lower_equal_elem", &lhs, out_dtype.into(), |a| {
            if DType::from(out_dtype) != DType::Bool(burn_backend::BoolStore::U32) {
                return Err(Error::InvalidParameter);
            }
            a.scalar(Elementwise::LessEqual, rhs.elem::<f32>())
        })
    }
    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        reduce("float_sum", &tensor, None, Reduction::Sum, tensor.dtype())
    }
    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce("float_sum_dim", &tensor, Some(dim), Reduction::Sum, tensor.dtype())
    }
    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce("float_mean_dim", &tensor, Some(dim), Reduction::Mean, tensor.dtype())
    }
    fn float_cumsum(tensor: FloatTensor<Self>, _dim: usize) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_cumsum", tensor.shape(), DType::F32, &[&tensor], "float_cumsum")
    }
    fn float_cumprod(tensor: FloatTensor<Self>, _dim: usize) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_cumprod", tensor.shape(), DType::F32, &[&tensor], "float_cumprod")
    }
    fn float_cummin(tensor: FloatTensor<Self>, _dim: usize) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_cummin", tensor.shape(), DType::F32, &[&tensor], "float_cummin")
    }
    fn float_cummax(tensor: FloatTensor<Self>, _dim: usize) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_cummax", tensor.shape(), DType::F32, &[&tensor], "float_cummax")
    }
    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        cast("float_cast", &tensor, dtype.into())
    }
    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_exp", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Exp))
    }
    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_log", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Log))
    }
    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_log1p", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Log1p))
    }
    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary("float_powf", &lhs, &rhs, lhs.dtype(), |a, b| a.binary(b, Elementwise::Power))
    }
    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        one("float_powf_scalar", &tensor, tensor.dtype(), |t| {
            t.scalar(Elementwise::Power, value.elem::<f32>())
        })
    }
    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_sqrt", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Sqrt))
    }
    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_abs", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Absolute))
    }
    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_cos", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Cos))
    }
    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_sin", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Sin))
    }
    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_tan", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Tan))
    }
    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_cosh", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Cosh))
    }
    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_sinh", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Sinh))
    }
    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_tanh", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Tanh))
    }
    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_acos", tensor.shape(), DType::F32, &[&tensor], "float_acos")
    }
    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_acosh", tensor.shape(), DType::F32, &[&tensor], "float_acosh")
    }
    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_asin", tensor.shape(), DType::F32, &[&tensor], "float_asin")
    }
    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_asinh", tensor.shape(), DType::F32, &[&tensor], "float_asinh")
    }
    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_atan", tensor.shape(), DType::F32, &[&tensor], "float_atan")
    }
    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_atanh", tensor.shape(), DType::F32, &[&tensor], "float_atanh")
    }
    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_atan2", lhs.shape(), DType::F32, &[&lhs, &rhs], "float_atan2")
    }
    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_round", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Round))
    }
    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_floor", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Floor))
    }
    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_ceil", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Ceil))
    }
    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        one("float_trunc", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Trunc))
    }
    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::unsupported("float_erf", tensor.shape(), DType::F32, &[&tensor], "float_erf")
    }
    fn float_argmax(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        reduce("float_argmax", &tensor, Some(dim), Reduction::ArgMax, out_dtype.into())
    }
    fn float_argmin(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        reduce("float_argmin", &tensor, Some(dim), Reduction::ArgMin, out_dtype.into())
    }
    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        view("float_expand", &tensor, layout(&tensor).and_then(|l| l.expand(shape)))
    }
    fn float_unfold(tensor: FloatTensor<Self>, dim: usize, size: usize, step: usize) -> FloatTensor<Self> {
        view("float_unfold", &tensor, layout(&tensor).and_then(|l| l.unfold(dim, size, step)))
    }
    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        cat("float_cat", tensors, dim)
    }
    fn float_zeros(shape: Shape, _device: &NativeDevice, dtype: FloatDType) -> BackendTensor {
        BackendTensor::build("float_zeros", shape.clone(), dtype.into(), || {
            NativeTensor::full(shape, dtype.into(), 0f32.to_bits())
        })
    }
    fn float_ones(shape: Shape, _device: &NativeDevice, dtype: FloatDType) -> BackendTensor {
        BackendTensor::build("float_ones", shape.clone(), dtype.into(), || {
            NativeTensor::full(shape, dtype.into(), 1f32.to_bits())
        })
    }
    fn float_full(shape: Shape, fill_value: Scalar, _device: &NativeDevice, dtype: FloatDType) -> BackendTensor {
        BackendTensor::build("float_full", shape.clone(), dtype.into(), || {
            NativeTensor::full(shape, dtype.into(), fill_value.elem::<f32>().to_bits())
        })
    }
    fn float_neg(tensor: BackendTensor) -> BackendTensor {
        one("float_neg", &tensor, tensor.dtype(), |t| t.unary(Elementwise::Negate))
    }
    fn float_prod(tensor: BackendTensor) -> BackendTensor {
        reduce("float_prod", &tensor, None, Reduction::Product, tensor.dtype())
    }
    fn float_prod_dim(tensor: BackendTensor, dim: usize) -> BackendTensor {
        reduce("float_prod_dim", &tensor, Some(dim), Reduction::Product, tensor.dtype())
    }
    fn float_scatter_nd(data: BackendTensor, indices: BackendTensor, values: BackendTensor, _update: IndexingUpdateOp) -> BackendTensor {
        BackendTensor::unsupported(
            "float_scatter_nd",
            data.shape(),
            data.dtype(),
            &[&data, &indices, &values],
            "multidimensional scatter",
        )
    }
    fn float_gather_nd(data: BackendTensor, indices: BackendTensor) -> BackendTensor {
        BackendTensor::unsupported("float_gather_nd", data.shape(), data.dtype(), &[&data, &indices], "multidimensional gather")
    }
    fn float_sort(tensor: BackendTensor, _dim: usize, _descending: bool) -> BackendTensor {
        BackendTensor::unsupported("float_sort", tensor.shape(), tensor.dtype(), &[&tensor], "GPU sorting")
    }
    fn float_sort_with_indices(tensor: BackendTensor, _dim: usize, _descending: bool, indices_dtype: IntDType) -> (BackendTensor, BackendTensor) {
        let output = BackendTensor::unsupported("float_sort_with_indices", tensor.shape(), tensor.dtype(), &[&tensor], "GPU sorting");
        let indices = BackendTensor::unsupported("float_sort_with_indices", tensor.shape(), indices_dtype.into(), &[&output], "GPU sorting");
        (output, indices)
    }
    fn float_argsort(tensor: BackendTensor, _dim: usize, _descending: bool, out_dtype: IntDType) -> BackendTensor {
        BackendTensor::unsupported("float_argsort", tensor.shape(), out_dtype.into(), &[&tensor], "GPU sorting")
    }
    async fn float_mask_select(tensor: BackendTensor, mask: BackendTensor) -> BackendTensor {
        BackendTensor::unsupported("float_mask_select", Shape::new([0]), tensor.dtype(), &[&tensor, &mask], "GPU compaction")
    }
}
