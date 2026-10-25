use super::*;
use burn_backend::ops::*;
use burn_backend::tensor::{BoolTensor, Device, FloatTensor, IntTensor};
use burn_backend::{BoolDType, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData};
impl BoolTensorOps<Self> for Native {
    fn bool_empty(shape: Shape, _device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self> {
        BackendTensor::build("bool_empty", shape.clone(), dtype.into(), || NativeTensor::full(shape, dtype.into(), 0))
    }
    fn bool_zeros(shape: Shape, _device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self> {
        BackendTensor::build("bool_zeros", shape.clone(), dtype.into(), || NativeTensor::full(shape, dtype.into(), 0))
    }
    fn bool_ones(shape: Shape, _device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self> {
        BackendTensor::build("bool_ones", shape.clone(), dtype.into(), || NativeTensor::full(shape, dtype.into(), 1))
    }
    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        tensor.read()
    }
    fn bool_from_data(data: TensorData, _device: &Device<Self>) -> BoolTensor<Self> {
        let dtype = if data.dtype() == DType::Bool(burn_backend::BoolStore::Native) {
            DType::Bool(burn_backend::BoolStore::U32)
        } else {
            data.dtype()
        };
        BackendTensor::build("bool_from_data", data.shape().clone(), dtype, || {
            if !data.dtype().is_bool() {
                return Err(Error::InvalidParameter);
            }
            NativeTensor::from_data(data)
        })
    }
    fn bool_into_int(tensor: BoolTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        cast("bool_into_int", &tensor, out_dtype.into())
    }
    fn bool_into_float(tensor: BoolTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        cast("bool_into_float", &tensor, out_dtype.into())
    }
    fn bool_to_device(tensor: BoolTensor<Self>, _device: &Device<Self>) -> BoolTensor<Self> {
        tensor
    }
    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        reshape("bool_reshape", &tensor, shape)
    }
    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        view("bool_slice", &tensor, layout(&tensor).and_then(|l| l.slice(slices)))
    }
    fn bool_slice_assign(tensor: BoolTensor<Self>, slices: &[Slice], value: BoolTensor<Self>) -> BoolTensor<Self> {
        BackendTensor::apply("bool_slice_assign", tensor.shape(), tensor.dtype(), &[&tensor, &value], |v| {
            v[0].slice_assign(slices, v[1])
        })
    }
    fn bool_mask_where(tensor: BoolTensor<Self>, mask: BoolTensor<Self>, value: BoolTensor<Self>) -> BoolTensor<Self> {
        BackendTensor::apply("bool_mask_where", tensor.shape(), tensor.dtype(), &[&tensor, &mask, &value], |v| {
            v[0].mask_where(v[1], v[2])
        })
    }
    fn bool_mask_fill(tensor: BoolTensor<Self>, mask: BoolTensor<Self>, value: Scalar) -> BoolTensor<Self> {
        BackendTensor::apply("bool_mask_fill", tensor.shape(), tensor.dtype(), &[&tensor, &mask], |v| {
            v[0].mask_fill_bool(v[1], value.elem::<bool>())
        })
    }
    fn bool_gather(dim: usize, tensor: BoolTensor<Self>, indices: IntTensor<Self>) -> BoolTensor<Self> {
        BackendTensor::apply("bool_gather", indices.shape(), tensor.dtype(), &[&tensor, &indices], |v| v[0].gather(dim, v[1]))
    }
    fn bool_scatter_or(dim: usize, tensor: BoolTensor<Self>, indices: IntTensor<Self>, value: BoolTensor<Self>) -> BoolTensor<Self> {
        BackendTensor::apply("bool_scatter_or", tensor.shape(), tensor.dtype(), &[&tensor, &indices, &value], |v| {
            v[0].scatter_or(dim, v[1], v[2])
        })
    }
    fn bool_select(tensor: BoolTensor<Self>, dim: usize, indices: IntTensor<Self>) -> BoolTensor<Self> {
        BackendTensor::apply(
            "bool_select",
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
    fn bool_select_or(tensor: BoolTensor<Self>, dim: usize, indices: IntTensor<Self>, value: BoolTensor<Self>) -> BoolTensor<Self> {
        BackendTensor::apply("bool_select_or", tensor.shape(), tensor.dtype(), &[&tensor, &indices, &value], |v| {
            v[0].select_or(dim, v[1], v[2])
        })
    }
    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        binary("bool_equal", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::BoolEqual))
    }
    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        one("bool_equal_elem", &lhs, lhs.dtype(), |a| {
            a.word_scalar(WordOp::BoolEqual, u32::from(rhs.elem::<bool>()))
        })
    }
    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        one("bool_not", &tensor, tensor.dtype(), |t| t.word_unary(WordOp::BoolNot))
    }
    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        binary("bool_and", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::BoolAnd))
    }
    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        binary("bool_or", &lhs, &rhs, lhs.dtype(), |a, b| a.word_binary(b, WordOp::BoolOr))
    }
    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        view("bool_swap_dims", &tensor, layout(&tensor).and_then(|l| l.swap_dims(dim1, dim2)))
    }
    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        view("bool_permute", &tensor, layout(&tensor).and_then(|l| l.permute(axes)))
    }
    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        view("bool_flip", &tensor, layout(&tensor).and_then(|l| l.flip(axes)))
    }
    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        view("bool_expand", &tensor, layout(&tensor).and_then(|l| l.expand(shape)))
    }
    fn bool_unfold(tensor: BoolTensor<Self>, dim: usize, size: usize, step: usize) -> BoolTensor<Self> {
        view("bool_unfold", &tensor, layout(&tensor).and_then(|l| l.unfold(dim, size, step)))
    }
    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        cat("bool_cat", tensors, dim)
    }
    async fn bool_argwhere(tensor: BackendTensor, out_dtype: IntDType) -> BackendTensor {
        BackendTensor::apply("bool_argwhere", Shape::new([0, tensor.rank()]), out_dtype.into(), &[&tensor], |v| {
            if out_dtype != IntDType::I32 {
                return Err(Error::InvalidParameter);
            }
            v[0].argwhere()
        })
    }
}
