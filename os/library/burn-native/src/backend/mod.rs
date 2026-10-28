//! Burn backend dispatch over the fallible native tensor implementation.
mod activation;
mod boolean;
mod execution;
mod float;
mod integer;
mod modules;
mod optimizer;
mod quantized;
mod qwen;

use crate::{
    NativeDevice, NativeTensor,
    layout::{Layout, broadcast, matmul_shape, nd_shape},
};
use alloc::{string::String, vec::Vec};
use burn_backend::{
    Backend, BackendTypes, DType, DTypeUsage, DTypeUsageSet, ExecutionError, GraphUnsupported, Shape, TensorMetadata, distributed::DistributedOps,
    ops::TransactionOps,
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::predicate::Predicate;
use d3os_nvidia::compute::scan::Scan;
use d3os_nvidia::compute::{elementwise::Elementwise, reduction::Reduction, words::WordOp};
pub use execution::BackendTensor;
use execution::CONTEXT;
pub use quantized::BackendQuantizedTensor;
pub use qwen::PrefillQuantization;
#[cfg(feature = "qwen-profiling")]
pub use qwen::profile::{ProjectionEntry, ProjectionProfiler};

/// D3OS's process-owned native NVIDIA backend.
#[derive(Clone, Debug, Default)]
pub struct Native;

impl BackendTypes for Native {
    type Device = NativeDevice;
    type FloatTensorPrimitive = BackendTensor;
    type IntTensorPrimitive = BackendTensor;
    type BoolTensorPrimitive = BackendTensor;
    type QuantizedTensorPrimitive = BackendQuantizedTensor;
    type GraphPrimitive = GraphUnsupported;
}

impl Backend for Native {
    fn name(_: &NativeDevice) -> String {
        String::from("d3os-native-nvidia")
    }
    fn seed(device: &NativeDevice, seed: u64) {
        device.seed(seed);
    }
    fn sync(_: &NativeDevice) -> Result<(), ExecutionError> {
        let semantic_error = CONTEXT.check();
        if semantic_error.as_ref().is_err_and(ExecutionError::is_device_poisoned) {
            return semantic_error;
        }
        Nvidia
            .with_commands(|| Ok::<_, Error>(()))
            .map_err(|error| CONTEXT.fail("synchronize", error).report())?;
        semantic_error?;
        CONTEXT.check()
    }
    fn flush(device: &NativeDevice) -> Result<(), ExecutionError> {
        Self::sync(device)
    }
    fn dtype_usage(_: &NativeDevice, dtype: DType) -> DTypeUsageSet {
        match dtype {
            DType::F32 => DTypeUsage::general() | DTypeUsage::Accelerated,
            DType::Bool(burn_backend::BoolStore::U32) => DTypeUsage::general(),
            DType::I32 => DTypeUsage::Storage.into(),
            DType::QFloat(scheme) if crate::quantization::supported_scheme(&scheme) => DTypeUsage::Storage.into(),
            _ => DTypeUsageSet::empty(),
        }
    }
    fn device_count(type_id: u16) -> usize {
        usize::from(type_id == 0)
    }
}

impl TransactionOps<Self> for Native {}
impl DistributedOps<Self> for Native {}

fn one(op: &'static str, tensor: &BackendTensor, dtype: DType, work: impl FnOnce(&NativeTensor) -> d3os_compute::Result<NativeTensor>) -> BackendTensor {
    BackendTensor::apply(op, tensor.shape(), dtype, &[tensor], |v| work(v[0]))
}

fn binary(
    op: &'static str, left: &BackendTensor, right: &BackendTensor, dtype: DType,
    work: impl FnOnce(&NativeTensor, &NativeTensor) -> d3os_compute::Result<NativeTensor>,
) -> BackendTensor {
    let shape = broadcast(&left.shape(), &right.shape()).unwrap_or_else(|_| left.shape());
    BackendTensor::apply(op, shape, dtype, &[left, right], |v| work(v[0], v[1]))
}

fn cast(op: &'static str, tensor: &BackendTensor, dtype: DType) -> BackendTensor {
    one(op, tensor, dtype, |t| t.cast(dtype))
}

fn sorted_pair(op: &'static str, tensor: &BackendTensor, axis: usize, descending: bool, index_dtype: DType) -> (BackendTensor, BackendTensor) {
    BackendTensor::pair(op, [tensor.shape(), tensor.shape()], [tensor.dtype(), index_dtype], &[tensor], |values| {
        if !index_dtype.is_int() && !index_dtype.is_uint() {
            return Err(Error::InvalidParameter);
        }
        let (values, indices) = values[0].sort(axis, descending)?;
        Ok((values, indices.cast(index_dtype)?))
    })
}

fn topk_shape(tensor: &BackendTensor, axis: usize, k: usize) -> Shape {
    let mut shape = tensor.shape();
    if let Some(size) = shape.get_mut(axis) {
        *size = k;
    }
    shape
}

fn topk_pair(op: &'static str, tensor: &BackendTensor, axis: usize, k: usize, index_dtype: DType) -> (BackendTensor, BackendTensor) {
    let shape = topk_shape(tensor, axis, k);
    BackendTensor::pair(op, [shape.clone(), shape], [tensor.dtype(), index_dtype], &[tensor], |values| {
        if !index_dtype.is_int() && !index_dtype.is_uint() {
            return Err(Error::InvalidParameter);
        }
        let (values, indices) = values[0].topk(axis, k)?;
        Ok((values, indices.cast(index_dtype)?))
    })
}

fn view(op: &'static str, tensor: &BackendTensor, layout: d3os_compute::Result<Layout>) -> BackendTensor {
    let shape = layout.as_ref().map_or_else(|_| tensor.shape(), |l| l.shape().clone());
    BackendTensor::apply(op, shape, tensor.dtype(), &[tensor], |v| v[0].with_layout(layout?))
}

fn layout(tensor: &BackendTensor) -> d3os_compute::Result<Layout> {
    tensor
        .value()
        .map(|t| Ok(t.layout().clone()))
        .unwrap_or_else(|| Layout::contiguous(tensor.shape()))
}

fn reshape(op: &'static str, tensor: &BackendTensor, shape: Shape) -> BackendTensor {
    BackendTensor::apply(op, shape.clone(), tensor.dtype(), &[tensor], |v| {
        let tensor = if v[0].layout().is_contiguous() { v[0].clone() } else { v[0].copy()? };
        tensor.reshape(shape)
    })
}

fn reduce(op: &'static str, tensor: &BackendTensor, axis: Option<usize>, reduction: Reduction, dtype: DType) -> BackendTensor {
    let mut shape = if axis.is_none() { Shape::new([1]) } else { tensor.shape() };
    if let Some(axis) = axis
        && let Some(d) = shape.get_mut(axis)
    {
        *d = 1;
    }
    BackendTensor::apply(op, shape, dtype, &[tensor], |v| {
        if reduction.index() && v[0].dtype.is_float() {
            return v[0].float_extrema(axis, reduction, dtype);
        }
        let reduced = if let Some(axis) = axis {
            v[0].reduce(axis, reduction)
        } else {
            v[0].reduce_all(reduction)
        }?;
        if reduction.index() { reduced.cast(dtype) } else { Ok(reduced) }
    })
}

fn reduction_pair(op: &'static str, tensor: &BackendTensor, axis: usize, reduction: Reduction, index_dtype: DType) -> (BackendTensor, BackendTensor) {
    let mut shape = tensor.shape();
    if let Some(dimension) = shape.get_mut(axis) {
        *dimension = 1;
    }
    BackendTensor::pair(op, [shape.clone(), shape], [tensor.dtype(), index_dtype], &[tensor], |values| {
        values[0].reduction_pair(axis, reduction, index_dtype)
    })
}

fn cat(op: &'static str, tensors: Vec<BackendTensor>, dim: usize) -> BackendTensor {
    let mut shape = tensors.first().map_or_else(|| Shape::new([0]), TensorMetadata::shape);
    let dtype = tensors.first().map_or(DType::F32, TensorMetadata::dtype);
    if let Some(length) = shape.get_mut(dim) {
        *length = tensors
            .iter()
            .try_fold(0usize, |n, t| n.checked_add(*t.shape().get(dim)?))
            .unwrap_or(usize::MAX);
    }
    let refs: Vec<_> = tensors.iter().collect();
    BackendTensor::apply(op, shape, dtype, &refs, |v| {
        NativeTensor::cat(&v.iter().map(|t| (*t).clone()).collect::<Vec<_>>(), dim)
    })
}

pub(crate) fn check_errors() -> Result<(), ExecutionError> {
    CONTEXT.check()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{
        IntDType,
        ops::{FloatTensorOps, IntTensorOps},
    };

    #[test]
    fn integer_reduction_failures_preserve_dtype_shape_and_the_original_error() {
        for dtype in [DType::I64, DType::I32, DType::I16, DType::I8, DType::U64, DType::U32, DType::U16, DType::U8] {
            let tensor = BackendTensor::build("input", Shape::new([2, 3]), dtype, || Err(Error::InvalidParameter));
            assert!(!tensor.failure().unwrap().report().is_device_poisoned());
            let failure = tensor.failure().unwrap();
            for reduced in [
                Native::int_sum(tensor.clone()),
                Native::int_prod(tensor.clone()),
                Native::int_mean(tensor.clone()),
                Native::int_min(tensor.clone()),
                Native::int_max(tensor.clone()),
            ] {
                assert_eq!(reduced.shape(), Shape::new([1]));
                assert_eq!(reduced.dtype(), dtype);
                assert!(alloc::sync::Arc::ptr_eq(&failure, &reduced.failure().unwrap()));
                assert!(!reduced.read().unwrap_err().is_device_poisoned());
            }
            for reduced in [
                Native::int_sum_dim(tensor.clone(), 1),
                Native::int_prod_dim(tensor.clone(), 1),
                Native::int_mean_dim(tensor.clone(), 1),
                Native::int_min_dim(tensor.clone(), 1),
                Native::int_max_dim(tensor.clone(), 1),
                Native::int_argmin(tensor.clone(), 1),
                Native::int_argmax(tensor.clone(), 1),
            ] {
                assert_eq!(reduced.shape(), Shape::new([2, 1]));
                assert_eq!(reduced.dtype(), dtype);
                assert!(alloc::sync::Arc::ptr_eq(&failure, &reduced.failure().unwrap()));
            }
            for (values, indices) in [
                Native::int_min_dim_with_indices(tensor.clone(), 1),
                Native::int_max_dim_with_indices(tensor.clone(), 1),
            ] {
                assert_eq!(values.shape(), Shape::new([2, 1]));
                assert_eq!(indices.shape(), values.shape());
                assert_eq!(values.dtype(), dtype);
                assert_eq!(indices.dtype(), dtype);
                assert!(alloc::sync::Arc::ptr_eq(&values.failure().unwrap(), &indices.failure().unwrap()));
                assert!(alloc::sync::Arc::ptr_eq(&failure, &values.failure().unwrap()));
            }
        }
    }

    #[test]
    fn float_pairs_keep_the_requested_index_metadata_and_shared_failure() {
        for dtype in [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16] {
            let tensor = BackendTensor::build("input", Shape::new([3, 257]), dtype, || Err(Error::InvalidParameter));
            assert!(!tensor.failure().unwrap().report().is_device_poisoned());
            for index_dtype in [
                IntDType::I64,
                IntDType::I32,
                IntDType::I16,
                IntDType::I8,
                IntDType::U64,
                IntDType::U32,
                IntDType::U16,
                IntDType::U8,
            ] {
                for (values, indices) in [
                    Native::float_min_dim_with_indices(tensor.clone(), 1, index_dtype),
                    Native::float_max_dim_with_indices(tensor.clone(), 1, index_dtype),
                ] {
                    assert_eq!(values.shape(), Shape::new([3, 1]));
                    assert_eq!(indices.shape(), values.shape());
                    assert_eq!(values.dtype(), dtype);
                    assert_eq!(indices.dtype(), DType::from(index_dtype));
                    assert!(alloc::sync::Arc::ptr_eq(&values.failure().unwrap(), &indices.failure().unwrap()));
                    assert!(!values.read().unwrap_err().is_device_poisoned());
                }
            }
        }
    }

    #[test]
    fn float_extrema_propagate_deferred_errors_with_reduced_metadata() {
        for dtype in [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16] {
            let tensor = BackendTensor::build("input", Shape::new([0, 257]), dtype, || Err(Error::InvalidParameter));
            let failure = tensor.failure().unwrap();
            assert!(!failure.report().is_device_poisoned());
            for reduced in [Native::float_min(tensor.clone()), Native::float_max(tensor.clone())] {
                assert_eq!(reduced.shape(), Shape::new([1]));
                assert_eq!(reduced.dtype(), dtype);
                assert!(alloc::sync::Arc::ptr_eq(&failure, &reduced.failure().unwrap()));
            }
            for reduced in [Native::float_min_dim(tensor.clone(), 1), Native::float_max_dim(tensor.clone(), 1)] {
                assert_eq!(reduced.shape(), Shape::new([0, 1]));
                assert_eq!(reduced.dtype(), dtype);
                assert!(alloc::sync::Arc::ptr_eq(&failure, &reduced.failure().unwrap()));
            }
            for index_dtype in [
                IntDType::I64,
                IntDType::I32,
                IntDType::I16,
                IntDType::I8,
                IntDType::U64,
                IntDType::U32,
                IntDType::U16,
                IntDType::U8,
            ] {
                for reduced in [
                    Native::float_argmin(tensor.clone(), 1, index_dtype),
                    Native::float_argmax(tensor.clone(), 1, index_dtype),
                ] {
                    assert_eq!(reduced.shape(), Shape::new([0, 1]));
                    assert_eq!(reduced.dtype(), DType::from(index_dtype));
                    assert!(alloc::sync::Arc::ptr_eq(&failure, &reduced.failure().unwrap()));
                    assert!(!reduced.read().unwrap_err().is_device_poisoned());
                }
            }
        }
    }
}
