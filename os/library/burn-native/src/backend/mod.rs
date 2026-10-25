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
    layout::{Layout, broadcast, matmul_shape},
};
use alloc::{string::String, vec::Vec};
use burn_backend::{
    Backend, BackendTypes, DType, DTypeUsage, DTypeUsageSet, ExecutionError, GraphUnsupported, Shape, TensorMetadata, distributed::DistributedOps,
    ops::TransactionOps,
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{elementwise::Elementwise, reduction::Reduction, words::WordOp};
pub use execution::BackendTensor;
use execution::{CONTEXT, Cause};
pub use quantized::BackendQuantizedTensor;

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
            .map_err(|error| CONTEXT.fail("synchronize", Cause::Native(error)).report())?;
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
        if let Some(axis) = axis {
            v[0].reduce(axis, reduction)
        } else {
            Nvidia.record_commands(|| {
                let mut tensor = v[0].clone();
                for axis in (0..tensor.shape().num_dims()).rev() {
                    tensor = tensor.reduce(axis, reduction)?;
                }
                tensor.reshape(Shape::new([1]))
            })
        }
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
