use super::*;
use crate::quantization::NativeQuantizedTensor;
use alloc::sync::Arc;
use burn_backend::ops::*;
use burn_backend::quantization::{QuantScheme, QuantizationParametersPrimitive};
use burn_backend::tensor::{Device, FloatTensor, QuantizedTensor};
use burn_backend::{ExecutionError, FloatDType, Shape, TensorData};
use execution::Failure;

/// Quantized values and scales share ownership independently of float tensors.
#[derive(Clone, Debug)]
pub struct BackendQuantizedTensor {
    shape: Shape,
    scheme: QuantScheme,
    value: Result<NativeQuantizedTensor, Arc<Failure>>,
}

impl TensorMetadata for BackendQuantizedTensor {
    type Device = NativeDevice;
    fn shape(&self) -> Shape {
        self.shape.clone()
    }
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }
    fn device(&self) -> NativeDevice {
        NativeDevice
    }
    fn can_mut(&self) -> bool {
        false
    }
}

impl BackendQuantizedTensor {
    fn build(
        operation: &'static str, shape: Shape, scheme: QuantScheme, inputs: &[&BackendTensor],
        work: impl FnOnce() -> d3os_compute::Result<NativeQuantizedTensor>,
    ) -> Self {
        let failure = inputs.iter().find_map(|t| t.failure()).or_else(|| CONTEXT.poisoned());
        let value = match failure {
            Some(failure) => Err(failure),
            None => work().map_err(|e| CONTEXT.fail(operation, Cause::Native(e))),
        };
        Self::from_result(shape, scheme, value)
    }

    fn from_result(shape: Shape, scheme: QuantScheme, value: Result<NativeQuantizedTensor, Arc<Failure>>) -> Self {
        match value {
            Ok(value) => Self {
                shape: value.shape(),
                scheme: value.scheme(),
                value: Ok(value),
            },
            Err(failure) => Self {
                shape,
                scheme,
                value: Err(failure),
            },
        }
    }

    fn map(self, operation: &'static str, shape: Shape, work: impl FnOnce(&NativeQuantizedTensor) -> d3os_compute::Result<NativeQuantizedTensor>) -> Self {
        let value = match self.value {
            Err(failure) => Err(failure),
            Ok(value) => match CONTEXT.poisoned() {
                Some(failure) => Err(failure),
                None => work(&value).map_err(|e| CONTEXT.fail(operation, Cause::Native(e))),
            },
        };
        Self::from_result(shape, self.scheme, value)
    }
}

impl QTensorOps<Self> for Native {
    fn q_from_data(data: TensorData, _device: &Device<Self>) -> QuantizedTensor<Self> {
        let scheme = match data.dtype() {
            DType::QFloat(scheme) => scheme,
            _ => QuantScheme::default(),
        };
        BackendQuantizedTensor::build("q_from_data", data.shape().clone(), scheme, &[], || NativeQuantizedTensor::from_data(data))
    }

    fn quantize(tensor: FloatTensor<Self>, scheme: &QuantScheme, qparams: QuantizationParametersPrimitive<Self>) -> QuantizedTensor<Self> {
        let mut inputs = alloc::vec![&tensor, &qparams.scales];
        if let Some(global) = &qparams.global {
            inputs.push(global);
        }
        BackendQuantizedTensor::build("quantize", tensor.shape(), *scheme, &inputs, || {
            if qparams.global.is_some() {
                return Err(Error::InvalidParameter);
            }
            NativeQuantizedTensor::quantize(
                tensor.value().ok_or(Error::InvalidParameter)?,
                qparams.scales.value().ok_or(Error::InvalidParameter)?,
                *scheme,
            )
        })
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let value = match tensor.value {
            Err(failure) => return BackendTensor::failed(tensor.shape, dtype.into(), failure),
            Ok(value) => value,
        };
        BackendTensor::build("dequantize", tensor.shape, dtype.into(), || {
            if dtype != FloatDType::F32 {
                return Err(Error::InvalidParameter);
            }
            value.dequantize()
        })
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, _device: &Device<Self>) -> QuantizedTensor<Self> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        tensor.map("q_reshape", shape.clone(), |v| v.reshape(shape))
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        let value = tensor.value.map_err(|failure| failure.report())?;
        if let Some(failure) = CONTEXT.poisoned() {
            return Err(failure.report());
        }
        value.read().map_err(|e| CONTEXT.fail("q_into_data", Cause::Native(e)).report())
    }

    fn q_swap_dims(tensor: QuantizedTensor<Self>, dim1: usize, dim2: usize) -> QuantizedTensor<Self> {
        let shape = tensor.shape();
        tensor.map("q_swap_dims", shape.clone(), |v| {
            if dim1 >= shape.num_dims() || dim2 >= shape.num_dims() {
                return Err(Error::InvalidParameter);
            }
            let mut axes: Vec<_> = (0..shape.num_dims()).collect();
            axes.swap(dim1, dim2);
            v.permute(&axes)
        })
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        let shape = tensor.shape();
        tensor.map("q_permute", shape, |v| v.permute(axes))
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        let shape = tensor.shape();
        tensor.map("q_flip", shape, |v| v.flip(axes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_dependencies_keep_an_acknowledged_float_failure() {
        let source = Native::float_from_data(TensorData::from([1i32]), &NativeDevice);
        let original = source.failure().unwrap();
        original.report();
        let quantized = Native::quantize_dynamic(source, &QuantScheme::default());
        let failure = quantized.value.as_ref().unwrap_err();
        assert!(Arc::ptr_eq(failure, &original));
        let decoded = Native::dequantize(quantized, FloatDType::F32);
        assert!(Arc::ptr_eq(&decoded.failure().unwrap(), &original));
    }

    #[test]
    fn malformed_quantized_storage_fails_before_any_gpu_upload() {
        for data in [
            TensorData::from_bytes_vec(alloc::vec![0; 4], [4], DType::QFloat(QuantScheme::default())),
            TensorData::from_bytes_vec(
                alloc::vec![0; 8],
                [4],
                DType::QFloat(QuantScheme::default().per_tensor(burn_backend::quantization::ScaleDtype::F16)),
            ),
        ] {
            let tensor = Native::q_from_data(data, &NativeDevice);
            let failure = tensor.value.unwrap_err();
            failure.report();
        }
    }
}
