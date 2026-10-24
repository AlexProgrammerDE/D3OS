use super::*;
use burn_backend::ops::*;
use burn_backend::quantization::{QuantScheme, QuantizationParametersPrimitive};
use burn_backend::tensor::{Device, FloatTensor, QuantizedTensor};
use burn_backend::{ExecutionError, FloatDType, Shape, TensorData};
impl QTensorOps<Self> for Native {
    fn q_from_data(data: TensorData, _device: &Device<Self>) -> QuantizedTensor<Self> {
        BackendTensor::unsupported("q_from_data", data.shape().clone(), data.dtype(), &[], "q_from_data")
    }
    fn quantize(tensor: FloatTensor<Self>, scheme: &QuantScheme, qparams: QuantizationParametersPrimitive<Self>) -> QuantizedTensor<Self> {
        let mut inputs = alloc::vec![&tensor, &qparams.scales];
        if let Some(global) = &qparams.global {
            inputs.push(global);
        }
        BackendTensor::unsupported("quantize", tensor.shape(), DType::QFloat(*scheme), &inputs, "quantize")
    }
    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        BackendTensor::unsupported("dequantize", tensor.shape(), dtype.into(), &[&tensor], "dequantize")
    }
    fn q_to_device(tensor: QuantizedTensor<Self>, _device: &Device<Self>) -> QuantizedTensor<Self> {
        tensor
    }
    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        BackendTensor::unsupported("q_reshape", shape, tensor.dtype(), &[&tensor], "q_reshape")
    }
    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        tensor.read()
    }
    fn q_swap_dims(tensor: QuantizedTensor<Self>, _dim1: usize, _dim2: usize) -> QuantizedTensor<Self> {
        BackendTensor::unsupported("q_swap_dims", tensor.shape(), tensor.dtype(), &[&tensor], "q_swap_dims")
    }
    fn q_permute(tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        BackendTensor::unsupported("q_permute", tensor.shape(), tensor.dtype(), &[&tensor], "q_permute")
    }
    fn q_flip(tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        BackendTensor::unsupported("q_flip", tensor.shape(), tensor.dtype(), &[&tensor], "q_flip")
    }
    fn quantize_dynamic(tensor: BackendTensor, scheme: &QuantScheme) -> BackendTensor {
        BackendTensor::unsupported("quantize_dynamic", tensor.shape(), DType::QFloat(*scheme), &[&tensor], "Burn quantization")
    }
}
