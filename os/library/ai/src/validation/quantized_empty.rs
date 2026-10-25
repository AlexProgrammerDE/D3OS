//! Empty quantized operations and record round trips, checked on each backend.
use alloc::{format, string::String, vec::Vec};
use burn::{
    backend::{DType, TensorData},
    tensor::{
        Bool, Device, Int, Tensor,
        quantization::{QuantScheme, QuantizationParameters},
    },
};

#[derive(Clone, Copy)]
pub enum Case {
    Direct,
    Select,
    Mask,
}

fn check<const D: usize>(tensor: Tensor<D>, expected: [usize; D], device: &Device) -> Result<Vec<u8>, String> {
    if tensor.dims() != expected || !matches!(tensor.dtype(), DType::QFloat(_)) {
        return Err(String::from("empty quantized shape or kind"));
    }
    let data = tensor.clone().try_into_data().map_err(|e| format!("empty quantized record: {e:?}"))?;
    let bytes = data.as_bytes().to_vec();
    let decoded = tensor
        .dequantize()
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("empty dequantization: {e:?}"))?;
    if !decoded.is_empty() {
        return Err(String::from("nonempty quantized decode"));
    }
    let restored = Tensor::<D>::from_data(data, device);
    let stored = restored.clone().try_into_data().map_err(|e| format!("empty quantized reload: {e:?}"))?;
    if stored.as_bytes() != bytes || restored.dims() != expected {
        return Err(String::from("empty quantized record changed on reload"));
    }
    if !restored
        .dequantize()
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("reloaded empty decode: {e:?}"))?
        .is_empty()
    {
        return Err(String::from("nonempty reloaded quantized decode"));
    }
    Ok(bytes)
}

pub async fn run(device: &Device, case: Case) -> Result<Vec<u8>, String> {
    let scheme = QuantScheme::default();
    let params = || QuantizationParameters {
        scales: Tensor::<1>::from_floats([1.], device),
        global: None,
    };
    if matches!(case, Case::Direct) {
        let input = Tensor::<1>::from_data(TensorData::new(Vec::<f32>::new(), [0]), device);
        return check(input.quantize(&scheme, params()), [0], device);
    }
    let input = Tensor::<2>::from_floats([[-2., 0., 2., 1.], [2., 0., -2., -1.]], device).quantize(&scheme, params());
    match case {
        Case::Mask => check(
            input.mask_select_async(Tensor::<2, Bool>::from_bool([[false; 4]; 2], device)).await,
            [0],
            device,
        ),
        Case::Select => check(
            input.select(1, Tensor::<1, Int>::from_data(TensorData::new(Vec::<i32>::new(), [0]), device)),
            [2, 0],
            device,
        ),
        Case::Direct => unreachable!(),
    }
}
