use super::ops::{MatrixPrimitive, QwenOps};
use alloc::vec::Vec;
use burn::{
    backend::{Flex, TensorMetadata, ops::FloatTensorOps, tensor::FloatTensor},
    tensor::{DType, Shape, TensorData},
};
use burn_flex::{FlexTensor, Layout};

fn decode(matrix: MatrixPrimitive<Flex>) -> FloatTensor<Flex> {
    // This implementation is selected only on an explicitly requested CPU device.
    let device = matrix.words.device();
    let data = matrix.words.to_contiguous().data_arc();
    if matrix.format == super::MatrixFormat::F32 {
        // I32 and F32 use the same four-byte storage. An immutable view keeps
        // the validated weight allocation shared across every projection.
        return FlexTensor::from_arc(data, Layout::contiguous(Shape::new(matrix.shape)), DType::F32);
    }
    let count = matrix.format.byte_len(matrix.shape).unwrap();
    let values = matrix.format.decode(matrix.shape, &data[..count]).unwrap();
    Flex::float_from_data(TensorData::new(values, matrix.shape), &device)
}

impl QwenOps for Flex {
    fn greedy(input: FloatTensor<Self>) -> core::result::Result<Option<u32>, burn::backend::ExecutionError> {
        let input = input.to_contiguous();
        if let Some(values) = input.as_slice::<f32>() {
            select(values.iter().copied().map(f64::from))
        } else {
            // Keep every supported CPU float dtype's precision. Model logits
            // use the borrowed F32 slice above without a copy or conversion.
            select(input.into_data().iter::<f64>())
        }
    }
    fn attention(input: super::ops::AttentionPrimitive<Self>, config: super::AttentionConfig) -> super::ops::AttentionResult<Self> {
        super::cpu_state::attention(input, config)
    }
    fn delta(input: super::ops::DeltaPrimitive<Self>, config: super::DeltaConfig) -> super::ops::DeltaResult<Self> {
        super::cpu_state::delta(input, config)
    }

    fn project(input: FloatTensor<Self>, matrix: MatrixPrimitive<Self>) -> FloatTensor<Self> {
        Self::float_matmul(input, Self::float_swap_dims(decode(matrix), 0, 1))
    }
    fn embedding(matrix: MatrixPrimitive<Self>, tokens: Vec<u32>) -> FloatTensor<Self> {
        let device = matrix.words.device();
        let width = matrix.shape[1];
        let rows = tokens.len();
        let row_bytes = matrix.format.byte_len([1, width]).unwrap();
        let data = matrix.words.to_contiguous().data_arc();
        let mut output = Vec::with_capacity(rows * width);
        for token in tokens {
            let start = token as usize * row_bytes;
            // Decode only selected rows, never the complete vocabulary table.
            output.extend(matrix.format.decode([1, width], &data[start..start + row_bytes]).unwrap());
        }
        Self::float_from_data(TensorData::new(output, [rows, width]), &device)
    }
}

fn select(values: impl Iterator<Item = f64>) -> core::result::Result<Option<u32>, burn::backend::ExecutionError> {
    let mut best = None;
    for (index, value) in values.enumerate() {
        if !value.is_finite() {
            return Ok(None);
        }
        if best.is_none_or(|(_, previous)| value > previous) {
            let index = u32::try_from(index).map_err(|_| burn::backend::ExecutionError::with_context("Qwen token index exceeds U32"))?;
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| Some(index))
        .ok_or_else(|| burn::backend::ExecutionError::with_context("Qwen logits are empty"))
}
