use super::ops::{MatrixPrimitive, QwenOps};
use alloc::vec::Vec;
use burn::{
    backend::{
        Flex, TensorMetadata,
        ops::{FloatTensorOps, IntTensorOps},
        tensor::FloatTensor,
    },
    tensor::{Shape, TensorData},
};

fn decode(matrix: MatrixPrimitive<Flex>) -> FloatTensor<Flex> {
    // This implementation is selected only on an explicitly requested CPU device.
    let device = matrix.words.device();
    let data = burn::backend::read_sync(Flex::int_into_data(matrix.words)).unwrap();
    let count = matrix.format.byte_len(matrix.shape).unwrap();
    let bytes = data.as_bytes();
    let values = matrix.format.decode(matrix.shape, &bytes[..count]).unwrap();
    Flex::float_from_data(TensorData::new(values, matrix.shape), &device)
}

impl QwenOps for Flex {
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
        let width = matrix.shape[1];
        let rows = tokens.len();
        let table = decode(matrix);
        let mut output = Vec::with_capacity(rows);
        for token in tokens {
            output.push(Self::float_slice(
                table.clone(),
                &[
                    burn::tensor::Slice::new(token as isize, Some(token as isize + 1), 1),
                    burn::tensor::Slice::new(0, Some(width as isize), 1),
                ],
            ));
        }
        Self::float_reshape(Self::float_cat(output, 0), Shape::new([rows, width]))
    }
}
