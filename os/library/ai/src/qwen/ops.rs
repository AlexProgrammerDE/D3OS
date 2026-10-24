#![cfg_attr(not(any(feature = "cpu", feature = "native")), allow(unused_variables))]
use super::MatrixFormat;
use alloc::vec::Vec;
use burn::backend::{
    Backend, ExtensionType, backend_extension,
    tensor::{FloatTensor, IntTensor},
};

#[derive(ExtensionType)]
pub struct MatrixPrimitive<B: Backend> {
    pub words: IntTensor<B>,
    pub shape: [usize; 2],
    pub format: MatrixFormat,
}

#[backend_extension(Native: cfg(feature = "native"), Flex: cfg(feature = "cpu"))]
pub trait QwenOps: Backend {
    fn project(input: FloatTensor<Self>, #[extension_type] matrix: MatrixPrimitive<Self>) -> FloatTensor<Self>;
    fn embedding(#[extension_type] matrix: MatrixPrimitive<Self>, tokens: Vec<u32>) -> FloatTensor<Self>;
    fn gated_project(
        input: FloatTensor<Self>, #[extension_type] gate: MatrixPrimitive<Self>, #[extension_type] up: MatrixPrimitive<Self>,
    ) -> FloatTensor<Self> {
        let gate = Self::project(input.clone(), gate);
        let up = Self::project(input, up);
        let sigmoid = Self::float_recip(Self::float_add_scalar(Self::float_exp(Self::float_neg(gate.clone())), 1f32.into()));
        Self::float_mul(Self::float_mul(gate, sigmoid), up)
    }
    fn project_add(input: FloatTensor<Self>, #[extension_type] matrix: MatrixPrimitive<Self>, residual: FloatTensor<Self>) -> FloatTensor<Self> {
        Self::float_add(Self::project(input, matrix), residual)
    }
    fn rms_norm(input: FloatTensor<Self>, weight: FloatTensor<Self>, epsilon: f32) -> FloatTensor<Self> {
        let square = Self::float_mul(input.clone(), input.clone());
        let inverse = Self::float_recip(Self::float_sqrt(Self::float_add_scalar(Self::float_mean_dim(square, 1), epsilon.into())));
        Self::float_mul(Self::float_mul(input, inverse), Self::float_add_scalar(weight, 1f32.into()))
    }
    fn residual_norm(input: FloatTensor<Self>, residual: FloatTensor<Self>, weight: FloatTensor<Self>, epsilon: f32) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let sum = Self::float_add(input, residual);
        let normalized = Self::rms_norm(sum.clone(), weight, epsilon);
        (sum, normalized)
    }
}

#[cfg(feature = "native")]
impl QwenOps for d3os_burn_native::backend::Native {
    fn project(input: FloatTensor<Self>, matrix: MatrixPrimitive<Self>) -> FloatTensor<Self> {
        Self::qwen_project(input, matrix.words, matrix.shape, matrix.format)
    }
    fn embedding(matrix: MatrixPrimitive<Self>, tokens: Vec<u32>) -> FloatTensor<Self> {
        Self::qwen_embedding(matrix.words, matrix.shape, matrix.format, tokens)
    }
    fn gated_project(input: FloatTensor<Self>, gate: MatrixPrimitive<Self>, up: MatrixPrimitive<Self>) -> FloatTensor<Self> {
        Self::qwen_gated_project(input, gate.words, up.words, gate.shape, gate.format, up.format)
    }
    fn project_add(input: FloatTensor<Self>, matrix: MatrixPrimitive<Self>, residual: FloatTensor<Self>) -> FloatTensor<Self> {
        Self::qwen_project_add(input, matrix.words, residual, matrix.shape, matrix.format)
    }
    fn rms_norm(input: FloatTensor<Self>, weight: FloatTensor<Self>, epsilon: f32) -> FloatTensor<Self> {
        Self::qwen_rms_norm(input, weight, epsilon)
    }
    fn residual_norm(input: FloatTensor<Self>, residual: FloatTensor<Self>, weight: FloatTensor<Self>, epsilon: f32) -> (FloatTensor<Self>, FloatTensor<Self>) {
        Self::qwen_residual_norm(input, residual, weight, epsilon)
    }
}
