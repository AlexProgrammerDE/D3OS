#![cfg_attr(not(any(feature = "cpu", feature = "native")), allow(unused_variables))]
use super::{MatrixFormat, MatrixLayout};
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
    pub layout: MatrixLayout,
    pub prefill_quantization: super::PrefillQuantization,
    pub q4_vector_kernel: super::Q4VectorKernel,
    pub q3_vector_kernel: super::Q3VectorKernel,
}

#[cfg(feature = "native")]
fn native_options<B: Backend>(matrix: &MatrixPrimitive<B>) -> crate::native::backend::ProjectionOptions {
    crate::native::backend::ProjectionOptions {
        prefill: matrix.prefill_quantization.native(),
        q4_vector: matrix.q4_vector_kernel.native(),
        q3_vector: matrix.q3_vector_kernel.native(),
    }
}

#[derive(ExtensionType)]
pub struct AttentionPrimitive<B: Backend> {
    pub query_gate: FloatTensor<B>,
    pub key: FloatTensor<B>,
    pub value: FloatTensor<B>,
    pub query_norm: FloatTensor<B>,
    pub key_norm: FloatTensor<B>,
    pub keys: FloatTensor<B>,
    pub values: FloatTensor<B>,
    pub position: usize,
}
#[derive(ExtensionType)]
pub struct AttentionResult<B: Backend> {
    pub output: FloatTensor<B>,
    pub keys: FloatTensor<B>,
    pub values: FloatTensor<B>,
}
#[derive(ExtensionType)]
pub struct DeltaParameterPrimitive<B: Backend> {
    pub convolution: FloatTensor<B>,
    pub log_decay: FloatTensor<B>,
    pub time_bias: FloatTensor<B>,
    pub norm: FloatTensor<B>,
}
#[derive(ExtensionType)]
pub struct DeltaCachePrimitive<B: Backend> {
    pub convolution: FloatTensor<B>,
    pub recurrent: FloatTensor<B>,
}
#[derive(ExtensionType)]
pub struct DeltaPrimitive<B: Backend> {
    pub qkv: FloatTensor<B>,
    pub gate: FloatTensor<B>,
    pub beta: FloatTensor<B>,
    pub alpha: FloatTensor<B>,
    #[extension_type]
    pub parameters: DeltaParameterPrimitive<B>,
    pub convolution: FloatTensor<B>,
    pub recurrent: FloatTensor<B>,
}
#[derive(ExtensionType)]
pub struct DeltaResult<B: Backend> {
    pub output: FloatTensor<B>,
    pub convolution: FloatTensor<B>,
    pub recurrent: FloatTensor<B>,
}
#[derive(ExtensionType)]
pub struct DeltaWeightPrimitive<B: Backend> {
    #[extension_type]
    pub qkv: MatrixPrimitive<B>,
    #[extension_type]
    pub gate: MatrixPrimitive<B>,
    #[extension_type]
    pub beta: MatrixPrimitive<B>,
    #[extension_type]
    pub alpha: MatrixPrimitive<B>,
}

#[backend_extension(Native: cfg(feature = "native"), Flex: cfg(feature = "cpu"))]
pub trait QwenOps: Backend {
    fn greedy(input: FloatTensor<Self>) -> core::result::Result<Option<u32>, burn::backend::ExecutionError>;
    fn attention(#[extension_type] input: AttentionPrimitive<Self>, config: super::AttentionConfig) -> AttentionResult<Self>;
    fn delta(#[extension_type] input: DeltaPrimitive<Self>, config: super::DeltaConfig) -> DeltaResult<Self>;
    fn project_delta(
        input: FloatTensor<Self>, #[extension_type] weights: DeltaWeightPrimitive<Self>, #[extension_type] parameters: DeltaParameterPrimitive<Self>,
        #[extension_type] cache: DeltaCachePrimitive<Self>, config: super::DeltaConfig,
    ) -> DeltaResult<Self> {
        let qkv = Self::project(input.clone(), weights.qkv);
        let gate = Self::project(input.clone(), weights.gate);
        let beta = Self::project(input.clone(), weights.beta);
        let alpha = Self::project(input, weights.alpha);
        Self::delta(
            DeltaPrimitive {
                qkv,
                gate,
                beta,
                alpha,
                parameters,
                convolution: cache.convolution,
                recurrent: cache.recurrent,
            },
            config,
        )
    }

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
    fn greedy(input: FloatTensor<Self>) -> core::result::Result<Option<u32>, burn::backend::ExecutionError> {
        Self::qwen_greedy(input)
    }
    fn attention(input: AttentionPrimitive<Self>, config: super::AttentionConfig) -> AttentionResult<Self> {
        let [output, keys, values] = Self::qwen_attention(
            [input.query_gate, input.key, input.value, input.query_norm, input.key_norm],
            [input.keys, input.values],
            config.geometry(),
            input.position,
            config.rope_theta,
            config.epsilon,
        );
        AttentionResult { output, keys, values }
    }
    fn delta(input: DeltaPrimitive<Self>, config: super::DeltaConfig) -> DeltaResult<Self> {
        let parameters = input.parameters;
        let [output, convolution, recurrent] = Self::qwen_delta(
            [
                input.qkv,
                input.gate,
                input.beta,
                input.alpha,
                parameters.convolution,
                parameters.log_decay,
                parameters.time_bias,
                parameters.norm,
            ],
            [input.convolution, input.recurrent],
            config.geometry(),
            config.epsilon,
        );
        DeltaResult {
            output,
            convolution,
            recurrent,
        }
    }
    fn project_delta(
        input: FloatTensor<Self>, weights: DeltaWeightPrimitive<Self>, parameters: DeltaParameterPrimitive<Self>, cache: DeltaCachePrimitive<Self>,
        config: super::DeltaConfig,
    ) -> DeltaResult<Self> {
        let storage = [
            ((weights.qkv.format, weights.qkv.layout), native_options(&weights.qkv)),
            ((weights.gate.format, weights.gate.layout), native_options(&weights.gate)),
            ((weights.beta.format, weights.beta.layout), native_options(&weights.beta)),
            ((weights.alpha.format, weights.alpha.layout), native_options(&weights.alpha)),
        ];
        let [output, convolution, recurrent] = Self::qwen_project_delta(
            input,
            [weights.qkv.words, weights.gate.words, weights.beta.words, weights.alpha.words],
            [parameters.convolution, parameters.log_decay, parameters.time_bias, parameters.norm],
            [cache.convolution, cache.recurrent],
            config.geometry(),
            config.epsilon,
            storage,
        );
        DeltaResult {
            output,
            convolution,
            recurrent,
        }
    }

    fn project(input: FloatTensor<Self>, matrix: MatrixPrimitive<Self>) -> FloatTensor<Self> {
        let options = native_options(&matrix);
        Self::qwen_project(input, matrix.words, matrix.shape, (matrix.format, matrix.layout), options)
    }
    fn embedding(matrix: MatrixPrimitive<Self>, tokens: Vec<u32>) -> FloatTensor<Self> {
        Self::qwen_embedding(matrix.words, matrix.shape, (matrix.format, matrix.layout), tokens)
    }
    fn gated_project(input: FloatTensor<Self>, gate: MatrixPrimitive<Self>, up: MatrixPrimitive<Self>) -> FloatTensor<Self> {
        let options = [native_options(&gate), native_options(&up)];
        Self::qwen_gated_project(
            input,
            gate.words,
            up.words,
            gate.shape,
            (gate.format, gate.layout),
            (up.format, up.layout),
            options,
        )
    }
    fn project_add(input: FloatTensor<Self>, matrix: MatrixPrimitive<Self>, residual: FloatTensor<Self>) -> FloatTensor<Self> {
        let options = native_options(&matrix);
        Self::qwen_project_add(input, matrix.words, residual, matrix.shape, (matrix.format, matrix.layout), options)
    }
    fn rms_norm(input: FloatTensor<Self>, weight: FloatTensor<Self>, epsilon: f32) -> FloatTensor<Self> {
        Self::qwen_rms_norm(input, weight, epsilon)
    }
    fn residual_norm(input: FloatTensor<Self>, residual: FloatTensor<Self>, weight: FloatTensor<Self>, epsilon: f32) -> (FloatTensor<Self>, FloatTensor<Self>) {
        Self::qwen_residual_norm(input, residual, weight, epsilon)
    }
}
