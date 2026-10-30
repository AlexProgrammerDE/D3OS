//! Packed transformer operations over Burn-owned storage.
#[cfg(feature = "qwen-profiling")]
pub mod profile;
mod state;
use super::{BackendTensor, Native};
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{DType, Shape, TensorMetadata};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, elementwise::Elementwise};
use d3os_tensor_file::{
    matrix::{MatrixFormat, MatrixLayout},
    quantized::Quantization,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefillQuantization {
    #[default]
    Accurate,
    Q8_1,
    Q8_1WithAttention,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Q4VectorKernel {
    #[default]
    Default,
    Narrow,
    Components,
    Aligned,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Q3VectorKernel {
    #[default]
    Default,
    Components,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjectionOptions {
    pub prefill: PrefillQuantization,
    pub q4_vector: Q4VectorKernel,
    pub q3_vector: Q3VectorKernel,
}

fn dense(input: &NativeTensor, dtype: DType) -> Result<NativeTensor> {
    if input.dtype() != dtype {
        return Err(Error::InvalidParameter);
    }
    if input.layout.is_contiguous() { Ok(input.clone()) } else { input.copy() }
}

fn weights(input: &NativeTensor, shape: [usize; 2], storage: (MatrixFormat, MatrixLayout)) -> Result<NativeTensor> {
    let words = storage.1.byte_len(shape, storage.0)?.div_ceil(4);
    if input.shape().as_slice() != [words] {
        return Err(Error::ShapeMismatch);
    }
    dense(input, DType::I32)
}

fn output(shape: [usize; 2], kernel: Kernel, inputs: &[(&NativeTensor, usize)], parameters: &[usize]) -> Result<NativeTensor> {
    let layout = Layout::contiguous(Shape::new(shape))?;
    let pointers: alloc::vec::Vec<_> = inputs.iter().map(|(t, offset)| (&*t.buffer, t.layout.offset() + offset)).collect();
    let produce = || Nvidia.produce_words(layout.len(), kernel, &pointers, parameters);
    #[cfg(feature = "qwen-profiling")]
    let buffer = profile::measure(kernel, parameters, produce)?;
    #[cfg(not(feature = "qwen-profiling"))]
    let buffer = produce()?;
    Ok(NativeTensor {
        buffer: Arc::new(buffer),
        layout,
        dtype: DType::F32,
    })
}

fn use_components(batch: usize, shape: [usize; 2], format: MatrixFormat) -> bool {
    // Only the measured full-batch 9B FFN geometries use this route. Tails,
    // attention projections and single-token decode keep their existing path.
    batch == 32 && matches!(shape, [12288, 4096] | [4096, 12288]) && format == MatrixFormat::Quantized(Quantization::Q8_0)
}

fn projection_components(batch: usize, shape: [usize; 2], storage: (MatrixFormat, MatrixLayout), mode: PrefillQuantization) -> Result<Option<usize>> {
    let (format, layout) = storage;
    if mode == PrefillQuantization::Q8_1WithAttention && matches!(shape, [1024 | 4096 | 8192, 4096]) {
        if format != MatrixFormat::Quantized(Quantization::Q8_0) {
            return Err(Error::InvalidParameter);
        }
        return Ok(if batch == 32 {
            Some(1)
        } else if layout == MatrixLayout::Q8Columns32 && batch > 8 {
            Some(3)
        } else {
            None
        });
    }
    if mode != PrefillQuantization::Accurate {
        if layout != MatrixLayout::Q8Columns32 || format != MatrixFormat::Quantized(Quantization::Q8_0) || !matches!(shape, [12288, 4096] | [4096, 12288]) {
            return Err(Error::InvalidParameter);
        }
        if batch > 1 {
            return Ok(Some(1));
        }
    }
    Ok((use_components(batch, shape, format) || (layout == MatrixLayout::Q8Columns32 && batch > 8)).then_some(3))
}

fn use_q3_prefill(batch: usize, shape: [usize; 2], storage: (MatrixFormat, MatrixLayout), options: ProjectionOptions) -> bool {
    batch == 32
        && shape == [17408, 5120]
        && storage == (MatrixFormat::Quantized(Quantization::Q3K), MatrixLayout::RowMajor)
        && options.prefill == PrefillQuantization::Accurate
        && options.q3_vector == Q3VectorKernel::Components
}

fn prepare_activations(input: &NativeTensor, outputs: usize, components: usize) -> Result<NativeTensor> {
    let [batch, inner] = *input.shape().as_slice() else {
        return Err(Error::ShapeMismatch);
    };
    let elements = batch.checked_mul(inner).ok_or(Error::SizeOverflow)?;
    let words = match components {
        1 => elements.checked_mul(9).ok_or(Error::SizeOverflow)? / 32,
        3 => elements.checked_mul(3).ok_or(Error::SizeOverflow)? / 4 + elements / 32,
        _ => return Err(Error::InvalidParameter),
    };
    output([1, words], Kernel::ProjectQuantized, &[(input, 0)], &[batch, inner, outputs, 8, 0, components])
}

fn project_prepared(
    activations: &NativeTensor, packed: &NativeTensor, batch: usize, shape: [usize; 2], layout: MatrixLayout, components: usize,
) -> Result<NativeTensor> {
    output(
        [batch, shape[0]],
        Kernel::ProjectQuantized,
        &[(activations, 0), (packed, 0)],
        &[batch, shape[1], shape[0], 8, if layout == MatrixLayout::RowMajor { 1 } else { 2 }, components],
    )
}

fn projection(
    input: &NativeTensor, packed: &NativeTensor, shape: [usize; 2], storage: (MatrixFormat, MatrixLayout), options: ProjectionOptions,
) -> Result<NativeTensor> {
    let (format, layout) = storage;
    let [batch, width] = *input.shape().as_slice() else {
        return Err(Error::ShapeMismatch);
    };
    let [outputs, inner] = shape;
    if batch == 0 || width != inner {
        return Err(Error::ShapeMismatch);
    }
    let components = projection_components(batch, shape, storage, options.prefill)?;
    let input = dense(input, DType::F32)?;
    let packed = weights(packed, shape, storage)?;
    if layout == MatrixLayout::Q3Padded112 {
        if format != MatrixFormat::Quantized(Quantization::Q3K) || shape != [17408, 5120] {
            return Err(Error::InvalidParameter);
        }
        let components = usize::from(batch > 1 || options.q3_vector == Q3VectorKernel::Components) * 3;
        let activations = if components == 3 { prepare_activations(&input, outputs, 3)? } else { input };
        return output(
            [batch, outputs],
            Kernel::ProjectQuantized,
            &[(&activations, 0), (&packed, 0)],
            &[batch, width, outputs, 11, 3, components],
        );
    }
    if use_q3_prefill(batch, shape, storage, options) {
        let activations = prepare_activations(&input, outputs, 3)?;
        return output(
            [batch, outputs],
            Kernel::ProjectQuantized,
            &[(&activations, 0), (&packed, 0)],
            &[batch, width, outputs, 11, 2, 3],
        );
    }
    if batch == 1 && format == MatrixFormat::Quantized(Quantization::Q4K) && options.q4_vector == Q4VectorKernel::Narrow {
        return output(
            [1, outputs],
            Kernel::ProjectQuantized,
            &[(&input, 0), (&packed, 0)],
            &[1, width, outputs, 12, 1],
        );
    }
    let vector_components = match format {
        MatrixFormat::Quantized(Quantization::Q3K) if options.q3_vector == Q3VectorKernel::Components => Some(11),
        MatrixFormat::Quantized(Quantization::Q4K) if matches!(options.q4_vector, Q4VectorKernel::Components | Q4VectorKernel::Aligned) => Some(12),
        _ => None,
    };
    if let (1, Some(code)) = (batch, vector_components) {
        let activations = prepare_activations(&input, outputs, 3)?;
        let phase = if code == 12 && options.q4_vector == Q4VectorKernel::Aligned && matches!((width, outputs), (6144 | 17408, 5120)) {
            4
        } else {
            1
        };
        return output(
            [1, outputs],
            Kernel::ProjectQuantized,
            &[(&activations, 0), (&packed, 0)],
            &[1, width, outputs, code, phase, 3],
        );
    }
    if let Some(components) = components {
        let activations = prepare_activations(&input, outputs, components)?;
        return project_prepared(&activations, &packed, batch, shape, layout, components);
    }
    if layout == MatrixLayout::Q8Columns32 {
        return output(
            [batch, outputs],
            Kernel::ProjectQuantized,
            &[(&input, 0), (&packed, 0)],
            &[batch, width, outputs, 8, 2],
        );
    }
    if format == MatrixFormat::Bf16 && (2..=32).contains(&batch) && (256..=2048).contains(&outputs) && width >= 512 {
        let partials = output(
            [16 * batch, outputs],
            Kernel::ProjectBf16Split,
            &[(&input, 0), (&packed, 0)],
            &[batch, width, outputs, 16],
        )?;
        return output([batch, outputs], Kernel::SumProjectionPartials, &[(&partials, 0)], &[batch, outputs, 16]);
    }
    let (kernel, code) = match format {
        MatrixFormat::F32 => (Kernel::Project, 0),
        MatrixFormat::Bf16 => (Kernel::ProjectBf16, 0),
        MatrixFormat::Quantized(format) => (Kernel::ProjectQuantized, format as usize),
    };
    let parameters = [batch, width, outputs, code];
    output(
        [batch, outputs],
        kernel,
        &[(&input, 0), (&packed, 0)],
        &parameters[..if code == 0 { 3 } else { 4 }],
    )
}

impl Native {
    pub fn qwen_pack_q8_weights(input: BackendTensor, shape: [usize; 2]) -> BackendTensor {
        BackendTensor::apply("pack Q8 weights", input.shape(), DType::I32, &[&input], |v| {
            let input = weights(v[0], shape, (MatrixFormat::Quantized(Quantization::Q8_0), MatrixLayout::Q8Columns32))?;
            let words = input.layout.len();
            let buffer = Nvidia.produce_words(
                words,
                Kernel::ProjectQuantized,
                &[(&input.buffer, input.layout.offset())],
                &[shape[0], shape[1], 8, 3],
            )?;
            Ok(NativeTensor {
                buffer: Arc::new(buffer),
                layout: Layout::contiguous(Shape::new([words]))?,
                dtype: DType::I32,
            })
        })
    }

    pub fn qwen_greedy(input: BackendTensor) -> core::result::Result<Option<u32>, burn_backend::ExecutionError> {
        input.read_value("greedy selection", |input| {
            let [1, columns] = *input.shape().as_slice() else {
                return Err(Error::ShapeMismatch);
            };
            if columns == 0 {
                return Err(Error::ShapeMismatch);
            }
            Nvidia.record_commands(|| {
                let input = dense(input, DType::F32)?;
                match Nvidia.argmax_words(&input.buffer, input.layout.offset(), columns) {
                    Ok(index) => u32::try_from(index).map(Some).map_err(|_| Error::SizeOverflow),
                    Err(Error::NonFinite) => Ok(None),
                    Err(error) => Err(error),
                }
            })
        })
    }

    pub fn qwen_project(
        input: BackendTensor, packed: BackendTensor, shape: [usize; 2], storage: (MatrixFormat, MatrixLayout), options: ProjectionOptions,
    ) -> BackendTensor {
        let batch = input.shape().first().copied().unwrap_or(0);
        BackendTensor::apply("packed projection", Shape::new([batch, shape[0]]), DType::F32, &[&input, &packed], |v| {
            projection(v[0], v[1], shape, storage, options)
        })
    }

    pub fn qwen_embedding(packed: BackendTensor, shape: [usize; 2], storage: (MatrixFormat, MatrixLayout), tokens: alloc::vec::Vec<u32>) -> BackendTensor {
        let [vocab, width] = shape;
        BackendTensor::apply("packed embedding", Shape::new([tokens.len(), width]), DType::F32, &[&packed], |v| {
            if tokens.is_empty() || tokens.iter().any(|&id| id as usize >= vocab) {
                return Err(Error::InvalidParameter);
            }
            let packed = weights(v[0], shape, storage)?;
            let (format, layout) = storage;
            let ids = Nvidia.upload_words(&tokens)?;
            let ids = NativeTensor {
                buffer: Arc::new(ids),
                layout: Layout::contiguous(Shape::new([tokens.len()]))?,
                dtype: DType::I32,
            };
            let (kernel, code) = match format {
                MatrixFormat::Quantized(format) => (Kernel::EmbeddingQuantized, format as usize),
                _ => (Kernel::Embedding, usize::from(format == MatrixFormat::Bf16)),
            };
            let layout_code = match layout {
                MatrixLayout::RowMajor => 0,
                MatrixLayout::Q8Columns32 => 2,
                MatrixLayout::Q3Padded112 => 3,
            };
            let parameters = [vocab, width, tokens.len(), code, layout_code];
            output(
                [tokens.len(), width],
                kernel,
                &[(&packed, 0), (&ids, 0)],
                &parameters[..if layout == MatrixLayout::RowMajor { 4 } else { 5 }],
            )
        })
    }

    pub fn qwen_gated_project(
        input: BackendTensor, gate: BackendTensor, up: BackendTensor, shape: [usize; 2], gate_storage: (MatrixFormat, MatrixLayout),
        up_storage: (MatrixFormat, MatrixLayout), options: [ProjectionOptions; 2],
    ) -> BackendTensor {
        let batch = input.shape().first().copied().unwrap_or(0);
        let (gate_format, gate_layout) = gate_storage;
        let (up_format, up_layout) = up_storage;
        BackendTensor::apply(
            "gated packed projection",
            Shape::new([batch, shape[0]]),
            DType::F32,
            &[&input, &gate, &up],
            |v| {
                let gate_components = projection_components(batch, shape, gate_storage, options[0].prefill)?;
                let up_components = projection_components(batch, shape, up_storage, options[1].prefill)?;
                if batch == 1 && gate_format == MatrixFormat::Bf16 && up_format == MatrixFormat::Bf16 {
                    if v[0].shape().as_slice() != [1, shape[1]] {
                        return Err(Error::ShapeMismatch);
                    }
                    let input = dense(v[0], DType::F32)?;
                    let gate = weights(v[1], shape, gate_storage)?;
                    let up = weights(v[2], shape, up_storage)?;
                    return output(
                        [1, shape[0]],
                        Kernel::GatedProjectBf16,
                        &[(&input, 0), (&gate, 0), (&up, 0)],
                        &[shape[1], shape[0]],
                    );
                }
                let shared_components = if gate_format == up_format
                    && gate_components == up_components
                    && (use_components(batch, shape, gate_format) || options[0].prefill != PrefillQuantization::Accurate)
                {
                    gate_components
                } else {
                    None
                };
                let (gate, up) = if let Some(components) = shared_components {
                    if v[0].shape().as_slice() != [batch, shape[1]] {
                        return Err(Error::ShapeMismatch);
                    }
                    let input = dense(v[0], DType::F32)?;
                    let gate = weights(v[1], shape, gate_storage)?;
                    let up = weights(v[2], shape, up_storage)?;
                    let activations = prepare_activations(&input, shape[0], components)?;
                    (
                        project_prepared(&activations, &gate, batch, shape, gate_layout, components)?,
                        project_prepared(&activations, &up, batch, shape, up_layout, components)?,
                    )
                } else {
                    (
                        projection(v[0], v[1], shape, gate_storage, options[0])?,
                        projection(v[0], v[2], shape, up_storage, options[1])?,
                    )
                };
                Nvidia.dispatch_words(Kernel::SiluMul, &[(&gate.buffer, 0), (&up.buffer, 0), (&gate.buffer, 0)], &[batch * shape[0]])?;
                Ok(gate)
            },
        )
    }

    pub fn qwen_project_add(
        input: BackendTensor, packed: BackendTensor, residual: BackendTensor, shape: [usize; 2], storage: (MatrixFormat, MatrixLayout),
        options: ProjectionOptions,
    ) -> BackendTensor {
        let result_shape = residual.shape();
        // Check before creating any temporary storage aliases. In-place dispatch is
        // valid only when no other Burn tensor can observe the residual allocation.
        let unique = residual.can_mut();
        BackendTensor::apply("packed residual projection", result_shape, DType::F32, &[&input, &packed, &residual], |v| {
            let [batch, width] = *v[0].shape().as_slice() else {
                return Err(Error::ShapeMismatch);
            };
            if width != shape[1] || v[2].shape().as_slice() != [batch, shape[0]] {
                return Err(Error::ShapeMismatch);
            }
            if options.prefill != PrefillQuantization::Accurate {
                projection_components(batch, shape, storage, options.prefill)?;
            }
            let (format, _) = storage;
            if batch == 1 && width.is_multiple_of(2) && shape[0] <= 8192 && format == MatrixFormat::Bf16 {
                let input = dense(v[0], DType::F32)?;
                let packed = weights(v[1], shape, storage)?;
                let residual = if unique { dense(v[2], DType::F32)? } else { v[2].copy()? };
                if residual.dtype() != DType::F32 {
                    return Err(Error::InvalidParameter);
                }
                Nvidia.dispatch_words(
                    Kernel::ProjectResidualBf16,
                    &[
                        (&residual.buffer, residual.layout.offset()),
                        (&input.buffer, input.layout.offset()),
                        (&packed.buffer, packed.layout.offset()),
                        (&residual.buffer, residual.layout.offset()),
                    ],
                    &[width, shape[0]],
                )?;
                return Ok(residual);
            }
            projection(v[0], v[1], shape, storage, options)?.binary(v[2], Elementwise::Add)
        })
    }

    pub fn qwen_rms_norm(input: BackendTensor, weight: BackendTensor, epsilon: f32) -> BackendTensor {
        BackendTensor::apply("zero-centered RMS normalization", input.shape(), DType::F32, &[&input, &weight], |v| {
            let [rows, width] = *v[0].shape().as_slice() else {
                return Err(Error::ShapeMismatch);
            };
            if rows == 0 || width == 0 || v[1].shape().as_slice() != [width] || !epsilon.is_finite() || epsilon <= 0. {
                return Err(Error::InvalidParameter);
            }
            let input = dense(v[0], DType::F32)?;
            let weight = dense(v[1], DType::F32)?;
            output(
                [rows, width],
                Kernel::RmsNorm,
                &[(&input, 0), (&weight, 0)],
                &[rows, width, epsilon.to_bits() as usize, 1f32.to_bits() as usize],
            )
        })
    }

    pub fn qwen_residual_norm(input: BackendTensor, residual: BackendTensor, weight: BackendTensor, epsilon: f32) -> (BackendTensor, BackendTensor) {
        let shape = input.shape();
        let unique = input.can_mut();
        BackendTensor::pair(
            "residual RMS normalization",
            [shape.clone(), shape],
            [DType::F32; 2],
            &[&input, &residual, &weight],
            |v| {
                let [rows, width] = *v[0].shape().as_slice() else {
                    return Err(Error::ShapeMismatch);
                };
                if rows == 0
                    || width == 0
                    || v[1].shape() != v[0].shape()
                    || v[2].shape().as_slice() != [width]
                    || !epsilon.is_finite()
                    || epsilon <= 0.
                    || v.iter().any(|t| t.dtype() != DType::F32)
                {
                    return Err(Error::InvalidParameter);
                }
                let residual = dense(v[1], DType::F32)?;
                let weight = dense(v[2], DType::F32)?;
                if width <= 1024 || width == 4096 {
                    let sum = if unique { dense(v[0], DType::F32)? } else { v[0].copy()? };
                    let normalized = output(
                        [rows, width],
                        Kernel::ResidualNorm,
                        &[(&sum, 0), (&residual, 0), (&weight, 0)],
                        &[rows, width, epsilon.to_bits() as usize],
                    )?;
                    Ok((sum, normalized))
                } else {
                    let sum = v[0].binary(&residual, Elementwise::Add)?;
                    let normalized = output(
                        [rows, width],
                        Kernel::RmsNorm,
                        &[(&sum, 0), (&weight, 0)],
                        &[rows, width, epsilon.to_bits() as usize, 1f32.to_bits() as usize],
                    )?;
                    Ok((sum, normalized))
                }
            },
        )
    }
}

#[cfg(test)]
mod projection_mode_tests {
    use super::*;

    #[test]
    fn attention_quantization_preserves_small_batches_and_decode() {
        let storage = (MatrixFormat::Quantized(Quantization::Q8_0), MatrixLayout::RowMajor);
        for outputs in [1024, 4096, 8192] {
            for batch in [1, 2, 5, 17, 31, 32, 33] {
                assert_eq!(
                    projection_components(batch, [outputs, 4096], storage, PrefillQuantization::Q8_1WithAttention).unwrap(),
                    (batch == 32).then_some(1)
                );
                assert_eq!(
                    projection_components(batch, [outputs, 4096], storage, PrefillQuantization::Accurate).unwrap(),
                    None
                );
            }
        }
        for shape in [[32, 4096], [4096, 1024], [12288, 4096]] {
            assert_eq!(
                projection_components(32, shape, storage, PrefillQuantization::Q8_1WithAttention),
                Err(Error::InvalidParameter)
            );
        }
    }

    #[test]
    fn tiled_attention_keeps_decode_and_uses_compensation_for_long_tails() {
        let storage = (MatrixFormat::Quantized(Quantization::Q8_0), MatrixLayout::Q8Columns32);
        for columns in [4096, 8192] {
            for rows in [1, 2, 5, 8] {
                assert_eq!(
                    projection_components(rows, [columns, 4096], storage, PrefillQuantization::Q8_1WithAttention).unwrap(),
                    None
                );
            }
            for rows in [17, 31] {
                assert_eq!(
                    projection_components(rows, [columns, 4096], storage, PrefillQuantization::Q8_1WithAttention).unwrap(),
                    Some(3)
                );
            }
            assert_eq!(
                projection_components(32, [columns, 4096], storage, PrefillQuantization::Q8_1WithAttention).unwrap(),
                Some(1)
            );
        }
    }

    #[test]
    fn quantized_prefill_keeps_single_token_and_accurate_routes() {
        let storage = (MatrixFormat::Quantized(Quantization::Q8_0), MatrixLayout::Q8Columns32);
        for shape in [[12288, 4096], [4096, 12288]] {
            assert_eq!(projection_components(1, shape, storage, PrefillQuantization::Q8_1).unwrap(), None);
            for batch in [2, 5, 32] {
                assert_eq!(projection_components(batch, shape, storage, PrefillQuantization::Q8_1).unwrap(), Some(1));
            }
            assert_eq!(projection_components(5, shape, storage, PrefillQuantization::Accurate).unwrap(), None);
            assert_eq!(projection_components(32, shape, storage, PrefillQuantization::Accurate).unwrap(), Some(3));
        }
    }

    #[test]
    fn q3_prefill_uses_only_the_measured_packed_geometry() {
        let storage = (MatrixFormat::Quantized(Quantization::Q3K), MatrixLayout::RowMajor);
        let options = ProjectionOptions {
            q3_vector: Q3VectorKernel::Components,
            ..Default::default()
        };
        assert!(use_q3_prefill(32, [17408, 5120], storage, options));
        for batch in [0, 1, 2, 16, 31, 33] {
            assert!(!use_q3_prefill(batch, [17408, 5120], storage, options));
        }
        for shape in [[5120, 17408], [17407, 5120], [17408, 5119]] {
            assert!(!use_q3_prefill(32, shape, storage, options));
        }
        for other in [
            (MatrixFormat::Quantized(Quantization::Q4K), MatrixLayout::RowMajor),
            (MatrixFormat::Quantized(Quantization::Q3K), MatrixLayout::Q8Columns32),
        ] {
            assert!(!use_q3_prefill(32, [17408, 5120], other, options));
        }
        assert!(!use_q3_prefill(32, [17408, 5120], storage, ProjectionOptions::default()));
        assert!(!use_q3_prefill(
            32,
            [17408, 5120],
            storage,
            ProjectionOptions {
                prefill: PrefillQuantization::Q8_1,
                ..options
            }
        ));
    }

    #[test]
    fn quantized_prefill_rejects_unqualified_weight_encodings() {
        for storage in [
            (MatrixFormat::Bf16, MatrixLayout::RowMajor),
            (MatrixFormat::Quantized(Quantization::Q8_0), MatrixLayout::RowMajor),
        ] {
            assert_eq!(
                projection_components(32, [12288, 4096], storage, PrefillQuantization::Q8_1),
                Err(Error::InvalidParameter)
            );
        }
        assert_eq!(
            projection_components(
                32,
                [4096, 4096],
                (MatrixFormat::Quantized(Quantization::Q8_0), MatrixLayout::Q8Columns32),
                PrefillQuantization::Q8_1
            ),
            Err(Error::InvalidParameter)
        );
    }
}
