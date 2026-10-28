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

fn prepare_components(input: &NativeTensor, outputs: usize) -> Result<NativeTensor> {
    let [batch, inner] = *input.shape().as_slice() else {
        return Err(Error::ShapeMismatch);
    };
    let elements = batch.checked_mul(inner).ok_or(Error::SizeOverflow)?;
    let words = elements.checked_mul(3).ok_or(Error::SizeOverflow)? / 4 + elements / 32;
    output([1, words], Kernel::ProjectQuantized, &[(input, 0)], &[batch, inner, outputs, 8, 0, 3])
}

fn project_components(components: &NativeTensor, packed: &NativeTensor, batch: usize, shape: [usize; 2], layout: MatrixLayout) -> Result<NativeTensor> {
    output(
        [batch, shape[0]],
        Kernel::ProjectQuantized,
        &[(components, 0), (packed, 0)],
        &[batch, shape[1], shape[0], 8, if layout == MatrixLayout::RowMajor { 1 } else { 2 }, 3],
    )
}

fn projection(input: &NativeTensor, packed: &NativeTensor, shape: [usize; 2], storage: (MatrixFormat, MatrixLayout)) -> Result<NativeTensor> {
    let (format, layout) = storage;
    let [batch, width] = *input.shape().as_slice() else {
        return Err(Error::ShapeMismatch);
    };
    let [outputs, inner] = shape;
    if batch == 0 || width != inner {
        return Err(Error::ShapeMismatch);
    }
    let input = dense(input, DType::F32)?;
    let packed = weights(packed, shape, storage)?;
    if use_components(batch, shape, format) || (layout == MatrixLayout::Q8Columns32 && batch > 8) {
        let components = prepare_components(&input, outputs)?;
        return project_components(&components, &packed, batch, shape, layout);
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

    pub fn qwen_project(input: BackendTensor, packed: BackendTensor, shape: [usize; 2], storage: (MatrixFormat, MatrixLayout)) -> BackendTensor {
        let batch = input.shape().first().copied().unwrap_or(0);
        BackendTensor::apply("packed projection", Shape::new([batch, shape[0]]), DType::F32, &[&input, &packed], |v| {
            projection(v[0], v[1], shape, storage)
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
            let parameters = [vocab, width, tokens.len(), code, 2];
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
        up_storage: (MatrixFormat, MatrixLayout),
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
                let (gate, up) = if use_components(batch, shape, gate_format) && gate_format == up_format {
                    if v[0].shape().as_slice() != [batch, shape[1]] {
                        return Err(Error::ShapeMismatch);
                    }
                    let input = dense(v[0], DType::F32)?;
                    let gate = weights(v[1], shape, gate_storage)?;
                    let up = weights(v[2], shape, up_storage)?;
                    let components = prepare_components(&input, shape[0])?;
                    (
                        project_components(&components, &gate, batch, shape, gate_layout)?,
                        project_components(&components, &up, batch, shape, up_layout)?,
                    )
                } else {
                    (projection(v[0], v[1], shape, gate_storage)?, projection(v[0], v[2], shape, up_storage)?)
                };
                Nvidia.dispatch_words(Kernel::SiluMul, &[(&gate.buffer, 0), (&up.buffer, 0), (&gate.buffer, 0)], &[batch * shape[0]])?;
                Ok(gate)
            },
        )
    }

    pub fn qwen_project_add(
        input: BackendTensor, packed: BackendTensor, residual: BackendTensor, shape: [usize; 2], storage: (MatrixFormat, MatrixLayout),
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
            projection(v[0], v[1], shape, storage)?.binary(v[2], Elementwise::Add)
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
