use super::*;
use burn_backend::{
    IntDType,
    ops::*,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
impl ModuleOps<Self> for Native {
    fn linear(x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>) -> FloatTensor<Self> {
        let mut refs = alloc::vec![&x, &weight];
        if let Some(bias) = &bias {
            refs.push(bias);
        }
        let shape = crate::linear::output_shape(&x.shape(), &weight.shape(), bias.as_ref().map(|b| b.shape()).as_ref()).unwrap_or_else(|_| x.shape());
        BackendTensor::apply("linear", shape, x.dtype(), &refs, |v| v[0].linear(v[1], v.get(2).copied()))
    }
    fn linear_x_backward(weight: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        if crate::mixed_matmul::supported(grad.dtype()) {
            let shape = crate::linear::output_shape(&grad.shape(), &Shape::from(weight.shape().iter().rev().copied().collect::<Vec<_>>()), None)
                .unwrap_or_else(|_| grad.shape());
            return BackendTensor::apply("linear_x_backward", shape, grad.dtype(), &[&weight, &grad], |v| {
                let weight = v[0].with_layout(v[0].layout.swap_dims(0, 1)?)?;
                v[1].linear(&weight, None)
            });
        }
        default_linear_x_backward(weight, grad)
    }
    fn linear_weight_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        if crate::mixed_matmul::supported(x.dtype()) {
            let shape = crate::linear::weight_gradient_shape(&x.shape(), &grad.shape())
                .unwrap_or_else(|_| Shape::new([x.shape().last().copied().unwrap_or(0), grad.shape().last().copied().unwrap_or(0)]));
            return BackendTensor::apply("linear_weight_backward", shape, x.dtype(), &[&x, &grad], |v| {
                if v[0].dtype != v[1].dtype {
                    return Err(Error::InvalidParameter);
                }
                let shape = crate::linear::weight_gradient_shape(v[0].layout.shape(), v[1].layout.shape())?;
                let flatten = |tensor: &NativeTensor| {
                    let shape = crate::linear::flattened_shape(tensor.layout.shape())?;
                    let tensor = if tensor.layout.is_contiguous() { tensor.clone() } else { tensor.copy()? };
                    tensor.reshape(shape)
                };
                crate::mixed_matmul::validate_tensor(v[0])?;
                crate::mixed_matmul::validate_tensor(v[1])?;
                let left_layout = crate::linear::flattened_layout(v[0])?.swap_dims(0, 1)?;
                let right_layout = crate::linear::flattened_layout(v[1])?;
                let width = crate::storage::width(v[0].dtype)?;
                let words = |layout: &Layout| (width.words(layout.storage_elements() as u32) as usize).max(1);
                crate::mixed_matmul::validate_product(&left_layout, v[0].dtype, words(&left_layout), &right_layout, v[1].dtype, words(&right_layout))?;
                if shape.as_slice().contains(&0) {
                    return Ok(NativeTensor {
                        buffer: v[0].buffer.clone(),
                        layout: Layout::contiguous(shape)?,
                        dtype: v[0].dtype,
                    });
                }
                Nvidia.record_commands(|| {
                    let x = flatten(v[0])?;
                    let x = x.with_layout(x.layout.swap_dims(0, 1)?)?;
                    x.matmul(&flatten(v[1])?)
                })
            });
        }
        let x = Self::float_swap_dims(default_flatten_leading(x), 0, 1);
        let grad = default_flatten_leading(grad);
        Self::float_matmul(x, grad)
    }
    fn linear_bias_backward(grad: FloatTensor<Self>) -> FloatTensor<Self> {
        if crate::mixed_matmul::supported(grad.dtype()) {
            let shape = Shape::new([grad.shape().last().copied().unwrap_or(0)]);
            return BackendTensor::apply("linear_bias_backward", shape.clone(), grad.dtype(), &[&grad], |v| {
                let flat = crate::linear::flattened_shape(v[0].layout.shape())?;
                crate::mixed_matmul::validate_tensor(v[0])?;
                if shape[0] == 0 {
                    return Ok(NativeTensor {
                        buffer: v[0].buffer.clone(),
                        layout: Layout::contiguous(shape)?,
                        dtype: v[0].dtype,
                    });
                }
                Nvidia.record_commands(|| {
                    let grad = if v[0].layout.is_contiguous() { v[0].clone() } else { v[0].copy()? };
                    grad.reshape(flat)?.reduce(0, Reduction::Sum)?.reshape(shape)
                })
            });
        }
        let shape = grad.shape();
        let outputs = shape[shape.num_dims() - 1];
        let grad = Self::float_sum_dim(default_flatten_leading(grad), 0);
        Self::float_reshape(grad, Shape::new([outputs]))
    }
    fn conv2d(x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvOptions<2>) -> FloatTensor<Self> {
        let mut refs = alloc::vec![&x, &weight];
        if let Some(bias) = &bias {
            refs.push(bias);
        }
        let shape = crate::convolution::Conv2d::new(x.shape(), weight.shape(), None, &options).map_or_else(|_| x.shape(), |p| p.output_shape().clone());
        BackendTensor::apply("conv2d", shape, DType::F32, &refs, |v| v[0].conv2d(v[1], v.get(2).copied(), &options))
    }
    fn deform_conv2d(
        x: FloatTensor<Self>, offset: FloatTensor<Self>, weight: FloatTensor<Self>, mask: Option<FloatTensor<Self>>, bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        let mut refs = alloc::vec![&x, &offset, &weight];
        refs.extend(mask.iter());
        refs.extend(bias.iter());
        let shape = crate::deform::DeformConv2d::new(
            x.shape(),
            offset.shape(),
            weight.shape(),
            mask.as_ref().map(|m| m.shape()).as_ref(),
            bias.as_ref().map(|b| b.shape()).as_ref(),
            &options,
        )
        .map_or_else(|_| x.shape(), |p| p.output_shape().clone());
        BackendTensor::apply("deform_conv2d", shape, DType::F32, &refs, |v| {
            v[0].deform_conv2d(
                v[1],
                v[2],
                mask.as_ref().map(|_| v[3]),
                bias.as_ref().map(|_| v[3 + usize::from(mask.is_some())]),
                &options,
            )
        })
    }
    fn deform_conv2d_backward(
        x: FloatTensor<Self>, offset: FloatTensor<Self>, weight: FloatTensor<Self>, mask: Option<FloatTensor<Self>>, bias: Option<FloatTensor<Self>>,
        grad: FloatTensor<Self>, options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let mut refs = alloc::vec![&x, &offset, &weight, &grad];
        refs.extend(mask.iter());
        refs.extend(bias.iter());
        let mut shapes = alloc::vec![x.shape(), offset.shape(), weight.shape()];
        shapes.extend(mask.iter().map(|m| m.shape()));
        shapes.extend(bias.iter().map(|b| b.shape()));
        let run = |v: &[&NativeTensor]| {
            let result = v[0].deform_conv2d_backward(
                v[1],
                v[2],
                mask.as_ref().map(|_| v[4]),
                bias.as_ref().map(|_| v[4 + usize::from(mask.is_some())]),
                v[3],
                &options,
            )?;
            let mut outputs = alloc::vec![result.input, result.offset, result.weights];
            outputs.extend(result.mask);
            outputs.extend(result.bias);
            Ok::<_, Error>(outputs)
        };
        let outputs = match shapes.len() {
            3 => BackendTensor::outputs(
                "deform_conv2d_backward",
                core::array::from_fn::<_, 3, _>(|i| shapes[i].clone()),
                [DType::F32; 3],
                &refs,
                |v| run(v)?.try_into().map_err(|_| Error::ShapeMismatch),
            )
            .to_vec(),
            4 => BackendTensor::outputs(
                "deform_conv2d_backward",
                core::array::from_fn::<_, 4, _>(|i| shapes[i].clone()),
                [DType::F32; 4],
                &refs,
                |v| run(v)?.try_into().map_err(|_| Error::ShapeMismatch),
            )
            .to_vec(),
            _ => BackendTensor::outputs(
                "deform_conv2d_backward",
                core::array::from_fn::<_, 5, _>(|i| shapes[i].clone()),
                [DType::F32; 5],
                &refs,
                |v| run(v)?.try_into().map_err(|_| Error::ShapeMismatch),
            )
            .to_vec(),
        };
        let mut outputs = outputs.into_iter();
        DeformConv2dBackward {
            x_grad: outputs.next().unwrap(),
            offset_grad: outputs.next().unwrap(),
            weight_grad: outputs.next().unwrap(),
            mask_grad: mask.map(|_| outputs.next().unwrap()),
            bias_grad: bias.map(|_| outputs.next().unwrap()),
        }
    }
    fn conv3d(x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvOptions<3>) -> FloatTensor<Self> {
        let mut refs = alloc::vec![&x, &weight];
        if let Some(bias) = &bias {
            refs.push(bias);
        }
        let shape = crate::spatial_convolution::SpatialConvolution::conv3d(x.shape(), weight.shape(), &options)
            .map_or_else(|_| x.shape(), |p| p.output_shape().clone());
        BackendTensor::apply("conv3d", shape, DType::F32, &refs, |v| v[0].conv3d(v[1], v.get(2).copied(), &options))
    }
    fn conv3d_x_backward(x: FloatTensor<Self>, weight: FloatTensor<Self>, grad: FloatTensor<Self>, options: ConvOptions<3>) -> FloatTensor<Self> {
        BackendTensor::apply("conv3d_x_backward", x.shape(), DType::F32, &[&x, &weight, &grad], |v| {
            v[0].conv3d_input_backward(v[1], v[2], &options)
        })
    }
    fn conv3d_weight_backward(x: FloatTensor<Self>, weight: FloatTensor<Self>, grad: FloatTensor<Self>, options: ConvOptions<3>) -> FloatTensor<Self> {
        BackendTensor::apply("conv3d_weight_backward", weight.shape(), DType::F32, &[&x, &weight, &grad], |v| {
            v[0].conv3d_weight_backward(v[1], v[2], &options)
        })
    }
    fn conv_transpose1d(
        x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvTransposeOptions<1>,
    ) -> FloatTensor<Self> {
        let mut refs = alloc::vec![&x, &weight];
        if let Some(bias) = &bias {
            refs.push(bias);
        }
        let shape = crate::spatial_convolution::SpatialConvolution::transpose(x.shape(), weight.shape(), &options)
            .map_or_else(|_| x.shape(), |p| p.output_shape().clone());
        BackendTensor::apply("conv_transpose1d", shape, DType::F32, &refs, |v| {
            v[0].conv_transpose(v[1], v.get(2).copied(), &options)
        })
    }
    fn conv_transpose1d_x_backward(weight: FloatTensor<Self>, grad: FloatTensor<Self>, options: ConvTransposeOptions<1>) -> FloatTensor<Self> {
        let shape = crate::spatial_convolution::SpatialConvolution::transpose_input(weight.shape(), grad.shape(), &options)
            .map_or_else(|_| grad.shape(), |p| p.input_shape().clone());
        BackendTensor::apply("conv_transpose1d_x_backward", shape, DType::F32, &[&weight, &grad], |v| {
            v[0].conv_transpose_input_backward(v[1], &options)
        })
    }
    fn conv_transpose1d_weight_backward(
        x: FloatTensor<Self>, weight: FloatTensor<Self>, grad: FloatTensor<Self>, options: ConvTransposeOptions<1>,
    ) -> FloatTensor<Self> {
        BackendTensor::apply("conv_transpose1d_weight_backward", weight.shape(), DType::F32, &[&x, &weight, &grad], |v| {
            v[0].conv_transpose_weight_backward(v[1], v[2], &options)
        })
    }
    fn conv_transpose2d(
        x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        let mut refs = alloc::vec![&x, &weight];
        if let Some(bias) = &bias {
            refs.push(bias);
        }
        let shape = crate::spatial_convolution::SpatialConvolution::transpose(x.shape(), weight.shape(), &options)
            .map_or_else(|_| x.shape(), |p| p.output_shape().clone());
        BackendTensor::apply("conv_transpose2d", shape, DType::F32, &refs, |v| {
            v[0].conv_transpose(v[1], v.get(2).copied(), &options)
        })
    }
    fn conv_transpose2d_x_backward(weight: FloatTensor<Self>, grad: FloatTensor<Self>, options: ConvTransposeOptions<2>) -> FloatTensor<Self> {
        let shape = crate::spatial_convolution::SpatialConvolution::transpose_input(weight.shape(), grad.shape(), &options)
            .map_or_else(|_| grad.shape(), |p| p.input_shape().clone());
        BackendTensor::apply("conv_transpose2d_x_backward", shape, DType::F32, &[&weight, &grad], |v| {
            v[0].conv_transpose_input_backward(v[1], &options)
        })
    }
    fn conv_transpose2d_weight_backward(
        x: FloatTensor<Self>, weight: FloatTensor<Self>, grad: FloatTensor<Self>, options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        BackendTensor::apply("conv_transpose2d_weight_backward", weight.shape(), DType::F32, &[&x, &weight, &grad], |v| {
            v[0].conv_transpose_weight_backward(v[1], v[2], &options)
        })
    }
    fn conv_transpose3d(
        x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        let mut refs = alloc::vec![&x, &weight];
        if let Some(bias) = &bias {
            refs.push(bias);
        }
        let shape = crate::spatial_convolution::SpatialConvolution::transpose(x.shape(), weight.shape(), &options)
            .map_or_else(|_| x.shape(), |p| p.output_shape().clone());
        BackendTensor::apply("conv_transpose3d", shape, DType::F32, &refs, |v| {
            v[0].conv_transpose(v[1], v.get(2).copied(), &options)
        })
    }
    fn conv_transpose3d_x_backward(weight: FloatTensor<Self>, grad: FloatTensor<Self>, options: ConvTransposeOptions<3>) -> FloatTensor<Self> {
        let shape = crate::spatial_convolution::SpatialConvolution::transpose_input(weight.shape(), grad.shape(), &options)
            .map_or_else(|_| grad.shape(), |p| p.input_shape().clone());
        BackendTensor::apply("conv_transpose3d_x_backward", shape, DType::F32, &[&weight, &grad], |v| {
            v[0].conv_transpose_input_backward(v[1], &options)
        })
    }
    fn conv_transpose3d_weight_backward(
        x: FloatTensor<Self>, weight: FloatTensor<Self>, grad: FloatTensor<Self>, options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        BackendTensor::apply("conv_transpose3d_weight_backward", weight.shape(), DType::F32, &[&x, &weight, &grad], |v| {
            v[0].conv_transpose_weight_backward(v[1], v[2], &options)
        })
    }
    fn avg_pool2d(
        x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], count_include_pad: bool, ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let options = AvgPoolOptions::new(kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_count_include_pad(count_include_pad)
            .with_ceil_mode(ceil_mode);
        let shape = crate::pooling::Pooling::average(x.shape(), &options).map_or_else(|_| x.shape(), |p| p.output_shape().clone());
        BackendTensor::apply("avg_pool2d", shape, DType::F32, &[&x], |v| v[0].avg_pool(&options))
    }
    fn avg_pool2d_backward(
        x: FloatTensor<Self>, grad: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let options = AvgPoolOptions::new(kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_count_include_pad(count_include_pad)
            .with_ceil_mode(ceil_mode);
        BackendTensor::apply("avg_pool2d_backward", x.shape(), DType::F32, &[&x, &grad], |v| {
            v[0].avg_pool_backward(v[1], &options)
        })
    }
    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        let shape = crate::pooling::Pooling::adaptive(x.shape(), &output_size).map_or_else(|_| x.shape(), |p| p.output_shape().clone());
        BackendTensor::apply("adaptive_avg_pool2d", shape, DType::F32, &[&x], |v| v[0].adaptive_avg_pool(output_size))
    }
    fn adaptive_avg_pool2d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::apply("adaptive_avg_pool2d_backward", x.shape(), DType::F32, &[&x, &grad], |v| {
            v[0].adaptive_avg_pool_backward(v[1])
        })
    }
    fn adaptive_avg_pool3d(x: FloatTensor<Self>, output_size: [usize; 3]) -> FloatTensor<Self> {
        let shape = crate::pooling::Pooling::adaptive(x.shape(), &output_size).map_or_else(|_| x.shape(), |p| p.output_shape().clone());
        BackendTensor::apply("adaptive_avg_pool3d", shape, DType::F32, &[&x], |v| v[0].adaptive_avg_pool(output_size))
    }
    fn adaptive_avg_pool3d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        BackendTensor::apply("adaptive_avg_pool3d_backward", x.shape(), DType::F32, &[&x, &grad], |v| {
            v[0].adaptive_avg_pool_backward(v[1])
        })
    }
    fn max_pool2d(
        x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], ceil_mode: bool,
    ) -> FloatTensor<Self> {
        Self::max_pool2d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode, IntDType::I32).output
    }
    fn max_pool2d_with_indices(
        x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], ceil_mode: bool, indices_dtype: IntDType,
    ) -> MaxPool2dWithIndices<Self> {
        let options = MaxPoolOptions::new(kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_dilation(dilation)
            .with_ceil_mode(ceil_mode);
        let shape = crate::pooling::Pooling::max(x.shape(), &options).map_or_else(|_| x.shape(), |p| p.output_shape().clone());
        let (output, indices) = BackendTensor::pair("max_pool2d", [shape.clone(), shape], [DType::F32, indices_dtype.into()], &[&x], |v| {
            if indices_dtype != IntDType::I32 {
                return Err(Error::InvalidParameter);
            }
            v[0].max_pool(&options)
        });
        MaxPool2dWithIndices { output, indices }
    }
    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], ceil_mode: bool,
        output_grad: FloatTensor<Self>, indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        let options = MaxPoolOptions::new(kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_dilation(dilation)
            .with_ceil_mode(ceil_mode);
        MaxPool2dBackward {
            x_grad: BackendTensor::apply("max_pool2d_backward", x.shape(), DType::F32, &[&x, &output_grad, &indices], |v| {
                v[0].max_pool_backward(v[1], v[2], &options)
            }),
        }
    }
    fn interpolate(x: FloatTensor<Self>, output_size: [usize; 2], options: InterpolateOptions) -> FloatTensor<Self> {
        let shape = crate::interpolate::Interpolation::new(x.shape(), output_size, &options).map_or_else(|_| x.shape(), |p| p.output_shape().clone());
        BackendTensor::apply("interpolate", shape, DType::F32, &[&x], |v| v[0].interpolate(output_size, &options))
    }
    fn interpolate_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>, output_size: [usize; 2], options: InterpolateOptions) -> FloatTensor<Self> {
        BackendTensor::apply("interpolate_backward", x.shape(), DType::F32, &[&x, &grad], |v| {
            v[0].interpolate_backward(v[1], output_size, &options)
        })
    }
    fn attention(
        query: FloatTensor<Self>, key: FloatTensor<Self>, value: FloatTensor<Self>, mask: Option<BoolTensor<Self>>, attn_bias: Option<FloatTensor<Self>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<Self> {
        burn_backend::ops::attention::attention_fallback::<Self>(query, key, value, mask, attn_bias, options)
    }
    fn conv2d_x_backward(x: FloatTensor<Self>, weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvOptions<2>) -> FloatTensor<Self> {
        BackendTensor::apply("conv2d_x_backward", x.shape(), DType::F32, &[&x, &weight, &output_grad], |v| {
            v[0].conv2d_input_backward(v[1], v[2], &options)
        })
    }
    fn conv2d_weight_backward(x: FloatTensor<Self>, weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvOptions<2>) -> FloatTensor<Self> {
        BackendTensor::apply("conv2d_weight_backward", weight.shape(), DType::F32, &[&x, &weight, &output_grad], |v| {
            v[0].conv2d_weight_backward(v[1], v[2], &options)
        })
    }
}

// Burn 0.22.1 keeps these default helpers crate-private. Preserve their F32
// operation sequence here so only the new dtypes use checked zero-width rows.
fn default_flatten_leading(tensor: BackendTensor) -> BackendTensor {
    let shape = tensor.shape();
    let rank = shape.num_dims();
    if rank == 2 {
        return tensor;
    }
    let width = shape[rank - 1];
    Native::float_reshape(tensor, Shape::new([shape.num_elements() / width, width]))
}

fn default_linear_x_backward(weight: BackendTensor, grad: BackendTensor) -> BackendTensor {
    let weight = Native::float_swap_dims(weight, 0, 1);
    let shape = grad.shape();
    let rank = shape.num_dims();
    // The pinned BetterTiling policy uses shape-only analysis for linear.
    // Weight is rank two and shared, so this is its exact merge predicate.
    let batches: usize = shape[..rank - 2].iter().product();
    let rows = shape[rank - 2];
    let outputs = weight.shape()[1];
    let merged = batches * rows;
    let merge = batches != 1 && !rows.is_multiple_of(32) && merged.min(outputs) * rows.max(outputs) > rows.min(outputs) * merged.max(outputs);
    if merge {
        let grad = Native::float_reshape(grad, Shape::new([merged, shape[rank - 1]]));
        let output = Native::float_matmul(grad, weight);
        let mut dimensions = shape.to_vec();
        dimensions[rank - 1] = outputs;
        Native::float_reshape(output, Shape::from(dimensions))
    } else {
        let weight = if rank > 2 {
            let mut dimensions = alloc::vec![1; rank - 2];
            dimensions.extend_from_slice(weight.shape().as_slice());
            Native::float_reshape(weight, Shape::from(dimensions))
        } else {
            weight
        };
        Native::float_matmul(grad, weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;

    #[test]
    fn typed_linear_failures_keep_destination_metadata_and_original_error() {
        for dtype in [DType::F16, DType::BF16, DType::Flex32] {
            for (input_shape, grad_shape) in [
                (Shape::new([0]), Shape::new([5])),
                (Shape::new([0, 3]), Shape::new([0, 5])),
                (Shape::new([2, 3, 0]), Shape::new([2, 3, 5])),
                (Shape::new([2, 3, 4]), Shape::new([2, 3, 0])),
            ] {
                let input = BackendTensor::build("input", input_shape.clone(), dtype, || Err(Error::InvalidParameter));
                let grad = BackendTensor::build("gradient", grad_shape.clone(), dtype, || Err(Error::ShapeMismatch));
                let weight_shape = crate::linear::weight_gradient_shape(&input_shape, &grad_shape).unwrap();
                let weight = BackendTensor::build("weight", weight_shape.clone(), dtype, || Err(Error::SizeOverflow));
                let bias = BackendTensor::build("bias", Shape::new([grad_shape[grad_shape.num_dims() - 1]]), dtype, || {
                    Err(Error::InvalidParameter)
                });
                for tensor in [&input, &grad, &weight, &bias] {
                    tensor.failure().unwrap().report();
                }
                let output = Native::linear(input.clone(), weight.clone(), Some(bias));
                let dx = Native::linear_x_backward(weight.clone(), grad.clone());
                let dw = Native::linear_weight_backward(input.clone(), grad.clone());
                let db = Native::linear_bias_backward(grad.clone());
                assert_eq!(output.shape(), grad_shape);
                assert_eq!(dx.shape(), input_shape);
                assert_eq!(dw.shape(), weight_shape);
                assert_eq!(db.shape(), Shape::new([grad_shape[grad_shape.num_dims() - 1]]));
                for value in [&output, &dx, &dw, &db] {
                    assert_eq!(value.dtype(), dtype);
                }
                assert!(Arc::ptr_eq(&output.failure().unwrap(), &input.failure().unwrap()));
                assert!(Arc::ptr_eq(&dx.failure().unwrap(), &weight.failure().unwrap()));
                assert!(Arc::ptr_eq(&dw.failure().unwrap(), &input.failure().unwrap()));
                assert!(Arc::ptr_eq(&db.failure().unwrap(), &grad.failure().unwrap()));
                assert!(!output.read().unwrap_err().is_device_poisoned());
                assert!(!grad.read().unwrap_err().is_device_poisoned());
                assert!(!weight.read().unwrap_err().is_device_poisoned());
            }
        }
    }
}
