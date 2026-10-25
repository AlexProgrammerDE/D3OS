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
        BackendTensor::apply("linear", shape, DType::F32, &refs, |v| v[0].linear(v[1], v.get(2).copied()))
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
