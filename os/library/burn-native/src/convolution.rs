//! Native NCHW convolution plans, Tensor Core forward passes, and gradients.
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{DType, Shape, ops::ConvOptions};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, convolution as abi, reduction::Reduction};

#[derive(Clone, Debug)]
pub struct Conv2d {
    geometry: [u32; abi::GEOMETRY],
    input: Shape,
    weights: Shape,
    output: Shape,
}

impl Conv2d {
    pub fn new(input: Shape, weights: Shape, bias: Option<&Shape>, options: &ConvOptions<2>) -> Result<Self> {
        Layout::contiguous(input.clone())?;
        Layout::contiguous(weights.clone())?;
        let ([n, c, h, w], [o, ci, kh, kw]) = (input.as_slice(), weights.as_slice()) else {
            return Err(Error::ShapeMismatch);
        };
        if options.groups == 0
            || c.checked_div(options.groups) != Some(*ci)
            || !c.is_multiple_of(options.groups)
            || !o.is_multiple_of(options.groups)
            || bias.is_some_and(|b| b.as_slice() != [*o])
        {
            return Err(Error::ShapeMismatch);
        }
        let mut p = [0u32; abi::GEOMETRY];
        let values = [
            *n,
            *c,
            *h,
            *w,
            *kh,
            *kw,
            1,
            1,
            options.stride[0],
            options.stride[1],
            options.padding[0].0,
            options.padding[1].0,
            options.dilation[0],
            options.dilation[1],
            options.groups,
            options.padding[0].1,
            options.padding[1].1,
            *o,
        ];
        for (p, v) in p.iter_mut().zip(values) {
            *p = u32::try_from(v).map_err(|_| Error::SizeOverflow)?;
        }
        p[6] = abi::output_extent(p[2], p[4], p[8], p[10], p[15], p[12]).map_err(|_| Error::InvalidParameter)?;
        p[7] = abi::output_extent(p[3], p[5], p[9], p[11], p[16], p[13]).map_err(|_| Error::InvalidParameter)?;
        abi::geometry(&p).map_err(|_| Error::InvalidParameter)?;
        let output = Shape::new([*n, *o, p[6] as usize, p[7] as usize]);
        Ok(Self {
            geometry: p,
            input,
            weights,
            output,
        })
    }

    pub fn output_shape(&self) -> &Shape {
        &self.output
    }
    pub fn columns_shape(&self) -> Shape {
        let p = self.geometry;
        Shape::new([p[0] as usize, p[14] as usize, (p[1] / p[14] * p[4] * p[5]) as usize, (p[6] * p[7]) as usize])
    }

    fn explicit_columns(&self, input: &Layout, weights: &Layout) -> bool {
        let p = self.geometry;
        let inner = self.weights[1] * self.weights[2] * self.weights[3];
        let spatial = self.output[2] * self.output[3];
        // Large dense filters reuse the same image windows across output
        // channels. Bound the workspace so other convolutions keep the direct
        // path, including small MNIST filters and grouped or dilated inputs.
        p[0] == 1
            && p[14] == 1
            && p[12] == 1
            && p[13] == 1
            && p[10] == p[15]
            && p[11] == p[16]
            && inner >= 256
            && p[17] >= 32
            && spatial >= 32
            && inner.checked_mul(spatial).is_some_and(|words| words <= 2 * 1024 * 1024)
            && input.is_contiguous()
            && weights.is_contiguous()
    }

    pub fn windows_parameters(&self, input: &Layout) -> Result<[usize; abi::WINDOW_PARAMETERS]> {
        if input.shape() != &self.input {
            return Err(Error::ShapeMismatch);
        }
        let mut p = [0u32; abi::WINDOW_PARAMETERS];
        p[..abi::GEOMETRY].copy_from_slice(&self.geometry);
        p[18..24].copy_from_slice(&view(input));
        abi::validate_windows(&p).map_err(|_| Error::InvalidParameter)?;
        Ok(p.map(|v| v as usize))
    }

    pub fn parameters(&self, input: &Layout, weights: &Layout) -> Result<[usize; abi::PARAMETERS]> {
        if weights.shape() != &self.weights {
            return Err(Error::ShapeMismatch);
        }
        let mut p = [0u32; abi::PARAMETERS];
        p[..24].copy_from_slice(&self.windows_parameters(input)?.map(|v| v as u32));
        p[24..30].copy_from_slice(&view(weights));
        abi::validate(&p).map_err(|_| Error::InvalidParameter)?;
        Ok(p.map(|v| v as usize))
    }

    fn gradient_matrix(&self, gradient: &NativeTensor) -> Result<NativeTensor> {
        let p = self.geometry.map(|v| v as usize);
        let [n, g, _, s]: [usize; 4] = self.columns_shape().as_slice().try_into().unwrap();
        let spatial_batch = n.checked_mul(s).ok_or(Error::SizeOverflow)?;
        let view = dense(gradient)?.reshape(Shape::new([n, g, p[17] / g, s]))?.permute(&[1, 2, 0, 3])?;
        // Collapse samples and output positions into one GEMM dimension, so
        // a group needs one Tensor Core launch rather than one per sample.
        dense(&view)?.reshape(Shape::new([g, p[17] / g, spatial_batch]))
    }

    fn input_gradient(&self, input: &NativeTensor, weights: &NativeTensor, dy: &NativeTensor) -> Result<NativeTensor> {
        let layout = Layout::contiguous(self.input.clone())?;
        if layout.is_empty() {
            return Ok(NativeTensor { buffer: input.buffer.clone(), layout, dtype: DType::F32 });
        }
        let p = self.geometry.map(|v| v as usize);
        let [n, g, k, s]: [usize; 4] = self.columns_shape().as_slice().try_into().unwrap();
        let matrix = dense(weights)?.reshape(Shape::new([g, p[17] / g, k]))?.permute(&[0, 2, 1])?;
        let columns = matrix.matmul(dy)?.reshape(Shape::new([g, k, n, s]))?.permute(&[2, 0, 1, 3])?;
        let columns = dense(&columns)?;
        let buffer = Nvidia.produce_words(layout.len(), Kernel::Conv2dFold, &[(&columns.buffer, 0)], &p)?;
        Ok(NativeTensor { buffer: Arc::new(buffer), layout, dtype: DType::F32 })
    }

    fn weight_gradient(&self, input: &NativeTensor, dy: &NativeTensor) -> Result<NativeTensor> {
        let [n, g, k, s]: [usize; 4] = self.columns_shape().as_slice().try_into().unwrap();
        let spatial_batch = n.checked_mul(s).ok_or(Error::SizeOverflow)?;
        let layout = Layout::contiguous(self.columns_shape())?;
        let columns = if layout.is_empty() {
            NativeTensor { buffer: input.buffer.clone(), layout, dtype: DType::F32 }
        } else {
            let parameters = self.windows_parameters(&input.layout)?;
            let buffer = Nvidia.produce_words(layout.len(), Kernel::Conv2dWindows, &[(&input.buffer, 0)], &parameters)?;
            NativeTensor { buffer: Arc::new(buffer), layout, dtype: DType::F32 }
        };
        let view = columns.permute(&[1, 0, 3, 2])?;
        let matrix = dense(&view)?.reshape(Shape::new([g, spatial_batch, k]))?;
        dy.matmul(&matrix)?.reshape(self.weights.clone())
    }
}

fn view(layout: &Layout) -> [u32; 6] {
    [
        layout.storage_elements().max(1) as u32,
        layout.offset() as u32,
        layout.strides()[0] as i32 as u32,
        layout.strides()[1] as i32 as u32,
        layout.strides()[2] as i32 as u32,
        layout.strides()[3] as i32 as u32,
    ]
}

fn dense(tensor: &NativeTensor) -> Result<NativeTensor> {
    if tensor.layout.is_contiguous() { Ok(tensor.clone()) } else { tensor.copy() }
}

impl NativeTensor {
    pub fn conv2d(&self, weights: &Self, bias: Option<&Self>, options: &ConvOptions<2>) -> Result<Self> {
        if self.dtype != DType::F32 || weights.dtype != DType::F32 || bias.is_some_and(|b| b.dtype != DType::F32) {
            return Err(Error::InvalidParameter);
        }
        let plan = Conv2d::new(
            self.layout.shape().clone(),
            weights.layout.shape().clone(),
            bias.map(|b| b.layout.shape()),
            options,
        )?;
        let mut parameters = plan.parameters(&self.layout, &weights.layout)?;
        parameters[30] = usize::from(bias.is_some());
        let layout = Layout::contiguous(plan.output.clone())?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: DType::F32,
            });
        }
        Nvidia.record_commands(|| {
            let bias = bias.map(dense).transpose()?;
            if plan.explicit_columns(&self.layout, &weights.layout) {
                let p = plan.geometry.map(|v| v as usize);
                let inner = p[1] * p[4] * p[5];
                let spatial = p[6] * p[7];
                let window_parameters = [p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]];
                let columns = Nvidia.produce_words(inner * spatial, Kernel::Im2col, &[(&self.buffer, self.layout.offset())], &window_parameters)?;
                let buffer = Nvidia.produce_words(
                    layout.len(),
                    Kernel::Matmul,
                    &[(&weights.buffer, weights.layout.offset()), (&columns, 0)],
                    &[p[17], inner, spatial],
                )?;
                if let Some(bias) = bias {
                    Nvidia.dispatch_words(
                        Kernel::BiasChannel,
                        &[(&buffer, 0), (&bias.buffer, bias.layout.offset()), (&buffer, 0)],
                        &[layout.len(), spatial, p[17]],
                    )?;
                }
                return Ok(Self {
                    buffer: Arc::new(buffer),
                    layout,
                    dtype: DType::F32,
                });
            }
            let bias_buffer = bias.as_ref().map_or((self.buffer.as_ref(), 0), |b| (b.buffer.as_ref(), b.layout.offset()));
            let buffer = Nvidia.produce_words(
                layout.len(),
                Kernel::Conv2d,
                &[(&self.buffer, 0), (&weights.buffer, 0), bias_buffer],
                &parameters,
            )?;
            Ok(Self {
                buffer: Arc::new(buffer),
                layout,
                dtype: DType::F32,
            })
        })
    }

    fn backward_plan(&self, weights: &Self, gradient: &Self, options: &ConvOptions<2>) -> Result<Conv2d> {
        if [self.dtype, weights.dtype, gradient.dtype].iter().any(|&d| d != DType::F32) {
            return Err(Error::InvalidParameter);
        }
        let plan = Conv2d::new(self.layout.shape().clone(), weights.layout.shape().clone(), None, options)?;
        if gradient.layout.shape() != plan.output_shape() {
            return Err(Error::ShapeMismatch);
        }
        Ok(plan)
    }

    pub fn conv2d_input_backward(&self, weights: &Self, gradient: &Self, options: &ConvOptions<2>) -> Result<Self> {
        let plan = self.backward_plan(weights, gradient, options)?;
        Nvidia.record_commands(|| plan.input_gradient(self, weights, &plan.gradient_matrix(gradient)?))
    }

    pub fn conv2d_weight_backward(&self, weights: &Self, gradient: &Self, options: &ConvOptions<2>) -> Result<Self> {
        let plan = self.backward_plan(weights, gradient, options)?;
        Nvidia.record_commands(|| plan.weight_gradient(self, &plan.gradient_matrix(gradient)?))
    }

    /// Return input, weight, and optional bias gradients. Every intermediate
    /// stays on the GPU; fold sums overlapping windows without atomic updates.
    pub fn conv2d_backward(&self, weights: &Self, gradient: &Self, bias: bool, options: &ConvOptions<2>) -> Result<(Self, Self, Option<Self>)> {
        let plan = self.backward_plan(weights, gradient, options)?;
        Nvidia.record_commands(|| {
            let p = plan.geometry.map(|v| v as usize);
            let dy = plan.gradient_matrix(gradient)?;
            let input_gradient = plan.input_gradient(self, weights, &dy)?;
            let weight_gradient = plan.weight_gradient(self, &dy)?;
            let bias_gradient = if bias {
                Some(
                    gradient
                        .reduce(0, Reduction::Sum)?
                        .reduce(2, Reduction::Sum)?
                        .reduce(3, Reduction::Sum)?
                        .reshape(Shape::new([p[17]]))?,
                )
            } else {
                None
            };
            Ok((input_gradient, weight_gradient, bias_gradient))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_windows_require_reuse_dense_views_and_a_bounded_workspace() {
        let input = Layout::contiguous(Shape::new([1, 32, 7, 9])).unwrap();
        let weights = Layout::contiguous(Shape::new([33, 32, 3, 3])).unwrap();
        let options = ConvOptions::new([2, 1], [1, 2], [1, 1], 1);
        let plan = Conv2d::new(input.shape().clone(), weights.shape().clone(), None, &options).unwrap();
        assert!(plan.explicit_columns(&input, &weights));
        assert!(!plan.explicit_columns(&input.clone().flip(&[3]).unwrap(), &weights));
        assert!(!plan.explicit_columns(&input, &weights.clone().flip(&[0]).unwrap()));
        for (shape, filter, options) in [
            ([2, 32, 7, 9], [33, 32, 3, 3], options.clone()),
            ([1, 1, 7, 9], [5, 1, 3, 3], options.clone()),
            ([1, 32, 1024, 1024], [33, 32, 3, 3], ConvOptions::new([1, 1], [1, 1], [1, 1], 1)),
            ([1, 32, 7, 9], [33, 32, 3, 3], ConvOptions::new([2, 1], [1, 2], [2, 1], 1)),
            (
                [1, 32, 7, 9],
                [33, 32, 3, 3],
                ConvOptions::new_with_padding([2, 1], [(1, 2), (2, 2)], [1, 1], 1),
            ),
        ] {
            let input = Layout::contiguous(Shape::new(shape)).unwrap();
            let weights = Layout::contiguous(Shape::new(filter)).unwrap();
            let plan = Conv2d::new(input.shape().clone(), weights.shape().clone(), None, &options).unwrap();
            assert!(!plan.explicit_columns(&input, &weights));
        }
    }
}
