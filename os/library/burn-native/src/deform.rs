//! Resident deformable convolution, grouped Tensor Core products and gradients.
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{DType, Shape, ops::DeformConvOptions};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, deform as abi, elementwise::Elementwise, reduction::Reduction};
#[derive(Clone, Debug)]
pub struct DeformConv2d {
    parameters: [u32; abi::PARAMETERS],
    input: Shape,
    offset: Shape,
    weights: Shape,
    output: Shape,
    mask: Option<Shape>,
}
pub struct DeformGradients {
    pub input: NativeTensor,
    pub offset: NativeTensor,
    pub weights: NativeTensor,
    pub mask: Option<NativeTensor>,
    pub bias: Option<NativeTensor>,
}
fn dense(t: &NativeTensor) -> Result<NativeTensor> {
    if t.layout.is_contiguous() { Ok(t.clone()) } else { t.copy() }
}
impl DeformConv2d {
    pub fn new(input: Shape, offset: Shape, weights: Shape, mask: Option<&Shape>, bias: Option<&Shape>, options: &DeformConvOptions<2>) -> Result<Self> {
        for shape in [&input, &offset, &weights] {
            Layout::contiguous(shape.clone())?;
        }
        let ([n, c, h, w], [o, ci, kh, kw]) = (input.as_slice(), weights.as_slice()) else {
            return Err(Error::ShapeMismatch);
        };
        if options.weight_groups == 0 || c.checked_div(options.weight_groups) != Some(*ci) || bias.is_some_and(|b| b.as_slice() != [*o]) {
            return Err(Error::ShapeMismatch);
        }
        let mut p = [0; abi::PARAMETERS];
        let values = [
            *n,
            *c,
            *h,
            *w,
            *o,
            *kh,
            *kw,
            1,
            1,
            options.stride[0],
            options.stride[1],
            options.padding[0],
            options.padding[1],
            options.dilation[0],
            options.dilation[1],
            options.weight_groups,
            options.offset_groups,
            usize::from(mask.is_some()),
        ];
        for (p, v) in p.iter_mut().zip(values) {
            *p = u32::try_from(v).map_err(|_| Error::SizeOverflow)?;
        }
        p[7] = d3os_nvidia::compute::spatial_convolution::output_extent(p[2], p[5], p[9], p[11], p[11], p[13]).map_err(|_| Error::InvalidParameter)?;
        p[8] = d3os_nvidia::compute::spatial_convolution::output_extent(p[3], p[6], p[10], p[12], p[12], p[14]).map_err(|_| Error::InvalidParameter)?;
        abi::geometry(&p).map_err(|_| Error::InvalidParameter)?;
        let mask_shape = Shape::new([*n, p[16] as usize * kh * kw, p[7] as usize, p[8] as usize]);
        let offset_shape = Shape::new([*n, 2 * mask_shape[1], p[7] as usize, p[8] as usize]);
        let output = Shape::new([*n, *o, p[7] as usize, p[8] as usize]);
        if offset != offset_shape || mask.is_some_and(|m| m != &mask_shape) {
            return Err(Error::ShapeMismatch);
        }
        if let Some(mask) = mask {
            Layout::contiguous(mask.clone())?;
        }
        Layout::contiguous(output.clone())?;
        Ok(Self {
            parameters: p,
            input,
            offset,
            weights,
            output,
            mask: mask.cloned(),
        })
    }
    pub fn output_shape(&self) -> &Shape {
        &self.output
    }
    pub fn parameters(&self) -> [usize; abi::PARAMETERS] {
        self.parameters.map(|p| p as usize)
    }
    fn dimensions(&self) -> [usize; 5] {
        let p = self.parameters;
        [
            p[0] as usize,
            p[15] as usize,
            p[4] as usize / p[15] as usize,
            p[1] as usize / p[15] as usize * p[5] as usize * p[6] as usize,
            p[7] as usize * p[8] as usize,
        ]
    }
    fn validate(
        &self, input: &NativeTensor, offset: &NativeTensor, weights: &NativeTensor, mask: Option<&NativeTensor>, bias: Option<&NativeTensor>,
        gradient: Option<&NativeTensor>,
    ) -> Result<()> {
        if [input, offset, weights]
            .into_iter()
            .chain(mask)
            .chain(bias)
            .chain(gradient)
            .any(|t| t.dtype != DType::F32)
        {
            return Err(Error::InvalidParameter);
        }
        if input.layout.shape() != &self.input
            || offset.layout.shape() != &self.offset
            || weights.layout.shape() != &self.weights
            || mask.map(|m| m.layout.shape()) != self.mask.as_ref()
            || bias.is_some_and(|b| b.layout.shape().as_slice() != [self.output[1]])
            || gradient.is_some_and(|g| g.layout.shape() != &self.output)
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
    fn columns(&self, input: &NativeTensor, offset: &NativeTensor, mask: Option<&NativeTensor>) -> Result<NativeTensor> {
        let [n, g, _, k, s] = self.dimensions();
        let layout = Layout::contiguous(Shape::new([g, k, n * s]))?;
        if layout.is_empty() {
            return Ok(NativeTensor {
                buffer: input.buffer.clone(),
                layout,
                dtype: DType::F32,
            });
        }
        let optional = mask.unwrap_or(offset);
        let buffer = Nvidia.produce_words(
            layout.len(),
            Kernel::DeformWindows,
            &[
                (&input.buffer, input.layout.offset()),
                (&offset.buffer, offset.layout.offset()),
                (&optional.buffer, optional.layout.offset()),
            ],
            &self.parameters(),
        )?;
        Ok(NativeTensor {
            buffer: Arc::new(buffer),
            layout,
            dtype: DType::F32,
        })
    }
    fn weights_matrix(&self, weights: &NativeTensor) -> Result<NativeTensor> {
        let [_, g, o, k, _] = self.dimensions();
        dense(weights)?.reshape(Shape::new([g, o, k]))
    }
    fn gradient_matrix(&self, gradient: &NativeTensor) -> Result<NativeTensor> {
        let [n, g, o, _, s] = self.dimensions();
        dense(&dense(gradient)?.reshape(Shape::new([n, g * o, s]))?.permute(&[1, 0, 2])?)?.reshape(Shape::new([g, o, n * s]))
    }
    fn forward(
        &self, input: &NativeTensor, offset: &NativeTensor, weights: &NativeTensor, mask: Option<&NativeTensor>, bias: Option<&NativeTensor>,
    ) -> Result<NativeTensor> {
        let layout = Layout::contiguous(self.output.clone())?;
        if layout.is_empty() {
            return Ok(NativeTensor {
                buffer: input.buffer.clone(),
                layout,
                dtype: DType::F32,
            });
        }
        let input = dense(input)?;
        let offset = dense(offset)?;
        let mask = mask.map(dense).transpose()?;
        let [n, g, o, _, s] = self.dimensions();
        let matrix = self.weights_matrix(weights)?.matmul(&self.columns(&input, &offset, mask.as_ref())?)?;
        let result = dense(&matrix.reshape(Shape::new([g * o, n, s]))?.permute(&[1, 0, 2])?)?.reshape(self.output.clone())?;
        if let Some(bias) = bias {
            result.binary(&dense(bias)?.reshape(Shape::new([1, g * o, 1, 1]))?, Elementwise::Add)
        } else {
            Ok(result)
        }
    }
    fn backward(
        &self, input: &NativeTensor, offset: &NativeTensor, weights: &NativeTensor, mask: Option<&NativeTensor>, gradient: &NativeTensor, bias: bool,
    ) -> Result<DeformGradients> {
        if self.input[0] == 0 {
            return Ok(DeformGradients {
                input: input.clone(),
                offset: offset.clone(),
                weights: NativeTensor::empty(self.weights.clone(), DType::F32)?,
                mask: mask.cloned(),
                bias: if bias {
                    Some(NativeTensor::empty(Shape::new([self.output[1]]), DType::F32)?)
                } else {
                    None
                },
            });
        }
        let input = dense(input)?;
        let offset = dense(offset)?;
        let mask = mask.map(dense).transpose()?;
        let dy = self.gradient_matrix(gradient)?;
        let columns = self.weights_matrix(weights)?.permute(&[0, 2, 1])?.matmul(&dy)?;
        let dw = dy
            .matmul(&self.columns(&input, &offset, mask.as_ref())?.permute(&[0, 2, 1])?)?
            .reshape(self.weights.clone())?;
        let dx = NativeTensor::empty(self.input.clone(), DType::F32)?;
        let (doffset, dmask) = {
            let p = self.parameters();
            let geometry = abi::geometry(&self.parameters).map_err(|_| Error::InvalidParameter)?;
            let optional = mask.as_ref().unwrap_or(&offset);
            Nvidia.dispatch_words(
                Kernel::DeformInput,
                &[
                    (&offset.buffer, offset.layout.offset()),
                    (&optional.buffer, optional.layout.offset()),
                    (&columns.buffer, columns.layout.offset()),
                    (&dx.buffer, 0),
                ],
                &p,
            )?;
            let buffer = Arc::new(Nvidia.produce_words(
                geometry.coordinates as usize,
                Kernel::DeformCoordinates,
                &[
                    (&input.buffer, input.layout.offset()),
                    (&offset.buffer, offset.layout.offset()),
                    (&optional.buffer, optional.layout.offset()),
                    (&columns.buffer, columns.layout.offset()),
                ],
                &p,
            )?);
            let view = |shape: Shape, at: usize| -> Result<NativeTensor> {
                let contiguous = Layout::contiguous(shape.clone())?;
                let layout = Layout::from_parts(shape, contiguous.strides().to_vec(), at as i64, geometry.coordinates as usize)?;
                Ok(NativeTensor {
                    buffer: buffer.clone(),
                    layout,
                    dtype: DType::F32,
                })
            };
            let doffset = view(self.offset.clone(), 0)?;
            let dmask = self.mask.as_ref().map(|shape| view(shape.clone(), geometry.offsets as usize)).transpose()?;
            (doffset, dmask)
        };
        let db = if bias {
            Some(
                gradient
                    .reduce(0, Reduction::Sum)?
                    .reduce(2, Reduction::Sum)?
                    .reduce(3, Reduction::Sum)?
                    .reshape(Shape::new([self.output[1]]))?,
            )
        } else {
            None
        };
        Ok(DeformGradients {
            input: dx,
            offset: doffset,
            weights: dw,
            mask: dmask,
            bias: db,
        })
    }
}
impl NativeTensor {
    pub fn deform_conv2d(&self, offset: &Self, weights: &Self, mask: Option<&Self>, bias: Option<&Self>, options: &DeformConvOptions<2>) -> Result<Self> {
        let plan = DeformConv2d::new(
            self.layout.shape().clone(),
            offset.layout.shape().clone(),
            weights.layout.shape().clone(),
            mask.map(|m| m.layout.shape()),
            bias.map(|b| b.layout.shape()),
            options,
        )?;
        plan.validate(self, offset, weights, mask, bias, None)?;
        Nvidia.record_commands(|| plan.forward(self, offset, weights, mask, bias))
    }
    pub fn deform_conv2d_backward(
        &self, offset: &Self, weights: &Self, mask: Option<&Self>, bias: Option<&Self>, gradient: &Self, options: &DeformConvOptions<2>,
    ) -> Result<DeformGradients> {
        let plan = DeformConv2d::new(
            self.layout.shape().clone(),
            offset.layout.shape().clone(),
            weights.layout.shape().clone(),
            mask.map(|m| m.layout.shape()),
            bias.map(|b| b.layout.shape()),
            options,
        )?;
        plan.validate(self, offset, weights, mask, bias, Some(gradient))?;
        Nvidia.record_commands(|| plan.backward(self, offset, weights, mask, gradient, bias.is_some()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_groups_require_matching_offset_and_mask_channels() {
        let input = Shape::new([2, 6, 5, 7]);
        let weights = Shape::new([4, 3, 2, 3]);
        let offset = Shape::new([2, 36, 4, 7]);
        let mask = Shape::new([2, 18, 4, 7]);
        let options = DeformConvOptions::new([1, 1], [0, 1], [1, 1], 2, 3);
        let plan = DeformConv2d::new(input.clone(), offset.clone(), weights.clone(), Some(&mask), Some(&Shape::new([4])), &options).unwrap();
        assert_eq!(plan.output_shape().as_slice(), [2, 4, 4, 7]);
        assert_eq!(&plan.parameters()[15..], [2, 3, 1]);
        for invalid_offset in [Shape::new([2, 24, 4, 7]), Shape::new([1, 36, 4, 7]), Shape::new([2, 36, 4, 8])] {
            assert!(DeformConv2d::new(input.clone(), invalid_offset, weights.clone(), Some(&mask), None, &options).is_err());
        }
        assert!(DeformConv2d::new(input.clone(), offset.clone(), weights.clone(), Some(&Shape::new([2, 19, 4, 7])), None, &options).is_err());
        assert!(DeformConv2d::new(input, offset, weights, Some(&mask), Some(&Shape::new([3])), &options).is_err());
    }
    #[test]
    fn malformed_geometry_never_reaches_unchecked_shape_products() {
        let options = DeformConvOptions::new([1, 1], [usize::MAX, 0], [1, 1], 1, 1);
        assert!(
            DeformConv2d::new(
                Shape::new([1, 1, 2, 2]),
                Shape::new([1, 2, 2, 2]),
                Shape::new([1, 1, 1, 1]),
                None,
                None,
                &options
            )
            .is_err()
        );
        let options = DeformConvOptions::new([1, 1], [0, 0], [1, 1], 1, 1);
        assert!(
            DeformConv2d::new(
                Shape::new([1, 1, usize::MAX, 2]),
                Shape::new([1, 2, 2, 2]),
                Shape::new([1, 1, 1, 1]),
                None,
                None,
                &options
            )
            .is_err()
        );
    }
}
