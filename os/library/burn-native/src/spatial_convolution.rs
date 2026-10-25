//! Tensor Core spatial convolutions through resident windows and transpose folds.
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use alloc::vec::Vec;
use burn_backend::{
    DType, Shape,
    ops::{ConvOptions, ConvTransposeOptions},
};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, elementwise::Elementwise, spatial_convolution as abi};
#[derive(Clone, Debug)]
pub struct SpatialConvolution {
    geometry: [u32; abi::GEOMETRY],
    image: Shape,
    grid: Shape,
    weights: Shape,
}
fn canonical(shape: &Shape) -> Result<[u32; 5]> {
    Layout::contiguous(shape.clone())?;
    let dims = match shape.as_slice() {
        [n, c, h, w] => [*n, *c, 1, *h, *w],
        [n, c, w] => [*n, *c, 1, 1, *w],
        [n, c, d, h, w] => [*n, *c, *d, *h, *w],
        _ => return Err(Error::ShapeMismatch),
    };
    let mut result = [0; 5];
    for (r, v) in result.iter_mut().zip(dims) {
        *r = u32::try_from(v).map_err(|_| Error::SizeOverflow)?;
    }
    Ok(result)
}
fn shape(dims: [u32; 5], rank: usize) -> Shape {
    Shape::from(
        dims.into_iter()
            .enumerate()
            .filter_map(|(i, v)| (i < 2 || i >= 7 - rank).then_some(v as usize))
            .collect::<Vec<_>>(),
    )
}
fn dense(t: &NativeTensor) -> Result<NativeTensor> {
    if t.layout.is_contiguous() { Ok(t.clone()) } else { t.copy() }
}
impl SpatialConvolution {
    pub fn conv3d(input: Shape, weights: Shape, options: &ConvOptions<3>) -> Result<Self> {
        if input.num_dims() != 5 || weights.num_dims() != 5 {
            return Err(Error::ShapeMismatch);
        }
        let image = canonical(&input)?;
        let filter = canonical(&weights)?;
        let mut p = [0u32; abi::GEOMETRY];
        p[..5].copy_from_slice(&image);
        p[5] = filter[0];
        p[6..9].copy_from_slice(&filter[2..]);
        p[21] = u32::try_from(options.groups).map_err(|_| Error::SizeOverflow)?;
        if p[21] == 0 || p[21] >= abi::TRANSPOSED || image[1] / p[21] != filter[1] {
            return Err(Error::ShapeMismatch);
        }
        for axis in 0..3 {
            for (at, v) in [
                (12, options.stride[axis]),
                (15, options.padding[axis].0),
                (18, options.dilation[axis]),
                (22, options.padding[axis].1),
            ] {
                p[at + axis] = u32::try_from(v).map_err(|_| Error::SizeOverflow)?;
            }
            p[9 + axis] =
                abi::output_extent(p[2 + axis], p[6 + axis], p[12 + axis], p[15 + axis], p[22 + axis], p[18 + axis]).map_err(|_| Error::InvalidParameter)?;
        }
        abi::geometry(&p).map_err(|_| Error::InvalidParameter)?;
        let grid = shape([p[0], p[5], p[9], p[10], p[11]], 5);
        Layout::contiguous(grid.clone())?;
        Ok(Self {
            geometry: p,
            image: input,
            grid,
            weights,
        })
    }
    pub fn transpose<const D: usize>(input: Shape, weights: Shape, options: &ConvTransposeOptions<D>) -> Result<Self> {
        if !(1..=3).contains(&D) || input.num_dims() != D + 2 || weights.num_dims() != D + 2 {
            return Err(Error::ShapeMismatch);
        }
        let grid = canonical(&input)?;
        let filter = canonical(&weights)?;
        let groups = u32::try_from(options.groups).map_err(|_| Error::SizeOverflow)?;
        if groups == 0 || groups >= abi::TRANSPOSED || grid[1] != filter[0] {
            return Err(Error::ShapeMismatch);
        }
        let mut p = [0; abi::GEOMETRY];
        p[0] = grid[0];
        p[1] = filter[1].checked_mul(groups).ok_or(Error::SizeOverflow)?;
        p[5] = grid[1];
        p[6..9].copy_from_slice(&filter[2..]);
        p[9..12].copy_from_slice(&grid[2..]);
        p[21] = groups | abi::TRANSPOSED;
        for axis in 0..3 {
            let v = if axis < 3 - D {
                [1, 0, 1, 0]
            } else {
                let i = axis - (3 - D);
                [options.stride[i], options.padding[i], options.dilation[i], options.padding_out[i]]
            };
            for (at, v) in [12, 15, 18, 22].into_iter().zip(v) {
                p[at + axis] = u32::try_from(v).map_err(|_| Error::SizeOverflow)?;
            }
            p[2 + axis] =
                abi::transpose_extent(p[9 + axis], p[6 + axis], p[12 + axis], p[15 + axis], p[22 + axis], p[18 + axis]).map_err(|_| Error::InvalidParameter)?;
        }
        abi::geometry(&p).map_err(|_| Error::InvalidParameter)?;
        let image = shape([p[0], p[1], p[2], p[3], p[4]], D + 2);
        Layout::contiguous(image.clone())?;
        Ok(Self {
            geometry: p,
            image,
            grid: input,
            weights,
        })
    }
    pub fn transpose_input<const D: usize>(weights: Shape, gradient: Shape, options: &ConvTransposeOptions<D>) -> Result<Self> {
        if !(1..=3).contains(&D) || weights.num_dims() != D + 2 || gradient.num_dims() != D + 2 {
            return Err(Error::ShapeMismatch);
        }
        canonical(&weights)?;
        canonical(&gradient)?;
        let mut input = gradient.to_vec();
        input[1] = weights[0];
        for axis in 0..D {
            if weights[axis + 2] == 0 || options.stride[axis] == 0 {
                return Err(Error::InvalidParameter);
            }
            let v = (gradient[axis + 2] as i128) + 2 * (options.padding[axis] as i128)
                - (options.dilation[axis] as i128) * ((weights[axis + 2] - 1) as i128)
                - (options.padding_out[axis] as i128)
                - 1;
            if v < 0 || v % (options.stride[axis] as i128) != 0 {
                return Err(Error::ShapeMismatch);
            }
            input[axis + 2] = usize::try_from(v / (options.stride[axis] as i128) + 1).map_err(|_| Error::SizeOverflow)?;
        }
        let plan = Self::transpose(Shape::from(input), weights, options)?;
        if plan.image != gradient {
            return Err(Error::ShapeMismatch);
        }
        Ok(plan)
    }
    pub fn output_shape(&self) -> &Shape {
        if self.geometry[21] & abi::TRANSPOSED != 0 { &self.image } else { &self.grid }
    }
    pub fn input_shape(&self) -> &Shape {
        if self.geometry[21] & abi::TRANSPOSED != 0 { &self.grid } else { &self.image }
    }
    fn dimensions(&self) -> [usize; 5] {
        let p = self.geometry;
        let g = (p[21] & !abi::TRANSPOSED) as usize;
        [
            p[0] as usize,
            g,
            p[5] as usize / g,
            (p[1] as usize / g) * p[6] as usize * p[7] as usize * p[8] as usize,
            p[9] as usize * p[10] as usize * p[11] as usize,
        ]
    }
    pub fn windows_parameters(&self, layout: &Layout) -> Result<[usize; abi::PARAMETERS]> {
        if layout.shape() != &self.image {
            return Err(Error::ShapeMismatch);
        }
        let mut p = [0u32; abi::PARAMETERS];
        p[..abi::GEOMETRY].copy_from_slice(&self.geometry);
        p[25] = layout.storage_elements().max(1) as u32;
        p[26] = layout.offset() as u32;
        let strides = layout.strides();
        let leading = 5 - strides.len();
        for axis in 0..5 {
            let stride = if axis < 2 {
                strides[axis]
            } else if axis < 2 + leading {
                0
            } else {
                strides[axis - leading]
            };
            p[27 + axis] = stride as i32 as u32;
        }
        abi::validate_windows(&p).map_err(|_| Error::InvalidParameter)?;
        Ok(p.map(|v| v as usize))
    }
    pub fn fold_parameters(&self) -> [usize; abi::GEOMETRY] {
        self.geometry.map(|v| v as usize)
    }
    fn windows(&self, input: &NativeTensor) -> Result<NativeTensor> {
        let [n, g, _, k, s] = self.dimensions();
        let layout = Layout::contiguous(Shape::new([n, g, k, s]))?;
        if layout.is_empty() {
            return Ok(NativeTensor {
                buffer: input.buffer.clone(),
                layout,
                dtype: DType::F32,
            });
        }
        let p = self.windows_parameters(&input.layout)?;
        let buffer = Nvidia.produce_words(layout.len(), Kernel::SpatialWindows, &[(&input.buffer, 0)], &p)?;
        Ok(NativeTensor {
            buffer: Arc::new(buffer),
            layout,
            dtype: DType::F32,
        })
    }
    fn image_matrix(&self, input: &NativeTensor) -> Result<NativeTensor> {
        let [n, g, _, k, s] = self.dimensions();
        dense(&self.windows(input)?.permute(&[1, 2, 0, 3])?)?.reshape(Shape::new([g, k, n * s]))
    }
    fn grid_matrix(&self, input: &NativeTensor) -> Result<NativeTensor> {
        if input.layout.shape() != &self.grid {
            return Err(Error::ShapeMismatch);
        }
        let [n, g, o, _, s] = self.dimensions();
        dense(&dense(input)?.reshape(Shape::new([n, g, o, s]))?.permute(&[1, 2, 0, 3])?)?.reshape(Shape::new([g, o, n * s]))
    }
    fn weights_matrix(&self, weights: &NativeTensor) -> Result<NativeTensor> {
        if weights.layout.shape() != &self.weights {
            return Err(Error::ShapeMismatch);
        }
        let [_, g, o, k, _] = self.dimensions();
        dense(weights)?.reshape(Shape::new([g, o, k]))
    }
    fn grid_result(&self, matrix: NativeTensor) -> Result<NativeTensor> {
        let [n, g, o, _, s] = self.dimensions();
        dense(&matrix.reshape(Shape::new([g, o, n, s]))?.permute(&[2, 0, 1, 3])?)?.reshape(self.grid.clone())
    }
    fn fold(&self, matrix: NativeTensor) -> Result<NativeTensor> {
        let [n, g, _, k, s] = self.dimensions();
        let columns = dense(&matrix.reshape(Shape::new([g, k, n, s]))?.permute(&[2, 0, 1, 3])?)?;
        let layout = Layout::contiguous(self.image.clone())?;
        if layout.is_empty() {
            return Ok(NativeTensor {
                buffer: columns.buffer,
                layout,
                dtype: DType::F32,
            });
        }
        let buffer = Nvidia.produce_words(layout.len(), Kernel::SpatialFold, &[(&columns.buffer, 0)], &self.fold_parameters())?;
        Ok(NativeTensor {
            buffer: Arc::new(buffer),
            layout,
            dtype: DType::F32,
        })
    }
    fn forward(&self, input: &NativeTensor, weights: &NativeTensor) -> Result<NativeTensor> {
        if self.grid[0] == 0 {
            return Ok(NativeTensor {
                buffer: input.buffer.clone(),
                layout: Layout::contiguous(self.grid.clone())?,
                dtype: DType::F32,
            });
        }
        self.grid_result(self.weights_matrix(weights)?.matmul(&self.image_matrix(input)?)?)
    }
    fn adjoint(&self, input: &NativeTensor, weights: &NativeTensor) -> Result<NativeTensor> {
        if self.image[0] == 0 {
            return Ok(NativeTensor {
                buffer: input.buffer.clone(),
                layout: Layout::contiguous(self.image.clone())?,
                dtype: DType::F32,
            });
        }
        self.fold(self.weights_matrix(weights)?.permute(&[0, 2, 1])?.matmul(&self.grid_matrix(input)?)?)
    }
    fn weight_gradient(&self, image: &NativeTensor, grid: &NativeTensor) -> Result<NativeTensor> {
        self.grid_matrix(grid)?
            .matmul(&self.image_matrix(image)?.permute(&[0, 2, 1])?)?
            .reshape(self.weights.clone())
    }
    fn check(&self, input: &NativeTensor, weights: &NativeTensor, bias: Option<&NativeTensor>) -> Result<()> {
        if input.dtype != DType::F32 || weights.dtype != DType::F32 || bias.is_some_and(|b| b.dtype != DType::F32) {
            return Err(Error::InvalidParameter);
        }
        if input.layout.shape() != self.input_shape()
            || weights.layout.shape() != &self.weights
            || bias.is_some_and(|b| b.layout.shape().as_slice() != [self.output_shape()[1]])
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
    fn biased(&self, result: NativeTensor, bias: Option<&NativeTensor>) -> Result<NativeTensor> {
        if result.layout.is_empty() {
            return Ok(result);
        }
        if let Some(bias) = bias {
            let mut dims = alloc::vec![1;self.output_shape().num_dims()];
            dims[1] = self.output_shape()[1];
            result.binary(&dense(bias)?.reshape(Shape::from(dims))?, Elementwise::Add)
        } else {
            Ok(result)
        }
    }
}
impl NativeTensor {
    pub fn conv3d(&self, weights: &Self, bias: Option<&Self>, options: &ConvOptions<3>) -> Result<Self> {
        let plan = SpatialConvolution::conv3d(self.layout.shape().clone(), weights.layout.shape().clone(), options)?;
        plan.check(self, weights, bias)?;
        Nvidia.record_commands(|| plan.biased(plan.forward(self, weights)?, bias))
    }
    pub fn conv3d_input_backward(&self, weights: &Self, gradient: &Self, options: &ConvOptions<3>) -> Result<Self> {
        let plan = SpatialConvolution::conv3d(self.layout.shape().clone(), weights.layout.shape().clone(), options)?;
        plan.check(self, weights, None)?;
        if gradient.dtype != DType::F32 {
            return Err(Error::InvalidParameter);
        }
        if gradient.layout.shape() != &plan.grid {
            return Err(Error::ShapeMismatch);
        }
        Nvidia.record_commands(|| plan.adjoint(gradient, weights))
    }
    pub fn conv3d_weight_backward(&self, weights: &Self, gradient: &Self, options: &ConvOptions<3>) -> Result<Self> {
        let plan = SpatialConvolution::conv3d(self.layout.shape().clone(), weights.layout.shape().clone(), options)?;
        plan.check(self, weights, None)?;
        if gradient.dtype != DType::F32 {
            return Err(Error::InvalidParameter);
        }
        if gradient.layout.shape() != &plan.grid {
            return Err(Error::ShapeMismatch);
        }
        Nvidia.record_commands(|| plan.weight_gradient(self, gradient))
    }
    pub fn conv_transpose<const D: usize>(&self, weights: &Self, bias: Option<&Self>, options: &ConvTransposeOptions<D>) -> Result<Self> {
        let plan = SpatialConvolution::transpose(self.layout.shape().clone(), weights.layout.shape().clone(), options)?;
        plan.check(self, weights, bias)?;
        Nvidia.record_commands(|| plan.biased(plan.adjoint(self, weights)?, bias))
    }
    pub fn conv_transpose_input_backward<const D: usize>(&self, gradient: &Self, options: &ConvTransposeOptions<D>) -> Result<Self> {
        let plan = SpatialConvolution::transpose_input(self.layout.shape().clone(), gradient.layout.shape().clone(), options)?;
        if self.dtype != DType::F32 || gradient.dtype != DType::F32 {
            return Err(Error::InvalidParameter);
        }
        Nvidia.record_commands(|| plan.forward(gradient, self))
    }
    pub fn conv_transpose_weight_backward<const D: usize>(&self, weights: &Self, gradient: &Self, options: &ConvTransposeOptions<D>) -> Result<Self> {
        let plan = SpatialConvolution::transpose(self.layout.shape().clone(), weights.layout.shape().clone(), options)?;
        plan.check(self, weights, None)?;
        if gradient.dtype != DType::F32 || gradient.layout.shape() != &plan.image {
            return Err(Error::ShapeMismatch);
        }
        Nvidia.record_commands(|| plan.weight_gradient(gradient, self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transpose_gradient_recovers_input_with_large_extra_padding() {
        let input = Shape::new([2, 6, 3, 5, 7]);
        let weights = Shape::new([6, 2, 2, 3, 2]);
        let options = ConvTransposeOptions::new([1, 2, 1], [0, 1, 0], [3, 4, 2], [2, 1, 3], 2);
        let plan = SpatialConvolution::transpose(input.clone(), weights.clone(), &options).unwrap();
        assert_eq!(plan.output_shape().as_slice(), [2, 4, 8, 13, 12]);
        let recovered = SpatialConvolution::transpose_input(weights.clone(), plan.output_shape().clone(), &options).unwrap();
        assert_eq!(recovered.input_shape(), &input);
        let mut wrong = plan.output_shape().to_vec();
        wrong[3] += 1;
        assert!(SpatialConvolution::transpose_input(weights, Shape::from(wrong), &options).is_err());
    }
    #[test]
    fn one_dimensional_views_insert_zero_strides_for_both_singleton_axes() {
        let input = Shape::new([2, 4, 5]);
        let weights = Shape::new([4, 2, 3]);
        let options = ConvTransposeOptions::new([2], [1], [3], [2], 2);
        let plan = SpatialConvolution::transpose(input.clone(), weights.clone(), &options).unwrap();
        assert_eq!(plan.output_shape().as_slice(), [2, 4, 14]);
        let view = Layout::contiguous(plan.output_shape().clone()).unwrap().flip(&[0, 1, 2]).unwrap();
        let parameters = plan.windows_parameters(&view).unwrap().map(|v| v as u32);
        assert_eq!(&parameters[..5], &[2, 4, 1, 1, 14]);
        assert_eq!(parameters[26], 111);
        assert_eq!(parameters[27..].iter().map(|&v| v as i32).collect::<Vec<_>>(), [-56, -14, 0, 0, -1]);
        assert_eq!(
            SpatialConvolution::transpose_input(weights, plan.output_shape().clone(), &options)
                .unwrap()
                .input_shape(),
            &input
        );
    }
    #[test]
    fn malformed_public_shapes_are_rejected_before_inverse_shape_arithmetic() {
        let options = ConvTransposeOptions::new([1; 3], [0; 3], [0; 3], [usize::MAX; 3], 1);
        assert!(SpatialConvolution::transpose_input(Shape::new([1, 1, usize::MAX, 1, 1]), Shape::new([1, 1, 1, 1, 1]), &options).is_err());
        assert!(SpatialConvolution::transpose_input(Shape::new([1, 1, 1, 1, 1]), Shape::new([usize::MAX, 1, 1, 1, 1]), &options).is_err());
    }
}
