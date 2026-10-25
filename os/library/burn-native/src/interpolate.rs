//! Separable native interpolation and deterministic gather-based transposes.
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{
    DType, Shape,
    ops::{InterpolateMode, InterpolateOptions},
};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{
    Kernel,
    interpolate::{self as abi, Mode},
};
#[derive(Clone, Debug)]
pub struct Interpolation {
    input: Shape,
    output: Shape,
    mode: Mode,
    align: bool,
}
impl Interpolation {
    pub fn new(input: Shape, size: [usize; 2], options: &InterpolateOptions) -> Result<Self> {
        if input.num_dims() != 4 || input[2..].contains(&0) || size.contains(&0) {
            return Err(Error::ShapeMismatch);
        }
        Layout::contiguous(input.clone())?;
        let output = Shape::new([input[0], input[1], size[0], size[1]]);
        Layout::contiguous(output.clone())?;
        let mode = match options.mode {
            InterpolateMode::Nearest => Mode::Nearest,
            InterpolateMode::NearestExact => Mode::NearestExact,
            InterpolateMode::Bilinear => Mode::Bilinear,
            InterpolateMode::Bicubic => Mode::Bicubic,
            InterpolateMode::Lanczos3 => Mode::Lanczos3,
        };
        for axis in 2..4 {
            abi::transform(input[axis] as u32, output[axis] as u32, mode, options.align_corners).map_err(|_| Error::InvalidParameter)?;
        }
        Ok(Self {
            input,
            output,
            mode,
            align: options.align_corners,
        })
    }
    pub fn output_shape(&self) -> &Shape {
        &self.output
    }
    pub fn parameters(&self, source: &Layout, axis: usize, backward: bool) -> Result<[usize; abi::PARAMETERS]> {
        if !(2..=3).contains(&axis) {
            return Err(Error::InvalidParameter);
        }
        let original = if axis == 3 {
            self.input.clone()
        } else {
            Shape::new([self.input[0], self.input[1], self.input[2], self.output[3]])
        };
        let target = self.output[axis];
        let mut resized = original.to_vec();
        resized[axis] = target;
        let expected = if backward { Shape::from(resized) } else { original.clone() };
        if source.shape() != &expected {
            return Err(Error::ShapeMismatch);
        }
        let mut p = [0u32; abi::PARAMETERS];
        p[..4].copy_from_slice(&[original[0] as u32, original[1] as u32, original[2] as u32, original[3] as u32]);
        p[4] = target as u32;
        p[5] = axis as u32;
        p[6] = self.mode as u32;
        p[7] = u32::from(self.align);
        p[8] = source.storage_elements().max(1) as u32;
        p[9] = source.offset() as u32;
        for (slot, &stride) in p[10..14].iter_mut().zip(source.strides()) {
            *slot = stride as i32 as u32;
        }
        p[14] = u32::from(backward);
        let transform = abi::transform(p[axis], p[4], self.mode, self.align).map_err(|_| Error::InvalidParameter)?;
        p[15..18].copy_from_slice(&transform);
        abi::validate(&p).map_err(|_| Error::InvalidParameter)?;
        Ok(p.map(|v| v as usize))
    }
}
impl NativeTensor {
    fn interpolate_axis(&self, plan: &Interpolation, axis: usize, backward: bool) -> Result<Self> {
        let parameters = plan.parameters(&self.layout, axis, backward)?;
        let mut shape = self.layout.shape().to_vec();
        shape[axis] = if backward { plan.input[axis] } else { plan.output[axis] };
        let layout = Layout::contiguous(Shape::from(shape))?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: self.dtype,
            });
        }
        if plan.input[axis] == plan.output[axis] {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout: self.layout.clone(),
                dtype: self.dtype,
            });
        }
        let buffer = Nvidia.produce_words(layout.len(), Kernel::Interpolate, &[(&self.buffer, 0)], &parameters)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype: DType::F32,
        })
    }
    pub fn interpolate(&self, size: [usize; 2], options: &InterpolateOptions) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::InvalidParameter);
        }
        let plan = Interpolation::new(self.layout.shape().clone(), size, options)?;
        Nvidia.record_commands(|| self.interpolate_axis(&plan, 3, false)?.interpolate_axis(&plan, 2, false))
    }
    pub fn interpolate_backward(&self, gradient: &Self, size: [usize; 2], options: &InterpolateOptions) -> Result<Self> {
        if self.dtype != DType::F32 || gradient.dtype != DType::F32 {
            return Err(Error::InvalidParameter);
        }
        let plan = Interpolation::new(self.layout.shape().clone(), size, options)?;
        if gradient.layout.shape() != &plan.output {
            return Err(Error::ShapeMismatch);
        }
        Nvidia.record_commands(|| gradient.interpolate_axis(&plan, 2, true)?.interpolate_axis(&plan, 3, true))
    }
}
