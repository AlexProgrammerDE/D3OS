//! Resident max, average, and adaptive pooling in one to three spatial dimensions.
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{
    DType, Shape,
    ops::{AvgPoolOptions, MaxPoolOptions},
};
use d3os_compute::{Backend, Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{
    Kernel,
    pooling::{self as abi, Mode},
    words::WordOp,
};

#[derive(Clone, Debug)]
pub struct Pooling {
    input: Shape,
    output: Shape,
    p: [u32; abi::GEOMETRY],
    mode: Mode,
}

impl Pooling {
    fn new(input: Shape, dimensions: usize, mode: Mode) -> Result<Self> {
        if !(1..=3).contains(&dimensions) || input.num_dims() != dimensions + 2 {
            return Err(Error::ShapeMismatch);
        }
        Layout::contiguous(input.clone())?;
        let mut p = [0u32; abi::GEOMETRY];
        p[0] = input[0] as u32;
        p[1] = input[1] as u32;
        p[2..14].fill(1);
        p[20..23].fill(1);
        for axis in 0..dimensions {
            p[5 - dimensions + axis] = input[2 + axis] as u32;
        }
        Ok(Self {
            output: input.clone(),
            input,
            p,
            mode,
        })
    }

    pub fn max<const D: usize>(input: Shape, options: &MaxPoolOptions<D>) -> Result<Self> {
        let mut plan = Self::new(input, D, Mode::Max)?;
        plan.set_options(options.kernel_size, options.stride, options.padding, options.dilation, options.ceil_mode)?;
        plan.finish()?;
        Ok(plan)
    }

    pub fn average<const D: usize>(input: Shape, options: &AvgPoolOptions<D>) -> Result<Self> {
        let mut plan = Self::new(input, D, Mode::Average)?;
        plan.set_options(options.kernel_size, options.stride, options.padding, [1; D], options.ceil_mode)?;
        if options.count_include_pad {
            plan.p[23] |= abi::INCLUDE_PAD;
        }
        plan.finish()?;
        Ok(plan)
    }

    pub fn adaptive(input: Shape, output: &[usize]) -> Result<Self> {
        let mut plan = Self::new(input, output.len(), Mode::Adaptive)?;
        for (axis, &size) in output.iter().enumerate() {
            plan.p[8 - output.len() + axis] = u32::try_from(size).map_err(|_| Error::SizeOverflow)?;
        }
        plan.finish()?;
        Ok(plan)
    }

    fn set_options<const D: usize>(
        &mut self, kernel: [usize; D], stride: [usize; D], padding: [(usize, usize); D], dilation: [usize; D], ceil: bool,
    ) -> Result<()> {
        self.p[23] = u32::from(ceil);
        for axis in 0..D {
            let at = 3 - D + axis;
            for (index, value) in [
                (8 + at, kernel[axis]),
                (11 + at, stride[axis]),
                (14 + at, padding[axis].0),
                (17 + at, padding[axis].1),
                (20 + at, dilation[axis]),
            ] {
                self.p[index] = u32::try_from(value).map_err(|_| Error::SizeOverflow)?;
            }
        }
        for axis in 0..3 {
            let p = self.p;
            self.p[5 + axis] = abi::output_extent(p[2 + axis], p[8 + axis], p[11 + axis], p[14 + axis], p[17 + axis], p[20 + axis], ceil)
                .map_err(|_| Error::InvalidParameter)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        abi::geometry(&self.p, self.mode).map_err(|_| Error::InvalidParameter)?;
        let dimensions = self.input.num_dims() - 2;
        let mut shape = self.input.to_vec();
        for axis in 0..dimensions {
            shape[2 + axis] = self.p[8 - dimensions + axis] as usize;
        }
        self.output = Shape::from(shape);
        Ok(())
    }

    pub fn output_shape(&self) -> &Shape {
        &self.output
    }

    pub fn parameters(&self, source: &Layout, backward: bool) -> Result<[usize; abi::PARAMETERS]> {
        if source.shape() != if backward { &self.output } else { &self.input } {
            return Err(Error::ShapeMismatch);
        }
        let dimensions = self.input.num_dims() - 2;
        let mut p = [0u32; abi::PARAMETERS];
        p[..abi::GEOMETRY].copy_from_slice(&self.p);
        p[24] = source.storage_elements().max(1) as u32;
        p[25] = source.offset() as u32;
        p[26] = source.strides()[0] as i32 as u32;
        p[27] = source.strides()[1] as i32 as u32;
        for axis in 0..dimensions {
            p[31 - dimensions + axis] = source.strides()[2 + axis] as i32 as u32;
        }
        p[31] = self.mode as u32;
        abi::validate(&p, backward).map_err(|_| Error::InvalidParameter)?;
        Ok(p.map(|v| v as usize))
    }
}

fn dense(tensor: &NativeTensor) -> Result<NativeTensor> {
    if tensor.layout.is_contiguous() { Ok(tensor.clone()) } else { tensor.copy() }
}

impl NativeTensor {
    pub fn max_pool<const D: usize>(&self, options: &MaxPoolOptions<D>) -> Result<(Self, Self)> {
        let plan = Pooling::max(self.layout.shape().clone(), options)?;
        let (values, indices) = self.pool_forward(&plan)?;
        Ok((values, indices.ok_or(Error::DeviceFault)?))
    }

    pub fn avg_pool<const D: usize>(&self, options: &AvgPoolOptions<D>) -> Result<Self> {
        let plan = Pooling::average(self.layout.shape().clone(), options)?;
        self.pool_forward(&plan).map(|(values, _)| values)
    }

    pub fn adaptive_avg_pool<const D: usize>(&self, output: [usize; D]) -> Result<Self> {
        let plan = Pooling::adaptive(self.layout.shape().clone(), &output)?;
        self.pool_forward(&plan).map(|(values, _)| values)
    }

    fn empty_pool(&self, shape: Shape, dtype: DType) -> Result<Self> {
        Ok(Self {
            buffer: self.buffer.clone(),
            layout: Layout::contiguous(shape)?,
            dtype,
        })
    }

    fn pool_forward(&self, plan: &Pooling) -> Result<(Self, Option<Self>)> {
        if self.dtype != DType::F32 {
            return Err(Error::InvalidParameter);
        }
        let parameters = plan.parameters(&self.layout, false)?;
        let layout = Layout::contiguous(plan.output.clone())?;
        if layout.is_empty() {
            return Ok((
                self.empty_pool(plan.output.clone(), DType::F32)?,
                if plan.mode == Mode::Max {
                    Some(self.empty_pool(plan.output.clone(), DType::I32)?)
                } else {
                    None
                },
            ));
        }
        let count = layout.len();
        let storage = if plan.mode == Mode::Max { count * 2 } else { count };
        let buffer = Arc::new(Nvidia.produce_words(storage, Kernel::Pool, &[(&self.buffer, 0)], &parameters)?);
        if plan.mode == Mode::Max {
            let value_layout = Layout::from_parts(plan.output.clone(), layout.strides().to_vec(), count as i64, storage)?;
            let index_layout = Layout::from_parts(plan.output.clone(), layout.strides().to_vec(), 0, storage)?;
            Ok((
                Self {
                    buffer: buffer.clone(),
                    layout: value_layout,
                    dtype: DType::F32,
                },
                Some(Self {
                    buffer,
                    layout: index_layout,
                    dtype: DType::I32,
                }),
            ))
        } else {
            Ok((
                Self {
                    buffer,
                    layout,
                    dtype: DType::F32,
                },
                None,
            ))
        }
    }

    pub fn max_pool_backward<const D: usize>(&self, gradient: &Self, indices: &Self, options: &MaxPoolOptions<D>) -> Result<Self> {
        if self.dtype != DType::F32 || gradient.dtype != DType::F32 || indices.dtype != DType::I32 {
            return Err(Error::InvalidParameter);
        }
        let plan = Pooling::max(self.layout.shape().clone(), options)?;
        if gradient.layout.shape() != &plan.output || indices.layout.shape() != &plan.output {
            return Err(Error::ShapeMismatch);
        }
        if self.layout.is_empty() {
            return self.empty_pool(plan.input, DType::F32);
        }
        Nvidia.with_commands(|| {
            let n = plan.input[0];
            let c = plan.input[1];
            let spatial = plan.input[2..].iter().product::<usize>();
            let out_spatial = plan.output[2..].iter().product::<usize>();
            let indices = dense(indices)?.reshape(Shape::new([n, c, out_spatial]))?;
            let gradient = dense(gradient)?.reshape(Shape::new([n, c, out_spatial]))?;
            // Only the -1 sentinel denotes an empty window. Other invalid
            // device indices still reach scatter's recoverable bounds check.
            let padding = indices.word_scalar(WordOp::Equal, u32::MAX)?;
            let indices = indices.mask_fill_i32(&padding, 0)?;
            let gradient = gradient.mask_fill(&padding, 0.0)?;
            Self::full(Shape::new([n, c, spatial]), DType::F32, 0)?
                .scatter_add(2, &indices, &gradient)?
                .reshape(plan.input)
        })
    }

    pub fn avg_pool_backward<const D: usize>(&self, gradient: &Self, options: &AvgPoolOptions<D>) -> Result<Self> {
        let plan = Pooling::average(self.layout.shape().clone(), options)?;
        self.pool_backward(gradient, &plan)
    }

    pub fn adaptive_avg_pool_backward(&self, gradient: &Self) -> Result<Self> {
        if gradient.layout.shape().num_dims() != self.layout.shape().num_dims() {
            return Err(Error::ShapeMismatch);
        }
        let plan = Pooling::adaptive(self.layout.shape().clone(), &gradient.layout.shape()[2..])?;
        self.pool_backward(gradient, &plan)
    }

    fn pool_backward(&self, gradient: &Self, plan: &Pooling) -> Result<Self> {
        if self.dtype != DType::F32 || gradient.dtype != DType::F32 {
            return Err(Error::InvalidParameter);
        }
        let parameters = plan.parameters(&gradient.layout, true)?;
        let layout = Layout::contiguous(plan.input.clone())?;
        if layout.is_empty() {
            return self.empty_pool(plan.input.clone(), DType::F32);
        }
        let buffer = Nvidia.produce_words(layout.len(), Kernel::PoolBackward, &[(&gradient.buffer, 0)], &parameters)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype: DType::F32,
        })
    }
}
