//! Native D3OS NVIDIA backend. Buffers remain in VRAM between operations.
use alloc::vec::Vec;
use d3os_nvidia::compute::{self as abi, Kernel, Request};

use crate::{Backend, Buffer, Conv2d, Error, Pool2d, Result, Tensor, element_count, output_extent};

mod transformer;

fn request(request: Request) -> Result<usize> {
    #[cfg(all(target_os = "none", target_arch = "x86_64"))]
    {
        use syscall::{SystemCall, return_vals::Errno, syscall};
        syscall(SystemCall::GpuCompute, &[&request as *const Request as usize]).map_err(|error| match error {
            Errno::ENOTSUP => Error::DeviceUnavailable,
            Errno::ENOMEM => Error::AllocationFailed,
            Errno::EINVAL | Errno::EINVALH => Error::InvalidParameter,
            _ => Error::DeviceFault,
        })
    }
    #[cfg(not(all(target_os = "none", target_arch = "x86_64")))]
    {
        let _ = request;
        Err(Error::DeviceUnavailable)
    }
}

fn operation(operation: u32) -> Request {
    Request {
        version: abi::VERSION,
        operation,
        ..Request::default()
    }
}

fn dimension(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::SizeOverflow)
}

#[derive(Debug)]
pub struct GpuBuffer {
    handle: u64,
    elements: usize,
}

impl Buffer for GpuBuffer {
    fn len(&self) -> usize {
        self.elements
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        let _ = request(Request {
            handle: self.handle,
            ..operation(abi::FREE)
        });
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Nvidia;

impl Nvidia {
    pub fn new() -> Result<Self> {
        if request(operation(abi::INFO))? != abi::VERSION as usize {
            return Err(Error::DeviceUnavailable);
        }
        Ok(Self)
    }

    fn allocate(&self, shape: &[usize]) -> Result<Tensor<Self>> {
        let elements = element_count(shape)?;
        dimension(elements)?;
        let handle = request(Request {
            elements: elements as u64,
            ..operation(abi::ALLOCATE)
        })? as u64;
        Tensor::from_buffer(shape, GpuBuffer { handle, elements })
    }

    fn launch(&self, kernel: Kernel, buffers: &[(&GpuBuffer, usize)], parameters: &[usize]) -> Result<()> {
        request(self.prepare(kernel, buffers, parameters)?)?;
        Ok(())
    }

    fn prepare(&self, kernel: Kernel, buffers: &[(&GpuBuffer, usize)], parameters: &[usize]) -> Result<Request> {
        let mut call = Request {
            kernel: kernel as u32,
            buffer_count: buffers.len() as u32,
            parameter_count: parameters.len() as u32,
            ..operation(abi::LAUNCH)
        };
        if buffers.len() > call.buffers.len() || parameters.len() > call.parameters.len() {
            return Err(Error::InvalidParameter);
        }
        for (dest, &value) in call.parameters.iter_mut().zip(parameters) {
            *dest = dimension(value)?;
        }
        let work = kernel.work(&call.parameters[..parameters.len()]).map_err(|_| Error::InvalidParameter)?;
        if buffers.len() != work.pointers {
            return Err(Error::InvalidParameter);
        }
        for (i, &(buffer, offset)) in buffers.iter().enumerate() {
            if (offset as u64).checked_add(work.extents[i]).is_none_or(|end| end > buffer.elements as u64) {
                return Err(Error::ShapeMismatch);
            }
            call.buffers[i] = buffer.handle;
            call.offsets[i] = offset as u64;
        }
        Ok(call)
    }
}

impl Backend for Nvidia {
    type Buffer = GpuBuffer;

    fn name(&self) -> &'static str {
        "nvidia"
    }

    fn upload(&self, values: Vec<f32>) -> Result<GpuBuffer> {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFinite);
        }
        let tensor = self.allocate(&[values.len()])?;
        request(Request {
            handle: tensor.buffer.handle,
            pointer: values.as_ptr() as u64,
            elements: values.len() as u64,
            ..operation(abi::UPLOAD)
        })?;
        Ok(tensor.buffer)
    }

    fn download(&self, tensor: &Tensor<Self>) -> Result<Vec<f32>> {
        let mut values = Vec::new();
        values.try_reserve_exact(tensor.len()).map_err(|_| Error::AllocationFailed)?;
        values.resize(tensor.len(), 0.0);
        request(Request {
            handle: tensor.buffer.handle,
            pointer: values.as_mut_ptr() as u64,
            elements: tensor.len() as u64,
            ..operation(abi::DOWNLOAD)
        })?;
        if values.iter().any(|value: &f32| !value.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(values)
    }

    fn matmul(&self, a: &Tensor<Self>, b: &Tensor<Self>) -> Result<Tensor<Self>> {
        let ([rows, inner], [other_inner, cols]) = (a.shape(), b.shape()) else {
            return Err(Error::ShapeMismatch);
        };
        if inner != other_inner {
            return Err(Error::ShapeMismatch);
        }
        let output = self.allocate(&[*rows, *cols])?;
        self.launch(Kernel::Matmul, &[(&a.buffer, 0), (&b.buffer, 0), (&output.buffer, 0)], &[*rows, *inner, *cols])?;
        Ok(output)
    }

    fn linear(&self, input: &Tensor<Self>, weights: &Tensor<Self>, bias: &Tensor<Self>) -> Result<Tensor<Self>> {
        let ([batch, features], [outputs, weight_features], [bias_outputs]) = (input.shape(), weights.shape(), bias.shape()) else {
            return Err(Error::ShapeMismatch);
        };
        if features != weight_features || outputs != bias_outputs {
            return Err(Error::ShapeMismatch);
        }
        let output = self.allocate(&[*batch, *outputs])?;
        self.launch(
            Kernel::Project,
            &[(&input.buffer, 0), (&weights.buffer, 0), (&output.buffer, 0)],
            &[*batch, *features, *outputs],
        )?;
        self.launch(
            Kernel::BiasRow,
            &[(&output.buffer, 0), (&bias.buffer, 0), (&output.buffer, 0)],
            &[output.len(), *outputs],
        )?;
        Ok(output)
    }

    fn conv2d(&self, input: &Tensor<Self>, weights: &Tensor<Self>, bias: &Tensor<Self>, config: Conv2d) -> Result<Tensor<Self>> {
        let ([batch, channels, height, width], [outputs, weight_channels, kh, kw], [bias_outputs]) = (input.shape(), weights.shape(), bias.shape()) else {
            return Err(Error::ShapeMismatch);
        };
        if channels != weight_channels || outputs != bias_outputs {
            return Err(Error::ShapeMismatch);
        }
        let oh = output_extent(*height, *kh, config.stride[0], config.padding[0])?;
        let ow = output_extent(*width, *kw, config.stride[1], config.padding[1])?;
        let inner = element_count(&[*channels, *kh, *kw])?;
        let spatial = element_count(&[oh, ow])?;
        let image = element_count(&[*channels, *height, *width])?;
        let output_image = element_count(&[*outputs, spatial])?;
        let output = self.allocate(&[*batch, *outputs, oh, ow])?;
        let columns = self.allocate(&[inner, spatial])?;
        for n in 0..*batch {
            self.launch(
                Kernel::Im2col,
                &[(&input.buffer, n * image), (&columns.buffer, 0)],
                &[
                    *channels,
                    *height,
                    *width,
                    *kh,
                    *kw,
                    oh,
                    ow,
                    config.stride[0],
                    config.stride[1],
                    config.padding[0],
                    config.padding[1],
                ],
            )?;
            self.launch(
                Kernel::Matmul,
                &[(&weights.buffer, 0), (&columns.buffer, 0), (&output.buffer, n * output_image)],
                &[*outputs, inner, spatial],
            )?;
        }
        self.launch(
            Kernel::BiasChannel,
            &[(&output.buffer, 0), (&bias.buffer, 0), (&output.buffer, 0)],
            &[output.len(), spatial, *outputs],
        )?;
        Ok(output)
    }

    fn relu(&self, input: Tensor<Self>) -> Result<Tensor<Self>> {
        self.launch(Kernel::Relu, &[(&input.buffer, 0), (&input.buffer, 0)], &[input.len()])?;
        Ok(input)
    }

    fn max_pool2d(&self, input: &Tensor<Self>, config: Pool2d) -> Result<Tensor<Self>> {
        let [batch, channels, height, width] = input.shape() else {
            return Err(Error::ShapeMismatch);
        };
        let oh = output_extent(*height, config.kernel[0], config.stride[0], 0)?;
        let ow = output_extent(*width, config.kernel[1], config.stride[1], 0)?;
        let output = self.allocate(&[*batch, *channels, oh, ow])?;
        let planes = element_count(&[*batch, *channels])?;
        self.launch(
            Kernel::MaxPool,
            &[(&input.buffer, 0), (&output.buffer, 0)],
            &[
                planes,
                *height,
                *width,
                config.kernel[0],
                config.kernel[1],
                config.stride[0],
                config.stride[1],
                oh,
                ow,
            ],
        )?;
        Ok(output)
    }

    fn softmax(&self, input: &Tensor<Self>) -> Result<Tensor<Self>> {
        let [rows, cols] = input.shape() else {
            return Err(Error::ShapeMismatch);
        };
        let output = self.allocate(input.shape())?;
        self.launch(Kernel::Softmax, &[(&input.buffer, 0), (&output.buffer, 0)], &[*rows, *cols])?;
        Ok(output)
    }
}
