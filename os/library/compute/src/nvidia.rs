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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Storage {
    F32,
    Bf16,
    Words,
}

#[derive(Debug)]
pub struct GpuBuffer<T = f32> {
    handle: u64,
    elements: usize,
    words: usize,
    storage: Storage,
    element: core::marker::PhantomData<T>,
}

impl<T> Buffer for GpuBuffer<T> {
    fn len(&self) -> usize {
        self.elements
    }
}

impl<T> Drop for GpuBuffer<T> {
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
    pub fn profile(&self) -> Result<[abi::ProfileEntry; abi::KERNEL_COUNT]> {
        let mut entries = [abi::ProfileEntry::default(); abi::KERNEL_COUNT];
        request(Request {
            pointer: entries.as_mut_ptr() as u64,
            elements: abi::KERNEL_COUNT as u64,
            ..operation(abi::PROFILE)
        })?;
        Ok(entries)
    }

    pub fn new() -> Result<Self> {
        if request(operation(abi::INFO))? != abi::VERSION as usize {
            return Err(Error::DeviceUnavailable);
        }
        Ok(Self)
    }

    fn allocate(&self, shape: &[usize]) -> Result<Tensor<Self>> {
        let elements = element_count(shape)?;
        Tensor::from_buffer(shape, self.allocate_storage(elements)?)
    }

    fn allocate_storage<T>(&self, elements: usize) -> Result<GpuBuffer<T>> {
        element_count(&[elements])?;
        dimension(elements)?;
        let handle = request(Request {
            elements: elements as u64,
            ..operation(abi::ALLOCATE)
        })? as u64;
        Ok(GpuBuffer {
            handle,
            elements,
            words: elements,
            storage: Storage::F32,
            element: core::marker::PhantomData,
        })
    }

    fn launch<T>(&self, kernel: Kernel, buffers: &[(&GpuBuffer<T>, usize)], parameters: &[usize]) -> Result<()> {
        request(self.prepare(kernel, buffers, parameters)?)?;
        Ok(())
    }

    fn command(&self, kernel: Kernel, parameters: &[usize]) -> Result<(Request, abi::Work)> {
        let mut call = Request {
            kernel: kernel as u32,
            parameter_count: parameters.len() as u32,
            ..operation(abi::LAUNCH)
        };
        if parameters.len() > call.parameters.len() {
            return Err(Error::InvalidParameter);
        }
        for (dest, &value) in call.parameters.iter_mut().zip(parameters) {
            *dest = dimension(value)?;
        }
        let work = kernel.work(&call.parameters[..parameters.len()]).map_err(|_| Error::InvalidParameter)?;
        call.buffer_count = work.pointers as u32;
        Ok((call, work))
    }

    fn bind<T>(call: &mut Request, buffers: &[(&GpuBuffer<T>, usize)], extents: &[u64]) -> Result<()> {
        if buffers.len() != extents.len() || buffers.len() > call.buffers.len() {
            return Err(Error::InvalidParameter);
        }
        let kernel = Kernel::from_raw(call.kernel).map_err(|_| Error::InvalidParameter)?;
        for (i, (&(buffer, offset), &extent)) in buffers.iter().zip(extents).enumerate() {
            let packed = matches!((kernel, i), (Kernel::ProjectBf16, 1) | (Kernel::DequantizeBf16, 0));
            if buffer.storage != Storage::Words && (buffer.storage == Storage::Bf16) != packed {
                return Err(Error::InvalidParameter);
            }
            if (offset as u64).checked_add(extent).is_none_or(|end| end > buffer.words as u64) {
                return Err(Error::ShapeMismatch);
            }
            call.buffers[i] = buffer.handle;
            call.offsets[i] = offset as u64;
        }
        Ok(())
    }

    fn prepare<T>(&self, kernel: Kernel, buffers: &[(&GpuBuffer<T>, usize)], parameters: &[usize]) -> Result<Request> {
        let (mut call, work) = self.command(kernel, parameters)?;
        Self::bind(&mut call, buffers, &work.extents[..work.pointers])?;
        Ok(call)
    }

    fn produce(&self, shape: &[usize], kernel: Kernel, inputs: &[(&GpuBuffer, usize)], parameters: &[usize]) -> Result<Tensor<Self>> {
        let elements = element_count(shape)?;
        Tensor::from_buffer(shape, self.produce_storage(elements, kernel, inputs, parameters, Storage::F32)?)
    }

    fn produce_storage<T>(
        &self, elements: usize, kernel: Kernel, inputs: &[(&GpuBuffer, usize)], parameters: &[usize], storage: Storage,
    ) -> Result<GpuBuffer<T>> {
        element_count(&[elements])?;
        let (mut call, work) = self.command(kernel, parameters)?;
        if !kernel.allocates_output() || work.output != inputs.len() || work.pointers != inputs.len() + 1 || work.extents[work.output] != elements as u64 {
            return Err(Error::InvalidParameter);
        }
        Self::bind(&mut call, inputs, &work.extents[..inputs.len()])?;
        call.operation = abi::LAUNCH_ALLOCATE;
        call.elements = elements as u64;
        let handle = request(call)? as u64;
        Ok(GpuBuffer {
            handle,
            elements,
            words: elements,
            storage,
            element: core::marker::PhantomData,
        })
    }

    /// Allocate zeroed, process-owned words for native non-tensor compute.
    pub fn allocate_words(&self, elements: usize) -> Result<GpuBuffer<u32>> {
        let mut buffer = self.allocate_storage(elements)?;
        buffer.storage = Storage::Words;
        Ok(buffer)
    }

    pub fn upload_words(&self, values: &[u32]) -> Result<GpuBuffer<u32>> {
        let buffer = self.allocate_words(values.len())?;
        self.write_words(&buffer, 0, values)?;
        Ok(buffer)
    }

    pub fn write_words(&self, buffer: &GpuBuffer<u32>, offset: usize, values: &[u32]) -> Result<()> {
        if offset.checked_add(values.len()).is_none_or(|end| end > buffer.elements) {
            return Err(Error::ShapeMismatch);
        }
        if !values.is_empty() {
            request(Request {
                handle: buffer.handle,
                pointer: values.as_ptr() as u64,
                elements: values.len() as u64,
                offset: offset as u64,
                ..operation(abi::UPLOAD)
            })?;
        }
        Ok(())
    }

    pub fn read_words_into(&self, buffer: &GpuBuffer<u32>, offset: usize, values: &mut [u32]) -> Result<()> {
        if offset.checked_add(values.len()).is_none_or(|end| end > buffer.elements) {
            return Err(Error::ShapeMismatch);
        }
        if !values.is_empty() {
            request(Request {
                handle: buffer.handle,
                pointer: values.as_mut_ptr() as u64,
                elements: values.len() as u64,
                offset: offset as u64,
                ..operation(abi::DOWNLOAD)
            })?;
        }
        Ok(())
    }

    /// Launch a built-in word kernel with the same checked ownership and range
    /// rules as tensor dispatch. Offsets and lengths use 32-bit words.
    pub fn dispatch_words(&self, kernel: Kernel, buffers: &[(&GpuBuffer<u32>, usize)], parameters: &[usize]) -> Result<()> {
        self.launch(kernel, buffers, parameters)
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

    fn argmax(&self, input: &Tensor<Self>) -> Result<usize> {
        // Each of 32 warps returns an exact index and value. Combining 32 pairs
        // avoids downloading and scanning the complete vocabulary on the CPU.
        let output = self.produce_storage(64, Kernel::ArgMax, &[(&input.buffer, 0)], &[input.len()], Storage::Words)?;
        let mut partials = [0u32; 64];
        self.read_words_into(&output, 0, &mut partials)?;
        let mut best = (usize::MAX, f32::NEG_INFINITY);
        for pair in partials.chunks_exact(2) {
            let index = pair[0] as usize;
            if index == u32::MAX as usize {
                continue;
            }
            let value = f32::from_bits(pair[1]);
            if !value.is_finite() {
                return Err(Error::NonFinite);
            }
            if index >= input.len() {
                return Err(Error::DeviceFault);
            }
            if value > best.1 || (value == best.1 && index < best.0) {
                best = (index, value);
            }
        }
        if best.0 == usize::MAX {
            return Err(Error::DeviceFault);
        }
        Ok(best.0)
    }

    fn upload_bf16(&self, values: impl ExactSizeIterator<Item = u16>) -> Result<GpuBuffer> {
        let elements = values.len();
        element_count(&[elements])?;
        dimension(elements)?;
        let mut words = Vec::new();
        words.try_reserve_exact(elements.div_ceil(2)).map_err(|_| Error::AllocationFailed)?;
        let mut count = 0;
        for value in values {
            if value & 0x7f80 == 0x7f80 {
                return Err(Error::NonFinite);
            }
            if count % 2 == 0 {
                words.push(u32::from(value));
            } else {
                *words.last_mut().ok_or(Error::InvalidParameter)? |= u32::from(value) << 16;
            }
            count += 1;
        }
        if count != elements {
            return Err(Error::InvalidParameter);
        }
        let mut buffer = self.allocate_storage::<f32>(words.len())?;
        request(Request {
            handle: buffer.handle,
            pointer: words.as_ptr() as u64,
            elements: words.len() as u64,
            ..operation(abi::UPLOAD)
        })?;
        buffer.elements = elements;
        buffer.storage = Storage::Bf16;
        Ok(buffer)
    }

    fn download(&self, tensor: &Tensor<Self>) -> Result<Vec<f32>> {
        if tensor.buffer.storage == Storage::Bf16 {
            let mut words = Vec::new();
            words.try_reserve_exact(tensor.buffer.words).map_err(|_| Error::AllocationFailed)?;
            words.resize(tensor.buffer.words, 0u32);
            request(Request {
                handle: tensor.buffer.handle,
                pointer: words.as_mut_ptr() as u64,
                elements: words.len() as u64,
                ..operation(abi::DOWNLOAD)
            })?;
            let mut values = Vec::new();
            values.try_reserve_exact(tensor.len()).map_err(|_| Error::AllocationFailed)?;
            for i in 0..tensor.len() {
                let bits = ((words[i / 2] >> ((i % 2) * 16)) & 0xffff) << 16;
                let value = f32::from_bits(bits);
                if !value.is_finite() {
                    return Err(Error::NonFinite);
                }
                values.push(value);
            }
            return Ok(values);
        }
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
        self.produce(&[*rows, *cols], Kernel::Matmul, &[(&a.buffer, 0), (&b.buffer, 0)], &[*rows, *inner, *cols])
    }

    fn linear(&self, input: &Tensor<Self>, weights: &Tensor<Self>, bias: &Tensor<Self>) -> Result<Tensor<Self>> {
        let ([batch, features], [outputs, weight_features], [bias_outputs]) = (input.shape(), weights.shape(), bias.shape()) else {
            return Err(Error::ShapeMismatch);
        };
        if features != weight_features || outputs != bias_outputs {
            return Err(Error::ShapeMismatch);
        }
        let output = self.produce(
            &[*batch, *outputs],
            if weights.buffer.storage == Storage::Bf16 {
                Kernel::ProjectBf16
            } else {
                Kernel::Project
            },
            &[(&input.buffer, 0), (&weights.buffer, 0)],
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
