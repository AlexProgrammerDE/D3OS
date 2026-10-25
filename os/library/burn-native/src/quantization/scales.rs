//! Resident raw scale codes, rounded on the GPU and packed only for records.
use super::*;
use d3os_nvidia::compute::quantization::{ScaleFormat, ScaleOperation};

pub(super) fn format(dtype: ScaleDtype) -> ScaleFormat {
    match dtype {
        ScaleDtype::F32 => ScaleFormat::F32,
        ScaleDtype::F16 => ScaleFormat::F16,
        ScaleDtype::BF16 => ScaleFormat::BF16,
        ScaleDtype::UE8M0 => ScaleFormat::UE8M0,
        ScaleDtype::UE4M3 => ScaleFormat::UE4M3,
    }
}

fn convert(input: &GpuBuffer<u32>, offset: usize, count: usize, dtype: ScaleDtype, operation: ScaleOperation) -> Result<GpuBuffer<u32>> {
    let elements = if operation == ScaleOperation::Pack {
        count.div_ceil(32 / format(dtype).bits() as usize)
    } else {
        count
    };
    if elements == 0 {
        return Nvidia.allocate_words(1);
    }
    Nvidia.produce_words(
        elements,
        Kernel::TensorScale,
        &[(input, offset)],
        &[count, format(dtype) as usize, operation as usize],
    )
}

pub(super) fn encode(input: &NativeTensor, dtype: ScaleDtype, shape: Shape) -> Result<NativeTensor> {
    if input.dtype() != DType::F32 || input.layout.len() != shape.num_elements() {
        return Err(Error::InvalidParameter);
    }
    let mut input = if input.layout.is_contiguous() { input.clone() } else { input.copy()? };
    input = input.reshape(shape.clone())?;
    if dtype == ScaleDtype::F32 {
        input.dtype = DType::I32;
        return Ok(input);
    }
    let buffer = convert(&input.buffer, input.layout.offset(), input.layout.len(), dtype, ScaleOperation::Encode)?;
    Ok(NativeTensor {
        buffer: Arc::new(buffer),
        layout: Layout::contiguous(shape)?,
        dtype: DType::I32,
    })
}

pub(super) fn from_bytes(bytes: &[u8], dtype: ScaleDtype, shape: Shape) -> Result<NativeTensor> {
    let count = shape.num_elements();
    let bytes_per_scale = format(dtype).bits() as usize / 8;
    if bytes.len() != count.checked_mul(bytes_per_scale).ok_or(Error::SizeOverflow)? {
        return Err(Error::InvalidParameter);
    }
    let packed = if bytes.is_empty() {
        Nvidia.allocate_words(1)?
    } else {
        Nvidia.upload_bytes(bytes)?
    };
    let buffer = if dtype == ScaleDtype::F32 {
        packed
    } else {
        convert(&packed, 0, count, dtype, ScaleOperation::Unpack)?
    };
    Ok(NativeTensor {
        buffer: Arc::new(buffer),
        layout: Layout::contiguous(shape)?,
        dtype: DType::I32,
    })
}

pub(super) fn read(input: &NativeTensor, dtype: ScaleDtype) -> Result<Vec<u8>> {
    let input = if input.layout.is_contiguous() { input.clone() } else { input.copy()? };
    let packed = convert(&input.buffer, input.layout.offset(), input.layout.len(), dtype, ScaleOperation::Pack)?;
    let length = input.layout.len() * (format(dtype).bits() as usize / 8);
    let mut words = alloc::vec![0; length.div_ceil(4)];
    if !words.is_empty() {
        Nvidia.read_words_into(&packed, 0, &mut words)?;
    }
    let mut bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    bytes.truncate(length);
    Ok(bytes)
}
