//! Packed integer and floating-point quantization with GPU-resident scales.
use crate::{NativeTensor, layout::Layout};
use alloc::{sync::Arc, vec::Vec};
use burn_backend::{
    DType, Shape, TensorData, TensorMetadata,
    quantization::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype, params_shape},
};
use d3os_compute::{
    Error, Result,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::{
    Kernel,
    quantization::{Codec, Operation},
};

#[derive(Clone, Debug)]
pub struct NativeQuantizedTensor {
    values: Arc<GpuBuffer<u32>>,
    layout: Layout,
    scales: NativeTensor,
    scheme: QuantScheme,
}

/// Validate schemes before allocating storage or invoking Burn's asserting
/// block-layout helpers. Scale grids follow rectangular blocks, not flat runs.
pub(crate) fn supported_scheme(scheme: &QuantScheme) -> bool {
    scheme.mode == QuantMode::Symmetric
        && scheme.scale_dtype() == ScaleDtype::F32
        && !(scheme.block_scale().is_some() && scheme.tensor_scale().is_some())
        && match scheme.store {
            QuantStore::PackedU32(0..8) => true,
            QuantStore::Native => scheme.value.size_bits() == 8,
            QuantStore::PackedNative(0..8) => scheme.value == QuantValue::E2M1,
            _ => false,
        }
}

pub(crate) fn validate_scheme(shape: &Shape, scheme: &QuantScheme) -> Result<()> {
    Layout::contiguous(shape.clone())?;
    if !supported_scheme(scheme) {
        return Err(Error::InvalidParameter);
    }
    match scheme.store {
        QuantStore::Native if scheme.value.size_bits() == 8 => {}
        QuantStore::PackedU32(axis) | QuantStore::PackedNative(axis) if axis < shape.num_dims() => {
            if !shape[shape.num_dims() - 1 - axis].is_multiple_of(scheme.num_quants()) {
                return Err(Error::InvalidShape);
            }
        }
        _ => return Err(Error::InvalidParameter),
    }
    if let Some(block) = scheme.block_size() {
        if block.len() > shape.num_dims() || block.iter().any(|&n| n == 0) {
            return Err(Error::InvalidShape);
        }
        let dimensions = block.to_dim_vec(shape.num_dims());
        if shape.iter().zip(dimensions).any(|(&n, b)| !n.is_multiple_of(b as usize)) {
            return Err(Error::InvalidShape);
        }
    }
    Ok(())
}

fn storage_axes(shape: &Shape, scheme: &QuantScheme) -> Vec<usize> {
    let mut axes: Vec<_> = (0..shape.num_dims()).collect();
    if let QuantStore::PackedU32(axis) | QuantStore::PackedNative(axis) = scheme.store {
        axes.swap(shape.num_dims() - 1 - axis, shape.num_dims() - 1);
    }
    axes
}

fn metadata(layout: &Layout, scheme: &QuantScheme) -> Result<[u32; 32]> {
    let mut out = [0; 32];
    out[..8].fill(1);
    out[16..24].fill(1);
    let blocks = scheme.block_size().map(|b| b.to_dim_vec(layout.shape().num_dims()));
    let grid = params_shape(layout.shape(), scheme);
    let mut scale_stride = 1u32;
    for axis in (0..layout.shape().num_dims()).rev() {
        out[axis] = layout.shape()[axis] as u32;
        out[8 + axis] = layout.strides()[axis] as i32 as u32;
        if let Some(blocks) = &blocks {
            out[16 + axis] = u32::from(blocks[axis]);
            out[24 + axis] = scale_stride;
            scale_stride = scale_stride.checked_mul(grid[axis] as u32).ok_or(Error::SizeOverflow)?;
        }
    }
    Ok(out)
}

fn run(input: &GpuBuffer<u32>, layout: &Layout, scales: &NativeTensor, scheme: &QuantScheme, operation: Operation) -> Result<GpuBuffer<u32>> {
    let bits = scheme.value.size_bits();
    let elements = if operation == Operation::Dequantize {
        layout.len()
    } else {
        layout.len().div_ceil(32 / bits)
    };
    if elements == 0 {
        return Nvidia.allocate_words(1);
    }
    let scales = if scales.layout.is_contiguous() { scales.clone() } else { scales.copy()? };
    let metadata = Nvidia.upload_words(&metadata(layout, scheme)?)?;
    let parameters = [
        layout.len(),
        layout.storage_elements(),
        scales.layout.len(),
        bits,
        operation as usize,
        match scheme.value {
            QuantValue::E4M3 => Codec::E4M3,
            QuantValue::E5M2 => Codec::E5M2,
            QuantValue::E2M1 => Codec::E2M1,
            value if value.is_symmetric() => Codec::IntegerSymmetric,
            _ => Codec::IntegerFull,
        } as usize,
        layout.offset(),
        0,
    ];
    Nvidia.produce_words(
        elements,
        Kernel::TensorQuantized,
        &[(input, 0), (&scales.buffer, scales.layout.offset()), (&metadata, 0)],
        &parameters,
    )
}

impl NativeQuantizedTensor {
    pub fn shape(&self) -> Shape {
        self.layout.shape().clone()
    }
    pub fn scheme(&self) -> QuantScheme {
        self.scheme
    }

    pub fn quantize(tensor: &NativeTensor, scales: &NativeTensor, scheme: QuantScheme) -> Result<Self> {
        let shape = tensor.shape();
        validate_scheme(&shape, &scheme)?;
        if tensor.dtype() != DType::F32 || scales.dtype() != DType::F32 || scales.layout.len() != params_shape(&shape, &scheme).num_elements() {
            return Err(Error::InvalidParameter);
        }
        Nvidia.record_commands(|| {
            let scales = if scales.layout.is_contiguous() { scales.clone() } else { scales.copy()? };
            let scales = scales.reshape(params_shape(&shape, &scheme))?;
            let axes = storage_axes(&shape, &scheme);
            let mut stored_scheme = scheme;
            stored_scheme.permute_block_dims(shape.num_dims(), &axes);
            let input = tensor.clone().permute(&axes)?;
            let stored_scales = if scheme.block_size().is_some() {
                scales.clone().permute(&axes)?
            } else {
                scales.clone()
            };
            let values = run(&input.buffer, &input.layout, &stored_scales, &stored_scheme, Operation::Quantize)?;
            let layout = Layout::contiguous(input.shape())?.permute(&axes)?;
            Ok(Self {
                values: Arc::new(values),
                layout,
                scales: scales.clone(),
                scheme,
            })
        })
    }

    pub fn dequantize(&self) -> Result<NativeTensor> {
        Nvidia.record_commands(|| {
            let buffer = run(&self.values, &self.layout, &self.scales, &self.scheme, Operation::Dequantize)?;
            Ok(NativeTensor {
                buffer: Arc::new(buffer),
                layout: Layout::contiguous(self.shape())?,
                dtype: DType::F32,
            })
        })
    }

    pub fn from_data(data: TensorData) -> Result<Self> {
        let DType::QFloat(scheme) = data.dtype() else {
            return Err(Error::InvalidParameter);
        };
        let shape = data.shape().clone();
        validate_scheme(&shape, &scheme)?;
        let axes = storage_axes(&shape, &scheme);
        let stored_shape = Shape::from(axes.iter().map(|&axis| shape[axis]).collect::<Vec<_>>());
        let mut stored_scheme = scheme;
        stored_scheme.permute_block_dims(shape.num_dims(), &axes);
        let scales_shape = params_shape(&stored_shape, &stored_scheme);
        let bytes = shape
            .num_elements()
            .checked_mul(scheme.value.size_bits())
            .ok_or(Error::SizeOverflow)?
            .div_ceil(32)
            * 4;
        let scale_bytes = scales_shape.num_elements().checked_mul(4).ok_or(Error::SizeOverflow)?;
        let prefix = data.as_bytes().len().checked_sub(scale_bytes).ok_or(Error::InvalidParameter)?;
        let compact = match scheme.store {
            QuantStore::Native | QuantStore::PackedNative(_) => shape
                .num_elements()
                .checked_mul(scheme.value.size_bits())
                .ok_or(Error::SizeOverflow)?
                .div_ceil(8),
            QuantStore::PackedU32(_) => bytes,
        };
        if prefix != compact && prefix != bytes {
            return Err(Error::InvalidParameter);
        }
        let values = if bytes == 0 {
            Nvidia.allocate_words(1)?
        } else {
            Nvidia.upload_bytes(&data.as_bytes()[..prefix])?
        };
        let scale_data = TensorData::try_from_bytes_vec(data.as_bytes()[prefix..].to_vec(), scales_shape, DType::F32).map_err(|_| Error::InvalidParameter)?;
        let scales = NativeTensor::from_data(scale_data)?;
        let scales = if scheme.block_size().is_some() { scales.permute(&axes)? } else { scales };
        Ok(Self {
            values: Arc::new(values),
            layout: Layout::contiguous(stored_shape)?.permute(&axes)?,
            scales,
            scheme,
        })
    }

    pub fn read(&self) -> Result<TensorData> {
        let axes = storage_axes(&self.shape(), &self.scheme);
        let layout = self.layout.permute(&axes)?;
        let mut stored_scheme = self.scheme;
        stored_scheme.permute_block_dims(self.shape().num_dims(), &axes);
        let scales = if self.scheme.block_size().is_some() {
            self.scales.clone().permute(&axes)?
        } else {
            self.scales.clone()
        };
        let values = run(&self.values, &layout, &scales, &stored_scheme, Operation::Copy)?;
        let mut words = alloc::vec![0u32; layout.len().div_ceil(32 / self.scheme.value.size_bits())];
        if !words.is_empty() {
            Nvidia.read_words_into(&values, 0, &mut words)?;
        }
        let mut bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        if matches!(self.scheme.store, QuantStore::Native | QuantStore::PackedNative(_)) {
            bytes.truncate(layout.len() * self.scheme.value.size_bits() / 8);
        }
        bytes.extend_from_slice(scales.read()?.as_bytes());
        TensorData::try_from_bytes_vec(bytes, self.shape(), DType::QFloat(self.scheme)).map_err(|_| Error::InvalidParameter)
    }

    pub fn reshape(&self, shape: Shape) -> Result<Self> {
        validate_scheme(&shape, &self.scheme)?;
        if shape.num_elements() != self.layout.len() {
            return Err(Error::ShapeMismatch);
        }
        if let Some(block) = self.scheme.block_size() {
            if block.len() > 1 && shape != self.shape() {
                return Err(Error::InvalidShape);
            }
            if params_shape(&shape, &self.scheme).num_elements() != self.scales.layout.len() {
                return Err(Error::InvalidShape);
            }
        }
        let (values, layout) = if self.layout.is_contiguous() {
            (self.values.clone(), self.layout.reshape(shape.clone())?)
        } else {
            let values = run(&self.values, &self.layout, &self.scales, &self.scheme, Operation::Copy)?;
            (Arc::new(values), Layout::contiguous(shape.clone())?)
        };
        let scales = if self.scales.layout.is_contiguous() {
            self.scales.clone()
        } else {
            self.scales.copy()?
        };
        let scales = scales.reshape(params_shape(&shape, &self.scheme))?;
        Ok(Self {
            values,
            layout,
            scales,
            scheme: self.scheme,
        })
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        let layout = self.layout.permute(axes)?;
        let mut scheme = self.scheme;
        scheme.permute_block_dims(axes.len(), axes);
        if let QuantStore::PackedU32(axis) | QuantStore::PackedNative(axis) = scheme.store {
            let packed_axis = axes.len() - 1 - axis;
            let index = axes.iter().position(|&a| a == packed_axis).ok_or(Error::InvalidParameter)?;
            let axis = axes.len() - 1 - index;
            scheme.store = match scheme.store {
                QuantStore::PackedNative(_) => QuantStore::PackedNative(axis),
                _ => QuantStore::PackedU32(axis),
            };
        }
        let scales = if scheme.block_size().is_some() {
            self.scales.clone().permute(axes)?
        } else {
            self.scales.clone()
        };
        Ok(Self {
            values: self.values.clone(),
            layout,
            scales,
            scheme,
        })
    }

    pub fn flip(&self, axes: &[usize]) -> Result<Self> {
        let layout = self.layout.flip(axes)?;
        let scales = if self.scheme.block_size().is_some() {
            self.scales.clone().flip(axes)?
        } else {
            self.scales.clone()
        };
        Ok(Self {
            values: self.values.clone(),
            layout,
            scales,
            scheme: self.scheme,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::quantization::BlockLayout;

    #[test]
    fn rectangular_scale_grids_follow_burn_block_coordinates() {
        let shape = Shape::new([2, 4, 8]);
        let scheme = QuantScheme::default().with_value(QuantValue::Q4S).per_block([2, 2, 4], ScaleDtype::F32);
        validate_scheme(&shape, &scheme).unwrap();
        let axes = [2, 0, 1];
        let layout = Layout::contiguous(shape).unwrap().permute(&axes).unwrap().flip(&[0]).unwrap();
        let mut scheme = scheme;
        scheme.permute_block_dims(3, &axes);
        let metadata = metadata(&layout, &scheme).unwrap();
        let reference = BlockLayout::new(layout.shape(), &scheme.block_size().unwrap());
        for i in 0..layout.len() {
            let mut linear = i;
            let mut address = layout.offset() as i64;
            let mut scale = 0;
            for axis in (0..8).rev() {
                let coordinate = linear % metadata[axis] as usize;
                linear /= metadata[axis] as usize;
                address += coordinate as i64 * i64::from(metadata[8 + axis] as i32);
                scale += coordinate / metadata[16 + axis] as usize * metadata[24 + axis] as usize;
            }
            assert_eq!(address as usize, layout.element_offset(i).unwrap());
            assert_eq!(scale, reference.block_of(i));
        }
    }

    #[test]
    fn reject_invalid_blocks_packing_and_unsupported_scales_before_gpu_work() {
        let shape = Shape::new([2, 16]);
        for scheme in [
            QuantScheme::default().with_value(QuantValue::Q4S).with_store(QuantStore::Native),
            QuantScheme::default().with_store(QuantStore::PackedU32(3)),
            QuantScheme::default().per_block([3], ScaleDtype::F32),
            QuantScheme::default().per_block([2, 2, 2], ScaleDtype::F32),
            QuantScheme::default().per_tensor(ScaleDtype::F16),
            QuantScheme::default().with_value(QuantValue::E2M1).with_store(QuantStore::Native),
        ] {
            assert!(validate_scheme(&shape, &scheme).is_err());
        }
        assert!(validate_scheme(&Shape::new([3]), &QuantScheme::default()).is_err());
        assert!(validate_scheme(&Shape::new([3]), &QuantScheme::default().with_store(QuantStore::Native)).is_ok());
    }

    #[test]
    fn native_fp4_requires_complete_pairs_on_its_own_packing_axis() {
        let native = QuantScheme::default().with_value(QuantValue::E2M1).with_store(QuantStore::PackedNative(0));
        assert!(validate_scheme(&Shape::new([2]), &native).is_ok());
        assert!(validate_scheme(&Shape::new([3]), &native).is_err());
        assert!(validate_scheme(&Shape::new([2]), &native.with_store(QuantStore::PackedU32(0))).is_err());
        assert!(validate_scheme(&Shape::new([8]), &native.with_store(QuantStore::PackedU32(0))).is_ok());
        let outer = native.with_store(QuantStore::PackedNative(1));
        let shape = Shape::new([2, 3]);
        assert!(validate_scheme(&shape, &outer).is_ok());
        assert_eq!(storage_axes(&shape, &outer), [1, 0]);
        assert!(validate_scheme(&shape, &native).is_err());
        assert!(validate_scheme(&shape, &outer.with_value(QuantValue::Q4F)).is_err());
    }
}
