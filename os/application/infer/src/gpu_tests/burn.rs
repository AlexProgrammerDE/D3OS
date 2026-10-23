//! Check Burn tensor ownership and views on real native GPU allocations.
use alloc::{format, string::String, vec::Vec};
use d3os_ai::{
    burn::backend::{BoolStore, DType, Shape, Slice, TensorData, TensorMetadata},
    native::{NativeDevice, NativeTensor},
};
use d3os_compute::{Error, nvidia::Nvidia};
use d3os_nvidia::compute as abi;

fn verify(backend: &Nvidia) -> Result<(), Error> {
    NativeDevice::open()?;
    let values: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 0.5).collect();
    let original = NativeTensor::from_data(TensorData::new(values.clone(), [2, 3, 4]))?;
    if !original.can_mut() {
        return Err(Error::DeviceFault);
    }
    let before = backend.service_profile()?;
    let view = original
        .clone()
        .permute(&[2, 0, 1])?
        .flip(&[0, 2])?
        .slice(&[Slice::new(1, Some(4), 2), Slice::full(), Slice::new(1, Some(3), -1)])?;
    let after = backend.service_profile()?;
    if original.can_mut()
        || view.can_mut()
        || [abi::ALLOCATE, abi::UPLOAD, abi::DOWNLOAD]
            .iter()
            .any(|&op| before[op as usize].calls != after[op as usize].calls)
    {
        return Err(Error::DeviceFault);
    }
    let expected: Vec<_> = [2, 6, 14, 18, 0, 4, 12, 16].iter().map(|&i| values[i]).collect();
    if view.read()?.as_slice::<f32>().map_err(|_| Error::DeviceFault)? != expected {
        return Err(Error::DeviceFault);
    }
    drop(view);
    if !original.can_mut() || original.read()?.as_slice::<f32>().map_err(|_| Error::DeviceFault)? != values {
        return Err(Error::DeviceFault);
    }
    let survivor = original.clone().slice(&[Slice::new(1, Some(2), 1)])?;
    drop(original);
    if !survivor.can_mut() || survivor.read()?.as_slice::<f32>().map_err(|_| Error::DeviceFault)? != &values[12..] {
        return Err(Error::DeviceFault);
    }
    let windows = NativeTensor::from_data(TensorData::new((0..12).collect::<Vec<i32>>(), [2, 6]))?.unfold(1, 3, 2)?;
    if windows.can_mut() || windows.read()?.as_slice::<i32>().map_err(|_| Error::DeviceFault)? != [0, 1, 2, 2, 3, 4, 6, 7, 8, 8, 9, 10] {
        return Err(Error::DeviceFault);
    }
    let bits = [0u32, 0x8000_0000, 0x7fc0_1234, 0x7f80_0000, 0xff80_0000, 0x0000_0001];
    let special = NativeTensor::from_data(TensorData::new(bits.iter().copied().map(f32::from_bits).collect(), [6]))?;
    let roundtrip: Vec<_> = special
        .read()?
        .as_slice::<f32>()
        .map_err(|_| Error::DeviceFault)?
        .iter()
        .map(|v| v.to_bits())
        .collect();
    if roundtrip != bits {
        return Err(Error::DeviceFault);
    }
    let booleans = NativeTensor::from_data(TensorData::new(alloc::vec![true, false, true], [3]))?;
    if booleans.dtype() != DType::Bool(BoolStore::U32) || booleans.read()?.as_bytes() != [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] {
        return Err(Error::DeviceFault);
    }
    let empty = NativeTensor::from_data(TensorData::new(Vec::<f32>::new(), [2, 0, 3]))?;
    if !empty.layout().is_empty() || !empty.read()?.as_bytes().is_empty() {
        return Err(Error::DeviceFault);
    }
    if survivor.with_layout(d3os_ai::native::layout::Layout::contiguous(Shape::new([1]))?).is_ok() {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    verify(backend).map_err(|e| format!("Burn native tensor storage: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_STORAGE_OK views=ok aliasing=ok lifetime=ok ieee_bits=ok bool=ok empty=ok backend=nvidia"
    ));
    Ok(())
}
