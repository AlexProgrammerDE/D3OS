use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::burn::{
    backend::TensorData,
    tensor::{DType, Shape},
};
use d3os_ai::native::NativeTensor;
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute as abi;

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let a = NativeTensor::from_data(TensorData::new((0..771).map(|i| (i / 257 + 1) as f32).collect::<Vec<_>>(), [3, 257]))
        .and_then(|t| t.flip(&[0, 1]))
        .map_err(|e| format!("{e}"))?;
    let b = NativeTensor::from_data(TensorData::new(vec![4.0f32, 5., 6.], [3, 1])).map_err(|e| format!("{e}"))?;
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let result = backend
        .with_commands(|| {
            let output = a.cross(&b, 0)?;
            drop(a);
            drop(b);
            Ok::<_, Error>(output)
        })
        .map_err(|e| format!("{e}"))?;
    let after = backend.service_profile().map_err(|e| format!("{e}"))?;
    if [abi::UPLOAD, abi::DOWNLOAD]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
    {
        return Err(String::from("cross transferred tensor data during execution"));
    }
    let data = result.read().map_err(|e| format!("{e}"))?;
    if data
        .as_slice::<f32>()
        .map_err(|e| format!("{e:?}"))?
        .iter()
        .enumerate()
        .any(|(i, &v)| v != [7., -14., 7.][i / 257])
    {
        return Err(String::from("resident cross or queued buffer lifetime mismatch"));
    }
    let empty = NativeTensor::full(Shape::new([0, 3]), DType::F32, 0).map_err(|e| format!("{e}"))?;
    let wrong_type = NativeTensor::full(Shape::new([0, 3]), DType::I32, 0).map_err(|e| format!("{e}"))?;
    let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
    let output = empty.cross(&empty, 1).map_err(|e| format!("{e}"))?;
    if !output.layout().is_empty()
        || !matches!(empty.cross(&empty, 2), Err(Error::ShapeMismatch))
        || !matches!(empty.cross(&empty, 0), Err(Error::ShapeMismatch))
        || !matches!(empty.cross(&wrong_type, 1), Err(Error::InvalidParameter))
        || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
    {
        return Err(String::from("empty cross shape, dtype, or launch validation"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_CROSS_OK resident=ok queued_lifetime=verified empty=no_launch invalid_inputs=rejected"
    ));
    Ok(())
}
