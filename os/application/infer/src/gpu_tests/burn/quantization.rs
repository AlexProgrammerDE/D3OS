//! Public Burn quantization, independently packed references, and guarded I/O.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::burn::{
    backend::{DType, TensorData},
    tensor::{
        Device, Tensor,
        quantization::{QuantScheme, QuantStore, QuantValue, QuantizationParameters, ScaleDtype},
    },
};
use d3os_compute::{Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel};

fn check<const D: usize>(tensor: Tensor<D>, expected: &[f32]) -> Result<(), String> {
    let actual = tensor.dequantize().try_into_vec_as::<f32>().map_err(|e| format!("quantized read: {e:?}"))?;
    if actual.len() != expected.len() || actual.iter().zip(expected).any(|(a, b)| a.to_bits() != b.to_bits()) {
        return Err(format!("quantized values: {actual:?} != {expected:?}"));
    }
    Ok(())
}

fn packed(values: &[i32], bits: usize) -> Vec<u8> {
    values
        .chunks(32 / bits)
        .flat_map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u32, |word, (i, &value)| word | ((value as u32 & ((1 << bits) - 1)) << (i * bits)))
                .to_le_bytes()
        })
        .collect()
}

fn public(backend: &Nvidia, device: &Device) -> Result<(), String> {
    let input = [-9., -8., -7.5, -7., -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 6.5, 7., 7.5, 8., 9., 0.];
    let rounded = [-9i32, -8, -8, -7, -2, -2, 0, 0, 2, 2, 6, 7, 8, 8, 9, 0];
    let values: Vec<_> = input.into_iter().cycle().take(32).collect();
    let source = Tensor::<2>::from_data(TensorData::new(values.clone(), [2, 16]), device);
    for value in [
        QuantValue::Q8S,
        QuantValue::Q8F,
        QuantValue::Q4S,
        QuantValue::Q4F,
        QuantValue::Q2S,
        QuantValue::Q2F,
    ] {
        let scheme = QuantScheme::default().with_value(value);
        let (low, high) = value.range();
        let expected: Vec<_> = rounded.into_iter().cycle().take(32).map(|n| n.clamp(low as i32, high as i32)).collect();
        let floats: Vec<_> = expected.iter().map(|&n| n as f32).collect();
        let scales = Tensor::<1>::from_floats([1.], device);
        let before = backend.service_profile().map_err(|e| format!("quantization profile: {e}"))?;
        let q = source.clone().quantize(&scheme, QuantizationParameters { scales, global: None });
        let decoded = q.clone().dequantize();
        let after = backend.service_profile().map_err(|e| format!("quantization profile: {e}"))?;
        if before[abi::DOWNLOAD as usize].calls != after[abi::DOWNLOAD as usize].calls || decoded.device() != *device {
            return Err(String::from("quantization transferred activations to the host or changed device"));
        }
        check(decoded, &floats)?;
        check(q.clone().flip([0, 1]), &floats.iter().rev().copied().collect::<Vec<_>>())?;
        let transposed: Vec<_> = (0..16).flat_map(|col| [floats[col], floats[16 + col]]).collect();
        check(q.clone().transpose(), &transposed)?;
        let outer_scheme = scheme.with_store(QuantStore::PackedU32(1));
        let outer = source.clone().transpose().quantize(
            &outer_scheme,
            QuantizationParameters {
                scales: Tensor::<1>::from_floats([1.], device),
                global: None,
            },
        );
        check(outer, &transposed)?;

        check(q.clone().reshape([1, 32]), &floats)?;
        let mut bytes = packed(&expected, value.size_bits());
        bytes.extend_from_slice(&1f32.to_le_bytes());
        let data = q.clone().try_into_data().map_err(|e| format!("quantized save: {e:?}"))?;
        if data.as_bytes() != bytes || data.dtype() != DType::QFloat(scheme) {
            return Err(format!("quantized packed serialization mismatch: {value:?}"));
        }
        check(Tensor::<2>::from_data(data, device), &floats)?;
        let data = q.transpose().try_into_data().map_err(|e| format!("quantized view save: {e:?}"))?;
        check(Tensor::<2>::from_data(data, device), &transposed)?;
        if value.is_symmetric() {
            let endpoints: Vec<_> = (0..32).map(|i| if i % 2 == 0 { high } else { -high }).collect();
            let x = Tensor::<2>::from_data(TensorData::new(endpoints.clone(), [2, 16]), device);
            check(x.quantize_dynamic(&scheme), &endpoints)?;
            check(Tensor::<2>::zeros([2, 16], device).quantize_dynamic(&scheme), &[0.; 32])?;
        }
    }
    check(source, &values)?;

    let scheme = QuantScheme::default().with_value(QuantValue::Q4S).per_block([2, 4], ScaleDtype::F32);
    let scales = [0.5, 1., 2., 4.];
    let values: Vec<_> = (0..32).map(|i| (i as i32 % 7 - 3) as f32 * scales[(i % 16) / 4]).collect();
    let q = Tensor::<2>::from_data(TensorData::new(values.clone(), [2, 16]), device).quantize(
        &scheme,
        QuantizationParameters {
            scales: Tensor::<1>::from_floats(scales, device),
            global: None,
        },
    );
    check(q.clone(), &values)?;
    let transposed: Vec<_> = (0..16).flat_map(|col| [values[col], values[16 + col]]).collect();
    check(q.clone().transpose(), &transposed)?;
    let data = q.transpose().try_into_data().map_err(|e| format!("block quantized save: {e:?}"))?;
    check(Tensor::<2>::from_data(data, device), &transposed)?;
    let endpoints: Vec<_> = (0..32).map(|i| (if i % 2 == 0 { 7. } else { -7. }) * scales[(i % 16) / 4]).collect();
    check(
        Tensor::<2>::from_data(TensorData::new(endpoints.clone(), [2, 16]), device).quantize_dynamic(&scheme),
        &endpoints,
    )?;
    let scheme = QuantScheme::default().with_store(QuantStore::Native);
    check(
        Tensor::<1>::zeros([0], device).quantize(
            &scheme,
            QuantizationParameters {
                scales: Tensor::<1>::from_floats([1.], device),
                global: None,
            },
        ),
        &[],
    )?;
    let data = TensorData::quantized(vec![-2i8, 0, 3], [3], scheme, &[0.5], None);
    let q = Tensor::<1>::from_data(data, device);
    check(q.clone(), &[-1., 0., 1.5])?;
    let data = q.try_into_data().map_err(|e| format!("native byte save: {e:?}"))?;
    check(Tensor::<1>::from_data(data, device), &[-1., 0., 1.5])?;
    Ok(())
}

fn guards(backend: &Nvidia) -> Result<(), Error> {
    const GUARD: u32 = 0xd305abcd;
    for bits in [2usize, 4, 8] {
        let mut meta = [0u32; 32];
        meta[..8].fill(1);
        meta[16..24].fill(1);
        meta[0] = 17;
        meta[8] = 1;
        let input = backend.upload_words(&[1f32.to_bits(); 17])?;
        let scales = backend.upload_words(&[1f32.to_bits()])?;
        let metadata = backend.upload_words(&meta)?;
        let words = 17usize.div_ceil(32 / bits);
        let output = backend.upload_words(&vec![GUARD; words + 2])?;
        backend.dispatch_words(
            Kernel::TensorQuantized,
            &[(&input, 0), (&scales, 0), (&metadata, 0), (&output, 1)],
            &[17, 17, 1, bits, 0, 1, 0, 0],
        )?;
        let mut actual = vec![0; words + 2];
        backend.read_words_into(&output, 0, &mut actual)?;
        let reference = packed(&[1; 17], bits);
        let bytes: Vec<_> = actual[1..words + 1].iter().flat_map(|n| n.to_le_bytes()).collect();
        if actual[0] != GUARD || actual[words + 1] != GUARD || bytes != reference {
            return Err(Error::DeviceFault);
        }
        let decoded = backend.upload_words(&[GUARD; 19])?;
        backend.dispatch_words(
            Kernel::TensorQuantized,
            &[(&output, 1), (&scales, 0), (&metadata, 0), (&decoded, 1)],
            &[17, 17, 1, bits, 1, 1, 0, 0],
        )?;
        let mut actual = [0; 19];
        backend.read_words_into(&decoded, 0, &mut actual)?;
        if actual[0] != GUARD || actual[18] != GUARD || actual[1..18] != [1f32.to_bits(); 17] {
            return Err(Error::DeviceFault);
        }
        // Metadata is a raw device buffer. Derived addresses outside declared
        // storage must be masked even if an application forges that metadata.
        meta[8] = 128;
        let metadata = backend.upload_words(&meta)?;
        backend.dispatch_words(
            Kernel::TensorQuantized,
            &[(&output, 1), (&scales, 0), (&metadata, 0), (&decoded, 1)],
            &[17, 17, 1, bits, 1, 1, 0, 0],
        )?;
        backend.read_words_into(&decoded, 0, &mut actual)?;
        if actual[0] != GUARD || actual[18] != GUARD || actual[2..18] != [0; 16] {
            return Err(Error::DeviceFault);
        }
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let device = d3os_ai::native_device().map_err(|e| format!("quantization device: {e}"))?;
    public(backend, &device)?;
    guards(backend).map_err(|e| format!("quantization guards: {e}"))?;
    device.sync().map_err(|e| format!("quantization synchronization: {e:?}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_QUANTIZATION_OK api=public integer_bits=8,4,2 formats=6 rounding=ties_even scales=per_tensor,rectangular_block serialization=verified views=verified guards=verified backend=nvidia"
    ));
    Ok(())
}
