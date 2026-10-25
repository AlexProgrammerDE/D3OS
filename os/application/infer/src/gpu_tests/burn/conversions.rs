//! Guarded resident conversions across every ordinary physical format.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{Shape, TensorData},
    native::{NativeTensor, layout::Layout},
    validation::{conversions as reference, storage::DTYPES},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel, cast::Format, storage::Mode};

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|b| {
            let mut word = [0; 4];
            word[..b.len()].copy_from_slice(b);
            u32::from_le_bytes(word)
        })
        .collect()
}
fn verify(source: d3os_ai::burn::backend::DType, destination: d3os_ai::burn::backend::DType, actual: &[u32], expected: &[u8]) -> Result<(), String> {
    let actual: Vec<_> = actual.iter().flat_map(|v| v.to_le_bytes()).collect();
    if actual[expected.len()..].iter().any(|&b| b != 0) {
        return Err(String::from("conversion output padding is not zero"));
    }
    reference::verify(source, destination, &actual[..expected.len()], expected)
}
pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    const GUARD: u32 = 0x6bad_beef;
    let layout = Layout::contiguous(Shape::new([3, 137]))
        .and_then(|l| l.permute(&[1, 0]))
        .and_then(|l| l.flip(&[0, 1]))
        .map_err(|e| format!("{e}"))?;
    for (source, source_format) in DTYPES.into_iter().zip(Format::ALL) {
        let raw = reference::input(source, source, 411, false);
        let logical = reference::view_bytes(&raw, source, 3, 137);
        let mut packed = words(&raw);
        if !raw.len().is_multiple_of(4) {
            *packed.last_mut().unwrap() |= 0xa5a5_a5a5u32 << ((raw.len() % 4) * 8);
        }
        let mut guarded = vec![GUARD];
        guarded.extend(&packed);
        guarded.push(GUARD);
        let input = backend.upload_words(&guarded).map_err(|e| format!("{e}"))?;
        let native = NativeTensor::from_data(TensorData::from_bytes_vec(raw.clone(), [3, 137], source))
            .and_then(|t| t.permute(&[1, 0]))
            .and_then(|t| t.flip(&[0, 1]))
            .map_err(|e| format!("{e}"))?;
        let scalar_bits = if source.is_float() {
            reference::reference(d3os_ai::burn::backend::DType::F64, source, (-13.25f64).to_bits())
        } else if source.is_bool() {
            1
        } else if source.is_int() {
            (-13i64) as u64
        } else {
            0xfedc_ba98_7654_3210
        };
        let scalar_bytes = scalar_bits.to_le_bytes()[..source.size()].to_vec();
        let scalar = NativeTensor::from_data(TensorData::from_bytes_vec(scalar_bytes.clone(), Shape::new([]), source)).map_err(|e| format!("{e}"))?;
        for (destination, destination_format) in DTYPES.into_iter().zip(Format::ALL) {
            let mut p = layout.storage_parameters(source_format.width(), Mode::Copy, 0).map_err(|e| format!("{e}"))?;
            p[2] = source_format as usize | ((destination_format as usize) << 8);
            let expected = reference::expected(source, destination, &logical);
            let count = expected.len().div_ceil(4);
            let output = backend.upload_words(&vec![GUARD; count + 2]).map_err(|e| format!("{e}"))?;
            backend
                .with_commands(|| backend.dispatch_words(Kernel::TensorCast, &[(&input, 1), (&output, 1)], &p))
                .map_err(|e| format!("{e}"))?;
            let mut actual = vec![0; count + 2];
            backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
            if actual[0] != GUARD || actual[count + 1] != GUARD {
                return Err(format!("cast guards {source:?}->{destination:?}"));
            }
            verify(source, destination, &actual[1..count + 1], &expected)?;
            let before = backend.service_profile().map_err(|e| format!("{e}"))?;
            let result = backend
                .with_commands(|| backend.produce_words(count, Kernel::TensorCast, &[(&input, 1)], &p))
                .map_err(|e| format!("{e}"))?;
            super::dispatch::resident(backend, &before, 0)?;
            let after = backend.service_profile().map_err(|e| format!("{e}"))?;
            if after[abi::ALLOCATE as usize].calls != before[abi::ALLOCATE as usize].calls
                || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 1
            {
                return Err(String::from("conversion full writer needed a separate clear"));
            }
            let mut actual = vec![0; count];
            backend.read_words_into(&result, 0, &mut actual).map_err(|e| format!("{e}"))?;
            verify(source, destination, &actual, &expected)?;
            let before = backend.service_profile().map_err(|e| format!("{e}"))?;
            let converted = backend
                .with_commands(|| native.cast(destination))
                .map_err(|e| format!("native {source:?}->{destination:?}: {e}"))?;
            super::dispatch::resident(backend, &before, 0)?;
            let data = converted.read().map_err(|e| format!("{e}"))?;
            if data.dtype() != destination || data.shape().as_slice() != [137, 3] {
                return Err(String::from("native conversion metadata"));
            }
            reference::verify(source, destination, data.as_bytes(), &expected)?;
            let before = backend.service_profile().map_err(|e| format!("{e}"))?;
            let converted_scalar = backend
                .with_commands(|| scalar.cast(destination))
                .map_err(|e| format!("scalar {source:?}->{destination:?}: {e}"))?;
            super::dispatch::resident(backend, &before, 0)?;
            let data = converted_scalar.read().map_err(|e| format!("{e}"))?;
            if data.dtype() != destination || data.shape().num_dims() != 0 {
                return Err(String::from("scalar conversion metadata"));
            }
            reference::verify(source, destination, data.as_bytes(), &reference::expected(source, destination, &scalar_bytes))?;
        }
        let mut preserved = vec![0; guarded.len()];
        backend.read_words_into(&input, 0, &mut preserved).map_err(|e| format!("{e}"))?;
        if preserved != guarded || native.read().map_err(|e| format!("{e}"))?.as_bytes() != logical {
            return Err(String::from("conversion changed source or padding"));
        }
        let empty = NativeTensor::from_data(TensorData::from_bytes_vec(Vec::new(), [0, 3], source)).map_err(|e| format!("{e}"))?;
        for destination in DTYPES {
            let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
            let converted = empty.cast(destination).map_err(|e| format!("{e}"))?;
            if converted.layout().shape().as_slice() != [0, 3] || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches {
                return Err(String::from("empty conversion launched work"));
            }
        }
    }
    let raw = reference::input(DTYPES[9], DTYPES[0], 129, false);
    let input = backend.upload_bytes(&raw).map_err(|e| format!("{e}"))?;
    let mut p = Layout::contiguous(Shape::new([129]))
        .and_then(|l| l.storage_parameters(Format::U64.width(), Mode::Copy, 0))
        .map_err(|e| format!("{e}"))?;
    p[2] = Format::U64 as usize | ((Format::F64 as usize) << 8);
    let output = backend
        .with_commands(|| {
            let output = backend.produce_words(258, Kernel::TensorCast, &[(&input, 0)], &p)?;
            drop(input);
            Ok::<_, Error>(output)
        })
        .map_err(|e| format!("{e}"))?;
    let mut actual = vec![0; 258];
    backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
    verify(DTYPES[9], DTYPES[0], &actual, &reference::expected(DTYPES[9], DTYPES[0], &raw))?;
    crate::report(format_args!(
        "NVIDIA_BURN_CONVERSIONS_OK pairs=225 widths=1,2,4,8 rounding=independent_binary guards=ok tails=zero full_writer=verified resident=true queued_lifetime=verified saturation=rust scalar=true empty=no_launch"
    ));
    Ok(())
}
