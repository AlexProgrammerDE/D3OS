//! Whole-word writers preserve packed elements and initialize physical tails.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{DType, Shape, TensorData},
    native::{NativeTensor, layout::Layout},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{
    self as abi, Kernel,
    elementwise::Elementwise,
    storage::{Mode, Width},
};

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

fn full_writer(backend: &Nvidia, before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS]) -> Result<(), String> {
    super::dispatch::resident(backend, before, 0)?;
    let after = backend.service_profile().map_err(|e| format!("{e}"))?;
    if after[abi::ALLOCATE as usize].calls != before[abi::ALLOCATE as usize].calls
        || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 1
    {
        return Err(String::from("typed storage full writer needed a separate allocation clear"));
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    const GUARD: u32 = 0x6bad_beef;
    for width in Width::ALL {
        let size = width.bytes() as usize;
        let raw: Vec<_> = (0..411u64)
            .flat_map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15).to_le_bytes()[..size].to_vec())
            .collect();
        let mut packed = words(&raw);
        if !raw.len().is_multiple_of(4) {
            *packed.last_mut().unwrap() |= 0xa5a5_a5a5u32 << ((raw.len() % 4) * 8);
        }
        let mut input = vec![GUARD];
        input.extend(&packed);
        input.push(GUARD);
        let source = backend.upload_words(&input).map_err(|e| format!("{e}"))?;
        let layout = Layout::contiguous(Shape::new([3, 137]))
            .and_then(|l| l.permute(&[1, 0]))
            .and_then(|l| l.flip(&[0, 1]))
            .map_err(|e| format!("{e}"))?;
        let parameters = layout.storage_parameters(width, Mode::Copy, 0).map_err(|e| format!("{e}"))?;
        let mut expected_bytes = Vec::new();
        for row in 0..137 {
            for col in 0..3 {
                let at = ((2 - col) * 137 + 136 - row) * size;
                expected_bytes.extend_from_slice(&raw[at..at + size]);
            }
        }
        let expected = words(&expected_bytes);
        let output = backend.upload_words(&vec![GUARD; expected.len() + 2]).map_err(|e| format!("{e}"))?;
        backend
            .with_commands(|| backend.dispatch_words(Kernel::TensorStorage, &[(&source, 1), (&output, 1)], &parameters))
            .map_err(|e| format!("{e}"))?;
        let mut actual = vec![0; expected.len() + 2];
        backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
        if actual[0] != GUARD || actual[actual.len() - 1] != GUARD || actual[1..actual.len() - 1] != expected {
            return Err(format!("typed {width:?} copy guards or packed tail"));
        }
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let result = backend
            .with_commands(|| backend.produce_words(expected.len(), Kernel::TensorStorage, &[(&source, 1)], &parameters))
            .map_err(|e| format!("{e}"))?;
        full_writer(backend, &before)?;
        let mut actual = vec![0; expected.len()];
        backend.read_words_into(&result, 0, &mut actual).map_err(|e| format!("{e}"))?;
        if actual != expected {
            return Err(format!("typed {width:?} fresh output"));
        }
        let mut preserved = vec![0; input.len()];
        backend.read_words_into(&source, 0, &mut preserved).map_err(|e| format!("{e}"))?;
        if preserved != input {
            return Err(String::from("typed copy changed source, padding, or guards"));
        }
        for count in [1, 3, 127, 128, 129] {
            let bits = if width == Width::Eight {
                0x0123_4567_89ab_cdefu64
            } else {
                0x89ab_cdefu64 & ((1u64 << (size * 8)) - 1)
            };
            let layout = Layout::contiguous(Shape::new([count])).map_err(|e| format!("{e}"))?;
            let p = layout.storage_parameters(width, Mode::Fill, bits).map_err(|e| format!("{e}"))?;
            let expected = words(&bits.to_le_bytes()[..size].repeat(count));
            let before = backend.service_profile().map_err(|e| format!("{e}"))?;
            let result = backend
                .with_commands(|| backend.produce_words(expected.len(), Kernel::TensorStorage, &[], &p))
                .map_err(|e| format!("{e}"))?;
            full_writer(backend, &before)?;
            let mut actual = vec![0; expected.len()];
            backend.read_words_into(&result, 0, &mut actual).map_err(|e| format!("{e}"))?;
            if actual != expected {
                return Err(format!("typed fill {width:?} count={count}"));
            }
        }
    }
    let raw: Vec<_> = (0..129u16).flat_map(u16::to_le_bytes).collect();
    let input = backend.upload_bytes(&raw).map_err(|e| format!("{e}"))?;
    let layout = Layout::contiguous(Shape::new([129])).map_err(|e| format!("{e}"))?;
    let p = layout.storage_parameters(Width::Two, Mode::Copy, 0).map_err(|e| format!("{e}"))?;
    let output = backend
        .with_commands(|| {
            let output = backend.produce_words(65, Kernel::TensorStorage, &[(&input, 0)], &p)?;
            drop(input);
            Ok::<_, Error>(output)
        })
        .map_err(|e| format!("{e}"))?;
    let mut actual = vec![0; 65];
    backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
    if actual != words(&raw) {
        return Err(String::from("packed storage queued source lifetime"));
    }
    let incompatible = NativeTensor::from_data(TensorData::from([0i32; 3])).map_err(|e| format!("{e}"))?;
    for dtype in [DType::F16, DType::BF16, DType::F64, DType::I64, DType::U64] {
        let source = NativeTensor::from_data(TensorData::from_bytes_vec(vec![0; 3 * dtype.size()], Shape::new([3]), dtype)).map_err(|e| format!("{e}"))?;
        let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
        if !matches!(source.binary(&incompatible, Elementwise::Add), Err(Error::InvalidParameter))
            || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
        {
            return Err(String::from("mixed typed arithmetic dispatched a word kernel"));
        }
    }
    crate::report(format_args!(
        "NVIDIA_BURN_STORAGE_OK widths=1,2,4,8 guards=ok tails=zero full_writer=verified queued_lifetime=verified resident=true unsupported_arithmetic=no_launch"
    ));
    Ok(())
}
