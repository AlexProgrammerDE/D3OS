//! Random123 anchors plus independent scalar integer and FP64 references.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{BoolStore, DType, Distribution, Shape, TensorMetadata},
    native::{
        NativeDevice, NativeTensor,
        random::{RandomGenerator, RandomState},
    },
};
use d3os_compute::{Backend, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel, random::Mode};

fn philox(mut c: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for _ in 0..10 {
        let x = u64::from(c[0]) * 0xd251_1f53u64;
        let y = u64::from(c[2]) * 0xcd9e_8d57u64;
        c = [(y >> 32) as u32 ^ c[1] ^ key[0], y as u32, (x >> 32) as u32 ^ c[3] ^ key[1], x as u32];
        key[0] = key[0].wrapping_add(0x9e37_79b9);
        key[1] = key[1].wrapping_add(0xbb67_ae85);
    }
    c
}

fn block(state: RandomState, index: usize, retry: u64) -> [u32; 4] {
    let counter = state.next_block + (index / 4) as u64;
    philox(
        [counter as u32, (counter >> 32) as u32, retry as u32, (retry >> 32) as u32],
        [state.seed as u32, (state.seed >> 32) as u32],
    )
}

fn raw(state: RandomState, index: usize, retry: u64) -> u32 {
    block(state, index, retry)[index % 4]
}

fn read(tensor: &NativeTensor) -> Result<Vec<u32>, Error> {
    Ok(tensor
        .read()?
        .as_bytes()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn resident(before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], after: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS]) -> Result<(), Error> {
    if [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
    {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn anchors() -> Result<(), Error> {
    // Published Random123 Philox4x32-10 known-answer vectors.
    for (c, k, want) in [
        ([0; 4], [0; 2], [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]),
        ([u32::MAX; 4], [u32::MAX; 2], [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]),
        (
            [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
            [0xa409_3822, 0x299f_31d0],
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
        ),
    ] {
        if philox(c, k) != want {
            return Err(Error::DeviceFault);
        }
    }
    Ok(())
}

fn words_and_checkpoints(backend: &Nvidia) -> Result<(), Error> {
    let generator = RandomGenerator::new(0);
    for count in [1usize, 2, 3, 4, 5, 127, 128, 129, 515, 4097] {
        for state in [
            RandomState { seed: 0, next_block: 0 },
            RandomState {
                seed: u64::MAX,
                next_block: u32::MAX as u64 - 1,
            },
            RandomState {
                seed: 0x1234_abcd_5678_90ef,
                next_block: u64::MAX - (count as u64).div_ceil(4),
            },
        ] {
            generator.restore(state);
            let before = backend.service_profile()?;
            let output = generator.random(Shape::new([count]), DType::I32, Distribution::Default)?;
            resident(&before, &backend.service_profile()?)?;
            let expected: Vec<_> = (0..count).map(|i| raw(state, i, 0)).collect();
            if read(&output)? != expected
                || output.dtype() != DType::I32
                || generator.state()
                    != (RandomState {
                        seed: state.seed,
                        next_block: state.next_block + (count as u64).div_ceil(4),
                    })
            {
                return Err(Error::DeviceFault);
            }
        }
    }
    generator.seed(19);
    let _first = generator.random(Shape::new([7]), DType::F32, Distribution::Default)?;
    let checkpoint = RandomState::from_bytes(generator.state().to_bytes())?;
    let uniform = generator.random(Shape::new([67]), DType::F32, Distribution::Uniform(-2.0, 5.0))?;
    let normal = generator.random(Shape::new([137]), DType::F32, Distribution::Normal(3.0, 2.0))?;
    let final_state = generator.state();
    generator.restore(checkpoint);
    let restored_uniform = generator.random(Shape::new([67]), DType::F32, Distribution::Uniform(-2.0, 5.0))?;
    let restored_normal = generator.random(Shape::new([137]), DType::F32, Distribution::Normal(3.0, 2.0))?;
    if read(&uniform)? != read(&restored_uniform)? || read(&normal)? != read(&restored_normal)? || final_state != generator.state() {
        return Err(Error::DeviceFault);
    }
    NativeDevice.seed(19);
    let device_checkpoint = NativeDevice.random_state();
    let a = NativeTensor::random(Shape::new([131]), DType::I32, Distribution::Default)?;
    NativeDevice.restore_random_state(device_checkpoint);
    let b = NativeTensor::random(Shape::new([131]), DType::I32, Distribution::Default)?;
    generator.seed(19);
    let c = generator.random(Shape::new([131]), DType::I32, Distribution::Default)?;
    if read(&a)? != read(&b)? || read(&a)? != read(&c)? {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn normal_reference(state: RandomState, index: usize, mean: f64, deviation: f64) -> f64 {
    let pair = (index % 4) / 2;
    let words = block(state, index, 0);
    let u1 = ((words[2 * pair] >> 8) as f64 + 1.0) / 16777216.0;
    let u2 = (words[2 * pair + 1] >> 8) as f64 / 16777216.0;
    let radius = libm::sqrt(-2.0 * libm::log(u1));
    let angle = core::f64::consts::TAU * u2;
    mean + deviation * radius * if index.is_multiple_of(2) { libm::cos(angle) } else { libm::sin(angle) }
}

fn distributions(backend: &Nvidia) -> Result<(), Error> {
    let generator = RandomGenerator::new(0x9912_8893_7690_abcd);
    let state = generator.state();
    for (lower, upper) in [
        (0.0f32, 1.0f32),
        (-17.5, 5.25),
        (-f32::MAX, f32::MAX),
        (0.0, f32::from_bits(1)),
        (-f32::from_bits(2), f32::from_bits(2)),
        (-1.0, f32::from_bits(0xbf7f_ffff)),
    ] {
        generator.restore(state);
        let before = backend.service_profile()?;
        let output = generator.random(Shape::new([17, 241]), DType::F32, Distribution::Uniform(lower as f64, upper as f64))?;
        resident(&before, &backend.service_profile()?)?;
        for (i, word) in read(&output)?.into_iter().enumerate() {
            let actual = f32::from_bits(word);
            let u = (raw(state, i, 0) >> 8) as f64 / 16777216.0;
            let limit = f32::from_bits(abi::random::predecessor(upper.to_bits()));
            let expected = ((lower as f64 * (1.0 - u) + upper as f64 * u) as f32).max(lower).min(limit);
            let tolerance = 3.0 * f32::EPSILON as f64 * (lower.abs() as f64 + upper.abs() as f64) + f32::from_bits(1) as f64;
            if !actual.is_finite() || actual < lower || actual >= upper || (actual as f64 - expected as f64).abs() > tolerance {
                return Err(Error::DeviceFault);
            }
            if lower == 0.0 && upper == 1.0 && actual.to_bits() != (u as f32).to_bits() {
                return Err(Error::DeviceFault);
            }
        }
    }
    for (mean, deviation) in [(0.0f32, 1.0f32), (3.25, 2.75), (-0.0, 0.0), (0.0, f32::from_bits(1))] {
        generator.restore(state);
        let before = backend.service_profile()?;
        let output = generator.random(Shape::new([17, 241]), DType::F32, Distribution::Normal(mean as f64, deviation as f64))?;
        resident(&before, &backend.service_profile()?)?;
        for (i, word) in read(&output)?.into_iter().enumerate() {
            let actual = f32::from_bits(word);
            let expected = normal_reference(state, i, mean as f64, deviation as f64);
            if !actual.is_finite() || (actual as f64 - expected).abs() > 6e-6 * (1.0 + expected.abs()) * deviation as f64 + 2.0 * f32::from_bits(1) as f64 {
                return Err(Error::DeviceFault);
            }
            if deviation == 0.0 && word != mean.to_bits() {
                return Err(Error::DeviceFault);
            }
        }
    }
    for dtype in [DType::I32, DType::F32, DType::Bool(BoolStore::U32)] {
        for probability in [0.0, 1.0, 0.27, 1.0 / 4294967296.0, 1.0 - 1.0 / 4294967296.0] {
            generator.restore(state);
            let threshold = (probability * 4294967296.0) as u64;
            let before = backend.service_profile()?;
            let output = generator.random(Shape::new([4097]), dtype, Distribution::Bernoulli(probability))?;
            resident(&before, &backend.service_profile()?)?;
            let expected: Vec<_> = (0..4097)
                .map(|i| {
                    let value = u32::from(u64::from(raw(state, i, 0)) < threshold);
                    if dtype == DType::F32 { (value as f32).to_bits() } else { value }
                })
                .collect();
            if read(&output)? != expected || output.dtype() != dtype {
                return Err(Error::DeviceFault);
            }
        }
    }
    let mut retries = 0;
    for (low, high) in [(i32::MIN, 1), (-17, 19), (0, 1), (i32::MIN, i32::MAX)] {
        generator.restore(state);
        let before = backend.service_profile()?;
        let output = generator.random(Shape::new([4097]), DType::I32, Distribution::Uniform(low as f64, high as f64))?;
        resident(&before, &backend.service_profile()?)?;
        let width = (i64::from(high) - i64::from(low)) as u32;
        let threshold = width.wrapping_neg() % width;
        for (i, actual) in read(&output)?.into_iter().enumerate() {
            let mut retry = 0;
            let mut value = raw(state, i, retry);
            while value < threshold {
                retry += 1;
                value = raw(state, i, retry);
            }
            retries += retry;
            let expected = (low as u32).wrapping_add(value % width);
            if actual != expected || (actual as i32) < low || (actual as i32) >= high {
                return Err(Error::DeviceFault);
            }
        }
    }
    if retries < 1000 {
        return Err(Error::DeviceFault);
    }
    generator.restore(state);
    let normal = generator.random(Shape::new([129]), DType::I32, Distribution::Normal(0.0, 3.0))?;
    let expected: Vec<_> = (0..129).map(|i| normal_reference(state, i, 0.0, 3.0) as i32 as u32).collect();
    if read(&normal)? != expected {
        return Err(Error::DeviceFault);
    }
    generator.restore(state);
    let boolean = generator.random(Shape::new([129]), DType::Bool(BoolStore::U32), Distribution::Default)?;
    let expected: Vec<_> = (0..129).map(|i| u32::from(raw(state, i, 0) < 0x8000_0000)).collect();
    if read(&boolean)? != expected {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn invalid_and_guards(backend: &Nvidia) -> Result<(), Error> {
    let generator = RandomGenerator::new(9);
    let state = generator.state();
    let before = backend.service_profile()?;
    let empty = generator.random(Shape::new([2, 0, 3]), DType::F32, Distribution::Default)?;
    let after = backend.service_profile()?;
    if !empty.layout().is_empty() || generator.state() != state || after[abi::LAUNCH_ALLOCATE as usize].calls != before[abi::LAUNCH_ALLOCATE as usize].calls {
        return Err(Error::DeviceFault);
    }
    for (dtype, dist) in [
        (DType::F32, Distribution::Uniform(1.0, 1.0)),
        (DType::F32, Distribution::Normal(0.0, -1.0)),
        (DType::I32, Distribution::Uniform(0.0, 0.5)),
        (DType::Bool(BoolStore::U32), Distribution::Uniform(0.0, 1.0)),
        (DType::F64, Distribution::Default),
    ] {
        let before = backend.service_profile()?;
        if generator.random(Shape::new([3]), dtype, dist).is_ok() {
            return Err(Error::DeviceFault);
        }
        resident(&before, &backend.service_profile()?)?;
        if generator.state() != state {
            return Err(Error::DeviceFault);
        }
    }
    generator.restore(RandomState { seed: 9, next_block: u64::MAX });
    if !matches!(generator.random(Shape::new([1]), DType::I32, Distribution::Default), Err(Error::SizeOverflow)) || generator.state().next_block != u64::MAX {
        return Err(Error::DeviceFault);
    }
    let count = 131;
    let guard = 0x6bad_beef;
    let output = backend.upload_words(&vec![guard; count + 5])?;
    for mode in Mode::ALL {
        let args = match mode {
            Mode::Words => [0, 0, 0],
            Mode::Uniform => [0, 1f32.to_bits(), abi::random::predecessor(1f32.to_bits())],
            Mode::Normal => [0, 0, 0],
            Mode::Bernoulli => [0, 1, 0],
            Mode::IntegerUniform => [42, 1, 0],
        };
        let p = [
            count,
            0,
            0,
            u32::MAX as usize,
            0,
            mode as usize,
            args[0] as usize,
            args[1] as usize,
            args[2] as usize,
        ];
        backend.dispatch_words(Kernel::Random, &[(&output, 2)], &p)?;
        let mut actual = vec![0; count + 5];
        backend.read_words_into(&output, 0, &mut actual)?;
        if actual[..2] != [guard; 2] || actual[count + 2..] != [guard; 3] {
            return Err(Error::DeviceFault);
        }
        let state = RandomState {
            seed: 0,
            next_block: u32::MAX as u64,
        };
        for i in 0..count {
            let expected = match mode {
                Mode::Words => raw(state, i, 0),
                Mode::Uniform => ((raw(state, i, 0) >> 8) as f32 / 16777216.0).to_bits(),
                Mode::Normal => 0,
                Mode::Bernoulli => 1,
                Mode::IntegerUniform => 42,
            };
            if actual[2 + i] != expected {
                return Err(Error::DeviceFault);
            }
        }
        backend.with_commands(|| {
            let mut invalid = p;
            invalid[0] = 0;
            if backend.dispatch_words(Kernel::Random, &[(&output, 2)], &invalid).is_ok() {
                return Err(Error::DeviceFault);
            }
            invalid = p;
            invalid[5] = 5;
            if backend.dispatch_words(Kernel::Random, &[(&output, 2)], &invalid).is_ok() {
                return Err(Error::DeviceFault);
            }
            invalid = p;
            invalid[3] = u32::MAX as usize;
            invalid[4] = u32::MAX as usize;
            if backend.dispatch_words(Kernel::Random, &[(&output, 2)], &invalid).is_ok() || backend.dispatch_words(Kernel::Random, &[(&output, 6)], &p).is_ok()
            {
                return Err(Error::DeviceFault);
            }
            Ok(())
        })?;
        let mut after = vec![0; count + 5];
        backend.read_words_into(&output, 0, &mut after)?;
        if after != actual {
            return Err(Error::DeviceFault);
        }
    }
    generator.seed(0);
    let recovered = generator.random(Shape::new([4]), DType::I32, Distribution::Default)?;
    if read(&recovered)? != [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8] {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    anchors().map_err(|e| format!("Philox known-answer reference: {e}"))?;
    words_and_checkpoints(backend).map_err(|e| format!("Philox words and checkpoints: {e}"))?;
    distributions(backend).map_err(|e| format!("Philox distributions: {e}"))?;
    invalid_and_guards(backend).map_err(|e| format!("Philox guards and recovery: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_RANDOM_OK algorithm=philox4x32-10 distributions=5 dtypes=3 checkpoint=exact integer_uniform=unbiased normal_reference=fp64 guards=ok resident=ok backend=nvidia"
    ));
    Ok(())
}
