//! Independent libm references for GPU math and exact IEEE edge cases.
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use d3os_ai::{
    burn::backend::{BoolStore, DType, Shape, TensorData, TensorMetadata},
    native::{NativeTensor, layout::Layout},
};
use d3os_compute::{Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, elementwise::Elementwise};

fn reference(op: Elementwise, x: f32) -> u32 {
    match op {
        Elementwise::Negate => x.to_bits() ^ 0x8000_0000,
        Elementwise::Absolute => x.to_bits() & 0x7fff_ffff,
        Elementwise::Reciprocal => (1.0 / x).to_bits(),
        Elementwise::Sqrt => libm::sqrtf(x).to_bits(),
        Elementwise::Exp => libm::expf(x).to_bits(),
        Elementwise::Log => libm::logf(x).to_bits(),
        Elementwise::Log1p => libm::log1pf(x).to_bits(),
        Elementwise::Tanh => libm::tanhf(x).to_bits(),
        Elementwise::Sin => libm::sinf(x).to_bits(),
        Elementwise::Cos => libm::cosf(x).to_bits(),
        Elementwise::Floor => libm::floorf(x).to_bits(),
        Elementwise::Ceil => libm::ceilf(x).to_bits(),
        Elementwise::Round => libm::roundevenf(x).to_bits(),
        Elementwise::Trunc => libm::truncf(x).to_bits(),
        Elementwise::Sign => (if x > 0.0 {
            1.0f32
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        })
        .to_bits(),
        Elementwise::IsNan => u32::from(x.is_nan()),
        Elementwise::IsInfinite => u32::from(x.is_infinite()),
        Elementwise::IsFinite => u32::from(x.is_finite()),
        Elementwise::Sinh => libm::sinhf(x).to_bits(),
        Elementwise::Cosh => libm::coshf(x).to_bits(),
        Elementwise::Tan => libm::tanf(x).to_bits(),
        _ => unreachable!(),
    }
}

fn check(tensor: &NativeTensor, expected: &[u32], op: Elementwise) -> Result<(), String> {
    let data = tensor.read().map_err(|e| e.to_string())?;
    if data.as_bytes().len() != expected.len() * 4 {
        return Err(format!("{op:?}: output length"));
    }
    let exact = op.boolean()
        || matches!(
            op,
            Elementwise::Negate
                | Elementwise::Absolute
                | Elementwise::Round
                | Elementwise::Floor
                | Elementwise::Ceil
                | Elementwise::Trunc
                | Elementwise::Sign
                | Elementwise::Reciprocal
                | Elementwise::Sqrt
        );
    for (index, (bytes, &expected)) in data.as_bytes().chunks_exact(4).zip(expected).enumerate() {
        let actual = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if actual == expected {
            continue;
        }
        let (a, e) = (f32::from_bits(actual), f32::from_bits(expected));
        if !op.boolean() && a.is_nan() && e.is_nan() && !matches!(op, Elementwise::Negate | Elementwise::Absolute | Elementwise::Round) {
            continue;
        }
        let ordered = |v: u32| if v & 0x8000_0000 == 0 { v | 0x8000_0000 } else { !v };
        if !exact && a.is_finite() && e.is_finite() && a != 0.0 && e != 0.0 && ordered(actual).abs_diff(ordered(expected)) <= 8 {
            continue;
        }
        return Err(format!("{op:?} index={index} actual={actual:#010x} expected={expected:#010x}"));
    }
    Ok(())
}

fn verify_unary(backend: &Nvidia, input: &NativeTensor, values: &[f32]) -> Result<(), String> {
    let before = backend.service_profile().map_err(|e| e.to_string())?;
    let outputs: Vec<_> = abi::elementwise::Elementwise::ALL
        .into_iter()
        .filter(|op| op.unary())
        .map(|op| input.unary(op).map(|t| (op, t)))
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let after = backend.service_profile().map_err(|e| e.to_string())?;
    if [abi::ALLOCATE, abi::UPLOAD, abi::DOWNLOAD]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
        || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != outputs.len() as u64
    {
        return Err(String::from("unary execution transferred or cleared storage"));
    }
    for (op, tensor) in outputs {
        let dtype = if op.boolean() { DType::Bool(BoolStore::U32) } else { DType::F32 };
        if tensor.shape() != input.shape() || tensor.dtype() != dtype {
            return Err(format!("{op:?}: shape or dtype"));
        }
        check(&tensor, &values.iter().map(|&x| reference(op, x)).collect::<Vec<_>>(), op)?;
    }
    Ok(())
}

fn verify_power(backend: &Nvidia, input: &NativeTensor, ordered: &[f32]) -> Result<(), String> {
    let bases = [-f32::INFINITY, -8.0, -1.0, -0.0, 0.0, 0.125, 1.0, 8.0, f32::INFINITY, f32::NAN];
    let exponents = [-f32::INFINITY, -3.0, -2.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0, 3.0, f32::INFINITY, f32::NAN];
    let a = NativeTensor::from_data(TensorData::new(bases.to_vec(), [bases.len(), 1])).map_err(|e| e.to_string())?;
    let b = NativeTensor::from_data(TensorData::new(exponents.to_vec(), [1, exponents.len()])).map_err(|e| e.to_string())?;
    let before = backend.service_profile().map_err(|e| e.to_string())?;
    let result = a.binary(&b, Elementwise::Power).map_err(|e| e.to_string())?;
    let after = backend.service_profile().map_err(|e| e.to_string())?;
    if [abi::UPLOAD, abi::DOWNLOAD]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
    {
        return Err(String::from("power execution transferred storage"));
    }
    let expected: Vec<_> = bases.into_iter().flat_map(|a| exponents.map(|b| libm::powf(a, b).to_bits())).collect();
    check(&result, &expected, Elementwise::Power)?;
    for exponent in [-3.0, -1.0, 0.0, 0.5, 2.0, 3.0] {
        let result = input.scalar(Elementwise::Power, exponent).map_err(|e| e.to_string())?;
        let expected: Vec<_> = ordered.iter().map(|&x| libm::powf(x, exponent).to_bits()).collect();
        check(&result, &expected, Elementwise::Power)?;
    }
    Ok(())
}

fn verify_fill(backend: &Nvidia) -> Result<(), Error> {
    let cases = [
        (DType::F32, 0x8000_0000),
        (DType::F32, 0x7fc0_1234),
        (DType::F32, 0x7f80_0000),
        (DType::I32, 0x8000_0000),
        (DType::I32, 0x7fff_ffff),
        (DType::Bool(BoolStore::U32), 0),
        (DType::Bool(BoolStore::U32), 1),
    ];
    let before = backend.service_profile()?;
    let outputs: Vec<_> = cases
        .into_iter()
        .map(|(dtype, word)| NativeTensor::full(Shape::new([3, 257]), dtype, word))
        .collect::<Result<_, _>>()?;
    let after = backend.service_profile()?;
    if [abi::ALLOCATE, abi::UPLOAD, abi::DOWNLOAD]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
        || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != cases.len() as u64
    {
        return Err(Error::DeviceFault);
    }
    for (tensor, (dtype, word)) in outputs.iter().zip(cases) {
        let expected: Vec<_> = (0..771).flat_map(|_| word.to_le_bytes()).collect();
        if tensor.dtype() != dtype || tensor.read()?.as_bytes() != expected {
            return Err(Error::DeviceFault);
        }
    }
    let before = backend.profile()?;
    let empty = NativeTensor::full(Shape::new([2, 0]), DType::F32, 1.0f32.to_bits())?;
    if !empty.read()?.as_bytes().is_empty()
        || before[abi::Kernel::Fill as usize].calls != backend.profile()?[abi::Kernel::Fill as usize].calls
        || !matches!(
            NativeTensor::full(Shape::new([3]), DType::Bool(BoolStore::U32), 2),
            Err(Error::InvalidParameter)
        )
    {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn verify_guards(backend: &Nvidia) -> Result<(), Error> {
    let guarded = backend.upload_words(&vec![0x6bad_beef; 259])?;
    backend.dispatch_words(abi::Kernel::Fill, &[(&guarded, 1)], &[257, 0x7fc0_1234])?;
    let mut actual = vec![0; 259];
    backend.read_words_into(&guarded, 0, &mut actual)?;
    let mut expected = vec![0x6bad_beef; 259];
    expected[1..258].fill(0x7fc0_1234);
    if actual != expected {
        return Err(Error::DeviceFault);
    }
    let values: Vec<_> = (0..257).map(|i| (i as f32 / 4.0 - 32.0).to_bits()).collect();
    let mut words = vec![0x6bad_beef];
    words.extend(&values);
    words.push(0x6bad_beef);
    let input = backend.upload_words(&words)?;
    let layout = Layout::contiguous(Shape::new([257]))?.flip(&[0])?;
    let parameters = layout.elementwise_parameters(&layout, Elementwise::Round, 0, false)?;
    backend.dispatch_words(abi::Kernel::TensorElementwise, &[(&input, 1), (&input, 1), (&guarded, 1)], &parameters)?;
    expected[1..258].copy_from_slice(&values.iter().rev().map(|&v| libm::roundevenf(f32::from_bits(v)).to_bits()).collect::<Vec<_>>());
    backend.read_words_into(&guarded, 0, &mut actual)?;
    if actual != expected {
        return Err(Error::DeviceFault);
    }
    let mut invalid = parameters;
    invalid[2] |= abi::elementwise::SCALAR_RHS as usize;
    if backend.dispatch_words(abi::Kernel::TensorElementwise, &[(&input, 1), (&input, 1), (&guarded, 1)], &invalid) != Err(Error::InvalidParameter) {
        return Err(Error::DeviceFault);
    }
    backend.read_words_into(&guarded, 0, &mut actual)?;
    if actual != expected {
        return Err(Error::DeviceFault);
    }
    backend.read_words_into(&input, 0, &mut actual)?;
    if actual != words {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let values: Vec<_> = (0..3 * 257).map(|i| (i % 257) as f32 / 16.0 - 8.0).collect();
    let original = NativeTensor::from_data(TensorData::new(values.clone(), [3, 257])).map_err(|e| e.to_string())?;
    let input = original.clone().permute(&[1, 0]).and_then(|t| t.flip(&[0])).map_err(|e| e.to_string())?;
    let ordered: Vec<_> = (0..771).map(|i| values[(i % 3) * 257 + 256 - i / 3]).collect();
    verify_unary(backend, &input, &ordered)?;
    verify_power(backend, &input, &ordered)?;
    let mut edges = Vec::new();
    for sign in [0, 0x8000_0000] {
        for exponent in 0..256 {
            for fraction in [0, 1, 0x3f_ffff, 0x7f_ffff] {
                edges.push(f32::from_bits(sign | exponent << 23 | fraction));
            }
        }
        for n in 0..128 {
            let bits = (n as f32 + 0.5).to_bits();
            edges.extend([bits - 1, bits, bits + 1].map(|b| f32::from_bits(b | sign)));
        }
    }
    let edge_input = NativeTensor::from_data(TensorData::new(edges.clone(), [edges.len()])).map_err(|e| e.to_string())?;
    verify_unary(backend, &edge_input, &edges)?;
    verify_fill(backend).map_err(|e| e.to_string())?;
    verify_guards(backend).map_err(|e| e.to_string())?;
    if original
        .read()
        .map_err(|e| e.to_string())?
        .as_slice::<f32>()
        .map_err(|_| String::from("input dtype"))?
        != values
    {
        return Err(String::from("unary math modified shared input"));
    }
    let empty = NativeTensor::full(Shape::new([2, 0]), DType::F32, 0).map_err(|e| e.to_string())?;
    let before = backend.queue_profile().map_err(|e| e.to_string())?;
    for op in Elementwise::ALL.into_iter().filter(|op| op.unary()) {
        let output = empty.unary(op).map_err(|e| e.to_string())?;
        if !output.layout().is_empty() {
            return Err(format!("{op:?}: empty shape"));
        }
    }
    if before.launches != backend.queue_profile().map_err(|e| e.to_string())?.launches {
        return Err(String::from("empty unary math dispatched GPU work"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_MATH_OK unary=21 power=ok fill=ok edge_values={} rounding=ties_even guards=ok resident=ok backend=nvidia",
        edges.len()
    ));
    Ok(())
}
