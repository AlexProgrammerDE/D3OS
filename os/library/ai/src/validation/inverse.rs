//! Public inverse math with independent libm and analytic derivative references.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{DType, TensorData},
    tensor::{Device, Tensor},
};

#[derive(Clone, Copy, Debug)]
pub enum Operation {
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
}
pub const OPERATIONS: [Operation; 6] = [
    Operation::Acos,
    Operation::Acosh,
    Operation::Asin,
    Operation::Asinh,
    Operation::Atan,
    Operation::Atanh,
];
#[derive(Clone, Copy, Debug)]
pub enum Domain {
    Ieee,
    Finite,
}
pub struct Report {
    pub unary_inputs: usize,
    pub atan2_pairs: usize,
    pub gradients: usize,
}

pub fn reference(op: Operation, x: f32) -> f32 {
    let x = f64::from(x);
    (match op {
        Operation::Acos => libm::acos(x),
        // The pinned libm also needs an explicit real-domain guard here.
        Operation::Acosh if x < 1.0 => f64::NAN,
        Operation::Acosh => libm::acosh(x),
        Operation::Asin => libm::asin(x),
        Operation::Asinh => libm::asinh(x),
        Operation::Atan => libm::atan(x),
        Operation::Atanh => libm::atanh(x),
    }) as f32
}
pub fn atan2_reference(y: f32, x: f32) -> f32 {
    libm::atan2(f64::from(y), f64::from(x)) as f32
}
fn apply<const D: usize>(op: Operation, t: Tensor<D>) -> Tensor<D> {
    match op {
        Operation::Acos => t.acos(),
        Operation::Acosh => t.acosh(),
        Operation::Asin => t.asin(),
        Operation::Asinh => t.asinh(),
        Operation::Atan => t.atan(),
        Operation::Atanh => t.atanh(),
    }
}
pub fn inputs(op: Operation, domain: Domain) -> Vec<f32> {
    if matches!(domain, Domain::Finite) {
        return (0..512)
            .map(|i| match op {
                Operation::Acos | Operation::Asin | Operation::Atanh => (i as f32 - 256.0) / 512.0,
                Operation::Acosh => 1.0 + (i as f32 + 1.0) / 128.0,
                _ => (i as f32 - 256.0) / 32.0,
            })
            .collect();
    }
    let mut values = Vec::new();
    for sign in [0, 0x8000_0000] {
        for exponent in 0..256 {
            for fraction in [0, 1, 0x3f_ffff, 0x7f_ffff] {
                values.push(f32::from_bits(sign | exponent << 23 | fraction));
            }
        }
    }
    values.extend((-256..=256).map(|i| i as f32 / 256.0));
    for center in [0x3980_0000u32, 0x3f00_0000, 0x3f80_0000, 0x4000_0000, 0x4580_0000] {
        for bits in center - 16..=center + 16 {
            for sign in [0, 0x8000_0000] {
                values.push(f32::from_bits(bits | sign));
            }
        }
    }
    let mut state = 0x7a1c_9453u32;
    for _ in 0..8192 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        values.push(f32::from_bits(state));
    }
    values.resize(values.len().next_multiple_of(2), 0.0);
    values
}
pub fn atan2_pairs(domain: Domain) -> (Vec<f32>, Vec<f32>) {
    let a = inputs(Operation::Atan, domain);
    let mut b: Vec<_> = (0..a.len()).map(|i| a[(i * 137 + 17) % a.len()]).collect();
    let mut a = a;
    let special: &[f32] = if matches!(domain, Domain::Finite) {
        &[-8.0, -1.0, -0.5, 0.5, 1.0, 8.0]
    } else {
        &[
            -f32::INFINITY,
            -f32::MAX,
            -1.0,
            -f32::MIN_POSITIVE,
            -f32::from_bits(1),
            -0.0,
            0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1.0,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
            f32::from_bits(0x7f80_0001),
        ]
    };
    for &y in special {
        for &x in special {
            a.push(y);
            b.push(x);
        }
    }
    (a, b)
}
pub fn check(actual: &[f32], expected: &[f32], context: &str) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("{context}: length"));
    }
    let ordered = |v: u32| if v & 0x8000_0000 == 0 { v | 0x8000_0000 } else { !v };
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()) {
            continue;
        }
        if a.is_finite() && e.is_finite() && a != 0.0 && e != 0.0 && ordered(a.to_bits()).abs_diff(ordered(e.to_bits())) <= 8 {
            continue;
        }
        return Err(format!(
            "{context}[{i}] actual={a:?} ({:08x}) expected={e:?} ({:08x})",
            a.to_bits(),
            e.to_bits()
        ));
    }
    Ok(())
}
fn read<const D: usize>(t: Tensor<D>) -> Result<Vec<f32>, String> {
    if t.dtype() != DType::F32 {
        return Err(String::from("inverse output dtype"));
    }
    t.try_into_vec_as::<f32>().map_err(|e| format!("inverse read: {e:?}"))
}
fn compare_gradient(a: &[f32], e: &[f64], context: &str) -> Result<(), String> {
    if a.len() != e.len() {
        return Err(format!("{context}: length"));
    }
    for (i, (&a, &e)) in a.iter().zip(e).enumerate() {
        if !a.is_finite() || (f64::from(a) - e).abs() > 2e-5 * e.abs() + 1e-6 {
            return Err(format!("{context}[{i}]: {a} != {e}"));
        }
    }
    Ok(())
}
fn gradients(device: &Device) -> Result<usize, String> {
    let device = device.clone().autodiff();
    for op in OPERATIONS {
        let x: Vec<_> = (0..64)
            .map(|i| match op {
                Operation::Acos | Operation::Asin | Operation::Atanh => (i as f32 - 32.0) / 64.0,
                Operation::Acosh => 1.25 + i as f32 / 16.0,
                _ => (i as f32 - 32.0) / 8.0,
            })
            .collect();
        let weights: Vec<_> = (0..64).map(|i| (i % 7 + 1) as f32 / 8.0).collect();
        let leaf = Tensor::<2>::from_data(TensorData::new(x.clone(), [2, 32]), &device).require_grad();
        let result = apply(op, leaf.clone().flip([0, 1]));
        let grads = (result * Tensor::<2>::from_data(TensorData::new(weights.clone(), [2, 32]), &device))
            .sum()
            .backward();
        let actual = read(leaf.grad(&grads).ok_or_else(|| format!("missing {op:?} gradient"))?)?;
        let expected: Vec<_> = x
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let x = f64::from(x);
                let d = match op {
                    Operation::Acos => -1.0 / libm::sqrt(1.0 - x * x),
                    Operation::Asin => 1.0 / libm::sqrt(1.0 - x * x),
                    Operation::Acosh => 1.0 / libm::sqrt(x * x - 1.0),
                    Operation::Asinh => 1.0 / libm::sqrt(x * x + 1.0),
                    Operation::Atan => 1.0 / (1.0 + x * x),
                    Operation::Atanh => 1.0 / (1.0 - x * x),
                };
                d * f64::from(weights[63 - i])
            })
            .collect();
        compare_gradient(&actual, &expected, &format!("{op:?} gradient"))?;
    }
    for (ry, rx) in [(2, 2), (2, 1), (1, 2)] {
        let y = [-3.125f32, 1.75, -0.625, 0.375, 2.125, 4.75];
        let x = [2.5f32, -4.5, 1.5, -2.5, 3.5, -1.5];
        let w = [0.25f32, 0.5, 0.75, 1.0, 1.25, 1.5];
        let a = Tensor::<2>::from_data(TensorData::new(y[..ry * 3].to_vec(), [ry, 3]), &device).require_grad();
        let b = Tensor::<2>::from_data(TensorData::new(x[..rx * 3].to_vec(), [rx, 3]), &device).require_grad();
        let result = a.clone().flip([0, 1]).atan2(b.clone().flip([0, 1]));
        let grads = (result * Tensor::<2>::from_data(TensorData::new(w.to_vec(), [2, 3]), &device)).sum().backward();
        let mut ey = vec![0f64; ry * 3];
        let mut ex = vec![0f64; rx * 3];
        for row in 0..2 {
            for col in 0..3 {
                let iy = (ry - 1 - row % ry) * 3 + 2 - col;
                let ix = (rx - 1 - row % rx) * 3 + 2 - col;
                let y = f64::from(y[iy]);
                let x = f64::from(x[ix]);
                let d = x * x + y * y;
                let weight = f64::from(w[row * 3 + col]);
                ey[iy] += weight * x / d;
                ex[ix] -= weight * y / d;
            }
        }
        compare_gradient(&read(a.grad(&grads).ok_or("missing atan2 y gradient")?)?, &ey, "atan2 y gradient")?;
        compare_gradient(&read(b.grad(&grads).ok_or("missing atan2 x gradient")?)?, &ex, "atan2 x gradient")?;
    }
    Ok(12)
}
fn unary_cases(device: &Device, op: Operation, domain: Domain) -> Result<usize, String> {
    let x = inputs(op, domain);
    let columns = x.len() / 2;
    let input = Tensor::<2>::from_data(TensorData::new(x.clone(), [2, columns]), device);
    let expected: Vec<_> = x.iter().map(|&x| reference(op, x)).collect();
    let output = apply(op, input.clone());
    if output.shape().as_slice() != [2, columns] {
        return Err(String::from("inverse shape"));
    }
    check(&read(output)?, &expected, &format!("{op:?}"))?;
    let expected: Vec<_> = (0..x.len()).map(|i| reference(op, x[(1 - i % 2) * columns + columns - 1 - i / 2])).collect();
    let view = apply(op, input.clone().transpose().flip([0, 1]));
    if view.shape().as_slice() != [columns, 2] {
        return Err(String::from("inverse view shape"));
    }
    check(&read(view)?, &expected, &format!("{op:?} signed view"))?;
    if read(input)?.iter().zip(&x).any(|(a, b)| a.to_bits() != b.to_bits()) {
        return Err(String::from("inverse changed shared input"));
    }
    if !read(apply(op, Tensor::<2>::zeros([0, 3], device)))?.is_empty() {
        return Err(String::from("inverse empty output"));
    }
    Ok(x.len())
}
fn atan2_cases(device: &Device, domain: Domain) -> Result<usize, String> {
    let (y, x) = atan2_pairs(domain);
    let columns = y.len() / 2;
    let a = Tensor::<2>::from_data(TensorData::new(y.clone(), [2, columns]), device);
    let b = Tensor::<2>::from_data(TensorData::new(x.clone(), [2, columns]), device);
    let expected: Vec<_> = y.iter().zip(&x).map(|(&y, &x)| atan2_reference(y, x)).collect();
    check(&read(a.clone().atan2(b.clone()))?, &expected, "atan2")?;
    let expected: Vec<_> = (0..y.len())
        .map(|i| {
            let j = (1 - i % 2) * columns + columns - 1 - i / 2;
            atan2_reference(y[j], x[j])
        })
        .collect();
    check(
        &read(a.clone().transpose().flip([0, 1]).atan2(b.clone().transpose().flip([0, 1])))?,
        &expected,
        "atan2 signed view",
    )?;
    for (t, original) in [(a, &y), (b, &x)] {
        if read(t)?.iter().zip(original).any(|(a, b)| a.to_bits() != b.to_bits()) {
            return Err(String::from("atan2 changed shared source"));
        }
    }
    for (ry, rx) in [(2, 1), (1, 2)] {
        let a = Tensor::<2>::from_data(TensorData::new(vec![0.5f32; ry * 3], [ry, 3]), device);
        let b = Tensor::<2>::from_data(TensorData::new(vec![-2f32; rx * 3], [rx, 3]), device);
        let out = a.atan2(b);
        if out.shape().as_slice() != [2, 3] {
            return Err(String::from("atan2 broadcast shape"));
        }
        check(&read(out)?, &[atan2_reference(0.5, -2.0); 6], "atan2 broadcast")?;
    }
    if !read(Tensor::<2>::zeros([0, 3], device).atan2(Tensor::<2>::ones([1, 3], device)))?.is_empty() {
        return Err(String::from("atan2 empty output"));
    }
    Ok(y.len())
}
pub struct Diagnostic {
    pub operations: Vec<(String, Result<usize, String>)>,
    pub gradients: Result<usize, String>,
}
pub fn diagnose(device: &Device, domain: Domain) -> Diagnostic {
    let mut operations: Vec<_> = OPERATIONS.into_iter().map(|op| (format!("{op:?}"), unary_cases(device, op, domain))).collect();
    operations.push((String::from("Atan2"), atan2_cases(device, domain)));
    Diagnostic {
        operations,
        gradients: gradients(device),
    }
}
pub fn run(device: &Device, domain: Domain) -> Result<Report, String> {
    let mut count = 0;
    for op in OPERATIONS {
        count = unary_cases(device, op, domain)?;
    }
    Ok(Report {
        unary_inputs: count,
        atan2_pairs: atan2_cases(device, domain)?,
        gradients: gradients(device)?,
    })
}
