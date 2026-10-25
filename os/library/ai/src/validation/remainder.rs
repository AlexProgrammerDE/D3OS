//! Public FP32 floor remainder with an independent libm reduction reference.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::TensorData,
    tensor::{Device, Tensor},
};

#[derive(Clone, Copy, Debug)]
pub enum Domain {
    Ieee,
    Finite,
    FiniteWide,
}

pub struct Report {
    pub pairs: usize,
    pub scalar_cases: usize,
    pub gradients: usize,
}

pub fn reference(x: f32, y: f32) -> f32 {
    let r = libm::fmodf(x, y);
    if r != 0.0 && r.is_sign_negative() != y.is_sign_negative() { r + y } else { r }
}

pub fn inputs() -> Vec<f32> {
    let mut values = Vec::with_capacity(2048);
    for sign in [0, 0x8000_0000] {
        for exponent in 0..256 {
            for fraction in [0, 1, 0x3f_ffff, 0x7f_ffff] {
                values.push(f32::from_bits(sign | exponent << 23 | fraction));
            }
        }
    }
    values
}

pub fn pairs(domain: Domain) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::new();
    let mut b = Vec::new();
    match domain {
        Domain::Finite => {
            for i in 0..512 {
                a.push((i as f32 - 256.0) * 0.125);
                b.push([-3.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 3.0][i % 8]);
            }
        }
        Domain::Ieee | Domain::FiniteWide => {
            for x in inputs() {
                for bits in [
                    0,
                    0x8000_0000,
                    1,
                    0x8000_0001,
                    0x7f_ffff,
                    0x807f_ffff,
                    0x80_0000,
                    0x8080_0000,
                    0x3eaa_aaab,
                    0xbeaa_aaab,
                    0x3f00_0000,
                    0xbf00_0000,
                    0x3f80_0000,
                    0xbf80_0000,
                    0x4040_0000,
                    0xc040_0000,
                    0x4b00_0001,
                    0xcb00_0001,
                    0x7f7f_ffff,
                    0xff7f_ffff,
                    0x7f80_0000,
                    0xff80_0000,
                    0x7fc0_1234,
                    0xffc0_5678,
                ] {
                    a.push(x);
                    b.push(f32::from_bits(bits));
                }
            }
            let mut state = 0x8a5c_19e3u32;
            for _ in 0..4096 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                a.push(f32::from_bits(state));
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                b.push(f32::from_bits(state));
            }
        }
    }
    if matches!(domain, Domain::FiniteWide) {
        let retained: Vec<_> = a.into_iter().zip(b).filter(|(a, b)| a.is_finite() && b.is_finite() && *b != 0.0).collect();
        let (mut a, mut b): (Vec<_>, Vec<_>) = retained.into_iter().unzip();
        if !a.len().is_multiple_of(2) {
            a.push(0.0);
            b.push(1.0);
        }
        return (a, b);
    }
    (a, b)
}

pub fn check(actual: &[f32], a: &[f32], b: &[f32], exact_zero: bool) -> Result<(), String> {
    if actual.len() != a.len() || actual.len() != b.len() {
        return Err(String::from("remainder result length"));
    }
    for (i, ((&actual, &x), &y)) in actual.iter().zip(a).zip(b).enumerate() {
        let expected = reference(x, y);
        if actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()) || (!exact_zero && actual == 0.0 && expected == 0.0) {
            continue;
        }
        return Err(format!(
            "remainder[{i}] x={x:?} ({:08x}) y={y:?} ({:08x}) actual={actual:?} ({:08x}) expected={expected:?} ({:08x})",
            x.to_bits(),
            y.to_bits(),
            actual.to_bits(),
            expected.to_bits()
        ));
    }
    Ok(())
}

fn values(tensor: Tensor<2>, shape: [usize; 2]) -> Result<Vec<f32>, String> {
    if tensor.shape().as_slice() != shape || tensor.dtype() != burn::backend::DType::F32 {
        return Err(String::from("remainder shape or dtype"));
    }
    tensor.try_into_vec_as::<f32>().map_err(|e| format!("remainder read: {e:?}"))
}

fn gradients(device: &Device) -> Result<usize, String> {
    let device = device.clone().autodiff();
    let a = [-5.375f32, -2.875, -0.375, 0.625, 3.125, 5.625];
    let b = [1.5f32, -2.5, 3.5, -1.5, 2.5, -3.5];
    let weights = [0.25f32, 0.5, 0.75, 1.0, 1.25, 1.5];
    for (rows_a, rows_b) in [(2, 2), (2, 1), (1, 2)] {
        let lhs = Tensor::<2>::from_data(TensorData::new(a[..rows_a * 3].to_vec(), [rows_a, 3]), &device).require_grad();
        let rhs = Tensor::<2>::from_data(TensorData::new(b[..rows_b * 3].to_vec(), [rows_b, 3]), &device).require_grad();
        let weight = Tensor::<2>::from_data(TensorData::new(weights.to_vec(), [2, 3]), &device);
        let result = lhs.clone().flip([0, 1]).remainder(rhs.clone().flip([0, 1]));
        let grads = (result * weight).sum().backward();
        let ga = values(lhs.grad(&grads).ok_or_else(|| String::from("missing remainder lhs gradient"))?, [rows_a, 3])?;
        let gb = values(rhs.grad(&grads).ok_or_else(|| String::from("missing remainder rhs gradient"))?, [rows_b, 3])?;
        let mut ea = vec![0f32; rows_a * 3];
        let mut eb = vec![0f32; rows_b * 3];
        for row in 0..2 {
            for col in 0..3 {
                let ia = (rows_a - 1 - row % rows_a) * 3 + 2 - col;
                let ib = (rows_b - 1 - row % rows_b) * 3 + 2 - col;
                let w = weights[row * 3 + col];
                ea[ia] += w;
                eb[ib] += -libm::floor(f64::from(a[ia]) / f64::from(b[ib])) as f32 * w;
            }
        }
        if ga != ea || gb != eb {
            return Err(format!(
                "remainder broadcast gradients rows={rows_a}/{rows_b} lhs={ga:?} != {ea:?} rhs={gb:?} != {eb:?}"
            ));
        }
    }
    for divisor in [-2.5f32, 2.5] {
        let lhs = Tensor::<2>::from_data(TensorData::new(a.to_vec(), [2, 3]), &device).require_grad();
        let weight = Tensor::<2>::from_data(TensorData::new(weights.to_vec(), [2, 3]), &device);
        let grads = (lhs.clone().flip([0, 1]).remainder_scalar(divisor) * weight).sum().backward();
        let actual = values(lhs.grad(&grads).ok_or_else(|| String::from("missing scalar remainder gradient"))?, [2, 3])?;
        if actual != weights.into_iter().rev().collect::<Vec<_>>() {
            return Err(String::from("scalar remainder gradient"));
        }
    }
    Ok(8)
}

pub fn run(device: &Device, domain: Domain) -> Result<Report, String> {
    let (a, b) = pairs(domain);
    let columns = a.len() / 2;
    let lhs = Tensor::<2>::from_data(TensorData::new(a.clone(), [2, columns]), device);
    let rhs = Tensor::<2>::from_data(TensorData::new(b.clone(), [2, columns]), device);
    let output = values(lhs.clone().remainder(rhs.clone()), [2, columns])?;
    check(&output, &a, &b, false)?;
    let order: Vec<_> = (0..a.len()).map(|i| (1 - i % 2) * columns + columns - 1 - i / 2).collect();
    let va: Vec<_> = order.iter().map(|&i| a[i]).collect();
    let vb: Vec<_> = order.iter().map(|&i| b[i]).collect();
    check(
        &values(
            lhs.clone().transpose().flip([0, 1]).remainder(rhs.clone().transpose().flip([0, 1])),
            [columns, 2],
        )?,
        &va,
        &vb,
        false,
    )?;
    for (tensor, original) in [(lhs, &a), (rhs, &b)] {
        if values(tensor, [2, columns])?.iter().zip(original).any(|(a, b)| a.to_bits() != b.to_bits()) {
            return Err(String::from("remainder changed shared input"));
        }
    }
    let scalar_inputs = match domain {
        Domain::Ieee => inputs(),
        Domain::FiniteWide => inputs().into_iter().filter(|v| v.is_finite()).collect(),
        Domain::Finite => a.clone(),
    };
    let input = Tensor::<2>::from_data(TensorData::new(scalar_inputs.clone(), [2, scalar_inputs.len() / 2]), device);
    let divisors: &[f32] = match domain {
        Domain::FiniteWide => &[
            -f32::MAX,
            -3.0,
            -0.5,
            -f32::MIN_POSITIVE,
            -f32::from_bits(1),
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            0.5,
            3.0,
            f32::MAX,
        ],
        Domain::Ieee => &[
            -f32::INFINITY,
            -f32::MAX,
            -3.0,
            -0.5,
            -f32::MIN_POSITIVE,
            -f32::from_bits(1),
            -0.0,
            0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            0.5,
            3.0,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ],
        Domain::Finite => &[-3.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 3.0],
    };
    for &y in divisors {
        let actual = values(input.clone().remainder_scalar(y), [2, scalar_inputs.len() / 2])?;
        check(&actual, &scalar_inputs, &vec![y; scalar_inputs.len()], false)?;
    }
    // Both broadcast directions use distinct, retained sources.
    for (ra, rb) in [(2, 1), (1, 2)] {
        let x = [-5.375f32, -2.875, -0.375, 0.625, 3.125, 5.625];
        let y = [1.5f32, -2.5, 3.5, -1.5, 2.5, -3.5];
        let lhs = Tensor::<2>::from_data(TensorData::new(x[..ra * 3].to_vec(), [ra, 3]), device).flip([0, 1]);
        let rhs = Tensor::<2>::from_data(TensorData::new(y[..rb * 3].to_vec(), [rb, 3]), device).flip([0, 1]);
        let mut ea = Vec::new();
        let mut eb = Vec::new();
        for row in 0..2 {
            for col in 0..3 {
                ea.push(x[(ra - 1 - row % ra) * 3 + 2 - col]);
                eb.push(y[(rb - 1 - row % rb) * 3 + 2 - col]);
            }
        }
        check(&values(lhs.remainder(rhs), [2, 3])?, &ea, &eb, false)?;
    }
    let empty = Tensor::<2>::zeros([0, 7], device);
    if !values(empty.clone().remainder(Tensor::<2>::ones([1, 7], device)), [0, 7])?.is_empty() || !values(empty.remainder_scalar(3.0), [0, 7])?.is_empty() {
        return Err(String::from("empty remainder"));
    }
    Ok(Report {
        pairs: a.len(),
        scalar_cases: divisors.len(),
        gradients: gradients(device)?,
    })
}
