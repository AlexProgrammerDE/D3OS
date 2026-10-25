//! Public erf/GELU conformance, independent of the native GPU approximation.
use alloc::{format, string::String, vec::Vec};
use burn::{
    backend::TensorData,
    tensor::{Device, Tensor, activation::gelu},
};

fn check_erf(actual: &[f32], input: &[f32]) -> Result<(), String> {
    if actual.len() != input.len() {
        return Err(String::from("erf output length"));
    }
    let ordered = |bits: u32| if bits & 0x8000_0000 == 0 { bits | 0x8000_0000 } else { !bits };
    for (i, (&actual, &x)) in actual.iter().zip(input).enumerate() {
        let expected = libm::erff(x);
        if actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()) {
            continue;
        }
        if actual.is_finite()
            && expected.is_finite()
            && actual != 0.0
            && expected != 0.0
            && ordered(actual.to_bits()).abs_diff(ordered(expected.to_bits())) <= 8
        {
            continue;
        }
        return Err(format!("erf[{i}] x={x:?}: {actual:?} != {expected:?}"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub enum GeluGradient {
    BurnDefault,
    Analytic,
}

fn inputs() -> Vec<f32> {
    let mut values: Vec<_> = (-256..=256).map(|i| i as f32 / 32.0).collect();
    for sign in [0, 0x8000_0000] {
        for exponent in 0..256 {
            for fraction in [0, 1, 0x3f_ffff, 0x7f_ffff] {
                values.push(f32::from_bits(sign | exponent << 23 | fraction));
            }
        }
        for center in [0x3180_0000u32, 0x3f58_0000, 0x3fa0_0000, 0x4036_db6d, 0x40c0_0000] {
            for bits in center - 4..=center + 4 {
                values.push(f32::from_bits(sign | bits));
            }
        }
    }
    values.resize(values.len().next_multiple_of(16), 0.0);
    values
}

pub fn run(device: &Device, gelu_gradient: GeluGradient) -> Result<(), String> {
    let values = inputs();
    let input = Tensor::<2>::from_data(TensorData::new(values.clone(), [2, values.len() / 2]), device);
    let actual = input.clone().erf().try_into_vec_as::<f32>().map_err(|e| format!("public erf read: {e:?}"))?;
    check_erf(&actual, &values)?;
    let columns = values.len() / 2;
    let view = input.clone().transpose().flip([0]);
    let expected: Vec<_> = (0..values.len()).map(|i| values[(i % 2) * columns + columns - 1 - i / 2]).collect();
    let actual = view.erf().try_into_vec_as::<f32>().map_err(|e| format!("strided erf read: {e:?}"))?;
    check_erf(&actual, &expected)?;
    let preserved = input.try_into_vec_as::<f32>().map_err(|e| format!("shared erf source: {e:?}"))?;
    if preserved.iter().zip(&values).any(|(a, b)| a.to_bits() != b.to_bits()) {
        return Err(String::from("erf changed shared source bits"));
    }
    if !Tensor::<2>::zeros([0, 7], device)
        .erf()
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("empty erf: {e:?}"))?
        .is_empty()
    {
        return Err(String::from("empty erf result"));
    }

    gelu_conformance(device, gelu_gradient)
}

pub fn gelu_conformance(device: &Device, gelu_gradient: GeluGradient) -> Result<(), String> {
    let autodiff = device.clone().autodiff();
    let x: Vec<_> = (-256..=256).map(|i| i as f32 / 32.0).collect();
    let erf_input = Tensor::<1>::from_data(TensorData::new(x.clone(), [x.len()]), &autodiff).require_grad();
    let gradients = erf_input.clone().erf().sum().backward();
    let erf_gradient = erf_input
        .grad(&gradients)
        .ok_or_else(|| String::from("missing erf gradient"))?
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("erf gradient read: {e:?}"))?;
    for (i, (&actual, &value)) in erf_gradient.iter().zip(&x).enumerate() {
        let expected = core::f32::consts::FRAC_2_SQRT_PI * libm::expf(-value * value);
        if !actual.is_finite() || (actual - expected).abs() > expected * 3e-6 + f32::MIN_POSITIVE {
            return Err(format!("erf gradient[{i}] {actual} != {expected}"));
        }
    }
    let tensor = Tensor::<1>::from_data(TensorData::new(x.clone(), [x.len()]), &autodiff).require_grad();
    let result = gelu(tensor.clone());
    let forward = result.clone().try_into_vec_as::<f32>().map_err(|e| format!("GELU read: {e:?}"))?;
    let gradients = result.sum().backward();
    let backward = tensor
        .grad(&gradients)
        .ok_or_else(|| String::from("missing GELU gradient"))?
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("GELU gradient read: {e:?}"))?;
    for (i, &x) in x.iter().enumerate() {
        let expected = x * (1.0 + libm::erff(x / core::f32::consts::SQRT_2)) * 0.5;
        // The pinned Burn backward uses the documented tanh approximation,
        // even though its forward uses erf. Match that API, not a new rule.
        let x = f64::from(x);
        let t = libm::tanh(0.0356774 * x * x * x + 0.797885 * x);
        let gradient = match gelu_gradient {
            GeluGradient::BurnDefault => (0.5 * t + (0.0535161 * x * x * x + 0.398942 * x) * (1.0 - t * t) + 0.5) as f32,
            GeluGradient::Analytic => {
                (0.5 * (1.0 + libm::erf(x / core::f64::consts::SQRT_2)) + x * libm::exp(-0.5 * x * x) / libm::sqrt(2.0 * core::f64::consts::PI)) as f32
            }
        };
        if !forward[i].is_finite() || !backward[i].is_finite() || (forward[i] - expected).abs() > 2e-6 || (backward[i] - gradient).abs() > 2e-6 {
            return Err(format!(
                "GELU[{i}] forward={} expected={expected} backward={} expected={gradient}",
                forward[i], backward[i]
            ));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct ErfProfile {
    pub input_bits: Vec<u32>,
    pub output_bits: Vec<u32>,
    pub strict_mismatches: usize,
    pub special_mismatches: usize,
    pub max_absolute_error: f32,
    pub max_ulp_error: u32,
}

/// Measure the implementation without converting a failed strict bound into
/// a conformance success. The vectors retain the raw evidence for review.
pub fn profile(device: &Device) -> Result<ErfProfile, String> {
    let values = inputs();
    let actual = Tensor::<2>::from_data(TensorData::new(values.clone(), [2, values.len() / 2]), device)
        .erf()
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("erf profile: {e:?}"))?;
    if actual.len() != values.len() {
        return Err(String::from("erf profile output length"));
    }
    let mut report = ErfProfile {
        input_bits: values.iter().map(|v| v.to_bits()).collect(),
        output_bits: actual.iter().map(|v| v.to_bits()).collect(),
        strict_mismatches: 0,
        special_mismatches: 0,
        max_absolute_error: 0.0,
        max_ulp_error: 0,
    };
    let ordered = |bits: u32| if bits & 0x8000_0000 == 0 { bits | 0x8000_0000 } else { !bits };
    for (&x, &actual) in values.iter().zip(&actual) {
        let expected = libm::erff(x);
        if check_erf(&[actual], &[x]).is_err() {
            report.strict_mismatches += 1;
        }
        if actual.is_finite() && expected.is_finite() {
            report.max_absolute_error = report.max_absolute_error.max((actual - expected).abs());
            report.max_ulp_error = report.max_ulp_error.max(ordered(actual.to_bits()).abs_diff(ordered(expected.to_bits())));
            if expected == 0.0 && expected.to_bits() != actual.to_bits() {
                report.special_mismatches += 1;
            }
        } else if !(expected.is_nan() && actual.is_nan()) && actual.to_bits() != expected.to_bits() {
            report.special_mismatches += 1;
        }
    }
    Ok(report)
}
