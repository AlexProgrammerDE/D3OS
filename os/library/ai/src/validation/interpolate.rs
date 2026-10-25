//! Independent FP64 resampling matrices and their transposes for public Burn APIs.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{
        DType, TensorData,
        ops::{InterpolateMode, InterpolateOptions},
    },
    tensor::{Device, Tensor, module::interpolate},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Nearest,
    Exact,
    Linear,
    Cubic,
    Lanczos3,
}
pub const MODES: [Mode; 5] = [Mode::Nearest, Mode::Exact, Mode::Linear, Mode::Cubic, Mode::Lanczos3];
impl Mode {
    pub fn options(self, size: [usize; 2], align: bool) -> InterpolateOptions {
        let mode = match self {
            Self::Nearest => InterpolateMode::Nearest,
            Self::Exact => InterpolateMode::NearestExact,
            Self::Linear => InterpolateMode::Bilinear,
            Self::Cubic => InterpolateMode::Bicubic,
            Self::Lanczos3 => InterpolateMode::Lanczos3,
        };
        let mut options = InterpolateOptions::new(mode).with_align_corners(align);
        options.output_size = Some(size);
        options
    }
    pub fn nearest(self) -> bool {
        matches!(self, Self::Nearest | Self::Exact)
    }
}
#[derive(Clone, Copy)]
pub struct Case {
    pub shape: [usize; 4],
    pub output: [usize; 2],
    pub transpose: bool,
}
pub const CASES: [Case; 10] = [
    Case {
        shape: [2, 3, 3, 5],
        output: [7, 9],
        transpose: false,
    },
    Case {
        shape: [1, 2, 7, 9],
        output: [3, 4],
        transpose: true,
    },
    Case {
        shape: [2, 1, 1, 5],
        output: [4, 1],
        transpose: false,
    },
    Case {
        shape: [1, 2, 5, 1],
        output: [1, 7],
        transpose: false,
    },
    Case {
        shape: [1, 1, 5, 7],
        output: [1, 1],
        transpose: false,
    },
    Case {
        shape: [1, 1, 1, 1],
        output: [3, 5],
        transpose: false,
    },
    Case {
        shape: [2, 3, 5, 7],
        output: [5, 7],
        transpose: false,
    },
    Case {
        shape: [1, 1, 17, 19],
        output: [13, 23],
        transpose: true,
    },
    Case {
        shape: [1, 1, 1, 65537],
        output: [1, 65539],
        transpose: false,
    },
    Case {
        shape: [1, 1, 1, 65539],
        output: [1, 65537],
        transpose: false,
    },
];
fn filter(mode: Mode, x: f64) -> f64 {
    let a = x.abs();
    match mode {
        Mode::Nearest | Mode::Exact => 1.0,
        Mode::Linear => (1.0 - a).max(0.0),
        Mode::Cubic if a < 1.0 => 1.25 * a * a * a - 2.25 * a * a + 1.0,
        Mode::Cubic if a < 2.0 => -0.75 * a * a * a + 3.75 * a * a - 6.0 * a + 3.0,
        Mode::Cubic => 0.0,
        Mode::Lanczos3 if a < 1e-12 => 1.0,
        Mode::Lanczos3 if a < 3.0 => {
            let p = core::f64::consts::PI * x;
            libm::sin(p) / p * libm::sin(p / 3.0) / (p / 3.0)
        }
        Mode::Lanczos3 => 0.0,
    }
}
pub fn weights(input: usize, output: usize, mode: Mode, align: bool) -> Vec<Vec<(usize, f64)>> {
    (0..output)
        .map(|j| {
            let i = input as i128;
            let o = output as i128;
            let j = j as i128;
            let (n, d) = match mode {
                Mode::Nearest => (j * i, o),
                Mode::Exact => ((2 * j + 1) * i, 2 * o),
                _ if align => (j * (i - 1), (o - 1).max(1)),
                _ => ((2 * j + 1) * i - o, 2 * o),
            };
            let whole = n.div_euclid(d);
            let phase = n.rem_euclid(d) as f64 / d as f64;
            let (taps, radius) = match mode {
                Mode::Nearest | Mode::Exact => (1, 0),
                Mode::Linear => (2, 0),
                Mode::Cubic => (4, 1),
                Mode::Lanczos3 => (6, 2),
            };
            let mut row = Vec::new();
            for tap in 0..taps {
                let raw = whole + tap - radius;
                if mode == Mode::Lanczos3 && (raw < 0 || raw >= i) {
                    continue;
                }
                let index = raw.clamp(0, i - 1) as usize;
                let w = filter(mode, (tap - radius) as f64 - phase);
                if w != 0.0 {
                    row.push((index, w));
                }
            }
            if mode == Mode::Lanczos3 {
                let sum: f64 = row.iter().map(|(_, w)| w).sum();
                for (_, w) in &mut row {
                    *w /= sum;
                }
            }
            row
        })
        .collect()
}
pub fn forward(values: &[f32], shape: [usize; 4], size: [usize; 2], mode: Mode, align: bool) -> Vec<f32> {
    let wy = weights(shape[2], size[0], mode, align);
    let wx = weights(shape[3], size[1], mode, align);
    let mut result = Vec::with_capacity(shape[0] * shape[1] * size[0] * size[1]);
    for nc in 0..shape[0] * shape[1] {
        for row in &wy {
            for col in &wx {
                if mode.nearest() {
                    result.push(values[(nc * shape[2] + row[0].0) * shape[3] + col[0].0]);
                } else {
                    let mut sum = 0f64;
                    for &(y, a) in row {
                        for &(x, b) in col {
                            sum += f64::from(values[(nc * shape[2] + y) * shape[3] + x]) * a * b;
                        }
                    }
                    result.push(sum as f32);
                }
            }
        }
    }
    result
}
pub fn backward(gradient: &[f32], shape: [usize; 4], size: [usize; 2], mode: Mode, align: bool) -> Vec<f64> {
    let wy = weights(shape[2], size[0], mode, align);
    let wx = weights(shape[3], size[1], mode, align);
    let mut result = vec![0f64; shape.iter().product()];
    for nc in 0..shape[0] * shape[1] {
        for (oy, row) in wy.iter().enumerate() {
            for (ox, col) in wx.iter().enumerate() {
                let g = f64::from(gradient[(nc * size[0] + oy) * size[1] + ox]);
                for &(y, a) in row {
                    for &(x, b) in col {
                        result[(nc * shape[2] + y) * shape[3] + x] += g * a * b;
                    }
                }
            }
        }
    }
    result
}
pub fn source(shape: [usize; 4]) -> Vec<f32> {
    (0..shape.iter().product()).map(|i| ((i * 17 % 37) as f32 - 18.0) * 0.125).collect()
}
pub fn gradient(shape: [usize; 4]) -> Vec<f32> {
    (0..shape.iter().product()).map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.125).collect()
}
fn order(shape: [usize; 4], transpose: bool) -> ([usize; 4], Vec<usize>) {
    let view = if transpose { [shape[0], shape[1], shape[3], shape[2]] } else { shape };
    let mut result = Vec::new();
    for n in 0..view[0] {
        for c in 0..view[1] {
            for h in 0..view[2] {
                for w in 0..view[3] {
                    let y = if transpose { view[3] - 1 - w } else { view[2] - 1 - h };
                    let x = if transpose { view[2] - 1 - h } else { view[3] - 1 - w };
                    result.push((((shape[0] - 1 - n) * shape[1] + shape[1] - 1 - c) * shape[2] + y) * shape[3] + x);
                }
            }
        }
    }
    (view, result)
}
fn view(input: Tensor<4>, transpose: bool) -> Tensor<4> {
    if transpose {
        input.swap_dims(2, 3).flip([0, 1, 2, 3])
    } else {
        input.flip([0, 1, 2, 3])
    }
}
fn read(t: Tensor<4>, shape: [usize; 4]) -> Result<Vec<f32>, String> {
    if t.shape().as_slice() != shape || t.dtype() != DType::F32 {
        return Err(String::from("interpolation shape or dtype"));
    }
    t.try_into_vec_as::<f32>().map_err(|e| format!("interpolation read: {e:?}"))
}
pub fn check(actual: &[f32], expected: &[f32], exact: bool, context: &str) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("{context}: length"));
    }
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if a.to_bits() == e.to_bits() {
            continue;
        }
        if !exact && a.is_finite() && e.is_finite() && (a - e).abs() <= 2e-5 * e.abs() + 3e-6 {
            continue;
        }
        return Err(format!("{context}[{i}] {a:?} ({:08x}) != {e:?} ({:08x})", a.to_bits(), e.to_bits()));
    }
    Ok(())
}
pub fn check_forward(device: &Device, wide: bool) -> Result<usize, String> {
    let mut cases = 0;
    for &case in &CASES[..if wide { 10 } else { 8 }] {
        let data = source(case.shape);
        let (shape, indices) = order(case.shape, case.transpose);
        let logical: Vec<_> = indices.iter().map(|&i| data[i]).collect();
        let input = Tensor::<4>::from_data(TensorData::new(data.clone(), case.shape), device);
        for mode in MODES {
            for align in [false, true] {
                let output = interpolate(view(input.clone(), case.transpose), mode.options(case.output, align));
                let expected = forward(&logical, shape, case.output, mode, align);
                let outshape = [shape[0], shape[1], case.output[0], case.output[1]];
                check(
                    &read(output, outshape)?,
                    &expected,
                    mode.nearest(),
                    &format!("{mode:?} align={align} {:?}->{:?}", shape, case.output),
                )?;
                cases += 1;
            }
        }
        if read(input, case.shape)?.iter().zip(data).any(|(a, b)| a.to_bits() != b.to_bits()) {
            return Err(String::from("interpolation changed shared source"));
        }
    }
    Ok(cases)
}
pub fn check_gradients(device: &Device, modes: &[Mode]) -> Result<usize, String> {
    let device = device.clone().autodiff();
    let mut cases = 0;
    for case in &CASES[..8] {
        let (shape, indices) = order(case.shape, case.transpose);
        let outshape = [shape[0], shape[1], case.output[0], case.output[1]];
        let g = gradient(outshape);
        for &mode in modes {
            for align in [false, true] {
                let input = Tensor::<4>::from_data(TensorData::new(source(case.shape), case.shape), &device).require_grad();
                let retained = input.clone();
                let output = interpolate(view(input.clone(), case.transpose), mode.options(case.output, align));
                let grads = (output * Tensor::<4>::from_data(TensorData::new(g.clone(), outshape), &device))
                    .sum()
                    .backward();
                let actual = read(
                    input.grad(&grads).ok_or_else(|| format!("missing interpolation gradient {mode:?}"))?,
                    case.shape,
                )?;
                let logical = backward(&g, shape, case.output, mode, align);
                let mut expected = vec![0f64; actual.len()];
                for (j, &i) in indices.iter().enumerate() {
                    expected[i] = logical[j];
                }
                for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                    if !a.is_finite() || (f64::from(a) - e).abs() > 3e-5 * e.abs() + 5e-6 {
                        return Err(format!("gradient {mode:?} align={align} case={:?}[{i}] {a} != {e}", case.shape));
                    }
                }
                if read(retained, case.shape)? != source(case.shape) {
                    return Err(String::from("interpolation backward changed retained leaf"));
                }
                cases += 1;
            }
        }
    }
    Ok(cases)
}
pub fn check_empty(device: &Device) -> Result<usize, String> {
    let mut cases = 0;
    for shape in [[0, 2, 3, 5], [2, 0, 3, 5]] {
        for mode in MODES {
            for align in [false, true] {
                let out = interpolate(Tensor::<4>::zeros(shape, device), mode.options([7, 9], align));
                if !read(out, [shape[0], shape[1], 7, 9])?.is_empty() {
                    return Err(String::from("nonempty interpolation empty batch/channel"));
                }
                cases += 1;
            }
        }
    }
    Ok(cases)
}
pub fn check_ieee(device: &Device) -> Result<usize, String> {
    let codes = [
        0u32, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fc01234, 0xffc05678, 0x7f800001, 0x3f800000, 0xbf800000, 0x00800000, 0x80800000,
        0x7f7fffff, 0xff7fffff,
    ];
    let input: Vec<_> = codes.into_iter().map(f32::from_bits).collect();
    let shape = [1, 1, 3, 5];
    let mut cases = 0;
    for mode in [Mode::Nearest, Mode::Exact] {
        for align in [false, true] {
            let tensor = Tensor::<4>::from_data(TensorData::new(input.clone(), shape), device);
            check(
                &read(interpolate(tensor, mode.options([7, 9], align)), [1, 1, 7, 9])?,
                &forward(&input, shape, [7, 9], mode, align),
                true,
                "nearest IEEE bits",
            )?;
            cases += 1;
        }
    }
    Ok(cases)
}
