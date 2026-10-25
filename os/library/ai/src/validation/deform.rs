//! Deformable convolution with independent FP64 samples and five derivatives.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{TensorData, ops::DeformConvOptions},
    tensor::{Device, Tensor, module::deform_conv2d},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    Fractional,
    Zero,
    Collision,
    Outside,
}
#[derive(Clone, Debug)]
pub struct Case {
    pub input: [usize; 4],
    pub weights: [usize; 4],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub weight_groups: usize,
    pub offset_groups: usize,
    pub pattern: Pattern,
}
pub fn cases() -> Vec<Case> {
    vec![
        Case {
            input: [1, 2, 4, 5],
            weights: [3, 2, 2, 3],
            stride: [1, 1],
            padding: [1, 0],
            dilation: [1, 1],
            weight_groups: 1,
            offset_groups: 1,
            pattern: Pattern::Fractional,
        },
        Case {
            input: [2, 4, 5, 7],
            weights: [6, 2, 2, 2],
            stride: [2, 1],
            padding: [1, 1],
            dilation: [1, 2],
            weight_groups: 2,
            offset_groups: 2,
            pattern: Pattern::Fractional,
        },
        Case {
            input: [1, 6, 3, 5],
            weights: [4, 3, 2, 2],
            stride: [1, 2],
            padding: [0, 1],
            dilation: [1, 1],
            weight_groups: 2,
            offset_groups: 3,
            pattern: Pattern::Fractional,
        },
        Case {
            input: [1, 3, 5, 4],
            weights: [6, 1, 3, 2],
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            weight_groups: 3,
            offset_groups: 1,
            pattern: Pattern::Fractional,
        },
        Case {
            input: [1, 4, 1, 137],
            weights: [5, 4, 1, 1],
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            weight_groups: 1,
            offset_groups: 2,
            pattern: Pattern::Fractional,
        },
        Case {
            input: [1, 1, 3, 4],
            weights: [2, 1, 2, 2],
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            weight_groups: 1,
            offset_groups: 1,
            pattern: Pattern::Zero,
        },
        Case {
            input: [1, 4, 5, 7],
            weights: [4, 2, 3, 3],
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            weight_groups: 2,
            offset_groups: 1,
            pattern: Pattern::Collision,
        },
        Case {
            input: [1, 2, 3, 5],
            weights: [3, 2, 2, 2],
            stride: [1, 1],
            padding: [1, 0],
            dilation: [1, 1],
            weight_groups: 1,
            offset_groups: 2,
            pattern: Pattern::Outside,
        },
    ]
}
impl Case {
    pub fn output(&self) -> [usize; 4] {
        [
            self.input[0],
            self.weights[0],
            (self.input[2] + 2 * self.padding[0] - (self.weights[2] - 1) * self.dilation[0] - 1) / self.stride[0] + 1,
            (self.input[3] + 2 * self.padding[1] - (self.weights[3] - 1) * self.dilation[1] - 1) / self.stride[1] + 1,
        ]
    }
    pub fn mask_shape(&self) -> [usize; 4] {
        let out = self.output();
        [out[0], self.offset_groups * self.weights[2] * self.weights[3], out[2], out[3]]
    }
    pub fn offset_shape(&self) -> [usize; 4] {
        let mut shape = self.mask_shape();
        shape[1] *= 2;
        shape
    }
    pub fn options(&self) -> DeformConvOptions<2> {
        DeformConvOptions::new(self.stride, self.padding, self.dilation, self.weight_groups, self.offset_groups)
    }
    pub fn offsets(&self) -> Vec<f32> {
        let shape = self.offset_shape();
        let out = self.output();
        let s = out[2] * out[3];
        (0..shape.iter().product())
            .map(|i| {
                let direction = i / s % 2;
                let k = i / (2 * s) % (self.weights[2] * self.weights[3]);
                let pos = i % s;
                match self.pattern {
                    Pattern::Zero => 0.0,
                    Pattern::Outside => {
                        if direction == 0 {
                            10000.0
                        } else {
                            -10000.0
                        }
                    }
                    Pattern::Fractional => {
                        let v = 0.125 + ((i * 7 + 3) % 5) as f32 * 0.25;
                        if (i / 3) % 2 == 0 { v } else { -v }
                    }
                    Pattern::Collision => {
                        let origin = if direction == 0 {
                            pos / out[3] * self.stride[0] + k / self.weights[3] * self.dilation[0]
                        } else {
                            pos % out[3] * self.stride[1] + k % self.weights[3] * self.dilation[1]
                        };
                        let target = if direction == 0 {
                            self.input[2] as f32 / 2.0 + 0.25
                        } else {
                            self.input[3] as f32 / 2.0 + 0.375
                        };
                        target + self.padding[direction] as f32 - origin as f32
                    }
                }
            })
            .collect()
    }
}
pub fn values(shape: &[usize], salt: usize, exact: bool) -> Vec<f32> {
    (0..shape.iter().product())
        .map(|i| {
            if exact {
                ((i * 7 + salt * 11) % 17) as f32 / 8.0 - 1.0
            } else {
                ((i * 17 + salt * 11) % 67) as f32 / 31.0 - 1.0
            }
        })
        .collect()
}
pub fn masks(shape: &[usize]) -> Vec<f32> {
    (0..shape.iter().product()).map(|i| (i % 4) as f32 * 0.5).collect()
}
fn index(shape: [usize; 4], n: usize, c: usize, y: usize, x: usize) -> usize {
    ((n * shape[1] + c) * shape[2] + y) * shape[3] + x
}
pub fn sample(x: &[f32], shape: [usize; 4], n: usize, c: usize, y: f64, z: f64) -> (f64, f64, f64, Vec<(usize, f64)>) {
    if !y.is_finite() || !z.is_finite() || y < -1.0 || z < -1.0 || y >= shape[2] as f64 || z >= shape[3] as f64 {
        return (0.0, 0.0, 0.0, Vec::new());
    }
    let fy = libm::floor(y);
    let fx = libm::floor(z);
    let ly = y - fy;
    let lx = z - fx;
    let mut corners = [0.0; 4];
    let mut contributions = Vec::new();
    for row in 0..2 {
        for col in 0..2 {
            let yy = fy as isize + row;
            let xx = fx as isize + col;
            if yy >= 0 && xx >= 0 && yy < shape[2] as isize && xx < shape[3] as isize {
                let at = index(shape, n, c, yy as usize, xx as usize);
                corners[row as usize * 2 + col as usize] = f64::from(x[at]);
                let weight = (if row == 0 { 1.0 - ly } else { ly }) * (if col == 0 { 1.0 - lx } else { lx });
                contributions.push((at, weight));
            }
        }
    }
    let [a, b, c, d] = corners;
    (
        (1.0 - ly) * (1.0 - lx) * a + (1.0 - ly) * lx * b + ly * (1.0 - lx) * c + ly * lx * d,
        (1.0 - lx) * (c - a) + lx * (d - b),
        (1.0 - ly) * (b - a) + ly * (d - c),
        contributions,
    )
}
pub struct Reference {
    pub output: Vec<f64>,
    pub dx: Vec<f64>,
    pub doffset: Vec<f64>,
    pub dw: Vec<f64>,
    pub dmask: Option<Vec<f64>>,
    pub db: Vec<f64>,
}
pub fn reference(case: &Case, x: &[f32], offset: &[f32], w: &[f32], mask: Option<&[f32]>, bias: Option<&[f32]>, dy: &[f32]) -> Reference {
    let out = case.output();
    let s = out[2] * out[3];
    let karea = case.weights[2] * case.weights[3];
    let mut r = Reference {
        output: vec![0.0; out.iter().product()],
        dx: vec![0.0; x.len()],
        doffset: vec![0.0; offset.len()],
        dw: vec![0.0; w.len()],
        dmask: mask.map(|m| vec![0.0; m.len()]),
        db: vec![0.0; out[1]],
    };
    for n in 0..out[0] {
        for oc in 0..out[1] {
            for oy in 0..out[2] {
                for ox in 0..out[3] {
                    let at = index(out, n, oc, oy, ox);
                    let g = f64::from(dy[at]);
                    r.output[at] = bias.map_or(0.0, |b| f64::from(b[oc]));
                    r.db[oc] += g;
                    let wg = oc / (out[1] / case.weight_groups);
                    for local in 0..case.weights[1] {
                        let ic = wg * case.weights[1] + local;
                        let og = ic / (case.input[1] / case.offset_groups);
                        for ky in 0..case.weights[2] {
                            for kx in 0..case.weights[3] {
                                let k = ky * case.weights[3] + kx;
                                let mi = ((n * case.offset_groups + og) * karea + k) * s + oy * out[3] + ox;
                                let oi = (mi / s) * 2 * s + mi % s;
                                let wi = ((oc * case.weights[1] + local) * case.weights[2] + ky) * case.weights[3] + kx;
                                let yy = (oy * case.stride[0] + ky * case.dilation[0]) as f64 - case.padding[0] as f64 + f64::from(offset[oi]);
                                let xx = (ox * case.stride[1] + kx * case.dilation[1]) as f64 - case.padding[1] as f64 + f64::from(offset[oi + s]);
                                let (value, gy, gx, pixels) = sample(x, case.input, n, ic, yy, xx);
                                let m = mask.map_or(1.0, |m| f64::from(m[mi]));
                                let weight = f64::from(w[wi]);
                                r.output[at] += value * m * weight;
                                r.dw[wi] += g * value * m;
                                r.doffset[oi] += g * weight * m * gy;
                                r.doffset[oi + s] += g * weight * m * gx;
                                if let Some(dm) = &mut r.dmask {
                                    dm[mi] += g * weight * value;
                                }
                                for (pixel, coefficient) in pixels {
                                    r.dx[pixel] += g * weight * m * coefficient;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    r
}
pub fn compare(actual: &[f32], expected: &[f64], label: &str) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("{label} length {} != {}", actual.len(), expected.len()));
    }
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (f64::from(a) - e).abs() > 4e-5 * (1.0 + e.abs()) {
            return Err(format!("{label}[{i}] {a} != {e}"));
        }
    }
    Ok(())
}
fn read<const D: usize>(t: Tensor<D>) -> Result<Vec<f32>, String> {
    t.try_into_vec_as::<f32>().map_err(|e| format!("deform read: {e:?}"))
}
pub fn check(device: &Device, empty: bool, exact: bool, selected: Option<usize>, backward: bool) -> Result<(usize, usize), String> {
    let device = device.clone().autodiff();
    let mut forward = 0;
    let mut gradients = 0;
    if selected.is_some_and(|n| n >= cases().len()) {
        return Err(String::from("invalid deform case"));
    }
    for (id, mut case) in cases().into_iter().enumerate() {
        if selected.is_some_and(|i| i != id) {
            continue;
        }
        if empty {
            case.input[0] = 0;
        }
        for flipped in [false, true] {
            for masked in [false, true] {
                for biased in [false, true] {
                    let mut x = values(&case.input, 1, exact);
                    let mut off = case.offsets();
                    let mut w = values(&case.weights, 2, exact);
                    let mut m = masks(&case.mask_shape());
                    let mut b = values(&[case.output()[1]], 3, exact);
                    let mut dy = values(&case.output(), 4, exact);
                    if flipped {
                        x.reverse();
                        off.reverse();
                        w.reverse();
                        m.reverse();
                        b.reverse();
                        dy.reverse();
                    }
                    let r = reference(&case, &x, &off, &w, masked.then_some(m.as_slice()), biased.then_some(b.as_slice()), &dy);
                    let upload = |mut values: Vec<f32>, shape: [usize; 4]| {
                        if flipped {
                            values.reverse();
                        }
                        let tensor = Tensor::<4>::from_data(TensorData::new(values, shape), &device);
                        if flipped { tensor.flip([0, 1, 2, 3]) } else { tensor }
                    };
                    let input = upload(x.clone(), case.input).require_grad();
                    let offsets = upload(off.clone(), case.offset_shape()).require_grad();
                    let weights = upload(w.clone(), case.weights).require_grad();
                    let mask = upload(m.clone(), case.mask_shape()).require_grad();
                    let mut bb = b.clone();
                    if flipped {
                        bb.reverse();
                    }
                    let bias = Tensor::<1>::from_data(TensorData::new(bb, [b.len()]), &device);
                    let bias = if flipped { bias.flip([0]) } else { bias };
                    let bias = bias.require_grad();
                    let output = deform_conv2d(
                        input.clone(),
                        offsets.clone(),
                        weights.clone(),
                        masked.then_some(mask.clone()),
                        biased.then_some(bias.clone()),
                        case.options(),
                    );
                    if output.dims() != case.output() {
                        return Err(format!("deform output shape case={id}"));
                    }
                    compare(
                        &read(output.clone())?,
                        &r.output,
                        &format!("forward case={id} flip={flipped} mask={masked} bias={biased}"),
                    )?;
                    if backward {
                        let grads = (output * upload(dy, case.output())).sum().backward();
                        compare(
                            &read(input.grad(&grads).ok_or_else(|| String::from("missing deform input gradient"))?)?,
                            &r.dx,
                            &format!("input gradient case={id}"),
                        )?;
                        compare(
                            &read(offsets.grad(&grads).ok_or_else(|| String::from("missing deform offset gradient"))?)?,
                            &r.doffset,
                            &format!("offset gradient case={id}"),
                        )?;
                        compare(
                            &read(weights.grad(&grads).ok_or_else(|| String::from("missing deform weight gradient"))?)?,
                            &r.dw,
                            &format!("weight gradient case={id}"),
                        )?;
                        gradients += 3;
                        if masked {
                            compare(
                                &read(mask.grad(&grads).ok_or_else(|| String::from("missing deform mask gradient"))?)?,
                                r.dmask.as_ref().unwrap(),
                                &format!("mask gradient case={id}"),
                            )?;
                            gradients += 1;
                        }
                        if biased {
                            compare(
                                &read(bias.grad(&grads).ok_or_else(|| String::from("missing deform bias gradient"))?)?,
                                &r.db,
                                &format!("bias gradient case={id}"),
                            )?;
                            gradients += 1;
                        }
                    }
                    for (label, actual, expected) in [
                        ("input", read(input)?, x),
                        ("offset", read(offsets)?, off),
                        ("weights", read(weights)?, w),
                        ("mask", read(mask)?, m),
                    ] {
                        compare(
                            &actual,
                            &expected.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                            &format!("retained {label}"),
                        )?;
                    }
                    forward += 1;
                }
            }
        }
    }
    device.sync().map_err(|e| format!("deform sync: {e:?}"))?;
    Ok((forward, gradients))
}
