//! Public convolution APIs against independent FP64 sums and analytic gradients.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::TensorData,
    backend::ops::{ConvOptions, ConvTransposeOptions},
    tensor::{
        Device, Tensor,
        module::{conv_transpose1d, conv_transpose2d, conv_transpose3d, conv3d},
    },
};
#[derive(Clone, Debug)]
pub struct Case {
    pub input: Vec<usize>,
    pub weights: Vec<usize>,
    pub stride: Vec<usize>,
    pub padding: Vec<usize>,
    pub dilation: Vec<usize>,
    pub extra: Vec<usize>,
    pub groups: usize,
    pub transpose: bool,
}
pub fn cases() -> Vec<Case> {
    let mut all = Vec::new();
    for transpose in [false, true] {
        for (input, weights, stride, padding, dilation, extra, groups) in [
            (
                vec![1, 2, 3, 4, 5],
                vec![3, 2, 2, 2, 3],
                vec![1, 1, 1],
                vec![0, 0, 1],
                vec![1, 1, 1],
                vec![0, 0, 0],
                1,
            ),
            (
                vec![2, 4, 3, 5, 7],
                vec![6, 2, 2, 3, 2],
                vec![2, 1, 2],
                vec![1, 1, 0],
                vec![1, 2, 1],
                vec![1, 0, 1],
                2,
            ),
            (
                vec![1, 3, 3, 4, 5],
                vec![3, 1, 2, 2, 2],
                vec![1, 2, 1],
                vec![1, 0, 1],
                vec![2, 1, 2],
                vec![0, 1, 0],
                3,
            ),
            (
                vec![1, 2, 1, 1, 3],
                vec![2, 2, 1, 1, 1],
                vec![1, 1, 1],
                vec![0, 0, 0],
                vec![1, 1, 1],
                vec![0, 0, 2],
                1,
            ),
        ] {
            let input = if transpose {
                let mut s = input;
                s[1] = weights[0];
                s
            } else {
                input
            };
            all.push(Case {
                input,
                weights,
                stride,
                padding,
                dilation,
                extra,
                groups,
                transpose,
            });
        }
    }
    for (input, weights, stride, padding, dilation, extra, groups) in [
        (vec![1, 2, 3, 5], vec![2, 3, 2, 3], vec![1, 2], vec![0, 1], vec![1, 1], vec![0, 1], 1),
        (vec![2, 4, 3, 4], vec![4, 2, 2, 2], vec![2, 1], vec![1, 0], vec![1, 2], vec![1, 0], 2),
        (vec![1, 3, 2, 3], vec![3, 1, 2, 3], vec![1, 1], vec![0, 1], vec![2, 1], vec![2, 1], 3),
        (vec![1, 2, 1, 1], vec![2, 2, 1, 1], vec![1, 1], vec![0, 0], vec![1, 1], vec![2, 3], 1),
    ] {
        all.push(Case {
            input,
            weights,
            stride,
            padding,
            dilation,
            extra,
            groups,
            transpose: true,
        });
    }
    for (input, weights, stride, padding, dilation, extra, groups) in [
        (vec![2, 4, 5], vec![4, 2, 3], vec![2], vec![1], vec![2], vec![1], 2),
        (vec![1, 3, 3], vec![3, 1, 2], vec![1], vec![0], vec![2], vec![3], 3),
    ] {
        all.push(Case {
            input,
            weights,
            stride,
            padding,
            dilation,
            extra,
            groups,
            transpose: true,
        });
    }

    all
}
fn canonical(shape: &[usize]) -> [usize; 5] {
    match shape {
        [n, c, d, h, w] => [*n, *c, *d, *h, *w],
        [n, c, h, w] => [*n, *c, 1, *h, *w],
        [n, c, w] => [*n, *c, 1, 1, *w],
        _ => panic!("reference rank"),
    }
}
fn spatial(values: &[usize], fill: usize) -> [usize; 3] {
    if values.len() == 1 {
        [fill, fill, values[0]]
    } else if values.len() == 2 {
        [fill, values[0], values[1]]
    } else {
        values.try_into().unwrap()
    }
}
impl Case {
    pub fn output(&self) -> Vec<usize> {
        let mut out = self.input.clone();
        out[1] = if self.transpose { self.weights[1] * self.groups } else { self.weights[0] };
        for axis in 0..self.stride.len() {
            let i = self.input[axis + 2] as i64;
            let k = self.weights[axis + 2] as i64;
            let s = self.stride[axis] as i64;
            let p = self.padding[axis] as i64;
            let d = self.dilation[axis] as i64;
            out[axis + 2] = if self.transpose {
                ((i - 1) * s + (k - 1) * d + 1 + self.extra[axis] as i64 - 2 * p) as usize
            } else {
                ((i + 2 * p - (k - 1) * d - 1) / s + 1) as usize
            };
        }
        out
    }
}
pub fn values(shape: &[usize], salt: usize) -> Vec<f32> {
    (0..shape.iter().product()).map(|i| ((i * 17 + salt * 11) % 67) as f32 / 31.0 - 1.0).collect()
}
pub struct Reference {
    pub output: Vec<f64>,
    pub dx: Vec<f64>,
    pub dw: Vec<f64>,
    pub db: Vec<f64>,
}
fn at(shape: [usize; 5], n: usize, c: usize, z: usize, y: usize, x: usize) -> usize {
    ((((n * shape[1] + c) * shape[2] + z) * shape[3] + y) * shape[4]) + x
}
pub fn reference(case: &Case, x: &[f32], w: &[f32], bias: Option<&[f32]>, dy: &[f32]) -> Reference {
    let output = case.output();
    let image = canonical(if case.transpose { &output } else { &case.input });
    let grid = canonical(if case.transpose { &case.input } else { &output });
    let filter = canonical(&case.weights);
    let stride = spatial(&case.stride, 1);
    let pad = spatial(&case.padding, 0);
    let dil = spatial(&case.dilation, 1);
    let ci = image[1] / case.groups;
    let co = grid[1] / case.groups;
    let mut r = Reference {
        output: vec![0.0; output.iter().product()],
        dx: vec![0.0; x.len()],
        dw: vec![0.0; w.len()],
        db: vec![0.0; output[1]],
    };
    let out = canonical(&output);
    let plane = out[2] * out[3] * out[4];
    for (i, result) in r.output.iter_mut().enumerate() {
        *result = bias.map_or(0.0, |b| f64::from(b[i / plane % out[1]]));
        r.db[i / plane % out[1]] += f64::from(dy[i]);
    }
    for n in 0..grid[0] {
        for oc in 0..grid[1] {
            for oz in 0..grid[2] {
                for oy in 0..grid[3] {
                    for ox in 0..grid[4] {
                        let g = at(grid, n, oc, oz, oy, ox);
                        let group = oc / co;
                        for ic in 0..ci {
                            for kz in 0..filter[2] {
                                for ky in 0..filter[3] {
                                    for kx in 0..filter[4] {
                                        let coords = [oz, oy, ox]
                                            .into_iter()
                                            .zip([kz, ky, kx])
                                            .enumerate()
                                            .map(|(a, (o, k))| (o * stride[a] + k * dil[a]) as isize - pad[a] as isize)
                                            .collect::<Vec<_>>();
                                        if coords.iter().enumerate().any(|(a, &v)| v < 0 || v >= image[a + 2] as isize) {
                                            continue;
                                        }
                                        let im = at(image, n, group * ci + ic, coords[0] as usize, coords[1] as usize, coords[2] as usize);
                                        let wi = at(filter, oc, ic, kz, ky, kx);
                                        let (src, dst) = if case.transpose { (g, im) } else { (im, g) };
                                        r.output[dst] += f64::from(x[src]) * f64::from(w[wi]);
                                        r.dx[src] += f64::from(dy[dst]) * f64::from(w[wi]);
                                        r.dw[wi] += f64::from(dy[dst]) * f64::from(x[src]);
                                    }
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
        return Err(format!("{label}: lengths {} != {}", actual.len(), expected.len()));
    }
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (f64::from(a) - e).abs() > 3e-5 * (1.0 + e.abs()) {
            return Err(format!("{label}[{i}] {a} != {e}"));
        }
    }
    Ok(())
}
fn read<const D: usize>(t: Tensor<D>) -> Result<Vec<f32>, String> {
    t.try_into_vec_as::<f32>().map_err(|e| format!("convolution read: {e:?}"))
}
pub fn check(device: &Device, empty: bool, selected: Option<usize>, exact_inputs: bool, empty_backward: bool) -> Result<(usize, usize), String> {
    let device = device.clone().autodiff();
    let mut count = 0;
    let mut gradients = 0;
    if selected.is_some_and(|index| index >= cases().len()) {
        return Err(String::from("invalid convolution case"));
    }
    for (index, mut case) in cases().into_iter().enumerate() {
        if selected.is_some_and(|chosen| chosen != index) {
            continue;
        }
        if empty {
            case.input[0] = 0;
        }
        for flipped in [false, true] {
            for biased in [false, true] {
                let data = |shape: &[usize], salt: usize| {
                    if exact_inputs {
                        (0..shape.iter().product())
                            .map(|i| (((i * 17 + salt * 11) % 67) as f32 - 32.0) / 32.0)
                            .collect()
                    } else {
                        values(shape, salt)
                    }
                };
                let mut x = data(&case.input, 1);
                let mut w = data(&case.weights, 2);
                let mut b = data(&[case.output()[1]], 3);
                let mut dy = data(&case.output(), 4);
                if flipped {
                    x.reverse();
                    w.reverse();
                    b.reverse();
                    dy.reverse();
                }
                let reference = reference(&case, &x, &w, biased.then_some(b.as_slice()), &dy);
                macro_rules! run {
                    ($rank:literal,$func:ident,$options:expr) => {{
                        let upload = |mut values: Vec<f32>, shape: Vec<usize>| {
                            if flipped {
                                values.reverse();
                            }
                            let tensor = Tensor::<$rank>::from_data(TensorData::new(values, shape), &device);
                            if flipped {
                                tensor.flip(core::array::from_fn::<usize, $rank, _>(|i| i))
                            } else {
                                tensor
                            }
                        };
                        let input = upload(x.clone(), case.input.clone()).require_grad();
                        let weights = upload(w.clone(), case.weights.clone()).require_grad();
                        let mut bias_data = b.clone();
                        if flipped {
                            bias_data.reverse();
                        }
                        let bias = Tensor::<1>::from_data(TensorData::new(bias_data, [b.len()]), &device);
                        let bias = if flipped { bias.flip([0]) } else { bias };
                        let bias = bias.require_grad();
                        let output = $func(input.clone(), weights.clone(), biased.then_some(bias.clone()), $options);
                        if output.dims().as_slice() != case.output().as_slice() {
                            return Err(format!("convolution output shape {case:?}"));
                        }
                        compare(
                            &read(output.clone())?,
                            &reference.output,
                            &format!("forward {case:?} flip={flipped} bias={biased}"),
                        )?;
                        if !empty || empty_backward {
                            let grads = (output * upload(dy.clone(), case.output())).sum().backward();
                            compare(
                                &read(input.grad(&grads).ok_or_else(|| String::from("missing input gradient"))?)?,
                                &reference.dx,
                                &format!("input gradient {case:?}"),
                            )?;
                            compare(
                                &read(weights.grad(&grads).ok_or_else(|| String::from("missing weight gradient"))?)?,
                                &reference.dw,
                                &format!("weight gradient {case:?}"),
                            )?;
                            if biased {
                                compare(
                                    &read(bias.grad(&grads).ok_or_else(|| String::from("missing bias gradient"))?)?,
                                    &reference.db,
                                    "bias gradient",
                                )?;
                            }
                            gradients += 2 + usize::from(biased);
                        }
                        compare(
                            &read(input)?,
                            &x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                            "retained input",
                        )?;
                        compare(
                            &read(weights)?,
                            &w.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                            "retained weights",
                        )?;
                    }};
                }
                match (case.transpose, case.input.len()) {
                    (false, 5) => run!(
                        5,
                        conv3d,
                        ConvOptions::new(
                            case.stride.clone().try_into().unwrap(),
                            case.padding.clone().try_into().unwrap(),
                            case.dilation.clone().try_into().unwrap(),
                            case.groups
                        )
                    ),
                    (true, 3) => run!(
                        3,
                        conv_transpose1d,
                        ConvTransposeOptions::new(
                            case.stride.clone().try_into().unwrap(),
                            case.padding.clone().try_into().unwrap(),
                            case.extra.clone().try_into().unwrap(),
                            case.dilation.clone().try_into().unwrap(),
                            case.groups
                        )
                    ),
                    (true, 4) => run!(
                        4,
                        conv_transpose2d,
                        ConvTransposeOptions::new(
                            case.stride.clone().try_into().unwrap(),
                            case.padding.clone().try_into().unwrap(),
                            case.extra.clone().try_into().unwrap(),
                            case.dilation.clone().try_into().unwrap(),
                            case.groups
                        )
                    ),
                    (true, 5) => run!(
                        5,
                        conv_transpose3d,
                        ConvTransposeOptions::new(
                            case.stride.clone().try_into().unwrap(),
                            case.padding.clone().try_into().unwrap(),
                            case.extra.clone().try_into().unwrap(),
                            case.dilation.clone().try_into().unwrap(),
                            case.groups
                        )
                    ),
                    _ => unreachable!(),
                }
                count += 1;
            }
        }
    }
    device.sync().map_err(|e| format!("convolution sync: {e:?}"))?;
    Ok((count, gradients))
}
