//! Direct FP64 convolution references and finite differences for GPU gradients.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{DType, Shape, Slice, TensorData, TensorMetadata, ops::ConvOptions},
    native::{NativeTensor, convolution::Conv2d},
};
use d3os_compute::{Error, nvidia::Nvidia};
use d3os_nvidia::compute as abi;

struct Fixture {
    tensor: NativeTensor,
    backing: NativeTensor,
    physical: Vec<f32>,
    logical: Vec<f64>,
}

fn fixture(shape: [usize; 4], strided: bool, seed: usize) -> Result<Fixture, Error> {
    let [n, c, h, w] = shape;
    let storage = if strided { [n + 2, c, w, h] } else { [n + 2, c, h, w] };
    let physical: Vec<_> = (0..storage.into_iter().product())
        .map(|i| (((i * 13 + seed * 7) % 37) as f32 - 18.0) / 37.0)
        .collect();
    let backing = NativeTensor::from_data(TensorData::new(physical.clone(), storage))?;
    let tensor = if strided {
        backing
            .clone()
            .slice(&[Slice::new(1, Some((n + 1) as isize), 1)])?
            .permute(&[0, 1, 3, 2])?
            .flip(&[0, 1, 3])?
    } else {
        backing.clone().slice(&[Slice::new(1, Some((n + 1) as isize), 1)])?
    };
    let logical = (0..n * c * h * w)
        .map(|i| {
            let x = i % w;
            let y = i / w % h;
            let ch = i / (w * h) % c;
            let batch = i / (c * h * w);
            let at = if strided {
                (((n - batch) * c + c - 1 - ch) * w + w - 1 - x) * h + y
            } else {
                i + c * h * w
            };
            f64::from(physical[at])
        })
        .collect();
    Ok(Fixture {
        tensor,
        backing,
        physical,
        logical,
    })
}

fn read(tensor: &NativeTensor) -> Result<Vec<f64>, Error> {
    Ok(tensor
        .read()?
        .as_slice::<f32>()
        .map_err(|_| Error::DeviceFault)?
        .iter()
        .map(|&v| f64::from(v))
        .collect())
}

fn compare(tensor: &NativeTensor, expected: &[f64]) -> Result<(), Error> {
    let actual = read(tensor)?;
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(&a, &b)| !a.is_finite() || (a - b).abs() > 2e-5 * (1.0 + b.abs()))
    {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

struct Reference {
    output: Vec<f64>,
    dx: Vec<f64>,
    dw: Vec<f64>,
    db: Vec<f64>,
}

fn reference(input: &[f64], weights: &[f64], bias: Option<&[f64]>, dy: &[f64], shapes: [[usize; 4]; 3], options: &ConvOptions<2>) -> Reference {
    let [shape, filter, out] = shapes;
    let [n, c, h, w] = shape;
    let [o, ci, kh, kw] = filter;
    let [_, _, oh, ow] = out;
    let mut r = Reference {
        output: vec![0.0; n * o * oh * ow],
        dx: vec![0.0; n * c * h * w],
        dw: vec![0.0; o * ci * kh * kw],
        db: vec![0.0; o],
    };
    for batch in 0..n {
        for oc in 0..o {
            for oy in 0..oh {
                for ox in 0..ow {
                    let at = ((batch * o + oc) * oh + oy) * ow + ox;
                    r.output[at] = bias.map_or(0.0, |b| b[oc]);
                    r.db[oc] += dy[at];
                    let group = oc / (o / options.groups);
                    for ic in 0..ci {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * options.stride[0] + ky * options.dilation[0]) as isize - options.padding[0].0 as isize;
                                let ix = (ox * options.stride[1] + kx * options.dilation[1]) as isize - options.padding[1].0 as isize;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                    continue;
                                }
                                let xi = ((batch * c + group * ci + ic) * h + iy as usize) * w + ix as usize;
                                let wi = ((oc * ci + ic) * kh + ky) * kw + kx;
                                r.output[at] += input[xi] * weights[wi];
                                r.dx[xi] += dy[at] * weights[wi];
                                r.dw[wi] += dy[at] * input[xi];
                            }
                        }
                    }
                }
            }
        }
    }
    r
}

fn case(
    backend: &Nvidia, shape: [usize; 4], filter: [usize; 4], options: ConvOptions<2>, strided: bool, biased: bool, forward: abi::Kernel,
) -> Result<(), Error> {
    let input = fixture(shape, strided, 1)?;
    let weights = fixture(filter, strided, 2)?;
    let plan = Conv2d::new(Shape::new(shape), Shape::new(filter), None, &options)?;
    let out: [usize; 4] = plan.output_shape().as_slice().try_into().unwrap();
    let gradient = fixture(out, strided, 3)?;
    let biases: Vec<_> = (0..filter[0]).map(|i| i as f32 / 17.0 - 0.2).collect();
    let bias = NativeTensor::from_data(TensorData::new(biases.clone(), [filter[0]]))?.flip(&[0])?;
    let logical_bias: Vec<_> = biases.iter().rev().map(|&v| f64::from(v)).collect();
    let reference = reference(
        &input.logical,
        &weights.logical,
        biased.then_some(logical_bias.as_slice()),
        &gradient.logical,
        [shape, filter, out],
        &options,
    );
    let before = backend.service_profile()?;
    let profiles = backend.profile()?;
    let output = input.tensor.conv2d(&weights.tensor, biased.then_some(&bias), &options)?;
    let after = backend.profile()?;
    if after[abi::Kernel::Conv2d as usize].calls - profiles[abi::Kernel::Conv2d as usize].calls != u64::from(shape[0] != 0 && forward == abi::Kernel::Conv2d)
        || after[abi::Kernel::Im2col as usize].calls - profiles[abi::Kernel::Im2col as usize].calls
            != u64::from(shape[0] != 0 && forward == abi::Kernel::Im2col)
        || after[abi::Kernel::Conv2dWindows as usize].calls != profiles[abi::Kernel::Conv2dWindows as usize].calls
        || (forward == abi::Kernel::Im2col && after[abi::Kernel::Matmul as usize].calls - profiles[abi::Kernel::Matmul as usize].calls != 1)
    {
        return Err(Error::DeviceFault);
    }
    let before_gradient = after[abi::Kernel::Matmul as usize].calls;
    let (dx, dw, db) = input.tensor.conv2d_backward(&weights.tensor, &gradient.tensor, biased, &options)?;
    let both_gradient = backend.profile()?[abi::Kernel::Matmul as usize].calls;
    let dx_only = input.tensor.conv2d_input_backward(&weights.tensor, &gradient.tensor, &options)?;
    let input_gradient = backend.profile()?[abi::Kernel::Matmul as usize].calls;
    let dw_only = input.tensor.conv2d_weight_backward(&weights.tensor, &gradient.tensor, &options)?;
    let weight_gradient = backend.profile()?[abi::Kernel::Matmul as usize].calls;
    if !input.tensor.layout().is_empty()
        && !gradient.tensor.layout().is_empty()
        && (both_gradient - before_gradient != 2 * options.groups as u64
            || input_gradient - both_gradient != options.groups as u64
            || weight_gradient - input_gradient != options.groups as u64)
    {
        return Err(Error::DeviceFault);
    }
    let after = backend.service_profile()?;
    if [abi::UPLOAD, abi::DOWNLOAD]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
        || output.shape().as_slice() != out
        || dx.shape().as_slice() != shape
        || dw.shape().as_slice() != filter
        || db.is_some() != biased
    {
        return Err(Error::DeviceFault);
    }
    compare(&output, &reference.output)?;
    compare(&dx, &reference.dx)?;
    compare(&dw, &reference.dw)?;
    compare(&dx_only, &reference.dx)?;
    compare(&dw_only, &reference.dw)?;
    if let Some(db) = db {
        compare(&db, &reference.db)?;
    }
    for fixture in [&input, &weights, &gradient] {
        let expected: Vec<_> = fixture.physical.iter().map(|&v| f64::from(v)).collect();
        if read(&fixture.backing)? != expected {
            return Err(Error::DeviceFault);
        }
    }
    Ok(())
}

fn finite_difference() -> Result<(), Error> {
    let shape = [1, 2, 3, 4];
    let filter = [2, 1, 2, 2];
    let options = ConvOptions::new([2, 1], [1, 1], [1, 2], 2);
    let input = fixture(shape, true, 5)?;
    let weights = fixture(filter, true, 6)?;
    let plan = Conv2d::new(Shape::new(shape), Shape::new(filter), None, &options)?;
    let out = plan.output_shape().as_slice().try_into().unwrap();
    let gradient = fixture(out, true, 7)?;
    let (dx, dw, db) = input.tensor.conv2d_backward(&weights.tensor, &gradient.tensor, true, &options)?;
    let objective = |x: &[f64], w: &[f64], b: &[f64]| {
        reference(x, w, Some(b), &gradient.logical, [shape, filter, out], &options)
            .output
            .iter()
            .zip(&gradient.logical)
            .map(|(&x, &g)| x * g)
            .sum::<f64>()
    };
    let x = input.logical;
    let w = weights.logical;
    let b = vec![0.0; 2];
    for (kind, actual) in [read(&dx)?, read(&dw)?, read(&db.ok_or(Error::DeviceFault)?)?].into_iter().enumerate() {
        for (index, actual) in actual.into_iter().enumerate() {
            let mut plus = [x.clone(), w.clone(), b.clone()];
            let mut minus = plus.clone();
            plus[kind][index] += 1e-4;
            minus[kind][index] -= 1e-4;
            let expected = (objective(&plus[0], &plus[1], &plus[2]) - objective(&minus[0], &minus[1], &minus[2])) / 2e-4;
            if !actual.is_finite() || (actual - expected).abs() > 2e-5 * (1.0 + expected.abs()) {
                return Err(Error::DeviceFault);
            }
        }
    }
    Ok(())
}

fn guards(backend: &Nvidia) -> Result<(), Error> {
    let shape = [2, 4, 7, 9];
    let filter = [6, 2, 3, 2];
    let options = ConvOptions::new_with_padding([2, 3], [(1, 2), (0, 1)], [2, 2], 2);
    let input = fixture(shape, true, 9)?;
    let weights = fixture(filter, true, 10)?;
    let plan = Conv2d::new(Shape::new(shape), Shape::new(filter), None, &options)?;
    let p = plan.parameters(input.tensor.layout(), weights.tensor.layout())?;
    let out = plan.output_shape().as_slice().try_into().unwrap();
    let expected = reference(
        &input.logical,
        &weights.logical,
        None,
        &vec![0.0; plan.output_shape().num_elements()],
        [shape, filter, out],
        &options,
    )
    .output;
    let guard = 0x6bad_beef;
    let mut x = vec![guard];
    x.extend(input.physical.iter().map(|v| v.to_bits()));
    x.push(guard);
    let mut w = vec![guard];
    w.extend(weights.physical.iter().map(|v| v.to_bits()));
    w.push(guard);
    let x_buffer = backend.upload_words(&x)?;
    let w_buffer = backend.upload_words(&w)?;
    let mut output_words = vec![guard; expected.len() + 2];
    let output = backend.upload_words(&output_words)?;
    backend.dispatch_words(abi::Kernel::Conv2d, &[(&x_buffer, 1), (&w_buffer, 1), (&x_buffer, 1), (&output, 1)], &p)?;
    backend.read_words_into(&output, 0, &mut output_words)?;
    if output_words[0] != guard
        || output_words[expected.len() + 1] != guard
        || output_words[1..expected.len() + 1].iter().zip(&expected).any(|(&a, &b)| {
            let a = f64::from(f32::from_bits(a));
            !a.is_finite() || (a - b).abs() > 2e-5 * (1.0 + b.abs())
        })
    {
        return Err(Error::DeviceFault);
    }
    let mut malformed = p;
    malformed[18] = input.tensor.layout().offset();
    if backend.dispatch_words(abi::Kernel::Conv2d, &[(&x_buffer, 1), (&w_buffer, 1), (&x_buffer, 1), (&output, 1)], &malformed) != Err(Error::InvalidParameter)
    {
        return Err(Error::DeviceFault);
    }
    if backend.dispatch_words(abi::Kernel::Conv2d, &[(&x_buffer, 1), (&w_buffer, 1), (&x_buffer, 1), (&x_buffer, 1)], &p) != Err(Error::InvalidParameter) {
        return Err(Error::DeviceFault);
    }
    let mut unchanged = vec![0; output_words.len()];
    backend.read_words_into(&output, 0, &mut unchanged)?;
    if unchanged != output_words {
        return Err(Error::DeviceFault);
    }
    for (buffer, expected) in [(&x_buffer, &x), (&w_buffer, &w)] {
        let mut actual = vec![0; expected.len()];
        backend.read_words_into(buffer, 0, &mut actual)?;
        if &actual != expected {
            return Err(Error::DeviceFault);
        }
    }
    // The window/fold pair is an adjoint even with overlap, dilation, and groups.
    let window_p = plan.windows_parameters(input.tensor.layout())?;
    let count = plan.columns_shape().num_elements();
    let columns = backend.upload_words(&vec![guard; count + 2])?;
    backend.dispatch_words(abi::Kernel::Conv2dWindows, &[(&x_buffer, 1), (&columns, 1)], &window_p)?;
    let mut values = vec![0; count + 2];
    backend.read_words_into(&columns, 0, &mut values)?;
    if values[0] != guard || values[count + 1] != guard {
        return Err(Error::DeviceFault);
    }
    let folded = backend.upload_words(&vec![guard; shape.into_iter().product::<usize>() + 2])?;
    backend.dispatch_words(abi::Kernel::Conv2dFold, &[(&columns, 1), (&folded, 1)], &p[..18])?;
    let mut folded_values = vec![0; shape.into_iter().product::<usize>() + 2];
    backend.read_words_into(&folded, 0, &mut folded_values)?;
    if folded_values[0] != guard || folded_values[folded_values.len() - 1] != guard {
        return Err(Error::DeviceFault);
    }
    let lhs: f64 = values[1..count + 1]
        .iter()
        .map(|&v| {
            let x = f64::from(f32::from_bits(v));
            x * x
        })
        .sum();
    let rhs: f64 = folded_values[1..folded_values.len() - 1]
        .iter()
        .zip(&input.logical)
        .map(|(&a, &b)| f64::from(f32::from_bits(a)) * b)
        .sum();
    if !lhs.is_finite() || !rhs.is_finite() || (lhs - rhs).abs() > 2e-5 * (1.0 + lhs.abs()) {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn rejected(backend: &Nvidia) -> Result<(), Error> {
    let input = fixture([1, 2, 3, 4], false, 1)?;
    let weights = fixture([2, 1, 2, 2], false, 2)?;
    let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 2);
    let wrong = NativeTensor::full(Shape::new([1, 2, 3, 4]), DType::I32, 0)?;
    let grad = NativeTensor::full(Shape::new([1, 2, 3, 4]), DType::F32, 0)?;
    let before = backend.queue_profile()?;
    if !matches!(wrong.conv2d(&weights.tensor, None, &options), Err(Error::InvalidParameter))
        || !matches!(input.tensor.conv2d_backward(&weights.tensor, &grad, false, &options), Err(Error::ShapeMismatch))
        || !matches!(input.tensor.conv2d(&weights.tensor, Some(&grad), &options), Err(Error::ShapeMismatch))
        || before.launches != backend.queue_profile()?.launches
    {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    for (shape, filter, options, strided, bias) in [
        (
            [2, 4, 7, 9],
            [6, 2, 3, 2],
            ConvOptions::new_with_padding([2, 3], [(1, 2), (0, 1)], [2, 2], 2),
            true,
            true,
        ),
        (
            [1, 2, 5, 6],
            [19, 2, 3, 2],
            ConvOptions::new_with_padding([1, 2], [(1, 0), (2, 1)], [1, 2], 1),
            false,
            true,
        ),
        ([2, 3, 4, 5], [6, 1, 2, 2], ConvOptions::new([1, 1], [0, 0], [1, 1], 3), true, false),
        ([1, 3, 8, 9], [17, 3, 3, 3], ConvOptions::new([1, 1], [1, 1], [1, 1], 1), true, true),
        ([0, 3, 5, 5], [4, 3, 3, 3], ConvOptions::new([1, 1], [0, 0], [1, 1], 1), true, true),
    ] {
        case(backend, shape, filter, options, strided, bias, abi::Kernel::Conv2d).map_err(|e| format!("convolution {shape:?} {filter:?}: {e}"))?;
    }
    for biased in [false, true] {
        for strided in [false, true] {
            case(
                backend,
                [1, 32, 7, 9],
                [33, 32, 3, 3],
                ConvOptions::new([2, 1], [1, 2], [1, 1], 1),
                strided,
                biased,
                if strided { abi::Kernel::Conv2d } else { abi::Kernel::Im2col },
            )
            .map_err(|e| format!("dense convolution strided={strided} bias={biased}: {e}"))?;
        }
    }
    finite_difference().map_err(|e| format!("convolution finite differences: {e}"))?;
    guards(backend).map_err(|e| format!("convolution guards: {e}"))?;
    rejected(backend).map_err(|e| format!("convolution invalid metadata: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_CONVOLUTION_OK forward=implicit,explicit-tensor-core gradients=input,weights,bias reference=fp64 finite-difference=ok groups=ok strides=ok guards=ok backend=nvidia"
    ));
    Ok(())
}
