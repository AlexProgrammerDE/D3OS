//! Independent pooling windows, FP64 gradients, and max-index edge cases.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{
        DType, Shape, Slice, TensorData, TensorMetadata,
        ops::{AvgPoolOptions, MaxPoolOptions},
    },
    native::{NativeTensor, pooling::Pooling},
};
use d3os_compute::{Backend, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, pooling::Mode};

struct Fixture {
    tensor: NativeTensor,
    backing: NativeTensor,
    physical: Vec<f32>,
    logical: Vec<f64>,
}

fn fixture(shape: Shape, seed: usize) -> Result<Fixture, Error> {
    let rank = shape.num_dims();
    let mut storage = shape.to_vec();
    storage[0] += 2;
    if rank >= 4 {
        storage.swap(rank - 1, rank - 2);
    }
    let physical: Vec<_> = (0..storage.iter().product())
        .map(|i| (((i * 13 + seed * 7) % 97) as f32 - 48.0) / 97.0)
        .collect();
    let backing = NativeTensor::from_data(TensorData::new(physical.clone(), Shape::from(storage.clone())))?;
    let mut axes: Vec<_> = (0..rank).collect();
    if rank >= 4 {
        axes.swap(rank - 1, rank - 2);
    }
    let tensor = backing
        .clone()
        .slice(&[Slice::new(1, Some((shape[0] + 1) as isize), 1)])?
        .permute(&axes)?
        .flip(&(0..rank).collect::<Vec<_>>())?;
    let logical = (0..shape.num_elements())
        .map(|mut i| {
            let mut c = vec![0; rank];
            for axis in (0..rank).rev() {
                c[axis] = shape[axis] - 1 - i % shape[axis];
                i /= shape[axis];
            }
            c[0] += 1;
            if rank >= 4 {
                c.swap(rank - 1, rank - 2);
            }
            let physical_index = c.iter().zip(&storage).fold(0, |at, (&c, &dim)| at * dim + c);
            f64::from(physical[physical_index])
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
fn words(tensor: &NativeTensor) -> Result<Vec<u32>, Error> {
    Ok(tensor
        .read()?
        .as_bytes()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
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
    indices: Vec<i32>,
    gradient: Vec<f64>,
}
struct Spec {
    input: [usize; 5],
    output: [usize; 5],
    kernel: [usize; 3],
    stride: [usize; 3],
    padding: [(usize, usize); 3],
    dilation: [usize; 3],
    mode: Mode,
    include: bool,
}

impl Spec {
    fn new<const D: usize>(shape: &Shape, options: &MaxPoolOptions<D>, include: Option<bool>, adaptive: Option<[usize; D]>) -> (Self, Shape) {
        let mut s = Self {
            input: [shape[0], shape[1], 1, 1, 1],
            output: [shape[0], shape[1], 1, 1, 1],
            kernel: [1; 3],
            stride: [1; 3],
            padding: [(0, 0); 3],
            dilation: [1; 3],
            mode: if adaptive.is_some() {
                Mode::Adaptive
            } else if include.is_some() {
                Mode::Average
            } else {
                Mode::Max
            },
            include: include.unwrap_or(false),
        };
        let mut output = shape.to_vec();
        for axis in 0..D {
            let a = 3 - D + axis;
            let input = shape[axis + 2];
            s.input[a + 2] = input;
            s.kernel[a] = options.kernel_size[axis];
            s.stride[a] = options.stride[axis];
            s.padding[a] = options.padding[axis];
            s.dilation[a] = options.dilation[axis];
            let value = if let Some(adaptive) = adaptive {
                adaptive[axis]
            } else {
                let effective = (s.kernel[a] - 1) * s.dilation[a] + 1;
                let available = input + s.padding[a].0 + s.padding[a].1 + if options.ceil_mode { s.stride[a] - 1 } else { 0 };
                let mut out = (available - effective) / s.stride[a] + 1;
                if options.ceil_mode && (out - 1) * s.stride[a] >= input + s.padding[a].0 {
                    out -= 1;
                }
                out
            };
            s.output[a + 2] = value;
            output[axis + 2] = value;
        }
        (s, Shape::from(output))
    }

    fn reference(&self, input: &[f64], gradient: &[f64]) -> Reference {
        let [n, c, d, h, w] = self.input;
        let [_, _, od, oh, ow] = self.output;
        let mut r = Reference {
            output: vec![0.0; n * c * od * oh * ow],
            indices: vec![-1; n * c * od * oh * ow],
            gradient: vec![0.0; input.len()],
        };
        for batch in 0..n {
            for channel in 0..c {
                for oz in 0..od {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let out = ((batch * c + channel) * od + oz) * oh * ow + oy * ow + ox;
                            let mut start = [0isize; 3];
                            let mut lengths = self.kernel;
                            let mut step = self.dilation;
                            for (axis, position) in [oz, oy, ox].into_iter().enumerate() {
                                if self.mode == Mode::Adaptive {
                                    let size = self.input[axis + 2];
                                    let count = self.output[axis + 2];
                                    start[axis] = (position * size / count) as isize;
                                    lengths[axis] = ((position + 1) * size).div_ceil(count) - start[axis] as usize;
                                    step[axis] = 1;
                                } else {
                                    start[axis] = (position * self.stride[axis]) as isize - self.padding[axis].0 as isize;
                                }
                            }
                            let mut sum = 0.0;
                            let mut maximum = f64::NEG_INFINITY;
                            let mut winner = None;
                            let mut denominator = 0usize;
                            let mut positions = Vec::new();
                            for z in 0..lengths[0] {
                                for y in 0..lengths[1] {
                                    for x in 0..lengths[2] {
                                        let p = [
                                            start[0] + (z * step[0]) as isize,
                                            start[1] + (y * step[1]) as isize,
                                            start[2] + (x * step[2]) as isize,
                                        ];
                                        let valid = p.iter().enumerate().all(|(a, &p)| p >= 0 && p < self.input[a + 2] as isize);
                                        let padded = p
                                            .iter()
                                            .enumerate()
                                            .all(|(a, &p)| p >= -(self.padding[a].0 as isize) && p < (self.input[a + 2] + self.padding[a].1) as isize);
                                        if if self.include && self.mode != Mode::Adaptive { padded } else { valid } {
                                            denominator += 1;
                                        }
                                        if !valid {
                                            continue;
                                        }
                                        let spatial = (p[0] as usize * h + p[1] as usize) * w + p[2] as usize;
                                        let index = (batch * c + channel) * d * h * w + spatial;
                                        let value = input[index];
                                        positions.push(index);
                                        sum += value;
                                        if winner.is_none() || value > maximum || value.is_nan() {
                                            maximum = value;
                                            winner = Some((index, spatial));
                                        }
                                    }
                                }
                            }
                            if self.mode == Mode::Max {
                                r.output[out] = maximum;
                                if let Some((index, spatial)) = winner {
                                    r.indices[out] = spatial as i32;
                                    r.gradient[index] += gradient[out];
                                }
                            } else {
                                r.output[out] = sum / denominator.max(1) as f64;
                                for index in positions {
                                    r.gradient[index] += gradient[out] / denominator.max(1) as f64;
                                }
                            }
                        }
                    }
                }
            }
        }
        r
    }
}

fn case<const D: usize>(backend: &Nvidia, shape: Shape, options: MaxPoolOptions<D>, include: Option<bool>, adaptive: Option<[usize; D]>) -> Result<(), Error> {
    let (spec, out) = Spec::new(&shape, &options, include, adaptive);
    let input = fixture(shape.clone(), 1)?;
    let gradient = fixture(out.clone(), 2)?;
    let expected = spec.reference(&input.logical, &gradient.logical);
    let average = AvgPoolOptions {
        kernel_size: options.kernel_size,
        stride: options.stride,
        padding: options.padding,
        count_include_pad: include.unwrap_or(false),
        ceil_mode: options.ceil_mode,
    };
    let before = backend.service_profile()?;
    let (output, indices) = if let Some(adaptive) = adaptive {
        (input.tensor.adaptive_avg_pool(adaptive)?, None)
    } else if include.is_some() {
        (input.tensor.avg_pool(&average)?, None)
    } else {
        let (v, i) = input.tensor.max_pool(&options)?;
        (v, Some(i))
    };
    let dx = if adaptive.is_some() {
        input.tensor.adaptive_avg_pool_backward(&gradient.tensor)?
    } else if include.is_some() {
        input.tensor.avg_pool_backward(&gradient.tensor, &average)?
    } else {
        input
            .tensor
            .max_pool_backward(&gradient.tensor, indices.as_ref().ok_or(Error::DeviceFault)?, &options)?
    };
    let after = backend.service_profile()?;
    let downloads = u64::from(spec.mode == Mode::Max && shape.num_elements() != 0);
    if [abi::UPLOAD, abi::ALLOCATE]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
        || after[abi::DOWNLOAD as usize].calls - before[abi::DOWNLOAD as usize].calls != downloads
        || output.shape() != out
        || dx.shape() != shape
    {
        return Err(Error::DeviceFault);
    }
    compare(&output, &expected.output)?;
    compare(&dx, &expected.gradient)?;
    if let Some(indices) = indices
        && (indices.dtype() != DType::I32 || words(&indices)? != expected.indices.iter().map(|&v| v as u32).collect::<Vec<_>>())
    {
        return Err(Error::DeviceFault);
    }
    for fixture in [&input, &gradient] {
        let expected: Vec<_> = fixture.physical.iter().map(|&v| f64::from(v)).collect();
        if read(&fixture.backing)? != expected {
            return Err(Error::DeviceFault);
        }
    }
    Ok(())
}

fn finite_difference(backend: &Nvidia) -> Result<(), Error> {
    let shape = Shape::new([1, 1, 4, 5]);
    let options = MaxPoolOptions::new([2, 3]).with_stride([1, 2]).with_padding([1, 1]).with_ceil_mode(true);
    let input = fixture(shape.clone(), 5)?;
    for mode in Mode::ALL {
        let include = (mode == Mode::Average).then_some(false);
        let adaptive = (mode == Mode::Adaptive).then_some([3, 7]);
        let (spec, out) = Spec::new(&shape, &options, include, adaptive);
        let gradient = fixture(out, 7)?;
        let dx = match mode {
            Mode::Max => {
                let (_, indices) = input.tensor.max_pool(&options)?;
                input.tensor.max_pool_backward(&gradient.tensor, &indices, &options)?
            }
            Mode::Average => input.tensor.avg_pool_backward(
                &gradient.tensor,
                &AvgPoolOptions::new([2, 3])
                    .with_stride([1, 2])
                    .with_padding([1, 1])
                    .with_count_include_pad(false)
                    .with_ceil_mode(true),
            )?,
            Mode::Adaptive => input.tensor.adaptive_avg_pool_backward(&gradient.tensor)?,
        };
        let objective = |x: &[f64]| {
            spec.reference(x, &gradient.logical)
                .output
                .iter()
                .zip(&gradient.logical)
                .map(|(&x, &g)| x * g)
                .sum::<f64>()
        };
        for (i, actual) in read(&dx)?.into_iter().enumerate() {
            let mut plus = input.logical.clone();
            let mut minus = plus.clone();
            plus[i] += 1e-4;
            minus[i] -= 1e-4;
            let expected = (objective(&plus) - objective(&minus)) / 2e-4;
            if !actual.is_finite() || (actual - expected).abs() > 2e-5 * (1.0 + expected.abs()) {
                return Err(Error::DeviceFault);
            }
        }
    }
    // A new command scope after all three backward paths remains usable.
    backend.with_commands(|| input.tensor.adaptive_avg_pool([1, 1]))?;
    Ok(())
}

fn max_edges(backend: &Nvidia) -> Result<(), Error> {
    let bits = [
        0x8000_0000,
        0,
        0x7fc0_1234,
        0x7fc0_5678,
        0xff80_0000,
        0xff80_0000,
        0x7f80_0000,
        1.0f32.to_bits(),
        0,
        0,
    ];
    let input = NativeTensor::from_data(TensorData::from_bytes_vec(
        bits.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Shape::new([1, 1, 2, 5]),
        DType::F32,
    ))?;
    let options = MaxPoolOptions::new([2, 3]).with_stride([1, 1]).with_padding_pairs([(2, 2), (3, 3)]);
    let (spec, out) = Spec::new(&input.shape(), &options, None, None);
    let expected = spec.reference(
        &bits.iter().map(|&v| f64::from(f32::from_bits(v))).collect::<Vec<_>>(),
        &vec![1.0; out.num_elements()],
    );
    let (output, indices) = input.max_pool(&options)?;
    let actual = words(&output)?;
    drop(output);
    let actual_indices = words(&indices)?;
    for ((actual, index), expected_index) in actual.into_iter().zip(actual_indices).zip(&expected.indices) {
        let expected = if *expected_index < 0 {
            f32::NEG_INFINITY.to_bits()
        } else {
            bits[*expected_index as usize]
        };
        if actual != expected || index != *expected_index as u32 {
            return Err(Error::DeviceFault);
        }
    }
    let gradient = NativeTensor::full(out, DType::F32, 1.0f32.to_bits())?;
    compare(&input.max_pool_backward(&gradient, &indices, &options)?, &expected.gradient)?;
    let tiny = NativeTensor::full(Shape::new([1, 1, 2]), DType::F32, 1)?;
    let tiny_grad = NativeTensor::full(Shape::new([1, 1, 1]), DType::F32, 2)?;
    let average = AvgPoolOptions::new([2]);
    if words(&tiny.avg_pool(&average)?)? != [1] || words(&tiny.avg_pool_backward(&tiny_grad, &average)?)? != [1, 1] {
        return Err(Error::DeviceFault);
    }
    let input = NativeTensor::full(Shape::new([1, 1, 2, 3]), DType::F32, 0)?;
    let options = MaxPoolOptions::new([2, 2]).with_stride([1, 1]);
    let gradient = NativeTensor::full(Shape::new([1, 1, 1, 2]), DType::F32, 1.0f32.to_bits())?;
    for bad in [-2i32, 6, i32::MIN, i32::MAX] {
        let indices = NativeTensor::from_data(TensorData::new(vec![bad, 0], [1, 1, 1, 2]))?;
        let before = backend.service_profile()?;
        if !matches!(
            backend.with_commands(|| input.max_pool_backward(&gradient, &indices, &options)),
            Err(Error::InvalidParameter)
        ) {
            return Err(Error::DeviceFault);
        }
        let after = backend.service_profile()?;
        if after[abi::FREE as usize].calls - before[abi::FREE as usize].calls
            != after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls
        {
            return Err(Error::DeviceFault);
        }
    }
    let indices = NativeTensor::from_data(TensorData::new(vec![-1i32, 0], [1, 1, 1, 2]))?;
    compare(
        &backend.with_commands(|| input.max_pool_backward(&gradient, &indices, &options))?,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )?;
    Ok(())
}

fn guarded_values(actual: &[u32], expected: &[f64], guard: u32) -> Result<(), Error> {
    if actual.len() != expected.len() + 2
        || actual[0] != guard
        || actual[actual.len() - 1] != guard
        || actual[1..actual.len() - 1].iter().zip(expected).any(|(&a, &b)| {
            let a = f64::from(f32::from_bits(a));
            !a.is_finite() || (a - b).abs() > 2e-5 * (1.0 + b.abs())
        })
    {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn guards(backend: &Nvidia) -> Result<(), Error> {
    let input = fixture(Shape::new([1, 2, 5, 7]), 3)?;
    let options = MaxPoolOptions::new([3, 2])
        .with_stride([2, 3])
        .with_padding_pairs([(1, 2), (0, 1)])
        .with_ceil_mode(true);
    let guard = 0x6bad_beef;
    let mut data = vec![guard];
    data.extend(input.physical.iter().map(|v| v.to_bits()));
    data.push(guard);
    let source = backend.upload_words(&data)?;
    for mode in Mode::ALL {
        let adaptive = (mode == Mode::Adaptive).then_some([3, 9]);
        let include = (mode == Mode::Average).then_some(true);
        let (spec, out) = Spec::new(&input.tensor.shape(), &options, include, adaptive);
        let gradient = fixture(out.clone(), 4)?;
        let expected = spec.reference(&input.logical, &gradient.logical);
        let plan = match mode {
            Mode::Max => Pooling::max(input.tensor.shape(), &options)?,
            Mode::Average => Pooling::average(
                input.tensor.shape(),
                &AvgPoolOptions::new([3, 2])
                    .with_stride([2, 3])
                    .with_padding_pairs([(1, 2), (0, 1)])
                    .with_ceil_mode(true),
            )?,
            Mode::Adaptive => Pooling::adaptive(input.tensor.shape(), &[3, 9])?,
        };
        let parameters = plan.parameters(input.tensor.layout(), false)?;
        let count = out.num_elements();
        let storage = if mode == Mode::Max { count * 2 } else { count };
        let output = backend.upload_words(&vec![guard; storage + 2])?;
        backend.dispatch_words(abi::Kernel::Pool, &[(&source, 1), (&output, 1)], &parameters)?;
        let mut actual = vec![0; storage + 2];
        backend.read_words_into(&output, 0, &mut actual)?;
        if mode == Mode::Max {
            if actual[1..count + 1] != expected.indices.iter().map(|&v| v as u32).collect::<Vec<_>>() || actual[0] != guard || actual[storage + 1] != guard {
                return Err(Error::DeviceFault);
            }
            let mut values = vec![guard];
            values.extend_from_slice(&actual[count + 1..storage + 1]);
            values.push(guard);
            guarded_values(&values, &expected.output, guard)?;
        } else {
            guarded_values(&actual, &expected.output, guard)?;
        }
        let mut malformed = parameters;
        malformed[24] = input.tensor.layout().offset();
        if backend.dispatch_words(abi::Kernel::Pool, &[(&source, 1), (&output, 1)], &malformed) != Err(Error::InvalidParameter)
            || backend.dispatch_words(abi::Kernel::Pool, &[(&source, 1), (&source, 1)], &parameters) != Err(Error::InvalidParameter)
        {
            return Err(Error::DeviceFault);
        }
        let mut unchanged = vec![0; actual.len()];
        backend.read_words_into(&output, 0, &mut unchanged)?;
        if unchanged != actual {
            return Err(Error::DeviceFault);
        }
        if mode != Mode::Max {
            let mut grad_data = vec![guard];
            grad_data.extend(gradient.physical.iter().map(|v| v.to_bits()));
            grad_data.push(guard);
            let grad_source = backend.upload_words(&grad_data)?;
            let backward = plan.parameters(gradient.tensor.layout(), true)?;
            let target = backend.upload_words(&vec![guard; input.logical.len() + 2])?;
            backend.dispatch_words(abi::Kernel::PoolBackward, &[(&grad_source, 1), (&target, 1)], &backward)?;
            let mut actual = vec![0; input.logical.len() + 2];
            backend.read_words_into(&target, 0, &mut actual)?;
            guarded_values(&actual, &expected.gradient, guard)?;
            let mut malformed = backward;
            malformed[24] = gradient.tensor.layout().offset();
            if backend.dispatch_words(abi::Kernel::PoolBackward, &[(&grad_source, 1), (&target, 1)], &malformed) != Err(Error::InvalidParameter) {
                return Err(Error::DeviceFault);
            }
            let mut unchanged = vec![0; actual.len()];
            backend.read_words_into(&target, 0, &mut unchanged)?;
            if unchanged != actual {
                return Err(Error::DeviceFault);
            }
            let mut unchanged = vec![0; grad_data.len()];
            backend.read_words_into(&grad_source, 0, &mut unchanged)?;
            if unchanged != grad_data {
                return Err(Error::DeviceFault);
            }
        }
    }
    let mut unchanged = vec![0; data.len()];
    backend.read_words_into(&source, 0, &mut unchanged)?;
    if unchanged != data {
        return Err(Error::DeviceFault);
    }
    let wrong = NativeTensor::full(input.tensor.shape(), DType::I32, 0)?;
    let gradient = NativeTensor::full(Shape::new([1, 2, 1, 1]), DType::F32, 0)?;
    let before = backend.queue_profile()?;
    if !matches!(wrong.max_pool(&options), Err(Error::InvalidParameter))
        || !matches!(
            input.tensor.avg_pool_backward(&gradient, &AvgPoolOptions::new([3, 2])),
            Err(Error::ShapeMismatch)
        )
        || !matches!(input.tensor.adaptive_avg_pool([0, 1]), Err(Error::InvalidParameter))
        || before.launches != backend.queue_profile()?.launches
    {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    case(
        backend,
        Shape::new([2, 2, 17]),
        MaxPoolOptions::new([3])
            .with_stride([2])
            .with_padding_pairs([(1, 2)])
            .with_dilation([2])
            .with_ceil_mode(true),
        None,
        None,
    )
    .map_err(|e| format!("max pool 1d: {e}"))?;
    case(
        backend,
        Shape::new([2, 3, 7, 9]),
        MaxPoolOptions::new([3, 2])
            .with_stride([2, 3])
            .with_padding_pairs([(1, 2), (0, 1)])
            .with_dilation([2, 2])
            .with_ceil_mode(true),
        None,
        None,
    )
    .map_err(|e| format!("max pool 2d: {e}"))?;
    case(
        backend,
        Shape::new([1, 2, 4, 5, 7]),
        MaxPoolOptions::new([2, 3, 2])
            .with_stride([1, 2, 2])
            .with_padding_pairs([(1, 0), (1, 1), (0, 1)])
            .with_dilation([1, 1, 2])
            .with_ceil_mode(true),
        None,
        None,
    )
    .map_err(|e| format!("max pool 3d: {e}"))?;
    for include in [true, false] {
        case(
            backend,
            Shape::new([2, 2, 17]),
            MaxPoolOptions::new([4]).with_stride([3]).with_padding([1]).with_ceil_mode(true),
            Some(include),
            None,
        )
        .map_err(|e| format!("avg pool 1d pad={include}: {e}"))?;
        case(
            backend,
            Shape::new([2, 3, 7, 9]),
            MaxPoolOptions::new([3, 2])
                .with_stride([2, 3])
                .with_padding_pairs([(1, 2), (0, 1)])
                .with_ceil_mode(true),
            Some(include),
            None,
        )
        .map_err(|e| format!("avg pool 2d pad={include}: {e}"))?;
        case(
            backend,
            Shape::new([1, 2, 5, 5, 5]),
            MaxPoolOptions::new([3, 3, 3])
                .with_stride([3, 3, 3])
                .with_padding([1, 1, 1])
                .with_ceil_mode(true),
            Some(include),
            None,
        )
        .map_err(|e| format!("avg pool 3d pad={include}: {e}"))?;
    }
    for (input, output) in [(17, 5), (5, 9), (65537, 65539)] {
        case(backend, Shape::new([1, 1, input]), MaxPoolOptions::new([1]), None, Some([output]))
            .map_err(|e| format!("adaptive pool 1d {input}->{output}: {e}"))?;
    }
    for output in [[3, 4], [9, 8]] {
        case(backend, Shape::new([1, 2, 5, 7]), MaxPoolOptions::new([1, 1]), None, Some(output)).map_err(|e| format!("adaptive pool 2d {output:?}: {e}"))?;
    }
    case(backend, Shape::new([1, 2, 3, 5, 7]), MaxPoolOptions::new([1, 1, 1]), None, Some([4, 2, 3])).map_err(|e| format!("adaptive pool 3d: {e}"))?;
    for shape in [Shape::new([0, 2, 5, 7]), Shape::new([1, 0, 5, 7])] {
        for include in [None, Some(false)] {
            case(backend, shape.clone(), MaxPoolOptions::new([2, 3]), include, None).map_err(|e| format!("empty pooling: {e}"))?;
        }
    }
    finite_difference(backend).map_err(|e| format!("pool finite differences: {e}"))?;
    max_edges(backend).map_err(|e| format!("pool edges: {e}"))?;
    guards(backend).map_err(|e| format!("pool guards: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_POOLING_OK modes=max,avg,adaptive dimensions=1,2,3 gradients=fp64 finite-difference=ok nan=last ties=first subnormals=preserved bounds=recoverable guards=ok backend=nvidia"
    ));
    Ok(())
}
