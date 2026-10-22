use d3os_compute::{Backend, Conv2d, Error, Pool2d, Tensor, cpu::Cpu, element_count, output_extent};

fn tensor(shape: &[usize], data: Vec<f32>) -> Tensor<Cpu> {
    Tensor::from_vec(&Cpu, shape, data).unwrap()
}

fn near(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && (a - b).abs() <= 1e-5 + 1e-5 * b.abs(), "{a} != {b}");
    }
}

#[test]
fn rectangular_matmul_and_linear_use_distinct_weight_layouts() {
    let a = tensor(&[2, 3], vec![1., -2., 3., 4., 0., -1.]);
    let b = tensor(&[3, 2], vec![2., 1., 0., -3., -1., 2.]);
    near(Cpu.values(&Cpu.matmul(&a, &b).unwrap()), &[-1., 13., 9., 2.]);
    let weights = tensor(&[2, 3], vec![2., 0., -1., 1., -3., 2.]);
    let bias = tensor(&[2], vec![0.5, -0.5]);
    near(Cpu.values(&Cpu.linear(&a, &weights, &bias).unwrap()), &[-0.5, 12.5, 9.5, 1.5]);
}

#[test]
fn linear_handles_vectorized_blocks_and_remainders() {
    for width in [1, 7, 8, 9, 17, 33] {
        let input: Vec<_> = (0..2 * width).map(|i| (i as f32 - 5.) / 3.).collect();
        let weights: Vec<_> = (0..3 * width).map(|i| ((i * 7) % 11) as f32 / 8.).collect();
        let expected: Vec<_> = input
            .chunks_exact(width)
            .flat_map(|row| {
                weights
                    .chunks_exact(width)
                    .map(move |weight| row.iter().zip(weight).map(|(a, b)| a * b).sum::<f32>() + 0.25)
            })
            .collect();
        let output = Cpu
            .linear(&tensor(&[2, width], input), &tensor(&[3, width], weights), &tensor(&[3], vec![0.25; 3]))
            .unwrap();
        near(Cpu.values(&output), &expected);
    }
}

#[test]
fn im2col_convolution_matches_direct_convolution_with_batch_padding_and_stride() {
    let (batch, channels, h, w, outputs, kh, kw) = (2, 2, 5, 7, 3, 3, 2);
    let config = Conv2d {
        stride: [2, 1],
        padding: [1, 2],
    };
    let oh = output_extent(h, kh, 2, 1).unwrap();
    let ow = output_extent(w, kw, 1, 2).unwrap();
    let image: Vec<_> = (0..batch * channels * h * w).map(|i| ((i * 13) % 23) as f32 / 7. - 1.).collect();
    let weights: Vec<_> = (0..outputs * channels * kh * kw).map(|i| ((i * 3) % 7) as f32 / 9. - 0.5).collect();
    let bias = [0.25, -1., 2.];
    let mut expected = Vec::new();
    for n in 0..batch {
        for o in 0..outputs {
            for y in 0..oh {
                for x in 0..ow {
                    let mut sum = bias[o];
                    for c in 0..channels {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (y * 2 + ky) as isize - 1;
                                let ix = (x + kx) as isize - 2;
                                if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                    sum += image[((n * channels + c) * h + iy as usize) * w + ix as usize] * weights[((o * channels + c) * kh + ky) * kw + kx];
                                }
                            }
                        }
                    }
                    expected.push(sum);
                }
            }
        }
    }
    let result = Cpu
        .conv2d(
            &tensor(&[batch, channels, h, w], image),
            &tensor(&[outputs, channels, kh, kw], weights),
            &tensor(&[outputs], bias.to_vec()),
            config,
        )
        .unwrap();
    assert_eq!(result.shape(), [batch, outputs, oh, ow]);
    near(Cpu.values(&result), &expected);
}

#[test]
fn pooling_keeps_negative_maxima_and_relu_preserves_shape() {
    let input = tensor(&[1, 1, 3, 4], vec![-8., -2., -9., -4., -3., -6., -1., -5., -7., -4., -2., -8.]);
    let pooled = Cpu
        .max_pool2d(
            &input,
            Pool2d {
                kernel: [2, 2],
                stride: [1, 2],
            },
        )
        .unwrap();
    assert_eq!(pooled.shape(), [1, 1, 2, 2]);
    near(Cpu.values(&pooled), &[-2., -1., -3., -1.]);
    let activated = Cpu.relu(pooled).unwrap();
    assert_eq!(activated.shape(), [1, 1, 2, 2]);
    near(Cpu.values(&activated), &[0.; 4]);
}

#[test]
fn softmax_is_stable_and_normalizes_each_row() {
    let input = tensor(&[2, 3], vec![1000., 1001., 1002., -1000., -999., -998.]);
    let result = Cpu.softmax(&input).unwrap();
    let values = Cpu.values(&result);
    near(&values[..3], &values[3..]);
    near(&[values[..3].iter().sum(), values[3..].iter().sum()], &[1., 1.]);
    assert!(values[0] < values[1] && values[1] < values[2]);
    assert!(matches!(Cpu.softmax(&tensor(&[1, 1], vec![f32::INFINITY])), Err(Error::NonFinite)));
}

#[test]
fn invalid_dimensions_fail_before_allocation_or_indexing() {
    assert_eq!(element_count(&[usize::MAX, 2]), Err(Error::SizeOverflow));
    assert_eq!(element_count(&[1, 0]), Err(Error::InvalidShape));
    assert_eq!(output_extent(3, 4, 1, 0), Err(Error::ShapeMismatch));
    assert_eq!(output_extent(3, 1, 0, 0), Err(Error::InvalidParameter));
    assert_eq!(output_extent(3, 1, 1, usize::MAX), Err(Error::SizeOverflow));
    assert!(matches!(Tensor::from_vec(&Cpu, &[2, 2], vec![1.]), Err(Error::ShapeMismatch)));
    assert!(matches!(tensor(&[2], vec![1., 2.]).reshape(&[3]), Err(Error::ShapeMismatch)));
    let a = tensor(&[2, 3], vec![0.; 6]);
    let b = tensor(&[2, 2], vec![0.; 4]);
    assert!(matches!(Cpu.matmul(&a, &b), Err(Error::ShapeMismatch)));
}
