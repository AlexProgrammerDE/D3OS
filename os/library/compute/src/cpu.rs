use alloc::vec::Vec;

use crate::{Backend, Buffer, Conv2d, Error, Pool2d, Result, Tensor, element_count, output_extent};

mod transformer;

#[derive(Debug)]
pub struct CpuBuffer(Vec<f32>);

impl Buffer for CpuBuffer {
    fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Cpu;

impl Cpu {
    pub fn values<'a>(&self, tensor: &'a Tensor<Self>) -> &'a [f32] {
        &tensor.buffer.0
    }
}

fn zeros(shape: &[usize]) -> Result<Vec<f32>> {
    let count = element_count(shape)?;
    let mut data = Vec::new();
    data.try_reserve_exact(count).map_err(|_| Error::AllocationFailed)?;
    data.resize(count, 0.0);
    Ok(data)
}

// Rows of B and output are contiguous. The inner loop can use SIMD without
// reassociating floating-point reductions or enabling fast-math globally.
fn gemm(a: &[f32], b: &[f32], out: &mut [f32], rows: usize, inner: usize, cols: usize) {
    for row in 0..rows {
        let output = &mut out[row * cols..(row + 1) * cols];
        for k in 0..inner {
            let value = a[row * inner + k];
            let input = &b[k * cols..(k + 1) * cols];
            for (dest, &rhs) in output.iter_mut().zip(input) {
                *dest += value * rhs;
            }
        }
    }
}

impl Backend for Cpu {
    type Buffer = CpuBuffer;

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn upload(&self, values: Vec<f32>) -> Result<CpuBuffer> {
        Ok(CpuBuffer(values))
    }

    fn download(&self, tensor: &Tensor<Self>) -> Result<Vec<f32>> {
        let mut copy = zeros(tensor.shape())?;
        copy.copy_from_slice(self.values(tensor));
        Ok(copy)
    }

    fn matmul(&self, a: &Tensor<Self>, b: &Tensor<Self>) -> Result<Tensor<Self>> {
        let ([rows, inner], [other_inner, cols]) = (a.shape(), b.shape()) else {
            return Err(Error::ShapeMismatch);
        };
        if inner != other_inner {
            return Err(Error::ShapeMismatch);
        }
        let shape = [*rows, *cols];
        let mut out = zeros(&shape)?;
        gemm(self.values(a), self.values(b), &mut out, *rows, *inner, *cols);
        Tensor::from_vec(self, &shape, out)
    }

    fn linear(&self, input: &Tensor<Self>, weights: &Tensor<Self>, bias: &Tensor<Self>) -> Result<Tensor<Self>> {
        let ([batch, features], [outputs, weight_features], [bias_outputs]) = (input.shape(), weights.shape(), bias.shape()) else {
            return Err(Error::ShapeMismatch);
        };
        if features != weight_features || outputs != bias_outputs {
            return Err(Error::ShapeMismatch);
        }
        let shape = [*batch, *outputs];
        let mut out = zeros(&shape)?;
        for (row, dest) in self.values(input).chunks_exact(*features).zip(out.chunks_exact_mut(*outputs)) {
            for ((weights, dest), &bias) in self.values(weights).chunks_exact(*features).zip(dest).zip(self.values(bias)) {
                // Independent accumulators allow vectorization of long dot products.
                let mut sums = [0.0; 8];
                let mut lhs = row.chunks_exact(8);
                let mut rhs = weights.chunks_exact(8);
                for (a, b) in lhs.by_ref().zip(rhs.by_ref()) {
                    for lane in 0..8 {
                        sums[lane] += a[lane] * b[lane];
                    }
                }
                let tail: f32 = lhs.remainder().iter().zip(rhs.remainder()).map(|(&a, &b)| a * b).sum();
                *dest = sums.iter().sum::<f32>() + tail + bias;
            }
        }
        Tensor::from_vec(self, &shape, out)
    }

    fn conv2d(&self, input: &Tensor<Self>, weights: &Tensor<Self>, bias: &Tensor<Self>, config: Conv2d) -> Result<Tensor<Self>> {
        let ([batch, channels, height, width], [outputs, weight_channels, kh, kw], [bias_outputs]) = (input.shape(), weights.shape(), bias.shape()) else {
            return Err(Error::ShapeMismatch);
        };
        if channels != weight_channels || outputs != bias_outputs {
            return Err(Error::ShapeMismatch);
        }
        let oh = output_extent(*height, *kh, config.stride[0], config.padding[0])?;
        let ow = output_extent(*width, *kw, config.stride[1], config.padding[1])?;
        let shape = [*batch, *outputs, oh, ow];
        let mut out = zeros(&shape)?;
        // Check products before address arithmetic and reuse scratch for each image.
        let inner = element_count(&[*channels, *kh, *kw])?;
        let spatial = element_count(&[oh, ow])?;
        let mut columns = zeros(&[inner, spatial])?;
        let image_size = element_count(&[*channels, *height, *width])?;
        for n in 0..*batch {
            let image = &self.values(input)[n * image_size..(n + 1) * image_size];
            for c in 0..*channels {
                for ky in 0..*kh {
                    for kx in 0..*kw {
                        let column = &mut columns[((c * kh + ky) * kw + kx) * spatial..((c * kh + ky) * kw + kx + 1) * spatial];
                        for y in 0..oh {
                            let iy = (y * config.stride[0] + ky).checked_sub(config.padding[0]);
                            for x in 0..ow {
                                let ix = (x * config.stride[1] + kx).checked_sub(config.padding[1]);
                                column[y * ow + x] = match (iy, ix) {
                                    (Some(iy), Some(ix)) if iy < *height && ix < *width => image[(c * height + iy) * width + ix],
                                    _ => 0.0,
                                };
                            }
                        }
                    }
                }
            }
            let output = &mut out[n * outputs * spatial..(n + 1) * outputs * spatial];
            gemm(self.values(weights), &columns, output, *outputs, inner, spatial);
            for (channel, &bias) in output.chunks_exact_mut(spatial).zip(self.values(bias)) {
                for value in channel {
                    *value += bias;
                }
            }
        }
        Tensor::from_vec(self, &shape, out)
    }

    fn relu(&self, mut input: Tensor<Self>) -> Result<Tensor<Self>> {
        for value in &mut input.buffer.0 {
            // Preserve NaNs so numerical errors remain visible to callers.
            if *value < 0.0 {
                *value = 0.0;
            }
        }
        Ok(input)
    }

    fn max_pool2d(&self, input: &Tensor<Self>, config: Pool2d) -> Result<Tensor<Self>> {
        let [batch, channels, height, width] = input.shape() else {
            return Err(Error::ShapeMismatch);
        };
        let oh = output_extent(*height, config.kernel[0], config.stride[0], 0)?;
        let ow = output_extent(*width, config.kernel[1], config.stride[1], 0)?;
        let shape = [*batch, *channels, oh, ow];
        let mut out = zeros(&shape)?;
        for (image, output) in self.values(input).chunks_exact(height * width).zip(out.chunks_exact_mut(oh * ow)) {
            for y in 0..oh {
                for x in 0..ow {
                    let mut largest = f32::NEG_INFINITY;
                    for ky in 0..config.kernel[0] {
                        for kx in 0..config.kernel[1] {
                            let value = image[(y * config.stride[0] + ky) * width + x * config.stride[1] + kx];
                            if value.is_nan() || value > largest {
                                largest = value;
                            }
                        }
                    }
                    output[y * ow + x] = largest;
                }
            }
        }
        Tensor::from_vec(self, &shape, out)
    }

    fn softmax(&self, input: &Tensor<Self>) -> Result<Tensor<Self>> {
        let [_, cols] = input.shape() else {
            return Err(Error::ShapeMismatch);
        };
        let mut out = self.download(input)?;
        for row in out.chunks_exact_mut(*cols) {
            if row.iter().any(|v| !v.is_finite()) {
                return Err(Error::NonFinite);
            }
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for value in row.iter_mut() {
                *value = libm::expf(*value - max);
                sum += *value;
            }
            for value in row {
                *value /= sum;
            }
        }
        Tensor::from_vec(self, input.shape(), out)
    }
}
