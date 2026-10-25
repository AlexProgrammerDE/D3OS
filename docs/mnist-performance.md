# MNIST GPU performance

The benchmark uses the shared Burn model and trainer on Linux CUDA and the native D3OS NVIDIA backend.
Both guests use the same RTX 4090 Laptop GPU, with one VM at a time.
Each guest has four virtual CPUs and 4 GiB of memory, with individual CPU pinning to host CPUs 8, 10, 12, and 14.

## Work and numerical references

The input artifact is the native seven-epoch model with 99.26 percent test accuracy.
Its SHA-256 is `d87a681e2cda7ed1c3301f334053265c76dc7beeb8e9c08e2d18120a6299061e`.
Each inference pass classifies all 10,000 test images. Batch sizes are 1, 64, and 256, including each final partial batch.
A complete warmup pass precedes three timed passes at each batch size.
The runners retain all 100,000 logits at each batch size after the timed work.

Each training repetition starts from the same artifact, fresh Adam state, and seed 42.
It uses batch size 64, learning rate 0.0001, beta values 0.9 and 0.999, epsilon 0.00001, and no AMSGrad.
Four warmup steps precede 64 timed steps, which process 4,096 images.
The runners retain every timed loss, 2,560 post-training logits, and all eight trained parameter tensors.
The three repetitions finish at step 68 and sample offset 4,352 in epoch zero.

Training timing includes input normalization, upload, forward execution, all eight gradients, Adam updates, synchronization, and the loss read.
Inference timing includes normalization, upload, forward execution, argmax, accuracy calculation, synchronization, and prediction reads.
Compilation, model loading, autotuning, warmup, reference export, and logging remain outside these intervals.
The model-load records have different runtime initialization boundaries and do not establish cold-start parity.

The strict numerical oracle uses Burn Flex on the Linux host with the same public model and trainer.
Its timings never enter the GPU comparison.
D3OS passes all 448,062 oracle values with zero failures at `2e-5 * (1 + abs(reference))`.
This is the existing convolution tolerance. The initial inference logits differ by at most `1.52587890625e-5`.
The comparison also verifies that training changes the starting parameters.

## Matrix precision differs

FP32 storage does not guarantee FP32 matrix arithmetic in the Linux reference.
The pinned Cubek implementation [promotes FP32 operands to TF32 for accelerated matrix products](https://docs.rs/crate/cubek-matmul/0.3.1/source/src/multi_level/definition/blueprint.rs).
Its convolution implementation uses the same selection helper.
D3OS retains FP32 arithmetic, with TF32x3 products on Tensor Cores.

The stock Linux run preserves all 10,000 classifications and prediction CRC32 `ce40d6c3`, but differs from D3OS by up to 0.015051 in logits.
The strict cross-backend comparison reports 292,696 tolerance failures.
The CPU oracle confirms D3OS's stricter numerical results without increasing the tolerance.
The comparison JSON records both arithmetic policies and marks performance precision as unmatched.
The stock Linux times remain a speed target. They cannot establish identical-precision parity.

Linux uses Rust 1.98.1 because its CUDA dependencies require newer features than D3OS's pinned compiler provides.
D3OS uses `nightly-2026-02-01`. Both applications target `x86-64-v3`.
The result manifests record the compiler versions, source hashes, input hashes, binary hashes, and CPU placement.

## Initial GPU timing cohort

Each cell is the mean of three timed intervals in one VM.
These are end-to-end GPU workload times, not isolated kernel event times.

| Work | Linux VM, Burn CUDA, TF32 allowed | D3OS VM, native Burn, FP32 | D3OS / Linux |
| --- | ---: | ---: | ---: |
| Inference, batch 1, 10,000 images | 2,667.736 ms | 1,579.000 ms | 0.592 |
| Inference, batch 64, 10,000 images | 70.431 ms | 48.000 ms | 0.682 |
| Inference, batch 256, 10,000 images | 27.293 ms | 30.000 ms | 1.099 |
| Training, batch 64, 64 steps | 88.054 ms | 257.000 ms | 2.919 |

Training remains the clearest gap in this cohort.
Additional VM repetitions and the matrix precision difference remain relevant to any parity claim.
The first functional Linux run overlapped a host rebuild. Its timings remain excluded in `linux-functional-overlap/`.

Accepted GPU evidence is under `target/nvidia/mnist-parity/linux-matched-v3/` and `d3os-baseline/`.
The strict numerical evidence is under `cpu-reference/`.
`comparison-baseline-oracle.json` separates the GPU timing target from the CPU numerical oracle.

## Reproduce the comparison

Build the production D3OS image and prepare the normal NVIDIA firmware and portable kernel inputs first.
Then build the Linux reference from its own directory:

```bash
cd tools/mnist-benchmark
cargo build --release --locked
cargo build --release --locked --no-default-features --features cpu \
  --target-dir ../../target/mnist-cpu-oracle
cd ../..

MNIST_MODEL="$HOME/.cache/d3os/mnist-trained/d87a681e2cda7ed1c3301f334053265c76dc7beeb8e9c08e2d18120a6299061e/mnist.model"

taskset -c 8,10,12,14 python3 tools/nvidia/benchmark-mnist.py \
  --os linux --model "$MNIST_MODEL" \
  --directory target/nvidia/mnist-parity/linux-new

taskset -c 8,10,12,14 python3 tools/nvidia/benchmark-mnist.py \
  --os d3os --model "$MNIST_MODEL" \
  --directory target/nvidia/mnist-parity/d3os-new

taskset -c 8,10,12,14 python3 tools/nvidia/benchmark-mnist.py \
  --os linux --cpu-reference --model "$MNIST_MODEL" \
  --directory target/nvidia/mnist-parity/oracle-new

python3 tools/nvidia/compare-mnist.py \
  --linux target/nvidia/mnist-parity/linux-new \
  --d3os target/nvidia/mnist-parity/d3os-new \
  --oracle target/nvidia/mnist-parity/oracle-new \
  --output target/nvidia/mnist-parity/comparison-new.json
```

Use fresh output directories. Keep host builds and CPU reference work separate from timed GPU runs.
Without `--oracle`, the comparison enforces the same tolerance directly against stock Linux CUDA and reports the precision mismatch as a failure.
Burn's own pack reader decodes the saved model parameters without a GPU.
