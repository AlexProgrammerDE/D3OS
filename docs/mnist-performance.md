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

## Batch convolution gradients

The native gradient path now combines samples and output positions into one matrix dimension.
Each requested gradient needs one Tensor Core matrix launch per group, instead of one per sample and group.
Burn's input-gradient and weight-gradient callbacks also compute only their requested result.
The combined gradient API shares its prepared output-gradient matrix between both products.
Every intermediate remains on the GPU, and fold still sums overlapping windows without atomic updates.

Three fresh training repetitions record 163, 157, and 158 ms for 64 steps, with a mean of 159.333 ms.
This D3OS cohort is 1.613 times faster than the initial 257 ms mean, reducing elapsed time by 38.0 percent.
It remains 1.810 times the retained stock Linux CUDA mean of 88.054 ms, with the arithmetic difference described above.
The change therefore reduces the training gap without establishing parity.

All 448,062 strict FP32 oracle values still pass with zero tolerance failures.
The complete NVIDIA operator suite also passes grouped, dilated, strided, empty-batch, and finite-difference convolution checks.
The operator checks count matrix launches for combined and separately requested gradients.
Inference means remain 1,594.667, 47.333, and 31.000 ms at batches 1, 64, and 256, respectively.
Every inference pass preserves 99.26 percent accuracy and prediction CRC32 `ce40d6c3`.

Evidence is under `target/nvidia/mnist-parity/d3os-batched-backward/` and `backward-operators/`.
The numerical and timing comparison is `comparison-batched-backward.json`.

## Separate training diagnostics

The native profile command runs eight training steps with ordinary counters, then eight steps with GPU timestamps.
Device timestamps force individual submissions and change execution time. These intervals never enter the throughput comparison.
The profiler also measures each matrix shape with resident constant inputs and preallocated outputs.
These isolated products exclude tensor views, allocation, upload, and optimizer work.

Before split reductions, an ordinary training step launches 196 kernels.
Matrix products account for about 51 percent of the instrumented GPU interval.
The first convolution's weight gradient multiplies `[16, 36864]` by `[36864, 25]`.
It uses one GPU block and takes 909 microseconds per isolated call in the retained diagnostic.
The second convolution's weight gradient takes 107 microseconds for `[32, 4096]` by `[4096, 400]`.
This identifies long reductions with few output tiles as the next optimization target.

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/benchmark-mnist.py \
  --os d3os --profile-only --model /absolute/path/to/mnist.model \
  --directory target/nvidia/mnist-parity/new-profile
```

The profile manifest marks `performance_comparison=false` and retains kernel, queue, service, and matrix diagnostics.
The initial evidence is under `target/nvidia/mnist-parity/matrix-profile-before/`.

### Split long matrix reductions

The native matrix request now accepts an optional split count.
Each block writes one partition of the reduction into temporary GPU storage.
A second kernel combines the partial results with compensated FP32 summation.
Both stages preserve the existing TF32x3 product arithmetic. No intermediate returns to the CPU.
The host exports portable Tile IR, and D3OS compiles the new entry for the detected GPU.
The separate host instruction audit verifies Tensor Core instructions and no register spills in that entry.

Burn selects splits from the matrix dimensions, targeting 128 output blocks with at most one split per 256 reduction elements.
It retains the ordinary path for short reductions or matrices with enough output tiles.
The measured MNIST weight-gradient shapes select 128 and eight splits, respectively.

The following D3OS diagnostic means include the final partial-sum kernel and use 16 resident calls per configuration.
They exclude ordinary training throughput and cannot establish Linux parity.

| Gradient matrix | D3OS, unsplit | D3OS, selected split count |
| --- | ---: | ---: |
| `[16, 36864]` by `[36864, 25]` | 909.366 microseconds | 18.104 microseconds, 128 splits |
| `[32, 4096]` by `[4096, 400]` | 106.998 microseconds | 29.958 microseconds, eight splits |

The split sweep is under `target/nvidia/mnist-parity/split-matrix-profile/`.
Portable inputs and host instruction audits are under `split-portable-kernels/` in the same parent directory.

The complete GPU operator suite passes with the new dispatch path.
New matrix checks cover uneven tiles, empty split ranges, input offsets, output guards, reversed views, and broadcast batches.
Their independent FP64 references use the established `2e-5 * (1 + abs(reference))` convolution bound.
The maximum scaled error across the new broadcast cases is `6.9341977e-6`.
Existing basic matrix checks retain their `1e-6 * (1 + abs(reference))` bound.
Accepted operator evidence is under `target/nvidia/mnist-parity/split-operators-reference/`.

The ordinary D3OS training repetitions record 97, 96, and 96 ms for 64 steps, with a mean of 96.333 ms.
This is 1.654 times faster than the preceding batched-gradient cohort, reducing elapsed time by 39.5 percent.
It is 2.668 times faster than the initial 257 ms mean.
The retained Linux CUDA mean is 88.054 ms, leaving D3OS 9.4 percent slower in this cohort.
The different matrix arithmetic and Linux timing spread still prevent an identical-precision parity claim.

All 448,062 strict CPU oracle values pass with zero tolerance failures, including every trained parameter and timed loss.
Inference means are 1,600.000, 46.667, and 30.000 ms at batches 1, 64, and 256.
Every inference repetition preserves 99.26 percent accuracy and prediction CRC32 `ce40d6c3`.
The batch-256 inference mean remains 9.9 percent above the retained Linux mean.

Ordinary execution evidence is under `target/nvidia/mnist-parity/d3os-split-matmul/`.
The comparison is `comparison-split-matmul.json` in the same parent directory.
It retains the Linux GPU speed target and the independent CPU numerical oracle as separate references.
AlexNet also passes its eight-layer Burn validation after the shared matrix dispatch change.
That regression log is under `target/nvidia/mnist-parity/split-alexnet/`; its load-inclusive interval is not a warm inference measurement.

## Fuse native Adam updates

The post-split diagnostic records 198 launches per training step, including 144 generic elementwise launches.
Those generic kernels account for about 55 percent of instrumented GPU time.
The diagnostic is under `target/nvidia/mnist-parity/training-profile-after-split/` and remains outside the throughput comparison.

Burn's optional native optimizer feature now dispatches one FP32 Adam kernel per parameter.
It retains the public optimizer API, timestep, moment tensors, and checkpoint format.
The parameter and moments occupy separate views of one new GPU allocation. Inputs and shared snapshots remain immutable.
AMSGrad adds its historical maximum to that allocation. Weight decay and gradient clipping retain their existing preprocessing paths.
The kernel uses explicit FP32 rounding, with the original bias correction and epsilon placement.
Other devices retain Burn's tensor-operation implementation.

The GPU conformance oracle uses canonical CPU arrays built from the native views' logical values.
This keeps the CPU reference independent of native view implementation.
Burn Flex 0.22.1 panics when Adam consumes an empty flipped CPU tensor; the separate Linux host probe reproduces this defect.
The native tests retain their reversed, transposed, and empty inputs.
The first diagnostic retains the host probe, guest console, and incomplete operator result under `adam-doom-operators/`.

The fused operation uses Burn's extension dispatcher to preserve untracked autodiff and checkpointing contexts.
Optional restored moments participate in that dispatch. Tracked optimizer inputs retain the differentiable tensor-operation path.
The complete GPU operator suite passes both checkpointing strategies, tracked gradients, and MNIST checkpoint resume.
Guarded direct updates preserve moment bits exactly against separate FP32 CPU operations.
Other checks cover repeated updates, AMSGrad reconfiguration, weight decay, shared snapshots, reversed views, and empty tensors.
Accepted operator evidence is under `target/nvidia/mnist-parity/adam-context-operators/`.

The separate post-fusion profile records 102 launches per step, down from 198.
Generic elementwise launches fall from 144 to 40, with eight fused Adam launches per step.
This removes 96 launches per step, or 48.5 percent. The profile remains excluded from ordinary performance comparisons.
Its evidence is under `target/nvidia/mnist-parity/training-profile-fused-adam/`.

Ordinary D3OS training repetitions take 79, 80, and 78 ms for 64 steps, with a mean of 79.000 ms.
This is 1.219 times faster than the preceding 96.333 ms cohort and 3.253 times faster than the initial 257 ms mean.
All 448,062 strict CPU oracle values pass with zero tolerance failures.

A fresh Linux CUDA cohort uses the rebuilt reference, matching inputs, and the same work contract.
Its training repetitions take 118.034, 90.850, and 96.999 ms, with a mean of 101.961 ms.
The earlier Linux mean remains 88.054 ms. All trials remain retained, including the faster 67.088 ms earlier Linux interval.
The timing spread and matrix arithmetic difference remain relevant to parity claims.

| Work | Fresh Linux VM, Burn CUDA, TF32 allowed | D3OS VM, native Burn, FP32 |
| --- | ---: | ---: |
| Inference, batch 1, 10,000 images | 3,699.530 ms | 1,577.000 ms |
| Inference, batch 64, 10,000 images | 59.774 ms | 48.333 ms |
| Inference, batch 256, 10,000 images | 30.568 ms | 30.000 ms |
| Training, batch 64, 64 steps | 101.961 ms | 79.000 ms |

Every inference repetition preserves 99.26 percent accuracy and prediction CRC32 `ce40d6c3`.
The older Linux batch-256 mean remains 27.293 ms, below D3OS's 30.000 ms. That inference gap remains open.

Fresh execution evidence is under `target/nvidia/mnist-parity/linux-fused-adam-cohort/` and `d3os-fused-adam/`.
The comparison files are `comparison-fused-adam-fresh-linux.json` and `comparison-fused-adam-retained-linux.json` in the same parent directory.
Both comparisons retain the independent CPU numerical oracle and explicitly mark matrix performance precision as unmatched.

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
