# Burn migration and training goal

Burn will provide the public AI framework for D3OS.
The migration covers tensor operations, model definitions, training, and inference.
The native D3OS driver and precompiled cuTile kernels remain the GPU execution stack.
DOOM keeps its native renderer and remains part of the separate Linux performance comparison.

This work extends the [native NVIDIA performance goal](nvidia-performance.md).
Qwen3.5-0.8B, Qwen3.5-9B Q8_0, Qwen3.6-27B Q3_K_M, and AlexNet must all migrate to Burn.
The goal also includes an MNIST convolutional model trained and exported inside D3OS.
Correctness alone does not complete the goal. Matched Linux performance comparisons remain required.

## Architecture

Applications use Burn tensors and modules through the D3OS AI library.
The D3OS backend owns tensor storage, native operation dispatch, synchronization, and GPU profiling.
The kernel driver owns GPU allocation, execution channels, and process isolation.
The host build compiles Rust cuTile kernels into GPU binaries before the VM starts.
Training decisions, backward passes, optimizer steps, and model export run inside the VM.

The backend must keep model weights and activations in GPU memory during GPU execution.
CPU execution is an explicit device choice. Unsupported GPU operations cannot silently execute on the CPU.
Shared tensors must preserve values needed by other tensors and backward passes.
Queued operations must retain their buffers until completion.
Allocation failures and GPU faults must reach the application without corrupting later work.

Burn's backend interface covers more operations than the current inference interface.
The implementation needs tensor views, broadcasting, reductions, indexing, and integer and boolean operations.
Training also needs gradient accumulation, convolution and pooling backward operations, and optimizer updates.

Fused Qwen operations remain native backend extensions.
These include packed quantized projections, normalization, attention, and recurrent state updates.
Extensions must preserve packed weight formats and avoid unnecessary intermediate tensors or host transfers.
An inference extension does not acquire a backward implementation merely through registration.
Training requires a correct backward implementation or an equivalent composition of differentiable primitives.

The public backend must integrate with Burn's device dispatch and autodiff.
A private implementation of its backend traits alone does not complete that integration.
The migration must use a pinned release and record any required upstream changes.
The D3OS target uses `no_std` with `alloc`. Host-only training interfaces are outside the guest dependency graph.

## Migration sequence

1. Pin Burn and validate the portable dependency graph on the D3OS target.
2. Implement and validate the native backend, including shared tensor ownership and command batching.
3. Add training operations and run the MNIST workflow inside the VM.
4. Migrate AlexNet with the existing eight layer references and matched Linux benchmark.
5. Migrate Qwen, including its optimized extensions, streaming loading, tokenizer, and sessions.
6. Validate 0.8B, 9B, and 27B correctness, GPU fit, prompt throughput, and generation throughput.
7. Remove replaced tensor and model APIs after their consumers migrate.
8. Run the complete regression suite and continue optimization against matched Linux measurements.

Existing implementations provide references during the migration.
The final architecture must not retain duplicate public tensor frameworks or unused compatibility layers.
The low-level NVIDIA interfaces remain available to the native DOOM renderer.

## MNIST acceptance criteria

The first training workload uses the full MNIST training set and the separate test set.
The host can stage the dataset and compile kernels. It cannot train the guest model.
The guest reports the seed, model configuration, batch size, optimizer, learning rate, and epoch count.
It also reports training loss, test accuracy, throughput, elapsed time, and GPU memory use.
The test accuracy target is at least 99 percent.

Gradient checks and an independent host reference validate the training operations.
A saved checkpoint contains weights, optimizer state, progress, and the random generator state.
Resuming the checkpoint must preserve the subsequent training sequence within the stated numerical tolerance.
The guest exports an inference artifact with model configuration, preprocessing information, and named weights.
The inference API reloads that artifact and reproduces the trained model's predictions.
Keeping artifacts across VM restarts requires persistent storage or a reliable host export path.

Initial numerical checks use FP32.
Mixed precision adds BF16 Tensor Core operations with FP32 accumulation and optimizer state.
The measurements must identify which precision each workload uses.

## Inference and performance acceptance criteria

AlexNet retains all eight existing reference comparisons and their original tolerances.
Qwen retains its layer, batch, cache, and tokenizer checks.
Large Qwen models also need independent numerical validation and complete output traces.
Their weights, context, and working storage must fit within the physical GPU's 16 GiB of VRAM.

Measurements separate file access, upload, allocation, prompt processing, and generation.
Each workload uses the same model, input, precision, context, and amount of work on Linux and D3OS.
The migration needs measurements before and after each workload change.
The existing retained measurements provide context, not a substitute for current matched measurements.
Compiler changes cannot justify relaxed output tolerances or an unreported performance regression.

DOOM retains its pixel references, cache checks, gameplay controls, and cleanup checks.
Its report includes uncapped benchmark FPS and the regular gameplay frame cap.
The titlebar provides FPS. The removed corner overlay remains removed.
Burn is not a dependency of the renderer.

## Current status

The native stack runs AlexNet, Qwen3.5-0.8B, and DOOM before the Burn migration.
Packed quantized GPU operations and grouped recurrent heads are validated independently.
Native 9B and 27B model integration remains in progress.
Burn 0.22.1 now compiles and links on the D3OS target with `no_std` and `alloc`.
Its module optimizer needs a [local portability patch](../third_party/README.md).
The D3OS AI library supplies native clock and timer wakeups for the portable runtime.
The first VM validation uses Burn's explicit Flex CPU device.
It passes analytic gradients, SGD updates, shared-tensor preservation, Adam checkpoint resume, and native timer wakeups.
The MNIST model definition uses Burn convolutions, pooling, and linear layers.
The native backend now has a Burn tensor primitive with shared GPU storage and checked tensor views.
Host checks cover every three-dimensional permutation and reversal, composed slices, broadcasts, overlapping windows, empty tensors, and invalid address ranges.
Physical GPU checks pass view addressing, aliasing, lifetime, exact floating-point bits, boolean storage, and empty tensors.
View construction performs no GPU allocation or transfer. The existing GPU operator checks also pass.
These checks validate storage. Burn device dispatch, native arithmetic, and autodiff integration still need implementation.
The native Burn GPU backend, full dataset training, and workload migration remain incomplete.
The retained Linux and D3OS measurements are in [Native NVIDIA performance](nvidia-performance.md).

## Portable runtime validation

Run the host checks:

```bash
cargo test --locked -p d3os-ai --features cpu,training
cargo clippy --locked -p d3os-ai --features cpu,training --all-targets -- -D warnings
```

Run the D3OS target check:

```bash
RUSTFLAGS='--cfg getrandom_backend="rdrand"' cargo clippy --locked \
  -p d3os-ai --features cpu,training \
  -Zbuild-std=core,alloc -Zbuild-std-features=compiler-builtins-mem \
  -Zjson-target-spec --target "$PWD/d3os_application.json" --no-deps -- -D warnings
```

After the production image build, run the VM validation:

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/run-qemu.py \
  --firmware target/nvidia/firmware --kernels target/nvidia/kernels \
  --memory-mib 4096 --boot-check --inference burn-self-test --timeout 120
```

The success marker starts with `INFERENCE_OK burn-self-test backend=cpu`.
This check establishes portable runtime behavior. It does not measure native Burn GPU training or inference performance.
