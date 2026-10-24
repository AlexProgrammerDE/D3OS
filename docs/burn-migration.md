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

Complete the native Burn backend before rewriting AI workloads against its public API.
Then migrate every AI workload and remove the replaced APIs.
Resume performance optimization after that migration. Correctness and regression checks continue throughout the work.
DOOM remains separate from Burn.

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

The existing native inference stack runs AlexNet and Qwen3.5-0.8B on the GPU.
DOOM uses its separate native renderer. These workloads have not migrated to Burn.
Native Qwen 9B and 27B model integration remains in progress.
Packed quantized GPU operations and grouped recurrent heads are validated independently.

Burn 0.22.1 compiles and links on D3OS with `no_std` and `alloc`.
Its optimizer needs a [local portability patch](../third_party/README.md).
The D3OS AI library supplies clock and timer wakeups for the portable runtime.
The explicit Flex CPU device passes analytic gradients, SGD updates, shared-tensor preservation, Adam checkpoint resume, and native timer wakeups inside D3OS.
The MNIST model definition uses Burn convolutions, pooling, and linear layers.

The native tensor primitive owns shared GPU storage and checked views.
It supports FP32, I32, and boolean storage, including exact floating-point bit preservation.
View construction performs no GPU allocation or transfer.
The GPU operations include FP32 arithmetic and math, reductions, exact tensor fill, and FP32 Tensor Core matrix products.
Another 29 entries implement I32 arithmetic, signed comparisons, bitwise operations, boolean logic, masks, and conversions.
These operations support strided views and broadcasting. Input tensors remain shared and immutable.
Full writers avoid a separate output clear. Empty logical outputs require no element work.
The matching SDK and driver use native compute protocol version 17.

The D3OS GPU self-test passes reversed and transposed views, overlapping windows, broadcast batches, subviews, empty dimensions, and neighboring guards.
Matrix references use independent FP64 accumulation. Math references use independent libm functions and include 2,816 FP32 edge values.
Checks cover every exponent range, rounding boundaries, signed zeros, nonfinite inputs, canonical booleans, and first-index extrema ties.
The driver rejects invalid address ranges and malformed launches before dispatch.
Resident-operation counters show no host activation transfers during the checked GPU operations.
The CI operator suite includes the portable AI library and native tensor package.
Typed-word GPU checks pass wrapping arithmetic, shift boundaries, signed comparisons, and conversions across every FP32 exponent range.
Boolean-to-I32 casts share resident storage. Ownership and service-counter checks confirm that they require no GPU allocation, transfer, or launch.
All 118 earlier GPU instruction sections remain byte-identical after the typed-word entries are added.

Native gather and vector select now support FP32, I32, and boolean tensors with checked, signed strides.
Indices remain on the GPU. A four-byte bounds result returns to the host after each nonempty operation.
This result requires synchronization. It prevents invalid device indices from escaping as successful tensors.
Invalid indices return an error, release private output storage, and leave the GPU usable for subsequent operations.
The checks cover reversed views, subviews, repeated indices, empty axes, command scopes, partial tiles, and neighboring guards.
All 147 earlier GPU instruction sections remain byte-identical after the gather entry is added.

Native scatter accumulation adds repeated FP32 and I32 values, or combines boolean values with OR.
Vector select accumulation uses the same checked view expansion as select.
The operations copy the destination on the GPU to preserve shared tensors.
Noncontiguous values also require a GPU copy. Indices retain their original signed strides.
Each nonempty operation reads a four-byte bounds result and synchronizes before exposing the output.
FP32 accumulation uses raw-word compare-and-swap with non-FTZ addition to preserve subnormal inputs and results.
Colliding additions have an unspecified order. Results can therefore differ by floating-point rounding between runs.
These primitives provide gradient accumulation for gather and select; they do not implement Burn autodiff registration.
Physical GPU checks pass all three operations across every axis of strided tensors, including repeated indices and offset views.
They also pass alias preservation, invalid-index recovery, guarded partial tiles, NaN handling, and exact subnormal accumulation.
A weighted gather loss and its scatter gradient agree with an independent FP64 finite-difference reference.
All 148 earlier GPU instruction sections remain byte-identical after the three scatter entries are added.

Native convolution now provides FP32 forward passes and input, weight, and optional bias gradients.
It supports NCHW tensors, groups, dilation, asymmetric padding, and signed views of inputs and weights.
The forward kernel uses compensated TF32 Tensor Core products and includes optional bias in the same launch.
Backward passes use resident matrix products, checked window extraction, and a fold kernel that sums overlapping windows without atomics.
Physical GPU checks compare forward results and all gradients with independent FP64 references and finite differences.
The checks include partial tiles, empty batches, offset views, shared storage, malformed metadata, and neighboring guards.
Service counters show no activation upload or download during these operations.
All 151 earlier GPU instruction sections remain byte-identical after the three convolution entries are added.
The existing AlexNet execution path remains unchanged. AI workload migration waits for the public Burn backend.

Native pooling provides max, average, and adaptive average operations with backward passes in one to three spatial dimensions.
The plans accept signed views, per-side padding, ceil windows, and max-pool dilation.
Max pooling preserves the first equal value and the last NaN, including its index and stored bits.
An empty window returns negative infinity and index -1. Backward ignores that sentinel and rejects other invalid device indices.
The value and index tensors share one initialized GPU allocation.
Average divisors distinguish explicit padding from the extra extent of a ceil window.
Adaptive pooling supports larger and smaller output dimensions, with 64-bit intermediate ratios.
Average backward operations gather overlapping contributions without atomics or host transfers.
Max backward uses checked scatter accumulation and reads one four-byte bounds flag before returning.
Physical D3OS GPU checks pass independent FP64 forward and gradient references, finite differences, signed views, and neighboring guards.
They also pass NaN payloads, tie selection, subnormal arithmetic, invalid-index recovery, empty dimensions, and adaptive ratios beyond 32-bit products.
All 154 earlier GPU instruction sections remain byte-identical. AlexNet, Qwen, and DOOM functional regression checks also pass.

Native masked selection copies exact FP32, I32, and boolean words from the selected input.
All three inputs support broadcasting and signed views. The mask requires a resident GPU copy when its layout is noncontiguous.
Slice assignment preserves shared tensors through a private GPU copy, including when the replacement shares overlapping source storage.
Concatenation and padding use checked resident copies. Partial writers always start from initialized destination storage.
The driver proves that writable views contain distinct addresses and rejects overlapping windows and broadcast destinations.

Physical D3OS GPU checks pass all three data types, signed views, broadcasting, overlapping-source snapshots, and concatenation across every axis.
They also pass exact NaN payloads, signed zeros, empty tensors, guarded partial tiles, and rejected invalid writes inside command scopes.
Service counters show no activation upload or download. All 159 earlier GPU instruction sections remain byte-identical.
AlexNet, Qwen, and DOOM functional regressions pass with the expanded bundle.

Native random generation uses Philox4x32-10 for raw I32 words, FP32 uniform and normal values, Bernoulli samples, and unbiased bounded integers.
The generator accepts Burn distribution types and supports FP32, I32, and boolean tensors.
Integer normal samples use the resident FP32 normal generator and a checked native cast.
FP32 Bernoulli samples use a resident I32-to-FP32 cast. No random tensor values originate on the host.

Each stream owns a seed and a checked 64-bit block counter. A mutex serializes reservation, submission, and state updates.
Each request reserves `ceil(elements / 4)` blocks. Partial final blocks discard their unused words.
Integer rejection retries use the counter's upper 64 bits and do not consume another request's reserved blocks.
Invalid requests and failed submissions leave the stream state unchanged. Empty tensors reserve no blocks and launch no element work.
A versioned, 24-byte checkpoint stores the seed and next block in little-endian form.
Checkpoint restore requires a training boundary with no concurrent stream users.

Physical D3OS GPU checks pass anchored Philox words, 64-bit counter carries, partial tiles, exact checkpoint replay, and guarded writes.
FP64 references validate normal samples. Scalar integer references validate rejection retries and all supported data types.
Service counters show no host upload or download during generation. All 161 earlier GPU instruction sections remain byte-identical.
Host tests also pass concurrent reservations, aborted submissions, malformed checkpoints, and counter exhaustion.
AlexNet, Qwen, and DOOM functional regressions pass with the expanded bundle.

The native backend now registers with Burn's public device dispatch and autodiff.
Applications open it through `d3os_ai::native_device()` and enable gradients with `device.autodiff()`.
The backend uses a [local registration patch](../third_party/README.md) for Burn's closed backend catalog.
CPU execution remains an explicit device choice. Native operations do not silently execute on the CPU.

A ready tensor owns real GPU storage. A failed tensor contains metadata and its original error, with no placeholder allocation.
Dependent operations propagate that error without allocating or launching GPU work.
Native readback and synchronization report failures. Recoverable errors leave later valid native operations usable.
A fatal device error remains poisoned and requires a process restart.

Physical D3OS checks pass public tensor views, matrix products, native error propagation, and convolution derivatives for inputs, weights, and bias.
They also pass analytic SGD and Adam checkpoint resume through Burn's public module and optimizer APIs.
The convolutional MNIST model produces gradients for all eight parameter tensors on synthetic inputs.
One SGD step decreases its loss and preserves a retained model clone.
The training step uploads no activations and downloads only the two four-byte bounds flags for max-pool gradients.
Batched gradient products use initialized GPU allocations because individual matrix launches write only one batch.

I32 reductions now provide sum, product, mean, extrema, and first-index selection without FP32 conversion.
Sum and product wrap at 32 bits. Mean divides the wrapped sum once and truncates toward zero.
Empty sums return zero, and empty products return one. Empty integer means and extrema return errors.
GPU index generation checks length and signed endpoints before dispatch and performs no host tensor upload.
These operations support strided and broadcast views and avoid separate output clears.

Physical D3OS checks pass exact references beyond FP32 integer precision, signed extrema, overflow, ties across tiles, and neighboring guards.
Burn cross-entropy passes an analytic loss and gradient reference. Its accuracy count uses resident integer reduction.
Four four-byte bounds flags return to the host before explicit result reads.
They come from target selection, maximum selection for stable log-softmax, and their two backward scatters.
All 166 earlier GPU instruction sections remain byte-identical after the eight new entries are added.
AlexNet, Qwen, and all 600 DOOM reference views pass with the matching SDK and kernel bundle.

Operator coverage remains incomplete. Unsupported native operations return execution errors instead of CPU results.
The gaps include integer division, integer matrix products, scans, some FP32 math, compaction, and sorting.
Generic Burn quantization, non-additive indexed updates, three-dimensional convolutions, transposed convolutions, and interpolation also remain incomplete.
Qwen's packed-weight extensions still need public Burn integration.
Full MNIST dataset training, complete checkpoint export, and AI workload migration remain incomplete.
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
