# Burn migration and training goal

Burn provides the public AI framework for D3OS.
The migration covers tensor operations, model definitions, training, and inference.
The native D3OS driver executes cuTile kernels compiled from portable Tile IR inside D3OS.
DOOM keeps its native renderer and remains part of the separate Linux performance comparison.

This work extends the [native NVIDIA performance goal](nvidia-performance.md).
Qwen3.5-0.8B, Qwen3.5-9B Q8_0, Qwen3.6-27B Q3_K_M, and AlexNet now use Burn.
The goal also includes an MNIST convolutional model trained and exported inside D3OS.
Correctness alone does not complete the goal. Matched Linux performance comparisons remain required.

## Primary goal: Linux backend API parity

Implement the full native Burn backend before returning to workload-specific performance optimization.
D3OS applications must be able to use the backend APIs supported by the pinned Burn Linux CUDA backend on the same GPU.
The current comparison uses Burn 0.22.1 and the RTX 4090 Laptop GPU.
Passing AlexNet, MNIST, and Qwen does not establish complete backend support.

The implementation must cover these areas:

| Area | Required behavior |
| --- | --- |
| Tensor operations | Floating-point, integer, boolean, indexing, reductions, scans, sorting, compaction, and view operations |
| Quantization | Public Burn quantization, dynamic calibration, packed storage, quantization parameters, dequantization, views, and quantized operations |
| Dtypes | The storage, conversion, arithmetic, and accelerated capabilities supported by the Linux reference on the same hardware |
| Modules and training | Supported convolution variants, pooling, interpolation, activations, backward operations, gradient accumulation, and optimizer updates |
| Data and model records | Quantized and ordinary tensor upload, download, serialization, model export, and reload |
| Execution | Device dispatch, synchronization, transactions, graph capture and replay, and asynchronous buffer lifetime |
| Semantics | Shared tensor preservation, broadcasting, strided views, empty shapes, numerical edge cases, and observable error behavior |

Audit every backend trait method, including methods inherited from Burn defaults.
Record support by operation, dtype, shape restrictions, and quantization scheme.
Distinguish an optimized kernel from a correct composition of resident GPU operations.
An implemented trait containing unsupported branches is not evidence of complete support.
Optional features require comparison with the Linux backend's actual capabilities on the available hardware.

Unsupported GPU operations must not silently execute on the CPU.
Use Burn's public APIs for general capabilities instead of adding parallel application-only interfaces.
The existing packed GGUF extensions can remain optimized implementations, but they do not replace public quantization support.
Activation quantization is authorized. Match the Linux comparison's precision and report its numerical effects explicitly.
Keep the independent FP32 path and references for validation; do not require all inference arithmetic to remain FP32.

Acceptance requires upstream backend conformance checks where portable, independent numerical references, and execution inside D3OS on the GPU.
Tests must cover quantized storage and parameters, shared and strided tensors, boundary shapes, serialization, and error propagation.
No unsupported operation or dtype advertised as supported by the matched Linux backend may remain without an explicit unresolved entry.
An unresolved entry keeps this goal incomplete.

The [backend coverage report](burn-backend-coverage.md) records the current inventory, matched capability declarations, and unresolved areas.

After backend parity, resume the full [performance acceptance table](nvidia-performance.md#performance-goal-and-acceptance).
It retains cold PTX and Tile IR compilation, DOOM FPS, MNIST training and inference, AlexNet, and all three Qwen variants.
Compilation and loading remain outside execution averages.

## Architecture

Applications use Burn tensors and modules through the D3OS AI library.
The D3OS backend owns tensor storage, native operation dispatch, synchronization, and GPU profiling.
The kernel driver owns GPU allocation, execution channels, and process isolation.
The host build exports portable Tile IR from Rust cuTile kernels.
D3OS runs the pinned NVIDIA compiler in userspace and selects machine code for the detected GPU.
The native driver retains the generated modules for the lifetime of the VM.
Compilation and validation are load phases outside model throughput measurements.
Training decisions, backward passes, optimizer steps, and model export run inside the VM.

The backend must keep model weights and activations in GPU memory during GPU execution.
CPU execution is an explicit device choice. Unsupported GPU operations cannot silently execute on the CPU.
Shared tensors must preserve values needed by other tensors and backward passes.
Queued operations must retain their buffers until completion.
Allocation failures and GPU faults must reach the application without corrupting later work.

Native compound operations use `Nvidia::record_commands()` to group their kernels.
Inside a parent scope, recording joins the parent's completion without adding a wait between layers.
A standalone recording scope waits before returning.
`with_commands()` retains explicit completion, including nested calls. Burn device synchronization also waits for outstanding work.
Reads, process cleanup, and page-table growth retain their completion barriers.

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
The host can stage the dataset and export portable kernel input. It cannot train the guest model.
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

AlexNet now uses Burn modules and public tensors on the native GPU and the explicit Flex CPU device.
Its legacy model implementation and execution API have been removed.
Qwen3.5-0.8B now uses Burn modules, packed parameters, sessions, and generation.
Its original layer, logit, cache, and tokenizer references pass on the physical GPU.
The previous inference crate is removed.
DOOM keeps its separate native renderer.
Native Qwen 9B and 27B now use the same Burn model and streaming GGUF loader.
Both pass independent FP32 layer and logit references at 165 positions, with matching sequential and batched final caches.
Packed quantized GPU operations and grouped recurrent heads are validated independently.

Burn 0.22.1 compiles and links on D3OS with `no_std` and `alloc`.
Its optimizer needs a [local portability patch](../third_party/README.md).
The D3OS AI library supplies clock and timer wakeups for the portable runtime.
The explicit Flex CPU device passes analytic gradients, SGD updates, shared-tensor preservation, Adam checkpoint resume, and native timer wakeups inside D3OS.
The MNIST model definition uses Burn convolutions, pooling, and linear layers.

The native tensor primitive owns shared GPU storage and checked views.
It supports 15 ordinary storage formats with their physical widths, including exact floating-point bit preservation.
View construction performs no GPU allocation or transfer.
The GPU operations include FP32 arithmetic and math, reductions, exact tensor fill, and FP32 Tensor Core matrix products.
Another 31 entries implement I32 arithmetic, signed comparisons, bitwise operations, boolean logic, masks, and conversions.
These operations support strided views and broadcasting. Input tensors remain shared and immutable.
Full writers avoid a separate output clear. Empty logical outputs require no element work.
The matching SDK and driver use native compute protocol version 32 and the 215-entry portable inference bundle.

The D3OS GPU self-test passes reversed and transposed views, overlapping windows, broadcast batches, subviews, empty dimensions, and neighboring guards.
Matrix references use independent FP64 accumulation. Math references use independent libm functions and include 2,816 FP32 edge values.
Checks cover every exponent range, rounding boundaries, signed zeros, nonfinite inputs, canonical booleans, and first-index extrema ties.
The driver rejects invalid address ranges and malformed launches before dispatch.
Resident-operation counters show no host activation transfers during the checked GPU operations.
The CI operator suite includes the portable AI library and native tensor package.
Typed-word GPU checks pass wrapping arithmetic, shift boundaries, signed comparisons, and conversions across every FP32 exponent range.
Boolean-to-I32 casts normalize nonzero records to one through a resident full writer.
Ownership and service-counter checks confirm independent output storage without activation transfers or a separate clear.
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
At this convolution-primitive milestone, AlexNet still used the legacy execution path.
The later model migration now uses Burn.

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
Resident I32 division and matrix products, FP32 erf, and cumulative operations now pass native GPU checks.
Sorting, argsort, top-k, FP32 cross products, coordinate extraction, and mask selection also pass their checked cases.
All single-axis scatter update modes pass matched public checks on D3OS and Linux CUDA.
Vector-selection updates also support all five modes through resident scatter and broadcast index views.
Multidimensional gather/scatter now pass native FP32/I32 checks, including scalar forward results and duplicate addition.
The [coverage report](burn-backend-coverage.md) records their shapes, numerical semantics, Linux failures, and remaining dtype restrictions.
The gaps include further dtype arithmetic, quantized conversion variants, additional FP32 math, further compaction formats, and scalar autodiff.
Complete Burn quantization, three-dimensional convolutions, transposed convolutions, and interpolation also remain incomplete.
The native backend also uses Burn's `GraphUnsupported` primitive, so captured graph replay remains a gap.
Its advertised dtype capabilities cover F32, I32 storage, U32 boolean storage, and the quantization storage described below.
Physical storage and ordinary numeric casts also support the other ordinary formats.
Their broader quantized and boolean-producing conversion paths remain incomplete, so the capability declarations stay disabled.
These formats do not yet provide general arithmetic or accelerated execution.
The public `QTensorOps` methods now execute for six integer formats, two FP8 formats, and FP4 with five scale dtypes.
Two-level schemes use an F32 tensor scale. Optimized public quantized matrix products remain open.
Packed GGUF projection support does not close those remaining public API or dtype gaps.
Qwen now has public Burn extensions for packed projections, embeddings, gates, and normalization.
Attention caches and recurrent state now use public Burn tensors.
The full model, loading, and generation now use these Burn extensions.
The native MNIST runner now accepts all 60,000 training images and the separate 10,000-image test split.
It supports complete Adam checkpoints, deterministic sample order, native random state, and inference export.
A physical D3OS run completes seven epochs and reaches 99.26 percent test accuracy.
The final model uses five epochs at learning rate 0.001 and two at 0.0001, without resetting Adam state.
A fresh VM loads the exported model and reproduces all 10,000 test predictions.
See [Train MNIST inside D3OS](mnist-training.howto.md) for commands and validation evidence.
The shared MNIST trainer now builds with the `training` feature independently of the native backend.
`Trainer::from_artifact` loads the same checked inference weights with fresh Adam state and a fresh deterministic sample order.
This API supports matched Linux and D3OS starting conditions. It does not resume a training checkpoint.
Complete checkpoint restore still preserves native random state and Adam state through the existing native API.
CPU tests cover imported weights, a subsequent gradient update, and invalid artifact rejection with and without the native feature.
AlexNet, all three Qwen profiles, and MNIST now use Burn. Performance optimization remains open.
The AlexNet migration preserves every value of all eight original layer references with absolute and relative tolerances of `2e-4`.
The host CPU reference and physical D3OS GPU both return class 258.
Forward execution uploads and downloads no activation data.
Its classifier weight layout is materialized once during loading through resident Burn tensor operations.
Tensor-file parsing now lives in the independent `d3os-tensor-file` package, with no execution backend dependency.

Fresh 100-pass measurements report 0.930 ms for legacy D3OS and 2.420 ms for Burn on D3OS.
Fresh Linux cuDNN/cuBLAS runs report 0.731 to 0.735 ms.
The 2.60-fold AlexNet regression remains open. Both model families now use Burn before further optimization.

The Qwen extension layer stores packed weights in immutable Burn integer parameters.
Its separate matrix metadata describes F32, BF16, Q8_0, Q3_K, Q4_K, Q5_K, or Q6_K rows.
Burn module records preserve the packed words and their execution device.
The matrix configuration supplies the logical shape and encoding when a record loads.
The shared format decoder now lives in `d3os-tensor-file`, without an execution dependency.

Native extensions retain the cuTile projection, split projection, embedding, gate, and normalization kernels.
Residual fusion copies shared storage before mutation. Other Burn tensors retain their original values.
Physical D3OS tests and explicit host CPU tests pass independent scalar and GGUF references.
Native forward checks report zero activation uploads and downloads. Host token IDs remain explicit embedding inputs.
Malformed weights fail before GPU allocation. Failed Burn inputs retain their original execution error.
These inference extensions reject autodiff devices because they do not implement backward passes.
The Qwen model and sessions now use Burn. The legacy model implementation and inference crate are removed.
This step adds no Linux or D3OS throughput measurement.
Evidence is under `target/nvidia/burn/qwen-packed/`.

Attention and recurrent operations now consume an owned state and return its continuation with the output tensor.
Cloned states retain their original cache values. Native updates reuse unique storage and copy shared storage on the GPU.
Validated constructors also accept resident cache tensors for restoration without a host readback.
A failed operation attaches the same error to its output and both cache tensors, without placeholder allocations.

Independent binary64 fixtures cover eight attention cases and sixteen recurrent cases across single-token and batched execution.
The cases include grouped heads, nonzero cache prefixes, signed views, long-context attention splits, and wider fallback dimensions.
Packed recurrent projection tests cover BF16 fusion, FP32 matrices, and mixed encodings.
Physical D3OS checks retain all native kernel paths and report zero activation uploads or downloads during state updates.
The native kernel bundle remains unchanged.

The fixture generator uses Python math and rounded FP32 inputs and cache writes.
Regenerate its data with `python3 tools/inference/prepare-burn-state-fixture.py`.
The fixture and provenance manifest live under `os/library/ai/tests/fixtures/`.
Logs and the fresh legacy Qwen baseline are under `target/nvidia/burn/qwen-state/`.
The subsequent model migration uses these state extensions. The Linux performance goal remains open.

The complete Qwen model, streaming weight interface, tokenizer, sessions, and generation now live in `d3os-ai`.
The old inference crate and its model execution API are removed. The host example and application callers use Burn.
Packed weights remain integer parameters in Burn module records, while normalization and convolution parameters use floating tensors.
The same model definition supports the explicit Flex CPU device and native NVIDIA device.
A session owns its caches and rejects further work after execution or observer errors.
Model outputs are resident Burn tensors. Greedy selection uses a native Burn extension.
The GPU reduces logits into 32 candidate pairs, which the host combines after one 256-byte read.
The selection preserves the first global maximum and rejects any nonfinite input, including values in losing partitions.
The explicit CPU backend preserves the input float precision and logical view order.
Selection detaches its borrowed logits without changing the caller's autodiff graph.

Physical D3OS verification passes every original 29-position, 24-layer reference and the complete final cache comparisons.
The original `2e-3` absolute and `5e-4` relative tolerances remain unchanged.
All 105 tokenizer cases pass, and streaming generation returns `4` for the default arithmetic prompt.
Burn module-record tests preserve packed parameters and identical predictions.
Synthetic CPU batch comparisons use `3e-5` absolute and relative tolerance because Flex matrix operations can use different accumulation orders.

The fresh D3OS comparison records 238.806 warm decode tokens/s after migration, versus 251.969 before migration.
Initial model preparation increases from 615 ms to 21880 ms.
Bulk packed-weight copies reduce it to 1940 ms in a subsequent D3OS GPU run.
The remaining preparation and generation gaps need fixes in the maintained Burn stack.
Large-model native correctness now passes; matched Linux parity remains open.

The retained Linux and D3OS measurements are in [Native NVIDIA performance](nvidia-performance.md).

## Public integer quantization

Burn's public `quantize`, `quantize_dynamic`, `dequantize`, and quantized data import and export now use native GPU storage and cuTile kernels.
The supported values are `Q8S`, `Q8F`, `Q4S`, `Q4F`, `Q2S`, and `Q2F`.
The `S` variants use symmetric integer ranges. The `F` variants use the full signed integer range.
These are integer formats; `Q8F` does not mean FP8.

The backend supports one F32 scale per tensor or per rectangular block.
Dynamic calibration uses Burn's min/max and scale calculations through resident GPU primitives.
Public one-dimensional scale tensors are reshaped to their block grid after their element counts are checked.
The block grid follows each tensor axis, including blocks that do not span the innermost dimension.

Quantized values occupy packed device words, with four, eight, or sixteen values per word.
Eight-bit values also support Burn's native byte storage format.
Packed storage can use any supported tensor axis, with that axis preserved through transposition and serialization.
Packed dimensions must contain complete words, and tensor dimensions must contain complete quantization blocks.
General reshapes of multidimensional block schemes remain restricted.

Quantized tensors own their values and scales through shared GPU storage.
Permutations and flips preserve the values required by other tensors.
Serialization copies packed values on the GPU before reading them and retains the scale grid's storage order.
Native byte imports accept both compact values and alignment padding before F32 scales.
Output conversion uses FP32; other output dtypes remain part of the dtype implementation work.

```rust
use d3os_ai::burn::tensor::{
    Tensor,
    quantization::{QuantScheme, QuantValue, ScaleDtype},
};

let device = d3os_ai::native_device()?;
let input = Tensor::<2>::ones([4, 32], &device);
let scheme = QuantScheme::default()
    .with_value(QuantValue::Q4S)
    .per_block([1, 32], ScaleDtype::F32);
let packed = input.quantize_dynamic(&scheme);
let restored = packed.dequantize();
```

The D3OS GPU check passes independent nearest-even rounding and packed-byte references for all six formats.
It covers serialization and reload, shared inputs, transposition, outer-axis packing, flips, simple reshapes, rectangular block scales, dynamic calibration, zero inputs, empty tensors, and native byte tails.
Guarded raw launches verify partial words and output boundaries for 8-, 4-, and 2-bit values.
Forged view metadata also remains inside the input and output allocations.
Quantization and dequantization perform no host activation readback.
Each launch uploads a small view-metadata buffer; the tensor values and scales remain on the GPU.

The complete pre-existing GPU operator suite also passes.
All 181 previous inference entries and their compiler constant banks remain byte-identical.
The portable bundle adds three spill-free entries and retains every Tensor Core instruction audit.
Its 184-entry inference module is compiled inside D3OS, with all generated module hashes matching the host audits.
This integer milestone used protocol 21; the current SDK and driver require protocol 23.

Evidence is under `target/nvidia/burn-quantization-validation/first/`.
The original integer bundle remains under `target/nvidia/burn-quantization-kernels/`. The current default includes the floating-point formats below.
These results establish native integer quantization functionality, not complete backend or Linux performance parity.
The common integer cases also pass the matched Linux conformance check below.
The floating-point formats and additional scale levels are covered below. Efficient quantized matrix execution remains required.
Qwen still uses its existing packed-GGUF extensions and FP32 activations until that execution work is integrated and validated.

### Matched integer conformance

The shared validator at `os/library/ai/src/validation/quantization.rs` runs through the public Burn API on Linux CUDA and D3OS's native GPU backend.
Both guests use Burn 0.22.1 and the same physical RTX 4090 Laptop GPU, with one VM at a time.
The archived manifests confirm that both runners use the same validator source by SHA-256.

All six integer formats pass the same independent nearest-even rounding and packed-byte references on both GPUs.
The common checks cover dequantization, transposition, outer-axis packing, serialization and reload, and dynamic calibration with endpoint and zero inputs.
Eight-bit reshapes also pass on both backends.
These are correctness checks, with no throughput measurement.

The first Linux trial exposed an upstream limitation rather than a D3OS failure.
Its CUDA backend cannot reshape sub-byte quantized values when the buffer needs to be rewritten.
The retained error is `Reshape with sub-byte values is not supported when the buffer must be recomputed`.
The shared validator therefore checks reshaping for eight-bit values.
D3OS separately retains reshape checks for all six formats, including packed two- and four-bit values.
Its additional rectangular-block, flip, empty-tensor, byte-tail, residency, and guard checks also pass.

Build the Linux runner from its own directory, after building the current D3OS production image:

```bash
cd tools/mnist-benchmark
cargo build --release --locked --bin quantization
cd ../..

taskset -c 8,10,12,14 python3 tools/nvidia/linux-reference.py \
  --burn-quantization tools/mnist-benchmark/target/release/quantization \
  --memory-mib 4096 --timeout 180 \
  --directory target/nvidia/burn-quantization-conformance/linux-new

taskset -c 8,10,12,14 python3 tools/nvidia/run-qemu.py \
  --firmware target/nvidia/firmware --kernels target/nvidia/kernels \
  --memory-mib 4096 --boot-check --inference gpu-self-test --timeout 180
```

The Linux launcher requires the CUDA success marker; a CPU runner cannot satisfy this check.
The accepted Linux evidence is under `target/nvidia/burn-quantization-conformance/linux-supported-integer/`.
The first failed Linux trial remains in `linux-integer/` in the same parent directory.
Native evidence is under `target/nvidia/burn-quantization-validation/shared-integer/`.
The comparison summary is `target/nvidia/burn-quantization-conformance/integer-comparison.json`.
This common-case conformance result does not establish full quantization or backend parity.

## Public FP8 and FP4 quantization

The native backend also supports `E4M3` and `E5M2` FP8 values and `E2M1` FP4 values.
These use the same public Burn methods. The original floating-point checks used F32 scales and output; additional scale formats are covered below.
FP8 supports packed U32 and native byte storage.
FP4 supports packed U32 and `PackedNative`, which stores two values per byte.
Both packed stores preserve their packing axis through views and serialization.
Native FP4 requires complete pairs on that axis; packed U32 FP4 requires groups of eight.

The cuTile kernels encode and decode the formats without requiring native FP8 or FP4 conversion instructions.
Encoding uses nearest-even rounding and finite saturation.
Stored negative zero retains its sign. Stored FP8 NaN and infinity encodings decode according to their format.
Burn's public quantization API clamps before encoding.
The Linux CUDA clamp maps either NaN sign to the positive saturation endpoint, and D3OS matches that behavior.
This differs from calling the low-level minifloat codec directly.

The shared validator enumerates every encoded value and every finite rounding midpoint, plus the adjacent F32 values on both sides.
An independent table search supplies the expected codes; it does not repeat the GPU's bit-rounding algorithm.
Checks also cover saturation, signed zeros, exact packed bytes, transposed views, reload, dynamic calibration, and compact byte tails.
FP8 rectangular blocks run on both backends.
These checks use the same physical GPU in separate Linux and D3OS guests.

The pinned Linux implementation has two relevant FP4 limitations:

- `PackedNative` FP4 is unsupported in the reference's data and quantization paths.
- A valid `[2, 32]` tensor with `[2, 8]` rectangular FP4 blocks returns `3.0` where the independent reference requires `4.0` at index 13.

The common Linux check retains packed-U32 FP4 with per-tensor scales.
D3OS separately runs the same exhaustive cases with native packed FP4 and retains the rectangular-block case.
The earlier Linux probe also rejects an FP4 block whose innermost extent is four because packed-U32 storage requires groups of eight.
These limitations do not justify removing the native backend's additional checks or copying the incorrect result.

This floating-point milestone used protocol 22; the current SDK and driver require protocol 23.
The packed-field codec is validated together with its bit width before dispatch.
That inference bundle contains 184 entries. The scale extension below adds three entries.
Only the quantize and dequantize machine-code sections changed; the other 182 entry sections and all constant banks remain identical.
The host audit retains the Tensor Core and spill checks.
Compilation remains inside D3OS, outside execution measurements.

The shared checks pass on Linux CUDA and the D3OS GPU with identical validator source hashes.
The D3OS run also passes the additional FP4 cases and the complete GPU operator suite.
All three modules compiled inside D3OS match their host audit hashes.
Native evidence is under `target/nvidia/burn-quantization-validation/minifloat-first/`.
Linux evidence is under `target/nvidia/burn-quantization-conformance/linux-minifloat-final/`.
The comparison summary is `target/nvidia/burn-quantization-conformance/minifloat-comparison.json`.
Failed Linux probes remain alongside that evidence, including the block diagnostic and unsupported native-FP4 trial.
The verified bundle is under `target/nvidia/burn-minifloat-final-kernels/` and now populates `target/nvidia/kernels/`.
These are correctness results. They do not establish quantized compute acceleration or full backend or performance parity.
The remaining dtype, operation, execution, and optimized quantized-compute gaps keep the primary goal open.

## Scale storage and two-level quantization

Quantization scales now support `F32`, `F16`, `BF16`, `UE8M0`, and `UE4M3`.
Narrow scales round upward to the next representable value, with finite saturation.
`UE8M0` has no zero, so a zero scale rounds to its minimum value, `2^-127`.
The encode kernel applies this policy before any quantized values use the scales.
Quantizing with a zero scale follows the Linux clamp policy, including integer formats: a NaN quotient takes the positive endpoint.

A quantized tensor retains one raw scale code per device word.
This representation preserves imported NaN payloads and makes scale views use the existing GPU tensor layout operations.
Record storage remains compact: four bytes for F32, two for F16 or BF16, and one for either unsigned format.
GPU pack and unpack kernels convert between device codes and record bytes without numeric conversion.
Partial words are padded only in device allocations; the serialized record contains exactly its defined bytes.

Two-level schemes retain a separate, shared F32 tensor scale.
Kernels reconstruct the effective scale as the decoded block scale times that tensor scale.
The product is formed in F32 before value conversion.
Views, simple reshapes, record export, and reload preserve both levels.
The tensor scale must use F32. This matches the pinned CUDA binding contract.
Burn's dynamic normalization permits F16 and UE4M3 block scales; wider block formats cannot use its maximum-based normalization recipe.
Missing or extra scale levels and malformed record lengths return execution errors before GPU access.

The scale paths add three cuTile entries: encode, pack, and unpack.
The portable inference bundle now has 187 entries.
Native compute protocol 23 validates scale format and global-scale flags and checks the separate tensor-scale binding.
Quantization and dequantization keep values and scales on the GPU.
Only explicit record export reads the packed storage back to the CPU.

The shared checks use independent ordered scale tables to find the first representable value at or above each input.
They cover all five dtypes, rounding boundaries, saturation, subnormal F16 scales, zero scales, explicit two-level parameters, and dynamic two-level calibration.
Record checks preserve raw scale bits, including NaN payloads, infinities, subnormals, and compact tails.
Two-level zero-input calibration and exported record reload also run through the public Burn API.

The pinned Linux CUDA mapper uses the input scale tensor's rank to index block coordinates.
The public API supplies flat scales, which produces incorrect results for some multidimensional inputs.
The retained `[7, 16]` Q4S example returns `0.0` instead of `1e-5` at index 16, even with F32 scales.
This also explains the earlier rectangular FP4 mismatch; it is not evidence that the FP4 codec itself rounds incorrectly.
Common scale checks therefore quantize one-dimensional inputs before reshaping and transposing the results.
D3OS separately checks multidimensional Q4S and FP4 inputs, all five scale dtypes, both scale levels, and reversed views.
Raw native launches also check prefix and suffix guards and tail padding for all three scale kernels.

Both Linux CUDA and D3OS pass the shared checks with identical validator source hashes.
D3OS also passes the additional multidimensional, reversed-view, and guarded-launch checks and the complete GPU operator suite.
The three modules compiled inside D3OS match their host audit hashes.
The 181 entries outside quantization and their constant banks remain byte-identical to the previous bundle.
Native evidence is under `target/nvidia/burn-quantization-validation/scales-final/`.
Linux evidence is under `target/nvidia/burn-quantization-conformance/linux-scales-final/`.
The comparison summary is `target/nvidia/burn-quantization-conformance/scales-comparison.json`.
The failed multidimensional Linux probe remains in `linux-scales-first/` beside the accepted Linux run.
The checked `target/nvidia/burn-scale-kernels/` bundle now populates the default kernel directory.
These are correctness checks. Full backend conformance, optimized quantized execution, and matched workload performance remain required.

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

The full pretrained reference also passes on the Linux host with Burn Flex CPU.
The initial D3OS software CPU emulation run exceeds its 900-second limit.
After the import and SIMD selection fixes, it passes all references in 704920 ms under TCG.
This is a D3OS CPU correctness result, not a GPU performance measurement.
