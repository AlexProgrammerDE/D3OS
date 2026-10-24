# Native NVIDIA performance

The native driver runs AlexNet, Qwen3.5-0.8B, and DOOM correctly on the RTX 4090 Laptop GPU.
The Linux performance target remains open. Correct output does not imply comparable throughput.

The goal now includes [Burn-based AI APIs and native training](burn-migration.md).
AlexNet and Qwen now use Burn with a native D3OS/cuTile backend.
DOOM keeps its native renderer. Its correctness and Linux performance requirements remain unchanged.

The optimization goal also includes Qwen3.5-9B at Q8_0 and Qwen3.6-27B at Q3_K_M.
Both targets must fit entirely in the GPU's 16 GiB of VRAM, including context and working storage.
The work includes pinned Linux measurements, native Rust/cuTile quantized kernels, numerical references, and matched performance comparisons.
The native loader also needs separate output weights and grouped key/value heads for recurrent attention.
Large model assets also need a storage path that avoids the current FAT32 boot file limit and duplicate archive copies.
These models are pending native validation. The existing 0.8B, AlexNet, and DOOM checks remain required.

## Measurement method

These measurements use the same physical GPU through VFIO, with four KVM virtual CPUs.
Only one guest owns the GPU at a time.
D3OS uses its Rust driver, signed 570.144 firmware, and native Rust kernels.
The Linux reference guest uses driver 615.78.08 and CUDA libraries installed on the host.
Those libraries remain inside the reference guest. D3OS does not link or load them.

The resident-model benchmarks separate file access, model upload, session allocation, prompt processing, and generation.
Reference comparisons run after the timed work.
AlexNet reports 100 forward passes after one warmup pass. Earlier measurements use 20 passes.
Qwen reports three independent sessions with a 22-token prompt, 64 decode steps, and capacity for 256 tokens.
The optional `long` benchmark generates 4,096 tokens per session with capacity for 4,352 tokens.
It writes complete token traces to the serial log without filling the interactive terminal output queue.
The decode loop continues after EOS to measure a fixed amount of work.
Its checksum describes benchmark work, not a conversational answer.

D3OS timers have millisecond resolution. The kernel benchmark amortizes this resolution across 100 launches.
Its times include CPU submission and synchronization.
The Linux kernel benchmark uses CUDA events across 30 queued launches, which excludes most CPU submission overhead.
D3OS also runs a separate instrumented pass with GPU timestamps before scheduling and at QMD completion.
The boot check calibrates a cuTile kernel to run for at least 10 ms and compares GPU timestamps with host elapsed time.
`device_mean_us` includes device scheduling and completion work, while `instrumented_mean_us` also includes CPU submission and waiting.
The first `mean_us` pass runs without timestamp instrumentation.
These measurements separate end-to-end overhead from device throughput. They are not interchangeable timers.

## Native Burn integer coverage

The I32 backend adds exact wrapping reductions and GPU-generated index ranges.
Physical D3OS checks use independent integer references, signed views, broadcast inputs, partial tiles, and neighboring guards.
Public Burn cross-entropy also passes an analytic loss, gradient, and accuracy reference on the NVIDIA device.
The matching native compute protocol is version 17, with 59 kernel families and 174 inference entries.
All 166 earlier GPU instruction sections remain byte-identical after the eight new entries are added.
Validation logs are under `target/nvidia/burn/integers/`.
This work adds no D3OS or Linux throughput measurement. Workload migration remains the priority before optimization.

## Native Burn training validation

D3OS now trains and exports the convolutional MNIST model through Burn's public native GPU API.
The physical RTX 4090 Laptop GPU run uses all 60,000 training images and the separate 10,000-image test set.
Seven FP32 epochs reach 99.26 percent test accuracy: five at learning rate 0.001, then two at 0.0001 with Adam state preserved.
The model also passes checkpoint resume across VM restarts and inference export checks.
See [Train MNIST inside D3OS](mnist-training.howto.md) for the procedure and retained evidence.
This validation contains no Linux training sample or matched performance comparison.

## AlexNet Burn migration

These are fresh measurements on the same passed-through RTX 4090 Laptop GPU, with four guest CPUs and host affinity `8,10,12,14`.
All three paths use the same FP32 checkpoint, 224-by-224 fixture, and eight complete layer references.
The original absolute and relative tolerances remain `2e-4`. The predicted class remains 258.

| OS and execution path | File read | Model preparation | Warm forward mean |
| --- | ---: | ---: | ---: |
| D3OS, legacy native model | 444 ms | 263 ms | 0.930 ms |
| D3OS, Burn native model | 242 ms | 319 ms | 2.420 ms |
| Linux, cuDNN/cuBLAS | 856.773 ms | 722.932 ms | 0.731–0.735 ms |

Each warm sample measures 100 passes after one warmup. Linux supplies three samples; its average is 0.732888 ms.
Both timers include CPU submission and synchronized completion. Input upload and reference readback remain outside the forward interval.
The file and preparation columns are single cold observations with different loading implementations, not a separate matched I/O benchmark.
Linux uses cuDNN FMA and default cuBLAS FP32 math. D3OS uses the native compensated TF32 Tensor Core kernels with FP32 storage.

Burn currently regresses D3OS warm forward time by 2.60 times.
The model remains on Burn. Kernel dispatch and allocation improvements follow the remaining workload migrations.
The forward residency check confirms zero activation uploads and downloads.
At this AlexNet milestone, Qwen retained its earlier execution path. Its later migration is described below.

A separate instrumented GPU pass records 1.437 ms across the five generic convolution entries and 0.570 ms across the three classifier matrix entries.
Those serialized device observations identify costs; they are not the uninstrumented forward measurement.
No device kernel binary changes in this migration.
The retained logs, Linux provenance, and comparison report are under `target/nvidia/burn/alexnet/`.
The Burn CPU reference, Qwen references, native Burn training/checkpoints, and all 600 DOOM views also pass.

## Qwen packed-weight Burn extensions

Fresh correctness checks pass on the physical NVIDIA GPU inside D3OS and on the explicit host Burn CPU device.
They cover F32, BF16, and five GGUF encodings, including independent packed-block reference values.
Projection, split projection, embeddings, fused gates, and zero-centered RMS normalization use public Burn tensors.
Burn module records preserve packed words. Residual fusion preserves cloned inputs and row views.
Native forward checks report zero activation uploads and downloads.
The device kernel bundle and compute protocol remain unchanged.

At this packed-weight milestone, the full model and state migration remained pending.
These checks add no D3OS throughput result and no Linux measurement.
Logs are under `target/nvidia/burn/qwen-packed/`.

## Qwen state extensions and migration baseline

Fresh physical D3OS checks pass public Burn attention and recurrent operations against independent binary64 fixtures.
Eight attention cases and sixteen recurrent cases cover single-token and batched execution, cache snapshots, and packed projection fusion.
Native state updates perform zero activation uploads or downloads.
Split attention, grouped heads, wider fallback dimensions, and error propagation also pass.
The kernel bundle and compute protocol remain unchanged.

These baseline samples use the legacy execution API before the Burn model migration.
The following fresh D3OS baseline measures that path before the model migration, with the passed-through RTX 4090 Laptop GPU.
Each session has capacity for 256 tokens, a 22-token prompt, and 64 decode steps.
The host affinity remains `8,10,12,14`, with four guest CPUs.

| D3OS legacy session | Prompt time | Prompt tokens/s | Decode time | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| First session | 13 ms | 1692.308 | 248 ms | 258.065 |
| Warm session 1 | 9 ms | 2444.444 | 254 ms | 251.969 |
| Warm session 2 | 9 ms | 2444.444 | 254 ms | 251.969 |

The single file-read observation is 1543 ms; model preparation takes 615 ms.
State allocation takes 21 ms initially and 6 ms for each later session.
All three decode traces share checksum `11137929795425568356`.
The original 29-position, 24-layer references and batched cache comparison pass after the benchmark.
No fresh Linux sample or complete Burn Qwen model result accompanies this baseline.

Fresh AlexNet references and all 600 DOOM views also pass.
DOOM reports 69 to 70 FPS after startup under its gameplay cap; this is not an uncapped rendering benchmark.
Evidence is under `target/nvidia/burn/qwen-state/`.

## Complete Qwen Burn model migration

Qwen3.5-0.8B now runs through Burn modules, packed parameters, public tensors, owned session state, and generation.
The tokenizer and streaming GGUF reader live in the AI library. The previous inference crate is removed.
Physical D3OS checks pass every original layer and logit reference at 29 positions, with all 105 tokenizer cases unchanged.
The original absolute and relative reference tolerances remain `2e-3` and `5e-4`.
Sequential and batched execution also pass the final cache comparisons.
Streaming generation answers the default arithmetic prompt with `4` and stops at EOS.

These are fresh D3OS before-and-after measurements on the same passed-through GPU, with four guest CPUs and affinity `8,10,12,14`.
Each session uses a 22-token prompt, 64 decode steps, and capacity for 256 tokens.
Timers include CPU submission and synchronized completion.

| D3OS execution path | Model preparation | First decode | Warm decode 1 | Warm decode 2 |
| --- | ---: | ---: | ---: | ---: |
| Legacy native model | 615 ms | 258.065 tokens/s | 251.969 tokens/s | 251.969 tokens/s |
| Burn native model | 21880 ms | 221.453 tokens/s | 242.424 tokens/s | 235.294 tokens/s |

The combined warm decode rate falls from 251.969 to 238.806 tokens/s, a 5.22 percent regression.
Warm prompt processing measures 9 ms before migration and 9 to 10 ms afterward.
The single file-read observations are 1543 ms before migration and 2666 ms afterward; they do not establish a matched I/O comparison.
State creation measures 21 ms initially and 6 ms warm before migration, versus 5 ms initially and below one timer tick warm afterward.
Model preparation regresses by 35.58 times. Service counters place almost all preparation time in guest application code.
These regressions remain open. Burn remains the model API.

All six decode traces share checksum `11137929795425568356`.
The resident-model check permits only explicit token-ID uploads and scalar selection reads, with no activation tensor transfer.
For the measured warm session, prefill uploads one token-ID batch and downloads no tensors.
Decode uploads 64 token IDs and reads 128 selection scalars: one finite-value flag and one selected index per step.
The kernel bundle and compute protocol remain unchanged.
No fresh Linux sample accompanies this migration comparison.
Fresh AlexNet references and all 600 DOOM views also pass; DOOM stays outside Burn and reaches its 70 FPS gameplay cap.
Evidence and the before-and-after report are under `target/nvidia/burn/qwen-model/`.

## Quantized Qwen models on Burn

Qwen3.5-9B Q8_0 and Qwen3.6-27B Q3_K_M now run through the native Burn stack.
Each passes 165 independent FP32 layer and logit references, with matching sequential and batched final caches.
The reference exporter runs PyTorch on the Linux host, one complete layer at a time.
It uses the original GGUF weights, the pinned Hugging Face configuration, and the Linux CUDA benchmark's token IDs.
This avoids loading a complete 27B FP32 model into host RAM.

The original absolute and relative tolerances remain `2e-3` and `5e-4`.
A separate check compares all 248320 logits at 129 positions after the prompt.
Both models pass all 32033280 values.
Maximum absolute errors are `0.00003064` for 9B and `0.00003553` for 27B.
The complete token, merge, and token-type tables match the 0.8B tokenizer.

GGUF exports grouped value heads in tiled order.
The reader restores input projections and state parameters to the model's grouped order.
It preserves packed output projection weights and reorders their input activation on the GPU.
This handles quantization blocks that span more than one value head without requantization.

These fresh speed samples use the passed-through RTX 4090 Laptop GPU, four guest CPUs, and affinity `8,10,12,14`.
Both guests use 15360 MiB of RAM, a 37-token prompt, 128 decode steps, and a 2048-token FP32 cache.
D3OS replays the Linux CUDA token IDs and continues all benchmark steps through EOS.
Transfer counters permit two prompt-ID uploads, 128 decode-ID uploads, and 256 selection-scalar reads per run.
They show no activation tensor upload or download during timing.
Numerical reference reads occur afterward.

| Model and weights | D3OS native Burn warm decode | Linux guest llama.cpp CUDA warm decode |
| --- | ---: | ---: |
| Qwen3.5-9B Q8_0 | 31.9205 tokens/s | 58.3715 tokens/s |
| Qwen3.6-27B Q3_K_M | 5.2260 tokens/s | 34.4025 tokens/s |

These are diagnostic speed comparisons. They do not establish Linux parity.
Native projections retain FP32 activation arithmetic.
The normal quantized Linux CUDA path produces different raw logits and one or two different greedy choices on these inputs.
D3OS instead agrees with the independent FP32 reference within the errors above.
The Linux CUDA speed results and Linux CPU accuracy results are separate evidence.
Some runs overlap CPU verification on separate cores; quiet matched timing and kernel profiling remain required.

Compute-buffer weight storage is 9516859392 bytes for 9B and 13575868416 bytes for 27B.
These counters exclude firmware and channel allocations.
Both models also fit their measured context and working storage in the physical GPU's 16 GiB VRAM.

Packed weight preparation now uses bulk copies and direct byte uploads.
The initial 0.8B Burn preparation regression falls from 21880 ms to 1940 ms in the first corrected D3OS run.
The earlier native model took 615 ms, so preparation still needs improvement.
Flex CPU model import widens BF16 constants to FP32 once, rather than for each projection.
Embedding lookup decodes only the requested token rows.
A local GEMM portability patch enables target-guaranteed SIMD without `std`.

D3OS CPU correctness checks pass all 29 positions, 24 layers, and 105 tokenizer cases.
The full check takes 704920 ms under TCG and 32072 ms under KVM in the retained runs.
These are CPU validation times, not GPU inference measurements.
The initial CPU emulation timeout is resolved without removing references or increasing its 900-second limit.

Commands are in [the Qwen guide](qwen.howto.md#run-the-quantized-9b-and-27b-models).
Reference manifests, source checksums, before-and-after failures, complete token traces, and comparisons are under `target/nvidia/burn/large-qwen/`.
CPU and loading evidence is under `target/nvidia/burn/qwen-load/`.

The final D3OS smoke suite also passes the GPU self-test, all eight AlexNet references, and the original 0.8B Qwen references.
Its latest 0.8B preparation sample is 1439 ms, with warm decode samples of 219.178 and 226.148 tokens/s.
These samples show remaining timing variation and a generation gap; they do not establish an improvement.
D3OS DOOM measures 1617.5 uncapped timedemo FPS and retains its 70 FPS gameplay cap.
All 600 verification frames, input controls, cache purges, cleanup, and post-exit GPU inference checks pass.
The GPU kernel bundle remains unchanged.

## Burn convolution algorithm selection

The next D3OS GPU change selects explicit image windows and Tensor Core matrix products for large dense inference convolutions.
It requires one image, ungrouped unit dilation, symmetric padding, contiguous views, and enough matrix work for reuse.
The window workspace is limited to 8 MiB.
Other geometries retain the direct convolution algorithm. Backward operations retain their independently validated implementation.

Fresh 100-pass D3OS GPU averages improve from 2.320 ms to 1.300 ms for AlexNet, a 43.97 percent reduction.
Fresh Linux GPU cuDNN/cuBLAS warm averages are 0.742077 and 0.732881 ms.
Both guests use four CPUs, affinity `8,10,12,14`, 4096 MiB of RAM, and the original FP32 weights and input.
The original eight layer references retain their `2e-4` absolute and relative tolerances.
The larger speed gap remains open.

The five D3OS convolution matrix products total about 0.327 ms of instrumented device time.
The earlier direct convolution kernels total about 1.427 ms.
Explicit window extraction and channel bias kernels add separate work, so these totals are not complete forward times.
Classifier matrix products and generic activation dispatch remain visible costs.

Physical GPU checks validate explicit and direct paths against independent FP64 forward and gradient references.
Cases cover optional bias, nonzero contiguous offsets, shared storage, and signed views.
Existing grouped, dilated, asymmetric, empty, finite-difference, residency, and guard checks pass.
The selector also has metadata tests for workspace limits and unsupported fast-path geometries.
The kernel bundle remains unchanged.
Evidence is under `target/nvidia/burn/alexnet-optimization/`.

## Earlier Burn primitive measurements

The following measurements describe earlier implementation stages. Current model migration results are above.

### Burn primitive regression

The measurements in this subsection are D3OS results from QEMU with the RTX 4090 Laptop GPU passed through via VFIO.
This regression round does not include a fresh Linux measurement.
Performance reports must label each result as D3OS or Linux and identify whether a comparison uses retained or fresh samples.

The first native Burn primitives now support shared GPU storage, checked views, and eleven specialized cuTile elementwise entries.
GPU checks cover signed strides, broadcasting, scalars, comparisons, mask filling, exact bit copies, partial tiles, and neighboring guards.
Chained arithmetic uses resident buffers without host activation transfers or separate output clears.
Invalid view metadata fails before dispatch. All 77 pre-existing device instruction sections remain byte-identical.

The next primitive checks add Tensor Core matrix products and seven strided reduction entries.
Matrix checks cover broadcast batches, noncontiguous inputs, subviews, empty outputs, and zero inner dimensions.
Reduction checks cover every tensor axis, repeated tiles, broadcasting, nonfinite values, first-index ties, empty axes, and neighboring guards.
Both operations keep tensors resident. All 88 earlier instruction sections remain byte-identical after the reductions are added.
The latest primitive checks add 21 unary math operations, broadcast power, and exact tensor fill.
Independent libm references cover 2,816 FP32 edge values, including rounding boundaries, zeros, infinities, and NaNs.
Fill performs no host upload and requires no separate clear. All 95 earlier GPU instruction sections remain byte-identical.
The current SDK and driver use native compute protocol version 18.

The accompanying regression passes AlexNet's eight layer references, Qwen's layer and cache references, and all 600 DOOM views.
D3OS AlexNet averages 0.960 ms across 100 passes.
Both D3OS Qwen warm 64-token samples reach 264.463 tokens/s, with the existing checksum unchanged.
D3OS DOOM reaches 1,596.76 uncapped FPS. Regular D3OS gameplay remains capped around 70 FPS.
These samples remain within the recent run-to-run variation and do not establish a throughput improvement.
The latest primitive checks and regression logs are under `target/nvidia/burn/math/`.
These are regression measurements of the existing workloads. Their migration to Burn remains pending.
The first Burn training check runs on the explicit CPU device inside D3OS and validates gradients and Adam checkpoint resume.
Full native Burn training and matched performance comparisons remain open.

The subsequent typed-word step adds I32 arithmetic, boolean logic, masks, and conversions.
All 118 earlier GPU instruction sections remain byte-identical after its 29 entries are added.
The D3OS self-test validates wrapping, conversion boundaries, shared ownership, guards, and resident execution.
This step uses functional workload checks. It adds no new D3OS or Linux throughput measurement.
The performance samples in this subsection remain from the preceding math regression, under `target/nvidia/burn/math/`.
The typed-word validation logs are under `target/nvidia/burn/words/`.

The subsequent indexing step adds resident gather and vector select for FP32, I32, and boolean tensors.
Each nonempty operation reads one four-byte bounds flag and synchronizes; tensor values do not cross to the host.
Physical D3OS GPU checks cover signed strides, offset views, duplicate indices, empty axes, partial tiles, and neighboring guards.
Invalid indices return an error without changing inputs or preventing later GPU operations, including inside command scopes.
All 147 previous GPU instruction sections remain byte-identical.
This step adds functional validation, with no new D3OS or Linux throughput measurement.
Its logs are under `target/nvidia/burn/indexing/`.

The next training primitive adds scatter accumulation and vector select accumulation.
FP32 and I32 values use addition. Boolean values use OR.
Inputs remain immutable through a private GPU copy of the destination and GPU materialization of noncontiguous values.
A four-byte bounds read synchronizes each nonempty operation. Colliding FP32 additions have an unspecified order.
FP32 compare-and-swap preserves subnormal values that Ada's global atomic addition can flush to zero.
NVIDIA describes this behavior in its [PTX atomic instruction reference](https://docs.nvidia.com/cuda/archive/12.1.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom).
Physical D3OS GPU checks pass strided scatter, duplicate indices, offset views, guards, shared inputs, and invalid-index recovery.
They also preserve subnormal accumulation and match a weighted gather gradient against independent FP64 finite differences.
All 148 previous GPU instruction sections remain byte-identical.
This step does not establish a D3OS or Linux throughput improvement.
Its evidence is under `target/nvidia/burn/scatter/`.

The convolution step adds native forward execution and gradients for the Burn backend.
Physical D3OS GPU checks pass independent FP64 references, finite differences, grouped and dilated cases, signed views, empty batches, and guards.
The operations keep activations resident. All 151 previous GPU instruction sections remain byte-identical.
This step leaves the existing AlexNet path unchanged and adds no performance improvement claim.
Its functional validation evidence is under `target/nvidia/burn/convolution/`.
Further performance optimization waits until the native Burn backend and AI workload migration are complete.

The pooling step adds max, average, and adaptive average forward and backward operations for the native Burn backend.
The operations retain tensor values on the GPU. Max backward reads one four-byte bounds flag and synchronizes.
Physical D3OS GPU checks pass FP64 references, finite differences, signed views, guarded writes, and recoverable index errors.
The checks include exact NaN payloads, subnormal arithmetic, empty dimensions, and adaptive ratios with products beyond 32 bits.
All 154 previous GPU instruction sections remain byte-identical. AlexNet, Qwen, and DOOM functional regressions also pass.
This step targets Burn operation coverage and correctness, with no new throughput comparison or optimization claim.
Its validation evidence is under `target/nvidia/burn/pooling/`.

The tensor update step adds exact masked selection, slice assignment, concatenation, and padding for the native Burn backend.
FP32, I32, and boolean values remain resident, including through signed views and broadcasting.
Updates preserve shared source storage through private destination copies. Partial writers start from initialized allocations.
Physical D3OS GPU checks pass exact words, signed views, overlapping-source snapshots, empty tensors, guards, and recoverable rejected launches.
All 159 previous GPU instruction sections remain byte-identical. AlexNet, Qwen, and DOOM functional regressions also pass.
This step adds operation coverage, with no new D3OS or Linux performance measurement.
Its validation evidence is under `target/nvidia/burn/updates/`.

The random-generation step adds seeded Philox4x32-10 streams and portable checkpoints for Burn initialization and training.
Five cuTile entries provide raw words, FP32 uniform and normal values, Bernoulli samples, and unbiased bounded integers.
Values remain on the GPU. Stream reservations are serialized and checked for counter exhaustion.
Physical D3OS GPU checks pass anchored Philox words, exact checkpoint replay, scalar integer references, FP64 normal references, and guarded writes.
All 161 previous GPU instruction sections remain byte-identical. AlexNet, Qwen, and DOOM functional regressions also pass.
This step adds training support, with no new D3OS or Linux performance measurement.
Its validation evidence is under `target/nvidia/burn/random/`.

The public backend step registers native GPU tensors and autodiff with Burn's device dispatcher.
Physical D3OS checks pass public matrix products, convolution gradients, analytic SGD, and Adam checkpoint resume.
The convolutional MNIST model also passes a synthetic SGD step, with gradients for all eight parameter tensors and preserved shared weights.
That step uploads no activations and reads only two four-byte bounds flags from max-pool gradients.
Batched gradient products retain initialized allocations for their partial writes.
Native failed tensors propagate execution errors without placeholder storage or CPU computation.
Operator coverage, complete MNIST training, and AI workload migration remain incomplete.
This step adds no D3OS or Linux performance comparison. Its evidence is under `target/nvidia/burn/dispatch/`.

### Grouped recurrent attention

The recurrent operator now accepts separate query/key and value head counts.
Its GPU checks cover the 16/32 heads of Qwen3.5-9B and the 16/48 heads of Qwen3.6-27B.
They also cover three smaller shapes with partial tiles and convolution histories.
Each shape compares seven sequential positions and the same positions in batches against the CPU reference.
The host test independently compares grouped channels against explicitly repeated query/key channels, including both final caches.
Invalid head groups leave both caches unchanged.

Grouped execution updates convolution history before the recurrent kernel runs.
Value heads share read-only query/key channels, while each head owns its recurrent matrix.
This avoids concurrent writes to shared convolution history.
All 76 existing device instruction sections remain byte-identical.
The Qwen-0.8B regression preserves its 64-token trace and passes all layer, batch, and cache references.
Full 9B and 27B execution still requires model configuration, output-weight, and asset-loading integration.

### Native quantized operators

The native Rust/cuTile backend now accepts all five packed weight formats used by the larger Qwen files.
Independent samples cover 5,120 values from the checksum-pinned checkpoints.
CPU and GPU decoding match the reference bits exactly.
The GPU checks also cover embedding gathers, single-token projections, batched Tensor Core projections, unaligned uploads, partial tiles, and neighboring guards.
Invalid raw embedding indices produce zero rows without reading outside the table.

All 56 existing device instruction sections remain byte-identical. The compiler audits actual Tensor Core instructions in each new batched projection.
The final regression passes AlexNet's eight layer references, Qwen-0.8B's layer and cache references, and all 600 DOOM views.
The accompanying DOOM run reaches 1,566.64 uncapped FPS. Regular gameplay remains capped around 70 FPS.
These checks validate the operators. Full native 9B and 27B model execution remains pending.

The small binary fixture and its provenance are under `os/library/compute/tests/fixtures`.
With the pinned models and reference tools available, regenerate it with:

```bash
target/qwen-venv/bin/python tools/inference/prepare-quantized-fixture.py
```

### Larger Qwen targets

The Linux reference now runs [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) and [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B).
The pinned quantized files come from Unsloth's [9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) and [27B](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) repositories.
The preparation script verifies each file's size and SHA-256 before staging it.
The reference also records the executable, headers, libraries, GPU memory, and complete prompt and output token sequences.

Each model uses FP32 attention and recurrent caches, a 2,048-token context, and a batch size of 32.
Every weight, including embeddings, resides in GPU memory. The reference rejects CPU weight buffers and incomplete layer offload.
The prompt asks for the largest country by land area and a detailed essay about its geography, climate, population, and history.
Each of three sessions generates 128 tokens. Fixed-length timing continues after EOS if the model emits it.

Qwen3.5-9B Q8_0 reaches 57.843, 58.117, and 58.073 tokens/s.
Its two warm samples average 58.095 tokens/s, with prompt times of 43.644 and 43.670 ms for 37 tokens.
Its measured GPU usage after decoding is 10,015,997,952 bytes, or 9.33 GiB.

Qwen3.6-27B Q3_K_M reaches 33.631, 33.990, and 33.979 tokens/s.
Its two warm samples average 33.985 tokens/s, with prompt times of 95.563 and 95.952 ms for the same 37 tokens.
Its measured GPU usage after decoding is 14,338,228,224 bytes, or 13.35 GiB.
This leaves about 2.30 GiB free in the memory pool reported by the NVIDIA driver.
Longer contexts require separate measurements.

Both outputs identify Russia. Each model preserves its exact output token sequence across all three sessions.
These results establish GPU fit and a Linux throughput reference. Native large-model validation and broader quality evaluation remain open.
The first Linux model loads take about 63 seconds and 87 seconds through the shared 9p filesystem.
Those load times include file access and GPU upload, so they do not match the native resident-archive upload timer.

Run the complete Linux suite with the existing pinned reference tools:

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/test-large-qwen.py 9b 27b
```

The suite downloads about 23.1 GB of model files and launches one GPU guest at a time.
It uses 15 GiB of guest RAM by default, within the session's existing 16 GiB memlock limit.
The logs, provenance manifests, and JSON results are under `target/nvidia/large-qwen`.
Use a new `--directory` to preserve an earlier suite's evidence.
The native API still requires quantized weight storage and kernels, separate output weights, and grouped recurrent heads before either model can run.
The 9B file contains Q8_0 matrices and FP32 tensors.
The 27B file mixes Q3_K, Q4_K, Q5_K, Q6_K, and FP32 tensors. Native support must preserve every tensor's actual format.

### Matched Linux AlexNet reference

The Linux reference uses the same FP32 AlexNet archive, image, and eight layer fixtures as D3OS.
It runs cuDNN convolutions and pooling, with cuBLAS dense layers.
Convolutions use FMA math; dense layers use the default FP32 math mode.
All eight outputs must meet the original absolute and relative tolerances of `2e-4`.
The full cuDNN 9.21.0.82 package is checksum-pinned and stays inside the reference guest.
The preparation manifest records the model, fixture, executable, headers, source, and library hashes.

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/linux-reference.py \
  --alexnet-benchmark --memory-mib 12288
```

The first three Linux samples average 0.7368, 0.7320, and 0.7308 ms over 20 forward passes each.
They pass every layer reference and produce class 258.
The preceding native control averages 1.25 ms over 20 passes.
The command-scope pilot reaches 1.05 ms, while the first full-writer sample reaches 1.15 ms.
The following alternating comparison uses longer runs to check those initial samples.

The following A-B-C-C-B-A comparison uses 100 passes per guest and one unchanged cuTile binary.

| Native variant | Samples, ms | Mean, ms | Launches per forward | Queue submissions |
| --- | ---: | ---: | ---: | ---: |
| Control | 1.33, 1.30 | 1.315 | 44 | 44 |
| Command scope | 1.11, 1.16 | 1.135 | 44 | 2 |
| Scope and full writers | 1.04, 1.17 | 1.105 | 31 | 2 |

The combined change reduces the mean by 16.0%, with all layer references passing in every guest.
The full-writer change alone has overlapping wall-time samples; its separate percentage benefit is not established.
The last-forward queue counters confirm that it removes 13 clear kernels while preserving the two-submission command scope.
The benchmark also provides matrix-device timings in a separate instrumented forward pass, outside the normal timing loop.

A separate A-B-B-A comparison keeps the scoped full-writer path fixed and changes only the FP32 matrix reduction tile.
Eight-wide samples reach 1.05 and 1.01 ms; 16-wide samples reach 0.97 and 0.91 ms.
The means are 1.03 and 0.94 ms, an 8.7% improvement.
The five convolution matrix kernels fall from 442.75 to 326.64 microseconds in the separate instrumented pass.
Every sample passes the original eight layer tolerances.
The 16-wide kernel uses 80 registers per thread and has no local spill instructions.
A 64-wide experiment reaches 1.14 ms, while a rank-preserving load variant reaches 1.09 ms. Both are removed.
All retained older kernel instruction sections remain byte-identical.
The final Linux comparison uses 100 passes per sample and reaches 0.7330, 0.7317, and 0.7311 ms.
It still passes all eight layer references. Native AlexNet's 0.94 ms mean remains about 28% slower.
Final regression checks pass GPU operators, Qwen model references, and all 600 DOOM reference views.
The accompanying uncapped DOOM sample reaches 1,607.06 FPS, with regular gameplay still capped around 70 FPS.

The sustained benchmark uses these commands, with one GPU guest at a time:

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/run-qemu.py \
  --firmware target/nvidia/firmware --kernels target/nvidia/kernels \
  --memory-mib 12288 --boot-check --inference gpu-qwen-bench \
  --qwen-long-bench --timeout 240
target/qwen-venv/bin/python tools/nvidia/prepare-linux-qwen.py \
  --gguf target/nvidia/linux-model/Qwen3.5-0.8B-BF16.gguf \
  --d3os-log target/nvidia/vm/serial.log --decode-tokens 4096 \
  --output target/nvidia/matched-qwen-long
taskset -c 8,10,12,14 python3 tools/nvidia/linux-reference.py \
  --llama-directory target/nvidia/linux-llama/llama-b11827 \
  --model target/nvidia/linux-model/Qwen3.5-0.8B-BF16.gguf \
  --matched-qwen-input target/nvidia/matched-qwen-long \
  --matched-qwen-report-selection --memory-mib 12288
```

The first four-tap baseline reaches 181.63, 174.06, and 169.30 tokens/s across its three sustained sessions.
All three produce the same 4,096-token trace and pass the numerical references after timing.
The first split-attention sessions reach 239.90, 235.62, and 232.05 tokens/s with the same complete trace.
The two warm sessions improve by about 36% over this baseline.
The corresponding Linux warm sessions range from 217.18 to 227.25 tokens/s.
These are initial samples. Repeated comparisons remain necessary because throughput changes with host and GPU conditions.

A subsequent A-B-B-A comparison retains the same cuTile binary and changes only the native attention dispatcher.
The control warm sessions reach 178.54, 173.70, 171.41, and 169.37 tokens/s.
The split sessions reach 235.19, 232.53, 233.94, and 232.27 tokens/s.
Their means are 173.25 and 233.48 tokens/s, an improvement of 34.8%.
All twelve sessions preserve the complete token trace and pass the layer, batch, and cache references.
The final short-decode regression reaches 252.96 and 263.37 tokens/s with the original 64-token checksum.
AlexNet averages 1.25 ms over 20 forward passes and passes all eight layer references.
DOOM passes 600 pixel-reference views and reports 1,596.76 FPS in its uncapped `demo1` benchmark.
Regular gameplay remains capped around 70 FPS. These regression samples do not establish a rendering improvement from attention changes.

Linux receives every native token as input, so both stacks perform the same decoding workload.
It selects a different token at position 138 in each run, before the split-attention threshold.
Its two leading scores differ by 0.000061; D3OS reverses their order with a margin of 0.002686.
The two scores differ across stacks by at most 0.004311.
The default audit rejects these traces because it requires identical greedy choices.
The explicit reporting option preserves input, cache, checksum, trace, and device checks while disclosing selection differences.
This option permits performance measurements; it does not establish identical generated answers.

### Borrowed BF16 uploads

The load profiler attributes approximately 86% of the earlier model-load cycles to application code rather than driver service calls.
The archive loader scanned each BF16 matrix, then repacked every value into a temporary word vector before uploading it.
It now sends borrowed little-endian bytes directly to the synchronous upload path after one finite-value check.
Unaligned CPU sources use byte copies. Odd final values use a padded private word instead of reading past the source.

The earlier short benchmark loads the resident model in 1,932 ms; the sustained control runs measure 1,863 to 1,894 ms.
The direct-upload short benchmark measures 748 ms, and its sustained run measures 676 ms.
The sustained warm decode rates remain 237.96 and 235.66 tokens/s with the complete original trace.
This change reduces loading time without changing weight representation or GPU inference arithmetic.
GPU tests preserve every source bit across all four alignments and odd tails, including a transfer larger than the staging buffer.
The boot transfer test also checks unaligned uploads and downloads with CPU and GPU boundary guards.

The first 2026-10-23 follow-up uses native copy-engine transfers, two QMD waves, fused Qwen residual operations, and a versioned DOOM texture cache.
An A-B-B-A comparison uses the same cuTile binary and host CPUs 8, 10, 12, and 14.
The control disables only the residual fusions and texture validation shortcut.
Host load averages range from approximately 0.2 to 2.2.

| Workload | Control | Follow-up |
| --- | ---: | ---: |
| Warm Qwen decode, four sessions | 220.69 to 230.22 tokens/s | 222.22 to 240.60 tokens/s |
| Mean warm Qwen decode | 227.42 tokens/s | 231.02 tokens/s |
| Warm 22-token prompt, median | 11 ms | 10.5 ms |
| Uncapped DOOM `demo1`, two guests | 1518.87 and 1509.67 FPS | 1628.07 and 1617.50 FPS |
| DOOM scene preparation, mean cycles/frame | 651,930 | 543,820 |

DOOM improves by about 7.2%, with almost unchanged palette conversion and display costs.
Qwen removes 48 launches per token, reducing 64-token decode work from 19,520 to 16,448 launches.
Its mean throughput improves by 1.6%, but individual sessions overlap substantially.
These samples do not establish a stable Qwen throughput gain from the fusions alone.
Every Qwen session preserves the exact token checksum and passes all layer, batch, and cache references.

A fresh matched Linux guest measures 262.31 and 262.82 tokens/s for independent warm sessions.
Its prompt times are 6.74 and 7.14 ms, with the same weights, token work, and FP32 caches.
All six Linux sessions reproduce every greedy token choice.
The native sample mean remains about 12% below Linux throughput, so the parity goal stays open.
The DOOM Linux reference still uses a headless software renderer and does not establish matched GPU rendering parity.

An optional serialized GPU-timestamp pass separates device work from CPU preparation.
It measures approximately 9.90 ms of prefill device work, including 4.12 ms in regular projections and 2.64 ms in recurrent blocks.
Split projections and their reduction use another 2.54 ms.
Timestamp instrumentation changes scheduling, so these values identify costs rather than replace normal throughput measurements.

The streaming essay demonstration generates 4,096 tokens in 24.393 seconds after the first token, averaging 167.88 tokens/s.
Its longer context, terminal output, and recording costs differ from the 64-token benchmark.
This result is not a regression measurement against the short benchmark.

### Recurrent prompt processing

The combined recurrent kernels reach 255 registers per thread and contain local-memory spill instructions.
Prompt batches now separate convolution history from the recurrent matrix and distribute each head across 32-column blocks.
The new kernels compile with 96 and 246 registers per thread, respectively, with 128 threads per block and no local-memory spills.
The operator suite and all layer, batch, and cache references pass with unchanged tolerances and exact greedy tokens.

An initial warm 22-token prompt falls from approximately 11 ms to 9 ms.
The serialized device profile reduces the recurrent portion from 2.64 ms to 1.13 ms across 18 layers.
A 16-column variant increases that portion to 1.46 ms and is removed.
Single-token decoding retains the combined kernel.
The separated stages reduce its device work but add launches, allocations, and request handling, reducing observed throughput.

### Grouped recurrent projections

The single-token recurrent path now groups its four independent input projections into one cuTile dispatch and one output allocation.
The recurrent kernel reads four disjoint ranges from that buffer.
The grouped entry uses the existing FP32 reduction arithmetic and 56 registers per thread.
Every earlier kernel instruction section matches the control byte for byte.
The model weights and archive format remain unchanged.

An A-B-B-A comparison uses the same compiled module, CPU affinity, and token work in all four guests.
The grouped guests each produce two warm samples of 250.98 tokens/s.
The controls measure 207.79 to 240.60 tokens/s, with noticeable variation between sessions.
The grouped mean is about 4.4% below the fresh Linux mean of 262.57 tokens/s.
These samples establish progress toward parity without removing the remaining prompt-latency or long-context work.
Warm prompts measure 8 to 9 ms in the grouped guests.
Every session retains the exact checksum and passes all layer, batch, and final-cache references.

Each generated token now requires 203 native kernel launches, compared with 257 before grouping.
The 64-token decode count falls from 16,448 to 12,992 launches, with queue batches falling from 768 to 704.
Protocol version 5 expands the per-kernel profile array to 41 entries, so applications require the matching kernel and SDK.
The operator suite covers all four output ranges, partial reduction tiles, unaligned word offsets, and neighboring output guards.

A separate 64-bit weight-load experiment preserves correctness but increases register pressure and reduces warm throughput to approximately 220 tokens/s.
Its immediate control measures 238.81 tokens/s in both warm sessions. The experiment is removed.
A single-token Tensor Core output-head variant also shows no complete-workload benefit and is removed.

### Four-tap decode specialization

Qwen uses four convolution taps, while the generic combined recurrent kernel reserves space for eight.
The four-tap specialization halves the live history and convolution-weight tiles.
It still uses 256 threads and reaches 255 registers per thread, but emits no local spill loads or stores.
The compiler audit rejects future spill instructions in this entry.
Larger convolution histories retain the generic kernel.
All earlier kernel instruction sections remain byte-identical.

The first warm decode samples reach 262.30 and 263.37 tokens/s with 8 and 9 ms prompt times.
The fresh matched Linux reference measures 262.31 and 262.82 tokens/s.
These short-decode samples reach the Linux throughput range while preserving the exact token checksum and every numerical reference.
A repeat reaches 234.43 and 252.96 tokens/s, while its following Linux reference reaches 262.76 and 262.93 tokens/s.
The first sample shows that throughput is achievable, but repeated runs do not yet establish consistent parity.
Prompt latency, cold loading, and longer contexts still require separate measurements.

## Earlier retained measurements

The final retained sample on 2026-10-23 uses the cached QMD templates and two submission waves described below.
All GPU programs still use cuTile Rust. The corner overlay is removed, and FPS remains beside the window title.
Both guests use host CPUs 8, 10, 12, and 14, with the same audited weights and token work.

| Workload | D3OS | Matched Linux reference |
| --- | ---: | ---: |
| Warm Qwen generation, independent sessions | 215.49 and 230.22 tokens/s | 253.11 and 251.36 tokens/s |
| Warm 22-token prompt | 11 and 10 ms | 7.61 and 7.66 ms |
| AlexNet forward pass, 20 repeats | 1.30 ms mean | No fresh matched run |
| Uncapped DOOM `demo1`, three guests | 1153.22, 1270.89, and 1270.89 FPS | Earlier headless software reference only |

The Qwen sample means leave D3OS about 12% below Linux generation throughput.
Prompt latency also remains higher, so the Linux performance goal stays open.
Every native session retains the exact 64-token checksum and passes all layer, batch, and final-cache comparisons.
All six Linux sessions reproduce every greedy choice, with FP32 caches and full GPU offload.
Linux reset-context warm sessions measure 255.18 to 255.55 tokens/s and 6.62 ms for the prompt.

DOOM renders through the native NVIDIA backend, with a 320-by-200 indexed surface scaled to 640 by 400 for presentation.
The demo runs uncapped for 7117 game ticks. These figures do not describe the interactive frame limiter.
The host runs concurrent Java builds; recorded load averages vary around 12 to 14 during this final suite.
The earlier A-B-B-A comparison below records the timing variation and does not isolate a driver-only DOOM regression.
The retained build also passes the separate 600-frame pixel comparison and GPU checks after game exit.
The production image, host tests, SDK Clippy checks, and both pushed CI workflows pass.

Further projection layouts, running Tensor Core accumulators, and bulk descriptor copies showed no complete-workload benefit in their measured runs.
Those experiments are removed from the maintained implementation.

## Maintained cuTile base

All 41 native inference and rendering operators now use NVIDIA cuTile Rust.
The former NVPTX crate and GPUOcelot test harness are removed.
The host compiler produces Tile IR 13.4 and assembles native Ada binaries with CUDA 13.4.92.
D3OS retains its Rust driver, inference API, and buffer ownership.
Explicit command scopes now group dependent operations before a completion wait.

The migration baseline below precedes the later follow-ups. It uses host CPUs 8, 10, 12, and 14 on 2026-10-23:

| Workload | Before migration | cuTile observations |
| --- | ---: | ---: |
| Qwen warm generation | 160.80 to 164.52 tokens/s | 152.75 to 153.85 tokens/s |
| Qwen 22-token prompt | 15 ms | 20 to 21 ms |
| AlexNet warm forward pass | 1.45 ms | 1.30 ms |
| DOOM uncapped `demo1` | 1423.40 FPS | 1127.13 to 1339.22 FPS |

These are samples from a shared development host, not fixed performance guarantees.
The fresh DOOM baseline has one run; its earlier native measurements also show substantial variation.
The cuTile results retain the existing precision policy and independent numerical checks.
Qwen produces identical 64-token benchmark output and passes all 24 layers across 29 reference positions.
AlexNet passes all eight layer references.
DOOM passes 600 frames of pixel comparisons, exits cleanly, and then passes the GPU operator suite.

The first complete cuTile port generated only 106.85 to 109.59 tokens/s.
Its combined vector and matrix entry reserved matrix resources during decoding.
Separate matrix, small-vector, and wide-vector entries recovered much of that loss.
Wide BF16 projections use paired 32-bit weight loads.
Their measured bandwidth increased from 272 to 556 GB/s.
Smaller projections retain separate cuTile implementations; their later layout changes are described below.
All variants use the same native request ABI.

D3OS now reads fixed thread counts from the binaries and rejects mismatched launches.
It binds compiler constant banks and selects an Ada shared-memory partition from each kernel's requirements.
The native timer check compares QMD timestamps against host elapsed time.
The scratch check runs a cuTile kernel with real local-memory spills and verifies every checksum.

cuTile remains the maintained base despite the remaining regressions.
The Linux throughput target is still open.
Projection latency, matrix scheduling, recurrent tile layout, submission overhead, and presentation remain optimization work.

### Copy-engine and service-profile follow-up

Host transfers now use the dedicated Ampere DMA COPY B engine instead of a compute copy kernel.
This path retains the existing channel and drains compute work before transferring ownership.
Its separate command page remains owned until both engine completion and the FIFO tail arrive.
The command encoding follows NVIDIA's [DMA COPY B methods](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/570.144/src/common/sdk/nvidia/inc/class/clc7b5.h).

The first exact-bit transfer suite passes all six sizes, including partial chunks and neighboring guards.
Its warm round-trip sample measures 7,529 MiB/s, compared with 6,736 MiB/s in an earlier compute-copy sample.
These initial samples do not isolate shared-host variation.
The first DOOM timedemo measures 1361.17 FPS, and the separate 600-frame pixel and input checks pass.
Qwen retains its exact tokens and full numerical references, with warm samples of 214.77 and 225.35 tokens/s.

`Nvidia::service_profile()` separates request handling from queue submission and waiting, indexed by the native operation ID.
The additive query retains request ABI version 3 and validates the complete writable destination before copying counters.
Service cycles include locking and buffer management; syscall pointer validation happens before this measurement.
The benchmark takes snapshots outside its timed intervals and reports the differences for prompt processing and decoding.
An initial 64-token decode uses 6.2 million CPU cycles for frees, excluding queue work.
Output-producing requests use 41.7 million.
This evidence favors reducing dependent launches rather than replacing the allocator without further measurements.

### Projection layout follow-up

BF16 matrix projection kernels now preserve two-dimensional tiles through masked memory loads.
The pinned Rust frontend broadcasts scalar load padding with the wrong rank for these loads.
An explicit tile selection supplies zero padding after the masked load.
FP32 matrix products, convolution, and single-token projections retain their measured layouts.

Fresh baseline sessions measured 20 ms for the 22-token prompt and 148.15 to 159.20 tokens/s for generation.
The final BF16 layout measured 15 ms and 156.10 to 160.00 tokens/s with the same native launcher.
This reduces prompt latency by 25%; these decode samples do not establish a generation speedup.
Every session retained the exact token checksum and passed the full model reference checks.
The binary audit found changes in only two entries: `tensor_project_bf16_weights` and `project_bf16_split`.
All other kernel instruction sections match the baseline byte for byte.

A further BF16 sweep compared 8-, 16-, and 32-element reduction tiles with the retained matrix layout.
At this stage, regular projections use 16 elements and split projections retain 8 elements.
The regular-projection phase decreased from 17.13 to 15.07 million dispatch cycles, including waiting.
The combined layout measured 14 to 15 ms for the prompt, with the same exact tokens and reference checks.
The 32-element variant measured 18 to 19 ms and was rejected.

Several experiments were rejected after measurement:

- Applying this layout to every matrix load increased the large FP32 projection microbenchmark from 33.49 to 47.54 microseconds.
- Increasing the Tensor Core reduction tile from 8 to 32 elements increased prompt time to 22 ms.
- Larger shared-memory carveouts showed no clear workload gain and inconsistent convolution timing.
- Expanding small-vector reduction tiles from 128 to 512 elements increased register use without improving generation throughput.

cuTile remains the maintained compiler throughout these comparisons.

The Linux kernel reference now includes the driver's dynamically loaded compiler libraries.
Without these libraries, loading the CUDA 13.4 CUBIN failed with error 221.
The reference records the driver library hashes and the benchmark source, executable, and CUBIN hashes.
A fresh same-binary comparison reports these FP32 projection times in microseconds:

| Shape (rows, inner, columns) | Native GPU timestamps | Linux CUDA events | Linux cuBLAS |
| --- | ---: | ---: | ---: |
| 1, 1024, 3584 | 10.62 | 8.67 | 8.06 |
| 1, 3584, 1024 | 11.71 | 10.04 | 8.53 |
| 1, 1024, 248320 | 1814.69 | 1813.33 | 1815.76 |
| 3025, 363, 64 | 33.59 | 30.52 | 14.95 |

The timestamp scopes differ as described above, and these are individual samples on a shared host.
The vocabulary kernel reaches comparable throughput in both systems.
The other results leave both scheduling costs and matrix-kernel efficiency as optimization work.

### CPU copies of DMA storage

The staging path uses bulk CPU copies after DMA completion and before the next GPU submission.
On x86, `REP MOVSB` preserves observable memory accesses without using SIMD registers in the kernel.
The existing completion waits and memory barriers still protect staging ownership.
Syscall entry also clears the direction flag before Rust runs, then restores the caller's flags on return.
A hardware check enters with the flag set and verifies the profile copy and restored user flag.

Three runs keep the same cuTile bundle, CPU affinity, WAD, and 7,116 rendered frames.
Mean palette-phase cycles per frame decrease from 465,954 to 431,296, about 7.4%.
Overall DOOM rates are 1324.97 to 1339.22 FPS before and 1332.06 to 1361.17 FPS after.
The overlapping ranges do not establish a reliable overall FPS improvement.
The final runs retain all model references and 600 frames of independent pixel verification.

### Immutable module images

The driver now uploads each CUBIN image once and shares it across its entry points.
Each entry retains its own register, stack, shared-memory, and constant-bank metadata.
The image remains private and immutable for the device lifetime.
This removes repeated PCIe writes of an entire module during first use of individual operators.

With the same cuTile bundle, first-use Qwen prompt processing falls from 479 ms to 18 to 19 ms across three fresh guests.
The first 64-step decode falls from 632 ms to 416 to 447 ms.
Model file access, model upload, and session allocation remain separate timings.
Warm generation still falls within the observed host variation and remains below the Linux target.
All three Qwen guests pass the complete token, layer, and cache comparisons.
AlexNet, the operator suite, and 600-frame DOOM verification also pass with shared images.

### Owned command batches

`Backend::with_commands()` now groups dependent GPU work while preserving the synchronous API outside explicit scopes.
Qwen records each transformer pass inside a scope.
The native queue owns 32 QMD slots, their constants, and their completion storage until the batch finishes.
QMD dependencies preserve order without a CPU wait after every operator.
Scope completion, CPU transfers, and profile snapshots drain pending work.
The driver also waits for the FIFO tail before reusing command storage.
Temporary allocations remain retired until that completion point.

The same 64-token decode records 19,584 launches in 896 submissions, about 22 launches per submission.
The first scope-only measurement generates 195.72 to 200.63 tokens/s, compared with roughly 150 to 160 before batching.
A 128-slot queue shows no clear overall gain and increases first-use allocation costs, so the maintained queue retains 32 slots.
A separate bulk-free API also shows no clear model gain and is removed.
Hardware checks cover queue wraparound, temporary allocation reuse, readback inside a scope, nested scopes, and returned errors.

### Small BF16 reductions

Small even-width BF16 projections now use three-dimensional `[2, 4, 32]` tiles with explicit masked loads.
The gated kernel's static shared storage decreases from 8,192 bytes to 64 bytes, excluding the driver's reserved prefix.
Its register count decreases from 64 to 48 per thread.
The compiler still chooses the physical lane layout.
Separate entries prevent odd-width fallback resources from increasing the even-width launch requirements.

The 1-by-1024 projection with 3,584 outputs decreases from 9.71 to 8.36 microseconds in GPU timestamp measurements.
The warm gated operation decreases from about 23 to 17 microseconds through the application API.
After 128 MiB of cache eviction, the gated operation remains near 43 microseconds.
The warm microbenchmark gain therefore does not establish the same improvement for model weights streamed from VRAM.
A reordered pair layout increases register use and reduces full-model throughput, so it is rejected.

### Measurements with command scopes

Three fresh native guests use the retained cuTile bundle and host CPUs 8, 10, 12, and 14.
The matched Linux guest runs afterward with the same GPU, token contract, FP32 caches, and 256-token capacity.

| Workload | D3OS | Linux reference |
| --- | ---: | ---: |
| First-use 22-token prompt | 17 ms | 245.96 ms |
| Warm 22-token prompt | 12 to 13 ms | 7.48 ms in a new context |
| Warm 64-token generation | 192.19 to 204.47 tokens/s | 256.03 to 256.78 tokens/s in a new context |
| Warm AlexNet forward | 1.45 ms | Separate reference required |
| Uncapped DOOM `demo1` | 1383.86 to 1423.40 FPS | Earlier headless software baseline only |

The six warm native generation samples average 199.56 tokens/s.
Linux reset-context samples generate 254.85 to 256.52 tokens/s, with prompt times of 6.55 to 6.96 ms.
The first-use intervals describe these prepared guests, with different kernel and graph setup costs.
They do not establish a general cold-start advantage for D3OS.
Warm native generation remains about 22% below this Linux reference, so the performance goal remains open.

DOOM uses the native NVIDIA compute renderer and a 640-by-400 output framebuffer.
The three runs complete the same 7,117 game ticks and 7,116 rendered frames, without a frame-rate cap.
The duplicate corner overlay is removed. FPS remains above the game image, beside the title.
The 600-frame pixel comparison and GPU operator checks after game exit still pass.
All three Qwen guests retain the exact 64-token checksum and pass every layer, logit, batch, and cache comparison.
AlexNet retains class 258 and passes all eight reference layers.

### Native BF16 Tensor Core products

Multi-token projections now load BF16 weights directly and split FP32 activations into a BF16 high part and residual.
Both products accumulate in FP32, with compensated summation across reduction tiles.
This changes intermediate rounding from the previous TF32 decomposition.
It retains more precision than a single BF16 activation cast, but does not promise an exact FP32 dot product.
All operator and model tolerances remain unchanged.

Regular projections use 64-element reduction tiles, while split projections use 16 elements.
The larger tile amortizes the compensation work across more inputs.
The binary audit now requires BF16 Tensor Core instructions in both entries.
Only these two instruction sections change. Single-token projections, AlexNet kernels, and rendering kernels remain byte-for-byte identical.

Three fresh guests measure 10 to 11 ms for the warm 22-token prompt, compared with 12 to 13 ms before this change.
The first measured prompt drops from 17 ms to 15 to 16 ms.
Prompt queue-wait cycles decrease from about 28.7 million to 23.8 million in the first comparison.
The full reference trace, final caches, and exact 64-token checksum pass in all three guests.
The unchanged decode kernels generate 193.94 to 204.47 tokens/s, within host variation.
Fresh AlexNet verification measures 1.25 ms per forward pass across 20 repeats.
Three fresh DOOM runs measure 1376.22, 1376.22, and 1311.03 FPS with the corner overlay removed.
These rendering samples overlap the wider observed variation and do not establish an effect from the matrix change.
The 600-frame pixel comparison and GPU checks after game exit pass again.
The complete compiler validation also passes, with the same CUBIN bytes as the hardware-tested bundle.

An independent recurrent experiment also reduced queue-wait cycles by about 4%, but showed no clear full-model throughput gain.
Its extra preparation kernels and allocations consumed the benefit, so the maintained path retains the simpler fused recurrent operation.

### Cached launch metadata

Each loaded program now retains a QMD template with its code addresses, resource metadata, and compiler constant banks.
Every launch copies the template and patches its dimensions, argument-buffer address, completion address, timing mode, and current scratch limit.
Dependency links modify only the submission copies. They cannot change the cached template.
The queue retains the existing compute and FIFO completion waits.

Three fresh guests record 27.0 to 32.8 million preparation cycles for the same 64-token decode.
The previous three guests record 52.8 to 57.7 million cycles.
These samples reduce preparation costs by about 38% to 53%, with host variation in both groups.
Three guests provide six warm generation samples from 203.82 to 210.53 tokens/s, averaging 206.37 tokens/s.
All guests pass every token, layer, logit, and cache comparison.
The GPU operator suite and timestamp measurements also pass with cached templates.
Warm prompt processing remains 10 to 11 ms, and AlexNet measures 1.25 ms per forward pass across 20 repeats.
Fresh DOOM runs measure 1423.40, 1383.86, and 1407.32 FPS with the title-bar FPS display.
The separate 600-frame pixel comparison and GPU operator checks after game exit also pass.

### CPU recording overlaps GPU execution

The queue now owns two waves of 32 slots, with separate command pages and completion reports.
The CPU records the next wave while the GPU executes the previous wave.
GPU semaphore acquires preserve stream order, and the CPU waits before it reuses a wave's storage.
Each allocation retains a retirement sequence until both compute and FIFO completion make reuse safe.
Transfers, profile snapshots, nested scope completion, process cleanup, and page-table growth still drain pending work.
GPU timestamp instrumentation retains individual synchronous submissions.

The completion reports use circular 64-bit sequence numbers and are never cleared during reuse.
A newer report also satisfies an older GPU consumer, so reuse cannot erase a pending acquire's signal.
Host tests cover counter wraparound and stale reports.
The command-scopes hardware test covers repeated wave reuse, temporary allocation reuse, readback, nesting, and returned errors.
The encoding follows NVIDIA's [Ampere FIFO semaphore fields](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/570.144/src/common/sdk/nvidia/inc/class/clc56f.h).

Three initial guests generate 214.77 to 231.88 tokens/s across six warm sessions, with the same exact token checksum.
Their complete layer, logit, batch, and final-cache comparisons pass.
Warm prompt processing remains 10 to 12 ms. AlexNet measures 1.15 ms across 20 forward passes.
The separate 600-frame DOOM pixel comparison and GPU checks after exit also pass.

A later A-B-B-A comparison runs the previous synchronous-wave driver and the new driver on the same kernel bundle.
The development host has concurrent Java builds, so the harness also records host load and CPU frequency samples.
The previous driver generates 185.51 to 209.15 tokens/s. The new driver generates 213.33 to 221.45 tokens/s.
The old DOOM samples measure 1283.99 and 1290.65 FPS. The new samples measure 1258.06 and 1169.46 FPS.
CPU display-copy cycles vary substantially as host frequencies change, so these game samples do not isolate a driver-only throughput difference.
All guests use host CPUs 8, 10, 12, and 14, with the same WAD and uncapped demo.
The synchronous path skips GPU acquires for dependencies that the CPU already observed as complete.
The corner overlay remains removed, and FPS remains beside the title.

The following sections preserve the earlier NVPTX measurements and implementation history.
They do not describe the current compiler-generated register layouts.

## Previous NVPTX measurements

Measurements on 2026-10-22 and 2026-10-23:

| Workload | Initial resident baseline | After native optimizations |
| --- | ---: | ---: |
| Qwen warm generation | 7.04 to 7.36 tokens/s | 146.79 to 162.85 tokens/s |
| Qwen warm prompt processing | 7.76 to 7.81 tokens/s | 1466.67 tokens/s |
| AlexNet forward pass | 10.0 ms | 1.45 to 1.55 ms |
| DOOM uncapped `demo1` | 536.84 to 676.89 FPS with word storage | 1290.65 to 1439.86 FPS with resident byte textures |

The AlexNet result excludes input upload and final download.
The DOOM result includes native GPU drawing, palette conversion, scaling, and framebuffer copies.
Its scene size is 320 by 200, with presentation at 640 by 400.
The regular game loop has separate timing behavior. Its HUD rate does not measure uncapped throughput.
The DOOM comparisons use three runs per renderer on host CPUs 8, 10, 12, and 14.
These CPU IDs belong to four distinct performance cores on this machine.
The earlier word renderer reports 676.89, 548.67, and 536.84 FPS.
The byte renderer reports 800.95, 943.54, and 892.81 FPS.
Both renderers use the same driver, kernel bundle, engine, WAD, and 7117-tick demo.
The baseline renderer also includes the same profiling counters.
Host CPU performance varies between runs, so these ranges do not establish a fixed percentage gain.

A new comparison keeps the byte shaders and driver fixed while changing upload and cache ownership.
The fresh per-batch baseline reports 880.19, 880.19, and 980.69 FPS.
The resident version reports 1431.58, 1431.58, and 1439.86 FPS.
Both render 7,116 frames across the same 7,117-tick demo, with the same WAD and host affinity.
The GPU shader bundle is identical in these six runs.
Later inference-kernel regression runs report 1290.65, 1407.32, 1376.22, and 1297.37 FPS with the resident renderer.
The final run retains the same 7,116 frames, upload counts, and five buffer-growth clears.
These later samples remain included in the table's range; they do not establish a rendering speedup from the inference changes.

Both launchers record `NVIDIA_HOST_CPU_AFFINITY` before starting a guest.
The pinned DOOM runs use the following commands, one guest at a time:

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/run-qemu.py \
  --firmware target/nvidia/firmware --kernels target/nvidia/kernels \
  --memory-mib 4096 --boot-check --inference gpu-doom-bench --timeout 180
taskset -c 8,10,12,14 python3 tools/nvidia/linux-reference.py --doom-benchmark
```

These CPU IDs are machine-specific. The earlier Qwen Linux measurements used unrestricted host affinity.
A new BF16 reference on these pinned cores reports 264.90 tokens/s and shuts down cleanly.
Those earlier results remain historical. The matched benchmark below replaces them for current model comparisons.

The Linux llama.cpp reference uses build `b11827`, full GPU offload, and three repetitions.
The earlier unrestricted runs report about 250 tokens/s with BF16 weights and 156 tokens/s with FP32 weights.
The FP32 file is a conversion of the same BF16 GGUF file without quantization.
The earlier `llama-bench` reference retains its default FP16 attention cache. D3OS uses FP32 state.
Both benchmarks request 22 prompt tokens and 64 generated tokens, but they use different token sequences.
The Linux hybrid-model benchmark can round prompt work to its required chunk size.
Its reported prompt throughput therefore cannot serve as an exact 22-token latency comparison.
These results establish the remaining gap, rather than an exact application speed ratio.

The GGUF reference comes from `unsloth/Qwen3.5-0.8B-GGUF` revision `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`.
The D3OS archive retains the pinned Qwen revision and original Transformers fixture described in [the Qwen guide](qwen.howto.md).

## Matched Qwen reference

The accepted comparison uses the same 22 prompt tokens, 64 decode inputs, batch limit of 32, and 256-token cache capacity.
Both guests use four virtual CPUs on host CPUs 8, 10, 12, and 14, with 12 GiB of guest RAM.
Attention caches, convolution history, and recurrent state use FP32 in both implementations.
Linux uses llama.cpp `b11827` with full GPU offload and backend greedy selection.
Automatic Flash Attention resolves to the enabled CUDA path in the accepted run.
Each decode step waits for the complete model result.
This historical comparison predates command scopes, when D3OS also waited after each individual kernel submission.

The weight audit covers all 320 text tensors and 19 model settings.
All 187 BF16 tensors match bit for bit.
The other 133 tensors match exact FP32 expansion or the converter's normalization and decay transformations.
Recomputed negative exponentials differ by at most two FP32 rounding units across the local and converter implementations.
Other converted tensors match exactly. The inference reference tolerances remain unchanged.
The archive hash, source checkpoint, GGUF hash, headers, libraries, and token contract are recorded in manifests.

D3OS exports its complete token traces outside the timed intervals.
The preparation tool independently checks the prompt with the pinned upstream tokenizer.
Linux feeds the exported decode tokens and separately computes every greedy choice.
All 12 Linux sessions across two VMs reproduce all 64 D3OS choices, with checksum `11137929795425568356`.
The launcher checks the full traces, actual FP32 cache reports, settings, offload, and evaluated token counts.
A differing greedy choice rejects the accepted comparison instead of silently changing the workload.

The fused-attention comparison uses three D3OS VMs and the two matched Linux reference VMs on the same GPU.
This comparison precedes the paired-load changes described below.
Every prompt and decode token in the new D3OS runs still matches those Linux references.

| Phase | D3OS | Linux, new context | Linux, reset context |
| --- | ---: | ---: | ---: |
| Read model into RAM | 1,683 to 2,056 ms | 5,543 to 7,077 ms | Same resident model |
| Load model from RAM | 1,900 to 1,988 ms | 815 to 1,148 ms | Same resident model |
| Create or reset session state | 6 to 20 ms | 308 to 375 ms | 0.062 to 0.068 ms after creation |
| First prompt after model load | 129 to 130 ms | 194 to 306 ms | Already warmed |
| Warm 22-token prompt | 15 ms | 6.78 to 6.90 ms | 6.32 to 6.43 ms |
| Warm prompt throughput | 1466.67 tokens/s | 3187.02 to 3246.19 tokens/s | 3423.12 to 3479.17 tokens/s |
| First 64-step decode | 442 to 458 ms | 251 to 260 ms | Already warmed |
| Warm 64-step decode | 393 to 414 ms | 245.35 to 248.14 ms | 245.97 to 247.49 ms |
| Warm generation | 154.59 to 162.85 tokens/s | 257.92 to 260.85 tokens/s | 258.60 to 260.19 tokens/s |

The first Linux model call includes initial kernel and graph setup.
These first-use values do not establish an uncached driver-installation cost.
New-context measurements match D3OS's independent-session structure.
Reset-context measurements retain Linux's context and graph allocations, with cleared model state and no prompt-cache reuse.
They provide a second target for a service that keeps its inference context alive.
Linux's long context-creation time remains outside the throughput intervals, just as D3OS session allocation does.

Both model files enter guest RAM before model loading starts.
Linux uses a private RAM file and disables model mmap. Its loader completes asynchronous weight uploads before returning.
Filesystem paths and archive layouts differ, so the file and load measurements do not establish identical storage performance.
Linux computes backend greedy selection with its model graph, including one unused candidate after the final decode step.
D3OS selects immediately before each decode step. This shifts one selection across the prompt/decode timing boundary.
The measured 64 choices and model inputs still match exactly.

Intermediate arithmetic can still differ between implementations.
The library's [BF16 GEMM path](https://github.com/ggml-org/llama.cpp/blob/b11827/ggml/src/ggml-cuda/ggml-cuda.cu) can convert activation operands to BF16 while accumulating in FP32.
D3OS retains FP32 activations through compensated TF32 decomposition.
Matching weights, cache storage, and token work does not establish identical intermediate rounding for every operator.
A separate precision policy must be measured explicitly before adopting lower-precision native operands.

The matched result confirms a prompt-latency gap of about 2.2 to 2.4 times and a decode-latency gap of about 1.6 to 1.7 times.
Host performance still varies; these samples do not promise a fixed ratio for other prompts or context lengths.
The profile continues to identify native projection work and synchronous submission as major decode costs.
Prompt attention now uses one fused launch per attention layer; projection work still dominates the remaining interval.
Read the [Linux comparison procedure](nvidia-vfio.howto.md#compare-against-linux) to reproduce the audit and both guests.

## Previous kernel implementation

The projection kernel selects a warp reduction for a single input row.
Each warp reads contiguous weights. FP32 matrices and very wide BF16 matrices reuse each input across eight outputs.
Single-row BF16 matrices with 2,049 to 8,192 outputs use four outputs per warp.
Narrow matrices use one output per warp to occupy more GPU multiprocessors.
This avoids the 15 unused rows in a padded tensor-core tile.
Larger matrix operations retain the TF32 tensor-core path.
The FP16 and BF16 tensor-core boot checks also remain active.

Aligned single-row BF16 projections now read two weights per lane through one 32-bit load.
The input width must be divisible by 64; other widths retain scalar reads.
The kernel checks weight alignment before selecting the paired reads, following the [PTX memory alignment requirements](https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands).
Both weights expand exactly into FP32, and the kernel retains FP32 fused multiply-add accumulation.
The paired lane layout changes the reduction order. Existing operator and model tolerances remain unchanged.

A pinned comparison uses the same benchmark code with the previous and current driver and GPU kernel pairs.
The four-output layout changes the driver grid; each run uses the matching pair.
The benchmark now measures packed BF16 weights directly and uses their two-byte size in bandwidth calculations.
It compares every output and runs a separate GPU timestamp pass across 100 launches.

| Single-row BF16 projection | Previous device time | Paired reads and medium-width tiling |
| --- | ---: | ---: |
| 1,024 inputs, 1,024 outputs | 5.156 µs | 5.121 µs |
| 1,024 inputs, 3,584 outputs | 9.769 µs | 6.888 µs |
| 3,584 inputs, 1,024 outputs | 8.900 µs | 7.376 µs |
| 1,024 inputs, 248,320 outputs | 918.465 µs | 914.735 µs |

The two middle cases improve by about 29% and 17% in this resident-weight test.
The small matrices can remain in GPU cache. These kernel results do not establish the same gain for full-model generation.
The vocabulary projection remains near 556 GB/s, with little change from the previous bundle.
This result points toward submission and other matrix work rather than expecting paired loads to remove the remaining Linux gap.

Two subsequent Qwen VMs retain 15 ms warm prompt times and report 146.79 to 158.81 tokens/s for generation.
Both pass all model references and reproduce every prompt and decode token from the accepted Linux comparison.
These full-model results do not demonstrate a generation speedup; host variability remains visible.
AlexNet still reports 1.45 ms per forward pass, and DOOM passes 600 pixel comparisons plus 6,447 forced atlas resets.

The driver stores QMDs, constants, commands, and completion signals in coherent guest RAM.
The CPU no longer writes each launch byte through the VRAM BAR or polls completion through PCIe reads.
The driver configures persistent channel state once and invalidates instruction caches when it loads a program.
Each dispatch retains its compute completion check and final FIFO completion check before buffer reuse.

GPU-owned VRAM uses cacheable mappings. Shared guest RAM retains volatile, system-coherent mappings.
Large application allocations use aligned 2 MiB GPU pages where possible, with 4 KiB pages for tails.
These mapping changes passed the transfer guard checks and full model references.
They did not remove the measured steady-state bandwidth gap by themselves.

RMSNorm uses one 128-thread block per row instead of one warp.
The block combines four partial sums in shared memory and publishes one common scale before writing the output.
In the measured decode profile, RMSNorm dispatch cycles fell from 137.95 million to 79.56 million across 3,136 calls.
This reduction changes the summation order. The original model tolerances and decode checksum still pass.

The recurrent delta step fuses convolution, Q/K normalization, decay, state updates, and gated output into one kernel.
One block handles each head and uses shared storage instead of global intermediate tensors.
Key and value dimensions up to 128 and convolution widths up to eight use this path.
Larger configurations retain the separate operator path.
The fused path removes 126 launches per generated token, including redundant intermediate clears.
Across 64 decode steps, zero kernels fell from 5,824 to 1,216.
The hardware suite compares output, convolution history, and recurrent state across 42 steps, including both paths.

Before cuTile, the loader rounded static shared storage plus the 1 KiB reserved prefix to 128 bytes.
It rejected requests above its fixed 32 KiB partition.
The cuTile migration replaces that limit with metadata-driven partition selection.

The fused delta step first raised warm generation from about 103 to 125–127 tokens/s.
The block-wide RMSNorm change then reached 138.83 tokens/s in both warm sessions of the next run.
A second VM reports 127.75 and 135.31 tokens/s, with the same checksum and complete reference pass.
Host variability remains visible; these runs do not establish a fixed gain percentage.
These measurements use host CPUs 8, 10, 12, and 14.
The complete Qwen reference still passes all 24 layers across 29 positions.

Matrix, RMSNorm, and fused delta operations allocate their output inside the validated dispatch.
A full writer initializes every element before the kernel publishes the new handle.
This removes a separate zero kernel while preserving zeroed ordinary allocations and process ownership.

Token selection reduces the FP32 vocabulary logits into 32 index/value pairs on the GPU.
The CPU combines these pairs after a 256-byte transfer, instead of downloading roughly 1 MB of logits per token.
Equal scores select the lowest index. Every non-finite score causes an error, even when it cannot win.

Qwen matrix weights retain their original BF16 bits in packed VRAM storage.
Embedding converts only the selected row into FP32.
Projection converts weights as it reads them and retains FP32 accumulation.
This changes storage density without rounding activations or weakening the model reference tolerances.

Qwen's feed-forward gate and up projections can share one dispatch with their SiLU multiplication.
The single-row BF16 kernel reads each input for both projections and writes only the final activation.
This removes two launches per layer, or 48 launches per generated token.
Accumulation stays FP32. Mixed storage, FP32 weights, and batched inputs retain the existing separate path.
Nine operator cases compare this operation with the CPU backend, including odd output and packed-weight tails.
Wide gated projections use four outputs per warp, compared with eight for ordinary projections.
Each gated output needs two accumulators. This layout increases active warps without doubling the accumulator count.
The assembled gated kernel uses 36 registers without stack storage or spills.
With this layout, two Qwen VMs report 148.49, 144.80, 143.18, and 140.97 tokens/s in their warm sessions.
Both VMs pass the complete model reference with the existing checksum.

The kernel benchmark also compares fused and separate feed-forward calls with resident weights.
It alternates their order across three rounds and includes allocation, dispatch, synchronization, and output release.
The small weight set can remain in GPU cache. These times do not represent streaming weights across a complete model.
The four-output layout reports 15 to 16 microseconds for fused calls, compared with 39 to 50 microseconds for separate calls.

A second pass writes a separate 128 MiB buffer before each call to create cache pressure.
GPU timestamps measure only the selected projection and activation kernels, excluding the eviction writes and CPU allocation work.
The three fused samples report 42.31 to 42.37 microseconds. Separate kernels report 71.86 to 71.99 microseconds.
Timestamp instrumentation adds scheduling and completion work, so these samples use a separate `instrumented_device_mean_us` field.

Qwen now processes prompt chunks of up to 32 tokens, one complete layer at a time.
Embedding gathers token rows on the GPU. Dense projections and normalization operate on the resulting token matrix.
Multi-row projections use the existing compensated TF32 tensor-core kernels with packed BF16 weights and FP32 activations.
Attention and recurrent cache updates retain token order within each layer.
The NVIDIA backend reuses scratch buffers and writes each token directly into the batch output.
Only the final prompt token receives a vocabulary projection during normal inference.
An independently owned GPU row copy selects that token without downloading activations.

The recorded 22-token warm prompt time falls from 126–135 ms before batching to 37–41 ms.
This gives 536.59–594.59 tokens/s across four batched VMs.
The first prompt in each VM takes 125–136 ms, including initial program loading; file and weight upload remain separate measurements.
Warm generation spans 125.00–149.18 tokens/s across those runs, with the same decode checksum.
The final VM also reports slower file and weight loading. These ranges retain that run and do not establish a fixed gain percentage.
This change targets prompt latency and does not establish a generation speedup.

Cooperative attention preparation assigns one warp to each token and head.
The warp combines its normalization sum, then rotates independent element pairs.
Queries and keys are prepared across the whole chunk. Values enter the cache through one batch copy.
Scoring still reads only the causal prefix for each token.
In the 22-token profile, preparation calls fall from 264 to 12, and cache-copy calls fall from 132 to six.
Preparation dispatch cycles fall from 17.97 million to about 0.30 million in the paired profiles.
The assembled kernel uses 27 registers, without local stack storage, spills, or block barriers.

BF16 weights have seven fraction bits and fit exactly within TF32's precision.
The BF16 matrix path therefore removes the zero weight-residual product while retaining explicit TF32 conversion.
It uses two tensor products per tile, with FP32 activations and compensated FP32 accumulation.
FP32 weight operations retain their three-product path.
The build audit checks Tensor Core instructions in the new split projection entrypoint as well as the original kernels.

Narrow prompt matrices provide too few output tiles to occupy the GPU well.
The backend now divides their input reduction into 16 independent partitions, then combines the partial results in a second kernel.
This path applies to BF16 projections with two to 32 rows, 256 to 2,048 outputs, and at least 512 input elements.
Other shapes retain their existing projection path.
The split kernel uses 37 registers, without local stack storage, spills, or block barriers.
Long, odd-sized operator checks compare against independent FP64 dot products.
Additional guard checks cover a single row, partial output tiles, and empty partitions that must write zeros.

The paired prompt profile reduces projection dispatch cycles from 33.30 million to 24.12 million, including the extra reduction launches.
Recurrent updates then account for about 30% of dispatch cycles; attention preparation accounts for less than 1%.
These cycle measurements include submission and waiting, rather than isolated device execution.

With these kernel changes, four VMs report warm prompt times of 23 to 35 ms, or 628.57 to 956.52 tokens/s.
Seven of the eight warm sessions take 23 to 26 ms; the 35 ms sample remains included in the range.
The final two VMs report 24, 24, 26, and 23 ms, with generation at 140.04 to 156.10 tokens/s.
Across all four VMs, generation spans 140.04 to 157.25 tokens/s.
The first prompt takes 142 to 152 ms, excluding file access and weight upload.
All runs preserve the decode checksum and complete sequential, batched, and cache reference checks.
These changes primarily improve prompt processing; the generation gap remains open.

Recurrent prompt updates now process the entire chunk within one launch per layer.
Each block owns one head and advances its tokens in order, without synchronization between blocks.
Its private recurrent state persists across the chunk, with one final write to the global cache.
The batch layout assigns eight value columns per warp and uses at most 512 threads per block.
Single-token decoding retains its existing four-column layout.
The batch kernel uses 40 registers and 128 bytes of local stack per thread, with no assembler-reported register spills.
The driver allocates this local storage from module metadata before launch.

In the 22-token profile, recurrent calls fall from 396 to 18, and separate output clears fall from 31 to 13.
Recurrent dispatch cycles fall from 16.68 million to 7.79 and 7.83 million in the two final VMs.
Their share of prompt dispatch cycles falls from about 30% to 16–17%.
An intermediate batch implementation without private state measured about 9.05 million recurrent dispatch cycles.
These profiles include CPU submission and completion polling; they do not isolate GPU execution.

The recurrent-kernel validation VMs report warm prompt times of 20, 20, 21, and 19 ms, or 1047.62 to 1157.89 tokens/s.
Their first prompt takes 146 and 156 ms, excluding file access and weight upload.
Warm generation spans 146.79 to 156.86 tokens/s, with the existing checksum.
Both VMs pass every sequential and batched layer, vocabulary, greedy-token, and final-cache comparison.
Operator checks also compare guarded batch buffers against consecutive single-token launches from nonzero initial state.
They retain the existing numerical tolerances, including for the changed batch reduction order.
The input guards and read-only contents remain exact.

Attention now fuses scores, softmax, value accumulation, and gating for an entire token batch.
One 128-thread block owns each token and query head, with grouped heads selecting their corresponding key/value head.
The block streams 128-position tiles and retains its running maximum, normalization mass, and normalized output.
It rescales earlier contributions when a later tile raises the maximum.
Weights are normalized before value multiplication, which avoids overflow from a large unnormalized weighted sum.
All query, key, value, and accumulator values remain FP32.
Head dimensions up to 256 use this kernel; larger dimensions retain the existing separate kernels.

Each block uses 540 bytes of static shared storage, independent of prefix length.
The driver allocates 1,664 bytes including CUDA's reserved prefix and alignment.
The assembled entrypoint uses 43 registers, without local stack storage or register spills.
The build checks its shared loads, stores, and block barriers in SASS.
The SDK validates the complete consumer before executing its checked cache-write requests.
The output uses a full-writer allocation, without a score tensor or output clear.

In the 22-token prompt profile, 396 scoring, softmax, and output calls become six fused calls.
Their dispatch cycles fall from 11.07 million to 0.23 million, with the same token work and cache capacity.
Prefill clear calls fall from 13 to one. Across 64 decode steps, clear calls fall from 832 to 64.
In the profiled decode interval, the three attention kernels previously used 49.73 million dispatch cycles; the fused kernel uses about 30.8 million.
These counters include CPU submission and completion waiting, rather than isolated GPU execution.

All six warm sessions across three new VMs report 15 ms for the 22-token prompt, compared with 19–21 ms in the preceding matched runs.
Warm generation spans 154.59 to 162.85 tokens/s. These overlapping ranges do not establish a fixed generation gain percentage.
Every new run passes the complete model fixture and matches the previous Linux prompt and decode traces exactly.
Raw guarded checks cover partial dimensions, multiple cache tiles, rising and falling score maxima, and prefixes through 1,025 positions.
A large-value case also remains finite where an unnormalized sum would overflow.
The independent CPU operator suite checks the fused path and a 257-dimensional fallback, with unchanged tolerances.
The final AlexNet regression reports 1.45 ms per forward pass. DOOM passes its full pixel checks and reports 1376.22 FPS.

The independent fixture now checks sequential and batched execution across all 24 layers and all vocabulary logits at 29 positions.
Its batches use 17 and 12 tokens to exercise cache continuation at a nonzero position.
It compares greedy selections and final attention, convolution, and recurrent cache contents.
Existing model tolerances remain unchanged.
A CPU test also crosses two 32-token boundaries after an existing prefix and checks subsequent decoding exactly.

The DOOM rasterizer builds ordered column and row lists for each command batch.
Each pixel merges its two relevant lists instead of scanning every command.
The merge preserves draw order, including overlap between columns and spans.
The production raster path stores indexed surfaces and texture data as bytes instead of words.
Each indexed-frame transfer now moves 64,000 bytes instead of 256,000 bytes.
The CPU no longer expands or contracts every indexed pixel between the C engine and the GPU.
Texture snapshots retain their content comparison, including reused zone allocations.
The renderer now retains an 8 MiB texture atlas across frames and uploads only appended snapshots.
A matching prefix can reuse a longer snapshot without extending the caller's readable memory range.
Changed contents append a new snapshot, preserving offsets already referenced by pending commands.
A hash table indexes texture identities and retains its storage across atlas generations.
Before atlas rollover, the renderer completes pending commands and then resets the CPU and GPU upload offsets.

Command and bin buffers grow as needed and remain allocated between batches.
They receive complete new prefixes before dispatch, while shader lengths bound access to those prefixes.
`Nvidia::write_bytes()` writes packed data into an existing word buffer and zero-pads its final partial word.
The byte upload and update paths use the same bulk packing helper.

The byte fuzz kernel assigns one active lane per warp to each framebuffer column.
It preserves pixel order and command order within each column, including overlapping and double-width commands.
Only vertical neighbor offsets are supported. The shader ignores invalid or nonvertical entries.
Independent columns can therefore execute concurrently without changing feedback results.
The original scan and serial fuzz kernels remain word-based references in the operator checks.
The byte checks also cover odd surface sizes, word and byte guards, unaligned tables, invalid offsets, and all four palette scales.

## Scheduling delay around GPU completion

The large vocabulary projection took about 3.36 ms through the application API, although GPU timestamps measured about 1.82 ms.
Its device bandwidth already matched the Linux kernel benchmark.
The GPU service used a spinlock whose owner could be preempted during synchronous completion polling.
The owner could remain descheduled after the GPU finished.

The service now holds a preemption guard while it owns the device lock, including process-buffer cleanup.
Interrupts remain enabled. The guard ends after the lock is released.
The same projection now takes about 1.82 ms through the application API.
Qwen generation increased from about 52 tokens/s to about 100 tokens/s with the same decode checksum and full reference results.

This follows the critical-region rule in the [Linux preemption locking guide](https://docs.kernel.org/locking/preempt-locking.html).
It retains a synchronous service, so long requests can delay scheduling on the calling CPU.
Queued submission and completion interrupts remain necessary for an efficient asynchronous service.

| FP32 projection shape | D3OS device time | Linux device time |
| --- | ---: | ---: |
| 1 by 9216, 4096 outputs | About 287 microseconds | 286.55 microseconds |
| 1 by 1024, 248320 outputs | About 1817 microseconds | 1815.62 microseconds |

The vocabulary projection reaches about 560 GB/s in both guests.
The native firmware clock sample reports a 9001 MHz memory target and a 2310 MHz SM target during boot.
Linux reports the same memory target and a 2040 MHz SM target in its startup query.
These are startup samples, rather than continuous model clock measurements.
The native clock probe also measures about 2309 MHz through GPU timer instructions.

The firmware clock query uses NVIDIA's [RM user shared data layout](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/570.144/src/common/sdk/nvidia/inc/class/cl00de.h).
The driver reads a stable timestamp around each snapshot and stops firmware polling after the boot diagnostic.
It retains ownership of the DMA region for the firmware lifetime.

## DOOM stage profile

`NVIDIA_DOOM_PROFILE` separates player-view rendering, palette/scaling transfers, and final framebuffer copying.
Each stage reports CPU cycles per rendered frame, including scheduling delays.
`NVIDIA_DOOM_KERNEL_PROFILE` reports CPU dispatch cycles per frame for each kernel.
These are synchronous CPU measurements, rather than isolated GPU timestamps.
The launcher waits for `NVIDIA_DOOM_PROFILE_DONE` before stopping a timedemo VM.

In the first matched pair, copy dispatch cycles decreased from about 578,000 to 442,000 per frame.
Fuzz dispatch cycles decreased from about 168,000 to 20,000 per frame.
The number of draw, fuzz, palette, copy, and clear operations stayed the same.
The full 600-frame C pixel comparison and the GPU self-test after game exit still pass.

The resident-upload comparison reduces zero kernels from 27,036 to five over the same demo.
Those five clears occur when command or bin buffers grow.
Copy kernels decrease from 55,793 to 49,640, while draw, fuzz, and palette kernel counts remain unchanged.
The resident atlas uploads 2,006,236 padded texture bytes over the entire demo.
It records 24,975,852 cache hits and 44,594 new snapshots, without atlas rollover in that run.
`NVIDIA_DOOM_UPLOAD_PROFILE` reports uploaded bytes, hits, new snapshots, rollover count, and buffer growth count.

Verification uses a 16 KiB atlas to force rollover within live draw batches.
The earlier 600-frame test reports 6,474 resets; the latest regression reports 6,447.
Both match C-rendered pixels, including low detail and fuzz feedback.
The launcher requires rollover coverage and checks that GPU inference still works after game exit.
The byte operator suite also checks incremental updates, shrinking odd-length writes, padding, untouched guards, and rejected ranges.

## Compiler and local-stack correctness

Warp shuffle instructions depend on execution in other lanes.
The inline assembly must not use `pure` or `nomem`, which permit assumptions that do not describe a synchronized collective.
A narrow-matrix check exposed the compiler sinking a shuffle into the lane-zero output branch.
The shared warp helper now preserves participation and uses immediate shuffle distances.
This also avoids a compiler-generated local shift table in the production projection kernels.

A larger validation run exposed a separate driver fault with a valid 40-byte local stack frame.
The same cubin passed under Linux but faulted during Qwen prefill in D3OS.
The loader now reads `EIATTR_MIN_STACK_SIZE` from CUDA ELF metadata and grows scratch storage before dispatch.
It updates the channel's local-memory base, per-TPC allocation, and QMD limit while the queue is idle.
The original failing cubin then passed the complete Qwen reference with the same decode checksum.
A boot diagnostic deliberately uses volatile local storage across 257 GPU threads to preserve this coverage.

A state-retention experiment requested 80 registers per thread in a 1,024-thread block and timed out before producing model output.
That launch exceeds Ada's 64K-register budget. The accepted batch layout uses at most 512 threads and 40 registers per thread.
The driver now rejects excessive register requests before queue submission, including rounded allocation units and partial warps.
Unit checks cover the failing configuration, exact limits, rounding, and overflow inputs.
The check follows NVIDIA's [Ada resource limits](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) and [register allocation guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy).
It validates a necessary launch limit and does not predict full kernel occupancy.

The QMD constant-buffer size uses 16-byte units. The encoded size now matches the actual 1 KiB launch buffer.
Mesa's [QMD v3 implementation](https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/+/a5af649fb690c96e3d1a1e5efb2ca199666c0ff1/src/nouveau/compiler/nak/qmd.rs) provides the same size conversion.

## Per-kernel profile

The resident Qwen benchmark reads `Nvidia::profile()` around prompt processing and decoding in the second session.
The `phase=prefill` and `phase=decode` records separate the two workloads. Logging occurs after both timed intervals.
With protocol version 3, each snapshot reports call counts, CPU preparation cycles, and immediate-launch wait cycles per built-in kernel.
`prepare_share_pct` divides a kernel's preparation cycles by the total preparation cycles in that interval.
It excludes application work and queue submission, so it does not equal that kernel's share of total model latency.
`NATIVE_QUEUE_PROFILE` separately reports submission batches, queued launches, submission cycles, and all completion-wait cycles.
Wait cycles include GPU execution and completion polling, rather than an isolated device timestamp.
GPU timestamp instrumentation forces individual submissions and changes the workload's queue behavior.
The counters cover the whole device. Concurrent GPU workloads invalidate attribution to one model.
CPU cycles are useful within a run and are not a portable time unit across machines.

## Remaining work

Qwen now batches prompt matrices and records dependent kernels inside command scopes.
Attention scoring, softmax, and output now share one tiled launch per attention layer and prompt chunk.
Recurrent updates now use one launch per layer and prompt chunk.
Earlier synchronous profiles put projections and their partial reductions at about 68% of dispatch cycles, and recurrent work at about 22%.
The current preparation counters exclude queue waiting and cannot establish the same attribution.
The next improvements must address matrix efficiency, recurrent execution, and submission costs.
The matched benchmark now uses identical token work, audited weights, FP32 caches, and equal cache capacity.
Longer contexts and prompts still need matched measurements.
The CPU token-selection work now combines only 32 GPU candidates.
Each change must preserve the existing model reference or define and measure a separate precision policy.

The previous large projection kernels matched Linux device throughput for the measured shapes.
The latest retained native sessions generate about 215 to 230 tokens/s, compared with about 251 to 256 in the fresh Linux reference.
The remaining work must reduce complete model latency, rather than optimize already comparable streaming bandwidth.

NVIDIA's cuTile Rust compiler is now the maintained base for every native GPU kernel.
Its [compile-only API](https://nvidia.github.io/cuda-rust/cutile/latest/guide/jit-compilation.html#compile-only-api) produces Tile IR without a GPU or CUDA driver at runtime.
The external `tileiras` assembler then produces a cubin for the requested target.
The [compatibility table](https://nvidia.github.io/cuda-rust/cutile/latest/reference/compatibility.html) includes Ada `sm_89` with Tile IR 13.2 or newer.
D3OS retains its native Rust driver and inference API.
The migration checks argument layouts, resource metadata, GPU execution, and independent numerical references.
The compiler does not replace the remaining work on submission ownership, synchronization, or presentation.

DOOM still transfers packed indexed frames for CPU HUD composition and copies RGB output into the console framebuffer.
Command and texture storage now remain resident, but command lists and bin indices still cross PCIe for each batch.
Each texture use still checks CPU contents to detect mutable or reused source memory.
Direct GPU presentation remains separate work.
Repeated VM starts also exposed an intermittent GSP timeout while allocating the golden compute engine.
The failed boot never reached a workload and is excluded from throughput results.
Successful reruns do not establish that this startup issue is fixed.
The Linux DOOM reference uses the same pinned engine and WAD with a headless framebuffer copy.
The two pinned Linux runs report 2306.44 and 1845.15 FPS for the same 7117-tick demo.
The earlier unrestricted-affinity Linux run reported 1586.59 FPS.
It measures a software-rendering baseline without display-server or physical-display costs.
It does not represent an OpenGL port or a Linux GPU renderer.
The Linux reference now uses a static PID 1 poweroff helper after the benchmark.
The previous asynchronous SysRq path could fault in the shell after device shutdown.
Runs with that shutdown panic remain excluded from the accepted results.

## Implementation sources

The MMU encoding follows NVIDIA's published [MMU-v2 fields](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/570.144/src/common/inc/swref/published/pascal/gp100/dev_mmu.h).
The mapping policy also follows the native [tinygrad NVIDIA page-table implementation](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/support/nv/nvdev.py).
The Linux comparison uses [llama.cpp CUDA kernels](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda).
NVIDIA's [matrix performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) explains the distinction between matrix tiling and memory-bound vector work.

The texture lookup table uses [hashbrown](https://docs.rs/hashbrown/0.17.1/hashbrown/) without its standard-library dependencies.

The shared-memory allocation follows [tinygrad’s native NVIDIA runtime](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_nv.py).

The fused recurrent mapping also uses [llama.cpp’s gated delta implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/gated_delta_net.cu) as a reference.
Its warp groups split independent value columns across key lanes. D3OS retains its existing row-major state layout.

The collective annotations follow the [Rust assembly options](https://doc.rust-lang.org/reference/inline-assembly.html#asm-options-supported-options-pure) and NVIDIA's [shuffle participation rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync).

The BF16 precision argument follows NVIDIA's [alternate floating-point format definitions](https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats).
The cuTile compiler emits TF32 conversions and selects register layouts for the maintained implementation.
The independent Rust split-reduction design follows the parallel reduction approach described in NVIDIA's [CUTLASS GEMM guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html).

The matched reference uses the pinned [llama.cpp C API](https://github.com/ggml-org/llama.cpp/blob/b11827/include/llama.h).
The weight audit follows its [Qwen conversion rules](https://github.com/ggml-org/llama.cpp/blob/b11827/conversion/qwen.py).

The tiled attention design uses the running softmax normalization described in [FlashAttention](https://arxiv.org/abs/2205.14135).
This Rust kernel retains FP32 operands and does not reuse the Linux library's attention kernels.
