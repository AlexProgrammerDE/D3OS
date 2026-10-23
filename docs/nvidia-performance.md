# Native NVIDIA performance

The native driver runs AlexNet, Qwen3.5-0.8B, and DOOM correctly on the RTX 4090 Laptop GPU.
The Linux performance target remains open. Correct output does not imply comparable throughput.

## Measurement method

These measurements use the same physical GPU through VFIO, with four KVM virtual CPUs.
Only one guest owns the GPU at a time.
D3OS uses its Rust driver, signed 570.144 firmware, and native Rust kernels.
The Linux reference guest uses driver 615.78.08 and CUDA libraries installed on the host.
Those libraries remain inside the reference guest. D3OS does not link or load them.

The resident-model benchmarks separate file access, model upload, session allocation, prompt processing, and generation.
Reference comparisons run after the timed work.
AlexNet reports 20 forward passes after one warmup pass.
Qwen reports three independent sessions with a 22-token prompt, 64 decode steps, and capacity for 256 tokens.
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

## Maintained cuTile base

All 38 native inference and rendering operators now use NVIDIA cuTile Rust.
The former NVPTX crate and GPUOcelot test harness are removed.
The host compiler produces Tile IR 13.4 and assembles native Ada binaries with CUDA 13.4.92.
D3OS retains its Rust driver, inference API, buffer ownership, and synchronous submission path.

Fresh measurements on 2026-10-23 use host CPUs 8, 10, 12, and 14:

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
Smaller projections retain separate cuTile implementations where paired loads measured slower.
All variants use the same native request ABI.

D3OS now reads fixed thread counts from the binaries and rejects mismatched launches.
It binds compiler constant banks and selects an Ada shared-memory partition from each kernel's requirements.
The native timer check compares QMD timestamps against host elapsed time.
The scratch check runs a cuTile kernel with real local-memory spills and verifies every checksum.

cuTile remains the maintained base despite the remaining regressions.
The Linux throughput target is still open.
Projection latency, matrix scheduling, recurrent tile layout, submission overhead, and presentation remain optimization work.

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
Regular projections use 16 elements; split projections retain 8 elements.
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
Each decode step waits for the complete model result. D3OS also waits after its individual kernel submissions.

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
The snapshot reports call counts, CPU launch cycles, and completion-wait cycles for each built-in kernel.
`dispatch_share_pct` divides a kernel's cycles by the total dispatch cycles in that interval.
It excludes application work outside dispatch and does not equal that kernel's share of total model latency.
`wait_pct` includes GPU execution and completion polling, rather than an isolated device timestamp.
The counters cover the whole device. Concurrent GPU workloads invalidate attribution to one model.
CPU cycles are useful within a run and are not a portable time unit across machines.

## Remaining work

Qwen now batches prompt matrices but still submits synchronous operations separately.
Attention scoring, softmax, and output now share one tiled launch per attention layer and prompt chunk.
Recurrent updates now use one launch per layer and prompt chunk.
The latest prompt profile puts projections and their partial reductions at about 68% of dispatch cycles, and recurrent work at about 22%.
The next improvements must address matrix efficiency, recurrent execution, and submission costs.
The matched benchmark now uses identical token work, audited weights, FP32 caches, and equal cache capacity.
Longer contexts and prompts still need matched measurements.
The CPU token-selection work now combines only 32 GPU candidates.
Each change must preserve the existing model reference or define and measure a separate precision policy.

The previous large projection kernels matched Linux device throughput for the measured shapes.
Recent native runs generate about 147 to 163 tokens/s, compared with about 258 to 261 tokens/s in the matched Linux reference.
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
