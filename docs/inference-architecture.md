# Native inference architecture and hardware handoff

D3OS owns the compute API, tensor operations, model execution, and application integration in this implementation.
The CPU backend is a real userspace implementation with `no_std` and `alloc`.
Host verification builds the same libraries for Linux.

The [Burn migration](burn-migration.md) replaces the public AI framework while retaining the native driver and cuTile kernels.
Its portable runtime and first native GPU primitives are operational. Model migration and GPU training remain in progress.
DOOM keeps its separate native rendering API.

The hardware-free phase delivers pretrained AlexNet and Qwen3.5-0.8B text inference inside D3OS.
The host compiler builds Rust cuTile kernels without GPU hardware.
The native NVIDIA hardware phase now runs AlexNet on the passed-through RTX 4090 Laptop GPU.
Firmware boot, GPU page tables, native kernel loading, and tensor execution are operational.
Qwen also passes all 24 layer checks and vocabulary-logit comparisons on the physical GPU.
Its native generation command replies `4` to the default arithmetic prompt and stops at EOS.
DOOM also runs through native GPU column, span, fuzz, palette, and scaling kernels.
The [renderer explanation](nvidia-rendering-design.md) describes command ordering, CPU synchronization, and game verification.

## Source layout

- `os/library/ai`: Burn-based model definitions and portable training validation.
- `os/library/burn-native`: Burn tensor primitives, checked views, and native GPU operations.
- `os/library/compute`: tensor storage, command scopes, CPU operators, and the native NVIDIA backend.
- `os/library/inference`: checked tensor archives, AlexNet evaluation, Qwen text generation, tokenization, and numerical verification.
- `os/application/infer`: D3OS file access, shell commands, and serial reports.
- `os/library/inference/examples/infer-host.rs`: host execution of the same model implementation.
- `tools/gpu-kernels`: the host cuTile compiler and Rust tile programs.
- `tools/inference`: asset preparation, cuTile compiler checks, and QEMU integration tests.
- `os/kernel/src/device/nvidia.rs`: native firmware startup, compute submission, and process-owned GPU storage for the RTX 4090 Laptop GPU.
- `tools/nvidia/run-qemu.py`: verified VFIO launch and passthrough boot checks.

The inference layer depends on the backend contract rather than CPU storage.
The NVIDIA backend implements the same contract through a D3OS compute syscall.
The model layer does not require Linux binary compatibility, a CUDA host runtime, or Python inside D3OS.

## CPU implementation

Each tensor owns its buffer and records its dimensions.
Shape products use checked arithmetic before allocation or indexing.
The CPU backend moves uploaded vectors into owned storage and exposes explicit downloads.

Convolution lowers each image into an `im2col` buffer and uses row-major matrix multiplication.
The inner matrix loop permits SIMD without global floating-point reassociation.
Fully connected layers use independent accumulators for long dot products.
Softmax subtracts each row's maximum before exponentiation.

AlexNet uses the torchvision evaluation architecture, including its preprocessing and disabled dropout.
The input size is fixed at 224 by 224 pixels.
At this size, the final feature map is already 6 by 6, so adaptive pooling is an identity.
Other input sizes and AlexNet variants are not supported by this model implementation.

## Tensor archives

An archive starts with eight magic bytes, `D3TENSR\0`.
A little-endian `u32` version and a little-endian `u32` record count follow.
Both supported versions accept one to 4,096 named tensors.
Version 1 stores every element as FP32.

Each record contains these fields in order:

1. Name length as a little-endian `u16`.
2. Rank as a little-endian `u16`.
3. Element count as a little-endian `u64`.
4. UTF-8 name bytes without a terminator.
5. Dimensions as little-endian `u64` values.
6. Contiguous little-endian IEEE 754 `f32` elements.

Names contain one to 256 bytes and cannot contain null bytes.
Ranks range from one to eight.
Dimensions must be nonzero and their product must equal the element count.
The parser rejects duplicate names, overflow, truncated records, unsupported versions, and trailing bytes.
Loading a tensor rejects nonfinite values.

Version 2 inserts a little-endian `u16` dtype after the element count in each record.
Dtype 0 stores FP32; dtype 1 stores BF16.
Two-dimensional BF16 records use the backend's packed-weight upload operation.
They pass borrowed little-endian bytes through `upload_bf16_le()`.
The NVIDIA backend validates these values once and copies them directly through its staging buffer, without a temporary packed-word vector.
CPU source alignment is unrestricted; the syscall still checks the complete readable byte range.
An odd final BF16 value uses a separate padded word, so the upload never reads past the record.
The upload completes before the borrowed storage can be released.
The CPU backend expands them exactly to FP32.
The NVIDIA backend retains packed weights for embedding, projection, and linear operations.
Other BF16 record shapes still expand before upload.
The rest of the record layout stays the same.
AlexNet archives from version 1 remain supported.

Qwen configuration records accept either 16 legacy fields or 18 fields.
The extra fields specify the recurrent query/key head count and whether word embeddings and output weights are tied.
An untied model loads `lm_head.weight` separately. Legacy records retain tied weights and equal recurrent head counts.
`Qwen::load_weights()` accepts a validated configuration and a callback that supplies each named tensor.
The callback can release its input bytes after the backend upload completes.
This interface supports streamed storage and packed weights without retaining a complete archive in application memory.

The format stores explicit data and dimensions rather than serialized Rust objects.
AlexNet separately validates every parameter name and shape before allocating its weight buffers.
The exporter writes a SHA-256 manifest alongside generated archives.
The archive itself does not authenticate its contents.

## What the cuTile checks establish

NVIDIA's pinned cuTile Rust compiler produces Tile IR 13.4.
The pinned CUDA 13.4 assembler produces native Ada `sm_89` binaries.
D3OS loads these binaries with its native Rust driver.
The guest uses no NVIDIA host runtime or Linux compatibility layer.

Host tests check SSA remapping when linking independently compiled entries.
The compiler verifies each binary's required thread count.
SASS audits require Tensor Core instructions and real local-memory spills in the scratch diagnostic.
CPU CI compiles all inference and rendering operators, but does not execute GPU instructions.

Hardware tests cover partial tiles, output guards, state feedback, and independent numerical references.
The VFIO launcher requires vector, Tensor Core, transfer, and local-memory checks before accepting a workload.
Model and rendering checks remain separate from compiler validation.

## Verified software results

Local verification on October 22, 2026 established these results:

- CPU operator and archive tests passed on the development host.
- The former PTX interpreter passed 22 cases before the cuTile migration.
- The maintained GPU compiler now targets Ada through cuTile.
- Pretrained AlexNet matched all eight independent NumPy layer references on the host.
- The small-network test passed inside D3OS under QEMU/KVM.
- Pretrained AlexNet passed all eight layer comparisons inside D3OS under QEMU/KVM.
- Both AlexNet guest tests also passed with QEMU TCG, without KVM or GPU access.
- Qwen3.5-0.8B passed all 24 layer checks and vocabulary-logit comparisons across 29 positions on the host and in D3OS KVM guests.
- Qwen passed three numerical verification runs each with two and four vCPUs after the context-switch interrupt fix.
- Qwen generated the expected arithmetic reply and stopped at EOS in three runs with four vCPUs.
- The context-switch interrupt-injection regression passed with two vCPUs under KVM and TCG, and four vCPUs under KVM.
- All 105 Qwen tokenizer cases matched the upstream tokenizer.
- Qwen passed the complete numerical trace inside D3OS under QEMU TCG without hardware virtualization.

The dog fixture produced class 258 with probability approximately 0.72448.
The first full-model guest run took approximately 1.28 seconds, including file access and verification.
That timing describes CPU execution in KVM on the development machine.
It is not a GPU performance measurement.

The test scripts preserve manifests and guest logs under `target`.
The CI workflow provides repeatable numerical tests and an optional pretrained QEMU run.

## NVIDIA hardware implementation

The current target is the RTX 4090 Laptop GPU (`10de:2757`, AD103) with one serialized compute queue.
The driver boots pinned signed firmware and allocates a private, virtually bound compute context.
The process-owned syscall interface accepts built-in kernels and opaque buffer handles.
Native compute protocol version 14 includes command scopes, queue profiles, and 55 built-in kernels.
Its expanded profile array requires the matching SDK.
Rebuild compute applications with the matching SDK. The kernel rejects incompatible request versions.

The new elementwise family dispatches eleven specialized cuTile entries for Burn tensor views.
It supports exact bit copies, FP32 arithmetic, comparisons, and mask filling, including scalar operands and strided broadcasting.
Its 32 parameters contain dimensions, signed strides, base offsets, storage extents, operation flags, and scalar bits.
The driver validates every reachable input address before dispatch and rejects output aliasing.
Every entry writes the complete output, so native allocation avoids a separate clear kernel.
Empty logical results reuse resident storage without a GPU launch.
Hardware checks cover partial tiles, neighboring guards, rejected addresses, exact floating-point copies, and chained resident operations.

The gather entry supports FP32, I32, and boolean values with I32 indices.
Its 32 parameters describe both views, the selected axis, and the output shape.
The driver validates reachable storage ranges and isolates the output and bounds flag from all input buffers.
The kernel masks invalid source reads and atomically records a bounds error.
The tensor API reads one four-byte flag and returns an error before exposing an invalid result.
This control read synchronizes queued work. Tensor values stay on the GPU until an explicit data read.
Vector select expands a checked index view and uses the same kernel without an index upload or materialization.

The scatter family reuses the indexing geometry with an operation tag in the rank word's high half.
Three entries provide FP32 addition, wrapping I32 addition, and boolean OR.
Values are contiguous at the kernel boundary. Indices and the destination support signed strides.
The tensor API copies shared destinations and materializes noncontiguous values on the GPU before dispatch.
The kernel updates only selected elements, so it cannot use the uninitialized full-writer allocation path.
Private copies isolate valid partial updates when another device index is invalid.
The same bounds flag and host synchronization report those errors before the caller receives an output.
FP32 updates use integer compare-and-swap and explicit non-FTZ addition, including for NaN values.
Colliding FP32 additions have an unspecified order. I32 updates wrap modulo 32 bits.

The native Burn convolution uses a checked NCHW plan with groups, dilation, and per-side padding.
Its 31 forward parameters include geometry, signed input and weight views, storage extents, and an optional bias flag.
The forward entry reads image windows directly into compensated TF32 Tensor Core products and writes a dense result.
It does not allocate an intermediate window matrix. A noncontiguous bias requires a GPU copy before the fused launch.
Backward passes extract resident windows for the weight gradient and fold matrix products into the input gradient.
Each fold thread sums the windows that contain its input element, so overlapping windows require no atomic writes.
Bias gradients use resident reductions. Empty batches return empty input gradients and zero weight and bias gradients.
The driver rejects invalid geometry, address ranges, and output aliasing before dispatch.

The pooling families normalize one to three spatial dimensions into NCDHW metadata.
Their 32 parameters contain input and output sizes, window geometry, padding, dilation, flags, a checked source view, and the operation.
Backward source views describe the output gradient rather than the original input.
Max pooling writes cached I32 indices and FP32 values into separate halves of one full-writer allocation.
The tensor views share ownership of that allocation. Padding-only windows contain index -1 and negative infinity.
Average and adaptive average backward entries sum contributions per input element without atomic writes.
Adaptive bounds use 64-bit products before division, while final element addresses retain the checked 32-bit range.
Max backward masks the -1 sentinel and uses resident scatter addition with its four-byte bounds result.

The reduction family provides sum, mean, product, minimum, maximum, and their two index operations for FP32 views.
Each output retains the reduced dimension with size one. Input strides can be negative or zero.
The 24 parameters describe the shape, reduction axis, operation, signed strides, and original storage extent.
The driver validates the complete address range and rejects output aliasing before dispatch.
Extrema propagate the first NaN and select the first logical index for ties, including signed zeros.
Empty reduction axes produce zero for sum, one for product, and NaN for mean.
Extrema reject empty reduction axes when the output is nonempty. Empty outputs require no launch.
Hardware checks cover partial tiles, multiple passes per row, strided broadcasting, nonfinite inputs, exact extrema bits, guards, and resident execution.
All 88 earlier GPU instruction sections remain byte-identical after the seven reduction entries are added.

The elementwise family also includes 21 unary math entries and a broadcast power entry.
They use the same checked strided layout protocol. Unary operations reject scalar-operand flags.
The compiler uses full-range exponential and hyperbolic tangent operations, with explicit rounding for square root and reciprocal.
The log1p implementation corrects cancellation near zero and preserves signed zeros.
Rounding uses ties to even and preserves large integral values and NaN payloads.
Sign follows Burn's contract: both zeros and NaNs produce zero.
Three classification entries produce canonical boolean words.
A separate full-writer kernel fills FP32, I32, and boolean storage without a host upload or a separate clear.
Native GPU checks compare these operations with independent libm references, including 2,816 FP32 edge values.
All 95 earlier GPU instruction sections remain byte-identical after the 23 math and fill entries are added.

The typed-word family adds 29 entries for I32 arithmetic, comparisons, bitwise operations, boolean logic, masks, and conversions.
Arithmetic wraps at 32 bits. Shift counts wrap modulo 32, and signed right shifts preserve the sign.
The float-to-I32 conversion truncates toward zero, clamps overflow, and maps NaN to zero.
It selects finite, representable operands before conversion to avoid undefined floating-to-integer conversion results.
Boolean outputs use canonical zero/one words.
Boolean-to-I32 casts share GPU storage and retain ownership without a launch, allocation, or transfer.
The driver reuses the checked strided address contract and rejects invalid operand flags and boolean scalar words.
D3OS hardware checks pass wrapping arithmetic, signed comparisons, conversions, shared ownership, partial tiles, and neighboring guards.
All 118 earlier GPU instruction sections remain byte-identical after the typed-word entries are added.
Integer division, remainder, and general indexed operations remain backend work.

`Backend::upload_quantized()` accepts borrowed Q8_0, Q3_K, Q4_K, Q5_K, and Q6_K weight blocks.
It checks block lengths and finite FP16 scales before allocating device storage.
The CPU backend decodes these blocks to FP32. The NVIDIA backend retains the original packed bytes in VRAM.
Embedding gathers and projections decode weights inside the GPU kernels.
Single-token projections use vector kernels. Batched projections use three compensated TF32 Tensor Core products.
Explicit weight downloads decode into a separate FP32 output buffer.
The tensor archive and complete large-model loader still require separate integration.

The D3OS kernel verifies handle arithmetic and launch dimensions before dispatch.
GPU tensor kernels use native Ada SASS compiled from Rust device code.

AlexNet passes every layer comparison on the physical GPU with the original reference tolerances.
Its matrix operations use three TF32 tensor products and compensated FP32 accumulation.
An earlier running accumulation missed one third-layer value by about `2.24e-4`.
Compensated accumulation corrects that mismatch without changing the reference tolerance.
Large FP32 matrix products use 16-wide reduction tiles when all three dispatch dimensions meet the specialization's bounds.
The remaining products retain the eight-wide kernel. Both use the same three TF32 products and compensated FP32 accumulation.
The larger tile amortizes compensation work without introducing register spills.
The compiler audit checks its Tensor Core instructions and rejects local spills.
AlexNet groups each forward pass in a command scope. Intermediate downloads and the end of the scope complete pending GPU work.
Convolution lowering and pooling allocate their outputs through full writers, so they do not need separate clear kernels.
Single-image convolutions also allocate their matrix result through its writer.
Batched convolutions retain a cleared combined output while reusing the complete column buffer for each image.
The hardware operator suite also covers partial tiles, long reductions, batched convolution, and buffer reuse.
The loader reads module stack metadata and sizes local scratch storage before submission.
It also reads each entrypoint’s static shared-memory section and includes the reserved CUDA prefix in its QMD allocation.
The driver selects a 32, 64, or 100 KiB shared-memory partition, including the reserved 1 KiB prefix.
Before submission, the driver also checks the entrypoint's register count against Ada's 64K-register budget per block.
It accounts for register allocation granularity and partial warps.
An oversized launch returns an error before the driver queues GPU work.
A GPU boot diagnostic exercises a real local stack frame across partial blocks.

Qwen matrix weights remain in packed BF16 storage while activations and accumulators retain FP32 values.
Single-row projections use coalesced warp reductions.
For even input widths, each lane reads two BF16 weights through one 32-bit load.
Bit expansion preserves both weights exactly. Activations and fused multiply-add accumulation remain FP32.
Other widths retain scalar reads, including odd rows and tails.
Small BF16 projections use tiles with shape `[2, 4, 32]`, with two groups of 32 weight pairs per output.
This layout reduces the shared storage for partial sums. cuTile selects the physical register and lane layout.
Wider projections retain eight outputs per warp and reuse inputs across the vocabulary matrix.
Multi-row projections load native BF16 weights and split FP32 activations into BF16 high and residual parts.
Two BF16 Tensor Core products use compensated FP32 accumulation.
Regular projections reduce 64 inputs per iteration. Split projections reduce 16 inputs per iteration.
This decomposition retains more activation precision than a single BF16 cast, but differs from the previous TF32 decomposition.
The existing operator and model tolerances remain unchanged.
FP32 weight operations retain three tensor products and compensated FP32 accumulation.
For two to 32 rows, 256 to 2,048 outputs, and at least 512 inputs, BF16 projections split the input reduction into 16 partitions.
A second kernel combines their FP32 partial results with compensated accumulation.
Other shapes retain their existing projection path.
The SDK checks storage types before dispatch and rejects packed weights in unsupported FP32 operations.
`TransformerBackend::gated_project()` combines the feed-forward gate projection, up projection, and SiLU multiplication.
The native single-row BF16 path writes one complete output without intermediate activation buffers.
FP32, mixed-storage, and multi-row calls retain the separate projection path.
The CPU backend provides the same operation through its existing primitives.

`TransformerBackend::residual_norm()` combines residual addition and zero-centered RMSNorm.
It returns the updated residual stream and an independent normalized tensor.
The native path supports widths up to 1,024 and writes both outputs in one launch.
`project_add()` adds a projection directly to an owned residual tensor.
The native single-row BF16 path supports even input widths and up to 8,192 outputs.
Other configurations use the separate operations through the same API.
These operations remove 48 launches per generated Qwen token without changing storage precision or reference tolerances.

`TransformerBackend::project_delta()` projects the hidden batch and applies its recurrent mixer.
`DeltaWeights` supplies the four projection matrices and the recurrent parameters.
The single-row native BF16 path groups QKV, gate, beta, and alpha projections into one dispatch.
It writes their results into one owned buffer, then binds its disjoint ranges to the recurrent kernel.
Temporary storage remains owned until all queued readers complete.
This removes another 54 launches per generated token without changing weight archives or accumulation precision.
Other shapes and storage types retain the separate projections through the same API.

Prompt processing uses chunks of up to 32 consecutive tokens and completes each layer across the whole chunk.
`embedding_batch()` gathers token rows, and batched attention and recurrent operations advance a single sequence in token order.
The NVIDIA backend prepares all attention queries and keys cooperatively, with one warp per token and head.
Each warp combines its normalization sum and rotates independent pairs without global scratch reads.
The backend copies the value batch once, then scores each token against only its causal cache prefix.
It checks the complete attention request list before the first cache write.
For head dimensions up to 256, one block per token and head fuses scoring, softmax, value accumulation, and output gating.
The block streams the causal prefix in tiles of 32 positions, with a running maximum and normalization mass.
It normalizes each tile's weights before multiplying values and rescales the previous normalized output when the maximum changes.
This keeps the intermediate storage bounded and avoids overflow from an unnormalized sum of large finite values.
All attention arithmetic remains FP32. Larger head dimensions retain the separate operator path.
Single-token decoding with at most 16 query heads splits prefixes of at least 512 positions across 16 blocks per head.
Each block writes a normalized partial output, its maximum score, and its normalization mass.
A second kernel rescales these partials against their shared maximum, combines them, and applies the output gate.
Empty tail partitions write zero mass and do not affect the result.
This follows the sequence partitioning used by [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html).
The fused path allocates its complete output during dispatch and removes the separate score tensor.
The backend checks the consumer and cache-write requests before executing the cache updates.
For supported recurrent dimensions, prompt batches separate convolution, recurrent updates, and output normalization.
The convolution kernel owns each channel's history across all rows and writes its final history after the chunk.
The recurrent kernel distributes each head across groups of 32 value columns.
Each block retains its private state columns across rows and writes their final values when the chunk completes.
This division removes register spills from the batch kernels and increases the number of independent blocks.
Single-token decoding retains the combined kernel because the additional launches outweigh its lower device work.
Convolution histories of up to four taps use a smaller compile-time tile in that kernel.
The eight-tap tile remains available for larger histories.
This specialization removes decode spills without adding dispatches or changing the recurrent state layout.
Larger recurrent configurations retain the separate operator path.
The batch dispatch validates all input, output, history, and recurrent extents before changing state.
It initializes the entire output directly, without a separate clear.
The NVIDIA backend binds row ranges directly and retains scratch storage for the fallback operations.
It validates the complete batch shape and cache capacity before the first state update.
`row()` returns independent storage, so consuming an operation's result cannot mutate its parent batch.
Normal prefill projects vocabulary logits only for the final row, without host activation transfers.
`prefill_observed()` exposes layer batches and their starting positions for verification.

The fixture verifier checks both sequential and batched layers, logits, greedy decisions, and final caches.
The numerical tolerances remain unchanged. A failed layer observer invalidates the session.
The single-token attention and delta APIs retain their one-row validation and use the same native batch primitives.

DOOM uses byte-addressed indexed surfaces and textures with word-addressed command buffers.
The byte raster kernels receive logical byte lengths, while the syscall validates the enclosing word ranges.
`Nvidia::upload_bytes()` packs data into aligned words and initializes trailing padding.
`Nvidia::write_bytes()` updates an existing word buffer at a word offset, with the same padding rule.
It checks the complete range before packing or transfer.

The renderer retains command and bin buffers and a bounded 8 MiB texture atlas across frames.
`raster::TextureAtlas` stores immutable snapshots with aligned offsets and detects changed bytes at reused identities.
Shorter requests can reuse matching prefixes of longer snapshots without reading beyond the supplied slice.
Only appended texture bytes cross PCIe. Previous snapshots remain valid until all pending draws complete.
Atlas rollover completes pending draws before reusing offsets, and retains CPU table and byte-storage capacity.
The verification mode uses a 16 KiB atlas and requires rollover during its pixel comparison.
The fuzz kernel preserves ordering within each column and permits only vertical neighbor offsets.
This permits concurrent columns without changing the original pixel feedback.

RMSNorm uses one 128-thread block per row, with shared partial sums and one common scale.
The hardware checks cover narrow rows, multiple rows, and widths that exceed or do not divide the block size.

The recurrent delta step fuses convolution, normalization, state updates, and gated output into one block per head.
This path supports key and value dimensions up to 128 and convolution widths up to eight.
Larger configurations retain the separate operator path.
Both paths keep state in VRAM and pass independent output and state comparisons after each step.

Matrix, row-copy, embedding-gather, RMSNorm, and fused delta operations combine output allocation with the kernel that fills it.
The driver validates a full writer before it publishes the new handle.
Outside command scopes, that writer completes before the call returns.
Inside a scope, GPU consumers retain queue order, and CPU readback waits for completion.
This removes a redundant zero kernel without exposing previous buffer contents to the CPU.
Ordinary allocations remain zeroed, including recycled storage.

`Backend::with_commands()` groups dependent operations without an immediate CPU wait after each launch.
The CPU backend runs the closure directly. The NVIDIA backend records process-scoped, nested command scopes.
Qwen uses one scope for the embedding and all transformer layers in each prompt chunk or decode step.
Readback, transfer, profile snapshots, and scope completion drain the queue.
Nested scope completion also drains the queue. A returned closure error still closes the scope.

The submission region owns two waves of 32 slots, with separate QMDs, launch constants, command pages, and completion reports.
QMD dependencies use GPU virtual addresses and preserve launch order.
Only the final QMD in each wave signals compute completion.
The CPU records the other wave while the GPU runs, then waits for compute and FIFO completion before slot reuse.
Circular 64-bit semaphore acquires order consecutive waves. An already completed dependency needs no additional GPU acquire.
The reports retain their monotonic values, so reusing slots cannot erase a signal that another wave still needs.
Dropped buffers retain their retirement sequence until that sequence completes, including during process cleanup.
CPU transfers and scope completion drain both waves. Page-table growth also drains the queue before it changes mappings.
Transfers now use the Ampere DMA COPY B engine on the existing channel, with separate staging, command, and completion storage.
The CPU waits for both the copy-engine release and FIFO tail before it reuses that storage.
The copy launch uses virtual addresses, pitched layouts, and a system-memory flush.
The transfer checks preserve exact bits across partial chunks and allocation guards.
GPU errors retain mapped storage and mark the device unhealthy.
The loader shares each immutable CUBIN image across entry points, with separate resource metadata for each entry.
Each program also caches its fixed QMD fields, including code addresses, resource limits, and compiler constant banks.
Every launch copies that template before it changes dimensions, constants, completion addresses, timing mode, and the current scratch limit.
Dependency links change only the owned submission copies.

The optional device profile returns per-kernel CPU cycle counters through the checked compute syscall.
Snapshots include dispatch counts, CPU preparation cycles, immediate-wait cycles, and instrumented GPU nanoseconds.
`QueueProfile` separately counts all submission batches, launches, submission cycles, and completion-wait cycles.
GPU timestamp instrumentation is disabled by default. Enabling it forces individual submissions for kernel attribution.
The counters cover the whole device, so exclusive workloads are required for useful attribution.

The [tinygrad direct PCI backend](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/support/nv/nvdev.py) provides a reference for initialization.
Its [firmware and GSP implementation](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/support/nv/ip.py) provides the corresponding management protocol.
[NVIDIA's hardware definitions](https://github.com/NVIDIA/open-gpu-doc) describe published command interfaces.
The hhuOS `origin/arc_driver` branch provides a Rust example of OS adapters, GPU memory, and compute submission.
Its Intel register encodings and page tables are specific to Intel hardware.

The implementation milestones are:

1. Add ECAM access and device capability handling to D3OS PCI support.
2. Map the GPU BARs with the required memory attributes.
3. Implement owned DMA allocations, reset handling, firmware boot, and GSP communication.
4. Implement VRAM allocation, GPU virtual addresses, page tables, and translation invalidation.
5. Create compute and copy channels, then submit work with completion fences and bounded timeouts.
6. Assemble device kernels for the chosen architecture and load their code, metadata, and constants.
7. Execute vector addition and matrix multiplication against CPU references on the real GPU.
8. Implement the remaining inference operators in the NVIDIA backend and run AlexNet through that backend.
9. Add process isolation, resource cleanup, fault diagnostics, and recovery tests.
10. Optimize measured bottlenecks and add larger model operators.

Milestones 1 through 4, native compute in 5, and milestones 6 through 8 are operational on the target GPU.
The userspace service validates buffer ownership and releases storage after process exit.
Copy-engine transfers, separate hardware address spaces for arbitrary user kernels, and recovery without VM reset remain unfinished.
Tensor execution and Qwen transformer operators are operational.

Initial bring-up can use driver-owned staging buffers and explicit copies.
It does not require shared CPU/GPU virtual addresses or demand-paged GPU memory.
Arbitrary userspace kernel submission still requires memory isolation and a documented unsafe launch boundary.

Qwen3.5-0.8B text generation now uses the native `TransformerBackend` interface.
It includes recurrent and grouped attention, normalization, rotary positions, SwiGLU, cached decoding, tokenization, and greedy sampling.
The NVIDIA backend implements these operations with GPU-resident state.
It validates each token row’s dispatch sequence before that row changes its cache.
The syscall checks every modified buffer, including convolution histories and recurrent state.
The hardware suite compares outputs and state across multiple attention and gated-delta steps.
The pretrained model then passes every original reference comparison at 29 token positions.
The transfer path uses a persistent, driver-owned 1 MiB staging allocation in coherent guest RAM.
GPU copy kernels move bounded chunks between that allocation and process-owned VRAM.
GPU clearing preserves buffer isolation when storage is fresh or recycled.
Token selection reduces vocabulary logits on the GPU and downloads 32 candidate pairs for the final CPU comparison.
It preserves the first-index tie rule and rejects every non-finite score.
See [native NVIDIA performance](nvidia-performance.md) for resident-model timings and Linux comparisons.
Read [the Qwen guide](qwen.howto.md) for API usage, reference verification, memory requirements, and multicore VM commands.

## GPU debug machine

The development host can keep Linux while a D3OS VM receives a secondary GPU through VFIO.
D3OS does not need installation on the machine's boot disk.

The hardware phase needs:

- Remote access to a Linux host with KVM and IOMMU support.
- An NVIDIA GPU that the host can release to VFIO, preferably separate from its display GPU.
- Permission to manage that specific device and start QEMU with passthrough.
- Access to serial logs, QEMU's monitor, and a debugger.
- A recovery method for GPU hangs, including a host reboot when device reset fails.

A CUDA container alone does not provide the necessary PCIe device ownership.
The actual PCI device and IOMMU group must be inspected before any driver unbinding.
The hardware phase must verify firmware initialization, DMA visibility, synchronization, and recovery on that device.

### Hybrid laptops

A laptop's display MUX can connect the internal panel directly to its NVIDIA GPU.
In that state, the host cannot release the GPU while the graphical session uses the panel.
The MUX must first route the panel to integrated graphics, usually through a firmware or vendor setting and a reboot.
The NVIDIA device must remain enumerated and available for VFIO.
Some integrated-only modes disable the discrete GPU, so the selected mode must preserve device access.

On supported ASUS systems, [supergfxctl documents a VFIO mode](https://gitlab.com/asus-linux/supergfxctl/-/blob/main/README.md).
Its read-only `--get` and `--supported` commands report the current switching state.
The available modes depend on the machine, firmware, and kernel.
Before detaching the GPU, verify that the active panel uses the Intel DRM device and that the complete NVIDIA IOMMU group is available.

The development laptop's read-only inspection found Intel Raptor Lake graphics and an RTX 4090 Laptop GPU (`10de:2757`).
The NVIDIA GPU at `0000:01:00.0` shares group 15 only with its audio function at `0000:01:00.1`.
The panel now uses Intel after switching the ASUS firmware to Hybrid mode.
Both NVIDIA functions now bind to `vfio-pci`, and their isolated group remains available to the VM.
The session's locked-memory limit was raised to 16 GiB for VFIO.
The [VFIO setup guide](nvidia-vfio.howto.md) gives the documented ASUS Hybrid-mode commands, display checks, VFIO reservation, and rollback steps.
The D3OS guest boots with the physical GPU and audio function present and completes its native CPU self-test with four vCPUs.
The native probe reads AD103 register identification, a 16 GiB BAR1 aperture, and 16,376 MiB of usable VRAM.
GPU firmware startup, scalar kernels, tensor kernels, AlexNet inference, and Qwen numerical verification pass on this physical GPU.
Read the [native inference commands](nvidia-vfio.howto.md#8-run-native-gpu-inference) for repeatable hardware checks.
The Ada kernel build uses `--hardware-sm sm_89`.
