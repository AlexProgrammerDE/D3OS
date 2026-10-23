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
Qwen reports three independent sessions with a 22-token prompt and 64 decode steps.
The decode loop continues after EOS to measure a fixed amount of work.
Its checksum describes benchmark work, not a conversational answer.

D3OS timers have millisecond resolution. The kernel benchmark amortizes this resolution across 100 launches.
Its times include CPU submission and synchronization.
The Linux kernel benchmark uses CUDA events across 30 queued launches, which excludes most CPU submission overhead.
D3OS also runs a separate instrumented pass with GPU timestamps before scheduling and at QMD completion.
The boot check compares these timestamps with a kernel that runs for 10 ms on the GPU.
`device_mean_us` includes device scheduling and completion work, while `instrumented_mean_us` also includes CPU submission and waiting.
The first `mean_us` pass runs without timestamp instrumentation.
These measurements separate end-to-end overhead from device throughput. They are not interchangeable timers.

## Measured progress

Measurements on 2026-10-22 and 2026-10-23:

| Workload | Initial resident baseline | After native optimizations |
| --- | ---: | ---: |
| Qwen warm generation | 7.04 to 7.36 tokens/s | 125.00 to 149.18 tokens/s |
| Qwen warm prompt processing | 7.76 to 7.81 tokens/s | 536.59 to 594.59 tokens/s |
| AlexNet forward pass | 10.0 ms | 1.45 to 1.55 ms |
| DOOM uncapped `demo1` | 536.84 to 676.89 FPS with word storage | 1431.58 to 1439.86 FPS with resident byte textures |

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
The other model-comparison limitations below still apply.

The Linux llama.cpp reference uses build `b11827`, full GPU offload, and three repetitions.
The earlier unrestricted runs report about 250 tokens/s with BF16 weights and 156 tokens/s with FP32 weights.
The FP32 file is a conversion of the same BF16 GGUF file without quantization.
The Linux reference retains its default FP16 attention cache. D3OS uses FP32 state.
Both benchmarks request 22 prompt tokens and 64 generated tokens, but they use different token sequences.
The Linux hybrid-model benchmark can round prompt work to its required chunk size.
Its reported prompt throughput therefore cannot serve as an exact 22-token latency comparison.
These results establish the remaining gap, rather than an exact application speed ratio.

The GGUF reference comes from `unsloth/Qwen3.5-0.8B-GGUF` revision `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`.
The D3OS archive retains the pinned Qwen revision and original Transformers fixture described in [the Qwen guide](qwen.howto.md).

## Changes that reduce overhead

The projection kernel selects a warp reduction for a single input row.
Each warp reads contiguous weights. Wide matrices reuse each input across eight outputs.
Narrow matrices use one output per warp to occupy more GPU multiprocessors.
This avoids the 15 unused rows in a padded tensor-core tile.
Larger matrix operations retain the TF32 tensor-core path.
The FP16 and BF16 tensor-core boot checks also remain active.

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

The loader now allocates each entrypoint's static shared memory plus the reserved 1 KiB CUDA prefix.
It rounds the allocation to 128 bytes and rejects requests above the configured 32 KiB partition.
The kernel build checks shared loads, shared stores, and barriers in the assembled normalization and recurrent kernels.

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
The final 600-frame test reports 6,474 resets while matching C-rendered pixels, including low detail and fuzz feedback.
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
Attention preparation and recurrent updates still launch once per token row.
In the 22-token prefill profile, packed projections account for about 41% of dispatch cycles.
Attention preparation accounts for about 21%, and recurrent updates about 19%.
The next improvements must address these kernels and submission costs.
The Linux prompt result still uses different tokens, archive provenance, and cache precision; a fully matched model benchmark remains necessary.
The CPU token-selection work now combines only 32 GPU candidates.
Each change must preserve the existing model reference or define and measure a separate precision policy.

The large projection kernels now match Linux device throughput for the measured shapes.
This does not establish model parity: Qwen still generates about 125 to 149 tokens/s, compared with about 265 tokens/s in the pinned Linux BF16 reference.
The remaining work must reduce complete model latency, rather than optimize already comparable streaming bandwidth.

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
