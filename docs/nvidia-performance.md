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
These measurements separate end-to-end overhead from device throughput. They are not interchangeable timers.

## Measured progress

Measurements on 2026-10-22:

| Workload | Initial resident baseline | After native optimizations |
| --- | ---: | ---: |
| Qwen warm generation | 7.04 to 7.36 tokens/s | 26.89 to 27.11 tokens/s |
| Qwen warm prompt processing | 7.76 to 7.81 tokens/s | 30.56 tokens/s |
| AlexNet forward pass | 10.0 ms | 2.5 ms |
| DOOM uncapped `demo1` | No matched initial measurement | 673.23 to 684.33 FPS |

The AlexNet result excludes input upload and final download.
The DOOM result includes native GPU drawing, palette conversion, scaling, and framebuffer copies.
Its scene size is 320 by 200, with presentation at 640 by 400.
The regular game loop has separate timing behavior. Its HUD rate does not measure uncapped throughput.

The Linux llama.cpp reference uses build `b11827`, full GPU offload, and three repetitions.
It reports about 250 tokens/s with BF16 weights and 156 tokens/s with FP32 weights.
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
Each warp reads contiguous weights and reuses each input across eight outputs.
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

The DOOM rasterizer builds ordered column and row lists for each command batch.
Each pixel merges its two relevant lists instead of scanning every command.
The merge preserves draw order, including overlap between columns and spans.
The fuzz pass retains sequential feedback semantics.
The original scan kernel remains the independent GPU reference in the operator checks.

## Remaining work

Qwen still submits synchronous operations separately and processes the prompt one token at a time.
It expands BF16 weights into FP32 VRAM storage and downloads the full vocabulary logits for CPU token selection.
A faster path needs batched submission, fused operations, native reduced-precision weights, and batched prompt processing.
Each change must preserve the existing model reference or define and measure a separate precision policy.

The large vocabulary projection reaches about 560 GB/s under Linux with either our kernel or cuBLAS FP32.
D3OS currently reports about 300 GB/s for that shape, including submission overhead.
The native SM clock probe reports about 2309 MHz. It does not measure the memory clock.
The remaining bandwidth difference needs device timestamps and memory-clock diagnostics before a cause can be assigned.

DOOM still transfers indexed frames for CPU HUD composition and copies RGB output into the console framebuffer.
Direct GPU presentation remains separate work.
The Linux DOOM reference uses the same pinned engine and WAD with a headless framebuffer copy.
That reference reports 1586.59 FPS for the same 7117-tick demo.
It measures a software-rendering baseline without display-server or physical-display costs.
It does not represent an OpenGL port or a Linux GPU renderer.

## Implementation sources

The MMU encoding follows NVIDIA's published [MMU-v2 fields](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/570.144/src/common/inc/swref/published/pascal/gp100/dev_mmu.h).
The mapping policy also follows the native [tinygrad NVIDIA page-table implementation](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/support/nv/nvdev.py).
The Linux comparison uses [llama.cpp CUDA kernels](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda).
NVIDIA's [matrix performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) explains the distinction between matrix tiling and memory-bound vector work.
