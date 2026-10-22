# Native inference architecture and hardware handoff

D3OS owns the compute API, tensor operations, model execution, and application integration in this implementation.
The CPU backend is a real userspace implementation with `no_std` and `alloc`.
Host verification builds the same libraries for Linux.

The hardware-free phase delivers pretrained AlexNet and Qwen3.5-0.8B text inference inside D3OS.
It also provides CPU execution of compiled Rust PTX kernels.
The native NVIDIA hardware phase now runs AlexNet on the passed-through RTX 4090 Laptop GPU.
Firmware boot, GPU page tables, native kernel loading, and tensor execution are operational.
Qwen also passes all 24 layer checks and vocabulary-logit comparisons on the physical GPU.
Its native generation command replies `4` to the default arithmetic prompt and stops at EOS.
GPU rendering remains unfinished.
The [DOOM rendering assessment](nvidia-rendering-design.md) describes a native compute path and the remaining transfer requirements.

## Source layout

- `os/library/compute`: tensor storage, the synchronous backend contract, CPU operators, and the native NVIDIA backend.
- `os/library/inference`: checked tensor archives, AlexNet evaluation, Qwen text generation, tokenization, and numerical verification.
- `os/application/infer`: D3OS file access, shell commands, and serial reports.
- `os/library/inference/examples/infer-host.rs`: host execution of the same model implementation.
- `tools/gpu-kernels`: Rust device kernels compiled to PTX.
- `tools/inference`: asset preparation, CPU PTX tests, and QEMU integration tests.
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
BF16 values expand to FP32 before backend upload.
The rest of the record layout stays the same.
AlexNet archives from version 1 remain supported.

The format stores explicit data and dimensions rather than serialized Rust objects.
AlexNet separately validates every parameter name and shape before allocating its weight buffers.
The exporter writes a SHA-256 manifest alongside generated archives.
The archive itself does not authenticate its contents.

## What the PTX tests establish

The Rust compiler generates the device code used by the numerical tests.
GPUOcelot executes that PTX on the host CPU.
The harness compares vector addition, ReLU, and matrix multiplication with independent scalar calculations.
Guard values detect writes immediately before or after the output allocation.

The interpreter accepts an older PTX instruction set.
The test build targets `sm_35`, while a separate compile targets Ampere `sm_86`.
Both builds use the same Rust kernel source and launch ABI.
Compilation of the Ampere output does not establish that it executes correctly on hardware.

The tests cover scalar floating-point kernels and launch bounds.
They do not establish Tensor Core support or inference performance on an NVIDIA GPU.
They also do not model a PCIe NVIDIA device that D3OS can discover.

## Verified software results

Local verification on October 22, 2026 established these results:

- CPU operator and archive tests passed on the development host.
- The PTX suite passed 22 cases across three compiled Rust kernels.
- The same kernels compiled for Ampere and Ada.
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
The D3OS kernel verifies handle arithmetic and launch dimensions before dispatch.
GPU tensor kernels use native Ada SASS compiled from Rust device code.

AlexNet passes every layer comparison on the physical GPU with the original reference tolerances.
Its matrix operations use three TF32 tensor products and compensated FP32 accumulation.
An earlier running accumulation missed one third-layer value by about `2.24e-4`.
Compensated accumulation corrects that mismatch without changing the reference tolerance.
The hardware operator suite also covers partial tiles, long reductions, batched convolution, and buffer reuse.

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
It prepares and validates every dispatch before the first cache mutation.
The syscall checks every modified buffer, including convolution histories and recurrent state.
The hardware suite compares outputs and state across multiple attention and gated-delta steps.
The pretrained model then passes every original reference comparison at 29 token positions.
The transfer path uses a persistent, driver-owned 1 MiB staging allocation in coherent guest RAM.
GPU copy kernels move bounded chunks between that allocation and process-owned VRAM.
GPU clearing preserves buffer isolation when storage is fresh or recycled.
The complete AlexNet and Qwen reference commands now take about one second and 11.5 seconds on the development GPU.
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
