# Native D3OS cuTile kernels

This host compiler exports portable Tile IR for native inference and rendering kernels.
NVIDIA cuTile Rust produces Tile IR 13.4.
D3OS runs `tileiras` and `ptxas` inside its Linux execution personality during module loading.
The compiler targets the detected GPU, currently Ada `sm_89` on the validation machine.
The native Rust driver then loads and executes the compiled modules.
D3OS requires no Linux CUDA execution libraries.

## Build and check

Install libclang and Rust 1.98.1 on a Linux x86-64 host.
From the repository root, run:

```bash
rustup toolchain install 1.98.1 --profile minimal --component clippy,rustfmt
python3 tools/inference/test-cutile-kernels.py
```

The script downloads checksum-pinned NVIDIA compiler packages into `target/nvidia/cutile-toolkit`.
It runs linker tests, Clippy, full compilation, and SASS audits without a GPU.
Run `python3 tools/nvidia/prepare-kernels.py` to rebuild only the device bundle.
An existing CUDA 13.4 toolkit can be supplied with `--toolkit`.

The output directory contains three binary families:

| Family | Contents |
| --- | --- |
| `vector-add` | Native boot arithmetic check |
| `tensor-matmul` | FP16 and BF16 Tensor Core boot checks |
| `inference` | All 59 public kernel families, operation specializations, and timing and scratch diagnostics |

Each family includes textual IR, binary Tile IR, and entry metadata.
The manifest records input hashes and the guest compilation requirement.
With `--audit`, host assembly and disassembly stay under `host-audit/`.
Those machine-code files never enter the guest boot image.
Compilation does not establish GPU execution or numerical correctness.
Follow [the VFIO guide](../../docs/nvidia-vfio.howto.md) for hardware checks.

## Add an operator

Write device code with cuTile operations inside a source fragment under `src`.
Add the fragment to `build.rs` so the compiler captures it in one cuTile module.
The `.rs.in` templates specialize concrete shapes and memory types before capture.
This avoids unresolved generic tuple and shape types in the current frontend.

Register public operations in `d3os_nvidia::compute::Kernel`.
Define buffer extents and launch dimensions in `Kernel::work`.
Use `Kernel::entry` for specialized device entries behind the same public operation.
Add specialized entries to the host compiler's inference list.
Keep pointers as 64-bit device addresses and scalar dimensions as `u32`.
Allocate buffers through the D3OS API.
The build rejects GPU relocations and runtime constant-bank dependencies that the native loader cannot satisfy.

The compiler metadata records each entry's required thread count.
The driver checks that contract again before launch.
It binds compiler constant banks and allocates shared memory and scratch from binary metadata.
Use dependency tokens when a read depends on an earlier global-memory write.
The recurrent batch path instead keeps history and state in tiles until the sequence completes.

BF16 matrix projections preserve two-dimensional pointer tiles through memory lowering.
Their masked loads select padding explicitly because the pinned frontend expands scalar padding with rank one.
FP32 matrix products and wide vector projections retain flattened loads, which measured faster for those paths.
BF16 matrix projections split FP32 activations into a BF16 high part and a BF16 residual, with compensated FP32 accumulation.
The binary audit requires native BF16 Tensor Core instructions for both regular and split projections.
Regular BF16 projections use 64-element reductions. Split BF16 projections use 16 elements, and FP32 products retain 8 elements.
Small even-width BF16 projections use `[2, 4, 32]` tiles to reduce shared storage for partial sums.
Their gated specialization combines two projections and SiLU multiplication without a separate activation buffer.
Odd-width inputs retain scalar BF16 loads, and vocabulary projections retain their wider tile.
Separate entries keep fallback shared-memory reservations out of the even-width launches.

Add numerical cases to the GPU operator suite when the operation has complex behavior.
Preserve partial-tile checks, buffer guards, and existing model tolerances.
Measure the complete workload as well as individual kernels.
The Linux kernel benchmark selects the same cuTile specialization and launch shape as D3OS.
