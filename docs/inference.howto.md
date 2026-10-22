# Develop native Rust inference

This guide is for developers who write compute applications for D3OS.
The CPU backend runs inside D3OS and on a development host.
The PTX harness executes compiled Rust GPU kernels on a CPU.
Neither path requires an NVIDIA GPU or NVIDIA host libraries.

The current model demonstration uses pretrained torchvision AlexNet in evaluation mode.
It accepts one normalized RGB image with shape `[1, 3, 224, 224]`.
Read [the architecture and hardware handoff](inference-architecture.md) for the implementation boundaries.
For the inspected Fedora laptop, follow [the NVIDIA VFIO setup commands](nvidia-vfio.howto.md) when hardware development can start.

## Verify the CPU backend

From the repository root, run:

```bash
cargo test --locked -p d3os-compute -p d3os-inference
cargo clippy --locked -p d3os-compute -p d3os-inference --all-targets -- -D warnings
cargo run --locked -p d3os-inference --example infer-host --release -- self-test
```

The tests cover convolution, matrix multiplication, pooling, activation, softmax, archive validation, and a small convolutional network.
The host example uses the same libraries as the D3OS application.

## Execute Rust GPU kernels without a GPU

Install the GPU compiler components for the repository's pinned toolchain:

```bash
rustup target add nvptx64-nvidia-cuda --toolchain nightly-2026-02-01
rustup component add llvm-tools llvm-bitcode-linker --toolchain nightly-2026-02-01
python3 tools/inference/test-gpu-kernels.py
```

On Linux x86-64, the script downloads a checksum-pinned [GPUOcelot interpreter](https://github.com/tinygrad/gpuocelot).
It stores this development dependency in `target/gpu-sim`.
The upstream binary requests an executable stack.
The script removes that request from the local copy before loading it, so the host keeps its stack protection.

On other platforms, build GPUOcelot from its source.
Then supply the library path:

```bash
python3 tools/inference/test-gpu-kernels.py --library /path/to/libgpuocelot.dylib
```

The suite executes 22 numerical cases across vector addition, ReLU, and matrix multiplication.
It includes partial thread blocks and guards around output buffers.
The final output is:

```text
GPU_PTX_SUITE_OK kernels=3 cases=22 hardware_validated=false
```

The interpreter requires PTX 3.2, so the simulated build targets `sm_35`.
The script also compiles the same Rust source for `sm_86` and preserves both outputs.
To compile an Ada artifact, add `--hardware-sm sm_89` to the command.
It does not rewrite generated PTX instructions.
The hardware artifacts in `target/gpu-sim/sm_86` or `target/gpu-sim/sm_89` still require assembly and hardware verification.
These tests do not cover Tensor Cores, PCIe, firmware, GPU page tables, or hardware scheduling.

## Prepare pretrained AlexNet

Create a Python environment with NumPy and Pillow:

```bash
python3 -m venv target/inference-venv
target/inference-venv/bin/python -m pip install numpy pillow
target/inference-venv/bin/python tools/inference/prepare-alexnet.py
```

The exporter downloads the official 233 MiB weights and the PyTorch dog photograph.
It verifies both downloads against fixed SHA-256 digests.
Its restricted checkpoint reader reconstructs float tensors without a PyTorch installation.
An independent NumPy implementation calculates reference outputs for five convolutional layers and three fully connected layers.

The exporter writes these files to `target/inference`:

- `alexnet.d3t`: pretrained weights in the native tensor format.
- `dog.d3t`: the normalized image and eight reference outputs.
- `manifest.json`: source checksums, library versions, output checksums, and reference predictions.

To prepare another photograph, pass its path:

```bash
target/inference-venv/bin/python tools/inference/prepare-alexnet.py --image /path/to/photo.jpg
```

The new fixture still uses the filename `dog.d3t`.
The manifest records the actual image checksum.

Verify the model on the host:

```bash
cargo run --locked -p d3os-inference --example infer-host --release -- \
  alexnet target/inference/alexnet.d3t target/inference/dog.d3t
```

Every layer must satisfy `abs(actual - reference) <= 0.0002 + 0.0002 * abs(reference)`.
Nonfinite values fail verification.
The official dog photograph produces class 258 as its highest-probability prediction, with probability approximately 0.72448.

## Verify inference inside D3OS

Install the OS build requirements from [the README](../README.md#requirements).
Install QEMU and OVMF on the development host.
Then build a release image:

```bash
cargo make --no-workspace --profile production image
```

Run the small-network test:

```bash
python3 tools/inference/test-qemu.py
```

After preparing the pretrained model, run the full verification:

```bash
python3 tools/inference/test-qemu.py --alexnet
```

The harness creates a separate boot image under `target`.
It starts D3OS, enters `infer` or `infer alexnet` through the normal shell, and reads the application's serial log.
The full-model VM receives 4 GiB of RAM.
The original boot image and boot configuration stay available for normal development.

If KVM is unavailable, the harness selects software CPU emulation.
To select this path explicitly, run:

```bash
python3 tools/inference/test-qemu.py --accel tcg --alexnet --timeout 300
```

If OVMF is not in a standard location, supply `--bios /path/to/firmware.fd`.
The scripts return a nonzero status on verification errors, guest crashes, and timeouts.
Read `serial.log` and `qemu.log` in the reported test directory for diagnostics.

## Use the normal D3OS terminal

The default OS image includes `infer`.
Enter `infer` to run the small-network test.

To include the pretrained demonstration in the normal image, copy the generated assets:

```bash
mkdir -p loader/initrd/usr/share/inference
cp target/inference/alexnet.d3t target/inference/dog.d3t loader/initrd/usr/share/inference/
QEMU_MEMORY='4G,slots=2,maxmem=8G' cargo make --no-workspace --profile production
```

In the D3OS shell, enter:

```text
infer alexnet
```

To use different archive paths, enter `infer alexnet MODEL FIXTURE`.
Generated weights and fixtures are excluded from Git.

## Write a compute application

Add a dependency on `os/library/compute` to a D3OS application.
This example multiplies two matrices through the public backend interface:

```rust
extern crate alloc;

use alloc::vec;
use d3os_compute::{Backend, Tensor, cpu::Cpu};

fn multiply() -> d3os_compute::Result<Tensor<Cpu>> {
    let backend = Cpu;
    let a = Tensor::from_vec(&backend, &[2, 3], vec![1., 2., 3., 4., 5., 6.])?;
    let b = Tensor::from_vec(&backend, &[3, 1], vec![1., 0., -1.])?;
    backend.matmul(&a, &b)
}
```

`Tensor` uses contiguous row-major storage.
Convolution uses NCHW input and OIHW weights.
Linear layers use weights with shape `[output_features, input_features]`.
All current backend operations complete before they return.

For model execution, add `os/library/inference`.
Load weights with `AlexNet::load`, supply a normalized input tensor, and call `AlexNet::forward`.
Use `|_, _| Ok(())` as the observer when reference-layer comparisons are unnecessary.
Apply `Backend::softmax` and `top_k` to the resulting logits.

Keep device kernels in a separate crate with the `nvptx64-nvidia-cuda` target.
The sample crate at `tools/gpu-kernels` defines its launch ABI with 64-bit device addresses and 32-bit dimensions.
The hardware driver must validate those dimensions and keep all referenced buffers alive until execution completes.

## Run CI verification

The `Native inference` workflow runs CPU tests, Clippy, and the PTX suite on pushes and pull requests.
For pretrained verification, start that workflow manually with its `pretrained` input enabled.
This option also builds D3OS and runs both QEMU tests with software CPU emulation.
The workflow uploads the manifest and guest logs.
