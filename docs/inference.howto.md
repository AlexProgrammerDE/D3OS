# Develop native Rust inference

This guide is for developers who write compute applications for D3OS.
For native text generation, read [the Qwen3.5-0.8B guide](qwen.howto.md).
The CPU backend runs inside D3OS and on a development host.
The cuTile compiler checks run without an NVIDIA GPU or host driver.
They require the pinned host compiler tools and development headers.

The current model demonstration uses pretrained torchvision AlexNet in evaluation mode.
It accepts one normalized RGB image with shape `[1, 3, 224, 224]`.
Read [the architecture and hardware handoff](inference-architecture.md) for the implementation boundaries.
The native NVIDIA backend also runs AlexNet on the RTX 4090 Laptop GPU with tensor cores.
For hardware setup and repeatable GPU checks, follow [the NVIDIA VFIO guide](nvidia-vfio.howto.md#8-run-native-gpu-inference).

## Verify the CPU backend

From the repository root, run:

```bash
cargo test --locked -p d3os-ai --features cpu,training,native
cargo clippy --locked -p d3os-ai --features cpu,training,native --all-targets -- -D warnings
cargo run --locked -p d3os-ai --features cpu --example infer-host --release -- self-test
```

The tests cover convolution, matrix multiplication, pooling, activation, softmax, archive validation, and a small convolutional network.
The host example uses the same libraries as the D3OS application.

## Check cuTile kernels without a GPU

On Linux x86-64, install libclang and the host Rust toolchain:

```bash
sudo apt-get install libclang-dev
rustup toolchain install 1.98.1 --profile minimal --component clippy,rustfmt
python3 tools/inference/test-cutile-kernels.py
```

The script downloads checksum-pinned NVIDIA compiler packages into `target/nvidia/cutile-toolkit`.
It runs linker tests, Clippy, compilation, and SASS audits.
It does not install a host GPU driver.
The expected final marker is:

```text
CUTILE_VALIDATION_OK hardware_validated=false
```

The build writes Tile IR, native binaries, disassembly, and resource metadata into `target/nvidia/kernels`.
It covers all 41 inference and rendering operators, plus boot diagnostics.
These checks do not execute GPU instructions or establish numerical correctness.
Run the [VFIO hardware checks](nvidia-vfio.howto.md#8-run-native-gpu-inference) for those guarantees.
The previous GPUOcelot PTX interpreter cannot execute the cuTile binaries and is removed.

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
cargo run --locked -p d3os-ai --features cpu --example infer-host --release -- \
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

## Write an AI application

Add `os/library/ai` as a path dependency with the `native` feature.
Use public Burn tensors and modules:

```rust
use d3os_ai::burn::tensor::{Device, Tensor};

fn multiply(device: &Device) -> Tensor<2> {
    let a = Tensor::<2>::from_floats([[1., 2., 3.], [4., 5., 6.]], device);
    let b = Tensor::<2>::from_floats([[1.], [0.], [-1.]], device);
    a.matmul(b)
}
```

Select `d3os_ai::native_device()?` for GPU execution.
With the `cpu` feature, select `Device::flex()` for explicit CPU execution.
Native operations never fall back to the CPU.
Call `device.sync()` or a checked tensor read to observe deferred execution errors.

Use `d3os_ai::alexnet::AlexNet` for the pretrained classifier and `d3os_ai::qwen::Qwen` for text generation.
Qwen creates native command scopes automatically and reports failures before returning model outputs.
Burn modules support device moves and parameter records. Model configuration must match the record.
BF16 and GGUF projection weights remain packed in GPU memory.
See [the Qwen guide](qwen.howto.md) for sessions, tokenization, and streaming replies.

The compute crate remains the low-level driver interface and an independent operator reference for diagnostics.
DOOM uses its native renderer without Burn.
The current compute protocol is version 18, with 59 kernel families and 174 compiled entries.
Rebuild the OS, applications, and kernel bundle together when that protocol changes.

Write device kernels in the cuTile module assembled by `tools/gpu-kernels/build.rs`.
The host compiler uses 64-bit device pointers and 32-bit dimensions for the native launch ABI.
Keep GPU operations inside the supported cuTile DSL.
D3OS validates the compiled thread count, register budget, shared-memory partition, and buffer extents.
The hardware driver must validate those dimensions and keep all referenced buffers alive until execution completes.

## Run CI verification

The `Native inference` workflow runs CPU tests, Clippy, and cuTile compiler checks on pushes and pull requests.
For pretrained verification, start that workflow manually with its `pretrained` input enabled.
This option also builds D3OS and runs both QEMU tests with software CPU emulation.
The workflow uploads the manifest and guest logs.
