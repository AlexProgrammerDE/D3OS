# Run Qwen3.5-0.8B in D3OS

D3OS runs the text decoder of [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) through its native Rust inference API.
This is the current official sub-billion Qwen model selected on October 22, 2026.
The newer Qwen3.8 releases are larger models.

The model runner, tokenizer, caches, and compute backends use Rust with `no_std` and `alloc`.
Python and Transformers run only on the development host to prepare assets and reference outputs.
The native NVIDIA backend passes the complete numerical fixture on the RTX 4090 Laptop GPU.
Read [the architecture guide](inference-architecture.md) for the driver and API boundaries.

## Prepare the model

Use a Linux development host with at least 16 GiB of RAM for the VM tests.
Allow approximately 12 GiB of free disk space for dependencies, downloads, model assets, and boot images.
The upstream checkpoint is 1.75 GB; the exported text weights are 1.50 GB.
The CPU backend expands BF16 weights to approximately 3 GB of FP32 storage.
The NVIDIA backend keeps matrix weights packed in BF16, with approximately 1.5 GB of weight data in VRAM.
Activations, normalization parameters, and recurrent state retain FP32 values.
Allocation padding and temporary buffers add to the VRAM requirement.

From the repository root, run:

```bash
python3 -m venv target/qwen-venv
target/qwen-venv/bin/python -m pip install -r tools/inference/requirements-qwen.txt
target/qwen-venv/bin/python tools/inference/prepare-qwen.py
```

The exporter pins the official checkpoint revision and checks SHA-256 digests for weights, configuration, and tokenizer assets.
It exports 320 text tensors and excludes the vision encoder and auxiliary prediction weights.
The reference implementation runs the official Transformers decoder with FP32 arithmetic and the original BF16 parameter values.

The output directory, `target/qwen`, contains:

- `qwen3.5-0.8b.d3t`: typed native tensor archive with model configuration and text weights.
- `qwen3.5.bpe`: native tokenizer vocabulary, merge ranks, and added tokens.
- `qwen-fixture.d3t`: input token IDs, residual streams from all 24 layers, and vocabulary logits.
- `tokenizer-fixture.bin`: independently encoded Unicode and control-token examples.
- `manifest.json`: source revision, checksums, reference versions, prompt, and continuation.

The default fixture contains 23 prompt tokens and six cached continuation positions.
The numerical test deliberately continues through EOS tokens to exercise cache updates at all six positions.
Normal generation stops at the first EOS token.

To select another reference prompt, run:

```bash
target/qwen-venv/bin/python tools/inference/prepare-qwen.py \
  --prompt 'Write one short sentence about the moon.' --steps 8
```

Use `--output DIRECTORY` to select an asset directory.
Use `--cache-dir DIRECTORY` to place the upstream downloads on another filesystem.
For deployment assets alone, `python3 tools/inference/prepare-qwen.py --skip-reference` uses only the Python standard library.
Reference verification requires the fixture files, so omit that flag for the steps below.

## Verify and generate on the host

Run the numerical and tokenizer checks:

```bash
cargo test --locked -p d3os-compute -p d3os-inference
cargo run --locked -p d3os-inference --example infer-host --release -- \
  qwen-verify target/qwen
```

The default fixture compares every vocabulary logit and every layer value at 29 token positions.
Each value must satisfy `abs(actual - reference) <= 0.002 + 0.0005 * abs(reference)`.
The verifier also compares the greedy token decision at every position.
All 105 tokenizer cases must produce exactly the upstream token IDs.

Generate a reply with a limit of 32 new tokens:

```bash
cargo run --locked -p d3os-inference --example infer-host --release -- \
  qwen target/qwen 32 'What is 2 + 2? Answer briefly.'
```

The verified checkpoint replies `4` and stops at EOS.
The limit counts generated tokens and excludes the prompt.

## Run inside D3OS

Install the [OS build requirements](../README.md#requirements), QEMU, OVMF, mtools, and the util-linux partition tools.
On Fedora, the extra image tools are available with `sudo dnf install mtools util-linux`.
On Ubuntu, use `sudo apt-get install mtools fdisk`.

Build D3OS and run the model checks:

```bash
cargo make --no-workspace --profile production image
python3 tools/inference/test-qemu.py --qwen-verify --timeout 900
python3 tools/inference/test-qemu.py --qwen --timeout 900
```

The harness builds a separate GPT/EFI image with the model assets and gives the VM 12 GiB of RAM.
It uses mtools to reserve sufficient FAT metadata space for the large initrd.
It does not copy model files into the normal repository boot image.

The VM uses two vCPUs by default.
Use `--cpus 4` to test four vCPUs and `--repeat 3` to repeat a command within the same VM.
The kernel masks interrupts across context transfers to prevent the earlier `local APIC not initialized` panic.
See [the context-switch regression guide](context-switch.howto.md) for the cause and debugger checks.

The complete numerical trace also passed with software CPU emulation, taking approximately six minutes on the development host.
To select this mode, add `--accel tcg`:

```bash
python3 tools/inference/test-qemu.py --qwen-verify --accel tcg --timeout 900
```

Use `--qwen-assets DIRECTORY` for assets prepared outside `target/qwen`.
The logs are `serial-qwen-verify-ACCEL.log`, `serial-qwen-ACCEL.log`, and `qemu.log` under `target/inference-qemu-qwen`.
`ACCEL` is `kvm` or `tcg`.
Both Qwen commands share the same image directory to limit disk use.
Run them sequentially.

The staged image includes these native shell commands:

```text
infer qwen
infer qwen 32 Write one short sentence about the moon.
infer qwen-verify
```

`infer qwen` uses the arithmetic prompt and a limit of 32 new tokens.
The application reads assets from `/usr/share/inference` through normal file syscalls.
It reports generation and verification results to the terminal and serial logger.

To open the prepared image in an interactive VM after the tests, run:

```bash
cp /usr/share/OVMF/OVMF_VARS_4M.fd target/qwen/interactive-vars.fd
qemu-system-x86_64 -machine q35 -accel kvm -cpu host -smp 2 -m 12G \
  -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
  -drive if=pflash,format=raw,file=target/qwen/interactive-vars.fd \
  -drive format=raw,file=target/inference-qemu-qwen/d3os.img \
  -device virtio-vga -device virtio-rng-pci -nic model=rtl8139 \
  -serial file:target/qwen/interactive-serial.log
```

On Fedora, use `/usr/share/edk2/ovmf/OVMF_CODE.fd` and `/usr/share/edk2/ovmf/OVMF_VARS.fd` as the firmware pair.
Keep the code image and variables template from the same firmware layout.

## Run on the NVIDIA GPU

First, complete the [VFIO setup and native GPU checks](nvidia-vfio.howto.md).
Prepare the model assets with the default fixture from this guide.
Build the OS image and native kernels:

```bash
cargo make --no-workspace --profile production image
python3 tools/nvidia/prepare-kernels.py
```

Run the operator suite and full-model verification sequentially:

```bash
python3 tools/nvidia/run-qemu.py --firmware target/nvidia/firmware \
  --kernels target/nvidia/kernels --memory-mib 2048 --boot-check \
  --inference gpu-self-test
python3 tools/nvidia/run-qemu.py --firmware target/nvidia/firmware \
  --kernels target/nvidia/kernels --memory-mib 12288 --boot-check \
  --inference gpu-qwen-verify --timeout 1200
```

The operator suite compares outputs, KV caches, convolution histories, and recurrent state against the CPU backend.
It covers 14 grouped-attention steps and 14 gated-delta steps, including the actual Qwen head dimensions.
It also verifies that a rejected cache update leaves the cache contents and position unchanged.

The full-model check requires `INFERENCE_OK qwen-verify backend=nvidia positions=29 layers=24`.
This check passed on the physical RTX GPU with four vCPUs.
Every layer value and vocabulary logit must satisfy the same tolerance as the CPU test.
Every greedy token choice must also match the reference.
With coherent guest-RAM staging, the verified run took 11.5 seconds, including file access and model upload.
The earlier BAR1 transfer path took about 325 seconds for the same command.

Generate the default arithmetic reply with:

```bash
python3 tools/nvidia/run-qemu.py --firmware target/nvidia/firmware \
  --kernels target/nvidia/kernels --memory-mib 12288 --boot-check \
  --inference gpu-qwen --timeout 1200
```

This command generated `4` and stopped at EOS on the physical GPU.
The result matched the host reference and reported one generated token.
The staging-transfer run took about 10 seconds, including model loading and prompt processing.

The corresponding native shell commands are:

```text
infer gpu-qwen
infer gpu-qwen 32 Write one short sentence about the moon.
infer gpu-qwen-verify
```

The launcher retains the virtual display for the console and passes the NVIDIA device to D3OS for computation.
The backend uses tensor cores for all model projections.
Normalization, activation, rotary positions, both attention paths, and cache updates also run on the GPU.
The tokenizer, scheduling, reference comparisons, and greedy selection run on the CPU.
No transformer operation downloads activations to compute its result on the CPU.

## Call the native API

Add `d3os-inference` and `d3os-compute` as path dependencies in your D3OS application.
This function accepts asset bytes from any file or storage interface:

```rust
extern crate alloc;

use alloc::string::String;
use d3os_compute::transformer::TransformerBackend;
use d3os_inference::{qwen::Qwen, tokenizer::Tokenizer};

fn reply<B: TransformerBackend>(
    backend: &B,
    weights: &[u8],
    vocabulary: &[u8],
    prompt: &str,
) -> d3os_inference::Result<String> {
    let tokenizer = Tokenizer::parse(vocabulary)?;
    let model = Qwen::load(backend, weights)?;
    let input = tokenizer.encode_chat(prompt)?;
    let output = model.generate(backend, &input, 32)?;
    tokenizer.decode(&output.tokens, true)
}
```

Keep the loaded `Qwen` instance to reuse its weights between requests.
Each `model.session(backend, capacity)` creates independent convolution, recurrent, and KV caches.
`session.prefill(&tokens)` processes the prompt and returns its final logits.
`session.step(token)` appends one token and returns the next-token logits.
These methods support custom sampling and multi-turn histories supplied as token IDs.
`encode_chat` supplies one user turn with the official non-thinking assistant prefix.

The public generator uses greedy decoding and reports EOS or token-limit termination.
A session rejects tokens outside its vocabulary and sequences beyond its allocated capacity.
A backend or observer error invalidates that session; create a new session before further execution.

## Implementation boundaries

The implementation includes all 24 text layers: 18 gated-delta layers and six grouped attention layers.
It supports the depthwise causal convolution, normalized delta rule, gated RMSNorm, partial RoPE, SwiGLU, and tied output embeddings.
The backend performs all numerical operations and owns cache tensors.
The model runner downloads only final logits or explicit verification observations.
The NVIDIA backend implements both `TransformerBackend` and `Backend`.
To select it, enable the compute crate's `nvidia` feature and pass `d3os_compute::nvidia::Nvidia::new()?` as the backend.
The same model code and tokenizer work with either backend.

The supported checkpoint profile uses equal key and value head counts in its gated-delta layers.
Image input, video input, auxiliary prediction heads, quantized CPU kernels, and stochastic sampling are outside this implementation.
The checkpoint advertises 262,144 positions; this work verifies short prompts and cached continuation, not full-length context behavior or performance.

For CI verification, dispatch the `Native inference` workflow with its `qwen` input enabled.
That job prepares the pinned model, verifies the host implementation, builds D3OS, and runs the guest checks with TCG.
