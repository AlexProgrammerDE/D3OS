# Run Qwen3.5-0.8B in D3OS

D3OS runs the text decoder of [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) through its native Rust inference API.
This is the current official sub-billion Qwen model selected on October 22, 2026.
The newer Qwen3.8 releases are larger models.

The model runner, tokenizer, caches, and CPU operators use Rust with `no_std` and `alloc`.
Python and Transformers run only on the development host to prepare assets and reference outputs.
NVIDIA execution still requires the hardware backend described in [the architecture guide](inference-architecture.md).

## Prepare the model

Use a Linux development host with at least 16 GiB of RAM for the VM tests.
Allow approximately 12 GiB of free disk space for dependencies, downloads, model assets, and boot images.
The upstream checkpoint is 1.75 GB; the exported text weights are 1.50 GB.
The CPU backend expands BF16 weights to approximately 3 GB of FP32 storage when it loads the model.

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
qemu-system-x86_64 -machine q35 -accel kvm -cpu host -smp 2 -m 12G \
  -bios /usr/share/OVMF/OVMF_CODE.fd \
  -drive format=raw,file=target/inference-qemu-qwen/d3os.img \
  -device virtio-vga -device virtio-rng-pci -nic model=rtl8139 \
  -serial file:target/qwen/interactive-serial.log
```

On Fedora, the firmware path is normally `/usr/share/edk2/ovmf/OVMF_CODE.fd`.
Select the path installed on your host.

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
A future NVIDIA backend must implement `TransformerBackend` in addition to `Backend`.

The supported checkpoint profile uses equal key and value head counts in its gated-delta layers.
Image input, video input, auxiliary prediction heads, quantized CPU kernels, and stochastic sampling are outside this implementation.
The checkpoint advertises 262,144 positions; this work verifies short prompts and cached continuation, not full-length context behavior or performance.

For CI verification, dispatch the `Native inference` workflow with its `qwen` input enabled.
That job prepares the pinned model, verifies the host implementation, builds D3OS, and runs the guest checks with TCG.
