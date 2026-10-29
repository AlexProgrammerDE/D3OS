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
cargo test --locked -p d3os-ai --features cpu,training,native
cargo run --locked -p d3os-ai --features cpu --example infer-host --release -- \
  qwen-verify target/qwen
```

The default fixture compares every vocabulary logit and every layer value at 29 token positions.
Each value must satisfy `abs(actual - reference) <= 0.002 + 0.0005 * abs(reference)`.
The verifier also compares the greedy token decision at every position.
All 105 tokenizer cases must produce exactly the upstream token IDs.

Generate a reply with a limit of 32 new tokens:

```bash
cargo run --locked -p d3os-ai --features cpu --example infer-host --release -- \
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
Generation streams text as tokens arrive and reports a cumulative rate every 256 tokens.
The final `QWEN_STREAM` line separates session setup and prompt processing from the interval after the first token.
Its rate divides subsequent generated tokens by that interval, including terminal output and synchronization costs.
It excludes model loading. Normal generation still stops at EOS or the requested token limit.
The native API exposes `Qwen::generate_stream()` and an incremental tokenizer decoder for other applications.
The decoder retains incomplete UTF-8 sequences between tokens.

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

Add `d3os-ai` as a path dependency, with its `native` feature for the D3OS GPU.
Use its `cpu` feature for the explicit Flex CPU device.
This function accepts asset bytes from any storage interface:

```rust
extern crate alloc;

use alloc::string::String;
use d3os_ai::{burn::tensor::Device, qwen::{self, Qwen, tokenizer::Tokenizer}};

fn reply(
    device: &Device,
    weights: &[u8],
    vocabulary: &[u8],
    prompt: &str,
) -> qwen::Result<String> {
    let tokenizer = Tokenizer::parse(vocabulary)?;
    let model = Qwen::load(device, weights)?;
    let input = tokenizer.encode_chat(prompt)?;
    let output = model.generate(&input, 32)?;
    tokenizer.decode(&output.tokens, true)
}
```

Open the GPU with `d3os_ai::native_device()`.
Keep the loaded `Qwen` instance to reuse its weights between requests.
Each `model.session(capacity)` creates independent convolution, recurrent, and KV caches as Burn tensors.
`session.prefill(&tokens)` processes the prompt and returns its final logits.
`session.step(token)` appends one token and returns the next-token logits.
These methods support custom sampling and multi-turn histories supplied as token IDs.
`encode_chat` supplies one user turn with the official non-thinking assistant prefix.

The public generator uses greedy decoding and reports EOS or token-limit termination.
A session rejects tokens outside its vocabulary and sequences beyond its allocated capacity.
A backend or observer error invalidates that session; create a new session before further execution.

## Implementation boundaries

The default 0.8B profile includes all 24 text layers: 18 gated-delta layers and six grouped attention layers.
It supports the depthwise causal convolution, normalized delta rule, gated RMSNorm, partial RoPE, SwiGLU, and tied output embeddings.
The backend performs all numerical operations and owns cache tensors.
The model returns resident Burn logits. Greedy selection reads a finite-value flag and the selected token ID.
Full tensor reads occur only when the caller requests them or runs reference checks.
Burn modules own the model parameters, including packed integer tensors for BF16 and GGUF matrices.
`Qwen::load_weights` accepts one encoded `WeightData` record at a time, including borrowed or owned bytes.
`qwen::gguf::Gguf` supplies the streaming GGUF reader through the same loader.
The previous inference crate and generic model execution API are removed.
The same model code and tokenizer work with either backend.
Flex import expands BF16 matrix constants to FP32 once. Native import retains their original BF16 encoding.
Packed model records need a template with the same physical encoding. Burn device moves preserve the existing encoding.

The native GGUF loader also supports grouped key heads, untied output weights, and the pinned 9B and 27B profiles.
Image input, video input, auxiliary prediction heads, quantized CPU kernels, and stochastic sampling are outside this implementation.
The checkpoint advertises 262,144 positions; this work verifies short prompts and cached continuation, not full-length context behavior or performance.

For CI verification, dispatch the `Native inference` workflow with its `qwen` input enabled.
That job prepares the pinned model, verifies the host implementation, builds D3OS, and runs the guest checks with TCG.

## Run the quantized 9B and 27B models

These workloads use the same Burn model, sessions, and generation API as 0.8B.
The native profiles are Qwen3.5-9B Q8_0 and Qwen3.6-27B Q3_K_M.
Both fit the passed-through RTX 4090 Laptop GPU with a 2048-token FP32 cache.
Their token tables, merge tables, and token types match the 0.8B tokenizer exactly.

First, prepare the pinned models and Linux CUDA speed references:

```bash
python3 tools/nvidia/prepare-large-qwen.py 9b 27b
taskset -c 8,10,12,14 python3 tools/nvidia/test-large-qwen.py 9b 27b \
  --memory-mib 15360 --timeout 600
```

Prepare the independent Linux CPU FP32 reference for 9B:

```bash
model=9b
weights=target/nvidia/large-qwen/Qwen3.5-9B-Q8_0.gguf
taskset -c 24,26,28,30 target/qwen-venv/bin/python \
  tools/nvidia/prepare-qwen-reference.py "$weights" \
  "target/nvidia/large-qwen/$model-config.json" \
  "target/nvidia/large-qwen/$model-linux-results.json" \
  "target/nvidia/burn/large-qwen/$model-fp32"
python3 tools/nvidia/prepare-qwen-shards.py "$weights" \
  "target/nvidia/burn/large-qwen/$model" \
  --linux-reference "target/nvidia/burn/large-qwen/$model-fp32/logits.bin"
```

For 27B, set `model=27b` and `weights=target/nvidia/large-qwen/Qwen3.6-27B-Q3_K_M.gguf`, then repeat those commands.
The reference exporter uses the pinned Qwen Python environment from the earlier preparation steps.
It loads one Transformers layer at a time and keeps inference arithmetic in FP32.
The complete 27B model does not need to fit in host RAM as FP32 weights.

Build D3OS, then verify the model on the physical GPU:

```bash
cargo make --no-workspace --profile production image
taskset -c 8,10,12,14 python3 tools/nvidia/run-qemu.py \
  --firmware target/nvidia/firmware --kernels target/nvidia/kernels \
  --memory-mib 15360 --boot-check --timeout 600 \
  --inference gpu-qwen-gguf-verify \
  --qwen-gguf-parts "target/nvidia/burn/large-qwen/$model"
```

The check covers 165 positions, every layer, all logits, and sequential versus batched final caches.
It retains the original `2e-3` absolute and `5e-4` relative reference tolerances.
For timing and transfer counters, replace `gpu-qwen-gguf-verify` with `gpu-qwen-gguf-bench`.
That benchmark replays the Linux CUDA reference token IDs, including all 128 decode steps after the prompt.
The full-logit reference check runs after timing.

For interactive generation, start the same image with `--display gtk` and omit `--boot-check`.
Use `--inference gpu-qwen-gguf` to stage its tokenizer and model parts.
At the D3OS shell, run:

```text
infer gpu-qwen-gguf 128 What country is largest by area?
```

The host preparer verifies the source and part checksums before creating the firmware index.
It uses filesystem sharing when available and a bounded copy otherwise.
The guest checks part counts, lengths, read bounds, GGUF offsets, shapes, and weight encodings.
Its reported source checksum comes from the host index; the guest does not hash the full model again.
Each source tensor is released after its Burn upload.

GGUF reorders grouped value heads during export.
The reader restores input projections and state parameters to the model's head order.
Output projection weights retain their packed bytes, including quantization blocks that span multiple heads.
Burn reorders the small input activation on the GPU before that projection.

### Experimental Q8_1 prefill for 9B

The default `PrefillQuantization::Accurate` mode retains the existing FP32 reference policy.
The optional `Q8_1` mode quantizes activations for the qualified 9B Q8_0 FFN matrices during batched prefill.
Attention and single-token decode retain their accurate routes.
Weights retain their packed bytes and one resident GPU copy.

The separate `Q8_1WithAttention` mode also quantizes qualified 32-row attention projections.
It supports row-major Q8_0 matrices with shapes `[1024,4096]`, `[4096,4096]`, and `[8192,4096]`.
Smaller attention batches and single-token decode retain their accurate routes.

For a loaded Native model, select the mode explicitly:

```rust
use d3os_ai::qwen::PrefillQuantization;

let model = model.with_prefill_quantization(PrefillQuantization::Q8_1)?;
```

Select `PrefillQuantization::Q8_1WithAttention` to include those attention projections.

The method rejects unsupported devices and weight layouts.
Mode metadata belongs to the destination model.
Record loading preserves that mode, but weight records do not encode it.
A quantized model moved to CPU rejects execution until its mode changes to `Accurate`.

For a diagnostic benchmark, add `--qwen-prefill-q8` to the host command for `gpu-qwen-gguf-bench`.
The corresponding guest command is:

```text
infer gpu-qwen-gguf-bench device-timing q8-prefill
```

For attention quantization, use the host flag `--qwen-prefill-q8-attention` or the guest option `q8-attention-prefill` instead.
The two quantized options are mutually exclusive.

Timing records precede the full-logit reference check.
The check keeps its original tolerances and reports `INFERENCE_FAIL` for the measured Q8_1 precision differences.
That result does not indicate FP32 equivalence or whole-model qualification.
The mode remains experimental.
Read the [measured performance and accuracy comparison](nvidia-performance.md#experimental-q8_1-prefill-on-the-9b-model).
The [attention comparison](nvidia-performance.md#optional-q8_1-attention-prefill) records the additional speed and precision changes.

### Paired Q4_K measurements within one VM

The paired benchmark compares Q4_K implementations with one resident model.
Both model configurations share the original packed weight tensors.
A memory and transfer check rejects an unexpected weight allocation or copy.

The control uses the existing Q4_K conversion path.
The candidate converts bounded unsigned weight operands through `u8` before FP32 arithmetic.
Both paths retain FP32 activations, multiplication, FMA, and reduction.
The candidate is an explicit choice. The default remains unchanged.

The measured order is `ABBA BAAB`, where A is the control and B is the candidate.
Each configuration first receives one warmup pass.
Two instrumented passes follow the eight ordinary samples.
The instrumented passes do not contribute to the speed comparison.

Each paired phase ends with `device.sync()` inside its timer.
These times include completion of the final submitted GPU work.
Earlier separate-VM times use a different completion scope and remain separate evidence.

The benchmark runs the complete original logit reference for each configuration.
Each reference covers 32,033,280 values with the original absolute and relative tolerances.
The launcher also requires matching generated tokens, projection workloads, and transfer counts.
The projection profiles identify the actual selected Q4_K entry.

Before a run, compare the source, image, module, model, and reference identities with existing receipts.
Reuse a matching completed capture.
For a new source or measurement scope, retain its reason and comparison rule before the run.

Use the normal GGUF benchmark command with these additional flags:

```sh
--inference gpu-qwen-gguf-bench --qwen-device-timing --qwen-paired-q4
```

The paired command requires Accurate activation arithmetic.
It rejects either Q8_1 prefill flag.
Use a short capture directory because QMP uses a UNIX socket within that directory.

The guest emits `NATIVE_LARGE_QWEN_PAIRED_*` records and the separate `INFERENCE_BENCH_PAIR_OK` footer.
The launcher rejects incomplete schedules and mismatched references.
Ordinary benchmark commands retain their existing schedule and records.
