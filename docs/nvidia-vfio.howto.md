# Prepare the NVIDIA debug machine

These commands prepare the inspected Fedora laptop for D3OS GPU development.
Run each stage separately and check its result before continuing.
The two reboot commands end your current session.
The Hybrid-mode and VFIO steps below were verified on this machine on October 22, 2026.

The hardware-free implementation already runs pretrained AlexNet inside D3OS on the CPU.
The native Rust NVIDIA driver boots FWSEC and GSP firmware on the physical GPU.
The driver also creates a compute channel and verifies GPU writes through its submission queue.
AlexNet, Qwen, and DOOM run through the native GPU driver. Kernel inputs compile inside D3OS during device load.
Read [the inference guide](inference.howto.md) to run the existing tests.

## Hardware this guide targets

- Intel graphics: `0000:00:02.0`, driver `i915`.
- RTX 4090 Laptop GPU: `0000:01:00.0`, PCI ID `10de:2757`.
- NVIDIA audio: `0000:01:00.1`, PCI ID `10de:22bb`.
- The two NVIDIA functions currently occupy IOMMU group 15 together.
- The internal panel initially used NVIDIA in `AsusMuxDgpu` mode; it now uses Intel in Hybrid mode.

These addresses and IDs are specific to this machine.
Disconnect external displays attached to NVIDIA during setup.
Keep access to the firmware GPU setting and a Linux text console for recovery.

## 1. Move the panel to Intel

Inspect the current firmware settings:

```bash
asusctl armoury get dgpu_disable
asusctl armoury get gpu_mux_mode
```

Set Hybrid mode, which keeps NVIDIA available while routing the panel through integrated graphics:

```bash
sudo asusctl armoury set dgpu_disable 0
sudo asusctl armoury set gpu_mux_mode 1
```

Save your work, then reboot:

```bash
sudo systemctl reboot
```

The [ASUS GPU switching documentation](https://opengamingcollective.github.io/asusctl/faq/gpu-switching.html) defines this pair as Hybrid mode and requires a reboot.
Do not select Integrated mode: it disables the NVIDIA device.
The installed `supergfxctl` reports only `AsusMuxDgpu`, so this guide uses the supported `asusctl armoury` interface.
If the commands fail or the settings revert, select Hybrid or Optimus in firmware and repeat the checks below.

## 2. Verify the host display and device isolation

After reboot, run:

```bash
asusctl armoury get dgpu_disable
asusctl armoury get gpu_mux_mode
lspci -nnk -s 00:02.0
lspci -nnk -s 01:00

for connector in /sys/class/drm/card*-*; do
    [ -f "$connector/enabled" ] || continue
    [ "$(cat "$connector/enabled")" = enabled ] || continue
    card="${connector##*/}"
    card="${card%%-*}"
    printf '%s -> %s\n' "${connector##*/}" \
        "$(basename "$(readlink -f "/sys/class/drm/$card/device")")"
done

readlink -f /sys/bus/pci/devices/0000:01:00.0/iommu_group
ls /sys/bus/pci/devices/0000:01:00.0/iommu_group/devices/
cat /sys/bus/pci/devices/0000:01:00.0/reset_method
```

Continue only when:

- `dgpu_disable` is 0 and `gpu_mux_mode` is 1.
- Intel uses `i915`, and each enabled display connector maps to `0000:00:02.0`.
- Both NVIDIA functions remain visible with the IDs listed above.
- Their IOMMU group contains exactly `0000:01:00.0` and `0000:01:00.1`.

The group number can change after reboot.
The device reports `flr bus` as available GPU reset methods.
Repeated QEMU sessions reset the GPU successfully after firmware boot.
If the display still maps to NVIDIA, stop here and resolve the MUX setting before reserving the GPU.

## 3. Reserve NVIDIA for VFIO on the next boot

Install the VM and inspection tools:

```bash
sudo dnf install qemu-system-x86 edk2-ovmf pciutils
modinfo vfio_pci
test -c /dev/kvm
```

The machine already has IOMMU groups, so no extra IOMMU boot options are needed for this inspected configuration.
[VFIO requires ownership of the complete group](https://www.kernel.org/doc/html/latest/driver-api/vfio.html), including the audio function.
NVIDIA will be unavailable to host applications while this reservation is active.

Create a new dracut configuration file.
If the file already exists, inspect it before replacing it:

```bash
sudo test ! -e /etc/dracut.conf.d/d3os-vfio.conf && \
sudo tee /etc/dracut.conf.d/d3os-vfio.conf >/dev/null <<'EOF'
force_drivers+=" vfio_pci "
add_drivers+=" vfio vfio_iommu_type1 i915 "
EOF
```

Rebuild the running kernel's initramfs and configure that kernel's boot entry:

```bash
sudo dracut --force --kver "$(uname -r)"
sudo grubby --update-kernel="/boot/vmlinuz-$(uname -r)" \
    --args='rd.driver.pre=vfio-pci vfio-pci.ids=10de:2757,10de:22bb'
sudo grubby --info="/boot/vmlinuz-$(uname -r)"
```

[Dracut's `force_drivers` option](https://man7.org/linux/man-pages/man5/dracut.conf.5.html) loads the VFIO driver early.
The PCI ID argument reserves both NVIDIA functions.
Only the running kernel's boot entry and initramfs are changed, leaving other installed kernels as a recovery option.
Do not continue if dracut or grubby reports an error.

Save your work and reboot into the same kernel version:

```bash
sudo systemctl reboot
```

## 4. Verify the handoff

Run these commands after reboot:

```bash
lspci -nnk -s 00:02.0
lspci -nnk -s 01:00
group_path=$(readlink -f /sys/bus/pci/devices/0000:01:00.0/iommu_group)
ls "$group_path/devices/"
ls -l /dev/kvm /dev/vfio/vfio "/dev/vfio/${group_path##*/}"
sudo journalctl -b -k --grep='DMAR|IOMMU|vfio'
```

Intel must still use `i915`.
Both NVIDIA functions must report `Kernel driver in use: vfio-pci`.
Repeat the enabled-connector loop from stage 2 to confirm the host panel still uses Intel.

For an unprivileged QEMU session, grant your login user access to this specific group:

```bash
group_path=$(readlink -f /sys/bus/pci/devices/0000:01:00.0/iommu_group)
sudo chown "$(id -un)" "/dev/vfio/${group_path##*/}"
test -r /dev/kvm && test -w /dev/kvm
test -r "/dev/vfio/${group_path##*/}" && test -w "/dev/vfio/${group_path##*/}"
```

The ownership change lasts until the device node is recreated, usually at reboot.
If the KVM permission checks fail, add your user to `kvm` with `sudo usermod -aG kvm "$(id -un)"`, then log out and back in.
Keep SELinux and the IOMMU enabled.

At this point the host is ready for the next development session.
Passing the GPU through does not yet make inference run on it.
Do not use the repository's older `qemu-pci.sh` template without adapting its example addresses and permissions.

## 5. Boot D3OS with the physical GPU

VFIO must pin the VM's RAM.
The launcher checks the current process's locked-memory limit before starting QEMU.
In the terminal that will start the VM, raise the shell's limit:

```bash
sudo prlimit --pid "$$" --memlock=17179869184:17179869184
ulimit -Hl
```

The expected limit is `16777216` KiB.
Child processes inherit that limit until the shell exits.
For an agent session, use its process ID instead of `$$`.

From the repository root, run:

```bash
cargo make --no-workspace --profile production image
python3 tools/nvidia/prepare-firmware.py
python3 tools/nvidia/run-qemu.py --firmware target/nvidia/firmware --memory-mib 2048 --boot-check
```

The launcher checks that the IOMMU group contains exactly the GPU and audio function and that both use `vfio-pci`.
It starts QEMU without root privileges and keeps the virtual display for the D3OS console.
The boot check requires both NVIDIA functions, a native register probe, and a CPU inference self-test.
With `--firmware`, it also requires a native GSP boot and three GPU semaphore submissions.
Each submission must write the expected value back through the GPU address space.
The firmware preparation tool verifies SHA-256 hashes and reuses installed firmware or downloads the pinned Linux firmware files.
Success prints `NVIDIA_PASSTHROUGH_BOOT_OK`.
Read `target/nvidia/vm/serial.log` and `target/nvidia/vm/qemu.log` for diagnostics.

Without `--boot-check`, the VM stays running until stopped.
Use `--memory-mib 12288` for a VM that will load the Qwen assets.
Use `--gdb-port 1234` to pause at reset for a local debugger connection.
The kernel reports `NVIDIA_PROBE_OK` when its native register probe succeeds.
That marker confirms register access.
`NVIDIA_FWSEC_BOOT_OK` confirms authenticated FWSEC execution and protected-region initialization.
`NVIDIA_GSP_BOOT_OK` confirms GSP startup and management messages through the Rust DMA queues.
`NVIDIA_COMPUTE_CHANNEL_OK` confirms channel construction and scheduling.
`NVIDIA_QUEUE_OK` confirms three GPU writes through GPFIFO submissions.
These markers do not confirm shader execution or GPU model execution.

The driver uses signed NVIDIA firmware with its own Rust loaders and message protocol.
It does not load host NVIDIA libraries.
The host-independent protocol and firmware-parser tests run with `cargo test -p d3os-nvidia`.
The generated ABI follows tinygrad revision `81d16543a19d61c82a94df30feba2af8396275af`.
Its reference license is in `os/library/nvidia/LICENSE.tinygrad`.
Firmware files stay under the ignored `target/nvidia/firmware` directory.

## 6. Run the shader development test

The shader test executes native vector addition on the passed-through AD103.
It verifies all 257 output values against CPU references.
The driver copies the initialized graphics template into a private image and relocates its internal pointer.
It binds that image through a GPU virtual address in RAMIN before it schedules the channel.
This path replaces the earlier user-context promotion, which left the image uninitialized.
The context layout follows NVIDIA's [image definitions](https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/volta/gv100/dev_ctxsw.ref.txt) and [RAMIN definitions](https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/turing/tu104/dev_ram.ref.txt).
Host tests reject malformed image headers and verify that relocation preserves unrelated fields.
The native userspace backend runs AlexNet on this GPU.
Qwen also passes its complete numerical fixture through the native GPU backend.

The host build exports portable Tile IR from Rust cuTile kernels.
Inside D3OS, CUDA 13.4.92 `tileiras` and its `ptxas` subprocess compile that IR for the detected GPU.
The current Ada driver selects `sm_89` from the GPU register identity.
D3OS retains its Rust loader, GPU page tables, QMD descriptors, and GPFIFO submissions.
The compiler uses the Linux execution personality and its interpreter libraries inside D3OS.
GPU execution uses the native Rust driver without `libcuda` or a Linux NVIDIA driver.

Build the kernels and run the boot checks:

```bash
rustup toolchain install 1.98.1 --profile minimal --component clippy,rustfmt
python3 tools/nvidia/prepare-kernels.py --audit
python3 tools/nvidia/run-qemu.py --firmware target/nvidia/firmware \
    --kernels target/nvidia/kernels --memory-mib 4096 --boot-check
```

The build downloads checksum-pinned compiler tools and headers without installing a GPU driver.
The launcher stages `.tilebc` inputs and the compiler runtime. It does not stage `.cubin` files.
The host audit also generates qualification records for the native loader.
D3OS verifies the runtime assets and requires its compiled output to match the qualified output hash.
These records qualify the unused launch-metadata slot and the supported CUDA data-pointer relocations.
Regenerate the bundle with `--audit` after kernel source changes.
The compiler and generated modules remain inside D3OS. The module cache lasts until the VM stops.
The boot log reports compilation and validation as separate load phases.
The launcher waits for `NVIDIA_COMPILE_READY` before it starts a workload or its timeout.
Use `--compile-timeout` for the separate compilation budget (600 seconds by default).
Install libclang before the first build.
Use `--toolkit /path/to/cuda-13.4` to supply an existing matching toolkit.

The launcher requires `NVIDIA_KERNEL_OK` for this test to pass.
A CPU self-test cannot satisfy that requirement.
`NVIDIA_CONTEXT_OK` confirms image construction and binding.
It does not confirm shader execution.
The shader test records GPU semaphore checkpoints, RAMIN fields, notifier contents, and firmware events in the serial log.

## 7. Build and verify tensor kernels

Tensor-core execution is a requirement for the AlexNet and Qwen GPU backends.
AlexNet uses the TF32 inference kernels described in section 8.
The [Qwen guide](qwen.howto.md#run-on-the-nvidia-gpu) gives its GPU commands and verification requirements.
The cuTile matrix programs use explicit `mmaf` operations.
The compiler chooses register layouts and emits Tensor Core instructions.
The FP16 and BF16 checks accumulate into FP32 and use masked 16-by-32 output tiles.
Inference uses compensated TF32 decomposition to preserve the existing FP32 accuracy policy.

Run the host checks:

```bash
python3 tools/inference/test-cutile-kernels.py
```

The build produces `tensor-matmul.tilebc` with `tensor_matmul_f16` and `tensor_matmul_bf16` entry points.
D3OS generates the executable module from that input.
The host validation command uses `--audit` and keeps its comparison binaries in `host-audit/`.
These audit binaries do not enter the boot image.
SASS audits require `HMMA.16816.F32` and `HMMA.16816.F32.BF16` instructions.
The inference audit also requires TF32 Tensor Core instructions.
The portable entry manifest records required thread counts. The assembled module supplies register usage and other launch metadata.
The manifest always records `hardware_validated: false` until separate hardware evidence exists.

D3OS binds compiler constant banks and selects a 32, 64, or 100 KiB Ada shared-memory partition.
It allocates scratch from the binary's stack metadata.
The scratch diagnostic must produce real local-memory loads and stores.
The timestamp diagnostic compares QMD GPU timestamps with host elapsed time.
It does not measure SM clock cycles.

The VM shader command in section 6 also stages the tensor kernels.
After vector addition succeeds, D3OS runs five matrix shapes through each tensor kernel.
It compares every output against CPU references and checks guards around the output buffers.
The input values are exact in both formats, so these cases verify layout and arithmetic rather than conversion accuracy.
Both kernels must report `NVIDIA_TENSOR_OK` before the launcher accepts the run.
These hardware checks pass on the RTX 4090 Laptop GPU.
They include partial tiles, multiple thread blocks, and a 1-by-1024 times 1024-by-1024 projection.
These FP16 and BF16 kernels remain separate from the TF32 AlexNet kernels.

## 8. Run native GPU inference

After the image and kernel builds, run the operator checks:

```bash
python3 tools/nvidia/run-qemu.py --firmware target/nvidia/firmware \
    --kernels target/nvidia/kernels --memory-mib 2048 --boot-check \
    --inference gpu-self-test
```

The command requires `INFERENCE_OK self-test backend=nvidia`.
It runs a small convolutional network through the userspace compute API.
Additional checks cover six tensor matrix cases, batch offsets, padding, rectangular kernels, pooling, and storage reuse.
The matrix checks include partial tiles and a 9,216-element reduction against FP64 references.
Softmax checks include multiple rows with large positive and negative logits.

With the AlexNet assets from the [inference guide](inference.howto.md), run:

```bash
python3 tools/nvidia/run-qemu.py --firmware target/nvidia/firmware \
    --kernels target/nvidia/kernels --memory-mib 4096 --boot-check \
    --inference gpu-alexnet --timeout 240
```

This command requires `INFERENCE_OK alexnet backend=nvidia layers=8`.
All eight layers must match the stored reference with the original `2e-4` absolute and relative tolerances.
Both commands passed with four vCPUs and the passed-through RTX 4090 Laptop GPU.
The AlexNet dog fixture selected class 258 with probability approximately 0.72448.

The shell commands are `infer gpu-self-test` and `infer gpu-alexnet [MODEL FIXTURE]`.
AI applications select the GPU with `d3os_ai::native_device()`.
They use Burn tensors and modules through that device. The native backend allocates storage and dispatches GPU operations.
The backend returns a device error when native compute is unavailable.
It never falls back to CPU execution.

The portable `inference.tilebc` module contains TF32 matrix kernels and FP32 elementwise operators.
Each matrix tile combines three `HMMA.1688.F32.TF32` products with compensated FP32 accumulation.
The required host `--audit` also verifies Tensor Core instructions in both matrix entry points.
Convolution uses GPU `im2col` followed by tensor matrix multiplication.
Fully connected layers use a tensor kernel that reads transposed weights directly.
ReLU, bias, pooling, and softmax also run on the GPU.

Weights and activations remain in process-owned VRAM buffers between operators.
The kernel validates handle ownership, buffer extents, shapes, launch bounds, and output aliasing.
Only built-in kernels are available through this syscall.
Completed operations permit buffer reuse, and process exit releases remaining buffers.
A dispatch error disables further compute requests and retains storage until VM reset.
Reference comparisons and top-five selection use explicit downloads.

Uploads and downloads use a driver-owned 1 MiB staging allocation in coherent guest RAM.
The GPU maps that allocation through its MMU-v2 system-coherent aperture.
Native copy kernels transfer each chunk, and completion fences separate CPU access from GPU access.
The allocation remains owned by the device even after a dispatch error.
The GPU clears both fresh and recycled application buffers before the driver returns a handle.

The boot check requires `NVIDIA_TRANSFER_OK cases=6 guards=ok bits=exact` when the inference module is present.
Those checks cover both transfer directions, staging boundaries, partial chunks, output guards, and arbitrary bit patterns including NaNs.
The `NVIDIA_TRANSFER_BENCH` line measures 128 MiB of aggregate uploads and downloads with repeated 1 MiB chunks.
That measurement includes CPU staging copies, kernel dispatch, and completion waits.
The warm-buffer run measured 5,333 MiB/s on the development GPU.

With this path, AlexNet passed all eight reference comparisons in 0.77–0.94 seconds instead of the earlier 26–27 seconds.
The Qwen reference run passed in 11.5 seconds instead of the earlier 325 seconds.
These application timings include file access, model upload, and reference comparisons.
Dedicated copy-engine transfers and optimized tensor tiling remain unfinished.
A later repeat encountered a GSP timeout during graphics-class initialization before model execution.
The next VM boot passed without a code change.
Repeated-reset reliability remains under investigation.

## 9. Run DOOM on the GPU

Prepare the checksum-pinned Freedoom assets:

```bash
python3 tools/nvidia/prepare-doom.py
```

The tool downloads Freedoom 0.13.0 and preserves its license in `target/doom/COPYING.txt`.
The launcher stages Phase 1 as `/usr/doom.wad` and includes the license in the VM image.
The normal image build includes the `doom` executable.
The kernel build from section 7 also includes the raster, fuzz, and palette kernels in `inference.tilebc`.

Run the comparison with the CPU renderer:

```bash
python3 tools/nvidia/run-qemu.py --firmware target/nvidia/firmware \
    --kernels target/nvidia/kernels --memory-mib 2048 --boot-check \
    --inference gpu-doom-verify
```

Run the game without CPU scene drawing:

```bash
python3 tools/nvidia/run-qemu.py --firmware target/nvidia/firmware \
    --kernels target/nvidia/kernels --memory-mib 2048 --boot-check \
    --inference gpu-doom
```

Both commands require 600 rendered views, control input, low-detail drawing, and invisibility rendering.
The comparison command requires exact equality for every indexed pixel after each GPU batch.
Both commands quit the game and require a successful GPU operator test in the same VM.
The launcher saves `target/nvidia/vm/doom.ppm` before exit and reports `NVIDIA_DOOM_OK` only after all checks pass.

For interactive play, start a visible VM:

```bash
python3 tools/nvidia/run-qemu.py --firmware target/nvidia/firmware \
    --kernels target/nvidia/kernels --memory-mib 2048 \
    --inference gpu-doom --display gtk
```

Then enter this command in the D3OS shell:

```text
doom --gpu -warp 1 1 -skill 2
```

Arrow keys move and turn, A/D strafe, E uses doors, Space fires, and Shift runs.
Number keys select weapons, Tab shows the automap, F5 changes detail, and Escape opens the menu.
F10 opens the quit prompt, and Y confirms exit.
The default `doom` command uses the CPU renderer.
An explicit GPU request returns an error if native compute is unavailable.

Walls, floors, ceilings, sprites, invisibility, palette conversion, and scaling run on the NVIDIA GPU.
Game logic, visibility, HUD, menus, automap, wipes, and the final console copy remain on the CPU.
The visible QEMU window uses the virtual console display, while native NVIDIA compute produces the scene pixels.
The [renderer explanation](nvidia-rendering-design.md) describes command ordering and buffer ownership.

The development GPU passed both 600-view checks with four vCPUs and 2 GiB of guest RAM.
A 30-second recording measured about 50–70 FPS after startup, including transfers and console presentation.
The window title shows the frame rate. The corner overlay is removed.

With host `ffmpeg` and Python Pillow installed, capture the same workloads:

```bash
python3 tools/nvidia/capture-demo.py qwen
python3 tools/nvidia/capture-demo.py doom --seconds 30 --fps 30
```

The first command asks Qwen3.5-0.8B which country is largest by size through `infer gpu-qwen`.
It saves the terminal screenshot in `target/nvidia/captures/qwen-largest-country.png`.
The second command records movement and firing directly from QEMU into `target/nvidia/captures/doom-nvidia-30s.mp4`.
The application draws the window title and FPS inside D3OS.
The recorder also preserves serial logs and a manifest with capture timing and game frame rates.

To record a long Qwen reply with live token rates, run:

```bash
python3 tools/nvidia/capture-demo.py qwen --video --seconds 120 --fps 15 \
  --output target/nvidia/captures/qwen-essay
```

The default prompt requests a detailed essay about hand washing, with a limit of 4,096 new tokens.
Use `--prompt` and `--max-tokens` to select another workload.
The recorder enters the command through QEMU keyboard input and captures the guest framebuffer at real speed.
It saves `qwen-nvidia.mp4`, the transcript, serial evidence, and a JSON file with measured rates.

## Measure resident workloads

Use the warm benchmarks to separate inference time from file access and model upload:

```bash
python3 tools/nvidia/run-qemu.py --boot-check --inference gpu-alexnet-bench \
    --firmware target/nvidia/firmware --kernels target/nvidia/kernels
python3 tools/nvidia/run-qemu.py --boot-check --inference gpu-qwen-bench \
    --memory-mib 12288 --firmware target/nvidia/firmware --kernels target/nvidia/kernels
python3 tools/nvidia/run-qemu.py --boot-check --inference gpu-kernel-bench \
    --memory-mib 12288 --firmware target/nvidia/firmware --kernels target/nvidia/kernels
python3 tools/nvidia/run-qemu.py --boot-check --inference gpu-doom-bench \
    --firmware target/nvidia/firmware --kernels target/nvidia/kernels
```

The DOOM benchmark runs the WAD's uncapped `demo1` and reports the engine's timedemo FPS.
Qwen performs 64 fixed decode steps, including steps after EOS.
These steps measure throughput and do not represent a chat response.
The model benchmarks run their full reference comparisons after the timed work.

For per-kernel GPU timestamps, add `--qwen-device-timing` to the Qwen benchmark command.
This option serializes run 1 and labels it `instrumented=true`.
The per-kernel profile reports device nanoseconds separately from CPU preparation and wait cycles.
Instrumentation changes scheduling and adds overhead. Use the uninstrumented runs for throughput comparisons.

### Compare against Linux

The reference tool requires a readable host kernel, matching NVIDIA modules, `nvidia-smi`, GCC, and installed CUDA libraries.
The current environment supplies CUDA libraries through `/usr/local/lib/ollama/cuda_v12`.
The tool copies these files into an isolated Linux guest without changing the host GPU binding.
It requires no host root privileges after VFIO access is configured.
Run only one GPU guest at a time.

Compare the native projection cubin against cuBLAS FP32:

```bash
python3 tools/nvidia/linux-reference.py --kernel-benchmark
```

Compare the pinned DOOM engine and WAD with Linux software rendering:

```bash
python3 tools/nvidia/linux-reference.py --doom-benchmark
```

This headless comparison includes a framebuffer copy, but excludes physical display costs.
It is not an OpenGL benchmark.

For the Qwen reference, download the pinned Linux binary and BF16 model:

```bash
mkdir -p target/nvidia/linux-llama target/nvidia/linux-model
curl --fail --location \
    https://github.com/ggml-org/llama.cpp/releases/download/b11827/llama-b11827-bin-ubuntu-cuda-12.8-x64.tar.gz \
    --output target/nvidia/linux-llama/llama-b11827-bin-ubuntu-cuda-12.8-x64.tar.gz
curl --fail --location \
    https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-BF16.gguf \
    --output target/nvidia/linux-model/Qwen3.5-0.8B-BF16.gguf
sha256sum --check <<'EOF'
c1f538781281ec07841badf0f3972ed88e4f4b60b5aa7598139cadfa3d51a484  target/nvidia/linux-llama/llama-b11827-bin-ubuntu-cuda-12.8-x64.tar.gz
cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5  target/nvidia/linux-model/Qwen3.5-0.8B-BF16.gguf
EOF
```

Build the current D3OS image and prepare the [native Qwen assets](qwen.howto.md#prepare-the-model) before continuing.
Use the existing Qwen Python environment for the model audit.
The preparation tool downloads pinned llama.cpp headers and extracts the checked binary archive.
If you used a separate exporter cache, add `--tokenizer-json PATH` to select its upstream `tokenizer.json`.
The selected CPU IDs must name four available performance cores on your host.

```bash
set -e
taskset -c 8,10,12,14 python3 tools/nvidia/run-qemu.py \
    --firmware target/nvidia/firmware --kernels target/nvidia/kernels \
    --memory-mib 12288 --boot-check --inference gpu-qwen-bench --timeout 180 \
    > target/nvidia/d3os-qwen-bench.log 2>&1
cp target/nvidia/vm/serial.log target/nvidia/d3os-qwen-bench-serial.log
target/qwen-venv/bin/python tools/nvidia/prepare-linux-qwen.py \
    --gguf target/nvidia/linux-model/Qwen3.5-0.8B-BF16.gguf \
    --d3os-log target/nvidia/d3os-qwen-bench-serial.log
taskset -c 8,10,12,14 python3 tools/nvidia/linux-reference.py --memory-mib 12288 \
    --llama-directory target/nvidia/linux-llama/llama-b11827 \
    --model target/nvidia/linux-model/Qwen3.5-0.8B-BF16.gguf \
    --matched-qwen-input target/nvidia/matched-qwen
cp target/nvidia/linux-reference/serial.log target/nvidia/linux-qwen-bench-serial.log
target/qwen-venv/bin/python tools/nvidia/test-matched-qwen.py \
    --gguf target/nvidia/linux-model/Qwen3.5-0.8B-BF16.gguf \
    --d3os-log target/nvidia/d3os-qwen-bench-serial.log \
    --linux-log target/nvidia/linux-qwen-bench-serial.log
```

The audit checks every text tensor and model setting against the native archive.
It exports exactly 22 prompt tokens and 64 decode inputs, with FP32 caches and capacity for 256 tokens.
Linux reports new-context and reset-context runs separately. Each reset clears model state instead of reusing the prompt.
The launcher requires complete GPU offload and identical greedy choices across all six runs.
The tests corrupt private model views and copied traces; the original model files stay unchanged.
Read `target/nvidia/matched-qwen/manifest.json` and `target/nvidia/linux-reference/matched-qwen-manifest.json` for checksums and settings.
Add `--llama-flash-attention off` to the Linux command to measure the unfused attention path.

The older synthetic `llama-bench` mode remains available by omitting `--matched-qwen-input`.
For its FP32 storage comparison, convert the same model without quantization:

```bash
LD_LIBRARY_PATH="$PWD/target/nvidia/linux-llama/llama-b11827:/usr/local/lib/ollama/cuda_v12" \
    target/nvidia/linux-llama/llama-b11827/llama-quantize \
    target/nvidia/linux-model/Qwen3.5-0.8B-BF16.gguf \
    target/nvidia/linux-model/Qwen3.5-0.8B-F32.gguf F32
python3 tools/nvidia/linux-reference.py --memory-mib 12288 \
    --llama-directory target/nvidia/linux-llama/llama-b11827 \
    --model target/nvidia/linux-model/Qwen3.5-0.8B-F32.gguf
```

The tool saves the guest log and driver manifest under `target/nvidia/linux-reference`.
A directory lock prevents another invocation from replacing files while that guest runs.
See [native NVIDIA performance](nvidia-performance.md) for measurement limits and the remaining performance gap.

## Restore NVIDIA to the host

Run this from the kernel entry changed in stage 3, with the VM stopped:

```bash
sudo mv /etc/dracut.conf.d/d3os-vfio.conf \
    /etc/dracut.conf.d/d3os-vfio.conf.disabled
sudo grubby --update-kernel="/boot/vmlinuz-$(uname -r)" \
    --remove-args='rd.driver.pre=vfio-pci vfio-pci.ids=10de:2757,10de:22bb'
sudo dracut --force --kver "$(uname -r)"
sudo systemctl reboot
```

Hybrid mode can remain enabled for normal host use.
To restore the original panel routing afterward, set `dgpu_disable` to 0 and `gpu_mux_mode` to 0 with `asusctl`, then reboot again.
If the desktop cannot start, use a text console or boot an unchanged kernel to repair the changed entry.
