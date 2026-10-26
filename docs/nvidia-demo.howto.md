# Run the NVIDIA vector-add demo

This demo boots the native driver on an RTX 4090 Laptop GPU (`10de:2757`, AD103).
It adds two vectors of 257 FP32 values and compares every GPU result with a CPU reference.
The final block includes unused threads to exercise the kernel's bounds check.

The driver loads signed NVIDIA firmware, creates GPU page tables and a compute context, then submits one vector-add kernel.
The GPU writes the output through its own address space.
The demo uses no CUDA driver, CUDA runtime, AI libraries, or compiler inside D3OS.
The host assembles a fixed PTX file into native `sm_89` instructions before boot.

## Requirements

- The normal [D3OS build tools](../README.md#requirements).
- Linux with KVM, QEMU, OVMF, and Python 3.
- The target GPU and its audio function reserved for `vfio-pci`.
- Read/write access to `/dev/kvm`, `/dev/vfio/vfio`, and the GPU's VFIO group.
- A standalone NVIDIA `ptxas` assembler with `sm_89` support.

The launch script requires an IOMMU group that contains exactly the GPU and its audio function.
It does not change host drivers, boot parameters, or device permissions.
This driver currently supports only the target device and firmware version 570.144.

## Build and run

From the repository root, build the normal D3OS image:

```bash
cargo make --no-workspace --profile production image
```

Prepare the firmware and assemble the fixed demo:

```bash
python3 tools/nvidia/prepare-firmware.py
python3 tools/nvidia/prepare-demo.py --ptxas /path/to/ptxas
```

The firmware script reuses installed files or downloads a pinned version.
It verifies each file's SHA-256 hash.
Generated assets remain under the ignored `target/nvidia` directory.

For the default 2048 MiB VM, set the shell's locked-memory limit to at least 2176 MiB.
If your hard limit is too low, raise it before the next command.
The launcher reports the required limit in bytes.

Run the hardware check:

```bash
python3 tools/nvidia/run-qemu.py --boot-check
```

For a different PCI address, add `--gpu 0000:01:00.0` with your address.
For a custom OVMF image, add `--ovmf /path/to/OVMF_CODE.fd`.
An `OVMF_CODE` image requires the corresponding `OVMF_VARS` template beside it.

The launcher builds a private image with the kernel, shell, terminal emulator, firmware, and vector-add binary.
The demo runs once during boot, after the kernel loads the initrd.
It requires no shell command or application syscall.
Without firmware assets, normal D3OS boots skip firmware startup and the demo.

Success prints `NVIDIA_DEMO_OK` on the host and stops QEMU.
Without `--boot-check`, QEMU continues until you stop it.
Logs are in `target/nvidia/vm/serial.log` and `target/nvidia/vm/qemu.log`.

The serial log records these milestones:

- `NVIDIA_PROBE_OK`: native PCI and register access.
- `NVIDIA_FWSEC_BOOT_OK`: authenticated FWSEC startup.
- `NVIDIA_GSP_BOOT_OK`: management firmware startup and RPC queues.
- `NVIDIA_CONTEXT_OK`: graphics context construction and virtual binding.
- `NVIDIA_COMPUTE_CHANNEL_OK`: compute channel setup.
- `NVIDIA_KERNEL_OK vector-add count=257 backend=nvidia`: all GPU outputs match.

This extraction passed the hardware check on the RTX 4090 Laptop GPU with four vCPUs and 2048 MiB RAM.

Only the final marker establishes successful vector-add execution.
The output starts as NaNs, so unchanged output cannot pass the comparison.

## Review the implementation

`os/kernel/src/device/nvidia.rs` owns device discovery and startup.
Its `flcn`, `gsp`, `rm`, `mmu`, and `compute` modules cover firmware execution, RPC, contexts, address spaces, and submission.
`os/library/nvidia` contains checked wire records, firmware parsers, and context relocation.
The driver retains DMA allocations for its lifetime, including after initialization errors.

The PCI changes map extended configuration space and serialize shared CF8/CFC registers.
The VirtIO change avoids probing the upper half of a 64-bit BAR as a separate BAR.
These changes support GPU passthrough alongside the virtual console.

Run the host protocol, firmware, bit-field, and context tests:

```bash
cargo test --locked -p d3os-nvidia
```

These tests do not establish hardware execution.
The generated ABI uses a pinned metadata reference, with attribution in `os/library/nvidia/LICENSE.tinygrad`.
`tools/nvidia/generate-abi.py` reproduces the checked record definitions.

The branch contains no tensor kernels, inference code, GPU service API, or general GPU translation pipeline.
