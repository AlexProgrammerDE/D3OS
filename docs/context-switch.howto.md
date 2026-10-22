# Test interrupt safety during a context switch

The kernel previously allowed interrupts during cooperative context switches.
An interrupt between the two `SWAPGS` instructions caused a panic with `local APIC not initialized`.

D3OS normally keeps its core-local storage pointer in the inactive `IA32_KERNEL_GS_BASE` register.
The switch temporarily exchanges GS bases to update the incoming thread's TSS stack pointer.
During that exchange, the interrupt read the user GS base as the core-local pointer.
The observed user GS base was zero, so interrupt acknowledgement could not find the local APIC.

The switch now saves the caller's flags before disabling interrupts.
It restores interrupts only after exchanging GS back, installing the incoming context, and unlocking the scheduler.
The unlock callback runs on an aligned stack before the incoming registers are restored.
Initial thread startup follows the same unlock and restore order.

## Run the regression

Install the [OS build requirements](../README.md#requirements), QEMU, OVMF, GDB, and GNU binutils.
From the repository root, run:

```bash
cargo make --no-workspace --profile production image
python3 tools/kernel/test-context-switch.py --accel tcg
```

The test boots two vCPUs with software CPU emulation.
It needs no model download, NVIDIA GPU, or hardware virtualization.
For a host with KVM access, run:

```bash
python3 tools/kernel/test-context-switch.py --accel kvm --cpus 4
```

The launcher detects Ubuntu's `OVMF_CODE_4M.fd` and Fedora's `OVMF_CODE.fd` firmware layouts.
It copies the matching variables template into the test directory before each boot.
For firmware outside the standard paths, add `--bios /path/to/firmware.fd`.

GDB stops a cooperative switch immediately after its first `SWAPGS` and queues a keyboard interrupt.
The test checks that interrupts remain masked through scheduler unlock and context restoration.
It also checks the callback's System V stack alignment and the restored GS base.
After GDB detaches, the guest must complete three native inference self-tests without a panic.

Before the fix, injecting a keyboard interrupt at this point reproduced the APIC panic.
The regression also rejects the old code because its interrupt flag remains enabled in the GS exchange window.

Successful runs print `CONTEXT_SWITCH_SUITE_OK`.
Read `target/context-switch/gdb.log` for debugger checks and `target/context-switch/guest.log` for guest results.
The generated debugger commands are in `target/context-switch/check.gdb`.
The `Native inference` workflow runs this regression on pushes and pull requests.

## Repeat the Qwen workload

Prepare the assets described in [the Qwen guide](qwen.howto.md), then run:

```bash
python3 tools/inference/test-qemu.py --qwen-verify --accel kvm --cpus 2 --repeat 3 --timeout 300
python3 tools/inference/test-qemu.py --qwen-verify --accel kvm --cpus 4 --repeat 3 --timeout 300
python3 tools/inference/test-qemu.py --qwen --accel kvm --cpus 4 --repeat 3 --timeout 300
```

These commands run sequentially because they share the staged model image.
Each numerical run checks all 24 layers, vocabulary logits at 29 positions, and 105 tokenizer cases.
Use a longer timeout on slower hosts.
Multiple guest CPUs exercise kernel scheduling; the inference API still uses its CPU backend.
