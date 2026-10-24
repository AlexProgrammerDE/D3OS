#!/usr/bin/env python3
"""Exercise interrupt delivery during a native D3OS context switch with GDB."""

import argparse
import os
import re
import signal
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def symbols(kernel):
    output = subprocess.check_output(
        ["nm", "-S", "-C", "--defined-only", str(kernel)], text=True
    )
    result = {}
    for line in output.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) == 4:
            address, size, _, name = fields
            result[name] = (int(address, 16), int(size, 16))
    return result


def gdb_script(kernel, port):
    table = symbols(kernel)
    entry, size = table["kernel::process::thread::thread_switch"]
    panic, _ = table["__rustc::rust_begin_unwind"]
    unlock, _ = table["unlock_scheduler"]
    allocate, _ = table["<linked_list_allocator::Heap>::allocate_first_fit"]
    deallocate, _ = table["<linked_list_allocator::Heap>::deallocate"]
    output = subprocess.check_output(
        [
            "objdump",
            "-d",
            f"--start-address={entry}",
            f"--stop-address={entry + size}",
            str(kernel),
        ],
        text=True,
    )
    instructions = []
    for line in output.splitlines():
        match = re.match(r"\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s+)+)\s*(\S+)", line)
        if match:
            address, encoding, mnemonic = match.groups()
            instructions.append((int(address, 16), len(encoding.split()), mnemonic))
    swaps = [
        (address, size)
        for address, size, mnemonic in instructions
        if mnemonic == "swapgs"
    ]
    if len(swaps) != 2:
        raise RuntimeError("context switch must contain a balanced GS exchange")
    swapped = swaps[0][0] + swaps[0][1]
    ready = next(
        address for address, _, mnemonic in instructions if mnemonic.startswith("popf")
    )

    def check(expression, message):
        return f'if !({expression})\n  printf "CONTEXT_SWITCH_FAIL {message}\\n"\n  quit 1\nend\n'

    # Hardware breakpoints survive the bootloader loading the kernel over low RAM.
    # Stop a switch on the boot CPU, where keyboard IRQs are routed.
    script = f"""set pagination off
set confirm off
set tcp auto-retry on
set tcp connect-timeout 30
set remotetimeout 10
target remote 127.0.0.1:{port}
hbreak *{panic}
hbreak *{entry} if $_thread == 1
continue
"""
    script += check(f"$pc == {entry}", "context switch did not start")
    script += f"""delete 2
hbreak *{swapped} if $_thread == 1
continue
"""
    script += check(f"$pc == {swapped}", "kernel panicked before the GS window")
    script += check(
        "($eflags & 0x200) == 0", "interrupts enabled while kernel GS is active"
    )
    # The active GS base is now the kernel CLS. Save its value before queueing an IRQ.
    script += f"""set $kernel_cls = $gs_base
printf "CONTEXT_SWITCH_GS_MASKED gs=%#lx\\n", $kernel_cls
delete 3
monitor sendkey shift
hbreak *{unlock} if $_thread == 1
continue
"""
    script += check(f"$pc == {unlock}", "kernel panicked before scheduler unlock")
    script += check(
        "($eflags & 0x200) == 0", "interrupts enabled before scheduler unlock"
    )
    script += check(
        "((unsigned long)$rsp & 15) == 8",
        "scheduler callback stack violates the SysV ABI",
    )
    script += f"""delete 4
hbreak *{ready} if $_thread == 1
continue
"""
    script += check(f"$pc == {ready}", "kernel panicked before flags restore")
    script += check(
        "($eflags & 0x200) == 0", "interrupts enabled before context restore"
    )
    script += check("$gs_base != $kernel_cls", "kernel GS was not exchanged back")
    script += """printf "CONTEXT_SWITCH_OK irq_queued=true context_restored=true callback_aligned=true\\n"
delete breakpoints
"""
    # Continue into normal runtime work after the switch. Both heap mutation
    # paths must hold an IRQ-save guard, so neither can lose its lock owner to
    # timer preemption or recursively allocate from an interrupt handler.
    for address, operation in ((allocate, "allocate"), (deallocate, "deallocate")):
        script += f"hbreak *{address}\ncontinue\n"
        script += check(f"$pc == {address}", f"heap {operation} did not run")
        script += check(
            "($eflags & 0x200) == 0", f"interrupts enabled during heap {operation}"
        )
        script += "delete breakpoints\n"
    script += 'printf "HEAP_PREEMPTION_OK allocate_masked=true deallocate_masked=true\\n"\ndetach\nquit\n'
    return script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accel", choices=("kvm", "tcg"), default="tcg")
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--bios", type=Path, help="OVMF firmware image")
    args = parser.parse_args()
    if not 2 <= args.cpus <= 64 or args.timeout < 10:
        parser.error("use 2 to 64 CPUs and a timeout of at least 10 seconds")
    directory = ROOT / "target/context-switch"
    directory.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    kernel = ROOT / "loader/kernel.elf"
    script = directory / "check.gdb"
    script.write_text(gdb_script(kernel, port))
    command = [
        sys.executable,
        str(ROOT / "tools/inference/test-qemu.py"),
        "--accel",
        args.accel,
        "--cpus",
        str(args.cpus),
        "--repeat",
        "3",
        "--gdb-port",
        str(port),
        "--timeout",
        str(args.timeout),
    ]
    if args.bios is not None:
        command.extend(["--bios", str(args.bios.resolve())])
    with (directory / "guest.log").open("w") as guest_log:
        harness = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=guest_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            with (directory / "gdb.log").open("w") as log:
                subprocess.run(
                    ["gdb", "-q", "-batch", str(kernel), "-x", str(script)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=args.timeout,
                )
            # The guest deadline is unchanged. Allow its failure-only GDB
            # snapshot to finish before terminating the harness process group.
            harness.wait(timeout=args.timeout + 30)
            if harness.returncode != 0:
                raise RuntimeError(
                    f"guest verification failed; read {directory / 'guest.log'}"
                )
        except (subprocess.SubprocessError, RuntimeError):
            for name in ("guest.log", "gdb.log"):
                path = directory / name
                if path.is_file():
                    print(
                        f"{path}:\n{path.read_text(errors='replace')}", file=sys.stderr
                    )
            raise
        finally:
            if harness.poll() is None:
                os.killpg(harness.pid, signal.SIGTERM)
                harness.wait(timeout=10)
    print(
        f"CONTEXT_SWITCH_SUITE_OK accelerator={args.accel} cpus={args.cpus} logs={directory}"
    )


if __name__ == "__main__":
    main()
