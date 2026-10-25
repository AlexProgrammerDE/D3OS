#!/usr/bin/env python3
"""Verify vector registers, glibc thread cleanup, and join with the staged runtime."""

import argparse
import importlib.util
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accel", choices=("kvm", "tcg"), default="tcg")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--memory-mib", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--bios", type=Path)
    parser.add_argument("--kernel-image", type=Path)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if not 2 <= args.cpus <= 64 or not 1 <= args.repeat <= 100 or args.timeout < 10 or args.memory_mib < 256:
        parser.error("use 2 to 64 CPUs, 1 to 100 repetitions, at least 256 MiB, and at least 10 seconds")
    harness = module("harness", ROOT / "tools/inference/test-qemu.py")
    compiler = module("compiler", ROOT / "tools/nvidia/guest-compiler.py")
    directory = (args.directory or ROOT / f"target/linux-threads-{args.accel}-{args.cpus}").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "threadexit"
    subprocess.run(
        ["gcc", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
         str(ROOT / "tools/kernel/fixtures/linux-thread-exit.c"), "-o", str(binary)],
        check=True,
    )
    assets = [(binary, Path("bin/threadexit")), *compiler.runtime_assets([binary])]
    image = harness.prepare(
        directory, "self-test", ROOT / "target/qwen", assets,
        kernel_image=args.kernel_image,
    )
    bios = args.bios or next(
        (path for path in (
            ROOT / "RELEASEX64_OVMF.fd",
            Path("/usr/share/OVMF/OVMF_CODE_4M.fd"),
            Path("/usr/share/OVMF/OVMF_CODE.fd"),
            Path("/usr/share/edk2/ovmf/OVMF_CODE.fd"),
        ) if path.is_file()),
        None,
    )
    if bios is None or not bios.is_file():
        parser.error("provide --bios with an OVMF firmware image")
    serial, monitor = directory / "serial.log", directory / "qmp.sock"
    serial.unlink(missing_ok=True)
    monitor.unlink(missing_ok=True)
    command = [
        "qemu-system-x86_64", "-machine", "q35", "-accel", args.accel,
        "-cpu", "host" if args.accel == "kvm" else "Haswell,fsgsbase",
        "-smp", str(args.cpus), "-m", str(args.memory_mib),
        *harness.firmware_options(bios.resolve(), directory),
        "-drive", f"format=raw,file={image}", "-display", "none",
        "-device", "virtio-vga", "-serial", f"file:{serial}",
        "-qmp", f"unix:{monitor},server=on,wait=off", "-no-reboot",
    ]
    qmp = None
    deadline = time.monotonic() + args.timeout
    with (directory / "qemu.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            qmp = harness.Qmp(monitor, process, deadline)

            def wait(marker, count=1):
                while time.monotonic() < deadline:
                    text = harness.read_serial(serial)
                    if any(error in text for error in (
                        "LINUX_THREAD_EXIT_FAIL", "Linux process fault", "panicked at",
                    )):
                        raise RuntimeError(f"Linux thread cleanup failed: {serial}")
                    if text.count(marker) >= count:
                        return
                    if process.poll() is not None:
                        raise RuntimeError(f"QEMU exited: {directory / 'qemu.log'}")
                    time.sleep(0.05)
                raise TimeoutError(f"waiting for {marker}: {serial}")

            wait("name = shell")
            time.sleep(1)
            for run in range(1, args.repeat + 1):
                qmp.type_command("threadexit")
                wait("LINUX_THREAD_EXIT_OK", run)
                wait("LINUX_RESERVATION_OK", run)
                time.sleep(0.2)
            print(f"LINUX_THREAD_EXIT_SUITE_OK accelerator={args.accel} cpus={args.cpus} memory_mib={args.memory_mib} runs={args.repeat} serial={serial}")
        finally:
            if qmp is not None:
                try:
                    qmp.execute("quit")
                except (OSError, RuntimeError):
                    pass
                qmp.close()
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
