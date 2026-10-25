#!/usr/bin/env python3
"""Verify that process exit stops busy or sleeping workers across native D3OS cores."""

import argparse
import importlib.util
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "inference_harness", ROOT / "tools/inference/test-qemu.py"
)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accel", choices=("kvm", "tcg"), default="kvm")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--bios", type=Path)
    parser.add_argument("--sleeping-workers", action="store_true")
    parser.add_argument("--kernel-image", type=Path)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if not 2 <= args.cpus <= 64 or not 1 <= args.repeat <= 100 or args.timeout < 10:
        parser.error("use 2 to 64 CPUs, 1 to 100 repetitions, and at least 10 seconds")
    bios = args.bios or next(
        (
            path
            for path in (
                ROOT / "RELEASEX64_OVMF.fd",
                Path("/usr/share/OVMF/OVMF_CODE_4M.fd"),
                Path("/usr/share/OVMF/OVMF_CODE.fd"),
                Path("/usr/share/edk2/ovmf/OVMF_CODE.fd"),
            )
            if path.is_file()
        ),
        None,
    )
    if bios is None or not bios.is_file():
        parser.error("provide --bios with an OVMF firmware image")
    suffix = "-sleeping" if args.sleeping_workers else ""
    directory = (
        args.directory or ROOT / f"target/process-exit-{args.accel}-{args.cpus}{suffix}"
    ).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    image = harness.prepare(
        directory,
        "self-test",
        ROOT / "target/qwen",
        [(ROOT / "loader/initrd/bin/proctest", Path("bin/proctest"))],
        kernel_image=args.kernel_image,
    )
    serial = directory / "serial.log"
    monitor = directory / "qmp.sock"
    serial.unlink(missing_ok=True)
    monitor.unlink(missing_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    (directory / "gdb.port").write_text(str(port))
    command = [
        "qemu-system-x86_64",
        "-machine",
        "q35",
        "-accel",
        args.accel,
        "-cpu",
        "host" if args.accel == "kvm" else "Haswell,fsgsbase",
        "-smp",
        str(args.cpus),
        "-m",
        "512M",
        *harness.firmware_options(bios.resolve(), directory),
        "-drive",
        f"format=raw,file={image}",
        "-display",
        "none",
        "-device",
        "virtio-vga",
        "-device",
        "virtio-rng-pci",
        "-nic",
        "model=rtl8139",
        "-serial",
        f"file:{serial}",
        "-qmp",
        f"unix:{monitor},server=on,wait=off",
        "-no-reboot",
        "-gdb",
        f"tcp:127.0.0.1:{port}",
    ]
    deadline = time.monotonic() + args.timeout
    qmp = None
    with (directory / "qemu.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log)
        try:
            qmp = harness.Qmp(monitor, process, deadline)

            def wait_for(marker, count=1):
                while time.monotonic() < deadline:
                    text = harness.read_serial(serial)
                    if "panicked at" in text or "PROCESS_LIFECYCLE_FAIL" in text:
                        raise RuntimeError(f"process lifecycle failed; read {serial}")
                    if text.count(marker) >= count:
                        return
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"QEMU exited; read {directory / 'qemu.log'}"
                        )
                    time.sleep(0.05)
                raise TimeoutError(f"waiting for {marker} run {count}; read {serial}")

            wait_for("name = shell")
            time.sleep(1)
            mode = "--sleeping-lifecycle" if args.sleeping_workers else "--lifecycle"
            for run in range(1, args.repeat + 1):
                qmp.type_command(f"proctest {mode}")
                wait_for("PROCESS_LIFECYCLE_OK", run)
                time.sleep(0.2)
            print(
                f"PROCESS_EXIT_SUITE_OK accelerator={args.accel} cpus={args.cpus} "
                f"runs={args.repeat} sleeping={args.sleeping_workers} serial={serial}"
            )
        except (RuntimeError, TimeoutError):
            harness.capture_failure(directory, port)
            raise
        finally:
            if qmp is not None:
                qmp.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
