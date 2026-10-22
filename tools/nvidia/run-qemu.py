#!/usr/bin/env python3
"""Boot D3OS with an already reserved NVIDIA GPU and its audio function."""

import argparse
import importlib.util
import os
import resource
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def vfio_devices(gpu):
    device = Path("/sys/bus/pci/devices") / gpu
    if not device.is_dir() or (device / "vendor").read_text().strip() != "0x10de":
        raise RuntimeError(f"{gpu} is not an NVIDIA PCI device")
    if (device / "device").read_text().strip() != "0x2757":
        raise RuntimeError(
            "this bring-up profile requires the RTX 4090 Laptop GPU (10de:2757)"
        )
    group = device / "iommu_group"
    if not group.exists():
        raise RuntimeError(f"{gpu} has no IOMMU group")
    functions = sorted(entry.name for entry in (group / "devices").iterdir())
    expected = [gpu, gpu.rsplit(".", 1)[0] + ".1"]
    if functions != expected:
        raise RuntimeError(
            f"IOMMU group must contain exactly GPU and audio: {functions}"
        )
    for address in functions:
        driver = Path("/sys/bus/pci/devices") / address / "driver"
        if not driver.exists() or driver.resolve().name != "vfio-pci":
            raise RuntimeError(f"{address} is not bound to vfio-pci")
    for node in [
        Path("/dev/kvm"),
        Path("/dev/vfio/vfio"),
        Path("/dev/vfio") / group.resolve().name,
    ]:
        if not os.access(node, os.R_OK | os.W_OK):
            raise RuntimeError(f"current user needs read/write access to {node}")
    return functions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0000:01:00.0")
    parser.add_argument("--image", type=Path, default=ROOT / "d3os.img")
    parser.add_argument("--memory-mib", type=int, default=4096)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument(
        "--boot-check",
        action="store_true",
        help="verify PCI discovery and native CPU self-test, then stop",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gdb-port", type=int, help="pause at reset for a local GDB connection"
    )
    args = parser.parse_args()
    if (
        not 512 <= args.memory_mib <= 32768
        or not 1 <= args.cpus <= 64
        or args.timeout <= 0
    ):
        parser.error("use 512 to 32768 MiB, 1 to 64 CPUs, and a positive timeout")
    devices = vfio_devices(args.gpu)
    if not args.image.is_file():
        parser.error(f"boot image does not exist: {args.image}")
    required_lock = (args.memory_mib + 128) * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if not args.dry_run and hard != resource.RLIM_INFINITY and hard < required_lock:
        parser.error(
            f"VFIO needs at least {required_lock} bytes of locked memory; this process has a hard limit of {hard}"
        )
    if not args.dry_run and soft != resource.RLIM_INFINITY and soft < required_lock:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (required_lock, hard))
    spec = importlib.util.spec_from_file_location(
        "inference_qemu", ROOT / "tools/inference/test-qemu.py"
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    directory = ROOT / "target/nvidia/vm"
    directory.mkdir(parents=True, exist_ok=True)
    firmware = next(
        (
            p
            for p in [
                Path("/usr/share/OVMF/OVMF_CODE_4M.fd"),
                Path("/usr/share/OVMF/OVMF_CODE.fd"),
                Path("/usr/share/edk2/ovmf/OVMF_CODE.fd"),
                ROOT / "RELEASEX64_OVMF.fd",
            ]
            if p.is_file()
        ),
        None,
    )
    if firmware is None:
        parser.error("install OVMF firmware")
    serial, monitor = directory / "serial.log", directory / "qmp.sock"
    command = [
        "qemu-system-x86_64",
        "-machine",
        "q35",
        "-accel",
        "kvm",
        "-cpu",
        "host",
        "-smp",
        str(args.cpus),
        "-m",
        str(args.memory_mib),
        *harness.firmware_options(firmware.resolve(), directory),
        "-drive",
        f"format=raw,file={args.image.resolve()}",
        "-boot",
        "c",
        "-display",
        "none",
        "-device",
        "virtio-vga",
        "-device",
        "virtio-rng-pci",
        "-nic",
        "model=rtl8139",
        "-device",
        "pcie-root-port,id=nvidia,chassis=1,slot=1",
        "-device",
        f"vfio-pci,host={devices[0]},bus=nvidia,addr=0.0,multifunction=on,rombar=0",
        "-device",
        f"vfio-pci,host={devices[1]},bus=nvidia,addr=0.1",
        "-serial",
        f"file:{serial}",
        "-qmp",
        f"unix:{monitor},server=on,wait=off",
        "-no-reboot",
    ]
    if args.gdb_port is not None:
        if not 1 <= args.gdb_port <= 65535:
            parser.error("--gdb-port must be from 1 to 65535")
        command.extend(["-gdb", f"tcp:127.0.0.1:{args.gdb_port}", "-S"])
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    serial.unlink(missing_ok=True)
    monitor.unlink(missing_ok=True)
    with (directory / "qemu.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log)
        qmp = None
        try:
            if not args.boot_check:
                print(
                    f"NVIDIA_VM_RUNNING pid={process.pid} serial={serial} qmp={monitor}",
                    flush=True,
                )
                process.wait()
                if process.returncode:
                    raise RuntimeError((directory / "qemu.log").read_text())
                return
            deadline = time.monotonic() + args.timeout
            qmp = harness.Qmp(monitor, process, deadline)
            started = False
            while time.monotonic() < deadline:
                text = serial.read_text(errors="replace") if serial.is_file() else ""
                if "panicked at" in text or "INFERENCE_FAIL" in text:
                    raise RuntimeError(f"guest failed: {serial}")
                if "name = shell" in text and not started:
                    time.sleep(1)
                    qmp.type_command("infer")
                    started = True
                if "INFERENCE_OK self-test" in text:
                    if "[0x10de:0x2757]" not in text or "[0x10de:0x22bb]" not in text:
                        raise RuntimeError(
                            "guest did not enumerate the target GPU and audio"
                        )
                    if "NVIDIA_PROBE_OK" not in text:
                        raise RuntimeError(
                            "native NVIDIA register probe did not succeed"
                        )
                    print(
                        f"NVIDIA_PASSTHROUGH_BOOT_OK cpus={args.cpus} serial={serial}",
                        flush=True,
                    )
                    return
                if process.poll() is not None:
                    raise RuntimeError((directory / "qemu.log").read_text())
                time.sleep(0.1)
            raise TimeoutError(f"D3OS boot check timed out: {serial}")
        finally:
            if qmp is not None:
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
