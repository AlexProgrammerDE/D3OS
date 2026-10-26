#!/usr/bin/env python3
"""Boot the native NVIDIA vector-add demo with an already reserved VFIO GPU."""

import argparse
import os
import resource
import shlex
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIRMWARE = ("gsp-570.144.bin", "bootloader-570.144.bin", "booter_load-570.144.bin")


def vfio_devices(gpu):
    device = Path("/sys/bus/pci/devices") / gpu
    if not device.is_dir() or (device / "vendor").read_text().strip() != "0x10de":
        raise RuntimeError(f"{gpu} is not an NVIDIA PCI device")
    if (device / "device").read_text().strip() != "0x2757":
        raise RuntimeError("this demo requires the RTX 4090 Laptop GPU (10de:2757)")
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
    for node in (
        Path("/dev/kvm"),
        Path("/dev/vfio/vfio"),
        Path("/dev/vfio") / group.resolve().name,
    ):
        if not os.access(node, os.R_OK | os.W_OK):
            raise RuntimeError(f"current user needs read/write access to {node}")
    return functions


def prepare(directory, firmware, cubin):
    stage = directory / "boot"
    stage.mkdir(parents=True, exist_ok=True)
    assets = [(firmware / name, f"usr/share/nvidia/{name}") for name in FIRMWARE]
    assets.append((cubin, "usr/share/nvidia/vector-add.cubin"))
    assets.extend(
        (ROOT / "loader/initrd/bin" / name, f"bin/{name}")
        for name in ("shell", "terminal_emulator")
    )
    with tarfile.open(
        stage / "initrd.tar", "w", format=tarfile.USTAR_FORMAT
    ) as archive:
        for source, destination in assets:
            archive.add(source, arcname=destination, recursive=False)
    for name in ("kernel.elf", "towboot.toml"):
        shutil.copyfile(ROOT / "loader" / name, stage / name)
    image = directory / "d3os.img"
    image.unlink(missing_ok=True)
    subprocess.run(
        [
            str(ROOT / "towbootctl"),
            "image",
            "--target",
            str(image),
            "--",
            "-config",
            "towboot.toml",
        ],
        cwd=stage,
        check=True,
    )
    return image


def firmware_options(image, directory):
    if not image.name.startswith("OVMF_CODE"):
        return ["-bios", str(image)]
    variables = image.with_name(image.name.replace("CODE", "VARS", 1))
    if not variables.is_file():
        raise RuntimeError(f"OVMF requires its variables template: {variables}")
    writable = directory / "ovmf-vars.fd"
    shutil.copyfile(variables, writable)
    return [
        "-drive",
        f"if=pflash,format=raw,readonly=on,file={image}",
        "-drive",
        f"if=pflash,format=raw,file={writable}",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0000:01:00.0")
    parser.add_argument(
        "--firmware", type=Path, default=ROOT / "target/nvidia/firmware"
    )
    parser.add_argument(
        "--cubin", type=Path, default=ROOT / "target/nvidia/vector-add.cubin"
    )
    parser.add_argument(
        "--ovmf", type=Path, help="OVMF code image or combined firmware"
    )
    parser.add_argument("--memory-mib", type=int, default=2048)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument(
        "--boot-check", action="store_true", help="verify GPU output, then stop"
    )
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if (
        not 512 <= args.memory_mib <= 32768
        or not 1 <= args.cpus <= 64
        or args.timeout <= 0
    ):
        parser.error("use 512 to 32768 MiB, 1 to 64 CPUs, and a positive timeout")
    devices = vfio_devices(args.gpu)
    required_lock = (args.memory_mib + 128) * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if hard != resource.RLIM_INFINITY and hard < required_lock:
        parser.error(
            f"VFIO requires a locked-memory limit of at least {required_lock} bytes"
        )
    if soft != resource.RLIM_INFINITY and soft < required_lock:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (required_lock, hard))
    ovmf = args.ovmf or next(
        (
            path
            for path in (
                Path("/usr/share/OVMF/OVMF_CODE_4M.fd"),
                Path("/usr/share/OVMF/OVMF_CODE.fd"),
                Path("/usr/share/edk2/ovmf/OVMF_CODE.fd"),
                ROOT / "RELEASEX64_OVMF.fd",
            )
            if path.is_file()
        ),
        None,
    )
    if ovmf is None:
        parser.error("install OVMF or provide --ovmf")
    directory = ROOT / "target/nvidia/vm"
    directory.mkdir(parents=True, exist_ok=True)
    image = prepare(directory, args.firmware, args.cubin)
    serial = directory / "serial.log"
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
        *firmware_options(ovmf.resolve(), directory),
        "-drive",
        f"format=raw,file={image}",
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
        "-no-reboot",
    ]
    print(shlex.join(command), flush=True)
    serial.unlink(missing_ok=True)
    with (directory / "qemu.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log)
        try:
            if not args.boot_check:
                print(
                    f"NVIDIA_VM_RUNNING pid={process.pid} serial={serial}", flush=True
                )
                if process.wait():
                    raise RuntimeError((directory / "qemu.log").read_text())
                return
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                text = serial.read_text(errors="replace") if serial.is_file() else ""
                if "panicked at" in text or "NVIDIA initialization failed:" in text:
                    raise RuntimeError(f"guest failed: {serial}")
                if "NVIDIA_KERNEL_OK vector-add count=257 backend=nvidia" in text:
                    print(f"NVIDIA_DEMO_OK serial={serial}", flush=True)
                    return
                if process.poll() is not None:
                    raise RuntimeError((directory / "qemu.log").read_text())
                time.sleep(0.1)
            raise TimeoutError(f"NVIDIA demo timed out: {serial}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
