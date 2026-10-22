#!/usr/bin/env python3
"""Boot D3OS and verify native inference through the normal shell and syscalls."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Qmp:
    def __init__(self, path, process, deadline):
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.settimeout(5)
        while True:
            try:
                self.socket.connect(str(path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if process.poll() is not None or time.monotonic() > deadline:
                    self.socket.close()
                    raise RuntimeError("QEMU did not open its monitor")
                time.sleep(0.05)
        self.file = self.socket.makefile("rwb")
        self.file.readline()
        self.sequence = 0
        self.execute("qmp_capabilities")

    def execute(self, command, arguments=None):
        self.sequence += 1
        request = {"execute": command, "id": self.sequence}
        if arguments is not None:
            request["arguments"] = arguments
        self.file.write(json.dumps(request).encode() + b"\n")
        self.file.flush()
        while line := self.file.readline():
            response = json.loads(line)
            if response.get("id") == self.sequence:
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response.get("return")
        raise RuntimeError("QEMU monitor disconnected")

    def type_command(self, command):
        for char in command:
            key = "spc" if char == " " else char
            self.execute("human-monitor-command", {"command-line": f"sendkey {key} 30"})
            time.sleep(0.06)
        self.execute("human-monitor-command", {"command-line": "sendkey ret 30"})

    def close(self):
        self.file.close()
        self.socket.close()


def prepare(directory, alexnet):
    stage = directory / "boot"
    initrd = stage / "initrd"
    (initrd / "bin").mkdir(parents=True, exist_ok=True)
    for name in ("infer", "shell", "terminal_emulator"):
        shutil.copyfile(ROOT / "loader/initrd/bin" / name, initrd / "bin" / name)
    if alexnet:
        assets = initrd / "usr/share/inference"
        assets.mkdir(parents=True, exist_ok=True)
        for name in ("alexnet.d3t", "dog.d3t"):
            shutil.copyfile(ROOT / "target/inference" / name, assets / name)
    with tarfile.open(
        stage / "initrd.tar", "w", format=tarfile.USTAR_FORMAT
    ) as archive:
        archive.add(initrd / "bin", arcname="bin")
        if alexnet:
            archive.add(initrd / "usr", arcname="usr")
    shutil.copyfile(ROOT / "loader/kernel.elf", stage / "kernel.elf")
    shutil.copyfile(ROOT / "loader/towboot.toml", stage / "towboot.toml")
    image = directory / "d3os.img"
    # towbootctl expects a new destination. The file is private to this test run.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alexnet",
        action="store_true",
        help="include pretrained weights and verify every AlexNet layer",
    )
    parser.add_argument("--bios", type=Path, help="OVMF firmware image")
    parser.add_argument(
        "--accel",
        choices=("kvm", "tcg"),
        default="kvm" if os.access("/dev/kvm", os.R_OK | os.W_OK) else "tcg",
    )
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    bios = args.bios or next(
        (
            p
            for p in [
                ROOT / "RELEASEX64_OVMF.fd",
                Path("/usr/share/OVMF/OVMF_CODE.fd"),
                Path("/usr/share/edk2/ovmf/OVMF_CODE.fd"),
            ]
            if p.exists()
        ),
        None,
    )
    if bios is None:
        parser.error(
            "provide --bios with an OVMF firmware image, or run cargo make --no-workspace ovmf"
        )
    directory = (
        ROOT
        / "target"
        / ("inference-qemu-alexnet" if args.alexnet else "inference-qemu-self-test")
    )
    directory.mkdir(parents=True, exist_ok=True)
    image = prepare(directory, args.alexnet)
    serial, monitor = directory / "serial.log", directory / "qmp.sock"
    serial.unlink(missing_ok=True)
    monitor.unlink(missing_ok=True)
    deadline = time.monotonic() + args.timeout
    # Keep the usual guest devices. A network card avoids changing unrelated boot paths.
    command = [
        "qemu-system-x86_64",
        "-machine",
        "q35",
        "-accel",
        args.accel,
        "-cpu",
        "host" if args.accel == "kvm" else "Haswell,fsgsbase",
        "-smp",
        "2",
        "-m",
        "4G" if args.alexnet else "512M",
        "-bios",
        str(bios.resolve()),
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
        "-serial",
        f"file:{serial}",
        "-qmp",
        f"unix:{monitor},server=on,wait=off",
        "-no-reboot",
    ]
    expected = "INFERENCE_OK alexnet" if args.alexnet else "INFERENCE_OK self-test"
    qmp = None
    with (directory / "qemu.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log)
        try:
            qmp = Qmp(monitor, process, deadline)
            while time.monotonic() < deadline:
                text = serial.read_text(errors="replace") if serial.exists() else ""
                if "name = shell" in text:
                    break
                if process.poll() is not None:
                    raise RuntimeError(
                        f"QEMU exited before boot. Read {directory / 'qemu.log'}"
                    )
                time.sleep(0.05)
            else:
                raise TimeoutError(f"D3OS did not start its shell. Read {serial}")
            time.sleep(1)
            qmp.type_command("infer alexnet" if args.alexnet else "infer")
            while time.monotonic() < deadline:
                text = serial.read_text(errors="replace")
                if "INFERENCE_FAIL" in text or "panicked at" in text:
                    raise RuntimeError(f"D3OS inference failed. Read {serial}")
                if expected in text:
                    print(
                        "\n".join(
                            line
                            for line in text.splitlines()
                            if "INFERENCE_" in line or "class=" in line
                        )
                    )
                    print(f"QEMU_INFERENCE_OK accelerator={args.accel} serial={serial}")
                    return
                if process.poll() is not None:
                    raise RuntimeError(f"QEMU exited during inference. Read {serial}")
                time.sleep(0.1)
            raise TimeoutError(f"D3OS did not complete inference. Read {serial}")
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
