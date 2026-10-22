#!/usr/bin/env python3
"""Boot D3OS and verify native inference through the normal shell and syscalls."""

import argparse
import errno
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
        # D3OS currently decodes a German keyboard layout. QEMU key names use US positions.
        for char in command:
            key = {" ": "spc", "-": "slash", "y": "z", "z": "y"}.get(char, char)
            self.execute("human-monitor-command", {"command-line": f"sendkey {key} 30"})
            time.sleep(0.06)
        self.execute("human-monitor-command", {"command-line": "sendkey ret 30"})

    def close(self):
        self.file.close()
        self.socket.close()


def large_image(stage, image):
    # towbootctl 0.10 reserves too little FAT metadata space for a GiB-sized initrd.
    # Build a GPT/EFI image with explicit headroom, streaming files through mtools.
    for command in ("sfdisk", "mformat", "mmd", "mcopy"):
        if shutil.which(command) is None:
            raise RuntimeError(
                f"large model images require {command}; install util-linux and mtools"
            )
    files = [stage / name for name in ("kernel.elf", "initrd.tar", "towboot.toml")]
    mib = 1024 * 1024
    sectors = (
        (sum(path.stat().st_size for path in files) + 64 * mib + mib - 1) // mib
    ) * 2048
    with image.open("wb") as file:
        file.truncate((sectors + 4096) * 512)
    subprocess.run(
        ["sfdisk", "--quiet", "--no-reread", "--no-tell-kernel", str(image)],
        input=f"label: gpt\n\nstart=2048,size={sectors},type=U\n",
        text=True,
        check=True,
    )
    volume = f"{image}@@{mib}"
    subprocess.run(
        ["mformat", "-i", volume, "-T", str(sectors), "-F", "::"], check=True
    )
    subprocess.run(["mmd", "-i", volume, "::/EFI", "::/EFI/BOOT"], check=True)
    bootloader = stage / "bootx64.efi"
    bootloader.unlink(missing_ok=True)
    subprocess.run(
        [str(ROOT / "towbootctl"), "extract", str(bootloader), "--x86-64"], check=True
    )
    subprocess.run(
        ["mcopy", "-i", volume, str(bootloader), "::/EFI/BOOT/BOOTX64.EFI"], check=True
    )
    subprocess.run(["mcopy", "-i", volume, *map(str, files), "::/"], check=True)


def prepare(directory, mode, qwen_assets):
    stage = directory / "boot"
    initrd = stage / "initrd"
    (initrd / "bin").mkdir(parents=True, exist_ok=True)
    for name in ("infer", "shell", "terminal_emulator"):
        shutil.copyfile(ROOT / "loader/initrd/bin" / name, initrd / "bin" / name)
    if mode != "self-test":
        assets = initrd / "usr/share/inference"
        assets.mkdir(parents=True, exist_ok=True)
        source = qwen_assets if mode.startswith("qwen") else ROOT / "target/inference"
        names = (
            (
                "qwen3.5-0.8b.d3t",
                "qwen3.5.bpe",
                "qwen-fixture.d3t",
                "tokenizer-fixture.bin",
            )
            if mode.startswith("qwen")
            else ("alexnet.d3t", "dog.d3t")
        )
        for name in names:
            destination = assets / name
            destination.unlink(missing_ok=True)
            try:
                os.link(source / name, destination)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(source / name, destination)
    with tarfile.open(
        stage / "initrd.tar", "w", format=tarfile.USTAR_FORMAT
    ) as archive:
        archive.add(initrd / "bin", arcname="bin")
        if mode != "self-test":
            archive.add(initrd / "usr", arcname="usr")
    shutil.copyfile(ROOT / "loader/kernel.elf", stage / "kernel.elf")
    shutil.copyfile(ROOT / "loader/towboot.toml", stage / "towboot.toml")
    image = directory / "d3os.img"
    # towbootctl expects a new destination. The file is private to this test run.
    image.unlink(missing_ok=True)
    if (stage / "initrd.tar").stat().st_size > 512 * 1024 * 1024:
        large_image(stage, image)
        return image
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
    models = parser.add_mutually_exclusive_group()
    models.add_argument(
        "--alexnet",
        action="store_true",
        help="include pretrained weights and verify every AlexNet layer",
    )
    models.add_argument(
        "--qwen", action="store_true", help="generate text with pretrained Qwen3.5-0.8B"
    )
    models.add_argument(
        "--qwen-verify",
        action="store_true",
        help="verify all Qwen layers, logits, and tokenizer fixtures",
    )
    parser.add_argument("--qwen-assets", type=Path, default=ROOT / "target/qwen")
    parser.add_argument("--bios", type=Path, help="OVMF firmware image")
    parser.add_argument(
        "--accel",
        choices=("kvm", "tcg"),
        default="kvm" if os.access("/dev/kvm", os.R_OK | os.W_OK) else "tcg",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--repeat", type=int, default=1, help="repeat the native command in the same VM"
    )
    parser.add_argument(
        "--gdb-port", type=int, help="start paused for GDB on a local TCP port"
    )
    parser.add_argument(
        "--cpus", type=int, default=2, help="guest CPU count (default: two)"
    )
    args = parser.parse_args()
    mode = (
        "qwen-verify"
        if args.qwen_verify
        else "qwen"
        if args.qwen
        else "alexnet"
        if args.alexnet
        else "self-test"
    )
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not 1 <= args.repeat <= 100:
        parser.error("--repeat must be from 1 to 100")
    if args.gdb_port is not None and not 1 <= args.gdb_port <= 65535:
        parser.error("--gdb-port must be from 1 to 65535")
    cpus = args.cpus
    if not 1 <= cpus <= 64:
        parser.error("--cpus must be from 1 to 64")
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
        / (
            "inference-qemu-qwen"
            if mode.startswith("qwen")
            else f"inference-qemu-{mode}"
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    image = prepare(directory, mode, args.qwen_assets.resolve())
    serial = directory / (
        f"serial-{mode}-{args.accel}.log" if mode.startswith("qwen") else "serial.log"
    )
    monitor = directory / "qmp.sock"
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
        str(cpus),
        "-m",
        "12G" if mode.startswith("qwen") else "4G" if args.alexnet else "512M",
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
    expected = (
        f"INFERENCE_OK {mode} " if mode != "self-test" else "INFERENCE_OK self-test"
    )
    if args.gdb_port is not None:
        command.extend(["-gdb", f"tcp:127.0.0.1:{args.gdb_port}", "-S"])
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
            qmp.type_command(f"infer {mode}" if mode != "self-test" else "infer")
            completed = 0
            while time.monotonic() < deadline:
                text = serial.read_text(errors="replace")
                if "INFERENCE_FAIL" in text or "panicked at" in text:
                    raise RuntimeError(f"D3OS inference failed. Read {serial}")
                count = text.count(expected)
                if count > completed and count < args.repeat:
                    completed = count
                    time.sleep(1)
                    qmp.type_command(
                        f"infer {mode}" if mode != "self-test" else "infer"
                    )
                if count >= args.repeat:
                    print(
                        "\n".join(
                            line
                            for line in text.splitlines()
                            if "INFERENCE_" in line
                            or "class=" in line
                            or "QWEN_" in line
                            or "infer.rs@" in line
                        )
                    )
                    print(
                        f"QEMU_INFERENCE_OK accelerator={args.accel} cpus={cpus} runs={count} serial={serial}"
                    )
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
