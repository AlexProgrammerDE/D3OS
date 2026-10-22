#!/usr/bin/env python3
"""Prepare an isolated Linux reference guest from installed kernel/driver files."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import stat
import subprocess
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]


def copy_file(root, source, destination=None):
    source = Path(source)
    target = root / str(destination or source).lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    shutil.copy2(source, target)
    target.chmod(source.stat().st_mode & 0o777)


def executable(root, name):
    source = Path(shutil.which(name) or name)
    copy_file(root, source, f"/bin/{name}")
    libraries = subprocess.check_output(["ldd", str(source)], text=True)
    for library in re.findall(r"/[^\s()]+", libraries):
        copy_file(root, library)


def kernel_benchmark(root):
    destination = root / "benchmark"
    subprocess.run(
        [
            "gcc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(ROOT / "tools/nvidia/linux-kernel-bench.c"),
            "-ldl",
            "-lm",
            "-o",
            str(destination),
        ],
        check=True,
    )
    dependencies = subprocess.check_output(["ldd", str(destination)], text=True)
    for library in re.findall(r"/[^\s()]+", dependencies):
        copy_file(root, library)
    for name in ("libcublas.so.12", "libcublasLt.so.12"):
        source = Path("/usr/local/lib/ollama/cuda_v12") / name
        copy_file(root, source, Path("/lib64") / name)
        dependencies = subprocess.check_output(["ldd", str(source)], text=True)
        for library in re.findall(r"/[^\s()]+", dependencies):
            copy_file(root, library)
    copy_file(root, ROOT / "target/nvidia/kernels/inference.cubin", "/inference.cubin")


def llama_benchmark(root, directory, model):
    environment = dict(
        os.environ, LD_LIBRARY_PATH=f"{directory}:/usr/local/lib/ollama/cuda_v12"
    )
    for source in directory.iterdir():
        if source.is_file() and (".so" in source.name or source.name == "llama-bench"):
            copy_file(root, source, Path("/bin") / source.name)
            dependencies = subprocess.check_output(
                ["ldd", str(source)], text=True, env=environment
            )
            for library in re.findall(r"/[^\s()]+", dependencies):
                copy_file(root, library, Path("/lib64") / Path(library).name)
    copy_file(root, model, "/model.gguf")
    benchmark = root / "benchmark"
    benchmark.write_text("""#!/bin/bash
set -eux
export LD_LIBRARY_PATH=/bin:/lib64
/bin/llama-bench -m /model.gguf -p 22 -n 64 -r 3 -ngl 99 -t 4 -o json
echo LINUX_LLAMA_BENCH_OK
""")
    benchmark.chmod(0o755)


def doom_benchmark(root):
    directory = ROOT / "os/application/doom"
    config = tomllib.loads((directory / "Makefile.toml").read_text())
    sources = [
        arg.replace("${CARGO_MAKE_WORKING_DIRECTORY}", str(directory)).replace(
            "${SOURCE_DIRECTORY}", str(directory / "src")
        )
        for arg in config["tasks"]["compile-doom"]["args"]
        if arg.endswith(".c") and "/libc/" not in arg
    ]
    binary = root / "bin/doom-linux"
    subprocess.run(
        [
            "gcc",
            "-std=gnu99",
            "-O3",
            "-march=x86-64-v3",
            "-D_DEFAULT_SOURCE",
            "-I",
            str(directory / "doomgeneric/doomgeneric"),
            *sources,
            str(ROOT / "tools/nvidia/linux-doom-bench.c"),
            "-Wl,--wrap=I_Error",
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
    )
    dependencies = subprocess.check_output(["ldd", str(binary)], text=True)
    for library in re.findall(r"/[^\s()]+", dependencies):
        copy_file(root, library)
    copy_file(root, ROOT / "target/doom/freedoom1.wad", "/doom.wad")
    benchmark = root / "benchmark"
    benchmark.write_text("""#!/bin/bash
/bin/doom-linux -iwad /doom.wad -timedemo demo1 -nogui
# The pinned engine reports timedemo completion through I_Error and exit(-1).
[ "$?" = 255 ] || exit 1
echo LINUX_DOOM_BENCH_OK
""")
    benchmark.chmod(0o755)


def modules(root, kernel, names):
    directory = Path("/lib/modules") / kernel
    for metadata in directory.glob("modules.*"):
        if metadata.is_file():
            copy_file(root, metadata)
    for name in names:
        dependencies = subprocess.check_output(
            ["modprobe", "--set-version", kernel, "--show-depends", name], text=True
        )
        for path in re.findall(r"^insmod (\S+)", dependencies, re.MULTILINE):
            copy_file(root, path)


def archive(root, output):
    # newc metadata creates device nodes inside the guest without requiring
    # mknod or root privileges on the host.
    with output.open("wb") as file:

        def entry(name, mode, data=b"", major=0, minor=0):
            name = name.encode() + b"\0"
            fields = (1, mode, 0, 0, 1, 0, len(data), 0, 0, major, minor, len(name), 0)
            file.write(
                b"070701" + b"".join(f"{field:08x}".encode() for field in fields)
            )
            file.write(name)
            file.write(b"\0" * (-file.tell() % 4))
            file.write(data)
            file.write(b"\0" * (-file.tell() % 4))

        for path in sorted(root.rglob("*")):
            mode = path.stat().st_mode
            entry(
                str(path.relative_to(root)),
                mode,
                path.read_bytes() if path.is_file() else b"",
            )
        entry("dev/console", stat.S_IFCHR | 0o600, major=5, minor=1)
        entry("dev/null", stat.S_IFCHR | 0o666, major=1, minor=3)
        entry("TRAILER!!!", 0)


def prepare(directory, kernel):
    bootstrap, root = directory / "bootstrap", directory / "root"
    for target in (bootstrap, root):
        target.mkdir(parents=True, exist_ok=True)
        for name in ("dev", "proc", "sys", "tmp", "newroot"):
            (target / name).mkdir(exist_ok=True)
        for name in ("bash", "mount", "modprobe", "mkdir", "switch_root"):
            executable(target, name)
        modules(target, kernel, ("virtio_pci", "9p", "9pnet_virtio"))
    for name in ("nvidia-smi", "mknod", "awk", "cat", "sleep"):
        executable(root, name)
    modules(root, kernel, ("nvidia", "nvidia_uvm"))
    subprocess.run(
        [
            "gcc",
            "-nostdlib",
            "-static",
            "-no-pie",
            "-Wl,--build-id=none,-z,noexecstack",
            str(ROOT / "tools/nvidia/linux-poweroff.S"),
            "-o",
            str(root / "bin/poweroff-guest"),
        ],
        check=True,
    )
    for library in ("libcuda.so.1", "libnvidia-ml.so.1"):
        copy_file(root, Path("/usr/lib64") / library, Path("/lib64") / library)
    firmware = subprocess.check_output(
        ["modinfo", "-k", kernel, "-F", "firmware", "nvidia"], text=True
    )
    for name in firmware.splitlines():
        if "ga10x" in name:
            copy_file(
                root, Path("/usr/lib/firmware") / name, Path("/lib/firmware") / name
            )
    init = bootstrap / "init"
    init.write_text("""#!/bin/bash
set -eux
export PATH=/bin
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev
modprobe virtio_pci
modprobe 9pnet_virtio
modprobe 9p
mount -t 9p -o trans=virtio,version=9p2000.L reference /newroot
exec switch_root /newroot /init
""")
    init.chmod(0o755)
    init = root / "init"
    init.write_text("""#!/bin/bash
set -eux
export PATH=/bin
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev
mount -t tmpfs tmpfs /tmp
modprobe nvidia NVreg_EnableGpuFirmware=1
modprobe nvidia_uvm
mknod /dev/nvidiactl c 195 255
mknod /dev/nvidia0 c 195 0
uvm=$(awk '$2 == "nvidia-uvm" {print $1}' /proc/devices)
mknod /dev/nvidia-uvm c "$uvm" 0
nvidia-smi --query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.limit --format=csv
echo LINUX_NVIDIA_REFERENCE_READY
if [ -x /benchmark ]; then
    if ! /benchmark; then
        echo LINUX_NVIDIA_REFERENCE_FAILED
        exec /bin/poweroff-guest
    fi
fi
echo LINUX_NVIDIA_REFERENCE_DONE
exec /bin/poweroff-guest
""")
    init.chmod(0o755)
    archive(bootstrap, directory / "bootstrap.cpio")
    manifest = {
        "kernel": kernel,
        "nvidia": subprocess.check_output(
            ["modinfo", "-k", kernel, "-F", "version", "nvidia"], text=True
        ).strip(),
    }
    manifest["initrd_sha256"] = hashlib.sha256(
        (directory / "bootstrap.cpio").read_bytes()
    ).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=ROOT / "target/nvidia/linux-reference"
    )
    parser.add_argument("--kernel", default=platform.release())
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--kernel-benchmark", action="store_true")
    parser.add_argument("--llama-directory", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--memory-mib", type=int, default=4096)
    parser.add_argument("--doom-benchmark", action="store_true")
    args = parser.parse_args()
    affinity = ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
    print(f"NVIDIA_HOST_CPU_AFFINITY cpus={affinity}", flush=True)
    if (
        sum((args.kernel_benchmark, bool(args.llama_directory), args.doom_benchmark))
        > 1
    ):
        parser.error("select only one Linux reference workload")
    if bool(args.llama_directory) != bool(args.model):
        parser.error("--llama-directory and --model must be supplied together")
    if args.memory_mib <= 0:
        parser.error("--memory-mib must be positive")
    args.directory = args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    lock = (args.directory / "guest.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("this Linux guest directory is already in use")
    prepare(args.directory, args.kernel)
    if args.kernel_benchmark:
        kernel_benchmark(args.directory / "root")
    elif args.llama_directory:
        if not args.model:
            parser.error("--llama-directory requires --model")
        llama_benchmark(
            args.directory / "root",
            args.llama_directory.resolve(),
            args.model.resolve(),
        )
    elif args.doom_benchmark:
        doom_benchmark(args.directory / "root")
    elif (args.directory / "root/benchmark").exists():
        (args.directory / "root/benchmark").unlink()
    if args.prepare_only:
        return
    # Reuse the same VFIO validation as D3OS. This guest never rebinds the host.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "nvidia_qemu", ROOT / "tools/nvidia/run-qemu.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    devices = module.vfio_devices("0000:01:00.0")
    memory = args.memory_mib
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    required = (memory + 128) << 20
    if soft != resource.RLIM_INFINITY and soft < required:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (required, hard))
    command = [
        "qemu-system-x86_64",
        "-machine",
        "q35",
        "-accel",
        "kvm",
        "-cpu",
        "host",
        "-smp",
        "4",
        "-m",
        str(memory),
        "-kernel",
        f"/boot/vmlinuz-{args.kernel}",
        "-initrd",
        str(args.directory / "bootstrap.cpio"),
        "-append",
        "console=ttyS0 loglevel=5 rdinit=/init panic=-1",
        "-display",
        "none",
        "-nic",
        "none",
        "-fsdev",
        f"local,id=reference,path={args.directory / 'root'},security_model=none",
        "-device",
        "virtio-9p-pci,fsdev=reference,mount_tag=reference",
        "-device",
        "pcie-root-port,id=nvidia,chassis=1,slot=1",
        "-device",
        f"vfio-pci,host={devices[0]},bus=nvidia,addr=0.0,multifunction=on,rombar=0",
        "-device",
        f"vfio-pci,host={devices[1]},bus=nvidia,addr=0.1",
        "-serial",
        f"file:{args.directory / 'serial.log'}",
        "-no-reboot",
    ]
    with (args.directory / "qemu.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            if process.wait(timeout=300):
                raise RuntimeError(f"Linux reference failed: {args.directory}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
    text = (args.directory / "serial.log").read_text(errors="replace")
    if "LINUX_NVIDIA_REFERENCE_DONE" not in text or "Kernel panic" in text:
        raise RuntimeError(
            f"Linux GPU reference did not complete: {args.directory / 'serial.log'}"
        )
    if args.kernel_benchmark and "LINUX_KERNEL_BENCH_OK" not in text:
        raise RuntimeError("Linux kernel benchmark did not pass")
    if args.llama_directory and "LINUX_LLAMA_BENCH_OK" not in text:
        raise RuntimeError("Linux llama benchmark did not pass")
    if args.doom_benchmark and not re.search(
        r"LINUX_DOOM_ENGINE_REPORT timed \d+ gametics in \d+ realtics \([0-9.]+ fps\)",
        text,
    ):
        raise RuntimeError("Linux DOOM benchmark did not report a result")
    print("LINUX_NVIDIA_REFERENCE_OK", flush=True)


if __name__ == "__main__":
    main()
