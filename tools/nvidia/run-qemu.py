#!/usr/bin/env python3
"""Boot D3OS with an already reserved NVIDIA GPU and its audio function."""

import argparse
import importlib.util
import json
import os
import re
import resource
import shlex
import struct
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
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument(
        "--firmware", type=Path, help="stage pinned NVIDIA firmware from this directory"
    )
    parser.add_argument(
        "--kernels",
        type=Path,
        help="stage portable Tile IR or PTX kernels for compilation inside D3OS",
    )
    parser.add_argument(
        "--compiler-toolkit",
        type=Path,
        default=ROOT / "target/nvidia/cutile-toolkit/packages/nvidia/cu13",
        help="pinned CUDA compiler tools to run inside D3OS",
    )
    parser.add_argument(
        "--inference",
        choices=(
            "cpu-self-test",
            "burn-self-test",
            "fwcfg-self-test",
            "gpu-mnist-train",
            "gpu-mnist-infer",
            "gpu-self-test",
            "gpu-kernel-bench",
            "gpu-alexnet",
            "gpu-alexnet-bench",
            "gpu-qwen",
            "gpu-qwen-verify",
            "gpu-qwen-bench",
            "gpu-qwen-gguf",
            "gpu-qwen-gguf-bench",
            "gpu-qwen-gguf-verify",
            "gpu-doom",
            "gpu-doom-verify",
            "gpu-doom-bench",
        ),
        default="cpu-self-test",
        help="native workload to validate during --boot-check",
    )
    parser.add_argument("--memory-mib", type=int, default=4096)
    parser.add_argument(
        "--qwen-gguf-parts",
        type=Path,
        help="firmware parts from prepare-qwen-shards.py",
    )
    parser.add_argument(
        "--qwen-device-timing",
        action="store_true",
        help="collect per-kernel GPU timestamps in a separate large-model pass or small-model run 1",
    )
    parser.add_argument(
        "--qwen-long-bench",
        action="store_true",
        help="generate 4096 tokens per Qwen benchmark run with a 4352-token cache",
    )
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument(
        "--fwcfg-file",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="expose a read-only file at /fwcfg/NAME",
    )
    parser.add_argument("--display", choices=("none", "gtk", "sdl"), default="none")
    parser.add_argument(
        "--boot-check",
        action="store_true",
        help="verify native device boot and the selected inference command, then stop",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--compile-timeout",
        type=float,
        default=600,
        help="separate boot and GPU compilation budget in seconds",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gdb-port", type=int, help="pause at reset for a local GDB connection"
    )
    args = parser.parse_args()
    firmware_files = {}
    if args.qwen_gguf_parts:
        if not args.inference.startswith("gpu-qwen-gguf"):
            parser.error("--qwen-gguf-parts requires a GGUF inference workload")
        directory = args.qwen_gguf_parts.resolve()
        index = (directory / "model.index").read_bytes()
        if len(index) != 60 or index[:8] != b"D3MODEL\0":
            parser.error("invalid model shard index")
        version, part_bytes, length, parts = struct.unpack_from("<IIQI", index, 8)
        if (
            version != 1
            or not (1 << 20) <= part_bytes <= (1 << 30)
            or part_bytes & (part_bytes - 1)
            or not 1 <= parts <= 64
            or (length + part_bytes - 1) // part_bytes != parts
        ):
            parser.error("invalid model shard extents")
        firmware_files["model.index"] = directory / "model.index"
        for part in range(parts):
            path = directory / f"model.{part:03}"
            if not path.is_file() or path.stat().st_size != min(
                part_bytes, length - part * part_bytes
            ):
                parser.error(f"missing or truncated model part: {path}")
            firmware_files[path.name] = path
        if args.inference == "gpu-qwen-gguf-bench":
            reference = directory / "model.reference"
            if (
                not reference.is_file()
                or not 64 <= reference.stat().st_size <= 0xFFFFFFFF
            ):
                parser.error(
                    "GGUF benchmark requires the prepared Linux logit reference"
                )
            firmware_files[reference.name] = reference
        if args.inference == "gpu-qwen-gguf-verify":
            fixture = directory / "model.fixture"
            if not fixture.is_file() or not 16 <= fixture.stat().st_size <= 0xFFFFFFFF:
                parser.error(
                    "GGUF verification requires the prepared full layer fixture"
                )
            firmware_files[fixture.name] = fixture
    elif args.inference.startswith("gpu-qwen-gguf"):
        parser.error("GGUF inference requires --qwen-gguf-parts")
    for argument in args.fwcfg_file:
        name, separator, filename = argument.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,45}", name):
            parser.error("--fwcfg-file requires NAME=PATH with a short ASCII filename")
        path = Path(filename).resolve()
        if (
            name in firmware_files
            or not path.is_file()
            or path.stat().st_size > 0xFFFFFFFF
        ):
            parser.error(
                "firmware filenames must be unique and files must be smaller than 4 GiB"
            )
        firmware_files[name] = path
    if args.qwen_device_timing and args.inference not in (
        "gpu-qwen-bench",
        "gpu-qwen-gguf-bench",
    ):
        parser.error("--qwen-device-timing requires a Qwen benchmark")
    if args.qwen_long_bench and args.inference != "gpu-qwen-bench":
        parser.error("--qwen-long-bench requires --inference gpu-qwen-bench")
    affinity = ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
    print(f"NVIDIA_HOST_CPU_AFFINITY cpus={affinity}", flush=True)
    if (
        not 512 <= args.memory_mib <= 32768
        or not 1 <= args.cpus <= 64
        or args.timeout <= 0
        or args.compile_timeout <= 0
    ):
        parser.error("use 512 to 32768 MiB, 1 to 64 CPUs, and a positive timeout")
    devices = vfio_devices(args.gpu)
    if args.kernels and not args.firmware:
        parser.error("--kernels requires --firmware")
    if args.inference not in (
        "cpu-self-test",
        "fwcfg-self-test",
        "burn-self-test",
    ) and (
        not args.kernels
        or not any(
            (args.kernels / f"inference.{suffix}").is_file()
            for suffix in ("tilebc", "ptx")
        )
    ):
        parser.error(
            "GPU inference requires --kernels with inference.tilebc or inference.ptx"
        )
    if args.image and args.firmware:
        parser.error("use either --image or --firmware")
    if not args.firmware:
        args.image = args.image or ROOT / "d3os.img"
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
    if args.firmware:
        names = ("gsp-570.144.bin", "bootloader-570.144.bin", "booter_load-570.144.bin")
        assets = [
            (args.firmware / name, Path("usr/share/nvidia") / name) for name in names
        ]
        if args.kernels:
            spec = importlib.util.spec_from_file_location(
                "guest_compiler", Path(__file__).with_name("guest-compiler.py")
            )
            compiler = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(compiler)
            assets.extend(
                compiler.prepare(
                    args.kernels.resolve(), args.compiler_toolkit.resolve(), directory
                )
            )
        if args.inference.startswith("gpu-doom"):
            assets.extend(
                [
                    (ROOT / "loader/initrd/bin/doom", Path("bin/doom")),
                    (ROOT / "target/doom/freedoom1.wad", Path("usr/doom.wad")),
                    (
                        ROOT / "target/doom/COPYING.txt",
                        Path("usr/share/doom/COPYING.txt"),
                    ),
                ]
            )
        for source, _ in assets:
            if not source.is_file():
                parser.error(f"firmware file does not exist: {source}")
        if args.dry_run:
            args.image = directory / "d3os.img"
        else:
            mode = "self-test"
            if args.inference.startswith("gpu-qwen-gguf"):
                assets.append(
                    (
                        ROOT / "target/qwen/qwen3.5.bpe",
                        Path("usr/share/inference/qwen3.5.bpe"),
                    )
                )
            elif args.inference.startswith("gpu-qwen"):
                mode = "qwen"
            elif args.inference.startswith("gpu-alexnet"):
                mode = "alexnet"
            args.image = harness.prepare(
                directory,
                mode,
                ROOT / "target/qwen",
                assets,
            )
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
        args.display,
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
    for name, path in firmware_files.items():
        filename = str(path).replace(",", ",,")
        command.extend(["-fw_cfg", f"name=opt/d3os/{name},file={filename}"])
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
            compilation_deadline = time.monotonic() + args.compile_timeout
            deadline = (
                time.monotonic()
                + args.timeout
                + (args.compile_timeout if args.kernels else 0)
            )
            qmp = harness.Qmp(monitor, process, deadline)
            started = False
            controls = False
            compilation_observed = None
            compilation_reported = False
            while time.monotonic() < deadline:
                text = serial.read_text(errors="replace") if serial.is_file() else ""
                if (
                    args.kernels
                    and compilation_observed is None
                    and "NVIDIA_COMPILE_START os=d3os" in text
                    and "NVIDIA_COMPILE_READY os=d3os" not in text
                ):
                    compilation_observed = time.monotonic()
                if (
                    compilation_observed is not None
                    and not compilation_reported
                    and "NVIDIA_COMPILE_READY os=d3os" in text
                ):
                    elapsed = (time.monotonic() - compilation_observed) * 1000
                    print(
                        f"NVIDIA_COMPILE_HOST_PHASE os=d3os observed_ms={elapsed:.3f} includes_gpu_validation=true source=serial_markers",
                        flush=True,
                    )
                    compilation_reported = True
                if (
                    args.kernels
                    and not started
                    and time.monotonic() > compilation_deadline
                ):
                    raise TimeoutError(f"D3OS GPU compilation timed out: {serial}")
                if (
                    "panicked at" in text
                    or "INFERENCE_FAIL" in text
                    or "NVIDIA initialization failed:" in text
                    or "NVIDIA_COMPILE_FAIL" in text
                    or "Linux process fault" in text
                    or "NVIDIA_DOOM_FAIL" in text
                    or "NVIDIA_DOOM_MISMATCH" in text
                ):
                    raise RuntimeError(f"guest failed: {serial}")
                if (
                    "name = shell" in text
                    and not started
                    and (not args.kernels or "NVIDIA_COMPILE_READY os=d3os" in text)
                ):
                    # Compile and validate before starting the workload budget.
                    deadline = time.monotonic() + args.timeout
                    time.sleep(1)
                    if args.inference.startswith("gpu-doom"):
                        flag = (
                            "--verify-gpu"
                            if args.inference.endswith("verify")
                            else "--gpu"
                        )
                        options = (
                            "-timedemo demo1"
                            if args.inference == "gpu-doom-bench"
                            else "-warp 1 1 -skill 2"
                        )
                        qmp.type_command(f"doom {flag} {options}")
                    else:
                        qmp.type_command(
                            "infer"
                            if args.inference == "cpu-self-test"
                            else f"infer {args.inference}"
                            + (" device-timing" if args.qwen_device_timing else "")
                            + (" long" if args.qwen_long_bench else "")
                        )
                    started = True
                if (
                    args.inference.startswith("gpu-doom")
                    and args.inference != "gpu-doom-bench"
                    and "NVIDIA_DOOM_FRAME_OK frame=60" in text
                    and not controls
                ):
                    for key in ("up", "left", "spc", "f5", "tab", "tab", "esc", "esc"):
                        qmp.execute(
                            "human-monitor-command",
                            {"command-line": f"sendkey {key} 300"},
                        )
                        time.sleep(0.4)
                    qmp.type_command("idbeholdi")
                    controls = True
                expected = {
                    "cpu-self-test": "INFERENCE_OK self-test backend=cpu",
                    "burn-self-test": "INFERENCE_OK burn-self-test backend=cpu gradients=ok aliasing=ok checkpoint=ok resume=ok",
                    "fwcfg-self-test": "INFERENCE_OK fwcfg-self-test",
                    "gpu-mnist-train": "INFERENCE_OK mnist mode=train backend=nvidia",
                    "gpu-mnist-infer": "INFERENCE_OK mnist mode=infer os=d3os backend=nvidia",
                    "gpu-self-test": "INFERENCE_OK self-test backend=nvidia",
                    "gpu-kernel-bench": "INFERENCE_BENCH_OK kernels backend=nvidia",
                    "gpu-alexnet": "INFERENCE_OK alexnet backend=nvidia layers=8",
                    "gpu-alexnet-bench": "INFERENCE_BENCH_OK alexnet backend=nvidia reference_layers=8",
                    "gpu-qwen": "INFERENCE_OK qwen backend=nvidia",
                    "gpu-qwen-verify": "INFERENCE_OK qwen-verify backend=nvidia positions=29 layers=24",
                    "gpu-qwen-bench": "INFERENCE_BENCH_OK qwen backend=nvidia reference_positions=29 reference_layers=24",
                    "gpu-qwen-gguf": "INFERENCE_OK qwen backend=nvidia",
                    "gpu-qwen-gguf-bench": "INFERENCE_BENCH_OK qwen-gguf backend=nvidia framework=burn os=d3os reference_tokens=128",
                    "gpu-qwen-gguf-verify": "INFERENCE_OK qwen-gguf-verify backend=nvidia framework=burn os=d3os positions=165",
                    "gpu-doom": "NVIDIA_DOOM_FRAME_OK frame=600",
                    "gpu-doom-verify": "NVIDIA_DOOM_FRAME_OK frame=600",
                    "gpu-doom-bench": "NVIDIA_DOOM_PROFILE_DONE",
                }[args.inference]
                if expected in text:
                    if args.inference == "gpu-doom-bench":
                        result = re.search(
                            r"DOOM_ENGINE_REPORT timed (\d+) gametics in (\d+) realtics \(([0-9.]+) fps\)",
                            text,
                        )
                        if not result:
                            raise RuntimeError(
                                "DOOM timedemo did not report a valid result"
                            )
                        print(
                            f"NVIDIA_DOOM_BENCH_OK gametics={result[1]} realtics={result[2]} fps={result[3]}",
                            flush=True,
                        )
                    elif args.inference.startswith("gpu-doom"):
                        for key in (173, 172, 32, 191, 9, 27):
                            for state in ("true", "false"):
                                if (
                                    not controls
                                    or f"NVIDIA_DOOM_INPUT key={key} pressed={state}"
                                    not in text
                                ):
                                    raise RuntimeError(
                                        f"DOOM did not consume key {key} pressed={state}"
                                    )
                        if not re.search(
                            r"NVIDIA_DOOM_FRAME_OK frame=600 .*low_primitives=[1-9]\d* fuzz_columns=[1-9]\d*",
                            text,
                        ):
                            raise RuntimeError(
                                "DOOM did not exercise low detail and invisibility rendering"
                            )
                        if args.inference == "gpu-doom-verify" and not re.search(
                            r"NVIDIA_DOOM_FRAME_OK frame=600 .*atlas_resets=[1-9]\d*",
                            text,
                        ):
                            raise RuntimeError(
                                "DOOM verification did not exercise texture atlas rollover"
                            )
                        if args.inference == "gpu-doom-verify":
                            purges = {
                                int(frame)
                                for frame in re.findall(
                                    r"NVIDIA_DOOM_CACHE_PURGE frame=(\d+)", text
                                )
                            }
                            if not set(range(60, 600, 60)).issubset(purges):
                                raise RuntimeError(
                                    "DOOM verification did not exercise all engine cache purges"
                                )
                        rates = [
                            int(n)
                            for n in re.findall(r"NVIDIA_DOOM_FPS fps=(\d+)", text)
                        ]
                        if not rates or max(rates[-3:]) < 30:
                            raise RuntimeError(
                                f"DOOM did not sustain a playable frame rate: {rates}"
                            )
                        qmp.execute(
                            "human-monitor-command",
                            {"command-line": f"screendump {directory / 'doom.ppm'}"},
                        )
                        qmp.execute(
                            "human-monitor-command", {"command-line": "sendkey f10 100"}
                        )
                        time.sleep(0.3)
                        qmp.type_command("y")
                        time.sleep(1)
                        qmp.type_command("infer gpu-self-test")
                        while time.monotonic() < deadline:
                            text = serial.read_text(errors="replace")
                            if (
                                "INFERENCE_OK self-test backend=nvidia" in text
                                and "NVIDIA_DOOM_EXIT frames=" in text
                            ):
                                break
                            if (
                                "NVIDIA_DOOM_FAIL" in text
                                or "INFERENCE_FAIL" in text
                                or "panicked at" in text
                            ):
                                raise RuntimeError(
                                    f"guest failed after DOOM exit: {serial}"
                                )
                            time.sleep(0.1)
                        else:
                            raise TimeoutError(
                                "DOOM exit did not return to a working GPU inference session"
                            )
                        print(
                            f"NVIDIA_DOOM_OK mode={args.inference} frames=600 fps={rates} cleanup=ok screenshot={directory / 'doom.ppm'}",
                            flush=True,
                        )
                    if (
                        args.kernels
                        and "NVIDIA_TRANSFER_OK cases=6 guards=ok bits=exact"
                        not in text
                    ):
                        raise RuntimeError(
                            "native coherent staging transfer verification did not succeed"
                        )
                    if (
                        args.kernels
                        and "NVIDIA_LOCAL_STACK_OK count=257 backend=nvidia" not in text
                    ):
                        raise RuntimeError(
                            "native GPU local-stack verification did not succeed"
                        )
                    if args.kernels and "NVIDIA_DEVICE_TIMER_OK" not in text:
                        raise RuntimeError(
                            "native GPU timestamp verification did not succeed"
                        )
                    if (
                        args.inference == "gpu-qwen"
                        and "INFERENCE_OK qwen backend=nvidia tokens=1 stop=EndOfSequence"
                        not in text
                    ):
                        raise RuntimeError(
                            "GPU generation did not match the default arithmetic fixture"
                        )
                    if "[0x10de:0x2757]" not in text or "[0x10de:0x22bb]" not in text:
                        raise RuntimeError(
                            "guest did not enumerate the target GPU and audio"
                        )
                    if "NVIDIA_PROBE_OK" not in text:
                        raise RuntimeError(
                            "native NVIDIA register probe did not succeed"
                        )
                    if args.firmware and "NVIDIA_QUEUE_OK" not in text:
                        raise RuntimeError(
                            "native GPU submission queue did not succeed"
                        )
                    if args.firmware and "NVIDIA_GSP_BOOT_OK" not in text:
                        raise RuntimeError("native GSP firmware boot did not succeed")
                    if args.kernels and "NVIDIA_COMPILE_READY os=d3os" not in text:
                        raise RuntimeError("in-guest GPU compilation did not succeed")
                    if args.kernels:
                        manifest = json.loads(
                            (directory / "guest-compiler-manifest.json").read_text()
                        )
                        for family in compiler.FAMILIES:
                            record = re.search(
                                rf"NVIDIA_COMPILE_MODULE os=d3os family={re.escape(family)} "
                                r"input=(?P<format>tile_ir|ptx) "
                                r"input_sha256=(?P<input_hash>[0-9a-f]{64}) "
                                r"target=sm_[0-9]+ compile_ms=(?P<total>[0-9]+) "
                                r"image_load_ms=(?P<load>[0-9]+) "
                                r"execute_ms=(?P<execute>[0-9]+) "
                                r"output_ms=(?P<output>[0-9]+) "
                                r"bytes=[0-9]+ sha256=[0-9a-f]{64} exit=0",
                                text,
                            )
                            suffix = (
                                "tilebc"
                                if record and record["format"] == "tile_ir"
                                else "ptx"
                            )
                            asset = f"usr/share/nvidia/{family}.{suffix}"
                            if (
                                not record
                                or record["input_hash"]
                                != manifest["assets"][asset]["sha256"]
                            ):
                                raise RuntimeError(
                                    f"guest did not compile the staged input: {family}"
                                )
                            if int(record["total"]) != sum(
                                int(record[phase])
                                for phase in ("load", "execute", "output")
                            ):
                                raise RuntimeError(
                                    f"guest compiler phase times disagree: {family}"
                                )
                    if args.kernels and "NVIDIA_KERNEL_OK" not in text:
                        raise RuntimeError(
                            "native GPU kernel verification did not succeed"
                        )
                    if args.kernels:
                        for symbol in ("tensor_matmul_f16", "tensor_matmul_bf16"):
                            if f"NVIDIA_TENSOR_OK kernel={symbol} cases=5" not in text:
                                raise RuntimeError(
                                    f"native tensor verification did not succeed: {symbol}"
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
        except BaseException:
            if qmp is not None:
                try:
                    registers = qmp.execute(
                        "human-monitor-command",
                        {"command-line": "info registers -a"},
                    )
                    (directory / "failure-registers.txt").write_text(registers)
                    qmp.execute(
                        "human-monitor-command",
                        {"command-line": f"screendump {directory / 'failure.ppm'}"},
                    )
                except (OSError, RuntimeError):
                    pass
            raise
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
