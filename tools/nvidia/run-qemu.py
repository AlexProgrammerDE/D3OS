#!/usr/bin/env python3
"""Boot D3OS with an already reserved NVIDIA GPU and its audio function."""

import argparse
import importlib.util
import json
import math
import os
import re
import resource
import shlex
import struct
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def host_cpu_ms(pid):
    # /proc/<pid>/stat sums CPU time across the process's threads. Split
    # after comm, which can contain spaces or parentheses, before indexing.
    fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) * 1000 / os.sysconf("SC_CLK_TCK")


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


def validate_module_admission(text, manifest, families, *, precompiled):
    """Bind each module record to its staged input, output, and loading mode."""
    expected_mode = "qualified_cache" if precompiled else "native_compile"
    if manifest.get("module_mode", "native_compile") != expected_mode:
        raise RuntimeError("staged module mode differs from the requested mode")
    marker = re.compile(r"(?<![A-Za-z0-9_])NVIDIA_COMPILE_MODULE ")
    pattern = re.compile(
        r"NVIDIA_COMPILE_MODULE os=d3os family=(?P<family>[a-z-]+) "
        r"input=(?P<format>tile_ir|ptx) "
        r"input_sha256=(?P<input_hash>[0-9a-f]{64}) "
        r"target=sm_89 compile_ms=(?P<total>[0-9]+) "
        r"image_load_ms=(?P<load>[0-9]+) execute_ms=(?P<execute>[0-9]+) "
        r"output_ms=(?P<output>[0-9]+) bytes=(?P<bytes>[0-9]+) "
        r"sha256=(?P<cubin_hash>[0-9a-f]{64}) exit=0"
        r"(?: mode=(?P<mode>native_compile|qualified_cache) "
        r"compiler_executed=(?P<executed>true|false))?\s*$"
    )
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = [line for line in clean.splitlines() if marker.search(line)]
    if len(lines) != len(families):
        raise RuntimeError("expected one admission record per kernel family")
    records = {}
    for line in lines:
        match = marker.search(line)
        record = pattern.fullmatch(line[match.start() :])
        if record is None:
            raise RuntimeError("malformed module admission record")
        record = record.groupdict()
        family = record["family"]
        if family not in families or family in records:
            raise RuntimeError("unknown or duplicate module family")
        suffix = "tilebc" if record["format"] == "tile_ir" else "ptx"
        asset = f"usr/share/nvidia/{family}.{suffix}"
        qualification = manifest["legacy_qualifications"][family]
        if (
            asset not in manifest["assets"]
            or record["input_hash"] != manifest["assets"][asset]["sha256"]
            or record["input_hash"] != qualification["source_sha256"]
            or record["cubin_hash"] != qualification["cubin_sha256"]
            or int(record["bytes"]) != qualification["cubin_bytes"]
        ):
            raise RuntimeError(f"module does not match staged qualification: {family}")
        if int(record["total"]) != sum(
            int(record[phase]) for phase in ("load", "execute", "output")
        ):
            raise RuntimeError(f"module phase times disagree: {family}")
        if precompiled:
            cached = manifest["assets"].get(f"usr/share/nvidia/{family}.cubin")
            if (
                record["mode"] != expected_mode
                or record["executed"] != "false"
                or int(record["load"]) != 0
                or int(record["execute"]) != 0
                or cached is None
                or cached["sha256"] != record["cubin_hash"]
                or cached["bytes"] != int(record["bytes"])
            ):
                raise RuntimeError(f"qualified cache was not used: {family}")
        elif record["mode"] not in (None, expected_mode) or record["executed"] not in (
            None,
            "true",
        ):
            raise RuntimeError(f"unexpected cache use in native compilation: {family}")
        records[family] = record
    return records


def validate_paired_q4(text):
    def records(marker):
        rows = []
        for payload in re.findall(
            rf"(?<![A-Za-z0-9_]){re.escape(marker)} ([^\r\n]+)", text
        ):
            tokens = None
            if marker == "NATIVE_LARGE_QWEN_PAIRED_TOKENS":
                payload, separator, encoded = payload.partition(" ids=")
                if not separator:
                    raise RuntimeError("paired Q4 token record has no sequence")
                tokens = json.loads(encoded)
            pairs = re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", payload)
            row = dict(pairs)
            if len(row) != len(pairs) or "status" in row:
                raise RuntimeError(
                    "paired Q4 record has duplicate or contradictory fields"
                )
            if tokens is not None:
                row["ids"] = tokens
            rows.append(row)
        return rows

    setup = records("NATIVE_LARGE_QWEN_PAIRED_SETUP")
    if (
        len(setup) != 1
        or any(
            setup[0].get(key) != value
            for key, value in {
                "variants": "2",
                "shared_weight_storage": "true",
                "weight_uploads": "0",
                "weight_downloads": "0",
                "ordinary_samples_per_variant": "4",
                "phase_completion": "device_sync",
            }.items()
        )
        or int(setup[0].get("resident_weight_bytes", "0")) <= 0
    ):
        raise RuntimeError("paired Q4 weights or measurement setup differs")
    roles = [
        "Control",
        "Candidate",
        "Control",
        "Candidate",
        "Candidate",
        "Control",
        "Candidate",
        "Control",
        "Control",
        "Candidate",
        "Control",
        "Candidate",
    ]
    kernels = {"Control": "Default", "Candidate": "Narrow"}
    benches = records("NATIVE_LARGE_QWEN_PAIRED_BENCH")
    resident = records("NATIVE_LARGE_QWEN_PAIRED_RESIDENT_OK")
    tokens = records("NATIVE_LARGE_QWEN_PAIRED_TOKENS")
    if len(benches) != 12 or len(resident) != 12 or len(tokens) != 12:
        raise RuntimeError("paired Q4 pass population differs")
    for run, (row, storage, token, role) in enumerate(
        zip(benches, resident, tokens, roles, strict=True)
    ):
        context = {
            "run": str(run),
            "variant": role,
            "q4_vector_kernel": kernels[role],
            "sample": f"Some({run - 2})" if 2 <= run < 10 else "None",
        }
        if any(
            row.get(key) != value
            or storage.get(key) != value
            or token.get(key) != value
            for key, value in context.items()
        ):
            raise RuntimeError("paired Q4 variant, order, or sample identity differs")
        sequence = token.get("ids")
        if (
            not isinstance(sequence, list)
            or len(sequence) != 128
            or any(
                type(value) is not int or not 0 <= value < 248320 for value in sequence
            )
            or sequence != tokens[0].get("ids")
        ):
            raise RuntimeError("paired Q4 token sequences differ or are invalid")
        checksum = 0
        for value in sequence:
            checksum = (checksum * 31 + value) & ((1 << 64) - 1)
        if checksum != int(row["checksum"]):
            raise RuntimeError("paired Q4 token checksum differs")
        expected = {
            "warm": "false" if run < 2 else "true",
            "instrumented": "true" if run >= 10 else "false",
            "prompt_tokens": "37",
            "decode_tokens": "128",
            "inputs": "linux_reference",
            "cache": "f32",
            "n_ctx": "2048",
            "eos": "continue_for_benchmark",
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise RuntimeError("paired Q4 workload or timing scope differs")
        ms, tps = int(row["decode_ms"]), float(row["decode_tps"])
        if (
            ms <= 0
            or int(row["prefill_ms"]) <= 0
            or not math.isfinite(tps)
            or abs(tps - 128000 / ms) > 0.00051
        ):
            raise RuntimeError("paired Q4 timing record is inconsistent")
        expected = {
            "prompt_id_uploads": "2",
            "decode_id_uploads": "128",
            "selection_read_calls": "128",
            "selection_bytes_per_read": "256",
            "selection_candidates_per_read": "32",
            "activation_uploads": "0",
            "activation_downloads": "0",
        }
        if any(storage.get(key) != value for key, value in expected.items()):
            raise RuntimeError("paired Q4 resident transfers differ")
    references = records("NATIVE_LARGE_QWEN_PAIRED_REFERENCE")
    if len(references) != 2 or [row.get("variant") for row in references] != [
        "Control",
        "Candidate",
    ]:
        raise RuntimeError("paired Q4 needs both complete numerical references")
    for row in references:
        expected = {
            "os": "d3os",
            "framework": "burn",
            "backend": "nvidia",
            "rows": "129",
            "vocab": "248320",
            "values": "32033280",
            "failures": "0",
            "absolute_tolerance": "0.002",
            "relative_tolerance": "0.0005",
            "inputs": "linux_reference",
            "q4_vector_kernel": kernels[row["variant"]],
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise RuntimeError("paired Q4 numerical policy or population differs")
        for key in ["max_absolute", "max_scaled_error", "squared_error"]:
            value = float(row[key])
            if (
                not math.isfinite(value)
                or value < 0
                or (key == "max_scaled_error" and value > 1)
            ):
                raise RuntimeError("paired Q4 numerical statistics are invalid")
        if not 0 <= int(row["selection_mismatches"]) <= 128:
            raise RuntimeError("paired Q4 selection count is invalid")
    for key in [
        "max_absolute",
        "max_scaled_error",
        "squared_error",
        "selection_mismatches",
    ]:
        if references[0][key] != references[1][key]:
            raise RuntimeError("paired Q4 changed numerical results")
    profiles = records("NATIVE_QWEN_PROJECTION_PROFILE")
    ledgers = []
    for role in ["Control", "Candidate"]:
        selected = [
            row
            for row in profiles
            if row.get("phase") == f"paired_{role}_decode_device"
        ]
        q4 = [row for row in selected if json.loads(row["parameters"])[3] == 12]
        expected_entry = "vector_project_q4k" + (
            "_narrow" if role == "Candidate" else ""
        )
        if not q4 or any(row["entry"] != expected_entry for row in q4):
            raise RuntimeError("paired Q4 did not execute the selected implementation")
        ledger = {}
        for row in selected:
            params = json.loads(row["parameters"])
            if row in q4 and (
                len(params) != (5 if role == "Candidate" else 4)
                or (len(params) == 5 and params[4] != 1)
            ):
                raise RuntimeError("paired Q4 executed an invalid selector packet")
            key = tuple(params[:4])
            if key in ledger or int(row["calls"]) <= 0:
                raise RuntimeError("paired Q4 projection ledger is invalid")
            ledger[key] = int(row["calls"])
        if sum(ledger.values()) != 51328:
            raise RuntimeError("paired Q4 projection count differs")
        ledgers.append(ledger)
    if ledgers[0] != ledgers[1]:
        raise RuntimeError("paired Q4 variants executed different projection workloads")
    footer = records("INFERENCE_BENCH_PAIR_OK qwen-gguf")
    if footer != [
        {
            "backend": "nvidia",
            "framework": "burn",
            "os": "d3os",
            "variants": "2",
            "reference_values_per_variant": "32033280",
        }
    ]:
        raise RuntimeError("paired Q4 completion record differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0000:01:00.0")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument(
        "--directory",
        type=Path,
        default=ROOT / "target/nvidia/vm",
        help="keep the staged image, compiler receipts, and capture in this directory",
    )
    parser.add_argument(
        "--firmware", type=Path, help="stage pinned NVIDIA firmware from this directory"
    )
    parser.add_argument(
        "--kernels",
        type=Path,
        help="stage portable Tile IR or PTX kernels for compilation inside D3OS",
    )
    parser.add_argument(
        "--precompiled-kernels",
        action="store_true",
        help="stage audited native modules with the same source and compiler bindings",
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
            "gpu-mnist-bench",
            "gpu-mnist-profile",
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
        "--qwen-prefill-q8",
        action="store_true",
        help="use experimental Q8_1 activation quantization for the large-model prefill benchmark",
    )
    parser.add_argument(
        "--qwen-prefill-q8-attention",
        action="store_true",
        help="extend experimental Q8_1 prefill to the qualified 32-row attention projections",
    )
    parser.add_argument(
        "--qwen-device-timing",
        action="store_true",
        help="collect per-kernel GPU timestamps in a separate large-model pass or small-model run 1",
    )
    parser.add_argument(
        "--qwen-paired-q4",
        action="store_true",
        help="compare accurate Q4 vector implementations with shared weights; requires --qwen-device-timing",
    )
    parser.add_argument(
        "--qwen-long-bench",
        action="store_true",
        help="generate 4096 tokens per Qwen benchmark run with a 4352-token cache",
    )
    parser.add_argument(
        "--qwen-firmware-model",
        action="store_true",
        help="stream the 0.8B benchmark model through firmware instead of the boot initrd",
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
        "--kernel-image",
        type=Path,
        help="use an alternate D3OS kernel ELF with --firmware",
    )
    parser.add_argument(
        "--application-binaries",
        type=Path,
        help="read infer, shell, and terminal_emulator from this directory with --firmware",
    )
    parser.add_argument(
        "--gdb-port", type=int, help="pause at reset for a local GDB connection"
    )
    parser.add_argument(
        "--pin-vcpus",
        action="store_true",
        help="pin guest CPUs individually to the caller's four host CPUs before boot",
    )
    args = parser.parse_args()
    if args.precompiled_kernels and not (args.firmware and args.kernels):
        parser.error("--precompiled-kernels requires --firmware and --kernels")
    if args.pin_vcpus and len(os.sched_getaffinity(0)) != 4:
        parser.error("--pin-vcpus requires an inherited affinity of four host CPUs")
    if args.kernel_image is not None and (
        not args.firmware or not args.kernel_image.is_file()
    ):
        parser.error("--kernel-image requires --firmware and a D3OS kernel ELF file")
    if args.application_binaries is not None and (
        not args.firmware
        or not all(
            (args.application_binaries / name).is_file()
            for name in ("infer", "shell", "terminal_emulator")
        )
    ):
        parser.error(
            "--application-binaries requires --firmware and infer, shell, terminal_emulator"
        )
    firmware_files = {}
    if args.qwen_firmware_model:
        if args.inference != "gpu-qwen-bench" or not args.firmware:
            parser.error(
                "--qwen-firmware-model requires --firmware and --inference gpu-qwen-bench"
            )
        model = ROOT / "target/qwen/qwen3.5-0.8b.d3t"
        if not model.is_file() or not 0 < model.stat().st_size <= 0xFFFFFFFF:
            parser.error(
                "the firmware benchmark model must exist and fit in a firmware file"
            )
        firmware_files[model.name] = model
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
    if args.qwen_prefill_q8 and args.inference != "gpu-qwen-gguf-bench":
        parser.error("--qwen-prefill-q8 requires --inference gpu-qwen-gguf-bench")
    if args.qwen_prefill_q8_attention and args.inference != "gpu-qwen-gguf-bench":
        parser.error(
            "--qwen-prefill-q8-attention requires --inference gpu-qwen-gguf-bench"
        )
    if args.qwen_prefill_q8 and args.qwen_prefill_q8_attention:
        parser.error("select one Qwen prefill quantization mode")
    if args.qwen_paired_q4 and (
        args.inference != "gpu-qwen-gguf-bench"
        or not args.qwen_device_timing
        or args.qwen_prefill_q8
        or args.qwen_prefill_q8_attention
    ):
        parser.error(
            "--qwen-paired-q4 requires the accurate GGUF benchmark and --qwen-device-timing"
        )
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
    directory = args.directory.resolve()
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
                    args.kernels.resolve(),
                    args.compiler_toolkit.resolve(),
                    directory,
                    precompiled=args.precompiled_kernels,
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
        if args.qwen_firmware_model:
            for name in ("qwen3.5.bpe", "qwen-fixture.d3t", "tokenizer-fixture.bin"):
                assets.append(
                    (ROOT / "target/qwen" / name, Path("usr/share/inference") / name)
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
            elif args.inference.startswith("gpu-qwen") and not args.qwen_firmware_model:
                mode = "qwen"
            elif args.inference.startswith("gpu-alexnet"):
                mode = "alexnet"
            args.image = harness.prepare(
                directory,
                mode,
                ROOT / "target/qwen",
                assets,
                kernel_image=args.kernel_image,
                application_binaries=args.application_binaries,
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
    elif args.pin_vcpus:
        command.append("-S")
    for name, path in firmware_files.items():
        filename = str(path).replace(",", ",,")
        command.extend(["-fw_cfg", f"name=opt/d3os/{name},file={filename}"])
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    serial.unlink(missing_ok=True)
    monitor.unlink(missing_ok=True)
    (directory / "vcpu-affinity.json").unlink(missing_ok=True)
    with (directory / "qemu.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log)
        qmp = None
        try:
            if args.pin_vcpus:
                qmp = harness.Qmp(
                    monitor, process, time.monotonic() + args.compile_timeout
                )
                mapping = {"execution_os": "d3os", **harness.pin_vcpus(qmp)}
                (directory / "vcpu-affinity.json").write_text(
                    json.dumps(mapping, indent=2) + "\n"
                )
                print(f"NVIDIA_VM_AFFINITY {json.dumps(mapping)}", flush=True)
                if args.gdb_port is None:
                    qmp.execute("cont")
            if not args.boot_check:
                if qmp is not None:
                    qmp.close()
                    qmp = None
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
            if qmp is None:
                qmp = harness.Qmp(monitor, process, deadline)
            started = False
            controls = False
            compilation_observed = None
            compilation_cpu = None
            compilation_reported = False
            while time.monotonic() < deadline:
                text = harness.read_serial(serial)
                if (
                    args.kernels
                    and compilation_observed is None
                    and "NVIDIA_COMPILE_START os=d3os" in text
                    and "NVIDIA_COMPILE_READY os=d3os" not in text
                ):
                    compilation_observed = time.monotonic()
                    compilation_cpu = host_cpu_ms(process.pid)
                if (
                    compilation_observed is not None
                    and not compilation_reported
                    and "NVIDIA_COMPILE_READY os=d3os" in text
                ):
                    elapsed = (time.monotonic() - compilation_observed) * 1000
                    cpu = host_cpu_ms(process.pid) - compilation_cpu
                    print(
                        f"NVIDIA_COMPILE_HOST_PHASE os=d3os observed_ms={elapsed:.3f} host_vm_cpu_ms={cpu:.3f} host_vm_cpu_percent={cpu / elapsed * 100:.3f} includes_gpu_validation=true source=serial_markers",
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
                            + (" q8-prefill" if args.qwen_prefill_q8 else "")
                            + (
                                " q8-attention-prefill"
                                if args.qwen_prefill_q8_attention
                                else ""
                            )
                            + (" paired-q4" if args.qwen_paired_q4 else "")
                            + (" long" if args.qwen_long_bench else "")
                            + (" firmware-model" if args.qwen_firmware_model else "")
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
                    "gpu-mnist-bench": "INFERENCE_OK mnist mode=bench os=d3os backend=nvidia",
                    "gpu-mnist-profile": "INFERENCE_OK mnist mode=profile os=d3os backend=nvidia",
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
                if args.qwen_paired_q4:
                    expected = "INFERENCE_BENCH_PAIR_OK qwen-gguf backend=nvidia framework=burn os=d3os variants=2 reference_values_per_variant=32033280"
                if expected in text:
                    if args.qwen_paired_q4:
                        validate_paired_q4(text)
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
                            text = harness.read_serial(serial)
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
                        validate_module_admission(
                            text,
                            manifest,
                            compiler.FAMILIES,
                            precompiled=args.precompiled_kernels,
                        )
                    if args.kernels and "NVIDIA_KERNEL_OK" not in text:
                        raise RuntimeError(
                            "native GPU kernel verification did not succeed"
                        )
                    if args.kernels and not re.search(
                        r"NVIDIA_RELOCATION_OK inputs=12 cases=4 comparisons=48 max_ulp=[01] "
                        r"table_addresses=gpu_virtual guards=ok tails=preserved source=preserved "
                        r"image_cache=reused backend=nvidia",
                        text,
                    ):
                        raise RuntimeError(
                            "native CUDA table relocation verification did not succeed"
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
