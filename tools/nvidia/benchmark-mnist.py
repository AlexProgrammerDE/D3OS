#!/usr/bin/env python3
"""Measure the same Burn MNIST training and inference work inside either GPU VM."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_FILES = {
    "mnist.train-images": "train-images-idx3-ubyte",
    "mnist.train-labels": "train-labels-idx1-ubyte",
    "mnist.test-images": "t10k-images-idx3-ubyte",
    "mnist.test-labels": "t10k-labels-idx1-ubyte",
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate(text, execution_os, artifacts, backend=None):
    contracts = re.findall(r"MNIST_BENCH_CONTRACT (\{[^\r\n]+\})", text)
    if len(contracts) != 1:
        raise RuntimeError("expected one complete MNIST benchmark contract")
    contract = json.loads(contracts[0])
    backend = backend or ("cuda" if execution_os == "linux" else "nvidia")
    if (
        contract["execution_os"] != execution_os
        or contract["backend"] != backend
        or contract["framework"] != "burn"
        or contract["precision"] != "fp32"
        or contract["cpu_isa"] != "x86-64-v3"
        or contract["training_samples"] != 60_000
        or contract["test_samples"] != 10_000
        or contract["warmup_steps"] != 4
        or contract["timed_steps"] != 64
        or contract["batch_size"] != 64
        or contract["repeats"] != 3
        or contract["inference_batches"] != [1, 64, 256]
    ):
        raise RuntimeError("unexpected MNIST benchmark work or backend")
    rows = []
    observed = set()
    for line in re.findall(r"MNIST_BENCH os=[^\r\n]+", text):
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if fields["os"] != execution_os or fields["backend"] != backend or fields["precision"] != "fp32":
            raise RuntimeError("MNIST result uses a different execution backend")
        mode, run, batch = fields["mode"], int(fields["run"]), int(fields["batch"])
        key = (mode, run, batch)
        if key in observed or not 0 <= run < 3:
            raise RuntimeError("duplicate or invalid MNIST repetition")
        observed.add(key)
        elapsed = float(fields["elapsed_ms"])
        rate = float(fields["images_per_second"])
        if not math.isfinite(elapsed) or elapsed <= 0 or not math.isfinite(rate):
            raise RuntimeError("invalid MNIST timing")
        samples = int(fields["samples"])
        if not math.isclose(rate, samples * 1000.0 / elapsed, rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError("MNIST throughput disagrees with elapsed time")
        if mode == "inference":
            if samples != 10_000 or int(fields["correct"]) < 9900 or batch not in (1, 64, 256):
                raise RuntimeError("incomplete or inaccurate MNIST inference")
        elif mode == "train":
            if (batch, samples, int(fields["warmup_steps"]), int(fields["timed_steps"]),
                int(fields["final_step"]), int(fields["final_epoch"]), int(fields["final_offset"])) != (64, 4096, 4, 64, 68, 0, 4352):
                raise RuntimeError("MNIST training progress differs from the contract")
        else:
            raise RuntimeError("unknown MNIST benchmark mode")
        rows.append(fields)
    expected = {("inference", run, batch) for run in range(3) for batch in (1, 64, 256)}
    expected.update(("train", run, 64) for run in range(3))
    if observed != expected:
        raise RuntimeError("missing MNIST benchmark repetitions")
    sizes = {f"inference-{batch}-logits.f32": 100_000 * 4 for batch in (1, 64, 256)}
    for run in range(3):
        sizes[f"train-{run}-losses.f32"] = 64 * 4
        sizes[f"train-{run}-logits.f32"] = 2560 * 4
        model = artifacts / f"train-{run}.model"
        if not model.is_file() or model.stat().st_size < 72:
            raise RuntimeError("missing trained MNIST artifact")
    for name, size in sizes.items():
        data = (artifacts / name).read_bytes()
        if len(data) != size or any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", data)):
            raise RuntimeError(f"incomplete or non-finite MNIST numerical reference: {name}")
    return contract, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--os", dest="execution_os", choices=("linux", "d3os"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "target/mnist")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cpu-reference", action="store_true", help="run a strict host CPU numerical oracle; never use its timings for GPU parity")
    parser.add_argument("--profile-only", action="store_true", help="collect separate native training CPU/queue/GPU diagnostics instead of throughput measurements")
    args = parser.parse_args()
    if len(os.sched_getaffinity(0)) != 4 or args.timeout < 60:
        parser.error("inherit exactly four host CPUs and allow at least 60 seconds")
    if args.cpu_reference and args.execution_os != "linux":
        parser.error("--cpu-reference requires --os linux")
    if args.profile_only and (args.execution_os != "d3os" or args.cpu_reference):
        parser.error("--profile-only requires --os d3os without --cpu-reference")
    model, data, output = args.model.resolve(), args.data.resolve(), args.directory.resolve()
    if not model.is_file():
        parser.error("--model must be an exported MNIST inference artifact")
    if output.exists():
        parser.error("use a new --directory to preserve previous evidence")
    output.mkdir(parents=True)
    staging = module("prepare_mnist", ROOT / "tools/inference/prepare-mnist.py")
    staging.prepare(data)
    inputs = {"mnist.model": model, **{name: data / source for name, source in DATA_FILES.items()}}
    manifest = {
        "execution_os": args.execution_os, "framework": "burn",
        "backend": "flex_cpu" if args.cpu_reference else "cuda" if args.execution_os == "linux" else "nvidia",
        "arithmetic_policy": "fp32" if args.cpu_reference else "cubecl_default_tf32_allowed" if args.execution_os == "linux" else "fp32_tf32x3",
        "memory_mib": None if args.cpu_reference else 4096, "vcpus": None if args.cpu_reference else 4,
        "host_affinity": sorted(os.sched_getaffinity(0)),
        "execution_location": "host" if args.cpu_reference else "vm",
        "host_training": args.cpu_reference,
        "reference_only": args.cpu_reference,
        "performance_comparison": not (args.cpu_reference or args.profile_only),
        "training_profile": args.profile_only,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "inputs": {name: {"bytes": path.stat().st_size, "sha256": digest(path)} for name, path in inputs.items()},
    }
    toolchain_file = ROOT / ("tools/mnist-benchmark/rust-toolchain.toml" if args.execution_os == "linux" else "rust-toolchain.toml")
    channel = tomllib.loads(toolchain_file.read_text())["toolchain"]["channel"]
    manifest["build_toolchain"] = {
        "channel": channel,
        "rustc": subprocess.check_output(["rustup", "run", channel, "rustc", "--version"], text=True).strip(),
    }
    if args.execution_os == "d3os":
        manifest["runtime"] = {name: digest(ROOT / name) for name in ("loader/kernel.elf", "loader/initrd/bin/infer")}
        (output / "runtime").mkdir()
        for name in ("loader/kernel.elf", "loader/initrd/bin/infer"):
            shutil.copyfile(ROOT / name, output / "runtime" / Path(name).name)
        kernels = ROOT / "target/nvidia/kernels"
        portable = json.loads((kernels / "manifest.json").read_text())
        manifest["portable_kernels"] = portable
        (output / "portable-kernels").mkdir()
        for name in ("manifest.json", *(entry["input"] for entry in portable["kernels"].values())):
            shutil.copyfile(kernels / name, output / "portable-kernels" / name)
    sources = (
        "os/library/ai/src/mnist.rs", "os/library/ai/src/mnist/training.rs",
        "os/library/ai/src/mnist/artifact.rs", "os/library/ai/src/mnist/data.rs",
        "os/application/infer/src/mnist.rs", "tools/mnist-benchmark/src/main.rs",
        "tools/mnist-benchmark/Cargo.toml", "tools/mnist-benchmark/Cargo.lock",
        "tools/mnist-benchmark/rust-toolchain.toml", "tools/mnist-benchmark/.cargo/config.toml",
        "tools/mnist-benchmark/src/bin/mnist-inspect-pack.rs", "rust-toolchain.toml", "d3os_application.json",
        "tools/nvidia/benchmark-mnist.py", "tools/nvidia/linux-reference.py",
        "tools/nvidia/run-mnist.py", "tools/nvidia/run-qemu.py",
        "os/library/burn-native/src/tensor.rs", "os/library/nvidia/src/compute.rs",
        "tools/gpu-kernels/src/matrix.rs", "tools/gpu-kernels/src/matrix.rs.in",
    )
    manifest["sources"] = {name: digest(ROOT / name) for name in sources}
    for name in sources:
        destination = output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))
    artifacts = output / "artifacts"
    if args.cpu_reference:
        guest = None
        binary = ROOT / "target/mnist-cpu-oracle/release/d3os-mnist-benchmark"
        manifest["binary_sha256"] = digest(binary)
        command = [str(binary), str(data), str(model), str(artifacts)]
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    elif args.execution_os == "linux":
        guest = output / "guest"
        command = [
            sys.executable, str(ROOT / "tools/nvidia/linux-reference.py"),
            "--directory", str(guest), "--mnist-benchmark", "--mnist-data", str(data),
            "--model", str(model), "--memory-mib", "4096", "--pin-vcpus", "--timeout", str(args.timeout),
        ]
    else:
        guest = ROOT / "target/nvidia/vm"
        command = [
            sys.executable, str(ROOT / "tools/nvidia/run-qemu.py"),
            "--firmware", str(ROOT / "target/nvidia/firmware"),
            "--kernels", str(ROOT / "target/nvidia/kernels"),
            "--memory-mib", "4096", "--pin-vcpus", "--boot-check",
            "--inference", "gpu-mnist-profile" if args.profile_only else "gpu-mnist-bench", "--timeout", str(args.timeout),
        ]
        for name, path in inputs.items():
            command.extend(["--fwcfg-file", f"{name}={path}"])
    with (output / "launcher.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=args.timeout if args.cpu_reference else None)
    for name in ("serial.log", "qemu.log", "vcpu-affinity.json", "guest-compiler-manifest.json"):
        if guest is not None and (guest / name).is_file():
            shutil.copyfile(guest / name, output / name)
    if result.returncode:
        raise RuntimeError(f"MNIST guest failed: {output / 'launcher.log'}")
    text = (output / ("launcher.log" if args.cpu_reference else "serial.log")).read_text(errors="replace")
    if args.profile_only:
        diagnostics = re.findall(r"MNIST_TRAIN_PROFILE [^\r\n]+", text)
        if len(diagnostics) != 2 or not all(f"instrumented={value}" in diagnostics[index] for index, value in enumerate(("false", "true"))):
            raise RuntimeError("missing ordinary or device-instrumented MNIST profile")
        records = [
            row for row in re.findall(r"NATIVE_(?:KERNEL|QUEUE|SERVICE)_PROFILE [^\r\n]+", text)
            if re.search(r"\bphase=mnist_train(?:_device)?(?:\s|$)", row)
        ]
        matrices = re.findall(r"MNIST_MATMUL_PROFILE [^\r\n]+", text)
        if len(matrices) != 18:
            raise RuntimeError("missing MNIST matrix shape diagnostics")
        (output / "results.json").write_text(json.dumps({**manifest, "diagnostics": diagnostics, "profiles": records, "matrices": matrices}, indent=2) + "\n")
        for line in diagnostics + matrices:
            print(line)
        return
    if not args.cpu_reference and args.execution_os == "linux":
        shutil.copytree(guest / "root/mnist-output", artifacts)
    elif not args.cpu_reference:
        artifacts.mkdir()
        runner = module("mnist_runner", ROOT / "tools/nvidia/run-mnist.py")
        _, pending = runner.extract_artifacts(text, artifacts)
        if pending:
            raise RuntimeError("interrupted MNIST reference export")
    contract, rows = validate(text, args.execution_os, artifacts, manifest["backend"])
    summary = {
        **manifest, "contract": contract, "measurements": rows,
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in sorted(artifacts.iterdir())},
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    for mode, batch in (("inference", 1), ("inference", 64), ("inference", 256), ("train", 64)):
        times = [float(row["elapsed_ms"]) for row in rows if row["mode"] == mode and int(row["batch"]) == batch]
        print(f"MNIST_BENCH_OK os={args.execution_os} backend={manifest['backend']} reference_only={args.cpu_reference} mode={mode} batch={batch} elapsed_ms={times} mean_ms={sum(times) / len(times):.6f}")


if __name__ == "__main__":
    main()
