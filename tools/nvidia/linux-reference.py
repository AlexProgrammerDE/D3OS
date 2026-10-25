#!/usr/bin/env python3
"""Prepare an isolated Linux reference guest from installed kernel/driver files."""

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import resource
import shlex
import shutil
import stat
import subprocess
import sys
import time
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


def stage_compiler(root, inputs=None):
    spec = importlib.util.spec_from_file_location(
        "guest_compiler", Path(__file__).with_name("guest-compiler.py")
    )
    compiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler)
    inputs = inputs or ROOT / "target/nvidia/kernels"
    assets = compiler.prepare(
        inputs,
        ROOT / "target/nvidia/cutile-toolkit/packages/nvidia/cu13",
        root.parent / "compiler",
        execution_os="linux",
    )
    for source, relative in assets:
        copy_file(root, source, relative)
    return compiler


def stage_burn_program(root, binary, destination):
    copy_file(root, binary, destination)
    dependencies = subprocess.check_output(["ldd", str(binary)], text=True)
    if "not found" in dependencies:
        raise RuntimeError("unresolved Linux Burn program library")
    for library in re.findall(r"/[^\s()]+", dependencies):
        copy_file(root, library)
    libdevice = ROOT / "target/nvidia/cutile-toolkit/packages/nvidia/cu13/nvvm/libdevice/libdevice.10.bc"
    copy_file(root, libdevice, "/usr/cuda/nvvm/libdevice/libdevice.10.bc")
    return libdevice


def burn_quantization(root, binary):
    libdevice = stage_burn_program(root, binary, "/burn-quantization")
    benchmark = root / "benchmark"
    benchmark.write_text("""#!/bin/bash
set -eux
export CUDA_PATH=/usr/cuda
export LD_LIBRARY_PATH=/lib64
export HOME=/tmp
export XDG_CACHE_HOME=/tmp/cache
export RUST_BACKTRACE=1
exec /burn-quantization
""")
    benchmark.chmod(0o755)
    sources = {
        "executable": binary,
        "runner": ROOT / "tools/mnist-benchmark/src/bin/quantization.rs",
        "launcher": Path(__file__).resolve(),
        "shared_validation": ROOT / "os/library/ai/src/validation/quantization.rs",
        "cargo_lock": ROOT / "tools/mnist-benchmark/Cargo.lock",
        "libdevice": libdevice,
    }
    (root.parent / "burn-quantization-manifest.json").write_text(json.dumps({
        "execution_os": "linux", "backend": "cuda GPU", "framework": "burn", "burn_version": "0.22.1",
        "purpose": "shared API conformance; no performance measurement",
        "inputs": {name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in sources.items()},
    }, indent=2) + "\n")


def mnist_benchmark(root, binary, data, model):
    if (root / "mnist-output").exists():
        raise RuntimeError("MNIST output already exists; use a fresh reference directory")
    libdevice = stage_burn_program(root, binary, "/mnist-benchmark")
    sources = {"mnist.model": model}
    for name in (
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
    ):
        copy_file(root, data / name, Path("/mnist-data") / name)
        sources[f"data/{name}"] = data / name
    copy_file(root, model, "/mnist.model")
    benchmark = root / "benchmark"
    benchmark.write_text("""#!/bin/bash
set -eux
export CUDA_PATH=/usr/cuda
export LD_LIBRARY_PATH=/lib64
export HOME=/tmp
export XDG_CACHE_HOME=/tmp/cache
exec /mnist-benchmark /mnist-data /mnist.model /mnist-output
""")
    benchmark.chmod(0o755)
    sources.update({
        "executable": binary,
        "cargo_lock": ROOT / "tools/mnist-benchmark/Cargo.lock",
        "source": ROOT / "tools/mnist-benchmark/src/main.rs",
        "libdevice": libdevice,
    })
    manifest = {
        "execution_os": "linux", "framework": "burn", "backend": "cuda",
        "compiler": "CubeCL default LLVM NVPTX",
        "inputs": {
            name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in sources.items()
        },
    }
    (root.parent / "mnist-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def compiler_benchmark(root, repeats, cpu_masks, inputs=None):
    stage_compiler(root, inputs)
    for name in ("date", "taskset", "sha256sum", "cp", "time"):
        executable(root, name)
    benchmark = root / "benchmark"
    benchmark.write_text(
        f"#!/bin/bash\nrepeats={repeats}\ncpu_masks={shlex.quote(' '.join(dict.fromkeys(cpu_masks)))}\n"
        + """
set -eux
capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits)
target=sm_${capability/./}
mkdir -p /tmp/cuda /tmp/kernels
cp -r /usr/cuda/. /tmp/cuda/
for input in /usr/share/nvidia/*.tilebc /usr/share/nvidia/*.ptx; do
    if [ -f "$input" ]; then cp "$input" /tmp/kernels/; fi
done
for cpus in $cpu_masks; do
    for ((run=0; run<repeats; run++)); do
        total_ms=0
        for family in vector-add tensor-matmul inference; do
            if [ -f "/tmp/kernels/$family.tilebc" ]; then
                tool=tileiras
                input="/tmp/kernels/$family.tilebc"
            else
                tool=ptxas
                input="/tmp/kernels/$family.ptx"
            fi
            start=$(date +%s%N)
            /bin/time -o /tmp/compiler-time -f "LINUX_COMPILE_CPU os=linux cpus=$cpus run=$run family=$family user_s=%U system_s=%S elapsed_s=%e max_rss_kib=%M major_faults=%F minor_faults=%R voluntary_switches=%w involuntary_switches=%c" taskset -c "$cpus" "/tmp/cuda/bin/$tool" --gpu-name="$target" --opt-level=3 "$input" -o /tmp/output.cubin
            end=$(date +%s%N)
            elapsed_ms=$(((end-start)/1000000))
            total_ms=$((total_ms+elapsed_ms))
            hash=$(sha256sum /tmp/output.cubin)
            echo "LINUX_COMPILE_MODULE os=linux cpus=$cpus run=$run family=$family target=$target compile_ms=$elapsed_ms sha256=${hash%% *} phase=load"
            cat /tmp/compiler-time
        done
        echo "LINUX_COMPILE_BENCH os=linux cpus=$cpus run=$run modules=3 compile_ms=$total_ms source=tmpfs phase=load"
    done
done
echo LINUX_COMPILER_BENCH_OK
"""
    )
    benchmark.chmod(0o755)


def kernel_benchmark(root):
    destination = root / "kernel-benchmark"
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
    compiler = stage_compiler(root)
    inputs = ROOT / "target/nvidia/kernels"
    portable = next(
        path
        for path in (inputs / "inference.tilebc", inputs / "inference.ptx")
        if path.is_file()
    )
    tool = "tileiras" if portable.suffix == ".tilebc" else "ptxas"
    benchmark = root / "benchmark"
    benchmark.write_text(f"""#!/bin/bash
set -eux
capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits)
target=sm_${{capability/./}}
SECONDS=0
/usr/cuda/bin/{tool} --gpu-name="$target" --opt-level=3 /usr/share/nvidia/{portable.name} -o /inference.cubin
echo "LINUX_COMPILE_READY target=$target compile_seconds=$SECONDS phase=load"
exec /kernel-benchmark
""")
    benchmark.chmod(0o755)
    manifest = {
        "source_sha256": hashlib.sha256(
            (ROOT / "tools/nvidia/linux-kernel-bench.c").read_bytes()
        ).hexdigest(),
        "binary_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "input": portable.name,
        "input_sha256": compiler.digest(portable),
        "code_generation_os": "linux",
    }
    (root.parent / "kernel-benchmark-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def llama_libraries(root, directory):
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


def quantized_profile(root, directories, nsight, inputs):
    target = nsight / "target/linux-desktop-glibc_2_11_3-x64"
    if not (target / "ncu").is_file():
        raise ValueError("Nsight directory must contain the pinned Linux x86-64 CLI")
    destination = root / "quantized-probe"
    source = ROOT / "tools/nvidia/linux-quantized-probe.c"
    subprocess.run([
        "gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        str(source), "-ldl", "-lm", "-o", str(destination),
    ], check=True)
    runtime = root / "opt/nsight/target/linux-desktop-glibc_2_11_3-x64"
    shutil.copytree(target, runtime, dirs_exist_ok=True)
    shutil.copytree(nsight / "sections", root / "opt/nsight/sections", dirs_exist_ok=True)
    environment = dict(os.environ, LD_LIBRARY_PATH=str(runtime))
    for binary in [destination, *runtime.iterdir()]:
        if not binary.is_file():
            continue
        dependencies = subprocess.check_output(["ldd", str(binary)], text=True, env=environment)
        if "not found" in dependencies:
            raise ValueError(f"unresolved profiler dependency: {binary}")
        for library in re.findall(r"/[^\s()]+", dependencies):
            if not Path(library).is_relative_to(runtime):
                copy_file(root, library)
    executable(root, "sha256sum")
    executable(root, "dmesg")
    for name in ("sh", "uname", "cp"):
        executable(root, name)
    copy_file(root, "/etc/os-release")
    (root / "etc/passwd").write_text("root:x:0:0:root:/tmp:/bin/bash\n")
    (root / "etc/group").write_text("root:x:0:\n")
    inputs = inputs or ROOT / "target/nvidia/kernels"
    stage_compiler(root, inputs)
    portable = inputs / "inference.tilebc"
    expected = json.loads((portable.parent / "manifest.json").read_text())["kernels"]["inference"]["host_audit"]["sha256"]
    sections = ("SpeedOfLight", "SchedulerStats", "WarpStateStats", "MemoryWorkloadAnalysis_Tables", "LaunchStats", "Occupancy")
    ncu = "/opt/nsight/target/linux-desktop-glibc_2_11_3-x64/ncu"
    commands = [
        "#!/bin/bash", "set -eux", "export HOME=/tmp", "export TMPDIR=/tmp",
        "export LD_LIBRARY_PATH=/opt/nsight/target/linux-desktop-glibc_2_11_3-x64:/lib64",
        "mkdir -p /dev/shm /profiles", "mount -t tmpfs tmpfs /dev/shm", "cd /profiles", "ulimit -c unlimited",
        "echo /tmp/core.%p >/proc/sys/kernel/core_pattern",
        f"{ncu} --version", 'capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits)',
        'target=sm_${capability/./}',
        '/usr/cuda/bin/tileiras --gpu-name="$target" --opt-level=3 /usr/share/nvidia/inference.tilebc -o /inference.cubin',
        'hash=$(sha256sum /inference.cubin)', f'test "${{hash%% *}}" = {shlex.quote(expected)}',
        'echo "LINUX_QUANTIZED_MODULE_SHA256 hash=${hash%% *} code_generation_os=linux"',
    ]
    cases = []
    for index, directory in enumerate(directories):
        manifest = json.loads((directory / "manifest.json").read_text())
        inner, outputs, format_code = (int(manifest[name]) for name in ("inner", "outputs", "format"))
        formats = {8: (32, 34), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210)}
        if format_code not in formats or inner <= 0 or outputs <= 0 or inner * outputs > 0xFFFFFFFF:
            raise ValueError("invalid quantized probe shape or format")
        block, block_bytes = formats[format_code]
        if inner % block:
            raise ValueError("quantized rows must contain complete blocks")
        sizes = {"weights.bin": inner * outputs // block * block_bytes, "input.bin": inner * 4, "reference.bin": outputs * 4}
        if manifest["files"].keys() != sizes.keys():
            raise ValueError("quantized probe requires weights, input, and reference")
        for name, data in manifest["files"].items():
            path = directory / name
            with path.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            if path.stat().st_size != sizes[name] or data["bytes"] != sizes[name] or digest != data["sha256"]:
                raise ValueError(f"quantized input checksum mismatch: {path}")
            copy_file(root, path, f"/probes/{index}/{name}")
        prefix = f"/profiles/case{index}"
        arguments = [
            ncu, "--profile-from-start", "off", "--clock-control", "boost", "--cache-control", "all",
            "--replay-mode", "kernel", "--kernel-name-base", "function", "--force-overwrite", "-o", prefix,
        ]
        for section in sections:
            arguments.extend(["--section", section])
        probe = [
            "/quantized-probe", f"/probes/{index}/weights.bin", f"/probes/{index}/input.bin",
            f"/probes/{index}/reference.bin", str(inner), str(outputs), str(format_code),
        ]
        commands.append(shlex.join(probe))
        arguments.extend(probe)
        arguments.append("profile")
        command = shlex.join(arguments)
        commands.extend([
            f'if ! {command} >{prefix}.log 2>&1; then cat {prefix}.log; dmesg; for core in /tmp/core.*; do if [ -f "$core" ]; then cp "$core" /profiles/; fi; done; exit 1; fi',
            f"cat {prefix}.log",
            f"{ncu} --import {prefix}.ncu-repz --page raw --csv --print-units base >{prefix}.csv",
        ])
        cases.append(manifest)
    commands.append("echo LINUX_QUANTIZED_PROFILE_OK")
    benchmark = root / "benchmark"
    benchmark.write_text("\n".join(commands) + "\n")
    benchmark.chmod(0o755)
    (root.parent / "quantized-profile-manifest.json").write_text(json.dumps({
        "execution_os": "linux", "backend": "native D3OS cuTile kernels", "instrumented": True,
        "ordinary_performance_comparison": False, "clock_control": "boost", "cache_control": "all",
        "buffer_alignment_bytes": 256, "warmup_launches": 256, "guest_cpus": 4,
        "host_affinity": sorted(os.sched_getaffinity(0)),
        "replay_mode": "kernel", "sections": sections, "cases": cases,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(portable.read_bytes()).hexdigest(), "expected_cubin_sha256": expected,
        "profiler": json.loads((nsight / "d3os-package.json").read_text()),
    }, indent=2) + "\n")


def validate_quantized_profile(directory):
    manifest = json.loads((directory / "quantized-profile-manifest.json").read_text())
    formats = {8: "q8_0", 11: "q3k", 12: "q4k", 13: "q5k", 14: "q6k"}
    results = []
    for index, case in enumerate(manifest["cases"]):
        path = directory / f"root/profiles/case{index}.csv"
        with path.open(newline="") as file:
            rows = list(csv.reader(file))
        if len(rows) != 3 or len(rows[0]) != len(set(rows[0])) or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("expected exactly one profiled launch with unique metric columns")
        names, units, values = rows
        fields = dict(zip(names, values))
        entry = f"vector_project_{formats[case['format']]}"
        if (fields.get("Kernel Name") != entry or fields.get("Block Size") != "(128, 1, 1)"
                or fields.get("Grid Size") != f"({(case['outputs'] + 3) // 4}, 1, 1)"):
            raise ValueError("counter report targets a different kernel or launch shape")
        metrics = {}
        for name, unit, value in zip(names, units, values):
            if "__" not in name or value in ("", "N/A"):
                continue
            try:
                number = float(value.replace(",", ""))
            except ValueError:
                continue
            if not math.isfinite(number):
                raise ValueError(f"nonfinite GPU metric: {name}")
            metrics[name] = {"value": number, "unit": unit}
        required = {
            "gpu__time_duration.sum": "ns",
            "dram__bytes_read.sum": "byte",
            "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed": "%",
        }
        for name, unit in required.items():
            if name not in metrics or metrics[name]["value"] <= 0 or metrics[name]["unit"] != unit:
                raise ValueError(f"missing hardware counter or incorrect unit: {name}")
        report = directory / f"root/profiles/case{index}.ncu-repz"
        results.append({"case": case, "launch": {name: fields[name] for name in names[:12]}, "metrics": metrics,
                        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()})
    result = {"execution_os": "linux", "backend": "native D3OS cuTile kernels", "instrumented": True,
              "ordinary_performance_comparison": False, "cases": results}
    (directory / "quantized-profile-results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def llama_benchmark(root, directory, model):
    llama_libraries(root, directory)
    copy_file(root, model, "/model.gguf")
    benchmark = root / "benchmark"
    benchmark.write_text("""#!/bin/bash
set -eux
export LD_LIBRARY_PATH=/bin:/lib64
/bin/llama-bench -m /model.gguf -p 22 -n 64 -r 3 -ngl 99 -t 4 -o json
echo LINUX_LLAMA_BENCH_OK
""")
    benchmark.chmod(0o755)


def large_qwen_benchmark(root, directory, source, model, target):
    spec = importlib.util.spec_from_file_location(
        "large_qwen", ROOT / "tools/nvidia/prepare-large-qwen.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.MODELS[target]
    if (
        model.stat().st_size != expected["bytes"]
        or module.digest(model) != expected["sha256"]
    ):
        raise ValueError("large Qwen model checksum mismatch")
    llama_libraries(root, directory)
    binary = root / "bin/linux-large-qwen-bench"
    source_file = ROOT / "tools/nvidia/linux-large-qwen-bench.c"
    subprocess.run(
        [
            "gcc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(source / "include"),
            "-I",
            str(source / "ggml/include"),
            str(source_file),
            "-L",
            str(directory),
            f"-Wl,-rpath-link,{directory}",
            "-lllama",
            "-lggml",
            "-lggml-base",
            "-o",
            str(binary),
        ],
        check=True,
    )
    for library in re.findall(
        r"/[^\s()]+",
        subprocess.check_output(
            ["ldd", str(binary)],
            text=True,
            env=dict(os.environ, LD_LIBRARY_PATH=str(directory)),
        ),
    ):
        copy_file(root, library)
    copy_file(root, model, "/model.gguf")
    benchmark = root / "benchmark"
    benchmark.write_text(
        "#!/bin/bash\nset -eux\nexport LD_LIBRARY_PATH=/bin:/lib64\n"
        f"export D3OS_QWEN_REFERENCE_OUT=/large-qwen-logits.bin D3OS_MODEL_SHA256={expected['sha256']}\n"
        f"/bin/linux-large-qwen-bench /model.gguf {expected['layers']} {expected['hidden']}\n"
    )
    benchmark.chmod(0o755)
    evidence = dict(expected)
    evidence.update(
        source_sha256=module.digest(source_file),
        executable_sha256=module.digest(binary),
        libraries={
            path.name: module.digest(path) for path in sorted(directory.glob("*.so*"))
        },
        headers={
            str(path.relative_to(source)): module.digest(path)
            for folder in (source / "include", source / "ggml/include")
            for path in sorted(folder.glob("*.h"))
        },
        cache="f32",
        context_capacity=2048,
        decode_tokens=128,
        batch=32,
        weights="all_cuda",
        comparison="Linux baseline; native comparison pending",
    )
    (root.parent / "large-qwen-manifest.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


def trace_cuda_kernels(root):
    source = ROOT / "tools/nvidia/linux-cuda-trace.c"
    library = root / "bin/linux-cuda-trace.so"
    subprocess.run(
        [
            "gcc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(source),
            "-ldl",
            "-pthread",
            "-o",
            str(library),
        ],
        check=True,
    )
    benchmark = root / "benchmark"
    script = benchmark.read_text()
    first, rest = script.split("\n", 1)
    benchmark.write_text(
        first + "\nexport LD_PRELOAD=/bin/linux-cuda-trace.so\n" + rest
    )
    (root.parent / "cuda-trace-manifest.json").write_text(
        json.dumps(
            {
                "os": "linux",
                "backend": "cuda",
                "instrumented": True,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


def validate_large_qwen(text, layers):
    if (
        "LINUX_LARGE_QWEN_OK" not in text
        or f"offloaded {layers + 1}/{layers + 1} layers to GPU" not in text
    ):
        raise ValueError("large Qwen benchmark requires success and full GPU offload")
    if "CPU_Mapped model buffer" in text or "CPU model buffer" in text:
        raise ValueError("large Qwen weights must all reside on the GPU")
    if any(marker not in text for marker in ("K (f32):", "V (f32):", "S (f32):")):
        raise ValueError("large Qwen requires FP32 attention and recurrent caches")
    contexts = re.findall(r"LINUX_LARGE_QWEN_CONTEXT ([^\r\n]*)", text)
    if len(contexts) != 1:
        raise ValueError("require one large Qwen context")
    context = dict(re.findall(r"(\w+)=([^\s]+)", contexts[0]))
    if any(
        context.get(key) != value
        for key, value in {"cache": "f32", "batch": "32", "n_ctx": "2048"}.items()
    ):
        raise ValueError("large Qwen context configuration mismatch")
    models = re.findall(
        r"LINUX_LARGE_QWEN_MODEL layers=(\d+) hidden=(\d+) parameters=(\d+) load_ms=([\d.]+)",
        text,
    )
    if (
        len(models) != 1
        or int(models[0][0]) != layers
        or int(models[0][1]) != (4096 if layers == 32 else 5120)
    ):
        raise ValueError("large Qwen model configuration mismatch")
    load_ms, state_ms = float(models[0][3]), float(context["state_ms"])
    if any(not math.isfinite(value) or value <= 0 for value in (load_ms, state_ms)):
        raise ValueError("invalid large Qwen setup timing")
    prompts = re.findall(r"LINUX_LARGE_QWEN_PROMPT ids=(\[.*?\])", text)
    if len(prompts) != 1:
        raise ValueError("require one large Qwen prompt trace")
    prompt = json.loads(prompts[0])
    if not 0 < len(prompt) <= 256 or any(
        type(token) is not int or not 0 <= token < 248320 for token in prompt
    ):
        raise ValueError("invalid large Qwen prompt")
    rows = re.findall(r"LINUX_LARGE_QWEN_BENCH ([^\r\n]*)", text)
    traces = re.findall(r"LINUX_LARGE_QWEN_TOKENS run=(\d+) ids=(\[.*?\])", text)
    if len(rows) != 3 or [int(run) for run, _ in traces] != [0, 1, 2]:
        raise ValueError("require three large Qwen runs and token traces")
    token_traces, samples = [], []
    for run, (row, (_, trace)) in enumerate(zip(rows, traces)):
        fields = dict(re.findall(r"(\w+)=([^\s]+)", row))
        tokens = json.loads(trace)
        if len(tokens) != 128 or any(
            type(token) is not int or not 0 <= token < 248320 for token in tokens
        ):
            raise ValueError("invalid large Qwen token trace")
        checksum = 0
        for token in tokens:
            checksum = (checksum * 31 + token) % (1 << 64)
        expected = {
            "run": run,
            "warm": "true" if run else "false",
            "decode_tokens": 128,
            "prompt_tokens": len(prompt),
            "checksum": checksum,
            "cache": "f32",
            "n_ctx": 2048,
            "sampler": "backend_greedy",
            "eos": "continue_for_benchmark",
        }
        if any(fields.get(key) != str(value) for key, value in expected.items()):
            raise ValueError("large Qwen settings or checksum mismatch")
        for key in ("prefill_ms", "decode_ms", "decode_tps"):
            if not math.isfinite(float(fields[key])) or float(fields[key]) <= 0:
                raise ValueError("invalid large Qwen timing")
        if not math.isclose(
            float(fields["decode_tps"]),
            128000 / float(fields["decode_ms"]),
            rel_tol=0.001,
        ):
            raise ValueError("large Qwen rate differs from evaluated work")
        token_traces.append(tokens)
        samples.append(
            {
                key: float(fields[key])
                for key in ("prefill_ms", "decode_ms", "decode_tps")
            }
        )
    if any(tokens != token_traces[0] for tokens in token_traces[1:]):
        raise ValueError("large Qwen independent sessions disagree")
    memory_rows = re.findall(
        r"LINUX_LARGE_QWEN_MEMORY phase=(\w+) free_bytes=(\d+) total_bytes=(\d+) used_bytes=(\d+)",
        text,
    )
    if [phase for phase, *_ in memory_rows] != [
        "before_load",
        "after_load",
        "after_context",
        "after_decode",
    ]:
        raise ValueError("require all four GPU memory measurements")
    memory = {}
    for phase, free, total, used in memory_rows:
        free, total, used = int(free), int(total), int(used)
        if not 0 < free < total <= 16 * 1024**3 or used != total - free:
            raise ValueError("invalid 16 GiB GPU memory measurement")
        memory[phase] = {"free_bytes": free, "total_bytes": total, "used_bytes": used}
    return {
        "model_load_ms": load_ms,
        "context_ms": state_ms,
        "samples": samples,
        "memory": memory,
        "prompt_ids": prompt,
        "decode_ids": token_traces[0],
        "warm_decode_tps_mean": sum(sample["decode_tps"] for sample in samples[1:]) / 2,
    }


def matched_qwen_benchmark(root, directory, source, model, inputs, flash_attention):
    manifest = json.loads((inputs / "manifest.json").read_text())

    def verify(path, expected):
        with path.open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"matched benchmark checksum mismatch: {path}")

    verify(model, manifest["gguf_sha256"])
    verify(inputs / "tokens.bin", manifest["tokens_sha256"])
    for name, expected in manifest["source_headers"].items():
        verify(source / name, expected)
    for name, expected in manifest["libraries"].items():
        verify(directory / name, expected)
    llama_libraries(root, directory)
    binary = root / "bin/linux-qwen-bench"
    subprocess.run(
        [
            "gcc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(source / "include"),
            "-I",
            str(source / "ggml/include"),
            str(ROOT / "tools/nvidia/linux-qwen-bench.c"),
            "-L",
            str(directory),
            f"-Wl,-rpath-link,{directory}",
            "-lllama",
            "-lggml",
            "-lggml-base",
            "-o",
            str(binary),
        ],
        check=True,
    )
    for library in re.findall(
        r"/[^\s()]+",
        subprocess.check_output(
            ["ldd", str(binary)],
            text=True,
            env=dict(os.environ, LD_LIBRARY_PATH=str(directory)),
        ),
    ):
        copy_file(root, library)
    copy_file(root, model, "/model.gguf")
    copy_file(root, inputs / "tokens.bin", "/tokens.bin")
    benchmark = root / "benchmark"
    benchmark.write_text(f"""#!/bin/bash
set -eux
export LD_LIBRARY_PATH=/bin:/lib64
/bin/linux-qwen-bench /model.gguf /tokens.bin {flash_attention}
""")
    benchmark.chmod(0o755)
    manifest["flash_attention_requested"] = flash_attention
    manifest["benchmark_source_sha256"] = hashlib.sha256(
        (ROOT / "tools/nvidia/linux-qwen-bench.c").read_bytes()
    ).hexdigest()
    manifest["benchmark_binary_sha256"] = hashlib.sha256(
        binary.read_bytes()
    ).hexdigest()
    (root.parent / "matched-qwen-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def validate_matched_qwen(text, manifest, flash_attention, *, strict_selection=True):
    rows = re.findall(r"LINUX_MATCHED_QWEN_BENCH ([^\r\n]*)", text)
    if len(rows) != 6 or "offloaded 25/25 layers to GPU" not in text:
        raise ValueError("matched benchmark requires six runs and full GPU offload")
    if "K (f32):" not in text or "V (f32):" not in text or "S (f32):" not in text:
        raise ValueError(
            "matched benchmark requires FP32 attention and recurrent caches"
        )
    checksum = 0
    for token in manifest["decode_ids"]:
        checksum = (checksum * 31 + token) % (1 << 64)
    sampled = re.findall(
        r"LINUX_MATCHED_QWEN_TOKENS phase=sampled run=(\d+) ids=(\[.*?\])", text
    )
    if [int(run) for run, _ in sampled] != [0, 1, 2, 0, 1, 2]:
        raise ValueError("require six ordered Linux selection traces")
    sampled = [json.loads(trace) for _, trace in sampled]
    differences = []
    for tokens in sampled:
        if len(tokens) != len(manifest["decode_ids"]) or any(
            type(token) is not int or not 0 <= token < 248320 for token in tokens
        ):
            raise ValueError("invalid Linux selection trace")
        differences.append(sum(a != b for a, b in zip(tokens, manifest["decode_ids"])))
    for index, row in enumerate(rows):
        run = index % 3
        fields = dict(re.findall(r"(\w+)=([^\s]+)", row))
        sampled_checksum = 0
        for token in sampled[index]:
            sampled_checksum = (sampled_checksum * 31 + token) % (1 << 64)
        expected = {
            "mode": "new_context" if index < 3 else "reset_context",
            "run": run,
            "warm": "true" if run else "false",
            "prompt_tokens": len(manifest["prompt_ids"]),
            "decode_tokens": len(manifest["decode_ids"]),
            "input_checksum": checksum,
            "sampled_checksum": sampled_checksum,
            "selection_mismatches": differences[index],
            "cache": "f32",
            "n_ctx": manifest["context_capacity"],
            "batch": manifest["batch_size"],
            "ubatch": manifest["batch_size"],
            "sampler": "backend_greedy",
            "flash_attention_requested": flash_attention,
        }
        if any(fields.get(key) != str(value) for key, value in expected.items()):
            raise ValueError(
                "Linux settings or reported choices differ from the matched workload"
            )
    for phase, expected, count in [
        ("prompt", manifest["prompt_ids"], 1),
        ("decode_input", manifest["decode_ids"], 1),
    ]:
        traces = re.findall(
            rf"LINUX_MATCHED_QWEN_TOKENS phase={phase} run=-?\d+ ids=(\[.*?\])", text
        )
        if len(traces) != count or any(
            json.loads(trace) != expected for trace in traces
        ):
            raise ValueError("Linux token trace differs from the matched workload")
    if strict_selection and any(differences):
        raise ValueError("Linux greedy choices differ from the matched workload")
    return differences


def alexnet_benchmark(root):
    cudnn = ROOT / "target/nvidia/linux-cudnn"
    headers = cudnn / "include"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/nvidia/prepare-linux-cudnn.py"),
            "--output",
            str(cudnn),
        ],
        check=True,
    )
    cuda_headers = ROOT / "target/nvidia/cutile-toolkit/packages/nvidia/cu13/include"
    binary = root / "benchmark"
    subprocess.run(
        [
            "gcc",
            "-D_POSIX_C_SOURCE=200809L",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-deprecated-declarations",
            "-I",
            str(headers),
            "-I",
            str(cuda_headers),
            str(ROOT / "tools/nvidia/linux-alexnet-bench.c"),
            "-ldl",
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
    )
    manifest = {
        "cudnn": json.loads((cudnn / "manifest.json").read_text()),
        "artifacts": {},
    }

    def remember(source, destination):
        source = Path(source)
        copy_file(root, source, destination)
        with source.open("rb") as file:
            manifest["artifacts"][destination] = hashlib.file_digest(
                file, "sha256"
            ).hexdigest()

    # Use the complete pinned cuDNN distribution in this reference guest.
    libraries = [
        *cudnn.joinpath("lib").glob("*.so*"),
        *(
            p
            for p in Path("/usr/local/lib/ollama/mlx_cuda_v13").glob("*.so*")
            if not p.name.startswith("libcudnn")
        ),
        *Path("/usr/local/lib/ollama/cuda_v13").glob("libcublas*.so*"),
    ]
    seen = {}
    for library in libraries:
        actual = library.resolve()
        if actual not in seen:
            dynamic = subprocess.check_output(["readelf", "-d", str(actual)], text=True)
            soname = re.search(r"\(SONAME\).*\[(.*?)\]", dynamic)
            name = soname[1] if soname else actual.name
            seen[actual] = name
            remember(actual, f"/lib64/{name}")
        else:
            name = seen[actual]
        alias = root / "lib64" / library.name
        if library.name != name:
            alias.unlink(missing_ok=True)
            alias.symlink_to(name)
    assets = ROOT / "target/inference"
    provenance = json.loads((assets / "manifest.json").read_text())
    if (
        provenance["weights_sha256"]
        != "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
    ):
        raise ValueError("unexpected AlexNet checkpoint provenance")
    for name, key in [("alexnet.d3t", "model_sha256"), ("dog.d3t", "fixture_sha256")]:
        remember(assets / name, f"/{name}")
        if manifest["artifacts"][f"/{name}"] != provenance[key]:
            raise ValueError(f"AlexNet asset checksum mismatch: {name}")
    for library in re.findall(
        r"/[^\s()]+", subprocess.check_output(["ldd", str(binary)], text=True)
    ):
        copy_file(root, library)
    search = f"{cudnn / 'lib'}:/usr/local/lib/ollama/mlx_cuda_v13:/usr/local/lib/ollama/cuda_v13"
    for library in seen:
        dependencies = subprocess.check_output(
            ["ldd", str(library)],
            text=True,
            env=dict(os.environ, LD_LIBRARY_PATH=search),
            stderr=subprocess.PIPE,
        )
        if "not found" in dependencies:
            raise ValueError(
                f"missing AlexNet library dependency: {library}\n{dependencies}"
            )
        for dependency in re.findall(r"/[^\s()]+", dependencies):
            if Path(dependency).resolve() not in seen:
                remember(dependency, dependency)
    remember(ROOT / "tools/nvidia/linux-alexnet-bench.c", "/alexnet-reference.c")
    with binary.open("rb") as file:
        manifest["artifacts"]["/bin/linux-alexnet-bench"] = hashlib.file_digest(
            file, "sha256"
        ).hexdigest()
    # prepare() launches /benchmark without arguments. Keep the executable separate.
    binary.rename(root / "bin/linux-alexnet-bench")
    binary.write_text(
        "#!/bin/bash\nset -eux\nexport LD_LIBRARY_PATH=/lib64:/bin\n/bin/linux-alexnet-bench /alexnet.d3t /dog.d3t\n"
    )
    binary.chmod(0o755)
    manifest["artifacts"]["/benchmark"] = hashlib.sha256(
        binary.read_bytes()
    ).hexdigest()
    (root.parent / "alexnet-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def validate_alexnet(text):
    rows = re.findall(r"LINUX_ALEXNET_BENCH ([^\r\n]*)", text)
    if len(rows) != 3 or "LINUX_ALEXNET_OK reference_layers=8 class=258" not in text:
        raise ValueError("AlexNet requires three runs and all layer references")
    names = re.findall(r"LINUX_ALEXNET_LAYER_OK name=([^\s]+)", text)
    if names != [
        "features.0",
        "features.3",
        "features.6",
        "features.8",
        "features.10",
        "classifier.1",
        "classifier.4",
        "logits",
    ]:
        raise ValueError("AlexNet reference did not compare every layer")
    for run, row in enumerate(rows):
        fields = dict(re.findall(r"(\w+)=([^\s]+)", row))
        for key, value in {
            "run": str(run),
            "repeats": "100",
            "weights": "f32",
            "arithmetic": "f32",
            "cudnn_math": "fma",
            "cublas_math": "default",
        }.items():
            if fields.get(key) != value:
                raise ValueError(f"unexpected AlexNet setting: {key}")
        for key in ["forward_mean_ms", "forward_total_ms"]:
            if not 0 < float(fields[key]) < 10000:
                raise ValueError("invalid AlexNet timing")


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
    # These compiler components are loaded through dlopen, so ldd cannot
    # discover them. Newer CUBIN files can require the driver's linker too.
    libraries = ["libcuda.so.1", "libnvidia-ml.so.1", "libnvidia-ptxjitcompiler.so.1"]
    driver_version = (
        (Path("/usr/lib64") / "libcuda.so.1").resolve().name.removeprefix("libcuda.so.")
    )
    for library in (
        "libnvidia-nvvm.so.4",
        f"libnvidia-gpucomp.so.{driver_version}",
        f"libnvidia-tileiras.so.{driver_version}",
    ):
        if (Path("/usr/lib64") / library).exists():
            libraries.append(library)
    driver_libraries = {}
    for library in libraries:
        source = Path("/usr/lib64") / library
        copy_file(root, source, Path("/lib64") / library)
        driver_libraries[library] = hashlib.sha256(source.read_bytes()).hexdigest()
        dependencies = subprocess.check_output(["ldd", str(source)], text=True)
        for dependency in re.findall(r"/[^\s()]+", dependencies):
            copy_file(root, dependency)
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
        "driver_libraries_sha256": driver_libraries,
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
    parser.add_argument("--quantized-profile", type=Path, nargs="+", help="checkpoint probe directories for Linux-only GPU hardware counters")
    parser.add_argument("--nsight-directory", type=Path, help="pinned Nsight Compute archive root; default downloads the pinned package")
    parser.add_argument("--compiler-benchmark", action="store_true")
    parser.add_argument("--mnist-benchmark", action="store_true")
    parser.add_argument("--burn-quantization", type=Path, help="run the shared Burn quantization conformance executable on Linux CUDA")
    parser.add_argument("--mnist-data", type=Path, default=ROOT / "target/mnist")
    parser.add_argument(
        "--mnist-executable", type=Path,
        default=ROOT / "tools/mnist-benchmark/target/release/d3os-mnist-benchmark",
    )
    parser.add_argument("--compiler-repeats", type=int, default=3)
    parser.add_argument(
        "--compiler-inputs",
        type=Path,
        help="portable inputs for compiler benchmarks or quantized profiles; default: target/nvidia/kernels",
    )
    parser.add_argument(
        "--compiler-cpus",
        choices=("0", "1", "2", "3", "0-3"),
        nargs="+",
        help="compiler CPU masks inside the four-CPU Linux guest; default: 0 0-3",
    )
    parser.add_argument("--llama-directory", type=Path)
    parser.add_argument("--large-qwen", choices=("9b", "27b"))
    parser.add_argument(
        "--trace-cuda-kernels",
        action="store_true",
        help="record Linux large-Qwen kernel names in a separate instrumented run",
    )
    parser.add_argument("--matched-qwen-input", type=Path)
    parser.add_argument(
        "--matched-qwen-report-selection",
        action="store_true",
        help="report greedy differences while enforcing identical teacher-forced inputs",
    )
    parser.add_argument(
        "--llama-flash-attention", choices=("auto", "off"), default="auto"
    )
    parser.add_argument(
        "--llama-source",
        type=Path,
        default=ROOT / "target/nvidia/linux-llama/source-b11827",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--memory-mib", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--pin-vcpus",
        action="store_true",
        help="pin guest CPUs individually to the caller's four host CPUs before boot",
    )
    parser.add_argument("--doom-benchmark", action="store_true")
    parser.add_argument("--alexnet-benchmark", action="store_true")
    args = parser.parse_args()
    if args.nsight_directory and not args.quantized_profile:
        parser.error("--nsight-directory requires --quantized-profile")
    if args.quantized_profile and any((args.kernel_benchmark, args.compiler_benchmark, args.mnist_benchmark, args.burn_quantization, args.llama_directory, args.doom_benchmark, args.alexnet_benchmark)):
        parser.error("--quantized-profile requires a separate diagnostic guest")
    if args.pin_vcpus and len(os.sched_getaffinity(0)) != 4:
        parser.error("--pin-vcpus requires an inherited affinity of four host CPUs")
    if args.compiler_cpus and not args.compiler_benchmark:
        parser.error("--compiler-cpus requires --compiler-benchmark")
    if args.compiler_inputs is not None:
        if not (args.compiler_benchmark or args.quantized_profile):
            parser.error("--compiler-inputs requires --compiler-benchmark or --quantized-profile")
        args.compiler_inputs = args.compiler_inputs.resolve()
        if not args.compiler_inputs.is_dir():
            parser.error("--compiler-inputs must be a directory")
    if args.trace_cuda_kernels and not (args.large_qwen and args.llama_directory):
        parser.error("--trace-cuda-kernels requires --large-qwen and --llama-directory")
    affinity = ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
    print(f"NVIDIA_HOST_CPU_AFFINITY cpus={affinity}", flush=True)
    if (
        sum(
            (
                args.kernel_benchmark,
                args.compiler_benchmark,
                args.mnist_benchmark,
                bool(args.burn_quantization),
                bool(args.llama_directory),
                args.doom_benchmark,
                args.alexnet_benchmark,
            )
        )
        > 1
    ):
        parser.error("select only one Linux reference workload")
    if not args.mnist_benchmark and bool(args.llama_directory) != bool(args.model):
        parser.error("--llama-directory and --model must be supplied together")
    if args.mnist_benchmark and (args.model is None or not args.mnist_executable.is_file()):
        parser.error("--mnist-benchmark requires --model and a built --mnist-executable")
    if args.burn_quantization and not args.burn_quantization.is_file():
        parser.error("--burn-quantization requires a built conformance executable")
    if args.matched_qwen_input and not args.llama_directory:
        parser.error("--matched-qwen-input requires --llama-directory and --model")
    if args.large_qwen and (not args.llama_directory or args.matched_qwen_input):
        parser.error(
            "--large-qwen requires --llama-directory and --model without --matched-qwen-input"
        )
    if args.matched_qwen_report_selection and not args.matched_qwen_input:
        parser.error("--matched-qwen-report-selection requires --matched-qwen-input")
    if args.memory_mib <= 0:
        parser.error("--memory-mib must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not 1 <= args.compiler_repeats <= 100:
        parser.error("--compiler-repeats must be from 1 to 100")
    args.directory = args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    lock = (args.directory / "guest.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("this Linux guest directory is already in use")
    prepare(args.directory, args.kernel)
    if args.quantized_profile:
        nsight = args.nsight_directory
        if nsight is None:
            spec = importlib.util.spec_from_file_location("nsight_package", Path(__file__).with_name("prepare-nsight-compute.py"))
            package = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(package)
            nsight = package.prepare(ROOT / "target/nvidia/nsight-compute")
        quantized_profile(args.directory / "root", [p.resolve() for p in args.quantized_profile], nsight.resolve(), args.compiler_inputs)
    elif args.burn_quantization:
        burn_quantization(args.directory / "root", args.burn_quantization.resolve())
    elif args.mnist_benchmark:
        mnist_benchmark(
            args.directory / "root", args.mnist_executable.resolve(),
            args.mnist_data.resolve(), args.model.resolve(),
        )
    elif args.compiler_benchmark:
        compiler_benchmark(
            args.directory / "root", args.compiler_repeats,
            args.compiler_cpus or ("0", "0-3"), args.compiler_inputs,
        )
    elif args.kernel_benchmark:
        kernel_benchmark(args.directory / "root")
    elif args.llama_directory:
        if args.large_qwen:
            large_qwen_benchmark(
                args.directory / "root",
                args.llama_directory.resolve(),
                args.llama_source.resolve(),
                args.model.resolve(),
                args.large_qwen,
            )
        elif args.matched_qwen_input:
            matched_qwen_benchmark(
                args.directory / "root",
                args.llama_directory.resolve(),
                args.llama_source.resolve(),
                args.model.resolve(),
                args.matched_qwen_input.resolve(),
                args.llama_flash_attention,
            )
        else:
            llama_benchmark(
                args.directory / "root",
                args.llama_directory.resolve(),
                args.model.resolve(),
            )
    elif args.doom_benchmark:
        doom_benchmark(args.directory / "root")
    elif args.alexnet_benchmark:
        alexnet_benchmark(args.directory / "root")
    elif (args.directory / "root/benchmark").exists():
        (args.directory / "root/benchmark").unlink()
    if args.trace_cuda_kernels:
        trace_cuda_kernels(args.directory / "root")
    if args.prepare_only:
        return
    # Reuse the same VFIO validation as D3OS. This guest never rebinds the host.
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
    monitor = args.directory / "qmp.sock"
    (args.directory / "vcpu-affinity.json").unlink(missing_ok=True)
    if args.pin_vcpus:
        monitor.unlink(missing_ok=True)
        command.extend(["-S", "-qmp", f"unix:{monitor},server=on,wait=off"])
    with (args.directory / "qemu.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        qmp = None
        try:
            if args.pin_vcpus:
                spec = importlib.util.spec_from_file_location(
                    "inference_qemu", ROOT / "tools/inference/test-qemu.py"
                )
                harness = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(harness)
                qmp = harness.Qmp(monitor, process, time.monotonic() + args.timeout)
                mapping = {"execution_os": "linux", **harness.pin_vcpus(qmp)}
                (args.directory / "vcpu-affinity.json").write_text(
                    json.dumps(mapping, indent=2) + "\n"
                )
                print(f"NVIDIA_VM_AFFINITY {json.dumps(mapping)}", flush=True)
                qmp.execute("cont")
            if process.wait(timeout=args.timeout):
                raise RuntimeError(f"Linux reference failed: {args.directory}")
        finally:
            if qmp is not None:
                qmp.close()
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
    if args.quantized_profile:
        if "LINUX_QUANTIZED_PROFILE_OK" not in text or text.count("LINUX_QUANTIZED_PROBE_OK") != 2 * len(args.quantized_profile):
            raise RuntimeError("Linux quantized hardware profile did not pass")
        validate_quantized_profile(args.directory)
        print("LINUX_QUANTIZED_COUNTERS_VALIDATED", flush=True)
    if args.compiler_benchmark and "LINUX_COMPILER_BENCH_OK" not in text:
        raise RuntimeError("Linux compiler benchmark did not pass")
    if args.mnist_benchmark and "LINUX_MNIST_BENCH_OK backend=cuda" not in text:
        raise RuntimeError("Linux CUDA MNIST benchmark did not pass")
    if args.burn_quantization and "BURN_QUANTIZATION_REFERENCE_OK os=linux backend=cuda formats=9 minifloat=e4m3,e5m2,e2m1 boundaries=exhaustive" not in text:
        raise RuntimeError("Linux CUDA Burn quantization conformance did not pass")
    if args.alexnet_benchmark:
        validate_alexnet(text)
    if args.llama_directory:
        if args.large_qwen:
            result = validate_large_qwen(text, 32 if args.large_qwen == "9b" else 64)
            result["instrumented"] = args.trace_cuda_kernels
            if args.trace_cuda_kernels:
                names = re.findall(r"LINUX_CUDA_KERNEL api=\S+ name=([^\r\n]+)", text)
                if not names or "LINUX_CUDA_TRACE_FAIL" in text:
                    raise RuntimeError("Linux CUDA kernel tracing did not complete")
                result["cuda_kernel_names"] = names
            (args.directory / "large-qwen-results.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print("LINUX_LARGE_QWEN_VALIDATED", flush=True)
            return
        marker = (
            "LINUX_MATCHED_QWEN_OK"
            if args.matched_qwen_input
            else "LINUX_LLAMA_BENCH_OK"
        )
        if marker not in text:
            raise RuntimeError("Linux llama benchmark did not pass")
        if args.matched_qwen_input:
            manifest = json.loads(
                (args.matched_qwen_input / "manifest.json").read_text()
            )
            differences = validate_matched_qwen(
                text,
                manifest,
                args.llama_flash_attention,
                strict_selection=not args.matched_qwen_report_selection,
            )
            policy = "report" if args.matched_qwen_report_selection else "strict"
            print(
                f"LINUX_MATCHED_QWEN_SELECTION policy={policy} "
                f"mismatches={sum(differences)} choices={6 * len(manifest['decode_ids'])}",
                flush=True,
            )
    if args.doom_benchmark and not re.search(
        r"LINUX_DOOM_ENGINE_REPORT timed \d+ gametics in \d+ realtics \([0-9.]+ fps\)",
        text,
    ):
        raise RuntimeError("Linux DOOM benchmark did not report a result")
    print("LINUX_NVIDIA_REFERENCE_OK", flush=True)


if __name__ == "__main__":
    main()
