#!/usr/bin/env python3
"""Measure the pinned 9B and 27B Linux baselines, one VFIO guest at a time."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", choices=("9b", "27b"), nargs="+")
    parser.add_argument(
        "--directory", type=Path, default=ROOT / "target/nvidia/large-qwen"
    )
    parser.add_argument("--memory-mib", type=int, default=15360)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--llama-directory",
        type=Path,
        default=ROOT / "target/nvidia/linux-llama/llama-b11827",
    )
    parser.add_argument(
        "--llama-source",
        type=Path,
        default=ROOT / "target/nvidia/linux-llama/source-b11827",
    )
    args = parser.parse_args()
    if args.memory_mib <= 0 or args.timeout <= 0:
        parser.error("memory and timeout must be positive")
    preparation = module(
        "large_qwen_models", ROOT / "tools/nvidia/prepare-large-qwen.py"
    )
    reference = module("linux_reference", ROOT / "tools/nvidia/linux-reference.py")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    guest = directory / "guest"
    for key in dict.fromkeys(args.models):
        preparation.prepare(directory, key)
        expected = preparation.MODELS[key]
        command = [
            sys.executable,
            str(ROOT / "tools/nvidia/linux-reference.py"),
            "--directory",
            str(guest),
            "--large-qwen",
            key,
            "--llama-directory",
            str(args.llama_directory.resolve()),
            "--llama-source",
            str(args.llama_source.resolve()),
            "--model",
            str(directory / expected["filename"]),
            "--memory-mib",
            str(args.memory_mib),
            "--timeout",
            str(args.timeout),
        ]
        with (directory / f"{key}-linux.log").open("w") as log:
            started_ns = time.time_ns()
            try:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            finally:
                # Preserve failures too. A later guest must not overwrite the
                # evidence for a model that could not load or finish decoding.
                for source, suffix in [
                    ("serial.log", "serial.log"),
                    ("qemu.log", "qemu.log"),
                    ("large-qwen-manifest.json", "manifest.json"),
                    ("manifest.json", "driver-manifest.json"),
                    ("root/large-qwen-logits.bin", "logits.bin"),
                ]:
                    path = guest / source
                    if path.exists() and path.stat().st_mtime_ns >= started_ns:
                        shutil.copy2(path, directory / f"{key}-linux-{suffix}")
        serial = directory / f"{key}-linux-serial.log"
        measured = reference.validate_large_qwen(
            serial.read_text(errors="replace"), expected["layers"]
        )
        measured.update(
            model=expected["model"],
            quantization=expected["quantization"],
            model_sha256=expected["sha256"],
            serial_sha256=preparation.digest(serial),
            backend="Linux CUDA",
            native_validation="pending",
            guest_memory_mib=args.memory_mib,
        )
        results[key] = measured
        (directory / f"{key}-linux-results.json").write_text(
            json.dumps(measured, indent=2) + "\n"
        )
        print(
            f"LARGE_QWEN_LINUX_OK model={measured['model']} "
            f"warm_decode_tps={measured['warm_decode_tps_mean']:.3f} "
            f"used_vram_bytes={measured['memory']['after_decode']['used_bytes']}",
            flush=True,
        )
    (directory / "linux-results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
