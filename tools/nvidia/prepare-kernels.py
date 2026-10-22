#!/usr/bin/env python3
"""Compile D3OS Rust kernels to native Ada SASS with the standalone assembler."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ptxas", type=Path, help="path to the standalone NVIDIA PTX assembler"
    )
    parser.add_argument(
        "--nvdisasm", type=Path, help="disassemble and verify tensor instructions"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "target/nvidia/kernels")
    args = parser.parse_args()
    assembler = args.ptxas or shutil.which("ptxas")
    if not assembler:
        parser.error("provide --ptxas or put the NVIDIA PTX assembler on PATH")
    assembler = Path(assembler).resolve()
    disassembler = args.nvdisasm or shutil.which("nvdisasm")
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(ROOT / "target/gpu-kernels")
    manifest = {"architecture": "sm_89", "hardware_validated": False, "kernels": {}}
    for kind in ("vector-add", "relu", "matmul", "tensor-matmul"):
        subprocess.run(
            [
                "cargo",
                "rustc",
                "--locked",
                "--manifest-path",
                str(ROOT / "tools/gpu-kernels/Cargo.toml"),
                "--target",
                "nvptx64-nvidia-cuda",
                "--release",
                "--no-default-features",
                "--features",
                kind,
                "--",
                "-C",
                "target-cpu=sm_89",
            ],
            cwd=ROOT,
            env=env,
            check=True,
            timeout=120,
        )
        ptx = args.output / f"{kind}.ptx"
        shutil.copyfile(
            Path(env["CARGO_TARGET_DIR"])
            / "nvptx64-nvidia-cuda/release/d3os_gpu_kernels.ptx",
            ptx,
        )
        cubin = args.output / f"{kind}.cubin"
        subprocess.run(
            [str(assembler), "-arch=sm_89", "-v", str(ptx), "-o", str(cubin)],
            check=True,
            timeout=120,
        )
        manifest["kernels"][kind] = {
            "sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
            "bytes": cubin.stat().st_size,
        }
        if kind == "tensor-matmul":
            tensor = manifest["kernels"][kind]
            tensor["entrypoints"] = ["tensor_matmul_f16", "tensor_matmul_bf16"]
            tensor["accumulator"] = "f32"
            tensor["sass_verified"] = False
            if disassembler:
                sass = subprocess.run(
                    [str(disassembler), str(cubin)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                ).stdout
                (args.output / f"{kind}.sass").write_text(sass)
                for entry, instruction in (
                    ("tensor_matmul_f16", r"\bHMMA\.16816\.F32\s"),
                    ("tensor_matmul_bf16", r"\bHMMA\.16816\.F32\.BF16\s"),
                ):
                    match = re.search(
                        rf"^\.text\.{entry}:$(.*?)(?=^\.text\.|\Z)",
                        sass,
                        re.MULTILINE | re.DOTALL,
                    )
                    if not match or not re.search(instruction, match.group(1)):
                        raise RuntimeError(
                            f"{entry} has no expected tensor instruction"
                        )
                tensor["sass_verified"] = True
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
