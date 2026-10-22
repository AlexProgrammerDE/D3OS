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
    for kind in ("vector-add", "relu", "matmul", "tensor-matmul", "inference"):
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
        if kind in ("tensor-matmul", "inference"):
            tensor = manifest["kernels"][kind]
            instructions = (
                (
                    ("tensor_matmul_f16", r"\bHMMA\.16816\.F32\s"),
                    ("tensor_matmul_bf16", r"\bHMMA\.16816\.F32\.BF16\s"),
                )
                if kind == "tensor-matmul"
                else (
                    ("tensor_matmul_tf32x3", r"\bHMMA\.1688\.F32\.TF32\s"),
                    ("tensor_project_tf32x3", r"\bHMMA\.1688\.F32\.TF32\s"),
                    ("tensor_project_bf16_weights", r"\bHMMA\.1688\.F32\.TF32\s"),
                )
            )
            tensor["entrypoints"] = [entry for entry, _ in instructions]
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
                for entry, instruction in instructions:
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
                if kind == "inference":
                    stack = re.search(
                        r"^\.text\.local_stack_probe:$(.*?)(?=^\.text\.|\Z)",
                        sass,
                        re.MULTILINE | re.DOTALL,
                    )
                    if not stack or not all(
                        re.search(pattern, stack.group(1))
                        for pattern in (r"\bLDL\b", r"\bSTL\b")
                    ):
                        raise RuntimeError(
                            "local_stack_probe has no local-memory load/store instructions"
                        )
                    tensor["local_stack_sass_verified"] = True
                    for entry, patterns in (
                        ("raster_binned_bytes", (r"\bLDG\.E\.U8\b", r"\bSTG\.E\.U8\b")),
                        ("raster_fuzz_bytes", (r"\bLDG\.E\.U8\b", r"\bSTG\.E\.U8\b")),
                        ("palette_scale_bytes", (r"\bLDG\.E\.U8\b",)),
                    ):
                        body = re.search(
                            rf"^\.text\.{entry}:$(.*?)(?=^\.text\.|\Z)",
                            sass,
                            re.MULTILINE | re.DOTALL,
                        )
                        if not body or not all(
                            re.search(pattern, body.group(1)) for pattern in patterns
                        ):
                            raise RuntimeError(
                                f"{entry} has no expected byte memory instructions"
                            )
                    tensor["byte_raster_sass_verified"] = True
                    for entry in ("rms_norm", "delta_fused"):
                        body = re.search(
                            rf"^\.text\.{entry}:$(.*?)(?=^\.text\.|\Z)",
                            sass,
                            re.MULTILINE | re.DOTALL,
                        )
                        if not body or not all(
                            re.search(pattern, body.group(1))
                            for pattern in (r"\bLDS\b", r"\bSTS\b", r"\bBAR\.SYNC\b")
                        ):
                            raise RuntimeError(
                                f"{entry} has no expected shared memory and barrier instructions"
                            )
                    tensor["shared_transformer_sass_verified"] = True
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
