#!/usr/bin/env python3
"""Export portable cuTile input for D3OS, with optional host SASS audits."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("vector-add", "tensor-matmul", "inference")
TENSOR_INSTRUCTIONS = {
    "tensor-matmul": {
        "tensor_matmul_f16": r"\bHMMA\.16816\.F32\s",
        "tensor_matmul_bf16": r"\bHMMA\.16816\.F32\.BF16\s",
    },
    "inference": {
        **dict.fromkeys(
            [
                "project_quantized",
                "project_q3k",
                "project_q4k",
                "project_q5k",
                "project_q6k",
                "project_m32_q8_0",
                "project_m32_q3k",
                "project_m32_q4k",
                "project_m32_q5k",
                "project_m32_q6k",
            ],
            r"\bHMMA\.1688\.F32\.TF32\s",
        ),
        "tensor_matmul_tf32x3": r"\bHMMA\.1688\.F32\.TF32\s",
        "tensor_matmul_tf32x3_k16": r"\bHMMA\.1688\.F32\.TF32\s",
        "tensor_matmul_tf32x3_split": r"\bHMMA\.1688\.F32\.TF32\s",
        "tensor_project_tf32x3": r"\bHMMA\.1688\.F32\.TF32\s",
        "tensor_project_bf16_weights": r"\bHMMA\.16816\.F32\.BF16\s",
        "project_bf16_split": r"\bHMMA\.16816\.F32\.BF16\s",
    },
}


def audit(family, sass):
    """Check actual device instructions, not compiler source annotations."""
    bodies = dict(
        re.findall(
            r"^\.text\.([\w]+):$(.*?)(?=^\.text\.|\Z)",
            sass,
            re.MULTILINE | re.DOTALL,
        )
    )
    for name, instruction in TENSOR_INSTRUCTIONS.get(family, {}).items():
        if name not in bodies or not re.search(instruction, bodies[name]):
            raise RuntimeError(f"{name}: missing Tensor Core instruction")
    if family == "inference":
        if re.search(r"\bFMNMX\b", bodies.get("inference_relu", "")):
            raise RuntimeError("ReLU must preserve NaN payloads")
        for name in (
            "delta_fused_four_taps",
            "tensor_matmul_tf32x3_k16",
            "tensor_matmul_tf32x3_split",
            "burn_adam",
            "burn_quantize",
            "burn_dequantize",
            "burn_quantized_copy",
            "burn_scale_encode",
            "burn_scale_pack",
            "burn_scale_unpack",
            "project_quantized",
            "project_q3k",
            "project_q4k",
            "project_q5k",
            "project_q6k",
            "project_m32_q8_0",
            "project_m32_q3k",
            "project_m32_q4k",
            "project_m32_q5k",
            "project_m32_q6k",
            "vector_project_q8_0",
            "vector_project_q3k",
            "vector_project_q4k",
            "vector_project_q5k",
            "vector_project_q6k",
        ):
            body = bodies.get(name, "")
            if not body or re.search(r"\b(?:LDL|STL)\b", body):
                raise RuntimeError(f"{name}: specialization must remain spill-free")
        for pattern in (r"\bLDL\b", r"\bSTL\b"):
            if not re.search(pattern, bodies.get("local_stack_probe", "")):
                raise RuntimeError("scratch diagnostic did not generate local spills")
        for name in ("raster_binned_bytes", "raster_fuzz_bytes"):
            for pattern in (r"\bLDG\.E\.U8\b", r"\bSTG\.E\.U8\b"):
                if not re.search(pattern, bodies.get(name, "")):
                    raise RuntimeError(f"{name}: missing byte memory instruction")
    # CB0 holds arguments; CB2 holds compiler constants. Reject banks used
    # by device allocation or other unsupported guest runtime calls.
    for name, body in bodies.items():
        instructions = "\n".join(
            line for line in body.splitlines() if re.search(r"/\*[0-9a-f]+\*/", line)
        )
        if re.search(r"c\[0x(?:[13456789a-f][0-9a-f]*|2[0-9a-f]+)\]", instructions):
            raise RuntimeError(f"{name}: unsupported device runtime constant bank")
    return bodies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--toolkit",
        type=Path,
        help="CUDA 13.4 toolkit; default downloads pinned host tools",
    )
    parser.add_argument(
        "--nvdisasm", type=Path, help="override the toolkit disassembler"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="also assemble sm_89 code on the host and audit its instructions",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "target/nvidia/kernels")
    args = parser.parse_args()
    if args.toolkit:
        toolkit = args.toolkit.resolve()
    else:
        spec = importlib.util.spec_from_file_location(
            "cutile_toolkit", Path(__file__).with_name("prepare-cutile-toolkit.py")
        )
        bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap)
        toolkit = bootstrap.prepare(ROOT / "target/nvidia/cutile-toolkit")
    disassembler = args.nvdisasm or toolkit / "bin/nvdisasm"
    env = os.environ.copy()
    env["CUDA_TOOLKIT_PATH"] = str(toolkit)
    env["CARGO_TARGET_DIR"] = str(ROOT / "target/cutile-build")
    crate = ROOT / "tools/gpu-kernels"
    toolchain = tomllib.loads((crate / "rust-toolchain.toml").read_text())["toolchain"][
        "channel"
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="cutile-", dir=args.output.parent
    ) as temporary:
        staging = Path(temporary)
        subprocess.run(
            [
                "cargo",
                f"+{toolchain}",
                "run",
                "--locked",
                "--manifest-path",
                str(crate / "Cargo.toml"),
                "--release",
                "--",
                str(staging),
                *(["--audit"] if args.audit else []),
            ],
            cwd=ROOT,
            env=env,
            check=True,
            timeout=900,
        )
        manifest = {
            "architecture": None,
            "format": "tile-ir",
            "code_generation_os": "d3os",
            "compiler": "NVIDIA cuTile Rust",
            "tile_ir": "13.4",
            "cuda_rust_revision": tomllib.loads((crate / "Cargo.toml").read_text())[
                "dependencies"
            ]["cutile"]["rev"],
            "hardware_validated": False,
            "kernels": {},
        }
        for family in FAMILIES:
            ir = staging / f"{family}.tilebc"
            entries = json.loads((staging / f"{family}.json").read_text())
            manifest["kernels"][family] = {
                "input": ir.name,
                "sha256": hashlib.sha256(ir.read_bytes()).hexdigest(),
                "bytes": ir.stat().st_size,
                "entries": entries,
            }
            if not args.audit:
                continue
            cubin = staging / f"{family}.cubin"
            sass = subprocess.run(
                [str(disassembler), str(cubin)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout
            bodies = audit(family, sass)
            (staging / f"{family}.sass").write_text(sass)
            if any(entry["entry"] not in bodies for entry in entries):
                raise RuntimeError(f"{family}: an entry has no disassembled code")
            manifest["kernels"][family]["host_audit"] = {
                "architecture": "sm_89",
                "sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
                "bytes": cubin.stat().st_size,
                "sass_verified": True,
            }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        args.output.mkdir(parents=True, exist_ok=True)
        for artifact in staging.iterdir():
            shutil.copyfile(artifact, args.output / artifact.name)
        # Remove machine code left by older builds. Optional audit artifacts
        # stay in a separate directory and never enter the guest boot image.
        audit_directory = args.output / "host-audit"
        if args.audit:
            audit_directory.mkdir(exist_ok=True)
        for artifact in args.output.iterdir():
            if artifact.suffix in (".cubin", ".sass"):
                if args.audit and (staging / artifact.name).is_file():
                    shutil.move(artifact, audit_directory / artifact.name)
                else:
                    artifact.unlink()
            elif artifact.suffix == ".ptx":
                artifact.unlink()
    print(
        f"CUTILE_BUILD_OK format=tile-ir code_generation_os=d3os host_audit={args.audit} families={len(FAMILIES)} output={args.output}"
    )


if __name__ == "__main__":
    main()
