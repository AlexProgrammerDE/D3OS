#!/usr/bin/env python3
"""Export portable cuTile and Rust PTX input, with optional core SASS audits."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("vector-add", "tensor-matmul", "inference")
# Cold builds and whole-family assembly can exceed fifteen minutes on CI CPUs.
COMPILATION_TIMEOUT = 30 * 60
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


@contextmanager
def staging_directory(output):
    """Keep failed compiler/audit output without replacing a working bundle."""
    with tempfile.TemporaryDirectory(prefix="cutile-", dir=output.parent) as temporary:
        staging = Path(temporary)
        try:
            yield staging
        except Exception:
            failure = Path(
                tempfile.mkdtemp(prefix=f"{output.name}-failed-", dir=output.parent)
            )
            shutil.copytree(staging, failure, dirs_exist_ok=True)
            print(f"Retained failed build artifacts: {failure}", flush=True)
            raise


def typed_math_scratch():
    spec = importlib.util.spec_from_file_location(
        "typed_math_scratch", Path(__file__).with_name("typed-math-scratch.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_qwen_module(staging, manifest, toolkit, target):
    spec = importlib.util.spec_from_file_location(
        "qwen_vector", Path(__file__).with_name("prepare-qwen-vector.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest["kernels"][module.FAMILY] = module.compile_module(staging, toolkit, target)


def publish_bundle(staging, output, manifest, *, audit):
    """Publish staged families while keeping host-only machine audits separate."""
    staging, output = staging.resolve(), output.resolve()
    if staging == output or staging in output.parents:
        raise ValueError("output must be outside the staging directory")
    staged_manifest = staging / "manifest.json"
    staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    output.mkdir(parents=True, exist_ok=True)
    for artifact in staging.iterdir():
        if artifact.is_dir():
            shutil.copytree(artifact, output / artifact.name, dirs_exist_ok=True)
        elif artifact.name != "manifest.json":
            shutil.copyfile(artifact, output / artifact.name)
    audit_directory = output / "host-audit"
    if audit:
        audit_directory.mkdir(exist_ok=True)
    portable_ptx = {
        record["input"]
        for record in manifest["kernels"].values()
        if record["input"].endswith(".ptx")
    }
    for artifact in output.iterdir():
        if artifact.suffix in (".cubin", ".sass"):
            if audit and (staging / artifact.name).is_file():
                shutil.move(artifact, audit_directory / artifact.name)
            else:
                artifact.unlink()
        elif artifact.suffix == ".ptx" and artifact.name not in portable_ptx:
            artifact.unlink()
    if "qwen-vector" not in manifest["kernels"]:
        for suffix in ("cubin", "sass"):
            (audit_directory / f"qwen-vector.{suffix}").unlink(missing_ok=True)
    # Replace the manifest only after all selected files are present.
    with tempfile.NamedTemporaryFile(dir=output, delete=False) as pending:
        pending.write(staged_manifest.read_bytes())
    shutil.copystat(staged_manifest, pending.name)
    Path(pending.name).replace(output / "manifest.json")


def audit(family, sass, *, cubin=None, entries=None, provenance=None):
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
            "burn_erf",
            "burn_remainder",
            "burn_acos",
            "burn_acosh",
            "burn_asin",
            "burn_asinh",
            "burn_atan",
            "burn_atanh",
            "burn_atan2",
            "burn_argwhere",
            "burn_scatter_ordered_f32",
            "burn_scatter_ordered_i32",
            "burn_cast",
            "burn_deform_windows",
            "burn_deform_coordinates",
            "burn_deform_input",
            "burn_spatial_windows",
            "burn_spatial_fold",
            *(
                f"burn_interpolate_{mode}{direction}"
                for mode in ("nearest", "exact", "linear", "cubic", "lanczos3")
                for direction in ("", "_backward")
            ),
            "burn_storage_copy",
            "burn_storage_fill",
            "burn_i32_divide",
            "burn_i32_remainder",
            "burn_i32_power",
            "burn_predicate",
            "burn_packed_gather",
            "burn_packed_assign",
            "burn_packed_mask",
            "burn_packed_bool_scatter",
            "burn_packed_scatter_assign",
            "burn_integer_arithmetic",
            "burn_float_arithmetic",
            "burn_i32_matmul",
            "burn_cumsum",
            "burn_cumprod",
            "burn_cummin",
            "burn_cummax",
            "burn_i32_cumsum",
            "burn_i32_cumprod",
            "burn_i32_cummin",
            "burn_i32_cummax",
            *(
                f"burn_sort_{dtype}_{order}{phase}"
                for dtype in ("f32", "i32")
                for order in ("ascending", "descending")
                for phase in ("", "_initial")
            ),
            "burn_sort_f32_topk",
            "burn_sort_f32_topk_initial",
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
            if name == "burn_float_arithmetic" and re.search(r"\b(?:LDL|STL)\b", body):
                if cubin is None or entries is None or provenance is None:
                    raise RuntimeError(
                        f"{name}: library scratch requires cubin, entry metadata and compiler provenance"
                    )
                try:
                    typed_math_scratch().qualify(cubin, sass, entries, provenance)
                except (
                    ValueError,
                    KeyError,
                    IndexError,
                    UnicodeError,
                    struct.error,
                ) as error:
                    raise RuntimeError(
                        f"{name}: library-scratch qualification failed: {error}"
                    ) from error
                continue
            if not body or re.search(r"\b(?:LDL|STL)\b", body):
                raise RuntimeError(f"{name}: specialization must remain spill-free")
        for pattern in (r"\bLDL\b", r"\bSTL\b"):
            if not re.search(pattern, bodies.get("local_stack_probe", "")):
                raise RuntimeError("scratch diagnostic did not generate local spills")
        for name in ("raster_binned_bytes", "raster_fuzz_bytes"):
            for pattern in (r"\bLDG\.E\.U8\b", r"\bSTG\.E\.U8\b"):
                if not re.search(pattern, bodies.get(name, "")):
                    raise RuntimeError(f"{name}: missing byte memory instruction")
    # CB0 holds arguments; CB2 holds compiler constants. CB4 contains the
    # qualified math-table pointers. The complete qualifier checks widths
    # and operand forms after this existing runtime-bank audit.
    for name, body in bodies.items():
        instructions = "\n".join(
            line for line in body.splitlines() if re.search(r"/\*[0-9a-f]+\*/", line)
        )
        for match in re.finditer(
            r"c\[0x([0-9a-f]+)\](?:\[0x([0-9a-f]+)\])?", instructions
        ):
            bank = int(match[1], 16)
            offset = int(match[2], 16) if match[2] is not None else None
            if bank in (0, 2):
                continue
            if bank == 4 and offset in (8, 12, 16, 20):
                continue
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
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="export only the three Tile IR families, without Q3/Q4 component decode",
    )
    parser.add_argument(
        "--qwen-target",
        type=Path,
        default=ROOT / "target/qwen-vector-build",
        help="Rust NVPTX build cache for the normal Qwen vector module",
    )
    args = parser.parse_args()
    qualification_spec = importlib.util.spec_from_file_location(
        "cubin_qualification", Path(__file__).with_name("cubin-qualification.py")
    )
    qualification = importlib.util.module_from_spec(qualification_spec)
    qualification_spec.loader.exec_module(qualification)
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
    scratch = typed_math_scratch() if args.audit else None
    provenance = (
        scratch.collect_provenance(crate, toolkit, disassembler) if scratch else None
    )
    toolchain = tomllib.loads((crate / "rust-toolchain.toml").read_text())["toolchain"][
        "channel"
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with staging_directory(args.output) as staging:
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
            timeout=COMPILATION_TIMEOUT,
        )
        if (
            scratch
            and scratch.collect_provenance(crate, toolkit, disassembler) != provenance
        ):
            raise RuntimeError(
                "typed-math source/compiler provenance changed during compilation"
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
            (staging / f"{family}.sass").write_text(sass)
            cubin_bytes = cubin.read_bytes()
            bodies = audit(
                family, sass, cubin=cubin_bytes, entries=entries, provenance=provenance
            )
            if family == "inference" and re.search(
                r"\b(?:LDL|STL)\b", bodies.get("burn_float_arithmetic", "")
            ):
                manifest["kernels"][family]["library_scratch_qualification"] = (
                    scratch.qualify(cubin_bytes, sass, entries, provenance)
                )
            if any(entry["entry"] not in bodies for entry in entries):
                raise RuntimeError(f"{family}: an entry has no disassembled code")
            manifest["kernels"][family]["host_audit"] = {
                "cubin": f"host-audit/{family}.cubin",
                "sass": f"host-audit/{family}.sass",
                "architecture": "sm_89",
                "sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
                "bytes": cubin.stat().st_size,
                "sass_verified": True,
            }
            manifest["kernels"][family]["legacy_qualification"] = qualification.qualify(
                cubin.read_bytes(), sass
            ) | {
                "source_sha256": manifest["kernels"][family]["sha256"],
                "source_bytes": ir.stat().st_size,
            }
        if not args.core_only:
            add_qwen_module(staging, manifest, toolkit, args.qwen_target)
        if (
            scratch
            and scratch.collect_provenance(crate, toolkit, disassembler) != provenance
        ):
            raise RuntimeError(
                "typed-math source/compiler provenance changed before publication"
            )
        publish_bundle(staging, args.output, manifest, audit=args.audit)
    print(
        f"CUTILE_BUILD_OK format=tile-ir code_generation_os=d3os host_audit={args.audit} families={len(manifest['kernels'])} output={args.output}"
    )


if __name__ == "__main__":
    main()
