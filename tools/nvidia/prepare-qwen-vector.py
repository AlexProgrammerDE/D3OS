#!/usr/bin/env python3
"""Add the optional Rust PTX Qwen projection module to an audited kernel bundle."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
ENTRIES = (
    "q3_components_dp4a",
    "q4_components_dp4a",
    "q3_warp_m16",
    "q3_components_padded112",
    "q3_default_padded112",
    "q3_warp_padded112",
    "embedding_q3_padded112",
    "q4_components_wide64",
    "q4_components_halfwarp",
    "q3_components_padded112_wide",
    "q4_warp_m16",
    "q4_shared_m16",
    "q4_shared_m32",
    "q3_shared_m16",
    "q3_negated_shared_m16",
    "q3_shared_m32",
    "q3_negated_shared_m32",
    "q3_components_halfwarp",
)
FAMILY = "qwen-vector"
CRATE = ROOT / "tools/qwen-vector-kernels"
FLAGS = ["-C", "target-cpu=sm_89", "-C", "target-feature=+ptx80", "--emit=asm"]


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(command, **options):
    return subprocess.run(command, check=True, timeout=180, **options)


def compile_module(directory, toolkit, target):
    """Build and qualify the Rust module inside a caller-owned staging directory."""
    directory, toolkit, target = (
        Path(path).resolve() for path in (directory, toolkit, target)
    )
    directory.mkdir(parents=True, exist_ok=True)
    audit_directory = directory / "host-audit"
    audit_directory.mkdir(exist_ok=True)
    guest = load("guest_compiler", Path(__file__).with_name("guest-compiler.py"))
    qualifier = guest.qualification_module()
    toolchain = tomllib.loads((CRATE / "rust-toolchain.toml").read_text())["toolchain"][
        "channel"
    ]
    pins = {
        str(path.relative_to(ROOT)): sha(path)
        for path in [
            CRATE / "Cargo.toml",
            CRATE / "Cargo.lock",
            CRATE / "rust-toolchain.toml",
            CRATE / "src/lib.rs",
            CRATE / "src/q4.rs",
            CRATE / "src/q4_aligned.rs",
            CRATE / "src/q4_halfwarp.rs",
            CRATE / "src/q4_prefill.rs",
            CRATE / "src/q4_shared.rs",
            CRATE / "src/q3_shared.rs",
            CRATE / "src/q3_negated_shared.rs",
            CRATE / "src/q3_negated.rs",
            CRATE / "src/q3_halfwarp.rs",
            CRATE / "src/q3_prefill.rs",
            CRATE / "src/q3_padded.rs",
        ]
    }
    env = dict(
        os.environ,
        CARGO_TARGET_DIR=str(target),
        CARGO_BUILD_JOBS="2",
        CARGO_INCREMENTAL="0",
        CARGO_PROFILE_RELEASE_DEBUG="0",
        RUSTFLAGS="",
    )
    for name in [
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_TARGET",
    ]:
        env.pop(name, None)
    run(
        [
            "cargo",
            f"+{toolchain}",
            "rustc",
            "--manifest-path",
            str(CRATE / "Cargo.toml"),
            "--offline",
            "--locked",
            "--release",
            "-Z",
            "build-std=core",
            "-Z",
            "build-std-features=compiler-builtins-mem",
            "--target",
            "nvptx64-nvidia-cuda",
            "--",
            *FLAGS,
        ],
        env=env,
        cwd=ROOT,
    )
    assembly = list(
        (target / "nvptx64-nvidia-cuda/release/deps").glob(
            "d3os_qwen_vector_kernels-*.s"
        )
    )
    if len(assembly) != 1:
        raise RuntimeError(f"expected one current PTX artifact, found {assembly}")
    source = directory / f"{FAMILY}.ptx"
    shutil.copyfile(assembly[0], source)
    cubin = audit_directory / f"{FAMILY}.cubin"
    run(
        [
            str(toolkit / "bin/ptxas"),
            "-arch=sm_89",
            "-O3",
            "-v",
            str(source),
            "-o",
            str(cubin),
        ]
    )
    sass = run(
        [str(toolkit / "bin/nvdisasm"), str(cubin)],
        capture_output=True,
        text=True,
    ).stdout
    evidence = qualifier.qualify(cubin.read_bytes(), sass)
    if {entry["name"] for entry in evidence["sections"]} != {
        f".text.{entry}" for entry in ENTRIES
    }:
        raise RuntimeError("unexpected Qwen module entry inventory")
    if evidence["cb4_accesses"] or any(
        instruction in sass for instruction in (" LDL", " STL")
    ):
        raise RuntimeError(
            "Qwen vector kernel must have no local memory or CB4 accesses"
        )
    if "IDP.4A.S8.S8" not in sass:
        raise RuntimeError("Qwen vector kernel must contain signed DP4A")
    if any(sha(ROOT / name) != value for name, value in pins.items()):
        raise RuntimeError("Qwen Rust source changed during compilation")
    (audit_directory / f"{FAMILY}.sass").write_text(sass)
    record = {
        "input": source.name,
        "sha256": sha(source),
        "bytes": source.stat().st_size,
        "compiler": "Rust NVPTX",
        "source_pins": pins,
        "rust_toolchain": toolchain,
        "rustc_version": run(
            ["rustc", f"+{toolchain}", "--version", "--verbose"],
            capture_output=True,
            text=True,
        ).stdout,
        "ptxas_sha256": sha(toolkit / "bin/ptxas"),
        "nvdisasm_sha256": sha(toolkit / "bin/nvdisasm"),
        "entries": [{"entry": entry, "threads": 128} for entry in ENTRIES],
        "host_audit": {
            "cubin": f"host-audit/{cubin.name}",
            "sass": f"host-audit/{FAMILY}.sass",
            "architecture": "sm_89",
            "sha256": sha(cubin),
            "bytes": cubin.stat().st_size,
            "sass_verified": True,
        },
        "legacy_qualification": evidence
        | {"source_sha256": sha(source), "source_bytes": source.stat().st_size},
    }
    return record


def publish_bundle(staging, output):
    """Claim a new output directory and publish its validated manifest last."""
    staging, output = staging.resolve(), output.resolve()
    if staging == output or staging in output.parents:
        raise ValueError("output must be outside the staging directory")
    output.mkdir()
    try:
        shutil.copytree(
            staging,
            output,
            dirs_exist_ok=True,
            ignore=lambda directory, names: (
                {"manifest.json"} if Path(directory) == staging else set()
            ),
        )
        pending = output / "manifest.json.tmp"
        shutil.copy2(staging / "manifest.json", pending)
        pending.replace(output / "manifest.json")
    except Exception:
        (output / "manifest.json").unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toolkit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "output already exists; validate and reuse it or choose a new directory"
        )
    if args.bundle.resolve() in args.output.resolve().parents:
        parser.error("output must be outside the input bundle")
    target = os.environ.get("CARGO_TARGET_DIR")
    if not target:
        parser.error("CARGO_TARGET_DIR must name the approved build cache")
    guest = load("guest_compiler", Path(__file__).with_name("guest-compiler.py"))
    qualifier = guest.qualification_module()
    manifest = json.loads(
        (args.bundle / "manifest.json").read_text(),
        object_pairs_hook=guest.unique_object,
    )
    if guest.family_names(manifest["kernels"]) != guest.FAMILIES:
        parser.error("input must be an unextended three-family audited bundle")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exporter = load("prepare_kernels", Path(__file__).with_name("prepare-kernels.py"))
    with exporter.staging_directory(args.output) as staging:
        shutil.copytree(args.bundle, staging, dirs_exist_ok=True)
        # Validate the copied bytes before adding a module or publishing a result.
        for family, record in manifest["kernels"].items():
            guest.checked_qualification(
                staging, family, staging / record["input"], record, qualifier
            )
        manifest["kernels"][FAMILY] = compile_module(staging, args.toolkit, target)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        publish_bundle(staging, args.output)
    print(
        f"QWEN_VECTOR_BUILD_OK family={FAMILY} entries={ENTRIES} output={args.output}"
    )


if __name__ == "__main__":
    main()
