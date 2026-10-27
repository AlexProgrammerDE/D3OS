#!/usr/bin/env python3
"""Verify cuTile compilation, linking, metadata, and SASS without claiming GPU execution."""

import importlib.util
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    spec = importlib.util.spec_from_file_location(
        "cutile_toolkit", ROOT / "tools/nvidia/prepare-cutile-toolkit.py"
    )
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    toolkit = bootstrap.prepare(ROOT / "target/nvidia/cutile-toolkit")
    spec = importlib.util.spec_from_file_location(
        "prepare_kernels", ROOT / "tools/nvidia/prepare-kernels.py"
    )
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    env = os.environ.copy()
    env["CUDA_TOOLKIT_PATH"] = str(toolkit)
    env["CARGO_TARGET_DIR"] = str(ROOT / "target/cutile-build")
    common = ["--locked", "--manifest-path", str(ROOT / "tools/gpu-kernels/Cargo.toml")]
    subprocess.run(
        ["cargo", "+1.98.1", "test", *common],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=600,
    )
    subprocess.run(
        [
            "cargo",
            "+1.98.1",
            "clippy",
            *common,
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=600,
    )
    subprocess.run(
        [
            "python3",
            str(ROOT / "tools/nvidia/prepare-kernels.py"),
            "--toolkit",
            str(toolkit),
            "--audit",
        ],
        cwd=ROOT,
        check=True,
        timeout=preparation.COMPILATION_TIMEOUT + 5 * 60,
    )
    print("CUTILE_VALIDATION_OK hardware_validated=false")


if __name__ == "__main__":
    main()
