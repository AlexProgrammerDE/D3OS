#!/usr/bin/env python3
"""Assemble the fixed vector-add demo for Ada (sm_89) on the host."""

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptxas", type=Path, help="standalone NVIDIA PTX assembler")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "target/nvidia/vector-add.cubin"
    )
    args = parser.parse_args()
    assembler = args.ptxas or shutil.which("ptxas")
    if not assembler:
        parser.error("provide --ptxas or put ptxas on PATH")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(assembler),
            "-arch=sm_89",
            "-v",
            str(ROOT / "tools/nvidia/vector-add.ptx"),
            "-o",
            str(args.output),
        ],
        check=True,
        timeout=120,
    )


if __name__ == "__main__":
    main()
