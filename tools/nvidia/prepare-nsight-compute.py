#!/usr/bin/env python3
"""Download the pinned Linux-only counter profiler without installing host drivers."""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "2026.3.1.2"
PACKAGE = f"nsight_compute-linux-x86_64-{VERSION}-archive"
URL = f"https://developer.download.nvidia.com/compute/cuda/redist/nsight_compute/linux-x86_64/{PACKAGE}.tar.xz"
SHA256 = "a0039be47cfe551f89d82087256080a9f33c6257d99a72c1164a87466ab072aa"
BYTES = 397165560
REDISTRIBUTION = "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.3.json"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"{PACKAGE}.tar.xz"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=60) as source, temporary.open("wb") as output:
            while block := source.read(1 << 20):
                output.write(block)
        if temporary.stat().st_size != BYTES or digest(temporary) != SHA256:
            raise ValueError("Nsight Compute archive checksum mismatch")
        temporary.replace(archive)
    if archive.stat().st_size != BYTES or digest(archive) != SHA256:
        raise ValueError("cached Nsight Compute archive checksum mismatch")
    target = directory / PACKAGE
    marker = target / "d3os-package.json"
    if not marker.exists():
        with tarfile.open(archive, "r:xz") as package:
            package.extractall(directory, filter="data")
        marker.write_text(json.dumps({
            "version": VERSION,
            "source": URL,
            "redistribution_manifest": REDISTRIBUTION,
            "archive_sha256": SHA256,
            "execution_os": "linux",
            "purpose": "diagnostic GPU counters; excluded from ordinary performance",
        }, indent=2) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "target/nvidia/nsight-compute")
    args = parser.parse_args()
    print(prepare(args.directory.resolve()), flush=True)


if __name__ == "__main__":
    main()
