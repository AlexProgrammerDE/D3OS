#!/usr/bin/env python3
"""Stage checksum-pinned MNIST IDX files without training a model on the host."""

import argparse
import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

# Mirrors and original checksums from the official torchvision MNIST dataset:
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py
RESOURCES = {
    "train-images-idx3-ubyte": ("f68b3c2dcbeaaa9fbdd348bbdeb94873", 60000, 2051),
    "train-labels-idx1-ubyte": ("d53e105ee54ea40749a09fcbcd1e9432", 60000, 2049),
    "t10k-images-idx3-ubyte": ("9fb629c4189551a2d022fa330f9573f3", 10000, 2051),
    "t10k-labels-idx1-ubyte": ("ec29112dd5afa0611ce80d1b7f02629c", 10000, 2049),
}


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "MNIST", "host_training": False, "files": {}}
    for filename, (checksum, count, magic) in RESOURCES.items():
        compressed = directory / f"{filename}.gz"
        url = f"https://ossci-datasets.s3.amazonaws.com/mnist/{filename}.gz"
        if (
            not compressed.is_file()
            or hashlib.md5(compressed.read_bytes()).hexdigest() != checksum
        ):
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(12 << 20)
            if hashlib.md5(data).hexdigest() != checksum:
                raise RuntimeError(f"MNIST archive checksum mismatch: {filename}")
            temporary = compressed.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(compressed)
        data = gzip.decompress(compressed.read_bytes())
        expected = 16 + count * 784 if magic == 2051 else 8 + count
        if len(data) != expected or struct.unpack_from(">II", data) != (magic, count):
            raise RuntimeError(f"MNIST IDX metadata mismatch: {filename}")
        (directory / filename).write_bytes(data)
        manifest["files"][filename] = {
            "url": url,
            "archive_md5": checksum,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "samples": count,
        }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "target/mnist",
    )
    args = parser.parse_args()
    prepare(args.output)
    print(f"MNIST_DATA_READY train=60000 test=10000 output={args.output}")
