#!/usr/bin/env python3
"""Extract a checkpoint projection and an independent FP64 CPU reference."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("tensor")
    parser.add_argument("output", type=Path)
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "target/nvidia/linux-llama/source-b11827/gguf-py")
    args = parser.parse_args()
    sys.path.insert(0, str(args.gguf_python.resolve()))
    import gguf
    import numpy as np

    reader = gguf.GGUFReader(str(args.model.resolve()))
    tensor = next((tensor for tensor in reader.tensors if tensor.name == args.tensor), None)
    if tensor is None or len(tensor.shape) != 2 or int(tensor.tensor_type) not in (8, 11, 12, 13, 14):
        parser.error("select a supported two-dimensional quantized tensor")
    outputs, inner = (int(n) for n in tensor.shape[::-1])
    if inner * outputs > 0xFFFFFFFF:
        parser.error("projection exceeds the native compute extent limit")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    raw = tensor.data.view(np.uint8).reshape(outputs, -1)
    raw.tofile(directory / "weights.bin")
    vector = np.random.default_rng(0xD305C0DE).standard_normal(inner, dtype=np.float32)
    vector.astype("<f4").tofile(directory / "input.bin")
    reference = np.empty(outputs, dtype="<f4")
    # Keep large vocabulary projections below a fixed CPU memory bound.
    for start in range(0, outputs, 256):
        end = min(start + 256, outputs)
        weights = gguf.dequantize(raw[start:end], tensor.tensor_type).reshape(end - start, inner)
        reference[start:end] = weights.astype(np.float64) @ vector.astype(np.float64)
    if not np.isfinite(reference).all():
        raise ValueError("projection reference contains a nonfinite value")
    reference.tofile(directory / "reference.bin")
    files = {
        name: {"bytes": (directory / name).stat().st_size, "sha256": digest(directory / name)}
        for name in ("weights.bin", "input.bin", "reference.bin")
    }
    manifest = {
        "tensor": tensor.name, "format": int(tensor.tensor_type), "inner": inner, "outputs": outputs,
        "model": str(args.model.resolve()), "model_sha256": digest(args.model),
        "reference": "GGUF dequantization, FP64 CPU dot product, FP32 output",
        "reference_execution_os": "host CPU", "numpy_version": np.__version__,
        "seed": 0xD305C0DE, "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
