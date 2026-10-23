#!/usr/bin/env python3
"""Export independent packed-weight references from the pinned large Qwen files."""

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", type=Path, default=ROOT / "target/nvidia/large-qwen"
    )
    parser.add_argument(
        "--source", type=Path, default=ROOT / "target/nvidia/linux-llama/source-b11827"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "os/library/compute/tests/fixtures"
    )
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "large_qwen", ROOT / "tools/nvidia/prepare-large-qwen.py"
    )
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    sys.path.insert(0, str(args.source.resolve() / "gguf-py"))
    import gguf
    import numpy as np

    if not Path(gguf.__file__).is_relative_to(args.source.resolve()):
        raise ValueError("use gguf-py from the pinned llama.cpp source tree")
    output = bytearray(b"D3QUANT\0")
    provenance, checksums = [], {}
    for key, formats in [("9b", [8]), ("27b", [11, 12, 13, 14])]:
        source = models.MODELS[key]
        path = args.models / source["filename"]
        if models.digest(path) != source["sha256"]:
            raise ValueError(f"model checksum mismatch: {path}")
        checksums[source["filename"]] = source["sha256"]
        reader = gguf.GGUFReader(path)
        for format_id in formats:
            tensor = next(t for t in reader.tensors if t.tensor_type == format_id)
            elements, width = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            indices = np.linspace(
                0, tensor.n_bytes // width - 1, 1024 // elements, dtype=np.int64
            )
            raw = tensor.data.view(np.uint8).reshape(-1, width)[indices].copy()
            reference = gguf.dequantize(raw, tensor.tensor_type).astype("<f4").ravel()
            output += struct.pack("<III", format_id, reference.size, raw.nbytes)
            output += raw.tobytes() + reference.tobytes()
            provenance.append(
                {
                    "file": source["filename"],
                    "tensor": tensor.name,
                    "format": tensor.tensor_type.name,
                    "block_indices": indices.tolist(),
                    "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                    "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                }
            )
    archive = args.source.with_suffix(".tar.gz")
    expected_source = "988d6834bd8b57ba973922ac0bda27f282a50f17c42fcdf30817915434f06c3a"
    if models.digest(archive) != expected_source:
        raise ValueError("llama.cpp source archive checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "quantized.bin").write_bytes(output)
    (args.output / "quantized.json").write_text(
        json.dumps(
            {
                "llama_source_sha256": expected_source,
                "models": checksums,
                "fixtures": provenance,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"QUANTIZED_FIXTURE_OK formats={len(provenance)} values={len(provenance) * 1024}"
    )


if __name__ == "__main__":
    main()
