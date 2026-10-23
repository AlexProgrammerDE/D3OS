#!/usr/bin/env python3
"""Download checksum-pinned quantized Qwen targets for the 16 GiB GPU."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "9b": {
        "model": "Qwen/Qwen3.5-9B",
        "repository": "unsloth/Qwen3.5-9B-GGUF",
        "revision": "3885219b6810b007914f3a7950a8d1b469d598a5",
        "filename": "Qwen3.5-9B-Q8_0.gguf",
        "bytes": 9527502048,
        "sha256": "809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4",
        "layers": 32,
        "hidden": 4096,
        "quantization": "Q8_0",
    },
    "27b": {
        "model": "Qwen/Qwen3.6-27B",
        "repository": "unsloth/Qwen3.6-27B-GGUF",
        "revision": "82d411acf4a06cfb8d9b073a5211bf410bfc29bf",
        "filename": "Qwen3.6-27B-Q3_K_M.gguf",
        "bytes": 13586217184,
        "sha256": "bdfa99d488b501f5ca711f35c930f005319a064eabcaae23290571a25c499eeb",
        "layers": 64,
        "hidden": 5120,
        "quantization": "Q3_K_M",
    },
}


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare(directory, key):
    model = MODELS[key]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / model["filename"]
    if not path.exists():
        temporary = path.with_suffix(".download")
        url = (
            "https://huggingface.co/{repository}/resolve/{revision}/{filename}".format(
                **model
            )
        )
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "--output",
                str(temporary),
                url,
            ],
            check=True,
        )
        if (
            temporary.stat().st_size != model["bytes"]
            or digest(temporary) != model["sha256"]
        ):
            raise ValueError(f"download checksum mismatch: {temporary}")
        temporary.replace(path)
    elif path.stat().st_size != model["bytes"] or digest(path) != model["sha256"]:
        raise ValueError(f"model checksum mismatch: {path}")
    manifest = dict(model, path=str(path.resolve()))
    (directory / f"{key}-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"LARGE_QWEN_MODEL_OK model={model['model']} quantization={model['quantization']} bytes={model['bytes']} path={path}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", choices=MODELS, nargs="+")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "target/nvidia/large-qwen"
    )
    args = parser.parse_args()
    for key in dict.fromkeys(args.models):
        prepare(args.output, key)


if __name__ == "__main__":
    main()
