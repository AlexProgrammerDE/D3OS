#!/usr/bin/env python3
"""Download checksum-pinned quantized Qwen targets for the 16 GiB GPU."""

import argparse
import hashlib
import json
import subprocess
import urllib.request
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
        "config_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "config_source_sha256": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05",
        "config_sha256": "cb1a0d3d7f169532311b6f5d5ee6a649c49667248878fdfe22f163dc16797070",
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
        "config_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "config_source_sha256": "69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c",
        "config_sha256": "1390512aaf85b8e294ea26bbe09ac3a79e1e687b9f6755e370e991c95f90f3a7",
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
    config = directory / f"{key}-config.json"
    if not config.exists():
        url = f"https://huggingface.co/{model['model']}/resolve/{model['config_revision']}/config.json"
        with urllib.request.urlopen(url, timeout=30) as source:
            raw = source.read()
        if hashlib.sha256(raw).hexdigest() != model["config_source_sha256"]:
            raise ValueError("Qwen configuration download checksum mismatch")
        canonical = (json.dumps(json.loads(raw), indent=2) + "\n").encode()
        if hashlib.sha256(canonical).hexdigest() != model["config_sha256"]:
            raise ValueError("Qwen canonical configuration checksum mismatch")
        temporary = config.with_suffix(".download")
        temporary.write_bytes(canonical)
        temporary.replace(config)
    elif digest(config) != model["config_sha256"]:
        raise ValueError(f"configuration checksum mismatch: {config}")
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
