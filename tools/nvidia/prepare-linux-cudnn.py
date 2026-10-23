#!/usr/bin/env python3
"""Prepare pinned NVIDIA cuDNN files for the isolated Linux reference."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "9.21.0.82"
URL = (
    "https://files.pythonhosted.org/packages/03/19/"
    "eb4b8a55e76f3474b0c2c10b9609cb1e65811a94d3e695e8e73c2fe7e72f/"
    "nvidia_cudnn_cu13-9.21.0.82-py3-none-manylinux_2_27_x86_64.whl"
)
SHA256 = "f11b5ef83acf79e82c07943c2f2f74830d31943e90050570d6ec9f6a99394cbb"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "target/nvidia/linux-cudnn"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "cudnn.whl"
    if not archive.is_file() or digest(archive) != SHA256:
        temporary = archive.with_suffix(".download")
        with (
            urllib.request.urlopen(URL, timeout=120) as source,
            temporary.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
        if digest(temporary) != SHA256:
            raise ValueError("cuDNN wheel checksum mismatch")
        temporary.replace(archive)
    files = {"include": {}, "lib": {}}
    with zipfile.ZipFile(archive) as source:
        for entry in source.namelist():
            path = Path(entry)
            if len(path.parts) != 4 or path.parts[:2] != ("nvidia", "cudnn"):
                continue
            kind = path.parts[2]
            if kind not in files or (kind == "include" and path.suffix != ".h"):
                continue
            if kind == "lib" and ".so" not in path.name:
                continue
            data = source.read(entry)
            directory = args.output / kind
            directory.mkdir(exist_ok=True)
            (directory / path.name).write_bytes(data)
            files[kind][path.name] = hashlib.sha256(data).hexdigest()
    if (
        "cudnn.h" not in files["include"]
        or "libcudnn_engines_tensor_ir.so.9" not in files["lib"]
    ):
        raise ValueError("cuDNN wheel lacks its headers or Tensor IR engine")
    (args.output / "manifest.json").write_text(
        json.dumps(
            {"version": VERSION, "url": URL, "sha256": SHA256, "files": files},
            indent=2,
        )
        + "\n"
    )
    print(
        f"LINUX_CUDNN_OK version={VERSION} headers={len(files['include'])} libraries={len(files['lib'])}"
    )


if __name__ == "__main__":
    main()
