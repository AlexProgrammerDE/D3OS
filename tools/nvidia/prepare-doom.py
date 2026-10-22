#!/usr/bin/env python3
"""Prepare checksum-pinned Freedoom assets for the native D3OS renderer."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = "freedoom-0.13.0"
URL = f"https://github.com/freedoom/freedoom/releases/download/v0.13.0/{RELEASE}.zip"
ARCHIVE_SHA = "3f9b264f3e3ce503b4fb7f6bdcb1f419d93c7b546f4df3e874dd878db9688f59"
FILES = {
    "freedoom1.wad": "7323bcc168c5a45ff10749b339960e98314740a734c30d4b9f3337001f9e703d",
    "freedoom2.wad": "a8772e088847032510d97ba2312406a6998f21cbab44d4ff10696faa9c0ecd4b",
    "COPYING.txt": "7c62f2c520769c798774f416637eec4921e5f0aafdac1245ae7c8a8cf65fe102",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target/doom")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"{RELEASE}.zip"
    if not archive.is_file():
        temporary = archive.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(URL, timeout=120) as response:
                data = response.read(64 * 1024 * 1024 + 1)
            if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA:
                raise RuntimeError("Freedoom archive SHA-256 mismatch")
            temporary.write_bytes(data)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise RuntimeError(f"cached Freedoom archive SHA-256 mismatch: {archive}")
    manifest = {"source": URL, "archive_sha256": ARCHIVE_SHA, "files": {}}
    with zipfile.ZipFile(archive) as contents:
        for name, expected in FILES.items():
            data = contents.read(f"{RELEASE}/{name}")
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"Freedoom asset SHA-256 mismatch: {name}")
            destination = args.output / name
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(destination)
            manifest["files"][name] = {"sha256": expected, "bytes": len(data)}
            print(f"FREEDOOM_OK {name} bytes={len(data)}", flush=True)
    (args.output / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
