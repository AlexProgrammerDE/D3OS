#!/usr/bin/env python3
"""Download the pinned host cuTile compiler dependencies without installing a driver."""

import argparse
import hashlib
import json
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = Path(__file__).with_name("cutile-toolkit.json")


def prepare(directory):
    directory = directory.resolve()
    wheels = directory / "wheels"
    packages = directory / "packages"
    wheels.mkdir(parents=True, exist_ok=True)
    packages.mkdir(parents=True, exist_ok=True)
    for item in json.loads(PACKAGES.read_text()):
        wheel = wheels / item["file"]
        if not wheel.is_file():
            temporary = wheel.with_suffix(".download")
            print(f"Download {item['name']} {item['version']}", flush=True)
            with (
                urllib.request.urlopen(item["url"], timeout=120) as source,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(source, output)
            temporary.replace(wheel)
        with wheel.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {wheel}")
        marker = directory / f"{item['name']}.sha256"
        if marker.is_file() and marker.read_text().strip() == digest:
            continue
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                destination = (packages / member.filename).resolve()
                if not destination.is_relative_to(packages):
                    raise RuntimeError(f"Unsafe archive path: {member.filename}")
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise RuntimeError(f"Unexpected symlink: {member.filename}")
                archive.extract(member, packages)
                if member.external_attr >> 16 & 0o111:
                    destination.chmod(destination.stat().st_mode | 0o111)
        marker.write_text(digest + "\n")
    toolkit = packages / "nvidia/cu13"
    for name in ("tileiras", "ptxas", "nvlink"):
        if not (toolkit / "bin" / name).is_file():
            raise RuntimeError(f"Incomplete compiler toolkit: missing {name}")
    return toolkit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "target/nvidia/cutile-toolkit"
    )
    args = parser.parse_args()
    print(prepare(args.output))


if __name__ == "__main__":
    main()
