#!/usr/bin/env python3
"""Prepare hash-verified GSP 570.144 firmware for the native D3OS driver."""

import argparse
import hashlib
import json
import lzma
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = "0a6871b19abf5d6e024b5d208b101ae53e7fa0de"
FILES = {
    "gsp-570.144.bin": (
        "ga102",
        "a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f",
    ),
    "bootloader-570.144.bin": (
        "ad102",
        "65ab2e6b6e0fca95365c4deac79a34582abcfeb15b6ae234138f22e7183118a8",
    ),
    "booter_load-570.144.bin": (
        "ad102",
        "8b293e19b637c5e22c87a2428d1c71bb13e0904e8a88ac6b3c6c1f2679c6e37a",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target/nvidia/firmware")
    parser.add_argument("--system-root", type=Path, default=Path("/usr/lib/firmware"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (architecture, expected) in FILES.items():
        destination = args.output / name
        relative = Path("nvidia") / architecture / "gsp" / name
        candidates = [destination, args.system_root / relative]
        candidates += [Path(str(args.system_root / relative) + ".xz")]
        data = None
        source = None
        for candidate in candidates:
            if not candidate.is_file():
                continue
            content = candidate.read_bytes()
            if candidate.suffix == ".xz":
                content = lzma.decompress(content)
            if hashlib.sha256(content).hexdigest() == expected:
                data, source = content, str(candidate)
                break
        if data is None:
            source = f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/{REVISION}/{relative.as_posix()}"
            print(f"Downloading {name}", flush=True)
            with urllib.request.urlopen(source, timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"firmware SHA-256 mismatch: {name}")
        if not destination.is_file() or destination.read_bytes() != data:
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(destination)
        manifest[name] = {"sha256": expected, "bytes": len(data), "source": source}
        print(f"FIRMWARE_OK {name} bytes={len(data)}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
