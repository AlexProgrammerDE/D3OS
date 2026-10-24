#!/usr/bin/env python3
"""Split a GGUF model into read-only firmware files with a checked index."""

import argparse
import errno
import fcntl
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

PART_BYTES = 1 << 30


def copy_part(source, destination, offset, length):
    """Use filesystem sharing when available, with a bounded copying fallback."""
    with destination.open("wb") as output:
        try:
            # Linux FICLONERANGE: source fd, source offset, length, target offset.
            fcntl.ioctl(
                output.fileno(),
                0x4020940D,
                struct.pack("=qQQQ", source.fileno(), offset, length, 0),
            )
            return
        except OSError as error:
            if error.errno not in (
                errno.EOPNOTSUPP,
                errno.EXDEV,
                errno.EINVAL,
                errno.ENOTTY,
            ):
                raise
        source.seek(offset)
        remaining = length
        while remaining:
            data = source.read(min(8 << 20, remaining))
            if not data:
                raise ValueError("GGUF model changed during preparation")
            output.write(data)
            remaining -= len(data)


def digest(path, combined=None):
    with path.open("rb") as file:
        local = hashlib.sha256()
        while data := file.read(8 << 20):
            local.update(data)
            if combined is not None:
                combined.update(data)
        return local.hexdigest()


def prepare(model, output, expected=None):
    model, output = Path(model).resolve(), Path(output).resolve()
    checksum = digest(model)
    if expected is not None and checksum != expected:
        raise ValueError("GGUF model checksum mismatch")
    size = model.stat().st_size
    count = (size + PART_BYTES - 1) // PART_BYTES
    if not 0 < count <= 64:
        raise ValueError("require a model between 1 byte and 64 GiB")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    parts = []
    combined = hashlib.sha256()
    with model.open("rb") as source:
        for i in range(count):
            name = f"model.{i:03}"
            path = output / name
            length = min(PART_BYTES, size - i * PART_BYTES)
            old = previous.get("parts", [])
            valid = (
                previous.get("sha256") == checksum
                and i < len(old)
                and old[i].get("name") == name
                and path.is_file()
                and path.stat().st_size == length
                and digest(path) == old[i].get("sha256")
            )
            if not valid:
                temporary = path.with_suffix(".download")
                copy_part(source, temporary, i * PART_BYTES, length)
                temporary.replace(path)
            parts.append(
                {"name": name, "bytes": length, "sha256": digest(path, combined)}
            )
    if combined.hexdigest() != checksum:
        raise ValueError("shard contents differ from the original model checksum")
    index = (
        b"D3MODEL\0"
        + struct.pack("<IIQI", 1, PART_BYTES, size, count)
        + bytes.fromhex(checksum)
    )
    (output / "model.index").write_bytes(index)
    manifest = {
        "source": str(model),
        "sha256": checksum,
        "bytes": size,
        "part_bytes": PART_BYTES,
        "parts": parts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument(
        "--linux-reference",
        type=Path,
        help="binary logits from the Linux CUDA reference guest",
    )
    args = parser.parse_args()
    result = prepare(args.model, args.output, args.sha256)
    if args.linux_reference:
        source = args.linux_reference.resolve()
        with source.open("rb") as file:
            header = file.read(64)
        if (
            len(header) != 64
            or header[:8] != b"D3QLOG\0\0"
            or header[32:].hex() != result["sha256"]
        ):
            raise ValueError("Linux logit reference targets a different model")
        version, vocab, prompt, decode, capacity, batch = struct.unpack_from(
            "<6I", header, 8
        )
        expected = 64 + 4 * (prompt + decode) + 4 * (decode + 1) * vocab
        if (
            (version, vocab, decode, capacity, batch) != (1, 248320, 128, 2048, 32)
            or not 1 <= prompt <= 256
            or source.stat().st_size != expected
        ):
            raise ValueError("Linux logit reference has unsupported dimensions")
        destination = args.output / "model.reference"
        if destination.resolve() != source:
            destination.unlink(missing_ok=True)
            try:
                os.link(source, destination)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(source, destination)
        result["reference"] = {
            "source": str(source),
            "bytes": expected,
            "sha256": digest(source),
        }
        provenance_path = source.with_name("manifest.json")
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text())
            if (
                provenance.get("gguf_model_sha256") != result["sha256"]
                or provenance.get("logits_sha256") != result["reference"]["sha256"]
            ):
                raise ValueError("reference provenance differs from the staged files")
            fixture = source.with_name("fixture.d3t")
            if digest(fixture) != provenance.get("fixture_sha256"):
                raise ValueError("full layer fixture checksum mismatch")
            target = args.output / "model.fixture"
            if target.resolve() != fixture:
                target.unlink(missing_ok=True)
                try:
                    os.link(fixture, target)
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
                    shutil.copyfile(fixture, target)
            result["reference"]["provenance"] = provenance
        (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"QWEN_SHARDS_OK bytes={result['bytes']} parts={len(result['parts'])} sha256={result['sha256']}"
    )


if __name__ == "__main__":
    main()
