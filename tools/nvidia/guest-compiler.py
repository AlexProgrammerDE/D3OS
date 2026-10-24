"""Stage portable GPU input and the compiler runtime for execution inside D3OS."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

FAMILIES = ("vector-add", "tensor-matmul", "inference")
VERSION = "13.4.92"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare(kernels, toolkit, directory, *, execution_os="d3os"):
    assets = []
    for family in FAMILIES:
        inputs = [kernels / f"{family}.{suffix}" for suffix in ("tilebc", "ptx")]
        present = [path for path in inputs if path.is_file()]
        if len(present) != 1:
            raise RuntimeError(f"{family}: provide exactly one Tile IR or PTX input")
        assets.append((present[0], Path("usr/share/nvidia") / present[0].name))

    libraries = {}
    for relative in (
        "bin/tileiras",
        "bin/ptxas",
        "lib/libnvvm.so.4",
        "nvvm/libdevice/libdevice.10.bc",
    ):
        source = toolkit / relative
        if not source.is_file():
            raise RuntimeError(f"missing guest compiler dependency: {source}")
        assets.append((source, Path("usr/cuda") / relative))
        if relative.startswith("bin/"):
            version = subprocess.check_output([str(source), "--version"], text=True)
            if not re.search(rf"\bV{re.escape(VERSION)}\b", version):
                raise RuntimeError(f"guest compiler must use CUDA {VERSION}: {source}")
        if relative.endswith(".bc"):
            continue
        result = subprocess.run(
            ["ldd", str(source)], text=True, capture_output=True, check=True
        )
        for line in result.stdout.splitlines():
            if "not found" in line:
                raise RuntimeError(
                    f"unresolved guest compiler dependency: {line.strip()}"
                )
            paths = re.findall(r"(/[^\s()]+)", line)
            if paths:
                library = Path(paths[-1]).resolve(strict=True)
                destination = Path("lib64") / Path(paths[0]).name
                if destination in libraries and libraries[destination] != library:
                    raise RuntimeError(f"conflicting guest library: {destination}")
                libraries[destination] = library
    assets.extend(
        (source, destination) for destination, source in sorted(libraries.items())
    )
    manifest = {
        "compiler_version": VERSION,
        "execution_os": execution_os,
        "assets": {
            str(destination): {
                "bytes": source.stat().st_size,
                "sha256": digest(source),
            }
            for source, destination in assets
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "guest-compiler-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    assets.append((path, Path("usr/share/nvidia/compiler-manifest.json")))
    return assets
