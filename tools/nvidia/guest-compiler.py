"""Stage portable GPU input and the compiler runtime for execution inside D3OS."""

import hashlib
import importlib.util
import json
import re
import shutil
import struct
import subprocess
from pathlib import Path, PurePosixPath

FAMILIES = ("vector-add", "tensor-matmul", "inference")
VERSION = "13.4.92"
MODULE_MAGIC = b"D3QMOD01"
RUNTIME_MAGIC = b"D3QRT001"
MAX_ASSETS = 128
MAX_PATH_BYTES = 512


def qualification_module():
    spec = importlib.util.spec_from_file_location(
        "cubin_qualification", Path(__file__).with_name("cubin-qualification.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate manifest field: {name}")
        result[name] = value
    return result


def normalized_path(path):
    """Require one bounded relative POSIX spelling for each staged asset."""
    if not isinstance(path, str):
        raise TypeError("asset path must be a string")
    encoded = path.encode("utf-8")
    parsed = PurePosixPath(path)
    if (
        not encoded
        or len(encoded) > MAX_PATH_BYTES
        or parsed.is_absolute()
        or str(parsed) != path
        or ".." in parsed.parts
        or path == "."
        or "\\" in path
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
    ):
        raise ValueError(f"invalid normalized relative asset path: {path!r}")
    return encoded


def bundle_file(kernels, relative):
    normalized_path(relative)
    root = kernels.resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(
            f"bundle input is outside the bundle or is not a file: {relative}"
        )
    return path


def checked_qualification(kernels, family, source, record, qualifier):
    """Recompute evidence from recorded audit inputs before producing a receipt."""
    if not isinstance(record, dict):
        raise TypeError(f"{family}: invalid kernel manifest record")
    if record.get("input") != source.name:
        raise ValueError(f"{family}: manifest input does not match the staged source")
    source_identity = {
        "source_sha256": digest(source),
        "source_bytes": source.stat().st_size,
    }
    if (
        record.get("sha256") != source_identity["source_sha256"]
        or type(record.get("bytes")) is not int
        or record["bytes"] != source_identity["source_bytes"]
    ):
        raise ValueError(f"{family}: manifest source identity mismatch")
    audit = record.get("host_audit")
    qualification = record.get("legacy_qualification")
    if audit is None or qualification is None:
        raise ValueError(
            f"{family}: missing host qualification; regenerate the kernel bundle with --audit"
        )
    if not isinstance(audit, dict) or not isinstance(qualification, dict):
        raise TypeError(f"{family}: invalid host qualification record")
    if audit.get("architecture") != "sm_89" or audit.get("sass_verified") is not True:
        raise ValueError(f"{family}: unsupported host audit profile")
    cubin = bundle_file(kernels, audit.get("cubin")).read_bytes()
    sass_bytes = bundle_file(kernels, audit.get("sass")).read_bytes()
    if (
        audit.get("sha256") != hashlib.sha256(cubin).hexdigest()
        or type(audit.get("bytes")) is not int
        or audit["bytes"] != len(cubin)
    ):
        raise ValueError(f"{family}: host audit cubin identity mismatch")
    # Do not normalize line endings: the receipt binds the exact UTF-8 SASS file.
    actual = qualifier.qualify(cubin, sass_bytes.decode("utf-8")) | source_identity
    if json.dumps(actual, sort_keys=True) != json.dumps(qualification, sort_keys=True):
        raise ValueError(
            f"{family}: qualification does not match the actual source/cubin/SASS evidence"
        )
    return actual


def runtime_inventory(records):
    """Encode sorted path/length/digest records, excluding generated sidecars.

    Paths have no leading slash. SHA256 covers these records only, without
    the D3QRT001 header or count. Module receipts and manifests are not inputs.
    """
    if not 0 < len(records) <= MAX_ASSETS:
        raise ValueError(f"runtime inventory must contain 1..{MAX_ASSETS} assets")
    encoded = bytearray()
    for path, record in sorted(records.items()):
        path_bytes = normalized_path(path)
        if PurePosixPath(path).suffix in (".qualification", ".json"):
            raise ValueError(
                "runtime inventory cannot include qualification files or manifests"
            )
        size = record["bytes"]
        if type(size) is not int or not 0 < size < 1 << 64:
            raise ValueError("invalid runtime asset byte count")
        checksum = bytes.fromhex(record["sha256"])
        if len(checksum) != 32 or not any(checksum):
            raise ValueError("invalid runtime asset SHA256")
        encoded.extend(struct.pack("<I", len(path_bytes)))
        encoded.extend(path_bytes)
        encoded.extend(struct.pack("<Q", size))
        encoded.extend(checksum)
    return bytes(encoded)


def module_receipt(qualification, input_kind, compiler_sha256, runtime_sha256):
    if input_kind not in (1, 2):
        raise ValueError("unsupported qualification input kind")
    receipt = bytearray(MODULE_MAGIC + struct.pack("<6I", 1, 1, 89, 3, input_kind, 0))
    for checksum in (
        qualification["source_sha256"],
        qualification["cubin_sha256"],
        qualification["sass_sha256"],
        compiler_sha256,
        runtime_sha256,
    ):
        raw = bytes.fromhex(checksum)
        if len(raw) != 32 or not any(raw):
            raise ValueError("invalid module receipt SHA256")
        receipt.extend(raw)
    return bytes(receipt)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def runtime_library(name):
    cache = subprocess.check_output(
        [shutil.which("ldconfig") or "/sbin/ldconfig", "-p"], text=True
    )
    for line in cache.splitlines():
        description, separator, path = line.partition(" => ")
        if separator and description.split()[0] == name and "x86-64" in description:
            return Path(path).resolve(strict=True)
    raise RuntimeError(f"missing x86-64 guest runtime library: {name}")


def runtime_assets(sources):
    # glibc loads the unwinder lazily for pthread_exit; it is absent from the
    # compiler binaries' DT_NEEDED entries and therefore from their ldd output.
    unwinder = runtime_library("libgcc_s.so.1")
    libraries = {Path("lib64/libgcc_s.so.1"): unwinder}

    def dependencies(source):
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

    for source in [unwinder, *sources]:
        dependencies(source)
    return [(source, destination) for destination, source in sorted(libraries.items())]


def prepare(kernels, toolkit, directory, *, execution_os="d3os"):
    try:
        bundle = json.loads(
            (kernels / "manifest.json").read_text(), object_pairs_hook=unique_object
        )
        if not isinstance(bundle, dict) or not isinstance(bundle.get("kernels"), dict):
            raise TypeError("invalid kernel bundle manifest")
        if set(bundle["kernels"]) != set(FAMILIES):
            raise ValueError(
                "kernel bundle must contain exactly the three supported families"
            )
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"invalid qualification bundle; regenerate with --audit: {error}"
        ) from error
    qualifier = qualification_module()
    qualifications = {}
    assets = []
    for family in FAMILIES:
        inputs = [kernels / f"{family}.{suffix}" for suffix in ("tilebc", "ptx")]
        present = [path for path in inputs if path.is_file()]
        if len(present) != 1:
            raise RuntimeError(f"{family}: provide exactly one Tile IR or PTX input")
        try:
            qualifications[family] = checked_qualification(
                kernels, family, present[0], bundle["kernels"][family], qualifier
            )
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(
                f"{family}: invalid host qualification: {error}"
            ) from error
        assets.append((present[0], Path("usr/share/nvidia") / present[0].name))

    elf_sources = []
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
        elf_sources.append(source)
    assets.extend(runtime_assets(elf_sources))
    destinations = [destination.as_posix() for _, destination in assets]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("duplicate guest runtime asset destinations")
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
        "legacy_qualifications": qualifications,
    }
    canonical_records = runtime_inventory(manifest["assets"])
    runtime_sha256 = hashlib.sha256(canonical_records).hexdigest()
    manifest["runtime_inventory_sha256"] = runtime_sha256
    directory.mkdir(parents=True, exist_ok=True)
    runtime_path = directory / "runtime.qualification"
    runtime_path.write_bytes(
        RUNTIME_MAGIC + struct.pack("<I", len(assets)) + canonical_records
    )
    assets.append((runtime_path, Path("usr/share/nvidia/runtime.qualification")))
    for family in FAMILIES:
        source_name = bundle["kernels"][family]["input"]
        input_kind = 1 if source_name.endswith(".tilebc") else 2
        compiler = "tileiras" if input_kind == 1 else "ptxas"
        receipt_path = directory / f"{family}.qualification"
        receipt_path.write_bytes(
            module_receipt(
                qualifications[family],
                input_kind,
                manifest["assets"][f"usr/cuda/bin/{compiler}"]["sha256"],
                runtime_sha256,
            )
        )
        assets.append((receipt_path, Path("usr/share/nvidia") / receipt_path.name))
    path = directory / "guest-compiler-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    assets.append((path, Path("usr/share/nvidia/compiler-manifest.json")))
    return assets
