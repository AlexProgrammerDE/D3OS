#!/usr/bin/env python3
"""Compile Rust GPU kernels and execute their PTX on a CPU through GPUOcelot."""

import argparse
import ctypes
import hashlib
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://github.com/tinygrad/gpuocelot/releases/download/v0.1.0/libgpuocelot.so"
SHA256 = "a24705276a9a187111371465987b3258f8836ef512a34266e3075bc4714e125a"
KINDS = ("vector-add", "relu", "matmul")


def nonexecutable_stack(data):
    """The upstream ELF has a legacy executable-stack flag. Remove that request.

    The interpreter uses no stack trampolines. This keeps the host's stack
    protection enabled, including on systems that reject executable stacks.
    """
    result = bytearray(data)
    if result[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected an ELF64 little-endian interpreter")
    offset = struct.unpack_from("<Q", result, 32)[0]
    size, count = struct.unpack_from("<HH", result, 54)
    for index in range(count):
        position = offset + index * size
        if struct.unpack_from("<I", result, position)[0] == 0x6474E551:  # PT_GNU_STACK
            flags = struct.unpack_from("<I", result, position + 4)[0]
            struct.pack_into("<I", result, position + 4, flags & ~1)  # clear PF_X
    return bytes(result)


def interpreter(directory):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            "provide --library with a locally built GPUOcelot library on this platform"
        )
    original = directory / "libgpuocelot.upstream.so"
    if not original.exists():
        temporary = original.with_suffix(".partial")
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("GPUOcelot download checksum mismatch")
        temporary.write_bytes(data)
        temporary.replace(original)
    data = original.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("GPUOcelot cache checksum mismatch")
    library = directory / "libgpuocelot.so"
    library.write_bytes(nonexecutable_stack(data))
    return library


def execute(library, ptx, buffers, scalars, count):
    run = ctypes.CDLL(str(library)).ptx_run
    run.restype = None
    run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)] + [
        ctypes.c_int
    ] * 7
    addresses = [ctypes.addressof(buffer) for buffer in buffers]
    # Output has a guard on either side. The kernel receives the first data element.
    addresses[-1] += ctypes.sizeof(ctypes.c_float)
    arguments = (ctypes.c_void_p * (len(addresses) + len(scalars)))(
        *addresses, *scalars
    )
    block = 32
    run(
        ptx,
        len(arguments),
        arguments,
        block,
        1,
        1,
        max(1, (count + block - 1) // block),
        1,
        1,
        0,
    )


def check(actual, expected):
    if actual[0] != 12345.0 or actual[-1] != -12345.0:
        raise AssertionError("kernel wrote outside its output allocation")
    for index, (a, b) in enumerate(zip(actual[1:-1], expected, strict=True)):
        if not math.isfinite(a) or abs(a - b) > 2e-5 + abs(b) * 2e-5:
            raise AssertionError(f"element {index}: {a} != {b}")


def values(count, salt):
    return (ctypes.c_float * count)(*(((i * salt) % 23 - 11) / 8 for i in range(count)))


def output(count):
    return (ctypes.c_float * (count + 2))(12345.0, *([math.nan] * count), -12345.0)


def worker(kind, ptx, library):
    source = ptx.read_bytes()
    cases = 0
    if kind in ("vector-add", "relu"):
        for count in (0, 1, 7, 31, 32, 33, 127, 257):
            a, result = values(count, 7), output(count)
            if kind == "vector-add":
                b = values(count, 13)
                expected = [ctypes.c_float(x + y).value for x, y in zip(a, b)]
                buffers = [a, b, result]
            else:
                expected = [max(x, 0) for x in a]
                buffers = [a, result]
            execute(library, source, buffers, [count], count)
            check(result, expected)
            cases += 1
    else:
        for rows, inner, cols in (
            (1, 1, 1),
            (2, 3, 4),
            (5, 9, 11),
            (3, 33, 7),
            (7, 4, 9),
            (1, 65, 33),
        ):
            a, b, result = (
                values(rows * inner, 3),
                values(inner * cols, 5),
                output(rows * cols),
            )
            expected = [
                sum(a[row * inner + k] * b[k * cols + col] for k in range(inner))
                for row in range(rows)
                for col in range(cols)
            ]
            execute(library, source, [a, b, result], [rows, inner, cols], rows * cols)
            check(result, expected)
            cases += 1
    print(f"GPU_PTX_OK kernel={kind} cases={cases}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware-sm",
        choices=("sm_80", "sm_86", "sm_89"),
        default="sm_86",
        help="compile a separate hardware artifact for Ampere or Ada (default: sm_86)",
    )
    parser.add_argument(
        "--library",
        type=Path,
        help="use an existing GPUOcelot library instead of the checksum-pinned download",
    )
    parser.add_argument("--worker", choices=KINDS, help=argparse.SUPPRESS)
    parser.add_argument("--ptx", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.ptx, args.library.resolve())
        return
    directory = ROOT / "target/gpu-sim"
    directory.mkdir(parents=True, exist_ok=True)
    library = args.library.resolve() if args.library else interpreter(directory)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(ROOT / "target/gpu-kernels")
    # Interpreter runs PTX 3.2. The same Rust source also compiles to modern PTX
    # for later hardware assembly. No generated instructions are rewritten.
    for kind in KINDS:
        for architecture in ("sm_35", args.hardware_sm):
            subprocess.run(
                [
                    "cargo",
                    "rustc",
                    "--locked",
                    "--manifest-path",
                    str(ROOT / "tools/gpu-kernels/Cargo.toml"),
                    "--target",
                    "nvptx64-nvidia-cuda",
                    "--release",
                    "--no-default-features",
                    "--features",
                    kind,
                    "--",
                    "-C",
                    f"target-cpu={architecture}",
                ],
                cwd=ROOT,
                env=env,
                check=True,
                timeout=120,
            )
            destination = directory / architecture
            destination.mkdir(exist_ok=True)
            ptx = destination / f"{kind}.ptx"
            shutil.copyfile(
                Path(env["CARGO_TARGET_DIR"])
                / "nvptx64-nvidia-cuda/release/d3os_gpu_kernels.ptx",
                ptx,
            )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                kind,
                "--ptx",
                str(directory / "sm_35" / f"{kind}.ptx"),
                "--library",
                str(library),
            ],
            cwd=directory,
            check=True,
            timeout=60,
        )
    print(f"GPU_PTX_BUILD_OK target={args.hardware_sm}")
    print("GPU_PTX_SUITE_OK kernels=3 cases=22 hardware_validated=false")


if __name__ == "__main__":
    main()
