#!/usr/bin/env python3
"""Train or evaluate MNIST inside D3OS and recover verified guest artifacts."""

import argparse
import binascii
import importlib.util
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_NAME = r"(?:mnist\.(?:checkpoint|model)|inference-(?:1|64|256)-logits\.f32|train-[0-2](?:\.model|-(?:losses|logits)\.f32))"
PATTERN = re.compile(
    rf"MNIST_ARTIFACT_(BEGIN name=({ARTIFACT_NAME}) bytes=(\d+) crc32=([0-9a-f]{{8}})|DATA name=({ARTIFACT_NAME}) offset=(\d+) hex=([0-9a-f]+)|END name=({ARTIFACT_NAME}))"
)


def extract_artifacts(text, output):
    pending = {}
    completed = {}
    for match in PATTERN.finditer(text):
        if match[2]:
            name, size, checksum = match[2], int(match[3]), int(match[4], 16)
            if name in pending or not 0 < size <= 16 << 20:
                raise RuntimeError("invalid or repeated artifact start")
            pending[name] = (size, checksum, bytearray())
        elif match[5]:
            name, offset = match[5], int(match[6])
            if name not in pending or len(match[7]) % 2:
                raise RuntimeError("artifact data without a valid start")
            size, checksum, data = pending[name]
            chunk = bytes.fromhex(match[7])
            if (
                offset != len(data)
                or not chunk
                or len(chunk) > 256
                or len(data) + len(chunk) > size
            ):
                raise RuntimeError(
                    "artifact chunk is missing, repeated, or outside the declared length"
                )
            data.extend(chunk)
        else:
            name = match[8]
            if name not in pending:
                raise RuntimeError("artifact end without a start")
            size, checksum, data = pending.pop(name)
            if len(data) != size or binascii.crc32(data) != checksum:
                raise RuntimeError("artifact transport length or checksum mismatch")
            if name.endswith(".f32"):
                if len(data) % 4 or any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", data)):
                    raise RuntimeError("invalid float reference artifact")
            else:
                kind = 1 if name.endswith("checkpoint") else 2
                if (
                    len(data) < 72
                    or data[:8] != b"D3MNIST1"
                    or struct.unpack_from("<I", data, 8)[0] != kind
                    or binascii.crc32(data[:-4])
                    != struct.unpack_from("<I", data, len(data) - 4)[0]
                ):
                    raise RuntimeError("invalid artifact container")
            temporary = output / f"{name}.tmp"
            temporary.write_bytes(data)
            temporary.replace(output / name)
            completed[name] = {"bytes": size, "crc32": f"{checksum:08x}"}
    return completed, sorted(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "target/mnist")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "target/nvidia/burn/mnist"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="stop at this global step; zero runs all epochs",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--override-learning-rate",
        action="store_true",
        help="explicitly change the resumed rate while preserving Adam state",
    )
    parser.add_argument(
        "--model", type=Path, help="evaluate an exported model instead of training"
    )
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    if args.override_learning_rate and not args.resume:
        parser.error("--override-learning-rate requires --resume")
    if args.resume and args.model:
        parser.error("use either --resume or --model")
    if (
        not 1 <= args.epochs <= 100
        or not 1 <= args.batch_size <= 4096
        or not 0 <= args.seed < 1 << 64
        or not 0 <= args.max_steps < 1 << 64
    ):
        parser.error("invalid training configuration")
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("use a positive finite learning rate and timeout")
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "prepare_mnist", ROOT / "tools/inference/prepare-mnist.py"
    )
    staging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staging)
    staging.prepare(args.data)
    files = {
        "mnist.test-images": args.data / "t10k-images-idx3-ubyte",
        "mnist.test-labels": args.data / "t10k-labels-idx1-ubyte",
    }
    mode = "gpu-mnist-infer" if args.model else "gpu-mnist-train"
    config = args.output / "mnist.config"
    config.write_bytes(
        struct.pack(
            "<QIIdQII",
            args.seed,
            args.batch_size,
            args.epochs,
            args.learning_rate,
            args.max_steps,
            int(args.resume is not None),
            int(args.override_learning_rate),
        )
    )
    files["mnist.config"] = config
    if args.model:
        files["mnist.model"] = args.model
    else:
        files.update(
            {
                "mnist.train-images": args.data / "train-images-idx3-ubyte",
                "mnist.train-labels": args.data / "train-labels-idx1-ubyte",
            }
        )
        if args.resume:
            files["mnist.checkpoint"] = args.resume
    command = [
        sys.executable,
        str(ROOT / "tools/nvidia/run-qemu.py"),
        "--firmware",
        str(ROOT / "target/nvidia/firmware"),
        "--kernels",
        str(ROOT / "target/nvidia/kernels"),
        "--memory-mib",
        "4096",
        "--boot-check",
        "--inference",
        mode,
        "--timeout",
        str(args.timeout),
    ]
    for name, path in files.items():
        command.extend(["--fwcfg-file", f"{name}={path.resolve()}"])
    serial_path = ROOT / "target/nvidia/vm/serial.log"
    previous_serial = serial_path.stat().st_mtime_ns if serial_path.exists() else None
    with (args.output / "vm.log").open("w") as log:
        result = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if not serial_path.is_file() or serial_path.stat().st_mtime_ns == previous_serial:
        raise RuntimeError(
            f"guest produced no serial log; see {args.output / 'vm.log'}"
        )
    serial = serial_path.read_text(errors="replace")
    (args.output / "serial.log").write_text(serial)
    artifacts, partial = extract_artifacts(serial, args.output)
    records = re.findall(
        r"MNIST_(?:START|LR_CHANGE|PROGRESS|EPOCH|RESULT)[^\r\n]*|INFERENCE_OK mnist[^\r\n]*",
        serial,
    )
    summary = {
        "os": "d3os",
        "backend": "nvidia",
        "host_training": False,
        "linux_measurement": False,
        "exit_code": result.returncode,
        "artifacts": artifacts,
        "partial_artifacts": partial,
        "records": records,
    }
    (args.output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    for record in records:
        print(record)
    if result.returncode:
        raise SystemExit(result.returncode)
    if partial or (
        not args.model and set(artifacts) != {"mnist.checkpoint", "mnist.model"}
    ):
        raise RuntimeError("guest did not export all complete artifacts")
    print(
        f"MNIST_GUEST_OK os=d3os backend=nvidia artifacts={len(artifacts)} output={args.output}"
    )


if __name__ == "__main__":
    main()
