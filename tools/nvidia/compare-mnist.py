#!/usr/bin/env python3
"""Compare matched MNIST timings, all logits, losses, and all eight trained parameters."""

import argparse
import binascii
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Preserve the existing native convolution reference tolerance.
TOLERANCE = 2e-5


def values(path):
    data = path.read_bytes()
    if len(data) % 4:
        raise RuntimeError(f"partial FP32 value: {path}")
    return [value[0] for value in struct.iter_unpack("<f", data)]


def difference(reference, actual):
    if len(reference) != len(actual) or not reference:
        raise RuntimeError("MNIST numerical reference length differs")
    absolute, scaled, failures = 0.0, 0.0, 0
    for expected, observed in zip(reference, actual):
        if not math.isfinite(expected) or not math.isfinite(observed):
            raise RuntimeError("non-finite MNIST comparison value")
        error = abs(expected - observed)
        normalized = error / (TOLERANCE * (1.0 + abs(expected)))
        absolute, scaled = max(absolute, error), max(scaled, normalized)
        failures += normalized > 1.0
    return {"values": len(reference), "max_abs": absolute, "max_scaled_error": scaled, "tolerance_failures": failures}


def model_parameters(path, inspector):
    data = path.read_bytes()
    if len(data) < 72 or data[:8] != b"D3MNIST1":
        raise RuntimeError(f"invalid MNIST model header: {path}")
    kind, metadata, weights, optimizer = struct.unpack_from("<IIQQ", data, 8)
    if (kind != 2 or metadata != 36 or optimizer != 0
        or len(data) != 32 + metadata + weights + 4
        or binascii.crc32(data[:-4]) != struct.unpack_from("<I", data, len(data) - 4)[0]):
        raise RuntimeError(f"invalid MNIST model container: {path}")
    result = subprocess.run(
        [str(inspector)], input=data[32 + metadata:32 + metadata + weights], capture_output=True, check=True,
    )
    tensors = json.loads(result.stdout)
    if len(tensors) != 8:
        raise RuntimeError("MNIST must retain all eight model parameters")
    return data[32:32 + metadata], tensors


def compare(linux, native, inspector, oracle=None):
    left, right = (json.loads((directory / "results.json").read_text()) for directory in (linux, native))
    if left["execution_os"] != "linux" or right["execution_os"] != "d3os":
        raise RuntimeError("supply the Linux and D3OS result directories in order")
    if not left.get("performance_comparison", False) or not right.get("performance_comparison", False):
        raise RuntimeError("a diagnostic or excluded run cannot establish performance parity")
    if left["inputs"] != right["inputs"]:
        raise RuntimeError("MNIST model or dataset bytes differ")
    shared_sources = (
        "os/library/ai/src/mnist.rs", "os/library/ai/src/mnist/training.rs",
        "os/library/ai/src/mnist/artifact.rs", "os/library/ai/src/mnist/data.rs",
    )
    for name in shared_sources:
        if left["sources"][name] != right["sources"][name]:
            raise RuntimeError(f"shared MNIST source differs: {name}")
    for key in ("vcpus", "memory_mib", "host_affinity"):
        if left[key] != right[key]:
            raise RuntimeError(f"VM comparison differs: {key}")
    ignored = {"execution_os", "backend", "model_load_ms"}
    contracts = [{key: value for key, value in result["contract"].items() if key not in ignored} for result in (left, right)]
    if contracts[0] != contracts[1]:
        raise RuntimeError("MNIST execution contracts differ")
    numerical_directory = linux / "artifacts"
    numerical_reference = {"execution_location": "linux_vm", "backend": "cuda", "arithmetic_policy": "cubecl_default_tf32_allowed"}
    if oracle is not None:
        oracle_manifest = json.loads((oracle / "results.json").read_text())
        oracle_contract = oracle_manifest["contract"]
        if not oracle_manifest["reference_only"] or oracle_manifest["backend"] != "flex_cpu" or oracle_manifest["inputs"] != left["inputs"]:
            raise RuntimeError("CPU numerical oracle uses different inputs or is not a reference run")
        if any(oracle_manifest["sources"][name] != left["sources"][name] for name in shared_sources):
            raise RuntimeError("CPU numerical oracle uses different shared model or trainer source")
        if {key: value for key, value in oracle_contract.items() if key not in ignored} != contracts[0]:
            raise RuntimeError("CPU numerical oracle uses a different training or inference contract")
        for name, record in oracle_manifest["artifacts"].items():
            if hashlib.sha256((oracle / "artifacts" / name).read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError(f"CPU oracle changed: {name}")
        numerical_directory = oracle / "artifacts"
        numerical_reference = {"execution_location": "linux_host", "backend": "flex_cpu", "arithmetic_policy": "fp32", "timings_used": False}
    for directory, result in ((linux, left), (native, right)):
        for name, record in result["artifacts"].items():
            path = directory / "artifacts" / name
            if path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError(f"MNIST evidence changed: {path}")
    numerical = {}
    reference_files = [f"inference-{batch}-logits.f32" for batch in (1, 64, 256)]
    reference_files.extend(f"train-{run}-{kind}.f32" for run in range(3) for kind in ("losses", "logits"))
    for name in reference_files:
        numerical[name] = difference(values(numerical_directory / name), values(native / "artifacts" / name))
    initial_model = linux / "guest/root/mnist.model"
    if hashlib.sha256(initial_model.read_bytes()).hexdigest() != left["inputs"]["mnist.model"]["sha256"]:
        raise RuntimeError("initial MNIST model changed after staging")
    initial_config, initial_parameters = model_parameters(initial_model, inspector)
    updates = []
    for run in range(3):
        expected_config, expected = model_parameters(numerical_directory / f"train-{run}.model", inspector)
        actual_config, actual = model_parameters(native / "artifacts" / f"train-{run}.model", inspector)
        if expected_config != actual_config or expected_config != initial_config or set(expected) != set(actual) or set(expected) != set(initial_parameters):
            raise RuntimeError("MNIST model configuration or parameter names differ")
        for name, tensor in expected.items():
            if tensor["shape"] != actual[name]["shape"]:
                raise RuntimeError(f"MNIST parameter shape differs: {name}")
            numerical[f"train-{run}:{name}"] = difference(tensor["values"], actual[name]["values"])
        changed = {
            backend: sum(value != original for name, tensor in parameters.items() for value, original in zip(tensor["values"], initial_parameters[name]["values"]))
            for backend, parameters in (("reference", expected), ("d3os", actual))
        }
        if not all(changed.values()):
            raise RuntimeError("MNIST training did not change model parameters")
        updates.append({"run": run, "changed_parameters": changed})
    timings = []
    for mode, batch in (("inference", 1), ("inference", 64), ("inference", 256), ("train", 64)):
        cohorts = []
        for result in (left, right):
            times = [float(row["elapsed_ms"]) for row in result["measurements"] if row["mode"] == mode and int(row["batch"]) == batch]
            if len(times) != 3 or any(not math.isfinite(time) or time <= 0 for time in times):
                raise RuntimeError("missing or invalid MNIST timing cohort")
            cohorts.append({"elapsed_ms": times, "mean_ms": sum(times) / len(times)})
        timings.append({"mode": mode, "batch": batch, "linux": cohorts[0], "d3os": cohorts[1], "d3os_over_linux": cohorts[1]["mean_ms"] / cohorts[0]["mean_ms"]})
    failures = sum(record["tolerance_failures"] for record in numerical.values())
    return {
        "tolerance": {"absolute": TOLERANCE, "relative": TOLERANCE},
        "numerical": numerical, "tolerance_failures": failures, "correctness_passed": failures == 0,
        "build_toolchains": {"linux": left["build_toolchain"], "d3os": right["build_toolchain"]},
        "training_updates": updates,
        "numerical_reference": numerical_reference,
        "performance_precision_matched": False,
        "performance_arithmetic": {"linux": "cubecl_default_tf32_allowed", "d3os": "fp32_tf32x3"},
        "timings": timings, "contract": contracts[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linux", type=Path, required=True)
    parser.add_argument("--d3os", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspector", type=Path, default=ROOT / "tools/mnist-benchmark/target/release/mnist-inspect-pack")
    parser.add_argument("--oracle", type=Path, help="strict FP32 CPU numerical oracle; its timings never enter the GPU comparison")
    args = parser.parse_args()
    result = compare(args.linux.resolve(), args.d3os.resolve(), args.inspector.resolve(), args.oracle.resolve() if args.oracle else None)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["timings"]:
        print(f"MNIST_COMPARE mode={row['mode']} batch={row['batch']} linux_mean_ms={row['linux']['mean_ms']:.6f} d3os_mean_ms={row['d3os']['mean_ms']:.6f} d3os_over_linux={row['d3os_over_linux']:.3f}")
    print(f"MNIST_REFERENCE values={sum(row['values'] for row in result['numerical'].values())} tolerance_failures={result['tolerance_failures']}")
    if result["tolerance_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
