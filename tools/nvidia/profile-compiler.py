#!/usr/bin/env python3
"""Count guest user/kernel CPU events during a separate compiler diagnostic run."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# perf_event_attr flags and generic hardware events from linux/perf_event.h.
EXCLUDE_USER, EXCLUDE_KERNEL, EXCLUDE_HV, EXCLUDE_HOST = (1 << n for n in (4, 5, 6, 19))
EVENTS = {
    "user_cycles": (0, 0, EXCLUDE_KERNEL),
    "user_instructions": (0, 1, EXCLUDE_KERNEL),
    "user_ref_cycles": (0, 9, EXCLUDE_KERNEL),
    "user_cache_misses": (0, 3, EXCLUDE_KERNEL),
    "user_branch_misses": (0, 5, EXCLUDE_KERNEL),
    "user_dtlb_read_misses": (3, 3 | (1 << 16), EXCLUDE_KERNEL),
    "kernel_cycles": (0, 0, EXCLUDE_USER),
}
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.syscall.restype = ctypes.c_long


class Counter:
    def __init__(self, thread, kind, config, exclude):
        attributes = ctypes.create_string_buffer(128)
        struct.pack_into("<IIQ", attributes, 0, kind, 128, config)
        # Include enabled/running time so multiplexed counts remain auditable.
        struct.pack_into("<Q", attributes, 32, 3)
        struct.pack_into("<Q", attributes, 40, EXCLUDE_HOST | EXCLUDE_HV | exclude)
        self.fd = LIBC.syscall(298, ctypes.byref(attributes), thread, -1, -1, 8)
        if self.fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))

    def read(self):
        value, enabled, running = struct.unpack("<QQQ", os.read(self.fd, 24))
        return {"value": value, "enabled_ns": enabled, "running_ns": running}

    def close(self):
        os.close(self.fd)


def counter_delta(before, after):
    delta = {key: after[key] - before[key] for key in ("value", "enabled_ns", "running_ns")}
    if min(delta.values()) < 0 or delta["running_ns"] > delta["enabled_ns"]:
        raise ValueError("performance counter interval is inconsistent")
    if delta["running_ns"] == 0:
        if delta["value"]:
            raise ValueError("performance counter advanced without running")
        return 0.0
    return delta["value"] * delta["enabled_ns"] / delta["running_ns"]


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if row["phase"] == "inference"]
    if len(rows) < 2:
        raise ValueError("inference compilation was not sampled")
    begin, end = rows[0], rows[-1]
    counts = {}
    for cpu, first in begin["cpus"].items():
        counts[cpu] = {
            name: counter_delta(before, end["cpus"][cpu]["events"][name])
            for name, before in first["events"].items()
        }
    complete = set.intersection(*(set(events) for events in counts.values()))
    totals = {name: sum(events[name] for events in counts.values()) for name in complete}
    if totals.get("user_instructions", 0) <= 0 or totals.get("user_cycles", 0) <= 0:
        raise ValueError("guest user-mode instructions and cycles were not counted")
    return {
        "inference_window_s": (end["monotonic_ns"] - begin["monotonic_ns"]) / 1e9,
        "samples": len(rows),
        "per_cpu": counts,
        "total_counts": totals,
        "user_ipc": totals["user_instructions"] / totals["user_cycles"],
        "kernel_cycle_percent": 100 * totals["kernel_cycles"] / (totals["kernel_cycles"] + totals["user_cycles"]),
        "user_cycles_per_ref_cycle": totals["user_cycles"] / totals["user_ref_cycles"] if totals.get("user_ref_cycles", 0) else None,
    }


def phase(text, execution_os):
    pattern = (
        r"NVIDIA_COMPILE_MODULE os=d3os family=([\w-]+)"
        if execution_os == "d3os"
        else r"^LINUX_COMPILE_MODULE [^\r\n]* family=([\w-]+)"
    )
    completed = set(re.findall(pattern, text, re.MULTILINE))
    if "inference" in completed:
        return "done"
    if "tensor-matmul" in completed:
        return "inference"
    if "vector-add" in completed:
        return "tensor-matmul"
    return "boot_or_vector-add"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--os", dest="execution_os", choices=("linux", "d3os"), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=480)
    args = parser.parse_args()
    if platform.machine() != "x86_64" or len(os.sched_getaffinity(0)) != 4:
        parser.error("use an x86-64 host with an inherited affinity of four CPUs")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    output = args.directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    (output / "profile-source.py").write_bytes(source)
    if args.execution_os == "linux":
        guest = output / "guest"
        command = [
            sys.executable, str(ROOT / "tools/nvidia/linux-reference.py"),
            "--compiler-benchmark", "--compiler-repeats", "1", "--compiler-cpus", "0-3",
            "--directory", str(guest), "--timeout", str(args.timeout), "--pin-vcpus",
        ]
    else:
        guest = ROOT / "target/nvidia/vm"
        command = [
            sys.executable, str(ROOT / "tools/nvidia/run-qemu.py"),
            "--firmware", str(ROOT / "target/nvidia/firmware"),
            "--kernels", str(ROOT / "target/nvidia/kernels"), "--pin-vcpus",
            "--boot-check", "--inference", "gpu-alexnet-bench", "--timeout", "90",
        ]
    manifest = guest / "vcpu-affinity.json"
    serial = guest / "serial.log"
    counters = {}
    errors = {}
    started_ns = time.time_ns()
    deadline = time.monotonic() + args.timeout
    with (output / "launcher.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"launcher exited before profiling: {output / 'launcher.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError("QEMU did not publish its CPU mapping")
                try:
                    if manifest.stat().st_mtime_ns >= started_ns:
                        mapping = json.loads(manifest.read_text())
                        break
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                time.sleep(0.05)
            for cpu in mapping["vcpus"]:
                thread = cpu["thread_id"]
                if os.getpgid(thread) != process.pid:
                    raise RuntimeError("CPU mapping does not belong to this diagnostic VM")
                counters[cpu["guest_cpu"]] = {}
                for name, arguments in EVENTS.items():
                    try:
                        counters[cpu["guest_cpu"]][name] = Counter(thread, *arguments)
                    except OSError as error:
                        if name in ("user_cycles", "user_instructions", "kernel_cycles"):
                            raise RuntimeError(f"guest CPU counters unavailable: {error}") from error
                        errors[f"{cpu['guest_cpu']}:{name}"] = str(error)
            metadata = {
                "execution_os": args.execution_os, "instrumented": True,
                "phase": "cold_compilation", "memory_mib": 4096,
                "guest_only": True, "events": EVENTS, "affinity": mapping,
                "optional_counter_errors": errors, "command": command,
                "source_sha256": hashlib.sha256(source).hexdigest(),
            }
            (output / "profile.json").write_text(json.dumps(metadata, indent=2) + "\n")
            with (output / "samples.jsonl").open("w") as samples:
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError("compiler profiling timed out")
                    text = serial.read_text(errors="replace") if serial.exists() else ""
                    current = phase(text, args.execution_os)
                    if current != "done":
                        row = {"monotonic_ns": time.monotonic_ns(), "phase": current, "cpus": {}}
                        for cpu in mapping["vcpus"]:
                            host = cpu["host_cpu"]
                            frequency = Path(f"/sys/devices/system/cpu/cpu{host}/cpufreq/scaling_cur_freq")
                            row["cpus"][cpu["guest_cpu"]] = {
                                "host_frequency_khz": int(frequency.read_text()) if frequency.exists() else None,
                                "events": {name: counter.read() for name, counter in counters[cpu["guest_cpu"]].items()},
                            }
                        samples.write(json.dumps(row) + "\n")
                        samples.flush()
                    time.sleep(0.25)
            if process.returncode:
                raise RuntimeError(f"compiler diagnostic failed: {output / 'launcher.log'}")
        finally:
            for events in counters.values():
                for counter in events.values():
                    counter.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            if args.execution_os == "d3os":
                for name in ("serial.log", "vcpu-affinity.json", "guest-compiler-manifest.json", "qemu.log"):
                    source = guest / name
                    if source.exists() and source.stat().st_mtime_ns >= started_ns:
                        shutil.copyfile(source, output / name)
    summary = {"execution_os": args.execution_os, "instrumented": True, **summarize(output / "samples.jsonl")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"COMPILER_PROFILE_OK os={args.execution_os} instrumented=true output={output}")


if __name__ == "__main__":
    main()
