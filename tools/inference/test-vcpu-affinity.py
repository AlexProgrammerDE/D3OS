#!/usr/bin/env python3
"""Verify vCPU affinity against real paused QEMU threads without GPU hardware."""

import importlib.util
import os
import subprocess
import tempfile
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "inference_qemu", Path(__file__).with_name("test-qemu.py")
)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def check(directory, count, reject, threading="multi"):
    monitor = directory / f"{count}-{threading}.sock"
    command = [
        "qemu-system-x86_64",
        "-machine",
        "q35",
        "-accel",
        f"tcg,thread={threading}",
        "-smp",
        str(count),
        "-m",
        "64",
        "-nodefaults",
        "-display",
        "none",
        "-S",
        "-qmp",
        f"unix:{monitor},server=on,wait=off",
    ]
    with (directory / f"{count}-{threading}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        qmp = None
        try:
            qmp = harness.Qmp(monitor, process, time.monotonic() + 10)
            guests = qmp.execute("query-cpus-fast")
            before = {
                cpu["thread-id"]: os.sched_getaffinity(cpu["thread-id"])
                for cpu in guests
            }
            if reject:
                try:
                    harness.pin_vcpus(qmp)
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid vCPU mapping was accepted")
                for thread, affinity in before.items():
                    assert os.sched_getaffinity(thread) == affinity
            else:
                mapping = harness.pin_vcpus(qmp)
                hosts = sorted(os.sched_getaffinity(0))
                assert [row["guest_cpu"] for row in mapping["vcpus"]] == list(range(count))
                assert [row["host_cpu"] for row in mapping["vcpus"]] == hosts
                for row in mapping["vcpus"]:
                    assert os.sched_getaffinity(row["thread_id"]) == {row["host_cpu"]}
            qmp.execute("quit")
            assert process.wait(timeout=10) == 0
        finally:
            if qmp is not None:
                qmp.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


def main():
    original = os.sched_getaffinity(0)
    hosts = sorted(original)[:4]
    try:
        os.sched_setaffinity(0, hosts)
        with tempfile.TemporaryDirectory(prefix="d3os-affinity-") as temporary:
            directory = Path(temporary)
            check(directory, len(hosts), False)
            check(directory, len(hosts) + 1, True)
            if len(hosts) > 1:
                check(directory, len(hosts), True, threading="single")
    finally:
        os.sched_setaffinity(0, original)
    print(f"VCPU_AFFINITY_OK host_cpus={hosts} mapping=verified mismatch=unchanged")


if __name__ == "__main__":
    main()
