#!/usr/bin/env python3
"""Capture real QEMU framebuffer output from native NVIDIA workloads."""

import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def wait_for(serial, process, expected, deadline):
    while time.monotonic() < deadline:
        text = serial.read_text(errors="replace") if serial.exists() else ""
        if any(
            marker in text
            for marker in (
                "NVIDIA initialization failed:",
                "NVIDIA_DOOM_FAIL",
                "INFERENCE_FAIL",
                "panicked at",
            )
        ):
            raise RuntimeError(f"guest failed: {serial}")
        if expected in text:
            return text
        if process.poll() is not None:
            raise RuntimeError("QEMU launcher exited before the capture")
        time.sleep(0.1)
    raise TimeoutError(f"waiting for {expected}: {serial}")


def record_qwen(qmp, serial, process, args):
    movie = args.output / "qwen-nvidia.mp4"
    ppm = args.output / "capture-frame.ppm"
    command = f"infer gpu-qwen {args.max_tokens} {args.prompt}"
    encoder_command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "image2pipe",
        "-vcodec",
        "ppm",
        "-framerate",
        str(args.fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(movie),
    ]
    typed = 0
    completed = None
    frames = 0
    with (args.output / "qwen-encoder.log").open("w") as log:
        encoder = subprocess.Popen(encoder_command, stdin=subprocess.PIPE, stderr=log)
        start = time.monotonic()
        try:
            for frame in range(args.seconds * args.fps):
                # Actual QMP keyboard input and framebuffer capture share one
                # connection. One character per frame keeps the prompt visible.
                if frame >= args.fps and typed <= len(command):
                    if typed == len(command):
                        key = "ret"
                    else:
                        char = command[typed]
                        key = {
                            " ": "spc",
                            "-": "slash",
                            "y": "z",
                            "z": "y",
                            "?": "shift-minus",
                            ".": "dot",
                            ",": "comma",
                        }.get(char.lower(), char.lower())
                        if char.isupper():
                            key = f"shift-{key}"
                    qmp.execute(
                        "human-monitor-command", {"command-line": f"sendkey {key} 30"}
                    )
                    typed += 1
                qmp.execute(
                    "human-monitor-command", {"command-line": f"screendump {ppm}"}
                )
                encoder.stdin.write(ppm.read_bytes())
                frames += 1
                text = serial.read_text(errors="replace")
                if process.poll() is not None or any(
                    marker in text for marker in ("INFERENCE_FAIL", "panicked at")
                ):
                    raise RuntimeError("Qwen recording failed; inspect qwen-serial.log")
                if "INFERENCE_OK qwen backend=nvidia" in text and completed is None:
                    completed = frame
                    Image.open(ppm).save(args.output / "qwen-complete.png")
                time.sleep(max(0, start + (frame + 1) / args.fps - time.monotonic()))
                if completed is not None and frame - completed >= 3 * args.fps:
                    break
            elapsed = time.monotonic() - start
        finally:
            encoder.stdin.close()
            encoder.wait(timeout=30)
    if encoder.returncode or completed is None or abs(elapsed - frames / args.fps) > 1:
        raise RuntimeError(
            "recording failed, did not finish, or could not maintain real time"
        )
    match = re.search(
        r"QWEN_STREAM tokens=(\d+) first_token_ms=(\d+) decode_ms=(\d+) tokens_per_second=([\d.]+)",
        text,
    )
    transcript = re.search(
        r"QWEN_TRANSCRIPT_BEGIN\n(.*?)\nQWEN_TRANSCRIPT_END", text, re.DOTALL
    )
    if match is None or transcript is None:
        raise RuntimeError("missing measured generation evidence")
    (args.output / "qwen-essay.txt").write_text(transcript[1] + "\n")
    ppm.unlink(missing_ok=True)
    return {
        "prompt": args.prompt,
        "command": command,
        "backend": "nvidia",
        "model": "Qwen3.5-0.8B",
        "max_tokens": args.max_tokens,
        "tokens": int(match[1]),
        "first_token_ms": int(match[2]),
        "decode_ms": int(match[3]),
        "tokens_per_second": float(match[4]),
        "capture_fps": args.fps,
        "frames": frames,
        "elapsed_seconds": elapsed,
        "video_seconds": frames / args.fps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=("qwen", "doom"))
    parser.add_argument("--output", type=Path, default=ROOT / "target/nvidia/captures")
    parser.add_argument(
        "--seconds",
        type=int,
        help="duration for DOOM, maximum duration for Qwen (defaults: 30 and 120)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--video", action="store_true", help="record Qwen typing and streaming output"
    )
    parser.add_argument(
        "--prompt",
        default="Write a very long essay about the importance of hand washing in as much detail as possible. Aim for at least 2500 words. Explain the science, everyday situations, proper technique, common mistakes, and benefits for communities.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.seconds is None:
        args.seconds = 120 if args.workload == "qwen" and args.video else 30
    if not 1 <= args.seconds <= 300 or not 1 <= args.fps <= 60:
        parser.error("use 1 to 300 seconds and 1 to 60 capture frames per second")
    if args.max_tokens <= 0 or not re.fullmatch(r"[a-zA-Z0-9 .,?\-]+", args.prompt):
        parser.error(
            "use a positive token limit and a prompt supported by the guest keyboard mapping"
        )
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "inference_qemu", ROOT / "tools/inference/test-qemu.py"
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    directory = ROOT / "target/nvidia/vm"
    serial = directory / "serial.log"
    command = [
        sys.executable,
        str(ROOT / "tools/nvidia/run-qemu.py"),
        "--firmware",
        str(ROOT / "target/nvidia/firmware"),
        "--kernels",
        str(ROOT / "target/nvidia/kernels"),
        "--memory-mib",
        "12288" if args.workload == "qwen" else "2048",
        "--inference",
        f"gpu-{args.workload}",
    ]
    # The launcher owns VFIO setup and VM cleanup. It leaves QMP available when
    # --boot-check is absent, so this process can drive input and capture pixels.
    with (args.output / f"{args.workload}-launcher.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True
        )
        qmp = None
        try:
            deadline = time.monotonic() + 300
            # Wait for this invocation's serial file, not the previous VM log.
            while True:
                log.flush()
                if (
                    "NVIDIA_VM_RUNNING"
                    in (args.output / f"{args.workload}-launcher.log").read_text()
                ):
                    break
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("QEMU launcher did not start")
                time.sleep(0.1)
            qmp = harness.Qmp(directory / "qmp.sock", process, deadline)
            text = wait_for(serial, process, "name = shell", deadline)
            for marker in (
                "NVIDIA_KERNEL_OK",
                "NVIDIA_TENSOR_OK kernel=tensor_matmul_f16",
                "NVIDIA_TRANSFER_OK",
            ):
                if marker not in text:
                    raise RuntimeError(f"missing hardware boot evidence: {marker}")
            time.sleep(1)
            if args.workload == "qwen" and args.video:
                manifest = record_qwen(qmp, serial, process, args)
            elif args.workload == "qwen":
                question = "What country is largest by size? Answer briefly."
                qmp.type_command(f"infer gpu-qwen 32 {question}")
                text = wait_for(
                    serial, process, "INFERENCE_OK qwen backend=nvidia", deadline
                )
                time.sleep(0.5)
                ppm = args.output / "qwen-largest-country.ppm"
                qmp.execute(
                    "human-monitor-command", {"command-line": f"screendump {ppm}"}
                )
                Image.open(ppm).save(args.output / "qwen-largest-country.png")
                manifest = {
                    "question": question,
                    "backend": "nvidia",
                    "model": "Qwen3.5-0.8B",
                }
            else:
                qmp.type_command("doom --gpu -warp 1 1 -skill 2")
                wait_for(serial, process, "NVIDIA_DOOM_FRAME_OK frame=60", deadline)
                ppm = args.output / "capture-frame.ppm"
                movie = args.output / f"doom-nvidia-{args.seconds}s.mp4"
                # Each frame comes directly from QEMU. The HUD is drawn inside
                # D3OS. Encoding adds no labels, synthesized frames, or speedups.
                encoder_command = [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "ppm",
                    "-framerate",
                    str(args.fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(movie),
                ]
                controls = [
                    (0, "up", 1500),
                    (2, "left", 500),
                    (3, "spc", 2000),
                    (6, "right", 800),
                    (8, "up", 1200),
                    (10, "spc", 1800),
                    (13, "left", 600),
                    (15, "up", 1000),
                    (17, "right", 600),
                    (19, "spc", 2000),
                    (22, "left", 800),
                    (24, "up", 1000),
                    (26, "spc", 1800),
                    (28, "right", 500),
                ]
                event = 0
                with (args.output / "doom-encoder.log").open("w") as encoder_log:
                    encoder = subprocess.Popen(
                        encoder_command, stdin=subprocess.PIPE, stderr=encoder_log
                    )
                    start = time.monotonic()
                    try:
                        for frame in range(args.seconds * args.fps):
                            elapsed = time.monotonic() - start
                            while (
                                event < len(controls) and controls[event][0] <= elapsed
                            ):
                                _, key, hold = controls[event]
                                qmp.execute(
                                    "human-monitor-command",
                                    {"command-line": f"sendkey {key} {hold}"},
                                )
                                event += 1
                            qmp.execute(
                                "human-monitor-command",
                                {"command-line": f"screendump {ppm}"},
                            )
                            encoder.stdin.write(ppm.read_bytes())
                            if frame == args.fps:
                                Image.open(ppm).save(
                                    args.output / "doom-nvidia-hud.png"
                                )
                            time.sleep(
                                max(
                                    0, start + (frame + 1) / args.fps - time.monotonic()
                                )
                            )
                    finally:
                        encoder.stdin.close()
                        encoder.wait(timeout=30)
                    elapsed = time.monotonic() - start
                    if encoder.returncode or elapsed > args.seconds + 1:
                        raise RuntimeError(
                            f"recording failed or exceeded real-time duration: elapsed={elapsed:.3f}s"
                        )
                text = serial.read_text(errors="replace")
                if (
                    "verify=false" not in text
                    or "NVIDIA_DOOM_INPUT" not in text
                    or "NVIDIA_DOOM_FAIL" in text
                ):
                    raise RuntimeError("recording lacks native GPU gameplay evidence")
                rates = [
                    int(value)
                    for value in re.findall(r"NVIDIA_DOOM_FPS fps=(\d+)", text)
                ]
                manifest = {
                    "backend": "nvidia",
                    "seconds": args.seconds,
                    "capture_fps": args.fps,
                    "elapsed_seconds": elapsed,
                    "game_fps": rates,
                }
                ppm.unlink(missing_ok=True)
            (args.output / f"{args.workload}-capture.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            print(json.dumps(manifest), flush=True)
        finally:
            if qmp is not None:
                qmp.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if serial.exists():
                (args.output / f"{args.workload}-serial.log").write_bytes(
                    serial.read_bytes()
                )


if __name__ == "__main__":
    main()
