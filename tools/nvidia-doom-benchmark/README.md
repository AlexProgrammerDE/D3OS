# Shared DOOM GPU reference

This Linux tool uses the Native DOOM application's Rust draw collector, texture atlas, and rasterizer.
It loads the same host-qualified inference cubin through the CUDA Driver API.
D3OS continues to use its native Rust driver.

Both paths validate kernel geometry, buffer extents, and aliases through the NVIDIA SDK metadata.
The Linux implementation owns its CUDA context and buffers, and completes transfers and dispatches before returning.
Its context handling follows NVIDIA's [context API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
Cubin loading uses the [module API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html).

## Build

Use the repository's pinned nightly and the same CPU target as the Native application:

```sh
CARGO_TARGET_DIR=target/doom-linux \
RUSTFLAGS='-Ctarget-cpu=x86-64-v3 -Cno-redzone=yes' \
cargo +nightly-2026-02-01 build --release --locked \
  --manifest-path tools/nvidia-doom-benchmark/Cargo.toml \
  --target x86_64-unknown-linux-gnu
```

The build uses the pinned DOOM C sources and wrapper list from the Native Makefile.
The standalone lock retains the Native mutex dependency.
GCC, ar, and the vendored doomgeneric sources must be available.
The tool dynamically loads `libcuda.so.1`; it does not compile GPU kernels.

Use the same command with `test` or `clippy --all-targets` for host checks.
Host tests validate address overflow, buffer extents, and rejected aliases without opening a GPU.

## Pixel verification

Select the exact qualified inference cubin used by the Native comparison.
Choose four fixed host cores; this example uses 8, 10, 12, and 14:

```sh
taskset -c 8,10,12,14 python3 -B tools/nvidia/linux-reference.py \
  --directory target/nvidia/doom-shared-linux-verify \
  --doom-benchmark \
  --doom-gpu-binary target/doom-linux/x86_64-unknown-linux-gnu/release/nvidia-doom-benchmark \
  --doom-gpu-cubin /path/to/qualified/inference.cubin \
  --doom-gpu-verify --pin-vcpus
```

Verification compares GPU pixels with the pinned C renderer throughout `demo1`.
It also forces texture-atlas rollover and cache purges.
A successful run reports `LINUX_DOOM_GPU_BENCH_OK` after the engine's timedemo completion.
The engine reports that completion through its error handler and returns 255.
Other errors, missing frame coverage, CUDA failures, and pixel mismatches fail validation.

## Timing

Omit `--doom-gpu-verify` in a separate capture to measure GPU rendering.
Verification timings include C reference drawing and must not be used for speed comparisons.
Record the engine, WAD, binary, cubin, CPU affinity, and guest settings with each result.
Check for matching completed evidence before launching another test.
The GPU mode rejects a directory that already contains `serial.log`.

Rendering and palette/readback use the same Rust implementation on both platforms.
Linux presentation copies the 640-by-400 image into an 800-by-600 memory framebuffer.
It has no display server, physical scanout, window title, or input thread.
D3OS retains its window presentation and input handling.
Use the separate stage counters and describe this difference when reporting total FPS.
This tool does not establish physical-display parity.
