# Third-party source changes

## burn-optim 0.22.1

Source: [Burn 0.22.1](https://crates.io/crates/burn-optim/0.22.1).
The original `.crate` archive has SHA-256 `3ae93fbc40f7cd0aab5deb1c94906a1bd40729ea6df9f3bcce586b1047a13b9c`.
The upstream MIT and Apache 2.0 licenses are included in `burn-optim`.

This release advertises `no_std` support, but its module optimizer imports a scheduler hidden behind `std`.
The local patch makes the scheduler available with `alloc` and keeps filesystem methods behind `std`.
It adds explicit allocation imports, uses `core` constants and formatting traits, and uses portable floating-point methods without `std`.
The logging dependency remains available without its `std` feature.
The optimizer algorithms and checkpoint format remain unchanged.
The patch also adds explicit device selection when loading optimizer records.
D3OS uses this path to restore Adam state directly to the native GPU.

The workspace uses this source through `[patch.crates-io]`.
The remaining Burn crates come from the pinned release in `Cargo.lock`.
The patch must remain until an upstream release supplies equivalent portable optimizer support.
Host and D3OS target checks cover the dependency graph.
Runtime validation checks analytic gradients, tensor aliasing, and Adam checkpoint resume.

The optional `d3os-native` feature also dispatches FP32 Adam updates to one native cuTile kernel per parameter.
The public Burn optimizer API, moment tensors, timestep, and checkpoint format remain unchanged.
Bias correction uses the shared host calculation, and the GPU retains explicit FP32 rounding and epsilon placement.
AMSGrad retains its historical maximum. Weight decay and gradient clipping keep their existing preprocessing paths.
Other devices retain the tensor-operation implementation. D3OS enables the feature when native AI and training are both enabled.
The fused operation preserves untracked autodiff contexts, including moments restored from checkpoints.
Tracked optimizer inputs retain the differentiable tensor-operation implementation.

## burn-dispatch and burn-backend-extension 0.22.1

Sources: [burn-dispatch 0.22.1](https://crates.io/crates/burn-dispatch/0.22.1) and [burn-backend-extension 0.22.1](https://crates.io/crates/burn-backend-extension/0.22.1).
The original archive SHA-256 values are:

- `burn-dispatch`: `64d255d7609901f7aaf2138b394a6fe8dd8c7b526d1ed2736ba2c643944028fa`.
- `burn-backend-extension`: `44e9f488bc7c4754f007d1885f72314dc5b91848b05a892b29cd522d083e1d71`.

Burn uses a closed catalog of execution backends. These patches register the D3OS native backend through the optional `d3os-native` feature.
The extension catalog recognizes `Native`. The dispatcher adds its device, tensor, graph, enumeration, transfer, and autodiff paths.
Existing backend identifiers retain their values. The native backend uses identifier 12.
The default device order retains the existing backends and uses Native when no other execution backend is enabled.
D3OS applications select the native device explicitly through `d3os_ai::native_device()`.

The runtime patch remains compatible with `no_std` and `alloc`. The extension compiler runs on the build host.
Both packages include the upstream MIT and Apache 2.0 licenses.
The workspace uses these sources through `[patch.crates-io]`.
These patches do not replace Burn's optimizer or autodiff algorithms.
The dispatcher also forwards linear forward and backward operations to the active backend.
The upstream dispatcher decomposes linear layers before backend-specific implementations can run.
Forwarding enables native linear execution while retaining each backend's autodiff behavior.
The extension frontend also recognizes optional tensor inputs through the shared routing parser.
Present inputs participate in backend selection, primitive conversion, and autodiff-context merging.
Native Adam uses this path for optional restored moments and AMSGrad history.

## gemm-common 0.19.0

Source: [gemm-common 0.19.0](https://crates.io/crates/gemm-common/0.19.0).
The original archive SHA-256 is `88027625910cc9b1085aaaa1c4bc46bb3a36aad323452b33c25b5e4e7c8e2a3e`.
The upstream MIT license is included in `gemm-common`.

Without `std`, the upstream feature detector checks Cargo features instead of CPU target features.
This selects scalar GEMM on D3OS even when the application target guarantees AVX2 and FMA.
The patch changes that check to `cfg!(target_feature = ...)`.
The standard-library runtime detector remains unchanged.
D3OS applications can then select the SIMD implementation guaranteed by their compilation target.
The workspace uses this source through `[patch.crates-io]`.
