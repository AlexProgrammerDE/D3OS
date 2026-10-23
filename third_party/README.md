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

The workspace uses this source through `[patch.crates-io]`.
Other Burn crates come from the pinned release in `Cargo.lock`.
The patch must remain until an upstream release supplies equivalent portable optimizer support.
Host and D3OS target checks cover the dependency graph.
Runtime validation checks analytic gradients, tensor aliasing, and Adam checkpoint resume.
