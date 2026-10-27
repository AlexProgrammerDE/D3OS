# Portable Burn state checks

This harness runs the registered tests from the two vendored crates.
It uses the actual BurnCore, Dispatch, Autodiff, and Flex implementations.
Each case runs in a fresh process because provider registration is permanent.

## Run the host feature matrix

From the repository root, run these commands:

```sh
cargo +1.98.1 test --locked --manifest-path tools/burn-state-conformance/Cargo.toml
cargo +1.98.1 test --locked --manifest-path tools/burn-state-conformance/Cargo.toml --features environment-std
cargo +1.98.1 test --locked --manifest-path tools/burn-state-conformance/Cargo.toml --features std
```

| Harness features | BurnCore | CubeCL environment |
| --- | --- | --- |
| Default | Without `std` | Without `std` |
| `environment-std` | Without `std` | With `std` |
| `std` | With `std` | With `std` |

The middle case checks independent feature unification.
The provider API remains available when the environment enables `std` independently.
BurnCore's standard branch continues to use `std::thread::ThreadId`.

The tests cover registration races, thread identity, retained updates from exited threads, shared clones, and records.
The host provider retains real Rust thread identifiers and assigns checked, unique tokens.
It does not use native thread handles that the host can reuse.

## Initialize a D3OS application

Enable the `burn` feature on the application's `runtime` dependency.
The runtime installs the D3OS provider after heap and thread setup, before application code.
`NativeDevice::open()` also calls the same initializer.
Repeated initialization returns the first result and never replaces another provider.

The D3OS provider reads the kernel thread ID through the existing syscall.
The kernel reserves each positive ID once and rejects user-thread creation when the ID space is exhausted.

After building the kernel and applications, run the state check:

```sh
python3 tools/inference/test-qemu.py --burn-state-self-test --accel tcg --cpus 4 --repeat 10
```

For an isolated build, supply `--kernel-image` and `--application-binaries`.
The binary directory must contain `infer`, `shell`, and `terminal_emulator`.

The guest check uses real D3OS threads and an explicit CPU device.
It checks pending updates after worker exit, repeated updates, and record restoration.
It also checks that record restoration preserves the destination training flag.

`RunningState::into_record()` does not flush pending updates.
For a record from one updating thread, call `value_sync()` on that thread before export.
A checkpoint across multiple updating threads requires a separate synchronization policy.
