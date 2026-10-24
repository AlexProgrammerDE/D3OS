# Linux application support

D3OS includes a Linux x86-64 execution personality in `os/library/linux-compat`.
The kernel selects it when an ELF executable has an interpreter or loadable segments below the native application boundary.
Native D3OS applications continue to use their existing runtime and syscall table.

## Launch a Linux application

Copy the Linux executable into `loader/initrd/bin/`.
Copy its interpreter and required shared libraries into `loader/initrd/lib64/`.
The executable's `PT_INTERP` path must match the interpreter's path in the image.
Statically linked Linux executables do not require an interpreter.

For the tested NVIDIA compiler, the directory contains these files:

```text
loader/initrd/bin/ptxas
loader/initrd/lib64/ld-linux-x86-64.so.2
loader/initrd/lib64/libc.so.6
loader/initrd/lib64/libm.so.6
loader/initrd/lib64/libpthread.so.0
```

Build and boot D3OS with the normal workflow:

```bash
cargo make --no-workspace --profile production
```

From the D3OS shell, run the executable by name or explicit path:

```text
ptxas --version
ptxas -arch=sm_80 /usr/vector-add.ptx -o /output.cubin
ls /
```

The normal application API also accepts the executable's name or path.
The caller can wait for the returned thread with the existing join API.
Linux output files remain in D3OS tmpfs after the process exits.
Compiler diagnostics use the normal D3OS terminal output buffer.
Tmpfs files do not survive a reboot.

## Implementation

The `linux-compat` crate owns ELF validation, the initial process stack, Linux syscall decoding, descriptors, and memory bookkeeping.
Its `Host` interface supplies kernel memory operations, filesystem access, clocks, TLS, and terminal output.
The kernel implementation lives in `os/kernel/src/linux`.
Each process has its own optional Linux execution state.

Linux executables can require addresses starting at 4 MiB.
D3OS reserves physical memory from 4 MiB to 64 MiB and links the kernel at 64 MiB.
This permits low Linux user mappings without replacing live kernel code or allocator data.
The kernel heap is 128 MiB to accommodate compiler loading and process metadata.
The ramdisk moves into the heap only when it overlaps the Linux executable window.

The loader reads executables through the naming service, including files created after boot.
Linux `exit` and `exit_group` release descriptors and use normal D3OS process cleanup.
The kernel records the exit status. The existing shell still assumes success after joining external applications.

## Verification

Run the portable crate's unit tests from the D3OS checkout:

```bash
cargo test -p linux-compat
cargo clippy -p linux-compat --all-targets --no-deps -- -D warnings
```

Compiler downloads, PTX fixtures, host traces, and VM comparison scripts remain in the separate `d3os-linux` tools project.
From that project, run:

```bash
python3 tools/bootstrap.py
python3 tools/test_ptxas.py --source /path/to/D3OS
```

The tools copy the current D3OS source, including local changes, into a disposable build directory.
They do not patch the implementation or replace the normal syscall path.
The `linux_test` kernel feature adds the supervisor and serial comparison protocol.
The supervisor waits for process exit before it stops QEMU.
A native userspace launcher exercises the existing application syscall.
A second native program reads the retained cubin and performs native file operations.

The suite compares compiler exit status, diagnostics, and cubin SHA-256 hashes against Linux.
Only numeric compilation-time messages are normalized.
It also checks repeated compilation in one VM and native application launch with four CPUs.

## Current limits

The verified workload is CUDA 12.9.86 `ptxas` with its copied Linux runtime libraries.
Other executables can run when their Linux interfaces fit the implemented subset.

- The execution personality supports one thread per process. Compiler worker threads and process creation are not implemented.
- Signal masks and actions can be stored, but asynchronous signal delivery is not implemented.
- Private file mappings use eager allocation. Shared mappings and file writeback are not implemented.
- Linux standard input, sockets, and device interfaces are not implemented.
- The loader supports the reserved low executable window and high user mappings. General PIE executable loading is not implemented.
- Clock precision follows D3OS timers and scheduler accounting.
- CUDA GPU loading and execution still require a GPU driver.
