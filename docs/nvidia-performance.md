# Native NVIDIA performance

The native driver runs AlexNet, Qwen3.5-0.8B, and DOOM correctly on the RTX 4090 Laptop GPU.
The Linux performance target remains open. Correct output does not imply comparable throughput.

The goal now includes [Burn-based AI APIs and native training](burn-migration.md).
AlexNet and Qwen now use Burn with a native D3OS/cuTile backend.
DOOM keeps its native renderer. Its correctness and Linux performance requirements remain unchanged.
Full [Burn backend API parity with Linux](burn-migration.md#primary-goal-linux-backend-api-parity) now takes priority over further workload-specific optimization.
The existing model migrations do not establish complete backend support, particularly for public quantization and additional dtypes.

The optimization goal also includes Qwen3.5-9B at Q8_0 and Qwen3.6-27B at Q3_K_M.
Both targets must fit entirely in the GPU's 16 GiB of VRAM, including context and working storage.
The work includes pinned Linux measurements, native Rust/cuTile quantized kernels, numerical references, and matched performance comparisons.
The native loader also needs separate output weights and grouped key/value heads for recurrent attention.
Large model assets also need a storage path that avoids the current FAT32 boot file limit and duplicate archive copies.
Both large models now pass native validation through Burn, as described below.
The existing 0.8B, AlexNet, and DOOM checks remain required.

## Performance goal and acceptance

The goal requires Linux performance parity for every workload in this table.
Correctness, compiler reuse, and a fast result for one model do not complete the goal.

| Workload | Required comparison |
| --- | --- |
| PTX and Tile IR compilation | Cold compilation inside D3OS versus inside Linux, with identical inputs, compiler versions, flags, and target GPU |
| DOOM | Uncapped FPS and frame time for the same engine, WAD, demo, resolution, and comparable rendering and presentation work |
| MNIST training | Training steps and images per second, including forward execution, backward execution, and optimizer updates |
| MNIST inference | Latency and images per second for the same trained convolutional model and batch sizes |
| AlexNet inference | Latency and images per second for the same model, inputs, precision, and batch sizes |
| Qwen 0.8B, 9B, and 27B | Separate model load, prompt processing, and token generation measurements for all three variants |

Each comparison identifies the execution OS, CPU or GPU backend, model, precision, workload size, and timing boundaries.
The GPU comparisons use the same physical RTX 4090 Laptop GPU, with one VM at a time.
Compiler comparisons also match guest memory, virtual CPUs, and host CPU affinity.
Training comparisons match the dataset, model, initialization, optimizer, batch size, and training work.
Model comparisons retain the numerical references, and DOOM retains its pixel references.

Repeated measurements must establish comparable performance within the observed measurement variation.
A remaining gap stays explicit for each workload, even after another workload reaches parity.
Compilation remains a separate load phase and never enters average inference, training, or rendering throughput.
Persistent cache hits must remain separate from cold compilation.
The current Linux headless software DOOM reference has different presentation costs and cannot establish GPU rendering parity.
MNIST now has GPU timing cohorts and a strict FP32 numerical oracle, with a matrix precision difference in stock Linux CUDA.
See [MNIST GPU performance](mnist-performance.md) for the work contract, numerical checks, and separate Linux and D3OS results.

## Measurement method

The protocol-54 mixed-matmul correctness capture passes its D3OS VM and strict host harness.
It checks 215,427 required values across shared, guarded, Native, and ordinary Autodiff routes.
Read-only verification and independent terminal review pass.
The full Linux CUDA selection fails, so this result establishes no Linux parity.
A separate BF16 broadcast-gradient graph passes 32,198 required comparisons with bit-exact Native agreement.
These correctness captures add no inference, training, rendering, compiler-latency, or transfer-bandwidth measurements.
The [mixed-matmul coverage reference](burn-backend-coverage.md#typed-mixed-precision-matrix-products-and-linear) records its stored arithmetic boundaries and limitations.

These measurements use the same physical GPU through VFIO, with four KVM virtual CPUs.
Only one guest owns the GPU at a time.
D3OS uses its Rust driver, signed 570.144 firmware, and native Rust kernels.
The Linux reference guest uses driver 615.78.08 and CUDA libraries installed on the host.
The Linux GPU driver and execution libraries remain inside the reference guest.
D3OS uses the NVIDIA compiler tools through its Linux execution personality, with its native Rust driver handling GPU execution.

The resident-model benchmarks separate file access, model upload, session allocation, prompt processing, and generation.
Reference comparisons run after the timed work.
AlexNet reports 100 forward passes after one warmup pass. Earlier measurements use 20 passes.
Qwen reports three independent sessions with a 22-token prompt, 64 decode steps, and capacity for 256 tokens.
The optional `long` benchmark generates 4,096 tokens per session with capacity for 4,352 tokens.
It writes complete token traces to the serial log without filling the interactive terminal output queue.
The decode loop continues after EOS to measure a fixed amount of work.
Its checksum describes benchmark work, not a conversational answer.

D3OS timers have millisecond resolution. The kernel benchmark amortizes this resolution across 100 launches.
Its times include CPU submission and synchronization.
The Linux kernel benchmark uses CUDA events across 30 queued launches, which excludes most CPU submission overhead.
D3OS also runs a separate instrumented pass with GPU timestamps before scheduling and at QMD completion.
The boot check calibrates a cuTile kernel to run for at least 10 ms and compares GPU timestamps with host elapsed time.
`device_mean_us` includes device scheduling and completion work, while `instrumented_mean_us` also includes CPU submission and waiting.
The first `mean_us` pass runs without timestamp instrumentation.
These measurements separate end-to-end overhead from device throughput. They are not interchangeable timers.

## Control guest CPU placement

Both GPU launchers accept `--pin-vcpus` for comparisons with fixed CPU placement.
Start the launcher with an affinity of exactly four host CPUs.
It pauses QEMU before boot, assigns one host CPU to each guest CPU, verifies the assignments, and then resumes the VM.
The `vcpu-affinity.json` file records the execution OS, guest CPU indices, host CPU indices, and QEMU thread IDs.

For the current four-CPU compiler comparison, use:

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/linux-reference.py \
  --compiler-benchmark --compiler-repeats 3 --compiler-cpus 0-3 \
  --memory-mib 4096 --pin-vcpus --timeout 480 \
  --directory target/nvidia/compile-parity/linux-pinned

taskset -c 8,10,12,14 python3 tools/nvidia/run-qemu.py \
  --firmware target/nvidia/firmware --kernels target/nvidia/kernels \
  --memory-mib 4096 --pin-vcpus --boot-check \
  --inference gpu-alexnet-bench --timeout 90
```

The Linux `--compiler-cpus` option selects compiler CPU masks inside the guest.
Its default compares CPU 0 with CPUs 0 through 3. Select `0-3` for the primary four-CPU comparison.
For another portable input bundle, pass its directory to Linux `--compiler-inputs` and D3OS `--kernels`.
Each family still requires exactly one Tile IR or PTX input. Both compiler manifests record the staged input hashes.
Each D3OS sample requires a new VM because its compiled modules remain cached for the VM's lifetime.
Keep compilation, GPU validation, and AlexNet execution as separate phases.

Pinning controls vCPU placement; it does not reserve host cores or fix their clock rates.
The Linux control trials still show timing variation after pinning.
The GPU-free CI check verifies real QEMU thread affinities, rejects a mismatched CPU count before changing any affinity, and rejects a shared vCPU thread.

## Cold compiler baseline

These trials use the validated paired-byte Q8 bundle, CUDA 13.4.92, four KVM CPUs, and 4 GiB of guest RAM.
Both OSes use fixed guest-to-host CPU assignments on `8,10,12,14`.
All 13 staged compiler assets, including runtime libraries and portable inputs, match by SHA-256.
All generated GPU binaries match the host audits byte for byte.
No host builds overlap these trials. Other host workloads and CPU clock changes remain outside the launcher's control.

| Execution OS | Compiler CPUs | Cold compilation of all three modules | Mean |
| --- | --- | --- | ---: |
| Linux VM | CPUs 0 through 3 | 79.321, 84.868 s | 82.095 s |
| D3OS VM | Four online CPUs | 79.073, 83.351, 92.723 s | 85.049 s |

The D3OS samples use three fresh boots, with all eight AlexNet references passing afterward.
The Linux samples use separate compiler processes in one VM, with inputs copied to guest tmpfs before timing.
The 3.60 percent mean difference and overlapping ranges do not settle the remaining timing spread.
Small-module startup also remains slower: vector compilation takes 258–337 ms on D3OS versus 38–43 ms on Linux.
Tensor test compilation takes 891–1,159 ms on D3OS versus 314–534 ms on Linux.
PTX compilation latency still needs its own matched comparison.
Compilation parity therefore remains open.

### Separate hardware-counter diagnostics

The compiler profiler counts guest user and kernel CPU events through the host's `perf_event_open` API.
It samples cumulative counters every 250 ms and retains enabled and running times for multiplexing corrections.
It also records each pinned host CPU's current frequency and archives its source.
These diagnostic runs have `instrumented=true` and remain separate from ordinary compilation timing.

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/profile-compiler.py \
  --os linux --directory target/nvidia/compile-profile/linux

taskset -c 8,10,12,14 python3 tools/nvidia/profile-compiler.py \
  --os d3os --directory target/nvidia/compile-profile/d3os
```

The retained diagnostics cover inference-module compilation after the tensor test module finishes.
The following counts sum all four guest CPUs, excluding host execution.
Kernel-cycle share includes activity on otherwise idle guest CPUs. It is not a wall-time percentage.

| Inference compiler diagnostic | Linux VM | D3OS VM |
| --- | ---: | ---: |
| Guest user instructions | 522.788 billion | 523.447 billion |
| Guest user cycles | 422.247 billion | 432.349 billion |
| Guest kernel cycles | 1.874 billion | 39.773 billion |
| Guest kernel-cycle share | 0.442 percent | 8.424 percent |
| User instructions per cycle | 1.238 | 1.211 |
| User cycles per reference cycle | 2.099 | 1.998 |

The nearly identical instruction totals establish comparable compiler work in these runs.
D3OS records much more kernel activity and a lower effective clock ratio.
The counters do not establish why its clock ratio differs.
A separate native kernel sampling run captures 23,354 guest-kernel samples with no lost samples.
Disassembly locates 47.51 percent in the Linux dispatcher's global process-manager read-lock loop and 10.51 percent in its per-process state-lock loop.
Kernel heap deallocation is the next major hotspot.
The profiled process cleanup thread destroys exited processes while holding the process-manager write lock.
That destruction can free thousands of mapping descriptors while other CPUs attempt syscalls.

Evidence is under `target/nvidia/compile-parity-q8/`.
The ordinary runs are `linux-pinned/`, `pinned-before-linux/`, and `pinned-after-linux/`.
The counter traces, source snapshots, and summaries are under `linux-profile/` and `d3os-profile/`.
The sampled kernel image, raw perf data, and program-counter report are under `kernel-samples/`.

### Reduce compiler kernel contention

The syscall dispatcher now obtains its process from the current thread without locking the global process registry.
The cleanup thread detaches retired processes under the registry lock, then destroys them after releasing that lock.
Process destruction releases mapping descriptors and other heap allocations.

Removing registry contention alone exposes more allocator contention.
Its separate diagnostic records 52.583 billion guest kernel cycles, above the original 39.773 billion.
Kernel samples then locate 20.60 percent in first-fit allocation and 14.81 percent in free-list deallocation.
Per-process state-lock waits account for another 40.70 percent.

The kernel heap now uses Talc 5.1.1 with its manual memory source.
It retains the fixed 128 MiB region and existing IRQ-save spinlock.
The allocation counter charges successful allocations only, and both allocator interfaces share that accounting path.

| D3OS inference compiler diagnostic | User instructions | Kernel cycles | Kernel-cycle share |
| --- | ---: | ---: | ---: |
| Original registry and first-fit heap | 523.447 billion | 39.773 billion | 8.424 percent |
| Registry changes with first-fit heap | 524.109 billion | 52.583 billion | 11.174 percent |
| Registry changes with Talc heap | 521.468 billion | 6.740 billion | 1.672 percent |

The final diagnostic reduces aggregate guest kernel cycles by 83.05 percent relative to the original.
This percentage describes CPU cycles across all four guest CPUs. It is not a wall-time speedup.
The retained Linux diagnostic records 1.874 billion kernel cycles, so D3OS still has excess kernel work.
All three generated GPU binaries match the host audits, and all eight AlexNet layer references pass.

Three additional fresh D3OS boots measure ordinary cold compilation without counter instrumentation:

| Execution OS and kernel | Cold compilation of all three modules | Mean |
| --- | --- | ---: |
| Linux VM, retained baseline | 79.321, 84.868 s | 82.095 s |
| D3OS VM, original kernel | 79.073, 83.351, 92.723 s | 85.049 s |
| D3OS VM, registry changes and Talc | 75.941, 75.924, 80.228 s | 77.364 s |

The new D3OS mean is 9.04 percent below its earlier mean, with the same input bundle and CPU placement.
GPU validation takes another 626–636 ms and remains outside compilation and inference throughput.
All three boots pass all eight AlexNet references and produce the same audited binaries.
Host clock changes and unrelated host work remain possible contributors to timing variation.

Full-bundle timings overlap the retained Linux range, but compilation parity remains open.
Vector compilation still takes 383–553 ms on D3OS versus 38–43 ms on Linux.
PTX compilation also needs a matched latency comparison.
The context-switch GDB test verifies masked interrupts inside both Talc allocation and deallocation.
Four multicore process-exit suites pass 30 repetitions each, covering busy and sleeping workers under KVM and TCG.
Ten KVM keyboard and foreground-cleanup repetitions and all ten Linux compatibility unit tests also pass.

Evidence is under `target/nvidia/compile-parity-q8/`.
The intermediate diagnostics are `fixed-profile/` and `fixed-kernel-samples/`.
The final counter trace is `talc-profile/`, and ordinary timing records are under `pinned-talc/`.
Kernel regression logs use the `talc-test-` prefix and `talc-tests/` directory.

The Talc allocation experiment also passes DOOM's 600 pixel-reference frames, controls, cache purges, atlas rollover, and GPU inference after game exit.
Qwen 0.8B passes all 29 positions across 24 layers, batched execution, cache comparison, and tokenizer checks.
Qwen 9B and 27B each pass all 32,033,280 independent FP32 logit references with zero tolerance failures.
Warm D3OS decode checks record 48.669 and 48.540 tokens/s for 9B, and 14.339 and 14.230 tokens/s for 27B.
These checks retain the existing precision, inputs, and tolerances. They do not establish Linux performance parity.

The native MNIST training regression repeats the complete seven-epoch recipe and reaches 99.26 percent accuracy at step 6,566.
It preserves Adam state across the VM restart after epoch five.
Exported-model inference in a fresh D3OS VM reproduces all 10,000 classifications and the retained prediction CRC32, `ce40d6c3`.
The final epoch's mean loss remains `0.006849`.
This is a D3OS training and inference correctness result, with no matched Linux timing sample.
Its progress timer includes checkpoint export and evaluation, so it does not measure training execution alone.

Runtime logs and MNIST artifacts are under `target/nvidia/compile-parity-q8/talc-workloads/`.
The trained model and checkpoint also have a persistent copy under `~/.cache/d3os/mnist-trained/`, indexed by model SHA-256.
The model SHA-256 is `d87a681e2cda7ed1c3301f334053265c76dc7beeb8e9c08e2d18120a6299061e`.

An additional boot exposes a missing compiler runtime dependency: glibc loads `libgcc_s.so.1` dynamically for `pthread_exit`.
The compiler's `ldd` output omits that library, so the original staging logic does too.
A small thread-exit fixture reproduces the abort without the library and completes cleanup and join checks when it is present.
The packager now stages the unwinder and its dependencies explicitly, bringing the compiler manifest to 14 assets.
Production Linux stderr also reaches the serial log, so a failed compiler reports its diagnostic before its exit status.

The GPU-free thread regression uses the same runtime dependency collector as compiler staging.
Each invocation creates and joins 64 workers, checking returned pointers and exactly one cleanup handler per worker.
Three invocations pass under both KVM and TCG with four guest CPUs and 2 GiB of RAM.
The 512 MiB TCG trial exhausts physical memory during concurrent glibc arena reservations, which D3OS currently backs eagerly.
This memory inefficiency remains separate work for the compiler performance goal.
The regression now runs in CI through `tools/kernel/test-linux-threads.py`.

The DOOM comparison runs three identical timedemos per VM, with compilation excluded from FPS.
It uses the same native GPU presentation, WAD, engine, resolution, 7,117 game ticks, and fixed CPU placement.

| D3OS kernel and runtime | GPU timedemo FPS | Mean FPS |
| --- | --- | ---: |
| Original kernel, first control VM | 1,500.57, 1,423.40, 1,607.06 | 1,510.35 |
| Registry changes and Talc, original runtime | 1,518.87, 1,317.96, 1,509.67 | 1,448.83 |
| Original kernel, second control VM | 1,586.59, 1,607.06, 1,537.62 | 1,577.09 |
| Registry changes and Talc, corrected runtime | 1,547.17, 1,596.76, 1,509.67 | 1,551.20 |

The six original-kernel samples average 1,543.72 FPS.
The final corrected-runtime sample set falls within the control spread; these runs do not isolate a kernel effect on FPS.
GPU operator checks also pass after the three consecutive timedemos in each VM.
The failed boot before the unwinder fix produces no FPS sample and remains recorded under `doom-talc-repeat/`.
Successful comparison logs and source snapshots are under `doom-heap-comparison/` and `doom-talc-unwinder/`.
These are D3OS kernel comparisons. Linux GPU FPS parity still requires equivalent presentation work.

### Preserve vector state across CPU migration

Further compiler retries expose a separate vector-register corruption bug.
The scheduler can migrate a queued FPU owner while its latest registers still exist only on the source CPU.
The old migration path discards that ownership record without saving the registers.
An incoming CPU with no previous owner also skips restoring the arriving thread's saved state.

Each context-switch path now saves the FPU owner before releasing the outgoing scheduler lock.
The first vector use on the incoming CPU restores the arriving thread's state, including when no previous owner exists.
Queued and blocked threads therefore carry their latest vector state in memory before migration.

The Linux thread fixture also holds unique bit patterns in XMM0 and XMM1 across 128 scheduler yields per worker.
The pre-fix Talc kernel fails this register check with fixture error code 4.
The fixed kernel passes three invocations under KVM and three under TCG, covering 384 worker lifetimes and 49,152 yields.
Cleanup handlers and joined return pointers also pass in those runs.
The same kernel repeats all 120 busy and sleeping process-exit checks, ten keyboard checks, and the interrupt-injection context-switch test successfully.

The failing kernel image and register-test evidence are under `target/nvidia/compile-parity-q8/simd-before-kvm/`.
Passing vector tests are under `simd-after-kvm/` and `simd-after-tcg/`.
The compiler and DOOM timing cohorts above precede this vector-state fix and remain retained as historical comparisons.

Three fresh boots after the vector-state fix and unwinder staging complete cold compilation in 73.327, 88.614, and 98.921 seconds.
The D3OS mean is 86.954 seconds. Each boot produces all three audited binaries and passes all eight AlexNet layer references.
These trials use the same four pinned CPUs, 4 GiB guest memory, compiler executables, and Tile IR inputs as the earlier comparison.
Their runtime now includes the additional unwind library.
They establish successful compilation and GPU validation, while the wide timing spread keeps compilation parity open.
The records are under `target/nvidia/compile-parity-q8/pinned-fpu-final/`.

The same fixed kernel passes the complete GPU regression matrix.
DOOM passes all 600 pixel-reference frames and records 1,518.872 FPS in the D3OS GPU timedemo.
Qwen 0.8B passes the full layer, batched execution, cache, and tokenizer references.
Qwen 9B and 27B each pass 32,033,280 independent FP32 logit references with zero tolerance failures.
Warm D3OS decode records 48.930 and 48.892 tokens/s for 9B, and 14.421 and 14.284 tokens/s for 27B.
MNIST checkpoint resume and fresh-VM inference preserve all 10,000 predictions, 99.26 percent accuracy, and prediction CRC32 `ce40d6c3`.
These are D3OS regression measurements, with no new matched Linux sample.
The logs are under `target/nvidia/compile-parity-q8/fpu-workloads/`.

### Back anonymous reservations only when access is enabled

Anonymous `PROT_NONE` mappings now reserve addresses without allocating physical pages.
`mprotect` allocates zeroed pages when access first becomes nonzero.
Removing access from an existing allocation preserves its backing and contents.
File mappings still receive backing before the file contents load, including mappings that initially deny access.
Fixed-address replacement and unmapping handle both reserved and backed pages.
This change does not implement demand paging for accessible mappings.

The thread fixture now reserves 1 GiB, enables eight separated pages, and verifies initial zeroes and contents after permission changes.
It also checks inaccessible file mappings, syscall access rejection, mixed fixed-address replacement, and unmapping with holes.
The prior kernel panics out of physical memory in the 512 MiB KVM trial.
The patched kernel passes three fixture invocations under KVM and three under TCG, each with four CPUs and 512 MiB.
Those runs retain all vector-register, cleanup-handler, and thread-join checks.
The final fixture also unmaps a backed page while access remains disabled.
Evidence is under `target/nvidia/compile-parity-q8/reservation-before/`, `reservation-final-kvm/`, and `reservation-final-tcg/`.

A fresh Linux VM repeats cold compilation with the corrected 14-asset runtime and the same four pinned CPUs and 4 GiB memory.
Its three totals are 75.376, 75.328, and 79.686 seconds, with a mean of 76.797 seconds.
All nine generated binaries match the host audits. Its vector module takes 39–48 ms, so the D3OS startup gap remains explicit.
These Linux results are under `target/nvidia/compile-parity-q8/linux-unwinder-final/`.

Three fresh D3OS boots after deferred backing record 79.260, 74.867, and 75.464 seconds, with a mean of 76.530 seconds.
All staged assets match the fresh Linux runtime, including the input and compiler hashes.
All three GPU binaries match their audits in every boot, and all eight AlexNet layer references pass each time.
The earlier D3OS mean was 86.954 seconds. Host clock variation still limits attribution of the wall-time change.
The current D3OS and Linux full-bundle totals are comparable within this cohort's variation.
However, D3OS vector compilation takes 283–370 ms versus 39–48 ms in Linux, and matched direct PTX timing remains open.
Compilation parity therefore remains incomplete.
The D3OS evidence, saved kernel, and source changes are under `target/nvidia/compile-parity-q8/pinned-reservation-final/`.

## Compile portable kernels inside D3OS

The host now exports Tile IR instead of injecting GPU machine code.
The D3OS worker runs CUDA 13.4.92 `tileiras` and its `ptxas` subprocess inside the guest.
PTX inputs use `ptxas` directly. Each kernel family accepts exactly one input format.
The driver selects `sm_89` from the current Ada GPU's register identity.
Generated modules remain immutable in a cache for the lifetime of the VM.

The audited boot image contains three Tile IR modules and the compiler runtime, with no cubin files.
The generated vector, tensor, and 174-entry inference modules match the retained host binaries byte for byte.
The same final inputs, compiler executables, and runtime libraries also produce identical outputs inside Linux.
The final comparison uses four KVM CPUs, 4 GiB of guest RAM, and host CPUs `8,10,12,14`.
Linux copies compiler files and inputs into guest tmpfs before starting its timer.

| Execution OS | Compiler CPU availability | Cold compilation of all three modules |
| --- | --- | --- |
| Linux VM | Restricted to CPU 0 | 72.094 s |
| Linux VM | CPUs 0 through 3 | 72.141 s |
| D3OS VM, early GPU request | Four online CPUs | 153.175 s |
| D3OS VM, DOOM boot | Four online CPUs | 83.027 s |

These are individual final-input trials. D3OS ranges from 1.15 to 2.12 times the four-CPU Linux sample.
The DOOM image uses identical compiler assets and inputs and produces identical binaries.
The large variation needs scheduler and CPU-time profiling before attributing a fixed slowdown to compilation itself.
D3OS separately spends 0.634 and 0.635 seconds validating the compiled code on the GPU.
Compilation and validation are load phases, excluded from warm model throughput.
The earlier Linux bundle also completed three trials per CPU mask, averaging 72.637 and 72.044 seconds respectively.
Its architecture hints differ from the final input, while its generated binaries are identical.

An early GPU request blocks without holding the GPU lock, then resumes after compilation and validation.
That path passes the native GPU operator suite and all eight AlexNet layer references.
The following AlexNet sample reports 1.090 ms per forward pass over 100 repetitions.
The preceding D3OS samples were 0.970 and 1.060 ms. This migration establishes no throughput improvement.
The kernel context-switch interrupt test and three repeated native CPU inference runs also pass.

Qwen 0.8B passes all 29 reference positions, 24 layers, batched execution, final cache comparisons, and 105 tokenizer cases.
Its two uninstrumented warm D3OS samples measure 239.700 and 258.065 tokens per second, with 9 ms prompt processing.
DOOM passes 600 pixel-reference frames, controls, cache purges, atlas rollover, and GPU inference after exit.
Its subsequent D3OS GPU timedemo measures 1,617.50 FPS for 7,117 game ticks and 154 real ticks.
The preceding D3OS GPU result was 1,607.06 FPS. The one-tick difference does not establish an optimization.
The retained Linux headless software samples remain 1,845.15 and 2,306.44 FPS, with different presentation costs.

The early request exposed a kernel-thread exit bug: startup retained a CPU-local scheduler across a migratable entry function.
Thread return now selects the current CPU's scheduler with interrupts masked until the switch.
Wait queues also register parking before releasing their queue lock, and remote wakes preserve that parking transition.
These changes prevent discarded wakes and exiting through a scheduler from another CPU.
The final waiter fix also cancels parking when the target has already entered a local ready queue.
The final image repeats the context-switch, early-request, GPU operator, and AlexNet checks successfully.
That early-request compilation takes 155.565 seconds, with a separate 0.632-second validation phase.

Reproduce the compiler comparison with:

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/large-qwen/guest --compiler-benchmark \
  --compiler-repeats 1 --memory-mib 4096 --timeout 600
```

The default repeat count is three per CPU mask.
Evidence is under `target/nvidia/guest-compile/`, including `matched-compiler-results.json`, input manifests, and serial logs.

The cache does not survive reboot, and startup still compiles the complete bundle.
Persistent caching and smaller, demand-loaded modules remain required startup improvements.
NVIDIA's [compute cache](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html) avoids repeated PTX compilation.
PyTorch also [caches generated kernels across processes](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).

Portable hints preserve the CTA contract for all twelve targets supported by the pinned assembler.
The vector input also assembles for `sm_80` and `sm_120` on the host, without claiming execution on those GPUs.
The native GPU driver still targets the RTX 4090 Laptop GPU. Portable input does not implement drivers for other architectures.

The 9B regression uses `tools/nvidia/fixtures/vector-add.ptx` in place of the vector Tile IR input.
The other two families retain their normal Tile IR inputs. The boot passes vector and Tensor Core execution checks.
All 32,033,280 independent FP32 reference logits pass with zero tolerance failures.
Warm D3OS generation measures 31.715 and 31.667 tokens per second; the preceding samples were 32.611 and 32.454.
One greedy selection differs from the retained Linux CUDA trace, while the benchmark uses the same teacher-forced inputs.
The retained Linux CUDA 9B baseline is 58.3715 tokens per second. The throughput gap remains open.

The 27B regression also passes all 32,033,280 independent FP32 reference logits with zero tolerance failures.
Warm D3OS generation measures 5.339 and 5.118 tokens per second, compared with the preceding 5.461 and 5.224 samples.
Two greedy selections differ from the retained Linux CUDA trace; the same teacher-forced inputs remain enforced.
The retained Linux CUDA 27B baseline is 34.4025 tokens per second. Comparable throughput is not established.

## Earlier compiler scheduling comparison

Timer preemption now keeps the current runnable thread when its ready queue has no peer.
Previously, that path queued the runnable thread and selected idle until the next scheduling opportunity.
The kernel also uses explicit cooperative yields in input and network polling loops.
Those yields retain their existing behavior.

This earlier compiler comparison uses CUDA 13.4.92 and the packed-block kernel inputs from the following section.
Both guests use four KVM CPUs, 4 GiB of RAM, affinity `8,10,12,14`, and target `sm_89`.
The Linux trial first copies compiler inputs and executables to guest tmpfs.
Linux repeats fresh compiler invocations in one VM. Each D3OS sample uses a fresh boot.
All three generated GPU binaries match between Linux and D3OS in every completed trial.

| Execution OS | Compiler CPUs | Three cold compilation samples | Mean |
| --- | --- | --- | ---: |
| Linux | CPU 0 | 70.668, 71.081, 70.817 s | 70.855 s |
| Linux | CPUs 0 through 3 | 70.403, 71.037, 70.765 s | 70.735 s |
| D3OS, timer policy only | Four online CPUs | 83.754, 73.914, 74.635 s | 77.434 s |
| D3OS, scheduler and process fixes | Four online CPUs | 74.024, 78.954, 74.892 s | 75.957 s |

The preceding matched 4 GiB D3OS AlexNet boot compiled this bundle in 149.398 seconds.
Earlier 15 GiB D3OS boots took 82.223 and 82.527 seconds, so the previous timing spread remains relevant.
The final D3OS mean is 7.38 percent slower than the four-CPU Linux mean.
Compilation parity remains open. The final D3OS samples still span 4.930 seconds.

Linux now records user CPU time, system CPU time, memory, page faults, and context switches for each compiler invocation.
Its first inference-module trial spends 70.03 seconds in user code and 0.12 seconds in the kernel.
D3OS records executable image loading, compiler execution, and output validation separately.
The execution interval includes dynamic linking and waiting for compiler subprocesses.
The launcher also measures host wall time between compilation start and GPU validation completion.
These observations agree with the guest phase totals inside the launcher's 100 ms polling interval.
GPU validation remains outside compilation and inference throughput.

All three final D3OS boots pass the eight AlexNet references, with forward means of 1.020, 1.000, and 1.010 ms.
The context-switch interrupt test and its three repeated CPU inference checks also pass under TCG.
Evidence and per-module phase records are under `target/nvidia/compile-parity/`.
The `linux-results.json`, `d3os-results.json`, and `cleanup-d3os-results.json` files retain the Linux, timer-only, and final samples respectively.

## Multicore process cleanup

The DOOM regression exposed an input thread that survived game exit on another CPU.
It consumed shell input as raw keyboard events and panicked with `invalid key state 10!`.
The old process lookup searched only the calling CPU's ready queue.
It missed running, sleeping, blocked, and remote threads.

The scheduler now records process ownership for every live thread across all CPUs.
Process exit uses that registry to request cleanup of every other thread.
Remote cancellation moves a running target off its CPU before removing it.
The removal logic also compares each queue's own length and decrements runnable counts only for runnable targets.

Longer regressions exposed additional scheduler and lifetime failures:

- The heap lock allowed timer preemption while held. Scheduler heap-lock polling could then prevent its owner from running again.
- A suspended Rust mutex guard could later unlock its original CPU's scheduler after the thread resumed on another CPU.
- Scheduler lookup could select a CPU before a preemption and use that CPU's scheduler after migration.
- The syscall dispatcher retained a process reference across native handlers that never returned, preventing address-space cleanup.

The heap now uses an IRQ-save lock. Scheduler lookup also masks local interrupts until the scheduler operation finishes.
Blocking operations restore the caller's interrupt state when they resume, including after migration.
The assembly switch callback releases the lock transferred by a context switch.
It releases the outgoing thread reference after switching stacks.
Native dispatch releases its process reference before entering a native handler.
Process exit also releases its lookup guard before acquiring the process-manager write lock.

The CPU regression launches four busy workers in a child process, then exits that process without joining them.
Shared counters must stop, and the global thread count must return to its previous value.
Thirty repetitions pass with four KVM CPUs and with two TCG CPUs, each in a 512 MiB guest.
Before the native reference fix, that guest exhausted physical memory after 17 successful repetitions.
The context-switch interrupt test, heap interrupt-mask checks, and three CPU inference repetitions also pass under TCG.
CI runs the process-exit regression without NVIDIA hardware.

The native NVIDIA DOOM regression passes 600 pixel-reference frames, control events, cache purges, and atlas resets.
GPU operator checks also pass after quitting DOOM, through the same terminal that previously failed.
The following native GPU timedemo completes 7,117 game ticks at 1,596.763 FPS.
This D3OS result does not establish parity with the retained Linux headless software reference.

Reproduce the CPU lifecycle check after building the production image:

```sh
python3 tools/kernel/test-process-exit.py --accel kvm --cpus 4 --repeat 30
python3 tools/kernel/test-process-exit.py --accel tcg --cpus 2 --repeat 30
```

The GPU evidence is under `target/nvidia/compile-parity/policy-doom-*`.
The CPU evidence is under `target/process-exit-kvm-4/` and `target/process-exit-tcg-2/`.

The four-step MNIST training smoke test exports its checkpoint and verifies inference after reloading the model.
Its 188,272-byte model matches the previous GPU smoke-test artifact byte for byte.
This partial training run establishes a regression check, with no matched Linux training or inference performance result.
Evidence is under `target/nvidia/compile-parity/mnist-smoke/`.

The final Qwen regressions retain the numerical references and benchmark checksums.
The 0.8B model passes 29 reference positions across 24 layers.
Each large model passes 32,033,280 reference logits with the original tolerances.
The large models retain their prior one-token and two-token greedy-selection differences from Linux.

| Model | D3OS native Burn warm decode | Retained Linux CUDA warm decode |
| --- | ---: | ---: |
| Qwen3.5-0.8B | 259.109, 259.109 tokens/s | 251 to 256 tokens/s |
| Qwen3.5-9B Q8_0 | 32.242, 32.193 tokens/s | 58.3715 tokens/s |
| Qwen3.6-27B Q3_K_M | 14.449, 14.321 tokens/s | 34.4025 tokens/s |

These decode results do not establish prompt-processing parity.
The 0.8B D3OS warm prompt samples take 9 and 10 ms for 22 tokens.
The retained Linux prompt samples take about 6.6 ms.
The large-model execution gaps remain open.
The final Qwen evidence is in `cleanup-ai-serial.log` and `policy-27b-serial.log` under `target/nvidia/compile-parity/`.

## Terminal input, idle polling, and process aborts

An idle terminal previously polled continuously and allocated a buffer on every output poll.
Each worker now reports activity. The terminal sleeps for one millisecond when no worker reports activity.
Output polling uses a fixed 128-byte stack buffer.

These CPU measurements cover the whole QEMU process, including its virtual CPUs and device emulation.
Both trials use the same kernel, four KVM CPUs, 512 MiB of RAM, and host affinity `8,10,12,14`.
Each trial contains three five-second idle windows. CPU percentages use one host core as 100 percent.

| D3OS terminal | Host CPU samples | Mean host CPU |
| --- | --- | ---: |
| Continuous polling | 101.797%, 101.796%, 101.797% | 101.796% |
| Sleep when idle | 2.400%, 2.600%, 2.000% | 2.333% |

Shell input and native CPU inference pass after both idle trials.
These trials include the TTY queue change and precede the sleep-list cleanup fix below.
This resource reduction does not establish compilation or GPU execution parity.
The launcher also records host VM CPU time between compilation markers, including GPU validation.
That counter includes every QEMU thread. It differs from Linux's per-compiler user and system CPU times.
Evidence and artifact hashes are under `target/nvidia/restored-idle/`.

Repeated input tests exposed a separate queue error.
The TTY accepted events only while an application waited inside a read call.
Events disappeared while the application handled an earlier event.
The TTY now keeps the selected encoding active and queues input between reads.
A change of encoding clears pending bytes from the previous mode.

The regression sends six key presses and releases while the reader pauses after each event.
It alternates sequential input and a burst of all six keys.
The old queue received only the first key's two events in the burst case.
The fixed queue receives all twelve events.
Each cycle then exits normally or panics, followed by CPU inference through the same shell.

Ten cycles pass with four KVM CPUs and idle sleeping after the sleep-list fix.
Every cycle also completes a CPU DOOM timedemo and another shell inference command after exit.
Thirty cycles without DOOM pass with four KVM CPUs and thirty with four TCG CPUs.
CI uses the thirty-cycle TCG configuration.

The pinned DOOM engine reports a completed timedemo through its fatal-error handler.
That handler tried to probe a Linux desktop dialog through the unimplemented `system()` function.
DOOM now selects console error reporting with `-nogui`, so the handler reaches native process exit.

The native panic handler previously used thread exit, which left sibling threads alive.
A surviving raw-input reader consumed the shell's next command and panicked while decoding its bytes.
Native panic handling now exits the whole process, consistent with the target's `panic=abort` configuration.
The public process-exit API now has a non-returning type.

DOOM also waits for its input worker to start and select raw mode before the engine begins rendering.
Starting the worker alone did not establish that ordering.
Without this wait, the first GPU trial lost the first four controls despite matching all 600 reference frames.
With the wait, three consecutive GPU cycles pass every control and all 600 reference frames.
Each cycle also exercises low detail, invisibility, texture atlas rollover, and every scheduled cache purge.
GPU inference passes after each manual exit and each timedemo exit.

Before the sleep-list fix below, D3OS GPU timedemos report 1,607.065, 1,596.763, and 1,607.065 FPS.
Each timedemo covers 7,117 game ticks, with 155, 156, and 155 real ticks respectively.
The one-tick spread does not establish a rendering speedup.
The Linux headless software reference still has different presentation costs, so DOOM FPS parity remains open.
The three-cycle evidence is under `target/nvidia/restored-doom-input-ready/`.
The preceding failed control check remains under `target/nvidia/restored-doom/`.

```sh
python3 tools/nvidia/prepare-doom.py
python3 tools/kernel/test-terminal-input.py --accel kvm --cpus 4 --repeat 10 --doom-timedemo --directory target/sleep-lock-fixed-doom
python3 tools/kernel/test-terminal-input.py --accel tcg --cpus 4 --repeat 30 --timeout 240 --directory target/sleep-lock-raw-tcg
```

The harness accepts `--directory` to preserve separate comparisons.
Input evidence is under `target/sleep-lock-fixed-doom/`, `target/sleep-lock-after/`, and `target/sleep-lock-raw-tcg/`.
The proctest Clippy check passes with warnings denied.
The terminal Clippy check still reports pre-existing lints in its input helpers, module layout, initialization, and process joins.

## Idle-core scheduling fairness

A 27B validation boot failed a Burn timer check before the model benchmark started.
Both wake counters remained zero at the existing 200 ms deadline.
The same problem reproduced in a CPU-only VM, without NVIDIA hardware or model assets.

The scheduler balanced queued work before checking whether its current thread was idle.
An idle CPU could donate a newly arrived worker before executing it.
That ordering can repeatedly move a new thread between CPUs instead of executing it.
Idle CPUs now select queued work first. CPUs that already execute an application retain load balancing.

A controlled comparison used identical userspace with two kernel binaries.
The old kernel failed its first timer check, with both counters at zero.
The fixed kernel passed 30 repetitions with four KVM CPUs and 30 with four TCG CPUs.
The rebuilt kernel also passes ten repetitions on each CPU backend after the TTY queue change.
The 200 ms deadline and wake assertions remain unchanged.

```sh
python3 tools/inference/test-qemu.py --burn-self-test --accel kvm --cpus 4 --repeat 10
python3 tools/inference/test-qemu.py --burn-self-test --accel tcg --cpus 4 --repeat 10
```

CI runs ten TCG repetitions.
Both QEMU launchers accept `--kernel-image PATH` for controlled kernel comparisons with unchanged userspace.
The NVIDIA launcher requires `--firmware` with that option.
The latest timer evidence is under `target/restored-burn-kvm/` and `target/restored-burn-tcg/`.

## Sleep-list lock during process cleanup

A longer mixed input/DOOM run stalled after its ninth input cycle.
GDB showed one CPU inside `kill_locally`, called through `drain_inbox_into_ready` and `block_and_switch`.
The other three CPUs were idle.
The active CPU repeatedly tried to acquire the sleep-list lock with interrupts disabled.

`block_and_switch` held that lock while dispatching inbox commands.
A kill command also inspects the sleep list, so the CPU tried to acquire its own lock again.
The scheduler now releases the sleep-list lock before inbox dispatch.
The ready-state lock still protects the scheduling decision.

The process-exit regression now supports four sleeping workers alongside its existing CPU-bound worker case.
Each case requires the workers to stop and the live thread count to return to its initial value.
With identical userspace, the old kernel stalls after four successful sleeping-worker cycles.
The fixed kernel passes 30 cycles with four KVM CPUs and 30 with four TCG CPUs.
Both kernels also pass 30 raw-input cycles without DOOM.
CI covers both busy and sleeping worker exit, plus 30 input/panic cycles with four CPUs.

```sh
python3 tools/kernel/test-process-exit.py --accel kvm --cpus 4 --repeat 30 --sleeping-workers
python3 tools/kernel/test-process-exit.py --accel tcg --cpus 4 --repeat 30 --sleeping-workers --timeout 240
```

Both lifecycle harnesses accept `--kernel-image` and `--directory` for separate kernel comparisons.
GDB diagnostics use the kernel ELF staged in the failing VM's boot directory.
This keeps the symbols correct when a later build replaces `loader/kernel.elf`.
Evidence is under `target/sleeping-exit-before/`, `target/sleeping-exit-after/`, and `target/sleeping-exit-tcg/`.
The original mixed-workload failure remains under `target/terminal-input-final-kvm/`.

The fixed kernel also passes the original ten-cycle input/DOOM sequence and thirty input/panic cycles under four-CPU TCG.
The final GPU validation repeats all three DOOM cycles, including every control, 600 reference frames, cache exercises, and GPU inference after both exits.
Its D3OS GPU timedemos report 1,482.708, 1,556.844, and 1,596.763 FPS.
This batch overlaps CPU regression tests. These diagnostic samples do not isolate the performance effect of the scheduler change.
The final boot records 108.768 seconds of compilation and 0.713 seconds of GPU validation, both outside execution timing.
Compilation parity and matched Linux DOOM FPS parity remain open.
Evidence is under `target/nvidia/restored-doom-final/`.

The harness now consumes complete serial records before evaluating their fields.
A partial frame record previously caused a false atlas-counter failure; its completed record contained 6,450 atlas resets.
The preceding failed harness run remains under `target/nvidia/restored-doom-sleep-lock/`.

The restored Qwen 27B GPU run passes all 32,033,280 logit comparisons, with zero tolerance failures.
It retains the two recorded greedy-choice differences and the existing decode checksum.
Its warm D3OS samples are 14.183 and 13.588 tokens/s, with background CPU regression tests active.
Fresh Linux CUDA samples are 34.229 and 34.237 tokens/s.
These diagnostic timings do not establish an isolated performance comparison or Linux parity.
The restored MNIST four-step check also preserves its model bytes and prediction checksum `663f1993`.
It covers checkpoint export and model reload, without a matched Linux training benchmark.

## Packed-block quantized projections

The large-model benchmark now accepts `device-timing` and records separate prefill and decode profiles.
It preserves three ordinary samples and adds a fourth sample with serialized GPU timestamps.
Instrumented throughput remains separate from ordinary warm throughput.

The initial 27B profile attributes 96.34 percent of measured GPU execution time to quantized projections.
Those projections consume 188.392 ms per token, out of 195.559 ms of instrumented GPU execution.
The new Q3_K, Q4_K, Q5_K, and Q6_K vector kernels decode complete 256-weight blocks.
They reuse packed payload and scale bytes across eight groups, with four independent accumulators.
Q8_0 and the batched Tensor Core projection paths retain their existing implementations.
Activations remain FP32, and the dequantization multiplication order remains unchanged.

The following before/after runs use D3OS, Burn, the native NVIDIA backend, and the same physical GPU.
They use four guest CPUs, affinity `8,10,12,14`, 15360 MiB of RAM, and a 2048-token FP32 cache.
Both runs process the same 37 prompt tokens and 128 teacher-forced decode steps.
The host performs no concurrent builds or numerical verification during these GPU runs.

| 27B measurement | Before | Packed-block kernels |
| --- | ---: | ---: |
| Ordinary warm generation, sample 1 | 5.338 tokens/s | 14.481 tokens/s |
| Ordinary warm generation, sample 2 | 5.121 tokens/s | 14.343 tokens/s |
| Quantized projection device time | 188.392 ms/token | 59.410 ms/token |
| Total instrumented GPU execution | 195.559 ms/token | 66.134 ms/token |

Mean warm throughput improves by 2.76 times in these samples.
The retained Linux llama.cpp CUDA baseline is 34.4025 tokens/s, so the large execution gap remains open.
All 32,033,280 independent FP32 logit comparisons pass with the original tolerances.
The maximum absolute error remains `0.00003552437`, with the same two greedy differences from the Linux CUDA trace.
The complete native selection checksum remains unchanged.
The GPU operator suite also passes all five quantized formats, including longer reductions and partial output tiles.

The four audited vector kernels have no local register spills.
Q3_K uses 64 registers instead of 80, Q4_K uses 64 instead of 66, and Q5_K uses 70 instead of 85.
Q6_K uses 84 registers instead of 88.
This D3OS boot compiles all three portable modules in 82.223 seconds, followed by 0.632 seconds of GPU validation.
The earlier compiler variation prevents attributing this load-time difference to the kernel change.

The regression VM also passes all 0.8B model, cache, batch, and tokenizer references.
Its warm D3OS samples measure 248.062 and 222.222 tokens/s, showing continued timing variation.
The 9B regression passes all 32,033,280 FP32 logits, with warm samples of 31.644 and 31.558 tokens/s.
Its maximum absolute error remains `0.000030636787`, with one greedy difference from Linux CUDA.
AlexNet passes all eight layer references and measures 0.980 ms per forward over 100 repetitions.
DOOM passes all 600 pixel references, controls, cache purges, atlas rollover, and post-exit GPU inference.
The following D3OS GPU timedemo reports 1,607.06 FPS, with 7,117 game ticks and 155 real ticks.
These control workloads establish no performance improvement from the quantized kernel change.
The regression and AlexNet VMs compile the same bundle in 82.527 and 149.398 seconds respectively.
This spread confirms that compilation variance remains unresolved.

Evidence is under `target/nvidia/large-profile/`.
The `27b-serial.log` and `block-27b-serial.log` files retain both complete sessions.
The `block-comparison.json` file separates ordinary samples from device measurements.

## Paired-byte Q8 projections

The Qwen 9B profile identifies single-token Q8_0 projections as the main decode cost.
The previous vector kernel broadcasts FP32 inputs across a four-row tile and reduces 128 values per row.
Its replacement assigns 32 logical lanes to each row and loads two adjacent quantized weights through one aligned 16-bit read.
Eight FP32 accumulators process eight Q8 blocks per iteration, followed by a 32-value row reduction.
Activations remain FP32. Weight scaling and the numerical tolerances remain unchanged.
Prompt processing retains its existing Tensor Core path.

The host audit finds no local register spills in the new vector kernel.
It uses 64 registers instead of 43 and reduces shared storage from 4,096 to 512 bytes.
Static shared-memory load/store instructions fall from 5/17 to 3/3.
These are instruction counts in the disassembly, not dynamic operation counts.
Raw executable-section comparison finds only `vector_project_q8_0` changed among the 174 inference entries.

The repeated comparison uses D3OS, Burn, the native NVIDIA backend, and the same RTX 4090 Laptop GPU.
Both bundles compile from Tile IR inside D3OS before operator validation and timing.
Each VM uses four KVM CPUs, affinity `8,10,12,14`, and 15360 MiB of RAM.
Work remains 37 prompt tokens, 128 teacher-forced decode tokens, and a 2048-token FP32 cache.
Each fresh boot runs the GPU operator suite, three ordinary model sessions, and a separate device-timestamp session.
The table includes only the two ordinary warm sessions from each boot.

| Qwen3.5-9B Q8_0, D3OS GPU | Previous vector kernel | Paired-byte vector kernel |
| --- | ---: | ---: |
| Fresh validation boots | 2 | 3 |
| Ordinary warm samples | 4 | 6 |
| Warm generation range | 31.809–32.072 tokens/s | 48.448–48.911 tokens/s |
| Mean warm generation | 31.984 tokens/s | 48.663 tokens/s |
| Warm prompt processing range | 413–421 ms | 410–423 ms |
| Quantized projection device time | 27.664–27.888 ms/token | 18.112–18.164 ms/token |

Mean generation improves by 52.15 percent across these samples.
A fresh, uninstrumented Linux llama.cpp CUDA VM measures 58.204 and 58.101 tokens/s, averaging 58.153.
Its warm prompt processing takes 43.293 and 43.248 ms.
These Linux results use the same model, prompt, cache settings, guest resources, and host affinity.
The projection arithmetic difference described below still needs a matched precision comparison.
Prompt processing remains unchanged on D3OS, with a large gap against Linux.
An independent host IR export overlaps some boots on CPUs `0,2,4,6`; these results do not establish an isolated compilation improvement.
Each ordinary sample excludes compilation, operator checks, reference validation, and device-timestamp instrumentation.

All five boots pass the quantized operator tests, partial blocks, output tails, and buffer guards.
All 32,033,280 independent FP32 logit comparisons pass with the original tolerances in each validation.
The maximum absolute error changes from `0.000024318695` to `0.000025749207`.
The selection checksum stays `11168047765639420708`, with the same one greedy difference from Linux CUDA.
Host linker tests, Clippy, and the Tensor Core instruction audits also pass.
Evidence is under `target/nvidia/q8-optimization/`, with individual serial logs and `comparison.json`.
The fresh ordinary Linux reference is under its `linux-current/` subdirectory.

The final source rebuild produces the same three audited GPU binaries as the measured candidate, byte for byte.
A separate D3OS boot then recompiles those final IR inputs and reproduces all three binary hashes.
It passes the complete GPU operator suite and all 32,033,280 FP32 logit comparisons.
Its warm generation samples are 48.799 and 48.614 tokens/s, with 422 and 412 ms prompt processing.
This final boot runs without the concurrent host IR export.
Its logs are under `target/nvidia/q8-optimization/final-validation/`.
The default kernel bundle now uses these validated portable inputs.

### Read adjacent packed weights for prompt processing

Quantized prompt projections now decode adjacent values within each packed weight row.
They transpose the decoded tile before Tensor Core multiplication.
This preserves the reduction order, FP32 activations, TF32x3 products, and resident model storage.
The change adds no intermediate allocation and leaves single-token vector kernels unchanged.

The host instruction audit finds exactly five changed executable sections, one batched projection for each quantized format.
Every other inference entry remains byte-identical, including the decode, BF16, AlexNet, MNIST, and DOOM kernels.
The changed entries retain Tensor Core instructions without register spills.
Each D3OS VM compiles its portable inputs internally and reproduces the audited machine-code hashes.

Fresh baseline and revised D3OS VMs use four KVM CPUs, host affinity `8,10,12,14`, and 15360 MiB of RAM.
Each model uses its existing weights, 37 prompt tokens, 128 teacher-forced decode steps, and a 2048-token FP32 cache.
The host runs one GPU VM at a time, without concurrent builds during timing.
The following means use two ordinary warm samples per model and bundle.
Compilation, loading, operator checks, numerical validation, and device instrumentation remain outside those intervals.

| D3OS GPU prompt processing | Previous packed reads | Adjacent packed reads | Speedup |
| --- | ---: | ---: | ---: |
| Qwen3.5-9B Q8_0 | 428.000 ms | 209.000 ms | 2.048 times |
| Qwen3.6-27B Q3_K_M | 1,733.500 ms | 1,045.500 ms | 1.658 times |

Separate instrumented profiles support the ordinary timing improvement.
Quantized projection device time falls from 408.194 to 199.004 ms for 9B, and from 1,854.827 to 1,077.995 ms for 27B.
These serialized device profiles remain excluded from the ordinary means.

Both bundles pass the complete GPU operator suite, including all five quantized formats, partial tiles, and buffer guards.
Each model also passes all 32,033,280 independent FP32 logit comparisons with zero tolerance failures.
Maximum absolute errors remain `0.000025749207` for 9B and `0.000034809113` for 27B.
Both model selection checksums remain unchanged.
The existing greedy differences from Linux also remain unchanged: one for 9B and two for 27B.

Decode performance does not improve in these samples.
The 9B means are 48.8925 tokens/s before and 48.9200 after.
The 27B means are 14.0895 tokens/s before and 13.5565 after, despite identical single-token kernel code.
The recorded timing difference remains explicit, without attributing it to an unmeasured cause.

Fresh Linux llama.cpp CUDA 9B samples average 43.143 ms for prompt processing and 58.303 tokens/s for decode.
Fresh Linux 27B samples average 93.213 ms and 34.3745 tokens/s respectively.
Both models retain the prompt and decode token IDs from the native reference inputs.
The revised D3OS prompt intervals remain 4.844 times longer than Linux for 9B and 11.216 times longer for 27B.
Linux quantizes activations during projection, while D3OS retains FP32 activations and the original numerical tolerances.
The Linux throughput target and its arithmetic difference both remain explicit.

The baseline native boots compile their full portable bundles in 83.992 and 84.311 seconds.
The revised native boots take 95.664 and 104.326 seconds, measured separately from execution.
These load-time observations do not establish compilation parity or isolate the cause of the difference.
A matched Linux cold-compilation comparison of the revised bundle remains required.

Baseline 9B evidence is under `target/nvidia/q8-optimization/prefill-before-20261025/`.
The other native cohorts and fresh Linux baselines are under `target/nvidia/qwen-prefill-optimization/`.
Portable inputs, host instruction audits, and the executable-section comparison are under `target/nvidia/qwen-prefill-coalesced-kernels/`.
The previous portable bundle remains under `target/nvidia/qwen-prefill-original-kernels/`.

### Share packed weights across prompt rows

Batched quantized projections now use 32-row tiles for batches above 16 tokens.
Smaller batches retain their 16-row kernels, and single-token vector kernels remain unchanged.
This reuses each decoded weight tile across more prompt tokens without enlarging the reduction tile.
The SDK and driver use protocol 20 so the launch grid matches the selected tile.
Applications and portable inputs require matching rebuilds.

Fresh D3OS VMs retain the preceding weights, prompt, decode trace, FP32 cache, guest resources, and host affinity.
Each VM passes the complete GPU operator suite, including new 33-row cases and guarded output tails.
The following means use two ordinary warm intervals per model.
Compilation, model loading, numerical checks, and device instrumentation remain outside execution timing.

| Prompt processing | Previous D3OS GPU | Revised D3OS GPU | Retained Linux CUDA GPU |
| --- | ---: | ---: | ---: |
| Qwen3.5-9B Q8_0 | 209.000 ms | 174.000 ms | 43.143 ms |
| Qwen3.6-27B Q3_K_M | 1,045.500 ms | 774.500 ms | 93.213 ms |

The D3OS prompt intervals improve by 1.201 times for 9B and 1.350 times for 27B.
Separate device profiles reduce quantized projection time from 199.004 to 167.134 ms, and from 1,077.995 to 816.567 ms respectively.
These instrumented samples remain separate from ordinary throughput.
The revised prompt intervals remain 4.033 and 8.309 times longer than the retained Linux GPU intervals.
Linux uses quantized activations during projection, while D3OS retains FP32 activations and the existing tolerances.

Each model passes all 32,033,280 independent FP32 logit comparisons with zero tolerance failures.
Maximum errors, selection checksums, and the existing greedy differences remain unchanged.
All 176 existing inference executable sections remain byte-identical. The bundle adds five 32-row projection entries.
Host audits enforce Tensor Core instructions and reject local register spills in both matrix variants.

Current D3OS decode means are 49.0045 tokens/s for 9B and 14.2580 tokens/s for 27B.
The retained Linux CUDA means are 58.303 and 34.3745 tokens/s respectively.
The unchanged vector code and timing variation do not establish a decode optimization.
D3OS cold compilation takes 91.875 and 88.212 seconds in these two boots, separately from execution.
A matched Linux cold-compilation comparison of this revised bundle remains required.

A rejected 64-value reduction tile passes the full 9B numerical reference but increases warm prompt processing to 325.500 ms.
A 32-row kernel for every batched projection averages 190.000 ms on 9B.
Keeping the 16-row kernel for short batches reduces that mean to 174.000 ms.
Only the adaptive selection becomes the default.

The host exporter now combines entries in one transaction, with per-entry checks and one final module verification.
This removes repeated verification of the growing module. It does not change the compiler work inside D3OS or Linux.

Evidence is under `target/nvidia/qwen-prefill-optimization/adaptive-m32-9b/`, `adaptive-m32-27b/`, and `row-tile-comparison.json`.
The same parent directory retains the rejected `k64-9b/` and `m32-9b/` experiments.
Accepted portable inputs and host audits are under `target/nvidia/qwen-adaptive-m32-kernels/` and now populate the default kernel directory.

### Remove masked-load padding from quantized decode

The single-token Q3_K, Q4_K, Q5_K, and Q6_K kernels now clamp inactive output lanes to the last valid weight row.
Their packed weight loads become unconditional, while output stores retain their original bounds masks.
The native launch contract rejects zero dimensions and incomplete quantization blocks.
Consequently, these extra reads stay inside valid storage, including projections with fewer than four outputs.
FP32 activations, accumulation order, output values, and numerical tolerances remain unchanged.

This removes repeated padding work inside the decoding loop.
Only four of the 181 inference executable sections change; all other executable sections and constant banks remain byte-identical.
The launch grid and protocol 20 remain unchanged, and all register-spill and Tensor Core audits pass.

The Linux hardware-counter diagnostic uses the same checkpoint tensors and measurement settings before and after the change:

| Instrumented Linux GPU kernel | Previous duration | Revised duration | Registers per thread, before to after |
| --- | ---: | ---: | ---: |
| Q3_K gate, 17,408 by 5,120 | 169.472 us | 156.800 us | 64 to 69 |
| Q4_K recurrent output, 5,120 by 6,144 | 62.304 us | 60.064 us | 64 to 70 |

These diagnostic GPU intervals shorten by 7.48 and 3.60 percent respectively.
The extra registers reduce occupancy, but both absolute execution intervals improve.
Instrumented Linux kernel durations remain separate from ordinary D3OS model throughput.

Fresh D3OS VMs compare the preceding bundle against this change on the same GPU.
Both use four vCPUs, 15,360 MiB of RAM, host affinity `8,10,12,14`, FP32 activations and cache, and the same teacher-forced trace.
The following ordinary warm samples exclude compilation, loading, reference checks, and device instrumentation.

| Qwen3.6-27B Q3_K_M measurement | Previous D3OS GPU | Revised D3OS GPU | Retained Linux CUDA GPU |
| --- | ---: | ---: | ---: |
| Decode samples | 14.429, 14.376 tokens/s | 15.322, 15.184 tokens/s | Mean 34.3745 tokens/s |
| Mean decode | 14.4025 tokens/s | 15.2530 tokens/s | 34.3745 tokens/s |
| Mean prompt processing | 764 ms | 757 ms | 93.213 ms |

Native decode improves by 5.91 percent in this comparison.
The small prompt difference does not establish a prompt optimization; the batched kernels are unchanged.
Native decode remains at 44.37 percent of the retained Linux rate, and the Linux activation-precision difference still applies.
The performance goal remains open.

Both native runs pass the complete GPU operator suite, including all five quantized formats and guarded output tails.
Each run also passes all 32,033,280 FP32 reference logits with zero tolerance failures.
The maximum absolute error remains `0.000034809113`, the checksum remains `14424479105642919525`, and the existing two greedy differences remain unchanged.
Separate native cold compilation takes 89.731 seconds for the control and 90.609 seconds for the revision.
These single compiler samples do not establish a compilation improvement or regression.

Ordinary model evidence is under `target/nvidia/qwen-prefill-optimization/masked-control-27b/` and `unmasked-27b/`.
Their summary is `unmasked-comparison.json` in the same parent directory.
Counter evidence is under `target/nvidia/quantized-counter-profile/unmasked-q3-q4/`, with its preceding baseline described below.
The revised portable inputs and audits are under `target/nvidia/quantized-unmasked-kernels/` and populate the default kernel directory.

### Profile native quantized kernels on Linux

The Linux reference guest can collect hardware counters for the exact cuTile kernels used by D3OS.
This diagnostic runs through Linux `libcuda`; the profiler and execution library remain inside that guest.
The guest compiles the portable Tile IR and checks the resulting binary against the native bundle's host audit.
D3OS continues to receive portable inputs and compile them inside D3OS.

The probe extracts an actual checkpoint tensor and creates a deterministic FP32 input vector.
Its independent CPU oracle uses GGUF dequantization and FP64 accumulation, then stores FP32 outputs.
Both an ordinary control launch and the profiled launch must pass every output comparison and all buffer guards.
The operator tolerance remains `0.0002 + 0.0002 * abs(reference)`.
Buffer data starts at a 256-byte boundary, matching native allocation alignment.

Prepare the checkpoint probes before starting a GPU timing run:

```bash
python3 tools/nvidia/prepare-nsight-compute.py

OPENBLAS_NUM_THREADS=4 taskset -c 8,10,12,14 \
  python3 tools/nvidia/prepare-quantized-probe.py \
  ~/.cache/d3os/large-qwen/Qwen3.6-27B-Q3_K_M.gguf \
  blk.0.ffn_gate.weight target/nvidia/quantized-probes/q3-gate

OPENBLAS_NUM_THREADS=4 taskset -c 8,10,12,14 \
  python3 tools/nvidia/prepare-quantized-probe.py \
  ~/.cache/d3os/large-qwen/Qwen3.6-27B-Q3_K_M.gguf \
  blk.0.ssm_out.weight target/nvidia/quantized-probes/q4-ssm

taskset -c 8,10,12,14 python3 tools/nvidia/linux-reference.py \
  --quantized-profile target/nvidia/quantized-probes/q3-gate \
  target/nvidia/quantized-probes/q4-ssm \
  --memory-mib 4096 --timeout 480 \
  --directory target/nvidia/quantized-counter-profile/baseline
```

Each output directory must be fresh. Pass `--compiler-inputs PATH` to profile another portable kernel bundle.
The preparer verifies the pinned Nsight Compute archive's size and SHA-256 without installing a host GPU driver.
Input manifests record the model, tensor, dimensions, raw weights, input vector, and oracle hashes.
The diagnostic retains compressed Nsight reports, raw CSV counters, compiler hashes, and validation logs.
Its parser rejects missing counters, unexpected units, multiple launches, and mismatched kernel geometry.

Nsight performs kernel replay with boost clock control and cache flushing after 256 warmup launches.
These are instrumented Linux GPU diagnostics, excluded from ordinary D3OS and Linux throughput comparisons.
NVIDIA documents these measurement effects in its [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

The initial Q3_K gate projection has 17,408 outputs and 5,120 inputs.
The Q4_K recurrent output projection has 5,120 outputs and 6,144 inputs.
Their respective integer ALU utilization is 84.69 and 76.28 percent of peak.
DRAM utilization is 39.31 and 49.48 percent, while math-pipeline throttle stalls average 3.32 and 2.61 cycles per issued instruction.
These counters identify integer decoding work as the main target for these two kernels.
They do not establish the bottleneck for every model tensor.
NVIDIA's [compute triage guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/) explains how to interpret these pipeline counters.

The accepted baseline reports are under `target/nvidia/quantized-counter-profile/baseline-q3-q4-final/`.
Earlier failed profiler and exporter attempts remain in separate directories and are excluded from accepted results.

Further Q3_K experiments compare adjacent packed weights and wider scale loads.
The original adjacent-load candidate passes all 17,408 CPU reference values and buffer guards but takes 161.280 us.
Adding 16-bit scale loads reduces the instrumented Linux GPU interval to 140.832 us.
A fresh accepted-kernel control takes 156.864 us, consistent with the preceding 156.800 us sample.
An alternative integer bit expression takes 142.816 us and is not selected.
These are Linux hardware-counter diagnostics, not ordinary D3OS model measurements.

The complete wider-load bundle passes host Tensor Core and spill audits.
Only `vector_project_q3k` changes; the other 203 executable and compiler-constant sections remain byte-identical.
Its focused and complete-module Q3_K instruction sections also match byte for byte.
The candidate is not adopted because it still needs full D3OS model validation and ordinary throughput measurements.
Work now prioritizes the complete public Burn backend and quantization API.
The accepted default bundle remains unchanged.
Candidate source, portable inputs, audits, and the experiment summary remain under `target/nvidia/q3-wide-paired-kernels/`.
Counter reports use the `q3-` directories under `target/nvidia/quantized-counter-profile/`.

### Audit the Linux projection precision

The Linux large-model reference now accepts `--trace-cuda-kernels` for a separate diagnostic run.
It records unique CUDA runtime kernel names and marks its result as `instrumented=true`.
Use its ordinary, uninstrumented runs for performance comparisons.
The trace library remains inside the Linux reference guest and never enters D3OS.

The pinned llama.cpp b11827 9B run launches `quantize_q8_1`, `quantize_mmq_q8_1`, and `mul_mat_vec_q` kernels for Q8_0 weights.
Its [quantized vector implementation](https://github.com/ggml-org/llama.cpp/blob/b11827/ggml/src/ggml-cuda/mmvq.cu) quantizes FP32 inputs to Q8_1 before projection.
Its [Q8 dot product](https://github.com/ggml-org/llama.cpp/blob/b11827/ggml/src/ggml-cuda/vecdotq.cuh) uses integer dot instructions and block scales.
FP32 cache and output settings therefore do not establish identical activation arithmetic between Linux and D3OS.
The stock Linux throughput remains a performance target, with that precision difference stated explicitly.
D3OS retains its independent FP32 references and their original tolerances.

For example, collect the diagnostic separately from the ordinary Linux baseline:

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/linux-reference.py \
  --large-qwen 9b \
  --llama-directory target/nvidia/linux-llama/llama-b11827 \
  --model /path/to/Qwen3.5-9B-Q8_0.gguf \
  --memory-mib 15360 --timeout 240 \
  --directory target/nvidia/linux-qwen-9b-trace \
  --trace-cuda-kernels
```

The validated trace records 39 unique kernel names.
Its manifest records the tracing source and library hashes.
Evidence is under `target/nvidia/q8-optimization/linux-trace-fixed/`.

## Native Burn integer coverage

The I32 backend adds exact wrapping reductions and GPU-generated index ranges.
Physical D3OS checks use independent integer references, signed views, broadcast inputs, partial tiles, and neighboring guards.
Public Burn cross-entropy also passes an analytic loss, gradient, and accuracy reference on the NVIDIA device.
The matching native compute protocol is version 17, with 59 kernel families and 174 inference entries.
All 166 earlier GPU instruction sections remain byte-identical after the eight new entries are added.
Validation logs are under `target/nvidia/burn/integers/`.
This work adds no D3OS or Linux throughput measurement. Workload migration remains the priority before optimization.

## Native Burn training validation

D3OS now trains and exports the convolutional MNIST model through Burn's public native GPU API.
The physical RTX 4090 Laptop GPU run uses all 60,000 training images and the separate 10,000-image test set.
Seven FP32 epochs reach 99.26 percent test accuracy: five at learning rate 0.001, then two at 0.0001 with Adam state preserved.
The model also passes checkpoint resume across VM restarts and inference export checks.
See [Train MNIST inside D3OS](mnist-training.howto.md) for the procedure and retained evidence.
This validation contains no Linux training sample or matched performance comparison.

## AlexNet Burn migration

These are fresh measurements on the same passed-through RTX 4090 Laptop GPU, with four guest CPUs and host affinity `8,10,12,14`.
All three paths use the same FP32 checkpoint, 224-by-224 fixture, and eight complete layer references.
The original absolute and relative tolerances remain `2e-4`. The predicted class remains 258.

| OS and execution path | File read | Model preparation | Warm forward mean |
| --- | ---: | ---: | ---: |
| D3OS, legacy native model | 444 ms | 263 ms | 0.930 ms |
| D3OS, Burn native model | 242 ms | 319 ms | 2.420 ms |
| Linux, cuDNN/cuBLAS | 856.773 ms | 722.932 ms | 0.731–0.735 ms |

Each warm sample measures 100 passes after one warmup. Linux supplies three samples; its average is 0.732888 ms.
Both timers include CPU submission and synchronized completion. Input upload and reference readback remain outside the forward interval.
The file and preparation columns are single cold observations with different loading implementations, not a separate matched I/O benchmark.
Linux uses cuDNN FMA and default cuBLAS FP32 math. D3OS uses the native compensated TF32 Tensor Core kernels with FP32 storage.

Burn currently regresses D3OS warm forward time by 2.60 times.
The model remains on Burn. Kernel dispatch and allocation improvements follow the remaining workload migrations.
The forward residency check confirms zero activation uploads and downloads.
At this AlexNet milestone, Qwen retained its earlier execution path. Its later migration is described below.

A separate instrumented GPU pass records 1.437 ms across the five generic convolution entries and 0.570 ms across the three classifier matrix entries.
Those serialized device observations identify costs; they are not the uninstrumented forward measurement.
No device kernel binary changes in this migration.
The retained logs, Linux provenance, and comparison report are under `target/nvidia/burn/alexnet/`.
The Burn CPU reference, Qwen references, native Burn training/checkpoints, and all 600 DOOM views also pass.

## Qwen packed-weight Burn extensions

Fresh correctness checks pass on the physical NVIDIA GPU inside D3OS and on the explicit host Burn CPU device.
They cover F32, BF16, and five GGUF encodings, including independent packed-block reference values.
Projection, split projection, embeddings, fused gates, and zero-centered RMS normalization use public Burn tensors.
Burn module records preserve packed words. Residual fusion preserves cloned inputs and row views.
Native forward checks report zero activation uploads and downloads.
The device kernel bundle and compute protocol remain unchanged.

At this packed-weight milestone, the full model and state migration remained pending.
These checks add no D3OS throughput result and no Linux measurement.
Logs are under `target/nvidia/burn/qwen-packed/`.

## Qwen state extensions and migration baseline

Fresh physical D3OS checks pass public Burn attention and recurrent operations against independent binary64 fixtures.
Eight attention cases and sixteen recurrent cases cover single-token and batched execution, cache snapshots, and packed projection fusion.
Native state updates perform zero activation uploads or downloads.
Split attention, grouped heads, wider fallback dimensions, and error propagation also pass.
The kernel bundle and compute protocol remain unchanged.

These baseline samples use the legacy execution API before the Burn model migration.
The following fresh D3OS baseline measures that path before the model migration, with the passed-through RTX 4090 Laptop GPU.
Each session has capacity for 256 tokens, a 22-token prompt, and 64 decode steps.
The host affinity remains `8,10,12,14`, with four guest CPUs.

| D3OS legacy session | Prompt time | Prompt tokens/s | Decode time | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| First session | 13 ms | 1692.308 | 248 ms | 258.065 |
| Warm session 1 | 9 ms | 2444.444 | 254 ms | 251.969 |
| Warm session 2 | 9 ms | 2444.444 | 254 ms | 251.969 |

The single file-read observation is 1543 ms; model preparation takes 615 ms.
State allocation takes 21 ms initially and 6 ms for each later session.
All three decode traces share checksum `11137929795425568356`.
The original 29-position, 24-layer references and batched cache comparison pass after the benchmark.
No fresh Linux sample or complete Burn Qwen model result accompanies this baseline.

Fresh AlexNet references and all 600 DOOM views also pass.
DOOM reports 69 to 70 FPS after startup under its gameplay cap; this is not an uncapped rendering benchmark.
Evidence is under `target/nvidia/burn/qwen-state/`.

## Complete Qwen Burn model migration

Qwen3.5-0.8B now runs through Burn modules, packed parameters, public tensors, owned session state, and generation.
The tokenizer and streaming GGUF reader live in the AI library. The previous inference crate is removed.
Physical D3OS checks pass every original layer and logit reference at 29 positions, with all 105 tokenizer cases unchanged.
The original absolute and relative reference tolerances remain `2e-3` and `5e-4`.
Sequential and batched execution also pass the final cache comparisons.
Streaming generation answers the default arithmetic prompt with `4` and stops at EOS.

These are fresh D3OS before-and-after measurements on the same passed-through GPU, with four guest CPUs and affinity `8,10,12,14`.
Each session uses a 22-token prompt, 64 decode steps, and capacity for 256 tokens.
Timers include CPU submission and synchronized completion.

| D3OS execution path | Model preparation | First decode | Warm decode 1 | Warm decode 2 |
| --- | ---: | ---: | ---: | ---: |
| Legacy native model | 615 ms | 258.065 tokens/s | 251.969 tokens/s | 251.969 tokens/s |
| Burn native model | 21880 ms | 221.453 tokens/s | 242.424 tokens/s | 235.294 tokens/s |

The combined warm decode rate falls from 251.969 to 238.806 tokens/s, a 5.22 percent regression.
Warm prompt processing measures 9 ms before migration and 9 to 10 ms afterward.
The single file-read observations are 1543 ms before migration and 2666 ms afterward; they do not establish a matched I/O comparison.
State creation measures 21 ms initially and 6 ms warm before migration, versus 5 ms initially and below one timer tick warm afterward.
Model preparation regresses by 35.58 times. Service counters place almost all preparation time in guest application code.
These regressions remain open. Burn remains the model API.

All six decode traces share checksum `11137929795425568356`.
The resident-model check permits only explicit token-ID uploads and scalar selection reads, with no activation tensor transfer.
For the measured warm session, prefill uploads one token-ID batch and downloads no tensors.
Decode uploads 64 token IDs and reads 128 selection scalars: one finite-value flag and one selected index per step.
The kernel bundle and compute protocol remain unchanged.
No fresh Linux sample accompanies this migration comparison.
Fresh AlexNet references and all 600 DOOM views also pass; DOOM stays outside Burn and reaches its 70 FPS gameplay cap.
Evidence and the before-and-after report are under `target/nvidia/burn/qwen-model/`.

## Quantized Qwen models on Burn

Qwen3.5-9B Q8_0 and Qwen3.6-27B Q3_K_M now run through the native Burn stack.
Each passes 165 independent FP32 layer and logit references, with matching sequential and batched final caches.
The reference exporter runs PyTorch on the Linux host, one complete layer at a time.
It uses the original GGUF weights, the pinned Hugging Face configuration, and the Linux CUDA benchmark's token IDs.
This avoids loading a complete 27B FP32 model into host RAM.

The original absolute and relative tolerances remain `2e-3` and `5e-4`.
A separate check compares all 248320 logits at 129 positions after the prompt.
Both models pass all 32033280 values.
Maximum absolute errors are `0.00003064` for 9B and `0.00003553` for 27B.
The complete token, merge, and token-type tables match the 0.8B tokenizer.

GGUF exports grouped value heads in tiled order.
The reader restores input projections and state parameters to the model's grouped order.
It preserves packed output projection weights and reorders their input activation on the GPU.
This handles quantization blocks that span more than one value head without requantization.

These fresh speed samples use the passed-through RTX 4090 Laptop GPU, four guest CPUs, and affinity `8,10,12,14`.
Both guests use 15360 MiB of RAM, a 37-token prompt, 128 decode steps, and a 2048-token FP32 cache.
D3OS replays the Linux CUDA token IDs and continues all benchmark steps through EOS.
Transfer counters permit two prompt-ID uploads, 128 decode-ID uploads, and 256 selection-scalar reads per run.
They show no activation tensor upload or download during timing.
Numerical reference reads occur afterward.

| Model and weights | D3OS native Burn warm decode | Linux guest llama.cpp CUDA warm decode |
| --- | ---: | ---: |
| Qwen3.5-9B Q8_0 | 31.9205 tokens/s | 58.3715 tokens/s |
| Qwen3.6-27B Q3_K_M | 5.2260 tokens/s | 34.4025 tokens/s |

These are diagnostic speed comparisons. They do not establish Linux parity.
Native projections retain FP32 activation arithmetic.
The normal quantized Linux CUDA path produces different raw logits and one or two different greedy choices on these inputs.
D3OS instead agrees with the independent FP32 reference within the errors above.
The Linux CUDA speed results and Linux CPU accuracy results are separate evidence.
Some runs overlap CPU verification on separate cores; quiet matched timing and kernel profiling remain required.

Compute-buffer weight storage is 9516859392 bytes for 9B and 13575868416 bytes for 27B.
These counters exclude firmware and channel allocations.
Both models also fit their measured context and working storage in the physical GPU's 16 GiB VRAM.

Packed weight preparation now uses bulk copies and direct byte uploads.
The initial 0.8B Burn preparation regression falls from 21880 ms to 1940 ms in the first corrected D3OS run.
The earlier native model took 615 ms, so preparation still needs improvement.
Flex CPU model import widens BF16 constants to FP32 once, rather than for each projection.
Embedding lookup decodes only the requested token rows.
A local GEMM portability patch enables target-guaranteed SIMD without `std`.

D3OS CPU correctness checks pass all 29 positions, 24 layers, and 105 tokenizer cases.
The full check takes 704920 ms under TCG and 32072 ms under KVM in the retained runs.
These are CPU validation times, not GPU inference measurements.
The initial CPU emulation timeout is resolved without removing references or increasing its 900-second limit.

Commands are in [the Qwen guide](qwen.howto.md#run-the-quantized-9b-and-27b-models).
Reference manifests, source checksums, before-and-after failures, complete token traces, and comparisons are under `target/nvidia/burn/large-qwen/`.
CPU and loading evidence is under `target/nvidia/burn/qwen-load/`.

The final D3OS smoke suite also passes the GPU self-test, all eight AlexNet references, and the original 0.8B Qwen references.
Its latest 0.8B preparation sample is 1439 ms, with warm decode samples of 219.178 and 226.148 tokens/s.
These samples show remaining timing variation and a generation gap; they do not establish an improvement.
D3OS DOOM measures 1617.5 uncapped timedemo FPS and retains its 70 FPS gameplay cap.
All 600 verification frames, input controls, cache purges, cleanup, and post-exit GPU inference checks pass.
The GPU kernel bundle remains unchanged.

## Burn convolution algorithm selection

The next D3OS GPU change selects explicit image windows and Tensor Core matrix products for large dense inference convolutions.
It requires one image, ungrouped unit dilation, symmetric padding, contiguous views, and enough matrix work for reuse.
The window workspace is limited to 8 MiB.
Other geometries retain the direct convolution algorithm. Backward operations retain their independently validated implementation.

Fresh 100-pass D3OS GPU averages improve from 2.320 ms to 1.300 ms for AlexNet, a 43.97 percent reduction.
Fresh Linux GPU cuDNN/cuBLAS warm averages are 0.742077 and 0.732881 ms.
Both guests use four CPUs, affinity `8,10,12,14`, 4096 MiB of RAM, and the original FP32 weights and input.
The original eight layer references retain their `2e-4` absolute and relative tolerances.
The larger speed gap remains open.

The five D3OS convolution matrix products total about 0.327 ms of instrumented device time.
The earlier direct convolution kernels total about 1.427 ms.
Explicit window extraction and channel bias kernels add separate work, so these totals are not complete forward times.
Classifier matrix products and generic activation dispatch remain visible costs.

Physical GPU checks validate explicit and direct paths against independent FP64 forward and gradient references.
Cases cover optional bias, nonzero contiguous offsets, shared storage, and signed views.
Existing grouped, dilated, asymmetric, empty, finite-difference, residency, and guard checks pass.
The selector also has metadata tests for workspace limits and unsupported fast-path geometries.
The kernel bundle remains unchanged.
Evidence is under `target/nvidia/burn/alexnet-optimization/`.

A subsequent dispatcher fix activates backend-specific linear forward and backward operations.
The native forward folds leading dimensions into matrix rows and adds bias to its private GPU result.
AlexNet then uses three row-bias kernels and 14 generic elementwise kernels, instead of 17 generic elementwise kernels.
Two D3OS GPU 100-pass samples measure 1.230 and 1.290 ms.
The variation does not establish a separate throughput improvement over the preceding 1.300 ms sample.
Public API tests check signed views, preserved inputs, folded batches, empty outputs, and zero input width.
Native MNIST autodiff and SGD checks and all eight AlexNet references pass.
The kernel bundle remains unchanged.

### Native linear storage, ReLU, and recording scopes

AlexNet now keeps its dense classifier weights in their original physical `[output, input]` order.
Burn sees a transposed `[input, output]` view. The native linear operation selects the existing projection kernels without copying those weights.
Single-row projections use the vector specialization. Batched projections retain Tensor Core execution.
Public API checks cover positive offsets, signed views, source preservation, and analytic input, weight, and bias gradients.
The D3OS CPU AlexNet run also passes every original layer reference.

ReLU now uses one native kernel and reuses uniquely owned contiguous storage.
Shared inputs receive a private result. Strided inputs first use the checked resident copy operation.
The integer selection preserves quiet and signaling NaN payloads, infinities, and positive subnormals. Both signed zeros produce positive zero.
The first floating selection compiled into a NaN-canonicalizing maximum instruction and failed the bit-pattern check.
The corrected integer implementation passes that check. The compiler audit rejects the unsafe maximum instruction.
Only the ReLU instruction section changes among the 174 inference entries. All Tensor Core and spill audits pass.

Compound Burn operations now use recording scopes that join their parent transaction.
Explicit `with_commands()` calls retain their synchronous contract, including nested calls.
Hardware checks cover joined recording scopes, handled callback errors, explicit nested completion, ordered writes, and allocation lifetimes.
AlexNet uses 31 launches in two submission batches, versus 15 batches with synchronous layer scopes.

Fresh D3OS GPU 100-pass means are 1.030 and 1.010 ms with the corrected ReLU and synchronous layer scopes.
The recording build measures 0.970 and 1.060 ms. Their overlap does not establish a separate stable speedup from scope composition alone.
The preceding linear build measures 1.230 and 1.290 ms, and column-storage samples measure 1.150 and 1.250 ms.
Fresh Linux GPU cuDNN/cuBLAS samples measure 0.733869, 0.732027, and 0.732209 ms.
All guests use four CPUs and affinity `8,10,12,14`.
AlexNet guests use 4096 MiB RAM and the same FP32 weights and input.
Every D3OS sample passes the original eight layer references, without changing their tolerances.

The final D3OS serialized device sample totals about 0.908 ms.
It includes 0.326 ms for convolution matrix products, 0.453 ms for classifier products, and 0.053 ms for window extraction.
ReLU, bias, and pooling kernels account for the remaining device time.
These device timestamps are diagnostic and remain separate from the normal forward timer.
Evidence, binary identities, retained comparison images, and the complete kernel profile are under `target/nvidia/burn/alexnet-optimization/`.

The same final build passes Qwen 0.8B layer, logit, tokenizer, and final-cache references.
Its two uninstrumented warm D3OS GPU samples measure 211.221 and 242.424 tokens/s.
An earlier diagnostic session measures 243.346 tokens/s without instrumentation. Its instrumented sample is excluded from that throughput figure.
The retained Linux GPU reference remains about 251 to 256 tokens/s.
Fresh D3OS GPU warm rates are 32.064 to 32.250 tokens/s for Qwen 9B and 5.180 to 5.420 for Qwen 27B.
The retained Linux CUDA rates are 58.3715 and 34.4025 tokens/s, with the arithmetic differences described above.
Each large model again passes 32,033,280 independent FP32 reference logits with zero tolerance failures.
These changes do not establish a Qwen throughput improvement.

The final D3OS GPU DOOM timedemo measures 1607.064575 FPS for 7117 game ticks and 155 real ticks.
The separate 600-frame pixel, control, cache-purge, atlas-rollover, and post-exit inference checks pass at the 70 FPS gameplay cap.
DOOM retains its title-bar FPS display and no corner overlay.
The retained Linux headless software results still lack matching presentation costs. They do not establish GPU rendering parity.

## Parallel Burn token selection

Qwen now selects tokens through a native Burn extension with 32 parallel GPU partitions.
The host combines exact index/value pairs and preserves the first global tie.
The operation rejects nonfinite values in every partition, including losing partitions.
Physical D3OS checks cover partition boundaries, full vocabulary width, signed views, positive offsets, source guards, and recoverable execution errors.
The explicit CPU backend preserves F64 precision. Selection also preserves the caller's autodiff graph.

Each token now needs one 256-byte metadata read instead of two four-byte scalar reads.
This reduces host transactions but increases transferred metadata from 8 to 256 bytes per token.
No activation tensor crosses to the host. The existing kernel bundle remains unchanged.
A fresh serialized D3OS GPU sample measures 1.106144 ms for 64 selection launches, or 0.017284 ms per token.
The preceding serialized reduction sample measures 10.393120 ms for 64 steps, or about 0.162393 ms per token.
These device samples diagnose selection cost. They do not establish an end-to-end model speedup.
The uninstrumented warm sample in the new diagnostic VM measures 240.602 tokens/s with the original checksum and all references unchanged.
No fresh Linux sample accompanies this change. Evidence is under `target/nvidia/burn/selection/`.

## Earlier Burn primitive measurements

The following measurements describe earlier implementation stages. Current model migration results are above.

### Burn primitive regression

The measurements in this subsection are D3OS results from QEMU with the RTX 4090 Laptop GPU passed through via VFIO.
This regression round does not include a fresh Linux measurement.
Performance reports must label each result as D3OS or Linux and identify whether a comparison uses retained or fresh samples.

The first native Burn primitives now support shared GPU storage, checked views, and eleven specialized cuTile elementwise entries.
GPU checks cover signed strides, broadcasting, scalars, comparisons, mask filling, exact bit copies, partial tiles, and neighboring guards.
Chained arithmetic uses resident buffers without host activation transfers or separate output clears.
Invalid view metadata fails before dispatch. All 77 pre-existing device instruction sections remain byte-identical.

The next primitive checks add Tensor Core matrix products and seven strided reduction entries.
Matrix checks cover broadcast batches, noncontiguous inputs, subviews, empty outputs, and zero inner dimensions.
Reduction checks cover every tensor axis, repeated tiles, broadcasting, nonfinite values, first-index ties, empty axes, and neighboring guards.
Both operations keep tensors resident. All 88 earlier instruction sections remain byte-identical after the reductions are added.
The latest primitive checks add 21 unary math operations, broadcast power, and exact tensor fill.
Independent libm references cover 2,816 FP32 edge values, including rounding boundaries, zeros, infinities, and NaNs.
Fill performs no host upload and requires no separate clear. All 95 earlier GPU instruction sections remain byte-identical.
The current SDK and driver use native compute protocol version 18.

The accompanying regression passes AlexNet's eight layer references, Qwen's layer and cache references, and all 600 DOOM views.
D3OS AlexNet averages 0.960 ms across 100 passes.
Both D3OS Qwen warm 64-token samples reach 264.463 tokens/s, with the existing checksum unchanged.
D3OS DOOM reaches 1,596.76 uncapped FPS. Regular D3OS gameplay remains capped around 70 FPS.
These samples remain within the recent run-to-run variation and do not establish a throughput improvement.
The latest primitive checks and regression logs are under `target/nvidia/burn/math/`.
These are regression measurements of the existing workloads. Their migration to Burn remains pending.
The first Burn training check runs on the explicit CPU device inside D3OS and validates gradients and Adam checkpoint resume.
Full native Burn training and matched performance comparisons remain open.

The subsequent typed-word step adds I32 arithmetic, boolean logic, masks, and conversions.
All 118 earlier GPU instruction sections remain byte-identical after its 29 entries are added.
The D3OS self-test validates wrapping, conversion boundaries, shared ownership, guards, and resident execution.
This step uses functional workload checks. It adds no new D3OS or Linux throughput measurement.
The performance samples in this subsection remain from the preceding math regression, under `target/nvidia/burn/math/`.
The typed-word validation logs are under `target/nvidia/burn/words/`.

The subsequent indexing step adds resident gather and vector select for FP32, I32, and boolean tensors.
Each nonempty operation reads one four-byte bounds flag and synchronizes; tensor values do not cross to the host.
Physical D3OS GPU checks cover signed strides, offset views, duplicate indices, empty axes, partial tiles, and neighboring guards.
Invalid indices return an error without changing inputs or preventing later GPU operations, including inside command scopes.
All 147 previous GPU instruction sections remain byte-identical.
This step adds functional validation, with no new D3OS or Linux throughput measurement.
Its logs are under `target/nvidia/burn/indexing/`.

The next training primitive adds scatter accumulation and vector select accumulation.
FP32 and I32 values use addition. Boolean values use OR.
Inputs remain immutable through a private GPU copy of the destination and GPU materialization of noncontiguous values.
A four-byte bounds read synchronizes each nonempty operation. Colliding FP32 additions have an unspecified order.
FP32 compare-and-swap preserves subnormal values that Ada's global atomic addition can flush to zero.
NVIDIA describes this behavior in its [PTX atomic instruction reference](https://docs.nvidia.com/cuda/archive/12.1.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom).
Physical D3OS GPU checks pass strided scatter, duplicate indices, offset views, guards, shared inputs, and invalid-index recovery.
They also preserve subnormal accumulation and match a weighted gather gradient against independent FP64 finite differences.
All 148 previous GPU instruction sections remain byte-identical.
This step does not establish a D3OS or Linux throughput improvement.
Its evidence is under `target/nvidia/burn/scatter/`.

The convolution step adds native forward execution and gradients for the Burn backend.
Physical D3OS GPU checks pass independent FP64 references, finite differences, grouped and dilated cases, signed views, empty batches, and guards.
The operations keep activations resident. All 151 previous GPU instruction sections remain byte-identical.
This step leaves the existing AlexNet path unchanged and adds no performance improvement claim.
Its functional validation evidence is under `target/nvidia/burn/convolution/`.
Further performance optimization waits until the native Burn backend and AI workload migration are complete.

The pooling step adds max, average, and adaptive average forward and backward operations for the native Burn backend.
The operations retain tensor values on the GPU. Max backward reads one four-byte bounds flag and synchronizes.
Physical D3OS GPU checks pass FP64 references, finite differences, signed views, guarded writes, and recoverable index errors.
The checks include exact NaN payloads, subnormal arithmetic, empty dimensions, and adaptive ratios with products beyond 32 bits.
All 154 previous GPU instruction sections remain byte-identical. AlexNet, Qwen, and DOOM functional regressions also pass.
This step targets Burn operation coverage and correctness, with no new throughput comparison or optimization claim.
Its validation evidence is under `target/nvidia/burn/pooling/`.

The tensor update step adds exact masked selection, slice assignment, concatenation, and padding for the native Burn backend.
FP32, I32, and boolean values remain resident, including through signed views and broadcasting.
Updates preserve shared source storage through private destination copies. Partial writers start from initialized allocations.
Physical D3OS GPU checks pass exact words, signed views, overlapping-source snapshots, empty tensors, guards, and recoverable rejected launches.
All 159 previous GPU instruction sections remain byte-identical. AlexNet, Qwen, and DOOM functional regressions also pass.
This step adds operation coverage, with no new D3OS or Linux performance measurement.
Its validation evidence is under `target/nvidia/burn/updates/`.

The random-generation step adds seeded Philox4x32-10 streams and portable checkpoints for Burn initialization and training.
Five cuTile entries provide raw words, FP32 uniform and normal values, Bernoulli samples, and unbiased bounded integers.
Values remain on the GPU. Stream reservations are serialized and checked for counter exhaustion.
Physical D3OS GPU checks pass anchored Philox words, exact checkpoint replay, scalar integer references, FP64 normal references, and guarded writes.
All 161 previous GPU instruction sections remain byte-identical. AlexNet, Qwen, and DOOM functional regressions also pass.
This step adds training support, with no new D3OS or Linux performance measurement.
Its validation evidence is under `target/nvidia/burn/random/`.

The public backend step registers native GPU tensors and autodiff with Burn's device dispatcher.
Physical D3OS checks pass public matrix products, convolution gradients, analytic SGD, and Adam checkpoint resume.
The convolutional MNIST model also passes a synthetic SGD step, with gradients for all eight parameter tensors and preserved shared weights.
That step uploads no activations and reads only two four-byte bounds flags from max-pool gradients.
Batched gradient products retain initialized allocations for their partial writes.
Native failed tensors propagate execution errors without placeholder storage or CPU computation.
Operator coverage, complete MNIST training, and AI workload migration remain incomplete.
This step adds no D3OS or Linux performance comparison. Its evidence is under `target/nvidia/burn/dispatch/`.

### Grouped recurrent attention

The recurrent operator now accepts separate query/key and value head counts.
Its GPU checks cover the 16/32 heads of Qwen3.5-9B and the 16/48 heads of Qwen3.6-27B.
They also cover three smaller shapes with partial tiles and convolution histories.
Each shape compares seven sequential positions and the same positions in batches against the CPU reference.
The host test independently compares grouped channels against explicitly repeated query/key channels, including both final caches.
Invalid head groups leave both caches unchanged.

Grouped execution updates convolution history before the recurrent kernel runs.
Value heads share read-only query/key channels, while each head owns its recurrent matrix.
This avoids concurrent writes to shared convolution history.
All 76 existing device instruction sections remain byte-identical.
The Qwen-0.8B regression preserves its 64-token trace and passes all layer, batch, and cache references.
Full 9B and 27B execution still requires model configuration, output-weight, and asset-loading integration.

### Native quantized operators

The native Rust/cuTile backend now accepts all five packed weight formats used by the larger Qwen files.
Independent samples cover 5,120 values from the checksum-pinned checkpoints.
CPU and GPU decoding match the reference bits exactly.
The GPU checks also cover embedding gathers, single-token projections, batched Tensor Core projections, unaligned uploads, partial tiles, and neighboring guards.
Invalid raw embedding indices produce zero rows without reading outside the table.

All 56 existing device instruction sections remain byte-identical. The compiler audits actual Tensor Core instructions in each new batched projection.
The final regression passes AlexNet's eight layer references, Qwen-0.8B's layer and cache references, and all 600 DOOM views.
The accompanying DOOM run reaches 1,566.64 uncapped FPS. Regular gameplay remains capped around 70 FPS.
These checks validate the operators. Full native 9B and 27B model execution remains pending.

The small binary fixture and its provenance are under `os/library/compute/tests/fixtures`.
With the pinned models and reference tools available, regenerate it with:

```bash
target/qwen-venv/bin/python tools/inference/prepare-quantized-fixture.py
```

### Larger Qwen targets

The Linux reference now runs [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) and [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B).
The pinned quantized files come from Unsloth's [9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) and [27B](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) repositories.
The preparation script verifies each file's size and SHA-256 before staging it.
The reference also records the executable, headers, libraries, GPU memory, and complete prompt and output token sequences.

Each model uses FP32 attention and recurrent caches, a 2,048-token context, and a batch size of 32.
Every weight, including embeddings, resides in GPU memory. The reference rejects CPU weight buffers and incomplete layer offload.
The prompt asks for the largest country by land area and a detailed essay about its geography, climate, population, and history.
Each of three sessions generates 128 tokens. Fixed-length timing continues after EOS if the model emits it.

Qwen3.5-9B Q8_0 reaches 57.843, 58.117, and 58.073 tokens/s.
Its two warm samples average 58.095 tokens/s, with prompt times of 43.644 and 43.670 ms for 37 tokens.
Its measured GPU usage after decoding is 10,015,997,952 bytes, or 9.33 GiB.

Qwen3.6-27B Q3_K_M reaches 33.631, 33.990, and 33.979 tokens/s.
Its two warm samples average 33.985 tokens/s, with prompt times of 95.563 and 95.952 ms for the same 37 tokens.
Its measured GPU usage after decoding is 14,338,228,224 bytes, or 13.35 GiB.
This leaves about 2.30 GiB free in the memory pool reported by the NVIDIA driver.
Longer contexts require separate measurements.

Both outputs identify Russia. Each model preserves its exact output token sequence across all three sessions.
These results establish GPU fit and a Linux throughput reference. Native large-model validation and broader quality evaluation remain open.
The first Linux model loads take about 63 seconds and 87 seconds through the shared 9p filesystem.
Those load times include file access and GPU upload, so they do not match the native resident-archive upload timer.

Run the complete Linux suite with the existing pinned reference tools:

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/test-large-qwen.py 9b 27b
```

The suite downloads about 23.1 GB of model files and launches one GPU guest at a time.
It uses 15 GiB of guest RAM by default, within the session's existing 16 GiB memlock limit.
The logs, provenance manifests, and JSON results are under `target/nvidia/large-qwen`.
Use a new `--directory` to preserve an earlier suite's evidence.
The native API still requires quantized weight storage and kernels, separate output weights, and grouped recurrent heads before either model can run.
The 9B file contains Q8_0 matrices and FP32 tensors.
The 27B file mixes Q3_K, Q4_K, Q5_K, Q6_K, and FP32 tensors. Native support must preserve every tensor's actual format.

### Matched Linux AlexNet reference

The Linux reference uses the same FP32 AlexNet archive, image, and eight layer fixtures as D3OS.
It runs cuDNN convolutions and pooling, with cuBLAS dense layers.
Convolutions use FMA math; dense layers use the default FP32 math mode.
All eight outputs must meet the original absolute and relative tolerances of `2e-4`.
The full cuDNN 9.21.0.82 package is checksum-pinned and stays inside the reference guest.
The preparation manifest records the model, fixture, executable, headers, source, and library hashes.

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/linux-reference.py \
  --alexnet-benchmark --memory-mib 12288
```

The first three Linux samples average 0.7368, 0.7320, and 0.7308 ms over 20 forward passes each.
They pass every layer reference and produce class 258.
The preceding native control averages 1.25 ms over 20 passes.
The command-scope pilot reaches 1.05 ms, while the first full-writer sample reaches 1.15 ms.
The following alternating comparison uses longer runs to check those initial samples.

The following A-B-C-C-B-A comparison uses 100 passes per guest and one unchanged cuTile binary.

| Native variant | Samples, ms | Mean, ms | Launches per forward | Queue submissions |
| --- | ---: | ---: | ---: | ---: |
| Control | 1.33, 1.30 | 1.315 | 44 | 44 |
| Command scope | 1.11, 1.16 | 1.135 | 44 | 2 |
| Scope and full writers | 1.04, 1.17 | 1.105 | 31 | 2 |

The combined change reduces the mean by 16.0%, with all layer references passing in every guest.
The full-writer change alone has overlapping wall-time samples; its separate percentage benefit is not established.
The last-forward queue counters confirm that it removes 13 clear kernels while preserving the two-submission command scope.
The benchmark also provides matrix-device timings in a separate instrumented forward pass, outside the normal timing loop.

A separate A-B-B-A comparison keeps the scoped full-writer path fixed and changes only the FP32 matrix reduction tile.
Eight-wide samples reach 1.05 and 1.01 ms; 16-wide samples reach 0.97 and 0.91 ms.
The means are 1.03 and 0.94 ms, an 8.7% improvement.
The five convolution matrix kernels fall from 442.75 to 326.64 microseconds in the separate instrumented pass.
Every sample passes the original eight layer tolerances.
The 16-wide kernel uses 80 registers per thread and has no local spill instructions.
A 64-wide experiment reaches 1.14 ms, while a rank-preserving load variant reaches 1.09 ms. Both are removed.
All retained older kernel instruction sections remain byte-identical.
The final Linux comparison uses 100 passes per sample and reaches 0.7330, 0.7317, and 0.7311 ms.
It still passes all eight layer references. Native AlexNet's 0.94 ms mean remains about 28% slower.
Final regression checks pass GPU operators, Qwen model references, and all 600 DOOM reference views.
The accompanying uncapped DOOM sample reaches 1,607.06 FPS, with regular gameplay still capped around 70 FPS.

The sustained benchmark uses these commands, with one GPU guest at a time:

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/run-qemu.py \
  --firmware target/nvidia/firmware --kernels target/nvidia/kernels \
  --memory-mib 12288 --boot-check --inference gpu-qwen-bench \
  --qwen-long-bench --timeout 240
target/qwen-venv/bin/python tools/nvidia/prepare-linux-qwen.py \
  --gguf target/nvidia/linux-model/Qwen3.5-0.8B-BF16.gguf \
  --d3os-log target/nvidia/vm/serial.log --decode-tokens 4096 \
  --output target/nvidia/matched-qwen-long
taskset -c 8,10,12,14 python3 tools/nvidia/linux-reference.py \
  --llama-directory target/nvidia/linux-llama/llama-b11827 \
  --model target/nvidia/linux-model/Qwen3.5-0.8B-BF16.gguf \
  --matched-qwen-input target/nvidia/matched-qwen-long \
  --matched-qwen-report-selection --memory-mib 12288
```

The first four-tap baseline reaches 181.63, 174.06, and 169.30 tokens/s across its three sustained sessions.
All three produce the same 4,096-token trace and pass the numerical references after timing.
The first split-attention sessions reach 239.90, 235.62, and 232.05 tokens/s with the same complete trace.
The two warm sessions improve by about 36% over this baseline.
The corresponding Linux warm sessions range from 217.18 to 227.25 tokens/s.
These are initial samples. Repeated comparisons remain necessary because throughput changes with host and GPU conditions.

A subsequent A-B-B-A comparison retains the same cuTile binary and changes only the native attention dispatcher.
The control warm sessions reach 178.54, 173.70, 171.41, and 169.37 tokens/s.
The split sessions reach 235.19, 232.53, 233.94, and 232.27 tokens/s.
Their means are 173.25 and 233.48 tokens/s, an improvement of 34.8%.
All twelve sessions preserve the complete token trace and pass the layer, batch, and cache references.
The final short-decode regression reaches 252.96 and 263.37 tokens/s with the original 64-token checksum.
AlexNet averages 1.25 ms over 20 forward passes and passes all eight layer references.
DOOM passes 600 pixel-reference views and reports 1,596.76 FPS in its uncapped `demo1` benchmark.
Regular gameplay remains capped around 70 FPS. These regression samples do not establish a rendering improvement from attention changes.

Linux receives every native token as input, so both stacks perform the same decoding workload.
It selects a different token at position 138 in each run, before the split-attention threshold.
Its two leading scores differ by 0.000061; D3OS reverses their order with a margin of 0.002686.
The two scores differ across stacks by at most 0.004311.
The default audit rejects these traces because it requires identical greedy choices.
The explicit reporting option preserves input, cache, checksum, trace, and device checks while disclosing selection differences.
This option permits performance measurements; it does not establish identical generated answers.

### Borrowed BF16 uploads

The load profiler attributes approximately 86% of the earlier model-load cycles to application code rather than driver service calls.
The archive loader scanned each BF16 matrix, then repacked every value into a temporary word vector before uploading it.
It now sends borrowed little-endian bytes directly to the synchronous upload path after one finite-value check.
Unaligned CPU sources use byte copies. Odd final values use a padded private word instead of reading past the source.

The earlier short benchmark loads the resident model in 1,932 ms; the sustained control runs measure 1,863 to 1,894 ms.
The direct-upload short benchmark measures 748 ms, and its sustained run measures 676 ms.
The sustained warm decode rates remain 237.96 and 235.66 tokens/s with the complete original trace.
This change reduces loading time without changing weight representation or GPU inference arithmetic.
GPU tests preserve every source bit across all four alignments and odd tails, including a transfer larger than the staging buffer.
The boot transfer test also checks unaligned uploads and downloads with CPU and GPU boundary guards.

The first 2026-10-23 follow-up uses native copy-engine transfers, two QMD waves, fused Qwen residual operations, and a versioned DOOM texture cache.
An A-B-B-A comparison uses the same cuTile binary and host CPUs 8, 10, 12, and 14.
The control disables only the residual fusions and texture validation shortcut.
Host load averages range from approximately 0.2 to 2.2.

| Workload | Control | Follow-up |
| --- | ---: | ---: |
| Warm Qwen decode, four sessions | 220.69 to 230.22 tokens/s | 222.22 to 240.60 tokens/s |
| Mean warm Qwen decode | 227.42 tokens/s | 231.02 tokens/s |
| Warm 22-token prompt, median | 11 ms | 10.5 ms |
| Uncapped DOOM `demo1`, two guests | 1518.87 and 1509.67 FPS | 1628.07 and 1617.50 FPS |
| DOOM scene preparation, mean cycles/frame | 651,930 | 543,820 |

DOOM improves by about 7.2%, with almost unchanged palette conversion and display costs.
Qwen removes 48 launches per token, reducing 64-token decode work from 19,520 to 16,448 launches.
Its mean throughput improves by 1.6%, but individual sessions overlap substantially.
These samples do not establish a stable Qwen throughput gain from the fusions alone.
Every Qwen session preserves the exact token checksum and passes all layer, batch, and cache references.

A fresh matched Linux guest measures 262.31 and 262.82 tokens/s for independent warm sessions.
Its prompt times are 6.74 and 7.14 ms, with the same weights, token work, and FP32 caches.
All six Linux sessions reproduce every greedy token choice.
The native sample mean remains about 12% below Linux throughput, so the parity goal stays open.
The DOOM Linux reference still uses a headless software renderer and does not establish matched GPU rendering parity.

An optional serialized GPU-timestamp pass separates device work from CPU preparation.
It measures approximately 9.90 ms of prefill device work, including 4.12 ms in regular projections and 2.64 ms in recurrent blocks.
Split projections and their reduction use another 2.54 ms.
Timestamp instrumentation changes scheduling, so these values identify costs rather than replace normal throughput measurements.

The streaming essay demonstration generates 4,096 tokens in 24.393 seconds after the first token, averaging 167.88 tokens/s.
Its longer context, terminal output, and recording costs differ from the 64-token benchmark.
This result is not a regression measurement against the short benchmark.

### Recurrent prompt processing

The combined recurrent kernels reach 255 registers per thread and contain local-memory spill instructions.
Prompt batches now separate convolution history from the recurrent matrix and distribute each head across 32-column blocks.
The new kernels compile with 96 and 246 registers per thread, respectively, with 128 threads per block and no local-memory spills.
The operator suite and all layer, batch, and cache references pass with unchanged tolerances and exact greedy tokens.

An initial warm 22-token prompt falls from approximately 11 ms to 9 ms.
The serialized device profile reduces the recurrent portion from 2.64 ms to 1.13 ms across 18 layers.
A 16-column variant increases that portion to 1.46 ms and is removed.
Single-token decoding retains the combined kernel.
The separated stages reduce its device work but add launches, allocations, and request handling, reducing observed throughput.

### Grouped recurrent projections

The single-token recurrent path now groups its four independent input projections into one cuTile dispatch and one output allocation.
The recurrent kernel reads four disjoint ranges from that buffer.
The grouped entry uses the existing FP32 reduction arithmetic and 56 registers per thread.
Every earlier kernel instruction section matches the control byte for byte.
The model weights and archive format remain unchanged.

An A-B-B-A comparison uses the same compiled module, CPU affinity, and token work in all four guests.
The grouped guests each produce two warm samples of 250.98 tokens/s.
The controls measure 207.79 to 240.60 tokens/s, with noticeable variation between sessions.
The grouped mean is about 4.4% below the fresh Linux mean of 262.57 tokens/s.
These samples establish progress toward parity without removing the remaining prompt-latency or long-context work.
Warm prompts measure 8 to 9 ms in the grouped guests.
Every session retains the exact checksum and passes all layer, batch, and final-cache references.

Each generated token now requires 203 native kernel launches, compared with 257 before grouping.
The 64-token decode count falls from 16,448 to 12,992 launches, with queue batches falling from 768 to 704.
Protocol version 5 expands the per-kernel profile array to 41 entries, so applications require the matching kernel and SDK.
The operator suite covers all four output ranges, partial reduction tiles, unaligned word offsets, and neighboring output guards.

A separate 64-bit weight-load experiment preserves correctness but increases register pressure and reduces warm throughput to approximately 220 tokens/s.
Its immediate control measures 238.81 tokens/s in both warm sessions. The experiment is removed.
A single-token Tensor Core output-head variant also shows no complete-workload benefit and is removed.

### Four-tap decode specialization

Qwen uses four convolution taps, while the generic combined recurrent kernel reserves space for eight.
The four-tap specialization halves the live history and convolution-weight tiles.
It still uses 256 threads and reaches 255 registers per thread, but emits no local spill loads or stores.
The compiler audit rejects future spill instructions in this entry.
Larger convolution histories retain the generic kernel.
All earlier kernel instruction sections remain byte-identical.

The first warm decode samples reach 262.30 and 263.37 tokens/s with 8 and 9 ms prompt times.
The fresh matched Linux reference measures 262.31 and 262.82 tokens/s.
These short-decode samples reach the Linux throughput range while preserving the exact token checksum and every numerical reference.
A repeat reaches 234.43 and 252.96 tokens/s, while its following Linux reference reaches 262.76 and 262.93 tokens/s.
The first sample shows that throughput is achievable, but repeated runs do not yet establish consistent parity.
Prompt latency, cold loading, and longer contexts still require separate measurements.

## Earlier retained measurements

The final retained sample on 2026-10-23 uses the cached QMD templates and two submission waves described below.
All GPU programs still use cuTile Rust. The corner overlay is removed, and FPS remains beside the window title.
Both guests use host CPUs 8, 10, 12, and 14, with the same audited weights and token work.

| Workload | D3OS | Matched Linux reference |
| --- | ---: | ---: |
| Warm Qwen generation, independent sessions | 215.49 and 230.22 tokens/s | 253.11 and 251.36 tokens/s |
| Warm 22-token prompt | 11 and 10 ms | 7.61 and 7.66 ms |
| AlexNet forward pass, 20 repeats | 1.30 ms mean | No fresh matched run |
| Uncapped DOOM `demo1`, three guests | 1153.22, 1270.89, and 1270.89 FPS | Earlier headless software reference only |

The Qwen sample means leave D3OS about 12% below Linux generation throughput.
Prompt latency also remains higher, so the Linux performance goal stays open.
Every native session retains the exact 64-token checksum and passes all layer, batch, and final-cache comparisons.
All six Linux sessions reproduce every greedy choice, with FP32 caches and full GPU offload.
Linux reset-context warm sessions measure 255.18 to 255.55 tokens/s and 6.62 ms for the prompt.

DOOM renders through the native NVIDIA backend, with a 320-by-200 indexed surface scaled to 640 by 400 for presentation.
The demo runs uncapped for 7117 game ticks. These figures do not describe the interactive frame limiter.
The host runs concurrent Java builds; recorded load averages vary around 12 to 14 during this final suite.
The earlier A-B-B-A comparison below records the timing variation and does not isolate a driver-only DOOM regression.
The retained build also passes the separate 600-frame pixel comparison and GPU checks after game exit.
The production image, host tests, SDK Clippy checks, and both pushed CI workflows pass.

Further projection layouts, running Tensor Core accumulators, and bulk descriptor copies showed no complete-workload benefit in their measured runs.
Those experiments are removed from the maintained implementation.

## Maintained cuTile base

All 41 native inference and rendering operators now use NVIDIA cuTile Rust.
The former NVPTX crate and GPUOcelot test harness are removed.
The host compiler produces Tile IR 13.4 and assembles native Ada binaries with CUDA 13.4.92.
D3OS retains its Rust driver, inference API, and buffer ownership.
Explicit command scopes now group dependent operations before a completion wait.

The migration baseline below precedes the later follow-ups. It uses host CPUs 8, 10, 12, and 14 on 2026-10-23:

| Workload | Before migration | cuTile observations |
| --- | ---: | ---: |
| Qwen warm generation | 160.80 to 164.52 tokens/s | 152.75 to 153.85 tokens/s |
| Qwen 22-token prompt | 15 ms | 20 to 21 ms |
| AlexNet warm forward pass | 1.45 ms | 1.30 ms |
| DOOM uncapped `demo1` | 1423.40 FPS | 1127.13 to 1339.22 FPS |

These are samples from a shared development host, not fixed performance guarantees.
The fresh DOOM baseline has one run; its earlier native measurements also show substantial variation.
The cuTile results retain the existing precision policy and independent numerical checks.
Qwen produces identical 64-token benchmark output and passes all 24 layers across 29 reference positions.
AlexNet passes all eight layer references.
DOOM passes 600 frames of pixel comparisons, exits cleanly, and then passes the GPU operator suite.

The first complete cuTile port generated only 106.85 to 109.59 tokens/s.
Its combined vector and matrix entry reserved matrix resources during decoding.
Separate matrix, small-vector, and wide-vector entries recovered much of that loss.
Wide BF16 projections use paired 32-bit weight loads.
Their measured bandwidth increased from 272 to 556 GB/s.
Smaller projections retain separate cuTile implementations; their later layout changes are described below.
All variants use the same native request ABI.

D3OS now reads fixed thread counts from the binaries and rejects mismatched launches.
It binds compiler constant banks and selects an Ada shared-memory partition from each kernel's requirements.
The native timer check compares QMD timestamps against host elapsed time.
The scratch check runs a cuTile kernel with real local-memory spills and verifies every checksum.

cuTile remains the maintained base despite the remaining regressions.
The Linux throughput target is still open.
Projection latency, matrix scheduling, recurrent tile layout, submission overhead, and presentation remain optimization work.

### Copy-engine and service-profile follow-up

Host transfers now use the dedicated Ampere DMA COPY B engine instead of a compute copy kernel.
This path retains the existing channel and drains compute work before transferring ownership.
Its separate command page remains owned until both engine completion and the FIFO tail arrive.
The command encoding follows NVIDIA's [DMA COPY B methods](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/570.144/src/common/sdk/nvidia/inc/class/clc7b5.h).

The first exact-bit transfer suite passes all six sizes, including partial chunks and neighboring guards.
Its warm round-trip sample measures 7,529 MiB/s, compared with 6,736 MiB/s in an earlier compute-copy sample.
These initial samples do not isolate shared-host variation.
The first DOOM timedemo measures 1361.17 FPS, and the separate 600-frame pixel and input checks pass.
Qwen retains its exact tokens and full numerical references, with warm samples of 214.77 and 225.35 tokens/s.

`Nvidia::service_profile()` separates request handling from queue submission and waiting, indexed by the native operation ID.
The additive query retains request ABI version 3 and validates the complete writable destination before copying counters.
Service cycles include locking and buffer management; syscall pointer validation happens before this measurement.
The benchmark takes snapshots outside its timed intervals and reports the differences for prompt processing and decoding.
An initial 64-token decode uses 6.2 million CPU cycles for frees, excluding queue work.
Output-producing requests use 41.7 million.
This evidence favors reducing dependent launches rather than replacing the allocator without further measurements.

### Projection layout follow-up

BF16 matrix projection kernels now preserve two-dimensional tiles through masked memory loads.
The pinned Rust frontend broadcasts scalar load padding with the wrong rank for these loads.
An explicit tile selection supplies zero padding after the masked load.
FP32 matrix products, convolution, and single-token projections retain their measured layouts.

Fresh baseline sessions measured 20 ms for the 22-token prompt and 148.15 to 159.20 tokens/s for generation.
The final BF16 layout measured 15 ms and 156.10 to 160.00 tokens/s with the same native launcher.
This reduces prompt latency by 25%; these decode samples do not establish a generation speedup.
Every session retained the exact token checksum and passed the full model reference checks.
The binary audit found changes in only two entries: `tensor_project_bf16_weights` and `project_bf16_split`.
All other kernel instruction sections match the baseline byte for byte.

A further BF16 sweep compared 8-, 16-, and 32-element reduction tiles with the retained matrix layout.
At this stage, regular projections use 16 elements and split projections retain 8 elements.
The regular-projection phase decreased from 17.13 to 15.07 million dispatch cycles, including waiting.
The combined layout measured 14 to 15 ms for the prompt, with the same exact tokens and reference checks.
The 32-element variant measured 18 to 19 ms and was rejected.

Several experiments were rejected after measurement:

- Applying this layout to every matrix load increased the large FP32 projection microbenchmark from 33.49 to 47.54 microseconds.
- Increasing the Tensor Core reduction tile from 8 to 32 elements increased prompt time to 22 ms.
- Larger shared-memory carveouts showed no clear workload gain and inconsistent convolution timing.
- Expanding small-vector reduction tiles from 128 to 512 elements increased register use without improving generation throughput.

cuTile remains the maintained compiler throughout these comparisons.

The Linux kernel reference now includes the driver's dynamically loaded compiler libraries.
Without these libraries, loading the CUDA 13.4 CUBIN failed with error 221.
The reference records the driver library hashes and the benchmark source, executable, and CUBIN hashes.
A fresh same-binary comparison reports these FP32 projection times in microseconds:

| Shape (rows, inner, columns) | Native GPU timestamps | Linux CUDA events | Linux cuBLAS |
| --- | ---: | ---: | ---: |
| 1, 1024, 3584 | 10.62 | 8.67 | 8.06 |
| 1, 3584, 1024 | 11.71 | 10.04 | 8.53 |
| 1, 1024, 248320 | 1814.69 | 1813.33 | 1815.76 |
| 3025, 363, 64 | 33.59 | 30.52 | 14.95 |

The timestamp scopes differ as described above, and these are individual samples on a shared host.
The vocabulary kernel reaches comparable throughput in both systems.
The other results leave both scheduling costs and matrix-kernel efficiency as optimization work.

### CPU copies of DMA storage

The staging path uses bulk CPU copies after DMA completion and before the next GPU submission.
On x86, `REP MOVSB` preserves observable memory accesses without using SIMD registers in the kernel.
The existing completion waits and memory barriers still protect staging ownership.
Syscall entry also clears the direction flag before Rust runs, then restores the caller's flags on return.
A hardware check enters with the flag set and verifies the profile copy and restored user flag.

Three runs keep the same cuTile bundle, CPU affinity, WAD, and 7,116 rendered frames.
Mean palette-phase cycles per frame decrease from 465,954 to 431,296, about 7.4%.
Overall DOOM rates are 1324.97 to 1339.22 FPS before and 1332.06 to 1361.17 FPS after.
The overlapping ranges do not establish a reliable overall FPS improvement.
The final runs retain all model references and 600 frames of independent pixel verification.

### Immutable module images

The driver now uploads each CUBIN image once and shares it across its entry points.
Each entry retains its own register, stack, shared-memory, and constant-bank metadata.
The image remains private and immutable for the device lifetime.
This removes repeated PCIe writes of an entire module during first use of individual operators.

With the same cuTile bundle, first-use Qwen prompt processing falls from 479 ms to 18 to 19 ms across three fresh guests.
The first 64-step decode falls from 632 ms to 416 to 447 ms.
Model file access, model upload, and session allocation remain separate timings.
Warm generation still falls within the observed host variation and remains below the Linux target.
All three Qwen guests pass the complete token, layer, and cache comparisons.
AlexNet, the operator suite, and 600-frame DOOM verification also pass with shared images.

### Owned command batches

`Backend::with_commands()` now groups dependent GPU work while preserving the synchronous API outside explicit scopes.
Qwen records each transformer pass inside a scope.
The native queue owns 32 QMD slots, their constants, and their completion storage until the batch finishes.
QMD dependencies preserve order without a CPU wait after every operator.
Scope completion, CPU transfers, and profile snapshots drain pending work.
The driver also waits for the FIFO tail before reusing command storage.
Temporary allocations remain retired until that completion point.

The same 64-token decode records 19,584 launches in 896 submissions, about 22 launches per submission.
The first scope-only measurement generates 195.72 to 200.63 tokens/s, compared with roughly 150 to 160 before batching.
A 128-slot queue shows no clear overall gain and increases first-use allocation costs, so the maintained queue retains 32 slots.
A separate bulk-free API also shows no clear model gain and is removed.
Hardware checks cover queue wraparound, temporary allocation reuse, readback inside a scope, nested scopes, and returned errors.

### Small BF16 reductions

Small even-width BF16 projections now use three-dimensional `[2, 4, 32]` tiles with explicit masked loads.
The gated kernel's static shared storage decreases from 8,192 bytes to 64 bytes, excluding the driver's reserved prefix.
Its register count decreases from 64 to 48 per thread.
The compiler still chooses the physical lane layout.
Separate entries prevent odd-width fallback resources from increasing the even-width launch requirements.

The 1-by-1024 projection with 3,584 outputs decreases from 9.71 to 8.36 microseconds in GPU timestamp measurements.
The warm gated operation decreases from about 23 to 17 microseconds through the application API.
After 128 MiB of cache eviction, the gated operation remains near 43 microseconds.
The warm microbenchmark gain therefore does not establish the same improvement for model weights streamed from VRAM.
A reordered pair layout increases register use and reduces full-model throughput, so it is rejected.

### Measurements with command scopes

Three fresh native guests use the retained cuTile bundle and host CPUs 8, 10, 12, and 14.
The matched Linux guest runs afterward with the same GPU, token contract, FP32 caches, and 256-token capacity.

| Workload | D3OS | Linux reference |
| --- | ---: | ---: |
| First-use 22-token prompt | 17 ms | 245.96 ms |
| Warm 22-token prompt | 12 to 13 ms | 7.48 ms in a new context |
| Warm 64-token generation | 192.19 to 204.47 tokens/s | 256.03 to 256.78 tokens/s in a new context |
| Warm AlexNet forward | 1.45 ms | Separate reference required |
| Uncapped DOOM `demo1` | 1383.86 to 1423.40 FPS | Earlier headless software baseline only |

The six warm native generation samples average 199.56 tokens/s.
Linux reset-context samples generate 254.85 to 256.52 tokens/s, with prompt times of 6.55 to 6.96 ms.
The first-use intervals describe these prepared guests, with different kernel and graph setup costs.
They do not establish a general cold-start advantage for D3OS.
Warm native generation remains about 22% below this Linux reference, so the performance goal remains open.

DOOM uses the native NVIDIA compute renderer and a 640-by-400 output framebuffer.
The three runs complete the same 7,117 game ticks and 7,116 rendered frames, without a frame-rate cap.
The duplicate corner overlay is removed. FPS remains above the game image, beside the title.
The 600-frame pixel comparison and GPU operator checks after game exit still pass.
All three Qwen guests retain the exact 64-token checksum and pass every layer, logit, batch, and cache comparison.
AlexNet retains class 258 and passes all eight reference layers.

### Native BF16 Tensor Core products

Multi-token projections now load BF16 weights directly and split FP32 activations into a BF16 high part and residual.
Both products accumulate in FP32, with compensated summation across reduction tiles.
This changes intermediate rounding from the previous TF32 decomposition.
It retains more precision than a single BF16 activation cast, but does not promise an exact FP32 dot product.
All operator and model tolerances remain unchanged.

Regular projections use 64-element reduction tiles, while split projections use 16 elements.
The larger tile amortizes the compensation work across more inputs.
The binary audit now requires BF16 Tensor Core instructions in both entries.
Only these two instruction sections change. Single-token projections, AlexNet kernels, and rendering kernels remain byte-for-byte identical.

Three fresh guests measure 10 to 11 ms for the warm 22-token prompt, compared with 12 to 13 ms before this change.
The first measured prompt drops from 17 ms to 15 to 16 ms.
Prompt queue-wait cycles decrease from about 28.7 million to 23.8 million in the first comparison.
The full reference trace, final caches, and exact 64-token checksum pass in all three guests.
The unchanged decode kernels generate 193.94 to 204.47 tokens/s, within host variation.
Fresh AlexNet verification measures 1.25 ms per forward pass across 20 repeats.
Three fresh DOOM runs measure 1376.22, 1376.22, and 1311.03 FPS with the corner overlay removed.
These rendering samples overlap the wider observed variation and do not establish an effect from the matrix change.
The 600-frame pixel comparison and GPU checks after game exit pass again.
The complete compiler validation also passes, with the same CUBIN bytes as the hardware-tested bundle.

An independent recurrent experiment also reduced queue-wait cycles by about 4%, but showed no clear full-model throughput gain.
Its extra preparation kernels and allocations consumed the benefit, so the maintained path retains the simpler fused recurrent operation.

### Cached launch metadata

Each loaded program now retains a QMD template with its code addresses, resource metadata, and compiler constant banks.
Every launch copies the template and patches its dimensions, argument-buffer address, completion address, timing mode, and current scratch limit.
Dependency links modify only the submission copies. They cannot change the cached template.
The queue retains the existing compute and FIFO completion waits.

Three fresh guests record 27.0 to 32.8 million preparation cycles for the same 64-token decode.
The previous three guests record 52.8 to 57.7 million cycles.
These samples reduce preparation costs by about 38% to 53%, with host variation in both groups.
Three guests provide six warm generation samples from 203.82 to 210.53 tokens/s, averaging 206.37 tokens/s.
All guests pass every token, layer, logit, and cache comparison.
The GPU operator suite and timestamp measurements also pass with cached templates.
Warm prompt processing remains 10 to 11 ms, and AlexNet measures 1.25 ms per forward pass across 20 repeats.
Fresh DOOM runs measure 1423.40, 1383.86, and 1407.32 FPS with the title-bar FPS display.
The separate 600-frame pixel comparison and GPU operator checks after game exit also pass.

### CPU recording overlaps GPU execution

The queue now owns two waves of 32 slots, with separate command pages and completion reports.
The CPU records the next wave while the GPU executes the previous wave.
GPU semaphore acquires preserve stream order, and the CPU waits before it reuses a wave's storage.
Each allocation retains a retirement sequence until both compute and FIFO completion make reuse safe.
Transfers, profile snapshots, nested scope completion, process cleanup, and page-table growth still drain pending work.
GPU timestamp instrumentation retains individual synchronous submissions.

The completion reports use circular 64-bit sequence numbers and are never cleared during reuse.
A newer report also satisfies an older GPU consumer, so reuse cannot erase a pending acquire's signal.
Host tests cover counter wraparound and stale reports.
The command-scopes hardware test covers repeated wave reuse, temporary allocation reuse, readback, nesting, and returned errors.
The encoding follows NVIDIA's [Ampere FIFO semaphore fields](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/570.144/src/common/sdk/nvidia/inc/class/clc56f.h).

Three initial guests generate 214.77 to 231.88 tokens/s across six warm sessions, with the same exact token checksum.
Their complete layer, logit, batch, and final-cache comparisons pass.
Warm prompt processing remains 10 to 12 ms. AlexNet measures 1.15 ms across 20 forward passes.
The separate 600-frame DOOM pixel comparison and GPU checks after exit also pass.

A later A-B-B-A comparison runs the previous synchronous-wave driver and the new driver on the same kernel bundle.
The development host has concurrent Java builds, so the harness also records host load and CPU frequency samples.
The previous driver generates 185.51 to 209.15 tokens/s. The new driver generates 213.33 to 221.45 tokens/s.
The old DOOM samples measure 1283.99 and 1290.65 FPS. The new samples measure 1258.06 and 1169.46 FPS.
CPU display-copy cycles vary substantially as host frequencies change, so these game samples do not isolate a driver-only throughput difference.
All guests use host CPUs 8, 10, 12, and 14, with the same WAD and uncapped demo.
The synchronous path skips GPU acquires for dependencies that the CPU already observed as complete.
The corner overlay remains removed, and FPS remains beside the title.

The following sections preserve the earlier NVPTX measurements and implementation history.
They do not describe the current compiler-generated register layouts.

## Previous NVPTX measurements

Measurements on 2026-10-22 and 2026-10-23:

| Workload | Initial resident baseline | After native optimizations |
| --- | ---: | ---: |
| Qwen warm generation | 7.04 to 7.36 tokens/s | 146.79 to 162.85 tokens/s |
| Qwen warm prompt processing | 7.76 to 7.81 tokens/s | 1466.67 tokens/s |
| AlexNet forward pass | 10.0 ms | 1.45 to 1.55 ms |
| DOOM uncapped `demo1` | 536.84 to 676.89 FPS with word storage | 1290.65 to 1439.86 FPS with resident byte textures |

The AlexNet result excludes input upload and final download.
The DOOM result includes native GPU drawing, palette conversion, scaling, and framebuffer copies.
Its scene size is 320 by 200, with presentation at 640 by 400.
The regular game loop has separate timing behavior. Its HUD rate does not measure uncapped throughput.
The DOOM comparisons use three runs per renderer on host CPUs 8, 10, 12, and 14.
These CPU IDs belong to four distinct performance cores on this machine.
The earlier word renderer reports 676.89, 548.67, and 536.84 FPS.
The byte renderer reports 800.95, 943.54, and 892.81 FPS.
Both renderers use the same driver, kernel bundle, engine, WAD, and 7117-tick demo.
The baseline renderer also includes the same profiling counters.
Host CPU performance varies between runs, so these ranges do not establish a fixed percentage gain.

A new comparison keeps the byte shaders and driver fixed while changing upload and cache ownership.
The fresh per-batch baseline reports 880.19, 880.19, and 980.69 FPS.
The resident version reports 1431.58, 1431.58, and 1439.86 FPS.
Both render 7,116 frames across the same 7,117-tick demo, with the same WAD and host affinity.
The GPU shader bundle is identical in these six runs.
Later inference-kernel regression runs report 1290.65, 1407.32, 1376.22, and 1297.37 FPS with the resident renderer.
The final run retains the same 7,116 frames, upload counts, and five buffer-growth clears.
These later samples remain included in the table's range; they do not establish a rendering speedup from the inference changes.

Both launchers record `NVIDIA_HOST_CPU_AFFINITY` before starting a guest.
The pinned DOOM runs use the following commands, one guest at a time:

```bash
taskset -c 8,10,12,14 python3 tools/nvidia/run-qemu.py \
  --firmware target/nvidia/firmware --kernels target/nvidia/kernels \
  --memory-mib 4096 --boot-check --inference gpu-doom-bench --timeout 180
taskset -c 8,10,12,14 python3 tools/nvidia/linux-reference.py --doom-benchmark
```

These CPU IDs are machine-specific. The earlier Qwen Linux measurements used unrestricted host affinity.
A new BF16 reference on these pinned cores reports 264.90 tokens/s and shuts down cleanly.
Those earlier results remain historical. The matched benchmark below replaces them for current model comparisons.

The Linux llama.cpp reference uses build `b11827`, full GPU offload, and three repetitions.
The earlier unrestricted runs report about 250 tokens/s with BF16 weights and 156 tokens/s with FP32 weights.
The FP32 file is a conversion of the same BF16 GGUF file without quantization.
The earlier `llama-bench` reference retains its default FP16 attention cache. D3OS uses FP32 state.
Both benchmarks request 22 prompt tokens and 64 generated tokens, but they use different token sequences.
The Linux hybrid-model benchmark can round prompt work to its required chunk size.
Its reported prompt throughput therefore cannot serve as an exact 22-token latency comparison.
These results establish the remaining gap, rather than an exact application speed ratio.

The GGUF reference comes from `unsloth/Qwen3.5-0.8B-GGUF` revision `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`.
The D3OS archive retains the pinned Qwen revision and original Transformers fixture described in [the Qwen guide](qwen.howto.md).

## Matched Qwen reference

The accepted comparison uses the same 22 prompt tokens, 64 decode inputs, batch limit of 32, and 256-token cache capacity.
Both guests use four virtual CPUs on host CPUs 8, 10, 12, and 14, with 12 GiB of guest RAM.
Attention caches, convolution history, and recurrent state use FP32 in both implementations.
Linux uses llama.cpp `b11827` with full GPU offload and backend greedy selection.
Automatic Flash Attention resolves to the enabled CUDA path in the accepted run.
Each decode step waits for the complete model result.
This historical comparison predates command scopes, when D3OS also waited after each individual kernel submission.

The weight audit covers all 320 text tensors and 19 model settings.
All 187 BF16 tensors match bit for bit.
The other 133 tensors match exact FP32 expansion or the converter's normalization and decay transformations.
Recomputed negative exponentials differ by at most two FP32 rounding units across the local and converter implementations.
Other converted tensors match exactly. The inference reference tolerances remain unchanged.
The archive hash, source checkpoint, GGUF hash, headers, libraries, and token contract are recorded in manifests.

D3OS exports its complete token traces outside the timed intervals.
The preparation tool independently checks the prompt with the pinned upstream tokenizer.
Linux feeds the exported decode tokens and separately computes every greedy choice.
All 12 Linux sessions across two VMs reproduce all 64 D3OS choices, with checksum `11137929795425568356`.
The launcher checks the full traces, actual FP32 cache reports, settings, offload, and evaluated token counts.
A differing greedy choice rejects the accepted comparison instead of silently changing the workload.

The fused-attention comparison uses three D3OS VMs and the two matched Linux reference VMs on the same GPU.
This comparison precedes the paired-load changes described below.
Every prompt and decode token in the new D3OS runs still matches those Linux references.

| Phase | D3OS | Linux, new context | Linux, reset context |
| --- | ---: | ---: | ---: |
| Read model into RAM | 1,683 to 2,056 ms | 5,543 to 7,077 ms | Same resident model |
| Load model from RAM | 1,900 to 1,988 ms | 815 to 1,148 ms | Same resident model |
| Create or reset session state | 6 to 20 ms | 308 to 375 ms | 0.062 to 0.068 ms after creation |
| First prompt after model load | 129 to 130 ms | 194 to 306 ms | Already warmed |
| Warm 22-token prompt | 15 ms | 6.78 to 6.90 ms | 6.32 to 6.43 ms |
| Warm prompt throughput | 1466.67 tokens/s | 3187.02 to 3246.19 tokens/s | 3423.12 to 3479.17 tokens/s |
| First 64-step decode | 442 to 458 ms | 251 to 260 ms | Already warmed |
| Warm 64-step decode | 393 to 414 ms | 245.35 to 248.14 ms | 245.97 to 247.49 ms |
| Warm generation | 154.59 to 162.85 tokens/s | 257.92 to 260.85 tokens/s | 258.60 to 260.19 tokens/s |

The first Linux model call includes initial kernel and graph setup.
These first-use values do not establish an uncached driver-installation cost.
New-context measurements match D3OS's independent-session structure.
Reset-context measurements retain Linux's context and graph allocations, with cleared model state and no prompt-cache reuse.
They provide a second target for a service that keeps its inference context alive.
Linux's long context-creation time remains outside the throughput intervals, just as D3OS session allocation does.

Both model files enter guest RAM before model loading starts.
Linux uses a private RAM file and disables model mmap. Its loader completes asynchronous weight uploads before returning.
Filesystem paths and archive layouts differ, so the file and load measurements do not establish identical storage performance.
Linux computes backend greedy selection with its model graph, including one unused candidate after the final decode step.
D3OS selects immediately before each decode step. This shifts one selection across the prompt/decode timing boundary.
The measured 64 choices and model inputs still match exactly.

Intermediate arithmetic can still differ between implementations.
The library's [BF16 GEMM path](https://github.com/ggml-org/llama.cpp/blob/b11827/ggml/src/ggml-cuda/ggml-cuda.cu) can convert activation operands to BF16 while accumulating in FP32.
D3OS retains FP32 activations through compensated TF32 decomposition.
Matching weights, cache storage, and token work does not establish identical intermediate rounding for every operator.
A separate precision policy must be measured explicitly before adopting lower-precision native operands.

The matched result confirms a prompt-latency gap of about 2.2 to 2.4 times and a decode-latency gap of about 1.6 to 1.7 times.
Host performance still varies; these samples do not promise a fixed ratio for other prompts or context lengths.
The profile continues to identify native projection work and synchronous submission as major decode costs.
Prompt attention now uses one fused launch per attention layer; projection work still dominates the remaining interval.
Read the [Linux comparison procedure](nvidia-vfio.howto.md#compare-against-linux) to reproduce the audit and both guests.

## Previous kernel implementation

The projection kernel selects a warp reduction for a single input row.
Each warp reads contiguous weights. FP32 matrices and very wide BF16 matrices reuse each input across eight outputs.
Single-row BF16 matrices with 2,049 to 8,192 outputs use four outputs per warp.
Narrow matrices use one output per warp to occupy more GPU multiprocessors.
This avoids the 15 unused rows in a padded tensor-core tile.
Larger matrix operations retain the TF32 tensor-core path.
The FP16 and BF16 tensor-core boot checks also remain active.

Aligned single-row BF16 projections now read two weights per lane through one 32-bit load.
The input width must be divisible by 64; other widths retain scalar reads.
The kernel checks weight alignment before selecting the paired reads, following the [PTX memory alignment requirements](https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands).
Both weights expand exactly into FP32, and the kernel retains FP32 fused multiply-add accumulation.
The paired lane layout changes the reduction order. Existing operator and model tolerances remain unchanged.

A pinned comparison uses the same benchmark code with the previous and current driver and GPU kernel pairs.
The four-output layout changes the driver grid; each run uses the matching pair.
The benchmark now measures packed BF16 weights directly and uses their two-byte size in bandwidth calculations.
It compares every output and runs a separate GPU timestamp pass across 100 launches.

| Single-row BF16 projection | Previous device time | Paired reads and medium-width tiling |
| --- | ---: | ---: |
| 1,024 inputs, 1,024 outputs | 5.156 µs | 5.121 µs |
| 1,024 inputs, 3,584 outputs | 9.769 µs | 6.888 µs |
| 3,584 inputs, 1,024 outputs | 8.900 µs | 7.376 µs |
| 1,024 inputs, 248,320 outputs | 918.465 µs | 914.735 µs |

The two middle cases improve by about 29% and 17% in this resident-weight test.
The small matrices can remain in GPU cache. These kernel results do not establish the same gain for full-model generation.
The vocabulary projection remains near 556 GB/s, with little change from the previous bundle.
This result points toward submission and other matrix work rather than expecting paired loads to remove the remaining Linux gap.

Two subsequent Qwen VMs retain 15 ms warm prompt times and report 146.79 to 158.81 tokens/s for generation.
Both pass all model references and reproduce every prompt and decode token from the accepted Linux comparison.
These full-model results do not demonstrate a generation speedup; host variability remains visible.
AlexNet still reports 1.45 ms per forward pass, and DOOM passes 600 pixel comparisons plus 6,447 forced atlas resets.

The driver stores QMDs, constants, commands, and completion signals in coherent guest RAM.
The CPU no longer writes each launch byte through the VRAM BAR or polls completion through PCIe reads.
The driver configures persistent channel state once and invalidates instruction caches when it loads a program.
Each dispatch retains its compute completion check and final FIFO completion check before buffer reuse.

GPU-owned VRAM uses cacheable mappings. Shared guest RAM retains volatile, system-coherent mappings.
Large application allocations use aligned 2 MiB GPU pages where possible, with 4 KiB pages for tails.
These mapping changes passed the transfer guard checks and full model references.
They did not remove the measured steady-state bandwidth gap by themselves.

RMSNorm uses one 128-thread block per row instead of one warp.
The block combines four partial sums in shared memory and publishes one common scale before writing the output.
In the measured decode profile, RMSNorm dispatch cycles fell from 137.95 million to 79.56 million across 3,136 calls.
This reduction changes the summation order. The original model tolerances and decode checksum still pass.

The recurrent delta step fuses convolution, Q/K normalization, decay, state updates, and gated output into one kernel.
One block handles each head and uses shared storage instead of global intermediate tensors.
Key and value dimensions up to 128 and convolution widths up to eight use this path.
Larger configurations retain the separate operator path.
The fused path removes 126 launches per generated token, including redundant intermediate clears.
Across 64 decode steps, zero kernels fell from 5,824 to 1,216.
The hardware suite compares output, convolution history, and recurrent state across 42 steps, including both paths.

Before cuTile, the loader rounded static shared storage plus the 1 KiB reserved prefix to 128 bytes.
It rejected requests above its fixed 32 KiB partition.
The cuTile migration replaces that limit with metadata-driven partition selection.

The fused delta step first raised warm generation from about 103 to 125–127 tokens/s.
The block-wide RMSNorm change then reached 138.83 tokens/s in both warm sessions of the next run.
A second VM reports 127.75 and 135.31 tokens/s, with the same checksum and complete reference pass.
Host variability remains visible; these runs do not establish a fixed gain percentage.
These measurements use host CPUs 8, 10, 12, and 14.
The complete Qwen reference still passes all 24 layers across 29 positions.

Matrix, RMSNorm, and fused delta operations allocate their output inside the validated dispatch.
A full writer initializes every element before the kernel publishes the new handle.
This removes a separate zero kernel while preserving zeroed ordinary allocations and process ownership.

Token selection reduces the FP32 vocabulary logits into 32 index/value pairs on the GPU.
The CPU combines these pairs after a 256-byte transfer, instead of downloading roughly 1 MB of logits per token.
Equal scores select the lowest index. Every non-finite score causes an error, even when it cannot win.

Qwen matrix weights retain their original BF16 bits in packed VRAM storage.
Embedding converts only the selected row into FP32.
Projection converts weights as it reads them and retains FP32 accumulation.
This changes storage density without rounding activations or weakening the model reference tolerances.

Qwen's feed-forward gate and up projections can share one dispatch with their SiLU multiplication.
The single-row BF16 kernel reads each input for both projections and writes only the final activation.
This removes two launches per layer, or 48 launches per generated token.
Accumulation stays FP32. Mixed storage, FP32 weights, and batched inputs retain the existing separate path.
Nine operator cases compare this operation with the CPU backend, including odd output and packed-weight tails.
Wide gated projections use four outputs per warp, compared with eight for ordinary projections.
Each gated output needs two accumulators. This layout increases active warps without doubling the accumulator count.
The assembled gated kernel uses 36 registers without stack storage or spills.
With this layout, two Qwen VMs report 148.49, 144.80, 143.18, and 140.97 tokens/s in their warm sessions.
Both VMs pass the complete model reference with the existing checksum.

The kernel benchmark also compares fused and separate feed-forward calls with resident weights.
It alternates their order across three rounds and includes allocation, dispatch, synchronization, and output release.
The small weight set can remain in GPU cache. These times do not represent streaming weights across a complete model.
The four-output layout reports 15 to 16 microseconds for fused calls, compared with 39 to 50 microseconds for separate calls.

A second pass writes a separate 128 MiB buffer before each call to create cache pressure.
GPU timestamps measure only the selected projection and activation kernels, excluding the eviction writes and CPU allocation work.
The three fused samples report 42.31 to 42.37 microseconds. Separate kernels report 71.86 to 71.99 microseconds.
Timestamp instrumentation adds scheduling and completion work, so these samples use a separate `instrumented_device_mean_us` field.

Qwen now processes prompt chunks of up to 32 tokens, one complete layer at a time.
Embedding gathers token rows on the GPU. Dense projections and normalization operate on the resulting token matrix.
Multi-row projections use the existing compensated TF32 tensor-core kernels with packed BF16 weights and FP32 activations.
Attention and recurrent cache updates retain token order within each layer.
The NVIDIA backend reuses scratch buffers and writes each token directly into the batch output.
Only the final prompt token receives a vocabulary projection during normal inference.
An independently owned GPU row copy selects that token without downloading activations.

The recorded 22-token warm prompt time falls from 126–135 ms before batching to 37–41 ms.
This gives 536.59–594.59 tokens/s across four batched VMs.
The first prompt in each VM takes 125–136 ms, including initial program loading; file and weight upload remain separate measurements.
Warm generation spans 125.00–149.18 tokens/s across those runs, with the same decode checksum.
The final VM also reports slower file and weight loading. These ranges retain that run and do not establish a fixed gain percentage.
This change targets prompt latency and does not establish a generation speedup.

Cooperative attention preparation assigns one warp to each token and head.
The warp combines its normalization sum, then rotates independent element pairs.
Queries and keys are prepared across the whole chunk. Values enter the cache through one batch copy.
Scoring still reads only the causal prefix for each token.
In the 22-token profile, preparation calls fall from 264 to 12, and cache-copy calls fall from 132 to six.
Preparation dispatch cycles fall from 17.97 million to about 0.30 million in the paired profiles.
The assembled kernel uses 27 registers, without local stack storage, spills, or block barriers.

BF16 weights have seven fraction bits and fit exactly within TF32's precision.
The BF16 matrix path therefore removes the zero weight-residual product while retaining explicit TF32 conversion.
It uses two tensor products per tile, with FP32 activations and compensated FP32 accumulation.
FP32 weight operations retain their three-product path.
The build audit checks Tensor Core instructions in the new split projection entrypoint as well as the original kernels.

Narrow prompt matrices provide too few output tiles to occupy the GPU well.
The backend now divides their input reduction into 16 independent partitions, then combines the partial results in a second kernel.
This path applies to BF16 projections with two to 32 rows, 256 to 2,048 outputs, and at least 512 input elements.
Other shapes retain their existing projection path.
The split kernel uses 37 registers, without local stack storage, spills, or block barriers.
Long, odd-sized operator checks compare against independent FP64 dot products.
Additional guard checks cover a single row, partial output tiles, and empty partitions that must write zeros.

The paired prompt profile reduces projection dispatch cycles from 33.30 million to 24.12 million, including the extra reduction launches.
Recurrent updates then account for about 30% of dispatch cycles; attention preparation accounts for less than 1%.
These cycle measurements include submission and waiting, rather than isolated device execution.

With these kernel changes, four VMs report warm prompt times of 23 to 35 ms, or 628.57 to 956.52 tokens/s.
Seven of the eight warm sessions take 23 to 26 ms; the 35 ms sample remains included in the range.
The final two VMs report 24, 24, 26, and 23 ms, with generation at 140.04 to 156.10 tokens/s.
Across all four VMs, generation spans 140.04 to 157.25 tokens/s.
The first prompt takes 142 to 152 ms, excluding file access and weight upload.
All runs preserve the decode checksum and complete sequential, batched, and cache reference checks.
These changes primarily improve prompt processing; the generation gap remains open.

Recurrent prompt updates now process the entire chunk within one launch per layer.
Each block owns one head and advances its tokens in order, without synchronization between blocks.
Its private recurrent state persists across the chunk, with one final write to the global cache.
The batch layout assigns eight value columns per warp and uses at most 512 threads per block.
Single-token decoding retains its existing four-column layout.
The batch kernel uses 40 registers and 128 bytes of local stack per thread, with no assembler-reported register spills.
The driver allocates this local storage from module metadata before launch.

In the 22-token profile, recurrent calls fall from 396 to 18, and separate output clears fall from 31 to 13.
Recurrent dispatch cycles fall from 16.68 million to 7.79 and 7.83 million in the two final VMs.
Their share of prompt dispatch cycles falls from about 30% to 16–17%.
An intermediate batch implementation without private state measured about 9.05 million recurrent dispatch cycles.
These profiles include CPU submission and completion polling; they do not isolate GPU execution.

The recurrent-kernel validation VMs report warm prompt times of 20, 20, 21, and 19 ms, or 1047.62 to 1157.89 tokens/s.
Their first prompt takes 146 and 156 ms, excluding file access and weight upload.
Warm generation spans 146.79 to 156.86 tokens/s, with the existing checksum.
Both VMs pass every sequential and batched layer, vocabulary, greedy-token, and final-cache comparison.
Operator checks also compare guarded batch buffers against consecutive single-token launches from nonzero initial state.
They retain the existing numerical tolerances, including for the changed batch reduction order.
The input guards and read-only contents remain exact.

Attention now fuses scores, softmax, value accumulation, and gating for an entire token batch.
One 128-thread block owns each token and query head, with grouped heads selecting their corresponding key/value head.
The block streams 128-position tiles and retains its running maximum, normalization mass, and normalized output.
It rescales earlier contributions when a later tile raises the maximum.
Weights are normalized before value multiplication, which avoids overflow from a large unnormalized weighted sum.
All query, key, value, and accumulator values remain FP32.
Head dimensions up to 256 use this kernel; larger dimensions retain the existing separate kernels.

Each block uses 540 bytes of static shared storage, independent of prefix length.
The driver allocates 1,664 bytes including CUDA's reserved prefix and alignment.
The assembled entrypoint uses 43 registers, without local stack storage or register spills.
The build checks its shared loads, stores, and block barriers in SASS.
The SDK validates the complete consumer before executing its checked cache-write requests.
The output uses a full-writer allocation, without a score tensor or output clear.

In the 22-token prompt profile, 396 scoring, softmax, and output calls become six fused calls.
Their dispatch cycles fall from 11.07 million to 0.23 million, with the same token work and cache capacity.
Prefill clear calls fall from 13 to one. Across 64 decode steps, clear calls fall from 832 to 64.
In the profiled decode interval, the three attention kernels previously used 49.73 million dispatch cycles; the fused kernel uses about 30.8 million.
These counters include CPU submission and completion waiting, rather than isolated GPU execution.

All six warm sessions across three new VMs report 15 ms for the 22-token prompt, compared with 19–21 ms in the preceding matched runs.
Warm generation spans 154.59 to 162.85 tokens/s. These overlapping ranges do not establish a fixed generation gain percentage.
Every new run passes the complete model fixture and matches the previous Linux prompt and decode traces exactly.
Raw guarded checks cover partial dimensions, multiple cache tiles, rising and falling score maxima, and prefixes through 1,025 positions.
A large-value case also remains finite where an unnormalized sum would overflow.
The independent CPU operator suite checks the fused path and a 257-dimensional fallback, with unchanged tolerances.
The final AlexNet regression reports 1.45 ms per forward pass. DOOM passes its full pixel checks and reports 1376.22 FPS.

The independent fixture now checks sequential and batched execution across all 24 layers and all vocabulary logits at 29 positions.
Its batches use 17 and 12 tokens to exercise cache continuation at a nonzero position.
It compares greedy selections and final attention, convolution, and recurrent cache contents.
Existing model tolerances remain unchanged.
A CPU test also crosses two 32-token boundaries after an existing prefix and checks subsequent decoding exactly.

The DOOM rasterizer builds ordered column and row lists for each command batch.
Each pixel merges its two relevant lists instead of scanning every command.
The merge preserves draw order, including overlap between columns and spans.
The production raster path stores indexed surfaces and texture data as bytes instead of words.
Each indexed-frame transfer now moves 64,000 bytes instead of 256,000 bytes.
The CPU no longer expands or contracts every indexed pixel between the C engine and the GPU.
Texture snapshots retain their content comparison, including reused zone allocations.
The renderer now retains an 8 MiB texture atlas across frames and uploads only appended snapshots.
A matching prefix can reuse a longer snapshot without extending the caller's readable memory range.
Changed contents append a new snapshot, preserving offsets already referenced by pending commands.
A hash table indexes texture identities and retains its storage across atlas generations.
Before atlas rollover, the renderer completes pending commands and then resets the CPU and GPU upload offsets.

Command and bin buffers grow as needed and remain allocated between batches.
They receive complete new prefixes before dispatch, while shader lengths bound access to those prefixes.
`Nvidia::write_bytes()` writes packed data into an existing word buffer and zero-pads its final partial word.
The byte upload and update paths use the same bulk packing helper.

The byte fuzz kernel assigns one active lane per warp to each framebuffer column.
It preserves pixel order and command order within each column, including overlapping and double-width commands.
Only vertical neighbor offsets are supported. The shader ignores invalid or nonvertical entries.
Independent columns can therefore execute concurrently without changing feedback results.
The original scan and serial fuzz kernels remain word-based references in the operator checks.
The byte checks also cover odd surface sizes, word and byte guards, unaligned tables, invalid offsets, and all four palette scales.

## Scheduling delay around GPU completion

The large vocabulary projection took about 3.36 ms through the application API, although GPU timestamps measured about 1.82 ms.
Its device bandwidth already matched the Linux kernel benchmark.
The GPU service used a spinlock whose owner could be preempted during synchronous completion polling.
The owner could remain descheduled after the GPU finished.

The service now holds a preemption guard while it owns the device lock, including process-buffer cleanup.
Interrupts remain enabled. The guard ends after the lock is released.
The same projection now takes about 1.82 ms through the application API.
Qwen generation increased from about 52 tokens/s to about 100 tokens/s with the same decode checksum and full reference results.

This follows the critical-region rule in the [Linux preemption locking guide](https://docs.kernel.org/locking/preempt-locking.html).
It retains a synchronous service, so long requests can delay scheduling on the calling CPU.
Queued submission and completion interrupts remain necessary for an efficient asynchronous service.

| FP32 projection shape | D3OS device time | Linux device time |
| --- | ---: | ---: |
| 1 by 9216, 4096 outputs | About 287 microseconds | 286.55 microseconds |
| 1 by 1024, 248320 outputs | About 1817 microseconds | 1815.62 microseconds |

The vocabulary projection reaches about 560 GB/s in both guests.
The native firmware clock sample reports a 9001 MHz memory target and a 2310 MHz SM target during boot.
Linux reports the same memory target and a 2040 MHz SM target in its startup query.
These are startup samples, rather than continuous model clock measurements.
The native clock probe also measures about 2309 MHz through GPU timer instructions.

The firmware clock query uses NVIDIA's [RM user shared data layout](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/570.144/src/common/sdk/nvidia/inc/class/cl00de.h).
The driver reads a stable timestamp around each snapshot and stops firmware polling after the boot diagnostic.
It retains ownership of the DMA region for the firmware lifetime.

## DOOM stage profile

Two protocol-19 D3OS VMs record six uncapped GPU timedemos at 1,547.174, 1,566.635, 1,566.635, 1,638.783, 1,628.072, and 1,628.072 FPS.
Their combined mean is 1,595.895 FPS. Each timedemo executes 7,117 gametics with native GPU window presentation and compilation excluded.
Both cohorts use matching kernel and DOOM executable hashes, four pinned virtual CPUs, and 4 GiB of guest memory.
The timing evidence is under `target/nvidia/mnist-parity/adam-doom-operators/` and `adam-doom-operators-canonical/`.
Those intervals finish before the subsequent optimizer validation failures, which remain retained with their guest consoles.
The corrected complete GPU operator suite passes separately under `adam-context-operators/`.
See [the MNIST report](mnist-performance.md#fuse-native-adam-updates) for the fixes and training measurements.
An equivalent Linux GPU rendering and presentation reference remains required; these FPS values do not establish Linux parity.

`NVIDIA_DOOM_PROFILE` separates player-view rendering, palette/scaling transfers, and final framebuffer copying.
Each stage reports CPU cycles per rendered frame, including scheduling delays.
`NVIDIA_DOOM_KERNEL_PROFILE` reports CPU dispatch cycles per frame for each kernel.
These are synchronous CPU measurements, rather than isolated GPU timestamps.
The launcher waits for `NVIDIA_DOOM_PROFILE_DONE` before stopping a timedemo VM.

In the first matched pair, copy dispatch cycles decreased from about 578,000 to 442,000 per frame.
Fuzz dispatch cycles decreased from about 168,000 to 20,000 per frame.
The number of draw, fuzz, palette, copy, and clear operations stayed the same.
The full 600-frame C pixel comparison and the GPU self-test after game exit still pass.

The resident-upload comparison reduces zero kernels from 27,036 to five over the same demo.
Those five clears occur when command or bin buffers grow.
Copy kernels decrease from 55,793 to 49,640, while draw, fuzz, and palette kernel counts remain unchanged.
The resident atlas uploads 2,006,236 padded texture bytes over the entire demo.
It records 24,975,852 cache hits and 44,594 new snapshots, without atlas rollover in that run.
`NVIDIA_DOOM_UPLOAD_PROFILE` reports uploaded bytes, hits, new snapshots, rollover count, and buffer growth count.

Verification uses a 16 KiB atlas to force rollover within live draw batches.
The earlier 600-frame test reports 6,474 resets; the latest regression reports 6,447.
Both match C-rendered pixels, including low detail and fuzz feedback.
The launcher requires rollover coverage and checks that GPU inference still works after game exit.
The byte operator suite also checks incremental updates, shrinking odd-length writes, padding, untouched guards, and rejected ranges.

## Compiler and local-stack correctness

Warp shuffle instructions depend on execution in other lanes.
The inline assembly must not use `pure` or `nomem`, which permit assumptions that do not describe a synchronized collective.
A narrow-matrix check exposed the compiler sinking a shuffle into the lane-zero output branch.
The shared warp helper now preserves participation and uses immediate shuffle distances.
This also avoids a compiler-generated local shift table in the production projection kernels.

A larger validation run exposed a separate driver fault with a valid 40-byte local stack frame.
The same cubin passed under Linux but faulted during Qwen prefill in D3OS.
The loader now reads `EIATTR_MIN_STACK_SIZE` from CUDA ELF metadata and grows scratch storage before dispatch.
It updates the channel's local-memory base, per-TPC allocation, and QMD limit while the queue is idle.
The original failing cubin then passed the complete Qwen reference with the same decode checksum.
A boot diagnostic deliberately uses volatile local storage across 257 GPU threads to preserve this coverage.

A state-retention experiment requested 80 registers per thread in a 1,024-thread block and timed out before producing model output.
That launch exceeds Ada's 64K-register budget. The accepted batch layout uses at most 512 threads and 40 registers per thread.
The driver now rejects excessive register requests before queue submission, including rounded allocation units and partial warps.
Unit checks cover the failing configuration, exact limits, rounding, and overflow inputs.
The check follows NVIDIA's [Ada resource limits](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) and [register allocation guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy).
It validates a necessary launch limit and does not predict full kernel occupancy.

The QMD constant-buffer size uses 16-byte units. The encoded size now matches the actual 1 KiB launch buffer.
Mesa's [QMD v3 implementation](https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/+/a5af649fb690c96e3d1a1e5efb2ca199666c0ff1/src/nouveau/compiler/nak/qmd.rs) provides the same size conversion.

## Per-kernel profile

The resident Qwen benchmark reads `Nvidia::profile()` around prompt processing and decoding in the second session.
The `phase=prefill` and `phase=decode` records separate the two workloads. Logging occurs after both timed intervals.
With protocol version 3, each snapshot reports call counts, CPU preparation cycles, and immediate-launch wait cycles per built-in kernel.
`prepare_share_pct` divides a kernel's preparation cycles by the total preparation cycles in that interval.
It excludes application work and queue submission, so it does not equal that kernel's share of total model latency.
`NATIVE_QUEUE_PROFILE` separately reports submission batches, queued launches, submission cycles, and all completion-wait cycles.
Wait cycles include GPU execution and completion polling, rather than an isolated device timestamp.
GPU timestamp instrumentation forces individual submissions and changes the workload's queue behavior.
The counters cover the whole device. Concurrent GPU workloads invalidate attribution to one model.
CPU cycles are useful within a run and are not a portable time unit across machines.

## Remaining work

Qwen now batches prompt matrices and records dependent kernels inside command scopes.
Attention scoring, softmax, and output now share one tiled launch per attention layer and prompt chunk.
Recurrent updates now use one launch per layer and prompt chunk.
Earlier synchronous profiles put projections and their partial reductions at about 68% of dispatch cycles, and recurrent work at about 22%.
The current preparation counters exclude queue waiting and cannot establish the same attribution.
The next improvements must address matrix efficiency, recurrent execution, and submission costs.
The matched benchmark now uses identical token work, audited weights, FP32 caches, and equal cache capacity.
Longer contexts and prompts still need matched measurements.
The CPU token-selection work now combines only 32 GPU candidates.
Each change must preserve the existing model reference or define and measure a separate precision policy.

The previous large projection kernels matched Linux device throughput for the measured shapes.
The latest retained native sessions generate about 215 to 230 tokens/s, compared with about 251 to 256 in the fresh Linux reference.
The remaining work must reduce complete model latency, rather than optimize already comparable streaming bandwidth.

NVIDIA's cuTile Rust compiler is now the maintained base for every native GPU kernel.
Its [compile-only API](https://nvidia.github.io/cuda-rust/cutile/latest/guide/jit-compilation.html#compile-only-api) produces Tile IR without a GPU or CUDA driver at runtime.
The external `tileiras` assembler then produces a cubin for the requested target.
The [compatibility table](https://nvidia.github.io/cuda-rust/cutile/latest/reference/compatibility.html) includes Ada `sm_89` with Tile IR 13.2 or newer.
D3OS retains its native Rust driver and inference API.
The migration checks argument layouts, resource metadata, GPU execution, and independent numerical references.
The compiler does not replace the remaining work on submission ownership, synchronization, or presentation.

DOOM still transfers packed indexed frames for CPU HUD composition and copies RGB output into the console framebuffer.
Command and texture storage now remain resident, but command lists and bin indices still cross PCIe for each batch.
Each texture use still checks CPU contents to detect mutable or reused source memory.
Direct GPU presentation remains separate work.
Repeated VM starts also exposed an intermittent GSP timeout while allocating the golden compute engine.
The failed boot never reached a workload and is excluded from throughput results.
Successful reruns do not establish that this startup issue is fixed.
The Linux DOOM reference uses the same pinned engine and WAD with a headless framebuffer copy.
The two pinned Linux runs report 2306.44 and 1845.15 FPS for the same 7117-tick demo.
The earlier unrestricted-affinity Linux run reported 1586.59 FPS.
It measures a software-rendering baseline without display-server or physical-display costs.
It does not represent an OpenGL port or a Linux GPU renderer.
The Linux reference now uses a static PID 1 poweroff helper after the benchmark.
The previous asynchronous SysRq path could fault in the shell after device shutdown.
Runs with that shutdown panic remain excluded from the accepted results.

## Implementation sources

The MMU encoding follows NVIDIA's published [MMU-v2 fields](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/570.144/src/common/inc/swref/published/pascal/gp100/dev_mmu.h).
The mapping policy also follows the native [tinygrad NVIDIA page-table implementation](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/support/nv/nvdev.py).
The Linux comparison uses [llama.cpp CUDA kernels](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda).
NVIDIA's [matrix performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) explains the distinction between matrix tiling and memory-bound vector work.

The texture lookup table uses [hashbrown](https://docs.rs/hashbrown/0.17.1/hashbrown/) without its standard-library dependencies.

The shared-memory allocation follows [tinygrad’s native NVIDIA runtime](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_nv.py).

The fused recurrent mapping also uses [llama.cpp’s gated delta implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/gated_delta_net.cu) as a reference.
Its warp groups split independent value columns across key lanes. D3OS retains its existing row-major state layout.

The collective annotations follow the [Rust assembly options](https://doc.rust-lang.org/reference/inline-assembly.html#asm-options-supported-options-pure) and NVIDIA's [shuffle participation rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync).

The BF16 precision argument follows NVIDIA's [alternate floating-point format definitions](https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats).
The cuTile compiler emits TF32 conversions and selects register layouts for the maintained implementation.
The independent Rust split-reduction design follows the parallel reduction approach described in NVIDIA's [CUTLASS GEMM guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html).

The matched reference uses the pinned [llama.cpp C API](https://github.com/ggml-org/llama.cpp/blob/b11827/include/llama.h).
The weight audit follows its [Qwen conversion rules](https://github.com/ggml-org/llama.cpp/blob/b11827/conversion/qwen.py).

The tiled attention design uses the running softmax normalization described in [FlashAttention](https://arxiv.org/abs/2205.14135).
This Rust kernel retains FP32 operands and does not reuse the Linux library's attention kernels.


## Scheduler inbox saturation during compilation

The first interpolation validation stopped during inference-module compilation, before any interpolation kernel ran.
A cross-core broadcast filled the scheduler's 100-slot inbox and panicked with `TrySendError::Full`.
The thread-transfer path also discarded a runnable thread if its bounded send failed.
Retrying a full inbox can deadlock because producers can hold scheduler locks with interrupts disabled.

Each core now owns one growable FIFO for thread transfers and control messages.
An IRQ-save lock protects insertion and removal.
The consumer releases this lock before it dispatches a message.
This preserves insertion order and permits handlers to enqueue follow-up commands.
The shared FIFO keeps control messages behind earlier thread transfers.
The initial allocation remains 100 entries. Growth uses the existing IRQ-save allocator and retains normal kernel OOM behavior.
Neither the inbox lock nor the allocator calls the scheduler while held.
The obsolete channel receiver bank and sender interface are removed.

Host checks use the production FIFO and lock, with thread-local interrupt state in place of privileged instructions.
They check 12,288 queued thread/control messages, retained ownership, nested interrupt state, and requeue after removal.
Four concurrent producers also deliver 65,536 messages with exact per-producer order and no missing or duplicate messages.
CI runs these checks before the OS build.

The rebuilt D3OS kernel passes ten thread/vector/cleanup runs with four KVM CPUs.
It also passes 30 sleeping-worker process-exit cycles with four TCG CPUs.
These regressions overlap GPU validation and do not establish a performance change.
The original panic remains under `target/nvidia/burn-backend-validation/interpolate-first/`.
The regression logs use the `target/interpolate-inbox-` prefix.

```sh
rustc --edition=2024 --test tools/kernel/test-scheduler-inbox.rs -o target/test-scheduler-inbox
target/test-scheduler-inbox
python3 tools/kernel/test-linux-threads.py --accel kvm --cpus 4 --repeat 10
python3 tools/kernel/test-process-exit.py --accel tcg --cpus 4 --repeat 30 --sleeping-workers --timeout 240
```

## Experimental Q8_1 prefill on the 9B model

The optional Q8_1 mode reduces activation precision for the 9B Q8_0 FFN projections during batched prefill.
The default accurate mode, attention projections, and single-token decode retain their existing arithmetic.
The [Qwen guide](qwen.howto.md#experimental-q8_1-prefill-for-9b) describes the API and benchmark flag.

The activation packer follows the pinned [llama.cpp MMQ D4 layout](https://github.com/ggml-org/llama.cpp/blob/b11827/ggml/src/ggml-cuda/quantize.cu).
Each transposed block contains four FP32 scales and 128 signed bytes.
The Rust packer matches the actual pinned CUDA template and an independent CPU implementation byte-for-byte on the measured cases.
Sampled projection outputs also match the quantized CPU oracle bit-for-bit.
The matrix kernel uses signed INT8 MMA and FP32 accumulation.
Gate and up projections share one activation conversion.
Packed weight bytes and the single resident weight allocation remain unchanged.

On October 28, 2026, paired Native runs used accurate, Q8_1, Q8_1, accurate order on the RTX 4090 Laptop GPU.
Each mode contributed four ordinary warm samples across two VMs.
The fourth pass in each VM collected device timings separately.
The workload used 37 prompt tokens, 128 teacher-forced decode steps, a 2048-token context, and FP32 KV storage.

| Native mode | Warm prefill | Decode |
| --- | ---: | ---: |
| Accurate | 121 ms | 52.75 tokens/s |
| Q8_1 prefill | 91.5 ms | 52.79 tokens/s |
| Linux llama.cpp b11827 | 43.30 ms | 58.18 tokens/s |

The Linux run used the same model, prompt, decode inputs, context, and KV format.
Its two ordinary warm samples exclude the first pass.
The arithmetic differs between modes, so this table does not establish FP32 equivalence.

The quantized Native mode used 24.4% less prefill time than accurate Native mode.
Decode throughput remained effectively unchanged.
Both modes retained 9,516,859,392 bytes of model weights and transferred no activation tensors between host and GPU during inference.
Cold model loading averaged 6912 ms and 6878 ms respectively, with identical import paths.
Automatic GPU clocks and four warm samples do not establish a statistical guarantee.

### Accuracy scope

All 32,033,280 logit comparisons pass the original FP32 policy in accurate mode.
In Q8_1 mode, 16,776,142 values exceed that policy.
The benchmark retains this reference failure and does not emit its numerical success marker.
Timing records precede the reference check.

The Q8_1 run has FP32 logit RMSE of 0.00736 and maximum absolute error of 0.295 on this fixture.
The matched Linux llama.cpp run has RMSE of 0.04668 and maximum absolute error of 0.518 against the same CPU reference.
Both Native modes produce the same 128 teacher-forced token decisions, including the existing row-102 difference from Linux.
These measurements do not qualify other prompts or autonomous generation trajectories.
The mode remains experimental and does not replace the accurate default.

The combined module contains 271 entries.
All 269 prior inference machine-text sections and both smaller cubins remain byte-exact.
Math certificate rules remain unchanged.
This checkpoint grants no whole-Burn, DOOM, general quality, or cross-OS performance-parity claim.
