# Burn backend coverage

The native D3OS backend does not yet match the full Burn Linux CUDA backend.
This report separates source inventory, device declarations, and actual GPU conformance.
The reference uses Burn 0.22.1, CubeCL 0.11.3, and the same physical RTX 4090 Laptop GPU.
The [migration goal](burn-migration.md#primary-goal-linux-backend-api-parity) defines acceptance.

## Source inventory

The Rust audit tool parses the pinned backend traits, their defaults, and selected implementations with `syn`.
It includes `BackendTypes`, `Backend`, `AutodiffBackend`, `DeviceOps`, `Device`, and `TensorMetadata`.
It also includes float, integer, boolean, quantized, module, activation, transaction, and distributed operations.
Each method records its signature, source location, conditional compilation attributes, and implementation or inherited default.
Associated types and source SHA-256 hashes accompany the report.

The current inventory contains 484 methods: 267 native overrides and 217 inherited defaults.
All required methods have declarations or implementations. This count does not establish runtime support.
The report also records 106 native helper bodies for manual review of dtype and shape restrictions.

The tool follows syntactic associated backend calls and expands the pinned `dequant_op_quant!` and `dequant_op_flow!` macros.
It identifies 38 methods with possible paths to explicit declines or panic declarations.
This number is not an unsupported-operation count. Some paths represent expected invalid-input errors or conditional restrictions.

The analysis does not resolve method types, evaluate conditional compilation, or expand other macros and free helper functions.
Fusion and autodiff wrappers do not prove that their inner kernels support an operation.
Every method therefore retains an unverified runtime status. The report always declares full parity false.

## Declared device capabilities

The shared capability probe queries Burn's public dtype usage API on each GPU backend.
It emits 226 cases: 16 ordinary dtypes and 210 representative quantization schemes.
These flags describe backend declarations. They do not prove execution for every operation or shape.

Linux CUDA declares ordinary storage support missing from D3OS for these 12 formats:

`F64`, `Flex32`, `F16`, `BF16`, `I64`, `I16`, `I8`, `U64`, `U32`, `U16`, `U8`, and `Bool(U8)`.

Linux also declares arithmetic support for these formats and `I32`.
D3OS already implements many I32 operations, but its incomplete arithmetic support remains an explicit capability gap.
The missing accelerated declarations cover `F16`, `BF16`, `I32`, `I8`, `U8`, and `Bool(U8)`.
Existing model kernels use Tensor Cores, but that does not establish general support for these public dtypes.

All sampled quantized storage declarations available on Linux are available on D3OS.
However, 110 sampled scheme variants lack Linux's arithmetic declaration, and 20 lack its accelerated declaration.
These counts describe scheme variants, not distinct operations.
Resident dequantization and requantization can implement some operations without an accelerated quantized kernel.

## Native erf and GELU validation

The native FP32 erf implementation uses piecewise rational approximations adapted from FreeBSD msun and Rust libm 0.2.16.
The kernel source retains the original permission notice.
Native compute protocol 24 added the erf opcode and the `burn_erf` entry.
That inference bundle contains 188 entries. All 187 previous instruction and constant sections remain byte-identical.

The public Burn checks cover 2,656 erf inputs, exponent ranges, approximation boundaries, signed zeros, infinities, and NaNs.
They also cover shared inputs, transposed and reversed views, empty tensors, erf gradients, and GELU forward and backward operations.
The ordinary native math suite retains its existing eight-ULP bound and includes 2,816 IEEE edge values per unary operation.

| Result on the physical GPU | D3OS native | Linux CUDA |
| --- | --- | --- |
| Erf cases | 2,656 | 2,656 |
| Strict eight-ULP erf check | Pass | Fail |
| Largest absolute erf error | 5.9604645e-8 | 2.9802322e-7 |
| Largest erf error in ULP | 1 | Dominated by cancellation near zero |
| GELU forward and backward | Pass | Pass |

The Linux CubeCL erf polyfill uses an approximation with cancellation near zero.
Its separate diagnostic run reports 1,036 strict mismatches and eight special-value mismatches.
The failed strict run remains part of the evidence. A successful diagnostic run does not replace strict conformance.
Both GPU backends use the pinned Burn default GELU backward approximation as their reference.
The explicit Flex CPU check uses its analytic GELU derivative instead.

D3OS compiles all portable modules inside the VM. Each generated module matches its host audit hash.
The full resident operator and quantization suites also pass.
These runs establish correctness for the checked cases. They contain no execution-performance or compile-time parity measurement.

## Native integer division and remainder

Protocol 25 adds exact I32 division and floor remainder to the public Burn backend.
Both tensor and scalar operations support broadcasting and signed views.
Division truncates toward zero. A nonzero remainder has the divisor's sign and a smaller magnitude.
The kernels use unsigned integer magnitudes, including the exact magnitude of `I32::MIN`.
The source retains integer division semantics instead of a floating-point quotient.
The NVIDIA compiler lowers division with a reciprocal estimate and integer correction, which preserves exact full-range results.

The portable checks exclude zero divisors and `I32::MIN / -1` or `I32::MIN % -1`.
Burn does not guarantee portable behavior for these remainder inputs.
Native division by zero returns -1, and native remainder by zero returns the dividend.
Native `I32::MIN / -1` wraps to `I32::MIN`, and the corresponding remainder is zero.
These native extensions do not promise the same result on Linux.

The shared conformance checks use independent I64 quotient and floor-modulo references.
They cover full-range I32 values, scalar operands, broadcasting, reversed transposes, empty outputs, and shared input preservation.
The low-level native suite also checks exceptional inputs, resident execution, partial tiles, and neighboring guards.
Both Linux CUDA and D3OS pass the shared public checks on the physical GPU.
D3OS also passes the complete operator suite after in-VM compilation, with all module hashes matching their host audits.
That bundle contains 190 entries. All 188 previous instruction and constant sections remain byte-identical.
The I32 arithmetic capability declaration remains incomplete pending further operation and precision checks.

## Native integer matrix products

Protocol 26 adds public I32 matrix multiplication with exact wrapping products and accumulation.
The operation uses the existing matrix path for batch broadcasting, signed views, offset views, and shared storage.
Noncontiguous matrix operands require resident GPU copies before multiplication.
Empty output shapes perform no element work. A zero inner dimension produces initialized zeros.

The I32 kernel uses integer SIMD operations. Existing FP32 model kernels retain their Tensor Core implementations.
The kernel writes every output element and rejects source/output overlap.
Driver validation bounds all three buffers and rejects malformed dimensions before dispatch.
That portable bundle contains 191 entries. All 190 previous instruction and constant sections remain byte-identical.

The shared reference accumulates exact products in I128 and then converts each result to I32.
It covers full-range values, overflow, partial tiles, transposes, reversed axes, offset views, broadcast batches, and shared inputs.
Seven ordinary shape groups and one batch group form the complete eight-group check.
Five groups have nonempty matrix dimensions.

Linux CUDA passes all five nonempty groups on the physical GPU.
Its separate all-shape run fails at `[16,0] @ [0,5]`: the first output is 465362048 instead of zero.
The input readbacks pass before multiplication. The failure therefore concerns the product, not fixture upload.
The failed all-shape run remains separate evidence. The nonempty result does not establish empty-shape support.

D3OS passes all eight groups, including zero K and empty outputs, on the same physical GPU.
Its additional native checks pass neighboring guards, invalid bounds, signed broadcast batches, and resident execution without tensor transfers.
All modules compiled inside D3OS match their host audit hashes. The complete existing operator suite also passes.
This validation adds no performance measurement or claim of full backend parity.

## Native cumulative operations

Protocol 27 adds inclusive cumulative sum, product, minimum, and maximum for FP32 and I32 tensors.
The output retains the input shape. Signed and broadcast strides remain on the GPU.
Empty tensors return without a GPU launch, and an invalid axis returns an error even for an empty tensor.
Each output allocation has a complete writer. Driver validation rejects overlapping input/output storage and invalid address ranges.

The kernels use cuTile parallel scans within 128-element chunks.
The final prefix of each chunk supplies the carry for the next chunk.
Each independent scan row runs in one block. The implementation performs linear work along the scan dimension.
The current launch contract limits scans to 33,554,431 independent rows, in addition to the existing signed 32-bit layout limits.
The pinned Linux CubeCL source instead reads each earlier element for each output, with quadratic work along that dimension.
These algorithm descriptions do not establish a measured performance difference.

Integer sums and products wrap exactly at 32 bits. Integer minimum and maximum use signed comparisons.
Floating minimum and maximum preserve signed zeros and propagate the first NaN payload in the native raw-bit checks.
Floating sums and products can differ from sequential accumulation because parallel scans change the operation order.
The shared reference uses F64 accumulation for ordinary finite values and checks nonfinite classifications separately.
The sum tolerance is `2e-6 * (1 + prefix L1 norm)`. The product tolerance is `2e-5 * (1 + absolute reference)`.
Finite extrema have zero tolerance. The native IEEE checks also require exact stored bits.

The shared checks cover lengths 1, 127, 128, 129, 257, and 531 across every axis of signed views.
They include broadcast storage, empty shapes, shared inputs, subnormal values, infinities, and NaNs.
Independent I128 references check integer wrapping.
Weighted gradient checks cover all four floating operations, including repeated extrema and one or multiple zeros in cumulative products.
Linux CUDA, D3OS native GPU, and the explicit Flex CPU reference pass these public checks.
D3OS also passes the raw-bit IEEE, neighboring guard, invalid-bound, and resident-execution checks.
Empty native scans enqueue no GPU launch. All guest-compiled module hashes match their host audits.
The complete existing operator suite passes with the expanded bundle.

The cumulative-operation bundle contains 199 entries. All 191 previous instruction and constant sections remain byte-identical.
All eight new entries pass the host spill audit.

## Native sorting and top-k

Protocol 28 adds resident FP32 and I32 sorting, sorting with indices, argsort, and top-k.
Ascending and descending sorts use FP32 total ordering, including signed zeros and NaN payloads.
Equal keys retain their original axis order. Returned indices use I32 storage.
Native floating top-k places NaNs first and treats both signed zeros as equal.
Original axis order resolves top-k ties. This policy remains distinct from the total ordering used by full sorts.

Each merge pass writes every output element, including partial tiles and unmatched tails.
Gather-based merge partitions keep output addresses independent of index payloads.
The first pass reads signed or broadcast views directly and initializes original axis indices.

Subsequent passes alternate between two allocations, each containing values followed by indices.
The two working allocations contain four words per input element in total, excluding the input.
A top-k result slices the final allocation through tensor metadata without a host bounds readback.
This implementation sorts the complete axis even when k is small. Specialized partial selection remains a performance opportunity.

Command recording groups the passes and retains queued buffers through completion.
Empty results enqueue no GPU launch. Invalid axes remain errors for empty tensors.
Top-k accepts k equal to the axis length. Burn's public argtopk requires k smaller than that length.
The paired allocation limits sorting to 1,073,741,823 logical elements, within the existing rank and layout limits.
Other ordinary dtypes and index dtypes remain unresolved.

The shared checks cover lengths 1, 2, 3, 127, 128, 129, 257, and 1025 across all axes of signed views.
They check exact stored bits, index permutations, broadcast storage, empty shapes, and retained input values.
Six independent gradient references cover both directions of sorting and top-k, through values-only and indexed APIs.
D3OS passes the IEEE and finite-input checks on the physical GPU.
Its raw checks also pass all ten modes, neighboring guards, invalid bounds, complete writers, and queued buffer lifetime.
The resident multi-pass checks require exactly two allocations and no tensor upload, download, or separate clear.

The complete existing operator suite passes, and all guest-compiled modules match their host audit hashes.

The pinned Linux backend uses a host fallback for sort and argsort, but runs top-k and its gradients on the GPU.
Its IEEE run fails: top-k returns the most negative finite FP32 value where the input contains negative infinity.
The exact output words are `0xff7fffff` instead of `0xff800000`.
That failed run remains evidence. D3OS does not copy the Linux sentinel result.

The first Linux finite-input run reached its 300-second deadline without a success marker.
The diagnostic ended with a guest out-of-memory kill of `burn-sort` at 583 seconds.
That run used 4 GiB of guest RAM and retained basic CubeCL compilation logs.
A retry with 12 GiB reached its 1,200-second deadline without completing the finite-input probe.
The finite Linux comparison therefore remains unverified. These runs contain no matched execution-performance measurement.

CPU checks and restricted comparisons do not establish complete GPU parity.

The current portable inference bundle contains 209 entries and 65 native kernel profiles.
All 199 earlier instruction and constant sections remain byte-identical. The ten new entries pass the host spill audit.
These checks establish correctness for the tested cases. They contain no performance or compilation-time parity measurement.

## Native vector products

The FP32 cross product uses resident slice views, products, subtraction, and concatenation.
Its vector axis must have length three in both inputs. Other axes support broadcasting and signed strides.
Command recording groups all component operations and keeps queued inputs alive through completion.
Empty outputs return without a GPU launch, after dtype, shape, and axis validation.
This composition reuses the existing portable kernels. Protocol 28 and its 209-entry bundle remain unchanged.

The shared forward reference covers all three axes, reversed transposes, broadcast inputs, retained tensors, and partial tiles.
Six independent gradient references cover both input derivatives on nonbroadcast tensors across all axes.
A separate infinity case checks the scalar vector path.
Broadcast gradient reduction remains unverified and is outside these six checks.
Both Linux CUDA and D3OS pass this public corpus, including empty outputs, on the same physical GPU.
Both VMs use 4 GiB of guest RAM.

D3OS also passes resident execution without tensor transfers, queued input lifetime, and invalid native input checks.
Empty native cross products enqueue no GPU launch.
The complete operator suite passes, and every guest-compiled module matches its host audit hash.

The explicit Flex CPU reference passes the nonempty corpus.
Its all-shape run panics in the pinned `burn-flex` binary operation when cross-product slices reference empty storage.
That failed CPU run remains evidence. CPU execution does not establish native GPU conformance.

## Remaining implementation areas

The unresolved work includes ordinary dtypes, complete integer arithmetic, matrix products for further dtypes, and additional floating-point operations.
Further sorting dtypes, compaction, non-additive updates, further convolution variants, deformable convolution, and interpolation also require coverage.
Broadcast cross-product gradients, graph capture, profiling, memory capabilities, records, and observable errors require matched Linux capability checks.
The current layout also restricts rank to eight and extents to signed 32-bit values.
The method inventory records defaults, but each composed path still needs dtype, shape, ownership, and execution checks.

## Reproduce the checks

Run the source inventory from the repository root:

```bash
cargo +1.98.1 run --locked --manifest-path tools/burn-backend-audit/Cargo.toml -- \
  . target/nvidia/burn-backend-audit/traits.json
```

Build the Linux probes with their default CUDA feature:

```bash
cd tools/mnist-benchmark
cargo build --release --locked --bin capabilities --bin erf --bin integer --bin integer_matrix --bin scan --bin sort --bin cross
cd ../..
```

Run the probes sequentially because both VMs use the same GPU:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-capabilities \
  --burn-capabilities tools/mnist-benchmark/target/release/capabilities
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-erf \
  --burn-erf tools/mnist-benchmark/target/release/erf
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-integer \
  --burn-integer tools/mnist-benchmark/target/release/integer
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-integer-matmul \
  --burn-integer-matmul tools/mnist-benchmark/target/release/integer_matrix
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-scan \
  --burn-scan tools/mnist-benchmark/target/release/scan
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-sort \
  --burn-sort tools/mnist-benchmark/target/release/sort
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-cross \
  --burn-cross tools/mnist-benchmark/target/release/cross
```

If the all-shape matrix check fails, run the nonempty comparison in a separate directory:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-integer-matmul-nonempty \
  --burn-integer-matmul tools/mnist-benchmark/target/release/integer_matrix \
  --burn-integer-matmul-nonempty
```

If the strict erf check fails, use a separate directory for its numerical diagnostic:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-erf-profile \
  --burn-erf tools/mnist-benchmark/target/release/erf --burn-erf-profile
```

If the IEEE sorting check fails, retain its evidence and run the finite corpus in a separate directory:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-sort-finite \
  --burn-sort tools/mnist-benchmark/target/release/sort --burn-sort-finite
```

For a slow Linux probe, add `--burn-sort-compile-log` and an explicit `--timeout 600` in a fresh directory.
This enables basic CubeCL compilation diagnostics. Its elapsed time is not an execution-performance measurement.

Run the D3OS GPU suite with the matching production image and portable kernel bundle:

```bash
python3 tools/nvidia/run-qemu.py --firmware target/nvidia/firmware \
  --kernels target/nvidia/kernels --memory-mib 4096 \
  --boot-check --inference gpu-self-test --timeout 300
```

The retained native evidence lives under `target/nvidia/burn-backend-validation/erf-final/`.
The Linux evidence lives under `target/nvidia/burn-backend-audit/linux-capabilities-final/`, `linux-erf/`, and `linux-erf-profile/`.
The audit directory contains `traits-final.json`, `capability-comparison.json`, and `erf-comparison.json`.
The integer evidence lives under `target/nvidia/burn-backend-validation/integer-final/` and `target/nvidia/burn-backend-audit/linux-integer/`.
The audit directory also contains the updated `traits-integer.json` and `integer-comparison.json`.
The matrix evidence lives under `target/nvidia/burn-backend-validation/i32-matrix-final/`.
Its Linux counterparts are `target/nvidia/burn-backend-audit/linux-i32-matrix-all/` and `linux-i32-matrix-nonempty/`.
The audit directory contains `traits-i32-matrix.json` and `i32-matrix-comparison.json`.
The cumulative evidence lives under `target/nvidia/burn-backend-validation/scans-first/` and `target/nvidia/burn-backend-audit/linux-scan-final/`.
The audit directory contains `traits-scan.json` and `scan-comparison.json`.
The sorting evidence lives under `target/nvidia/burn-backend-validation/sorts-final/`.
Its Linux counterparts are `target/nvidia/burn-backend-audit/linux-sort-ieee-diagnostic/`, `linux-sort-finite-diagnostic/`, and `linux-sort-finite-12g/`.
The audit directory contains `traits-sort-final.json` and `sort-comparison.json`.
The vector-product evidence lives under `target/nvidia/burn-backend-validation/cross-final/` and `target/nvidia/burn-backend-audit/linux-cross/`.
The audit directory contains `traits-cross.json`, `cross-comparison.json`, and the retained Flex CPU failure.
Manifests record the shared validation source hashes, executable hashes, and execution backend.
These generated local artifacts are not Git-tracked. The probes and audit tool are tracked for reproduction.
