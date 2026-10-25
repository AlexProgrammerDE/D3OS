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

The current inventory contains 484 methods: 277 native overrides and 207 inherited defaults.
All required methods have declarations or implementations. This count does not establish runtime support.
The report also records 134 native helper bodies for manual review of dtype and shape restrictions.

The tool follows syntactic associated backend calls and expands the pinned `dequant_op_quant!` and `dequant_op_flow!` macros.
It identifies 8 methods with possible paths to explicit declines or panic declarations.
This number is not an unsupported-operation count. Some paths represent expected invalid-input errors or conditional restrictions.

The analysis does not resolve method types, evaluate conditional compilation, or expand other macros and free helper functions.
Fusion and autodiff wrappers do not prove that their inner kernels support an operation.
Every method therefore retains an unverified runtime status. The report always declares full parity false.

## Declared device capabilities

The shared capability probe queries Burn's public dtype usage API on each GPU backend.
It emits 226 cases: 16 ordinary dtypes and 210 representative quantization schemes.
These flags describe backend declarations. They do not prove execution for every operation or shape.

Linux CUDA declares storage support missing from D3OS for these 12 formats:

`F64`, `Flex32`, `F16`, `BF16`, `I64`, `I16`, `I8`, `U64`, `U32`, `U16`, `U8`, and `Bool(U8)`.

Burn's `Storage` flag requires conversion to and from other supported dtypes as well as physical storage.
Ordinary numeric casts now execute on the GPU, as described below.
Quantized conversions, further index formats, and arithmetic remain incomplete for the new formats.
The usage flags stay disabled until the complete conversion contract passes.
Both backends decline the separate `Bool(Native)` declaration.

Linux also declares arithmetic support for these formats and `I32`.
D3OS already implements many I32 operations, but its incomplete arithmetic support remains an explicit capability gap.
The missing accelerated declarations cover `F16`, `BF16`, `I32`, `I8`, `U8`, and `Bool(U8)`.
Existing model kernels use Tensor Cores, but that does not establish general support for these public dtypes.

All sampled quantized storage declarations available on Linux are available on D3OS.
However, 110 sampled scheme variants lack Linux's arithmetic declaration, and 20 lack its accelerated declaration.
These counts describe scheme variants, not distinct operations.
Resident dequantization and requantization can implement some operations without an accelerated quantized kernel.

## Packed ordinary dtype storage

Protocol 31 introduced packed storage copy and fill kernels with 214 inference entries and 67 compute profile slots.
All 212 earlier instruction entries and their constant sections remain byte-identical.
The current protocol and bundle include the conversions described below.

| Physical bytes per value | Ordinary formats |
| --- | --- |
| 1 | I8, U8, Bool(U8) |
| 2 | F16, BF16, I16, U16 |
| 4 | F32, Flex32, I32, U32, Bool(U32) |
| 8 | F64, I64, U64 |

Layouts retain element-based offsets and signed strides.
The driver converts validated element extents to packed word extents before checking GPU address ranges.
Each copy or fill lane writes a complete output word and initializes unused tail bytes to zero.
The kernels preserve stored bits without a floating-point conversion.
The existing four-byte copy and fill paths remain in use.

The shared corpus checks six lengths around packed-word and tile boundaries for each format.
Checks include reversed transposes, stepped slices, offset broadcasts, overlapping windows, empty shapes, input preservation, and record reloads.
Fills cover signed zeros, exact half values, signed integers, and U64 values beyond the signed range.
F16 and BF16 checks preserve all 65,536 bit codes of each format through a transposed GPU copy.

Burn's high-level boolean creation normalizes storage to the device default, including uploads with an explicit dtype option.
The shared probe therefore uses public backend operations to check both boolean widths directly.
Floating-point and integer checks use public Burn tensors.

Linux CUDA passes the complete 14-format corpus with Flex32 excluded explicitly.
Its pinned backend rejects directly tagged Flex32 uploads despite declaring storage support.
A separate attempt to initialize Flex32 through an FP32 assignment fails with `DTypeMismatch` in Burn fusion.
Both diagnostics remain separate from the passing Linux result.
The original boolean fixture failures remain recorded because they incorrectly expected high-level creation to retain the requested width.
The explicit Flex CPU diagnostic also fails for a U64 fill beyond I64 range in its `to_i64().unwrap()` conversion.
CPU behavior does not replace the GPU comparison.

D3OS passes the complete 15-format corpus, including direct Flex32 uploads.
Its full operator and quantization regression suites also pass after compilation inside the VM.
All three generated modules match their host audit hashes.
Native validation also checks neighboring guards, nonzero source padding, zero output padding, and queued source lifetimes.
Its service counters check resident execution and output allocation without a separate clear.
The original storage checkpoint rejected F16, BF16, F64, I64, and U64 gathers.
The movement checkpoint replaces that obsolete rejection with exact-byte positive checks.
Arithmetic checks for these formats still reject unsupported work without a GPU launch.

These additions provide storage, fills, views, materialization, and tensor record reloads.
General arithmetic, matrix products, and accelerated execution for the additional formats remain incomplete.
Ordinary numeric casts now pass the checks in the next section.
This checkpoint contains no execution-performance or compile-time parity measurement.

## Ordinary numeric conversions

Protocol 32 introduced one resident conversion kernel for all 15 ordinary formats.
That version uses a matching SDK, driver, application image, and 215-entry portable inference bundle.
The current protocol also includes the remainder operation described below.
The compute profile contains 68 slots.
All 214 earlier instruction entries and their constant sections remain byte-identical.
The existing FP32/I32 conversion paths remain in use.

Each lane writes a complete output word, including zero padding after the final narrow value.
Source offsets and signed strides use logical elements, while the driver checks packed physical bounds for both buffers.
Conversions need no host activation transfer or separate output clear.
Identity conversions retain shared storage and exact bits.
The F32/Flex32 alias also retains shared storage because both formats use the same physical representation.
Other conversions produce independent storage and preserve the source.

Floating-point conversions use nearest-even rounding.
Integer narrowing retains the low destination bits after signed source extension.
Native float-to-integer conversion truncates toward zero, clamps overflow, and maps NaN to zero.
The Linux comparison uses finite inputs within the destination range because its overflow behavior is not portable.
Finite results require exact bits. Conversions between different floating-point formats compare NaN classification instead of payload bits.

The independent reference rounds binary significands with integer arithmetic.
It includes U64 values whose correct result differs from an intermediate FP32 or FP64 conversion.
The native raw suite checks 225 format pairs, scalar results, empty casts, neighboring guards, zero padding, and queued source lifetimes.
Service counters check resident execution and allocation through a single full writer.
Boolean cases include zero, one, and all-ones records.
Boolean-to-I32 conversion now produces numeric zero or one instead of reinterpreting the original bits.
Its output owns independent storage, so the previous allocation-free alias check now checks full-writer and ownership behavior.

The shared suite covers 195 Burn API pairs on D3OS and 168 on Linux CUDA with Flex32 excluded explicitly.
Numeric cases use public Burn tensors. Boolean cases use public backend operations to request both physical storage widths.
Checks include reversed transposes, empty shapes, and eight native or six Linux cast-gradient cases.
Every F16 and BF16 bit code converts to each floating-point format in the checked backend's domain.
BF16 checks also cover 195,840 FP64 boundary inputs and 99,072 integer boundary inputs across signed and unsigned 16/32/64-bit formats.
These inputs include both sides of each midpoint, ties, both signs, subnormals, and overflow boundaries where applicable.

The first native run detected double rounding in the pinned NVIDIA compiler's FP64-to-BF16 lowering.
It returned `0x3f80` where the independent reference and Linux returned `0x3f81`.
The instruction dump shows FP64-to-FP32 conversion followed by FP32-to-BF16 conversion.
The integer-to-BF16 paths also used an intermediate FP32 value.
The replacement rounds directly from FP64 bits or the original integer's leading, guard, and sticky bits.
The [Tile IR 13.4 conversion rules](https://docs.nvidia.com/cuda/tile-ir/13.4/sections/operations.html) define the rounding and saturating operations used by the other paths.

The pinned Linux F32-to-Flex32 cast returns F32 metadata, so the full Flex32 corpus remains a separate failed diagnostic.
Another retained failure came from the gradient fixture's constructor, before casting.
Burn autodiff inherits `float_full`, which uses `TensorData::full_dtype` and tags a Flex32 fill as F32.
The corrected cast fixture creates explicitly tagged leaves and checks their source dtype first.
This avoids treating a constructor result as proof about cast gradients.

D3OS passes the full native corpus and existing operator regressions after compilation inside the VM.
All three generated modules match their host audit hashes.
Linux CUDA passes the shared corpus with Flex32 excluded explicitly.

These checks cover ordinary casts, not full dtype arithmetic, quantized conversion variants, or accelerated matrix products for the additional formats.
The capability declarations remain conservative. This checkpoint contains no matched execution-performance or compile-time parity measurement.

## Packed ordinary predicates

Protocol 39 adds resident predicates for all 13 ordinary numeric formats and both boolean storage widths.
The matching driver, SDK, image, and portable bundle contain 240 inference entries and 75 compute profiles.
The new complete writer packs canonical U8 or U32 boolean results and clears unused tail bytes.
Logical offsets and signed strides retain element units. The driver checks physical word bounds for each input and output.
The 32-word parameter layout retains both words of a 64-bit scalar without changing the request structure.

Numeric comparisons cover equality, inequality, greater, greater-or-equal, less, and less-or-equal through Burn's public backend API.
Scalar comparisons use the source dtype before encoding their bits.
Boolean equality, inequality, AND, OR, XOR, and NOT preserve the requested storage width.
Floating-point classification covers NaNs, infinities, and the public composition for finite values.

The kernel compares unsigned bit keys for floats and full-width integer representations.
It normalizes signed zero and excludes unordered NaN operands from ordered comparisons and equality.
This avoids rounding adjacent I64/U64 values through floating point and preserves subnormal comparisons.
FP32/I32 comparisons and U32 boolean logic retain the existing one-word fast paths.
FP32 classification also uses the existing specialized entries.

The independent reference decodes host storage types and uses their ordinary comparisons and classification.
Its unit checks include adjacent 64-bit integers beyond FP64 precision, signed boundaries, NaNs, infinities, signed zeros, and subnormals.
The shared corpus includes scalar operands, Cartesian edge pairs, empty tensors, reversed/transposed/offset views, and broadcast operands.
It checks both boolean widths at tail lengths 1, 127, 128, 129, and 257.
Both half formats also cover every one of their 65,536 storage patterns.
The raw native corpus checks 382 direct dispatches, 382 tensor cases, and 431 rejected contracts.
It includes neighboring guards, zero tails, complete writers, resident counters, queued lifetimes, and ranks zero and eight.

D3OS passes all 6,898 public cases across 13 numeric formats and both boolean widths.
It also passes the raw/native contract cases, every rejection check, and the existing operator and quantization regressions.
All three modules compiled inside D3OS match their host audit hashes.
The new entry passes the spill audit without local loads or stores.
All 239 earlier instruction entries and their constant sections remain byte-identical.

Linux CUDA passes 6,316 public cases with Flex32 excluded, including all half patterns and the complete IEEE corpus.
Its full corpus fails the direct Flex32 upload in the pinned backend.
The initial non-Flex32 GPU run passed, but the first harness expected too few cases because it omitted empty broadcasts.
The corrected harness and separate final run pass. Both records remain available.

The stored-value movement checkpoint extends masking, compaction, concatenation, and view assignment to these packed boolean results.
Indexed assignment and additional index dtypes remain separate work.
Dtype capability declarations remain conservative. These checks do not establish execution-performance or compile-time parity.

## Packed stored-value movement

Protocol 40 extends gather/select, masked selection/fill, view assignment, concatenation, and padding across all 15 ordinary storage formats.
The matching driver, SDK, image, and portable bundle contain 244 inference entries and 79 compute profiles.
All 240 previous instruction entries and their constant sections remain byte-identical.
The four new entries pass the spill audit without local loads or stores.

Gathers retain I32 index values, signed source/index strides, and exact stored bits.
Narrow lanes own complete output words, including zero tail padding.
A 64-bit lane loads each selected value once and writes both output words.
Only the four-byte bounds flag returns to the host. Invalid device indices remain recoverable errors.

Partial assignment retains the existing proof that logical destinations do not overlap.
Byte and half-word destinations can still share a physical word.
Raw-word compare-and-swap merges each field into initialized storage without losing neighboring updates.
Four-byte values retain the existing writer. Eight-byte values write their two private words directly.
Slice assignment preserves the original destination and overlapping source snapshots through a private GPU copy.
Concatenation starts from initialized packed storage. Padding starts from a typed exact fill.

Masked selection supports all three inputs' broadcasts and either U8 or U32 masks.
Expanded noncontiguous masks require a resident GPU copy.
Contiguous U8 masks use a word-base pointer and byte remainder, including offsets that do not align with a word.
Scalar fills retain both words of their destination-typed metadata. They do not truncate F64/I64/U64 values through FP32 or I32.
Full writers initialize every packed output byte, including the last word's unused bytes.

BoolU8 scatter/select OR uses word atomics with byte-field contributions, so duplicates and neighboring destinations remain safe.
False update values still require valid indices.
BoolU8 argwhere uses the existing resident I32 conversion and prefix scan, with one four-byte count readback.
True and false scalar masks produce shapes `[1, 0]` and `[0, 0]` without coordinate writes.
The existing four-byte fast paths remain in use.

`NativeTensor::pad` now accepts U64 raw fill bits. Existing U32 variables require `u64::from(fill)` at the call site.
Burn's public constant-padding value remains FP32 because `PadMode::Constant` defines that input type.
Burn exposes no separate boolean pad operation. Numeric constant, edge, and reflect padding compose the resident movement operations.

The independent shared reference manipulates opaque byte records rather than decoded numeric values.
Its complete corpus contains 4,793 public cases, 2,994 source checks, and 1,273,900 output values across all 15 formats.
The raw corpus contains 60 gather, 400 mask, 120 assignment, 50 boolean-scatter, and 14 queued-lifetime cases.
Native checks add 165 movement cases, eight boolean/compaction cases, 32 empty/error cases, and four scalar argwhere cases.
The checks cover NaN payloads, signed zero, wide integer bits, byte offsets, partial tails, ranks zero/eight, and source ownership.
D3OS passes the complete shared corpus and every raw/native check, including the four scalar argwhere cases.
All earlier operator and quantization regressions also pass.
All three modules compiled inside D3OS match their host audit hashes.

Linux's full corpus retains its unsupported Flex32 upload failure.
The separate corpus without Flex32 fails at empty boolean OR scatter because the pinned CUDA kernel divides by a zero dimension.
Skipping only empty boolean scatter reveals incorrect mixed-width boolean mask selection.
The pinned boolean wrapper passes the data storage type as the mask type.
A separate diagnostic restricts boolean where/fill to matching stores while retaining both mask widths for numeric tensors.
The isolated Linux diagnostic passes 4,328 cases, 2,792 source checks, and 1,147,836 output values.
It excludes Flex32 uploads, four empty boolean OR cases, and only mixed-store boolean where/fill operations.
Numeric where/fill still checks both mask stores. Mask selection retains mixed-width stores and all other empty cases.
The three failed diagnostics and passing subset use the same frozen reference source and Linux executable.
Every diagnostic remains separate from the complete D3OS corpus and its expected totals.

Wider ND reads, indexed assignment, additional index formats, and arithmetic updates remain outside this checkpoint.
General dtype arithmetic, full quantized contracts, graph capture, and other execution capabilities also remain incomplete.
These correctness checks do not establish execution-performance or compile-time parity.

## Inverse FP32 math

Protocol 34 adds `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, and `atan2` through Burn's public API.
The matching SDK, driver, application image, and 223-entry portable inference bundle must use this protocol together.
The compute profile retains 68 slots because these operations use elementwise dispatch.
All 216 earlier instruction entries and their constant sections remain byte-identical.

The circular functions use [NVIDIA Tile IR `atan2`](https://docs.nvidia.com/cuda/tile-ir/13.4/sections/operations.html#cuda-tile-atan2).
The `asin` and `acos` radicand uses `(1 - x) * (1 + x)` to retain accuracy near both endpoints.
The hyperbolic functions use bounded FP32 range reductions adapted from Rust libm 0.2.16, with its MIT notice retained.
Tiny `asinh` and `atanh` inputs retain their original bits.
Large hyperbolic inputs use logarithmic forms that avoid intermediate squares.

A shared corrected `log1p` helper serves the new functions and the existing public `log1p` operation.
Its extracted implementation produces the same machine code as before.

Each unary operation receives 11,084 inputs across every FP32 exponent range, including random bit patterns and boundary neighbors.
The `atan2` corpus contains 11,280 pairs, including Cartesian combinations of signed zeros, subnormals, infinities, and NaNs.
A separate bounded corpus contains 512 inputs per unary operation and 548 `atan2` pairs.
Checks include reversed transposes, source preservation, broadcasting in both directions, and empty outputs.
Twelve analytic gradient cases cover all six unary functions and both `atan2` operands, including broadcast reductions.

Finite nonzero forward results have an eight-ULP limit against independent FP64 libm results rounded to FP32.
Zeros require their exact sign, infinities require exact bits, and NaNs require matching classification.
Gradients use independent FP64 derivatives away from singularities, with a relative tolerance of `2e-5` and absolute tolerance of `1e-6`.
The native common math suite also checks all six new unary entries against its 2,816 IEEE edge values and resident counters.
Native `atan2` checks cover seven guarded lengths, output allocation, resident execution, queued lifetimes, scalar rank zero, and empty inputs.
The first native run exposed an invalid-domain result in the pinned FP32 libm reference.
For input `0xc27fffff`, approximately `-64`, it returned a negative finite value instead of NaN.
The corrected reference enforces the real `acosh` domain and uses FP64 libm results rounded to FP32 for all seven operations.
The first failed run remains under `target/nvidia/burn-backend-validation/inverse-first/`.

A second native run passes both inverse corpora and all gradient cases, then reaches an obsolete Qwen error fixture.
Five error fixtures used `acos` to create an unsupported-operation error.
They now use out-of-range gathers, so future operation support cannot make their inputs valid.
Counters still require failed dependent operations to allocate no storage and launch no work.

The complete D3OS GPU suite passes after these fixture corrections.
Both inverse corpora and all 12 gradient cases pass against the corrected reference.
The existing operator, quantization, Qwen extension, and error-propagation checks also pass.
All three modules compiled inside D3OS match their host audit hashes.

Linux CUDA passes the full circular corpus and all 12 gradient cases.
Its `acosh` returns infinity for a large finite input whose reference is `45.054565`.
Its full `asinh` and `atanh` checks return zero for the smallest positive FP32 input.
The bounded `asinh` and `atanh` checks each differ by 13 ULP at their first mismatch.

The diagnostic mode checks every operation and reports each strict result independently.
Diagnostic completion does not establish strict conformance. Both Linux domains retain `strict_conformance=false`.

These checks contain no execution-performance or compile-time parity measurement.

## Native interpolation

Protocol 35 adds FP32 NCHW interpolation with nearest, nearest-exact, bilinear, bicubic, and Lanczos3 modes.
All five modes include native backward operations and both `align_corners` settings.
Nearest modes ignore `align_corners`, as in the pinned Linux backend.
The matching driver, SDK, application image, and portable bundle use 233 inference entries and 69 compute profiles.
All 223 earlier instruction entries and their constant sections remain byte-identical.

The forward operation resizes width, then height. The backward operation applies the transposes in reverse order.
Each backward lane gathers its contributing gradients without floating-point atomics or host activation transfers.
Both directions read signed and broadcast views and write complete dense outputs.
Identity axes retain shared storage. Empty batch or channel dimensions require no element launch or output allocation.
Spatial dimensions must remain nonzero.

Exact rational transforms use 64-bit coordinate arithmetic.
This avoids overflow and nearest-neighbor errors when coordinate products exceed 32 bits.
Bilinear and bicubic filters clamp boundary samples. Bicubic uses Keys coefficients with alpha `-0.75`.
Lanczos3 discards samples outside the input and normalizes the retained weights.

The independent reference constructs FP64 resampling matrices and their transposes.
The public corpus contains 100 forward cases, 80 gradient cases, 20 empty cases, and four exact nearest IEEE cases.
It covers signed views, transposed spatial axes, singletons, identity axes, upsampling, downsampling, and retained inputs.
Wide cases resize 65,537 elements to 65,539 and reverse that transformation.
The raw native suite adds 40 guarded mode/axis/direction/broadcast cases.
It also checks full writers, queued input lifetime, resident counters, invalid launch metadata, and empty operations.
Two host oracle tests anchor nearest mappings, singleton corner alignment, and known cubic forward/transpose coefficients.

The initial Lanczos3 export failed the spill audit because general sine lowering used local scratch memory.
Its source, portable input, machine code, and instruction dumps remain under `interpolate-sin-spill/`.
The replacement reflects its bounded angle into `[0, pi/2]` and evaluates a degree-13 sine polynomial with FMA.
A triple-angle identity supplies the Lanczos product. This change does not replace general sine operations.
All ten new entries pass the spill audit without local loads or stores.

Linux CUDA passes the shared compact corpus: 80 forward cases across all five modes and 32 nearest-mode gradient cases.
Its wide nearest diagnostic fails at output index 65,536, returning `-1.625` instead of `2.0`.
Its separate empty diagnostic panics in the CubeCL autotuner.
Its exact nearest IEEE diagnostic changes negative zero to positive zero.
Bilinear, bicubic, and Lanczos3 backward probes each report explicit unsupported-operation errors on the pinned Linux backend.
The additional D3OS backward modes extend that baseline.
These failed Linux diagnostics remain separate from the passing compact result.

The first D3OS run panicked during compiler startup when a scheduler inbox filled, before interpolation execution.
The [scheduler correction](nvidia-performance.md#scheduler-inbox-saturation-during-compilation) replaces that bounded delivery path.
D3OS passes the complete public interpolation corpus, all raw native checks, and the existing operator and quantization regressions.
All three modules compiled inside D3OS match their host audit hashes.
These correctness checks do not establish execution-performance or compile-time parity.

## Native 3D and transposed convolutions

Protocol 36 adds FP32 3D convolution and transposed 1D/2D/3D convolutions through public Burn APIs.
Their input and weight gradients use native implementations. Bias gradients use Burn's resident reduction defaults.
The matching driver, SDK, application image, and portable bundle contain 235 inference entries and 71 compute profiles.
All 233 earlier instruction entries and their constant sections remain byte-identical.

Two new kernels extract canonical NCDHW windows and apply the transpose of that extraction.
Matrix products reuse compensated TF32x3 Tensor Core kernels.
Each group combines batch and spatial positions into one matrix dimension.
The backward fold gathers overlapping contributions without floating-point atomics.
Intermediate columns, products, and gradients remain on the GPU.
The explicit column workspace must fit the existing signed 32-bit layout limits and available device memory.

The shared geometry supports grouped and depthwise convolutions, stride, dilation, and output padding.
Transposed output padding can exceed the stride. An explicit input-shape calculation avoids inverse rounding assumptions in that case.
The kernel computes spatial coordinates in 64 bits and clamps masked coordinates before signed view addressing.
The new geometry and view fields fit the existing 32-scalar request without changing its structure.
Normal 3D plans also accept asymmetric padding through backend operations.
Burn's high-level `conv3d` function rejects asymmetric padding before backend dispatch, so the shared public corpus uses symmetric padding.

The independent FP64 reference uses scalar convolution sums and analytic input, weight, and bias derivatives.
Two host tests anchor volume orientation, transposed output padding, and centered finite differences across every checked geometry.
Fourteen shape groups combine both view directions and optional bias into 56 forward and 140 gradient checks per numerical corpus.
The 1D transposed API uses the same native spatial algebra, including direct input and weight gradients.
The general FP32 corpus uses values outside TF32's exact representation.
A separate corpus uses dyadic inputs exactly representable in TF32.
Both use the same `3e-5 * (1 + abs(reference))` tolerance.
Another 56 cases check empty batches, with a separate set of 140 empty-batch gradient checks.
D3OS passes the empty-batch gradient corpus, including exact zero weight and bias gradients.

The raw native suite contains 56 guarded window/fold cases, including broadcast and negative-stride reads.
It checks source preservation, complete writers, resident counters, queued source lifetime, and invalid geometry rejection before launch.
Both new entries pass the spill audit without local loads or stores.

Linux CUDA passes 56 forward and 140 gradient checks with TF32-exact inputs, plus 56 empty forward cases.
Its empty-batch backward diagnostic returns `0.96774197` for a weight gradient whose independent reference is zero.
This failed diagnostic remains separate from the passing empty forward result.
Its general FP32 run fails at the first forward output, returning `0.2412498` instead of `0.24037458584038518`.
A scalar diagnostic reproduces that result after rounding operands to ten fraction bits.
FP16 rounding also reproduces it for these bounded operands, so this sample alone does not identify Linux's selected instruction type.
The strict failed run remains separate from the passing exact-input corpus.
These results do not establish strict general FP32 conformance on Linux.

D3OS passes both complete numerical corpora, all empty and guarded cases, and the existing operator and quantization regressions.
All three modules compiled inside D3OS match their host audit hashes.
These correctness checks do not establish execution-performance or compile-time parity.

## Native deformable convolution

Protocol 37 adds FP32 deformable 2D convolution and all five gradients through Burn's public module API.
The gradients cover the input, offsets, weights, optional modulation mask, and optional bias.
The matching driver, SDK, application image, and portable bundle contain 238 inference entries and 74 compute profiles.
All 235 earlier instruction entries and their constant sections remain byte-identical.

Three new kernels sample bilinear windows, compute offset/mask derivatives, and accumulate input derivatives.
Grouped compensated TF32x3 Tensor Core products implement forward projections and weight/column gradients.
Weight groups and offset groups remain independent, including partitions that cross each other.
Noncontiguous inputs require resident GPU copies. No activation tensor returns to the host during these operations.

Forward windows and paired offset/mask gradients use complete writers without a separate output clear.
The paired gradients share private storage through checked views.
Input gradients accumulate into initialized zero storage with raw-word compare-and-swap and explicit non-FTZ addition.
This avoids native floating-point atomics that can flush subnormals on Ada.
The accumulation order can vary across blocks, so ordinary gradients use a numerical tolerance.

Coordinates use 64-bit integer base arithmetic followed by FP32 offset addition and bilinear sampling.
The kernel sanitizes invalid coordinates before integer conversion and clamps masked pixel addresses.
Native nonfinite offsets contribute zero sampled values and derivatives.
Finite coordinate boundaries use the right derivative of the piecewise bilinear function, including the `-1` boundary.
Other nonfinite-input arithmetic and wider dtype conformance remain unverified.

The independent reference uses FP64 bilinear samples and analytic derivatives for every input kind.
Host tests anchor all five derivatives, zero modulation, and coordinate boundaries.
Centered finite differences check eight geometries, excluding offset discontinuities in the separate zero-offset fixture.
The corpus includes stride, dilation, depthwise and crossing groups, collisions, partial tiles, zero offsets, and out-of-range samples.
Optional masks, optional bias, and reversed views produce 64 forward cases and 256 gradient checks per numerical corpus.
Both general FP32 and TF32-exact input corpora use the same `4e-5 * (1 + abs(reference))` tolerance.
The empty-batch corpus also checks all required and optional gradients.

The raw suite checks 96 guarded primitive cases, full writers, source preservation, queued lifetime, resident counters, and rejected geometry.
It includes nonfinite offsets and an exact smallest-subnormal input-gradient check.
All three new entries pass the spill audit without local loads or stores.
The first export exposed a pointer-type inference error in the pinned cuTile compiler.
Explicit types on each pointer reshape/broadcast step correct that error. Its generated source and diagnostic remain archived separately.

The general Linux CUDA run fails its first forward result: `0.033498675` instead of `0.03331488615597683`.
The TF32-exact run passes that forward result, then fails an offset derivative: `0.38085938` instead of `-0.94482421875`.
The pinned coordinate-gradient kernel omits the kernel-tap displacement from its column base.
A scalar reproduction that uses tap zero returns exactly the observed value.
The source and reproduction remain separate evidence from the shared independent reference.
The empty Linux diagnostic returns a nonzero weight gradient where the independent reference requires zero.
D3OS passes both complete numerical corpora, the empty-batch gradient corpus, and every guarded primitive check.
All three guest-compiled modules match their host audit hashes. The existing operator and quantization regressions also pass.
Linux passes all 64 forward cases with TF32-exact inputs and all 64 empty forward cases.
Separate 1x1-filter and outside-sample probes each pass eight forwards and all 32 required/optional gradient checks.
Those limited passing gradient domains do not replace the failed complete corpus.
These checks do not establish execution-performance or compile-time parity.

## Native floating-point remainder

Protocol 33 adds FP32 tensor and scalar remainder through Burn's public API.
The matching SDK, driver, application image, and 216-entry portable inference bundle must use this protocol together.
The compute profile retains 68 slots because remainder uses the existing elementwise dispatch.

The GPU kernel uses NVIDIA's Tile IR `remf` operation for the truncated residue.
Burn's floor modulo then adjusts a nonzero residue whose sign differs from the divisor.
The correction uses non-FTZ nearest-even addition.
The [Tile IR remainder specification](https://docs.nvidia.com/cuda/tile-ir/13.4/sections/operations.html#cuda-tile-remf) defines the residue and special cases.
Floating-point rounding can produce a nonzero result equal to the divisor.

A zero residue retains the dividend's sign.
Zero divisors, infinite dividends, and NaN operands return NaN.
An infinite divisor follows the independent libm reduction and sign-correction reference.
NaN checks compare classification instead of payload bits.
The shared Linux checks permit either sign of zero. Native raw checks require the dividend's sign.

The complete corpus contains 53,248 input pairs across every FP32 exponent range, including 4,096 deterministic raw-bit pairs.
Public checks include 15 scalar divisors, reversed transposes, broadcasting in both directions, input preservation, and empty outputs.
Eight gradient checks cover both operands, broadcast reduction, signed views, and scalar remainder.
The analytical gradients use one for the dividend and negative floor quotient for the divisor, away from discontinuities.

Linux CUDA passes the ordinary finite corpus: 512 pairs, eight scalar divisors, and all eight gradient checks.
Its full IEEE diagnostic returns NaN for `0 % infinity`, where the libm-based reference returns zero.
A separate 40,790-pair finite diagnostic excludes nonfinite operands and zero divisors.
It fails at `f32::from_bits(1) % -3`, returning the positive dividend instead of the rounded negative divisor.
The pinned CubeCL C++ lowering computes `x - y * floor(x / y)`.
Its quotient underflows to negative zero in this case, which loses the floor correction.
Both failed diagnostics remain separate from the passing ordinary finite result.

D3OS passes the complete IEEE corpus and all eight gradient checks.
It also passes the identical finite corpus used by Linux CUDA.
Native checks cover guarded writes at seven lengths, including partial tiles and the full corpus.
Service counters confirm resident execution and one full-writer allocation without a separate clear.

The queue retains buffers after their source tensors leave scope.
Empty outputs and invalid inputs launch no GPU work.
Scalar rank-zero results retain their shape.
All three modules compiled inside D3OS match their host audit hashes.
The existing operator and quantization suites also pass.
All 215 earlier instruction entries and their constant sections remain byte-identical.
This checkpoint contains no matched execution-performance or compile-time parity measurement.

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

## Native integer powers

Protocol 38 adds exact I32 tensor powers and generic scalar powers through Burn's public API.
The matching portable bundle contains 239 inference entries and 74 compute profiles.
The new operation uses the existing typed-word layout and complete-writer contract.

The inherited Burn fallback converts integer bases to floating point before exponentiation.
That conversion can lose base precision, even when the exponent is one.
The native replacement uses unsigned multiplication and exponentiation by squaring, with modulo-2^32 results.
It interprets exponent bits as U32, matching the pinned Linux CUDA backend.
A negative I32 exponent therefore represents a large unsigned exponent, rather than a fractional reciprocal.
The result of zero raised to zero is one.
Burn retains its existing scalar shortcuts for exponents zero, one, and two.

The independent reference uses left-to-right binary exponentiation with I128 products and explicit modular reduction.
Host checks compare it with primitive wrapping powers and modular group identities.
The shared corpus checks 94 public cases, including signed views, broadcasting, rank eight, empty shapes, and retained source tensors.
It includes all pairs of 15 boundary bases and 17 exponent patterns.
The raw corpus checks ten guarded tensor/scalar cases at lengths 1, 127, 128, 129, and 257.
Separate checks cover a scalar result, queued source lifetimes, allocation through one complete writer, and resident execution.
Invalid aliases, invalid dtypes, incompatible shapes, and empty outputs enqueue no element work.

Linux CUDA passes all 94 public cases on the same physical GPU.
D3OS passes the same 94 public cases, all ten guarded cases, and the existing operator and quantization regressions.
All three modules compiled inside D3OS match their host audit hashes.
The new entry passes the spill audit without local loads or stores.
All 238 earlier instruction entries and their constant sections remain byte-identical.
These correctness checks do not establish execution-performance or compile-time parity.
The broader I32 arithmetic capability remains conservative until the remaining contracts pass.

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

The sorting bundle contains 209 entries and 65 native kernel profiles.
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

## Native dynamic selection

Protocol 29 adds GPU coordinate extraction for boolean masks, with I32 output indices.
Burn's inherited mask-selection methods now compose this operation with reshape and resident gather for FP32, I32, and boolean tensors.
The public quantized selection path also uses these GPU indices.
Nonzero coordinates retain logical row order, including reversed transposes and broadcast mask storage.

A resident integer prefix sum counts canonical boolean values.
Only its final four-byte count returns to the host to determine the output shape.
Noncontiguous masks require a resident copy before flattening.
Each selected output row finds its input position through bounded binary search of the prefix counts.
The coordinate kernel writes every output word, including when prefix contents are malformed.
Invalid prefix matches produce -1 coordinates. Prefix values never determine output addresses.

This contract permits a fresh output allocation without a separate clear.
The driver rejects overlapping source/output ranges and malformed dimensions before dispatch.
Queued prefix storage remains alive through completion of the coordinate kernel.
An empty input returns shape `[0, rank]` without GPU work or a count readback.
An all-false nonempty mask still requires its prefix scan and count readback, but launches no coordinate kernel.

Argwhere requires one count readback. The inherited ordinary mask-selection path also reads one four-byte gather-bounds flag.
Tensor values and index arrays remain on the GPU during execution.
The existing layout limits still apply: rank at most eight and signed 32-bit extents.
The product of selected count and rank must also fit a signed 32-bit extent.

The shared checks cover lengths 1, 127, 128, 129, 257, and 531, with empty, full, alternating, sparse, and broadcast masks.
They cover ranks one, three, and eight, exact FP32 payload bits, full-range integers, boolean values, and retained inputs.
Four independent weighted gradient checks include reversed transposes, empty selections, and full selections.
The Q8 cases retain quantized output kind and check both aligned `PackedU32` storage and compact `Native` byte storage.
Unaligned `PackedU32` fixtures are rejected by both backends. Their failed runs remain diagnostic evidence.

Quantized selection can dequantize and requantize through Burn's default implementation.
The Q8 reference therefore permits half a quantization step plus four FP32 rounding ULPs.
For this corpus, the independent F64 bound is `max_abs / 255 + 4 * F32_EPSILON * max_abs`.
FP32, integer, and boolean selection retain exact comparisons.
The initial lossless Q8 assertion and Linux's returned packed bytes remain in the diagnostic evidence.
Both D3OS and Linux CUDA pass the corrected public corpus on the same physical GPU, with 4 GiB per VM.

The raw native checks cover source/output guards, malformed prefix contents, complete writers, and queued prefix lifetime.
They also check coordinate 16,777,219, beyond FP32's consecutive integer range, using a prefix constructed on the GPU.
Service counters check one readback for argwhere and two for ordinary mask selection, without tensor uploads.
The readback sizes follow the checked one-word count and bounds buffers.

The protocol-29 portable bundle contains 210 inference entries and 66 kernel profiles.
All 209 prior instruction and constant sections remain byte-identical, and the new entry passes the host spill audit.
The pinned Linux CUDA backend reads argwhere masks on the host and uses GPU gathers for selection.
Its public API comparison therefore contains no GPU-compaction performance baseline.
D3OS also passes the full operator suite, and every guest-compiled module matches its host audit hash.
These checks add no execution-performance or compile-time parity measurement.

## Native ordered scatter updates

Protocol 30 adds scatter assignment, multiplication, minimum, and maximum for FP32 and I32 tensors.
Burn routes its float and integer scatter methods through these native operations.
The existing atomic addition and boolean OR kernels retain their previous implementations.

One GPU lane owns each destination row and processes its update axis in logical order.
The driver requires dense destination strides to prove that different rows cannot write the same element.
The native API first copies a strided destination into dense storage and materializes strided update values when necessary.
Indices retain signed strides, including reversed views.
Explicit memory tokens order repeated reads and writes within a row.

Integer products wrap modulo 2^32. FP32 products use nearest-even rounding and preserve subnormal values.
Assignment copies stored bits, including NaN payloads and signed zeros.
Minimum and maximum use numeric minimum/maximum semantics when one operand is NaN.
A four-byte status buffer reports negative and out-of-range indices as recoverable errors.
Tensor payloads remain on the GPU. Empty updates enqueue no kernel.

Burn's public scatter contract requires unique indices within each row.
The shared corpus respects that requirement for both forward and backward checks.
It covers all three axes, reversed transposes, shared inputs, and update lengths 1, 127, 128, and 129.
Twelve independent gradient cases check both input derivatives, including equal-value minimum and maximum cases.
The public IEEE corpus checks signed zeros, NaNs, infinities, and subnormal products.
Linux CUDA passes this complete corpus on the physical RTX 4090 Laptop GPU.

The explicit Flex CPU backend passes the finite corpus, including gradients and empty updates.
Its IEEE run returns positive zero for `min(+0, -0)`, where the strict reference expects negative zero.
That diagnostic remains separate from the successful Linux CUDA result.

D3OS passes the same complete public corpus on the same physical GPU, with 4 GiB per VM.
Its raw checks cover all eight dtype/update combinations, 129 destination rows, and 257 updates per row.
They check neighboring guards, reversed index storage, input preservation, and rejected destination aliasing.
Repeated-index forward checks confirm logical update order as a native extension.
They do not extend Burn's unique-index gradient contract.
Service counters check zero tensor uploads and one index-status read for the native operation.
A queued consumer also verifies destination lifetime after the source tensors leave scope.

The current bundle contains 212 inference entries and 66 kernel profiles.
All 210 prior entries retain identical instruction and constant sections.
The two new entries pass the host spill audit, and all three guest-compiled modules match their host audit hashes.
The complete D3OS operator suite passes. These results add no workload-throughput or compile-time parity measurement.
Protocol 30 requires a matching driver, SDK, application image, and portable bundle.

## Native vector-selection updates

The native backend now supports all five FP32 and I32 `select_assign` update modes.
Assignment, multiplication, minimum, and maximum join the existing addition path.
The index vector becomes a signed-stride broadcast view across the other tensor dimensions.
The operation reuses resident scatter, including its private destination copy and four-byte bounds status.
This composition requires no index expansion kernel, tensor readback, or additional portable entry.
Protocol 30, 66 kernel profiles, and the 212-entry bundle remain unchanged.

The shared corpus checks all three axes, offset and reversed index vectors, reversed tensor views, and retained inputs.
It covers update lengths 1, 127, 128, and 129, FP32 IEEE values, full-range I32 arithmetic, and empty FP32 updates.
Fifteen independent weighted gradient cases check both input derivatives, including minimum and maximum ties.
A rank-eight case checks the broadcast index view across singleton dimensions.
The repeated-index corpus checks addition, multiplication, minimum, and maximum with exact, order-independent references.
It excludes duplicate assignment and duplicate gradients, whose public contracts are undefined.

Linux CUDA passes the complete corpus on the physical RTX 4090 Laptop GPU.
The explicit Flex CPU backend passes the finite corpus.
Its strict IEEE run retains the same signed-zero minimum difference as the scatter diagnostic.

D3OS passes the same complete public corpus on the same GPU, with 4 GiB per VM.
Its raw checks cover all ten dtype/update combinations and retain the original index storage, including neighboring guards.
The checks use a nonzero index offset and stride -2, with zero broadcast strides across the other dimensions.
Native counters report no tensor uploads and one four-byte bounds read per nonempty operation.
Invalid indices return recoverable errors. Empty FP32 and I32 updates enqueue no GPU work.
The full native operator suite also passes, including the refactored scatter corpus.

All three guest-compiled modules match their host audit hashes, and the portable bundle is byte-identical to the preceding checkpoint.
This step adds operation coverage without a new throughput or compile-time parity measurement.
Quantized selection-update execution remains unverified across the supported format matrix.

## Native multidimensional indexing

The native backend implements FP32 and I32 `gather_nd` and all five `scatter_nd` update modes.
An index tuple addresses leading data dimensions. The output preserves index batch dimensions and trailing data slices.
Empty index tensors remain invalid under Burn's public contract.
Empty trailing data slices perform no GPU element work.

The implementation forms exact I32 linear row indices on the GPU.
It validates each coordinate independently before the final gather or scatter bounds check.
This prevents an invalid tuple such as `[1, -1]` from aliasing an otherwise valid flattened position.
Nonempty operations replace invalid tuples with sentinel indices and produce a recoverable bounds error.
Signed source, index, and value views use resident copies where flattening requires contiguous storage.
Tensor payloads remain on the GPU. Each nonempty operation reads one four-byte bounds flag.

A single full-coordinate tuple can produce a rank-zero scalar.
The native layout now represents that scalar as one logical element with no axes.
Its strided launch descriptor uses one physical dimension of length one, preserving the scalar's logical shape.
The existing portable kernels therefore handle scalar copies, arithmetic, broadcasting, and ND values without a new GPU entry.
Protocol 30, 66 kernel profiles, and the 212-entry bundle remain unchanged.

The shared corpus covers tuple widths one through three and index ranks one through three.
It checks reversed and offset views, batches, update lengths 1, 127, 128, and 129, and shared input preservation.
Fifteen scatter-gradient cases check both input derivatives for all five modes.
Three gather-gradient cases use independent weighted references.
Large I32 gathers check flattened positions 16,785,407 and 16,785,408 without FP32 conversion.
The full Flex CPU corpus passes, including scalar forward results, duplicate addition, and empty trailing slices.

The initial Linux CUDA run fails scalar readback with CubeCL's `Unsupported rank 0` error.
The subsequent non-scalar run returns a duplicate gather gradient of 1, where 513 repeated indices require 513.
The pinned ND scatter kernel uses ordinary loads and stores for addition, without atomic accumulation.
These failures remain separate evidence. They are not successful conformance cases.

Linux CUDA passes the separate unique, non-scalar corpus, including all 18 gradient cases and empty trailing slices.
D3OS passes the full corpus on the same GPU, including scalar forward results and duplicate accumulation.
Its raw checks reject cancelling invalid tuples, preserve source guards, and retain exact IEEE payload bits.
They also cover a rank-eight scalar result and FP32/I32 scalar arithmetic.
Native counters check zero tensor uploads and one four-byte bounds read per nonempty operation.
Empty results reuse the source buffer and enqueue no GPU work.

The first native run caught an extra clear launch from an empty-result allocation.
The corrected run passes the complete operator suite, and all guest-compiled modules match the unchanged host audit hashes.
This step adds no execution-throughput or compile-time parity measurement.

An isolated scalar-gradient diagnostic also covers a pinned Burn autodiff limitation.
Its CPU run panics in `unsqueeze_like` when a rank-one gradient targets a rank-zero shape.
The isolated Linux scalar-gradient run fails earlier during rank-zero readback.
Native scalar autodiff remains unverified. Scalar forward support does not establish scalar autodiff support.

## Empty quantized diagnostics

An empty Q8 tensor with an explicit unit scale round-trips through quantized records on both D3OS and Linux CUDA.
Its record contains the four F32 scale bytes `[0, 0, 128, 63]` and no payload bytes.
Both backends retain the empty shape, quantized kind, exact record bytes, and empty dequantized result after reload.

Empty quantized selection and all-false quantized mask selection fail on both GPU backends.
The pinned Linux CUDA default recalibration reports `ReduceAxisTooSmall` for an empty reduction.
D3OS reports recoverable `float_argmin` and `float_argmax` errors and acknowledges each error before subsequent work.
The native operator suite then completes successfully.
These rejected selections are diagnostics, not successful conformance cases or an additional Linux-supported capability gap.
Other quantization formats and layouts still require separate coverage.

## Remaining implementation areas

The unresolved work includes ordinary dtype arithmetic, quantized conversions, further index formats, complete integer arithmetic, further matrix dtypes, and additional floating-point operations.
Further sorting and compaction dtypes also require coverage.
Further convolution dtypes and numerical domains beyond the checked finite corpus remain unresolved.
Compaction formats beyond the checked Q8 cases remain unverified.
Broadcast cross-product gradients, graph capture, profiling, memory capabilities, records, and observable errors require matched Linux capability checks.
The current layout also restricts rank to eight and extents to signed 32-bit values.
The method inventory records defaults, but each composed path still needs dtype, shape, ownership, and execution checks.

Scalar autodiff and the full quantized ND format matrix also remain unresolved.

## Reproduce the checks

Run the source inventory from the repository root:

```bash
cargo +1.98.1 run --locked --manifest-path tools/burn-backend-audit/Cargo.toml -- \
  . target/nvidia/burn-backend-audit/traits.json
```

Build the Linux probes with their default CUDA feature:

```bash
cd tools/mnist-benchmark
cargo build --release --locked --bin capabilities --bin erf --bin integer --bin integer_power --bin predicate --bin movement --bin integer_matrix --bin scan --bin sort --bin cross --bin compaction --bin scatter_updates --bin quantized_empty --bin nd_indexing --bin storage --bin conversions --bin remainder --bin inverse --bin interpolate --bin spatial_convolution --bin deform
cd ../..
```

Run the strict inverse comparison:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-inverse \
  --burn-inverse tools/mnist-benchmark/target/release/inverse
```

For per-operation diagnostics, use a fresh directory and add `--burn-inverse-diagnostic`.
For the bounded corpus, also add `--burn-inverse-finite`.
The D3OS `gpu-self-test` requires both complete corpora to pass the same strict bound.

Run the ordinary finite remainder comparison:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-remainder-finite \
  --burn-remainder tools/mnist-benchmark/target/release/remainder \
  --burn-remainder-finite
```

For the full IEEE diagnostic, use a fresh directory and omit `--burn-remainder-finite`.
For the wide finite diagnostic, replace that flag with `--burn-remainder-wide`.
The D3OS `gpu-self-test` always includes the full IEEE corpus.

Run the conversion comparison with the pinned Linux Flex32 failure excluded:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-conversions \
  --burn-conversions tools/mnist-benchmark/target/release/conversions \
  --burn-conversions-no-flex32
```

For the full Flex32 diagnostic, use a fresh directory and omit `--burn-conversions-no-flex32`.
The D3OS `gpu-self-test` always includes all 15 formats and the complete native gradient and boundary corpus.

Run the storage comparison with the pinned Linux Flex32 failure excluded:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-storage \
  --burn-storage tools/mnist-benchmark/target/release/storage \
  --burn-storage-no-flex32
```

For the isolated Flex32 diagnostic, use a fresh directory and replace `--burn-storage-no-flex32` with `--burn-storage-flex32-direct`.
Without either flag, the probe includes all 15 formats and retains the Linux Flex32 error.
The D3OS `gpu-self-test` always includes all 15 formats.

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
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-compaction \
  --burn-compaction tools/mnist-benchmark/target/release/compaction
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

Run the ordered scatter comparison on Linux CUDA:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-scatter-updates \
  --burn-scatter-updates tools/mnist-benchmark/target/release/scatter_updates
```

Run the vector-selection corpus with the same executable:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-select-updates \
  --burn-scatter-updates tools/mnist-benchmark/target/release/scatter_updates \
  --burn-scatter-updates-select
```

Run the full ND diagnostic first and retain any failure:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-nd-indexing-all \
  --burn-nd-indexing tools/mnist-benchmark/target/release/nd_indexing
```

Run the accepted Linux ND domain separately:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-nd-indexing-unique \
  --burn-nd-indexing tools/mnist-benchmark/target/release/nd_indexing \
  --burn-nd-indexing-no-scalar --burn-nd-indexing-unique
```

The flags exclude the retained scalar-read and duplicate-gradient failures.
They leave the D3OS suite unchanged.
The `--burn-nd-indexing-scalar-grad` flag runs the isolated scalar autodiff diagnostic.

Run each empty quantized diagnostic in a separate Linux VM:

```bash
for case in direct select mask; do
  python3 tools/nvidia/linux-reference.py \
    --directory "target/nvidia/linux-burn-quantized-empty-$case" \
    --burn-quantized-empty tools/mnist-benchmark/target/release/quantized_empty \
    --burn-quantized-empty-case "$case"
done
```

The diagnostic's JSON `status` distinguishes success from an execution error.
A successful launcher exit alone does not establish API support.

Run the full Linux interpolation diagnostic and retain its wide-coordinate error:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-interpolate-wide \
  --burn-interpolate tools/mnist-benchmark/target/release/interpolate
```

Run the accepted compact comparison in a separate VM:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-interpolate-compact \
  --burn-interpolate tools/mnist-benchmark/target/release/interpolate \
  --burn-interpolate-case compact
```

The compact flag excludes only the two wide-coordinate shape groups. D3OS retains all ten groups.
For isolated diagnostics, use fresh directories with cases `empty`, `ieee`, `linear_backward`, `cubic_backward`, and `lanczos3_backward`.

Run the strict spatial-convolution comparison first and retain its FP32 diagnostic:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-spatial-strict \
  --burn-spatial-convolution tools/mnist-benchmark/target/release/spatial_convolution
```

Run the separate TF32-exact input corpus:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-spatial-exact \
  --burn-spatial-convolution tools/mnist-benchmark/target/release/spatial_convolution \
  --burn-spatial-tf32-exact
```

The flag changes the checked inputs. It does not change the numerical tolerance or establish general FP32 conformance.
For empty diagnostics, use fresh directories with `--burn-spatial-empty` and `--burn-spatial-empty-backward` separately.
The `--burn-spatial-case` option selects one of the 14 shape groups for isolated diagnosis.
D3OS retains both complete numerical corpora and both empty checks.

Run the complete stored-value movement comparison:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-movement \
  --burn-movement tools/mnist-benchmark/target/release/movement
```

Use fresh directories for each retained diagnostic.
The isolated comparison adds `--burn-movement-no-flex32`, `--burn-movement-skip-empty-bool-scatter`, and `--burn-movement-matching-bool-masks`.
Those flags preserve all other empty cases, both numeric mask stores, and mixed-width mask selection.
The runner's `--expected-summary` mode computes corpus metadata without initializing CUDA.
It reports counts, not a GPU execution result.

Run the ordinary predicate comparison:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-predicate \
  --burn-predicate tools/mnist-benchmark/target/release/predicate
```

For the separate corpus without Flex32 uploads, add `--burn-predicate-no-flex32` and use a fresh directory.
For an isolated finite diagnostic, add `--burn-predicate-finite` and use another directory.
The finite diagnostic retains signed zeros and subnormals.

Run the exact integer-power comparison:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-integer-power \
  --burn-integer-power tools/mnist-benchmark/target/release/integer_power
```

Run the strict deformable-convolution comparison and retain its diagnostic:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-deform-strict \
  --burn-deform tools/mnist-benchmark/target/release/deform
```

Run the full forward-only comparison with TF32-exact inputs:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-deform-forward \
  --burn-deform tools/mnist-benchmark/target/release/deform \
  --burn-deform-tf32-exact --burn-deform-forward-only
```

For the failed complete gradient diagnostic, use a fresh directory and omit `--burn-deform-forward-only`.
For isolated passing gradient domains, add `--burn-deform-case 4` or `--burn-deform-case 7` with TF32-exact inputs.
For empty diagnostics, use `--burn-deform-empty`. Its separate forward-only run also needs `--burn-deform-forward-only`.
D3OS retains the complete general, exact-input, and empty-batch gradient corpora.

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
The compaction evidence lives under `target/nvidia/burn-backend-validation/compaction-final/` and `target/nvidia/burn-backend-audit/linux-compaction-native-tails/`.
The audit directory contains `traits-compaction.json`, `compaction-comparison.json`, and retained format and quantization diagnostics.
The ordered scatter evidence lives under `target/nvidia/burn-backend-validation/ordered-scatter-first/` and `target/nvidia/burn-backend-audit/linux-scatter-updates-ieee/`.
The audit directory contains `traits-ordered-scatter.json`, `ordered-scatter-comparison.json`, and `ordered-scatter-section-comparison.json`.
The empty quantized diagnostics live under `linux-quantized-empty-final-direct/`, `linux-quantized-empty-final-select/`, and `linux-quantized-empty-final-mask/`.
The native scatter run contains the corresponding D3OS diagnostics.
The vector-selection evidence lives under `target/nvidia/burn-backend-validation/select-updates-first/` and `target/nvidia/burn-backend-audit/linux-select-updates-ieee/`.
The refactored scatter comparison lives under `target/nvidia/burn-backend-audit/linux-scatter-updates-refactored/`.
The audit directory contains `traits-select-updates.json`, `select-updates-comparison.json`, and separate Flex CPU diagnostics.
The accepted ND evidence lives under `target/nvidia/burn-backend-validation/nd-indexing-empty-fixed/` and `target/nvidia/burn-backend-audit/linux-nd-indexing-unique/`.
The first native failure remains under `target/nvidia/burn-backend-validation/nd-indexing-first/`.
The Linux scalar, duplicate, and scalar-gradient failures remain under `linux-nd-indexing-all/`, `linux-nd-indexing-no-scalar/`, and `linux-nd-indexing-scalar-grad/`.
The audit directory contains `traits-nd-indexing.json`, `nd-indexing-comparison.json`, and separate CPU diagnostics.
The accepted storage evidence lives under `target/nvidia/burn-backend-validation/storage-conservative-flags/` and `target/nvidia/burn-backend-audit/linux-storage-explicit-bool/`.
The audit directory contains `traits-storage-final.json`, `storage-comparison.json`, and `storage-section-comparison.json`.
The isolated Linux Flex32 failure remains under `linux-storage-flex32-direct/`, with the assignment diagnostic under `linux-storage-alias/`.
The first native boolean fixture failure remains under `target/nvidia/burn-backend-validation/storage-first/`.
The accepted conversion evidence lives under `target/nvidia/burn-backend-validation/cast-resident-final/` and `target/nvidia/burn-backend-audit/linux-conversions-final/`.
The audit directory contains `traits-cast-final.json`, `cast-comparison.json`, and `cast-section-comparison.json`.
The first rounding failure remains under `cast-first/`, with the old machine code, portable input, and instruction dumps retained.
The gradient-constructor diagnostic remains under `cast-direct-bf16/`; the replaced Boolean alias check remains under `cast-final/`.
The Linux Flex32 metadata diagnostic remains under `linux-conversions-flex32/` with its exact shared source archive.
The remainder evidence lives under `target/nvidia/burn-backend-validation/remainder-remf-final/` and `target/nvidia/burn-backend-audit/linux-remainder-finite-final/`.
The audit directory contains `traits-remainder-final.json`, `remainder-comparison.json`, and `remainder-section-comparison.json`.
The failed Linux diagnostics remain under `linux-remainder-ieee/` and `linux-remainder-wide/`.
The earlier integer-significand implementation passes under `target/nvidia/burn-backend-validation/remainder-final/`.
Its full IEEE run also remains under `target/nvidia/burn-backend-validation/remainder-first/`.
The accepted inverse evidence lives under `target/nvidia/burn-backend-validation/inverse-errors-fixed/`.
Its Linux counterparts are `target/nvidia/burn-backend-audit/linux-inverse-ieee-domain-fixed/` and `linux-inverse-finite-domain-fixed/`.
The strict Linux failures remain under `linux-inverse-ieee/` and `linux-inverse-finite/`.
The audit directory contains `traits-inverse.json`, `inverse-comparison.json`, and `inverse-section-comparison.json`.
Manifests record the shared validation source hashes, executable hashes, and execution backend.
These generated local artifacts are not Git-tracked. The probes and audit tool are tracked for reproduction.

The interpolation evidence lives under `target/nvidia/burn-backend-validation/interpolate-final/`.
Its Linux counterparts use the `target/nvidia/burn-backend-audit/linux-interpolate-` prefix.
The audit directory contains `traits-interpolate-final.json`, `interpolate-comparison.json`, and `interpolate-section-comparison.json`.
The original scheduler panic and initial Lanczos scratch export remain in separate diagnostic directories.

The spatial convolution evidence lives under `target/nvidia/burn-backend-validation/spatial-final/`.
Its Linux counterparts use `target/nvidia/burn-backend-audit/linux-spatial-` with the `normal-final`, `exact-final`, `empty-final`, and `empty-backward-final` suffixes.
The audit directory contains `traits-spatial-final.json`, `spatial-comparison.json`, and `spatial-section-comparison.json`.
Earlier 48-case runs remain separate from the final 56-case corpus.

The deformable-convolution evidence lives under `target/nvidia/burn-backend-validation/deform-first/`.
Its Linux counterparts use the `target/nvidia/burn-backend-audit/linux-deform-` prefix.
The audit directory contains `traits-deform-final.json`, `deform-comparison.json`, and `deform-section-comparison.json`.
The coordinate-row diagnosis includes `deform-linux-row-index-diagnostic.json` and the archived pinned coordinate-kernel source.
The pointer-inference compiler diagnostic remains under `deform-pointer-inference/`.

The integer-power evidence uses `target/nvidia/burn-backend-validation/power-first/` and `target/nvidia/burn-backend-audit/linux-power/`.
The audit directory contains `traits-power-final.json`, `power-comparison.json`, and `power-section-comparison.json`.

The predicate evidence uses `target/nvidia/burn-backend-validation/predicate-first/` and the `target/nvidia/burn-backend-audit/linux-predicate-` prefix.
The audit directory contains `traits-predicate-final.json`, `predicate-comparison.json`, and `predicate-section-comparison.json`.

The movement evidence uses `target/nvidia/burn-backend-validation/movement-first/` and the `target/nvidia/burn-backend-audit/linux-movement-` prefix.
The audit directory contains `traits-movement-final.json`, `movement-comparison.json`, and `movement-section-comparison.json`.
Each Linux directory retains its shared source, executable, runner, expected summary, launcher, and hash manifest.
