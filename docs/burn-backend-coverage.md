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

The current inventory contains 484 methods: 296 native overrides and 188 inherited defaults.
All required methods have declarations or implementations. This count does not establish runtime support.
The report also records 154 native helper bodies for manual review of dtype and shape restrictions.

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
Quantized conversions and general arithmetic remain incomplete for the new formats.
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
The protocol-41 section below covers indexed assignment and integer index formats.
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

The protocol-41 section extends ND reads, indexed assignment, and integer index formats.
[Protocol 55](#typed-indexed-accumulation-and-embedding-gradients) adds wider arithmetic updates.
General dtype arithmetic, full quantized contracts, graph capture, and other execution capabilities also remain incomplete.
These correctness checks do not establish execution-performance or compile-time parity.

## Typed indexed payloads and integer indices

Protocol 41 adds packed indexed assignment with 245 inference entries and 80 compute profiles.
Gather and select retain all 15 ordinary payload formats.
Native ND gather also accepts these formats, including both boolean stores.
Burn's public boolean bridge rejects ND operations on both backends.
The public corpus therefore excludes boolean ND operations, while native checks cover boolean ND gather separately.

Indexed assignment, vector assignment, and ND assignment accept all 13 ordinary numeric payload formats.
At protocol 41, other update modes require F32 or I32 values.
[Protocol 55](#typed-indexed-accumulation-and-embedding-gradients) extends these modes to the remaining ordinary numeric formats.
Boolean OR retains its separate contract.
The existing F32/I32 assignment routes remain unchanged.
The new kernel assigns one lane to each destination row outside the indexed axis.
A loop-carried token orders repeated destinations within each row.
Narrow values use word-level CAS to preserve neighboring fields and initialized tail bytes.

Eight-byte values retain both raw words. Logical and unsigned physical bounds guard every payload access.
The driver requires a dense destination and rejects overlap involving either writer.
Native operations copy the destination and snapshot noncontiguous or nonzero-offset values before dispatch.

Gather, select, scatter, and ND coordinates accept all eight integer index formats.
Original-dtype comparisons reject negative and oversized values before their resident I32 conversion.
Invalid I64/U64 values therefore cannot alias valid positions through discarded high bits.
I8/I16/U8/U16 indices avoid thresholds that their original formats cannot represent.
Only the existing isolated bounds result crosses to the host.
ND indexing retains independent coordinate checks before linearization.

Float argmin/argmax, argsort, sort/top-k index pairs, argtopk, and boolean argwhere honor their requested integer output dtype.
At protocol 41, native ordering and reductions require F32/I32 values.
The later typed reduction checkpoints extend the reduction formats.
Output casts preserve the existing integer narrowing rule: low bits remain, including when an index exceeds the requested format.
Pooling index outputs and their backward sentinel rules remain I32-only.
General typed arithmetic, quantized contracts, and execution capabilities remain separate work.

D3OS passes 52,306 public cases, 223,260 source-preservation checks, and 4,507,438 output values on the passed-through RTX 4090 Laptop GPU.
The corpus uses independent byte records and checks actual output dtype, shape, and bytes.
It covers ranks one through eight, signed views, shared storage, large unique ND updates, and valid empty trailing slices.
ND coordinate tensors remain nonempty, as Burn requires.

| D3OS public operation group | Cases |
| --- | ---: |
| Gather and vector select | 10,080 |
| ND gather and ND assignment | 28,912 |
| One-axis and vector assignment | 8,736 |
| Extrema indices, ordering indices, and boolean coordinates | 4,578 |
| Total | 52,306 |

The guarded suite also passes 160 raw checks and 122 native API checks.
These checks cover packed tails, nonzero offsets, cross-block CAS, queued input drops, and original-width index bounds.
Native scalar-fill checks retain an F64 value one ULP above one through both boolean mask stores.
Public assignment comparisons use unique destinations. Guarded native checks separately establish deterministic duplicate ordering.
The final full suite passes all earlier regression gates, and all three guest compiler hashes match their host audits.
All 244 earlier entries retain 527 identical instruction and constant sections.
Nine host reference tests and the production image build also pass.

The first D3OS run passed indexing, then stopped at an older negative fixture that treated integer indices as unsupported.
The corrected fixture uses F32 indices and retains its zero-launch contract.
The complete rerun passes this fixture and the unchanged positive indexing corpus.

The retained Linux run stops at the pinned CubeCL rank-zero readback panic.
The diagnostic without scalar ND results stops at the unsupported direct Flex32 upload.
The diagnostic without both domains runs out of its 4 GiB guest RAM allocation after 725 seconds.
These records report no tensor mismatch before their failure. They do not establish complete Linux conformance.
The 12 GiB Linux diagnostic reached its 2,400-second deadline without a completion marker or reported tensor mismatch.
All Linux records use the same frozen reference, runner, executable, and launcher hashes.
These are correctness checks, with no execution-performance or compile-time parity claim.

## Typed basic floating arithmetic

Protocol 43 adds eight basic floating operations across F64, F32, Flex32, F16, and BF16.
The matching bundle contains 247 inference entries and 82 compute profiles.
All 246 earlier entries retain 532 identical instruction and constant sections.

The operations are addition, subtraction, multiplication, division, negation, absolute value, reciprocal, and sign.
Each native tensor result retains its input dtype.
The Native API rejects mixed tensor dtypes with a recoverable error.
Matching F32 operations retain their existing GPU entry points.
F64 arithmetic uses actual F64 operands and results, including both storage words.
Half and BF16 operations use typed arithmetic with destination rounding.
Division and reciprocal use nearest-even division without FTZ.

Backend scalar conversion follows the pinned Linux schedule through F64, F32, and the destination half format.
The four scalar arithmetic routes now retain F64 precision, range, and subnormals that the previous F32 conversion discarded.
The public corpus includes F64 scalar precision checks and half scalar rounding checks.
Stored intermediate results establish each primitive's rounding boundary separately from a fused operation.

The independent fixture generator uses exact integers, rational arithmetic, and an IEEE nearest-even encoder.
Finite results, infinities, and signed zeros require exact bits. Arithmetic NaNs require classification.
Native negation and absolute-value guards also require exact NaN payload transformations.
Sign returns positive zero for either zero sign or a NaN.
Fixture verification includes 6,948 comparisons against local MPFR.

D3OS passes 900 public cases, covering 101,825 results, 1,170 source checks, and 1,140 alias checks.
Guarded checks pass 214 raw dispatches, 219 fresh outputs, 520 Native results, 557 expected rejections, and 172 public results.
They cover signed views, broadcasts, ranks zero and eight, packed guards, zero tails, resident execution, error recovery, and queued input drops.
All earlier regression gates pass, including typed integer arithmetic and the 52,306-case indexing corpus.
All three guest compiler hashes match their host audits.

The first GPU run exposed an incorrect negative F16 sign result.
Its Tile IR encoded the negative half constant as zero. The generated instructions lacked the negative-result branch.
The corrected kernel classifies stored IEEE fields with integer operations instead of floating comparisons.
The same guarded input now returns the expected minus one.
An independent review also corrected an error fixture that reconstructed an I32 failed primitive as a Float tensor.
The corrected fixture reads that backend failure directly and preserves its recovery checks.

Linux CUDA passes the 680-case diagnostic across F64, F32, F16, and BF16, covering 82,928 result values.
This diagnostic excludes rank-zero results and direct Flex32 uploads. It retains the IEEE cases and materializes each chain intermediate.
The complete Linux run retains CubeCL's rank-zero readback panic.
The diagnostic without rank-zero results retains its direct Flex32 upload rejection.
Every Linux record preserves the same shared reference, fixture, executable, and launcher hashes.

Eight SDK tests, eight host reference tests, 24 Native unit tests, compiler audits, Clippy checks, and the production image build pass.
These are correctness checks, with no performance-parity claim.
Other floating math, typed reductions, scans, ordering, matrices, arithmetic updates, and broader quantized APIs remain incomplete.

## Typed square root and integral rounding

Protocol 44 extends the typed floating entry with square root, floor, ceil, ties-to-even round, and truncation.
The bundle retains 247 inference entries and 82 compute profiles.
All 246 other entries retain 532 identical instruction and constant sections.
The new operations support F64, F32, Flex32, F16, and BF16 without changing the tensor dtype.
Both systems use the passed-through RTX 4090 Laptop GPU.
Native F32 retains its existing elementwise entry points.

F64 square root uses actual F64 operands and results.
Half square root uses the compiler's F32 refinement and destination rounding.
The guarded GPU checks compare every F16 and BF16 input encoding against independently generated expected results.
The four typed integral operations transform stored IEEE fields with integer arithmetic.
They preserve signed zeros, infinities, and NaN payloads, including integral values beyond the integer-conversion range.
Native F32 floor, ceil, and truncation retain their earlier NaN classification contract.
Public arithmetic NaNs require classification. Raw typed integral results require exact input NaN bits.

The fixture generator uses exact rational arithmetic and integer square root with a squared-midpoint comparison.
It includes exhaustive half tables, F64 rounding boundaries, subnormal inputs, and separate primitive boundaries in materialized chains.
The local MPFR comparison passes 579,958 checks.
Ideal half square root and staged F32 square root agree wherever their results are non-NaN.

D3OS passes all 465 public cases across five formats, covering 32,740 output values and 15 materialized chains.
Guarded checks pass 255 raw dispatches, 270 fresh outputs, 565 Native results, 1,422 expected rejections, and 72 public results.
Ten exhaustive GPU checks cover 655,360 half results.
The checks retain guards, zero tails, signed views, broadcasting, resident execution, queued ownership, and recoverable errors.
All earlier regression gates pass, including 900 basic float cases, 7,349 integer cases, and 52,306 indexing cases.
All three guest compiler hashes match their host audits.

Linux CUDA passes 352 diagnostic cases across F64, F32, F16, and BF16, covering 26,172 output values.
The diagnostic excludes rank-zero results and direct Flex32 uploads, while retaining the full IEEE input domain.
The complete Linux attempt stops at CubeCL's rank-zero readback panic.
The diagnostic without rank-zero results stops at its direct Flex32 upload rejection.
All three Linux records retain matching reference, fixture, executable, launcher, and dependency hashes.
These are correctness checks. Full backend coverage, compile-time parity, and execution-performance parity remain incomplete.

## Typed ReLU and its backward operation

Native ReLU now accepts F64, F16, BF16, and Flex32 alongside the existing F32 path.
The new paths compose a resident ordered comparison with mask fill. They retain the input shape and dtype.
Both zero signs become positive zero. Negative finite values and negative infinity also become positive zero.
Positive values and every NaN retain their stored bits, including signaling payloads and signs.
The backward operation masks the stored forward output and preserves each unmasked gradient byte.

The existing F32 ownership logic and GPU entry remain unchanged.
Typed nonempty operations allocate a predicate and a full output without host activation transfers.
Empty typed operations return their input without a kernel launch or output allocation.
The native default mask uses U32. Separate explicit compositions validate U8 and U32 masks.
The SDK, all 247 inference entries, and the 82 compute profiles retain protocol 44.
This change does not broaden the general dtype capability declarations.

D3OS passes 85 public cases, 245 materialized stages, 12,200 exact floating values, and 320 boolean values across five formats.
The corpus checks forward results, stored-output backward results, idempotence, aliases, poison allocations, and explicit masks.
It includes true rank zero, rank eight, empty shapes, signed views, broadcasts, odd offsets, and packed boundaries.
Guarded checks pass 260 raw compositions, 260 Native forward results, 390 backward results, and 260 explicit mask compositions.
They also pass eight public results, eight expected rejections, 32 empty checks, and eight queued ownership checks.
Two exhaustive cases cover all 131,072 F16 and BF16 input encodings through every checked raw and Native route.
The existing F32 ownership and autodiff gate also passes. Broader non-F32 autodiff remains unclaimed.
All earlier regression gates pass, including typed rounding, basic float arithmetic, integer arithmetic, and indexing.
All three guest compiler hashes match the unchanged host audits.

Linux CUDA passes the 64-case diagnostic across F64, F32, F16, and BF16 on the same passed-through GPU.
It checks 9,748 exact floating values, 256 boolean values, and 184 materialized stages.
The diagnostic excludes rank-zero results and direct Flex32 uploads. It retains empty shapes and exact IEEE payload checks.
The complete Linux attempt retains a fusion vectorization panic on the true rank-zero F64 case.
The diagnostic without rank zero retains the direct Flex32 upload rejection.
Every Linux record preserves matching executable, source, expected-summary, and libdevice hashes.

Four shared host tests, three raw fixture audits, 24 Native unit tests, Clippy, and production image compilation pass.
The independent reference uses integer IEEE fields. It performs no host floating arithmetic to compute expected ReLU results.
These are correctness results. Full Burn coverage and Linux compile-time and execution-performance parity remain incomplete.

## Typed floating remainder

Protocol 45 adds tensor and scalar remainder across F64, F32, Flex32, F16, and BF16.
The existing floating entry now supports 14 operations. The bundle retains 247 entries and 82 compute profiles.
All 246 other entries retain 532 identical instruction and constant sections.
The floating entry adds a constant-bank section alongside its changed instructions.

The kernel computes a truncated residue, then corrects a nonzero residue whose sign differs from the divisor.
F64 residue and correction use actual F64 arithmetic. F32 and Flex32 use F32 arithmetic.
Half operands first retain their destination storage values, then widen to F32 for residue, correction, and selection.
The selected result rounds once to destination storage. The correction uses nearest-even addition without FTZ.
Finite values, infinities, and signed zeros require exact bits. A zero result keeps the dividend sign.
Arithmetic NaNs require classification. Preserved source tensors retain every byte, including NaN payloads.

The scalar wrapper now uses the canonical conversion through F64, F32, and destination storage where applicable.
This removes F64 narrowing and corrects rare F32 integer midpoint differences from its former direct conversion.
For `-3 % Int((1<<62)+(1<<38)+1)`, F32 and Flex32 now return `0x5e800000` instead of `0x5e800001`.
The F32 GPU entry and its mathematical implementation remain unchanged.
Public scalar wrappers convert numeric arguments to Float through Burn's `Scalar::new`.
Separate Dispatch checks preserve the original Float, Int, UInt, and Bool variants.

The independent generator uses exact rational quotients, integer nearest-even encoding, and explicit rounding at each specified stage.
Its original numerical corpus contains 1,608 tensor rows, 221 scalar specifications, and 2,210 scalar results.
Supplemental half witnesses distinguish destination scalar storage from a scalar that incorrectly stays in F32.
The host reference passes 1,008 MPFR residue comparisons, including 388 subnormal results and 256 zero results.
The half schedule comparison found no differences in its bounded search. The compiler audit separately verifies the required staging.

D3OS passes 692 public cases and 20,200 output values across all five formats.
The complete guarded gate passes 589 raw dispatches, 604 fresh outputs, 805 Native operations, and 762 public operations.
It also checks 383 expected rejections, 240 empty operations, and 20 queued ownership cases.
Operation counters verify 2,458 typed launches and 302 launches through the existing F32 entry.
The checks retain signed views, broadcasts, ranks zero and eight, packed guards, zero tails, exact aliases, and recoverable errors.
All earlier regression gates pass, and all three guest compiler hashes match their host audits.
These checks do not establish broader typed autodiff or reduction support.

Linux retains three explicit failures against the same strict numerical expectations.
The complete run stops at CubeCL's rank-zero readback panic.
The diagnostic without rank zero stops at the direct Flex32 upload rejection.
The diagnostic without either domain reaches numerical checks and reports 4,986 element mismatches.
Those diagnostics affect 97 F64 cases, 88 F32 cases, 92 F16 cases, and 92 BF16 cases.
They include zero signs, subnormal residues, extreme ratios, and infinities. These counts describe recorded mismatches, not independent conformance totals.
The complete diagnostic remains in the serial log. No Linux passing subset or performance-parity claim follows from these records.

All 116 SDK tests, six shared host tests, four raw fixture audits, repository Clippy checks, and the production image build pass.
These are correctness checks. Full Burn coverage, compile-time parity, and workload performance parity remain incomplete.

## CUDA module data pointers

F64 sine and cosine require pointers to coefficient and argument-reduction tables.
The native loader supports the observed SM89 ELF64 `REL` type-2 form with zero implicit addends.
It computes each pointer from the GPU virtual base, the packed section offset, and the symbol offset.
The physical VRAM address selects the upload destination only.

The loader validates the complete module and its entry metadata before GPU allocation.
It rejects unknown live relocations, invalid symbol extents, duplicate slots, and unsupported addends.
One uploaded image retains the metadata for all entries. Cached entry loads reuse its patched pointers.
Validation and patch writes are atomic. This does not establish allocator or hardware rollback after a device fault.

The unused Tile launch-metadata slot requires a generated qualification record.
The host checker requires complete instruction coverage and rejects accesses that can reach that slot within its supported disassembly profile.
It permits the observed fixed-bank CB2 indexing and the specified CB4 table-pointer operands.
This check does not provide a general SASS decoder or dataflow proof.

D3OS verifies the immutable compiler assets, portable input, and qualification record.
The generated cubin must match the qualified host output hash before admission.
The launcher stages the portable code and qualification records. Host cubin files remain audit artifacts.
Rebuilding a bundle generates new records without a permanent output-hash list in Rust.

The diagnostic kernel uses F64 sine and cosine with 12 MPFR reference inputs, including very large arguments.
D3OS passes 48 comparisons across four launches, including an empty case, with a maximum error of one representable step.
Guards, untouched tails, source preservation, and image-cache reuse also pass.
All three guest-compiled modules match the qualified host output hashes.
The preceding bundle's 534 instruction and constant sections remain byte-identical.
This diagnostic does not establish full typed trigonometric support through Burn.
The complete earlier D3OS GPU regression suite also passes.

## Typed exponential and logarithm

Protocol 46 adds Exp and Log across F64, F32, Flex32, F16, and BF16.
The floating entry supports 16 operations, with 247 bundle entries and 82 compute profiles.
All other 246 entries retain 532 identical instruction and constant sections.
The corrected floating entry uses 71 registers, 48,512 code bytes, and 128 threads.

F64 retains double operands, computation, and results.
Half operands retain their stored values, widen to F32, and round once to destination storage after Exp Full or Log.
F32 retains its existing elementwise entries. Raw checks also cover the typed F32 entry.

The independent generator uses directed MPFR intervals and exact integer destination rounding.
Its 306,572 reference rows include every F16 and BF16 encoding for both operations.
Three F16 inputs distinguish direct destination rounding from ideal F32 staging.
The checks retain both reference schedules and their measured distances separately.

The numerical policy defines project qualification targets:

| Format | Finite nonzero target |
| --- | --- |
| F64 | Exp and Log within one representable step |
| F32, Flex32 | Exp within two steps and Log within one step |
| F16, BF16 | Destination RNE applied to permitted neighboring F32 results |

NVIDIA publishes [empirical error figures](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#exponential-functions).
These figures do not guarantee every input. The project verifies classifications independently of finite numerical distance.
An unexpected zero fails that classification check, including a lost minimum subnormal.
Zeros and infinities require the expected sign. NaNs require classification, while preserved source tensors require every original byte.

F64 Exp uses a cutoff from directed MPFR intervals at the zero/subnormal boundary.
The first representable input greater than `ln(2^-1075)` is `0xc0874910d52d3051`.
Smaller inputs return positive zero. For inputs at this cutoff or greater, a zero Full result becomes the minimum subnormal.
The wrapper handles both zero signs and retains actual F64 computation.

The first Native run exposed 14 lost-subnormal observations. The corrected kernel passes those checks without a policy or fixture change.

D3OS passes 861,244 guarded comparisons and 1,190 shared cases with 709,814 observed values.
The guarded plan checks 2,496 launches, 595 rejections, 200 empty operations, and 30 queued ownership cases.
Guards, packed tails, residency, source preservation, recovery, and kernel routing also pass.
The complete regression suite passes, including every earlier remainder and arithmetic gate.

The complete Linux CUDA attempt stops at rank-zero readback in the pinned CubeCL implementation.
Excluding rank zero exposes its direct Flex32 upload rejection.
Excluding both domains completes 936 cases and 679,980 observations, with 14 F64 Exp classification errors.
Those errors return zero for a minimum-subnormal reference. Other selected dtype/operation cells record zero failures.

Zero and the minimum subnormal differ by one representable step. The separate classification gate still rejects that change.

These checks cover forward Exp and Log. General typed autodiff, full backend coverage, and workload performance parity remain incomplete.

## Typed floating power

Protocol 49 adds mathematical Power across F64, F32, Flex32, F16, and BF16.
The floating entry contains 24 operations. The portable bundle retains 248 entries and 82 compute profiles.
F64 preserves double operands and results. Half operands widen to F32, then round once to destination storage.

Native and Dispatch preserve the original scalar variant through canonical destination conversion.
The scalar wrappers retain Burn's five stored shortcuts for exponents `0`, `1`, `2`, `-1`, and `-2`.
Square and square-then-reciprocal therefore follow stored arithmetic, which can differ from mathematical power.
The public Float `powi_scalar` constructor produces a floating scalar.
Native rejects fractional or nonfinite scalar integer exponents through recoverable errors.
The backend integer-tensor route casts exponents to the base dtype before mathematical power.
Empty zero-exponent wrappers allocate storage through the normal allocation and zeroing path.

Project qualification permits two representable steps for F64 and four for F32/Flex32 mathematical power.
Half results must belong to the destination rounding image of four F32 neighbors on each side of the correctly rounded F32 reference.
Special classes, result signs, exact identities, and zero/nonzero preservation have separate gates.
The F64 helper implements the named identities and classifies power-of-two underflow with exact integer significands.
Its boundary proof includes subnormal bases and negative-base exponent parity.
The remaining generic approximation still depends on the pinned math implementation and its tested error allowance.

The independent MPFR registry contains 369,836 records for mathematical and stored-wrapper schedules.
On the D3OS GPU, the guarded suite passes 5,141,923 comparisons across 145 planned observation cells.
The shared Burn API suite passes 2,076 cases and 506,214 values across 160 planned cells.
All earlier GPU regression gates pass. All three portable families compile inside D3OS and match their qualified host code bytes.
The other 247 instruction entries remain unchanged from the accepted unary checkpoint.

The host certificate binds 80 registers, 182,272 code bytes, a 96-byte frame, and all 194 library-scratch instruction sites.
The tests bind both operands, scalar routes, stored schedules, class populations, error metrics, and worst-case reference witnesses.
Both earlier stopped Native attempts remain failure evidence. One exposed F64 numerical errors, and both exposed incomplete empty-allocation test expectations.

Three Linux CUDA attempts stop in the pinned scalar-integer probe.
Rank-zero and Flex32 exclusions do not remove that panic.
The explicit common-Linux four-format run checks 452,063 values and retains 116 F64 qualification failures.
A separate filtered F32 CUDA run passes 331 cases and 41,152 values.
These runs measure correctness only. General typed autodiff, full Burn coverage, and performance parity remain incomplete.

The accepted Native archive is `target/nvidia/burn-backend-validation/typed-power-final/`.
The earlier `typed-power-first/` and `typed-power-corrected/` directories retain their terminal failures.
Linux archives use `target/nvidia/burn-backend-audit/linux-typed-power-`.
Comparison records use the `target/nvidia/burn-backend-audit/typed-power-` prefix.

Build the Linux runner from the repository root:

```bash
cargo +1.98.1 build --manifest-path tools/mnist-benchmark/Cargo.toml --release --locked --bin typed_power
```

Run `python3 tools/nvidia/typed-power-generate.py --check` to reproduce the numerical registry.
Run `python3 tools/nvidia/test-typed-power.py` to check the evidence validator without a GPU.
Use `--burn-typed-power` with the Linux launcher and a fresh output directory for each attempt.
The `--common-linux`, dtype, finite, rank-zero, and Flex32 filters declare the selected comparison domain.

## Typed integer reductions

Protocol 51 adds Sum, Product, Mean, Min, Max, ArgMin, ArgMax, and paired extrema across all eight ordinary integer formats.
The portable bundle contains 250 entries and 83 compute profiles.
All 248 preceding instruction sections remain byte-identical.

Axis arithmetic wraps in a 32-bit accumulator for narrow integers and a 64-bit accumulator for I64 and U64.
Axis mean divides before the final narrowing conversion.
Global mean instead materializes the sum in the input dtype, then divides by a count representable in that dtype.
For I8 `[127,127]`, axis mean returns `127` and global mean returns `-1` under this Native policy.
Invalid global counts return recoverable errors before GPU work.

Minima, maxima, and winner selection retain every input bit.
Equal values select the first logical coordinate.
Exposed integer indices retain their low bits after narrowing.
Paired values come from an independent reduction of the original source, so a wrapped index cannot select the wrong value.
These pairs require no tensor payload or bounds-status readback.
The F32 paired path uses the same resident composition and preserves the requested index dtype.

Short axes use one GPU lane per packed output word.
Axes of at least 128 elements use one cooperative CTA per output word.
Both paths write complete words and zero unused tail bytes.
Wide loads and stores use two U32 words and support four-byte pointer alignment.
Existing F32 and I32 reductions keep their cooperative kernels.
For empty outputs, these legacy paths reuse storage without GPU work.
Other integer formats allocate one placeholder and launch only the allocator's clear kernel.

On the D3OS GPU, guarded and Native checks pass 4,445 result values.
The shared corpus passes 323 valid cases and 42 recoverable-error cases, with another 2,360 result values.
The Native scope checks source and alias preservation for all 365 cases.
All 464 rejection checks pass, together with both poisoned-output passes, fresh writers, queued lifetimes, and resident value/index consumers.
The complete earlier operator suite also passes, including classification gradients and MNIST training, checkpoint, and export checks.
All three portable families compile inside D3OS and match their qualified host code bytes.

The immutable GPU capture is `~/.cache/d3os/burn-integer-reduction-harness-complete/`.
A separate host-only replay corrects the expected scope for the 42 additional error-preservation checks.
It reruns every host gate without changing the 307-file capture or any runtime source.
The three earlier failed captures remain separate evidence.

The complete Linux CUDA run stops at the pinned CubeCL rank-zero layout rejection.
With rank zero excluded, all eight formats pass 283 valid cases and 2,320 values.
This filtered run includes 52 pair steps and 16 chains. It skips 34 invalid-input diagnostics.
An isolated I8 global-mean count diagnostic triggers the pinned CubeCL scalar-conversion panic.
Native rejects that count recoverably before GPU work.
These results qualify the selected integer corpus, not complete Linux backend parity.
The Linux captures remain under `~/.cache/d3os/burn-ir-linux-1/`, `burn-ir-linux-2/`, and `burn-ir-linux-3/`.

These checks measure correctness, not execution speed.
Further floating reductions, typed scans, quantized contracts, broader training support, and full backend and performance parity remain incomplete.
The broad dtype usage flags remain disabled until their complete conversion and arithmetic contracts pass.

Build the reference runner from the repository root:

```bash
cargo +1.98.1 build --manifest-path tools/mnist-benchmark/Cargo.toml --release --locked --bin integer_reduction
```

Run the Linux CUDA comparison in a fresh directory:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory /tmp/d3os-linux-ir \
  --burn-integer-reduction tools/mnist-benchmark/target/release/integer_reduction
```

The runner's `--expected-summary` and `--diagnostics` commands initialize no device.
The Linux launcher supports explicit dtype, rank-zero, and nonempty filters.
Its isolated error-case flag runs one diagnostic instead of treating a panic as conformance.

## Typed floating extrema

Protocol 52 adds typed floating extrema with 252 portable entries and 84 compute profiles.
All 250 preceding instruction sections remain byte-identical.
Min, Max, ArgMin, ArgMax, and value/index pairs support F64, F32, Flex32, F16, and BF16.
Index outputs support all eight ordinary integer formats.
Existing F32 cooperative reductions retain their kernel and empty-storage behavior.

Selection compares stored values without reducing their precision.
The first logical NaN wins, including its original payload, sign, and signaling bit.
Equal numeric values, including opposite signed zeros, select the first logical coordinate.
Exposed indices retain their low destination bits.
Paired values use an independent fold of the original source, so narrow indices cannot select an incorrect value.
Both folds remain resident and require no bounds-status readback.

Short and cooperative paths write complete output words with zero tail bytes.
F64 bindings permit four-byte alignment and load both U32 halves.
Legacy F32 empty outputs share source storage without allocation or GPU work.
The four additional formats allocate an independent one-word placeholder and clear it through the allocator.
Ownership checks cover individual outputs, pairs, source aliases, and the source after output release.

On the D3OS GPU, the new guarded and Native checks pass 11,400 values and 360 rejection checks.
The shared corpus passes another 735 values across 116 valid cases and 18 recoverable-error cases.
Checks include signed views, both broadcast directions, rank eight, packed tails, poisoned outputs, queued lifetimes, and exact ownership.
The complete earlier operator suite also passes.
All three portable families compile inside D3OS and match their qualified host code bytes.
The corrected VM and strict host harness both exit zero.

The immutable successful capture is `~/.cache/d3os/burn-float-extrema-harness-corrected/`.
Its external inventory binds 724 files, including source, compiler, image, result, and serial records.
The first capture remains failed evidence: image assembly stopped before QEMU because its target path was relative.
The corrected harness uses an absolute image directory and passes an independent path review.

The full Linux CUDA comparison stops at the pinned CubeCL rank-zero layout rejection.
With rank zero excluded, the all-format run stops at the pinned backend's direct Flex32 upload rejection.
Separate dtype runs retain the strict Native source-bit policy and exclude rank zero explicitly.

| Linux CUDA format | Attempted cases | Passed cases | Cases with source-bit differences |
| --- | --- | --- | --- |
| F64 | 23 | 14 | 9 |
| F32 | 23 | 20 | 3 |
| F16 | 22 | 19 | 3 |
| BF16 | 21 | 12 | 9 |

All 24 differing cases concern signed-zero ties or NaN payload/signaling bits.
They describe a source-bit policy difference, not a finite arithmetic tolerance comparison.
The six Linux captures remain under `~/.cache/d3os/burn-fe-linux-1/` through `burn-fe-linux-6/`.
No filtered run establishes complete Linux parity.
These correctness checks do not establish general floating arithmetic reductions, typed autodiff, full backend coverage, or performance parity.

Build and run the Linux reference from the repository root:

```bash
cargo +1.98.1 build --manifest-path tools/mnist-benchmark/Cargo.toml \
  --release --locked --target x86_64-unknown-linux-gnu --bin float_extrema
python3 tools/nvidia/linux-reference.py \
  --directory /tmp/d3os-linux-fe \
  --burn-float-extrema tools/mnist-benchmark/target/x86_64-unknown-linux-gnu/release/float_extrema
```

The runner's `--expected-summary` and `--diagnostics` commands initialize no device.
The launcher supports explicit dtype, rank-zero, nonempty, and isolated error-case diagnostics.
Each filtered run requires a fresh directory and retains its selected scope.

## Typed floating sums, products, and means

Protocol 53 adds floating Sum, Product, and Mean with 253 portable entries and 85 compute profiles.
All 252 preceding instruction sections remain byte-identical.
F64, Flex32, F16, and BF16 use the new typed axis kernel.
F32 retains its existing cooperative kernels and global reduction schedule.

Each new axis reduction visits logical coordinates in increasing order.
F64 uses F64 arithmetic. Flex32, F16, and BF16 use F32 accumulation.
Arithmetic requests nearest-even rounding without flush-to-zero.
Mean converts the integer axis length directly to the accumulator type before division.
Half outputs round once to their destination format after each axis reduction.
Arithmetic NaNs use a canonical positive quiet encoding.

Global reductions process axes by increasing original dimension length, with original axis order as the tie-breaker.
Each axis result returns to its storage dtype before the next stage.
Global Mean repeats axis Mean at each stage.
F32 instead retains its stored global Sum followed by division through the existing F32 scalar conversion.
Rank-zero globals preserve their stored bits through a shared-buffer reshape.
Empty globals produce the operation's exact identity or canonical NaN without a source read.

Complete word writers initialize packed tails to zero.
Checked signed views retain logical element offsets, while physical word addresses retain their full unsigned range.
Resident command composition keeps source clones, intermediate results, and queued consumers alive without activation readback.
The guarded checks cover original output words, neighboring guards, both poison patterns, deferred errors, and recovery.

D3OS passes 3,313 raw and Native value checks plus 356 shared final values.
The shared plan contains 71 valid device cases, two expected-error cases, and seven separate symbolic cases.
All 73 device attempts pass their required checks, including the two recoverable errors.
The seven symbolic cases prove bounded metadata and arithmetic properties without device execution.
They do not increase the device attempt or value counts.
All earlier operator regressions pass, and all three guest-compiled modules match their qualified host bytes.
The VM and strict host harness both exit zero.

The immutable capture is `~/.cache/d3os/burn-float-reduction-harness-work/first/`.
Its external inventory binds 757 files, including 492 runtime source files.
The archived reduction result retains a stale `vm_executed=false` preparation field.
Actual execution is recorded separately by the launcher exit, `vm_launched=true`, and source-bound GPU telemetry.
Acceptance uses those terminal records and preserves the original result unchanged.

The full Linux comparison stops at direct Flex32 upload rejection.
Separate F64, F16, and BF16 runs exclude rank zero explicitly and pass their selected device cases.

| Linux CUDA selection | Passed device cases | Separate model cases | Result |
| --- | --- | --- | --- |
| F64, rank zero excluded | 18 | 2 | Pass |
| F16, rank zero excluded | 15 | 4 | Pass |
| BF16, rank zero excluded | 14 | 0 | Pass |
| F32, rank zero excluded | 9 of 10 | 0 | Existing extrema NaN source-bit difference |

The F32 difference belongs to the existing exact-selection chain, not the new arithmetic kernel.
Separate full-rank F64 and invalid-axis captures retain the pinned CubeCL layout and error-path panics.
All seven Linux histories remain under `~/.cache/d3os/burn-fr-linux-1/` through `burn-fr-linux-7/`.
The Linux binary uses the source before a metadata-only lint correction.
Independent review proves unchanged arithmetic, corpus counts, and metadata behavior after that correction.
These selected observations establish neither general Linux reduction-order equivalence nor performance parity.

Build and run the Linux reference from the repository root:

```bash
cargo +1.98.1 build --manifest-path tools/mnist-benchmark/Cargo.toml \
  --release --locked --target x86_64-unknown-linux-gnu --bin float_reduction
python3 tools/nvidia/linux-reference.py \
  --directory /tmp/d3os-linux-fr \
  --burn-float-reduction tools/mnist-benchmark/target/x86_64-unknown-linux-gnu/release/float_reduction
```

Pure `--expected-summary` and `--diagnostics` commands initialize no device.
The launcher retains each selected dtype, rank-zero, nonempty, or isolated error scope in a fresh archive.
This protocol-53 checkpoint does not establish typed backward reductions, mixed-precision matrix products, full backend support, or performance parity.

## Typed mixed-precision matrix products and linear

Protocol 54 adds F16, BF16, and Flex32 matrix products, with 256 portable entries and 88 compute profiles.
All 253 preceding inference instruction sections and the auxiliary instruction sections remain byte-identical.
F32 and I32 retain their existing matrix paths.
The completed D3OS capture passed read-only verification and independent terminal review.
Protocol 55 now supplies the accepted default and preserves the protocol-54 bundle separately.
The protocol-53 bundle remains intact under `target/nvidia/burn-mixed-matmul-default-reduction53/`.

F16 and BF16 inputs retain their packed storage bits before F32 Tensor Core accumulation.
Flex32 retains F32 storage but rounds each matrix operand to F16 before multiplication.
For F16 and BF16, one cast writes the complete concatenated result into its destination format.
This gives one owner to each packed output word, including odd rows and batch boundaries.
Unused output half slots become zero.
Flex32 returns the private F32 scratch allocation with Flex32 metadata, without an additional cast kernel.

Checked matrix views support ranks two through eight, signed strides, offsets, and broadcast batches.
Empty outputs share source storage without device work.
Nonempty K=0 products produce exact positive zeros without reading inputs or running MMA.
Invalid dimensions, address ranges, or overlapping writes fail before submission.

Linear stores its matrix product in the destination dtype before adding typed bias.
Input, weight, and bias gradients use the same stored boundaries.
Broadcast gradient reductions process original axis indices in ascending order and store each intermediate result.
Ordinary public Autodiff checks include both tracked operands, either single tracked operand, shared sources, and linear backward.
Public tensor constructors need explicit dtype options because omitted options select the device default, even when TensorData contains half values.

The numerical corpus retains 121 rows and three unused reserves.
Shared qualification selects 93 rows: 39 geometry, 24 numerical, and 30 gradient rows.
Exact rational references and independent gradient scatter checks cover the exact-normal domain.
Four finite intervals are project empirical targets, not guarantees about vendor accumulation order.
Cancellation observations and nonempty zero signs remain diagnostic.
Required exception and subnormal policies remain separate from those observations.

The completed D3OS run passes these source-bound gates:

- Shared: 1,380 required routes and 143,013 required values, plus 14 diagnostic values.
- Raw and Native: 72,414 required values, 71 diagnostics, and 45,116 source words.
- Physical writers: 33 recipes, 66 guarded and 66 fresh scratch trials, and 44 guarded and 44 fresh packed casts.
- Runtime contracts: 342 Native outputs, 83 rejections, 15 deferred cases, three queued cases, and 21 public Autodiff routes.

The eight conditional-model values are a subset of the required values, not eight extra device comparisons.
Physical checks read the original Native allocation, including packed tails.
Both poison patterns require every live word or half slot to change, independently of the numerical policy.
Counters cover all 88 kernels and 14 services, with downloads confined to explicit validation reads.
All 57 inherited regression markers pass. The VM launcher and strict host harness both exit zero.

The immutable successful capture is `~/.cache/d3os/burn-mixed-matmul-harness-work/second/`.
Its inventory binds 784 files and 511 runtime sources.
The first D3OS capture remains failed evidence after its public fixtures accidentally selected default F32 and retained an AD context during downcasts.
The repair changes fixture construction and primitive extraction, with unchanged numerical policies, counts, and counter requirements.

The full Linux CUDA selection reports 843 failures and 642 unobserved routes.
Its complete report remains under `~/.cache/d3os/burn-mm-linux-1/`.
CPU diagnostics, earlier direct CUDA probes, and separate external scopes do not replace that failure.

A fresh BF16 broadcast-gradient selection passes 28 required routes and 32,198 required comparisons.
Both operands are tracked, with two broadcast axes.
All observed bits match the completed Native capture.
Its 117-file capture is `~/.cache/d3os/burn-mm-linux-2/`.
This result covers the selected graph. Full Linux parity remains open.

NativeExtensions, Preservation, and F64Diagnostics remain separately labeled and unqualified in this corpus.
The pinned high-level rank-one zero-input linear frontend also panics before reaching Native.

Build and run the Linux reference from the repository root:

```bash
cargo +1.98.1 build --manifest-path tools/mnist-benchmark/Cargo.toml \
  --release --locked --no-default-features --features cuda --bin mixed_matmul
python3 tools/nvidia/linux-reference.py \
  --directory /tmp/d3os-linux-mm \
  --burn-mixed-matmul tools/mnist-benchmark/target/release/mixed_matmul
```

Run the isolated BF16 gradient selection in a separate archive:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory /tmp/d3os-linux-mm-bf16-grad \
  --burn-mixed-matmul tools/mnist-benchmark/target/release/mixed_matmul \
  --burn-mixed-matmul-dtype bf16 \
  --burn-mixed-matmul-case bf16_both_multi_axis_batch_broadcast
```

Pure `--expected-summary` and `--diagnostics` commands initialize no device.
Each filtered or isolated diagnostic run needs a fresh archive and retains its explicit selection.
General F64 matmul and half-precision convolution or pooling remain incomplete.
Full Burn support and performance parity also remain open.

Run the independent fixture, CPU reference, and evidence-parser checks separately:

```bash
python3 tools/nvidia/generate-mixed-matmul-fixtures.py --check --self-test
cargo test -p d3os-ai --features training,cpu --test mixed-matmul-reference \
  --target x86_64-unknown-linux-gnu --locked
python3 tools/nvidia/test-mixed-matmul.py
```

These checks initialize no GPU. Device execution evidence remains a separate requirement.

## Typed indexed accumulation and embedding gradients

Protocol 55 adds one typed indexed-fold entry, with 257 portable entries and 89 compute profiles.
Read-only replay and independent terminal review pass. Protocol 55 is the accepted default.
The complete protocol-54 bundle remains under `target/nvidia/burn-indexed-accumulation-default-matmul54/`.
A later source-only renewal removes three overwritten initial assignments and pins the math certificate as `sm89-typed-erf25-library-scratch-v7`.
Strict all-targets Clippy and all 48 host checks pass.
All three complete cubins and fresh disassemblies remain byte-identical to the GPU-validated protocol-55 bundle.
The first full exporter attempt stops at its CPU guard. Retained TileIR assembly and exact original metadata checks complete the qualification separately.
That stopped attempt remains failed evidence. The renewal adds no GPU run or performance claim.
The prior v6-source bundle remains under `target/nvidia/burn-indexed-accumulation-default-v6-source/`.
All 256 earlier inference instruction sections remain byte-identical. The vector and tensor instruction sections also remain unchanged.
F32, I32, and assignment retain their existing Native routes.
The added Native routes cover F64, Flex32, F16, BF16, I64, I16, I8, U64, U32, U16, and U8.
Scatter, select updates, and ND slice updates support Add, Mul, Min, and Max through resident checked indices.

Each destination owner processes matching updates in logical order, starting from the actual initialized destination.
Packed elements share one physical-word owner. A 64-bit element has one owner for both words.
Integer Add and Mul wrap to the destination width. Signed comparisons and full-width U64 comparisons retain their integer interpretation.
F64 uses binary64 arithmetic. Flex32 uses F32 arithmetic without Tensor Core narrowing.
F16 and BF16 widen each stored operand to F32, perform one update, then round back to storage after every update.
Touched Add and Mul results use the selected canonical NaN policy. Untouched source bits remain exact.
Min and Max select the numeric operand against one NaN and retain a quieted running-base payload against two NaNs.
Min selects negative zero and Max selects positive zero for mixed zero signs.
These are explicit project contracts. They do not establish a universal CUDA numerical policy.

Native validates metadata before resident index conversion or work submission.
All eight integer index formats use device-side conversion and coordinate checks.
ND coordinates are checked before flattening, so an invalid coordinate cannot alias a valid row.
Nonempty updates use a private dense destination and exactly one synchronous four-byte status read.
That control read is part of the contract. Tensor payloads and index arrays stay on the device during the primitive.
Valid empty updates share source storage without device work.

The completed Native capture passes these source-bound checks:

- Shared NativeValues: 184 cases, including 20 ordinary public gradient graphs, with 2,556 required observations.
- Shared NativeErrors: eight semantic-error cases, with 122 required observations and same-index-dtype recovery.
- Raw: 138 recipes and 228 launches, including two initialized-base trials and same-context recovery.
- Special values: 264 records bind two trials for each of the 132 original C3 keys.
- Native physical checks: 99 cases, 33 empty aliases, 21 errors and recoveries, 26 metadata rejections, and 11 queued cases.
- Five supplemental public Autodiff lifetime graphs retain separate logical and counter evidence.

The physical checks read original Native allocations, including packed tails, guards, source words, and private output ownership.
Public gradient reads establish logical original-leaf results. They do not independently expose every physical word of a public leaf.
All intervals retain exact arrays for 89 kernels and 14 services, including observer requests.
All 131 inherited parser gates and 57 regression markers remain enforced.
The original launcher and host harness exit zero.
The separate MODEL evaluator passes 283 attachments with zero device attempts. Those results are not GPU passes.

The sealed successful capture is `/dev/shm/burn55-second/`, with 1,279 files and 918 source paths.
The original producer records 32 writes. Read-only replay reproduces those writes and all 31 final output files byte for byte.
Independent terminal review rejects all 786 altered-evidence controls.
The capture contains one boot archive-parser diagnostic. The original strict gates pass without an invented zero-error-log requirement.
The complete capture uses volatile shared memory. A disk audit subset preserves 1,246 file identities and omits 33 large binary files.
That subset does not replace the complete capture. The complete inventory and terminal results remain separately hash-bound.

| Evidence | SHA-256 |
| --- | --- |
| Complete capture inventory | `38546d0a1f5095a0844eb84efb3248338e0b92aac3f020963035e12a9fa5a4d1` |
| Runtime results | `ac2c430d63fad4939f80d695e35a46ca4e3d79bf0b49c698b85966a82dbc2a02` |
| Original write trace | `ec2e5b0bb636c97fc49a5f3ceb3e223e6b9b31ab61c3d56536d15da8acff8143` |
| Independent terminal review | `238d8e6c691317a45f6507460792d589e4d0f090f30f4564a46b22a8fadd9b79` |

The first Native preparation attempt remains failed evidence: a per-user tmpfs quota stopped archival before VM creation.
The second attempt uses a separate tmpfs mount, without deleting the failed evidence.

The full 167-case Linux CUDA selection remains failed.
Pinned CubeCL rejects direct Flex32 construction and interrupts the executor, so complete device attempt counts remain unknown.
A separate F16 selection reports 15 passes and two failures across 17 cases.
R087 loses a duplicate ND Add contribution. R154 loses a duplicate contribution to the original-leaf gradient.
Both expected results are exact under every update order in the selected finite domain.
The Native capture passes those reference cases. The Linux failures remain preserved without tolerance changes.
These results do not establish full Linux parity, full Burn support, or performance parity.

The portable reference generator, host tests, pure CLI inventories, and MODEL evaluation need no GPU.
The Linux runner retains the exact selected scope and a fresh archive for each attempt.
Common half-precision modules, typed RNG and ranges, broader dtype support, and matched workload performance remain open.

## Portable RunningState

BurnCore's non-`std` RunningState now uses a process-wide thread identity provider.
The D3OS runtime installs this provider before application code when its `burn` feature is enabled.
`NativeDevice::open()` uses the same one-time initialization.
The standard BurnCore branch retains `std::thread::ThreadId`.
The provider API remains available when another dependency independently enables the environment's `std` feature.

The kernel reserves each positive thread ID once, including after thread exit.
It rejects user-thread creation when the ID space is exhausted, instead of wrapping or reusing an ID.
Host tests cover the final ID, permanent exhaustion, concurrent reservations, and reservation ownership.

The host state suite passes 51 isolated cases across three feature combinations.
The cases cover registration, retained updates, shared clones, and records through actual Flex and Autodiff APIs.
On D3OS, the CPU state command passes ten runs with four guest CPUs.
These runs check 30 workers and 1,920 identity comparisons across yields, plus pending updates and record restoration.
The destination training flag remains independent of the serialized state.

The existing timer, gradient, aliasing, and optimizer-resume command also passes ten runs.
Process-exit checks pass 30 busy-worker runs and 30 sleeping-worker runs with the new kernel.
Independent source reviews find no introduced P1/P2 defects in the provider integration or checked ID allocation.
Three test files received formatting only after review. Their output matches the formatter applied to the reviewed sources exactly.

These checks close the tested platform dependency for RunningState.
They do not qualify half-precision BatchNorm, normalization, attention numerics, or GPU performance.
The GPU protocol and numerical policies remain unchanged.
The [state test guide](../tools/burn-state-conformance/README.md) lists the feature matrix and guest commands.

## Typed Erf

Protocol 50 adds Erf across F64, F32, Flex32, F16, and BF16.
The floating entry contains 25 operations. The portable bundle retains 248 entries and 82 compute profiles.
F64 uses double arithmetic. Half operands widen to F32, then round once to destination storage.
The precise F32 helper also replaces the previous helper used by the existing Erf and GELU checks.

Project qualification permits two representable steps for F64, F32, and Flex32.
Half outputs must belong to the destination rounding image of two F32 neighbors on each side of the reference.
Special classes, signed zeros, nonzero results, the output range, and exact oddness have separate gates.
These are bounded qualification targets, not an error guarantee for every possible wide input.

The independent MPFR registry contains 230,618 rows, including every F16 and BF16 encoding.
On the D3OS GPU, the guarded suite passes 1,085,172 comparisons.
The shared Burn API suite passes 680 cases and 510,046 values.
All five formats pass their exact oddness counts. These checks do not increase the reported output counts.
Observed wide error is at most one representable step. Both half formats exactly match their staged references.
F16 retains the expected one-step differences from direct rounding of the mathematical result.

All earlier GPU regression gates pass, including the original F32 Erf and GELU campaign with its unchanged eight-step target.
All three portable families compile inside D3OS and match their qualified host code bytes.
Only the Erf and typed floating arithmetic instruction entries change from the accepted Power bundle.
The other 246 instruction entries remain byte-identical.
The scratch certificate binds 88 registers, 224,512 code bytes, a 96-byte frame, and all 194 library-scratch sites.

The original Native capture stays immutable at `~/.cache/d3os/burn-typed-erf-harness-first/`.
A separate host-validation replay binds the corrected evidence parser and the original 292-file inventory.
It changes neither the captured GPU results nor their runtime source inventory.
The parser checks per-image error capacities, both distance histograms, class transitions, reference witnesses, and separate oddness failures.

The strict Linux CUDA attempt stops at the pinned rank-zero layout rejection.
Excluding rank zero exposes the separate Flex32 upload rejection.
Excluding both permits all 449,092 selected outputs, with 225,308 value qualification failures and eight oddness failures.
The pinned CubeCL Erf polyfill uses F32 coefficients and an absolute-error approximation.
It does not meet this project's stricter targets near zero, including the exact signed-zero requirement.
These results describe the Erf approximation in the pinned Burn CUDA backend.
All three attempts remain under `target/nvidia/burn-backend-audit/linux-typed-erf-`.
Comparison records use the `target/nvidia/burn-backend-audit/typed-erf-` prefix.

These runs measure correctness only. General typed autodiff, full Burn coverage, and performance parity remain incomplete.

Build the Linux runner from the repository root:

```bash
cargo +1.98.1 build --manifest-path tools/mnist-benchmark/Cargo.toml --release --locked --bin typed_erf
```

Run `python3 tools/nvidia/typed-erf-generate.py --check` to reproduce the numerical registry.
Run `python3 tools/nvidia/test-typed-erf.py` to check the evidence validator without a GPU.
Use `--burn-typed-erf` with the Linux launcher and a fresh output directory for each attempt.
The dtype, finite, rank-zero, and Flex32 filters declare the selected comparison domain.

## Typed Log1p, Sinh, Cosh, and Tan

Protocol 48 adds Log1p, Sinh, Cosh, and Tan across F64, F32, Flex32, F16, and BF16.
The floating entry contains 23 operations. The portable bundle retains 248 entries and 82 compute profiles.
F64 keeps double operands and results. Half inputs widen to F32, then round once to destination storage.

Log1p uses a precise F32/F64 polynomial implementation with explicit nearest-even arithmetic.
It preserves tiny inputs, subnormals, and signed zero without first rounding `1 + x`.
The existing F32 Log1p, Asinh, Acosh, and Atanh entries also use this helper.
The other 243 instruction entries remain unchanged from the trigonometric checkpoint.

The table gives project qualification targets in representable steps.

| Operation | F64 target | F32/Flex32 target |
|---|---:|---:|
| Log1p | 1 | 1 |
| Sinh | 2 | 3 |
| Cosh | 1 | 2 |
| Tan | 2 | 4 |

Half results must belong to the destination rounding image of the permitted F32 neighbors.
Special values, zero signs, nonzero preservation, result signs, and the Cosh range have separate gates.
The independent MPFR corpus contains 700,819 rows, including every F16 and BF16 encoding for each operation.

On the D3OS GPU, 7,640 guarded launches pass 3,443,996 comparisons with zero qualification failures.
The shared Burn API suite passes 2,560 cases and 1,595,518 result values.
All earlier GPU regression gates pass, including the inverse functions that use the new Log1p helper.
The guest compiles all three Tile IR families inside D3OS. Their code bytes match the qualified host outputs.

The renewed host certificate binds 79 registers, 152,832 code bytes, a 96-byte frame, and all 194 reduction-scratch instruction sites.
It retains the zero-local-memory checks for other protected entries.
Aggregate evidence binds reference-class populations to the selected workload and checks worst-case witnesses against immutable reference rows.
Materialized chains retain bounded class populations because their actual stored intermediates can vary within the numerical allowance.

The strict Linux CUDA attempt retains the pinned rank-zero rejection.
The next attempt excludes rank zero and retains the Flex32 upload rejection.
With both excluded, Linux checks 1,487,496 values and reports 163,294 Log1p failures.
Its observed Sinh, Cosh, and Tan cells report zero failures.
Both original and updated-metadata Linux records remain preserved.
These correctness results do not establish full Burn coverage, general typed autodiff, or performance parity.

The Native archive is `target/nvidia/burn-backend-validation/typed-unary-first/`.
Linux archives use `target/nvidia/burn-backend-audit/linux-typed-unary-`.
The comparison and section records use the `target/nvidia/burn-backend-audit/typed-unary-` prefix.

Build the Linux runner from the repository root:

```bash
cargo +1.98.1 build --manifest-path tools/mnist-benchmark/Cargo.toml --release --locked --bin typed_unary
```
Run `python3 tools/nvidia/typed-unary-generate.py --check` to reproduce the numerical references.
Run `python3 tools/nvidia/test-typed-unary.py` to check the evidence validator without a GPU.
Use `--burn-typed-unary` with the Linux launcher and a fresh output directory for each attempt.
Its dtype, operation, finite, rank-zero, and Flex32 filters declare the selected comparison domain.

## Typed trigonometric functions

Protocol 47 adds Tanh, Sin, and Cos across F64, F32, Flex32, F16, and BF16.
The floating entry supports 19 operations. The bundle retains 248 entries and 82 compute profiles.
The other 247 entries retain 534 identical instruction and constant sections.

F64 keeps double operands and results. Half inputs widen to F32, then round once to destination storage.
Native F32 retains its existing elementwise entries. Guarded checks also exercise the typed F32 entry.

The independent MPFR corpus contains 439,735 rows, including every F16 and BF16 encoding for each operation.
It includes 175 additional rows for three-step chains that look up references using each actual stored intermediate.
F64 Tanh permits one representable step; F64 Sin/Cos and F32/Flex32 permit two.
Half results must belong to the destination rounding image of the five permitted F32 neighbors.
Special values, zero signs, nonzero preservation, result signs, and the output range have separate gates.
Finite normal/subnormal transitions remain diagnostic within the numerical allowance.

These limits are project qualification targets, not vendor guarantees.

The enlarged entry uses 71 registers, 106,112 code bytes, and a 96-byte frame with 128 threads.
Its vendor argument-reduction routines use two 28-byte F32 arrays and one 40-byte F64 array.
The host audit recognizes the complete reviewed instruction schedule, constants, helper linkage, resources, and compiler provenance.
All 131 local instruction sites must match that certificate. Other previously spill-free entries retain their zero-local-memory checks.
A changed schedule requires new qualification. This certificate does not prove arbitrary SASS dataflow or numerical accuracy.

On D3OS, the guarded harness checks 5,634 launches and passes 2,239,480 comparisons with zero failures.
The shared Burn corpus passes 1,815 cases and 1,024,565 result observations, including rank zero and direct Flex32 inputs.
Checks include guards, zero packed tails, output-poison independence, source preservation, routing, queued ownership, and recoverable errors.
All earlier GPU regression gates also pass. Guest-compiled code matches the qualified host output for all three modules.

Linux CUDA's strict attempt stops at rank-zero readback. Excluding rank zero exposes the direct Flex32 upload rejection.
Excluding both domains passes 979,334 observations across F64, F32, F16, and BF16 with zero qualification failures.
The CUDA runner records fusion as enabled and materializes each chain step.
The failing strict attempts and passing filtered result remain separate records.

This checkpoint qualifies forward Tanh, Sin, and Cos. Broader typed autodiff, full backend coverage, and performance parity remain incomplete.

## Typed integer elementwise arithmetic

Protocol 42 adds 15 integer operations across I8, I16, I32, I64, U8, U16, U32, and U64.
The matching bundle contains 246 inference entries and 81 compute profiles.
All 245 earlier entries retain 530 identical instruction and constant sections.

The operations include addition, subtraction, multiplication, division, floor remainder, powers, negation, absolute value, sign, bitwise logic, and shifts.
Tensor, scalar, and unary inputs retain their actual integer format.
Matching I32 operations retain the existing fast path.
Other formats use a packed full writer with exact integer arithmetic and zero output tails.
No integer value passes through a floating-point intermediate.

Native arithmetic wraps at the destination width.
Division truncates toward zero, and floor remainder follows the divisor sign.
The portable Linux corpus excludes ordinary signed overflow, invalid divisions, invalid shifts, and signed minimum negation or absolute value.
Tensor powers use the low 32 exponent bits after signed narrow extension.
Mixed exponent formats and unsigned negation remain explicitly labeled Native extensions.
Scalar exponent two follows Burn's multiplication shortcut and its separate overflow domain.

Native backend scalar entry checks the received `Scalar` before dispatch, including empty outputs.
The Dispatch scalar-power override forwards that check instead of applying an inherited panicking I32 probe.
Invalid generic scalar arguments can still fail inside Burn's public `Scalar::new` before backend entry.
This extension therefore does not claim that every generic scalar argument returns a recoverable backend error.

D3OS passes 7,349 public cases, including 998 extension cases, with 2,765,465 exact output values and 10,001 source checks.
Guarded checks pass 460 raw dispatches, 468 fresh outputs, 804 Native results, 561 expected rejections, and 84 public results.
They cover signed views, broadcasts, ranks zero and eight, packed guards, zero tails, high integer bits, error recovery, and queued input drops.
All earlier regression gates pass, including the 52,306-case indexing corpus.
All three guest compiler hashes match their host audits.

The first GPU run stopped at a public test constructor that selected the default I32 dtype instead of its intended U8 dtype.
Every public fixture now requests its intended dtype explicitly, including mixed-format exponent tensors.
The complete rerun passes without a production arithmetic change.

Seven SDK tests, eight host reference tests, 21 Native unit tests, compiler audits, Clippy checks, and the production image build pass.
The Linux portable run stops at the pinned CubeCL rank-zero readback panic.
Its frozen executable, reference, runner, expected summary, and launcher hashes remain archived.
This failure does not establish complete Linux conformance or a numerical mismatch.
These are correctness checks. They establish neither complete backend support nor performance parity.
Typed reductions, scans, ordering, matrices, and arithmetic updates remain outside this extension.

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
  --boot-check --inference gpu-self-test --timeout 900
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

The final typed-index evidence lives under `target/nvidia/burn-backend-validation/indexed-final/`.
The first negative-fixture diagnostic remains under `indexed-first/`, including its passing public indexing marker and overall failure.
The Linux records use the `target/nvidia/burn-backend-audit/linux-indexed-` prefix.
The audit directory contains `traits-indexed-final.json`, `indexed-comparison.json`, and `indexed-section-comparison.json`.

Build the Linux indexed runner from its pinned toolchain directory:

```bash
cd tools/mnist-benchmark
cargo build --release --locked --bin indexed
cd ../..
```

Run the larger Linux diagnostic with the two retained upstream domains excluded:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-indexed-diagnostic \
  --burn-indexed tools/mnist-benchmark/target/release/indexed \
  --burn-indexed-no-flex32 --burn-indexed-skip-scalar-nd \
  --memory-mib 12288 --timeout 2400
```

For the complete Linux diagnostic, omit both domain flags and use a fresh directory.
The runner's `--expected-summary` command returns CPU metadata before device construction. It does not run GPU checks.

The typed integer evidence lives under `target/nvidia/burn-backend-validation/int-arithmetic-explicit-dtypes/`.
The initial fixture diagnostic remains under `int-arithmetic-runtime/`.
The Linux arithmetic records use the `target/nvidia/burn-backend-audit/linux-int-arithmetic-` prefix.
The audit directory contains `traits-int-arithmetic-final.json`, `int-arithmetic-comparison.json`, and `int-arithmetic-section-comparison.json`.

Build the portable arithmetic runner from its pinned toolchain directory:

```bash
cd tools/mnist-benchmark
cargo build --release --locked --bin integer_arithmetic
cd ../..
```

Run the complete portable Linux corpus in a fresh directory:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-integer-arithmetic \
  --burn-integer-arithmetic tools/mnist-benchmark/target/release/integer_arithmetic \
  --memory-mib 12288 --timeout 900
```

For a diagnostic subset, use `--burn-integer-arithmetic-dtype` or `--burn-integer-arithmetic-op`.
The runner's `--expected-summary` describes the exact selected corpus before device construction.

The typed float evidence lives under `target/nvidia/burn-backend-validation/float-arithmetic-sign-fixed/`.
The initial sign failure remains under `float-arithmetic-first/`, with its original source and portable manifest.
The Linux records use the `target/nvidia/burn-backend-audit/linux-float-arithmetic-` prefix.
The audit directory contains `traits-float-arithmetic-final.json`, `float-arithmetic-comparison.json`, and `float-arithmetic-section-comparison.json`.
The archived sign diagnosis includes `float-arithmetic-sign-lowering-diagnostic.json` in the local cache.
These records establish correctness within the stated domains. They contain no matched performance benchmark.

Build the typed float runner from its pinned toolchain directory:

```bash
cd tools/mnist-benchmark
cargo build --release --locked --bin float_arithmetic
cd ../..
```

Verify the deterministic fixtures and their independent local MPFR comparison:

```bash
python3 tools/nvidia/float-arithmetic-generate.py --check --mpfr
```

The `--mpfr` option requires the local MPFR library. The fixture generator has no Python package dependency.

Run the supported Linux diagnostic in a fresh directory:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-float-arithmetic \
  --burn-float-arithmetic tools/mnist-benchmark/target/release/float_arithmetic \
  --burn-float-arithmetic-skip-rank-zero --burn-float-arithmetic-no-flex32 \
  --memory-mib 12288 --timeout 900
```

For the complete Linux attempt, omit both diagnostic flags and use a fresh directory.
The additional `--burn-float-arithmetic-finite` option requires finite operands and finite expected results.
The dtype and operation filters select smaller diagnostic groups. Each selection has its own exact expected summary.

The typed rounding evidence lives under `target/nvidia/burn-backend-validation/float-rounding-first/`.
The Linux records use the `target/nvidia/burn-backend-audit/linux-float-rounding-` prefix.
The audit directory contains `float-rounding-comparison.json` and `float-rounding-section-comparison.json`.
These records preserve the complete D3OS suite and the explicitly restricted Linux diagnostic.

Build the typed rounding runner from its pinned toolchain directory:

```bash
cd tools/mnist-benchmark
cargo build --release --locked --bin float_rounding
cd ../..
```

Verify the deterministic fixtures and their independent local MPFR comparison:

```bash
python3 tools/nvidia/float-rounding-generate.py --check --mpfr
```

Run the supported Linux diagnostic in a fresh directory:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-float-rounding \
  --burn-float-rounding tools/mnist-benchmark/target/release/float_rounding \
  --burn-float-rounding-skip-rank-zero --burn-float-rounding-no-flex32 \
  --memory-mib 12288 --timeout 900
```

For the complete Linux attempt, omit both diagnostic flags and use a fresh directory.
The `--burn-float-rounding-finite` option requires finite inputs and finite expected results.
The `--burn-float-rounding-dtype` and `--burn-float-rounding-op` filters select smaller diagnostic groups.
The runner's `--expected-summary` reports exact selected counts before device construction.

The typed ReLU evidence lives under `target/nvidia/burn-backend-validation/typed-relu-first/`.
The Linux records use the `target/nvidia/burn-backend-audit/linux-typed-relu-` prefix.
The audit directory contains `typed-relu-comparison.json`, `typed-relu-source-baseline.json`, and `traits-typed-relu-final.json`.

Build the typed ReLU runner from its pinned toolchain directory:

```bash
cd tools/mnist-benchmark
cargo build --release --locked --bin relu
cd ../..
```

Run the supported Linux diagnostic in a fresh directory:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-typed-relu \
  --burn-relu tools/mnist-benchmark/target/release/relu \
  --burn-relu-skip-rank-zero --burn-relu-no-flex32 \
  --memory-mib 12288 --timeout 900
```

For the complete Linux attempt, omit both diagnostic flags and use a fresh directory.
The `--burn-relu-dtype` filter selects a smaller diagnostic group.
The runner's `--expected-summary` reports exact selected counts before device construction.

The typed remainder evidence lives under `target/nvidia/burn-backend-validation/typed-remainder-first/`.
The Linux records use the `target/nvidia/burn-backend-audit/linux-typed-remainder-` prefix.
The audit directory contains `typed-remainder-comparison.json` and `typed-remainder-section-comparison.json`.
The filtered Linux directory also contains `numerical-failure-summary.json`. Its complete diagnostics remain in `serial.log`.

Build the runner from its pinned toolchain directory:

```bash
cd tools/mnist-benchmark
cargo build --release --locked --bin typed_remainder
cd ../..
```

Verify the fixtures and the independent MPFR residue checks:

```bash
python3 tools/nvidia/typed-remainder-generate.py --check --mpfr
```

Reproduce the retained filtered Linux failure in a fresh directory:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-typed-remainder \
  --burn-typed-remainder tools/mnist-benchmark/target/release/typed_remainder \
  --burn-typed-remainder-skip-rank-zero --burn-typed-remainder-no-flex32 \
  --memory-mib 12288 --timeout 900
```

For the complete attempt, omit both diagnostic flags and use a fresh directory.
The `--burn-typed-remainder-finite` option requires finite effective operands and finite expected results.
The dtype and group filters select explicit diagnostic domains. A group selection omits scalar, geometry, and chain cases.
The runner's `--expected-summary` reports exact selected counts before device construction.

### Exp/log qualification

The final Native archive uses `target/nvidia/burn-backend-validation/typed-exp-log-final/`.
The first Native failure and the partial regression stop remain separate archives.
Linux archives use the `target/nvidia/burn-backend-audit/linux-typed-exp-log-` prefix.
The `v4` records retain the current launcher, executable, fixture, policy, and per-operation count identities.

Build the runner from its pinned toolchain directory:

```bash
cd tools/mnist-benchmark
cargo build --release --locked --bin typed_exp_log
cd ../..
```

Verify the interval reference fixtures:

```bash
python3 tools/nvidia/typed-exp-log-generate.py --check
```

Prepare a portable kernel bundle in a fresh output directory:

```bash
python3 tools/nvidia/prepare-kernels.py --audit \
  --output target/nvidia/burn-typed-exp-log-reproduce-kernels
```

Build the OS image:

```bash
cargo make --no-workspace --profile production image
```

Run the complete Native GPU checks:

```bash
python3 tools/nvidia/run-qemu.py \
  --firmware target/nvidia/firmware \
  --kernels target/nvidia/burn-typed-exp-log-reproduce-kernels \
  --memory-mib 4096 --boot-check --inference gpu-self-test --timeout 1200
```

Reproduce the retained Linux classification errors in another fresh directory:

```bash
python3 tools/nvidia/linux-reference.py \
  --directory target/nvidia/linux-burn-typed-exp-log-reproduce \
  --burn-typed-exp-log tools/mnist-benchmark/target/release/typed_exp_log \
  --burn-typed-exp-log-skip-rank-zero --burn-typed-exp-log-no-flex32 \
  --memory-mib 12288 --timeout 900
```

For the full Linux attempt, omit both diagnostic flags and use a fresh directory.

The finite filter retains finite inputs and expected outputs, including subnormal values.
The dtype and operation filters narrow the declared comparison domain.
The runner's `--expected-summary` reports selected counts before device creation.

### Trigonometric qualification

The Native archive uses `target/nvidia/burn-backend-validation/typed-trig-first/`.
Linux archives use the `target/nvidia/burn-backend-audit/linux-typed-trig-` prefix.
The `strict-first` and `no-rank-zero-first` records retain the upstream rejections.
The `no-rank-zero-no-flex32-first` record contains the passing filtered CUDA result.

Build the runner and verify its independent references:

```bash
cd tools/mnist-benchmark
cargo build --release --locked --bin typed_trig
cd ../..
python3 tools/nvidia/typed-trig-generate.py --check
python3 tools/nvidia/test-typed-trig.py
```

Use `prepare-kernels.py --audit` and the complete Native command above to test the current source.
Select `--burn-typed-trig tools/mnist-benchmark/target/release/typed_trig` for the Linux launcher.
The matching diagnostic flags are `--burn-typed-trig-skip-rank-zero` and `--burn-typed-trig-no-flex32`.
Omit both flags for the strict attempt. Use a fresh output directory for each attempt.
