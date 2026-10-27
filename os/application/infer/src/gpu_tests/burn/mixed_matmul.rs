//! Mixed matmul physical writers, typed graph boundaries, and resident lifetimes.
//! Host checks audit plans only. ROOT approves the exact freeze before GPU use.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{DType, Shape, TensorData, TensorMetadata, ops::FloatTensorOps, tensor::FloatTensor},
        tensor::try_read_sync,
    },
    native::{Native, NativeDevice, NativeTensor, layout::Layout},
    validation::mixed_matmul::{self as oracle, Case, Expected, Group, Matrix, Operation, Policy, Scope, Tracking},
};
use d3os_compute::{
    Backend as _, Error,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::{
    self as abi, Kernel,
    cast::Format,
    mixed_matmul::{self as raw, Kind},
    reduction::Reduction,
    storage::Mode,
};

type Result<T> = core::result::Result<T, String>;
const FROZEN: bool = true;
const SOURCE_PINS: &[(&str, &str)] = &[
    ("shared", "5d99a747cbcbdeed09ba5c5d01bc444ff607c8e4a9c6a4d3f4431a3ac4e3c4eb"),
    ("descriptor", "c4082b796538a782364e860658a816f547809aae0f6f9497961ea8a494c2526d"),
    ("fixture_blob", "c45e6e6eac8a1ee445a470318ed43576fb4652ed256bd4385d3b60b3a5e9266d"),
    ("generated_manifest", "9dc3fcfa1fedeae0e7489e53c5138a8bb18fb277f7521af93766f13487089c8a"),
    ("reference_policy", "e17e389ded361ae999a5747bd6aed4acbaa53d338316c6c2293eebe810874efe"),
];
const GUARD: u32 = 0x6bad_beef;
const POISONS: [u32; 2] = [0xdead_beef, 0x5ac3_9248];
const SCRATCH_POISON_MASKS: [u32; 2] = [0xa53c_9248, 0x5ac3_6db7];
const DTYPES: [DType; 3] = [DType::F16, DType::BF16, DType::Flex32];

fn gpu<T>(result: d3os_compute::Result<T>) -> Result<T> {
    result.map_err(|e| format!("mixed matmul: {e}"))
}
fn kind(dtype: DType) -> Kind {
    match dtype {
        DType::F16 => Kind::F16,
        DType::BF16 => Kind::BF16,
        DType::Flex32 => Kind::Flex32,
        _ => unreachable!("mixed dtype"),
    }
}
fn format_of(dtype: DType) -> Format {
    match dtype {
        DType::F16 => Format::F16,
        DType::BF16 => Format::BF16,
        DType::Flex32 => Format::Flex32,
        DType::F32 => Format::F32,
        _ => unreachable!("mixed storage"),
    }
}
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut word = [0; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(word)
        })
        .collect()
}
fn data(matrix: &Matrix) -> TensorData {
    TensorData::from_bytes_vec(matrix.values.bytes().to_vec(), matrix.values.shape.to_vec(), matrix.values.dtype)
}
fn public_tensor<const D: usize>(matrix: &Matrix, device: &d3os_ai::burn::tensor::Device) -> d3os_ai::burn::tensor::Tensor<D> {
    d3os_ai::burn::tensor::Tensor::from_data(data(matrix), (device, matrix.values.dtype)).autodiff()
}
fn fixture(id: &str) -> Result<&'static Case> {
    oracle::fixture(id).ok_or_else(|| format!("missing frozen row {id}"))
}
fn main_expected(case: &Case) -> Result<&Expected> {
    case.expectation("forward")
        .or_else(|| case.expectation("expected"))
        .ok_or_else(|| format!("{} output absent", case.id))
}
fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Work {
    kernels: [u64; abi::KERNEL_COUNT],
    services: [u64; abi::SERVICE_OPERATIONS],
}
impl Default for Work {
    fn default() -> Self {
        Self {
            kernels: [0; abi::KERNEL_COUNT],
            services: [0; abi::SERVICE_OPERATIONS],
        }
    }
}
impl Work {
    fn scope(&mut self) {
        self.services[abi::BEGIN_COMMANDS as usize] += 1;
        self.services[abi::END_COMMANDS as usize] += 1;
    }
    fn launch(&mut self, kernel: Kernel, fresh: bool) {
        self.kernels[kernel as usize] += 1;
        self.services[if fresh { abi::LAUNCH_ALLOCATE } else { abi::LAUNCH } as usize] += 1;
    }
    fn allocate(&mut self) {
        self.services[abi::ALLOCATE as usize] += 1;
        self.kernels[Kernel::Zero as usize] += 1;
    }
    fn free(&mut self, count: usize) {
        self.services[abi::FREE as usize] += count as u64;
    }
}
#[derive(Clone)]
struct Snapshot {
    kernels: [abi::ProfileEntry; abi::KERNEL_COUNT],
    services: [abi::ServiceEntry; abi::SERVICE_OPERATIONS],
}
impl Snapshot {
    fn take(backend: &Nvidia) -> Result<Self> {
        Ok(Self {
            kernels: gpu(backend.profile())?,
            services: gpu(backend.service_profile())?,
        })
    }
    fn check(&self, backend: &Nvidia, want: &Work) -> Result<()> {
        check_counters(self, &Self::take(backend)?, want)
    }
}
fn check_counters(before: &Snapshot, after: &Snapshot, want: &Work) -> Result<()> {
    for (i, ((old, new), &count)) in before.kernels.iter().zip(&after.kernels).zip(&want.kernels).enumerate() {
        if new.wait_cycles != old.wait_cycles {
            return Err(format!("unexpected standalone kernel wait kernel={i}"));
        }
        if new.calls.checked_sub(old.calls) != Some(count) {
            return Err(format!("kernel={i} expected={count} before={} after={}", old.calls, new.calls));
        }
    }
    for (i, ((old, new), &count)) in before.services.iter().zip(&after.services).zip(&want.services).enumerate() {
        let count = count + u64::from(i == abi::PROFILE as usize || i == abi::SERVICE_PROFILE as usize);
        if new.calls.checked_sub(old.calls) != Some(count) {
            return Err(format!("service={i} expected={count} before={} after={}", old.calls, new.calls));
        }
    }
    Ok(())
}
fn download(backend: &Nvidia, buffer: &GpuBuffer<u32>, len: usize) -> Result<Vec<u32>> {
    let before = Snapshot::take(backend)?;
    let mut actual = vec![0; len];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    let mut work = Work::default();
    work.services[abi::DOWNLOAD as usize] = u64::from(len != 0);
    before.check(backend, &work)?;
    Ok(actual)
}
fn native_words(backend: &Nvidia, tensor: &NativeTensor) -> Result<Vec<u32>> {
    let before = Snapshot::take(backend)?;
    let actual = gpu(tensor.read_storage_words())?;
    let mut work = Work::default();
    work.services[abi::DOWNLOAD as usize] = 1;
    before.check(backend, &work)?;
    Ok(actual)
}
fn check_guards(actual: &[u32], len: usize) -> Result<&[u32]> {
    if actual.len() != len + 2 || actual.first() != Some(&GUARD) || actual.last() != Some(&GUARD) {
        return Err(String::from("physical output guard/extent changed"));
    }
    Ok(&actual[1..len + 1])
}
fn compare_values(dtype: DType, policies: &[Policy], expected: &[u64], actual: &[u64]) -> Result<()> {
    if expected.len() != actual.len() || expected.len() != policies.len() {
        return Err(String::from("logical count/policy mismatch"));
    }
    for (i, ((&expected, &actual), &policy)) in expected.iter().zip(actual).zip(policies).enumerate() {
        if !oracle::compare(dtype, policy, expected, actual) {
            return Err(format!(
                "cell={i} dtype={dtype:?} policy={policy:?} actual={actual:#x} expected={expected:#x} class={:?}",
                oracle::classification(dtype, actual)
            ));
        }
    }
    Ok(())
}
fn check_output(expected: &Expected, actual: &[u32]) -> Result<()> {
    let bytes: Vec<_> = actual.iter().flat_map(|w| w.to_le_bytes()).collect();
    let byte_count = expected.values.len * expected.values.dtype.size();
    if bytes.len() != byte_count.div_ceil(4) * 4 {
        return Err(String::from("physical output word extent mismatch"));
    }
    if bytes[byte_count..].iter().any(|&b| b != 0) {
        return Err(String::from("original destination packed tail is not zero"));
    }
    let actual = oracle::unpack(expected.values.dtype, &bytes[..byte_count])?;
    compare_values(expected.values.dtype, expected.policies, &expected.values.bits(), &actual)
}

/// Identical inputs must not depend on prior destination contents. This is a
/// physical coherence check, independent of every numerical fixture policy.
/// NaN payloads and nonempty zero signs remain outside the result contract.
fn check_scratch_coherence(baseline: &[u32], actual: &[u32]) -> Result<()> {
    if baseline.len() != actual.len() {
        return Err(String::from("scratch coherence word count mismatch"));
    }
    for (i, (&baseline, &actual)) in baseline.iter().zip(actual).enumerate() {
        let same = match oracle::classification(DType::F32, u64::from(baseline)) {
            oracle::Class::NaN => oracle::classification(DType::F32, u64::from(actual)) == oracle::Class::NaN,
            oracle::Class::Zero => oracle::classification(DType::F32, u64::from(actual)) == oracle::Class::Zero,
            _ => baseline == actual,
        };
        if !same {
            return Err(format!(
                "scratch destination-dependent word={i} baseline={baseline:#010x} actual={actual:#010x}"
            ));
        }
    }
    Ok(())
}
fn scratch_poisons(baseline: &[u32], mask: u32) -> Vec<u32> {
    baseline.iter().map(|&word| word ^ mask).collect()
}
fn check_scratch_overwrite(baseline: &[u32], poisons: &[u32], actual: &[u32]) -> Result<()> {
    if baseline.len() != poisons.len() || baseline.len() != actual.len() {
        return Err(String::from("scratch overwrite word count mismatch"));
    }
    for (i, (&poison, &actual)) in poisons.iter().zip(actual).enumerate() {
        if actual == poison {
            return Err(format!("scratch word={i} retained destination poison={poison:#010x}"));
        }
    }
    check_scratch_coherence(baseline, actual)
}
fn packed_element(words: &[u32], index: usize) -> u16 {
    (words[index / 2] >> ((index % 2) * 16)) as u16
}
fn check_packed_storage(dtype: DType, count: usize, words: &[u32]) -> Result<()> {
    if !matches!(dtype, DType::F16 | DType::BF16) || words.len() != count.div_ceil(2) {
        return Err(String::from("packed physical extent/dtype mismatch"));
    }
    if count % 2 == 1 && words.last().is_some_and(|word| word >> 16 != 0) {
        return Err(String::from("original packed cast unused tail is not zero"));
    }
    Ok(())
}
/// Storage identity is checked per live element. NaN payloads and nonempty
/// zero signs may vary, but repairing the unused tail cannot hide a live store.
fn check_packed_coherence(dtype: DType, count: usize, baseline: &[u32], actual: &[u32]) -> Result<()> {
    check_packed_storage(dtype, count, baseline)?;
    check_packed_storage(dtype, count, actual)?;
    for i in 0..count {
        let baseline = packed_element(baseline, i);
        let actual = packed_element(actual, i);
        let same = match oracle::classification(dtype, u64::from(baseline)) {
            oracle::Class::NaN => oracle::classification(dtype, u64::from(actual)) == oracle::Class::NaN,
            oracle::Class::Zero => oracle::classification(dtype, u64::from(actual)) == oracle::Class::Zero,
            _ => baseline == actual,
        };
        if !same {
            return Err(format!(
                "packed cast destination-dependent element={i} baseline={baseline:#06x} actual={actual:#06x}"
            ));
        }
    }
    Ok(())
}
fn check_packed_overwrite(dtype: DType, count: usize, baseline: &[u32], poisons: &[u32], actual: &[u32]) -> Result<()> {
    check_packed_storage(dtype, count, baseline)?;
    check_packed_storage(dtype, count, actual)?;
    if poisons.len() != baseline.len() {
        return Err(String::from("packed poison word count mismatch"));
    }
    for i in 0..count {
        let poison = packed_element(poisons, i);
        if packed_element(actual, i) == poison {
            return Err(format!("packed cast element={i} retained destination poison={poison:#06x}"));
        }
    }
    check_packed_coherence(dtype, count, baseline, actual)
}
fn check_source_words(expected: &[u32], actual: &[u32]) -> Result<()> {
    if expected != actual {
        return Err(String::from("original raw source/guard/sibling changed"));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct RawCase {
    case: &'static Case,
    tile: bool,
    lhs_slot: usize,
    rhs_slot: usize,
}
impl RawCase {
    fn matrix(&self, lhs: bool) -> Result<&Matrix> {
        let name = match (self.tile, lhs) {
            (true, true) => "tile.lhs",
            (true, false) => "tile.rhs",
            (false, true) => "lhs",
            (false, false) => "rhs",
        };
        if self.tile {
            self.case.raw_matrices.iter().find(|m| m.name == name)
        } else {
            self.case.matrix(name)
        }
        .ok_or_else(|| format!("{} matrix {name} absent", self.case.id))
    }
    fn expected(&self, scratch: bool) -> Result<&Expected> {
        if self.tile {
            self.case
                .raw_expected
                .iter()
                .find(|e| e.name == if scratch { "tile.scratch" } else { "tile.output" })
        } else if scratch {
            oracle::scratch_expectation(self.case)
        } else {
            self.case.expectation("expected")
        }
        .ok_or_else(|| format!("{} raw expected absent tile={} scratch={scratch}", self.case.id, self.tile))
    }
    fn dimensions(&self) -> Result<[usize; 3]> {
        let a = self.matrix(true)?.values.shape;
        let b = self.matrix(false)?.values.shape;
        Ok([a[a.len() - 2], a[a.len() - 1], b[b.len() - 1]])
    }
    fn batches(&self) -> Result<usize> {
        let e = self.expected(false)?.values.shape;
        Ok(product(&e[..e.len() - 2]))
    }
    fn parameters(&self, batch: usize) -> Result<[usize; 5]> {
        let [m, k, n] = self.dimensions()?;
        let packed = self.case.dtype != Some(DType::Flex32);
        Ok([
            m,
            k,
            n,
            if packed { (self.lhs_slot + batch * m * k) % 2 } else { 0 },
            if packed { (self.rhs_slot + batch * k * n) % 2 } else { 0 },
        ])
    }
    fn input_bits(&self, lhs: bool) -> Result<Vec<u64>> {
        let m = self.matrix(lhs)?;
        let [rows, k, cols] = self.dimensions()?;
        let count = if lhs { rows * k } else { k * cols };
        let batch = self.batches()?;
        let values = m.values.bits();
        if values.len() == batch * count {
            return Ok(values);
        }
        if values.len() == count {
            return Ok((0..batch).flat_map(|_| values.iter().copied()).collect());
        }
        Err(String::from("raw selection requires dense equal batch or whole-matrix broadcast"))
    }
    fn source_words(&self, lhs: bool) -> Result<Vec<u32>> {
        let dtype = self.case.dtype.ok_or("raw dtype missing")?;
        let packed = dtype != DType::Flex32;
        let slot = if lhs { self.lhs_slot } else { self.rhs_slot };
        let poison = match dtype {
            DType::F16 => 0x7e35,
            DType::BF16 => 0x7fc5,
            _ => 0x7fc5_a53c,
        };
        let mut bits = vec![poison; if packed { slot } else { 0 }];
        bits.extend(self.input_bits(lhs)?);
        bits.push(poison);
        bits.push(poison);
        let mut output = vec![GUARD];
        output.extend(words(&oracle::pack(dtype, &bits)));
        output.push(GUARD);
        Ok(output)
    }
    fn input_offset(&self, lhs: bool, batch: usize) -> Result<usize> {
        let [m, k, n] = self.dimensions()?;
        let base = if lhs { self.lhs_slot + batch * m * k } else { self.rhs_slot + batch * k * n };
        Ok(1 + if self.case.dtype == Some(DType::Flex32) { base } else { base / 2 })
    }
}
fn raw_plan() -> Vec<RawCase> {
    let suffixes = [
        "dense_odd",
        "rank_eight",
        "exact_finite_dyadic",
        "destination_ties_both_parities_and_sides",
        "cancellation_order",
        "K32_boundary_cancellation",
        "minimum_subnormal_input",
        "accumulator_or_output_underflow",
        "live_nan_inf_and_masked_padding",
        "operand_range_and_output_overflow",
    ];
    let mut plan = Vec::new();
    for (prefix, dtype) in ["f16", "bf16", "flex32"].into_iter().zip(DTYPES) {
        for (i, suffix) in suffixes.into_iter().enumerate() {
            let id = format!("{prefix}_{suffix}");
            let case = oracle::fixture(&id).expect("frozen raw base row");
            plan.push(RawCase {
                case,
                tile: false,
                lhs_slot: usize::from(dtype != DType::Flex32 && i % 2 == 0),
                rhs_slot: usize::from(dtype != DType::Flex32 && i % 3 != 0),
            });
        }
        let case = oracle::fixture(&format!("{prefix}_dense_odd")).expect("tile parent");
        plan.push(RawCase {
            case,
            tile: true,
            lhs_slot: usize::from(dtype != DType::Flex32),
            rhs_slot: usize::from(dtype != DType::Flex32),
        });
    }
    plan
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Counts {
    raw_cases: usize,
    guarded_scratch: usize,
    fresh_scratch: usize,
    guarded_cast: usize,
    fresh_cast: usize,
    native_outputs: usize,
    mandatory_values: usize,
    diagnostic_values: usize,
    source_words: usize,
    rejections: usize,
    empty_aliases: usize,
    zero_k: usize,
    raw_zero_k: usize,
    deferred: usize,
    queued: usize,
    autodiff: usize,
}
const EXPECTED: Counts = Counts {
    raw_cases: 33,
    guarded_scratch: 66,
    fresh_scratch: 66,
    guarded_cast: 44,
    fresh_cast: 44,
    native_outputs: 342,
    mandatory_values: 72414,
    diagnostic_values: 71,
    source_words: 45116,
    rejections: 83,
    empty_aliases: 9,
    zero_k: 3,
    raw_zero_k: 3,
    deferred: 15,
    queued: 3,
    autodiff: 21,
};
fn frozen_shared_summary() -> oracle::Summary {
    oracle::Summary {
        selected_rows: 93,
        required_routes: 1380,
        diagnostic_routes: 14,
        required_values: 143013,
        diagnostic_values: 14,
        model_values: 8,
        source_values: 72552,
        gradient_values: 3863,
        consumer_routes: 141,
        external_gates: 0,
        geometry_rows: 39,
        numeric_rows: 24,
        gradient_rows: 30,
        native_extension_rows: 0,
        preservation_rows: 0,
        f64_rows: 0,
        f16_values: 47701,
        bf16_values: 47707,
        flex32_values: 47619,
        f64_values: 0,
    }
}
#[derive(Default)]
struct Observed {
    counts: Counts,
    failed: usize,
    examples: Vec<String>,
}
impl Observed {
    fn record(&mut self, context: &str, result: Result<()>) {
        if let Err(e) = result {
            self.failed += 1;
            if self.examples.len() < 12 {
                self.examples.push(format!("{context}: {e}"));
            }
        }
    }
}
fn count_values(counts: &mut Counts, e: &Expected) {
    let diagnostics = e.policies.iter().filter(|&&p| p == Policy::Observation).count();
    counts.mandatory_values += e.values.len - diagnostics;
    counts.diagnostic_values += diagnostics;
}
fn raw_work(case: &RawCase, fresh: bool, cast: bool) -> Result<Work> {
    let mut w = Work::default();
    w.scope();
    if cast {
        w.launch(Kernel::TensorCast, fresh);
    } else {
        for _ in 0..case.batches()? {
            w.launch(kind(case.case.dtype.unwrap()).kernel(), false);
        }
    }
    Ok(w)
}
fn cast_parameters(count: usize, dtype: DType) -> Result<[usize; 32]> {
    let layout = gpu(Layout::contiguous(Shape::new([count])))?;
    let mut p = gpu(layout.storage_parameters(Format::F32.width(), Mode::Copy, 0))?;
    p[2] = Format::F32 as usize | ((format_of(dtype) as usize) << 8);
    Ok(p)
}
fn dispatch_product(backend: &Nvidia, case: &RawCase, a: &GpuBuffer<u32>, b: &GpuBuffer<u32>, out: &GpuBuffer<u32>, offset: usize) -> d3os_compute::Result<()> {
    let [m, _, n] = case.dimensions().map_err(|_| Error::InvalidParameter)?;
    for batch in 0..case.batches().map_err(|_| Error::InvalidParameter)? {
        backend.dispatch_words(
            kind(case.case.dtype.unwrap()).kernel(),
            &[
                (a, case.input_offset(true, batch).map_err(|_| Error::InvalidParameter)?),
                (b, case.input_offset(false, batch).map_err(|_| Error::InvalidParameter)?),
                (out, offset + batch * m * n),
            ],
            &case.parameters(batch).map_err(|_| Error::InvalidParameter)?,
        )?;
    }
    Ok(())
}
fn emit_scratch(case: &RawCase, route: &str, expected: &Expected, actual: &[u32]) {
    if case.case.group != Group::Numeric {
        return;
    }
    let target = expected.values.bits();
    for (i, (&word, &target)) in actual.iter().zip(&target).enumerate() {
        let a = f32::from_bits(word);
        let e = f32::from_bits(target as u32);
        let ideal = case
            .case
            .numeric
            .get(i)
            .and_then(|c| {
                if let Some((n, d)) = c.ideal_rational.split_once('/') {
                    Some(n.parse::<f64>().ok()? / d.parse::<f64>().ok()?)
                } else {
                    c.ideal_rational.parse::<f64>().ok()
                }
            })
            .filter(|v| v.is_finite());
        let absolute = ideal.filter(|_| a.is_finite()).map(|e| (f64::from(a) - e).abs());
        let relative = absolute.zip(ideal).and_then(|(error, e)| if e == 0.0 { None } else { Some(error / e.abs()) });
        let order = |bits: u32| if bits >> 31 == 0 { bits | 0x8000_0000 } else { !bits };
        let ulp = if a.is_finite() && e.is_finite() {
            Some(order(word).abs_diff(order(target as u32)))
        } else {
            None
        };
        crate::report(format_args!(
            "NVIDIA_BURN_MIXED_MATMUL_RAW_NUMERIC case={} route={route} stage={} cell={i} dtype=F32 actual={word:#010x} expected={target:#010x} class={:?} policy={:?} absolute_error={absolute:?} relative_error={relative:?} storage_ulp={ulp:?} conditional_model={}",
            case.case.id,
            expected.stage,
            oracle::classification(DType::F32, u64::from(word)),
            expected.policies[i],
            oracle::MODEL_LABEL
        ));
    }
}
fn raw_case(backend: &Nvidia, case: &RawCase, counts: &mut Counts) -> Result<()> {
    let dtype = case.case.dtype.ok_or("raw dtype missing")?;
    let a_words = case.source_words(true)?;
    let b_words = case.source_words(false)?;
    let a = gpu(backend.upload_words(&a_words))?;
    let b = gpu(backend.upload_words(&b_words))?;
    let scratch = case.expected(true)?;
    let output = case.expected(false)?;
    let count = scratch.values.len;
    let mut previous_fresh: Option<Vec<u32>> = None;
    let mut previous_cast: Option<Vec<u32>> = None;
    for (poison, mask) in POISONS.into_iter().zip(SCRATCH_POISON_MASKS) {
        let dirty = gpu(backend.upload_words(&vec![poison; count]))?;
        drop(dirty);
        let before = Snapshot::take(backend)?;
        // A single raw full writer can use LAUNCH_ALLOCATE; batched writers
        // intentionally share an explicitly allocated concatenated allocation.
        let fresh = if case.batches()? == 1 {
            gpu(backend.with_commands(|| {
                backend.produce_words(
                    count,
                    kind(dtype).kernel(),
                    &[
                        (&a, case.input_offset(true, 0).map_err(|_| Error::InvalidParameter)?),
                        (&b, case.input_offset(false, 0).map_err(|_| Error::InvalidParameter)?),
                    ],
                    &case.parameters(0).map_err(|_| Error::InvalidParameter)?,
                )
            }))?
        } else {
            gpu(backend.with_commands(|| {
                let out = backend.allocate_words(count)?;
                dispatch_product(backend, case, &a, &b, &out, 0)?;
                Ok::<_, Error>(out)
            }))?
        };
        let mut work = raw_work(case, false, false)?;
        if case.batches()? == 1 {
            work.services[abi::LAUNCH as usize] = 0;
            work.services[abi::LAUNCH_ALLOCATE as usize] = 1;
        } else {
            work.allocate();
        }
        before.check(backend, &work)?;
        let baseline = download(backend, &fresh, count)?;
        emit_scratch(case, "fresh", scratch, &baseline);
        // Frozen fixture policies remain the sole numerical acceptance oracle.
        check_output(scratch, &baseline)?;
        if let Some(previous) = &previous_fresh {
            check_scratch_coherence(previous, &baseline)?;
        }
        counts.fresh_scratch += 1;
        count_values(counts, scratch);
        // Fresh bytes select physical poisons only. No value from this
        // allocation becomes an expected numerical result or widens a policy.
        let poisons = scratch_poisons(&baseline, mask);
        let mut initial = vec![GUARD];
        initial.extend_from_slice(&poisons);
        initial.push(GUARD);
        let out = gpu(backend.upload_words(&initial))?;
        let before = Snapshot::take(backend)?;
        gpu(backend.with_commands(|| dispatch_product(backend, case, &a, &b, &out, 1)))?;
        before.check(backend, &raw_work(case, false, false)?)?;
        let actual = download(backend, &out, count + 2)?;
        let actual = check_guards(&actual, count)?;
        emit_scratch(case, "guarded", scratch, actual);
        check_scratch_overwrite(&baseline, &poisons, actual)?;
        check_output(scratch, actual)?;
        counts.guarded_scratch += 1;
        count_values(counts, scratch);
        previous_fresh = Some(baseline);
        if dtype != DType::Flex32 {
            let p = cast_parameters(count, dtype)?;
            let output_count = count.div_ceil(2);
            let dirty = gpu(backend.upload_words(&vec![poison; output_count]))?;
            drop(dirty);
            let before = Snapshot::take(backend)?;
            let fresh_cast = gpu(backend.with_commands(|| backend.produce_words(output_count, Kernel::TensorCast, &[(&fresh, 0)], &p)))?;
            before.check(backend, &raw_work(case, true, true)?)?;
            let cast_baseline = download(backend, &fresh_cast, output_count)?;
            // Fresh still passes the independent frozen numerical oracle.
            // Its bits select physical poisons, never expected MMA values.
            check_output(output, &cast_baseline)?;
            if let Some(previous) = &previous_cast {
                check_packed_coherence(dtype, count, previous, &cast_baseline)?;
            }
            counts.fresh_cast += 1;
            count_values(counts, output);
            // Both half masks change every byte and toggle exponent bits for
            // F16 and BF16. NaN/Inf/zero baselines therefore get finite, nonzero
            // poison halves: legal NaN payload/zero-sign changes cannot collide.
            let poisons = scratch_poisons(&cast_baseline, mask);
            let mut initial = vec![GUARD];
            initial.extend_from_slice(&poisons);
            initial.push(GUARD);
            let cast = gpu(backend.upload_words(&initial))?;
            let before = Snapshot::take(backend)?;
            // One whole-result writer owns row/batch borders and the zero tail.
            gpu(backend.with_commands(|| backend.dispatch_words(Kernel::TensorCast, &[(&fresh, 0), (&cast, 1)], &p)))?;
            before.check(backend, &raw_work(case, false, true)?)?;
            let actual = download(backend, &cast, output_count + 2)?;
            let actual = check_guards(&actual, output_count)?;
            check_packed_overwrite(dtype, count, &cast_baseline, &poisons, actual)?;
            check_output(output, actual)?;
            counts.guarded_cast += 1;
            count_values(counts, output);
            previous_cast = Some(cast_baseline);
        }
        for (buffer, expected) in [(&a, &a_words), (&b, &b_words)] {
            check_source_words(expected, &download(backend, buffer, expected.len())?)?;
            counts.source_words += expected.len();
        }
    }
    counts.raw_cases += 1;
    Ok(())
}

fn matrix_layout(matrix: &Matrix) -> Result<Layout> {
    if let Some(v) = matrix.view {
        gpu(Layout::from_parts(
            Shape::from(matrix.values.shape.to_vec()),
            v.strides.iter().map(|&s| s as i64).collect(),
            v.base as i64,
            v.root.len,
        ))
    } else {
        gpu(Layout::contiguous(Shape::from(matrix.values.shape.to_vec())))
    }
}
fn native_matrix(matrix: &Matrix) -> Result<(NativeTensor, NativeTensor)> {
    let root = matrix.view.map_or(matrix.values, |v| v.root);
    let source = gpu(NativeTensor::from_data(TensorData::from_bytes_vec(
        root.bytes().to_vec(),
        root.shape.to_vec(),
        root.dtype,
    )))?;
    let tensor = gpu(source.with_layout(matrix_layout(matrix)?))?;
    Ok((source, tensor))
}
fn matrix_contiguous(layout: &Layout) -> bool {
    let rank = layout.shape().num_dims();
    let shape = layout.shape();
    (shape[rank - 2] == 1 || layout.strides()[rank - 2] == shape[rank - 1] as i64) && (shape[rank - 1] == 1 || layout.strides()[rank - 1] == 1)
}
/// Independent host inventory of Native's allocation, writer, and materialization
/// boundaries. It uses public layout/SDK contracts, never measured counters.
fn matmul_work(dtype: DType, a: &Layout, b: &Layout) -> Result<Work> {
    let shape = gpu(d3os_ai::native::layout::matmul_shape(a.shape(), b.shape()))?;
    let mut work = Work::default();
    if shape.as_slice().contains(&0) {
        return Ok(work);
    }
    work.scope();
    work.allocate();
    let rank = shape.num_dims();
    let k = a.shape()[a.shape().num_dims() - 1];
    if k == 0 {
        return Ok(work);
    }
    for layout in [a, b] {
        if !matrix_contiguous(layout) {
            work.launch(
                if dtype == DType::Flex32 {
                    Kernel::TensorElementwise
                } else {
                    Kernel::TensorStorage
                },
                true,
            );
            work.free(1);
        }
    }
    for _ in 0..product(&shape[..rank - 2]) {
        work.launch(kind(dtype).kernel(), false);
    }
    if dtype != DType::Flex32 {
        work.launch(Kernel::TensorCast, true);
        work.free(1);
    }
    Ok(work)
}
fn inspect_native(backend: &Nvidia, tensor: &NativeTensor, expected: &Expected, counts: &mut Counts) -> Result<()> {
    if tensor.dtype() != expected.values.dtype || tensor.shape().as_slice() != expected.values.shape {
        return Err(format!(
            "native metadata {:?}/{:?} != {:?}/{:?}",
            tensor.dtype(),
            tensor.shape(),
            expected.values.dtype,
            expected.values.shape
        ));
    }
    if !expected.values.shape.contains(&0) {
        check_output(expected, &native_words(backend, tensor)?)?;
    }
    counts.native_outputs += 1;
    count_values(counts, expected);
    Ok(())
}
fn source_identity(backend: &Nvidia, matrix: &Matrix, root: &NativeTensor, counts: &mut Counts) -> Result<()> {
    let values = matrix.view.map_or(matrix.values, |v| v.root);
    let expected = words(values.bytes());
    let actual = native_words(backend, root)?;
    // Empty Native sources have one driver placeholder; it is allocator zero.
    let expected = if expected.is_empty() { vec![0] } else { expected };
    if actual != expected {
        return Err(format!("{} original source bytes/sibling changed", matrix.name));
    }
    counts.source_words += actual.len();
    Ok(())
}
fn native_case(backend: &Nvidia, case: &Case, tile: bool, counts: &mut Counts) -> Result<()> {
    let (a, b, expected) = if tile {
        (
            case.raw_matrices.iter().find(|m| m.name == "tile.lhs").ok_or("tile lhs absent")?,
            case.raw_matrices.iter().find(|m| m.name == "tile.rhs").ok_or("tile rhs absent")?,
            case.raw_expected.iter().find(|e| e.name == "tile.output").ok_or("tile expected absent")?,
        )
    } else {
        (
            case.matrix("lhs").ok_or("lhs absent")?,
            case.matrix("rhs").ok_or("rhs absent")?,
            main_expected(case)?,
        )
    };
    let (ar, a_tensor) = native_matrix(a)?;
    let (br, b_tensor) = native_matrix(b)?;
    let saved_a = a_tensor.clone();
    let saved_b = b_tensor.clone();
    source_identity(backend, a, &ar, counts)?;
    source_identity(backend, b, &br, counts)?;
    let before = Snapshot::take(backend)?;
    let output = gpu(backend.record_commands(|| a_tensor.matmul(&b_tensor)))?;
    let mut work = matmul_work(a.values.dtype, a_tensor.layout(), b_tensor.layout())?;
    work.scope();
    before.check(backend, &work)?;
    inspect_native(backend, &output, expected, counts)?;
    // Reading aliases does not materialize or cast their original backing.
    source_identity(backend, a, &saved_a, counts)?;
    source_identity(backend, b, &saved_b, counts)?;
    drop(output);
    drop(a_tensor);
    drop(b_tensor);
    drop(saved_a);
    drop(saved_b);
    if !ar.can_mut() || !br.can_mut() {
        return Err(String::from("source did not become unique after output/view drops"));
    }
    source_identity(backend, a, &ar, counts)?;
    source_identity(backend, b, &br, counts)?;
    Ok(())
}
fn checked_matmul(backend: &Nvidia, a: &NativeTensor, b: &NativeTensor, expected: &Expected, counts: &mut Counts) -> Result<NativeTensor> {
    let before = Snapshot::take(backend)?;
    let output = gpu(a.matmul(b))?;
    before.check(backend, &matmul_work(a.dtype(), a.layout(), b.layout())?)?;
    inspect_native(backend, &output, expected, counts)?;
    Ok(output)
}
fn checked_sum(backend: &Nvidia, a: &NativeTensor, axis: usize, expected: &Expected, counts: &mut Counts) -> Result<NativeTensor> {
    let before = Snapshot::take(backend)?;
    let output = gpu(backend.record_commands(|| a.reduce(axis, Reduction::Sum)))?;
    let mut work = Work::default();
    work.scope();
    if expected.values.shape.contains(&0) {
        work.allocate();
    } else {
        work.launch(Kernel::FloatReduction, true);
    }
    before.check(backend, &work)?;
    inspect_native(backend, &output, expected, counts)?;
    Ok(output)
}
fn checked_add(backend: &Nvidia, a: &NativeTensor, b: &NativeTensor, expected: &Expected, counts: &mut Counts) -> Result<NativeTensor> {
    let before = Snapshot::take(backend)?;
    let output = gpu(backend.record_commands(|| a.binary(b, d3os_nvidia::compute::elementwise::Elementwise::Add)))?;
    let mut work = Work::default();
    work.scope();
    if !expected.values.shape.contains(&0) {
        work.launch(Kernel::FloatArithmetic, true);
    }
    before.check(backend, &work)?;
    inspect_native(backend, &output, expected, counts)?;
    Ok(output)
}
fn transpose(t: &NativeTensor) -> Result<NativeTensor> {
    let rank = t.shape().num_dims();
    gpu(t.with_layout(gpu(t.layout().swap_dims(rank - 2, rank - 1))?))
}
fn native_gradients(backend: &Nvidia, case: &Case, counts: &mut Counts) -> Result<()> {
    if case.operation == Operation::Linear {
        return native_linear(backend, case, counts);
    }
    let a = case.matrix("lhs").ok_or("gradient lhs absent")?;
    let b = case.matrix("rhs").ok_or("gradient rhs absent")?;
    let g = case.matrix("upstream").ok_or("upstream absent")?;
    let (ar, a) = native_matrix(a)?;
    let (br, b) = native_matrix(b)?;
    let (gr, g) = native_matrix(g)?;
    let mut outputs: Vec<(&str, NativeTensor)> = Vec::new();
    let y = checked_matmul(backend, &a, &b, main_expected(case)?, counts)?;
    outputs.push(("forward", y));
    for (name, left, right, target) in [("dA", g.clone(), transpose(&b)?, a.shape()), ("dB", transpose(&a)?, g.clone(), b.shape())] {
        let before_name = format!("{name}_before_sum");
        let expected = case.expectation(&before_name).ok_or("stored gradient missing")?;
        let mut grad = checked_matmul(backend, &left, &right, expected, counts)?;
        // Keeping each stored stage alive makes FREE ownership unambiguous.
        outputs.push((expected.name, grad.clone()));
        for axis in 0..target.num_dims() {
            if grad.shape()[axis] != target[axis] {
                let label = format!("{name}_axes.{axis}");
                let expected = case.expectation(&label).ok_or("axis gradient missing")?;
                grad = checked_sum(backend, &grad, axis, expected, counts)?;
                outputs.push((expected.name, grad.clone()));
            }
        }
        inspect_native(backend, &grad, case.expectation(name).ok_or("final gradient absent")?, counts)?;
        outputs.push((name, grad));
    }
    if case.tracking == Tracking::Shared {
        let a = &outputs.iter().find(|(n, _)| *n == "dA").unwrap().1;
        let b = &outputs.iter().find(|(n, _)| *n == "dB").unwrap().1;
        let expected = case.expectation("shared_source_gradient").ok_or("shared join absent")?;
        let joined = checked_add(backend, a, b, expected, counts)?;
        outputs.push((expected.name, joined));
    }
    drop(a);
    drop(b);
    drop(g);
    for consumer in case.consumers {
        let output = &outputs.iter().find(|(name, _)| *name == consumer.input).ok_or("last owner consumer absent")?.1;
        let _ = checked_sum(backend, output, consumer.axis, &consumer.expected, counts)?;
    }
    for (name, root) in [("lhs", &ar), ("rhs", &br), ("upstream", &gr)] {
        source_identity(backend, case.matrix(name).ok_or("gradient original source")?, root, counts)?;
    }
    Ok(())
}
fn linear_work(dtype: DType, input: &Layout, weight: &Layout, bias: bool) -> Result<Work> {
    let shape = input.shape();
    let flat = Shape::new([product(&shape[..shape.num_dims() - 1]), shape[shape.num_dims() - 1]]);
    let mut work = Work::default();
    if flat[0] == 0 || weight.shape()[1] == 0 {
        return Ok(work);
    }
    work.scope();
    let layout = gpu(Layout::contiguous(flat))?;
    if !input.is_contiguous() {
        work.launch(
            if dtype == DType::Flex32 {
                Kernel::TensorElementwise
            } else {
                Kernel::TensorStorage
            },
            true,
        );
        work.free(1);
    }
    let product = matmul_work(dtype, &layout, weight)?;
    for (i, &v) in product.kernels.iter().enumerate() {
        work.kernels[i] += v;
    }
    for (i, &v) in product.services.iter().enumerate() {
        work.services[i] += v;
    }
    if bias {
        work.launch(Kernel::FloatArithmetic, true);
        work.free(1);
    }
    Ok(work)
}
fn native_linear(backend: &Nvidia, case: &Case, counts: &mut Counts) -> Result<()> {
    let xm = case.matrix("input").ok_or("linear input absent")?;
    let wm = case.matrix("weight").ok_or("linear weight absent")?;
    let bm = case.matrix("bias").ok_or("linear bias absent")?;
    let gm = case.matrix("upstream").ok_or("linear upstream absent")?;
    let (xr, x) = native_matrix(xm)?;
    let (wr, w) = native_matrix(wm)?;
    let (br, b) = native_matrix(bm)?;
    let (gr, g) = native_matrix(gm)?;
    let before = Snapshot::take(backend)?;
    let y = gpu(x.linear(&w, Some(&b)))?;
    before.check(backend, &linear_work(x.dtype(), x.layout(), w.layout(), true)?)?;
    inspect_native(backend, &y, main_expected(case)?, counts)?;
    let wt = transpose(&w)?;
    let before = Snapshot::take(backend)?;
    let dx = gpu(g.linear(&wt, None))?;
    before.check(backend, &linear_work(g.dtype(), g.layout(), wt.layout(), false)?)?;
    inspect_native(backend, &dx, case.expectation("dx").ok_or("dx absent")?, counts)?;
    let xshape = x.shape();
    let gshape = g.shape();
    let rows = product(&xshape[..xshape.num_dims() - 1]);
    let xf = gpu(x.clone().reshape(Shape::new([rows, xshape[xshape.num_dims() - 1]])))?;
    let gf = gpu(g.clone().reshape(Shape::new([rows, gshape[gshape.num_dims() - 1]])))?;
    let dw = checked_matmul(backend, &transpose(&xf)?, &gf, case.expectation("dW").ok_or("dW absent")?, counts)?;
    // Bias backward keeps one axis and reshapes only metadata.
    let before = Snapshot::take(backend)?;
    let db = gpu(backend.record_commands(|| gf.reduce(0, Reduction::Sum)))?;
    let db = gpu(db.reshape(Shape::new([gshape[gshape.num_dims() - 1]])))?;
    let mut work = Work::default();
    work.scope();
    if gshape[gshape.num_dims() - 1] == 0 {
        work.allocate();
    } else {
        work.launch(Kernel::FloatReduction, true);
    }
    before.check(backend, &work)?;
    inspect_native(backend, &db, case.expectation("db").ok_or("db absent")?, counts)?;
    for c in case.consumers {
        let tensor = match c.input {
            "forward" => &y,
            "dx" => &dx,
            "dW" => &dw,
            "db" => &db,
            _ => return Err(String::from("linear consumer mismatch")),
        };
        let _ = checked_sum(backend, tensor, c.axis, &c.expected, counts)?;
    }
    for (m, root) in [(xm, &xr), (wm, &wr), (bm, &br), (gm, &gr)] {
        source_identity(backend, m, root, counts)?;
    }
    Ok(())
}

fn bad_parameters() -> Vec<Vec<usize>> {
    let good = [3, 5, 7, 0, 0];
    let mut out = vec![good[..4].to_vec(), vec![3, 5, 7, 0, 0, 0]];
    for (field, value) in [
        (0, 0),
        (2, 0),
        (0, i32::MAX as usize + 1),
        (1, i32::MAX as usize + 1),
        (2, i32::MAX as usize + 1),
        (3, 2),
        (4, 2),
        (0, i32::MAX as usize),
        (1, i32::MAX as usize),
    ] {
        let mut p = good;
        p[field] = value;
        out.push(p.to_vec());
    }
    out.push(vec![i32::MAX as usize, 0, 1, 0, 0]);
    out.push(vec![3, 0, 7, 1, 0]);
    out
}
fn rejections(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    let host = vec![GUARD; 256];
    let a = gpu(backend.upload_words(&host))?;
    let b = gpu(backend.upload_words(&host))?;
    let out = gpu(backend.upload_words(&host))?;
    for dtype in DTYPES {
        for p in bad_parameters() {
            let before = Snapshot::take(backend)?;
            let result = backend.dispatch_words(kind(dtype).kernel(), &[(&a, 0), (&b, 0), (&out, 0)], &p);
            if result.is_ok() {
                return Err(format!("accepted bad raw metadata {dtype:?}/{p:?}"));
            }
            before.check(backend, &Work::default())?;
            counts.rejections += 1;
        }
        if dtype == DType::Flex32 {
            for p in [[3, 5, 7, 1, 0], [3, 5, 7, 0, 1]] {
                let before = Snapshot::take(backend)?;
                if backend.dispatch_words(kind(dtype).kernel(), &[(&a, 0), (&b, 0), (&out, 0)], &p).is_ok() {
                    return Err(String::from("Flex32 accepted packed slot"));
                }
                before.check(backend, &Work::default())?;
                counts.rejections += 1;
            }
        }
        // Driver aliases are actual rejected service attempts. No kernel is
        // queued; input-input aliases remain valid in the repeated-source row.
        for buffers in [[(&a, 0), (&b, 0), (&a, 0)], [(&a, 0), (&b, 0), (&b, 1)]] {
            let before = Snapshot::take(backend)?;
            if backend.dispatch_words(kind(dtype).kernel(), &buffers, &[3, 5, 7, 0, 0]).is_ok() {
                return Err(String::from("driver accepted output/source overlap"));
            }
            let mut work = Work::default();
            work.services[abi::LAUNCH as usize] = 1;
            before.check(backend, &work)?;
            counts.rejections += 1;
        }
        for buffers in [
            [(&a, 256), (&b, 0), (&out, 0)],
            [(&a, 0), (&b, 256), (&out, 0)],
            [(&a, 0), (&b, 0), (&out, 256)],
        ] {
            let before = Snapshot::take(backend)?;
            if backend.dispatch_words(kind(dtype).kernel(), &buffers, &[3, 5, 7, 0, 0]).is_ok() {
                return Err(String::from("accepted short physical extent"));
            }
            before.check(backend, &Work::default())?;
            counts.rejections += 1;
        }
    }
    let nan = gpu(backend.upload_words(&[0x7fc1_dead]))?;
    for (prefix, dtype) in ["f16", "bf16", "flex32"].into_iter().zip(DTYPES) {
        let initial = vec![GUARD; 23];
        let zero = gpu(backend.upload_words(&initial))?;
        let before = Snapshot::take(backend)?;
        gpu(backend.record_commands(|| backend.dispatch_words(kind(dtype).kernel(), &[(&nan, 0), (&nan, 0), (&zero, 1)], &[3, 0, 7, 0, 0])))?;
        let mut work = Work::default();
        work.scope();
        work.launch(kind(dtype).kernel(), false);
        before.check(backend, &work)?;
        let expected = oracle::scratch_expectation(fixture(&format!("{prefix}_zero_K"))?).ok_or("K0 scratch absent")?;
        let actual = download(backend, &zero, 23)?;
        check_output(expected, check_guards(&actual, 21)?)?;
        count_values(counts, expected);
        counts.raw_zero_k += 1;
    }
    if download(backend, &nan, 1)? != [0x7fc1_dead] {
        return Err(String::from("raw zero-K placeholder changed"));
    }
    counts.source_words += 1;
    for (buffer, expected) in [(&a, &host), (&b, &host), (&out, &host)] {
        if download(backend, buffer, expected.len())? != *expected {
            return Err(String::from("rejection changed guarded source/output"));
        }
        counts.source_words += expected.len();
    }
    Ok(())
}
fn empty_and_zero(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in DTYPES {
        let prefix = match dtype {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => "flex32",
        };
        for suffix in ["zero_batch", "zero_M", "zero_N", "zero_K"] {
            let case = fixture(&format!("{prefix}_{suffix}"))?;
            let am = case.matrix("lhs").unwrap();
            let bm = case.matrix("rhs").unwrap();
            let (ar, a) = native_matrix(am)?;
            let (br, b) = native_matrix(bm)?;
            let before = Snapshot::take(backend)?;
            let output = gpu(a.matmul(&b))?;
            before.check(backend, &matmul_work(dtype, a.layout(), b.layout())?)?;
            let expected = main_expected(case)?;
            if expected.values.len == 0 {
                if output.can_mut() {
                    return Err(String::from("empty matmul failed source Arc alias ownership"));
                }
                counts.empty_aliases += 1;
            } else {
                let original = native_words(backend, &output)?;
                if original.iter().any(|&word| word != 0) || !output.can_mut() {
                    return Err(String::from("zero K requires fresh original positive-zero allocation"));
                }
                check_output(expected, &original)?;
                counts.zero_k += 1;
            }
            if output.dtype() != dtype || output.shape().as_slice() != expected.values.shape {
                return Err(String::from("empty/zero metadata changed"));
            }
            drop(output);
            drop(a);
            drop(b);
            if !ar.can_mut() || !br.can_mut() {
                return Err(String::from("empty aliases retained source after output drop"));
            }
            source_identity(backend, am, &ar, counts)?;
            source_identity(backend, bm, &br, counts)?;
        }
        // Native validates type, rank, K and incompatible zero batches before
        // recording, allocation, or an empty-output alias can occur.
        let cases = [
            (vec![0, 3, 5], vec![2, 5, 7], dtype),
            (vec![0, 3, 5], vec![1, 6, 0], dtype),
            (vec![0, 3, 5], vec![1, 5, 0], if dtype == DType::F16 { DType::BF16 } else { DType::F16 }),
            (vec![3], vec![3, 0], dtype),
            (vec![], vec![1, 0], dtype),
        ];
        for (a_shape, b_shape, b_dtype) in cases {
            let a = gpu(NativeTensor::empty(Shape::from(a_shape), dtype))?;
            let b = gpu(NativeTensor::empty(Shape::from(b_shape), b_dtype))?;
            let before = Snapshot::take(backend)?;
            if a.matmul(&b).is_ok() {
                return Err(String::from("Native accepted invalid empty metadata"));
            }
            before.check(backend, &Work::default())?;
            counts.rejections += 1;
        }
        for shape in [vec![1; 9], vec![0, i32::MAX as usize + 1], vec![46341, 46341]] {
            let before = Snapshot::take(backend)?;
            if NativeTensor::empty(Shape::from(shape), dtype).is_ok() {
                return Err(String::from("oversized metadata allocated before validation"));
            }
            before.check(backend, &Work::default())?;
            counts.rejections += 1;
        }
        let before = Snapshot::take(backend)?;
        if Layout::from_parts(Shape::new([3, 5]), vec![-5, 1], 0, 15).is_ok() {
            return Err(String::from("out-of-bounds signed view accepted"));
        }
        before.check(backend, &Work::default())?;
        counts.rejections += 1;
    }
    Ok(())
}
fn deferred(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in DTYPES {
        let other = if dtype == DType::F16 { DType::BF16 } else { DType::F16 };
        for (a_shape, b_shape, b_dtype) in [
            (vec![3, 5], vec![6, 2], dtype),
            (vec![0, 3, 5], vec![1, 5, 0], other),
            (vec![0, 3, 5], vec![2, 5, 7], dtype),
            (vec![3], vec![3, 0], dtype),
            (vec![], vec![1, 0], dtype),
        ] {
            let a = Native::float_from_data(
                TensorData::from_bytes_vec(oracle::pack(dtype, &vec![0; product(&a_shape)]), a_shape.clone(), dtype),
                &NativeDevice,
            );
            let b = Native::float_from_data(
                TensorData::from_bytes_vec(oracle::pack(b_dtype, &vec![0; product(&b_shape)]), b_shape, b_dtype),
                &NativeDevice,
            );
            let before = Snapshot::take(backend)?;
            let bad = Native::float_matmul(a.clone(), b.clone());
            let expected_shape = d3os_ai::native::layout::matmul_shape(&a.shape(), &b.shape()).unwrap_or_else(|_| a.shape());
            if bad.dtype() != dtype || bad.shape() != expected_shape {
                return Err(String::from("deferred matmul lost dtype/shape metadata"));
            }
            let error = try_read_sync(Native::float_into_data(bad)).ok_or("deferred read pending")?;
            if error.is_ok() {
                return Err(String::from("deferred malformed matmul failed to report original error"));
            }
            before.check(backend, &Work::default())?;
            counts.deferred += 1;
        }
        let prefix = match dtype {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => "flex32",
        };
        native_case(backend, fixture(&format!("{prefix}_dense_odd"))?, false, counts)?;
    }
    Ok(())
}
fn queued(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for prefix in ["f16", "bf16", "flex32"] {
        let case = fixture(&format!("{prefix}_dense_odd"))?;
        let am = case.matrix("lhs").unwrap();
        let bm = case.matrix("rhs").unwrap();
        let (ar, a) = native_matrix(am)?;
        let (br, b) = native_matrix(bm)?;
        let saved = ar.clone();
        drop(br);
        let expected = main_expected(case)?;
        let consumer = case.consumers.first().ok_or("queued consumer absent")?;
        let before = Snapshot::take(backend)?;
        let result = gpu(backend.record_commands(|| {
            let output = backend.record_commands(|| {
                let temporary = a.matmul(&b)?;
                drop(a);
                drop(b);
                Ok::<_, Error>(temporary)
            })?;
            let output = backend.record_commands(|| {
                let temporary = output.reduce(consumer.axis, Reduction::Sum)?;
                drop(output);
                // Retired inputs/scratch/output must not be recycled while
                // pending consumers still reference their last owners.
                let dirty = backend.allocate_words(expected.values.len.max(1))?;
                drop(dirty);
                Ok::<_, Error>(temporary)
            })?;
            Ok::<_, Error>(output)
        }))?;
        let mut work = matmul_work(am.values.dtype, &matrix_layout(am)?, &matrix_layout(bm)?)?;
        for _ in 0..3 {
            work.scope();
        }
        work.launch(Kernel::FloatReduction, true);
        work.allocate();
        work.free(3);
        before.check(backend, &work)?;
        inspect_native(backend, &result, &consumer.expected, counts)?;
        source_identity(backend, am, &saved, counts)?;
        drop(result);
        drop(saved);
        if !ar.can_mut() {
            return Err(String::from("queued last-owner recovery failed"));
        }
        counts.queued += 1;
    }
    Ok(())
}

fn merge_work(target: &mut Work, source: &Work) {
    for (i, &n) in source.kernels.iter().enumerate() {
        target.kernels[i] += n;
    }
    for (i, &n) in source.services.iter().enumerate() {
        target.services[i] += n;
    }
}
fn read_primitive(backend: &Nvidia, tensor: &FloatTensor<Native>, expected: &Expected, counts: &mut Counts) -> Result<()> {
    let before = Snapshot::take(backend)?;
    let data = try_read_sync(Native::float_into_data(tensor.clone()))
        .ok_or("public read pending")?
        .map_err(|e| format!("public read: {e:?}"))?;
    let mut work = Work::default();
    work.services[abi::DOWNLOAD as usize] = u64::from(expected.values.len != 0);
    before.check(backend, &work)?;
    if data.dtype() != expected.values.dtype || data.shape().as_slice() != expected.values.shape {
        return Err(String::from("public primitive metadata mismatch"));
    }
    compare_values(
        data.dtype(),
        expected.policies,
        &expected.values.bits(),
        &oracle::unpack(data.dtype(), data.as_bytes())?,
    )?;
    count_values(counts, expected);
    Ok(())
}
fn public_backward_work(case: &Case, track_left: bool, track_right: bool) -> Result<Work> {
    let a = case.matrix("lhs").ok_or("public lhs absent")?;
    let b = case.matrix("rhs").ok_or("public rhs absent")?;
    let g = case.matrix("upstream").ok_or("public upstream absent")?;
    let dtype = a.values.dtype;
    let mut work = Work::default();
    work.scope();
    // Backward starts with ones at the nonscalar weighted output. Multiplying
    // those ones by frozen upstream values seeds the real ordinary graph.
    work.launch(if dtype == DType::Flex32 { Kernel::Fill } else { Kernel::TensorStorage }, true);
    work.launch(Kernel::FloatArithmetic, true);
    work.free(3 + usize::from(!track_left) + usize::from(!track_right));
    let al = gpu(Layout::contiguous(Shape::from(a.values.shape.to_vec())))?;
    let bl = gpu(Layout::contiguous(Shape::from(b.values.shape.to_vec())))?;
    let gl = gpu(Layout::contiguous(Shape::from(g.values.shape.to_vec())))?;
    for (active, left, right, target) in [
        (
            track_left,
            gl.clone(),
            gpu(bl.swap_dims(bl.shape().num_dims() - 2, bl.shape().num_dims() - 1))?,
            a.values.shape,
        ),
        (
            track_right,
            gpu(al.swap_dims(al.shape().num_dims() - 2, al.shape().num_dims() - 1))?,
            gl,
            b.values.shape,
        ),
    ] {
        if active {
            let output = gpu(d3os_ai::native::layout::matmul_shape(left.shape(), right.shape()))?;
            merge_work(&mut work, &matmul_work(dtype, &left, &right)?);
            for (axis, &n) in target.iter().enumerate() {
                if output[axis] != n {
                    work.launch(Kernel::FloatReduction, true);
                    work.free(1);
                }
            }
        }
    }
    if case.tracking == Tracking::Shared {
        work.launch(Kernel::FloatArithmetic, true);
        work.free(2);
    }
    Ok(work)
}
fn public_autodiff<const D: usize>(backend: &Nvidia, case: &Case, counts: &mut Counts) -> Result<()> {
    use d3os_ai::burn::tensor::Device;
    let device = Device::new(NativeDevice);
    let am = case.matrix("lhs").ok_or("public lhs absent")?;
    let bm = case.matrix("rhs").ok_or("public rhs absent")?;
    let tracked_left = !matches!(case.tracking, Tracking::Rhs | Tracking::None);
    let tracked_right = !matches!(case.tracking, Tracking::Lhs | Tracking::None);
    let a = public_tensor::<D>(am, &device).set_require_grad(tracked_left);
    let b = if case.tracking == Tracking::Shared {
        a.clone()
    } else {
        public_tensor::<D>(bm, &device).set_require_grad(tracked_right)
    };
    let saved_a = a.clone();
    let saved_b = b.clone();
    let before = Snapshot::take(backend)?;
    let output = a.matmul(b);
    let mut work = matmul_work(
        am.values.dtype,
        &gpu(Layout::contiguous(Shape::from(am.values.shape.to_vec())))?,
        &gpu(Layout::contiguous(Shape::from(bm.values.shape.to_vec())))?,
    )?;
    before.check(backend, &work)?;
    if output.is_tracked() != (tracked_left || tracked_right) {
        return Err(String::from("ordinary public tracking changed"));
    }
    let forward = output
        .clone()
        .inner()
        .try_into_primitive::<Native>()
        .map_err(|_| String::from("public Native output route absent"))?;
    read_primitive(backend, &forward, main_expected(case)?, counts)?;
    if !tracked_left && !tracked_right {
        counts.autodiff += 1;
        return Ok(());
    }
    // The untracked opposite input now survives only in ordinary Autodiff's
    // saved state. No plain buffer or spare user clone can mask its lifetime.
    let saved_a = tracked_left.then_some(saved_a);
    let saved_b = tracked_right.then_some(saved_b);
    let gm = case.matrix("upstream").ok_or("public seed absent")?;
    let upstream = public_tensor::<D>(gm, &device);
    let before = Snapshot::take(backend)?;
    let weighted = gpu(backend.record_commands(|| Ok::<_, Error>(output * upstream)))?;
    work = Work::default();
    work.scope();
    work.launch(Kernel::FloatArithmetic, true);
    before.check(backend, &work)?;
    let saved_weighted = weighted.clone();
    let before = Snapshot::take(backend)?;
    let gradients = gpu(backend.record_commands(|| Ok::<_, Error>(weighted.backward())))?;
    before.check(backend, &public_backward_work(case, tracked_left, tracked_right)?)?;
    let da = saved_a.as_ref().and_then(|a| a.grad(&gradients));
    let db = saved_b.as_ref().and_then(|b| b.grad(&gradients));
    if da.is_some() != tracked_left || db.is_some() != tracked_right {
        return Err(String::from("ordinary public tracked/single/saved leaf gradients changed"));
    }
    if case.tracking == Tracking::Shared {
        let tensor = da
            .ok_or("shared gradient absent")?
            .inner()
            .try_into_primitive::<Native>()
            .map_err(|_| String::from("shared Native gradient absent"))?;
        read_primitive(
            backend,
            &tensor,
            case.expectation("shared_source_gradient").ok_or("shared expected absent")?,
            counts,
        )?;
    } else {
        for (tensor, name) in [(da, "dA"), (db, "dB")] {
            if let Some(tensor) = tensor {
                let tensor = tensor
                    .inner()
                    .try_into_primitive::<Native>()
                    .map_err(|_| String::from("Native gradient route absent"))?;
                read_primitive(backend, &tensor, case.expectation(name).ok_or("gradient expected absent")?, counts)?;
            }
        }
    }
    drop(saved_weighted);
    counts.autodiff += 1;
    Ok(())
}
fn numeric_staging_native(backend: &Nvidia, case: &Case, counts: &mut Counts) -> Result<()> {
    let mut roots = Vec::new();
    let mut input = |name: &str| -> Result<NativeTensor> {
        let m = case.matrix(name).ok_or("staging source absent")?;
        let (root, tensor) = native_matrix(m)?;
        roots.push((m, root));
        Ok(tensor)
    };
    let x = input("linear.input")?;
    let w = input("linear.weight")?;
    let b = input("linear.bias")?;
    let stored = checked_matmul(backend, &x, &w, case.expectation("linear.stored").ok_or("stored linear absent")?, counts)?;
    let _ = checked_add(
        backend,
        &stored,
        &b,
        case.expectation("linear.output").ok_or("linear staged output absent")?,
        counts,
    )?;
    let before = Snapshot::take(backend)?;
    let y = gpu(x.linear(&w, Some(&b)))?;
    before.check(backend, &linear_work(x.dtype(), x.layout(), w.layout(), true)?)?;
    inspect_native(backend, &y, case.expectation("linear.output").unwrap(), counts)?;
    let a = input("gradient.lhs")?;
    let rhs = input("gradient.rhs")?;
    let g = input("gradient.upstream")?;
    let _ = checked_matmul(
        backend,
        &a,
        &rhs,
        case.expectation("gradient.forward").ok_or("staged gradient forward absent")?,
        counts,
    )?;
    let da0 = checked_matmul(
        backend,
        &g,
        &transpose(&rhs)?,
        case.expectation("gradient.dA_before_sum").ok_or("stored midpoint gradient absent")?,
        counts,
    )?;
    let _ = checked_sum(
        backend,
        &da0,
        0,
        case.expectation("gradient.dA").ok_or("post-storage gradient sum absent")?,
        counts,
    )?;
    for (m, root) in roots {
        source_identity(backend, m, &root, counts)?;
    }
    Ok(())
}
fn public_linear_backward_work(case: &Case) -> Result<Work> {
    let xm = case.matrix("input").ok_or("linear backward plan x")?;
    let wm = case.matrix("weight").ok_or("linear backward plan w")?;
    let gm = case.matrix("upstream").ok_or("linear backward plan g")?;
    let dtype = xm.values.dtype;
    let mut work = Work::default();
    work.scope();
    work.launch(if dtype == DType::Flex32 { Kernel::Fill } else { Kernel::TensorStorage }, true);
    work.launch(Kernel::FloatArithmetic, true);
    work.free(3);
    let gl = matrix_layout(gm)?;
    let wl = matrix_layout(wm)?;
    let wt = gpu(wl.swap_dims(0, 1))?;
    merge_work(&mut work, &linear_work(dtype, &gl, &wt, false)?);
    let rows = product(&xm.values.shape[..2]);
    let xf = gpu(Layout::contiguous(Shape::new([rows, xm.values.shape[2]])))?;
    let gf = gpu(Layout::contiguous(Shape::new([rows, gm.values.shape[2]])))?;
    work.scope();
    merge_work(&mut work, &matmul_work(dtype, &gpu(xf.swap_dims(0, 1))?, &gf)?);
    work.scope();
    work.launch(Kernel::FloatReduction, true);
    Ok(work)
}
fn public_linear(backend: &Nvidia, case: &Case, counts: &mut Counts) -> Result<()> {
    use d3os_ai::burn::tensor::{Device, module::linear};
    let device = Device::new(NativeDevice);
    let xm = case.matrix("input").ok_or("public linear x")?;
    let wm = case.matrix("weight").ok_or("public linear w")?;
    let bm = case.matrix("bias").ok_or("public linear b")?;
    let gm = case.matrix("upstream").ok_or("public linear g")?;
    let x = public_tensor::<3>(xm, &device).require_grad();
    let w = public_tensor::<2>(wm, &device).require_grad();
    let b = public_tensor::<1>(bm, &device).require_grad();
    let before = Snapshot::take(backend)?;
    let y = linear(x.clone(), w.clone(), Some(b.clone()));
    before.check(backend, &linear_work(xm.values.dtype, &matrix_layout(xm)?, &matrix_layout(wm)?, true)?)?;
    let plain = y
        .clone()
        .inner()
        .try_into_primitive::<Native>()
        .map_err(|_| String::from("public Native linear output"))?;
    read_primitive(backend, &plain, main_expected(case)?, counts)?;
    let g = public_tensor::<3>(gm, &device);
    let before = Snapshot::take(backend)?;
    let weighted = gpu(backend.record_commands(|| Ok::<_, Error>(y * g)))?;
    let mut work = Work::default();
    work.scope();
    work.launch(Kernel::FloatArithmetic, true);
    before.check(backend, &work)?;
    let saved = weighted.clone();
    let before = Snapshot::take(backend)?;
    let gradients = gpu(backend.record_commands(|| Ok::<_, Error>(weighted.backward())))?;
    let work = public_linear_backward_work(case)?;
    before.check(backend, &work)?;
    let dx = x
        .grad(&gradients)
        .ok_or("public linear dx absent")?
        .inner()
        .try_into_primitive::<Native>()
        .map_err(|_| String::from("Native dx route"))?;
    let dw = w
        .grad(&gradients)
        .ok_or("public linear dW absent")?
        .inner()
        .try_into_primitive::<Native>()
        .map_err(|_| String::from("Native dW route"))?;
    let db = b
        .grad(&gradients)
        .ok_or("public linear db absent")?
        .inner()
        .try_into_primitive::<Native>()
        .map_err(|_| String::from("Native db route"))?;
    for (t, name) in [(&dx, "dx"), (&dw, "dW"), (&db, "db")] {
        read_primitive(backend, t, case.expectation(name).ok_or("public linear expected")?, counts)?;
    }
    drop(saved);
    counts.autodiff += 1;
    Ok(())
}
fn public_routes(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for prefix in ["f16", "bf16", "flex32"] {
        for suffix in [
            "dense_odd",
            "both_operands_dense",
            "lhs_tracked_only",
            "rhs_tracked_only",
            "shared_source_both_operands",
        ] {
            public_autodiff::<2>(backend, fixture(&format!("{prefix}_{suffix}"))?, counts)?;
        }
        public_autodiff::<6>(backend, fixture(&format!("{prefix}_both_multi_axis_batch_broadcast"))?, counts)?;
        public_linear(backend, fixture(&format!("{prefix}_linear_with_bias"))?, counts)?;
    }
    Ok(())
}

fn root_words(matrix: &Matrix) -> usize {
    let root = matrix.view.map_or(matrix.values, |v| v.root);
    (root.len * root.dtype.size()).div_ceil(4).max(1)
}
fn inventory_native(case: &Case, tile: bool, counts: &mut Counts) -> Result<()> {
    let (a, b, e) = if tile {
        (
            case.raw_matrices.iter().find(|m| m.name == "tile.lhs").ok_or("inventory tile lhs")?,
            case.raw_matrices.iter().find(|m| m.name == "tile.rhs").ok_or("inventory tile rhs")?,
            case.raw_expected.iter().find(|e| e.name == "tile.output").ok_or("inventory tile output")?,
        )
    } else {
        (
            case.matrix("lhs").ok_or("inventory lhs")?,
            case.matrix("rhs").ok_or("inventory rhs")?,
            main_expected(case)?,
        )
    };
    counts.native_outputs += 1;
    count_values(counts, e);
    counts.source_words += 3 * (root_words(a) + root_words(b));
    Ok(())
}
fn inventory_gradient(case: &Case, counts: &mut Counts) -> Result<()> {
    let names = if case.operation == Operation::Linear {
        vec!["forward", "dx", "dW", "db"]
    } else {
        vec!["forward", "dA_before_sum", "dA", "dB_before_sum", "dB"]
    };
    for name in names {
        let e = case.expectation(name).ok_or("inventory gradient missing")?;
        counts.native_outputs += 1;
        count_values(counts, e);
    }
    counts.source_words += case.matrices.iter().map(root_words).sum::<usize>();
    if case.operation != Operation::Linear {
        for e in case
            .expected
            .iter()
            .filter(|e| e.name.starts_with("dA_axes.") || e.name.starts_with("dB_axes.") || e.name == "shared_source_gradient")
        {
            counts.native_outputs += 1;
            count_values(counts, e);
        }
    }
    for c in case.consumers {
        counts.native_outputs += 1;
        count_values(counts, &c.expected);
    }
    Ok(())
}
/// Every population is enumerated from frozen descriptors before execution.
fn expected_counts(plan: &[RawCase]) -> Result<Counts> {
    let mut counts = Counts::default();
    for r in plan {
        counts.raw_cases += 1;
        counts.guarded_scratch += 2;
        counts.fresh_scratch += 2;
        for _ in 0..4 {
            count_values(&mut counts, r.expected(true)?);
        }
        if r.case.dtype != Some(DType::Flex32) {
            counts.guarded_cast += 2;
            counts.fresh_cast += 2;
            for _ in 0..4 {
                count_values(&mut counts, r.expected(false)?);
            }
        }
        counts.source_words += 2 * (r.source_words(true)?.len() + r.source_words(false)?.len());
    }
    for c in oracle::fixtures() {
        match c.group {
            Group::Geometry | Group::Numeric => inventory_native(c, false, &mut counts)?,
            Group::Gradient => inventory_gradient(c, &mut counts)?,
            _ => {}
        }
    }
    for c in oracle::fixtures().iter().filter(|c| c.matrix("linear.input").is_some()) {
        for name in [
            "linear.stored",
            "linear.output",
            "linear.output",
            "gradient.forward",
            "gradient.dA_before_sum",
            "gradient.dA",
        ] {
            counts.native_outputs += 1;
            count_values(&mut counts, c.expectation(name).ok_or("staging inventory")?);
        }
        for name in [
            "linear.input",
            "linear.weight",
            "linear.bias",
            "gradient.lhs",
            "gradient.rhs",
            "gradient.upstream",
        ] {
            counts.source_words += root_words(c.matrix(name).unwrap());
        }
    }
    for prefix in ["f16", "bf16", "flex32"] {
        let c = fixture(&format!("{prefix}_linear_with_bias"))?;
        counts.autodiff += 1;
        for name in ["forward", "dx", "dW", "db"] {
            count_values(&mut counts, c.expectation(name).unwrap());
        }
        let c = fixture(&format!("{prefix}_dense_odd"))?;
        inventory_native(c, true, &mut counts)?;
        inventory_native(c, false, &mut counts)?;
        counts.queued += 1;
        counts.native_outputs += 1;
        count_values(&mut counts, &c.consumers[0].expected);
        counts.source_words += root_words(c.matrix("lhs").unwrap());
        for suffix in ["zero_batch", "zero_M", "zero_N", "zero_K"] {
            let c = fixture(&format!("{prefix}_{suffix}"))?;
            counts.source_words += c.matrices.iter().map(root_words).sum::<usize>();
            if suffix == "zero_K" {
                counts.zero_k += 1;
            } else {
                counts.empty_aliases += 1;
            }
        }
        for suffix in [
            "dense_odd",
            "both_operands_dense",
            "lhs_tracked_only",
            "rhs_tracked_only",
            "shared_source_both_operands",
            "both_multi_axis_batch_broadcast",
        ] {
            let c = fixture(&format!("{prefix}_{suffix}"))?;
            counts.autodiff += 1;
            count_values(&mut counts, main_expected(c)?);
            if c.tracking == Tracking::Shared {
                count_values(&mut counts, c.expectation("shared_source_gradient").unwrap());
            } else {
                if !matches!(c.tracking, Tracking::None | Tracking::Rhs) {
                    count_values(&mut counts, c.expectation("dA").unwrap());
                }
                if !matches!(c.tracking, Tracking::None | Tracking::Lhs) {
                    count_values(&mut counts, c.expectation("dB").unwrap());
                }
            }
        }
    }
    counts.source_words += 3 * 256 + 1;
    counts.rejections = DTYPES.len() * (bad_parameters().len() + 2 + 3 + 5 + 4) + 2;
    counts.deferred = 15;
    counts.raw_zero_k = 3;
    for prefix in ["f16", "bf16", "flex32"] {
        count_values(
            &mut counts,
            oracle::scratch_expectation(fixture(&format!("{prefix}_zero_K"))?).ok_or("K0 scratch descriptor")?,
        );
    }
    Ok(counts)
}
fn audit_raw(case: &RawCase) -> Result<()> {
    let dtype = case.case.dtype.ok_or("audit dtype")?;
    let [m, k, n] = case.dimensions()?;
    let scratch = case.expected(true)?;
    let output = case.expected(false)?;
    if scratch.values.dtype != DType::F32
        || scratch.values.len != output.values.len
        || scratch.values.shape != output.values.shape
        || scratch.policies.len() != scratch.values.len
    {
        return Err(String::from("independent F32 scratch descriptor mismatch"));
    }
    for batch in 0..case.batches()? {
        let p = case.parameters(batch)?.map(|v| v as u32);
        let work = kind(dtype).kernel().work(&p).map_err(|e| format!("registered mixed work: {e:?}"))?;
        let word_count = |count: usize, slot: usize| if dtype == DType::Flex32 { count } else { (count + slot).div_ceil(2) };
        let expected = [word_count(m * k, p[3] as usize) as u64, word_count(k * n, p[4] as usize) as u64, (m * n) as u64];
        if work.threads != 128
            || work.blocks as usize != m.div_ceil(16) * n.div_ceil(32)
            || work.extents[..3] != expected
            || (work.pointers, work.output, work.in_place, work.mutated) != (3, 2, false, 0)
        {
            return Err(String::from("registered mixed geometry/extents/ownership mismatch"));
        }
        let symbol = match dtype {
            DType::F16 => "burn_matmul_f16",
            DType::BF16 => "burn_matmul_bf16",
            _ => "burn_matmul_flex32",
        };
        if kind(dtype).kernel().symbol() != symbol || kind(dtype).kernel().entry(&p) != symbol {
            return Err(String::from("mixed kernel static entry mismatch"));
        }
        let input_address = 0x100004;
        let right_address = input_address + work.extents[0] * 4;
        if work.validate_aliases(&[input_address, input_address, 0x800004]).is_err()
            || work.validate_aliases(&[input_address, right_address, input_address]).is_ok()
        {
            return Err(String::from("raw repeated-input/output alias contract changed"));
        }
    }
    if case.tile && (m <= 16 || n <= 32 || k <= 32) {
        return Err(String::from("tile refinement does not cross all tile boundaries"));
    }
    Ok(())
}
fn plan_json(plan: &[RawCase]) -> Result<String> {
    let mut rows = Vec::new();
    for c in plan {
        let scratch = c.expected(true)?;
        let destination = c.expected(false)?;
        let guarded = raw_work(c, false, false)?;
        let cast = if c.case.dtype == Some(DType::Flex32) {
            Work::default()
        } else {
            raw_work(c, false, true)?
        };
        rows.push(format!("{{\"case_id\":\"{}\",\"tile\":{},\"dtype\":\"{:?}\",\"dimensions\":{:?},\"batches\":{},\"slots\":[{},{}],\"scratch_values\":{},\"destination_values\":{},\"guarded_scratch_kernels\":{:?},\"guarded_scratch_services\":{:?},\"guarded_cast_kernels\":{:?},\"guarded_cast_services\":{:?}}}",c.case.id,c.tile,c.case.dtype.unwrap(),c.dimensions()?,c.batches()?,c.lhs_slot,c.rhs_slot,scratch.values.len,destination.values.len,guarded.kernels,guarded.services,cast.kernels,cast.services));
    }
    let mut native = Vec::new();
    let mut public = Vec::new();
    for c in oracle::fixtures()
        .iter()
        .filter(|c| matches!(c.group, Group::Geometry | Group::Numeric | Group::Gradient))
    {
        let linear = c.operation == Operation::Linear;
        let a = c.matrix(if linear { "input" } else { "lhs" }).ok_or("plan Native lhs")?;
        let b = c.matrix(if linear { "weight" } else { "rhs" }).ok_or("plan Native rhs")?;
        let w = if linear {
            linear_work(a.values.dtype, &matrix_layout(a)?, &matrix_layout(b)?, true)?
        } else {
            matmul_work(a.values.dtype, &matrix_layout(a)?, &matrix_layout(b)?)?
        };
        native.push(format!(
            "{{\"case_id\":\"{}\",\"kernels\":{:?},\"services\":{:?}}}",
            c.id, w.kernels, w.services
        ));
    }
    for prefix in ["f16", "bf16", "flex32"] {
        for suffix in [
            "both_operands_dense",
            "lhs_tracked_only",
            "rhs_tracked_only",
            "shared_source_both_operands",
            "both_multi_axis_batch_broadcast",
        ] {
            let c = fixture(&format!("{prefix}_{suffix}"))?;
            let w = public_backward_work(c, c.tracking != Tracking::Rhs, c.tracking != Tracking::Lhs)?;
            public.push(format!(
                "{{\"case_id\":\"{}\",\"kernels\":{:?},\"services\":{:?}}}",
                c.id, w.kernels, w.services
            ));
        }
        let c = fixture(&format!("{prefix}_linear_with_bias"))?;
        let w = public_linear_backward_work(c)?;
        public.push(format!(
            "{{\"case_id\":\"{}\",\"kernels\":{:?},\"services\":{:?}}}",
            c.id, w.kernels, w.services
        ));
    }
    Ok(format!(
        "{{\"protocol\":{},\"kernel_count\":{},\"service_count\":{},\"scratch_trial_order\":[\"fresh\",\"guarded\"],\"scratch_poison_masks\":{:?},\"scratch_physical_overwrite_all_words\":true,\"fresh_cross_trial_coherence\":true,\"cast_trial_order\":[\"fresh\",\"guarded\"],\"cast_poison_masks\":{:?},\"cast_physical_overwrite_all_live_elements\":true,\"cast_fresh_cross_trial_coherence\":true,\"cast_unused_tail_zero\":true,\"raw_recipes\":[{}],\"native_forward_recipes\":[{}],\"public_backward_recipes\":[{}],\"shared\":{}}}",
        abi::VERSION,
        abi::KERNEL_COUNT,
        abi::SERVICE_OPERATIONS,
        SCRATCH_POISON_MASKS,
        SCRATCH_POISON_MASKS,
        rows.join(","),
        native.join(","),
        public.join(","),
        oracle::summary_json(&frozen_shared_summary())
    ))
}
fn options() -> oracle::Options {
    oracle::Options {
        max_failures: 12,
        ..Default::default()
    }
}
fn contract(plan: &[RawCase]) -> Result<()> {
    if !matches!((abi::VERSION, abi::KERNEL_COUNT, abi::SERVICE_OPERATIONS), (54, 88, 14) | (55, 89, 14))
        || raw::PARAMETERS != 5
        || raw::THREADS != 128
        || [Kernel::MatmulF16 as usize, Kernel::MatmulBf16 as usize, Kernel::MatmulFlex32 as usize] != [85, 86, 87]
        || oracle::CORPUS_VERSION != 1
        || oracle::SELECTED_ROWS != 121
        || oracle::RESERVED_ROWS != 3
        || oracle::NUMERIC_CELLS != 65
        || oracle::LAST_DROP_CONSUMERS != 141
    {
        return Err(String::from("mixed matmul protocol/corpus/API registration changed"));
    }
    for c in plan {
        audit_raw(c)?;
    }
    if expected_counts(plan)? != EXPECTED {
        return Err(String::from("mixed raw count freeze changed"));
    }
    if oracle::expected_summary(&options(), Scope::Shared) != frozen_shared_summary() {
        return Err(String::from("mixed Shared literal population freeze changed"));
    }
    let s = oracle::expected_summary(&options(), Scope::Shared);
    if (s.selected_rows, s.geometry_rows, s.numeric_rows, s.gradient_rows, s.external_gates) != (93, 39, 24, 30, 0) {
        return Err(String::from("mixed strict Shared scope changed"));
    }
    Ok(())
}
pub(super) fn run(backend: &Nvidia) -> Result<()> {
    let plan = raw_plan();
    contract(&plan)?;
    let expected = EXPECTED;
    let options = options();
    let shared_expected = oracle::expected_summary(&options, Scope::Shared);
    crate::report(format_args!(
        "NVIDIA_BURN_MIXED_MATMUL_PLAN frozen={FROZEN} protocol={} kernels={} source_pins={SOURCE_PINS:?} raw={expected:?} shared={} population={} conditional_model={} full_public_qualification=false",
        abi::VERSION,
        abi::KERNEL_COUNT,
        oracle::summary_json(&shared_expected),
        plan_json(&plan)?,
        oracle::MODEL_LABEL
    ));
    if !FROZEN {
        return Err(String::from(
            "mixed matmul UNFROZEN: ROOT exact numerical/corpus/source/count approval required",
        ));
    }
    let mut observed = Observed::default();
    for c in &plan {
        let result = raw_case(backend, c, &mut observed.counts);
        observed.record(&format!("raw/{}/tile={}", c.case.id, c.tile), result);
    }
    for c in oracle::fixtures() {
        let result = match c.group {
            Group::Geometry | Group::Numeric => native_case(backend, c, false, &mut observed.counts),
            Group::Gradient => native_gradients(backend, c, &mut observed.counts),
            _ => continue,
        };
        observed.record(&format!("native/{}", c.id), result);
        if c.matrix("linear.input").is_some() {
            let result = numeric_staging_native(backend, c, &mut observed.counts);
            observed.record(&format!("staging/{}", c.id), result);
        }
    }
    for prefix in ["f16", "bf16", "flex32"] {
        let c = fixture(&format!("{prefix}_dense_odd"))?;
        let result = native_case(backend, c, true, &mut observed.counts);
        observed.record(&format!("tile/{prefix}"), result);
    }
    for (name, suite) in [
        ("rejections", rejections as fn(&Nvidia, &mut Counts) -> Result<()>),
        ("empty_zero", empty_and_zero),
        ("deferred", deferred),
        ("queued", queued),
        ("ordinary_public_autodiff", public_routes),
    ] {
        let result = suite(backend, &mut observed.counts);
        observed.record(name, result);
    }
    let device = d3os_ai::native_device().map_err(|e| format!("mixed shared Native device: {e}"))?;
    let shared = try_read_sync(oracle::run(&device, &options, Scope::Shared)).ok_or("shared mixed matmul pending")?;
    // Emit every bounded observation, class, stage and worst-error witness even
    // if a mandatory numerical, queue, counter, or source gate failed.
    crate::report(format_args!("NVIDIA_BURN_MIXED_MATMUL_SHARED {}", oracle::report_json(&shared)));
    for scope in [Scope::NativeExtensions, Scope::Preservation, Scope::F64Diagnostics] {
        let report = try_read_sync(oracle::run(&device, &options, scope)).ok_or("separate mixed diagnostic pending")?;
        crate::report(format_args!(
            "NVIDIA_BURN_MIXED_MATMUL_SEPARATE scope={scope:?} qualification=false {}",
            oracle::report_json(&report)
        ));
    }
    let accepted = observed.failed == 0 && observed.counts == expected && shared.expected == shared_expected && shared.is_success();
    crate::report(format_args!(
        "NVIDIA_BURN_MIXED_MATMUL_OBSERVED accepted={accepted} expected={expected:?} observed={:?} failed={} examples={:?} external_extension_evidence=raw_separate preservation=separate f64=diagnostic_only full_public_qualification=false",
        observed.counts, observed.failed, observed.examples
    ));
    if !accepted {
        return Err(format!(
            "mixed matmul strict failure expected={expected:?} observed={:?} failed={} shared_failed={}",
            observed.counts, observed.failed, shared.failed
        ));
    }
    crate::report(format_args!(
        "{} os=d3os backend=native fusion=disabled",
        oracle::success_footer(&options, Scope::Shared, &shared)?
    ));
    crate::report(format_args!(
        "NVIDIA_BURN_MIXED_MATMUL_STRICT_OK scope=shared93_plus_raw protocol={} kernels={} observed={:?} original_allocation_tail=checked full_public_qualification=false",
        abi::VERSION,
        abi::KERNEL_COUNT,
        observed.counts
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_plan_checks_registered_work_and_distinct_packed_batch_slots() {
        let plan = raw_plan();
        contract(&plan).unwrap();
        assert_eq!(plan.len(), 33);
        for c in &plan {
            for lhs in [true, false] {
                let dtype = c.case.dtype.unwrap();
                let source = c.source_words(lhs).unwrap();
                let bits = c.input_bits(lhs).unwrap();
                let [m, k, n] = c.dimensions().unwrap();
                let size = if lhs { m * k } else { k * n };
                for batch in 0..c.batches().unwrap() {
                    let offset = c.input_offset(lhs, batch).unwrap();
                    let slot = c.parameters(batch).unwrap()[if lhs { 3 } else { 4 }];
                    for i in 0..size {
                        let actual = if dtype == DType::Flex32 {
                            u64::from(source[offset + i])
                        } else {
                            u64::from(source[offset + (slot + i) / 2] >> (((slot + i) % 2) * 16) & 0xffff)
                        };
                        assert_eq!(actual, bits[batch * size + i], "{} lhs={lhs} batch={batch} i={i}", c.case.id);
                    }
                }
            }
        }
        for dtype in [DType::F16, DType::BF16] {
            let c = plan
                .iter()
                .find(|c| c.case.dtype == Some(dtype) && !c.tile && c.batches().unwrap() > 1 && c.dimensions().unwrap()[1] % 2 == 1)
                .unwrap();
            assert_ne!(c.parameters(0).unwrap()[3], c.parameters(1).unwrap()[3]);
            assert_ne!(c.parameters(0).unwrap()[4], c.parameters(1).unwrap()[4]);
        }
    }
    #[test]
    fn original_full_words_reject_poison_partial_writers_and_wrong_tail() {
        for c in raw_plan() {
            for scratch in [true, false] {
                let e = c.expected(scratch).unwrap();
                let expected = words(e.values.bytes());
                assert!(check_output(e, &expected).is_ok());
                for poison in POISONS {
                    let mut poisoned = expected.clone();
                    let index = e.policies.iter().position(|p| *p != Policy::Observation).unwrap_or(0);
                    let word = index * e.values.dtype.size() / 4;
                    poisoned[word] = poison;
                    if e.policies.iter().any(|p| *p != Policy::Observation) {
                        assert!(check_output(e, &poisoned).is_err());
                    }
                }
                if e.values.dtype.size() == 2 && e.values.len % 2 == 1 {
                    let mut stale = expected;
                    *stale.last_mut().unwrap() |= 0xa53c_0000;
                    assert!(check_output(e, &stale).is_err());
                }
            }
        }
    }
    #[test]
    fn every_observation_no_write_reproducer_fails_the_physical_gate() {
        let (mut recipes, mut cells, mut old_untouched) = (0, 0, 0);
        for c in raw_plan() {
            let e = c.expected(true).unwrap();
            if !e.policies.contains(&Policy::Observation) {
                continue;
            }
            let baseline = words(e.values.bytes());
            let mut dirty_trials = Vec::new();
            for poison in POISONS {
                let mut actual = baseline.clone();
                for (i, p) in e.policies.iter().enumerate() {
                    if *p == Policy::Observation {
                        actual[i] = poison;
                    }
                }
                // R1's numerical-only checker must remain observation-only.
                assert!(check_output(e, &actual).is_ok());
                let known_poisons = vec![poison; baseline.len()];
                assert!(check_scratch_overwrite(&baseline, &known_poisons, &actual).is_err());
                // A stale Fresh observation cannot serve as numerical truth:
                // an independent guarded writer must still cohere with it.
                let derived = scratch_poisons(&actual, SCRATCH_POISON_MASKS[0]);
                assert!(check_scratch_overwrite(&actual, &derived, &baseline).is_err());
                dirty_trials.push(actual);
                old_untouched += 1;
            }
            // Fresh nonwrites carrying the two dirty allocations cannot hide
            // behind otherwise unconstrained numerical Observation policies.
            assert!(check_scratch_coherence(&dirty_trials[0], &dirty_trials[1]).is_err());
            for mask in SCRATCH_POISON_MASKS {
                let poisons = scratch_poisons(&baseline, mask);
                assert!(check_scratch_overwrite(&baseline, &poisons, &baseline).is_ok());
                for (i, p) in e.policies.iter().enumerate() {
                    if *p == Policy::Observation {
                        let mut partial = baseline.clone();
                        partial[i] = poisons[i];
                        assert!(check_scratch_overwrite(&baseline, &poisons, &partial).is_err());
                    }
                }
            }
            recipes += 1;
            cells += e.policies.iter().filter(|p| **p == Policy::Observation).count();
        }
        assert_eq!((recipes, cells, old_untouched), (11, 11, 22));
        println!("R1_OBSERVATION_REPRO_REJECTED recipes={recipes} cells={cells} old_untouched={old_untouched}");
    }
    #[test]
    fn scratch_overwrite_is_independent_of_nan_payloads_and_zero_signs() {
        let baseline = [0x7fc0_1234, 0xffc0_5678, 0, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 1, 0x3f80_0000];
        let same_values = [0xffc1_4321, 0x7fc2_8765, 0x8000_0000, 0, 0x7f80_0000, 0xff80_0000, 1, 0x3f80_0000];
        for mask in SCRATCH_POISON_MASKS {
            assert!(mask.to_le_bytes().iter().all(|&byte| byte != 0));
            let poisons = scratch_poisons(&baseline, mask);
            assert!(check_scratch_overwrite(&baseline, &poisons, &same_values).is_ok());
            for i in 0..baseline.len() {
                let mut partial = same_values;
                partial[i] = poisons[i];
                assert!(check_scratch_overwrite(&baseline, &poisons, &partial).is_err());
            }
            for i in [0, 1, 4, 5] {
                assert_eq!(oracle::classification(DType::F32, u64::from(poisons[i])), oracle::Class::Finite);
            }
        }
        assert!((SCRATCH_POISON_MASKS[0] ^ SCRATCH_POISON_MASKS[1]).to_le_bytes().iter().all(|&byte| byte != 0));
        let mut wrong = baseline;
        wrong[4] = 0xff80_0000;
        assert!(check_scratch_coherence(&baseline, &wrong).is_err());
        wrong = baseline;
        wrong[7] ^= 1;
        assert!(check_scratch_coherence(&baseline, &wrong).is_err());
        assert!(check_scratch_overwrite(&baseline, &baseline, &same_values[..7]).is_err());
    }
    #[test]
    fn every_scratch_word_requires_a_store_and_sources_remain_exact() {
        let mut physical_words = 0;
        for c in raw_plan() {
            let baseline = words(c.expected(true).unwrap().values.bytes());
            for mask in SCRATCH_POISON_MASKS {
                let poisons = scratch_poisons(&baseline, mask);
                for i in 0..baseline.len() {
                    let mut partial = baseline.clone();
                    partial[i] = poisons[i];
                    assert!(check_scratch_overwrite(&baseline, &poisons, &partial).is_err());
                    physical_words += 1;
                }
            }
            for lhs in [true, false] {
                let source = c.source_words(lhs).unwrap();
                assert!(check_source_words(&source, &source).is_ok());
                // Include source NaN payloads, packed sibling halves and both
                // word guards. Numerical class acceptance never applies here.
                for i in 0..source.len() {
                    let mut corrupt = source.clone();
                    corrupt[i] ^= 1;
                    assert!(check_source_words(&source, &corrupt).is_err());
                }
            }
        }
        assert!(physical_words > 1000);
    }
    #[test]
    fn packed_observation_tail_only_and_stale_fresh_reproducers_are_rejected() {
        let (mut recipes, mut rejected) = (0, 0);
        for c in raw_plan() {
            let dtype = c.case.dtype.unwrap();
            if dtype == DType::Flex32 {
                continue;
            }
            let e = c.expected(false).unwrap();
            if !e.policies.contains(&Policy::Observation) {
                continue;
            }
            assert_eq!(e.values.len, 1);
            assert_eq!(e.policies, &[Policy::Observation]);
            let baseline = words(e.values.bytes());
            let mut stale_trials = Vec::new();
            for poison in POISONS {
                let tail_only = [poison & 0xffff];
                let guarded = [GUARD, tail_only[0], GUARD];
                // Preserve the R2 numerical-only reproducer. Observation is
                // still not a promise about cancellation arithmetic.
                assert!(check_output(e, check_guards(&guarded, 1).unwrap()).is_ok());
                assert!(check_output(e, &tail_only).is_ok());
                assert!(check_packed_overwrite(dtype, 1, &baseline, &[poison], &tail_only).is_err());
                assert!(check_packed_coherence(dtype, 1, &baseline, &tail_only).is_err());
                let derived = scratch_poisons(&tail_only, SCRATCH_POISON_MASKS[0]);
                assert!(check_packed_overwrite(dtype, 1, &tail_only, &derived, &baseline).is_err());
                stale_trials.push(tail_only);
                rejected += 2;
            }
            assert!(check_packed_coherence(dtype, 1, &stale_trials[0], &stale_trials[1]).is_err());
            recipes += 1;
        }
        assert_eq!((recipes, rejected), (5, 20));
        println!("R2_PACKED_REPRO_REJECTED recipes={recipes} old_checker_paths={rejected}");
    }
    #[test]
    fn every_live_packed_half_requires_a_store_even_when_tail_and_sibling_are_written() {
        let (mut low, mut high, mut odd_tails) = (0, 0, 0);
        for c in raw_plan() {
            let dtype = c.case.dtype.unwrap();
            if dtype == DType::Flex32 {
                continue;
            }
            let e = c.expected(false).unwrap();
            let baseline = words(e.values.bytes());
            let count = e.values.len;
            for mask in SCRATCH_POISON_MASKS {
                let poisons = scratch_poisons(&baseline, mask);
                assert!(check_packed_overwrite(dtype, count, &baseline, &poisons, &baseline).is_ok());
                let mut tail_only = poisons.clone();
                if count % 2 == 1 {
                    *tail_only.last_mut().unwrap() &= 0xffff;
                }
                assert!(check_packed_overwrite(dtype, count, &baseline, &poisons, &tail_only).is_err());
                for i in 0..count {
                    let mut partial = baseline.clone();
                    let shift = (i % 2) * 16;
                    let half_mask = 0xffffu32 << shift;
                    partial[i / 2] = (partial[i / 2] & !half_mask) | (poisons[i / 2] & half_mask);
                    assert!(check_packed_overwrite(dtype, count, &baseline, &poisons, &partial).is_err());
                    if i % 2 == 0 {
                        low += 1;
                    } else {
                        high += 1;
                    }
                }
                if count % 2 == 1 {
                    let mut bad_tail = baseline.clone();
                    *bad_tail.last_mut().unwrap() |= 0xa53c_0000;
                    assert!(check_packed_overwrite(dtype, count, &baseline, &poisons, &bad_tail).is_err());
                    odd_tails += 1;
                }
                let mut guarded = vec![GUARD];
                guarded.extend_from_slice(&baseline);
                guarded.push(GUARD);
                assert!(check_guards(&guarded, baseline.len()).is_ok());
                guarded[0] ^= 1;
                assert!(check_guards(&guarded, baseline.len()).is_err());
            }
        }
        assert!(low > 1000 && high > 1000 && odd_tails > 0);
        println!("PACKED_HALF_PARTIALS_REJECTED low={low} high={high} odd_tails={odd_tails}");
    }
    #[test]
    fn packed_poison_bounds_preserve_every_legal_nan_payload_and_both_zero_signs() {
        for dtype in [DType::F16, DType::BF16] {
            let (nan, inf, one) = if dtype == DType::F16 {
                (0x7e01u16, 0x7c00u16, 0x3c00u16)
            } else {
                (0x7fc1u16, 0x7f80u16, 0x3f80u16)
            };
            let baseline_bits = [nan, 0, nan | 0x8000, 0x8000, inf, inf | 0x8000, 1, one, one + 1];
            let pack = |bits: &[u16]| words(&oracle::pack(dtype, &bits.iter().map(|&b| u64::from(b)).collect::<Vec<_>>()));
            let baseline = pack(&baseline_bits);
            for mask in SCRATCH_POISON_MASKS {
                let poisons = scratch_poisons(&baseline, mask);
                for bits in 0..=u16::MAX {
                    let class = oracle::classification(dtype, u64::from(bits));
                    for half_mask in [mask as u16, (mask >> 16) as u16] {
                        assert!(half_mask.to_le_bytes().iter().all(|&b| b != 0));
                        let poisoned = bits ^ half_mask;
                        assert_ne!(bits, poisoned);
                        if matches!(class, oracle::Class::NaN | oracle::Class::Infinity | oracle::Class::Zero) {
                            // This bounds every legal payload/sign variant:
                            // none can equal a finite, nonzero poison half.
                            assert_eq!(oracle::classification(dtype, u64::from(poisoned)), oracle::Class::Finite);
                        }
                    }
                    if class == oracle::Class::NaN {
                        let mut legal = baseline_bits;
                        legal[0] = bits;
                        legal[2] = bits ^ 0x8000;
                        legal[1] = 0x8000;
                        legal[3] = 0;
                        assert!(check_packed_overwrite(dtype, legal.len(), &baseline, &poisons, &pack(&legal)).is_ok());
                    }
                }
                let mut wrong = baseline_bits;
                wrong[4] ^= 0x8000;
                assert!(check_packed_coherence(dtype, wrong.len(), &baseline, &pack(&wrong)).is_err());
                wrong = baseline_bits;
                wrong[7] ^= 1;
                assert!(check_packed_coherence(dtype, wrong.len(), &baseline, &pack(&wrong)).is_err());
                assert!(check_packed_overwrite(dtype, baseline_bits.len(), &baseline, &poisons[..poisons.len() - 1], &baseline).is_err());
            }
        }
    }
    #[test]
    fn frozen_scratch_classes_are_independent_of_destination_underflow_and_overflow() {
        let c = fixture("f16_operand_range_and_output_overflow").unwrap();
        let scratch = oracle::scratch_expectation(c).unwrap();
        let output = main_expected(c).unwrap();
        assert_eq!(scratch.values.dtype, DType::F32);
        assert_eq!(oracle::classification(DType::F32, scratch.values.bits()[0]), oracle::Class::Finite);
        assert_eq!(oracle::classification(DType::F16, output.values.bits()[0]), oracle::Class::Infinity);
        let c = fixture("f16_accumulator_or_output_underflow").unwrap();
        let scratch = oracle::scratch_expectation(c).unwrap();
        let output = main_expected(c).unwrap();
        assert_ne!(scratch.values.bits()[1], 0);
        assert_eq!(output.values.bits()[1], 0);
    }
    #[test]
    fn every_counter_slot_and_observer_adjustment_is_required() {
        let before = Snapshot {
            kernels: [abi::ProfileEntry::default(); abi::KERNEL_COUNT],
            services: [abi::ServiceEntry::default(); abi::SERVICE_OPERATIONS],
        };
        let mut after = before.clone();
        after.services[abi::PROFILE as usize].calls = 1;
        after.services[abi::SERVICE_PROFILE as usize].calls = 1;
        assert!(check_counters(&before, &after, &Work::default()).is_ok());
        for i in 0..abi::KERNEL_COUNT {
            let mut bad = after.clone();
            bad.kernels[i].calls += 1;
            assert!(check_counters(&before, &bad, &Work::default()).is_err());
        }
        for i in 0..abi::SERVICE_OPERATIONS {
            let mut bad = after.clone();
            bad.services[i].calls += 1;
            assert!(check_counters(&before, &bad, &Work::default()).is_err());
        }
        let mut waited = after.clone();
        waited.kernels[Kernel::MatmulF16 as usize].wait_cycles = 1;
        assert!(check_counters(&before, &waited, &Work::default()).is_err());
        let mut download = Work::default();
        download.services[abi::DOWNLOAD as usize] = 1;
        let mut after_read = after;
        after_read.services[abi::DOWNLOAD as usize].calls += 1;
        assert!(check_counters(&before, &after_read, &download).is_ok());
    }
    #[test]
    fn public_fixtures_preserve_dtype_and_extract_the_inner_native_primitive() {
        use d3os_ai::burn::tensor::Device;
        let device = Device::new(NativeDevice);
        for prefix in ["f16", "bf16", "flex32"] {
            let case = fixture(&format!("{prefix}_dense_odd")).unwrap();
            let matrix = case.matrix("lhs").unwrap();
            let tensor = public_tensor::<2>(matrix, &device);
            // Host Native tensors retain metadata and deferred errors without
            // executing device work. The constructor must select this dtype
            // explicitly rather than the device's default F32 dtype.
            assert_eq!(tensor.dtype(), matrix.values.dtype);
            assert!(tensor.is_autodiff());
            assert!(tensor.clone().try_into_primitive::<Native>().is_err());
            let inner = tensor.inner();
            assert!(!inner.is_autodiff());
            let primitive = inner.try_into_primitive::<Native>().unwrap();
            assert_eq!(primitive.dtype(), matrix.values.dtype);
            assert_eq!(primitive.shape().as_slice(), matrix.values.shape);
        }
    }
    #[test]
    fn malformed_packet_and_alias_rules_reject_before_a_writer_can_exist() {
        for kind in Kind::ALL {
            for p in bad_parameters() {
                let p: Vec<_> = p.into_iter().map(|n| n as u32).collect();
                assert!(kind.kernel().work(&p).is_err());
            }
            let w = kind.kernel().work(&[3, 5, 7, 0, 0]).unwrap();
            assert!(w.validate_aliases(&[0x100004, 0x100004, 0x800004]).is_ok());
            for addresses in [[0x100004, 0x200004, 0x100004], [0x100004, 0x200004, 0x200008]] {
                assert!(w.validate_aliases(&addresses).is_err());
            }
        }
    }
    #[test]
    fn count_inventory_is_pure_bounded_and_keeps_diagnostics_separate() {
        let plan = raw_plan();
        let a = expected_counts(&plan).unwrap();
        let b = expected_counts(&plan).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, EXPECTED);
        assert_eq!(
            (a.raw_cases, a.guarded_scratch, a.fresh_scratch, a.guarded_cast, a.fresh_cast),
            (33, 66, 66, 44, 44)
        );
        assert_eq!((a.empty_aliases, a.zero_k, a.deferred, a.queued, a.autodiff), (9, 3, 15, 3, 21));
        assert!(a.mandatory_values > 0 && a.diagnostic_values > 0);
        println!("COUNTS {a:?}");
        println!("RAW_PLAN {}", plan_json(&plan).unwrap());
        println!("SHARED {}", oracle::summary_json(&oracle::expected_summary(&options(), Scope::Shared)));
    }
}
