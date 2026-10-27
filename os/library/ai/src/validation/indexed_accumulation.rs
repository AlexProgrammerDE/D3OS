//! Immutable indexed accumulation references and selected public Dispatch checks.
//!
//! Shared reads prove logical values and retained source-clone lineage. Physical
//! allocation, original tails, ownership and residency require separate Raw proof.
//! Symbolic attachments are MODEL evidence and never enter device row counts.
use super::float_reduction::{self as ieee, Arithmetic};
use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use burn::{
    backend::{
        Backend, DType, Dispatch, DispatchTensor, ExecutionError, Shape, TensorData, TensorMetadata,
        ops::{FloatTensorOps, IntTensorOps},
    },
    tensor::{Device, IndexingUpdateOp, Int, Slice, Tensor, module},
};
use core::fmt::Write;
mod fixtures;

pub const CORPUS_VERSION: u32 = 1;
pub const DTYPES: [DType; 13] = [
    DType::F64,
    DType::F32,
    DType::Flex32,
    DType::F16,
    DType::BF16,
    DType::I64,
    DType::I32,
    DType::I16,
    DType::I8,
    DType::U64,
    DType::U32,
    DType::U16,
    DType::U8,
];
pub const NUMERICAL_FREEZE_SHA256: &str = "e159c1b3bd1f6f78bf7b64913e30765d6d167e4fb335fbcf499ec94d648521c8";
const MAX_FAILURES: usize = 16;
const MAX_MISMATCHES: usize = 8;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionScope {
    LinuxCommon,
    NativeValues,
    NativeErrors,
    ModelReference,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub scope: ExecutionScope,
    pub dtype: Option<DType>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            scope: ExecutionScope::LinuxCommon,
            dtype: None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Add,
    Mul,
    Min,
    Max,
    Assign,
}
impl Operation {
    fn update(self) -> IndexingUpdateOp {
        match self {
            Self::Add => IndexingUpdateOp::Add,
            Self::Mul => IndexingUpdateOp::Mul,
            Self::Min => IndexingUpdateOp::Min,
            Self::Max => IndexingUpdateOp::Max,
            Self::Assign => IndexingUpdateOp::Assign,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Scatter,
    SelectUpdate,
    ScatterNd,
    GatherBackward,
    SelectBackward,
    GatherNdBackward,
    EmbeddingBackward,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    LegacyExactFiniteGuard,
    AcceptedAssignBitwiseGuard,
    DeclaredWidthInteger,
    NativeSerialC3,
    PublicLinearAdjointExact,
    NativeRecoverableSemanticError,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticError {
    IndexRange,
    CoordinateBounds,
}
#[derive(Clone, Copy, Debug)]
pub enum ViewStep {
    Reshape(&'static [usize]),
    Slice { start: usize, end: usize, step: usize },
    Permute(&'static [usize]),
    Flip(&'static [usize]),
    Expand(&'static [usize]),
}
#[derive(Clone, Copy, Debug)]
pub struct SourceSpec {
    pub role: &'static str,
    pub storage_id: &'static str,
    pub dtype: DType,
    pub storage_shape: &'static [usize],
    pub storage_bits: &'static [u64],
    pub shape: &'static [usize],
    pub strides_elements: &'static [i64],
    pub offset_elements: usize,
    pub logical_to_storage: &'static [usize],
    pub construction: &'static [ViewStep],
}
impl SourceSpec {
    pub fn logical_bits(&self) -> Vec<u64> {
        self.logical_to_storage.iter().map(|&i| self.storage_bits[i]).collect()
    }
}
#[derive(Clone, Copy, Debug)]
pub struct ValueSpec {
    pub dtype: DType,
    pub shape: &'static [usize],
    pub bits: &'static [u64],
}
#[derive(Clone, Copy, Debug)]
pub struct GradientSpec {
    pub forward: ValueSpec,
    pub leaf_gradient: ValueSpec,
    pub root: ValueSpec,
    pub seed: ValueSpec,
    pub upstream: &'static [u64],
}
#[derive(Clone, Copy, Debug)]
pub struct PublicGraph {
    pub branches: usize,
    pub product: ValueSpec,
    pub branch_sum: ValueSpec,
}
#[derive(Clone, Copy, Debug)]
pub enum Expected {
    Value(ValueSpec),
    Error(SemanticError),
    Gradient(&'static GradientSpec),
}
#[derive(Debug)]
pub struct Case {
    pub id: &'static str,
    pub dtype: DType,
    pub index_dtype: DType,
    pub operation: Operation,
    pub route: Route,
    pub family: &'static str,
    pub design_rank: usize,
    pub view_profile: &'static str,
    pub axis: Option<usize>,
    pub sources: &'static [SourceSpec],
    pub normalized_indices: Option<&'static [i32]>,
    pub expected: Expected,
    pub public_graph: Option<PublicGraph>,
    pub policy: Policy,
    /// Exact reviewed row. Execution membership never rewrites this metadata.
    pub frozen_json: &'static str,
    pub row_sha256: &'static str,
}
impl Case {
    fn source(&self, role: &str) -> &SourceSpec {
        self.sources.iter().find(|s| s.role == role).expect("generated source role")
    }
}
#[derive(Clone, Copy, Debug)]
pub enum Node {
    Null,
    Bool(bool),
    Integer(i128),
    String(&'static str),
    Array(&'static [Node]),
    Object(&'static [(&'static str, Node)]),
}
impl Node {
    fn get(&self, key: &str) -> Result<&Node, String> {
        if let Self::Object(fields) = self {
            fields
                .iter()
                .find_map(|(k, v)| (*k == key).then_some(v))
                .ok_or_else(|| format!("missing model field {key}"))
        } else {
            Err(String::from("model object required"))
        }
    }
    fn optional(&self, key: &str) -> Option<&Node> {
        self.get(key).ok()
    }
    fn array(&self) -> Result<&'static [Node], String> {
        if let Self::Array(v) = self {
            Ok(v)
        } else {
            Err(String::from("model array required"))
        }
    }
    fn integer(&self) -> Result<i128, String> {
        if let Self::Integer(v) = self {
            Ok(*v)
        } else {
            Err(String::from("model integer required"))
        }
    }
    fn text(&self) -> Result<&'static str, String> {
        if let Self::String(v) = self {
            Ok(v)
        } else {
            Err(String::from("model string required"))
        }
    }
    fn boolean(&self) -> Result<bool, String> {
        if let Self::Bool(v) = self {
            Ok(*v)
        } else {
            Err(String::from("model boolean required"))
        }
    }
    fn bits(&self) -> Result<u64, String> {
        u64::from_str_radix(self.text()?.strip_prefix("0x").ok_or("hex prefix required")?, 16).map_err(|e| format!("model bits: {e}"))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AttachmentKey {
    pub template: &'static str,
    pub content_sha256: &'static str,
}
#[derive(Debug)]
pub struct ModelAttachment {
    pub key: AttachmentKey,
    pub label: &'static str,
    pub kind: &'static str,
    pub dtype: Option<DType>,
    pub class_pattern: bool,
    pub frozen_json: &'static str,
    pub data: Node,
}
#[derive(Debug)]
pub struct SymbolicTemplate {
    pub id: &'static str,
    pub name: &'static str,
    pub attachments: &'static [ModelAttachment],
}
pub fn corpus() -> &'static [Case] {
    fixtures::CASES
}
pub fn symbolic_corpus() -> &'static [SymbolicTemplate] {
    fixtures::SYMBOLIC
}
pub fn fixtures_sha256() -> &'static str {
    fixtures::FIXTURES_SHA256
}
pub fn structured_policy_json() -> &'static str {
    fixtures::POLICY_JSON
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CheckKind {
    InputStorage,
    InputView,
    RetainedStorage,
    RetainedView,
    Output,
    PublicStage,
    SemanticError,
    NoPublishedOutput,
    Recovery,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ObservationKey {
    pub case_id: &'static str,
    pub check: CheckKind,
    pub subject: &'static str,
}
fn key(case: &Case, check: CheckKind, subject: &'static str) -> ObservationKey {
    ObservationKey {
        case_id: case.id,
        check,
        subject,
    }
}
fn branch_names(branch: usize) -> (&'static str, &'static str) {
    if branch == 0 {
        ("branch_1_product", "branch_1_sum")
    } else {
        ("branch_2_product", "branch_2_sum")
    }
}
fn required_keys(case: &Case) -> BTreeSet<ObservationKey> {
    let mut keys = BTreeSet::new();
    for source in case.sources {
        keys.insert(key(case, CheckKind::InputStorage, source.storage_id));
        keys.insert(key(case, CheckKind::RetainedStorage, source.storage_id));
        keys.insert(key(case, CheckKind::InputView, source.role));
        keys.insert(key(case, CheckKind::RetainedView, source.role));
    }
    match case.expected {
        Expected::Value(_) => {
            keys.insert(key(case, CheckKind::Output, "result"));
        }
        Expected::Error(_) => {
            for (kind, subject) in [
                (CheckKind::SemanticError, "result"),
                (CheckKind::NoPublishedOutput, "result"),
                (CheckKind::Recovery, "same_context_valid_operation"),
            ] {
                keys.insert(key(case, kind, subject));
            }
        }
        Expected::Gradient(_) => {
            for name in ["forward", "root", "original_leaf_gradient"] {
                keys.insert(key(case, CheckKind::PublicStage, name));
            }
            for branch in 0..case.public_graph.expect("generated graph").branches {
                let (product, sum) = branch_names(branch);
                keys.insert(key(case, CheckKind::PublicStage, product));
                keys.insert(key(case, CheckKind::PublicStage, sum));
            }
        }
    }
    keys
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectionError {
    UnsupportedDtype,
    EmptyPopulation,
}
#[derive(Clone, Debug)]
pub struct Population {
    options: Options,
    cases: Vec<&'static Case>,
    models: Vec<&'static ModelAttachment>,
    keys: BTreeSet<ObservationKey>,
}
impl Population {
    pub fn cases(&self) -> &[&'static Case] {
        &self.cases
    }
    pub fn models(&self) -> &[&'static ModelAttachment] {
        &self.models
    }
    pub fn required_observations(&self) -> &BTreeSet<ObservationKey> {
        &self.keys
    }
}
pub fn population(options: &Options) -> Result<Population, SelectionError> {
    if options.dtype.is_some_and(|d| !DTYPES.contains(&d)) {
        return Err(SelectionError::UnsupportedDtype);
    }
    let cases: Vec<_> = corpus()
        .iter()
        .filter(|c| {
            options.dtype.is_none_or(|d| c.dtype == d)
                && match options.scope {
                    ExecutionScope::LinuxCommon => c.dtype != DType::F64 && !matches!(c.expected, Expected::Error(_)),
                    ExecutionScope::NativeValues => !matches!(c.expected, Expected::Error(_)),
                    ExecutionScope::NativeErrors => matches!(c.expected, Expected::Error(_)),
                    ExecutionScope::ModelReference => false,
                }
        })
        .collect();
    let models: Vec<_> = if options.scope == ExecutionScope::ModelReference {
        symbolic_corpus()
            .iter()
            .flat_map(|b| b.attachments)
            .filter(|a| options.dtype.is_none_or(|d| a.dtype == Some(d)))
            .collect()
    } else {
        Vec::new()
    };
    if cases.is_empty() && models.is_empty() {
        return Err(SelectionError::EmptyPopulation);
    }
    let keys = cases.iter().flat_map(|c| required_keys(c)).collect();
    Ok(Population {
        options: *options,
        cases,
        models,
        keys,
    })
}
#[derive(Clone, Debug)]
pub struct ExpectedSummary {
    pub scope: ExecutionScope,
    pub cases: usize,
    pub values: usize,
    pub errors: usize,
    pub public_gradients: usize,
    pub templates: usize,
    pub attachments: usize,
    pub class_patterns_model: usize,
    pub source_roles: usize,
    pub source_storages: usize,
    pub observation_keys: usize,
}
pub fn expected_summary(options: &Options) -> Result<ExpectedSummary, SelectionError> {
    let p = population(options)?;
    Ok(ExpectedSummary {
        scope: options.scope,
        cases: p.cases.len(),
        values: p.cases.iter().filter(|c| matches!(c.expected, Expected::Value(_))).count(),
        errors: p.cases.iter().filter(|c| matches!(c.expected, Expected::Error(_))).count(),
        public_gradients: p.cases.iter().filter(|c| matches!(c.expected, Expected::Gradient(_))).count(),
        templates: p.models.iter().map(|a| a.key.template).collect::<BTreeSet<_>>().len(),
        attachments: p.models.len(),
        class_patterns_model: p.models.iter().filter(|a| a.class_pattern).count(),
        source_roles: p.cases.iter().map(|c| c.sources.len()).sum(),
        source_storages: p.keys.iter().filter(|k| k.check == CheckKind::InputStorage).count(),
        observation_keys: p.keys.len(),
    })
}
fn quoted(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
fn debug_json(value: impl core::fmt::Debug) -> String {
    quoted(&format!("{value:?}"))
}
fn list_json(items: impl IntoIterator<Item = String>) -> String {
    format!("[{}]", items.into_iter().collect::<Vec<_>>().join(","))
}
fn key_json(k: &ObservationKey) -> String {
    format!(
        "{{\"id\":{},\"check\":{},\"subject\":{}}}",
        quoted(k.case_id),
        debug_json(k.check),
        quoted(k.subject)
    )
}
fn attachment_key_json(k: AttachmentKey) -> String {
    format!("{{\"template\":{},\"content_sha256\":{}}}", quoted(k.template), quoted(k.content_sha256))
}
pub fn summary_json(s: &ExpectedSummary) -> String {
    format!(
        "{{\"scope\":{},\"cases\":{},\"values\":{},\"errors\":{},\"public_gradients\":{},\"templates\":{},\"attachments\":{},\"class_patterns_MODEL\":{},\"source_roles\":{},\"source_storages\":{},\"observation_keys\":{},\"device_initialized\":false,\"FROZEN\":false,\"source_binding_approved\":false}}",
        debug_json(s.scope),
        s.cases,
        s.values,
        s.errors,
        s.public_gradients,
        s.templates,
        s.attachments,
        s.class_patterns_model,
        s.source_roles,
        s.source_storages,
        s.observation_keys
    )
}
pub fn population_json(p: &Population) -> String {
    let cases = list_json(p.cases.iter().map(|c| {
        format!(
            "{{\"frozen_row\":{},\"row_sha256\":{},\"execution_policy\":{}}}",
            c.frozen_json,
            quoted(c.row_sha256),
            debug_json(c.policy)
        )
    }));
    let models = list_json(p.models.iter().map(|a| {
        format!(
            "{{\"key\":{},\"evidence\":\"MODEL\",\"attachment\":{}}}",
            attachment_key_json(a.key),
            a.frozen_json
        )
    }));
    let selected: BTreeSet<_> = p.cases.iter().map(|c| c.id).collect();
    let excluded = list_json(corpus().iter().filter(|c| !selected.contains(c.id)).map(|c| {
        format!(
            "{{\"id\":{},\"reason\":{}}}",
            quoted(c.id),
            quoted(if matches!(c.expected, Expected::Error(_)) {
                "Native_error_scope_only"
            } else if p.options.scope == ExecutionScope::LinuxCommon && c.dtype == DType::F64 {
                "public_CUDA_F64_source_declined"
            } else {
                "outside_selected_scope_or_dtype"
            })
        )
    }));
    format!(
        "{{\"corpus_version\":1,\"scope\":{},\"dtype_filter\":{},\"fixtures_sha256\":{},\"numerical_freeze_sha256\":{},\"FROZEN\":false,\"source_binding_approved\":false,\"observation\":\"NOT_EXECUTED\",\"structured_policies\":{},\"cases\":{},\"models\":{},\"excluded\":{},\"required_observations\":{}}}",
        debug_json(p.options.scope),
        p.options.dtype.map_or_else(|| String::from("null"), debug_json),
        quoted(fixtures_sha256()),
        quoted(NUMERICAL_FREEZE_SHA256),
        structured_policy_json(),
        cases,
        models,
        excluded,
        list_json(p.keys.iter().map(key_json))
    )
}

fn width(dtype: DType) -> u32 {
    (dtype.size() * 8) as u32
}
fn mask(dtype: DType) -> u64 {
    u64::MAX >> (64 - width(dtype))
}
fn signed_value(dtype: DType, bits: u64) -> i128 {
    let v = i128::from(bits & mask(dtype));
    if dtype.is_int() && bits & (1 << (width(dtype) - 1)) != 0 {
        v - (1i128 << width(dtype))
    } else {
        v
    }
}
fn fields(dtype: DType) -> (u32, u32) {
    match dtype {
        DType::F16 => (5, 10),
        DType::BF16 => (8, 7),
        DType::F64 => (11, 52),
        _ => (8, 23),
    }
}
fn nan(dtype: DType, bits: u64) -> bool {
    let (e, f) = fields(dtype);
    bits & ((1 << (e + f)) - 1) > ((1 << e) - 1) << f
}
/// Exact selected Native scalar step. This is a host reference, never a fallback.
pub fn reference_step(dtype: DType, a: u64, b: u64, op: Operation) -> u64 {
    if op == Operation::Assign {
        return b;
    }
    if !dtype.is_float() {
        return match op {
            Operation::Add => a.wrapping_add(b) & mask(dtype),
            Operation::Mul => a.wrapping_mul(b) & mask(dtype),
            Operation::Min => {
                if signed_value(dtype, a) <= signed_value(dtype, b) {
                    a
                } else {
                    b
                }
            }
            Operation::Max => {
                if signed_value(dtype, a) >= signed_value(dtype, b) {
                    a
                } else {
                    b
                }
            }
            Operation::Assign => b,
        };
    }
    if matches!(op, Operation::Min | Operation::Max) {
        let (_, f) = fields(dtype);
        if nan(dtype, a) || nan(dtype, b) {
            return if nan(dtype, a) && nan(dtype, b) {
                a | (1 << (f - 1))
            } else if nan(dtype, a) {
                b
            } else {
                a
            };
        }
        let sign = 1u64 << (width(dtype) - 1);
        if a & !sign == 0 && b & !sign == 0 {
            return if op == Operation::Min { a | b } else { a & b };
        }
        let order = |v: u64| if v & sign != 0 { !v & mask(dtype) } else { v ^ sign };
        return if (op == Operation::Min && order(a) <= order(b)) || (op == Operation::Max && order(a) >= order(b)) {
            a
        } else {
            b
        };
    }
    let compute = if matches!(dtype, DType::F16 | DType::BF16) { DType::F32 } else { dtype };
    let result = ieee::ieee_binary(
        compute,
        ieee::cast_bits(dtype, compute, a),
        ieee::cast_bits(dtype, compute, b),
        if op == Operation::Add { Arithmetic::Add } else { Arithmetic::Multiply },
    );
    ieee::cast_bits(compute, dtype, result)
}
fn parse_dtype(text: &str) -> Result<DType, String> {
    match text {
        "F64" => Ok(DType::F64),
        "F32" => Ok(DType::F32),
        "Flex32" => Ok(DType::Flex32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "I64" => Ok(DType::I64),
        "I32" => Ok(DType::I32),
        "I16" => Ok(DType::I16),
        "I8" => Ok(DType::I8),
        "U64" => Ok(DType::U64),
        "U32" => Ok(DType::U32),
        "U16" => Ok(DType::U16),
        "U8" => Ok(DType::U8),
        _ => Err(String::from("unknown model dtype")),
    }
}
fn parse_operation(text: &str) -> Result<Operation, String> {
    match text {
        "Add" => Ok(Operation::Add),
        "Mul" => Ok(Operation::Mul),
        "Min" => Ok(Operation::Min),
        "Max" => Ok(Operation::Max),
        _ => Err(String::from("unknown model operation")),
    }
}
fn require(value: bool, message: &str) -> Result<(), String> {
    if value { Ok(()) } else { Err(message.into()) }
}
fn integers(node: &Node) -> Result<Vec<i128>, String> {
    node.array()?.iter().map(Node::integer).collect()
}
fn raw_bits(node: &Node) -> Result<Vec<u64>, String> {
    node.array()?.iter().map(Node::bits).collect()
}
fn checked_product(values: impl IntoIterator<Item = i128>) -> Option<i128> {
    values.into_iter().try_fold(1i128, |a, b| a.checked_mul(b))
}
fn index_error(index: i128, length: i128) -> Option<&'static str> {
    if !(0..=i128::from(i32::MAX)).contains(&index) {
        Some("recoverable_index_range")
    } else if index >= length {
        Some("recoverable_coordinate_bounds")
    } else {
        None
    }
}
fn expected_error(node: &Node) -> Result<Option<&'static str>, String> {
    if matches!(node, Node::Null) { Ok(None) } else { Ok(Some(node.text()?)) }
}
fn model_numeric(model: &ModelAttachment) -> Result<(), String> {
    let n = &model.data;
    let dtype = model.dtype.ok_or("numeric dtype missing")?;
    let op = parse_operation(n.get("operation")?.text()?)?;
    let mut accumulator = n.get("base_bits")?.bits()?;
    let updates = raw_bits(n.get("update_bits")?)?;
    let stored = raw_bits(n.get("stored_trace")?)?;
    require(updates.len() == stored.len(), "stored trace length")?;
    let compute_trace = n.get("compute_trace")?.array()?;
    for (i, (&update, &expected)) in updates.iter().zip(&stored).enumerate() {
        let previous = accumulator;
        if dtype.is_float() && matches!(op, Operation::Add | Operation::Mul) {
            let cf = if matches!(dtype, DType::F16 | DType::BF16 | DType::Flex32) {
                DType::F32
            } else {
                DType::F64
            };
            let operands = [ieee::cast_bits(dtype, cf, previous), ieee::cast_bits(dtype, cf, update)];
            let value = ieee::ieee_binary(
                cf,
                operands[0],
                operands[1],
                if op == Operation::Add { Arithmetic::Add } else { Arithmetic::Multiply },
            );
            let trace = compute_trace.get(i).ok_or("missing compute stage")?;
            if raw_bits(trace.get("operand_bits")?)? != operands
                || trace.get("result_bits")?.bits()? != value
                || parse_dtype(trace.get("dtype")?.text()?)? != cf
            {
                return Err(format!(
                    "MODEL compute stage={i} dtype={cf:?} actual_operands={operands:x?} actual={value:#x} expected={:?}",
                    trace
                ));
            }
        }
        accumulator = reference_step(dtype, previous, update, op);
        if accumulator != expected {
            return Err(format!(
                "MODEL store stage={i} dtype={dtype:?} actual_operands=[{previous:#x},{update:#x}] actual={accumulator:#x} expected={expected:#x}"
            ));
        }
    }
    require(accumulator == n.get("expected_bits")?.bits()?, "numeric final bits")
}
fn proposed_validator(n: &Node) -> Result<bool, String> {
    let p = integers(n.get("parameters")?)?;
    let buffers = integers(n.get("buffer_words")?)?;
    let starts = integers(n.get("addresses")?)?;
    if p.len() != 32 || buffers.len() != 4 || starts.len() != 4 {
        return Ok(false);
    }
    let rank = (p[0] & 255) as usize;
    let fmt = ((p[0] >> 8) & 255) as usize;
    let op = (p[0] >> 16) & 255;
    if !(1..=8).contains(&rank) || p[0] >> 24 != 0 || fmt > 12 || op > 3 {
        return Ok(false);
    }
    let dims = &p[8..8 + rank];
    if p[1] < 0 || p[1] >= rank as i128 || dims.iter().any(|&x| !(1..=i128::from(i32::MAX)).contains(&x)) || checked_product(dims.iter().copied()) != Some(p[2])
    {
        return Ok(false);
    }
    if p[8 + rank..16].iter().any(|&x| x != 1) || p[16 + rank..24].iter().chain(&p[24 + rank..32]).any(|&x| x != 0) {
        return Ok(false);
    }
    let mut ds = dims.to_vec();
    ds[p[1] as usize] = p[7];
    let mut stride = 1;
    for i in (0..rank).rev() {
        if p[16 + i] != stride {
            return Ok(false);
        }
        stride *= ds[i];
    }
    if p[4] != 0 || checked_product(ds) != Some(p[3]) || [p[2], p[3], p[5], p[7]].iter().any(|&x| !(1..=i128::from(i32::MAX)).contains(&x)) {
        return Ok(false);
    }
    let (mut lo, mut hi) = (p[6], p[6]);
    for (&d, &s) in dims.iter().zip(&p[24..24 + rank]) {
        let s = if s & (1 << 31) != 0 { s - (1 << 32) } else { s };
        let delta = (d - 1) * s;
        lo += delta.min(0);
        hi += delta.max(0);
    }
    if lo < 0 || hi < lo || hi >= p[5] {
        return Ok(false);
    }
    let width = [64, 32, 32, 16, 16, 64, 32, 16, 8, 64, 32, 16, 8][fmt];
    let needs = [(p[2] * width + 31) / 32, p[5], 1, (p[3] * width + 31) / 32];
    if buffers.iter().zip(needs).any(|(&a, b)| a < b) {
        return Ok(false);
    }
    let ends: Vec<_> = starts.iter().zip(needs).map(|(&a, b)| a + b * 4).collect();
    if starts.iter().any(|&x| x < 0) || ends.iter().any(|&x| x > i128::from(u64::MAX)) {
        return Ok(false);
    }
    Ok(!(2..4).any(|a| (0..4).any(|b| a != b && starts[a] < ends[b] && starts[b] < ends[a])))
}
fn packed(bits: &[u64], dtype: DType, fill: u32) -> Vec<u32> {
    let w = width(dtype) as usize;
    let mut out = vec![fill; (bits.len() * w).div_ceil(32).max(1)];
    for (i, &value) in bits.iter().enumerate() {
        if w == 64 {
            out[2 * i] = value as u32;
            out[2 * i + 1] = (value >> 32) as u32;
        } else {
            let shift = i * w % 32;
            let mask = if w == 32 { u32::MAX } else { (1u32 << w) - 1 };
            out[i * w / 32] = (out[i * w / 32] & !(mask << shift)) | ((value as u32 & mask) << shift);
        }
    }
    out
}
/// Evaluate a bounded attachment using host algebra. It never creates a tensor.
pub fn check_model(model: &ModelAttachment) -> Result<(), String> {
    let n = &model.data;
    match model.kind {
        "numeric_reference" => model_numeric(model),
        "proposed_validator_model" => require(proposed_validator(n)? == n.get("model_accepts")?.boolean()?, "proposed validator result"),
        "descriptor" => require(
            (1..=8).contains(&n.get("valid_rank")?.integer()?) && !(1..=8).contains(&n.get("reject_rank")?.integer()?),
            "rank bound",
        ),
        "descriptor_error" => {
            let product = checked_product(integers(n.get("shape")?)?).ok_or("shape algebra overflow")?;
            require(
                product == n.get("exact_product")?.integer()? && product > i128::from(i32::MAX),
                "count overflow rejection",
            )
        }
        "descriptor_algebra" => {
            if let Some(points) = n.optional("points") {
                for point in points.array()? {
                    let j = point.get("logical_index")?.integer()?;
                    let words = raw_bits(point.get("u32_word_offsets")?)?;
                    let wide = raw_bits(point.get("zero_extended_u64_offsets")?)?;
                    let bytes = integers(point.get("positive_byte_offsets")?)?;
                    require(
                        words == [2 * j as u64, 2 * j as u64 + 1] && words == wide && bytes == [2 * j * 4, (2 * j + 1) * 4],
                        "wide pointer zero extension",
                    )?;
                }
                for masked in n.get("masked")?.array()? {
                    let j = masked.get("raw_logical")?.integer()?;
                    let active = masked.get("active")?.boolean()?;
                    let safe = if active && (0..i128::from(i32::MAX)).contains(&j) { j } else { 0 };
                    require(integers(masked.get("safe_pair")?)? == [2 * safe, 2 * safe + 1], "masked pointer pair")?;
                }
                Ok(())
            } else if let Some(spans) = n.optional("spans") {
                for span in spans.array()? {
                    let shape = integers(span.get("shape")?)?;
                    let strides = integers(span.get("strides_elements")?)?;
                    let mut lo = span.get("offset")?.integer()?;
                    let mut hi = lo;
                    for (d, s) in shape.into_iter().zip(strides) {
                        let delta = (d - 1) * s;
                        lo += delta.min(0);
                        hi += delta.max(0);
                    }
                    let valid = 0 <= lo && lo <= hi && hi < span.get("storage_extent")?.integer()?;
                    require(
                        integers(span.get("span")?)? == [lo, hi] && valid == span.get("expected_valid")?.boolean()?,
                        "signed view span",
                    )?;
                }
                Ok(())
            } else {
                let max = n.get("valid_logical_max")?.integer()?;
                require(
                    max == i128::from(i32::MAX) && n.get("reject_logical")?.integer()? == max + 1 && n.get("physical_64_words")?.integer()? == 2 * max,
                    "count boundary",
                )
            }
        }
        "index_error" => {
            if let Some(source) = n.optional("source") {
                let dtype = parse_dtype(source.get("dtype")?.text()?)?;
                let values = raw_bits(source.get("storage_bits")?)?;
                let invalid = values.iter().any(|&b| !(0..=i128::from(i32::MAX)).contains(&signed_value(dtype, b)));
                require(
                    invalid && n.get("expected")?.get("error")?.text()? == "recoverable_index_range",
                    "exact index range",
                )
            } else {
                let dtype = parse_dtype(n.get("index_dtype")?.text()?)?;
                let invalid = raw_bits(n.get("index_bits")?)?
                    .iter()
                    .any(|&b| !(0..=i128::from(i32::MAX)).contains(&signed_value(dtype, b)));
                require(invalid, "ND validation before linearization")
            }
        }
        "index_bounds" => require(
            index_error(n.get("index")?.integer()?, n.get("axis_length")?.integer()?) == expected_error(n.get("expected_error")?)?,
            "coordinate upper bound",
        ),
        "descriptor_index" => {
            let dtype = parse_dtype(n.get("dtype")?.text()?)?;
            let value = signed_value(dtype, n.get("index_bits")?.bits()?);
            let length = n.get("axis_length")?.integer()?;
            require(
                value == n.get("expected_normalized")?.integer()?
                    && index_error(value, length).is_none()
                    && (length as u64 & mask(dtype)) == n.get("forbidden_bound_cast")?.bits()?,
                "narrow dtype wide bound",
            )
        }
        "metadata_error_and_value" => {
            for axis in n.get("axis_cases")?.array()? {
                require(
                    axis.get("rank")?.integer()? == 0 && axis.get("axis")?.integer()? == 0,
                    "rank zero axis rejection",
                )?;
            }
            let scalar = n.get("nd_scalar")?;
            let index = raw_bits(scalar.get("index_bits")?)?[0] as usize;
            let source = raw_bits(scalar.get("source_bits")?)?;
            require(
                scalar.get("expected_shape")?.array()?.is_empty() && raw_bits(scalar.get("expected_bits")?)? == [source[index]],
                "scalar ND value",
            )
        }
        "metadata_error" => {
            for case in n.get("cases")?.array()? {
                let shape = integers(case.get("index_shape")?)?;
                require(shape.is_empty() || shape.contains(&0), "ND index metadata rejection")?;
            }
            Ok(())
        }
        "empty_reference" => {
            if let Some(base) = n.optional("base_bits") {
                require(
                    raw_bits(base)? == raw_bits(n.get("expected_bits")?)? && checked_product(integers(n.get("update_shape")?)?) == Some(0),
                    "empty base preservation",
                )?;
                for entry in n.get("all_dtype_empty_allocations")?.array()? {
                    require(
                        entry.get("logical_elements")?.integer()? == 0
                            && entry.get("unused_u32_words")?.integer()? == 1
                            && entry.get("element_dispatches")?.integer()? == 0,
                        "empty allocation descriptor",
                    )?;
                }
                Ok(())
            } else {
                require(
                    checked_product(integers(n.get("values_shape")?)?) == Some(0)
                        && n.get("expected_output_bits")?.array()?.is_empty()
                        && n.get("current_Native_coordinate_reads")?.integer()? == 0,
                    "empty trailing slice shortcut",
                )
            }
        }
        "packed_algebra" => {
            let dtype = model.dtype.ok_or("packed dtype")?;
            let base = raw_bits(n.get("base_bits")?)?;
            let at = n.get("logical_field_updated")?.integer()? as usize;
            let mut actual = base.clone();
            actual[at] = reference_step(dtype, base[at], base[at], Operation::Add);
            let before = packed(&base, dtype, 0xa5a5a5a5);
            let after = packed(&actual, dtype, 0xa5a5a5a5);
            require(
                actual == raw_bits(n.get("expected_bits")?)?
                    && before.iter().map(|&v| u64::from(v)).collect::<Vec<_>>() == raw_bits(n.get("before")?.get("u32_words")?)?
                    && after.iter().map(|&v| u64::from(v)).collect::<Vec<_>>() == raw_bits(n.get("after")?.get("u32_words")?)?,
                "packed neighbor and tail algebra",
            )
        }
        "counterexample_algebra" => {
            let initial = n.get("initial")?.bits()?;
            let committed = n.get("high_update_committed")?.bits()?;
            let replacement = n.get("low_update_stale_replacement")?.bits()?;
            let actual = (committed & 0xffff0000) | (replacement & 0xffff);
            require(
                initial != committed
                    && n.get("CAS_expected_mismatch")?.boolean()?
                    && actual == n.get("recomputed_replacement")?.bits()?
                    && replacement != actual,
                "stale whole word CAS counterexample",
            )
        }
        "converter_only" => {
            let one = ieee::integer_bits(DType::F64, 1);
            let exact = ieee::ieee_binary(
                DType::F64,
                ieee::ieee_binary(DType::F64, one, ((1023 - 8) as u64) << 52, Arithmetic::Add),
                ((1023 - 25) as u64) << 52,
                Arithmetic::Add,
            );
            let direct = ieee::cast_bits(DType::F64, DType::BF16, exact);
            let staged = ieee::cast_bits(DType::F32, DType::BF16, ieee::cast_bits(DType::F64, DType::F32, exact));
            require(
                direct == n.get("direct_BF16")?.bits()?
                    && staged == n.get("F32_then_BF16")?.bits()?
                    && direct != staged
                    && !n.get("reachable_stored_BF16_Add_Mul_counterexample_claimed")?.boolean()?,
                "converter-only double rounding",
            )
        }
        "analytic_gradient" => {
            let dtype = model.dtype.ok_or("gradient dtype")?;
            let base = n.get("base_bits")?.bits()?;
            let updates = raw_bits(n.get("update_bits")?)?;
            let upstream = n.get("upstream_bits")?.bits()?;
            require(
                updates == [base]
                    && reference_step(dtype, base, updates[0], parse_operation(n.get("operation")?.text()?)?) == base
                    && n.get("expected_base_gradient_bits")?.bits()? == upstream
                    && raw_bits(n.get("expected_update_gradient_bits")?)? == [upstream],
                "inclusive unique tie subgradient",
            )
        }
        "error_sequence_model" => require(
            index_error(n.get("first_index")?.integer()?, 4) == Some("recoverable_index_range")
                && index_error(n.get("second_index")?.integer()?, 4).is_none()
                && n.get("second_index")?.integer()? == n.get("second_expected_normalized")?.integer()?,
            "error sequence model",
        ),
        _ => Err(format!("unhandled model kind {}", model.kind)),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completion {
    Running,
    Finished,
    Aborted,
    Rejected,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bits: Vec<u64>,
}
impl From<ValueSpec> for Snapshot {
    fn from(v: ValueSpec) -> Self {
        Self {
            dtype: v.dtype,
            shape: v.shape.to_vec(),
            bits: v.bits.to_vec(),
        }
    }
}
#[derive(Clone, Debug)]
pub struct OperandObservation {
    pub key: ObservationKey,
    pub actual: Option<Snapshot>,
    pub unavailable: Option<String>,
}
#[derive(Clone, Debug)]
pub struct StageObservation {
    pub key: ObservationKey,
    pub expected: Snapshot,
    pub actual: Option<Snapshot>,
    pub read_error: Option<String>,
    pub actual_operands: Vec<OperandObservation>,
}
#[derive(Clone, Debug)]
pub struct SemanticObservation {
    pub case_id: &'static str,
    pub expected: SemanticError,
    pub actual_error: Option<String>,
    pub output_metadata_valid: bool,
    pub accepted: bool,
}
#[derive(Clone, Debug)]
pub struct Failure {
    pub key: Option<ObservationKey>,
    pub attachment: Option<AttachmentKey>,
    pub message: String,
    pub mismatches: Vec<(usize, u64, u64)>,
    pub actual_operands: Vec<OperandObservation>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceCounts {
    pub attempted: usize,
    pub passed: usize,
    pub failed: usize,
}
/// Only the runner can record observations or finalize a report.
#[derive(Debug)]
pub struct Report {
    options: Options,
    expected: Option<Population>,
    completion: Completion,
    backend: String,
    attempted: BTreeSet<&'static str>,
    successful: BTreeSet<ObservationKey>,
    failed_keys: BTreeSet<ObservationKey>,
    model_results: BTreeMap<AttachmentKey, bool>,
    stages: Vec<StageObservation>,
    semantic_events: Vec<SemanticObservation>,
    failures: Vec<Failure>,
    failure_count: usize,
    sync_complete: bool,
    poisoned: bool,
    counts_reliable: bool,
}
impl Report {
    fn new(options: &Options) -> Self {
        Self {
            options: *options,
            expected: population(options).ok(),
            completion: Completion::Running,
            backend: String::from("not_initialized"),
            attempted: BTreeSet::new(),
            successful: BTreeSet::new(),
            failed_keys: BTreeSet::new(),
            model_results: BTreeMap::new(),
            stages: Vec::new(),
            semantic_events: Vec::new(),
            failures: Vec::new(),
            failure_count: 0,
            sync_complete: false,
            poisoned: false,
            counts_reliable: true,
        }
    }
    fn failure(&mut self, failure: Failure) {
        self.failure_count += 1;
        if self.failures.len() < MAX_FAILURES {
            self.failures.push(failure);
        }
    }
    fn problem(&mut self, message: String) {
        self.failure(Failure {
            key: None,
            attachment: None,
            message,
            mismatches: Vec::new(),
            actual_operands: Vec::new(),
        });
    }
    fn record(&mut self, key: ObservationKey, result: Result<(), Box<Failure>>) {
        if self.successful.contains(&key) || self.failed_keys.contains(&key) {
            self.problem(format!("duplicate observation {key:?}"));
            return;
        }
        if self.expected.as_ref().is_none_or(|p| !p.keys.contains(&key)) {
            self.problem(format!("unexpected observation {key:?}"));
            return;
        }
        match result {
            Ok(()) => {
                self.successful.insert(key);
            }
            Err(mut failure) => {
                failure.key = Some(key);
                self.failed_keys.insert(key);
                self.failure(*failure);
            }
        }
    }
    fn finish(&mut self) {
        if self.completion != Completion::Running {
            self.problem(String::from("report finalized more than once"));
            self.completion = Completion::Rejected;
            return;
        }
        self.completion = if self.poisoned { Completion::Aborted } else { Completion::Finished };
    }
    pub fn completion(&self) -> Completion {
        self.completion
    }
    pub fn failures(&self) -> &[Failure] {
        &self.failures
    }
    pub fn stages(&self) -> &[StageObservation] {
        &self.stages
    }
    pub fn semantic_events(&self) -> &[SemanticObservation] {
        &self.semantic_events
    }
    fn attempted(&self) -> usize {
        self.attempted.len()
    }
    fn passed(&self) -> usize {
        self.attempted
            .iter()
            .filter(|&&id| {
                self.expected
                    .as_ref()
                    .is_some_and(|p| p.keys.iter().filter(|k| k.case_id == id).all(|k| self.successful.contains(k)))
            })
            .count()
    }
    fn failed(&self) -> usize {
        self.attempted() - self.passed()
    }
    pub fn device_counts(&self) -> Option<DeviceCounts> {
        self.counts_reliable.then(|| DeviceCounts {
            attempted: self.attempted(),
            passed: self.passed(),
            failed: self.failed(),
        })
    }
    pub fn model_attempted(&self) -> usize {
        self.model_results.len()
    }
    pub fn model_passed(&self) -> usize {
        self.model_results.values().filter(|&&v| v).count()
    }
    pub fn is_ok(&self) -> bool {
        success_footer(self).is_ok()
    }
}
/// Initialization and executor failures can only create a rejected report.
pub fn rejected_report(options: &Options, reason: &str) -> Report {
    let mut r = Report::new(options);
    r.problem(reason.into());
    r.completion = Completion::Rejected;
    r
}
/// A lost executor cannot truthfully report zero actual attempts.
pub fn interrupted_report(options: &Options, reason: &str) -> Report {
    let mut report = rejected_report(options, reason);
    report.counts_reliable = false;
    report
}
fn plain_failure(message: impl Into<String>) -> Box<Failure> {
    Box::new(Failure {
        key: None,
        attachment: None,
        message: message.into(),
        mismatches: Vec::new(),
        actual_operands: Vec::new(),
    })
}
pub fn success_footer(report: &Report) -> Result<String, String> {
    if report.completion != Completion::Finished || report.failure_count != 0 || report.poisoned || !report.counts_reliable {
        return Err(String::from("unfinished, failed, rejected or poisoned report"));
    }
    let current = population(&report.options).map_err(|e| format!("selection: {e:?}"))?;
    let sealed = report.expected.as_ref().ok_or("missing expected population")?;
    if current.keys != sealed.keys
        || current.cases.iter().map(|c| (c.id, c.row_sha256)).collect::<Vec<_>>() != sealed.cases.iter().map(|c| (c.id, c.row_sha256)).collect::<Vec<_>>()
        || current.models.iter().map(|a| a.key).collect::<Vec<_>>() != sealed.models.iter().map(|a| a.key).collect::<Vec<_>>()
    {
        return Err(String::from("population identity changed"));
    }
    if report.options.scope == ExecutionScope::ModelReference {
        let expected: BTreeSet<_> = current.models.iter().map(|a| a.key).collect();
        if !report.attempted.is_empty()
            || !report.successful.is_empty()
            || report.model_results.keys().copied().collect::<BTreeSet<_>>() != expected
            || !report.model_results.values().all(|&p| p)
        {
            return Err(String::from("MODEL attachment identity/count mismatch"));
        }
        return Ok(format!(
            "BURN_INDEXED_ACCUMULATION_MODEL_OK evidence=MODEL templates={} attachments={} device_attempted=0 device_passed=0",
            expected.iter().map(|k| k.template).collect::<BTreeSet<_>>().len(),
            expected.len()
        ));
    }
    let expected_ids: BTreeSet<_> = current.cases.iter().map(|c| c.id).collect();
    if !report.sync_complete
        || report.attempted != expected_ids
        || report.successful != current.keys
        || !report.failed_keys.is_empty()
        || !report.model_results.is_empty()
        || report.passed() != current.cases.len()
        || report.attempted() != report.passed() + report.failed()
    {
        return Err(String::from("device rows, source keys, observation keys or synchronization incomplete"));
    }
    Ok(format!(
        "BURN_INDEXED_ACCUMULATION_OK scope={:?} attempted={} passed={} source_roles={} source_storages={} observations={} logical_only=true full_parity=false",
        report.options.scope,
        report.attempted(),
        report.passed(),
        current.keys.iter().filter(|k| k.check == CheckKind::InputView).count(),
        current.keys.iter().filter(|k| k.check == CheckKind::InputStorage).count(),
        report.successful.len()
    ))
}
fn snapshot_json(s: &Snapshot) -> String {
    format!(
        "{{\"dtype\":{},\"shape\":{:?},\"bits\":{}}}",
        debug_json(s.dtype),
        s.shape,
        list_json(s.bits.iter().map(|b| quoted(&format!("0x{b:0width$x}", width = s.dtype.size() * 2))))
    )
}
fn operand_json(o: &OperandObservation) -> String {
    format!(
        "{{\"key\":{},\"actual\":{},\"unavailable\":{}}}",
        key_json(&o.key),
        o.actual.as_ref().map_or_else(|| String::from("null"), snapshot_json),
        o.unavailable.as_ref().map_or_else(|| String::from("null"), |s| quoted(s))
    )
}
pub fn observed_json(r: &Report) -> String {
    let missing = list_json(r.expected.iter().flat_map(|p| p.keys.difference(&r.successful)).map(key_json));
    let failures = list_json(r.failures.iter().map(|f| {
        format!(
            "{{\"key\":{},\"attachment\":{},\"message\":{},\"mismatches\":{},\"actual_operands\":{}}}",
            f.key.as_ref().map_or_else(|| String::from("null"), key_json),
            f.attachment.map_or_else(|| String::from("null"), attachment_key_json),
            quoted(&f.message),
            list_json(
                f.mismatches
                    .iter()
                    .map(|(i, a, e)| format!("{{\"element\":{i},\"actual_bits\":\"0x{a:x}\",\"expected_bits\":\"0x{e:x}\"}}"))
            ),
            list_json(f.actual_operands.iter().map(operand_json))
        )
    }));
    let stages = list_json(r.stages.iter().map(|s| {
        format!(
            "{{\"key\":{},\"expected\":{},\"actual\":{},\"read_error\":{},\"actual_operands\":{}}}",
            key_json(&s.key),
            snapshot_json(&s.expected),
            s.actual.as_ref().map_or_else(|| String::from("null"), snapshot_json),
            s.read_error.as_ref().map_or_else(|| String::from("null"), |s| quoted(s)),
            list_json(s.actual_operands.iter().map(operand_json))
        )
    }));
    let model_keys = list_json(
        r.model_results
            .iter()
            .map(|(k, passed)| format!("{{\"key\":{},\"passed\":{passed},\"evidence\":\"MODEL\"}}", attachment_key_json(*k))),
    );
    let seeds = list_json(r.expected.iter().flat_map(|p| p.cases.iter()).filter_map(|c| {
        if let Expected::Gradient(g) = c.expected {
            Some(format!(
                "{{\"id\":{},\"status\":\"expected_only_source_backed\",\"expected\":{},\"actual\":null}}",
                quoted(c.id),
                snapshot_json(&g.seed.into())
            ))
        } else {
            None
        }
    }));
    let mut json = format!(
        "{{\"corpus_version\":1,\"scope\":{},\"dtype_filter\":{},\"backend\":{},\"completion\":{},\"finalized\":{},\"ok\":{},\"FROZEN\":false,\"source_binding_approved\":false,\"numerical_freeze_sha256\":{},\"fixtures_sha256\":{},\"device_attempted\":{},\"device_passed\":{},\"device_failed\":{},\"model_attempted\":{},\"model_passed\":{},\"successful_observations\":{},\"failed_observations\":{},\"missing_observations\":{},\"failure_count\":{},\"omitted_failure_details\":{},\"failures\":{},\"stage_observations\":{},\"seeds\":{},\"model_results\":{},\"synchronized\":{},\"device_poisoned\":{},\"source_observation\":\"retained_clone_lineage_logical_reads_after_whole_graph\",\"physical_tails_ownership_residency_proven\":false,\"full_parity\":false,\"structured_policies\":{}}}",
        debug_json(r.options.scope),
        r.options.dtype.map_or_else(|| String::from("null"), debug_json),
        quoted(&r.backend),
        debug_json(r.completion),
        r.completion != Completion::Running,
        r.is_ok(),
        quoted(NUMERICAL_FREEZE_SHA256),
        quoted(fixtures_sha256()),
        r.device_counts().map_or_else(|| String::from("null"), |c| c.attempted.to_string()),
        r.device_counts().map_or_else(|| String::from("null"), |c| c.passed.to_string()),
        r.device_counts().map_or_else(|| String::from("null"), |c| c.failed.to_string()),
        r.model_attempted(),
        r.model_passed(),
        list_json(r.successful.iter().map(key_json)),
        list_json(r.failed_keys.iter().map(key_json)),
        missing,
        r.failure_count,
        r.failure_count - r.failures.len(),
        failures,
        stages,
        seeds,
        model_keys,
        r.sync_complete,
        r.poisoned,
        structured_policy_json()
    );
    json.pop();
    let events = list_json(r.semantic_events.iter().map(|e| {
        format!(
            "{{\"id\":{},\"expected\":{},\"actual_error\":{},\"output_metadata_valid\":{},\"accepted\":{}}}",
            quoted(e.case_id),
            debug_json(e.expected),
            e.actual_error.as_ref().map_or_else(|| String::from("null"), |s| quoted(s)),
            e.output_metadata_valid,
            e.accepted
        )
    }));
    let selected = list_json(r.expected.iter().flat_map(|p| p.cases.iter()).map(|c| {
        format!(
            "{{\"id\":{},\"row_sha256\":{},\"policy\":{}}}",
            quoted(c.id),
            quoted(c.row_sha256),
            debug_json(c.policy)
        )
    }));
    let _ = write!(
        json,
        ",\"semantic_events\":{},\"selected_rows\":{},\"attempted_ids\":{}}}",
        events,
        selected,
        if r.counts_reliable {
            list_json(r.attempted.iter().map(|s| quoted(s)))
        } else {
            String::from("null")
        }
    );
    json
}
pub fn run_models(options: &Options) -> Report {
    if options.scope != ExecutionScope::ModelReference {
        return rejected_report(options, "run_models requires ModelReference");
    }
    let mut r = Report::new(options);
    r.backend = String::from("host_model_no_device");
    let Some(p) = r.expected.clone() else {
        return rejected_report(options, "empty or invalid MODEL population");
    };
    for a in p.models {
        let result = check_model(a);
        r.model_results.insert(a.key, result.is_ok());
        if let Err(message) = result {
            r.failure(Failure {
                attachment: Some(a.key),
                ..*plain_failure(message)
            });
        }
    }
    r.finish();
    r
}

#[derive(Clone)]
struct Pending {
    key: ObservationKey,
    handle: Option<DispatchTensor>,
    expected: Snapshot,
    operands: Vec<ObservationKey>,
}
fn bytes(dtype: DType, bits: &[u64]) -> Vec<u8> {
    bits.iter().flat_map(|v| v.to_le_bytes().into_iter().take(dtype.size())).collect()
}
fn create(source: &SourceSpec, device: &Device) -> DispatchTensor {
    let data = TensorData::from_bytes_vec(bytes(source.dtype, source.storage_bits), source.storage_shape.to_vec(), source.dtype);
    if source.dtype.is_float() {
        Tensor::<1>::from_data(data, (device, source.dtype)).into_dispatch()
    } else {
        Tensor::<1, Int>::from_data(data, (device, source.dtype)).into_dispatch()
    }
}
fn view(mut tensor: DispatchTensor, steps: &[ViewStep]) -> DispatchTensor {
    for step in steps {
        let float = tensor.dtype().is_float();
        tensor = match step {
            ViewStep::Reshape(shape) => {
                if tensor.shape().as_slice() == *shape {
                    tensor
                } else if float {
                    Dispatch::float_reshape(tensor, Shape::from(shape.to_vec()))
                } else {
                    Dispatch::int_reshape(tensor, Shape::from(shape.to_vec()))
                }
            }
            ViewStep::Slice { start, end, step } => {
                let slice = [Slice::new(*start as isize, Some(*end as isize), *step as isize)];
                if float {
                    Dispatch::float_slice(tensor, &slice)
                } else {
                    Dispatch::int_slice(tensor, &slice)
                }
            }
            ViewStep::Permute(axes) => {
                if float {
                    Dispatch::float_permute(tensor, axes)
                } else {
                    Dispatch::int_permute(tensor, axes)
                }
            }
            ViewStep::Flip(axes) => {
                if axes.is_empty() {
                    tensor
                } else if float {
                    Dispatch::float_flip(tensor, axes)
                } else {
                    Dispatch::int_flip(tensor, axes)
                }
            }
            ViewStep::Expand(shape) => {
                if tensor.shape().as_slice() == *shape {
                    tensor
                } else if float {
                    Dispatch::float_expand(tensor, Shape::from(shape.to_vec()))
                } else {
                    Dispatch::int_expand(tensor, Shape::from(shape.to_vec()))
                }
            }
        };
    }
    tensor
}
struct Inputs {
    views: BTreeMap<&'static str, DispatchTensor>,
    pending: Vec<Pending>,
}
fn inputs(case: &Case, device: &Device, leaf: Option<DispatchTensor>) -> Inputs {
    let mut roots = BTreeMap::new();
    let mut pending = Vec::new();
    for source in case.sources {
        if roots.contains_key(source.storage_id) {
            continue;
        }
        let (tensor, shape) = if source.storage_id == "original_tracked_leaf" {
            (leaf.as_ref().expect("tracked leaf supplied").clone(), case.source("original_leaf").shape)
        } else {
            (create(source, device), source.storage_shape)
        };
        let expected = Snapshot {
            dtype: source.dtype,
            shape: shape.to_vec(),
            bits: source.storage_bits.to_vec(),
        };
        for kind in [CheckKind::InputStorage, CheckKind::RetainedStorage] {
            pending.push(Pending {
                key: key(case, kind, source.storage_id),
                handle: Some(tensor.clone()),
                expected: expected.clone(),
                operands: Vec::new(),
            });
        }
        roots.insert(source.storage_id, tensor);
    }
    let mut views = BTreeMap::new();
    for source in case.sources {
        let root = roots[source.storage_id].clone();
        let tensor = if source.role == "original_leaf" {
            root
        } else {
            view(root, source.construction)
        };
        let expected = Snapshot {
            dtype: source.dtype,
            shape: source.shape.to_vec(),
            bits: source.logical_bits(),
        };
        for kind in [CheckKind::InputView, CheckKind::RetainedView] {
            pending.push(Pending {
                key: key(case, kind, source.role),
                handle: Some(tensor.clone()),
                expected: expected.clone(),
                operands: Vec::new(),
            });
        }
        views.insert(source.role, tensor);
    }
    Inputs { views, pending }
}
fn update(case: &Case, views: &BTreeMap<&str, DispatchTensor>) -> DispatchTensor {
    let destination = views["destination"].clone();
    let indices = views["indices"].clone();
    let values = views["values"].clone();
    let op = case.operation.update();
    match (case.dtype.is_float(), case.route) {
        (true, Route::Scatter) => Dispatch::float_scatter(case.axis.expect("axis"), destination, indices, values, op),
        (false, Route::Scatter) => Dispatch::int_scatter(case.axis.expect("axis"), destination, indices, values, op),
        (true, Route::SelectUpdate) => Dispatch::float_select_assign(destination, case.axis.expect("axis"), indices, values, op),
        (false, Route::SelectUpdate) => Dispatch::int_select_assign(destination, case.axis.expect("axis"), indices, values, op),
        (true, Route::ScatterNd) => Dispatch::float_scatter_nd(destination, indices, values, op),
        (false, Route::ScatterNd) => Dispatch::int_scatter_nd(destination, indices, values, op),
        _ => unreachable!("generated value route"),
    }
}
async fn read_data(tensor: DispatchTensor) -> Result<TensorData, ExecutionError> {
    if tensor.dtype().is_float() {
        Dispatch::float_into_data(tensor).await
    } else {
        Dispatch::int_into_data(tensor).await
    }
}
fn snapshot(data: TensorData) -> Result<Snapshot, String> {
    let dtype = data.dtype();
    let width = dtype.size();
    if width == 0 || width > 8 || !data.as_bytes().len().is_multiple_of(width) {
        return Err(String::from("invalid readback byte width"));
    }
    let bits = data
        .as_bytes()
        .chunks_exact(width)
        .map(|chunk| {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(word)
        })
        .collect::<Vec<_>>();
    if data.shape().as_slice().iter().product::<usize>() != bits.len() {
        return Err(String::from("readback shape/byte count mismatch"));
    }
    Ok(Snapshot {
        dtype,
        shape: data.shape().to_vec(),
        bits,
    })
}
fn compare(actual: &Snapshot, expected: &Snapshot) -> Result<(), Box<Failure>> {
    let mismatches = actual
        .bits
        .iter()
        .zip(&expected.bits)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .take(MAX_MISMATCHES)
        .map(|(i, (&a, &e))| (i, a, e))
        .collect::<Vec<_>>();
    if actual == expected {
        Ok(())
    } else {
        Err(Box::new(Failure {
            mismatches,
            ..*plain_failure(format!(
                "dtype/shape/bits actual={:?}/{:?}/{} expected={:?}/{:?}/{}",
                actual.dtype,
                actual.shape,
                actual.bits.len(),
                expected.dtype,
                expected.shape,
                expected.bits.len()
            ))
        }))
    }
}
async fn observe(report: &mut Report, pending: Vec<Pending>, recovery_device: Option<&Device>) {
    // All handles and the complete queued graph exist before the first read.
    let mut actuals = BTreeMap::new();
    for check in &pending {
        let result = if let Some(handle) = &check.handle {
            match read_data(handle.clone()).await {
                Ok(data) => snapshot(data),
                Err(error) => {
                    report.poisoned |= error.is_device_poisoned();
                    Err(format!("readback: {error}"))
                }
            }
        } else {
            Err(String::from("missing original-leaf gradient"))
        };
        actuals.insert(check.key, result);
    }
    let recovery_sync = recovery_device.and_then(|device| device.sync().err()).map(|error| {
        report.poisoned |= error.is_device_poisoned();
        format!("same-context recovery synchronization: {error}")
    });
    for check in pending {
        let operands = check
            .operands
            .iter()
            .map(|k| match actuals.get(k) {
                Some(Ok(value)) => OperandObservation {
                    key: *k,
                    actual: Some(value.clone()),
                    unavailable: None,
                },
                Some(Err(reason)) => OperandObservation {
                    key: *k,
                    actual: None,
                    unavailable: Some(reason.clone()),
                },
                None => OperandObservation {
                    key: *k,
                    actual: None,
                    unavailable: Some(String::from("stage operand snapshot unavailable")),
                },
            })
            .collect::<Vec<_>>();
        let result = &actuals[&check.key];
        if matches!(check.key.check, CheckKind::PublicStage | CheckKind::Output | CheckKind::Recovery) {
            report.stages.push(StageObservation {
                key: check.key,
                expected: check.expected.clone(),
                actual: result.as_ref().ok().cloned(),
                read_error: result.as_ref().err().cloned(),
                actual_operands: operands.clone(),
            });
        }
        let outcome = match result {
            Ok(actual) => compare(actual, &check.expected),
            Err(reason) => Err(plain_failure(reason.clone())),
        }
        .map_err(|mut f| {
            f.actual_operands = operands;
            f
        });
        let outcome = if check.key.check == CheckKind::Recovery {
            recovery_sync.as_ref().map_or(outcome, |reason| Err(plain_failure(reason.clone())))
        } else {
            outcome
        };
        report.record(check.key, outcome);
    }
}
fn public_graph<const D: usize, const O: usize>(case: &Case, leaf: Tensor<D>, output: Tensor<O>, coefficient: Tensor<O>, pending: &mut Vec<Pending>) {
    let Expected::Gradient(expected) = case.expected else { unreachable!() };
    let graph = case.public_graph.expect("public graph");
    let stage = |subject| key(case, CheckKind::PublicStage, subject);
    pending.push(Pending {
        key: stage("forward"),
        handle: Some(output.clone().into_dispatch()),
        expected: expected.forward.into(),
        operands: vec![key(case, CheckKind::InputView, "operand"), key(case, CheckKind::InputView, "indices")],
    });
    let mut sums = Vec::new();
    for branch in 0..graph.branches {
        let product = output.clone() * coefficient.clone();
        let sum = product.clone().sum();
        let (product_name, sum_name) = branch_names(branch);
        pending.push(Pending {
            key: stage(product_name),
            handle: Some(product.into_dispatch()),
            expected: graph.product.into(),
            operands: vec![stage("forward"), key(case, CheckKind::InputView, "upstream_coefficient")],
        });
        pending.push(Pending {
            key: stage(sum_name),
            handle: Some(sum.clone().into_dispatch()),
            expected: graph.branch_sum.into(),
            operands: vec![stage(product_name)],
        });
        sums.push(sum);
    }
    let root = if graph.branches == 2 {
        sums[0].clone() + sums[1].clone()
    } else {
        sums[0].clone()
    };
    pending.push(Pending {
        key: stage("root"),
        handle: Some(root.clone().into_dispatch()),
        expected: expected.root.into(),
        operands: (0..graph.branches).map(|b| stage(branch_names(b).1)).collect(),
    });
    let gradients = root.backward();
    pending.push(Pending {
        key: stage("original_leaf_gradient"),
        handle: leaf.grad(&gradients).map(Tensor::into_dispatch),
        expected: expected.leaf_gradient.into(),
        operands: vec![
            key(case, CheckKind::InputView, "operand"),
            key(case, CheckKind::InputView, "indices"),
            key(case, CheckKind::InputView, "upstream_coefficient"),
            stage("root"),
        ],
    });
}
fn public_case<const D: usize>(case: &Case, device: &Device) -> Vec<Pending> {
    let device = device.clone().autodiff();
    let source = case.source("original_leaf");
    let leaf = Tensor::<D>::from_data(
        TensorData::from_bytes_vec(bytes(source.dtype, source.storage_bits), source.shape.to_vec(), source.dtype),
        (&device, source.dtype),
    )
    .autodiff()
    .require_grad();
    let mut inputs = inputs(case, &device, Some(leaf.clone().into_dispatch()));
    let operand = Tensor::<D>::from_dispatch(inputs.views["operand"].clone());
    match case.route {
        Route::GatherBackward => {
            let output = operand.gather(
                case.axis.expect("gather axis"),
                Tensor::<D, Int>::from_dispatch(inputs.views["indices"].clone()),
            );
            public_graph(
                case,
                leaf,
                output,
                Tensor::<D>::from_dispatch(inputs.views["upstream_coefficient"].clone()),
                &mut inputs.pending,
            );
        }
        Route::SelectBackward => {
            let output = operand.select(
                case.axis.expect("select axis"),
                Tensor::<1, Int>::from_dispatch(inputs.views["indices"].clone()),
            );
            public_graph(
                case,
                leaf,
                output,
                Tensor::<D>::from_dispatch(inputs.views["upstream_coefficient"].clone()),
                &mut inputs.pending,
            );
        }
        Route::GatherNdBackward => {
            let output: Tensor<2> = operand.gather_nd(Tensor::<2, Int>::from_dispatch(inputs.views["indices"].clone()));
            public_graph(
                case,
                leaf,
                output,
                Tensor::<2>::from_dispatch(inputs.views["upstream_coefficient"].clone()),
                &mut inputs.pending,
            );
        }
        Route::EmbeddingBackward => {
            let output = module::embedding(
                Tensor::<2>::from_dispatch(operand.into_dispatch()),
                Tensor::<2, Int>::from_dispatch(inputs.views["indices"].clone()),
            );
            public_graph(
                case,
                leaf,
                output,
                Tensor::<3>::from_dispatch(inputs.views["upstream_coefficient"].clone()),
                &mut inputs.pending,
            );
        }
        _ => unreachable!("generated public route"),
    }
    inputs.pending
}
fn pending_public(case: &Case, device: &Device) -> Vec<Pending> {
    match case.design_rank {
        1 => public_case::<1>(case, device),
        2 => public_case::<2>(case, device),
        4 => public_case::<4>(case, device),
        6 => public_case::<6>(case, device),
        7 => public_case::<7>(case, device),
        8 => public_case::<8>(case, device),
        _ => unreachable!("generated public rank"),
    }
}
fn reference_index_error(case: &Case) -> Option<SemanticError> {
    let values = case
        .source("indices")
        .logical_bits()
        .into_iter()
        .map(|b| signed_value(case.index_dtype, b))
        .collect::<Vec<_>>();
    if values.iter().any(|v| !(0..=i128::from(i32::MAX)).contains(v)) {
        return Some(SemanticError::IndexRange);
    }
    let axis = case.axis?;
    let bound = case.source("destination").shape[axis] as i128;
    values.iter().any(|&v| v >= bound).then_some(SemanticError::CoordinateBounds)
}
fn recoverable_error(error: &ExecutionError, case: &Case) -> bool {
    let operation = if case.dtype.is_float() { "float_scatter" } else { "int_scatter" };
    matches!(error,ExecutionError::WithContext{reason} if reason==&format!("D3OS native Burn {operation}: invalid operation parameter"))
}
async fn error_case(report: &mut Report, case: &Case, device: &Device) {
    let mut inputs = inputs(case, device, None);
    let output = update(case, &inputs.views);
    let metadata = output.dtype() == case.dtype && output.shape().as_slice() == case.source("destination").shape;
    let result = read_data(output).await;
    let Expected::Error(expected) = case.expected else { unreachable!() };
    let actual_error = result.as_ref().err().map(|e| format!("{e}"));
    let accepted = metadata && reference_index_error(case) == Some(expected) && result.as_ref().is_err_and(|e| recoverable_error(e, case));
    report.semantic_events.push(SemanticObservation {
        case_id: case.id,
        expected,
        actual_error: actual_error.clone(),
        output_metadata_valid: metadata,
        accepted,
    });
    report.poisoned |= result.as_ref().is_err_and(ExecutionError::is_device_poisoned);
    for kind in [CheckKind::SemanticError, CheckKind::NoPublishedOutput] {
        report.record(
            key(case, kind, "result"),
            if accepted {
                Ok(())
            } else {
                Err(plain_failure(format!(
                    "expected={expected:?} metadata_valid={metadata} actual_error={actual_error:?} published_output={}",
                    result.is_ok()
                )))
            },
        );
    }
    // The failed tensor's read reports its deferred error before a fresh valid
    // operation on the same device. Poison and stale errors cannot pass recovery.
    let one = if case.dtype.is_float() { ieee::integer_bits(case.dtype, 1) } else { 1 };
    let two = reference_step(case.dtype, one, one, Operation::Add);
    let recovery = if case.dtype.is_float() {
        Tensor::<1>::from_data(
            TensorData::from_bytes_vec(bytes(case.dtype, &[one, one]), vec![2], case.dtype),
            (device, case.dtype),
        )
        .scatter(
            0,
            Tensor::<1, Int>::from_data(
                TensorData::from_bytes_vec(bytes(case.index_dtype, &[0]), vec![1], case.index_dtype),
                (device, case.index_dtype),
            ),
            Tensor::<1>::from_data(TensorData::from_bytes_vec(bytes(case.dtype, &[one]), vec![1], case.dtype), (device, case.dtype)),
            IndexingUpdateOp::Add,
        )
        .into_dispatch()
    } else {
        Tensor::<1, Int>::from_data(
            TensorData::from_bytes_vec(bytes(case.dtype, &[one, one]), vec![2], case.dtype),
            (device, case.dtype),
        )
        .scatter(
            0,
            Tensor::<1, Int>::from_data(
                TensorData::from_bytes_vec(bytes(case.index_dtype, &[0]), vec![1], case.index_dtype),
                (device, case.index_dtype),
            ),
            Tensor::<1, Int>::from_data(TensorData::from_bytes_vec(bytes(case.dtype, &[one]), vec![1], case.dtype), (device, case.dtype)),
            IndexingUpdateOp::Add,
        )
        .into_dispatch()
    };
    inputs.pending.push(Pending {
        key: key(case, CheckKind::Recovery, "same_context_valid_operation"),
        handle: Some(recovery),
        expected: Snapshot {
            dtype: case.dtype,
            shape: vec![2],
            bits: vec![two, one],
        },
        operands: Vec::new(),
    });
    observe(report, inputs.pending, Some(device)).await;
}
fn is_native_backend(backend: &str) -> bool {
    backend == "dispatch<d3os-native-nvidia>"
}

/// Run a selected population through registered Dispatch and standard AD.
/// Source and stage readbacks follow the entire queued graph, including backward.
pub async fn run(device: &Device, options: &Options) -> Report {
    if options.scope == ExecutionScope::ModelReference {
        return rejected_report(options, "MODEL execution uses run_models without a device");
    }
    let plain = device.clone().without_autodiff();
    let backend = Dispatch::name(plain.as_dispatch());
    let native = is_native_backend(&backend);
    if (matches!(options.scope, ExecutionScope::NativeValues | ExecutionScope::NativeErrors) && !native)
        || (options.scope == ExecutionScope::LinuxCommon && native)
    {
        return rejected_report(options, "execution scope does not match the registered backend");
    }
    let mut report = Report::new(options);
    report.backend = backend;
    let Some(p) = report.expected.clone() else {
        return rejected_report(options, "empty or unsupported execution population");
    };
    for case in p.cases {
        report.attempted.insert(case.id);
        match case.expected {
            Expected::Error(_) => error_case(&mut report, case, &plain).await,
            Expected::Gradient(_) => {
                let pending = pending_public(case, &plain);
                observe(&mut report, pending, None).await;
            }
            Expected::Value(expected) => {
                let mut inputs = inputs(case, &plain, None);
                let output = update(case, &inputs.views);
                inputs.pending.push(Pending {
                    key: key(case, CheckKind::Output, "result"),
                    handle: Some(output),
                    expected: expected.into(),
                    operands: vec![
                        key(case, CheckKind::InputView, "destination"),
                        key(case, CheckKind::InputView, "values"),
                        key(case, CheckKind::InputView, "indices"),
                    ],
                });
                observe(&mut report, inputs.pending, None).await;
            }
        }
        if report.poisoned {
            break;
        }
    }
    match plain.sync() {
        Ok(()) => report.sync_complete = true,
        Err(error) => {
            report.poisoned |= error.is_device_poisoned();
            report.problem(format!("final synchronization: {error}"));
        }
    }
    report.finish();
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "native")]
    #[test]
    fn native_scope_guard_recognizes_registered_dispatch_metadata() {
        let native = crate::native::NativeDevice;
        let device = Device::new(native);
        let dispatched = Dispatch::name(device.as_dispatch());
        assert!(is_native_backend(&dispatched));
        assert!(!is_native_backend(&crate::native::Native::name(&native)));

        for foreign in [
            String::from("dispatch<foreign>"),
            format!("prefix{dispatched}"),
            format!("{dispatched}suffix"),
            format!("dispatch<{dispatched}>"),
        ] {
            assert!(!is_native_backend(&foreign));
        }
    }

    #[test]
    fn completion_requires_observations_and_cannot_be_repeated() {
        for scope in [
            ExecutionScope::LinuxCommon,
            ExecutionScope::NativeValues,
            ExecutionScope::NativeErrors,
            ExecutionScope::ModelReference,
        ] {
            let mut report = Report::new(&Options { scope, dtype: None });
            assert!(!report.is_ok());
            report.finish();
            assert!(!report.is_ok());
            report.finish();
            assert_eq!(report.completion(), Completion::Rejected);
        }
    }

    #[test]
    fn duplicate_and_unexpected_keys_remain_failures() {
        let mut report = Report::new(&Options::default());
        let first = *report.expected.as_ref().unwrap().keys.first().unwrap();
        report.record(first, Err(plain_failure(String::new())));
        report.record(first, Err(plain_failure(String::new())));
        let outside = population(&Options {
            scope: ExecutionScope::NativeErrors,
            dtype: None,
        })
        .unwrap();
        report.record(*outside.keys.first().unwrap(), Err(plain_failure(String::new())));
        report.finish();
        assert!(!report.is_ok());
        assert_eq!(report.failure_count, 3);
        assert_eq!(report.failed_keys.len(), 1);
        assert!(report.successful.is_empty());
    }

    #[test]
    fn poisoned_or_unrelated_errors_never_satisfy_semantic_checks() {
        for case in corpus().iter().filter(|c| matches!(c.expected, Expected::Error(_))) {
            assert!(!recoverable_error(&ExecutionError::device_poisoned(String::new()), case));
            assert!(!recoverable_error(&ExecutionError::with_context(String::new()), case));
            let Expected::Error(expected) = case.expected else { unreachable!() };
            assert_eq!(reference_index_error(case), Some(expected));
        }
    }

    #[test]
    fn lost_executor_keeps_counts_unknown_and_cannot_pass() {
        let report = interrupted_report(&Options::default(), "");
        assert!(report.device_counts().is_none());
        assert!(!report.is_ok());
        assert_eq!(report.completion(), Completion::Rejected);
    }
}
