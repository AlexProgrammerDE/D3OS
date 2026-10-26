//! Frozen mixed matrix fixtures and logical public-Dispatch conformance.
//! Hardware exception/subnormal gates remain mandatory. Physical allocation,
//! guard, queue and service-counter evidence belongs to the raw harness.
use alloc::{
    collections::BTreeSet,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
pub use burn::backend::DType;
use burn::{
    backend::{
        Dispatch, TensorData, TensorMetadata,
        ops::{FloatTensorOps, ModuleOps},
        tensor::FloatTensor,
    },
    tensor::{Device, Slice, Tensor},
};

pub type Shape = &'static [usize];
pub const CORPUS_VERSION: u32 = 1;
pub const SELECTED_ROWS: usize = 121;
pub const RESERVED_ROWS: usize = 3;
pub const NUMERIC_CELLS: usize = 65;
pub const LAST_DROP_CONSUMERS: usize = 141;
pub const MODEL_LABEL: &str = "project_empirical_conditional_normal_tree_not_vendor_guarantee";
pub const FRONTEND_ZERO_LINEAR_ID: &str = "public_linear_rank1_zero_frontend_unsqueeze";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Geometry,
    Numeric,
    Gradient,
    NativeExtension,
    Preservation,
    F64Diagnostic,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Matmul,
    Linear,
    ExternalGate,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tracking {
    None,
    Both,
    Lhs,
    Rhs,
    Shared,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Shared,
    NativeExtensions,
    Preservation,
    F64Diagnostics,
    All,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    pub dtype: Option<DType>,
    pub case_id: Option<String>,
    pub include_empty: bool,
    pub materialize_chain_steps: bool,
    pub max_failures: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            dtype: None,
            case_id: None,
            include_empty: true,
            materialize_chain_steps: false,
            max_failures: 32,
        }
    }
}
/// Data descriptor into the independently generated little-endian fixture blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Values {
    pub dtype: DType,
    pub shape: Shape,
    pub offset: usize,
    pub len: usize,
}
impl Values {
    pub fn bits(self) -> Vec<u64> {
        unpack(self.dtype, self.bytes()).expect("generated fixture alignment")
    }
    pub fn bytes(self) -> &'static [u8] {
        &FIXTURE_BYTES[self.offset..self.offset + self.len * self.dtype.size()]
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    pub strides: &'static [isize],
    pub base: usize,
    pub root: Values,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Matrix {
    pub name: &'static str,
    pub values: Values,
    pub view: Option<View>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Exact,
    ZeroClass,
    NaNClass,
    Infinity,
    ProjectInterval { low: u64, high: u64 },
    Observation,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expected {
    pub name: &'static str,
    pub values: Values,
    pub policies: &'static [Policy],
    pub stage: &'static str,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumericCell {
    pub name: &'static str,
    pub ideal_bits: u64,
    pub serial_bits: u64,
    pub ideal_rational: &'static str,
    pub sum_abs_products: &'static str,
    pub scratch_low: &'static str,
    pub scratch_high: &'static str,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Consumer {
    pub input: &'static str,
    pub expected: Expected,
    pub axis: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Case {
    pub id: &'static str,
    pub group: Group,
    pub dtype: Option<DType>,
    pub operation: Operation,
    pub tracking: Tracking,
    pub matrices: &'static [Matrix],
    pub expected: &'static [Expected],
    pub numeric: &'static [NumericCell],
    pub consumers: &'static [Consumer],
    pub raw_matrices: &'static [Matrix],
    pub raw_expected: &'static [Expected],
}
impl Case {
    pub fn matrix(&self, name: &str) -> Option<&Matrix> {
        self.matrices.iter().find(|x| x.name == name)
    }
    pub fn expectation(&self, name: &str) -> Option<&Expected> {
        self.expected.iter().find(|x| x.name == name)
    }
}
include!("mixed_matmul_index.rs");
pub fn scratch_expectation(case: &Case) -> Option<&Expected> {
    case.raw_expected.iter().find(|e| e.name == "scratch")
}
pub fn fixtures() -> &'static [Case] {
    CASES
}
pub fn fixture(id: &str) -> Option<&'static Case> {
    CASES.iter().find(|x| x.id == id)
}
pub fn source_roots(case: &Case) -> Vec<Values> {
    case.matrices.iter().map(|m| m.view.map_or(m.values, |v| v.root)).collect()
}
pub fn pack(dtype: DType, values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes().into_iter().take(dtype.size())).collect()
}
pub fn unpack(dtype: DType, bytes: &[u8]) -> Result<Vec<u64>, String> {
    if !bytes.len().is_multiple_of(dtype.size()) {
        return Err(String::from("partial storage element"));
    }
    Ok(bytes
        .chunks_exact(dtype.size())
        .map(|b| {
            let mut x = [0; 8];
            x[..b.len()].copy_from_slice(b);
            u64::from_le_bytes(x)
        })
        .collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Finite,
    Zero,
    Subnormal,
    Infinity,
    NaN,
}
fn fields(dtype: DType) -> (u32, u32) {
    match dtype {
        DType::F16 => (5, 10),
        DType::BF16 => (8, 7),
        DType::F32 | DType::Flex32 => (8, 23),
        DType::F64 => (11, 52),
        _ => panic!("floating fixture dtype required"),
    }
}
pub fn classification(dtype: DType, bits: u64) -> Class {
    let (e, f) = fields(dtype);
    let exponent = bits >> f & ((1 << e) - 1);
    let fraction = bits & ((1 << f) - 1);
    if exponent == (1 << e) - 1 {
        if fraction == 0 { Class::Infinity } else { Class::NaN }
    } else if exponent == 0 {
        if fraction == 0 { Class::Zero } else { Class::Subnormal }
    } else {
        Class::Finite
    }
}
fn ordered(dtype: DType, bits: u64) -> u64 {
    let sign = 1u64 << (dtype.size() * 8 - 1);
    let mask = u64::MAX >> (64 - dtype.size() * 8);
    if bits & sign != 0 { !bits & mask } else { bits | sign }
}
/// Compare one frozen destination. Observation has no acceptance interval.
/// Source identity uses a separate byte comparison, including NaN payloads.
pub fn compare(dtype: DType, policy: Policy, expected: u64, actual: u64) -> bool {
    if dtype.size() < 8 && (expected >> (dtype.size() * 8) != 0 || actual >> (dtype.size() * 8) != 0) {
        return false;
    }
    match policy {
        Policy::Exact => expected == actual,
        Policy::ZeroClass => classification(dtype, actual) == Class::Zero,
        Policy::NaNClass => classification(dtype, actual) == Class::NaN,
        Policy::Infinity => classification(dtype, actual) == Class::Infinity && expected == actual,
        Policy::ProjectInterval { low, high } => {
            matches!(classification(dtype, actual), Class::Finite | Class::Subnormal | Class::Zero)
                && ordered(dtype, low) <= ordered(dtype, actual)
                && ordered(dtype, actual) <= ordered(dtype, high)
        }
        Policy::Observation => true,
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteKey {
    pub case_id: &'static str,
    pub route: &'static str,
    pub output: &'static str,
}
impl RouteKey {
    pub fn text(&self) -> String {
        format!("{}/{}/{}", self.case_id, self.route, self.output)
    }
}
#[derive(Clone, Debug)]
struct Plan {
    key: RouteKey,
    expected: Values,
    policies: Vec<Policy>,
    source: bool,
    stage: &'static str,
}
/// Immutable operand binding for one stored stage. Derived operands retain an
/// explicit producer graph when no separate frozen intermediate exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessOperand {
    pub origin: &'static str,
    pub source: &'static str,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bits: Option<Vec<u64>>,
    pub views: Vec<OperandView>,
    pub producer: Option<alloc::boxed::Box<StageWitness>>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperandView {
    Reshape(Vec<usize>),
    Transpose(usize, usize),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageWitness {
    pub operation: &'static str,
    pub operands: Vec<WitnessOperand>,
    pub axis: Option<usize>,
    /// Immediate operation result, before a metadata-only output reshape.
    pub result_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub source_view: Option<View>,
}
impl WitnessOperand {
    fn frozen(origin: &'static str, source: &'static str, values: Values) -> Self {
        Self {
            origin,
            source,
            dtype: values.dtype,
            shape: values.shape.to_vec(),
            bits: Some(values.bits()),
            views: Vec::new(),
            producer: None,
        }
    }
    fn derived(source: &'static str, dtype: DType, producer: StageWitness) -> Self {
        Self {
            origin: "derived_stage",
            source,
            dtype,
            shape: producer.output_shape.clone(),
            bits: None,
            views: Vec::new(),
            producer: Some(alloc::boxed::Box::new(producer)),
        }
    }
    fn reshape(mut self, shape: Vec<usize>) -> Self {
        assert_eq!(self.shape.iter().product::<usize>(), shape.iter().product::<usize>());
        if self.shape != shape {
            self.views.push(OperandView::Reshape(shape.clone()));
            self.shape = shape;
        }
        self
    }
    fn flatten(self) -> Self {
        let rank = self.shape.len();
        let shape = vec![self.shape[..rank - 1].iter().product(), self.shape[rank - 1]];
        self.reshape(shape)
    }
    fn transpose(mut self) -> Self {
        let rank = self.shape.len();
        let (rows, columns) = (self.shape[rank - 2], self.shape[rank - 1]);
        if let Some(bits) = self.bits.as_mut() {
            let mut transposed = Vec::with_capacity(bits.len());
            for batch in 0..self.shape[..rank - 2].iter().product() {
                for column in 0..columns {
                    for row in 0..rows {
                        transposed.push(bits[(batch * rows + row) * columns + column]);
                    }
                }
            }
            *bits = transposed;
        }
        self.views.push(OperandView::Transpose(rank - 2, rank - 1));
        self.shape.swap(rank - 2, rank - 1);
        self
    }
}
fn matrix_operand(c: &Case, name: &str) -> Option<WitnessOperand> {
    let m = c.matrix(name)?;
    Some(WitnessOperand::frozen("matrix", m.name, m.values))
}
fn expected_operand(c: &Case, name: &str) -> Option<WitnessOperand> {
    let e = c.expectation(name)?;
    Some(WitnessOperand::frozen("expected_stage", e.name, e.values))
}
fn stage(operation: &'static str, operands: Vec<WitnessOperand>, output_shape: &[usize]) -> StageWitness {
    StageWitness {
        operation,
        operands,
        axis: None,
        result_shape: output_shape.to_vec(),
        output_shape: output_shape.to_vec(),
        source_view: None,
    }
}
fn dot_stage(lhs: WitnessOperand, rhs: WitnessOperand, output_shape: &[usize]) -> StageWitness {
    // Leading output dimensions may be broadcast, but its last two dimensions
    // are the actual matrix product. A linear output view can restore rank.
    let mut result_shape = output_shape.to_vec();
    if lhs.shape.len() == 2 && rhs.shape.len() == 2 {
        result_shape = vec![lhs.shape[0], rhs.shape[1]];
    }
    let mut witness = stage("matmul", vec![lhs, rhs], output_shape);
    witness.result_shape = result_shape;
    witness
}
fn sum_stage(input: WitnessOperand, axis: usize, output_shape: &[usize]) -> StageWitness {
    let mut result_shape = input.shape.clone();
    result_shape[axis] = 1;
    let mut witness = stage("typed_sum", vec![input], output_shape);
    witness.axis = Some(axis);
    witness.result_shape = result_shape;
    witness
}
fn gradient_product(c: &Case, lhs_gradient: bool, output_shape: &[usize]) -> Option<StageWitness> {
    let g = matrix_operand(c, "upstream")?;
    Some(if lhs_gradient {
        dot_stage(g, matrix_operand(c, "rhs")?.transpose(), output_shape)
    } else {
        dot_stage(matrix_operand(c, "lhs")?.transpose(), g, output_shape)
    })
}
fn gradient_stage(c: &Case, lhs_gradient: bool, axis: Option<usize>, output_shape: &[usize]) -> Option<StageWitness> {
    let prefix = if lhs_gradient { "dA_axes." } else { "dB_axes." };
    let before = if lhs_gradient { "dA_before_sum" } else { "dB_before_sum" };
    let last_axis = axis.or_else(|| c.expected.iter().filter_map(|e| e.name.strip_prefix(prefix)?.parse::<usize>().ok()).max());
    let Some(axis) = last_axis else {
        return gradient_product(c, lhs_gradient, output_shape);
    };
    let previous = c
        .expected
        .iter()
        .filter_map(|e| Some((e.name.strip_prefix(prefix)?.parse::<usize>().ok()?, e)))
        .filter(|(prior, _)| *prior < axis)
        .max_by_key(|(prior, _)| *prior)
        .map_or(before, |(_, e)| e.name);
    Some(sum_stage(expected_operand(c, previous)?, axis, output_shape))
}
/// Bind a route to its own frozen operands. This only reshapes/permutates bits
/// or names a stored producer; it never computes a numerical expectation.
fn stage_witness(c: &Case, key: &RouteKey, output_shape: &[usize]) -> Option<StageWitness> {
    let name = key.output;
    if key.route == "consumer" {
        let consumer = c.consumers.iter().find(|consumer| consumer.expected.name == name)?;
        return Some(sum_stage(expected_operand(c, consumer.input)?, consumer.axis, output_shape));
    }
    if let Some(axis) = name.strip_prefix("dA_axes.").and_then(|x| x.parse().ok()) {
        return gradient_stage(c, true, Some(axis), output_shape);
    }
    if let Some(axis) = name.strip_prefix("dB_axes.").and_then(|x| x.parse().ok()) {
        return gradient_stage(c, false, Some(axis), output_shape);
    }
    Some(match name {
        "gradient.forward" => dot_stage(matrix_operand(c, "gradient.lhs")?, matrix_operand(c, "gradient.rhs")?, output_shape),
        "gradient.dA_before_sum" => dot_stage(
            matrix_operand(c, "gradient.upstream")?,
            matrix_operand(c, "gradient.rhs")?.transpose(),
            output_shape,
        ),
        "gradient.dA" => sum_stage(expected_operand(c, "gradient.dA_before_sum")?, 0, output_shape),
        "linear.stored" => dot_stage(matrix_operand(c, "linear.input")?, matrix_operand(c, "linear.weight")?, output_shape),
        "linear.output" => stage(
            "typed_add",
            vec![expected_operand(c, "linear.stored")?, matrix_operand(c, "linear.bias")?],
            output_shape,
        ),
        "stored_forward" => dot_stage(matrix_operand(c, "input")?.flatten(), matrix_operand(c, "weight")?, output_shape),
        "dx" => dot_stage(matrix_operand(c, "upstream")?.flatten(), matrix_operand(c, "weight")?.transpose(), output_shape),
        "dW" => dot_stage(
            matrix_operand(c, "input")?.flatten().transpose(),
            matrix_operand(c, "upstream")?.flatten(),
            output_shape,
        ),
        "db" => sum_stage(matrix_operand(c, "upstream")?.flatten(), 0, output_shape),
        "dA_before_sum" => gradient_product(c, true, output_shape)?,
        "dB_before_sum" => gradient_product(c, false, output_shape)?,
        "dA" => gradient_stage(c, true, None, output_shape)?,
        "dB" => gradient_stage(c, false, None, output_shape)?,
        "shared_source_gradient" => stage("typed_add", vec![expected_operand(c, "dA")?, expected_operand(c, "dB")?], output_shape),
        "shared_transpose.forward" => dot_stage(matrix_operand(c, "lhs")?, matrix_operand(c, "lhs")?.transpose(), output_shape),
        "shared_transpose.gradient" => {
            let dtype = c.dtype?;
            let da = dot_stage(matrix_operand(c, "upstream")?, matrix_operand(c, "lhs")?, output_shape);
            let db = dot_stage(matrix_operand(c, "lhs")?.transpose(), matrix_operand(c, "upstream")?, output_shape);
            stage(
                "typed_add",
                vec![
                    WitnessOperand::derived("shared_transpose.dA", dtype, da),
                    WitnessOperand::derived("shared_transpose.dB", dtype, db).transpose(),
                ],
                output_shape,
            )
        }
        "lhs_backing_adjoint" | "rhs_backing_adjoint" => {
            let (matrix_name, gradient_name) = if name == "lhs_backing_adjoint" { ("lhs", "dA") } else { ("rhs", "dB") };
            let mut witness = stage("view_adjoint", vec![expected_operand(c, gradient_name)?], output_shape);
            witness.source_view = c.matrix(matrix_name)?.view;
            witness
        }
        "zero0.expected" => dot_stage(matrix_operand(c, "zero0.lhs")?, matrix_operand(c, "zero0.rhs")?, output_shape),
        "zero1.expected" => dot_stage(matrix_operand(c, "zero1.lhs")?, matrix_operand(c, "zero1.rhs")?, output_shape),
        "expected" | "forward" if c.operation == Operation::Linear => stage(
            "typed_add",
            vec![
                expected_operand(c, "stored_forward")?.reshape(output_shape.to_vec()),
                matrix_operand(c, "bias")?,
            ],
            output_shape,
        ),
        "expected" | "forward" => {
            let lhs = matrix_operand(c, "lhs")?;
            let mut rhs = matrix_operand(c, "rhs")?;
            if key.route == "public" && rhs.shape.len() < lhs.shape.len() {
                let mut shape = vec![1; lhs.shape.len() - rhs.shape.len()];
                shape.extend_from_slice(&rhs.shape);
                rhs = rhs.reshape(shape);
            }
            dot_stage(lhs, rhs, output_shape)
        }
        _ => return None,
    })
}

fn selected(c: &Case, o: &Options, s: Scope) -> bool {
    let scope = match s {
        Scope::Shared => matches!(c.group, Group::Geometry | Group::Numeric | Group::Gradient),
        Scope::NativeExtensions => c.group == Group::NativeExtension,
        Scope::Preservation => c.group == Group::Preservation,
        Scope::F64Diagnostics => c.group == Group::F64Diagnostic,
        Scope::All => true,
    };
    scope
        && o.dtype.is_none_or(|d| c.dtype == Some(d))
        && o.case_id.as_ref().is_none_or(|id| id == c.id)
        && (o.include_empty || !c.matrices.iter().any(|m| m.values.shape.contains(&0)))
}
fn value_plan(c: &Case, route: &'static str, e: &Expected) -> Plan {
    Plan {
        key: RouteKey {
            case_id: c.id,
            route,
            output: e.name,
        },
        expected: e.values,
        policies: e.policies.to_vec(),
        source: false,
        stage: e.stage,
    }
}
fn plans(c: &Case) -> Vec<Plan> {
    let mut out = Vec::new();
    if c.operation == Operation::ExternalGate {
        return out;
    }
    for e in c.expected {
        if e.name.starts_with("shared_transpose.") {
            continue;
        }
        let route = if e.stage == "source_view_adjoint" { "autodiff_backing" } else { "primitive" };
        out.push(value_plan(c, route, e));
    }
    let name = if c.expectation("forward").is_some() { "forward" } else { "expected" };
    if let Some(e) = c.expectation(name) {
        out.push(value_plan(c, "public", e));
    }
    if c.tracking != Tracking::None {
        for matrix in c
            .matrices
            .iter()
            .filter(|m| matches!(m.name, "lhs" | "rhs" | "input" | "weight" | "bias" | "upstream"))
        {
            out.push(Plan {
                key: RouteKey {
                    case_id: c.id,
                    route: "autodiff_source_after",
                    output: matrix.name,
                },
                expected: matrix.values,
                policies: vec![Policy::Exact; matrix.values.len],
                source: true,
                stage: "saved_autodiff_source_bytes",
            });
            if c.expectation("lhs_backing_adjoint").is_some()
                && let Some(view) = matrix.view
            {
                out.push(Plan {
                    key: RouteKey {
                        case_id: c.id,
                        route: "autodiff_backing_source_after",
                        output: matrix.name,
                    },
                    expected: view.root,
                    policies: vec![Policy::Exact; view.root.len],
                    source: true,
                    stage: "view_backing_source_bytes",
                });
            }
        }
        for name in ["forward", "dA", "dB", "dx", "dW", "db", "shared_source_gradient"] {
            let active = match name {
                "dA" => !matches!(c.tracking, Tracking::Rhs | Tracking::Shared),
                "dB" => !matches!(c.tracking, Tracking::Lhs | Tracking::Shared),
                "shared_source_gradient" => c.tracking == Tracking::Shared,
                _ => true,
            };
            if active && let Some(e) = c.expectation(name) {
                out.push(value_plan(c, "autodiff", e));
            }
        }
        if c.tracking == Tracking::Shared {
            for name in ["shared_transpose.forward", "shared_transpose.gradient"] {
                if let Some(e) = c.expectation(name) {
                    out.push(value_plan(c, "autodiff_transpose", e));
                }
            }
        }
    }
    for (name, route) in [("linear.output", "public_linear_staging"), ("gradient.dA", "autodiff_staging")] {
        if let Some(e) = c.expectation(name) {
            out.push(value_plan(c, route, e));
        }
    }
    for consumer in c.consumers {
        out.push(value_plan(c, "consumer", &consumer.expected));
    }
    for m in c.matrices {
        let root = m.view.map_or(m.values, |v| v.root);
        for route in ["source_before", "source_after", "alias_after"] {
            out.push(Plan {
                key: RouteKey {
                    case_id: c.id,
                    route,
                    output: m.name,
                },
                expected: root,
                policies: vec![Policy::Exact; root.len],
                source: true,
                stage: "untouched_source_bytes",
            });
        }
    }
    out
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub selected_rows: usize,
    pub required_routes: usize,
    pub diagnostic_routes: usize,
    pub required_values: usize,
    pub diagnostic_values: usize,
    pub model_values: usize,
    pub source_values: usize,
    pub gradient_values: usize,
    pub consumer_routes: usize,
    pub external_gates: usize,
    pub geometry_rows: usize,
    pub numeric_rows: usize,
    pub gradient_rows: usize,
    pub native_extension_rows: usize,
    pub preservation_rows: usize,
    pub f64_rows: usize,
    pub f16_values: usize,
    pub bf16_values: usize,
    pub flex32_values: usize,
    pub f64_values: usize,
}
/// Pure inventory. Expected populations never depend on measured observations.
pub fn expected_summary(options: &Options, scope: Scope) -> Summary {
    let mut s = Summary::default();
    for c in CASES.iter().filter(|c| selected(c, options, scope)) {
        s.selected_rows += 1;
        match c.group {
            Group::Geometry => s.geometry_rows += 1,
            Group::Numeric => s.numeric_rows += 1,
            Group::Gradient => s.gradient_rows += 1,
            Group::NativeExtension => s.native_extension_rows += 1,
            Group::Preservation => s.preservation_rows += 1,
            Group::F64Diagnostic => s.f64_rows += 1,
        }
        if c.operation == Operation::ExternalGate {
            s.external_gates += 1;
        }
        for p in plans(c) {
            let diagnostic = p.policies.iter().filter(|&&p| p == Policy::Observation).count();
            let required = p.expected.len - diagnostic;
            if required > 0 || p.expected.len == 0 {
                s.required_routes += 1;
            }
            if diagnostic > 0 {
                s.diagnostic_routes += 1;
            }
            s.required_values += required;
            s.diagnostic_values += diagnostic;
            s.model_values += p.policies.iter().filter(|p| matches!(p, Policy::ProjectInterval { .. })).count();
            if p.source {
                s.source_values += p.expected.len;
            }
            if p.key.route == "consumer" {
                s.consumer_routes += 1;
            }
            if matches!(p.key.route, "autodiff" | "autodiff_backing" | "autodiff_transpose" | "autodiff_staging") && !p.key.output.ends_with("forward") {
                s.gradient_values += p.expected.len;
            }
            match p.expected.dtype {
                DType::F16 => s.f16_values += p.expected.len,
                DType::BF16 => s.bf16_values += p.expected.len,
                DType::Flex32 => s.flex32_values += p.expected.len,
                DType::F64 => s.f64_values += p.expected.len,
                _ => {}
            }
        }
    }
    s
}
#[derive(Clone, Debug)]
pub struct Failure {
    pub key: RouteKey,
    pub message: String,
}
#[derive(Clone, Debug)]
pub struct NumericWitness {
    pub key: RouteKey,
    pub cell: usize,
    pub stored_stage: &'static str,
    pub dtype: DType,
    pub actual: u64,
    pub expected: u64,
    pub ulp_distance: u64,
    pub ideal_rational: &'static str,
    pub sum_abs_products: &'static str,
    pub absolute_error: Option<f64>,
    pub relative_error: Option<f64>,
    pub lhs: Vec<u64>,
    pub rhs: Vec<u64>,
    pub lhs_shape: Vec<usize>,
    pub rhs_shape: Vec<usize>,
    pub stage_witness: StageWitness,
}
#[derive(Clone, Debug)]
pub struct Observation {
    pub key: RouteKey,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub values: Vec<u64>,
    pub required_checked: usize,
    pub diagnostic_observed: usize,
    pub model_checked: usize,
    pub zero_sign_differences: usize,
}
#[derive(Clone, Debug)]
pub struct Report {
    pub expected: Summary,
    pub attempted: usize,
    pub passed: usize,
    pub failed: usize,
    pub required_checked: usize,
    pub diagnostic_observed: usize,
    pub model_checked: usize,
    pub checked_required_routes: usize,
    pub observed_diagnostic_routes: usize,
    pub failures: Vec<Failure>,
    pub observations: Vec<Observation>,
    pub worst: Option<NumericWitness>,
    pub unobserved: usize,
    pub external_unobserved: usize,
    pub tracking_failures: usize,
    max_failures: usize,
    finalized: bool,
    seen: BTreeSet<String>,
    planned: Vec<Plan>,
    #[cfg(target_os = "none")]
    pending_reads: Vec<(RouteKey, FloatTensor<Dispatch>)>,
}
impl Report {
    pub fn new(options: &Options, scope: Scope) -> Self {
        let expected = expected_summary(options, scope);
        let external_unobserved = expected.external_gates;
        Self {
            expected,
            attempted: 0,
            passed: 0,
            failed: 0,
            required_checked: 0,
            diagnostic_observed: 0,
            model_checked: 0,
            checked_required_routes: 0,
            observed_diagnostic_routes: 0,
            failures: Vec::new(),
            observations: Vec::new(),
            worst: None,
            unobserved: 0,
            external_unobserved,
            tracking_failures: 0,
            max_failures: options.max_failures,
            finalized: false,
            seen: BTreeSet::new(),
            planned: CASES.iter().filter(|c| selected(c, options, scope)).flat_map(plans).collect(),
            #[cfg(target_os = "none")]
            pending_reads: Vec::new(),
        }
    }
    fn fail(&mut self, key: RouteKey, message: String) {
        self.failed += 1;
        if self.failures.len() < self.max_failures {
            self.failures.push(Failure { key, message });
        }
    }
    /// Reject duplicate/unexpected route keys and wrong metadata before counting.
    pub fn observe(&mut self, key: RouteKey, dtype: DType, shape: &[usize], values: Vec<u64>) {
        if self.finalized {
            self.fail(key, String::from("observation after report finalization"));
            return;
        }
        let Some(p) = self.planned.iter().find(|p| p.key == key).cloned() else {
            self.fail(key, String::from("unexpected route key"));
            return;
        };
        if !self.seen.insert(key.text()) {
            self.fail(key, String::from("duplicate route key"));
            return;
        }
        self.attempted += 1;
        if dtype != p.expected.dtype || shape != p.expected.shape || values.len() != p.expected.len {
            self.fail(
                key,
                format!(
                    "metadata/count actual={dtype:?}/{shape:?}/{} expected={:?}/{:?}/{}",
                    values.len(),
                    p.expected.dtype,
                    p.expected.shape,
                    p.expected.len
                ),
            );
            return;
        }
        let expected = p.expected.bits();
        let mut required = 0;
        let mut diagnostic = 0;
        let mut model = 0;
        let mut zeros = 0;
        let mut failures = 0;
        for (i, ((&a, &e), &policy)) in values.iter().zip(&expected).zip(&p.policies).enumerate() {
            if policy == Policy::Observation {
                diagnostic += 1;
            } else {
                required += 1;
            }
            if matches!(policy, Policy::ProjectInterval { .. }) {
                model += 1;
            }
            if classification(dtype, a) == Class::Zero && classification(dtype, e) == Class::Zero && a != e {
                zeros += 1;
            }
            if !compare(dtype, policy, e, a) {
                failures += 1;
                self.fail(
                    key.clone(),
                    format!(
                        "cell={i} stage={} policy={policy:?} actual={a:#x} expected={e:#x} class={:?}",
                        p.stage,
                        classification(dtype, a)
                    ),
                );
            }
            if !p.source && numeric_value(dtype, a).is_some() && numeric_value(dtype, e).is_some() {
                let ulp = ordered(dtype, a).abs_diff(ordered(dtype, e));
                if self.worst.as_ref().is_none_or(|w| ulp > w.ulp_distance) {
                    let c = fixture(key.case_id).expect("planned case");
                    let Some(stage_witness) = stage_witness(c, &key, p.expected.shape) else {
                        self.fail(key.clone(), String::from("stored stage operand binding absent"));
                        continue;
                    };
                    let lhs = stage_witness.operands.first();
                    let rhs = stage_witness.operands.get(1);
                    let cell = c.numeric.get(i).filter(|_| key.output == "expected");
                    let ideal_rational = cell.map_or("", |c| c.ideal_rational);
                    let e_value = if ideal_rational.is_empty() {
                        numeric_value(dtype, e)
                    } else {
                        rational_approximation(ideal_rational)
                    };
                    let a_value = numeric_value(dtype, a);
                    let absolute = a_value.zip(e_value).map(|(a, e)| (a - e).abs());
                    self.worst = Some(NumericWitness {
                        key: key.clone(),
                        cell: i,
                        stored_stage: p.stage,
                        dtype,
                        actual: a,
                        expected: e,
                        ulp_distance: ulp,
                        ideal_rational,
                        sum_abs_products: cell.map_or("", |c| c.sum_abs_products),
                        absolute_error: absolute,
                        relative_error: absolute.zip(e_value).and_then(|(a, e)| if e != 0.0 { Some(a / e.abs()) } else { None }),
                        lhs: lhs.and_then(|operand| operand.bits.clone()).unwrap_or_default(),
                        rhs: rhs.and_then(|operand| operand.bits.clone()).unwrap_or_default(),
                        lhs_shape: lhs.map_or_else(Vec::new, |operand| operand.shape.clone()),
                        rhs_shape: rhs.map_or_else(Vec::new, |operand| operand.shape.clone()),
                        stage_witness,
                    });
                }
            }
        }
        self.required_checked += required;
        self.diagnostic_observed += diagnostic;
        self.model_checked += model;
        if required > 0 || values.is_empty() {
            self.checked_required_routes += 1;
        }
        if diagnostic > 0 {
            self.observed_diagnostic_routes += 1;
        }
        if failures == 0 && (required > 0 || values.is_empty()) {
            self.passed += 1;
        }
        self.observations.push(Observation {
            key,
            dtype,
            shape: shape.to_vec(),
            values,
            required_checked: required,
            diagnostic_observed: diagnostic,
            model_checked: model,
            zero_sign_differences: zeros,
        });
    }
    pub fn finish(&mut self) {
        if self.finalized {
            return;
        }
        let missing: Vec<_> = self
            .planned
            .iter()
            .filter(|p| !self.seen.contains(&p.key.text()))
            .map(|p| p.key.clone())
            .collect();
        self.unobserved = missing.len();
        for key in missing {
            self.fail(key, String::from("required route unobserved (including mandatory diagnostic observation)"));
        }
        self.external_unobserved = self.expected.external_gates;
        self.finalized = true;
    }
    pub fn is_success(&self) -> bool {
        self.finalized
            && self.expected.selected_rows > 0
            && self.failed == 0
            && self.unobserved == 0
            && self.external_unobserved == 0
            && self.tracking_failures == 0
            && self.seen.len() == self.planned.len()
            && self.required_checked == self.expected.required_values
            && self.diagnostic_observed == self.expected.diagnostic_values
            && self.model_checked == self.expected.model_values
            && self.checked_required_routes == self.expected.required_routes
            && self.observed_diagnostic_routes == self.expected.diagnostic_routes
    }
}
/// Diagnostic conversion only. No host value supplies an expected result.
fn rational_approximation(value: &str) -> Option<f64> {
    let (numerator, denominator) = value.split_once('/').unwrap_or((value, "1"));
    let result = numerator.parse::<f64>().ok()? / denominator.parse::<f64>().ok()?;
    result.is_finite().then_some(result)
}
fn numeric_value(dtype: DType, bits: u64) -> Option<f64> {
    let class = classification(dtype, bits);
    if matches!(class, Class::NaN | Class::Infinity) {
        return None;
    }
    if dtype == DType::F64 {
        return Some(f64::from_bits(bits));
    }
    if matches!(dtype, DType::F32 | DType::Flex32) {
        return Some(f32::from_bits(bits as u32) as f64);
    }
    let (e, f) = fields(dtype);
    let sign = if bits >> (e + f) == 0 { 1.0 } else { -1.0 };
    let exp = bits >> f & ((1 << e) - 1);
    let fraction = bits & ((1 << f) - 1);
    let bias = (1 << (e - 1)) - 1;
    let (mantissa, power) = if exp == 0 {
        (fraction, 1 - bias - f as i32)
    } else {
        ((1 << f) + fraction, exp as i32 - bias - f as i32)
    };
    Some(sign * mantissa as f64 * libm::pow(2.0, power as f64))
}

type FT = FloatTensor<Dispatch>;
fn upload(values: Values, device: &Device) -> FT {
    Dispatch::float_from_data(
        TensorData::from_bytes_vec(values.bytes().to_vec(), values.shape.to_vec(), values.dtype),
        device.as_dispatch(),
    )
}
/// Rebuild the frozen affine views through actual Dispatch slice/reshape/flip.
/// Root reads establish byte identity, not physical kernel guard coverage.
fn build(matrix: &Matrix, device: &Device) -> Result<(FT, FT), String> {
    let Some(v) = matrix.view else {
        let t = upload(matrix.values, device);
        return Ok((t.clone(), t));
    };
    let root = upload(v.root, device);
    let t = build_view(matrix, root.clone())?;
    Ok((root, t))
}
fn build_view(matrix: &Matrix, root: FT) -> Result<FT, String> {
    let v = matrix.view.ok_or("view descriptor absent")?;
    let shape = matrix.values.shape;
    if matrix.values.len == 0 {
        return Ok(Dispatch::float_reshape(
            Dispatch::float_slice(root, &[Slice::new(0, Some(0), 1)]),
            shape.to_vec().into(),
        ));
    }
    let low = shape
        .iter()
        .zip(v.strides)
        .fold(v.base as isize, |n, (&d, &s)| n + ((d - 1) as isize * s).min(0));
    let dense: Vec<_> = (0..shape.len()).map(|i| shape[i + 1..].iter().product::<usize>() as isize).collect();
    let transposed = v.strides.len() >= 2 && v.strides[v.strides.len() - 2] == 1;
    let step = if transposed {
        1
    } else {
        v.strides.last().copied().unwrap_or(1).unsigned_abs()
    };
    if low < 0 || step == 0 {
        return Err(String::from("invalid generated affine view"));
    }
    let mut t = Dispatch::float_slice(root.clone(), &[Slice::new(low, Some(low + (matrix.values.len * step) as isize), step as isize)]);
    if transposed {
        let mut physical = shape.to_vec();
        let n = physical.len();
        physical.swap(n - 2, n - 1);
        t = Dispatch::float_reshape(t, physical.into());
        t = Dispatch::float_swap_dims(t, n - 2, n - 1);
    } else {
        if !v.strides.iter().zip(&dense).all(|(&a, &b)| a.unsigned_abs() == b.unsigned_abs() * step) {
            return Err(String::from("unsupported generated affine strides"));
        }
        t = Dispatch::float_reshape(t, shape.to_vec().into());
        let axes: Vec<_> = v.strides.iter().enumerate().filter(|(_, s)| **s < 0).map(|(i, _)| i).collect();
        if !axes.is_empty() {
            t = Dispatch::float_flip(t, &axes);
        }
    }
    Ok(t)
}
async fn boundary(t: FT, device: &Device, o: &Options) -> Result<FT, String> {
    #[cfg(not(target_os = "none"))]
    if o.materialize_chain_steps {
        device.flush().map_err(|e| format!("nonblocking graph flush: {e:?}"))?;
        Dispatch::float_into_data(t.clone())
            .await
            .map_err(|e| format!("stored graph materialization: {e:?}"))?;
    }
    #[cfg(target_os = "none")]
    let _ = (device, o);
    Ok(t)
}
fn public_matmul<const D: usize>(a: FT, b: FT) -> FT {
    let mut shape = b.shape().to_vec();
    if shape.len() < D {
        shape.splice(0..0, core::iter::repeat_n(1, D - shape.len()));
    }
    let b = if b.shape().num_dims() == D {
        b
    } else {
        Dispatch::float_reshape(b, shape.into())
    };
    Tensor::<D>::from_dispatch(a).matmul(Tensor::<D>::from_dispatch(b)).into_dispatch()
}
fn public_linear<const D: usize>(a: FT, b: FT, bias: Option<FT>) -> FT {
    burn::tensor::module::linear(
        Tensor::<D>::from_dispatch(a),
        Tensor::<2>::from_dispatch(b),
        bias.map(Tensor::<1>::from_dispatch),
    )
    .into_dispatch()
}
fn public_forward(a: FT, b: FT, bias: Option<FT>, linear: bool) -> FT {
    macro_rules! call {
        ($d:literal) => {
            if linear {
                public_linear::<$d>(a, b, bias)
            } else {
                public_matmul::<$d>(a, b)
            }
        };
    }
    match a.shape().num_dims() {
        1 => call!(1),
        2 => call!(2),
        3 => call!(3),
        4 => call!(4),
        5 => call!(5),
        6 => call!(6),
        7 => call!(7),
        8 => call!(8),
        _ => panic!("fixture rank outside 1..8"),
    }
}
async fn read_report(r: &mut Report, key: RouteKey, t: FT) {
    #[cfg(target_os = "none")]
    {
        r.pending_reads.push((key, t));
    }
    #[cfg(not(target_os = "none"))]
    read_report_now(r, key, t).await;
}
async fn read_report_now(r: &mut Report, key: RouteKey, t: FT) {
    let dtype = t.dtype();
    let shape = t.shape().to_vec();
    match Dispatch::float_into_data(t).await {
        Ok(data) => match unpack(dtype, data.as_bytes()) {
            Ok(bits) => r.observe(key, data.dtype(), data.shape().as_slice(), bits),
            Err(e) => r.fail(key, e),
        },
        Err(e) => r.fail(key, format!("device read dtype={dtype:?} shape={shape:?}: {e:?}")),
    }
}
async fn output(r: &mut Report, c: &Case, route: &'static str, name: &'static str, t: FT, outputs: &mut Vec<(&'static str, FT)>) {
    read_report(
        r,
        RouteKey {
            case_id: c.id,
            route,
            output: name,
        },
        t.clone(),
    )
    .await;
    outputs.push((name, t));
}
fn get(outputs: &[(&'static str, FT)], name: &str) -> Option<FT> {
    outputs.iter().find(|(n, _)| *n == name).map(|(_, t)| t.clone())
}
fn transpose(t: FT) -> FT {
    let rank = t.shape().num_dims();
    Dispatch::float_swap_dims(t, rank - 2, rank - 1)
}
async fn explicit_backward(
    c: &Case, inputs: (FT, FT, FT), device: &Device, o: &Options, r: &mut Report, outputs: &mut Vec<(&'static str, FT)>,
) -> Result<(), String> {
    let (a, b, g) = inputs;
    let mut da = boundary(Dispatch::float_matmul(g.clone(), transpose(b.clone())), device, o).await?;
    let mut db = boundary(Dispatch::float_matmul(transpose(a.clone()), g), device, o).await?;
    for (name, t) in [("dA_before_sum", da.clone()), ("dB_before_sum", db.clone())] {
        if c.expectation(name).is_some() {
            output(r, c, "primitive", name, t, outputs).await;
        }
    }
    for (side, gradient, target) in [("dA_axes", &mut da, a.shape()), ("dB_axes", &mut db, b.shape())] {
        // Original axis indices remain stable because SUM keeps dimensions.
        for axis in 0..target.num_dims() {
            if gradient.shape()[axis] != target[axis] {
                *gradient = boundary(Dispatch::float_sum_dim(gradient.clone(), axis), device, o).await?;
                let label = format!("{side}.{axis}");
                if let Some(e) = c.expected.iter().find(|e| e.name == label) {
                    output(r, c, "primitive", e.name, gradient.clone(), outputs).await;
                }
            }
        }
    }
    output(r, c, "primitive", "dA", da.clone(), outputs).await;
    output(r, c, "primitive", "dB", db.clone(), outputs).await;
    if c.tracking == Tracking::Shared {
        output(r, c, "primitive", "shared_source_gradient", Dispatch::float_add(da, db), outputs).await;
    }
    Ok(())
}
async fn numeric_staging(
    c: &Case, device: &Device, o: &Options, r: &mut Report, inputs: &[(&'static str, FT)], outputs: &mut Vec<(&'static str, FT)>,
) -> Result<(), String> {
    if let (Some(x), Some(w), Some(b)) = (c.matrix("linear.input"), c.matrix("linear.weight"), c.matrix("linear.bias")) {
        let (x, w, b) = (
            get(inputs, x.name).ok_or("staging input absent")?,
            get(inputs, w.name).ok_or("staging weight absent")?,
            get(inputs, b.name).ok_or("staging bias absent")?,
        );
        let stored = boundary(Dispatch::float_matmul(x.clone(), w.clone()), device, o).await?;
        output(r, c, "primitive", "linear.stored", stored.clone(), outputs).await;
        let y = boundary(Dispatch::float_add(stored, b.clone()), device, o).await?;
        output(r, c, "primitive", "linear.output", y, outputs).await;
        // Public linear is independently checked against the same staged bits.
        let y = public_forward(x, w, Some(b), true);
        if let Some(e) = c.expectation("linear.output") {
            read_against(r, c, "public_linear_staging", e, y).await;
        }
    }
    if let (Some(a), Some(b), Some(g)) = (c.matrix("gradient.lhs"), c.matrix("gradient.rhs"), c.matrix("gradient.upstream")) {
        let (a, b, g) = (
            get(inputs, a.name).ok_or("staging lhs absent")?,
            get(inputs, b.name).ok_or("staging rhs absent")?,
            get(inputs, g.name).ok_or("staging upstream absent")?,
        );
        let y = boundary(Dispatch::float_matmul(a.clone(), b.clone()), device, o).await?;
        output(r, c, "primitive", "gradient.forward", y, outputs).await;
        let da0 = boundary(Dispatch::float_matmul(g.clone(), transpose(b)), device, o).await?;
        output(r, c, "primitive", "gradient.dA_before_sum", da0.clone(), outputs).await;
        let da = boundary(Dispatch::float_sum_dim(da0, 0), device, o).await?;
        output(r, c, "primitive", "gradient.dA", da, outputs).await;
        // Standard tracked public path for the midpoint storage-before-SUM gate.
        let a = Tensor::<3>::from_dispatch(a).autodiff().require_grad();
        let b = Tensor::<3>::from_dispatch(upload(c.matrix("gradient.rhs").unwrap().values, device)).autodiff();
        let y = a.clone().matmul(b);
        let g = Tensor::<3>::from_dispatch(g).autodiff();
        let grads = (y * g).sum().backward();
        let da = a.grad(&grads).ok_or("midpoint autodiff dA absent")?.into_dispatch();
        if let Some(e) = c.expectation("gradient.dA") {
            read_against(r, c, "autodiff_staging", e, da).await;
        }
    }
    Ok(())
}
async fn read_against(r: &mut Report, c: &Case, route: &'static str, e: &Expected, t: FT) {
    // This route is part of the pure plan; it cannot add runtime expectations.
    read_report(
        r,
        RouteKey {
            case_id: c.id,
            route,
            output: e.name,
        },
        t,
    )
    .await;
}
async fn autodiff_case<const D: usize>(c: &Case, device: &Device, o: &Options, r: &mut Report) -> Result<(), String> {
    let linear = c.operation == Operation::Linear;
    let am = c.matrix(if linear { "input" } else { "lhs" }).ok_or("missing autodiff lhs")?;
    let bm = c.matrix(if linear { "weight" } else { "rhs" }).ok_or("missing autodiff rhs")?;
    let (_, av) = build(am, device)?;
    let (_, bv) = build(bm, device)?;
    // Leaf handles retained for gradient lookup are saved clones. User handles
    // go out of scope before backward; the graph owns its saved operands.
    let backing = c.expectation("lhs_backing_adjoint").is_some();
    let mut a = Tensor::<D>::from_dispatch(av).autodiff();
    let mut b_dispatch = bv;
    if b_dispatch.shape().num_dims() < D && !linear {
        let mut shape = b_dispatch.shape().to_vec();
        shape.splice(0..0, core::iter::repeat_n(1, D - shape.len()));
        b_dispatch = Dispatch::float_reshape(b_dispatch, shape.into());
    }
    if !matches!(c.tracking, Tracking::Rhs) {
        a = a.require_grad();
    }
    let saved_a = a.clone();
    let (y, saved_b, bias) = if linear {
        let b = Tensor::<2>::from_dispatch(b_dispatch).autodiff().require_grad();
        let bias = Tensor::<1>::from_dispatch(upload(c.matrix("bias").ok_or("missing bias")?.values, device))
            .autodiff()
            .require_grad();
        let y = burn::tensor::module::linear(a, b.clone(), Some(bias.clone()));
        (y, Some(b.into_dispatch()), Some(bias))
    } else {
        let mut b = Tensor::<D>::from_dispatch(b_dispatch).autodiff();
        if !matches!(c.tracking, Tracking::Lhs) {
            b = b.require_grad();
        }
        if c.tracking == Tracking::Shared {
            b = saved_a.clone();
        }
        let y = a.matmul(b.clone());
        (y, Some(b.into_dispatch()), None)
    };
    let y = Tensor::<D>::from_dispatch(boundary(y.into_dispatch(), device, o).await?);
    read_report(
        r,
        RouteKey {
            case_id: c.id,
            route: "autodiff",
            output: "forward",
        },
        y.clone().into_dispatch(),
    )
    .await;
    let g = Tensor::<D>::from_dispatch(upload(c.matrix("upstream").ok_or("missing upstream")?.values, device)).autodiff();
    let saved_g = g.clone();
    let grads = (y * g).sum().backward();
    read_report(
        r,
        RouteKey {
            case_id: c.id,
            route: "autodiff_source_after",
            output: "upstream",
        },
        saved_g.into_dispatch(),
    )
    .await;
    read_report(
        r,
        RouteKey {
            case_id: c.id,
            route: "autodiff_source_after",
            output: am.name,
        },
        saved_a.clone().into_dispatch(),
    )
    .await;
    if let Some(b) = saved_b.as_ref() {
        read_report(
            r,
            RouteKey {
                case_id: c.id,
                route: "autodiff_source_after",
                output: bm.name,
            },
            b.clone(),
        )
        .await;
    }
    if let Some(b) = bias.as_ref() {
        read_report(
            r,
            RouteKey {
                case_id: c.id,
                route: "autodiff_source_after",
                output: "bias",
            },
            b.clone().into_dispatch(),
        )
        .await;
    }
    if let Some(ga) = saved_a.grad(&grads) {
        let name = if linear {
            "dx"
        } else if c.tracking == Tracking::Shared {
            "shared_source_gradient"
        } else {
            "dA"
        };
        read_report(
            r,
            RouteKey {
                case_id: c.id,
                route: "autodiff",
                output: name,
            },
            ga.into_dispatch(),
        )
        .await;
    } else if c.tracking != Tracking::Rhs {
        return Err(String::from("tracked lhs gradient absent"));
    }
    if let Some(b) = saved_b {
        if linear {
            let b = Tensor::<2>::from_dispatch(b);
            let gb = b.grad(&grads).ok_or("weight gradient absent")?;
            read_report(
                r,
                RouteKey {
                    case_id: c.id,
                    route: "autodiff",
                    output: "dW",
                },
                gb.into_dispatch(),
            )
            .await;
        } else {
            let b = Tensor::<D>::from_dispatch(b);
            if matches!(c.tracking, Tracking::Lhs) {
                if b.grad(&grads).is_some() {
                    r.tracking_failures += 1;
                    return Err(String::from("untracked rhs unexpectedly has gradient"));
                }
            } else if c.tracking != Tracking::Shared {
                read_report(
                    r,
                    RouteKey {
                        case_id: c.id,
                        route: "autodiff",
                        output: "dB",
                    },
                    b.grad(&grads).ok_or("rhs gradient absent")?.into_dispatch(),
                )
                .await;
            }
        }
    }
    if c.tracking == Tracking::Rhs && saved_a.grad(&grads).is_some() {
        r.tracking_failures += 1;
        return Err(String::from("untracked lhs unexpectedly has gradient"));
    }
    if let Some(bias) = bias {
        read_report(
            r,
            RouteKey {
                case_id: c.id,
                route: "autodiff",
                output: "db",
            },
            bias.grad(&grads).ok_or("bias gradient absent")?.into_dispatch(),
        )
        .await;
    }
    if backing {
        // Separate public affine graph roots, not a host scatter masquerading
        // as a device gradient. Construct views from tracked roots below.
        autodiff_backing::<D>(c, device, r).await?;
    }
    Ok(())
}
async fn autodiff_backing<const D: usize>(c: &Case, device: &Device, r: &mut Report) -> Result<(), String> {
    let am = c.matrix("lhs").ok_or("missing view lhs")?;
    let bm = c.matrix("rhs").ok_or("missing view rhs")?;
    let ar = Tensor::<1>::from_dispatch(upload(am.view.ok_or("lhs view absent")?.root, device))
        .autodiff()
        .require_grad();
    let br = Tensor::<1>::from_dispatch(upload(bm.view.ok_or("rhs view absent")?.root, device))
        .autodiff()
        .require_grad();
    let a = Tensor::<D>::from_dispatch(build_view(am, ar.clone().into_dispatch())?);
    let b = Tensor::<D>::from_dispatch(build_view(bm, br.clone().into_dispatch())?);
    let y = a.matmul(b);
    let g = Tensor::<D>::from_dispatch(upload(c.matrix("upstream").ok_or("upstream absent")?.values, device)).autodiff();
    let grads = (y * g).sum().backward();
    for (name, t) in [("lhs_backing_adjoint", ar.grad(&grads)), ("rhs_backing_adjoint", br.grad(&grads))] {
        read_report(
            r,
            RouteKey {
                case_id: c.id,
                route: "autodiff_backing",
                output: name,
            },
            t.ok_or("physical view root adjoint absent")?.into_dispatch(),
        )
        .await;
    }
    read_report(
        r,
        RouteKey {
            case_id: c.id,
            route: "autodiff_backing_source_after",
            output: "lhs",
        },
        ar.into_dispatch(),
    )
    .await;
    read_report(
        r,
        RouteKey {
            case_id: c.id,
            route: "autodiff_backing_source_after",
            output: "rhs",
        },
        br.into_dispatch(),
    )
    .await;
    Ok(())
}
async fn shared_transpose(c: &Case, device: &Device, r: &mut Report) -> Result<(), String> {
    let a = Tensor::<2>::from_dispatch(upload(c.matrix("lhs").ok_or("shared source absent")?.values, device))
        .autodiff()
        .require_grad();
    let y = a.clone().matmul(a.clone().transpose());
    read_report(
        r,
        RouteKey {
            case_id: c.id,
            route: "autodiff_transpose",
            output: "shared_transpose.forward",
        },
        y.clone().into_dispatch(),
    )
    .await;
    let g = Tensor::<2>::from_dispatch(upload(c.matrix("upstream").ok_or("shared upstream absent")?.values, device)).autodiff();
    let grads = (y * g).sum().backward();
    read_report(
        r,
        RouteKey {
            case_id: c.id,
            route: "autodiff_transpose",
            output: "shared_transpose.gradient",
        },
        a.grad(&grads).ok_or("shared transpose gradient absent")?.into_dispatch(),
    )
    .await;
    Ok(())
}
async fn autodiff_by_rank(c: &Case, device: &Device, o: &Options, r: &mut Report) -> Result<(), String> {
    let rank = c.matrix("lhs").or_else(|| c.matrix("input")).ok_or("autodiff input absent")?.values.shape.len();
    match rank {
        2 => autodiff_case::<2>(c, device, o, r).await,
        3 => autodiff_case::<3>(c, device, o, r).await,
        4 => autodiff_case::<4>(c, device, o, r).await,
        5 => autodiff_case::<5>(c, device, o, r).await,
        6 => autodiff_case::<6>(c, device, o, r).await,
        7 => autodiff_case::<7>(c, device, o, r).await,
        8 => autodiff_case::<8>(c, device, o, r).await,
        _ => Err(String::from("autodiff rank outside selected corpus")),
    }
}
async fn run_case(c: &Case, device: &Device, o: &Options, r: &mut Report) -> Result<(), String> {
    let device = &device.clone().without_autodiff();
    let mut inputs = Vec::new();
    let mut roots = Vec::new();
    let mut outputs = Vec::new();
    for m in c.matrices {
        let (root, t) = build(m, device)?;
        read_report(
            r,
            RouteKey {
                case_id: c.id,
                route: "source_before",
                output: m.name,
            },
            root.clone(),
        )
        .await;
        roots.push((m.name, root.clone(), root));
        inputs.push((m.name, t));
    }
    if c.group == Group::F64Diagnostic && c.matrix("lhs").is_none() {
        for i in 0..2 {
            let an = format!("zero{i}.lhs");
            let bn = format!("zero{i}.rhs");
            let en = format!("zero{i}.expected");
            if let (Some(a), Some(b), Some(e)) = (get(&inputs, &an), get(&inputs, &bn), c.expected.iter().find(|e| e.name == en)) {
                output(r, c, "primitive", e.name, Dispatch::float_matmul(a, b), &mut outputs).await;
            }
        }
    } else {
        let linear = c.operation == Operation::Linear;
        let a = get(&inputs, if linear { "input" } else { "lhs" }).ok_or("lhs missing")?;
        let b = get(&inputs, if linear { "weight" } else { "rhs" }).ok_or("rhs missing")?;
        let name = if c.expectation("forward").is_some() { "forward" } else { "expected" };
        let y = if linear {
            let shape = a.shape().to_vec();
            let rows = shape[..shape.len() - 1].iter().product::<usize>();
            let flat = Dispatch::float_reshape(a.clone(), vec![rows, shape[shape.len() - 1]].into());
            let stored = boundary(Dispatch::float_matmul(flat, b.clone()), device, o).await?;
            let out_shape = c.expectation("forward").ok_or("linear forward absent")?.values.shape;
            output(r, c, "primitive", "stored_forward", stored.clone(), &mut outputs).await;
            let stored = Dispatch::float_reshape(stored, out_shape.to_vec().into());
            boundary(Dispatch::float_add(stored, get(&inputs, "bias").ok_or("bias missing")?), device, o).await?
        } else {
            boundary(Dispatch::float_matmul(a.clone(), b.clone()), device, o).await?
        };
        output(r, c, "primitive", name, y, &mut outputs).await;
        let y = public_forward(a.clone(), b.clone(), get(&inputs, "bias"), linear);
        output(r, c, "public", name, y, &mut Vec::new()).await;
        if c.tracking != Tracking::None {
            let g = get(&inputs, "upstream").ok_or("upstream missing")?;
            if linear {
                let dx = boundary(Dispatch::linear_x_backward(b.clone(), g.clone()), device, o).await?;
                let dw = boundary(Dispatch::linear_weight_backward(a.clone(), g.clone()), device, o).await?;
                let db = boundary(Dispatch::linear_bias_backward(g), device, o).await?;
                for (name, t) in [("dx", dx), ("dW", dw), ("db", db)] {
                    output(r, c, "primitive", name, t, &mut outputs).await;
                }
            } else {
                explicit_backward(c, (a, b, g), device, o, r, &mut outputs).await?;
            }
            autodiff_by_rank(c, device, o, r).await?;
            if c.tracking == Tracking::Shared {
                shared_transpose(c, device, r).await?;
            }
        }
        if c.group == Group::Numeric {
            numeric_staging(c, device, o, r, &inputs, &mut outputs).await?;
        }
    }
    // Consumers own the producer handle after its final temporary drops.
    drop(inputs);
    let consumers = c
        .consumers
        .iter()
        .map(|consumer| {
            get(&outputs, consumer.input)
                .map(|t| (consumer, t))
                .ok_or_else(|| format!("consumer producer {} absent", consumer.input))
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(outputs);
    for (consumer, t) in consumers {
        let t = boundary(t, device, o).await?;
        let y = Dispatch::float_sum_dim(t, consumer.axis);
        read_report(
            r,
            RouteKey {
                case_id: c.id,
                route: "consumer",
                output: consumer.expected.name,
            },
            y,
        )
        .await;
    }
    for (name, root, alias) in roots {
        read_report(
            r,
            RouteKey {
                case_id: c.id,
                route: "source_after",
                output: name,
            },
            root,
        )
        .await;
        read_report(
            r,
            RouteKey {
                case_id: c.id,
                route: "alias_after",
                output: name,
            },
            alias,
        )
        .await;
    }
    Ok(())
}
#[cfg(not(target_os = "none"))]
extern crate std;
async fn guarded_case(c: &Case, device: &Device, o: &Options, r: &mut Report) -> Result<(), String> {
    #[cfg(not(target_os = "none"))]
    {
        use core::{future::Future, task::Poll};
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let mut future = core::pin::pin!(run_case(c, device, o, r));
        core::future::poll_fn(|cx| match catch_unwind(AssertUnwindSafe(|| future.as_mut().poll(cx))) {
            Ok(poll) => poll,
            Err(payload) => {
                let message = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| String::from(*s)))
                    .unwrap_or_else(|| String::from("non-string panic"));
                Poll::Ready(Err(format!("backend panicked: {message}")))
            }
        })
        .await
    }
    #[cfg(target_os = "none")]
    run_case(c, device, o, r).await
}
/// Execute every selected case when the device remains usable. Panics on Linux
/// become failed routes; unobserved and poisoned work never qualifies.
pub async fn run(device: &Device, options: &Options, scope: Scope) -> Report {
    let mut report = Report::new(options, scope);
    for c in CASES.iter().filter(|c| selected(c, options, scope)) {
        if c.operation == Operation::ExternalGate {
            continue;
        }
        if let Err(message) = guarded_case(c, device, options, &mut report).await {
            report.fail(
                RouteKey {
                    case_id: c.id,
                    route: "execution",
                    output: "case",
                },
                message,
            );
        }
    }
    #[cfg(target_os = "none")]
    for (key, tensor) in core::mem::take(&mut report.pending_reads) {
        read_report_now(&mut report, key, tensor).await;
    }
    report.finish();
    report
}
/// Isolated source-derived frontend diagnostic. A panic occurs before Native
/// dispatch for rank-one empty input. This is never a passing public row.
pub fn frontend_zero_linear_diagnostic(device: &Device) {
    let dtype = DType::F16;
    let x = Tensor::<1>::from_data(TensorData::from_bytes_vec(Vec::new(), [0], dtype), device);
    let w = Tensor::<2>::from_data(TensorData::from_bytes_vec(Vec::new(), [0, 7], dtype), device);
    let bias = Tensor::<1>::from_data(TensorData::from_bytes_vec(pack(dtype, &[0; 7]), [7], dtype), device);
    let _ = burn::tensor::module::linear(x, w, Some(bias));
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
fn key_json(k: &RouteKey) -> String {
    format!(
        "{{\"case_id\":{},\"route\":{},\"output\":{}}}",
        json_string(k.case_id),
        json_string(k.route),
        json_string(k.output)
    )
}
fn optional_number(value: Option<f64>) -> String {
    value.filter(|x| x.is_finite()).map_or_else(|| String::from("null"), |x| format!("{x}"))
}
pub fn summary_json(s: &Summary) -> String {
    format!(
        concat!(
            "{{\"version\":{},\"corpus_rows\":121,\"reserves\":3,\"numeric_cells\":65,\"corpus_last_drop_consumers\":141,",
            "\"selected_rows\":{},\"required_routes\":{},\"diagnostic_routes\":{},\"required_values\":{},\"diagnostic_values\":{},\"model_values\":{},",
            "\"source_values\":{},\"gradient_values\":{},\"consumer_routes\":{},\"external_gates\":{},",
            "\"groups\":{{\"geometry\":{},\"numeric\":{},\"gradient\":{},\"native_extensions\":{},\"preservation\":{},\"f64_diagnostics\":{}}},",
            "\"dtype_values\":{{\"F16\":{},\"BF16\":{},\"Flex32\":{},\"F64\":{}}},",
            "\"conditional_model\":{},\"vendor_rounding_guarantee\":false,\"device_observations\":0,\"physical_proof\":false}}"
        ),
        CORPUS_VERSION,
        s.selected_rows,
        s.required_routes,
        s.diagnostic_routes,
        s.required_values,
        s.diagnostic_values,
        s.model_values,
        s.source_values,
        s.gradient_values,
        s.consumer_routes,
        s.external_gates,
        s.geometry_rows,
        s.numeric_rows,
        s.gradient_rows,
        s.native_extension_rows,
        s.preservation_rows,
        s.f64_rows,
        s.f16_values,
        s.bf16_values,
        s.flex32_values,
        s.f64_values,
        json_string(MODEL_LABEL)
    )
}
/// Expected metadata populations, useful to audit a filtered run independently
/// from observations. Keys are unique across sources, consumers and AD routes.
fn policy_json(policy: Policy) -> String {
    match policy {
        Policy::Exact => String::from("{\"kind\":\"exact\"}"),
        Policy::ZeroClass => String::from("{\"kind\":\"zero_class\"}"),
        Policy::NaNClass => String::from("{\"kind\":\"nan_class\"}"),
        Policy::Infinity => String::from("{\"kind\":\"infinity\"}"),
        Policy::ProjectInterval { low, high } => format!("{{\"kind\":\"project_interval\",\"low\":{low},\"high\":{high}}}"),
        Policy::Observation => String::from("{\"kind\":\"observation\"}"),
    }
}
pub fn population_json(options: &Options, scope: Scope) -> String {
    let population: Vec<_> = CASES
        .iter()
        .filter(|c| selected(c, options, scope))
        .flat_map(plans)
        .map(|p| {
            format!(
                "{{\"key\":{},\"dtype\":{},\"shape\":{:?},\"values\":{},\"required_values\":{},\"diagnostic_values\":{},\"model_values\":{},\"expected_bits\":{:?},\"policies\":[{}],\"stored_stage\":{},\"source\":{},\"stage_witness\":{}}}",
                key_json(&p.key),
                json_string(&format!("{:?}", p.expected.dtype)),
                p.expected.shape,
                p.expected.len,
                p.policies.iter().filter(|&&x| x != Policy::Observation).count(),
                p.policies.iter().filter(|&&x| x == Policy::Observation).count(),
                p.policies.iter().filter(|x| matches!(x, Policy::ProjectInterval { .. })).count(),
                p.expected.bits(),p.policies.iter().map(|&policy|policy_json(policy)).collect::<Vec<_>>().join(","),json_string(p.stage),p.source,if p.source {String::from("null")} else {stage_witness(fixture(p.key.case_id).expect("planned fixture"),&p.key,p.expected.shape).as_ref().map_or_else(||String::from("null"),stage_witness_json)}
            )
        })
        .collect();
    format!("[{}]", population.join(","))
}
fn operand_bits_json(operand: Option<&WitnessOperand>) -> String {
    operand
        .and_then(|operand| operand.bits.as_ref())
        .map_or_else(|| String::from("null"), |bits| format!("{bits:?}"))
}
fn witness_operand_json(operand: &WitnessOperand) -> String {
    let views = operand
        .views
        .iter()
        .map(|view| match view {
            OperandView::Reshape(shape) => format!("{{\"kind\":\"reshape\",\"shape\":{shape:?}}}"),
            OperandView::Transpose(first, second) => format!("{{\"kind\":\"transpose\",\"first\":{first},\"second\":{second}}}"),
        })
        .collect::<Vec<_>>();
    format!(
        "{{\"origin\":{},\"source\":{},\"dtype\":{},\"shape\":{:?},\"bits\":{},\"views\":[{}],\"producer\":{}}}",
        json_string(operand.origin),
        json_string(operand.source),
        json_string(&format!("{:?}", operand.dtype)),
        operand.shape,
        operand_bits_json(Some(operand)),
        views.join(","),
        operand
            .producer
            .as_ref()
            .map_or_else(|| String::from("null"), |producer| stage_witness_json(producer))
    )
}
fn stage_witness_json(witness: &StageWitness) -> String {
    let operands = witness.operands.iter().map(witness_operand_json).collect::<Vec<_>>();
    let view = witness.source_view.map_or_else(
        || String::from("null"),
        |view| format!("{{\"strides\":{:?},\"base\":{},\"root_shape\":{:?}}}", view.strides, view.base, view.root.shape),
    );
    format!(
        "{{\"operation\":{},\"operands\":[{}],\"axis\":{},\"result_shape\":{:?},\"output_shape\":{:?},\"source_view\":{},\"bits_origin\":\"immutable_fixture_not_device_capture\"}}",
        json_string(witness.operation),
        operands.join(","),
        witness.axis.map_or_else(|| String::from("null"), |axis| format!("{axis}")),
        witness.result_shape,
        witness.output_shape,
        view
    )
}

fn numeric_measurements_json(r: &Report, diagnostic_only: bool) -> String {
    let mut measurements = Vec::new();
    for observation in &r.observations {
        let Some(case) = fixture(observation.key.case_id) else {
            continue;
        };
        if observation.key.output != "expected" || case.numeric.is_empty() {
            continue;
        }
        let policies = case.expectation("expected").expect("numeric destination").policies;
        for (i, ((&actual, cell), &policy)) in observation.values.iter().zip(case.numeric).zip(policies).enumerate() {
            if diagnostic_only && policy != Policy::Observation {
                continue;
            }
            let ideal = rational_approximation(cell.ideal_rational);
            let absolute = numeric_value(observation.dtype, actual).zip(ideal).map(|(a, e)| (a - e).abs());
            let relative = absolute.zip(ideal).and_then(|(a, e)| if e != 0.0 { Some(a / e.abs()) } else { None });
            let ulp = if numeric_value(observation.dtype, actual).is_some() && numeric_value(observation.dtype, cell.ideal_bits).is_some() {
                ordered(observation.dtype, actual)
                    .abs_diff(ordered(observation.dtype, cell.ideal_bits))
                    .to_string()
            } else {
                String::from("null")
            };
            measurements.push(format!(concat!("{{\"key\":{},\"cell\":{},\"name\":{},\"policy\":{},\"required\":{},\"actual_class\":{},\"actual_bits\":{},\"ideal_stored_bits\":{},\"serial_diagnostic_bits\":{},",
                "\"exact_operand_dot\":{},\"sum_abs_products\":{},\"conditional_scratch_interval\":[{},{}],\"approximate_absolute_error\":{},\"approximate_relative_error\":{},\"storage_ULP_distance\":{},\"metrics_are_acceptance\":false}}"),
                key_json(&observation.key),i,json_string(cell.name),json_string(&format!("{policy:?}")),policy!=Policy::Observation,json_string(&format!("{:?}",classification(observation.dtype,actual))),actual,cell.ideal_bits,cell.serial_bits,json_string(cell.ideal_rational),json_string(cell.sum_abs_products),json_string(cell.scratch_low),json_string(cell.scratch_high),optional_number(absolute),optional_number(relative),ulp));
        }
    }
    format!("[{}]", measurements.join(","))
}
pub fn report_json(r: &Report) -> String {
    let failures: Vec<_> = r
        .failures
        .iter()
        .map(|f| format!("{{\"key\":{},\"message\":{}}}", key_json(&f.key), json_string(&f.message)))
        .collect();
    let observations:Vec<_>=r.observations.iter().map(|o|format!("{{\"key\":{},\"dtype\":{},\"shape\":{:?},\"values\":{:?},\"required_checked\":{},\"diagnostic_observed\":{},\"model_checked\":{},\"zero_sign_differences\":{}}}",key_json(&o.key),json_string(&format!("{:?}",o.dtype)),o.shape,o.values,o.required_checked,o.diagnostic_observed,o.model_checked,o.zero_sign_differences)).collect();
    let worst=r.worst.as_ref().map_or_else(||String::from("null"),|w|{
        let lhs=w.stage_witness.operands.first();
        let rhs=w.stage_witness.operands.get(1);
        format!(concat!("{{\"key\":{},\"operation\":{},\"cell\":{},\"stored_stage\":{},\"dtype\":{},\"actual\":{},\"expected\":{},\"storage_ULP_distance\":{},",
            "\"approximate_absolute_error\":{},\"approximate_relative_error\":{},\"exact_ideal_rational\":{},\"sum_abs_products\":{},\"lhs_shape\":{},\"rhs_shape\":{},\"lhs_bits\":{},\"rhs_bits\":{},\"stage_witness\":{},\"metrics_are_acceptance\":false}}"),
            key_json(&w.key),json_string(w.stage_witness.operation),w.cell,json_string(w.stored_stage),json_string(&format!("{:?}",w.dtype)),w.actual,w.expected,w.ulp_distance,
            optional_number(w.absolute_error),optional_number(w.relative_error),json_string(w.ideal_rational),json_string(w.sum_abs_products),
            lhs.map_or_else(||String::from("null"),|operand|format!("{:?}",operand.shape)),rhs.map_or_else(||String::from("null"),|operand|format!("{:?}",operand.shape)),
            operand_bits_json(lhs),operand_bits_json(rhs),stage_witness_json(&w.stage_witness))
    });
    format!(
        concat!(
            "{{\"expected\":{},\"attempted_routes\":{},\"passed_required_routes\":{},\"failed\":{},\"required_checked\":{},\"diagnostic_observed\":{},\"model_checked\":{},",
            "\"checked_required_routes\":{},\"observed_diagnostic_routes\":{},\"unobserved_routes\":{},\"external_unobserved\":{},\"tracking_failures\":{},\"logical_success\":{},\"finalized\":{},",
            "\"failures\":[{}],\"observations\":[{}],\"worst_numeric_witness\":{},\"numeric_measurements\":{},\"diagnostic_measurements\":{},\"full_public_qualification\":false,\"physical_proof\":false}}"
        ),
        summary_json(&r.expected),
        r.attempted,
        r.passed,
        r.failed,
        r.required_checked,
        r.diagnostic_observed,
        r.model_checked,
        r.checked_required_routes,
        r.observed_diagnostic_routes,
        r.unobserved,
        r.external_unobserved,
        r.tracking_failures,
        r.is_success(),
        r.finalized,
        failures.join(","),
        observations.join(","),
        worst,
        numeric_measurements_json(r, false),
        numeric_measurements_json(r, true)
    )
}
pub fn success_footer(options: &Options, scope: Scope, report: &Report) -> Result<String, String> {
    if report.expected != expected_summary(options, scope) || !report.is_success() {
        return Err(format!(
            "mixed matrix logical conformance failed: failed={} unobserved={} external={}",
            report.failed, report.unobserved, report.external_unobserved
        ));
    }
    Ok(format!(
        "BURN_MIXED_MATMUL_LOGICAL_OK version={CORPUS_VERSION} scope={scope:?} rows={} required_checked={} diagnostic_observed={} model_checked={} physical_proof=false full_public_qualification=false conditional_model={MODEL_LABEL}",
        report.expected.selected_rows, report.required_checked, report.diagnostic_observed, report.model_checked
    ))
}
