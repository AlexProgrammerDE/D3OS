//! Exact integer elementwise fixtures, independent of Native arithmetic helpers.
//!
//! Portable signed ordinary arithmetic stays representable. Native extensions
//! cover overflow, invalid division and masked shifts explicitly. The pinned
//! CubeCL power kernel reads both operands using the lhs storage type, so mixed
//! exponent storage is a Native extension. Its inherited unsigned negation also
//! rejects the scalar -1 before dispatch. Both belong only to --extensions;
//! the default corpus is the portable differential domain for all eight formats.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{Backend as BurnBackend, DType, Dispatch, Scalar, TensorData, TensorMetadata, tensor::IntTensor},
    tensor::{Device, Slice},
};

pub const DTYPES: [DType; 8] = [DType::I8, DType::I16, DType::I32, DType::I64, DType::U8, DType::U16, DType::U32, DType::U64];
const BATCH: usize = 8191;
const LENGTHS: [usize; 6] = [0, 1, 127, 128, 129, 257];
const TAILS: [usize; 10] = [0, 1, 2, 3, 4, 5, 127, 128, 129, 257];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    FloorRem,
    Powi,
    Neg,
    Abs,
    Sign,
    And,
    Or,
    Xor,
    Not,
    Shl,
    Shr,
}
impl Operation {
    pub const ALL: [Self; 15] = [
        Self::Add,
        Self::Sub,
        Self::Mul,
        Self::Div,
        Self::FloorRem,
        Self::Powi,
        Self::Neg,
        Self::Abs,
        Self::Sign,
        Self::And,
        Self::Or,
        Self::Xor,
        Self::Not,
        Self::Shl,
        Self::Shr,
    ];
    pub fn unary(self) -> bool {
        matches!(self, Self::Neg | Self::Abs | Self::Sign | Self::Not)
    }
}

pub fn width(dtype: DType) -> u32 {
    assert!(DTYPES.contains(&dtype), "integer format required");
    (dtype.size() * 8) as u32
}
pub fn mask(dtype: DType) -> u64 {
    u64::MAX >> (64 - width(dtype))
}
/// Decode in I128 so abs(I64::MIN) and U64::MAX both remain exact.
pub fn decode(dtype: DType, bits: u64) -> i128 {
    let n = width(dtype);
    let value = i128::from(bits & mask(dtype));
    if dtype.is_int() && value & (1i128 << (n - 1)) != 0 {
        value - (1i128 << n)
    } else {
        value
    }
}
fn encode(dtype: DType, value: i128) -> u64 {
    (value.rem_euclid(1i128 << width(dtype))) as u64
}
/// Signed narrow exponents sign-extend; wide exponents retain their low 32 bits.
pub fn exponent_bits(dtype: DType, bits: u64) -> u32 {
    decode(dtype, bits) as u32
}

/// Left-to-right U128 modular power, exactly 32 exponent steps.
pub fn power_reference(dtype: DType, base: u64, exponent: u32) -> u64 {
    let modulus = 1u128 << width(dtype);
    let base = u128::from(base & mask(dtype));
    let mut result = 1u128;
    for bit in (0..32).rev() {
        result = result * result % modulus;
        if exponent & (1 << bit) != 0 {
            result = result * base % modulus;
        }
    }
    result as u64
}

/// Exact modulo arithmetic plus the explicitly named Native exceptional policy.
pub fn reference(dtype: DType, op: Operation, lhs: u64, rhs_dtype: DType, rhs: u64) -> u64 {
    let (a, b) = (u128::from(lhs & mask(dtype)), u128::from(rhs & mask(dtype)));
    let modulus = 1u128 << width(dtype);
    let (x, y) = (decode(dtype, lhs), decode(dtype, rhs));
    let result = match op {
        Operation::Add => (a + b) % modulus,
        Operation::Sub => (modulus + a - b) % modulus,
        Operation::Mul => (a * b) % modulus,
        Operation::Div if b == 0 => modulus - 1,
        Operation::FloorRem if b == 0 => a,
        Operation::Div => return encode(dtype, x / y),
        Operation::FloorRem => {
            let rem = x % y;
            return encode(dtype, if rem != 0 && (rem < 0) != (y < 0) { rem + y } else { rem });
        }
        Operation::Powi => return power_reference(dtype, lhs, exponent_bits(rhs_dtype, rhs)),
        Operation::Neg => (modulus - a) % modulus,
        Operation::Abs => return encode(dtype, x.abs()),
        Operation::Sign => return encode(dtype, x.signum()),
        Operation::And => a & b,
        Operation::Or => a | b,
        Operation::Xor => a ^ b,
        Operation::Not => (modulus - 1) ^ a,
        Operation::Shl => (a << (rhs & u64::from(width(dtype) - 1))) % modulus,
        Operation::Shr => return encode(dtype, x >> (rhs & u64::from(width(dtype) - 1))),
    };
    result as u64
}

/// Restrict only the ordinary signed operations whose C++ behavior is undefined.
pub fn portable(dtype: DType, op: Operation, lhs: u64, rhs: u64) -> bool {
    let (x, y) = (decode(dtype, lhs), decode(dtype, rhs));
    let min = -(1i128 << (width(dtype) - 1));
    let max = (1i128 << (width(dtype) - 1)) - 1;
    let fits = |v| v >= min && v <= max;
    match op {
        Operation::Div | Operation::FloorRem => y != 0 && !(dtype.is_int() && x == min && y == -1),
        Operation::Shl => y >= 0 && y < i128::from(width(dtype)) && (!dtype.is_int() || (x >= 0 && fits(x << y))),
        Operation::Shr => y >= 0 && y < i128::from(width(dtype)),
        Operation::Add if dtype.is_int() => fits(x + y),
        Operation::Sub if dtype.is_int() => fits(x - y),
        Operation::Mul if dtype.is_int() => fits(x * y),
        Operation::Neg if dtype.is_uint() => false,
        Operation::Neg | Operation::Abs if dtype.is_int() => x != min,
        _ => true,
    }
}

/// Check scalar conversion independently; never convert integer scalars to float.
pub fn normalize_scalar(dtype: DType, scalar: Scalar) -> Result<u64, String> {
    let value = match scalar {
        Scalar::Int(v) => i128::from(v),
        Scalar::UInt(v) => i128::from(v),
        Scalar::Bool(v) => i128::from(v),
        Scalar::Float(v) => {
            // Fractional fixtures are small and exact in F64. Wide fixtures use
            // Scalar::Int/UInt so their payload never enters floating arithmetic.
            if !v.is_finite() || !(-2147483649.0..2147483648.0).contains(&v) {
                return Err(String::from("float scalar outside checked fixture range"));
            }
            v as i128
        }
    };
    let min = if dtype.is_int() { -(1i128 << (width(dtype) - 1)) } else { 0 };
    let max = if dtype.is_int() {
        (1i128 << (width(dtype) - 1)) - 1
    } else {
        i128::from(mask(dtype))
    };
    if value < min || value > max {
        return Err(format!("scalar {scalar:?} outside {dtype:?}"));
    }
    Ok(encode(dtype, value))
}
/// The public shortcut runs before normalization at the base format.
pub fn scalar_exponent(dtype: DType, scalar: Scalar) -> Result<u32, String> {
    let probe = decode(DType::I32, normalize_scalar(DType::I32, scalar)?) as i32;
    if matches!(probe, 0..=2) {
        return Ok(probe as u32);
    }
    Ok(exponent_bits(dtype, normalize_scalar(dtype, scalar)?))
}

pub fn pack(dtype: DType, values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()[..dtype.size()].to_vec()).collect()
}
pub fn unpack(dtype: DType, bytes: &[u8]) -> Vec<u64> {
    assert_eq!(bytes.len() % dtype.size(), 0);
    bytes
        .chunks_exact(dtype.size())
        .map(|v| {
            let mut word = [0; 8];
            word[..v.len()].copy_from_slice(v);
            u64::from_le_bytes(word)
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Values {
    pub shape: Vec<usize>,
    pub values: Vec<u64>,
}
fn coordinates(shape: &[usize], mut linear: usize) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coords[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    coords
}
fn offset(shape: &[usize], coords: &[usize]) -> usize {
    assert_eq!(shape.len(), coords.len());
    shape.iter().zip(coords).fold(0, |n, (&size, &i)| {
        assert!(i < size);
        n * size + i
    })
}
impl Values {
    pub fn new(shape: &[usize], values: Vec<u64>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self { shape: shape.to_vec(), values }
    }
    fn mapped(&self, shape: &[usize], map: impl Fn(&[usize]) -> Vec<usize>) -> Self {
        Self::new(
            shape,
            (0..shape.iter().product())
                .map(|i| self.values[offset(&self.shape, &map(&coordinates(shape, i)))])
                .collect(),
        )
    }
    pub fn slice(&self, starts: &[usize], steps: &[usize], shape: &[usize]) -> Self {
        assert_eq!(self.shape.len(), starts.len());
        assert_eq!(starts.len(), steps.len());
        self.mapped(shape, |c| c.iter().enumerate().map(|(axis, &i)| starts[axis] + steps[axis] * i).collect())
    }
    pub fn permute(&self, axes: &[usize]) -> Self {
        let shape: Vec<_> = axes.iter().map(|&a| self.shape[a]).collect();
        self.mapped(&shape, |c| {
            let mut original = vec![0; axes.len()];
            for (&axis, &i) in axes.iter().zip(c) {
                original[axis] = i;
            }
            original
        })
    }
    pub fn flip(&self, axes: &[usize]) -> Self {
        self.mapped(&self.shape, |c| {
            c.iter()
                .enumerate()
                .map(|(axis, &i)| if axes.contains(&axis) { self.shape[axis] - 1 - i } else { i })
                .collect()
        })
    }
    fn broadcast_at(&self, output_shape: &[usize], linear: usize) -> u64 {
        let coords = coordinates(output_shape, linear);
        let skip = output_shape.len() - self.shape.len();
        let own: Vec<_> = self
            .shape
            .iter()
            .enumerate()
            .map(|(axis, &size)| if size == 1 { 0 } else { coords[skip + axis] })
            .collect();
        self.values[offset(&self.shape, &own)]
    }
}
pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Vec<usize> {
    let rank = lhs.len().max(rhs.len());
    (0..rank)
        .map(|axis| {
            let a = if axis + lhs.len() >= rank { lhs[axis + lhs.len() - rank] } else { 1 };
            let b = if axis + rhs.len() >= rank { rhs[axis + rhs.len() - rank] } else { 1 };
            assert!(a == b || a == 1 || b == 1, "incompatible broadcast");
            if a == 1 { b } else { a }
        })
        .collect()
}

#[derive(Clone, Debug)]
enum View {
    Slice { starts: Vec<usize>, steps: Vec<usize>, shape: Vec<usize> },
    Permute(Vec<usize>),
    Flip(Vec<usize>),
}
#[derive(Clone, Debug)]
struct Input {
    dtype: DType,
    root: Values,
    views: Vec<View>,
}
impl Input {
    fn flat(dtype: DType, values: Vec<u64>) -> Self {
        Self {
            dtype,
            root: Values::new(&[values.len()], values),
            views: vec![],
        }
    }
    fn logical(&self) -> Values {
        self.views.iter().fold(self.root.clone(), |data, view| match view {
            View::Slice { starts, steps, shape } => data.slice(starts, steps, shape),
            View::Permute(axes) => data.permute(axes),
            View::Flip(axes) => data.flip(axes),
        })
    }
    fn build<B: BurnBackend>(&self, device: &B::Device) -> (IntTensor<B>, IntTensor<B>) {
        let source = B::int_from_data(
            TensorData::from_bytes_vec(pack(self.dtype, &self.root.values), self.root.shape.clone(), self.dtype),
            device,
        );
        let mut tensor = source.clone();
        for view in &self.views {
            tensor = match view {
                View::Slice { starts, steps, shape } => {
                    let slices: Vec<_> = starts
                        .iter()
                        .zip(steps)
                        .zip(shape)
                        .map(|((&start, &step), &length)| Slice::new(start as isize, Some((start + step * length) as isize), step as isize))
                        .collect();
                    B::int_slice(tensor, &slices)
                }
                View::Permute(axes) => B::int_permute(tensor, axes),
                View::Flip(axes) => B::int_flip(tensor, axes),
            };
        }
        (source, tensor)
    }
}
#[derive(Clone, Debug)]
enum Right {
    Tensor(Input),
    Scalar(Scalar),
    None,
}
#[derive(Clone, Debug)]
struct Case {
    label: &'static str,
    op: Operation,
    lhs: Input,
    rhs: Right,
    extension: bool,
}
impl Case {
    fn expected(&self) -> Values {
        let lhs = self.lhs.logical();
        match &self.rhs {
            Right::Tensor(rhs) => {
                let right = rhs.logical();
                let shape = broadcast_shape(&lhs.shape, &right.shape);
                let values = (0..shape.iter().product())
                    .map(|i| reference(self.lhs.dtype, self.op, lhs.broadcast_at(&shape, i), rhs.dtype, right.broadcast_at(&shape, i)))
                    .collect();
                Values::new(&shape, values)
            }
            Right::Scalar(scalar) => {
                let exponent = if self.op == Operation::Powi {
                    Some(scalar_exponent(self.lhs.dtype, *scalar).expect("valid scalar fixture"))
                } else {
                    None
                };
                let rhs = if exponent.is_none() {
                    normalize_scalar(self.lhs.dtype, *scalar).expect("valid scalar fixture")
                } else {
                    0
                };
                Values::new(
                    &lhs.shape,
                    lhs.values
                        .iter()
                        .map(|&x| match exponent {
                            Some(exp) => power_reference(self.lhs.dtype, x, exp),
                            None => reference(self.lhs.dtype, self.op, x, self.lhs.dtype, rhs),
                        })
                        .collect(),
                )
            }
            Right::None => Values::new(
                &lhs.shape,
                lhs.values.iter().map(|&x| reference(self.lhs.dtype, self.op, x, self.lhs.dtype, 0)).collect(),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub extensions: bool,
    pub dtype: Option<DType>,
    pub operation: Option<Operation>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub dtypes: usize,
    pub operations: [usize; 15],
    pub tensor_cases: usize,
    pub scalar_cases: usize,
    pub unary_cases: usize,
    pub extension_cases: usize,
    pub mixed_exponent_cases: usize,
    pub source_checks: usize,
    pub guard_checks: usize,
    pub values: usize,
    pub exhaustive_pairs: usize,
    pub exhaustive_unary_values: usize,
    pub unsigned_neg_cases: usize,
}
impl Summary {
    pub fn cases(&self) -> usize {
        self.tensor_cases + self.scalar_cases + self.unary_cases
    }
    fn record(&mut self, case: &Case) {
        self.operations[case.op as usize] += 1;
        self.extension_cases += usize::from(case.extension);
        self.unsigned_neg_cases += usize::from(case.op == Operation::Neg && case.lhs.dtype.is_uint());
        self.source_checks += 1;
        self.guard_checks += 1;
        match &case.rhs {
            Right::Tensor(rhs) => {
                self.tensor_cases += 1;
                self.source_checks += 1;
                self.mixed_exponent_cases += usize::from(case.op == Operation::Powi && rhs.dtype != case.lhs.dtype);
            }
            Right::Scalar(_) => self.scalar_cases += 1,
            Right::None => self.unary_cases += 1,
        }
        let lhs = case.lhs.logical();
        let n = if let Right::Tensor(rhs) = &case.rhs {
            broadcast_shape(&lhs.shape, &rhs.logical().shape).iter().product()
        } else {
            lhs.values.len()
        };
        self.values += n;
        if case.label == "exhaustive pairs" {
            self.exhaustive_pairs += n;
        }
        if case.label == "exhaustive unary" {
            self.exhaustive_unary_values += n;
        }
    }
}

fn edges(dtype: DType) -> Vec<u64> {
    let n = width(dtype);
    let max = mask(dtype);
    let sign = 1u64 << (n - 1);
    let mut values = vec![
        0,
        1,
        2,
        3,
        7,
        max,
        max - 1,
        max - 2,
        sign,
        sign + 1,
        sign - 1,
        sign - 2,
        0xaaaa_aaaa_aaaa_aaaa & max,
        0x5555_5555_5555_5555 & max,
    ];
    for bit in [4, 5, 6, 7, 8, 15, 16, 24, 31, 32, 53, 63] {
        if bit < n {
            let v = 1u64 << bit;
            values.extend([v - 1, v, (v + 1) & max]);
        }
    }
    if n == 64 {
        values.extend([0xffff_ffff, 0x1_0000_0001, 0xffff_ffff_0000_0001, 0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]);
    }
    if dtype.is_int() {
        for value in [-7, -257, -65537, -16_777_217, -(1i128 << 53) - 1] {
            if value >= -(1i128 << (n - 1)) {
                values.push(encode(dtype, value));
            }
        }
    }
    values.sort_unstable();
    values.dedup();
    values
}
fn scalar_value(dtype: DType, bits: u64) -> Scalar {
    if dtype.is_int() {
        Scalar::Int(decode(dtype, bits) as i64)
    } else {
        Scalar::UInt(bits & mask(dtype))
    }
}
fn scalars(dtype: DType, op: Operation) -> Vec<Scalar> {
    let mut values: Vec<_> = if matches!(op, Operation::Shl | Operation::Shr) {
        [0, 1, width(dtype) / 2 - 1, width(dtype) / 2, width(dtype) - 1]
            .into_iter()
            .map(|v| Scalar::UInt(u64::from(v)))
            .collect()
    } else {
        edges(dtype)
            .into_iter()
            .map(|v| scalar_value(dtype, v))
            .filter(|v| op != Operation::Powi || scalar_exponent(dtype, *v).is_ok())
            .collect()
    };
    if op == Operation::Powi {
        // These exercise the public I32 shortcut before base normalization.
        values.extend([
            Scalar::Float(-0.75),
            Scalar::Float(0.75),
            Scalar::Float(1.75),
            Scalar::Float(2.75),
            Scalar::Float(3.75),
        ]);
        if normalize_scalar(dtype, Scalar::UInt(200)).is_ok() {
            values.push(Scalar::UInt(200));
        }
    } else if !matches!(op, Operation::Shl | Operation::Shr) {
        values.push(Scalar::Float(3.75));
        if dtype.is_int() {
            values.push(Scalar::Float(-3.75));
        }
    }
    values
}
fn push_pairs(cases: &mut Vec<Case>, dtype: DType, rhs_dtype: DType, op: Operation, pairs: &[(u64, u64)], label: &'static str, extension: bool) {
    for batch in pairs.chunks(BATCH) {
        cases.push(Case {
            label,
            op,
            lhs: Input::flat(dtype, batch.iter().map(|p| p.0).collect()),
            rhs: Right::Tensor(Input::flat(rhs_dtype, batch.iter().map(|p| p.1).collect())),
            extension,
        });
    }
}
fn push_unary(cases: &mut Vec<Case>, dtype: DType, op: Operation, values: &[u64], label: &'static str, extension: bool) {
    for batch in values.chunks(BATCH) {
        cases.push(Case {
            label,
            op,
            lhs: Input::flat(dtype, batch.to_vec()),
            rhs: Right::None,
            extension,
        });
    }
}
fn safe_geometry_pair(dtype: DType, op: Operation, i: usize) -> (u64, u64) {
    let n = width(dtype);
    let values = edges(dtype);
    let a = values[i % values.len()];
    let b = values[(i * 7 + 3) % values.len()];
    match op {
        Operation::Add | Operation::Sub if dtype.is_int() => (encode(dtype, i as i128 % 31 - 15), encode(dtype, i as i128 % 29 - 14)),
        Operation::Mul if dtype.is_int() => (encode(dtype, i as i128 % 15 - 7), encode(dtype, i as i128 % 13 - 6)),
        Operation::Div | Operation::FloorRem => (
            a,
            if dtype.is_int() {
                encode(dtype, if i.is_multiple_of(2) { 3 } else { -3 })
            } else {
                3
            },
        ),
        Operation::Shl => (
            if dtype.is_int() { (i % 2) as u64 } else { a },
            (i as u64) % u64::from(if dtype.is_int() { n - 1 } else { n }),
        ),
        Operation::Shr => (a, (i as u64) % u64::from(n)),
        Operation::Neg | Operation::Abs if dtype.is_int() && decode(dtype, a) == -(1i128 << (n - 1)) => (a + 1, b),
        _ => (a, b),
    }
}
/// Offset, stepped, transposed and reversed input with poisoned unused elements.
fn strided_input(dtype: DType, values: &[u64], length: usize) -> Input {
    assert_eq!(values.len(), 3 * length);
    let pitch = 2 * length + 3;
    let mut root = vec![0xa5c3_7e19_d68b_f042 & mask(dtype); 3 * pitch];
    for row in 0..3 {
        for col in 0..length {
            root[row * pitch + 1 + 2 * col] = values[row * length + col];
        }
    }
    Input {
        dtype,
        root: Values::new(&[3, pitch], root),
        views: vec![
            View::Slice {
                starts: vec![0, 1],
                steps: vec![1, 2],
                shape: vec![3, length],
            },
            View::Permute(vec![1, 0]),
            View::Flip(vec![0]),
        ],
    }
}
fn shaped(dtype: DType, shape: &[usize], values: Vec<u64>) -> Input {
    Input {
        dtype,
        root: Values::new(shape, values),
        views: vec![],
    }
}
fn geometry(cases: &mut Vec<Case>, dtype: DType, op: Operation) {
    for length in TAILS {
        let pairs: Vec<_> = (0..length).map(|i| safe_geometry_pair(dtype, op, i)).collect();
        cases.push(Case {
            label: "linear packed tail",
            op,
            lhs: Input::flat(dtype, pairs.iter().map(|p| p.0).collect()),
            rhs: if op.unary() {
                Right::None
            } else {
                Right::Tensor(Input::flat(dtype, pairs.iter().map(|p| p.1).collect()))
            },
            extension: false,
        });
    }
    for length in LENGTHS {
        let pairs: Vec<_> = (0..3 * length).map(|i| safe_geometry_pair(dtype, op, i)).collect();
        let left = strided_input(dtype, &pairs.iter().map(|p| p.0).collect::<Vec<_>>(), length);
        if op.unary() {
            cases.push(Case {
                label: "strided unary tails",
                op,
                lhs: left,
                rhs: Right::None,
                extension: false,
            });
        } else {
            let right = shaped(dtype, &[1, 3], (0..3).map(|i| safe_geometry_pair(dtype, op, i + 11).1).collect());
            cases.push(Case {
                label: "broadcast rhs",
                op,
                lhs: left,
                rhs: Right::Tensor(right),
                extension: false,
            });
            let left = shaped(dtype, &[1, 3], (0..3).map(|i| safe_geometry_pair(dtype, op, i + 7).0).collect());
            let right = strided_input(dtype, &pairs.iter().map(|p| p.1).collect::<Vec<_>>(), length);
            cases.push(Case {
                label: "broadcast lhs",
                op,
                lhs: left,
                rhs: Right::Tensor(right),
                extension: false,
            });
            let scalar = if matches!(op, Operation::Div | Operation::FloorRem) {
                Scalar::Int(3)
            } else {
                Scalar::Int(1)
            };
            let lhs = strided_input(dtype, &pairs.iter().map(|p| p.0).collect::<Vec<_>>(), length);
            // Geometry samples are safe for the chosen scalar, except signed
            // add/sub at the full-width edges (their samples are bounded above).
            cases.push(Case {
                label: "strided scalar tails",
                op,
                lhs,
                rhs: Right::Scalar(scalar),
                extension: false,
            });
        }
    }
    for shape in [vec![], vec![1, 1, 1, 1, 1, 1, 1, 1]] {
        let (a, b) = safe_geometry_pair(dtype, op, 19);
        cases.push(Case {
            label: if shape.is_empty() { "rank zero" } else { "rank eight" },
            op,
            lhs: shaped(dtype, &shape, vec![a]),
            rhs: if op.unary() {
                Right::None
            } else {
                Right::Tensor(shaped(dtype, &shape, vec![b]))
            },
            extension: false,
        });
    }
    // Rank eight with a genuine axis permutation and negative stride.
    let shape = [1, 1, 1, 1, 1, 1, 3, 129];
    let mut input = shaped(dtype, &shape, (0..387).map(|i| safe_geometry_pair(dtype, op, i).0).collect());
    input.views = vec![View::Permute(vec![0, 1, 2, 3, 4, 5, 7, 6]), View::Flip(vec![6, 7])];
    cases.push(Case {
        label: "rank eight view",
        op,
        lhs: input,
        rhs: if op.unary() { Right::None } else { Right::Scalar(Scalar::Int(1)) },
        extension: false,
    });
}
fn extension_cases(cases: &mut Vec<Case>, dtype: DType, op: Operation) {
    let edge = edges(dtype);
    if op.unary() {
        let values: Vec<_> = edge.into_iter().filter(|&a| !portable(dtype, op, a, 0)).collect();
        push_unary(cases, dtype, op, &values, "Native exceptional unary", true);
        return;
    }
    if op == Operation::Powi && dtype.is_int() {
        let values: Vec<_> = edge.iter().copied().filter(|&a| !portable(dtype, Operation::Mul, a, a)).collect();
        for scalar in [Scalar::Int(2), Scalar::Float(2.75)] {
            cases.push(Case {
                label: "Native public square overflow",
                op,
                lhs: Input::flat(dtype, values.clone()),
                rhs: Right::Scalar(scalar),
                extension: true,
            });
        }
    }
    let mut pairs: Vec<_> = edge
        .iter()
        .flat_map(|&a| edge.iter().map(move |&b| (a, b)))
        .filter(|&(a, b)| !portable(dtype, op, a, b))
        .collect();
    if matches!(op, Operation::Shl | Operation::Shr) {
        let n = u64::from(width(dtype));
        pairs.extend(
            edge.iter()
                .flat_map(|&a| [n, n + 1, 2 * n - 1, mask(dtype)].into_iter().map(move |b| (a, b & mask(dtype)))),
        );
    }
    push_pairs(cases, dtype, dtype, op, &pairs, "Native exceptional pairs", true);
    // Scalar exceptional cases still obey checked scalar conversion. Wide
    // out-of-I32 power scalars belong to pre-dispatch rejection tests elsewhere.
    let mut scalar_values = scalars(dtype, op);
    if matches!(op, Operation::Shl | Operation::Shr) {
        scalar_values.extend([Scalar::UInt(u64::from(width(dtype))), Scalar::UInt(u64::from(width(dtype)) + 1)]);
        scalar_values.push(scalar_value(dtype, mask(dtype)));
        if normalize_scalar(dtype, Scalar::UInt(200)).is_ok() {
            scalar_values.push(Scalar::UInt(200));
        }
    }
    for scalar in scalar_values {
        if op == Operation::Powi {
            continue;
        }
        let Ok(rhs) = normalize_scalar(dtype, scalar) else { continue };
        let values: Vec<_> = edges(dtype).into_iter().filter(|&a| !portable(dtype, op, a, rhs)).collect();
        if !values.is_empty() {
            cases.push(Case {
                label: "Native exceptional scalar",
                op,
                lhs: Input::flat(dtype, values),
                rhs: Right::Scalar(scalar),
                extension: true,
            });
        }
    }
}
fn cases(dtype: DType, options: &Options) -> Vec<Case> {
    let mut result = Vec::new();
    for op in Operation::ALL {
        let native_only = op == Operation::Neg && dtype.is_uint();
        if options.operation.is_some_and(|only| only != op) || (native_only && !options.extensions) {
            continue;
        }
        let first_case = result.len();
        if op.unary() {
            let exhaustive = width(dtype) <= 16;
            let values: Vec<_> = if exhaustive { (0..=mask(dtype)).collect() } else { edges(dtype) };
            let values: Vec<_> = values.into_iter().filter(|&a| native_only || portable(dtype, op, a, 0)).collect();
            push_unary(
                &mut result,
                dtype,
                op,
                &values,
                if exhaustive { "exhaustive unary" } else { "unary edges" },
                false,
            );
        } else {
            let exhaustive = width(dtype) == 8;
            let values: Vec<_> = if exhaustive { (0..=255).collect() } else { edges(dtype) };
            let pairs: Vec<_> = values
                .iter()
                .flat_map(|&a| values.iter().map(move |&b| (a, b)))
                .filter(|&(a, b)| portable(dtype, op, a, b))
                .collect();
            push_pairs(
                &mut result,
                dtype,
                dtype,
                op,
                &pairs,
                if exhaustive { "exhaustive pairs" } else { "binary edges" },
                false,
            );
            for scalar in scalars(dtype, op) {
                let exponent = if op == Operation::Powi {
                    Some(scalar_exponent(dtype, scalar).expect("valid exponent"))
                } else {
                    None
                };
                let rhs = if exponent.is_none() {
                    normalize_scalar(dtype, scalar).expect("valid scalar")
                } else {
                    0
                };
                let values: Vec<_> = values
                    .iter()
                    .copied()
                    .filter(|&a| {
                        // Public power exponent two uses ordinary multiplication.
                        if exponent == Some(2) {
                            portable(dtype, Operation::Mul, a, a)
                        } else {
                            exponent.is_some() || portable(dtype, op, a, rhs)
                        }
                    })
                    .collect();
                if !values.is_empty() {
                    result.push(Case {
                        label: "scalar edges",
                        op,
                        lhs: Input::flat(dtype, values),
                        rhs: Right::Scalar(scalar),
                        extension: false,
                    });
                }
            }
            if op == Operation::Powi && options.extensions {
                for rhs_dtype in DTYPES {
                    if rhs_dtype == dtype {
                        continue;
                    }
                    let powers = edges(rhs_dtype);
                    let bases = edges(dtype);
                    let pairs: Vec<_> = bases.iter().flat_map(|&a| powers.iter().map(move |&b| (a, b))).collect();
                    push_pairs(&mut result, dtype, rhs_dtype, op, &pairs, "Native actual mixed exponent dtype", true);
                    for length in [1, 129] {
                        let left = strided_input(dtype, &(0..3 * length).map(|i| bases[i % bases.len()]).collect::<Vec<_>>(), length);
                        let mut right = shaped(rhs_dtype, &[1, 3], vec![powers[0], mask(rhs_dtype), powers[powers.len() / 2]]);
                        right.views.push(View::Flip(vec![1]));
                        result.push(Case {
                            label: "Native mixed exponent broadcast rhs",
                            op,
                            lhs: left,
                            rhs: Right::Tensor(right),
                            extension: true,
                        });
                        let left = shaped(dtype, &[1, 3], vec![1, 3, mask(dtype)]);
                        let right = strided_input(rhs_dtype, &(0..3 * length).map(|i| powers[i % powers.len()]).collect::<Vec<_>>(), length);
                        result.push(Case {
                            label: "Native mixed exponent broadcast lhs",
                            op,
                            lhs: left,
                            rhs: Right::Tensor(right),
                            extension: true,
                        });
                    }
                }
            }
        }
        geometry(&mut result, dtype, op);
        if native_only {
            for case in &mut result[first_case..] {
                case.extension = true;
            }
        } else if options.extensions {
            extension_cases(&mut result, dtype, op);
        }
    }
    result
}
fn selected(dtype: DType, options: &Options) -> bool {
    options.dtype.is_none_or(|only| only == dtype)
}
fn initialize_summary(options: &Options) -> Summary {
    Summary {
        dtypes: DTYPES.iter().filter(|&&dtype| selected(dtype, options)).count(),
        ..Summary::default()
    }
}
/// Enumerate the same generated cases used by the runner, without a device.
pub fn expected_summary(options: &Options) -> Summary {
    let mut summary = initialize_summary(options);
    for dtype in DTYPES {
        if selected(dtype, options) {
            for case in cases(dtype, options) {
                summary.record(&case);
            }
        }
    }
    summary
}

fn apply_tensor<B: BurnBackend>(op: Operation, lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
    match op {
        Operation::Add => B::int_add(lhs, rhs),
        Operation::Sub => B::int_sub(lhs, rhs),
        Operation::Mul => B::int_mul(lhs, rhs),
        Operation::Div => B::int_div(lhs, rhs),
        Operation::FloorRem => B::int_remainder(lhs, rhs),
        Operation::Powi => B::int_powi(lhs, rhs),
        Operation::And => B::bitwise_and(lhs, rhs),
        Operation::Or => B::bitwise_or(lhs, rhs),
        Operation::Xor => B::bitwise_xor(lhs, rhs),
        Operation::Shl => B::bitwise_left_shift(lhs, rhs),
        Operation::Shr => B::bitwise_right_shift(lhs, rhs),
        _ => unreachable!(),
    }
}
fn apply_scalar<B: BurnBackend>(op: Operation, lhs: IntTensor<B>, rhs: Scalar) -> IntTensor<B> {
    match op {
        Operation::Add => B::int_add_scalar(lhs, rhs),
        Operation::Sub => B::int_sub_scalar(lhs, rhs),
        Operation::Mul => B::int_mul_scalar(lhs, rhs),
        Operation::Div => B::int_div_scalar(lhs, rhs),
        Operation::FloorRem => B::int_remainder_scalar(lhs, rhs),
        Operation::Powi => B::int_powi_scalar(lhs, rhs),
        Operation::And => B::bitwise_and_scalar(lhs, rhs),
        Operation::Or => B::bitwise_or_scalar(lhs, rhs),
        Operation::Xor => B::bitwise_xor_scalar(lhs, rhs),
        Operation::Shl => B::bitwise_left_shift_scalar(lhs, rhs),
        Operation::Shr => B::bitwise_right_shift_scalar(lhs, rhs),
        _ => unreachable!(),
    }
}
async fn check<B: BurnBackend>(tensor: IntTensor<B>, dtype: DType, expected: &Values) -> Result<(), String> {
    if tensor.dtype() != dtype || tensor.shape().as_slice() != expected.shape {
        return Err(format!(
            "metadata {:?} {:?}, expected {dtype:?} {:?}",
            tensor.dtype(),
            tensor.shape(),
            expected.shape
        ));
    }
    let data = B::int_into_data(tensor).await.map_err(|e| format!("read: {e:?}"))?;
    let bytes = pack(dtype, &expected.values);
    if data.dtype() != dtype || data.shape().as_slice() != expected.shape || data.as_bytes().len() != bytes.len() {
        return Err(format!(
            "storage metadata {:?} {:?} bytes={}, expected {dtype:?} {:?} bytes={}",
            data.dtype(),
            data.shape(),
            data.as_bytes().len(),
            expected.shape,
            bytes.len()
        ));
    }
    if let Some(at) = data.as_bytes().iter().zip(&bytes).position(|(a, b)| a != b) {
        let index = at / dtype.size();
        return Err(format!(
            "element={index} actual={:#x} expected={:#x}",
            unpack(dtype, data.as_bytes())[index],
            expected.values[index]
        ));
    }
    Ok(())
}
async fn run_case<B: BurnBackend>(device: &B::Device, case: &Case, summary: &mut Summary) -> Result<(), String> {
    let dtype = case.lhs.dtype;
    let expected = case.expected();
    let (source, lhs) = case.lhs.build::<B>(device);
    // Public APIs expose logical bytes, not unused packed slots or allocation
    // guards. Retain a separate poisoned allocation and every source alias.
    let poison = Input::flat(
        dtype,
        (0..expected.values.len() + 9)
            .map(|i| {
                if i.is_multiple_of(2) {
                    mask(dtype)
                } else {
                    0xa53c_d791_6e28_b40f & mask(dtype)
                }
            })
            .collect(),
    );
    let (guard, _) = poison.build::<B>(device);
    let mut retained_rhs = None;
    let output = match &case.rhs {
        Right::Tensor(rhs) => {
            let (source, tensor) = rhs.build::<B>(device);
            retained_rhs = Some((source, rhs));
            apply_tensor::<B>(case.op, lhs, tensor)
        }
        Right::Scalar(scalar) => apply_scalar::<B>(case.op, lhs, *scalar),
        Right::None => match case.op {
            Operation::Neg => B::int_neg(lhs),
            Operation::Abs => B::int_abs(lhs),
            Operation::Sign => B::int_sign(lhs),
            Operation::Not => B::bitwise_not(lhs),
            _ => unreachable!(),
        },
    };
    check::<B>(output, dtype, &expected).await?;
    check::<B>(source, dtype, &case.lhs.root).await.map_err(|e| format!("lhs preservation: {e}"))?;
    if let Some((source, rhs)) = retained_rhs {
        check::<B>(source, rhs.dtype, &rhs.root).await.map_err(|e| format!("rhs preservation: {e}"))?;
    }
    check::<B>(guard, dtype, &poison.root).await.map_err(|e| format!("poison preservation: {e}"))?;
    summary.record(case);
    Ok(())
}
/// Run public backend operations; host arithmetic is used only as an oracle.
pub async fn run_backend<B: BurnBackend>(device: &B::Device, options: &Options) -> Result<Summary, String> {
    let mut summary = initialize_summary(options);
    for dtype in DTYPES {
        if !selected(dtype, options) {
            continue;
        }
        for case in cases(dtype, options) {
            run_case::<B>(device, &case, &mut summary).await.map_err(|e| {
                format!(
                    "{dtype:?} {:?} {} extension={} rhs={:?}: {e}",
                    case.op,
                    case.label,
                    case.extension,
                    match &case.rhs {
                        Right::Tensor(t) => Some(t.dtype),
                        _ => None,
                    }
                )
            })?;
        }
    }
    let expected = expected_summary(options);
    if summary != expected {
        return Err(format!("corpus totals {summary:?}, expected {expected:?}"));
    }
    Ok(summary)
}
pub async fn run(device: &Device, options: &Options) -> Result<Summary, String> {
    run_backend::<Dispatch>(device.as_dispatch(), options).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_portable_cases_obey_ordinary_arithmetic_and_scalar_domains() {
        for dtype in DTYPES {
            for case in cases(dtype, &Options::default()) {
                let lhs = case.lhs.logical();
                match &case.rhs {
                    Right::Tensor(input) => {
                        assert!(input.dtype == dtype || case.op == Operation::Powi);
                        let rhs = input.logical();
                        let shape = broadcast_shape(&lhs.shape, &rhs.shape);
                        for i in 0..shape.iter().product() {
                            let a = lhs.broadcast_at(&shape, i);
                            let b = rhs.broadcast_at(&shape, i);
                            assert!(portable(dtype, case.op, a, b), "{dtype:?} {:?} {} a={a:#x} b={b:#x}", case.op, case.label);
                        }
                    }
                    Right::Scalar(scalar) => {
                        if case.op == Operation::Powi {
                            let exp = scalar_exponent(dtype, *scalar).unwrap();
                            if exp == 2 {
                                assert!(lhs.values.iter().all(|&a| portable(dtype, Operation::Mul, a, a)));
                            }
                        } else {
                            let rhs = normalize_scalar(dtype, *scalar).unwrap();
                            assert!(
                                lhs.values.iter().all(|&a| portable(dtype, case.op, a, rhs)),
                                "{dtype:?} {:?} {} scalar={scalar:?}",
                                case.op,
                                case.label
                            );
                        }
                    }
                    Right::None => assert!(lhs.values.iter().all(|&a| portable(dtype, case.op, a, 0))),
                }
            }
        }
    }

    #[test]
    fn mixed_exponent_cases_retain_every_rhs_format_and_full_base_bits() {
        for dtype in DTYPES {
            let options = Options {
                extensions: true,
                operation: Some(Operation::Powi),
                ..Options::default()
            };
            let corpus = cases(dtype, &options);
            for rhs_dtype in DTYPES {
                assert!(corpus.iter().any(|case| matches!(&case.rhs, Right::Tensor(input) if input.dtype == rhs_dtype)));
            }
            if width(dtype) == 64 {
                assert!(corpus.iter().any(|case| case.lhs.root.values.contains(&((1 << 53) + 1))));
            }
            let restricted = cases(dtype, &Options { extensions: false, ..options });
            assert!(restricted.iter().all(|case| !matches!(&case.rhs, Right::Tensor(input) if input.dtype != dtype)));
        }
    }

    #[test]
    fn summary_uses_generated_counts_and_explicit_scope_restrictions() {
        let full = expected_summary(&Options::default());
        let unsigned_neg = expected_summary(&Options {
            operation: Some(Operation::Neg),
            ..Options::default()
        });
        let extended = expected_summary(&Options {
            extensions: true,
            ..Options::default()
        });
        assert_eq!(full.dtypes, 8);
        assert!(full.operations.iter().all(|&n| n > 0));
        assert_eq!(full.extension_cases, 0);
        assert_eq!(full.cases(), full.operations.iter().sum::<usize>());
        assert_eq!(full.source_checks, full.cases() + full.tensor_cases);
        assert_eq!(full.guard_checks, full.cases());
        assert!(full.exhaustive_pairs > 500_000);
        assert_eq!(full.exhaustive_unary_values, 460_540);
        assert_eq!(full.mixed_exponent_cases, 0);
        assert_eq!(full.unsigned_neg_cases, 0);
        assert_eq!(unsigned_neg.unsigned_neg_cases, 0);
        assert_eq!(extended.mixed_exponent_cases, 56 * 5);
        assert!(extended.unsigned_neg_cases > 0);
        assert_eq!(extended.cases() - full.cases(), extended.extension_cases);
        assert!(extended.values > full.values);
    }
}
