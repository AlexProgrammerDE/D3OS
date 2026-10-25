//! Exact stored-value movement through the public Dispatch backend.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{
        BoolDType, DType, Dispatch, IntDType, Scalar, TensorData, TensorMetadata,
        ops::{BoolTensorOps, FloatTensorOps, IntTensorOps, PadMode},
        tensor::{BoolTensor, FloatTensor, IntTensor},
    },
    tensor::{Device, Slice, read_sync},
};

pub const DTYPES: [DType; 15] = [
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
    DType::Bool(BoolDType::U8),
    DType::Bool(BoolDType::U32),
];
const LENGTHS: [usize; 6] = [0, 1, 127, 128, 129, 257];
const MASK_DTYPES: [BoolDType; 2] = [BoolDType::U8, BoolDType::U32];

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub include_flex32: bool,
    /// Isolate the pinned Linux empty Boolean scatter division failure.
    pub skip_empty_bool_scatter: bool,
    /// Isolate mixed-width Boolean-mask reinterpretation in pinned Linux.
    pub matching_bool_masks_only: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            include_flex32: true,
            skip_empty_bool_scatter: false,
            matching_bool_masks_only: false,
        }
    }
}

/// A host tensor stores opaque element records, never decoded numeric values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Records {
    pub shape: Vec<usize>,
    width: usize,
    elements: Vec<Vec<u8>>,
}
fn coordinates(shape: &[usize]) -> Vec<Vec<usize>> {
    let mut rows = vec![Vec::new()];
    for &size in shape {
        rows = rows
            .into_iter()
            .flat_map(|row| {
                (0..size).map(move |i| {
                    let mut next = row.clone();
                    next.push(i);
                    next
                })
            })
            .collect();
    }
    rows
}
impl Records {
    pub fn new(shape: &[usize], width: usize, bytes: &[u8]) -> Self {
        assert!(width > 0);
        assert_eq!(bytes.len(), shape.iter().product::<usize>() * width);
        Self {
            shape: shape.to_vec(),
            width,
            elements: bytes.chunks_exact(width).map(<[u8]>::to_vec).collect(),
        }
    }
    pub fn bytes(&self) -> Vec<u8> {
        self.elements.concat()
    }
    fn index(&self, coordinate: &[usize]) -> usize {
        assert_eq!(coordinate.len(), self.shape.len());
        coordinate.iter().zip(&self.shape).fold(0, |offset, (&i, &size)| {
            assert!(i < size);
            offset * size + i
        })
    }
    fn at(&self, coordinate: &[usize]) -> &Vec<u8> {
        &self.elements[self.index(coordinate)]
    }
    fn mapped(&self, shape: &[usize], map: impl Fn(&[usize]) -> Vec<usize>) -> Self {
        Self {
            shape: shape.to_vec(),
            width: self.width,
            elements: coordinates(shape).iter().map(|c| self.at(&map(c)).clone()).collect(),
        }
    }
    pub fn slice(&self, starts: &[usize], steps: &[usize], shape: &[usize]) -> Self {
        self.mapped(shape, |c| c.iter().enumerate().map(|(axis, &i)| starts[axis] + steps[axis] * i).collect())
    }
    pub fn flip(&self, axes: &[usize]) -> Self {
        self.mapped(&self.shape, |c| {
            c.iter()
                .enumerate()
                .map(|(a, &i)| if axes.contains(&a) { self.shape[a] - 1 - i } else { i })
                .collect()
        })
    }
    pub fn permute(&self, axes: &[usize]) -> Self {
        let shape: Vec<_> = axes.iter().map(|&a| self.shape[a]).collect();
        self.mapped(&shape, |c| {
            let mut original = vec![0; axes.len()];
            for (&a, &i) in axes.iter().zip(c) {
                original[a] = i;
            }
            original
        })
    }
    pub fn broadcast(&self, shape: &[usize]) -> Self {
        assert_eq!(shape.len(), self.shape.len());
        for (&source, &target) in self.shape.iter().zip(shape) {
            assert!(source == 1 || source == target);
        }
        self.mapped(shape, |c| c.iter().enumerate().map(|(a, &i)| if self.shape[a] == 1 { 0 } else { i }).collect())
    }
    pub fn select(&self, axis: usize, indices: &[usize]) -> Self {
        let mut shape = self.shape.clone();
        shape[axis] = indices.len();
        self.mapped(&shape, |c| {
            let mut original = c.to_vec();
            original[axis] = indices[c[axis]];
            original
        })
    }
    pub fn gather(&self, axis: usize, shape: &[usize], indices: &[usize]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), indices.len());
        let elements = coordinates(shape)
            .iter()
            .zip(indices)
            .map(|(c, &i)| {
                let mut original = c.clone();
                original[axis] = i;
                self.at(&original).clone()
            })
            .collect();
        Self {
            shape: shape.to_vec(),
            width: self.width,
            elements,
        }
    }
    pub fn assign(&self, starts: &[usize], steps: &[usize], value: &Self) -> Self {
        assert_eq!(self.width, value.width);
        let mut result = self.clone();
        for (c, record) in coordinates(&value.shape).iter().zip(&value.elements) {
            let target: Vec<_> = c.iter().enumerate().map(|(a, &i)| starts[a] + i * steps[a]).collect();
            let index = result.index(&target);
            result.elements[index] = record.clone();
        }
        result
    }
    pub fn cat(&self, rhs: &Self, axis: usize) -> Self {
        assert_eq!(self.width, rhs.width);
        let mut shape = self.shape.clone();
        shape[axis] += rhs.shape[axis];
        let elements = coordinates(&shape)
            .iter()
            .map(|c| {
                if c[axis] < self.shape[axis] {
                    self.at(c).clone()
                } else {
                    let mut other = c.clone();
                    other[axis] -= self.shape[axis];
                    rhs.at(&other).clone()
                }
            })
            .collect();
        Self {
            shape,
            width: self.width,
            elements,
        }
    }
    pub fn mask_where(&self, mask: &[bool], value: &Self) -> Self {
        let rhs = value.broadcast(&self.shape);
        assert_eq!(mask.len(), self.elements.len());
        let mut result = self.clone();
        for ((record, replacement), &take) in result.elements.iter_mut().zip(rhs.elements).zip(mask) {
            if take {
                *record = replacement;
            }
        }
        result
    }
    pub fn mask_select(&self, mask: &[bool]) -> Self {
        assert_eq!(mask.len(), self.elements.len());
        let elements: Vec<_> = self.elements.iter().zip(mask).filter_map(|(v, &keep)| keep.then_some(v.clone())).collect();
        Self {
            shape: vec![elements.len()],
            width: self.width,
            elements,
        }
    }
    pub fn pad(&self, padding: &[(usize, usize)], mode: PadMode, constant: &[u8]) -> Self {
        assert_eq!(self.width, constant.len());
        let shape: Vec<_> = self.shape.iter().zip(padding).map(|(&n, &(a, b))| n + a + b).collect();
        let elements = coordinates(&shape)
            .iter()
            .map(|c| {
                let mut original = Vec::new();
                for (a, &position) in c.iter().enumerate() {
                    let (before, _) = padding[a];
                    let n = self.shape[a];
                    if position >= before && position < before + n {
                        original.push(position - before);
                    } else {
                        match mode {
                            PadMode::Constant(_) => return constant.to_vec(),
                            PadMode::Edge => original.push(if position < before { 0 } else { n - 1 }),
                            PadMode::Reflect => original.push(if position < before {
                                before - position
                            } else {
                                2 * n + before - 2 - position
                            }),
                        }
                    }
                }
                self.at(&original).clone()
            })
            .collect();
        Self {
            shape,
            width: self.width,
            elements,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Gather,
    Select,
    MaskWhere,
    MaskFill,
    SliceAssign,
    Cat,
    Pad,
    MaskSelect,
    Argwhere,
    BoolScatterOr,
}
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub dtypes: usize,
    pub gather: usize,
    pub select: usize,
    pub mask_where: usize,
    pub mask_fill: usize,
    pub slice_assign: usize,
    pub cat: usize,
    pub pad: usize,
    pub mask_select: usize,
    pub argwhere: usize,
    pub bool_scatter_or: usize,
    pub source_checks: usize,
    pub values: usize,
}
impl Summary {
    pub fn cases(&self) -> usize {
        self.gather
            + self.select
            + self.mask_where
            + self.mask_fill
            + self.slice_assign
            + self.cat
            + self.pad
            + self.mask_select
            + self.argwhere
            + self.bool_scatter_or
    }
    fn record(&mut self, op: Operation, count: usize) {
        *match op {
            Operation::Gather => &mut self.gather,
            Operation::Select => &mut self.select,
            Operation::MaskWhere => &mut self.mask_where,
            Operation::MaskFill => &mut self.mask_fill,
            Operation::SliceAssign => &mut self.slice_assign,
            Operation::Cat => &mut self.cat,
            Operation::Pad => &mut self.pad,
            Operation::MaskSelect => &mut self.mask_select,
            Operation::Argwhere => &mut self.argwhere,
            Operation::BoolScatterOr => &mut self.bool_scatter_or,
        } += 1;
        self.values += count;
    }
}

fn encoded(dtype: DType, values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()[..dtype.size()].to_vec()).collect()
}
fn payload(dtype: DType, count: usize, salt: usize) -> Records {
    let edges: &[u64] = match dtype {
        DType::F64 => &[
            0,
            1 << 63,
            1,
            0x8000_0000_0000_0001,
            0x7ff0_0000_0000_0001,
            0xfff8_1234_5678_9abc,
            0x7ff0_0000_0000_0000,
            0x0010_0000_0000_0000,
            0x3ff0_0000_0000_0001,
        ],
        DType::F32 | DType::Flex32 => &[0, 0x8000_0000, 1, 0x8000_0001, 0x7f80_0001, 0xffc0_1234, 0x7f80_0000, 0x0080_0000, 0x3f80_0001],
        DType::F16 => &[0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3bff, 0x3c01, 0x7bff, 0x7c00, 0x7c01, 0xfe34],
        DType::BF16 => &[0, 0x8000, 1, 0x8001, 0x007f, 0x0080, 0x3f7f, 0x3f81, 0x7f7f, 0x7f80, 0x7f81, 0xffd3],
        _ => &[
            0,
            1,
            2,
            u64::MAX,
            u64::MAX - 1,
            1 << 63,
            (1 << 63) - 1,
            (1 << 53) + 1,
            0xfedc_ba98_7654_3210,
            0x0123_4567_89ab_cdef,
            127,
            128,
            255,
        ],
    };
    let values: Vec<_> = (0..count)
        .map(|i| {
            if dtype.is_bool() {
                u64::from((i + salt) % 5 < 2)
            } else if i % 3 != 2 {
                edges[(i + salt) % edges.len()]
            } else {
                (i as u64 + salt as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left((i % 64) as u32)
            }
        })
        .collect();
    Records::new(&[count], dtype.size(), &encoded(dtype, &values))
}
fn scalar(dtype: DType) -> (Scalar, Vec<u8>) {
    let (value, bits) = match dtype {
        DType::F64 => ((-2f64).into(), (-2f64).to_bits()),
        DType::F32 | DType::Flex32 => ((-2f32).into(), u64::from((-2f32).to_bits())),
        DType::F16 => ((-2f32).into(), 0xc000),
        DType::BF16 => ((-2f32).into(), 0xc000),
        DType::I64 => ((i64::MIN + 0x1234_5678).into(), (i64::MIN + 0x1234_5678) as u64),
        DType::U64 => (0xfedc_ba98_7654_3210u64.into(), 0xfedc_ba98_7654_3210),
        DType::I32 | DType::I16 | DType::I8 => ((-2i32).into(), (-2i64) as u64),
        DType::Bool(_) => (true.into(), 1),
        _ => (254u32.into(), 254),
    };
    (value, encoded(dtype, &[bits]))
}

#[derive(Clone)]
enum Stored {
    Float(FloatTensor<Dispatch>),
    Int(IntTensor<Dispatch>),
    Bool(BoolTensor<Dispatch>),
}
macro_rules! unary {
    ($self:expr, $float:ident, $int:ident, $bool:ident $(, $arg:expr)*) => {
        match $self {
            Stored::Float(t) => Stored::Float(Dispatch::$float(t $(, $arg)*)),
            Stored::Int(t) => Stored::Int(Dispatch::$int(t $(, $arg)*)),
            Stored::Bool(t) => Stored::Bool(Dispatch::$bool(t $(, $arg)*)),
        }
    };
}
impl Stored {
    fn new(device: &Device, dtype: DType, records: &Records) -> Self {
        let data = TensorData::from_bytes_vec(records.bytes(), records.shape.clone(), dtype);
        if dtype.is_float() {
            Self::Float(Dispatch::float_from_data(data, device.as_dispatch()))
        } else if dtype.is_bool() {
            Self::Bool(Dispatch::bool_from_data(data, device.as_dispatch()))
        } else {
            Self::Int(Dispatch::int_from_data(data, device.as_dispatch()))
        }
    }
    fn slice(self, slices: &[Slice]) -> Self {
        unary!(self, float_slice, int_slice, bool_slice, slices)
    }
    fn flip(self, axes: &[usize]) -> Self {
        unary!(self, float_flip, int_flip, bool_flip, axes)
    }
    fn permute(self, axes: &[usize]) -> Self {
        unary!(self, float_permute, int_permute, bool_permute, axes)
    }
    fn select(self, axis: usize, indices: IntTensor<Dispatch>) -> Self {
        unary!(self, float_select, int_select, bool_select, axis, indices)
    }
    fn gather(self, axis: usize, indices: IntTensor<Dispatch>) -> Self {
        match self {
            Self::Float(t) => Self::Float(Dispatch::float_gather(axis, t, indices)),
            Self::Int(t) => Self::Int(Dispatch::int_gather(axis, t, indices)),
            Self::Bool(t) => Self::Bool(Dispatch::bool_gather(axis, t, indices)),
        }
    }
    fn mask_where(self, mask: BoolTensor<Dispatch>, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Float(t), Self::Float(v)) => Self::Float(Dispatch::float_mask_where(t, mask, v)),
            (Self::Int(t), Self::Int(v)) => Self::Int(Dispatch::int_mask_where(t, mask, v)),
            (Self::Bool(t), Self::Bool(v)) => Self::Bool(Dispatch::bool_mask_where(t, mask, v)),
            _ => unreachable!(),
        }
    }
    fn mask_fill(self, mask: BoolTensor<Dispatch>, value: Scalar) -> Self {
        unary!(self, float_mask_fill, int_mask_fill, bool_mask_fill, mask, value)
    }
    async fn mask_select(self, mask: BoolTensor<Dispatch>) -> Self {
        match self {
            Self::Float(t) => Self::Float(Dispatch::float_mask_select(t, mask).await),
            Self::Int(t) => Self::Int(Dispatch::int_mask_select(t, mask).await),
            Self::Bool(t) => Self::Bool(Dispatch::bool_mask_select(t, mask).await),
        }
    }
    fn assign(self, slices: &[Slice], value: Self) -> Self {
        match (self, value) {
            (Self::Float(t), Self::Float(v)) => Self::Float(Dispatch::float_slice_assign(t, slices, v)),
            (Self::Int(t), Self::Int(v)) => Self::Int(Dispatch::int_slice_assign(t, slices, v)),
            (Self::Bool(t), Self::Bool(v)) => Self::Bool(Dispatch::bool_slice_assign(t, slices, v)),
            _ => unreachable!(),
        }
    }
    fn cat(self, rhs: Self, axis: usize) -> Self {
        match (self, rhs) {
            (Self::Float(t), Self::Float(v)) => Self::Float(Dispatch::float_cat(vec![t, v], axis)),
            (Self::Int(t), Self::Int(v)) => Self::Int(Dispatch::int_cat(vec![t, v], axis)),
            (Self::Bool(t), Self::Bool(v)) => Self::Bool(Dispatch::bool_cat(vec![t, v], axis)),
            _ => unreachable!(),
        }
    }
    fn pad(self, padding: &[(usize, usize)], mode: PadMode) -> Self {
        match self {
            Self::Float(t) => Self::Float(Dispatch::float_pad(t, padding, mode)),
            Self::Int(t) => Self::Int(Dispatch::int_pad(t, padding, mode)),
            Self::Bool(_) => unreachable!("Burn has no boolean pad API"),
        }
    }
    fn check(self, dtype: DType, expected: &Records) -> Result<(), String> {
        let (actual_dtype, shape) = match &self {
            Self::Float(t) => (t.dtype(), t.shape()),
            Self::Int(t) => (t.dtype(), t.shape()),
            Self::Bool(t) => (t.dtype(), t.shape()),
        };
        if actual_dtype != dtype || shape.as_slice() != expected.shape {
            return Err(format!("metadata {actual_dtype:?} {shape:?}, expected {dtype:?} {:?}", expected.shape));
        }
        let data = match self {
            Self::Float(t) => read_sync(Dispatch::float_into_data(t)),
            Self::Int(t) => read_sync(Dispatch::int_into_data(t)),
            Self::Bool(t) => read_sync(Dispatch::bool_into_data(t)),
        }
        .map_err(|e| format!("read: {e:?}"))?;
        let bytes = expected.bytes();
        if data.dtype() != dtype || data.as_bytes().len() != bytes.len() {
            return Err(format!(
                "storage metadata {:?} bytes={} expected={}",
                data.dtype(),
                data.as_bytes().len(),
                bytes.len()
            ));
        }
        if let Some(byte) = data.as_bytes().iter().zip(&bytes).position(|(a, b)| a != b) {
            return Err(format!(
                "element={} byte={byte} actual={:#04x} expected={:#04x}",
                byte / dtype.size(),
                data.as_bytes()[byte],
                bytes[byte]
            ));
        }
        Ok(())
    }
}
fn check(actual: Stored, dtype: DType, expected: &Records, op: Operation, summary: &mut Summary) -> Result<(), String> {
    actual.check(dtype, expected).map_err(|e| format!("{op:?}: {e}"))?;
    summary.record(op, expected.elements.len());
    Ok(())
}
fn preserve(actual: Stored, dtype: DType, expected: &Records, summary: &mut Summary) -> Result<(), String> {
    actual.check(dtype, expected).map_err(|e| format!("source preservation: {e}"))?;
    summary.source_checks += 1;
    Ok(())
}
fn indices(device: &Device, shape: &[usize], values: &[usize]) -> (IntTensor<Dispatch>, Records) {
    let records = Records::new(shape, 4, &encoded(DType::I32, &values.iter().map(|&v| v as u64).collect::<Vec<_>>()));
    let Stored::Int(t) = Stored::new(device, DType::I32, &records) else {
        unreachable!()
    };
    (t, records)
}
fn view_slices(length: usize) -> [Slice; 2] {
    [Slice::full(), Slice::new(1, Some((2 * length + 1) as isize), 2)]
}
fn selection(length: usize, axis: usize) -> Vec<usize> {
    if axis == 1 {
        vec![2, 0, 2, 1]
    } else if length == 0 {
        vec![]
    } else {
        vec![length - 1, 0, length / 2, 0]
    }
}
fn pattern(i: usize, variant: usize) -> bool {
    match variant {
        0 => false,
        1 => true,
        _ => i.is_multiple_of(7) || i % 129 == 128,
    }
}

struct MaskPlan {
    source: Records,
    logical: Records,
    truth: Vec<bool>,
}
fn mask_plan(length: usize, variant: usize, dtype: BoolDType) -> MaskPlan {
    let shape = match variant {
        3 => vec![1, 3],
        4 => vec![length, 1],
        _ => vec![3, 2 * length + 3],
    };
    let truth: Vec<_> = (0..shape.iter().product())
        .map(|i| match variant {
            3 => i != 0,
            4 => pattern(i, 2),
            _ => pattern(i, variant),
        })
        .collect();
    let width = DType::Bool(dtype).size();
    let source = Records::new(
        &shape,
        width,
        &encoded(DType::Bool(dtype), &truth.iter().map(|&v| u64::from(v)).collect::<Vec<_>>()),
    );
    let logical = match variant {
        3 => source.flip(&[1]),
        4 => source.flip(&[0]),
        _ => source.slice(&[0, 1], &[1, 2], &[3, length]).permute(&[1, 0]).flip(&[0, 1]),
    };
    let truth = logical.broadcast(&[length, 3]).elements.iter().map(|v| v[0] != 0).collect();
    MaskPlan { source, logical, truth }
}
fn mask_view(source: BoolTensor<Dispatch>, length: usize, variant: usize) -> BoolTensor<Dispatch> {
    match variant {
        3 => Dispatch::bool_flip(source, &[1]),
        4 => Dispatch::bool_flip(source, &[0]),
        _ => Dispatch::bool_flip(Dispatch::bool_permute(Dispatch::bool_slice(source, &view_slices(length)), &[1, 0]), &[0, 1]),
    }
}
fn pad_constant(dtype: DType) -> Vec<u8> {
    encoded(
        dtype,
        &[match dtype {
            DType::F64 => 2f64.to_bits(),
            DType::F32 | DType::Flex32 => u64::from(2f32.to_bits()),
            DType::F16 | DType::BF16 => 0x4000,
            _ => 2,
        }],
    )
}
fn pad_modes(length: usize) -> Vec<(Vec<(usize, usize)>, PadMode)> {
    let mut modes = vec![
        (vec![(1, 2), (2, 1)], PadMode::Constant(2.)),
        (vec![(usize::from(length > 0), if length > 0 { 2 } else { 0 }), (2, 1)], PadMode::Edge),
        (vec![if length > 1 { (1, 1) } else { (0, 0) }, (1, 1)], PadMode::Reflect),
    ];
    // Burn rejects reflect padding whenever any input dimension is empty.
    if length == 0 {
        modes.pop();
    }
    modes
}
fn coordinate_records(shape: &[usize], truth: &[bool]) -> Records {
    let rows: Vec<_> = coordinates(shape).into_iter().zip(truth).filter_map(|(c, &take)| take.then_some(c)).collect();
    let bytes = encoded(DType::I32, &rows.iter().flatten().map(|&v| v as u64).collect::<Vec<_>>());
    Records::new(&[rows.len(), shape.len()], 4, &bytes)
}
/// Sequential host OR at duplicate destinations, with opaque canonical records.
pub fn scatter_or_reference(base: &Records, axis: usize, indices: &[usize], value: &Records) -> Records {
    assert_eq!(base.width, value.width);
    assert_eq!(indices.len(), value.elements.len());
    let mut result = base.clone();
    for ((mut destination, &index), record) in coordinates(&value.shape).into_iter().zip(indices).zip(&value.elements) {
        destination[axis] = index;
        let offset = result.index(&destination);
        if record.iter().any(|&byte| byte != 0) {
            result.elements[offset].fill(0);
            result.elements[offset][0] = 1;
        }
    }
    result
}

fn mask_operations(dtype: DType, mask_dtype: DType, options: &Options) -> bool {
    !options.matching_bool_masks_only || !dtype.is_bool() || dtype == mask_dtype
}

async fn flat_tails(device: &Device, dtype: DType, length: usize, options: &Options, summary: &mut Summary) -> Result<(), String> {
    let records = payload(dtype, length + 2, 11);
    let source = Stored::new(device, dtype, &records);
    let slice = [Slice::new(1, Some((length + 1) as isize), 1)];
    let view = source.clone().slice(&slice).flip(&[0]);
    let logical = records.slice(&[1], &[1], &[length]).flip(&[0]);
    let wanted: Vec<_> = (0..length).map(|i| (i * 17 + 3) % length).collect();
    let reversed: Vec<_> = wanted.iter().rev().copied().collect();
    let (index, index_records) = indices(device, &[length], &reversed);
    let index_view = Dispatch::int_flip(index.clone(), &[0]);
    check(
        view.clone().select(0, index_view.clone()),
        dtype,
        &logical.select(0, &wanted),
        Operation::Select,
        summary,
    )?;
    check(
        view.clone().gather(0, index_view),
        dtype,
        &logical.gather(0, &[length], &wanted),
        Operation::Gather,
        summary,
    )?;
    let reverse = logical.flip(&[0]);
    check(
        view.clone().assign(&[Slice::full()], view.clone().flip(&[0])),
        dtype,
        &reverse,
        Operation::SliceAssign,
        summary,
    )?;
    let midpoint = length / 2;
    let left = view.clone().slice(&[Slice::new(0, Some(midpoint as isize), 1)]);
    let right = view.clone().slice(&[Slice::new(midpoint as isize, Some(length as isize), 1)]);
    check(left.cat(right, 0), dtype, &logical, Operation::Cat, summary)?;
    for mask_dtype in MASK_DTYPES {
        let mask_dtype = DType::Bool(mask_dtype);
        let mask_records = Records::new(&[2 * length + 3], mask_dtype.size(), &encoded(mask_dtype, &vec![1; 2 * length + 3]));
        let Stored::Bool(mask_source) = Stored::new(device, mask_dtype, &mask_records) else {
            unreachable!()
        };
        let mask = Dispatch::bool_flip(
            Dispatch::bool_slice(mask_source.clone(), &[Slice::new(1, Some((2 * length + 1) as isize), 2)]),
            &[0],
        );
        if mask_operations(dtype, mask_dtype, options) {
            check(
                view.clone().mask_where(mask.clone(), view.clone().flip(&[0])),
                dtype,
                &reverse,
                Operation::MaskWhere,
                summary,
            )?;
            let (fill, bytes) = scalar(dtype);
            let expected = Records::new(&[length], dtype.size(), &bytes.repeat(length));
            check(view.clone().mask_fill(mask.clone(), fill), dtype, &expected, Operation::MaskFill, summary)?;
        }
        check(view.clone().mask_select(mask).await, dtype, &logical, Operation::MaskSelect, summary)?;
        preserve(Stored::Bool(mask_source), mask_dtype, &mask_records, summary)?;
    }
    if !dtype.is_bool() {
        check(view.clone().pad(&[(0, 0)], PadMode::Constant(2.)), dtype, &logical, Operation::Pad, summary)?;
    }
    preserve(Stored::Int(index), DType::I32, &index_records, summary)?;
    preserve(source, dtype, &records, summary)?;
    preserve(view, dtype, &logical, summary)?;
    Ok(())
}

async fn batch(device: &Device, dtype: DType, length: usize, options: &Options, summary: &mut Summary) -> Result<(), String> {
    let mut source_records = payload(dtype, 3 * (2 * length + 3), 0);
    source_records.shape = vec![3, 2 * length + 3];
    let source = Stored::new(device, dtype, &source_records);
    let view = source.clone().slice(&view_slices(length)).permute(&[1, 0]).flip(&[0, 1]);
    let logical = source_records.slice(&[0, 1], &[1, 2], &[3, length]).permute(&[1, 0]).flip(&[0, 1]);
    let mut rhs_records = payload(dtype, 3, 7);
    rhs_records.shape = vec![1, 3];
    let rhs = Stored::new(device, dtype, &rhs_records);
    for axis in 0..2 {
        let wanted = selection(length, axis);
        let reversed: Vec<_> = wanted.iter().rev().copied().collect();
        let (index_source, index_records) = indices(device, &[wanted.len()], &reversed);
        let index_view = Dispatch::int_flip(index_source.clone(), &[0]);
        check(
            view.clone().select(axis, index_view),
            dtype,
            &logical.select(axis, &wanted),
            Operation::Select,
            summary,
        )
        .map_err(|e| format!("axis={axis}: {e}"))?;
        preserve(Stored::Int(index_source), DType::I32, &index_records, summary)?;
        let wanted: Vec<_> = (0..length)
            .flat_map(|row| (0..3).map(move |col| if axis == 0 { (row * 5 + col * 3) % length } else { (row + col * 2) % 3 }))
            .collect();
        let reversed: Vec<_> = wanted.iter().rev().copied().collect();
        let (index_source, index_records) = indices(device, &[length, 3], &reversed);
        let index_view = Dispatch::int_flip(index_source.clone(), &[0, 1]);
        check(
            view.clone().gather(axis, index_view),
            dtype,
            &logical.gather(axis, &[length, 3], &wanted),
            Operation::Gather,
            summary,
        )
        .map_err(|e| format!("axis={axis}: {e}"))?;
        preserve(Stored::Int(index_source), DType::I32, &index_records, summary)?;
    }
    let reversed = logical.flip(&[0, 1]);
    check(
        view.clone().assign(&[Slice::full(), Slice::full()], view.clone().flip(&[0, 1])),
        dtype,
        &reversed,
        Operation::SliceAssign,
        summary,
    )
    .map_err(|e| format!("reversed destination view: {e}"))?;
    let stepped_expected = source_records.assign(&[0, 1], &[1, 2], &reversed.permute(&[1, 0]));
    check(
        source.clone().assign(&view_slices(length), view.clone().flip(&[0, 1]).permute(&[1, 0])),
        dtype,
        &stepped_expected,
        Operation::SliceAssign,
        summary,
    )
    .map_err(|e| format!("stepped alias destination: {e}"))?;
    let overlap_records = payload(dtype, length + 2, 9);
    let overlap = Stored::new(device, dtype, &overlap_records);
    let overlap_value = overlap.clone().slice(&[Slice::new(0, Some(length as isize), 1)]);
    let expected = overlap_records.assign(&[1], &[1], &overlap_records.slice(&[0], &[1], &[length]));
    check(
        overlap.clone().assign(&[Slice::new(1, Some((length + 1) as isize), 1)], overlap_value),
        dtype,
        &expected,
        Operation::SliceAssign,
        summary,
    )
    .map_err(|e| format!("overlapping forward copy: {e}"))?;
    for axis in 0..2 {
        check(
            view.clone().cat(view.clone().flip(&[axis]), axis),
            dtype,
            &logical.cat(&logical.flip(&[axis]), axis),
            Operation::Cat,
            summary,
        )
        .map_err(|e| format!("axis={axis}: {e}"))?;
    }
    for mask_dtype in MASK_DTYPES {
        for variant in 0..5 {
            let plan = mask_plan(length, variant, mask_dtype);
            let Stored::Bool(mask_source) = Stored::new(device, DType::Bool(mask_dtype), &plan.source) else {
                unreachable!()
            };
            let mask_view = mask_view(mask_source.clone(), length, variant);
            let (replacement, replacement_records) = if variant == 4 {
                (
                    view.clone().slice(&[Slice::full(), Slice::new(1, Some(2), 1)]),
                    logical.slice(&[0, 1], &[1, 1], &[length, 1]),
                )
            } else {
                (rhs.clone(), rhs_records.clone())
            };
            if mask_operations(dtype, DType::Bool(mask_dtype), options) {
                check(
                    view.clone().mask_where(mask_view.clone(), replacement),
                    dtype,
                    &logical.mask_where(&plan.truth, &replacement_records),
                    Operation::MaskWhere,
                    summary,
                )
                .map_err(|e| format!("mask={mask_dtype:?} variant={variant}: {e}"))?;
                let (fill, fill_bytes) = scalar(dtype);
                let constant = Records::new(&[1, 1], dtype.size(), &fill_bytes);
                check(
                    view.clone().mask_fill(mask_view.clone(), fill),
                    dtype,
                    &logical.mask_where(&plan.truth, &constant),
                    Operation::MaskFill,
                    summary,
                )
                .map_err(|e| format!("mask={mask_dtype:?} variant={variant}: {e}"))?;
            }
            let expanded = Dispatch::bool_expand(mask_view, [length, 3].into());
            check(
                view.clone().mask_select(expanded.clone()).await,
                dtype,
                &logical.mask_select(&plan.truth),
                Operation::MaskSelect,
                summary,
            )
            .map_err(|e| format!("mask={mask_dtype:?} variant={variant}: {e}"))?;
            if dtype == DType::Bool(mask_dtype) {
                let output = Dispatch::bool_argwhere(expanded.clone(), IntDType::I32).await;
                check(
                    Stored::Int(output),
                    DType::I32,
                    &coordinate_records(&[length, 3], &plan.truth),
                    Operation::Argwhere,
                    summary,
                )
                .map_err(|e| format!("mask={mask_dtype:?} variant={variant}: {e}"))?;
            }
            preserve(Stored::Bool(mask_source), DType::Bool(mask_dtype), &plan.source, summary)?;
            preserve(Stored::Bool(expanded), DType::Bool(mask_dtype), &plan.logical.broadcast(&[length, 3]), summary)?;
        }
    }
    if dtype.is_bool() {
        let Stored::Bool(base) = view.clone() else { unreachable!() };
        let Stored::Bool(value) = view.clone().flip(&[0, 1]) else { unreachable!() };
        for axis in 0..if length == 0 && options.skip_empty_bool_scatter { 0 } else { 2 } {
            let wanted: Vec<_> = (0..length)
                .flat_map(|row| (0..3).map(move |col| if axis == 0 { row / 2 } else { col / 2 }))
                .collect();
            let (index, index_records) = indices(device, &[length, 3], &wanted);
            let output = Dispatch::bool_scatter_or(axis, base.clone(), index.clone(), value.clone());
            check(
                Stored::Bool(output),
                dtype,
                &scatter_or_reference(&logical, axis, &wanted, &reversed),
                Operation::BoolScatterOr,
                summary,
            )
            .map_err(|e| format!("axis={axis}: {e}"))?;
            preserve(Stored::Int(index), DType::I32, &index_records, summary)?;
        }
    } else {
        for (padding, mode) in pad_modes(length) {
            check(
                view.clone().pad(&padding, mode),
                dtype,
                &logical.pad(&padding, mode, &pad_constant(dtype)),
                Operation::Pad,
                summary,
            )
            .map_err(|e| format!("mode={mode:?}: {e}"))?;
        }
    }
    preserve(source, dtype, &source_records, summary)?;
    preserve(view, dtype, &logical, summary)?;
    preserve(rhs, dtype, &rhs_records, summary)?;
    preserve(overlap, dtype, &overlap_records, summary)?;
    Ok(())
}

/// Pure expected totals include output elements, not source preservation reads.
pub fn expected_summary(options: &Options) -> Summary {
    let mut summary = Summary::default();
    for dtype in DTYPES {
        if dtype == DType::Flex32 && !options.include_flex32 {
            continue;
        }
        summary.dtypes += 1;
        for length in LENGTHS {
            for op in [
                Operation::Select,
                Operation::Gather,
                Operation::SliceAssign,
                Operation::Cat,
                Operation::MaskSelect,
                Operation::MaskSelect,
            ] {
                summary.record(op, length);
            }
            for mask in MASK_DTYPES {
                if mask_operations(dtype, DType::Bool(mask), options) {
                    summary.record(Operation::MaskWhere, length);
                    summary.record(Operation::MaskFill, length);
                }
            }
            if !dtype.is_bool() {
                summary.record(Operation::Pad, length);
            }
            summary.source_checks += 5;
            for axis in 0..2 {
                summary.record(Operation::Select, if axis == 0 { if length == 0 { 0 } else { 12 } } else { 4 * length });
                summary.record(Operation::Gather, 3 * length);
            }
            for count in [3 * length, 3 * (2 * length + 3), length + 2] {
                summary.record(Operation::SliceAssign, count);
            }
            for _ in 0..2 {
                summary.record(Operation::Cat, 6 * length);
            }
            for mask_dtype in MASK_DTYPES {
                for variant in 0..5 {
                    let plan = mask_plan(length, variant, mask_dtype);
                    if mask_operations(dtype, DType::Bool(mask_dtype), options) {
                        summary.record(Operation::MaskWhere, 3 * length);
                        summary.record(Operation::MaskFill, 3 * length);
                    }
                    let selected = plan.truth.iter().filter(|&&v| v).count();
                    summary.record(Operation::MaskSelect, selected);
                    if dtype == DType::Bool(mask_dtype) {
                        summary.record(Operation::Argwhere, 2 * selected);
                    }
                }
            }
            if dtype.is_bool() {
                for _ in 0..if length == 0 && options.skip_empty_bool_scatter { 0 } else { 2 } {
                    summary.record(Operation::BoolScatterOr, 3 * length);
                }
            } else {
                for (padding, _) in pad_modes(length) {
                    summary.record(Operation::Pad, (length + padding[0].0 + padding[0].1) * (3 + padding[1].0 + padding[1].1));
                }
            }
            summary.source_checks += if dtype.is_bool() && !(length == 0 && options.skip_empty_bool_scatter) {
                30
            } else {
                28
            };
        }
    }
    summary
}

/// Run identical strict byte checks on Linux and native D3OS Dispatch devices.
pub async fn run(device: &Device, options: &Options) -> Result<Summary, String> {
    let mut summary = Summary::default();
    for dtype in DTYPES {
        if dtype == DType::Flex32 && !options.include_flex32 {
            continue;
        }
        for length in LENGTHS {
            flat_tails(device, dtype, length, options, &mut summary)
                .await
                .map_err(|e| format!("{dtype:?} flat length={length}: {e}"))?;
            batch(device, dtype, length, options, &mut summary)
                .await
                .map_err(|e| format!("{dtype:?} length={length}: {e}"))?;
        }
        summary.dtypes += 1;
    }
    let expected = expected_summary(options);
    if summary != expected {
        return Err(format!("corpus totals {summary:?}, expected {expected:?}"));
    }
    Ok(summary)
}
