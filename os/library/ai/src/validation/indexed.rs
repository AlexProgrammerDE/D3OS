//! Exact byte-record references for public indexed backend operations.
//!
//! Boolean ND indexing is absent from Burn's backend trait. Integer ordering
//! fixes its index dtype through the input dtype; requested output dtype checks
//! use the float APIs and Boolean argwhere. Invalid indices stay in raw backend
//! tests because these APIs offer no recoverable bounds-error result.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{
        Backend as BurnBackend, BoolDType, DType, Dispatch, IntDType, TensorData, TensorMetadata,
        tensor::{BoolTensor, FloatTensor, IntTensor},
    },
    tensor::{Device, IndexingUpdateOp, Slice},
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
pub const INDEX_DTYPES: [IntDType; 8] = [
    IntDType::I64,
    IntDType::I32,
    IntDType::I16,
    IntDType::I8,
    IntDType::U64,
    IntDType::U32,
    IntDType::U16,
    IntDType::U8,
];
const LENGTHS: [usize; 6] = [0, 1, 127, 128, 129, 257];

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub include_flex32: bool,
    /// Isolate the pinned Linux rank-zero ND result failure.
    pub include_scalar_nd: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            include_flex32: true,
            include_scalar_nd: true,
        }
    }
}

/// Each element is an opaque fixed-width record. Movement never decodes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Records {
    pub shape: Vec<usize>,
    width: usize,
    bytes: Vec<u8>,
}
/// Enumerate coordinates lexicographically, including one coordinate for a scalar.
fn coordinates(shape: &[usize]) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]];
    for &size in shape {
        result = result
            .into_iter()
            .flat_map(|prefix| {
                (0..size).map(move |i| {
                    let mut c = prefix.clone();
                    c.push(i);
                    c
                })
            })
            .collect();
    }
    result
}
fn offset(shape: &[usize], coordinate: &[usize]) -> usize {
    assert_eq!(shape.len(), coordinate.len());
    shape.iter().zip(coordinate).fold(0, |n, (&size, &i)| {
        assert!(i < size);
        n * size + i
    })
}
impl Records {
    pub fn new(shape: &[usize], width: usize, bytes: &[u8]) -> Self {
        assert!(width > 0);
        assert_eq!(shape.iter().product::<usize>() * width, bytes.len());
        Self {
            shape: shape.to_vec(),
            width,
            bytes: bytes.to_vec(),
        }
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn reshape(&self, shape: &[usize]) -> Self {
        Self::new(shape, self.width, &self.bytes)
    }
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
    pub fn len(&self) -> usize {
        self.bytes.len() / self.width
    }
    fn at(&self, c: &[usize]) -> &[u8] {
        let start = offset(&self.shape, c) * self.width;
        &self.bytes[start..start + self.width]
    }
    fn mapped(&self, shape: &[usize], map: impl Fn(&[usize]) -> Vec<usize>) -> Self {
        let bytes = coordinates(shape).iter().flat_map(|c| self.at(&map(c)).to_vec()).collect::<Vec<_>>();
        Self::new(shape, self.width, &bytes)
    }
    pub fn slice(&self, starts: &[usize], steps: &[usize], shape: &[usize]) -> Self {
        assert_eq!(starts.len(), self.shape.len());
        assert_eq!(steps.len(), starts.len());
        self.mapped(shape, |c| c.iter().enumerate().map(|(a, &i)| starts[a] + steps[a] * i).collect())
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
    pub fn gather(&self, axis: usize, index_shape: &[usize], indices: &[usize]) -> Self {
        assert_eq!(index_shape.len(), self.shape.len());
        assert_eq!(index_shape.iter().product::<usize>(), indices.len());
        let mut row = 0;
        let bytes = coordinates(index_shape)
            .into_iter()
            .flat_map(|mut c| {
                c[axis] = indices[row];
                row += 1;
                self.at(&c).to_vec()
            })
            .collect::<Vec<_>>();
        Self::new(index_shape, self.width, &bytes)
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
    /// Follow the ND definition: each coordinate tuple selects a trailing slice.
    pub fn gather_nd(&self, index_shape: &[usize], indices: &[usize]) -> Self {
        let width = *index_shape.last().unwrap();
        assert!(width > 0 && width <= self.shape.len());
        assert!(!indices.is_empty(), "ND index tensors must not be empty");
        assert_eq!(index_shape.iter().product::<usize>(), indices.len());
        let mut shape = index_shape[..index_shape.len() - 1].to_vec();
        shape.extend_from_slice(&self.shape[width..]);
        let mut bytes = Vec::new();
        for tuple in indices.chunks_exact(width) {
            assert!(tuple.iter().zip(&self.shape).all(|(&i, &size)| i < size));
            for suffix in coordinates(&self.shape[width..]) {
                let mut c = tuple.to_vec();
                c.extend(suffix);
                bytes.extend_from_slice(self.at(&c));
            }
        }
        Self::new(&shape, self.width, &bytes)
    }
    /// Assignment requires unique full destination coordinates in the corpus.
    pub fn scatter(&self, axis: usize, indices: &[usize], values: &Self) -> Self {
        assert_eq!(indices.len(), values.len());
        self.assign_coordinates(
            values,
            coordinates(&values.shape)
                .into_iter()
                .zip(indices)
                .map(|(mut c, &i)| {
                    c[axis] = i;
                    c
                })
                .collect(),
        )
    }
    pub fn select_assign(&self, axis: usize, indices: &[usize], values: &Self) -> Self {
        assert_eq!(values.shape[axis], indices.len());
        self.assign_coordinates(
            values,
            coordinates(&values.shape)
                .into_iter()
                .map(|mut c| {
                    c[axis] = indices[c[axis]];
                    c
                })
                .collect(),
        )
    }
    pub fn scatter_nd(&self, index_shape: &[usize], indices: &[usize], values: &Self) -> Self {
        let width = *index_shape.last().unwrap();
        assert_eq!(self.gather_nd(index_shape, indices).shape, values.shape);
        let mut destinations = Vec::new();
        for tuple in indices.chunks_exact(width) {
            assert!(tuple.iter().zip(&self.shape).all(|(&i, &size)| i < size));
            for suffix in coordinates(&self.shape[width..]) {
                let mut c = tuple.to_vec();
                c.extend(suffix);
                destinations.push(c);
            }
        }
        self.assign_coordinates(values, destinations)
    }
    fn assign_coordinates(&self, values: &Self, destinations: Vec<Vec<usize>>) -> Self {
        assert_eq!(self.width, values.width);
        assert_eq!(destinations.len(), values.len());
        if values.is_empty() {
            return self.clone();
        }
        let mut seen = vec![false; self.len()];
        let mut output = self.clone();
        for (c, value) in destinations.iter().zip(values.bytes.chunks_exact(values.width)) {
            let at = offset(&self.shape, c);
            assert!(!seen[at], "public exact assignment requires unique destinations");
            seen[at] = true;
            output.bytes[at * self.width..(at + 1) * self.width].copy_from_slice(value);
        }
        output
    }
}
fn encoded(dtype: DType, values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()[..dtype.size()].to_vec()).collect()
}
fn index_records(shape: &[usize], dtype: IntDType, values: &[usize]) -> Records {
    let dtype = DType::from(dtype);
    Records::new(shape, dtype.size(), &encoded(dtype, &values.iter().map(|&i| i as u64).collect::<Vec<_>>()))
}
fn decoded_indices(records: &Records) -> Vec<usize> {
    records
        .bytes
        .chunks_exact(records.width)
        .map(|bytes| {
            let mut word = [0; 8];
            word[..bytes.len()].copy_from_slice(bytes);
            usize::try_from(u64::from_le_bytes(word)).expect("corpus indices fit usize")
        })
        .collect()
}
fn payload(shape: &[usize], dtype: DType, salt: usize) -> Records {
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
    let values: Vec<_> = (0..shape.iter().product())
        .map(|i: usize| {
            if dtype.is_bool() {
                u64::from((i + salt) % 5 < 2)
            } else if i % 3 != 2 {
                edges[(i + salt) % edges.len()]
            } else {
                (i as u64 + salt as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left((i % 64) as u32)
            }
        })
        .collect();
    Records::new(shape, dtype.size(), &encoded(dtype, &values))
}

#[derive(Clone, Debug)]
enum View {
    Slice { starts: Vec<usize>, steps: Vec<usize>, shape: Vec<usize> },
    Flip(Vec<usize>),
    Permute(Vec<usize>),
    Reshape(Vec<usize>),
}
impl View {
    fn reference(&self, input: &Records) -> Records {
        match self {
            Self::Slice { starts, steps, shape } => input.slice(starts, steps, shape),
            Self::Flip(axes) => input.flip(axes),
            Self::Permute(axes) => input.permute(axes),
            Self::Reshape(shape) => input.reshape(shape),
        }
    }
}
#[derive(Clone, Debug)]
struct Input {
    root: Records,
    views: Vec<View>,
}
impl Input {
    fn logical(&self) -> Records {
        self.views.iter().fold(self.root.clone(), |data, view| view.reference(&data))
    }
    fn payload(shape: &[usize], dtype: DType, salt: usize) -> Self {
        // Reverse the physical axes, then take an offset step-two slice, transpose,
        // and reverse logical axes. Every nonempty view has an actual offset.
        let axes: Vec<_> = (0..shape.len()).rev().collect();
        let physical: Vec<_> = axes.iter().map(|&a| shape[a]).collect();
        let last = shape.len() - 1;
        let mut root_shape = physical.clone();
        root_shape[last] = 2 * physical[last] + 3;
        let mut starts = vec![0; shape.len()];
        starts[last] = 1;
        let mut steps = vec![1; shape.len()];
        steps[last] = 2;
        Self {
            root: payload(&root_shape, dtype, salt),
            views: vec![
                View::Slice {
                    starts,
                    steps,
                    shape: physical,
                },
                View::Permute(axes),
                View::Flip((0..shape.len()).collect()),
            ],
        }
    }
    fn indices(shape: &[usize], dtype: IntDType, values: &[usize]) -> Self {
        Self::strided(index_records(shape, dtype, values))
    }
    fn strided(desired: Records) -> Self {
        let shape = &desired.shape;
        let axes: Vec<_> = (0..shape.len()).collect();
        let reversed = desired.flip(&axes);
        let last = shape.len() - 1;
        let mut root_shape = shape.to_vec();
        root_shape[last] = 2 * shape[last] + 3;
        let mut root = Records::new(&root_shape, desired.width, &vec![0; root_shape.iter().product::<usize>() * desired.width]);
        for (c, value) in coordinates(shape).iter().zip(reversed.bytes.chunks_exact(reversed.width)) {
            let mut target = c.clone();
            target[last] = 1 + 2 * c[last];
            let at = offset(&root_shape, &target) * root.width;
            root.bytes[at..at + root.width].copy_from_slice(value);
        }
        let mut starts = vec![0; shape.len()];
        starts[last] = 1;
        let mut steps = vec![1; shape.len()];
        steps[last] = 2;
        Self {
            root,
            views: vec![
                View::Slice {
                    starts,
                    steps,
                    shape: shape.to_vec(),
                },
                View::Flip(axes),
            ],
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Gather,
    Select,
    GatherNd,
    ScatterAssign,
    SelectAssign,
    ScatterNdAssign,
}
#[derive(Clone, Debug)]
struct Case {
    label: String,
    op: Operation,
    dtype: DType,
    index_dtype: IntDType,
    axis: usize,
    source: Input,
    indices: Input,
    // Updates share the source allocation, including overlapping reversed slices.
    value_views: Vec<View>,
    expected: Records,
}
struct CaseBuilder {
    dtype: DType,
    index_dtype: IntDType,
}
impl CaseBuilder {
    fn build(&self, label: String, op: Operation, axis: usize, source: Input, indices: Input, value_views: Vec<View>) -> Case {
        let Self { dtype, index_dtype } = *self;
        let data = source.logical();
        let index = indices.logical();
        let ids = decoded_indices(&index);
        let values = value_views.iter().fold(data.clone(), |data, view| view.reference(&data));
        let expected = match op {
            Operation::Gather => data.gather(axis, &index.shape, &ids),
            Operation::Select => data.select(axis, &ids),
            Operation::GatherNd => data.gather_nd(&index.shape, &ids),
            Operation::ScatterAssign => data.scatter(axis, &ids, &values),
            Operation::SelectAssign => data.select_assign(axis, &ids, &values),
            Operation::ScatterNdAssign => data.scatter_nd(&index.shape, &ids, &values),
        };
        Case {
            label,
            op,
            dtype,
            index_dtype,
            axis,
            source,
            indices,
            value_views,
            expected,
        }
    }
}
impl Case {
    fn assignment(&self) -> bool {
        matches!(self.op, Operation::ScatterAssign | Operation::SelectAssign | Operation::ScatterNdAssign)
    }
}
fn prefix_views(shape: &[usize], axis: usize, length: usize) -> Vec<View> {
    let mut sliced = shape.to_vec();
    sliced[axis] = length;
    vec![
        View::Slice {
            starts: (0..shape.len()).map(|a| usize::from(a == axis)).collect(),
            steps: vec![1; shape.len()],
            shape: sliced,
        },
        View::Flip((0..shape.len()).collect()),
    ]
}
/// Cases are generated a dtype at a time to bound host memory.
fn indexed_cases(dtype: DType, index_dtype: IntDType, options: &Options) -> Vec<Case> {
    let mut cases = Vec::new();
    let builder = CaseBuilder { dtype, index_dtype };
    for length in LENGTHS {
        let shape = [7.max(length + 2)];
        let source = Input::payload(&shape, dtype, length + 3);
        let reads: Vec<_> = (0..length).map(|i| (i * 13 + 3) % shape[0].min(127)).collect();
        let index = Input::indices(&[length], index_dtype, &reads);
        for op in [Operation::Gather, Operation::Select] {
            cases.push(builder.build(format!("tail={length}"), op, 0, source.clone(), index.clone(), vec![]));
        }
        if !dtype.is_bool() {
            let count = length.min(127);
            let ids: Vec<_> = (0..count).rev().collect();
            let index = Input::indices(&[count], index_dtype, &ids);
            for op in [Operation::ScatterAssign, Operation::SelectAssign] {
                cases.push(builder.build(
                    format!("overlap tail={length}"),
                    op,
                    0,
                    source.clone(),
                    index.clone(),
                    prefix_views(&shape, 0, count),
                ));
            }
        }
    }
    if !dtype.is_bool() {
        // Multicomponent coordinates keep every coordinate representable in I8
        // while providing enough unique destinations for a 257-element tail.
        let shape = [17, 19, 3];
        let source = Input::payload(&shape, dtype, 71);
        for length in LENGTHS.into_iter().filter(|&length| length > 0) {
            for width in [2, 3] {
                let tuples = coordinates(&shape[..width]);
                let ids: Vec<_> = tuples.iter().skip(5).take(length).flatten().copied().collect();
                for index_shape in [vec![length, width], vec![1, length, width]] {
                    let index = Input::indices(&index_shape, index_dtype, &ids);
                    let data = source.logical();
                    let value_shape = data.gather_nd(&index_shape, &ids).shape;
                    let needed = value_shape.iter().product::<usize>();
                    let views = vec![
                        View::Flip(vec![0, 1, 2]),
                        View::Reshape(vec![data.len()]),
                        View::Slice {
                            starts: vec![0],
                            steps: vec![1],
                            shape: vec![needed],
                        },
                        View::Reshape(value_shape),
                    ];
                    for op in [Operation::GatherNd, Operation::ScatterNdAssign] {
                        cases.push(builder.build(
                            format!("ND tail={length} tuples={index_shape:?}"),
                            op,
                            0,
                            source.clone(),
                            index.clone(),
                            views.clone(),
                        ));
                    }
                }
            }
        }
    }
    if !dtype.is_bool() {
        // Empty ND outputs are supported when valid, nonempty coordinates
        // select slices containing a zero-sized trailing data dimension.
        for shape in [vec![2, 0], vec![2, 0, 3], vec![2, 3, 0]] {
            let source = Input::payload(&shape, dtype, 83);
            let zero_axis = shape.iter().position(|&size| size == 0).unwrap();
            for width in 1..=zero_axis {
                let tuples = coordinates(&shape[..width]);
                let count = tuples.len().min(2);
                let ids: Vec<_> = tuples.iter().rev().take(count).flatten().copied().collect();
                for index_shape in [vec![count, width], vec![1, count, width], vec![width]] {
                    let tuple_count = index_shape[..index_shape.len() - 1].iter().product::<usize>();
                    let ids = &ids[..tuple_count * width];
                    let index = Input::indices(&index_shape, index_dtype, ids);
                    let data = source.logical();
                    let value_shape = data.gather_nd(&index_shape, ids).shape;
                    let views = vec![View::Reshape(value_shape)];
                    for op in [Operation::GatherNd, Operation::ScatterNdAssign] {
                        cases.push(builder.build(
                            format!("ND empty data={shape:?} tuples={index_shape:?}"),
                            op,
                            0,
                            source.clone(),
                            index.clone(),
                            views.clone(),
                        ));
                    }
                }
            }
        }
    }
    for rank in 1..=8 {
        let mut shape = vec![1; rank];
        shape[rank - 1] = 7;
        if rank > 1 {
            shape[0] = 2;
        }
        if rank > 2 {
            shape[rank / 2] = 3;
        }
        let source = Input::payload(&shape, dtype, rank * 17);
        for axis in 0..rank {
            let size = shape[axis];
            let ids: Vec<_> = (0..size).map(|i| (i + 1) % size).collect();
            let vector = Input::indices(&[size], index_dtype, &ids);
            cases.push(builder.build(
                format!("rank={rank} axis={axis}"),
                Operation::Select,
                axis,
                source.clone(),
                vector.clone(),
                vec![],
            ));
            let ids: Vec<_> = coordinates(&shape).iter().map(|c| (c[axis] + 1) % size).collect();
            let full = Input::indices(&shape, index_dtype, &ids);
            cases.push(builder.build(
                format!("rank={rank} axis={axis}"),
                Operation::Gather,
                axis,
                source.clone(),
                full.clone(),
                vec![],
            ));
            if !dtype.is_bool() {
                for (op, index) in [(Operation::ScatterAssign, full), (Operation::SelectAssign, vector)] {
                    cases.push(builder.build(
                        format!("rank={rank} axis={axis} alias"),
                        op,
                        axis,
                        source.clone(),
                        index,
                        vec![View::Flip((0..rank).collect())],
                    ));
                }
            }
        }
        if dtype.is_bool() {
            continue;
        }
        for width in 1..=rank {
            let available = coordinates(&shape[..width]);
            let count = available.len().min(4);
            let ids: Vec<_> = available.iter().cycle().skip(1).take(count).flatten().copied().collect();
            // Multidimensional tuple grids plus rank-one tuple tensors. Keep
            // output rank <= 8 even when the data itself has rank eight.
            let mut grids = vec![vec![count, width]];
            if rank - width + 2 <= 8 {
                grids.push(vec![1, count, width]);
            }
            grids.push(vec![width]);
            for index_shape in grids {
                let scalar = index_shape.len() == 1 && width == rank;
                if scalar && !options.include_scalar_nd {
                    continue;
                }
                let tuple_count = index_shape[..index_shape.len() - 1].iter().product::<usize>();
                let index = Input::indices(&index_shape, index_dtype, &ids[..tuple_count * width]);
                cases.push(builder.build(
                    format!("ND rank={rank} tuples={index_shape:?}"),
                    Operation::GatherNd,
                    0,
                    source.clone(),
                    index.clone(),
                    vec![],
                ));
                let data = source.logical();
                let value_shape = data.gather_nd(&index_shape, &ids[..tuple_count * width]).shape;
                let needed = value_shape.iter().product::<usize>();
                let value_views = vec![
                    View::Flip((0..rank).collect()),
                    View::Reshape(vec![data.len()]),
                    View::Slice {
                        starts: vec![0],
                        steps: vec![1],
                        shape: vec![needed],
                    },
                    View::Reshape(value_shape),
                ];
                cases.push(builder.build(
                    format!("ND rank={rank} tuples={index_shape:?}"),
                    Operation::ScatterNdAssign,
                    0,
                    source.clone(),
                    index,
                    value_views,
                ));
            }
        }
    }
    cases
}

#[derive(Clone)]
enum Stored<B: BurnBackend> {
    Float(FloatTensor<B>),
    Int(IntTensor<B>),
    Bool(BoolTensor<B>),
}
macro_rules! unary {
    ($self:expr, $float:ident, $int:ident, $bool:ident $(, $arg:expr)*) => {
        match $self {
            Stored::Float(t) => Stored::Float(B::$float(t $(, $arg)*)),
            Stored::Int(t) => Stored::Int(B::$int(t $(, $arg)*)),
            Stored::Bool(t) => Stored::Bool(B::$bool(t $(, $arg)*)),
        }
    };
}
impl<B: BurnBackend> Stored<B> {
    fn new(device: &B::Device, dtype: DType, records: &Records) -> Self {
        let data = TensorData::from_bytes_vec(records.bytes().to_vec(), records.shape.clone(), dtype);
        if dtype.is_float() {
            Self::Float(B::float_from_data(data, device))
        } else if dtype.is_bool() {
            Self::Bool(B::bool_from_data(data, device))
        } else {
            Self::Int(B::int_from_data(data, device))
        }
    }
    fn view(self, view: &View) -> Self {
        match view {
            View::Slice { starts, steps, shape } => {
                let slices: Vec<_> = starts
                    .iter()
                    .zip(steps)
                    .zip(shape)
                    .map(|((&start, &step), &length)| Slice::new(start as isize, Some((start + step * length) as isize), step as isize))
                    .collect();
                unary!(self, float_slice, int_slice, bool_slice, &slices)
            }
            View::Flip(axes) => unary!(self, float_flip, int_flip, bool_flip, axes),
            View::Permute(axes) => unary!(self, float_permute, int_permute, bool_permute, axes),
            View::Reshape(shape) => self.reshape(shape),
        }
    }
    fn views(mut self, views: &[View]) -> Self {
        for view in views {
            self = self.view(view);
        }
        self
    }
    fn reshape(self, shape: &[usize]) -> Self {
        unary!(self, float_reshape, int_reshape, bool_reshape, shape.into())
    }
    fn int(self) -> IntTensor<B> {
        let Self::Int(t) = self else { unreachable!() };
        t
    }
    async fn check(self, dtype: DType, expected: &Records) -> Result<(), String> {
        let (actual_dtype, shape) = match &self {
            Self::Float(t) => (t.dtype(), t.shape()),
            Self::Int(t) => (t.dtype(), t.shape()),
            Self::Bool(t) => (t.dtype(), t.shape()),
        };
        if actual_dtype != dtype || shape.as_slice() != expected.shape {
            return Err(format!("metadata {actual_dtype:?} {shape:?}, expected {dtype:?} {:?}", expected.shape));
        }
        let data = match self {
            Self::Float(t) => B::float_into_data(t).await,
            Self::Int(t) => B::int_into_data(t).await,
            Self::Bool(t) => B::bool_into_data(t).await,
        }
        .map_err(|e| format!("read: {e:?}"))?;
        if data.dtype() != dtype || data.shape().as_slice() != expected.shape || data.as_bytes().len() != expected.bytes.len() {
            return Err(format!(
                "storage metadata {:?} {:?} bytes={} expected={}",
                data.dtype(),
                data.shape(),
                data.as_bytes().len(),
                expected.bytes.len()
            ));
        }
        if let Some(at) = data.as_bytes().iter().zip(&expected.bytes).position(|(a, b)| a != b) {
            return Err(format!(
                "element={} byte={at} actual={:#04x} expected={:#04x}",
                at / expected.width,
                data.as_bytes()[at],
                expected.bytes[at]
            ));
        }
        Ok(())
    }
    fn indexed(self, case: &Case, indices: IntTensor<B>, values: Option<Self>) -> Self {
        match case.op {
            Operation::Gather => match self {
                Self::Float(t) => Self::Float(B::float_gather(case.axis, t, indices)),
                Self::Int(t) => Self::Int(B::int_gather(case.axis, t, indices)),
                Self::Bool(t) => Self::Bool(B::bool_gather(case.axis, t, indices)),
            },
            Operation::Select => unary!(self, float_select, int_select, bool_select, case.axis, indices),
            Operation::GatherNd => match self {
                Self::Float(t) => Self::Float(B::float_gather_nd(t, indices)),
                Self::Int(t) => Self::Int(B::int_gather_nd(t, indices)),
                Self::Bool(_) => unreachable!(),
            },
            op => match (self, values.unwrap()) {
                (Self::Float(t), Self::Float(v)) => Self::Float(match op {
                    Operation::ScatterAssign => B::float_scatter(case.axis, t, indices, v, IndexingUpdateOp::Assign),
                    Operation::SelectAssign => B::float_select_assign(t, case.axis, indices, v, IndexingUpdateOp::Assign),
                    Operation::ScatterNdAssign => B::float_scatter_nd(t, indices, v, IndexingUpdateOp::Assign),
                    _ => unreachable!(),
                }),
                (Self::Int(t), Self::Int(v)) => Self::Int(match op {
                    Operation::ScatterAssign => B::int_scatter(case.axis, t, indices, v, IndexingUpdateOp::Assign),
                    Operation::SelectAssign => B::int_select_assign(t, case.axis, indices, v, IndexingUpdateOp::Assign),
                    Operation::ScatterNdAssign => B::int_scatter_nd(t, indices, v, IndexingUpdateOp::Assign),
                    _ => unreachable!(),
                }),
                _ => unreachable!(),
            },
        }
    }
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub dtypes: usize,
    pub index_dtypes: usize,
    /// Two ND operations, each absent for both Boolean storage formats.
    pub unsupported_bool_nd: usize,
    pub gather: usize,
    pub select: usize,
    pub gather_nd: usize,
    pub scatter_assign: usize,
    pub select_assign: usize,
    pub scatter_nd_assign: usize,
    pub argmin: usize,
    pub argmax: usize,
    pub sort_pairs: usize,
    pub topk_pairs: usize,
    pub argsort: usize,
    pub argtopk: usize,
    pub argwhere: usize,
    pub source_checks: usize,
    pub values: usize,
}
impl Summary {
    pub fn cases(&self) -> usize {
        self.gather
            + self.select
            + self.gather_nd
            + self.scatter_assign
            + self.select_assign
            + self.scatter_nd_assign
            + self.argmin
            + self.argmax
            + self.sort_pairs
            + self.topk_pairs
            + self.argsort
            + self.argtopk
            + self.argwhere
    }
    fn indexed(&mut self, case: &Case) {
        *match case.op {
            Operation::Gather => &mut self.gather,
            Operation::Select => &mut self.select,
            Operation::GatherNd => &mut self.gather_nd,
            Operation::ScatterAssign => &mut self.scatter_assign,
            Operation::SelectAssign => &mut self.select_assign,
            Operation::ScatterNdAssign => &mut self.scatter_nd_assign,
        } += 1;
        self.values += case.expected.len();
        self.source_checks += 4 + usize::from(case.assignment());
    }
}
async fn run_case<B: BurnBackend>(device: &B::Device, case: &Case, summary: &mut Summary) -> Result<(), String> {
    let root = Stored::<B>::new(device, case.dtype, &case.source.root);
    let source = root.clone().views(&case.source.views);
    let index_root = Stored::<B>::new(device, case.index_dtype.into(), &case.indices.root);
    let index = index_root.clone().views(&case.indices.views);
    let source_records = case.source.logical();
    let (values, value_records) = if case.assignment() {
        (
            Some(source.clone().views(&case.value_views)),
            Some(case.value_views.iter().fold(source_records.clone(), |data, view| view.reference(&data))),
        )
    } else {
        (None, None)
    };
    source
        .clone()
        .indexed(case, index.clone().int(), values.clone())
        .check(case.dtype, &case.expected)
        .await?;
    root.check(case.dtype, &case.source.root).await?;
    source.check(case.dtype, &source_records).await?;
    index_root.check(case.index_dtype.into(), &case.indices.root).await?;
    index.check(case.index_dtype.into(), &case.indices.logical()).await?;
    if let (Some(value), Some(records)) = (values, value_records) {
        value.check(case.dtype, &records).await?;
    }
    summary.indexed(case);
    Ok(())
}

/// CPU comparison uses finite, distinct F32 or signed I32 data. Original value
/// bytes are copied after sorting coordinates, never reconstructed from floats.
pub fn ordered_reference(data: &Records, dtype: DType, axis: usize, descending: bool, k: usize, index_dtype: IntDType) -> (Records, Records) {
    assert!(matches!(dtype, DType::F32 | DType::I32));
    assert!(k <= data.shape[axis]);
    let mut shape = data.shape.clone();
    shape[axis] = k;
    let mut bytes = Vec::new();
    let mut indices = Vec::new();
    for c in coordinates(&shape) {
        let mut lane: Vec<_> = (0..data.shape[axis]).collect();
        let record = |i| {
            let mut source = c.clone();
            source[axis] = i;
            u32::from_le_bytes(data.at(&source).try_into().unwrap())
        };
        lane.sort_by(|&a, &b| {
            let (a_bits, b_bits) = (record(a), record(b));
            let cmp = if dtype == DType::F32 {
                f32::from_bits(a_bits).total_cmp(&f32::from_bits(b_bits))
            } else {
                (a_bits as i32).cmp(&(b_bits as i32))
            };
            (if descending { cmp.reverse() } else { cmp }).then(a.cmp(&b))
        });
        let chosen = lane[c[axis]];
        let mut source = c;
        source[axis] = chosen;
        bytes.extend_from_slice(data.at(&source));
        indices.push(chosen);
    }
    (Records::new(&shape, data.width, &bytes), index_records(&shape, index_dtype, &indices))
}
pub fn argwhere_reference(data: &Records, index_dtype: IntDType) -> Records {
    let positions: Vec<_> = coordinates(&data.shape).into_iter().filter(|c| data.at(c).iter().any(|&b| b != 0)).collect();
    index_records(
        &[positions.len(), data.shape.len()],
        index_dtype,
        &positions.into_iter().flatten().collect::<Vec<_>>(),
    )
}
#[derive(Clone, Copy, Debug)]
enum OrderingOp {
    Argmin,
    Argmax,
    SortPairs,
    TopkPairs,
    Argsort,
    Argtopk,
    Argwhere,
}
struct OrderingCase {
    label: String,
    op: OrderingOp,
    dtype: DType,
    out_dtype: IntDType,
    axis: usize,
    descending: bool,
    k: usize,
    source: Input,
    expected: Vec<(DType, Records)>,
}
impl OrderingCase {
    fn new(op: OrderingOp, dtype: DType, out_dtype: IntDType, axis: usize, descending: bool, k: usize, source: Input) -> Self {
        let data = source.logical();
        let expected = if matches!(op, OrderingOp::Argwhere) {
            vec![(out_dtype.into(), argwhere_reference(&data, out_dtype))]
        } else {
            let (values, indices) = ordered_reference(&data, dtype, axis, descending, k, out_dtype);
            match op {
                OrderingOp::SortPairs | OrderingOp::TopkPairs => vec![(dtype, values), (out_dtype.into(), indices)],
                _ => vec![(out_dtype.into(), indices)],
            }
        };
        Self {
            label: format!("shape={:?} axis={axis} descending={descending} k={k}", data.shape),
            op,
            dtype,
            out_dtype,
            axis,
            descending,
            k,
            source,
            expected,
        }
    }
}
fn ordering_cases() -> Vec<OrderingCase> {
    let mut result = Vec::new();
    // Axis lengths stay <= 127, so every requested index is representable even
    // in I8. Tail lengths elsewhere still exercise 128/129/257 element buffers.
    let mut shapes = vec![vec![1], vec![127], vec![2, 7, 3], vec![129, 7], vec![257, 3]];
    for rank in 3..=8 {
        let mut shape = vec![1; rank];
        shape[0] = 2;
        shape[rank / 2] = 3;
        shape[rank - 1] = 7;
        shapes.push(shape);
    }
    for dtype in [DType::F32, DType::I32] {
        for shape in &shapes {
            let count = shape.iter().product::<usize>();
            let words: Vec<_> = (0..count)
                .map(|i| {
                    // The odd multiplier permutes these bounded power-of-two bins,
                    // making each value distinct with varied signed ordering.
                    let n = ((i * 73 + 19) % 2048) as i32 - 1024;
                    if dtype == DType::F32 {
                        u64::from((n as f32 / 8.).to_bits())
                    } else {
                        u64::from((n * 2_000_003) as u32)
                    }
                })
                .collect();
            let source = Input::strided(Records::new(shape, 4, &encoded(dtype, &words)));
            for (axis, &size) in shape.iter().enumerate() {
                if size > 127 {
                    continue;
                }
                let outputs: &[IntDType] = if dtype == DType::F32 { &INDEX_DTYPES } else { &[IntDType::I32] };
                for &out in outputs {
                    for (op, desc) in [(OrderingOp::Argmin, false), (OrderingOp::Argmax, true)] {
                        result.push(OrderingCase::new(op, dtype, out, axis, desc, 1, source.clone()));
                    }
                    for descending in [false, true] {
                        for op in [OrderingOp::SortPairs, OrderingOp::Argsort] {
                            result.push(OrderingCase::new(op, dtype, out, axis, descending, size, source.clone()));
                        }
                    }
                    let mut counts = vec![0, 1, 3.min(size), size];
                    counts.sort_unstable();
                    counts.dedup();
                    for k in counts {
                        for op in [OrderingOp::TopkPairs, OrderingOp::Argtopk] {
                            result.push(OrderingCase::new(op, dtype, out, axis, true, k, source.clone()));
                        }
                    }
                }
            }
        }
    }
    for bool_dtype in [BoolDType::U8, BoolDType::U32] {
        let dtype = DType::Bool(bool_dtype);
        for rank in 1..=8 {
            let mut shape = vec![1; rank];
            shape[rank - 1] = 7;
            if rank > 1 {
                shape[0] = 3;
            }
            for variant in 0..3 {
                let values: Vec<_> = (0..shape.iter().product())
                    .map(|i: usize| {
                        u64::from(match variant {
                            0 => false,
                            1 => true,
                            _ => i % 3 == 1 || i.is_multiple_of(7),
                        })
                    })
                    .collect();
                let source = Input::strided(Records::new(&shape, dtype.size(), &encoded(dtype, &values)));
                for out_dtype in INDEX_DTYPES {
                    result.push(OrderingCase::new(OrderingOp::Argwhere, dtype, out_dtype, 0, false, 0, source.clone()));
                }
            }
        }
    }
    result
}
impl Summary {
    fn ordering(&mut self, case: &OrderingCase) {
        *match case.op {
            OrderingOp::Argmin => &mut self.argmin,
            OrderingOp::Argmax => &mut self.argmax,
            OrderingOp::SortPairs => &mut self.sort_pairs,
            OrderingOp::TopkPairs => &mut self.topk_pairs,
            OrderingOp::Argsort => &mut self.argsort,
            OrderingOp::Argtopk => &mut self.argtopk,
            OrderingOp::Argwhere => &mut self.argwhere,
        } += 1;
        self.values += case.expected.iter().map(|(_, records)| records.len()).sum::<usize>();
        self.source_checks += 2;
    }
}
async fn run_ordering<B: BurnBackend>(device: &B::Device, case: &OrderingCase, summary: &mut Summary) -> Result<(), String> {
    let root = Stored::<B>::new(device, case.dtype, &case.source.root);
    let source = root.clone().views(&case.source.views);
    let output: Vec<Stored<B>> = match source.clone() {
        Stored::Float(t) => match case.op {
            OrderingOp::Argmin => vec![Stored::Int(B::float_argmin(t, case.axis, case.out_dtype))],
            OrderingOp::Argmax => vec![Stored::Int(B::float_argmax(t, case.axis, case.out_dtype))],
            OrderingOp::SortPairs => {
                let (v, i) = B::float_sort_with_indices(t, case.axis, case.descending, case.out_dtype);
                vec![Stored::Float(v), Stored::Int(i)]
            }
            OrderingOp::TopkPairs => {
                let (v, i) = B::float_topk_with_indices(t, case.axis, case.k, case.out_dtype);
                vec![Stored::Float(v), Stored::Int(i)]
            }
            OrderingOp::Argsort => vec![Stored::Int(B::float_argsort(t, case.axis, case.descending, case.out_dtype))],
            OrderingOp::Argtopk => vec![Stored::Int(B::float_argtopk(t, case.axis, case.k, case.out_dtype))],
            OrderingOp::Argwhere => unreachable!(),
        },
        Stored::Int(t) => match case.op {
            OrderingOp::Argmin => vec![Stored::Int(B::int_argmin(t, case.axis))],
            OrderingOp::Argmax => vec![Stored::Int(B::int_argmax(t, case.axis))],
            OrderingOp::SortPairs => {
                let (v, i) = B::int_sort_with_indices(t, case.axis, case.descending);
                vec![Stored::Int(v), Stored::Int(i)]
            }
            OrderingOp::TopkPairs => {
                let (v, i) = B::int_topk_with_indices(t, case.axis, case.k);
                vec![Stored::Int(v), Stored::Int(i)]
            }
            OrderingOp::Argsort => vec![Stored::Int(B::int_argsort(t, case.axis, case.descending))],
            OrderingOp::Argtopk => vec![Stored::Int(B::int_argtopk(t, case.axis, case.k))],
            OrderingOp::Argwhere => unreachable!(),
        },
        Stored::Bool(t) => vec![Stored::Int(B::bool_argwhere(t, case.out_dtype).await)],
    };
    assert_eq!(output.len(), case.expected.len());
    for (actual, (dtype, expected)) in output.into_iter().zip(&case.expected) {
        actual.check(*dtype, expected).await?;
    }
    root.check(case.dtype, &case.source.root).await?;
    source.check(case.dtype, &case.source.logical()).await?;
    summary.ordering(case);
    Ok(())
}

/// Pure CPU totals traverse the exact same case generators as backend execution.
/// Output values include both tensors from ordering pairs, but exclude reads of
/// retained sources. This function neither creates a device nor initializes one.
pub fn expected_summary(options: &Options) -> Summary {
    let mut summary = Summary::default();
    for dtype in DTYPES {
        if dtype == DType::Flex32 && !options.include_flex32 {
            continue;
        }
        summary.dtypes += 1;
        for index_dtype in INDEX_DTYPES {
            for case in indexed_cases(dtype, index_dtype, options) {
                summary.indexed(&case);
            }
        }
    }
    summary.index_dtypes = INDEX_DTYPES.len();
    summary.unsupported_bool_nd = 4;
    for case in ordering_cases() {
        summary.ordering(&case);
    }
    summary
}
/// Run the corpus on any backend without routing indices through host memory.
pub async fn run_backend<B: BurnBackend>(device: &B::Device, options: &Options) -> Result<Summary, String> {
    let mut summary = Summary::default();
    for dtype in DTYPES {
        if dtype == DType::Flex32 && !options.include_flex32 {
            continue;
        }
        for index_dtype in INDEX_DTYPES {
            for case in indexed_cases(dtype, index_dtype, options) {
                run_case::<B>(device, &case, &mut summary)
                    .await
                    .map_err(|e| format!("{dtype:?} index={index_dtype:?} {:?} {}: {e}", case.op, case.label))?;
            }
        }
        summary.dtypes += 1;
    }
    summary.index_dtypes = INDEX_DTYPES.len();
    summary.unsupported_bool_nd = 4;
    for case in ordering_cases() {
        run_ordering::<B>(device, &case, &mut summary)
            .await
            .map_err(|e| format!("{:?} out={:?} {:?} {}: {e}", case.dtype, case.out_dtype, case.op, case.label))?;
    }
    let expected = expected_summary(options);
    if summary != expected {
        return Err(format!("corpus totals {summary:?}, expected {expected:?}"));
    }
    Ok(summary)
}
/// Public Dispatch adapter shared by the Linux runner and Native command.
pub async fn run(device: &Device, options: &Options) -> Result<Summary, String> {
    run_backend::<Dispatch>(device.as_dispatch(), options).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logical_shape(input: &Input) -> Vec<usize> {
        input.views.iter().fold(input.root.shape.clone(), |shape, view| match view {
            View::Slice { shape, .. } | View::Reshape(shape) => shape.clone(),
            View::Flip(_) => shape,
            View::Permute(axes) => axes.iter().map(|&axis| shape[axis]).collect(),
        })
    }

    #[test]
    fn generated_nd_cases_follow_public_shape_contract_and_cover_empty_slices() {
        for dtype in DTYPES {
            for index_dtype in INDEX_DTYPES {
                let mut empty_gather = 0;
                let mut empty_scatter = 0;
                for case in indexed_cases(dtype, index_dtype, &Options::default()) {
                    if !matches!(case.op, Operation::GatherNd | Operation::ScatterNdAssign) {
                        continue;
                    }
                    assert!(!dtype.is_bool());
                    let data_shape = logical_shape(&case.source);
                    let index = case.indices.logical();
                    assert!(!index.shape.is_empty());
                    assert!(!index.is_empty());
                    let width = *index.shape.last().unwrap();
                    assert!(width > 0 && width <= data_shape.len());
                    let ids = decoded_indices(&index);
                    for tuple in ids.chunks_exact(width) {
                        assert!(tuple.iter().zip(&data_shape).all(|(&i, &size)| i < size));
                    }
                    let mut output_shape = index.shape[..index.shape.len() - 1].to_vec();
                    output_shape.extend_from_slice(&data_shape[width..]);
                    if matches!(case.op, Operation::GatherNd) {
                        assert_eq!(case.expected.shape, output_shape);
                        if case.expected.is_empty() {
                            empty_gather += 1;
                        }
                    } else {
                        let value_shape = case.value_views.iter().fold(data_shape.clone(), |shape, view| match view {
                            View::Slice { shape, .. } | View::Reshape(shape) => shape.clone(),
                            View::Flip(_) => shape,
                            View::Permute(axes) => axes.iter().map(|&axis| shape[axis]).collect(),
                        });
                        assert_eq!(value_shape, output_shape);
                        assert_eq!(case.expected.shape, data_shape);
                        if case.expected.is_empty() {
                            empty_scatter += 1;
                        }
                    }
                }
                if dtype.is_bool() {
                    assert_eq!((empty_gather, empty_scatter), (0, 0));
                } else {
                    assert_eq!((empty_gather, empty_scatter), (12, 12));
                }
            }
        }
    }
}
