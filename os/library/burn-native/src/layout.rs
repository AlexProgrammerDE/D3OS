//! Checked tensor views. Offsets and strides count storage elements, not bytes.
use alloc::{vec, vec::Vec};
use burn_backend::{Shape, Slice};
use d3os_compute::{Error, Result};
use d3os_nvidia::compute::elementwise::{self, Elementwise};
use d3os_nvidia::compute::indexing;
use d3os_nvidia::compute::reduction::{self, Reduction};
use d3os_nvidia::compute::scan::{self, Scan};
use d3os_nvidia::compute::sort::{self, Mode as SortMode};
use d3os_nvidia::compute::words::{self, WordOp};

/// Native kernels use signed 32-bit element indices, including reversed views.
const INDEX_MAX: usize = i32::MAX as usize;
pub const MAX_RANK: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Vec<i64>,
    offset: i64,
    storage_elements: usize,
    elements: usize,
}

fn count(shape: &[usize]) -> Result<usize> {
    if shape.len() > MAX_RANK || shape.iter().any(|&dim| dim > INDEX_MAX) {
        return Err(Error::InvalidShape);
    }
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d).filter(|&n| n <= INDEX_MAX).ok_or(Error::SizeOverflow))
}

impl Layout {
    pub fn contiguous(shape: Shape) -> Result<Self> {
        let elements = count(shape.as_slice())?;
        let mut strides = vec![0; shape.num_dims()];
        if elements == 0 {
            return Self::from_parts(shape, strides, 0, 0);
        }
        let mut stride = 1i64;
        for (&dim, out) in shape.as_slice().iter().zip(&mut strides).rev() {
            *out = stride;
            stride = stride.checked_mul(dim as i64).ok_or(Error::SizeOverflow)?;
        }
        Self::from_parts(shape, strides, 0, elements)
    }

    pub fn from_parts(shape: Shape, strides: Vec<i64>, offset: i64, storage_elements: usize) -> Result<Self> {
        let elements = count(shape.as_slice())?;
        if strides.len() != shape.num_dims() || storage_elements > INDEX_MAX || strides.iter().any(|&s| i32::try_from(s).is_err()) {
            return Err(Error::InvalidShape);
        }
        let result = Self {
            shape,
            strides,
            offset,
            storage_elements,
            elements,
        };
        result.validate_bounds()?;
        Ok(result)
    }

    fn validate_bounds(&self) -> Result<()> {
        if self.offset < 0 || self.offset as u64 > self.storage_elements as u64 {
            return Err(Error::ShapeMismatch);
        }
        if self.is_empty() {
            return Ok(());
        }
        let mut lower = i128::from(self.offset);
        let mut upper = lower;
        for (&dim, &stride) in self.shape.as_slice().iter().zip(&self.strides) {
            let span = (dim - 1) as i128 * i128::from(stride);
            lower += span.min(0);
            upper += span.max(0);
        }
        if lower < 0 || upper >= self.storage_elements as i128 {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }
    pub fn offset(&self) -> usize {
        self.offset as usize
    }
    pub fn storage_elements(&self) -> usize {
        self.storage_elements
    }
    pub fn len(&self) -> usize {
        self.elements
    }
    pub fn is_empty(&self) -> bool {
        self.elements == 0
    }

    pub fn is_contiguous(&self) -> bool {
        if self.is_empty() {
            return true;
        }
        let mut expected = 1i64;
        for (&dim, &stride) in self.shape.as_slice().iter().zip(&self.strides).rev() {
            if dim != 1 && stride != expected {
                return false;
            }
            expected *= dim as i64;
        }
        true
    }

    /// Address one element in logical row-major order, including arbitrary views.
    pub fn element_offset(&self, mut linear: usize) -> Result<usize> {
        if linear >= self.elements {
            return Err(Error::InvalidParameter);
        }
        let mut address = self.offset;
        for (&dim, &stride) in self.shape.as_slice().iter().zip(&self.strides).rev() {
            address += (linear % dim) as i64 * stride;
            linear /= dim;
        }
        usize::try_from(address).map_err(|_| Error::ShapeMismatch)
    }

    pub fn reshape(&self, shape: Shape) -> Result<Self> {
        if count(shape.as_slice())? != self.elements {
            return Err(Error::ShapeMismatch);
        }
        if !self.is_contiguous() {
            return Err(Error::InvalidParameter);
        }
        let mut layout = Self::contiguous(shape)?;
        layout.offset = self.offset;
        layout.storage_elements = self.storage_elements;
        layout.validate_bounds()?;
        Ok(layout)
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        if axes.len() != self.strides.len() {
            return Err(Error::InvalidParameter);
        }
        let mut seen = 0u16;
        for &axis in axes {
            if axis >= axes.len() || seen & (1 << axis) != 0 {
                return Err(Error::InvalidParameter);
            }
            seen |= 1 << axis;
        }
        Self::from_parts(
            Shape::from(axes.iter().map(|&a| self.shape.as_slice()[a]).collect::<Vec<_>>()),
            axes.iter().map(|&a| self.strides[a]).collect(),
            self.offset,
            self.storage_elements,
        )
    }

    pub fn swap_dims(&self, first: usize, second: usize) -> Result<Self> {
        if first >= self.strides.len() || second >= self.strides.len() {
            return Err(Error::InvalidParameter);
        }
        let mut axes: Vec<_> = (0..self.strides.len()).collect();
        axes.swap(first, second);
        self.permute(&axes)
    }

    pub fn flip(&self, axes: &[usize]) -> Result<Self> {
        let mut layout = self.clone();
        let mut seen = 0u16;
        for &axis in axes {
            if axis >= self.strides.len() || seen & (1 << axis) != 0 {
                return Err(Error::InvalidParameter);
            }
            seen |= 1 << axis;
            if !self.is_empty() {
                layout.offset += (layout.shape.as_slice()[axis] - 1) as i64 * layout.strides[axis];
            }
            layout.strides[axis] = -layout.strides[axis];
        }
        Self::from_parts(layout.shape, layout.strides, layout.offset, layout.storage_elements)
    }

    pub fn expand(&self, shape: Shape) -> Result<Self> {
        count(shape.as_slice())?;
        if shape.num_dims() < self.shape.num_dims() {
            return Err(Error::ShapeMismatch);
        }
        let pad = shape.num_dims() - self.shape.num_dims();
        let mut strides = vec![0; shape.num_dims()];
        for (axis, stride) in strides.iter_mut().enumerate().skip(pad) {
            let old = self.shape.as_slice()[axis - pad];
            if old != 1 && old != shape.as_slice()[axis] {
                return Err(Error::ShapeMismatch);
            }
            if old != 1 {
                *stride = self.strides[axis - pad];
            }
        }
        Self::from_parts(shape, strides, self.offset, self.storage_elements)
    }

    pub fn slice(&self, slices: &[Slice]) -> Result<Self> {
        if slices.len() > self.strides.len() || slices.iter().any(|s| s.step == 0) {
            return Err(Error::InvalidParameter);
        }
        let mut dimensions = self.shape.as_slice().to_vec();
        let mut strides = self.strides.clone();
        let mut offset = self.offset;
        for (axis, slice) in slices.iter().enumerate() {
            let range = slice.to_range(dimensions[axis]);
            if range.start > range.end {
                return Err(Error::InvalidParameter);
            }
            dimensions[axis] = slice.output_size(dimensions[axis]);
            if dimensions[axis] != 0 {
                let first = if slice.is_reversed() { range.end - 1 } else { range.start };
                offset += first as i64 * strides[axis];
            }
            strides[axis] = strides[axis].checked_mul(slice.step as i64).ok_or(Error::SizeOverflow)?;
        }
        if dimensions.contains(&0) {
            offset = offset.clamp(0, self.storage_elements as i64);
        }
        Self::from_parts(Shape::from(dimensions), strides, offset, self.storage_elements)
    }

    pub fn unfold(&self, dim: usize, size: usize, step: usize) -> Result<Self> {
        if dim >= self.strides.len() || size == 0 || step == 0 || size > self.shape.as_slice()[dim] {
            return Err(Error::InvalidParameter);
        }
        let mut dimensions = self.shape.as_slice().to_vec();
        let mut strides = self.strides.clone();
        dimensions[dim] = (dimensions[dim] - size) / step + 1;
        dimensions.push(size);
        let stride = strides[dim];
        strides[dim] = stride.checked_mul(step as i64).ok_or(Error::SizeOverflow)?;
        strides.push(stride);
        Self::from_parts(Shape::from(dimensions), strides, self.offset, self.storage_elements)
    }

    /// Encode two equally shaped views into the checked native launch protocol.
    pub fn elementwise_parameters(&self, rhs: &Self, op: Elementwise, scalar_bits: u32, scalar_rhs: bool) -> Result<[usize; 32]> {
        let parameters = self.strided_parameters(rhs, op as u32, scalar_bits, scalar_rhs)?;
        elementwise::validate(&parameters).map_err(|_| Error::InvalidParameter)?;
        Ok(parameters.map(|value| value as usize))
    }

    pub fn word_parameters(&self, rhs: &Self, op: WordOp, scalar_bits: u32, scalar_rhs: bool) -> Result<[usize; 32]> {
        let parameters = self.strided_parameters(rhs, op as u32, scalar_bits, scalar_rhs)?;
        words::validate(&parameters).map_err(|_| Error::InvalidParameter)?;
        Ok(parameters.map(|value| value as usize))
    }

    /// Encode a partial write with a provably nonoverlapping destination view.
    pub fn assignment_parameters(&self, source: &Self) -> Result<[usize; 32]> {
        let parameters = self.strided_parameters(source, 0, 0, false)?;
        d3os_nvidia::compute::updates::validate_assign(&parameters).map_err(|_| Error::InvalidParameter)?;
        Ok(parameters.map(|value| value as usize))
    }

    fn strided_parameters(&self, rhs: &Self, operation: u32, scalar_bits: u32, scalar_rhs: bool) -> Result<[u32; 32]> {
        if self.shape != rhs.shape {
            return Err(Error::ShapeMismatch);
        }
        let mut parameters = [0u32; 32];
        parameters[..8].copy_from_slice(&[
            self.elements as u32,
            self.shape.num_dims().max(1) as u32,
            operation | if scalar_rhs { elementwise::SCALAR_RHS } else { 0 },
            scalar_bits,
            self.offset as u32,
            rhs.offset as u32,
            self.storage_elements as u32,
            rhs.storage_elements as u32,
        ]);
        parameters[8..16].fill(1);
        for (axis, &dim) in self.shape.iter().enumerate() {
            parameters[8 + axis] = dim as u32;
            parameters[16 + axis] = self.strides[axis] as i32 as u32;
            parameters[24 + axis] = rhs.strides[axis] as i32 as u32;
        }
        Ok(parameters)
    }

    /// Encode logical element geometry with a packed physical word contract.
    pub fn storage_parameters(&self, width: d3os_nvidia::compute::storage::Width, mode: d3os_nvidia::compute::storage::Mode, bits: u64) -> Result<[usize; 32]> {
        use d3os_nvidia::compute::storage::{self, Mode};
        let mut p = [0u32; 32];
        p[..8].copy_from_slice(&[
            self.len() as u32,
            self.shape.num_dims().max(1) as u32,
            width as u32 | ((mode as u32) << 8),
            bits as u32,
            self.offset as u32,
            (bits >> 32) as u32,
            self.storage_elements as u32,
            0,
        ]);
        p[8..16].fill(1);
        for (axis, &dim) in self.shape.iter().enumerate() {
            p[8 + axis] = dim as u32;
            p[16 + axis] = self.strides[axis] as i32 as u32;
        }
        if mode == Mode::Fill && self.shape.num_dims() == 0 {
            p[16] = 1;
        }
        storage::validate(&p).map_err(|_| Error::InvalidParameter)?;
        Ok(p.map(|v| v as usize))
    }

    pub fn reduction_shape(&self, axis: usize) -> Result<Shape> {
        let mut shape = self.shape.to_vec();
        *shape.get_mut(axis).ok_or(Error::InvalidParameter)? = 1;
        count(&shape)?;
        Ok(Shape::from(shape))
    }

    pub fn gather_shape(&self, indices: &Self, axis: usize) -> Result<Shape> {
        if axis >= self.shape.num_dims()
            || indices.shape.num_dims() != self.shape.num_dims()
            || self.shape.iter().zip(indices.shape.iter()).enumerate().any(|(a, (&s, &i))| a != axis && s != i)
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(indices.shape.clone())
    }

    pub fn gather_parameters(&self, indices: &Self, axis: usize) -> Result<[usize; 32]> {
        let shape = self.gather_shape(indices, axis)?;
        let mut p = [0u32; 32];
        p[..8].copy_from_slice(&[
            shape.num_dims() as u32,
            axis as u32,
            indices.len() as u32,
            self.storage_elements.max(1) as u32,
            self.offset as u32,
            indices.storage_elements.max(1) as u32,
            indices.offset as u32,
            self.shape[axis] as u32,
        ]);
        p[8..16].fill(1);
        for (a, &dim) in shape.iter().enumerate() {
            p[8 + a] = dim as u32;
            p[16 + a] = self.strides[a] as i32 as u32;
            p[24 + a] = indices.strides[a] as i32 as u32;
        }
        indexing::validate(&p).map_err(|_| Error::InvalidParameter)?;
        Ok(p.map(|v| v as usize))
    }

    pub fn scatter_parameters(&self, indices: &Self, axis: usize, operation: indexing::Scatter) -> Result<[usize; 32]> {
        let mut p = self.gather_parameters(indices, axis)?.map(|v| v as u32);
        p[0] |= (operation as u32) << 16;
        indexing::validate_scatter(&p).map_err(|_| Error::InvalidParameter)?;
        Ok(p.map(|v| v as usize))
    }

    /// Encode one-axis reductions without materializing strided input views.
    pub fn reduction_parameters(&self, axis: usize, op: Reduction) -> Result<[usize; 24]> {
        let shape = self.reduction_shape(axis)?;
        let mut parameters = [0u32; 24];
        parameters[..8].copy_from_slice(&[
            count(shape.as_slice())? as u32,
            self.shape.num_dims() as u32,
            op as u32,
            axis as u32,
            self.offset as u32,
            self.storage_elements.max(1) as u32,
            self.shape[axis] as u32,
            0,
        ]);
        parameters[8..16].fill(1);
        for (axis, &dim) in self.shape.iter().enumerate() {
            parameters[8 + axis] = dim as u32;
            parameters[16 + axis] = self.strides[axis] as i32 as u32;
        }
        reduction::validate(&parameters).map_err(|_| Error::InvalidParameter)?;
        Ok(parameters.map(|value| value as usize))
    }

    pub fn scan_parameters(&self, axis: usize, op: Scan) -> Result<[usize; 24]> {
        let mut parameters = self.reduction_parameters(axis, op.reduction())?.map(|v| v as u32);
        parameters[2] = op as u32;
        parameters[7] = self.shape[axis + 1..].iter().product::<usize>() as u32;
        scan::validate(&parameters).map_err(|_| Error::InvalidParameter)?;
        Ok(parameters.map(|v| v as usize))
    }

    pub fn sort_parameters(&self, axis: usize, mode: SortMode, width: usize) -> Result<[usize; 32]> {
        let scan = self.scan_parameters(axis, Scan::Sum)?;
        let mut p = [0u32; 32];
        p[..24].copy_from_slice(&scan.map(|v| v as u32));
        p[0] = self.len() as u32;
        p[2] = mode as u32;
        p[24] = u32::try_from(width).map_err(|_| Error::InvalidParameter)?;
        if width == 0 {
            return Err(Error::InvalidParameter);
        }
        p[25] = p[24].ilog2() + 1;
        sort::validate(&p).map_err(|_| Error::InvalidParameter)?;
        Ok(p.map(|v| v as usize))
    }
}

/// NumPy-style broadcasting, including zero-length dimensions.
pub fn broadcast(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    count(lhs.as_slice())?;
    count(rhs.as_slice())?;
    let rank = lhs.num_dims().max(rhs.num_dims());
    let mut dimensions = vec![1; rank];
    for (axis, dim) in dimensions.iter_mut().enumerate() {
        let left = axis.checked_sub(rank - lhs.num_dims()).map_or(1, |a| lhs.as_slice()[a]);
        let right = axis.checked_sub(rank - rhs.num_dims()).map_or(1, |a| rhs.as_slice()[a]);
        if left == right {
            *dim = left;
        } else if left == 1 {
            *dim = right;
        } else if right == 1 {
            *dim = left;
        } else {
            return Err(Error::ShapeMismatch);
        }
    }
    count(&dimensions)?;
    Ok(Shape::from(dimensions))
}

/// Broadcast batch dimensions and retain the two matrix output dimensions.
pub fn matmul_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    count(lhs.as_slice())?;
    count(rhs.as_slice())?;
    let (left_rank, right_rank) = (lhs.num_dims(), rhs.num_dims());
    if left_rank < 2 || right_rank < 2 || lhs[left_rank - 1] != rhs[right_rank - 2] {
        return Err(Error::ShapeMismatch);
    }
    let rank = left_rank.max(right_rank);
    let mut dimensions = vec![1; rank];
    for (axis, dim) in dimensions.iter_mut().enumerate().take(rank - 2) {
        let left = axis.checked_sub(rank - left_rank).map_or(1, |a| lhs[a]);
        let right = axis.checked_sub(rank - right_rank).map_or(1, |a| rhs[a]);
        *dim = if left == right || right == 1 {
            left
        } else if left == 1 {
            right
        } else {
            return Err(Error::ShapeMismatch);
        };
    }
    dimensions[rank - 2] = lhs[left_rank - 2];
    dimensions[rank - 1] = rhs[right_rank - 1];
    count(&dimensions)?;
    Ok(Shape::from(dimensions))
}

/// Shape of ND gather output and ND scatter values. Empty index tensors are
/// rejected by Burn's public contract; selecting a single point is a scalar.
pub fn nd_shape(data: &Shape, indices: &Shape) -> Result<Shape> {
    count(data.as_slice())?;
    let index_count = count(indices.as_slice())?;
    let Some(&width) = indices.last() else {
        return Err(Error::InvalidShape);
    };
    if index_count == 0 || width == 0 || width > data.num_dims() {
        return Err(Error::InvalidShape);
    }
    let mut dims = indices[..indices.num_dims() - 1].to_vec();
    dims.extend_from_slice(&data[width..]);
    count(&dims)?;
    Ok(Shape::from(dims))
}
