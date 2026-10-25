//! Resident predicates with packed canonical boolean results.
use crate::{
    NativeTensor,
    layout::{Layout, broadcast},
    storage::{format, scalar_bits},
};
use alloc::sync::Arc;
use burn_backend::{BoolStore, DType, Scalar};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{
    Kernel,
    elementwise::Elementwise,
    predicate::{self as abi, Predicate},
    words::WordOp,
};
fn boolean(dtype: DType) -> Result<bool> {
    match dtype {
        DType::Bool(BoolStore::U8) => Ok(true),
        DType::Bool(BoolStore::U32) => Ok(false),
        _ => Err(Error::InvalidParameter),
    }
}
impl NativeTensor {
    pub fn compare(&self, rhs: &Self, op: Predicate, dtype: DType) -> Result<Self> {
        if !op.binary() || self.dtype != rhs.dtype {
            return Err(Error::InvalidParameter);
        }
        self.predicate(rhs, op, dtype, None)
    }
    pub fn compare_scalar(&self, value: Scalar, op: Predicate, dtype: DType) -> Result<Self> {
        if !op.binary() {
            return Err(Error::InvalidParameter);
        }
        self.predicate(self, op, dtype, Some(scalar_bits(self.dtype, value)?))
    }
    pub fn classify(&self, op: Predicate, dtype: DType) -> Result<Self> {
        if op.binary() {
            return Err(Error::InvalidParameter);
        }
        self.predicate(self, op, dtype, None)
    }
    fn predicate(&self, rhs: &Self, op: Predicate, dtype: DType, scalar: Option<u64>) -> Result<Self> {
        let packed = boolean(dtype)?;
        let format = format(self.dtype)?;
        if !op.supports(format) {
            return Err(Error::InvalidParameter);
        }
        // Retain existing one-word fast paths and their established semantics.
        if !packed {
            let elementwise = match op {
                Predicate::Equal => Some(Elementwise::Equal),
                Predicate::Greater => Some(Elementwise::Greater),
                Predicate::GreaterEqual => Some(Elementwise::GreaterEqual),
                Predicate::Less => Some(Elementwise::Less),
                Predicate::LessEqual => Some(Elementwise::LessEqual),
                Predicate::IsNan => Some(Elementwise::IsNan),
                Predicate::IsInfinite => Some(Elementwise::IsInfinite),
                Predicate::IsFinite => Some(Elementwise::IsFinite),
                _ => None,
            };
            if self.dtype == DType::F32
                && let Some(operation) = elementwise
            {
                return if !op.binary() {
                    self.unary(operation)
                } else if let Some(bits) = scalar {
                    self.scalar(operation, f32::from_bits(bits as u32))
                } else {
                    self.binary(rhs, operation)
                };
            }
            let word = match op {
                Predicate::Equal => Some(if self.dtype.is_bool() { WordOp::BoolEqual } else { WordOp::Equal }),
                Predicate::Greater => Some(WordOp::Greater),
                Predicate::GreaterEqual => Some(WordOp::GreaterEqual),
                Predicate::Less => Some(WordOp::Less),
                Predicate::LessEqual => Some(WordOp::LessEqual),
                Predicate::And => Some(WordOp::BoolAnd),
                Predicate::Or => Some(WordOp::BoolOr),
                Predicate::Xor => Some(WordOp::BoolXor),
                Predicate::Not => Some(WordOp::BoolNot),
                _ => None,
            };
            if matches!(self.dtype, DType::I32 | DType::Bool(BoolStore::U32))
                && let Some(operation) = word
            {
                return if !op.binary() {
                    self.word_unary(operation)
                } else if let Some(bits) = scalar {
                    self.word_scalar(operation, bits as u32)
                } else {
                    self.word_binary(rhs, operation)
                };
            }
        }
        let shape = if scalar.is_some() || !op.binary() {
            self.layout.shape().clone()
        } else {
            broadcast(self.layout.shape(), rhs.layout.shape())?
        };
        let layout = Layout::contiguous(shape.clone())?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype,
            });
        }
        let left = self.layout.expand(shape.clone())?;
        let right = rhs.layout.expand(shape)?;
        let mut p = left.word_parameters(&right, WordOp::Equal, 0, false)?;
        p[2] = op as usize | ((format as usize) << 8) | if packed { abi::PACKED_BOOL as usize } else { 0 };
        if let Some(bits) = scalar {
            p[2] |= abi::SCALAR_RHS as usize;
            p[3] = bits as u32 as usize;
            p[5] = (bits >> 32) as usize;
        }
        if scalar.is_some() || !op.binary() {
            if scalar.is_none() {
                p[5] = 0;
            }
            p[7] = 0;
            p[24..32].fill(0);
        }
        let words = if packed { layout.len().div_ceil(4) } else { layout.len() };
        let buffer = Nvidia.produce_words(words, Kernel::TensorPredicate, &[(&self.buffer, 0), (&rhs.buffer, 0)], &p)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype,
        })
    }
}
