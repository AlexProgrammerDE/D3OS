//! Deferred errors for Burn's infallible operation traits. A failed tensor
//! contains metadata and its original error, never a placeholder allocation.
use crate::{NativeDevice, NativeTensor};
use alloc::{format, sync::Arc, vec::Vec};
use burn_backend::{DType, ExecutionError, Shape, TensorData, TensorMetadata};
use core::sync::atomic::{AtomicBool, Ordering};
use d3os_compute::Error;
use spin::Mutex;

#[derive(Clone, Debug)]
pub(super) enum Cause {
    Native(Error),
    Capability(&'static str),
}

#[derive(Debug)]
pub(super) struct Failure {
    operation: &'static str,
    cause: Cause,
    reported: AtomicBool,
}

impl Failure {
    fn is_poisoned(&self) -> bool {
        matches!(self.cause, Cause::Native(Error::DeviceFault))
    }

    fn error(&self) -> ExecutionError {
        let reason = match &self.cause {
            Cause::Native(error) => format!("D3OS native Burn {}: {error}", self.operation),
            Cause::Capability(capability) => format!("D3OS native Burn {} requires unsupported {capability}", self.operation),
        };
        if self.is_poisoned() {
            ExecutionError::device_poisoned(reason)
        } else {
            ExecutionError::with_context(reason)
        }
    }

    pub(super) fn report(&self) -> ExecutionError {
        self.reported.store(true, Ordering::Release);
        self.error()
    }
}

struct Errors {
    pending: Vec<Arc<Failure>>,
    poisoned: Option<Arc<Failure>>,
}

pub(super) struct Context {
    errors: Mutex<Errors>,
}

pub(super) static CONTEXT: Context = Context::new();

impl Context {
    const fn new() -> Self {
        Self {
            errors: Mutex::new(Errors {
                pending: Vec::new(),
                poisoned: None,
            }),
        }
    }

    pub(super) fn poisoned(&self) -> Option<Arc<Failure>> {
        self.errors.lock().poisoned.clone()
    }

    pub(super) fn fail(&self, operation: &'static str, cause: Cause) -> Arc<Failure> {
        let mut errors = self.errors.lock();
        if let Some(failure) = &errors.poisoned {
            return failure.clone();
        }
        let failure = Arc::new(Failure {
            operation,
            cause,
            reported: AtomicBool::new(false),
        });
        errors.pending.retain(|error| !error.reported.load(Ordering::Acquire));
        errors.pending.push(failure.clone());
        if failure.is_poisoned() {
            errors.poisoned = Some(failure.clone());
        }
        failure
    }

    pub(super) fn check(&self) -> Result<(), ExecutionError> {
        let mut errors = self.errors.lock();
        if let Some(failure) = &errors.poisoned {
            return Err(failure.report());
        }
        errors.pending.retain(|failure| !failure.reported.load(Ordering::Acquire));
        if errors.pending.is_empty() {
            return Ok(());
        }
        // Return one root failure at a time. Later independent errors remain
        // observable, and dependent failed tensors share their original root.
        let first = errors.pending.remove(0);
        Err(first.report())
    }
}

/// Burn's native primitive, containing either real device storage or an error.
#[derive(Clone, Debug)]
pub struct BackendTensor {
    shape: Shape,
    dtype: DType,
    value: Result<NativeTensor, Arc<Failure>>,
}

impl BackendTensor {
    pub(super) fn read_value<T>(&self, operation: &'static str, work: impl FnOnce(&NativeTensor) -> d3os_compute::Result<T>) -> Result<T, ExecutionError> {
        let value = self.value.as_ref().map_err(|failure| failure.report())?;
        if let Some(failure) = CONTEXT.poisoned() {
            return Err(failure.report());
        }
        work(value).map_err(|error| CONTEXT.fail(operation, Cause::Native(error)).report())
    }

    pub(super) fn value(&self) -> Option<&NativeTensor> {
        self.value.as_ref().ok()
    }

    pub(super) fn pair(
        operation: &'static str, shapes: [Shape; 2], dtypes: [DType; 2], inputs: &[&Self],
        work: impl FnOnce(&[&NativeTensor]) -> d3os_compute::Result<(NativeTensor, NativeTensor)>,
    ) -> (Self, Self) {
        let [first, second] = Self::outputs(operation, shapes, dtypes, inputs, |values| {
            let (first, second) = work(values)?;
            Ok([first, second])
        });
        (first, second)
    }

    pub(super) fn outputs<const N: usize>(
        operation: &'static str, shapes: [Shape; N], dtypes: [DType; N], inputs: &[&Self],
        work: impl FnOnce(&[&NativeTensor]) -> d3os_compute::Result<[NativeTensor; N]>,
    ) -> [Self; N] {
        let failure = inputs
            .iter()
            .find_map(|tensor| tensor.value.as_ref().err().cloned())
            .or_else(|| CONTEXT.poisoned());
        let result = if let Some(failure) = failure {
            Err(failure)
        } else {
            let values: Vec<_> = inputs.iter().filter_map(|tensor| tensor.value()).collect();
            work(&values)
                .and_then(|outputs| {
                    if outputs
                        .iter()
                        .enumerate()
                        .any(|(index, tensor)| tensor.shape() != shapes[index] || tensor.dtype() != dtypes[index])
                    {
                        Err(Error::ShapeMismatch)
                    } else {
                        Ok(outputs)
                    }
                })
                .map_err(|error| CONTEXT.fail(operation, Cause::Native(error)))
        };
        match result {
            Ok(outputs) => outputs.map(Self::ready),
            Err(failure) => core::array::from_fn(|index| Self::failed(shapes[index].clone(), dtypes[index], failure.clone())),
        }
    }
    pub(super) fn ready(value: NativeTensor) -> Self {
        Self {
            shape: value.shape(),
            dtype: value.dtype(),
            value: Ok(value),
        }
    }

    pub(super) fn failed(shape: Shape, dtype: DType, failure: Arc<Failure>) -> Self {
        Self {
            shape,
            dtype,
            value: Err(failure),
        }
    }

    pub(super) fn build(operation: &'static str, shape: Shape, dtype: DType, work: impl FnOnce() -> d3os_compute::Result<NativeTensor>) -> Self {
        if let Some(failure) = CONTEXT.poisoned() {
            return Self::failed(shape, dtype, failure);
        }
        let result = work().and_then(|value| if value.dtype() == dtype { Ok(value) } else { Err(Error::InvalidParameter) });
        match result {
            Ok(value) => Self::ready(value),
            Err(error) => Self::failed(shape, dtype, CONTEXT.fail(operation, Cause::Native(error))),
        }
    }

    pub(super) fn apply(
        operation: &'static str, shape: Shape, dtype: DType, inputs: &[&Self], work: impl FnOnce(&[&NativeTensor]) -> d3os_compute::Result<NativeTensor>,
    ) -> Self {
        let mut values = Vec::with_capacity(inputs.len());
        for input in inputs {
            match &input.value {
                Ok(value) => values.push(value),
                Err(failure) => return Self::failed(shape, dtype, failure.clone()),
            }
        }
        Self::build(operation, shape, dtype, || work(&values))
    }

    pub(super) fn unsupported(operation: &'static str, shape: Shape, dtype: DType, inputs: &[&Self], capability: &'static str) -> Self {
        for input in inputs {
            if let Err(failure) = &input.value {
                return Self::failed(shape, dtype, failure.clone());
            }
        }
        Self::failed(shape, dtype, CONTEXT.fail(operation, Cause::Capability(capability)))
    }

    pub(super) fn read(self) -> Result<TensorData, ExecutionError> {
        match self.value {
            Err(failure) => Err(failure.report()),
            Ok(value) => value.read().map_err(|error| CONTEXT.fail("readback", Cause::Native(error)).report()),
        }
    }
}

impl TensorMetadata for BackendTensor {
    type Device = NativeDevice;
    fn shape(&self) -> Shape {
        self.shape.clone()
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn device(&self) -> NativeDevice {
        NativeDevice
    }
    fn can_mut(&self) -> bool {
        self.value.as_ref().is_ok_and(TensorMetadata::can_mut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatched_tensor_kinds_fail_before_attempting_a_device_upload() {
        use crate::backend::Native;
        use burn_backend::ops::{BoolTensorOps, FloatTensorOps, IntTensorOps};
        let invalid = [
            Native::float_from_data(TensorData::from([1i32]), &NativeDevice),
            Native::int_from_data(TensorData::from([1f32]), &NativeDevice),
            Native::bool_from_data(TensorData::from([1f32]), &NativeDevice),
        ];
        for tensor in invalid {
            let failure = tensor.value.unwrap_err();
            assert!(matches!(failure.cause, Cause::Native(Error::InvalidParameter)));
            failure.report();
        }
        assert!(CONTEXT.check().is_ok());
    }

    #[test]
    fn failed_tensors_preserve_roots_and_never_execute_dependent_work() {
        let context = Context::new();
        let failure = context.fail("gather", Cause::Native(Error::InvalidParameter));
        let tensor = BackendTensor::failed(Shape::new([3, 4]), DType::F32, failure.clone());
        let output = BackendTensor::apply("reshape", Shape::new([12]), DType::F32, &[&tensor], |_| panic!());
        assert_eq!(output.shape().as_slice(), [12]);
        assert!(!output.can_mut());
        assert!(Arc::ptr_eq(output.value.as_ref().unwrap_err(), &failure));
        assert!(!output.clone().read().unwrap_err().is_device_poisoned());
        assert!(context.check().is_ok());
        assert!(output.read().is_err());
        assert!(context.check().is_ok());
    }

    #[test]
    fn acknowledgment_does_not_hide_independent_errors_or_clear_a_poisoned_device() {
        let context = Context::new();
        let first = context.fail("allocation", Cause::Native(Error::AllocationFailed));
        let second = context.fail("dtype", Cause::Capability("F64"));
        first.report();
        assert!(context.check().is_err());
        assert!(second.reported.load(Ordering::Acquire));
        assert!(context.check().is_ok());
        let fault = context.fail("kernel", Cause::Native(Error::DeviceFault));
        assert!(context.check().unwrap_err().is_device_poisoned());
        assert!(context.check().unwrap_err().is_device_poisoned());
        assert!(Arc::ptr_eq(&context.fail("later", Cause::Native(Error::InvalidParameter)), &fault));
        assert!(fault.report().is_device_poisoned());
        assert!(context.check().is_err());
    }
}
