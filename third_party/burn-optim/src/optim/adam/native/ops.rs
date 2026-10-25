//! Preserve Burn's autodiff context when fusing untracked optimizer tensors.
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, Native, autodiff::checkpoint::strategy::CheckpointStrategy,
    backend_extension, ops::FloatTensorOps, tensor::FloatTensor,
};
use burn_core as burn;

#[backend_extension(Native, Autodiff)]
pub(super) trait AdamOps: Backend {
    fn adam_update(
        parameter: FloatTensor<Self>, gradient: FloatTensor<Self>,
        first: Option<FloatTensor<Self>>, second: Option<FloatTensor<Self>>,
        maximum: Option<FloatTensor<Self>>, amsgrad: bool, coefficients: [f32; 7],
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        Option<FloatTensor<Self>>,
    );
}

impl AdamOps for Native {
    fn adam_update(
        parameter: FloatTensor<Self>, gradient: FloatTensor<Self>,
        first: Option<FloatTensor<Self>>, second: Option<FloatTensor<Self>>,
        maximum: Option<FloatTensor<Self>>, amsgrad: bool, coefficients: [f32; 7],
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        Option<FloatTensor<Self>>,
    ) {
        assert_eq!(
            first.is_some(),
            second.is_some(),
            "Adam moments must be restored together"
        );
        Self::adam(
            parameter,
            gradient,
            first.zip(second),
            maximum,
            amsgrad,
            coefficients,
        )
    }
}

impl<C: CheckpointStrategy> AdamOps for Autodiff<Native, C> {
    fn adam_update(
        parameter: FloatTensor<Self>, gradient: FloatTensor<Self>,
        first: Option<FloatTensor<Self>>, second: Option<FloatTensor<Self>>,
        maximum: Option<FloatTensor<Self>>, amsgrad: bool, coefficients: [f32; 7],
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        Option<FloatTensor<Self>>,
    ) {
        // Tracked tensors use the ordinary differentiable implementation.
        // Untracked tensors retain their surrounding checkpointing context.
        assert!(
            !Self::float_is_require_grad(&parameter) && !Self::float_is_require_grad(&gradient)
        );
        assert!(
            first
                .as_ref()
                .is_none_or(|tensor| !Self::float_is_require_grad(tensor))
        );
        assert!(
            second
                .as_ref()
                .is_none_or(|tensor| !Self::float_is_require_grad(tensor))
        );
        assert!(
            maximum
                .as_ref()
                .is_none_or(|tensor| !Self::float_is_require_grad(tensor))
        );
        let (parameter, first, second, maximum) = Native::adam_update(
            Self::inner(parameter),
            Self::inner(gradient),
            first.map(Self::inner),
            second.map(Self::inner),
            maximum.map(Self::inner),
            amsgrad,
            coefficients,
        );
        (
            Self::from_inner(parameter),
            Self::from_inner(first),
            Self::from_inner(second),
            maximum.map(Self::from_inner),
        )
    }
}
