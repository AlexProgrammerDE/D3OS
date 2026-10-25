//! Native fusion behind the public Burn Adam optimizer and record format.
mod ops;

use super::{AdamState, AdaptiveMomentum, AdaptiveMomentumState};
use crate::LearningRate;
use burn_core::tensor::{DType, Tensor};
use burn_dispatch::{Dispatch, DispatchDevice};
use ops::AdamOps;

pub(super) fn supports<const D: usize>(
    tensor: &Tensor<D>, grad: &Tensor<D>, state: Option<&AdaptiveMomentumState<D>>,
) -> bool {
    let untracked = |tensor: &Tensor<D>| {
        tensor.dtype() == DType::F32
            && !tensor.is_require_grad()
            && matches!(
                tensor.device().inner().as_dispatch(),
                DispatchDevice::Native(_)
            )
    };
    untracked(tensor)
        && untracked(grad)
        && state.is_none_or(|state| {
            untracked(&state.moment_1)
                && untracked(&state.moment_2)
                && state.max_moment_2.as_ref().is_none_or(untracked)
        })
}

pub(super) fn step<const D: usize>(
    momentum: &AdaptiveMomentum, lr: LearningRate, tensor: Tensor<D>, grad: Tensor<D>,
    state: Option<AdaptiveMomentumState<D>>,
) -> (Tensor<D>, Option<AdamState<D>>) {
    let (time, first, second, mut maximum) = match state {
        Some(state) => (
            state.time + 1,
            Some(state.moment_1.into_dispatch()),
            Some(state.moment_2.into_dispatch()),
            state.max_moment_2,
        ),
        None => (1, None, None, None),
    };
    let (bias_sqrt, combined) = momentum.bias_factors(time);
    let coefficients = [
        momentum.beta_1,
        1.0 - momentum.beta_1,
        momentum.beta_2,
        1.0 - momentum.beta_2,
        combined,
        momentum.epsilon * bias_sqrt,
        lr as f32,
    ];
    let history = if momentum.amsgrad {
        maximum.take().map(Tensor::into_dispatch)
    } else {
        None
    };
    let (parameter, first, second, next_maximum) = Dispatch::adam_update(
        tensor.into_dispatch(),
        grad.into_dispatch(),
        first,
        second,
        history,
        momentum.amsgrad,
        coefficients,
    );
    let state = AdaptiveMomentumState {
        time,
        moment_1: Tensor::from_dispatch(first),
        moment_2: Tensor::from_dispatch(second),
        max_moment_2: next_maximum.map(Tensor::from_dispatch).or(maximum),
    };
    (
        Tensor::from_dispatch(parameter),
        Some(AdamState::new(state)),
    )
}
