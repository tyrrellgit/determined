//! Model traits used by the filter implementations.
//!
//! The core traits hold only what every filter needs: propagating the state
//! (`TransitionModel`), projecting it into the measurement domain with the
//! measurement noise (`MeasurementModel`), and the filter's update step
//! (`UpdateModel`). Capabilities that only some filters or models have are
//! separate traits: Jacobians (`TransitionJacobian`, `MeasurementJacobian`,
//! needed by the Extended Kalman Filter) and the inverse measurement model
//! (`InverseMeasurement`).
//! 

use crate::common::na as na;
use crate::epoch::Epoch;
use crate::measurement::Observation;
use crate::state::{ State, StatePtr };

pub trait DefaultFromState{

    type DefaultType;
    type StateType;

    fn default_from_state(state: Self::StateType) -> Self::DefaultType;
}

/// State transition model trait.
pub trait TransitionModel<N> 
where
    N: na::Dim,
    na::DefaultAllocator: na::allocator::Allocator<N, na::U1>
        + na::allocator::Allocator<N, N>,
{
    /// x(t) -> x(t')
    fn state(&mut self, epoch: &Epoch) -> &StatePtr<N>;
}

/// Jacobian of a state transition.
pub trait TransitionJacobian<N>
where
    N: na::Dim,
    na::DefaultAllocator: na::allocator::Allocator<N, na::U1>
        + na::allocator::Allocator<N, N>,
{
    /// Jacobian df/dx (N x N)
    fn jacobian(&self, state: &State<N>) -> na::OMatrix<f64, N, N>;
}

/// Measurement model trait.
pub trait MeasurementModel<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: na::allocator::Allocator<N, na::U1>
        + na::allocator::Allocator<M, na::U1>
        + na::allocator::Allocator<N, N>
        + na::allocator::Allocator<M, M>
        + na::allocator::Allocator<M, N> 
{
    /// measurement projection h(x) -> z
    fn projection(&self, state: &State<N>) -> Observation<M>;

    /// measurement noise covariance R (M x M)
    fn noise(&self) -> na::OMatrix<f64, M, M>;
}

/// Jacobian of a measurement model.
pub trait MeasurementJacobian<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: na::allocator::Allocator<N, na::U1>
        + na::allocator::Allocator<M, na::U1>
        + na::allocator::Allocator<N, N>
        + na::allocator::Allocator<M, M>
        + na::allocator::Allocator<M, N> 
{
    /// Jacobian dh/dx (M x N)
    fn jacobian(&self, state: &State<N>) -> na::OMatrix<f64, M, N>;
}

/// Inverse measurement model.
pub trait InverseMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: na::allocator::Allocator<N, na::U1>
        + na::allocator::Allocator<M, na::U1>
        + na::allocator::Allocator<N, N>
        + na::allocator::Allocator<M, M>
{
    /// inverse measurement model h^-1(z) -> x, or `None` if this model has
    /// no inverse (e.g. a rank deficient measurement matrix)
    fn inverse(&self, obs: &Observation<M>) -> Option<State<N>>;
}

/// Update model trait.
pub trait UpdateModel<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: na::allocator::Allocator<N, na::U1>
        + na::allocator::Allocator<M, na::U1>
        + na::allocator::Allocator<N, N>
        + na::allocator::Allocator<M, M>
        + na::allocator::Allocator<M, N> 
{   
    /// State accessor
    fn state(&mut self, epoch: Option<&Epoch>) -> &StatePtr<N>;

    /// Compute updated gains
    fn apply(&mut self, observation: &Observation<M>) -> &StatePtr<N>;
}
