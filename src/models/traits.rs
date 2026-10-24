//! Model traits used by the Extended Kalman Filter implementation.
//!
//! These traits allow users to supply nonlinear state transition and
//! measurement models with accompanying Jacobians so the EKF can operate.
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

    /// inverse measurement model h^-1(z) -> x
    fn inverse(&self, obs: &Observation<M>) -> State<N>;

    /// Jacobian dh/dx (M x N)
    fn jacobian(&self, state: &State<N>) -> na::OMatrix<f64, M, N>;
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
    /// State transition model
    fn state(&mut self, epoch: &Epoch) -> &StatePtr<N>;

    /// Compute updated gains
    fn apply(&mut self, observation: &Observation<M>) -> &StatePtr<N>;

    /// Jacobian dh/dx (M x N)
    fn jacobian(&self, x: &State<N>) -> na::OMatrix<f64, M, N>;
}
