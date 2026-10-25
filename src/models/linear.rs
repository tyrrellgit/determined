//! Linear models used by the Kalman Filter implementation.
//!
//! These models take linear state transition and measurement matrices
//! defining a standard Kalman Filter. They are generic over the dimension
//! type, so work with statically sized (`na::Const<N>`, see the `...N` /
//! `...NM` aliases) and dynamically sized (`na::Dyn`) matrices.

use crate::common::{ na, FilterAllocator };
use crate::common::linalg::moore_penrose_inverse;
use crate::epoch::Epoch;
use crate::measurement::Observation;
use crate::state::{ State, StatePtr };

use na::allocator::Allocator;

use crate::models::traits::{ 
    TransitionModel, TransitionJacobian,
    MeasurementModel, MeasurementJacobian, InverseMeasurement,
    UpdateModel, DefaultFromState
};
use crate::models::update::kalman_update;

// Transition Models
// Linear (Discrete Time)
#[derive(Clone, Debug)]
pub struct LinearTransition<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    pub state: StatePtr<N>,                    // state vector (N x 1)
    pub f: na::OMatrix<f64, N, N>,             // state transition (N x N)
    pub q: na::OMatrix<f64, N, N>,             // process noise covariance (N x N)
    f_inv: Option<na::OMatrix<f64, N, N>>,     // inverse of F
}

/// Statically sized `LinearTransition`.
pub type LinearTransitionN<const N: usize> = LinearTransition<na::Const<N>>;

impl<N> LinearTransition<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{ 
    pub fn new(
        state: StatePtr<N>,
        f: na::OMatrix<f64, N, N>,
        q: na::OMatrix<f64, N, N> ) -> Self {

        let f_inv = f.clone().try_inverse();
        if f_inv.is_none() {
            spdlog::warn!(
                "State transition matrix F is non-invertible; backward propagation will do nothing."
            );
        }
        LinearTransition { state, f, q, f_inv }
    }
}   

impl<N> DefaultFromState for LinearTransition<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    type DefaultType = LinearTransition<N>;
    type StateType = StatePtr<N>;

    fn default_from_state(state: Self::StateType) -> Self::DefaultType
    where Self: Sized {
        let n = N::from_usize(state.read().unwrap().dim());
        LinearTransition::new(
            state,
            na::OMatrix::<f64, N, N>::identity_generic(n, n),
            na::OMatrix::<f64, N, N>::zeros_generic(n, n),
        )
    }
}

impl<N> TransitionModel<N> for LinearTransition<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    fn state(&mut self, epoch: &Epoch) -> &StatePtr<N> {
        // Determine timesteps to propagate
        let mut state = self.state.write().unwrap();

        let delta = { *epoch - state.epoch }.value;
        if delta == 0 {
            return &self.state;
        }

        if delta > 0 {
            // propagate forward
            for _ in 0..delta {
                let cov = state.covariance();
                state.value = &self.f * &state.value;
                state.covariance = Some(&self.f * cov * self.f.transpose() + &self.q);
            }
        } else {
            // propagate backward
            let Some(f_inv) = &self.f_inv else {
                spdlog::warn!(
                    "State transition matrix F is non-invertible; backward propagation will do nothing."
                );
                return &self.state;
            };
            for _ in 0..delta.abs() {
                let cov = state.covariance();
                state.value = f_inv * &state.value;
                state.covariance = Some(f_inv * (cov - &self.q) * f_inv.transpose());
            }
        };
        state.epoch = *epoch;
        &self.state
    }
}

impl<N> TransitionJacobian<N> for LinearTransition<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    /// The Jacobian of a linear transition is F.
    fn jacobian(&self, _state: &State<N>) -> na::OMatrix<f64, N, N> {
        self.f.clone()
    }
}

/// Measurement Models
/// Linear
#[derive(Clone, Debug)]
pub struct LinearMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub h: na::OMatrix<f64, M, N>,             // measurement matrix
    pub r: na::OMatrix<f64, M, M>,             // measurement noise covariance(M x M)
    h_inv: Option<na::OMatrix<f64, N, M>>,     // pseudo-inverse of H, if full rank
}

/// Statically sized `LinearMeasurement`.
pub type LinearMeasurementNM<const N: usize, const M: usize> =
    LinearMeasurement<na::Const<N>, na::Const<M>>;

impl<N, M> LinearMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub fn new(h: na::OMatrix<f64, M, N>, r: na::OMatrix<f64, M, M>) -> Self {
        let h_inv = moore_penrose_inverse(&h);
        LinearMeasurement { h, r, h_inv }
    }
}

impl<N, M> Default for LinearMeasurement<N, M>
where
    N: na::DimName,
    M: na::DimName,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    fn default() -> Self {
        LinearMeasurement::new(
            na::OMatrix::<f64, M, N>::identity(),
            na::OMatrix::<f64, M, M>::identity(),
        )
    }
}

impl<N, M> MeasurementModel<N, M> for LinearMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    fn projection(&self, state: &State<N>) -> Observation<M> {
        Observation {
            value: &self.h * &state.value,
            epoch: state.epoch
        }
    }

    fn noise(&self) -> na::OMatrix<f64, M, M> {
        self.r.clone()
    }
}

impl<N, M> MeasurementJacobian<N, M> for LinearMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    /// The Jacobian of a linear measurement is H.
    fn jacobian(&self, _state: &State<N>) -> na::OMatrix<f64, M, N> {
        self.h.clone()
    }
}

impl<N, M> InverseMeasurement<N, M> for LinearMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    /// Pseudo-inverse projection; `None` if H is rank deficient.
    fn inverse(&self, obs: &Observation<M>) -> Option<State<N>> {
        self.h_inv.as_ref().map(|h_inv| State {
            value: h_inv * &obs.value,
            covariance: None,
            epoch: obs.epoch,
        })
    }
}

// Update Models
// Linear
pub struct LinearUpdate<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub state: StatePtr<N>,                      // mutable ref. to state
    pub measurement: LinearMeasurement<N, M>,    // measurement model
    pub transition: LinearTransition<N>,         // transition model
}

/// Statically sized `LinearUpdate`.
pub type LinearUpdateNM<const N: usize, const M: usize> =
    LinearUpdate<na::Const<N>, na::Const<M>>;

impl<N, M> LinearUpdate<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub fn new(
        measurement: LinearMeasurement<N, M>,
        transition: LinearTransition<N>,
    ) -> Self {
        let _state = transition.state.clone();
        LinearUpdate {
            state: _state,
            measurement,
            transition,
        }
    }
}

impl<N, M> DefaultFromState for LinearUpdate<N, M>
where
    N: na::DimName,
    M: na::DimName,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    type DefaultType = LinearUpdate<N, M>;
    type StateType = StatePtr<N>;

    fn default_from_state(state: Self::StateType) -> Self::DefaultType {
        LinearUpdate::new(
            LinearMeasurement::<N, M>::default(),
            LinearTransition::<N>::default_from_state(state)
        )
    }
}

impl<N, M> UpdateModel<N, M> for LinearUpdate<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    fn state(&mut self, epoch: Option<&Epoch>) -> &StatePtr<N> {
        match epoch {
            Some(epoch) => self.transition.state(epoch),
            _ => &self.transition.state
        }
    }
    
    fn apply(&mut self, observation: &Observation<M>) -> &StatePtr<N> {

        // propagate state to observation epoch
        _ = self.transition.state(&observation.epoch);
        let mut state = self.state.write().unwrap();

        // project state to measurement domain
        let z_x = self.measurement.projection(&state).value;

        let h = self.measurement.jacobian(&state);
        let r = self.measurement.noise();
        kalman_update(&mut state, &observation.value, &z_x, &h, &r);

        &self.state
    }
}
