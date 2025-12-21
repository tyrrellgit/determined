//! Algorithms module
//!
//! Contains concrete filter algorithm implementations. The implementations are
//! intentionally written using nalgebra's statically sized `SMatrix` types so
//! they can be used in `no_std`/embedded contexts with predictable memory
//! usage.

use crate::common::na as na;
use crate::state::StatePtr;
use crate::measurement::Observation;
use crate::common::Epoch;
use crate::filter::Filter;
use crate::models::{TransitionModel, UpdateModel, MeasurementModel};
use crate::models::{LinearTransition, LinearMeasurement, LinearUpdate};
use crate::models::DefaultFromState;


/// Generic Kalman Filter implementation for statically sized
/// state transition and measurement models.
pub struct KalmanFilterTUH<const N: usize, const M: usize, T, U, H>
where
    T: TransitionModel<N>,
    U: UpdateModel<N, M>,
    H: MeasurementModel<N, M>,
{
    pub state: StatePtr<na::Const<N>>,
    pub transition: T,
    pub update: U,
    pub measurement: H,
}

/// Implementation for constructors
impl<const N: usize, const M: usize, T, U, H> KalmanFilterTUH<N, M, T, U, H>
where
    T: TransitionModel<N>
        + DefaultFromState<StateType = StatePtr<na::Const<N>>, DefaultType = T>,
    U: UpdateModel<N, M>
        + DefaultFromState<StateType = StatePtr<na::Const<N>>, DefaultType = U>,
    H: MeasurementModel<N, M> + Default,
{
    pub fn new(
        state: StatePtr<na::Const<N>>,
        transition: T,
        measurement: H,
        update: U,
    ) -> Self {
        KalmanFilterTUH {
            state,
            transition,
            update,
            measurement,
        }
    }

    pub fn default_from_state(state: StatePtr<na::Const<N>>) -> Self {
        KalmanFilterTUH {
            state: state.clone(),
            transition: T::default_from_state(state.clone()),
            update: U::default_from_state(state),
            measurement: H::default(),
        }
    }
}

/// Filter trait implementation
impl<const N: usize, const M: usize, T, U, H> Filter for KalmanFilterTUH<N, M, T, U, H>
where
    T: TransitionModel<N>,
    U: UpdateModel<N, M>,
    H: MeasurementModel<N, M>,
{
    type StateType = StatePtr<na::Const<N>>;
    type ObservationType = Observation<na::Const<M>>;

    fn predict(&mut self, _epoch: &Epoch) -> Self::StateType {
        self.transition.state(&_epoch).clone()
    }

    fn update<'a>(&'a mut self, observation: &Self::ObservationType) -> Self::StateType {
        self.update.apply(observation).clone()
    }

    fn state<'a>(&'a self) -> Self::StateType {
        self.state.clone()
    }

    fn reset(&mut self) {
        self.state.borrow_mut().value.fill(0.0);
        self.state.borrow_mut().covariance = Some(na::SMatrix::<f64, N, N>::identity());
    }
}

/// KalmanFilter
pub type KalmanFilterNM<const N: usize, const M: usize> =
    KalmanFilterTUH<
        N, M, 
        LinearTransition<N>,
        LinearUpdate<N, M>,
        LinearMeasurement<N, M>,>;