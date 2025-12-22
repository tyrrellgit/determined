//! Algorithms module
//!
//! Contains concrete filter algorithm implementations. The implementations are
//! intentionally written using nalgebra's statically sized `SMatrix` types so
//! they can be used in `no_std`/embedded contexts with predictable memory
//! usage.

use crate::common::na;
use crate::state::StatePtr;
use crate::measurement::Observation;
use crate::epoch::Epoch;
use crate::filter::Filter;
use crate::models::{
    TransitionModel, UpdateModel, MeasurementModel,
    LinearTransition, LinearMeasurement, LinearUpdate,
    DefaultFromState
};


/// Generic Kalman Filter implementation for statically sized
/// state transition and measurement models.
pub struct KalmanFilterTUH<const N: usize, const M: usize, T, U, H>
where
    T: TransitionModel<na::Const<N>>,
    U: UpdateModel<na::Const<N>, na::Const<M>>,
    H: MeasurementModel<na::Const<N>, na::Const<M>>,
{
    pub state: StatePtr<na::Const<N>>,
    pub transition: T,
    pub update: U,
    pub measurement: H,
}

/// Implementation for constructors
impl<const N: usize, const M: usize, T, U, H> KalmanFilterTUH<N, M, T, U, H>
where
    T: TransitionModel<na::Const<N>>
        + DefaultFromState<StateType = StatePtr<na::Const<N>>, DefaultType = T>,
    U: UpdateModel<na::Const<N>, na::Const<M>>
        + DefaultFromState<StateType = StatePtr<na::Const<N>>, DefaultType = U>,
    H: MeasurementModel<na::Const<N>, na::Const<M>> + Default,
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
    T: TransitionModel<na::Const<N>>,
    U: UpdateModel<na::Const<N>, na::Const<M>>,
    H: MeasurementModel<na::Const<N>, na::Const<M>>,
{
    type StateType = StatePtr<na::Const<N>>;
    type ObservationType = Observation<na::Const<M>>;

    fn predict(&mut self, _epoch: &Epoch) -> Self::StateType {
        self.transition.state(_epoch).clone()
    }

    fn update(&mut self, observation: &Self::ObservationType) -> Self::StateType {
        self.update.apply(observation).clone()
    }

    fn state(&self) -> Self::StateType {
        self.state.clone()
    }

    fn reset(&mut self) {
        let mut state = self.state.write().unwrap();
        state.value.fill(0.0);
        state.covariance = Some(na::SMatrix::<f64, N, N>::identity());
    }
}

/// KalmanFilter
pub type KalmanFilterNM<const N: usize, const M: usize> =
    KalmanFilterTUH<
        N, M, 
        LinearTransition<N>,
        LinearUpdate<N, M>,
        LinearMeasurement<N, M>,>;