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
    UpdateModel, LinearUpdate,
    DefaultFromState
};


/// Generic Kalman Filter implementation for statically sized
/// state transition and measurement models.
pub struct KalmanFilterT<const N: usize, const M: usize, T>
where
    T: UpdateModel<na::Const<N>, na::Const<M>>
{
    pub state: StatePtr<na::Const<N>>,
    pub update: T,
}

/// Implementation for constructors
impl<const N: usize, const M: usize, T> KalmanFilterT<N, M, T>
where
    T: UpdateModel<na::Const<N>, na::Const<M>>
        + DefaultFromState<StateType = StatePtr<na::Const<N>>, DefaultType = T>,
{
    pub fn new(
        state: StatePtr<na::Const<N>>,
        update: T,
    ) -> Self {
        KalmanFilterT {
            state,
            update,
        }
    }

    pub fn default_from_state(state: StatePtr<na::Const<N>>) -> Self {
        KalmanFilterT {
            state: state.clone(),
            update: T::default_from_state(state),
        }
    }
}

/// Filter trait implementation
impl<const N: usize, const M: usize, T> Filter for KalmanFilterT<N, M, T>
where
    T: UpdateModel<na::Const<N>, na::Const<M>>,
{
    type StateType = StatePtr<na::Const<N>>;
    type ObservationType = Observation<na::Const<M>>;

    fn predict(&mut self, _epoch: &Epoch) -> Self::StateType {
        self.update.state(_epoch).clone()
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
    KalmanFilterT<
        N, M, 
        LinearUpdate<N, M>
    >;