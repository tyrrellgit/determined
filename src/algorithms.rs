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
    UpdateModel,
    LinearUpdate, NonLinearUpdate,
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

/// Impementation for Defaults
impl<const N: usize, const M: usize, T> DefaultFromState for KalmanFilterT<N, M, T>
where
    T: UpdateModel<na::Const<N>, na::Const<M>>
        + DefaultFromState<StateType = StatePtr<na::Const<N>>, DefaultType = T>,
{   
    type DefaultType = Self;
    type StateType = StatePtr<na::Const<N>>;
    
    fn default_from_state(state: Self::StateType) -> Self::DefaultType {
        KalmanFilterT {
            state: state.clone(),
            update: T::default_from_state(state),
        }
    }

}

/// Implementation for constructors
impl<const N: usize, const M: usize, T> KalmanFilterT<N, M, T>
where
    T: UpdateModel<na::Const<N>, na::Const<M>>,
{
    pub fn new(
        mut update: T,
    ) -> Self {
        let state = update.state(None);
        KalmanFilterT {
            state: state.clone(),
            update,
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
        self.update.state(Some(_epoch)).clone()
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

/// Extended Kalman Filter
pub type ExtendedKalmanFilterNM<const N: usize, const M: usize> =
    KalmanFilterT<
        N, M,
        NonLinearUpdate<N, M>
    >;