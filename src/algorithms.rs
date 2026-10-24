//! Algorithms module
//!
//! Contains concrete filter algorithm implementations. They are generic over
//! the dimension type: use the `...NM` aliases for nalgebra's statically sized
//! matrices (predictable, allocation-free memory usage for `no_std`/embedded
//! contexts), or the `...Dyn` aliases for sizes only known at runtime (as used
//! by the Python bindings).

use std::marker::PhantomData;

use crate::common::{ na, FilterAllocator };
use crate::state::StatePtr;
use crate::measurement::Observation;
use crate::epoch::Epoch;
use crate::filter::Filter;
use crate::models::{
    UpdateModel,
    LinearUpdate, NonLinearUpdate, UnscentedUpdate,
    LinearUDUpdate, NonLinearUDUpdate,
    DefaultFromState
};


/// Generic Kalman Filter implementation for a state of dimension `N` and
/// measurements of dimension `M`, built from an update model `T`.
pub struct KalmanFilterT<N, M, T>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
    T: UpdateModel<N, M>
{
    pub state: StatePtr<N>,
    pub update: T,
    _measurement: PhantomData<fn() -> M>,
}

/// Impementation for Defaults
impl<N, M, T> DefaultFromState for KalmanFilterT<N, M, T>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
    T: UpdateModel<N, M>
        + DefaultFromState<StateType = StatePtr<N>, DefaultType = T>,
{   
    type DefaultType = Self;
    type StateType = StatePtr<N>;
    
    fn default_from_state(state: Self::StateType) -> Self::DefaultType {
        KalmanFilterT {
            state: state.clone(),
            update: T::default_from_state(state),
            _measurement: PhantomData,
        }
    }

}

/// Implementation for constructors
impl<N, M, T> KalmanFilterT<N, M, T>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
    T: UpdateModel<N, M>,
{
    pub fn new(
        mut update: T,
    ) -> Self {
        let state = update.state(None);
        KalmanFilterT {
            state: state.clone(),
            update,
            _measurement: PhantomData,
        }
    }
}

/// Filter trait implementation
impl<N, M, T> Filter for KalmanFilterT<N, M, T>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
    T: UpdateModel<N, M>,
{
    type StateType = StatePtr<N>;
    type ObservationType = Observation<M>;

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
        let (n, _) = state.value.shape_generic();
        state.value.fill(0.0);
        state.covariance = Some(na::OMatrix::<f64, N, N>::identity_generic(n, n));
    }
}

/// KalmanFilter
pub type KalmanFilterNM<const N: usize, const M: usize> =
    KalmanFilterT<
        na::Const<N>, na::Const<M>,
        LinearUpdate<na::Const<N>, na::Const<M>>
    >;

/// KalmanFilter with runtime sized state and measurement
pub type KalmanFilterDyn =
    KalmanFilterT<na::Dyn, na::Dyn, LinearUpdate<na::Dyn, na::Dyn>>;

/// Extended Kalman Filter
pub type ExtendedKalmanFilterNM<const N: usize, const M: usize> =
    KalmanFilterT<
        na::Const<N>, na::Const<M>,
        NonLinearUpdate<na::Const<N>, na::Const<M>>
    >;

/// Extended Kalman Filter with runtime sized state and measurement
pub type ExtendedKalmanFilterDyn =
    KalmanFilterT<na::Dyn, na::Dyn, NonLinearUpdate<na::Dyn, na::Dyn>>;

/// Unscented Kalman Filter
pub type UnscentedKalmanFilterNM<const N: usize, const M: usize> =
    KalmanFilterT<
        na::Const<N>, na::Const<M>,
        UnscentedUpdate<na::Const<N>, na::Const<M>>
    >;

/// Unscented Kalman Filter with runtime sized state and measurement
pub type UnscentedKalmanFilterDyn =
    KalmanFilterT<na::Dyn, na::Dyn, UnscentedUpdate<na::Dyn, na::Dyn>>;

/// UD factorised Kalman Filter
pub type UDKalmanFilterNM<const N: usize, const M: usize> =
    KalmanFilterT<
        na::Const<N>, na::Const<M>,
        LinearUDUpdate<na::Const<N>, na::Const<M>>
    >;

/// UD factorised Kalman Filter with runtime sized state and measurement
pub type UDKalmanFilterDyn =
    KalmanFilterT<na::Dyn, na::Dyn, LinearUDUpdate<na::Dyn, na::Dyn>>;

/// UD factorised Extended Kalman Filter
pub type UDExtendedKalmanFilterNM<const N: usize, const M: usize> =
    KalmanFilterT<
        na::Const<N>, na::Const<M>,
        NonLinearUDUpdate<na::Const<N>, na::Const<M>>
    >;

/// UD factorised Extended Kalman Filter with runtime sized state and measurement
pub type UDExtendedKalmanFilterDyn =
    KalmanFilterT<na::Dyn, na::Dyn, NonLinearUDUpdate<na::Dyn, na::Dyn>>;
