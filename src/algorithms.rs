//! Algorithms module
//!
//! Contains concrete filter algorithm implementations. The implementations are
//! intentionally written using nalgebra's statically sized `SMatrix` types so
//! they can be used in `no_std`/embedded contexts with predictable memory
//! usage.

use spdlog;

use crate::common::na as na;
use crate::state::State;
use crate::common::Epoch;
use crate::filter::Filter;
use crate::models::{StateTransition, MeasurementModel};

/// KalmanFilter: statically sized (SMatrix) implementation.
#[derive(Clone, Debug)]
pub struct KalmanFilter<const N: usize, const M: usize> {
    pub x: State<na::Const<N>>,                         // state vector (N x 1)
    pub p: na::SMatrix<f64, N, N>,             // covariance (N x N)
    pub f: na::SMatrix<f64, N, N>,             // state transition (N x N)
    pub q: na::SMatrix<f64, N, N>,             // process noise (N x N)
    pub h: na::SMatrix<f64, M, N>,             // measurement matrix (M x N)
    pub r: na::SMatrix<f64, M, M>,             // measurement noise (M x M)
}

impl<const N: usize, const M: usize> Default for KalmanFilter<N, M> {
    fn default() -> Self {
        KalmanFilter {
            x: State::<na::Const<N>>::default(),
            p: na::SMatrix::<f64, N, N>::identity(),
            f: na::SMatrix::<f64, N, N>::identity(),
            q: na::SMatrix::<f64, N, N>::zeros(),
            h: na::SMatrix::<f64, M, N>::zeros(),
            r:na:: SMatrix::<f64, M, M>::identity(),
        }
    }
}

impl<const N: usize, const M: usize> Filter for KalmanFilter<N, M> {
    type StateType = State<na::Const<N>>;
    type ObservationType = State<na::Const<M>>;

    fn predict<'a>(&'a mut self, _epoch: &Epoch) -> &'a Self::StateType {
        self.x.value = &self.f * &self.x.value;
        self.p = &self.f * &self.p * self.f.transpose() + &self.q;
        &self.x
    }

    fn update<'a>(&'a mut self, observation: &Self::ObservationType) -> &'a Self::StateType {
        let z = &observation.value;
        let s = &self.h * &self.p * self.h.transpose() + &self.r;
        let s_inv = s.try_inverse().expect("innovation covariance S is singular");
        let k = &self.p * self.h.transpose() * s_inv;
        let y = z - &self.h * &self.x.value;

        self.x.value = &self.x.value + &k * y;

        let i = na::SMatrix::<f64, N, N>::identity();
        self.p = (i - &k * &self.h) * &self.p;
        &self.x
    }

    fn state(&self) -> &Self::StateType {
        &self.x
    }

    fn state_mut(&mut self) -> &mut Self::StateType {
        &mut self.x
    }

    fn reset(&mut self) {
        self.x.value.fill(0.0);
        self.p = na::SMatrix::<f64, N, N>::identity();
    }
}

// DynKalmanFilter is KalmanFilter with State and SMatrix using dynamic sizing
// This is less efficient than the statically sized version but allows for
// flexibility in state and measurement dimensions at runtime.
#[derive(Clone, Debug)]
pub struct DynKalmanFilter {
    pub x: State<na::Dyn>,                 // state vector
    pub p: na::DMatrix<f64>,                           // covariance
    pub f: na::DMatrix<f64>,                           // state transition
    pub q: na::DMatrix<f64>,                           // process noise
    pub h: na::DMatrix<f64>,                           // measurement matrix
    pub r: na::DMatrix<f64>,                           // measurement noise
}

impl DynKalmanFilter {
    pub fn new(state_dim: usize, meas_dim: usize) -> Self {
        DynKalmanFilter {
            x: State::<na::Dyn>::zeros(state_dim, 0),
            p: na::DMatrix::<f64>::identity(state_dim, state_dim),
            f: na::DMatrix::<f64>::identity(state_dim, state_dim),
            q: na::DMatrix::<f64>::zeros(state_dim, state_dim),
            h: na::DMatrix::<f64>::zeros(meas_dim, state_dim),
            r: na::DMatrix::<f64>::identity(meas_dim, meas_dim),
        }
    }
}

impl Default for DynKalmanFilter {
    fn default() -> Self {
        let state_dim: usize = 1;
        let meas_dim: usize = 1;
        DynKalmanFilter {
            x: State::<na::Dyn>::zeros(state_dim, 0),
            p: na::DMatrix::<f64>::identity(state_dim, state_dim),
            f: na::DMatrix::<f64>::identity(state_dim, state_dim),
            q: na::DMatrix::<f64>::zeros(state_dim, state_dim),
            h: na::DMatrix::<f64>::zeros(meas_dim, state_dim),
            r: na::DMatrix::<f64>::identity(meas_dim, meas_dim),
        }
    }
}

impl Filter for DynKalmanFilter {
    type StateType = State<na::Dyn>;
    type ObservationType = State<na::Dyn>;

    fn predict<'a>(&'a mut self, _epoch: &Epoch) -> &'a Self::StateType {
        self.x.value = &self.f * &self.x.value;
        self.p = &self.f * &self.p * self.f.transpose() + &self.q;
        &self.x
    }

    fn update<'a>(&'a mut self, observation: &Self::ObservationType) -> &'a Self::StateType {
        let z = &observation.value;
        let s = &self.h * &self.p * self.h.transpose() + &self.r;
        let s_inv = s.try_inverse().expect("innovation covariance S is singular");
        let k = &self.p * self.h.transpose() * s_inv;
        let y = z - &self.h * &self.x.value;

        self.x.value = &self.x.value + &k * y;

        let i = na::DMatrix::<f64>::identity(self.p.nrows(), self.p.ncols());
        self.p = (i - &k * &self.h) * &self.p;
        &self.x
    }

    fn state(&self) -> &Self::StateType {
        &self.x
    }

    fn state_mut(&mut self) -> &mut Self::StateType {
        &mut self.x
    }

    fn reset(&mut self) {
        self.x.value.fill(0.0);
        self.p = na::DMatrix::<f64>::identity(self.p.nrows(), self.p.ncols());
    }
}

/// Generic Kalman Filter implementation for arbitrarystate transition and measurement models.
pub struct KalmanFilterTU<const N: usize, const M: usize, T, U>
where
    T: StateTransition<N>,
    U: MeasurementModel<N, M>,
{
    pub x: State<na::Const<N>>,
    pub p: na::SMatrix<f64, N, N>,
    pub q: na::SMatrix<f64, N, N>,
    pub r: na::SMatrix<f64, M, M>,
    pub transition: T,
    pub measurement: U,
}

impl<const N: usize, const M: usize, T, U> Default for KalmanFilterTU<N, M, T, U>
where
    T: StateTransition<N> + Default,
    U: MeasurementModel<N, M> + Default,
{
    fn default() -> Self {
        KalmanFilterTU {
            x: State::<na::Const<N>>::default(),
            p: na::SMatrix::<f64, N, N>::identity(),
            q: na::SMatrix::<f64, N, N>::zeros(),
            r: na::SMatrix::<f64, M, M>::identity(),
            transition: T::default(),
            measurement: U::default(),
        }
    }
}

impl<const N: usize, const M: usize, T, U> Filter for KalmanFilterTU<N, M, T, U>
where
    T: StateTransition<N>,
    U: MeasurementModel<N, M>,
{
    type StateType = State<na::Const<N>>;
    type ObservationType = State<na::Const<M>>;

    fn predict<'a>(&'a mut self, _epoch: &Epoch) -> &'a Self::StateType {
        let x_pred = self.transition.f(&self.x.value, None);
        self.x.value = x_pred;
        let j = self.transition.jacobian(&self.x.value);
        self.p = &j * &self.p * j.transpose() + &self.q;
        &self.x
    }

    fn update<'a>(&'a mut self, observation: &Self::ObservationType) -> &'a Self::StateType {
        let z = &observation.value;
        let h_x = self.measurement.h(&self.x.value);
        let h = self.measurement.jacobian(&self.x.value);
        let s = &h * &self.p * h.transpose() + &self.r;
        let s_inv = match s.try_inverse(){
            None => {
                spdlog::error!(
                    "Invalid update: Innovation covariance S is singular; state will not be updated."
                );
                return &self.x;
            }
            Some(s_inv) => s_inv,
        };
        
        let k = &self.p * h.transpose() * s_inv;
        let y = z - h_x;
        self.x.value = &self.x.value + &k * y;
        let i = na::SMatrix::<f64, N, N>::identity();
        self.p = (i - &k * &h) * &self.p;
        &self.x
    }

    fn state(&self) -> &Self::StateType {
        &self.x
    }

    fn state_mut(&mut self) -> &mut Self::StateType {
        &mut self.x
    }

    fn reset(&mut self) {
        self.x.value.fill(0.0);
        self.p = na::SMatrix::<f64, N, N>::identity();
    }
}
