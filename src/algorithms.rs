//! Algorithms module
//!
//! Contains concrete filter algorithm implementations. The implementations are
//! intentionally written using nalgebra's statically sized `SMatrix` types so
//! they can be used in `no_std`/embedded contexts with predictable memory
//! usage.

use crate::common::na as na;
use crate::state::State;
use crate::common::Epoch;
use crate::common::Algorithm;
use crate::models::{StateTransition, MeasurementModel};

/// KalmanFilter: statically sized (SMatrix) implementation.
#[derive(Clone, Debug)]
pub struct KalmanFilter<const N: usize, const M: usize> {
    pub x: State<N, 1>,                         // state vector (N x 1)
    pub p: na::SMatrix<f64, N, N>,             // covariance (N x N)
    pub f: na::SMatrix<f64, N, N>,             // state transition (N x N)
    pub q: na::SMatrix<f64, N, N>,             // process noise (N x N)
    pub h: na::SMatrix<f64, M, N>,             // measurement matrix (M x N)
    pub r: na::SMatrix<f64, M, M>,             // measurement noise (M x M)
}

impl<const N: usize, const M: usize> KalmanFilter<N, M> {
    pub fn builder() -> Self {
        // convenience builder -> default values
        KalmanFilter::new()
    }
}

impl<const N: usize, const M: usize> Algorithm for KalmanFilter<N, M> {
    type StateType = State<N, 1>;
    type ObservationType = State<M, 1>;

    fn new() -> Self {
        KalmanFilter {
            x: State::new(0.0, 0),
            p: na::SMatrix::<f64, N, N>::identity(),
            f: na::SMatrix::<f64, N, N>::identity(),
            q: na::SMatrix::<f64, N, N>::zeros(),
            h: na::SMatrix::<f64, M, N>::zeros(),
            r: na::SMatrix::<f64, M, M>::identity(),
        }
    }

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

/// Extended Kalman Filter implementation using given state transition and measurement models.
pub struct Ekf<const N: usize, const M: usize, T, U>
where
    T: StateTransition<N>,
    U: MeasurementModel<N, M>,
{
    pub x: State<N, 1>,
    pub p: na::SMatrix<f64, N, N>,
    pub q: na::SMatrix<f64, N, N>,
    pub r: na::SMatrix<f64, M, M>,
    pub transition: T,
    pub measurement: U,
}

impl<const N: usize, const M: usize, T, U> Ekf<N, M, T, U>
where
    T: StateTransition<N>,
    U: MeasurementModel<N, M>,
{
    pub fn new(transition: T, measurement: U) -> Self {
        Ekf {
            x: State::<N, 1>::new(0.0, 0),
            p: na::SMatrix::<f64, N, N>::identity(),
            q: na::SMatrix::<f64, N, N>::zeros(),
            r: na::SMatrix::<f64, M, M>::identity(),
            transition,
            measurement,
        }
    }
}

impl<const N: usize, const M: usize, T, U> Algorithm for Ekf<N, M, T, U>
where
    T: StateTransition<N> + Default,
    U: MeasurementModel<N, M> + Default,
{
    type StateType = State<N, 1>;
    type ObservationType = State<M, 1>;

    fn new() -> Self
    where
        Self: Sized,
    {
        Ekf::new(T::default(), U::default())
    }

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
        let s_inv = s.try_inverse().expect("innovation covariance S is singular");
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
