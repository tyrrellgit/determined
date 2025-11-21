//! Model traits used by the Extended Kalman Filter implementation.
//!
//! These traits allow users to supply nonlinear state transition and
//! measurement models with accompanying Jacobians so the EKF can operate.

use crate::common::na as na;

/// State transition model trait used by EKF and related filters.
pub trait StateTransition<const N: usize> {
    /// f(x, u) -> x'
    fn f(&self, x: &na::SMatrix<f64, N, 1>, u: Option<&na::SMatrix<f64, N, 1>>) -> na::SMatrix<f64, N, 1>;

    /// Jacobian df/dx (N x N)
    fn jacobian(&self, x: &na::SMatrix<f64, N, 1>) -> na::SMatrix<f64, N, N>;
}

/// Measurement model trait used by EKF.
pub trait MeasurementModel<const N: usize, const M: usize> {
    /// h(x) -> z
    fn h(&self, x: &na::SMatrix<f64, N, 1>) -> na::SMatrix<f64, M, 1>;

    /// Jacobian dh/dx (M x N)
    fn jacobian(&self, x: &na::SMatrix<f64, N, 1>) -> na::SMatrix<f64, M, N>;
}
