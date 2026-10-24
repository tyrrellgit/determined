//! Kalman measurement update shared by the linear and nonlinear update models.

use crate::common::{ na, FilterAllocator };
use crate::state::State;

/// Apply the Kalman measurement update to `state` in place.
///
/// `z_x` is the predicted measurement h(x) and `h` its Jacobian at `x` (the
/// measurement matrix for a linear model). The covariance uses the Joseph form
/// `(I - KH) P (I - KH)^T + K R K^T`, which stays symmetric positive
/// semi-definite under rounding. Returns `false`, leaving `state` unchanged,
/// if the innovation covariance is singular.
pub fn kalman_update<N, M>(
    state: &mut State<N>,
    z: &na::OVector<f64, M>,
    z_x: &na::OVector<f64, M>,
    h: &na::OMatrix<f64, M, N>,
    r: &na::OMatrix<f64, M, M>,
) -> bool
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    let cov = state.covariance();

    // innovation covariance and gain
    let h_t = h.transpose();
    let s = h * &cov * &h_t + r;
    let s_inv = match s.try_inverse() {
        Some(s_inv) => s_inv,
        None => {
            spdlog::error!(
                "Invalid update: Innovation covariance S is singular; state will not be updated."
            );
            return false;
        }
    };
    let k = &cov * &h_t * s_inv;

    // update state with the innovation
    state.value = &state.value + &k * (z - z_x);

    // Joseph form covariance
    let (n, _) = cov.shape_generic();
    let i_kh = na::OMatrix::<f64, N, N>::identity_generic(n, n) - &k * h;
    state.covariance = Some(&i_kh * &cov * i_kh.transpose() + &k * r * k.transpose());

    true
}
