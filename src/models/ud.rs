//! UD factorised Kalman filters.
//!
//! The covariance is held as factors `P = U D U^T` (`U` unit upper triangular,
//! `D` diagonal) and the filter propagates and updates the factors directly,
//! never forming `P`. This keeps the covariance symmetric and positive
//! semi-definite by construction. Following Ramos et al., "A summary on the
//! UD Kalman Filter" (arXiv:2203.06105):
//!
//! - initialise: `[U, D] = udu(P)`
//! - time update: weighted modified Gram-Schmidt on `W = [F U, G]`,
//!   `D̆ = diag(D, D_Q)`, where `Q = G D_Q G^T` is factorised the same way
//! - measurement update: scalar measurements processed sequentially with the
//!   modified Agee-Turner rank-one update, after decorrelating a non-diagonal
//!   `R = U_R D_R U_R^T`
//!
//! `LinearUDUpdate` and `NonLinearUDUpdate` use the existing linear and
//! nonlinear models; they differ only in how the state value is stepped and
//! how the transition Jacobian `F` is obtained.
//!
//! The shared `State` still carries `P = U D U^T` after every step, so it can
//! be read like any other filter's state, but the factors are what the filter
//! uses. If the state's covariance is changed from outside, the factors are
//! recomputed from it.

use crate::common::{ na, FilterAllocator };
use crate::common::linalg::udu_decomposition;
use crate::epoch::Epoch;
use crate::measurement::Observation;
use crate::state::{ State, StatePtr };

use na::allocator::Allocator;

use crate::models::traits::{
    MeasurementModel, MeasurementJacobian, TransitionJacobian, UpdateModel
};
use crate::models::linear::{ LinearTransition, LinearMeasurement };
use crate::models::nonlinear::{ NonLinearTransition, NonLinearMeasurement };


/// UD factors of a covariance, `P = U diag(d) U^T`.
#[derive(Clone, Debug)]
pub struct UDFactors<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    pub u: na::OMatrix<f64, N, N>,
    pub d: na::OVector<f64, N>,
    /// covariance last written to the state, to detect outside changes
    written: Option<na::OMatrix<f64, N, N>>,
}

impl<N> UDFactors<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    pub fn from_covariance(p: &na::OMatrix<f64, N, N>) -> Self {
        let (u, d) = udu_decomposition(p);
        if d.iter().any(|x| *x < 0.0) {
            spdlog::error!(
                "Covariance is not positive semi-definite; its UD factors have negative entries."
            );
        }
        UDFactors { u, d, written: None }
    }

    /// `P = U D U^T`
    pub fn covariance(&self) -> na::OMatrix<f64, N, N> {
        let p = &self.u * na::OMatrix::<f64, N, N>::from_diagonal(&self.d) * self.u.transpose();
        // remove rounding asymmetry from the product
        (&p + p.transpose()) * 0.5
    }

    /// Recompute the factors if the state's covariance is not the one these
    /// factors last wrote, i.e. it was set from outside the filter.
    fn sync(&mut self, state: &State<N>) {
        let p = state.covariance();
        if self.written.as_ref() != Some(&p) {
            *self = Self::from_covariance(&p);
        }
    }

    /// Write `P = U D U^T` into the state.
    fn write(&mut self, state: &mut State<N>) {
        let p = self.covariance();
        state.covariance = Some(p.clone());
        self.written = Some(p);
    }

    /// Time update: replace the factors with those of `F P F^T + Q`, by
    /// weighted modified Gram-Schmidt on `W = [F U, G]` with weights
    /// `diag(D, D_Q)` (§II.A). `W` is kept as its two blocks `F U` and `G`.
    fn predict(&mut self, f: &na::OMatrix<f64, N, N>, q: &na::OMatrix<f64, N, N>) {
        let n = self.d.len();
        let (g, dq) = udu_decomposition(q);
        let mut a = f * &self.u;    // rows of W, first block
        let mut b = g;              // rows of W, second block

        let (nd, _) = self.u.shape_generic();
        let mut u = na::OMatrix::<f64, N, N>::identity_generic(nd, nd);
        let mut d = na::OVector::<f64, N>::zeros_generic(nd, na::Const::<1>);

        for k in (0..n).rev() {
            // weighted norm of the (already orthogonalised) row k
            let dk: f64 = (0..n).map(|j| a[(k, j)] * self.d[j] * a[(k, j)] + b[(k, j)] * dq[j] * b[(k, j)]).sum();
            // a weighted sum of squares: negative only through rounding
            d[k] = dk.max(0.0);
            if dk <= 0.0 {
                continue;
            }
            // orthogonalise the rows above against row k
            for i in 0..k {
                let dot: f64 = (0..n).map(|j| a[(i, j)] * self.d[j] * a[(k, j)] + b[(i, j)] * dq[j] * b[(k, j)]).sum();
                let uik = dot / dk;
                u[(i, k)] = uik;
                for j in 0..n {
                    a[(i, j)] -= uik * a[(k, j)];
                    b[(i, j)] -= uik * b[(k, j)];
                }
            }
        }
        self.u = u;
        self.d = d;
    }

    /// Measurement update for innovation `y = z - h(x)`, measurement Jacobian
    /// `h` at `x`, and noise `r`. Measurements are processed one at a time
    /// with the modified Agee-Turner update (§III, Appendix B), after
    /// decorrelating `r` if it is not diagonal (Appendix D). Returns the state
    /// correction, or `None` (logged, factors unchanged) if an innovation
    /// variance is not positive or the factors are not positive semi-definite.
    fn update<M>(
        &mut self,
        h: &na::OMatrix<f64, M, N>,
        y: &na::OVector<f64, M>,
        r: &na::OMatrix<f64, M, M>,
    ) -> Option<na::OVector<f64, N>>
    where
        M: na::Dim,
        na::DefaultAllocator: FilterAllocator<N, M>,
    {
        let m = r.nrows();
        let n = self.d.len();

        let diagonal = (0..m).all(|i| (0..m).all(|j| i == j || r[(i, j)] == 0.0));
        let (h, y, r) = if diagonal {
            (h.clone(), y.clone(), r.diagonal())
        } else {
            // R = U_R D_R U_R^T: filter on U_R^-1 H, U_R^-1 y with noise D_R
            let (ur, dr) = udu_decomposition(r);
            (ur.solve_upper_triangular(h)?, ur.solve_upper_triangular(y)?, dr)
        };

        let mut u = self.u.clone();
        let mut d = self.d.clone();
        let (nd, _) = u.shape_generic();
        let mut dx = na::OVector::<f64, N>::zeros_generic(nd, na::Const::<1>);

        for i in 0..m {
            // innovation of this measurement, given the corrections so far
            let e = y[i] - (0..n).map(|j| h[(i, j)] * dx[j]).sum::<f64>();

            // w = U^T h_i, v = D w
            let w: Vec<f64> = (0..n).map(|j| (0..n).map(|k| u[(k, j)] * h[(i, k)]).sum()).collect();
            let v: Vec<f64> = (0..n).map(|j| d[j] * w[j]).collect();

            let mut k = na::OVector::<f64, N>::zeros_generic(nd, na::Const::<1>);
            k[0] = v[0];

            // alpha_j = r + sum_{l <= j} d_l w_l^2 is the innovation variance
            // accumulated over the first j states. It is zero for a perfect
            // measurement (r = 0) of states not reached yet; the updates take
            // the limit for that case. A negative alpha means r or the factors
            // are not positive semi-definite.
            let invalid = || {
                spdlog::error!(
                    "Invalid update: measurement noise or covariance factors are not positive semi-definite; state will not be updated."
                );
                None
            };
            let mut alpha = r[i] + v[0] * w[0];
            if alpha < 0.0 {
                return invalid();
            }
            if alpha > 0.0 {
                d[0] = r[i] * d[0] / alpha;
            }

            for j in 1..n {
                let alpha_prev = alpha;
                alpha = alpha_prev + v[j] * w[j];
                if alpha < 0.0 {
                    return invalid();
                }
                if alpha > 0.0 {
                    d[j] = d[j] * alpha_prev / alpha;
                }
                // alpha_prev = 0 only when the gain so far (k) is zero, so the
                // value of lambda does not matter then
                let lambda = if alpha_prev > 0.0 { -w[j] / alpha_prev } else { 0.0 };
                for row in 0..n {
                    let u_old = u[(row, j)];
                    u[(row, j)] = u_old + lambda * k[row];
                    k[row] += v[j] * u_old;
                }
            }

            if alpha <= 0.0 {
                spdlog::error!(
                    "Invalid update: innovation variance is not positive; state will not be updated."
                );
                return None;
            }

            // gain K = k / alpha
            dx += k * (e / alpha);
        }

        self.u = u;
        self.d = d;
        Some(dx)
    }
}

/// Propagate `state_ptr` to `epoch` one step at a time. `step` returns the
/// next state value and the transition Jacobian `F`, both at the state before
/// the step.
fn propagate<N>(
    state_ptr: &StatePtr<N>,
    factors: &mut UDFactors<N>,
    q: &na::OMatrix<f64, N, N>,
    epoch: &Epoch,
    mut step: impl FnMut(&State<N>) -> (na::OVector<f64, N>, na::OMatrix<f64, N, N>),
)
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    let mut state = state_ptr.read().unwrap().clone();

    let delta = { *epoch - state.epoch }.value;
    if delta == 0 {
        return;
    }
    if delta < 0 {
        spdlog::warn!(
            "Backward propagation is not supported by the UD filters; state is unchanged."
        );
        return;
    }

    factors.sync(&state);
    for _ in 0..delta {
        let (value, f) = step(&state);
        factors.predict(&f, q);
        state.value = value;
        state.epoch = state.epoch + Epoch::new(1);
    }
    factors.write(&mut state);
    *state_ptr.write().unwrap() = state;
}

/// Measurement update of `state_ptr`, linearised about the current estimate.
fn apply<N, M, Z>(
    state_ptr: &StatePtr<N>,
    factors: &mut UDFactors<N>,
    measurement: &Z,
    observation: &Observation<M>,
)
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
    Z: MeasurementModel<N, M> + MeasurementJacobian<N, M>,
{
    let mut state = state_ptr.read().unwrap().clone();
    factors.sync(&state);

    let z_x = measurement.projection(&state).value;
    let h = measurement.jacobian(&state);
    let y = &observation.value - z_x;

    if let Some(dx) = factors.update(&h, &y, &measurement.noise()) {
        state.value += dx;
        factors.write(&mut state);
        *state_ptr.write().unwrap() = state;
    }
}


// Update Models
// Linear UD
pub struct LinearUDUpdate<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub state: StatePtr<N>,                      // mutable ref. to state
    pub measurement: LinearMeasurement<N, M>,    // measurement model
    pub transition: LinearTransition<N>,         // transition model
    pub factors: UDFactors<N>,                   // covariance factors
}

/// Statically sized `LinearUDUpdate`.
pub type LinearUDUpdateNM<const N: usize, const M: usize> =
    LinearUDUpdate<na::Const<N>, na::Const<M>>;

impl<N, M> LinearUDUpdate<N, M>
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
        let factors = UDFactors::from_covariance(&_state.read().unwrap().covariance());
        LinearUDUpdate {
            state: _state,
            measurement,
            transition,
            factors,
        }
    }
}

impl<N, M> UpdateModel<N, M> for LinearUDUpdate<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    fn state(&mut self, epoch: Option<&Epoch>) -> &StatePtr<N> {
        if let Some(epoch) = epoch {
            let f = &self.transition.f;
            propagate(&self.state, &mut self.factors, &self.transition.q, epoch,
                |state| (f * &state.value, f.clone()));
        }
        &self.state
    }

    fn apply(&mut self, observation: &Observation<M>) -> &StatePtr<N> {
        // propagate state to observation epoch
        self.state(Some(&observation.epoch));
        apply(&self.state, &mut self.factors, &self.measurement, observation);
        &self.state
    }
}

// NonLinear UD (UD Extended Kalman Filter)
pub struct NonLinearUDUpdate<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub state: StatePtr<N>,                          // mutable ref. to state
    pub measurement: NonLinearMeasurement<N, M>,     // measurement model
    pub transition: NonLinearTransition<N>,          // transition model
    pub factors: UDFactors<N>,                       // covariance factors
}

/// Statically sized `NonLinearUDUpdate`.
pub type NonLinearUDUpdateNM<const N: usize, const M: usize> =
    NonLinearUDUpdate<na::Const<N>, na::Const<M>>;

impl<N, M> NonLinearUDUpdate<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub fn new(
        measurement: NonLinearMeasurement<N, M>,
        transition: NonLinearTransition<N>,
    ) -> Self {
        let _state = transition.state.clone();
        let factors = UDFactors::from_covariance(&_state.read().unwrap().covariance());
        NonLinearUDUpdate {
            state: _state,
            measurement,
            transition,
            factors,
        }
    }
}

impl<N, M> UpdateModel<N, M> for NonLinearUDUpdate<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    fn state(&mut self, epoch: Option<&Epoch>) -> &StatePtr<N> {
        if let Some(epoch) = epoch {
            let transition = &mut self.transition;
            let q = transition.q.clone();
            propagate(&self.state, &mut self.factors, &q, epoch,
                |state| {
                    let f = transition.jacobian(state);
                    ((transition.f)(state), f)
                });
        }
        &self.state
    }

    fn apply(&mut self, observation: &Observation<M>) -> &StatePtr<N> {
        // propagate state to observation epoch
        self.state(Some(&observation.epoch));
        apply(&self.state, &mut self.factors, &self.measurement, observation);
        &self.state
    }
}
