//! Unscented models used by the Unscented Kalman Filter implementation.
//!
//! Like the nonlinear models, these are built from user supplied closures: a
//! propagation function `f` and a measurement function `h`. No Jacobians are
//! needed; instead a set of sigma points is pushed through `f` / `h` and the
//! mean and covariance are recovered from them (the unscented transform).
//! They are generic over the dimension type, so work with statically sized
//! (`na::Const<N>`, see the `...N` / `...NM` aliases) and dynamically sized
//! (`na::Dyn`) matrices.
//!
//! The closures are run on copies of the state, so no lock is held while user
//! code runs.

use crate::common::{ na, FilterAllocator };
use crate::epoch::Epoch;
use crate::measurement::Observation;
use crate::state::{ State, StatePtr };

use na::allocator::Allocator;

use crate::models::traits::{
    TransitionModel, MeasurementModel, InverseMeasurement, UpdateModel
};
// same closure types as the nonlinear models for h and the inverse measurement
pub use crate::models::nonlinear::{ H, HInv };

/// One-step propagation: x(t) -> x(t + 1). Called once per sigma point, so
/// it must not keep state between calls (hence `Fn`, not `FnMut`).
pub type F<N> = Box<dyn Fn(&State<N>) -> na::OVector<f64, N> + Send + Sync>;


/// Scaled sigma point parameters (van der Merwe).
///
/// - `alpha`: spread of the sigma points around the mean.
/// - `beta`: prior knowledge of the distribution; 2 is optimal for Gaussians.
/// - `kappa`: secondary scaling.
///
/// The default `alpha = 1, beta = 2, kappa = 0` keeps every sigma point
/// weight non-negative, so the predicted covariance stays positive
/// semi-definite. Small `alpha` (e.g. 1e-3) gives a large negative centre
/// weight, which is numerically fragile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnscentedParams {
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
}

impl Default for UnscentedParams {
    fn default() -> Self {
        UnscentedParams { alpha: 1.0, beta: 2.0, kappa: 0.0 }
    }
}

/// Sigma points for a state, with their mean and covariance weights.
struct SigmaPoints<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    points: Vec<State<N>>,
    wm: Vec<f64>,
    wc: Vec<f64>,
}

impl<N> SigmaPoints<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    /// 2n + 1 sigma points for `state`, or `None` (logged) if its covariance
    /// is not positive definite.
    fn new(state: &State<N>, params: &UnscentedParams) -> Option<Self> {
        let n = state.dim();
        let nf = n as f64;
        let lambda = params.alpha * params.alpha * (nf + params.kappa) - nf;

        let sqrt = match na::Cholesky::new(state.covariance() * (nf + lambda)) {
            Some(chol) => chol.unpack(),
            None => {
                spdlog::error!(
                    "Invalid unscented transform: covariance is not positive definite; state will not be updated."
                );
                return None;
            }
        };

        let at = |value| State { value, covariance: None, epoch: state.epoch };
        let mut points = Vec::with_capacity(2 * n + 1);
        points.push(at(state.value.clone()));
        for i in 0..n {
            points.push(at(&state.value + sqrt.column(i)));
        }
        for i in 0..n {
            points.push(at(&state.value - sqrt.column(i)));
        }

        let w = 1.0 / (2.0 * (nf + lambda));
        let wm0 = lambda / (nf + lambda);
        let wc0 = wm0 + (1.0 - params.alpha * params.alpha + params.beta);
        let mut wm = vec![w; 2 * n + 1];
        let mut wc = vec![w; 2 * n + 1];
        wm[0] = wm0;
        wc[0] = wc0;

        Some(SigmaPoints { points, wm, wc })
    }
}

/// Weighted mean of transformed sigma points.
fn weighted_mean<D>(values: &[na::OVector<f64, D>], wm: &[f64]) -> na::OVector<f64, D>
where
    D: na::Dim,
    na::DefaultAllocator: Allocator<D>,
{
    let (d, _) = values[0].shape_generic();
    let mut mean = na::OVector::<f64, D>::zeros_generic(d, na::Const::<1>);
    for (v, w) in values.iter().zip(wm) {
        mean.axpy(*w, v, 1.0);
    }
    mean
}

/// Weighted cross covariance sum_i wc_i (a_i - a_mean)(b_i - b_mean)^T.
fn weighted_covariance<A, B>(
    a: &[na::OVector<f64, A>], a_mean: &na::OVector<f64, A>,
    b: &[na::OVector<f64, B>], b_mean: &na::OVector<f64, B>,
    wc: &[f64],
) -> na::OMatrix<f64, A, B>
where
    A: na::Dim,
    B: na::Dim,
    na::DefaultAllocator: Allocator<A> + Allocator<B> + Allocator<A, B>,
{
    let (ra, _) = a_mean.shape_generic();
    let (rb, _) = b_mean.shape_generic();
    let mut cov = na::OMatrix::<f64, A, B>::zeros_generic(ra, rb);
    for ((ai, bi), w) in a.iter().zip(b).zip(wc) {
        cov.ger(*w, &(ai - a_mean), &(bi - b_mean), 1.0);
    }
    cov
}

// Transition Models
// Unscented (Discrete Time)
pub struct UnscentedTransition<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    pub state: StatePtr<N>,
    pub f: F<N>,
    pub q: na::OMatrix<f64, N, N>,
    pub params: UnscentedParams,
}

/// Statically sized `UnscentedTransition`.
pub type UnscentedTransitionN<const N: usize> = UnscentedTransition<na::Const<N>>;

impl<N> UnscentedTransition<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    /// `f` advances the state by a single epoch; it is applied to every sigma
    /// point once per epoch step. Uses the default `UnscentedParams`.
    pub fn new(state: StatePtr<N>, f: F<N>, q: na::OMatrix<f64, N, N>) -> Self {
        UnscentedTransition { state, f, q, params: UnscentedParams::default() }
    }

    /// Use different sigma point parameters.
    pub fn with_params(mut self, params: UnscentedParams) -> Self {
        self.params = params;
        self
    }
}

impl<N> TransitionModel<N> for UnscentedTransition<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    fn state(&mut self, epoch: &Epoch) -> &StatePtr<N> {
        let mut state = self.state.read().unwrap().clone();

        let delta = { *epoch - state.epoch }.value;
        if delta == 0 {
            return &self.state;
        }
        if delta < 0 {
            // a general nonlinear f has no inverse
            spdlog::warn!(
                "Backward propagation is not supported for unscented transitions; state is unchanged."
            );
            return &self.state;
        }

        for _ in 0..delta {
            let sigma = match SigmaPoints::new(&state, &self.params) {
                Some(sigma) => sigma,
                None => return &self.state,
            };
            let propagated: Vec<_> = sigma.points.iter().map(|p| (self.f)(p)).collect();

            let mean = weighted_mean(&propagated, &sigma.wm);
            let cov = weighted_covariance(&propagated, &mean, &propagated, &mean, &sigma.wc) + &self.q;

            state.value = mean;
            state.covariance = Some(cov);
            state.epoch = state.epoch + Epoch::new(1);
        }

        *self.state.write().unwrap() = state;
        &self.state
    }
}

// Measurement Models
// Unscented
pub struct UnscentedMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub h: H<N, M>,
    pub r: na::OMatrix<f64, M, M>,
    h_inv: Option<HInv<N, M>>,
}

/// Statically sized `UnscentedMeasurement`.
pub type UnscentedMeasurementNM<const N: usize, const M: usize> =
    UnscentedMeasurement<na::Const<N>, na::Const<M>>;

impl<N, M> UnscentedMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub fn new(h: H<N, M>, r: na::OMatrix<f64, M, M>) -> Self {
        UnscentedMeasurement { h, r, h_inv: None }
    }

    /// Supply an inverse measurement function h^-1(z) -> x.
    /// Without one, `inverse` warns and returns a zero state.
    pub fn with_inverse(mut self, h_inv: HInv<N, M>) -> Self {
        self.h_inv = Some(h_inv);
        self
    }
}

impl<N, M> MeasurementModel<N, M> for UnscentedMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    fn projection(&self, state: &State<N>) -> Observation<M> {
        Observation {
            value: (self.h)(state),
            epoch: state.epoch
        }
    }

    fn noise(&self) -> na::OMatrix<f64, M, M> {
        self.r.clone()
    }
}

impl<N, M> InverseMeasurement<N, M> for UnscentedMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    /// Uses the function supplied with `with_inverse`, if any.
    fn inverse(&self, obs: &Observation<M>) -> State<N> {
        match &self.h_inv {
            Some(h_inv) => h_inv(obs),
            None => {
                spdlog::warn!(
                    "No inverse measurement function supplied; inverse projection will return zero."
                );
                // the state dimension is only known statically
                let n = N::from_usize(N::try_to_usize().unwrap_or(0));
                State {
                    value: na::OVector::<f64, N>::zeros_generic(n, na::Const::<1>),
                    covariance: None,
                    epoch: obs.epoch,
                }
            }
        }
    }
}

// Update
pub struct UnscentedUpdate<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub state: StatePtr<N>,                          // mutable ref. to state
    pub measurement: UnscentedMeasurement<N, M>,     // measurement model
    pub transition: UnscentedTransition<N>,          // transition model (and sigma point parameters)
}

/// Statically sized `UnscentedUpdate`.
pub type UnscentedUpdateNM<const N: usize, const M: usize> =
    UnscentedUpdate<na::Const<N>, na::Const<M>>;

impl<N, M> UnscentedUpdate<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub fn new(
        measurement: UnscentedMeasurement<N, M>,
        transition: UnscentedTransition<N>,
    ) -> Self {
        let _state = transition.state.clone();
        UnscentedUpdate {
            state: _state,
            measurement,
            transition,
        }
    }
}

impl<N, M> UpdateModel<N, M> for UnscentedUpdate<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    fn state(&mut self, epoch: Option<&Epoch>) -> &StatePtr<N> {
        match epoch {
            Some(epoch) => self.transition.state(epoch),
            _ => &self.transition.state
        }
    }

    fn apply(&mut self, observation: &Observation<M>) -> &StatePtr<N> {

        // propagate state to observation epoch
        _ = self.transition.state(&observation.epoch);
        let mut state = self.state.read().unwrap().clone();

        // push sigma points through the measurement function
        let sigma = match SigmaPoints::new(&state, &self.transition.params) {
            Some(sigma) => sigma,
            None => return &self.state,
        };
        let values: Vec<_> = sigma.points.iter().map(|p| p.value.clone()).collect();
        let projected: Vec<_> = sigma.points.iter().map(|p| (self.measurement.h)(p)).collect();

        // predicted measurement, innovation covariance and cross covariance
        let z_mean = weighted_mean(&projected, &sigma.wm);
        let s = weighted_covariance(&projected, &z_mean, &projected, &z_mean, &sigma.wc) + self.measurement.noise();
        let pxz = weighted_covariance(&values, &state.value, &projected, &z_mean, &sigma.wc);

        let s_inv = match s.clone().try_inverse() {
            Some(s_inv) => s_inv,
            None => {
                spdlog::error!(
                    "Invalid update: Innovation covariance S is singular; state will not be updated."
                );
                return &self.state;
            }
        };

        // gain, state and covariance update
        let k = &pxz * s_inv;
        state.value = &state.value + &k * (&observation.value - &z_mean);
        let cov = state.covariance() - &k * &s * k.transpose();
        // remove rounding asymmetry
        state.covariance = Some((&cov + cov.transpose()) * 0.5);

        *self.state.write().unwrap() = state;
        &self.state
    }
}
