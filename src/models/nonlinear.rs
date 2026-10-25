//! Nonlinear models used by the Extended Kalman Filter implementation.
//!
//! These models are built from user supplied closures: a propagation function
//! `f` and measurement function `h`, each with an accompanying Jacobian, so the
//! EKF can linearise about the current state estimate. They are generic over
//! the dimension type, so work with statically sized (`na::Const<N>`, see the
//! `...N` / `...NM` aliases) and dynamically sized (`na::Dyn`) matrices.
//!
//! The closures are run on a copy of the state, so no lock is held while user
//! code runs.

use crate::common::{ na, FilterAllocator };
use crate::epoch::Epoch;
use crate::measurement::Observation;
use crate::state::{ State, StatePtr };

use na::allocator::Allocator;

use crate::models::traits::{ 
    TransitionModel, TransitionJacobian,
    MeasurementModel, MeasurementJacobian, InverseMeasurement,
    UpdateModel,
};
use crate::models::update::kalman_update;

/// One-step propagation: x(t) -> x(t + 1)
pub type F<N> = Box<dyn FnMut(&State<N>) -> na::OVector<f64, N> + Send + Sync>;
/// Jacobian of `F` with respect to the state (N x N)
pub type J<N> = Box<dyn Fn(&State<N>) -> na::OMatrix<f64, N, N> + Send + Sync>;

/// Measurement function: h(x) -> z
pub type H<N, M> = Box<dyn Fn(&State<N>) -> na::OVector<f64, M> + Send + Sync>;
/// Jacobian of `H` with respect to the state (M x N)
pub type JH<N, M> = Box<dyn Fn(&State<N>) -> na::OMatrix<f64, M, N> + Send + Sync>;
/// Optional inverse measurement: h^-1(z) -> x
pub type HInv<N, M> = Box<dyn Fn(&Observation<M>) -> State<N> + Send + Sync>;


// Transition Models
// NonLinear (Discrete Time)
pub struct NonLinearTransition<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    pub state: StatePtr<N>,
    pub f: F<N>,
    pub j: J<N>,
    pub q: na::OMatrix<f64, N, N>,
}

/// Statically sized `NonLinearTransition`.
pub type NonLinearTransitionN<const N: usize> = NonLinearTransition<na::Const<N>>;

impl<N> NonLinearTransition<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{   
    /// `f` advances the state by a single epoch; it is called once per epoch
    /// step, and `j` is evaluated at the state *before* each step.
    pub fn new(
        state: StatePtr<N>,
        f: F<N>,
        j: J<N>,
        q: na::OMatrix<f64, N, N>
    ) -> Self {
        NonLinearTransition { state, f, j, q }
    }
}   

impl<N> TransitionModel<N> for NonLinearTransition<N>
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
                "Backward propagation is not supported for nonlinear transitions; state is unchanged."
            );
            return &self.state;
        }

        for _ in 0..delta {
            let jac = (self.j)(&state);
            let value = (self.f)(&state);
            let cov = &jac * state.covariance() * jac.transpose() + &self.q;

            state.value = value;
            state.covariance = Some(cov);
            state.epoch = state.epoch + Epoch::new(1);
        }

        *self.state.write().unwrap() = state;
        &self.state
    }
}

impl<N> TransitionJacobian<N> for NonLinearTransition<N>
where
    N: na::Dim,
    na::DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    fn jacobian(&self, _state: &State<N>) -> na::OMatrix<f64, N, N> {
        (self.j)(_state)
    }
}

// Measurement Models
// NonLinear
pub struct NonLinearMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub h: H<N, M>,
    pub j: JH<N, M>,
    pub r: na::OMatrix<f64, M, M>,
    h_inv: Option<HInv<N, M>>,
}

/// Statically sized `NonLinearMeasurement`.
pub type NonLinearMeasurementNM<const N: usize, const M: usize> =
    NonLinearMeasurement<na::Const<N>, na::Const<M>>;

impl<N, M> NonLinearMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{   
    pub fn new(
        h: H<N, M>,
        j: JH<N, M>,
        r: na::OMatrix<f64, M, M>
    ) -> Self {
        NonLinearMeasurement{ h, j, r, h_inv: None }
    }

    /// Supply an inverse measurement function h^-1(z) -> x.
    /// Without one, `inverse` returns `None`.
    pub fn with_inverse(mut self, h_inv: HInv<N, M>) -> Self {
        self.h_inv = Some(h_inv);
        self
    }

    /// Linear sensor `z = H x` expressed as a nonlinear model, so a linear
    /// measurement can be paired with a nonlinear transition.
    pub fn linear(h: na::OMatrix<f64, M, N>, r: na::OMatrix<f64, M, M>) -> Self
    where
        na::OMatrix<f64, M, N>: Send + Sync + 'static,
    {
        let jac = h.clone();
        Self::new(
            Box::new(move |state| &h * &state.value),
            Box::new(move |_| jac.clone()),
            r,
        )
    }
} 

impl<N, M> MeasurementModel<N, M> for NonLinearMeasurement<N, M>
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

impl<N, M> MeasurementJacobian<N, M> for NonLinearMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    fn jacobian(&self, state: &State<N>) -> na::OMatrix<f64, M, N> {
        (self.j)(state)
    }
}

impl<N, M> InverseMeasurement<N, M> for NonLinearMeasurement<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    /// Uses the function supplied with `with_inverse`, if any.
    fn inverse(&self, obs: &Observation<M>) -> Option<State<N>> {
        self.h_inv.as_ref().map(|h_inv| h_inv(obs))
    }
}

// Update
pub struct NonLinearUpdate<N, M>
where
    N: na::Dim,
    M: na::Dim,
    na::DefaultAllocator: FilterAllocator<N, M>,
{
    pub state: StatePtr<N>,                          // mutable ref. to state
    pub measurement: NonLinearMeasurement<N, M>,     // measurement model
    pub transition: NonLinearTransition<N>,          // transition model
}

/// Statically sized `NonLinearUpdate`.
pub type NonLinearUpdateNM<const N: usize, const M: usize> =
    NonLinearUpdate<na::Const<N>, na::Const<M>>;

impl<N, M> NonLinearUpdate<N, M>
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
        NonLinearUpdate {
            state: _state,
            measurement,
            transition,
        }
    }
}

impl<N, M> UpdateModel<N, M> for NonLinearUpdate<N, M>
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

        // project state to measurement domain, linearised about the current estimate
        let z_x = self.measurement.projection(&state).value;
        let h = self.measurement.jacobian(&state);

        if kalman_update(&mut state, &observation.value, &z_x, &h, &self.measurement.noise()) {
            *self.state.write().unwrap() = state;
        }

        &self.state
    }
}
