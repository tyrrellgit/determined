//! Nonlinear models used by the Extended Kalman Filter implementation.
//!
//! These models are built from user supplied closures: a propagation function
//! `f` and measurement function `h`, each with an accompanying Jacobian, so the
//! EKF can linearise about the current state estimate.

use crate::common::na as na;
use crate::epoch::Epoch;
use crate::measurement::Observation;
use crate::state::{ State, StatePtr };

use crate::models::traits::{ 
    TransitionModel, MeasurementModel, UpdateModel
};

/// One-step propagation: x(t) -> x(t + 1)
type F<const N: usize> = Box<dyn FnMut(&State<na::Const<N>>) -> na::SMatrix<f64, N, 1>>;
/// Jacobian of `F` with respect to the state (N x N)
type J<const N: usize> = Box<dyn Fn(&State<na::Const<N>>) -> na::SMatrix<f64, N, N>>;

/// Measurement function: h(x) -> z
type H<const N: usize, const M: usize> = Box<dyn Fn(&State<na::Const<N>>) -> na::SMatrix<f64, M, 1>>;
/// Jacobian of `H` with respect to the state (M x N)
type JH<const N: usize, const M: usize> = Box<dyn Fn(&State<na::Const<N>>) -> na::SMatrix<f64, M, N>>;
/// Optional inverse measurement: h^-1(z) -> x
type HInv<const N: usize, const M: usize> = Box<dyn Fn(&Observation<na::Const<M>>) -> State<na::Const<N>>>;


// Transition Models
// NonLinear (Discrete Time)
pub struct NonLinearTransition<const N: usize> {
    pub state: StatePtr<na::Const<N>>,
    pub f: F<N>,
    pub j: J<N>,
    pub q: na::SMatrix<f64, N, N>,
}

impl<const N: usize> NonLinearTransition<N> {   
    /// `f` advances the state by a single epoch; it is called once per epoch
    /// step, and `j` is evaluated at the state *before* each step.
    pub fn new(
        state: StatePtr<na::Const<N>>,
        f: F<N>,
        j: J<N>,
        q: na::SMatrix<f64, N, N>
    ) -> Self {
        NonLinearTransition { state, f, j, q }
    }
}   

impl<const N: usize> TransitionModel<na::Const<N>> for NonLinearTransition<N> {
    fn state(&mut self, epoch: &Epoch) -> &StatePtr<na::Const<N>> {
        let mut state = self.state.write().unwrap();

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
            let cov = jac * state.covariance() * jac.transpose() + self.q;

            state.value = value;
            state.covariance = Some(cov);
            state.epoch = state.epoch + Epoch::new(1);
        }

        &self.state
    }

    fn jacobian(&self, _state: &State<na::Const<N>>) -> na::SMatrix<f64, N, N> {
        (self.j)(_state)
    }
}

// Measurement Models
// NonLinear
pub struct NonLinearMeasurement<const N: usize, const M: usize> {
    pub h: H<N, M>,
    pub j: JH<N, M>,
    pub r: na::SMatrix<f64, M, M>,
    h_inv: Option<HInv<N, M>>,
}

impl<const N: usize, const M: usize> NonLinearMeasurement<N, M> {   
    pub fn new(
        h: H<N, M>,
        j: JH<N, M>,
        r: na::SMatrix<f64, M, M>
    ) -> Self {
        NonLinearMeasurement{ h, j, r, h_inv: None }
    }

    /// Supply an inverse measurement function h^-1(z) -> x.
    /// Without one, `inverse` warns and returns a zero state.
    pub fn with_inverse(mut self, h_inv: HInv<N, M>) -> Self {
        self.h_inv = Some(h_inv);
        self
    }

    /// Linear sensor `z = H x` expressed as a nonlinear model, so a linear
    /// measurement can be paired with a nonlinear transition.
    pub fn linear(h: na::SMatrix<f64, M, N>, r: na::SMatrix<f64, M, M>) -> Self {
        Self::new(
            Box::new(move |state| h * state.value),
            Box::new(move |_| h),
            r,
        )
    }
} 

impl<const N: usize, const M: usize> MeasurementModel<na::Const<N>, na::Const<M>> for NonLinearMeasurement<N, M> {
    fn projection(&self, state: &State<na::Const<N>>) -> Observation<na::Const<M>> {
        Observation {
            value: (self.h)(state),
            epoch: state.epoch
        }
    }

    fn inverse(&self, obs: &Observation<na::Const<M>>) -> State<na::Const<N>> {
        match &self.h_inv {
            Some(h_inv) => h_inv(obs),
            None => {
                spdlog::warn!(
                    "No inverse measurement function supplied; inverse projection will return zero."
                );
                State {
                    value: na::SMatrix::<f64, N, 1>::zeros(),
                    covariance: None,
                    epoch: obs.epoch,
                }
            }
        }
    }

    fn jacobian(&self, state: &State<na::Const<N>>) -> na::SMatrix<f64, M, N> {
        (self.j)(state)
    }
}

// Update
pub struct NonLinearUpdate<const N: usize, const M: usize> {
    pub state: StatePtr<na::Const<N>>,               // mutable ref. to state
    pub measurement: NonLinearMeasurement<N, M>,     // measurement model
    pub transition: NonLinearTransition<N>,          // transition model
    identity: na::SMatrix<f64, N, N>,                // identity matrix
}

impl<const N: usize, const M: usize> NonLinearUpdate<N, M> {
    pub fn new(
        measurement: NonLinearMeasurement<N, M>,
        transition: NonLinearTransition<N>,
    ) -> Self {
        let _state = transition.state.clone();
        NonLinearUpdate {
            state: _state,
            measurement,
            transition,
            identity: na::SMatrix::<f64, N, N>::identity(),
        }
    }
}

impl<const N: usize, const M: usize> UpdateModel<na::Const<N>, na::Const<M>> for NonLinearUpdate<N, M> {
    fn state(&mut self, epoch: Option<&Epoch>) -> &StatePtr<na::Const<N>> {
        match epoch {
            Some(epoch) => self.transition.state(epoch),
            _ => &self.transition.state
        }
    }
    
    fn apply(&mut self, observation: &Observation<na::Const<M>>) -> &StatePtr<na::Const<N>> {

        // propagate state to observation epoch
        _ = self.transition.state(&observation.epoch);
        let mut state = self.state.write().unwrap();

        // project state to measurement domain
        let z_x = &self.measurement.projection(&state).value;
        let z = &observation.value;

        let cov = &state.covariance();

        // compute gain matrix, linearised about the current estimate
        let h = &self.measurement.jacobian(&state);
        let h_t = &h.transpose();
        let s = h * cov * h_t + self.measurement.r;
        let s_inv = match s.try_inverse(){
            Some(s_inv) => s_inv,
            _ => {
                spdlog::error!(
                    "Invalid update: Innovation covariance S is singular; state will not be updated."
                );
                return &self.state;
            },

        };
        
        // Compute gain and innovation
        let k = cov * h_t * s_inv;
        let y = z - z_x;
        
        // Update state
        state.value = state.value + k * y;

        // Update Covariance (Joseph form: stays symmetric positive semi-definite)
        let i_kh = self.identity - k * h;
        state.covariance = Some(i_kh * cov * i_kh.transpose() + k * self.measurement.r * k.transpose());

        &self.state

    }

    fn jacobian(&self, _x: &State<na::Const<N>>) -> na::SMatrix<f64, M, N> {
        self.measurement.jacobian(_x)
    }
}
