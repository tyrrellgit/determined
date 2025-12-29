//! Model traits used by the Extended Kalman Filter implementation.
//!
//! These traits allow users to supply nonlinear state transition and
//! measurement models with accompanying Jacobians for a standard
//! Extended Kalman Filter implementation.
//! 

use crate::common::na as na;
use crate::epoch::Epoch;
use crate::measurement::Observation;
use crate::state::{ State, StatePtr };

use crate::models::traits::{ 
    TransitionModel, MeasurementModel, UpdateModel
};

use crate::models::LinearMeasurement;

type F<const N: usize> = Box<dyn FnMut(&Epoch) -> na::SMatrix<f64, N, 1>>;
type J<const N: usize> = Box<dyn Fn(&State<na::Const<N>>) -> na::SMatrix<f64, N, N>>;


// Transition Models
// NonLinear (Discrete Time)
pub struct NonLinearTransition<const N: usize> {
    pub state: StatePtr<na::Const<N>>,
    pub f: F<N>,
    pub j: J<N>,
    pub q: na::SMatrix<f64, N, N>,
}

impl<const N: usize> NonLinearTransition<N> {   
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

        // propagate state
        let _value = (self.f)(epoch);

        // compute jacobian --> TODO: cloning!
        let _cov = {
            let _state = self.state.read().unwrap();
            let _f = self.jacobian(&_state.clone());
            Some(_f * _state.covariance() * _f.transpose() + self.q)
        };     

        // update internal reference
        let mut state = self.state.write().unwrap();

        state.value = _value;
        state.covariance = _cov;
        state.epoch = *epoch;

        &self.state
    }

    fn jacobian(&self, _state: &State<na::Const<N>>) -> na::SMatrix<f64, N, N> {
        (self.j)(_state)
    }
}

// Measurement
pub struct NonLinearMeasurement<const N: usize, const M: usize> {
    pub h: F<N>,
    pub j: J<N>,
    pub r: na::SMatrix<f64, N, N>,
}

// Update
pub struct NonLinearUpdate<const N: usize, const M: usize> {
    pub state: StatePtr<na::Const<N>>,               // mutable ref. to state
    pub measurement: LinearMeasurement<N, M>,        // measurement model
    pub transition: NonLinearTransition<N>,    // transition model
    identity: na::SMatrix<f64, N, N>,                // identity matrix
}

impl<const N: usize, const M: usize> NonLinearUpdate<N, M> {
    pub fn new(
        measurement: LinearMeasurement<N, M>,
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

        // compute gain matrix
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

        // Update Covariance
        state.covariance = Some((self.identity - k * h) * cov);

        &self.state

    }

    // code proper jacobian
    fn jacobian(&self, _x: &State<na::Const<N>>) -> na::SMatrix<f64, M, N> {
        self.measurement.jacobian(_x)
    }
}