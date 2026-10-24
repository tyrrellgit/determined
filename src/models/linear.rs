//! Model traits used by the Extended Kalman Filter implementation.
//!
//! These traits allow users to supply nonlinear state transition and
//! measurement models with accompanying Jacobians so the EKF can operate.
//! 

use crate::common::na as na;
use crate::epoch::Epoch;
use crate::measurement::Observation;
use crate::state::{ State, StatePtr };

use crate::common::linalg::{ moore_penrose_right_inverse };

use crate::models::traits::{ 
    TransitionModel, MeasurementModel, UpdateModel, DefaultFromState
};

// Transition Models
// Linear (Discrete Time)
#[derive(Clone, Debug)]
pub struct LinearTransition<const N: usize> {
    pub state: StatePtr<na::Const<N>>,    // state vector (N x 1)
    pub f: na::SMatrix<f64, N, N>,             // state transition (N x N)
    pub q: na::SMatrix<f64, N, N>,             // process noise covariance (N x N)
    f_inv: Option<na::SMatrix<f64, N, N>>,     // inverse of F
}

impl<const N: usize> LinearTransition<N> { 
    pub fn new(
        state: StatePtr<na::Const<N>>,
        f: na::SMatrix<f64, N, N>,
        q: na::SMatrix<f64, N, N> ) -> Self {

        let f_inv = match f.try_inverse() {
            None => { 
                spdlog::warn!("State transition matrix F is non-invertible;
                backward propagation will do nothing.");
                None
            },
            Some(f_inv) => Some(f_inv),
        };
        LinearTransition { state, f, q, f_inv }
    }

    fn f_inv(&self) -> na::SMatrix<f64, N, N> {
        // getter for F inverse with default zero matrix
        self.f_inv.unwrap_or(na::SMatrix::<f64, N, N>::zeros())
    }
}   

impl<const N: usize> DefaultFromState for LinearTransition<N> {

    type DefaultType = LinearTransition<N>;
    type StateType = StatePtr<na::Const<N>>;

    fn default_from_state(state: Self::StateType) -> Self::DefaultType
    where Self: Sized {
        LinearTransition::new(
            state,
            na::SMatrix::<f64, N, N>::identity(),
            na::SMatrix::<f64, N, N>::zeros(),
        )
    }
}

impl<const N: usize> TransitionModel<na::Const<N>> for LinearTransition<N> {

    fn state(&mut self, epoch: &Epoch) -> &StatePtr<na::Const<N>> {
        // Determine timesteps to propagate
        let mut state = self.state.write().unwrap();

        let delta = { *epoch - state.epoch }.value;
        if delta == 0 {
            return &self.state;
        }

        let cov =  &state.covariance();

        if delta > 0 {
            // propagate forward
            for _ in 0..delta {
                state.value = self.f * state.value;
                state.covariance = Some(self.f * cov * self.f.transpose() + self.q);
            }
        } else {
            // propagate backward
            let f_inv = self.f_inv();
            for _ in 0..delta.abs() {
                state.value = f_inv * state.value;
                state.covariance = Some(f_inv * (cov - self.q) * f_inv.transpose());
            }
        };
        &self.state
    }

    fn jacobian(&self, _state: &State<na::Const<N>>) -> na::SMatrix<f64, N, N> {
        self.f
    }
}

/// Measurement Models
/// Linear
#[derive(Clone, Debug)]
pub struct LinearMeasurement<const N: usize, const M: usize> {
    pub h: na::SMatrix<f64, M, N>,             // measurement matrix
    pub r: na::SMatrix<f64, M, M>,             // measurement noise covariance(M x M)
    h_t: na::SMatrix<f64, N, M>,               // transpose of H
    h_inv: na::SMatrix<f64, N, M>,             // inverse of H
}

impl<const N: usize, const M: usize> LinearMeasurement<N, M> {
    pub fn new(h: na::SMatrix<f64, M, N>, r: na::SMatrix<f64, M, M>) -> Self {
        let h_inv = match moore_penrose_right_inverse(&h) {
            None => {
                spdlog::warn!("Measurement matrix H is non-invertible; inverse projection will return zero.");
                na::SMatrix::<f64, N, M>::zeros()
            },
            Some(h_inv) => h_inv,
        };
        LinearMeasurement {
            h,
            r,
            h_t: h.transpose(),
            h_inv,
        }
    }
}


impl<const N: usize, const M: usize> Default for LinearMeasurement<N, M> {
    fn default() -> Self {
        let _h = na::SMatrix::<f64, M, N>::identity();
        LinearMeasurement {
            h: _h,
            r: na::SMatrix::<f64, M, M>::identity(),
            h_t: _h.transpose(),
            h_inv:  moore_penrose_right_inverse(&_h).unwrap_or(na::SMatrix::<f64, N, M>::zeros()),
        }
    }
}

impl<const N: usize, const M: usize> MeasurementModel<na::Const<N>, na::Const<M>> for LinearMeasurement<N, M> {
    fn projection(&self, state: &State<na::Const<N>>) -> Observation<na::Const<M>> {
        Observation {
            value: self.h * state.value,
            epoch: state.epoch
        }
    }

    fn inverse(&self, obs: &Observation<na::Const<M>>) -> State<na::Const<N>> {
        State {
            value: self.h_inv * obs.value,
            covariance: None,
            epoch: obs.epoch,
        }
    } // TODO: probably dont need this since its could often be ill-defined

    fn jacobian(&self, _state: &State<na::Const<N>>) -> na::SMatrix<f64, M, N> {
        self.h // TODO: code proper jacobian
    }
}

// Update Models
// Linear
pub struct LinearUpdate<const N: usize, const M: usize > {
    pub state: StatePtr<na::Const<N>>,         // mutable ref. to state
    pub measurement: LinearMeasurement<N, M>,    // measurement model
    pub transition: LinearTransition<N>,         // transition model
    identity: na::SMatrix<f64, N, N>,                // identity matrix
}

impl<const N: usize, const M: usize> LinearUpdate<N, M> {
    pub fn new(
        state: StatePtr<na::Const<N>>,
        measurement: LinearMeasurement<N, M>,
        transition: LinearTransition<N>,
    ) -> Self {
        LinearUpdate {
            state,
            measurement,
            transition,
            identity: na::SMatrix::<f64, N, N>::identity(),
        }
    }
}

impl<const N: usize, const M: usize> DefaultFromState for LinearUpdate<N, M> {

    type DefaultType = LinearUpdate<N, M>;
    type StateType = StatePtr<na::Const<N>>;

    fn default_from_state(state: Self::StateType) -> Self::DefaultType {
        LinearUpdate::new(
            state.clone(),
            LinearMeasurement::<N, M>::default(),
            LinearTransition::<N>::default_from_state(state)
        )
    }
}

impl<const N: usize, const M: usize> UpdateModel<na::Const<N>, na::Const<M>> for LinearUpdate<N, M> {

    fn state(&mut self, epoch: &Epoch) -> &StatePtr<na::Const<N>> {
        self.transition.state(epoch)
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
        let h = &self.measurement.h;
        let h_t = &self.measurement.h_t;
        let s = h * cov * h_t + self.measurement.r;
        let s_inv = match s.try_inverse(){
            None => {
                spdlog::error!(
                    "Invalid update: Innovation covariance S is singular; state will not be updated."
                );
                return &self.state;
            }
            Some(s_inv) => s_inv,
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

    fn jacobian(&self, _x: &State<na::Const<N>>) -> na::SMatrix<f64, M, N> {
        self.measurement.h
    }
}