//! Measurement helpers and observation types.
//!
//! `Measurement` is a small container that groups a state, observation and a
//! joint covariance / jacobian. It provides convenience constructors and a
//! minimal update/predict placeholder useful for tests and examples.

use crate::common::na as na;
use crate::common::Epoch;
use crate::state::State;

/// A convenience alias for a column vector observation of length `O`.
pub type Observation<O> = State<O>;

#[derive(Clone, Debug)]
pub struct Measurement<const S: usize, const O: usize, const N: usize> {
    pub state: State<na::Const<S>>,                 // state vector (S x 1)
    pub observation: State<na::Const<O>>,           // observation vector (O x 1)
    pub covariance: na::SMatrix<f64, N, N>, // joint covariance (N x N), N should == S + O
    pub jacobian: na::SMatrix<f64, O, S>,   // measurement jacobian (O x S)

}

// OMatrix allocator for <Dim, Dim> is not directly implemented

impl<const S: usize, const O: usize, const N: usize> Measurement<S, O, N> {
    pub fn new(state: State<na::Const<S>>, observation: State<na::Const<O>>) -> Self {
        Measurement {
            state,
            observation,
            covariance: na::SMatrix::<f64, N, N>::identity(),
            jacobian: na::SMatrix::<f64, O, S>::zeros(),
        }
    }

    /// TODO: Time-predict the measurement's state
    pub fn predict(&self, epoch: &Epoch) -> State<na::Const<S>> {
        let mut predicted = self.state.clone();
        predicted.epoch = epoch.value;
        predicted
    }

    /// TODO: Incorporate a new observation into this measurement
    pub fn update(&mut self, observation: &State<na::Const<O>>) {
        self.observation = observation.clone();
    }
}