use crate::common::na as na;
use crate::common::Epoch;
use crate::state::State;
use crate::common::Composable;

pub type Observation<const O: usize> = State<O, 1>;

#[derive(Clone, Debug)]
pub struct Measurement<const S: usize, const O: usize, const N: usize> {
    pub state: State<S, 1>,                 // state vector (S x 1)
    pub observation: State<O, 1>,           // observation vector (O x 1)
    pub covariance: na::SMatrix<f64, N, N>, // joint covariance (N x N), N should == S + O
    pub jacobian: na::SMatrix<f64, O, S>,   // measurement jacobian (O x S)
}

impl<const S: usize, const O: usize, const N: usize> Measurement<S, O, N> {
    /// Construct a Measurement, validating that the provided `N` matches
    /// the intended combined size `S + O`. If the sizes don't match this
    /// constructor will panic. Covariance will be initialized to identity
    /// and jacobian to zeros by default; callers can replace them.
    pub fn new(state: State<S, 1>, observation: State<O, 1>) -> Self {
        // runtime check: ensure N == S + O
        if N != (S + O) {
            panic!("Measurement::new: N must equal S + O (got N={} but S+O={})", N, S + O);
        }

        Measurement {
            state,
            observation,
            covariance: na::SMatrix::<f64, N, N>::identity(),
            jacobian: na::SMatrix::<f64, O, S>::zeros(),
        }
    }

    /// Time-predict the measurement's state (placeholder).
    /// Returns a new State<S,1> with epoch set to the provided epoch.
    pub fn predict(&self, epoch: &Epoch) -> State<S, 1> {
        let mut predicted = self.state.clone();
        predicted.epoch = epoch.value;
        predicted
    }

    /// Incorporate a new observation into this measurement (minimal placeholder).
    /// This intentionally avoids coupling to any concrete Filter trait; actual
    /// filter processing should be done externally by passing this measurement
    /// and/or its observation to the appropriate algorithm.
    pub fn update(&mut self, observation: &State<O, 1>) {
        self.observation = observation.clone();
    }
}

impl<const S: usize, const O: usize, const N: usize> Composable for Measurement<S, O, N> {
    type Output = Measurement<S, O, N>;

    /// Combine two measurements by composing their components.
    /// - states and observations are combined via their Composable::add impls
    /// - covariance and jacobian are summed (simple fusion; replace with
    ///   more appropriate combination if needed)
    fn add(self, other: Measurement<S, O, N>) -> Measurement<S, O, N> {
        Measurement {
            state: self.state.add(other.state),
            observation: self.observation.add(other.observation),
            covariance: self.covariance + other.covariance,
            jacobian: self.jacobian + other.jacobian,
        }
    }
}