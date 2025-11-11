use crate::common::Composable;
use crate::common::na as na;
use crate::common::Epoch;
use crate::state::State;

type Observation< int S, int O > = State< int S, int O >;

pub struct Measurement< int S, int O > {
    pub state: State< 1, S >, // nalgebra is column-major like Eigen
    pub observation: Observation< 1, O > // ---^
    pub covariance: na::Matrix< f64, S+O, S+O >,
    pub jacobian: na::Matrix< f64, R, C >,
}

impl Measurement {

    pub fn predict(&self, epoch: &Epoch) -> State {
        // Time update placeholder
        return self.state;
    }

    pub fn update(&self, observation: &Observation, filter: &Filter) -> Self {
        // Measurement update placeholder
        filter.process(observation);
        self.state = filter.state(self)
    }
}

impl Composable for Measurement {
    type Output = Measurement;

    fn add(self, other: Measurement) -> Measurement {
        /* Implementation of adding two measurements
    
        Couple things need to happen here: 
        
            1. Combined the new state covariance
            2. Combined the new state jacobian

        */
        Measurement { /* fields */ }
    }
}