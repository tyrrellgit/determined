use std::collections::HashMap;

use crate::composition::Composable;
use crate::measurement::{Measurement, Observation};
use crate::state::State;
use crate::algorithm::Algorithm;

pub struct Filter {
    algorithm: Algorithm,
    measurement: Measurement,
    measurements: HashMap<String, Measurement>,
}

impl Filter {

    pub fn predict(&self, epoch: &Epoch) -> State {
        // Time update placeholder
        self.algorithm.predict(epoch)
    }

    pub fn update(&self, observation: &Observation) -> Self {
        // Measurement update placeholder
        self.process(observation);
        self.algorithm.update(observation)
    }

    pub fn process(&self, observation: &Observation) {
        // Process the observation
    }

    pub fn state(&self, measurement: &Measurement) -> State {
        // Return the state based on the measurement
        State { /* fields */ }
    }
}

impl Composable< Measurement > for Filter {
    type Output = Filter;
    fn add(mut self, other: Measurement) -> Filter {
        self.measurements.insert(other.id.clone(), other);
        self.measurement.add(measurement);
        return self;
    }
}
