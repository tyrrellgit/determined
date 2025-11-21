use std::collections::HashMap;

use crate::measurement::{Measurement, Observation};
use crate::state::State;
use crate::common::Algorithm;
use crate::common::Epoch;

/// Generic Filter container that holds an Algorithm implementation and a set
/// of named measurements. The filter is generic over the algorithm type to
/// allow users to plug in their own filters.
pub struct Filter<T, const S: usize, const O: usize, const N: usize>
where
    T: Algorithm<StateType = State<S, 1>, ObservationType = State<O, 1>>,
{
    algorithm: T,
    measurement: Measurement<S, O, N>,
    measurements: HashMap<String, Measurement<S, O, N>>,
}

impl<T, const S: usize, const O: usize, const N: usize> Filter<T, S, O, N>
where
    T: Algorithm<StateType = State<S, 1>, ObservationType = State<O, 1>>,
{
    pub fn predict<'a>(&'a mut self, epoch: &Epoch) -> &'a State<S, 1> {
        self.algorithm.predict(epoch)
    }

    pub fn update<'a>(&'a mut self, observation: &Observation<O>) -> &'a State<S, 1> {
        // Process the observation and forward to the algorithm
        self.process(observation);
        self.algorithm.update(observation)
    }

    pub fn process(&mut self, _observation: &Observation<O>) {
        // Application-specific processing can go here. Placeholder.
    }

    pub fn state<'a>(&self, measurement: &'a Measurement<S, O, N>) -> &'a State<S, 1> {
        &measurement.state
    }
}
