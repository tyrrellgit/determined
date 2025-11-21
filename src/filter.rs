//! `Filter` is a small convenience wrapper that pairs an `Algorithm` with a
//! set of named measurements. It's currently minimal and provided as an easy
//! place to attach application-specific measurement routing or multi-sensor
//! bookkeeping.
//!
//! Note: in the current tree `Filter` is present for examples and may be
//! unused by the core algorithms. It can be removed safely if you prefer a
//! slimmer crate surface.

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
    /// Forward the predict call to the underlying algorithm.
    pub fn predict<'a>(&'a mut self, epoch: &Epoch) -> &'a State<S, 1> {
        self.algorithm.predict(epoch)
    }

    /// Forward an update with an observation to the algorithm.
    pub fn update<'a>(&'a mut self, observation: &Observation<O>) -> &'a State<S, 1> {
        // Process the observation and forward to the algorithm
        self.process(observation);
        self.algorithm.update(observation)
    }

    /// Placeholder hook for application-specific processing (e.g. gating,
    /// buffering or measurement association).
    pub fn process(&mut self, _observation: &Observation<O>) {
        // Application-specific processing can go here. Placeholder.
    }

    /// Access the state part of a measurement reference.
    pub fn state<'a>(&self, measurement: &'a Measurement<S, O, N>) -> &'a State<S, 1> {
        &measurement.state
    }
}
