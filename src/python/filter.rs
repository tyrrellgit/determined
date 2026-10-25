//! The part every Python filter class shares.
//!
//! PyO3 classes cannot be generic, so each Python filter class in
//! `algorithms` is its own struct. Each holds a `FilterCore` around its Rust
//! filter, and its `predict`/`update`/`state`/`reset` methods forward to it,
//! so that behaviour is written once.

use pyo3::prelude::*;

use crate::common::na;
use crate::filter::Filter;
use crate::measurement::Observation;
use crate::state::StatePtr;

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;
use crate::python::measurement::PyObservation;
use crate::python::arrays::check_length;
use crate::python::callbacks::{ ErrorSlot, guarded };

/// A Rust filter, the state it shares with Python, and what is needed to run
/// it safely from Python.
pub struct FilterCore<F> {
    pub filter: F,
    state: PyState,
    /// error slot of the filter's Python model functions, if it has any
    error: Option<ErrorSlot>,
    /// number of elements an observation must have, if known
    measurement_dim: Option<usize>,
}

impl<F> FilterCore<F>
where
    F: Filter<StateType = StatePtr<na::Dyn>, ObservationType = Observation<na::Dyn>>,
{
    pub fn new(filter: F, state: PyState, error: Option<ErrorSlot>, measurement_dim: Option<usize>) -> Self {
        FilterCore { filter, state, error, measurement_dim }
    }

    /// Run `op` on the filter; if it has Python model functions and one of
    /// them raised, restore the state and re-raise.
    fn run(&mut self, op: impl FnOnce(&mut F) -> StatePtr<na::Dyn>) -> PyResult<PyState> {
        let inner = match &self.error {
            Some(error) => guarded(&mut self.filter, &self.state.inner, error, op)?,
            None => op(&mut self.filter),
        };
        Ok(PyState { inner })
    }

    pub fn predict(&mut self, epoch: &PyEpoch) -> PyResult<PyState> {
        self.run(|filter| filter.predict(&epoch.inner))
    }

    pub fn update(&mut self, observation: &PyObservation) -> PyResult<PyState> {
        if let Some(m) = self.measurement_dim {
            check_length(&observation.inner.value, m, "observation")?;
        }
        self.run(|filter| filter.update(&observation.inner))
    }

    pub fn get_state(&self) -> PyState {
        PyState { inner: self.filter.state() }
    }

    pub fn set_state(&self, state: &PyState) -> PyResult<()> {
        self.state.assign(state)
    }

    /// Zero the state and reset the covariance to identity.
    pub fn reset(&mut self) {
        self.filter.reset();
    }
}
