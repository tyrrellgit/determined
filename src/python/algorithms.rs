#![allow(unsafe_op_in_unsafe_fn)]

use spdlog;

use pyo3::prelude::*;

use crate::common::na as na;
use crate::algorithms::KalmanFilterT;
use crate::filter::Filter;

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;
use crate::python::measurement::PyObservation;
use crate::python::update::PyUpdateModel;


/// Filter built from a Python `UpdateModel`: the Python equivalent of a Rust
/// `KalmanFilterT` with your own update model. The update maths is whatever
/// your model implements. See `KalmanFilter` for the Rust-computed filter.
#[pyclass(name="CustomFilter")]
pub struct PyCustomFilter {
    filter: KalmanFilterT<na::Dyn, na::Dyn, PyUpdateModel>,
    state: PyState,
}

#[pymethods]
impl PyCustomFilter {
    #[new]
    pub fn new(update: PyRef<'_, PyUpdateModel>) -> Self {
        let update = update.clone();
        let state = update.state.clone();
        PyCustomFilter {
            filter: KalmanFilterT::new(update),
            state,
        }
    }

    fn predict(&mut self, epoch: &PyEpoch) -> PyState {
        PyState { inner: self.filter.predict(&epoch.inner) }
    }

    fn update(&mut self, observation: &PyObservation) -> PyState {
        PyState { inner: self.filter.update(&observation.inner) }
    }

    #[getter]
    fn get_state(&self) -> PyState {
        PyState { inner: self.filter.state() }
    }

    #[setter]
    fn set_state(&mut self, state: PyState) -> PyResult<()> {
        self.state.assign(&state)
    }

    /// Calls the update model's `reset()`, if it has one. The state belongs to
    /// your models, so the filter does not reset it itself.
    fn reset(&mut self) {
        Python::attach(|py| {
            if self.filter.update.model.bind(py).call_method0("reset").is_err() {
                spdlog::warn!("No update.reset() method found.");
            }
        });
    }
}
