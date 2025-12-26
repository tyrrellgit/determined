#![allow(unsafe_op_in_unsafe_fn)]

use spdlog;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::common::na as na;
use crate::filter::Filter;
use crate::epoch::Epoch;
use crate::measurement::Observation;
use crate::state::StatePtr;

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;
use crate::python::measurement::PyObservation;


#[pyclass(name="KalmanFilter")]
pub struct PyKalmanFilter {
    update: Py<PyAny>,
    state: PyState,
}

#[pymethods]
impl PyKalmanFilter {
    #[new]
    pub fn new(update: Py<PyAny>, state: PyState) -> Self {
        PyKalmanFilter {
            update: update,
            state: state
        }
    }

    #[pyo3(name="predict")]
    fn predict_state(&mut self, epoch: &PyEpoch) -> PyState {
        let state = self.predict(&epoch.inner).clone();
        PyState { inner: state }
    }

    #[pyo3(name="update")]
    fn update_state(&mut self, observation: &PyObservation) -> PyState {
        let state = self.update(&observation.inner).clone();
        PyState { inner: state }
    }

    #[getter] //TODO: conflict with Filter.state()
    fn get_state(&self) -> PyState {
        let _state = self.state();
        PyState { inner: _state }
    }

    #[setter]
    fn set_state(&mut self, state: PyState) {

        let py_state = state.inner.read().unwrap();
        let mut inner_state = self.state.inner.write().unwrap();

        inner_state.value = py_state.value.clone();
        inner_state.covariance = py_state.covariance.clone();
        inner_state.epoch = py_state.epoch;

    }

    #[pyo3(name="reset")]
    fn reset_state(&mut self) {
            self.reset();
        }
}

impl Filter for PyKalmanFilter {

    type ObservationType = Observation<na::Dyn>;
    type StateType = StatePtr<na::Dyn>;

    fn predict(&mut self, epoch: &Epoch) -> Self::StateType {

        let py_epoch = PyEpoch { inner: *epoch };

        Python::attach(| py | {

            let mut state = self.state.inner.write().unwrap();

            let py_obj = self.update.bind(py);
            let result = py_obj
                .call_method("state", (py_epoch,), None)
                .expect("Failed to call update.state()");

            let result_state: PyState = result
                .extract()
                .expect("update.state() must return <State>");

            let _state = result_state.inner.read().unwrap();

            // Update internal state
            state.value = _state.value.clone();
            state.covariance = _state.covariance.clone();
            state.epoch = *epoch;

        });

        // return a reference counted ptr
        self.state.inner.clone()
    }

    fn update(&mut self, observation: &Observation<na::Dyn>) -> Self::StateType {

        let py_observation = PyObservation { inner: observation.clone() };

        Python::attach(| py | {
            
            let mut state = self.state.inner.write().unwrap();

            let py_obj = self.update.bind(py);
            let result = py_obj
                .call_method("apply", (py_observation,), None)
                .expect("Failed to call update.apply()");

            let result_state: PyState = result
                .extract()
                .expect("update.apply() must return <State>");

            let _state = result_state.inner.read().unwrap();

            // Update internal state
            state.value = _state.value.clone();
            state.covariance = _state.covariance.clone();
            state.epoch = _state.epoch;

        });

        // return a reference counted ptr
        self.state.inner.clone()
    }

    fn state(&self) -> Self::StateType {
        // return a reference counted ptr
        self.state.inner.clone()
    }

    fn reset(&mut self) {
        // make this call optional such that the expects dont fail shoudl the method not exists
        Python::attach(| py| {
            let py_obj = self.update.bind(py);
            let result = py_obj
                .call_method("reset", (), None);

            let _ = match result {
                Ok(res) => res,
                Err(_) => {
                    spdlog::warn!("No reset() method found.");
                    return;
                }
            };
        });
    }
}

