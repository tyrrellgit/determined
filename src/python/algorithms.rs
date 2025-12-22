#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::{ PyArray1, PyArray2, PyReadonlyArray1 };
use numpy::{ ToPyArray ,PyArrayMethods };

use crate::common::na as na;
use crate::filter::Filter;
use crate::epoch::Epoch;
use crate::measurement::Observation;
use crate::models::MeasurementModel;
use crate::state::StatePtr;

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;
use crate::python::measurement::PyObservation;

/// Array types bound to python
type PyVector<'py> = Bound<'py, PyArray1<f64>>;
type PyMatrix<'py> = Bound<'py, PyArray2<f64>>;

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
    fn predict_state(&mut self, observation: &PyObservation) -> PyState {
        todo!()
    }

    #[pyo3(name="update")]
    fn update_state(&mut self, observation: &PyObservation) -> PyState {
        todo!()
    }

    #[getter]
    fn state(&self) -> PyState {
        self.state.clone()
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
        todo!()
    }

    fn update(&mut self, observation: &Observation<na::Dyn>) -> Self::StateType {
        todo!()
    }

    fn state(&self) -> Self::StateType {
        self.state.inner.clone()
    }

    fn reset(&mut self) {
        todo!()
    }
}

