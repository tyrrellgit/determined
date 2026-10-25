#![allow(unsafe_op_in_unsafe_fn)]

//! Python-implemented models: wrappers that let a Python object implement the
//! Rust model traits from `crate::models`, so it can be used with
//! `CustomFilter`. They mirror the Rust traits:
//!
//! | Wrapper            | Rust traits                                                  | Python methods                              |
//! |--------------------|--------------------------------------------------------------|---------------------------------------------|
//! | `TransitionModel`  | `TransitionModel`, `TransitionJacobian`                      | `state(epoch)`, `jacobian(state)`           |
//! | `MeasurementModel` | `MeasurementModel`, `MeasurementJacobian`, `InverseMeasurement` | `projection(state)`, `noise()`, `jacobian(state)`, `inverse(observation)` |
//! | `UpdateModel`      | `UpdateModel`                                                | `apply(observation)`                        |
//!
//! Each Python method is only called when the matching Rust method is.

use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::PyArrayMethods;

use crate::common::na;
use crate::epoch::Epoch;
use crate::measurement::Observation;
use crate::models::{
    TransitionModel, TransitionJacobian,
    MeasurementModel, MeasurementJacobian, InverseMeasurement,
    UpdateModel,
};
use crate::state::{ State, StatePtr };

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;
use crate::python::measurement::PyObservation;
use crate::python::arrays::{ PyMatrix, to_dmatrix, to_pymatrix };


// Wrapper for Python-defined state transition - DYNAMIC only
#[pyclass(name="TransitionModel")]
pub struct PyTransitionModel {
    pub model: Py<PyAny>,
    pub state: PyState, 
}

impl Clone for PyTransitionModel {
    fn clone(&self) -> Self {
        Python::attach(| py| {
            let _ref = self.model.clone_ref(py);
            Self {
                state: self.state.clone(),
                model: _ref,
            }
        })
    }
}

#[pymethods]
impl PyTransitionModel {

    #[new]
    #[pyo3(signature = (model, *, state))]
    pub fn new(model: Py<PyAny>, state: PyState) -> Self {
        PyTransitionModel { model: model, state }
    }

    #[getter]
    fn get_model(&self) -> &Py<PyAny> {
        &self.model
    }

    #[pyo3(name="state")]
    fn state_transition(&mut self, epoch: &PyEpoch) -> PyState {
        let state = self.state(&epoch.inner).clone();
        PyState { inner: state }
    }

    #[pyo3(name="jacobian")]
    fn jacobian_matrix<'py>(&mut self, py: Python<'py>, state: &'py PyState) -> PyMatrix<'py> {
        let state_data = state.inner.read().unwrap();
        let jac = self.jacobian(&state_data);
        to_pymatrix(py, &jac)
    }

}


impl TransitionModel<na::Dyn> for PyTransitionModel {
    fn state(&mut self, epoch: &Epoch) -> &StatePtr<na::Dyn> {

        let py_epoch = PyEpoch{ inner: *epoch };
        
        Python::attach(|py| {
            let py_obj = self.model.bind(py);
            
            // Call Python method with state and epoch
            let result = py_obj
                .call_method("state", (py_epoch,), None)
                .expect("Failed to call state()");
            
            let result_state: PyState = result
                .extract()
                .expect("state() must return <State>");
            
            // copy out first: the returned state may share this state's lock
            let _state = result_state.inner.read().unwrap().clone();

            {   // lock and update state
                let mut state = self.state.inner.write().unwrap();
                state.value = _state.value;
                state.covariance = _state.covariance;
                state.epoch = *epoch;
            }

        });
        &self.state.inner
    }
}

impl TransitionJacobian<na::Dyn> for PyTransitionModel {
    fn jacobian(&self, state: &State<na::Dyn>) -> na::OMatrix<f64, na::Dyn, na::Dyn> {
        let py_state = PyState{ inner: state.ptr() };

        Python::attach(|py| {
            let py_obj = self.model.bind(py);
            let result = py_obj
                .call_method(
                    "jacobian",
                    (py_state,),
                    None)
                .expect("Failed to call jacobian()");
            
            let result_vec: PyMatrix<'_> = result
                .extract()
                .expect("jacobian() must return ndarray");
            
            // numpy -> ndarray (zero copy view) -> nalgebra (single copy)
            to_dmatrix(unsafe { result_vec.as_array() })
        })
    }
}


// Wrapper for Python-defined Measurement models
#[pyclass(name="MeasurementModel")]
pub struct PyMeasurementModel {
    pub model: Py<PyAny>,
}

impl Clone for PyMeasurementModel {
    fn clone(&self) -> Self {
        Python::attach(| py| {
            let _ref = self.model.clone_ref(py);
            Self {
                model: _ref,
            }
        })
    }
}

#[pymethods]
impl PyMeasurementModel {
    
    #[new]
    pub fn new(model: Py<PyAny>) -> Self {
        PyMeasurementModel { model: model }
    }

    #[getter]
    fn get_model(&self) -> &Py<PyAny> {
        &self.model
    }

    #[pyo3(name="projection")]
    pub fn project_state(&mut self, state: &PyState) -> PyObservation {
        let state_data = state.inner.read().unwrap();
        let proj = self.projection(&state_data);
        PyObservation { inner: proj }
    }

    #[pyo3(name="inverse")]
    pub fn project_observation(&mut self, observation: PyObservation) -> PyState {
        let proj = self.inverse(&observation.inner);
        PyState { inner: proj.ptr() }
    }

    #[pyo3(name="jacobian")]
    pub fn jacobian_matrix<'py>(&mut self, py: Python<'py>, state: &PyState) -> PyMatrix<'py> {
        let state_data = state.inner.read().unwrap();
        let jac = self.jacobian(&state_data);
        to_pymatrix(py, &jac)
    }

    #[pyo3(name="noise")]
    pub fn noise_matrix<'py>(&mut self, py: Python<'py>) -> PyMatrix<'py> {
        to_pymatrix(py, &self.noise())
    }
}

impl MeasurementModel<na::Dyn, na::Dyn> for PyMeasurementModel {
    fn projection(&self, state: &State<na::Dyn>) -> Observation<na::Dyn> {
        let py_state = PyState { inner: state.ptr() };

        Python::attach(|py| {
            let py_obj = self.model.bind(py);
            
            let result = py_obj
                .call_method(
                    "projection",
                    (py_state,),
                    None
                )
                .expect("Failed to call projection()");

            let obs: PyObservation = result
                .extract()
                .expect("projection() must return <Observation>");

            obs.inner
        })
    }

    fn noise(&self) -> na::OMatrix<f64, na::Dyn, na::Dyn> {
        Python::attach(|py| {
            let py_obj = self.model.bind(py);
            let result = py_obj
                .call_method0("noise")
                .expect("Failed to call noise()");

            let result_mat: PyMatrix<'_> = result
                .extract()
                .expect("noise() must return ndarray");

            // numpy -> ndarray (zero copy view) -> nalgebra (single copy)
            to_dmatrix(unsafe { result_mat.as_array() })
        })
    }
}

impl MeasurementJacobian<na::Dyn, na::Dyn> for PyMeasurementModel {
    fn jacobian(&self, state: &State<na::Dyn>) -> na::OMatrix<f64, na::Dyn, na::Dyn> {
        let py_state = PyState{ inner: state.ptr() };
        
        Python::attach(|py| {
            let py_obj = self.model.bind(py);            
            let result = py_obj
                .call_method(
                    "jacobian",
                    (py_state,),
                    None)
                .expect("Failed to call jacobian()");
            
            let result_vec: PyMatrix<'_> = result
                .extract()
                .expect("jacobian() must return <ndarray>");

            // numpy -> ndarray (zero copy view) -> nalgebra (single copy)
            to_dmatrix(unsafe { result_vec.as_array() })
        })
    }
}

impl InverseMeasurement<na::Dyn, na::Dyn> for PyMeasurementModel {
    fn inverse(&self, obs: &Observation<na::Dyn>) -> State<na::Dyn> {

        let py_obs = PyObservation {
                inner: Observation {
                    value: obs.value.clone(),
                    epoch: obs.epoch,
                }
            };

        Python::attach(|py| {
            let py_obj = self.model.bind(py);

            let result = py_obj
                .call_method(
                    "inverse",
                    (py_obs,),
                    None
                )
                .expect("Failed to call inverse()");

            let state: PyState = result
                .extract()
                .expect("inverse() must return <State>");

            state.inner.read().unwrap().clone()
        })
    }
}


// Wrapper for Python-defined update models
#[pyclass(name="UpdateModel")]
pub struct PyUpdateModel {
    pub model: Py<PyAny>,
    pub state: PyState,
    pub transition: PyTransitionModel,
    pub measurement: PyMeasurementModel,
}

impl Clone for PyUpdateModel {
    fn clone(&self) -> Self {
        Python::attach(| py| {
            let _ref = self.model.clone_ref(py);
            Self {
                model: _ref,
                state: self.state.clone(),
                transition: self.transition.clone(),
                measurement: self.measurement.clone(),
            }
        })
    }
}

#[pymethods]
impl PyUpdateModel {
    #[new]
    #[pyo3(signature = (model, *, transition, measurement))]
    pub fn new(
        model: Py<PyAny>,
        transition: Py<PyTransitionModel>,
        measurement: Py<PyMeasurementModel>) -> Self {
        
        Python::attach(|py| {

            let _transition: PyTransitionModel = transition.extract(py).unwrap();
            let _measurement: PyMeasurementModel = measurement.extract(py).unwrap();

            PyUpdateModel {
                model: model,
                state: _transition.state.clone(),
                transition: _transition,
                measurement: _measurement,
            }
        })
    }

    #[getter]
    fn get_model(&self) -> &Py<PyAny> {
        &self.model
    }

    #[pyo3(name="state")]
    fn state_transition(&mut self, epoch: &PyEpoch) -> PyState {
        let state = self.state(Some(&epoch.inner));
        PyState { inner: state.clone() }
    }

    #[pyo3(name="apply")]
    fn apply_update(&mut self, observation: &PyObservation) -> PyState {
        let obs = &observation.inner;
        let state = self.apply(obs).clone();
        PyState { inner: state }
    }
}

impl UpdateModel<na::Dyn, na::Dyn> for PyUpdateModel {

    fn state(&mut self, epoch: Option<&Epoch>) -> &StatePtr<na::Dyn> {
        match epoch {
            Some(epoch) => self.transition.state(epoch),
            _ => &self.transition.state.inner
        }
    }

    fn apply(&mut self, observation: &Observation<na::Dyn>) -> &StatePtr<na::Dyn> {
        
        let py_obs = PyObservation{ inner: observation.clone() };
        
        Python::attach(|py| {
            let py_obj = self.model.bind(py);
            
            // Call Python method with state and epoch
            let result = py_obj
                .call_method(
                    "apply",
                    (py_obs,),
                    None)
                .expect("Failed to call apply()");
            
            let result_state: PyState = result
                .extract()
                .expect("apply() must return <State>");
            
            // copy out first: the returned state may share this state's lock
            let _state = result_state.inner.read().unwrap().clone();

            {   // lock and update state
                *self.state.inner.write().unwrap() = _state;
            }

        });
        &self.state.inner
    }
}
