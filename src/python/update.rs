#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::{ ToPyArray, PyArrayMethods };

use crate::common::na;
use crate::measurement::Observation;
use crate::models::{TransitionModel, UpdateModel};
use crate::state::{ State, StatePtr };

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;
use crate::python::transition::PyTransitionModel;
use crate::python::measurement::{PyMeasurementModel, PyObservation};
use crate::python::arrays::{ PyMatrix };


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
        let state = self.state(&epoch.inner);
        PyState { inner: state.clone() }
    }

    #[pyo3(name="apply")]
    fn apply_update(&mut self, observation: &PyObservation) -> PyState {
        let obs = &observation.inner;
        let state = self.apply(obs).clone();
        PyState { inner: state }
    }

    #[pyo3(name="jacobian")]
    fn jacobian_matrix<'py>(&mut self, py: Python<'py>, state: &'py PyState) -> PyMatrix<'py> {
        let state_data = state.inner.read().unwrap();
        let jac = self.jacobian(&state_data);
        jac.as_slice()
            .to_pyarray(py)
            .reshape([jac.nrows(), jac.ncols()])
            .unwrap()
    }
}

impl UpdateModel<na::Dyn, na::Dyn> for PyUpdateModel {

    fn state(&mut self, epoch: &crate::epoch::Epoch) -> &StatePtr<na::Dyn> {
        self.transition.state(epoch)
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
            
            let _state = result_state.inner.read().unwrap();
            
            {   // lock and update state
                let mut state = self.state.inner.write().unwrap();
                state.value = _state.value.clone();
                state.covariance = _state.covariance.clone();
                state.epoch = _state.epoch;
            }

        });
        &self.state.inner
    }

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
            
            // numpy -> ndarray (zero copy view)
            let array = unsafe { result_vec.as_array() };
            let (nrows, ncols) = array.dim();
        
            // ndarray -> nalgebra (single copy, row-major)
            na::DMatrix::<f64>::from_iterator(
                nrows,
                ncols,
                array.iter().copied()
            )
        })
    }
}