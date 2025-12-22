#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::{ PyArray2 };
use numpy::{ ToPyArray ,PyArrayMethods };

use crate::common::na;
use crate::measurement::Observation;
use crate::models::UpdateModel;
use crate::state::{ State, StatePtr };

use crate::python::state::PyState;
use crate::python::measurement::PyObservation;

/// Array types bound to python
type PyMatrix<'py> = Bound<'py, PyArray2<f64>>;

#[pyclass(name="UpdateModel")]
pub struct PyUpdateModel {
    py_obj: Py<PyAny>,
    state: PyState,
}

#[pymethods]
impl PyUpdateModel {
    #[new]
    pub fn new(model: Py<PyAny>, state: PyState) -> Self {
        PyUpdateModel {
            py_obj: model,
            state: state
        }
    }

    #[pyo3(name="apply")]
    fn apply_update<'py>(&mut self, observation: &'py PyObservation) -> PyState {
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

    fn apply(&mut self, observation: &Observation<na::Dyn>) -> &StatePtr<na::Dyn> {
        
        let mut state = self.state.inner.write().unwrap();
        let py_obs = PyObservation{ inner: observation.clone() };
        
        Python::attach(|py| {
            let py_obj = self.py_obj.bind(py);
            
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

            state.value = _state.value.clone();
            state.covariance = _state.covariance.clone();
            state.epoch = _state.epoch;

        });
        &self.state.inner
    }

    fn jacobian(&self, state: &State<na::Dyn>) -> na::OMatrix<f64, na::Dyn, na::Dyn> {
        let py_state = PyState{ inner: state.ptr() };

        Python::attach(|py| {
            let py_obj = self.py_obj.bind(py);           
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