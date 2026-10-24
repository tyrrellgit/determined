#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::{ ToPyArray, PyArrayMethods };

use crate::common::na as na;
use crate::epoch::Epoch;
use crate::models::TransitionModel;
use crate::state::{State, StatePtr};

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;
use crate::python::arrays::{ PyMatrix };


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
        jac.as_slice()
            .to_pyarray(py)
            .reshape([jac.nrows(), jac.ncols()])
            .unwrap()
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
            
            let _state = result_state.inner.read().unwrap();

            {   // lock and update state
                let mut state = self.state.inner.write().unwrap();
                state.value = _state.value.clone();
                state.covariance = _state.covariance.clone();
                state.epoch = *epoch;
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