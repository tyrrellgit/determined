#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::{ PyArray1, PyArray2 };
use numpy::{ ToPyArray ,PyArrayMethods };

use crate::common::na as na;
use crate::epoch::Epoch;
use crate::models::TransitionModel;
use crate::state::{State, StatePtr};

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;

/// Array types bound to python
type PyVector<'py> = Bound<'py, PyArray1<f64>>;
type PyMatrix<'py> = Bound<'py, PyArray2<f64>>;

// Wrapper for Python-defined state transition - DYNAMIC only
#[pyclass(name="TransitionModel")]
pub struct PyTransitionModel {
    py_obj: Py<PyAny>,
    state: PyState, 
}

#[pymethods]
impl PyTransitionModel {

    #[new]
    pub fn new(model: Py<PyAny>, state: PyState) -> Self {
        PyTransitionModel { py_obj: model, state }
    }

    #[pyo3(name="state")]
    fn state_transition<'py>(&mut self, epoch: &'py PyEpoch) -> PyState {
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

        let mut state = self.state.inner.write().unwrap();
        let py_epoch = PyEpoch{ inner: *epoch };
        
        Python::attach(|py| {
            let py_obj = self.py_obj.bind(py);
            
            // Call Python method with state and epoch
            let result = py_obj
                .call_method("state", (py_epoch,), None)
                .expect("Failed to call state()");
            
            let result_state: PyState = result
                .extract()
                .expect("state() must return <State>");
            
            let _state = result_state.inner.read().unwrap();

            state.value = _state.value.clone();
            state.covariance = _state.covariance.clone();
            state.epoch = *epoch;

        });
        &self.state.inner
    }

    fn jacobian(&self, state: &State<na::Dyn>) -> na::OMatrix<f64, na::Dyn, na::Dyn> {
        Python::attach(|py| {
            let py_obj = self.py_obj.bind(py);
            let vec: PyVector<'_> = state.value.as_slice().to_pyarray(py);
            
            let result = py_obj
                .call_method("jacobian", (vec,), None)
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