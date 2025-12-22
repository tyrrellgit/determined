#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::{ PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2 };
use numpy::{ ToPyArray ,PyArrayMethods };

use crate::common::na as na;
use crate::common::Epoch;
use crate::models::TransitionModel;
use crate::state::{State, StatePtr};

/// Array types bound to python
type PyVector<'py> = Bound<'py, PyArray1<f64>>;
type PyMatrix<'py> = Bound<'py, PyArray2<f64>>;


#[pyclass(name="Epoch")]
#[derive(Clone, Debug)]
pub struct PyEpoch {
    inner: Epoch,
}

#[pyclass(name="State")]
#[derive(Clone, Debug)]
pub struct PyState {
    inner: StatePtr<na::Dyn>,
}

#[pymethods]
impl PyEpoch {
    #[new]
    fn new(value: i64) -> PyEpoch {
        PyEpoch{
            inner: Epoch::new(value)
        }
    }  

    #[getter]
    fn value(&self) -> i64 {
        self.inner.value
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }
}

#[pymethods]
impl PyState {
    
    #[new]
    fn new(
        value: PyReadonlyArray1<f64>,
        covariance: Option<PyReadonlyArray2<f64>>,
        epoch: &PyEpoch
    ) -> PyResult<Self> {
        
    // numpy -> ndarray (zero-copy view) -> nalgebra
    let array = value.as_array();
    let vector = na::DVector::<f64>::from_iterator(
        array.len(),
        array.iter().copied()
    );
    
    // Same for covariance
    let cov = covariance.map(|cov| {
        let nd_cov = cov.as_array();
        let (nrows, ncols) = nd_cov.dim();
        
        // ndarray -> nalgebra (single copy, row-major)
        na::DMatrix::<f64>::from_iterator(
            nrows,
            ncols,
            nd_cov.iter().copied()
        )
    });
        
        Ok(PyState {
            inner: State {
                value: vector,
                covariance: cov,
                epoch: epoch.inner,
            }.ptr()
        })
    }

    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> PyVector<'py> {
        // nalgebra -> numpy
        let state = self.inner.read().unwrap();
        state.value.as_slice().to_pyarray(py)
    }

    #[getter]
    fn epoch(&self) -> i64 {
        let state = self.inner.read().unwrap();
        state.epoch.value
    }

    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Option<PyMatrix<'py>> {
        let state = self.inner.read().unwrap();
        state.covariance.as_ref().map(|cov| {
            cov.as_slice().to_pyarray(py)
                .reshape([cov.nrows(), cov.ncols()])
                .unwrap()
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:0.6}", self.inner.read().unwrap()))
    }
}

// Wrapper for Python-defined state transition - DYNAMIC only
#[pyclass(name="TransitionModel")]
pub struct PyTransitionModel {
    py_obj: Py<PyAny>,
    state: PyState, 
}

#[pymethods]
impl PyTransitionModel {

    #[new]
    pub fn new(py_obj: Py<PyAny>, state: PyState) -> Self {
        PyTransitionModel { py_obj, state }
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

///TODO Measurement model, Update Model, KalmanFilter
#[pymodule]
fn determined(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEpoch>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyTransitionModel>()?;
    Ok(())
}