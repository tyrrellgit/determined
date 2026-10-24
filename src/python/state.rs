#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use numpy::{ PyReadonlyArray1, PyReadonlyArray2 };
use numpy::{ ToPyArray, PyArrayMethods };

use crate::common::na as na;
use crate::python::epoch::PyEpoch;
use crate::state::{State, StatePtr};
use crate::python::arrays::{ PyVector, PyMatrix };


#[pyclass(name="State")]
#[derive(Clone, Debug)]
pub struct PyState {
    pub inner: StatePtr<na::Dyn>,
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
    fn epoch(&self) -> PyEpoch {
        let state = self.inner.read().unwrap();
        PyEpoch { inner: state.epoch }
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