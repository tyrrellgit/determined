#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use numpy::{ PyReadonlyArray1, PyReadonlyArray2 };
use numpy::ToPyArray;

use crate::common::na as na;
use crate::python::epoch::PyEpoch;
use crate::state::{State, StatePtr};
use crate::python::arrays::{ PyVector, PyMatrix, to_dmatrix, to_pymatrix, check_square };


#[pyclass(name="State")]
#[derive(Clone, Debug)]
pub struct PyState {
    pub inner: StatePtr<na::Dyn>,
}

#[pymethods]
impl PyState {
    
    #[new]
    #[pyo3(signature = (value, *, covariance, epoch))]
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
        // ndarray -> nalgebra (single copy)
        to_dmatrix(cov.as_array())
    });
    if let Some(cov) = &cov {
        check_square(cov, vector.len(), "covariance")?;
    }
        
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
        state.covariance.as_ref().map(|cov| to_pymatrix(py, cov))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:0.6}", self.inner.read().unwrap()))
    }
}

impl PyState {
    /// Copy `other` into this state's storage, keeping this state's identity,
    /// so everything sharing it (e.g. a filter) sees the new values.
    pub fn assign(&self, other: &PyState) -> PyResult<()> {
        // copy out first: `other` may share this state's lock
        let incoming = other.inner.read().unwrap().clone();
        let dim = self.inner.read().unwrap().dim();
        if incoming.dim() != dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "state has dimension {}, expected {dim}", incoming.dim())));
        }
        *self.inner.write().unwrap() = incoming;
        Ok(())
    }
}
