#![allow(unsafe_op_in_unsafe_fn)]

use crate::epoch::Epoch;
use pyo3::prelude::*;

#[pyclass(name = "Epoch")]
#[derive(Clone, Debug)]
pub struct PyEpoch {
    pub inner: Epoch,
}

#[pymethods]
impl PyEpoch {
    #[new]
    fn new(value: i64) -> PyEpoch {
        PyEpoch {
            inner: Epoch::new(value),
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
