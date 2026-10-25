#![allow(unsafe_op_in_unsafe_fn)]

use numpy::PyReadonlyArray1;
use numpy::ToPyArray;
use pyo3::prelude::*;

use crate::common::na;
use crate::measurement::Observation;

use crate::python::arrays::PyVector;
use crate::python::epoch::PyEpoch;

#[pyclass(name = "Observation")]
#[derive(Clone, Debug)]
pub struct PyObservation {
    pub inner: Observation<na::Dyn>,
}

#[pymethods]
impl PyObservation {
    #[new]
    #[pyo3(signature = (value, *, epoch))]
    fn new(value: PyReadonlyArray1<f64>, epoch: &PyEpoch) -> PyResult<Self> {
        // numpy -> ndarray (zero-copy view) -> nalgebra
        let array = value.as_array();

        // TODO: handle ndarray with more than 1 dimension!
        let vector = na::DVector::<f64>::from_iterator(array.len(), array.iter().copied());

        Ok(PyObservation {
            inner: Observation {
                value: vector,
                epoch: epoch.inner,
            },
        })
    }

    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> PyVector<'py> {
        self.inner.value.as_slice().to_pyarray(py)
    }

    #[getter]
    fn epoch(&self) -> PyEpoch {
        PyEpoch {
            inner: self.inner.epoch,
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:0.6}", self.inner))
    }
}
