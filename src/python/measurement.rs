#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::{ PyArray1, PyArray2, PyReadonlyArray1 };
use numpy::{ ToPyArray ,PyArrayMethods };

use crate::common::na as na;
use crate::measurement::Observation;
use crate::models::MeasurementModel;
use crate::state::State;

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;

/// Array types bound to python
type PyVector<'py> = Bound<'py, PyArray1<f64>>;
type PyMatrix<'py> = Bound<'py, PyArray2<f64>>;

#[pyclass(name="Observation")]
#[derive(Clone, Debug)]
pub struct PyObservation {
    pub inner: Observation<na::Dyn>,
}

#[pymethods]
impl PyObservation {
    #[new]
    fn new(
        value: PyReadonlyArray1<f64>,
        epoch: &PyEpoch
    ) -> PyResult<Self> {
        // numpy -> ndarray (zero-copy view) -> nalgebra
        let array = value.as_array();
        
        // TODO: handle ndarray with more than 1 dimension!
        let vector = na::DVector::<f64>::from_iterator(
            array.len(),
            array.iter().copied()
        );
            
        Ok(PyObservation {
            inner: Observation {
                value: vector,
                epoch: epoch.inner,
            }
        })
    }

    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> PyVector<'py> {
        self.inner.value.as_slice()
            .to_pyarray(py)
    }

    #[getter]
    fn epoch(&self) -> PyEpoch {
        PyEpoch { inner: self.inner.epoch }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:0.6}", self.inner))
    }
}

// Wrapper for Python-defined Measurement models
#[pyclass(name="MeasurementModel")]
pub struct PyMeasurementModel {
    py_obj: Py<PyAny>,
}

#[pymethods]
impl PyMeasurementModel {
    
    #[new]
    pub fn new(model: Py<PyAny>) -> Self {
        PyMeasurementModel { py_obj: model }
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
        jac.as_slice()
            .to_pyarray(py)
            .reshape([jac.nrows(), jac.ncols()])
            .unwrap()
    }
}

impl MeasurementModel<na::Dyn, na::Dyn> for PyMeasurementModel {
    fn projection(&self, state: &State<na::Dyn>) -> Observation<na::Dyn> {
        let py_state = PyState { inner: state.ptr() };

        Python::attach(|py| {
            let py_obj = self.py_obj.bind(py);
            
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

    fn inverse(&self, obs: &Observation<na::Dyn>) -> State<na::Dyn> {

        let py_obs = PyObservation {
                inner: Observation {
                    value: obs.value.clone(),
                    epoch: obs.epoch,
                }
            };

        Python::attach(|py| {
            let py_obj = self.py_obj.bind(py);

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
                .expect("jacobian() must return <ndarray>");

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
