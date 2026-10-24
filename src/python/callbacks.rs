//! Python model functions as Rust model closures.
//!
//! Used by `ExtendedKalmanFilter` to pass Python callables `f(x, t)` /
//! `h(x, t)` (and their Jacobians) to the Rust `NonLinear*` models. The
//! callables take the state value as a numpy array and the epoch as an int,
//! and may return anything array-like (so autodiff libraries such as JAX work).

use std::sync::{ Arc, Mutex };

use pyo3::prelude::*;
use pyo3::exceptions::{ PyTypeError, PyValueError };
use pyo3::types::{ PyAny, PyDict };
use numpy::{ PyReadonlyArray2, PyReadonlyArrayDyn, ToPyArray };

use crate::common::na;
use crate::state::State;

use crate::python::arrays::to_dmatrix;

/// First error raised by a Python model function during a filter call.
///
/// The Rust model closures cannot fail, so on error they record it here and
/// return a placeholder; the wrapper then restores the state and raises.
pub type ErrorSlot = Arc<Mutex<Option<PyErr>>>;

/// Coerce anything array-like (numpy, jax, lists...) to a float64 ndarray.
fn as_f64_array<'py>(py: Python<'py>, obj: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("float64")?)?;
    np.call_method("asarray", (obj,), Some(&kwargs))
}

pub fn check_callable(func: &Py<PyAny>, py: Python<'_>, name: &str) -> PyResult<()> {
    if func.bind(py).is_callable() {
        Ok(())
    } else {
        Err(PyTypeError::new_err(format!("{name} must be callable")))
    }
}

/// Call `func(x, t)` and read the result as a vector of `len` elements.
fn call_vector(
    py: Python<'_>,
    func: &Py<PyAny>,
    name: &str,
    state: &State<na::Dyn>,
    len: usize,
) -> PyResult<na::DVector<f64>> {
    let x = state.value.as_slice().to_pyarray(py);
    let result = as_f64_array(py, func.bind(py).call1((x, state.epoch.value))?)?;
    let array: PyReadonlyArrayDyn<f64> = result.extract()?;
    let view = array.as_array();
    if view.len() != len {
        return Err(PyValueError::new_err(format!(
            "{name}() returned {} elements, expected {len}", view.len())));
    }
    Ok(na::DVector::<f64>::from_iterator(len, view.iter().copied()))
}

/// Call `func(x, t)` and read the result as an `nrows` x `ncols` matrix.
fn call_matrix(
    py: Python<'_>,
    func: &Py<PyAny>,
    name: &str,
    state: &State<na::Dyn>,
    nrows: usize,
    ncols: usize,
) -> PyResult<na::DMatrix<f64>> {
    let x = state.value.as_slice().to_pyarray(py);
    let result = as_f64_array(py, func.bind(py).call1((x, state.epoch.value))?)?;
    let array: PyReadonlyArray2<f64> = result.extract().map_err(|_| {
        PyValueError::new_err(format!("{name}() must return a 2D array"))
    })?;
    if array.as_array().dim() != (nrows, ncols) {
        return Err(PyValueError::new_err(format!(
            "{name}() returned shape {:?}, expected ({nrows}, {ncols})", array.as_array().dim())));
    }
    Ok(to_dmatrix(array.as_array()))
}

/// Wrap a Python model function as a Rust model closure returning a vector.
pub fn vector_model(
    func: Py<PyAny>,
    name: &'static str,
    len: usize,
    error: ErrorSlot,
) -> impl Fn(&State<na::Dyn>) -> na::DVector<f64> + Send + Sync {
    move |state| {
        let placeholder = || na::DVector::<f64>::zeros(len);
        if error.lock().unwrap().is_some() {
            return placeholder();
        }
        Python::attach(|py| call_vector(py, &func, name, state, len))
            .unwrap_or_else(|err| {
                *error.lock().unwrap() = Some(err);
                placeholder()
            })
    }
}

/// Wrap a Python model function as a Rust model closure returning a matrix.
pub fn matrix_model(
    func: Py<PyAny>,
    name: &'static str,
    nrows: usize,
    ncols: usize,
    error: ErrorSlot,
) -> impl Fn(&State<na::Dyn>) -> na::DMatrix<f64> + Send + Sync {
    move |state| {
        let placeholder = || na::DMatrix::<f64>::zeros(nrows, ncols);
        if error.lock().unwrap().is_some() {
            return placeholder();
        }
        Python::attach(|py| call_matrix(py, &func, name, state, nrows, ncols))
            .unwrap_or_else(|err| {
                *error.lock().unwrap() = Some(err);
                placeholder()
            })
    }
}
