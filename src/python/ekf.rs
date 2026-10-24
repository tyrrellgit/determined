#![allow(unsafe_op_in_unsafe_fn)]

//! Python wrapper around the Rust Extended Kalman Filter (`ExtendedKalmanFilterDyn`).
//!
//! Unlike the type-erased `CustomFilter` wrapper (where the Python object
//! implements the whole update), this filter does the EKF maths in Rust and
//! only calls back into Python for the model functions. The callables take
//! the state *value* as a numpy array and the epoch `t` as an int, and return
//! plain numpy arrays, which suits automatic differentiation libraries, e.g.
//! `jac_f = jax.jacfwd(f)` (differentiates the first argument, `x`).
//!
//! States returned by `predict`/`update`/`state` are live, zero-copy views of
//! the filter's internal state, not snapshots: they change on the next call.
//! Copy the arrays (`state.value.copy()`) if you need to keep one.

use std::sync::{ Arc, Mutex };

use pyo3::prelude::*;
use pyo3::exceptions::{ PyTypeError, PyValueError };
use pyo3::types::{ PyAny, PyDict };
use numpy::{ PyReadonlyArray2, PyReadonlyArrayDyn, ToPyArray };

use crate::common::na;
use crate::algorithms::ExtendedKalmanFilterDyn;
use crate::filter::Filter;
use crate::models::{ NonLinearMeasurement, NonLinearTransition, NonLinearUpdate };
use crate::state::State;

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;
use crate::python::measurement::PyObservation;
use crate::python::arrays::{ to_dmatrix, check_square, check_length };

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

#[pyclass(name="ExtendedKalmanFilter")]
pub struct PyExtendedKalmanFilter {
    filter: ExtendedKalmanFilterDyn,
    state: PyState,
    error: ErrorSlot,
}

impl PyExtendedKalmanFilter {
    /// Run a filter operation; if a Python model function raised, restore
    /// the state to what it was before and re-raise that error.
    fn guarded<T>(&mut self, op: impl FnOnce(&mut ExtendedKalmanFilterDyn) -> T) -> PyResult<T> {
        let backup = self.filter.state.read().unwrap().clone();
        *self.error.lock().unwrap() = None;

        let result = op(&mut self.filter);

        match self.error.lock().unwrap().take() {
            Some(err) => {
                *self.filter.state.write().unwrap() = backup;
                Err(err)
            }
            None => Ok(result),
        }
    }
}

#[pymethods]
impl PyExtendedKalmanFilter {
    /// ExtendedKalmanFilter(state, *, f, jac_f, q, h, jac_h, r)
    ///
    /// f(x, t) -> x'     one-step propagation from epoch t to t + 1, shape (N,)
    /// jac_f(x, t) -> F  Jacobian of f, shape (N, N)
    /// h(x, t) -> z      measurement function at epoch t, shape (M,)
    /// jac_h(x, t) -> H  Jacobian of h, shape (M, N)
    /// q, r              process (N, N) and measurement (M, M) noise
    ///
    /// `t` is the integer epoch value. All arguments after `state` are
    /// keyword-only. The returned states are live views of the filter state.
    #[new]
    #[pyo3(signature = (state, *, f, jac_f, q, h, jac_h, r))]
    fn new(
        py: Python<'_>,
        state: PyState,
        f: Py<PyAny>,
        jac_f: Py<PyAny>,
        q: PyReadonlyArray2<f64>,
        h: Py<PyAny>,
        jac_h: Py<PyAny>,
        r: PyReadonlyArray2<f64>,
    ) -> PyResult<Self> {
        for (func, name) in [(&f, "f"), (&jac_f, "jac_f"), (&h, "h"), (&jac_h, "jac_h")] {
            check_callable(func, py, name)?;
        }

        let q = to_dmatrix(q.as_array());
        let r = to_dmatrix(r.as_array());
        let n = state.inner.read().unwrap().dim();
        let m = r.nrows();
        check_square(&q, n, "q")?;
        check_square(&r, m, "r")?;

        let error = ErrorSlot::default();
        let transition = NonLinearTransition::new(
            state.inner.clone(),
            Box::new(vector_model(f, "f", n, error.clone())),
            Box::new(matrix_model(jac_f, "jac_f", n, n, error.clone())),
            q,
        );
        let measurement = NonLinearMeasurement::new(
            Box::new(vector_model(h, "h", m, error.clone())),
            Box::new(matrix_model(jac_h, "jac_h", m, n, error.clone())),
            r,
        );
        let filter = ExtendedKalmanFilterDyn::new(NonLinearUpdate::new(measurement, transition));

        Ok(PyExtendedKalmanFilter { filter, state, error })
    }

    fn predict(&mut self, epoch: &PyEpoch) -> PyResult<PyState> {
        let inner = self.guarded(|filter| filter.predict(&epoch.inner))?;
        Ok(PyState { inner })
    }

    fn update(&mut self, observation: &PyObservation) -> PyResult<PyState> {
        check_length(&observation.inner.value, self.filter.update.measurement.r.nrows(), "observation")?;
        let inner = self.guarded(|filter| filter.update(&observation.inner))?;
        Ok(PyState { inner })
    }

    #[getter]
    fn get_state(&self) -> PyState {
        PyState { inner: self.filter.state() }
    }

    #[setter]
    fn set_state(&self, state: PyState) -> PyResult<()> {
        self.state.assign(&state)
    }

    /// Zero the state and reset the covariance to identity.
    fn reset(&mut self) {
        self.filter.reset();
    }
}
