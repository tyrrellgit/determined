#![allow(unsafe_op_in_unsafe_fn)]

//! Python-facing Extended Kalman Filter.
//!
//! Unlike the type-erased `KalmanFilter` wrapper (where the Python object
//! implements the whole update), this filter does the EKF maths in Rust and
//! only calls back into Python for the model functions. The callables take
//! and return plain numpy arrays (the state *value*), which suits automatic
//! differentiation libraries, e.g. `jac_f = jax.jacfwd(f)`.

use pyo3::prelude::*;
use pyo3::exceptions::{ PyTypeError, PyValueError };
use pyo3::types::{ PyAny, PyDict };
use numpy::{ PyReadonlyArray2, PyReadonlyArrayDyn, ToPyArray };

use crate::common::na;
use crate::epoch::Epoch;

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;
use crate::python::measurement::PyObservation;
use crate::python::arrays::to_dmatrix;

/// Coerce anything array-like (numpy, jax, lists...) to a float64 ndarray.
fn as_f64_array<'py>(py: Python<'py>, obj: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("float64")?)?;
    np.call_method("asarray", (obj,), Some(&kwargs))
}

fn check_callable(func: &Py<PyAny>, py: Python<'_>, name: &str) -> PyResult<()> {
    if func.bind(py).is_callable() {
        Ok(())
    } else {
        Err(PyTypeError::new_err(format!("{name} must be callable")))
    }
}

fn check_square(m: &na::DMatrix<f64>, dim: usize, name: &str) -> PyResult<()> {
    if m.shape() == (dim, dim) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must have shape ({dim}, {dim}), got {:?}", m.shape())))
    }
}

/// Call `func(x)` and read the result as a vector of `len` elements.
fn call_vector(
    py: Python<'_>,
    func: &Py<PyAny>,
    name: &str,
    x: &na::DVector<f64>,
    len: usize,
) -> PyResult<na::DVector<f64>> {
    let result = func.bind(py).call1((x.as_slice().to_pyarray(py),))?;
    let result = as_f64_array(py, result)?;
    let array: PyReadonlyArrayDyn<f64> = result.extract()?;
    let view = array.as_array();
    if view.len() != len {
        return Err(PyValueError::new_err(format!(
            "{name}() returned {} elements, expected {len}", view.len())));
    }
    Ok(na::DVector::<f64>::from_iterator(len, view.iter().copied()))
}

/// Call `func(x)` and read the result as an `nrows` x `ncols` matrix.
fn call_matrix(
    py: Python<'_>,
    func: &Py<PyAny>,
    name: &str,
    x: &na::DVector<f64>,
    nrows: usize,
    ncols: usize,
) -> PyResult<na::DMatrix<f64>> {
    let result = func.bind(py).call1((x.as_slice().to_pyarray(py),))?;
    let result = as_f64_array(py, result)?;
    let array: PyReadonlyArray2<f64> = result.extract().map_err(|_| {
        PyValueError::new_err(format!("{name}() must return a 2D array"))
    })?;
    if array.as_array().dim() != (nrows, ncols) {
        return Err(PyValueError::new_err(format!(
            "{name}() returned shape {:?}, expected ({nrows}, {ncols})", array.as_array().dim())));
    }
    Ok(to_dmatrix(array.as_array()))
}

#[pyclass(name="ExtendedKalmanFilter")]
pub struct PyExtendedKalmanFilter {
    state: PyState,
    f: Py<PyAny>,
    jac_f: Py<PyAny>,
    h: Py<PyAny>,
    jac_h: Py<PyAny>,
    q: na::DMatrix<f64>,
    r: na::DMatrix<f64>,
}

impl PyExtendedKalmanFilter {
    fn dim(&self) -> usize {
        self.state.inner.read().unwrap().dim()
    }

    /// Copy the state out so no lock is held while Python code runs.
    fn snapshot(&self) -> (na::DVector<f64>, na::DMatrix<f64>, Epoch) {
        let state = self.state.inner.read().unwrap();
        (state.value.clone(), state.covariance(), state.epoch)
    }

    fn write_back(&self, value: na::DVector<f64>, cov: na::DMatrix<f64>, epoch: Epoch) {
        let mut state = self.state.inner.write().unwrap();
        state.value = value;
        state.covariance = Some(cov);
        state.epoch = epoch;
    }

    /// Advance the state one epoch at a time, as the Rust `NonLinearTransition`.
    fn propagate(&self, py: Python<'_>, epoch: Epoch) -> PyResult<()> {
        let n = self.dim();
        let (mut x, mut p, mut current) = self.snapshot();

        let delta = (epoch - current).value;
        if delta == 0 {
            return Ok(());
        }
        if delta < 0 {
            spdlog::warn!(
                "Backward propagation is not supported for nonlinear transitions; state is unchanged."
            );
            return Ok(());
        }

        for _ in 0..delta {
            let jac = call_matrix(py, &self.jac_f, "jac_f", &x, n, n)?;
            let next = call_vector(py, &self.f, "f", &x, n)?;
            p = &jac * &p * jac.transpose() + &self.q;
            x = next;
            current = current + Epoch::new(1);
        }

        self.write_back(x, p, current);
        Ok(())
    }
}

#[pymethods]
impl PyExtendedKalmanFilter {
    /// ExtendedKalmanFilter(state, f, jac_f, q, h, jac_h, r)
    ///
    /// f(x) -> x'        one-step state propagation, shape (N,)
    /// jac_f(x) -> F     Jacobian of f, shape (N, N)
    /// h(x) -> z         measurement function, shape (M,)
    /// jac_h(x) -> H     Jacobian of h, shape (M, N)
    /// q, r              process (N, N) and measurement (M, M) noise
    #[new]
    #[pyo3(signature = (state, f, jac_f, q, h, jac_h, r))]
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
        check_square(&q, n, "q")?;
        if r.nrows() != r.ncols() {
            return Err(PyValueError::new_err(format!(
                "r must be square, got {:?}", r.shape())));
        }

        Ok(PyExtendedKalmanFilter { state, f, jac_f, h, jac_h, q, r })
    }

    fn predict(&self, py: Python<'_>, epoch: &PyEpoch) -> PyResult<PyState> {
        self.propagate(py, epoch.inner)?;
        Ok(PyState { inner: self.state.inner.clone() })
    }

    fn update(&self, py: Python<'_>, observation: &PyObservation) -> PyResult<PyState> {
        let n = self.dim();
        let m = self.r.nrows();

        let z = &observation.inner.value;
        if z.len() != m {
            return Err(PyValueError::new_err(format!(
                "observation has {} elements, expected {m}", z.len())));
        }

        // propagate state to observation epoch
        self.propagate(py, observation.inner.epoch)?;
        let (x, p, epoch) = self.snapshot();

        // project the state to the measurement domain, linearised about it
        let z_x = call_vector(py, &self.h, "h", &x, m)?;
        let h = call_matrix(py, &self.jac_h, "jac_h", &x, m, n)?;

        // innovation covariance and gain
        let h_t = h.transpose();
        let s = &h * &p * &h_t + &self.r;
        let s_inv = match s.try_inverse() {
            Some(s_inv) => s_inv,
            None => {
                spdlog::error!(
                    "Invalid update: Innovation covariance S is singular; state will not be updated."
                );
                return Ok(PyState { inner: self.state.inner.clone() });
            }
        };
        let k = &p * &h_t * s_inv;

        let value = &x + &k * (z - z_x);
        let cov = (na::DMatrix::<f64>::identity(n, n) - &k * &h) * &p;

        self.write_back(value, cov, epoch);
        Ok(PyState { inner: self.state.inner.clone() })
    }

    #[getter]
    fn get_state(&self) -> PyState {
        PyState { inner: self.state.inner.clone() }
    }

    #[setter]
    fn set_state(&self, state: PyState) -> PyResult<()> {
        let incoming = state.inner.read().unwrap();
        if incoming.dim() != self.dim() {
            return Err(PyValueError::new_err(format!(
                "state has dimension {}, expected {}", incoming.dim(), self.dim())));
        }
        let mut inner = self.state.inner.write().unwrap();
        inner.value = incoming.value.clone();
        inner.covariance = incoming.covariance.clone();
        inner.epoch = incoming.epoch;
        Ok(())
    }

    /// Zero the state and reset the covariance to identity.
    fn reset(&self) {
        let mut state = self.state.inner.write().unwrap();
        let n = state.dim();
        state.value.fill(0.0);
        state.covariance = Some(na::DMatrix::<f64>::identity(n, n));
    }
}
