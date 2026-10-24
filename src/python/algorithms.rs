#![allow(unsafe_op_in_unsafe_fn)]

//! Python wrappers around the Rust filters in `crate::algorithms`:
//!
//! | Rust                                     | Python                 |
//! |------------------------------------------|------------------------|
//! | `KalmanFilterDyn`                        | `KalmanFilter`         |
//! | `ExtendedKalmanFilterDyn`                | `ExtendedKalmanFilter` |
//! | `UnscentedKalmanFilterDyn`               | `UnscentedKalmanFilter`|
//! | `UDKalmanFilterDyn`                      | `UDKalmanFilter`       |
//! | `UDExtendedKalmanFilterDyn`              | `UDExtendedKalmanFilter` |
//! | `KalmanFilterT<Dyn, Dyn, PyUpdateModel>` | `CustomFilter`         |
//!
//! `KalmanFilter` and `UDKalmanFilter` are computed entirely in Rust from numpy
//! matrices. The extended, UD extended and unscented filters do the maths in Rust and
//! call back into Python only for the model functions (see `callbacks`). `CustomFilter` runs
//! whatever update your Python `UpdateModel` implements.
//!
//! States returned by `predict`/`update`/`state` are live, zero-copy views of
//! the filter's internal state, not snapshots: they change on the next call.
//! Copy the arrays (`state.value.copy()`) if you need to keep one.

use pyo3::prelude::*;
use numpy::PyReadonlyArray2;

use crate::common::na;
use crate::algorithms::{
    KalmanFilterT, KalmanFilterDyn, ExtendedKalmanFilterDyn, UnscentedKalmanFilterDyn,
    UDKalmanFilterDyn, UDExtendedKalmanFilterDyn,
};
use crate::filter::Filter;
use crate::models::{
    LinearMeasurement, LinearTransition, LinearUpdate,
    NonLinearMeasurement, NonLinearTransition, NonLinearUpdate,
    UnscentedMeasurement, UnscentedTransition, UnscentedUpdate, UnscentedParams,
    LinearUDUpdate, NonLinearUDUpdate,
};

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;
use crate::python::measurement::PyObservation;
use crate::python::models::PyUpdateModel;
use crate::python::arrays::{ to_dmatrix, check_shape, check_square, check_length };
use crate::python::callbacks::{ ErrorSlot, check_callable, guarded, vector_model, matrix_model };


// Kalman Filter
#[pyclass(name="KalmanFilter")]
pub struct PyKalmanFilter {
    filter: KalmanFilterDyn,
    state: PyState,
}

#[pymethods]
impl PyKalmanFilter {
    /// KalmanFilter(state, *, f, q, h, r)
    ///
    /// f  state transition, shape (N, N)
    /// q  process noise covariance, shape (N, N)
    /// h  measurement matrix, shape (M, N)
    /// r  measurement noise covariance, shape (M, M)
    ///
    /// All arguments after `state` are keyword-only. The returned states are
    /// live views of the filter state.
    #[new]
    #[pyo3(signature = (state, *, f, q, h, r))]
    fn new(
        state: PyState,
        f: PyReadonlyArray2<f64>,
        q: PyReadonlyArray2<f64>,
        h: PyReadonlyArray2<f64>,
        r: PyReadonlyArray2<f64>,
    ) -> PyResult<Self> {
        let (f, q) = (to_dmatrix(f.as_array()), to_dmatrix(q.as_array()));
        let (h, r) = (to_dmatrix(h.as_array()), to_dmatrix(r.as_array()));

        let n = state.inner.read().unwrap().dim();
        let m = h.nrows();
        check_square(&f, n, "f")?;
        check_square(&q, n, "q")?;
        check_shape(&h, (m, n), "h")?;
        check_square(&r, m, "r")?;

        let filter = KalmanFilterDyn::new(LinearUpdate::new(
            LinearMeasurement::new(h, r),
            LinearTransition::new(state.inner.clone(), f, q),
        ));
        Ok(PyKalmanFilter { filter, state })
    }

    fn predict(&mut self, epoch: &PyEpoch) -> PyState {
        PyState { inner: self.filter.predict(&epoch.inner) }
    }

    fn update(&mut self, observation: &PyObservation) -> PyResult<PyState> {
        check_length(&observation.inner.value, self.filter.update.measurement.h.nrows(), "observation")?;
        Ok(PyState { inner: self.filter.update(&observation.inner) })
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


// Extended Kalman Filter
#[pyclass(name="ExtendedKalmanFilter")]
pub struct PyExtendedKalmanFilter {
    filter: ExtendedKalmanFilterDyn,
    state: PyState,
    error: ErrorSlot,
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
        let inner = guarded(&mut self.filter, &self.state.inner, &self.error,
            |filter| filter.predict(&epoch.inner))?;
        Ok(PyState { inner })
    }

    fn update(&mut self, observation: &PyObservation) -> PyResult<PyState> {
        check_length(&observation.inner.value, self.filter.update.measurement.r.nrows(), "observation")?;
        let inner = guarded(&mut self.filter, &self.state.inner, &self.error,
            |filter| filter.update(&observation.inner))?;
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


// Unscented Kalman Filter
#[pyclass(name="UnscentedKalmanFilter")]
pub struct PyUnscentedKalmanFilter {
    filter: UnscentedKalmanFilterDyn,
    state: PyState,
    error: ErrorSlot,
}

#[pymethods]
impl PyUnscentedKalmanFilter {
    /// UnscentedKalmanFilter(state, *, f, q, h, r, alpha=1.0, beta=2.0, kappa=0.0)
    ///
    /// f(x, t) -> x'     one-step propagation from epoch t to t + 1, shape (N,)
    /// h(x, t) -> z      measurement function at epoch t, shape (M,)
    /// q, r              process (N, N) and measurement (M, M) noise
    /// alpha, beta, kappa  scaled sigma point parameters (van der Merwe)
    ///
    /// No Jacobians are needed: `f` and `h` are evaluated at 2N + 1 sigma
    /// points per step, so they must not keep state between calls. The
    /// default parameters keep all sigma point weights non-negative.
    ///
    /// `t` is the integer epoch value. All arguments after `state` are
    /// keyword-only. The returned states are live views of the filter state.
    #[new]
    #[pyo3(signature = (state, *, f, q, h, r, alpha=1.0, beta=2.0, kappa=0.0))]
    fn new(
        py: Python<'_>,
        state: PyState,
        f: Py<PyAny>,
        q: PyReadonlyArray2<f64>,
        h: Py<PyAny>,
        r: PyReadonlyArray2<f64>,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> PyResult<Self> {
        for (func, name) in [(&f, "f"), (&h, "h")] {
            check_callable(func, py, name)?;
        }

        let q = to_dmatrix(q.as_array());
        let r = to_dmatrix(r.as_array());
        let n = state.inner.read().unwrap().dim();
        let m = r.nrows();
        check_square(&q, n, "q")?;
        check_square(&r, m, "r")?;

        let error = ErrorSlot::default();
        let transition = UnscentedTransition::new(
            state.inner.clone(),
            Box::new(vector_model(f, "f", n, error.clone())),
            q,
        ).with_params(UnscentedParams { alpha, beta, kappa });
        let measurement = UnscentedMeasurement::new(
            Box::new(vector_model(h, "h", m, error.clone())),
            r,
        );
        let filter = UnscentedKalmanFilterDyn::new(UnscentedUpdate::new(measurement, transition));

        Ok(PyUnscentedKalmanFilter { filter, state, error })
    }

    fn predict(&mut self, epoch: &PyEpoch) -> PyResult<PyState> {
        let inner = guarded(&mut self.filter, &self.state.inner, &self.error,
            |filter| filter.predict(&epoch.inner))?;
        Ok(PyState { inner })
    }

    fn update(&mut self, observation: &PyObservation) -> PyResult<PyState> {
        check_length(&observation.inner.value, self.filter.update.measurement.r.nrows(), "observation")?;
        let inner = guarded(&mut self.filter, &self.state.inner, &self.error,
            |filter| filter.update(&observation.inner))?;
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


// UD Kalman Filter
#[pyclass(name="UDKalmanFilter")]
pub struct PyUDKalmanFilter {
    filter: UDKalmanFilterDyn,
    state: PyState,
}

#[pymethods]
impl PyUDKalmanFilter {
    /// UDKalmanFilter(state, *, f, q, h, r)
    ///
    /// The Kalman filter in UD factorised form: the covariance is held as
    /// P = U D U^T and the factors are propagated and updated directly, which
    /// keeps P symmetric positive semi-definite. Same models as KalmanFilter:
    ///
    /// f  state transition, shape (N, N)
    /// q  process noise covariance, shape (N, N)
    /// h  measurement matrix, shape (M, N)
    /// r  measurement noise covariance, shape (M, M); decorrelated if not diagonal
    ///
    /// All arguments after `state` are keyword-only. The returned states are
    /// live views of the filter state, carrying P = U D U^T. Predicting
    /// backwards is not supported.
    #[new]
    #[pyo3(signature = (state, *, f, q, h, r))]
    fn new(
        state: PyState,
        f: PyReadonlyArray2<f64>,
        q: PyReadonlyArray2<f64>,
        h: PyReadonlyArray2<f64>,
        r: PyReadonlyArray2<f64>,
    ) -> PyResult<Self> {
        let (f, q) = (to_dmatrix(f.as_array()), to_dmatrix(q.as_array()));
        let (h, r) = (to_dmatrix(h.as_array()), to_dmatrix(r.as_array()));

        let n = state.inner.read().unwrap().dim();
        let m = h.nrows();
        check_square(&f, n, "f")?;
        check_square(&q, n, "q")?;
        check_shape(&h, (m, n), "h")?;
        check_square(&r, m, "r")?;

        let filter = UDKalmanFilterDyn::new(LinearUDUpdate::new(
            LinearMeasurement::new(h, r),
            LinearTransition::new(state.inner.clone(), f, q),
        ));
        Ok(PyUDKalmanFilter { filter, state })
    }

    fn predict(&mut self, epoch: &PyEpoch) -> PyState {
        PyState { inner: self.filter.predict(&epoch.inner) }
    }

    fn update(&mut self, observation: &PyObservation) -> PyResult<PyState> {
        check_length(&observation.inner.value, self.filter.update.measurement.h.nrows(), "observation")?;
        Ok(PyState { inner: self.filter.update(&observation.inner) })
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


// UD Extended Kalman Filter
#[pyclass(name="UDExtendedKalmanFilter")]
pub struct PyUDExtendedKalmanFilter {
    filter: UDExtendedKalmanFilterDyn,
    state: PyState,
    error: ErrorSlot,
}

#[pymethods]
impl PyUDExtendedKalmanFilter {
    /// UDExtendedKalmanFilter(state, *, f, jac_f, q, h, jac_h, r)
    ///
    /// The Extended Kalman filter in UD factorised form (P = U D U^T, factors
    /// propagated and updated directly). Same models as ExtendedKalmanFilter:
    ///
    /// f(x, t) -> x'     one-step propagation from epoch t to t + 1, shape (N,)
    /// jac_f(x, t) -> F  Jacobian of f, shape (N, N)
    /// h(x, t) -> z      measurement function at epoch t, shape (M,)
    /// jac_h(x, t) -> H  Jacobian of h, shape (M, N)
    /// q, r              process (N, N) and measurement (M, M) noise; r is
    ///                   decorrelated if not diagonal
    ///
    /// `t` is the integer epoch value. All arguments after `state` are
    /// keyword-only. The returned states are live views of the filter state,
    /// carrying P = U D U^T.
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
        let filter = UDExtendedKalmanFilterDyn::new(NonLinearUDUpdate::new(measurement, transition));

        Ok(PyUDExtendedKalmanFilter { filter, state, error })
    }

    fn predict(&mut self, epoch: &PyEpoch) -> PyResult<PyState> {
        let inner = guarded(&mut self.filter, &self.state.inner, &self.error,
            |filter| filter.predict(&epoch.inner))?;
        Ok(PyState { inner })
    }

    fn update(&mut self, observation: &PyObservation) -> PyResult<PyState> {
        check_length(&observation.inner.value, self.filter.update.measurement.r.nrows(), "observation")?;
        let inner = guarded(&mut self.filter, &self.state.inner, &self.error,
            |filter| filter.update(&observation.inner))?;
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


// Custom Filter
/// Filter built from a Python `UpdateModel`: the Python equivalent of a Rust
/// `KalmanFilterT` with your own update model. The update maths is whatever
/// your model implements. See `KalmanFilter` for the Rust-computed filter.
#[pyclass(name="CustomFilter")]
pub struct PyCustomFilter {
    filter: KalmanFilterT<na::Dyn, na::Dyn, PyUpdateModel>,
    state: PyState,
}

#[pymethods]
impl PyCustomFilter {
    #[new]
    pub fn new(update: PyRef<'_, PyUpdateModel>) -> Self {
        let update = update.clone();
        let state = update.state.clone();
        PyCustomFilter {
            filter: KalmanFilterT::new(update),
            state,
        }
    }

    fn predict(&mut self, epoch: &PyEpoch) -> PyState {
        PyState { inner: self.filter.predict(&epoch.inner) }
    }

    fn update(&mut self, observation: &PyObservation) -> PyState {
        PyState { inner: self.filter.update(&observation.inner) }
    }

    #[getter]
    fn get_state(&self) -> PyState {
        PyState { inner: self.filter.state() }
    }

    #[setter]
    fn set_state(&mut self, state: PyState) -> PyResult<()> {
        self.state.assign(&state)
    }

    /// Calls the update model's `reset()`, if it has one. The state belongs to
    /// your models, so the filter does not reset it itself.
    fn reset(&mut self) {
        Python::attach(|py| {
            if self.filter.update.model.bind(py).call_method0("reset").is_err() {
                spdlog::warn!("No update.reset() method found.");
            }
        });
    }
}
