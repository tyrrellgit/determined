#![allow(unsafe_op_in_unsafe_fn)]

//! Python wrapper around the Rust linear Kalman filter (`KalmanFilterDyn`).
//!
//! The filter is fully computed in Rust from numpy matrices; no Python code
//! runs during `predict`/`update`. For nonlinear models see
//! `ExtendedKalmanFilter`, and for filters whose update you implement in
//! Python see `CustomFilter`.
//!
//! States returned by `predict`/`update`/`state` are live, zero-copy views of
//! the filter's internal state, not snapshots: they change on the next call.
//! Copy the arrays (`state.value.copy()`) if you need to keep one.

use pyo3::prelude::*;
use numpy::PyReadonlyArray2;

use crate::algorithms::KalmanFilterDyn;
use crate::filter::Filter;
use crate::models::{ LinearMeasurement, LinearTransition, LinearUpdate };

use crate::python::epoch::PyEpoch;
use crate::python::state::PyState;
use crate::python::measurement::PyObservation;
use crate::python::arrays::{ to_dmatrix, check_shape, check_square, check_length };

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
