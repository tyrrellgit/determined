//! Python bindings for the determined Kalman filter library
//! 
//! This module exposes the KalmanFilter to Python via PyO3.

#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::algorithms::DynKalmanFilter;
use crate::filter::Filter;
use crate::state::State;
use crate::common::Epoch;
use crate::common::na as na;

/// A Python wrapper around the Kalman Filter
#[pyclass]
pub struct PyKalmanFilter {
    /// The underlying filter (only supports 4,2 for now)
    kf: DynKalmanFilter,
}

#[pymethods]
impl PyKalmanFilter {
    /// Create a new Kalman filter for a system with N state dimensions and M measurement dimensions
    #[new]
    fn new(state_dim: usize, meas_dim: usize) -> Self {
        PyKalmanFilter {
            kf: DynKalmanFilter::new(state_dim, meas_dim),
        }
    }

    /// Predict the state forward in time
    fn predict(&mut self) -> PyResult<Vec<f64>> {
        let state = self.kf.predict(&Epoch { value: 0 });
        Ok(state.value.as_slice().to_vec())
    }

    /// Update the filter with a measurement
    fn update(&mut self, measurement: Vec<f64>) -> PyResult<Vec<f64>> {
        if measurement.len() != 2 {
            return Err(PyValueError::new_err("measurement must have 2 elements"));
        }

        let meas_matrix = na::OMatrix::<f64, na::Dyn, na::U1>::from_column_slice(&measurement);
        let obs = State{value: meas_matrix, epoch: 0};
        
        let state = self.kf.update(&obs);
        Ok(state.value.as_slice().to_vec())
    }

    /// Get the current state estimate
    fn get_state(&self) -> PyResult<Vec<f64>> {
        Ok(self.kf.state().value.as_slice().to_vec())
    }

    /// Get the current covariance trace (uncertainty)
    fn get_covariance_trace(&self) -> PyResult<f64> {
        Ok(self.kf.p.trace())
    }

    /// Set the state transition matrix F
    fn set_state_transition(&mut self, f: Vec<f64>) -> PyResult<()> {
        let dim = self.kf.x.dim();
        self.kf.f = na::DMatrix::from_row_slice(dim, dim, &f);
        Ok(())
    }

    /// Set the measurement matrix H
    fn set_measurement_matrix(&mut self, h: Vec<Vec<f64>>) -> PyResult<()> {
        let state_dim = self.kf.x.dim();
        let meas_dim = self.kf.h.nrows();

        // Flatten the nested Vec<Vec<f64>> into a flat Vec<f64>
        let flat: Vec<f64> = h.into_iter().flatten().collect();
        if flat.len() != meas_dim * state_dim {
            return Err(PyValueError::new_err("H must be a matrix of size (meas_dim, state_dim)"));
        }
        let matrix = na::DMatrix::from_row_slice(meas_dim, state_dim, &flat);
        self.kf.h = matrix;
        Ok(())
    }

    /// Set the process noise covariance Q
    fn set_process_noise(&mut self, q: Vec<Vec<f64>>) -> PyResult<()> {
        let dim = self.kf.x.dim();
        let matrix = na::DMatrix::from_row_slice(dim, dim, &q.concat());
        self.kf.q = matrix;
        Ok(())
    }

    /// Set the measurement noise covariance R
    fn set_measurement_noise(&mut self, r: Vec<Vec<f64>>) -> PyResult<()> {
        let dim = self.kf.h.nrows();
        let matrix = na::DMatrix::from_row_slice(dim, dim, &r.concat());
        self.kf.r = matrix;
        Ok(())
    }

    /// Set the initial state estimate
    fn set_state(&mut self, state: Vec<f64>) -> PyResult<()> {   
        let state_vec = na::OMatrix::<f64, na::Dyn, na::U1>::from_column_slice(&state);    
        self.kf.x = State{value: state_vec, epoch: 0};
        Ok(())
    }

    /// Set the initial covariance (uses diagonal matrix with given value)
    fn set_covariance(&mut self, initial_uncertainty: f64) -> PyResult<()> {
        let dim = self.kf.x.dim();
        self.kf.p = na::DMatrix::<f64>::identity(dim, dim) * initial_uncertainty;
        Ok(())
    }

    /// Reset the filter to initial state
    fn reset(&mut self) {
        self.kf.reset();
    }
}

/// Module initialization
#[pymodule]
    fn determined(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyKalmanFilter>()?;
        
        let submodule = PyModule::new_bound(_py, "kalman")?;
        submodule.add_class::<PyKalmanFilter>()?;
        m.add_submodule(&submodule)?;
        
        Ok(())
    }
