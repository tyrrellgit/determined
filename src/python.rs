//! Python bindings for the determined Kalman filter library
//! 
//! This module exposes the KalmanFilter to Python via PyO3.

#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::algorithms::KalmanFilter;
use crate::filter::Filter;
use crate::state::State;
use crate::common::Epoch;
use crate::common::na as na;

/// A Python wrapper around the Kalman Filter
#[pyclass]
pub struct PyKalmanFilter {
    /// The underlying filter (only supports 4,2 for now)
    kf: KalmanFilter<4, 2>,
}

#[pymethods]
impl PyKalmanFilter {
    /// Create a new Kalman filter for a system with N state dimensions and M measurement dimensions
    #[new]
    fn new() -> Self {
        PyKalmanFilter {
            kf: KalmanFilter::new(),
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
        
        let meas_matrix = na::SMatrix::<f64, 2, 1>::from_column_slice(&measurement);
        let obs = State::from_matrix(meas_matrix, 0);
        
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

    /// Set the state transition matrix F (4x4)
    fn set_state_transition(&mut self, f: Vec<Vec<f64>>) -> PyResult<()> {
        if f.len() != 4 || f.iter().any(|row| row.len() != 4) {
            return Err(PyValueError::new_err("F must be a 4x4 matrix"));
        }
        
        let mut matrix = na::SMatrix::<f64, 4, 4>::zeros();
        for i in 0..4 {
            for j in 0..4 {
                matrix[(i, j)] = f[i][j];
            }
        }
        self.kf.f = matrix;
        Ok(())
    }

    /// Set the measurement matrix H (2x4)
    fn set_measurement_matrix(&mut self, h: Vec<Vec<f64>>) -> PyResult<()> {
        if h.len() != 2 || h.iter().any(|row| row.len() != 4) {
            return Err(PyValueError::new_err("H must be a 2x4 matrix"));
        }
        
        let mut matrix = na::SMatrix::<f64, 2, 4>::zeros();
        for i in 0..2 {
            for j in 0..4 {
                matrix[(i, j)] = h[i][j];
            }
        }
        self.kf.h = matrix;
        Ok(())
    }

    /// Set the process noise covariance Q (4x4)
    fn set_process_noise(&mut self, q: Vec<Vec<f64>>) -> PyResult<()> {
        if q.len() != 4 || q.iter().any(|row| row.len() != 4) {
            return Err(PyValueError::new_err("Q must be a 4x4 matrix"));
        }
        
        let mut matrix = na::SMatrix::<f64, 4, 4>::zeros();
        for i in 0..4 {
            for j in 0..4 {
                matrix[(i, j)] = q[i][j];
            }
        }
        self.kf.q = matrix;
        Ok(())
    }

    /// Set the measurement noise covariance R (2x2)
    fn set_measurement_noise(&mut self, r: Vec<Vec<f64>>) -> PyResult<()> {
        if r.len() != 2 || r.iter().any(|row| row.len() != 2) {
            return Err(PyValueError::new_err("R must be a 2x2 matrix"));
        }
        
        let mut matrix = na::SMatrix::<f64, 2, 2>::zeros();
        for i in 0..2 {
            for j in 0..2 {
                matrix[(i, j)] = r[i][j];
            }
        }
        self.kf.r = matrix;
        Ok(())
    }

    /// Set the initial state estimate
    fn set_state(&mut self, state: Vec<f64>) -> PyResult<()> {
        if state.len() != 4 {
            return Err(PyValueError::new_err("state must have 4 elements"));
        }
        
        let state_vec = na::SVector::<f64, 4>::from_row_slice(&state);
        self.kf.x = State::from_matrix(state_vec.into(), 0);
        Ok(())
    }

    /// Set the initial covariance (uses diagonal matrix with given value)
    fn set_covariance(&mut self, initial_uncertainty: f64) -> PyResult<()> {
        self.kf.p = na::SMatrix::<f64, 4, 4>::identity() * initial_uncertainty;
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
