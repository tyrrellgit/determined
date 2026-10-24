#![allow(unsafe_op_in_unsafe_fn)]

//! Python bindings. The modules mirror the crate:
//!
//! - `epoch`, `state`, `measurement`: `Epoch`, `State`, `Observation`
//! - `models`: Python-implemented `TransitionModel`, `MeasurementModel`, `UpdateModel`
//! - `algorithms`: `KalmanFilter`, `ExtendedKalmanFilter`, `CustomFilter`
//!
//! plus binding helpers: `arrays` (numpy <-> nalgebra), `callbacks` (Python
//! functions as Rust model closures) and `logging` (spdlog -> Python logging).

pub mod epoch;
pub mod state;
pub mod measurement;
pub mod models;
pub mod algorithms;

pub mod arrays;
pub mod callbacks;
pub mod logging;

use pyo3::prelude::*;

#[pymodule]
fn determined(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Route Rust log messages to Python's `logging`
    logging::install()?;

    // Structs
    m.add_class::<epoch::PyEpoch>()?;
    m.add_class::<state::PyState>()?;
    m.add_class::<measurement::PyObservation>()?;

    // Models
    m.add_class::<models::PyTransitionModel>()?;
    m.add_class::<models::PyMeasurementModel>()?;
    m.add_class::<models::PyUpdateModel>()?;

    // Algorithms
    m.add_class::<algorithms::PyKalmanFilter>()?;
    m.add_class::<algorithms::PyExtendedKalmanFilter>()?;
    m.add_class::<algorithms::PyCustomFilter>()?;

    Ok(())
}
