#![allow(unsafe_op_in_unsafe_fn)]

pub mod epoch;
pub mod state;
pub mod transition;
pub mod measurement;
pub mod update;
pub mod algorithms;
pub mod arrays;

use pyo3::prelude::*;

///TODO: Update Model, KalmanFilter
#[pymodule]
fn determined(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Structs
    m.add_class::<epoch::PyEpoch>()?;
    m.add_class::<state::PyState>()?;
    m.add_class::<measurement::PyObservation>()?;

    // Models
    m.add_class::<transition::PyTransitionModel>()?;
    m.add_class::<measurement::PyMeasurementModel>()?;
    m.add_class::<update::PyUpdateModel>()?;
    
    // Algorithms
    m.add_class::<algorithms::PyKalmanFilter>()?;

    Ok(())
}