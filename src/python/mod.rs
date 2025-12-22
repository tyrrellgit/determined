#![allow(unsafe_op_in_unsafe_fn)]

pub mod epoch;
pub mod state;
pub mod transition;
pub mod measurement;

use pyo3::prelude::*;

///TODO Measurement model, Update Model, KalmanFilter
#[pymodule]
fn determined(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Structs
    m.add_class::<epoch::PyEpoch>()?;
    m.add_class::<state::PyState>()?;
    m.add_class::<measurement::PyObservation>()?;

    // Models
    m.add_class::<transition::PyTransitionModel>()?;
    m.add_class::<measurement::PyMeasurementModel>()?;
    
    Ok(())
}