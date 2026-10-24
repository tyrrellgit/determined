use pyo3::prelude::*;
use numpy::{ PyArray1, PyArray2 };

pub type PyVector<'py> = Bound<'py, PyArray1<f64>>;
pub type PyMatrix<'py> = Bound<'py, PyArray2<f64>>;