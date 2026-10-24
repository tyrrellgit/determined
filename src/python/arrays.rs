use pyo3::prelude::*;
use numpy::{ PyArray1, PyArray2, ToPyArray, PyArrayMethods };
use numpy::ndarray::ArrayView2;

use crate::common::na;

pub type PyVector<'py> = Bound<'py, PyArray1<f64>>;
pub type PyMatrix<'py> = Bound<'py, PyArray2<f64>>;

/// numpy (row-major view) -> nalgebra matrix, preserving the element layout.
///
/// nalgebra stores column-major, so the elements must be read row by row.
pub fn to_dmatrix(array: ArrayView2<'_, f64>) -> na::DMatrix<f64> {
    let (nrows, ncols) = array.dim();
    na::DMatrix::<f64>::from_row_iterator(nrows, ncols, array.iter().copied())
}

/// nalgebra matrix -> numpy (C-order) array of the same shape.
pub fn to_pymatrix<'py>(py: Python<'py>, matrix: &na::DMatrix<f64>) -> PyMatrix<'py> {
    // transpose so the column-major slice reads out in row-major order
    matrix.transpose()
        .as_slice()
        .to_pyarray(py)
        .reshape([matrix.nrows(), matrix.ncols()])
        .unwrap()
}
