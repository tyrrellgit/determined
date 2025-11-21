//! Small typed wrapper around nalgebra static matrices representing a state or
//! observation. `State<R,C>` is a thin container that carries a `SMatrix` and
//! a simple `epoch` value used for timestamping.

use crate::common::na as na;
use crate::common::Composable;


#[derive(Clone, Debug)]
pub struct State<const R: usize, const C: usize> {
    pub value: na::SMatrix<f64, R, C>,
    pub epoch: u64,
}

impl<const R: usize, const C: usize> State<R, C> {
    pub fn new(fill: f64, epoch: u64) -> Self {
        State {
            value: na::SMatrix::<f64, R, C>::from_element(fill),
            epoch,
        }
    }

    pub fn from_matrix(value: na::SMatrix<f64, R, C>, epoch: u64) -> Self {
        State { value, epoch }
    }
}

impl<const R: usize, const C: usize> Composable for State<R, C> {
    type Output = State<R, C>;

    fn add(self, other: State<R, C>) -> State<R, C> {
        State {
            value: self.value + other.value,
            epoch: std::cmp::max(self.epoch, other.epoch),
        }
    }
}