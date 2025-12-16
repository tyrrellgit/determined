//! Small typed wrapper around nalgebra static matrices representing a state or
//! observation. `State<R,C>` is a thin container that carries a `SMatrix` and
//! a simple `epoch` value used for timestamping.

use crate::common::{Composable, na as na};

#[derive(Clone, Debug)]
pub struct State<T>
where
    T: na::Dim,
    na::DefaultAllocator: na::base::allocator::Allocator<T, na::U1>,
{
    pub value: na::Matrix<f64, T, na::U1, na::Owned<f64, T, na::U1>>,
    pub epoch: u64,
}

impl<T> State<T>
where
    T: na::Dim,
    na::DefaultAllocator: na::allocator::Allocator<T, na::U1>,
{
    pub fn new(value: Vec<f64>, epoch: u64) -> Self {
        let dim = value.len();
        State {
            value: na::Matrix::<f64, T, na::U1, _>::from_row_slice_generic(
                T::from_usize(dim),
                na::U1,
                &value,
            ),
            epoch,
        }
    }

    pub fn zeros(size: usize, epoch: u64) -> Self {
        State {
            value: na::Matrix::<f64, T, na::U1, _>::zeros_generic(
                T::from_usize(size),
                na::U1,
            ),
            epoch,
        }
    }

    pub fn dim(&self) -> usize {
        self.value.nrows()
    }
}

impl<T> Default for State<T>
where
    T: na::Dim + na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<T, na::U1>,
{
    fn default() -> Self {
        Self {
            value: na::Matrix::<f64, T, na::U1, _>::zeros(),
            epoch: 0,
        }
    }
}

impl<T> Composable for State<T>
where
    T: na::Dim,
    na::DefaultAllocator: na::allocator::Allocator<T, na::U1>,
{   
    type Output = State<T>;

    fn add(self, other: Self) -> Self {
        State {
            value: &self.value + &other.value,
            epoch: self.epoch,
        }
    }
}