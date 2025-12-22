//! Small typed wrapper around nalgebra static matrices representing a state or
//! observation. `State<R,C>` is a thin container that carries a `SMatrix` and
//! a simple `epoch` value used for timestamping.

use std::sync::Arc;
use std::sync::RwLock;
use std::fmt;

use crate::common::{Composable, na as na};
use crate::common::Epoch;

#[derive(Clone, Debug)]
pub struct State<T>
where
    T: na::Dim,
    na::DefaultAllocator: na::base::allocator::Allocator<T, na::U1>
        + na::allocator::Allocator<T, T>,
{
    pub value: na::Matrix<f64, T, na::U1, na::Owned<f64, T, na::U1>>,
    pub covariance: Option<na::Matrix<f64, T, T, na::Owned<f64, T, T>>>,
    pub epoch: Epoch,
}

pub type StateN<const N: usize> = State<na::Const<N>>;
pub type StateDyn = State<na::Dyn>;
pub type StatePtr<T> = Arc<RwLock<State<T>>>;

impl<T> State<T>
where
    T: na::Dim,
    na::DefaultAllocator: na::allocator::Allocator<T, na::U1>
        + na::allocator::Allocator<T, T>,
{
    pub fn new(value: Vec<f64>, covariance: Vec<Vec<f64>>, epoch: Epoch) -> Self {
        let dim = value.len();
        State {
            value: na::Matrix::<f64, T, na::U1, _>::from_row_slice_generic(
                T::from_usize(dim),
                na::U1,
                &value,
            ),
            covariance: Some(na::Matrix::<f64, T, T, _>::from_row_slice_generic(
                T::from_usize(dim),
                T::from_usize(dim),
                &covariance.concat(),
            )),
            epoch,
        }
    }

    pub fn zeros(size: usize, epoch: Epoch) -> Self {
        State {
            value: na::Matrix::<f64, T, na::U1, _>::zeros_generic(
                T::from_usize(size),
                na::U1,
            ),
            covariance: Some(na::Matrix::<f64, T, T, _>::zeros_generic(
                T::from_usize(size),
                T::from_usize(size),
            )),
            epoch,
        }
    }

    pub fn dim(&self) -> usize {
        self.value.nrows()
    }

    pub fn covariance(&self) -> na::Matrix<f64, T, T, na::Owned<f64, T, T>> {
        // TODO: stop this covariance from cloning
        self.covariance.clone().unwrap_or_else(|| {
            na::Matrix::<f64, T, T, na::Owned<f64, T, T>>::zeros_generic(
                T::from_usize(self.dim()),
                T::from_usize(self.dim()),
            )
        })
    }

    pub fn ptr(&self) -> StatePtr<T> {
        Arc::new(RwLock::new(self.clone()))
    }
}

impl<T> Default for State<T>
where
    T: na::Dim + na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<T, na::U1>
        + na::allocator::Allocator<T, T>,
{
    fn default() -> Self {
        Self {
            value: na::Matrix::<f64, T, na::U1, _>::zeros(),
            covariance: Some(na::Matrix::<f64, T, T, _>::identity()),
            epoch: Epoch::default(),
        }
    }
}

impl<T> Composable for State<T>
where
    T: na::Dim,
    na::DefaultAllocator: na::allocator::Allocator<T, na::U1>
        + na::allocator::Allocator<T, T>,
{   
    type Output = State<T>;

    fn add(self, other: Self) -> Self {
        State {
            value: &self.value + &other.value,
            covariance: Some(self.covariance() + &other.covariance()),
            epoch: self.epoch,
        }
    }
}

impl<T> fmt::Display for State<T>
where
    T: na::Dim,
    na::DefaultAllocator: na::base::allocator::Allocator<T, na::U1>
        + na::allocator::Allocator<T, T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format `value`, `covariance`, and `epoch` as desired
        write!(f, "State:\n")?;
        write!(f, "  Value: {:0.6}\n", self.value)?; // or custom formatting
        let _ = match &self.covariance {
            Some(cov) => write!(f, "  Covariance: {:0.6}\n", cov),
            None => write!(f, "  Covariance: None\n"),
        };
        write!(f, "  Epoch: {}", self.epoch.value)
    }
}