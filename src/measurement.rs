//! Measurement helpers and observation types.
//!
//! `Measurement` is a small container that groups a state, observation and a
//! joint covariance / jacobian. It provides convenience constructors and a
//! minimal update/predict placeholder useful for tests and examples.

use std::fmt;

use crate::common::na;
use crate::epoch::Epoch;

#[derive(Clone, Debug)]
pub struct Observation<T>
where
    T: na::Dim,
    na::DefaultAllocator: na::base::allocator::Allocator<T, na::U1>
        + na::allocator::Allocator<T, T>,
{
    pub value: na::Matrix<f64, T, na::U1, na::Owned<f64, T, na::U1>>,
    pub epoch: Epoch,
}

impl<T> fmt::Display for Observation<T>
where
    T: na::Dim,
    na::DefaultAllocator: na::base::allocator::Allocator<T, na::U1>
        + na::allocator::Allocator<T, T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format `value`, `covariance`, and `epoch` as desired
        writeln!(f, "Observation:")?;
        writeln!(f, "‾‾‾‾‾‾‾‾‾‾‾‾")?;
        write!(f, "  Value: {:0.6}", self.value)?; 
        writeln!(f, "  Epoch: {}", self.epoch.value)
    }
}