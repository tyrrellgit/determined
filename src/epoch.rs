//! Common utilities and core traits used across the crate.
//!
//! This module re-exports `nalgebra` as `na` for convenience and provides the
//! `Algorithm` trait used by filter implementations.

use std::ops::{Add, Sub};

pub use nalgebra as na;
pub use crate::common::Composable;


#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Epoch {
    pub value: i64, // signed to allow negative time
}

impl Epoch {
    pub fn new(value: i64) -> Self {
        Epoch { value }
    }   
}

impl Add for Epoch {
    type Output = Epoch;

    fn add(self, other: Epoch) -> Epoch {
        Epoch {
            value: self.value + other.value,
        }
    }
}

impl Add for &Epoch {
    type Output = Epoch;

    fn add(self, other: &Epoch) -> Epoch {
        Epoch {
            value: self.value + other.value,
        }
    }
}

impl Sub for Epoch {
    type Output = Epoch;

    fn sub(self, other: Epoch) -> Epoch {
        Epoch {
            value: self.value - other.value,
        }
    }
}

impl Sub for &Epoch {
    type Output = Epoch;

    fn sub(self, other: &Epoch) -> Epoch {
        Epoch {
            value: self.value - other.value,
        }
    }
}