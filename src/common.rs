//! Common utilities and core traits used across the crate.
//!
//! This module re-exports `nalgebra` as `na` for convenience and provides the
//! `Algorithm` trait used by filter implementations.

pub use nalgebra as na;
pub use crate::composition::Composable;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Epoch {
    pub value: u64,
}