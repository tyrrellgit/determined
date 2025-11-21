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

/// Core algorithm trait shared across algorithm implementations.
///
/// Implementations expose their concrete `StateType` and `ObservationType` as
/// associated types. The trait intentionally returns references from
/// `predict`/`update` to avoid unnecessary cloning of large state vectors.
///
/// Minimal contract:
/// - `new()` constructs a reasonable default instance.
/// - `predict(epoch)` advances internal state (e.g. x, P) and returns a
///   reference to the predicted state.
/// - `update(obs)` fuses an observation into the internal state and returns
///   a reference to the posterior state.
pub trait Algorithm {
    type StateType;
    type ObservationType;

    fn new() -> Self
    where
        Self: Sized;

    fn predict<'a>(&'a mut self, epoch: &Epoch) -> &'a Self::StateType;
    fn update<'a>(&'a mut self, observation: &Self::ObservationType) -> &'a Self::StateType;
    fn state(&self) -> &Self::StateType;
    fn state_mut(&mut self) -> &mut Self::StateType;
    fn reset(&mut self);
}