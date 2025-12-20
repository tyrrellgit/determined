//! `Filter` is a trait for basic algorithms. 

use crate::common::Epoch;

/// Core Filter trait shared across algorithm implementations.
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
pub trait Filter {
    type StateType;
    type ObservationType;

    fn predict<'a>(&'a mut self, epoch: &Epoch) -> Self::StateType;
    fn update<'a>(&'a mut self, observation: &Self::ObservationType) -> Self::StateType;
    fn state<'a>(&'a self) -> Self::StateType;
    fn reset(&mut self);
}
