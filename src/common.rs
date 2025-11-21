pub use nalgebra as na;
pub use crate::composition::Composable;

pub struct Epoch {
    pub value: u64
}

/// Core algorithm trait shared across algorithm implementations.
/// Implementations can choose their own StateType and ObservationType.
pub trait Algorithm {
    type StateType;
    type ObservationType;

    fn new() -> Self where Self: Sized;
    fn predict<'a>(&'a mut self, epoch: &Epoch) -> &'a Self::StateType;
    fn update<'a>(&'a mut self, observation: &Self::ObservationType) -> &'a Self::StateType;
    fn state(&self) -> &Self::StateType;
    fn state_mut(&mut self) -> &mut Self::StateType;
    fn reset(&mut self);
}