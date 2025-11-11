use crate::common::Composable;

use crate::common::na as na;

pub struct State< const R: usize, const C: usize > {
    pub value: na::Matrix< f64, R, C >,
    pub epoch: u64,
};

impl State {
    pub fn new(value: f64, epoch: u64) -> Self {
        State { value, epoch }
    }
};

impl Composable for State {
    type Output = State;

    fn add(self, other: State) -> State {
        // Implementation of adding two states
        State { /* fields */ }
    }
}