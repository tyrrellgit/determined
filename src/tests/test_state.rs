use crate::state::State;
use crate::common::Composable;
use crate::common::na as na;

#[test]
fn test_add() {
    // 1x1 states behave like scalars
    let a: State<na::U1> = State::new(vec![1.0], 0);
    let b: State<na::U1> = State::new(vec![2.0], 0);
    let result: State<na::U1> = a.add(b);
    assert_eq!(result.value[(0, 0)], 3.0);
    assert_eq!(result.epoch, 0);
}
