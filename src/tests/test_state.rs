use crate::state::State;
use crate::common::Composable;

#[test]
fn test_add() {
    // 1x1 states behave like scalars
    let a = State::<1, 1>::new(1.0, 0);
    let b = State::<1, 1>::new(2.0, 0);
    let result = a.add(b);
    assert_eq!(result.value[(0, 0)], 3.0);
    assert_eq!(result.epoch, 0);
}
