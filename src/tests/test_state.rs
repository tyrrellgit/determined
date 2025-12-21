use crate::common::Epoch;
use crate::state::{ State, StateN, StateDyn };
use crate::common::Composable;
use crate::common::na as na;

#[test]
fn test_add() {
    // 1x1 states behave like scalars
    let a: State<na::U1> = State::new(vec![1.0], vec![vec![0.0]], Epoch::new(0));
    let b: State<na::U1> = State::new(vec![2.0], vec![vec![0.0]], Epoch::new(0));
    let result: State<na::U1> = a.add(b);
    assert_eq!(result.value[(0, 0)], 3.0);
    assert_eq!(result.epoch.value, 0);
}

#[test]
fn static_alloc() {
    // Static 4D state (compile-time size)
    let _ = State::<na::Const<4>> {
        value: na::SVector::<f64, 4>::zeros(),
        covariance: Some(na::SMatrix::<f64, 4, 4>::identity()),
        epoch: Epoch::new(0),
    };

    let _ = StateN::<4> {
        value: na::SVector::<f64, 4>::zeros(),
        covariance: Some(na::SMatrix::<f64, 4, 4>::identity()),
        epoch: Epoch::new(0),
    };
}


#[test]
fn runtime_alloc() {
    // Dynamic state (runtime size)
    let _ = State::<na::Dyn> {
        value: na::DVector::<f64>::zeros(4),
        covariance: Some(na::DMatrix::<f64>::identity(4, 4)),
        epoch: Epoch::new(0),
    };

    let _ = StateDyn {
        value: na::DVector::<f64>::zeros(4),
        covariance: Some(na::DMatrix::<f64>::identity(4, 4)),
        epoch: Epoch::new(0),
    };
}
