use crate::measurement::Observation;
use crate::state::State;
use crate::filter::Filter;
use crate::algorithms::KalmanFilterNM;
use crate::common::na as na;
use crate::epoch::Epoch;

#[test]
fn test_kalman_update_scalar() {
    // Small 1D Kalman filter: N=1, M=1

    let x = State::<na::Const<1>>::new(
        vec![0.0], 
        vec![vec![1.0]],
        Epoch::new(0)).ptr();

    let mut filter = KalmanFilterNM::<1, 1>::default_from_state(x);
    // initial state x = 0

    assert_eq!(filter.state().read().unwrap().value[(0, 0)], 0.0);

    let obs = Observation{
        value: na::SMatrix::<f64, 1, 1>::from_row_slice(&[1.0]),
        epoch: Epoch::new(0),
    };

    // predict (block pattern to avoid borrow conflict)
    { let _ = filter.predict(&Epoch::new(1)); }

    // measurement update
    filter.measurement.h = na::SMatrix::<f64, 1, 1>::identity();
    filter.measurement.r = na::SMatrix::<f64, 1, 1>::identity();
    let _ = filter.update(&obs);

    // With P=I, R=I, H=I we expect K = 1/2 and x -> 0.5
    let x_val = filter.state().read().unwrap().value[(0, 0)];
    assert!((x_val - 0.5).abs() < 1e-12, "x after update = {}", x_val);
}

#[test]
fn test_kalman_update_2d() {
    let x = State::<na::Const<2>>::new(
        vec![0.0, 0.0], 
        vec![
            vec![1.0, 0.0],
            vec![0.0,1.0]],
        Epoch::new(0)).ptr();

    // N=2, M=1. Observe only the first state element.
    let mut filter = KalmanFilterNM::<2, 1>::default_from_state(x);
    
    filter.measurement.h = na::SMatrix::<f64, 1, 2>::from_row_slice(&[1.0, 0.0]);

    let obs = Observation{
        value: na::SMatrix::<f64, 1, 1>::from_row_slice(&[2.0]),
        epoch: Epoch::new(0),
    };

    // block pattern to avoid borrow conflict
    { let _ = filter.predict(&Epoch { value: 1 }); }
    { let _ = filter.update(&obs); }

    // Expect first state to be 1.0 (K = [0.5, 0]^T, y = 2)
    let state_ptr = filter.state();
    let state = state_ptr.read().unwrap();
    let x0 = state.value[(0, 0)];
    let x1 = state.value[(1, 0)];
    assert!((x0 - 1.0).abs() < 1e-12, "x0 after update = {}", x0);
    assert!((x1 - 0.0).abs() < 1e-12, "x1 after update = {}", x1);
}