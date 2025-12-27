use crate::measurement::Observation;
use crate::state::{State, StatePtr};
use crate::filter::Filter;
use crate::algorithms::ExtendedKalmanFilterNM;
use crate::models::*;
use crate::common::na as na;
use crate::epoch::Epoch;

#[test]
fn test_extended_kalman() {

    // initial state
    let state = State::<na::Const<2>>::new(
        vec![1.0, 1.0], 
        vec![
            vec![1.0, 0.0], 
            vec![0.0, 1.0]
        ],
        Epoch::new(0)).ptr();

    // Linear Transition model
    let _f= na::SMatrix::<f64, 2, 2>::from_diagonal(
        &na::SVector::from_row_slice(&[1.5, 2.0]));

    let one_step_state = _f * state.read().unwrap().value;

    let mut _s = state.read().unwrap().value.clone();
    let f = Box::new(move | _epoch: &Epoch | {
        _s = _f * _s;
        _s
    });

    let j = Box::new(| _state: &State<na::Const<2>> |{
        na::SMatrix::<f64, 2, 2>::identity()
    });
    
    let q = &na::SMatrix::<f64, 2,2>::identity() * 0.05;

    let transition = NonLinearTransition::new(
        state.clone(),
        f,
        j,
        q
    );

    // Linear Measurement model
    let h = na::SMatrix::<f64, 2, 2>::identity();
    let r = &na::SMatrix::<f64, 2, 2>::identity() * 0.01;
    let measurement = LinearMeasurement::<2, 2>::new(
        h, // observation model
        r  // observation noise
    );

    // Kalman Update Model
    let update = NonLinearUpdate::new(
        measurement,
        transition
    );

    // Filter wrapper
    let mut filter = ExtendedKalmanFilterNM::new(
        update
    );

    let obs = Observation{
        value: na::SMatrix::<f64, 2, 1>::from_row_slice(&[1.0, 1.0]),
        epoch: Epoch::new(0),
    };

    // predict
    let step_state = filter.predict(&Epoch::new(1));
    let _value = step_state.read().unwrap().value;

    let delta = (_value - one_step_state).norm();
    assert!(delta < 1e-12, 
        "Incorrect State update: Abs Err: {}\n Expected: {} Found: {}",
        delta, one_step_state, _value);

    // measurement update
    let _ = filter.update(&obs);

}