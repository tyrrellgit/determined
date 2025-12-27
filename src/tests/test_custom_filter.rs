use crate::measurement::Observation;
use crate::state::State;
use crate::filter::Filter;
use crate::algorithms::KalmanFilterNM;
use crate::models::*;
use crate::common::na as na;
use crate::epoch::Epoch;

#[test]
fn test_kalman_custom() {

    // initial state
    let state = State::<na::Const<2>>::new(
        vec![1.0, 1.0], 
        vec![
            vec![1.0, 0.0], 
            vec![0.0, 1.0]
        ],
        Epoch::new(0)).ptr();

    // Linear Transition model
    let f= na::SMatrix::<f64, 2, 2>::from_diagonal(
        &na::SVector::from_row_slice(&[1.5, 2.0]));

    let one_step_state = f * state.read().unwrap().value;
    
    let q = &na::SMatrix::<f64, 2,2>::identity() * 0.05;

    let transition = LinearTransition::<2>::new(
        state.clone(),
        f,
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
    let update = LinearUpdate::<2, 2>::new(
        measurement,
        transition
    );

    // Filter wrapper
    let mut filter = KalmanFilterNM::new(
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