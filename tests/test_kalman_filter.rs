use determined::algorithms::KalmanFilterNM;
use determined::filter::Filter;
use determined::common::na as na;
use determined::measurement::Observation;
use determined::state::{ State, StatePtr };
use determined::epoch::Epoch;

#[test]
fn test_kalman_filter() {

    let x: StatePtr<na::Const<1>> = State::<na::Const<1>>::new(
        vec![1.0], 
        vec![vec![1.0]],
        Epoch{ value: 0 }).ptr();

    let mut kf: KalmanFilterNM::<1, 1> = 
        KalmanFilterNM::default_from_state(x);

    let epoch = Epoch { value: 0 };
    let z = Observation::<na::Const<1>> {
        value: na::SMatrix::<f64, 1, 1>::from_row_slice(&[1.2]),
        epoch: epoch,
    }; 

    let _ = kf.predict(&epoch);
    let _ = kf.update(&z);

    let s = kf.state();
    println!("state = {:?}", s);
}