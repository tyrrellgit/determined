use determined::measurement::Observation;
use determined::state::State;
use determined::filter::Filter;
use determined::algorithms::KalmanFilterNM;
use determined::models::*;
use determined::common::na as na;
use determined::epoch::Epoch;

use spdlog;

fn main() {

    spdlog::info!("Setting up initial state...");
    // initial state
    let state = State::<na::Const<2>>::new(
        vec![1.0, 1.0], 
        vec![
            vec![1.0, 0.0], 
            vec![0.0, 1.0]
        ],
        Epoch::new(0)).ptr();
    
    spdlog::info!("Setting up state dynamics...");
    // Linear Transition model
    let f= na::SMatrix::<f64, 2, 2>::from_diagonal(
        &na::SVector::from_row_slice(&[1.5, 2.0]));    
    let q = &na::SMatrix::<f64, 2,2>::identity() * 0.05;

    let transition = LinearTransition::<2>::new(
        state.clone(),
        f,
        q
    );

    spdlog::info!("Setting up measurement model...");
    // Linear Measurement model
    let h = na::SMatrix::<f64, 2, 2>::identity();
    let r = &na::SMatrix::<f64, 2, 2>::identity() * 0.01;

    let measurement = LinearMeasurement::<2, 2>::new(
        h, // observation model
        r  // observation noise
    );

    spdlog::info!("Building kalman filter...");
    // Kalman Update Model
    let update = LinearUpdate::<2, 2>::new(
        measurement,
        transition
    );

    // Filter wrapper
    let mut filter = KalmanFilterNM::new(
        update
    );

    spdlog::info!("Generating dummy observation...");
    let obs = Observation{
        value: na::SMatrix::<f64, 2, 1>::from_row_slice(&[1.0, 1.0]),
        epoch: Epoch::new(0),
    };

    spdlog::info!("Predicting...");
    // predict
    let _ = filter.predict(&Epoch::new(1));
    
    spdlog::info!("Updating...");
    // measurement update
    let _ = filter.update(&obs);

    spdlog::info!("Done!");
}