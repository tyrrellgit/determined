//! Unscented Kalman filter tracking a target moving at constant velocity with
//! a range-and-bearing sensor (a radar). No Jacobians are needed: sigma points
//! are pushed through f and h instead.

use determined::algorithms::UnscentedKalmanFilterNM;
use determined::common::na;
use determined::epoch::Epoch;
use determined::filter::Filter;
use determined::measurement::Observation;
use determined::models::*;
use determined::state::State;

fn main() {
    // state x = (px, py, vx, vy), one epoch per second
    let f = na::SMatrix::<f64, 4, 4>::new(
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    );
    let q = na::SMatrix::<f64, 4, 4>::from_diagonal(&na::SVector::<f64, 4>::new(
        0.01, 0.01, 0.001, 0.001,
    ));
    let r = na::SMatrix::<f64, 2, 2>::from_diagonal(&na::SVector::<f64, 2>::new(
        0.1 * 0.1,
        0.01 * 0.01,
    ));
    let radar =
        |x: &na::SVector<f64, 4>| na::SVector::<f64, 2>::new(x[0].hypot(x[1]), x[1].atan2(x[0]));

    // the target starts at (10, 5) moving at (-0.5, 0.3); the initial guess is poor
    let mut truth = na::SVector::<f64, 4>::new(10.0, 5.0, -0.5, 0.3);
    let state = State::<na::Const<4>>::new(
        vec![9.0, 6.0, 0.0, 0.0],
        vec![
            vec![4.0, 0.0, 0.0, 0.0],
            vec![0.0, 4.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ],
        Epoch::new(0),
    )
    .ptr();

    let transition = UnscentedTransitionN::<4>::new(state, Box::new(move |s| f * s.value), q);
    let measurement = UnscentedMeasurementNM::<4, 2>::new(Box::new(move |s| radar(&s.value)), r);
    let mut filter =
        UnscentedKalmanFilterNM::new(UnscentedUpdateNM::<4, 2>::new(measurement, transition));

    for t in 1..=20 {
        truth = f * truth;
        // deterministic stand-in for sensor noise
        let noise =
            na::SVector::<f64, 2>::new(0.1 * (1.7 * t as f64).sin(), 0.01 * (2.3 * t as f64).cos());
        let obs = Observation {
            value: radar(&truth) + noise,
            epoch: Epoch::new(t),
        };

        filter.predict(&obs.epoch);
        let estimate = filter.update(&obs);
        if t % 5 == 0 {
            let x = estimate.read().unwrap().value;
            println!(
                "t={:2} position=({:.3}, {:.3}) velocity=({:.3}, {:.3}) position error={:.3}",
                t,
                x[0],
                x[1],
                x[2],
                x[3],
                ((x[0] - truth[0]).powi(2) + (x[1] - truth[1]).powi(2)).sqrt()
            );
        }
    }
}
