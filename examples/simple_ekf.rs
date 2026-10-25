//! Extended Kalman filter tracking a static 2D target from range-only
//! measurements, a nonlinear sensor h(x) = |x|.

use determined::algorithms::ExtendedKalmanFilterNM;
use determined::common::na;
use determined::epoch::Epoch;
use determined::filter::Filter;
use determined::measurement::Observation;
use determined::models::*;
use determined::state::State;

fn main() {
    // initial guess (2, 2) with a wide covariance; the target is really at (3, 4)
    let state = State::<na::Const<2>>::new(
        vec![2.0, 2.0],
        vec![vec![4.0, 0.0], vec![0.0, 4.0]],
        Epoch::new(0),
    )
    .ptr();

    // static target: x(t + 1) = x(t)
    let transition = NonLinearTransitionN::<2>::new(
        state.clone(),
        Box::new(|state| state.value),
        Box::new(|_| na::SMatrix::<f64, 2, 2>::identity()),
        na::SMatrix::<f64, 2, 2>::zeros(),
    );

    // range sensor and its Jacobian
    let measurement = NonLinearMeasurementNM::<2, 1>::new(
        Box::new(|state| na::SMatrix::<f64, 1, 1>::new(state.value.norm())),
        Box::new(|state| {
            let range = state.value.norm();
            na::SMatrix::<f64, 1, 2>::new(state.value[0] / range, state.value[1] / range)
        }),
        na::SMatrix::<f64, 1, 1>::new(0.01),
    );

    let mut filter =
        ExtendedKalmanFilterNM::new(NonLinearUpdateNM::<2, 1>::new(measurement, transition));

    for t in 1..=5 {
        let obs = Observation {
            value: na::SMatrix::<f64, 1, 1>::new(5.0),
            epoch: Epoch::new(t),
        };
        filter.predict(&obs.epoch);
        let estimate = filter.update(&obs);
        println!("t={} {}", t, estimate.read().unwrap());
    }
}
