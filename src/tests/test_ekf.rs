use crate::measurement::Observation;
use crate::state::State;
use crate::filter::Filter;
use crate::algorithms::{ ExtendedKalmanFilterNM, KalmanFilterNM };
use crate::models::*;
use crate::common::na as na;
use crate::epoch::Epoch;

fn initial_state() -> State<na::Const<2>> {
    State::<na::Const<2>>::new(
        vec![1.0, 1.0], 
        vec![
            vec![1.0, 0.0], 
            vec![0.0, 1.0]
        ],
        Epoch::new(0))
}

/// A linear problem run through the EKF must match the linear Kalman filter.
#[test]
fn test_extended_kalman_matches_linear() {
    let f = na::SMatrix::<f64, 2, 2>::from_diagonal(
        &na::SVector::from_row_slice(&[1.5, 0.5]));
    let q = &na::SMatrix::<f64, 2, 2>::identity() * 0.05;
    let h = na::SMatrix::<f64, 2, 2>::identity();
    let r = &na::SMatrix::<f64, 2, 2>::identity() * 0.01;

    // Linear filter
    let mut kf = KalmanFilterNM::new(LinearUpdateNM::<2, 2>::new(
        LinearMeasurementNM::<2, 2>::new(h, r),
        LinearTransitionN::<2>::new(initial_state().ptr(), f, q),
    ));

    // Extended filter
    let mut ekf = ExtendedKalmanFilterNM::new(NonLinearUpdateNM::<2, 2>::new(
        NonLinearMeasurementNM::<2, 2>::linear(h, r),
        NonLinearTransitionN::<2>::new(
            initial_state().ptr(),
            Box::new(move |state| f * state.value),
            Box::new(move |_| f),
            q,
        ),
    ));

    for t in 1..=3 {
        let obs = Observation{
            value: na::SMatrix::<f64, 2, 1>::from_row_slice(&[1.0 + t as f64, 0.5]),
            epoch: Epoch::new(t),
        };

        kf.predict(&obs.epoch);
        ekf.predict(&obs.epoch);
        let kf_state = kf.update(&obs);
        let ekf_state = ekf.update(&obs);

        let kf_state = kf_state.read().unwrap();
        let ekf_state = ekf_state.read().unwrap();

        let dx = (kf_state.value - ekf_state.value).norm();
        let dp = (kf_state.covariance() - ekf_state.covariance()).norm();
        assert!(dx < 1e-12, "State mismatch at t={}: {}", t, dx);
        assert!(dp < 1e-12, "Covariance mismatch at t={}: {}", t, dp);
        assert_eq!(ekf_state.epoch, Epoch::new(t));
    }
}

/// Multi-step prediction applies `f` once per epoch, and is idempotent.
#[test]
fn test_extended_kalman_predict_steps() {
    let mut ekf = ExtendedKalmanFilterNM::new(NonLinearUpdateNM::<2, 2>::new(
        NonLinearMeasurementNM::<2, 2>::linear(
            na::SMatrix::<f64, 2, 2>::identity(),
            na::SMatrix::<f64, 2, 2>::identity()),
        NonLinearTransitionN::<2>::new(
            initial_state().ptr(),
            Box::new(|state| state.value * 2.0),
            Box::new(|_| na::SMatrix::<f64, 2, 2>::identity() * 2.0),
            na::SMatrix::<f64, 2, 2>::zeros(),
        ),
    ));

    let state = ekf.predict(&Epoch::new(3));
    assert_eq!(state.read().unwrap().value, na::SVector::<f64, 2>::new(8.0, 8.0));
    assert_eq!(state.read().unwrap().covariance(), na::SMatrix::<f64, 2, 2>::identity() * 64.0);

    // predicting to the same epoch again must not propagate
    let state = ekf.predict(&Epoch::new(3));
    assert_eq!(state.read().unwrap().value, na::SVector::<f64, 2>::new(8.0, 8.0));
}

/// A genuinely nonlinear (range) sensor pulls the estimate towards the truth
/// and shrinks the uncertainty.
#[test]
fn test_extended_kalman_range_sensor() {
    let truth_range = 5.0; // true position is (3, 4)

    let measurement = NonLinearMeasurementNM::<2, 1>::new(
        Box::new(|state| na::SMatrix::<f64, 1, 1>::new(state.value.norm())),
        Box::new(|state| {
            let rng = state.value.norm();
            na::SMatrix::<f64, 1, 2>::new(state.value[0] / rng, state.value[1] / rng)
        }),
        na::SMatrix::<f64, 1, 1>::new(0.01),
    );

    // static target
    let transition = NonLinearTransitionN::<2>::new(
        State::<na::Const<2>>::new(
            vec![2.0, 2.0],
            vec![vec![4.0, 0.0], vec![0.0, 4.0]],
            Epoch::new(0)).ptr(),
        Box::new(|state| state.value),
        Box::new(|_| na::SMatrix::<f64, 2, 2>::identity()),
        na::SMatrix::<f64, 2, 2>::zeros(),
    );

    let mut ekf = ExtendedKalmanFilterNM::new(
        NonLinearUpdateNM::<2, 1>::new(measurement, transition));

    let mut err = (2.0f64 * 2.0 + 2.0 * 2.0).sqrt() - truth_range;
    let mut trace = 8.0;
    for t in 1..=5 {
        let state = ekf.update(&Observation{
            value: na::SMatrix::<f64, 1, 1>::new(truth_range),
            epoch: Epoch::new(t),
        });
        let state = state.read().unwrap();

        let new_err = state.value.norm() - truth_range;
        let new_trace = state.covariance().trace();
        assert!(new_err.abs() < err.abs(), "Range error did not shrink at t={}", t);
        assert!(new_trace < trace, "Uncertainty did not shrink at t={}", t);
        err = new_err;
        trace = new_trace;
    }
    assert!(err.abs() < 1e-2, "Range error too large: {}", err);
}
