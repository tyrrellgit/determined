use crate::measurement::Observation;
use crate::state::State;
use crate::filter::Filter;
use crate::algorithms::{ KalmanFilterNM, UnscentedKalmanFilterNM, UnscentedKalmanFilterDyn };
use crate::models::*;
use crate::common::na as na;
use crate::epoch::Epoch;

fn state2(value: [f64; 2], var: f64) -> State<na::Const<2>> {
    State::<na::Const<2>>::new(
        value.to_vec(), vec![vec![var, 0.0], vec![0.0, var]], Epoch::new(0))
}

/// The unscented transform is exact for linear models, so the UKF must
/// match the linear Kalman filter, whatever the sigma point parameters.
#[test]
fn test_unscented_matches_linear_kalman() {
    let f = na::SMatrix::<f64, 2, 2>::new(1.0, 0.5, 0.0, 1.0);
    let h = na::SMatrix::<f64, 1, 2>::new(1.0, 0.0);
    let q = na::SMatrix::<f64, 2, 2>::identity() * 0.05;
    let r = na::SMatrix::<f64, 1, 1>::new(0.1);

    for params in [UnscentedParams::default(), UnscentedParams { alpha: 0.1, beta: 2.0, kappa: 1.0 }] {
        let mut kf = KalmanFilterNM::new(LinearUpdateNM::<2, 1>::new(
            LinearMeasurementNM::<2, 1>::new(h, r),
            LinearTransitionN::<2>::new(state2([0.0, 1.0], 1.0).ptr(), f, q),
        ));
        let mut ukf = UnscentedKalmanFilterNM::new(UnscentedUpdateNM::<2, 1>::new(
            UnscentedMeasurementNM::<2, 1>::new(Box::new(move |s| h * s.value), r),
            UnscentedTransitionN::<2>::new(state2([0.0, 1.0], 1.0).ptr(), Box::new(move |s| f * s.value), q)
                .with_params(params),
        ));

        for t in [1, 2, 4, 5] {
            let obs = Observation { value: na::SVector::<f64, 1>::new(0.4 * t as f64), epoch: Epoch::new(t) };
            let a = kf.update(&obs);
            let b = ukf.update(&obs);
            let (a, b) = (a.read().unwrap(), b.read().unwrap());

            assert!((a.value - b.value).norm() < 1e-9, "{:?}: state {} vs {}", params, a.value, b.value);
            assert!((a.covariance() - b.covariance()).norm() < 1e-9, "{:?}: covariance", params);
            assert_eq!(a.epoch, b.epoch);
        }
    }
}

/// For f(x) = x^2 with x ~ N(m, s^2), the default sigma points recover the
/// exact mean m^2 + s^2 and variance 4 m^2 s^2 + 2 s^4. (The EKF gives m^2.)
#[test]
fn test_unscented_transform_moments() {
    let (m, var) = (2.0, 0.5);
    let state = State::<na::Const<1>>::new(vec![m], vec![vec![var]], Epoch::new(0)).ptr();
    let mut transition = UnscentedTransitionN::<1>::new(
        state, Box::new(|s| s.value.map(|x| x * x)), na::SMatrix::<f64, 1, 1>::zeros());

    let out = transition.state(&Epoch::new(1)).read().unwrap().clone();
    assert!((out.value[0] - (m * m + var)).abs() < 1e-12, "mean {}", out.value[0]);
    let expected_var = 4.0 * m * m * var + 2.0 * var * var;
    assert!((out.covariance()[(0, 0)] - expected_var).abs() < 1e-12, "variance {}", out.covariance()[(0, 0)]);
}

/// Range-only sensor on a static target. Matches an independent numpy UKF
/// (same sigma points). The range constrains only the radius, so with large
/// tangential uncertainty the posterior is ring-shaped and its mean lies
/// inside the ring: E|x|^2 = |mean|^2 + tr(P) approaches 25, not |mean|.
#[test]
fn test_unscented_range_sensor() {
    // radius of the mean after each update, from the numpy reference
    let reference = [3.8639, 4.3361, 4.4525, 4.4974, 4.5210];

    let mut ukf = UnscentedKalmanFilterNM::new(UnscentedUpdateNM::<2, 1>::new(
        UnscentedMeasurementNM::<2, 1>::new(
            Box::new(|s| na::SVector::<f64, 1>::new(s.value.norm())),
            na::SMatrix::<f64, 1, 1>::new(0.01)),
        UnscentedTransitionN::<2>::new(
            state2([2.0, 2.0], 4.0).ptr(), Box::new(|s| s.value), na::SMatrix::<f64, 2, 2>::zeros()),
    ));

    let mut second_moment = 8.0 + 8.0;
    for (t, expected) in (1..=5).zip(reference) {
        let state = ukf.update(&Observation { value: na::SVector::<f64, 1>::new(5.0), epoch: Epoch::new(t) });
        let state = state.read().unwrap();

        assert!((state.value.norm() - expected).abs() < 1e-4,
            "t={}: radius {} vs reference {}", t, state.value.norm(), expected);

        let moment = state.value.norm_squared() + state.covariance().trace();
        assert!(moment > second_moment && moment < 25.0, "t={}: E|x|^2 = {}", t, moment);
        second_moment = moment;
    }
}

/// Runtime sized (`na::Dyn`, as used by Python) matches statically sized exactly.
#[test]
fn test_unscented_dyn_matches_static() {
    let range = |x: &[f64]| (x[0] * x[0] + x[1] * x[1]).sqrt();

    let mut fixed = UnscentedKalmanFilterNM::new(UnscentedUpdateNM::<2, 1>::new(
        UnscentedMeasurementNM::<2, 1>::new(
            Box::new(move |s| na::SVector::<f64, 1>::new(range(s.value.as_slice()))),
            na::SMatrix::<f64, 1, 1>::new(0.01)),
        UnscentedTransitionN::<2>::new(
            state2([2.0, 2.0], 4.0).ptr(), Box::new(|s| s.value * 1.01),
            na::SMatrix::<f64, 2, 2>::identity() * 0.01),
    ));
    let mut dynamic = UnscentedKalmanFilterDyn::new(UnscentedUpdate::new(
        UnscentedMeasurement::new(
            Box::new(move |s| na::DVector::from_element(1, range(s.value.as_slice()))),
            na::DMatrix::from_element(1, 1, 0.01)),
        UnscentedTransition::new(
            State::<na::Dyn>::new(vec![2.0, 2.0], vec![vec![4.0, 0.0], vec![0.0, 4.0]], Epoch::new(0)).ptr(),
            Box::new(|s| &s.value * 1.01),
            na::DMatrix::identity(2, 2) * 0.01),
    ));

    for t in 1..=4 {
        let epoch = Epoch::new(t);
        let a = fixed.update(&Observation { value: na::SVector::<f64, 1>::new(5.0), epoch });
        let b = dynamic.update(&Observation { value: na::DVector::from_element(1, 5.0), epoch });
        let (a, b) = (a.read().unwrap(), b.read().unwrap());
        assert_eq!(a.value.as_slice(), b.value.as_slice());
        assert_eq!(a.covariance().as_slice(), b.covariance().as_slice());
    }
}

/// Same-epoch predict does nothing; backward predict and a covariance that is
/// not positive definite leave the state unchanged.
#[test]
fn test_unscented_edge_cases() {
    let mut transition = UnscentedTransitionN::<2>::new(
        state2([1.0, 1.0], 1.0).ptr(), Box::new(|s| s.value * 2.0), na::SMatrix::<f64, 2, 2>::zeros());

    let before = transition.state.read().unwrap().clone();
    transition.state(&Epoch::new(0));
    transition.state(&Epoch::new(-1));
    let after = transition.state.read().unwrap().clone();
    assert_eq!(before.value, after.value);
    assert_eq!(before.epoch, after.epoch);

    let mut bad = UnscentedTransitionN::<2>::new(
        state2([1.0, 1.0], -1.0).ptr(), Box::new(|s| s.value * 2.0), na::SMatrix::<f64, 2, 2>::zeros());
    bad.state(&Epoch::new(1));
    let after = bad.state.read().unwrap().clone();
    assert_eq!(after.value, na::SVector::<f64, 2>::new(1.0, 1.0));
    assert_eq!(after.epoch, Epoch::new(0));
}

/// The measurement reports its noise, and uses the inverse supplied with
/// `with_inverse`.
#[test]
fn test_unscented_measurement_noise_and_inverse() {
    let r = na::SMatrix::<f64, 1, 1>::new(0.25);
    let measurement = UnscentedMeasurementNM::<2, 1>::new(
        Box::new(|s| na::SVector::<f64, 1>::new(s.value[0])), r)
        .with_inverse(Box::new(|obs| State::<na::Const<2>>::new(
            vec![obs.value[0], 0.0], vec![vec![1.0, 0.0], vec![0.0, 1.0]], obs.epoch)));

    assert_eq!(measurement.noise(), r);
    let obs = Observation { value: na::SVector::<f64, 1>::new(3.0), epoch: Epoch::new(2) };
    let x = measurement.inverse(&obs);
    assert_eq!(x.value, na::SVector::<f64, 2>::new(3.0, 0.0));
    assert_eq!(x.epoch, Epoch::new(2));
}
