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

    let transition = LinearTransitionN::<2>::new(
        state.clone(),
        f,
        q
    );

    // Linear Measurement model
    let h = na::SMatrix::<f64, 2, 2>::identity();
    let r = &na::SMatrix::<f64, 2, 2>::identity() * 0.01;

    let measurement = LinearMeasurementNM::<2, 2>::new(
        h, // observation model
        r  // observation noise
    );

    // Kalman Update Model
    let update = LinearUpdateNM::<2, 2>::new(
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
/// The pseudo-inverse is chosen by shape, so a wide H gets the right inverse
/// (H H+ = I) and a tall H the left inverse (H+ H = I).
#[test]
fn test_moore_penrose_inverse_by_shape() {
    use crate::common::linalg::moore_penrose_inverse;

    // det(H^T H) of this wide H rounds to a tiny non-zero value
    let wide = na::SMatrix::<f64, 1, 2>::new(0.7, 0.9);
    let wide_inv = moore_penrose_inverse(&wide).unwrap();
    assert!(((wide * wide_inv)[(0, 0)] - 1.0).abs() < 1e-12, "H H+ = {}", wide * wide_inv);

    let tall = wide.transpose();
    let tall_inv = moore_penrose_inverse(&tall).unwrap();
    assert!(((tall_inv * tall)[(0, 0)] - 1.0).abs() < 1e-12, "H+ H = {}", tall_inv * tall);

    // rank deficient
    assert!(moore_penrose_inverse(&na::SMatrix::<f64, 1, 2>::zeros()).is_none());
}

/// With a non-invertible F, backward prediction leaves the state unchanged.
#[test]
fn test_backward_predict_with_singular_transition() {
    let _quiet = crate::tests::expect_logs();
    let state = State::<na::Const<2>>::new(vec![1.0, 2.0], vec![vec![1.0, 0.0], vec![0.0, 1.0]], Epoch::new(0));
    let mut filter = KalmanFilterNM::new(LinearUpdateNM::<2, 2>::new(
        LinearMeasurementNM::<2, 2>::new(na::SMatrix::identity(), na::SMatrix::identity()),
        LinearTransitionN::<2>::new(state.ptr(), na::SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, 0.0),
            na::SMatrix::identity())));

    let forward = filter.predict(&Epoch::new(2)).read().unwrap().clone();
    let backward = filter.predict(&Epoch::new(0)).read().unwrap().clone();
    assert_eq!(backward.value, forward.value);
    assert_eq!(backward.covariance(), forward.covariance());
    assert_eq!(backward.epoch, Epoch::new(2));
}

/// `inverse()` is `None` when a model has no inverse, at any dimension type.
#[test]
fn test_inverse_measurement_is_optional() {
    let obs = Observation { value: na::SVector::<f64, 1>::new(2.0), epoch: Epoch::new(1) };

    // full rank wide H: the right inverse
    let wide = LinearMeasurementNM::<2, 1>::new(na::SMatrix::<f64, 1, 2>::new(0.7, 0.9), na::SMatrix::identity());
    let x = wide.inverse(&obs).expect("full rank H has a pseudo-inverse");
    assert!(((0.7 * x.value[0] + 0.9 * x.value[1]) - 2.0).abs() < 1e-12);

    // rank deficient H: no inverse
    let deficient = LinearMeasurementNM::<2, 1>::new(na::SMatrix::<f64, 1, 2>::zeros(), na::SMatrix::identity());
    assert!(deficient.inverse(&obs).is_none());

    // nonlinear without with_inverse, runtime sized (used to return a 0-length state)
    let nonlinear = NonLinearMeasurement::<na::Dyn, na::Dyn>::new(
        Box::new(|s| s.value.clone()), Box::new(|_| na::DMatrix::identity(2, 2)), na::DMatrix::identity(2, 2));
    let dyn_obs = Observation { value: na::DVector::from_element(2, 1.0), epoch: Epoch::new(0) };
    assert!(nonlinear.inverse(&dyn_obs).is_none());
}
