use crate::measurement::Observation;
use crate::state::State;
use crate::filter::Filter;
use crate::algorithms::{ KalmanFilterNM, KalmanFilterDyn, ExtendedKalmanFilterNM, ExtendedKalmanFilterDyn };
use crate::models::*;
use crate::common::na as na;
use crate::epoch::Epoch;

const F: [f64; 4] = [1.0, 0.5, 0.0, 1.0];
const H: [f64; 2] = [1.0, 0.0];

fn obs_at(t: i64) -> (f64, Epoch) {
    (0.4 * t as f64, Epoch::new(t))
}

/// The same models with runtime sizes (`na::Dyn`, as used by Python) must
/// give exactly the statically sized result.
#[test]
fn test_dyn_matches_static_kalman() {
    let f = na::SMatrix::<f64, 2, 2>::from_row_slice(&F);
    let h = na::SMatrix::<f64, 1, 2>::from_row_slice(&H);
    let q = na::SMatrix::<f64, 2, 2>::identity() * 0.05;
    let r = na::SMatrix::<f64, 1, 1>::new(0.1);

    let state = || State::<na::Const<2>>::new(
        vec![0.0, 1.0], vec![vec![1.0, 0.0], vec![0.0, 1.0]], Epoch::new(0));
    let dyn_state = || State::<na::Dyn>::new(
        vec![0.0, 1.0], vec![vec![1.0, 0.0], vec![0.0, 1.0]], Epoch::new(0));

    let mut fixed = KalmanFilterNM::new(LinearUpdateNM::<2, 1>::new(
        LinearMeasurementNM::<2, 1>::new(h, r),
        LinearTransitionN::<2>::new(state().ptr(), f, q),
    ));
    let mut dynamic = KalmanFilterDyn::new(LinearUpdate::new(
        LinearMeasurement::new(na::DMatrix::from_row_slice(1, 2, &H), na::DMatrix::from_element(1, 1, 0.1)),
        LinearTransition::new(dyn_state().ptr(), na::DMatrix::from_row_slice(2, 2, &F), na::DMatrix::identity(2, 2) * 0.05),
    ));

    for t in [1, 2, 5, 6] {
        let (z, epoch) = obs_at(t);
        let a = fixed.update(&Observation { value: na::SVector::<f64, 1>::new(z), epoch });
        let b = dynamic.update(&Observation { value: na::DVector::from_element(1, z), epoch });
        let (a, b) = (a.read().unwrap(), b.read().unwrap());

        assert_eq!(a.value.as_slice(), b.value.as_slice());
        assert_eq!(a.covariance().as_slice(), b.covariance().as_slice());
        assert_eq!(a.epoch, b.epoch);
    }
}

#[test]
fn test_dyn_matches_static_extended_kalman() {
    let range = |x: &[f64]| (x[0] * x[0] + x[1] * x[1]).sqrt();

    let mut fixed = ExtendedKalmanFilterNM::new(NonLinearUpdateNM::<2, 1>::new(
        NonLinearMeasurementNM::<2, 1>::new(
            Box::new(move |s| na::SVector::<f64, 1>::new(range(s.value.as_slice()))),
            Box::new(move |s| na::SMatrix::<f64, 1, 2>::new(
                s.value[0] / range(s.value.as_slice()), s.value[1] / range(s.value.as_slice()))),
            na::SMatrix::<f64, 1, 1>::new(0.01)),
        NonLinearTransitionN::<2>::new(
            State::<na::Const<2>>::new(vec![2.0, 2.0], vec![vec![4.0, 0.0], vec![0.0, 4.0]], Epoch::new(0)).ptr(),
            Box::new(|s| s.value * 1.01),
            Box::new(|_| na::SMatrix::<f64, 2, 2>::identity() * 1.01),
            na::SMatrix::<f64, 2, 2>::identity() * 0.01),
    ));
    let mut dynamic = ExtendedKalmanFilterDyn::new(NonLinearUpdate::new(
        NonLinearMeasurement::new(
            Box::new(move |s| na::DVector::from_element(1, range(s.value.as_slice()))),
            Box::new(move |s| na::DMatrix::from_row_slice(1, 2, &[
                s.value[0] / range(s.value.as_slice()), s.value[1] / range(s.value.as_slice())])),
            na::DMatrix::from_element(1, 1, 0.01)),
        NonLinearTransition::new(
            State::<na::Dyn>::new(vec![2.0, 2.0], vec![vec![4.0, 0.0], vec![0.0, 4.0]], Epoch::new(0)).ptr(),
            Box::new(|s| &s.value * 1.01),
            Box::new(|_| na::DMatrix::identity(2, 2) * 1.01),
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
