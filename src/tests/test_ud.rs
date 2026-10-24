use crate::measurement::Observation;
use crate::state::State;
use crate::filter::Filter;
use crate::algorithms::{
    KalmanFilterNM, ExtendedKalmanFilterNM,
    UDKalmanFilterNM, UDKalmanFilterDyn, UDExtendedKalmanFilterNM,
};
use crate::common::linalg::udu_decomposition;
use crate::models::*;
use crate::common::na as na;
use crate::epoch::Epoch;

fn state3() -> State<na::Const<3>> {
    State::<na::Const<3>>::new(
        vec![1.0, -0.5, 2.0],
        vec![vec![2.0, 0.3, 0.1], vec![0.3, 1.0, -0.2], vec![0.1, -0.2, 0.5]],
        Epoch::new(0))
}

fn close<R: na::Dim, C: na::Dim, S1, S2>(a: &na::Matrix<f64, R, C, S1>, b: &na::Matrix<f64, R, C, S2>) -> bool
where
    S1: na::Storage<f64, R, C>,
    S2: na::Storage<f64, R, C>,
{
    let diff = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    diff <= 1e-9 * (1.0 + b.norm())
}

#[test]
fn test_udu_decomposition() {
    let m = state3().covariance();
    let (u, d) = udu_decomposition(&m);
    for i in 0..3 {
        assert_eq!(u[(i, i)], 1.0);
        for j in 0..i {
            assert_eq!(u[(i, j)], 0.0, "U must be upper triangular");
        }
    }
    let rebuilt = u * na::SMatrix::<f64, 3, 3>::from_diagonal(&d) * u.transpose();
    assert!(close(&rebuilt, &m), "{} vs {}", rebuilt, m);

    // semi-definite: a zero variance gives a zero pivot, not a NaN
    let semi = na::SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, 0.0);
    let (u, d) = udu_decomposition(&semi);
    assert!(u.iter().chain(d.iter()).all(|x| x.is_finite()));
    assert!(close(&(u * na::SMatrix::<f64, 2, 2>::from_diagonal(&d) * u.transpose()), &semi));
}

/// The UD filter is the Kalman filter in factored form: same estimates, for
/// diagonal and non-diagonal R (decorrelated), non-diagonal Q and epoch gaps.
#[test]
fn test_ud_matches_kalman() {
    let f = na::SMatrix::<f64, 3, 3>::new(1.0, 0.1, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.95);
    let q = na::SMatrix::<f64, 3, 3>::new(0.02, 0.005, 0.0, 0.005, 0.01, 0.002, 0.0, 0.002, 0.03);
    let h = na::SMatrix::<f64, 2, 3>::new(1.0, 0.0, 0.0, 0.0, 1.0, 1.0);

    for r in [
        na::SMatrix::<f64, 2, 2>::new(0.1, 0.0, 0.0, 0.2),     // diagonal
        na::SMatrix::<f64, 2, 2>::new(0.1, 0.04, 0.04, 0.2),   // correlated
    ] {
        let mut kf = KalmanFilterNM::new(LinearUpdateNM::<3, 2>::new(
            LinearMeasurementNM::<3, 2>::new(h, r),
            LinearTransitionN::<3>::new(state3().ptr(), f, q)));
        let mut ud = UDKalmanFilterNM::new(LinearUDUpdateNM::<3, 2>::new(
            LinearMeasurementNM::<3, 2>::new(h, r),
            LinearTransitionN::<3>::new(state3().ptr(), f, q)));

        for t in [1, 2, 5, 6, 9] {
            let obs = Observation { value: na::SVector::<f64, 2>::new(0.3 * t as f64, -0.1 * t as f64), epoch: Epoch::new(t) };
            let a = kf.update(&obs);
            let b = ud.update(&obs);
            let (a, b) = (a.read().unwrap(), b.read().unwrap());
            assert!(close(&b.value, &a.value), "t={}: {} vs {}", t, b.value, a.value);
            assert!(close(&b.covariance(), &a.covariance()), "t={}: covariance", t);
            assert_eq!(a.epoch, b.epoch);
        }
    }
}

/// The UD EKF matches the EKF for nonlinear models and several measurements.
#[test]
fn test_ud_extended_matches_extended() {
    let transition = || NonLinearTransitionN::<2>::new(
        State::<na::Const<2>>::new(vec![2.0, 2.0], vec![vec![1.0, 0.2], vec![0.2, 1.0]], Epoch::new(0)).ptr(),
        Box::new(|s| na::SVector::<f64, 2>::new(s.value[0] + 0.1 * s.value[1].sin(), 0.98 * s.value[1])),
        Box::new(|s| na::SMatrix::<f64, 2, 2>::new(1.0, 0.1 * s.value[1].cos(), 0.0, 0.98)),
        na::SMatrix::<f64, 2, 2>::identity() * 0.01);
    // range and bearing
    let measurement = || NonLinearMeasurementNM::<2, 2>::new(
        Box::new(|s| na::SVector::<f64, 2>::new(s.value.norm(), s.value[1].atan2(s.value[0]))),
        Box::new(|s| {
            let (x, y) = (s.value[0], s.value[1]);
            let r2 = x * x + y * y;
            let r = r2.sqrt();
            na::SMatrix::<f64, 2, 2>::new(x / r, y / r, -y / r2, x / r2)
        }),
        na::SMatrix::<f64, 2, 2>::new(0.05, 0.0, 0.0, 0.01));

    let mut ekf = ExtendedKalmanFilterNM::new(NonLinearUpdateNM::<2, 2>::new(measurement(), transition()));
    let mut ud = UDExtendedKalmanFilterNM::new(NonLinearUDUpdateNM::<2, 2>::new(measurement(), transition()));

    for t in [1, 2, 4, 5] {
        let obs = Observation { value: na::SVector::<f64, 2>::new(3.0, 0.8), epoch: Epoch::new(t) };
        let a = ekf.update(&obs);
        let b = ud.update(&obs);
        let (a, b) = (a.read().unwrap(), b.read().unwrap());
        assert!(close(&b.value, &a.value), "t={}: {} vs {}", t, b.value, a.value);
        assert!(close(&b.covariance(), &a.covariance()), "t={}: covariance", t);
    }
}

/// Badly scaled problem: the factors keep P symmetric with non-negative D.
#[test]
fn test_ud_stays_positive_semi_definite() {
    let mut ud = UDKalmanFilterNM::new(LinearUDUpdateNM::<2, 1>::new(
        LinearMeasurementNM::<2, 1>::new(na::SMatrix::<f64, 1, 2>::new(1.0, 1.0), na::SMatrix::<f64, 1, 1>::new(1e-8)),
        LinearTransitionN::<2>::new(
            State::<na::Const<2>>::new(vec![0.0, 0.0], vec![vec![1e8, 0.0], vec![0.0, 1.0]], Epoch::new(0)).ptr(),
            na::SMatrix::<f64, 2, 2>::identity(), na::SMatrix::<f64, 2, 2>::zeros())));

    for t in 1..50 {
        let state = ud.update(&Observation { value: na::SVector::<f64, 1>::new(1.0), epoch: Epoch::new(t) });
        let p = state.read().unwrap().covariance();
        assert_eq!(p, p.transpose());
        assert!(ud.update.factors.d.iter().all(|d| *d >= 0.0), "t={}: D = {}", t, ud.update.factors.d);
    }
}

/// Setting the state from outside is picked up: the filter continues from it
/// exactly as a Kalman filter started there would.
#[test]
fn test_ud_refactorises_after_outside_change() {
    let f = na::SMatrix::<f64, 3, 3>::identity() * 1.1;
    let q = na::SMatrix::<f64, 3, 3>::identity() * 0.01;
    let h = na::SMatrix::<f64, 2, 3>::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    let r = na::SMatrix::<f64, 2, 2>::identity() * 0.1;

    let mut ud = UDKalmanFilterNM::new(LinearUDUpdateNM::<3, 2>::new(
        LinearMeasurementNM::<3, 2>::new(h, r), LinearTransitionN::<3>::new(state3().ptr(), f, q)));
    ud.update(&Observation { value: na::SVector::<f64, 2>::new(1.0, 1.0), epoch: Epoch::new(1) });

    // replace the state from outside
    let mut fresh = state3();
    fresh.epoch = Epoch::new(1);
    *ud.state.write().unwrap() = fresh.clone();

    let mut kf = KalmanFilterNM::new(LinearUpdateNM::<3, 2>::new(
        LinearMeasurementNM::<3, 2>::new(h, r), LinearTransitionN::<3>::new(fresh.ptr(), f, q)));

    let obs = Observation { value: na::SVector::<f64, 2>::new(0.5, -0.5), epoch: Epoch::new(3) };
    let a = kf.update(&obs);
    let b = ud.update(&obs);
    let (a, b) = (a.read().unwrap(), b.read().unwrap());
    assert!(close(&b.value, &a.value));
    assert!(close(&b.covariance(), &a.covariance()));
}

/// Runtime sized (`na::Dyn`, as used by Python) matches statically sized exactly.
#[test]
fn test_ud_dyn_matches_static() {
    let s = state3();
    let f = na::SMatrix::<f64, 3, 3>::new(1.0, 0.1, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.95);
    let q = na::SMatrix::<f64, 3, 3>::identity() * 0.02;
    let h = na::SMatrix::<f64, 2, 3>::new(1.0, 0.0, 0.0, 0.0, 1.0, 1.0);
    let r = na::SMatrix::<f64, 2, 2>::new(0.1, 0.04, 0.04, 0.2);
    let dynm = |m: &dyn Fn(usize, usize) -> f64, rows, cols| na::DMatrix::from_fn(rows, cols, |i, j| m(i, j));

    let mut fixed = UDKalmanFilterNM::new(LinearUDUpdateNM::<3, 2>::new(
        LinearMeasurementNM::<3, 2>::new(h, r), LinearTransitionN::<3>::new(s.ptr(), f, q)));
    let mut dynamic = UDKalmanFilterDyn::new(LinearUDUpdate::new(
        LinearMeasurement::new(dynm(&|i, j| h[(i, j)], 2, 3), dynm(&|i, j| r[(i, j)], 2, 2)),
        LinearTransition::new(
            State::<na::Dyn>::new(s.value.as_slice().to_vec(),
                (0..3).map(|i| (0..3).map(|j| s.covariance()[(i, j)]).collect()).collect(), Epoch::new(0)).ptr(),
            dynm(&|i, j| f[(i, j)], 3, 3), dynm(&|i, j| q[(i, j)], 3, 3))));

    for t in [1, 3, 4] {
        let z = [0.3 * t as f64, -0.2];
        let a = fixed.update(&Observation { value: na::SVector::<f64, 2>::from_row_slice(&z), epoch: Epoch::new(t) });
        let b = dynamic.update(&Observation { value: na::DVector::from_row_slice(&z), epoch: Epoch::new(t) });
        let (a, b) = (a.read().unwrap(), b.read().unwrap());
        assert_eq!(a.value.as_slice(), b.value.as_slice());
        assert_eq!(a.covariance().as_slice(), b.covariance().as_slice());
    }
}

/// Same-epoch predict does nothing; backward predict and a singular
/// innovation leave the state unchanged.
#[test]
fn test_ud_edge_cases() {
    let mut ud = UDKalmanFilterNM::new(LinearUDUpdateNM::<3, 2>::new(
        LinearMeasurementNM::<3, 2>::new(na::SMatrix::<f64, 2, 3>::zeros(), na::SMatrix::<f64, 2, 2>::zeros()),
        LinearTransitionN::<3>::new(state3().ptr(), na::SMatrix::<f64, 3, 3>::identity() * 2.0,
            na::SMatrix::<f64, 3, 3>::zeros())));
    let before = state3();

    ud.predict(&Epoch::new(0));
    ud.predict(&Epoch::new(-1));
    let after = ud.state();
    assert_eq!(after.read().unwrap().value, before.value);
    assert_eq!(after.read().unwrap().epoch, Epoch::new(0));

    // H = 0 and R = 0: the innovation variance is zero
    let state = ud.update(&Observation { value: na::SVector::<f64, 2>::new(1.0, 1.0), epoch: Epoch::new(0) });
    assert_eq!(state.read().unwrap().value, before.value);
}
