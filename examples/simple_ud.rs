//! UD Kalman filter tracking a target moving at constant velocity from
//! correlated position measurements, alongside the standard Kalman filter.
//! The UD filter propagates the factors P = U D U^T, keeping P symmetric
//! positive semi-definite by construction, and gives the same estimates.

use determined::measurement::Observation;
use determined::state::State;
use determined::filter::Filter;
use determined::algorithms::{ KalmanFilterNM, UDKalmanFilterNM };
use determined::models::*;
use determined::common::na as na;
use determined::epoch::Epoch;

fn main() {
    // state x = (px, py, vx, vy), one epoch per second, position measured
    let f = na::SMatrix::<f64, 4, 4>::new(
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0);
    let q = na::SMatrix::<f64, 4, 4>::from_diagonal(&na::SVector::<f64, 4>::new(0.01, 0.01, 0.001, 0.001));
    let h = na::SMatrix::<f64, 2, 4>::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0);
    // correlated measurement noise: decorrelated internally by the UD filter
    let r = na::SMatrix::<f64, 2, 2>::new(0.04, 0.03, 0.03, 0.09);

    let initial = || State::<na::Const<4>>::new(
        vec![0.0; 4],
        vec![vec![10.0, 0.0, 0.0, 0.0], vec![0.0, 10.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]],
        Epoch::new(0)).ptr();

    let mut ud = UDKalmanFilterNM::new(LinearUDUpdateNM::<4, 2>::new(
        LinearMeasurementNM::<4, 2>::new(h, r), LinearTransitionN::<4>::new(initial(), f, q)));
    let mut kf = KalmanFilterNM::new(LinearUpdateNM::<4, 2>::new(
        LinearMeasurementNM::<4, 2>::new(h, r), LinearTransitionN::<4>::new(initial(), f, q)));

    let mut truth = na::SVector::<f64, 4>::new(0.0, 0.0, 1.0, 0.5);
    for t in 1..=30 {
        truth = f * truth;
        // deterministic stand-in for sensor noise
        let noise = na::SVector::<f64, 2>::new(0.2 * (1.3 * t as f64).sin(), 0.3 * (0.7 * t as f64).cos());
        let obs = Observation { value: h * truth + noise, epoch: Epoch::new(t) };

        let a = ud.update(&obs);
        let b = kf.update(&obs);
        if t % 10 == 0 {
            let (a, b) = (a.read().unwrap(), b.read().unwrap());
            println!("t={:2} estimate={:.3?} truth={:.3?}", t, a.value.as_slice(), truth.as_slice());
            println!("      |UD - KF| state={:.1e} covariance={:.1e}",
                (a.value - b.value).abs().max(), (a.covariance() - b.covariance()).abs().max());
        }
    }

    let d: Vec<String> = ud.update.factors.d.iter().map(|d| format!("{:.3e}", d)).collect();
    println!("UD factors: D = [{}]", d.join(", "));
}
