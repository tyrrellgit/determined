use crate::state::State;
use crate::measurement::Measurement;
use crate::filter::Filter;
use crate::algorithms::KalmanFilter;
use crate::common::na as na;
use crate::common::Epoch;

#[test]
fn test_kalman_update_scalar() {
    // Small 1D Kalman filter: N=1, M=1
    // bring Algorithm trait into scope so we can call the trait-associated `new`
    let mut filter = <KalmanFilter::<1, 1> as Filter>::new();

    // initial state x = 0
    assert_eq!(filter.x.value[(0, 0)], 0.0);

    // measurement z = 1
    let meas = Measurement::<1, 1, 2> {
        state: State::<1, 1>::new(0.0, 0),
        observation: State::<1, 1>::new(1.0, 0),
        covariance: na::SMatrix::<f64, 2, 2>::identity(),
        jacobian: na::SMatrix::<f64, 1, 1>::identity(),
    };

    // predict (no-op since F=I)
    let _pred = filter.predict(&Epoch { value: 1 });

    // update with measurement; update returns a reference to the state
    // set h to identity for scalar case so measurement maps directly
    filter.h = na::SMatrix::<f64, 1, 1>::identity();
    let _ = filter.update(&meas.observation);

    // With P=I, R=I, H=I we expect K = 1/2 and x -> 0.5
    let x_val = filter.x.value[(0, 0)];
    assert!((x_val - 0.5).abs() < 1e-12, "x after update = {}", x_val);
}

#[test]
fn test_kalman_update_2d() {
    // N=2, M=1. Observe only the first state element.
    let mut filter = <KalmanFilter::<2, 1> as Filter>::new();
    // set H = [1 0]
    filter.h = na::SMatrix::<f64, 1, 2>::from_row_slice(&[1.0, 0.0]);

    // measurement z = 2
    let meas = Measurement::<2, 1, 3>::new(State::<2, 1>::new(0.0, 0), State::<1, 1>::new(2.0, 0));

    let _ = filter.predict(&Epoch { value: 1 });
    let _ = filter.update(&meas.observation);

    // Expect first state to be 1.0 (K = [0.5, 0]^T, y = 2)
    let x0 = filter.x.value[(0, 0)];
    let x1 = filter.x.value[(1, 0)];
    assert!((x0 - 1.0).abs() < 1e-12, "x0 after update = {}", x0);
    assert!((x1 - 0.0).abs() < 1e-12, "x1 after update = {}", x1);
}

#[test]
#[should_panic(expected = "innovation covariance S is singular")]
fn test_kalman_singular_innovation() {
    // Construct filter where p and r are zero so S == 0 and inverse fails.
    let mut filter = <KalmanFilter::<1, 1> as Filter>::new();
    filter.p = na::SMatrix::<f64, 1, 1>::zeros();
    filter.r = na::SMatrix::<f64, 1, 1>::zeros();
    filter.h = na::SMatrix::<f64, 1, 1>::identity();

    let meas = Measurement::<1, 1, 2>::new(State::<1, 1>::new(0.0, 0), State::<1, 1>::new(1.0, 0));
    // This should panic when update tries to invert S
    let _ = filter.update(&meas.observation);
}