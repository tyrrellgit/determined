use determined::algorithms::KalmanFilter;
use determined::common::Algorithm;
use determined::state::State;
use determined::common::Epoch;

#[test]
fn test_kalman_filter() {
    let mut kf: KalmanFilter::<1, 1> = KalmanFilter::new();
    // make sure measurement mapping is set
    kf.h = determined::common::na::SMatrix::<f64, 1, 1>::identity();

    let epoch = Epoch { value: 0 };
    kf.predict(&epoch);

    let z: State<1, 1> = State::new(1.0, 1);
    kf.update(&z);

    let s = kf.state();
    println!("state = {:?}", s);
}