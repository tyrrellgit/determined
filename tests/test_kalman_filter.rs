use determined::algorithms::KalmanFilter;
use determined::filter::Filter;
use determined::state::State;
use determined::common::Epoch;

#[test]
fn test_kalman_filter() {
    let mut kf: KalmanFilter::<1, 1> = KalmanFilter::default();
    // make sure measurement mapping is set
    kf.h = determined::common::na::SMatrix::<f64, 1, 1>::identity();

    let epoch = Epoch { value: 0 };
    kf.predict(&epoch);

    let z = State::new(vec![1.0], 1);
    kf.update(&z);

    let s = kf.state();
    println!("state = {:?}", s);
}