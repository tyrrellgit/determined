#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update() {
        let x = State::new(1.0, 0);
        let o = Observation::new(2.0, 0);
        let m = Measurement {
            state: a,
            observation: b,
            covariance: na::Matrix::<f64, 2, 2>::identity(),
            jacobian: na::Matrix::<f64, 2, 2>::identity(),
        };
        
        let filter = KalmanFilter {
            P: 1.0,
            Q: 1.0,
            R: 1.0,
            x: 0.0, 
            K: 0.0,
            state: State::new(0.0, 0),
            H: 1.0,
            F: 1.0,
        };
        
        filter.update(m);
    }
}