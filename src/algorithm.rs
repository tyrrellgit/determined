trait Algorithm {
    fn new() -> Self;
    fn predict(&mut self, epoch: Epoch) -> State;
    fn update(&mut self, observation: Observation) -> State;
    fn reset(&mut self);
};

struct KalmanFilter< const N: usize, const M: usize >{
    P : f64, // Process noise covariance
    Q : f64, // Measurement noise covariance
    R : f64, // State covariance matrix
    K : f64, // Kalman gain
    F : f64, // State transition matrix
    H : f64, // Measurement matrix
    state: State<N, M>, // state vector
};


impl Algorithm for KalmanFilter {
    fn new() -> Self {
        KalmanFilter {
            P: 1.0,
            Q: 1.0,
            R: 1.0,
            x: 0.0,
            K: 0.0,
            state: State::new(0.0, 0),
            H: 1.0,
            F: 1.0,
        };
    }

    fn predict(&mut self, input: f64) -> f64 {
        // Prediction logic
        self.x = self.F * self.x + input;
        self.P = self.F * self.P * self.F + self.Q;
        self.x
    }

    fn update(&mut self, measurement: f64) -> f64 {
        // Update logic
        self.K = self.P * self.H / (self.H * self.P * self.H + self.R);
        self.x = self.x + self.K * (measurement - self.H * self.x);
        self.P = (1.0 - self.K * self.H) * self.P;
        self.x
    }

    fn reset(&mut self) {
        // Reset logic
        self.x = 0.0;
        self.P = 1.0;
    }
};