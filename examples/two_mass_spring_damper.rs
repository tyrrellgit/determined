/// Example: Simulate the two-mass-spring-damper Kalman filter.
/// 
/// This example demonstrates the Kalman filter applied to the two-mass-spring-damper
/// system from: https://solmaz.eng.uci.edu/Teaching/MAE195_code/Kalman_filter_Two_Mass_spring_damper.html
/// 
/// Run with: cargo run --example two_mass_spring_damper --release

use statrs::distribution::MultivariateNormal;
use rand;
use rand_distr::Distribution;

use determined::algorithms::{ KalmanFilterNM };
use determined::models::{ LinearTransition, LinearMeasurement, LinearUpdate };
use determined::filter::Filter;
use determined::state::State;
use determined::measurement::Observation;
use determined::epoch::Epoch;
use determined::common::na;

fn setup_two_mass_spring_damper_filter() -> (
    KalmanFilterNM<4, 2>,
    impl Fn(&State<na::Const<4>>) -> State<na::Const<4>>,
    impl Fn(&State<na::Const<4>>) -> Observation<na::Const<2>>) {
    // Physical parameters from UCI example
    let m = 0.1;              // mass 2 (kg)
    let m_large = 1.0;        // mass 1 (kg)
    let k = 0.091;            // spring constant (N/m)
    let b = 0.0036;           // damping coefficient (N·s/m)
    
    // Continuous-time state transition matrix A
    let a = na::SMatrix::<f64, 4, 4>::new(
        0.0,              1.0,                    0.0,              0.0,
        -k/m,             -b/m,                   k/m,              b/m,
        0.0,              0.0,                    0.0,              1.0,
        k/m_large,        b/m_large,              -k/m_large,      -b/m_large,
    );

    // Discretization parameters
    let delta_t = 0.1;
    let identity = na::SMatrix::<f64, 4, 4>::identity();
    
    // Discrete state transition matrix: F = I + Δt·A
    let f = identity + delta_t * a;
    let f_model = f.clone(); // for model closure

    // Measurement matrix H: observe only positions
    let h = na::SMatrix::<f64, 2, 4>::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    );
    let h_model = h.clone(); // for measurement closure
    
    // Process noise covariance Q
    let sigma_q = na::SMatrix::<f64, 4, 4>::from_diagonal(&na::SVector::from_row_slice(&[
        0.3 * delta_t,
        0.05 * delta_t,
        0.3 * delta_t,
        0.05 * delta_t,
    ]));
    let q = &sigma_q * sigma_q.transpose();
    
    // Measurement noise covariance R
    let sigma_r = na::SMatrix::<f64, 2, 2>::from_diagonal(&na::SVector::from_row_slice(&[0.1, 0.1]));
    let r = &sigma_r * sigma_r.transpose();

    let x_hat_init = na::SVector::<f64, 4>::new(-1.0, 0.0, -0.5, 0.0);

    let x = State{
            value: x_hat_init.into(),
            covariance: Some(identity * 2.0),
            epoch: Epoch::new(0)
        }.ptr();

    let transition = LinearTransition::<4>::new(
        x.clone(),
        f,
        q
    );

    let measurement = LinearMeasurement::<4, 2>::new(
        h,
        r
    );

    // build algorithm update model
    let update = LinearUpdate::<4, 2>::new(
        measurement,
        transition
    );
    
    // build kalman filter with give update model
    let kf =
        KalmanFilterNM::new(
            update,
        );

    // function closure for system dynamics
    let model = move | state: &State<na::Const<4>> | -> State<na::Const<4>> {
        let new_value = f_model * state.value;
        State{
            value: new_value,
            covariance: None,
            epoch: Epoch::new(state.epoch.value + 1),
        }
    };

    // function closure for measurement model
    let h = move | state: &State<na::Const<4>> | -> Observation<na::Const<2>> {
        let measurement_value = h_model * &state.value;
        Observation{
            value: measurement_value,
            epoch: Epoch::new(state.epoch.value),
        }
    };

    ( kf, model, h)
}

fn main() {
    let (mut kf, model, h) = setup_two_mass_spring_damper_filter();

    let mut x_true = State {
        value: na::SVector::<f64, 4>::new(1.0, 0.0, 0.0, 0.0),
        covariance: None,
        epoch: Epoch { value: 0 },
    };

    let num_steps = 100 as i64;
    let epoch = Epoch { value: 0 };
    let noise_std_dev = 0.05;
    
    let initial_cov_trace = kf.state().read().unwrap().covariance().trace();
    let mut max_error_x1: f64 = 0.0;
    let mut max_error_x2: f64 = 0.0;

    let mean = na::SMatrix::<f64, 2, 1>::from_column_slice(&[0.0, 0.0]);
    let cov = na::SMatrix::<f64, 2, 2>::from_diagonal(
        &na::SVector::from_row_slice(
            &[noise_std_dev * noise_std_dev, noise_std_dev * noise_std_dev]
        ));

    let mvn = MultivariateNormal::new_from_nalgebra(mean, cov).unwrap();
    let rng = &mut rand::thread_rng();
    let mut sampler = mvn.sample_iter(rng);

    for step in 0..num_steps {
        // Predict
        kf.predict(&epoch);
        
        // Propagate true system
        x_true = model(&x_true);
        
        // Simulate measurement with noise
        let noise_matrix = sampler.next().unwrap();
        let measurement_value = h(&x_true).value + noise_matrix;
        let observation = Observation{
            value: measurement_value,
            epoch: Epoch::new(step as i64),
        };

        // Update filter
        kf.update(&observation);
        
        // Track errors
        let x_est = kf.state().read().unwrap().value;
        max_error_x1 = max_error_x1.max((x_est[0] - x_true.value[0]).abs());
        max_error_x2 = max_error_x2.max((x_est[2] - x_true.value[2]).abs());
    }
    
    let final_cov_trace = kf.state().read().unwrap().covariance().trace();
    let uncertainty_reduction = (1.0 - final_cov_trace / initial_cov_trace) * 100.0;
    
    println!("=== Two-Mass-Spring-Damper Kalman Filter ===");
    println!("Simulation: {} steps ({:.1} seconds)", num_steps, (num_steps - 1) as f64 * 0.1);
    println!("\nResults:");
    println!("  Max error (x1): {:.6} m", max_error_x1);
    println!("  Max error (x2): {:.6} m", max_error_x2);
    println!("  Initial uncertainty (trace P): {:.6}", initial_cov_trace);
    println!("  Final uncertainty (trace P):   {:.6}", final_cov_trace);
    println!("  Uncertainty reduction: {:.2}%", uncertainty_reduction);
}
