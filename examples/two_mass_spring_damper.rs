/// Example: Simulate the two-mass-spring-damper Kalman filter.
/// 
/// This example demonstrates the Kalman filter applied to the two-mass-spring-damper
/// system from: https://solmaz.eng.uci.edu/Teaching/MAE195_code/Kalman_filter_Two_Mass_spring_damper.html
/// 
/// Run with: cargo run --example two_mass_spring_damper --release

use determined::algorithms::{ KalmanFilterNM, KalmanFilterTUH};
use determined::models::{ LinearTransition, LinearMeasurement, LinearUpdate };
use determined::filter::Filter;
use determined::state::State;
use determined::measurement::Observation;
use determined::common::Epoch;
use determined::common::na as na;

fn setup_two_mass_spring_damper_filter() -> KalmanFilterNM<4, 2> {
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

    // Measurement matrix H: observe only positions
    let h = na::SMatrix::<f64, 2, 4>::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    );
    
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

    //TODO: this cloning needs to stop...
    let update = LinearUpdate::<4, 2>::new(
        x.clone(),
        measurement.clone(),
        transition.clone()
    );
    
    // TODO: simplify this constructor
    let kf =
        KalmanFilterTUH::new(
            x,
            transition,
            measurement,
            update,
        );
    kf
}

fn main() {
    let mut kf = setup_two_mass_spring_damper_filter();
    
    let h = kf.measurement.h;
    let mut x_true = na::SVector::<f64, 4>::new(1.0, 0.0, 0.0, 0.0);
    
    let num_steps = 100;
    let epoch = Epoch { value: 0 };
    let noise_std_dev = 0.05;
    
    let initial_cov_trace = kf.state().borrow().covariance().trace();
    let mut max_error_x1: f64 = 0.0;
    let mut max_error_x2: f64 = 0.0;
    
    for step in 0..num_steps {
        // Predict
        kf.predict(&epoch);
        
        // Propagate true system
        x_true = kf.transition.f * x_true;
        
        // Simulate measurement with noise
        let noise_matrix = na::SMatrix::<f64, 2, 1>::from_column_slice(
            &[noise_std_dev, noise_std_dev]
        );
        let measurement_value = h * &x_true + noise_matrix;
        let observation = Observation{
            value: measurement_value,
            epoch: Epoch::new(step as i64),
        };

        // Update filter
        kf.update(&observation);
        
        // Track errors
        let x_est = kf.state().borrow().value;
        max_error_x1 = max_error_x1.max((x_est[0] - x_true[0]).abs());
        max_error_x2 = max_error_x2.max((x_est[2] - x_true[2]).abs());
    }
    
    let final_cov_trace = kf.state().borrow().covariance().trace();
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
