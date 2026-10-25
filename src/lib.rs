//! Determined — small filtering primitives
//!
//! This crate contains compact implementations of Kalman and Extended Kalman
//! filters on nalgebra matrices. The models are generic over the dimension
//! type: statically sized (`na::Const<N>`, via the `...N`/`...NM` aliases such
//! as `KalmanFilterNM`) for allocation-free embedded use, or runtime sized
//! (`na::Dyn`, e.g. `KalmanFilterDyn`), which the Python bindings wrap.
//!
//! Run `cargo doc --open` to generate API documentation locally.
//!
//! Main modules:
//! - `common` — core traits and aliases (e.g. `Algorithm`).
//! - `algorithms` — Kalman and EKF implementations.
//! - `state` — `State<R,C>` wrapper around nalgebra matrices.
//! - `measurement` — measurement/observation helpers.
//! - `models` — traits for `StateTransition` and `MeasurementModel` used by EKF.

pub mod algorithms;
pub mod common;
pub mod epoch;
pub mod filter;
pub mod measurement;
pub mod models;
pub mod python;
pub mod state;

#[cfg(test)]
mod tests;
