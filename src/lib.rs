//! Determined — small filtering primitives
//!
//! This crate contains compact implementations of static-size Kalman and
//! Extended Kalman filters using nalgebra's `SMatrix`/`SVector`. It's designed
//! for clarity and small embedded-friendly usage.
//!
//! Run `cargo doc --open` to generate API documentation locally.
//!
//! Main modules:
//! - `common` — core traits and aliases (e.g. `Algorithm`).
//! - `algorithms` — Kalman and EKF implementations.
//! - `state` — `State<R,C>` wrapper around nalgebra matrices.
//! - `measurement` — measurement/observation helpers.
//! - `models` — traits for `StateTransition` and `MeasurementModel` used by EKF.

pub mod common;
pub mod composition;
pub mod algorithms;
pub mod measurement;
pub mod state;
pub mod models;
pub mod filter;
pub mod python;

mod linalg;

#[cfg(test)]
mod tests;