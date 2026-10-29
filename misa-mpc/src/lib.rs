//! Model-predictive control for legged robots, on misa-wbc's QP layer.
//!
//! - [`wrench_mpc`] -- single-rigid-body MPC over contact wrenches, with the
//!   next landing as a decision variable. Robot- and simulator-agnostic: it
//!   takes the centroidal state and the contact schedule as plain data.

pub mod wrench_mpc;
