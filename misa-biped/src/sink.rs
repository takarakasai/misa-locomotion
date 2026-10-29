//! Where the walker's output goes: its log lines ([`LogSink`]) and its
//! opt-in diagnostic records ([`WalkerDiag`]).
//!
//! The walker prints nothing and opens no file itself. Each line it would
//! have printed goes to the sink at the moment it is decided, so a harness
//! that writes `line` to stdout and `err_line` to stderr reproduces the old
//! log byte for byte (the two streams are kept apart because a log taken with
//! `2>&1` interleaves them in the order they were written). The diagnostics
//! the control used to write into CSV / JSON files (the CoM preview's plan,
//! the wrench MPC's input, solve and plan) are handed to a [`WalkerDiag`]
//! with the values they were written from; the harness formats them.
//! (S13 of doc/from_claude/20261029_walker_extraction_plan.md, §2.4.)

use crate::preview::{ComPreview, PreviewSample};
use misa_mpc::wrench_mpc;
use nalgebra as na;

/// The walker's log output, one line at a time (no trailing newline).
pub trait LogSink {
    /// A line of the standard log (stdout).
    fn line(&mut self, s: &str);
    /// A line of the error log (stderr).
    fn err_line(&mut self, s: &str);
}

/// A sink that drops everything.
pub struct NullSink;

impl LogSink for NullSink {
    fn line(&mut self, _: &str) {}
    fn err_line(&mut self, _: &str) {}
}

/// `outln!(out, "fmt", args..)`: one line to a [`LogSink`]'s standard log.
#[macro_export]
macro_rules! outln {
    ($out:expr, $($arg:tt)*) => {
        $crate::sink::LogSink::line(&mut *$out, &format!($($arg)*))
    };
}

/// `errln!(out, "fmt", args..)`: one line to a [`LogSink`]'s error log.
#[macro_export]
macro_rules! errln {
    ($out:expr, $($arg:tt)*) => {
        $crate::sink::LogSink::err_line(&mut *$out, &format!($($arg)*))
    };
}

/// The walker's opt-in diagnostic records. Every method has an empty
/// default, so a harness implements only what it records; the `wants_*` /
/// `keeps_*` questions let the walker skip the work (and the timing) of a
/// record nobody keeps, exactly as the old `Option<file>.is_some()` did.
#[allow(unused_variables)]
pub trait WalkerDiag {
    /// An accepted CoM-preview solve (PREVIEW_TRAJ_CSV): its samples and
    /// solution, at time `t`.
    fn preview_solved(&mut self, t: f64, samples: &[PreviewSample], solution: &ComPreview) {}

    /// Every CoM-preview solve (COM_PREVIEW_RUNTIME_CSV): how long it took,
    /// whether it was accepted, the terminal error then held, and the
    /// horizon's sample count.
    fn preview_runtime(&mut self, t: f64, solve_ns: u128, accepted: bool, terminal_error: f64, horizon: usize) {}

    /// Whether [`WalkerDiag::mpc_input`] records anything (MPC_DIAG_DIR).
    /// The walker times the record only when it does.
    fn wants_mpc_input(&self) -> bool {
        false
    }

    /// A wrench-MPC solve's input and configuration, before the solve
    /// (MPC_DIAG_DIR). `ctrl_dt` is the control period, `update_dt` the
    /// MPC's update period.
    fn mpc_input(
        &mut self,
        t: f64,
        ctrl_dt: f64,
        update_dt: f64,
        input: &wrench_mpc::Input,
        config: &wrench_mpc::Config,
    ) {
    }

    /// Every wrench-MPC solve (MPC_RUNTIME_CSV): the stance mask (bit per
    /// side), the configuration, the update period, how long it took,
    /// whether it returned a solution, and the age of the held one before it
    /// (-1 = none).
    #[allow(clippy::too_many_arguments)]
    fn mpc_runtime(
        &mut self,
        t: f64,
        stance_mask: u8,
        config: &wrench_mpc::Config,
        update_dt: f64,
        solve_ns: u128,
        accepted: bool,
        previous_age: f64,
    ) {
    }

    /// Whether the walker keeps the MPC's contact frames with its held
    /// answer (`TickRecord::wrench_mpc_frames`; QP_WRENCH_CSV reads them).
    fn keeps_mpc_frames(&self) -> bool {
        false
    }

    /// A wrench-MPC solution (MPC_TRAJ_CSV): the CoM it started from, its
    /// input and solution, the net pressure point of its first node, the
    /// landing variable it was given, and its per-side wrenches.
    #[allow(clippy::too_many_arguments)]
    fn mpc_solved(
        &mut self,
        t: f64,
        com: &na::Vector3<f64>,
        input: &wrench_mpc::Input,
        solution: &wrench_mpc::Solution,
        net_cop: [f64; 2],
        landing: Option<&wrench_mpc::Landing>,
        by_side: &[wrench_mpc::Wrench; 2],
    ) {
    }
}

/// A [`WalkerDiag`] that records nothing.
pub struct NoDiag;

impl WalkerDiag for NoDiag {}
