//! Biped locomotion planning, independent of any simulator.
//!
//! Lifted out of articara's `biped` module so a second driver (a real-robot
//! runner, another simulator) can use the same planners. The parts that need
//! MuJoCo or articara's model editor -- plant bring-up, the plant write, the
//! trajectory CSV -- stay in articara.
//!
//! - [`profile`]      -- per-machine constants (kyo46rs, Unitree G1).
//! - [`gait`]         -- support schedule, footstep plan, swing-foot trajectory.
//! - [`dcm`]          -- LIPM / divergent-component-of-motion balance reference.
//! - [`preview`]      -- ZMP preview control over the DCM reference.
//! - [`contact_plan`] -- when the current step ends and where it lands.
//! - [`reach`]        -- leg-reach limits on the landing.
//! - [`terrain`]      -- the known-terrain description (stairs, slopes).
//! - [`height`]       -- CoM height reference over that terrain.
//! - [`plan`]         -- the walker's plan in one place (schedule, footsteps,
//!   DCM, height, landing clock) and its DCM rebuild.
//! - [`recovery`]     -- push-recovery references (trajopt) and their world frame.
//! - [`config`]       -- every control knob of the walker, with the code's
//!   defaults (the harness maps its environment onto it).
//! - [`kv`]           -- the knobs by name: a key/value lookup -> [`config::Config`].
//! - [`robot`]        -- the rigid-body model the walker computes with, the
//!   state it assembles from an estimate (M, h, FK, CoM Jacobian), and the
//!   per-tick observation.
//! - [`estimator`]    -- sensor reading -> state estimate (velocity fusion with
//!   an accelerometer, low-pass filters, the joint-rate clamp).
//! - [`command`]      -- what is commanded when the WBC degrades (the policy,
//!   its tally, the gravity + posture fallback).
//! - [`wbc`]          -- the seam to the whole-body QP: the request the walker
//!   makes each tick, the solution, and the `WbcStack` trait a stack of QP
//!   levels implements.
//! - [`contact`]      -- contact anchors, contact Jacobians, the sole wrench map.
//! - [`tasks`]        -- one builder per priority level of the hierarchical QP.
//! - [`task_stack`]   -- the `TaskStack`: those levels, ordered, weighted and
//!   solved behind `WbcStack` (articara's `biped::wbc_stack`).
//! - [`sink`]         -- where the walker's output goes: its log lines
//!   (`LogSink`) and its opt-in diagnostic records (`WalkerDiag`).
//! - [`walker`]       -- the walker itself: the plan made at start-up, the
//!   tick's control (`Walker::step`) and its end (`Walker::end_tick`).

pub mod profile;
pub mod gait;
pub mod dcm;
pub mod preview;
pub mod contact_plan;
pub mod reach;
pub mod terrain;
pub mod height;
pub mod plan;
pub mod recovery;
pub mod config;
pub mod kv;
pub mod robot;
pub mod wbc;
pub mod contact;
pub mod tasks;
pub mod task_stack;
pub mod sink;
pub mod estimator;
pub mod command;
pub mod walker;

pub use profile::Profile;
