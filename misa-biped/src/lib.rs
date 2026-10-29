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

pub mod profile;
pub mod gait;
pub mod dcm;
pub mod preview;
pub mod contact_plan;
pub mod reach;
pub mod terrain;
pub mod height;
pub mod plan;

pub use profile::Profile;
