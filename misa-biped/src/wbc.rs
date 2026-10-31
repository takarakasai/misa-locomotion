//! The seam between the walker and its whole-body QP: what the walker asks
//! for each tick ([`WbcRequest`]), what it gets back ([`WbcSolution`]), and
//! the trait a stack of QP levels implements ([`WbcStack`]).
//!
//! Added on 2026-10-30 (S12 of doc/from_claude/20261029_walker_extraction_plan.md,
//! §2.6, plan A: one coarse trait). The walker computes the REFERENCES --
//! numbers: the CoM acceleration, the trunk's orientation target, the
//! momentum and arm-hold targets, the swing foot's target and gains, the
//! load split, the MPC's wrenches -- and the CONTACT REQUEST (which feet,
//! their load shares, rigid or managed, a moved CoP box, the anchor
//! events). Building the levels, ordering and weighting them, solving and
//! extracting the result are the stack's. The stack keeps its own state
//! across ticks: the solver's warm start, the last QP structure and the
//! contact anchors.
//!
//! [`crate::task_stack::TaskStack`] implements it with the level builders of
//! [`crate::tasks`] / [`crate::contact`] and `misa_wbc::Solver` (articara
//! re-exports it as `biped::wbc_stack`); another runner (another QP, a real
//! robot) can put its own stack behind the same request.
use crate::gait::Support;
use crate::robot::{RobotModel, RobotState};
use crate::sink::LogSink;
use nalgebra as na;

/// A change to a foot's contact anchor, in the order the walker decided it
/// (the lift-off / touchdown of the schedule, then a confirmed landing).
/// The stack applies them in this order before it builds the contact level.
#[derive(Clone, Debug)]
pub enum AnchorEvent {
    /// The foot leaves the contact set: forget its anchor.
    Release(usize),
    /// The foot has landed: anchor it here (sole position, rotation).
    Touchdown { side: usize, pos: na::Vector3<f64>, rot: na::Matrix3<f64> },
}

/// How a stance foot's contact is held and which rows of the swing foot's
/// pose are tracked: the level builders' own types (one copy since S13d).
pub use crate::tasks::{ContactMode, SwingAxes};

/// A foot's CoP box moved and shrunk: the sole point in the foot frame and
/// the box's half-extents (x, y).
pub type CopPatch = ([f64; 3], (f64, f64));

/// The contact model: the anchors' Baumgarte gains and leak, the sole in
/// the foot frame, the friction, the CoP box, the torsion coefficient, the
/// per-foot force ceiling and the control period. Fixed at start-up; the
/// walker reads the CoP box and the torsion coefficient too (the MPC plans
/// against the same contact), so it is built on the walker's side and handed
/// to the stack. The level builders know it as [`crate::tasks::ContactCfg`].
#[derive(Clone, Debug)]
pub struct ContactParams {
    pub kp_c: f64,
    pub kd_c: f64,
    pub anchor_leak: f64,
    pub anchor_leak_rot: f64,
    /// Sole plane in the foot link frame: `[centre_x, 0, -below_origin]`.
    pub sole_offset_local: [f64; 3],
    pub friction_mu: f64,
    /// CoP box half-extents, already scaled by the margin factor.
    pub cop_half: (f64, f64),
    pub mu_torsion: f64,
    pub f_max_per_foot: f64,
    /// Drop the contact patch (the CoP box and the friction cone) entirely,
    /// to test whether the CoP box is what is binding.
    pub no_patch: bool,
    pub dt: f64,
}

/// The trunk: its Jacobian and bias, the roll / pitch target and, when the
/// yaw is regulated, the yaw target.
pub struct TrunkRef<'a> {
    pub j: &'a na::DMatrix<f64>,
    pub djv: &'a na::DVector<f64>,
    pub rp_ref: [f64; 2],
    pub yaw: Option<f64>,
}

/// The centroidal angular-momentum task (MOM, arms_neutral): the CMM, its
/// bias, the momentum now, its reference and rate, the gain, the axes.
pub struct MomRef {
    pub cmm: na::DMatrix<f64>,
    pub bias: na::DVector<f64>,
    pub h: na::DVector<f64>,
    pub h_ref: [f64; 3],
    pub dh_ref: [f64; 3],
    pub kp: f64,
    pub axes: [bool; 3],
}

/// MOM_CMP's inputs (the CMM is the tick's, see [`MomCmp`]): the bias, the
/// momentum now, its reference (itself) and the asked rate.
pub struct CmpRef {
    pub bias: na::DVector<f64>,
    pub h: na::DVector<f64>,
    pub h_ref: [f64; 3],
    pub dh: [f64; 3],
}

/// MOM_CMP as the stack is handed it: the tick's CMM and the inputs.
pub struct MomCmp<'a> {
    pub cmm: &'a na::DMatrix<f64>,
    pub r: &'a CmpRef,
}

/// The arm hold's rows (actuated joint, velocity index) and its reference
/// (the seed, with a jab swept in).
pub struct ArmRef {
    pub arms: Vec<(usize, usize)>,
    pub q_ref: Vec<f64>,
}

/// The swing foot's task inputs: its Jacobian and bias, where it is and how
/// fast, the target position and velocity, the gains, the axes, the
/// orientation (measured, target, angular velocity, gains), and the
/// feed-forward acceleration (`None` for a foot put back on its footstep,
/// which has none).
pub struct SwingRef {
    pub jf: na::DMatrix<f64>,
    pub djv: na::DVector<f64>,
    pub pos: na::Vector3<f64>,
    pub vel: na::Vector3<f64>,
    pub tgt: na::Vector3<f64>,
    pub tgt_v: na::Vector3<f64>,
    pub kp_xy: f64,
    pub kp_z: f64,
    pub kd: f64,
    pub axes: SwingAxes,
    pub rot: Option<(na::Matrix3<f64>, na::Matrix3<f64>, na::Vector3<f64>, f64, f64)>,
    pub acc: Option<na::Vector3<f64>>,
}

/// The MPC's sole wrenches (sole-centre frame, `[m, f]`, per side) and how
/// they enter the nominal forces: `authority` replaces the load split's
/// nominal, otherwise it is blended in by `blend`.
pub struct MpcNominal<'a> {
    pub wrench: &'a [na::Vector6<f64>; 2],
    pub authority: bool,
    pub blend: f64,
}

/// What the walker asks of the stack in one tick.
pub struct WbcRequest<'a> {
    pub t: f64,
    pub tick: usize,
    pub robot: &'a RobotModel,
    pub state: &'a RobotState,
    /// The planned support (the QP_DIAG dump and the level debug print
    /// report it).
    pub support: Support,
    /// The feet the QP solves with as contacts: sides and model indices, in
    /// slot order.
    pub stance_sides: &'a [usize],
    pub stance: &'a [usize],
    /// The anchor changes decided this tick, in order.
    pub anchor_events: &'a [AnchorEvent],
    /// Per side: the load share (the force ceiling's factor), rigid or
    /// managed, and a moved CoP box (`None` = the full sole).
    pub load_share: [f64; 2],
    pub contact_mode: [ContactMode; 2],
    pub cop_patch: [Option<CopPatch>; 2],
    /// The CoM acceleration.
    pub a_com: na::Vector3<f64>,
    pub trunk: TrunkRef<'a>,
    /// The coupled level (CoM, momentum and the MPC's wrench in one).
    pub coupled: bool,
    pub momentum: Option<&'a MomRef>,
    pub mom_cmp: Option<MomCmp<'a>>,
    pub arm_hold: Option<&'a ArmRef>,
    /// The joint angles the posture and arm-hold levels hold against
    /// (harness order; today the plant's, §6.6 of the plan).
    pub joint_pos: &'a [f64],
    /// The torque regulariser's target and the nominal forces: the load
    /// split per slot and, when the MPC applies, its wrenches.
    pub tau_gravity: &'a na::DVector<f64>,
    pub shares: &'a [f64],
    pub mpc: Option<MpcNominal<'a>>,
    pub swing: Option<&'a SwingRef>,
    /// Running: the second airborne foot (same level as `swing`).
    pub swing2: Option<&'a SwingRef>,
    /// The measured vertical contact forces (diagnostics only).
    pub contact_fz: [f64; 2],
}

/// How long the stack spent [ns]: the solve, the physical check, the
/// diagnostic capture.
#[derive(Clone, Copy, Debug, Default)]
pub struct QpTimings {
    pub solve_ns: u128,
    pub check_ns: u128,
    pub capture_ns: u128,
}

/// What the stack returns.
pub struct WbcSolution {
    /// The solve (its status already `Degraded` when the gate rejected it)
    /// and the physical quantities extracted from it.
    pub sol: misa_wbc::Solution,
    pub extracted: misa_wbc::Extracted,
    /// The levels by name, in priority order.
    pub level_names: Vec<&'static str>,
    /// The nominal contact forces the regulariser pulled toward (world axes
    /// per stance slot).
    pub forces_nominal: na::DVector<f64>,
    /// Per stance slot, the solved contact wrench in the sole-centre frame
    /// (`[m, f]`).
    pub sole_wrench: Vec<na::DVector<f64>>,
    /// The left foot's contact acceleration reference (z) and anchor error
    /// (z), when it stands.
    pub acc_dbg: [f64; 2],
    /// The swing level's residual `A x - b` (unweighted), when there is one.
    pub swing_residual: Option<na::DVector<f64>>,
    /// The trunk's achieved roll / pitch acceleration.
    pub trunk_acc_rp: [f64; 2],
    pub timings: QpTimings,
    /// The physical rows' worst normalised violation (0 when not measured).
    pub physical_violation: f64,
    /// The solver was reset for a new QP structure.
    pub reset: bool,
    /// The validation gate rejected the solve.
    pub rejected: bool,
    /// The solve is not `Optimal` (after the gate).
    pub degraded: bool,
}

/// A stack of QP levels behind the walker.
pub trait WbcStack {
    /// Apply the request's anchor events, build the levels, solve, extract.
    /// The stack's own debug lines go to `out`, in the walker's order.
    fn solve(&mut self, req: &WbcRequest, out: &mut dyn LogSink) -> WbcSolution;

    /// A foot's contact anchor (position, rotation), as the last solve left
    /// it.
    fn anchor(&self, side: usize) -> Option<(na::Vector3<f64>, na::Matrix3<f64>)>;
}

/// Centre of pressure implied by a sole-frame wrench `w = [m(0..3); f(3..6)]`,
/// as `(x, y, fz)`. `fz <= 0` means the foot is unsupported, and the CoP is
/// undefined rather than zero. (Moved from articara's `biped::contact`, which
/// re-exports it.)
pub fn cop_from_sole_wrench(w: &na::DVector<f64>) -> Option<(f64, f64, f64)> {
    let fz = w[5];
    if fz.abs() < 1e-6 {
        None
    } else {
        Some((-w[1] / fz, w[0] / fz, fz))
    }
}

/// World-frame reference angular acceleration for the trunk level, built from
/// a rotation-vector error rather than from Euler components. (Moved from
/// articara's `biped::tasks`, which re-exports it: it is a reference the
/// walker computes, not a level.)
///
/// Feeding a ZYX roll/pitch error straight into the WORLD angular-acceleration
/// rows is only the same thing at zero heading. A quarter turn later the
/// body's roll axis points along world +y, so the roll correction comes out on
/// the wrong axis; past 90 deg it is also wrong-signed. Measured on kyo46rs:
/// every turn command fell after a fixed ACCUMULATED heading -- 120 deg at
/// wz=0.10, 104 deg at 0.20, 90 deg at 0.40 -- while tracking its commanded
/// rate to ~90% right up to the fall.
///
/// `e = log(R_des * R^T)` is the world-frame rotation carrying the body to
/// upright-at-`yaw_des`, so it is correct at any heading. When `yaw_ref` is
/// `None` the desired heading is the current one and the z component is zero,
/// which is what the caller drops.
pub fn trunk_ori_ref(
    r_body: &na::UnitQuaternion<f64>,
    yaw_ref: Option<f64>,
    omega_world: &[f64; 3],
    gains: TrunkGains,
) -> [f64; 3] {
    let (_, _, yaw_now) = r_body.euler_angles();
    let yaw_des = yaw_ref.unwrap_or(yaw_now);
    let r_des = na::UnitQuaternion::from_euler_angles(0.0, 0.0, yaw_des);
    let e = (r_des * r_body.inverse()).scaled_axis();
    let dead = |v: f64| {
        if v.abs() <= gains.deadband {
            0.0
        } else {
            v - gains.deadband * v.signum()
        }
    };
    [
        gains.sign * (gains.kp * dead(e[0]) - gains.kd * omega_world[0]),
        gains.sign * (gains.kp * dead(e[1]) - gains.kd * omega_world[1]),
        gains.kp_yaw * e[2] + gains.kd_yaw * (gains.wz_ref - omega_world[2]),
    ]
}

/// Gains for [`trunk_ori_ref`]. Roll/pitch and yaw keep separate gains because
/// they are tuned against different things -- tilt rejection versus tracking a
/// commanded turn.
#[derive(Clone, Copy, Debug)]
pub struct TrunkGains {
    pub kp: f64,
    pub kd: f64,
    pub deadband: f64,
    pub sign: f64,
    pub kp_yaw: f64,
    pub kd_yaw: f64,
    /// Commanded yaw RATE, fed forward into the damping term.
    pub wz_ref: f64,
}
