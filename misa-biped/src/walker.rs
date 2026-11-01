//! The walker: the G1 / kyo46rs biped's whole control loop behind one
//! object (S13 of doc/from_claude/20261029_walker_extraction_plan.md).
//!
//! [`WalkerInit::new`] makes the plan from the settled pose (the footsteps,
//! the schedule, the speed schedule, the staircase's footfalls, COM_PREPLAN,
//! the DCM plan and the landing clock) and initialises the controller's
//! state; [`WalkerInit::with_stack`] attaches the WBC stack (built by the
//! harness from [`WalkerInit::contact_params`]) and gives the [`Walker`].
//! Each tick the harness calls [`Walker::step`] with the [`Observation`]
//! (the state estimate, from which the walker assembles the dynamics and
//! kinematics itself; phases 1-12: the landing clock, the support and contacts, the DCM
//! reference and its push responses, the commanded ZMP, the CoM preview, the
//! height, the wrench MPC, the levels' references, the stack's solve and the
//! solve's diagnostics) and, after its own scenario (the push) has fired,
//! [`Walker::end_tick`] (phase 13). Lines go to a [`LogSink`] as they are
//! decided and opt-in records to a [`WalkerDiag`]; the walker reads no
//! environment variable and opens no file.
//!
//! The phase functions themselves are in `phases.rs`, moved verbatim from
//! the example's control loop (S11); the runs are byte-identical to it.
use crate::config::Config;
use crate::dcm::G;
use crate::gait::{ContactCorrection, FootstepPlan, Footsteps, GaitParams, GaitPlan, Support};
use crate::outln;
use crate::plan::Plan;
use crate::recovery::{RefLib, RefMode};
use crate::robot::{Observation, RobotModel, RobotState};
use crate::sink::{LogSink, WalkerDiag};
use crate::terrain::Stairs;
use crate::wbc::{AnchorEvent, ContactParams, WbcSolution, WbcStack};
use nalgebra as na;

mod phases;
use phases::*;

/// The controller's state carried from tick to tick (the planned
/// schedule, footsteps and DCM aside, which are the plan). Each field is the
/// `let` of the same name before the loop, which gives its initial value;
/// gathered here so what the controller remembers is one thing.
pub(crate) struct CtlState {
    /// Speed-scheduled CoM drop, slewed.
    pub vx2_z_track: crate::height::SlewedDrop,
    /// PUSH_Z_DROP holds the lowered CoM until this time.
    pub push_z_drop_until: f64,
    /// Measured-contact correction of the planned contact set.
    pub correction: crate::gait::ContactCorrection,
    /// Terrain height under the support, slewed.
    pub terrain_track: f64,
    /// contact_plan: the landing shift applied to `cp_slice` (the slice
    /// whose end it has retimed is the plan's claim, `SliceClaims::end_time`).
    pub cp_applied: na::Vector2<f64>,
    pub cp_slice: Option<usize>,
    /// Contact admission: the last scheduled admit, the sole height measured
    /// at it, and the double support each foot is certified for.
    pub last_scheduled_admit: [Option<usize>; 2],
    pub admit_measured_z: [Option<f64>; 2],
    pub confirmed_admit: [Option<usize>; 2],
    /// Push-recovery reference: the one picked, the cooldown, the lateral
    /// quiet time, and the engaged reference's frame.
    pub ref_lib: Option<RefLib>,
    pub ref_cooldown_until: f64,
    pub ref_lat_quiet_t: f64,
    pub ref_mode: Option<RefMode>,
    /// PUSH_REPLAN: (slice, original footsteps, original DCM plan, original
    /// landing, last applied landing) and the hold shift (the slice whose
    /// landing it owns is the plan's claim, `SliceClaims::landing`).
    pub push_replan_state: Option<(
        usize,
        crate::gait::FootstepPlan,
        crate::dcm::DcmPlan,
        na::Vector2<f64>,
        Option<na::Vector2<f64>>,
    )>,
    pub push_replan_shift: f64,
    /// (slice, inward is +y) latched when the MPC crossing gate opens.
    pub mpc_fs_cross_latched: Option<(usize, bool)>,
    /// CoM preview: the reference ZMP, the predicted CoM and velocity, and
    /// the time of the next solve.
    pub preview_p: na::Vector2<f64>,
    pub preview_c: na::Vector2<f64>,
    pub preview_v: na::Vector2<f64>,
    pub preview_next: f64,
    /// Support of the previous tick (anchors are released and set on a change).
    pub prev_support: crate::gait::Support,
    /// Where the swing sole left the ground.
    pub swing_lift_off: Option<na::Vector3<f64>>,
    /// Running: where and when each foot last left the ground (its own
    /// swing spans flight + the other foot's stance + flight).
    pub run_lift_off: [Option<(na::Vector3<f64>, f64)>; 2],
    /// Heel strike / toe-off: the edge each foot is rolling on, if any.
    pub edge: [Option<EdgeState>; 2],
    /// Running: the CoM velocity filtered for the foothold correction.
    pub run_vel_ema: Option<na::Vector2<f64>>,
    /// Running: (flight slice, its original end) while a late touchdown
    /// holds it open.
    pub run_flight_end: Option<(usize, f64)>,
    /// Load-change rate per foot (the managed-contact trigger).
    pub fz_prev: [f64; 2],
    pub fz_rate_ema: [f64; 2],
    pub managed_until: [f64; 2],
    /// Last commanded horizontal CoM acceleration (COM_XY_JERK_MAX).
    pub previous_a_xy: Option<na::Vector2<f64>>,
    /// Wrench MPC: the time of the next solve and the held answer.
    pub wrench_mpc_next: f64,
    pub wrench_mpc_command: Option<[misa_mpc::wrench_mpc::Wrench; 2]>,
    /// MPC_FOOTSTEP: the slice whose landing is being moved, how much of the
    /// MPC's offset is applied, the (first, last) slice it is armed for, and
    /// the last total.
    pub mpc_fs_slice: Option<usize>,
    pub mpc_fs_applied: na::Vector2<f64>,
    pub mpc_fs_armed: Option<(usize, usize)>,
    pub mpc_fs_last: na::Vector2<f64>,
    /// The held MPC answer's diagnostics, read only by the logs: the
    /// reference vertical forces and the time it was solved at, the net
    /// pressure point its wrenches imply, and their contact frames (kept
    /// only for QP_WRENCH_CSV). Held across ticks with `wrench_mpc_command`.
    pub diag_mpc_reference: [f64; 2],
    pub diag_mpc_time: Option<f64>,
    pub diag_mpc_net_cop: [f64; 2],
    pub wrench_mpc_frames: Option<[misa_mpc::wrench_mpc::Contact; 2]>,
}

/// What the controller computed this tick, for the harness to read: the
/// trajectory CSV, the 40-tick print and the fall test read these instead of
/// the control loop's locals. Built once per tick, after the solve and the
/// per-foot CoP, from the values as they stand there (`z_ref` is the second
/// binding, `p_cmd` the one after the preview's correction, `r` the DCM
/// reference after the recovery reference's override).
pub struct TickRecord {
    /// The measured soles (FK), as the phases read them.
    pub actual_feet: crate::gait::Footsteps,
    /// The schedule's slice now (by value), its index, and how far into it.
    pub slice: crate::gait::Slice,
    pub slice_idx: usize,
    pub slice_frac: f64,
    /// Support from the schedule (`Double` throughout with NO_LIFT), steps
    /// taken, and the planned soles of this slice.
    pub support: crate::gait::Support,
    pub step_idx: usize,
    pub steps: crate::gait::Footsteps,
    /// The feet the QP solved with as contacts, and how many.
    pub stance_sides: Vec<usize>,
    pub nc: usize,
    /// Measured vertical force per foot, as the controller read it.
    pub fz_meas: [f64; 2],
    /// Measured DCM, its reference, and the distance between them.
    pub xi: na::Vector2<f64>,
    pub r: crate::dcm::DcmRef,
    pub dcm_err: f64,
    /// Commanded ZMP after the support-box clamp (and how far it was moved).
    pub p_cmd: na::Vector2<f64>,
    pub clamped: f64,
    /// CoM height reference and the commanded CoM acceleration.
    pub z_ref: f64,
    pub a_com: na::Vector3<f64>,
    /// Load share per stance foot.
    pub shares: Vec<f64>,
    /// Trunk roll / pitch (the estimate), their reference, and what the
    /// trunk task commanded and achieved [rad/s^2].
    pub roll: f64,
    pub pitch: f64,
    pub rp_ref: [f64; 2],
    pub trunk_cmd_rp: [f64; 2],
    pub trunk_acc_rp: [f64; 2],
    /// Swing-foot target and its tracking diagnostics.
    pub swing_tgt: na::Vector3<f64>,
    pub swing_cmd_acc_z: f64,
    pub swing_achieved_acc_z: f64,
    pub swing_res_z: f64,
    pub swing_res_rows: f64,
    pub swing_task_residual_rms: f64,
    pub swing_pos_err_xy: f64,
    pub swing_pos_err_z: f64,
    /// CoM preview: whether it has engaged, its CoM and velocity.
    pub replan_engaged: bool,
    pub preview_c: na::Vector2<f64>,
    pub preview_v: na::Vector2<f64>,
    /// The held MPC answer, the net pressure point it implies, and the
    /// landing offset it last asked for; its reference vertical forces, when
    /// it was solved, and its contact frames (see `CtlState`).
    pub wrench_mpc_command: Option<[misa_mpc::wrench_mpc::Wrench; 2]>,
    pub diag_mpc_net_cop: [f64; 2],
    pub mpc_fs_last: na::Vector2<f64>,
    pub diag_mpc_reference: [f64; 2],
    pub diag_mpc_time: Option<f64>,
    pub wrench_mpc_frames: Option<[misa_mpc::wrench_mpc::Contact; 2]>,
    /// The QP's levels by name, and the nominal contact forces it was
    /// pulled toward (`force_nominal`, world axes per stance slot).
    pub level_names: Vec<&'static str>,
    pub forces_nominal: na::DVector<f64>,
    /// QP timing [ns] (solve, physical check, QP_DIAG capture, MPC_DIAG
    /// capture), the normalised physical violation, and whether the solver
    /// was reset for a new QP structure.
    pub qp_solve_ns: u128,
    pub qp_check_ns: u128,
    pub qp_capture_ns: u128,
    pub mpc_capture_ns: u128,
    pub qp_physical_violation: f64,
    pub qp_reset: bool,
    /// The QP's per-foot CoP (sole frame, with fz) and force (world), the
    /// CoP utilisation, and the net pressure point of its forces.
    pub cop_qp: [[f64; 3]; 2],
    pub f_qp_w: [[f64; 3]; 2],
    pub cop_use: f64,
    pub diag_wbc_net_cop: [f64; 2],
    pub acc_dbg: [f64; 2],
    /// Leg reach used per side, hip heights, and the stride they allow now.
    pub reach_used: [f64; 2],
    pub hip_z: [f64; 2],
    pub stride_max_now: f64,
}

/// The controller's counters, reported after the loop (`DegradedTally`, the
/// harness's, stays apart). All start at zero.
#[derive(Clone, Debug, Default)]
pub struct WalkerStats {
    pub n_mom_cmp_ticks: u32,
    pub mom_cmp_peak: f64,
    pub n_push_z_drop: u32,
    pub n_cp_shift: u32,
    pub n_cp_retime: u32,
    pub cp_retime_total: f64,
    pub n_confirmed_admissions: u32,
    pub ref_engagements: u32,
    pub n_ds_capture: u32,
    pub ds_capture_total: f64,
    pub n_ds_feasible: u32,
    pub n_ds_continuous: u32,
    pub ds_feasible_total: f64,
    pub n_push_replan: u32,
    pub n_mpc_fs_cross: u32,
    pub preview_count: u32,
    pub preview_failures: u32,
    pub preview_error: f64,
    pub landing_retimes: u32,
    pub landing_delay: f64,
    pub n_replan: u32,
    pub n_corrected: u32,
    pub wrench_mpc_solves: u32,
    pub wrench_mpc_failures: u32,
}

/// The scenario and the command the walker is started with, and the values
/// it is still handed from the harness's plant (the leaks, §6 of the plan:
/// each is kept as it is, because fixing it changes the behaviour).
pub struct WalkerSetup {
    /// Realised control period [s] (substeps x the simulator's dt; the
    /// Config's `CTRL_DT` is the nominal one).
    pub dt: f64,
    /// End of the schedule [s] (`GaitParams.t_end`; today the run's length
    /// `T`, §6.11).
    pub t_end: f64,
    /// The known terrain. `Some` also for the zero-riser "stair" of the flat
    /// benches (§6.2).
    pub terrain: Option<Stairs>,
    /// The command, woven into the plan at start-up (§6.9).
    pub command: CommandSchedule,
    /// The push-recovery references (REF_LIBRARY; the harness reads them).
    pub recovery_refs: Vec<RefLib>,
    /// LEAK (§6.7): the WBC's and the MPC's friction coefficient,
    /// `cfg.wbc.friction_mu(MU_GROUND, ..)` -- built from the PLANT's floor
    /// friction, unless `cfg.wbc.mu_assumed` (`MU_ASSUMED`) replaces it.
    pub friction_mu: f64,
    /// LEAK (§6.6): the forward kinematics of the settled pose from the
    /// PLANT's joint angles (`rig.fk_now()`): the initial footsteps, the
    /// initial CoM and the feet's planned orientation come from it. Under
    /// `cfg.wbc.joint_q_estimate` the harness builds it from the t=0
    /// estimate instead ([`crate::robot::RobotModel::fk_of`]).
    pub init_fk: misarta::data::Data<f64>,
    /// LEAK: the leg-reach model, `LEG_REACH` (read by the harness) over the
    /// profile's length, with the descent's margin.
    pub leg_reach: crate::reach::LegReach,
    /// LEAK: the hips' model indices, from the harness's rig
    /// (`rig.link_to_idx` of `Profile::hip_links`).
    pub hip_mi: Option<[usize; 2]>,
}

/// The command (VX / VY / WZ, VX2 / VX2_AT / VX2_RAMP, SQUAT_*, JAB*).
#[derive(Clone, Debug)]
pub struct CommandSchedule {
    /// Forward / lateral speed [m/s] and turn rate [rad/s].
    pub vx: f64,
    pub vy: f64,
    pub wz: f64,
    /// Crouch/rise amplitude [m] and period [s].
    pub squat_amp: f64,
    pub squat_period: f64,
    /// Speed change: (from step, to speed [m/s], over steps).
    pub vx2: Option<(usize, f64, usize)>,
    /// The jab: on, its start [s], period per character [s], the L/R
    /// pattern, and the extended pose (shoulder pitch, elbow) [rad].
    pub jab: bool,
    pub jab_start: f64,
    pub jab_period: f64,
    pub jab_pattern: String,
    pub jab_pitch: f64,
    pub jab_elbow: f64,
}

/// Where the jab is at a time (see [`CommandSchedule::jab_at`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JabPhase {
    /// The arm the pattern names for this period: 0 = left, 1 = right.
    pub side: usize,
    /// How far through its period, `[0, 1)`.
    pub phase: f64,
    /// `0.5 (1 - cos 2 pi phase)`: 0 = in guard, 1 = fully extended.
    pub extension: f64,
}

impl CommandSchedule {
    /// The jab's own clock at `t` (one pattern character per `jab_period`
    /// from `jab_start`). Before `jab_start` it reads the start of the first
    /// period. The walker's arm reference (which only asks from `jab_start`
    /// on) and the harness's push line share it.
    pub fn jab_at(&self, t: f64) -> JabPhase {
        let ph = ((t - self.jab_start) / self.jab_period).max(0.0);
        let side = if self
            .jab_pattern
            .chars()
            .nth(ph.floor() as usize % self.jab_pattern.chars().count())
            .is_some_and(|c| c.eq_ignore_ascii_case(&'l'))
        {
            0
        } else {
            1
        };
        JabPhase { side, phase: ph.fract(), extension: 0.5 * (1.0 - (2.0 * std::f64::consts::PI * ph.fract()).cos()) }
    }
}

/// What `step` hands the harness for the plant: the WBC's solve (status,
/// extracted torques / accelerations / forces), the gravity torque the
/// degraded-solve fallback is built from, and whether the gate rejected it
/// or it is degraded.
pub struct Command {
    pub sol: misa_wbc::Solution,
    pub extracted: misa_wbc::Extracted,
    pub tau_gravity: na::DVector<f64>,
    pub rejected: bool,
    pub degraded: bool,
}

/// One tick of the walker: the command, the tick's record, and the state
/// the walker assembled from the estimate (the harness's measurements and
/// logs read its FK, CMM inputs and `q` / `v`).
pub struct StepOutput {
    pub command: Command,
    pub record: TickRecord,
    pub state: RobotState,
}

/// The walker with its plan made and its state initialised, before the WBC
/// stack is attached (the harness builds the stack from
/// [`WalkerInit::contact_params`], after the start-up prints that come
/// between the plan's and the stack's).
pub struct WalkerInit {
    cfg: Config,
    k: WalkerConsts,
    plan: Plan,
    ctl: CtlState,
}

/// The walker: its knobs, the start-up constants, the plan, the controller's
/// state and counters, and the WBC stack behind it.
pub struct Walker<S: WbcStack> {
    cfg: Config,
    k: WalkerConsts,
    plan: Plan,
    ctl: CtlState,
    stats: WalkerStats,
    stack: S,
    /// The time of the last `step` (`end_tick` closes that tick).
    t: f64,
}

impl WalkerInit {
    /// The plan, captured once from the settled pose (the initial footsteps,
    /// the schedule with its settles, the speed schedule, the staircase's
    /// footfalls, COM_PREPLAN, the DCM plan and the landing clock), the
    /// contact model, and the controller's initial state. The plan's lines
    /// go to `out`, in the order they always came.
    pub fn new(cfg: Config, robot: RobotModel, setup: WalkerSetup, out: &mut dyn LogSink) -> Self {
        let wcfg = &cfg;
        let prof = robot.prof;
        let dt = setup.dt;
        let total_mass = robot.total_mass;
        let total_t = setup.t_end;
        let stairs = &setup.terrain;
        let command = &setup.command;
        let (sole_half_l, sole_half_w) = prof.cop_half;
        let friction_mu = setup.friction_mu;
        let f_max_per_foot = wcfg.wbc.f_max_per_foot(total_mass, G);
        assert!(
            f_max_per_foot >= total_mass * G,
            "f_max ({f_max_per_foot:.1} N) is below the single-support nominal ({:.1} N)",
            total_mass * G
        );
        let gp = GaitParams {
            t_start: wcfg.gait.t_start,
            t_ds: wcfg.gait.t_ds,
            t_ss: wcfg.gait.t_ss,
            n_steps: wcfg.gait.n_steps,
            first_swing: wcfg.gait.first_swing,
            t_end: total_t,
            t_flight: wcfg.gait.t_flight,
        };
        // W1: run the schedule (so the ZMP reference moves) but never actually
        // release a foot. Isolates "can the balance reference move at all" from
        // everything that touchdown brings with it.
        let no_lift = wcfg.gait.no_lift;
        let cop_frac = wcfg.dcm.cop_frac;
        let anchor_leak = wcfg.contact.anchor_leak;
        // Contact-stabiliser (Baumgarte) gain. 1600 was picked for a two-foot
        // stance and concentrated on one foot it kicks the robot off the ground --
        // the same finding that fixed G1's single-leg stance (6ba64d2). Measured
        // here, stepping in place at a 99.4 mm stance, T=40 / N_STEPS=40:
        //
        //   kp_c   60  100  120  150  180  220  250  400  600  1600
        //   steps   9   16   17   25   10   13   17    9    6     3
        //
        // 150 is the DEFAULT, not a tuned value. The sweep is non-monotone and
        // both neighbours drop hard, and the sensitivity is sharper than that:
        // kd_c = 24 gives 25 steps where kd_c = 2*sqrt(150) = 24.495 gives 18.
        // A 2% change in the damping is worth 7 steps, so no single number here
        // is a result. What the data supports is "1600 is wrong and anything in
        // 60-250 is several times better".
        let kp_c = wcfg.contact.kp;
        // Critically damped against kp_c. Moving the two independently is how the
        // pair drifted apart in the first place: kp_c=400 with the old kd_c=80
        // gives 4 steps, and the same kp_c with 2*sqrt(kp_c) gives 9.
        let kd_c = wcfg.contact.kd();

        let kp_sw_xy = wcfg.swing.kp_xy();

        // DCM tracking. k is dimensionless: the DCM error decays as
        // exp(-k_dcm * omega * t), so 2.0 is two e-folds per DCM time constant.
        let k_dcm = wcfg.dcm.k_dcm;

        // ── Plan, captured ONCE from the settled pose ──────────────────────
        let d0 = &setup.init_fk;
        let com0 = robot.com_of(d0);
        let steps = Footsteps::from_fk(d0, robot.foot_mi, prof.sole_centre_x, prof.sole_below_origin);
        // The orientation each foot is PLANNED to land in, captured once from the
        // settled pose. Stepping in place does not turn, so this never changes --
        // which is the point: it is a plan, not a measurement, and a swing foot
        // that drifts is correcting toward it rather than tracking its own drift.
        let foot_rot0: [na::Matrix3<f64>; 2] =
            std::array::from_fn(|s| misarta::se3::rotation_matrix(&d0.oMi[robot.foot_mi[s]]));
        // TRACKING_FF_MIN_RISER [m]: apply the cubic acceleration feedforward
        // only on a staircase whose riser is at least this (0 = every stair
        // swing). Measured 2026-10-20: the feedforward takes the -150 mm descent
        // from 18 to 41/48 but, applied to every stair swing, needs the hand
        // retreat biases removed and then costs the 60/80 mm descents 3-7
        // members (round trip 192 -> 185). The gate is the RISER, not the
        // swing's height change: each foot climbs two risers per swing, so 60 mm
        // stairs already give 0.12 while the one-step deep bench gives 0.10-0.15.
        let tracking_ff_min_riser = wcfg.swing.tracking_ff_min_riser;
        let tracking_ff_riser_ok = stairs.as_ref().is_some_and(|st| st.riser.abs() >= tracking_ff_min_riser);
        let mut gait = GaitPlan::new(&gp);
        // A separate hold AFTER the normal inter-step pressure transfer. Extending
        // T_DS itself changes the transfer slope and proved non-robust for fixed
        // arms: 20 ms changes swapped survival between the 0.5/0.25 ms plants.
        // Consecutive DS slices make this one a constant-ZMP segment, so it adds
        // settling time without changing the validated transfer duration.
        let ds_settle = wcfg.gait.ds_settle;
        let ds_settle_after_step = wcfg.gait.ds_settle_after_step;
        assert!(
            ds_settle.is_finite() && ds_settle >= 0.0,
            "DS_SETTLE must be finite and non-negative"
        );
        let n_ds_settles = gait.insert_ds_settles(ds_settle, ds_settle_after_step);
        if n_ds_settles > 0 {
            outln!(out, 
                "inter-step settle: {n_ds_settles} mid-foot holds x {ds_settle:.3}s, after step {ds_settle_after_step}"
            );
        }
        // Z_COM_OFFSET [m]: the walking CoM height RELATIVE to the measured
        // initial one. `Z_COM` is an absolute world height, which on a descent
        // (spawn on a platform) silently adds the platform height to the crouch
        // -- measured 2026-10-05: Z_COM=0.66 took every descent below 6 cm to
        // 0/48. 0 = off = bit-identical.
        let z_com = wcfg.com_plan.z_com(com0.z);
        // How far the ZMP is asked to travel laterally, as a fraction of the
        // half stance width. 1.0 is true single support (all load on one sole);
        // anything less is only reachable with both feet down, so it is a W1
        // instrument, not a gait parameter.
        let zmp_lat_scale = wcfg.dcm.zmp_lat_scale;
        // Forward travel per step, in metres of BODY advance. 0 is stepping in
        // place, which stays a special case of this rather than a separate path.
        // A commanded speed maps to it as stride = v * (t_ss + t_ds).
        // Velocity command. STRIDE is kept as a shorthand for VX in m/step.
        let t_step = gp.t_ss + gp.t_ds + gp.t_flight;
        let stride = wcfg.gait.stride;
        let (vx, vy, wz) = (command.vx, command.vy, command.wz);
        // Crouch/rise while doing it, to load the sagittal chain during locomotion
        // rather than only when standing.
        let (squat_amp, squat_period) = (command.squat_amp, command.squat_period);
        // Steps over which the stride ramps in from standing.
        let stride_ramp = wcfg.gait.stride_ramp;
        // ALTERNATE=0 restores the step-and-close plan, for A/B.
        // Speed-change benchmark: from step VX2_AT the forward speed ramps to
        // VX2 over VX2_RAMP steps (gait::FootstepPlan::velocity_with_change).
        // Unset = the fixed-speed plan, bit-identical.
        let vx2_change = command.vx2;
        // A speed the fixed step time would turn into a longer stride than the
        // VX2_SCHED cap is reached by STEPPING FASTER instead -- the footfalls
        // are planned at the capped stride and each step from VX2_AT on (its
        // single support and the double support after it) is shortened by
        // stride_max / (v * t_step). Measured 2026-10-03 on G1 v6.3 with the
        // step time fixed at 1.21 s: 0.25 m/s 47/48, 0.30 m/s 12/48, 0.35 m/s
        // 1/48 -- the stride grows to 0.36-0.42 m and the landing leg is at
        // 99.8% of its length. VX2_TSCALE_MIN bounds the shortening. (The
        // hand-set cap VX2_STRIDE_MAX is removed.)
        let vx2_tscale_min = wcfg.speed.tscale_min;
        // VX2_SCHED: ONE speed schedule instead of the two hand-set knobs.
        // Measured 2026-10-14 (doc/from_claude/20261014_speed_design.md): the
        // swing's peak acceleration goes as v / T_step, so a speed is reached
        // by the LONGEST stride the legs allow (cap VXS_STRIDE_MAX) and only
        // then by cadence; a longer stride needs a lower CoM to stay within
        // leg reach, z_drop = 0.024 + VXS_Z_SLOPE (stride - 0.35) above a
        // 0.35 m stride, fading to 0 below 0.25 m. Reproduces the measured
        // settings (0.35/0.024 to 0.50 m/s, 0.45/0.06 at 0.55). The swing
        // acceleration bound v / T <= 0.7 then puts the ceiling at ~0.56 m/s
        // with a 0.45 m stride. (The two hand-set knobs, VX2_STRIDE_MAX and
        // VX2_Z_DROP, are removed.) Unset = no cap and no drop.
        let vx2_sched = wcfg.speed.sched;
        // (VX2_REANCHOR, re-anchoring the plan on the stance foot at every
        // single support under a speed command, is removed.)
        let (vx2_stride_max, vx2_z_drop_sched) = match vx2_change {
            Some((_, vx2, _)) if vx2_sched => {
                // The cap itself rises with speed: a 0.42 m stride at the
                // nominal cadence (0.35 m/s) stretches the stance leg over a
                // 0.84 s single support and falls even 5 cm lower, while the
                // same stride at 0.55 m/s (0.82 s step) walks. 0.35 m up to
                // VXS_V_LONG, then +VXS_CAP_SLOPE per m/s to VXS_STRIDE_MAX.
                let cap = wcfg.speed.stride_cap(vx2);
                let stride = (vx2 * t_step).min(cap);
                let zd = if stride >= 0.35 {
                    0.024 + wcfg.speed.z_slope * (stride - 0.35)
                } else {
                    0.024 * ((stride - 0.25) / 0.10).clamp(0.0, 1.0)
                }
                .min(wcfg.speed.z_max)
                // VXS_Z_MIN: a floor on the scheduled drop once a speed command
                // is active. Measured 2026-10-16 (push x speed, N=48, v6.10):
                // at 0.30 m/s the table drops 0.024 and the pushes go 77/192;
                // +0.02 gives 122, +0.04 gives 127 (backward 11 N.s 14 -> 48).
                // At 0.50 the table already drops 0.06 and more removes lateral
                // margin (swing-side 5 N.s 11 -> 6). 0 = the table alone.
                .max(wcfg.speed.z_min);
                // (VXS_Z_EXTRA, more drop on top of the schedule, is removed.)
                outln!(out, "speed schedule: v={vx2:.2} m/s -> stride {stride:.3} m (cap {cap:.2}), step {:.3} s, CoM drop {zd:.3} m", stride / vx2.max(1e-6));
                (cap, Some(zd))
            }
            _ => (0.0, None),
        };
            let vx2_cmd_speed: Option<f64> = vx2_change.map(|(_, v, _)| v);
        let vx2_change = match vx2_change {
            Some((at, vx2, ramp2)) if vx2_stride_max > 0.0 && vx2 * t_step > vx2_stride_max => {
                // Per-step time scale along the ramp, then the capped plan speed.
                let mut k = 0usize;
                let mut i = 0usize;
                while i < gait.slices.len() {
                    if matches!(gait.slices[i].support, Support::Single { .. }) {
                        if k >= at {
                            let g = if ramp2 == 0 { 1.0 } else { ((k - at + 1) as f64 / ramp2 as f64).min(1.0) };
                            let v_k = vx + (vx2 - vx) * g;
                            let scale = (vx2_stride_max / (v_k * t_step)).clamp(vx2_tscale_min, 1.0);
                            // The single support and EVERY double-support slice
                            // after it (transfer + DS_SETTLE hold).
                            let mut j = i;
                            while j < gait.slices.len()
                                && (j == i || matches!(gait.slices[j].support, Support::Double))
                            {
                                let t0 = gait.slices[j].t0;
                                // (VX2_DS_SCALE, an extra factor on the double
                                // supports, defaulted to 1 and is removed.)
                                let d = gait.slices[j].duration() * scale;
                                gait.retime(j, t0 + d);
                                j += 1;
                            }
                        }
                        k += 1;
                    }
                    i += 1;
                }
                outln!(out, "speed change: stride capped at {vx2_stride_max:.3} m, steps from {at} shortened (>= x{vx2_tscale_min:.2})");
                // The position plan advances `v * t_step` per step, so the capped
                // stride is the speed stride_max / t_step in the plan's units.
                Some((at, (vx2_stride_max / t_step).max(vx), ramp2))
            }
            other => other,
        };
        // Lower the walking CoM height by the schedule's drop from one step
        // before the speed change on. Measured 2026-10-05: the 0.45 m/s wall is
        // leg reach (stance leg at 99%, the heel lifts and the foot slips), and
        // 24 mm lower takes it from 7/48 to 48/48 -- but lowering the CoM for
        // the whole run trades descents and lateral pushes away. Tied to the
        // speed command it leaves every other bench bit-identical. (The
        // hand-set VX2_Z_DROP is removed.)
        let vx2_z_drop = vx2_z_drop_sched.unwrap_or(0.0);
            let vx2_z_from: Option<f64> = match vx2_change {
            Some((at, _, _)) if vx2_z_drop > 0.0 => gait
                .slices
                .iter()
                .filter(|sl| matches!(sl.support, Support::Single { .. }))
                .nth(at.saturating_sub(1))
                .map(|sl| sl.t0),
            _ => None,
        };
        let mut footsteps = if wcfg.gait.alternate {
            FootstepPlan::velocity_with_change(&gait, &steps, vx, vy, wz, t_step, stride_ramp, vx2_change)
        } else {
            FootstepPlan::ramped_stride(&gait, &steps, vx * t_step, stride_ramp)
        };
        // Stairs: rebuild the flat-ground footstep plan at tread centres. Terrain
        // enters the controller HERE and nowhere else -- the ZMP plan, the swing
        // touchdown, the per-slice anchors and the support polygon all read this
        // plan, so one edit moves all of them consistently.
        if let Some(st) = &stairs {
            // Rebuild the on-stairs part of the plan DETERMINISTICALLY instead
            // of snapping the velocity plan's wishes to the tread grid. Snapping
            // was tried first and quantised different nominal targets onto the
            // same spot: the approach degenerated into stepping against the
            // first riser and even a 20 mm riser fell. The sequence a stair
            // wants is an approach followed by alternating feet on successive
            // treads, with one riser of body advance per step.
            // Toe margin to a riser face. 10 mm was not enough: the heel-to-toe
            // pressure roll pushes the stance toe forward and it ground into the
            // first riser at 46 N while the climb was being launched off it.
            // The 20 mm default is 41% of kyo46rs's 49 mm sole half-length but only
            // 24% of G1's 85 mm, so the same number means two different things.
            // Measured 2026-09-20 on G1: the last approach footfall lands ON this
            // stand line, which puts the swing toe 20 mm from the riser face -- the
            // ankle advances 20 mm and the toe is already at the riser while the
            // foot has barely left the ground. Touchdown was detected with the foot
            // still RISING (td_vz +0.114), DCM error went 0.0042 -> 0.1857, cop_use
            // saturated and fz nearly doubled. Overridable so a longer-footed robot
            // can buy the clearance; the default keeps kyo46rs bit-identical.
            let m = sole_half_l + wcfg.stair_plan.toe_margin;
            let a_pos = st.start - m; // stand line: toes just clear of riser 1
            // Widen the on-stairs footfalls. Foot-over-foot passes the swing
            // shank beside the stance foot at tread height, and the measured
            // clearance was negative: left_shank <-> right_foot at 75 N on the
            // second climb. Walking a little wide opens it.
            let y_out = wcfg.stair_plan.y_out;
            let stair_gait_pair = wcfg.stair_plan.gait_pair;
            let n = st.n as i64;
            let land = st.start + st.n as f64 * st.tread + m;
            // Aim tread landings BEHIND the centre. The measured touchdown
            // overshoots its target by ~45 mm in x (travel-phase momentum); on
            // the one-step case the tread and the landing form one continuous
            // surface and the overshoot was harmless, but on a real flight each
            // tread is a 110 mm island and centre+45 is the nosing. Biasing the
            // TARGET rearward spends the sole's rear margin on the overshoot.
            let mut td_bias = wcfg.stair_plan.td_bias;
            // Descent-only rearward bias. Measured 2026-10-03 on G1 (v6.2, two
            // steps down, 60 mm risers, 29/48): the step-down landing overshoots
            // its goal by ~21 mm in x, a 170 mm sole on a 211 mm tread has 20 mm
            // to spare, so the toe contact spheres land PAST the nosing and the
            // stance foot pitches over the edge during the next swing. Default =
            // STAIR_TD_BIAS, i.e. bit-identical.
            // Applied only to treads that HAVE a nosing ahead (levels 1..n-1):
            // the last level of a descent is the ground, and biasing it rearward
            // walks the heel toward the riser face behind for nothing (measured:
            // one step down 192 -> 188, 150 mm 18 -> 0 with the bias on it).
            // Ascent twin of the descent bias: treads with a RISER AHEAD (levels
            // 1..n-1 of a flight going up). Measured 2026-10-03 on G1 v6.3, three
            // steps up: the landing overshoots by ~30 mm, the toe ends past the
            // tread (20 mm to spare) and comes down ON the next step -- at 4 and
            // 6 cm the sole rests half on it (centre 20 mm high, 0/48), at 8 cm
            // the riser face stops the toe (48/48). Default = STAIR_TD_BIAS, so
            // the one-step benches are bit-identical.
            let td_bias_asc_mid = wcfg.stair_plan.td_bias_asc_mid();
            let td_bias_desc = if st.riser < 0.0 || st.down_n > 0 {
                wcfg.descent.td_bias_desc(&wcfg.stair_plan)
            } else {
                td_bias
            };
            if wcfg.stair_plan.full_sole {
                let margin = wcfg.stair_plan.sole_margin;
                let available = 0.5 * st.tread - sole_half_l - margin;
                assert!(
                    margin.is_finite() && margin >= 0.0 && available >= 0.0,
                    "tread cannot contain the requested full sole and margin"
                );
                let requested = td_bias;
                td_bias = td_bias.clamp(-available, available);
                outln!(out, "full-sole plan: bias {requested:.6} -> {td_bias:.6} m, margin {margin:.4} m");
            }
            // Stride the walk-out takes on the landing, as a fraction of the tread.
            // Half a tread is 1.31x kyo46rs's 42 mm cruise stride (an acceleration)
            // but only 0.43x G1's 242 mm (a halving in one step). Measured
            // 2026-09-20 on G1: with the stair itself removed (riser 0.000) the run
            // still fell at 3.98 m -- on the landing, at exactly this deceleration
            // -- while the same recipe without STAIRS walked 6.23 m. `pos_of` and
            // `lvl_of` must read the SAME value or the grade and its inverse
            // disagree. The default keeps kyo46rs bit-identical.
            let landing_step = wcfg.stair_plan.landing_frac * st.tread;
            let slope_keep_stride = wcfg.slope.keep_stride;
            let pos_of = |lvl: i64| -> f64 {
                if lvl <= 0 {
                    a_pos
                } else if lvl > n {
                    // KEEP WALKING on the landing at half-tread strides instead
                    // of pinning both feet to one point. The pinned version made
                    // every run that reached the top fall THERE: the climb ends
                    // at ~0.15 m/s and a frozen dwell asks that momentum to die
                    // in one step -- measured, all eight one-step-ascent
                    // configurations fell at x = 0.43..0.49, i.e. on the
                    // landing, not on the stair.
                    land + (lvl - n - 1) as f64 * landing_step
                } else {
                    st.start + (lvl as f64 - 0.5) * st.tread
                        - if st.riser < 0.0 && lvl < n {
                            td_bias_desc
                        } else if st.riser > 0.0 && lvl < n {
                            td_bias_asc_mid
                        } else {
                            td_bias
                        }
                }
            };
            let lvl_of = |x: f64| -> i64 {
                // Landing positions grade upward too, or the walk-out targets
                // would converge on one point and re-create the pinned stop.
                if x >= st.start + st.n as f64 * st.tread {
                    // `pos_of` and this inverse must agree on the positions
                    // `pos_of` itself generates, and in floating point they do not:
                    // measured 2026-09-22, `pos_of(3)` produced a position whose
                    // `(x - land) / landing_step` evaluates to 0.999999999999999,
                    // so `floor` returned 0 and the point mapped back to level 2.
                    // The next swing then aimed at `pos_of(3)` again and BOTH feet
                    // were pinned to one x -- the walk-out stop this grading exists
                    // to prevent. Nudging by an epsilon before the floor makes the
                    // round trip exact for every level (verified 2..5).
                    return n + 1
                        + (((x - land) / landing_step + 1e-9).floor() as i64).max(0);
                }
                if x < st.start - 0.5 * m {
                    0
                } else if x >= st.start + st.n as f64 * st.tread {
                    n + 1
                } else {
                    (((x - st.start) / st.tread) as i64 + 1).clamp(1, n)
                }
            };
            // Land one foot ON the stand line before the first tread, instead of
            // jumping to the tread from wherever the approach grid happens to stop.
            //
            // OPT-IN (default off): the comment below records that squaring up was
            // measured to fell kyo46rs, and that result stands for kyo46rs's
            // geometry, where the un-squared jump is already small. G1 is the case
            // it was not measured on. Measured 2026-09-21: G1 cruises at 242 mm,
            // its last approach footfall lands at 1.739 m and tread 1 sits at
            // 2.106 m, so the un-squared jump is 367 mm -- half again the stride
            // the machine has been shown to take (242 mm, 16/16 under the 0.5 ms
            // audit). Squaring splits it into 141 mm and 226 mm, both inside that.
            // The knob exists so the trade (one short step vs one oversized one)
            // is a measured choice per robot rather than a constant.
            let align_stand_line = wcfg.stair_plan.align_stand_line;
            // Up-and-over course (STAIRS_DOWN): the footfall positions by level,
            // walked with explicit per-foot levels instead of the `lvl_of`
            // inverse. Level 0 = stand line, 1..n = up treads, then the landing
            // at `landing_step` strides up to a second stand line before the
            // descending flight, the down treads (rearward bias on the ones with
            // a nosing ahead), and the walk-out below.
            let course: Option<Vec<f64>> = (st.down_n > 0).then(|| {
                let le = st.landing_end();
                // Stand line before the descending flight: the toe must stay
                // short of the landing's edge. Measured 2026-10-03 (4 cm course,
                // v6 and v6.2 alike): with the toe AT the edge (the ascent's
                // STAIR_TOE_MARGIN=0 reused) the landing overshoots by 7 mm, the
                // toe spheres hang over the edge and the sole centre sinks 20 mm.
                let stand2 = le - sole_half_l - wcfg.descent.stand_margin;
                let mut v = vec![a_pos];
                for k in 1..=n {
                    v.push(pos_of(k));
                }
                let mut x = land;
                while x < stand2 - 0.5 * landing_step {
                    v.push(x);
                    x += landing_step;
                }
                v.push(stand2);
                for k in 1..=st.down_n {
                    let nosing_ahead = k < st.down_n;
                    v.push(
                        le + (k as f64 - 0.5) * st.tread
                            - if nosing_ahead { td_bias_desc } else { td_bias },
                    );
                }
                let last = *v.last().unwrap();
                for j in 1..=60 {
                    v.push(last + j as f64 * landing_step);
                }
                v
            });
            let mut course_lvl = [0usize; 2];
            let mut aligned = false;
            let mut cur = footsteps.initial;
            let mut ladder: Vec<(usize, f64, f64)> = Vec::new();
            for (i, sl) in gait.slices.iter().enumerate() {
                if let Support::Single { swing, .. } = sl.support {
                    let stance = 1 - swing;
                    let nom_x = footsteps.at[i].sole[swing].x;
                    // FOOT OVER FOOT, with no pause at the bottom: the first
                    // swing whose nominal target would cross the stand line goes
                    // straight to tread 1, and every later one to the level
                    // above its stance foot -- alternate feet on alternate
                    // treads. Two gaits were tried and rejected first: pair-up
                    // (both feet per tread) alternates a 110 mm ZMP jump with a
                    // 0 mm one, a stop-and-go CoM shuttle whose amplitude does
                    // not fit on a 98 mm sole; and squaring both feet up on a
                    // stand line before the first riser inserts the same
                    // stop-start at the bottom -- measured, THAT is what felled
                    // even a 20 mm riser, height-independently.
                    // STAIR_GAIT=pair revisits the rejected gait WITH the tools
                    // that did not exist at rejection time: ZMP_ROLL bridges
                    // half the dwell jump inside the sole, and the measured
                    // killer of foot-over-foot is its 220-270 mm swings -- every
                    // one lands short, high and late (dead-stretched leg waiting
                    // for the body), and the next lift-off inherits the mess.
                    // Pair-up halves the swing to ~110 mm.
                    // SLOPE_KEEP_STRIDE=1: on a known slope (no treads) keep the
                    // velocity plan's stride and only lift each footfall to the
                    // surface. The tread grid otherwise re-lays every footfall at
                    // the landing stride (G1: 0.199 m vs the commanded 0.149, a
                    // 35% faster climb) -- measured 2026-10-19 as the root of the
                    // pushes-on-a-slope collapse (late, long, shortening steps).
                    let keep_nominal = nom_x < a_pos - 1e-9 || (slope_keep_stride && st.n == 0 && st.slope != 0.0);
                    let tx = if keep_nominal {
                        nom_x // flat approach: the velocity plan's own wish
                    } else if align_stand_line && !aligned {
                        // First swing that would overshoot the stand line: put it
                        // ON the line. Only ever fires once; the next swing takes
                        // the tread from a known start, so both halves of the
                        // entry are bounded by the stand-line offset and half a
                        // tread rather than by where the approach grid landed.
                        aligned = true;
                        a_pos
                    } else if let Some(v) = &course {
                        let l = (course_lvl[stance] + 1).min(v.len() - 1);
                        course_lvl[swing] = l;
                        v[l]
                    } else if stair_gait_pair {
                        let sk = lvl_of(cur.sole[stance].x);
                        let wk = lvl_of(cur.sole[swing].x);
                        if wk < sk { pos_of(sk) } else { pos_of(sk + 1) }
                    } else {
                        pos_of(lvl_of(cur.sole[stance].x) + 1)
                    };
                    cur.sole[swing].x = tx;
                    cur.sole[swing].y = footsteps.at[i].sole[swing].y
                        + if keep_nominal {
                            0.0
                        } else if swing == 0 {
                            y_out
                        } else {
                            -y_out
                        };
                    cur.sole[swing].z = st.h(tx);
                    cur.yaw[swing] = footsteps.at[i].yaw[swing];
                    ladder.push((swing, tx, cur.sole[swing].z));
                }
                footsteps.at[i] = cur;
            }
            let s: Vec<String> = ladder
                .iter()
                .map(|(sw, x, z)| format!("{}({:.3},{:.3})", if *sw == 0 { "L" } else { "R" }, x, z))
                .collect();
            outln!(out, "stairs footfalls: {}", s.join(" "));
        }
        // Reserve swing time before the approach DCM is generated. Waiting until
        // touchdown to discover a longer step can already put the DCM beyond the
        // stance sole's capture range. Optionally also reserve time for long
        // same-height swings, including the transition onto the final landing.
        if wcfg.com_plan.preplan() {
            let vx_limit = wcfg.landing.speed_xy;
            let vz_limit = wcfg.landing.speed_z;
            let max_ss = wcfg.com_plan.stair_ss_max.max(gp.t_ss);
            // (COM_DESC_DS / COM_DESC_XFER, the double supports before and
            // after a descent swing stretched, are removed.)
            let xy_min = wcfg.com_plan.preplan_xy_min;
            for i in 1..gait.slices.len() {
                if let Some(side) = gait.slices[i].support.swing() {
                    let delta = footsteps.at_slice(i).sole[side] - footsteps.at_slice(i - 1).sole[side];
                    if delta.z.abs() > 0.005 || delta.xy().norm() > xy_min {
                        let duration =
                            (delta.xy().norm() / vx_limit).max(delta.z.abs() / vz_limit) + 0.05;
                        let new_t1 = gait.slices[i].t0 + duration.clamp(gp.t_ss, max_ss);
                        // Say so. The landing-speed sweep that motivated this
                        // line produced four bit-identical falls, and whether
                        // the extension had fired at all was not observable.
                        outln!(out, 
                            "preplan: slice {i} swing {} d=({:.3},{:+.3}) SS {:.2} -> {:.2}s",
                            if side == 0 { "L" } else { "R" },
                            delta.xy().norm(),
                            delta.z,
                            gait.slices[i].duration(),
                            new_t1 - gait.slices[i].t0
                        );
                        gait.retime(i, new_t1);
                    }
                }
            }
        }
        // The LIPM height is the CoM's height ABOVE THE WALKING SURFACE, which
        // on a descent platform is not its world z.
        // (STAIR_CROUCH, a fixed crouch on stairs that this height also took
        // out, was opt-in and is removed; STAIR_ZONE_CROUCH is the one in use.)
        // Heel-to-toe travel of the planned pressure inside each stance (see
        // DcmPlan::from_footsteps_rolled). On for stairs by default: the tread
        // pitch fixes the footstep pitch at ~2x what the commanded speed builds
        // momentum for, and the sole's own length is where the difference has
        // to come from. Must stay inside the CoP box (cop_half.0 * COP_FRAC).
        // Both are `DcmConfig::params`.
        let dcm_params = wcfg.dcm.params(z_com, stairs.as_ref().map(|st| st.base()));
        // (DCM_PREVIEW_HORIZON, a finite DCM preview, and DCM_REF_BLEND_T / _MIN,
        // a continuity guard that blended a jump of the DCM reference across a
        // plan rebuild, were not adopted and are removed. Measured 2026-09-23
        // on G1: a one-tick jump of xi_ref above 5 cm separated fallers from
        // survivors at every riser -- the guard was the experiment on it.)
        // The plan from here on: the schedule and footsteps built above, their
        // DCM reference (built now), the CoM height plan (level at z_com until
        // COM_REPLAN retargets it) and the landing clock with its knobs set
        // (`Config::landing_timing`; the force on the swing sole that counts
        // toward confirming a landing is LAND_CONFIRM_FZ_FRAC of the weight --
        // historically 5% (17 N on G1). Measured 2026-09-24 on the stair swing:
        // the foot rests on the tread for 1.4-2.8 s carrying a MEDIAN of 10 N,
        // because the swing task still holds it 4 mm above where it physically
        // sits (swing_pos_err_z = +4 mm) -- 10 N never reaches the 17 N gate,
        // and only impact chatter (spikes to 464 N) does). Every later edit goes
        // through it.
        let plan = Plan::new(
            gait,
            footsteps,
            dcm_params,
            crate::preview::HeightPlan::constant(z_com),
            wcfg.landing_timing(total_mass, G),
        );
        let omega = plan.dcm().omega;
        outln!(out, 
            "gait: {} steps, DS {:.2}s / SS {:.2}s (start {:.2}s), first swing = {}",
            gp.n_steps,
            gp.t_ds,
            gp.t_ss,
            gp.t_start,
            if gp.first_swing == 0 { "left" } else { "right" }
        );
        if zmp_lat_scale != 1.0 {
            outln!(out, "ZMP_LAT_SCALE={zmp_lat_scale}: the plan does NOT reach true single support");
        }
        outln!(out, 
            "lipm: z_com={z_com:.4} m  omega={omega:.3} rad/s (1/omega={:.3} s)  k_dcm={k_dcm}",
            1.0 / omega
        );
        outln!(out, 
            "footsteps (sole centres, world): L=({:+.4},{:+.4})  R=({:+.4},{:+.4})  stance width={:.1} mm",
            steps.sole[0].x,
            steps.sole[0].y,
            steps.sole[1].x,
            steps.sole[1].y,
            (steps.sole[0].y - steps.sole[1].y).abs() * 1e3
        );
        if vx != 0.0 || vy != 0.0 || wz != 0.0 || squat_amp != 0.0 {
            outln!(out, 
                "command: vx={vx:+.3} vy={vy:+.3} m/s  wz={wz:+.3} rad/s  squat={:.0} mm @ {squat_period:.1} s",
                squat_amp * 1e3
            );
        }
        if stride != 0.0 {
            outln!(out, 
                "stride: {:.3} m/step -> {:.3} m/s commanded, {:.2} m of travel planned over {} steps",
                stride,
                stride / (gp.t_ss + gp.t_ds),
                plan.footsteps().travel_x(),
                gp.n_steps
            );
        }
        if no_lift {
            outln!(out, "NO_LIFT: the schedule drives the ZMP reference, but both feet stay in contact");
        }

        // The contact model: the walker's (the MPC plans against its CoP box and
        // torsion) and the stack's (the harness builds it from
        // `contact_params()`).
        let contact_cfg = ContactParams {
            kp_c,
            kd_c,
            anchor_leak,
            anchor_leak_rot: 0.0,
            sole_offset_local: [prof.sole_centre_x, 0.0, -prof.sole_below_origin],
            friction_mu,
            cop_half: (sole_half_l * cop_frac, sole_half_w * cop_frac),
            mu_torsion: 0.05,
            f_max_per_foot,
            no_patch: false,
            dt,
        };

        // The controller's initial state (`CtlState` gives what each is).
        let mut correction = ContactCorrection::new(total_mass * G, wcfg.contact.ticks);
        let contact_on_bw = wcfg.contact.on_bw;
        assert!(
            contact_on_bw.is_finite() && contact_on_bw > 0.02 && contact_on_bw <= 1.0,
            "CONTACT_ON_BW must exceed the 0.02 BW release threshold and be <= 1"
        );
        correction.on_n = contact_on_bw * total_mass * G;
        // Terrain height under the support starts at the spawn platform's.
        let terrain_track = stairs.as_ref().map_or(0.0, |st| st.base());
        let ctl = CtlState {
            vx2_z_track: crate::height::SlewedDrop::default(),
            push_z_drop_until: f64::NEG_INFINITY,
            correction,
            terrain_track,
            cp_applied: na::Vector2::zeros(),
            cp_slice: None,
            last_scheduled_admit: [None; 2],
            admit_measured_z: [None; 2],
            confirmed_admit: [None; 2],
            ref_lib: None,
            ref_cooldown_until: f64::NEG_INFINITY,
            ref_lat_quiet_t: f64::NEG_INFINITY,
            ref_mode: None,
            push_replan_state: None,
            push_replan_shift: 0.0,
            mpc_fs_cross_latched: None,
            preview_p: steps.mid_xy(),
            preview_c: com0.xy(),
            preview_v: na::Vector2::zeros(),
            preview_next: 0.0,
            prev_support: plan.gait().support_at(0.0),
            swing_lift_off: None,
            run_lift_off: [None, None],
            edge: [None, None],
            run_vel_ema: None,
            run_flight_end: None,
            fz_prev: [0.0; 2],
            fz_rate_ema: [0.0; 2],
            managed_until: [f64::NEG_INFINITY; 2],
            previous_a_xy: None,
            wrench_mpc_next: 0.0,
            wrench_mpc_command: None,
            mpc_fs_slice: None,
            mpc_fs_applied: na::Vector2::zeros(),
            mpc_fs_armed: None,
            mpc_fs_last: na::Vector2::zeros(),
            diag_mpc_reference: [f64::NAN; 2],
            diag_mpc_time: None,
            diag_mpc_net_cop: [f64::NAN; 2],
            wrench_mpc_frames: None,
        };
        // What the phases read that start-up fixed.
        let land_extend_desc = wcfg.descent.land_extend(&wcfg.landing);
        let mpc_gate = wcfg.wbc.qp_gate() || wcfg.mpc.bench_gate;
        let k = WalkerConsts {
            dt,
            omega,
            total_mass,
            z_com,
            terrain_base: terrain_track,
            sole_half_l,
            sole_half_w,
            land_extend_desc,
            vx,
            t_step,
            gp,
            squat_amp,
            squat_period,
            vx2_change,
            vx2_cmd_speed,
            vx2_z_from,
            vx2_z_drop,
            contact_plan_cfg: wcfg.contact_plan.planner(),
            com_preview_dt: wcfg.com_plan.preview_dt.max(dt),
            com_replan_dt: wcfg.com_plan.replan_dt.max(dt),
            wz,
            command: command.clone(),
            kp_sw_xy,
            tracking_ff_riser_ok,
            foot_rot0,
            hip_mi: setup.hip_mi,
            leg_reach: setup.leg_reach,
            f_max_per_foot,
            friction_mu,
            mpc_gate,
            wrench_mpc_apply: wcfg.mpc.applies(),
            stair_wrench_dt: wcfg.mpc.dt.max(dt),
            stair_wrench_preview_dt: wcfg.mpc.preview_dt.max(dt),
            contact_cfg,
            steps0: steps,
            stairs: setup.terrain,
            ref_libs: setup.recovery_refs,
            robot,
        };
        WalkerInit { cfg, k, plan, ctl }
    }

    /// The contact model the stack is to be built with (the walker's MPC
    /// reads the same one).
    pub fn contact_params(&self) -> &ContactParams {
        &self.k.contact_cfg
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Attach the WBC stack.
    pub fn with_stack<S: WbcStack>(self, stack: S) -> Walker<S> {
        Walker {
            cfg: self.cfg,
            k: self.k,
            plan: self.plan,
            ctl: self.ctl,
            stats: WalkerStats::default(),
            stack,
            t: 0.0,
        }
    }
}

impl<S: WbcStack> Walker<S> {
    /// One tick of the control (phases 1-12): the observation in, the
    /// command and the tick's record out. The control's lines go to `out` as
    /// they are decided, its opt-in records to `diag`.
    pub fn step(&mut self, obs: &Observation, out: &mut dyn LogSink, diag: &mut dyn WalkerDiag) -> StepOutput {
        let (wcfg, k) = (&self.cfg, &self.k);
        let (ctl, plan, stats) = (&mut self.ctl, &mut self.plan, &mut self.stats);
        let t = obs.tick as f64 * k.dt;
        self.t = t;
        let robot = &k.robot;
        let prof = robot.prof;
        // Phase 1: the landing detector's settings (it reads no observation).
        phase_landing_setup(plan, wcfg, k, t);
        // The state from the estimate (S14; the harness assembled it
        // before): q / v, M with the armature, h with the damping, the FK,
        // the CoM and its Jacobian. The phases read it through `TickObs`.
        let state = robot.assemble(obs.estimate);
        let obs = &TickObs {
            tick: obs.tick,
            state: &state,
            contact_fz: obs.contact_fz,
            // The joint angles the posture levels read: the plant's (LEAK,
            // §6.6), or the estimate's under JOINT_Q_EST.
            joint_pos: if wcfg.wbc.joint_q_estimate { &obs.estimate.joint_q } else { obs.joint_pos_plant },
        };
        let st = obs_state(obs);
        let data = &st.data;
        // The measured soles (FK), once per tick: the phases and the
        // trajectory CSV read this one.
        let actual_feet = Footsteps::from_fk(data, robot.foot_mi, prof.sole_centre_x, prof.sole_below_origin);
        // The centroidal momentum matrix (and its rate term), computed once
        // per tick on first use (the MPC, the momentum task, MOM_CMP).
        let cent = Centroidal::new(robot, st);

        phase_landing_clock(ctl, plan, stats, wcfg, k, obs, &actual_feet, t, out);
        // The anchor changes phases 3 and 4 decide, in order (the stack
        // applies them in phase 11, before its contact level).
        let mut anchor_events: Vec<AnchorEvent> = Vec::new();
        let ph3 = phase_support(ctl, plan, &mut anchor_events, wcfg, k, obs, t, out);

        let ph4 = phase_contacts(ctl, plan, stats, &mut anchor_events, wcfg, k, obs, &ph3, t, out);
        let SupportOut { slice, slice_frac, support, step_idx, slice_idx, steps, .. } = ph3;
        let ContactsOut { fz_meas, stance_sides, stance, nc, ramp } = ph4;

        let ph5 = phase_dcm(ctl, plan, stats, wcfg, k, obs, &actual_feet, &ph3, t, out);

        let CmdZmpOut { p_cmd, clamped, zmp_deficit } = phase_cmd_zmp(ctl, plan, stats, wcfg, k, &ph3, &ph5, t, out);
        let DcmOut { xi, r, dcm_err } = ph5;
        let p_cmd = phase_preview(ctl, plan, stats, diag, wcfg, k, obs, &actual_feet, &stance_sides, p_cmd, t);
        let HeightOut { z_ref, a_com } = phase_height(ctl, plan, wcfg, k, obs, &ph3, &stance_sides, &p_cmd, t);

        let mpc_capture_ns = phase_wrench_mpc(
            ctl, plan, stats, diag, wcfg, k, obs, &actual_feet, &cent, &ph3, &stance_sides, &ph5, &a_com, t, out,
        );

        // ---- the WBC: the levels' references (phase 10), the request and
        // the stack's solve (phase 11), the solve's diagnostics (phase 12).
        let refs = phase_task_refs(
            ctl, plan, stats, wcfg, k, obs, &cent, &ph3, &stance_sides, &stance, nc, &ramp, &ph5, &p_cmd, &zmp_deficit,
            t,
        );
        let wbc = phase_wbc(
            &mut self.stack, ctl, wcfg, k, obs, &cent, &ph3, &stance_sides, &stance, nc, &ramp, &anchor_events, &a_com,
            &refs, t, out,
        );
        let sd = phase_solve_diag(wcfg, k, obs, &stance, &refs, &wbc);
        let TaskRefs { roll, pitch, rp_ref, tau_gravity, shares, swing_tgt, swing_cmd_acc_z, .. } = refs;
        let WbcSolution {
            sol,
            extracted,
            level_names,
            forces_nominal,
            acc_dbg,
            trunk_acc_rp,
            timings: crate::wbc::QpTimings { solve_ns: qp_solve_ns, check_ns: qp_check_ns, capture_ns: qp_capture_ns },
            physical_violation: qp_physical_violation,
            reset: qp_reset,
            rejected,
            degraded,
            ..
        } = wbc;
        // What the trunk task asked for (its roll / pitch reference).
        let trunk_cmd_rp = rp_ref;
        let SolveDiag {
            swing_task_residual_rms,
            swing_res_z,
            swing_res_rows,
            reach_used,
            hip_z,
            stride_max_now,
            swing_achieved_acc_z,
            swing_pos_err_xy,
            swing_pos_err_z,
            cop_qp,
            f_qp_w,
            cop_use,
            diag_wbc_net_cop,
        } = sd;
        // The tick's record: what the controller computed, for the harness's
        // logs and measurements.
        let record = TickRecord {
            actual_feet,
            slice,
            slice_idx,
            slice_frac,
            support,
            step_idx,
            steps,
            stance_sides,
            nc,
            fz_meas,
            xi,
            r,
            dcm_err,
            p_cmd,
            clamped,
            z_ref,
            a_com,
            shares,
            roll,
            pitch,
            rp_ref,
            trunk_cmd_rp,
            trunk_acc_rp,
            swing_tgt,
            swing_cmd_acc_z,
            swing_achieved_acc_z,
            swing_res_z,
            swing_res_rows,
            swing_task_residual_rms,
            swing_pos_err_xy,
            swing_pos_err_z,
            replan_engaged: plan.claims().done(crate::plan::Writer::ReplanStart),
            preview_c: ctl.preview_c,
            preview_v: ctl.preview_v,
            wrench_mpc_command: ctl.wrench_mpc_command,
            diag_mpc_net_cop: ctl.diag_mpc_net_cop,
            mpc_fs_last: ctl.mpc_fs_last,
            diag_mpc_reference: ctl.diag_mpc_reference,
            diag_mpc_time: ctl.diag_mpc_time,
            wrench_mpc_frames: ctl.wrench_mpc_frames.clone(),
            level_names,
            forces_nominal,
            qp_solve_ns,
            qp_check_ns,
            qp_capture_ns,
            mpc_capture_ns,
            qp_physical_violation,
            qp_reset,
            cop_qp,
            f_qp_w,
            cop_use,
            diag_wbc_net_cop,
            acc_dbg,
            reach_used,
            hip_z,
            stride_max_now,
        };
        StepOutput { command: Command { sol, extracted, tau_gravity, rejected, degraded }, record, state }
    }

    /// The end of the tick (phase 13), after the harness has fired its push:
    /// a recovery reference whose time is up hands the feet back to the gait.
    pub fn end_tick(&mut self, out: &mut dyn LogSink) {
        phase_end_tick(&mut self.ctl, &mut self.plan, &self.cfg, self.t, out);
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn robot(&self) -> &RobotModel {
        &self.k.robot
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    pub fn stats(&self) -> &WalkerStats {
        &self.stats
    }

    /// The WBC stack (its contact anchors, for the trajectory CSV).
    pub fn stack(&self) -> &S {
        &self.stack
    }

    /// The planned LIPM's omega.
    pub fn omega(&self) -> f64 {
        self.k.omega
    }

    /// The gait's timing (with `t_end`).
    pub fn gait_params(&self) -> &GaitParams {
        &self.k.gp
    }

    /// The feet at start-up (sole centres, world).
    pub fn initial_feet(&self) -> &Footsteps {
        &self.k.steps0
    }

    /// The command the walker was started with.
    pub fn command(&self) -> &CommandSchedule {
        &self.k.command
    }
}
/// A foot rolling on its heel or toe edge (HEEL_TOE).
#[derive(Clone, Copy, Debug)]
pub struct EdgeState {
    /// Edge point in the foot-link frame, and where it is held (world).
    pub r_local: [f64; 3],
    pub p_anchor: na::Vector3<f64>,
    /// The flat foot orientation the pitch is measured from.
    pub r0: na::Matrix3<f64>,
    /// Pitch about the sole y axis from `pitch0` to `pitch1` over [t0, t0 + dur].
    pub pitch0: f64,
    pub pitch1: f64,
    pub t0: f64,
    pub dur: f64,
    /// true = heel (ends flat, re-anchored), false = toe (ends at lift-off).
    pub heel: bool,
}

impl EdgeState {
    pub fn pitch_at(&self, t: f64) -> (f64, f64) {
        let u = ((t - self.t0) / self.dur.max(1e-6)).clamp(0.0, 1.0);
        let s = u * u * (3.0 - 2.0 * u);
        let ds = if (0.0..1.0).contains(&u) { 6.0 * u * (1.0 - u) / self.dur.max(1e-6) } else { 0.0 };
        (self.pitch0 + (self.pitch1 - self.pitch0) * s, (self.pitch1 - self.pitch0) * ds)
    }
}


