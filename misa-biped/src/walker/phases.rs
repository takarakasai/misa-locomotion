//! The walker's control, phase by phase (S11 of
//! doc/from_claude/20261029_walker_extraction_plan.md).
//!
//! Each `phase_*` function is one block of the tick's control, moved here
//! verbatim from the example's loop: the knobs it reads are bound at its top
//! from the [`Config`] (the same `let x = cfg.sub.x;` aliases the loop bound
//! once from its config), the start-up constants come from [`WalkerConsts`],
//! and what it computed for the later phases is returned in its own `*Out`
//! struct. The plan, the controller's state and its counters are passed by
//! `&mut`; [`super::Walker::step`] calls them in order and builds the tick's
//! `TickRecord` from their outputs. The arithmetic, the order of the prints
//! and the order of the plan's rebuilds are the loop's own.
use super::{CtlState, WalkerStats};
use crate::dcm::{DcmRef, SupportBox, StepTiming, com_accel_xy, commanded_zmp, dcm_of, step_timing};
use crate::gait::{
    Footsteps, GaitParams, Slice, Support, swing_position, swing_velocity,
};
use crate::dcm::G;
use crate::terrain::Stairs;
use crate::config::Config;
use crate::plan::{Plan, Writer, claim_check};
use crate::recovery::{RefLib, RefMode};
use crate::robot::{RobotModel, RobotState};
use crate::sink::{LogSink, WalkerDiag};
use crate::wbc::{
    AnchorEvent, ArmRef, CmpRef, ContactMode, ContactParams, MomCmp, MomRef, MpcNominal, SwingAxes, SwingRef,
    TrunkGains, TrunkRef, WbcRequest, WbcSolution, WbcStack, cop_from_sole_wrench, trunk_ori_ref,
};
use crate::{errln, outln};
use nalgebra as na;
use std::f64::consts::PI;

/// What the control reads that is fixed at start-up but is not a knob: the
/// robot model, the realised control period, the planned LIPM's omega, the
/// nominal CoM height, and the scenario values the walker is still handed
/// (the known terrain, the sole half-size the harness read).
pub(crate) struct WalkerConsts {
    pub robot: RobotModel,
    /// Realised control period (substeps x sim dt).
    pub dt: f64,
    /// The DCM plan's omega (`plan.dcm().omega`).
    pub omega: f64,
    pub total_mass: f64,
    /// Nominal CoM height and the terrain height it was measured over.
    pub z_com: f64,
    pub terrain_base: f64,
    /// The sole's half-size (`Profile::cop_half`).
    pub sole_half_l: f64,
    pub sole_half_w: f64,
    /// The known terrain (`Some` also for the zero-riser "stair" of the flat
    /// benches, §6.2 of the plan).
    pub stairs: Option<Stairs>,
    /// `cfg.descent.land_extend(&cfg.landing)`.
    pub land_extend_desc: f64,
    /// The command (until a command schedule replaces them, §6.9): the
    /// forward speed, the step period and the gait's timing, the squat, the
    /// speed change (VX2: at step, speed, ramp steps; after the stride cap)
    /// with the speed it commands, the time its CoM drop starts and the drop.
    pub vx: f64,
    pub t_step: f64,
    pub gp: GaitParams,
    pub squat_amp: f64,
    pub squat_period: f64,
    pub vx2_change: Option<(usize, f64, usize)>,
    pub vx2_cmd_speed: Option<f64>,
    pub vx2_z_from: Option<f64>,
    pub vx2_z_drop: f64,
    /// The push-recovery references (REF_LIBRARY, read by the harness).
    pub ref_libs: Vec<RefLib>,
    /// `cfg.contact_plan.planner()`, and the CoM preview's clocks
    /// (`cfg.com_plan.{preview_dt, replan_dt}.max(dt)`).
    pub contact_plan_cfg: crate::contact_plan::Config,
    pub com_preview_dt: f64,
    pub com_replan_dt: f64,
    /// The wrench MPC: the per-foot force ceiling
    /// (`cfg.wbc.f_max_per_foot(total_mass, G)`), the WBC's friction
    /// coefficient (`cfg.wbc.friction_mu(MU_GROUND, ..)` -- built from the
    /// PLANT's friction, §6.7), whether the solution is validated
    /// (QP_BENCH_GATE or MPC_BENCH_GATE), `cfg.mpc.applies()`, its update and
    /// node periods (`.max(dt)`), and the contact model (the MPC's CoP box
    /// and torsion coefficient are the WBC's; the stack is built from the
    /// same one).
    pub f_max_per_foot: f64,
    pub friction_mu: f64,
    pub mpc_gate: bool,
    pub wrench_mpc_apply: bool,
    pub stair_wrench_dt: f64,
    pub stair_wrench_preview_dt: f64,
    pub contact_cfg: ContactParams,
    /// The levels' references: the turn rate (WZ), the command (the jab's
    /// clock and pose, JAB*), the swing's xy gain (`cfg.swing.kp_xy()`), whether this
    /// staircase takes the swing's cubic feed-forward (TRACKING_FF_MIN_RISER
    /// against the known riser), and the feet's orientation at start-up
    /// (a planned yaw rotates it).
    pub wz: f64,
    pub command: super::CommandSchedule,
    pub kp_sw_xy: f64,
    pub tracking_ff_riser_ok: bool,
    pub foot_rot0: [na::Matrix3<f64>; 2],
    /// The hips (leg reach) and the reach model (phase 12's diagnostics).
    pub hip_mi: Option<[usize; 2]>,
    pub leg_reach: crate::reach::LegReach,
    /// The feet at start-up (the settled pose's FK; the plan starts here).
    pub steps0: Footsteps,
}

/// The centroidal momentum matrix `A_G(q)` and `Ȧ_G v`, computed once per
/// tick on first use. Both are pure functions of the model and the tick's
/// `q` / `v`, which nothing changes inside the tick, so the MPC (`h`), the
/// momentum task and MOM_CMP read the same values they used to compute three
/// times (the result is bit-identical). `A_G` is kept with the state
/// ([`RobotModel::cmm`]), so the harness's measurements read the same one.
pub(crate) struct Centroidal<'a> {
    robot: &'a crate::robot::RobotModel,
    st: &'a RobotState,
    dcmm_v: std::cell::OnceCell<na::Vector6<f64>>,
}

impl<'a> Centroidal<'a> {
    pub fn new(robot: &'a crate::robot::RobotModel, st: &'a RobotState) -> Self {
        Centroidal { robot, st, dcmm_v: Default::default() }
    }

    pub fn cmm(&self) -> &na::DMatrix<f64> {
        self.robot.cmm(self.st)
    }

    pub fn dcmm_v(&self) -> &na::Vector6<f64> {
        self.dcmm_v
            .get_or_init(|| misarta::centroidal::compute_cmm_dot_times_v(&self.robot.model, &self.st.q, &self.st.v))
    }
}

/// The observation as the phases read it: the harness's [`crate::robot::Observation`]
/// with its estimate replaced by the state the walker assembled from it
/// (S14).
pub(crate) struct TickObs<'a> {
    pub tick: usize,
    pub state: &'a RobotState,
    pub contact_fz: [f64; 2],
    /// `Observation::sole_cop`.
    pub sole_cop: [Option<[f64; 2]>; 2],
    /// The joint angles the posture and arm-hold levels read (harness
    /// order): the plant's `Observation::joint_pos_plant` (LEAK), or the
    /// estimate's `joint_q` under `JOINT_Q_EST`.
    pub joint_pos: &'a [f64],
}

/// The tick's assembled state.
pub(crate) fn obs_state<'a>(obs: &TickObs<'a>) -> &'a RobotState {
    obs.state
}

/// `COP_LEASH`: the CoP box of foot `side`, cut to within the leash of the
/// CoP its sole measures, as (centre from the sole centre, half-extents) in
/// the sole frame; None when the leash is off or does not apply (no
/// reading, too little load, a level sole under `COP_LEASH_TILT`, or not
/// in single support -- the seesaw is the lone stance foot's, and in double
/// support the spawn drop rocks both soles on their corners -- or in the
/// stair zone, where the planner knows the edge and the descent rolls the
/// stance foot over the nosing on purpose: leashed, the 20 mm descent fell
/// 14 -> 4 of 48). `full` is the uncut half box.
pub(crate) fn cop_leash_box(
    cfg: &Config,
    k: &WalkerConsts,
    obs: &TickObs,
    sup: &SupportOut,
    side: usize,
    full: (f64, f64),
) -> Option<([f64; 2], (f64, f64))> {
    let c = &cfg.contact;
    if c.cop_leash <= 0.0
        || !matches!(sup.support, Support::Single { stance, .. } if stance == side)
        || obs.contact_fz[side] < c.cop_leash_fz * k.total_mass * G
    {
        return None;
    }
    let in_stairs = k.stairs.as_ref().is_some_and(|st| {
        (st.riser.abs() > 1e-6 || st.down_n > 0)
            && crate::height::stair_zone_crouch(sup.steps.mid_xy().x, st, cfg.stair_zone.lead, cfg.stair_zone.tail, 1.0)
                > 0.0
    });
    if in_stairs {
        return None;
    }
    let m = obs.sole_cop[side]?;
    if c.cop_leash_tilt > 0.0 {
        let r = misarta::se3::rotation_matrix(&obs_state(obs).data.oMi[k.robot.foot_mi[side]]);
        if r[(2, 2)].clamp(-1.0, 1.0).acos() < c.cop_leash_tilt {
            return None;
        }
    }
    let cut = |m: f64, h: f64| {
        let m = m.clamp(-h, h);
        let (lo, hi) = ((m - c.cop_leash).max(-h), (m + c.cop_leash).min(h));
        (0.5 * (lo + hi), (0.5 * (hi - lo)).max(0.002))
    };
    // Fore/aft only: the seesaw is about the pitch axis (the bumps run the
    // full width), and a lateral leash pinned the box to a corner on the
    // spawn drop (fell at 0.5 s, 2026-11-01).
    let (cx, hx) = cut(m[0], full.0);
    Some(([cx, 0.0], (hx, full.1)))
}

/// Phase 1: the landing detector's tolerances and extension for the slice
/// the clock is in (ascending, descending, or a step whose end the contact
/// planner owns). Assigned every tick, before the plant is read.
pub(crate) fn phase_landing_setup(plan: &mut Plan, cfg: &Config, k: &WalkerConsts, t: f64) {
    // Extra landing-deadline slack for the swing that CHANGES HEIGHT, i.e. the
    // step onto a tread. Zero (the default) means "same as everywhere else".
    //
    // Measured 2026-09-22 on G1: the step onto a 20 mm tread needs roughly a
    // second of slack before its load crosses the confirm threshold -- the foot
    // rests on the tread at 4-6 N for ~500 ms first -- but applying that slack
    // to EVERY swing stalls the gait clock afterwards. With
    // `COM_LAND_EXTEND=1.08` the walk reached the tread 15/16 and then died:
    // 2 landing retimes totalling 2.16 s, the stair swing taking 2.035 s
    // against a planned 0.681, and the swings after it collapsing to
    // 1.075 / 0.500 / 0.260 / 0.035 s. With the default 0.25 it never reaches
    // the tread at all (0/16). The two ends of that trade want different
    // numbers, so they get different numbers.
    let land_extend_rise = cfg.landing.extend_rise;
    // STEP_DZ_MIN: planned sole-height change above which a swing is a
    // stair swing (rise/descent dwell and confirmation, stair landing box,
    // landing-timing activation). 0.005 = shipped, bit-identical; a known
    // slope (TERRAIN_SLOPE, 10-15 mm per step) sets ~0.03 so its steps are
    // level steps (measured 2026-10-15: at 0.005 every step waited the
    // 1.08 s rise dwell and the walk collapsed). STEP_DZ_SWING_MIN does the
    // same for the swing PROFILE choice (stair quintic vs smoothstep).
    let step_dz_min = cfg.landing.step_dz_min;
    // 0 = a descending swing keeps its extension as a dwell too (double
    // support starts at the deadline, as the historical timeout did).
    let contact_event_end_on_touch = cfg.landing.event.end_on_touch;
    // xy tolerance of the landing confirmation for DESCENDING swings only.
    // Measured 2026-10-03 on the up-and-over course (8 cm, 2/48): the long
    // step-down swing overshoots its goal by ~23 mm in x, the 12 mm gate
    // never confirms a foot that is flat on the ground, the swing times out
    // and the MPC then moves the landing 15 cm while the foot is down and
    // drags it. Unset = LAND_CONFIRM_XY (bit-identical).
    // (LAND_CONFIRM_XY_DESC_CP, COM_LAND_EXTEND_DESC_CP and
    // LAND_CONFIRM_FZ_DESC_FRAC -- variants of the descent confirmation --
    // and LAND_CONFIRM_XY_RISE -- confirming rising swings, which ended the
    // CEM's 1.08 s ascent dwell (1 step up 192 -> 114) -- were not adopted
    // and are removed.)
    // z tolerance that goes with the wider xy one: the 10 mm default lets
    // the confirmation fire on the HEEL STRIKE of a toe-up foot (sole centre
    // still 8 mm high), 60 ms before the sole is flat (measured: 2 steps x
    // 6 cm 46 -> 33 with the wide xy alone). Unset = LAND_CONFIRM_Z.
    // (Unset = LAND_CONFIRM_Z: the landing clock's `z_tol_override`.)
    let land_confirm_xy_desc = cfg.descent.land_confirm_xy;
    let contact_plan_on = cfg.contact_plan.on;
    // (COM_LAND_PREDICT_FRAC is the landing clock's `min_fraction`.)
    let land_extend_base = cfg.landing.extend;
    let land_extend_desc = k.land_extend_desc;
    if land_extend_rise > 0.0 {
        // Same test the landing detector itself uses to decide a swing is
        // a stair swing (`preview.rs`: the planned sole height changes by
        // more than 5 mm), so the slack applies to exactly the swings that
        // need it and to no others.
        let i = plan.gait().index_at(t);
        let dz_swing = plan.gait().slices[i].support.swing().map_or(0.0, |side| {
            plan.footsteps().at_slice(i).sole[side].z
                - plan.footsteps().at_slice(i.saturating_sub(1)).sole[side].z
        });
        let rising = dz_swing.abs() > step_dz_min;
        let descending = dz_swing < -step_dz_min;
        // The contact planner owns the end of a step it has retimed; the
        // landing extension must not re-open it (assigned every tick, so
        // the planner's own zeroing would be undone here otherwise).
        let cp_owns_end = plan.claims().end_time_owner(i) == Some(Writer::ContactPlan);
        let mut ed = plan.edit(Writer::LandingSetup);
        ed.landing_mut().event_end_on_touch = descending && contact_event_end_on_touch;
        ed.landing_mut().xy_tol_override = if descending { land_confirm_xy_desc } else { None };
        ed.landing_mut().max_extension = if contact_plan_on && cp_owns_end {
            0.0
        } else if descending {
            land_extend_desc
        } else if rising {
            land_extend_rise
        } else {
            land_extend_base
        };
        ed.commit();
    }
}

/// Phase 2: the landing clock (touchdown, timeout, the confirmed admission's
/// certificate, the touchdown event's shift), DS_FEASIBLE, DS_CAPTURE, and the
/// engagement of the CoM re-plan. `actual` is the measured soles (FK).
#[allow(clippy::too_many_arguments)]
pub(crate) fn phase_landing_clock(
    ctl: &mut CtlState,
    plan: &mut Plan,
    stats: &mut WalkerStats,
    cfg: &Config,
    k: &WalkerConsts,
    obs: &TickObs,
    actual: &Footsteps,
    t: f64,
    out: &mut dyn LogSink,
) {
    // ---- contact-driven phase transitions -----------------------------
    //
    // The schedule says when the step ENDS; the ground says when it actually
    // ended. Measured on this robot, the swing foot lands about 30 ms early
    // on a 0.35 s single support -- 8.6% of the phase, every step, same sign
    // -- and a time-based schedule keeps commanding a swing that has already
    // finished while the QP solves against a contact set that is a step
    // behind. Terminating the swing on contact is what a passive-ankle biped
    // does in hardware (Kim et al. 2019, arXiv:1901.08100: contact switches
    // "terminate swing foot motion controls when the swing foot touches the
    // ground earlier than anticipated").
    let com_replan = cfg.com_plan.replan;
    let contact_admit_confirmed = cfg.contact.admit_confirmed;
    // A landing that ends by the extension TIMEOUT counts too when the sole
    // is within this of its planned tread height: measured 2026-09-24, the
    // champion's stair landings mostly end that way (retime median 2.16 s =
    // 1.0 s swing + 1.08 s extension) with the sole on the tread all along.
    let contact_admit_z = cfg.contact.admit_z;
    // (PHASE_BY_CONTACT / PHASE_EXTEND / PHASE_MIN_FRAC / PHASE_LOAD /
    // PHASE_TICKS, the phase clock advanced by measured contact, are removed;
    // its summary line keeps the zeros it printed when off.)
    // The control loop's debug prints (`DIAG_REPLAN`, `*_DEBUG`, ...).
    let debug = cfg.debug;
    let contact_event_shift = cfg.landing.event.shift;
    // Run the DS_FEASIBLE re-plan on a timeout end as well (it is keyed on a
    // non-zero retime, which a timeout never produces). Default off.
    let ds_feasible_on_timeout = cfg.double_support.feasible_on_timeout;
    let ds_feasible = cfg.double_support.feasible;
    let cop_frac = cfg.dcm.cop_frac;
    // Keep the commanded ZMP inside the support polygon, shrunk by this
    // factor. Riding the exact edge means the next disturbance makes P0
    // infeasible outright.
    let zmp_margin = cfg.dcm.zmp_margin;
    // Pivot box for the DS sizing: "both" = the double-support box (the
    // historical choice), "stance" = the OLD stance sole only. At the cut the
    // new foot carries ~25% of the weight (measured), so a pivot under it is
    // not realisable until the load has moved; the stance-only box is what
    // the pressure can reach NOW.
    let ds_feasible_box_stance = cfg.double_support.feasible_box_stance;
    let ds_feasible_continuous = cfg.double_support.feasible_continuous;
    let ds_feasible_max = cfg.double_support.feasible_max;
    // Ramp variant: the double-support ZMP ramps from a solved p0 to the
    // next stance's own entry point p1, so the new foot takes load INSIDE
    // the double support. With the constant solve above the ZMP sits behind
    // the leading foot for the whole DS (measured 2026-09-24: the admitted
    // foot carried 0-6 N until single support). For p(t) = p0 + v t:
    //     xi(0) = p0 A + p1 B + xi_eos e^{-wT},
    //     A = 1 - (1 - e^{-wT})/(wT),  B = (1 - e^{-wT})/(wT) - e^{-wT}
    // solved for p0 and clamped to the DS box. Default off.
    let ds_feasible_ramp = cfg.double_support.feasible_ramp;
    // DS_CAPTURE: end the double support EARLY when the DCM has left the
    // double-support box along x (no pressure point inside it can hold
    // it: 0-step capture is gone, only the next step can). Measured
    // 2026-10-11 on the late-phase forward push (11 N*s at swing 0.63):
    // the foot lands short, then the nominal 0.37 s double support runs
    // while the CoM walks past the front foot, and the catch step comes
    // 0.9 s after the push -- too late (12/12 falls). The DS end moves to
    // `t0 + DS_CAPTURE_T_MIN` (not before the landed foot carries
    // DS_CAPTURE_LOAD of the weight), once per slice; the next single
    // support is then shortened by the contact_plan and its landing by
    // the MPC as usual. DS_FEASIBLE (which only lengthens, when the DS
    // pressure CAN still catch the DCM) is complementary. Default off.
    let ds_capture = cfg.double_support.capture;
    // "Hopeless": the time the DS pressure needs along x exceeds this
    // multiple of the remaining double support.
    let ds_capture_hopeless = cfg.double_support.capture_hopeless;
    // Fire only when the DCM has run AHEAD of the DS-end target (a forward
    // push). Measured 2026-10-11 (N=48, 4 conditions): without the gate the
    // forward push went 36 -> 45 but the backward one 34 -> 32 -- stepping
    // sooner backward does not help a leg that has no authority (the
    // backward failure, design note 10-10). Default on.
    let ds_capture_fwd_only = cfg.double_support.capture_fwd_only;
    // (DS_CAPTURE_LAT, the same test along y, was not adopted and is
    // removed.)
    // The forward excursion must exceed this [m] (noise floor).
    let ds_capture_margin = cfg.double_support.capture_margin;
    let ds_capture_t_min = cfg.double_support.capture_t_min;
    let ds_capture_load = cfg.double_support.capture_load;
    // Only shorten when it saves at least this much [s].
    let ds_capture_min_gain = cfg.double_support.capture_min_gain;
    let k_dcm = cfg.dcm.k_dcm;
    let (dt, omega, total_mass) = (k.dt, k.omega, k.total_mass);
    let (z_com, terrain_base) = (k.z_com, k.terrain_base);
    let (sole_half_l, sole_half_w) = (k.sole_half_l, k.sole_half_w);
    let st = obs_state(obs);
    let com = st.com;
    let com_vel = st.com_vel;
    if com_replan {
        let touchdowns_before = plan.landing().touchdowns;
        let timeouts_before = plan.landing().timeouts;
        // The clock's own retime (extend, confirm, time out) is rebuilt below
        // where it always was: by the touchdown event's commit, and by the
        // clock's second commit when it moved the end or the timeout
        // re-plan asks. Its touchdown event is taken here (read only below).
        let mut clock = plan.edit(Writer::LandingClock);
        let shift = clock.update_landing(&actual, obs.contact_fz, t, dt, out);
        let last_event = clock.landing_mut().last_event.take();
        clock.commit();
        let landing_timed_out = plan.landing().timeouts > timeouts_before;
        // The slice that just ended is the timer's clock slice, not
        // `gait.index_at(t)`: on a timeout tick the index already points
        // at the double support, whose swing side is None.
        if contact_admit_confirmed
            && (plan.landing().touchdowns > touchdowns_before || landing_timed_out)
            && let Some((ended, _, _)) = plan.landing().clock
            && let Some(sw) = plan.gait().slices[ended].support.swing()
            && (plan.footsteps().at_slice(ended).sole[sw].z - actual.sole[sw].z).abs() < contact_admit_z
        {
            ctl.confirmed_admit[sw] = Some(ended + 1);
            if debug.contact_admit {
                outln!(out, "  [adm] t={t:.3} certificate side={sw} ds_slice={} (confirm={} timeout={})", ended + 1, plan.landing().touchdowns > touchdowns_before, landing_timed_out);
            }
        }
        if let Some((li, lsw, err_xy)) = last_event
            && contact_event_shift
            && err_xy.norm() > 1e-4
        {
            // plan - actual = err  ->  move the plan by -err onto the foot.
            let mut ed = plan.edit(Writer::ContactEvent);
            ed.shift_next(li, lsw, -err_xy);
            ed.commit();
        }
        if shift != 0.0 || (ds_feasible_on_timeout && landing_timed_out) {
            if shift != 0.0 {
                stats.landing_retimes += 1;
                stats.landing_delay += -shift;
            }
            let mut clock = plan.edit(Writer::LandingClock);
            clock.touch();
            clock.commit();
            ctl.preview_next = t;
            if ds_feasible {
                let ds = plan.gait().index_at(t + 1e-6);
                if matches!(plan.gait().slices[ds].support, Support::Double) {
                    let xi_meas = dcm_of(&com, &com_vel, omega);
                    let seg = plan.dcm().index_at(t + 1e-6);
                    let target = plan.dcm().xi_eos[seg];
                    let ds_box_full = SupportBox::from_stance_yawed(
                        &actual,
                        Support::Double,
                        (sole_half_l * cop_frac, sole_half_w * cop_frac),
                        zmp_margin,
                    );
                    // The support that just ended tells which foot carries the load.
                    let prev_sw = ds
                        .checked_sub(1)
                        .and_then(|j| plan.gait().slices[j].support.swing());
                    let ds_box = match (ds_feasible_box_stance, prev_sw) {
                        (true, Some(sw)) => SupportBox::from_stance_yawed(
                            &actual,
                            Support::Single { stance: 1 - sw, swing: sw },
                            (sole_half_l * cop_frac, sole_half_w * cop_frac),
                            zmp_margin,
                        ),
                        _ => SupportBox { lo: ds_box_full.lo, hi: ds_box_full.hi },
                    };
                    // Pivot: a box point on the far side of the DCM from
                    // the target, so the DCM is pushed TOWARD the target
                    // (`xi_dot = omega (xi - p)`). Clamping the DCM itself
                    // put the pivot on the DCM when it was inside the box
                    // (gap zero, `Degenerate`) and the branch never fired
                    // -- measured 2026-09-24, 0 retimes in 384 runs.
                    let nominal = plan.gait().slices[ds].duration();
                    let p_continuous = |tt: f64| -> na::Vector2<f64> {
                        let d = (-omega * tt).exp();
                        ds_box.clamp(&((xi_meas - target * d) / (1.0 - d))).0
                    };
                    let p = if ds_feasible_continuous {
                        p_continuous(nominal)
                    } else {
                        ds_box.clamp(&(xi_meas * 2.0 - target)).0
                    };
                    let mut need = 0.0_f64;
                    for axis in 0..2 {
                        if let StepTiming::Time(tt) = step_timing(&xi_meas, &target, &p, omega, axis) {
                            need = need.max(tt);
                        }
                    }
                    let want = need.clamp(nominal, ds_feasible_max);
                    if ds_feasible_continuous {
                        if want > nominal + 1e-3 {
                            let mut ed = plan.edit(Writer::DsFeasible);
                            let dd = ed.retime(ds, ed.gait().slices[ds].t0 + want);
                            stats.n_ds_feasible += 1;
                            stats.ds_feasible_total += dd.abs();
                            ed.commit();
                        }
                        // The double support's ZMP onto the BUILT reference,
                        // after the commit. KNOWN LEAK (plan.rs, §7 of the
                        // extraction plan, item 8): the next commit that
                        // rebuilds -- any writer's, this tick or later --
                        // drops it. Kept as it is (changing it changes the
                        // walk).
                        let seg_ds = plan.dcm().index_at(plan.gait().slices[ds].t0 + 1e-6);
                        let p_final = if ds_feasible_ramp && seg_ds + 1 < plan.dcm().segs.len() {
                            let tt = plan.gait().slices[ds].duration();
                            let wt = omega * tt;
                            let d = (-wt).exp();
                            let b = (1.0 - d) / wt - d;
                            let a = 1.0 - (1.0 - d) / wt;
                            let p1 = plan.dcm().segs[seg_ds + 1].p0;
                            let p0 = ds_box.clamp(&((xi_meas - target * d - p1 * b) / a)).0;
                            plan.set_segment_zmp_ramp(seg_ds, p0, p1);
                            p0
                        } else {
                            let p = p_continuous(plan.gait().slices[ds].duration());
                            plan.set_segment_zmp(seg_ds, p);
                            p
                        };
                        stats.n_ds_continuous += 1;
                        if debug.ds_feasible {
                            let r = plan.dcm().reference(plan.gait().slices[ds].t0 + 1e-6);
                            outln!(out, "  [dsc] t={t:.3} ds={ds} xi=({:.3},{:+.3}) target=({:.3},{:+.3}) p=({:.3},{:+.3}) T={:.3} xi_ref(t0)=({:.3},{:+.3})", xi_meas.x, xi_meas.y, target.x, target.y, p_final.x, p_final.y, plan.gait().slices[ds].duration(), r.xi.x, r.xi.y);
                        }
                    } else if want > nominal + 1e-3 {
                        let mut ed = plan.edit(Writer::DsFeasible);
                        let dd = ed.retime(ds, ed.gait().slices[ds].t0 + want);
                        stats.n_ds_feasible += 1;
                        stats.ds_feasible_total += dd.abs();
                        ed.commit();
                    }
                    if debug.ds_feasible {
                        outln!(out, "  [dsf] t={t:.3} ds={ds} xi=({:.3},{:+.3}) target=({:.3},{:+.3}) p=({:.3},{:+.3}) need={need:.3} nominal={nominal:.3} -> {want:.3}", xi_meas.x, xi_meas.y, target.x, target.y, p.x, p.y);
                    }
                }
            }
        }
        if ds_capture {
            let ds = plan.gait().index_at(t + 1e-6);
            let is_ds = |j: usize| matches!(plan.gait().slices[j].support, Support::Double);
            // Not while a recovery reference owns the phase durations.
            let ref_owns = claim_check!(plan.claims().whole_owner() == Some(Writer::RecoveryRef), ctl.ref_mode.is_some());
            let run_done = plan.claims().ds_run_claimed(ds);
            if is_ds(ds) && !ref_owns && !run_done {
                // The whole run of double-support slices the clock is in
                // (a DS is split into ramp / hold / ramp slices).
                let mut first = ds;
                while first > 0 && is_ds(first - 1) {
                    first -= 1;
                }
                let mut last = ds;
                while last + 1 < plan.gait().slices.len() && is_ds(last + 1) {
                    last += 1;
                }
                let t0_run = plan.gait().slices[first].t0;
                let t1_run = plan.gait().slices[last].t1;
                let remaining = t1_run - t;
                let xi_meas = dcm_of(&com, &com_vel, omega);
                let seg_last = plan.dcm().index_at(t1_run - 1e-6);
                let target = plan.dcm().xi_eos[seg_last];
                let ds_box = SupportBox::from_stance_yawed(
                    &actual,
                    Support::Double,
                    (sole_half_l * cop_frac, sole_half_w * cop_frac),
                    zmp_margin,
                );
                // The pivot that would bring the DCM to the DS-end target
                // in the remaining time, clamped to the box; the time it
                // then actually needs along x (DS_FEASIBLE's own test).
                let dcy = (-omega * remaining.max(1e-3)).exp();
                let p = ds_box.clamp(&((xi_meas - target * dcy) / (1.0 - dcy))).0;
                // Degenerate = the DCM is already at the target (or the
                // pivot sits on it): nothing to do, NOT hopeless. Measured
                // 2026-10-11 (7@90 m19): a 0.1 mm gap was read as
                // unreachable and a healthy double support lost 300 ms.
                let need = match step_timing(&xi_meas, &target, &p, omega, 0) {
                    StepTiming::Time(tt) => tt,
                    StepTiming::Degenerate => 0.0,
                    StepTiming::Unreachable => f64::INFINITY,
                };
                let hopeless_x = remaining > 1e-3
                    && need > remaining * ds_capture_hopeless
                    && (!ds_capture_fwd_only || xi_meas.x > target.x + ds_capture_margin);
                let hopeless = hopeless_x;
                let prev_sw = first.checked_sub(1).and_then(|j| plan.gait().slices[j].support.swing());
                let landed_load = prev_sw.map_or(0.0, |sw| ctl.fz_prev[sw]);
                let earliest = (t0_run + ds_capture_t_min).max(t + 2.0 * dt);
                if debug.ds_capture {
                    outln!(out, 
                        "  [dsc-cap?] t={t:.3} run {first}..{last} [{t0_run:.3},{t1_run:.3}] xi.x={:.3} target.x={:.3} p.x={:.3} need={need:.3} rem={remaining:.3} hopeless={hopeless} load={landed_load:.0} earliest={earliest:.3}",
                        xi_meas.x, target.x, p.x
                    );
                }
                if hopeless
                    && landed_load >= ds_capture_load * total_mass * G
                    && earliest < t1_run - ds_capture_min_gain
                {
                    let f = (earliest - t) / remaining;
                    let targets: Vec<(usize, f64)> =
                        (ds..=last).map(|j| (j, t + (plan.gait().slices[j].t1 - t) * f)).collect();
                    let mut ed = plan.edit(Writer::DsCapture);
                    for (j, nt) in targets {
                        stats.ds_capture_total += ed.retime(j, nt).abs();
                    }
                    ed.claim_ds_run(last);
                    stats.n_ds_capture += 1;
                    ed.commit();
                    if debug.ds_capture {
                        outln!(out, "  [dsc-cap] t={t:.3} DS end {t1_run:.3} -> {earliest:.3}");
                    }
                }
            }
        }
        // (COM_REPLAN_ALWAYS, engaging before the first landing clock, is
        // removed.)
        let engaged = plan.claims().done(Writer::ReplanStart);
        if !engaged && plan.landing().clock.is_some() {
            let cmd_ref = plan.dcm().reference(t);
            ctl.preview_p = commanded_zmp(
                &dcm_of(&com, &com_vel, omega),
                &cmd_ref,
                omega,
                k_dcm,
            );
            let mut ed = plan.edit(Writer::ReplanStart);
            *ed.height_mut() = crate::preview::HeightPlan::constant(
                z_com + ctl.terrain_track - terrain_base,
            );
            ed.claim_once();
            ed.commit();
        }
    }
    // LAND_REANCHOR [m]: the plan onto the foot that just landed. Measured
    // 2026-10-31 on kyo46rs (0.9 N*s toward the stance side, early phase):
    // the pushed swing lands 2-4 cm wide of its target (the WBC uses the
    // swing leg for the trunk while the stance CoP is at its edge), and the
    // plan keeps that foot where it was meant to land -- the DCM reference
    // and the ZMP box of the next single support then sit 4 cm inside the
    // real stance sole, the CoP saturates at its inner edge and the next
    // swing is flung inward into the stance leg. Once per double support,
    // when the landed foot carries LAND_REANCHOR_LOAD of the weight, its
    // planned position (until it lifts again, `shift_next`) moves onto the
    // measured sole if they differ by more than LAND_REANCHOR and less than
    // LAND_REANCHOR_MAX; not on a known staircase (its zone). 0 = off.
    let reanchor = cfg.landing.reanchor;
    if reanchor > 0.0 {
        let i = plan.gait().index_at(t + 1e-6);
        let is_ds = |j: usize| matches!(plan.gait().slices[j].support, Support::Double);
        if is_ds(i) {
            let mut first = i;
            while first > 0 && is_ds(first - 1) {
                first -= 1;
            }
            let in_stair_zone = k.stairs.as_ref().is_some_and(|st| {
                (st.riser.abs() > 1e-6 || st.down_n > 0)
                    && crate::height::stair_zone_crouch(
                        plan.footsteps().at_slice(i).mid_xy().x,
                        st,
                        cfg.stair_zone.lead,
                        cfg.stair_zone.tail,
                        1.0,
                    ) > 0.0
            });
            if ctl.reanchor_ds != Some(first)
                && !in_stair_zone
                && let Some(sw) = first.checked_sub(1).and_then(|j| plan.gait().slices[j].support.swing())
                && obs.contact_fz[sw] >= cfg.landing.reanchor_load * total_mass * G
            {
                ctl.reanchor_ds = Some(first);
                let planned = plan.footsteps().at_slice(i).xy(sw);
                let d = actual.sole[sw].xy() - planned;
                if d.norm() > reanchor && d.norm() < cfg.landing.reanchor_max {
                    let mut ed = plan.edit(Writer::ContactEvent);
                    ed.shift_next(i, sw, d);
                    ed.commit();
                    stats.n_reanchor += 1;
                    if debug.replan {
                        outln!(out, "  [reanchor] t={t:.3} ds {first} foot {sw} plan ({:.3},{:+.3}) -> sole ({:.3},{:+.3})", planned.x, planned.y, actual.sole[sw].x, actual.sole[sw].y);
                    }
                }
            }
        }
    }
}

/// What phase 3 found: the schedule's slice now, the support (all `Double`
/// with NO_LIFT), the steps taken, the planned soles of this slice, and the
/// swing's rotation mode (the stair zone's own inside it).
pub(crate) struct SupportOut {
    pub slice: Slice,
    pub slice_frac: f64,
    pub support: Support,
    pub step_idx: usize,
    pub slice_idx: usize,
    pub steps: Footsteps,
    pub swing_rot_mode: u32,
}

/// Phase 3: the schedule -> the support, and the contact anchors released at
/// a lift-off and set at a touchdown (recorded in `anchor_events`; the stack
/// applies them before its contact level).
pub(crate) fn phase_support(
    ctl: &mut CtlState,
    plan: &Plan,
    anchor_events: &mut Vec<AnchorEvent>,
    cfg: &Config,
    k: &WalkerConsts,
    obs: &TickObs,
    t: f64,
    out: &mut dyn LogSink,
) -> SupportOut {
    let no_lift = cfg.gait.no_lift;
    let swing_rot_mode = cfg.swing.rot_mode;
    // STAIR_ZONE_SWING_ROT=mode: the SWING_ROT mode used only while the
    // mid-feet point is inside a known staircase's zone (the same zone as
    // STAIR_ZONE_CROUCH, including its lead/tail). Measured 2026-10-20 on
    // kyo46rs (N=24): holding the swing foot's pitch (mode 3) is what lets
    // it take the 0.13-0.16 m stair step (8 -> 12 of 12) and the 10 mm
    // descent (1 -> 24 of 24 with the zone crouch), but applied everywhere
    // it costs the 0.9 N.s forward push (21 -> 16). 0 = off = SWING_ROT
    // everywhere, bit-identical.
    let stair_zone_swing_rot = cfg.swing.zone_rot_mode;
    let stair_crouch_lead = cfg.stair_zone.lead;
    let stair_crouch_tail = cfg.stair_zone.tail;
    let debug = cfg.debug;
    let robot = &k.robot;
    let prof = &robot.prof;
    let stairs = &k.stairs;
    let data = &obs_state(obs).data;
    // By value, not by reference: `Slice` is `Copy`, and holding a
    // borrow on `gait` here is what stops the step-timing adaptation
    // below from retiming it.
    let (slice, slice_frac) = {
        let (sl, f) = plan.gait().at(t);
        (*sl, f)
    };
    let support = if no_lift {
        Support::Double
    } else {
        slice.support
    };
    let step_idx = plan.gait().steps_taken(t.max(1e-9));
    let slice_idx = plan.gait().index_at(t);
    // Everything below reads the footsteps for THIS slice. With stride 0
    // they are the same pair every time and this is the old behaviour.
    let steps = *plan.footsteps().at_slice(slice_idx);
    let swing_rot_mode = if stair_zone_swing_rot > 0
        && stairs.as_ref().is_some_and(|st| {
            crate::height::stair_zone_crouch(steps.mid_xy().x, st, stair_crouch_lead, stair_crouch_tail, 1.0) > 0.0
        }) {
        stair_zone_swing_rot
    } else {
        swing_rot_mode
    };

    // Touchdown and lift-off are the only two events that may edit an
    // anchor. Doing it on the schedule (rather than lazily, the way
    // standing did with get_or_insert) is what makes a foot that has been
    // somewhere else since stop arguing for where it used to be.
    if support != ctl.prev_support {
        match (ctl.prev_support, support) {
            (Support::Double, Support::Single { swing, .. }) => {
                anchor_events.push(AnchorEvent::Release(swing));
                let p = misarta::se3::translation(&data.oMi[robot.foot_mi[swing]]);
                ctl.swing_lift_off = Some(p);
            }
            (Support::Single { swing, .. }, Support::Double) => {
                // The foot has landed somewhere other than nominal, so
                // every future segment is relative to where it ACTUALLY
                // is. Rebuilding the DCM plan here is safe rather than
                // discontinuous: a segment's influence on the present
                // reference decays by exp(-omega*T) per step -- about a
                // factor of 7 at these timings -- so a few mm one step
                // ahead moves the current reference by a fraction of a
                // millimetre.
                // Re-anchor the landed foot where it ACTUALLY is, not
                // where it was planned to be: the anchor's job is to stop
                // drift from here, and seeding it with a position the
                // foot is not at demands a force to close a gap that the
                // contact cannot close.
                let mi = robot.foot_mi[swing];
                let mut p = misarta::se3::translation(&data.oMi[mi]);
                // Rotation from the PLAN (flat sole at the planned yaw),
                // not from the measurement. The anchor's rotation is
                // deliberately leak-free ("roll cannot be conceded"), so
                // anchoring a toe-first landing at its measured pitch
                // HOLDS the foot on its toe -- measured on the descent,
                // the sole stayed at +27..+47 deg after touchdown, the
                // robot vaulted over the toe and fell forward. Anchoring
                // flat makes the Baumgarte roll the heel down instead,
                // which is what a toe-first landing is FOR.
                // (STAIR_ANCHOR_FLAT, the plan rotation, is removed.)
                let r = misarta::se3::rotation_matrix(&data.oMi[mi]);
                // x, y and orientation from the measurement -- but z on
                // the WALKING SURFACE (the plan's sole height), same as
                // the contact-driven branch below. Anchoring a foot at
                // its measured z was fine when landings finished on
                // time; a stair swing that is still 20 mm up at the
                // slice boundary was being anchored IN MID-AIR, held
                // there by Baumgarte, and never loaded -- the double
                // support then had one real foot, and the next lift-off
                // fell. Measured: the entire knife-edge sensitivity of
                // the 4-step recipe was "does this landing finish
                // before the boundary".
                let p_meas_z = p.z;
                p.z = steps.sole[swing].z + prof.sole_below_origin;
                if debug.anchor {
                    errln!(out, 
                        "  [anchor] t={t:.3} swing {} touchdown: meas z {:.4} -> plan z {:.4}",
                        if swing == 0 { "L" } else { "R" },
                        p_meas_z,
                        p.z
                    );
                }
                anchor_events.push(AnchorEvent::Touchdown { side: swing, pos: p, rot: r });
                ctl.swing_lift_off = None;
            }
            _ => {}
        }
        ctl.prev_support = support;
    }
    SupportOut { slice, slice_frac, support, step_idx, slice_idx, steps, swing_rot_mode }
}

/// The double support's load ramp: the foot that just landed takes its share
/// on over `ramp_frac` of the slice, the foot about to lift gives it up.
pub(crate) struct LoadRamp {
    pub landing: Option<usize>,
    pub lifting: Option<usize>,
    slice_frac: f64,
    ramp_frac: f64,
    foot_mi: [usize; 2],
}

impl LoadRamp {
    fn f_share(&self, side: usize) -> f64 {
        if let Some(l) = self.landing {
            if l == side {
                return (self.slice_frac / self.ramp_frac).clamp(0.0, 1.0);
            }
        }
        if let Some(l) = self.lifting {
            if l == side {
                return ((1.0 - self.slice_frac) / self.ramp_frac).clamp(0.0, 1.0);
            }
        }
        1.0
    }

    /// Load share of the foot whose model index is `mi` (floored at 2%).
    pub fn load_share(&self, mi: usize) -> f64 {
        let side = usize::from(mi != self.foot_mi[0]);
        self.f_share(side).max(0.02)
    }
}

/// What phase 4 found: the measured vertical forces as the controller read
/// them, the feet the QP solves with as contacts (sides, model indices,
/// count), and the load ramp.
pub(crate) struct ContactsOut {
    pub fz_meas: [f64; 2],
    pub stance_sides: Vec<usize>,
    pub stance: Vec<usize>,
    pub nc: usize,
    pub ramp: LoadRamp,
}

/// Phase 4: the managed-contact rate trigger, the contact set (the measured
/// correction, the scheduled / loaded / confirmed admissions, the fresh
/// anchors) and the double support's load ramp.
#[allow(clippy::too_many_arguments)]
pub(crate) fn phase_contacts(
    ctl: &mut CtlState,
    plan: &Plan,
    stats: &mut WalkerStats,
    anchor_events: &mut Vec<AnchorEvent>,
    cfg: &Config,
    k: &WalkerConsts,
    obs: &TickObs,
    sup: &SupportOut,
    t: f64,
    out: &mut dyn LogSink,
) -> ContactsOut {
    // Trigger Managed on the RATE of the normal force, not only its level.
    // Measured on the fixed-arm stair transfer: during the chatter fz swings
    // 15 <-> 78 N, so a level test catches only the dips and leaves the
    // contact rigid through the rest of the oscillation. The chatter runs at
    // ~1200 N/s against ~200 N/s for an intended transfer, so the rate
    // separates them. 0 = off (shipped behaviour).
    let stair_managed_rate = cfg.managed.rate;
    // Hold Managed this long once tripped, so the trigger does not itself
    // chatter at the rate it is suppressing.
    let stair_managed_latch = cfg.managed.latch;
    // Solve against the contact set the FEET report, not the one the schedule
    // assumes. CONTACT_DRIVEN=0 restores the schedule-only behaviour for A/B.
    let contact_driven = cfg.contact.driven;
    let contact_admit_scheduled = cfg.contact.admit_scheduled;
    let contact_admit_z = cfg.contact.admit_z;
    // CONTACT_ADMIT_LOADED="z_tol,fz_frac": a scheduled landing whose sole
    // is OFF its planned height by more than CONTACT_ADMIT_Z but within
    // z_tol, and which measures at least fz_frac of body weight, is admitted
    // too, anchored at its MEASURED height. Measured 2026-10-15 on the
    // unmodelled-bump bench: a foot landing on a 12 mm bump failed the 10 mm
    // gate, carried 27 N (under the 10% BW hysteresis) for the whole double
    // support while the QP balanced on the rear foot (n_stance = 1), the DCM
    // error grew, the next swing was retimed to 0.2 s and stomped at 1860 N.
    // Unset = the 10 mm gate alone (bit-identical).
    let contact_admit_loaded = cfg.contact.admit_loaded;
    let debug = cfg.debug;
    let contact_admit_confirmed = cfg.contact.admit_confirmed;
    let no_lift = cfg.gait.no_lift;
    // Double-support load ramp: the fraction of the double support over
    // which the landed foot takes the load.
    let ramp_frac = cfg.contact.load_ramp_frac;
    let (dt, total_mass) = (k.dt, k.total_mass);
    let robot = &k.robot;
    let prof = &robot.prof;
    let data = &obs_state(obs).data;
    let (support, steps, slice_frac) = (sup.support, &sup.steps, sup.slice_frac);
    let fz_meas = [obs.contact_fz[0], obs.contact_fz[1]];
    if stair_managed_rate > 0.0 {
        for side in 0..2 {
            // EMA over ~4 ticks: one noisy sample must not trip it.
            let r = (fz_meas[side] - ctl.fz_prev[side]).abs() / dt.max(1e-9);
            ctl.fz_rate_ema[side] += 0.25 * (r - ctl.fz_rate_ema[side]);
            if ctl.fz_rate_ema[side] > stair_managed_rate {
                ctl.managed_until[side] = t + stair_managed_latch;
            }
        }
    }
    ctl.fz_prev = fz_meas;
    let stance_sides: Vec<usize> = if contact_driven {
        let (mut sides, mut fresh) = ctl.correction.update(support, fz_meas);
        // Planned admission on ANY scheduled touchdown, flat included:
        // entering a double support, a foot the schedule now calls
        // stance whose sole is within CONTACT_ADMIT_Z of its planned
        // height is certified for that double support. Measured
        // 2026-09-24 under the weighted QP: a gentle flat touchdown
        // carried 5 N, stayed under the 10% BW hysteresis for the whole
        // double support, and the QP balanced on the rear foot alone.
        // (CONTACT_ADMIT_SCHEDULED_OFF_STAIRS, only outside the stair
        // zone, is removed.)
        if contact_admit_scheduled
            && matches!(support, Support::Double)
        {
            let now_slice = plan.gait().index_at(t);
            if let Some(prev) = now_slice.checked_sub(1)
                && let Some(landed) = plan.gait().slices[prev].support.swing()
                && ctl.confirmed_admit[landed].is_none()
                && !sides.contains(&landed)
                && ctl.last_scheduled_admit[landed] != Some(now_slice)
            {
                let mi = robot.foot_mi[landed];
                let sole_z = misarta::se3::translation(&data.oMi[mi]).z - prof.sole_below_origin;
                let dz_plan = (sole_z - steps.sole[landed].z).abs();
                if dz_plan < contact_admit_z {
                    ctl.confirmed_admit[landed] = Some(now_slice);
                    ctl.last_scheduled_admit[landed] = Some(now_slice);
                    ctl.admit_measured_z[landed] = None;
                } else if let Some((z_tol, fz_frac)) = contact_admit_loaded
                    && dz_plan < z_tol
                    && fz_meas[landed] >= fz_frac * total_mass * G
                {
                    ctl.confirmed_admit[landed] = Some(now_slice);
                    ctl.last_scheduled_admit[landed] = Some(now_slice);
                    ctl.admit_measured_z[landed] = Some(sole_z);
                    if debug.contact_admit {
                        outln!(out, "  [adm-loaded] t={t:.3} side={landed} sole_z={sole_z:.4} plan_z={:.4} fz={:.0}", steps.sole[landed].z, fz_meas[landed]);
                    }
                }
            }
        }
        if contact_admit_confirmed || contact_admit_scheduled {
            let now_slice = plan.gait().index_at(t);
            for side in 0..2 {
                let Some(ds_slice) = ctl.confirmed_admit[side] else { continue };
                if now_slice > ds_slice {
                    // The double support after the landing is over.
                    ctl.confirmed_admit[side] = None;
                    continue;
                }
                // Only inside that double support (not on the confirm
                // tick itself, whose schedule may still say "swing").
                // Re-admit as often as the hysteresis drops the foot:
                // measured 2026-09-24, a confirmed landing is admitted
                // by its own 106 N impact, bounces under the 2% BW
                // release threshold two ticks later, and then rests at
                // 30 N -- under the 10% BW re-entry -- for the rest of
                // the double support. The lifting foot at the END of a
                // later double support is not covered by this: its
                // certificate expired with its own landing's DS.
                if now_slice == ds_slice
                    && support.is_stance(side)
                    && !sides.contains(&side)
                    && ctl.correction.admit(side)
                {
                    sides.push(side);
                    sides.sort_unstable();
                    fresh.push(side);
                    stats.n_confirmed_admissions += 1;
                    if debug.contact_admit {
                        let mi = robot.foot_mi[side];
                        let sole_z = misarta::se3::translation(&data.oMi[mi]).z - prof.sole_below_origin;
                        outln!(out, 
                            "  [adm] t={t:.3} side={side} sole_z={sole_z:.4} plan_z={:.4} fz=({:.0},{:.0}) support={support:?}",
                            steps.sole[side].z, fz_meas[0], fz_meas[1]
                        );
                    }
                }
            }
        }
        for side in fresh {
            // A foot that has just become load-bearing needs a fresh
            // anchor: inheriting the stale one demands a force to close a
            // gap the contact cannot close.
            //
            // Take x, y and orientation from the measurement, but put z on
            // the GROUND. A contact anchor describes where the foot meets
            // the floor, and admitting a foot that is still 3-5 mm up --
            // which is exactly what a bouncing touchdown looks like --
            // anchors it in mid-air, so the Baumgarte term then works to
            // HOLD IT THERE at kp_c = 1600 while the QP plans a reaction
            // that does not exist. Measured, that is what turned a landed
            // foot into a hovering one for the rest of the run.
            let mi = robot.foot_mi[side];
            let mut p = misarta::se3::translation(&data.oMi[mi]);
            // A landing admitted on an unmodelled surface (see
            // CONTACT_ADMIT_LOADED) is anchored where it actually is.
            p.z = ctl.admit_measured_z[side].unwrap_or(steps.sole[side].z) + prof.sole_below_origin;
            let rr = misarta::se3::rotation_matrix(&data.oMi[mi]);
            anchor_events.push(AnchorEvent::Touchdown { side, pos: p, rot: rr });
        }
        let sched: Vec<usize> = (0..2).filter(|&s| support.is_stance(s)).collect();
        if sides != sched {
            stats.n_corrected += 1;
        }
        sides
    } else {
        (0..2).filter(|&s| support.is_stance(s)).collect()
    };
    let stance: Vec<usize> = stance_sides.iter().map(|&s| robot.foot_mi[s]).collect();
    let nc = stance.len();

    // Load ramps across double support. The foot that just landed takes
    // its share on gradually and the foot about to lift gives it up
    // gradually: the CoP box is |m| <= L*fz, so ramping fz is the only
    // way to shrink a foot's authority without a step change in the
    // constraint set.
    let (landing, lifting) = if matches!(support, Support::Double) && !no_lift {
        let i = plan.gait().index_at(t);
        let landed = if i > 0 {
            plan.gait().slices[i - 1].support.swing()
        } else {
            None
        };
        let next = plan.gait().slices.get(i + 1).and_then(|s| s.support.swing());
        (landed, next)
    } else {
        (None, None)
    };
    // (LOAD_RAMP_TD, keeping the share the landed foot already carries,
    // and LOAD_RAMP_CONTACT_T, ramping from the contact-entry tick too,
    // were not adopted and are removed.)
    let ramp = LoadRamp { landing, lifting, slice_frac, ramp_frac, foot_mi: robot.foot_mi };
    ContactsOut { fz_meas, stance_sides, stance, nc, ramp }
}

/// Which lateral direction is "inward" (toward the stance foot) for the
/// landing of `swing` planned at slice `i`: returns (lateral gap, true when
/// inward is +y). The gap is |stance.y - landing.y| in world y.
fn landing_inward(
    footsteps: &crate::gait::FootstepPlan,
    i: usize,
    swing: usize,
) -> (f64, bool) {
    let f = footsteps.at_slice(i);
    let stance = 1 - swing;
    let dy = f.sole[stance].y - f.sole[swing].y;
    (dy.abs(), dy > 0.0)
}

/// What phase 5 found: the measured DCM, the DCM reference (a landed
/// recovery reference's own when one is engaged) and the error between them.
pub(crate) struct DcmOut {
    pub xi: na::Vector2<f64>,
    pub r: DcmRef,
    pub dcm_err: f64,
    /// The DCM error the push triggers read: `xi - r.xi`, low-passed with
    /// PUSH_TRIG_LPF_HZ (unfiltered when off, so `trig_err.norm()` is then
    /// `dcm_err` exactly).
    pub trig_err: na::Vector2<f64>,
}

/// Phase 5: the DCM and its reference, then the plan's push responses in
/// their order: PUSH_YIELD (slow the walk), PUSH_REPLAN (re-lay the landing
/// after a backward push), PUSH_Z_DROP (detect; the drop is phase 8's) and
/// the engagement of a recovery reference.
#[allow(clippy::too_many_arguments)]
pub(crate) fn phase_dcm(
    ctl: &mut CtlState,
    plan: &mut Plan,
    stats: &mut WalkerStats,
    cfg: &Config,
    k: &WalkerConsts,
    obs: &TickObs,
    actual: &Footsteps,
    sup: &SupportOut,
    t: f64,
    out: &mut dyn LogSink,
) -> DcmOut {
    // PUSH_YIELD_V [m/s]: when a push is detected during a speed command
    // (DCM error above PUSH_YIELD_TRIG), re-plan the remaining steps at this
    // lower speed with the schedule's stride cap and step time, once per
    // run. Measured 2026-10-16: the pushes at 0.50 m/s are lost 2-3 steps
    // after the push through the lateral authority, and the same pushes at
    // 0.30 survive 133/192. 0 = off (bit-identical).
    let push_yield_v = cfg.speed.yield_v;
    let vx2_sched = cfg.speed.sched;
    let push_yield_vmin = cfg.speed.yield_vmin;
    let push_yield_trig = cfg.speed.yield_trig;
    let push_yield_no_fwd = cfg.speed.yield_no_fwd;
    let vx2_tscale_min = cfg.speed.tscale_min;
    let debug = cfg.debug;
    let push_replan_trig = cfg.push_replan.trig;
    let stair_crouch_lead = cfg.stair_zone.lead;
    let stair_crouch_tail = cfg.stair_zone.tail;
    let push_replan_slope_lead = cfg.push_replan.slope_lead;
    let push_replan_freeze = cfg.push_replan.freeze;
    // PUSH_REPLAN_ANKLE="ax,ay": fraction of the sole's half length / width
    // the ankle is credited with holding before the error is carried forward.
    // Measured 2026-10-21: crediting the full 85 mm half length took the
    // backward 14 N.s fallers from 8/12 to 0/12 (a heel CoP does not hold a
    // backward push); laterally it keeps a small error from reading as a 1 m
    // step. Default 0 / 1.
    let push_replan_ankle = cfg.push_replan.ankle;
    // PUSH_REPLAN_BACK_ONLY=1: engage only on a backward-dominant error
    // (behind, and more than sideways). PUSH_REPLAN_VMAX [m/s]: never while a
    // VX2 speed command above this is in force (the speed yield re-plans
    // there). Measured 2026-10-21 (tu42, all directions): backward 14 N.s
    // 9 -> 42, but forward / lateral 3-5 members lost each and 0.50 m/s
    // backward 44 -> 21.
    let push_replan_back_only = cfg.push_replan.back_only;
    let push_replan_vmax = cfg.push_replan.vmax;
    // The current DCM error must also exceed this [m] (a 2 mm error at the
    // start of a long first support carried 0.8 s forward read as 4 cm).
    let push_replan_emin = cfg.push_replan.emin;
    let push_replan_max_back = cfg.push_replan.max_back;
    let push_replan_max_fwd = cfg.push_replan.max_fwd;
    let push_replan_min_w = cfg.push_replan.min_w;
    // Widest outward step [m] from the stance foot.
    let push_replan_max_w = cfg.push_replan.max_w;
    let push_replan_stand = cfg.push_replan.stand;
    let push_replan_ramp = cfg.push_replan.ramp;
    let push_z_drop = cfg.push_crouch.drop;
    let push_z_drop_trig = cfg.push_crouch.trig;
    let push_z_drop_lat_max = cfg.push_crouch.lat_max;
    let push_z_drop_t = cfg.push_crouch.t;
    // No reference when the push lands this late in the swing (the walker
    // copes on its own there; measured in-place 5 N s at phase 0.7: 3/3).
    let ref_trig_max_frac = cfg.recovery.trig_max_frac;
    // Detector: lateral DCM error toward the stance side [m] that engages a
    // reference (0 = off). (The bench trigger at the push, REF_TRIG_AT_PUSH /
    // REF_TRIG_MIN_IMPULSE -- an oracle a robot does not have -- and the
    // forward-push detector REF_TRIG_DCM_X are removed.)
    let ref_trig_dcm = cfg.recovery.trig_dcm;
    // 0 = enter the reference at its start even when the detector fired late
    // (for references solved with the detection delay in their initial state).
    let ref_delay_comp = cfg.recovery.delay_comp;
    let omega = k.omega;
    let (sole_half_l, sole_half_w) = (k.sole_half_l, k.sole_half_w);
    let (vx, t_step, gp, stairs) = (k.vx, k.t_step, &k.gp, &k.stairs);
    let (vx2_change, vx2_cmd_speed, vx2_z_from) = (k.vx2_change, k.vx2_cmd_speed, k.vx2_z_from);
    let vxs_cap_of = |v: f64| -> f64 { cfg.speed.stride_cap(v) };
    let ref_libs = &k.ref_libs[..];
    let ref_any = !ref_libs.is_empty();
    let st = obs_state(obs);
    let (com, com_vel) = (st.com, st.com_vel);
    let (support, slice_frac, slice_idx) = (sup.support, sup.slice_frac, sup.slice_idx);
    // (DCM_LEAD_S, a sensor-delay lead on the measured DCM, is removed.)
    let xi = dcm_of(&com, &com_vel, omega);
    let mut r = plan.dcm().reference(t);
    // For a LANDED (re-step) reference the DCM law tracks the
    // reference's own DCM and CoP (its CoM and contact forces) instead of
    // the re-planned LIPM. Measured 2026-10-13 on ld7_a: without it the
    // walker's CoM x lagged the reference by 6 cm at the second landing
    // and 12 cm at the end and the robot fell backward; with it the CoM
    // tracks within 3 cm. (REF_WBC_LANDED, default on, is removed with
    // REF_WBC; the detectors only pick a landed reference when the
    // library has no other.)
    if let (Some(mode), Some(lib)) = (ctl.ref_mode, ctl.ref_lib.as_ref())
        && lib.landed
    {
        let tau = t - mode.t0 + mode.tau0;
        if tau <= lib.total() {
            let xi = mode.world_xy(lib, lib.dcm_at(tau, omega));
            let xi1 = mode.world_xy(lib, lib.dcm_at(tau + 0.01, omega));
            r.xi = xi;
            r.xi_dot = (xi1 - xi) / 0.01;
            r.p = mode.world_xy(lib, lib.cop_at(tau));
        }
    }
    let dcm_err = (xi - r.xi).norm();
    // PUSH_TRIG_LPF_HZ: the triggers' error, low-passed. Measured 2026-11-01
    // on kyo46rs (v3 + the G1 push keys): with the sensor noise of the
    // family's large-noise condition (velocity 0.03, joint rate 0.1) a
    // single tick's DCM error exceeds the MPC landing's 7.5 mm trigger 42 %
    // of the time (0 % without noise, 1.4 % under a 0.9 N*s push); the
    // landing variable armed on every member (median 113 times) and
    // PUSH_REPLAN engaged on noise, and the condition fell 2 better / 41
    // worse and 3 / 30. Filtering the error, not the DCM, keeps the
    // reference's own motion out of the lag.
    let trig_err = {
        let e = xi - r.xi;
        let hz = cfg.dcm.trig_lpf_hz;
        if hz > 0.0 {
            let a = k.dt / (1.0 / (2.0 * std::f64::consts::PI * hz) + k.dt);
            let f = ctl.trig_err_f.get_or_insert(e);
            *f += a * (e - *f);
            *f
        } else {
            e
        }
    };
    if push_yield_v > 0.0
        && !plan.claims().done(Writer::PushYield)
        && vx2_sched
        && let Some(v_hi) = vx2_cmd_speed
        && v_hi > push_yield_v
        // PUSH_YIELD_VMIN: only from this commanded speed up (at 0.30 m/s
        // the lowering alone handles the pushes and the yield cost the
        // backward 11 N.s 48 -> 46, 2026-10-16). 0 = any speed.
        && v_hi >= push_yield_vmin
        && vx2_z_from.is_some_and(|t0| t >= t0)
        && dcm_err > push_yield_trig
        // PUSH_YIELD_NO_FWD=1: never yield on a FORWARD error. Measured
        // 2026-10-16 at 0.50 m/s (N=48): yielding to 0.20 took the
        // backward 11 N.s from 19 to 44 but the forward 11 N.s from 45
        // to 34 -- slowing the plan under a forward push leaves the body
        // ahead of it.
        && !(push_yield_no_fwd && (xi.x - r.xi.x) > 0.5 * push_yield_trig)
    {
        let mut ed = plan.edit(Writer::PushYield);
        ed.claim_once();
        let v_lo = push_yield_v;
        let stride_hi = (v_hi * t_step).min(vxs_cap_of(v_hi));
        let stride_lo = (v_lo * t_step).min(vxs_cap_of(v_lo));
        let scale_hi = (stride_hi / (v_hi * t_step)).clamp(vx2_tscale_min, 1.0);
        let scale_lo = (stride_lo / (v_lo * t_step)).clamp(vx2_tscale_min, 1.0);
        let ratio = scale_lo / scale_hi;
        // Every future landing advances by stride_lo instead of
        // stride_hi: the n-th future step is pulled back n * delta.
        let delta = stride_hi - stride_lo;
        let i_now = ed.gait().index_at(t);
        let mut n = 0usize;
        let mut shift = [0.0_f64; 2];
        for j in (i_now + 1)..ed.gait().slices.len() {
            if let Some(sw) = ed.gait().slices[j].support.swing() {
                n += 1;
                shift[sw] = n as f64 * delta;
            }
            if j < ed.footsteps().at.len() {
                for side in 0..2 {
                    ed.footsteps_mut().at[j].sole[side].x -= shift[side];
                }
            }
        }
        // And every future slice runs at the slower speed's duration.
        for j in (i_now + 1)..ed.gait().slices.len() {
            let t0 = ed.gait().slices[j].t0;
            let d = ed.gait().slices[j].duration() * ratio;
            ed.retime(j, t0 + d);
        }
        ed.commit();
        stats.n_replan += 1;
        if debug.replan { outln!(out, "  [diag-replan] site 0 t={t:.3} (line 6798)"); }
        outln!(out, "  [yield] t={t:.3} dcm_err={dcm_err:.3}: speed {v_hi:.2} -> {v_lo:.2} m/s, stride {stride_hi:.3} -> {stride_lo:.3}, slices x{ratio:.3}, {n} steps re-planned");
    }
    if push_replan_trig > 0.0 {
        let i = plan.gait().index_at(t);
        let sl = plan.gait().slices[i].clone();
        match sl.support {
            Support::Single { swing, stance }
                if !claim_check!(plan.claims().whole_owner() == Some(Writer::RecoveryRef), ctl.ref_mode.is_some())
                    // Not on a known staircase (its zone, with the crouch
                    // lead / tail): a deep step-down reads as a backward
                    // error and the re-plan cost the -150 mm descent
                    // 41 -> 30 (measured 2026-10-22).
                    && !stairs.as_ref().is_some_and(|st| {
                        // A zero-riser "stair" (the flat benches park one
                        // at x = 2 m) is not a staircase.
                        (st.riser.abs() > 1e-6 || st.down_n > 0)
                            && crate::height::stair_zone_crouch(
                            plan.footsteps().at_slice(i).mid_xy().x,
                            st,
                            stair_crouch_lead,
                            stair_crouch_tail,
                            1.0,
                        ) > 0.0
                    })
                    // PUSH_REPLAN_SLOPE_LEAD (m): nor on a known slope,
                    // from this far before its start to its end. The
                    // zero-riser test above lets a slope through, and
                    // there the re-plan costs the backward 11 N.s 18 -> 4
                    // (measured 2026-10-19: the re-laid plan meets the
                    // slope entry with a backward DCM error, and the
                    // second firing steps back off the slope). 0 = off.
                    && !(push_replan_slope_lead > 0.0
                        && stairs.as_ref().is_some_and(|st| {
                            let x = plan.footsteps().at_slice(i).mid_xy().x;
                            st.n == 0
                                && st.slope != 0.0
                                && x >= st.start - push_replan_slope_lead
                                && x <= st.start + st.slope_len
                        })) =>
            {
                // The plan in force when this single support began.
                if ctl.push_replan_state.as_ref().is_none_or(|st| st.0 != i) {
                    let base = plan.footsteps().at_slice(i).xy(swing);
                    ctl.push_replan_state = Some((i, plan.footsteps().clone(), plan.dcm().clone(), base, None));
                }
                let frac = (t - sl.t0) / sl.duration().max(1e-9);
                let rem = (sl.t1 - t).clamp(0.0, gp.t_ss);
                if let Some((_, orig_fs, orig_plan, base, last)) = ctl.push_replan_state.as_mut()
                    && frac < push_replan_freeze
                {
                    let e = xi - orig_plan.reference(t).xi;
                    // The ankle can hold an error up to the sole's half
                    // size by moving the CoP; only the rest diverges.
                    let half = na::Vector2::new(sole_half_l * push_replan_ankle[0], sole_half_w * push_replan_ankle[1]);
                    let grow = (omega * rem).exp();
                    let project = |e: na::Vector2<f64>| {
                        na::Vector2::new(
                            {
                                let dp = e.x.clamp(-half.x, half.x);
                                dp + (e.x - dp) * grow
                            },
                            {
                                let dp = e.y.clamp(-half.y, half.y);
                                dp + (e.y - dp) * grow
                            },
                        )
                    };
                    let d = project(e);
                    // PUSH_TRIG_LPF_HZ: engage on the low-passed error (the
                    // filtered `xi - r.xi` plus the reference's own offset
                    // from the plan in force); the landing still follows `d`.
                    let (e_trig, d_trig) = if cfg.dcm.trig_lpf_hz > 0.0 {
                        let e_t = trig_err + (r.xi - orig_plan.reference(t).xi);
                        (e_t, project(e_t))
                    } else {
                        (e, d)
                    };
                    let v_now = (vx2_sched && vx2_z_from.is_some_and(|t0| t >= t0))
                        .then_some(vx2_cmd_speed)
                        .flatten();
                    let back_ok = (!push_replan_back_only || (e.x < 0.0 && e.y.abs() < -e.x))
                        && !v_now.is_some_and(|v| v > push_replan_vmax);
                    // (PUSH_REPLAN_LAT, the outward lateral step too, is
                    // removed.)
                    let owns_landing = plan.claims().landing_owner(i) == Some(Writer::PushReplan);
                    if owns_landing
                        || (back_ok && d_trig.norm() > push_replan_trig && e_trig.norm() > push_replan_emin)
                    {
                        let st = orig_fs.at_slice(i).xy(stance);
                        let mut l = *base + d;
                        l.x = l.x.clamp(st.x - push_replan_max_back, st.x + push_replan_max_fwd);
                        // Lateral: +1 when the swing foot normally lands at +y of the stance foot.
                        let side = if base.y > st.y { 1.0 } else { -1.0 };
                        let w = (l.y - st.y) * side;
                        // Never widen beyond what the plan already had: the
                        // floor is the smaller of MIN_W and the planned gap
                        // (G1 walks at ~0.104 m; a 0.12 floor widened every
                        // backward re-plan by 2 cm, measured 2026-10-21).
                        let min_w = push_replan_min_w.min(((base.y - st.y) * side).max(0.0));
                        // Inward of the minimum gap (a crossing): leave this
                        // landing to the MPC (its crossing gate,
                        // MPC_FOOTSTEP_CROSS_*) -- do not engage, and once
                        // engaged keep the last landing.
                        let inward_skip = w < 0.0;
                        let w = if w >= min_w {
                            w.min(push_replan_max_w)
                        } else {
                            min_w
                        };
                        l.y = st.y + side * w;
                        if !inward_skip && last.is_none_or(|p| (p - l).norm() > 0.005) {
                            let mut fs = orig_fs.clone();
                            fs.replan_from_landing(plan.gait(), i, swing, l, push_replan_stand, push_replan_ramp);
                            let mut ed = plan.edit(Writer::PushReplan);
                            ed.replace_footsteps(fs);
                            ed.claim_landing(i);
                            ed.commit();
                            if !owns_landing {
                                stats.n_push_replan += 1;
                                stats.n_replan += 1;
                                if debug.replan { outln!(out, "  [diag-replan] site 1 t={t:.3} (line 6901)"); }
                            }
                            ctl.push_replan_shift = (l - *base).norm();
                            if debug.push_replan {
                                outln!(out, "  [replan] t={t:.3} slice {i} frac={frac:.2} e=({:+.3},{:+.3}) d=({:+.3},{:+.3}) landing ({:.3},{:+.3}) -> ({:.3},{:+.3})", e.x, e.y, d.x, d.y, base.x, base.y, l.x, l.y);
                            }
                            *last = Some(l);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    // Backward-dominant only, and never under an engaged recovery
    // reference (measured: firing on lateral pushes cost the lateral
    // ladder 136 -> 129 while the reference was running).
    // (The rate-predicted error, REF_TRIG_PRED, is removed: at its
    // default 0 the prediction was the error itself.)
    let back_err = (xi - r.xi).x;
    if push_z_drop > 0.0
        && back_err < -push_z_drop_trig
        && (xi - r.xi).y.abs() < push_z_drop_lat_max
        && !claim_check!(plan.claims().whole_owner() == Some(Writer::RecoveryRef), ctl.ref_mode.is_some())
        && t >= ctl.push_z_drop_until
    {
        ctl.push_z_drop_until = t + push_z_drop_t;
        stats.n_push_z_drop += 1;
        outln!(out, "  [zdrop] t={t:.3} backward DCM error {:+.3} m -> CoM {:.3} m lower for {push_z_drop_t:.2} s", (xi - r.xi).x, push_z_drop);
    }
    // Engage a recovery reference when the DCM error toward the STANCE
    // side exceeds REF_TRIG_DCM in single support (the detector).
    // (REF_SLOPE_GATE, no reference on a known slope, is removed.)
    if ref_any
        && !claim_check!(plan.claims().whole_owner() == Some(Writer::RecoveryRef), ctl.ref_mode.is_some())
        && t >= ctl.ref_cooldown_until
        && slice_frac <= ref_trig_max_frac
        && let Support::Single { stance, swing } = support
    {
        let lateral = (xi - r.xi).y * if stance == 0 { 1.0 } else { -1.0 };
        if lateral.abs() < 0.005 {
            ctl.ref_lat_quiet_t = t;
        }
        let forward = (xi - r.xi).x;
        // The library's lateral-push references (`impulse_deg` 90).
        let want_deg = 90.0;
        if ref_trig_dcm > 0.0 && lateral > ref_trig_dcm && ref_libs.iter().any(|l| l.impulse_deg == want_deg) {
            // The push happened when the lateral error last was quiet;
            // the reference is entered that far in.
            let tau0 = if !ref_delay_comp { 0.0 } else { (t - ctl.ref_lat_quiet_t).clamp(0.0, 0.3) };
            outln!(out, "  [ref] t={t:.3} trigger: dcm (lateral DCM error toward stance {lateral:+.3} m, forward {forward:+.3} m, dir {want_deg:.0}, tau0 {tau0:.3})");
            // Pick the reference meant for the nearest swing phase.
            // (Selecting by the swing foot's height was tried first:
            // on the flat in-place gait the apex comes late, so the
            // height does not order the phases.)
            // Phase AT THE PUSH (the detector fires tau0 later).
            let frac_push = slice_frac - tau0 / plan.gait().slices[slice_idx].duration().max(1e-6);
            // Nearest walking speed first (the commanded speed once the
            // speed change has ramped in, else the nominal), then phase.
            let v_now = match vx2_change {
                Some((at, v2, ramp2)) if plan.gait().steps_taken(t) >= at + ramp2 => v2,
                _ => vx,
            };
            let pick = (0..ref_libs.len())
                .filter(|&k| ref_libs[k].impulse_deg == want_deg && !ref_libs[k].landed)
                .min_by(|&a, &b| {
                    let da = ((ref_libs[a].vx - v_now).abs() * 10.0, (ref_libs[a].swing_phase - frac_push).abs());
                    let db = ((ref_libs[b].vx - v_now).abs() * 10.0, (ref_libs[b].swing_phase - frac_push).abs());
                    (da.0 + da.1).partial_cmp(&(db.0 + db.1)).unwrap()
                })
                .unwrap_or(0);
            ctl.ref_lib = Some(ref_libs[pick].clone());
            let lib = ctl.ref_lib.as_ref().unwrap();
            outln!(out, "  [ref] swing phase {slice_frac:.2} -> reference {} (meant for {:.2})", pick, lib.swing_phase);
            let mode = RefMode { t0: t, tau0, stance, origin: actual.sole[stance], mirror: if stance == 0 { 1.0 } else { -1.0 } };
            // Phases: this single support ends after phases[0]; the
            // double support after it (every DS slice up to the next
            // single support) gets phases[1]; the next single support
            // phases[2]; the double support after that phases[3].
            let mut ed = plan.edit(Writer::RecoveryRef);
            let i0 = slice_idx;
            let mut j = i0;
            let mut ph = 0usize;
            let mut t_cursor = t;
            let mut first_ss_after: Option<usize> = None;
            while j < ed.gait().slices.len() && ph < lib.phases.len() {
                let is_ss = matches!(ed.gait().slices[j].support, Support::Single { .. });
                // Count the slices of this phase (DS phases may be split).
                let mut jend = j + 1;
                if !is_ss {
                    while jend < ed.gait().slices.len() && matches!(ed.gait().slices[jend].support, Support::Double) {
                        jend += 1;
                    }
                }
                let n = (jend - j) as f64;
                for jj in j..jend {
                    let remaining = if ph == 0 { (lib.phases[0] - tau0).max(0.05) } else { lib.phases[ph] };
                    let end = t_cursor + remaining / n;
                    ed.retime(jj, end);
                    t_cursor = end;
                }
                if is_ss && j != i0 && first_ss_after.is_none() {
                    first_ss_after = Some(j);
                }
                j = jend;
                ph += 1;
            }
            // Footfalls: every single-support phase of the reference
            // lands its swing foot (reference right, left, right, ...)
            // where the reference puts it at the end of that phase.
            let mut t_end = 0.0;
            let mut jj = i0;
            let mut ref_sw = 1usize;
            let mut land1 = na::Vector3::zeros();
            let mut land2 = na::Vector3::zeros();
            for (ph, dur) in lib.phases.iter().enumerate() {
                t_end += dur;
                if ph % 2 == 0 {
                    // find the single-support slice of this phase
                    while jj < ed.gait().slices.len() && ed.gait().slices[jj].support.swing().is_none() {
                        jj += 1;
                    }
                    if jj >= ed.gait().slices.len() {
                        break;
                    }
                    let foot = mode.foot(ref_sw);
                    let land = mode.world(lib, lib.sole_at(ref_sw, t_end));
                    let d = (land - ed.footsteps().at_slice(jj).sole[foot]).xy();
                    ed.shift_next(jj, foot, d);
                    if ph == 0 { land1 = land; } else if ph == 2 { land2 = land; }
                    ref_sw = 1 - ref_sw;
                    jj += 1;
                }
            }
            let _ = first_ss_after;
            ed.landing_mut().event_mode = true;
            ed.claim_whole();
            ed.commit();
            ctl.ref_mode = Some(mode);
            stats.ref_engagements += 1;
            outln!(out, 
                "  [ref] t={t:.3} engaged: stance={} mirror={:+.0} origin=({:.3},{:.3}) phases={:?} land1=({:.3},{:.3}) land2=({:.3},{:.3}) total={:.3}",
                stance, mode.mirror, mode.origin.x, mode.origin.y, lib.phases, land1.x, land1.y,
                land2.x, land2.y, lib.total()
            );
        
        }
    }
    DcmOut { xi, r, dcm_err, trig_err }
}

/// What phase 6 found: the commanded ZMP clamped into the support, how far
/// the clamp moved it, and the pressure it refused (MOM_CMP's input).
pub(crate) struct CmdZmpOut {
    pub p_cmd: na::Vector2<f64>,
    pub clamped: f64,
    pub zmp_deficit: na::Vector2<f64>,
}

/// Phase 6: the commanded ZMP and the contact planner (CONTACT_PLAN: the
/// landing shift and the step's end time).
#[allow(clippy::too_many_arguments)]
pub(crate) fn phase_cmd_zmp(
    ctl: &mut CtlState,
    plan: &mut Plan,
    stats: &mut WalkerStats,
    cfg: &Config,
    k: &WalkerConsts,
    sup: &SupportOut,
    dcm: &DcmOut,
    t: f64,
    out: &mut dyn LogSink,
) -> CmdZmpOut {
    let k_dcm = cfg.dcm.k_dcm;
    let cop_frac = cfg.dcm.cop_frac;
    let zmp_margin = cfg.dcm.zmp_margin;
    let clamp_zmp = cfg.dcm.clamp_zmp;
    let contact_plan_on = cfg.contact_plan.on;
    // Only plan when the DCM is actually off the plan (same reason as the
    // removed ADAPT_TIME_TRIG: the undisturbed gait must reproduce bit for
    // bit).
    let contact_plan_trig = cfg.contact_plan.trig;
    let debug = cfg.debug;
    let (dt, omega) = (k.dt, k.omega);
    let (sole_half_l, sole_half_w) = (k.sole_half_l, k.sole_half_w);
    let contact_plan_cfg = k.contact_plan_cfg;
    let (support, steps) = (sup.support, &sup.steps);
    let (xi, r, dcm_err) = (dcm.xi, &dcm.r, dcm.dcm_err);
    let p_raw = commanded_zmp(&xi, &r, omega, k_dcm);
    // Clamp against the support that is REALLY there, which in NO_LIFT is
    // both feet even where the plan says one.
    let sbox = SupportBox::from_stance_yawed(
        &steps,
        support,
        (sole_half_l * cop_frac, sole_half_w * cop_frac),
        zmp_margin,
    );
    let (p_cmd, clamped) = if clamp_zmp {
        sbox.clamp(&p_raw)
    } else {
        (p_raw, 0.0)
    };
    // Pressure the polygon refused (world x, y), for MOM_CMP.
    let zmp_deficit = p_raw - p_cmd;

    // ---- contact plan (CONTACT_PLAN) --------------------------------
    if contact_plan_on
        && !claim_check!(plan.claims().whole_owner() == Some(Writer::RecoveryRef), ctl.ref_mode.is_some())
        && let Support::Single { swing, .. } = support
        && dcm_err >= contact_plan_trig
    {
        let i = plan.gait().index_at(t);
        if ctl.cp_slice != Some(i) {
            ctl.cp_slice = Some(i);
            ctl.cp_applied = na::Vector2::zeros();
        }
        let seg = &plan.dcm().segs[plan.dcm().index_at(t)];
        let sit = crate::contact_plan::Situation {
            xi,
            p: seg.at(t),
            p_dot: seg.slope(),
            omega,
            t,
            nominal_end: plan.gait().slices[i].t1,
            xi_eos_plan: plan.dcm().eos_at(t),
            swing_sign: if swing == 0 { 1.0 } else { -1.0 },
        };
        // (CP_J_AUTO, the sensitivity differenced from the plan, and
        // CP_TOUCHDOWN, planning for the DCM at touchdown, were not
        // adopted and are removed.)
        let dec = crate::contact_plan::plan(&sit, &contact_plan_cfg);
        if debug.cp {
            outln!(out, 
                "  [cp] t={t:.3} slice={i} err={dcm_err:.4} xi=({:.3},{:+.3}) eos_plan=({:.3},{:+.3}) pred_nom=({:.3},{:+.3}) shift=({:+.3},{:+.3}) dT={:+.3} unmet={:.4} cost={:.3}",
                xi.x, xi.y, sit.xi_eos_plan.x, sit.xi_eos_plan.y,
                crate::contact_plan::predict(&sit, (sit.nominal_end - t).max(0.0)).x,
                crate::contact_plan::predict(&sit, (sit.nominal_end - t).max(0.0)).y,
                dec.shift.x, dec.shift.y, dec.new_end.map_or(0.0, |e| e - sit.nominal_end), dec.unmet, dec.cost
            );
        }
        // (CP_TREAD_BOX, snapping the landing into the tread's sole
        // window, MEASURED HARMFUL 2026-09-25: landing fully on the tread
        // does not predict survival at the 60 mm riser, and snapping to
        // the window edge took 26/32 -> 17/32. Removed; terrain awareness
        // belongs in a cost with margin, not a snap to an edge.)
        let mut ed = plan.edit(Writer::ContactPlan);
        // Like ADAPT_STEP did: place the foot once per step, keeping the
        // largest demand seen, instead of re-placing it every tick.
        //
        // That is a RATCHET: the landing only ever moves further from
        // nominal. With the shipped 15 mm trigger it fires a handful of
        // times in an emergency and the ratchet is harmless, but run
        // continuously it walks the foot out to its bound and the stair
        // approach dies (measured 2026-09-24 on the randomised stairs:
        // 0/192 at every shift weight, falling at 21-23 s). (CP_TRACK,
        // following the current decision instead, is removed: with the
        // champion's CP_MAX_*=0 the shift is always zero and both rules
        // give zero.)
        let d = if dec.shift.norm() > ctl.cp_applied.norm() {
            dec.shift - ctl.cp_applied
        } else {
            na::Vector2::zeros()
        };
        if d.norm() > 1e-9 && !ed.landing_closed(i) {
            // `shift_from` shifts the swing foot's landing in slice `i`
            // AND every later slice, so the adjusted position becomes the
            // nominal that the NEXT step's residual is measured against.
            // Under continuous firing the foot walks outward one bound per
            // step (measured 2026-09-25: lateral spacing 127 -> 153 mm
            // median against a 30 mm limit). CP_SHIFT_NEXT, moving only
            // this landing, was not adopted and is removed.
            ed.shift_from(i, swing, d);
            ctl.cp_applied = dec.shift;
            stats.n_cp_shift += 1;
        }
        if let Some(ne) = dec.new_end
            && ne > t + dt
            && ed.claims().end_time_owner(i) != Some(Writer::ContactPlan)
        {
            let dd = ed.retime(i, ne);
            stats.n_cp_retime += 1;
            stats.cp_retime_total += dd.abs();
            ed.claim_end_time(i);
            // The planner has now set this step's end; the landing
            // extension must not move it again (the two would fight
            // over the same boundary, and extension IS mechanism B).
            ed.landing_mut().max_extension = 0.0;
        }
        // Rebuilt iff the landing moved or the end was set.
        if ed.commit() {
            stats.n_replan += 1;
            if debug.replan { outln!(out, "  [diag-replan] site 2 t={t:.3} (line 7454)"); }
        }
    }
    CmdZmpOut { p_cmd, clamped, zmp_deficit }
}

/// Phase 7: the CoM preview (COM_REPLAN once engaged): re-solve on its own
/// clock and pull the commanded ZMP's x toward the preview's. Returns the
/// commanded ZMP after that correction. Writes PREVIEW_TRAJ_CSV and
/// COM_PREVIEW_RUNTIME_CSV (the harness's sinks, until S13's hook).
#[allow(clippy::too_many_arguments)]
pub(crate) fn phase_preview(
    ctl: &mut CtlState,
    plan: &Plan,
    stats: &mut WalkerStats,
    diag: &mut dyn WalkerDiag,
    cfg: &Config,
    k: &WalkerConsts,
    obs: &TickObs,
    actual: &Footsteps,
    stance_sides: &[usize],
    mut p_cmd: na::Vector2<f64>,
    t: f64,
) -> na::Vector2<f64> {
    let cop_frac = cfg.dcm.cop_frac;
    let zmp_margin = cfg.dcm.zmp_margin;
    let com_preview_n = cfg.com_plan.preview_n;
    let com_preview_smooth = cfg.com_plan.preview_smooth;
    let com_replan_forward_gain = cfg.com_plan.replan_forward_gain;
    let omega = k.omega;
    let (sole_half_l, sole_half_w) = (k.sole_half_l, k.sole_half_w);
    let (com_preview_dt, com_replan_dt) = (k.com_preview_dt, k.com_replan_dt);
    let st = obs_state(obs);
    let (com, com_vel) = (st.com, st.com_vel);
    if plan.claims().done(Writer::ReplanStart) {
        let measured_support = if stance_sides.len() == 2 {
            Support::Double
        } else {
            let side = stance_sides[0];
            Support::Single {
                stance: side,
                swing: 1 - side,
            }
        };
        let current_box = SupportBox::from_stance_yawed(
            &actual,
            measured_support,
            (sole_half_l * cop_frac, sole_half_w * cop_frac),
            zmp_margin,
        );
        if t + 1e-9 >= ctl.preview_next {
            let mut samples = Vec::with_capacity(com_preview_n);
            for k in 0..com_preview_n {
                let begin = t + k as f64 * com_preview_dt;
                let end = begin + com_preview_dt;
                let f0 = crate::preview::forecast_feet(
                    &plan.gait(),
                    &plan.footsteps(),
                    &actual,
                    &stance_sides,
                    t,
                    begin,
                );
                let f1 = crate::preview::forecast_feet(
                    &plan.gait(),
                    &plan.footsteps(),
                    &actual,
                    &stance_sides,
                    t,
                    end - 1e-9,
                );
                let b0 = SupportBox::from_stance_yawed(
                    &f0,
                    plan.gait().support_at(begin),
                    (sole_half_l * cop_frac, sole_half_w * cop_frac),
                    zmp_margin,
                );
                let b1 = SupportBox::from_stance_yawed(
                    &f1,
                    plan.gait().support_at(end - 1e-9),
                    (sole_half_l * cop_frac, sole_half_w * cop_frac),
                    zmp_margin,
                );
                let mut bounds = SupportBox {
                    lo: b0.lo.sup(&b1.lo),
                    hi: b0.hi.inf(&b1.hi),
                };
                if k == 0 {
                    bounds = SupportBox {
                        lo: current_box.lo,
                        hi: current_box.hi,
                    };
                }
                let (zmp_ref, xi_ref) = (
                    plan.dcm().reference(begin + 0.5 * com_preview_dt).p,
                    plan.dcm().reference(end).xi,
                );
                samples.push(crate::preview::PreviewSample {
                    duration: com_preview_dt,
                    bounds,
                    zmp: zmp_ref,
                    xi: xi_ref,
                });
            }
            let preview_solve_start = std::time::Instant::now();
            let preview_solution = crate::preview::replan(
                omega,
                com.xy(),
                com_vel.xy(),
                ctl.preview_p,
                &samples,
                com_preview_smooth,
            );
            let preview_solve_ns = preview_solve_start.elapsed().as_nanos();
            let preview_accepted = preview_solution.is_some();
            match preview_solution {
                Some(plan) => {
                    ctl.preview_p = plan.zmp[0];
                    ctl.preview_c = plan.com[0];
                    ctl.preview_v = plan.velocity[0];
                    stats.preview_error = plan.terminal_error;
                    stats.preview_count += 1;
                    diag.preview_solved(t, &samples, &plan);
                }
                None => {
                    ctl.preview_p = current_box.clamp(&p_cmd).0;
                    stats.preview_failures += 1;
                }
            }
            // (applied_lateral: COM_REPLAN_LATERAL, removed, was off.)
            diag.preview_runtime(t, preview_solve_ns, preview_accepted, stats.preview_error, com_preview_n);
            // (COM_PREVIEW_PHASE_LOCK, a discrete-clock update rate, is
            // removed.)
            ctl.preview_next = t + com_replan_dt;
        }
        let optimized = current_box.clamp(&ctl.preview_p).0;
        // Forward only (COM_REPLAN_LATERAL and its gain are removed).
        p_cmd.x += com_replan_forward_gain * (optimized.x - p_cmd.x);
    }
    p_cmd
}

/// What phase 8 found: the CoM height reference (after the speed / push
/// drops) and the commanded CoM acceleration.
pub(crate) struct HeightOut {
    pub z_ref: f64,
    pub a_com: na::Vector3<f64>,
}

/// Phase 8: the horizontal CoM acceleration from the commanded ZMP (the
/// variable-height form on stairs, the jerk limit), the height reference
/// (terrain, stance level, descent lead, stair crouch, squat, the speed and
/// push drops) and the vertical PD.
#[allow(clippy::too_many_arguments)]
pub(crate) fn phase_height(
    ctl: &mut CtlState,
    plan: &mut Plan,
    cfg: &Config,
    k: &WalkerConsts,
    obs: &TickObs,
    sup: &SupportOut,
    stance_sides: &[usize],
    p_cmd: &na::Vector2<f64>,
    t: f64,
) -> HeightOut {
    // VH_DCM=1: the height-varying DCM law on a modelled terrain (see its use).
    let vh_dcm = cfg.dcm.vh_dcm;
    let com_xy_jerk_max = cfg.dcm.com_xy_jerk_max;
    // Risers per step the height reference is allowed to move. Small spreads
    // one riser's climb over several steps; large finishes it within the
    // swing that takes the tread.
    let stair_climb_rate = cfg.com_plan.stair_climb_rate;
    let com_height_replan = cfg.com_plan.height_replan;
    // (LAND_WAIT_HOLD, holding the DCM reference at the stance sole while a
    // stair landing waits, was not adopted and is removed.)
    // Lower the CoM height target while the feet are in the stair zone.
    // Measured 2026-09-24 on G1: the second step onto an 80 mm riser puts
    // the swing sole 0.20 m ahead of the stance sole with the CoM 0.684 m
    // above the tread -- a 0.71 m reach, the leg's full length. The knee
    // arrives straight (0.01-0.06 rad), the swing task can no longer push
    // the sole down (residual 20 -> 600), and the QP throws the leg up.
    // The 0.233 m landing stride (STAIR_LANDING_FRAC) has the same reach.
    // Zone: mid-feet x from `start - lead` to `landing_start + tail`. The
    // target moves through the usual `COM_HEIGHT_TIME` retarget, so it is
    // smooth. Default 0 = shipped height.
    // (Distinct from the older global `STAIR_CROUCH` (removed), which
    // lowered the CoM from t=0 at 50 mm/s and rescaled the LIPM height;
    // measured 2026-09-24 that one fells the flat approach at t=5 on G1.)
    // Aim the CoM height at the LOWER level a descent is about to step
    // onto, before the swing. Measured 2026-10-03 on G1 (v6 + descent
    // extension, 80 mm step-down, 45 fallers of 48): the CoM reference
    // follows the STANCE soles, so it stays at platform height through the
    // whole step-down swing; the swing leg reaches 99.7-99.9% of its length
    // and stops 15-20 mm above the lower surface; the schedule then admits
    // the airborne foot, whose orientation is uncontrolled for ~80 ms while
    // it is loaded, and the corner it happens to land on decides the run
    // (heel-outer corner -> survives, toe-outer corner -> slides 12 cm
    // forward on the toe and falls). Fraction of the drop to the coming
    // landing that the height reference takes in advance: 1 = the full
    // riser, through the usual COM_HEIGHT_TIME retarget. Only a landing
    // BELOW the stance soles counts, so ascents and level walking are
    // bit-identical. Default 0 = off = shipped.
    let com_height_desc_lead = cfg.descent.com_height_lead;
    let stair_crouch_lead = cfg.stair_zone.lead;
    let stair_crouch_tail = cfg.stair_zone.tail;
    let stair_zone_crouch = cfg.descent.zone_crouch;
    let com_height_time = cfg.com_plan.height_time;
    let push_z_drop = cfg.push_crouch.drop;
    let kp_com = cfg.gains.kp_com;
    let kd_com = cfg.gains.kd_com;
    let diag_vz_bias_height = cfg.gains.diag_vz_bias_height;
    let (dt, omega, z_com, terrain_base) = (k.dt, k.omega, k.z_com, k.terrain_base);
    let (vx, t_step, stairs) = (k.vx, k.t_step, &k.stairs);
    let (squat_amp, squat_period) = (k.squat_amp, k.squat_period);
    let (vx2_z_from, vx2_z_drop) = (k.vx2_z_from, k.vx2_z_drop);
    let st = obs_state(obs);
    let (com, com_vel) = (st.com, st.com_vel);
    let (slice_idx, steps) = (sup.slice_idx, &sup.steps);
    // Variable-height form when a stair is in the scene: the LIPM's
    // constant-height omega is ~20% too stiff while the CoM is being
    // driven DOWN a riser. Gated so level and validated runs are bit-
    // identical (VH_DCM=0 restores the classic form).
    let a_xy_raw = if stairs.is_some() && vh_dcm {
        // Height above the surface the pressure actually acts on, which
        // on a stair is the STANCE sole -- not the fixed `z_lip` omega
        // was built from. Measured on a descent: omega^2 claims 32.9
        // while the true g/h is 39.7, a 21% error in the commanded ZMP
        // response, in the unsafe direction (too soft).
        let surf = if stance_sides.is_empty() {
            0.0
        } else {
            stance_sides.iter().map(|&s| steps.sole[s].z).sum::<f64>()
                / stance_sides.len() as f64
        };
        crate::dcm::com_accel_xy_vh(&com, &p_cmd, omega, 0.0, surf)
    } else {
        com_accel_xy(&com, &p_cmd, omega)
    };
    let a_xy = if let Some(previous) = ctl.previous_a_xy {
        let delta = a_xy_raw - previous;
        let max_delta = com_xy_jerk_max * dt;
        if delta.norm() > max_delta {
            previous + delta * (max_delta / delta.norm())
        } else {
            a_xy_raw
        }
    } else {
        a_xy_raw
    };
    ctl.previous_a_xy = Some(a_xy);
    // z stays a PD on the nominal height: the LIPM says nothing about it,
    // and a constant-height assumption is exactly what makes omega a
    // constant in the first place.
    // Height reference. `omega` stays at the nominal height: the LIPM is
    // derived for a constant one, and re-deriving it per tick is the
    // variable-height DCM formulation, which this is not. The squat is
    // therefore a disturbance the horizontal loop has to survive, which is
    // the point of having it in the benchmark.
    // Stairs: the height reference rides the terrain. The target is the
    // mean of the two planned sole heights for THIS slice, slewed at
    // 1.5 risers per step so a riser is absorbed over roughly one step.
    // omega stays at the flat-ground value -- same argument as the squat
    // below: this is not the variable-height DCM formulation, and the
    // height change is a disturbance the horizontal loop must survive.
    let zd_terrain = if let Some(st) = &stairs {
        // A DS-only ramp was tried, but left the height reference a
        // riser behind the feet. Slew through both support phases.
        // Foot-over-foot climbs one riser per step, so the rate limit
        // turns the staircase of slice targets into a near-constant
        // slope; no gating on support phase (a DS-only ramp was tried
        // and left the reference a riser behind the feet by tread 2).
        let fs = plan.footsteps().at_slice(slice_idx);
        // The mean planned sole height (STAIR_DESC_MIN, the lower of the
        // two on a descent, is removed).
        let tgt_h = 0.5 * (fs.sole[0].z + fs.sole[1].z);
        // A slope (TERRAIN_SLOPE) has no riser: its rise per step is
        // the grade times the nominal stride, else the rate is zero and
        // the height reference never leaves the spawn level (measured
        // 2026-10-15 on +3 deg: CoM 0.683 absolute for the whole climb).
        let riser_eff = if st.n == 0 && st.slope != 0.0 {
            st.slope.abs() * (vx.abs() * t_step).max(0.10)
        } else {
            st.riser.abs()
        };
        let rate = stair_climb_rate * riser_eff / t_step.max(1e-6);
        let dh = (tgt_h - ctl.terrain_track).clamp(-rate * dt, rate * dt);
        ctl.terrain_track += dh;
        dh / dt
    } else {
        0.0
    };
    let (z_ref, zd_ref, zdd_ref) = if plan.claims().done(Writer::ReplanStart) && com_height_replan && plan.landing().clock.is_some() {
        // Where the CoM height aims while BOTH feet count: historically
        // the mean sole height. Measured 2026-09-24 on the 80 mm riser
        // with the landed foot admitted at the start of double support
        // (CONTACT_ADMIT_CONFIRMED): the half-rise then completes before
        // the trailing leg has left the lower tread, that leg reaches
        // single support with its knee at 0.09 rad (champion: 0.34), and
        // the next swing task has no authority left -- the foot launches
        // to 1.2 m. (COM_HEIGHT_DS_FRAC, keeping the CoM at the lower
        // sole through double support, was not adopted and is removed;
        // the level is the mean.)
        // All height rules live in src/biped/height.rs (one place for the
        // stance level, the descent lead, the stair crouch, the reach
        // caps and the slewed drops); the arithmetic is the one used
        // inline before 2026-10-19, bit-identical.
        use crate::height as hr;
        let soles_z: Vec<f64> = stance_sides.iter().map(|&side| steps.sole[side].z).collect();
        let h = hr::stance_level(&soles_z, None);
        // The landing this swing is heading for, or -- in a double
        // support -- the landing of the swing that follows it.
        let next_landing = {
            let i = plan.gait().index_at(t);
            (i..plan.gait().slices.len().min(i + 4))
                .find_map(|j| plan.gait().slices[j].support.swing().map(|sw| (j, sw)))
                .map(|(j, sw)| (sw, plan.footsteps().at_slice(j).sole[sw]))
        };
        let h = hr::descent_lead(h, next_landing.map(|(_, l)| l.z), com_height_desc_lead);
        let crouch = match &stairs {
            Some(st) => hr::stair_zone_crouch(steps.mid_xy().x, st, stair_crouch_lead, stair_crouch_tail, stair_zone_crouch),
            None => 0.0,
        };
        let target = if crouch > 0.0 {
            z_com + h - terrain_base - crouch
        } else {
            z_com + h - terrain_base
        };
        // (REACH_HEIGHT / _FLOOR, the height capped by the legs' reach
        // over every sole, and COM_HEIGHT_DESC_REACH, the same cap for
        // the landing of a descent, were not adopted and are removed.)
        if (target - plan.height().target).abs() > 1e-6 {
            let mut ed = plan.edit(Writer::Height);
            ed.height_mut().retarget(t, target, com_height_time);
            ed.commit();
        }
        plan.height().at(t)
    } else if squat_amp > 0.0 {
        let w = 2.0 * PI / squat_period;
        let ph = w * t;
        (
            z_com - squat_amp * (1.0 - ph.cos()) * 0.5,
            -squat_amp * 0.5 * w * ph.sin(),
            -squat_amp * 0.5 * w * w * ph.cos(),
        )
    } else {
        (
            z_com + (ctl.terrain_track - terrain_base),
            zd_terrain,
            0.0,
        )
    };
    // VX2_Z_DROP: eased in at 50 mm/s on top of whichever height
    // reference is live (the flat walk never enters the replan branch).
    // PUSH_Z_DROP: lower the CoM for PUSH_Z_DROP_T after a BACKWARD
    // push (DCM error behind the plan by more than PUSH_Z_DROP_TRIG).
    // Measured 2026-10-05: 24 mm lower for the whole run doubles the
    // backward-push survival (21 -> 43/48) but trades descents away;
    // tied to the detection it leaves the other benches alone.
    let z_drop_target = crate::height::drop_target(
        if vx2_z_from.is_some_and(|t0| t >= t0) { vx2_z_drop } else { 0.0 },
        if t < ctl.push_z_drop_until { push_z_drop } else { 0.0 },
    );
    let (z_ref, zd_ref) = if z_drop_target > 0.0 || ctl.vx2_z_track.track > 0.0 {
        let d = ctl.vx2_z_track.step(z_drop_target, 0.05, dt);
        (z_ref - ctl.vx2_z_track.track, zd_ref - d / dt)
    } else {
        (z_ref, zd_ref)
    };
    // (MPC_REACH, the CoM height capped by the MPC from the legs' reach,
    // was not adopted and is removed.)
    let a_z = zdd_ref + kp_com * (z_ref - com.z) + kd_com * (zd_ref - com_vel.z - diag_vz_bias_height);
    let a_com = na::Vector3::new(a_xy.x, a_xy.y, a_z);
    HeightOut { z_ref, a_com }
}

/// Phase 9: the centroidal wrench MPC (STAIR_WRENCH_MPC with
/// WRENCH_MPC_ALWAYS). It replans both sole wrenches from the measured
/// centroidal state; the first horizon command becomes the WBC's force
/// regularisation target and its resultant acceleration the CoM task's
/// reference, which removes the old conflict where ZMP, height and load
/// split came from independent controllers. On its own clock, build the input (load and contact
/// schedules, the contacts and their forecast landings, the crossing gate,
/// the landing variable), solve, hold the answer in `ctl`, and move the
/// swing's landing by the MPC's shift (MPC_FOOTSTEP). Writes MPC_DIAG_DIR,
/// MPC_RUNTIME_CSV and MPC_TRAJ_CSV (the harness's sinks, until S13's hook).
/// Returns the time spent on the MPC_DIAG_DIR capture [ns].
#[allow(clippy::too_many_arguments)]
pub(crate) fn phase_wrench_mpc(
    ctl: &mut CtlState,
    plan: &mut Plan,
    stats: &mut WalkerStats,
    diag: &mut dyn WalkerDiag,
    cfg: &Config,
    k: &WalkerConsts,
    obs: &TickObs,
    actual: &Footsteps,
    cent: &Centroidal,
    sup: &SupportOut,
    stance_sides: &[usize],
    dcm: &DcmOut,
    a_com: &na::Vector3<f64>,
    t: f64,
    out: &mut dyn LogSink,
) -> u128 {
    let stair_wrench_mpc = cfg.mpc.enabled;
    // Run the wrench MPC outside a stair transfer too.
    //
    // The block below was gated on a running stair transfer (since removed),
    // so the MPC solved only while a transfer was armed. Measured
    // 2026-09-22: at 60 and 80 mm risers no transfer armed in 16 of 16 runs,
    // i.e. the best high-riser figures of that day were obtained with the
    // MPC executing ZERO ticks. A predictive layer that only runs during a
    // hand-written primitive cannot carry a momentum reference or a foothold
    // decision, so this is infrastructure for both rather than a result on
    // its own.
    //
    // Default OFF: the walking branch builds its own references (below) and
    // must be measured against the transfer-only lineage before it replaces
    // it.
    let wrench_mpc_always = cfg.mpc.always;
    let stair_wrench_horizon = cfg.mpc.horizon;
    // Align the fz schedules with `contact_schedule` (node centre).
    let mpc_node_centre = cfg.mpc.node_centre;
    let mpc_forecast_swing_only = cfg.mpc.forecast_swing_only;
    let mpc_schedule_early_fz = cfg.mpc.schedule_early_fz;
    let mpc_cop_scale_x = cfg.mpc.cop_scale_x;
    let (mpc_auth_stair_off, mpc_auth_tilt) = (cfg.mpc.auth_stair_off, cfg.mpc.auth_tilt);
    let stair_wrench_low_min = cfg.mpc.low_min;
    // 0 = off, 1 = seed the FIRST solve from the measured load, 2 = anchor
    // EVERY solve to the measured load (the stair-transfer ramp's rule, made
    // general): the smoothness term then says "do not jump away from what the
    // robot is actually carrying" instead of "do not jump away from what the
    // MPC said last time", which is what makes it lag every load transfer.
    let mpc_seed_previous = cfg.mpc.seed_previous;
    // Couple the plant's measured contact set into both the horizon and WBC.
    // An absent foot is recovered kinematically and cannot provide a wrench
    // until ContactCorrection's load hysteresis admits it again.
    let stair_hybrid_contact = cfg.mpc.hybrid_contact;
    let mpc_footstep_cross_y = cfg.mpc_cross.y;
    let mpc_footstep_cross_ex = cfg.mpc_cross.ex;
    let debug = cfg.debug;
    // Largest lateral shift TOWARD the stance foot (a narrower step). The
    // contact_plan adjuster forbids it in v4.1 (CP_MAX_Y_IN=0); the MPC box
    // allowed it. Measured 2026-09-27 (D+J+T, 11 N.s backward, m32): the
    // landing moved 27 mm inward, the lateral DCM error then ran away
    // -19 -> -126 mm and the robot fell. Default = MPC_FOOTSTEP_MAX_Y
    // (symmetric box), bit-identical.
    let mpc_footstep_max_y_in = cfg.mpc_footstep.max_y_in;
    // Inward bound while the SAGITTAL DCM error is forward by at least
    // MPC_FOOTSTEP_Y_IN_FWD_TRIG [m] (world x; the walk heads +x). Measured
    // 2026-09-27: the inward shifts that save 11 N.s forward pushes happen
    // at x error +10..+17 mm; the harmful ones after backward pushes at
    // x error -16..-34 mm (push step) or ~0 with a large lateral error (the
    // next step). The retimed-step rule above does not separate them (a
    // backward push shortens steps too). 0 = off = shipped.
    let mpc_footstep_y_in_fwd = cfg.mpc_footstep.y_in_fwd;
    let mpc_footstep_y_in_fwd_trig = cfg.mpc_footstep.y_in_fwd_trig;
    let mpc_footstep = cfg.mpc_footstep.on;
    // MPC_FOOTSTEP_STAIR_OFF: no landing variable on a known staircase.
    // Measured 2026-10-31 on kyo46rs (v3 + the G1 push keys): the one-step
    // descent's long stair step (5x the cruise step) carries a DCM error
    // above MPC_FOOTSTEP_TRIG with no push at all, the variable arms on every
    // member and moves the landing, and 20 mm descents fall 0 better / 63
    // worse over three seeds; without the variable they are unchanged.
    let mpc_footstep_stair_off = cfg.mpc_footstep.stair_off;
    // Arm the landing variable only once the DCM error of this single
    // support reaches this value [m] (latched for the rest of the step), the
    // rule the contact_plan adjuster uses (CP_TRIG): an undisturbed step
    // then solves exactly the fixed-contact problem. Measured 2026-09-27:
    // with the variable always present, D+J loses 6-8 cm wide-holdout
    // stairs on two independent seeds (pooled 1 better / 10 worse) while
    // the variable itself stays near 0 where the runs part -- the cost is
    // its presence, not its choice. 0 = always armed = shipped.
    let mpc_footstep_trig = cfg.mpc_footstep.trig;
    let mpc_footstep_w = cfg.mpc_footstep.w;
    // (MPC_FOOTSTEP_TRIG_HOLD, arming the following steps too, and
    // VXS_BOX_GAIN, a box grown with the speed command, are removed.)
    let mpc_footstep_max = cfg.mpc_footstep.max;
    // MPC_FOOTSTEP_CROSS_FLOOR [fraction of the sole half width]: while
    // crossing, a lower bound on the inward shift from the LIPM: the DCM at
    // the end of this single support, with the stance CoP held at the
    // stance sole's outer edge (the best the ankle can do), must lie within
    // this fraction of the half width of the landing sole. (Its x twin,
    // MPC_FOOTSTEP_CAPTURE_FLOOR, is removed.) Measured 2026-10-20: with
    // only the bound opened the MPC lands at y +0.02..+0.04 for a DCM at
    // +0.09 (stance sole +0.05) -- the SRBD model has no attitude and
    // believes it can hold the excursion. 0 = off.
    let mpc_footstep_cross_floor = cfg.mpc_cross.floor;
    // DIAG_VZ_BIAS_MPC / DIAG_VZ_BIAS_HEIGHT (m/s): a vertical CoM velocity
    // error seen by ONE consumer only -- the wrench MPC's initial state, or
    // the CoM height PD. Diagnostic for where SENSE_VEL_BIAS_Z acts; 0 = off.
    let diag_vz_bias_mpc = cfg.gains.diag_vz_bias_mpc;
    let mpc_net_cop_weight = cfg.mpc.net_cop_w;
    // MPC_FOOTSTEP: the swing foot's landing becomes a decision variable of
    // the wrench MPC (`wrench_mpc::Landing`, one SQP step about the planned
    // load). The chosen offset is applied the way the contact_plan adjuster
    // applies its shift -- `shift_next` (non-accumulating) and a rebuilt DCM
    // plan -- and the two are mutually exclusive: both on would be two
    // authorities over the same landing. Off by default.
    // Step-4 body, stage 1 (design note Sec.9): the landing recaptures the DCM.
    let mpc_landing_capture_w = cfg.mpc_footstep.capture_w;
    // Give the capture rows the plan's own sensitivity to the landing
    // (`Input::reference_dcm_sensitivity`), differenced from the real DCM
    // plan the way CP_J_AUTO does: shift the landing by 2 cm per axis,
    // rebuild the rolled plan, read the reference DCM at every node. Off =
    // the shipped rows, bit-identical.
    let mpc_landing_capture_j = cfg.mpc_footstep.capture_j;
    let mpc_w_pos = cfg.mpc.w_pos;
    let mpc_w_vel = cfg.mpc.w_vel;
    let mpc_w_mom = cfg.mpc.w_mom;
    let mpc_w_acc = cfg.mpc.w_acc;
    let mpc_w_wrench = cfg.mpc.w_wrench;
    // Decide the landing lexicographically (capture first, exactly, inside
    // the box; then the wrench MPC with it fixed) instead of through the
    // penalty weight above, which then only has to be > 0 to switch the
    // capture inputs on. See wrench_mpc::Config::landing_lexicographic.
    let mpc_landing_lex = cfg.mpc_footstep.lex;
    // 1 = shipped: the landing offset also enters the momentum and net-CoP
    // rows as `dp x F`. 0 = capture rows and landing cost only (2026-09-27:
    // the moment coupling made the landing jitter +-30-47 mm on
    // undisturbed steps, see the design note).
    let mpc_footstep_moment = cfg.mpc_footstep.moment;
    let mpc_w_smooth = cfg.mpc.w_smooth;
    // KKT complementarity tolerance of the MPC's solution check (see its use).
    let mpc_kkt_comp_tol = cfg.mpc.kkt_comp_tol;
    let mpc_backend_override = cfg.mpc.backend;
    let (dt, omega, total_mass) = (k.dt, k.omega, k.total_mass);
    let sole_half_l = k.sole_half_l;
    let sole_half_w = k.sole_half_w;
    let robot = &k.robot;
    let (f_max_per_foot, friction_mu, mpc_gate) = (k.f_max_per_foot, k.friction_mu, k.mpc_gate);
    let wrench_mpc_apply = k.wrench_mpc_apply;
    let (stair_wrench_dt, stair_wrench_preview_dt) = (k.stair_wrench_dt, k.stair_wrench_preview_dt);
    let contact_cfg = &k.contact_cfg;
    let st = obs_state(obs);
    let (com, com_vel, v_dvec, data) = (st.com, st.com_vel, &st.v_dvec, &st.data);
    let support = sup.support;
    let xi = dcm.xi;
    // The landing variable arms on the push triggers' error (PUSH_TRIG_LPF_HZ;
    // its norm is `dcm_err` itself when the filter is off).
    let dcm_err_trig = dcm.trig_err.norm();
    let a_com = *a_com;
    let mut mpc_capture_ns = 0u128;
    if stair_wrench_mpc && wrench_mpc_always {
        // A held result has no age limit (STAIR_WRENCH_MAX_AGE removed).
        if t + 1e-9 >= ctl.wrench_mpc_next {
            let horizon_time = stair_wrench_preview_dt * stair_wrench_horizon as f64;
            let weight = total_mass * G;
            // Where the MPC is asked to be at the end of its horizon.
            //
            // Historically this was the state PROPAGATED from now at the
            // DCM controller's current acceleration -- i.e. "wherever we
            // are already heading", which is a prediction, not a target.
            // An MPC given its own prediction has nothing to correct, and
            // measured 2026-09-24 the walking branch's horizontal target
            // carried no plan at all (only the height came from one).
            // (MPC_TARGET_PREVIEW, the horizontal target from the CoM
            // preview QP, was not adopted and is removed.)
            let mut target_com =
                com + com_vel * horizon_time + a_com * (0.5 * horizon_time * horizon_time);
            let mut target_velocity = com_vel + a_com * horizon_time;
            // `high` is the CURRENT stance foot, purely so the code below
            // has a name for "the loaded side"; the schedules are built
            // from the gait.
            let high = usize::from(!stance_sides.contains(&0));
            // The height reference is the walk's own.
            let (hz, hzd, _) = plan.height().at(t + horizon_time);
            target_com.z = hz;
            target_velocity.z = hzd;
            let low = 1 - high;
            let mut reference_fz_schedule = Vec::with_capacity(stair_wrench_horizon);
            let mut min_fz_schedule = Vec::with_capacity(stair_wrench_horizon);
            for k in 0..stair_wrench_horizon {
                // Sample the gait at the node's CENTRE, the same instant
                // `contact_schedule` uses.
                //
                // These two schedules used to disagree by half a node:
                // this one read `k * dt`, the contact set read
                // `(k + 0.5) * dt`. A node that straddles a lift-off then
                // gets "swing" from one and "double support" from the
                // other, and the MPC is handed a minimum vertical force on
                // a foot it has been told is in the air. Measured
                // 2026-09-24 with the MPC applied: the first node at
                // t=22.06 said contact [false, true] and min_fz
                // [40.2, 0.0] -- infeasible, the solve failed, the walker
                // held the last wrench across a support change and put
                // 770 N through the wrong foot.
                let stage_dt = if mpc_node_centre {
                    (k as f64 + 0.5) * stair_wrench_preview_dt
                } else {
                    k as f64 * stair_wrench_preview_dt
                };
                // The load split is the GAIT's, read forward along the
                // horizon. Double support is an even split; single
                // support puts the whole weight on the stance sole. This
                // is the schedule the walk is already executing, so the
                // MPC predicts the motion rather than a primitive.
                let high_share = {
                    let idx = plan.gait().index_at(t + stage_dt);
                    match plan.gait().slices[idx].support.swing() {
                        None => 0.5,
                        Some(sw) => {
                            if sw == high {
                                0.0
                            } else {
                                1.0
                            }
                        }
                    }
                };
                let mut reference = [0.0; 2];
                let mut minimum = [0.0; 2];
                reference[high] = high_share * weight;
                reference[low] = (1.0 - high_share) * weight;
                minimum[high] = stair_wrench_low_min * weight * high_share;
                minimum[low] = stair_wrench_low_min * weight * (1.0 - high_share);
                for side in 0..2 {
                    let capacity = f_max_per_foot;
                    minimum[side] = minimum[side].min(capacity);
                    reference[side] = reference[side].max(minimum[side]).min(capacity);
                }
                reference_fz_schedule.push(reference);
                min_fz_schedule.push(minimum);
            }
            let mut previous =
                ctl.wrench_mpc_command.unwrap_or([misa_mpc::wrench_mpc::Wrench::zeros(); 2]);
            // MPC_SEED_PREVIOUS: start the smoothness anchor at the
            // MEASURED load instead of at zero.
            //
            // The smoothness term penalises distance from `previous`, and
            // on the first solve `previous` is zero. With the weight that
            // kills the chatter (MPC_W_SMOOTH=10, 1e4x the default) the
            // commanded wrench then ramps up at ~150 N/s: measured
            // 2026-09-26, 148 N total at t=1.0 s and 113 N on the stance
            // leg as the first single support began, against 353 N the
            // WBC needed. Under MPC_SOLE_AUTHORITY that under-load IS the
            // regulariser target; the welded robot absorbed it (151/192),
            // the free-arm robot touched down early at 2.6 s and fell at
            // 3.2 s in half the members. The stair-transfer ramp already
            // seeds `previous` this way (`seed_previous`); this does the
            // same for the first solve of a walk. Off = zero seed = the
            // shipped behaviour.
            if mpc_seed_previous == 2 || (mpc_seed_previous == 1 && ctl.wrench_mpc_command.is_none()) {
                for side in 0..2 {
                    previous[side][5] = obs.contact_fz[side].max(0.0);
                }
            }
            let mut contacts: Vec<misa_mpc::wrench_mpc::Contact> = Vec::with_capacity(2);
            for side in 0..2 {
                let rotation = misarta::se3::rotation_matrix(&data.oMi[robot.foot_mi[side]]);
                let s_pitch = -rotation[(2, 0)];
                let rolling = stair_hybrid_contact && s_pitch.abs() >= 0.087;
                let leash = if cfg.contact.cop_leash_mpc && !rolling {
                    cop_leash_box(cfg, k, obs, sup, side, contact_cfg.cop_half)
                } else {
                    None
                };
                let cop_half = if let Some((_, h)) = leash {
                    h
                } else if rolling {
                    (0.006, contact_cfg.cop_half.1)
                } else if mpc_cop_scale_x < 1.0 {
                    (contact_cfg.cop_half.0 * mpc_cop_scale_x, contact_cfg.cop_half.1)
                } else {
                    contact_cfg.cop_half
                };
                // The WBC selector and MPC wrench must use the same
                // point. During rolling contact that point is the toe or
                // heel strip centre rather than the sole centre.
                let (patch_x, patch_y) = if let Some((c, _)) = leash {
                    (c[0], c[1])
                } else if rolling {
                    ((sole_half_l - cop_half.0) * s_pitch.signum(), 0.0)
                } else {
                    (0.0, 0.0)
                };
                let position =
                    actual.sole[side] + rotation * na::Vector3::new(patch_x, patch_y, 0.0);
                contacts.push(misa_mpc::wrench_mpc::Contact {
                    position,
                    rotation,
                    cop_half,
                });
            }
            // A foot in the air now but scheduled to be in contact within
            // the horizon: place its MPC contact at the forecast landing
            // rather than where it is mid-swing.
            if wrench_mpc_apply {
                let end = t + stair_wrench_horizon as f64 * stair_wrench_preview_dt;
                let fc = crate::preview::forecast_feet(
                    &plan.gait(),
                    &plan.footsteps(),
                    &actual,
                    &stance_sides,
                    t,
                    end - 1e-9,
                );
                for side in 0..2 {
                    if !stance_sides.contains(&side)
                        && (!mpc_forecast_swing_only
                            || matches!(plan.gait().support_at(t), Support::Single { swing, .. } if swing == side))
                    {
                        let yaw = fc.yaw[side];
                        let (sy, cy) = yaw.sin_cos();
                        let rot = na::Matrix3::new(cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0);
                        contacts[side].position = fc.sole[side];
                        contacts[side].rotation = rot;
                    }
                }
            }
            if mpc_auth_stair_off || mpc_auth_tilt > 0.0 {
                let in_stairs = mpc_auth_stair_off
                    && k.stairs.as_ref().is_some_and(|st| {
                        (st.riser.abs() > 1e-6 || st.down_n > 0)
                            && crate::height::stair_zone_crouch(
                                plan.footsteps().at_slice(plan.gait().index_at(t)).mid_xy().x,
                                st,
                                cfg.stair_zone.lead,
                                cfg.stair_zone.tail,
                                1.0,
                            ) > 0.0
                    });
                let tilted = mpc_auth_tilt > 0.0
                    && stance_sides.iter().any(|&s| {
                        let r = misarta::se3::rotation_matrix(&data.oMi[robot.foot_mi[s]]);
                        r[(2, 2)].clamp(-1.0, 1.0).acos() > mpc_auth_tilt
                    });
                ctl.mpc_auth_veto = in_stairs || tilted;
            }
            let cmm = cent.cmm();
            let h = cmm * v_dvec;
            // MPC_FOOTSTEP: in single support, the swing foot's landing is
            // a variable from the first node at which the schedule has it
            // down.
            // Inward (toward-stance) bound on the landing shift and which
            // sign of y is inward: MPC_FOOTSTEP_MAX_Y_IN. (The v6.13
            // backward parts MPC_FOOTSTEP_SHIFT_REST / _BACK_ONLY /
            // _BACK_PERSIST / _FLOOR_BACK_ERR / _BACK_VMAX / _HOLD_SHIFT /
            // _CAPTURE_FLOOR went to PUSH_REPLAN in v6.14 and are removed.)
            // MPC_FOOTSTEP_CROSS_Y gate (see its declaration).
            let cross_now: Option<(usize, bool)> = if mpc_footstep_cross_y > 0.0
                && let Support::Single { swing: sw_c, .. } = support
            {
                let i = plan.gait().index_at(t);
                if let Some((li, lp)) = ctl.mpc_fs_cross_latched
                    && li == i
                {
                    Some((li, lp))
                } else {
                    let stance_c = 1 - sw_c;
                    let f = plan.footsteps().at_slice(i);
                    // The stance foot lies at +y of the swing foot.
                    let plus = f.sole[stance_c].y > f.sole[sw_c].y;
                    let ex = (xi.y - f.sole[stance_c].y) * if plus { 1.0 } else { -1.0 };
                    if ex > mpc_footstep_cross_ex {
                        ctl.mpc_fs_cross_latched = Some((i, plus));
                        stats.n_mpc_fs_cross += 1;
                        if debug.mpc_fs_cross {
                            outln!(out, "  [fs-cross] t={t:.3} slice {i} swing {sw_c} DCM {ex:+.3} m past the stance sole");
                        }
                        Some((i, plus))
                    } else {
                        None
                    }
                }
            } else {
                None
            };
            let inward_bound = |i: usize| -> (f64, bool) {
                let (_, plus) = support
                    .swing()
                    .map(|sw| landing_inward(&plan.footsteps(), i, sw))
                    .unwrap_or((0.0, true));
                let mut b = mpc_footstep_max_y_in;
                if mpc_footstep_y_in_fwd > 0.0
                    && xi.x - plan.dcm().reference(t).xi.x >= mpc_footstep_y_in_fwd_trig
                {
                    b = b.max(mpc_footstep_y_in_fwd);
                }
                if let Some((ci, cplus)) = cross_now
                    && ci == i
                {
                    return (b.max(mpc_footstep_cross_y), cplus);
                }
                (b, plus)
            };
            let landing = if mpc_footstep
                && plan.claims().landing_owner(plan.gait().index_at(t)) != Some(Writer::PushReplan)
                && wrench_mpc_apply
                && let Support::Single { swing, .. } = support
                && !(mpc_footstep_stair_off
                    && k.stairs.as_ref().is_some_and(|st| {
                        (st.riser.abs() > 1e-6 || st.down_n > 0)
                            && crate::height::stair_zone_crouch(
                                plan.footsteps().at_slice(plan.gait().index_at(t)).mid_xy().x,
                                st,
                                cfg.stair_zone.lead,
                                cfg.stair_zone.tail,
                                1.0,
                            ) > 0.0
                    }))
                && {
                    let i = plan.gait().index_at(t);
                    if mpc_footstep_trig > 0.0
                        && dcm_err_trig >= mpc_footstep_trig
                        && !ctl.mpc_fs_armed.is_some_and(|(a, u)| a <= i && i <= u)
                    {
                        // A step is two slices (single + double support).
                        ctl.mpc_fs_armed = Some((i, i));
                    }
                    mpc_footstep_trig == 0.0
                        || ctl.mpc_fs_armed.is_some_and(|(a, u)| a <= i && i <= u)
                }
                && {
                    // (The freeze by swing phase, MPC_FOOTSTEP_FREEZE_FRAC /
                    // _RISE, defaulted to inf = never and is removed.)
                    // Event mode: no landing moves while the foot is
                    // probing for the ground (measured 2026-10-04: the
                    // goal moved 5-12 cm under a foot 2 cm from contact).
                    !plan.landing_closed(plan.gait().index_at(t))
                        && !claim_check!(plan.claims().whole_owner() == Some(Writer::RecoveryRef), ctl.ref_mode.is_some())
                }
            {
                (0..stair_wrench_horizon)
                    .find(|&k| {
                        let at = t + (k as f64 + 0.5) * stair_wrench_preview_dt;
                        match plan.gait().support_at(at) {
                            Support::Double => true,
                            Support::Single { swing: sw, .. } => sw != swing,
                        }
                    })
                    .map(|first_node| misa_mpc::wrench_mpc::Landing {
                        foot: swing,
                        first_node,
                        min_first_node: first_node,
                        position: [contacts[swing].position.x, contacts[swing].position.y],
                        // MPC_FOOTSTEP_TERRAIN_W (terrain as a cost) removed.
                        terrain_x: None,
                        weight: mpc_footstep_w,
                        max: {
                            let i = plan.gait().index_at(t);
                            let (inward, inward_is_plus) = inward_bound(i);
                            let mut m = mpc_footstep_max;
                            if inward_is_plus {
                                m[1] = if cross_now.is_some_and(|(ci, _)| ci == i) { inward } else { m[1].min(inward) };
                            }
                            m
                        },
                        // The answer is the total since the slice began
                        // (MPC_FOOTSTEP_INCREMENT removed).
                        prior: [0.0; 2],
                        min: {
                            let i = plan.gait().index_at(t);
                            // (MPC_FOOTSTEP_STAIR_BACK, inf by default, removed.)
                            let mut m = mpc_footstep_max;
                            let (inward, inward_is_plus) = inward_bound(i);
                            if !inward_is_plus {
                                m[1] = if cross_now.is_some_and(|(ci, _)| ci == i) { inward } else { m[1].min(inward) };
                            }
                            m
                        },
                    })
                    .map(|mut land| {
                        // Crossing: the LIPM capture floor on the inward shift.
                        if mpc_footstep_cross_floor > 0.0
                            && let Some((ci, cplus)) = cross_now
                            && ci == plan.gait().index_at(t)
                        {
                            let sl = &plan.gait().slices[ci];
                            let rem = (sl.t1 - t).max(0.0);
                            let sgn = if cplus { 1.0 } else { -1.0 };
                            let stance_y = plan.footsteps().at_slice(ci).sole[1 - land.foot].y;
                            let edge = stance_y + sgn * sole_half_w;
                            let xi_td = edge + (xi.y - edge) * (omega * rem).exp();
                            // The ORIGINAL landing: the contact position already
                            // carries the shift applied so far this slice, and
                            // the box acts on the total shift from the original.
                            let applied_y = if ctl.mpc_fs_slice == Some(ci) { ctl.mpc_fs_applied.y } else { 0.0 };
                            let base = land.position[1] - applied_y;
                            // Inward shift needed for the landing to come within
                            // floor * half width of xi_td.
                            let need = (xi_td - sgn * mpc_footstep_cross_floor * sole_half_w - base) * sgn;
                            if need > 0.0 {
                                if cplus {
                                    land.min[1] = land.min[1].min(-need);
                                    land.max[1] = land.max[1].max(need);
                                } else {
                                    land.max[1] = land.max[1].min(-need);
                                    land.min[1] = land.min[1].max(need);
                                }
                            }
                            if debug.mpc_fs_cross {
                                outln!(out, "  [fs-cross-floor] t={t:.3} rem={rem:.3} xi.y={:+.3} xi_td={xi_td:+.3} base={base:+.3} need={need:+.3} max={:?} min={:?}", xi.y, land.max, land.min);
                            }
                        }
                        land
                    })
            } else {
                None
            };
            let input = misa_mpc::wrench_mpc::Input {
                com,
                velocity: if diag_vz_bias_mpc != 0.0 {
                    com_vel + na::Vector3::new(0.0, 0.0, diag_vz_bias_mpc)
                } else {
                    com_vel
                },
                angular_momentum: na::Vector3::new(h[0], h[1], h[2]),
                target_com,
                target_velocity,
                target_acceleration: a_com,
                com_z_max: None,
                contacts: [contacts[0].clone(), contacts[1].clone()],
                // Walking branch: the contact SET over the horizon comes
                // from the gait, node by node (lift-off and landing inside
                // the horizon), the same way the CoM preview builds its
                // support boxes. Repeating the current stance for 0.4 s
                // -- the historical construction, kept for transfers --
                // gave a solution that, once applied (WRENCH_MPC_APPLY),
                // could not walk at all (0/192, measured 2026-09-24).
                contact_schedule: if wrench_mpc_apply {
                    let i_now = plan.gait().index_at(t);
                    let early_swing = match plan.gait().support_at(t) {
                        Support::Single { swing, .. }
                            if mpc_schedule_early_fz > 0.0 && obs.contact_fz[swing] > mpc_schedule_early_fz =>
                        {
                            Some(swing)
                        }
                        _ => None,
                    };
                    (0..stair_wrench_horizon)
                        .map(|k| {
                            let at = t + (k as f64 + 0.5) * stair_wrench_preview_dt;
                            match plan.gait().support_at(at) {
                                Support::Double => [true, true],
                                Support::Single { swing, .. }
                                    if early_swing == Some(swing) && plan.gait().index_at(at) == i_now =>
                                {
                                    [true, true]
                                }
                                Support::Single { swing, .. } => [swing != 0, swing != 1],
                            }
                        })
                        .collect()
                } else {
                    vec![
                        [stance_sides.contains(&0), stance_sides.contains(&1)];
                        stair_wrench_horizon
                    ]
                },
                // What fraction of its capacity each foot may use at
                // each node: all of it. (MPC_LOAD_SCHEDULE, the WBC's
                // double-support load ramp shown to the MPC, was not
                // adopted and is removed.)
                activation_schedule: vec![[1.0; 2]; stair_wrench_horizon],
                reference_fz_schedule,
                // Where the walk wants the pressure, node by node, in
                // each sole's own frame. Without it the MPC has no
                // opinion about the CoP and picks the split its wrench
                // terms like, which is a different CoP from the one the
                // DCM plan is steering by (measured 2026-09-25: totals
                // agree to 1%, the split differs by 40-55 N, and applying
                // the MPC's answer cannot walk at any blend).
                reference_cop_schedule: None,
                // Where the walk wants the NET pressure of both feet,
                // node by node, in world. This is what the DCM plan
                // actually constrains; asking each sole for its own CoP
                // made the disagreement worse (43 -> 63 N, 2026-09-25).
                // (MPC_COP_REF_CMD, the schedule shifted to the WBC's
                // p_cmd, was not adopted -- the champion sets 0 -- and is
                // removed; it added an exact zero when off.)
                reference_net_cop_schedule: (mpc_net_cop_weight > 0.0)
                    .then(|| {
                    (0..stair_wrench_horizon)
                        .map(|k| {
                            let at = t + (k as f64 + 0.5) * stair_wrench_preview_dt;
                            plan.dcm().reference(at).p
                        })
                        .collect()
                }),
                reference_dcm_schedule: (mpc_landing_capture_w > 0.0)
                    .then(|| {
                    (0..stair_wrench_horizon)
                        .map(|k| {
                            let at = t + (k as f64 + 0.5) * stair_wrench_preview_dt;
                            plan.dcm().reference(at).xi
                        })
                        .collect()
                }),
                min_fz_schedule,
                previous,
                landing,
                dcm_now: (mpc_landing_capture_w > 0.0).then_some(xi),
                reference_dcm_sensitivity: (mpc_landing_capture_j
                    && mpc_landing_capture_w > 0.0)
                    .then(|| landing)
                    .flatten()
                    .map(|land| {
                        let i = plan.gait().index_at(t);
                        let du = 0.02;
                        let mut cols = [Vec::new(), Vec::new()];
                        for axis in 0..2 {
                            let mut probe = plan.footsteps().clone();
                            let mut d = na::Vector2::zeros();
                            d[axis] = du;
                            // The sensitivity describes the plan the shift
                            // is actually applied to.
                            probe.shift_next(i, land.foot, d);
                            let moved = plan.dcm_for(&probe);
                            cols[axis] = (0..stair_wrench_horizon)
                                .map(|k| {
                                    let at = t + (k as f64 + 0.5) * stair_wrench_preview_dt;
                                    (moved.reference(at).xi - plan.dcm().reference(at).xi) / du
                                })
                                .collect();
                        }
                        (0..stair_wrench_horizon)
                            .map(|k| {
                                na::Matrix2::from_columns(&[cols[0][k], cols[1][k]])
                            })
                            .collect()
                    }),
                // MPC_ARM_ABOUT_PRIOR (removed) was the only user.
                prior_com: None,
            };
            let mut config = misa_mpc::wrench_mpc::Config {
                horizon: stair_wrench_horizon,
                dt: stair_wrench_preview_dt,
                mass: total_mass,
                gravity: G,
                friction_mu,
                cop_half: contact_cfg.cop_half,
                torsion_mu: contact_cfg.mu_torsion,
                // (MPC_F_MAX_SCALE, a ceiling of its own, is removed:
                // its default NaN meant the WBC's.)
                max_fz: f_max_per_foot,
                validate_solution: mpc_gate,
                // The cost weights, so the ones that compete with the CoP
                // term can be measured against it. Defaults are the
                // shipped values, so leaving them alone is bit-identical.
                // The removed experiments (MPC_DCM_W, MPC_LANDING_CAPTURE_N,
                // MPC_LANDING_TIME*, MPC_FZ_SPLIT_EQ, MPC_CAPTURE_COP_W,
                // MPC_CAPTURE_LEX*, MPC_ARM_ABOUT_PRIOR, MPC_W_LDOT,
                // MPC_COP_W, MPC_NET_COP_EQ, MPC_REACH_W) keep `Config::default()`,
                // which is what their defaults were.
                position_weight: mpc_w_pos,
                velocity_weight: mpc_w_vel,
                momentum_weight: mpc_w_mom,
                acceleration_weight: mpc_w_acc,
                wrench_weight: mpc_w_wrench,
                omega,
                landing_capture_weight: mpc_landing_capture_w,
                landing_lexicographic: mpc_landing_lex,
                landing_moment_coupling: mpc_footstep_moment,
                smooth_weight: mpc_w_smooth,
                net_cop_weight: mpc_net_cop_weight,
                ..Default::default()
            };
            config.qp.absolute_gap = mpc_gate;
            // Interior-point solutions carry complementarity slack at the
            // gap setting; the KKT gate's 1e-6 rejected every one.
            config.kkt_complementarity_tolerance = mpc_kkt_comp_tol;
            if let Some(backend) = mpc_backend_override {
                config.qp.solver = backend;
            }
            config.trace_landing_time = cfg.mpc.trace_landing_time;
            config.trace_failures = cfg.mpc.trace_failures;
            if diag.wants_mpc_input() {
                let capture_start = std::time::Instant::now();
                diag.mpc_input(t, dt, stair_wrench_dt, &input, &config);
                mpc_capture_ns = capture_start.elapsed().as_nanos();
            }
            let previous_age = ctl.diag_mpc_time.map_or(-1.0, |stamp| t - stamp);
            let mpc_solve_start = std::time::Instant::now();
            let mut mpc_trace = Vec::new();
            let mpc_result = misa_mpc::wrench_mpc::solve_traced(
                &input,
                &config,
                None, // no warm start (MPC_QP_WARM removed)
                &mut mpc_trace,
            );
            let mpc_solve_ns = mpc_solve_start.elapsed().as_nanos();
            for line in &mpc_trace {
                out.err_line(line);
            }
            {
                let mask = u8::from(stance_sides.contains(&0))
                    | (u8::from(stance_sides.contains(&1)) << 1);
                diag.mpc_runtime(t, mask, &config, stair_wrench_dt, mpc_solve_ns, mpc_result.is_some(), previous_age);
            }
            match mpc_result {
                Some(solution) => {
                    let mut by_side = [misa_mpc::wrench_mpc::Wrench::zeros(); 2];
                    by_side.copy_from_slice(&solution.sole_wrenches);
                    ctl.diag_mpc_reference = input.reference_fz_schedule[0];
                    ctl.diag_mpc_time = Some(t);
                    ctl.wrench_mpc_command = Some(by_side);
                    {
                        let (mut mx, mut my, mut fz) = (0.0, 0.0, 0.0);
                        for foot in 0..2 {
                            let c = &input.contacts[foot];
                            let w = by_side[foot];
                            let m = c.rotation * na::Vector3::new(w[0], w[1], w[2]);
                            let f = c.rotation * na::Vector3::new(w[3], w[4], w[5]);
                            // Moment of this contact about the world origin,
                            // horizontal components only.
                            mx += m.x + c.position.y * f.z - c.position.z * f.y;
                            my += m.y - c.position.x * f.z + c.position.z * f.x;
                            fz += f.z;
                        }
                        ctl.diag_mpc_net_cop = if fz > 1.0 {
                            [-my / fz, mx / fz]
                        } else {
                            [f64::NAN; 2]
                        };
                    }
                    if diag.keeps_mpc_frames() {
                        ctl.wrench_mpc_frames = Some(input.contacts.clone());
                    }
                    diag.mpc_solved(t, &com, &input, &solution, ctl.diag_mpc_net_cop, landing.as_ref(), &by_side);
                    // (The lexicographic CoP's soft fallback and its
                    // MPC_LEX_DEBUG_* prints went with MPC_CAPTURE_LEX.)
                    if let Some(dp) = solution.landing_shift
                        && let Some(land) = landing
                        && !plan.landing_closed(plan.gait().index_at(t))
                    {
                        let i = plan.gait().index_at(t);
                        if ctl.mpc_fs_slice != Some(i) {
                            ctl.mpc_fs_slice = Some(i);
                            ctl.mpc_fs_applied = na::Vector2::zeros();
                        }
                        let (d, total) = (dp - ctl.mpc_fs_applied, dp);
                        ctl.mpc_fs_last = total;
                        if debug.mpc_fs_landing {
                            outln!(out, 
                                "  [fs-decision] t={t:.3} first_node={} prior=({:+.3},{:+.3}) dp=({:+.3},{:+.3}) total=({:+.3},{:+.3}) max={:?} min={:?} xi-ref=({:+.3},{:+.3})",
                                land.first_node, land.prior[0], land.prior[1], dp.x, dp.y, total.x, total.y, land.max, land.min,
                                xi.x - plan.dcm().reference(t).xi.x, xi.y - plan.dcm().reference(t).xi.y
                            );
                        }
                        if d.norm() > 1e-6 {
                            let mut ed = plan.edit(Writer::MpcLanding);
                            ed.shift_next(i, land.foot, d);
                            ctl.mpc_fs_applied = total;
                            ed.commit();
                            stats.n_replan += 1;
                            if debug.replan { outln!(out, "  [diag-replan] site 6 t={t:.3} (line 9380)"); }
                        }
                    }
                    if debug.mpc_traj && t < 2.0 {
                        let xi0 = com.xy() + com_vel.xy() / omega;
                        let r0 = plan.dcm().reference(t);
                        let mut line = format!("  [traj] t={t:.3} xi_meas=({:+.3},{:+.3}) xi_plan=({:+.3},{:+.3}) tgt=({:+.3},{:+.3}) |",
                            xi0.x, xi0.y, r0.xi.x, r0.xi.y, target_com.x, target_com.y);
                        for (k, [c, v]) in solution.trajectory.iter().enumerate() {
                            let xi = c.xy() + v.xy() / omega;
                            let rp = plan.dcm().reference(t + (k as f64 + 1.0) * stair_wrench_preview_dt);
                            line += &format!(" k{k}: pred({:+.3},{:+.3}) plan({:+.3},{:+.3})", xi.x, xi.y, rp.xi.x, rp.xi.y);
                        }
                        out.line(&line);
                    }
                    stats.wrench_mpc_solves += 1;
                }
                None => {
                    stats.wrench_mpc_failures += 1;
                }
            }
            ctl.wrench_mpc_next = t + stair_wrench_dt;
        }
    } else {
        ctl.wrench_mpc_command = None;
        ctl.wrench_mpc_next = t;
    }
    mpc_capture_ns
}

/// What phase 10 computed for the levels: the trunk's Jacobian and bias,
/// the trunk's roll and pitch (the estimate), their reference, the yaw
/// reference and the full orientation command; whether the coupled level is
/// on; the momentum, MOM_CMP and arm-hold inputs; the gravity torque and the
/// load shares; the foot out of contact and its task inputs, target and
/// commanded vertical acceleration (diagnostics).
pub(crate) struct TaskRefs {
    pub j_trunk: na::DMatrix<f64>,
    pub djv_trunk: na::DVector<f64>,
    pub roll: f64,
    pub pitch: f64,
    pub rp_ref: [f64; 2],
    pub yaw_ref: Option<f64>,
    pub ori: [f64; 3],
    pub coupled_active: bool,
    pub mom: Option<MomRef>,
    pub cmp: Option<CmpRef>,
    pub arm: Option<ArmRef>,
    pub tau_gravity: na::DVector<f64>,
    pub shares: Vec<f64>,
    pub free_side: Option<usize>,
    pub swing: Option<SwingRef>,
    pub swing_tgt: na::Vector3<f64>,
    pub swing_cmd_acc_z: f64,
}

/// Phase 10: the references of the WBC's levels, computed before any level
/// is built: the trunk (roll / pitch estimate, the planned yaw, the
/// orientation command), the centroidal momentum (MOM, arms_neutral), the
/// hip strategy (MOM_CMP), the arm hold (with the jab), the gravity torque
/// and the load split, and the swing foot's targets (the planned arc, the
/// stair profiles, the holds, the recovery reference's sole path, the probe
/// depth) with its axes and orientation target. The levels themselves are
/// phase 11's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn phase_task_refs(
    ctl: &CtlState,
    plan: &Plan,
    stats: &mut WalkerStats,
    cfg: &Config,
    k: &WalkerConsts,
    obs: &TickObs,
    cent: &Centroidal,
    sup: &SupportOut,
    stance_sides: &[usize],
    stance: &[usize],
    nc: usize,
    ramp: &LoadRamp,
    dcm: &DcmOut,
    p_cmd: &na::Vector2<f64>,
    zmp_deficit: &na::Vector2<f64>,
    t: f64,
) -> TaskRefs {
    // TURN_IN_DS: the trunk turns toward a new heading during the double
    // support, not the single support before it (see its use).
    let turn_in_ds = cfg.gait.turn_in_ds;
    // Regulate the trunk's yaw as well as its roll and pitch.
    let trunk_yaw = cfg.gains.trunk_yaw;
    let kp_trunk = cfg.gains.kp_trunk;
    let kd_trunk = cfg.gains.kd_trunk;
    let trunk_sign = cfg.gains.trunk_sign;
    let kp_yaw = cfg.gains.kp_yaw;
    let kd_yaw = cfg.gains.kd_yaw;
    let stair_wrench_mpc = cfg.mpc.enabled;
    let stair_wrench_coupled = cfg.mpc.coupled;
    // Centroidal angular-momentum task (doc Sec.21.3, reopened by the v6
    // shoulder roll -- see `bt::angular_momentum`). `MOM_AXES` is a 3-char
    // mask over roll/pitch/yaw, e.g. "x--" for roll only. Default OFF: it
    // costs two extra CMM evaluations a tick and has never been measured.
    let use_mom = cfg.arm.mom;
    // 角運動量タスクが「追従」する参照の作り方。
    //   zero         : h_ref = 0 = 従来(抑制)。歩行と両立しないことを実測済み。
    //   arms_neutral : h_ref = h - A_arms·v_arms、つまり残差が「腕の寄与そのもの」に
    //                  なる。タスクは「腕の角運動量寄与をゼロに。脚の分には触るな」。
    // 本来は A2 の可行な全身参照から h_ref(t) を取るべきで、これはその足場。
    let mom_ref_mode = cfg.arm.mom_ref.clone();
    // Ask the ARMS for roll angular momentum when the lateral DCM escapes.
    //
    // This is the eCMP/VRP mechanism (Englsberger IROS 2013) written as a task
    // reference. The centroidal moment pivot is the ZMP displaced by the rate
    // of change of angular momentum,
    //
    //     CMP_y = ZMP_y - h_dot_x / (m g)
    //
    // and the DCM is driven by the CMP, not the ZMP. So when the ZMP is pinned
    // to the lateral sole edge -- measured as the terminal state of every
    // 60 mm-riser fall, `zmp_y` clamped at -0.0919 for 2 s -- the only pivot
    // left is angular momentum. Driving the DCM back toward its reference in
    // +y needs CMP_y to move +y, hence h_dot_x NEGATIVE: the gain is applied
    // with a minus sign below.
    //
    // Only meaningful with `MOM_REF=arms_neutral`, where the task's Jacobian
    // is masked to the arm columns: the reference then says "the ARMS shall
    // produce this roll momentum rate" and the legs' own momentum is left
    // alone. With the unmasked CMM the same reference would instead forbid the
    // legs the momentum they make by construction -- the failure measured
    // 2026-09-20 (arm travel 1.8 -> 297 deg, unmoved by a 60x gain sweep).
    //
    // Default 0.0 = the zero reference, i.e. previous behaviour.
    let mom_arm_dcm = cfg.arm.mom_arm_dcm;
    // Ceiling on the requested rate. The arms are light and near the CoM, so a
    // large request is one the solver cannot fill and will chase to the joint
    // limits; this bounds the ask rather than the gain.
    let mom_arm_dcm_max = cfg.arm.mom_arm_dcm_max;
    let kp_mom = cfg.arm.kp_mom;
    let mom_axes = cfg.arm.mom_axes;
    let stair_crouch_lead = cfg.stair_zone.lead;
    let stair_crouch_tail = cfg.stair_zone.tail;
    // MOM_CMP: the hip strategy, written as the centroidal moment pivot.
    // When the DCM law's pressure point saturates at the support edge, the
    // deficit `p_raw - p_cmd` is pressure the polygon cannot give; the same
    // horizontal force follows from an angular-momentum rate instead
    // (CMP_y = CoP_y - h_dot_x / (m g), CMP_x = CoP_x + h_dot_y / (m g)).
    // This asks the WHOLE body (unmasked CMM, roll and pitch) for
    //     h_dot = MOM_CMP * m g * (-deficit_y, +deficit_x, 0),
    // capped at MOM_CMP_MAX [N m], with no P term, as a weighted task in
    // the centroidal level. The arms-only MOM_ARM_DCM task stays as it is.
    // Design: doc/from_claude/20261012_restep_design.md Sec.6 (the WBC's
    // lateral authority ends at the sole edge). 0 = off, bit-identical.
    let mom_cmp = cfg.mom_cmp.gain;
    let mom_cmp_dead = cfg.mom_cmp.dead;
    // Sign check (+1 = the derivation above) and a cap on the angular
    // momentum the strategy may build up [N m s]: a hip strategy is a
    // transient -- past the cap the rate in that direction is cut to zero
    // and a P term winds the momentum back (MOM_CMP_KH [1/s]).
    let mom_cmp_sign = cfg.mom_cmp.sign;
    let mom_cmp_max = cfg.mom_cmp.max;
    let mom_cmp_hmax = cfg.mom_cmp.hmax;
    let mom_cmp_kh = cfg.mom_cmp.kh;
    // Hold the arms, at their own level ABOVE posture.
    //
    // Sec.29.6 measured the v6 arm swinging 111 deg peak-to-peak at the
    // shoulder roll and dragging pitch and elbow to 64/26 deg with it, where
    // every pre-v6 model and `v6fixed` sit at 2-10 deg. The posture level is
    // the only thing regulating any of it, and it runs at one uniform gain
    // BELOW swing -- for the legs that is harmless because the contact/CoM/
    // swing tasks pin them anyway, and for an arm in every one of those
    // tasks' null space it is the whole budget.
    //
    // NOT free, though it looked like it should be: the arms carry no load, so
    // a level of their own reads like it can only take null space nothing else
    // wanted. Measured (Sec.30.3), the same task on v4 -- a model with no
    // shoulder roll and no problem -- costs 22/28 -> 16/28. On v6 it is a net
    // win only because the flailing was costing more than the level does.
    // Default OFF for that reason.
    let arm_hold = cfg.arm.hold;
    // WHICH arm rows to hold. Holding all of them keeps a commanded pose but
    // costs the top lateral rung: at VY=0.073 the held arm survives 1 of 16
    // probes where the free arm survives 16 (Sec.38). The free arm keeps the
    // guard's SHAPE anyway -- measured, the elbow sits at -128 deg against a
    // -130 seed and travels 23 deg, while the shoulder swings 132 deg about
    // its seed. So the shoulder swing is what the walk wants and the elbow
    // fold is what the guard wants, and they are separable.
    //   all      -- shoulder + elbow (what Sec.30 measured)
    //   elbow    -- fold held, shoulder free to swing
    //   shoulder -- the opposite, for symmetry of the experiment
    let arm_hold_joints = cfg.arm.hold_joints.clone();
    // 0 = translation, 1 = +yaw, 2 = full pose, 3 = XYZ+roll+pitch (no yaw).
    let stair_absolute_td = cfg.swing.absolute_td;
    // STAIR_ABSOLUTE_TD_DESC=1: the surveyed-tread touchdown height for
    // DESCENDING swings only. Measured 2026-10-19 with the acceleration
    // feedforward (which tracks the target to a few mm): a descent target
    // built from the lift-off height sits 6 cm above the lower tread when
    // the foot lifted off hovering, the precise swing then never touches and
    // the 60 mm round trip falls; the absolute height fixes that, but on
    // the 80 mm ascent's top landing it costs the run (foot lands 4-10 cm
    // short, the 4-step 80 mm ascent 48 -> fall). Descent-only keeps both.
    let stair_absolute_td_desc = cfg.swing.absolute_td_desc;
    // Full per-foot wrench MPC for the three-phase stair transfer.
    // Opt-in until the fixed-arm staircase benchmark has been revalidated.
    // Command the swing to the ANKLE pose the footstep plan implies, rather
    // than to the sole pose directly. See the use site for the measurement.
    let stair_td_sole_frame = cfg.swing.td_sole_frame;
    let com_replan = cfg.com_plan.replan;
    let step_dz_swing_min = cfg.swing.step_dz_min;
    let stair_swing_ff = cfg.swing.stair_ff;
    let lift_h = cfg.swing.lift_h;
    // (CONTACT_FZ_RATE, a load-split damper on the measured normal force,
    // was FALSIFIED -- worse than undamped at every weight -- and is
    // removed, as is DS_FZ_FLOOR_WEIGHT / _BW, a soft double-support load
    // floor that was not adopted.)
    let stair_swing_windows = cfg.swing.stair_windows;
    // (STAIR_DESC_MIN, riding the lower of the two soles down a descent, and
    // STAIR_CROUCH, a fixed crouch on stairs, were opt-in and are removed.)
    // Phase windows of the stair swing, `rise_end,travel_start,travel_end,
    // drop_start` (see gait::stair_swing_windows). The default is the
    // shipped profile.
    // Landing overdrive of the stair swing (metres COMMANDED below the
    // surface). See gait::stair_swing_sink.
    let stair_td_sink = cfg.swing.td_sink;
    let stair_land_sink = cfg.swing.land_sink;
    // Controlled ablation: retain the cubic profile and add its analytic acceleration.
    let tracking_cubic_ff = cfg.swing.tracking_cubic_ff;
    let swing_z_hold = cfg.swing.z_hold;
    let swing_z_hold_min_dur = cfg.swing.z_hold_min_dur;
    let swing_z_drop = cfg.swing.z_drop;
    // (STAIR_DESC_TOE, a toe-first descent landing, and SWING_TOE_UP / _W,
    // the toe lifted through every swing, were not adopted and are removed.)
    // SWING_XY_DELAY / SWING_Z_HOLD / SWING_Z_DROP: ROUGH_MODE sets them.
    let swing_xy_delay = cfg.swing.xy_delay;
    let push_replan_trig = cfg.push_replan.trig;
    let push_replan_hold = cfg.push_replan.hold;
    // MPC_FOOTSTEP_CROSS_HOLD_Z [m]: while crossing, hold the swing foot at
    // least this high above its landing until it is within
    // MPC_FOOTSTEP_CROSS_HOLD_XY [m] of the landing horizontally. Measured
    // 2026-10-20: with the crossing landing fixed the foot still came down on
    // the nominal clock 5 cm short of it (the lateral travel lags), touched
    // at y +0.01 and slid. The landing check then extends the single support
    // (LandingTiming) instead of the foot landing short. 0 = off.
    let mpc_footstep_cross_hold_z = cfg.mpc_cross.hold_z;
    let mpc_footstep_cross_hold_xy = cfg.mpc_cross.hold_xy;
    let ref_swing = cfg.recovery.swing;
    let tracking_ff_scale = cfg.swing.tracking_ff_scale;
    // Swing. kp is split by axis: z is clearance and must not scuff, x/y is
    // placement, and while the stance leg is paying the reaction for both it
    // is x/y that can afford to be soft. Holding the swing foot's world x,y
    // HARD is what toppled the single-leg experiment (section 4.2) -- there
    // the target was frozen while the body translated 70 mm, which is a
    // kinematic conflict. Here the target is a planned footstep, so the
    // conflict is gone in principle; the split gain is how much that is
    // trusted in practice.
    let kp_sw_z = cfg.swing.kp;
    let kd_sw = cfg.swing.kd;
    // TERRAIN_PITCH=1: on a known slope the swing foot's orientation target
    // is pitched to the surface at its landing (toe up on an upslope) and
    // pitch is tracked (SwingAxes::XyzRollPitch). A level sole meets a 5 deg
    // slope on its toe or heel only (measured 2026-10-15: the sole centre
    // hovered 7 mm above the surface with no load, then the stance rolled).
    // Off = shipped, bit-identical.
    let terrain_pitch = cfg.slope.terrain_pitch;
    let terrain_pitch_yaw = cfg.slope.terrain_pitch_yaw;
    let kp_sw_rot = cfg.swing.kp_rot;
    let kd_sw_rot = cfg.swing.kd_rot;
    let (dt, total_mass) = (k.dt, k.total_mass);
    let robot = &k.robot;
    let prof = &robot.prof;
    let stairs = &k.stairs;
    let (wz, t_step, gp) = (k.wz, k.t_step, &k.gp);
    let wrench_mpc_apply = k.wrench_mpc_apply;
    let (jab, jab_start) = (k.command.jab, k.command.jab_start);
    let (jab_pitch, jab_elbow) = (k.command.jab_pitch, k.command.jab_elbow);
    let (kp_sw_xy, tracking_ff_riser_ok) = (k.kp_sw_xy, k.tracking_ff_riser_ok);
    // Rotate the nominal foot orientation to a planned yaw. A turn moves the
    // footstep AND the direction the foot points; commanding the original
    // orientation would land the foot square to the world under a turned body.
    let foot_rot_at = |side: usize, yaw: f64| -> na::Matrix3<f64> {
        let d = yaw - k.foot_rot0[side][(1, 0)].atan2(k.foot_rot0[side][(0, 0)]);
        na::Rotation3::from_euler_angles(0.0, 0.0, d).matrix() * k.foot_rot0[side]
    };
    let st = obs_state(obs);
    let (q, v, v_dvec, data) = (&st.q, &st.v, &st.v_dvec, &st.data);
    let (support, slice, slice_frac, slice_idx) = (sup.support, sup.slice, sup.slice_frac, sup.slice_idx);
    let (steps, swing_rot_mode) = (&sup.steps, sup.swing_rot_mode);
    let (xi, r) = (dcm.xi, &dcm.r);
    let (p_cmd, zmp_deficit) = (*p_cmd, *zmp_deficit);

    // ---- P2: trunk upright -----------------------------------------
    let j_trunk =
        misarta::jacobian::compute_joint_jacobian_from_data(&robot.model, q, data, robot.trunk_mi);
    let dj_trunk = misarta::jacobian::compute_joint_jacobian_time_derivative(
        &robot.model,
        q,
        v,
        robot.trunk_mi,
    );
    let djv_trunk = &dj_trunk * v_dvec;
    let (roll, pitch, trunk_quat, trunk_omega_w) = if robot.trunk_from_base {
        let (r, p, _) = st.body_quat.euler_angles();
        (r, p, st.body_quat, st.v_ang_w)
    } else {
        let rot = misarta::se3::rotation_matrix(&data.oMi[robot.trunk_mi]);
        let omega = j_trunk.rows(0, 3) * v_dvec;
        let rotation = na::Rotation3::from_matrix_unchecked(rot);
        (
            rotation[(2, 1)].atan2(rotation[(2, 2)]),
            (-rotation[(2, 0)]).asin(),
            na::UnitQuaternion::from_rotation_matrix(&rotation),
            [omega[0], omega[1], omega[2]],
        )
    };
    let _ = (roll, pitch);
    // Yaw error against the pose the plan was built in, not against the
    // last tick: the reference is a fixed world direction.
    // Trunk yaw follows the PLAN, not the initial heading.
    //
    // Regulating to `yaw0` with a zero rate reference is right for
    // straight walking -- it is what stopped the legs twisting to their
    // hip_yaw stops -- and it makes turning impossible, because the task
    // then fights the commanded turn directly. Measured: every turn
    // command in the benchmark fell, down to 0.05 rad/s. The planned
    // heading is the mean of the two footstep yaws, which already carries
    // the stride ramp.
    // WHEN the body turns matters as much as how much.
    //
    // Yawing during SINGLE support has to be paid for out of the stance
    // sole's torsional friction -- mu_torsion 0.05 against 65 N is
    // 3.25 N*m, on a 38 mm-wide foot. Yawing during DOUBLE support is paid
    // for by a couple between two feet 99.4 mm apart, which is a different
    // order of authority. So hold the heading through single support and
    // turn while both feet are down.
    //
    // `at_slice(i)` already has the swing foot at its LANDING yaw, so
    // reading it during single support asks the trunk to rotate before the
    // foot that will carry the rotation is even down.
    let yaw_plan = {
        let mean = |i: usize| {
            let y = plan.footsteps().at_slice(i).yaw;
            0.5 * (y[0] + y[1])
        };
        let prev = mean(slice_idx.saturating_sub(1));
        if !turn_in_ds {
            mean(slice_idx)
        } else if matches!(support, Support::Single { .. }) {
            prev
        } else {
            // Rotate across the double support.
            prev + (mean(slice_idx) - prev) * slice_frac
        }
    };
    // Rate reference follows the same shape as the heading: zero while a
    // single foot is carrying the robot, the whole turn compressed into DS.
    let wz_ref = if turn_in_ds {
        if matches!(support, Support::Single { .. }) {
            0.0
        } else {
            wz * (t_step / gp.t_ds.max(1e-6))
        }
    } else {
        wz
    };
    // Now an absolute target heading -- trunk_ori_ref forms the error.
    let yaw_ref = trunk_yaw.then_some(yaw_plan);
    // One world-frame rotation-vector error for all three rows. The
    // roll/pitch rows are frame-correct at any heading this way; the old
    // Euler form silently rotated its own correction away.
    let ori = trunk_ori_ref(
        &trunk_quat,
        yaw_ref,
        &trunk_omega_w,
        TrunkGains {
            kp: kp_trunk,
            kd: kd_trunk,
            deadband: 0.0,
            sign: trunk_sign,
            kp_yaw,
            kd_yaw,
            wz_ref,
        },
    );
    let rp_ref = [ori[0], ori[1]];

    // ---- centroidal angular momentum --------------------------------
    //
    // `A_G` and `Ȧ_G v` are recomputed here rather than cached: the CMM is
    // a function of q, and this loop's whole point is that q moves.
    let coupled_active = stair_wrench_mpc
        && stair_wrench_coupled
        && wrench_mpc_apply
        && ctl.wrench_mpc_command.is_some()
        && nc > 0;
    let mom = if use_mom || coupled_active {
        let cmm = cent.cmm();
        let dcmm_v = cent.dcmm_v();
        let h_now = cmm * v_dvec;
        // arms_neutral: 「腕が角運動量を生むな。脚の分には触るな」。
        //
        // 参照だけ差し替えるのでは表現できない ── ヤコビアンが全 CMM 行の
        // ままだと、式は ḣ_total = -kp·h_arms という意味を成さない目的になる
        // (実測: 腕の振れが 299.8 -> 318.2 deg と増えた)。腕の列だけを残した
        // CMM を渡し、状態も腕の寄与だけにする。
        //
        // Ȧ·v の腕成分は直接得られないのでゼロとする。減衰型の目的としては
        // 許容できる近似で、本来 A2 が h_ref(t) を供給すれば不要になる経路。
        let arm_cols: Vec<usize> = robot
            .actuated()
            .into_iter()
            .filter(|&(ji, _)| {
                let n = &robot.joint_names[ji];
                n.contains("shoulder") || n.contains("elbow") || n.contains("wrist")
            })
            .map(|(_, vi)| vi)
            .collect();
        let arms_neutral = mom_ref_mode == "arms_neutral" && !arm_cols.is_empty();
        let (cmm_use, h_use, bias_use) = if arms_neutral {
            let mut masked = na::DMatrix::zeros(cmm.nrows(), cmm.ncols());
            for &vi in &arm_cols {
                masked.set_column(vi, &cmm.column(vi));
            }
            let h_arm = &masked * v_dvec;
            let zero = na::DVector::zeros(cmm.nrows());
            (masked, h_arm, zero)
        } else {
            (
                cmm.clone(),
                h_now.clone(),
                na::DVector::from_row_slice(dcmm_v.as_slice()),
            )
        };
        // Zero references (MOM_REF_MPC, the MPC's planned rate as the
        // reference, was not adopted and is removed); the arms-only DCM
        // ask below is the one rate this task is given.
        let h_ref = [0.0_f64; 3];
        let mut dh_ref = [0.0_f64; 3];
        if arms_neutral && mom_arm_dcm != 0.0 {
            dh_ref[0] = (-mom_arm_dcm * (xi.y - r.xi.y))
                .clamp(-mom_arm_dcm_max, mom_arm_dcm_max);
        }
        let dh_ref = dh_ref;
        Some(MomRef {
            cmm: cmm_use,
            bias: bias_use,
            h: h_use,
            h_ref,
            dh_ref,
            kp: kp_mom,
            axes: if coupled_active { [true; 3] } else { mom_axes },
        })
    } else {
        None
    };

    // ---- hip strategy: centroidal moment pivot (MOM_CMP) --------------
    let in_stair_zone = match &stairs {
        Some(st) if st.riser.abs() > 0.005 => {
            let x = steps.mid_xy().x;
            (x >= st.start - stair_crouch_lead && x <= st.landing_start() + stair_crouch_tail)
                || (st.down_n > 0
                    && x >= st.landing_end() - stair_crouch_lead
                    && x <= st.landing_end() + st.down_n as f64 * st.tread + stair_crouch_tail)
        }
        _ => false,
    };
    let cmp = if mom_cmp > 0.0
        && !in_stair_zone
        && zmp_deficit.norm() > mom_cmp_dead
    {
        let cmm = cent.cmm();
        let dcmm_v = cent.dcmm_v();
        let h_now = cmm * v_dvec;
        let mg = total_mass * G;
        let mut dh = [
            (-mom_cmp_sign * mom_cmp * mg * zmp_deficit.y).clamp(-mom_cmp_max, mom_cmp_max),
            (mom_cmp_sign * mom_cmp * mg * zmp_deficit.x).clamp(-mom_cmp_max, mom_cmp_max),
            0.0,
        ];
        for a in 0..2 {
            if mom_cmp_hmax > 0.0 && h_now[a].abs() > mom_cmp_hmax && dh[a] * h_now[a] > 0.0 {
                dh[a] = 0.0;
            }
            dh[a] -= mom_cmp_kh * h_now[a];
        }
        stats.n_mom_cmp_ticks += 1;
        stats.mom_cmp_peak = stats.mom_cmp_peak.max(dh[0].abs()).max(dh[1].abs());
        Some(CmpRef {
            bias: na::DVector::from_row_slice(dcmm_v.as_slice()),
            h_ref: [h_now[0], h_now[1], h_now[2]],
            h: h_now,
            dh,
        })
    } else {
        None
    };

    // ---- arm hold ----------------------------------------------------
    //
    // Same task as posture, restricted to the arm rows. Rows outside the
    // filter stay all-zero with a zero reference, i.e. `0 = 0`, which the
    // solver satisfies for free -- so this really is "posture, arms only,
    // at its own gain".
    // Jab. The arm-hold reference stops being the seed and becomes a
    // function of time for ONE arm at a time, out and back along a raised
    // cosine so the reversal has no corner.
    //
    // The punching arm needs its ELBOW driven too, and Sec.38 measured that
    // holding both rows of both arms costs four cells. So the elbow is
    // added to the held set only for the arm that is currently punching,
    // and only while it is: the other arm keeps whatever
    // `ARM_HOLD_JOINTS` asked for.
    let (jab_side, jab_a) = if jab && t >= jab_start {
        let j = k.command.jab_at(t);
        (Some(j.side), j.extension)
    } else {
        (None, 0.0)
    };
    let arm = if arm_hold {
        let jab_names: Option<(&str, &str)> = jab_side.map(|s| {
            if s == 0 {
                ("left_shoulder_pitch_joint", "left_elbow_joint")
            } else {
                ("right_shoulder_pitch_joint", "right_elbow_joint")
            }
        });
        let arms: Vec<(usize, usize)> = robot
            .actuated()
            .into_iter()
            .filter(|&(ji, _)| {
                let n = robot.joint_names[ji].as_str();
                let base = match arm_hold_joints.as_str() {
                    "elbow" => n.contains("elbow"),
                    "shoulder" => n.contains("shoulder"),
                    _ => n.contains("shoulder") || n.contains("elbow"),
                };
                base || jab_names.is_some_and(|(_, e)| n == e)
            })
            .collect();
        // Reference = the seed, with the punching arm's two rows swept
        // toward the extended pose.
        let mut q_ref = robot.q_seed.clone();
        if let Some((sh, el)) = jab_names {
            for (name, target) in [(sh, jab_pitch), (el, jab_elbow)] {
                if let Some(&ji) = robot.joint_map.get(name) {
                    q_ref[ji] += jab_a * (target - q_ref[ji]);
                }
            }
        }
        (!arms.is_empty()).then(|| ArmRef { arms, q_ref })
    } else {
        None
    };
    // ---- lowest: gravity comp + the load split the ZMP implies -------
    let tau_gravity = robot.gravity_torque(q);
    // Split the nominal load so that the net CoP lands where the DCM
    // controller asked for it. Projecting p_cmd onto the line between the
    // sole centres generalises the y-interpolation the standing version
    // used, and it is the same argument: an equal-share target makes the
    // regulariser walk the CoP outward instead of transferring load.
    let shares: Vec<f64> = match nc {
        2 => {
            let s0 = steps.xy(0);
            let s1 = steps.xy(1);
            let d = s0 - s1;
            let a = if d.norm_squared() > 1e-12 {
                ((p_cmd - s1).dot(&d) / d.norm_squared()).clamp(0.0, 1.0)
            } else {
                0.5
            };
            vec![a * ramp.load_share(stance[0]), (1.0 - a) * ramp.load_share(stance[1])]
        }
        _ => vec![1.0; nc],
    };

    // ---- P3: the foot that is NOT in contact ------------------------
    //
    // EVERY foot outside the contact set needs a Cartesian task. A foot
    // that is in neither is unconstrained except by the weak posture
    // level, and the QP will happily use the free leg as a reaction mass
    // for the CoM task: measured, a foot that had just landed was flung
    // to 80 mm and stayed there for a second while the robot toppled on
    // the other leg. "The schedule says double support" is not a task.
    let free_side: Option<usize> = (0..2).find(|s| !stance_sides.contains(s));
    let mut swing_tgt = na::Vector3::zeros();
    let mut swing_cmd_acc_z = 0.0_f64;
    let swing = match (free_side, ctl.swing_lift_off) {
        (None, _) => None,
        // The scheduled swing, following its planned arc.
        (Some(side), Some(lift_off))
            if support.swing() == Some(side) =>
        {
            let mi = robot.foot_mi[side];
            let jf =
                misarta::jacobian::compute_joint_jacobian_from_data(&robot.model, q, data, mi);
            let djf =
                misarta::jacobian::compute_joint_jacobian_time_derivative(&robot.model, q, v, mi);
            let djv = &djf * v_dvec;
            let pos = misarta::se3::translation(&data.oMi[mi]);
            let vel3 = &jf.rows(3, 3).into_owned() * v_dvec;
            let vel = na::Vector3::new(vel3[0], vel3[1], vel3[2]);
            // Land on the PLANNED footstep. Its z stays the lift-off z
            // rather than the plan's: the ground is flat, so they agree to
            // within a millimetre, and using the measured one keeps a foot
            // that started slightly high from being driven into the floor.
            // With stride 0 this is exactly "land where you took off".
            let plan_xy = plan.footsteps().at_slice(slice_idx).sole[side];
            // On stairs the surfaces differ in height; carry the CHANGE
            // in surface height on top of the measured lift-off z, so a
            // foot that started a few millimetres high keeps its offset
            // instead of being driven into the tread. Flat ground makes
            // dz zero and this is exactly the old `lift_off.z`.
            // The plan's own height change decides "descending" for the
            // descent-only absolute target (the measured-lift-off form
            // of dz is what it replaces, so it cannot be read first).
            let plan_dz_sign = stairs
                .as_ref()
                .map_or(0.0, |st| st.h(plan_xy.x) - st.h(lift_off.x));
            let dz = if (stair_absolute_td || (stair_absolute_td_desc && plan_dz_sign < -1e-9)) && stairs.is_some() {
                plan_xy.z - plan.footsteps().at_slice(slice_idx.saturating_sub(1)).sole[side].z
            } else {
                stairs
                    .as_ref()
                    .map_or(0.0, |st| st.h(plan_xy.x) - st.h(lift_off.x))
            };
            let touch_down = if (stair_absolute_td || (stair_absolute_td_desc && dz < -1e-9)) && dz.abs() > 1e-9 {
                // The target is a surveyed tread, not the measured height
                // of the previous (possibly rocking) foot plus a riser.
                let offset = foot_rot_at(side, plan.footsteps().at_slice(slice_idx).yaw[side])
                    * na::Vector3::new(prof.sole_centre_x, 0.0, -prof.sole_below_origin);
                plan_xy - offset
            } else {
                // `plan_xy` is the SOLE centre; the swing task tracks the
                // ANKLE. On a robot whose sole centre sits under the ankle
                // the two coincide and the difference is exactly zero --
                // kyo46rs has `sole_centre_x: 0.0`. G1 has 0.035, so the
                // swing is commanded 35 mm beyond where the footstep plan
                // wants the sole, and the landing check (which compares
                // sole to sole, `preview.rs`) sees that as a miss it can
                // never confirm: measured 2026-09-21, the sole lands 62 mm
                // long against a 12 mm tolerance and the stair transfer
                // therefore never arms.
                //
                // `STAIR_ABSOLUTE_TD` already subtracts this offset, but it
                // also switches z from "lift-off height plus riser" to the
                // surveyed tread, and that second change is what fells the
                // G1 approach (measured: x = 0.21 m). This knob is the
                // frame correction ALONE, in xy -- z is already an ankle
                // height here, being lift-off plus a surface delta.
                // OPT-IN: zero on kyo46rs either way, so the default keeps
                // every existing result bit-identical.
                let mut td = na::Vector3::new(plan_xy.x, plan_xy.y, lift_off.z + dz);
                if stair_td_sole_frame {
                    let off = foot_rot_at(side, plan.footsteps().at_slice(slice_idx).yaw[side])
                        * na::Vector3::new(prof.sole_centre_x, 0.0, 0.0);
                    td.x -= off.x;
                    td.y -= off.y;
                }
                td
            };
            // Flat ground keeps the shipped arc. A stair swing cannot:
            // its xy smoothstep advances while the sin^2 bump is still
            // low, and the TOE (sole half-length ahead of the centre)
            // reaches the riser FACE a few percent into the swing --
            // watched in render, the toe slams the wall at ~5 mm of
            // lift and the "climb" lands at the riser's base. So climb
            // in phases instead: rise to clearance FIRST, translate at
            // height, drop onto the tread last. Each phase edge uses
            // smoothstep, so velocity is zero at every joint and the
            // feedforward stays continuous.
            let (swing_fraction, swing_duration) = if com_replan {
                plan.landing()
                    .clock
                    .filter(|(i, _, _)| *i == slice_idx)
                    .map(|(_, start, duration)| {
                        (((t - start) / duration).clamp(0.0, 1.0), duration)
                    })
                    .unwrap_or((slice_frac, slice.duration()))
            } else {
                (slice_frac, slice.duration())
            };
            let (mut tgt, mut tgt_v, tgt_a) =
                if dz.abs() > step_dz_swing_min && stair_swing_ff {
                    crate::gait::stair_swing_sink(
                        lift_off,
                        touch_down,
                        lift_h,
                        swing_fraction,
                        swing_duration,
                        stair_swing_windows,
                        stair_td_sink,
                    )
                } else if dz.abs() > 1e-9 {
                    let f = swing_fraction;
                    let dur = swing_duration;
                    let ss = |u: f64| -> (f64, f64) {
                        let u = u.clamp(0.0, 1.0);
                        (u * u * (3.0 - 2.0 * u), 6.0 * u * (1.0 - u))
                    };
                    let z_max = lift_off.z.max(touch_down.z) + lift_h;
                    // rise [0,0.30] -> travel [0.25,0.70] -> drop [0.60,0.95].
                    // The drop OVERSHOOTS the surface by a few millimetres
                    // and ends before the slice does: a profile that arrives
                    // exactly at the tread with exactly zero velocity at
                    // frac = 1.0 leaves no margin for tracking lag, and the
                    // measured result was a foot still hovering when the
                    // schedule declared double support -- the anchor then
                    // slams it down (197 N spike) and the step after dies.
                    // How far INTO the tread the default stair swing aims.
                    // Historically 4 mm. Measured 2026-09-24: with 4 mm the
                    // foot rests on the tread carrying a median of 10 N for
                    // 1.4-2.8 s (the task still holds the leg's weight) and
                    // the landing gate (17 N) is met only by impact chatter;
                    // the feed-forward profile with 15 mm loaded it at
                    // 102 N within 0.07 s. Knob so the depth can be set on
                    // THIS profile; default reproduces the constant.
                    let z_land = touch_down.z - stair_land_sink;
                    let (su, dsu) = ss((f - 0.25) / 0.45);
                    let (ru, dru) = ss(f / 0.30);
                    let (du, ddu) = ss((f - 0.60) / 0.35);
                    let xy = lift_off.xy() + (touch_down.xy() - lift_off.xy()) * su;
                    let z = lift_off.z + (z_max - lift_off.z) * ru + (z_land - z_max) * du;
                    let vxy = (touch_down.xy() - lift_off.xy()) * (dsu / 0.45 / dur);
                    let vz = (z_max - lift_off.z) * (dru / 0.30 / dur)
                        + (z_land - z_max) * (ddu / 0.35 / dur);
                    (
                        na::Vector3::new(xy.x, xy.y, z),
                        na::Vector3::new(vxy.x, vxy.y, vz),
                        if tracking_cubic_ff && tracking_ff_riser_ok {
                            // Exact second derivative of the unchanged cubic profile.
                            let dd = |u: f64, span: f64| {
                                if u > 0.0 && u < 1.0 {
                                    (6.0 - 12.0 * u) / (span * dur).powi(2)
                                } else {
                                    0.0
                                }
                            };
                            let axy =
                                (touch_down.xy() - lift_off.xy()) * dd((f - 0.25) / 0.45, 0.45);
                            let az = (z_max - lift_off.z) * dd(f / 0.30, 0.30)
                                + (z_land - z_max) * dd((f - 0.60) / 0.35, 0.35);
                            na::Vector3::new(axy.x, axy.y, az)
                        } else {
                            na::Vector3::zeros()
                        },
                    )
                } else {
                    let mut pos0 = swing_position(
                        lift_off,
                        touch_down,
                        lift_h,
                        swing_fraction,
                    );
                    let mut vel0 = swing_velocity(
                        lift_off,
                        touch_down,
                        lift_h,
                        swing_fraction,
                        swing_duration,
                    );
                    // Level swings only: a swing that changes height
                    // (the shipped arc interpolates z linearly under its
                    // bump) would hold the apex and drop the whole change
                    // in the last 35% -- measured 2026-10-15, the 100 mm
                    // descent stomped at 1376 N and fell on every member.
                    // ... and only away from known stairs: inside the
                    // stair zone the validated stair machinery (landing
                    // clocks, descent lead) owns the swing; with the hold
                    // active there the 100 mm descent started its swing
                    // 20 cm ahead of the foot and fell on every member.
                    // ... and only for a swing on its NOMINAL clock: a
                    // push retimes the swing shorter, and the held apex
                    // then has to come down in the last 35% of a short
                    // swing. Measured 2026-10-16 on the 24 worst backward
                    // and lateral push members: hold unconditional 17 -> 11,
                    // start delay alone 17 -> 17 (SWING_Z_HOLD_MIN_DUR,
                    // fraction of the nominal single support; 0 = always).
                    if swing_z_hold
                        && !in_stair_zone
                        && (touch_down.z - lift_off.z).abs() < 0.02
                        && swing_duration >= swing_z_hold_min_dur * gp.t_ss
                    {
                        // SWING_Z_HOLD: the three-phase z of
                        // STAIR_SWING_PROFILE (rise [0,0.30] to the apex,
                        // hold, drop [0.60,0.95]) with the SHIPPED xy.
                        // Measured 2026-10-15 on the unmodelled-bump bench
                        // (joint-sum foot pitch): the swing foot hangs
                        // 20 deg toe-down, so with the sin^2 lift the toe
                        // is ~1 cm up when it reaches a 16 mm edge; the
                        // three-phase lift is at 5 cm before x moves and
                        // clears. Its compressed xy is what cost the push
                        // family (326 -> 235), so only z is taken.
                        let f = swing_fraction;
                        let dur = swing_duration.max(1e-6);
                        let ss = |u: f64| -> (f64, f64) {
                            let u = u.clamp(0.0, 1.0);
                            (u * u * (3.0 - 2.0 * u), 6.0 * u * (1.0 - u))
                        };
                        let z_max = lift_off.z.max(touch_down.z) + lift_h;
                        // SWING_Z_DROP="d0,d1": the drop window (default
                        // 0.60..0.95 as in the three-phase profile; 1.0
                        // lands on the slice end like the shipped arc).
                        let (d0, d1) = (swing_z_drop[0], swing_z_drop[1].max(swing_z_drop[0] + 0.05));
                        let (ru, dru) = ss(f / 0.30);
                        let (du, ddu) = ss((f - d0) / (d1 - d0));
                        pos0.z = lift_off.z + (z_max - lift_off.z) * ru + (touch_down.z - z_max) * du;
                        vel0.z = (z_max - lift_off.z) * (dru / 0.30 / dur)
                            + (touch_down.z - z_max) * (ddu / (d1 - d0) / dur);
                    }
                    if swing_xy_delay > 0.0 {
                        // (SWING_XY_END, ending the horizontal motion
                        // before 1, is removed.)
                        // SWING_XY_DELAY: hold the horizontal motion for
                        // the first `delay` of the swing while the shipped
                        // sin^2 lift runs, then smoothstep the xy over the
                        // rest. The unmodelled-bump bench measured the
                        // toe catching a 15 mm edge as soon as the swing
                        // starts (x advances before z has risen); the
                        // three-phase STAIR_SWING_PROFILE fixes that but
                        // costs the push family 326 -> 235 (2026-10-15).
                        // This keeps the lift and landing timing as
                        // shipped. 0 = off (bit-identical).
                        let d = swing_xy_delay;
                        let e = 1.0_f64.max(d + 0.05);
                        let u = ((swing_fraction - d) / (e - d)).clamp(0.0, 1.0);
                        let s_ = (3.0 - 2.0 * u) * u * u;
                        let ds = 6.0 * u * (1.0 - u) / ((e - d) * swing_duration.max(1e-6));
                        let dxy = touch_down.xy() - lift_off.xy();
                        pos0.x = lift_off.x + dxy.x * s_;
                        pos0.y = lift_off.y + dxy.y * s_;
                        vel0.x = dxy.x * ds;
                        vel0.y = dxy.y * ds;
                    }
                    (pos0, vel0, na::Vector3::zeros())
                };
            let replan_hold = push_replan_trig > 0.0
                && plan.claims().landing_owner(slice_idx) == Some(Writer::PushReplan)
                && ctl.push_replan_shift > push_replan_hold;
            // 0.05: the default of PUSH_REPLAN_HOLD_Z (removed; the
            // champion turns the hold off with PUSH_REPLAN_HOLD=10).
            let hold_z = if replan_hold { 0.05 } else { mpc_footstep_cross_hold_z };
            if hold_z > 0.0
                && (replan_hold || ctl.mpc_fs_cross_latched.is_some_and(|(ci, _)| ci == slice_idx))
            {
                let foot = misarta::se3::translation(&data.oMi[robot.foot_mi[side]]);
                if (touch_down.xy() - foot.xy()).norm() > mpc_footstep_cross_hold_xy {
                    let hold = touch_down.z + hold_z;
                    if tgt.z < hold {
                        tgt.z = hold;
                        tgt_v.z = tgt_v.z.max(0.0);
                    }
                }
            }
            if let Some(mode) = ctl.ref_mode
                && let Some(lib) = &ctl.ref_lib
                && ref_swing
            {
                // Follow the reference's sole trajectory for this foot
                // (the reference's right foot is the one that swings
                // first). Ankle target = sole centre minus the sole offset.
                let ref_side = if side == mode.stance { 0 } else { 1 };
                // The CoM dynamics ran on during the detection delay, but
                // the swing foot did not: compress the first swing's path
                // into the time that remains instead of skipping its start.
                let elapsed = t - mode.t0;
                let first = (lib.phases[0] - mode.tau0).max(0.05);
                let warp = if ref_side == 1 && elapsed < first { lib.phases[0] / first } else { 1.0 };
                let tau = if ref_side == 1 && elapsed < first { elapsed * warp } else { elapsed + mode.tau0 };
                let sole_w = mode.world(lib, lib.sole_at(ref_side, tau));
                let sole_w1 = mode.world(lib, lib.sole_at(ref_side, tau + dt * warp));
                let offset = foot_rot_at(side, plan.footsteps().at_slice(slice_idx).yaw[side])
                    * na::Vector3::new(prof.sole_centre_x, 0.0, -prof.sole_below_origin);
                tgt = sole_w - offset;
                tgt_v = (sole_w1 - sole_w) / dt;
            }
            if plan.landing().probe_depth > 0.0
                && plan.landing().clock.is_some_and(|(ci, _, _)| ci == slice_idx)
            {
                tgt.z -= plan.landing().probe_depth;
            }
            swing_tgt = tgt;
            swing_cmd_acc_z = tgt_a.z * tracking_ff_scale
                + kp_sw_z * (tgt.z - pos.z)
                + kd_sw * (tgt_v.z - vel.z);
            let om3 = &jf.rows(0, 3).into_owned() * v_dvec;
            // The task itself is built with the other levels (phase 11);
            // here its targets, axes and orientation target, in the
            // order the call evaluated them.
            // (STAIR_RISE_SOLE_POSE / _MIN_DZ, roll and pitch held
            // through a height-changing swing, were opt-in -- the
            // champion sets 0 -- and are removed.)
            let axes = if terrain_pitch
                && stairs.as_ref().is_some_and(|st| st.pitch_at(plan.footsteps().at_slice(slice_idx).sole[side].x) != 0.0)
            {
                // TERRAIN_PITCH_YAW=1: keep the walking default's yaw
                // and add pitch only, roll free (see SwingAxes::XyzYawPitch).
                if terrain_pitch_yaw { SwingAxes::XyzYawPitch } else { SwingAxes::XyzRollPitch }
            } else {
                match swing_rot_mode {
                    0 => SwingAxes::Xyz,
                    2 => SwingAxes::Pose,
                    3 => SwingAxes::XyzRollPitch,
                    _ => SwingAxes::XyzYaw,
                }
            };
            let rot = {
                let mut rot_tgt =
                    foot_rot_at(side, plan.footsteps().at_slice(slice_idx).yaw[side]);
                if terrain_pitch && let Some(st) = &stairs {
                    let th = st.pitch_at(plan.footsteps().at_slice(slice_idx).sole[side].x);
                    if th != 0.0 {
                        // Toe up by the slope angle: rotate about world y by -th.
                        rot_tgt = na::Rotation3::from_axis_angle(&na::Vector3::y_axis(), -th).matrix() * rot_tgt;
                    }
                }
                Some((
                    misarta::se3::rotation_matrix(&data.oMi[mi]),
                    rot_tgt,
                    na::Vector3::new(om3[0], om3[1], om3[2]),
                    kp_sw_rot,
                    // (LAND_KD_ROT_BOOST, a multiplier through a stair
                    // swing's drop phase, defaulted to 1 and is removed.)
                    kd_sw_rot,
                ))
            };
            let acc = tgt_a * tracking_ff_scale;
            Some(SwingRef { jf, djv, pos, vel, tgt, tgt_v, kp_xy: kp_sw_xy, kp_z: kp_sw_z, kd: kd_sw, axes, rot, acc: Some(acc) })
        }
        // Out of contact for any other reason -- it has not re-established
        // yet, or it bounced. Put it back on its planned footstep. The
        // target is the foot LINK origin, which sits `sole_below_origin`
        // above the sole.
        (Some(side), _) => {
            let mi = robot.foot_mi[side];
            let jf =
                misarta::jacobian::compute_joint_jacobian_from_data(&robot.model, q, data, mi);
            let djf =
                misarta::jacobian::compute_joint_jacobian_time_derivative(&robot.model, q, v, mi);
            let djv = &djf * v_dvec;
            let pos = misarta::se3::translation(&data.oMi[mi]);
            let vel3 = &jf.rows(3, 3).into_owned() * v_dvec;
            let vel = na::Vector3::new(vel3[0], vel3[1], vel3[2]);
            let tgt = steps.sole[side] + na::Vector3::new(0.0, 0.0, prof.sole_below_origin);
            swing_tgt = tgt;
            swing_cmd_acc_z = kp_sw_z * (tgt.z - pos.z) - kd_sw * vel.z;
            let om3 = &jf.rows(0, 3).into_owned() * v_dvec;
            let axes = match swing_rot_mode {
                0 => SwingAxes::Xyz,
                2 => SwingAxes::Pose,
                3 => SwingAxes::XyzRollPitch,
                _ => SwingAxes::XyzYaw,
            };
            let rot = Some((
                misarta::se3::rotation_matrix(&data.oMi[mi]),
                foot_rot_at(side, plan.footsteps().at_slice(slice_idx).yaw[side]),
                na::Vector3::new(om3[0], om3[1], om3[2]),
                kp_sw_rot,
                kd_sw_rot,
            ));
            Some(SwingRef { jf, djv, pos, vel, tgt, tgt_v: na::Vector3::zeros(), kp_xy: kp_sw_xy, kp_z: kp_sw_z, kd: kd_sw, axes, rot, acc: None })
        }
    };


    TaskRefs {
        j_trunk,
        djv_trunk,
        roll,
        pitch,
        rp_ref,
        yaw_ref,
        ori,
        coupled_active,
        mom,
        cmp,
        arm,
        tau_gravity,
        shares,
        free_side,
        swing,
        swing_tgt,
        swing_cmd_acc_z,
    }
}

/// Phase 11: the WBC request, and the solve by the stack (S12: the levels,
/// their order and weights, the solve, the gate and QP_DIAG_DIR are the
/// stack's, `crate::task_stack::TaskStack` by default). The walker hands it
/// phase 10's references and the contact request: the feet, the anchor
/// events of phases 3 and 4 (in order), the load share per side, rigid or
/// managed contact per side (STAIR_MANAGED_*), the slope's heel cut as a
/// moved CoP box per side (SLOPE_HEEL_CUT), and the MPC's wrenches when they
/// apply (MPC_SOLE_AUTHORITY, STAIR_WRENCH_BLEND). The per-side values are
/// the closures the contact level used to call, evaluated for both sides
/// (they are pure).
#[allow(clippy::too_many_arguments)]
pub(crate) fn phase_wbc(
    stack: &mut impl WbcStack,
    ctl: &CtlState,
    cfg: &Config,
    k: &WalkerConsts,
    obs: &TickObs,
    cent: &Centroidal,
    sup: &SupportOut,
    stance_sides: &[usize],
    stance: &[usize],
    nc: usize,
    ramp: &LoadRamp,
    anchor_events: &[AnchorEvent],
    a_com: &na::Vector3<f64>,
    refs: &TaskRefs,
    t: f64,
    out: &mut dyn LogSink,
) -> WbcSolution {
    let stair_wrench_mpc = cfg.mpc.enabled;
    let stair_managed_contact = cfg.managed.on;
    let stair_managed_tangent = cfg.managed.tangent;
    let stair_managed_rotation = cfg.managed.rotation;
    let stair_managed_az_max = cfg.managed.az_max;
    // (SWING_CAUTIOUS_ERR, STAIR_ANCHOR_FLAT and STAIR_ROLL_PATCH -- the
    // latter two for the toe-first descent -- are removed.)
    let slope_heel_cut = cfg.slope.heel_cut;
    let slope_heel_cut_max = cfg.slope.heel_cut_max;
    let mpc_sole_authority = cfg.mpc.sole_authority;
    let mpc_auth_fz_ratio = cfg.mpc.auth_fz_ratio;
    let stair_wrench_blend = cfg.mpc.blend;
    let sole_half_l = k.sole_half_l;
    let robot = &k.robot;
    let prof = &robot.prof;
    let stairs = &k.stairs;
    let contact_cfg = &k.contact_cfg;
    let wrench_mpc_apply = k.wrench_mpc_apply;
    let (support, steps) = (sup.support, &sup.steps);
    let lifting = ramp.lifting;
    let contact_mode = [0, 1].map(|side: usize| {
        if stair_wrench_mpc
            && stair_managed_contact
            // Rate trigger (see STAIR_MANAGED_RATE): the level
            // test misses the chatter, which swings fz 15<->78 N.
            && t < ctl.managed_until[side]
        {
            ContactMode::Managed {
                tangent_weight: stair_managed_tangent,
                rotation_weight: stair_managed_rotation,
                normal_weight: 1.0,
                normal_accel_max: stair_managed_az_max,
            }
        } else {
            ContactMode::Rigid
        }
    });
    let cop_patch = [0, 1].map(|_side: usize| {
        // SLOPE_HEEL_CUT [m/deg]: on a known UPSLOPE, during double
        // support, move the REAR foot's CoP box forward by
        // cut * slope_deg and shrink it by the same, so its heel
        // region is excluded. Measured 2026-10-17 (threshold table,
        // terrain bench doc Sec.10): the rear foot's CoP goes to the
        // heel in proportion to the slope (-12 mm/deg), reaches the
        // heel edge at 5 deg and the foot lifts off 3/4 of the way
        // through the double support. With the heel excluded the QP
        // has to realise the same net CoP by loading the front
        // foot, which is what the climb needs. 0 = off.
        if slope_heel_cut > 0.0
            && matches!(support, Support::Double)
            && let Some(st) = &stairs
            && st.n == 0
            && st.slope > 0.0
            && lifting == Some(_side)
            // ... and only while that foot stands ON the slope (the
            // cut applied on the flat approach at 7 deg emptied the
            // box and fell at 3.3 s, 2026-10-17).
            && {
                let x = steps.sole[_side].x;
                x >= st.start && x <= st.start + st.slope_len
            }
        {
            let deg = st.slope.atan().to_degrees();
            let d = (slope_heel_cut * deg).min(slope_heel_cut_max);
            return Some((
                [prof.sole_centre_x + d, 0.0, -prof.sole_below_origin],
                ((sole_half_l - d).max(0.005), contact_cfg.cop_half.1),
            ));
        }
        // (STAIR_ROLL_PATCH, a narrow strip at the edge of a pitched
        // sole, is removed.)
        None
    });
    // COP_LEASH: a loaded stance foot's box cut to the leash around its
    // measured CoP (the slope's heel cut, when it applies, wins).
    let cop_patch: [_; 2] = std::array::from_fn(|side| {
        cop_patch[side].or_else(|| {
            cop_leash_box(cfg, k, obs, sup, side, contact_cfg.cop_half).map(|(c, h)| {
                ([prof.sole_centre_x + c[0], c[1], -prof.sole_below_origin], h)
            })
        })
    });
    let req = WbcRequest {
        t,
        tick: obs.tick,
        robot,
        state: obs_state(obs),
        support,
        stance_sides,
        stance,
        anchor_events,
        load_share: [0, 1].map(|side| ramp.load_share(robot.foot_mi[side])),
        contact_mode,
        cop_patch,
        a_com: *a_com,
        trunk: TrunkRef {
            j: &refs.j_trunk,
            djv: &refs.djv_trunk,
            rp_ref: refs.rp_ref,
            yaw: refs.yaw_ref.map(|_| refs.ori[2]),
        },
        coupled: refs.coupled_active,
        momentum: refs.mom.as_ref(),
        mom_cmp: refs.cmp.as_ref().map(|r| MomCmp { cmm: cent.cmm(), r }),
        arm_hold: refs.arm.as_ref(),
        joint_pos: obs.joint_pos,
        tau_gravity: &refs.tau_gravity,
        shares: &refs.shares,
        // The MPC's wrenches replace (MPC_SOLE_AUTHORITY, double support
        // only) or blend into the load split's nominal forces.
        mpc: ctl
            .wrench_mpc_command
            .as_ref()
            .filter(|_| wrench_mpc_apply && nc > 0)
            .map(|wrench| MpcNominal {
                wrench,
                authority: mpc_sole_authority
                    && nc == 2
                    && !ctl.mpc_auth_veto
                    && (mpc_auth_fz_ratio <= 0.0
                        || wrench[0][5] + wrench[1][5] <= mpc_auth_fz_ratio * k.total_mass * G),
                blend: stair_wrench_blend,
            }),
        swing: refs.swing.as_ref(),
        contact_fz: obs.contact_fz,
    };
    stack.solve(&req, out)
}

/// What phase 12 measured on the solution: the swing level's residuals,
/// the legs' reach, the swing foot's achieved acceleration and position
/// error, and the QP's per-foot CoP and force, CoP use and net CoP.
pub(crate) struct SolveDiag {
    pub swing_task_residual_rms: f64,
    pub swing_res_z: f64,
    pub swing_res_rows: f64,
    pub reach_used: [f64; 2],
    pub hip_z: [f64; 2],
    pub stride_max_now: f64,
    pub swing_achieved_acc_z: f64,
    pub swing_pos_err_xy: f64,
    pub swing_pos_err_z: f64,
    pub cop_qp: [[f64; 3]; 2],
    pub f_qp_w: [[f64; 3]; 2],
    pub cop_use: f64,
    pub diag_wbc_net_cop: [f64; 2],
}

/// Phase 12: the diagnostics of the solve (read by the record only).
pub(crate) fn phase_solve_diag(
    cfg: &Config,
    k: &WalkerConsts,
    obs: &TickObs,
    stance: &[usize],
    refs: &TaskRefs,
    w: &WbcSolution,
) -> SolveDiag {
    let cop_frac = cfg.dcm.cop_frac;
    let (total_mass, sole_half_l, sole_half_w) = (k.total_mass, k.sole_half_l, k.sole_half_w);
    let (hip_mi, leg_reach) = (k.hip_mi, &k.leg_reach);
    let robot = &k.robot;
    let prof = &robot.prof;
    let data = &obs_state(obs).data;
    let extracted = &w.extracted;
    let (free_side, swing_tgt, swing_cmd_acc_z) = (refs.free_side, refs.swing_tgt, refs.swing_cmd_acc_z);
    // The swing level's residual `A x - b` (the stack computed it on the
    // unweighted task), one value per row.
    let swing_task_residual_rms = w.swing_residual.as_ref().map_or(0.0, |r| {
        if r.len() == 0 {
            0.0
        } else {
            r.norm() / (r.len() as f64).sqrt()
        }
    });
    // Which ROW of the swing task is being refused. `z` is the last row
    // in every `SwingAxes` layout, and after the ascent the step that
    // never lands is exactly the one whose z target is descending.
    let (swing_res_z, swing_res_rows) = w.swing_residual.as_ref().map_or((0.0, 0.0), |r| {
        if r.len() == 0 {
            (0.0, 0.0)
        } else {
            (r[r.len() - 1], r.len() as f64)
        }
    });
    // Leg reach this tick. `steps.sole[side]` is the PLANNED sole, so the
    // measured sole comes from FK like everywhere else in this loop.
    let mut reach_used = [0.0_f64; 2];
    let mut hip_z = [0.0_f64; 2];
    let mut stride_max_now = 0.0_f64;
    if let Some(mis) = hip_mi
        && leg_reach.max > 0.0
    {
        let mut soles = [na::Vector3::zeros(); 2];
        for side in 0..2 {
            let hip = misarta::se3::translation(&data.oMi[mis[side]]);
            let mut sole = misarta::se3::translation(&data.oMi[robot.foot_mi[side]]);
            sole.z -= prof.sole_below_origin;
            reach_used[side] = leg_reach.used(&hip, &sole);
            hip_z[side] = hip.z;
            soles[side] = sole;
        }
        // The stride the two legs allow at this instant, with the hip
        // midway (see reach::max_stride).
        let hip_mid = 0.5 * (hip_z[0] + hip_z[1]);
        stride_max_now = leg_reach.max_stride(hip_mid, soles[0].z, soles[1].z);
    }
    // `swing_res_z = achieved - commanded`: misa-wbc Cartesian residual.
    let swing_achieved_acc_z = swing_cmd_acc_z + swing_res_z;
    let (swing_pos_err_xy, swing_pos_err_z) = free_side.map_or((0.0, 0.0), |side| {
        let pos = misarta::se3::translation(&data.oMi[robot.foot_mi[side]]);
        (
            (pos.xy() - swing_tgt.xy()).norm(),
            (pos.z - swing_tgt.z).abs(),
        )
    });

    let mut cop_qp = [[0.0_f64; 3]; 2];
    let mut wbc_net_cop_acc = [f64::NAN; 3];
    let mut f_qp_w = [[0.0_f64; 3]; 2];
    let mut cop_use = 0.0_f64;
    for (slot, mi) in stance.iter().copied().enumerate() {
        let side = robot.side_of(mi);
        if let Some((cx, cy, fz)) = cop_from_sole_wrench(&w.sole_wrench[slot]) {
            if fz > 1e-6 {
                cop_qp[side] = [cx, cy, fz];
                // Same net pressure point as `diag_mpc_net_cop`, for the
                // forces the WBC asked for, so the two are comparable
                // against the commanded ZMP.
                {
                    // `cop_from_sole_wrench` gives the point in the SOLE
                    // CENTRE frame, and `oMi` is the ankle. On G1 the sole
                    // centre sits `sole_centre_x` = 35 mm ahead of the
                    // ankle, so adding the two directly biased this column
                    // by -35 mm in x -- which read as "the WBC puts its
                    // pressure 32 mm behind the command" and was reported
                    // that way on 2026-09-25 (mpc_in_the_loop.md Sec.29).
                    // In single support the same column read -37.5 mm,
                    // which a CoP confined to one sole cannot do; that is
                    // how the bias was caught (2026-09-26). Same
                    // construction as `Footsteps::from_fk`.
                    let pose = &data.oMi[mi];
                    let rot = misarta::se3::rotation_matrix(pose);
                    let origin = misarta::se3::translation(pose);
                    let point = origin
                        + rot
                            * na::Vector3::new(
                                prof.sole_centre_x + cx,
                                cy,
                                -prof.sole_below_origin,
                            );
                    if !wbc_net_cop_acc[2].is_finite() {
                        wbc_net_cop_acc = [0.0; 3];
                    }
                    wbc_net_cop_acc[0] += point.x * fz;
                    wbc_net_cop_acc[1] += point.y * fz;
                    wbc_net_cop_acc[2] += fz;
                }
                // Only count a foot that is actually carrying something.
                // A foot at 1 N with its CoP in the box corner is a
                // 0.07 N*m wrench, and reporting that as a saturated
                // constraint hides the one that matters.
                if fz > 0.1 * total_mass * G {
                    cop_use = cop_use.max(
                        (cx.abs() / (sole_half_l * cop_frac))
                            .max(cy.abs() / (sole_half_w * cop_frac)),
                    );
                }
            }
        }
        for k in 0..3 {
            f_qp_w[side][k] = extracted.forces[6 * slot + 3 + k];
        }
    }
    // The same net pressure point as `diag_mpc_net_cop`, for the forces
    // the WBC actually asked the robot for.
    let diag_wbc_net_cop = if wbc_net_cop_acc[2] > 1.0 {
        [
            wbc_net_cop_acc[0] / wbc_net_cop_acc[2],
            wbc_net_cop_acc[1] / wbc_net_cop_acc[2],
        ]
    } else {
        [f64::NAN; 2]
    };
    SolveDiag {
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
    }
}

/// Phase 13, the end of the tick (after the harness has fired its push): a
/// recovery reference whose time is up hands the feet back to the gait,
/// re-centring every later footfall on where the feet are now.
pub(crate) fn phase_end_tick(ctl: &mut CtlState, plan: &mut Plan, cfg: &Config, t: f64, out: &mut dyn LogSink) {
    let ref_cooldown = cfg.recovery.cooldown;
    // CONTACT_EVENT (read once: the landing clock's `event_mode`, which a
    // reference turns on while it is engaged and its end restores to this).
    let event_mode_cfg = cfg.landing.event.on;
    let ref_reset_stance = cfg.recovery.reset_stance;
    // The re-centring after a reference skipped the swing already under way
    // when the reference ended (the gait's first step then went to the
    // pre-push plan: 0.38 m ahead, 0.28 m wide). With this on, that swing is
    // re-centred too, at the feet's midpoint (REF_RESET_LEAD, a lead ahead
    // of it, is removed).
    let ref_reset_incl_current = cfg.recovery.reset_incl_current;
    if let Some(mode) = ctl.ref_mode
        && let Some(lib) = &ctl.ref_lib
        && t >= mode.t0 + lib.total() - mode.tau0
    {
        ctl.ref_mode = None;
        ctl.ref_cooldown_until = t + ref_cooldown;
        let mut ed = plan.edit(Writer::RecoveryEnd);
        ed.release_whole();
        ed.landing_mut().event_mode = event_mode_cfg;
        // The reference leaves the feet where the recovery put them
        // (in tandem, narrow). Every later footfall is re-centred on the
        // feet as they are now, at the nominal stance width, so the gait
        // that resumes is the nominal one and not a stride-0 echo of the
        // recovery (measured: the echo fell 1.7 s after the reference).
        if ref_reset_stance {
            let i = ed.gait().index_at(t);
            let cur = *ed.footsteps().at_slice(i);
            let mid = 0.5 * (cur.sole[0].xy() + cur.sole[1].xy());
            let half_w = 0.5 * (ed.footsteps().initial.sole[0].y - ed.footsteps().initial.sole[1].y).abs();
            let j0 = if ref_reset_incl_current { i } else { i + 1 };
            for side in 0..2 {
                let want = na::Vector2::new(mid.x, mid.y + if side == 0 { half_w } else { -half_w });
                if let Some(j) = (j0..ed.gait().slices.len()).find(|&j| ed.gait().slices[j].support.swing() == Some(side)) {
                    if j == i {
                        outln!(out, "  [ref] t={t:.3} re-centre the swing under way: foot {side} -> ({:.3},{:.3})", want.x, want.y);
                    }
                    let d = want - ed.footsteps().at_slice(j).sole[side].xy();
                    ed.shift_from(j, side, d);
                }
            }
            // Rebuilt even when neither foot had a later swing to move.
            ed.touch();
        }
        ed.commit();
        outln!(out, "  [ref] t={t:.3} reference over, back to the gait");
    }
}
