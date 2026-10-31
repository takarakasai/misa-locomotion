//! Convex receding-horizon centroidal MPC for two sole contact wrenches.
//!
//! The prediction freezes each contact moment arm for one QP solve, which
//! makes the SRBD dynamics linear. Wrenches are expressed at the sole centre
//! in the sole frame as `[mx, my, mz, fx, fy, fz]`.

use misa_wbc::qp::{QpConfig, QpSolution, QpStatus, QpWorkspace, solve_qp, solve_qp_warm};
use nalgebra as na;

pub type Wrench = na::SVector<f64, 6>;

#[derive(Clone, Debug)]
pub struct Contact {
    pub position: na::Vector3<f64>,
    pub rotation: na::Matrix3<f64>,
    /// Available rectangular CoP region for this contact. This may be a
    /// narrow toe/heel patch while the sole is rolling onto a tread.
    pub cop_half: (f64, f64),
}

/// A footstep the MPC may move: the landing of `foot`, active from node
/// `first_node` on, as a decision variable `dp` in R^2 (world x, y).
///
/// The contact position enters the problem only through the angular-momentum
/// rows and the net-CoP rows, both bilinear in (p, f); the friction, CoP-box,
/// torsion and normal-force constraints live in the sole frame and do not
/// see it. So one SQP step is enough to make the landing a variable: the
/// cross term is linearised about a prior world force `F_k` per node,
///     (p + dp - c) x F  =  (p - c) x F  +  dp x F_k  =  ...  - skew(F_k) dp,
/// with `F_k = R [0, 0, fz_ref_k]`, the planned load -- already available per
/// node, so no solution history is needed. `weight * |dp|^2` is the cost of
/// moving the foot off the plan; `max` boxes it (reach comes later).
#[derive(Clone, Copy, Debug)]
pub struct Landing {
    pub foot: usize,
    pub first_node: usize,
    pub weight: f64,
    pub max: [f64; 2],
    /// Offset of this landing ALREADY applied to the plan the rest of the
    /// input was built from (world x, y). The variable `dp` is then an
    /// increment on top of it, and the landing cost and box act on the
    /// total `prior + dp`. Zero = the shipped semantics (cost and box on
    /// `dp` itself), bit-identical.
    pub prior: [f64; 2],
    /// Bound on the NEGATIVE side of each axis (|min| on `prior + dp` below
    /// zero). Equal to `max` = the shipped symmetric box, bit-identical.
    /// A stair landing uses a smaller backward bound: the MPC has no terrain
    /// model and was measured pulling a first-tread landing 20 mm toward the
    /// nosing (design note, 09-27 10:10).
    pub min: [f64; 2],
    /// Earliest node the landing may be brought forward to (step timing as
    /// a decision, `Config::landing_timing`). Equal to `first_node` = the
    /// nominal timing only, bit-identical.
    pub min_first_node: usize,
    /// Terrain window for the landing's total x (`position.x + prior.x +
    /// dp.x`): `(lo, hi, weight)`. Outside the window the lexicographic
    /// landing pays `weight * distance^2`, a soft pull back onto the surface
    /// the sole can stand on (a hard snap to the window edge was measured
    /// harmful, design note 09-25: it commands a landing AT a boundary).
    /// `None` = no terrain term, bit-identical.
    pub terrain_x: Option<(f64, f64, f64)>,
    /// Planned landing position (world x, y), where the pressure goes from
    /// an EARLIER touchdown on: before the nominal first node the plan's own
    /// net CoP is still the stance foot's, so an early landing's capture is
    /// modelled as `position + prior + dp` on those nodes.
    pub position: [f64; 2],
}

#[derive(Clone, Debug)]
pub struct Input {
    pub com: na::Vector3<f64>,
    pub velocity: na::Vector3<f64>,
    pub angular_momentum: na::Vector3<f64>,
    pub target_com: na::Vector3<f64>,
    pub target_velocity: na::Vector3<f64>,
    pub target_acceleration: na::Vector3<f64>,
    pub contacts: [Contact; 2],
    /// Fixed contact availability for every prediction stage. An unavailable
    /// contact has all six wrench components constrained to zero. Keeping the
    /// schedule in the optimisation input prevents the SRBD model from using
    /// a reaction wrench that the plant cannot realise.
    pub contact_schedule: Vec<[bool; 2]>,
    /// Continuous contact authority per prediction stage. Zero removes the
    /// wrench and one permits `Config::max_fz`; intermediate values cap the
    /// full friction/CoP cone through its normal-force bound.
    pub activation_schedule: Vec<[f64; 2]>,
    /// Per-stage desired vertical force in each sole frame.
    pub reference_fz_schedule: Vec<[f64; 2]>,
    /// Where the NET pressure of both feet should sit, per node, in world
    /// [m]. Only x and y; the height is taken from each contact.
    ///
    /// The first version of this asked for a CoP on each sole separately and
    /// made things WORSE (the left/right force gap grew 43 -> 63 N, measured
    /// 2026-09-25): the walk's plan is a NET pressure point, and in double
    /// support each sole's own CoP need not sit anywhere near it. What the
    /// gait actually constrains is the net, which is the force-weighted mean
    /// of the two -- and that is still linear in the wrench:
    ///
    /// ```text
    /// sum_i [ m_i + (p_i - c) x f_i ]_z-free  =>  two rows
    /// ```
    ///
    /// written out, with `p_i` the contact origin and `c` the target:
    ///     sum_i ( m_x,i + (p_iy - c_y) f_z,i - (p_iz - c_z) f_y,i ) = 0
    ///     sum_i ( m_y,i - (p_ix - c_x) f_z,i + (p_iz - c_z) f_x,i ) = 0
    pub reference_net_cop_schedule: Option<Vec<na::Vector2<f64>>>,
    /// Superseded, kept so the field名 does not silently change meaning.
    /// Where the pressure should sit on each sole, per node, in that sole's
    /// own frame [m]. `None` (the default) adds nothing, which is the
    /// behaviour every recipe measured before 2026-09-25 was tuned under.
    ///
    /// Without this the MPC has no opinion about the CoP, so in double
    /// support -- where the split of a given total wrench is underdetermined
    /// -- it picks whatever its wrench and smoothness terms like. Measured
    /// 2026-09-25: the totals it asks for are right (median 1.008 body
    /// weights) but the left/right split differs from the walk's by 40-55 N,
    /// and since that split IS the CoP, applying the MPC's answer moved the
    /// pressure a few centimetres and no blend fraction, replacement or
    /// horizon change could walk with it. The gait's intent lived only in the
    /// DCM plan; this is how the MPC is told about it.
    ///
    /// The CoP is a ratio of moment to normal force, but the condition
    /// "pressure at `c`" is linear in the wrench:
    ///     m_y + c_x f_z = 0,   m_x - c_y f_z = 0
    /// so it costs the problem nothing in structure.
    pub reference_cop_schedule: Option<Vec<[na::Vector2<f64>; 2]>>,
    /// Where the divergent component of motion should be, per node, in
    /// world [m]; horizontal only.
    ///
    /// The MPC's own feedback law is whatever its cost implies. With the CoM
    /// position and velocity penalised SEPARATELY that law is not a DCM law
    /// and cannot be made into one by choosing the two weights, because a
    /// diagonal cost in (c, c_dot) has no cross term and the DCM cost
    ///
    /// ```text
    /// |c + c_dot/omega - xi_ref|^2
    /// ```
    ///
    /// is exactly a cross term. Meanwhile the WBC's commanded pressure point
    /// IS a DCM law. So the two ran different feedback on the same plant, and
    /// measured 2026-09-25 their double-support force split would not come
    /// closer than 33 N however the other weights were moved -- the floor was
    /// the mismatch itself, not a tuning failure.
    ///
    /// Only x and y: the vertical has no divergent LIPM mode.
    pub reference_dcm_schedule: Option<Vec<na::Vector2<f64>>>,
    /// Per-stage lower bound on vertical force. This lets touchdown load be
    /// admitted continuously instead of imposing a full support load on the
    /// first sample that reports contact.
    pub min_fz_schedule: Vec<[f64; 2]>,
    pub previous: [Wrench; 2],
    /// `None` (the default) is the shipped problem with fixed contacts.
    pub landing: Option<Landing>,
    /// Predicted CoM at the end of each node from the PREVIOUS solve
    /// (`Solution::trajectory[k][0]`), used with `Config::arm_about_prior`
    /// as the linearisation point of each node's moment arm. `None` keeps
    /// the shipped arm about `target_com`.
    pub prior_com: Option<Vec<na::Vector3<f64>>>,
    /// Measured DCM now (world x, y), for the landing-capture rows. `None`
    /// disables them even when `Config::landing_capture_weight` is set.
    pub dcm_now: Option<na::Vector2<f64>>,
    /// How the plan's OWN reference DCM at node k moves when the landing
    /// moves: d xi_ref_k / d dp, one 2x2 per node (world x, y). The reference
    /// is a backward recursion from the future footsteps, so shifting the
    /// next landing by dp moves the reference of the current step by about
    /// 0.65 dp (measured against the real recursion, see contact_plan
    /// CP_J_AUTO). Without it the capture rows solve
    ///     a + b dp = xi_ref
    /// for a target that then moves under them once the shift is applied:
    /// the next solve sees a residual -J dp and answers -J dp / b, a loop
    /// with gain J/|b| that exceeds 1 whenever the landing is within ~3
    /// nodes of the horizon end (b ~ -(1+w dt)^m + 1 is small there). That is
    /// the 20 ms sign-alternating landing measured 2026-09-27 (design note).
    /// With it the rows solve a + (b - J) dp = xi_ref, and |b - J| >= J
    /// bounds the shift by the error over J -- the capture-point step
    /// adjustment. `None` = the shipped rows (J = 0), bit-identical.
    pub reference_dcm_sensitivity: Option<Vec<na::Matrix2<f64>>>,
    /// Upper bound on the predicted CoM height at the END of each node,
    /// world z [m] -- what the legs can reach from the feet the schedule has
    /// down at that node (`Config::reach_weight`). The bound is soft: node k
    /// gets a slack `s_k >= 0` with `z_k - s_k <= cap_k` and `reach_weight *
    /// s_k^2` in the cost, so a cap the dynamics cannot meet in time does
    /// not make the problem infeasible. The leg's reach is a circle about
    /// the sole, which is nonlinear in the horizontal CoM; the caller
    /// linearises it at the previous solve's predicted trajectory (one SQP
    /// step per replan, as the moment arm is). `None` = no rows, bit-identical.
    pub com_z_max: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub horizon: usize,
    pub dt: f64,
    pub mass: f64,
    pub gravity: f64,
    pub friction_mu: f64,
    pub cop_half: (f64, f64),
    pub torsion_mu: f64,
    pub max_fz: f64,
    pub position_weight: f64,
    pub velocity_weight: f64,
    pub momentum_weight: f64,
    pub acceleration_weight: f64,
    pub wrench_weight: f64,
    /// Weight on the CoP residual (see `Input::reference_cop_schedule`).
    /// 0 = off = the behaviour before 2026-09-25.
    pub cop_weight: f64,
    /// Weight on the NET CoP residual (see `Input::reference_net_cop_schedule`).
    pub net_cop_weight: f64,
    /// Make the net CoP an EQUALITY instead of a cost.
    ///
    /// As a cost it is only a wish: measured 2026-09-25, raising the weight
    /// saturated at a left/right force gap of 57 N against the walk's own
    /// split, because the MPC still trades the CoP off against its wrench and
    /// smoothness terms. The CoP has a centimetre-scale tolerance, so a wish
    /// is not enough -- and two things wishing for different CoPs is exactly
    /// the authority conflict that stopped the MPC being usable at all. As an
    /// equality the MPC instead picks the best force split AMONG those that
    /// put the pressure where the walk asked.
    ///
    /// Only where both feet are scheduled down: in single support the net CoP
    /// is that one sole's own CoP, already bounded by its box, and an equality
    /// there can only make the problem infeasible.
    pub net_cop_equality: bool,
    /// Weight on the DCM residual (see `Input::reference_dcm_schedule`).
    /// 0 = off = the behaviour before 2026-09-25.
    /// Take each node's moment arm about the previous solve's predicted CoM
    /// for that node (`Input::prior_com`) instead of about `target_com`.
    /// This is one SQP step per solve: with the arm frozen about the target,
    /// the contact position enters only as a moment about a point the CoM
    /// is not at, and (measured 2026-09-26, Sec.50) a landing variable then
    /// only trades angular momentum. Off = shipped behaviour.
    /// Weight on the LANDING-CAPTURE rows: the DCM propagated through the
    /// horizon with the planned net CoP, where the CoP after the landing
    /// moves rigidly with the landing offset `dp`,
    ///     xi_{k+1} = xi_k + dt*omega*(xi_k - p_k),   p_k = p_bar_k (+ dp after touchdown)
    /// tracked to `Input::reference_dcm_schedule`. This is what a footstep is
    /// FOR in a LIPM -- the next pressure point that recaptures the DCM --
    /// and it is what the moment-arm coupling alone cannot express: with only
    /// that, the landing variable sat at its box regardless of weights
    /// (design note Sec.8). Needs `Input::dcm_now`, `Input::landing`,
    /// `reference_net_cop_schedule`, `reference_dcm_schedule`, `omega`. 0 = off.
    pub landing_capture_weight: f64,
    /// Horizon of the landing-capture rows in nodes, when longer than the
    /// wrench horizon (the reference schedules must be that long; past
    /// their end the last entry repeats). 0 = the wrench horizon. Measured
    /// 2026-10-02 (design note Sec.18): with 8 nodes the required backward
    /// step shrank from -200 to -107 mm as touchdown approached, because
    /// within 400 ms a small step already brings the horizon's DCM back to
    /// the plan; the long-run capture condition needs the next step inside.
    pub landing_capture_horizon: usize,
    /// Decide the landing LEXICOGRAPHICALLY: first the offset that
    /// minimises the capture residual inside the box (a 2-variable box QP,
    /// solved exactly), then the wrench MPC with the landing fixed there.
    /// This is the limit of `landing_capture_weight -> inf`: capture always
    /// wins over the moment coupling and the landing cost, with no weight to
    /// tune (measured 2026-09-27 on a 15/8 cm box: 1e4 falls at the first
    /// step, 1e5 breaks stairs, 1e6 works, 1e7 kills everything). The landing
    /// cost only breaks ties. Needs the same inputs as the capture rows.
    /// false = shipped (penalty rows).
    pub landing_lexicographic: bool,
    /// Step TIMING as a decision of the lexicographic landing stage: the
    /// touchdown node is enumerated from `Landing::first_node` down to
    /// `Landing::min_first_node`, each with its own capture box-QP, and the
    /// (node, dp) of least capture residual plus `landing_time_weight` per
    /// node of shortening wins (BD decides placement and timing together;
    /// v5.2 took the timing from the contact_plan heuristic). Off = nominal
    /// timing, bit-identical.
    pub landing_timing: bool,
    pub landing_time_weight: f64,
    /// Judge the timing on the SAGITTAL capture residual only. The lateral
    /// DCM needs the whole swing to come back; landing early on a lateral
    /// error puts the foot down before it has, which is why the contact_plan
    /// heuristic runs with CP_T_FROM_X (2026-10-02 01:42: with both axes the
    /// stage shortened an undisturbed step by 100 ms on a 43 mm lateral
    /// residual and fell). true = x only (default), false = both axes.
    pub landing_time_from_x: bool,
    /// Shorten only when it buys at least this much capture residual (m^2
    /// summed over the nominal landing's nodes). The rolled plan and the
    /// LIPM propagation disagree by ~25 mm RMS on an undisturbed step, and
    /// without a dead band that bias alone made the stage shorten a quiet
    /// step by 100 ms (2026-10-02 01:45).
    pub landing_time_gain_min: f64,
    /// Fix each foot's vertical force to `Input::reference_fz_schedule` by
    /// an EQUALITY at every node where both feet are down. Measured
    /// 2026-10-01 (design note Sec.14): with the CoP-capture rows the MPC
    /// satisfies a heel-ward net CoP by loading the front foot and leaving
    /// the sole moments near zero; the WBC's regulariser then relaxes the
    /// ankles exactly when a backward push needs them. With the split fixed
    /// the net CoP can only move through the sole moments, which is what
    /// the WBC can realise. false = shipped.
    pub fz_split_equality: bool,
    /// Decide the net CoP FIRST, as a LIPM plan: per node, the pressure point
    /// inside the box the scheduled load split can reach that recaptures the
    /// DCM (`lexicographic_cop`), then hold the wrench MPC to it by equality
    /// (net CoP at every contact node, and the TOTAL vertical force at every
    /// node). Replaces the penalty capture rows, which the MPC satisfied with
    /// excess vertical force on the front foot (2026-10-01, design note
    /// Sec.14): with the total held, the net CoP can only be moved by the
    /// split and the sole moments, both of which the WBC can realise. (A
    /// per-foot split equality was tried first: it forbids the load transfer
    /// a double support exists for and is infeasible from 0.7 s.) The CoP
    /// plan's weights are `capture_cop_wp` (deviation from the plan's net
    /// CoP) and `capture_cop_ws` (node-to-node change), both against a DCM
    /// residual of weight 1 (x4 at the terminal node); `capture_cop_margin`
    /// shrinks every box. false = shipped.
    pub capture_cop_lexicographic: bool,
    pub capture_cop_wp: f64,
    pub capture_cop_ws: f64,
    pub capture_cop_margin: f64,
    /// Diagnostics for the lexicographic CoP: hold the net CoP by equality
    /// (false = weighted residual, `net_cop_weight`) and hold the TOTAL
    /// vertical force by equality (false = free). Both true = the design.
    pub capture_cop_net_eq: bool,
    pub capture_cop_total_eq: bool,
    /// Build the reachable box as if every foot's minimum load were zero
    /// (the touchdown-admission ramp in `min_fz_schedule` pulls the box's
    /// rear edge forward through a double support, and the DCM law the WBC
    /// runs ignores that ramp). The wrench stage keeps the real minimum; an
    /// unreachable plan then falls back to the soft rows. false = exact box.
    pub capture_cop_box_ignores_min: bool,
    /// Keep the plan off the box's vertices: when the reachable range is
    /// computed, each foot is assumed to need its minimum load PLUS this
    /// fraction of the scheduled total. At a vertex the wrench stage must
    /// hold one foot at its minimum and both sole moments at their limit at
    /// once, and the interior-point solve of that equality fails (2026-10-02
    /// 01:11: the plan sat at 0.872 = the vertex, the strict solve failed,
    /// the soft retry answered 0.898 and the backward push was lost).
    pub capture_cop_force_margin: f64,
    /// Weight on the DCM-capture rows built from the MPC's OWN net CoP:
    /// the DCM is propagated through the horizon with
    ///     p_k = p_bar_k + delta_p_k(u),  delta_p_k = (-M_y, M_x) / Fz_ref_k
    /// where M is the net moment about p_bar_k that the net-CoP rows already
    /// write (linear in the wrenches and, through the moment coupling, in the
    /// landing), linearised at the planned vertical load. Every node is a
    /// row, landing or not: the MPC then uses the pressure point inside the
    /// support (ankle strategy) and the landing for the same capture goal
    /// instead of tracking a CoP schedule and a landing rule separately.
    /// Needs the capture inputs (`dcm_now`, both reference schedules) and
    /// `omega`. 0 = off = shipped.
    pub capture_cop_weight: f64,
    /// Let the landing offset enter the momentum rows and the net-CoP rows
    /// as the extra moment `dp x F_k` (the 2026-09-26 coupling). Measured
    /// 2026-09-27 (V5A i8_d0/m01): with it the landing jitters +-30-47 mm
    /// during undisturbed steps while the DCM error stays within 5 mm -- the
    /// optimiser trades the landing against angular momentum, not against
    /// capture -- and the swing target lurches late in swing. `false` keeps
    /// the landing in the capture rows and its own cost only.
    /// true = shipped behaviour.
    pub landing_moment_coupling: bool,
    pub arm_about_prior: bool,
    /// Weight on the node-to-node CHANGE of angular momentum, (L_{k+1} -
    /// L_k)/dt. The shipped cost tracks L itself to zero, which a walk
    /// cannot satisfy and which, weighted hard, broke the baseline
    /// (Sec.50). 0 = off.
    pub ldot_weight: f64,
    pub dcm_weight: f64,
    /// LIPM frequency sqrt(g/z) [1/s] for the DCM term. The term is skipped
    /// unless this and `dcm_weight` are both positive.
    pub omega: f64,
    pub smooth_weight: f64,
    /// Independent MPC backend configuration; defaults preserve ActiveSet.
    pub qp: QpConfig,
    /// Opt-in common formal-status/KKT gate, including the default backend.
    /// Closed-loop comparisons enable this for baseline and candidates alike.
    pub validate_solution: bool,
    /// Tolerances of that gate: all scaled KKT residuals must be below
    /// `kkt_tolerance`, except complementarity, which must be below
    /// `kkt_complementarity_tolerance`. An interior-point backend (Clarabel)
    /// stops at its duality gap and leaves complementarity around 1e-5 to
    /// 1e-3 at the walker's gap setting while feasibility and stationarity
    /// sit at 1e-10 (2026-10-02): with both at 1e-6 every Clarabel solve was
    /// rejected and the walker ran on its fallback. Defaults keep 1e-6.
    pub kkt_tolerance: f64,
    pub kkt_complementarity_tolerance: f64,
    /// Weight on the height-cap slack (see `Input::com_z_max`).
    pub reach_weight: f64,
    /// Diagnostic traces, written as lines to the `trace` of
    /// [`solve_traced`] (nothing is printed; the solve is unchanged):
    /// the touchdown node the timed landing chose when it shortened the step
    /// (`[landing-time]`, was `MPC_LANDING_TIME_DEBUG`), and why a solve
    /// failed -- the strict lexicographic problem's broken equality group,
    /// the KKT gate, a non-optimal status (`[lex-diag]`, `[mpc-gate]`,
    /// `[mpc-fail]`, was `MPC_FS_DEBUG`).
    pub trace_landing_time: bool,
    pub trace_failures: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            horizon: 8,
            dt: 0.05,
            mass: 1.0,
            gravity: 9.81,
            friction_mu: 0.6,
            cop_half: (0.05, 0.025),
            torsion_mu: 0.02,
            max_fz: 1000.0,
            position_weight: 20.0,
            velocity_weight: 4.0,
            momentum_weight: 2.0,
            acceleration_weight: 6.0,
            wrench_weight: 2e-4,
            cop_weight: 0.0,
            net_cop_weight: 0.0,
            net_cop_equality: false,
            landing_capture_weight: 0.0,
            landing_capture_horizon: 0,
            landing_lexicographic: false,
            landing_timing: false,
            landing_time_weight: 1e-3,
            landing_time_from_x: true,
            landing_time_gain_min: 1e-3,
            fz_split_equality: false,
            capture_cop_lexicographic: false,
            capture_cop_wp: 0.1,
            capture_cop_ws: 0.1,
            capture_cop_margin: 0.005,
            capture_cop_net_eq: true,
            capture_cop_total_eq: true,
            capture_cop_box_ignores_min: false,
            capture_cop_force_margin: 0.1,
            capture_cop_weight: 0.0,
            landing_moment_coupling: true,
            arm_about_prior: false,
            ldot_weight: 0.0,
            dcm_weight: 0.0,
            omega: 0.0,
            smooth_weight: 1e-3,
            qp: QpConfig::default(),
            validate_solution: false,
            kkt_tolerance: 1e-6,
            kkt_complementarity_tolerance: 1e-6,
            reach_weight: 1e6,
            trace_landing_time: false,
            trace_failures: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub sole_wrenches: [Wrench; 2],
    pub acceleration: na::Vector3<f64>,
    /// Predicted CoM `[position, velocity]` at the END of each node, node 0
    /// first. The node-0 acceleration alone proved unusable as a WBC
    /// reference (a 50 ms average handed to a 5 ms loop; measured
    /// 2026-09-26 it collapsed the robot through a vertical feedback loop),
    /// so a consumer that wants the MPC to steer the CoM takes the
    /// trajectory and keeps its own tracking law.
    pub trajectory: Vec<[na::Vector3<f64>; 2]>,
    /// The landing offset the MPC chose (world x, y), when `Input::landing`
    /// was given.
    pub landing_shift: Option<na::Vector2<f64>>,
    /// The net-CoP plan the lexicographic CoP stage fixed, one per node.
    pub cop_plan: Option<Vec<na::Vector2<f64>>>,
    /// The strict CoP equality could not be met and this is the soft retry.
    pub soft_fallback: bool,
    /// Touchdown node the timed lexicographic landing chose (`Config::
    /// landing_timing`); equals `Landing::first_node` when unchanged.
    pub landing_node: Option<usize>,
    /// Predicted angular momentum at the END of each node, node 0 first.
    pub momentum_trajectory: Vec<na::Vector3<f64>>,
    pub terminal_com: na::Vector3<f64>,
    pub terminal_momentum: na::Vector3<f64>,
}

fn skew(v: &na::Vector3<f64>) -> na::Matrix3<f64> {
    na::Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Dense original QP, exposed for identical-input solver comparisons.
#[derive(Clone, Debug)]
pub struct DenseProblem {
    pub h: na::DMatrix<f64>,
    pub linear: na::DVector<f64>,
    pub d: na::DMatrix<f64>,
    pub rhs: na::DVector<f64>,
    pub initial: na::DVector<f64>,
    /// Constant omitted from the solver objective (for OCP equivalence).
    pub objective_constant: f64,
    pub terminal_map: na::DMatrix<f64>,
    pub terminal_bias: na::DVector<f64>,
    /// Per-node CoM position and velocity as affine functions of the
    /// decision vector: `(rows 0..6 of the state map, bias 0..6)` at the END
    /// of each node. Kept so a solution can report its predicted trajectory,
    /// not just its terminal state (see `Solution::trajectory`).
    pub node_maps: Vec<(na::DMatrix<f64>, na::DVector<f64>)>,
    /// Equality rows, when something asks for one (see
    /// `Config::net_cop_equality`). `None` keeps the shipped problem, which
    /// has inequalities only.
    pub a_eq: Option<na::DMatrix<f64>>,
    pub b_eq: Option<na::DVector<f64>>,
    /// The net-CoP plan the lexicographic CoP stage fixed (see `Config`).
    pub cop_plan: Option<Vec<na::Vector2<f64>>>,
    /// Touchdown node chosen by the timed landing stage.
    pub landing_node: Option<usize>,
}

/// Assemble without running a solver; allocation time is separate from solve time.
/// Rows of the landing-capture residual, one 2x2 block per node from the
/// landing on: residual_k = C_k dp + o_k (world x, y). Empty when the
/// inputs it needs are missing. The arithmetic is exactly the one the
/// penalty rows use, so both paths see the same numbers.
fn capture_rows(input: &Input, cfg: &Config, land: &Landing) -> Vec<(na::Matrix2<f64>, na::Vector2<f64>)> {
    capture_rows_at(input, cfg, land, land.first_node)
}

/// [`capture_rows`] with the touchdown brought forward to node `k_td`
/// (<= `first_node`): on the nodes in between the pressure is the landing
/// itself (`position + prior + dp`) instead of the plan's net CoP.
fn capture_rows_at(input: &Input, cfg: &Config, land: &Landing, k_td: usize) -> Vec<(na::Matrix2<f64>, na::Vector2<f64>)> {
    let n = cfg.horizon.max(cfg.landing_capture_horizon);
    let mut out = Vec::new();
    if cfg.omega <= 0.0 {
        return out;
    }
    let (Some(xi0), Some(pbar), Some(xiref)) = (
        input.dcm_now,
        input.reference_net_cop_schedule.as_ref(),
        input.reference_dcm_schedule.as_ref(),
    ) else {
        return out;
    };
    let wdt = cfg.omega * cfg.dt;
    let mut a = xi0;
    let mut b = na::Matrix2::<f64>::zeros();
    let early = na::Vector2::new(land.position[0] + land.prior[0], land.position[1] + land.prior[1]);
    for k in 0..n {
        let p = if k >= k_td && k < land.first_node {
            early
        } else {
            pbar.get(k).copied().unwrap_or_else(|| pbar[pbar.len() - 1])
        };
        a = a * (1.0 + wdt) - p * wdt;
        b *= 1.0 + wdt;
        if k >= k_td {
            b -= na::Matrix2::identity() * wdt;
        }
        if k >= k_td && k + 1 <= n {
            let target = xiref.get(k).copied().unwrap_or_else(|| xiref[xiref.len() - 1]);
            let jk = input
                .reference_dcm_sensitivity
                .as_ref()
                .and_then(|s| s.get(k).or(s.last()))
                .copied()
                .unwrap_or_else(na::Matrix2::zeros);
            out.push((b - jk, a - target));
        }
    }
    out
}

/// Exact minimiser of 0.5 x'Hx + g'x over lo <= x <= hi in two variables
/// (H symmetric positive definite): the optimum is the unconstrained point,
/// a point on one edge with the other variable free, or a corner; take the
/// best feasible candidate.
fn box_qp_2d(h: &na::Matrix2<f64>, g: &na::Vector2<f64>, lo: [f64; 2], hi: [f64; 2]) -> na::Vector2<f64> {
    let f = |x: &na::Vector2<f64>| 0.5 * x.dot(&(h * x)) + g.dot(x);
    let inside = |x: &na::Vector2<f64>| (0..2).all(|j| x[j] >= lo[j] - 1e-12 && x[j] <= hi[j] + 1e-12);
    let mut cands: Vec<na::Vector2<f64>> = Vec::new();
    if let Some(inv) = h.try_inverse() {
        cands.push(-(inv * g));
    }
    for j in 0..2 {
        let o = 1 - j;
        for v in [lo[j], hi[j]] {
            // Fix x_j = v, minimise over x_o, then clamp.
            let mut x = na::Vector2::zeros();
            x[j] = v;
            if h[(o, o)] > 0.0 {
                x[o] = (-(g[o] + h[(o, j)] * v) / h[(o, o)]).clamp(lo[o], hi[o]);
            } else {
                x[o] = lo[o];
            }
            cands.push(x);
        }
    }
    let mut best = na::Vector2::new(lo[0].max(hi[0].min(0.0)), lo[1].max(hi[1].min(0.0)));
    let mut best_f = f(&best);
    for c in cands {
        if inside(&c) {
            let v = f(&c);
            if v < best_f {
                best_f = v;
                best = c;
            }
        }
    }
    best
}

/// The landing offset the lexicographic mode fixes (see
/// `Config::landing_lexicographic`), or `None` when there is no landing or
/// no capture rows.
pub fn lexicographic_landing(input: &Input, cfg: &Config) -> Option<na::Vector2<f64>> {
    let land = input.landing.as_ref()?;
    landing_at(input, cfg, land, land.first_node).map(|(dp, _)| dp)
}

/// Stage one of the lexicographic CoP: the net-CoP plan over the horizon,
/// or `None` when the inputs it needs are missing.
///
/// LIPM in the DCM, `xi_{k+1} = (1 + w dt) xi_k - w dt p_k`, from the
/// measured `dcm_now`. Per node the pressure point is boxed to what the
/// SCHEDULED load split can reach through the sole moments: the share-
/// weighted sum of the soles' CoP boxes (their world-axis bounding boxes),
/// shrunk by `capture_cop_margin`. That box is exactly the set the wrench
/// stage can hit with the split held, so the equalities it then writes are
/// feasible. The two axes decouple, so it is two small box QPs.
pub fn lexicographic_cop(input: &Input, cfg: &Config) -> Option<Vec<na::Vector2<f64>>> {
    let landing = if cfg.landing_lexicographic { lexicographic_landing(input, cfg) } else { None };
    lexicographic_cop_with_landing(input, cfg, landing)
}

/// [`lexicographic_cop`] with the landing's decided offset: from the
/// landing's first node on, that foot's box moves by `dp`.
pub fn lexicographic_cop_with_landing(
    input: &Input,
    cfg: &Config,
    landing_shift: Option<na::Vector2<f64>>,
) -> Option<Vec<na::Vector2<f64>>> {
    let n = cfg.horizon;
    let xi0 = input.dcm_now?;
    let xiref = input.reference_dcm_schedule.as_ref()?;
    let pbar = input.reference_net_cop_schedule.as_ref()?;
    if cfg.omega <= 0.0 || n == 0 {
        return None;
    }
    let wdt = cfg.omega * cfg.dt;
    let m = cfg.capture_cop_margin.max(0.0);
    // Reachable net-CoP polygon per node: the convex hull of
    // `s * corner_a + (1 - s) * corner_b` over the corners of the two soles'
    // (yawed) CoP boxes and the admissible shares s. The share is ONE
    // number for both axes, which is what an axis-separable box got wrong
    // (2026-10-02 01:28: x wanted 45% on the landed foot, y wanted 78% on
    // the other; each alone reachable, together not). Half-planes are
    // shifted inward by the margin. Single support: the yawed box itself.
    let mut polys: Vec<Vec<(na::Vector2<f64>, f64)>> = Vec::with_capacity(n); // (normal, offset): n.p <= offset
    let mut last_poly: Option<Vec<(na::Vector2<f64>, f64)>> = None;
    let mut last_centre = na::Vector2::new(pbar[0].x, pbar[0].y);
    for k in 0..n {
        let feet: Vec<usize> = (0..2).filter(|&f| input.contact_schedule[k][f]).collect();
        if feet.is_empty() {
            polys.push(last_poly.clone().unwrap_or_default());
            continue;
        }
        let corners = |f: usize| -> [na::Vector2<f64>; 4] {
            let c = &input.contacts[f];
            let r = &c.rotation;
            let ux = na::Vector2::new(r[(0, 0)], r[(1, 0)]);
            let uy = na::Vector2::new(r[(0, 1)], r[(1, 1)]);
            let shift = match (input.landing, landing_shift) {
                (Some(land), Some(dp)) if land.foot == f && k >= land.first_node => dp,
                _ => na::Vector2::zeros(),
            };
            let centre = na::Vector2::new(c.position.x, c.position.y) + shift;
            let (hx, hy) = (c.cop_half.0.max(0.0), c.cop_half.1.max(0.0));
            [
                centre + ux * hx + uy * hy,
                centre + ux * hx - uy * hy,
                centre - ux * hx + uy * hy,
                centre - ux * hx - uy * hy,
            ]
        };
        let mut pts: Vec<na::Vector2<f64>> = Vec::new();
        if feet.len() == 1 {
            pts.extend_from_slice(&corners(feet[0]));
        } else {
            let (a, b) = (feet[0], feet[1]);
            let total_ref: f64 = feet.iter().map(|&f| input.reference_fz_schedule[k][f].max(0.0)).sum();
            let mins = if cfg.capture_cop_box_ignores_min {
                [0.0, 0.0]
            } else {
                [input.min_fz_schedule[k][a].max(0.0), input.min_fz_schedule[k][b].max(0.0)]
            };
            let f_tot = total_ref.max(mins[0] + mins[1]).max(1e-9);
            let slack = cfg.capture_cop_force_margin.max(0.0) * f_tot;
            let caps = [
                cfg.max_fz * input.activation_schedule[k][a].clamp(0.0, 1.0),
                cfg.max_fz * input.activation_schedule[k][b].clamp(0.0, 1.0),
            ];
            // share of foot a: at least its minimum (+slack), at most what b's
            // minimum (+slack) and a's own ceiling allow.
            let s_lo = ((mins[0] + slack) / f_tot).max(1.0 - (caps[1] - slack) / f_tot).clamp(0.0, 1.0);
            let s_hi = ((f_tot - mins[1] - slack) / f_tot).min((caps[0] - slack) / f_tot).clamp(0.0, 1.0);
            let (s_lo, s_hi) = if s_lo <= s_hi { (s_lo, s_hi) } else { (0.5, 0.5) };
            let (ca, cb) = (corners(a), corners(b));
            for &s in &[s_lo, s_hi] {
                for pa in &ca {
                    for pb in &cb {
                        pts.push(pa * s + pb * (1.0 - s));
                    }
                }
            }
        }
        let hull = convex_hull(&pts);
        let centre = hull.iter().fold(na::Vector2::zeros(), |acc, p| acc + p) / hull.len().max(1) as f64;
        let mut half: Vec<(na::Vector2<f64>, f64)> = Vec::new();
        if hull.len() >= 3 {
            for i in 0..hull.len() {
                let a = hull[i];
                let b = hull[(i + 1) % hull.len()];
                let e = b - a;
                let len = e.norm();
                if len < 1e-9 {
                    continue;
                }
                // Outward normal of a counter-clockwise polygon.
                let nrm = na::Vector2::new(e.y, -e.x) / len;
                half.push((nrm, nrm.dot(&a) - m));
            }
        } else {
            // Degenerate (a point or a segment): a small box around it.
            for (nrm, ext) in [
                (na::Vector2::new(1.0, 0.0), 1e-3),
                (na::Vector2::new(-1.0, 0.0), 1e-3),
                (na::Vector2::new(0.0, 1.0), 1e-3),
                (na::Vector2::new(0.0, -1.0), 1e-3),
            ] {
                half.push((nrm, nrm.dot(&centre) + ext));
            }
        }
        last_centre = centre;
        last_poly = Some(half.clone());
        polys.push(half);
    }
    let _ = last_centre;
    // One QP in the 2n pressure points: DCM tracking through the LIPM,
    // deviation from the plan's net CoP, node-to-node change; polygon rows.
    let nv = 2 * n;
    let mut h = na::DMatrix::zeros(nv, nv);
    let mut g = na::DVector::zeros(nv);
    for axis in 0..2 {
        for k in 0..n {
            let a = (1.0 + wdt).powi(k as i32 + 1) * xi0[axis];
            let target = xiref.get(k).copied().unwrap_or_else(|| xiref[xiref.len() - 1])[axis];
            let w = if k + 1 == n { 4.0 } else { 1.0 };
            let mut row = na::DVector::zeros(nv);
            for j in 0..=k {
                row[2 * j + axis] = -wdt * (1.0 + wdt).powi((k - j) as i32);
            }
            h += &row * row.transpose() * w;
            g += &row * ((a - target) * w);
        }
        for k in 0..n {
            let p = pbar.get(k).copied().unwrap_or_else(|| pbar[pbar.len() - 1])[axis];
            let i = 2 * k + axis;
            h[(i, i)] += cfg.capture_cop_wp;
            g[i] -= cfg.capture_cop_wp * p;
            if k > 0 {
                let ws = cfg.capture_cop_ws;
                let j = 2 * (k - 1) + axis;
                h[(i, i)] += ws;
                h[(j, j)] += ws;
                h[(i, j)] -= ws;
                h[(j, i)] -= ws;
            }
        }
    }
    let n_rows: usize = polys.iter().map(|p| p.len()).sum();
    let mut d = na::DMatrix::zeros(n_rows, nv);
    let mut rhs = na::DVector::zeros(n_rows);
    let mut row = 0;
    for (k, poly) in polys.iter().enumerate() {
        for (nrm, off) in poly {
            d[(row, 2 * k)] = nrm.x;
            d[(row, 2 * k + 1)] = nrm.y;
            rhs[row] = *off;
            row += 1;
        }
    }
    let mut qp = QpConfig::default();
    qp.solver = misa_wbc::QpSolver::Clarabel;
    let sol = solve_qp(&h, &g, None, None, Some(&d), Some(&rhs), None, &qp);
    if sol.status != QpStatus::Optimal || sol.x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some((0..n).map(|k| na::Vector2::new(sol.x[2 * k], sol.x[2 * k + 1])).collect())
}

/// Convex hull (counter-clockwise, no repeated points) of a small point set.
fn convex_hull(pts: &[na::Vector2<f64>]) -> Vec<na::Vector2<f64>> {
    let mut p: Vec<na::Vector2<f64>> = pts.to_vec();
    p.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));
    p.dedup_by(|a, b| (a.x - b.x).abs() < 1e-12 && (a.y - b.y).abs() < 1e-12);
    if p.len() < 3 {
        return p;
    }
    let cross = |o: &na::Vector2<f64>, a: &na::Vector2<f64>, b: &na::Vector2<f64>| {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    };
    let mut lower: Vec<na::Vector2<f64>> = Vec::new();
    for q in &p {
        while lower.len() >= 2 && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], q) <= 1e-14 {
            lower.pop();
        }
        lower.push(*q);
    }
    let mut upper: Vec<na::Vector2<f64>> = Vec::new();
    for q in p.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], q) <= 1e-14 {
            upper.pop();
        }
        upper.push(*q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Capture box-QP for one touchdown node: (dp, residual cost).
fn landing_at(input: &Input, cfg: &Config, land: &Landing, k_td: usize) -> Option<(na::Vector2<f64>, f64)> {
    let rows = capture_rows_at(input, cfg, land, k_td);
    if rows.is_empty() {
        return None;
    }
    let mut h = na::Matrix2::<f64>::zeros();
    let mut g = na::Vector2::<f64>::zeros();
    let mut c0 = 0.0;
    for (c, o) in &rows {
        h += c.transpose() * c;
        g += c.transpose() * o;
        c0 += o.dot(o);
    }
    let rho = 1e-6 * (h.trace() * 0.5).max(1e-12);
    let prior = na::Vector2::new(land.prior[0], land.prior[1]);
    h += na::Matrix2::identity() * rho;
    g += prior * rho;
    // `max`/`min` may be NEGATIVE: a negative `min` is a lower bound above
    // zero (a shift the landing MUST take, e.g. a capture floor) and a
    // negative `max` an upper bound below zero. Clamping them to zero, as
    // this did until 2026-10-20, silently dropped every such floor; for the
    // non-negative bounds every recipe uses the result is unchanged.
    let hi = [land.max[0] - land.prior[0], land.max[1] - land.prior[1]];
    let lo = [-(land.min[0] + land.prior[0]), -(land.min[1] + land.prior[1])];
    let lo = [lo[0].min(hi[0]), lo[1].min(hi[1])];
    let dp = match land.terrain_x {
        Some((wlo, whi, w)) if w > 0.0 && wlo <= whi => {
            // Convex piecewise quadratic in x: zero inside the window, a
            // one-sided quadratic outside. Exact minimiser: the best of the
            // three regions, each a box QP (the region's x bound added).
            let base = land.position[0] + land.prior[0];
            let mut best: Option<(na::Vector2<f64>, f64)> = None;
            let f = |x: &na::Vector2<f64>, hh: &na::Matrix2<f64>, gg: &na::Vector2<f64>| 0.5 * x.dot(&(hh * x)) + gg.dot(x);
            // Inside: dp.x in [wlo - base, whi - base] intersected with the box.
            let lo_in = [lo[0].max(wlo - base), lo[1]];
            let hi_in = [hi[0].min(whi - base), hi[1]];
            if lo_in[0] <= hi_in[0] {
                let x = box_qp_2d(&h, &g, lo_in, hi_in);
                best = Some((x, f(&x, &h, &g)));
            }
            // Below: cost w (base + dp.x - wlo)^2 for dp.x <= wlo - base.
            for (edge, side_hi) in [(wlo, true), (whi, false)] {
                let mut hh = h;
                let mut gg = g;
                hh[(0, 0)] += 2.0 * w;
                gg[0] += 2.0 * w * (base - edge);
                let (rlo, rhi) = if side_hi {
                    ([lo[0], lo[1]], [hi[0].min(edge - base), hi[1]])
                } else {
                    ([lo[0].max(edge - base), lo[1]], [hi[0], hi[1]])
                };
                if rlo[0] <= rhi[0] {
                    let x = box_qp_2d(&hh, &gg, rlo, rhi);
                    let v = f(&x, &hh, &gg) + w * (base - edge) * (base - edge);
                    if best.is_none_or(|(_, b)| v < b) {
                        best = Some((x, v));
                    }
                }
            }
            best.map_or_else(|| box_qp_2d(&h, &g, lo, hi), |(x, _)| x)
        }
        _ => box_qp_2d(&h, &g, lo, hi),
    };
    // Residual on the NOMINAL landing's nodes only (k >= first_node, the
    // last rows): an earlier touchdown has more rows, and summing them all
    // would charge it for capturing sooner.
    let n_nom = cfg.horizon.saturating_sub(land.first_node).min(rows.len());
    let cost = rows[rows.len() - n_nom..]
        .iter()
        .map(|(c, o)| {
            let r = c * dp + o;
            if cfg.landing_time_from_x { r.x * r.x } else { r.norm_squared() }
        })
        .sum::<f64>();
    let _ = c0;
    Some((dp, cost))
}

/// The timed lexicographic landing: `(dp, touchdown node)`. With no timing
/// freedom (`min_first_node >= first_node`) or timing off it is
/// [`lexicographic_landing`] at the nominal node.
pub fn lexicographic_landing_timed(input: &Input, cfg: &Config) -> Option<(na::Vector2<f64>, usize)> {
    landing_timed_traced(input, cfg, &mut Vec::new())
}

fn landing_timed_traced(input: &Input, cfg: &Config, trace_out: &mut Vec<String>) -> Option<(na::Vector2<f64>, usize)> {
    let land = input.landing.as_ref()?;
    if !cfg.landing_timing || land.min_first_node >= land.first_node {
        return lexicographic_landing(input, cfg).map(|dp| (dp, land.first_node));
    }
    let mut best: Option<(na::Vector2<f64>, usize, f64)> = None;
    let debug = cfg.trace_landing_time;
    let mut trace = String::new();
    let nominal_cost = landing_at(input, cfg, land, land.first_node).map(|(_, c)| c);
    for k_td in (land.min_first_node..=land.first_node).rev() {
        let Some((dp, cost)) = landing_at(input, cfg, land, k_td) else { continue };
        let shortened = (land.first_node - k_td) as f64 * cfg.dt;
        let total = cost + cfg.landing_time_weight * shortened * shortened
            + if k_td < land.first_node { cfg.landing_time_gain_min } else { 0.0 };
        let _ = nominal_cost;
        if debug {
            trace.push_str(&format!(" k{k_td}:{:.2e}+{:.1e}", cost, total - cost));
        }
        if best.is_none_or(|(_, _, b)| total < b - 1e-15) {
            best = Some((dp, k_td, total));
        }
    }
    if debug && let Some((dp, k, _)) = best && k < land.first_node {
        trace_out.push(format!(
            "  [landing-time] first_node={} chosen={k} dp=({:+.3},{:+.3}) xi0={:?}{trace}",
            land.first_node, dp.x, dp.y, input.dcm_now.map(|v| (v.x, v.y))
        ));
    }
    best.map(|(dp, k, _)| (dp, k))
}

pub fn assemble(input: &Input, cfg: &Config) -> Option<DenseProblem> {
    assemble_traced(input, cfg, &mut Vec::new())
}

fn assemble_traced(input: &Input, cfg: &Config, trace: &mut Vec<String>) -> Option<DenseProblem> {
    let (landing_fixed, landing_node) = if cfg.landing_lexicographic {
        match landing_timed_traced(input, cfg, trace) {
            Some((dp, k)) => (Some(dp), Some(k)),
            None => (None, None),
        }
    } else {
        (None, None)
    };
    let cop_fixed = if cfg.capture_cop_lexicographic {
        lexicographic_cop_with_landing(input, cfg, landing_fixed)
    } else {
        None
    };
    // With the landing decided before assembly, the lexicographic CoP mode
    // puts it into the lever arms EXACTLY (position + dp from its first
    // node) instead of the moment coupling linearised at the reference load:
    // that coupling is exact only at fz = F_ref, and after a push the landed
    // foot carries far more, so the strict net-CoP equality asked for a
    // moment the sole could not give (2026-10-02 01:22, 54 soft fallbacks).
    let exact_landing = if cfg.capture_cop_lexicographic { landing_fixed } else { None };
    let foot_position = |k: usize, foot: usize| -> na::Vector3<f64> {
        let mut p = input.contacts[foot].position;
        if let (Some(land), Some(dp)) = (input.landing, exact_landing)
            && land.foot == foot
            && k >= land.first_node
        {
            p.x += dp.x;
            p.y += dp.y;
        }
        p
    };
    let n = cfg.horizon;
    let n_land = if input.landing.is_some() { 2 } else { 0 };
    // Height-cap slacks, one per node, after the landing variables.
    let n_reach = if input.com_z_max.as_ref().is_some_and(|c| c.len() >= n) { n } else { 0 };
    let nu = 12 * n + n_land + n_reach;
    if n == 0
        || [
            cfg.dt,
            cfg.mass,
            cfg.gravity,
            cfg.friction_mu,
            cfg.cop_half.0,
            cfg.cop_half.1,
            cfg.torsion_mu,
            cfg.max_fz,
            cfg.position_weight,
            cfg.velocity_weight,
            cfg.momentum_weight,
            cfg.acceleration_weight,
            cfg.wrench_weight,
            cfg.cop_weight,
            cfg.net_cop_weight,
            cfg.dcm_weight,
            cfg.omega,
            cfg.ldot_weight,
            cfg.landing_capture_weight,
            cfg.smooth_weight,
        ]
        .iter()
        .any(|v| !v.is_finite())
        || [
            cfg.position_weight,
            cfg.velocity_weight,
            cfg.momentum_weight,
            cfg.acceleration_weight,
            cfg.wrench_weight,
            cfg.cop_weight,
            cfg.net_cop_weight,
            cfg.dcm_weight,
            cfg.omega,
            cfg.ldot_weight,
            cfg.landing_capture_weight,
            cfg.smooth_weight,
            cfg.torsion_mu,
        ]
        .iter()
        .any(|v| *v < 0.0)
        || cfg.dt <= 0.0
        || cfg.mass <= 0.0
        || cfg.gravity <= 0.0
        || cfg.friction_mu < 0.0
        || cfg.cop_half.0 <= 0.0
        || cfg.cop_half.1 <= 0.0
        || cfg.max_fz <= 0.0
        || input.contact_schedule.len() != n
        || input.activation_schedule.len() != n
        || input.reference_fz_schedule.len() != n
        || input
            .reference_cop_schedule
            .as_ref()
            .is_some_and(|c| c.len() < n)
        || input
            .reference_net_cop_schedule
            .as_ref()
            .is_some_and(|c| c.len() < n)
        || input.min_fz_schedule.len() != n
        || input
            .com_z_max
            .as_ref()
            .is_some_and(|c| c.len() < n || c.iter().any(|z| !z.is_finite()))
        || input.contacts.iter().any(|c| {
            !c.cop_half.0.is_finite()
                || !c.cop_half.1.is_finite()
                || c.cop_half.0 <= 0.0
                || c.cop_half.1 <= 0.0
                || c.rotation
                    .iter()
                    .chain(c.position.iter())
                    .any(|x| !x.is_finite())
        })
        || input
            .reference_fz_schedule
            .iter()
            .zip(&input.min_fz_schedule)
            .zip(&input.activation_schedule)
            .any(|((reference, minimum), activation)| {
                (0..2).any(|foot| {
                    !reference[foot].is_finite()
                        || !minimum[foot].is_finite()
                        || !activation[foot].is_finite()
                        || activation[foot] < 0.0
                        || activation[foot] > 1.0
                        || minimum[foot] < 0.0
                        || reference[foot] < minimum[foot]
                        || minimum[foot] > cfg.max_fz * activation[foot]
                        || reference[foot] > cfg.max_fz
                })
            })
    {
        return None;
    }
    if input
        .com
        .iter()
        .chain(input.velocity.iter())
        .chain(input.angular_momentum.iter())
        .chain(input.target_com.iter())
        .chain(input.target_velocity.iter())
        .chain(input.target_acceleration.iter())
        .chain(input.previous.iter().flat_map(|w| w.iter()))
        .any(|x| !x.is_finite())
    {
        return None;
    }

    // x = [com, velocity, angular momentum] = map * U + bias.
    let mut map = na::DMatrix::<f64>::zeros(9, nu);
    let mut bias = na::DVector::from_iterator(
        9,
        input
            .com
            .iter()
            .chain(input.velocity.iter())
            .chain(input.angular_momentum.iter())
            .copied(),
    );
    let mut rows: Vec<(na::DVector<f64>, f64)> = Vec::new();
    let mut eq_rows: Vec<na::DVector<f64>> = Vec::new();
    // Per node: the map from the decision vector to the net-CoP offset
    // (x, y) from the planned net CoP (see `Config::capture_cop_weight`).
    let mut cop_maps: Vec<Option<(na::DVector<f64>, na::DVector<f64>)>> = vec![None; n];
    let add_residual = |rows: &mut Vec<_>, coeff: na::DVector<f64>, offset: f64, weight: f64| {
        if weight > 0.0 {
            rows.push((coeff * weight.sqrt(), offset * weight.sqrt()));
        }
    };
    let gravity = na::Vector3::new(0.0, 0.0, -cfg.gravity);
    let mut node_maps: Vec<(na::DMatrix<f64>, na::DVector<f64>)> = Vec::with_capacity(n);
    // Angular-momentum rows at the end of the previous node, for the L-dot
    // residual. Before node 0 that is the measured momentum with no
    // dependence on the decision vector.
    let mut prev_mom: (na::DMatrix<f64>, na::DVector<f64>) =
        (na::DMatrix::zeros(3, nu), bias.rows(6, 3).into_owned());

    for k in 0..n {
        let arm_centre = match (&input.prior_com, cfg.arm_about_prior) {
            (Some(prior), true) if k < prior.len() => prior[k],
            _ => input.target_com,
        };
        for axis in 0..3 {
            let old_c = map.row(axis).into_owned();
            let old_v = map.row(3 + axis).into_owned();
            map.row_mut(axis).copy_from(&(old_c + old_v.scale(cfg.dt)));
            bias[axis] += cfg.dt * bias[3 + axis] + 0.5 * cfg.dt * cfg.dt * gravity[axis];
            bias[3 + axis] += cfg.dt * gravity[axis];
        }
        for foot in 0..2 {
            let base = 12 * k + 6 * foot;
            let contact = &input.contacts[foot];
            let force_gain = contact.rotation / cfg.mass;
            let torque_gain = skew(&(foot_position(k, foot) - arm_centre)) * contact.rotation;
            for r in 0..3 {
                for j in 0..3 {
                    map[(r, base + 3 + j)] += 0.5 * cfg.dt * cfg.dt * force_gain[(r, j)];
                    map[(3 + r, base + 3 + j)] += cfg.dt * force_gain[(r, j)];
                    map[(6 + r, base + j)] += cfg.dt * contact.rotation[(r, j)];
                    map[(6 + r, base + 3 + j)] += cfg.dt * torque_gain[(r, j)];
                }
            }
            // Landing as a variable: the extra moment `dp x F_k` about the
            // target, linearised at the planned load (see `Landing`).
            if let Some(land) = input.landing
                && cfg.landing_moment_coupling
                && exact_landing.is_none()
                && land.foot == foot
                && k >= land.first_node
                && input.contact_schedule[k][foot]
            {
                let f_prior = contact.rotation
                    * na::Vector3::new(0.0, 0.0, input.reference_fz_schedule[k][foot]);
                let gain = -skew(&f_prior);
                for r in 0..3 {
                    for j in 0..2 {
                        map[(6 + r, 12 * n + j)] += cfg.dt * gain[(r, j)];
                    }
                }
            }
        }

        // State at the END of node k, with node k's own wrench included.
        node_maps.push((map.rows(0, 9).into_owned(), bias.rows(0, 9).into_owned()));
        if n_reach > 0 {
            let mut coeff = na::DVector::zeros(nu);
            coeff[12 * n + n_land + k] = 1.0;
            add_residual(&mut rows, coeff, 0.0, cfg.reach_weight);
        }
        if cfg.ldot_weight > 0.0 {
            for axis in 0..3 {
                let coeff = (map.row(6 + axis).transpose() - prev_mom.0.row(axis).transpose())
                    / cfg.dt;
                let offset = (bias[6 + axis] - prev_mom.1[axis]) / cfg.dt;
                add_residual(&mut rows, coeff, offset, cfg.ldot_weight);
            }
        }
        prev_mom = (map.rows(6, 3).into_owned(), bias.rows(6, 3).into_owned());

        let frac = (k + 1) as f64 / n as f64;
        let ref_c = input.com + (input.target_com - input.com) * frac;
        let ref_v = input.velocity + (input.target_velocity - input.velocity) * frac;
        let terminal = if k + 1 == n { 4.0 } else { 1.0 };
        for axis in 0..3 {
            add_residual(
                &mut rows,
                map.row(axis).transpose(),
                bias[axis] - ref_c[axis],
                cfg.position_weight * terminal,
            );
            add_residual(
                &mut rows,
                map.row(3 + axis).transpose(),
                bias[3 + axis] - ref_v[axis],
                cfg.velocity_weight * terminal,
            );
            add_residual(
                &mut rows,
                map.row(6 + axis).transpose(),
                bias[6 + axis],
                cfg.momentum_weight * terminal,
            );
        }

        // Penalise the DIVERGENT component, xi = c + c_dot/omega, which is
        // the only thing the pressure point can steer. Built from the rows
        // already assembled above, so it costs one axpy per axis.
        if cfg.dcm_weight > 0.0 && cfg.omega > 0.0 {
            for axis in 0..2 {
                let reference = input
                    .reference_dcm_schedule
                    .as_ref()
                    .and_then(|s| s.get(k))
                    .map(|xi| xi[axis])
                    .unwrap_or(ref_c[axis] + ref_v[axis] / cfg.omega);
                let coeff =
                    map.row(axis).transpose() + map.row(3 + axis).transpose() / cfg.omega;
                add_residual(
                    &mut rows,
                    coeff,
                    bias[axis] + bias[3 + axis] / cfg.omega - reference,
                    cfg.dcm_weight * terminal,
                );
            }
        }

        for axis in 0..3 {
            let mut coeff = na::DVector::zeros(nu);
            for foot in 0..2 {
                let base = 12 * k + 6 * foot;
                for j in 0..3 {
                    coeff[base + 3 + j] = input.contacts[foot].rotation[(axis, j)] / cfg.mass;
                }
            }
            add_residual(
                &mut rows,
                coeff,
                gravity[axis] - input.target_acceleration[axis],
                cfg.acceleration_weight,
            );
        }

        for foot in 0..2 {
            for j in 0..6 {
                let idx = 12 * k + 6 * foot + j;
                let mut coeff = na::DVector::zeros(nu);
                coeff[idx] = 1.0;
                let reference = if j == 5 && input.contact_schedule[k][foot] {
                    input.reference_fz_schedule[k][foot]
                } else {
                    0.0
                };
                add_residual(&mut rows, coeff.clone(), -reference, cfg.wrench_weight);
                // The wrench is (m_x, m_y, m_z, f_x, f_y, f_z) -- the CoP box
                // constraints below are written on components 0 and 1, and
                // friction on 3 and 4. A first draft of this term used 3 and 4
                // for the moments and constrained the tangential forces
                // instead; the unit test made the same mistake and passed.
                if cfg.cop_weight > 0.0
                    && input.contact_schedule[k][foot]
                    && let Some(cops) = input.reference_cop_schedule.as_ref()
                    && (j == 0 || j == 1)
                {
                    let c = cops[k][foot];
                    let mut row = na::DVector::zeros(nu);
                    row[idx] = 1.0;
                    // m_x - c_y f_z = 0 ;  m_y + c_x f_z = 0
                    row[12 * k + 6 * foot + 5] = if j == 0 { -c.y } else { c.x };
                    add_residual(&mut rows, row, 0.0, cfg.cop_weight);
                }
                if k == 0 {
                    add_residual(
                        &mut rows,
                        coeff,
                        -input.previous[foot][j],
                        cfg.smooth_weight,
                    );
                } else {
                    coeff[idx - 12] = -1.0;
                    add_residual(&mut rows, coeff, 0.0, cfg.smooth_weight);
                }
            }
        }
        // Net CoP over both feet (see `reference_net_cop_schedule`). Two rows
        // per node, spanning both feet, so it constrains the SPLIT rather than
        // each foot's own pressure point.
        if (cfg.net_cop_weight > 0.0 || cfg.net_cop_equality || cop_fixed.is_some())
            && let Some(nets) = input.reference_net_cop_schedule.as_ref()
        {
            let c = cop_fixed.as_ref().map_or(nets[k], |plan| plan[k]);
            let mut rx = na::DVector::zeros(nu);
            let mut ry = na::DVector::zeros(nu);
            let mut any = false;
            for foot in 0..2 {
                if !input.contact_schedule[k][foot] {
                    continue;
                }
                any = true;
                let contact = &input.contacts[foot];
                let p = foot_position(k, foot);
                let rot = contact.rotation;
                let b = 12 * k + 6 * foot;
                // The wrench is in the SOLE's frame: the dynamics multiplies
                // it by `contact.rotation` before use, and the friction and
                // CoP boxes are written on its local components. A first
                // version of these rows added the LOCAL moment to a world
                // lever arm; with flat feet at zero yaw that is nearly right,
                // which is how it half-worked (43 -> 32 N) and hid the error.
                //
                // World moment about `c`:  R m + (p - c) x (R f).
                let d = na::Vector3::new(p.x - c.x, p.y - c.y, p.z);
                let cross = skew(&d) * rot;
                if let Some(land) = input.landing
                    && cfg.landing_moment_coupling
                    && exact_landing.is_none()
                    && land.foot == foot
                    && k >= land.first_node
                {
                    let f_prior =
                        rot * na::Vector3::new(0.0, 0.0, input.reference_fz_schedule[k][foot]);
                    let gain = -skew(&f_prior);
                    for j in 0..2 {
                        rx[12 * n + j] += gain[(0, j)];
                        ry[12 * n + j] += gain[(1, j)];
                    }
                }
                for j in 0..3 {
                    rx[b + j] += rot[(0, j)];
                    ry[b + j] += rot[(1, j)];
                    rx[b + 3 + j] += cross[(0, j)];
                    ry[b + 3 + j] += cross[(1, j)];
                }
            }
            if cfg.capture_cop_weight > 0.0 && any {
                let fz: f64 = (0..2)
                    .filter(|&f| input.contact_schedule[k][f])
                    .map(|f| input.reference_fz_schedule[k][f])
                    .sum();
                if fz > 1e-6 {
                    // CoP offset from p_bar: (-M_y, M_x) / Fz.
                    cop_maps[k] = Some((-ry.clone() / fz, rx.clone() / fz));
                }
            }
            let both = input.contact_schedule[k][0] && input.contact_schedule[k][1];
            if any
                && ((cop_fixed.is_some() && cfg.capture_cop_net_eq)
                    || (cfg.net_cop_equality && both))
            {
                eq_rows.push(rx.clone());
                eq_rows.push(ry.clone());
            } else if any && cfg.net_cop_weight > 0.0 {
                add_residual(&mut rows, rx, 0.0, cfg.net_cop_weight);
                add_residual(&mut rows, ry, 0.0, cfg.net_cop_weight);
            }
        }
    }

    // Right-hand sides of the equality rows (the net-CoP rows are all zero).
    let mut eq_rhs = vec![0.0; eq_rows.len()];
    if let (Some(fixed), Some(_)) = (landing_fixed, input.landing) {
        // The landing is decided by the capture stage: dp = fixed.
        for j in 0..2 {
            let mut r = na::DVector::zeros(nu);
            r[12 * n + j] = 1.0;
            eq_rows.push(r);
            eq_rhs.push(fixed[j]);
        }
    }
    if cop_fixed.is_some() && cfg.capture_cop_total_eq {
        for k in 0..n {
            let feet: Vec<usize> = (0..2).filter(|&f| input.contact_schedule[k][f]).collect();
            if feet.is_empty() {
                continue;
            }
            let mut r = na::DVector::zeros(nu);
            let mut total = 0.0;
            for &foot in &feet {
                r[12 * k + 6 * foot + 5] = 1.0;
                total += input.reference_fz_schedule[k][foot];
            }
            eq_rows.push(r);
            eq_rhs.push(total);
        }
    }
    if cfg.fz_split_equality {
        for k in 0..n {
            if !(input.contact_schedule[k][0] && input.contact_schedule[k][1]) {
                continue;
            }
            for foot in 0..2 {
                let mut r = na::DVector::zeros(nu);
                r[12 * k + 6 * foot + 5] = 1.0;
                eq_rows.push(r);
                eq_rhs.push(input.reference_fz_schedule[k][foot]);
            }
        }
    }
    let (a_eq, b_eq) = if eq_rows.is_empty() {
        (None, None)
    } else {
        let mut a = na::DMatrix::zeros(eq_rows.len(), nu);
        for (i, r) in eq_rows.iter().enumerate() {
            a.row_mut(i).copy_from(&r.transpose());
        }
        (Some(a), Some(na::DVector::from_vec(eq_rhs)))
    };
    if let Some(land) = input.landing {
        for j in 0..2 {
            let mut coeff = na::DVector::zeros(nu);
            coeff[12 * n + j] = 1.0;
            add_residual(&mut rows, coeff, land.prior[j], land.weight);
        }
        // Landing capture: propagate the DCM through the horizon with the
        // planned CoP, the CoP after touchdown moving with `dp`, and ask it
        // to meet the plan's DCM. Affine in dp, so it is two extra columns.
        if cfg.landing_capture_weight > 0.0
            && landing_fixed.is_none()
            && cfg.omega > 0.0
            && let (Some(xi0), Some(pbar), Some(xiref)) = (
                input.dcm_now,
                input.reference_net_cop_schedule.as_ref(),
                input.reference_dcm_schedule.as_ref(),
            )
        {
            let wdt = cfg.omega * cfg.dt;
            let mut a = xi0; // xi_k as a constant part ...
            let mut b = na::Matrix2::<f64>::zeros(); // ... plus b * dp
            for k in 0..n {
                let p = pbar.get(k).copied().unwrap_or_else(|| pbar[pbar.len() - 1]);
                // xi_{k+1} = (1 + w dt) xi_k - w dt p_k, with p_k = p + dp after touchdown.
                a = a * (1.0 + wdt) - p * wdt;
                b *= 1.0 + wdt;
                if k >= land.first_node {
                    b -= na::Matrix2::identity() * wdt;
                }
                if k >= land.first_node && k + 1 <= n {
                    let target = xiref.get(k).copied().unwrap_or_else(|| xiref[xiref.len() - 1]);
                    let jk = input
                        .reference_dcm_sensitivity
                        .as_ref()
                        .and_then(|s| s.get(k).or(s.last()))
                        .copied()
                        .unwrap_or_else(na::Matrix2::zeros);
                    for axis in 0..2 {
                        let mut coeff = na::DVector::zeros(nu);
                        coeff[12 * n] = b[(axis, 0)] - jk[(axis, 0)];
                        coeff[12 * n + 1] = b[(axis, 1)] - jk[(axis, 1)];
                        add_residual(
                            &mut rows,
                            coeff,
                            a[axis] - target[axis],
                            cfg.landing_capture_weight,
                        );
                    }
                }
            }
        }
    }
    if cfg.capture_cop_weight > 0.0
        && cfg.omega > 0.0
        && let (Some(xi0), Some(pbar), Some(xiref)) = (
            input.dcm_now,
            input.reference_net_cop_schedule.as_ref(),
            input.reference_dcm_schedule.as_ref(),
        )
    {
        let wdt = cfg.omega * cfg.dt;
        let mut a = xi0;
        let mut bx = na::DVector::<f64>::zeros(nu);
        let mut by = na::DVector::<f64>::zeros(nu);
        for k in 0..n {
            let p = pbar.get(k).copied().unwrap_or_else(|| pbar[pbar.len() - 1]);
            // xi_{k+1} = (1 + w dt) xi_k - w dt (p_bar_k + delta_p_k(u)).
            a = a * (1.0 + wdt) - p * wdt;
            bx *= 1.0 + wdt;
            by *= 1.0 + wdt;
            if let Some((dx, dy)) = cop_maps[k].as_ref() {
                bx -= dx * wdt;
                by -= dy * wdt;
            }
            let target = xiref.get(k).copied().unwrap_or_else(|| xiref[xiref.len() - 1]);
            // The plan's own reference moves with the landing (J dp).
            let jk = input
                .landing
                .and(input.reference_dcm_sensitivity.as_ref())
                .and_then(|s| s.get(k).or(s.last()))
                .copied()
                .unwrap_or_else(na::Matrix2::zeros);
            for (axis, b) in [(0, &bx), (1, &by)] {
                let mut coeff = b.clone();
                if input.landing.is_some() {
                    coeff[12 * n] -= jk[(axis, 0)];
                    coeff[12 * n + 1] -= jk[(axis, 1)];
                }
                add_residual(&mut rows, coeff, a[axis] - target[axis], cfg.capture_cop_weight);
            }
        }
    }
    let mut objective = na::DMatrix::zeros(rows.len(), nu);
    let mut offset = na::DVector::zeros(rows.len());
    for (r, (a, b)) in rows.into_iter().enumerate() {
        objective.row_mut(r).copy_from(&a.transpose());
        offset[r] = b;
    }
    let h = objective.transpose() * &objective + na::DMatrix::identity(nu, nu) * 1e-10;
    let objective_constant = 0.5 * offset.norm_squared();
    let linear = objective.transpose() * offset;

    // Sole-frame friction, CoP, torsion, and normal-force constraints.
    let n_box = if input.landing.is_some() { 4 } else { 0 };
    let mut d = na::DMatrix::zeros(24 * n + n_box + 2 * n_reach, nu);
    let mut rhs = na::DVector::zeros(24 * n + n_box + 2 * n_reach);
    if n_reach > 0 {
        let caps = input.com_z_max.as_ref().unwrap();
        for k in 0..n {
            let r0 = 24 * n + n_box + 2 * k;
            let sidx = 12 * n + n_land + k;
            let (m, b) = &node_maps[k];
            // z_k - s_k <= cap_k   (z_k = m.row(2) U + b[2])
            d.row_mut(r0).copy_from(&m.row(2));
            d[(r0, sidx)] = -1.0;
            rhs[r0] = caps[k] - b[2];
            // -s_k <= 0
            d[(r0 + 1, sidx)] = -1.0;
            rhs[r0 + 1] = 0.0;
        }
    }
    if let Some(land) = input.landing {
        for j in 0..2 {
            // |prior + dp| <= max, written on dp.
            d[(24 * n + 2 * j, 12 * n + j)] = 1.0;
            // Signed bounds (see the lexicographic stage): a negative `min`
            // is a floor above zero. Kept feasible: the floor never passes
            // the ceiling.
            let ub = land.max[j] - land.prior[j];
            let lb = (-(land.min[j] + land.prior[j])).min(ub);
            rhs[24 * n + 2 * j] = ub;
            d[(24 * n + 2 * j + 1, 12 * n + j)] = -1.0;
            rhs[24 * n + 2 * j + 1] = -lb;

        }
    }
    let mut row = 0;
    for k in 0..n {
        for foot in 0..2 {
            let base = 12 * k + 6 * foot;
            let fz = base + 5;
            let bounds = [
                (base + 3, 1.0, cfg.friction_mu),
                (base + 3, -1.0, cfg.friction_mu),
                (base + 4, 1.0, cfg.friction_mu),
                (base + 4, -1.0, cfg.friction_mu),
                (base, 1.0, input.contacts[foot].cop_half.1),
                (base, -1.0, input.contacts[foot].cop_half.1),
                (base + 1, 1.0, input.contacts[foot].cop_half.0),
                (base + 1, -1.0, input.contacts[foot].cop_half.0),
                (base + 2, 1.0, cfg.torsion_mu),
                (base + 2, -1.0, cfg.torsion_mu),
            ];
            for &(idx, sign, scale) in &bounds {
                d[(row, idx)] = sign;
                d[(row, fz)] = -scale;
                row += 1;
            }
            let active = input.contact_schedule[k][foot];
            let activation = if active {
                input.activation_schedule[k][foot]
            } else {
                0.0
            };
            d[(row, fz)] = 1.0;
            rhs[row] = cfg.max_fz * activation;
            row += 1;
            d[(row, fz)] = -1.0;
            rhs[row] = if active {
                -input.min_fz_schedule[k][foot]
            } else {
                0.0
            };
            row += 1;
        }
    }

    let mut initial = na::DVector::zeros(nu);
    for k in 0..n {
        for foot in 0..2 {
            if input.contact_schedule[k][foot] {
                initial[12 * k + 6 * foot + 5] = input.reference_fz_schedule[k][foot]
                    .clamp(input.min_fz_schedule[k][foot], cfg.max_fz);
            }
        }
    }
    if n_reach > 0 {
        let caps = input.com_z_max.as_ref().unwrap();
        for k in 0..n {
            let (m, b) = &node_maps[k];
            let z = (m.row(2) * &initial)[0] + b[2];
            initial[12 * n + n_land + k] = (z - caps[k]).max(0.0);
        }
    }
    Some(DenseProblem {
        h,
        linear,
        d,
        rhs,
        initial,
        objective_constant,
        terminal_map: map,
        terminal_bias: bias,
        node_maps,
        a_eq,
        b_eq,
        cop_plan: cop_fixed,
        landing_node,
    })
}

/// State is invalidated when contact availability or constraint structure changes.
#[derive(Default)]
pub struct Workspace {
    pub qp: QpWorkspace,
    signature: Vec<u64>,
    last_report: Option<SolveReport>,
}
/// Diagnostic snapshot for the most recent valid assembled MPC problem.
#[derive(Clone, Debug)]
pub struct SolveReport {
    pub status: QpStatus,
    pub iterations: usize,
    pub native: Option<misa_wbc::qp::QpDiagnostics>,
    pub residuals: Residuals,
    pub assembly_ns: u128,
    pub solver_call_ns: u128,
    pub validation_ns: u128,
}

impl Workspace {
    pub fn clear(&mut self) {
        self.qp.clear();
        self.signature.clear();
        self.last_report = None;
    }
    pub fn last_report(&self) -> Option<&SolveReport> {
        self.last_report.as_ref()
    }
    pub fn prepare(&mut self, input: &Input, problem: &DenseProblem) {
        let mut signature = vec![
            problem.h.nrows() as u64,
            problem.d.nrows() as u64,
            problem.a_eq.as_ref().map_or(0, |a| a.nrows()) as u64,
        ];
        signature.extend(
            input
                .contact_schedule
                .iter()
                .flat_map(|v| v.iter().map(|b| *b as u64)),
        );
        signature.extend(
            input
                .activation_schedule
                .iter()
                .flat_map(|v| v.iter().map(|a| (*a > 0.0) as u64)),
        );
        signature.extend(problem.d.iter().map(|v| (*v != 0.0) as u64));
        if signature != self.signature {
            self.qp.clear();
            self.signature = signature;
        }
    }
}

pub fn solve_dense(
    problem: &DenseProblem,
    cfg: &QpConfig,
    workspace: Option<&mut QpWorkspace>,
) -> QpSolution {
    let mut cfg = cfg.clone();
    cfg.prox_weight = 0.0;
    // The warm start is built from the vertical-force schedule and satisfies
    // no equality we might add. The active-set backend answers `Infeasible`
    // for a feasible problem when `x0` violates an equality (misa-wbc
    // `equality_qp_is_solved_even_from_a_violating_warm_start`), so drop it
    // when there are equality rows.
    let initial = problem.a_eq.is_none().then_some(&problem.initial);
    match workspace {
        Some(ws) => solve_qp_warm(
            &problem.h,
            &problem.linear,
            problem.a_eq.as_ref(),
            problem.b_eq.as_ref(),
            Some(&problem.d),
            Some(&problem.rhs),
            initial,
            &cfg,
            ws,
        ),
        None => solve_qp(
            &problem.h,
            &problem.linear,
            problem.a_eq.as_ref(),
            problem.b_eq.as_ref(),
            Some(&problem.d),
            Some(&problem.rhs),
            initial,
            &cfg,
        ),
    }
}

/// Solve the linearised SRBD problem and return the first horizon command.
pub fn solve(input: &Input, cfg: &Config) -> Option<Solution> {
    solve_with_workspace(input, cfg, None)
}

pub fn solve_with_workspace(
    input: &Input,
    cfg: &Config,
    workspace: Option<&mut Workspace>,
) -> Option<Solution> {
    solve_traced(input, cfg, workspace, &mut Vec::new())
}

/// [`solve_with_workspace`] with the traces [`Config::trace_landing_time`]
/// and [`Config::trace_failures`] ask for appended to `trace`, one line each
/// (the caller prints them, or not).
pub fn solve_traced(
    input: &Input,
    cfg: &Config,
    mut workspace: Option<&mut Workspace>,
    trace: &mut Vec<String>,
) -> Option<Solution> {
    if let Some(ws) = workspace.as_deref_mut() {
        ws.last_report = None;
    }
    let assembly_started = std::time::Instant::now();
    // Lexicographic CoP: a plan that the wrench stage cannot meet exactly
    // (a corner of the reachable box, a contact the schedule and the robot
    // disagree about) must not cost the tick. Retry once with the net-CoP
    // rows soft.
    if cfg.capture_cop_lexicographic && cfg.capture_cop_net_eq {
        let strict = solve_with_workspace_inner(input, cfg, workspace.as_deref_mut(), assembly_started, trace);
        if strict.is_some() {
            return strict;
        }
        if cfg.trace_failures {
            let dp = lexicographic_landing(input, cfg);
            let plan = lexicographic_cop_with_landing(input, cfg, dp);
            trace.push(format!(
                "    [lex-diag] landing={:?} dp={:?} pos0=({:.3},{:.3}) pos1=({:.3},{:.3}) cop_half0={:?} cop_half1={:?}",
                input.landing.map(|l| (l.foot, l.first_node, l.prior, l.max, l.min)),
                dp.map(|d| (d.x, d.y)),
                input.contacts[0].position.x, input.contacts[0].position.y,
                input.contacts[1].position.x, input.contacts[1].position.y,
                input.contacts[0].cop_half, input.contacts[1].cop_half
            ));
            if let Some(plan) = plan {
                trace.push(format!("    [lex-diag] plan: {}", plan.iter().map(|p| format!("({:.3},{:.3})", p.x, p.y)).collect::<Vec<_>>().join(" ")));
            }
            // Which equality group breaks the strict problem?
            for (label, net, total, lex_land) in [
                ("net_eq off", false, true, true),
                ("total_eq off", true, false, true),
                ("both off", false, false, true),
                ("landing free", true, true, false),
                ("all off", false, false, false),
            ] {
                let mut c = cfg.clone();
                c.capture_cop_net_eq = net;
                c.capture_cop_total_eq = total;
                c.landing_lexicographic = lex_land;
                let st = assemble_traced(input, &c, trace).map(|p| solve_dense(&p, &c.qp, None).status);
                trace.push(format!("    [lex-diag] {label}: {st:?}"));
            }
        }
        let mut soft = cfg.clone();
        soft.capture_cop_net_eq = false;
        if soft.net_cop_weight <= 0.0 {
            soft.net_cop_weight = 10.0;
        }
        return solve_with_workspace_inner(input, &soft, workspace, std::time::Instant::now(), trace)
            .map(|mut s| {
                s.soft_fallback = true;
                s
            });
    }
    solve_with_workspace_inner(input, cfg, workspace, assembly_started, trace)
}

fn solve_with_workspace_inner(
    input: &Input,
    cfg: &Config,
    mut workspace: Option<&mut Workspace>,
    assembly_started: std::time::Instant,
    trace: &mut Vec<String>,
) -> Option<Solution> {
    let problem = assemble_traced(input, cfg, trace)?;
    #[cfg(feature = "qp-hpipm")]
    let use_ocp = cfg.qp.solver == misa_wbc::qp::QpSolver::Hpipm;
    #[cfg(not(feature = "qp-hpipm"))]
    let use_ocp = false;
    #[cfg(feature = "qp-hpipm")]
    let ocp = if use_ocp {
        Some(assemble_ocp(input, cfg)?)
    } else {
        None
    };
    let assembly_ns = assembly_started.elapsed().as_nanos();
    let solve_started = std::time::Instant::now();
    let result = if use_ocp {
        #[cfg(feature = "qp-hpipm")]
        {
            misa_wbc::ocp::solve_ocp(ocp.as_ref().unwrap(), &cfg.qp)
        }
        #[cfg(not(feature = "qp-hpipm"))]
        {
            unreachable!()
        }
    } else if let Some(ws) = workspace.as_deref_mut() {
        ws.prepare(input, &problem);
        solve_dense(&problem, &cfg.qp, Some(&mut ws.qp))
    } else {
        solve_dense(&problem, &cfg.qp, None)
    };
    let solver_call_ns = solve_started.elapsed().as_nanos();
    // A finite Optimal status alone is insufficient for newly selected backends.
    let validation_started = std::time::Instant::now();
    let residual = residuals(&problem, &result);
    let validation_ns = validation_started.elapsed().as_nanos();
    if let Some(ws) = workspace {
        ws.last_report = Some(SolveReport {
            status: result.status,
            iterations: result.iterations,
            native: result.diagnostics.clone(),
            residuals: residual.clone(),
            assembly_ns,
            solver_call_ns,
            validation_ns,
        });
    }
    if (cfg.validate_solution || cfg.qp.solver != misa_wbc::qp::QpSolver::ActiveSet)
        && (!result.formally_solved()
            || !residual.accepted_with(cfg.kkt_tolerance, cfg.kkt_complementarity_tolerance))
    {
        if cfg.trace_failures {
            trace.push(format!(
                "  [mpc-gate] status={:?} formally={} iters={} feas={:.2e}/{:.2e} stat={:.2e}/{:.2e} comp={:.2e} dual={:.2e} eq_rows={} lambda_eq={}",
                result.status, result.formally_solved(), result.iterations,
                residual.feasibility, residual.normalized_feasibility,
                residual.stationarity, residual.normalized_stationarity,
                residual.complementarity, residual.dual_violation,
                problem.a_eq.as_ref().map_or(0, |a| a.nrows()), result.lambda_eq.len()
            ));
        }
        return None;
    }
    let map = &problem.terminal_map;
    let bias = &problem.terminal_bias;
    let gravity = na::Vector3::new(0.0, 0.0, -cfg.gravity);
    if result.status != QpStatus::Optimal || result.x.iter().any(|x| !x.is_finite()) {
        if cfg.trace_failures {
            trace.push(format!(
                "  [mpc-fail] status={:?} eq_rows={} cop_plan={:?}",
                result.status,
                problem.a_eq.as_ref().map_or(0, |a| a.nrows()),
                problem.cop_plan.as_ref().map(|p| (p[0].x, p[0].y)),
            ));
            for k in 0..cfg.horizon.min(3) {
                trace.push(format!(
                    "    k{k} contact={:?} act={:?} min={:?} ref={:?} pos0=({:.3},{:.3}) pos1=({:.3},{:.3}) cop_half={:?}/{:?}",
                    input.contact_schedule[k], input.activation_schedule[k], input.min_fz_schedule[k],
                    input.reference_fz_schedule[k], input.contacts[0].position.x, input.contacts[0].position.y,
                    input.contacts[1].position.x, input.contacts[1].position.y,
                    input.contacts[0].cop_half, input.contacts[1].cop_half
                ));
            }
        }
        return None;
    }
    let wrenches = [
        Wrench::from_column_slice(&result.x.as_slice()[0..6]),
        Wrench::from_column_slice(&result.x.as_slice()[6..12]),
    ];
    let net_force = input.contacts[0].rotation * wrenches[0].fixed_rows::<3>(3)
        + input.contacts[1].rotation * wrenches[1].fixed_rows::<3>(3);
    let states: Vec<na::DVector<f64>> =
        problem.node_maps.iter().map(|(m, b)| m * &result.x + b).collect();
    let trajectory = states
        .iter()
        .map(|st| {
            [
                na::Vector3::new(st[0], st[1], st[2]),
                na::Vector3::new(st[3], st[4], st[5]),
            ]
        })
        .collect();
    let momentum_trajectory = states
        .iter()
        .map(|st| na::Vector3::new(st[6], st[7], st[8]))
        .collect();
    let landing_shift = input.landing.map(|_| {
        let n = cfg.horizon;
        na::Vector2::new(result.x[12 * n], result.x[12 * n + 1])
    });
    Some(Solution {
        sole_wrenches: wrenches,
        acceleration: net_force / cfg.mass + gravity,
        trajectory,
        landing_shift,
        cop_plan: problem.cop_plan.clone(),
        soft_fallback: false,
        landing_node: problem.landing_node,
        momentum_trajectory,
        terminal_com: na::Vector3::new(bias[0], bias[1], bias[2])
            + na::Vector3::new(
                (&map.row(0) * &result.x)[0],
                (&map.row(1) * &result.x)[0],
                (&map.row(2) * &result.x)[0],
            ),
        terminal_momentum: na::Vector3::new(bias[6], bias[7], bias[8])
            + na::Vector3::new(
                (&map.row(6) * &result.x)[0],
                (&map.row(7) * &result.x)[0],
                (&map.row(8) * &result.x)[0],
            ),
    })
}

/// Independent residuals in physical units and row/term-relative normalization.
#[derive(Clone, Debug)]
pub struct Residuals {
    pub feasibility: f64,
    pub normalized_feasibility: f64,
    pub stationarity: f64,
    pub normalized_stationarity: f64,
    pub complementarity: f64,
    pub dual_violation: f64,
}
impl Residuals {
    pub fn accepted(&self, tolerance: f64) -> bool {
        self.accepted_with(tolerance, tolerance)
    }
    pub fn accepted_with(&self, tolerance: f64, complementarity_tolerance: f64) -> bool {
        [self.normalized_feasibility, self.normalized_stationarity, self.dual_violation]
            .iter()
            .all(|v| v.is_finite() && *v <= tolerance)
            && self.complementarity.is_finite()
            && self.complementarity <= complementarity_tolerance
    }
}
pub fn residuals(p: &DenseProblem, result: &QpSolution) -> Residuals {
    // The equality rows must be part of the KKT check: without them the
    // stationarity residual carries the whole equality multiplier and every
    // solution with an equality row is rejected (2026-10-02: every Clarabel
    // solve of the lexicographic CoP, and of v5.2's own lexicographic
    // landing on any non-active-set backend).
    let eq = p.a_eq.as_ref().zip(p.b_eq.as_ref());
    let r = misa_wbc::qp_check::check(&p.h, &p.linear, eq, Some((&p.d, &p.rhs)), result);
    let invalid = if r.finite { 0.0 } else { f64::INFINITY };
    Residuals {
        feasibility: r.primal_raw.max(invalid),
        normalized_feasibility: r.primal_scaled.max(invalid),
        stationarity: r.stationarity_raw.max(invalid),
        normalized_stationarity: r.stationarity_scaled.max(invalid),
        complementarity: r.complementarity_scaled.max(invalid),
        dual_violation: r.dual_sign.max(invalid),
    }
}

/// Equivalent OCP with 21 states: physical SRBD state plus previous 12 wrenches.
/// Augmentation expresses the wrench-rate cost without cross-stage couplings.
pub fn assemble_ocp(input: &Input, cfg: &Config) -> Option<misa_wbc::ocp::OcpProblem> {
    use misa_wbc::ocp::{OcpProblem, OcpStage};
    // The stage-wise form has no column for the height-cap slacks.
    if input.com_z_max.is_some() {
        return None;
    }
    let dense = assemble(input, cfg)?;
    let mut initial = na::DVector::zeros(21);
    initial.rows_mut(0, 3).copy_from(&input.com);
    initial.rows_mut(3, 3).copy_from(&input.velocity);
    initial.rows_mut(6, 3).copy_from(&input.angular_momentum);
    for foot in 0..2 {
        initial
            .rows_mut(9 + 6 * foot, 6)
            .copy_from(&input.previous[foot]);
    }
    let mut a = na::DMatrix::zeros(21, 21);
    a.view_mut((0, 0), (9, 9))
        .copy_from(&na::DMatrix::identity(9, 9));
    for j in 0..3 {
        a[(j, 3 + j)] = cfg.dt;
    }
    let mut b = na::DMatrix::zeros(21, 12);
    b.view_mut((9, 0), (12, 12))
        .copy_from(&na::DMatrix::identity(12, 12));
    let mut bias = na::DVector::zeros(21);
    bias[2] = -0.5 * cfg.dt * cfg.dt * cfg.gravity;
    bias[5] = -cfg.dt * cfg.gravity;
    for foot in 0..2 {
        let c = &input.contacts[foot];
        let torque = skew(&(c.position - input.target_com)) * c.rotation;
        for i in 0..3 {
            for j in 0..3 {
                b[(i, 6 * foot + 3 + j)] = 0.5 * cfg.dt * cfg.dt * c.rotation[(i, j)] / cfg.mass;
                b[(3 + i, 6 * foot + 3 + j)] = cfg.dt * c.rotation[(i, j)] / cfg.mass;
                b[(6 + i, 6 * foot + j)] = cfg.dt * c.rotation[(i, j)];
                b[(6 + i, 6 * foot + 3 + j)] = cfg.dt * torque[(i, j)];
            }
        }
    }
    let mut stages = Vec::new();
    for k in 0..cfg.horizon {
        let mut lx = na::DMatrix::zeros(36, 21);
        let mut lu = na::DMatrix::zeros(36, 12);
        let mut offset = na::DVector::zeros(36);
        let terminal = if k + 1 == cfg.horizon { 4.0 } else { 1.0 };
        let frac = (k + 1) as f64 / cfg.horizon as f64;
        let rc = input.com + (input.target_com - input.com) * frac;
        let rv = input.velocity + (input.target_velocity - input.velocity) * frac;
        for axis in 0..9 {
            let weight = (terminal
                * match axis {
                    0..=2 => cfg.position_weight,
                    3..=5 => cfg.velocity_weight,
                    _ => cfg.momentum_weight,
                })
            .sqrt();
            lx.row_mut(axis).copy_from(&(a.row(axis) * weight));
            lu.row_mut(axis).copy_from(&(b.row(axis) * weight));
            offset[axis] = weight
                * (bias[axis]
                    - match axis {
                        0..=2 => rc[axis],
                        3..=5 => rv[axis - 3],
                        _ => 0.0,
                    });
        }
        for axis in 0..3 {
            for foot in 0..2 {
                for j in 0..3 {
                    lu[(9 + axis, 6 * foot + 3 + j)] = cfg.acceleration_weight.sqrt()
                        * input.contacts[foot].rotation[(axis, j)]
                        / cfg.mass;
                }
            }
            offset[9 + axis] = cfg.acceleration_weight.sqrt()
                * ((if axis == 2 { -cfg.gravity } else { 0.0 }) - input.target_acceleration[axis]);
        }
        for j in 0..12 {
            lu[(12 + j, j)] = cfg.wrench_weight.sqrt();
            let foot = j / 6;
            let reference = if j % 6 == 5 && input.contact_schedule[k][foot] {
                input.reference_fz_schedule[k][foot]
            } else {
                0.0
            };
            offset[12 + j] = -cfg.wrench_weight.sqrt() * reference;
            lu[(24 + j, j)] = cfg.smooth_weight.sqrt();
            lx[(24 + j, 9 + j)] = -cfg.smooth_weight.sqrt();
        }
        stages.push(OcpStage {
            a: a.clone(),
            b: b.clone(),
            bias: bias.clone(),
            q_matrix: lx.transpose() * &lx,
            r_matrix: lu.transpose() * &lu + na::DMatrix::identity(12, 12) * 1e-10,
            s_matrix: lu.transpose() * &lx,
            q: lx.transpose() * &offset,
            r: lu.transpose() * &offset,
            c: na::DMatrix::zeros(24, 21),
            d: dense.d.view((24 * k, 12 * k), (24, 12)).into_owned(),
            lower: na::DVector::from_element(24, f64::NEG_INFINITY),
            upper: dense.rhs.rows(24 * k, 24).into_owned(),
            constant: 0.5 * offset.norm_squared(),
        });
    }
    Some(OcpProblem { initial, stages })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CoP term has to actually move the pressure, and do nothing when
    /// the reference is where the solution already puts it.
    #[test]
    fn the_cop_reference_moves_the_pressure() {
        let (input, cfg) = standing([0.0, 0.0]);
        let plain = solve(&input, &cfg).expect("baseline solves");
        // Ask for the pressure 3 cm forward on the LEFT sole only.
        let want = na::Vector2::new(0.03, 0.0);
        let mut asked = input.clone();
        asked.reference_cop_schedule = Some(vec![[want, na::Vector2::zeros()]; cfg.horizon]);
        let cfg2 = Config { cop_weight: 1.0, ..cfg.clone() };
        let moved = solve(&asked, &cfg2).expect("with a CoP reference it still solves");
        // CoP on a sole is (-m_y / f_z, m_x / f_z).
        let cop_x = |w: &Wrench| -w[1] / w[5].max(1e-9);
        assert!(
            cop_x(&moved.sole_wrenches[0]) > cop_x(&plain.sole_wrenches[0]) + 0.005,
            "left CoP {} -> {}",
            cop_x(&plain.sole_wrenches[0]),
            cop_x(&moved.sole_wrenches[0])
        );
        // The right foot was asked for its centre and should not run away.
        assert!(cop_x(&moved.sole_wrenches[1]).abs() < 0.02);
    }

    /// Net CoP where the physics allows it.
    ///
    /// Asking for a CoP offset while also asking the CoM to stay still is a
    /// contradiction -- the pressure point IS what accelerates the CoM -- and
    /// the first version of this test did exactly that, then read the
    /// resulting infeasibility as a bug in the rows. The lateral acceleration
    /// that a CoP offset `d` implies is `g d / z`, so the target has to move
    /// with it.
    fn leaning(offset_y: f64) -> (Input, Config) {
        let (mut input, cfg) = standing([0.0, 0.0]);
        let z = input.com.z;
        let a = cfg.gravity * offset_y / z;
        input.target_acceleration = na::Vector3::new(0.0, a, 0.0);
        input.reference_net_cop_schedule =
            Some(vec![na::Vector2::new(0.0, offset_y); cfg.horizon]);
        (input, cfg)
    }

    fn net_cop_y(s: &Solution) -> f64 {
        let (wl, wr) = (&s.sole_wrenches[0], &s.sole_wrenches[1]);
        let (fl, fr) = (wl[5], wr[5]);
        ((wl[0] + 0.08 * fl) + (wr[0] - 0.08 * fr)) / (fl + fr).max(1e-9)
    }

    /// The net CoP term must move the force SPLIT, which is the thing the
    /// per-sole version could not do.
    #[test]
    fn the_net_cop_reference_moves_the_split() {
        let (input, cfg) = leaning(0.04);
        let plain = solve(&input, &cfg).expect("baseline solves");
        let cfg2 = Config { net_cop_weight: 10.0, ..cfg.clone() };
        let moved = solve(&input, &cfg2).expect("solves");
        assert!(
            net_cop_y(&moved) > net_cop_y(&plain) + 0.005,
            "net CoP {:.4} -> {:.4}",
            net_cop_y(&plain),
            net_cop_y(&moved)
        );
    }

    /// As an EQUALITY the net CoP is met, not merely preferred.
    #[test]
    fn the_net_cop_equality_is_met() {
        let (input, cfg) = leaning(0.015);
        let soft = Config { net_cop_weight: 10.0, ..cfg.clone() };
        let hard = Config { net_cop_equality: true, ..cfg.clone() };
        let a = solve(&input, &soft).expect("soft solves");
        let b = solve(&input, &hard).expect("hard solves");
        assert!(
            (net_cop_y(&b) - 0.015).abs() < (net_cop_y(&a) - 0.015).abs(),
            "soft {:.4} hard {:.4}",
            net_cop_y(&a),
            net_cop_y(&b)
        );
        assert!((net_cop_y(&b) - 0.015).abs() < 2e-3, "hard {:.4}", net_cop_y(&b));
    }

    /// The equality must not appear in single support, where it could only
    /// make the problem infeasible.
    #[test]
    fn the_equality_is_skipped_in_single_support() {
        let (mut input, cfg) = standing([0.0, 0.0]);
        input.contact_schedule = vec![[true, false]; cfg.horizon];
        input.reference_fz_schedule = vec![[98.1, 0.0]; cfg.horizon];
        input.reference_net_cop_schedule =
            Some(vec![na::Vector2::new(0.0, 0.5); cfg.horizon]);
        let hard = Config { net_cop_equality: true, ..cfg.clone() };
        let problem = assemble(&input, &hard).expect("assembles");
        assert!(problem.a_eq.is_none(), "no equality rows in single support");
        assert!(solve(&input, &hard).is_some(), "and it still solves");
    }

    /// The net CoP rows must be right for a YAWED foot, which is where the
    /// frame mistake shows: with zero yaw a local moment and a world moment
    /// coincide and a wrong row still looks plausible.
    #[test]
    fn the_net_cop_rows_are_in_world_for_a_yawed_foot() {
        let (mut input, cfg) = standing([0.0, 0.0]);
        let yaw = 0.6_f64;
        let (sy, cy) = yaw.sin_cos();
        let rot = na::Matrix3::new(cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0);
        input.contacts[0].rotation = rot;
        input.contacts[1].rotation = rot;
        input.reference_net_cop_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        let hard = Config { net_cop_equality: true, ..cfg.clone() };
        let pr = assemble(&input, &hard).expect("assembles");
        let a = pr.a_eq.as_ref().expect("equality rows");
        // Build a wrench by hand whose WORLD net moment about the origin is
        // zero, and check the rows agree: equal vertical loads at +-0.08 with
        // local moments that cancel the lever arms.
        let mut x = na::DVector::zeros(pr.h.ncols());
        for (foot, sign) in [(0usize, 1.0_f64), (1, -1.0)] {
            let b = 6 * foot;
            let fz = 49.05;
            x[b + 5] = fz;
            // world moment from the lever arm is (p x f)_x = p_y * fz
            // so the local moment must produce -p_y*fz in world x:
            // R m = (-p_y fz, 0, 0)  =>  m = R^T (-p_y fz, 0, 0)
            let want = na::Vector3::new(-sign * 0.08 * fz, 0.0, 0.0);
            let m = rot.transpose() * want;
            for j in 0..3 {
                x[b + j] = m[j];
            }
        }
        let r = a * &x;
        assert!(r[0].abs() < 1e-9 && r[1].abs() < 1e-9, "rows give {:?}", (r[0], r[1]));
    }

    #[test]
    fn a_zero_net_cop_weight_changes_nothing() {
        let (input, cfg) = standing([0.0, 0.0]);
        let plain = solve(&input, &cfg).expect("solves");
        let mut asked = input.clone();
        asked.reference_net_cop_schedule =
            Some(vec![na::Vector2::new(0.05, 0.05); cfg.horizon]);
        let same = solve(&asked, &cfg).expect("solves");
        for foot in 0..2 {
            for j in 0..6 {
                assert_eq!(plain.sole_wrenches[foot][j], same.sole_wrenches[foot][j]);
            }
        }
    }

    /// Weight 0 must reproduce the shipped problem exactly, whatever the
    /// schedule says.
    #[test]
    fn a_zero_cop_weight_changes_nothing() {
        let (input, cfg) = standing([0.0, 0.0]);
        let plain = solve(&input, &cfg).expect("solves");
        let mut asked = input.clone();
        asked.reference_cop_schedule =
            Some(vec![[na::Vector2::new(0.04, 0.02); 2]; cfg.horizon]);
        let same = solve(&asked, &cfg).expect("solves");
        for foot in 0..2 {
            for j in 0..6 {
                assert_eq!(plain.sole_wrenches[foot][j], same.sole_wrenches[foot][j]);
            }
        }
    }

    fn standing(min: [f64; 2]) -> (Input, Config) {
        let cfg = Config {
            mass: 10.0,
            max_fz: 150.0,
            horizon: 5,
            ..Config::default()
        };
        let contact = |y: f64, _i: usize| Contact {
            position: na::Vector3::new(0.0, y, 0.0),
            rotation: na::Matrix3::identity(),
            cop_half: cfg.cop_half,
        };
        let com = na::Vector3::new(0.0, 0.0, 0.6);
        (
            Input {
                com,
                velocity: na::Vector3::zeros(),
                angular_momentum: na::Vector3::zeros(),
                target_com: com,
                target_velocity: na::Vector3::zeros(),
                target_acceleration: na::Vector3::zeros(),
                contacts: [contact(0.08, 0), contact(-0.08, 1)],
                contact_schedule: vec![[true; 2]; cfg.horizon],
                activation_schedule: vec![[1.0; 2]; cfg.horizon],
                reference_fz_schedule: vec![[49.05; 2]; cfg.horizon],
                min_fz_schedule: vec![min; cfg.horizon],
                reference_cop_schedule: None,
                reference_net_cop_schedule: None,
                reference_dcm_schedule: None,
                previous: [Wrench::zeros(), Wrench::zeros()],
                landing: None,
                prior_com: None,
                dcm_now: None,
                reference_dcm_sensitivity: None,
                com_z_max: None,
            },
            cfg,
        )
    }

    #[test]
    fn a_height_cap_lowers_the_predicted_com_and_a_loose_one_changes_nothing() {
        let (input, cfg) = standing([0.0, 0.0]);
        let plain = solve(&input, &cfg).expect("solves");
        let mut loose = input.clone();
        loose.com_z_max = Some(vec![10.0; cfg.horizon]);
        let same = solve(&loose, &cfg).expect("solves");
        for k in 0..cfg.horizon {
            assert!((plain.trajectory[k][0].z - same.trajectory[k][0].z).abs() < 1e-6);
        }
        let cap = 0.58;
        let mut capped = input.clone();
        capped.com_z_max = Some(vec![cap; cfg.horizon]);
        let low = solve(&capped, &cfg).expect("solves");
        let last = cfg.horizon - 1;
        assert!(
            low.trajectory[last][0].z < plain.trajectory[last][0].z - 0.005,
            "{} vs {}",
            low.trajectory[last][0].z,
            plain.trajectory[last][0].z
        );
        assert!(low.trajectory[last][0].z <= cap + 0.002, "{}", low.trajectory[last][0].z);
    }

    /// The stabilising pressure point for a LIPM must lie BEYOND the DCM,
    /// because xi_dot = omega (xi - p): only p > xi brings a forward-running
    /// xi back. The expected value here comes from that identity, not from
    /// the assembled rows -- the last time a test here was written the same
    /// way as the code, it passed while both were wrong.
    #[test]
    fn the_dcm_term_puts_the_pressure_beyond_the_divergent_component() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        let omega = (cfg.gravity / input.com.z).sqrt();
        // Running forward with the CoM still at the origin: xi = v/omega.
        input.velocity.x = 0.2;
        let xi = input.velocity.x / omega;
        // Ask to come to rest at the origin, and give the CoM tracking terms
        // nothing to argue with by leaving the targets where the CoM is.
        input.reference_dcm_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);

        let without = solve(&input, &cfg).unwrap();
        cfg.dcm_weight = 200.0;
        cfg.omega = omega;
        let with = solve(&input, &cfg).unwrap();

        let net_cop_x = |w: &Solution| {
            let (mut m, mut f) = (0.0, 0.0);
            for foot in 0..2 {
                let c = &input.contacts[foot];
                let wr = w.sole_wrenches[foot];
                let moment = c.rotation * na::Vector3::new(wr[0], wr[1], wr[2]);
                let force = c.rotation * na::Vector3::new(wr[3], wr[4], wr[5]);
                m += moment.y + (c.position.x - input.com.x) * force.z;
                f += force.z;
            }
            // CoP_x = -M_y/F_z about the CoM, plus the CoM's own x.
            input.com.x - m / f
        };
        cfg.dcm_weight = 2000.0;
        let harder = solve(&input, &cfg).unwrap();

        let p_without = net_cop_x(&without);
        let p_with = net_cop_x(&with);
        let p_harder = net_cop_x(&harder);
        assert!(
            p_with > p_without,
            "the DCM term must move the pressure forward ({p_without:.4} -> {p_with:.4})"
        );
        assert!(
            p_harder > p_with,
            "more weight must push further ({p_with:.4} -> {p_harder:.4})"
        );
        // The shipped cost barely reacts: 2.7 mm of pressure against a 49 mm
        // divergence. That is the gap this term exists to close, and it is
        // why the MPC and the WBC -- which runs a real DCM law -- could not
        // be made to agree on the double-support split by any reweighting.
        assert!(
            p_without < 0.2 * xi && p_harder > 0.8 * xi,
            "xi={xi:.4} without={p_without:.4} with={p_with:.4} harder={p_harder:.4}"
        );
    }

    /// The trajectory's last node must be the terminal state the solver
    /// already reports, and a robot at rest asked to stay at rest must be
    /// predicted to stay put. Expected values come from the inputs, not from
    /// the node bookkeeping being tested.
    #[test]
    fn the_trajectory_ends_at_the_terminal_state_and_a_standing_robot_stays_put() {
        let (input, cfg) = standing([0.0, 0.0]);
        let sol = solve(&input, &cfg).unwrap();
        assert_eq!(sol.trajectory.len(), cfg.horizon);
        let last = sol.trajectory[cfg.horizon - 1][0];
        assert!((last - sol.terminal_com).norm() < 1e-9, "{last:?} vs {:?}", sol.terminal_com);
        for (k, [c, v]) in sol.trajectory.iter().enumerate() {
            assert!((c - input.com).norm() < 0.01, "node {k}: c={c:?} drifted from {:?}", input.com);
            assert!(v.norm() < 0.05, "node {k}: v={v:?} should be ~0 at rest");
        }
    }

    /// With the landing boxed to zero the problem must reproduce the
    /// fixed-contact solution exactly: the extra variable cannot move, and
    /// its rows contribute nothing.
    #[test]
    fn a_landing_boxed_to_zero_reproduces_the_fixed_contact_solution() {
        let (mut input, cfg) = standing([0.0, 0.0]);
        input.target_com.x += 0.05;
        let fixed = solve(&input, &cfg).unwrap();
        input.landing = Some(Landing { foot: 1, first_node: 0, weight: 1.0, max: [0.0, 0.0], prior: [0.0; 2], min: [0.0, 0.0], min_first_node: 0, position: [0.0; 2], terrain_x: None });
        let boxed = solve(&input, &cfg).unwrap();
        let dp = boxed.landing_shift.unwrap();
        assert!(dp.norm() < 1e-9, "{dp:?}");
        for foot in 0..2 {
            for j in 0..6 {
                assert!(
                    (fixed.sole_wrenches[foot][j] - boxed.sole_wrenches[foot][j]).abs() < 1e-6,
                    "foot {foot} comp {j}: {} vs {}", fixed.sole_wrenches[foot][j], boxed.sole_wrenches[foot][j]
                );
            }
        }
    }

    /// With the moment coupling off and no capture rows, the landing has
    /// only its own cost: it must stay at zero and the wrenches must equal
    /// the fixed-contact solution even with a wide box. (With the coupling
    /// on, the same box lets the momentum cost move it -- see the next test.)
    #[test]
    fn without_moment_coupling_the_landing_only_moves_for_capture() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        input.target_velocity.x = 0.3;
        let fixed = solve(&input, &cfg).unwrap();
        input.landing = Some(Landing { foot: 1, first_node: 0, weight: 1.0, max: [0.10, 0.05], prior: [0.0; 2], min: [0.10, 0.05], min_first_node: 0, position: [0.0; 2], terrain_x: None });
        cfg.landing_moment_coupling = false;
        let free = solve(&input, &cfg).unwrap();
        let dp = free.landing_shift.unwrap();
        assert!(dp.norm() < 1e-7, "decoupled landing moved: {dp:?}");
        for foot in 0..2 {
            for j in 0..6 {
                assert!(
                    (fixed.sole_wrenches[foot][j] - free.sole_wrenches[foot][j]).abs() < 1e-5,
                    "foot {foot} comp {j}: {} vs {}", fixed.sole_wrenches[foot][j], free.sole_wrenches[foot][j]
                );
            }
        }
        cfg.landing_moment_coupling = true;
        let coupled = solve(&input, &cfg).unwrap();
        assert!(coupled.landing_shift.unwrap().norm() > 1e-3, "coupling on must move it");
    }

    /// Asking for forward VELOCITY (target position unchanged, so the frozen
    /// moment arm about the target has no x component) needs f_x > 0 at
    /// ground level, which is a negative pitch angular momentum; the
    /// momentum cost then wants the free foot BEHIND (dp.x < 0), because
    /// dp x F_k with F_k upward gives -dp.x * fz in y. The expected sign is
    /// derived from the LIPM here, not read off the rows. (A first draft put
    /// the target 10 cm ahead as well; the arm term (p - c) x F then
    /// dominates and the correct answer is the OPPOSITE sign -- the test was
    /// wrong, not the code.)
    #[test]
    fn a_forward_target_moves_the_free_landing_back() {
        let (mut input, cfg) = standing([0.0, 0.0]);
        input.target_velocity.x = 0.3;
        input.landing = Some(Landing { foot: 1, first_node: 0, weight: 1.0, max: [0.06, 0.03], prior: [0.0; 2], min: [0.06, 0.03], min_first_node: 0, position: [0.0; 2], terrain_x: None });
        let sol = solve(&input, &cfg).unwrap();
        let dp = sol.landing_shift.unwrap();
        assert!(dp.x < -1e-3, "expected the free foot to move back, got {dp:?}");
        assert!(dp.x.abs() <= 0.06 + 1e-9 && dp.y.abs() <= 0.03 + 1e-9, "box violated: {dp:?}");
    }

    /// A prior equal to the target at every node must reproduce the frozen
    /// arm exactly, and a different prior must change the moment rows.
    #[test]
    fn an_arm_about_a_prior_equal_to_the_target_is_the_shipped_arm() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        input.target_com.x += 0.03;
        let fixed = solve(&input, &cfg).unwrap();
        cfg.arm_about_prior = true;
        input.prior_com = Some(vec![input.target_com; cfg.horizon]);
        let same = solve(&input, &cfg).unwrap();
        for foot in 0..2 {
            for j in 0..6 {
                assert_eq!(fixed.sole_wrenches[foot][j], same.sole_wrenches[foot][j]);
            }
        }
        input.prior_com = Some(vec![input.target_com + na::Vector3::new(0.05, 0.0, 0.0); cfg.horizon]);
        let moved = solve(&input, &cfg).unwrap();
        let diff: f64 = (0..2)
            .map(|f| (moved.sole_wrenches[f] - same.sole_wrenches[f]).norm())
            .sum();
        assert!(diff > 1e-6, "a moved prior must move the solution");
    }

    /// Penalising L-dot must make the predicted angular momentum change
    /// less from node to node than the shipped cost does, on a robot asked
    /// to pick up forward speed (which needs a pitch moment to be traded
    /// off). The expected direction comes from what the term penalises.
    #[test]
    fn the_ldot_weight_smooths_the_predicted_angular_momentum() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        input.target_velocity.x = 0.3;
        cfg.momentum_weight = 0.0;
        let free = solve(&input, &cfg).unwrap();
        cfg.ldot_weight = 1e4;
        let damped = solve(&input, &cfg).unwrap();
        let jump = |m: &Vec<na::Vector3<f64>>| {
            let mut worst = (m[0] - input.angular_momentum).norm();
            for k in 1..m.len() {
                worst = worst.max((m[k] - m[k - 1]).norm());
            }
            worst
        };
        let (jf, jd) = (jump(&free.momentum_trajectory), jump(&damped.momentum_trajectory));
        assert!(jd < 0.5 * jf, "L-dot weight did not smooth: free {jf:.4} damped {jd:.4}");
    }

    /// LIPM: xi_dot = omega (xi - p). A DCM that sits AHEAD of where the
    /// plan wants it is brought back only by a pressure point ahead of it, so
    /// the capture rows must move the free landing FORWARD (dp.x > 0). The
    /// moment-arm coupling alone asks the opposite (see
    /// `a_forward_target_moves_the_free_landing_back`), so this also checks
    /// that the capture rows win when weighted. Weight 0 = bit-identical.
    #[test]
    fn the_capture_rows_move_the_landing_toward_the_escaping_dcm() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        cfg.omega = (cfg.gravity / input.com.z).sqrt();
        cfg.net_cop_weight = 1.0;
        input.reference_net_cop_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.reference_dcm_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.landing = Some(Landing { foot: 1, first_node: 0, weight: 1.0, max: [0.10, 0.05], prior: [0.0; 2], min: [0.10, 0.05], min_first_node: 0, position: [0.0; 2], terrain_x: None });
        input.dcm_now = Some(na::Vector2::new(0.04, 0.0)); // DCM 4 cm ahead of the plan
        let without = solve(&input, &cfg).unwrap();
        cfg.landing_capture_weight = 1e4;
        let with = solve(&input, &cfg).unwrap();
        let (d0, d1) = (without.landing_shift.unwrap(), with.landing_shift.unwrap());
        assert!(d1.x > 0.01, "capture must move the landing forward, got {d1:?} (without: {d0:?})");
        assert!(d1.x > d0.x + 0.005, "capture must move it further forward than the moment coupling alone");
        // Off = identical.
        cfg.landing_capture_weight = 0.0;
        let again = solve(&input, &cfg).unwrap();
        assert_eq!(again.landing_shift.unwrap(), d0);
    }

    /// With the plan's own sensitivity J in the rows, the same escaping DCM
    /// asks for a SMALLER shift of the same sign: the rows now know that the
    /// reference moves 0.65 dp with the landing, so the error is closed by
    /// (b - J) dp instead of b dp. `None` must reproduce the rows exactly.
    #[test]
    fn the_plan_sensitivity_shrinks_the_capture_shift_without_changing_its_sign() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        cfg.omega = (cfg.gravity / input.com.z).sqrt();
        cfg.net_cop_weight = 1.0;
        cfg.landing_capture_weight = 1e4;
        input.reference_net_cop_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.reference_dcm_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        // Landing late in the horizon: the shipped rows have a small b there
        // and ask for a shift the box has to clamp.
        input.landing = Some(Landing { foot: 1, first_node: cfg.horizon - 2, weight: 1.0, max: [0.30, 0.30], prior: [0.0; 2], min: [0.30, 0.30], min_first_node: 0, position: [0.0; 2], terrain_x: None });
        input.dcm_now = Some(na::Vector2::new(0.02, 0.0));
        let plain = solve(&input, &cfg).unwrap().landing_shift.unwrap();
        input.reference_dcm_sensitivity = Some(vec![na::Matrix2::zeros(); cfg.horizon]);
        let zero_j = solve(&input, &cfg).unwrap().landing_shift.unwrap();
        assert!((zero_j - plain).norm() < 1e-9, "J = 0 must reproduce the rows: {zero_j:?} vs {plain:?}");
        input.reference_dcm_sensitivity = Some(vec![na::Matrix2::identity() * 0.65; cfg.horizon]);
        let with_j = solve(&input, &cfg).unwrap().landing_shift.unwrap();
        assert!(plain.x > 0.0 && with_j.x > 0.0, "same sign: {plain:?} vs {with_j:?}");
        assert!(with_j.x < 0.5 * plain.x, "J must shrink the shift: {plain:?} -> {with_j:?}");
    }

    /// With an offset already applied, the box and cost act on the TOTAL:
    /// a landing already at the box edge can only come back, and with no
    /// other pull the increment returns it toward zero.
    #[test]
    fn a_prior_offset_moves_the_box_and_the_cost_to_the_total() {
        let (mut input, cfg) = standing([0.0, 0.0]);
        input.landing = Some(Landing { foot: 1, first_node: 0, weight: 1e6, max: [0.06, 0.03], prior: [0.06, -0.03], min: [0.06, 0.03], min_first_node: 0, position: [0.0; 2], terrain_x: None });
        let dp = solve(&input, &cfg).unwrap().landing_shift.unwrap();
        assert!(dp.x <= 1e-9 && dp.y >= -1e-9, "box on the total: {dp:?}");
        assert!((dp.x + 0.06).abs() < 1e-3 && (dp.y - 0.03).abs() < 1e-3, "cost on the total pulls it back to zero: {dp:?}");
    }

    /// A zero backward bound forbids dp.x < 0 even when the rest of the
    /// problem pulls the landing back; the forward side is untouched.
    #[test]
    fn a_zero_backward_bound_keeps_the_landing_from_moving_back() {
        let (mut input, cfg) = standing([0.0, 0.0]);
        input.target_velocity.x = 0.3; // pulls the free landing back (see below)
        input.landing = Some(Landing { foot: 1, first_node: 0, weight: 1.0, max: [0.06, 0.03], prior: [0.0; 2], min: [0.06, 0.03], min_first_node: 0, position: [0.0; 2], terrain_x: None });
        let free = solve(&input, &cfg).unwrap().landing_shift.unwrap();
        assert!(free.x < -1e-3, "premise: the free landing moves back, got {free:?}");
        input.landing = Some(Landing { foot: 1, first_node: 0, weight: 1.0, max: [0.06, 0.03], prior: [0.0; 2], min: [0.0, 0.03], min_first_node: 0, position: [0.0; 2], terrain_x: None });
        let held = solve(&input, &cfg).unwrap().landing_shift.unwrap();
        assert!(held.x >= -1e-9, "backward bound 0: {held:?}");
    }

    /// The 2-D box QP must match brute force on a grid, inside and on the
    /// boundary.
    #[test]
    fn the_2d_box_qp_is_exact() {
        let h = na::Matrix2::new(3.0, 1.2, 1.2, 2.0);
        for g in [na::Vector2::new(-0.3, 0.1), na::Vector2::new(2.0, -3.0), na::Vector2::new(-5.0, -5.0)] {
            let (lo, hi) = ([-0.5, -0.2], [0.4, 0.6]);
            let x = box_qp_2d(&h, &g, lo, hi);
            let f = |x: &na::Vector2<f64>| 0.5 * x.dot(&(h * x)) + g.dot(x);
            let mut best = f64::INFINITY;
            for i in 0..=400 {
                for j in 0..=400 {
                    let y = na::Vector2::new(lo[0] + (hi[0] - lo[0]) * i as f64 / 400.0, lo[1] + (hi[1] - lo[1]) * j as f64 / 400.0);
                    best = best.min(f(&y));
                }
            }
            assert!(f(&x) <= best + 1e-9, "g={g:?}: {} vs grid {best}", f(&x));
            assert!((0..2).all(|k| x[k] >= lo[k] - 1e-12 && x[k] <= hi[k] + 1e-12));
        }
    }

    /// Lexicographic mode = the limit of an infinite capture weight: it
    /// must agree with a very large penalty weight (same sign, close), and
    /// the solved landing must equal the fixed one. Off = bit-identical.
    #[test]
    fn the_lexicographic_landing_is_the_infinite_weight_limit() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        cfg.omega = (cfg.gravity / input.com.z).sqrt();
        cfg.net_cop_weight = 1.0;
        input.reference_net_cop_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.reference_dcm_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.landing = Some(Landing { foot: 1, first_node: 0, weight: 1.0, max: [0.10, 0.05], prior: [0.0; 2], min: [0.10, 0.05], min_first_node: 0, position: [0.0; 2], terrain_x: None });
        input.dcm_now = Some(na::Vector2::new(0.02, 0.01));
        cfg.landing_capture_weight = 1e4;
        let shipped = solve(&input, &cfg).unwrap();
        cfg.landing_capture_weight = 1e9;
        let heavy = solve(&input, &cfg).unwrap().landing_shift.unwrap();
        cfg.landing_capture_weight = 1e4;
        cfg.landing_lexicographic = true;
        let fixed = lexicographic_landing(&input, &cfg).unwrap();
        let lex = solve(&input, &cfg).unwrap();
        let got = lex.landing_shift.unwrap();
        assert!((got - fixed).norm() < 1e-5, "solved {got:?} vs fixed {fixed:?}");
        assert!((fixed - heavy).norm() < 2e-3, "lex {fixed:?} vs 1e9 penalty {heavy:?}");
        cfg.landing_lexicographic = false;
        let again = solve(&input, &cfg).unwrap();
        assert_eq!(again.landing_shift, shipped.landing_shift);
    }

    /// LIPM: a DCM ahead of the plan is brought back only by a pressure
    /// point ahead of it. With the capture rows built on the MPC's own CoP,
    /// the solved net CoP (first node) must move FORWARD of the planned one
    /// when the DCM runs ahead, and off must be bit-identical.
    #[test]
    fn the_capture_cop_rows_move_the_pressure_point_ahead_of_an_escaping_dcm() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        cfg.omega = (cfg.gravity / input.com.z).sqrt();
        cfg.net_cop_weight = 1.0;
        input.reference_net_cop_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.reference_dcm_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.dcm_now = Some(na::Vector2::new(0.02, 0.0));
        let net_cop_x = |sol: &Solution| {
            // Net CoP x of node 0 from the sole wrenches: feet at x = 0, flat.
            let fz: f64 = sol.sole_wrenches.iter().map(|w| w[5]).sum();
            let my: f64 = sol.sole_wrenches.iter().map(|w| w[1]).sum();
            -my / fz
        };
        let off = solve(&input, &cfg).unwrap();
        cfg.capture_cop_weight = 1e3;
        let on = solve(&input, &cfg).unwrap();
        assert!(net_cop_x(&on) > net_cop_x(&off) + 0.005, "on {} off {}", net_cop_x(&on), net_cop_x(&off));
        cfg.capture_cop_weight = 0.0;
        let again = solve(&input, &cfg).unwrap();
        assert_eq!(again.sole_wrenches, off.sole_wrenches);
    }

    /// LIPM: a DCM AHEAD of the plan is recaptured only by a pressure point
    /// ahead of it. The CoP plan must put node 0 ahead of the DCM, inside the
    /// reachable box, and the wrench stage must then realise exactly that
    /// net CoP at node 0 (equality). Off = the shipped problem, bit for bit.
    #[test]
    fn the_lexicographic_cop_recaptures_an_escaping_dcm_and_the_wrench_realises_it() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        cfg.omega = (cfg.gravity / input.com.z).sqrt();
        input.reference_net_cop_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.reference_dcm_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.dcm_now = Some(na::Vector2::new(0.02, 0.0));
        let shipped = assemble(&input, &cfg).unwrap();
        cfg.capture_cop_lexicographic = true;
        let plan = lexicographic_cop(&input, &cfg).unwrap();
        assert!(plan[0].x > 0.02, "p_0 must be ahead of the DCM: {:?}", plan[0]);
        // Reachable box of a 50/50 split of two soles at y = +-0.08 with the
        // test's cop_half: x within +-cop_half.0 - margin.
        assert!(plan[0].x <= cfg.cop_half.0 - cfg.capture_cop_margin + 1e-9);
        let sol = solve(&input, &cfg).unwrap();
        assert_eq!(sol.cop_plan.as_ref().unwrap()[0], plan[0]);
        // Net CoP of the solved node-0 wrench.
        let (mut mx, mut my, mut fz) = (0.0, 0.0, 0.0);
        for foot in 0..2 {
            let c = &input.contacts[foot];
            let w = sol.sole_wrenches[foot];
            let m = c.rotation * na::Vector3::new(w[0], w[1], w[2]);
            let f = c.rotation * na::Vector3::new(w[3], w[4], w[5]);
            mx += m.x + c.position.y * f.z - c.position.z * f.y;
            my += m.y - c.position.x * f.z + c.position.z * f.x;
            fz += f.z;
        }
        let cop = na::Vector2::new(-my / fz, mx / fz);
        assert!((cop - plan[0]).norm() < 2e-3, "wrench must realise the plan: {cop:?} vs {:?}", plan[0]);
        // Total vertical force held to the schedule.
        let total: f64 = (0..2).map(|f| input.reference_fz_schedule[0][f]).sum();
        assert!((sol.sole_wrenches[0][5] + sol.sole_wrenches[1][5] - total).abs() < 1e-3);
        cfg.capture_cop_lexicographic = false;
        let again = assemble(&input, &cfg).unwrap();
        assert_eq!(again.h, shipped.h);
        assert_eq!(again.linear, shipped.linear);
        assert!(again.cop_plan.is_none());
    }

    /// A large error saturates the plan at the box edge, never beyond.
    #[test]
    fn the_lexicographic_cop_stays_inside_the_reachable_box() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        cfg.omega = (cfg.gravity / input.com.z).sqrt();
        cfg.capture_cop_lexicographic = true;
        input.reference_net_cop_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.reference_dcm_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.dcm_now = Some(na::Vector2::new(0.5, -0.5));
        let plan = lexicographic_cop(&input, &cfg).unwrap();
        let hx = cfg.cop_half.0 - cfg.capture_cop_margin;
        for p in &plan {
            assert!(p.x <= hx + 1e-9 && p.x >= -hx - 1e-9, "{p:?}");
        }
        assert!((plan[0].x - hx).abs() < 1e-9, "must sit at the forward edge: {:?}", plan[0]);
        assert!(plan[0].y < 0.0, "must move toward the escaping side in y: {:?}", plan[0]);
    }

    /// The walker's standing input of 2026-10-02 00:55 that the lexicographic
    /// CoP stage made "infeasible": a net CoP 3 cm to the side of mid-feet
    /// with 20 N minimum per foot is reachable (load the near foot 55/45 and
    /// use its sole moment), and Clarabel solves it. The active-set backend
    /// reports Infeasible for the same problem with ANY positive minimum
    /// load, so the lexicographic mode must not run on it.
    #[test]
    fn a_sideways_net_cop_equality_is_feasible_on_clarabel_but_not_the_active_set() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        cfg.mass = 34.134;
        cfg.max_fz = 2.3 * cfg.mass * cfg.gravity;
        cfg.omega = (cfg.gravity / 0.683).sqrt();
        cfg.net_cop_weight = 10.0;
        cfg.net_cop_equality = true;
        input.com = na::Vector3::new(0.0465, 0.003, 0.683);
        input.target_com = input.com;
        for foot in 0..2 {
            input.contacts[foot].cop_half = (0.077590975, 0.02738505);
        }
        input.contacts[0].position = na::Vector3::new(0.041, 0.056, 0.0);
        input.contacts[1].position = na::Vector3::new(0.052, -0.050, 0.0);
        input.min_fz_schedule = vec![[20.091; 2]; cfg.horizon];
        input.reference_fz_schedule = vec![[167.4266; 2]; cfg.horizon];
        input.previous[0][5] = 167.4266;
        input.previous[1][5] = 167.4266;
        input.reference_net_cop_schedule = Some(vec![na::Vector2::new(0.0465, -0.03); cfg.horizon]);
        let problem = assemble(&input, &cfg).unwrap();
        cfg.qp.solver = misa_wbc::QpSolver::ActiveSet;
        let active = solve_dense(&problem, &cfg.qp, None);
        cfg.qp.solver = misa_wbc::QpSolver::Clarabel;
        let clarabel = solve_dense(&problem, &cfg.qp, None);
        assert_eq!(clarabel.status, QpStatus::Optimal, "{:?}", clarabel.status);
        assert!(clarabel.x[11] > clarabel.x[5], "the near foot must carry more: {} vs {}", clarabel.x[11], clarabel.x[5]);
        assert_ne!(active.status, QpStatus::Optimal, "if the active set now solves this, the lexicographic mode may use it");
        // The lexicographic stage on Clarabel from a DCM 2.5 cm ahead and to
        // the side: feasible, with the plan inside the reachable box.
        cfg.net_cop_equality = false;
        cfg.capture_cop_lexicographic = true;
        input.reference_dcm_schedule = Some(vec![na::Vector2::new(0.0465, 0.003); cfg.horizon]);
        input.reference_net_cop_schedule = Some(vec![na::Vector2::new(0.0465, 0.003); cfg.horizon]);
        for xi in [(0.05, 0.0), (0.07, 0.0), (0.0465, -0.01), (0.07, -0.01)] {
            input.dcm_now = Some(na::Vector2::new(xi.0, xi.1));
            let plan = lexicographic_cop(&input, &cfg).unwrap();
            let problem = assemble(&input, &cfg).unwrap();
            let qp = solve_dense(&problem, &cfg.qp, None);
            eprintln!("lex xi={xi:?} plan0=({:.4},{:.4}) status={:?}", plan[0].x, plan[0].y, qp.status);
        }
        input.dcm_now = Some(na::Vector2::new(0.05, 0.0));
        let sol = solve(&input, &cfg);
        assert!(sol.is_some(), "lexicographic CoP must solve on Clarabel");
    }

    /// Step timing: a DCM far ahead of the plan with the nominal touchdown
    /// 4 nodes away is captured better by landing earlier; with the timing
    /// weight the stage must pick an earlier node and a lower capture
    /// residual than the nominal one. Timing off = the nominal node.
    #[test]
    fn the_timed_landing_lands_earlier_when_the_dcm_runs_away() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        cfg.omega = (cfg.gravity / input.com.z).sqrt();
        cfg.landing_lexicographic = true;
        cfg.net_cop_weight = 1.0;
        input.reference_net_cop_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.reference_dcm_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.contact_schedule = (0..cfg.horizon).map(|k| [true, k >= 4]).collect();
        input.landing = Some(Landing {
            foot: 1, first_node: 4, weight: 1.0, max: [0.10, 0.05], prior: [0.0; 2], min: [0.10, 0.05],
            min_first_node: 1, position: [0.0, -0.08], terrain_x: None,
        });
        input.dcm_now = Some(na::Vector2::new(0.08, 0.0));
        let nominal = lexicographic_landing_timed(&input, &cfg).unwrap();
        assert_eq!(nominal.1, 4, "timing off keeps the nominal node");
        cfg.landing_timing = true;
        cfg.landing_time_weight = 1e-3;
        let timed = lexicographic_landing_timed(&input, &cfg).unwrap();
        assert!(timed.1 < 4, "must land earlier: {:?}", timed);
        let (_, c_nom) = landing_at(&input, &cfg, input.landing.as_ref().unwrap(), 4).unwrap();
        let (_, c_timed) = landing_at(&input, &cfg, input.landing.as_ref().unwrap(), timed.1).unwrap();
        assert!(c_timed < c_nom, "earlier landing must capture better: {c_timed} vs {c_nom}");
        // A DCM on the plan keeps the nominal timing.
        input.dcm_now = Some(na::Vector2::new(0.0, 0.0));
        let calm = lexicographic_landing_timed(&input, &cfg).unwrap();
        assert_eq!(calm.1, 4, "no error, no shortening: {:?}", calm);
    }

    /// A landing the capture wants 6 cm ahead, on a tread whose window ends
    /// 2 cm ahead: with the terrain term the landing stays near the window
    /// edge (within the margin the weight allows), without it it goes to
    /// 6 cm; a landing already inside the window is unchanged.
    #[test]
    fn the_terrain_term_pulls_a_landing_back_onto_the_tread() {
        let (mut input, mut cfg) = standing([0.0, 0.0]);
        cfg.omega = (cfg.gravity / input.com.z).sqrt();
        cfg.landing_lexicographic = true;
        input.reference_net_cop_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.reference_dcm_schedule = Some(vec![na::Vector2::zeros(); cfg.horizon]);
        input.dcm_now = Some(na::Vector2::new(0.03, 0.0));
        let mut land = Landing { foot: 1, first_node: 0, weight: 1.0, max: [0.10, 0.05], prior: [0.0; 2], min: [0.10, 0.05], min_first_node: 0, position: [0.0, -0.08], terrain_x: None };
        input.landing = Some(land);
        let free = lexicographic_landing(&input, &cfg).unwrap();
        assert!(free.x > 0.04, "{free:?}");
        land.terrain_x = Some((-0.10, 0.02, 100.0));
        input.landing = Some(land);
        let held = lexicographic_landing(&input, &cfg).unwrap();
        assert!(held.x < free.x - 0.01 && held.x > 0.0, "terrain must pull it back: {held:?} vs {free:?}");
        land.terrain_x = Some((-0.10, 0.20, 100.0));
        input.landing = Some(land);
        let inside = lexicographic_landing(&input, &cfg).unwrap();
        assert!((inside - free).norm() < 1e-9, "inside the window nothing changes: {inside:?} vs {free:?}");
    }

    #[test]
    fn dense_and_structured_ocp_are_equivalent() {
        for horizon in [1, 8, 16, 32] {
            let (mut input, mut cfg) = standing([0.0, 0.0]);
            cfg.horizon = horizon;
            input.contact_schedule = vec![[true, false]; horizon];
            input.activation_schedule = vec![[0.7, 0.0]; horizon];
            input.reference_fz_schedule = vec![[70.0, 0.0]; horizon];
            input.min_fz_schedule = vec![[0.0, 0.0]; horizon];
            input.previous[0][5] = 49.0;
            input.target_com.x = 0.1;
            input.contacts[0].rotation =
                na::Rotation3::from_euler_angles(0.1, -0.2, 0.3).into_inner();
            input.contacts[1].cop_half = (0.035, 0.018);
            let dense = assemble(&input, &cfg).unwrap();
            let ocp = assemble_ocp(&input, &cfg).unwrap();
            let u = na::DVector::from_iterator(
                12 * horizon,
                (0..12 * horizon).map(|j| ((j * 17 % 71) as f64 - 35.0) * 0.1),
            );
            let mut x = ocp.initial.clone();
            let mut value = 0.0;
            for (k, stage) in ocp.stages.iter().enumerate() {
                let uk = u.rows(12 * k, 12).into_owned();
                value += 0.5 * x.dot(&(&stage.q_matrix * &x))
                    + 0.5 * uk.dot(&(&stage.r_matrix * &uk))
                    + uk.dot(&(&stage.s_matrix * &x))
                    + stage.q.dot(&x)
                    + stage.r.dot(&uk)
                    + stage.constant;
                let violation = &stage.c * &x + &stage.d * &uk - &stage.upper;
                let dense_violation = dense.d.rows(24 * k, 24) * &u - dense.rhs.rows(24 * k, 24);
                assert!((violation - dense_violation).amax() < 1e-10);
                x = &stage.a * &x + &stage.b * &uk + &stage.bias;
            }
            let expected =
                0.5 * u.dot(&(&dense.h * &u)) + dense.linear.dot(&u) + dense.objective_constant;
            assert!(
                (value - expected).abs() < 1e-8 * (1.0 + expected.abs()),
                "{value} != {expected}"
            );
            let terminal = &dense.terminal_map * &u + &dense.terminal_bias;
            assert!((x.rows(0, 9) - terminal).amax() < 1e-10);
        }
    }

    #[test]
    fn rejects_nonfinite_weights_and_failed_residuals() {
        let (input, mut cfg) = standing([0.0, 0.0]);
        cfg.position_weight = f64::NAN;
        assert!(assemble(&input, &cfg).is_none());
        cfg.position_weight = 20.0;
        let p = assemble(&input, &cfg).unwrap();
        let mut result = solve_dense(&p, &cfg.qp, None);
        result.x[5] = -1.0;
        assert!(!residuals(&p, &result).accepted(1e-6));
    }

    #[cfg(feature = "qp-hpipm")]
    #[test]
    fn hpipm_ocp_matches_dense_original_qp() {
        for horizon in [1, 8, 16, 32] {
            let (mut input, mut cfg) = standing([0.0, 0.0]);
            cfg.horizon = horizon;
            input.contact_schedule = vec![[true, true]; horizon];
            input.activation_schedule = vec![[1.0, 1.0]; horizon];
            input.reference_fz_schedule = vec![[49.05, 49.05]; horizon];
            input.min_fz_schedule = vec![[0.0, 0.0]; horizon];
            let p = assemble(&input, &cfg).unwrap();
            let ocp = assemble_ocp(&input, &cfg).unwrap();
            let mut qp = cfg.qp.clone();
            qp.solver = misa_wbc::qp::QpSolver::Hpipm;
            let structured = misa_wbc::ocp::solve_ocp(&ocp, &qp);
            let dense = solve_dense(&p, &qp, None);
            assert_eq!(
                structured.status,
                QpStatus::Optimal,
                "N={horizon} {:?}",
                structured.diagnostics
            );
            assert_eq!(dense.status, QpStatus::Optimal, "N={horizon}");
            let r = residuals(&p, &structured);
            assert!(r.accepted(1e-6), "N={horizon} {r:?}");
            let original =
                0.5 * structured.x.dot(&(&p.h * &structured.x)) + p.linear.dot(&structured.x);
            assert!(
                (original - dense.objective).abs() < 1e-6 * (1.0 + dense.objective.abs()),
                "N={horizon}: {original} != {}",
                dense.objective
            );
            assert!(solve(&input, &Config { qp, ..cfg }).is_some());
        }
    }

    #[test]
    fn common_validation_flag_rejects_loose_default_backend_solution() {
        let (input, mut cfg) = standing([0.0, 0.0]);
        // A deliberately loose stopping tolerance labels the initial feasible
        // reference Optimal although smoothing leaves a stationarity residual.
        cfg.qp.optimality_tol = 1e6;
        let mut workspace = Workspace::default();
        assert!(solve_with_workspace(&input, &cfg, Some(&mut workspace)).is_some());
        assert!(!workspace.last_report().unwrap().residuals.accepted(1e-6));
        cfg.validate_solution = true;
        assert!(solve_with_workspace(&input, &cfg, Some(&mut workspace)).is_none());
        assert!(!workspace.last_report().unwrap().residuals.accepted(1e-6));
    }

    #[test]
    fn static_balance_shares_weight() {
        let (input, cfg) = standing([0.0, 0.0]);
        let sol = solve(&input, &cfg).unwrap();
        let fz = sol.sole_wrenches[0][5] + sol.sole_wrenches[1][5];
        assert!((fz - cfg.mass * cfg.gravity).abs() < 2.0, "fz={fz}");
        assert!(sol.acceleration.norm() < 0.2, "a={:?}", sol.acceleration);
    }

    #[test]
    fn preserves_low_foot_minimum_load() {
        let (mut input, cfg) = standing([25.0, 0.0]);
        input.reference_fz_schedule.fill([25.0, 73.1]);
        let sol = solve(&input, &cfg).unwrap();
        assert!(sol.sole_wrenches[0][5] >= 25.0 - 1e-6);
    }

    #[test]
    fn returned_wrench_respects_patch_constraints() {
        let (mut input, cfg) = standing([5.0, 5.0]);
        input.target_acceleration = na::Vector3::new(20.0, -20.0, 0.0);
        let sol = solve(&input, &cfg).unwrap();
        for wrench in sol.sole_wrenches {
            let fz = wrench[5];
            assert!(wrench[3].abs() <= cfg.friction_mu * fz + 1e-6);
            assert!(wrench[4].abs() <= cfg.friction_mu * fz + 1e-6);
            assert!(wrench[0].abs() <= cfg.cop_half.1 * fz + 1e-6);
            assert!(wrench[1].abs() <= cfg.cop_half.0 * fz + 1e-6);
            assert!(wrench[2].abs() <= cfg.torsion_mu * fz + 1e-6);
        }
    }

    #[test]
    fn rejects_impossible_minimum() {
        let (input, mut cfg) = standing([160.0, 0.0]);
        cfg.max_fz = 150.0;
        assert!(solve(&input, &cfg).is_none());
    }

    #[test]
    fn unavailable_contact_has_zero_wrench() {
        let (mut input, cfg) = standing([25.0, 5.0]);
        input.contact_schedule = vec![[false, true]; cfg.horizon];
        input
            .reference_fz_schedule
            .fill([25.0, cfg.mass * cfg.gravity]);
        let sol = solve(&input, &cfg).unwrap();
        assert!(
            sol.sole_wrenches[0].norm() < 1e-7,
            "w={:?}",
            sol.sole_wrenches[0]
        );
        assert!(sol.sole_wrenches[1][5] > 0.0);
    }

    #[test]
    fn fractional_activation_caps_normal_force() {
        let (mut input, cfg) = standing([0.0, 0.0]);
        input.activation_schedule.fill([0.4, 1.0]);
        let sol = solve(&input, &cfg).unwrap();
        assert!(sol.sole_wrenches[0][5] <= 0.4 * cfg.max_fz + 1e-6);
    }

    #[test]
    fn rejects_wrong_schedule_length() {
        let (mut input, cfg) = standing([0.0, 0.0]);
        input.contact_schedule.pop();
        assert!(solve(&input, &cfg).is_none());
    }

    #[test]
    fn stage_force_schedule_sets_first_stage_minimum() {
        let (mut input, cfg) = standing([0.0, 0.0]);
        input.reference_fz_schedule = (0..cfg.horizon)
            .map(|k| [10.0 + 8.0 * k as f64, 88.1 - 8.0 * k as f64])
            .collect();
        input.min_fz_schedule = (0..cfg.horizon)
            .map(|k| [5.0 + 5.0 * k as f64, 0.0])
            .collect();
        let sol = solve(&input, &cfg).unwrap();
        assert!(sol.sole_wrenches[0][5] >= 5.0 - 1e-6);
    }

    #[test]
    fn rejects_wrong_force_schedule_length() {
        let (mut input, cfg) = standing([0.0, 0.0]);
        input.reference_fz_schedule.pop();
        assert!(solve(&input, &cfg).is_none());
    }
}
