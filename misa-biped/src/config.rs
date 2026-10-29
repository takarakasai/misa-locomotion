//! The walker's configuration: every control knob, in one value.
//!
//! Moved here from the environment reads of `examples/kyo46rs_walk/main.rs`
//! on 2026-10-29 (S9 of doc/from_claude/20261029_walker_extraction_plan.md).
//! This crate never reads the environment; the harness maps its environment
//! (or a JSON of the same keys) onto a [`Config`] once at start-up
//! (`examples/kyo46rs_walk/config_env.rs`), starting from
//! [`Config::code_defaults`]. The doc of each field names the key it is read
//! from.
//!
//! Values are stored AFTER the read's own post-processing (`.max(0.0)`,
//! `.clamp(..)`, `as usize`), which is the value the walker uses. Two kinds of
//! default cannot be fixed here and are `Option`s, `None` meaning "derive it":
//!
//! - from another knob (`KD_CONTACT` = 2 sqrt(`KP_CONTACT`), ...), resolved by
//!   a method on the sub-struct;
//! - from the run's context (the plant's friction, the body mass, the initial
//!   CoM height, whether a terrain model exists), resolved by a method that
//!   takes it as an argument.
//!
//! A bound against the realised control period (`.max(dt)`) or another
//! sub-struct's knob stays at the use site.
//!
//! There is no `Default`: the G1 champion's values as the default would move
//! kyo46rs's baseline, and the code's own defaults depend on the machine
//! ([`Config::code_defaults`] takes the [`Profile`]).
use crate::plan::DcmParams;
use crate::profile::Profile;

/// Every control knob of the walker.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub wbc: WbcConfig,
    pub joint_limit: JointLimitConfig,
    pub gains: BalanceGains,
    pub dcm: DcmConfig,
    pub gait: GaitConfig,
    pub stair_plan: StairPlanConfig,
    pub descent: DescentConfig,
    pub slope: SlopeConfig,
}

impl Config {
    /// The values the walker runs with when no knob is set: the code's own
    /// defaults, some of them the machine's (`prof.*`).
    pub fn code_defaults(prof: &Profile) -> Self {
        Self {
            wbc: WbcConfig {
                feas_tol: misa_wbc::SolveConfig::default().feasibility_tol,
                strategy: misa_wbc::SolveConfig::default().strategy,
                backend: None,
                bench_gate: false,
                friction_mu: None,
                f_max_scale: 2.3,
                f_max: None,
                eom_hard: true,
                hold_kp: prof.hold_kp,
                hold_kd: prof.hold_kd,
                hold_bridge: 8,
                fallback_level: 999,
                level_weights: LevelWeights::defaults(),
                ctrl_dt: prof.ctrl_dt,
            },
            joint_limit: JointLimitConfig {
                enabled: false,
                alpha: [10.0; 3],
                a_max: 50.0,
                v_frac: 1.0,
            },
            gains: BalanceGains {
                kp_com: 300.0,
                kd_com: 80.0,
                kp_trunk: 200.0,
                kd_trunk: 40.0,
                kp_post: 100.0,
                kd_post: 20.0,
                posture: true,
                trunk: true,
                trunk_sign: 1.0,
                trunk_yaw: true,
                kp_yaw: 200.0,
                kd_yaw: 40.0,
                diag_vz_bias_mpc: 0.0,
                diag_vz_bias_height: 0.0,
            },
            dcm: DcmConfig {
                cop_frac: 1.0,
                k_dcm: 2.0,
                vh_dcm: false,
                zmp_margin: 0.8,
                clamp_zmp: true,
                zmp_lat_scale: 1.0,
                zmp_roll: None,
                com_xy_jerk_max: f64::INFINITY,
            },
            gait: GaitConfig {
                t_start: 2.0,
                t_ds: 0.20,
                t_ss: 0.45,
                n_steps: 20,
                first_swing: 1,
                no_lift: false,
                turn_in_ds: true,
                ds_settle: 0.0,
                ds_settle_after_step: 1,
                stride: 0.0,
                stride_ramp: 6,
                alternate: true,
            },
            stair_plan: StairPlanConfig {
                toe_margin: 0.020,
                y_out: 0.010,
                gait_pair: false,
                td_bias: 0.0,
                td_bias_asc_mid: None,
                full_sole: false,
                sole_margin: 0.002,
                landing_frac: 0.5,
                align_stand_line: false,
            },
            descent: DescentConfig {
                td_bias_desc: None,
                stand_margin: 0.03,
                com_height_lead: 0.0,
                zone_crouch: 0.0,
                land_extend: None,
                land_confirm_xy: None,
                land_confirm_z: None,
                leg_reach_margin: 0.95,
            },
            slope: SlopeConfig {
                keep_stride: false,
                heel_cut: 0.0,
                heel_cut_max: 0.05,
                terrain_pitch: false,
                terrain_pitch_yaw: false,
            },
        }
    }
}

/// The whole-body QP: its solver, its contact model's bounds, the
/// degraded-solve fallback and the weighted strategy's per-level weights.
#[derive(Clone, Debug, PartialEq)]
pub struct WbcConfig {
    /// `WBC_FEAS_TOL` (positive and finite; checked by the harness).
    pub feas_tol: f64,
    /// `WBC_STRATEGY`: `null_space` (unset), `weighted` (with
    /// `WBC_WEIGHT_RATIO`, at least 1), `force_budget`.
    pub strategy: misa_wbc::HqpStrategy,
    /// `WBC_QP_BACKEND`. None = the solver's default.
    pub backend: Option<misa_wbc::QpSolver>,
    /// `QP_BENCH_GATE`: validate every solve (also on for the native
    /// backends proxqp / eiquadprog / qpoases, see [`WbcConfig::qp_gate`]).
    pub bench_gate: bool,
    /// `friction_mu`. None = the plant's `MU_GROUND` x the profile's
    /// friction margin ([`WbcConfig::friction_mu`]).
    pub friction_mu: Option<f64>,
    /// `F_MAX_SCALE` [body weights] and `F_MAX` [N] per foot. None = the
    /// scale x body weight ([`WbcConfig::f_max_per_foot`]).
    pub f_max_scale: f64,
    pub f_max: Option<f64>,
    /// `EOM_HARD`.
    pub eom_hard: bool,
    /// `HOLD_KP`, `HOLD_KD` (profile), `HOLD_BRIDGE` [ticks],
    /// `FALLBACK_LEVEL`: the degraded-solve policy.
    pub hold_kp: f64,
    pub hold_kd: f64,
    pub hold_bridge: u32,
    pub fallback_level: usize,
    /// `WBCW_<LEVEL>`: the weighted strategy's per-level weights.
    pub level_weights: LevelWeights,
    /// `CTRL_DT` (profile): the NOMINAL control period. The realised one
    /// (a whole number of plant steps) is the harness's.
    pub ctrl_dt: f64,
}

impl WbcConfig {
    /// Whether every solve is validated: asked for, or a native backend.
    pub fn qp_gate(&self) -> bool {
        self.bench_gate
            || matches!(
                self.backend,
                Some(misa_wbc::QpSolver::Proxqp | misa_wbc::QpSolver::Eiquadprog | misa_wbc::QpSolver::Qpoases)
            )
    }

    /// The friction the QP plans against: `friction_mu`, or the plant's
    /// friction `mu_ground` times the profile's margin.
    pub fn friction_mu(&self, mu_ground: f64, friction_margin: f64) -> f64 {
        self.friction_mu.unwrap_or(mu_ground * friction_margin)
    }

    /// The normal-force bound per foot [N]: `F_MAX`, or `F_MAX_SCALE` x m g.
    pub fn f_max_per_foot(&self, total_mass: f64, g: f64) -> f64 {
        self.f_max.unwrap_or(self.f_max_scale * total_mass * g)
    }
}

/// The weighted strategy's weight per WBC level, for every level name the
/// walker's stack produces. An array, not a map, so a lookup cannot depend
/// on an iteration order.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelWeights(pub [(&'static str, f64); 13]);

impl LevelWeights {
    pub const NAMES: [&'static str; 13] = [
        "eom",
        "contact+cones",
        "dynamics+contact+cones",
        "com",
        "com+momentum+wrench",
        "trunk",
        "momentum",
        "arm hold(before swing)",
        "swing",
        "arm hold",
        "posture",
        "posture+momentum",
        "regularise",
    ];

    /// The default weight of a level, by name (0.1 for a name not listed).
    pub fn default_for(name: &str) -> f64 {
        match name {
            "com" | "com+momentum+wrench" => 1.0,
            "swing" | "contact recovery" => 1.0,
            "trunk" => 0.3,
            "momentum" => 0.1,
            "posture" | "posture+momentum" => 0.01,
            "regularise" => 1e-4,
            _ => 0.1,
        }
    }

    /// The knob of a level: the name upper-cased with non-alphanumerics as
    /// '_' after `WBCW_` (e.g. `WBCW_COM`, `WBCW_POSTURE_MOMENTUM`).
    pub fn key(name: &str) -> String {
        let key: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
            .collect();
        format!("WBCW_{key}")
    }

    pub fn defaults() -> Self {
        Self(Self::NAMES.map(|name| (name, Self::default_for(name))))
    }

    /// The weight of a level; None for a name not in the table.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.0.iter().find(|(n, _)| *n == name).map(|&(_, w)| w)
    }

    pub fn set(&mut self, name: &str, w: f64) {
        if let Some(e) = self.0.iter_mut().find(|(n, _)| *n == name) {
            e.1 = w;
        }
    }
}

/// Joint position / velocity safety limits (`misa_wbc::tasks::joint_limit_cbf`).
#[derive(Clone, Debug, PartialEq)]
pub struct JointLimitConfig {
    /// `JLIM`.
    pub enabled: bool,
    /// `JLIM_ALPHA1`, `JLIM_ALPHA2`, `JLIM_ALPHA3`.
    pub alpha: [f64; 3],
    /// `JLIM_AMAX`.
    pub a_max: f64,
    /// `JLIM_V_FRAC` (0.05..1): the CBF's velocity limit as a fraction of
    /// the mechanical one. Below 1 needs `JLIM`.
    pub v_frac: f64,
}

/// Task-space PD gains of the CoM, the trunk and the posture.
#[derive(Clone, Debug, PartialEq)]
pub struct BalanceGains {
    /// `KP_COM`, `KD_COM`.
    pub kp_com: f64,
    pub kd_com: f64,
    /// `KP_TRUNK`, `KD_TRUNK`.
    pub kp_trunk: f64,
    pub kd_trunk: f64,
    /// `KP_POST`, `KD_POST`, and `POST` (the posture level on).
    pub kp_post: f64,
    pub kd_post: f64,
    pub posture: bool,
    /// `TRUNK` (the trunk level on), `TRUNK_SIGN`.
    pub trunk: bool,
    pub trunk_sign: f64,
    /// `TRUNK_YAW` (regulate yaw too), `KP_YAW`, `KD_YAW`.
    pub trunk_yaw: bool,
    pub kp_yaw: f64,
    pub kd_yaw: f64,
    /// `DIAG_VZ_BIAS_MPC`, `DIAG_VZ_BIAS_HEIGHT` [m/s]: a vertical CoM
    /// velocity error seen by one consumer only (diagnostic; 0 = off).
    pub diag_vz_bias_mpc: f64,
    pub diag_vz_bias_height: f64,
}

/// The DCM law and its ZMP reference.
#[derive(Clone, Debug, PartialEq)]
pub struct DcmConfig {
    /// `COP_FRAC`: the CoP box as a fraction of the sole.
    pub cop_frac: f64,
    /// `K_DCM`.
    pub k_dcm: f64,
    /// `VH_DCM`: the height-varying DCM law on a modelled terrain.
    pub vh_dcm: bool,
    /// `ZMP_MARGIN`, `CLAMP_ZMP`: keep the commanded ZMP inside the support
    /// polygon shrunk by this factor.
    pub zmp_margin: f64,
    pub clamp_zmp: bool,
    /// `ZMP_LAT_SCALE`.
    pub zmp_lat_scale: f64,
    /// `ZMP_ROLL` [m]. None = 0.030 with a terrain model, 0 without
    /// ([`DcmConfig::params`]).
    pub zmp_roll: Option<f64>,
    /// `COM_XY_JERK_MAX` (positive, or infinite = off; checked by the harness).
    pub com_xy_jerk_max: f64,
}

impl DcmConfig {
    /// The DCM reference's constants: the LIPM height is the CoM's above the
    /// terrain model's base (`terrain_base`, None = no terrain model), and
    /// the heel-to-toe roll defaults on with a terrain model.
    pub fn params(&self, z_com: f64, terrain_base: Option<f64>) -> DcmParams {
        DcmParams {
            z_lip: (z_com - terrain_base.unwrap_or(0.0)).max(0.05),
            lat_scale: self.zmp_lat_scale,
            roll: self.zmp_roll.unwrap_or(if terrain_base.is_some() { 0.030 } else { 0.0 }),
        }
    }
}

/// The support schedule and the stride.
#[derive(Clone, Debug, PartialEq)]
pub struct GaitConfig {
    /// `T_START`, `T_DS`, `T_SS` [s], `N_STEPS`, `FIRST_SWING` (0 = left).
    pub t_start: f64,
    pub t_ds: f64,
    pub t_ss: f64,
    pub n_steps: usize,
    pub first_swing: usize,
    /// `NO_LIFT`: run the schedule but never release a foot (W1).
    pub no_lift: bool,
    /// `TURN_IN_DS`: turn the trunk during the double support.
    pub turn_in_ds: bool,
    /// `DS_SETTLE` [s] (finite, non-negative; checked by the harness) after
    /// step `DS_SETTLE_AFTER_STEP` (at least 1).
    pub ds_settle: f64,
    pub ds_settle_after_step: usize,
    /// `STRIDE` [m/step]: shorthand for the forward speed (`VX`'s default).
    pub stride: f64,
    /// `STRIDE_RAMP` [steps].
    pub stride_ramp: usize,
    /// `ALTERNATE` (0 = the step-and-close plan).
    pub alternate: bool,
}

/// The footfalls on a known staircase.
#[derive(Clone, Debug, PartialEq)]
pub struct StairPlanConfig {
    /// `STAIR_TOE_MARGIN` [m]: toe clearance to riser 1 on the stand line.
    pub toe_margin: f64,
    /// `STAIR_Y_OUT` [m]: widen the on-stairs footfalls.
    pub y_out: f64,
    /// `STAIR_GAIT=pair`: both feet per tread.
    pub gait_pair: bool,
    /// `STAIR_TD_BIAS` [m]: tread landings behind the centre, and
    /// `STAIR_TD_BIAS_ASC_MID` for treads with a riser ahead (None = the same).
    pub td_bias: f64,
    pub td_bias_asc_mid: Option<f64>,
    /// `STAIR_FULL_SOLE`, `STAIR_SOLE_MARGIN` [m]: clamp the bias so the
    /// whole sole stays on the tread.
    pub full_sole: bool,
    pub sole_margin: f64,
    /// `STAIR_LANDING_FRAC` (at least 1e-3): the walk-out stride on the
    /// landing, as a fraction of the tread.
    pub landing_frac: f64,
    /// `STAIR_ALIGN_STAND_LINE`.
    pub align_stand_line: bool,
}

impl StairPlanConfig {
    pub fn td_bias_asc_mid(&self) -> f64 {
        self.td_bias_asc_mid.unwrap_or(self.td_bias)
    }
}

/// Stepping down.
#[derive(Clone, Debug, PartialEq)]
pub struct DescentConfig {
    /// `STAIR_TD_BIAS_DESC` [m]: the rearward bias on descending treads with
    /// a nosing ahead. None = `STAIR_TD_BIAS`.
    pub td_bias_desc: Option<f64>,
    /// `STAIR_DESC_STAND_MARGIN` [m] (non-negative): the toe's distance from
    /// the landing's edge before a descending flight.
    pub stand_margin: f64,
    /// `COM_HEIGHT_DESC_LEAD` (0..1): the fraction of the coming drop the
    /// height reference takes before the swing.
    pub com_height_lead: f64,
    /// `STAIR_ZONE_CROUCH` [m] (0..0.15): the CoM lowered in the stair zone.
    pub zone_crouch: f64,
    /// `COM_LAND_EXTEND_DESC` [s] (non-negative): the landing-deadline slack
    /// of a descending swing. None = `COM_LAND_EXTEND_RISE`.
    pub land_extend: Option<f64>,
    /// `LAND_CONFIRM_XY_DESC`, `LAND_CONFIRM_Z_DESC` [m]: the landing
    /// confirmation's tolerances for descending swings. None = the general ones.
    pub land_confirm_xy: Option<f64>,
    pub land_confirm_z: Option<f64>,
    /// `LEG_REACH_MARGIN` (0.1..1): the fraction of the leg a plan may use.
    pub leg_reach_margin: f64,
}

impl DescentConfig {
    pub fn td_bias_desc(&self, stair: &StairPlanConfig) -> f64 {
        self.td_bias_desc.unwrap_or(stair.td_bias)
    }
}

/// Walking on a known slope.
#[derive(Clone, Debug, PartialEq)]
pub struct SlopeConfig {
    /// `SLOPE_KEEP_STRIDE`: keep the velocity plan's stride on a slope.
    pub keep_stride: bool,
    /// `SLOPE_HEEL_CUT` [m] (non-negative), `SLOPE_HEEL_CUT_MAX` (0..0.08).
    pub heel_cut: f64,
    pub heel_cut_max: f64,
    /// `TERRAIN_PITCH`, `TERRAIN_PITCH_YAW`: pitch the swing foot to the surface.
    pub terrain_pitch: bool,
    pub terrain_pitch_yaw: bool,
}
