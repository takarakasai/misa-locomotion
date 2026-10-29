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
use crate::preview::LandingTiming;
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
    pub swing: SwingConfig,
    pub arm: ArmMomentumConfig,
    pub mom_cmp: MomCmpConfig,
    pub contact: ContactConfig,
    pub managed: ManagedContactConfig,
    pub landing: LandingConfig,
    pub com_plan: ComPlanConfig,
    pub stair_zone: StairZoneConfig,
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
            swing: SwingConfig {
                kp: prof.kp_swing,
                kp_xy: None,
                kd: prof.kd_swing,
                kp_rot: 400.0,
                kd_rot: 40.0,
                lift_h: 0.5 * prof.lift_h,
                rot_mode: 1,
                zone_rot_mode: 0,
                tracking_cubic_ff: false,
                tracking_ff_min_riser: 0.0,
                tracking_ff_scale: 1.0,
                td_sole_frame: false,
                td_sink: 0.004,
                land_sink: 0.004,
                stair_windows: [0.30, 0.25, 0.70, 0.60],
                stair_ff: false,
                step_dz_min: 1e-9,
                absolute_td: false,
                absolute_td_desc: false,
                xy_delay: 0.0,
                z_hold: false,
                z_hold_min_dur: 0.0,
                z_drop: [0.60, 0.95],
            },
            arm: ArmMomentumConfig {
                hold: false,
                hold_joints: "all".into(),
                before_swing: false,
                kp: 400.0,
                kd: 40.0,
                mom: false,
                kp_mom: 10.0,
                mom_axes: [true, false, false],
                mom_level: "swing_after".into(),
                mom_ref: "zero".into(),
                mom_arm_dcm: 0.0,
                mom_arm_dcm_max: 20.0,
            },
            mom_cmp: MomCmpConfig {
                gain: 0.0,
                max: 20.0,
                dead: 0.003,
                w: 1.0,
                sign: 1.0,
                hmax: 0.0,
                kh: 0.0,
            },
            contact: ContactConfig {
                kp: 150.0,
                kd: None,
                anchor_leak: 0.2,
                driven: true,
                ticks: 2,
                on_bw: 0.10,
                load_ramp_frac: 0.6,
                admit_confirmed: false,
                admit_scheduled: false,
                admit_z: 0.010,
                admit_loaded: None,
            },
            managed: ManagedContactConfig {
                on: true,
                tangent: 0.03,
                rotation: 0.05,
                az_max: 15.0,
                rate: 0.0,
                latch: 0.10,
            },
            landing: {
                let lt = LandingTiming::new(0.0);
                LandingConfig {
                    confirm_fz_frac: 0.05,
                    confirm_xy: lt.xy_tol,
                    confirm_z: lt.z_tol,
                    predict_frac: 0.0,
                    extend: 0.25,
                    extend_rise: 0.0,
                    step_dz_min: 0.005,
                    speed_xy: 0.35,
                    speed_z: 0.20,
                    event: ContactEventConfig {
                        on: false,
                        z_tol: 0.003,
                        shift: true,
                        end_on_touch: true,
                        probe_speed: 0.05,
                        probe_max: 0.03,
                    },
                }
            },
            com_plan: ComPlanConfig {
                z_com: None,
                z_com_offset: 0.0,
                replan: false,
                preplan: None,
                preplan_xy_min: f64::INFINITY,
                stair_ss_max: 0.80,
                replan_forward_gain: 1.0,
                replan_dt: 0.02,
                preview_dt: 0.05,
                preview_n: 20,
                preview_smooth: 0.1,
                height_replan: true,
                height_time: 0.60,
                stair_climb_rate: 1.5,
            },
            stair_zone: StairZoneConfig { lead: 0.30, tail: 0.60 },
        }
    }

    /// The landing clock with its knobs set: the confirmation's force gate
    /// (`LAND_CONFIRM_FZ_FRAC` of m g) and tolerances, the deadline slack and
    /// arrival speeds, the event-driven contact, and the descent's z
    /// tolerance. What a swing sets for itself (its xy override, its
    /// extension) the walker sets every tick.
    pub fn landing_timing(&self, total_mass: f64, g: f64) -> LandingTiming {
        let l = &self.landing;
        let mut lt = LandingTiming::new(l.confirm_fz_frac * total_mass * g);
        lt.xy_tol = l.confirm_xy;
        lt.z_tol = l.confirm_z;
        lt.event_mode = l.event.on;
        lt.event_z_tol = l.event.z_tol;
        lt.probe_speed = l.event.probe_speed;
        lt.probe_max = l.event.probe_max;
        lt.min_fraction = l.predict_frac;
        lt.dz_min = l.step_dz_min;
        lt.z_tol_override = self.descent.land_confirm_z;
        lt.max_extension = l.extend;
        lt.speed_z = l.speed_z;
        lt.speed_xy = l.speed_xy;
        lt
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
    /// of a descending swing. None = `COM_LAND_EXTEND_RISE`
    /// ([`DescentConfig::land_extend`]).
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

    pub fn land_extend(&self, landing: &LandingConfig) -> f64 {
        self.land_extend.unwrap_or(landing.extend_rise)
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

/// The swing foot: its gains, its apex, its profile on stairs, and the
/// rough-ground hold (`ROUGH_MODE` sets `LIFT_H` and the three `SWING_*`).
#[derive(Clone, Debug, PartialEq)]
pub struct SwingConfig {
    /// `KP_SWING` (z, profile), `KP_SWING_XY` (None = a quarter of z),
    /// `KD_SWING` (profile).
    pub kp: f64,
    pub kp_xy: Option<f64>,
    pub kd: f64,
    /// `KP_SWING_ROT`, `KD_SWING_ROT`: the swing foot's orientation.
    pub kp_rot: f64,
    pub kd_rot: f64,
    /// `LIFT_H` [m] (half the profile's).
    pub lift_h: f64,
    /// `SWING_ROT` (0 translation, 1 +yaw, 2 full pose, 3 no yaw), and
    /// `STAIR_ZONE_SWING_ROT` (non-negative) inside a stair zone (0 = the same).
    pub rot_mode: u32,
    pub zone_rot_mode: u32,
    /// `TRACKING_CUBIC_FF`: the cubic profile's acceleration feedforward,
    /// on stairs whose riser is at least `TRACKING_FF_MIN_RISER` [m]
    /// (non-negative), scaled by `TRACKING_FF_SCALE`.
    pub tracking_cubic_ff: bool,
    pub tracking_ff_min_riser: f64,
    pub tracking_ff_scale: f64,
    /// `STAIR_TD_SOLE_FRAME`: command the ankle pose the footstep implies.
    pub td_sole_frame: bool,
    /// `STAIR_TD_SINK`, `STAIR_LAND_SINK` (non-negative) [m]: landing
    /// overdrive below the surface.
    pub td_sink: f64,
    pub land_sink: f64,
    /// `STAIR_SWING_WINDOWS` (four numbers, else the default): rise end,
    /// travel start, travel end, drop start.
    pub stair_windows: [f64; 4],
    /// `STAIR_SWING_FF`.
    pub stair_ff: bool,
    /// `STEP_DZ_SWING_MIN` [m] (non-negative): height change above which a
    /// swing takes the stair profile.
    pub step_dz_min: f64,
    /// `STAIR_ABSOLUTE_TD`, `STAIR_ABSOLUTE_TD_DESC`: the surveyed tread
    /// height as the touchdown target (all swings / descending only).
    pub absolute_td: bool,
    pub absolute_td_desc: bool,
    /// `SWING_XY_DELAY` (0..0.6), `SWING_Z_HOLD`, `SWING_Z_HOLD_MIN_DUR`
    /// (0..1), `SWING_Z_DROP` ("a,b": 0.3..0.95, 0.5..1).
    pub xy_delay: f64,
    pub z_hold: bool,
    pub z_hold_min_dur: f64,
    pub z_drop: [f64; 2],
}

impl SwingConfig {
    pub fn kp_xy(&self) -> f64 {
        self.kp_xy.unwrap_or(0.25 * self.kp)
    }
}

/// The arms: their hold, and the centroidal angular-momentum task.
#[derive(Clone, Debug, PartialEq)]
pub struct ArmMomentumConfig {
    /// `ARM_HOLD`, `ARM_HOLD_JOINTS` (all / elbow / shoulder),
    /// `ARM_BEFORE_SWING`, `KP_ARM`, `KD_ARM`.
    pub hold: bool,
    pub hold_joints: String,
    pub before_swing: bool,
    pub kp: f64,
    pub kd: f64,
    /// `MOM`, `KP_MOM`, `MOM_AXES` (roll/pitch/yaw mask, "x--"),
    /// `MOM_LEVEL`, `MOM_REF` (zero / arms_neutral).
    pub mom: bool,
    pub kp_mom: f64,
    pub mom_axes: [bool; 3],
    pub mom_level: String,
    pub mom_ref: String,
    /// `MOM_ARM_DCM`, `MOM_ARM_DCM_MAX` (non-negative): roll momentum asked
    /// of the arms when the lateral DCM escapes.
    pub mom_arm_dcm: f64,
    pub mom_arm_dcm_max: f64,
}

/// MOM_CMP: the hip strategy written as the centroidal moment pivot.
#[derive(Clone, Debug, PartialEq)]
pub struct MomCmpConfig {
    /// `MOM_CMP`, `MOM_CMP_MAX`, `MOM_CMP_DEAD`, `MOM_CMP_W`, `MOM_CMP_HMAX`,
    /// `MOM_CMP_KH` (all non-negative), `MOM_CMP_SIGN`. 0 gain = off.
    pub gain: f64,
    pub max: f64,
    pub dead: f64,
    pub w: f64,
    pub sign: f64,
    pub hmax: f64,
    pub kh: f64,
}

/// The contact model, the measured contact set and the load ramp.
#[derive(Clone, Debug, PartialEq)]
pub struct ContactConfig {
    /// `KP_CONTACT`, `KD_CONTACT` (None = 2 sqrt(kp)): the contact stabiliser.
    pub kp: f64,
    pub kd: Option<f64>,
    /// `ANCHOR_LEAK`.
    pub anchor_leak: f64,
    /// `CONTACT_DRIVEN`, `CONTACT_TICKS`, `CONTACT_ON_BW` (checked by the
    /// harness): solve against the contact set the feet report.
    pub driven: bool,
    pub ticks: u32,
    pub on_bw: f64,
    /// `LOAD_RAMP_FRAC` (0.05..1).
    pub load_ramp_frac: f64,
    /// `CONTACT_ADMIT_CONFIRMED`, `CONTACT_ADMIT_SCHEDULED`,
    /// `CONTACT_ADMIT_Z` (non-negative), `CONTACT_ADMIT_LOADED`
    /// ("z_tol,fz_frac", both non-negative).
    pub admit_confirmed: bool,
    pub admit_scheduled: bool,
    pub admit_z: f64,
    pub admit_loaded: Option<(f64, f64)>,
}

impl ContactConfig {
    pub fn kd(&self) -> f64 {
        self.kd.unwrap_or(2.0 * self.kp.sqrt())
    }
}

/// Managed (soft) contact on a stair landing.
#[derive(Clone, Debug, PartialEq)]
pub struct ManagedContactConfig {
    /// `STAIR_MANAGED_CONTACT`, `_TANGENT` (0..1), `_ROTATION` (0..1),
    /// `_AZ_MAX` (at least 0.1), `_RATE`, `_LATCH` (non-negative).
    pub on: bool,
    pub tangent: f64,
    pub rotation: f64,
    pub az_max: f64,
    pub rate: f64,
    pub latch: f64,
}

/// The landing clock (`LandingTiming`; see [`Config::landing_timing`]).
#[derive(Clone, Debug, PartialEq)]
pub struct LandingConfig {
    /// `LAND_CONFIRM_FZ_FRAC` (non-negative): the confirmation's force gate
    /// in body weights, `LAND_CONFIRM_XY`, `LAND_CONFIRM_Z` [m].
    pub confirm_fz_frac: f64,
    pub confirm_xy: f64,
    pub confirm_z: f64,
    /// `COM_LAND_PREDICT_FRAC` (0..1).
    pub predict_frac: f64,
    /// `COM_LAND_EXTEND`, `COM_LAND_EXTEND_RISE` [s] (non-negative).
    pub extend: f64,
    pub extend_rise: f64,
    /// `STEP_DZ_MIN` [m] (non-negative): height change above which a swing
    /// is a stair swing.
    pub step_dz_min: f64,
    /// `COM_LAND_VXY`, `COM_LAND_VZ` [m/s] (at least 0.01): the swing speeds
    /// the clock and `COM_PREPLAN` assume.
    pub speed_xy: f64,
    pub speed_z: f64,
    pub event: ContactEventConfig,
}

/// Event-driven contact.
#[derive(Clone, Debug, PartialEq)]
pub struct ContactEventConfig {
    /// `CONTACT_EVENT`, `CONTACT_EVENT_Z` [m], `CONTACT_EVENT_SHIFT`,
    /// `CONTACT_EVENT_END_ON_TOUCH`, `CONTACT_PROBE_VZ` [m/s],
    /// `CONTACT_PROBE_MAX` [m] (the numbers non-negative).
    pub on: bool,
    pub z_tol: f64,
    pub shift: bool,
    pub end_on_touch: bool,
    pub probe_speed: f64,
    pub probe_max: f64,
}

/// The CoM: its height, the swing time reserved ahead, and the preview.
#[derive(Clone, Debug, PartialEq)]
pub struct ComPlanConfig {
    /// `Z_COM` [m, world]. None = the initial CoM height + `Z_COM_OFFSET`
    /// ([`ComPlanConfig::z_com`]).
    pub z_com: Option<f64>,
    pub z_com_offset: f64,
    /// `COM_REPLAN`, and `COM_PREPLAN` (None = `COM_REPLAN`).
    pub replan: bool,
    pub preplan: Option<bool>,
    /// `COM_PREPLAN_XY_MIN` [m], `COM_STAIR_SS_MAX` [s] (at least `T_SS`, at
    /// the use).
    pub preplan_xy_min: f64,
    pub stair_ss_max: f64,
    /// `COM_REPLAN_FORWARD_GAIN` (0..1; checked by the harness).
    pub replan_forward_gain: f64,
    /// `COM_REPLAN_DT`, `COM_PREVIEW_DT` [s] (at least the control period,
    /// at the use), `COM_PREVIEW_N` (at least 2), `COM_PREVIEW_SMOOTH`
    /// (non-negative).
    pub replan_dt: f64,
    pub preview_dt: f64,
    pub preview_n: usize,
    pub preview_smooth: f64,
    /// `COM_HEIGHT_REPLAN`, `COM_HEIGHT_TIME` [s] (at least 0.10).
    pub height_replan: bool,
    pub height_time: f64,
    /// `STAIR_CLIMB_RATE` [risers per step].
    pub stair_climb_rate: f64,
}

impl ComPlanConfig {
    pub fn z_com(&self, com0_z: f64) -> f64 {
        self.z_com.unwrap_or(com0_z + self.z_com_offset)
    }

    pub fn preplan(&self) -> bool {
        self.preplan.unwrap_or(self.replan)
    }
}

/// The stair zone: mid-feet x from `start - lead` to `landing_start + tail`
/// (shared by the zone crouch, MOM_CMP's stair exclusion, PUSH_REPLAN and the
/// zone swing mode).
#[derive(Clone, Debug, PartialEq)]
pub struct StairZoneConfig {
    /// `STAIR_ZONE_CROUCH_LEAD`, `STAIR_ZONE_CROUCH_TAIL` [m] (non-negative).
    pub lead: f64,
    pub tail: f64,
}
