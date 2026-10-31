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
//! ([`Config::code_defaults`] takes the [`Profile`]). The two adopted
//! baselines are presets, [`Config::g1_v619`] and [`Config::kyo46rs_v2`]
//! (with the v3 candidate [`Config::kyo46rs_v3`]).
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
    pub contact_plan: ContactPlanConfig,
    pub double_support: DoubleSupportConfig,
    pub mpc: WrenchMpcConfig,
    pub mpc_footstep: MpcFootstepConfig,
    pub mpc_cross: MpcCrossConfig,
    pub push_replan: PushReplanConfig,
    pub push_crouch: PushCrouchConfig,
    pub speed: SpeedScheduleConfig,
    pub recovery: RecoveryRefConfig,
    pub debug: DebugFlags,
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
                mu_assumed: None,
                joint_q_estimate: false,
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
                t_flight: 0.0,
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
                run_bulge: 0.0,
                heel_toe: false,
                heel_strike_deg: 6.0,
                toe_off_deg: 12.0,
                heel_roll_time: 0.10,
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
                walk_vault: 0.0,
            },
            stair_zone: StairZoneConfig { lead: 0.30, tail: 0.60 },
            contact_plan: ContactPlanConfig {
                on: false,
                trig: 0.015,
                t_min_frac: 0.5,
                t_max_frac: 1.0,
                dt_grid: 0.01,
                max_x: 0.06,
                max_y: 0.03,
                max_y_in: None,
                timing_from_x: false,
                timing_lat_outward: false,
                j: 0.5,
                w_shift: 0.0,
                w_u: 1.0,
                w_t: 1.0,
                w_res: 100.0,
                dead_u: 0.002,
                dead_t: 0.005,
            },
            double_support: DoubleSupportConfig {
                feasible: false,
                feasible_on_timeout: false,
                feasible_continuous: false,
                feasible_ramp: false,
                feasible_max: 1.5,
                feasible_box_stance: false,
                capture: false,
                capture_hopeless: 1.5,
                capture_min_gain: 0.05,
                capture_fwd_only: true,
                capture_margin: 0.01,
                capture_t_min: 0.15,
                capture_load: 0.4,
            },
            mpc: WrenchMpcConfig {
                enabled: false,
                always: false,
                apply: false,
                coupled: true,
                node_centre: false,
                net_cop_w: 0.0,
                seed_previous: 0,
                w_pos: 20.0,
                w_vel: 4.0,
                w_mom: 2.0,
                w_acc: 6.0,
                w_wrench: 2e-4,
                w_smooth: 1e-3,
                sole_authority: false,
                dt: 0.02,
                horizon: 8,
                preview_dt: 0.05,
                low_min: 0.12,
                blend: 0.25,
                hybrid_contact: true,
                backend: None,
                kkt_comp_tol: 1e-6,
                bench_gate: false,
            },
            mpc_footstep: MpcFootstepConfig {
                on: false,
                w: 100.0,
                moment: true,
                trig: 0.0,
                max: [0.06, 0.03],
                max_y_in: f64::INFINITY,
                y_in_fwd: 0.0,
                y_in_fwd_trig: 0.005,
                capture_w: 0.0,
                capture_j: false,
                lex: false,
            },
            mpc_cross: MpcCrossConfig { y: 0.0, ex: 0.02, floor: 0.0, hold_z: 0.0, hold_xy: 0.02 },
            push_replan: PushReplanConfig {
                trig: 0.0,
                emin: 0.015,
                freeze: 0.85,
                stand: 2,
                ramp: 2,
                max_fwd: 0.35,
                max_back: 0.30,
                min_w: 0.12,
                max_w: 0.40,
                hold: 0.06,
                back_only: false,
                vmax: f64::INFINITY,
                ankle: [0.0, 1.0],
                slope_lead: 0.0,
            },
            push_crouch: PushCrouchConfig { drop: 0.0, trig: 0.03, t: 1.5, lat_max: 0.02 },
            speed: SpeedScheduleConfig {
                sched: false,
                tscale_min: 0.4,
                stride_max: 0.45,
                cap_slope: 2.0,
                v_long: 0.45,
                z_slope: 0.36,
                z_max: 0.08,
                z_min: 0.0,
                yield_v: 0.0,
                yield_trig: 0.03,
                yield_no_fwd: false,
                yield_vmin: 0.0,
            },
            recovery: RecoveryRefConfig {
                trig_dcm: 0.0,
                trig_max_frac: 1.0,
                delay_comp: true,
                swing: true,
                reset_stance: true,
                reset_incl_current: false,
                cooldown: 1.0,
            },
            debug: DebugFlags::default(),
        }
    }

    /// The G1 champion v6.19: the code defaults of [`crate::profile::G1_23DOF`]
    /// with the control keys of `log/claude_g1champion_20261029v619/controls.json`.
    /// The rest of that file is the rig (`URDF`, `SOLE_HALF_L`, `SIM_DT`,
    /// `QD_CLAMP`, `ARMATURE`, `DAMPING_OVERRIDES`, the seeds, the sensor
    /// model), the scenario (`T`, `STAIR_LANDING_LEN`), the command
    /// (`STRIDE`'s speed), the reference library and keys no longer read.
    /// Checked against that file by crates/misa-biped/tests/presets.rs.
    pub fn g1_v619() -> Self {
        let mut c = Self::code_defaults(&crate::profile::G1_23DOF);
        c.wbc.feas_tol = 1e-8;
        c.wbc.strategy = misa_wbc::HqpStrategy::WeightedQp { level_ratio: 1 };
        c.wbc.fallback_level = 3;
        c.wbc.level_weights.set("com", 100.0);
        c.wbc.level_weights.set("posture", 0.001);
        c.wbc.ctrl_dt = 0.005;
        c.joint_limit.enabled = true;
        c.joint_limit.alpha = [25.6107, 10.0, 20.6833];
        c.joint_limit.a_max = 351.019;
        c.gains.kp_com = 419.064;
        c.gains.kd_com = 48.1952;
        c.gains.kp_trunk = 73.0103;
        c.gains.kd_trunk = 28.2069;
        c.dcm.cop_frac = 0.912835;
        c.dcm.k_dcm = 3.09069;
        c.dcm.vh_dcm = true;
        c.gait.t_ds = 0.365543;
        c.gait.t_ss = 0.843964;
        c.gait.n_steps = 28;
        c.gait.ds_settle = 0.0840506;
        c.gait.stride = 0.148842;
        c.gait.stride_ramp = 3; // STRIDE_RAMP=3.15826, read as a whole number of steps
        c.stair_plan.toe_margin = 0.0;
        c.stair_plan.y_out = 0.00442314;
        c.stair_plan.td_bias = 0.000780737;
        c.stair_plan.td_bias_asc_mid = Some(0.02);
        c.stair_plan.landing_frac = 0.942684;
        c.stair_plan.align_stand_line = true;
        c.descent.td_bias_desc = Some(0.02);
        c.descent.com_height_lead = 0.5;
        c.descent.zone_crouch = 0.04;
        c.descent.land_extend = Some(0.25);
        c.descent.land_confirm_xy = Some(0.03);
        c.descent.land_confirm_z = Some(0.003);
        c.slope.keep_stride = true;
        c.slope.heel_cut = 0.010;
        c.slope.terrain_pitch_yaw = true;
        c.swing.kp = 477.044;
        c.swing.kd = 31.493;
        c.swing.lift_h = 0.0512879;
        c.swing.tracking_cubic_ff = true;
        c.swing.tracking_ff_min_riser = 0.10;
        c.swing.td_sole_frame = true;
        c.arm.hold = true;
        c.arm.before_swing = true;
        c.arm.kp = 400.0;
        c.arm.kd = 40.0;
        c.arm.mom = true;
        c.arm.kp_mom = 77.0372;
        c.arm.mom_ref = "arms_neutral".into();
        c.arm.mom_arm_dcm = 27.4033;
        c.mom_cmp.gain = 1.0;
        c.mom_cmp.max = 20.0;
        c.mom_cmp.sign = -1.0;
        c.mom_cmp.hmax = 1.0;
        c.mom_cmp.kh = 2.0;
        c.contact.admit_confirmed = true;
        c.contact.admit_scheduled = true;
        c.contact.admit_loaded = Some((0.04, 0.02));
        c.landing.extend = 0.25;
        c.landing.extend_rise = 1.08;
        c.landing.speed_xy = 0.0776444;
        c.landing.speed_z = 0.05;
        c.com_plan.replan = true;
        c.contact_plan.on = true;
        c.contact_plan.max_x = 0.0;
        c.contact_plan.max_y = 0.0;
        c.contact_plan.max_y_in = Some(0.0);
        c.contact_plan.timing_from_x = true;
        c.contact_plan.timing_lat_outward = true;
        c.double_support.feasible = true;
        c.double_support.feasible_on_timeout = true;
        c.double_support.feasible_continuous = true;
        c.double_support.feasible_ramp = true;
        c.double_support.capture = true;
        c.double_support.capture_margin = 0.005;
        c.mpc.enabled = true;
        c.mpc.always = true;
        c.mpc.apply = true;
        c.mpc.coupled = false;
        c.mpc.node_centre = true;
        c.mpc.net_cop_w = 10.0;
        c.mpc.seed_previous = 2;
        c.mpc.w_smooth = 10.0;
        c.mpc.sole_authority = true;
        c.mpc.horizon = 8;
        c.mpc_footstep.on = true;
        c.mpc_footstep.w = 100.0;
        c.mpc_footstep.trig = 0.015;
        c.mpc_footstep.max = [0.15, 0.08];
        c.mpc_footstep.max_y_in = 0.0;
        c.mpc_footstep.y_in_fwd = 0.035;
        c.mpc_footstep.y_in_fwd_trig = 0.005;
        c.mpc_footstep.capture_w = 1000000.0;
        c.mpc_footstep.capture_j = true;
        c.mpc_footstep.lex = true;
        c.mpc_cross.y = 0.20;
        c.mpc_cross.ex = 0.03;
        c.mpc_cross.floor = 0.5;
        c.mpc_cross.hold_z = 0.05;
        c.push_replan.trig = 0.03;
        c.push_replan.back_only = true;
        c.push_replan.vmax = 0.15;
        c.push_replan.hold = 10.0; // in effect no hold (see §7 of the extraction plan)
        c.push_replan.slope_lead = 0.6;
        c.push_crouch.drop = 0.035;
        c.speed.z_min = 0.06;
        c.speed.yield_v = 0.20;
        c.speed.yield_no_fwd = true;
        c.speed.yield_vmin = 0.35;
        c.recovery.trig_dcm = 0.036;
        c.recovery.trig_max_frac = 0.65;
        c.recovery.delay_comp = false;
        c.recovery.reset_incl_current = true;
        c
    }

    /// kyo46rs base v2: the code defaults of [`crate::profile::KYO46RS`] with
    /// the control keys of `log/claude_g1rand_20260925/kyo_base_v2.json`
    /// (the rest of it is the rig, `T`, the command `VX` and keys no longer
    /// read). Checked against that file by crates/misa-biped/tests/presets.rs.
    pub fn kyo46rs_v2() -> Self {
        let mut c = Self::code_defaults(&crate::profile::KYO46RS);
        c.wbc.feas_tol = 1e-8;
        c.wbc.fallback_level = 3;
        c.wbc.ctrl_dt = 0.005;
        c.joint_limit.enabled = true;
        c.joint_limit.a_max = 200.0;
        c.dcm.k_dcm = 2.0;
        c.dcm.zmp_roll = Some(0.02107);
        c.dcm.com_xy_jerk_max = 100.0;
        c.gait.t_ds = 0.3384;
        c.gait.t_ss = 0.45;
        c.gait.n_steps = 40;
        c.gait.stride_ramp = 6;
        c.stair_plan.gait_pair = true;
        c.stair_plan.td_bias = 0.0;
        c.stair_plan.full_sole = true;
        c.stair_plan.sole_margin = 0.002;
        c.swing.kp_xy = Some(400.0);
        c.swing.lift_h = 0.03;
        c.swing.rot_mode = 1;
        c.swing.tracking_cubic_ff = false;
        c.swing.tracking_ff_scale = 1.0;
        c.swing.td_sink = 0.009;
        c.swing.stair_windows = [0.50, 0.38, 0.80, 0.70];
        c.swing.stair_ff = true;
        c.swing.absolute_td = false;
        c.arm.hold = true;
        c.arm.hold_joints = "shoulder".into();
        c.arm.kp = 200.0;
        c.arm.kd = 20.0;
        c.contact.on_bw = 0.05;
        c.contact.admit_scheduled = true;
        c.landing.predict_frac = 1.0;
        c.landing.extend = 0.10;
        c.landing.speed_xy = 0.05;
        c.landing.speed_z = 0.05;
        c.com_plan.replan = true;
        c.com_plan.preplan = Some(true);
        c.com_plan.preplan_xy_min = 0.075;
        c.com_plan.stair_ss_max = 0.7254;
        c
    }

    /// kyo46rs base v3 candidate (2026-10-20, doc/from_claude/20261018_kyo46rs_port_scoping.md
    /// §11): [`Config::kyo46rs_v2`] with the five stair-zone keys of
    /// `log/claude_g1rand_20260925/kyo_base_v3.json` (`STAIR_SWING_FF=0`,
    /// `STAIR_ABSOLUTE_TD_DESC=1`, `STAIR_ALIGN_STAND_LINE=1`,
    /// `STAIR_ZONE_CROUCH=0.02`, `STAIR_ZONE_SWING_ROT=3`). Off the stair zone
    /// it walks as v2 does. Checked against that file by
    /// crates/misa-biped/tests/presets.rs.
    pub fn kyo46rs_v3() -> Self {
        let mut c = Self::kyo46rs_v2();
        c.swing.stair_ff = false;
        c.swing.absolute_td_desc = true;
        c.stair_plan.align_stand_line = true;
        c.descent.zone_crouch = 0.02;
        c.swing.zone_rot_mode = 3;
        c
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
        lt.debug = self.debug.land_timing;
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
    /// `friction_mu`. None = the floor friction (`mu_assumed`, or the
    /// plant's `MU_GROUND`) x the profile's friction margin
    /// ([`WbcConfig::friction_mu`]).
    pub friction_mu: Option<f64>,
    /// `MU_ASSUMED`: the floor friction the controller ASSUMES, in place of
    /// the plant's `MU_GROUND` (which the benches randomise per member).
    /// None = the plant's value (the LEAK of §6.7 of
    /// doc/from_claude/20261029_walker_extraction_plan.md, bit-identical).
    /// The harness's nominal floor is 0.70 (its `MU_GROUND` default).
    pub mu_assumed: Option<f64>,
    /// `JOINT_Q_EST`: the posture and arm-hold levels read the ESTIMATE's
    /// joint angles (`StateEstimate::joint_q`, with the sensor model's delay
    /// and noise) instead of the plant's (`Observation::joint_pos_plant`,
    /// the LEAK of §6.6). The harness reads the same switch for the
    /// start-up FK and the degraded-solve fallback. false = the plant's
    /// (bit-identical); with no delay and no joint noise the two are equal.
    pub joint_q_estimate: bool,
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

    /// The friction the QP plans against: `friction_mu`, or the floor's
    /// friction times the profile's margin -- the assumed floor
    /// (`mu_assumed`) when set, else the plant's `mu_ground` (LEAK).
    pub fn friction_mu(&self, mu_ground: f64, friction_margin: f64) -> f64 {
        self.friction_mu.unwrap_or(self.mu_assumed.unwrap_or(mu_ground) * friction_margin)
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
    /// Flight between steps (running), both feet off the ground; 0 = walk.
    pub t_flight: f64,
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
    /// `RUN_SWING_BULGE`: running only, the airborne foot bows outward by
    /// this much mid-swing (sin² profile) to clear the stance leg [m].
    pub run_bulge: f64,
    /// `HEEL_TOE`: walking with heel strike and toe-off -- the landing
    /// foot comes down toe-up (`HEEL_STRIKE_DEG`) on its heel edge and rolls
    /// flat over `HEEL_ROLL_TIME` [s]; the rear foot rolls onto its toe edge
    /// through the double support, heel rising to `TOE_OFF_DEG`.
    pub heel_toe: bool,
    pub heel_strike_deg: f64,
    pub toe_off_deg: f64,
    pub heel_roll_time: f64,
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
    /// `WALK_VAULT` [m]: walking only, the CoM rises by this much over each
    /// single support (sin² of the slice fraction) -- an inverted-pendulum
    /// arc over a straightening stance leg instead of a constant height.
    pub walk_vault: f64,
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

/// The contact planner (P2-1, `contact_plan`): when the current single
/// support ends and where it lands, once the DCM is off the plan.
#[derive(Clone, Debug, PartialEq)]
pub struct ContactPlanConfig {
    /// `CONTACT_PLAN`, and `CP_TRIG` [m]: plan only above this DCM error.
    pub on: bool,
    pub trig: f64,
    /// `CP_T_MIN`, `CP_T_MAX` (1 = never extend the single support),
    /// `CP_DT` (at least 1 ms).
    pub t_min_frac: f64,
    pub t_max_frac: f64,
    pub dt_grid: f64,
    /// `CP_MAX_X`, `CP_MAX_Y`, `CP_MAX_Y_IN` (None = `CP_MAX_Y`). All three
    /// zero make the planner a pure step-retimer.
    pub max_x: f64,
    pub max_y: f64,
    pub max_y_in: Option<f64>,
    /// `CP_T_FROM_X`, `CP_T_LAT_OUT`.
    pub timing_from_x: bool,
    pub timing_lat_outward: bool,
    /// `CP_J` (at least 0.05), `CP_W_SHIFT` (non-negative), `CP_W_U`,
    /// `CP_W_T`, `CP_W_RES`, `CP_DEAD_U`, `CP_DEAD_T`.
    pub j: f64,
    pub w_shift: f64,
    pub w_u: f64,
    pub w_t: f64,
    pub w_res: f64,
    pub dead_u: f64,
    pub dead_t: f64,
}

impl ContactPlanConfig {
    /// The planner's own tunables.
    pub fn planner(&self) -> crate::contact_plan::Config {
        crate::contact_plan::Config {
            t_min_frac: self.t_min_frac,
            t_max_frac: self.t_max_frac,
            dt_grid: self.dt_grid,
            max_x: self.max_x,
            max_y: self.max_y,
            max_y_in: self.max_y_in.unwrap_or(self.max_y),
            timing_from_x: self.timing_from_x,
            timing_lat_outward: self.timing_lat_outward,
            j: self.j,
            w_shift: self.w_shift,
            w_u: self.w_u,
            w_t: self.w_t,
            w_res: self.w_res,
            dead_u: self.dead_u,
            dead_t: self.dead_t,
        }
    }
}

/// The double support after a landing: DS_FEASIBLE sizes it from the
/// measured DCM, DS_CAPTURE ends it early when the DCM has left its box.
#[derive(Clone, Debug, PartialEq)]
pub struct DoubleSupportConfig {
    /// `DS_FEASIBLE`, `DS_FEASIBLE_ON_TIMEOUT`, `DS_FEASIBLE_CONTINUOUS`,
    /// `DS_FEASIBLE_RAMP`, `DS_FEASIBLE_MAX` [s], `DS_FEASIBLE_BOX=stance`
    /// (the old stance sole as the pivot box).
    pub feasible: bool,
    pub feasible_on_timeout: bool,
    pub feasible_continuous: bool,
    pub feasible_ramp: bool,
    pub feasible_max: f64,
    pub feasible_box_stance: bool,
    /// `DS_CAPTURE`, `_HOPELESS` (at least 1), `_MIN_GAIN` [s]
    /// (non-negative), `_FWD_ONLY`, `_MARGIN` [m], `_T_MIN` [s] (at least
    /// 0.02), `_LOAD` (0..1).
    pub capture: bool,
    pub capture_hopeless: f64,
    pub capture_min_gain: f64,
    pub capture_fwd_only: bool,
    pub capture_margin: f64,
    pub capture_t_min: f64,
    pub capture_load: f64,
}

/// The per-foot wrench MPC.
#[derive(Clone, Debug, PartialEq)]
pub struct WrenchMpcConfig {
    /// `STAIR_WRENCH_MPC`, `WRENCH_MPC_ALWAYS`, `WRENCH_MPC_APPLY` (only with
    /// ALWAYS: [`WrenchMpcConfig::applies`]), `STAIR_WRENCH_COUPLED`.
    pub enabled: bool,
    pub always: bool,
    pub apply: bool,
    pub coupled: bool,
    /// `MPC_NODE_CENTRE`, `MPC_NET_COP_W` (non-negative),
    /// `MPC_SEED_PREVIOUS` (0 off, 1 seed the first solve, 2 every solve).
    pub node_centre: bool,
    pub net_cop_w: f64,
    pub seed_previous: i32,
    /// `MPC_W_POS`, `_VEL`, `_MOM`, `_ACC`, `_WRENCH`, `_SMOOTH` (non-negative).
    pub w_pos: f64,
    pub w_vel: f64,
    pub w_mom: f64,
    pub w_acc: f64,
    pub w_wrench: f64,
    pub w_smooth: f64,
    /// `MPC_SOLE_AUTHORITY`.
    pub sole_authority: bool,
    /// `STAIR_WRENCH_DT`, `STAIR_WRENCH_PREVIEW_DT` [s] (at least the control
    /// period, at the use), `STAIR_WRENCH_N` (at least 2),
    /// `STAIR_WRENCH_LOW_MIN` (0..0.45), `STAIR_WRENCH_BLEND` (0..1).
    pub dt: f64,
    pub horizon: usize,
    pub preview_dt: f64,
    pub low_min: f64,
    pub blend: f64,
    /// `STAIR_HYBRID_CONTACT`: the measured contact set in the horizon and WBC.
    pub hybrid_contact: bool,
    /// `MPC_QP_BACKEND`, `MPC_KKT_COMP_TOL`, `MPC_BENCH_GATE`.
    pub backend: Option<misa_wbc::QpSolver>,
    pub kkt_comp_tol: f64,
    pub bench_gate: bool,
}

impl WrenchMpcConfig {
    /// Whether the MPC's answer reaches the WBC outside a stair transfer
    /// (`WRENCH_MPC_APPLY` counts only with `WRENCH_MPC_ALWAYS`).
    pub fn applies(&self) -> bool {
        self.apply && self.always
    }
}

/// MPC_FOOTSTEP: the swing foot's landing as a variable of the wrench MPC.
#[derive(Clone, Debug, PartialEq)]
pub struct MpcFootstepConfig {
    /// `MPC_FOOTSTEP`, `MPC_FOOTSTEP_W`, `MPC_FOOTSTEP_MOMENT`,
    /// `MPC_FOOTSTEP_TRIG` [m] (the numbers non-negative).
    pub on: bool,
    pub w: f64,
    pub moment: bool,
    pub trig: f64,
    /// `MPC_FOOTSTEP_MAX_X`, `_MAX_Y`, `_MAX_Y_IN` [m] (non-negative).
    pub max: [f64; 2],
    pub max_y_in: f64,
    /// `MPC_FOOTSTEP_Y_IN_FWD` [m] (non-negative) while the sagittal DCM
    /// error is forward by `MPC_FOOTSTEP_Y_IN_FWD_TRIG`.
    pub y_in_fwd: f64,
    pub y_in_fwd_trig: f64,
    /// `MPC_LANDING_CAPTURE_W` (non-negative), `MPC_LANDING_CAPTURE_J`,
    /// `MPC_LANDING_LEX`.
    pub capture_w: f64,
    pub capture_j: bool,
    pub lex: bool,
}

/// The MPC landing crossing the stance foot.
#[derive(Clone, Debug, PartialEq)]
pub struct MpcCrossConfig {
    /// `MPC_FOOTSTEP_CROSS_Y`, `_EX`, `_FLOOR`, `_HOLD_Z`, `_HOLD_XY` (all
    /// but `_EX` non-negative). 0 `y` = off.
    pub y: f64,
    pub ex: f64,
    pub floor: f64,
    pub hold_z: f64,
    pub hold_xy: f64,
}

/// PUSH_REPLAN: one re-planner for the step after a push.
#[derive(Clone, Debug, PartialEq)]
pub struct PushReplanConfig {
    /// `PUSH_REPLAN` [m] (0 = off), `PUSH_REPLAN_EMIN`, `PUSH_REPLAN_FREEZE`.
    pub trig: f64,
    pub emin: f64,
    pub freeze: f64,
    /// `PUSH_REPLAN_STAND`, `PUSH_REPLAN_RAMP` [steps].
    pub stand: usize,
    pub ramp: usize,
    /// `PUSH_REPLAN_MAX_FWD`, `_MAX_BACK`, `_MIN_W`, `_MAX_W`, `_HOLD` [m]
    /// (non-negative).
    pub max_fwd: f64,
    pub max_back: f64,
    pub min_w: f64,
    pub max_w: f64,
    pub hold: f64,
    /// `PUSH_REPLAN_BACK_ONLY`, `PUSH_REPLAN_VMAX` [m/s] (non-negative).
    pub back_only: bool,
    pub vmax: f64,
    /// `PUSH_REPLAN_ANKLE` ("ax,ay", each 0..1).
    pub ankle: [f64; 2],
    /// `PUSH_REPLAN_SLOPE_LEAD` [m] (non-negative).
    pub slope_lead: f64,
}

/// PUSH_Z_DROP: lower the CoM after a detected push.
#[derive(Clone, Debug, PartialEq)]
pub struct PushCrouchConfig {
    /// `PUSH_Z_DROP` [m] (0 = off), `_TRIG` [m], `_T` [s], `_LAT_MAX` [m]
    /// (all non-negative).
    pub drop: f64,
    pub trig: f64,
    pub t: f64,
    pub lat_max: f64,
}

/// The speed schedule under a `VX2` command, and PUSH_YIELD.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeedScheduleConfig {
    /// `VX2_SCHED`, `VX2_TSCALE_MIN` (0.1..1).
    pub sched: bool,
    pub tscale_min: f64,
    /// `VXS_STRIDE_MAX` (at least 0.05), `VXS_CAP_SLOPE`, `VXS_V_LONG`:
    /// [`SpeedScheduleConfig::stride_cap`].
    pub stride_max: f64,
    pub cap_slope: f64,
    pub v_long: f64,
    /// `VXS_Z_SLOPE` (non-negative), `VXS_Z_MAX`, `VXS_Z_MIN` (non-negative):
    /// the scheduled CoM drop.
    pub z_slope: f64,
    pub z_max: f64,
    pub z_min: f64,
    /// `PUSH_YIELD_V`, `_TRIG`, `_VMIN` (non-negative), `_NO_FWD`.
    pub yield_v: f64,
    pub yield_trig: f64,
    pub yield_no_fwd: bool,
    pub yield_vmin: f64,
}

impl SpeedScheduleConfig {
    /// The stride cap at speed `v`: 0.35 m up to `v_long`, then rising by
    /// `cap_slope` per m/s to `stride_max`.
    pub fn stride_cap(&self, v: f64) -> f64 {
        (0.35 + self.cap_slope * (v - self.v_long)).clamp(0.35, self.stride_max)
    }
}

/// Push-recovery references (the library itself is the harness's to load).
#[derive(Clone, Debug, PartialEq)]
pub struct RecoveryRefConfig {
    /// `REF_TRIG_DCM` [m] (non-negative; 0 = off), `REF_TRIG_MAX_FRAC`,
    /// `REF_DELAY_COMP`.
    pub trig_dcm: f64,
    pub trig_max_frac: f64,
    pub delay_comp: bool,
    /// `REF_SWING`, `REF_RESET_STANCE`, `REF_RESET_INCL_CURRENT`,
    /// `REF_COOLDOWN` [s] (non-negative).
    pub swing: bool,
    pub reset_stance: bool,
    pub reset_incl_current: bool,
    pub cooldown: f64,
}

/// The control loop's debug prints. All off by default.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DebugFlags {
    /// `CONTACT_ADMIT_DEBUG` (set): the `[adm]` lines (admit certificates).
    pub contact_admit: bool,
    /// `DS_FEASIBLE_DEBUG` (set): the `[ds]`/`[dsf]` lines.
    pub ds_feasible: bool,
    /// `DS_CAPTURE_DEBUG` (set): the `[dsc-cap]` lines.
    pub ds_capture: bool,
    /// `DEBUG_ANCHOR` (a switch: "0" = off): the `[anchor]` lines.
    pub anchor: bool,
    /// `CP_DEBUG` (set): the `[cp]` lines (contact_plan).
    pub cp: bool,
    /// `PUSH_REPLAN_DEBUG` (set): the `[replan]` lines.
    pub push_replan: bool,
    /// `MPC_FS_CROSS_DEBUG` (set): the `[fs-cross]` lines (the crossing gate).
    pub mpc_fs_cross: bool,
    /// `MPC_FS_DEBUG_LANDING` (set): the `[fs-decision]` lines.
    pub mpc_fs_landing: bool,
    /// `MPC_TRAJ_DEBUG` (set): the `[traj]` lines (before t = 2 s).
    pub mpc_traj: bool,
    /// `DIAG_REPLAN` (a switch): print each live replan with its site.
    pub replan: bool,
    /// `WBC_LEVEL_DEBUG_T=<t>`: the `[wbc-levels]` lines within 11 ms of t.
    /// Unparsable = None (silently off).
    pub wbc_level_t: Option<f64>,
    /// `LAND_TIMING_DEBUG` (set): the landing clock's `[lt]` lines
    /// (`LandingTiming::debug`).
    pub land_timing: bool,
}
