//! Key/value -> [`Config`]: the walker's knobs by name.
//!
//! [`crate::config`] holds the knobs and their code defaults; this module is
//! the table from key to field, with each read's own post-processing
//! (`.max`, `.clamp`, `as usize`) next to it. `get` is any lookup -- a host
//! passes `|k| std::env::var(k).ok()` to read its environment, the preset
//! tests (tests/presets.rs) a JSON of the same keys -- so the library itself
//! reads no environment. Keys the walker does not own (the rig, the scenario,
//! the command, the diagnostics) are the host's.
//!
//! The lookups match a harness's `env_f64` / `flag`: a number that is unset
//! or does not parse is the default, and a switch is on when set to anything
//! but "0".
use crate::config::*;
use crate::profile::Profile;

/// `ROUGH_MODE=1`: the rough-terrain preset for an operator who knows the
/// ground is uneven. Lift fully before advancing (`SWING_Z_HOLD` with the
/// drop starting at 0.50 so early landings still work), a 0.15 start delay,
/// and an 8 cm apex. Measured 2026-10-16/17 (N=48, v6.10): unmodelled bumps
/// 10/20/30 mm = 48/46/48 (champion 35/13/5), but the lateral pushes at
/// 0.12 m/s go 157 -> 150, so it is not the default.
///
/// The mode WINS over the four knobs (the champion's controls carry a
/// CEM-tuned `LIFT_H` of 0.051, which the mode must replace): these are the
/// values [`config_from_kv`] reads for them, as the strings the harness used
/// to put into the environment, so they go through the same parse and clamp.
/// The apex is `ROUGH_MODE_LIFT_H` -- on a KNOWN slope (`TERRAIN_SLOPE` set)
/// 6 cm by default, since the 8 cm apex falls on +3 deg (measured
/// 2026-10-17, `LIFT_H=0.08` alone: 16.0 s) and 6 cm walks; 8 cm elsewhere.
pub fn rough_mode(get: &dyn Fn(&str) -> Option<String>) -> Option<[(&'static str, String); 4]> {
    if !get("ROUGH_MODE").map(|v| v == "1").unwrap_or(false) {
        return None;
    }
    let lift = get("ROUGH_MODE_LIFT_H")
        .unwrap_or_else(|| if get("TERRAIN_SLOPE").is_some() { "0.06".into() } else { "0.08".into() });
    Some([
        ("SWING_Z_HOLD", "1".into()),
        ("SWING_Z_DROP", "0.50,1.0".into()),
        ("SWING_XY_DELAY", "0.15".into()),
        ("LIFT_H", lift),
    ])
}

/// The robot's profile with its sole half-size read over it: `SOLE_HALF_L` /
/// `SOLE_HALF_W` [m] replace `Profile::cop_half` (fore/aft, lateral), the
/// sole the walker plans, clamps its CoP box and builds its contact model
/// from. Unset = the profile's own.
pub fn profile_from_kv(mut prof: Profile, get: &dyn Fn(&str) -> Option<String>) -> Profile {
    let num = |k: &str, d: f64| -> f64 { get(k).and_then(|s| s.parse().ok()).unwrap_or(d) };
    prof.cop_half = (num("SOLE_HALF_L", prof.cop_half.0), num("SOLE_HALF_W", prof.cop_half.1));
    prof
}

/// The Config a run starts from (S17): `WALKER_PRESET=g1_v619`, `kyo46rs_v2`
/// or `kyo46rs_v3` names a preset ([`Config::g1_v619`], [`Config::kyo46rs_v2`],
/// [`Config::kyo46rs_v3`]),
/// unset is [`Config::code_defaults`] of the robot. A preset is made for one
/// robot; naming it for another panics (its profile-derived values would be
/// the wrong robot's).
pub fn base_config(prof: &Profile, get: &dyn Fn(&str) -> Option<String>) -> Config {
    match get("WALKER_PRESET").as_deref() {
        None => Config::code_defaults(prof),
        Some("g1_v619") => {
            assert_eq!(prof.name, crate::profile::G1_23DOF.name, "WALKER_PRESET=g1_v619 is the G1 preset (ROBOT=g1)");
            Config::g1_v619()
        }
        Some("kyo46rs_v2") => {
            assert_eq!(prof.name, crate::profile::KYO46RS.name, "WALKER_PRESET=kyo46rs_v2 is the kyo46rs preset (ROBOT unset or kyo46rs)");
            Config::kyo46rs_v2()
        }
        Some("kyo46rs_v3") => {
            assert_eq!(prof.name, crate::profile::KYO46RS.name, "WALKER_PRESET=kyo46rs_v3 is the kyo46rs preset (ROBOT unset or kyo46rs)");
            Config::kyo46rs_v3()
        }
        Some(other) => panic!("unknown WALKER_PRESET={other:?} (g1_v619 | kyo46rs_v2 | kyo46rs_v3)"),
    }
}

/// Every walker knob from `get` (`|k| std::env::var(k).ok()` for the
/// environment), over the base [`base_config`] picks (`WALKER_PRESET`, or
/// the code defaults), with [`rough_mode`]'s values over its four knobs. A
/// key that is unset (or does not parse) keeps the base's value, so with no
/// preset this is the code defaults' table it always was. Panics on an
/// unsupported `WBC_STRATEGY` / `WBC_QP_BACKEND` / `MPC_QP_BACKEND` and on a
/// preset named for another robot.
pub fn config_from_kv(prof: &Profile, get: &dyn Fn(&str) -> Option<String>) -> Config {
    let rough = rough_mode(get);
    let get = |k: &str| -> Option<String> {
        match rough.as_ref().and_then(|r| r.iter().find(|(n, _)| *n == k)) {
            Some((_, v)) => Some(v.clone()),
            None => get(k),
        }
    };
    let num = |k: &str, d: f64| -> f64 { get(k).and_then(|s| s.parse().ok()).unwrap_or(d) };
    let flag = |k: &str, d: bool| -> bool { get(k).map(|v| v != "0").unwrap_or(d) };
    let opt = |k: &str| -> Option<f64> { get(k).and_then(|v| v.parse().ok()) };
    let d = base_config(prof, &get);

    let wbc = {
        let d = &d.wbc;
        let mut level_weights = d.level_weights.clone();
        for name in LevelWeights::NAMES {
            level_weights.set(name, num(&LevelWeights::key(name), d.level_weights.get(name).unwrap_or(LevelWeights::default_for(name))));
        }
        WbcConfig {
            feas_tol: num("WBC_FEAS_TOL", d.feas_tol),
            strategy: match get("WBC_STRATEGY").as_deref() {
                // Per-task weights are applied to the levels themselves
                // (WBCW_*); the ratio defaults to 1 so they are not scaled
                // again by position.
                Some("weighted") => misa_wbc::HqpStrategy::WeightedQp {
                    level_ratio: num("WBC_WEIGHT_RATIO", 1.0).max(1.0) as u32,
                },
                Some("force_budget") => misa_wbc::HqpStrategy::ForceBudgetCascade,
                Some("null_space") => misa_wbc::HqpStrategy::NullSpace,
                None => d.strategy,
                Some(other) => panic!("unsupported WBC_STRATEGY: {other}"),
            },
            backend: get("WBC_QP_BACKEND").map(|name| match name.as_str() {
                "clarabel" => misa_wbc::QpSolver::Clarabel,
                "active_set" => misa_wbc::QpSolver::ActiveSet,
                "proxqp" => misa_wbc::QpSolver::Proxqp,
                "eiquadprog" => misa_wbc::QpSolver::Eiquadprog,
                "qpoases" => misa_wbc::QpSolver::Qpoases,
                _ => panic!("unsupported WBC_QP_BACKEND: {name}"),
            }).or(d.backend),
            bench_gate: flag("QP_BENCH_GATE", d.bench_gate),
            friction_mu: opt("friction_mu").or(d.friction_mu),
            mu_assumed: opt("MU_ASSUMED").or(d.mu_assumed),
            joint_q_estimate: flag("JOINT_Q_EST", d.joint_q_estimate),
            f_max_scale: num("F_MAX_SCALE", d.f_max_scale),
            f_max: opt("F_MAX").or(d.f_max),
            eom_hard: flag("EOM_HARD", d.eom_hard),
            hold_kp: num("HOLD_KP", d.hold_kp),
            hold_kd: num("HOLD_KD", d.hold_kd),
            hold_bridge: num("HOLD_BRIDGE", d.hold_bridge as f64) as u32,
            fallback_level: num("FALLBACK_LEVEL", d.fallback_level as f64) as usize,
            level_weights,
            ctrl_dt: num("CTRL_DT", d.ctrl_dt),
        }
    };

    let joint_limit = {
        let d = &d.joint_limit;
        JointLimitConfig {
            enabled: flag("JLIM", d.enabled),
            alpha: [
                num("JLIM_ALPHA1", d.alpha[0]),
                num("JLIM_ALPHA2", d.alpha[1]),
                num("JLIM_ALPHA3", d.alpha[2]),
            ],
            a_max: num("JLIM_AMAX", d.a_max),
            v_frac: num("JLIM_V_FRAC", d.v_frac).clamp(0.05, 1.0),
        }
    };

    let gains = {
        let d = &d.gains;
        BalanceGains {
            kp_com: num("KP_COM", d.kp_com),
            kd_com: num("KD_COM", d.kd_com),
            kp_trunk: num("KP_TRUNK", d.kp_trunk),
            kd_trunk: num("KD_TRUNK", d.kd_trunk),
            kp_post: num("KP_POST", d.kp_post),
            kd_post: num("KD_POST", d.kd_post),
            posture: flag("POST", d.posture),
            trunk: flag("TRUNK", d.trunk),
            trunk_sign: num("TRUNK_SIGN", d.trunk_sign),
            trunk_yaw: flag("TRUNK_YAW", d.trunk_yaw),
            kp_yaw: num("KP_YAW", d.kp_yaw),
            kd_yaw: num("KD_YAW", d.kd_yaw),
            diag_vz_bias_mpc: num("DIAG_VZ_BIAS_MPC", d.diag_vz_bias_mpc),
            diag_vz_bias_height: num("DIAG_VZ_BIAS_HEIGHT", d.diag_vz_bias_height),
        }
    };

    let dcm = {
        let d = &d.dcm;
        DcmConfig {
            cop_frac: num("COP_FRAC", d.cop_frac),
            k_dcm: num("K_DCM", d.k_dcm),
            vh_dcm: flag("VH_DCM", d.vh_dcm),
            zmp_margin: num("ZMP_MARGIN", d.zmp_margin),
            clamp_zmp: flag("CLAMP_ZMP", d.clamp_zmp),
            zmp_lat_scale: num("ZMP_LAT_SCALE", d.zmp_lat_scale),
            zmp_roll: opt("ZMP_ROLL").or(d.zmp_roll),
            com_xy_jerk_max: num("COM_XY_JERK_MAX", d.com_xy_jerk_max),
        }
    };

    let gait = {
        let d = &d.gait;
        GaitConfig {
            t_start: num("T_START", d.t_start),
            t_ds: num("T_DS", d.t_ds),
            t_ss: num("T_SS", d.t_ss),
            n_steps: num("N_STEPS", d.n_steps as f64) as usize,
            first_swing: num("FIRST_SWING", d.first_swing as f64) as usize,
            no_lift: flag("NO_LIFT", d.no_lift),
            turn_in_ds: flag("TURN_IN_DS", d.turn_in_ds),
            ds_settle: num("DS_SETTLE", d.ds_settle),
            ds_settle_after_step: num("DS_SETTLE_AFTER_STEP", d.ds_settle_after_step as f64).max(1.0) as usize,
            stride: num("STRIDE", d.stride),
            stride_ramp: num("STRIDE_RAMP", d.stride_ramp as f64) as usize,
            alternate: flag("ALTERNATE", d.alternate),
        }
    };

    let stair_plan = {
        let d = &d.stair_plan;
        StairPlanConfig {
            toe_margin: num("STAIR_TOE_MARGIN", d.toe_margin),
            y_out: num("STAIR_Y_OUT", d.y_out),
            gait_pair: get("STAIR_GAIT").map(|v| v == "pair").unwrap_or(d.gait_pair),
            td_bias: num("STAIR_TD_BIAS", d.td_bias),
            td_bias_asc_mid: opt("STAIR_TD_BIAS_ASC_MID").or(d.td_bias_asc_mid),
            full_sole: flag("STAIR_FULL_SOLE", d.full_sole),
            sole_margin: num("STAIR_SOLE_MARGIN", d.sole_margin),
            landing_frac: num("STAIR_LANDING_FRAC", d.landing_frac).max(1e-3),
            align_stand_line: flag("STAIR_ALIGN_STAND_LINE", d.align_stand_line),
        }
    };

    let descent = {
        let d = &d.descent;
        DescentConfig {
            td_bias_desc: opt("STAIR_TD_BIAS_DESC").or(d.td_bias_desc),
            stand_margin: num("STAIR_DESC_STAND_MARGIN", d.stand_margin).max(0.0),
            com_height_lead: num("COM_HEIGHT_DESC_LEAD", d.com_height_lead).clamp(0.0, 1.0),
            zone_crouch: num("STAIR_ZONE_CROUCH", d.zone_crouch).clamp(0.0, 0.15),
            land_extend: opt("COM_LAND_EXTEND_DESC").map(|v| v.max(0.0)).or(d.land_extend),
            land_confirm_xy: opt("LAND_CONFIRM_XY_DESC").or(d.land_confirm_xy),
            land_confirm_z: opt("LAND_CONFIRM_Z_DESC").or(d.land_confirm_z),
            leg_reach_margin: num("LEG_REACH_MARGIN", d.leg_reach_margin).clamp(0.1, 1.0),
        }
    };

    let slope = {
        let d = &d.slope;
        SlopeConfig {
            keep_stride: flag("SLOPE_KEEP_STRIDE", d.keep_stride),
            heel_cut: num("SLOPE_HEEL_CUT", d.heel_cut).max(0.0),
            heel_cut_max: num("SLOPE_HEEL_CUT_MAX", d.heel_cut_max).clamp(0.0, 0.08),
            terrain_pitch: flag("TERRAIN_PITCH", d.terrain_pitch),
            terrain_pitch_yaw: flag("TERRAIN_PITCH_YAW", d.terrain_pitch_yaw),
        }
    };

    let swing = {
        let d = &d.swing;
        let kp = num("KP_SWING", d.kp);
        SwingConfig {
            kp,
            kp_xy: opt("KP_SWING_XY").or(d.kp_xy),
            kd: num("KD_SWING", d.kd),
            kp_rot: num("KP_SWING_ROT", d.kp_rot),
            kd_rot: num("KD_SWING_ROT", d.kd_rot),
            lift_h: num("LIFT_H", d.lift_h),
            rot_mode: num("SWING_ROT", d.rot_mode as f64) as u32,
            zone_rot_mode: num("STAIR_ZONE_SWING_ROT", d.zone_rot_mode as f64).max(0.0) as u32,
            tracking_cubic_ff: flag("TRACKING_CUBIC_FF", d.tracking_cubic_ff),
            tracking_ff_min_riser: num("TRACKING_FF_MIN_RISER", d.tracking_ff_min_riser).max(0.0),
            tracking_ff_scale: num("TRACKING_FF_SCALE", d.tracking_ff_scale),
            td_sole_frame: flag("STAIR_TD_SOLE_FRAME", d.td_sole_frame),
            td_sink: num("STAIR_TD_SINK", d.td_sink),
            land_sink: num("STAIR_LAND_SINK", d.land_sink).max(0.0),
            stair_windows: {
                let s = get("STAIR_SWING_WINDOWS").unwrap_or_default();
                let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if v.len() == 4 { [v[0], v[1], v[2], v[3]] } else { d.stair_windows }
            },
            stair_ff: flag("STAIR_SWING_FF", d.stair_ff),
            step_dz_min: num("STEP_DZ_SWING_MIN", d.step_dz_min).max(0.0),
            absolute_td: flag("STAIR_ABSOLUTE_TD", d.absolute_td),
            absolute_td_desc: flag("STAIR_ABSOLUTE_TD_DESC", d.absolute_td_desc),
            xy_delay: num("SWING_XY_DELAY", d.xy_delay).clamp(0.0, 0.6),
            z_hold: flag("SWING_Z_HOLD", d.z_hold),
            z_hold_min_dur: num("SWING_Z_HOLD_MIN_DUR", d.z_hold_min_dur).clamp(0.0, 1.0),
            z_drop: {
                let v: Vec<f64> = get("SWING_Z_DROP")
                    .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                    .unwrap_or_default();
                [
                    v.first().copied().unwrap_or(d.z_drop[0]).clamp(0.3, 0.95),
                    v.get(1).copied().unwrap_or(d.z_drop[1]).clamp(0.5, 1.0),
                ]
            },
        }
    };

    let arm = {
        let d = &d.arm;
        ArmMomentumConfig {
            hold: flag("ARM_HOLD", d.hold),
            hold_joints: get("ARM_HOLD_JOINTS").unwrap_or_else(|| d.hold_joints.clone()),
            before_swing: flag("ARM_BEFORE_SWING", d.before_swing),
            kp: num("KP_ARM", d.kp),
            kd: num("KD_ARM", d.kd),
            mom: flag("MOM", d.mom),
            kp_mom: num("KP_MOM", d.kp_mom),
            mom_axes: match get("MOM_AXES") {
                Some(s) => {
                    let b = s.as_bytes();
                    [
                        b.first().is_some_and(|c| *c == b'x'),
                        b.get(1).is_some_and(|c| *c == b'y'),
                        b.get(2).is_some_and(|c| *c == b'z'),
                    ]
                }
                None => d.mom_axes,
            },
            mom_level: get("MOM_LEVEL").unwrap_or_else(|| d.mom_level.clone()),
            mom_ref: get("MOM_REF").unwrap_or_else(|| d.mom_ref.clone()),
            mom_arm_dcm: num("MOM_ARM_DCM", d.mom_arm_dcm),
            mom_arm_dcm_max: num("MOM_ARM_DCM_MAX", d.mom_arm_dcm_max).max(0.0),
        }
    };

    let mom_cmp = {
        let d = &d.mom_cmp;
        MomCmpConfig {
            gain: num("MOM_CMP", d.gain).max(0.0),
            max: num("MOM_CMP_MAX", d.max).max(0.0),
            dead: num("MOM_CMP_DEAD", d.dead).max(0.0),
            w: num("MOM_CMP_W", d.w).max(0.0),
            sign: num("MOM_CMP_SIGN", d.sign),
            hmax: num("MOM_CMP_HMAX", d.hmax).max(0.0),
            kh: num("MOM_CMP_KH", d.kh).max(0.0),
        }
    };

    let contact = {
        let d = &d.contact;
        ContactConfig {
            kp: num("KP_CONTACT", d.kp),
            kd: opt("KD_CONTACT").or(d.kd),
            anchor_leak: num("ANCHOR_LEAK", d.anchor_leak),
            driven: flag("CONTACT_DRIVEN", d.driven),
            ticks: num("CONTACT_TICKS", d.ticks as f64) as u32,
            on_bw: num("CONTACT_ON_BW", d.on_bw),
            load_ramp_frac: num("LOAD_RAMP_FRAC", d.load_ramp_frac).clamp(0.05, 1.0),
            admit_confirmed: flag("CONTACT_ADMIT_CONFIRMED", d.admit_confirmed),
            admit_scheduled: flag("CONTACT_ADMIT_SCHEDULED", d.admit_scheduled),
            admit_z: num("CONTACT_ADMIT_Z", d.admit_z).max(0.0),
            admit_loaded: get("CONTACT_ADMIT_LOADED").and_then(|v| {
                let mut it = v.split(',').map(|x| x.trim().parse::<f64>().ok());
                Some((it.next()??.max(0.0), it.next()??.max(0.0)))
            }).or(d.admit_loaded),
        }
    };

    let managed = {
        let d = &d.managed;
        ManagedContactConfig {
            on: flag("STAIR_MANAGED_CONTACT", d.on),
            tangent: num("STAIR_MANAGED_TANGENT", d.tangent).clamp(0.0, 1.0),
            rotation: num("STAIR_MANAGED_ROTATION", d.rotation).clamp(0.0, 1.0),
            az_max: num("STAIR_MANAGED_AZ_MAX", d.az_max).max(0.1),
            rate: num("STAIR_MANAGED_RATE", d.rate).max(0.0),
            latch: num("STAIR_MANAGED_LATCH", d.latch).max(0.0),
        }
    };

    let landing = {
        let d = &d.landing;
        LandingConfig {
            confirm_fz_frac: num("LAND_CONFIRM_FZ_FRAC", d.confirm_fz_frac).max(0.0),
            confirm_xy: num("LAND_CONFIRM_XY", d.confirm_xy),
            confirm_z: num("LAND_CONFIRM_Z", d.confirm_z),
            predict_frac: num("COM_LAND_PREDICT_FRAC", d.predict_frac).clamp(0.0, 1.0),
            extend: num("COM_LAND_EXTEND", d.extend).max(0.0),
            extend_rise: num("COM_LAND_EXTEND_RISE", d.extend_rise).max(0.0),
            step_dz_min: num("STEP_DZ_MIN", d.step_dz_min).max(0.0),
            speed_xy: num("COM_LAND_VXY", d.speed_xy).max(0.01),
            speed_z: num("COM_LAND_VZ", d.speed_z).max(0.01),
            event: ContactEventConfig {
                on: flag("CONTACT_EVENT", d.event.on),
                z_tol: num("CONTACT_EVENT_Z", d.event.z_tol).max(0.0),
                shift: flag("CONTACT_EVENT_SHIFT", d.event.shift),
                end_on_touch: flag("CONTACT_EVENT_END_ON_TOUCH", d.event.end_on_touch),
                probe_speed: num("CONTACT_PROBE_VZ", d.event.probe_speed).max(0.0),
                probe_max: num("CONTACT_PROBE_MAX", d.event.probe_max).max(0.0),
            },
            reanchor: num("LAND_REANCHOR", d.reanchor).max(0.0),
            reanchor_max: num("LAND_REANCHOR_MAX", d.reanchor_max).max(0.0),
            reanchor_load: num("LAND_REANCHOR_LOAD", d.reanchor_load).max(0.0),
        }
    };

    let com_plan = {
        let d = &d.com_plan;
        ComPlanConfig {
            z_com: opt("Z_COM").or(d.z_com),
            z_com_offset: num("Z_COM_OFFSET", d.z_com_offset),
            replan: flag("COM_REPLAN", d.replan),
            preplan: get("COM_PREPLAN").map(|v| v != "0").or(d.preplan),
            preplan_xy_min: num("COM_PREPLAN_XY_MIN", d.preplan_xy_min),
            stair_ss_max: num("COM_STAIR_SS_MAX", d.stair_ss_max),
            replan_forward_gain: num("COM_REPLAN_FORWARD_GAIN", d.replan_forward_gain),
            replan_dt: num("COM_REPLAN_DT", d.replan_dt),
            preview_dt: num("COM_PREVIEW_DT", d.preview_dt),
            preview_n: num("COM_PREVIEW_N", d.preview_n as f64).max(2.0) as usize,
            preview_smooth: num("COM_PREVIEW_SMOOTH", d.preview_smooth).max(0.0),
            height_replan: flag("COM_HEIGHT_REPLAN", d.height_replan),
            height_time: num("COM_HEIGHT_TIME", d.height_time).max(0.10),
            stair_climb_rate: num("STAIR_CLIMB_RATE", d.stair_climb_rate),
        }
    };

    let stair_zone = {
        let d = &d.stair_zone;
        StairZoneConfig {
            lead: num("STAIR_ZONE_CROUCH_LEAD", d.lead).max(0.0),
            tail: num("STAIR_ZONE_CROUCH_TAIL", d.tail).max(0.0),
        }
    };

    let contact_plan = {
        let d = &d.contact_plan;
        ContactPlanConfig {
            on: flag("CONTACT_PLAN", d.on),
            trig: num("CP_TRIG", d.trig),
            t_min_frac: num("CP_T_MIN", d.t_min_frac),
            t_max_frac: num("CP_T_MAX", d.t_max_frac),
            dt_grid: num("CP_DT", d.dt_grid).max(0.001),
            max_x: num("CP_MAX_X", d.max_x),
            max_y: num("CP_MAX_Y", d.max_y),
            max_y_in: opt("CP_MAX_Y_IN").or(d.max_y_in),
            timing_from_x: flag("CP_T_FROM_X", d.timing_from_x),
            timing_lat_outward: flag("CP_T_LAT_OUT", d.timing_lat_outward),
            j: num("CP_J", d.j).max(0.05),
            w_shift: num("CP_W_SHIFT", d.w_shift).max(0.0),
            w_u: num("CP_W_U", d.w_u),
            w_t: num("CP_W_T", d.w_t),
            w_res: num("CP_W_RES", d.w_res),
            dead_u: num("CP_DEAD_U", d.dead_u),
            dead_t: num("CP_DEAD_T", d.dead_t),
        }
    };

    let double_support = {
        let d = &d.double_support;
        DoubleSupportConfig {
            feasible: flag("DS_FEASIBLE", d.feasible),
            feasible_on_timeout: flag("DS_FEASIBLE_ON_TIMEOUT", d.feasible_on_timeout),
            feasible_continuous: flag("DS_FEASIBLE_CONTINUOUS", d.feasible_continuous),
            feasible_ramp: flag("DS_FEASIBLE_RAMP", d.feasible_ramp),
            feasible_max: num("DS_FEASIBLE_MAX", d.feasible_max),
            feasible_box_stance: get("DS_FEASIBLE_BOX").map(|v| v == "stance").unwrap_or(d.feasible_box_stance),
            capture: flag("DS_CAPTURE", d.capture),
            capture_hopeless: num("DS_CAPTURE_HOPELESS", d.capture_hopeless).max(1.0),
            capture_min_gain: num("DS_CAPTURE_MIN_GAIN", d.capture_min_gain).max(0.0),
            capture_fwd_only: flag("DS_CAPTURE_FWD_ONLY", d.capture_fwd_only),
            capture_margin: num("DS_CAPTURE_MARGIN", d.capture_margin),
            capture_t_min: num("DS_CAPTURE_T_MIN", d.capture_t_min).max(0.02),
            capture_load: num("DS_CAPTURE_LOAD", d.capture_load).clamp(0.0, 1.0),
        }
    };

    let mpc = {
        let d = &d.mpc;
        WrenchMpcConfig {
            enabled: flag("STAIR_WRENCH_MPC", d.enabled),
            always: flag("WRENCH_MPC_ALWAYS", d.always),
            apply: flag("WRENCH_MPC_APPLY", d.apply),
            coupled: flag("STAIR_WRENCH_COUPLED", d.coupled),
            node_centre: flag("MPC_NODE_CENTRE", d.node_centre),
            net_cop_w: num("MPC_NET_COP_W", d.net_cop_w).max(0.0),
            seed_previous: num("MPC_SEED_PREVIOUS", d.seed_previous as f64) as i32,
            w_pos: num("MPC_W_POS", d.w_pos).max(0.0),
            w_vel: num("MPC_W_VEL", d.w_vel).max(0.0),
            w_mom: num("MPC_W_MOM", d.w_mom).max(0.0),
            w_acc: num("MPC_W_ACC", d.w_acc).max(0.0),
            w_wrench: num("MPC_W_WRENCH", d.w_wrench).max(0.0),
            w_smooth: num("MPC_W_SMOOTH", d.w_smooth).max(0.0),
            sole_authority: flag("MPC_SOLE_AUTHORITY", d.sole_authority),
            dt: num("STAIR_WRENCH_DT", d.dt),
            horizon: num("STAIR_WRENCH_N", d.horizon as f64).max(2.0) as usize,
            preview_dt: num("STAIR_WRENCH_PREVIEW_DT", d.preview_dt),
            low_min: num("STAIR_WRENCH_LOW_MIN", d.low_min).clamp(0.0, 0.45),
            blend: num("STAIR_WRENCH_BLEND", d.blend).clamp(0.0, 1.0),
            hybrid_contact: flag("STAIR_HYBRID_CONTACT", d.hybrid_contact),
            backend: get("MPC_QP_BACKEND").map(|name| match name.as_str() {
                "active_set" => misa_wbc::QpSolver::ActiveSet,
                "clarabel" => misa_wbc::QpSolver::Clarabel,
                "proxqp" => misa_wbc::QpSolver::Proxqp,
                "osqp" => misa_wbc::QpSolver::Osqp,
                "hpipm" => misa_wbc::QpSolver::Hpipm,
                _ => panic!("unsupported MPC_QP_BACKEND: {name}"),
            }).or(d.backend),
            kkt_comp_tol: num("MPC_KKT_COMP_TOL", d.kkt_comp_tol),
            bench_gate: flag("MPC_BENCH_GATE", d.bench_gate),
            trace_landing_time: get("MPC_LANDING_TIME_DEBUG").is_some() || d.trace_landing_time,
            trace_failures: get("MPC_FS_DEBUG").is_some() || d.trace_failures,
        }
    };

    let mpc_footstep = {
        let d = &d.mpc_footstep;
        MpcFootstepConfig {
            on: flag("MPC_FOOTSTEP", d.on),
            w: num("MPC_FOOTSTEP_W", d.w).max(0.0),
            moment: flag("MPC_FOOTSTEP_MOMENT", d.moment),
            trig: num("MPC_FOOTSTEP_TRIG", d.trig).max(0.0),
            max: [
                num("MPC_FOOTSTEP_MAX_X", d.max[0]).max(0.0),
                num("MPC_FOOTSTEP_MAX_Y", d.max[1]).max(0.0),
            ],
            max_y_in: num("MPC_FOOTSTEP_MAX_Y_IN", d.max_y_in).max(0.0),
            y_in_fwd: num("MPC_FOOTSTEP_Y_IN_FWD", d.y_in_fwd).max(0.0),
            y_in_fwd_trig: num("MPC_FOOTSTEP_Y_IN_FWD_TRIG", d.y_in_fwd_trig),
            capture_w: num("MPC_LANDING_CAPTURE_W", d.capture_w).max(0.0),
            capture_j: flag("MPC_LANDING_CAPTURE_J", d.capture_j),
            lex: flag("MPC_LANDING_LEX", d.lex),
        }
    };

    let mpc_cross = {
        let d = &d.mpc_cross;
        MpcCrossConfig {
            y: num("MPC_FOOTSTEP_CROSS_Y", d.y).max(0.0),
            ex: num("MPC_FOOTSTEP_CROSS_EX", d.ex),
            floor: num("MPC_FOOTSTEP_CROSS_FLOOR", d.floor).max(0.0),
            hold_z: num("MPC_FOOTSTEP_CROSS_HOLD_Z", d.hold_z).max(0.0),
            hold_xy: num("MPC_FOOTSTEP_CROSS_HOLD_XY", d.hold_xy).max(0.0),
        }
    };

    let push_replan = {
        let d = &d.push_replan;
        PushReplanConfig {
            trig: num("PUSH_REPLAN", d.trig).max(0.0),
            emin: num("PUSH_REPLAN_EMIN", d.emin).max(0.0),
            freeze: num("PUSH_REPLAN_FREEZE", d.freeze),
            stand: num("PUSH_REPLAN_STAND", d.stand as f64).max(0.0) as usize,
            ramp: num("PUSH_REPLAN_RAMP", d.ramp as f64).max(0.0) as usize,
            max_fwd: num("PUSH_REPLAN_MAX_FWD", d.max_fwd).max(0.0),
            max_back: num("PUSH_REPLAN_MAX_BACK", d.max_back).max(0.0),
            min_w: num("PUSH_REPLAN_MIN_W", d.min_w).max(0.0),
            max_w: num("PUSH_REPLAN_MAX_W", d.max_w).max(0.0),
            hold: num("PUSH_REPLAN_HOLD", d.hold).max(0.0),
            back_only: flag("PUSH_REPLAN_BACK_ONLY", d.back_only),
            vmax: num("PUSH_REPLAN_VMAX", d.vmax).max(0.0),
            ankle: {
                let v: Vec<f64> = get("PUSH_REPLAN_ANKLE")
                    .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                    .unwrap_or_default();
                [
                    v.first().copied().unwrap_or(d.ankle[0]).clamp(0.0, 1.0),
                    v.get(1).copied().unwrap_or(d.ankle[1]).clamp(0.0, 1.0),
                ]
            },
            slope_lead: num("PUSH_REPLAN_SLOPE_LEAD", d.slope_lead).max(0.0),
        }
    };

    let push_crouch = {
        let d = &d.push_crouch;
        PushCrouchConfig {
            drop: num("PUSH_Z_DROP", d.drop).max(0.0),
            trig: num("PUSH_Z_DROP_TRIG", d.trig).max(0.0),
            t: num("PUSH_Z_DROP_T", d.t).max(0.0),
            lat_max: num("PUSH_Z_DROP_LAT_MAX", d.lat_max).max(0.0),
        }
    };

    let speed = {
        let d = &d.speed;
        SpeedScheduleConfig {
            sched: flag("VX2_SCHED", d.sched),
            tscale_min: num("VX2_TSCALE_MIN", d.tscale_min).clamp(0.1, 1.0),
            stride_max: num("VXS_STRIDE_MAX", d.stride_max).max(0.05),
            cap_slope: num("VXS_CAP_SLOPE", d.cap_slope),
            v_long: num("VXS_V_LONG", d.v_long),
            z_slope: num("VXS_Z_SLOPE", d.z_slope).max(0.0),
            z_max: num("VXS_Z_MAX", d.z_max),
            z_min: num("VXS_Z_MIN", d.z_min).max(0.0),
            yield_v: num("PUSH_YIELD_V", d.yield_v).max(0.0),
            yield_trig: num("PUSH_YIELD_TRIG", d.yield_trig).max(0.0),
            yield_no_fwd: flag("PUSH_YIELD_NO_FWD", d.yield_no_fwd),
            yield_vmin: num("PUSH_YIELD_VMIN", d.yield_vmin).max(0.0),
        }
    };

    let recovery = {
        let d = &d.recovery;
        RecoveryRefConfig {
            trig_dcm: num("REF_TRIG_DCM", d.trig_dcm).max(0.0),
            trig_max_frac: num("REF_TRIG_MAX_FRAC", d.trig_max_frac),
            delay_comp: flag("REF_DELAY_COMP", d.delay_comp),
            swing: flag("REF_SWING", d.swing),
            reset_stance: flag("REF_RESET_STANCE", d.reset_stance),
            reset_incl_current: flag("REF_RESET_INCL_CURRENT", d.reset_incl_current),
            cooldown: num("REF_COOLDOWN", d.cooldown).max(0.0),
        }
    };

    // The debug prints: most are on when the key is SET (to anything),
    // DEBUG_ANCHOR and DIAG_REPLAN are switches.
    let debug = {
        let set = |k: &str| get(k).is_some();
        DebugFlags {
            contact_admit: set("CONTACT_ADMIT_DEBUG"),
            ds_feasible: set("DS_FEASIBLE_DEBUG"),
            ds_capture: set("DS_CAPTURE_DEBUG"),
            anchor: flag("DEBUG_ANCHOR", d.debug.anchor),
            cp: set("CP_DEBUG"),
            push_replan: set("PUSH_REPLAN_DEBUG"),
            mpc_fs_cross: set("MPC_FS_CROSS_DEBUG"),
            mpc_fs_landing: set("MPC_FS_DEBUG_LANDING"),
            mpc_traj: set("MPC_TRAJ_DEBUG"),
            replan: flag("DIAG_REPLAN", d.debug.replan),
            wbc_level_t: get("WBC_LEVEL_DEBUG_T").and_then(|v| v.parse::<f64>().ok()),
            land_timing: set("LAND_TIMING_DEBUG"),
        }
    };

    Config {
        wbc,
        joint_limit,
        gains,
        dcm,
        gait,
        stair_plan,
        descent,
        slope,
        swing,
        arm,
        mom_cmp,
        contact,
        managed,
        landing,
        com_plan,
        stair_zone,
        contact_plan,
        double_support,
        mpc,
        mpc_footstep,
        mpc_cross,
        push_replan,
        push_crouch,
        speed,
        recovery,
        debug,
    }
}
