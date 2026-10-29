//! Contact planning at the DCM layer: pick WHEN the current single support
//! ends and WHERE the swing foot lands so that the DCM at the end of the step
//! is where the gait plan wants it.
//!
//! P2-1 of doc/from_claude/20260923_p2_contact_planner_design_v2.md. This is
//! the predictive replacement for the reactive `ADAPT_STEP` (move the foot by
//! the ZMP clamp deficit after saturation) and `ADAPT_TIME` (invert the
//! single-support closed form on one axis). Both were measured neutral on G1
//! (T=40, landing 3.0 m, N=48 x 4 risers, 2026-09-23).
//!
//! Why a grid over T and a closed form for `u`, not a QP over `(u, tau)`:
//! with `ZMP_ROLL > 0` the stance ZMP moves linearly during single support,
//! so the DCM at the end of the step is affine in `u` but NOT affine in
//! `tau = exp(omega T)` -- the convexification of Khadiv et al. 2020 assumes a
//! fixed pivot. A 10 ms grid over T with the exact rolled prediction and the
//! box-clamped closed form for `u` is exact and cheaper than a QP.

use nalgebra as na;

/// Tunables. Defaults are deliberately conservative; every one is an env
/// knob in the walker (`CP_*`).
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Remaining single-support duration the plan may choose, as fractions
    /// of the nominal remaining duration.
    pub t_min_frac: f64,
    pub t_max_frac: f64,
    /// Grid spacing over T [s].
    pub dt_grid: f64,
    /// Reachability box for the landing shift [m], relative to the plan.
    pub max_x: f64,
    pub max_y: f64,
    /// Lateral bound TOWARD the stance foot (inward) [m]; `max_y` bounds the
    /// outward side. Measured 2026-09-24 on the flat push benchmark: after an
    /// 11 N*s backward push the planner shortened the step by 0.17 s, and the
    /// shortening moved the predicted lateral end point so that the chosen
    /// landing went 30 mm INWARD although the lateral error was nil; two
    /// steps later the feet crossed and the run fell sideways. Only used when
    /// `Situation::swing_sign` is non-zero.
    pub max_y_in: f64,
    /// Choose the step TIMING from the sagittal (x) residual only; the
    /// lateral axis is then corrected by placement alone. Measured
    /// 2026-09-24 on the flat push benchmark: a 3 N*s push toward the stance
    /// side leaves a lateral residual placement cannot fix (it would need a
    /// cross-over step), so the planner shortened the step by 0.17 s to
    /// shrink it; the rebuilt plan moved the lateral DCM reference 30 mm at
    /// once, the ZMP command sat on the stance foot's outer edge for 0.45 s
    /// and the next swing crossed under the body. ADAPT_STEP, which never
    /// retimes, survived the same member.
    pub timing_from_x: bool,
    /// With `timing_from_x`: still let the LATERAL residual drive the timing
    /// when it points toward the swing side (outward), where landing the
    /// swing foot sooner is what catches the DCM. Measured 2026-09-24: a
    /// 5 N*s push toward the swing side survived 9/16 with lateral timing and
    /// 4/16 without; a push toward the stance side needs the opposite (11/16
    /// without, 7/16 with). Needs `Situation::swing_sign`.
    pub timing_lat_outward: bool,
    /// Sensitivity of the plan's end-of-step DCM to the landing: moving the
    /// next foot by `u` moves `xi_eos` by about `j * u`. Measured on G1 the
    /// DCM sits at the mid-point of the two feet at the end of single
    /// support, hence 0.5 (P2 theory review). A numeric Jacobian from the
    /// plan replaces this in P2-1b.
    pub j: f64,
    /// How much a shift COSTS when the shift itself is chosen. 0 (the
    /// default) reproduces the original "cancel the whole residual",
    /// `u = resid / j`; larger values leave part of the residual to the
    /// timing and to the next step, which is what makes the planner safe to
    /// run continuously. Separate from `w_u`, which weighs the same shift
    /// when the grid picks `T` -- changing that would move the timing of
    /// every recipe measured so far.
    pub w_shift: f64,
    /// Cost weights: landing shift, timing change, unmet residual.
    pub w_u: f64,
    pub w_t: f64,
    pub w_res: f64,
    /// Dead bands: below these nothing is applied (keeps the nominal gait
    /// bit-for-bit when undisturbed).
    pub dead_u: f64,
    pub dead_t: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            t_min_frac: 0.5,
            t_max_frac: 1.5,
            dt_grid: 0.01,
            max_x: 0.06,
            max_y: 0.03,
            max_y_in: 0.03,
            timing_from_x: false,
            timing_lat_outward: false,
            j: 0.5,
            w_shift: 0.0,
            w_u: 1.0,
            w_t: 1.0,
            w_res: 100.0,
            dead_u: 0.002,
            dead_t: 0.005,
        }
    }
}

/// What the current single support looks like from the planner's side.
#[derive(Clone, Copy, Debug)]
pub struct Situation {
    /// Measured DCM now.
    pub xi: na::Vector2<f64>,
    /// Planned ZMP now and its planned rate during this single support
    /// (`ZmpSeg::at(t)` and `ZmpSeg::slope()`).
    pub p: na::Vector2<f64>,
    pub p_dot: na::Vector2<f64>,
    pub omega: f64,
    pub t: f64,
    /// Planned end of the current single support.
    pub nominal_end: f64,
    /// Where the plan wants the DCM at that end (`DcmPlan::eos_at`).
    pub xi_eos_plan: na::Vector2<f64>,
    /// +1 when the swing foot is the LEFT one (outward = +y), -1 for the
    /// right, 0 when unknown (symmetric `max_y` bound).
    pub swing_sign: f64,
}

/// The decision: shift the next landing by `shift` and end the single
/// support at `new_end` (None = keep the plan's end).
#[derive(Clone, Copy, Debug, Default)]
pub struct Decision {
    pub shift: na::Vector2<f64>,
    pub new_end: Option<f64>,
    pub cost: f64,
    /// Residual the plan could not absorb (clamped `u` and bounded T).
    pub unmet: f64,
}

/// DCM at `t + T` under a ZMP moving linearly from `p` at rate `p_dot`:
/// `xi(t+T) = p + p_dot T + p_dot/omega + (xi - p - p_dot/omega) e^{omega T}`.
/// Same expression as `DcmPlan::predict_eos` with the end moved to `t + T`.
pub fn predict(s: &Situation, big_t: f64) -> na::Vector2<f64> {
    let v_over_w = s.p_dot / s.omega;
    s.p + s.p_dot * big_t + v_over_w + (s.xi - s.p - v_over_w) * (s.omega * big_t).exp()
}

pub fn plan(s: &Situation, cfg: &Config) -> Decision {
    let remaining = (s.nominal_end - s.t).max(0.0);
    let t_lo = (remaining * cfg.t_min_frac).max(0.0);
    let t_hi = (remaining * cfg.t_max_frac).max(t_lo);
    let mut best: Option<Decision> = None;
    let n = ((t_hi - t_lo) / cfg.dt_grid).floor() as usize + 1;
    for k in 0..=n {
        let big_t = (t_lo + k as f64 * cfg.dt_grid).min(t_hi);
        let resid = predict(s, big_t) - s.xi_eos_plan;
        // The shift that MINIMISES the cost, not the one that cancels the
        // residual.
        //
        //     min_u  w_u |u|^2 + w_res |resid - j u|^2
        //     =>     u = w_res j resid / (w_u + w_res j^2)
        //
        // The first draft set `u = resid / j`, i.e. always cancel, and used
        // the cost only to pick `T` on the grid. Nothing then traded "move
        // the foot" against "accept some error", so with the 15 mm trigger
        // removed the planner rewrote the gait every tick: measured
        // 2026-09-24 on the randomised stairs, firing continuously was
        // 0/192 and fell at 21 s, BEFORE the stair, while the shipped
        // trigger gave 137/192. Raising `w_u` now buys a real trade-off
        // instead of a dead band -- this is the Khadiv/Griffin step
        // adjustment written the way its cost says.
        let want = resid * (cfg.w_res * cfg.j / (cfg.w_shift + cfg.w_res * cfg.j * cfg.j));
        let u = na::Vector2::new(
            want.x.clamp(-cfg.max_x, cfg.max_x),
            {
                let (lo, hi) = if s.swing_sign > 0.0 {
                    (-cfg.max_y_in, cfg.max_y)
                } else if s.swing_sign < 0.0 {
                    (-cfg.max_y, cfg.max_y_in)
                } else {
                    (-cfg.max_y, cfg.max_y)
                };
                want.y.clamp(lo, hi)
            },
        );
        let left = resid - u * cfg.j;
        let dt = big_t - remaining;
        let cost = if cfg.timing_from_x {
            let lat = cfg.timing_lat_outward && s.swing_sign != 0.0 && resid.y * s.swing_sign > 0.0;
            if lat {
                cfg.w_u * u.norm_squared() + cfg.w_t * dt * dt + cfg.w_res * left.norm_squared()
            } else {
                cfg.w_u * u.x * u.x + cfg.w_t * dt * dt + cfg.w_res * left.x * left.x
            }
        } else {
            cfg.w_u * u.norm_squared() + cfg.w_t * dt * dt + cfg.w_res * left.norm_squared()
        };
        if best.is_none_or(|b| cost < b.cost) {
            best = Some(Decision {
                shift: u,
                new_end: Some(s.t + big_t),
                cost,
                unmet: left.norm(),
            });
        }
    }
    let mut d = best.unwrap_or_default();
    if d.shift.norm() < cfg.dead_u {
        d.shift = na::Vector2::zeros();
    }
    if let Some(e) = d.new_end
        && (e - s.nominal_end).abs() < cfg.dead_t
    {
        d.new_end = None;
    }
    d
}

/// What the touchdown planner needs on top of [`Situation`].
#[derive(Clone, Copy, Debug)]
pub struct Touchdown {
    /// Length of the double support that follows this single support [s].
    pub ds_duration: f64,
    /// Where the pressure enters the NEXT single support, relative to that
    /// stance sole: `-roll` along its yaw (see `DcmPlan::from_footsteps_rolled`).
    pub entry_offset: na::Vector2<f64>,
    /// The landing the plan intends, as the pressure point the DCM plan uses
    /// (sole centre pulled toward mid-stance by the lateral scale).
    pub u_nom: na::Vector2<f64>,
    /// Where the DCM should sit relative to that landing when the next single
    /// support begins. Taken from the plan, so an undisturbed step asks for
    /// no shift at all.
    pub b_nom: na::Vector2<f64>,
}

/// Step adjustment written the way the DCM literature writes it: put the foot
/// where the DCM will BE when it lands.
///
/// [`plan`] compares the predicted end-of-single-support DCM against the
/// plan's value for it, and converts the difference into a shift through a
/// sensitivity `j` -- how much the plan's own backward recursion moves when
/// the landing moves. Measured 2026-09-24, that `j` is 0.646 (the shipped
/// constant said 0.5), and it is the WRONG channel: it is the landing's
/// effect on the plan, not on the robot. The landing's real effect is that
/// the foot becomes the pressure point, and the first place that acts is the
/// double support that follows.
///
/// Propagating the DCM through that double support, with the ZMP ramping from
/// where it ends the single support to the new foot's entry point, gives
///
/// ```text
/// xi_td(u) = alpha + beta u,   beta = 1 + (1 - e^{w Tds}) / (w Tds)
/// ```
///
/// and the condition "the DCM lands at its nominal offset from the new foot"
/// is `xi_td(u) = u + b_nom`, whose coefficient in `u` is `beta - 1` = -2.25
/// on this robot: 3.5x the authority of the shipped channel, and with the
/// sign the physics asks for (pressure further forward slows the DCM's
/// forward divergence).
///
/// Two steps ahead was the first design; it is not used. The DCM at the end
/// of the NEXT single support carries `e^{w T} = 45`, so its coefficient in
/// `u` is about -100 and the least squares is stiff enough to be meaningless
/// against any model error. Landing where the DCM will be is the same
/// guarantee, well conditioned.
pub fn plan_touchdown(s: &Situation, td: &Touchdown, cfg: &Config) -> Decision {
    let remaining = (s.nominal_end - s.t).max(0.0);
    let t_lo = (remaining * cfg.t_min_frac).max(0.0);
    let t_hi = (remaining * cfg.t_max_frac).max(t_lo);
    let tds = td.ds_duration.max(1e-3);
    let e2 = (s.omega * tds).exp();
    let beta = 1.0 + (1.0 - e2) / (s.omega * tds);
    let g = beta - 1.0;
    let mut best: Option<Decision> = None;
    let n = ((t_hi - t_lo) / cfg.dt_grid).floor() as usize + 1;
    for k in 0..=n {
        let big_t = (t_lo + k as f64 * cfg.dt_grid).min(t_hi);
        // DCM and ZMP at the end of the single support.
        let xi1 = predict(s, big_t);
        let pa = s.p + s.p_dot * big_t;
        // xi_td(u) = alpha + beta u, from the linear-ZMP solution over the
        // double support with the ZMP going pa -> (u + entry_offset).
        let c = td.entry_offset;
        // xi_td = (u + c) + v/w + (xi1 - pa - v/w) e2,  v = (u + c - pa)/Tds.
        // Splitting off the terms in `u` leaves beta above and:
        let k = (c - pa) / (s.omega * tds);
        let alpha = c + k + (xi1 - pa - k) * e2;
        // Residual at touchdown if the foot does not move, and the shift that
        // minimises w_shift |u - u_nom|^2 + w_res |resid|^2.
        let r0 = alpha - td.b_nom + td.u_nom * g;
        let denom = cfg.w_shift + cfg.w_res * g * g;
        let u_abs = if denom > 0.0 {
            (td.u_nom * cfg.w_shift - (r0 - td.u_nom * g) * (cfg.w_res * g)) / denom
        } else {
            td.u_nom
        };
        let want = u_abs - td.u_nom;
        let (lo, hi) = if s.swing_sign > 0.0 {
            (-cfg.max_y_in, cfg.max_y)
        } else if s.swing_sign < 0.0 {
            (-cfg.max_y, cfg.max_y_in)
        } else {
            (-cfg.max_y, cfg.max_y)
        };
        let u = na::Vector2::new(
            want.x.clamp(-cfg.max_x, cfg.max_x),
            want.y.clamp(lo, hi),
        );
        let left = alpha - td.b_nom + (td.u_nom + u) * g;
        let dt = big_t - remaining;
        let cost =
            cfg.w_u * u.norm_squared() + cfg.w_t * dt * dt + cfg.w_res * left.norm_squared();
        if best.is_none_or(|b| cost < b.cost) {
            best = Some(Decision {
                shift: u,
                new_end: Some(s.t + big_t),
                cost,
                unmet: left.norm(),
            });
        }
    }
    let mut d = best.unwrap_or_default();
    if d.shift.norm() < cfg.dead_u {
        d.shift = na::Vector2::zeros();
    }
    if let Some(e) = d.new_end
        && (e - s.nominal_end).abs() < cfg.dead_t
    {
        d.new_end = None;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dcm::DcmPlan;
    use crate::gait::{FootstepPlan, Footsteps, GaitParams, GaitPlan};

    /// What `Config::j` is supposed to be: how far the plan's end-of-step DCM
    /// moves when the NEXT landing moves. The shipped 0.5 came from "the DCM
    /// sits midway between the feet at the end of single support". This
    /// measures it against the real backward recursion instead, because the
    /// planner divides by it -- a `j` that is half the truth doubles every
    /// correction, and the step adjustment then overshoots exactly when it
    /// runs often enough to matter.
    #[test]
    fn the_footstep_sensitivity_matches_the_plan() {
        let gait = GaitPlan::new(&GaitParams::default());
        let actual = Footsteps {
            sole: [
                na::Vector3::new(0.0, 0.05, 0.0),
                na::Vector3::new(0.0, -0.05, 0.0),
            ],
            yaw: [0.0; 2],
        };
        let feet = FootstepPlan::constant_stride(&gait, &actual, 0.14);
        let build = |f: &FootstepPlan| DcmPlan::from_footsteps_rolled(&gait, f, 0.68, 1.0, 0.03);
        // A single support in the middle of the walk, and the foot that lands
        // at its end.
        let i = (0..gait.slices.len())
            .find(|&k| k > 4 && gait.slices[k].support.swing().is_some())
            .expect("a single support");
        let sw = gait.slices[i].support.swing().unwrap();
        let base = build(&feet).xi_eos[i];
        let du = 0.02;
        let mut moved = FootstepPlan::constant_stride(&gait, &actual, 0.14);
        moved.shift_from(i, sw, na::Vector2::new(du, 0.0));
        let j = (build(&moved).xi_eos[i].x - base.x) / du;
        // Report it in the failure message so the number is visible.
        // Measured 2026-09-24 with the shipped gait: 0.646, against the
        // 0.5 the planner assumes -- so every correction is 29% too big.
        // `CP_J_AUTO` computes this at run time instead of trusting either
        // number, because it moves with the gait timing and the roll.
        assert!((j - 0.646).abs() < 0.02, "measured j = {j:.4}");
        assert!(
            (j - Config::default().j).abs() > 0.1,
            "the constant is no longer wrong; re-check why"
        );
    }

    fn sit(xi: (f64, f64), eos: (f64, f64)) -> Situation {
        Situation {
            xi: na::Vector2::new(xi.0, xi.1),
            p: na::Vector2::new(0.0, -0.06),
            p_dot: na::Vector2::new(0.06, 0.0),
            omega: 3.788,
            t: 10.0,
            nominal_end: 10.4,
            xi_eos_plan: na::Vector2::new(eos.0, eos.1),
            swing_sign: 0.0,
        }
    }

    #[test]
    fn timing_from_x_ignores_a_lateral_residual() {
        // Pure lateral residual: with timing_from_x the step is not retimed.
        let cfg = Config { timing_from_x: true, ..Config::default() };
        let s0 = sit((0.02, -0.03), (0.0, 0.0));
        let base = predict(&s0, 0.4);
        let s = Situation { xi_eos_plan: base + na::Vector2::new(0.0, 0.08), ..s0 };
        let d = plan(&s, &cfg);
        assert!(d.new_end.is_none(), "{:?}", d);
        // residual = predicted - plan = -0.08 in y: placement follows it.
        assert!(d.shift.y < 0.0, "{:?}", d);
    }

    #[test]
    fn outward_lateral_residual_may_retime_inward_may_not() {
        let cfg = Config { timing_from_x: true, timing_lat_outward: true, max_y: 0.005, ..Config::default() };
        let s0 = sit((0.02, -0.03), (0.0, 0.0));
        let base = predict(&s0, 0.4);
        // swing = left (outward +y). Predicted end 8 cm further +y than plan.
        let out = Situation { xi_eos_plan: base - na::Vector2::new(0.0, 0.08), swing_sign: 1.0, ..s0 };
        assert!(plan(&out, &cfg).new_end.is_some());
        let inw = Situation { xi_eos_plan: base + na::Vector2::new(0.0, 0.08), swing_sign: 1.0, ..s0 };
        assert!(plan(&inw, &cfg).new_end.is_none());
    }

    #[test]
    fn inward_bound_limits_only_the_inward_side() {
        // Swing foot is the LEFT one (outward = +y). A residual asking for a
        // large inward (-y) shift is cut at max_y_in; outward stays at max_y.
        let cfg = Config { max_y_in: 0.005, t_min_frac: 1.0, t_max_frac: 1.0, ..Config::default() };
        let s0 = sit((0.02, -0.03), (0.0, 0.0));
        let base = predict(&s0, 0.4);
        let inward = Situation { xi_eos_plan: base + na::Vector2::new(0.0, 0.05), swing_sign: 1.0, ..s0 };
        let d = plan(&inward, &cfg);
        assert!((d.shift.y + 0.005).abs() < 1e-9, "{:?}", d);
        let outward = Situation { xi_eos_plan: base - na::Vector2::new(0.0, 0.05), swing_sign: 1.0, ..s0 };
        let d = plan(&outward, &cfg);
        assert!((d.shift.y - cfg.max_y).abs() < 1e-9, "{:?}", d);
    }

    fn td(u_nom: (f64, f64), b_nom: (f64, f64)) -> Touchdown {
        Touchdown {
            ds_duration: 0.382,
            entry_offset: na::Vector2::new(-0.03, 0.0),
            u_nom: na::Vector2::new(u_nom.0, u_nom.1),
            b_nom: na::Vector2::new(b_nom.0, b_nom.1),
        }
    }

    /// The whole point: the coefficient of the landing in the touchdown
    /// residual. -2.25 on this robot against the 0.646 the other channel has.
    #[test]
    fn the_touchdown_channel_has_the_authority() {
        let w: f64 = 3.788;
        let tds: f64 = 0.382;
        let beta = 1.0 + (1.0 - (w * tds).exp()) / (w * tds);
        assert!((beta - 1.0 + 2.246).abs() < 0.01, "beta-1 = {}", beta - 1.0);
    }

    /// A step the plan already gets right is left alone: build `b_nom` from
    /// the propagation itself, then the minimiser is the nominal landing.
    #[test]
    fn a_consistent_touchdown_asks_for_no_shift() {
        let s = sit((0.02, -0.03), (0.0, 0.0));
        let mut t = td((0.14, 0.05), (0.0, 0.0));
        let cfg = Config { t_min_frac: 1.0, t_max_frac: 1.0, w_shift: 1.0, ..Config::default() };
        // What the DCM actually is at touchdown with the nominal landing.
        let big_t = s.nominal_end - s.t;
        let xi1 = predict(&s, big_t);
        let pa = s.p + s.p_dot * big_t;
        let e2 = (s.omega * t.ds_duration).exp();
        let c = t.entry_offset + t.u_nom;
        let v = (c - pa) / t.ds_duration;
        let xi_td = c + v / s.omega + (xi1 - pa - v / s.omega) * e2;
        t.b_nom = xi_td - t.u_nom;
        let d = plan_touchdown(&s, &t, &cfg);
        assert!(d.shift.norm() < 1e-9, "{:?}", d);
        assert!(d.unmet < 1e-9, "{:?}", d);
    }

    /// A DCM that will land 5 cm too far forward moves the foot FORWARD.
    #[test]
    fn a_fast_dcm_moves_the_landing_out() {
        let s = sit((0.02, -0.03), (0.0, 0.0));
        let mut t = td((0.14, 0.05), (0.0, 0.0));
        let cfg = Config { t_min_frac: 1.0, t_max_frac: 1.0, w_shift: 0.0, ..Config::default() };
        let big_t = s.nominal_end - s.t;
        let xi1 = predict(&s, big_t);
        let pa = s.p + s.p_dot * big_t;
        let e2 = (s.omega * t.ds_duration).exp();
        let c = t.entry_offset + t.u_nom;
        let v = (c - pa) / t.ds_duration;
        let xi_td = c + v / s.omega + (xi1 - pa - v / s.omega) * e2;
        // Ask for the DCM to sit 5 cm FURTHER BACK than it will: the foot has
        // to come forward, because pressure ahead slows the divergence.
        t.b_nom = xi_td - t.u_nom - na::Vector2::new(0.05, 0.0);
        let d = plan_touchdown(&s, &t, &cfg);
        assert!(d.shift.x > 0.0, "{:?}", d);
    }

    #[test]
    fn the_shift_trades_effort_against_residual() {
        // w_u -> 0 recovers the old "cancel it all" behaviour; a large w_u
        // barely moves the foot. Both with the same residual.
        let s0 = sit((0.02, -0.03), (0.0, 0.0));
        let target = predict(&s0, 0.4);
        let s = Situation { xi_eos_plan: target - na::Vector2::new(0.05, 0.0), ..s0 };
        let soft = Config { w_shift: 0.0, t_min_frac: 1.0, t_max_frac: 1.0, ..Config::default() };
        let stiff = Config { w_shift: 1e4, t_min_frac: 1.0, t_max_frac: 1.0, ..Config::default() };
        let du_soft = plan(&s, &soft).shift.x;
        let du_stiff = plan(&s, &stiff).shift.x;
        assert!(du_soft > 4.0 * du_stiff.abs().max(1e-9), "soft {du_soft} stiff {du_stiff}");
        // The soft one is the full cancellation resid/j, up to the clamp.
        assert!((du_soft - 0.05 / soft.j).abs() < 1e-3 || du_soft >= soft.max_x - 1e-9);
        // And w_shift = 0 is exactly the shipped closed form.
        let plain = Config { t_min_frac: 1.0, t_max_frac: 1.0, ..Config::default() };
        assert_eq!(plan(&s, &plain).shift.x, du_soft);
    }

    #[test]
    fn undisturbed_step_is_left_alone() {
        // Build the "plan" so that the nominal end already lands the DCM on
        // the target: then the best decision is no shift and no retime.
        let s0 = sit((0.02, -0.03), (0.0, 0.0));
        let target = predict(&s0, 0.4);
        let s = Situation { xi_eos_plan: target, ..s0 };
        let d = plan(&s, &Config::default());
        assert!(d.shift.norm() < 1e-9, "{:?}", d);
        assert!(d.new_end.is_none(), "{:?}", d);
    }

    #[test]
    fn lateral_escape_moves_the_foot_outward_and_ends_earlier() {
        // DCM already 3 cm outboard of where the plan expects it at the end:
        // the planner should push the landing outward (same sign) and not
        // wait longer.
        let s0 = sit((0.02, -0.03), (0.0, 0.0));
        let target = predict(&s0, 0.4);
        let s = Situation {
            xi: s0.xi + na::Vector2::new(0.0, -0.03),
            xi_eos_plan: target,
            ..s0
        };
        let d = plan(&s, &Config::default());
        assert!(d.shift.y < 0.0, "{:?}", d);
        if let Some(e) = d.new_end {
            assert!(e <= s.nominal_end + 1e-9, "{:?}", d);
        }
    }
}
