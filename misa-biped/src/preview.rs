//! Receding-horizon horizontal CoM planning with bounded ZMP controls.
//!
//! Each solve starts at the measured CoM state. The caller supplies the
//! contact schedule (including delayed touchdowns) and its support bounds.
//! This is a constant-height LIPM approximation, not a full centroidal MPC.
use misa_wbc::qp::{QpConfig, QpStatus, solve_qp};
use nalgebra as na;

use super::dcm::SupportBox;

pub struct PreviewSample {
    pub duration: f64,
    pub bounds: SupportBox,
    pub zmp: na::Vector2<f64>,
    pub xi: na::Vector2<f64>,
}

pub struct ComPreview {
    pub zmp: Vec<na::Vector2<f64>>,
    /// State at each interval end; the initial state is supplied by the caller.
    pub com: Vec<na::Vector2<f64>>,
    pub velocity: Vec<na::Vector2<f64>>,
    pub terminal_error: f64,
}

/// Solve two convex QPs for piecewise-constant pressure positions. Position
/// and velocity follow the exact LIPM transition over each sample. DCM error
/// is soft (the current state can be outside the recoverable set), while
/// every pressure bound is hard. A failed/non-finite solve returns None.
pub fn replan(
    omega: f64,
    com: na::Vector2<f64>,
    velocity: na::Vector2<f64>,
    previous_zmp: na::Vector2<f64>,
    samples: &[PreviewSample],
    smooth_weight: f64,
) -> Option<ComPreview> {
    let n = samples.len();
    if n == 0
        || !omega.is_finite()
        || omega <= 0.0
        || !smooth_weight.is_finite()
        || smooth_weight < 0.0
        || com
            .iter()
            .chain(velocity.iter())
            .chain(previous_zmp.iter())
            .any(|x| !x.is_finite())
    {
        return None;
    }
    let mut a = na::DMatrix::zeros(3 * n, n);
    let mut b = na::DMatrix::zeros(3 * n, 2);
    let mut cm = na::DVector::zeros(n);
    let mut vm = cm.clone();
    let (mut cf, mut vf) = (com, velocity);
    for (k, s) in samples.iter().enumerate() {
        if s.zmp.iter().chain(s.xi.iter()).any(|x| !x.is_finite())
            || !s.duration.is_finite()
            || s.duration <= 0.0
            || (0..2).any(|j| {
                !s.bounds.lo[j].is_finite()
                    || !s.bounds.hi[j].is_finite()
                    || s.bounds.lo[j] > s.bounds.hi[j]
            })
        {
            return None;
        }
        let ch = (omega * s.duration).cosh();
        let sh = (omega * s.duration).sinh();
        let nc = &cm * ch + &vm * (sh / omega);
        vm = &cm * (omega * sh) + &vm * ch;
        cm = nc;
        cm[k] += 1.0 - ch;
        vm[k] -= omega * sh;
        let next = cf * ch + vf * (sh / omega);
        vf = cf * (omega * sh) + vf * ch;
        cf = next;
        let weight = if k + 1 == n { 5.0_f64.sqrt() } else { 1.0 };
        a.row_mut(k)
            .copy_from(&((&cm + &vm / omega) * weight).transpose());
        for j in 0..2 {
            b[(k, j)] = (cf[j] + vf[j] / omega - s.xi[j]) * weight;
            b[(n + k, j)] = -s.zmp[j] * 0.03_f64.sqrt();
        }
        a[(n + k, k)] = 0.03_f64.sqrt();
        let w = smooth_weight.sqrt();
        a[(2 * n + k, k)] = w;
        if k > 0 {
            a[(2 * n + k, k - 1)] = -w;
        } else {
            for j in 0..2 {
                b[(2 * n + k, j)] = -previous_zmp[j] * w;
            }
        }
    }
    let h = a.transpose() * &a + na::DMatrix::identity(n, n) * 1e-10;
    let g = a.transpose() * b;
    if h.iter().chain(g.iter()).any(|x| !x.is_finite()) {
        return None;
    }
    let mut constraints = na::DMatrix::zeros(2 * n, n);
    for k in 0..n {
        constraints[(k, k)] = 1.0;
        constraints[(n + k, k)] = -1.0;
    }
    let mut zmp = vec![na::Vector2::zeros(); n];
    for j in 0..2 {
        let mut rhs = na::DVector::zeros(2 * n);
        let mut initial = na::DVector::zeros(n);
        for (k, s) in samples.iter().enumerate() {
            rhs[k] = s.bounds.hi[j];
            rhs[n + k] = -s.bounds.lo[j];
            initial[k] = s.zmp[j].clamp(s.bounds.lo[j], s.bounds.hi[j]);
        }
        let solution = solve_qp(
            &h,
            &g.column(j).into_owned(),
            None,
            None,
            Some(&constraints),
            Some(&rhs),
            Some(&initial),
            &QpConfig::default(),
        );
        if solution.status != QpStatus::Optimal || solution.x.iter().any(|x| !x.is_finite()) {
            return None;
        }
        for (k, s) in samples.iter().enumerate() {
            if solution.x[k] < s.bounds.lo[j] - 1e-6 || solution.x[k] > s.bounds.hi[j] + 1e-6 {
                return None;
            }
            zmp[k][j] = solution.x[k].clamp(s.bounds.lo[j], s.bounds.hi[j]);
        }
    }
    let (mut c, mut v) = (com, velocity);
    let (mut cs, mut vs) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for (s, p) in samples.iter().zip(&zmp) {
        let ch = (omega * s.duration).cosh();
        let sh = (omega * s.duration).sinh();
        let next = *p + (c - *p) * ch + v * (sh / omega);
        v = (c - *p) * (omega * sh) + v * ch;
        c = next;
        cs.push(c);
        vs.push(v);
    }
    let terminal_error = (c + v / omega - samples[n - 1].xi).norm();
    Some(ComPreview {
        zmp,
        com: cs,
        velocity: vs,
        terminal_error,
    })
}

/// Contact-triggered stair transfer shared by the horizontal pressure,
/// vertical CoM and load references.  All three use one minimum-jerk clock,
/// so none of them can commit to the upper foot before the others.
#[derive(Clone, Copy, Debug)]
pub struct StairTransferPlan {
    pub start: f64,
    pub duration: f64,
    pub p0: na::Vector2<f64>,
    pub p1: na::Vector2<f64>,
    pub z0: f64,
    pub z1: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct StairTransferRef {
    pub alpha: f64,
    pub p: na::Vector2<f64>,
    pub z: f64,
    pub zd: f64,
    pub zdd: f64,
}

impl StairTransferPlan {
    pub fn at(&self, t: f64) -> StairTransferRef {
        let u = ((t - self.start) / self.duration.max(1e-9)).clamp(0.0, 1.0);
        let alpha = u.powi(3) * (10.0 + u * (-15.0 + 6.0 * u));
        let alpha_d = 30.0 * u.powi(2) * (1.0 - u).powi(2) / self.duration.max(1e-9);
        let alpha_dd = 60.0 * u * (1.0 - u) * (1.0 - 2.0 * u) / self.duration.max(1e-9).powi(2);
        let dz = self.z1 - self.z0;
        StairTransferRef {
            alpha,
            p: self.p0 + (self.p1 - self.p0) * alpha,
            z: self.z0 + dz * alpha,
            zd: dz * alpha_d,
            zdd: dz * alpha_dd,
        }
    }
}

/// Quintic height segment with continuous position, velocity and acceleration
/// across a retarget. End velocity/acceleration are zero.
pub struct HeightPlan {
    start: f64,
    duration: f64,
    coefficients: [f64; 6],
    pub target: f64,
}
impl HeightPlan {
    pub fn constant(z: f64) -> Self {
        Self {
            start: 0.0,
            duration: 1.0,
            coefficients: [z, 0.0, 0.0, 0.0, 0.0, 0.0],
            target: z,
        }
    }
    pub fn at(&self, t: f64) -> (f64, f64, f64) {
        if t >= self.start + self.duration {
            return (self.target, 0.0, 0.0);
        }
        let x = (t - self.start).max(0.0);
        let a = &self.coefficients;
        (
            a[0] + x * (a[1] + x * (a[2] + x * (a[3] + x * (a[4] + x * a[5])))),
            a[1] + x * (2.0 * a[2] + x * (3.0 * a[3] + x * (4.0 * a[4] + x * 5.0 * a[5]))),
            2.0 * a[2] + x * (6.0 * a[3] + x * (12.0 * a[4] + x * 20.0 * a[5])),
        )
    }
    pub fn retarget(&mut self, t: f64, target: f64, duration: f64) {
        assert!(duration > 0.0 && duration.is_finite());
        let (z, v, acc) = self.at(t);
        let d = target - z;
        let h = duration;
        self.coefficients = [
            z,
            v,
            0.5 * acc,
            (10.0 * d - 6.0 * v * h - 1.5 * acc * h * h) / h.powi(3),
            (-15.0 * d + 8.0 * v * h + 1.5 * acc * h * h) / h.powi(4),
            (6.0 * d - 3.0 * v * h - 0.5 * acc * h * h) / h.powi(5),
        ];
        self.start = t;
        self.duration = h;
        self.target = target;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(lo: f64, hi: f64) -> PreviewSample {
        PreviewSample {
            duration: 0.05,
            bounds: SupportBox {
                lo: na::Vector2::repeat(lo),
                hi: na::Vector2::repeat(hi),
            },
            zmp: na::Vector2::zeros(),
            xi: na::Vector2::zeros(),
        }
    }
    #[test]
    fn equilibrium_stays_at_rest() {
        let samples: Vec<_> = (0..20).map(|_| sample(-0.04, 0.04)).collect();
        let p = replan(
            5.4,
            na::Vector2::zeros(),
            na::Vector2::zeros(),
            na::Vector2::zeros(),
            &samples,
            0.1,
        )
        .unwrap();
        assert!(p.com.iter().all(|c| c.norm() < 1e-9));
        assert!(p.terminal_error < 1e-9);
    }
    #[test]
    fn disturbed_state_respects_pressure_bounds_and_exact_dynamics() {
        let samples: Vec<_> = (0..20).map(|_| sample(-0.03, 0.03)).collect();
        let c = na::Vector2::new(0.01, -0.005);
        let v = na::Vector2::new(0.08, -0.03);
        let p = replan(5.4, c, v, na::Vector2::zeros(), &samples, 0.1).unwrap();
        assert!(p.zmp.iter().all(|u| u.iter().all(|x| x.abs() <= 0.030001)));
        let expected = p.zmp[0]
            + (c - p.zmp[0]) * (5.4_f64 * 0.05).cosh()
            + v * ((5.4_f64 * 0.05).sinh() / 5.4);
        assert!((p.com[0] - expected).norm() < 1e-12);
        assert!(p.terminal_error < 0.01);
    }
    #[test]
    fn unavailable_future_support_cannot_be_used_early() {
        let samples: Vec<_> = (0..20)
            .map(|i| {
                let mut s = sample(
                    if i < 10 { -0.04 } else { 0.06 },
                    if i < 10 { 0.04 } else { 0.14 },
                );
                s.xi = na::Vector2::repeat(0.10);
                s
            })
            .collect();
        let p = replan(
            5.4,
            na::Vector2::zeros(),
            na::Vector2::zeros(),
            na::Vector2::zeros(),
            &samples,
            0.1,
        )
        .unwrap();
        assert!(p.zmp[..10].iter().all(|u| u.x <= 0.04));
        assert!(p.zmp[10..].iter().all(|u| u.x >= 0.06));
    }
    #[test]
    fn unreachable_capture_state_reports_residual_instead_of_inventing_support() {
        let samples: Vec<_> = (0..20).map(|_| sample(-0.03, 0.03)).collect();
        let plan = replan(
            5.4,
            na::Vector2::repeat(0.04),
            na::Vector2::zeros(),
            na::Vector2::zeros(),
            &samples,
            0.1,
        )
        .unwrap();
        assert!(plan.zmp.iter().all(|p| p.x <= 0.030001 && p.y <= 0.030001));
        assert!(plan.terminal_error > 0.1);
    }
    #[test]
    fn nonfinite_state_is_rejected_before_entering_solver() {
        assert!(
            replan(
                5.4,
                na::Vector2::repeat(f64::NAN),
                na::Vector2::zeros(),
                na::Vector2::zeros(),
                &[sample(-0.04, 0.04)],
                0.1
            )
            .is_none()
        );
    }
    #[test]
    fn contradictory_bounds_are_rejected() {
        assert!(
            replan(
                5.4,
                na::Vector2::zeros(),
                na::Vector2::zeros(),
                na::Vector2::zeros(),
                &[sample(0.04, -0.04)],
                0.1
            )
            .is_none()
        );
    }
    #[test]
    fn stair_transfer_uses_one_smooth_clock_for_pressure_height_and_load() {
        let plan = StairTransferPlan {
            start: 2.0,
            duration: 0.8,
            p0: na::Vector2::new(0.0, -0.05),
            p1: na::Vector2::new(0.12, 0.05),
            z0: 0.34,
            z1: 0.392,
        };
        let a = plan.at(2.0);
        let m = plan.at(2.4);
        let b = plan.at(2.8);
        assert!(a.alpha.abs() < 1e-12 && b.alpha > 1.0 - 1e-12);
        assert!((a.p - plan.p0).norm() < 1e-12);
        assert!((b.p - plan.p1).norm() < 1e-12);
        assert!((m.alpha - 0.5).abs() < 1e-12);
        assert!((m.z - 0.366).abs() < 1e-12);
        assert!(a.zd.abs() < 1e-12 && a.zdd.abs() < 1e-12);
        assert!(b.zd.abs() < 1e-12 && b.zdd.abs() < 1e-12);
    }

    #[test]
    fn height_retarget_preserves_position_velocity_and_acceleration() {
        let mut p = HeightPlan::constant(0.34);
        p.retarget(1.0, 0.392, 0.6);
        let old = p.at(1.2);
        p.retarget(1.2, 0.444, 0.6);
        let new = p.at(1.2);
        assert!(
            (old.0 - new.0).abs() < 1e-12
                && (old.1 - new.1).abs() < 1e-12
                && (old.2 - new.2).abs() < 1e-12
        );
        let end = p.at(1.8);
        assert!((end.0 - 0.444).abs() < 1e-12 && end.1.abs() < 1e-12 && end.2.abs() < 1e-12);
    }
}

/// Keep a currently planted foot at its measured position throughout the
/// preview, until its next planned lift-off. Using the measurement only for
/// the first sample makes the foot jump back to its nominal target 50 ms
/// later, which invents support after a placement error.
pub fn forecast_feet(
    gait: &super::gait::GaitPlan,
    feet: &super::gait::FootstepPlan,
    actual: &super::gait::Footsteps,
    current_sides: &[usize],
    now: f64,
    at: f64,
) -> super::gait::Footsteps {
    let mut result = *feet.at_slice(gait.index_at(at));
    for &side in current_sides {
        let lifted = gait
            .slices
            .iter()
            .any(|sl| sl.t1 > now && sl.t0 <= at && sl.support.swing() == Some(side));
        if !lifted {
            result.sole[side] = actual.sole[side];
            result.yaw[side] = actual.yaw[side];
        }
    }
    result
}

/// Arrival-time estimate used before constructing the preview support horizon.
/// The trajectory clock remains nominal even when support is extended.
pub struct LandingTiming {
    pub max_extension: f64,
    /// Planned sole-height change above which a swing counts as a stair
    /// swing (confirmed landing, extension). 0.005 = the shipped test; a
    /// slope sets it higher so its 10-15 mm steps stay level steps.
    pub dz_min: f64,
    /// Per-swing override of the confirmation's xy tolerance (the caller
    /// sets it for the swing in progress). `None` = `LAND_CONFIRM_XY`.
    pub xy_tol_override: Option<f64>,
    /// z tolerance that goes with `xy_tol_override` (applied only while that
    /// override is set). `None` = `LAND_CONFIRM_Z`.
    pub z_tol_override: Option<f64>,
    /// Per-swing override of the confirmation's force gate [N] (the caller
    /// sets it for the swing in progress). `None` = the constructor's value.
    pub load_threshold_override: Option<f64>,
    /// Event-driven contact (doc/from_claude/20261004_event_driven_contact_design.md).
    /// Touchdown is the measured event "load above the gate with the sole
    /// within `event_z_tol` of its planned height" -- the xy error is NOT a
    /// gate. The event is reported once in `last_event` so the caller can
    /// move the plan onto the foot. A swing with `event_end_on_touch` ends
    /// at the event; otherwise it keeps its extension as an explicit dwell
    /// and ends at the deadline. With no event by the deadline the swing is
    /// kept open while the caller lowers the foot by `probe_depth`, and only
    /// past `probe_max` does it time out. false = the historical machine.
    pub event_mode: bool,
    pub event_z_tol: f64,
    pub event_end_on_touch: bool,
    pub probe_speed: f64,
    pub probe_max: f64,
    /// How far below its target the caller should hold the swing foot [m].
    pub probe_depth: f64,
    event_slice: Option<usize>,
    pub last_event: Option<(usize, usize, na::Vector2<f64>)>,
    pub events: u32,
    pub probes: u32,
    pub min_fraction: f64,
    /// Confirm LEVEL swings too, not only ones that change height.
    ///
    /// The height test below exists because this was built for the stair,
    /// where the swing that matters is the one crossing a riser. After the
    /// ascent the robot is on the landing and every remaining step is level,
    /// so the test switches the whole mechanism off exactly where the gait
    /// resumes -- and the only other landing confirmation, `ContactPhase`,
    /// is mutually exclusive with `COM_REPLAN`. Measured on the fixed-arm
    /// one-step, support then transferred on the clock with 3.66 N (5% of
    /// body weight) under the incoming foot and 13.4 mm of swing Z error.
    pub level_steps: bool,
    pub speed_xy: f64,
    pub speed_z: f64,
    pub load_threshold: f64,
    pub clock: Option<(usize, f64, f64)>,
    /// Optional earlier contact-check threshold for one scheduled swing.
    /// The default remains 65% of its original duration.
    pub early_confirm_slice: Option<(usize, f64)>,
    active: Option<usize>,
    finished: Option<usize>,
    confirmations: u32,
    pub timeouts: u32,
    pub touchdowns: u32,
    /// P2-2 (doc/from_claude/20260923_p2_contact_planner_design_v2.md):
    /// confirm the landing on REAL LOAD, regardless of the xy position error,
    /// as long as the sole is at the planned height. Measured 2026-09-23 on
    /// G1, 80 mm riser: runs whose swing sole reached tread height with an
    /// xy error above the 12 mm gate were never confirmed, rested on the
    /// tread unloaded and fell 2-3 s later (13 of 15); loosening the gate
    /// instead confirmed feet still in the air and was catastrophic. Load is
    /// the discriminator position is not. Default false = previous behaviour.
    pub load_confirm: bool,
    /// Force on the swing sole that counts as "landed" [N]; higher than
    /// `load_threshold` so a brushing contact does not qualify.
    pub load_confirm_threshold: f64,
    pub load_confirm_ticks: u32,
    load_confirmations: u32,
    /// Set when a landing was confirmed by load: (slice, side, plan - actual
    /// xy). The caller moves the plan onto the foot (`shift_next`).
    pub last_load_confirm: Option<(usize, usize, na::Vector2<f64>)>,
    pub load_confirm_count: u32,
}
impl LandingTiming {
    pub fn new(load_threshold: f64) -> Self {
        Self {
            dz_min: 0.005,
            max_extension: 0.25,
            xy_tol_override: None,
            z_tol_override: None,
            load_threshold_override: None,
            event_mode: false,
            event_z_tol: 0.003,
            event_end_on_touch: false,
            probe_speed: 0.05,
            probe_max: 0.03,
            probe_depth: 0.0,
            event_slice: None,
            last_event: None,
            events: 0,
            probes: 0,
            min_fraction: 0.0,
            level_steps: false,
            speed_xy: 0.35,
            speed_z: 0.20,
            load_threshold,
            clock: None,
            early_confirm_slice: None,
            active: None,
            finished: None,
            confirmations: 0,
            timeouts: 0,
            touchdowns: 0,
            load_confirm: false,
            load_confirm_threshold: 0.0,
            load_confirm_ticks: 3,
            load_confirmations: 0,
            last_load_confirm: None,
            load_confirm_count: 0,
        }
    }
    pub fn update(
        &mut self,
        gait: &mut super::gait::GaitPlan,
        feet: &super::gait::FootstepPlan,
        actual: &super::gait::Footsteps,
        fz: [f64; 2],
        t: f64,
        dt: f64,
    ) -> f64 {
        if self.active.is_none() {
            let i = gait.index_at(t);
            if self.finished != Some(i) {
                if let Some(sw) = gait.slices[i].support.swing() {
                    let from = feet.at_slice(i.saturating_sub(1)).sole[sw].z;
                    if self.level_steps || (feet.at_slice(i).sole[sw].z - from).abs() > self.dz_min {
                        let sl = gait.slices[i];
                        if std::env::var("LAND_TIMING_DEBUG").is_ok() {
                            println!("  [lt] t={t:.3} activate slice={i} sw={sw} z {from:.3} -> {:.3} t0={:.3} dur={:.3}", feet.at_slice(i).sole[sw].z, sl.t0, sl.duration());
                        }
                        self.active = Some(i);
                        self.probe_depth = 0.0;
                        self.clock = Some((i, sl.t0, sl.duration()));
                        self.confirmations = 0;
                        self.load_confirmations = 0;
                    }
                }
            }
        }
        let Some(i) = self.active else { return 0.0 };
        let (_, start, duration) = self.clock.unwrap();
        let sw = gait.slices[i].support.swing().unwrap();
        let error = feet.at_slice(i).sole[sw] - actual.sole[sw];
        if t < start + duration * self.min_fraction {
            return 0.0;
        }
        let confirm_fraction = self
            .early_confirm_slice
            .filter(|(slice, _)| *slice == i)
            .map_or(0.65, |(_, fraction)| fraction);
        // Position tolerances of the confirmation. Historically the constants
        // 0.012 / 0.010. Measured 2026-09-23 on G1, 80 mm riser, T=40, landing
        // 3.0 m, transfer not armed: at the tick the swing sole first reached
        // tread height, runs that later fell had a median xy error of 21.5 mm
        // (13 of 15 above 12 mm) and runs that survived 3.7 mm (3 of 18). A
        // landing that misses the gate is never confirmed, the foot rests on
        // the tread UNLOADED (fz ~ 0 while the sole sits at tread height), the
        // stance leg is held for another 2-3 s and the lateral DCM escapes.
        // The tolerances are the discriminator, so they are knobs; defaults
        // reproduce the constants.
        if self.event_mode {
            return self.update_event(gait, i, sw, error, fz[sw], start, duration, confirm_fraction, t, dt);
        }
        let xy_tol: f64 = std::env::var("LAND_CONFIRM_XY").ok().and_then(|v| v.parse().ok()).unwrap_or(0.012);
        let z_tol: f64 = std::env::var("LAND_CONFIRM_Z").ok().and_then(|v| v.parse().ok()).unwrap_or(0.010);
        let z_tol = match (self.xy_tol_override, self.z_tol_override) {
            (Some(_), Some(z)) => z,
            _ => z_tol,
        };
        let landed = t >= start + duration * confirm_fraction
            && fz[sw] > self.load_threshold_override.unwrap_or(self.load_threshold)
            && match self.xy_tol_override {
                // The override widens the FORWARD axis only (the swing's x
                // overshoot); the lateral error keeps the normal tolerance.
                // Measured 2026-10-03: widening both confirmed a landing
                // 27 mm off laterally and the next step fell (1 step x 8 cm
                // 48 -> 44).
                Some(x_tol) => error.x.abs() < x_tol && error.y.abs() < xy_tol,
                None => error.xy().norm() < xy_tol,
            }
            && error.z.abs() < z_tol;
        self.confirmations = if landed { self.confirmations + 1 } else { 0 };
        if self.confirmations >= 2 {
            if std::env::var("LAND_TIMING_DEBUG").is_ok() {
                println!("  [lt] t={t:.3} CONFIRM slice={i} sw={sw} fz={:.0} err=({:+.3},{:+.3},{:+.3})", fz[sw], error.x, error.y, error.z);
            }
            self.active = None;
            self.finished = Some(i);
            self.touchdowns += 1;
            return gait.retime(i, t);
        }
        if self.load_confirm {
            let by_load = t >= start + duration * confirm_fraction
                && fz[sw] > self.load_confirm_threshold
                && error.z.abs() < z_tol;
            self.load_confirmations = if by_load { self.load_confirmations + 1 } else { 0 };
            if self.load_confirmations >= self.load_confirm_ticks {
                self.active = None;
                self.finished = Some(i);
                self.touchdowns += 1;
                self.load_confirm_count += 1;
                self.last_load_confirm = Some((i, sw, error.xy()));
                return gait.retime(i, t);
            }
        }
        let deadline = start + duration + self.max_extension;
        if t >= deadline - 1e-9 {
            if std::env::var("LAND_TIMING_DEBUG").is_ok() {
                println!("  [lt] t={t:.3} TIMEOUT slice={i} sw={sw} fz={:.0} err=({:+.3},{:+.3},{:+.3}) ext={:.3}", fz[sw], error.x, error.y, error.z, self.max_extension);
            }
            self.timeouts += 1;
            self.active = None;
            self.finished = Some(i);
            return 0.0;
        }
        let remaining =
            (error.xy().norm() / self.speed_xy).max(error.z.abs() / self.speed_z) + 0.03;
        let end = (t + remaining.max(dt)).min(deadline).max(gait.slices[i].t1);
        if end > gait.slices[i].t1 + 1e-6 {
            gait.retime(i, end)
        } else {
            0.0
        }
    }

    /// Event mode: true once the swing of slice `i` has touched down or has
    /// started probing -- from then on its landing must not be moved.
    pub fn landing_closed(&self, i: usize) -> bool {
        self.event_mode
            && (self.event_slice == Some(i)
                || (self.probe_depth > 0.0 && self.clock.is_some_and(|(ci, _, _)| ci == i)))
    }

    #[allow(clippy::too_many_arguments)]
    fn update_event(
        &mut self,
        gait: &mut super::gait::GaitPlan,
        i: usize,
        sw: usize,
        error: na::Vector3<f64>,
        fz: f64,
        start: f64,
        duration: f64,
        confirm_fraction: f64,
        t: f64,
        dt: f64,
    ) -> f64 {
        let debug = std::env::var("LAND_TIMING_DEBUG").is_ok();
        let deadline = start + duration + self.max_extension;
        let touching = t >= start + duration * confirm_fraction
            && fz > self.load_threshold_override.unwrap_or(self.load_threshold)
            // While probing the target is BELOW the plan, so the sole may be
            // lower than planned by up to the probe depth.
            // (error = plan - actual: negative z = the sole is above plan.)
            && error.z > -self.event_z_tol
            && error.z < self.event_z_tol + self.probe_depth;
        self.confirmations = if touching { self.confirmations + 1 } else { 0 };
        if self.confirmations >= 2 && self.event_slice != Some(i) {
            self.event_slice = Some(i);
            self.last_event = Some((i, sw, error.xy()));
            self.events += 1;
            if debug {
                println!("  [lt] t={t:.3} EVENT slice={i} sw={sw} fz={fz:.0} err=({:+.3},{:+.3},{:+.3}) probe={:.3}", error.x, error.y, error.z, self.probe_depth);
            }
        }
        let evented = self.event_slice == Some(i);
        if evented && (self.event_end_on_touch || t >= deadline - 1e-9) {
            if debug {
                println!("  [lt] t={t:.3} END slice={i} sw={sw} (event, {})", if t >= deadline - 1e-9 { "dwell over" } else { "on touch" });
            }
            self.active = None;
            self.finished = Some(i);
            self.probe_depth = 0.0;
            if t >= deadline - 1e-9 {
                // The dwell ran out exactly as the historical extension did;
                // report it the same way so the caller's timeout path runs.
                self.timeouts += 1;
                return 0.0;
            }
            self.touchdowns += 1;
            return gait.retime(i, t);
        }
        // The sole is at its planned height but carries less than the gate:
        // it is resting on the surface with the swing task still holding the
        // leg (measured 2026-10-04: 13 N against a 17 N gate, the wait ran to
        // the deadline while the DCM error grew 0.02 -> 0.16). Press it down
        // now rather than wait.
        let resting = !evented
            && t >= start + duration * confirm_fraction
            && error.z.abs() < self.event_z_tol + self.probe_depth;
        if (t >= deadline - 1e-9 || resting) && !evented {
            // No contact by the deadline (or contact without load): probe
            // down instead of declaring double support over a foot that is
            // still in the air.
            let depth = self.probe_depth + self.probe_speed * dt;
            if depth > self.probe_max && t >= deadline - 1e-9 {
                if debug {
                    println!("  [lt] t={t:.3} TIMEOUT slice={i} sw={sw} fz={fz:.0} err=({:+.3},{:+.3},{:+.3}) after probe {:.3}", error.x, error.y, error.z, self.probe_depth);
                }
                self.timeouts += 1;
                self.active = None;
                self.finished = Some(i);
                self.probe_depth = 0.0;
                return 0.0;
            }
            if self.probe_depth == 0.0 {
                self.probes += 1;
                if debug {
                    println!("  [lt] t={t:.3} PROBE slice={i} sw={sw} fz={fz:.0} err=({:+.3},{:+.3},{:+.3})", error.x, error.y, error.z);
                }
            }
            self.probe_depth = depth.min(self.probe_max).max(1e-9);
            let end = (t + 0.03_f64.max(dt)).max(gait.slices[i].t1);
            return if end > gait.slices[i].t1 + 1e-6 { gait.retime(i, end) } else { 0.0 };
        }
        // Before the deadline: the historical rolling extension.
        let remaining =
            (error.xy().norm() / self.speed_xy).max(error.z.abs() / self.speed_z) + 0.03;
        let end = (t + remaining.max(dt)).min(deadline).max(gait.slices[i].t1);
        if end > gait.slices[i].t1 + 1e-6 {
            gait.retime(i, end)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod landing_tests {
    use super::*;
    use crate::gait::{FootstepPlan, Footsteps, GaitParams, GaitPlan};
    fn setup(height: f64) -> (GaitPlan, FootstepPlan, Footsteps) {
        let gait = GaitPlan::new(&GaitParams::default());
        let actual = Footsteps {
            sole: [
                na::Vector3::new(0.0, 0.05, 0.001),
                na::Vector3::new(0.0, -0.05, 0.001),
            ],
            yaw: [0.0; 2],
        };
        let mut feet = FootstepPlan::constant_stride(&gait, &actual, 0.0);
        for f in feet.at.iter_mut().skip(1) {
            f.sole[1].z = height;
        }
        (gait, feet, actual)
    }
    #[test]
    fn measured_stance_is_kept_until_next_liftoff() {
        let (gait, feet, mut actual) = setup(0.052);
        actual.sole[0].x = 0.06;
        let before = forecast_feet(&gait, &feet, &actual, &[0], 2.0, 2.5);
        assert_eq!(before.sole[0].x, 0.06);
        let after = forecast_feet(&gait, &feet, &actual, &[0], 2.0, 3.2);
        assert_eq!(after.sole[0].x, feet.at_slice(gait.index_at(3.2)).sole[0].x);
        assert_ne!(after.sole[0].x, 0.06);
    }
    #[test]
    fn settled_clearance_is_not_a_stair() {
        let (mut gait, feet, actual) = setup(0.0);
        let mut timing = LandingTiming::new(5.0);
        for k in 0..100 {
            assert_eq!(
                timing.update(
                    &mut gait,
                    &feet,
                    &actual,
                    [50.0, 0.0],
                    2.0 + k as f64 * 0.005,
                    0.005
                ),
                0.0
            );
        }
        assert!(timing.clock.is_none());
    }
    #[test]
    fn waiting_does_not_rewind_swing_or_move_the_deadline() {
        let (mut gait, feet, mut actual) = setup(0.052);
        let mut timing = LandingTiming::new(5.0);
        actual.sole[1].z = 0.12;
        timing.update(&mut gait, &feet, &actual, [50.0, 0.0], 2.0, 0.005);
        for k in 0..150 {
            timing.update(
                &mut gait,
                &feet,
                &actual,
                [50.0, 0.0],
                2.0 + k as f64 * 0.005,
                0.005,
            );
        }
        let (i, t, d) = timing.clock.unwrap();
        assert_eq!(i, 1);
        assert!((t - 2.0).abs() < 1e-12 && (d - 0.45).abs() < 1e-12);
        assert!((gait.slices[1].t1 - 2.70).abs() < 1e-9);
        assert_eq!(timing.timeouts, 1);
    }
    #[test]
    fn early_confirmation_only_applies_to_selected_swing() {
        let (mut gait, feet, actual) = setup(0.0);
        let mut timing = LandingTiming::new(5.0);
        timing.level_steps = true;
        timing.early_confirm_slice = Some((1, 0.65 * 0.70));
        timing.update(&mut gait, &feet, &actual, [50.0, 20.0], 2.0, 0.005);
        for t in [2.250, 2.255] {
            timing.update(&mut gait, &feet, &actual, [50.0, 20.0], t, 0.005);
        }
        assert_eq!(timing.touchdowns, 1);
        assert!((gait.slices[1].t1 - 2.255).abs() < 1e-9);

        let (mut gait, feet, actual) = setup(0.0);
        let mut timing = LandingTiming::new(5.0);
        timing.level_steps = true;
        timing.early_confirm_slice = Some((3, 0.65 * 0.70));
        timing.update(&mut gait, &feet, &actual, [50.0, 20.0], 2.0, 0.005);
        for t in [2.250, 2.255] {
            timing.update(&mut gait, &feet, &actual, [50.0, 20.0], t, 0.005);
        }
        assert_eq!(timing.touchdowns, 0);
    }
    #[test]
    fn load_without_tread_height_does_not_confirm_touchdown() {
        let (mut gait, feet, mut actual) = setup(0.052);
        let mut timing = LandingTiming::new(5.0);
        actual.sole[1].z = 0.08;
        timing.update(&mut gait, &feet, &actual, [50.0, 20.0], 2.0, 0.005);
        for t in [2.35, 2.355, 2.36] {
            timing.update(&mut gait, &feet, &actual, [50.0, 20.0], t, 0.005);
        }
        assert_eq!(timing.touchdowns, 0);
        actual.sole[1].z = 0.053;
        for t in [2.365, 2.37] {
            timing.update(&mut gait, &feet, &actual, [50.0, 20.0], t, 0.005);
        }
        assert_eq!(timing.touchdowns, 1);
        assert!((gait.slices[1].t1 - 2.37).abs() < 1e-9);
    }
}
