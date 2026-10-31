//! Balance reference from the divergent component of motion.
//!
//! Under the linear inverted pendulum model the CoM splits into a stable and
//! an unstable part. The unstable one,
//!
//! ```text
//! xi = c + c_dot / omega        omega = sqrt(g / z_com)
//! ```
//!
//! obeys `xi_dot = omega (xi - p)` where `p` is the ZMP: first order, and
//! directly steerable by where the pressure is put. The CoM then follows
//! `c_dot = -omega (c - xi)`, which is stable on its own. So planning the
//! ZMP and tracking the DCM is the whole balance problem, and it needs no QP
//! of its own -- the output is a commanded CoM acceleration, which is exactly
//! what the P1 task already consumes.
//!
//! # Why this replaces the latched lateral target
//!
//! Single-leg stance used a ramp to a lateral CoM target that was frozen on
//! the first tick (`LATCH_STANCE`). Freezing was correct as far as it went --
//! reading the stance foot's position every tick put the plant inside its own
//! reference and cost a fall with ZERO degraded solves. But a frozen target
//! cannot walk. The fix is not to unfreeze it; it is to generate it from the
//! FOOTSTEP PLAN, which is world-frame data that no measurement feeds back
//! into. If the foot moves, that is a disturbance to reject, not a new goal.

use nalgebra as na;

use super::gait::{FootstepPlan, Footsteps, GaitPlan, Support};

pub const G: f64 = 9.81;

/// One segment of the planned centre of pressure: linear from `p0` to `p1`
/// over `duration`. Constant segments are the `p0 == p1` case, so single and
/// double support share one closed form.
#[derive(Clone, Copy, Debug)]
pub struct ZmpSeg {
    pub p0: na::Vector2<f64>,
    pub p1: na::Vector2<f64>,
    pub t0: f64,
    pub duration: f64,
}

impl ZmpSeg {
    pub fn at(&self, t: f64) -> na::Vector2<f64> {
        let f = if self.duration <= 1e-12 {
            0.0
        } else {
            ((t - self.t0) / self.duration).clamp(0.0, 1.0)
        };
        self.p0 + (self.p1 - self.p0) * f
    }
    /// `d(p)/dt`, constant within the segment.
    pub fn slope(&self) -> na::Vector2<f64> {
        if self.duration <= 1e-12 {
            na::Vector2::zeros()
        } else {
            (self.p1 - self.p0) / self.duration
        }
    }
}

/// A planned ZMP trajectory and the DCM trajectory that lands on it.
#[derive(Clone)]
pub struct DcmPlan {
    pub omega: f64,
    pub segs: Vec<ZmpSeg>,
    /// DCM at the END of each segment, from the backward recursion.
    ///
    /// The END, not the start, and that is not a presentation choice. See
    /// [`DcmPlan::reference`]: evaluating forward from the start needs
    /// `exp(+omega t)`, and the DCM is the DIVERGENT component, so that
    /// exponential amplifies round-off instead of decaying it.
    pub xi_eos: Vec<na::Vector2<f64>>,
}

/// Instantaneous reference: where the DCM should be, how fast, and where the
/// pressure was planned to be while it does that.
#[derive(Clone, Copy, Debug)]
pub struct DcmRef {
    pub xi: na::Vector2<f64>,
    pub xi_dot: na::Vector2<f64>,
    pub p: na::Vector2<f64>,
}

impl DcmPlan {
    /// Build the ZMP plan from the support schedule and the footstep plan.
    ///
    /// Single support puts the pressure at the stance sole centre and holds
    /// it. Double support walks it linearly from the previous stance foot to
    /// the next one -- a step in the ZMP reference would be a step in the
    /// commanded CoM acceleration, and that lands straight in the contact.
    ///
    /// `lateral_scale` pulls the single-support target back toward mid-stance:
    /// 1.0 is the sole centre, 0.0 never leaves the middle. It is not a fudge
    /// factor, it is the lateral authority the machine is being asked for --
    /// full transfer means the CoM leans out by the half stance width, and on
    /// kyo46rs that alone consumes 76% of the ankle_roll travel with both feet
    /// still flat on the floor. Below 1.0 the plan no longer reaches true
    /// single support, so it is only meaningful while both feet are down.
    pub fn from_schedule(
        plan: &GaitPlan,
        steps: &Footsteps,
        z_com: f64,
        lateral_scale: f64,
    ) -> Self {
        let fp = FootstepPlan::constant_stride(plan, steps, 0.0);
        Self::from_footsteps(plan, &fp, z_com, lateral_scale)
    }

    /// The walking form: the pressure target comes from the footstep the
    /// stance foot is standing on IN THAT SLICE, not from a fixed pair.
    pub fn from_footsteps(
        plan: &GaitPlan,
        fp: &FootstepPlan,
        z_com: f64,
        lateral_scale: f64,
    ) -> Self {
        Self::from_footsteps_rolled(plan, fp, z_com, lateral_scale, 0.0)
    }

    /// [`Self::from_footsteps`], with the single-support pressure ROLLING
    /// heel-to-toe: it enters each stance `roll` metres behind the sole
    /// centre (along the stance yaw) and leaves `roll` ahead, instead of
    /// dwelling at the centre. A dwell plan advances the reference in jumps
    /// of one footstep pitch; when the pitch is set by terrain (a staircase
    /// tread) rather than by the commanded speed, those jumps outrun what
    /// the sole can drive. Rolling spends the sole's own length on forward
    /// travel -- 2*roll of every pitch happens inside the stance -- which is
    /// what a human foot does. `roll` must stay inside the CoP box or the
    /// reference asks for pressure off the sole.
    pub fn from_footsteps_rolled(
        plan: &GaitPlan,
        fp: &FootstepPlan,
        z_com: f64,
        lateral_scale: f64,
        roll: f64,
    ) -> Self {
        let omega = (G / z_com).sqrt();
        // Pass 1: where the pressure sits during each SINGLE support phase,
        // and the resting point (mid-stance) for double support at the ends.
        let target = |i: usize| -> Option<(na::Vector2<f64>, na::Vector2<f64>)> {
            let st = fp.at_slice(i);
            match plan.slices[i].support {
                Support::Single { stance, .. } => {
                    let mid = st.mid_xy();
                    let p = mid + (st.xy(stance) - mid) * lateral_scale;
                    let u = na::Vector2::new(st.yaw[stance].cos(), st.yaw[stance].sin());
                    Some((p - u * roll, p + u * roll))
                }
                // flight: no pressure point; the short flight is ramped
                // like a double support (a virtual ZMP between the soles)
                Support::Double | Support::Flight { .. } => None,
            }
        };
        let n = plan.slices.len();
        let mut segs: Vec<ZmpSeg> = Vec::with_capacity(n);
        for (i, s) in plan.slices.iter().enumerate() {
            let (p0, p1) = match target(i) {
                Some(p) => p,
                None => {
                    // Start where the previous segment ENDED, not at the last
                    // single-support point. Between two steps the two are the
                    // same thing, so mid-gait behaviour is unchanged; at the
                    // END of the plan they are not, and searching backward for
                    // a single support makes every trailing double-support
                    // slice re-ramp from the last stance sole. With more than
                    // one trailing slice that means ramping to mid-stance over
                    // and over, each time across the full slice -- which is
                    // how a 25 s tail turned into eleven seconds of standing
                    // on one foot.
                    let prev = segs
                        .last()
                        .map(|s: &ZmpSeg| s.p1)
                        .unwrap_or_else(|| fp.at_slice(i).mid_xy());
                    let next_single = || {
                        (i + 1..n)
                            .find_map(target)
                            .map(|(entry, _)| entry)
                            .unwrap_or_else(|| fp.at_slice(i).mid_xy())
                    };
                    // `GaitPlan::insert_ds_settles` replaces an inter-step
                    // DS with three consecutive DS slices. Route pressure to
                    // mid-feet on the first, keep it there on the middle, and
                    // route it to the new stance on the last. Other DS runs
                    // retain the original direct-to-next-stance behaviour.
                    let run_start = (0..i)
                        .rev()
                        .find(|&k| !matches!(plan.slices[k].support, Support::Double))
                        .map_or(0, |k| k + 1);
                    let run_end = (i + 1..n)
                        .find(|&k| !matches!(plan.slices[k].support, Support::Double))
                        .map_or(n - 1, |k| k - 1);
                    let next = if run_end + 1 - run_start >= 3 && i < run_end {
                        fp.at_slice(i).mid_xy()
                    } else {
                        next_single()
                    };
                    (prev, next)
                }
            };
            segs.push(ZmpSeg {
                p0,
                p1,
                t0: s.t0,
                duration: s.duration(),
            });
        }

        // Pass 2: backward recursion for the DCM.
        //
        // For a linear ZMP p(t) = p0 + v t on [0, T], the solution of
        // xi_dot = omega (xi - p) is
        //     xi(t) = p(t) + v/omega + (xi_0 - p0 - v/omega) e^{omega t}
        // so, going backwards from the segment's exit value,
        //     xi_0 = p0 + v/omega + (xi_T - p1 - v/omega) e^{-omega T}
        //
        // The terminal condition is rest at the final ZMP: xi = p, c_dot = 0.
        // Anything else asks the robot to still be moving when the plan runs
        // out, and the exponential means that error is the LARGEST one at the
        // start of the plan, not the smallest.
        let mut plan = DcmPlan {
            omega,
            segs,
            xi_eos: Vec::new(),
        };
        plan.recompute_eos();
        plan
    }

    /// Pass 2 as a method, so a plan whose segments were edited after
    /// construction can be made consistent again.
    pub fn recompute_eos(&mut self) {
        let omega = self.omega;
        let segs = &self.segs;
        let mut xi_eos = vec![na::Vector2::zeros(); segs.len()];
        let mut xi_end = segs.last().map(|s| s.p1).unwrap_or_else(na::Vector2::zeros);
        for i in (0..segs.len()).rev() {
            let s = &segs[i];
            let v_over_w = s.slope() / omega;
            xi_eos[i] = xi_end;
            xi_end = s.p0 + v_over_w + (xi_end - s.p1 - v_over_w) * (-omega * s.duration).exp();
        }
        self.xi_eos = xi_eos;
    }

    /// Replace segment `i`'s ZMP with the constant point `p` and make the plan
    /// consistent again. Used after a landing cut: the double support that
    /// follows is re-sized from the measured DCM (`step_timing` with a chosen
    /// pivot), and the plan must then actually PLACE its ZMP at that pivot
    /// for that long, or the reference it produces still starts at the rest
    /// point and the controller is asked for the same infeasible jump.
    /// Measured 2026-09-24 on G1: sizing the DS alone rescued 4 of 59
    /// fallers; the reference stayed discontinuous because the plan did not
    /// know about the pivot.
    pub fn set_segment_zmp(&mut self, i: usize, p: na::Vector2<f64>) {
        if let Some(s) = self.segs.get_mut(i) {
            s.p0 = p;
            s.p1 = p;
            self.recompute_eos();
        }
    }

    /// Like [`set_segment_zmp`] but the ZMP moves linearly from `p0` to `p1`
    /// over the segment -- the pressure can only be where the LOAD is, and at
    /// a landing cut the load is still on the old stance foot.
    pub fn set_segment_zmp_ramp(&mut self, i: usize, p0: na::Vector2<f64>, p1: na::Vector2<f64>) {
        if let Some(s) = self.segs.get_mut(i) {
            s.p0 = p0;
            s.p1 = p1;
            self.recompute_eos();
        }
    }

    pub fn index_at(&self, t: f64) -> usize {
        match self.segs.iter().position(|s| t < s.t0 + s.duration) {
            Some(i) => i,
            None => self.segs.len() - 1,
        }
    }

    /// Reference DCM, its rate, and the planned ZMP, at time `t`.
    ///
    /// Evaluated BACKWARD from the segment's end:
    ///
    /// ```text
    /// xi(t) = p(t) + v/omega + (xi_eos - p1 - v/omega) exp(-omega (T - t))
    /// ```
    ///
    /// which is the same solution as the forward form, but with a DECAYING
    /// exponential. That is not a nicety. The forward form,
    /// `xi_ini + ... exp(+omega t)`, pairs a coefficient that the backward
    /// recursion already shrank by `exp(-omega T)` with a factor that grows by
    /// `exp(+omega T)`, so the coefficient is a catastrophic cancellation and
    /// the exponential amplifies what survives.
    ///
    /// Measured, on a 10 s final double-support segment at omega = 5.585:
    /// `exp(omega T) = 1.8e24`, the cancelled coefficient carries ~9e-18 of
    /// round-off instead of its true ~1e-26, and the reference DCM came out
    /// at -0.39 m -- ten times the stance width, for a plan whose ZMP never
    /// leaves +-0.042 m. The robot tracked it, faithfully, into the floor.
    pub fn reference(&self, t: f64) -> DcmRef {
        let i = self.index_at(t);
        let s = &self.segs[i];
        let dt = (t - s.t0).clamp(0.0, s.duration);
        let v_over_w = s.slope() / self.omega;
        let p = s.at(t);
        let decay = (-self.omega * (s.duration - dt)).exp();
        let xi = p + v_over_w + (self.xi_eos[i] - s.p1 - v_over_w) * decay;
        DcmRef {
            xi,
            xi_dot: (xi - p) * self.omega,
            p,
        }
    }
    /// A receding, finite-horizon DCM reference. The terminal DCM is set to
    /// the planned ZMP at `t + horizon_s`; future footsteps beyond that point
    /// cannot change the current reference. Evaluating backward over partial
    /// segments keeps the reference continuous as the horizon slides.
    /// A nonpositive horizon reproduces the full-plan reference exactly.
    pub fn reference_preview(&self, t: f64, horizon_s: f64) -> DcmRef {
        if !horizon_s.is_finite() || horizon_s <= 0.0 {
            return self.reference(t);
        }
        let last = self.segs.last().expect("DCM plan has segments");
        let plan_end = last.t0 + last.duration;
        let start = t.clamp(self.segs[0].t0, plan_end);
        let end = (start + horizon_s).min(plan_end);
        let first_i = self.index_at(start);
        let last_i = self.index_at(end);
        let mut xi = self.segs[last_i].at(end);
        for i in (first_i..=last_i).rev() {
            let seg = &self.segs[i];
            let a = if i == first_i { start } else { seg.t0 };
            let b = if i == last_i {
                end
            } else {
                seg.t0 + seg.duration
            };
            if b <= a {
                continue;
            }
            let v_over_w = seg.slope() / self.omega;
            xi = seg.at(a) + v_over_w + (xi - seg.at(b) - v_over_w) * (-self.omega * (b - a)).exp();
        }
        let p = self.segs[first_i].at(start);
        // A moving terminal contributes to the derivative of the receding
        // reference. Without this term xi_dot would describe the frozen
        // terminal LIPM flow, not d(reference_preview)/dt.
        let terminal_motion = if end < plan_end - 1e-12 {
            self.segs[last_i].slope() * (-self.omega * (end - start)).exp()
        } else {
            na::Vector2::zeros()
        };
        DcmRef {
            xi,
            xi_dot: (xi - p) * self.omega + terminal_motion,
            p,
        }
    }
}

impl DcmPlan {
    /// The plan's DCM at the END of the segment covering `t`.
    pub fn eos_at(&self, t: f64) -> na::Vector2<f64> {
        self.xi_eos[self.index_at(t)]
    }

    /// Where the MEASURED DCM will be at the end of the current segment, if
    /// the ZMP follows the plan from here.
    ///
    /// This is the quantity footstep adaptation steers on: the next foot
    /// should land under where the DCM is actually going, not under where the
    /// plan assumed it would go.
    ///
    /// The `exp(+omega dt)` here is honest -- it is a forward PREDICTION over
    /// a shrinking horizon, not an evaluation of a stored coefficient
    /// (contrast [`DcmPlan::reference`]). It is noisy at the start of a step,
    /// where dt is a full single support and the factor is ~7, and exact at
    /// touchdown, where dt is zero. Applying the correction continuously is
    /// what makes that acceptable: it converges as the step runs out.
    pub fn predict_eos(&self, t: f64, xi: &na::Vector2<f64>) -> na::Vector2<f64> {
        let i = self.index_at(t);
        let s = &self.segs[i];
        let t_end = s.t0 + s.duration;
        let dt_left = (t_end - t).max(0.0);
        let v_over_w = s.slope() / self.omega;
        s.at(t_end) + v_over_w + (xi - s.at(t) - v_over_w) * (self.omega * dt_left).exp()
    }
}

/// Measured DCM from the measured CoM state.
pub fn dcm_of(com: &na::Vector3<f64>, com_vel: &na::Vector3<f64>, omega: f64) -> na::Vector2<f64> {
    na::Vector2::new(com.x + com_vel.x / omega, com.y + com_vel.y / omega)
}

/// The DCM tracking law, returning the ZMP the controller wants.
///
/// Solving `omega (xi - p) = xi_dot_ref - k (xi - xi_ref)` for `p` gives
///
/// ```text
/// p_cmd = xi - xi_dot_ref/omega + (k/omega) (xi - xi_ref)
/// ```
///
/// so `k_dcm = k/omega` is dimensionless and the error decays as
/// `exp(-k_dcm * omega * t)`. `k_dcm = 2` is one DCM time constant of settling
/// per half of `1/omega`, which is the usual starting point.
pub fn commanded_zmp(
    xi_meas: &na::Vector2<f64>,
    r: &DcmRef,
    omega: f64,
    k_dcm: f64,
) -> na::Vector2<f64> {
    xi_meas - r.xi_dot / omega + (xi_meas - r.xi) * k_dcm
}

/// Commanded CoM acceleration in the horizontal plane, `omega^2 (c - p)`.
pub fn com_accel_xy(
    com: &na::Vector3<f64>,
    p_cmd: &na::Vector2<f64>,
    omega: f64,
) -> na::Vector2<f64> {
    na::Vector2::new(com.x - p_cmd.x, com.y - p_cmd.y) * (omega * omega)
}

/// [`com_accel_xy`] without the constant-height assumption.
///
/// The exact centroidal relation for a single contact is
/// `(c_xy - p) * (g + c_z_ddot) / (c_z - p_z)`; the LIPM drops `c_z_ddot`
/// and fixes the height, which is right for level walking and wrong exactly
/// when the CoM is being driven up or down a stair. Descending, `c_z_ddot`
/// is NEGATIVE -- a 52 mm drop inside one step peaks near -2 m/s^2, so the
/// true horizontal stiffness is ~20% BELOW what `omega^2` claims and every
/// commanded ZMP overshoots. Ascending the sign flips and the error is
/// conservative, which is why level and upward gaits survived it.
///
/// `zdd_ref` is the planned vertical CoM acceleration and `surface_z` the
/// height of the contact the pressure acts on; passing zeros reproduces the
/// classic form.
pub fn com_accel_xy_vh(
    com: &na::Vector3<f64>,
    p_cmd: &na::Vector2<f64>,
    omega: f64,
    zdd_ref: f64,
    surface_z: f64,
) -> na::Vector2<f64> {
    let dxy = na::Vector2::new(com.x - p_cmd.x, com.y - p_cmd.y);
    let h = com.z - surface_z;
    if h <= 0.05 || !h.is_finite() || !zdd_ref.is_finite() {
        return dxy * (omega * omega);
    }
    // Keep the effective stiffness positive and bounded: a free-fall
    // (zdd = -g) leaves the contact unable to steer at all, and anything
    // past that is not a contact any more.
    let k = ((G + zdd_ref) / h).clamp(0.1 * omega * omega, 4.0 * omega * omega);
    dxy * k
}

/// Clamp the commanded ZMP into the current support polygon.
///
/// The QP's `patch_contact` already refuses to produce a CoP outside the
/// sole, so an out-of-box command is not dangerous -- it is DISHONEST. The
/// solver saturates, the CoM tracks something other than what was asked, and
/// the logs show a healthy `Optimal` status the whole way down. Clamping here
/// makes the saturation a number the summary can report instead of a silent
/// difference between the reference and the plant.
///
/// The polygon is approximated by the axis-aligned hull of the stance soles'
/// CoP boxes, which is exact while the feet are parallel and level -- the
/// only case stepping in place produces.
pub struct SupportBox {
    pub lo: na::Vector2<f64>,
    pub hi: na::Vector2<f64>,
}

impl SupportBox {
    pub fn from_stance(
        steps: &Footsteps,
        support: Support,
        cop_half: (f64, f64),
        margin: f64,
    ) -> Self {
        let mut lo = na::Vector2::new(f64::INFINITY, f64::INFINITY);
        let mut hi = na::Vector2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        let half = na::Vector2::new(cop_half.0 * margin, cop_half.1 * margin);
        for side in 0..2 {
            if !support.is_stance(side) {
                continue;
            }
            let c = steps.xy(side);
            lo = na::Vector2::new(lo.x.min(c.x - half.x), lo.y.min(c.y - half.y));
            hi = na::Vector2::new(hi.x.max(c.x + half.x), hi.y.max(c.y + half.y));
        }
        SupportBox { lo, hi }
    }

    /// The support polygon with the soles ROTATED to their planned yaw.
    ///
    /// The axis-aligned version describes a footprint the robot does not have
    /// the moment it turns: a 98 x 38 mm sole yawed by 30 degrees has an
    /// axis-aligned hull 34 mm wider than the sole in y and 30 mm shorter in
    /// x, so the clamp permits pressure the foot cannot deliver and forbids
    /// pressure it can. Take the hull of the rotated corners instead. Still a
    /// box, so the clamp stays a clamp -- but a box around the right shape.
    pub fn from_stance_yawed(
        steps: &Footsteps,
        support: Support,
        cop_half: (f64, f64),
        margin: f64,
    ) -> Self {
        let mut lo = na::Vector2::new(f64::INFINITY, f64::INFINITY);
        let mut hi = na::Vector2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        let (hx, hy) = (cop_half.0 * margin, cop_half.1 * margin);
        for side in 0..2 {
            if !support.is_stance(side) {
                continue;
            }
            let c = steps.xy(side);
            let (cs, sn) = (steps.yaw[side].cos(), steps.yaw[side].sin());
            for (sx, sy) in [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
                let (dx, dy) = (sx * hx, sy * hy);
                let p = na::Vector2::new(c.x + cs * dx - sn * dy, c.y + sn * dx + cs * dy);
                lo = na::Vector2::new(lo.x.min(p.x), lo.y.min(p.y));
                hi = na::Vector2::new(hi.x.max(p.x), hi.y.max(p.y));
            }
        }
        SupportBox { lo, hi }
    }

    /// Returns the clamped point and how far it had to move (0 = inside).
    pub fn clamp(&self, p: &na::Vector2<f64>) -> (na::Vector2<f64>, f64) {
        // No stance foot (flight): there is no polygon to clamp into.
        if !(self.lo.x <= self.hi.x && self.lo.y <= self.hi.y) {
            return (*p, 0.0);
        }
        let c = na::Vector2::new(
            p.x.clamp(self.lo.x, self.hi.x),
            p.y.clamp(self.lo.y, self.hi.y),
        );
        (c, (c - p).norm())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gait::{GaitParams, GaitPlan, LEFT, RIGHT};

    fn steps() -> Footsteps {
        Footsteps {
            sole: [
                na::Vector3::new(0.0, 0.0706, 0.0),
                na::Vector3::new(0.0, -0.0706, 0.0),
            ],
            yaw: [0.0, 0.0],
        }
    }

    #[test]
    fn dcm_reference_is_continuous_across_segment_boundaries() {
        let p = GaitParams {
            t_start: 1.0,
            t_ds: 0.2,
            t_ss: 0.4,
            n_steps: 4,
            first_swing: RIGHT,
            t_end: 6.0,
            t_flight: 0.0,
        };
        let plan = DcmPlan::from_schedule(&GaitPlan::new(&p), &steps(), 0.315, 1.0);
        for s in plan.segs.iter().skip(1) {
            let before = plan.reference(s.t0 - 1e-7).xi;
            let after = plan.reference(s.t0 + 1e-7).xi;
            assert!(
                (before - after).norm() < 1e-5,
                "jump at t={}: {before:?} vs {after:?}",
                s.t0
            );
        }
    }

    #[test]
    fn finite_preview_ignores_distant_terminal_steps_and_stays_continuous() {
        let make = |n_steps| {
            let p = GaitParams {
                t_start: 1.0,
                t_ds: 0.2,
                t_ss: 0.4,
                n_steps,
                first_swing: RIGHT,
                t_end: 30.0,
                t_flight: 0.0,
            };
            let gait = GaitPlan::new(&p);
            let fp = FootstepPlan::alternating(&gait, &steps(), 0.03, 0);
            DcmPlan::from_footsteps(&gait, &fp, 0.315, 1.0)
        };
        let short = make(12);
        let long = make(20);
        for t in [1.5, 2.0, 3.0, 5.0] {
            let a = short.reference_preview(t, 1.5);
            let b = long.reference_preview(t, 1.5);
            assert!((a.xi - b.xi).norm() < 1e-12, "preview differs at {t}");
            assert!((a.p - b.p).norm() < 1e-12);
            assert!((short.reference_preview(t, 0.0).xi - short.reference(t).xi).norm() < 1e-15);
        }
        let h = 1e-6;
        for t in [1.25, 2.15, 3.05, 4.35] {
            let rate = (short.reference_preview(t + h, 1.5).xi
                - short.reference_preview(t - h, 1.5).xi)
                / (2.0 * h);
            let reported = short.reference_preview(t, 1.5).xi_dot;
            assert!((rate - reported).norm() < 1e-5, "rate mismatch at {t}");
        }
        for seg in short.segs.iter().skip(1).take(15) {
            let before = short.reference_preview(seg.t0 - 1e-7, 1.5).xi;
            let after = short.reference_preview(seg.t0 + 1e-7, 1.5).xi;
            assert!((before - after).norm() < 1e-5, "jump at {}", seg.t0);
        }
    }

    #[test]
    fn closed_form_satisfies_the_dcm_ode_pointwise() {
        // `xi_dot = omega (xi - p)` is the whole model, so check it as a
        // residual against a central difference of the closed form.
        //
        // NOT by forward-integrating: the DCM is the DIVERGENT component, the
        // eigenvalue is +omega, and at omega = 5.6 rad/s a 3 s march
        // multiplies any truncation error by e^16.7 ~ 2e7. An integration
        // test here fails on a correct plan, which is exactly what the first
        // version of this test did.
        let p = GaitParams {
            t_start: 1.0,
            t_ds: 0.2,
            t_ss: 0.4,
            n_steps: 3,
            first_swing: RIGHT,
            t_end: 4.0,
            t_flight: 0.0,
        };
        let plan = DcmPlan::from_schedule(&GaitPlan::new(&p), &steps(), 0.315, 1.0);
        let h = 1e-6;
        let mut worst: f64 = 0.0;
        let mut t = 0.05;
        while t < 3.5 {
            // Skip segment boundaries, where the central difference straddles
            // a slope change in p and is not a derivative of anything.
            let near_edge = plan
                .segs
                .iter()
                .any(|s| (t - s.t0).abs() < 1e-3 || (t - (s.t0 + s.duration)).abs() < 1e-3);
            if !near_edge {
                let fd = (plan.reference(t + h).xi - plan.reference(t - h).xi) / (2.0 * h);
                let r = plan.reference(t);
                worst = worst.max((fd - (r.xi - r.p) * plan.omega).norm());
                // The published xi_dot must agree with the same derivative.
                worst = worst.max((fd - r.xi_dot).norm());
            }
            t += 1e-3;
        }
        assert!(worst < 1e-5, "worst DCM ODE residual {worst:.3e}");
    }

    #[test]
    fn plan_ends_at_rest_over_the_final_support() {
        let p = GaitParams {
            t_start: 1.0,
            t_ds: 0.2,
            t_ss: 0.4,
            n_steps: 2,
            first_swing: RIGHT,
            t_end: 5.0,
            t_flight: 0.0,
        };
        let plan = DcmPlan::from_schedule(&GaitPlan::new(&p), &steps(), 0.315, 1.0);
        let end = plan.reference(5.0);
        // At rest means xi == p, i.e. c_dot == 0.
        assert!((end.xi - end.p).norm() < 1e-6);
        // ...and the final ZMP is mid-stance, not over one foot.
        assert!((end.p - steps().mid_xy()).norm() < 1e-9);
    }

    #[test]
    fn single_support_zmp_sits_on_the_stance_sole() {
        let p = GaitParams {
            t_start: 1.0,
            t_ds: 0.2,
            t_ss: 0.4,
            n_steps: 2,
            first_swing: RIGHT,
            t_end: 5.0,
            t_flight: 0.0,
        };
        let gait = GaitPlan::new(&p);
        let plan = DcmPlan::from_schedule(&gait, &steps(), 0.315, 1.0);
        // First SS: right swings, so the left foot carries the pressure.
        let mid_ss = 1.0 + 0.2;
        assert!(matches!(
            gait.support_at(mid_ss),
            Support::Single { stance: LEFT, .. }
        ));
        assert!((plan.reference(mid_ss).p - steps().xy(LEFT)).norm() < 1e-12);
    }

    #[test]
    fn inserted_settle_routes_zmp_via_mid_feet() {
        let p = GaitParams {
            t_start: 1.0,
            t_ds: 0.2,
            t_ss: 0.4,
            n_steps: 3,
            first_swing: RIGHT,
            t_end: 6.0,
            t_flight: 0.0,
        };
        let mut gait = GaitPlan::new(&p);
        assert_eq!(gait.insert_ds_settles(0.15, 1), 2);
        let fp = FootstepPlan::alternating(&gait, &steps(), 0.03, 0);
        let plan = DcmPlan::from_footsteps(&gait, &fp, 0.315, 1.0);

        // First inter-step DS run: transfer to the current mid-feet point,
        // hold exactly there, then transfer to the next stance foot.
        let first = 2;
        let hold = 3;
        let last = 4;
        let next_ss = 5;
        let mid = fp.at_slice(hold).mid_xy();
        let next_stance = match gait.slices[next_ss].support {
            Support::Single { stance, .. } => fp.at_slice(next_ss).xy(stance),
            Support::Double | Support::Flight { .. } => panic!("expected single support after settle"),
        };
        assert!((plan.segs[first].p1 - mid).norm() < 1e-12);
        assert!((plan.segs[hold].p0 - mid).norm() < 1e-12);
        assert!((plan.segs[hold].p1 - mid).norm() < 1e-12);
        assert!((plan.segs[last].p0 - mid).norm() < 1e-12);
        assert!((plan.segs[last].p1 - next_stance).norm() < 1e-12);
    }

    #[test]
    fn a_long_trailing_segment_does_not_blow_the_reference_up() {
        // The regression that cost a 22 s run. `t_end` far past the last step
        // leaves one double-support segment ten seconds long; at
        // omega = 5.585 that is exp(55.9) = 1.8e24 of forward amplification
        // on a coefficient that is pure cancellation. Bound the reference by
        // something physical: the DCM of a plan whose ZMP stays inside the
        // stance can never leave the stance by more than a few centimetres.
        let p = GaitParams {
            t_start: 2.0,
            t_ds: 0.2,
            t_ss: 0.45,
            n_steps: 20,
            first_swing: RIGHT,
            t_end: 25.0,
            t_flight: 0.0,
        };
        let plan = DcmPlan::from_schedule(&GaitPlan::new(&p), &steps(), 0.315, 1.0);
        let mut t = 0.0;
        while t <= 25.0 {
            let r = plan.reference(t);
            assert!(
                r.xi.norm() < 0.15,
                "reference DCM {:?} at t={t:.2} is outside anything the plan can produce",
                r.xi
            );
            t += 0.005;
        }
        // ...and it must still END at rest over the final ZMP.
        let end = plan.reference(25.0);
        assert!((end.xi - end.p).norm() < 1e-9);
    }

    #[test]
    fn support_box_shrinks_to_one_foot_in_single_support() {
        let s = steps();
        let ds = SupportBox::from_stance(&s, Support::Double, (0.049, 0.019), 1.0);
        let ss = SupportBox::from_stance(
            &s,
            Support::Single {
                stance: LEFT,
                swing: RIGHT,
            },
            (0.049, 0.019),
            1.0,
        );
        assert!((ds.hi.y - ds.lo.y) > (ss.hi.y - ss.lo.y));
        // A command over the right foot is outside the left-foot-only box.
        let (c, moved) = ss.clamp(&s.xy(RIGHT));
        assert!(moved > 0.0);
        assert!(c.y >= ss.lo.y - 1e-12);
    }
}

/// Step-timing adjustment: how long the current single support should still
/// last so that the DCM arrives where the plan wanted it.
///
/// # Why timing and not placement
///
/// During single support the ZMP is parked at the stance sole, so the DCM
/// obeys `xi(t) = p + (xi_0 - p) exp(omega t)` in closed form. Two knobs can
/// steer where it ends up at touchdown: WHERE the next foot goes, and WHEN it
/// gets there. Placement is [`crate::gait::FootstepPlan`]'s adaptation
/// (`ADAPT_STEP`), and on this machine it is measured harmful -- 26 grid
/// points improved against 83 worsened over 672 runs (doc Sec.19.6). Timing
/// has never been tried: `t_ss` is a constant, so a disturbed machine walks
/// the same schedule as an undisturbed one and commits to the next stance
/// foot at a time chosen before the push happened.
///
/// # The solve
///
/// Fix the footstep, target the plan's own end-of-segment DCM `xi_eos`, and
/// invert the closed form for the remaining time:
///
/// ```text
/// xi_eos = p + (xi_meas - p) exp(omega tau)
/// tau    = ln((xi_eos - p) / (xi_meas - p)) / omega
/// ```
///
/// This is a division and a log, not an optimisation -- there is no QP and no
/// MPC here. The catch is that it is one equation per axis and only one
/// unknown, so the two axes disagree; `axis` picks which one to honour.
/// Lateral (`1`) is the default because that is the axis this machine falls
/// on: the lateral CoP box is +-19 mm against +-49 mm fore-aft, and every
/// failure traced in Sec.19 was the stance CoP pinned to the lateral edge.
///
/// The outcome of [`step_timing`], which is not always a time.
///
/// The distinction matters and measuring it changed the reading of the whole
/// feature: on a run that falls, the solve reports [`Unreachable`] on 75 of 83
/// ticks and returns a time on only 8, while runs that survive ask for a time
/// once or not at all. So the timing adaptation, as first written, was silent
/// on exactly the disturbance class that kills the machine -- its measured
/// gain came from correcting steps that were merely late.
///
/// [`Unreachable`]: StepTiming::Unreachable
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepTiming {
    /// Remaining single-support time that lands the DCM on its target.
    Time(f64),
    /// The DCM is on the FAR SIDE of the ZMP from the target. During single
    /// support the DCM only ever moves away from the ZMP, so no positive time
    /// reaches the target: this step cannot be rescued by waiting. It can only
    /// be ENDED -- which is what a reduced-step reflex does with this.
    Unreachable,
    /// One of the two gaps is too small to divide by; the ratio would be noise.
    Degenerate,
}

/// Solve for the remaining single-support duration. See [`StepTiming`] for
/// what the non-numeric outcomes mean; they are reported rather than folded
/// into a `None` because "no answer" and "the answer is: stop waiting" call
/// for opposite actions.
pub fn step_timing(
    xi_meas: &na::Vector2<f64>,
    xi_target: &na::Vector2<f64>,
    p: &na::Vector2<f64>,
    omega: f64,
    axis: usize,
) -> StepTiming {
    const MIN_GAP: f64 = 1e-4; // 0.1 mm; below this the ratio is noise
    let now = xi_meas[axis] - p[axis];
    let want = xi_target[axis] - p[axis];
    if now.abs() < MIN_GAP || want.abs() < MIN_GAP {
        return StepTiming::Degenerate;
    }
    let ratio = want / now;
    if ratio <= 0.0 {
        return StepTiming::Unreachable;
    }
    StepTiming::Time(ratio.ln() / omega)
}

#[cfg(test)]
mod timing_tests {
    use super::*;

    #[test]
    fn timing_recovers_the_undisturbed_duration() {
        // Roll the closed form forward by a known tau, then ask for it back.
        let omega = 5.585;
        let p = na::Vector2::new(0.0, -0.05);
        let xi0 = na::Vector2::new(0.0, -0.02);
        let tau: f64 = 0.23;
        let xi_end = p + (xi0 - p) * (omega * tau).exp() as f64;
        let StepTiming::Time(got) = step_timing(&xi0, &xi_end, &p, omega, 1) else {
            panic!("expected a solvable time");
        };
        assert!((got - tau).abs() < 1e-9, "got {got}, want {tau}");
    }

    #[test]
    fn a_dcm_that_ran_ahead_shortens_the_step() {
        // Pushed toward the target: less time left than nominal.
        let omega = 5.585;
        let p = na::Vector2::new(0.0, -0.05);
        let nominal = na::Vector2::new(0.0, -0.02);
        let target = p + (nominal - p) * (omega * 0.30_f64).exp();
        let ahead = na::Vector2::new(0.0, -0.005); // further from the ZMP
        let StepTiming::Time(got) = step_timing(&ahead, &target, &p, omega, 1) else {
            panic!("expected a solvable time");
        };
        assert!(got < 0.30, "expected a shorter step, got {got}");
    }

    #[test]
    fn wrong_side_of_the_zmp_is_unsolvable() {
        let omega = 5.585;
        let p = na::Vector2::new(0.0, 0.0);
        let xi = na::Vector2::new(0.0, 0.03);
        let target = na::Vector2::new(0.0, -0.03);
        assert_eq!(
            step_timing(&xi, &target, &p, omega, 1),
            StepTiming::Unreachable
        );
    }
}
