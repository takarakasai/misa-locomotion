//! Support schedule and swing-foot trajectory for a biped.
//!
//! The schedule is a plain list of time slices, built once, because for
//! stepping in place it IS known in advance and a plan the controller can
//! look ahead into is what the DCM reference needs ([`super::dcm`]). A
//! contact-driven correction belongs on top of this, not instead of it: the
//! nominal schedule is still what the reference was planned against.

use nalgebra as na;

/// 0 = left, 1 = right, matching `BipedRig::foot_mi` (articara).
pub type Side = usize;
pub const LEFT: Side = 0;
pub const RIGHT: Side = 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Support {
    /// Both feet down.
    Double,
    /// One foot down. `stance` carries the load, `swing` is in the air.
    Single { stance: Side, swing: Side },
}

impl Support {
    pub fn swing(self) -> Option<Side> {
        match self {
            Support::Double => None,
            Support::Single { swing, .. } => Some(swing),
        }
    }
    pub fn is_stance(self, side: Side) -> bool {
        match self {
            Support::Double => true,
            Support::Single { stance, .. } => stance == side,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Slice {
    pub support: Support,
    pub t0: f64,
    pub t1: f64,
}

impl Slice {
    pub fn duration(&self) -> f64 {
        self.t1 - self.t0
    }
    /// Position within the slice, 0 at entry and 1 at exit.
    pub fn frac(&self, t: f64) -> f64 {
        let d = self.duration();
        if d <= 1e-12 {
            0.0
        } else {
            ((t - self.t0) / d).clamp(0.0, 1.0)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GaitParams {
    /// Double support before the first step. This is the weight shift, and it
    /// is deliberately longer than the inter-step double support: the CoM has
    /// to travel from the middle of the stance to over one foot, from rest.
    pub t_start: f64,
    /// Double support between steps.
    pub t_ds: f64,
    /// Single support per step.
    pub t_ss: f64,
    /// How many steps to take. The plan ends with a double-support segment
    /// that runs to `t_end`, so the robot comes back to rest centred.
    pub n_steps: usize,
    /// Which foot swings first.
    pub first_swing: Side,
    pub t_end: f64,
}

impl Default for GaitParams {
    fn default() -> Self {
        GaitParams {
            t_start: 2.0,
            t_ds: 0.20,
            t_ss: 0.45,
            n_steps: 20,
            first_swing: RIGHT,
            t_end: 25.0,
        }
    }
}

pub struct GaitPlan {
    pub slices: Vec<Slice>,
}

/// Contact confirmation around touchdown. Remember the active swing across
/// its end boundary and capture the extension budget once per step.
pub struct ContactPhase {
    pub min_frac: f64,
    pub load_threshold: f64,
    pub confirm_ticks: u32,
    pub extend_frac: f64,
    active: Option<(usize, f64, f64)>, // slice, original end, duration
    loaded_ticks: u32,
}

impl ContactPhase {
    /// The swing's ORIGINAL clock for `slice`, if this phase is holding it
    /// open: `(t0, duration)` as scheduled before any waiting. A late-contact
    /// extension stretches the slice, and a swing fraction recomputed from
    /// the stretched duration REWINDS -- measured on the one-step descent,
    /// the foot 5 mm above the lower tread was pulled 50 mm back up to the
    /// hold-phase height and the run tipped. The trajectory must keep its
    /// own clock and simply HOLD at its endpoint while the schedule waits.
    pub fn frozen_clock(&self, slice: usize) -> Option<(f64, f64)> {
        self.active
            .filter(|&(i, _, _)| i == slice)
            .map(|(_, end, dur)| (end - dur, dur))
    }
}

impl ContactPhase {
    pub fn new(min_frac: f64, load_threshold: f64, confirm_ticks: u32, extend_frac: f64) -> Self {
        Self {
            min_frac,
            load_threshold,
            confirm_ticks: confirm_ticks.max(1),
            extend_frac: extend_frac.max(0.0),
            active: None,
            loaded_ticks: 0,
        }
    }

    /// Signed schedule shift: positive for early touchdown, negative for
    /// waiting. Call before looking up this tick's support.
    pub fn update(&mut self, gait: &mut GaitPlan, t: f64, dt: f64, fz: [f64; 2]) -> f64 {
        if self.active.is_none() {
            let i = gait.index_at(t);
            if gait.slices[i].support.swing().is_some() {
                let sl = gait.slices[i];
                self.active = Some((i, sl.t1, sl.duration()));
                self.loaded_ticks = 0;
            }
        }
        let Some((i, original_end, duration)) = self.active else {
            return 0.0;
        };
        let sl = gait.slices[i];
        let swing = sl
            .support
            .swing()
            .expect("active contact phase is single support");
        let eligible = t >= original_end - duration * (1.0 - self.min_frac);
        self.loaded_ticks = if eligible && fz[swing] > self.load_threshold {
            self.loaded_ticks + 1
        } else {
            0
        };
        if self.loaded_ticks >= self.confirm_ticks {
            self.active = None;
            return gait.retime(i, t);
        }
        let deadline = original_end + duration * self.extend_frac;
        if t >= deadline - 1e-9 {
            self.active = None;
            return 0.0;
        }
        // At t == t1 index_at(t) already points at double support. Use the
        // remembered swing and keep its interval open beyond this tick.
        if t >= sl.t1 - 1e-9 {
            return gait.retime(i, (t + dt).min(deadline));
        }
        0.0
    }
}

impl GaitPlan {
    pub fn new(p: &GaitParams) -> Self {
        let mut slices = Vec::with_capacity(2 * p.n_steps + 2);
        let mut t = 0.0;
        slices.push(Slice {
            support: Support::Double,
            t0: t,
            t1: t + p.t_start,
        });
        t += p.t_start;
        let mut swing = p.first_swing;
        for _ in 0..p.n_steps {
            let stance = 1 - swing;
            slices.push(Slice {
                support: Support::Single { stance, swing },
                t0: t,
                t1: t + p.t_ss,
            });
            t += p.t_ss;
            slices.push(Slice {
                support: Support::Double,
                t0: t,
                t1: t + p.t_ds,
            });
            t += p.t_ds;
            swing = 1 - swing;
        }
        // Run out to the end of the experiment on both feet, as a SEPARATE
        // slice rather than by stretching the last double support.
        //
        // Stretching it was a real bug. The ZMP plan ramps linearly across a
        // double-support slice from the last stance sole to mid-stance, so a
        // slice extended to `t_end` spreads "come to rest in the middle" over
        // the entire tail. Measured with a 25 s tail: the robot finished the
        // walk standing on ONE foot -- 64.3 N against 0.6 N -- and stayed
        // there for eleven seconds while its CoM crept back at 2 mm/s, until
        // the unloaded foot finally caught 18 N and knocked it over. The
        // settle has to have its own, bounded duration.
        if p.t_end > t {
            slices.push(Slice {
                support: Support::Double,
                t0: t,
                t1: p.t_end,
            });
        }
        GaitPlan { slices }
    }

    /// End the slice covering `at` at time `new_t1`, sliding every later
    /// slice by the same amount.
    ///
    /// This is how a contact-driven phase transition stays consistent with
    /// the DCM plan. The alternative -- keeping a nominal schedule and running
    /// a separate "actual" clock that jumps -- leaves the DCM reference
    /// describing a step that is no longer being taken. Retiming the PLAN and
    /// rebuilding the reference from it keeps one clock and one story.
    ///
    /// Rebuilding is cheap and, more importantly, nearly continuous: a
    /// segment's influence on the present DCM reference decays by
    /// `exp(-omega*T)` per step, about a factor of 20 at these timings, so
    /// moving a boundary by tens of milliseconds moves the current reference
    /// by microns.
    ///
    /// Returns the shift applied (positive = the step was cut short).
    pub fn retime(&mut self, at: usize, new_t1: f64) -> f64 {
        let old_t1 = self.slices[at].t1;
        let d = old_t1 - new_t1;
        if d.abs() < 1e-12 {
            return 0.0;
        }
        self.slices[at].t1 = new_t1;
        for s in self.slices.iter_mut().skip(at + 1) {
            s.t0 -= d;
            s.t1 -= d;
        }
        d
    }

    /// Insert one slice of `duration` right after slice `after`, sliding
    /// every later slice back by that much. Returns the new slice's index.
    /// Used by the re-step (doc/from_claude/20261012_restep_design.md).
    pub fn insert_after(&mut self, after: usize, support: Support, duration: f64) -> usize {
        let t0 = self.slices[after].t1;
        for s in self.slices.iter_mut().skip(after + 1) {
            s.t0 += duration;
            s.t1 += duration;
        }
        self.slices.insert(after + 1, Slice { support, t0, t1: t0 + duration });
        after + 1
    }

    /// Insert a separate constant-ZMP double-support hold between steps.
    ///
    /// The existing inter-step double-support slice transfers pressure from
    /// the previous stance foot to the next one. Merely making that slice
    /// longer also slows the transfer and changes the complete DCM plan. A
    /// second consecutive double-support slice starts where the transfer
    /// ended and has the same next target, so `DcmPlan` makes it a constant
    /// segment. This lets a controller wait for the measured DCM and contact
    /// load to settle without changing the transfer profile.
    pub fn insert_ds_settles(&mut self, duration: f64, first_after_step: usize) -> usize {
        assert!(duration.is_finite() && duration >= 0.0);
        if duration <= 0.0 || self.slices.len() < 3 {
            return 0;
        }
        let old = std::mem::take(&mut self.slices);
        let mut out = Vec::with_capacity(old.len() * 2);
        let mut t = old.first().map_or(0.0, |s| s.t0);
        let mut inserted = 0;
        let mut steps_seen = 0;
        for (i, sl) in old.iter().copied().enumerate() {
            if matches!(sl.support, Support::Single { .. }) {
                steps_seen += 1;
            }
            let d = sl.duration();
            let between_steps = matches!(sl.support, Support::Double)
                && i > 0
                && steps_seen >= first_after_step.max(1)
                && matches!(old[i - 1].support, Support::Single { .. })
                && old
                    .get(i + 1)
                    .is_some_and(|n| matches!(n.support, Support::Single { .. }));
            if between_steps {
                // Split the old direct stance-to-stance transfer in half.
                // DcmPlan interprets this three-DS run as old stance ->
                // mid-feet, constant mid-feet hold, mid-feet -> new stance.
                out.push(Slice {
                    support: Support::Double,
                    t0: t,
                    t1: t + 0.5 * d,
                });
                t += 0.5 * d;
                out.push(Slice {
                    support: Support::Double,
                    t0: t,
                    t1: t + duration,
                });
                t += duration;
                out.push(Slice {
                    support: Support::Double,
                    t0: t,
                    t1: t + 0.5 * d,
                });
                t += 0.5 * d;
                inserted += 1;
            } else {
                out.push(Slice {
                    support: sl.support,
                    t0: t,
                    t1: t + d,
                });
                t += d;
            }
        }
        self.slices = out;
        inserted
    }

    /// The slice covering `t`, clamped to the ends.
    pub fn index_at(&self, t: f64) -> usize {
        match self.slices.iter().position(|s| t < s.t1) {
            Some(i) => i,
            None => self.slices.len() - 1,
        }
    }

    pub fn at(&self, t: f64) -> (&Slice, f64) {
        let i = self.index_at(t);
        (&self.slices[i], self.slices[i].frac(t))
    }

    pub fn support_at(&self, t: f64) -> Support {
        self.slices[self.index_at(t)].support
    }

    /// Number of single-support phases entered by time `t` -- the step count
    /// a summary line should report.
    pub fn steps_taken(&self, t: f64) -> usize {
        self.slices
            .iter()
            .filter(|s| matches!(s.support, Support::Single { .. }) && s.t0 < t)
            .count()
    }
}

/// Where each foot is planted, in world coordinates, at the SOLE.
///
/// For stepping in place this is captured once from the settled pose and
/// never updated. That is the point: the balance reference must be built from
/// the PLAN, not from where the feet currently measure. Reading the stance
/// foot every tick is what put the plant inside its own reference and cost a
/// fall with zero degraded solves -- the foot rolled 4.6 mm and the CoM
/// target jumped 37 mm in two ticks.
#[derive(Clone, Copy, Debug)]
pub struct Footsteps {
    pub sole: [na::Vector3<f64>; 2],
    /// World yaw of each sole. Constant for straight walking; a turn rotates
    /// the feet with the body, and the swing task has to be told, or the foot
    /// lands square to the world while the body has turned under it.
    pub yaw: [f64; 2],
}

impl Footsteps {
    /// Sole centre of each foot in world coordinates, from an FK snapshot.
    pub fn from_fk(
        data: &misarta::data::Data<f64>,
        foot_mi: [usize; 2],
        sole_centre_x: f64,
        sole_below_origin: f64,
    ) -> Self {
        let mut sole = [na::Vector3::zeros(); 2];
        let mut yaw = [0.0; 2];
        for side in 0..2 {
            let o = misarta::se3::translation(&data.oMi[foot_mi[side]]);
            let r = misarta::se3::rotation_matrix(&data.oMi[foot_mi[side]]);
            sole[side] = o + r * na::Vector3::new(sole_centre_x, 0.0, -sole_below_origin);
            yaw[side] = r[(1, 0)].atan2(r[(0, 0)]);
        }
        Footsteps { sole, yaw }
    }

    pub fn xy(&self, side: Side) -> na::Vector2<f64> {
        na::Vector2::new(self.sole[side].x, self.sole[side].y)
    }

    pub fn mid_xy(&self) -> na::Vector2<f64> {
        (self.xy(LEFT) + self.xy(RIGHT)) * 0.5
    }
}

/// Where both feet are planted during each slice of the schedule.
///
/// Stepping in place needed one fixed pair; walking needs one pair PER SLICE,
/// and everything downstream reads it -- the ZMP plan, the swing foot's
/// touchdown target, the support polygon the commanded ZMP is clamped into,
/// the load split, and the z a touchdown anchor is projected onto. Threading
/// a sequence through instead of a constant is the whole difference between
/// the two gaits.
///
/// For a slice in single support the swing foot's entry is where it is GOING,
/// i.e. its target at the end of the slice. For a double-support slice both
/// entries are where the feet already are.
#[derive(Clone)]
pub struct FootstepPlan {
    pub at: Vec<Footsteps>,
    /// Where each foot started, kept so a stopped plan can be read back.
    pub initial: Footsteps,
}

impl FootstepPlan {
    /// Constant-stride walk. `stride` is the distance the BODY advances per
    /// step, so each foot moves 2x that when it swings -- the standard
    /// bookkeeping slip is to move the foot by one stride and wonder why the
    /// robot travels at half the commanded speed.
    ///
    /// `stride = 0` reproduces stepping in place exactly, which is what keeps
    /// the working gait a special case of this one rather than a separate
    /// path.
    pub fn constant_stride(plan: &GaitPlan, initial: &Footsteps, stride: f64) -> Self {
        let mut cur = *initial;
        let mut at = Vec::with_capacity(plan.slices.len());
        for s in &plan.slices {
            match s.support {
                Support::Single { swing, .. } => {
                    // The swing foot advances two strides from where it was,
                    // landing one stride ahead of the stance foot.
                    let mut next = cur;
                    next.sole[swing].x += 2.0 * stride;
                    at.push(next);
                    cur = next;
                }
                Support::Double => at.push(cur),
            }
        }
        FootstepPlan {
            at,
            initial: *initial,
        }
    }

    /// A walk from a velocity command: forward/backward, sideways, and turning.
    ///
    /// Every footstep is the nominal foot offset expressed in a BODY FRAME
    /// that advances by `(vx, vy) * T_step` and turns by `wz * T_step` each
    /// step. The foot swinging at step k lands where the body will be at step
    /// k+1, so the body moves one stride per step while the foot moves two --
    /// which is the alternating-lead pattern, arrived at by construction
    /// rather than by the half-first-step correction the x-only version
    /// needed. For `vy = wz = 0` it reproduces [`FootstepPlan::alternating`]
    /// exactly.
    ///
    /// Sideways and turning fall out of the same construction, and so does the
    /// non-crossing property: the two offsets are fixed in the body frame, so
    /// the feet can never swap sides however the frame moves. The lateral gap
    /// does breathe by one stride either side of nominal, so `|vy| * T_step`
    /// has to stay below the stance width.
    pub fn velocity(
        plan: &GaitPlan,
        initial: &Footsteps,
        vx: f64,
        vy: f64,
        wz: f64,
        t_step: f64,
        ramp: usize,
    ) -> Self {
        Self::velocity_with_change(plan, initial, vx, vy, wz, t_step, ramp, None)
    }

    /// [`Self::velocity`] with a commanded SPEED CHANGE: from step `at` on,
    /// the forward speed ramps linearly from `vx` to `vx2` over `ramp2`
    /// steps (`change = Some((at, vx2, ramp2))`). A benchmark for the
    /// controller's response to a command step, which the fixed-speed plan
    /// never exercises. `None` = [`Self::velocity`], bit for bit.
    #[allow(clippy::too_many_arguments)]
    pub fn velocity_with_change(
        plan: &GaitPlan,
        initial: &Footsteps,
        vx: f64,
        vy: f64,
        wz: f64,
        t_step: f64,
        ramp: usize,
        change: Option<(usize, f64, usize)>,
    ) -> Self {
        // The nominal offsets, read off the starting stance so a command of
        // zero reproduces exactly where the robot already is.
        let mid = initial.mid_xy();
        let yaw0 = 0.5 * (initial.yaw[LEFT] + initial.yaw[RIGHT]);
        let (c0, s0) = (yaw0.cos(), yaw0.sin());
        let off: [na::Vector2<f64>; 2] = std::array::from_fn(|k| {
            let d = initial.xy(k) - mid;
            // into the body frame
            na::Vector2::new(c0 * d.x + s0 * d.y, -s0 * d.x + c0 * d.y)
        });

        // Body pose after n steps, with the ramp applied step by step.
        let mut pose = Vec::with_capacity(plan.slices.len() + 2);
        let mut p = mid;
        let mut th = yaw0;
        pose.push((p, th));
        let n_steps = plan
            .slices
            .iter()
            .filter(|s| matches!(s.support, Support::Single { .. }))
            .count();
        for k in 0..=n_steps {
            let f = if ramp == 0 {
                1.0
            } else {
                ((k + 1) as f64 / ramp as f64).min(1.0)
            };
            let (c, s) = (th.cos(), th.sin());
            let vx_k = match change {
                Some((at, vx2, ramp2)) if k >= at => {
                    let g = if ramp2 == 0 { 1.0 } else { ((k - at + 1) as f64 / ramp2 as f64).min(1.0) };
                    vx + (vx2 - vx) * g
                }
                _ => vx,
            };
            p += na::Vector2::new(c * vx_k - s * vy, s * vx_k + c * vy) * (t_step * f);
            th += wz * t_step * f;
            pose.push((p, th));
        }
        let place = |k: usize, side: Side| -> (na::Vector3<f64>, f64) {
            let (p, th) = pose[k.min(pose.len() - 1)];
            let (c, s) = (th.cos(), th.sin());
            let o = off[side];
            (
                na::Vector3::new(
                    p.x + c * o.x - s * o.y,
                    p.y + s * o.x + c * o.y,
                    initial.sole[side].z,
                ),
                th,
            )
        };

        let mut cur = *initial;
        let mut at = Vec::with_capacity(plan.slices.len());
        let mut step = 0usize;
        for sl in &plan.slices {
            match sl.support {
                Support::Single { swing, .. } => {
                    // Close the feet on the last step so the plan ends square
                    // -- land at the body pose the OTHER foot is already at,
                    // not one stride beyond it. Without this the plan finishes
                    // mid-stride and the settle has to absorb the difference.
                    let target = if step + 1 == n_steps { step } else { step + 1 };
                    let (pos, th) = place(target, swing);
                    cur.sole[swing] = pos;
                    cur.yaw[swing] = th;
                    at.push(cur);
                    step += 1;
                }
                Support::Double => at.push(cur),
            }
        }
        FootstepPlan {
            at,
            initial: *initial,
        }
    }

    /// Alternating-lead walk: the feet stay one stride apart and take turns
    /// leading, instead of one foot stepping out and the other closing up.
    ///
    /// The naive version -- every swing advances 2x stride from where it was
    /// -- looks right and is not. From feet together it produces
    ///
    /// ```text
    ///   after step 1 (R swings)   L 0.000  R 0.100   R leads
    ///   after step 2 (L swings)   L 0.100  R 0.100   TOGETHER
    ///   after step 3 (R swings)   L 0.100  R 0.200   R leads
    /// ```
    ///
    /// so the right foot always leads and the left only ever catches up. That
    /// is a step-and-close gait, left/right asymmetric, and it spends every
    /// other double support with no fore/aft offset between the feet at all.
    ///
    /// The fix is the first step: make it a HALF stride and the phase is right
    /// from then on, with each foot alternately one stride ahead. The last
    /// step is likewise a half stride so the plan ends with the feet level,
    /// which is what the DCM plan's "come to rest at mid-stance" terminal
    /// condition assumes.
    ///
    /// `ramp` steps of stride build-up on top, because a robot that has been
    /// standing still cannot take a full step immediately -- the DCM has to
    /// already be moving at the walking rate the instant single support
    /// begins.
    pub fn alternating(plan: &GaitPlan, initial: &Footsteps, stride: f64, ramp: usize) -> Self {
        let n_steps = plan
            .slices
            .iter()
            .filter(|s| matches!(s.support, Support::Single { .. }))
            .count();
        let mut cur = *initial;
        let mut at = Vec::with_capacity(plan.slices.len());
        let mut step = 0usize;
        for s in &plan.slices {
            match s.support {
                Support::Single { swing, .. } => {
                    let ramp_f = if ramp == 0 {
                        1.0
                    } else {
                        ((step + 1) as f64 / ramp as f64).min(1.0)
                    };
                    // Half on the way in and on the way out, full in between.
                    let half = step == 0 || step + 1 == n_steps;
                    let advance = if half { stride } else { 2.0 * stride };
                    let mut next = cur;
                    next.sole[swing].x += advance * ramp_f;
                    at.push(next);
                    cur = next;
                    step += 1;
                }
                Support::Double => at.push(cur),
            }
        }
        FootstepPlan {
            at,
            initial: *initial,
        }
    }

    /// Constant stride, reached over `ramp` steps from standing.
    ///
    /// Every failure above stride 0.04 happened 1 to 1.7 s after the FIRST
    /// step, not after any accumulation, which points at the start rather
    /// than at a steady-state ceiling: the plan asks a robot that has been
    /// standing still to take a full-length step immediately, and the DCM has
    /// to be moving at the walking rate the moment single support begins.
    /// Ramping asks for it over several steps instead.
    pub fn ramped_stride(plan: &GaitPlan, initial: &Footsteps, stride: f64, ramp: usize) -> Self {
        let mut cur = *initial;
        let mut at = Vec::with_capacity(plan.slices.len());
        let mut step = 0usize;
        for s in &plan.slices {
            match s.support {
                Support::Single { swing, .. } => {
                    let f = if ramp == 0 {
                        1.0
                    } else {
                        ((step + 1) as f64 / ramp as f64).min(1.0)
                    };
                    let mut next = cur;
                    next.sole[swing].x += 2.0 * stride * f;
                    at.push(next);
                    cur = next;
                    step += 1;
                }
                Support::Double => at.push(cur),
            }
        }
        FootstepPlan {
            at,
            initial: *initial,
        }
    }

    /// Move one foot's planted position by `d`, from slice `from` onward.
    ///
    /// This is how a footstep adaptation becomes part of the plan rather than
    /// a per-tick override: once the foot has landed somewhere other than
    /// nominal, every future segment is relative to where it actually is.
    /// Insert a footstep entry right after slice `after` (the plan is one
    /// entry per gait slice; keep the two in step).
    pub fn insert_after(&mut self, after: usize, fs: Footsteps) {
        self.at.insert(after + 1, fs);
    }

    pub fn shift_from(&mut self, from: usize, side: Side, d: na::Vector2<f64>) {
        for f in self.at.iter_mut().skip(from) {
            f.sole[side].x += d.x;
            f.sole[side].y += d.y;
        }
    }

    /// Shift only the landing of `side` that slice `from` belongs to: every
    /// slice from `from` on while that foot stays planted at the same place,
    /// stopping at the first slice where the plan moves it again. Later
    /// landings keep their planned positions (a second tread is not missed).
    /// Announced to Codex in doc/from_claude/20260923_to_codex_shift_next.md.
    pub fn shift_next(&mut self, from: usize, side: Side, d: na::Vector2<f64>) {
        let Some(first) = self.at.get(from) else { return };
        let anchor = first.sole[side];
        for f in self.at.iter_mut().skip(from) {
            if (f.sole[side] - anchor).norm() > 1e-9 {
                break;
            }
            f.sole[side].x += d.x;
            f.sole[side].y += d.y;
        }
    }

    /// [`Self::shift_next`] plus everything the plan does AFTER that landing:
    /// the landing foot's later positions and the other foot's positions once
    /// it has left where it stands now. The walk then continues from the
    /// moved landing instead of returning to the pre-shift ladder. The foot
    /// standing at `from` is not moved.
    pub fn shift_rest(&mut self, from: usize, side: Side, d: na::Vector2<f64>) {
        let Some(first) = self.at.get(from).cloned() else { return };
        let other = 1 - side;
        let other_anchor = first.sole[other];
        let mut other_left = false;
        for f in self.at.iter_mut().skip(from) {
            f.sole[side].x += d.x;
            f.sole[side].y += d.y;
            if !other_left && (f.sole[other] - other_anchor).norm() > 1e-9 {
                other_left = true;
            }
            if other_left {
                f.sole[other].x += d.x;
                f.sole[other].y += d.y;
            }
        }
    }

    /// Re-plan from slice `from` (a single support with `swing` in the air):
    /// the swing lands at `landing` (x, y; z kept), and every LATER step is
    /// placed relative to the foot it steps past exactly as the ORIGINAL plan
    /// placed it (so its stride / width / speed schedule is kept), except
    /// that the first `stand_steps` later steps take no forward offset
    /// (stepping in place) and the next `ramp_steps` grow back to it.
    pub fn replan_from_landing(
        &mut self,
        plan: &GaitPlan,
        from: usize,
        swing: Side,
        landing: na::Vector2<f64>,
        stand_steps: usize,
        ramp_steps: usize,
    ) {
        if from >= self.at.len() {
            return;
        }
        let orig = self.at.clone();
        let mut cur = self.at[from];
        cur.sole[swing].x = landing.x;
        cur.sole[swing].y = landing.y;
        let mut in_first = true;
        let mut k = 0usize;
        for j in from..self.at.len().min(plan.slices.len()) {
            match plan.slices[j].support {
                Support::Double => {
                    in_first = false;
                    self.at[j] = cur;
                }
                Support::Single { swing: s, .. } => {
                    if in_first {
                        self.at[j] = cur;
                        continue;
                    }
                    let o = &orig[j];
                    let mut off = o.sole[s].xy() - o.sole[1 - s].xy();
                    if k < stand_steps {
                        off.x = 0.0;
                    } else if k < stand_steps + ramp_steps {
                        off.x *= (k - stand_steps + 1) as f64 / (ramp_steps + 1) as f64;
                    }
                    let mut next = cur;
                    next.sole[s].x = cur.sole[1 - s].x + off.x;
                    next.sole[s].y = cur.sole[1 - s].y + off.y;
                    next.sole[s].z = o.sole[s].z;
                    self.at[j] = next;
                    cur = next;
                    k += 1;
                }
            }
        }
    }

    pub fn at_slice(&self, i: usize) -> &Footsteps {
        &self.at[i.min(self.at.len() - 1)]
    }

    /// Total forward travel the plan asks the body to make.
    pub fn travel_x(&self) -> f64 {
        let last = self.at.last().unwrap_or(&self.initial);
        (last.mid_xy().x - self.initial.mid_xy().x).max(0.0)
    }
}

/// Corrects the scheduled support set with what the feet are ACTUALLY doing.
///
/// The schedule is a plan, and the plant does not read it. Measured on
/// kyo46rs stepping in place, both directions of disagreement occur inside a
/// single 200 ms double-support phase:
///
/// - the swing foot lands 30 ms EARLY and carries 10.7 N while the QP still
///   has it out of the contact set, and
/// - a scheduled stance foot BOUNCES to `fz = 0` while the QP is solving for
///   it to carry 14-31 N.
///
/// Either way the QP is solving against a contact set that does not exist,
/// and the force distribution it computes is applied to a robot that cannot
/// realise it -- measured, the per-foot vertical loads came out completely
/// swapped from the plan, QP (0.0, 63.4) N against a plant doing (29.8, 30.2).
///
/// Hysteresis matters in both directions: a foot momentarily unloaded during
/// a weight transfer is normal and must not be dropped, and contact chatter on
/// touchdown must not add and remove the foot on alternate ticks.
#[derive(Clone, Debug)]
pub struct ContactCorrection {
    /// Load above which an unscheduled foot is admitted to the contact set.
    pub on_n: f64,
    /// Load below which a scheduled stance foot is dropped from it.
    pub off_n: f64,
    /// Consecutive ticks the condition must hold before acting.
    pub ticks: u32,
    on_count: [u32; 2],
    off_count: [u32; 2],
    state: [bool; 2],
    previous_nominal: Option<Support>,
    /// Re-admit a loaded scheduled stance foot immediately on every tick.
    /// Opt-in replay of the 2026-09-12 contact rule for regression diagnosis.
    pub scheduled_reentry: bool,
}

impl ContactCorrection {
    /// Thresholds as a fraction of body weight: admit at 10%, drop at 2%.
    pub fn new(weight_n: f64, ticks: u32) -> Self {
        ContactCorrection {
            on_n: 0.10 * weight_n,
            off_n: 0.02 * weight_n,
            ticks,
            on_count: [0; 2],
            off_count: [0; 2],
            state: [true; 2],
            previous_nominal: None,
            scheduled_reentry: false,
        }
    }

    /// Feed this tick's measured vertical load per foot and the scheduled
    /// support; returns the support set to actually solve against, plus the
    /// sides whose contact was newly established (they need a fresh anchor).
    /// Admit `side` on the SCHEDULE's say-so, bypassing the load hysteresis.
    /// Returns true when the contact is newly established (needs a fresh
    /// anchor). Measured 2026-09-24 on G1 stairs: a foot confirmed on a
    /// tread rests there carrying 2-25 N, under the 10% BW `on_n`, so the
    /// QP never models it, never plans a force through it, and it never
    /// loads -- the very loop this call breaks. The caller decides WHEN
    /// (a confirmed landing entering its scheduled stance).
    pub fn admit(&mut self, side: Side) -> bool {
        let was = self.state[side];
        self.state[side] = true;
        self.on_count[side] = self.ticks;
        self.off_count[side] = 0;
        !was
    }

    pub fn update(&mut self, nominal: Support, fz: [f64; 2]) -> (Vec<Side>, Vec<Side>) {
        let mut stance = Vec::with_capacity(2);
        let mut fresh = Vec::new();
        for side in 0..2 {
            let sched = nominal.is_stance(side);
            let entering_stance = sched
                && self
                    .previous_nominal
                    .is_none_or(|previous| !previous.is_stance(side));
            let loaded = fz[side] > self.on_n;
            let unloaded = fz[side] < self.off_n;
            self.on_count[side] = if loaded { self.on_count[side] + 1 } else { 0 };
            self.off_count[side] = if unloaded {
                self.off_count[side] + 1
            } else {
                0
            };

            // A commanded LIFT is not negotiable. The correction exists to
            // notice contacts the plan did not predict, not to veto the plan:
            // keeping a scheduled swing foot in the contact set because it has
            // not unloaded yet means nothing ever unloads it -- the contact
            // constraint pins it and the swing task never runs. Measured, that
            // turned 20 steps into a shuffle, foot apex 0.2-1.2 mm.
            if nominal.swing() == Some(side) {
                let was = self.state[side];
                self.state[side] = false;
                let _ = was;
                continue;
            }
            let was = self.state[side];
            // Hysteresis on the MEASURED state, not on the schedule.
            //
            // Keying the branch off `sched` looks natural and is wrong: a foot
            // arrives at its scheduled double-support phase with `off_count`
            // already saturated from the swing, so "scheduled stance, keep
            // unless it has left" drops it on the first tick -- and the
            // re-admission path was on the `!sched` branch, so it could never
            // come back. Measured, that stranded the foot out of the contact
            // set for the whole rest of the run.
            let now = if was {
                !(self.off_count[side] >= self.ticks)
            } else {
                self.on_count[side] >= self.ticks
            };
            // The schedule still gets a vote at one moment: entering a phase
            // that calls a foot stance, with the foot already loaded, admits
            // it immediately rather than after the hysteresis.
            let now = now || ((entering_stance || self.scheduled_reentry) && sched && loaded);
            self.state[side] = now;
            if now {
                stance.push(side);
                if !was {
                    fresh.push(side);
                }
            }
        }
        // Never hand the QP an empty contact set: with no contact the level-0
        // problem has no way to hold the robot up at all, and the fallback
        // would be commanding a posture PD in free flight. Keep the foot the
        // schedule believes in.
        self.previous_nominal = Some(nominal);
        if stance.is_empty() {
            let keep = match nominal {
                Support::Single { stance, .. } => stance,
                Support::Double => usize::from(fz[1] > fz[0]),
            };
            self.state[keep] = true;
            stance.push(keep);
        }
        (stance, fresh)
    }

    /// The side currently believed to be in contact, for logging.
    pub fn state(&self) -> [bool; 2] {
        self.state
    }
}

/// Swing-foot trajectory: smoothstep in xy, `sin^2` bump in z.
///
/// Both axes start and end with zero velocity, so the handover to stance
/// generates no impulsive torque on the leg. The vertical zero-velocity
/// property is the one that matters -- a cubic Bezier with nonzero touchdown
/// vz is what produced trunk bobbing on the quadruped.
///
/// This duplicates `quadruped_gait::swing_traj::swing_position`, which is
/// leg-count agnostic and would be reusable if it were `pub`. Twenty lines is
/// cheaper than a cross-crate change to reach it.
pub fn swing_position(
    lift_off: na::Vector3<f64>,
    touch_down: na::Vector3<f64>,
    swing_height: f64,
    frac: f64,
) -> na::Vector3<f64> {
    let t = frac.clamp(0.0, 1.0);
    // xy: smoothstep S(t) = 3t^2 - 2t^3, S'(0) = S'(1) = 0.
    let s = (3.0 - 2.0 * t) * t * t;
    let mut p = lift_off + (touch_down - lift_off) * s;
    // z bump: sin^2(pi t) h, zero derivative at both ends, peak at t = 0.5.
    p.z += (std::f64::consts::PI * t).sin().powi(2) * swing_height;
    p
}

/// Analytic derivative of [`swing_position`], in m/s, given the swing's real
/// duration. Feeding the task a velocity reference rather than damping to
/// zero is what stops the tracking error from growing with swing speed.
pub fn swing_velocity(
    lift_off: na::Vector3<f64>,
    touch_down: na::Vector3<f64>,
    swing_height: f64,
    frac: f64,
    duration_s: f64,
) -> na::Vector3<f64> {
    let t = frac.clamp(0.0, 1.0);
    let tt = duration_s.max(1e-6);
    let ds = 6.0 * t * (1.0 - t) / tt; // d/dt of smoothstep
    let mut v = (touch_down - lift_off) * ds;
    v.z += swing_height * std::f64::consts::PI * (2.0 * std::f64::consts::PI * t).sin() / tt;
    v
}

/// Stair clearance trajectory, with analytic velocity and acceleration.
/// Quintic phases start/end at rest with zero acceleration, including at the
/// final hold. `from` and `to` are foot-link origins in world coordinates.
pub fn stair_swing(
    from: na::Vector3<f64>,
    to: na::Vector3<f64>,
    clearance: f64,
    frac: f64,
    duration: f64,
) -> (na::Vector3<f64>, na::Vector3<f64>, na::Vector3<f64>) {
    stair_swing_windows(
        from,
        to,
        clearance,
        frac,
        duration,
        [0.30, 0.25, 0.70, 0.60],
    )
}

/// [`stair_swing_windows`] with the landing overdrive (how far below the
/// surface the drop is COMMANDED to end) exposed. 4 mm was tuned when the
/// swing tracked its target 25-45 mm loose; with the velocity-clamp fix the
/// foot goes where it is told, and a 4 mm sink leaves the last of the drop
/// unfinished at the slice end -- measured, the mount foot was still 20 mm
/// up when double support started, bounced, and the run died two slices
/// later.
pub fn stair_swing_sink(
    from: na::Vector3<f64>,
    to: na::Vector3<f64>,
    clearance: f64,
    frac: f64,
    duration: f64,
    w: [f64; 4],
    sink: f64,
) -> (na::Vector3<f64>, na::Vector3<f64>, na::Vector3<f64>) {
    stair_swing_impl(from, to, clearance, frac, duration, w, sink)
}

/// [`stair_swing`] with the phase windows exposed: `[rise_end,
/// travel_start, travel_end, drop_start]` as fractions of the swing (the
/// drop always ends at 0.95). The defaults reproduce `stair_swing`.
/// Lengthening the rise is the lever that matters: the quintic's peak
/// acceleration scales with 1/span^2, and the measured failure mode of a
/// short rise is not the rise itself but its REACTION -- the hip flexion
/// that lifts the foot at 15 m/s^2 sweeps it 45 mm backward first, which
/// the weaker xy feedback then has to recover mid-swing.
pub fn stair_swing_windows(
    from: na::Vector3<f64>,
    to: na::Vector3<f64>,
    clearance: f64,
    frac: f64,
    duration: f64,
    w: [f64; 4],
) -> (na::Vector3<f64>, na::Vector3<f64>, na::Vector3<f64>) {
    stair_swing_impl(from, to, clearance, frac, duration, w, 0.004)
}

fn stair_swing_impl(
    from: na::Vector3<f64>,
    to: na::Vector3<f64>,
    clearance: f64,
    frac: f64,
    duration: f64,
    w: [f64; 4],
    sink: f64,
) -> (na::Vector3<f64>, na::Vector3<f64>, na::Vector3<f64>) {
    assert!(duration > 0.0);
    let phase = |start: f64, span: f64| {
        let u = ((frac - start) / span).clamp(0.0, 1.0);
        let d = span * duration;
        (
            u.powi(3) * (10.0 + u * (-15.0 + 6.0 * u)),
            30.0 * u.powi(2) * (1.0 - u).powi(2) / d,
            60.0 * u * (1.0 - u) * (1.0 - 2.0 * u) / (d * d),
        )
    };
    let (h, hd, hdd) = phase(w[1], (w[2] - w[1]).max(1e-6));
    let (up, upd, updd) = phase(0.0, w[0].max(1e-6));
    let (down, downd, downdd) = phase(w[3], (0.95 - w[3]).max(1e-6));
    let top = from.z.max(to.z) + clearance;
    let rise = top - from.z;
    let drop = to.z - sink - top;
    let delta = to - from;
    (
        na::Vector3::new(
            from.x + delta.x * h,
            from.y + delta.y * h,
            from.z + rise * up + drop * down,
        ),
        na::Vector3::new(delta.x * hd, delta.y * hd, rise * upd + drop * downd),
        na::Vector3::new(delta.x * hdd, delta.y * hdd, rise * updd + drop * downdd),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stair_swing_derivatives_and_endpoints_for_both_directions() {
        let from = na::Vector3::new(0.1, -0.05, 0.20);
        for dz in [-0.104, 0.104] {
            let to = from + na::Vector3::new(0.22, 0.01, dz);
            let (p, v, a) = stair_swing(from, to, 0.02, 0.0, 0.45);
            assert!((p - from).norm() < 1e-12);
            assert!(v.norm() < 1e-12 && a.norm() < 1e-12);
            let (p, v, a) = stair_swing(from, to, 0.02, 1.0, 0.45);
            assert!((p - (to - na::Vector3::new(0.0, 0.0, 0.004))).norm() < 1e-12);
            assert!(v.norm() < 1e-12 && a.norm() < 1e-12);
            for f in [0.1, 0.27, 0.4, 0.65, 0.8] {
                let eps = 1e-6;
                let (_, v, a) = stair_swing(from, to, 0.02, f, 0.45);
                let (pm, vm, _) = stair_swing(from, to, 0.02, f - eps, 0.45);
                let (pp, vp, _) = stair_swing(from, to, 0.02, f + eps, 0.45);
                assert!(((pp - pm) / (2.0 * eps * 0.45) - v).norm() < 1e-7);
                assert!(((vp - vm) / (2.0 * eps * 0.45) - a).norm() < 1e-6);
            }
        }
    }

    #[test]
    fn late_contact_waits_across_boundary_but_has_fixed_deadline() {
        let mut gait = GaitPlan::new(&GaitParams::default());
        let mut phase = ContactPhase::new(0.75, 20.0, 3, 0.5);
        let end = gait.slices[1].t1;
        let limit = end + gait.slices[1].duration() * 0.5;
        phase.update(&mut gait, 2.0, 0.005, [0.0; 2]);
        assert!(phase.update(&mut gait, end, 0.005, [0.0; 2]) < 0.0);
        assert!(matches!(gait.support_at(end), Support::Single { .. }));
        for k in 1..=46 {
            phase.update(&mut gait, end + k as f64 * 0.005, 0.005, [0.0; 2]);
        }
        assert!((gait.slices[1].t1 - limit).abs() < 1e-9);
        assert_eq!(gait.support_at(limit + 1e-9), Support::Double);
        assert!((gait.slices[2].duration() - 0.20).abs() < 1e-9);
    }

    #[test]
    fn contact_confirmation_rejects_scuffs_and_finishes_late_swing() {
        let mut gait = GaitPlan::new(&GaitParams::default());
        let mut phase = ContactPhase::new(0.75, 20.0, 3, 0.5);
        phase.update(&mut gait, 2.0, 0.005, [0.0, 40.0]);
        for t in [2.10, 2.11, 2.12] {
            assert_eq!(phase.update(&mut gait, t, 0.005, [0.0, 40.0]), 0.0);
        }
        let end = gait.slices[1].t1;
        for (k, load) in [40.0, 0.0, 40.0, 40.0, 40.0].into_iter().enumerate() {
            let t = end + k as f64 * 0.005;
            phase.update(&mut gait, t, 0.005, [0.0, load]);
            if k < 4 {
                assert!(matches!(gait.support_at(t), Support::Single { .. }));
            }
        }
        assert_eq!(gait.support_at(end + 0.020), Support::Double);
    }

    #[test]
    fn early_contact_shifts_future_slices_without_changing_their_duration() {
        let mut gait = GaitPlan::new(&GaitParams::default());
        let mut phase = ContactPhase::new(0.75, 20.0, 3, 0.5);
        phase.update(&mut gait, 2.0, 0.005, [0.0; 2]);
        for t in [2.35, 2.355, 2.36] {
            phase.update(&mut gait, t, 0.005, [0.0, 40.0]);
        }
        assert!((gait.slices[1].t1 - 2.36).abs() < 1e-9);
        assert_eq!(gait.support_at(2.36), Support::Double);
        assert!((gait.slices[3].duration() - 0.45).abs() < 1e-9);
    }

    #[test]
    fn plan_alternates_and_covers_the_timeline() {
        let p = GaitParams {
            t_start: 1.0,
            t_ds: 0.2,
            t_ss: 0.4,
            n_steps: 3,
            first_swing: RIGHT,
            t_end: 10.0,
        };
        let plan = GaitPlan::new(&p);
        // start DS, then (SS, DS) x 3, then the SETTLE slice. The settle is
        // separate on purpose -- stretching the last inter-step double support
        // to t_end spreads the ZMP's return to mid-stance over the whole tail.
        assert_eq!(plan.slices.len(), 8);
        assert_eq!(plan.slices[6].support, Support::Double);
        assert_eq!(plan.slices[7].support, Support::Double);
        // ...and the inter-step double support keeps its own short duration.
        assert!((plan.slices[6].duration() - 0.2).abs() < 1e-12);
        assert_eq!(plan.slices[0].support, Support::Double);
        assert_eq!(
            plan.slices[1].support,
            Support::Single {
                stance: LEFT,
                swing: RIGHT
            }
        );
        assert_eq!(
            plan.slices[3].support,
            Support::Single {
                stance: RIGHT,
                swing: LEFT
            }
        );
        assert_eq!(
            plan.slices[5].support,
            Support::Single {
                stance: LEFT,
                swing: RIGHT
            }
        );
        // contiguous, and the tail runs to t_end
        for w in plan.slices.windows(2) {
            assert!((w[0].t1 - w[1].t0).abs() < 1e-12);
        }
        assert_eq!(plan.slices.last().unwrap().t1, 10.0);
        assert_eq!(plan.steps_taken(10.0), 3);
    }

    #[test]
    fn no_steps_is_double_support_throughout() {
        let p = GaitParams {
            n_steps: 0,
            t_start: 1.0,
            t_end: 5.0,
            ..Default::default()
        };
        let plan = GaitPlan::new(&p);
        // The initial weight shift and the settle, both double support.
        assert_eq!(plan.slices.len(), 2);
        assert_eq!(plan.support_at(0.5), Support::Double);
        assert_eq!(plan.support_at(4.0), Support::Double);
        assert_eq!(plan.slices.last().unwrap().t1, 5.0);
    }

    #[test]
    fn settle_slices_are_separate_and_only_between_steps() {
        let p = GaitParams {
            n_steps: 3,
            t_end: 8.0,
            ..GaitParams::default()
        };
        let mut plan = GaitPlan::new(&p);
        let old_duration: f64 = plan.slices.iter().map(Slice::duration).sum();
        assert_eq!(plan.insert_ds_settles(0.15, 1), 2);
        let new_duration: f64 = plan.slices.iter().map(Slice::duration).sum();
        assert!((new_duration - old_duration - 0.30).abs() < 1e-12);
        let consecutive = plan
            .slices
            .windows(2)
            .filter(|w| {
                matches!(w[0].support, Support::Double) && matches!(w[1].support, Support::Double)
            })
            .count();
        // Each three-DS inter-step run has two consecutive pairs, plus the
        // existing final settle pair.
        assert_eq!(consecutive, 5);
    }

    #[test]
    fn settle_insertion_can_start_after_approach_steps() {
        let p = GaitParams {
            n_steps: 4,
            t_end: 8.0,
            ..GaitParams::default()
        };
        let mut plan = GaitPlan::new(&p);
        assert_eq!(plan.insert_ds_settles(0.15, 3), 1);
    }

    #[test]
    fn a_speed_change_leaves_the_plan_identical_before_it_and_moves_further_after() {
        let p = GaitParams { t_start: 1.0, t_ds: 0.2, t_ss: 0.35, n_steps: 10, first_swing: RIGHT, t_end: 20.0 };
        let plan = GaitPlan::new(&p);
        let init = Footsteps {
            sole: [na::Vector3::new(0.0, 0.05, 0.0), na::Vector3::new(0.0, -0.05, 0.0)],
            yaw: [0.0, 0.0],
        };
        let t_step = p.t_ss + p.t_ds;
        let a = FootstepPlan::velocity(&plan, &init, 0.2, 0.0, 0.0, t_step, 0);
        let b = FootstepPlan::velocity_with_change(&plan, &init, 0.2, 0.0, 0.0, t_step, 0, Some((4, 0.4, 0)));
        let singles: Vec<usize> = plan.slices.iter().enumerate()
            .filter(|(_, s)| matches!(s.support, Support::Single { .. })).map(|(i, _)| i).collect();
        let (i0, il) = (singles[0], *singles.last().unwrap());
        assert!((a.at[i0].mid_xy().x - b.at[i0].mid_xy().x).abs() < 1e-12);
        assert!(b.at[il].mid_xy().x > a.at[il].mid_xy().x + 0.1, "{} vs {}", b.at[il].mid_xy().x, a.at[il].mid_xy().x);
    }

    #[test]
    fn alternating_keeps_the_feet_one_stride_apart_and_swaps_the_lead() {
        let p = GaitParams {
            t_start: 1.0,
            t_ds: 0.2,
            t_ss: 0.35,
            n_steps: 8,
            first_swing: RIGHT,
            t_end: 20.0,
        };
        let plan = GaitPlan::new(&p);
        let init = Footsteps {
            sole: [
                na::Vector3::new(0.0, 0.05, 0.0),
                na::Vector3::new(0.0, -0.05, 0.0),
            ],
            yaw: [0.0, 0.0],
        };
        let stride = 0.05;
        let fp = FootstepPlan::alternating(&plan, &init, stride, 0);

        // Gap after each single-support slice, in step order.
        let gaps: Vec<f64> = plan
            .slices
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s.support, Support::Single { .. }))
            .map(|(i, _)| fp.at_slice(i).sole[LEFT].x - fp.at_slice(i).sole[RIGHT].x)
            .collect();

        // Steady steps: one stride apart, lead alternating. Excludes the
        // half step in and the half step out.
        for (k, g) in gaps.iter().enumerate().take(gaps.len() - 1).skip(1) {
            assert!(
                (g.abs() - stride).abs() < 1e-12,
                "step {k}: gap {g} is not one stride"
            );
            assert!(
                g * gaps[k - 1] < 0.0 || k == 1,
                "step {k}: the lead did not swap ({} then {g})",
                gaps[k - 1]
            );
        }
        // ...and the plan closes with the feet level.
        assert!(
            gaps.last().unwrap().abs() < 1e-12,
            "the last step does not close the feet"
        );

        // The naive version is what this replaces: it alternates one stride
        // apart with dead level, which is a step-and-close.
        let naive = FootstepPlan::constant_stride(&plan, &init, stride);
        let ngaps: Vec<f64> = plan
            .slices
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s.support, Support::Single { .. }))
            .map(|(i, _)| naive.at_slice(i).sole[LEFT].x - naive.at_slice(i).sole[RIGHT].x)
            .collect();
        assert!(ngaps[1].abs() < 1e-12 && ngaps[3].abs() < 1e-12);
    }

    #[test]
    fn contact_correction_requires_sustained_recontact_within_one_phase() {
        let mut correction = ContactCorrection::new(100.0, 3);
        for _ in 0..3 {
            correction.update(Support::Double, [0.0, 20.0]);
        }
        assert_eq!(correction.state(), [false, true]);

        for _ in 0..2 {
            let (stance, fresh) = correction.update(Support::Double, [20.0, 20.0]);
            assert!(!stance.contains(&LEFT));
            assert!(fresh.is_empty());
        }
        let (stance, fresh) = correction.update(Support::Double, [20.0, 20.0]);
        assert!(stance.contains(&LEFT));
        assert_eq!(fresh, vec![LEFT]);
    }

    #[test]
    fn contact_correction_admits_loaded_foot_on_phase_entry() {
        let mut correction = ContactCorrection::new(100.0, 3);
        for _ in 0..3 {
            correction.update(
                Support::Single {
                    stance: RIGHT,
                    swing: LEFT,
                },
                [0.0, 20.0],
            );
        }
        let (stance, fresh) = correction.update(Support::Double, [20.0, 20.0]);
        assert!(stance.contains(&LEFT));
        assert_eq!(fresh, vec![LEFT]);
    }

    #[test]
    fn velocity_command_reproduces_the_x_only_plan() {
        // The generalisation has to be a strict superset, or every result
        // measured with `alternating` stops applying.
        let p = GaitParams {
            t_start: 1.0,
            t_ds: 0.2,
            t_ss: 0.35,
            n_steps: 10,
            first_swing: RIGHT,
            t_end: 30.0,
        };
        let plan = GaitPlan::new(&p);
        let init = Footsteps {
            sole: [
                na::Vector3::new(0.0, 0.05, 0.0),
                na::Vector3::new(0.0, -0.05, 0.0),
            ],
            yaw: [0.0, 0.0],
        };
        let t_step = p.t_ss + p.t_ds;
        let stride = 0.05;
        let a = FootstepPlan::alternating(&plan, &init, stride, 0);
        let b = FootstepPlan::velocity(&plan, &init, stride / t_step, 0.0, 0.0, t_step, 0);
        for i in 0..plan.slices.len() {
            for side in 0..2 {
                let (x, y) = (a.at_slice(i).sole[side], b.at_slice(i).sole[side]);
                assert!(
                    (x - y).norm() < 1e-12,
                    "slice {i} side {side}: {x:?} vs {y:?}"
                );
            }
        }
    }

    #[test]
    fn sideways_and_turning_keep_the_feet_on_their_own_sides() {
        let p = GaitParams {
            t_start: 1.0,
            t_ds: 0.2,
            t_ss: 0.35,
            n_steps: 12,
            first_swing: RIGHT,
            t_end: 30.0,
        };
        let plan = GaitPlan::new(&p);
        let init = Footsteps {
            sole: [
                na::Vector3::new(0.0, 0.05, 0.0),
                na::Vector3::new(0.0, -0.05, 0.0),
            ],
            yaw: [0.0, 0.0],
        };
        let t_step = p.t_ss + p.t_ds;
        for (vx, vy, wz) in [
            (0.0, 0.05, 0.0),
            (0.0, -0.05, 0.0),
            (0.0, 0.0, 0.3),
            (0.05, 0.0, -0.3),
        ] {
            let fp = FootstepPlan::velocity(&plan, &init, vx, vy, wz, t_step, 0);
            for i in 0..plan.slices.len() {
                let f = fp.at_slice(i);
                // Left must stay to the left of right IN THE BODY FRAME, which
                // for a turn means rotating the separation back.
                let th = 0.5 * (f.yaw[LEFT] + f.yaw[RIGHT]);
                let d = f.xy(LEFT) - f.xy(RIGHT);
                let lateral = -th.sin() * d.x + th.cos() * d.y;
                assert!(
                    lateral > 0.02,
                    "({vx},{vy},{wz}) slice {i}: feet crossed, lateral gap {lateral:.4}"
                );
            }
        }
    }

    #[test]
    fn turning_rotates_the_footstep_yaw() {
        let p = GaitParams {
            t_start: 1.0,
            t_ds: 0.2,
            t_ss: 0.35,
            n_steps: 10,
            first_swing: RIGHT,
            t_end: 30.0,
        };
        let plan = GaitPlan::new(&p);
        let init = Footsteps {
            sole: [
                na::Vector3::new(0.0, 0.05, 0.0),
                na::Vector3::new(0.0, -0.05, 0.0),
            ],
            yaw: [0.0, 0.0],
        };
        let t_step = 0.55;
        let wz = 0.4;
        let fp = FootstepPlan::velocity(&plan, &init, 0.0, 0.0, wz, t_step, 0);
        let last = fp.at.last().unwrap();
        let turned = 0.5 * (last.yaw[LEFT] + last.yaw[RIGHT]);
        // Ten steps at 0.4 rad/s over 0.55 s each, the feet lag the body by
        // about half a step.
        assert!(
            turned > 0.5 * 10.0 * wz * t_step,
            "only turned {turned:.3} rad"
        );
        assert!(turned <= 11.0 * wz * t_step + 1e-9);
    }

    #[test]
    fn the_settle_does_not_stretch_the_last_double_support() {
        // The bug: extending the final inter-step double support to `t_end`
        // made the ZMP ramp to mid-stance across the entire tail, so the robot
        // finished the walk standing on one foot for eleven seconds and then
        // fell over. The settle needs its own slice.
        let p = GaitParams {
            t_start: 1.0,
            t_ds: 0.2,
            t_ss: 0.35,
            n_steps: 4,
            first_swing: RIGHT,
            t_end: 60.0,
        };
        let plan = GaitPlan::new(&p);
        let last_step_end = plan
            .slices
            .iter()
            .rposition(|s| matches!(s.support, Support::Single { .. }))
            .expect("there are steps");
        // The double support right after the last step is a NORMAL one.
        assert!(
            (plan.slices[last_step_end + 1].duration() - 0.2).abs() < 1e-12,
            "the post-step double support was stretched to {} s",
            plan.slices[last_step_end + 1].duration()
        );
        // The tail is a separate slice and carries all the remaining time.
        assert_eq!(plan.slices.last().unwrap().t1, 60.0);
        assert!(plan.slices.last().unwrap().duration() > 50.0);
    }

    #[test]
    fn swing_endpoints_are_exact_and_boundary_velocity_is_zero() {
        let a = na::Vector3::new(0.0, 0.1, 0.0);
        let b = na::Vector3::new(0.2, 0.1, 0.0);
        assert!((swing_position(a, b, 0.04, 0.0) - a).norm() < 1e-15);
        assert!((swing_position(a, b, 0.04, 1.0) - b).norm() < 1e-15);
        // Peak clearance at mid-swing.
        assert!((swing_position(a, b, 0.04, 0.5).z - 0.04).abs() < 1e-15);
        // Touchdown vz is EXACTLY zero -- this is the property that matters.
        assert!(swing_velocity(a, b, 0.04, 1.0, 0.4).norm() < 1e-12);
        assert!(swing_velocity(a, b, 0.04, 0.0, 0.4).norm() < 1e-12);
    }
}

#[cfg(test)]
mod shift_next_tests {
    use super::*;
    use nalgebra as na;

    #[test]
    fn shift_next_moves_one_landing_only() {
        // Foot 0 planted at x=0 for slices 0..3, lands at x=1 for 3..6, then 2.
        let mk = |x: f64| Footsteps {
            sole: [na::Vector3::new(x, 0.1, 0.0), na::Vector3::new(0.5, -0.1, 0.0)],
            yaw: [0.0; 2],
        };
        let mut fp = FootstepPlan {
            initial: mk(0.0),
            at: vec![mk(0.0), mk(0.0), mk(0.0), mk(1.0), mk(1.0), mk(1.0), mk(2.0)],
        };
        fp.shift_next(3, 0, na::Vector2::new(0.02, -0.01));
        let xs: Vec<f64> = fp.at.iter().map(|f| f.sole[0].x).collect();
        assert_eq!(xs, vec![0.0, 0.0, 0.0, 1.02, 1.02, 1.02, 2.0]);
        assert!((fp.at[4].sole[0].y - 0.09).abs() < 1e-12);
        assert!((fp.at[6].sole[0].y - 0.1).abs() < 1e-12);
        // Foot 1 untouched.
        assert!(fp.at.iter().all(|f| (f.sole[1].x - 0.5).abs() < 1e-12));
    }
}
