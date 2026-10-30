//! The walker's plan, in one place: the support schedule, the footsteps, the
//! DCM reference built from them, the CoM height plan and the landing clock.
//!
//! Moved here from `examples/kyo46rs_walk.rs` on 2026-10-29 (S6 of
//! doc/from_claude/20261029_walker_extraction_plan.md). Thirteen writers edit
//! these in turn inside one control tick (the landing clock, DS_FEASIBLE,
//! PUSH_REPLAN, the contact planner, the MPC landing, ...), and each of them
//! rebuilt the DCM reference with the same five arguments. Bundling the plan
//! puts that rebuild behind [`Plan::rebuild_dcm`].
//!
//! The rebuild stays EAGER: a writer that changes the schedule or the
//! footsteps commits where it always rebuilt, and nothing is deferred. The
//! walker reads the old DCM reference between some edits and their rebuild
//! (and DS_FEASIBLE writes segment pressures that the next rebuild discards),
//! so moving a rebuild changes the result.
//!
//! S16 (the second stage of §3.2): the fields are private. A writer edits
//! through [`Plan::edit`] with its name ([`Writer`]); the [`PlanEditor`]
//! retimes, shifts or replaces, and [`PlanEditor::commit`] rebuilds the DCM
//! reference when something was edited and counts the rebuild against the
//! writer. Who owns what of a slice -- its end time, its landing, the whole
//! plan, the once-per-run and once-per-run-of-double-support writers -- is
//! [`SliceClaims`], set by the writers through their editor and read by the
//! others' gates. The landing clock's own claim (a landing that is confirmed
//! or being probed for) is its state: [`Plan::landing_closed`].
//!
//! KNOWN LEAK, kept as it is (§7 of the extraction plan, item 8): DS_FEASIBLE
//! writes the double support's ZMP onto the built reference after its commit
//! ([`Plan::set_segment_zmp`] / [`Plan::set_segment_zmp_ramp`]), and the next
//! commit that rebuilds (any writer's, later in the same tick or after)
//! silently drops it. Making the pressure part of the plan's state would
//! change the walk, so it is not done here.
use super::dcm::DcmPlan;

/// A gate's answer from [`SliceClaims`] (the first expression). With the
/// `walker-claims-check` feature, the controller-state judgement it
/// replaced (the second) is computed too and asserted equal -- the S16
/// check that the claims are a faithful copy before the old fields go.
macro_rules! claim_check {
    ($new:expr, $old:expr) => {{
        let new = $new;
        #[cfg(feature = "walker-claims-check")]
        {
            let old = $old;
            assert_eq!(new, old, "SliceClaims disagrees: {} vs {}", stringify!($new), stringify!($old));
        }
        new
    }};
}
pub(crate) use claim_check;
use super::gait::{FootstepPlan, GaitPlan};
use super::preview::{HeightPlan, LandingTiming};

/// The constants the DCM reference is built with, fixed at start-up.
#[derive(Clone, Copy, Debug)]
pub struct DcmParams {
    /// LIPM height: the CoM above the walking surface [m].
    pub z_lip: f64,
    /// Lateral reach of the single-support pressure (1.0 = the sole centre;
    /// see [`DcmPlan::from_footsteps`]).
    pub lat_scale: f64,
    /// Heel-to-toe pressure travel inside each stance [m] (see
    /// [`DcmPlan::from_footsteps_rolled`]).
    pub roll: f64,
}

impl DcmParams {
    /// The DCM reference of a schedule and a footstep plan.
    pub fn build(&self, gait: &GaitPlan, footsteps: &FootstepPlan) -> DcmPlan {
        DcmPlan::from_footsteps_rolled(gait, footsteps, self.z_lip, self.lat_scale, self.roll)
    }
}

/// Everything the walker plans ahead. Read through the accessors; written
/// only through [`Plan::edit`] (and DS_FEASIBLE's segment pressures, see the
/// module's note).
///
/// Invariant: `dcm` is `dcm_params.build(&gait, &footsteps)` as of the last
/// rebuild, plus any segment pressures written onto it since (DS_FEASIBLE's
/// `set_segment_zmp*`), which the next rebuild drops.
pub struct Plan {
    /// When each foot is in support.
    gait: GaitPlan,
    /// Where each foot stands in each slice.
    footsteps: FootstepPlan,
    /// The ZMP / DCM reference built from the two above.
    dcm: DcmPlan,
    /// CoM height reference (quintic retargets).
    height: HeightPlan,
    /// When a swing actually lands (extensions, confirmations, events).
    landing: LandingTiming,
    pub dcm_params: DcmParams,
    /// Who owns what of the slices (S16).
    claims: SliceClaims,
    /// Rebuilds of the DCM reference per writer ([`PlanEditor::commit`]).
    rebuilds: [u32; Writer::COUNT],
}

impl Plan {
    /// Bundle a schedule and footsteps with their DCM reference, built now.
    pub fn new(
        gait: GaitPlan,
        footsteps: FootstepPlan,
        dcm_params: DcmParams,
        height: HeightPlan,
        landing: LandingTiming,
    ) -> Self {
        let dcm = dcm_params.build(&gait, &footsteps);
        Plan {
            gait,
            footsteps,
            dcm,
            height,
            landing,
            dcm_params,
            claims: SliceClaims::default(),
            rebuilds: [0; Writer::COUNT],
        }
    }

    pub fn gait(&self) -> &GaitPlan {
        &self.gait
    }

    pub fn footsteps(&self) -> &FootstepPlan {
        &self.footsteps
    }

    pub fn dcm(&self) -> &DcmPlan {
        &self.dcm
    }

    pub fn height(&self) -> &HeightPlan {
        &self.height
    }

    pub fn landing(&self) -> &LandingTiming {
        &self.landing
    }

    pub fn claims(&self) -> &SliceClaims {
        &self.claims
    }

    /// How many times `by`'s commits rebuilt the DCM reference.
    pub fn rebuilds(&self, by: Writer) -> u32 {
        self.rebuilds[by as usize]
    }

    /// The landing clock's claim on slice `i`'s landing: in event mode, a
    /// landing that has touched or whose swing is probing for the ground is
    /// closed to the writers that would move it (the contact planner, the
    /// MPC landing). It is the clock's state, not a stored claim.
    pub fn landing_closed(&self, i: usize) -> bool {
        self.landing.landing_closed(i)
    }

    /// Start an edit by `by`. Nothing is rebuilt until [`PlanEditor::commit`].
    pub fn edit(&mut self, by: Writer) -> PlanEditor<'_> {
        PlanEditor { plan: self, by, touched: false }
    }

    /// DS_FEASIBLE's double-support ZMP, written onto the BUILT reference
    /// after its commit (the KNOWN LEAK of the module's note: the next
    /// rebuild drops it). The only write that is not through an editor.
    pub fn set_segment_zmp(&mut self, seg: usize, p: nalgebra::Vector2<f64>) {
        self.dcm.set_segment_zmp(seg, p);
    }

    /// The ramp variant of [`Plan::set_segment_zmp`] (the same leak).
    pub fn set_segment_zmp_ramp(&mut self, seg: usize, p0: nalgebra::Vector2<f64>, p1: nalgebra::Vector2<f64>) {
        self.dcm.set_segment_zmp_ramp(seg, p0, p1);
    }

    /// Rebuild the DCM reference from the current schedule and footsteps.
    fn rebuild_dcm(&mut self) {
        self.dcm = self.dcm_params.build(&self.gait, &self.footsteps);
    }

    /// The DCM reference the current schedule would have with footsteps
    /// `fs` instead (a what-if; the plan is not changed).
    pub fn dcm_for(&self, fs: &FootstepPlan) -> DcmPlan {
        self.dcm_params.build(&self.gait, fs)
    }
}

/// The plan's writers inside a control tick, in the order they run
/// (§3.1 of doc/from_claude/20261029_walker_extraction_plan.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Writer {
    /// #1: the landing detector's settings for this slice (extension,
    /// descent tolerance, end on touch). Writes the landing clock only.
    LandingSetup,
    /// #2: the landing clock (touchdown, timeout, extension); its retime is
    /// rebuilt when it moved the end or the timeout re-plan asks.
    LandingClock,
    /// #3: the touchdown event moves the plan onto the foot.
    ContactEvent,
    /// #4: DS_FEASIBLE sizes the double support (and writes its ZMP after
    /// the commit, the leak).
    DsFeasible,
    /// #5: DS_CAPTURE ends a hopeless double support early.
    DsCapture,
    /// #6: the CoM re-plan's engagement (the height plan restarts).
    ReplanStart,
    /// #7: PUSH_YIELD slows the rest of the plan.
    PushYield,
    /// #8: PUSH_REPLAN re-lays the plan from a new landing.
    PushReplan,
    /// #9: a recovery reference takes the phases and the next footfalls.
    RecoveryRef,
    /// #10: the contact planner (the landing shift, the step's end).
    ContactPlan,
    /// #11: the CoM height plan's retarget.
    Height,
    /// #12: the MPC's landing variable.
    MpcLanding,
    /// #13: the recovery reference ends (event mode back, re-centring).
    RecoveryEnd,
}

impl Writer {
    pub const COUNT: usize = 13;
    pub const ALL: [Writer; Writer::COUNT] = [
        Writer::LandingSetup,
        Writer::LandingClock,
        Writer::ContactEvent,
        Writer::DsFeasible,
        Writer::DsCapture,
        Writer::ReplanStart,
        Writer::PushYield,
        Writer::PushReplan,
        Writer::RecoveryRef,
        Writer::ContactPlan,
        Writer::Height,
        Writer::MpcLanding,
        Writer::RecoveryEnd,
    ];
}

/// One writer's edit of the plan. Reads see the plan as edited so far
/// (`Deref`); the schedule and the footsteps change only through the
/// methods here, and [`PlanEditor::commit`] rebuilds the DCM reference if
/// any of them was called (or [`PlanEditor::touch`]).
#[must_use = "an edit is rebuilt only by commit()"]
pub struct PlanEditor<'a> {
    plan: &'a mut Plan,
    by: Writer,
    touched: bool,
}

impl std::ops::Deref for PlanEditor<'_> {
    type Target = Plan;
    fn deref(&self) -> &Plan {
        self.plan
    }
}

impl PlanEditor<'_> {
    pub fn by(&self) -> Writer {
        self.by
    }

    /// Move slice `slice`'s end to `t1` (`GaitPlan::retime`); the shift.
    pub fn retime(&mut self, slice: usize, t1: f64) -> f64 {
        self.touched = true;
        self.plan.gait.retime(slice, t1)
    }

    /// `FootstepPlan::shift_next`.
    pub fn shift_next(&mut self, slice: usize, side: usize, d: nalgebra::Vector2<f64>) {
        self.touched = true;
        self.plan.footsteps.shift_next(slice, side, d);
    }

    /// `FootstepPlan::shift_from`.
    pub fn shift_from(&mut self, slice: usize, side: usize, d: nalgebra::Vector2<f64>) {
        self.touched = true;
        self.plan.footsteps.shift_from(slice, side, d);
    }

    /// Replace the footsteps wholesale (PUSH_REPLAN's re-laid plan).
    pub fn replace_footsteps(&mut self, fs: FootstepPlan) {
        self.touched = true;
        self.plan.footsteps = fs;
    }

    /// The footsteps to edit in place (PUSH_YIELD pulls every future
    /// landing back).
    pub fn footsteps_mut(&mut self) -> &mut FootstepPlan {
        self.touched = true;
        &mut self.plan.footsteps
    }

    /// Rebuild at the commit without an edit of this session's own: the
    /// landing clock's retime (made in its update, which does not touch),
    /// a timeout re-plan, a re-centring that may move nothing.
    pub fn touch(&mut self) {
        self.touched = true;
    }

    /// The landing clock's tick (`LandingTiming::update`): it may retime the
    /// slice it times (extend, confirm, time out) and returns the shift of
    /// its end. NOT touched: the walker rebuilds the clock's retime where it
    /// always did (the caller touches when the shift is non-zero or the
    /// timeout re-plan asks).
    pub fn update_landing(
        &mut self,
        actual: &crate::gait::Footsteps,
        fz: [f64; 2],
        t: f64,
        dt: f64,
        out: &mut dyn crate::sink::LogSink,
    ) -> f64 {
        let plan = &mut *self.plan;
        plan.landing.update(&mut plan.gait, &plan.footsteps, actual, fz, t, dt, out)
    }

    /// The landing clock's settings and events (not part of the DCM
    /// reference; no rebuild).
    pub fn landing_mut(&mut self) -> &mut LandingTiming {
        &mut self.plan.landing
    }

    /// The CoM height plan (no rebuild).
    pub fn height_mut(&mut self) -> &mut HeightPlan {
        &mut self.plan.height
    }

    /// This writer owns slice `i`'s end time from now on.
    pub fn claim_end_time(&mut self, i: usize) {
        self.plan.claims.end_time = Some((i, self.by));
    }

    /// This writer owns slice `i`'s landing from now on.
    pub fn claim_landing(&mut self, i: usize) {
        self.plan.claims.landing = Some((i, self.by));
    }

    /// This writer owns the whole plan until it is released.
    pub fn claim_whole(&mut self) {
        self.plan.claims.whole = Some(self.by);
    }

    /// The whole-plan claim ends (the recovery reference is over).
    pub fn release_whole(&mut self) {
        self.plan.claims.whole = None;
    }

    /// This writer has acted on the run of double-support slices up to
    /// `last` (once per run).
    pub fn claim_ds_run(&mut self, last: usize) {
        self.plan.claims.ds_run = Some((last, self.by));
    }

    /// This writer has acted, once for the run.
    pub fn claim_once(&mut self) {
        self.plan.claims.once |= 1 << self.by as u32;
    }

    /// End the edit: rebuild the DCM reference if anything was edited (or
    /// touched), and count it against the writer. True = rebuilt.
    pub fn commit(self) -> bool {
        if self.touched {
            self.plan.rebuild_dcm();
            self.plan.rebuilds[self.by as usize] += 1;
        }
        self.touched
    }
}

/// Who owns what of the slices (S16: the writers' agreements in one place;
/// before, each was its own field of the controller state). Set through
/// [`PlanEditor`]'s `claim_*`, read by the writers' gates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SliceClaims {
    /// Slice `i`'s end time and who set it (the contact planner's retime,
    /// once per slice; the landing extension must not re-open it).
    end_time: Option<(usize, Writer)>,
    /// Slice `i`'s landing and who placed it (PUSH_REPLAN; the MPC landing
    /// keeps off it, and the swing follows it).
    landing: Option<(usize, Writer)>,
    /// A writer that owns the whole plan (a recovery reference while it
    /// runs: the double-support capture, PUSH_REPLAN, the CoM drop, the
    /// contact planner and the MPC landing all stand back).
    whole: Option<Writer>,
    /// The run of double-support slices up to this one has been acted on
    /// (DS_CAPTURE, once per run).
    ds_run: Option<(usize, Writer)>,
    /// Writers that act once per run (PUSH_YIELD, the re-plan's
    /// engagement), as bits of `1 << Writer`.
    once: u32,
}

impl SliceClaims {
    /// Who owns slice `i`'s end time.
    pub fn end_time_owner(&self, i: usize) -> Option<Writer> {
        self.end_time.and_then(|(s, w)| (s == i).then_some(w))
    }

    /// Who owns slice `i`'s landing.
    pub fn landing_owner(&self, i: usize) -> Option<Writer> {
        self.landing.and_then(|(s, w)| (s == i).then_some(w))
    }

    /// Who owns the whole plan.
    pub fn whole_owner(&self) -> Option<Writer> {
        self.whole
    }

    /// The double-support run that contains slice `ds` has been acted on
    /// (the claimed run ends at or after `ds`).
    pub fn ds_run_claimed(&self, ds: usize) -> bool {
        self.ds_run.is_some_and(|(l, _)| l >= ds)
    }

    /// `by` has acted (once per run).
    pub fn done(&self, by: Writer) -> bool {
        self.once & (1 << by as u32) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gait::{Footsteps, GaitParams, RIGHT};
    use nalgebra as na;

    fn plan() -> Plan {
        let p = GaitParams {
            t_start: 1.0,
            t_ds: 0.2,
            t_ss: 0.4,
            n_steps: 6,
            first_swing: RIGHT,
            t_end: 8.0,
        };
        let gait = GaitPlan::new(&p);
        let steps = Footsteps {
            sole: [na::Vector3::new(0.0, 0.0706, 0.0), na::Vector3::new(0.0, -0.0706, 0.0)],
            yaw: [0.0, 0.0],
        };
        let fs = FootstepPlan::alternating(&gait, &steps, 0.1, 0);
        let params = DcmParams { z_lip: 0.66, lat_scale: 1.0, roll: 0.03 };
        Plan::new(gait, fs, params, HeightPlan::constant(0.66), LandingTiming::new(17.0))
    }

    /// The first single support: (slice, swing side).
    fn first_swing(p: &Plan) -> (usize, usize) {
        p.gait.slices.iter().enumerate().find_map(|(i, s)| s.support.swing().map(|sw| (i, sw))).unwrap()
    }

    fn same(a: &DcmPlan, b: &DcmPlan) -> bool {
        a.omega == b.omega
            && a.xi_eos == b.xi_eos
            && a.segs.len() == b.segs.len()
            && a.segs.iter().zip(&b.segs).all(|(x, y)| {
                x.t0 == y.t0 && x.duration == y.duration && x.p0 == y.p0 && x.p1 == y.p1
            })
    }

    #[test]
    fn rebuild_is_the_constructor_arithmetic() {
        let mut p = plan();
        let direct = DcmPlan::from_footsteps_rolled(&p.gait, &p.footsteps, 0.66, 1.0, 0.03);
        assert!(same(&p.dcm, &direct));
        // An edit is invisible until the rebuild, and then it is the direct build.
        let (i, side) = first_swing(&p);
        p.footsteps.shift_next(i, side, na::Vector2::new(0.02, -0.01));
        assert!(same(&p.dcm, &direct));
        p.rebuild_dcm();
        let moved = DcmPlan::from_footsteps_rolled(&p.gait, &p.footsteps, 0.66, 1.0, 0.03);
        assert!(same(&p.dcm, &moved));
        assert!(!same(&p.dcm, &direct));
    }

    #[test]
    fn commit_rebuilds_only_an_edit() {
        let mut p = plan();
        let before = DcmPlan::from_footsteps_rolled(&p.gait, &p.footsteps, 0.66, 1.0, 0.03);
        // A writer that only reads, or writes the landing clock, does not rebuild.
        let mut ed = p.edit(Writer::LandingSetup);
        ed.landing_mut().max_extension = 0.25;
        assert!(!ed.commit());
        assert!(same(&p.dcm, &before));
        assert_eq!(p.landing().max_extension, 0.25);
        assert_eq!(p.rebuilds(Writer::LandingSetup), 0);
        // An edit is invisible to the reference until its commit, and then
        // it is the direct build; the rebuild is counted against the writer.
        let (i, side) = first_swing(&p);
        let mut ed = p.edit(Writer::MpcLanding);
        ed.shift_next(i, side, na::Vector2::new(0.02, -0.01));
        assert!(same(&ed.dcm, &before));
        assert!(ed.commit());
        assert!(same(&p.dcm, &DcmPlan::from_footsteps_rolled(&p.gait, &p.footsteps, 0.66, 1.0, 0.03)));
        assert!(!same(&p.dcm, &before));
        assert_eq!(p.rebuilds(Writer::MpcLanding), 1);
        // touch(): rebuilt with no edit of its own.
        let mut ed = p.edit(Writer::LandingClock);
        ed.touch();
        assert!(ed.commit());
        assert_eq!(p.rebuilds(Writer::LandingClock), 1);
        assert_eq!(Writer::ALL.iter().map(|&w| p.rebuilds(w)).sum::<u32>(), 2);
    }

    /// The writers in their tick order: the landing clock's retime waits for
    /// a later commit, DS_FEASIBLE's segment pressure sits on the built
    /// reference, and the next writer's commit drops it (the KNOWN LEAK the
    /// module keeps).
    #[test]
    fn writers_in_tick_order_and_the_segment_leak() {
        let mut p = plan();
        let (i, side) = first_swing(&p);
        let ds = i + 1;
        assert!(matches!(p.gait().slices[ds].support, crate::gait::Support::Double));
        // #2: the clock extends the swing; its session does not rebuild.
        let mut clock = p.edit(Writer::LandingClock);
        let t1 = clock.gait().slices[i].t1;
        let t1_before = DcmPlan::from_footsteps_rolled(&clock.gait, &clock.footsteps, 0.66, 1.0, 0.03);
        clock.plan.gait.retime(i, t1 + 0.05); // what `update_landing` may do (untracked)
        assert!(!clock.commit());
        assert!(same(&p.dcm, &t1_before));
        // #3: the touchdown event's shift rebuilds, and so takes the clock's retime in.
        let mut ed = p.edit(Writer::ContactEvent);
        ed.shift_next(i, side, na::Vector2::new(0.0, 0.01));
        assert!(ed.commit());
        let built = DcmPlan::from_footsteps_rolled(&p.gait, &p.footsteps, 0.66, 1.0, 0.03);
        assert!(same(&p.dcm, &built));
        // #4: DS_FEASIBLE lengthens the double support, commits, then writes
        // its pressure onto the built reference.
        let mut ed = p.edit(Writer::DsFeasible);
        let t0 = ed.gait().slices[ds].t0;
        let dd = ed.retime(ds, t0 + 0.3);
        assert!(dd < 0.0); // lengthened (retime: positive = cut short)
        assert!(ed.commit());
        let seg = p.dcm().index_at(t0 + 1e-6);
        let pinned = na::Vector2::new(0.123, -0.045);
        p.set_segment_zmp(seg, pinned);
        assert_eq!(p.dcm().segs[seg].p0, pinned);
        // #12: the MPC landing's commit rebuilds and the pressure is gone.
        let mut ed = p.edit(Writer::MpcLanding);
        ed.shift_next(i + 2, 1 - side, na::Vector2::new(0.01, 0.0));
        assert!(ed.commit());
        assert_ne!(p.dcm().segs[seg].p0, pinned);
        assert!(same(&p.dcm, &DcmPlan::from_footsteps_rolled(&p.gait, &p.footsteps, 0.66, 1.0, 0.03)));
        assert_eq!(
            Writer::ALL.map(|w| p.rebuilds(w)),
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        );
    }

    #[test]
    fn slice_claims_name_their_owner() {
        let mut p = plan();
        assert_eq!(p.claims(), &SliceClaims::default());
        let mut ed = p.edit(Writer::ContactPlan);
        ed.claim_end_time(3);
        assert!(!ed.commit());
        assert_eq!(p.claims().end_time_owner(3), Some(Writer::ContactPlan));
        assert_eq!(p.claims().end_time_owner(4), None);
        // A later slice's claim replaces the earlier one (one slice at a time).
        p.edit(Writer::ContactPlan).claim_end_time(5);
        assert_eq!(p.claims().end_time_owner(3), None);
        assert_eq!(p.claims().end_time_owner(5), Some(Writer::ContactPlan));

        p.edit(Writer::PushReplan).claim_landing(7);
        assert_eq!(p.claims().landing_owner(7), Some(Writer::PushReplan));
        assert_eq!(p.claims().landing_owner(6), None);

        assert_eq!(p.claims().whole_owner(), None);
        p.edit(Writer::RecoveryRef).claim_whole();
        assert_eq!(p.claims().whole_owner(), Some(Writer::RecoveryRef));
        p.edit(Writer::RecoveryEnd).release_whole();
        assert_eq!(p.claims().whole_owner(), None);

        p.edit(Writer::DsCapture).claim_ds_run(9);
        assert!(p.claims().ds_run_claimed(8) && p.claims().ds_run_claimed(9));
        assert!(!p.claims().ds_run_claimed(10));

        assert!(!p.claims().done(Writer::PushYield));
        p.edit(Writer::PushYield).claim_once();
        assert!(p.claims().done(Writer::PushYield));
        assert!(!p.claims().done(Writer::ReplanStart));
        p.edit(Writer::ReplanStart).claim_once();
        assert!(p.claims().done(Writer::ReplanStart) && p.claims().done(Writer::PushYield));
        // Claims never rebuild.
        assert_eq!(Writer::ALL.iter().map(|&w| p.rebuilds(w)).sum::<u32>(), 0);
    }

    #[test]
    fn writer_indices_follow_all() {
        for (k, w) in Writer::ALL.iter().enumerate() {
            assert_eq!(*w as usize, k);
        }
    }

    #[test]
    fn dcm_for_leaves_the_plan_alone() {
        let p = plan();
        let mut probe = p.footsteps.clone();
        let (i, side) = first_swing(&p);
        probe.shift_next(i, side, na::Vector2::new(0.03, 0.0));
        let what_if = p.dcm_for(&probe);
        assert!(same(&what_if, &DcmPlan::from_footsteps_rolled(&p.gait, &probe, 0.66, 1.0, 0.03)));
        assert!(!same(&what_if, &p.dcm));
    }
}
