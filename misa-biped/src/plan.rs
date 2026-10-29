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
//! footsteps calls `rebuild_dcm` where it always rebuilt, and nothing is
//! deferred. The walker reads the old DCM reference between some edits and
//! their rebuild (and DS_FEASIBLE writes segment pressures that the next
//! rebuild discards), so moving a rebuild changes the result.
use super::dcm::DcmPlan;
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

/// Everything the walker plans ahead. The fields are public: writers edit
/// `gait` and `footsteps` directly and then call [`Plan::rebuild_dcm`].
///
/// Invariant: `dcm` is `dcm_params.build(&gait, &footsteps)` as of the last
/// rebuild, plus any segment pressures written onto it since (DS_FEASIBLE's
/// `set_segment_zmp*`), which the next rebuild drops.
pub struct Plan {
    /// When each foot is in support.
    pub gait: GaitPlan,
    /// Where each foot stands in each slice.
    pub footsteps: FootstepPlan,
    /// The ZMP / DCM reference built from the two above.
    pub dcm: DcmPlan,
    /// CoM height reference (quintic retargets).
    pub height: HeightPlan,
    /// When a swing actually lands (extensions, confirmations, events).
    pub landing: LandingTiming,
    pub dcm_params: DcmParams,
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
        Plan { gait, footsteps, dcm, height, landing, dcm_params }
    }

    /// Rebuild the DCM reference from the current schedule and footsteps.
    pub fn rebuild_dcm(&mut self) {
        self.dcm = self.dcm_params.build(&self.gait, &self.footsteps);
    }

    /// The DCM reference the current schedule would have with footsteps
    /// `fs` instead (a what-if; the plan is not changed).
    pub fn dcm_for(&self, fs: &FootstepPlan) -> DcmPlan {
        self.dcm_params.build(&self.gait, fs)
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
