//! Push-recovery references: a trajectory-optimised recovery (phase
//! durations, sole and CoM nodes) and the frame that plays it back in the
//! world once the walker engages it.
//!
//! Moved here from `examples/kyo46rs_walk.rs` on 2026-10-29 (S7 of
//! doc/from_claude/20261029_walker_extraction_plan.md). The interpolation
//! (`dcm_at`, `cop_at`, `sole_at`) is the example's arithmetic verbatim -- a
//! push run that engages a reference is covered by the bit-identity check.
//!
//! The references come from scripts/trajopt/push_recovery.py as JSON. This
//! crate does not read JSON: the caller parses the file and fills the public
//! fields (see doc/from_claude/20261005_trajopt_reference_plan.md §9 for the
//! format).
use nalgebra as na;

/// A recovery reference: phase durations, node times, sole-centre
/// trajectories and the CoM, all in the optimiser's frame (left foot stance
/// at y = `stance_y0`, push toward +y).
#[derive(Clone)]
pub struct RefLib {
    /// Phase durations [s]; their sum is the reference's length.
    pub phases: Vec<f64>,
    /// Swing sole height above the stance sole at the first node [m].
    pub swing_z0: f64,
    /// Swing phase (fraction of the single support) this reference is meant for.
    pub swing_phase: f64,
    /// Direction of the push this reference answers [deg]: 90 = toward the stance foot, 0 = forward.
    pub impulse_deg: f64,
    /// Solved from the LANDED state (the swing foot on the ground, unloaded):
    /// the re-step reference. Only RESTEP (removed) engaged it; the
    /// detectors never pick it.
    pub landed: bool,
    /// Walking speed the reference was solved at [m/s] (0.12 = the nominal
    /// gait; the library is picked by the nearest speed first, then phase).
    pub vx: f64,
    /// Node times [s], from 0.
    pub t: Vec<f64>,
    /// Sole centres at the nodes: [left (the stance), right].
    pub sole: [Vec<na::Vector3<f64>>; 2],
    /// CoM at the nodes, and the vertical force of each foot per interval.
    pub com: Vec<na::Vector3<f64>>,
    pub fz: [Vec<f64>; 2],
    /// y of the stance (left) sole at the first node: the frame's lateral origin.
    pub stance_y0: f64,
}

impl RefLib {
    /// Length of the reference [s].
    pub fn total(&self) -> f64 {
        self.phases.iter().sum()
    }
    fn interp(&self, xs: &[na::Vector3<f64>], tau: f64) -> na::Vector3<f64> {
        let t = &self.t;
        if tau <= t[0] {
            return xs[0];
        }
        for k in 0..t.len() - 1 {
            if tau <= t[k + 1] {
                let a = ((tau - t[k]) / (t[k + 1] - t[k]).max(1e-9)).clamp(0.0, 1.0);
                return xs[k] * (1.0 - a) + xs[k + 1] * a;
            }
        }
        *xs.last().unwrap()
    }
    /// Reference DCM (xy) at `tau`: c + c_dot / omega, c_dot by finite difference.
    pub fn dcm_at(&self, tau: f64, omega: f64) -> na::Vector2<f64> {
        let h = 0.01;
        let c = self.interp(&self.com, tau);
        let cd = (self.interp(&self.com, tau + h) - self.interp(&self.com, (tau - h).max(0.0))) / (h + (tau - (tau - h).max(0.0)));
        (c + cd / omega).xy()
    }
    /// Reference net CoP (xy) at `tau`: the soles weighted by their vertical force.
    pub fn cop_at(&self, tau: f64) -> na::Vector2<f64> {
        let t = &self.t;
        let mut k = 0usize;
        while k + 1 < self.fz[0].len() && tau > t[k + 1] {
            k += 1;
        }
        let (fl, fr) = (self.fz[0][k].max(0.0), self.fz[1][k].max(0.0));
        let (pl, pr) = (self.sole_at(0, tau), self.sole_at(1, tau));
        if fl + fr < 1.0 {
            return (0.5 * (pl + pr)).xy();
        }
        ((pl * fl + pr * fr) / (fl + fr)).xy()
    }
    /// Sole centre of reference foot `side` at reference time `tau`, linearly
    /// interpolated between nodes and held at the ends.
    pub fn sole_at(&self, side: usize, tau: f64) -> na::Vector3<f64> {
        let t = &self.t;
        if tau <= t[0] {
            return self.sole[side][0];
        }
        for k in 0..t.len() - 1 {
            if tau <= t[k + 1] {
                let a = ((tau - t[k]) / (t[k + 1] - t[k]).max(1e-9)).clamp(0.0, 1.0);
                return self.sole[side][k] * (1.0 - a) + self.sole[side][k + 1] * a;
            }
        }
        *self.sole[side].last().unwrap()
    }
}

/// An engaged reference: when it started, which walker foot plays the
/// reference's LEFT (stance) foot, and the frame that maps the reference into
/// the world (origin = measured stance sole at engagement, `mirror` = -1 when
/// the stance is the walker's right foot).
#[derive(Clone, Copy)]
pub struct RefMode {
    pub t0: f64,
    /// Reference time already elapsed at engagement (the detector fires
    /// some tens of ms after the push; the reference starts at the push).
    pub tau0: f64,
    pub stance: usize,
    pub origin: na::Vector3<f64>,
    pub mirror: f64,
}

impl RefMode {
    /// A point of the reference in the world.
    pub fn world(&self, lib: &RefLib, p: na::Vector3<f64>) -> na::Vector3<f64> {
        self.origin + na::Vector3::new(p.x, self.mirror * (p.y - lib.stance_y0), p.z)
    }
    pub fn world_xy(&self, lib: &RefLib, p: na::Vector2<f64>) -> na::Vector2<f64> {
        self.world(lib, na::Vector3::new(p.x, p.y, 0.0)).xy()
    }
    /// Walker foot index that plays reference foot `ref_side` (0 = its stance).
    pub fn foot(&self, ref_side: usize) -> usize {
        if ref_side == 0 { self.stance } else { 1 - self.stance }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two intervals: the CoM moves 0.1 m in y over 0.2 s, the swing (right)
    /// foot lifts and moves out, the stance carries all the load first.
    fn lib() -> RefLib {
        let v = na::Vector3::new;
        RefLib {
            phases: vec![0.1, 0.1],
            swing_z0: 0.0,
            swing_phase: 0.5,
            impulse_deg: 90.0,
            landed: false,
            vx: 0.12,
            t: vec![0.0, 0.1, 0.2],
            sole: [
                vec![v(0.0, 0.1, 0.0), v(0.0, 0.1, 0.0), v(0.0, 0.1, 0.0)],
                vec![v(0.0, -0.1, 0.0), v(0.05, -0.15, 0.05), v(0.1, -0.2, 0.0)],
            ],
            com: vec![v(0.0, 0.0, 0.7), v(0.0, 0.05, 0.7), v(0.0, 0.1, 0.7)],
            fz: [vec![300.0, 150.0], vec![0.0, 150.0]],
            stance_y0: 0.1,
        }
    }

    #[test]
    fn soles_interpolate_and_hold_at_the_ends() {
        let l = lib();
        assert_eq!(l.total(), 0.2);
        assert_eq!(l.sole_at(1, -1.0), l.sole[1][0]);
        assert_eq!(l.sole_at(1, 9.0), l.sole[1][2]);
        assert!((l.sole_at(1, 0.05) - na::Vector3::new(0.025, -0.125, 0.025)).norm() < 1e-12);
    }

    #[test]
    fn the_cop_follows_the_load() {
        let l = lib();
        // First interval: all the load on the stance (left) sole.
        assert!((l.cop_at(0.05) - na::Vector2::new(0.0, 0.1)).norm() < 1e-12);
        // Second: shared evenly, the midpoint of the two soles.
        let mid = 0.5 * (l.sole_at(0, 0.15) + l.sole_at(1, 0.15));
        assert!((l.cop_at(0.15) - mid.xy()).norm() < 1e-12);
    }

    #[test]
    fn the_dcm_leads_the_com_by_its_velocity() {
        let l = lib();
        let omega = 3.7;
        // Constant CoM velocity 0.5 m/s in y inside the reference.
        let xi = l.dcm_at(0.1, omega);
        assert!((xi.y - (0.05 + 0.5 / omega)).abs() < 1e-9, "{xi:?}");
    }

    #[test]
    fn the_right_stance_mirrors_the_reference() {
        let l = lib();
        let left = RefMode { t0: 0.0, tau0: 0.0, stance: 0, origin: na::Vector3::new(1.0, 0.2, 0.0), mirror: 1.0 };
        let right = RefMode { stance: 1, mirror: -1.0, ..left };
        let p = l.sole_at(1, 0.2);
        assert!((left.world(&l, p) - na::Vector3::new(1.1, -0.1, 0.0)).norm() < 1e-12);
        assert!((right.world(&l, p) - na::Vector3::new(1.1, 0.5, 0.0)).norm() < 1e-12);
        assert_eq!((left.foot(0), left.foot(1)), (0, 1));
        assert_eq!((right.foot(0), right.foot(1)), (1, 0));
    }
}
