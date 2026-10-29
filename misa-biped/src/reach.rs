//! How far the legs can actually put a foot.
//!
//! The footstep plan and the step adjustment (`contact_plan`) both move
//! landing points around inside FIXED boxes (`ADAPT_MAX_X`, `CP_MAX_X`, ...).
//! Nothing in either of them knows the leg has a length. Measured on G1
//! 2026-09-24: the landing walk-out after a stair is where the runs die, the
//! step adjustment sits on its box bound while it happens, and the fix that
//! helped (shortening the planned landing stride) is a constant chosen by
//! hand for one riser height -- the symptom-level version of this module.
//!
//! The bound used here is the double-support one, which is the binding case
//! for a long stride: at touchdown BOTH soles must be within reach of the
//! hips. With the hip at height `h_back` over the trailing sole and
//! `h_front` over the landing sole, the widest stride is
//!
//! ```text
//! s_max = sqrt(L^2 - h_back^2) + sqrt(L^2 - h_front^2)
//! ```
//!
//! which is just "each leg spans a right triangle". `L` is the straight-leg
//! distance from the hip pivot to the sole plane, scaled by `margin` because
//! a leg at full extension has no vertical authority left (and the knee
//! singularity is what throws the swing).
use nalgebra as na;

#[derive(Clone, Copy, Debug)]
pub struct LegReach {
    /// Hip pivot to sole plane, straight leg [m].
    pub max: f64,
    /// Fraction of `max` a PLAN may ask for. 1.0 = the kinematic limit.
    pub margin: f64,
}

impl LegReach {
    /// From the lengths of the segments between the hip pivot and the ankle,
    /// plus the sole's offset below the ankle.
    pub fn from_segments(segments: &[f64], sole_below_origin: f64, margin: f64) -> Self {
        LegReach {
            max: segments.iter().sum::<f64>() + sole_below_origin,
            margin: margin.clamp(0.1, 1.0),
        }
    }

    /// Usable length after the margin.
    pub fn usable(&self) -> f64 {
        self.max * self.margin
    }

    /// Horizontal distance a single leg can span with the sole `drop` below
    /// the hip. Zero when the drop alone exhausts the leg.
    pub fn horizontal(&self, drop: f64) -> f64 {
        let l = self.usable();
        (l * l - drop * drop).max(0.0).sqrt()
    }

    /// Widest stride (trailing sole -> landing sole, horizontal) that leaves
    /// both legs inside the margin at touchdown. `hip_z` is the hip height,
    /// `back_z` / `front_z` the two sole heights, all in the same frame.
    pub fn max_stride(&self, hip_z: f64, back_z: f64, front_z: f64) -> f64 {
        self.horizontal(hip_z - back_z) + self.horizontal(hip_z - front_z)
    }

    /// How much of the leg a pose is using, as a fraction of `max`. 1.0 is
    /// straight. This is the number to log: the reach CLAIM about a fall is
    /// only worth anything if the leg was actually near 1.
    pub fn used(&self, hip: &na::Vector3<f64>, sole: &na::Vector3<f64>) -> f64 {
        if self.max <= 0.0 {
            return 0.0;
        }
        (hip - sole).norm() / self.max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g1() -> LegReach {
        // hip_yaw -> knee 0.1939, knee -> ankle_pitch 0.3000,
        // ankle_pitch -> ankle_roll 0.0176, sole 0.035 below.
        LegReach::from_segments(&[0.1939, 0.3000, 0.0176], 0.035, 1.0)
    }

    #[test]
    fn straight_leg_length_matches_the_urdf_chain() {
        assert!((g1().max - 0.5465).abs() < 1e-3, "{}", g1().max);
    }

    #[test]
    fn a_drop_eats_the_horizontal_span() {
        let r = g1();
        assert!((r.horizontal(0.0) - r.max).abs() < 1e-12);
        assert!(r.horizontal(0.3) < r.horizontal(0.2));
        // Past the leg's length nothing is left, and it never goes negative.
        assert_eq!(r.horizontal(r.max + 0.1), 0.0);
    }

    #[test]
    fn stepping_up_widens_and_stepping_down_narrows_the_stride() {
        // Not the intuition the first draft of this test had: for a given hip
        // height a HIGHER landing surface is vertically closer to the hip, so
        // more of the leg is left for horizontal span. Descending is the
        // case that costs reach. What makes a step up hard is the hip having
        // to rise, which shows up as `hip_z` below.
        let r = LegReach { margin: 0.95, ..g1() };
        let hip = 0.50;
        let flat = r.max_stride(hip, 0.0, 0.0);
        assert!(r.max_stride(hip, 0.0, 0.08) > flat);
        assert!(r.max_stride(hip, 0.0, -0.08) < flat);
        // Both legs share the stride, so it beats one leg's span.
        assert!(flat > r.horizontal(hip));
    }

    #[test]
    fn a_higher_hip_costs_stride() {
        let r = LegReach { margin: 0.95, ..g1() };
        assert!(r.max_stride(0.52, 0.0, 0.0) < r.max_stride(0.46, 0.0, 0.0));
    }

    #[test]
    fn margin_scales_everything() {
        let full = g1();
        let tight = LegReach { margin: 0.5, ..full };
        assert!((tight.usable() - 0.5 * full.usable()).abs() < 1e-12);
        assert!(tight.horizontal(0.1) < full.horizontal(0.1));
    }
}
