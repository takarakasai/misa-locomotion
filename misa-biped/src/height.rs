//! The CoM height reference: every rule that moves it, in one place.
//!
//! The walker composes these in a fixed order (2026-10-19, G1 v6.10):
//!
//! 1. the level the feet stand on ([`stance_level`]: mean of the planned
//!    soles, or a fraction between them in double support),
//! 2. the descent lead ([`descent_lead`]: start down toward a lower landing
//!    before the swing),
//! 3. the stair-zone crouch ([`stair_zone_crouch`]),
//! 4. caps by what the legs can reach ([`reach_cap_landing`] for the next
//!    landing, [`reach_cap_soles`] for every sole the plan stands on),
//! 5. slewed drops that ride on top of whichever reference is live
//!    ([`drop_target`] = the larger of the speed table's drop and the
//!    backward-push drop, tracked by [`SlewedDrop`] at 50 mm/s).
//!
//! Each function is the exact arithmetic the walker used inline before the
//! refactor, so a run with the same knobs is bit-identical.
use nalgebra as na;

use super::terrain::Stairs;

/// The height the feet stand on: the mean of the planned sole heights, or
/// -- in a double support with `ds_frac` given -- that fraction of the way
/// from the lower sole to the higher one.
pub fn stance_level(soles_z: &[f64], ds_frac: Option<f64>) -> f64 {
    match ds_frac {
        Some(frac) if soles_z.len() == 2 => {
            let (z0, z1) = (soles_z[0], soles_z[1]);
            z0.min(z1) + frac * (z0.max(z1) - z0.min(z1))
        }
        // no sole down (flight): the ground the robot left, taken as 0
        _ if soles_z.is_empty() => 0.0,
        _ => soles_z.iter().sum::<f64>() / soles_z.len() as f64,
    }
}

/// Start down toward a LOWER landing before the swing: `lead` of the way
/// from the stance level to the landing height. A landing that is not
/// lower leaves the level alone.
pub fn descent_lead(h: f64, landing_z: Option<f64>, lead: f64) -> f64 {
    if lead <= 0.0 {
        return h;
    }
    match landing_z {
        Some(lz) if lz < h - 1e-6 => h - lead * (h - lz),
        _ => h,
    }
}

/// The crouch inside a known staircase's zone (from `lead` before its start
/// to `tail` past its landing, and the same around a descent flight).
pub fn stair_zone_crouch(x: f64, st: &Stairs, lead: f64, tail: f64, crouch: f64) -> f64 {
    if crouch <= 0.0 {
        return 0.0;
    }
    if (x >= st.start - lead && x <= st.landing_start() + tail)
        || (st.down_n > 0
            && x >= st.landing_end() - lead
            && x <= st.landing_end() + st.down_n as f64 * st.tread + tail)
    {
        crouch
    } else {
        0.0
    }
}

/// Cap the target by the leg's reach to the next LOWER landing: the CoM may
/// not sit higher than the point from which the landing is within `span`
/// of the hip, with the horizontal lever taken as half the stance-to-landing
/// distance and the hip `hip_drop` below the CoM.
pub fn reach_cap_landing(
    target: f64,
    landing: &na::Vector3<f64>,
    stance: &na::Vector3<f64>,
    span: f64,
    hip_drop: f64,
) -> f64 {
    if !(landing.z < stance.z - 0.005 && hip_drop.is_finite()) {
        return target;
    }
    let d = 0.5 * (landing.xy() - stance.xy()).norm();
    if d < span {
        target.min(landing.z + (span * span - d * d).sqrt() + hip_drop)
    } else {
        target
    }
}

/// Cap the target by the leg's reach to every sole in `soles` (the feet the
/// plan believes in plus the landing the swing is heading for), measured
/// from the CoM's horizontal position, and never below `floor`.
pub fn reach_cap_soles<I: Iterator<Item = na::Vector3<f64>>>(
    target: f64,
    com_xy: &na::Vector2<f64>,
    soles: I,
    span: f64,
    hip_drop: f64,
    floor: f64,
) -> f64 {
    if !hip_drop.is_finite() {
        return target;
    }
    let mut cap = f64::INFINITY;
    for sole in soles {
        let d = (com_xy - sole.xy()).norm();
        if d < span {
            cap = cap.min(sole.z + (span * span - d * d).sqrt() + hip_drop);
        }
    }
    if cap.is_finite() {
        target.min(cap).max(floor)
    } else {
        target
    }
}

/// The drop that rides on top of the live reference: the larger of the
/// speed table's drop (active from the speed change on) and the
/// backward-push drop (active while the detector holds).
pub fn drop_target(speed_drop: f64, push_drop: f64) -> f64 {
    speed_drop.max(push_drop)
}

/// A drop tracked at a bounded rate. `step` returns this tick's increment
/// (so the caller can turn it into a velocity reference).
#[derive(Clone, Copy, Debug, Default)]
pub struct SlewedDrop {
    pub track: f64,
}

impl SlewedDrop {
    pub fn step(&mut self, target: f64, rate_per_s: f64, dt: f64) -> f64 {
        let d = (target - self.track).clamp(-rate_per_s * dt, rate_per_s * dt);
        self.track += d;
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stance_level_is_the_mean_or_the_fraction_between_two_soles() {
        assert!((stance_level(&[0.0, 0.1], None) - 0.05).abs() < 1e-12);
        assert!((stance_level(&[0.1, 0.0], Some(0.0)) - 0.0).abs() < 1e-12);
        assert!((stance_level(&[0.1, 0.0], Some(1.0)) - 0.1).abs() < 1e-12);
        assert!((stance_level(&[0.2], Some(0.5)) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn descent_lead_only_goes_down() {
        assert_eq!(descent_lead(0.1, Some(0.0), 0.5), 0.05);
        assert_eq!(descent_lead(0.1, Some(0.2), 0.5), 0.1);
        assert_eq!(descent_lead(0.1, None, 0.5), 0.1);
        assert_eq!(descent_lead(0.1, Some(0.0), 0.0), 0.1);
    }

    #[test]
    fn reach_caps_bound_from_above_only() {
        let landing = na::Vector3::new(0.3, 0.0, -0.1);
        let stance = na::Vector3::new(0.0, 0.0, 0.0);
        let capped = reach_cap_landing(0.70, &landing, &stance, 0.5, 0.1);
        assert!(capped < 0.70 && capped > 0.0);
        // a level landing leaves the target alone
        assert_eq!(reach_cap_landing(0.70, &stance, &stance, 0.5, 0.1), 0.70);
        let soles = [na::Vector3::new(0.0, 0.0, 0.0), na::Vector3::new(0.4, 0.0, 0.0)];
        let c = reach_cap_soles(0.70, &na::Vector2::new(0.2, 0.0), soles.iter().copied(), 0.5, 0.1, 0.45);
        assert!(c <= 0.70 && c >= 0.45);
        assert_eq!(reach_cap_soles(0.70, &na::Vector2::zeros(), soles.iter().copied(), 0.5, f64::NAN, 0.45), 0.70);
    }

    #[test]
    fn slewed_drop_is_rate_bounded_and_returns_the_increment() {
        let mut s = SlewedDrop::default();
        let d = s.step(0.035, 0.05, 0.005);
        assert!((d - 0.00025).abs() < 1e-12 && (s.track - 0.00025).abs() < 1e-12);
        for _ in 0..1000 {
            s.step(0.035, 0.05, 0.005);
        }
        assert!((s.track - 0.035).abs() < 1e-12);
        assert_eq!(drop_target(0.024, 0.035), 0.035);
    }
}
