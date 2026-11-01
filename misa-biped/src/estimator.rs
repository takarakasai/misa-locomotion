//! The state estimator: a sensor reading -> the [`StateEstimate`] the walker
//! runs on.
//!
//! The reading is whatever the robot's sensors report, in the estimate's own
//! shape (floating base pose and velocities in world axes, joint angles and
//! rates in the host's joint order). What makes a reading imperfect -- delay,
//! noise, bias, how an accelerometer sees the base -- is the host's sensor
//! model (articara's rig, or the real hardware); this is only what the
//! controller side does with it, in this order:
//!
//! 1. `vel_fuse_hz`: a complementary filter on the base linear velocity. The
//!    accelerometer (world-frame base acceleration, gravity removed) carries
//!    the high band between ticks; the velocity reading corrects below the
//!    crossover. With `vel_fuse_bias_t` an accelerometer-bias state, driven
//!    by the integral of the innovation, removes a constant bias's steady
//!    velocity error (it settles in roughly that many seconds).
//! 2. `filter_hz`: a first-order low-pass on every velocity (joint rates,
//!    base linear and angular velocity).
//! 3. `filter_v_hz`: the same low-pass on the base linear velocity only.
//! 4. `qd_clamp`: the joint rates the controller sees are saturated.
//!
//! Each stage at 0 (the default) is off and passes the reading through
//! unchanged; `qd_clamp` at infinity likewise.
use crate::robot::StateEstimate;

/// Estimator settings (all off by default; see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct EstimatorConfig {
    /// Sample time of the filters and the fusion [s] (the control period).
    pub dt: f64,
    /// Complementary-filter crossover [Hz] (0 = off).
    pub vel_fuse_hz: f64,
    /// Accelerometer-bias time constant [s] (0 = no bias state).
    pub vel_fuse_bias_t: f64,
    /// Low-pass cut-off on every velocity [Hz] (0 = off).
    pub filter_hz: f64,
    /// Low-pass cut-off on the base linear velocity only [Hz] (0 = off).
    pub filter_v_hz: f64,
    /// Joint-rate saturation [rad/s].
    pub qd_clamp: f64,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self { dt: 0.005, vel_fuse_hz: 0.0, vel_fuse_bias_t: 0.0, filter_hz: 0.0, filter_v_hz: 0.0, qd_clamp: f64::INFINITY }
    }
}

/// The estimator's state between ticks (see the module docs).
#[derive(Clone, Debug)]
pub struct Estimator {
    pub cfg: EstimatorConfig,
    /// (fused base linear velocity, accelerometer-bias estimate)
    fuse: Option<([f64; 3], [f64; 3])>,
    /// (joint rates, base linear velocity, base angular velocity)
    filt: Option<(Vec<Option<f64>>, [f64; 3], [f64; 3])>,
    filt_v: Option<[f64; 3]>,
}

impl Estimator {
    pub fn new(cfg: EstimatorConfig) -> Self {
        Self { cfg, fuse: None, filt: None, filt_v: None }
    }

    /// One tick. `accel_w` is the accelerometer's world-frame base
    /// acceleration [m/s^2] since the last tick; the fusion needs it on
    /// every tick but the first (which only seeds the fused velocity with
    /// the reading). Without it the fusion skips its prediction and only
    /// corrects.
    pub fn update(&mut self, mut r: StateEstimate, accel_w: Option<[f64; 3]>) -> StateEstimate {
        let c = &self.cfg;
        if c.vel_fuse_hz > 0.0 {
            let dt = c.dt;
            let tau = 1.0 / (2.0 * std::f64::consts::PI * c.vel_fuse_hz);
            let k = dt / (tau + dt);
            match self.fuse.as_mut() {
                Some((est, bias)) => {
                    for i in 0..3 {
                        if let Some(acc) = accel_w {
                            est[i] += (acc[i] - bias[i]) * dt;
                        }
                        let innov = r.base_lin_vel_w[i] - est[i];
                        est[i] += k * innov;
                        if c.vel_fuse_bias_t > 0.0 {
                            bias[i] -= innov * dt / (tau * c.vel_fuse_bias_t);
                        }
                    }
                    r.base_lin_vel_w = *est;
                }
                None => self.fuse = Some((r.base_lin_vel_w, [0.0; 3])),
            }
        }
        if c.filter_hz > 0.0 {
            let tau = 1.0 / (2.0 * std::f64::consts::PI * c.filter_hz);
            let a = c.dt / (tau + c.dt);
            match self.filt.as_mut() {
                Some((qd, v, w)) => {
                    for (x, y) in qd.iter_mut().zip(r.joint_qd.iter()) {
                        if let (Some(xv), Some(yv)) = (x.as_mut(), y) {
                            *xv += a * (yv - *xv);
                        } else {
                            *x = *y;
                        }
                    }
                    for i in 0..3 {
                        v[i] += a * (r.base_lin_vel_w[i] - v[i]);
                        w[i] += a * (r.base_ang_vel_w[i] - w[i]);
                    }
                    r.joint_qd = qd.clone();
                    r.base_lin_vel_w = *v;
                    r.base_ang_vel_w = *w;
                }
                None => self.filt = Some((r.joint_qd.clone(), r.base_lin_vel_w, r.base_ang_vel_w)),
            }
        }
        if c.filter_v_hz > 0.0 {
            let tau = 1.0 / (2.0 * std::f64::consts::PI * c.filter_v_hz);
            let a = c.dt / (tau + c.dt);
            match self.filt_v.as_mut() {
                Some(f) => {
                    for i in 0..3 {
                        f[i] += a * (r.base_lin_vel_w[i] - f[i]);
                    }
                    r.base_lin_vel_w = *f;
                }
                None => self.filt_v = Some(r.base_lin_vel_w),
            }
        }
        let qd_clamp = c.qd_clamp;
        r.joint_qd = r.joint_qd.into_iter().map(|qd| qd.map(|x| x.clamp(-qd_clamp, qd_clamp))).collect();
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    fn reading(v: [f64; 3], qd: f64) -> StateEstimate {
        StateEstimate {
            base_pos: [0.0; 3],
            base_quat: na::UnitQuaternion::identity(),
            base_lin_vel_w: v,
            base_ang_vel_w: [0.0; 3],
            joint_q: vec![0.0; 2],
            joint_qd: vec![Some(qd), None],
        }
    }

    #[test]
    fn off_passes_through_but_clamps() {
        let mut e = Estimator::new(EstimatorConfig { qd_clamp: 5.0, ..Default::default() });
        let out = e.update(reading([0.1, 0.2, 0.3], 9.0), None);
        assert_eq!(out.base_lin_vel_w, [0.1, 0.2, 0.3]);
        assert_eq!(out.joint_qd, vec![Some(5.0), None]);
    }

    #[test]
    fn fusion_follows_the_accelerometer_through_a_stuck_reading() {
        // The velocity sensor sticks at 0 while the base accelerates at 1 m/s^2:
        // the fused estimate rises with the accelerometer, pulled back below
        // the crossover.
        let cfg = EstimatorConfig { vel_fuse_hz: 1.0, ..Default::default() };
        let mut e = Estimator::new(cfg);
        e.update(reading([0.0; 3], 0.0), None);
        let mut v = 0.0;
        for _ in 0..20 {
            v = e.update(reading([0.0; 3], 0.0), Some([1.0, 0.0, 0.0])).base_lin_vel_w[0];
        }
        assert!(v > 0.05 && v < 0.1, "{v}");
    }

    #[test]
    fn low_pass_converges() {
        let mut e = Estimator::new(EstimatorConfig { filter_hz: 10.0, ..Default::default() });
        e.update(reading([0.0; 3], 0.0), None);
        let mut out = reading([0.0; 3], 0.0);
        for _ in 0..400 {
            out = e.update(reading([1.0, 0.0, 0.0], 2.0), None);
        }
        assert!((out.base_lin_vel_w[0] - 1.0).abs() < 1e-6);
        assert!((out.joint_qd[0].unwrap() - 2.0).abs() < 1e-6);
    }
}
