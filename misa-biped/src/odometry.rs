//! Leg odometry: the floating base's position and linear velocity from the
//! IMU and the stance feet's kinematics, for a robot that has no sensor of
//! either (every real one).
//!
//! A linear Kalman filter in world axes (Bloesch et al. 2012, "State
//! estimation for legged robots"; the linear form of the MIT Cheetah 3
//! estimator, Bledt et al. 2018). The base orientation and angular velocity
//! are taken as the IMU's attitude filter reports them; they make every
//! relation below linear.
//!
//! State `x = [p, v, f_L, f_R, b]` (15): base position and velocity, the two
//! feet's (foot link origins') positions, and the accelerometer's bias in
//! world axes (a random walk; observable through the feet's velocity rows,
//! it would otherwise leave a steady velocity error).
//!
//! - Predict with the accelerometer: `a = R f_b + g - b` (specific force in
//!   the base frame, rotated, gravity added, bias removed), `p += v dt + a dt^2 / 2`,
//!   `v += a dt`. A foot in contact stays where it is (small process noise);
//!   a foot in the air is free (its noise is divided by the contact weight).
//! - Correct with each foot's kinematics: `f_i - p = R r_i(q)` and, for a
//!   foot in contact, `-v = R (w_b x r_i + J_i(q) qd)`. The measurement
//!   noise of both is divided by the contact weight, so a swing foot's rows
//!   carry nothing about the base and only drag its own state along.
//! - Optionally (`foot_z_sigma` > 0) a foot in contact is at the ground
//!   height it had at the start, which keeps the vertical position from
//!   drifting step by step (flat ground only).
//!
//! With `tilt_cal_t` > 0 the IMU's roll and pitch are corrected by the stance
//! feet (flat ground only): a sole in full contact is level, so the tilt the
//! kinematics give it is the attitude's error (plus the leg's encoder
//! offsets, which is what the controller needs corrected anyway). The
//! correction integrates that tilt with this time constant, from feet in
//! full contact whose tilt is under `tilt_gate` (a heel or toe roll is not
//! taken).
//!
//! The gyro's bias is taken as its mean over the first `gyro_cal_s` seconds
//! (the robot standing still at start-up) and removed afterwards.
//!
//! The contact weight `c_i` in [0, 1] is a smooth step of the foot's normal
//! force between `fz_lo` and `fz_hi` (fractions of the body weight).
use crate::robot::{RobotModel, StateEstimate};
use nalgebra as na;

const N: usize = 15;
type V12 = na::SVector<f64, N>;
type M12 = na::SMatrix<f64, N, N>;

/// Leg-odometry settings (all standard deviations per sample).
#[derive(Clone, Debug, PartialEq)]
pub struct LegOdometryConfig {
    /// Filter sample time [s] (the control period).
    pub dt: f64,
    /// Accelerometer noise [m/s^2] (process noise of the velocity).
    pub sigma_acc: f64,
    /// Foot drift while in contact [m/s] (process noise of a stance foot).
    pub sigma_foot: f64,
    /// Kinematic foot position [m] (encoder noise, link-length error).
    pub sigma_kin_pos: f64,
    /// Kinematic foot velocity [m/s] (joint-rate noise, slip).
    pub sigma_kin_vel: f64,
    /// Ground height of a foot in contact [m] (0 = no height row).
    pub foot_z_sigma: f64,
    /// Contact weight 0 below / 1 above this normal force, as a fraction of
    /// the body weight.
    pub fz_lo: f64,
    pub fz_hi: f64,
    /// Accelerometer bias: initial standard deviation [m/s^2] and random
    /// walk [m/s^2 per sqrt(s)] (initial 0 = no bias state).
    pub acc_bias0: f64,
    pub acc_bias_walk: f64,
    /// Gyro-bias calibration window at start-up [s] (0 = none).
    pub gyro_cal_s: f64,
    /// First-order low-pass on the joint rates handed to the walker [Hz]
    /// (0 = off; the filter's own velocity rows use the raw rates).
    pub qd_lp_hz: f64,
    /// Roll / pitch correction from the stance feet: time constant [s]
    /// (0 = off) and the largest foot tilt taken [rad].
    pub tilt_cal_t: f64,
    pub tilt_gate: f64,
}

impl Default for LegOdometryConfig {
    fn default() -> Self {
        Self {
            dt: 0.002,
            sigma_acc: 0.5,
            sigma_foot: 0.002,
            sigma_kin_pos: 0.003,
            sigma_kin_vel: 0.05,
            foot_z_sigma: 0.0,
            fz_lo: 0.05,
            fz_hi: 0.2,
            acc_bias0: 0.0,
            acc_bias_walk: 0.01,
            gyro_cal_s: 0.0,
            qd_lp_hz: 0.0,
            tilt_cal_t: 0.0,
            tilt_gate: 0.03,
        }
    }
}

impl LegOdometryConfig {
    /// `ODOM_*` knobs by name over the defaults (`dt` given).
    pub fn from_kv(dt: f64, get: &dyn Fn(&str) -> Option<String>) -> Self {
        let d = Self { dt, ..Self::default() };
        let num = |k: &str, v: f64| get(k).and_then(|s| s.parse().ok()).unwrap_or(v);
        Self {
            dt,
            sigma_acc: num("ODOM_SIGMA_ACC", d.sigma_acc),
            sigma_foot: num("ODOM_SIGMA_FOOT", d.sigma_foot),
            sigma_kin_pos: num("ODOM_SIGMA_KIN_POS", d.sigma_kin_pos),
            sigma_kin_vel: num("ODOM_SIGMA_KIN_VEL", d.sigma_kin_vel),
            foot_z_sigma: num("ODOM_FOOT_Z_SIGMA", d.foot_z_sigma),
            fz_lo: num("ODOM_FZ_LO", d.fz_lo),
            fz_hi: num("ODOM_FZ_HI", d.fz_hi),
            acc_bias0: num("ODOM_ACC_BIAS0", d.acc_bias0),
            acc_bias_walk: num("ODOM_ACC_BIAS_WALK", d.acc_bias_walk),
            gyro_cal_s: num("ODOM_GYRO_CAL_S", d.gyro_cal_s),
            qd_lp_hz: num("ODOM_QD_LP_HZ", d.qd_lp_hz),
            tilt_cal_t: num("ODOM_TILT_CAL_T", d.tilt_cal_t),
            tilt_gate: num("ODOM_TILT_GATE", d.tilt_gate),
        }
    }
}

/// One tick's sensor reading for [`LegOdometry::update`].
#[derive(Clone, Debug)]
pub struct ImuLegReading {
    /// The IMU attitude filter's base orientation.
    pub base_quat: na::UnitQuaternion<f64>,
    /// Gyro, base frame [rad/s].
    pub gyro_b: [f64; 3],
    /// Accelerometer specific force, base frame [m/s^2] (`a - g`).
    pub acc_b: [f64; 3],
    /// Joint angles and rates, harness order.
    pub joint_q: Vec<f64>,
    pub joint_qd: Vec<Option<f64>>,
    /// Normal force of each foot [N].
    pub contact_fz: [f64; 2],
}

/// The filter's state between ticks (see the module docs).
#[derive(Clone, Debug)]
pub struct LegOdometry {
    pub cfg: LegOdometryConfig,
    x: V12,
    p: M12,
    ground_z: [f64; 2],
    body_weight: f64,
    /// (sum of gyro readings, count) during the calibration window, then the
    /// bias.
    gyro_sum: ([f64; 3], usize),
    gyro_bias: [f64; 3],
    tick: usize,
    qd_lp: Option<Vec<Option<f64>>>,
    /// Roll / pitch correction (world rotation vector, z = 0).
    tilt: na::Vector3<f64>,
}

const G: f64 = 9.81;

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

impl LegOdometry {
    /// Start at a known base position and velocity (the start-up pose); the
    /// feet from the kinematics there.
    pub fn new(cfg: LegOdometryConfig, robot: &RobotModel, est0: &StateEstimate) -> Self {
        let r = Self::feet_rel(robot, &est0.base_quat, &est0.joint_q, None).0;
        let mut x = V12::zeros();
        for i in 0..3 {
            x[i] = est0.base_pos[i];
            x[3 + i] = est0.base_lin_vel_w[i];
            x[6 + i] = est0.base_pos[i] + r[0][i];
            x[9 + i] = est0.base_pos[i] + r[1][i];
        }
        let mut p = M12::zeros();
        for i in 0..12 {
            p[(i, i)] = 1e-6;
        }
        for i in 12..15 {
            p[(i, i)] = cfg.acc_bias0.powi(2).max(1e-12);
        }
        Self {
            cfg,
            x,
            p,
            ground_z: [x[8], x[11]],
            body_weight: robot.total_mass * G,
            gyro_sum: ([0.0; 3], 0),
            gyro_bias: [0.0; 3],
            tick: 0,
            qd_lp: None,
            tilt: na::Vector3::zeros(),
        }
    }

    /// The feet's positions relative to the base in world axes, `R r_i(q)`,
    /// and (with the rates) their velocities relative to the base,
    /// `R (w_b x r_i + J_i qd)` (the joint part by a forward difference).
    fn feet_rel(
        robot: &RobotModel,
        quat: &na::UnitQuaternion<f64>,
        q: &[f64],
        rates: Option<(&[Option<f64>], &na::Vector3<f64>)>,
    ) -> ([na::Vector3<f64>; 2], [na::Vector3<f64>; 2], [na::Matrix3<f64>; 2]) {
        let body = |q: &[f64]| {
            let e = StateEstimate {
                base_pos: [0.0; 3],
                base_quat: na::UnitQuaternion::identity(),
                base_lin_vel_w: [0.0; 3],
                base_ang_vel_w: [0.0; 3],
                joint_q: q.to_vec(),
                joint_qd: vec![None; q.len()],
            };
            let data = robot.fk_of(&e);
            robot.foot_mi.map(|mi| (misarta::se3::translation(&data.oMi[mi]), misarta::se3::rotation_matrix(&data.oMi[mi])))
        };
        let fk = body(q);
        let rb = fk.map(|f| f.0);
        let rot_b = fk.map(|f| f.1);
        let rot = quat.to_rotation_matrix();
        let pos = rb.map(|r| rot * r);
        let vel = match rates {
            Some((qd, w_b)) => {
                let eps = 1e-3;
                let q2: Vec<f64> = q.iter().zip(qd).map(|(x, d)| x + eps * d.unwrap_or(0.0)).collect();
                let rb2 = body(&q2).map(|f| f.0);
                [0, 1].map(|i| rot * (w_b.cross(&rb[i]) + (rb2[i] - rb[i]) / eps))
            }
            None => [na::Vector3::zeros(); 2],
        };
        (pos, vel, rot_b)
    }

    /// The contact weight of each foot from its normal force.
    pub fn contact_weight(&self, fz: [f64; 2]) -> [f64; 2] {
        let (lo, hi) = (self.cfg.fz_lo * self.body_weight, self.cfg.fz_hi * self.body_weight);
        fz.map(|f| smoothstep((f - lo) / (hi - lo).max(1e-9)))
    }

    /// The roll / pitch correction from the stance feet (world rotation
    /// vector) [rad].
    pub fn tilt_correction(&self) -> [f64; 3] {
        [self.tilt.x, self.tilt.y, self.tilt.z]
    }

    /// Current estimate: base position, base velocity, feet (world).
    pub fn state(&self) -> ([f64; 3], [f64; 3], [[f64; 3]; 2]) {
        let g = |i: usize| [self.x[i], self.x[i + 1], self.x[i + 2]];
        (g(0), g(3), [g(6), g(9)])
    }

    /// One tick: the reading -> the walker's state estimate (orientation and
    /// angular velocity from the IMU, position and velocity from the filter,
    /// joints as read).
    pub fn update(&mut self, robot: &RobotModel, r: &ImuLegReading) -> StateEstimate {
        let c = self.cfg.clone();
        let dt = c.dt;
        let quat = na::UnitQuaternion::from_scaled_axis(self.tilt) * r.base_quat;
        let rot = quat.to_rotation_matrix();
        // gyro bias: averaged while standing at start-up, then removed
        if (self.tick as f64) * dt < c.gyro_cal_s {
            for i in 0..3 {
                self.gyro_sum.0[i] += r.gyro_b[i];
            }
            self.gyro_sum.1 += 1;
            let n = self.gyro_sum.1 as f64;
            self.gyro_bias = self.gyro_sum.0.map(|s| s / n);
        }
        self.tick += 1;
        let w_b = na::Vector3::from(r.gyro_b) - na::Vector3::from(self.gyro_bias);
        let a_w = rot * na::Vector3::from(r.acc_b) + na::Vector3::new(0.0, 0.0, -G);
        let cw = self.contact_weight(r.contact_fz);
        let wmin = 1e-3;

        // predict
        let mut f = M12::identity();
        for i in 0..3 {
            f[(i, 3 + i)] = dt;
            if c.acc_bias0 > 0.0 {
                f[(i, 12 + i)] = -0.5 * dt * dt;
                f[(3 + i, 12 + i)] = -dt;
            }
        }
        let mut b = V12::zeros();
        for i in 0..3 {
            b[i] = 0.5 * dt * dt * a_w[i];
            b[3 + i] = dt * a_w[i];
        }
        self.x = f * self.x + b;
        let mut q = M12::zeros();
        let qa = (c.sigma_acc * dt).powi(2);
        for i in 0..3 {
            q[(i, i)] = qa * dt * dt / 4.0;
            q[(i, 3 + i)] = qa * dt / 2.0;
            q[(3 + i, i)] = qa * dt / 2.0;
            q[(3 + i, 3 + i)] = qa;
            for k in 0..2 {
                // a foot in the air moves freely (1 m/s of process noise)
                let s = c.sigma_foot * dt / cw[k].max(wmin);
                q[(6 + 3 * k + i, 6 + 3 * k + i)] = s.min(1.0 * dt).powi(2);
            }
            q[(12 + i, 12 + i)] = if c.acc_bias0 > 0.0 { c.acc_bias_walk.powi(2) * dt } else { 0.0 };
        }
        self.p = f * self.p * f.transpose() + q;

        // correct: per foot, 3 position rows, 3 velocity rows, 1 height row
        let (pos, vel, foot_rot_b) = Self::feet_rel(robot, &quat, &r.joint_q, Some((&r.joint_qd, &w_b)));
        if c.tilt_cal_t > 0.0 {
            // a level sole's normal is vertical: the rotation that makes it so
            for k in 0..2 {
                let z = rot * foot_rot_b[k].column(2);
                let phi = na::Vector3::new(z.y, -z.x, 0.0);
                if cw[k] > 0.9 && phi.norm() < c.tilt_gate {
                    self.tilt += phi * (cw[k] * dt / c.tilt_cal_t);
                }
            }
        }
        let mut rows: Vec<([f64; N], f64, f64)> = Vec::with_capacity(14);
        for k in 0..2 {
            let w = cw[k].max(wmin);
            for i in 0..3 {
                let mut h = [0.0; N];
                h[6 + 3 * k + i] = 1.0;
                h[i] = -1.0;
                rows.push((h, pos[k][i], (c.sigma_kin_pos).powi(2) / w));
            }
            if cw[k] > 0.01 {
                for i in 0..3 {
                    let mut h = [0.0; N];
                    h[3 + i] = 1.0;
                    rows.push((h, -vel[k][i], c.sigma_kin_vel.powi(2) / cw[k]));
                }
                if c.foot_z_sigma > 0.0 {
                    let mut h = [0.0; N];
                    h[6 + 3 * k + 2] = 1.0;
                    rows.push((h, self.ground_z[k], c.foot_z_sigma.powi(2) / cw[k]));
                }
            }
        }
        // sequential scalar updates (the rows' noises are independent)
        for (h, y, rvar) in rows {
            let hv = na::SVector::<f64, N>::from(h);
            let ph = self.p * hv;
            let s = hv.dot(&ph) + rvar;
            let k = ph / s;
            let innov = y - hv.dot(&self.x);
            self.x += k * innov;
            self.p -= k * ph.transpose();
        }
        self.p = 0.5 * (self.p + self.p.transpose());

        let w_w = rot * w_b;
        let joint_qd = if c.qd_lp_hz > 0.0 {
            let a = dt / (1.0 / (2.0 * std::f64::consts::PI * c.qd_lp_hz) + dt);
            let f = self.qd_lp.get_or_insert_with(|| r.joint_qd.clone());
            for (x, y) in f.iter_mut().zip(&r.joint_qd) {
                *x = match (*x, y) {
                    (Some(xv), Some(yv)) => Some(xv + a * (yv - xv)),
                    _ => *y,
                };
            }
            f.clone()
        } else {
            r.joint_qd.clone()
        };
        StateEstimate {
            base_pos: [self.x[0], self.x[1], self.x[2]],
            base_quat: quat,
            base_lin_vel_w: [self.x[3], self.x[4], self.x[5]],
            base_ang_vel_w: [w_w.x, w_w.y, w_w.z],
            joint_q: r.joint_q.clone(),
            joint_qd,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothstep_is_a_weight() {
        assert_eq!(smoothstep(-1.0), 0.0);
        assert_eq!(smoothstep(2.0), 1.0);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-12);
    }
}
