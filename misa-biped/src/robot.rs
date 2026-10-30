//! The robot model the walker computes with, the state it computes from an
//! estimate, and the observation it is given each tick.
//!
//! Moved here from articara's `biped::rig` on 2026-10-29 (S10 of
//! doc/from_claude/20261029_walker_extraction_plan.md). `BipedRig::sync`
//! read the plant, passed it through the sensor model, and then -- with no
//! further use of MuJoCo -- built `q`/`v`, the mass matrix with the rotor
//! inertia, `h` with the joint damping, the FK and the centroidal Jacobian.
//! That second half is [`RobotModel::assemble`]; the rig keeps the plant and
//! the sensor model (`BipedRig::sense`, which hands out a [`StateEstimate`]),
//! and since S14 the walker calls `assemble` itself on the estimate it is
//! given.
//!
//! The arithmetic is the rig's, in the rig's order (the joint loops, the
//! armature added to `M` and the damping to `h` per joint, the CoM sums per
//! mass link): the walker's results are bit-identical across the move.
use crate::profile::Profile;
use nalgebra as na;
use std::collections::HashMap;

/// A link that carries mass, paired with its misarta index and the CoM
/// offset in the link frame -- everything `J_com` needs.
#[derive(Clone, Debug)]
pub struct MassLink {
    pub mi: usize,
    pub m: f64,
    pub com_local: na::Vector3<f64>,
}

/// The rigid-body model and the joint maps the walker computes with. Built
/// once by the harness (`BipedRig::walker_model`) and not changed after.
///
/// Joints are indexed two ways: the harness's joint order (`ji`, the order
/// of `joint_names`, which `StateEstimate` and the plant use) and misarta's
/// (`mi` joints, `v` rows). `a2m` maps the first to the second.
#[derive(Clone, Debug)]
pub struct RobotModel {
    pub model: misarta::model::Model<f64>,
    pub prof: Profile,
    /// The harness's joints, by name, and the name -> index map.
    pub joint_names: Vec<String>,
    pub joint_map: HashMap<String, usize>,
    /// Harness joint index -> misarta joint index.
    pub a2m: Vec<Option<usize>>,
    /// misarta indices of [left, right] foot links.
    pub foot_mi: [usize; 2],
    /// Body the trunk task holds upright, and whether it IS the FreeFlyer.
    pub trunk_mi: usize,
    pub trunk_from_base: bool,
    pub mass_links: Vec<MassLink>,
    pub total_mass: f64,
    /// Per harness joint: the reflected rotor inertia and the viscous
    /// damping the plant carries, which misarta's model does not (added to
    /// `M` and `h` in [`RobotModel::assemble`]).
    pub armature: Vec<f64>,
    pub joint_damping: Vec<f64>,
    /// Per actuated row (misarta v-index - 6): torque, position and velocity
    /// limits.
    pub torque_max: na::DVector<f64>,
    pub q_min: na::DVector<f64>,
    pub q_max: na::DVector<f64>,
    pub v_max: na::DVector<f64>,
    /// Joint angles (harness order) the posture task and the degraded-solve
    /// fallback hold.
    pub q_seed: Vec<f64>,
    pub nv: usize,
    pub na: usize,
}

/// What the state estimator hands the walker: the floating base and the
/// joints, in world axes and the harness's joint order. Joint rates the
/// plant does not report are `None`. The joint rates arrive already
/// saturated (articara's rig applies `QD_CLAMP` in `sense`); the walker
/// uses them as they are.
#[derive(Clone, Debug)]
pub struct StateEstimate {
    pub base_pos: [f64; 3],
    pub base_quat: na::UnitQuaternion<f64>,
    pub base_lin_vel_w: [f64; 3],
    pub base_ang_vel_w: [f64; 3],
    pub joint_q: Vec<f64>,
    pub joint_qd: Vec<Option<f64>>,
}

/// One tick's worth of state in misarta's coordinates, plus the dynamics
/// quantities every task needs.
pub struct RobotState {
    pub q: Vec<f64>,
    pub v: Vec<f64>,
    pub v_dvec: na::DVector<f64>,
    pub data: misarta::data::Data<f64>,
    /// Mass matrix WITH reflected rotor inertia on the actuated diagonal.
    pub mass: na::DMatrix<f64>,
    /// Nonlinear effects WITH viscous joint damping.
    pub h: na::DVector<f64>,
    pub com: na::Vector3<f64>,
    pub com_vel: na::Vector3<f64>,
    pub j_com: na::DMatrix<f64>,
    pub djv_com: na::Vector3<f64>,
    pub body_pos: [f64; 3],
    pub body_quat: na::UnitQuaternion<f64>,
    pub v_ang_w: [f64; 3],
    /// The centroidal momentum matrix `A_G(q)`, computed on first use
    /// ([`RobotModel::cmm`]) and kept with the state, so the walker (the
    /// MPC's `h`, the momentum task, MOM_CMP) and the harness's measurements
    /// read one computation per tick (S15).
    cmm: std::cell::OnceCell<na::DMatrix<f64>>,
}

/// What the walker is given each tick.
pub struct Observation<'a> {
    /// The tick; the walker's time is `tick as f64 * dt`.
    pub tick: usize,
    /// The state estimate; the walker assembles `M`, `h`, the FK and the
    /// centroidal Jacobian from it ([`RobotModel::assemble`]) and hands the
    /// result back with the command (S14; before, the harness assembled
    /// it).
    pub estimate: &'a StateEstimate,
    /// Vertical contact force per foot [N] (the only part of the measured
    /// contacts the controller reads).
    pub contact_fz: [f64; 2],
    /// LEAK (§6.6 of the extraction plan): the PLANT's joint angles
    /// (harness order), with none of the sensor model's delay or noise.
    /// Read by the posture and arm-hold levels, as before the split; a leak
    /// of the truth into the controller, kept visible here.
    pub joint_pos_plant: &'a [f64],
}

impl RobotModel {
    /// The estimate in misarta's coordinates: `q` with the base pose, `v`
    /// with the base twist in the BODY frame ([angular; linear], misarta's
    /// FreeFlyer convention) and the joint rates (already saturated by the
    /// estimator).
    pub fn configuration(&self, est: &StateEstimate) -> (Vec<f64>, Vec<f64>) {
        let model = &self.model;
        let body_pos = est.base_pos;
        let body_quat = est.base_quat;
        let v_lin_w = est.base_lin_vel_w;
        let v_ang_w = est.base_ang_vel_w;
        let r_wb = body_quat.to_rotation_matrix();
        let r_bw = r_wb.transpose();
        let v_lin_body = r_bw * na::Vector3::new(v_lin_w[0], v_lin_w[1], v_lin_w[2]);
        let v_ang_body = r_bw * na::Vector3::new(v_ang_w[0], v_ang_w[1], v_ang_w[2]);

        let mut q = model.neutral_q();
        q[0] = body_pos[0];
        q[1] = body_pos[1];
        q[2] = body_pos[2];
        q[3] = body_quat.i;
        q[4] = body_quat.j;
        q[5] = body_quat.k;
        q[6] = body_quat.w;
        let mut v = vec![0.0_f64; self.nv];
        // FreeFlyer motion subspace is I6 in the BODY frame with row order
        // [angular; linear] (misarta joint.rs), so v[0..3] is omega_body and
        // v[3..6] is v_body.
        v[0] = v_ang_body.x;
        v[1] = v_ang_body.y;
        v[2] = v_ang_body.z;
        v[3] = v_lin_body.x;
        v[4] = v_lin_body.y;
        v[5] = v_lin_body.z;
        for ji in 0..self.joint_names.len() {
            let Some(mi) = self.a2m[ji] else { continue };
            if model.joints[mi].joint_type.nq() == 1 {
                q[model.q_idx[mi]] = est.joint_q[ji];
            }
            if model.joints[mi].joint_type.nv() == 1 {
                if let Some(qd) = est.joint_qd[ji] {
                    v[model.v_idx[mi]] = qd;
                }
            }
        }
        (q, v)
    }

    /// The dynamics at `(q, v)`: `M` (with armature), `h` (with joint
    /// damping), FK, and the centroidal Jacobian with its bias. The base
    /// pose and angular velocity of `est` are carried into the state as they
    /// are.
    pub fn state(&self, q: Vec<f64>, v: Vec<f64>, est: &StateEstimate) -> RobotState {
        let model = &self.model;
        let nv = self.nv;
        let v_dvec = na::DVector::from_column_slice(&v);

        let mut mass = misarta::crba::crba(model, &q);
        let mut h = misarta::rnea::nonlinear_effects(model, &q, &v);
        // misarta's dynamics Model carries no rotor inertia and no joint
        // damping -- `armature`/`joint_damping` are MJCF export fields and
        // reach the plant only. So the WBC has been solving a model that
        // differs from the simulator on EVERY actuated joint. Reflected rotor
        // inertia is a diagonal add to M; viscous damping is a velocity term
        // in h. Add both at this boundary so the two agree.
        for ji in 0..self.joint_names.len() {
            let Some(mi) = self.a2m[ji] else { continue };
            if model.joints[mi].joint_type.nv() != 1 {
                continue;
            }
            let vi = model.v_idx[mi];
            if vi < 6 {
                continue;
            }
            mass[(vi, vi)] += self.armature[ji];
            // Per-joint: the plant carries each joint's own damping (uniform
            // value plus any name-matched override), so the compensation
            // must read the same number or the two models diverge exactly
            // on the joints being designed.
            h[vi] += self.joint_damping[ji] * v[vi];
        }
        let data = misarta::fk::forward_kinematics(model, &q);

        // ---- CoM position, Jacobian and bias, all world-frame ----------
        // Per link: take its parent joint's world Jacobian (rows 0..3 =
        // angular, 3..6 = linear, at the JOINT origin) and shift the linear
        // part out to the link's own CoM,
        //     v_c = v_o + omega x r      =>  J_lin_c = J_lin_o - [r]x J_ang
        // then mass-average. The bias picks up the centripetal term:
        //     dJv_lin_c = dJv_lin_o - [r]x dJv_ang + omega x (omega x r)
        let com = self.com_of(&data);
        let mut j_com = na::DMatrix::zeros(3, nv);
        let mut djv_com = na::Vector3::zeros();
        for l in &self.mass_links {
            let rot = misarta::se3::rotation_matrix(&data.oMi[l.mi]);
            let r = rot * l.com_local;
            let skew = na::Matrix3::new(0.0, -r.z, r.y, r.z, 0.0, -r.x, -r.y, r.x, 0.0);
            let j = misarta::jacobian::compute_joint_jacobian_from_data(model, &q, &data, l.mi);
            let j_ang = j.rows(0, 3).into_owned();
            let j_lin = j.rows(3, 3).into_owned();
            let j_lin_c = &j_lin - &skew * &j_ang;
            j_com += l.m * j_lin_c;

            let dj = misarta::jacobian::compute_joint_jacobian_time_derivative(model, &q, &v, l.mi);
            let djv = &dj * &v_dvec;
            let djv_ang = na::Vector3::new(djv[0], djv[1], djv[2]);
            let djv_lin = na::Vector3::new(djv[3], djv[4], djv[5]);
            let omega = &j_ang * &v_dvec;
            let omega = na::Vector3::new(omega[0], omega[1], omega[2]);
            djv_com += l.m * (djv_lin - skew * djv_ang + omega.cross(&omega.cross(&r)));
        }
        j_com /= self.total_mass;
        djv_com /= self.total_mass;

        let com_vel = &j_com * &v_dvec;
        let com_vel = na::Vector3::new(com_vel[0], com_vel[1], com_vel[2]);

        RobotState {
            q,
            v,
            v_dvec,
            data,
            mass,
            h,
            com,
            com_vel,
            j_com,
            djv_com,
            body_pos: est.base_pos,
            body_quat: est.base_quat,
            v_ang_w: est.base_ang_vel_w,
            cmm: std::cell::OnceCell::new(),
        }
    }

    /// The centroidal momentum matrix `A_G(q)` of `st` (`h = A_G v`, rows
    /// 0..3 angular about the CoM, 3..6 linear), computed on the first call
    /// for this state and then kept with it. A pure function of the model
    /// and `st.q`, so who asks first does not change the value.
    pub fn cmm<'s>(&self, st: &'s RobotState) -> &'s na::DMatrix<f64> {
        st.cmm.get_or_init(|| misarta::centroidal::compute_centroidal_momentum_matrix(&self.model, &st.q))
    }

    /// [`RobotModel::configuration`] then [`RobotModel::state`].
    pub fn assemble(&self, est: &StateEstimate) -> RobotState {
        let (q, v) = self.configuration(est);
        self.state(q, v, est)
    }

    /// World-frame CoM position from an FK snapshot.
    pub fn com_of(&self, data: &misarta::data::Data<f64>) -> na::Vector3<f64> {
        let mut c = na::Vector3::zeros();
        for l in &self.mass_links {
            let r = misarta::se3::rotation_matrix(&data.oMi[l.mi]);
            let o = misarta::se3::translation(&data.oMi[l.mi]);
            c += l.m * (o + r * l.com_local);
        }
        c / self.total_mass
    }

    /// Gravity-compensation torque on the actuated rows, at `q`.
    pub fn gravity_torque(&self, q: &[f64]) -> na::DVector<f64> {
        let g_full = misarta::rnea::compute_gravity(&self.model, q);
        let mut tau_gravity = na::DVector::zeros(self.na);
        for ji in 0..self.joint_names.len() {
            let Some(mi) = self.a2m[ji] else { continue };
            if self.model.joints[mi].joint_type.nv() != 1 {
                continue;
            }
            let vi = self.model.v_idx[mi];
            if vi >= 6 {
                tau_gravity[vi - 6] = g_full[vi];
            }
        }
        tau_gravity
    }

    /// `(harness joint index, actuated row index)` over the joints that map
    /// to a single-DoF actuated misarta row. Every loop in the controller
    /// that touches tau, qddot or the posture task walks exactly this set.
    pub fn actuated(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(self.na);
        for ji in 0..self.joint_names.len() {
            let Some(mi) = self.a2m[ji] else { continue };
            if self.model.joints[mi].joint_type.nv() != 1 {
                continue;
            }
            let vi = self.model.v_idx[mi];
            if vi >= 6 {
                out.push((ji, vi));
            }
        }
        out
    }

    /// 0 = left, 1 = right.
    pub fn side_of(&self, foot_mi: usize) -> usize {
        usize::from(foot_mi != self.foot_mi[0])
    }
}
