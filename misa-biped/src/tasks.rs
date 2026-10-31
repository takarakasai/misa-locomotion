//! One builder per priority level of the biped's hierarchical QP.
//!
//! The level ORDER is the driver's business, not this module's -- it changes
//! between double and single support and between experiments, and hiding it
//! behind a fixed list is how a "level 4" tally came to mislabel every
//! single-support run by one. What lives here is how each level's matrices
//! and reference are built, so that two drivers (squat and walk) cannot drift
//! apart in the details.
//!
//! Moved from articara's `biped::tasks` on 2026-10-30 (S13d of
//! doc/from_claude/20261029_walker_extraction_plan.md), which re-exports it
//! under its old path. [`crate::task_stack::TaskStack`] is the walker's
//! driver of these builders.

use misa_wbc::{Affine, AsAffine, Dynamics, Task, tasks as wt};
use nalgebra as na;

use crate::contact::{Anchors, sole_selection};

/// P1: track a CoM acceleration reference.
pub fn com(
    qddot: &Affine,
    j_com: &na::DMatrix<f64>,
    djv_com: &na::Vector3<f64>,
    accel_ref: &na::Vector3<f64>,
) -> Task {
    wt::cartesian_acceleration(
        qddot,
        j_com,
        &na::DVector::from_vec(vec![djv_com.x, djv_com.y, djv_com.z]),
        &na::DVector::from_vec(vec![accel_ref.x, accel_ref.y, accel_ref.z]),
    )
}

/// P2: hold the trunk upright, via the WORLD-frame angular Jacobian.
///
/// Not `qddot[0..2]`: those are body-frame and carry a sign inversion. Rows
/// 0..2 of the trunk's own world Jacobian map qddot to world angular
/// acceleration, which is what matches a world-frame roll/pitch error.
pub fn trunk(
    qddot: &Affine,
    j_trunk: &na::DMatrix<f64>,
    djv_trunk: &na::DVector<f64>,
    rp_ref: &[f64; 2],
    nv: usize,
) -> Task {
    trunk_rpy(qddot, j_trunk, djv_trunk, rp_ref, None, nv)
}

/// [`trunk`] with an optional YAW row.
///
/// Roll and pitch alone leave the whole stack with nothing regulating
/// rotation about the vertical. Measured on kyo46rs stepping in place: the
/// swing leg twists (its own orientation being unconstrained), that injects
/// vertical-axis angular momentum, and the base counter-rotates +-8 deg to
/// conserve it while hip_yaw walks to its +-30 deg stop. Regulating the trunk's
/// yaw asks for the quantity that matters and lets the QP decide which joints
/// pay for it, rather than dictating a swing-foot angle.
pub fn trunk_rpy(
    qddot: &Affine,
    j_trunk: &na::DMatrix<f64>,
    djv_trunk: &na::DVector<f64>,
    rp_ref: &[f64; 2],
    yaw_ref: Option<f64>,
    nv: usize,
) -> Task {
    let rows = if yaw_ref.is_some() { 3 } else { 2 };
    let mut j_rp = na::DMatrix::zeros(rows, nv);
    for c in 0..nv {
        j_rp[(0, c)] = j_trunk[(0, c)];
        j_rp[(1, c)] = j_trunk[(1, c)];
        if yaw_ref.is_some() {
            j_rp[(2, c)] = j_trunk[(2, c)];
        }
    }
    let (mut dj, mut r) = (vec![djv_trunk[0], djv_trunk[1]], vec![rp_ref[0], rp_ref[1]]);
    if let Some(y) = yaw_ref {
        dj.push(djv_trunk[2]);
        r.push(y);
    }
    wt::cartesian_acceleration(
        qddot,
        &j_rp,
        &na::DVector::from_vec(dj),
        &na::DVector::from_vec(r),
    )
}

/// Roll/pitch reference acceleration for [`trunk`].
///
/// `dead` is a deadband on the POSITION error only; damping stays live, so
/// the task still resists rate inside the band instead of going open.
///
/// `sign` is +1, the textbook convention. A period when -1 was needed was
/// compensating for a `crba` bug in misarta, not a convention mismatch.
pub fn trunk_rp_ref(
    roll: f64,
    pitch: f64,
    omega_world: &[f64; 3],
    kp: f64,
    kd: f64,
    deadband: f64,
    sign: f64,
) -> [f64; 2] {
    let dead = |e: f64| {
        if e.abs() <= deadband {
            0.0
        } else {
            e - deadband * e.signum()
        }
    };
    [
        sign * (kp * (0.0 - dead(roll)) + kd * (0.0 - omega_world[0])),
        sign * (kp * (0.0 - dead(pitch)) + kd * (0.0 - omega_world[1])),
    ]
}

/// The trunk level's orientation reference and its gains live in
/// [`crate::wbc`] (a reference the walker computes); re-exported under their
/// old path.
pub use crate::wbc::{TrunkGains, trunk_ori_ref};

/// Centroidal ANGULAR momentum, driven toward zero: `ḣ_ang = -kp · h_ang`.
///
/// `cmm` is misarta's `A_G` with `h = [h_ang; h_lin]`, so the angular rows are
/// 0..3 -- the linear rows are deliberately not used, because `h_lin = m·ṗ_com`
/// is the CoM task's variable and asking two levels for the same quantity only
/// decides which one loses.
///
/// `axes` selects which of roll/pitch/yaw to regulate. Doc Sec.21.3 ruled this
/// task out before implementing it, on two grounds: the roll component was
/// architecturally zero (shoulder and elbow were both sagittal), and the `h_x`
/// seen while falling reached 1.10 kg·m²/s against roughly 0.3 available from
/// the arms. **The first ground died with the v6 shoulder roll; the second did
/// not** -- 1.10 is the value `h_x` grows TO during a fall, while the value at
/// single-support entry was 0.178, and that one is inside what the arms can
/// produce. Which of the two numbers this task has to beat is exactly what
/// was never measured.
/// `h_ref` is the angular momentum this task should TRACK, not suppress.
///
/// With `h_ref = 0` the task says "do not let angular momentum exist", which
/// is what it said before this parameter was added, and which locomotion
/// cannot satisfy: legs swinging produce angular momentum by construction, so
/// the solver spends the cheapest joints it has -- the arms -- cancelling
/// them. Measured on the free-arm G1 (doc/from_claude/20260920_bdi_task_structure.md):
/// enabling this task at all took arm travel from 1.8 deg to 170-295 deg
/// during a SQUAT, and sweeping `kp` over 60x did not change it, because the
/// reference, not the gain, is what forbids the motion.
///
/// Boston Dynamics describe the opposite: angular momentum is a quantity their
/// MPC plans, which is what lets Atlas "move its arms and legs to change its
/// inertia" -- tuck to rotate faster, spread to slow down. That behaviour is
/// only expressible if something asks for a particular h(t).
pub fn angular_momentum(
    qddot: &Affine,
    cmm: &na::DMatrix<f64>,
    dcmm_v: &na::DVector<f64>,
    h_now: &na::DVector<f64>,
    h_ref: &[f64; 3],
    // Planned rate of change, fed FORWARD. `-K(h - h_ref)` alone can only
    // correct an error that has already happened; a reference that moves needs
    // the plan's own `dh`, or the task lags it by construction. Codex raised
    // this against the h_ref-only form and they are right: their A2 reference
    // carries `h` and `dh` at 5701 nodes precisely so both terms can be
    // supplied. Zero here reproduces the P-only behaviour exactly.
    dh_ref: &[f64; 3],
    kp: f64,
    axes: [bool; 3],
    nv: usize,
) -> Option<Task> {
    let rows: Vec<usize> = (0..3).filter(|&i| axes[i]).collect();
    if rows.is_empty() {
        return None;
    }
    let mut j = na::DMatrix::zeros(rows.len(), nv);
    let mut bias = na::DVector::zeros(rows.len());
    let mut href = na::DVector::zeros(rows.len());
    for (r, &i) in rows.iter().enumerate() {
        j.row_mut(r).copy_from(&cmm.row(i));
        bias[r] = dcmm_v[i];
        href[r] = dh_ref[i] - kp * (h_now[i] - h_ref[i]);
    }
    Some(wt::cartesian_acceleration(qddot, &j, &bias, &href))
}

/// P4: weak posture, so the null space does not wander.
///
/// `actuated` is `(articara joint index, misarta v index)` as
/// [`crate::robot::RobotModel::actuated`] (and articara's
/// `BipedRig::actuated`) returns it.
pub fn posture(
    qddot: &Affine,
    actuated: &[(usize, usize)],
    q_now: &[f64],
    q_seed: &[f64],
    v: &[f64],
    kp: f64,
    kd: f64,
    na_count: usize,
    nv: usize,
) -> Task {
    let mut j_post = na::DMatrix::zeros(na_count, nv);
    let mut post_ref = na::DVector::zeros(na_count);
    for &(ji, vi) in actuated {
        j_post[(vi - 6, vi)] = 1.0;
        post_ref[vi - 6] = kp * (q_seed[ji] - q_now[ji]) + kd * (0.0 - v[vi]);
    }
    wt::cartesian_acceleration(qddot, &j_post, &na::DVector::zeros(na_count), &post_ref)
}

/// Joint position/velocity safety limits, as `misa_wbc::tasks::joint_limit_cbf`
/// on the actuated rows. Not in any level by default -- only the torque box
/// is enforced today, so a redundant leg can be commanded into a solution
/// that is only realisable by running a joint past its physical stop
/// (measured: ankle_roll pushed to 196% of its URDF limit while falling,
/// doc Sec.18.5). See `JLIM` in kyo46rs_walk.rs for the level this feeds.
///
/// `limits` (gains and `q_min`/`q_max`/`v_max`/`a_max`) is built once by the
/// caller from the URDF-derived bounds (articara's `BipedRig`). `q_act`/`v_act`
/// are this tick's LIVE actuated-row position/velocity, read through
/// `rig.model.q_idx`/`v_idx` off misarta's raw state.
///
/// The older wording here said `rig.robot.joint_positions` is "frozen at the
/// burn-in seed pose and never updated" -- **that is wrong, and it cost a
/// session's worth of chasing a bug that was not there.** `BipedRig::new`
/// writes it once at the end of the settle, but the plant step also writes it
/// every tick (`MujocoSim::step_n_frames` takes `&mut robot`), so it does
/// track the joint. Both readings are live; use whichever the surrounding
/// code already uses.
pub fn joint_limits(
    qddot: &Affine,
    actuated: &[(usize, usize)],
    q_act: &na::DVector<f64>,
    v_act: &na::DVector<f64>,
    limits: &wt::JointLimitCbf,
    na_count: usize,
    nv: usize,
) -> Task {
    let mut j_sel = na::DMatrix::zeros(na_count, nv);
    for &(_, vi) in actuated {
        j_sel[(vi - 6, vi)] = 1.0;
    }
    let qddot_act = &j_sel * qddot;
    wt::joint_limit_cbf(&qddot_act, q_act, v_act, limits)
}

/// Which rows of the swing foot's linear Jacobian to constrain.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SwingAxes {
    /// Clearance only. What single-leg STANCE wants: constraining x and y as
    /// well pins the swing foot to the world position it happened to occupy
    /// at t=0, on a robot that is deliberately translating its body sideways,
    /// and the reaction for holding it there lands on the stance leg.
    ZOnly,
    /// Full 3-D tracking. What STEPPING wants: the target is a planned
    /// footstep that moves, so there is nothing to fight.
    Xyz,
    /// Position plus YAW only, four rows.
    ///
    /// The middle ground, and usually the right one. Taking all six of the
    /// foot's DoF leaves the swing leg no null space at all -- with the
    /// stance contact already spending 6 rows and CoM + trunk another 5, a
    /// six-row swing task drops the robot in one or two steps at every gain
    /// tried. Yaw is the DoF that actually ran away.
    XyzYaw,
    /// Position plus roll and pitch, five rows. Stair climbing needs the sole
    /// kept flat, while leaving yaw free preserves one leg redundancy.
    XyzRollPitch,
    /// Position plus yaw and pitch, five rows: the walking default (XyzYaw)
    /// with the sole's pitch added, roll left free. For a known slope, where
    /// the landing needs the sole pitched to the surface but the push
    /// recovery was tuned with yaw held and roll free (measured 2026-10-20:
    /// XyzRollPitch on the +3 deg slope pushes cost the forward 11 N.s
    /// 44 -> 28 against no orientation tracking).
    XyzYawPitch,
    /// Position AND orientation, six rows.
    ///
    /// Leaving the swing foot's three ROTATIONAL DoF unconstrained is the same
    /// mistake as leaving its translation unconstrained, and it is less
    /// obvious because nothing flies through the air -- the leg simply twists.
    /// Measured on kyo46rs stepping in place with translation-only swing:
    /// hip_yaw walked from 0.04 deg to its +-30 deg STOP over 16 steps, the
    /// feet landed yawed by up to 31 deg, and the base yawed +-8 deg trying to
    /// conserve the angular momentum the twisting legs kept injecting. The QP
    /// was using the free leg's yaw as a null-space dumping ground.
    Pose,
}

/// P3: swing-foot tracking. `kp_xy` is separate from `kp_z` because the two
/// axes are doing different jobs -- z is clearance and must not scuff, x/y is
/// placement and can afford to be soft while the stance leg is the one paying
/// for it.
#[allow(clippy::too_many_arguments)]
pub fn swing(
    qddot: &Affine,
    j_foot: &na::DMatrix<f64>,
    djv: &na::DVector<f64>,
    pos: &na::Vector3<f64>,
    vel: &na::Vector3<f64>,
    target: &na::Vector3<f64>,
    target_vel: &na::Vector3<f64>,
    kp_xy: f64,
    kp_z: f64,
    kd: f64,
    axes: SwingAxes,
) -> Task {
    swing_with_pose(
        qddot, j_foot, djv, pos, vel, target, target_vel, kp_xy, kp_z, kd, axes, None,
    )
}

/// [`swing`] with an optional orientation target for [`SwingAxes::Pose`].
///
/// `orientation` is `(R_current, R_target, omega, kp, kd)`. The target should
/// come from the PLAN -- the orientation the foot is meant to land in -- not
/// from where the foot happens to be, or the error it is meant to remove
/// becomes the thing it tracks.
#[allow(clippy::too_many_arguments)]
pub fn swing_with_pose(
    qddot: &Affine,
    j_foot: &na::DMatrix<f64>,
    djv: &na::DVector<f64>,
    pos: &na::Vector3<f64>,
    vel: &na::Vector3<f64>,
    target: &na::Vector3<f64>,
    target_vel: &na::Vector3<f64>,
    kp_xy: f64,
    kp_z: f64,
    kd: f64,
    axes: SwingAxes,
    orientation: Option<(
        na::Matrix3<f64>,
        na::Matrix3<f64>,
        na::Vector3<f64>,
        f64,
        f64,
    )>,
) -> Task {
    swing_with_pose_acceleration(
        qddot,
        j_foot,
        djv,
        pos,
        vel,
        target,
        target_vel,
        kp_xy,
        kp_z,
        kd,
        axes,
        orientation,
        &na::Vector3::zeros(),
    )
}

/// Pose tracking with a planned linear acceleration feedforward, in world
/// coordinates. The acceleration is combined with feedback before building
/// misa-wbc's Cartesian acceleration task.
#[allow(clippy::too_many_arguments)]
pub fn swing_with_pose_acceleration(
    qddot: &Affine,
    j_foot: &na::DMatrix<f64>,
    djv: &na::DVector<f64>,
    pos: &na::Vector3<f64>,
    vel: &na::Vector3<f64>,
    target: &na::Vector3<f64>,
    target_vel: &na::Vector3<f64>,
    kp_xy: f64,
    kp_z: f64,
    kd: f64,
    axes: SwingAxes,
    orientation: Option<(
        na::Matrix3<f64>,
        na::Matrix3<f64>,
        na::Vector3<f64>,
        f64,
        f64,
    )>,
    target_acc: &na::Vector3<f64>,
) -> Task {
    let a = target_acc
        + na::Vector3::new(
            kp_xy * (target.x - pos.x) + kd * (target_vel.x - vel.x),
            kp_xy * (target.y - pos.y) + kd * (target_vel.y - vel.y),
            kp_z * (target.z - pos.z) + kd * (target_vel.z - vel.z),
        );
    match axes {
        SwingAxes::ZOnly => wt::cartesian_acceleration(
            qddot,
            &j_foot.rows(5, 1).into_owned(),
            &na::DVector::from_vec(vec![djv[5]]),
            &na::DVector::from_vec(vec![a.z]),
        ),
        SwingAxes::Xyz => wt::cartesian_acceleration(
            qddot,
            &j_foot.rows(3, 3).into_owned(),
            &na::DVector::from_vec(vec![djv[3], djv[4], djv[5]]),
            &na::DVector::from_vec(vec![a.x, a.y, a.z]),
        ),
        SwingAxes::XyzYaw => {
            let (rot, rot_tgt, omega, kp_r, kd_r) =
                orientation.expect("SwingAxes::XyzYaw needs an orientation target");
            let dr = rot_tgt * rot.transpose();
            let e_yaw = (dr[(1, 0)] - dr[(0, 1)]) * 0.5;
            let a_yaw = kp_r * e_yaw - kd_r * omega.z;
            let mut j = na::DMatrix::zeros(4, j_foot.ncols());
            for c in 0..j_foot.ncols() {
                j[(0, c)] = j_foot[(2, c)];
                for r in 0..3 {
                    j[(1 + r, c)] = j_foot[(3 + r, c)];
                }
            }
            wt::cartesian_acceleration(
                qddot,
                &j,
                &na::DVector::from_vec(vec![djv[2], djv[3], djv[4], djv[5]]),
                &na::DVector::from_vec(vec![a_yaw, a.x, a.y, a.z]),
            )
        }
        SwingAxes::XyzRollPitch => {
            let (rot, rot_tgt, omega, kp_r, kd_r) =
                orientation.expect("SwingAxes::XyzRollPitch needs an orientation target");
            let dr = rot_tgt * rot.transpose();
            let e_roll = (dr[(2, 1)] - dr[(1, 2)]) * 0.5;
            let e_pitch = (dr[(0, 2)] - dr[(2, 0)]) * 0.5;
            let a_roll = kp_r * e_roll - kd_r * omega.x;
            let a_pitch = kp_r * e_pitch - kd_r * omega.y;
            let mut j = na::DMatrix::zeros(5, j_foot.ncols());
            for c in 0..j_foot.ncols() {
                j[(0, c)] = j_foot[(0, c)];
                j[(1, c)] = j_foot[(1, c)];
                for r in 0..3 {
                    j[(2 + r, c)] = j_foot[(3 + r, c)];
                }
            }
            wt::cartesian_acceleration(
                qddot,
                &j,
                &na::DVector::from_vec(vec![djv[0], djv[1], djv[3], djv[4], djv[5]]),
                &na::DVector::from_vec(vec![a_roll, a_pitch, a.x, a.y, a.z]),
            )
        }
        SwingAxes::XyzYawPitch => {
            let (rot, rot_tgt, omega, kp_r, kd_r) =
                orientation.expect("SwingAxes::XyzYawPitch needs an orientation target");
            let dr = rot_tgt * rot.transpose();
            let e_pitch = (dr[(0, 2)] - dr[(2, 0)]) * 0.5;
            let e_yaw = (dr[(1, 0)] - dr[(0, 1)]) * 0.5;
            let a_pitch = kp_r * e_pitch - kd_r * omega.y;
            let a_yaw = kp_r * e_yaw - kd_r * omega.z;
            let mut j = na::DMatrix::zeros(5, j_foot.ncols());
            for c in 0..j_foot.ncols() {
                j[(0, c)] = j_foot[(1, c)];
                j[(1, c)] = j_foot[(2, c)];
                for r in 0..3 {
                    j[(2 + r, c)] = j_foot[(3 + r, c)];
                }
            }
            wt::cartesian_acceleration(
                qddot,
                &j,
                &na::DVector::from_vec(vec![djv[1], djv[2], djv[3], djv[4], djv[5]]),
                &na::DVector::from_vec(vec![a_pitch, a_yaw, a.x, a.y, a.z]),
            )
        }
        SwingAxes::Pose => {
            let (rot, rot_tgt, omega, kp_r, kd_r) =
                orientation.expect("SwingAxes::Pose needs an orientation target");
            // Same small-angle extraction the contact anchor uses: the skew
            // part of R_target * R_current^T.
            let dr = rot_tgt * rot.transpose();
            let e = na::Vector3::new(
                dr[(2, 1)] - dr[(1, 2)],
                dr[(0, 2)] - dr[(2, 0)],
                dr[(1, 0)] - dr[(0, 1)],
            ) * 0.5;
            let a_ang = e * kp_r - omega * kd_r;
            wt::cartesian_acceleration(
                qddot,
                &j_foot.rows(0, 6).into_owned(),
                &na::DVector::from_vec(vec![djv[0], djv[1], djv[2], djv[3], djv[4], djv[5]]),
                &na::DVector::from_vec(vec![a_ang.x, a_ang.y, a_ang.z, a.x, a.y, a.z]),
            )
        }
    }
}

/// Lowest level: gravity-comp torque + the nominal load split.
///
/// The load split is NOT an equal share. In the CoM task's null space,
/// "transfer load between the feet" and "walk the CoP outward" are
/// interchangeable, and a 50/50 force target makes the regulariser pick the
/// second one every time. Measured with the equal-share target: at CoM
/// y = +15 mm the split was still 50.1/49.9 and the stance CoP was already
/// 74% of the way to its edge, when 60.8/39.2 would have held it centred.
pub fn regulariser(
    tau: &Affine,
    tau_gravity: &na::DVector<f64>,
    forces: &impl AsAffine,
    forces_nominal: &na::DVector<f64>,
) -> Task {
    wt::regularize(tau, tau_gravity) + wt::regularize(forces, forces_nominal)
}

/// Build `forces_nominal`: each stance foot's share of body weight, in the
/// vertical row of its wrench block.
pub fn force_nominal(forces_size: usize, shares: &[f64], weight_n: f64) -> na::DVector<f64> {
    let mut forces_nominal = na::DVector::zeros(forces_size);
    let share_tot: f64 = shares.iter().sum::<f64>().max(1e-6);
    for (slot, s) in shares.iter().enumerate() {
        forces_nominal[6 * slot + 5] = weight_n * s / share_tot;
    }
    forces_nominal
}

/// Everything the P0 contact level needs that is not per-foot state: the
/// walker's contact model ([`crate::wbc::ContactParams`]) under the name the
/// level builders have always used. (One struct since S13d; the two were
/// field-for-field copies.)
pub use crate::wbc::ContactParams as ContactCfg;

/// How a stance foot's contact acceleration is held.
#[derive(Clone, Copy, Debug)]
pub enum ContactMode {
    /// All six rows as one hard contact.
    Rigid,
    /// Keep normal recovery authoritative while allowing a lightly loaded
    /// sole to slide/rotate enough to re-establish a physical contact.
    Managed {
        tangent_weight: f64,
        rotation_weight: f64,
        normal_weight: f64,
        normal_accel_max: f64,
    },
    /// Rolling on a sole edge (heel strike / toe-off): the edge point
    /// `r_local` (foot-link frame) is held at `p_anchor`, and the foot's
    /// orientation follows `r0` pitched by `pitch` about the sole y axis
    /// (rate `pitch_rate`), so the foot rotates about the edge.
    Edge {
        r_local: [f64; 3],
        p_anchor: na::Vector3<f64>,
        r0: na::Matrix3<f64>,
        pitch: f64,
        pitch_rate: f64,
    },
}

pub struct ContactLevel {
    pub task: Task,
    /// Per stance slot: the map from the force block to that foot's
    /// sole-frame wrench, kept so the solved CoP can be measured against the
    /// box that constrained it.
    pub sole_sel: Vec<na::DMatrix<f64>>,
    /// `[vertical Baumgarte accel_ref, vertical pose error]` for the LEFT
    /// foot: what the contact constraint is asking for, against what the foot
    /// is actually doing.
    pub acc_dbg: [f64; 2],
}

/// P0: contact acceleration (Baumgarte) + friction cone + CoP box + `f_max`,
/// added on top of `base` (the torque box, and the EoM too unless it has been
/// promoted to a level of its own).
#[allow(clippy::too_many_arguments)]
pub fn contact_level(
    dyn_ctx: &Dynamics,
    base: Task,
    j_contact: &na::DMatrix<f64>,
    dj_v: &na::DVector<f64>,
    data: &misarta::data::Data<f64>,
    v_dvec: &na::DVector<f64>,
    stance: &[usize],
    left_foot_mi: usize,
    anchors: &mut Anchors,
    load_share: &dyn Fn(usize) -> f64,
    cfg: &ContactCfg,
    contact_mode: &dyn Fn(usize) -> ContactMode,
    // Per-foot override of (sole_offset_local, cop_half) from the MEASURED
    // sole orientation, for ROLLING contact: a toe-first landing touches on
    // a line at the leading edge, and keeping the full-sole CoP box lets
    // the QP claim pressure the patch cannot carry -- measured on the
    // descent, the heel then never comes down (ankle torque stays ~1 N*m of
    // its 6) while the body vaults over the toe. None = the full sole.
    patch_mod: &dyn Fn(usize, &na::Matrix3<f64>) -> Option<([f64; 3], (f64, f64))>,
) -> ContactLevel {
    let forces = dyn_ctx.forces();
    let mut p0 = base;
    let mut sole_sel: Vec<na::DMatrix<f64>> = Vec::with_capacity(stance.len());
    let mut acc_dbg = [0.0_f64; 2];

    for (slot, foot_mi) in stance.iter().copied().enumerate() {
        let js = j_contact.rows(6 * slot, 6).into_owned();
        let djvs = dj_v.rows(6 * slot, 6).into_owned();
        let pos = misarta::se3::translation(&data.oMi[foot_mi]);
        let rot = misarta::se3::rotation_matrix(&data.oMi[foot_mi]);
        let side = usize::from(foot_mi != left_foot_mi);
        anchors.ensure(side, pos, rot);
        anchors.leak(side, pos, rot, cfg.anchor_leak, cfg.anchor_leak_rot, cfg.dt);

        let vel = &js * v_dvec;
        let acc_ref = anchors.accel_ref(side, pos, rot, &vel, cfg.kp_c, cfg.kd_c);
        if foot_mi == left_foot_mi {
            acc_dbg = [acc_ref[5], anchors.lin_error(side, pos).z];
        }

        let over = patch_mod(side, &rot);
        let (off_local, cop_half) = match over {
            Some((o, c)) => (o, c),
            None => (cfg.sole_offset_local, cfg.cop_half),
        };
        let sel = sole_selection(&rot, off_local, forces.size(), slot);
        let w_sole = &sel * &forces.as_affine();
        sole_sel.push(sel);

        p0 = p0
            + match contact_mode(side) {
                ContactMode::Rigid => {
                    wt::cartesian_acceleration(dyn_ctx.qddot(), &js, &djvs, &acc_ref)
                }
                ContactMode::Managed {
                    tangent_weight,
                    rotation_weight,
                    normal_weight,
                    normal_accel_max,
                } => {
                    let subset = |rows: &[usize], reference: &na::DVector<f64>| {
                        let mut j = na::DMatrix::zeros(rows.len(), js.ncols());
                        let mut bias = na::DVector::zeros(rows.len());
                        for (dst, &src) in rows.iter().enumerate() {
                            j.row_mut(dst).copy_from(&js.row(src));
                            bias[dst] = djvs[src];
                        }
                        wt::cartesian_acceleration(dyn_ctx.qddot(), &j, &bias, reference)
                    };
                    let rotation = subset(
                        &[0, 1, 2],
                        &na::DVector::from_iterator(3, (0..3).map(|i| acc_ref[i])),
                    )
                    .weight(rotation_weight);
                    let tangent = subset(
                        &[3, 4],
                        &na::DVector::from_iterator(2, (3..5).map(|i| acc_ref[i])),
                    )
                    .weight(tangent_weight);
                    let normal = subset(
                        &[5],
                        &na::DVector::from_element(
                            1,
                            acc_ref[5].clamp(-normal_accel_max, normal_accel_max),
                        ),
                    )
                    .weight(normal_weight);
                    rotation + tangent + normal
                }
                ContactMode::Edge { r_local, p_anchor, r0, pitch, pitch_rate } => {
                    let rr = rot * na::Vector3::new(r_local[0], r_local[1], r_local[2]);
                    let w = na::Vector3::new(vel[0], vel[1], vel[2]);
                    let v_o = na::Vector3::new(vel[3], vel[4], vel[5]);
                    let sk = rr.cross_matrix();
                    let jw = js.rows(0, 3).into_owned();
                    let jp = js.rows(3, 3).into_owned() - &sk * &jw;
                    let dw = na::Vector3::new(djvs[0], djvs[1], djvs[2]);
                    let dv = na::Vector3::new(djvs[3], djvs[4], djvs[5]);
                    let bias_p = dv - sk * dw + w.cross(&w.cross(&rr));
                    let p = pos + rr;
                    let v_p = v_o + w.cross(&rr);
                    let a_p = (p_anchor - p) * cfg.kp_c - v_p * cfg.kd_c;
                    // desired orientation: r0 pitched about its own y axis
                    let (sp, cp) = pitch.sin_cos();
                    let ry = na::Matrix3::new(cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp);
                    let rd = r0 * ry;
                    let m = rd * rot.transpose() - rot * rd.transpose();
                    let e = na::Vector3::new(m[(2, 1)], m[(0, 2)], m[(1, 0)]) * 0.5;
                    let wd = r0.column(1).into_owned() * pitch_rate;
                    let alpha = e * cfg.kp_c + (wd - w) * cfg.kd_c;
                    let jrow = na::DMatrix::from_fn(6, js.ncols(), |r, c| if r < 3 { jw[(r, c)] } else { jp[(r - 3, c)] });
                    let bias = na::DVector::from_column_slice(&[dw[0], dw[1], dw[2], bias_p[0], bias_p[1], bias_p[2]]);
                    let refv = na::DVector::from_column_slice(&[alpha[0], alpha[1], alpha[2], a_p[0], a_p[1], a_p[2]]);
                    wt::cartesian_acceleration(dyn_ctx.qddot(), &jrow, &bias, &refv)
                }
            };
        if !cfg.no_patch {
            // f_max carries the load ramp. The CoP box is |m| <= L*fz, so
            // squeezing fz shrinks the box with it -- a foot that is being
            // unloaded stops being able to argue for a CoP it is about to
            // lose.
            let share = load_share(foot_mi);
            let sole_patch = wt::ContactPatch {
                mu: cfg.friction_mu,
                cop_half,
                mu_torsion: cfg.mu_torsion,
                f_max: (cfg.f_max_per_foot * share).max(0.5),
            };
            p0 = p0 + wt::patch_contact(&w_sole, &sole_patch);
        }
    }

    ContactLevel {
        task: p0,
        sole_sel,
        acc_dbg,
    }
}
