//! Contact anchors, contact Jacobians, and the foot-link -> sole wrench map.
//!
//! Moved from articara's `biped::contact` on 2026-10-30 (S13d of
//! doc/from_claude/20261029_walker_extraction_plan.md), which re-exports it
//! under its old path.
//!
//! Three things live here that each cost a day to find:
//!
//! 1. **The contact constraint needs Baumgarte.** `zero_contact_acceleration`
//!    pins `J*qddot + Jdot*v = 0`, an ACCELERATION constraint: give the foot
//!    any angular velocity and its orientation drifts at that rate forever,
//!    because nothing feeds the pose error back. Invisible in a symmetric
//!    squat (the foot never acquires roll rate) and fatal in a lateral weight
//!    shift, where the stance sole rolled to 19 deg while the solver still
//!    reported the contact satisfied. Once the sole is on its edge, the
//!    rectangular CoP box describes a footprint that is no longer touching
//!    the floor.
//! 2. **A frozen anchor becomes a phantom force.** See [`Anchors::leak`].
//! 3. **The CoP box is about the SOLE, not the foot link origin.** See
//!    [`sole_selection`].

use nalgebra as na;

/// The pose a foot's contact is holding: where it was when it touched down.
pub type AnchorPose = (na::Vector3<f64>, na::Matrix3<f64>);

/// Per-foot touchdown pose, indexed by side (0 = left, 1 = right).
#[derive(Default, Clone)]
pub struct Anchors {
    pub a: [Option<AnchorPose>; 2],
    /// Vertical offset added to the anchor when the Baumgarte error is formed,
    /// in metres, positive up. The anchor itself is left alone so that
    /// [`Anchors::leak`] keeps tracking the real foot.
    ///
    /// This is the actuation channel for foot force difference control: with a
    /// unilateral contact the only way to ask a foot for more normal force is
    /// to ask it to go down. Zero everywhere unless a caller sets it, which
    /// makes the whole mechanism inert by default.
    pub bias_z: [f64; 2],
}

impl Anchors {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the anchor for a foot that has just made contact, discarding
    /// whatever it held before.
    ///
    /// Walking needs this and standing did not, which is why it did not exist:
    /// with `get_or_insert` alone, a foot that lifts and lands somewhere else
    /// keeps arguing for the place it left. That is the same failure as a
    /// stale anchor ([`Anchors::leak`]) except the error is a whole step
    /// long instead of 12 mm.
    pub fn touchdown(&mut self, side: usize, pos: na::Vector3<f64>, rot: na::Matrix3<f64>) {
        self.a[side] = Some((pos, rot));
    }

    /// Forget a foot's anchor as it leaves the contact set.
    pub fn release(&mut self, side: usize) {
        self.a[side] = None;
    }

    /// Set the anchor only if the foot does not already have one.
    pub fn ensure(&mut self, side: usize, pos: na::Vector3<f64>, rot: na::Matrix3<f64>) {
        self.a[side].get_or_insert((pos, rot));
    }

    /// Let the anchor follow a foot that has genuinely moved.
    ///
    /// The anchor used to be frozen once and never revisited, and the stance
    /// foot slides ~12 mm during a single-support transition. A stale anchor
    /// is not a small error: `kp_c * 12 mm = 19.6 m/s^2` of lateral foot
    /// acceleration demanded forever, which the QP pays for by planning a
    /// contact force that does not exist. Measured in the settled single-leg
    /// pose, the QP believed it was applying 24 N of tangential force -- 37%
    /// of body weight -- while MuJoCo's contacts summed to 0.0 N, and it
    /// thought fz was 71.3 N when the true value is 65.1 N = mg exactly. The
    /// torque it sends is computed against that phantom reaction.
    ///
    /// Baumgarte is there to reject drift, not to relitigate where the foot
    /// ought to be, so the anchor leaks toward the current pose with a time
    /// constant far slower than the contact dynamics.
    ///
    /// `leak_rot` is deliberately 0 by default. Rotational drift is the
    /// failure the Baumgarte term exists for -- unchecked, the stance sole
    /// rolled to 19 deg while the solver still called the contact satisfied.
    /// Translation can be conceded; roll cannot.
    pub fn leak(
        &mut self,
        side: usize,
        pos: na::Vector3<f64>,
        rot: na::Matrix3<f64>,
        leak: f64,
        leak_rot: f64,
        dt: f64,
    ) {
        if leak <= 0.0 {
            return;
        }
        if let Some((ap, ar)) = self.a[side].as_mut() {
            let a = (leak * dt).min(1.0);
            *ap += (pos - *ap) * a;
            let ar_a = (leak_rot * dt).min(1.0);
            *ar = *ar + (rot - *ar) * ar_a;
        }
    }

    /// Baumgarte reference acceleration for one foot, `[angular; linear]` in
    /// the world frame: a PD on the pose error against the anchor.
    pub fn accel_ref(
        &self,
        side: usize,
        pos: na::Vector3<f64>,
        rot: na::Matrix3<f64>,
        vel: &na::DVector<f64>,
        kp: f64,
        kd: f64,
    ) -> na::DVector<f64> {
        let (p0, r0) = self.a[side].expect("anchor must be set before accel_ref");
        let dr = r0 * rot.transpose();
        // rotation vector of dr (small-angle: the skew part)
        let e_ang = na::Vector3::new(
            dr[(2, 1)] - dr[(1, 2)],
            dr[(0, 2)] - dr[(2, 0)],
            dr[(1, 0)] - dr[(0, 1)],
        ) * 0.5;
        let mut e_lin = p0 - pos;
        e_lin.z += self.bias_z[side];
        let mut acc_ref = na::DVector::zeros(6);
        for k in 0..3 {
            acc_ref[k] = kp * e_ang[k] - kd * vel[k];
            acc_ref[3 + k] = kp * e_lin[k] - kd * vel[3 + k];
        }
        acc_ref
    }

    /// Linear part of the pose error against the anchor, for logging.
    pub fn lin_error(&self, side: usize, pos: na::Vector3<f64>) -> na::Vector3<f64> {
        match self.a[side] {
            Some((p0, _)) => {
                let mut e = p0 - pos;
                e.z += self.bias_z[side];
                e
            }
            None => na::Vector3::zeros(),
        }
    }
}

/// Stacked `J_c` (6 rows per stance foot) and `Jdot_c * v`, in the order the
/// stance list gives.
pub fn contact_jacobians(
    model: &misarta::model::Model<f64>,
    q: &[f64],
    v: &[f64],
    data: &misarta::data::Data<f64>,
    v_dvec: &na::DVector<f64>,
    stance: &[usize],
    nv: usize,
) -> (na::DMatrix<f64>, na::DVector<f64>) {
    let nc = stance.len();
    let mut j_contact = na::DMatrix::zeros(6 * nc, nv);
    let mut dj_v = na::DVector::zeros(6 * nc);
    for (slot, foot_mi) in stance.iter().copied().enumerate() {
        let jf = misarta::jacobian::compute_joint_jacobian_from_data(model, q, data, foot_mi);
        let djf = misarta::jacobian::compute_joint_jacobian_time_derivative(model, q, v, foot_mi);
        let djv = &djf * v_dvec;
        for r in 0..6 {
            for c in 0..nv {
                j_contact[(6 * slot + r, c)] = jf[(r, c)];
            }
            dj_v[6 * slot + r] = djv[r];
        }
    }
    (j_contact, dj_v)
}

/// Map the decision-variable force block to one foot's wrench ABOUT ITS SOLE,
/// expressed in the sole frame.
///
/// The CoP box is only the real centre-of-pressure condition about the sole;
/// the force variables are a wrench about the foot LINK ORIGIN, 35-59 mm
/// higher, where a tangential `fx` fakes `h*fx` of moment. Constraining the
/// link-origin wrench instead defends a footprint the robot does not have.
///
/// `sole_offset_local` MUST track the URDF's foot collision box.
pub fn sole_selection(
    rot: &na::Matrix3<f64>,
    sole_offset_local: [f64; 3],
    forces_size: usize,
    slot: usize,
) -> na::DMatrix<f64> {
    let r_w = rot
        * na::Vector3::new(
            sole_offset_local[0],
            sole_offset_local[1],
            sole_offset_local[2],
        );
    let rt = rot.transpose();
    let skew = na::Matrix3::new(0.0, -r_w.z, r_w.y, r_w.z, 0.0, -r_w.x, -r_w.y, r_w.x, 0.0);
    let top_right = -(rt * skew);
    let mut sel = na::DMatrix::zeros(6, forces_size);
    for i in 0..3 {
        for jj in 0..3 {
            sel[(i, 6 * slot + jj)] = rt[(i, jj)];
            sel[(i, 6 * slot + 3 + jj)] = top_right[(i, jj)];
            sel[(3 + i, 6 * slot + 3 + jj)] = rt[(i, jj)];
        }
    }
    sel
}

/// The CoP of a sole-frame wrench lives in [`crate::wbc`] (the walker reads
/// it from the stack's solution); re-exported under its old path.
pub use crate::wbc::cop_from_sole_wrench;
