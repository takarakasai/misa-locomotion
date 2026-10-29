//! Per-machine constants for the biped WBC.
//!
//! Everything in the controller that is a property of the ROBOT rather than
//! of the control law. It was all inlined for kyo46rs, which is exactly how
//! articara's `WbcPipeline` ended up unusable for anything but a
//! quadruped -- the leg count was spelled `[String; 4]` in a dozen places.
//! Adding a second machine is the moment to pay that off, not after.

/// Sole plane, CoP box, seed pose, gains and periods for one machine.
#[derive(Clone, Copy, Debug)]
pub struct Profile {
    pub name: &'static str,
    pub urdf: &'static str,
    pub root_link: &'static str,
    pub foot_links: [&'static str; 2],
    /// hip_pitch / knee / ankle_pitch, per side, for the crouch seed.
    pub sagittal: [[&'static str; 3]; 2],
    pub hip_roll: [&'static str; 2],
    /// Sole plane, in the foot link's own frame: how far below the origin it
    /// sits, and where its centre is fore/aft. MUST match the URDF -- the CoP
    /// box is described in this frame and a wrong centre silently describes a
    /// footprint the robot does not have.
    /// Straight-leg distance from the hip pivot (the last link above the
    /// thigh) to the SOLE PLANE [m]. 0 = not measured for this robot, which
    /// disables every reach-aware feature rather than guessing a length.
    /// G1: hip_yaw->knee 0.1939 + knee->ankle_pitch 0.3000 +
    /// ankle_pitch->ankle_roll 0.0176 + sole 0.035 = 0.5465 (from the URDF
    /// joint origins, 2026-09-24).
    pub leg_reach: f64,
    /// Links the thigh hangs from, [left, right]. `None` where they have not
    /// been identified, which disables the reach logging with the length.
    pub hip_links: Option<[&'static str; 2]>,
    pub sole_below_origin: f64,
    pub sole_centre_x: f64,
    /// CoP box half-extents (fore/aft, lateral).
    pub cop_half: (f64, f64),
    /// Height to drop the model from while measuring where the soles land.
    pub probe_z: f64,
    /// Crouch seed: knee angle. hip_pitch and ankle_pitch are -knee/2 so the
    /// three sum to zero and the sole stays parallel to the floor.
    pub knee_seed: f64,
    /// Joints written to the trajectory CSV, in order.
    pub log_joints: &'static [&'static str],
    /// Burn-in position PD, and the rotor inertia / viscous damping added to
    /// the WBC's M and h. All four scale with the machine: a gain sized for
    /// a 6.6 kg robot with 6 N*m joints does nothing to a 34 kg one with
    /// 139 N*m knees, and armature copied from the wrong motor puts a
    /// systematic error on every actuated row of the mass matrix.
    /// kv must stay under 2*I/dt -- the plant's joint PD is explicit.
    pub burnin_kp: f64,
    pub burnin_kv: f64,
    pub armature: f64,
    pub joint_damping: f64,
    /// Drop mesh collision geoms and contact only on the URDF's primitives.
    /// This is what Unitree's own `g1_23dof.xml` does -- every mesh geom in
    /// it carries `contype="0" conaffinity="0" group="1"`, i.e. visual only,
    /// and the 8 spheres + 4 cylinders carry all the contact. Converting the
    /// URDF naively instead collides the detailed meshes against each other,
    /// and on G1 the pelvis cover overlaps the hip links BY DESIGN: measured,
    /// 241.9 kN per side, 722x body weight, present on every tick. That is
    /// the same brace that made kyo46rs's single-leg stance look solved.
    pub collide_primitives_only: bool,
    /// Body whose orientation P2 holds upright. None = the FreeFlyer body
    /// itself. kyo46rs's floating base IS its torso; G1's is the pelvis, with
    /// a waist joint between it and the upper body, so "hold the trunk level"
    /// means different things on the two machines.
    pub trunk_link: Option<&'static str>,
    /// WBC period. Not a preference: G1 holds 6.6 s at 2 ms against 0.96 s at
    /// 5 ms, because between ticks the plant runs open-loop on a torque
    /// computed for a state it has since left, and how far it drifts scales
    /// with the machine.
    pub ctrl_dt: f64,
    /// Swing-foot clearance (m) and the gains that hold it. Geometric and
    /// leg-length dependent -- 40 mm is a real step on a 0.66 m robot and a
    /// scuff on a 1.3 m one.
    pub lift_h: f64,
    pub kp_swing: f64,
    pub kd_swing: f64,
    /// Degraded-solve fallback PD. TORQUE dimension, so it does not carry
    /// across machines the way the acceleration-level task gains do.
    pub hold_kp: f64,
    pub hold_kd: f64,
    /// Seconds of base-welded settling before the free-base run.
    pub burnin_s: f64,
    /// Friction the QP plans against. Deliberately BELOW the plant's
    /// MU_GROUND so the solver is the conservative one; keeping the margin
    /// explicit stops the two drifting apart silently.
    pub friction_margin: f64,
}

pub const KYO46RS: Profile = Profile {
    name: "kyo46rs",
    urdf: "/home/takara/work/dp/humanoid/kyo46rs_description/urdf/kyo46rs.urdf",
    root_link: "torso",
    foot_links: ["left_foot_link", "right_foot_link"],
    sagittal: [
        [
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
        ],
        [
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        ],
    ],
    hip_roll: ["left_hip_roll_joint", "right_hip_roll_joint"],
    sole_below_origin: 0.035,
    // hip_pitch -> knee 0.112 + knee -> ankle_pitch 0.100 + ankle_pitch ->
    // foot 0.046 + foot origin -> sole 0.035, from the URDF (2026-10-18).
    // Measured against the thigh link (= the hip pitch joint), see
    // `hip_links`. Only the `reach_*` log columns and the opt-in reach caps
    // read it; the shipped kyo46rs recipe is otherwise unchanged.
    leg_reach: 0.293,
    hip_links: Some(["left_thigh_link", "right_thigh_link"]),
    sole_centre_x: 0.0,
    cop_half: (0.049, 0.019),
    probe_z: 0.47,
    knee_seed: 0.70,
    burnin_kp: 150.0,
    burnin_kv: 2.0,
    armature: 0.0005,
    joint_damping: 0.15,
    collide_primitives_only: false,
    trunk_link: None,
    ctrl_dt: 0.005,
    lift_h: 0.04,
    kp_swing: 400.0,
    kd_swing: 40.0,
    hold_kp: 15.0,
    hold_kd: 2.0,
    burnin_s: 1.2,
    friction_margin: 0.857, // 0.6 against a 0.7 plant, the measured pair
    log_joints: &[
        "left_hip_yaw_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        // A joint missing from this list is not merely absent from the CSV:
        // the renderer defaults an unknown joint to 0.0, so it is DRAWN as
        // never having moved, and "it did not move" cannot be told apart from
        // "the log does not carry it". The v6 shoulder roll spent a session in
        // exactly that state.
        "left_shoulder_roll_joint",
        "left_shoulder_pitch_joint",
        "left_elbow_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_pitch_joint",
        "right_elbow_joint",
    ],
};

/// kyo46rs v2: the same robot with CAD geometry and mesh-derived inertia.
///
/// Kinematics, joint limits, link masses and **collision geometry** are
/// bit-identical to [`KYO46RS`] -- verified, 18 geoms, exact match, not merely
/// claimed by its README. So it does NOT relieve the forearm-against-hip
/// interference that blocks stepping (doc section 10.4): that is a function of
/// the collision primitives and the missing shoulder abduction DoF, and both
/// are unchanged.
///
/// What IS different is the inertia: integrated over the real solid rather
/// than hand-lumped boxes. Torso ixx x1.94, forearm ixx x0.72, and every link
/// gains a small CoM offset. That reaches the WBC through `crba`, so it is
/// worth running as the more faithful plant -- just not as a fix for the
/// geometry.
pub const KYO46RS2: Profile = Profile {
    urdf: "/home/takara/work/dp/humanoid/kyo46rs2_description/urdf/kyo46rs2.urdf",
    name: "kyo46rs2",
    ..KYO46RS
};

/// Unitree G1, 23-DOF variant. 34.13 kg against kyo46rs's 6.64, and every
/// torque limit is a real per-joint number (knee 139, ankle 35, hip 88)
/// rather than an estimate, which is the point of running it: it separates
/// "the control law is fragile" from "the model is fragile".
///
/// Foot: four 5 mm contact spheres at the corners of a 170 x 60 mm footprint,
/// 35 mm below the ankle_roll origin, and the patch is NOT centred on the
/// ankle -- it runs -50..+120 mm fore/aft, so its centre is 35 mm forward.
pub const G1_23DOF: Profile = Profile {
    name: "g1_23dof",
    urdf: "/home/takara/work/dp/articara/models/unitree_g1_src/robots/g1_description/g1_23dof.urdf",
    root_link: "pelvis",
    foot_links: ["left_ankle_roll_link", "right_ankle_roll_link"],
    sagittal: [
        [
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
        ],
        [
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        ],
    ],
    hip_roll: ["left_hip_roll_joint", "right_hip_roll_joint"],
    sole_below_origin: 0.035,
    leg_reach: 0.5465,
    hip_links: Some(["left_hip_yaw_link", "right_hip_yaw_link"]),
    sole_centre_x: 0.035,
    cop_half: (0.085, 0.030),
    probe_z: 0.90,
    knee_seed: 0.70,
    // Sized off the torque limits: G1's knee is 139 N*m against kyo46rs's 12,
    // and 34 kg against 6.6. The URDF declares no damping, friction or
    // armature at all, so these are engineering placeholders, not data --
    // flagged here because the same gap on kyo46rs cost a day.
    burnin_kp: 2000.0,
    burnin_kv: 20.0,
    armature: 0.01,
    joint_damping: 1.0,
    collide_primitives_only: true,
    trunk_link: Some("torso_link"),
    // 2 ms measured; 5 ms loses the machine in under a second.
    ctrl_dt: 0.002,
    // Scaled by leg length (G1's is roughly 2x kyo46rs's). NOT measured --
    // single support has never run far enough on G1 to tune the swing.
    lift_h: 0.08,
    kp_swing: 400.0,
    kd_swing: 40.0,
    // Left at kyo46rs's values ON PURPOSE. The torque-dimension argument says
    // these should scale with the actuators (25-139 N*m against 6-12), but
    // measured, raising them hurts: 15 -> 200 takes G1 from 6.63 s to 0.81 s,
    // and 600 gives 1.08 s. Whatever the fallback is doing, it is not
    // limited by its authority.
    hold_kp: 15.0,
    hold_kd: 2.0,
    burnin_s: 1.2,
    friction_margin: 0.857,
    log_joints: &[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    ],
};

/// `ROBOT=kyo46rs|g1`. Panics on an unknown name rather than defaulting --
/// silently running the wrong machine's sole geometry is the single most
/// expensive mistake available here.
pub fn by_name(name: &str) -> Profile {
    match name {
        "" | "kyo46rs" => KYO46RS,
        "kyo46rs2" | "v2" => KYO46RS2,
        "g1" | "g1_23dof" => G1_23DOF,
        other => panic!("unknown ROBOT={other:?} (kyo46rs | kyo46rs2 | g1)"),
    }
}
