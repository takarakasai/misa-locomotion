//! Leg odometry on a toy biped: a free-floating base with two legs of two
//! prismatic joints each (x, then z), so a foot's position relative to the
//! base is set directly by its joints.
use misa_biped::odometry::{ImuLegReading, LegOdometry, LegOdometryConfig};
use misa_biped::profile::KYO46RS;
use misa_biped::robot::{MassLink, RobotModel, StateEstimate};
use misarta::joint;
use misarta::model::{LinkInertia, ModelBuilder};
use misarta::se3;
use nalgebra as na;
use std::collections::HashMap;

fn offset(x: f64, y: f64, z: f64) -> se3::SE3<f64> {
    se3::from_rotation_and_translation(&na::Rotation3::identity(), &na::Vector3::new(x, y, z))
}

fn toy() -> RobotModel {
    let mass = |m: f64| LinkInertia { mass: m, center_of_mass: na::Vector3::zeros(), rotational_inertia: na::Matrix3::identity() * 0.01 };
    let model = ModelBuilder::new()
        .add_joint_with_link("root", 0, misarta::joint::JointType::FreeFlyer, se3::identity(), mass(20.0), "base")
        .add_joint_with_link("lx", 1, joint::prismatic_x(), offset(0.0, 0.1, 0.0), mass(1.0), "lxl")
        .add_joint_with_link("lz", 2, joint::prismatic_z(), se3::identity(), mass(1.0), "lfoot")
        .add_joint_with_link("rx", 1, joint::prismatic_x(), offset(0.0, -0.1, 0.0), mass(1.0), "rxl")
        .add_joint_with_link("rz", 4, joint::prismatic_z(), se3::identity(), mass(1.0), "rfoot")
        .build();
    let names: Vec<String> = ["lx", "lz", "rx", "rz"].iter().map(|s| s.to_string()).collect();
    let a2m = vec![Some(2), Some(3), Some(4), Some(5)];
    let mass_links = (1..6).map(|mi| MassLink { mi, m: model.inertias[mi].mass, com_local: na::Vector3::zeros() }).collect();
    let nv = model.nv;
    RobotModel {
        prof: KYO46RS,
        joint_map: names.iter().cloned().enumerate().map(|(i, n)| (n, i)).collect::<HashMap<_, _>>(),
        joint_names: names,
        a2m,
        foot_mi: [3, 5],
        trunk_mi: 1,
        trunk_from_base: true,
        mass_links,
        total_mass: 24.0,
        armature: vec![0.0; 4],
        joint_damping: vec![0.0; 4],
        torque_max: na::DVector::from_element(4, 100.0),
        q_min: na::DVector::from_element(4, -10.0),
        q_max: na::DVector::from_element(4, 10.0),
        v_max: na::DVector::from_element(4, 10.0),
        q_seed: vec![0.0; 4],
        nv,
        na: nv - 6,
        model,
    }
}

/// Base at `p` moving at `v` (world x), feet fixed at x = 0, z = 0, base
/// 0.6 m up: the joints that put the feet there and their rates.
fn reading(p: f64, v: f64, acc_bias: [f64; 3]) -> ImuLegReading {
    ImuLegReading {
        base_quat: na::UnitQuaternion::identity(),
        gyro_b: [0.0; 3],
        acc_b: [acc_bias[0], acc_bias[1], 9.81 + acc_bias[2]],
        joint_q: vec![-p, -0.6, -p, -0.6],
        joint_qd: vec![Some(-v), Some(0.0), Some(-v), Some(0.0)],
        contact_fz: [120.0, 120.0],
    }
}

fn est0(p: f64, v: f64) -> StateEstimate {
    StateEstimate {
        base_pos: [p, 0.0, 0.6],
        base_quat: na::UnitQuaternion::identity(),
        base_lin_vel_w: [v, 0.0, 0.0],
        base_ang_vel_w: [0.0; 3],
        joint_q: vec![-p, -0.6, -p, -0.6],
        joint_qd: vec![Some(-v), Some(0.0), Some(-v), Some(0.0)],
    }
}

#[test]
fn a_moving_base_over_planted_feet_is_tracked() {
    let robot = toy();
    let dt = 0.002;
    // starts believing it stands still while the base moves at 0.3 m/s
    let mut odo = LegOdometry::new(LegOdometryConfig { dt, ..Default::default() }, &robot, &est0(0.0, 0.0));
    let v = 0.3;
    let mut e = est0(0.0, 0.0);
    for k in 1..=500 {
        let p = v * dt * k as f64;
        e = odo.update(&robot, &reading(p, v, [0.0; 3]));
    }
    assert!((e.base_lin_vel_w[0] - v).abs() < 0.01, "{:?}", e.base_lin_vel_w);
    let (_, _, feet) = odo.state();
    // the feet stay where they were planted (base error from the slow start
    // shows up as a shift of both)
    assert!((feet[0][0] - feet[1][0]).abs() < 1e-6);
    assert!((e.base_pos[0] - feet[0][0] - v * dt * 500.0).abs() < 1e-3, "{:?} {:?}", e.base_pos, feet);
}

#[test]
fn an_accelerometer_bias_is_learned_not_integrated() {
    let robot = toy();
    let dt = 0.002;
    let bias = [0.2, -0.15, 0.0];
    let run = |acc_bias0: f64| {
        let cfg = LegOdometryConfig { dt, acc_bias0, ..Default::default() };
        let mut odo = LegOdometry::new(cfg, &robot, &est0(0.0, 0.0));
        let mut e = est0(0.0, 0.0);
        for _ in 0..2500 {
            e = odo.update(&robot, &reading(0.0, 0.0, bias));
        }
        e
    };
    let without = run(0.0);
    let with = run(0.3);
    let speed = |e: &StateEstimate| e.base_lin_vel_w[0].hypot(e.base_lin_vel_w[1]);
    assert!(speed(&without) > 2e-3, "{:?}", without.base_lin_vel_w);
    assert!(speed(&with) < 0.2 * speed(&without), "{:?} vs {:?}", with.base_lin_vel_w, without.base_lin_vel_w);
}
