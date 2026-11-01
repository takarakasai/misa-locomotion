//! What to do with the WBC's answer, including when it did not produce one.
//!
//! **A degraded solve is not a slightly-worse solve.** misa-wbc's `HoQp`
//! returns `x_new = prev.x` on a failed inner QP, and at level 0 that is the
//! ZERO vector -- so the "solution" satisfies the HOMOGENEOUS equation of
//! motion, with all of gravity dropped and the contact Baumgarte rows
//! discarded along with it, on exactly the transition ticks where they work
//! hardest. Commanding it is worse than doing nothing.

//!
//! The host hands the result to its plant (articara's `write_to_plant`, a
//! real robot's driver); what is commanded is decided here. Lines go to the
//! [`LogSink`].

use misa_wbc::SolveStatus;
use nalgebra as na;

use crate::sink::LogSink;

/// Running tally of degraded solves, labelled by the level that failed.
pub struct DegradedTally {
    pub n: u32,
    by_level: Vec<u32>,
    /// The stack changes shape between double and single support, so the
    /// tally has to be labelled with the names of the LAST configuration
    /// actually solved -- a fixed list mislabels every single-support run by
    /// one.
    final_level_names: Vec<String>,
}

impl Default for DegradedTally {
    fn default() -> Self {
        Self::new()
    }
}

impl DegradedTally {
    pub fn new() -> Self {
        DegradedTally {
            n: 0,
            by_level: vec![0; 12],
            final_level_names: Vec::new(),
        }
    }

    pub fn observe(
        &mut self,
        status: &SolveStatus,
        t: f64,
        tick: usize,
        nc: usize,
        level_names: &[&str],
        out: &mut dyn LogSink,
    ) {
        self.final_level_names = level_names.iter().map(|s| s.to_string()).collect();
        if matches!(status, SolveStatus::Optimal) {
            return;
        }
        self.n += 1;
        if let SolveStatus::Degraded { level, .. } = status {
            if *level < self.by_level.len() {
                self.by_level[*level] += 1;
            }
        }
        if self.n <= 6 || tick % 200 == 0 {
            let nm = match status {
                SolveStatus::Degraded { level, .. } => {
                    level_names.get(*level).copied().unwrap_or("?")
                }
                _ => "-",
            };
            out.line(&format!("    [degraded] t={t:6.3} nc={nc} level={nm} status={status:?}"));
        }
    }

    /// Final tally by NAME rather than index.
    pub fn report(&self, out: &mut dyn LogSink) {
        let parts: Vec<String> = self
            .by_level
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(i, c)| {
                format!(
                    "{}={c}",
                    self.final_level_names
                        .get(i)
                        .map(|s| s.as_str())
                        .unwrap_or("?")
                )
            })
            .collect();
        if !parts.is_empty() {
            out.line(&format!("    by level: {}", parts.join("  ")));
        }
    }
}

/// How a degraded solve is turned into something safe to command.
pub struct CommandPolicy {
    /// Which priority levels, if they degrade, invalidate the whole solution.
    ///
    /// A level-2 failure returns the level-1 solution, which still satisfies
    /// the EoM, the contacts, the friction and CoP cones, the torque box AND
    /// the CoM task; only trunk orientation is compromised. Replacing that
    /// with gravity comp ought to be strictly worse -- but measured both
    /// ways, kyo46rs single-leg goes SURVIVED -> FELL (tilt 0.146 -> 0.529)
    /// and G1 goes 0.147 -> 0.522 rad. The reason is the BOTTOM of the stack:
    /// the last level is the regulariser, and a solution that stopped before
    /// it has an unconstrained null-space component, so tau can be anything
    /// the higher levels did not pin. "Satisfies the hard constraints" does
    /// not mean "is a sane torque". Default 999 keeps the conservative
    /// behaviour; the knob stays so the experiment is repeatable.
    pub fallback_max_level: usize,
    /// Consecutive degraded ticks bridged with the last good torque before
    /// switching to the recomputed one. Over a few ticks freezing is the
    /// smoother choice; swapping controllers every other tick just chatters
    /// (a straight swap measured a 256 N contact spike, 4x body weight).
    pub hold_bridge: u32,
    /// Restore the old always-freeze-the-last-good-torque behaviour, for A/B.
    pub hold_last: bool,
    /// Crossfade length on each fallback <-> QP handover. Default 0, and that
    /// is measured, not lazy: 10/20/40/80 ticks all topple (20 drives the
    /// knee to -25.8 deg). The step at handover is not what it looked like --
    /// the same stance-foot liftoff happens with the fallback disabled
    /// entirely, always on the tick a degraded RUN ends, so the discontinuity
    /// is between the QP's broken solution and its recovered one, and
    /// crossfading into a broken solution cannot help.
    pub blend_ticks: u32,

    last_good: Option<Vec<f64>>,
    consec_degraded: u32,
    cmd_prev: Vec<f64>,
    blend_from: Vec<f64>,
    blend_left: u32,
    in_fallback_prev: bool,
}

impl CommandPolicy {
    pub fn new(
        n_joints: usize,
        fallback_max_level: usize,
        hold_bridge: u32,
        hold_last: bool,
        blend_ticks: u32,
    ) -> Self {
        CommandPolicy {
            fallback_max_level,
            hold_bridge,
            hold_last,
            blend_ticks: if hold_last { 0 } else { blend_ticks },
            last_good: None,
            consec_degraded: 0,
            cmd_prev: vec![0.0; n_joints],
            blend_from: vec![0.0; n_joints],
            blend_left: 0,
            in_fallback_prev: false,
        }
    }

    /// Replace `robot_taus` with something safe to command, given how the
    /// solve went. `fallback` recomputes a support torque for the pose the
    /// robot is ACTUALLY in (gravity comp + posture PD) -- see
    /// [`gravity_plus_posture`].
    ///
    /// Returns true if the fallback (or a frozen command) is driving.
    pub fn apply(
        &mut self,
        status: &SolveStatus,
        robot_taus: &mut [f64],
        t: f64,
        dt: f64,
        nc: usize,
        fallback: &dyn Fn(&mut [f64]),
        out: &mut dyn LogSink,
    ) -> bool {
        let bad_level = match status {
            SolveStatus::Optimal => None,
            SolveStatus::Degraded { level, .. } => Some(*level),
        };
        let unusable = bad_level.is_some_and(|l| l <= self.fallback_max_level);
        let mut in_fallback = false;
        match self.last_good.as_ref() {
            Some(prev) if unusable => {
                in_fallback = true;
                if self.hold_last || self.consec_degraded < self.hold_bridge {
                    robot_taus.copy_from_slice(prev);
                } else {
                    fallback(robot_taus);
                }
                self.consec_degraded += 1;
                // Holding the last good torque bridges an occasional failed
                // solve. It must not quietly become the controller: a long
                // run of failures means the robot is open-loop on a stale
                // command, which reads in the logs as a smooth mechanical
                // collapse rather than as a control fault. (Measured: 540 ms
                // of frozen torque while the stance knee folded 48 -> 11 deg,
                // with the torque columns byte-identical throughout.)
                if self.consec_degraded == 10 {
                    let src = if self.hold_last {
                        format!(
                            "still commanding the torque from t={:.3}",
                            t - self.consec_degraded as f64 * dt
                        )
                    } else {
                        "running on gravity comp + posture PD".to_string()
                    };
                    out.line(&format!(
                        "  [OPEN LOOP] t={t:6.3} nc={nc}: {} consecutive degraded solves, {src}",
                        self.consec_degraded
                    ));
                }
            }
            _ => {
                if self.consec_degraded >= 10 {
                    out.line(&format!(
                        "  [recovered] t={t:6.3} after {} degraded ticks",
                        self.consec_degraded
                    ));
                }
                self.consec_degraded = 0;
                // Only a clean solve becomes the held command. A level-3
                // solution is good enough to send but not to freeze.
                if bad_level.is_none() {
                    self.last_good = Some(robot_taus.to_vec());
                }
            }
        }
        // Crossfade whenever the commanding controller changes, in either
        // direction, from whatever was last actually sent. `last_good` keeps
        // the QP's own output rather than this blended command, so the bridge
        // still freezes a real solution.
        if self.blend_ticks > 0 && in_fallback != self.in_fallback_prev {
            self.blend_left = self.blend_ticks;
            self.blend_from.copy_from_slice(&self.cmd_prev);
        }
        self.in_fallback_prev = in_fallback;
        if self.blend_left > 0 {
            let a = 1.0 - f64::from(self.blend_left) / f64::from(self.blend_ticks);
            for k in 0..robot_taus.len() {
                robot_taus[k] = self.blend_from[k] * (1.0 - a) + robot_taus[k] * a;
            }
            self.blend_left -= 1;
        }
        self.cmd_prev.copy_from_slice(robot_taus);
        in_fallback
    }

    /// Whether the policy is currently running open-loop on a stale command.
    pub fn consecutive_degraded(&self) -> u32 {
        self.consec_degraded
    }
}

/// The degraded-solve fallback: gravity compensation plus a PD onto the seed
/// posture, clamped to the torque box.
///
/// This replaced freezing the last good torque, which was measured to be
/// actively harmful: a torque solved for two feet supplies about half the
/// support one foot needs, so freezing it let the stance knee sag 49 -> 11 deg
/// over 705 ms, and the QP then had to recover from a leg whose own 6x6
/// Jacobian had gone from cond 49 to cond 206.
///
/// `actuated` is the `(host joint index, misarta v-index)` pairs
/// ([`crate::robot::RobotModel::actuated`]); `q_seed` (host order) the
/// posture held, `torque_max` the box (v-index - 6). `q_joint` is the joint
/// angles the PD reads (host order): the state estimate's, or the plant's
/// if the host chooses.
#[allow(clippy::too_many_arguments)]
pub fn gravity_plus_posture(
    actuated: &[(usize, usize)],
    q_seed: &[f64],
    torque_max: &na::DVector<f64>,
    q_joint: &[f64],
    tau_gravity: &na::DVector<f64>,
    v: &[f64],
    hold_kp: f64,
    hold_kd: f64,
    out: &mut [f64],
) {
    for &(ji, vi) in actuated {
        let e = q_seed[ji] - q_joint[ji];
        let tau = tau_gravity[vi - 6] + hold_kp * e - hold_kd * v[vi];
        let lim = torque_max[vi - 6];
        out[ji] = tau.clamp(-lim, lim);
    }
}
