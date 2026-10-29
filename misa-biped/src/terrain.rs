//! Stepped terrain the walker plans against: a flight of equal risers and
//! treads followed by a landing.
//!
//! Moved here from `examples/kyo46rs_walk.rs` on 2026-09-23 (P2-0 of
//! doc/from_claude/20260923_p2_contact_planner_design_v2.md) so that a contact
//! planner in this crate can read the tread spans. `h` and `base` are the
//! example's expressions verbatim -- they decide every footfall height and are
//! covered by the bit-identity regression -- and the new accessors are
//! additions only.

/// A flight of `n` steps of height `riser` (negative = descending) and depth
/// `tread`, whose first riser stands at world x = `start`, then a landing of
/// length `landing_len` at the top.
#[derive(Clone, Copy, Debug)]
pub struct Stairs {
    pub n: usize,
    pub riser: f64,
    pub tread: f64,
    pub start: f64,
    /// Length of the physical landing past the last tread. Historically the
    /// example hard-coded 0.60 m while `h()` treated the top as infinite;
    /// measured 2026-09-23 on G1, every walk-out fell at exactly this edge
    /// (see doc/from_claude/20260923_landing_edge.md).
    pub landing_len: f64,
    /// Steps of a DESCENDING flight (same |riser| and tread) that starts at
    /// `landing_end()`: an up-and-over course. 0 = none (the historical
    /// terrain, bit-identical). Only meaningful for an ascending `riser`.
    pub down_n: usize,
    /// A SLOPE instead of a flight (`n == 0`): rise per metre of x from
    /// `start` over a horizontal run of `slope_len`, flat beyond. Negative
    /// descends from a platform `slope_len * |slope|` high. 0 = no slope,
    /// bit-identical (doc/from_claude/20261015_terrain_bench.md).
    pub slope: f64,
    pub slope_len: f64,
}

impl Stairs {
    /// Parse the `STAIRS` string `n,riser,tread[,start_x]`.
    pub fn parse(spec: &str, landing_len: f64) -> Self {
        let v: Vec<f64> = spec
            .split(',')
            .map(|x| {
                x.trim()
                    .parse()
                    .expect("STAIRS wants n,riser,tread[,start_x]")
            })
            .collect();
        assert!(v.len() >= 3, "STAIRS wants n,riser,tread[,start_x]");
        Stairs {
            n: v[0] as usize,
            riser: v[1],
            tread: v[2],
            start: v.get(3).copied().unwrap_or(0.25),
            landing_len,
            down_n: 0,
            slope: 0.0,
            slope_len: 0.0,
        }
    }

    /// A slope (`TERRAIN_SLOPE="deg[,start[,run]]"`): `n = 0`, rising
    /// `tan(deg)` per metre over `run` metres of x from `start`.
    pub fn slope(deg: f64, start: f64, run: f64, landing_len: f64) -> Self {
        Stairs {
            n: 0,
            riser: 0.0,
            tread: 0.211,
            start,
            landing_len,
            down_n: 0,
            slope: deg.to_radians().tan(),
            slope_len: run.max(0.0),
        }
    }

    /// Pitch of the walking surface at `x` [rad] (positive = rising ahead):
    /// the slope's angle inside its run, 0 elsewhere (treads are level).
    pub fn pitch_at(&self, x: f64) -> f64 {
        if self.n == 0 && self.slope != 0.0 && x >= self.start && x <= self.start + self.slope_len {
            self.slope.atan()
        } else {
            0.0
        }
    }

    /// Platform height the robot starts on (0 when ascending).
    pub fn base(&self) -> f64 {
        if self.n == 0 && self.slope < 0.0 {
            self.slope_len * -self.slope
        } else if self.riser < 0.0 {
            self.n as f64 * -self.riser
        } else {
            0.0
        }
    }

    /// World z of the walking surface under sole-centre x.
    ///
    /// NOTE: clamps the level to `n`, i.e. the top surface is infinite here
    /// while the physical landing ends at `landing_end()`. Planners that place
    /// feet past the last tread must check `landing_span()` themselves.
    pub fn h(&self, x: f64) -> f64 {
        if self.n == 0 && self.slope != 0.0 {
            return self.base() + self.slope * (x - self.start).clamp(0.0, self.slope_len);
        }
        if self.down_n > 0 && self.riser > 0.0 && x >= self.landing_end() {
            let k2 = (((x - self.landing_end()) / self.tread) as i64 + 1).clamp(0, self.down_n as i64);
            return (self.n as i64 - k2).max(0) as f64 * self.riser;
        }
        let k = if x < self.start {
            0
        } else {
            ((x - self.start) / self.tread) as i64 + 1
        };
        self.base() + k.clamp(0, self.n as i64) as f64 * self.riser
    }

    /// x interval `[x0, x1)` of tread `k` (1-based; `k == 0` is the approach
    /// floor, unbounded below).
    pub fn tread_span(&self, k: usize) -> (f64, f64) {
        if k == 0 {
            return (f64::NEG_INFINITY, self.start);
        }
        let x0 = self.start + (k - 1) as f64 * self.tread;
        (x0, x0 + self.tread)
    }

    /// x where the top surface begins (end of the last tread).
    pub fn landing_start(&self) -> f64 {
        self.start + self.n as f64 * self.tread
    }

    /// x where the physical landing ends.
    pub fn landing_end(&self) -> f64 {
        self.landing_start() + self.landing_len
    }

    /// x interval of the landing.
    pub fn landing_span(&self) -> (f64, f64) {
        (self.landing_start(), self.landing_end())
    }

    /// x interval of the flat surface that contains `x` -- the approach
    /// floor, one tread, or the landing.
    ///
    /// A tread is 0.211 m and a G1 sole is 0.170 m, so the sole centre has a
    /// 41 mm window to sit fully on one. A step adjustment with +-60 mm of
    /// authority in x will walk the foot off that window unless it is told
    /// where the edges are: measured 2026-09-25, per-cycle adjustment put
    /// 16/24 landings fully on the tread against 20/24 for the throttled
    /// one, and the falls moved from the landing beyond the step (27.8 s) to
    /// the step itself (22.3 s).
    pub fn surface_span(&self, x: f64) -> (f64, f64) {
        if self.n == 0 && self.slope != 0.0 {
            // One continuous surface: no edges to keep a sole off.
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        if x < self.start {
            return (f64::NEG_INFINITY, self.start);
        }
        let k = ((x - self.start) / self.tread) as usize + 1;
        if k > self.n {
            self.landing_span()
        } else {
            self.tread_span(k)
        }
    }

    /// Where the sole centre may land so the whole sole sits on the surface
    /// that `x` is on. `None` when that surface is narrower than the sole.
    pub fn sole_window(&self, x: f64, sole_half_l: f64) -> Option<(f64, f64)> {
        let (lo, hi) = self.surface_span(x);
        let (lo, hi) = (lo + sole_half_l, hi - sole_half_l);
        (lo <= hi).then_some((lo, hi))
    }

    /// The nearest `x` to `want` at which the whole sole sits on SOME flat
    /// surface, and whether `want` itself already did.
    ///
    /// Not `sole_window(want)`: that names one surface, and clamping to the
    /// surface a step is currently standing over forbids the step that climbs
    /// onto the next one. What the terrain actually demands is only that the
    /// foot land fully on something. A `want` in the dead zone that straddles
    /// a nosing is pushed to whichever side is closer.
    pub fn snap_to_sole_window(&self, want: f64, sole_half_l: f64) -> (f64, bool) {
        if let Some((lo, hi)) = self.sole_window(want, sole_half_l)
            && (lo..=hi).contains(&want)
        {
            return (want, true);
        }
        // `want` straddles an edge (or sits on a surface too narrow for the
        // sole). Consider this surface and its neighbours and take the
        // closest admissible point.
        let probe = [
            want - self.tread.max(sole_half_l * 2.0),
            want,
            want + self.tread.max(sole_half_l * 2.0),
        ];
        let mut best: Option<f64> = None;
        for x in probe {
            let Some((lo, hi)) = self.sole_window(x, sole_half_l) else {
                continue;
            };
            let c = want.clamp(lo, hi);
            if best.is_none_or(|b| (c - want).abs() < (b - want).abs()) {
                best = Some(c);
            }
        }
        (best.unwrap_or(want), false)
    }

    /// Height of the top surface.
    pub fn top(&self) -> f64 {
        self.base() + self.n as f64 * self.riser.max(0.0)
    }
}

#[cfg(test)]
mod tests {

    /// Hand-computed against the STAIRS spec, not against `surface_span`:
    /// one 0.211 m tread starting at 2.00, a 0.170 m sole. The tread runs
    /// [2.000, 2.211], so the sole centre fits in [2.085, 2.126] -- 41 mm.
    #[test]
    fn the_sole_window_on_a_tread_is_the_tread_less_the_sole() {
        let s = Stairs::parse("1,0.060,0.211,2.00", 3.0);
        let (lo, hi) = s.sole_window(2.10, 0.085).unwrap();
        assert!((lo - 2.085).abs() < 1e-12, "{lo}");
        assert!((hi - 2.126).abs() < 1e-12, "{hi}");
        // The approach floor is unbounded below and ends at the riser.
        let (lo, hi) = s.sole_window(1.50, 0.085).unwrap();
        assert!(lo.is_infinite() && lo < 0.0);
        assert!((hi - (2.00 - 0.085)).abs() < 1e-12, "{hi}");
        // Past the last tread is the landing: 2.211 to 5.211.
        let (lo, hi) = s.sole_window(3.00, 0.085).unwrap();
        assert!((lo - (2.211 + 0.085)).abs() < 1e-12, "{lo}");
        assert!((hi - (5.211 - 0.085)).abs() < 1e-12, "{hi}");
        // A sole longer than the tread has no window at all.
        assert!(s.sole_window(2.10, 0.150).is_none());
    }

    /// The snap must never forbid climbing: a foot reaching for the tread
    /// from the approach floor has to be allowed onto it, not pinned behind
    /// the riser. Expected values are read off the geometry by hand.
    #[test]
    fn the_snap_lets_a_foot_climb_onto_the_tread() {
        let s = Stairs::parse("1,0.060,0.211,2.00", 3.0);
        // Already on the tread: untouched.
        let (x, ok) = s.snap_to_sole_window(2.10, 0.085);
        assert!(ok && (x - 2.10).abs() < 1e-12, "{x}");
        // Well back on the approach floor: untouched (the floor is
        // unbounded below and ends at 1.915).
        let (x, ok) = s.snap_to_sole_window(1.50, 0.085);
        assert!(ok && (x - 1.50).abs() < 1e-12, "{x}");
        // Straddling the nosing at 2.00: the dead zone is (1.915, 2.085).
        // 2.07 is 15 mm from the tread window and 155 mm from the floor's,
        // so it goes UP, not back.
        let (x, ok) = s.snap_to_sole_window(2.07, 0.085);
        assert!(!ok && (x - 2.085).abs() < 1e-12, "{x}");
        // 1.93 is 15 mm past the floor's edge and 155 mm short of the
        // tread's, so it goes back.
        let (x, ok) = s.snap_to_sole_window(1.93, 0.085);
        assert!(!ok && (x - 1.915).abs() < 1e-12, "{x}");
    }
    use super::*;

    /// The example's old inline `h` (kept verbatim) against the moved one.
    #[test]
    fn h_matches_the_example_expression_bit_for_bit() {
        fn old_h(n: usize, riser: f64, tread: f64, start: f64, x: f64) -> f64 {
            let base = if riser < 0.0 { n as f64 * -riser } else { 0.0 };
            let k = if x < start { 0 } else { ((x - start) / tread) as i64 + 1 };
            base + k.clamp(0, n as i64) as f64 * riser
        }
        for &(n, riser, tread, start) in &[
            (1usize, 0.020, 0.211, 2.00),
            (1, 0.060, 0.211, 2.00),
            (1, 0.052, 0.110, 0.30),
            (3, -0.052, 0.110, 0.30),
            (4, 0.080, 0.25, 1.0),
        ] {
            let st = Stairs { n, riser, tread, start, landing_len: 0.60, down_n: 0, slope: 0.0, slope_len: 0.0 };
            let mut x = -0.5;
            while x < 6.0 {
                assert_eq!(st.h(x).to_bits(), old_h(n, riser, tread, start, x).to_bits(), "x={x}");
                x += 0.00731;
            }
        }
    }

    #[test]
    fn spans_are_contiguous() {
        let st = Stairs::parse("3,0.05,0.2,1.0", 2.0);
        let near = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(near(st.tread_span(1).0, 1.0) && near(st.tread_span(1).1, 1.2));
        assert!(near(st.tread_span(3).1, st.landing_start()));
        assert!(near(st.landing_span().0, 1.6) && near(st.landing_span().1, 3.6));
        assert!(near(st.top(), 0.15));
    }

    #[test]
    fn up_and_over_course_heights() {
        let mut st = Stairs::parse("2,0.05,0.2,1.0", 1.0);
        let flat = st;
        st.down_n = 2;
        // Up flight and landing are unchanged by the down flight.
        for x in [0.5, 1.1, 1.3, 1.5, 2.39] {
            assert_eq!(st.h(x), flat.h(x));
        }
        assert!((st.h(2.41) - 0.05).abs() < 1e-12); // landing_end = 2.4
        assert!(st.h(2.61).abs() < 1e-12);
        assert!(st.h(5.0).abs() < 1e-12);
    }
}
