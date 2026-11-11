//! Decisions on a POOL of adjacent frames, as pure functions.
//!
//! A camera delivers video, so `mean`-mode liveness and identity are judged
//! on a window of temporally adjacent frames instead of one deciding frame:
//!
//! * [`select_tail`]: the last `size` scored frames must be contiguous in
//!   camera capture order (no gap larger than `[liveness] pool_max_gap_frames`
//!   camera frames), otherwise the attempt has too few usable frames;
//! * [`judge_liveness`]: robust aggregate of the per-frame depth relief
//!   (hard floor for every frame, minimum pass fraction, median);
//! * [`judge_identity`]: per-frame cosine consistency (floor for every frame,
//!   mean at the threshold, bounded spread).
//!
//! No camera or model code; `ipc.rs` feeds the numbers and applies the verdicts.
//! The absolute depth threshold is an input and is never changed here.

use doorman_shared::config::{Config, POOL_HARD_FLOOR_FRACTION};

/// Tolerance for the pass-fraction arithmetic in `f32`.
const FRACTION_EPS: f32 = 1e-4;

/// Failing criterion of the pool liveness verdict (stable names for logs).
pub const CRITERION_HARD_FLOOR: &str = "pool_hard_floor";
pub const CRITERION_PASS_FRACTION: &str = "pool_pass_fraction";
pub const CRITERION_MEDIAN: &str = "pool_median";
/// Failing criterion of the pool identity verdict.
pub const CRITERION_IDENTITY_FLOOR: &str = "pool_identity_floor";
pub const CRITERION_IDENTITY_MEAN: &str = "pool_identity_mean";
pub const CRITERION_IDENTITY_SPREAD: &str = "pool_identity_spread";

/// Pool settings of one `mean`-mode attempt, resolved from the config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoolPolicy {
    /// Frames in the pool (`min(pool_frames, auth_frames)`, at least 1).
    pub size: usize,
    /// Largest camera-sequence difference between neighbouring pool frames.
    pub max_gap: u64,
    /// Absolute depth threshold (`authentication.liveness_depth_threshold`).
    pub depth_threshold: f32,
    pub min_pass_fraction: f32,
    /// Relief below which a single frame fails the whole pool.
    pub hard_floor: f32,
    /// `decision.frame_identity_floor` (0 = off).
    pub identity_floor: f32,
    pub max_score_spread: f32,
    /// Match threshold the mean per-frame cosine must reach.
    pub similarity_threshold: f32,
}

impl PoolPolicy {
    /// `None` when the pool is switched off (`pool_frames = 0`).
    pub fn from_config(config: &Config, similarity_threshold: f32) -> Option<Self> {
        let l = &config.liveness;
        if l.pool_frames == 0 {
            return None;
        }
        let depth_threshold = config.authentication.liveness_depth_threshold;
        let hard_floor = if l.pool_hard_floor > 0.0 {
            l.pool_hard_floor
        } else {
            depth_threshold * POOL_HARD_FLOOR_FRACTION
        };
        Some(Self {
            size: l.pool_frames.min(config.authentication.auth_frames).max(1),
            max_gap: u64::from(l.pool_max_gap_frames),
            depth_threshold,
            min_pass_fraction: l.pool_min_pass_fraction,
            hard_floor,
            identity_floor: config.decision.frame_identity_floor,
            max_score_spread: config.decision.pool_max_score_spread,
            similarity_threshold,
        })
    }
}

/// Why the collected frames cannot form a pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolShape {
    /// Fewer scored frames than the pool size.
    TooFew { have: usize, need: usize },
    /// Two neighbouring pool frames are further apart than the allowed gap, or
    /// their sequence numbers do not increase (a camera restart).
    NonContiguous { after_sequence: u64 },
}

/// Start index of the pool: the last `size` of `sequences` (camera sequence
/// numbers of the scored frames, in scoring order), if they are contiguous.
pub fn select_tail(sequences: &[u64], size: usize, max_gap: u64) -> Result<usize, PoolShape> {
    if sequences.len() < size {
        return Err(PoolShape::TooFew {
            have: sequences.len(),
            need: size,
        });
    }
    let start = sequences.len() - size;
    for pair in sequences[start..].windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if b <= a || b - a > max_gap {
            return Err(PoolShape::NonContiguous { after_sequence: a });
        }
    }
    Ok(start)
}

/// Largest sequence difference between neighbouring frames (0 for < 2 frames).
pub fn max_sequence_gap(sequences: &[u64]) -> u64 {
    sequences
        .windows(2)
        .map(|p| p[1].saturating_sub(p[0]))
        .max()
        .unwrap_or(0)
}

/// Aggregate of the per-frame reliefs of a pool.
#[derive(Debug, Clone, PartialEq)]
pub struct LivenessSummary {
    pub reliefs: Vec<f32>,
    pub median: f32,
    pub min: f32,
    /// Frames at or above the depth threshold.
    pub passes: usize,
}

/// Verdict of [`judge_liveness`]: the summary, and the criterion that failed.
#[derive(Debug, Clone, PartialEq)]
pub struct LivenessVerdict {
    pub summary: LivenessSummary,
    /// `None` = the pool is live.
    pub failed: Option<&'static str>,
}

fn median(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
    }
}

/// Frames needed to pass for `fraction` of `n` (rounded up).
fn required_passes(fraction: f32, n: usize) -> usize {
    ((fraction * n as f32) - FRACTION_EPS).ceil().max(1.0) as usize
}

/// Robust liveness verdict over the per-frame reliefs of a pool. Fails, in this
/// order, when
///
/// 1. any relief is below `hard_floor` (or not a number): a photo frame cannot
///    hide among live ones;
/// 2. fewer than `min_pass_fraction` of the frames reach the depth threshold;
/// 3. the median relief is below the depth threshold.
///
/// An empty pool fails closed. The threshold is never lowered.
pub fn judge_liveness(reliefs: &[f32], policy: &PoolPolicy) -> LivenessVerdict {
    let passes = reliefs
        .iter()
        .filter(|r| **r >= policy.depth_threshold)
        .count();
    let summary = LivenessSummary {
        reliefs: reliefs.to_vec(),
        median: median(reliefs),
        min: reliefs.iter().copied().fold(f32::INFINITY, f32::min),
        passes,
    };
    let failed =
        if reliefs.is_empty() || reliefs.iter().any(|r| r.is_nan() || *r < policy.hard_floor) {
            Some(CRITERION_HARD_FLOOR)
        } else if passes < required_passes(policy.min_pass_fraction, reliefs.len()) {
            Some(CRITERION_PASS_FRACTION)
        } else if summary.median < policy.depth_threshold {
            Some(CRITERION_MEDIAN)
        } else {
            None
        };
    LivenessVerdict { summary, failed }
}

/// Identity consistency of a pool: `Err(criterion)` when
///
/// * any per-frame cosine is below `identity_floor` (when it is on) or NaN;
/// * the mean per-frame cosine is below the match threshold;
/// * best minus worst cosine exceeds `max_score_spread` (mixed identities).
pub fn judge_identity(scores: &[f32], policy: &PoolPolicy) -> Result<f32, &'static str> {
    if scores.is_empty() || scores.iter().any(|s| s.is_nan()) {
        return Err(CRITERION_IDENTITY_FLOOR);
    }
    if policy.identity_floor > 0.0 && scores.iter().any(|s| *s < policy.identity_floor) {
        return Err(CRITERION_IDENTITY_FLOOR);
    }
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    if mean < policy.similarity_threshold {
        return Err(CRITERION_IDENTITY_MEAN);
    }
    let best = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let worst = scores.iter().copied().fold(f32::INFINITY, f32::min);
    if best - worst > policy.max_score_spread {
        return Err(CRITERION_IDENTITY_SPREAD);
    }
    Ok(mean)
}

/// One-line description of a pool for the decision log and shadow records.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PoolReport {
    pub size: usize,
    /// Camera sequence numbers of the pool frames.
    pub sequences: Vec<u64>,
    pub max_gap: u64,
    /// Per-frame depth relief, in pool order.
    pub reliefs: Vec<f32>,
    pub relief_median: Option<f32>,
    pub relief_min: Option<f32>,
    pub passes: usize,
    /// Per-frame template score, in pool order.
    pub scores: Vec<f32>,
    /// Failing criterion (liveness or identity); `None` = the pool passed.
    pub failed: Option<&'static str>,
}

impl PoolReport {
    pub fn log_fields(&self) -> String {
        let fmt = |v: &[f32]| {
            v.iter()
                .map(|x| format!("{x:.4}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            "pool={} seq={:?} max_gap={} relief=[{}] relief_median={} relief_min={} pass={}/{} cos=[{}] failed={}",
            self.size,
            self.sequences,
            self.max_gap,
            fmt(&self.reliefs),
            self.relief_median
                .map_or("n/a".to_string(), |m| format!("{m:.4}")),
            self.relief_min
                .map_or("n/a".to_string(), |m| format!("{m:.4}")),
            self.passes,
            self.reliefs.len(),
            fmt(&self.scores),
            self.failed.unwrap_or("none"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: f32 = 0.05;

    fn policy() -> PoolPolicy {
        PoolPolicy::from_config(&Config::default(), 0.4).expect("pool on by default")
    }

    #[test]
    fn defaults_resolve_to_the_documented_policy() {
        let p = policy();
        assert_eq!(p.size, 5);
        assert_eq!(p.max_gap, 3);
        assert_eq!(p.depth_threshold, T);
        assert!((p.hard_floor - 0.025).abs() < 1e-9, "0.5 x threshold");
        assert_eq!(p.min_pass_fraction, 0.6);
    }

    #[test]
    fn pool_is_off_at_zero_and_clamped_to_auth_frames() {
        let mut c = Config::default();
        c.liveness.pool_frames = 0;
        assert!(PoolPolicy::from_config(&c, 0.4).is_none());
        c.liveness.pool_frames = 9;
        c.authentication.auth_frames = 3;
        assert_eq!(PoolPolicy::from_config(&c, 0.4).unwrap().size, 3);
        c.liveness.pool_hard_floor = 0.04;
        assert_eq!(PoolPolicy::from_config(&c, 0.4).unwrap().hard_floor, 0.04);
    }

    #[test]
    fn genuine_pool_passes() {
        let v = judge_liveness(&[0.2, 0.18, 0.21, 0.19, 0.2], &policy());
        assert_eq!(v.failed, None);
        assert_eq!(v.summary.passes, 5);
    }

    #[test]
    fn photo_pool_fails() {
        let v = judge_liveness(&[0.01; 5], &policy());
        assert_eq!(v.failed, Some(CRITERION_HARD_FLOOR));
    }

    #[test]
    fn near_threshold_photo_pool_fails_on_median_or_fraction() {
        // All above the hard floor (0.025) but below the threshold: a flat
        // print that is not flat enough for the floor still fails.
        let v = judge_liveness(&[0.04; 5], &policy());
        assert_eq!(v.failed, Some(CRITERION_PASS_FRACTION));
    }

    #[test]
    fn mixed_photo_and_live_fails() {
        // Two photo frames among three live ones: hard floor.
        let v = judge_liveness(&[0.2, 0.01, 0.2, 0.01, 0.2], &policy());
        assert_eq!(v.failed, Some(CRITERION_HARD_FLOOR));
        // Photo-ish frames above the floor: pass fraction 3/5 would pass, but
        // 2 of 5 below the threshold with a live median is still accepted
        // only when each is above the floor.
        let v = judge_liveness(&[0.2, 0.03, 0.2, 0.03, 0.2], &policy());
        assert_eq!(v.failed, None, "3 of 5 live, none below the floor");
        // 3 photo-like (above floor) and 2 live: fraction fails.
        let v = judge_liveness(&[0.2, 0.03, 0.03, 0.03, 0.2], &policy());
        assert_eq!(v.failed, Some(CRITERION_PASS_FRACTION));
    }

    #[test]
    fn single_borderline_frame_is_tolerated_only_above_the_floor() {
        let live = [0.2, 0.2, 0.2, 0.2];
        let mut r = live.to_vec();
        r.push(0.04); // below the threshold, above the floor
        assert_eq!(judge_liveness(&r, &policy()).failed, None);
        let mut r = live.to_vec();
        r.push(0.02); // below the floor
        assert_eq!(
            judge_liveness(&r, &policy()).failed,
            Some(CRITERION_HARD_FLOOR)
        );
    }

    #[test]
    fn the_absolute_threshold_is_not_lowered() {
        // The thin genuine margin seen on the host: the pool fails like the
        // single frame did.
        let v = judge_liveness(&[0.037, 0.043, 0.04, 0.041, 0.039], &policy());
        assert_eq!(v.failed, Some(CRITERION_PASS_FRACTION));
        assert!(v.summary.median < T);
        // Exactly at the threshold passes.
        assert_eq!(judge_liveness(&[T; 5], &policy()).failed, None);
    }

    #[test]
    fn median_criterion_is_reached_when_the_fraction_allows_it() {
        // Four frames, one passes: the fraction (0.5 -> 2 passes) fails first;
        // with a looser fraction the median (0.03) is what fails.
        let mut p = policy();
        p.size = 4;
        p.min_pass_fraction = 0.5;
        let v = judge_liveness(&[0.03, 0.03, 0.06, 0.03], &p);
        assert_eq!(v.failed, Some(CRITERION_PASS_FRACTION));
        p.min_pass_fraction = 0.25;
        let v = judge_liveness(&[0.03, 0.03, 0.06, 0.03], &p);
        assert_eq!(v.failed, Some(CRITERION_MEDIAN));
    }

    #[test]
    fn nan_and_empty_fail_closed() {
        assert_eq!(
            judge_liveness(&[0.2, f32::NAN, 0.2], &policy()).failed,
            Some(CRITERION_HARD_FLOOR)
        );
        assert_eq!(
            judge_liveness(&[], &policy()).failed,
            Some(CRITERION_HARD_FLOOR)
        );
    }

    #[test]
    fn required_passes_rounds_up() {
        assert_eq!(required_passes(0.6, 5), 3);
        assert_eq!(required_passes(0.6, 3), 2);
        assert_eq!(required_passes(1.0, 5), 5);
        assert_eq!(required_passes(0.5, 4), 2);
    }

    #[test]
    fn contiguous_tail_is_accepted() {
        assert_eq!(select_tail(&[10, 11, 12, 13, 14], 5, 3), Ok(0));
        assert_eq!(select_tail(&[1, 5, 6, 8, 9, 11], 5, 3), Ok(1), "gaps of 3");
        assert_eq!(select_tail(&[7], 1, 3), Ok(0));
    }

    #[test]
    fn non_contiguous_pool_is_rejected() {
        assert_eq!(
            select_tail(&[10, 11, 20, 21, 22], 5, 3),
            Err(PoolShape::NonContiguous { after_sequence: 11 })
        );
        // Only the tail matters: an old gap before the pool is fine.
        assert_eq!(select_tail(&[1, 50, 51, 52], 3, 3), Ok(1));
        // A sequence reset (camera restart) is not contiguous.
        assert!(matches!(
            select_tail(&[5, 6, 1, 2], 4, 3),
            Err(PoolShape::NonContiguous { .. })
        ));
        // A repeated frame is not an advance.
        assert!(select_tail(&[5, 5, 6], 3, 3).is_err());
    }

    #[test]
    fn too_few_frames_are_reported() {
        assert_eq!(
            select_tail(&[1, 2], 5, 3),
            Err(PoolShape::TooFew { have: 2, need: 5 })
        );
    }

    #[test]
    fn max_gap_helper() {
        assert_eq!(max_sequence_gap(&[]), 0);
        assert_eq!(max_sequence_gap(&[4]), 0);
        assert_eq!(max_sequence_gap(&[1, 2, 5, 6]), 3);
    }

    #[test]
    fn identity_consistent_pool_passes_and_returns_the_mean() {
        let mean = judge_identity(&[0.78, 0.8, 0.75, 0.82, 0.79], &policy()).unwrap();
        assert!((mean - 0.788).abs() < 1e-4);
    }

    #[test]
    fn identity_pool_failures() {
        let p = policy();
        assert_eq!(
            judge_identity(&[0.8, 0.8, 0.29, 0.8, 0.8], &p),
            Err(CRITERION_IDENTITY_FLOOR)
        );
        assert_eq!(
            judge_identity(&[0.35, 0.36, 0.37, 0.35, 0.36], &p),
            Err(CRITERION_IDENTITY_MEAN)
        );
        // Two identities: every frame above the floor, the mean above the
        // threshold, but a wide spread.
        assert_eq!(
            judge_identity(&[0.9, 0.9, 0.9, 0.35, 0.35], &p),
            Err(CRITERION_IDENTITY_SPREAD)
        );
        assert_eq!(
            judge_identity(&[0.8, f32::NAN], &p),
            Err(CRITERION_IDENTITY_FLOOR)
        );
        assert_eq!(judge_identity(&[], &p), Err(CRITERION_IDENTITY_FLOOR));
    }

    #[test]
    fn identity_floor_zero_disables_only_the_floor() {
        let mut p = policy();
        p.identity_floor = 0.0;
        p.max_score_spread = 2.0;
        let mean = judge_identity(&[0.9, 0.9, 0.9, 0.9, 0.0], &p).unwrap();
        assert!((mean - 0.72).abs() < 1e-5);
        assert_eq!(
            judge_identity(&[0.3, 0.3, 0.3], &p),
            Err(CRITERION_IDENTITY_MEAN)
        );
    }

    #[test]
    fn report_line_names_every_number() {
        let r = PoolReport {
            size: 2,
            sequences: vec![3, 4],
            max_gap: 1,
            reliefs: vec![0.2, 0.1],
            relief_median: Some(0.15),
            relief_min: Some(0.1),
            passes: 2,
            scores: vec![0.8, 0.7],
            failed: None,
        };
        let line = r.log_fields();
        for needle in [
            "pool=2",
            "seq=[3, 4]",
            "relief=[0.2000,0.1000]",
            "relief_median=0.1500",
            "pass=2/2",
            "cos=[0.8000,0.7000]",
            "failed=none",
        ] {
            assert!(line.contains(needle), "{needle} in {line}");
        }
    }
}
