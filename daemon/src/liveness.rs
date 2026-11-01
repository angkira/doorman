//! Typed result of one depth-liveness evaluation.
//!
//! The relief value is carried so callers can log it, put it in the timing
//! record and calibrate the threshold from it (docs/decision-and-liveness.md).

use doorman_shared::config::ReliefStat;

/// Guards the division by the reference depth range.
const RELIEF_EPS: f32 = 1e-8;
/// Percentiles of the robust relief statistic.
const RELIEF_LOW_PERCENTILE: f32 = 0.10;
const RELIEF_HIGH_PERCENTILE: f32 = 0.90;

/// Outcome of evaluating liveness on one frame. A gate that cannot run is a
/// distinct, fail-closed outcome so "model unavailable" is never confused with
/// "spoof suspected".
#[derive(Debug, Clone, PartialEq)]
pub enum LivenessOutcome {
    /// Relief at or above the threshold.
    Pass { relief: f32 },
    /// Relief below the threshold: suspected spoof.
    Fail { relief: f32 },
    /// The check could not run (model missing, inference error). Fail closed.
    Unavailable { reason: String },
}

/// Error code of a liveness check that could not run (model missing, inference
/// error). Distinct from [`LIVENESS_REJECTED_CODE`] so "unavailable" is never
/// mistaken for "spoof suspected" in logs, `doorman test` output and PAM.
pub const LIVENESS_UNAVAILABLE_CODE: &str = "liveness_unavailable";
/// Error code of a liveness check that ran and rejected the face.
pub const LIVENESS_REJECTED_CODE: &str = "liveness_rejected";

impl LivenessOutcome {
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass { .. })
    }

    /// The relief value, when the check produced one.
    pub fn relief(&self) -> Option<f32> {
        match self {
            Self::Pass { relief } | Self::Fail { relief } => Some(*relief),
            Self::Unavailable { .. } => None,
        }
    }
}

/// Progress of a multi-frame liveness evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum LivenessProgress {
    /// More frames are needed.
    Pending,
    /// Final verdict, with the median relief of the frames evaluated.
    Decided(LivenessOutcome),
}

/// Combines per-frame outcomes into one verdict (`[liveness] aggregate =
/// "median"`). The aggregate passes when the median relief of `frames` frames
/// is at or above the threshold; for an odd `frames` that is exactly "a
/// majority (`frames / 2 + 1`) of the frames pass", so the verdict is final as
/// soon as one side holds a majority. For the recommended 3 frames this is the
/// early exit "the first two frames agree". A frame whose check is unavailable
/// decides at once as unavailable (fail closed).
pub struct LivenessAggregator {
    majority: usize,
    outcomes: Vec<LivenessOutcome>,
}

impl LivenessAggregator {
    /// `frames` must be odd (validated by the config).
    pub fn new(frames: usize) -> Self {
        Self {
            majority: frames / 2 + 1,
            outcomes: Vec::with_capacity(frames),
        }
    }

    /// Frames evaluated so far.
    pub fn evaluated(&self) -> usize {
        self.outcomes.len()
    }

    pub fn push(&mut self, outcome: LivenessOutcome) -> LivenessProgress {
        if matches!(outcome, LivenessOutcome::Unavailable { .. }) {
            return LivenessProgress::Decided(outcome);
        }
        self.outcomes.push(outcome);
        let passes = self.outcomes.iter().filter(|o| o.is_pass()).count();
        let fails = self.outcomes.len() - passes;
        let relief = self.median_relief();
        if passes >= self.majority {
            LivenessProgress::Decided(LivenessOutcome::Pass { relief })
        } else if fails >= self.majority {
            LivenessProgress::Decided(LivenessOutcome::Fail { relief })
        } else {
            LivenessProgress::Pending
        }
    }

    /// Median of the collected reliefs (mean of the middle two for an even count).
    fn median_relief(&self) -> f32 {
        let mut reliefs: Vec<f32> = self.outcomes.iter().filter_map(|o| o.relief()).collect();
        reliefs.sort_by(|a, b| a.total_cmp(b));
        let n = reliefs.len();
        if n == 0 {
            0.0
        } else if n % 2 == 1 {
            reliefs[n / 2]
        } else {
            (reliefs[n / 2 - 1] + reliefs[n / 2]) * 0.5
        }
    }
}

/// Linear-interpolated percentile (`q` in 0..=1) of `values`; 0.0 when empty.
fn percentile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let pos = q * (sorted.len() - 1) as f32;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f32)
}

/// Relief of a face depth patch, in `[0, 1]`, higher = more 3D structure.
///
/// * `patch`: depth values of the face region; `mask` (same length) keeps only
///   the entries that are `true`, `None` keeps all. An empty selection falls
///   back to `reference`, which yields a near-zero relief (as before).
/// * `reference`: the whole depth map, used to normalise the spread.
/// * `ReliefStat::MinMax`: `std(patch) / (max - min of reference)`.
/// * `ReliefStat::P10P90`: `(p90 - p10 of patch) / (p90 - p10 of reference)`.
pub fn relief_from_patch(
    stat: ReliefStat,
    patch: &[f32],
    mask: Option<&[bool]>,
    reference: &[f32],
) -> f32 {
    let selected: Vec<f32> = match mask {
        Some(mask) => patch
            .iter()
            .zip(mask)
            .filter_map(|(v, keep)| keep.then_some(*v))
            .collect(),
        None => patch.to_vec(),
    };
    let face = if selected.is_empty() { reference } else { &selected[..] };

    let relief = match stat {
        ReliefStat::MinMax => {
            let (min, max) = reference
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
            let count = face.len() as f32;
            let mean = face.iter().sum::<f32>() / count;
            let var = face.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / count;
            var.sqrt() / ((max - min) + RELIEF_EPS)
        }
        ReliefStat::P10P90 => {
            let spread = |v: &[f32]| {
                percentile(v, RELIEF_HIGH_PERCENTILE) - percentile(v, RELIEF_LOW_PERCENTILE)
            };
            spread(face) / (spread(reference) + RELIEF_EPS)
        }
    };
    relief.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whole map: 100 samples ramping 0..=99, so min/max range is 99 and
    /// p10..p90 range is 79.2.
    fn ramp() -> Vec<f32> {
        (0..100).map(|v| v as f32).collect()
    }

    #[test]
    fn minmax_matches_the_original_std_over_global_range() {
        let reference = ramp();
        let patch = [10.0, 20.0, 30.0, 40.0];
        let std = (125.0f32).sqrt(); // population std of the patch
        let expected = std / (99.0 + RELIEF_EPS);
        let got = relief_from_patch(ReliefStat::MinMax, &patch, None, &reference);
        assert!((got - expected).abs() < 1e-6, "{got} vs {expected}");
    }

    #[test]
    fn flat_patch_has_zero_relief_for_both_stats() {
        let reference = ramp();
        let flat = [42.0; 16];
        for stat in [ReliefStat::MinMax, ReliefStat::P10P90] {
            assert_eq!(relief_from_patch(stat, &flat, None, &reference), 0.0);
        }
    }

    #[test]
    fn p10_p90_ignores_a_single_outlier_pixel_that_minmax_counts() {
        let reference = ramp();
        let mut patch = vec![50.0f32; 50];
        patch[0] = 99.0; // one hot pixel
        let mm = relief_from_patch(ReliefStat::MinMax, &patch, None, &reference);
        let pp = relief_from_patch(ReliefStat::P10P90, &patch, None, &reference);
        assert!(mm > 0.05, "minmax is moved by one pixel: {mm}");
        assert_eq!(pp, 0.0, "p10..p90 of the patch is still flat");
    }

    #[test]
    fn p10_p90_value_on_a_known_patch() {
        let reference = ramp();
        let patch: Vec<f32> = (0..=10).map(|v| v as f32 * 10.0).collect(); // 0..100 step 10
        // p10 = 10, p90 = 90 -> 80 / 79.2, clamped to 1.
        assert_eq!(relief_from_patch(ReliefStat::P10P90, &patch, None, &reference), 1.0);
        let narrow: Vec<f32> = (0..=10).map(|v| 40.0 + v as f32).collect(); // 40..50
        // p10 = 41, p90 = 49 -> 8 / 79.2.
        let got = relief_from_patch(ReliefStat::P10P90, &narrow, None, &reference);
        assert!((got - 8.0 / (79.2 + RELIEF_EPS)).abs() < 1e-4, "{got}");
    }

    #[test]
    fn mask_selects_values_and_empty_selection_falls_back_to_reference() {
        let reference = ramp();
        let patch = [0.0, 100.0, 50.0, 50.0];
        let mask = [false, false, true, true];
        assert_eq!(
            relief_from_patch(ReliefStat::MinMax, &patch, Some(&mask), &reference),
            0.0,
            "only the two equal values remain"
        );
        let none = [false; 4];
        let fallback = relief_from_patch(ReliefStat::MinMax, &patch, Some(&none), &reference);
        let whole = relief_from_patch(ReliefStat::MinMax, &reference, None, &reference);
        assert_eq!(fallback, whole);
    }

    fn pass(relief: f32) -> LivenessOutcome {
        LivenessOutcome::Pass { relief }
    }
    fn fail(relief: f32) -> LivenessOutcome {
        LivenessOutcome::Fail { relief }
    }

    #[test]
    fn single_frame_decides_immediately_like_the_original_gate() {
        let mut agg = LivenessAggregator::new(1);
        assert_eq!(agg.push(pass(0.2)), LivenessProgress::Decided(pass(0.2)));
        let mut agg = LivenessAggregator::new(1);
        assert_eq!(agg.push(fail(0.01)), LivenessProgress::Decided(fail(0.01)));
    }

    #[test]
    fn three_frames_exit_early_when_the_first_two_agree() {
        let mut agg = LivenessAggregator::new(3);
        assert_eq!(agg.push(pass(0.30)), LivenessProgress::Pending);
        assert_eq!(agg.push(pass(0.10)), LivenessProgress::Decided(pass(0.20)));
        assert_eq!(agg.evaluated(), 2);

        let mut agg = LivenessAggregator::new(3);
        agg.push(fail(0.01));
        assert_eq!(agg.push(fail(0.03)), LivenessProgress::Decided(fail(0.02)));
    }

    #[test]
    fn three_frames_disagreeing_use_the_third_as_tiebreak_and_match_the_median() {
        let mut agg = LivenessAggregator::new(3);
        agg.push(pass(0.30));
        assert_eq!(agg.push(fail(0.02)), LivenessProgress::Pending);
        // Median of {0.30, 0.02, 0.04} is 0.04: below a 0.05 threshold -> fail.
        assert_eq!(agg.push(fail(0.04)), LivenessProgress::Decided(fail(0.04)));
    }

    /// Property (exhaustive over pass/fail patterns): the majority verdict
    /// equals "median relief >= threshold" for every odd frame count.
    #[test]
    fn majority_verdict_equals_the_median_threshold_rule() {
        const THRESHOLD: f32 = 0.05;
        for frames in [1usize, 3, 5, 7] {
            for pattern in 0u32..(1 << frames) {
                let reliefs: Vec<f32> = (0..frames)
                    .map(|i| {
                        let base = if pattern >> i & 1 == 1 { 0.10 } else { 0.01 };
                        base + i as f32 * 0.001
                    })
                    .collect();
                let mut sorted = reliefs.clone();
                sorted.sort_by(|a, b| a.total_cmp(b));
                let expected_pass = sorted[frames / 2] >= THRESHOLD;
                let mut agg = LivenessAggregator::new(frames);
                let mut verdict = None;
                for r in &reliefs {
                    let o = if *r >= THRESHOLD { pass(*r) } else { fail(*r) };
                    if let LivenessProgress::Decided(v) = agg.push(o) {
                        verdict = Some(v.is_pass());
                        break;
                    }
                }
                assert_eq!(verdict, Some(expected_pass), "frames {frames} pattern {pattern:b}");
            }
        }
    }

    #[test]
    fn unavailable_frame_decides_unavailable_at_once() {
        let mut agg = LivenessAggregator::new(5);
        agg.push(pass(0.3));
        let u = LivenessOutcome::Unavailable { reason: "x".into() };
        assert_eq!(agg.push(u.clone()), LivenessProgress::Decided(u));
    }

    #[test]
    fn default_depth_input_size_matches_the_model_profile() {
        assert_eq!(
            crate::ml::DepthPadConfig::DEPTH_ANYTHING_V2.input_size,
            doorman_shared::config::DEFAULT_DEPTH_INPUT_SIZE
        );
    }

    #[test]
    fn result_is_clamped_to_unit_interval() {
        let reference = [0.0, 1.0];
        let patch = [-100.0, 100.0];
        assert_eq!(relief_from_patch(ReliefStat::MinMax, &patch, None, &reference), 1.0);
    }

    #[test]
    fn only_pass_is_pass_and_relief_is_exposed() {
        assert!(LivenessOutcome::Pass { relief: 0.2 }.is_pass());
        assert!(!LivenessOutcome::Fail { relief: 0.01 }.is_pass());
        let u = LivenessOutcome::Unavailable { reason: "x".into() };
        assert!(!u.is_pass());
        assert_eq!(u.relief(), None);
        assert_eq!(LivenessOutcome::Fail { relief: 0.01 }.relief(), Some(0.01));
    }
}
