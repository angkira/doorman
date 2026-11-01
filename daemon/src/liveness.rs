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

/// A liveness signal independent of depth relief (for example a screen-flash
/// reflectance challenge or a head-pose challenge). Interface only: no
/// implementation exists and `[liveness] signals` rejects every entry until
/// one does. See docs/decision-and-liveness.md for the known weaknesses of the
/// candidates.
pub trait LivenessSignal: Send + Sync {
    /// Name used in `[liveness] signals` and in logs.
    fn name(&self) -> &'static str;

    /// Evaluate the signal on the quality frame and its face box (x, y, w, h,
    /// normalized). An active challenge drives the display or asks the user
    /// to move before answering; it must fail closed (`Unavailable`).
    fn evaluate(
        &self,
        frame: &image::DynamicImage,
        face_bbox: (f32, f32, f32, f32),
    ) -> LivenessOutcome;
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

/// Binds liveness to the frame that completes an accepting decision.
///
/// The first quality frames are liveness-checked before the embeddings that
/// decide the match are collected, so a live face followed by a photo would
/// otherwise pass. Before accepting, the deciding frame (the last quality frame
/// scored) must itself be live: `deciding_frame_checked` says it already was
/// (no second inference); otherwise `recheck` evaluates it. `recheck` yields
/// `None` when the check is disabled. A failed or unavailable recheck is
/// returned as `Err` so the caller rejects fail-closed.
pub async fn confirm_on_deciding_frame<F, Fut>(
    deciding_frame_checked: bool,
    recheck: F,
) -> Result<(), LivenessOutcome>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Option<LivenessOutcome>>,
{
    if deciding_frame_checked {
        return Ok(());
    }
    match recheck().await {
        None | Some(LivenessOutcome::Pass { .. }) => Ok(()),
        Some(not_live) => Err(not_live),
    }
}

/// Picks up to `samples` frames evenly spread over the frames not yet depth-checked.
///
/// `already_checked[i]` is true when frame `i` had a depth check (those are
/// excluded). Returns ascending, distinct frame indices; fewer than `samples`
/// when fewer unchecked frames exist. Each pick is the midpoint of an equal
/// slice of the unchecked frames, so the first and last slices are not biased
/// towards the sequence ends.
pub fn sample_unchecked_frames(already_checked: &[bool], samples: usize) -> Vec<usize> {
    let unchecked: Vec<usize> = already_checked
        .iter()
        .enumerate()
        .filter_map(|(i, &checked)| (!checked).then_some(i))
        .collect();
    let picks = samples.min(unchecked.len());
    // Midpoint of slice k of `picks` equal slices: (k + 1/2) * len / picks.
    (0..picks)
        .map(|k| unchecked[(2 * k + 1) * unchecked.len() / (2 * picks)])
        .collect()
}

/// `mean` mode: depth-checks the sampled frames of an accepting attempt.
///
/// Every frame index picked by [`sample_unchecked_frames`] is evaluated with
/// `check`; the first one that is not a pass is returned as `Err` so the caller
/// rejects fail-closed. `Ok(n)` is the number of depth checks performed (for
/// timing). `check` yields `None` when the check is disabled.
pub async fn confirm_sampled_frames<F, Fut>(
    already_checked: &[bool],
    samples: usize,
    mut check: F,
) -> Result<usize, LivenessOutcome>
where
    F: FnMut(usize) -> Fut,
    Fut: std::future::Future<Output = Option<LivenessOutcome>>,
{
    let mut performed = 0;
    for index in sample_unchecked_frames(already_checked, samples) {
        performed += 1;
        match check(index).await {
            None | Some(LivenessOutcome::Pass { .. }) => {}
            Some(not_live) => return Err(not_live),
        }
    }
    Ok(performed)
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
    let face = if selected.is_empty() {
        reference
    } else {
        &selected[..]
    };

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
        assert_eq!(
            relief_from_patch(ReliefStat::P10P90, &patch, None, &reference),
            1.0
        );
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
                assert_eq!(
                    verdict,
                    Some(expected_pass),
                    "frames {frames} pattern {pattern:b}"
                );
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
        assert_eq!(
            relief_from_patch(ReliefStat::MinMax, &patch, None, &reference),
            1.0
        );
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

    #[test]
    fn sampling_spreads_evenly_and_skips_checked_frames() {
        let none = [false; 8];
        assert_eq!(sample_unchecked_frames(&none, 0), Vec::<usize>::new());
        assert_eq!(sample_unchecked_frames(&none, 1), vec![4]);
        assert_eq!(sample_unchecked_frames(&none, 2), vec![2, 6]);
        assert_eq!(sample_unchecked_frames(&none, 4), vec![1, 3, 5, 7]);
        // Asking for more than exist yields every unchecked frame once.
        assert_eq!(
            sample_unchecked_frames(&none, 20),
            (0..8).collect::<Vec<_>>()
        );
        // Frame 0 was depth-checked already: it is never picked.
        let first_checked = [true, false, false, false];
        assert_eq!(sample_unchecked_frames(&first_checked, 3), vec![1, 2, 3]);
        assert!(sample_unchecked_frames(&[true, true], 2).is_empty());
        assert!(sample_unchecked_frames(&[], 2).is_empty());
    }

    #[tokio::test]
    async fn sampled_frames_reject_on_first_failure_and_count_checks() {
        let frames = [0.2_f32, 0.2, 0.01, 0.2, 0.2];
        let check = |index: usize| {
            let relief = frames[index];
            async move {
                Some(if relief >= 0.05 {
                    LivenessOutcome::Pass { relief }
                } else {
                    LivenessOutcome::Fail { relief }
                })
            }
        };
        let none = [false; 5];
        // One sample is frame 2, the photo.
        assert!(confirm_sampled_frames(&none, 1, check).await.is_err());
        // Zero samples never check anything.
        assert_eq!(confirm_sampled_frames(&none, 0, check).await, Ok(0));
        // Frame 2 counted as checked: the remaining live frames pass.
        let skip_photo = [false, false, true, false, false];
        assert_eq!(confirm_sampled_frames(&skip_photo, 4, check).await, Ok(4));
    }

    #[tokio::test]
    async fn already_checked_deciding_frame_costs_no_second_inference() {
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let r = confirm_on_deciding_frame(true, || async {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            None
        })
        .await;
        assert!(r.is_ok());
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn unavailable_or_failed_recheck_is_an_error_and_disabled_is_ok() {
        let unavailable = LivenessOutcome::Unavailable {
            reason: "no model".into(),
        };
        assert!(
            confirm_on_deciding_frame(false, || async { Some(unavailable) })
                .await
                .is_err()
        );
        assert!(confirm_on_deciding_frame(false, || async { None })
            .await
            .is_ok());
    }
}
