//! `[decision] mode = "n_of_m"`: the sequential accept/reject state machine.
//!
//! Each quality frame contributes one score (its template score, see
//! `scoring.rs`). The decider accepts as soon as `n` of the last `m` scores are
//! at or above the threshold AND liveness has passed. It rejects early when
//! liveness failed, or when `n` can no longer be reached within the remaining
//! frame budget (`budget` quality frames in total; the wall-clock bound is the
//! caller's authentication timeout, which ends the run as "not matched").
//!
//! Pure state machine: no I/O, no clock, deterministic.

use std::collections::VecDeque;

/// Liveness verdict as known when a frame is scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LivenessState {
    /// Not decided yet (multi-frame liveness still collecting).
    Pending,
    Passed,
    Failed,
}

/// Why the decider gave up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    LivenessFailed,
    /// `n` frames above the threshold are no longer reachable in the budget.
    CannotReachN,
    /// Enough frames matched but liveness never reached a pass in the budget.
    LivenessUndecided,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Step {
    Continue,
    /// `similarity` is the mean of the window scores that reached the threshold.
    Accept { similarity: f32 },
    Reject(RejectReason),
}

pub struct SequentialDecider {
    threshold: f32,
    n: usize,
    m: usize,
    budget: usize,
    seen: usize,
    window: VecDeque<f32>,
}

impl SequentialDecider {
    /// Requires `1 <= n <= m` and `n <= budget` (validated by the config).
    pub fn new(threshold: f32, n: usize, m: usize, budget: usize) -> Self {
        Self {
            threshold,
            n,
            m,
            budget,
            seen: 0,
            window: VecDeque::with_capacity(m),
        }
    }

    fn hit(&self, score: f32) -> bool {
        score >= self.threshold
    }

    fn hits(&self) -> usize {
        self.window.iter().filter(|s| self.hit(**s)).count()
    }

    /// Frames scored so far.
    pub fn frames_seen(&self) -> usize {
        self.seen
    }

    /// Highest score seen in the current window (0.0 if none), for reporting.
    pub fn best_in_window(&self) -> f32 {
        self.window.iter().copied().fold(0.0f32, f32::max)
    }

    /// Most hits the window could hold if every remaining frame were a hit.
    fn best_reachable_hits(&self) -> usize {
        let remaining = self.budget.saturating_sub(self.seen);
        let kept = self.m - remaining.min(self.m);
        let skip = self.window.len().saturating_sub(kept);
        let carried = self
            .window
            .iter()
            .skip(skip)
            .filter(|s| self.hit(**s))
            .count();
        carried + remaining.min(self.m)
    }

    /// Feed the next quality frame's score together with the current liveness
    /// state. Do not call again after `Accept` or `Reject`.
    pub fn push(&mut self, score: f32, liveness: LivenessState) -> Step {
        self.seen += 1;
        if self.window.len() == self.m {
            self.window.pop_front();
        }
        self.window.push_back(score);

        if liveness == LivenessState::Failed {
            return Step::Reject(RejectReason::LivenessFailed);
        }
        if self.hits() >= self.n && liveness == LivenessState::Passed {
            let (sum, count) = self
                .window
                .iter()
                .filter(|s| self.hit(**s))
                .fold((0.0f32, 0usize), |(s, c), v| (s + v, c + 1));
            return Step::Accept {
                similarity: sum / count as f32,
            };
        }
        if self.hits() >= self.n {
            // Matching enough, only liveness is outstanding.
            return if self.seen >= self.budget {
                Step::Reject(RejectReason::LivenessUndecided)
            } else {
                Step::Continue
            };
        }
        if self.best_reachable_hits() < self.n {
            return Step::Reject(RejectReason::CannotReachN);
        }
        Step::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use LivenessState::{Failed, Passed, Pending};

    const T: f32 = 0.4;
    const HI: f32 = 0.8;
    const LO: f32 = 0.1;

    fn run(d: &mut SequentialDecider, scores: &[f32], live: LivenessState) -> Vec<Step> {
        let mut out = Vec::new();
        for &s in scores {
            let step = d.push(s, live);
            out.push(step);
            if step != Step::Continue {
                break;
            }
        }
        out
    }

    #[test]
    fn accepts_at_the_third_matching_frame() {
        let mut d = SequentialDecider::new(T, 3, 5, 10);
        let steps = run(&mut d, &[HI, HI, HI, HI], Passed);
        assert_eq!(steps.len(), 3);
        assert!(matches!(steps[2], Step::Accept { similarity } if (similarity - HI).abs() < 1e-6));
        assert_eq!(d.frames_seen(), 3);
    }

    #[test]
    fn isolated_spikes_never_accept() {
        // One hit in every three frames: at most 2 of any 5 are hits.
        let mut d = SequentialDecider::new(T, 3, 5, 10);
        let scores = [HI, LO, LO, HI, LO, LO, HI, LO, LO, HI];
        let steps = run(&mut d, &scores, Passed);
        assert!(!steps.iter().any(|s| matches!(s, Step::Accept { .. })));
        assert_eq!(*steps.last().unwrap(), Step::Reject(RejectReason::CannotReachN));
    }

    #[test]
    fn hits_must_fall_inside_the_window() {
        // Hits at frames 1, 2 then a gap longer than the window, then one more:
        // never 3 within 5.
        let mut d = SequentialDecider::new(T, 3, 5, 12);
        let scores = [HI, HI, LO, LO, LO, LO, HI, LO, LO, LO, LO, LO];
        let steps = run(&mut d, &scores, Passed);
        assert!(!steps.iter().any(|s| matches!(s, Step::Accept { .. })));
    }

    #[test]
    fn early_reject_when_n_is_unreachable_in_the_budget() {
        // Budget 5, n 3: after three misses at most 2 frames remain.
        let mut d = SequentialDecider::new(T, 3, 5, 5);
        assert_eq!(d.push(LO, Passed), Step::Continue);
        assert_eq!(d.push(LO, Passed), Step::Continue);
        assert_eq!(
            d.push(LO, Passed),
            Step::Reject(RejectReason::CannotReachN)
        );
    }

    #[test]
    fn budget_exhaustion_without_n_hits_rejects() {
        let mut d = SequentialDecider::new(T, 2, 3, 4);
        let steps = run(&mut d, &[HI, LO, LO, LO], Passed);
        assert_eq!(*steps.last().unwrap(), Step::Reject(RejectReason::CannotReachN));
    }

    #[test]
    fn failed_liveness_rejects_immediately_even_with_matching_frames() {
        let mut d = SequentialDecider::new(T, 1, 1, 10);
        assert_eq!(d.push(HI, Failed), Step::Reject(RejectReason::LivenessFailed));
    }

    #[test]
    fn pending_liveness_never_accepts_and_resolves_later() {
        let mut d = SequentialDecider::new(T, 2, 3, 10);
        assert_eq!(d.push(HI, Pending), Step::Continue);
        assert_eq!(d.push(HI, Pending), Step::Continue);
        assert!(matches!(d.push(HI, Passed), Step::Accept { .. }));
    }

    #[test]
    fn pending_liveness_at_budget_end_is_a_distinct_reject() {
        let mut d = SequentialDecider::new(T, 1, 1, 2);
        assert_eq!(d.push(HI, Pending), Step::Continue);
        assert_eq!(
            d.push(HI, Pending),
            Step::Reject(RejectReason::LivenessUndecided)
        );
    }

    #[test]
    fn n_equal_m_needs_a_full_streak() {
        let mut d = SequentialDecider::new(T, 3, 3, 10);
        let steps = run(&mut d, &[HI, HI, LO, HI, HI, HI], Passed);
        assert!(matches!(steps.last().unwrap(), Step::Accept { .. }));
        assert_eq!(d.frames_seen(), 6);
    }

    /// Deterministic xorshift so the property loop is seeded and reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Property (seeded loop): over random sequences and liveness traces the
    /// decider (1) accepts only when >= n of the last m scores are hits and
    /// liveness is Passed on that frame, (2) never accepts after a Failed,
    /// (3) rejects as CannotReachN only if n is truly unreachable.
    #[test]
    fn property_never_accepts_without_n_hits_or_liveness_pass() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for _case in 0..2000 {
            let m = 1 + rng.below(6) as usize;
            let n = 1 + rng.below(m as u64) as usize;
            let budget = n + rng.below(10) as usize;
            let mut d = SequentialDecider::new(T, n, m, budget);
            let mut history: Vec<f32> = Vec::new();
            let mut failed_seen = false;
            for _ in 0..budget + 3 {
                let score = if rng.below(2) == 0 { HI } else { LO };
                let live = match rng.below(4) {
                    0 => Pending,
                    1 | 2 => Passed,
                    _ => Failed,
                };
                failed_seen |= live == Failed;
                history.push(score);
                let step = d.push(score, live);
                let window_hits = history
                    .iter()
                    .rev()
                    .take(m)
                    .filter(|s| **s >= T)
                    .count();
                match step {
                    Step::Accept { .. } => {
                        assert!(window_hits >= n, "accepted with {window_hits} < {n} hits");
                        assert_eq!(live, Passed, "accepted without liveness pass");
                        assert!(!failed_seen, "accepted after a liveness failure");
                        assert!(history.len() >= n);
                        break;
                    }
                    Step::Reject(RejectReason::CannotReachN) => {
                        let remaining = budget.saturating_sub(history.len());
                        let best = history
                            .iter()
                            .rev()
                            .take(m - remaining.min(m))
                            .filter(|s| **s >= T)
                            .count()
                            + remaining.min(m);
                        assert!(best < n, "early reject although {best} >= {n} reachable");
                        break;
                    }
                    Step::Reject(_) => break,
                    Step::Continue => {}
                }
            }
        }
    }
}
