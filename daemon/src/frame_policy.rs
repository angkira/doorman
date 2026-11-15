//! Per-frame acceptance rules of the decision core, as pure functions.
//!
//! * identity floor: a frame that feeds a `mean` decision must itself resemble
//!   the enrolled face (`[decision] frame_identity_floor`);
//! * multi-face gate (`[recognition] multi_face_policy`);
//! * frame age (`[camera] max_frame_age_ms`);
//! * minimum number of judged-good frames (`[authentication] min_decision_frames`).
//!
//! No camera or model code, so each rule is tested in isolation with injected
//! values; `ipc.rs` applies them inside `run_authenticate`.

use doorman_shared::config::MultiFacePolicy;
use std::time::{Duration, Instant};

/// A frame whose score against the enrolled templates is below `floor` fails
/// the identity floor. `floor <= 0` disables the check. A NaN score fails
/// (closed): it is not a number that reaches the floor.
pub fn below_identity_floor(score: f32, floor: f32) -> bool {
    floor > 0.0 && (score.is_nan() || score < floor)
}

/// Whether a frame captured at `captured` may still be used at `now`.
/// `max_age == 0` disables the guard. A capture time in the future (clock
/// oddity) counts as age zero.
pub fn is_recent(captured: Instant, now: Instant, max_age: Duration) -> bool {
    max_age.is_zero() || now.saturating_duration_since(captured) <= max_age
}

/// Enough judged-good frames collected for a `mean` decision.
pub fn enough_frames(collected: usize, required: usize) -> bool {
    collected >= required
}

/// A second face seen in the same frame: normalized box area and detector score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sighting {
    pub area: f32,
    pub score: f32,
}

/// What to do with a frame that has `others` besides the main face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceChoice {
    /// Use the main (highest-scoring) face.
    Main,
    /// Use `others[i]` instead (policy `largest`).
    Other(usize),
    /// The frame is not accepted (policy `reject`).
    Ambiguous,
}

/// Apply `[recognition] multi_face_policy`. A second face counts when its area
/// is at least `min_area_ratio` of the main face's and its score at least
/// `min_score`.
pub fn choose_face(
    policy: MultiFacePolicy,
    min_area_ratio: f32,
    min_score: f32,
    main: Sighting,
    others: &[Sighting],
) -> FaceChoice {
    if policy == MultiFacePolicy::Off {
        return FaceChoice::Main;
    }
    let counts = |o: &Sighting| o.score >= min_score && o.area >= main.area * min_area_ratio;
    match policy {
        MultiFacePolicy::Off => FaceChoice::Main,
        MultiFacePolicy::Reject => {
            if others.iter().any(counts) {
                FaceChoice::Ambiguous
            } else {
                FaceChoice::Main
            }
        }
        MultiFacePolicy::Largest => others
            .iter()
            .enumerate()
            .filter(|(_, o)| o.score >= min_score && o.area > main.area)
            .max_by(|a, b| a.1.area.total_cmp(&b.1.area))
            .map_or(FaceChoice::Main, |(i, _)| FaceChoice::Other(i)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    #[test]
    fn identity_floor_boundaries() {
        assert!(!below_identity_floor(0.30, 0.30), "equal passes");
        assert!(below_identity_floor(0.299, 0.30));
        assert!(!below_identity_floor(0.8, 0.30));
        assert!(!below_identity_floor(-1.0, 0.0), "floor 0 disables");
        assert!(below_identity_floor(f32::NAN, 0.30), "NaN fails closed");
    }

    #[test]
    fn frame_age_uses_the_injected_clock() {
        let t0 = Instant::now();
        let max = Duration::from_millis(1_000);
        assert!(is_recent(t0, t0, max));
        assert!(
            is_recent(t0, t0 + max, max),
            "exactly at the limit is usable"
        );
        assert!(!is_recent(t0, t0 + max + Duration::from_millis(1), max));
        assert!(is_recent(t0 + S, t0, max), "future capture time is age 0");
        assert!(is_recent(t0, t0 + 3600 * S, Duration::ZERO), "0 disables");
    }

    #[test]
    fn minimum_frames() {
        assert!(enough_frames(8, 8));
        assert!(!enough_frames(7, 8));
        assert!(!enough_frames(1, 8));
    }

    fn sight(area: f32, score: f32) -> Sighting {
        Sighting { area, score }
    }

    #[test]
    fn reject_policy_drops_a_frame_with_a_qualifying_second_face() {
        let main = sight(0.10, 0.95);
        let r =
            |others: &[Sighting]| choose_face(MultiFacePolicy::Reject, 0.25, 0.75, main, others);
        assert_eq!(r(&[]), FaceChoice::Main);
        assert_eq!(
            r(&[sight(0.05, 0.9)]),
            FaceChoice::Ambiguous,
            "half the area"
        );
        assert_eq!(
            r(&[sight(0.025, 0.9)]),
            FaceChoice::Ambiguous,
            "exactly 25%"
        );
        assert_eq!(
            r(&[sight(0.02, 0.9)]),
            FaceChoice::Main,
            "small background face"
        );
        assert_eq!(
            r(&[sight(0.10, 0.5)]),
            FaceChoice::Main,
            "low-score detection"
        );
        assert_eq!(
            r(&[sight(0.01, 0.9), sight(0.08, 0.8)]),
            FaceChoice::Ambiguous
        );
    }

    #[test]
    fn largest_policy_picks_the_biggest_face_and_off_ignores_the_rest() {
        let main = sight(0.05, 0.95);
        let others = [sight(0.04, 0.9), sight(0.12, 0.8), sight(0.20, 0.5)];
        assert_eq!(
            choose_face(MultiFacePolicy::Largest, 0.25, 0.75, main, &others),
            FaceChoice::Other(1),
            "the 0.20 face is below the score floor"
        );
        assert_eq!(
            choose_face(
                MultiFacePolicy::Largest,
                0.25,
                0.75,
                sight(0.3, 0.9),
                &others
            ),
            FaceChoice::Main
        );
        assert_eq!(
            choose_face(MultiFacePolicy::Off, 0.25, 0.75, main, &others),
            FaceChoice::Main
        );
    }
}
