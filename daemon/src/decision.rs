//! Structured result of one `Authenticate` run.
//!
//! The decision core returns an [`AuthDecision`]; the wire [`Response`] and the
//! `WaitForMatch` [`AttemptOutcome`] are both built from it, so nothing has to
//! re-parse a message string to learn the similarity.

use crate::liveness::{LivenessOutcome, LIVENESS_UNAVAILABLE_CODE};
use crate::unlock_wait::AttemptOutcome;
use doorman_shared::Response;

/// How an authentication run ended.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthOutcome {
    Matched,
    NotMatched,
    NotEnrolled,
    /// Camera lease failed; `message` is the user-facing reason.
    CameraUnavailable {
        message: String,
    },
    NoFrames,
    NoClearFace,
    LivenessRejected,
    LivenessError,
    /// Mean mode: a frame that fed the decision did not reach the per-frame
    /// identity floor (`[decision] frame_identity_floor`). A judged rejection.
    IdentityFloor,
    /// More than one face was in view (`[recognition] multi_face_policy =
    /// "reject"`) and too few single-face frames remained. A judged rejection.
    AmbiguousFaces,
    /// The attempt ended with fewer judged-good frames than
    /// `[authentication] min_decision_frames`. Not a judged failure.
    InsufficientFrames,
    /// Face auth for this user is suspended by the failure budget
    /// (`[ipc] face_max_judged_failures`); no camera was opened.
    Suspended {
        retry_after_s: u64,
    },
}

/// Stable code for a rejection by the per-frame identity floor.
pub const IDENTITY_FLOOR_CODE: &str = "identity_floor";
/// Stable code for a rejection by the multi-face gate.
pub const AMBIGUOUS_FACES_CODE: &str = "ambiguous_faces";
/// Stable code for an attempt that ended with too few judged-good frames.
pub const INSUFFICIENT_FRAMES_CODE: &str = "insufficient_frames";

#[derive(Debug, Clone, PartialEq)]
pub struct AuthDecision {
    pub outcome: AuthOutcome,
    /// Final cosine similarity, when a decision on embeddings was reached.
    pub similarity: Option<f32>,
    /// Quality embeddings that entered the decision.
    pub frames_used: usize,
    /// Frames pulled from the camera.
    pub frames_seen: usize,
    /// Liveness verdict; `None` when liveness was disabled or never evaluated.
    pub liveness: Option<LivenessOutcome>,
}

/// Wire reason of the `Denied` answer while face auth is suspended. The code
/// is stable; scripts and the PAM module fall through to the password path.
pub fn suspended_reason(retry_after_s: u64) -> String {
    format!(
        "face auth suspended [{}]: use the password (retry in {retry_after_s} s)",
        crate::face_budget::FACE_SUSPENDED_CODE
    )
}

/// Success message of `Authenticate` (wire format, unchanged).
pub fn authenticated_message(similarity: f32) -> String {
    format!("Authenticated (confidence: {:.2}%)", similarity * 100.0)
}

impl AuthDecision {
    /// A decision that carries only an outcome (early exits).
    pub fn bare(outcome: AuthOutcome) -> Self {
        Self {
            outcome,
            similarity: None,
            frames_used: 0,
            frames_seen: 0,
            liveness: None,
        }
    }

    /// The wire response. A `Matched` without a similarity cannot be built by
    /// the core; it fails closed here.
    pub fn into_response(self) -> Response {
        let failure = |reason: &str| Response::Failure {
            reason: reason.to_string(),
        };
        let unavailable_reason = match &self.liveness {
            Some(LivenessOutcome::Unavailable { reason }) => reason.as_str(),
            _ => "no detail",
        };
        match (self.outcome, self.similarity) {
            (AuthOutcome::Matched, Some(similarity)) => Response::Success {
                message: Some(authenticated_message(similarity)),
                data: None,
            },
            (AuthOutcome::Matched, None) => failure("Authentication decision incomplete."),
            (AuthOutcome::NotMatched, similarity) => Response::Failure {
                reason: format!(
                    "Face not recognized (confidence: {:.2}%)",
                    similarity.unwrap_or(0.0) * 100.0
                ),
            },
            (AuthOutcome::NotEnrolled, _) => failure("User not enrolled"),
            (AuthOutcome::CameraUnavailable { message }, _) => {
                Response::Failure { reason: message }
            }
            (AuthOutcome::NoFrames, _) => failure("Failed to capture frames"),
            (AuthOutcome::NoClearFace, _) => failure(
                "No clear face detected. Please ensure good lighting and look at the camera.",
            ),
            (AuthOutcome::LivenessRejected, _) => {
                failure("Liveness check failed (not a live face).")
            }
            (AuthOutcome::IdentityFloor, _) => failure(&format!(
                "Face not recognized [{IDENTITY_FLOOR_CODE}]: a frame did not match the enrolled face."
            )),
            (AuthOutcome::AmbiguousFaces, _) => failure(&format!(
                "More than one face in view [{AMBIGUOUS_FACES_CODE}]; only the enrolled user may be in front of the camera."
            )),
            (AuthOutcome::InsufficientFrames, _) => failure(&format!(
                "Too few usable frames [{INSUFFICIENT_FRAMES_CODE}]; please try again."
            )),
            (AuthOutcome::Suspended { retry_after_s }, _) => Response::Denied {
                reason: suspended_reason(retry_after_s),
            },
            (AuthOutcome::LivenessError, _) => Response::Failure {
                reason: format!(
                    "Liveness check unavailable [{LIVENESS_UNAVAILABLE_CODE}]: {unavailable_reason}"
                ),
            },
        }
    }

    /// The `WaitForMatch` view of this decision.
    pub fn attempt_outcome(&self) -> AttemptOutcome {
        match (&self.outcome, self.similarity) {
            (AuthOutcome::Matched, Some(similarity)) => AttemptOutcome::Matched { similarity },
            (AuthOutcome::NotEnrolled, _) => AttemptOutcome::NotEnrolled,
            (AuthOutcome::Suspended { retry_after_s }, _) => AttemptOutcome::Suspended {
                retry_after_s: *retry_after_s,
            },
            _ => AttemptOutcome::NotMatched,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decision(outcome: AuthOutcome, similarity: Option<f32>) -> AuthDecision {
        AuthDecision {
            similarity,
            ..AuthDecision::bare(outcome)
        }
    }

    #[test]
    fn matched_maps_to_wire_success_and_attempt_matched() {
        let d = decision(AuthOutcome::Matched, Some(0.8123));
        assert_eq!(
            d.attempt_outcome(),
            AttemptOutcome::Matched { similarity: 0.8123 }
        );
        match d.into_response() {
            Response::Success { message, .. } => {
                assert_eq!(
                    message.as_deref(),
                    Some("Authenticated (confidence: 81.23%)")
                )
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn matched_without_similarity_fails_closed() {
        let d = decision(AuthOutcome::Matched, None);
        assert_eq!(d.attempt_outcome(), AttemptOutcome::NotMatched);
        assert!(matches!(d.into_response(), Response::Failure { .. }));
    }

    #[test]
    fn failure_reasons_keep_their_wire_text() {
        let reason = |o| match decision(o, None).into_response() {
            Response::Failure { reason } => reason,
            other => panic!("{other:?}"),
        };
        assert_eq!(reason(AuthOutcome::NotEnrolled), "User not enrolled");
        assert_eq!(reason(AuthOutcome::NoFrames), "Failed to capture frames");
        assert!(reason(AuthOutcome::LivenessError).contains("[liveness_unavailable]"));
        assert_eq!(
            reason(AuthOutcome::LivenessRejected),
            "Liveness check failed (not a live face)."
        );
        match decision(AuthOutcome::NotMatched, Some(0.3)).into_response() {
            Response::Failure { reason } => {
                assert_eq!(reason, "Face not recognized (confidence: 30.00%)")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn unavailable_liveness_reports_its_own_code_and_reason() {
        let d = AuthDecision {
            liveness: Some(LivenessOutcome::Unavailable {
                reason: "depth model not loaded".into(),
            }),
            ..AuthDecision::bare(AuthOutcome::LivenessError)
        };
        match d.into_response() {
            Response::Failure { reason } => assert_eq!(
                reason,
                "Liveness check unavailable [liveness_unavailable]: depth model not loaded"
            ),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn not_enrolled_is_terminal_for_wait_for_match() {
        assert_eq!(
            decision(AuthOutcome::NotEnrolled, None).attempt_outcome(),
            AttemptOutcome::NotEnrolled
        );
        assert_eq!(
            decision(AuthOutcome::LivenessRejected, None).attempt_outcome(),
            AttemptOutcome::NotMatched
        );
    }
}
