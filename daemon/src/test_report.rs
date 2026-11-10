//! Structured result of `doorman test --json`.
//!
//! The daemon answers `Authenticate` with a wire `Response` whose text is the
//! stable contract (see `decision.rs`). This module classifies that answer into
//! a small record that carries no frames, embeddings or exact similarity:
//! only the bucketed similarity, a reason code and timing.

use crate::decision::{AMBIGUOUS_FACES_CODE, IDENTITY_FLOOR_CODE, INSUFFICIENT_FRAMES_CODE};
use crate::face_budget::FACE_SUSPENDED_CODE;
use crate::liveness::{LIVENESS_REJECTED_CODE, LIVENESS_UNAVAILABLE_CODE};
use crate::sudo_policy::similarity_bucket;
use doorman_shared::Response;
use serde::Serialize;

/// Exit code: the face was recognized.
pub const EXIT_RECOGNIZED: i32 = 0;
/// Exit code: CLI or daemon-communication error.
pub const EXIT_ERROR: i32 = 1;
/// Exit code: not recognized, rejected, unavailable or suspended.
pub const EXIT_NOT_RECOGNIZED: i32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TestReport {
    /// `recognized`, `not_recognized`, `suspended` or `error`.
    pub decision: &'static str,
    /// Stable machine code, see `docs/CONFIG.md`/`docs/unlock-path.md`.
    pub reason_code: String,
    /// Similarity rounded to 0.05, when the daemon reported one.
    pub similarity_bucket: Option<f32>,
    /// `Some(false)` when the liveness gate rejected, `Some(true)` when the
    /// attempt got past it to the identity decision, `None` when unknown.
    pub liveness_ok: Option<bool>,
    /// Frames that entered the decision. Not carried over the wire yet: `null`.
    pub frames_judged: Option<u32>,
    pub duration_ms: u64,
    /// RFC 3339 UTC time until which face auth is suspended.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suspended_until: Option<String>,
    /// Process exit code (not part of the JSON).
    #[serde(skip)]
    pub exit_code: i32,
}

/// The `NN.NN%` after `confidence: ` in the daemon's texts, as a fraction.
fn confidence(text: &str) -> Option<f32> {
    let rest = text.split("confidence: ").nth(1)?;
    let pct: f32 = rest.split('%').next()?.trim().parse().ok()?;
    Some(pct / 100.0)
}

/// Seconds in `retry in N s`.
fn retry_secs(text: &str) -> Option<u64> {
    text.split("retry in ")
        .nth(1)?
        .split(' ')
        .next()?
        .parse()
        .ok()
}

/// Classify the daemon's answer. `now_unix` is the current Unix time (injected
/// for tests); `duration_ms` is measured by the caller.
pub fn classify(response: &Response, duration_ms: u64, now_unix: i64) -> TestReport {
    let report = |decision, code: &str, exit_code| TestReport {
        decision,
        reason_code: code.to_string(),
        similarity_bucket: None,
        liveness_ok: None,
        frames_judged: None,
        duration_ms,
        suspended_until: None,
        exit_code,
    };
    match response {
        Response::Success { message, .. } => TestReport {
            similarity_bucket: message
                .as_deref()
                .and_then(confidence)
                .map(similarity_bucket),
            liveness_ok: Some(true),
            ..report("recognized", "matched", EXIT_RECOGNIZED)
        },
        Response::Failure { reason } => {
            let has = |code: &str| reason.contains(&format!("[{code}]"));
            let (code, liveness_ok) = if has(LIVENESS_UNAVAILABLE_CODE) {
                (LIVENESS_UNAVAILABLE_CODE, None)
            } else if has(IDENTITY_FLOOR_CODE) {
                (IDENTITY_FLOOR_CODE, Some(true))
            } else if has(AMBIGUOUS_FACES_CODE) {
                (AMBIGUOUS_FACES_CODE, None)
            } else if has(INSUFFICIENT_FRAMES_CODE) {
                (INSUFFICIENT_FRAMES_CODE, None)
            } else if reason.starts_with("Liveness check failed") {
                (LIVENESS_REJECTED_CODE, Some(false))
            } else if reason.starts_with("Face not recognized") {
                ("not_matched", Some(true))
            } else if reason.starts_with("No clear face") {
                ("no_clear_face", None)
            } else if reason == "Failed to capture frames" {
                ("no_frames", None)
            } else if reason == "User not enrolled" {
                ("not_enrolled", None)
            } else {
                ("other_failure", None)
            };
            TestReport {
                similarity_bucket: confidence(reason).map(similarity_bucket),
                liveness_ok,
                ..report("not_recognized", code, EXIT_NOT_RECOGNIZED)
            }
        }
        Response::Denied { reason } if reason.contains(FACE_SUSPENDED_CODE) => {
            let until = retry_secs(reason).and_then(|s| {
                chrono::DateTime::from_timestamp(now_unix.saturating_add(s as i64), 0)
                    .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
            });
            TestReport {
                suspended_until: until,
                ..report("suspended", FACE_SUSPENDED_CODE, EXIT_NOT_RECOGNIZED)
            }
        }
        Response::Denied { .. } => report("error", "denied", EXIT_ERROR),
        Response::RateLimited { .. } => report("error", "rate_limited", EXIT_ERROR),
        _ => report("error", "unexpected_response", EXIT_ERROR),
    }
}

/// Report for a failure before an answer arrived (daemon unreachable, timeout).
pub fn error_report(duration_ms: u64) -> TestReport {
    TestReport {
        decision: "error",
        reason_code: "daemon_error".to_string(),
        similarity_bucket: None,
        liveness_ok: None,
        frames_judged: None,
        duration_ms,
        suspended_until: None,
        exit_code: EXIT_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::{AuthDecision, AuthOutcome};

    fn wire(outcome: AuthOutcome, similarity: Option<f32>) -> Response {
        AuthDecision {
            similarity,
            ..AuthDecision::bare(outcome)
        }
        .into_response()
    }

    fn c(r: &Response) -> TestReport {
        classify(r, 123, 1_000_000)
    }

    #[test]
    fn every_daemon_outcome_has_a_stable_code_and_exit_code() {
        let cases: Vec<(Response, &str, &str, i32)> = vec![
            (
                wire(AuthOutcome::Matched, Some(0.81)),
                "recognized",
                "matched",
                0,
            ),
            (
                wire(AuthOutcome::NotMatched, Some(0.31)),
                "not_recognized",
                "not_matched",
                2,
            ),
            (
                wire(AuthOutcome::LivenessRejected, None),
                "not_recognized",
                "liveness_rejected",
                2,
            ),
            (
                wire(AuthOutcome::LivenessError, None),
                "not_recognized",
                "liveness_unavailable",
                2,
            ),
            (
                wire(AuthOutcome::IdentityFloor, None),
                "not_recognized",
                "identity_floor",
                2,
            ),
            (
                wire(AuthOutcome::AmbiguousFaces, None),
                "not_recognized",
                "ambiguous_faces",
                2,
            ),
            (
                wire(AuthOutcome::InsufficientFrames, None),
                "not_recognized",
                "insufficient_frames",
                2,
            ),
            (
                wire(AuthOutcome::NoClearFace, None),
                "not_recognized",
                "no_clear_face",
                2,
            ),
            (
                wire(AuthOutcome::NoFrames, None),
                "not_recognized",
                "no_frames",
                2,
            ),
            (
                wire(AuthOutcome::NotEnrolled, None),
                "not_recognized",
                "not_enrolled",
                2,
            ),
            (
                wire(AuthOutcome::Suspended { retry_after_s: 30 }, None),
                "suspended",
                "face_auth_suspended",
                2,
            ),
            (
                Response::Denied {
                    reason: "peer uid".into(),
                },
                "error",
                "denied",
                1,
            ),
            (
                Response::RateLimited { retry_after_s: 1 },
                "error",
                "rate_limited",
                1,
            ),
        ];
        for (resp, decision, code, exit) in cases {
            let r = c(&resp);
            assert_eq!(
                (r.decision, r.reason_code.as_str(), r.exit_code),
                (decision, code, exit),
                "{resp:?}"
            );
            assert_eq!(r.duration_ms, 123);
        }
    }

    #[test]
    fn similarity_is_bucketed_never_exact() {
        let r = c(&wire(AuthOutcome::Matched, Some(0.8123)));
        assert_eq!(r.similarity_bucket, Some(0.80));
        let r = c(&wire(AuthOutcome::NotMatched, Some(0.31)));
        assert_eq!(r.similarity_bucket, Some(0.30));
        let json = serde_json::to_string(&c(&wire(AuthOutcome::Matched, Some(0.8123)))).unwrap();
        assert!(!json.contains("0.8123") && !json.contains("81.2"), "{json}");
    }

    #[test]
    fn liveness_ok_distinguishes_a_spoof_reject_from_an_identity_reject() {
        assert_eq!(
            c(&wire(AuthOutcome::LivenessRejected, None)).liveness_ok,
            Some(false)
        );
        assert_eq!(
            c(&wire(AuthOutcome::NotMatched, Some(0.1))).liveness_ok,
            Some(true)
        );
        assert_eq!(c(&wire(AuthOutcome::NoClearFace, None)).liveness_ok, None);
    }

    #[test]
    fn suspension_reports_the_end_time() {
        let r = c(&wire(AuthOutcome::Suspended { retry_after_s: 90 }, None));
        // 1_000_000 + 90 s = 1970-01-12T13:48:10Z
        assert_eq!(r.suspended_until.as_deref(), Some("1970-01-12T13:48:10Z"));
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["decision"], "suspended");
        assert!(json.get("exit_code").is_none());
        let other = serde_json::to_value(c(&wire(AuthOutcome::NotMatched, Some(0.1)))).unwrap();
        assert!(other.get("suspended_until").is_none());
    }

    #[test]
    fn json_carries_exactly_the_documented_fields() {
        let v = serde_json::to_value(c(&wire(AuthOutcome::Matched, Some(0.9)))).unwrap();
        let mut keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "decision",
                "duration_ms",
                "frames_judged",
                "liveness_ok",
                "reason_code",
                "similarity_bucket"
            ]
        );
    }
}
