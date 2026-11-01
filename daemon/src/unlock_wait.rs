//! `WaitForMatch`: hold a connection until the user is recognized.
//!
//! The decision is NOT made here. Every attempt is delegated to a
//! [`DecisionSource`]; in the daemon that is the very same pipeline `Authenticate`
//! runs (quality gate, liveness, aggregated threshold), so `Matched` can never
//! be weaker than an `Authenticate` success. This module only adds the waiting
//! policy: lock gating, retry until the deadline, one terminal answer.
//!
//! No camera or model code lives here, so it is testable with a fake source.

use async_trait::async_trait;
use doorman_shared::{NoMatchReason, Response};
use std::future::pending;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;
use tokio::time::Instant;

/// Result of one full decision attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttemptOutcome {
    /// The full decision accepted the user.
    Matched { similarity: f32 },
    /// Not accepted (no clear face, liveness reject, below threshold, no
    /// camera, ...). Worth another attempt while time remains.
    NotMatched,
    /// The user has no enrollment; retrying cannot help.
    NotEnrolled,
}

/// One authentication attempt for `username`. Implementations must apply the
/// complete `Authenticate` decision.
#[async_trait]
pub trait DecisionSource: Send + Sync {
    async fn attempt(&self, username: &str) -> AttemptOutcome;
}

/// Run decision attempts until one matches, the deadline passes, or retrying is
/// pointless.
///
/// `armed` is the lock supervisor's `locked && !idle` gate; `None` means the
/// daemon is not lock-gated (user/dev mode). While disarmed this awaits the
/// gate (no polling) and no attempt, hence no camera lease, is made. An attempt
/// in flight is dropped when the gate closes or the deadline passes. Attempts
/// follow each other immediately; each one waits on camera frames, so there is
/// no busy loop. Returns exactly one terminal response.
pub async fn wait_for_match(
    source: &dyn DecisionSource,
    mut armed: Option<watch::Receiver<bool>>,
    username: &str,
    timeout: Duration,
) -> Response {
    let deadline = Instant::now() + timeout;
    let mut reason = NoMatchReason::Timeout;

    loop {
        if let Some(rx) = armed.as_mut() {
            if !*rx.borrow_and_update() {
                reason = NoMatchReason::NotArmed;
                match tokio::time::timeout_at(deadline, rx.wait_for(|a| *a)).await {
                    Ok(Ok(_)) => continue,
                    // Deadline passed, or the supervisor is gone: never armed.
                    Ok(Err(_)) | Err(_) => return Response::NoMatch { reason },
                }
            }
        }

        let disarmed = async {
            match armed.as_mut() {
                Some(rx) => {
                    if rx.wait_for(|a| !*a).await.is_err() {
                        pending::<()>().await;
                    }
                }
                None => pending::<()>().await,
            }
        };

        tokio::select! {
            biased;
            _ = tokio::time::sleep_until(deadline) => return Response::NoMatch { reason },
            _ = disarmed => continue,
            outcome = source.attempt(username) => match outcome {
                AttemptOutcome::Matched { similarity } => {
                    return Response::Matched {
                        username: username.to_string(),
                        similarity,
                        ts_unix_ms: unix_ms_now(),
                    };
                }
                AttemptOutcome::NotMatched => reason = NoMatchReason::Timeout,
                AttemptOutcome::NotEnrolled => {
                    return Response::NoMatch { reason: NoMatchReason::NotEnrolled };
                }
            },
        }
    }
}

/// The requested timeout capped by `[ipc] wait_for_match_max_ms`.
pub fn effective_timeout(requested_ms: u64, max_ms: u64) -> Duration {
    Duration::from_millis(requested_ms.min(max_ms))
}

/// Success message of `Authenticate`; `parse_authenticated_similarity` is its inverse.
pub fn authenticated_message(similarity: f32) -> String {
    format!("Authenticated (confidence: {:.2}%)", similarity * 100.0)
}

/// Similarity carried by an `Authenticate` success message (percent, 2 decimals).
pub fn parse_authenticated_similarity(message: &str) -> Option<f32> {
    let percent = message
        .strip_prefix("Authenticated (confidence: ")?
        .strip_suffix("%)")?;
    Some(percent.parse::<f32>().ok()? / 100.0)
}

fn unix_ms_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct Script {
        outcomes: std::sync::Mutex<Vec<AttemptOutcome>>,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl DecisionSource for Script {
        async fn attempt(&self, _username: &str) -> AttemptOutcome {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let next = {
                let mut o = self.outcomes.lock().unwrap();
                (!o.is_empty()).then(|| o.remove(0))
            };
            match next {
                Some(outcome) => outcome,
                // Out of script: behave like a real attempt that waits for frames.
                None => pending().await,
            }
        }
    }

    fn script(outcomes: Vec<AttemptOutcome>) -> (Script, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Script {
                outcomes: std::sync::Mutex::new(outcomes),
                calls: calls.clone(),
            },
            calls,
        )
    }

    const LONG: Duration = Duration::from_secs(30);

    #[tokio::test(start_paused = true)]
    async fn retries_until_match_then_stops() {
        let (s, calls) = script(vec![
            AttemptOutcome::NotMatched,
            AttemptOutcome::NotMatched,
            AttemptOutcome::Matched { similarity: 0.8 },
            AttemptOutcome::Matched { similarity: 0.9 },
        ]);
        let r = wait_for_match(&s, None, "alice", LONG).await;
        assert!(
            matches!(&r, Response::Matched { username, similarity, .. }
                if username == "alice" && (*similarity - 0.8).abs() < 1e-6),
            "{r:?}"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "no attempt after the match"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn times_out_without_match() {
        let (s, _) = script(vec![AttemptOutcome::NotMatched]);
        let r = wait_for_match(&s, None, "alice", Duration::from_secs(5)).await;
        assert!(
            matches!(
                r,
                Response::NoMatch {
                    reason: NoMatchReason::Timeout
                }
            ),
            "{r:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn not_enrolled_is_terminal() {
        let (s, calls) = script(vec![AttemptOutcome::NotEnrolled]);
        let r = wait_for_match(&s, None, "alice", LONG).await;
        assert!(matches!(
            r,
            Response::NoMatch {
                reason: NoMatchReason::NotEnrolled
            }
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn disarmed_makes_no_attempt_and_reports_not_armed() {
        let (s, calls) = script(vec![AttemptOutcome::Matched { similarity: 0.9 }]);
        let (_tx, rx) = watch::channel(false);
        let r = wait_for_match(&s, Some(rx), "alice", Duration::from_secs(5)).await;
        assert!(matches!(
            r,
            Response::NoMatch {
                reason: NoMatchReason::NotArmed
            }
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn waits_for_arming_then_matches() {
        let (s, calls) = script(vec![AttemptOutcome::Matched { similarity: 0.9 }]);
        let (tx, rx) = watch::channel(false);
        let arm = async {
            tokio::time::sleep(Duration::from_secs(2)).await;
            tx.send(true).unwrap();
        };
        let (r, _) = tokio::join!(wait_for_match(&s, Some(rx), "alice", LONG), arm);
        assert!(matches!(r, Response::Matched { .. }), "{r:?}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn disarming_drops_the_attempt_in_flight() {
        // Script is empty: the attempt waits forever, like a camera that never delivers.
        let (s, calls) = script(vec![]);
        let (tx, rx) = watch::channel(true);
        let disarm = async {
            tokio::time::sleep(Duration::from_secs(1)).await;
            tx.send(false).unwrap();
        };
        let (r, _) = tokio::join!(
            wait_for_match(&s, Some(rx), "alice", Duration::from_secs(5)),
            disarm
        );
        assert!(matches!(
            r,
            Response::NoMatch {
                reason: NoMatchReason::NotArmed
            }
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn authenticated_message_roundtrips_similarity() {
        let s = parse_authenticated_similarity(&authenticated_message(0.8123)).unwrap();
        assert!((s - 0.8123).abs() < 0.0001, "{s}");
        assert_eq!(parse_authenticated_similarity("Face not recognized"), None);
    }

    #[test]
    fn timeout_is_capped() {
        assert_eq!(effective_timeout(5, 60_000), Duration::from_millis(5));
        assert_eq!(effective_timeout(u64::MAX, 60_000), Duration::from_secs(60));
        assert_eq!(effective_timeout(10, 0), Duration::ZERO);
    }
}
