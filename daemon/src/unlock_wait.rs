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
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant as StdInstant, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;
use tokio::time::Instant;

/// Pause after a rejected attempt before the next one starts, so a source that
/// answers instantly (camera unavailable, suspended-then-expired budget) can
/// never turn the retry loop into a busy loop that hammers the camera and the
/// models. Small against an attempt (1-3 s); the failure budget
/// (`[ipc] face_max_judged_failures`) is what bounds the number of attempts.
pub const RETRY_PAUSE: Duration = Duration::from_millis(250);

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
    /// Face auth is suspended by the failure budget; retrying before
    /// `retry_after_s` is pointless and would only be refused again.
    Suspended { retry_after_s: u64 },
}

/// One authentication attempt for `username`. Implementations must apply the
/// complete `Authenticate` decision.
#[async_trait]
pub trait DecisionSource: Send + Sync {
    async fn attempt(&self, username: &str) -> AttemptOutcome;
}

/// Request-relative clock of one `WaitForMatch`, shared with the decision
/// source so its timing record can attribute time to the stages
/// (`hint_wait`, `lock_to_camera_ready`). Also the freshness floor: frames
/// captured before `request_start` are never used for this call.
#[derive(Debug, Clone)]
pub struct WaitTiming {
    pub request_start: StdInstant,
    armed_at: Arc<OnceLock<StdInstant>>,
}

impl WaitTiming {
    pub fn new() -> Self {
        Self {
            request_start: StdInstant::now(),
            armed_at: Arc::new(OnceLock::new()),
        }
    }

    /// Record the first moment the lock gate was seen armed (later calls are no-ops).
    pub fn mark_armed(&self) {
        let _ = self.armed_at.set(StdInstant::now());
    }

    /// The shared cell, for the attempt's timing record.
    pub fn armed_cell(&self) -> Arc<OnceLock<StdInstant>> {
        self.armed_at.clone()
    }

    pub fn armed_at(&self) -> Option<StdInstant> {
        self.armed_at.get().copied()
    }

    /// Time from the request until the gate was first seen armed.
    pub fn hint_wait(&self) -> Option<Duration> {
        self.armed_at()
            .map(|t| t.saturating_duration_since(self.request_start))
    }
}

impl Default for WaitTiming {
    fn default() -> Self {
        Self::new()
    }
}

/// How one pre-arm phase ended.
enum Prearm {
    /// Accepted by the full decision AND the gate was armed when it was delivered.
    Matched {
        similarity: f32,
    },
    NotEnrolled,
    /// The failure budget suspended face auth; nothing to hold or wait for.
    Suspended,
    /// Hold expired, the gate closed again, or the supervisor is gone. Any
    /// result is discarded; the caller falls back to the standard loop.
    Over,
    /// The call deadline passed first.
    Deadline,
}

/// Pre-arm: run decision attempts while the gate is still closed, so camera
/// open and frame scoring overlap the wait for the lock hint.
///
/// A result is only ever *held* until the gate opens; it is delivered the
/// moment the gate is armed, never before. Everything is dropped (camera lease
/// included) when `hold_until` passes without the gate opening, so a lock that
/// never completes cannot keep the camera on. The source must use only frames
/// captured after `WaitTiming::request_start`.
async fn prearm_phase(
    source: &dyn DecisionSource,
    rx: &mut watch::Receiver<bool>,
    username: &str,
    hold_until: Instant,
    deadline: Instant,
    timing: &WaitTiming,
) -> Prearm {
    enum Done {
        Matched(f32),
        NotEnrolled,
        Suspended,
    }
    let attempts = async {
        loop {
            match source.attempt(username).await {
                AttemptOutcome::Matched { similarity } => return Done::Matched(similarity),
                AttemptOutcome::NotEnrolled => return Done::NotEnrolled,
                AttemptOutcome::Suspended { .. } => return Done::Suspended,
                // Never spin: pause, and let the hold/gate branches of the select run.
                AttemptOutcome::NotMatched => tokio::time::sleep(RETRY_PAUSE).await,
            }
        }
    };
    tokio::pin!(attempts);
    let finish = |done: Done| match done {
        Done::Matched(similarity) => Prearm::Matched { similarity },
        Done::NotEnrolled => Prearm::NotEnrolled,
        Done::Suspended => Prearm::Suspended,
    };

    // Stage A: the gate is closed. Nothing may be delivered yet.
    let mut held: Option<Done> = None;
    loop {
        let opened = {
            let hold_expired = tokio::time::sleep_until(hold_until);
            tokio::pin!(hold_expired);
            tokio::select! {
                biased;
                _ = &mut hold_expired => return Prearm::Over,
                opened = rx.wait_for(|a| *a) => Some(opened.is_ok()),
                done = &mut attempts, if held.is_none() => { held = Some(done); None }
            }
        };
        match opened {
            Some(true) => break,
            Some(false) => return Prearm::Over,
            // NotEnrolled needs no lock to be final; a match must wait for it.
            None if matches!(held, Some(Done::NotEnrolled)) => return Prearm::NotEnrolled,
            // A refusal needs no lock either, and nothing is delivered by it.
            None if matches!(held, Some(Done::Suspended)) => return Prearm::Suspended,
            None => {}
        }
    }
    timing.mark_armed();
    if let Some(done) = held {
        return finish(done);
    }

    // Stage B: the gate opened while the attempt is still in flight. Let it
    // finish; drop it if the gate closes again or the deadline passes.
    tokio::select! {
        biased;
        _ = tokio::time::sleep_until(deadline) => Prearm::Deadline,
        _ = async {
            if rx.wait_for(|a| !*a).await.is_err() {
                pending::<()>().await;
            }
        } => Prearm::Over,
        done = &mut attempts => finish(done),
    }
}

/// Run decision attempts until one matches, the deadline passes, or retrying is
/// pointless.
///
/// `armed` is the lock supervisor's `locked && !idle` gate; `None` means the
/// daemon is not lock-gated (user/dev mode). `Matched` is never returned while
/// the gate is closed. While disarmed this awaits the gate (no polling).
///
/// With `prearm_hold = Some(h)` and a closed gate at the start, decision
/// attempts run immediately (see [`prearm_phase`]) for at most `h`; the camera
/// lease inside the source is released when that phase ends. Afterwards, or
/// without pre-arm, no attempt, hence no camera lease, is made while disarmed.
/// An attempt in flight is dropped when the gate closes after having been open
/// or the deadline passes. Returns exactly one terminal response.
pub async fn wait_for_match(
    source: &dyn DecisionSource,
    mut armed: Option<watch::Receiver<bool>>,
    username: &str,
    timeout: Duration,
    prearm_hold: Option<Duration>,
    timing: &WaitTiming,
) -> Response {
    let started = Instant::now();
    let deadline = started + timeout;
    let mut reason = NoMatchReason::Timeout;

    if let (Some(rx), Some(hold)) = (armed.as_mut(), prearm_hold) {
        if !*rx.borrow_and_update() {
            let hold_until = (started + hold).min(deadline);
            match prearm_phase(source, rx, username, hold_until, deadline, timing).await {
                Prearm::Matched { similarity } => {
                    return Response::Matched {
                        username: username.to_string(),
                        similarity,
                        ts_unix_ms: unix_ms_now(),
                    };
                }
                Prearm::NotEnrolled => {
                    return Response::NoMatch {
                        reason: NoMatchReason::NotEnrolled,
                    };
                }
                Prearm::Suspended => {
                    return Response::NoMatch {
                        reason: NoMatchReason::Suspended,
                    };
                }
                Prearm::Deadline => return Response::NoMatch { reason },
                Prearm::Over => {}
            }
        }
    }

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
        timing.mark_armed();

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
                AttemptOutcome::Suspended { .. } => {
                    return Response::NoMatch { reason: NoMatchReason::Suspended };
                }
            },
        }
        // Pause before the next attempt (never a back-to-back retry).
        tokio::select! {
            biased;
            _ = tokio::time::sleep_until(deadline) => return Response::NoMatch { reason },
            _ = tokio::time::sleep(RETRY_PAUSE) => {}
        }
    }
}

/// The requested timeout capped by `[ipc] wait_for_match_max_ms`.
pub fn effective_timeout(requested_ms: u64, max_ms: u64) -> Duration {
    Duration::from_millis(requested_ms.min(max_ms))
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
        let r = wait_for_match(&s, None, "alice", LONG, None, &WaitTiming::new()).await;
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
        let r = wait_for_match(
            &s,
            None,
            "alice",
            Duration::from_secs(5),
            None,
            &WaitTiming::new(),
        )
        .await;
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
        let r = wait_for_match(&s, None, "alice", LONG, None, &WaitTiming::new()).await;
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
        let r = wait_for_match(
            &s,
            Some(rx),
            "alice",
            Duration::from_secs(5),
            None,
            &WaitTiming::new(),
        )
        .await;
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
        let timing = WaitTiming::new();
        let (r, _) = tokio::join!(
            wait_for_match(&s, Some(rx), "alice", LONG, None, &timing),
            arm
        );
        assert!(matches!(r, Response::Matched { .. }), "{r:?}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn disarming_drops_the_attempt_in_flight() {
        // Script is empty: the attempt waits forever, like a camera that never delivers.
        let (s, calls) = script(vec![]);
        let (tx, rx) = watch::channel(true);
        let timing = WaitTiming::new();
        let disarm = async {
            tokio::time::sleep(Duration::from_secs(1)).await;
            tx.send(false).unwrap();
        };
        let (r, _) = tokio::join!(
            wait_for_match(&s, Some(rx), "alice", Duration::from_secs(5), None, &timing),
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
    fn timeout_is_capped() {
        assert_eq!(effective_timeout(5, 60_000), Duration::from_millis(5));
        assert_eq!(effective_timeout(u64::MAX, 60_000), Duration::from_secs(60));
        assert_eq!(effective_timeout(10, 0), Duration::ZERO);
    }

    /// Fake camera + recognizer: every attempt holds a "lease" for `takes`
    /// (camera open plus frame scoring), then yields `outcome`.
    struct Timed {
        takes: Duration,
        outcome: AttemptOutcome,
        calls: Arc<AtomicUsize>,
        live_leases: Arc<AtomicUsize>,
    }

    struct Lease(Arc<AtomicUsize>);
    impl Drop for Lease {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl DecisionSource for Timed {
        async fn attempt(&self, _username: &str) -> AttemptOutcome {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.live_leases.fetch_add(1, Ordering::SeqCst);
            let _lease = Lease(self.live_leases.clone());
            tokio::time::sleep(self.takes).await;
            self.outcome
        }
    }

    struct Rig {
        source: Timed,
        calls: Arc<AtomicUsize>,
        live: Arc<AtomicUsize>,
    }

    fn rig(takes_ms: u64, outcome: AttemptOutcome) -> Rig {
        let calls = Arc::new(AtomicUsize::new(0));
        let live = Arc::new(AtomicUsize::new(0));
        Rig {
            source: Timed {
                takes: Duration::from_millis(takes_ms),
                outcome,
                calls: calls.clone(),
                live_leases: live.clone(),
            },
            calls,
            live,
        }
    }

    const HOLD: Duration = Duration::from_secs(5);
    const MATCH: AttemptOutcome = AttemptOutcome::Matched { similarity: 0.9 };

    /// Run `wait_for_match` with the gate opened at `open_after` (never when
    /// `None`); returns the response and the virtual time it took.
    async fn run_gate(
        rig: &Rig,
        open_after: Option<Duration>,
        prearm: Option<Duration>,
        timeout: Duration,
    ) -> (Response, Duration) {
        let (tx, rx) = watch::channel(false);
        let t0 = Instant::now();
        let open = async {
            if let Some(d) = open_after {
                tokio::time::sleep(d).await;
                tx.send(true).unwrap();
            }
            pending::<()>().await;
        };
        let timing = WaitTiming::new();
        let wait = wait_for_match(&rig.source, Some(rx), "alice", timeout, prearm, &timing);
        let r = tokio::select! { r = wait => r, _ = open => unreachable!() };
        (r, t0.elapsed())
    }

    #[tokio::test(start_paused = true)]
    async fn prearm_matches_right_after_the_hint_when_scoring_is_done() {
        let rig = rig(1_000, MATCH);
        let (r, took) = run_gate(&rig, Some(Duration::from_secs(2)), Some(HOLD), LONG).await;
        assert!(matches!(r, Response::Matched { .. }), "{r:?}");
        assert_eq!(
            took,
            Duration::from_secs(2),
            "delivered at the hint, not later"
        );
        assert_eq!(rig.calls.load(Ordering::SeqCst), 1);
        assert_eq!(rig.live.load(Ordering::SeqCst), 0, "lease released");
    }

    #[tokio::test(start_paused = true)]
    async fn prearm_overlaps_scoring_with_the_hint_wait() {
        // Scoring takes 3 s, the hint comes at 1 s: done at 3 s, not 4 s.
        let rig = rig(3_000, MATCH);
        let (r, took) = run_gate(&rig, Some(Duration::from_secs(1)), Some(HOLD), LONG).await;
        assert!(matches!(r, Response::Matched { .. }), "{r:?}");
        assert_eq!(took, Duration::from_secs(3));
        assert_eq!(
            rig.calls.load(Ordering::SeqCst),
            1,
            "one attempt, not restarted"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn without_prearm_the_attempt_starts_only_at_the_hint() {
        let rig = rig(3_000, MATCH);
        let (r, took) = run_gate(&rig, Some(Duration::from_secs(1)), None, LONG).await;
        assert!(matches!(r, Response::Matched { .. }), "{r:?}");
        assert_eq!(took, Duration::from_secs(4));
    }

    #[tokio::test(start_paused = true)]
    async fn never_matched_while_the_gate_is_closed() {
        let rig = rig(10, MATCH);
        let (r, took) = run_gate(&rig, None, Some(HOLD), Duration::from_secs(20)).await;
        assert!(
            matches!(
                r,
                Response::NoMatch {
                    reason: NoMatchReason::NotArmed
                }
            ),
            "{r:?}"
        );
        assert_eq!(took, Duration::from_secs(20));
    }

    #[tokio::test(start_paused = true)]
    async fn camera_is_released_after_the_hold_when_the_hint_never_comes() {
        let rig = rig(10_000, MATCH); // still scoring when the hold ends
        let live = rig.live.clone();
        let probe = async {
            tokio::time::sleep(Duration::from_secs(1)).await;
            assert_eq!(live.load(Ordering::SeqCst), 1, "held during the hold");
            tokio::time::sleep(HOLD).await; // t = 6 s, hold ended at 5 s
            assert_eq!(live.load(Ordering::SeqCst), 0, "released after the hold");
        };
        let ((r, _), ()) = tokio::join!(
            run_gate(&rig, None, Some(HOLD), Duration::from_secs(10)),
            probe
        );
        assert!(matches!(r, Response::NoMatch { .. }), "{r:?}");
        assert_eq!(
            rig.calls.load(Ordering::SeqCst),
            1,
            "no new attempt while disarmed"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_held_result_is_discarded_when_the_hint_comes_after_the_hold() {
        let rig = rig(1_000, MATCH);
        let (r, took) = run_gate(&rig, Some(Duration::from_secs(7)), Some(HOLD), LONG).await;
        assert!(matches!(r, Response::Matched { .. }), "{r:?}");
        assert_eq!(took, Duration::from_secs(8), "fresh attempt after the hint");
        assert_eq!(
            rig.calls.load(Ordering::SeqCst),
            2,
            "stale result not reused"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn hold_is_capped_by_the_call_timeout() {
        let rig = rig(10, MATCH);
        let (r, took) = run_gate(&rig, None, Some(HOLD), Duration::from_secs(2)).await;
        assert!(matches!(r, Response::NoMatch { .. }), "{r:?}");
        assert_eq!(took, Duration::from_secs(2));
        assert_eq!(rig.live.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn no_match_during_prearm_keeps_trying_until_the_hint() {
        let rig = rig(500, AttemptOutcome::NotMatched);
        let (r, took) = run_gate(
            &rig,
            Some(Duration::from_secs(2)),
            Some(HOLD),
            Duration::from_secs(4),
        )
        .await;
        assert!(matches!(r, Response::NoMatch { .. }), "{r:?}");
        assert_eq!(took, Duration::from_secs(4));
        assert!(rig.calls.load(Ordering::SeqCst) >= 4);
    }

    #[tokio::test(start_paused = true)]
    async fn not_enrolled_is_final_during_prearm() {
        let rig = rig(100, AttemptOutcome::NotEnrolled);
        let (r, took) = run_gate(&rig, None, Some(HOLD), LONG).await;
        assert!(matches!(
            r,
            Response::NoMatch {
                reason: NoMatchReason::NotEnrolled
            }
        ));
        assert_eq!(took, Duration::from_millis(100));
    }

    #[tokio::test(start_paused = true)]
    async fn closing_the_gate_after_it_opened_drops_the_attempt_in_flight() {
        let rig = rig(10_000, MATCH);
        let (tx, rx) = watch::channel(false);
        let flip = async {
            tokio::time::sleep(Duration::from_secs(1)).await;
            tx.send(true).unwrap();
            tokio::time::sleep(Duration::from_secs(1)).await;
            tx.send(false).unwrap();
        };
        let timing = WaitTiming::new();
        let (r, ()) = tokio::join!(
            wait_for_match(
                &rig.source,
                Some(rx),
                "alice",
                Duration::from_secs(6),
                Some(HOLD),
                &timing
            ),
            flip
        );
        assert!(
            matches!(
                r,
                Response::NoMatch {
                    reason: NoMatchReason::NotArmed
                }
            ),
            "{r:?}"
        );
        assert_eq!(rig.live.load(Ordering::SeqCst), 0);
        assert!(timing.hint_wait().is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn an_already_open_gate_skips_prearm() {
        let rig = rig(1_000, MATCH);
        let (_tx, rx) = watch::channel(true);
        let timing = WaitTiming::new();
        let r = wait_for_match(&rig.source, Some(rx), "alice", LONG, Some(HOLD), &timing).await;
        assert!(matches!(r, Response::Matched { .. }));
        assert_eq!(rig.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn not_lock_gated_mode_is_unchanged_by_prearm() {
        let rig = rig(1_000, MATCH);
        let r = wait_for_match(
            &rig.source,
            None,
            "alice",
            LONG,
            Some(HOLD),
            &WaitTiming::new(),
        )
        .await;
        assert!(matches!(r, Response::Matched { .. }));
    }
    #[tokio::test(start_paused = true)]
    async fn a_suspended_budget_ends_the_wait_at_once_without_retrying() {
        let (s, calls) = script(vec![
            AttemptOutcome::Suspended { retry_after_s: 30 },
            AttemptOutcome::Matched { similarity: 0.9 },
        ]);
        let r = wait_for_match(&s, None, "alice", LONG, None, &WaitTiming::new()).await;
        assert!(
            matches!(
                r,
                Response::NoMatch {
                    reason: NoMatchReason::Suspended
                }
            ),
            "{r:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn a_suspension_during_prearm_is_final_and_needs_no_lock() {
        let (s, calls) = script(vec![AttemptOutcome::Suspended { retry_after_s: 30 }]);
        let (_tx, rx) = watch::channel(false);
        let r = wait_for_match(
            &s,
            Some(rx),
            "alice",
            LONG,
            Some(Duration::from_secs(5)),
            &WaitTiming::new(),
        )
        .await;
        assert!(
            matches!(
                r,
                Response::NoMatch {
                    reason: NoMatchReason::Suspended
                }
            ),
            "{r:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// A source that rejects instantly (camera unavailable) must not turn the
    /// retry loop into a busy loop: attempts are spaced by `RETRY_PAUSE`.
    #[tokio::test(start_paused = true)]
    async fn rejected_attempts_are_spaced_not_back_to_back() {
        struct Instant_(Arc<AtomicUsize>);
        #[async_trait]
        impl DecisionSource for Instant_ {
            async fn attempt(&self, _u: &str) -> AttemptOutcome {
                self.0.fetch_add(1, Ordering::SeqCst);
                AttemptOutcome::NotMatched
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let src = Instant_(calls.clone());
        let r = wait_for_match(
            &src,
            None,
            "alice",
            Duration::from_secs(5),
            None,
            &WaitTiming::new(),
        )
        .await;
        assert!(matches!(r, Response::NoMatch { .. }));
        // At least 100 ms between attempts, independent of the constant's value.
        let max = 5_000 / 100;
        let n = calls.load(Ordering::SeqCst);
        assert!(n >= 2 && n <= max, "{n} attempts in 5 s (max {max})");
    }

    #[tokio::test(start_paused = true)]
    async fn rejected_attempts_in_prearm_are_spaced_too() {
        struct Instant_(Arc<AtomicUsize>);
        #[async_trait]
        impl DecisionSource for Instant_ {
            async fn attempt(&self, _u: &str) -> AttemptOutcome {
                self.0.fetch_add(1, Ordering::SeqCst);
                AttemptOutcome::NotMatched
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let src = Instant_(calls.clone());
        let (_tx, rx) = watch::channel(false);
        let r = wait_for_match(
            &src,
            Some(rx),
            "alice",
            Duration::from_secs(5),
            Some(Duration::from_secs(5)),
            &WaitTiming::new(),
        )
        .await;
        assert!(matches!(r, Response::NoMatch { .. }));
        // At least 100 ms between attempts, independent of the constant's value.
        let max = 5_000 / 100;
        let n = calls.load(Ordering::SeqCst);
        assert!(n >= 2 && n <= max, "{n} attempts in 5 s (max {max})");
    }
}
