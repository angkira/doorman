//! Opt-in strict profile for `sudo` face authentication (`[pam.sudo]`).
//!
//! Pure policy, independent of camera and models: profile selection from the
//! PAM service name, the presence short circuit, the per-user failure
//! suspension (injectable clock), the frame freshness rule and the audit
//! helpers. `ipc.rs` applies them around the shared decision core.
//! See docs/SUDO.md.

use crate::decision::AuthOutcome;
use doorman_shared::Config;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Reason of the `Denied` answer while `[pam.sudo] enabled = false`.
pub const DENIED_SUDO_DISABLED: &str = "sudo face auth disabled";
/// Reason when the session is locked (nobody is at an unlocked desktop).
pub const DENIED_SUDO_LOCKED: &str = "sudo face auth unavailable while the session is locked";
/// Reason when the lock supervisor has no session to judge presence by.
pub const DENIED_SUDO_NO_SESSION: &str =
    "sudo face auth unavailable: no tracked session (fail closed)";
/// Reason while a user's sudo face auth is suspended after repeated failures.
pub const DENIED_SUDO_SUSPENDED: &str = "sudo face auth suspended after repeated failures";

/// Tracing target of the audit lines: `journalctl -u doormand | grep doorman::audit`.
pub const AUDIT_TARGET: &str = "doorman::audit";
/// Similarity is reported rounded to this step, never exactly.
pub const AUDIT_SIMILARITY_STEP: f32 = 0.05;

/// Audit reason codes (stable strings).
pub mod code {
    pub const DISABLED: &str = "disabled";
    pub const LOCKED: &str = "session_locked";
    pub const NO_SESSION: &str = "no_session";
    pub const SUSPENDED: &str = "suspended";
    pub const MATCHED: &str = "matched";
    pub const NOT_MATCHED: &str = "not_matched";
    pub const NO_CLEAR_FACE: &str = "no_clear_face";
    pub const LIVENESS_REJECTED: &str = "liveness_rejected";
    pub const LIVENESS_UNAVAILABLE: &str = "liveness_unavailable";
    pub const NO_FRAMES: &str = "no_frames";
    pub const CAMERA_UNAVAILABLE: &str = "camera_unavailable";
    pub const NOT_ENROLLED: &str = "not_enrolled";
}

/// Settings of the strict profile, resolved from the config.
#[derive(Debug, Clone, PartialEq)]
pub struct StrictProfile {
    pub threshold: f32,
    pub accept_n: usize,
    pub window_m: usize,
    pub liveness_frames: usize,
    pub timeout: Duration,
}

/// Which policy an `Authenticate` runs under.
#[derive(Debug, Clone, PartialEq)]
pub enum Profile {
    /// The existing lock-screen behaviour, untouched.
    Default,
    Strict(StrictProfile),
    Denied(&'static str),
}

/// Pick the profile from the claimed PAM service.
///
/// The peer uid is deliberately not an input. PAM runs as root, but a non-root
/// peer that claims a sudo service name only gets the stricter profile (or the
/// disabled denial), which is harmless; claiming any other service name yields
/// the default profile, which that peer could already get by sending no service
/// at all. Nothing is gained by lying, so the claim needs no authorization.
pub fn select_profile(service: Option<&str>, cfg: &Config) -> Profile {
    let sudo = &cfg.pam.sudo;
    match service {
        Some(name) if sudo.service_names.iter().any(|n| n == name) => {
            if !sudo.enabled {
                return Profile::Denied(DENIED_SUDO_DISABLED);
            }
            Profile::Strict(StrictProfile {
                threshold: sudo.effective_threshold(&cfg.authentication),
                accept_n: sudo.accept_n,
                window_m: sudo.window_m,
                liveness_frames: sudo.liveness_frames,
                timeout: Duration::from_millis(sudo.timeout_ms),
            })
        }
        _ => Profile::Default,
    }
}

/// What the lock supervisor knows about the seat session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    Unlocked,
    Locked,
    /// No supervisor, or it found no session to track.
    Unavailable,
}

/// Presence short circuit: `Err((audit code, denial reason))` unless the
/// tracked session is unlocked. Fails closed.
pub fn check_presence(presence: Presence) -> Result<(), (&'static str, &'static str)> {
    match presence {
        Presence::Unlocked => Ok(()),
        Presence::Locked => Err((code::LOCKED, DENIED_SUDO_LOCKED)),
        Presence::Unavailable => Err((code::NO_SESSION, DENIED_SUDO_NO_SESSION)),
    }
}

/// A frame is usable by the strict profile only if it was captured at or after
/// the moment the request arrived.
pub fn is_fresh(captured: Instant, request_start: Instant) -> bool {
    captured >= request_start
}

/// How long `acquire_camera` may wait for a frame: the default wait, capped by
/// the time left until the strict deadline. `None` once the deadline has passed
/// (the caller must fail at once instead of opening the camera).
pub fn camera_acquire_wait(
    default_wait: Duration,
    strict_deadline: Option<Instant>,
    now: Instant,
) -> Option<Duration> {
    match strict_deadline {
        None => Some(default_wait),
        Some(deadline) => {
            let remaining = deadline.saturating_duration_since(now);
            if remaining.is_zero() {
                None
            } else {
                Some(default_wait.min(remaining))
            }
        }
    }
}

#[derive(Default)]
struct UserFailures {
    /// Failure times inside the counting window.
    failures: VecDeque<Instant>,
    suspended_until: Option<Instant>,
}

/// Per-username suspension: `max_failures` consecutive failures within
/// `window` suspend sudo face auth for `window`; a success resets the count.
/// The clock is injected by the caller (`now`). In memory, reset on restart.
pub struct SudoGuard {
    max_failures: u32,
    window: Duration,
    users: Mutex<HashMap<String, UserFailures>>,
}

impl SudoGuard {
    pub fn new(max_failures: u32, window: Duration) -> Self {
        Self {
            max_failures,
            window,
            users: Mutex::new(HashMap::new()),
        }
    }

    /// `Err(retry_after_s)` while `user` is suspended.
    pub fn check(&self, user: &str, now: Instant) -> Result<(), u64> {
        let mut users = self.users.lock().unwrap_or_else(|e| e.into_inner());
        let Some(state) = users.get_mut(user) else {
            return Ok(());
        };
        match state.suspended_until {
            Some(until) if until > now => {
                let wait = until - now;
                Err(wait.as_secs() + u64::from(wait.subsec_nanos() > 0))
            }
            Some(_) => {
                state.suspended_until = None;
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// Record a failed strict attempt; suspends once the budget is used up.
    pub fn record_failure(&self, user: &str, now: Instant) {
        let mut users = self.users.lock().unwrap_or_else(|e| e.into_inner());
        let state = users.entry(user.to_string()).or_default();
        while state
            .failures
            .front()
            .is_some_and(|t| now.saturating_duration_since(*t) >= self.window)
        {
            state.failures.pop_front();
        }
        state.failures.push_back(now);
        if state.failures.len() >= self.max_failures as usize {
            state.failures.clear();
            state.suspended_until = Some(now + self.window);
        }
    }

    /// A success resets the user's failure count.
    pub fn record_success(&self, user: &str) {
        let mut users = self.users.lock().unwrap_or_else(|e| e.into_inner());
        users.remove(user);
    }
}

/// Audit reason code of a finished decision.
pub fn outcome_code(outcome: &AuthOutcome) -> &'static str {
    match outcome {
        AuthOutcome::Matched => code::MATCHED,
        AuthOutcome::NotMatched => code::NOT_MATCHED,
        AuthOutcome::NotEnrolled => code::NOT_ENROLLED,
        AuthOutcome::CameraUnavailable { .. } => code::CAMERA_UNAVAILABLE,
        AuthOutcome::NoFrames => code::NO_FRAMES,
        AuthOutcome::NoClearFace => code::NO_CLEAR_FACE,
        AuthOutcome::LivenessRejected => code::LIVENESS_REJECTED,
        AuthOutcome::LivenessError => code::LIVENESS_UNAVAILABLE,
    }
}

/// Whether a finished decision counts towards the suspension: the face (or
/// its absence or liveness) was judged. Missing enrollment and camera or
/// frame problems are infrastructure, not failed attempts.
pub fn counts_as_failure(outcome: &AuthOutcome) -> bool {
    matches!(
        outcome,
        AuthOutcome::NotMatched
            | AuthOutcome::NoClearFace
            | AuthOutcome::LivenessRejected
            | AuthOutcome::LivenessError
    )
}

/// `similarity` rounded to [`AUDIT_SIMILARITY_STEP`], for the audit line.
pub fn similarity_bucket(similarity: f32) -> f32 {
    (similarity / AUDIT_SIMILARITY_STEP).round() * AUDIT_SIMILARITY_STEP
}

/// One audit line per strict attempt. Carries no frames or embeddings and only
/// the bucketed similarity.
pub fn audit(
    service: &str,
    uid: u32,
    user: &str,
    success: bool,
    reason: &str,
    similarity: Option<f32>,
) {
    let bucket = similarity.map(|s| format!("{:.2}", similarity_bucket(s)));
    tracing::info!(
        target: "doorman::audit",
        service,
        uid,
        user,
        result = if success { "success" } else { "failure" },
        reason,
        similarity_bucket = bucket.as_deref().unwrap_or("none"),
        "sudo face auth attempt"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_acquire_wait_is_bounded_by_strict_deadline() {
        let now = Instant::now();
        let default_wait = Duration::from_secs(8);
        // Non-strict: unchanged default.
        assert_eq!(
            camera_acquire_wait(default_wait, None, now),
            Some(default_wait)
        );
        // Strict, plenty of time beyond the default: default wins.
        let far = now + Duration::from_secs(20);
        assert_eq!(
            camera_acquire_wait(default_wait, Some(far), now),
            Some(default_wait)
        );
        // Strict 4 s deadline, 1 s already spent: 3 s left.
        let deadline = now + Duration::from_secs(4);
        let later = now + Duration::from_secs(1);
        assert_eq!(
            camera_acquire_wait(default_wait, Some(deadline), later),
            Some(Duration::from_secs(3))
        );
        // Deadline reached or passed: no wait at all.
        assert_eq!(
            camera_acquire_wait(default_wait, Some(deadline), deadline),
            None
        );
        assert_eq!(
            camera_acquire_wait(
                default_wait,
                Some(deadline),
                deadline + Duration::from_secs(1)
            ),
            None
        );
    }

    fn cfg(enabled: bool) -> Config {
        let mut c = Config::default();
        c.pam.sudo.enabled = enabled;
        c
    }

    #[test]
    fn profile_selection_table() {
        let mut custom = cfg(true);
        custom.pam.sudo.service_names = vec!["sudo".into(), "sudo-i".into()];
        custom.pam.sudo.similarity_threshold = Some(0.55);
        let off = cfg(false);
        let on = cfg(true);

        let strict_default = Profile::Strict(StrictProfile {
            threshold: 0.4,
            accept_n: 4,
            window_m: 5,
            liveness_frames: 3,
            timeout: Duration::from_millis(4000),
        });
        let cases: Vec<(&str, Option<&str>, &Config, Profile)> = vec![
            ("no service, off", None, &off, Profile::Default),
            ("no service, on", None, &on, Profile::Default),
            ("login, off", Some("login"), &off, Profile::Default),
            ("login, on", Some("login"), &on, Profile::Default),
            ("sudo-i not listed", Some("sudo-i"), &on, Profile::Default),
            ("case differs", Some("Sudo"), &on, Profile::Default),
            ("empty service", Some(""), &on, Profile::Default),
            (
                "sudo, off",
                Some("sudo"),
                &off,
                Profile::Denied(DENIED_SUDO_DISABLED),
            ),
            ("sudo, on", Some("sudo"), &on, strict_default),
            (
                "sudo-i listed, threshold override",
                Some("sudo-i"),
                &custom,
                Profile::Strict(StrictProfile {
                    threshold: 0.55,
                    accept_n: 4,
                    window_m: 5,
                    liveness_frames: 3,
                    timeout: Duration::from_millis(4000),
                }),
            ),
        ];
        for (name, service, config, want) in cases {
            assert_eq!(select_profile(service, config), want, "{name}");
        }
    }

    #[test]
    fn disabled_sudo_message_is_stable() {
        assert_eq!(DENIED_SUDO_DISABLED, "sudo face auth disabled");
    }

    #[test]
    fn presence_fails_closed() {
        assert!(check_presence(Presence::Unlocked).is_ok());
        assert_eq!(
            check_presence(Presence::Locked).unwrap_err().0,
            code::LOCKED
        );
        assert_eq!(
            check_presence(Presence::Unavailable).unwrap_err().0,
            code::NO_SESSION
        );
    }

    #[test]
    fn freshness_uses_injected_timestamps() {
        let start = Instant::now();
        let before = start - Duration::from_millis(1);
        let after = start + Duration::from_millis(1);
        assert!(!is_fresh(before, start), "frame older than the request");
        assert!(is_fresh(start, start));
        assert!(is_fresh(after, start));
    }

    const WINDOW: Duration = Duration::from_secs(600);

    #[test]
    fn suspends_after_max_failures_and_recovers() {
        let g = SudoGuard::new(3, WINDOW);
        let t0 = Instant::now();
        for i in 0..2 {
            g.record_failure("alice", t0 + Duration::from_secs(i));
            assert!(g.check("alice", t0 + Duration::from_secs(i)).is_ok());
        }
        g.record_failure("alice", t0 + Duration::from_secs(2));
        assert_eq!(g.check("alice", t0 + Duration::from_secs(3)), Err(599));
        // Other users are unaffected.
        assert!(g.check("bob", t0 + Duration::from_secs(3)).is_ok());
        // Suspension ends after the window, with a fresh failure budget.
        let later = t0 + Duration::from_secs(2) + WINDOW;
        assert!(g.check("alice", later).is_ok());
        g.record_failure("alice", later);
        assert!(g.check("alice", later).is_ok());
    }

    #[test]
    fn success_resets_the_count() {
        let g = SudoGuard::new(3, WINDOW);
        let t0 = Instant::now();
        g.record_failure("alice", t0);
        g.record_failure("alice", t0);
        g.record_success("alice");
        g.record_failure("alice", t0);
        g.record_failure("alice", t0);
        assert!(g.check("alice", t0).is_ok(), "only 2 consecutive failures");
    }

    #[test]
    fn failures_outside_the_window_do_not_count() {
        let g = SudoGuard::new(3, WINDOW);
        let t0 = Instant::now();
        g.record_failure("alice", t0);
        g.record_failure("alice", t0);
        g.record_failure("alice", t0 + WINDOW);
        assert!(g.check("alice", t0 + WINDOW).is_ok());
    }

    #[test]
    fn only_judged_faces_count_as_failures() {
        assert!(counts_as_failure(&AuthOutcome::NotMatched));
        assert!(counts_as_failure(&AuthOutcome::NoClearFace));
        assert!(counts_as_failure(&AuthOutcome::LivenessRejected));
        assert!(counts_as_failure(&AuthOutcome::LivenessError));
        assert!(!counts_as_failure(&AuthOutcome::Matched));
        assert!(!counts_as_failure(&AuthOutcome::NotEnrolled));
        assert!(!counts_as_failure(&AuthOutcome::NoFrames));
        assert!(!counts_as_failure(&AuthOutcome::CameraUnavailable {
            message: String::new()
        }));
        assert_eq!(
            outcome_code(&AuthOutcome::LivenessError),
            code::LIVENESS_UNAVAILABLE
        );
    }

    #[test]
    fn similarity_is_bucketed_to_five_hundredths() {
        assert!((similarity_bucket(0.812) - 0.80).abs() < 1e-6);
        assert!((similarity_bucket(0.826) - 0.85).abs() < 1e-6);
        assert!((similarity_bucket(-0.01) - 0.0).abs() < 1e-6);
    }
}
