//! Failure budget and escalating cooldown for face auth (`[ipc] face_*`).
//!
//! Every entry point (`Authenticate`, PAM, `doorman test`, each `WaitForMatch`
//! retry) goes through the same decision core, which consults this budget
//! first. After `max_failures` *judged* failures inside `window` face auth for
//! that user is suspended for the next step of `cooldowns`; a match resets the
//! budget, and so does a quiet period. The password path never sees it.
//!
//! Pure policy with an injected clock (`now`), no camera or model code. In
//! memory: a daemon restart clears it (a restart needs root).

use crate::decision::AuthOutcome;
use doorman_shared::IpcConfig;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Stable code of the suspension, in the wire reason, logs and `doorman test --json`.
pub const FACE_SUSPENDED_CODE: &str = "face_auth_suspended";
/// Tracing target of the budget's audit lines.
pub const AUDIT_TARGET: &str = "doorman::audit";

/// Whether a finished decision is a *judged* failure: a complete decision on a
/// face that rejected it. Camera and frame problems, a missing enrollment, no
/// face in view (`NoClearFace`), an unavailable liveness model and too few
/// frames are infrastructure or absence, not a failed attempt.
pub fn is_judged_failure(outcome: &AuthOutcome) -> bool {
    matches!(
        outcome,
        AuthOutcome::NotMatched
            | AuthOutcome::LivenessRejected
            | AuthOutcome::IdentityFloor
            | AuthOutcome::AmbiguousFaces
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaceBudgetSettings {
    /// `0` disables the budget.
    pub max_failures: u32,
    pub window: Duration,
    pub cooldowns: Vec<Duration>,
    pub quiet_reset: Duration,
}

impl FaceBudgetSettings {
    pub fn from_config(ipc: &IpcConfig) -> Self {
        Self {
            max_failures: ipc.face_max_judged_failures,
            window: Duration::from_secs(ipc.face_failure_window_secs),
            cooldowns: ipc
                .face_cooldown_secs
                .iter()
                .map(|s| Duration::from_secs(*s))
                .collect(),
            quiet_reset: Duration::from_secs(ipc.face_budget_quiet_reset_secs),
        }
    }
}

#[derive(Default)]
struct UserBudget {
    /// Judged failures inside the window.
    failures: VecDeque<Instant>,
    /// How many suspensions happened since the last reset (indexes `cooldowns`).
    level: usize,
    suspended_until: Option<Instant>,
    /// Time of the most recent judged failure, for the quiet-period reset.
    last_failure: Option<Instant>,
}

pub struct FaceBudget {
    settings: FaceBudgetSettings,
    users: Mutex<HashMap<String, UserBudget>>,
}

impl FaceBudget {
    pub fn new(settings: FaceBudgetSettings) -> Self {
        Self {
            settings,
            users: Mutex::new(HashMap::new()),
        }
    }

    fn enabled(&self) -> bool {
        self.settings.max_failures > 0 && !self.settings.cooldowns.is_empty()
    }

    /// `Err(remaining)` while face auth for `user` is suspended.
    pub fn check(&self, user: &str, now: Instant) -> Result<(), Duration> {
        if !self.enabled() {
            return Ok(());
        }
        let mut users = self.users.lock().unwrap_or_else(|e| e.into_inner());
        let Some(state) = users.get_mut(user) else {
            return Ok(());
        };
        match state.suspended_until {
            Some(until) if until > now => Err(until - now),
            Some(_) => {
                state.suspended_until = None;
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// Record one judged failure. Returns the cooldown when this failure used
    /// up the budget.
    pub fn record_failure(&self, user: &str, now: Instant) -> Option<Duration> {
        if !self.enabled() {
            return None;
        }
        let mut users = self.users.lock().unwrap_or_else(|e| e.into_inner());
        let state = users.entry(user.to_string()).or_default();
        if state
            .last_failure
            .is_some_and(|t| now.saturating_duration_since(t) >= self.settings.quiet_reset)
        {
            state.level = 0;
        }
        while state
            .failures
            .front()
            .is_some_and(|t| now.saturating_duration_since(*t) >= self.settings.window)
        {
            state.failures.pop_front();
        }
        state.failures.push_back(now);
        state.last_failure = Some(now);
        if state.failures.len() < self.settings.max_failures as usize {
            return None;
        }
        state.failures.clear();
        let step = state.level.min(self.settings.cooldowns.len() - 1);
        let cooldown = self.settings.cooldowns[step];
        state.level = state.level.saturating_add(1);
        state.suspended_until = Some(now + cooldown);
        Some(cooldown)
    }

    /// A match resets the user's budget completely.
    pub fn record_success(&self, user: &str) {
        let mut users = self.users.lock().unwrap_or_else(|e| e.into_inner());
        users.remove(user);
    }

    /// Account for a finished decision: a match resets, a judged failure
    /// counts, anything else is ignored. Returns the cooldown if one started.
    pub fn record(&self, user: &str, outcome: &AuthOutcome, now: Instant) -> Option<Duration> {
        if *outcome == AuthOutcome::Matched {
            self.record_success(user);
            None
        } else if is_judged_failure(outcome) {
            self.record_failure(user, now)
        } else {
            None
        }
    }
}

/// Round a remaining suspension up to whole seconds (never 0 while suspended).
pub fn retry_after_secs(remaining: Duration) -> u64 {
    remaining.as_secs() + u64::from(remaining.subsec_nanos() > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEC: Duration = Duration::from_secs(1);

    fn budget() -> FaceBudget {
        FaceBudget::new(FaceBudgetSettings::from_config(&IpcConfig::default()))
    }

    fn fail_n(b: &FaceBudget, n: u32, at: Instant) -> Option<Duration> {
        let mut last = None;
        for _ in 0..n {
            last = b.record_failure("u", at);
        }
        last
    }

    #[test]
    fn the_fifth_judged_failure_suspends_for_the_first_cooldown() {
        let b = budget();
        let t0 = Instant::now();
        assert_eq!(fail_n(&b, 4, t0), None);
        assert!(b.check("u", t0).is_ok());
        assert_eq!(b.record_failure("u", t0), Some(30 * SEC));
        let remaining = b.check("u", t0 + 10 * SEC).unwrap_err();
        assert_eq!(remaining, 20 * SEC);
        assert_eq!(retry_after_secs(remaining), 20);
        assert!(b.check("u", t0 + 30 * SEC).is_ok(), "cooldown over");
    }

    #[test]
    fn cooldowns_escalate_and_the_last_step_repeats() {
        let b = budget();
        let mut t = Instant::now();
        let mut seen = Vec::new();
        for _ in 0..5 {
            let cooldown = fail_n(&b, 5, t).expect("suspended");
            seen.push(cooldown.as_secs());
            t += cooldown + SEC;
            assert!(b.check("u", t).is_ok());
        }
        assert_eq!(seen, vec![30, 120, 600, 600, 600]);
    }

    #[test]
    fn a_match_resets_the_budget_and_the_escalation() {
        let b = budget();
        let t0 = Instant::now();
        fail_n(&b, 4, t0);
        b.record("u", &AuthOutcome::Matched, t0);
        assert_eq!(
            fail_n(&b, 4, t0),
            None,
            "the four earlier failures are gone"
        );
        // Escalation restarts too.
        assert_eq!(b.record_failure("u", t0), Some(30 * SEC));
        b.record("u", &AuthOutcome::Matched, t0 + 31 * SEC);
        assert_eq!(fail_n(&b, 5, t0 + 40 * SEC), Some(30 * SEC));
    }

    #[test]
    fn failures_outside_the_window_do_not_count() {
        let b = budget();
        let t0 = Instant::now();
        fail_n(&b, 4, t0);
        // 301 s later (window 300 s) the old failures are gone.
        assert_eq!(b.record_failure("u", t0 + 301 * SEC), None);
    }

    #[test]
    fn a_quiet_period_restarts_the_escalation() {
        let b = budget();
        let t0 = Instant::now();
        assert_eq!(fail_n(&b, 5, t0), Some(30 * SEC));
        let t1 = t0 + 31 * SEC;
        assert_eq!(fail_n(&b, 5, t1), Some(120 * SEC));
        // 1800 s without a judged failure: back to the first step.
        let t2 = t1 + 121 * SEC + Duration::from_secs(1_800);
        assert_eq!(fail_n(&b, 5, t2), Some(30 * SEC));
    }

    #[test]
    fn only_completed_rejections_are_judged_failures() {
        let judged = [
            AuthOutcome::NotMatched,
            AuthOutcome::LivenessRejected,
            AuthOutcome::IdentityFloor,
            AuthOutcome::AmbiguousFaces,
        ];
        let not_judged = [
            AuthOutcome::Matched,
            AuthOutcome::NotEnrolled,
            AuthOutcome::CameraUnavailable {
                message: "x".into(),
            },
            AuthOutcome::NoFrames,
            AuthOutcome::NoClearFace,
            AuthOutcome::LivenessError,
            AuthOutcome::InsufficientFrames,
            AuthOutcome::Suspended { retry_after_s: 1 },
        ];
        for o in &judged {
            assert!(is_judged_failure(o), "{o:?}");
        }
        for o in &not_judged {
            assert!(!is_judged_failure(o), "{o:?}");
        }
        let b = budget();
        let t0 = Instant::now();
        for _ in 0..20 {
            for o in &not_judged {
                assert_eq!(b.record("u", o, t0), None);
            }
        }
        assert!(b.check("u", t0).is_ok());
    }

    #[test]
    fn users_are_independent() {
        let b = budget();
        let t0 = Instant::now();
        fail_n(&b, 5, t0);
        assert!(b.check("u", t0).is_err());
        assert!(b.check("other", t0).is_ok());
    }

    #[test]
    fn zero_failures_disables_the_budget() {
        let ipc = IpcConfig {
            face_max_judged_failures: 0,
            ..IpcConfig::default()
        };
        let b = FaceBudget::new(FaceBudgetSettings::from_config(&ipc));
        let t0 = Instant::now();
        assert_eq!(fail_n(&b, 50, t0), None);
        assert!(b.check("u", t0).is_ok());
    }
}
