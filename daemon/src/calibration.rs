//! Calibration mode: a root-only, time-limited, in-memory relaxation used to
//! measure presentation-attack behaviour (`sudo doorman calibrate`).
//!
//! # What it relaxes (and nothing else)
//!
//! 1. Shadow PAD logging is forced on (log-only, same collector as
//!    `[liveness] shadow_pad_logging`).
//! 2. The face failure budget ([`crate::face_budget::FaceBudget`]) is
//!    *frozen*: judged failures neither count toward a suspension nor reset
//!    it, and an existing suspension is not enforced. Judged failures are
//!    still counted here, in a separate counter.
//!
//! No threshold, liveness rule, N-of-M, identity floor, multi-face policy,
//! frame minimum or model-integrity check consults this module.
//!
//! # Why it fails safe
//!
//! * The mode is *derived from time*: it is active only while
//!   `now < expires`. A late or stuck expiry task cannot extend it.
//! * It lives in process memory only. A daemon restart ends it.
//! * Starting and ending it is root-only (`ipc_policy::authorize`).
//! * Because the budget is frozen, not rewritten, "restoring" it is the
//!   absence of an action: nothing was changed that has to be put back.
//!
//! Time is passed in (`now`), so tests use a fake clock.

use doorman_shared::CalibrationInfo;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Tracing target of the calibration audit lines.
pub const AUDIT_TARGET: &str = "doorman::calibration";

#[derive(Debug, Clone, Copy)]
struct Session {
    started: Instant,
    expires: Instant,
    next_warn: Instant,
    counted_failures: u64,
}

/// What the periodic task has to report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tick {
    /// Not active, nothing to say.
    Idle,
    /// Active and the warning interval elapsed.
    Warn { remaining_secs: u64 },
    /// The session ran out since the last tick (it is already inactive).
    Expired { counted_failures: u64 },
}

/// Shared calibration state.
#[derive(Debug, Default)]
pub struct Calibration {
    session: Mutex<Option<Session>>,
}

fn round_up_secs(d: Duration) -> u64 {
    d.as_secs() + u64::from(d.subsec_nanos() > 0)
}

impl Calibration {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Session>> {
        self.session.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Start (or re-arm: the new request replaces the session) a session of
    /// `ttl` from `now`.
    pub fn start(&self, now: Instant, ttl: Duration, warn_interval: Duration) -> CalibrationInfo {
        let session = Session {
            started: now,
            expires: now + ttl,
            next_warn: now + warn_interval,
            counted_failures: 0,
        };
        *self.lock() = Some(session);
        info_of(&session, now)
    }

    /// End the session now. Returns the failures counted, or `None` if none was active.
    pub fn end(&self, now: Instant) -> Option<u64> {
        let mut guard = self.lock();
        let ended = guard.take()?;
        (now < ended.expires).then_some(ended.counted_failures)
    }

    /// Whether calibration relaxations apply at `now`. Pure function of time.
    pub fn is_active(&self, now: Instant) -> bool {
        self.lock().is_some_and(|s| now < s.expires)
    }

    pub fn status(&self, now: Instant) -> CalibrationInfo {
        match *self.lock() {
            Some(s) if now < s.expires => info_of(&s, now),
            _ => inactive(),
        }
    }

    /// Seconds left, `None` when not active.
    pub fn remaining_secs(&self, now: Instant) -> Option<u64> {
        let info = self.status(now);
        info.active.then_some(info.remaining_secs)
    }

    /// Count one judged failure that the frozen budget did not take.
    /// Returns whether it was counted (the mode was active).
    pub fn count_judged_failure(&self, now: Instant) -> bool {
        match self.lock().as_mut() {
            Some(s) if now < s.expires => {
                s.counted_failures += 1;
                true
            }
            _ => false,
        }
    }

    /// Advance the periodic bookkeeping: report expiry once, and a warning each
    /// `warn_interval` while active.
    pub fn tick(&self, now: Instant, warn_interval: Duration) -> Tick {
        let mut guard = self.lock();
        let Some(s) = guard.as_mut() else {
            return Tick::Idle;
        };
        if now >= s.expires {
            let counted = s.counted_failures;
            *guard = None;
            return Tick::Expired {
                counted_failures: counted,
            };
        }
        if now >= s.next_warn {
            s.next_warn = now + warn_interval;
            return Tick::Warn {
                remaining_secs: round_up_secs(s.expires - now),
            };
        }
        Tick::Idle
    }
}

/// Parse a session length typed by the operator: `45m`, `2700s`, `2h`, `1h30m`;
/// a bare number is minutes. Range checking is the daemon's job.
pub fn parse_ttl(text: &str) -> Result<u64, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("empty duration".to_string());
    }
    if let Ok(minutes) = text.parse::<u64>() {
        return minutes
            .checked_mul(60)
            .ok_or_else(|| format!("`{text}` is too large"));
    }
    let mut total: u64 = 0;
    let mut digits = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            digits.push(c);
            continue;
        }
        let unit = match c.to_ascii_lowercase() {
            'h' => 3600,
            'm' => 60,
            's' => 1,
            _ => return Err(format!("`{text}`: use forms like 45m, 2700s, 2h or 1h30m")),
        };
        let n: u64 = digits
            .parse()
            .map_err(|_| format!("`{text}`: a number must come before `{c}`"))?;
        digits.clear();
        total = n
            .checked_mul(unit)
            .and_then(|v| total.checked_add(v))
            .ok_or_else(|| format!("`{text}` is too large"))?;
    }
    if !digits.is_empty() {
        return Err(format!(
            "`{text}`: the trailing {digits} needs a unit (h, m or s)"
        ));
    }
    Ok(total)
}

fn inactive() -> CalibrationInfo {
    CalibrationInfo {
        active: false,
        remaining_secs: 0,
        ttl_secs: 0,
        judged_failures_counted: 0,
    }
}

fn info_of(s: &Session, now: Instant) -> CalibrationInfo {
    CalibrationInfo {
        active: true,
        remaining_secs: round_up_secs(s.expires.saturating_duration_since(now)),
        ttl_secs: (s.expires - s.started).as_secs(),
        judged_failures_counted: s.counted_failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: Duration = Duration::from_secs(60);

    #[test]
    fn inactive_until_started_and_after_the_ttl() {
        let c = Calibration::new();
        let t0 = Instant::now();
        assert!(!c.is_active(t0));
        c.start(t0, 45 * MIN, 5 * MIN);
        assert!(c.is_active(t0));
        assert!(c.is_active(t0 + 45 * MIN - Duration::from_secs(1)));
        assert!(
            !c.is_active(t0 + 45 * MIN),
            "expiry is derived from time, no task needed"
        );
        assert_eq!(c.status(t0 + 46 * MIN), inactive());
    }

    #[test]
    fn remaining_time_counts_down_and_rounds_up() {
        let c = Calibration::new();
        let t0 = Instant::now();
        let info = c.start(t0, 10 * MIN, 5 * MIN);
        assert_eq!((info.remaining_secs, info.ttl_secs), (600, 600));
        assert_eq!(
            c.remaining_secs(t0 + Duration::from_millis(1500)),
            Some(599)
        );
        assert_eq!(c.remaining_secs(t0 + 10 * MIN), None);
    }

    #[test]
    fn end_stops_it_and_reports_the_counter_once() {
        let c = Calibration::new();
        let t0 = Instant::now();
        c.start(t0, 10 * MIN, 5 * MIN);
        assert!(c.count_judged_failure(t0));
        assert!(c.count_judged_failure(t0));
        assert_eq!(c.end(t0 + MIN), Some(2));
        assert!(!c.is_active(t0 + MIN));
        assert_eq!(c.end(t0 + MIN), None, "second end is a no-op");
        assert!(
            !c.count_judged_failure(t0 + MIN),
            "nothing counted when off"
        );
    }

    #[test]
    fn tick_warns_periodically_then_reports_expiry_exactly_once() {
        let c = Calibration::new();
        let t0 = Instant::now();
        let warn = 5 * MIN;
        c.start(t0, 12 * MIN, warn);
        assert_eq!(c.tick(t0 + MIN, warn), Tick::Idle);
        assert_eq!(
            c.tick(t0 + 5 * MIN, warn),
            Tick::Warn {
                remaining_secs: 7 * 60
            }
        );
        assert_eq!(c.tick(t0 + 6 * MIN, warn), Tick::Idle);
        assert_eq!(
            c.tick(t0 + 10 * MIN, warn),
            Tick::Warn {
                remaining_secs: 2 * 60
            }
        );
        c.count_judged_failure(t0 + 11 * MIN);
        assert_eq!(
            c.tick(t0 + 13 * MIN, warn),
            Tick::Expired {
                counted_failures: 1
            }
        );
        assert_eq!(c.tick(t0 + 14 * MIN, warn), Tick::Idle);
    }

    #[test]
    fn rearming_replaces_the_session_and_resets_the_counter() {
        let c = Calibration::new();
        let t0 = Instant::now();
        c.start(t0, 10 * MIN, 5 * MIN);
        c.count_judged_failure(t0);
        let info = c.start(t0 + 9 * MIN, 10 * MIN, 5 * MIN);
        assert_eq!(info.judged_failures_counted, 0);
        assert!(c.is_active(t0 + 18 * MIN));
        assert!(!c.is_active(t0 + 19 * MIN));
    }

    #[test]
    fn ttl_parsing() {
        for (text, want) in [
            ("45m", 2700),
            ("45", 2700),
            ("2700s", 2700),
            ("2h", 7200),
            ("1h30m", 5400),
            ("1H", 3600),
            (" 10m ", 600),
        ] {
            assert_eq!(parse_ttl(text), Ok(want), "{text}");
        }
        for bad in [
            "",
            "m",
            "-5m",
            "4x",
            "1h30",
            "abc",
            "1.5h",
            "99999999999999999999m",
        ] {
            assert!(parse_ttl(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn an_expired_session_cannot_be_ended_as_if_active() {
        let c = Calibration::new();
        let t0 = Instant::now();
        c.start(t0, MIN, 5 * MIN);
        assert_eq!(c.end(t0 + 2 * MIN), None);
    }
}
