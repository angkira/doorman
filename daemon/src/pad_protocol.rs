//! `doorman pad run`: guided presentation-attack screening, native port of
//! `scripts/pad_protocol.py`.
//!
//! The owner sits in front of the camera with a phone and a printed photo; the
//! session walks through genuine and attack classes, asks the daemon for one
//! authentication per attempt (the same IPC request as `doorman test`) and
//! records only the decision, reason code, similarity bucket, liveness flag,
//! duration and time. It never captures, stores or shows a frame.
//!
//! The session logic is independent of the socket and the terminal: it talks to
//! a [`Backend`] (the daemon) and a [`Terminal`] (prompts, clock, Ctrl-C), so
//! the tests drive it with scripted fakes and a fake clock.
//!
//! Results are SCREENING, not certification: a handful of attempts per class
//! gives wide bounds (0 of 10 accepted still allows a true rate of 25.9 %).

use crate::pad_stats;
use crate::test_report::TestReport;
use chrono::{DateTime, Utc};
use doorman_shared::CalibrationInfo;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Version of the JSON result format.
pub const PROTOCOL_VERSION: u32 = 1;
/// Printed (verbatim) when calibration mode is not active at the start.
pub const NO_CALIBRATION_NOTICE: &str = "Для 20-минутной сессии без пауз: sudo doorman calibrate";

pub const DEFAULT_GENUINE: usize = 10;
pub const DEFAULT_ATTACK: usize = 10;
/// Per genuine-condition class (dark, backlight, glasses, yaw, distance).
pub const DEFAULT_CONDITION: usize = 5;
pub const QUICK_GENUINE: usize = 3;
pub const QUICK_ATTACK: usize = 2;
pub const QUICK_CONDITION: usize = 2;

const COUNTDOWN: Duration = Duration::from_secs(3);
const PAUSE_BETWEEN_ATTEMPTS: Duration = Duration::from_secs(2);
const COOLDOWN_MARGIN: Duration = Duration::from_secs(2);
const FALLBACK_COOLDOWN: Duration = Duration::from_secs(30);
const COOLDOWN_STEP: Duration = Duration::from_secs(15);
/// Stop after this many infrastructure errors in a row.
const MAX_CONSECUTIVE_ERRORS: usize = 3;
/// Warn when the calibration mode has this long left.
const CALIBRATION_LOW_WATER: Duration = Duration::from_secs(5 * 60);
/// How long an attempt waits for its shadow record to show up in the log.
const SHADOW_WAIT: Duration = Duration::from_millis(1500);
const SHADOW_POLL: Duration = Duration::from_millis(100);
const SHADOW_MATCH_BEFORE_MS: i64 = 1000;
const SHADOW_MATCH_AFTER_MS: i64 = 4000;

/// Reason codes that mean the system found nothing to judge.
const NON_RESPONSE_CODES: [&str; 2] = ["no_clear_face", "no_frames"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Genuine,
    Attack,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Baseline,
    Attack,
    Condition,
}

#[derive(Debug, Clone, Copy)]
pub struct AttackClass {
    pub name: &'static str,
    pub kind: Kind,
    pub group: Group,
    pub title: &'static str,
    pub instruction: &'static str,
}

const fn class(
    name: &'static str,
    kind: Kind,
    group: Group,
    title: &'static str,
    instruction: &'static str,
) -> AttackClass {
    AttackClass {
        name,
        kind,
        group,
        title,
        instruction,
    }
}

/// Every class, in the order they are presented.
pub const CLASSES: [AttackClass; 13] = [
    class("genuine", Kind::Genuine, Group::Baseline, "Genuine owner, baseline",
        "Look at the camera normally, at your usual distance, good light."),
    class("phone_photo", Kind::Attack, Group::Attack, "Phone: still photo of the owner",
        "Show a photo of YOUR face on the phone screen, filling the view, brightness at maximum. No live face in view."),
    class("phone_replay", Kind::Attack, Group::Attack, "Phone: video replay of the owner",
        "Play a video of YOUR face on the phone (blinking, moving), screen to the camera, maximum brightness. No live face in view."),
    class("print_matte", Kind::Attack, Group::Attack, "Printed photo of the owner",
        "Hold a printed photo of YOUR face (face about 8-10 cm wide) in front of the camera. Move it slightly. No live face in view."),
    class("mixed_track", Kind::Attack, Group::Attack, "Photo next to a live face",
        "Show your LIVE face and, beside it in the same view, a phone or print with a face (yours or a lookalike)."),
    class("genuine_dark", Kind::Genuine, Group::Condition, "Genuine: dark room",
        "Switch the room light off or dim it a lot; look at the camera."),
    class("genuine_backlight", Kind::Genuine, Group::Condition, "Genuine: backlight",
        "Sit with a bright window or lamp behind you; look at the camera."),
    class("genuine_glasses_on", Kind::Genuine, Group::Condition, "Genuine: glasses on",
        "Wear your glasses (no glasses: skip this class)."),
    class("genuine_glasses_off", Kind::Genuine, Group::Condition, "Genuine: glasses off",
        "Take your glasses off (no glasses: skip this class)."),
    class("genuine_yaw_left", Kind::Genuine, Group::Condition, "Genuine: head turned about 20 degrees left",
        "Turn your head about 20 degrees to the left, eyes on the camera."),
    class("genuine_yaw_right", Kind::Genuine, Group::Condition, "Genuine: head turned about 20 degrees right",
        "Turn your head about 20 degrees to the right, eyes on the camera."),
    class("genuine_near", Kind::Genuine, Group::Condition, "Genuine: near",
        "Sit close: about 30 cm from the camera."),
    class("genuine_far", Kind::Genuine, Group::Condition, "Genuine: far",
        "Sit far: about 1 m from the camera."),
];

/// Attempts per group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    pub genuine: usize,
    pub attack: usize,
    pub condition: usize,
}

impl Counts {
    pub const FULL: Counts = Counts {
        genuine: DEFAULT_GENUINE,
        attack: DEFAULT_ATTACK,
        condition: DEFAULT_CONDITION,
    };
    /// `doorman pad run --quick`: a smoke test, bounds are very wide.
    pub const QUICK: Counts = Counts {
        genuine: QUICK_GENUINE,
        attack: QUICK_ATTACK,
        condition: QUICK_CONDITION,
    };

    fn for_group(self, group: Group) -> usize {
        match group {
            Group::Baseline => self.genuine,
            Group::Attack => self.attack,
            Group::Condition => self.condition,
        }
    }
}

// ---------------------------------------------------------------------------
// Backend and terminal
// ---------------------------------------------------------------------------

/// The daemon, as the session sees it.
pub trait Backend {
    /// One authentication attempt, classified like `doorman --json test`.
    fn attempt(&mut self) -> TestReport;
    /// The calibration mode state; `None` when the daemon cannot be asked.
    fn calibration(&mut self) -> Option<CalibrationInfo>;
}

/// Prompts, clock and Ctrl-C.
pub trait Terminal {
    fn say(&mut self, line: &str);
    /// One line typed by the user; `None` on end of input or Ctrl-C.
    fn ask(&mut self, prompt: &str) -> Option<String>;
    /// Sleep; returns `false` when interrupted (Ctrl-C) during the sleep.
    fn sleep(&mut self, d: Duration) -> bool;
    fn interrupted(&self) -> bool;
    fn now(&self) -> DateTime<Utc>;
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

/// Aggregates of the daemon's shadow record for one attempt.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ShadowSummary {
    pub attempt_id: Option<String>,
    pub frames_judged: Option<u64>,
    pub similarity_median: Option<f64>,
    pub similarity_min: Option<f64>,
    pub depth_relief_min: Option<f64>,
    pub depth_margin_min: Option<f64>,
    pub minifasnet_state: Option<String>,
    pub minifasnet_logit_diff: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Attempt {
    pub class: String,
    pub kind: Kind,
    pub index: usize,
    pub decision: String,
    pub reason_code: String,
    pub similarity_bucket: Option<f32>,
    pub liveness_ok: Option<bool>,
    pub duration_ms: u64,
    pub timestamp: String,
    /// Calibration mode was active when the attempt started.
    pub calibration: bool,
    pub shadow: Option<ShadowSummary>,
}

impl Attempt {
    fn is_error(&self) -> bool {
        !matches!(self.decision.as_str(), "recognized" | "not_recognized")
    }
    fn non_response(&self) -> bool {
        self.decision == "not_recognized" && NON_RESPONSE_CODES.contains(&self.reason_code.as_str())
    }
}

// ---------------------------------------------------------------------------
// Shadow log reader
// ---------------------------------------------------------------------------

/// Reader of records appended to the shadow log after it was opened.
///
/// Availability means the file can be OPENED FOR READING, not that it can be
/// stat-ed: the log is `doorman:doorman 0640` in a world-listable directory, so
/// `stat` succeeds for users who cannot read a byte.
pub struct ShadowLog {
    path: PathBuf,
    offset: u64,
}

impl ShadowLog {
    pub fn open(path: &Path) -> Option<Self> {
        let file = std::fs::File::open(path).ok()?;
        let offset = file.metadata().ok()?.len();
        Some(Self {
            path: path.to_path_buf(),
            offset,
        })
    }

    pub fn new_records(&mut self) -> Vec<serde_json::Value> {
        let Ok(mut file) = std::fs::File::open(&self.path) else {
            return Vec::new();
        };
        let Ok(len) = file.metadata().map(|m| m.len()) else {
            return Vec::new();
        };
        if len < self.offset {
            self.offset = 0; // rotated
        }
        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            return Vec::new();
        }
        let mut data = Vec::new();
        if file.read_to_end(&mut data).is_err() {
            return Vec::new();
        }
        self.offset += data.len() as u64;
        String::from_utf8_lossy(&data)
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter(serde_json::Value::is_object)
            .collect()
    }
}

fn summarize_shadow(rec: &serde_json::Value) -> ShadowSummary {
    let f = |v: &serde_json::Value| v.as_f64();
    let relief = &rec["depth"]["relief"];
    ShadowSummary {
        attempt_id: rec["attempt_id"].as_str().map(str::to_string),
        frames_judged: rec["frames"]["judged"].as_u64(),
        similarity_median: f(&rec["similarity"]["per_frame"]["median"]),
        similarity_min: f(&rec["similarity"]["per_frame"]["min"]),
        depth_relief_min: f(&relief["min"]),
        depth_margin_min: f(&rec["depth"]["margin_min"]),
        minifasnet_state: rec["minifasnet"]["state"].as_str().map(str::to_string),
        minifasnet_logit_diff: f(&rec["minifasnet"]["logit_diff"]),
    }
}

/// The last `source = test` record whose timestamp falls inside the attempt
/// (with some slack on both sides).
pub fn match_shadow(
    records: &[serde_json::Value],
    started_ms: i64,
    ended_ms: i64,
) -> Option<ShadowSummary> {
    let (lo, hi) = (
        started_ms - SHADOW_MATCH_BEFORE_MS,
        ended_ms + SHADOW_MATCH_AFTER_MS,
    );
    records
        .iter()
        .filter(|r| r["source"].as_str().is_none_or(|s| s == "test"))
        .filter(|r| {
            r["timestamp_unix_ms"]
                .as_i64()
                .is_some_and(|t| (lo..=hi).contains(&t))
        })
        .next_back()
        .map(summarize_shadow)
}

// ---------------------------------------------------------------------------
// The session
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct Suspension {
    pub at: String,
    pub wait_s: u64,
    pub why: String,
}

/// Why the session ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Ending {
    Completed,
    Interrupted,
    TooManyErrors,
    NotStarted,
}

pub struct Session<'a> {
    pub backend: &'a mut dyn Backend,
    pub term: &'a mut dyn Terminal,
    pub counts: Counts,
    pub shadow: Option<ShadowLog>,
    pub attempts: Vec<Attempt>,
    pub suspensions: Vec<Suspension>,
    pub skipped_classes: Vec<String>,
    pub started: DateTime<Utc>,
    pub ending: Ending,
    low_water_warned: bool,
    was_calibrating: bool,
}

impl<'a> Session<'a> {
    pub fn new(
        backend: &'a mut dyn Backend,
        term: &'a mut dyn Terminal,
        counts: Counts,
        shadow: Option<ShadowLog>,
    ) -> Self {
        let started = term.now();
        Self {
            backend,
            term,
            counts,
            shadow,
            attempts: Vec::new(),
            suspensions: Vec::new(),
            skipped_classes: Vec::new(),
            started,
            ending: Ending::NotStarted,
            low_water_warned: false,
            was_calibrating: false,
        }
    }

    /// Run the whole protocol; always returns, with `ending` and `attempts`
    /// holding whatever was recorded (partial results are valid results).
    pub fn run(&mut self) {
        self.term
            .say("doorman presentation-attack screening (screening, not certification)");
        self.term
            .say("  Nothing is saved except decisions: no frames, no screenshots.");
        let status = self.backend.calibration();
        self.was_calibrating = status.as_ref().is_some_and(|c| c.active);
        match &status {
            Some(c) if c.active => self.term.say(&format!(
                "  Calibration mode is active ({} left): no pauses.",
                fmt_secs(c.remaining_secs)
            )),
            Some(_) => {
                self.term.say(NO_CALIBRATION_NOTICE);
                self.term.say(
                    "  Without it every rejection counts toward the failure budget and the \
                     session pauses while face auth is suspended.",
                );
                if self
                    .term
                    .ask("Press Enter to continue in paced mode (Ctrl-C to stop) ... ")
                    .is_none()
                {
                    self.ending = Ending::Interrupted;
                    return;
                }
            }
            None => self
                .term
                .say("  (cannot read the calibration state: is the daemon running?)"),
        }
        self.ending = Ending::Completed;
        for class in CLASSES.iter() {
            let n = self.counts.for_group(class.group);
            if n == 0 {
                continue;
            }
            if !self.run_class(class, n) {
                return;
            }
        }
    }

    /// `false` when the session must stop.
    fn run_class(&mut self, class: &AttackClass, n: usize) -> bool {
        self.term.say("");
        self.term
            .say(&format!("== {} ({n} attempts) ==", class.title));
        self.term.say(&format!("   {}", class.instruction));
        match self
            .term
            .ask("Enter to start, s + Enter to skip this class ... ")
        {
            None => {
                self.ending = Ending::Interrupted;
                return false;
            }
            Some(a) if matches!(a.trim().to_ascii_lowercase().as_str(), "s" | "skip") => {
                self.skipped_classes.push(class.name.to_string());
                self.term.say("   skipped");
                return true;
            }
            Some(_) => {}
        }
        for index in 1..=n {
            match self.one_attempt(class, index, n) {
                Step::Recorded => {}
                Step::Stop(ending) => {
                    self.ending = ending;
                    return false;
                }
            }
        }
        true
    }

    fn check_calibration(&mut self) -> bool {
        let Some(c) = self.backend.calibration() else {
            return self.was_calibrating;
        };
        if self.was_calibrating && !c.active {
            self.term.say(
                "  Calibration mode ended: face auth is paced by the failure budget again. \
                 Re-arm with: sudo doorman calibrate",
            );
        } else if c.active
            && !self.low_water_warned
            && Duration::from_secs(c.remaining_secs) <= CALIBRATION_LOW_WATER
        {
            self.low_water_warned = true;
            self.term.say(&format!(
                "  Calibration mode ends in {}: re-arm with sudo doorman calibrate",
                fmt_secs(c.remaining_secs)
            ));
        }
        self.was_calibrating = c.active;
        c.active
    }

    fn one_attempt(&mut self, class: &AttackClass, index: usize, total: usize) -> Step {
        let mut consecutive_errors = 0;
        loop {
            if self.term.interrupted() {
                return Step::Stop(Ending::Interrupted);
            }
            let calibrating = self.check_calibration();
            let prompt = format!(
                "  [{} {index}/{total}] Prepare, then press Enter ... ",
                class.name
            );
            if self.term.ask(&prompt).is_none() {
                return Step::Stop(Ending::Interrupted);
            }
            self.term
                .say(&format!("  starting in {} s ...", COUNTDOWN.as_secs()));
            if !self.term.sleep(COUNTDOWN) {
                return Step::Stop(Ending::Interrupted);
            }
            let started = self.term.now();
            let report = self.backend.attempt();
            let ended = self.term.now();
            if report.decision == "suspended" {
                // The refused run did not look at a face: wait, then redo it.
                if self.wait_out_suspension(&report).is_err() {
                    return Step::Stop(Ending::Interrupted);
                }
                continue;
            }
            let shadow = self.collect_shadow(started, ended);
            let attempt = Attempt {
                class: class.name.to_string(),
                kind: class.kind,
                index,
                decision: report.decision.to_string(),
                reason_code: report.reason_code.clone(),
                similarity_bucket: report.similarity_bucket,
                liveness_ok: report.liveness_ok,
                duration_ms: report.duration_ms,
                timestamp: started.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                calibration: calibrating,
                shadow,
            };
            let label = match attempt.decision.as_str() {
                "recognized" => "ACCEPTED".to_string(),
                "not_recognized" => "rejected".to_string(),
                other => other.to_uppercase(),
            };
            self.term
                .say(&format!("  -> {label} ({})", attempt.reason_code));
            let is_error = attempt.is_error();
            self.attempts.push(attempt);
            if is_error {
                consecutive_errors += 1;
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                    self.term.say(&format!(
                        "  {MAX_CONSECUTIVE_ERRORS} errors in a row: is the daemon running and \
                         the user enrolled? Stopping."
                    ));
                    return Step::Stop(Ending::TooManyErrors);
                }
                self.term.say(
                    "  (error: recorded as infrastructure, excluded from the rates; retrying)",
                );
                continue;
            }
            if !self.term.sleep(PAUSE_BETWEEN_ATTEMPTS) {
                return Step::Stop(Ending::Interrupted);
            }
            return Step::Recorded;
        }
    }

    fn collect_shadow(
        &mut self,
        started: DateTime<Utc>,
        ended: DateTime<Utc>,
    ) -> Option<ShadowSummary> {
        self.shadow.as_ref()?;
        let (s, e) = (started.timestamp_millis(), ended.timestamp_millis());
        let mut records = Vec::new();
        let mut waited = Duration::ZERO;
        loop {
            if let Some(log) = self.shadow.as_mut() {
                records.extend(log.new_records());
            }
            if let Some(found) = match_shadow(&records, s, e) {
                return Some(found);
            }
            if waited >= SHADOW_WAIT || !self.term.sleep(SHADOW_POLL) {
                return None;
            }
            waited += SHADOW_POLL;
        }
    }

    /// Wait until face auth is allowed again, saying why. `Err` = interrupted.
    fn wait_out_suspension(&mut self, report: &TestReport) -> Result<(), ()> {
        let now = self.term.now();
        let remaining = report
            .suspended_until
            .as_deref()
            .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
            .and_then(|t| (t.with_timezone(&Utc) - now).to_std().ok())
            .unwrap_or(FALLBACK_COOLDOWN);
        let wait = remaining.max(Duration::from_secs(1)) + COOLDOWN_MARGIN;
        self.term.say(&format!(
            "  Face auth is suspended by the failure budget; waiting {} s. This is the budget \
             doing its job. {NO_CALIBRATION_NOTICE}",
            wait.as_secs()
        ));
        self.suspensions.push(Suspension {
            at: now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            wait_s: wait.as_secs(),
            why: "failure budget".to_string(),
        });
        let mut left = wait;
        while !left.is_zero() {
            let step = left.min(COOLDOWN_STEP);
            if !self.term.sleep(step) {
                return Err(());
            }
            left -= step;
            if !left.is_zero() && wait > Duration::from_secs(30) {
                self.term.say(&format!("  ... {} s left", left.as_secs()));
            }
        }
        Ok(())
    }
}

enum Step {
    Recorded,
    Stop(Ending),
}

fn fmt_secs(s: u64) -> String {
    format!("{}m{:02}s", s / 60, s % 60)
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ClassRow {
    pub class: String,
    pub kind: Kind,
    pub n_presented: usize,
    pub n_responded: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub non_response: usize,
    pub infra_excluded: usize,
    /// `APCER` (attack accepted) or `BPCER` (genuine rejected).
    pub rate_name: &'static str,
    pub rate: Option<f64>,
    pub upper_95: Option<f64>,
    pub reasons: BTreeMap<String, usize>,
}

/// Per-class counts and rates. Responded = accepted + rejected; no-face and
/// errors are not part of the denominator.
pub fn summarize(attempts: &[Attempt]) -> Vec<ClassRow> {
    let mut names: Vec<&str> = Vec::new();
    for a in attempts {
        if !names.contains(&a.class.as_str()) {
            names.push(&a.class);
        }
    }
    names
        .into_iter()
        .map(|name| {
            let group: Vec<&Attempt> = attempts.iter().filter(|a| a.class == name).collect();
            let kind = group[0].kind;
            let accepted = group.iter().filter(|a| a.decision == "recognized").count();
            let non_response = group.iter().filter(|a| a.non_response()).count();
            let rejected = group
                .iter()
                .filter(|a| a.decision == "not_recognized" && !a.non_response())
                .count();
            let responded = accepted + rejected;
            let mut reasons = BTreeMap::new();
            for a in &group {
                *reasons.entry(a.reason_code.clone()).or_insert(0) += 1;
            }
            let errors = if kind == Kind::Attack {
                accepted
            } else {
                rejected
            };
            ClassRow {
                class: name.to_string(),
                kind,
                n_presented: group.len(),
                n_responded: responded,
                accepted,
                rejected,
                non_response,
                infra_excluded: group.len() - responded - non_response,
                rate_name: if kind == Kind::Attack {
                    "APCER"
                } else {
                    "BPCER"
                },
                rate: (responded > 0).then(|| errors as f64 / responded as f64),
                upper_95: (responded > 0).then(|| pad_stats::upper_95(errors, responded)),
                reasons,
            }
        })
        .collect()
}

fn pct(x: Option<f64>) -> String {
    x.map_or("n/a".to_string(), |v| format!("{:.1}%", v * 100.0))
}

/// One-paragraph honest summary.
pub fn headline(rows: &[ClassRow]) -> String {
    let mut parts = Vec::new();
    let worst = rows
        .iter()
        .filter(|r| r.kind == Kind::Attack && r.n_responded > 0)
        .max_by(|a, b| {
            let key = |r: &ClassRow| (r.upper_95.unwrap_or(0.0), r.rate.unwrap_or(0.0));
            key(a)
                .partial_cmp(&key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    if let Some(w) = worst {
        parts.push(format!(
            "worst attack class: {} accepted {}/{} (APCER {}, 95% upper bound {})",
            w.class,
            w.accepted,
            w.n_responded,
            pct(w.rate),
            pct(w.upper_95)
        ));
    }
    let base = rows
        .iter()
        .find(|r| r.class == "genuine" && r.n_responded > 0)
        .or_else(|| {
            rows.iter()
                .find(|r| r.kind == Kind::Genuine && r.n_responded > 0)
        });
    if let Some(b) = base {
        parts.push(format!(
            "genuine baseline: rejected {}/{} (BPCER {}, 95% upper bound {})",
            b.rejected,
            b.n_responded,
            pct(b.rate),
            pct(b.upper_95)
        ));
    }
    parts.push(
        "SCREENING, NOT CERTIFICATION: small samples give wide bounds, one phone and one \
         lighting setup are not the world"
            .to_string(),
    );
    parts.join("; ")
}

pub fn format_table(rows: &[ClassRow]) -> String {
    let mut out = format!(
        "{:<22}{:<8}{:>4}{:>6}{:>5}{:>5}{:>4}{:>6}{:>8}{:>11}\n",
        "class", "kind", "n", "resp", "acc", "rej", "nr", "infra", "rate", "95% upper"
    );
    out.push_str(&"-".repeat(79));
    out.push('\n');
    for r in rows {
        let kind = match r.kind {
            Kind::Genuine => "genuine",
            Kind::Attack => "attack",
        };
        let _ = writeln!(
            out,
            "{:<22}{:<8}{:>4}{:>6}{:>5}{:>5}{:>4}{:>6}{:>8}{:>11}",
            r.class,
            kind,
            r.n_presented,
            r.n_responded,
            r.accepted,
            r.rejected,
            r.non_response,
            r.infra_excluded,
            pct(r.rate),
            pct(r.upper_95)
        );
    }
    out.push_str(
        "rate = APCER (attack accepted) or BPCER (genuine rejected); nr = no face / no frames; \
         infra = errors, excluded\n",
    );
    out
}

// ---------------------------------------------------------------------------
// Output files
// ---------------------------------------------------------------------------

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn opt<T: ToString>(v: Option<T>) -> String {
    v.map(|v| v.to_string()).unwrap_or_default()
}

pub fn attempts_csv(attempts: &[Attempt]) -> String {
    let mut out = String::from(
        "class,kind,index,decision,reason_code,similarity_bucket,liveness_ok,duration_ms,\
         timestamp,calibration,shadow_depth_margin_min,shadow_minifasnet_logit_diff,\
         shadow_similarity_median\n",
    );
    for a in attempts {
        let sh = a.shadow.as_ref();
        let kind = if a.kind == Kind::Attack {
            "attack"
        } else {
            "genuine"
        };
        let _ = writeln!(
            out,
            "{},{kind},{},{},{},{},{},{},{},{},{},{},{}",
            csv_field(&a.class),
            a.index,
            csv_field(&a.decision),
            csv_field(&a.reason_code),
            opt(a.similarity_bucket),
            opt(a.liveness_ok),
            a.duration_ms,
            a.timestamp,
            a.calibration,
            opt(sh.and_then(|s| s.depth_margin_min)),
            opt(sh.and_then(|s| s.minifasnet_logit_diff)),
            opt(sh.and_then(|s| s.similarity_median)),
        );
    }
    out
}

/// Input of `doorman-bench pad-report --threshold 0.5`: score 1.0 = accepted,
/// 0.0 = rejected; no-face / no-frames rows are non-responses; errors are left
/// out.
pub fn bench_csv(attempts: &[Attempt], session: &str) -> String {
    let mut out = String::from("label,species,score,response,session,subject\n");
    for a in attempts {
        let (response, score) = if a.decision == "recognized" {
            ("ok", 1.0)
        } else if a.non_response() {
            ("no_face", 0.0)
        } else if a.decision == "not_recognized" {
            ("ok", 0.0)
        } else {
            continue;
        };
        let (label, species) = if a.kind == Kind::Attack {
            ("attack", a.class.as_str())
        } else {
            ("bona_fide", "none")
        };
        let _ = writeln!(
            out,
            "{label},{species},{score:.1},{response},{session},owner"
        );
    }
    out
}

#[derive(Serialize)]
struct ResultsJson<'a> {
    protocol_version: u32,
    meta: Meta<'a>,
    summary: &'a [ClassRow],
    headline: &'a str,
    attempts: &'a [Attempt],
}

#[derive(Serialize)]
struct Meta<'a> {
    started: String,
    ending: Ending,
    completed: bool,
    confidence: f64,
    bound: &'static str,
    suspensions: &'a [Suspension],
    skipped_classes: &'a [String],
    shadow_log_used: bool,
    user_recorded: bool,
}

pub fn results_json(s: &Session<'_>, rows: &[ClassRow], line: &str) -> String {
    let doc = ResultsJson {
        protocol_version: PROTOCOL_VERSION,
        meta: Meta {
            started: s.started.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            ending: s.ending,
            completed: s.ending == Ending::Completed,
            confidence: pad_stats::CONFIDENCE,
            bound: "one-sided Clopper-Pearson",
            suspensions: &s.suspensions,
            skipped_classes: &s.skipped_classes,
            shadow_log_used: s.shadow.is_some(),
            user_recorded: false,
        },
        summary: rows,
        headline: line,
        attempts: &s.attempts,
    };
    let mut text = serde_json::to_string_pretty(&doc).expect("results serialize");
    text.push('\n');
    text
}

/// Write `text` to `path` with mode 0600 (also when the file existed).
pub fn write_private(path: &Path, text: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    file.write_all(text.as_bytes())?;
    file.sync_all()
}

/// The three output paths for a prefix: `.json`, `.csv`, `-bench.csv`.
pub fn output_paths(prefix: &Path) -> [PathBuf; 3] {
    let with = |suffix: &str| {
        let mut s = prefix.as_os_str().to_os_string();
        s.push(suffix);
        PathBuf::from(s)
    };
    [with(".json"), with(".csv"), with("-bench.csv")]
}

pub fn default_prefix(home: &Path, today: chrono::NaiveDate) -> PathBuf {
    home.join(format!("doorman-pad-{today}"))
}

/// Write all three files (mode 0600). Returns their paths.
pub fn write_results(
    prefix: &Path,
    session: &Session<'_>,
    rows: &[ClassRow],
    line: &str,
) -> std::io::Result<[PathBuf; 3]> {
    let paths = output_paths(prefix);
    let session_id = session.started.format("%Y%m%dT%H%M%S").to_string();
    write_private(&paths[0], &results_json(session, rows, line))?;
    write_private(&paths[1], &attempts_csv(&session.attempts))?;
    write_private(&paths[2], &bench_csv(&session.attempts, &session_id))?;
    Ok(paths)
}

/// What to print about the shadow records after the run.
pub fn shadow_hint(session: &Session<'_>, log_path: &str) -> String {
    if session.shadow.is_some() {
        "Shadow records were read and attached to each attempt.".to_string()
    } else {
        format!(
            "Shadow records are not readable by you. Read them with: sudo cat {log_path} \
             (match by timestamp; each record has source and calibration)."
        )
    }
}

/// Whether this process runs as root (file modes do not bind it).
pub fn running_as_root() -> bool {
    doorman_shared::config_trust::effective_uid() == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct FakeTerm {
        now: DateTime<Utc>,
        answers: VecDeque<Option<String>>,
        said: Vec<String>,
        interrupt_after_sleeps: Option<usize>,
        sleeps: usize,
        slept: Duration,
    }

    impl FakeTerm {
        fn new() -> Self {
            Self {
                now: DateTime::from_timestamp(1_800_000_000, 0).unwrap(),
                answers: VecDeque::new(),
                said: Vec::new(),
                interrupt_after_sleeps: None,
                sleeps: 0,
                slept: Duration::ZERO,
            }
        }
        fn said(&self, needle: &str) -> bool {
            self.said.iter().any(|l| l.contains(needle))
        }
    }

    impl Terminal for FakeTerm {
        fn say(&mut self, line: &str) {
            self.said.push(line.to_string());
        }
        fn ask(&mut self, _prompt: &str) -> Option<String> {
            // Default: the user just presses Enter.
            self.answers.pop_front().unwrap_or(Some(String::new()))
        }
        fn sleep(&mut self, d: Duration) -> bool {
            self.sleeps += 1;
            self.slept += d;
            self.now += chrono::Duration::from_std(d).unwrap();
            self.interrupt_after_sleeps.is_none_or(|n| self.sleeps <= n)
        }
        fn interrupted(&self) -> bool {
            self.interrupt_after_sleeps.is_some_and(|n| self.sleeps > n)
        }
        fn now(&self) -> DateTime<Utc> {
            self.now
        }
    }

    fn report(decision: &'static str, code: &str) -> TestReport {
        TestReport {
            decision,
            reason_code: code.to_string(),
            similarity_bucket: Some(0.5),
            liveness_ok: Some(true),
            frames_judged: None,
            duration_ms: 900,
            suspended_until: None,
            calibration: false,
            exit_code: 0,
        }
    }

    struct FakeBackend {
        script: VecDeque<TestReport>,
        calibration: Vec<Option<CalibrationInfo>>,
        attempts: usize,
        status_calls: usize,
    }

    impl FakeBackend {
        fn new(script: Vec<TestReport>, active: bool) -> Self {
            Self {
                script: script.into(),
                calibration: vec![Some(info(active, 2400))],
                attempts: 0,
                status_calls: 0,
            }
        }
    }

    fn info(active: bool, remaining: u64) -> CalibrationInfo {
        CalibrationInfo {
            active,
            remaining_secs: if active { remaining } else { 0 },
            ttl_secs: if active { 2700 } else { 0 },
            judged_failures_counted: 0,
        }
    }

    impl Backend for FakeBackend {
        fn attempt(&mut self) -> TestReport {
            self.attempts += 1;
            self.script
                .pop_front()
                .unwrap_or_else(|| report("not_recognized", "not_matched"))
        }
        fn calibration(&mut self) -> Option<CalibrationInfo> {
            let i = self.status_calls.min(self.calibration.len() - 1);
            self.status_calls += 1;
            self.calibration[i].clone()
        }
    }

    const TINY: Counts = Counts {
        genuine: 2,
        attack: 2,
        condition: 1,
    };

    fn run(
        backend: &mut FakeBackend,
        term: &mut FakeTerm,
        counts: Counts,
    ) -> (Vec<Attempt>, Ending, Vec<String>) {
        let mut s = Session::new(backend, term, counts, None);
        s.run();
        (s.attempts, s.ending, s.skipped_classes)
    }

    #[test]
    fn defaults_are_ten_ten_and_quick_is_three_two() {
        assert_eq!(
            (Counts::FULL.genuine, Counts::FULL.attack),
            (10, 10),
            "10 genuine + 10 per attack class"
        );
        assert_eq!((Counts::QUICK.genuine, Counts::QUICK.attack), (3, 2));
        let plan = |c: Counts| -> usize { CLASSES.iter().map(|k| c.for_group(k.group)).sum() };
        assert_eq!(plan(Counts::FULL), 10 + 4 * 10 + 8 * 5);
        assert_eq!(plan(Counts::QUICK), 3 + 4 * 2 + 8 * 2);
    }

    #[test]
    fn every_required_class_is_present() {
        let names: Vec<_> = CLASSES.iter().map(|c| c.name).collect();
        for want in [
            "genuine",
            "phone_photo",
            "phone_replay",
            "print_matte",
            "mixed_track",
            "genuine_dark",
            "genuine_backlight",
            "genuine_glasses_on",
            "genuine_glasses_off",
            "genuine_yaw_left",
            "genuine_yaw_right",
            "genuine_near",
            "genuine_far",
        ] {
            assert!(names.contains(&want), "{want}");
        }
        assert_eq!(CLASSES.iter().filter(|c| c.kind == Kind::Attack).count(), 4);
    }

    #[test]
    fn without_calibration_the_exact_notice_is_printed_and_enter_is_awaited() {
        let mut backend = FakeBackend::new(vec![], false);
        let mut term = FakeTerm::new();
        // First answer (the Enter after the notice) is None: Ctrl-C there stops cleanly.
        term.answers.push_back(None);
        let (attempts, ending, _) = run(&mut backend, &mut term, TINY);
        assert!(term
            .said
            .iter()
            .any(|l| l == "Для 20-минутной сессии без пауз: sudo doorman calibrate"));
        assert_eq!(
            (attempts.len(), ending, backend.attempts),
            (0, Ending::Interrupted, 0)
        );
    }

    #[test]
    fn with_calibration_active_there_is_no_notice_and_no_extra_prompt() {
        let mut backend = FakeBackend::new(vec![], true);
        let mut term = FakeTerm::new();
        let (attempts, ending, _) = run(&mut backend, &mut term, TINY);
        assert!(!term.said(NO_CALIBRATION_NOTICE));
        assert_eq!(ending, Ending::Completed);
        assert_eq!(attempts.len(), 2 + 4 * 2 + 8);
        assert!(attempts.iter().all(|a| a.calibration));
    }

    #[test]
    fn a_suspension_is_waited_out_explained_and_not_recorded() {
        let mut suspended = report("suspended", "face_auth_suspended");
        let until = DateTime::from_timestamp(1_800_000_000 + 40, 0).unwrap();
        suspended.suspended_until = Some(until.to_rfc3339_opts(chrono::SecondsFormat::Secs, true));
        let mut backend = FakeBackend::new(vec![suspended, report("recognized", "matched")], false);
        let mut term = FakeTerm::new();
        let mut s = Session::new(
            &mut backend,
            &mut term,
            Counts {
                genuine: 1,
                attack: 0,
                condition: 0,
            },
            None,
        );
        s.run();
        assert_eq!(s.attempts.len(), 1, "the refused run is not an attempt");
        assert_eq!(s.attempts[0].decision, "recognized");
        assert_eq!(s.suspensions.len(), 1);
        assert!(term.now >= until, "waited past suspended_until");
        let said = term.said.join("\n");
        assert!(said.contains("suspended by the failure budget"), "{said}");
    }

    #[test]
    fn three_errors_in_a_row_stop_the_session_but_keep_the_results() {
        let errors = (0..3).map(|_| report("error", "daemon_error")).collect();
        let mut backend = FakeBackend::new(errors, true);
        let mut term = FakeTerm::new();
        let (attempts, ending, _) = run(&mut backend, &mut term, TINY);
        assert_eq!(ending, Ending::TooManyErrors);
        assert_eq!(attempts.len(), 3);
        let rows = summarize(&attempts);
        assert_eq!(rows[0].infra_excluded, 3);
        assert_eq!(rows[0].n_responded, 0);
    }

    #[test]
    fn ctrl_c_ends_the_run_cleanly_with_partial_results_that_can_be_written() {
        let mut backend = FakeBackend::new(
            vec![
                report("recognized", "matched"),
                report("recognized", "matched"),
            ],
            true,
        );
        let mut term = FakeTerm::new();
        // Each attempt sleeps twice (countdown, pause): interrupt during the third attempt.
        term.interrupt_after_sleeps = Some(4);
        let mut s = Session::new(&mut backend, &mut term, TINY, None);
        s.run();
        assert_eq!(s.ending, Ending::Interrupted);
        assert_eq!(s.attempts.len(), 2);
        let rows = summarize(&s.attempts);
        let line = headline(&rows);
        let dir = tempfile::tempdir().unwrap();
        let paths = write_results(&dir.path().join("out"), &s, &rows, &line).unwrap();
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&paths[0]).unwrap()).unwrap();
        assert_eq!(json["meta"]["completed"], false);
        assert_eq!(json["meta"]["ending"], "interrupted");
        assert_eq!(json["attempts"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn skipping_a_class_is_recorded() {
        let mut backend = FakeBackend::new(vec![], true);
        let mut term = FakeTerm::new();
        // genuine: Enter to start, then 2 attempts (2 Enters); then skip the 2nd class.
        for a in ["", "", "", "s"] {
            term.answers.push_back(Some(a.to_string()));
        }
        let (attempts, _, skipped) = run(&mut backend, &mut term, TINY);
        assert_eq!(skipped, vec!["phone_photo".to_string()]);
        assert!(attempts.iter().all(|a| a.class != "phone_photo"));
    }

    #[test]
    fn calibration_ending_midway_is_announced() {
        let mut backend = FakeBackend::new(vec![], true);
        backend.calibration = vec![
            Some(info(true, 2400)),
            Some(info(true, 200)),
            Some(info(false, 0)),
        ];
        let mut term = FakeTerm::new();
        run(
            &mut backend,
            &mut term,
            Counts {
                genuine: 3,
                attack: 0,
                condition: 0,
            },
        );
        assert!(term.said("ends in 3m20s"), "{:?}", term.said);
        assert!(term.said("Calibration mode ended"));
    }

    fn att(class: &str, kind: Kind, decision: &str, code: &str) -> Attempt {
        Attempt {
            class: class.to_string(),
            kind,
            index: 1,
            decision: decision.to_string(),
            reason_code: code.to_string(),
            similarity_bucket: None,
            liveness_ok: None,
            duration_ms: 1,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            calibration: true,
            shadow: None,
        }
    }

    #[test]
    fn summary_counts_rates_and_the_exact_bound() {
        let mut v = Vec::new();
        for _ in 0..10 {
            v.push(att(
                "phone_photo",
                Kind::Attack,
                "not_recognized",
                "liveness_rejected",
            ));
        }
        v.push(att(
            "phone_photo",
            Kind::Attack,
            "not_recognized",
            "no_clear_face",
        ));
        v.push(att("phone_photo", Kind::Attack, "error", "daemon_error"));
        for _ in 0..9 {
            v.push(att("genuine", Kind::Genuine, "recognized", "matched"));
        }
        v.push(att(
            "genuine",
            Kind::Genuine,
            "not_recognized",
            "not_matched",
        ));
        let rows = summarize(&v);
        let photo = rows.iter().find(|r| r.class == "phone_photo").unwrap();
        assert_eq!(
            (
                photo.n_presented,
                photo.n_responded,
                photo.accepted,
                photo.non_response,
                photo.infra_excluded
            ),
            (12, 10, 0, 1, 1)
        );
        assert!(
            (photo.upper_95.unwrap() - 0.2588655509).abs() < 1e-8,
            "0/10 -> 25.89 %"
        );
        let genuine = rows.iter().find(|r| r.class == "genuine").unwrap();
        assert_eq!((genuine.rejected, genuine.rate_name), (1, "BPCER"));
        let line = headline(&rows);
        assert!(line.contains("SCREENING, NOT CERTIFICATION"));
        assert!(line.contains("phone_photo accepted 0/10"));
    }

    #[test]
    fn bench_csv_matches_the_pad_report_format() {
        let v = vec![
            att("phone_photo", Kind::Attack, "recognized", "matched"),
            att(
                "print_matte",
                Kind::Attack,
                "not_recognized",
                "liveness_rejected",
            ),
            att(
                "print_matte",
                Kind::Attack,
                "not_recognized",
                "no_clear_face",
            ),
            att("genuine", Kind::Genuine, "recognized", "matched"),
            att("genuine", Kind::Genuine, "error", "daemon_error"),
        ];
        assert_eq!(
            bench_csv(&v, "S1"),
            "label,species,score,response,session,subject\n\
             attack,phone_photo,1.0,ok,S1,owner\n\
             attack,print_matte,0.0,ok,S1,owner\n\
             attack,print_matte,0.0,no_face,S1,owner\n\
             bona_fide,none,1.0,ok,S1,owner\n"
        );
    }

    #[test]
    fn output_files_are_private_even_when_they_existed() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let mut backend = FakeBackend::new(vec![], true);
        let mut term = FakeTerm::new();
        let mut s = Session::new(
            &mut backend,
            &mut term,
            Counts {
                genuine: 1,
                attack: 0,
                condition: 0,
            },
            None,
        );
        s.run();
        let prefix = dir.path().join("doorman-pad-2026-01-01");
        // A pre-existing world-readable file must end up 0600.
        std::fs::write(&output_paths(&prefix)[1], "old").unwrap();
        std::fs::set_permissions(
            &output_paths(&prefix)[1],
            std::fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        let rows = summarize(&s.attempts);
        let paths = write_results(&prefix, &s, &rows, &headline(&rows)).unwrap();
        for p in &paths {
            let mode = std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "{}", p.display());
        }
        assert!(paths[2].to_string_lossy().ends_with("-bench.csv"));
        let json = std::fs::read_to_string(&paths[0]).unwrap();
        for forbidden in ["frame", "embedding", "username"] {
            assert!(!json.contains(forbidden), "{forbidden}");
        }
    }

    #[test]
    fn shadow_log_availability_is_readability_not_stat() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shadow.jsonl");
        assert!(ShadowLog::open(&path).is_none(), "missing");
        std::fs::write(&path, "").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
        // stat succeeds, reading does not (root ignores modes, so only assert for non-root).
        assert!(std::fs::metadata(&path).is_ok());
        if crate::pad_protocol::running_as_root() {
            return;
        }
        assert!(
            ShadowLog::open(&path).is_none(),
            "stat-able but unreadable is NOT available"
        );
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        assert!(ShadowLog::open(&path).is_some());
    }

    #[test]
    fn the_shadow_reader_sees_only_new_records_and_survives_rotation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shadow.jsonl");
        std::fs::write(&path, "{\"old\":1}\n").unwrap();
        let mut log = ShadowLog::open(&path).unwrap();
        assert!(log.new_records().is_empty(), "the old record is skipped");
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        std::io::Write::write_all(&mut f, b"{\"a\":1}\nnot json\n{\"b\":2}\n").unwrap();
        assert_eq!(log.new_records().len(), 2);
        std::fs::write(&path, "{\"c\":3}\n").unwrap(); // shorter: rotated
        assert_eq!(log.new_records().len(), 1);
    }

    #[test]
    fn shadow_records_are_matched_by_time_and_source() {
        let rec = |ts: i64, source: &str| {
            serde_json::json!({"timestamp_unix_ms": ts, "source": source, "attempt_id": format!("t-{ts}"),
                "frames": {"judged": 5}, "minifasnet": {"state": "scored", "logit_diff": 2.5}})
        };
        let records = vec![
            rec(10_000, "lock_screen"),
            rec(10_500, "test"),
            rec(99_000, "test"),
        ];
        let m = match_shadow(&records, 10_000, 11_000).unwrap();
        assert_eq!(m.attempt_id.as_deref(), Some("t-10500"));
        assert_eq!(m.minifasnet_logit_diff, Some(2.5));
        assert!(match_shadow(&records, 50_000, 51_000).is_none());
    }

    #[test]
    fn the_hint_says_how_to_read_unreadable_shadow_records() {
        let mut backend = FakeBackend::new(vec![], true);
        let mut term = FakeTerm::new();
        let s = Session::new(&mut backend, &mut term, TINY, None);
        assert!(shadow_hint(&s, "/run/doorman/shadow-pad.jsonl")
            .contains("sudo cat /run/doorman/shadow-pad.jsonl"));
    }
}
