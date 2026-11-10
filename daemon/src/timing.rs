//! Per-attempt stage timing telemetry (WS6).
//!
//! One [`AttemptTiming`] is created at the start of an authentication attempt,
//! filled in by hooks along the pipeline, and emitted when it is dropped:
//!
//! * always as a `debug!` line (via `tracing`), and
//! * as one JSON line appended to a bounded rotating file when
//!   `[telemetry] timing_log = true`.
//!
//! Privacy: a record carries stage durations, frame counters, the outcome, the
//! final cosine similarity and the liveness verdict. The similarity is kept
//! on purpose: it is the calibration signal for the threshold. A record never
//! carries a username, uid, frame, embedding or per-frame score. The attempt id
//! is a monotonic counter + wall-clock microseconds, unrelated to any identity.
//!
//! All durations come from `std::time::Instant` (monotonic).

use doorman_shared::config::TelemetryConfig;
use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// File mode of the timing log: owner read/write, group read.
const TIMING_LOG_MODE: u32 = 0o640;

static ATTEMPT_SEQ: AtomicU64 = AtomicU64::new(0);

/// How an authentication attempt ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingOutcome {
    /// Never set: the attempt exited through a path without a hook.
    Unknown,
    Matched,
    NotMatched,
    LivenessRejected,
    LivenessError,
    NoFrames,
    NoClearFace,
    NotEnrolled,
    CameraUnavailable,
    IdentityFloor,
    AmbiguousFaces,
    InsufficientFrames,
}

/// Sum + count of repeated stage invocations.
#[derive(Debug, Default, Clone, Copy)]
pub struct StageAccum {
    pub total: Duration,
    pub count: u32,
}

impl StageAccum {
    pub fn record(&mut self, d: Duration) {
        self.total += d;
        self.count += 1;
    }
}

/// Serialized form: durations in fractional milliseconds, `null` when a stage
/// never ran or cannot be measured.
#[derive(Debug, Serialize)]
pub struct TimingRecord {
    pub attempt_id: String,
    pub timestamp_unix_ms: u64,
    pub outcome: TimingOutcome,
    pub camera_open_ms: Option<f64>,
    pub first_frame_ms: Option<f64>,
    pub first_valid_frame_ms: Option<f64>,
    pub detect_sum_ms: Option<f64>,
    pub detect_count: u32,
    /// Always `null` today: alignment runs inside the backend's embedding call
    /// and is included in `embed_*`.
    pub align_ms: Option<f64>,
    pub embed_sum_ms: Option<f64>,
    pub embed_count: u32,
    pub liveness_ms: Option<f64>,
    pub liveness_passed: Option<bool>,
    /// Depth relief of the liveness evaluation (calibration signal, see
    /// docs/decision-and-liveness.md); `null` when it produced none.
    pub liveness_relief: Option<f32>,
    /// Frames the liveness check was evaluated on (1 for the single-frame gate).
    pub liveness_frames: u32,
    pub decision_ms: Option<f64>,
    pub total_ms: f64,
    pub frames_seen: usize,
    pub frames_used: usize,
    /// Final aggregate cosine similarity (calibration signal).
    pub similarity: Option<f32>,
    /// `WaitForMatch` only: request until the lock gate was first seen armed.
    pub hint_wait_ms: Option<f64>,
    /// `WaitForMatch` only: lock gate armed until the camera delivered (signed:
    /// negative when pre-arm had the camera ready before the lock was confirmed).
    pub lock_to_camera_ready_ms: Option<f64>,
    /// Camera ready until the last frame needed was scored.
    pub frames_collect_ms: Option<f64>,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Timing state of one authentication attempt. Emits on drop.
pub struct AttemptTiming {
    start: Instant,
    cfg: TelemetryConfig,
    pub attempt_id: String,
    pub outcome: TimingOutcome,
    pub camera_open: Option<Duration>,
    pub first_frame: Option<Duration>,
    pub first_valid_frame: Option<Duration>,
    pub detect: StageAccum,
    pub embed: StageAccum,
    pub liveness: Option<Duration>,
    pub liveness_passed: Option<bool>,
    pub liveness_relief: Option<f32>,
    pub liveness_frames: u32,
    pub decision: Option<Duration>,
    pub frames_seen: usize,
    pub frames_used: usize,
    pub similarity: Option<f32>,
    /// Set by `WaitForMatch` attempts (see [`AttemptTiming::set_wait`]).
    wait_request_start: Option<Instant>,
    wait_armed_at: Option<Arc<OnceLock<Instant>>>,
    pub frames_collect: Option<Duration>,
}

impl AttemptTiming {
    pub fn start(cfg: &TelemetryConfig) -> Self {
        let seq = ATTEMPT_SEQ.fetch_add(1, Ordering::Relaxed);
        let micros = chrono::Utc::now().timestamp_micros();
        Self {
            start: Instant::now(),
            cfg: cfg.clone(),
            attempt_id: format!("t-{micros}-{seq}"),
            outcome: TimingOutcome::Unknown,
            camera_open: None,
            first_frame: None,
            first_valid_frame: None,
            detect: StageAccum::default(),
            embed: StageAccum::default(),
            liveness: None,
            liveness_passed: None,
            liveness_relief: None,
            liveness_frames: 0,
            decision: None,
            frames_seen: 0,
            frames_used: 0,
            similarity: None,
            wait_request_start: None,
            wait_armed_at: None,
            frames_collect: None,
        }
    }

    /// Time elapsed since the attempt started.
    pub fn since_start(&self) -> Duration {
        self.start.elapsed()
    }

    /// Mark the first frame received (no-op after the first call).
    pub fn mark_first_frame(&mut self) {
        if self.first_frame.is_none() {
            self.first_frame = Some(self.start.elapsed());
        }
    }

    /// Mark the first frame that passed the quality gate (no-op after the first call).
    pub fn mark_first_valid_frame(&mut self) {
        if self.first_valid_frame.is_none() {
            self.first_valid_frame = Some(self.start.elapsed());
        }
    }

    /// Attach the `WaitForMatch` clock: when the request arrived and the cell
    /// the lock gate fills in when it is first seen armed.
    pub fn set_wait(&mut self, request_start: Instant, armed_at: Arc<OnceLock<Instant>>) {
        self.wait_request_start = Some(request_start);
        self.wait_armed_at = Some(armed_at);
    }

    fn armed_at(&self) -> Option<Instant> {
        self.wait_armed_at.as_ref()?.get().copied()
    }

    fn hint_wait_ms(&self) -> Option<f64> {
        let armed = self.armed_at()?;
        Some(ms(armed.saturating_duration_since(self.wait_request_start?)))
    }

    /// Armed-to-camera-ready in ms, negative when the camera was ready first.
    fn lock_to_camera_ready_ms(&self) -> Option<f64> {
        let armed = self.armed_at()?;
        let ready = self.start + self.camera_open?;
        Some(match ready.checked_duration_since(armed) {
            Some(after) => ms(after),
            None => -ms(armed.saturating_duration_since(ready)),
        })
    }

    pub fn record(&self) -> TimingRecord {
        let accum_ms = |a: &StageAccum| (a.count > 0).then(|| ms(a.total));
        TimingRecord {
            attempt_id: self.attempt_id.clone(),
            timestamp_unix_ms: chrono::Utc::now().timestamp_millis().max(0) as u64,
            outcome: self.outcome,
            camera_open_ms: self.camera_open.map(ms),
            first_frame_ms: self.first_frame.map(ms),
            first_valid_frame_ms: self.first_valid_frame.map(ms),
            detect_sum_ms: accum_ms(&self.detect),
            detect_count: self.detect.count,
            align_ms: None,
            embed_sum_ms: accum_ms(&self.embed),
            embed_count: self.embed.count,
            liveness_ms: self.liveness.map(ms),
            liveness_passed: self.liveness_passed,
            liveness_relief: self.liveness_relief,
            liveness_frames: self.liveness_frames,
            decision_ms: self.decision.map(ms),
            total_ms: ms(self.start.elapsed()),
            frames_seen: self.frames_seen,
            frames_used: self.frames_used,
            similarity: self.similarity,
            hint_wait_ms: self.hint_wait_ms(),
            lock_to_camera_ready_ms: self.lock_to_camera_ready_ms(),
            frames_collect_ms: self.frames_collect.map(ms),
        }
    }
}

impl Drop for AttemptTiming {
    fn drop(&mut self) {
        let rec = self.record();
        match serde_json::to_string(&rec) {
            Ok(line) => {
                debug!(target: "doorman::timing", "{}", line);
                if self.cfg.timing_log {
                    if let Err(e) = append_rotating(
                        Path::new(&self.cfg.timing_log_path),
                        self.cfg.timing_log_max_bytes,
                        &line,
                    ) {
                        warn!("timing log write failed: {}", e);
                    }
                }
            }
            Err(e) => warn!("timing record serialization failed: {}", e),
        }
    }
}

fn rotated_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".1");
    PathBuf::from(s)
}

/// Append `line` (plus newline) to `path`. If that would grow the file beyond
/// `max_bytes`, the file is first renamed to `<path>.1` (replacing the previous
/// generation), so at most about `2 * max_bytes` is ever on disk.
pub fn append_rotating(path: &Path, max_bytes: u64, line: &str) -> std::io::Result<()> {
    let incoming = line.len() as u64 + 1;
    if let Ok(meta) = fs::metadata(path) {
        if meta.len() > 0 && meta.len() + incoming > max_bytes {
            fs::rename(path, rotated_path(path))?;
        }
    }
    let mut opts = OpenOptions::new();
    opts.create(true).append(true);
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(TIMING_LOG_MODE);
    }
    let mut f = opts.open(path)?;
    f.write_all(line.as_bytes())?;
    f.write_all(b"\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_keeps_one_generation() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.jsonl");
        // Each line is 9 bytes with newline; cap of 20 fits two lines.
        for i in 0..5 {
            append_rotating(&p, 20, &format!("line-{i:03}")).unwrap();
        }
        let cur = fs::read_to_string(&p).unwrap();
        let old = fs::read_to_string(rotated_path(&p)).unwrap();
        assert_eq!(cur, "line-004\n");
        assert_eq!(old, "line-002\nline-003\n");
    }

    #[test]
    fn record_has_no_identity_fields_and_null_for_missing_stages() {
        let cfg = TelemetryConfig::default();
        let mut t = AttemptTiming::start(&cfg);
        t.detect.record(Duration::from_millis(10));
        t.detect.record(Duration::from_millis(20));
        t.outcome = TimingOutcome::Matched;
        let v = serde_json::to_value(t.record()).unwrap();
        assert_eq!(v["detect_sum_ms"], 30.0);
        assert_eq!(v["detect_count"], 2);
        assert!(v["embed_sum_ms"].is_null());
        assert!(v["align_ms"].is_null());
        assert!(v["liveness_relief"].is_null());
        assert_eq!(v["outcome"], "matched");
        t.liveness_relief = Some(0.25);
        assert_eq!(
            serde_json::to_value(t.record()).unwrap()["liveness_relief"],
            0.25
        );
        let s = v.to_string();
        assert!(!s.contains("user") && !s.contains("uid"));
    }

    #[test]
    fn lock_relative_stages_are_present_and_signed() {
        let cfg = TelemetryConfig::default();
        let mut t = AttemptTiming::start(&cfg);
        let v = serde_json::to_value(t.record()).unwrap();
        for key in [
            "hint_wait_ms",
            "lock_to_camera_ready_ms",
            "frames_collect_ms",
        ] {
            assert!(
                v.get(key).is_some_and(|x| x.is_null()),
                "{key} absent or set"
            );
        }
        // The gate opened 40 ms after the request; the camera became ready 100 ms
        // after the attempt started (here: the request).
        let request = t.start;
        let armed = Arc::new(OnceLock::new());
        armed.set(request + Duration::from_millis(40)).unwrap();
        t.set_wait(request, armed);
        t.camera_open = Some(Duration::from_millis(100));
        t.frames_collect = Some(Duration::from_millis(250));
        let v = serde_json::to_value(t.record()).unwrap();
        assert_eq!(v["hint_wait_ms"], 40.0);
        assert_eq!(v["lock_to_camera_ready_ms"], 60.0);
        assert_eq!(v["frames_collect_ms"], 250.0);
        // Camera ready before the gate opened: negative.
        t.camera_open = Some(Duration::from_millis(10));
        let v = serde_json::to_value(t.record()).unwrap();
        assert_eq!(v["lock_to_camera_ready_ms"], -30.0);
        assert!(!v.to_string().contains("user"));
    }
}
