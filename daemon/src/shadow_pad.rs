//! Shadow PAD logging: calibration data for the presentation-attack decision,
//! collected without any influence on it.
//!
//! `[liveness] shadow_pad_logging = true` makes each authentication attempt
//! append ONE JSON line to `[telemetry] shadow_pad_log_path` (mode 0640,
//! rotated like the timing log). The line holds per-attempt aggregates only:
//! the distribution of per-frame similarities and of depth-relief values, frame
//! counters with the reasons frames were dropped, quality metrics of the
//! deciding frame, the MiniFASNet score of that frame, the decision and its
//! reason code. It never holds a frame, an embedding or a name.
//!
//! # Why it cannot change a decision
//!
//! * The collector is fed by plain value copies at hooks in the decision loop
//!   (`similarity`, `depth_relief`, `frame_event`, `quality_frame`). None of
//!   them returns anything the decision reads.
//! * Every hook runs under `catch_unwind`; the first panic disables the
//!   collector for the rest of the attempt and is logged at WARN.
//! * The only expensive part, the MiniFASNet pass and the quality metrics, runs
//!   AFTER the decision exists, in a detached task ([`ShadowPad::finish`]); an
//!   error, panic, timeout or missing model there only changes the record.
//! * The legacy `check_liveness` path (which answers "pass" for a missing
//!   model) is not used: [`crate::ml::MLPipeline::shadow_minifasnet_logit_diff`]
//!   reports a missing model as `None`.
//!
//! Latency: collecting is a few `Vec` pushes per frame. The deciding frame is
//! kept as an `Arc` (no pixel copy). The extra MiniFASNet inference (128x128
//! input, a model of about 1.8 MB, so on the order of 5-20 ms on CPU, an
//! estimate that the owner's timing log can confirm) and the quality metrics
//! run after the response is built, off the async workers.

use crate::ml::{Face, MLPipeline};
use doorman_shared::Config;
use doormand::decision::AuthDecision;
use image::DynamicImage;
use serde::Serialize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Record format version; bump on an incompatible change.
pub const SCHEMA_VERSION: u32 = 2;
/// Upper bound for the shadow MiniFASNet pass; past it the record says `timeout`.
const MINIFASNET_TIMEOUT: Duration = Duration::from_secs(2);
const P10: f32 = 0.10;

/// Where an attempt came from; written to every record as `source`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptSource {
    /// A direct `Authenticate` that is not the sudo profile: `doorman test`,
    /// PAM login/unlock.
    Test,
    /// The strict sudo profile.
    Sudo,
    /// `WaitForMatch` (lock-screen unlock watcher, including pre-arm retries).
    LockScreen,
}

impl AttemptSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Test => "test",
            Self::Sudo => "sudo",
            Self::LockScreen => "lock_screen",
        }
    }
}

/// Why a frame did or did not reach the decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameEvent {
    /// Captured before the request (strict profile or `WaitForMatch`).
    SkippedPreRequest,
    /// Older than `camera.max_frame_age_ms`.
    SkippedStale,
    NoFace,
    DetectorError,
    /// Multi-face gate.
    Ambiguous,
    /// Face-area / sharpness / brightness gate.
    QualityRejected,
    EmbeddingError,
    /// Embedded and scored: the frame entered the decision.
    Judged,
}

/// Min / 10th percentile / median / max of a list of values.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Distribution {
    pub n: usize,
    pub min: f32,
    pub p10: f32,
    pub median: f32,
    pub max: f32,
}

/// Linear-interpolated percentile of a sorted, non-empty slice.
fn percentile(sorted: &[f32], q: f32) -> f32 {
    let pos = q * (sorted.len() - 1) as f32;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f32)
}

/// Distribution of the finite values; `None` when there are none.
pub fn distribution(values: &[f32]) -> Option<Distribution> {
    let mut sorted: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(|a, b| a.total_cmp(b));
    Some(Distribution {
        n: sorted.len(),
        min: sorted[0],
        p10: percentile(&sorted, P10),
        median: percentile(&sorted, 0.5),
        max: sorted[sorted.len() - 1],
    })
}

/// Yaw proxy from the 5-point landmarks (right-eye, left-eye, nose, ...): the
/// nose offset from the eye midpoint divided by the inter-ocular distance.
/// About 0 for a frontal face, growing in magnitude with head turn. `None`
/// without landmarks or with coincident eyes.
pub fn yaw_proxy(landmarks: &[(f32, f32); 5]) -> Option<f32> {
    let (right_eye, left_eye, nose) = (landmarks[0], landmarks[1], landmarks[2]);
    let inter_ocular = (left_eye.0 - right_eye.0).abs();
    if !inter_ocular.is_finite() || inter_ocular < 1e-6 {
        return None;
    }
    let mid_x = (left_eye.0 + right_eye.0) / 2.0;
    Some((nose.0 - mid_x) / inter_ocular)
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct FrameCounts {
    /// Frames pulled from the camera and examined (the decision's own count).
    pub seen: usize,
    pub skipped_pre_request: usize,
    pub skipped_stale: usize,
    pub no_face: usize,
    pub detector_error: usize,
    pub ambiguous_faces: usize,
    pub quality_rejected: usize,
    pub embedding_error: usize,
    /// Frames that entered the decision (embedded and scored).
    pub judged: usize,
}

#[derive(Debug, Serialize)]
pub struct SimilarityBlock {
    /// Per-frame template score of the judged frames.
    pub per_frame: Option<Distribution>,
    /// The attempt's final score (aggregate in `mean` mode, best window score
    /// in `n_of_m`).
    pub final_score: Option<f32>,
    pub threshold: Option<f32>,
    /// `final_score - threshold`.
    pub margin: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct DepthBlock {
    /// Depth relief of every frame the depth model ran on in this attempt.
    pub relief: Option<Distribution>,
    pub threshold: f32,
    /// `min relief - threshold` (the worst check) and `median - threshold`.
    pub margin_min: Option<f32>,
    pub margin_median: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct MiniFasnetBlock {
    /// `scored`, `model_not_loaded`, `no_frame`, `error` or `timeout`.
    pub state: &'static str,
    /// `real_logit - spoof_logit` of the deciding frame (positive = real).
    pub logit_diff: Option<f32>,
    /// Softmax real probability of the same frame.
    pub p_real: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct QualityBlock {
    pub face_area_frac: f32,
    /// Whole-frame Laplacian variance (the quality gate's own metric).
    pub sharpness: f32,
    /// Mean luma inside the face box.
    pub brightness: f32,
    /// See [`yaw_proxy`]; `null` without landmarks.
    pub yaw_proxy: Option<f32>,
}

/// One line of the shadow log.
#[derive(Debug, Serialize)]
pub struct ShadowRecord {
    pub schema_version: u32,
    /// `test`, `sudo` or `lock_screen`.
    pub source: &'static str,
    /// Calibration mode was active when the attempt started.
    pub calibration: bool,
    pub attempt_id: String,
    pub timestamp_unix_ms: u64,
    /// `recognized`, `not_recognized`, `suspended` or `error` (as `doorman --json test`).
    pub decision: String,
    pub reason_code: String,
    pub duration_ms: u64,
    pub decision_mode: &'static str,
    pub frames: FrameCounts,
    /// Frames that fed the decision, as reported by the decision itself.
    pub frames_used: usize,
    pub similarity: SimilarityBlock,
    pub depth: DepthBlock,
    pub minifasnet: MiniFasnetBlock,
    /// Metrics of the deciding (last quality) frame.
    pub quality: Option<QualityBlock>,
}

struct Inner {
    source: AttemptSource,
    calibration: bool,
    started: Instant,
    pipeline: Arc<MLPipeline>,
    log_path: PathBuf,
    log_max_bytes: u64,
    depth_threshold: f32,
    attempt_id: String,
    similarity_threshold: Option<f32>,
    n_of_m: bool,
    counts: FrameCounts,
    similarities: Vec<f32>,
    reliefs: Vec<f32>,
    /// Last frame that passed the quality gate: the deciding frame whenever the
    /// attempt reached a decision.
    last_quality_frame: Option<(Arc<DynamicImage>, Face)>,
}

/// Collector for one attempt. Disabled (a no-op) unless
/// `[liveness] shadow_pad_logging` is on.
pub struct ShadowPad {
    inner: Option<Inner>,
}

#[cfg(test)]
thread_local! {
    /// Test-only fault injection (mutation checks).
    pub(crate) static FAULT: std::cell::Cell<Option<Fault>> = const { std::cell::Cell::new(None) };
}

/// Where a test makes the shadow code panic.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fault {
    /// Inside every hook of the decision loop.
    Hook,
    /// While building the record in `finish`.
    Finish,
    /// In the detached task, before the MiniFASNet pass.
    Emit,
}

#[cfg(test)]
fn inject(at: Fault) {
    if FAULT.with(|f| f.get()) == Some(at) {
        panic!("injected shadow fault at {at:?}");
    }
}

impl ShadowPad {
    /// `calibrating`: calibration mode is active, which forces the collector on
    /// exactly like `[liveness] shadow_pad_logging`. Lock-screen attempts are
    /// collected only with `shadow_pad_include_lockscreen`, either way.
    pub fn new(
        config: &Config,
        pipeline: &Arc<MLPipeline>,
        source: AttemptSource,
        calibrating: bool,
    ) -> Self {
        if !(config.liveness.shadow_pad_logging || calibrating) {
            return Self::disabled();
        }
        if source == AttemptSource::LockScreen && !config.liveness.shadow_pad_include_lockscreen {
            return Self::disabled();
        }
        Self {
            inner: Some(Inner {
                source,
                calibration: calibrating,
                started: Instant::now(),
                pipeline: pipeline.clone(),
                log_path: PathBuf::from(&config.telemetry.shadow_pad_log_path),
                log_max_bytes: config.telemetry.timing_log_max_bytes,
                depth_threshold: config.authentication.liveness_depth_threshold,
                attempt_id: String::new(),
                similarity_threshold: None,
                n_of_m: false,
                counts: FrameCounts::default(),
                similarities: Vec::new(),
                reliefs: Vec::new(),
                last_quality_frame: None,
            }),
        }
    }

    pub fn disabled() -> Self {
        Self { inner: None }
    }

    #[cfg(test)]
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Run `hook` on the collector; a panic disables the collector instead of
    /// unwinding into the decision.
    fn guard(&mut self, hook: impl FnOnce(&mut Inner)) {
        let Some(inner) = self.inner.as_mut() else {
            return;
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            #[cfg(test)]
            inject(Fault::Hook);
            hook(inner);
        }));
        if outcome.is_err() {
            warn!("shadow PAD logging failed; disabled for this attempt (decision unaffected)");
            self.inner = None;
        }
    }

    pub fn attempt_id(&mut self, id: &str) {
        self.guard(|s| s.attempt_id = id.to_string());
    }

    /// The thresholds and decision mode this attempt runs with.
    pub fn context(&mut self, similarity_threshold: f32, n_of_m: bool) {
        self.guard(|s| {
            s.similarity_threshold = Some(similarity_threshold);
            s.n_of_m = n_of_m;
        });
    }

    pub fn frame_event(&mut self, event: FrameEvent) {
        self.guard(|s| {
            let c = &mut s.counts;
            match event {
                FrameEvent::SkippedPreRequest => c.skipped_pre_request += 1,
                FrameEvent::SkippedStale => c.skipped_stale += 1,
                FrameEvent::NoFace => c.no_face += 1,
                FrameEvent::DetectorError => c.detector_error += 1,
                FrameEvent::Ambiguous => c.ambiguous_faces += 1,
                FrameEvent::QualityRejected => c.quality_rejected += 1,
                FrameEvent::EmbeddingError => c.embedding_error += 1,
                FrameEvent::Judged => c.judged += 1,
            }
        });
    }

    /// Per-frame template score of a judged frame.
    pub fn similarity(&mut self, score: f32) {
        self.guard(|s| s.similarities.push(score));
    }

    /// Depth relief of one depth check (`None` when the check produced no value).
    pub fn depth_relief(&mut self, relief: Option<f32>) {
        self.guard(|s| s.reliefs.extend(relief));
    }

    /// The latest frame that passed the quality gate (an `Arc` clone, no pixels copied).
    pub fn quality_frame(&mut self, image: &Arc<DynamicImage>, face: &Face) {
        self.guard(|s| {
            let mut face = face.clone();
            face.other_faces.clear();
            s.last_quality_frame = Some((image.clone(), face));
        });
    }

    /// Build the record once the decision exists and hand the slow parts to a
    /// detached task. Returns the task so tests can await it; production code
    /// drops it. Never fails and never blocks the caller.
    pub fn finish(mut self, decision: &AuthDecision) -> Option<tokio::task::JoinHandle<()>> {
        let inner = self.inner.take()?;
        let decision = decision.clone();
        let built = catch_unwind(AssertUnwindSafe(move || {
            #[cfg(test)]
            inject(Fault::Finish);
            let handle = tokio::runtime::Handle::try_current().ok()?;
            let (record, job) = build(inner, &decision);
            Some(handle.spawn(emit(record, job)))
        }));
        match built {
            Ok(handle) => handle,
            Err(_) => {
                warn!("shadow PAD record failed (decision unaffected)");
                None
            }
        }
    }
}

/// What the detached task still has to do.
struct EmitJob {
    pipeline: Arc<MLPipeline>,
    frame: Option<(Arc<DynamicImage>, Face)>,
    log_path: PathBuf,
    log_max_bytes: u64,
}

fn build(inner: Inner, decision: &AuthDecision) -> (ShadowRecord, EmitJob) {
    let duration_ms = inner.started.elapsed().as_millis() as u64;
    let now = chrono::Utc::now();
    let report = doormand::test_report::classify(
        &decision.clone().into_response(),
        duration_ms,
        now.timestamp(),
    );
    let relief = distribution(&inner.reliefs);
    let threshold = inner.depth_threshold;
    let final_score = decision.similarity;
    let record = ShadowRecord {
        schema_version: SCHEMA_VERSION,
        source: inner.source.as_str(),
        calibration: inner.calibration,
        attempt_id: inner.attempt_id,
        timestamp_unix_ms: now.timestamp_millis().max(0) as u64,
        decision: report.decision.to_string(),
        reason_code: report.reason_code,
        duration_ms,
        decision_mode: if inner.n_of_m { "n_of_m" } else { "mean" },
        frames: FrameCounts {
            seen: decision.frames_seen,
            ..inner.counts
        },
        frames_used: decision.frames_used,
        similarity: SimilarityBlock {
            per_frame: distribution(&inner.similarities),
            final_score,
            threshold: inner.similarity_threshold,
            margin: final_score
                .zip(inner.similarity_threshold)
                .map(|(score, t)| score - t),
        },
        depth: DepthBlock {
            margin_min: relief.as_ref().map(|d| d.min - threshold),
            margin_median: relief.as_ref().map(|d| d.median - threshold),
            relief,
            threshold,
        },
        minifasnet: MiniFasnetBlock {
            state: "no_frame",
            logit_diff: None,
            p_real: None,
        },
        quality: None,
    };
    let job = EmitJob {
        pipeline: inner.pipeline,
        frame: inner.last_quality_frame,
        log_path: inner.log_path,
        log_max_bytes: inner.log_max_bytes,
    };
    (record, job)
}

fn quality_of(image: &DynamicImage, face: &Face) -> QualityBlock {
    QualityBlock {
        face_area_frac: face.area_frac(),
        sharpness: crate::pipeline::sharpness_score(image),
        brightness: crate::pipeline::face_brightness(image, face.bbox),
        yaw_proxy: face.landmarks.as_ref().and_then(yaw_proxy),
    }
}

/// The detached half: quality metrics, the MiniFASNet pass, the file write.
async fn emit(mut record: ShadowRecord, job: EmitJob) {
    #[cfg(test)]
    inject(Fault::Emit);
    if let Some((image, face)) = job.frame {
        let (q_image, q_face) = (image.clone(), face.clone());
        record.quality = tokio::task::spawn_blocking(move || quality_of(&q_image, &q_face))
            .await
            .ok();
        let scored = tokio::time::timeout(
            MINIFASNET_TIMEOUT,
            job.pipeline.shadow_minifasnet_logit_diff(&image, &face),
        )
        .await;
        record.minifasnet = match scored {
            Ok(Ok(Some(diff))) => MiniFasnetBlock {
                state: "scored",
                logit_diff: Some(diff),
                p_real: Some(1.0 / (1.0 + (-diff).exp())),
            },
            Ok(Ok(None)) => MiniFasnetBlock {
                state: "model_not_loaded",
                logit_diff: None,
                p_real: None,
            },
            Ok(Err(e)) => {
                debug!("shadow MiniFASNet error: {e}");
                MiniFasnetBlock {
                    state: "error",
                    logit_diff: None,
                    p_real: None,
                }
            }
            Err(_) => MiniFasnetBlock {
                state: "timeout",
                logit_diff: None,
                p_real: None,
            },
        };
    }
    match serde_json::to_string(&record) {
        Ok(line) => {
            debug!(target: "doorman::shadow_pad", "{line}");
            if let Err(e) = crate::timing::append_rotating(&job.log_path, job.log_max_bytes, &line)
            {
                warn!("shadow PAD log write failed: {e}");
            }
        }
        Err(e) => warn!("shadow PAD record serialization failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_of_known_values() {
        let d = distribution(&[0.9, 0.1, 0.5, 0.3, 0.7]).unwrap();
        assert_eq!(d.n, 5);
        assert_eq!((d.min, d.median, d.max), (0.1, 0.5, 0.9));
        // p10 at position 0.4 between 0.1 and 0.3.
        assert!((d.p10 - 0.18).abs() < 1e-6, "{}", d.p10);
    }

    #[test]
    fn distribution_ignores_non_finite_and_handles_empty() {
        assert!(distribution(&[]).is_none());
        assert!(distribution(&[f32::NAN, f32::INFINITY]).is_none());
        let d = distribution(&[f32::NAN, 0.4]).unwrap();
        assert_eq!((d.n, d.min, d.max), (1, 0.4, 0.4));
    }

    #[test]
    fn yaw_proxy_is_zero_frontal_and_signed_when_turned() {
        let frontal = [(0.4, 0.4), (0.6, 0.4), (0.5, 0.5), (0.43, 0.6), (0.57, 0.6)];
        assert!(yaw_proxy(&frontal).unwrap().abs() < 1e-6);
        let turned = [
            (0.4, 0.4),
            (0.6, 0.4),
            (0.56, 0.5),
            (0.43, 0.6),
            (0.57, 0.6),
        ];
        assert!((yaw_proxy(&turned).unwrap() - 0.3).abs() < 1e-5);
        let degenerate = [(0.5, 0.4), (0.5, 0.4), (0.5, 0.5), (0.4, 0.6), (0.6, 0.6)];
        assert!(yaw_proxy(&degenerate).is_none());
    }

    #[test]
    fn disabled_collector_is_inert() {
        let mut pad = ShadowPad::disabled();
        pad.similarity(0.5);
        pad.frame_event(FrameEvent::Judged);
        assert!(!pad.is_enabled());
    }
}
