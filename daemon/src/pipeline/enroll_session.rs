//! Enrollment capture sessions: per-frame gates (brightness, pose, second
//! face), the per-run acceptance rules (minimum usable frames, minimum median
//! relief), session bookkeeping over stored templates, and the plain-language
//! report. Pure functions over measurements; no camera, models or I/O.
//!
//! One `doorman enroll` run is one capture session. Sessions are stored as they
//! finish (so progress is never lost); an enrollment is *complete* once the
//! stored templates span `enroll_min_sessions` sessions that differ from one
//! another. This module never relaxes the depth-liveness gate: it only adds
//! requirements, and the relief it stores is not consulted at authentication.

use super::enroll_plan::{plan_templates, EnrollMode};
use crate::ml::cosine_similarity;
use crate::ml::Face;
use crate::storage::TemplateMeta;
use chrono::{DateTime, Local, TimeZone};
use doorman_shared::config::{EnrollmentConfig, MultiFacePolicy};
use doormand::frame_policy::{choose_face, FaceChoice, Sighting};

/// Session id of templates stored before metadata existed (one pseudo-session).
const LEGACY_SESSION_ID: u64 = 0;

/// How far ahead of "now" a stored session id may be and still count as the
/// present (small clock adjustments); anything later is treated as expired.
const CLOCK_SLACK_MS: u64 = 60_000;

/// What was measured on one frame's face.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameMeasure {
    pub brightness: f32,
    pub area_frac: f32,
    pub yaw_ratio: Option<f32>,
    pub pitch_ratio: Option<f32>,
    /// A second face counted by the multi-face criteria was in view.
    pub second_face: bool,
}

/// Why a frame was not used for enrollment (besides detection/quality/depth).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameReject {
    SecondFace,
    TooDark,
    TooBright,
    HeadTurned,
}

/// Horizontal and vertical head-pose proxies from YuNet's 5 landmarks (right
/// eye, left eye, nose, right mouth corner, left mouth corner). Yaw is
/// `|nose offset from the eye midpoint| / eye distance` (0 = frontal); pitch is
/// the nose's position between the eye line (0) and the mouth line (1).
pub fn pose_ratios(face: &Face) -> (Option<f32>, Option<f32>) {
    let Some([re, le, nose, rm, lm]) = face.landmarks else {
        return (None, None);
    };
    let all = [re, le, nose, rm, lm];
    if all.iter().any(|p| !p.0.is_finite() || !p.1.is_finite()) {
        // Unusable landmarks: report NaN so `check_frame` fails the frame closed.
        return (Some(f32::NAN), Some(f32::NAN));
    }
    let eye_dx = (le.0 - re.0).abs();
    let eye_mid = ((re.0 + le.0) / 2.0, (re.1 + le.1) / 2.0);
    let yaw = (eye_dx > 1e-4).then(|| (nose.0 - eye_mid.0).abs() / eye_dx);
    let mouth_dy = (rm.1 + lm.1) / 2.0 - eye_mid.1;
    let pitch = (mouth_dy > 1e-4).then(|| (nose.1 - eye_mid.1) / mouth_dy);
    (yaw, pitch)
}

/// Measure a detected face. `rec` supplies the multi-face area/score criteria.
pub fn measure_face(
    image: &image::DynamicImage,
    face: &Face,
    rec: &doorman_shared::RecognitionConfig,
) -> FrameMeasure {
    let (yaw_ratio, pitch_ratio) = pose_ratios(face);
    let others: Vec<Sighting> = face
        .other_faces
        .iter()
        .map(|o| Sighting {
            area: o.area_frac(),
            score: o.confidence,
        })
        .collect();
    let second_face = choose_face(
        MultiFacePolicy::Reject,
        rec.multi_face_min_area_ratio,
        rec.multi_face_min_score,
        Sighting {
            area: face.area_frac(),
            score: face.confidence,
        },
        &others,
    ) == FaceChoice::Ambiguous;
    FrameMeasure {
        brightness: super::face_brightness(image, face.bbox),
        area_frac: face.area_frac(),
        yaw_ratio,
        pitch_ratio,
        second_face,
    }
}

/// `value >= floor`, false when either is NaN (a NaN measurement must fail).
fn at_least(value: f32, floor: f32) -> bool {
    value >= floor
}

/// Apply the enrollment-only frame gates. NaN measurements fail.
pub fn check_frame(cfg: &EnrollmentConfig, m: &FrameMeasure) -> Result<(), FrameReject> {
    if m.second_face {
        return Err(FrameReject::SecondFace);
    }
    // Non-finite pose (NaN landmarks) fails closed even with the pose gate off.
    if m.yaw_ratio.is_some_and(|y| !y.is_finite()) || m.pitch_ratio.is_some_and(|p| !p.is_finite())
    {
        return Err(FrameReject::HeadTurned);
    }
    if cfg.min_brightness > 0.0 && !at_least(m.brightness, cfg.min_brightness) {
        return Err(FrameReject::TooDark);
    }
    if cfg.max_brightness < doorman_shared::config::MAX_LUMA
        && !at_least(cfg.max_brightness, m.brightness)
    {
        return Err(FrameReject::TooBright);
    }
    if cfg.max_yaw_ratio > 0.0 {
        if let Some(yaw) = m.yaw_ratio {
            if !at_least(cfg.max_yaw_ratio, yaw) {
                return Err(FrameReject::HeadTurned);
            }
        }
    }
    Ok(())
}

/// Counts of frames dropped per reason in one run.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RejectTally {
    pub no_face: usize,
    pub quality: usize,
    pub second_face: usize,
    pub too_dark: usize,
    pub too_bright: usize,
    pub head_turned: usize,
    /// Depth relief below the liveness threshold (or not a number).
    pub not_live: usize,
    /// Depth check could not run (model missing / inference error).
    pub liveness_unavailable: usize,
    pub embed_error: usize,
}

impl RejectTally {
    pub fn note(&mut self, reject: FrameReject) {
        match reject {
            FrameReject::SecondFace => self.second_face += 1,
            FrameReject::TooDark => self.too_dark += 1,
            FrameReject::TooBright => self.too_bright += 1,
            FrameReject::HeadTurned => self.head_turned += 1,
        }
    }

    fn describe(&self) -> String {
        let parts = [
            (self.no_face, "no face found"),
            (self.quality, "face too small, blurry or dim"),
            (self.second_face, "a second face in view"),
            (self.too_dark, "too dark"),
            (self.too_bright, "over-exposed"),
            (self.head_turned, "head turned too far"),
            (self.not_live, "below the liveness depth threshold"),
            (self.liveness_unavailable, "depth check unavailable"),
            (self.embed_error, "embedding failed"),
        ];
        let shown: Vec<String> = parts
            .iter()
            .filter(|(n, _)| *n > 0)
            .map(|(n, why)| format!("{n} {why}"))
            .collect();
        if shown.is_empty() {
            "none".to_string()
        } else {
            shown.join(", ")
        }
    }
}

/// One frame that passed every gate.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub embedding: Vec<f32>,
    pub relief: Option<f32>,
    pub measure: FrameMeasure,
}

/// A template already stored for the user.
#[derive(Debug, Clone)]
pub struct StoredTemplate {
    pub embedding: Vec<f32>,
    pub meta: Option<TemplateMeta>,
}

/// Fixed inputs of one run besides the frames.
pub struct RunPolicy<'a> {
    pub cfg: &'a EnrollmentConfig,
    /// `authentication.liveness_depth_threshold`, or `None` when liveness is off.
    pub relief_threshold: Option<f32>,
    /// Frames sampled this run (for messages).
    pub frames_sampled: usize,
    /// Capacity of the template gallery.
    pub target: usize,
}

/// Summary of one capture session among the stored templates.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionSummary {
    /// `None` for templates stored before metadata existed.
    pub session_id: Option<u64>,
    pub templates: usize,
    pub relief_min: Option<f32>,
    pub relief_median: Option<f32>,
    pub relief_max: Option<f32>,
    pub brightness_median: Option<f32>,
    /// Differs enough from the earlier counted sessions to count.
    pub counted: bool,
    /// Why it does not count (novelty / brightness difference), when it does not.
    pub similarity_note: Option<String>,
}

/// Result of an accepted run: what to store and what to tell the owner.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub embeddings: Vec<Vec<f32>>,
    pub metas: Vec<Option<TemplateMeta>>,
    pub sessions: Vec<SessionSummary>,
    pub this_session_id: u64,
    pub counted_sessions: usize,
    pub required_sessions: usize,
    pub complete: bool,
    pub usable_frames: usize,
    pub frames_sampled: usize,
    pub tally: RejectTally,
    pub relief_threshold: Option<f32>,
}

/// Median of `values` (mean of the middle two for an even count).
pub fn median(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f32::total_cmp);
    let mid = v.len() / 2;
    Some(if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2.0
    })
}

fn min_max(values: &[f32]) -> (Option<f32>, Option<f32>) {
    (
        values.iter().copied().min_by(f32::total_cmp),
        values.iter().copied().max_by(f32::total_cmp),
    )
}

/// Accept or refuse a run, and plan the full template set to store.
///
/// Refusals (nothing is stored): fewer than `min_valid_frames` usable frames;
/// median relief below `min_median_relief_margin` x the liveness threshold;
/// nothing new to add in append mode / full gallery.
pub fn finish_run(
    policy: &RunPolicy<'_>,
    candidates: Vec<Candidate>,
    existing: Vec<StoredTemplate>,
    mode: EnrollMode,
    tally: RejectTally,
    now: DateTime<Local>,
) -> Result<RunOutcome, String> {
    let cfg = policy.cfg;
    let usable = candidates.len();
    if usable < cfg.min_valid_frames {
        return Err(format!(
            "Only {usable} of the {} usable frames required were captured (of {} sampled; dropped: {}). \
             Face the camera in even light, keep other faces out of view, and try again. Nothing was stored.",
            cfg.min_valid_frames,
            policy.frames_sampled,
            tally.describe()
        ));
    }

    let reliefs: Vec<f32> = candidates.iter().filter_map(|c| c.relief).collect();
    if let (Some(threshold), true) = (policy.relief_threshold, cfg.min_median_relief_margin > 0.0) {
        let required = cfg.min_median_relief_margin * threshold;
        // Every candidate carries a relief when liveness is on; none would mean
        // the depth stage did not run, which is also a refusal.
        let med = median(&reliefs).unwrap_or(f32::NAN);
        if !at_least(med, required) {
            tracing::warn!(
                "Enrollment refused: median depth relief {med:.3} is below {required:.3} (enrollment.min_median_relief_margin = {} x the liveness threshold); root can set that key (0 disables) if this environment is genuinely fine",
                cfg.min_median_relief_margin
            );
            return Err(
                "The depth relief of your face is too marginal here, so genuine unlocks would fail often. Nothing was stored. \
                 Try again in better, even light (face a window or lamp, avoid strong backlight, sit closer). \
                 If the light cannot be improved, ask the system administrator (root) to relax the enrollment depth-margin setting in the daemon configuration."
                    .to_string(),
            );
        }
    }

    // A further session must be the same person as the stored templates: the
    // median of the frames' best cosine, and the weakest of the three best
    // frames, must reach the floor. The first enrolment has nothing to compare.
    if !existing.is_empty() {
        let mut best: Vec<f32> = candidates
            .iter()
            .map(|c| {
                existing
                    .iter()
                    .map(|t| cosine_similarity(&c.embedding, &t.embedding))
                    .filter(|s| !s.is_nan())
                    .fold(-1.0_f32, f32::max)
            })
            .collect();
        let med = median(&best).unwrap_or(f32::NAN);
        best.sort_by(|a, b| b.total_cmp(a));
        let top_min = best.iter().take(3).copied().min_by(f32::total_cmp);
        let floor = cfg.enroll_same_person_min_similarity;
        if !at_least(med, floor) || !top_min.is_some_and(|t| at_least(t, floor)) {
            tracing::warn!(
                "Enrollment refused: median best similarity to the stored templates {med:.3}, weakest of top 3 {top_min:?}, required {floor:.3} (enrollment.enroll_same_person_min_similarity)"
            );
            return Err(
                "These frames do not look like the person already enrolled for this user. Nothing was stored. \
                 Enroll the same person, or remove the user first to start over."
                    .to_string(),
            );
        }
    }

    // Join the newest stored session when this run starts within the gap. A
    // stored session id in the future (the clock was stepped and restored) is
    // never joined: it would otherwise absorb every later run.
    let now_ms = u64::try_from(now.timestamp_millis()).unwrap_or(0);
    let newest = existing
        .iter()
        .filter_map(|t| t.meta.as_ref().map(|m| m.session_id))
        .filter(|id| *id <= now_ms.saturating_add(CLOCK_SLACK_MS))
        .max();
    let gap_ms = cfg.session_min_gap_secs.saturating_mul(1000);
    let session_id = match newest {
        Some(id) if now_ms.saturating_sub(id) < gap_ms => id,
        _ => now_ms.max(1),
    };

    let build = env!("CARGO_PKG_VERSION");
    let captured_at = now.to_rfc3339();
    let new_pairs: Vec<StoredTemplate> = candidates
        .iter()
        .map(|c| StoredTemplate {
            embedding: c.embedding.clone(),
            meta: Some(TemplateMeta {
                session_id,
                captured_at: captured_at.clone(),
                relief: c.relief,
                brightness: c.measure.brightness,
                face_area_frac: c.measure.area_frac,
                yaw_ratio: c.measure.yaw_ratio,
                pitch_ratio: c.measure.pitch_ratio,
                build: build.to_string(),
            }),
        })
        .collect();

    let existing_embeddings: Vec<Vec<f32>> = existing.iter().map(|t| t.embedding.clone()).collect();
    let planned = plan_templates(
        new_pairs.iter().map(|t| t.embedding.clone()).collect(),
        &existing_embeddings,
        mode,
        cfg.prune_cosine,
        policy.target,
    )?;
    let metas: Vec<Option<TemplateMeta>> = planned
        .iter()
        .map(|e| {
            existing
                .iter()
                .chain(new_pairs.iter())
                .find(|t| &t.embedding == e)
                .and_then(|t| t.meta.clone())
        })
        .collect();

    let sessions = summarize_sessions(cfg, &planned, &metas);
    let counted_sessions = sessions.iter().filter(|s| s.counted).count();
    Ok(RunOutcome {
        complete: counted_sessions >= cfg.enroll_min_sessions,
        embeddings: planned,
        metas,
        sessions,
        this_session_id: session_id,
        counted_sessions,
        required_sessions: cfg.enroll_min_sessions,
        usable_frames: usable,
        frames_sampled: policy.frames_sampled,
        tally,
        relief_threshold: policy.relief_threshold,
    })
}

/// Group stored templates into sessions (oldest first; metadata-free templates
/// form one leading legacy session) and decide which sessions count.
pub fn summarize_sessions(
    cfg: &EnrollmentConfig,
    embeddings: &[Vec<f32>],
    metas: &[Option<TemplateMeta>],
) -> Vec<SessionSummary> {
    let mut ids: Vec<u64> = metas
        .iter()
        .map(|m| m.as_ref().map_or(LEGACY_SESSION_ID, |m| m.session_id))
        .collect();
    ids.sort_unstable();
    ids.dedup();

    let mut out: Vec<SessionSummary> = Vec::with_capacity(ids.len());
    // Embeddings of the sessions that counted so far, and their brightness.
    let mut counted_embeddings: Vec<&Vec<f32>> = Vec::new();
    let mut counted_brightness: Vec<Option<f32>> = Vec::new();
    for id in ids {
        let members: Vec<usize> = (0..embeddings.len())
            .filter(|&i| {
                metas[i]
                    .as_ref()
                    .map_or(LEGACY_SESSION_ID, |m| m.session_id)
                    == id
            })
            .collect();
        let reliefs: Vec<f32> = members
            .iter()
            .filter_map(|&i| metas[i].as_ref().and_then(|m| m.relief))
            .collect();
        let brights: Vec<f32> = members
            .iter()
            .filter_map(|&i| metas[i].as_ref().map(|m| m.brightness))
            .collect();
        let brightness_median = median(&brights);
        let (relief_min, relief_max) = min_max(&reliefs);

        let (counted, similarity_note) = if counted_embeddings.is_empty() {
            (true, None)
        } else {
            let novelty = members
                .iter()
                .map(|&i| {
                    let best = counted_embeddings
                        .iter()
                        .map(|t| cosine_similarity(&embeddings[i], t))
                        .fold(f32::NEG_INFINITY, f32::max);
                    (1.0 - best).max(0.0)
                })
                .sum::<f32>()
                / members.len().max(1) as f32;
            let delta = brightness_median.map(|b| {
                counted_brightness
                    .iter()
                    .flatten()
                    .map(|c| (b - c).abs())
                    .fold(f32::INFINITY, f32::min)
            });
            let novel = novelty >= cfg.session_min_novelty;
            let lit_differently = cfg.session_min_brightness_delta > 0.0
                && delta.is_some_and(|d| d.is_finite() && d >= cfg.session_min_brightness_delta);
            if novel || lit_differently {
                (true, None)
            } else {
                let lit = delta
                    .filter(|d| d.is_finite())
                    .map_or("unknown".to_string(), |d| format!("{d:.0}"));
                (
                    false,
                    Some(format!(
                        "too similar to earlier sessions (embedding novelty {novelty:.2}, needs {:.2}; brightness difference {lit}, needs {:.0})",
                        cfg.session_min_novelty, cfg.session_min_brightness_delta
                    )),
                )
            }
        };
        if counted {
            counted_embeddings.extend(members.iter().map(|&i| &embeddings[i]));
            counted_brightness.push(brightness_median);
        }
        out.push(SessionSummary {
            session_id: (id != LEGACY_SESSION_ID).then_some(id),
            templates: members.len(),
            relief_min,
            relief_median: median(&reliefs),
            relief_max,
            brightness_median,
            counted,
            similarity_note,
        });
    }
    out
}

fn fmt_opt(v: Option<f32>, prec: usize) -> String {
    v.map_or("n/a".to_string(), |v| format!("{v:.prec$}"))
}

fn session_label(id: Option<u64>) -> String {
    match id.and_then(|ms| i64::try_from(ms).ok()) {
        None => "earlier (before session tracking)".to_string(),
        Some(ms) => Local.timestamp_millis_opt(ms).single().map_or_else(
            || "unknown time".to_string(),
            |t| t.format("%Y-%m-%d %H:%M").to_string(),
        ),
    }
}

/// Plain-language report for the `doorman enroll` output. Shown only to the
/// admin peer that issued the request.
pub fn render_report(username: &str, o: &RunOutcome) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let threshold = o
        .relief_threshold
        .map_or(String::new(), |t| format!(" (liveness threshold {t:.3})"));
    let _ = writeln!(
        s,
        "Capture recorded for '{username}': {} usable frames of {} sampled{threshold}.",
        o.usable_frames, o.frames_sampled
    );
    let _ = writeln!(s, "Dropped frames: {}.", o.tally.describe());
    let _ = writeln!(
        s,
        "Stored {} templates across {} session(s):",
        o.embeddings.len(),
        o.sessions.len()
    );
    for sess in &o.sessions {
        let here = if sess.session_id == Some(o.this_session_id) {
            " <- this run"
        } else {
            ""
        };
        let status = if sess.counted {
            "counts"
        } else {
            "does not count"
        };
        let _ = writeln!(
            s,
            "  {}: {} templates, depth relief min/median/max {}/{}/{}, face brightness {}; {status}{here}",
            session_label(sess.session_id),
            sess.templates,
            fmt_opt(sess.relief_min, 3),
            fmt_opt(sess.relief_median, 3),
            fmt_opt(sess.relief_max, 3),
            fmt_opt(sess.brightness_median, 0),
        );
        if let Some(note) = &sess.similarity_note {
            let _ = writeln!(s, "    ({note})");
        }
    }
    if o.complete {
        let _ = write!(
            s,
            "Enrollment COMPLETE: {} of {} required distinct sessions.",
            o.counted_sessions, o.required_sessions
        );
    } else {
        let _ = write!(
            s,
            "Enrollment INCOMPLETE: {} of {} required distinct sessions. Run 'doorman enroll {username}' again \
             after a few minutes, in different lighting (daylight versus lamp) or with a slightly different head position. \
             Unlocking is not blocked meanwhile, but reliability improves with more varied sessions.",
            o.counted_sessions, o.required_sessions
        );
    }
    s
}

/// Reason text when no frame at all was usable.
pub fn nothing_usable_reason(tally: &RejectTally, sampled: usize) -> String {
    format!(
        "No usable face frames out of {sampled} sampled (dropped: {}). \
         Make sure you are alone in view, facing the camera, in even light.",
        tally.describe()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(deg: f32) -> Vec<f32> {
        let r = deg.to_radians();
        vec![r.cos(), r.sin()]
    }

    fn measure(brightness: f32) -> FrameMeasure {
        FrameMeasure {
            brightness,
            area_frac: 0.1,
            yaw_ratio: Some(0.05),
            pitch_ratio: Some(0.55),
            second_face: false,
        }
    }

    fn cand(deg: f32, relief: Option<f32>, brightness: f32) -> Candidate {
        Candidate {
            embedding: unit(deg),
            relief,
            measure: measure(brightness),
        }
    }

    fn policy<'a>(cfg: &'a EnrollmentConfig, thr: Option<f32>) -> RunPolicy<'a> {
        RunPolicy {
            cfg,
            relief_threshold: thr,
            frames_sampled: 40,
            target: 30,
        }
    }

    fn at(hour: u32, min: u32) -> DateTime<Local> {
        Local.with_ymd_and_hms(2026, 11, 11, hour, min, 0).unwrap()
    }

    fn cfg() -> EnrollmentConfig {
        EnrollmentConfig {
            min_valid_frames: 3,
            // These fixtures use far-apart "conditions"; the identity check has
            // its own tests with the default.
            enroll_same_person_min_similarity: -1.0,
            ..EnrollmentConfig::default()
        }
    }

    fn run(
        cfg: &EnrollmentConfig,
        cands: Vec<Candidate>,
        existing: Vec<StoredTemplate>,
        now: DateTime<Local>,
    ) -> Result<RunOutcome, String> {
        finish_run(
            &policy(cfg, Some(0.05)),
            cands,
            existing,
            EnrollMode::Merge,
            RejectTally::default(),
            now,
        )
    }

    fn stored(o: &RunOutcome) -> Vec<StoredTemplate> {
        o.embeddings
            .iter()
            .cloned()
            .zip(o.metas.iter().cloned())
            .map(|(embedding, meta)| StoredTemplate { embedding, meta })
            .collect()
    }

    fn good(base: f32, brightness: f32) -> Vec<Candidate> {
        vec![
            cand(base, Some(0.10), brightness),
            cand(base + 20.0, Some(0.12), brightness),
            cand(base + 40.0, Some(0.14), brightness),
        ]
    }

    #[test]
    fn frame_gates_reject_dark_bright_turned_and_second_face() {
        let c = EnrollmentConfig::default();
        assert_eq!(check_frame(&c, &measure(120.0)), Ok(()));
        assert_eq!(check_frame(&c, &measure(10.0)), Err(FrameReject::TooDark));
        assert_eq!(
            check_frame(&c, &measure(250.0)),
            Err(FrameReject::TooBright)
        );
        assert_eq!(
            check_frame(&c, &measure(f32::NAN)),
            Err(FrameReject::TooDark),
            "NaN fails"
        );
        let mut turned = measure(120.0);
        turned.yaw_ratio = Some(0.6);
        assert_eq!(check_frame(&c, &turned), Err(FrameReject::HeadTurned));
        let mut second = measure(120.0);
        second.second_face = true;
        assert_eq!(check_frame(&c, &second), Err(FrameReject::SecondFace));
        // No landmarks: pose cannot be checked and the frame passes.
        let mut blind = measure(120.0);
        blind.yaw_ratio = None;
        assert_eq!(check_frame(&c, &blind), Ok(()));
    }

    #[test]
    fn gates_can_be_switched_off() {
        let c = EnrollmentConfig {
            min_brightness: 0.0,
            max_brightness: 255.0,
            max_yaw_ratio: 0.0,
            ..EnrollmentConfig::default()
        };
        let mut m = measure(0.0);
        m.yaw_ratio = Some(1.5);
        assert_eq!(check_frame(&c, &m), Ok(()));
        m.brightness = 255.0;
        assert_eq!(check_frame(&c, &m), Ok(()));
    }

    #[test]
    fn pose_ratios_are_zero_for_a_frontal_face_and_grow_when_turned() {
        let mut face = Face {
            bbox: (0.3, 0.3, 0.3, 0.3),
            confidence: 0.9,
            frame_dimensions: (640, 480),
            landmarks: Some([(0.4, 0.4), (0.6, 0.4), (0.5, 0.5), (0.43, 0.6), (0.57, 0.6)]),
            other_faces: Vec::new(),
        };
        let (yaw, pitch) = pose_ratios(&face);
        assert!(yaw.unwrap() < 1e-4);
        assert!((pitch.unwrap() - 0.5).abs() < 1e-4);
        face.landmarks = Some([
            (0.4, 0.4),
            (0.6, 0.4),
            (0.57, 0.5),
            (0.43, 0.6),
            (0.57, 0.6),
        ]);
        assert!((pose_ratios(&face).0.unwrap() - 0.35).abs() < 1e-3);
        face.landmarks = None;
        assert_eq!(pose_ratios(&face), (None, None));
    }

    #[test]
    fn too_few_usable_frames_refuses_and_stores_nothing() {
        let c = cfg();
        let err = run(&c, good(0.0, 120.0)[..2].to_vec(), vec![], at(9, 0)).unwrap_err();
        assert!(err.contains("Only 2 of the 3"), "{err}");
        assert!(err.contains("Nothing was stored"), "{err}");
    }

    #[test]
    fn marginal_median_relief_refuses_the_whole_run() {
        let c = cfg();
        // Threshold 0.05 -> required median 0.075.
        let low = vec![
            cand(0.0, Some(0.060), 120.0),
            cand(20.0, Some(0.070), 120.0),
            cand(40.0, Some(0.074), 120.0),
        ];
        let err = run(&c, low, vec![], at(9, 0)).unwrap_err();
        assert!(err.contains("too marginal"), "{err}");
        assert!(
            !err.contains("0.075") && !err.contains("min_median_relief_margin"),
            "{err}"
        );
        assert!(
            err.contains("better, even light") && err.contains("administrator"),
            "{err}"
        );
        // Exactly at the margin passes.
        let edge = vec![
            cand(0.0, Some(0.075), 120.0),
            cand(20.0, Some(0.075), 120.0),
            cand(40.0, Some(0.075), 120.0),
        ];
        assert!(run(&c, edge, vec![], at(9, 0)).is_ok());
    }

    #[test]
    fn margin_check_can_be_turned_off_and_is_skipped_without_liveness() {
        let off = EnrollmentConfig {
            min_median_relief_margin: 0.0,
            ..cfg()
        };
        let low = || {
            good(0.0, 120.0)
                .into_iter()
                .map(|mut c| {
                    c.relief = Some(0.051);
                    c
                })
                .collect::<Vec<_>>()
        };
        assert!(run(&off, low(), vec![], at(9, 0)).is_ok());
        let on = cfg();
        let no_liveness = finish_run(
            &policy(&on, None),
            good(0.0, 120.0)
                .into_iter()
                .map(|mut c| {
                    c.relief = None;
                    c
                })
                .collect(),
            vec![],
            EnrollMode::Merge,
            RejectTally::default(),
            at(9, 0),
        );
        assert!(no_liveness.is_ok());
    }

    #[test]
    fn a_lone_session_is_incomplete_and_a_distinct_second_one_completes() {
        let c = cfg();
        let first = run(&c, good(0.0, 120.0), vec![], at(9, 0)).unwrap();
        assert!(!first.complete);
        assert_eq!((first.counted_sessions, first.required_sessions), (1, 2));
        assert!(render_report("alice", &first).contains("INCOMPLETE"));

        // 10 minutes later, different light and clearly different embeddings.
        let second = run(&c, good(100.0, 70.0), stored(&first), at(9, 10)).unwrap();
        assert_eq!(second.sessions.len(), 2);
        assert!(second.complete, "{:?}", second.sessions);
        assert!(render_report("alice", &second).contains("COMPLETE"));
        assert!(second.metas.iter().all(Option::is_some));
    }

    #[test]
    fn a_run_inside_the_gap_joins_the_same_session() {
        let c = cfg();
        let first = run(&c, good(0.0, 120.0), vec![], at(9, 0)).unwrap();
        let second = run(&c, good(100.0, 70.0), stored(&first), at(9, 2)).unwrap();
        assert_eq!(second.this_session_id, first.this_session_id);
        assert_eq!(second.sessions.len(), 1);
        assert!(!second.complete);
        // With the gap disabled the same run is its own session.
        let c0 = EnrollmentConfig {
            session_min_gap_secs: 0,
            ..cfg()
        };
        let third = run(&c0, good(100.0, 70.0), stored(&first), at(9, 2)).unwrap();
        assert_eq!(third.sessions.len(), 2);
    }

    #[test]
    fn a_session_in_the_same_conditions_does_not_count() {
        let c = cfg();
        let first = run(&c, good(0.0, 120.0), vec![], at(9, 0)).unwrap();
        // Same pose and the same light: embeddings near the stored ones, 5
        // luma levels apart. Prune keeps nothing new, so use Append-free merge:
        // the stored set is unchanged and the session map shows no new session.
        let again = run(&c, good(1.0, 125.0), stored(&first), at(9, 30));
        // Near-duplicates are pruned, so no templates of the new session remain.
        let again = again.unwrap();
        assert_eq!(again.sessions.len(), 1);
        assert!(!again.complete);

        // Slightly different embeddings (not pruned) but same brightness: too similar.
        let tight = EnrollmentConfig {
            prune_cosine: 1.0,
            session_min_novelty: 0.5,
            ..cfg()
        };
        let again = run(&tight, good(5.0, 124.0), stored(&first), at(9, 30)).unwrap();
        assert_eq!(again.sessions.len(), 2);
        assert!(!again.sessions[1].counted);
        assert!(again.sessions[1]
            .similarity_note
            .as_deref()
            .unwrap()
            .contains("too similar"));
        assert!(!again.complete);
        // Brightness alone can make it distinct.
        let bright = run(&tight, good(5.0, 60.0), stored(&first), at(9, 30)).unwrap();
        assert!(bright.sessions[1].counted);
        assert!(bright.complete);
    }

    #[test]
    fn legacy_templates_form_one_counted_session_and_one_new_run_completes() {
        let c = cfg();
        let legacy: Vec<StoredTemplate> = [0.0, 25.0, 50.0]
            .iter()
            .map(|d| StoredTemplate {
                embedding: unit(*d),
                meta: None,
            })
            .collect();
        let o = run(&c, good(120.0, 100.0), legacy, at(9, 0)).unwrap();
        assert_eq!(o.sessions.len(), 2);
        assert_eq!(o.sessions[0].session_id, None);
        assert_eq!(o.sessions[0].templates, 3);
        assert!(o.sessions[0].counted && o.sessions[1].counted);
        assert!(o.complete);
        assert!(render_report("a", &o).contains("before session tracking"));
    }

    #[test]
    fn summaries_report_relief_and_brightness_per_session() {
        let c = cfg();
        let o = run(&c, good(0.0, 130.0), vec![], at(9, 0)).unwrap();
        let s = &o.sessions[0];
        assert_eq!(s.templates, 3);
        assert_eq!(s.relief_min, Some(0.10));
        assert_eq!(s.relief_median, Some(0.12));
        assert_eq!(s.relief_max, Some(0.14));
        assert_eq!(s.brightness_median, Some(130.0));
    }

    fn identity_cfg() -> EnrollmentConfig {
        EnrollmentConfig {
            min_valid_frames: 3,
            ..EnrollmentConfig::default()
        }
    }

    #[test]
    fn a_session_of_another_person_is_refused_and_nothing_is_planned() {
        let c = identity_cfg();
        let first = run(&c, good(0.0, 120.0), vec![], at(9, 0)).unwrap();
        // 120+ degrees away from every stored template: a different face.
        let err = run(&c, good(160.0, 70.0), stored(&first), at(9, 30)).unwrap_err();
        assert!(err.contains("do not look like the person"), "{err}");
        assert!(err.contains("Nothing was stored"), "{err}");
        assert!(!err.contains("0.5") && !err.contains("similarity"), "{err}");
        // Same person under other light is fine.
        assert!(run(&c, good(15.0, 70.0), stored(&first), at(9, 30)).is_ok());
    }

    #[test]
    fn first_enrolment_skips_the_identity_check() {
        let c = EnrollmentConfig {
            enroll_same_person_min_similarity: 1.0,
            ..identity_cfg()
        };
        assert!(run(&c, good(0.0, 120.0), vec![], at(9, 0)).is_ok());
    }

    #[test]
    fn identity_check_boundary_is_inclusive_and_needs_the_weakest_top_frames() {
        // Stored template at 0 degrees; candidates exactly at the floor cosine.
        let floor = 0.5_f32;
        let c = EnrollmentConfig {
            enroll_same_person_min_similarity: floor,
            ..identity_cfg()
        };
        let existing = vec![StoredTemplate {
            embedding: vec![1.0, 0.0],
            meta: None,
        }];
        let edge = |x: f32| Candidate {
            embedding: vec![x, (1.0 - x * x).sqrt()],
            relief: Some(0.1),
            measure: measure(120.0),
        };
        let all_at_floor = vec![edge(floor), edge(floor), edge(floor)];
        assert!(run(&c, all_at_floor, existing.clone(), at(9, 0)).is_ok());
        // Median ok (0.9) but one of the three best frames is below the floor.
        let weak = vec![edge(0.9), edge(0.9), edge(0.49)];
        assert!(run(&c, weak, existing.clone(), at(9, 0)).is_err());
        // NaN similarity fails closed.
        let nan = vec![edge(f32::NAN), edge(f32::NAN), edge(f32::NAN)];
        assert!(run(&c, nan, existing, at(9, 0)).is_err());
    }

    #[test]
    fn session_gap_boundary_is_exclusive() {
        let c = cfg(); // gap 300 s
        let first = run(&c, good(0.0, 120.0), vec![], at(9, 0)).unwrap();
        let just_inside = at(9, 0) + chrono::Duration::seconds(299);
        let joined = run(&c, good(100.0, 70.0), stored(&first), just_inside).unwrap();
        assert_eq!(joined.this_session_id, first.this_session_id);
        let exactly = at(9, 5);
        let split = run(&c, good(100.0, 70.0), stored(&first), exactly).unwrap();
        assert_ne!(split.this_session_id, first.this_session_id);
    }

    #[test]
    fn a_stored_session_in_the_future_is_never_joined() {
        let c = cfg();
        let first = run(&c, good(0.0, 120.0), vec![], at(9, 0)).unwrap();
        // Clock now reads 3 days earlier than the stored session (stepped back).
        let earlier = at(9, 0) - chrono::Duration::days(3);
        let later = run(&c, good(100.0, 70.0), stored(&first), earlier).unwrap();
        assert_ne!(later.this_session_id, first.this_session_id);
        assert_eq!(later.sessions.len(), 2);
        // A small step (within the slack) still joins.
        let near = at(9, 0) - chrono::Duration::seconds(30);
        let joined = run(&c, good(100.0, 70.0), stored(&first), near).unwrap();
        assert_eq!(joined.this_session_id, first.this_session_id);
    }

    #[test]
    fn novelty_boundary_is_inclusive() {
        let c = EnrollmentConfig {
            session_min_novelty: 1.0,
            session_min_brightness_delta: 20.0,
            ..cfg()
        };
        let meta = |id: u64| {
            Some(TemplateMeta {
                session_id: id,
                captured_at: String::new(),
                relief: Some(0.1),
                brightness: 100.0,
                face_area_frac: 0.1,
                yaw_ratio: None,
                pitch_ratio: None,
                build: String::new(),
            })
        };
        // Orthogonal embeddings: novelty is exactly 1.0 = the requirement.
        let s = summarize_sessions(&c, &[vec![1.0, 0.0], vec![0.0, 1.0]], &[meta(1), meta(2)]);
        assert!(s[1].counted, "{:?}", s[1]);
    }

    #[test]
    fn nan_landmarks_fail_the_frame_closed_even_with_the_pose_gate_off() {
        let mut face = Face {
            bbox: (0.3, 0.3, 0.3, 0.3),
            confidence: 0.9,
            frame_dimensions: (640, 480),
            landmarks: Some([
                (0.4, 0.4),
                (0.6, 0.4),
                (f32::NAN, 0.5),
                (0.43, 0.6),
                (0.57, 0.6),
            ]),
            other_faces: Vec::new(),
        };
        let (yaw, pitch) = pose_ratios(&face);
        let m = FrameMeasure {
            yaw_ratio: yaw,
            pitch_ratio: pitch,
            ..measure(120.0)
        };
        assert_eq!(
            check_frame(&EnrollmentConfig::default(), &m),
            Err(FrameReject::HeadTurned)
        );
        let off = EnrollmentConfig {
            max_yaw_ratio: 0.0,
            ..EnrollmentConfig::default()
        };
        assert_eq!(check_frame(&off, &m), Err(FrameReject::HeadTurned));
        face.landmarks = Some([
            (f32::INFINITY, 0.4),
            (0.6, 0.4),
            (0.5, 0.5),
            (0.4, 0.6),
            (0.6, 0.6),
        ]);
        assert!(pose_ratios(&face).0.unwrap().is_nan());
    }

    #[test]
    fn median_handles_even_odd_and_empty() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
    }

    #[test]
    fn append_mode_without_new_templates_is_refused() {
        let c = cfg();
        let first = run(&c, good(0.0, 120.0), vec![], at(9, 0)).unwrap();
        let err = finish_run(
            &policy(&c, Some(0.05)),
            good(0.5, 120.0),
            stored(&first),
            EnrollMode::Append,
            RejectTally::default(),
            at(9, 30),
        )
        .unwrap_err();
        assert!(err.contains("near-duplicate"), "{err}");
    }

    #[test]
    fn tally_describes_reasons() {
        let mut t = RejectTally::default();
        assert_eq!(t.describe(), "none");
        t.note(FrameReject::TooDark);
        t.note(FrameReject::SecondFace);
        t.not_live = 4;
        let d = t.describe();
        assert!(d.contains("4 below the liveness depth threshold"), "{d}");
        assert!(
            d.contains("1 too dark") && d.contains("1 a second face"),
            "{d}"
        );
    }
}
