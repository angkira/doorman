//! Attempt-log dataset (R&D milestone M1, see
//! `.plans/2026-06-23_learnable-projector-rnd.md`): an append-only, per-unlock-attempt
//! record of model embeddings + scores + outcome, to seed the future learnable depth
//! projector / fusion / self-training work.
//!
//! SEPARATE from the live gallery (`embeddings.bin`): the auth hot path keeps reading
//! only the gallery; this dataset is WRITE-ONLY from the hot path and read-only for an
//! offline `doorman train`. Default OFF (`[dataset] logging_enabled`). Best-effort:
//! any logging error is swallowed so authentication is NEVER affected.
//!
//! Privacy: EMBEDDINGS ONLY — raw frames are never persisted here.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::{debug, warn};
use chrono::Timelike;

/// Outcome label for one attempt. The ground-truth-confirmed `ConfirmedGenuine` is set
/// ONLY by the (future, M2) GT-labeling step — cosine alone never sets it. M1 logs the
/// weaker, immediately-known states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Outcome {
    /// GT-confirmed genuine (PAM/password success). Reserved for M2 labeling.
    ConfirmedGenuine,
    /// Recognized (>= threshold) + liveness PASS at the unlock decision; GT not yet
    /// confirmed (a candidate positive).
    ProactiveUnlocked,
    /// Recognized but the FATAL liveness/anti-spoof gate FAILED — a hard negative.
    SuspectedSpoof,
    /// Below the recognition threshold.
    Rejected,
    Unknown,
}

/// Independent source for a ground-truth label. M1 writes `None`; M2 appends
/// label events when a trusted path confirms the attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GtSource {
    PamFaceAuth,
    ProactiveThenStayed,
    PasswordFallback,
    None,
}

/// Summary of how the scored embedding sat relative to the enrolled gallery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GallerySpreadStats {
    pub nearest: f32,
    pub mean: f32,
    pub max: f32,
}

impl Default for GallerySpreadStats {
    fn default() -> Self {
        Self {
            nearest: 0.0,
            mean: 0.0,
            max: 0.0,
        }
    }
}

/// Lightweight capture conditions for stratified offline evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptConditions {
    pub brightness: Option<f32>,
    pub face_area_frac: Option<f32>,
    pub sharpness: Option<f32>,
    pub hour_utc: Option<u32>,
}

impl Default for AttemptConditions {
    fn default() -> Self {
        Self {
            brightness: None,
            face_area_frac: None,
            sharpness: None,
            hour_utc: None,
        }
    }
}

/// One scored attempt. Vectors only — no raw frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptRecord {
    pub schema_version: u32,
    pub attempt_id: String,
    pub timestamp_utc: String,
    pub lock_generation: u64,
    pub matched_user: Option<String>,
    pub best_cosine: f32,
    pub liveness_relief: Option<f32>,
    pub liveness_pass: Option<bool>,
    pub outcome: Outcome,
    pub gt_source: GtSource,
    /// Aggregated EdgeFace RGB embedding (the one scored against the gallery).
    pub edgeface_emb: Vec<f32>,
    pub depth_proj_emb: Option<Vec<f32>>,
    pub gallery_spread: GallerySpreadStats,
    pub conditions: AttemptConditions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelEvent {
    pub schema_version: u32,
    pub attempt_id: String,
    pub timestamp_utc: String,
    pub outcome: Outcome,
    pub gt_source: GtSource,
}

/// The dataset directory for a given daemon data dir (`<data_dir>/dataset`).
pub fn dataset_dir(data_dir: &str) -> PathBuf {
    Path::new(data_dir).join("dataset")
}

pub fn gallery_spread_from_similarities(similarities: &[f32]) -> GallerySpreadStats {
    if similarities.is_empty() {
        return GallerySpreadStats::default();
    }
    let max = similarities.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = similarities.iter().sum::<f32>() / similarities.len() as f32;
    GallerySpreadStats {
        nearest: max,
        mean,
        max,
    }
}

pub fn outcome_from_match(matched: bool, liveness_pass: Option<bool>) -> Outcome {
    if !matched {
        return Outcome::Rejected;
    }
    match liveness_pass {
        Some(true) => Outcome::ProactiveUnlocked,
        Some(false) => Outcome::SuspectedSpoof,
        None => Outcome::Unknown,
    }
}

pub fn new_attempt_record(
    attempt_id: String,
    lock_generation: u64,
    matched_user: Option<String>,
    best_cosine: f32,
    liveness_relief: Option<f32>,
    liveness_pass: Option<bool>,
    outcome: Outcome,
    gt_source: GtSource,
    edgeface_emb: &[f32],
    depth_proj_emb: Option<Vec<f32>>,
    gallery_spread: GallerySpreadStats,
    mut conditions: AttemptConditions,
) -> AttemptRecord {
    let now = chrono::Utc::now();
    if conditions.hour_utc.is_none() {
        conditions.hour_utc = Some(now.hour());
    }
    AttemptRecord {
        schema_version: 1,
        attempt_id,
        timestamp_utc: now.to_rfc3339(),
        lock_generation,
        matched_user,
        best_cosine,
        liveness_relief,
        liveness_pass,
        outcome,
        gt_source,
        edgeface_emb: edgeface_emb.to_vec(),
        depth_proj_emb,
        gallery_spread,
        conditions,
    }
}

/// Append one record to the day's shard IF logging is enabled. Best-effort — never
/// returns an error to the caller, never panics: a logging failure must not affect auth.
pub fn log_attempt_capped(
    data_dir: &str,
    enabled: bool,
    max_records_per_day: usize,
    rec: &AttemptRecord,
) {
    if !enabled {
        return;
    }
    match append_capped(&dataset_dir(data_dir), "attempts", max_records_per_day, rec) {
        Ok(()) => debug!(
            "dataset: logged attempt {} ({:?})",
            rec.attempt_id, rec.outcome
        ),
        Err(e) => warn!("dataset: failed to log attempt (non-fatal): {}", e),
    }
}

pub fn log_label_event(data_dir: &str, enabled: bool, event: &LabelEvent) {
    if !enabled {
        return;
    }
    match append_jsonl(&dataset_dir(data_dir), "labels", event) {
        Ok(()) => debug!("dataset: logged label event for attempt {}", event.attempt_id),
        Err(e) => warn!("dataset: failed to log label event (non-fatal): {}", e),
    }
}

fn append_capped<T: Serialize>(
    dir: &Path,
    prefix: &str,
    max_records_per_day: usize,
    rec: &T,
) -> std::io::Result<()> {
    if max_records_per_day == 0 {
        return Ok(());
    }
    let day = chrono::Utc::now().format("%Y%m%d");
    let path = dir.join(format!("{prefix}-{day}.jsonl"));
    if path.exists() {
        let existing = std::fs::read_to_string(&path)?.lines().count();
        if existing >= max_records_per_day {
            return Ok(());
        }
    }
    append_jsonl(dir, prefix, rec)
}

fn append_jsonl<T: Serialize>(dir: &Path, prefix: &str, rec: &T) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
    }
    let day = chrono::Utc::now().format("%Y%m%d");
    let path = dir.join(format!("{prefix}-{day}.jsonl"));
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = f.set_permissions(std::fs::Permissions::from_mode(0o600));
    }
    let line = serde_json::to_string(rec)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writeln!(f, "{line}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> AttemptRecord {
        AttemptRecord {
            schema_version: 1,
            attempt_id: "attempt-1".to_string(),
            timestamp_utc: "2026-06-23T12:00:00Z".to_string(),
            lock_generation: 7,
            matched_user: Some("owner".to_string()),
            best_cosine: 0.91,
            liveness_relief: None,
            liveness_pass: Some(true),
            outcome: Outcome::ProactiveUnlocked,
            gt_source: GtSource::None,
            edgeface_emb: vec![0.1, 0.2, 0.3],
            depth_proj_emb: None,
            gallery_spread: GallerySpreadStats {
                nearest: 0.91,
                mean: 0.72,
                max: 0.91,
            },
            conditions: AttemptConditions {
                brightness: Some(0.5),
                face_area_frac: Some(0.08),
                sharpness: Some(12.0),
                hour_utc: Some(14),
            },
        }
    }

    #[test]
    fn disabled_logging_does_not_create_dataset_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().to_string_lossy();

        log_attempt_capped(&data_dir, false, usize::MAX, &sample_record());

        assert!(!dataset_dir(&data_dir).exists());
    }

    #[test]
    fn enabled_logging_appends_jsonl_record_under_dataset_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().to_string_lossy();

        log_attempt_capped(&data_dir, true, usize::MAX, &sample_record());

        let shard = dataset_dir(&data_dir).join(format!(
            "attempts-{}.jsonl",
            chrono::Utc::now().format("%Y%m%d")
        ));
        let content = std::fs::read_to_string(shard).unwrap();
        let stored: AttemptRecord = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(stored.attempt_id, "attempt-1");
        assert_eq!(stored.lock_generation, 7);
        assert!(matches!(stored.outcome, Outcome::ProactiveUnlocked));
        assert!(matches!(stored.gt_source, GtSource::None));
        assert_eq!(stored.edgeface_emb, vec![0.1, 0.2, 0.3]);
        assert_eq!(stored.schema_version, 1);
        assert!(stored.depth_proj_emb.is_none());
        assert_eq!(stored.gallery_spread.nearest, 0.91);
        assert_eq!(stored.gallery_spread.mean, 0.72);
        assert_eq!(stored.gallery_spread.max, 0.91);
        assert_eq!(stored.conditions.brightness, Some(0.5));
        assert_eq!(stored.conditions.face_area_frac, Some(0.08));
        assert_eq!(stored.conditions.sharpness, Some(12.0));
        assert_eq!(stored.conditions.hour_utc, Some(14));
    }

    #[test]
    fn capped_attempt_logging_skips_records_after_daily_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().to_string_lossy();

        log_attempt_capped(&data_dir, true, 1, &sample_record());
        log_attempt_capped(&data_dir, true, 1, &sample_record());

        let shard = dataset_dir(&data_dir).join(format!(
            "attempts-{}.jsonl",
            chrono::Utc::now().format("%Y%m%d")
        ));
        let content = std::fs::read_to_string(shard).unwrap();
        assert_eq!(content.lines().count(), 1);
    }

    #[test]
    fn gallery_spread_uses_nearest_mean_and_max_similarity() {
        let spread = gallery_spread_from_similarities(&[0.9, 0.7, 0.5]);

        assert_eq!(spread.nearest, 0.9);
        assert_eq!(spread.mean, 0.7);
        assert_eq!(spread.max, 0.9);
    }

    #[test]
    fn outcome_from_match_tracks_rejected_spoof_and_unknown() {
        assert!(matches!(
            outcome_from_match(false, None),
            Outcome::Rejected
        ));
        assert!(matches!(
            outcome_from_match(true, Some(false)),
            Outcome::SuspectedSpoof
        ));
        assert!(matches!(
            outcome_from_match(true, Some(true)),
            Outcome::ProactiveUnlocked
        ));
        assert!(matches!(
            outcome_from_match(true, None),
            Outcome::Unknown
        ));
    }

    #[test]
    fn enabled_label_logging_appends_jsonl_event() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().to_string_lossy();
        let event = LabelEvent {
            schema_version: 1,
            attempt_id: "attempt-1".to_string(),
            timestamp_utc: "2026-06-23T12:00:01Z".to_string(),
            outcome: Outcome::ConfirmedGenuine,
            gt_source: GtSource::PamFaceAuth,
        };

        log_label_event(&data_dir, true, &event);

        let shard = dataset_dir(&data_dir).join(format!(
            "labels-{}.jsonl",
            chrono::Utc::now().format("%Y%m%d")
        ));
        let content = std::fs::read_to_string(shard).unwrap();
        let stored: LabelEvent = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(stored.attempt_id, "attempt-1");
        assert!(matches!(stored.outcome, Outcome::ConfirmedGenuine));
        assert!(matches!(stored.gt_source, GtSource::PamFaceAuth));
    }
}
