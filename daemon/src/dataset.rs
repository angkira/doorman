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

use chrono::Timelike;
use image::codecs::jpeg::JpegEncoder;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

/// Outcome label for one attempt. The ground-truth-confirmed `ConfirmedGenuine` is set
/// ONLY by the (future, M2) GT-labeling step — cosine alone never sets it. M1 logs the
/// weaker, immediately-known states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    /// GT-confirmed genuine (PAM/password success). Reserved for M2 labeling.
    ConfirmedGenuine,
    /// Genuine user was present but the recognizer failed to accept the attempt.
    /// For training this is a positive/genuine sample with a failed model decision.
    FalseNegative,
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttemptConditions {
    pub brightness: Option<f32>,
    pub face_area_frac: Option<f32>,
    pub sharpness: Option<f32>,
    pub hour_utc: Option<u32>,
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
    #[serde(default)]
    pub failure_reason: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturePackManifest {
    pub schema_version: u32,
    pub attempt_id: String,
    pub timestamp_utc: String,
    pub matched_user: Option<String>,
    pub outcome: Outcome,
    pub gt_source: GtSource,
    pub frame_count: usize,
    pub frames: Vec<CaptureFrameMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureFrameMeta {
    pub filename: String,
    pub source_index: usize,
    pub frame_size: (u32, u32),
    pub bbox_norm: Option<[f32; 4]>,
    pub conditions: AttemptConditions,
}

#[derive(Debug, Clone)]
pub struct CaptureFrameImage {
    pub source_index: usize,
    pub image: image::DynamicImage,
    pub bbox_norm: Option<[f32; 4]>,
    pub conditions: AttemptConditions,
}

/// The dataset directory for a given daemon data dir (`<data_dir>/dataset`).
pub fn dataset_dir(data_dir: &str) -> PathBuf {
    Path::new(data_dir).join("dataset")
}

pub fn gallery_spread_from_similarities(similarities: &[f32]) -> GallerySpreadStats {
    if similarities.is_empty() {
        return GallerySpreadStats::default();
    }
    let max = similarities
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
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

// Plain data parameters of a persisted record; a params struct would be a refactor.
#[allow(clippy::too_many_arguments)]
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
        failure_reason: None,
        gt_source,
        edgeface_emb: edgeface_emb.to_vec(),
        depth_proj_emb,
        gallery_spread,
        conditions,
    }
}

pub fn new_failure_attempt_record(
    attempt_id: String,
    lock_generation: u64,
    matched_user: Option<String>,
    outcome: Outcome,
    liveness_pass: Option<bool>,
    failure_reason: String,
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
        best_cosine: 0.0,
        liveness_relief: None,
        liveness_pass,
        outcome,
        failure_reason: Some(failure_reason),
        gt_source: GtSource::None,
        edgeface_emb: Vec::new(),
        depth_proj_emb: None,
        gallery_spread: GallerySpreadStats::default(),
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
        Ok(()) => debug!(
            "dataset: logged label event for attempt {}",
            event.attempt_id
        ),
        Err(e) => warn!("dataset: failed to log label event (non-fatal): {}", e),
    }
}

/// Whether raw frames of an attempt with this outcome may be stored: needs
/// `logging_enabled` and `keep_raw_frames`, and a successful unlock
/// additionally needs `capture_successful_unlocks`.
pub fn capture_pack_allowed(cfg: &doorman_shared::DatasetConfig, outcome: Outcome) -> bool {
    cfg.logging_enabled
        && cfg.keep_raw_frames
        && (outcome != Outcome::ConfirmedGenuine || cfg.capture_successful_unlocks)
}

// Plain data parameters of a persisted record; a params struct would be a refactor.
#[allow(clippy::too_many_arguments)]
pub fn write_capture_pack(
    data_dir: &str,
    enabled: bool,
    attempt_id: &str,
    matched_user: Option<String>,
    outcome: Outcome,
    gt_source: GtSource,
    max_packs_per_day: usize,
    jpeg_quality: u8,
    frames: &[CaptureFrameImage],
) {
    if !enabled || max_packs_per_day == 0 || frames.is_empty() {
        return;
    }
    match write_capture_pack_inner(
        data_dir,
        attempt_id,
        matched_user,
        outcome,
        gt_source,
        max_packs_per_day,
        jpeg_quality,
        frames,
    ) {
        Ok(()) => debug!("dataset: wrote capture pack {}", attempt_id),
        Err(e) => warn!("dataset: failed to write capture pack (non-fatal): {}", e),
    }
}

// Plain data parameters of a persisted record; a params struct would be a refactor.
#[allow(clippy::too_many_arguments)]
fn write_capture_pack_inner(
    data_dir: &str,
    attempt_id: &str,
    matched_user: Option<String>,
    outcome: Outcome,
    gt_source: GtSource,
    max_packs_per_day: usize,
    jpeg_quality: u8,
    frames: &[CaptureFrameImage],
) -> std::io::Result<()> {
    let day = chrono::Utc::now().format("%Y%m%d").to_string();
    let day_dir = dataset_dir(data_dir).join("captures").join(day);
    if day_dir.exists() {
        let existing = std::fs::read_dir(&day_dir)?
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .count();
        if existing >= max_packs_per_day {
            return Ok(());
        }
    }
    let pack_dir = dataset_dir(data_dir)
        .join("captures")
        .join(chrono::Utc::now().format("%Y%m%d").to_string())
        .join(sanitize_path_component(attempt_id));
    create_private_dir_all(&pack_dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let dataset = dataset_dir(data_dir);
        let captures = dataset.join("captures");
        let day = captures.join(chrono::Utc::now().format("%Y%m%d").to_string());
        let _ = std::fs::set_permissions(&dataset, std::fs::Permissions::from_mode(0o700));
        let _ = std::fs::set_permissions(&captures, std::fs::Permissions::from_mode(0o700));
        let _ = std::fs::set_permissions(&day, std::fs::Permissions::from_mode(0o700));
        let _ = std::fs::set_permissions(&pack_dir, std::fs::Permissions::from_mode(0o700));
    }

    let mut metas = Vec::with_capacity(frames.len());
    for (idx, frame) in frames.iter().enumerate() {
        let filename = format!("frame-{idx:03}-src-{}.jpg", frame.source_index);
        let path = pack_dir.join(&filename);
        let mut file = create_private_file(&path)?;
        let rgb = frame.image.to_rgb8();
        let mut encoder = JpegEncoder::new_with_quality(&mut file, jpeg_quality);
        encoder.encode_image(&rgb).map_err(std::io::Error::other)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = file.set_permissions(std::fs::Permissions::from_mode(0o600));
        }
        metas.push(CaptureFrameMeta {
            filename,
            source_index: frame.source_index,
            frame_size: (frame.image.width(), frame.image.height()),
            bbox_norm: frame.bbox_norm,
            conditions: frame.conditions.clone(),
        });
    }

    let manifest = CapturePackManifest {
        schema_version: 1,
        attempt_id: attempt_id.to_string(),
        timestamp_utc: chrono::Utc::now().to_rfc3339(),
        matched_user,
        outcome,
        gt_source,
        frame_count: metas.len(),
        frames: metas,
    };
    let manifest_path = pack_dir.join("manifest.json");
    let mut manifest_file = create_private_file(&manifest_path)?;
    serde_json::to_writer_pretty(&mut manifest_file, &manifest).map_err(std::io::Error::other)?;
    writeln!(manifest_file)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = manifest_file.set_permissions(std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

fn create_private_dir_all(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder.create(path)
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(path)
    }
}

fn create_private_file(path: &Path) -> std::io::Result<File> {
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)
}

#[allow(dead_code)]
pub fn capture_stats(data_dir: &Path) -> std::io::Result<(usize, usize)> {
    let captures_dir = data_dir.join("dataset").join("captures");
    if !captures_dir.exists() {
        return Ok((0, 0));
    }
    let mut packs = 0usize;
    let mut frames = 0usize;
    for day in std::fs::read_dir(captures_dir)? {
        let day = day?;
        if !day.file_type()?.is_dir() {
            continue;
        }
        for pack in std::fs::read_dir(day.path())? {
            let pack = pack?;
            if !pack.file_type()?.is_dir() {
                continue;
            }
            if pack.path().join("manifest.json").exists() {
                packs += 1;
                frames += std::fs::read_dir(pack.path())?
                    .filter_map(Result::ok)
                    .filter(|entry| {
                        entry
                            .path()
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("jpg"))
                    })
                    .count();
            }
        }
    }
    Ok((packs, frames))
}

fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
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
    #[test]
    fn raw_frames_of_successful_unlocks_need_an_explicit_opt_in() {
        let mut cfg = doorman_shared::DatasetConfig {
            logging_enabled: true,
            keep_raw_frames: true,
            ..Default::default()
        };
        assert!(!capture_pack_allowed(&cfg, Outcome::ConfirmedGenuine));
        for failed in [
            Outcome::Rejected,
            Outcome::FalseNegative,
            Outcome::SuspectedSpoof,
        ] {
            assert!(capture_pack_allowed(&cfg, failed), "{failed:?}");
        }
        cfg.capture_successful_unlocks = true;
        assert!(capture_pack_allowed(&cfg, Outcome::ConfirmedGenuine));
        cfg.keep_raw_frames = false;
        assert!(!capture_pack_allowed(&cfg, Outcome::ConfirmedGenuine));
        assert!(!capture_pack_allowed(&cfg, Outcome::Rejected));
        cfg.keep_raw_frames = true;
        cfg.logging_enabled = false;
        assert!(!capture_pack_allowed(&cfg, Outcome::Rejected));
    }

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
            failure_reason: None,
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
    fn deserializes_legacy_attempt_record_with_missing_failure_reason() {
        let legacy_record = r#"{"schema_version":1,"attempt_id":"legacy-1","timestamp_utc":"2026-06-23T12:00:00Z","lock_generation":1,"matched_user":null,"best_cosine":0.11,"liveness_relief":null,"liveness_pass":null,"outcome":"Rejected","gt_source":"None","edgeface_emb":[0.1,0.2],"depth_proj_emb":null,"gallery_spread":{"nearest":0.1,"mean":0.1,"max":0.1},"conditions":{"brightness":null,"face_area_frac":null,"sharpness":null,"hour_utc":null}}"#;
        let stored: AttemptRecord = serde_json::from_str(legacy_record).unwrap();

        assert_eq!(stored.failure_reason, None);
        assert_eq!(stored.attempt_id, "legacy-1");
    }

    #[test]
    fn new_failure_attempt_record_populates_failure_fields() {
        let rec = new_failure_attempt_record(
            "fail-1".to_string(),
            0,
            Some("owner".to_string()),
            Outcome::SuspectedSpoof,
            Some(false),
            "liveness_rejected".to_string(),
            AttemptConditions {
                face_area_frac: Some(0.08),
                ..AttemptConditions::default()
            },
        );

        assert!(matches!(rec.outcome, Outcome::SuspectedSpoof));
        assert_eq!(rec.liveness_pass, Some(false));
        assert_eq!(rec.failure_reason, Some("liveness_rejected".to_string()));
        assert!(rec.edgeface_emb.is_empty());
        assert_eq!(rec.best_cosine, 0.0);
        assert!(rec.depth_proj_emb.is_none());
        assert_eq!(rec.gallery_spread.nearest, 0.0);
        assert_eq!(rec.gallery_spread.mean, 0.0);
        assert_eq!(rec.gallery_spread.max, 0.0);
        assert_eq!(rec.conditions.face_area_frac, Some(0.08));
        assert!(rec.conditions.hour_utc.is_some());
        assert_eq!(rec.matched_user, Some("owner".to_string()));
        assert!(matches!(rec.gt_source, GtSource::None));
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
        assert!(matches!(outcome_from_match(false, None), Outcome::Rejected));
        assert!(matches!(
            outcome_from_match(true, Some(false)),
            Outcome::SuspectedSpoof
        ));
        assert!(matches!(
            outcome_from_match(true, Some(true)),
            Outcome::ProactiveUnlocked
        ));
        assert!(matches!(outcome_from_match(true, None), Outcome::Unknown));
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

    #[test]
    fn capture_pack_writes_manifest_and_jpegs_with_daily_cap() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().to_string_lossy();
        let frame = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            8,
            6,
            image::Rgb([32, 64, 96]),
        ));
        let frames = vec![CaptureFrameImage {
            source_index: 3,
            image: frame,
            bbox_norm: Some([0.1, 0.2, 0.3, 0.4]),
            conditions: AttemptConditions {
                brightness: Some(0.5),
                face_area_frac: Some(0.12),
                sharpness: Some(42.0),
                hour_utc: Some(13),
            },
        }];

        write_capture_pack(
            &data_dir,
            true,
            "attempt/unsafe",
            Some("owner".to_string()),
            Outcome::ConfirmedGenuine,
            GtSource::PamFaceAuth,
            1,
            90,
            &frames,
        );
        write_capture_pack(
            &data_dir,
            true,
            "attempt-second",
            Some("owner".to_string()),
            Outcome::ConfirmedGenuine,
            GtSource::PamFaceAuth,
            1,
            90,
            &frames,
        );

        let (packs, saved_frames) = capture_stats(tmp.path()).unwrap();
        assert_eq!(packs, 1);
        assert_eq!(saved_frames, 1);

        let day = chrono::Utc::now().format("%Y%m%d").to_string();
        let manifest_path = dataset_dir(&data_dir)
            .join("captures")
            .join(day)
            .join("attempt_unsafe")
            .join("manifest.json");
        let manifest: CapturePackManifest =
            serde_json::from_str(&std::fs::read_to_string(manifest_path).unwrap()).unwrap();
        assert_eq!(manifest.attempt_id, "attempt/unsafe");
        assert_eq!(manifest.frame_count, 1);
        assert_eq!(manifest.frames[0].source_index, 3);
        assert_eq!(manifest.frames[0].frame_size, (8, 6));
        assert_eq!(manifest.frames[0].bbox_norm, Some([0.1, 0.2, 0.3, 0.4]));
        assert_eq!(manifest.frames[0].conditions.sharpness, Some(42.0));
        assert!(dataset_dir(&data_dir)
            .join("captures")
            .join(chrono::Utc::now().format("%Y%m%d").to_string())
            .join("attempt_unsafe")
            .join(&manifest.frames[0].filename)
            .exists());
    }
}
