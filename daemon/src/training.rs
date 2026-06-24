use crate::dataset::{AttemptRecord, GtSource, LabelEvent, Outcome};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    pub schema_version: u32,
    pub version: String,
    pub attempts_seen: usize,
    pub confirmed_genuine: usize,
    pub rejected: usize,
    pub suspected_spoof: usize,
    pub unknown: usize,
    pub output_dir: PathBuf,
    pub promoted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetReport {
    pub schema_version: u32,
    pub attempts_seen: usize,
    pub labels_seen: usize,
    pub confirmed_genuine: usize,
    pub rejected: usize,
    pub suspected_spoof: usize,
    pub unknown: usize,
    pub proactive_unlocked: usize,
    pub matched_attempts: usize,
    pub unlabeled_attempts: usize,
    pub capture_packs: usize,
    pub capture_frames: usize,
    #[serde(default)]
    pub failure_reasons: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ValidationMetrics {
    pub incumbent_far: f32,
    pub candidate_far: f32,
    pub incumbent_frr: f32,
    pub candidate_frr: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionReport {
    pub schema_version: u32,
    pub candidate_dir: PathBuf,
    pub current_path: PathBuf,
    pub validation: ValidationMetrics,
    pub promoted: bool,
    pub promoted_at_utc: String,
}

pub fn run_offline_training(
    data_dir: &Path,
    models_dir: &Path,
    min_confirmed: usize,
) -> Result<TrainingReport> {
    let dataset_report = inspect_dataset(data_dir)?;

    if dataset_report.confirmed_genuine < min_confirmed {
        return Err(anyhow!(
            "not enough confirmed-genuine labels: have {}, need {}",
            dataset_report.confirmed_genuine,
            min_confirmed
        ));
    }

    let version = format!("candidate-{}", chrono::Utc::now().format("%Y%m%dT%H%M%SZ"));
    let output_dir = models_dir.join("projector").join(&version);
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let report = TrainingReport {
        schema_version: 1,
        version,
        attempts_seen: dataset_report.attempts_seen,
        confirmed_genuine: dataset_report.confirmed_genuine,
        rejected: dataset_report.rejected,
        suspected_spoof: dataset_report.suspected_spoof,
        unknown: dataset_report.unknown,
        output_dir,
        promoted: false,
    };

    let manifest = serde_json::to_string_pretty(&report)?;
    std::fs::write(report.output_dir.join("manifest.json"), manifest)
        .with_context(|| "failed to write projector candidate manifest")?;
    Ok(report)
}

pub fn inspect_dataset(data_dir: &Path) -> Result<DatasetReport> {
    let dataset_dir = data_dir.join("dataset");
    let attempts = read_jsonl_prefix::<AttemptRecord>(&dataset_dir, "attempts")?;
    let labels = read_jsonl_prefix::<LabelEvent>(&dataset_dir, "labels")?;

    let confirmed_genuine = labels
        .iter()
        .filter(|event| {
            matches!(event.outcome, Outcome::ConfirmedGenuine)
                && !matches!(event.gt_source, GtSource::None)
        })
        .count();

    let mut failure_reasons = BTreeMap::new();
    for attempt in &attempts {
        if let Some(reason) = &attempt.failure_reason {
            *failure_reasons.entry(reason.clone()).or_insert(0) += 1;
        }
    }

    let labeled_attempt_ids: HashSet<&str> = labels
        .iter()
        .map(|event| event.attempt_id.as_str())
        .collect();
    let unlabeled_attempts = attempts
        .iter()
        .filter(|attempt| !labeled_attempt_ids.contains(attempt.attempt_id.as_str()))
        .count();
    let matched_attempts = attempts
        .iter()
        .filter(|attempt| !matches!(attempt.outcome, Outcome::Rejected))
        .count();
    let (capture_packs, capture_frames) = crate::dataset::capture_stats(data_dir)
        .with_context(|| format!("failed to inspect capture packs under {}", data_dir.display()))?;

    Ok(DatasetReport {
        schema_version: 1,
        attempts_seen: attempts.len(),
        labels_seen: labels.len(),
        confirmed_genuine,
        rejected: attempts
            .iter()
            .filter(|attempt| matches!(attempt.outcome, Outcome::Rejected))
            .count(),
        suspected_spoof: attempts
            .iter()
            .filter(|attempt| matches!(attempt.outcome, Outcome::SuspectedSpoof))
            .count(),
        unknown: attempts
            .iter()
            .filter(|attempt| matches!(attempt.outcome, Outcome::Unknown))
            .count(),
        proactive_unlocked: attempts
            .iter()
            .filter(|attempt| matches!(attempt.outcome, Outcome::ProactiveUnlocked))
            .count(),
        matched_attempts,
        unlabeled_attempts,
        capture_packs,
        capture_frames,
        failure_reasons,
    })
}

pub fn promote_projector_candidate(
    candidate_dir: &Path,
    validation: ValidationMetrics,
    max_frr_regression: f32,
) -> Result<PromotionReport> {
    if validation.candidate_far > validation.incumbent_far {
        write_validation(candidate_dir, validation, false)?;
        return Err(anyhow!(
            "FAR regression: candidate {:.6} > incumbent {:.6}",
            validation.candidate_far,
            validation.incumbent_far
        ));
    }
    if validation.candidate_frr > validation.incumbent_frr + max_frr_regression {
        write_validation(candidate_dir, validation, false)?;
        return Err(anyhow!(
            "FRR regression: candidate {:.6} > allowed {:.6}",
            validation.candidate_frr,
            validation.incumbent_frr + max_frr_regression
        ));
    }

    let manifest = candidate_dir.join("manifest.json");
    if !manifest.exists() {
        return Err(anyhow!(
            "candidate manifest missing: {}",
            manifest.display()
        ));
    }

    write_validation(candidate_dir, validation, true)?;
    let projector_dir = candidate_dir
        .parent()
        .ok_or_else(|| anyhow!("candidate has no projector parent directory"))?;
    let current_path = projector_dir.join("current");
    replace_current_link(&current_path, candidate_dir)?;

    let report = PromotionReport {
        schema_version: 1,
        candidate_dir: candidate_dir.to_path_buf(),
        current_path,
        validation,
        promoted: true,
        promoted_at_utc: chrono::Utc::now().to_rfc3339(),
    };
    std::fs::write(
        candidate_dir.join("promotion.json"),
        serde_json::to_string_pretty(&report)?,
    )
    .with_context(|| "failed to write promotion report")?;
    Ok(report)
}

#[derive(Debug, Clone, Serialize)]
struct ValidationReport {
    schema_version: u32,
    validation: ValidationMetrics,
    passed: bool,
    validated_at_utc: String,
}

fn write_validation(candidate_dir: &Path, validation: ValidationMetrics, passed: bool) -> Result<()> {
    std::fs::create_dir_all(candidate_dir)
        .with_context(|| format!("failed to create {}", candidate_dir.display()))?;
    let report = ValidationReport {
        schema_version: 1,
        validation,
        passed,
        validated_at_utc: chrono::Utc::now().to_rfc3339(),
    };
    std::fs::write(
        candidate_dir.join("validation.json"),
        serde_json::to_string_pretty(&report)?,
    )
    .with_context(|| "failed to write validation report")
}

fn replace_current_link(current_path: &Path, candidate_dir: &Path) -> Result<()> {
    if current_path.exists() || current_path.symlink_metadata().is_ok() {
        let metadata = current_path.symlink_metadata()?;
        if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() {
            std::fs::remove_dir_all(current_path)?;
        } else {
            std::fs::remove_file(current_path)?;
        }
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(candidate_dir, current_path)
            .with_context(|| format!("failed to update {}", current_path.display()))?;
    }
    #[cfg(not(unix))]
    {
        std::fs::write(current_path, candidate_dir.display().to_string())
            .with_context(|| format!("failed to update {}", current_path.display()))?;
    }
    Ok(())
}

fn read_jsonl_prefix<T: for<'de> Deserialize<'de>>(dir: &Path, prefix: &str) -> Result<Vec<T>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with(prefix) && name.ends_with(".jsonl") {
            paths.push(path);
        }
    }
    paths.sort();

    let mut records = Vec::new();
    for path in paths {
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        for (idx, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let record = serde_json::from_str(line).with_context(|| {
                format!("failed to parse {} line {}", path.display(), idx + 1)
            })?;
            records.push(record);
        }
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::dataset::{
        AttemptConditions, GallerySpreadStats, GtSource, LabelEvent, Outcome,
    };

    fn write_jsonl<T: serde::Serialize>(path: &std::path::Path, value: &T) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut line = serde_json::to_string(value).unwrap();
        line.push('\n');
        std::fs::write(path, line).unwrap();
    }

    fn append_jsonl<T: serde::Serialize>(path: &std::path::Path, value: &T) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut line = serde_json::to_string(value).unwrap();
        line.push('\n');
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        file.write_all(line.as_bytes()).unwrap();
    }

    fn sample_attempt() -> crate::dataset::AttemptRecord {
        crate::dataset::new_attempt_record(
            "attempt-1".to_string(),
            0,
            Some("owner".to_string()),
            0.91,
            None,
            Some(true),
            Outcome::Unknown,
            GtSource::None,
            &[0.1, 0.2, 0.3],
            None,
            GallerySpreadStats {
                nearest: 0.91,
                mean: 0.72,
                max: 0.91,
            },
            AttemptConditions::default(),
        )
    }

    fn sample_attempt_with(
        attempt_id: &str,
        outcome: Outcome,
        failure_reason: Option<&str>,
        matched_user: Option<&str>,
    ) -> crate::dataset::AttemptRecord {
        let mut attempt = sample_attempt();
        attempt.attempt_id = attempt_id.to_string();
        attempt.outcome = outcome;
        attempt.failure_reason = failure_reason.map(str::to_string);
        attempt.matched_user = matched_user.map(str::to_string);
        attempt
    }

    fn sample_label() -> LabelEvent {
        LabelEvent {
            schema_version: 1,
            attempt_id: "attempt-1".to_string(),
            timestamp_utc: "2026-06-23T12:00:01Z".to_string(),
            outcome: Outcome::ConfirmedGenuine,
            gt_source: GtSource::PamFaceAuth,
        }
    }

    #[test]
    fn inspects_dataset_and_counts_attempts_labels_and_failure_reasons() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        let dataset_dir = data_dir.join("dataset");
        append_jsonl(
            &dataset_dir.join("attempts-20260623.jsonl"),
            &sample_attempt_with(
                "attempt-1",
                Outcome::ProactiveUnlocked,
                Some("too_dark"),
                Some("owner"),
            ),
        );
        append_jsonl(
            &dataset_dir.join("attempts-20260623.jsonl"),
            &sample_attempt_with("attempt-2", Outcome::Unknown, None, Some("owner")),
        );
        append_jsonl(
            &dataset_dir.join("attempts-20260623.jsonl"),
            &sample_attempt_with(
                "attempt-3",
                Outcome::SuspectedSpoof,
                Some("spoof_pattern"),
                Some("guest"),
            ),
        );
        append_jsonl(
            &dataset_dir.join("attempts-20260623.jsonl"),
            &sample_attempt_with("attempt-4", Outcome::Rejected, None, None),
        );
        append_jsonl(
            &dataset_dir.join("labels-20260623.jsonl"),
            &LabelEvent {
                schema_version: 1,
                attempt_id: "attempt-1".to_string(),
                timestamp_utc: "2026-06-23T12:00:01Z".to_string(),
                outcome: Outcome::ConfirmedGenuine,
                gt_source: GtSource::PamFaceAuth,
            },
        );
        append_jsonl(
            &dataset_dir.join("labels-20260623.jsonl"),
            &LabelEvent {
                schema_version: 1,
                attempt_id: "attempt-4".to_string(),
                timestamp_utc: "2026-06-23T12:00:02Z".to_string(),
                outcome: Outcome::Rejected,
                gt_source: GtSource::PasswordFallback,
            },
        );

        let report = super::inspect_dataset(&data_dir).unwrap();

        assert_eq!(report.schema_version, 1);
        assert_eq!(report.attempts_seen, 4);
        assert_eq!(report.labels_seen, 2);
        assert_eq!(report.confirmed_genuine, 1);
        assert_eq!(report.proactive_unlocked, 1);
        assert_eq!(report.unknown, 1);
        assert_eq!(report.suspected_spoof, 1);
        assert_eq!(report.rejected, 1);
        assert_eq!(report.matched_attempts, 3);
        assert_eq!(report.unlabeled_attempts, 2);
        assert_eq!(report.capture_packs, 0);
        assert_eq!(report.capture_frames, 0);

        let mut expected_reasons = BTreeMap::new();
        expected_reasons.insert("spoof_pattern".to_string(), 1);
        expected_reasons.insert("too_dark".to_string(), 1);
        assert_eq!(report.failure_reasons, expected_reasons);
    }

    #[test]
    fn writes_projector_candidate_without_promoting() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        let models_dir = tmp.path().join("models");
        let dataset_dir = data_dir.join("dataset");
        write_jsonl(&dataset_dir.join("attempts-20260623.jsonl"), &sample_attempt());
        write_jsonl(&dataset_dir.join("labels-20260623.jsonl"), &sample_label());

        let report = super::run_offline_training(&data_dir, &models_dir, 1).unwrap();

        assert_eq!(report.attempts_seen, 1);
        assert_eq!(report.confirmed_genuine, 1);
        assert!(!report.promoted);
        assert!(report.output_dir.exists());
        assert!(report.output_dir.join("manifest.json").exists());
        assert!(!models_dir.join("projector/current").exists());
    }

    #[test]
    fn refuses_training_without_enough_confirmed_labels() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        let models_dir = tmp.path().join("models");
        let dataset_dir = data_dir.join("dataset");
        write_jsonl(&dataset_dir.join("attempts-20260623.jsonl"), &sample_attempt());
        write_jsonl(&dataset_dir.join("labels-20260623.jsonl"), &sample_label());

        let err = super::run_offline_training(&data_dir, &models_dir, 2).unwrap_err();

        assert!(err.to_string().contains("not enough confirmed-genuine labels"));
        assert!(!models_dir.join("projector").exists());
    }

    #[test]
    fn refuses_promotion_when_candidate_far_regresses() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        let models_dir = tmp.path().join("models");
        let dataset_dir = data_dir.join("dataset");
        write_jsonl(&dataset_dir.join("attempts-20260623.jsonl"), &sample_attempt());
        write_jsonl(&dataset_dir.join("labels-20260623.jsonl"), &sample_label());
        let report = super::run_offline_training(&data_dir, &models_dir, 1).unwrap();

        let err = super::promote_projector_candidate(
            &report.output_dir,
            super::ValidationMetrics {
                incumbent_far: 0.01,
                candidate_far: 0.02,
                incumbent_frr: 0.05,
                candidate_frr: 0.04,
            },
            0.01,
        )
        .unwrap_err();

        assert!(err.to_string().contains("FAR regression"));
        assert!(!models_dir.join("projector/current").exists());
    }

    #[test]
    fn promotes_candidate_when_validation_passes() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        let models_dir = tmp.path().join("models");
        let dataset_dir = data_dir.join("dataset");
        write_jsonl(&dataset_dir.join("attempts-20260623.jsonl"), &sample_attempt());
        write_jsonl(&dataset_dir.join("labels-20260623.jsonl"), &sample_label());
        let report = super::run_offline_training(&data_dir, &models_dir, 1).unwrap();

        let promotion = super::promote_projector_candidate(
            &report.output_dir,
            super::ValidationMetrics {
                incumbent_far: 0.01,
                candidate_far: 0.01,
                incumbent_frr: 0.05,
                candidate_frr: 0.055,
            },
            0.01,
        )
        .unwrap();

        assert!(promotion.promoted);
        assert_eq!(promotion.current_path, models_dir.join("projector/current"));
        assert!(models_dir.join("projector/current").exists());
        assert!(report.output_dir.join("validation.json").exists());
        assert!(report.output_dir.join("promotion.json").exists());
    }
}
