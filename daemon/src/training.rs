use crate::dataset::{AttemptRecord, GtSource, LabelEvent, Outcome};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
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

pub fn run_offline_training(
    data_dir: &Path,
    models_dir: &Path,
    min_confirmed: usize,
) -> Result<TrainingReport> {
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

    if confirmed_genuine < min_confirmed {
        return Err(anyhow!(
            "not enough confirmed-genuine labels: have {}, need {}",
            confirmed_genuine,
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
        attempts_seen: attempts.len(),
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
        output_dir,
        promoted: false,
    };

    let manifest = serde_json::to_string_pretty(&report)?;
    std::fs::write(report.output_dir.join("manifest.json"), manifest)
        .with_context(|| "failed to write projector candidate manifest")?;
    Ok(report)
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
    use crate::dataset::{
        AttemptConditions, GallerySpreadStats, GtSource, LabelEvent, Outcome,
    };

    fn write_jsonl<T: serde::Serialize>(path: &std::path::Path, value: &T) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut line = serde_json::to_string(value).unwrap();
        line.push('\n');
        std::fs::write(path, line).unwrap();
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
}
