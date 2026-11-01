//! Manual purge of old dataset files (`doorman dataset purge`).
//!
//! Dataset files are named after their UTC day: `attempts-YYYYMMDD.jsonl`,
//! `labels-YYYYMMDD.jsonl` and `captures/YYYYMMDD/`. A purge selects entries by
//! that date (not by mtime, which a backup restore would change). The daemon never
//! calls this on its own: deletion is always an explicit owner action.

use chrono::{Duration, NaiveDate};
use std::path::{Path, PathBuf};

/// Date format of the day embedded in dataset file and directory names.
const DAY_FORMAT: &str = "%Y%m%d";
/// Prefixes of the per-day JSONL shards in the dataset directory.
const SHARD_PREFIXES: [&str; 2] = ["attempts-", "labels-"];
/// Extension of the JSONL shards.
const SHARD_SUFFIX: &str = ".jsonl";
/// Name of the capture-pack directory inside the dataset directory.
const CAPTURES_DIR: &str = "captures";

/// Entries of `dataset_dir` whose day is strictly older than `older_than_days`
/// days before `today`. Entries with unrecognised names are never selected.
pub fn purge_candidates(
    dataset_dir: &Path,
    today: NaiveDate,
    older_than_days: u32,
) -> std::io::Result<Vec<PathBuf>> {
    let cutoff = today - Duration::days(i64::from(older_than_days));
    let mut out = Vec::new();
    if !dataset_dir.is_dir() {
        return Ok(out);
    }
    for entry in std::fs::read_dir(dataset_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name();
        if shard_day(&name.to_string_lossy()).is_some_and(|day| day < cutoff) {
            out.push(entry.path());
        }
    }
    let captures = dataset_dir.join(CAPTURES_DIR);
    if captures.is_dir() {
        for entry in std::fs::read_dir(&captures)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name();
            if NaiveDate::parse_from_str(&name.to_string_lossy(), DAY_FORMAT)
                .is_ok_and(|day| day < cutoff)
            {
                out.push(entry.path());
            }
        }
    }
    out.sort();
    Ok(out)
}

fn shard_day(file_name: &str) -> Option<NaiveDate> {
    let stem = file_name.strip_suffix(SHARD_SUFFIX)?;
    SHARD_PREFIXES
        .iter()
        .find_map(|prefix| stem.strip_prefix(prefix))
        .and_then(|day| NaiveDate::parse_from_str(day, DAY_FORMAT).ok())
}

/// Delete the selected entries (files or directory trees). Returns how many
/// were removed; stops at the first error so nothing is skipped silently.
pub fn delete_entries(entries: &[PathBuf]) -> std::io::Result<usize> {
    for path in entries {
        if path.is_dir() {
            std::fs::remove_dir_all(path)?;
        } else {
            std::fs::remove_file(path)?;
        }
    }
    Ok(entries.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn selects_only_entries_older_than_the_cutoff() {
        let tmp = tempfile::tempdir().unwrap();
        let ds = tmp.path();
        for name in [
            "attempts-20261001.jsonl", // old
            "labels-20261010.jsonl",   // exactly 14 days before 2026-10-24: kept
            "attempts-20261020.jsonl", // recent
            "attempts-notadate.jsonl", // unrecognised
            "notes.txt",               // unrecognised
        ] {
            std::fs::write(ds.join(name), b"x").unwrap();
        }
        std::fs::create_dir_all(ds.join("captures/20261002/pack")).unwrap();
        std::fs::create_dir_all(ds.join("captures/20261022/pack")).unwrap();
        std::fs::create_dir_all(ds.join("captures/junk")).unwrap();

        let got = purge_candidates(ds, day(2026, 10, 24), 14).unwrap();
        assert_eq!(
            got,
            vec![
                ds.join("attempts-20261001.jsonl"),
                ds.join("captures/20261002"),
            ]
        );
    }

    #[test]
    fn missing_dataset_dir_has_nothing_to_purge() {
        let tmp = tempfile::tempdir().unwrap();
        let got = purge_candidates(&tmp.path().join("absent"), day(2026, 10, 24), 14).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn delete_entries_removes_files_and_trees() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("attempts-20200101.jsonl");
        let dir = tmp.path().join("captures/20200101");
        std::fs::write(&file, b"x").unwrap();
        std::fs::create_dir_all(dir.join("pack")).unwrap();
        assert_eq!(delete_entries(&[file.clone(), dir.clone()]).unwrap(), 2);
        assert!(!file.exists() && !dir.exists());
    }
}
