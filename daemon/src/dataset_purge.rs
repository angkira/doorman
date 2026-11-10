//! Manual purge of old dataset files (`doorman dataset purge`).
//!
//! Dataset files are named after their UTC day: `attempts-YYYYMMDD.jsonl`,
//! `labels-YYYYMMDD.jsonl` and `captures/YYYYMMDD/`. A purge selects entries by
//! that date (not by mtime, which a backup restore would change). The daemon never
//! calls this on its own: deletion is always an explicit owner action.
//!
//! The purge runs as root over a directory the unprivileged `doorman` user can
//! write, so it never follows symlinks: a symlinked dataset or `captures`
//! directory is an error, symlinked entries are never selected, and deletion
//! re-checks that every entry is a real path inside the dataset directory.

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

/// Whether `path` is a real directory. A missing path is `Ok(false)`; a symlink
/// (even to a directory) is an error, because following it could point the
/// purge at files outside the dataset.
fn real_dir_exists(path: &Path) -> std::io::Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("refusing to follow symlink {}", path.display()),
        )),
        Ok(meta) => Ok(meta.is_dir()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// Entries of `dataset_dir` whose day is strictly older than `older_than_days`
/// days before `today`. Entries with unrecognised names are never selected.
pub fn purge_candidates(
    dataset_dir: &Path,
    today: NaiveDate,
    older_than_days: u32,
) -> std::io::Result<Vec<PathBuf>> {
    let cutoff = today - Duration::days(i64::from(older_than_days));
    let mut out = Vec::new();
    if !real_dir_exists(dataset_dir)? {
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
    if real_dir_exists(&captures)? {
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

/// Only the capture-pack day directories (`captures/YYYYMMDD`) strictly older
/// than `older_than_days` before `today`. Attempt and label shards are never
/// selected: this is the automatic retention of raw frames.
pub fn capture_candidates(
    dataset_dir: &Path,
    today: NaiveDate,
    older_than_days: u32,
) -> std::io::Result<Vec<PathBuf>> {
    Ok(purge_candidates(dataset_dir, today, older_than_days)?
        .into_iter()
        .filter(|p| p.is_dir() && p.parent().is_some_and(|d| d.ends_with(CAPTURES_DIR)))
        .collect())
}

/// Make the dataset, `captures` and every capture-day directory private
/// (0700). Real directories only (symlinks are skipped). Returns how many were
/// changed.
#[cfg(unix)]
pub fn enforce_private_dirs(dataset_dir: &Path) -> std::io::Result<usize> {
    use std::os::unix::fs::PermissionsExt;
    const PRIVATE: u32 = 0o700;
    let mut changed = 0;
    let mut fix = |dir: &Path| -> std::io::Result<()> {
        let meta = std::fs::symlink_metadata(dir)?;
        if meta.is_dir() && meta.permissions().mode() & 0o777 != PRIVATE {
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(PRIVATE))?;
            changed += 1;
        }
        Ok(())
    };
    if !real_dir_exists(dataset_dir)? {
        return Ok(0);
    }
    fix(dataset_dir)?;
    let captures = dataset_dir.join(CAPTURES_DIR);
    if real_dir_exists(&captures)? {
        fix(&captures)?;
        for entry in std::fs::read_dir(&captures)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                fix(&entry.path())?;
            }
        }
    }
    Ok(changed)
}

/// How automatic retention of raw frames is configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retention {
    /// `raw_frames_retention_days` unset: never delete automatically.
    Unconfigured,
    /// `raw_frames_retention_days = 0`: keep forever.
    KeepForever,
    /// Delete capture days older than this many days.
    Days(u32),
}

impl Retention {
    pub fn from_config(days: Option<u32>) -> Self {
        match days {
            None => Self::Unconfigured,
            Some(0) => Self::KeepForever,
            Some(n) => Self::Days(n),
        }
    }

    /// The INFO line logged at start.
    pub fn describe(self) -> String {
        match self {
            Self::Unconfigured => "raw frame retention: not configured (raw_frames_retention_days unset), nothing is deleted automatically".to_string(),
            Self::KeepForever => "raw frame retention: keep forever (raw_frames_retention_days = 0)".to_string(),
            Self::Days(n) => format!("raw frame retention: capture packs older than {n} days are deleted at start and daily"),
        }
    }
}

/// One retention pass: fix directory modes, then delete capture days older than
/// the configured retention. Does nothing to files when retention is not
/// `Days`. Returns the number of capture days removed.
pub fn run_retention(
    dataset_dir: &Path,
    retention: Retention,
    today: NaiveDate,
) -> std::io::Result<usize> {
    #[cfg(unix)]
    enforce_private_dirs(dataset_dir)?;
    let Retention::Days(days) = retention else {
        return Ok(0);
    };
    let old = capture_candidates(dataset_dir, today, days)?;
    delete_entries(dataset_dir, &old)
}

fn shard_day(file_name: &str) -> Option<NaiveDate> {
    let stem = file_name.strip_suffix(SHARD_SUFFIX)?;
    SHARD_PREFIXES
        .iter()
        .find_map(|prefix| stem.strip_prefix(prefix))
        .and_then(|day| NaiveDate::parse_from_str(day, DAY_FORMAT).ok())
}

/// Delete the selected entries (files or directory trees) of `dataset_dir`.
/// Returns how many were removed; stops at the first error so nothing is
/// skipped silently. Every entry must be a real file or directory (not a
/// symlink) whose canonical path lies inside `dataset_dir`.
pub fn delete_entries(dataset_dir: &Path, entries: &[PathBuf]) -> std::io::Result<usize> {
    let root = dataset_dir.canonicalize()?;
    for path in entries {
        let meta = std::fs::symlink_metadata(path)?;
        if meta.file_type().is_symlink() || !path.canonicalize()?.starts_with(&root) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "refusing to delete {}: symlink or outside {}",
                    path.display(),
                    dataset_dir.display()
                ),
            ));
        }
        if meta.is_dir() {
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
        assert_eq!(
            delete_entries(tmp.path(), &[file.clone(), dir.clone()]).unwrap(),
            2
        );
        assert!(!file.exists() && !dir.exists());
    }

    /// A directory outside the dataset with an old-looking day folder in it.
    fn victim(tmp: &Path) -> PathBuf {
        let victim = tmp.join("victim");
        std::fs::create_dir_all(victim.join("20200101")).unwrap();
        std::fs::write(victim.join("20200101/precious"), b"x").unwrap();
        victim
    }

    #[test]
    fn symlinked_captures_directory_is_refused_and_nothing_outside_is_touched() {
        let tmp = tempfile::tempdir().unwrap();
        let victim = victim(tmp.path());
        let ds = tmp.path().join("dataset");
        std::fs::create_dir_all(&ds).unwrap();
        std::os::unix::fs::symlink(&victim, ds.join("captures")).unwrap();

        let err = purge_candidates(&ds, day(2026, 10, 24), 14).unwrap_err();
        assert!(err.to_string().contains("symlink"), "{err}");
        assert!(victim.join("20200101/precious").exists());
    }

    #[test]
    fn symlinked_dataset_directory_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let victim = victim(tmp.path());
        std::fs::write(victim.join("attempts-20200101.jsonl"), b"x").unwrap();
        let ds = tmp.path().join("dataset");
        std::os::unix::fs::symlink(&victim, &ds).unwrap();

        assert!(purge_candidates(&ds, day(2026, 10, 24), 14).is_err());
        assert!(victim.join("attempts-20200101.jsonl").exists());
    }

    #[test]
    fn symlinked_entries_are_not_selected_and_delete_refuses_them() {
        let tmp = tempfile::tempdir().unwrap();
        let victim = victim(tmp.path());
        let ds = tmp.path().join("dataset");
        std::fs::create_dir_all(ds.join("captures")).unwrap();
        // An old-looking day folder that is really a link out of the dataset.
        let link = ds.join("captures/20200101");
        std::os::unix::fs::symlink(victim.join("20200101"), &link).unwrap();

        assert!(purge_candidates(&ds, day(2026, 10, 24), 14)
            .unwrap()
            .is_empty());
        // Even when handed the link directly, deletion refuses it.
        let err = delete_entries(&ds, &[link]).unwrap_err();
        assert!(err.to_string().contains("refusing"), "{err}");
        assert!(victim.join("20200101/precious").exists());
    }

    #[test]
    fn delete_refuses_paths_outside_the_dataset_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let victim = victim(tmp.path());
        let ds = tmp.path().join("dataset");
        std::fs::create_dir_all(&ds).unwrap();
        assert!(delete_entries(&ds, &[victim.join("20200101")]).is_err());
        assert!(victim.join("20200101/precious").exists());
    }

    #[test]
    fn retention_deletes_only_old_capture_days_and_only_when_configured() {
        let tmp = tempfile::tempdir().unwrap();
        let ds = tmp.path();
        for d in ["20261001", "20261020", "20261024"] {
            std::fs::create_dir_all(ds.join("captures").join(d).join("pack")).unwrap();
        }
        std::fs::write(ds.join("attempts-20261001.jsonl"), "x").unwrap();
        std::fs::write(ds.join("labels-20261001.jsonl"), "x").unwrap();
        let today = day(2026, 10, 24);

        // Unset and 0: nothing is deleted.
        for r in [
            Retention::from_config(None),
            Retention::from_config(Some(0)),
        ] {
            assert_eq!(run_retention(ds, r, today).unwrap(), 0);
            assert!(ds.join("captures/20261001").exists());
        }
        // 7 days: the 2026-10-01 pack goes, the others and the shards stay.
        assert_eq!(
            run_retention(ds, Retention::from_config(Some(7)), today).unwrap(),
            1
        );
        assert!(!ds.join("captures/20261001").exists());
        assert!(ds.join("captures/20261020").exists());
        assert!(ds.join("captures/20261024").exists());
        assert!(ds.join("attempts-20261001.jsonl").exists());
        assert!(ds.join("labels-20261001.jsonl").exists());
    }

    #[test]
    fn retention_state_is_described_for_the_start_up_log() {
        assert!(Retention::from_config(None)
            .describe()
            .contains("not configured"));
        assert!(Retention::from_config(Some(0))
            .describe()
            .contains("keep forever"));
        assert!(Retention::from_config(Some(7))
            .describe()
            .contains("7 days"));
    }

    #[cfg(unix)]
    #[test]
    fn dataset_directories_are_forced_to_0700() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let ds = tmp.path().join("dataset");
        let dayd = ds.join("captures").join("20261024");
        std::fs::create_dir_all(&dayd).unwrap();
        for d in [&ds, &ds.join("captures"), &dayd] {
            std::fs::set_permissions(d, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        assert_eq!(enforce_private_dirs(&ds).unwrap(), 3);
        for d in [&ds, &ds.join("captures"), &dayd] {
            let mode = std::fs::metadata(d).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o700, "{}", d.display());
        }
        assert_eq!(enforce_private_dirs(&ds).unwrap(), 0, "idempotent");
        assert_eq!(enforce_private_dirs(&tmp.path().join("none")).unwrap(), 0);
    }
}
