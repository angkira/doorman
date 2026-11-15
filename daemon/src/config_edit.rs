//! `doorman config get | set | unset`: validated, comment-preserving edits of
//! the daemon config.
//!
//! * The key must exist in the real schema (`config_check::known_keys`, derived
//!   from `Config::default()`), and the edited text must pass the same
//!   parse + `Config::validate` as the daemon's loader before anything is
//!   written. An unknown key is refused with did-you-mean suggestions, a wrong
//!   type or range with the loader's own error.
//! * Edits go through `toml_edit`, which keeps comments, key order and spacing
//!   of everything it does not touch (the `toml` crate would rewrite the file).
//! * Writing is atomic: backup (`<file>.bak.<utc timestamp>`, mode 0600, the
//!   newest [`MAX_BACKUPS`] kept), temporary file in the same directory,
//!   fsync, rename, fsync of the directory. Ownership and mode of the existing
//!   file are kept (root:root 0644 for the system file).
//! * A root process refuses a file that fails the config trust check (not
//!   root-owned, or group/other-writable file or directory).
//!
//! Nothing here restarts anything; the CLI prints [`RESTART_HINT`].

use doorman_shared::config_check::{self, check_str};
use doorman_shared::config_trust::{self, MetadataProvider};
use std::fmt::Write as _;
use std::io::Write;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use toml_edit::{DocumentMut, Item, Value};

/// Printed after a successful `set` / `unset`.
pub const RESTART_HINT: &str = "restart needed: sudo systemctl restart doormand";
/// Backups kept next to the config file.
pub const MAX_BACKUPS: usize = 10;
/// Mode of the backup files (they hold the old config).
const BACKUP_MODE: u32 = 0o600;
/// Mode of the temporary file until it is renamed over the config.
const TEMP_MODE: u32 = 0o600;
/// Suggestions offered for an unknown key.
const MAX_SUGGESTIONS: usize = 3;
/// Context lines around a changed region in the printed diff.
const DIFF_CONTEXT: usize = 2;

#[derive(Debug, thiserror::Error)]
pub enum EditError {
    #[error("unknown key `{key}`{}", fmt_suggestions(.suggestions))]
    UnknownKey {
        key: String,
        suggestions: Vec<String>,
    },
    #[error("`{key}` is a section, not a value; its keys: {}", .children.join(", "))]
    Section { key: String, children: Vec<String> },
    #[error("`{key}` is deprecated and ignored: {hint}")]
    Deprecated { key: String, hint: String },
    #[error("cannot read the value for `{key}`: {why}")]
    BadValue { key: String, why: String },
    #[error("{0}")]
    Invalid(String),
    #[error("cannot parse the config file: {0}")]
    Parse(String),
    #[error("{0}")]
    Trust(String),
    #[error("{0}")]
    Io(String),
}

fn fmt_suggestions(s: &[String]) -> String {
    if s.is_empty() {
        String::new()
    } else {
        format!(" (did you mean: {}?)", s.join(", "))
    }
}

fn io_err(what: &str, e: std::io::Error) -> EditError {
    EditError::Io(format!("{what}: {e}"))
}

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

fn levenshtein(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur.push((prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Known leaf keys close to `key` (whole path, or just the last segment).
pub fn suggest(key: &str) -> Vec<String> {
    let last = key.rsplit('.').next().unwrap_or(key);
    let mut scored: Vec<(usize, String)> = config_check::known_keys()
        .into_iter()
        .filter(|(_, is_table)| !is_table)
        .filter_map(|(candidate, _)| {
            let cand_last = candidate.rsplit('.').next().unwrap_or(&candidate);
            let whole = levenshtein(key, &candidate);
            let tail = levenshtein(last, cand_last);
            let limit = (key.len() / 3).max(3);
            let near_tail = tail <= (last.len() / 3).max(2);
            let best = if near_tail {
                whole.min(tail + 1)
            } else {
                whole
            };
            (best <= limit || near_tail).then_some((best, candidate))
        })
        .collect();
    scored.sort();
    scored
        .into_iter()
        .take(MAX_SUGGESTIONS)
        .map(|(_, k)| k)
        .collect()
}

/// Validate that `key` names a settable leaf of the schema.
fn check_leaf(key: &str) -> Result<serde_json::Value, EditError> {
    if let Some((_, hint)) = config_check::DEPRECATED_KEYS
        .iter()
        .find(|(old, _)| *old == key)
    {
        return Err(EditError::Deprecated {
            key: key.to_string(),
            hint: (*hint).to_string(),
        });
    }
    let known = config_check::known_keys();
    match known.get(key) {
        None => Err(EditError::UnknownKey {
            key: key.to_string(),
            suggestions: suggest(key),
        }),
        Some(true) => {
            let prefix = format!("{key}.");
            Err(EditError::Section {
                key: key.to_string(),
                children: known
                    .keys()
                    .filter(|k| k.starts_with(&prefix))
                    .cloned()
                    .collect(),
            })
        }
        Some(false) => Ok(config_check::default_value(key).unwrap_or(serde_json::Value::Null)),
    }
}

/// The value as written, without the comment or spacing around it.
fn repr(v: &Value) -> String {
    let mut bare = v.clone();
    *bare.decor_mut() = toml_edit::Decor::default();
    bare.to_string()
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

fn parse_value(key: &str, raw: &str, default: &serde_json::Value) -> Result<Value, EditError> {
    use serde_json::Value as J;
    let as_string = || Value::from(raw);
    let parsed: Result<Value, _> = raw.trim().parse::<Value>();
    match (parsed, default) {
        (Ok(v), J::String(_)) if !v.is_str() => Ok(as_string()),
        (Err(_), J::String(_) | J::Null) => Ok(as_string()),
        (Ok(v), J::Number(n)) if n.is_f64() => match v.as_integer() {
            Some(i) => Ok(Value::from(i as f64)),
            None => Ok(v),
        },
        (Ok(v), _) => Ok(v),
        (Err(e), _) => Err(EditError::BadValue {
            key: key.to_string(),
            why: format!(
                "`{raw}` is not a TOML value ({}); quote strings and write lists as [1, 2]",
                e.message()
            ),
        }),
    }
}

/// Navigate to (creating) the table holding the last segment of `key`.
fn parent_table<'a>(
    doc: &'a mut DocumentMut,
    segments: &[&str],
) -> Result<&'a mut dyn toml_edit::TableLike, EditError> {
    let mut table: &mut dyn toml_edit::TableLike = doc.as_table_mut();
    for seg in segments {
        if !table.contains_key(seg) {
            table.insert(seg, Item::Table(toml_edit::Table::new()));
        }
        table = table
            .get_mut(seg)
            .and_then(Item::as_table_like_mut)
            .ok_or_else(|| {
                EditError::Invalid(format!("`{seg}` exists in the file but is not a table"))
            })?;
    }
    Ok(table)
}

// ---------------------------------------------------------------------------
// get
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    File,
    Default,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub key: String,
    pub value: String,
    pub source: Source,
}

fn render_default(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "unset".to_string(),
        other => other.to_string(),
    }
}

/// The effective value of `key` (a leaf), or of every leaf below it (a
/// section): the file's value when set, else the built-in default.
pub fn get(content: &str, key: &str) -> Result<Vec<Entry>, EditError> {
    let known = config_check::known_keys();
    let leaves: Vec<String> = match known.get(key) {
        None => {
            return Err(EditError::UnknownKey {
                key: key.to_string(),
                suggestions: suggest(key),
            })
        }
        Some(false) => vec![key.to_string()],
        Some(true) => {
            let prefix = format!("{key}.");
            known
                .iter()
                .filter(|(k, is_table)| k.starts_with(&prefix) && !**is_table)
                .map(|(k, _)| k.clone())
                .collect()
        }
    };
    let doc: DocumentMut = content
        .parse()
        .map_err(|e: toml_edit::TomlError| EditError::Parse(e.to_string()))?;
    Ok(leaves
        .into_iter()
        .map(|leaf| {
            let mut item: Option<&Item> = Some(doc.as_item());
            for seg in leaf.split('.') {
                item = item.and_then(|i| i.get(seg));
            }
            match item {
                Some(Item::Value(v)) => Entry {
                    value: repr(v),
                    key: leaf,
                    source: Source::File,
                },
                _ => Entry {
                    value: render_default(&config_check::default_value(&leaf).unwrap_or_default()),
                    key: leaf,
                    source: Source::Default,
                },
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// set / unset
// ---------------------------------------------------------------------------

/// The result of planning an edit; nothing is written yet.
#[derive(Debug, Clone)]
pub struct Plan {
    pub new_content: String,
    pub changed: bool,
    /// The previous value as written in the file; `None` when the key was unset.
    pub before: Option<String>,
    pub after: Option<String>,
    pub diff: String,
}

fn validate(new_content: &str) -> Result<(), EditError> {
    check_str(new_content).map(|_| ()).map_err(|e| {
        EditError::Invalid(format!(
            "refused, the result would not be a valid config: {e}"
        ))
    })
}

/// Set `key` to `raw` (TOML syntax; a bare word is taken as a string for
/// string keys). The edited text is validated before it is returned.
pub fn plan_set(content: &str, key: &str, raw: &str) -> Result<Plan, EditError> {
    let default = check_leaf(key)?;
    let new_value = parse_value(key, raw, &default)?;
    let mut doc: DocumentMut = content
        .parse()
        .map_err(|e: toml_edit::TomlError| EditError::Parse(e.to_string()))?;
    let segments: Vec<&str> = key.split('.').collect();
    let (last, parents) = segments.split_last().expect("a key has a segment");
    let table = parent_table(&mut doc, parents)?;
    let before = table.get(last).and_then(Item::as_value).map(repr);
    let mut new_value = new_value;
    match table.get_mut(last) {
        Some(existing) => {
            // Keep the key's own decor (comments above it) and the trailing
            // comment of the old value.
            if let Some(old) = existing.as_value() {
                *new_value.decor_mut() = old.decor().clone();
            }
            *existing = Item::Value(new_value.clone());
        }
        None => {
            table.insert(last, Item::Value(new_value.clone()));
        }
    }
    let after = Some(repr(&new_value));
    let new_content = doc.to_string();
    validate(&new_content)?;
    Ok(Plan {
        changed: new_content != content,
        diff: diff(content, &new_content),
        new_content,
        before,
        after,
    })
}

/// Remove `key` from the file so the built-in default applies again.
pub fn plan_unset(content: &str, key: &str) -> Result<Plan, EditError> {
    check_leaf(key)?;
    let mut doc: DocumentMut = content
        .parse()
        .map_err(|e: toml_edit::TomlError| EditError::Parse(e.to_string()))?;
    let segments: Vec<&str> = key.split('.').collect();
    let (last, parents) = segments.split_last().expect("a key has a segment");
    // Do not create tables just to find nothing in them.
    let mut probe: Option<&Item> = Some(doc.as_item());
    for seg in parents {
        probe = probe.and_then(|i| i.get(seg));
    }
    if probe.and_then(|i| i.get(last)).is_none() {
        return Ok(Plan {
            new_content: content.to_string(),
            changed: false,
            before: None,
            after: None,
            diff: String::new(),
        });
    }
    let table = parent_table(&mut doc, parents)?;
    let before = table
        .remove(last)
        .and_then(|i| i.into_value().ok())
        .map(|v| repr(&v));
    let new_content = doc.to_string();
    validate(&new_content)?;
    Ok(Plan {
        changed: true,
        diff: diff(content, &new_content),
        new_content,
        before,
        after: None,
    })
}

/// Minimal unified-style diff: the changed middle between the common leading
/// and trailing lines, with [`DIFF_CONTEXT`] lines of context. Empty when equal.
pub fn diff(old: &str, new: &str) -> String {
    if old == new {
        return String::new();
    }
    let (o, n): (Vec<&str>, Vec<&str>) = (old.lines().collect(), new.lines().collect());
    let prefix = o.iter().zip(&n).take_while(|(a, b)| a == b).count();
    let max_suffix = o.len().min(n.len()) - prefix;
    let suffix = o
        .iter()
        .rev()
        .zip(n.iter().rev())
        .take_while(|(a, b)| a == b)
        .count()
        .min(max_suffix);
    let (o_mid, n_mid) = (&o[prefix..o.len() - suffix], &n[prefix..n.len() - suffix]);
    let ctx_start = prefix.saturating_sub(DIFF_CONTEXT);
    let mut out = format!("@@ line {} @@\n", ctx_start + 1);
    for line in &o[ctx_start..prefix] {
        let _ = writeln!(out, " {line}");
    }
    for line in o_mid {
        let _ = writeln!(out, "-{line}");
    }
    for line in n_mid {
        let _ = writeln!(out, "+{line}");
    }
    let tail_end = (o.len() - suffix + DIFF_CONTEXT).min(o.len());
    for line in &o[o.len() - suffix..tail_end] {
        let _ = writeln!(out, " {line}");
    }
    out
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// What `apply` did.
#[derive(Debug)]
pub struct Applied {
    pub backup: PathBuf,
}

fn backup_prefix(path: &Path) -> String {
    format!(
        "{}.bak.",
        path.file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default()
    )
}

/// Delete all but the newest `keep` backups of `path`.
fn prune_backups(path: &Path, keep: usize) {
    let Some(dir) = path.parent() else { return };
    let prefix = backup_prefix(path);
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    let mut names: Vec<String> = read
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| {
            n.strip_prefix(&prefix)
                .is_some_and(|ts| !ts.is_empty() && ts.chars().all(|c| c.is_ascii_alphanumeric()))
        })
        .collect();
    names.sort();
    let excess = names.len().saturating_sub(keep);
    for name in names.into_iter().take(excess) {
        let _ = std::fs::remove_file(dir.join(name));
    }
}

fn write_new_file(path: &Path, mode: u32, bytes: &[u8]) -> Result<(), EditError> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(mode)
        .open(path)
        .map_err(|e| io_err(&format!("cannot create {}", path.display()), e))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| io_err(&format!("cannot write {}", path.display()), e))
}

/// Replace the config at `path` with `new_content` atomically.
///
/// `expected_old` is the text the plan was made from: if the file changed in
/// between, nothing is written. `timestamp` names the backup
/// (`<file>.bak.<timestamp>`). With `enforce_trust` the config trust check
/// (root-owned, not group/other-writable, file and directory) must pass.
pub fn apply(
    path: &Path,
    expected_old: &str,
    new_content: &str,
    timestamp: &str,
    enforce_trust: Option<&dyn MetadataProvider>,
) -> Result<Applied, EditError> {
    if let Some(meta) = enforce_trust {
        config_trust::check_trusted(path, meta).map_err(|e| EditError::Trust(e.to_string()))?;
    }
    validate(new_content)?;
    let current = std::fs::read_to_string(path)
        .map_err(|e| io_err(&format!("cannot read {}", path.display()), e))?;
    if current != expected_old {
        return Err(EditError::Io(format!(
            "{} changed while it was being edited; nothing written, run the command again",
            path.display()
        )));
    }
    let md = std::fs::metadata(path).map_err(|e| io_err("cannot stat the config", e))?;
    let dir = path
        .parent()
        .ok_or_else(|| EditError::Io("config path has no directory".into()))?;

    let backup = dir.join(format!("{}{timestamp}", backup_prefix(path)));
    write_new_file(&backup, BACKUP_MODE, current.as_bytes())?;

    let tmp = dir.join(format!(
        ".{}.tmp.{}",
        path.file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default(),
        std::process::id()
    ));
    let finish = || -> Result<(), EditError> {
        write_new_file(&tmp, TEMP_MODE, new_content.as_bytes())?;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(md.mode() & 0o7777))
            .map_err(|e| io_err("cannot set the mode", e))?;
        std::os::unix::fs::chown(&tmp, Some(md.uid()), Some(md.gid()))
            .map_err(|e| io_err("cannot keep the owner of the config", e))?;
        std::fs::rename(&tmp, path).map_err(|e| io_err("cannot replace the config", e))?;
        std::fs::File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(|e| io_err("cannot sync the config directory", e))
    };
    if let Err(e) = finish() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    prune_backups(path, MAX_BACKUPS);

    // Read it back: what is on disk is what was validated.
    let written =
        std::fs::read_to_string(path).map_err(|e| io_err("cannot read back the config", e))?;
    if written != new_content {
        return Err(EditError::Io(
            "the config on disk differs from what was written".into(),
        ));
    }
    validate(&written)?;
    Ok(Applied { backup })
}

#[cfg(test)]
mod tests {
    use super::*;
    use doorman_shared::config_trust::FileFacts;
    use std::collections::HashMap;

    const SAMPLE: &str = "\
# doorman config (hand written)
schema_version = 1

[ipc]
# why: lab machine
face_max_judged_failures = 5   # five tries
face_cooldown_secs = [10, 30, 120]

[camera]
width = 640
";

    fn value_of(content: &str, key: &str) -> String {
        get(content, key).unwrap().remove(0).value
    }

    #[test]
    fn set_changes_one_value_and_keeps_every_comment_and_the_layout() {
        let plan = plan_set(SAMPLE, "ipc.face_max_judged_failures", "7").unwrap();
        assert!(plan.changed);
        assert_eq!(
            plan.new_content,
            SAMPLE.replace(
                "face_max_judged_failures = 5   # five tries",
                "face_max_judged_failures = 7   # five tries"
            )
        );
        assert_eq!(plan.before.as_deref(), Some("5"));
        assert_eq!(plan.after.as_deref(), Some("7"));
        assert!(
            plan.diff
                .contains("-face_max_judged_failures = 5   # five tries"),
            "{}",
            plan.diff
        );
        assert!(plan
            .diff
            .contains("+face_max_judged_failures = 7   # five tries"));
    }

    #[test]
    fn arrays_round_trip() {
        let plan = plan_set(SAMPLE, "ipc.face_cooldown_secs", "[5, 20, 60, 300]").unwrap();
        assert_eq!(
            value_of(&plan.new_content, "ipc.face_cooldown_secs"),
            "[5, 20, 60, 300]"
        );
        assert!(plan.new_content.contains("# why: lab machine"));
        let cfg = check_str(&plan.new_content).unwrap().config;
        assert_eq!(cfg.ipc.face_cooldown_secs, vec![5, 20, 60, 300]);
    }

    #[test]
    fn a_new_key_in_an_existing_table_and_a_new_table_are_added_without_touching_the_rest() {
        let a = plan_set(SAMPLE, "camera.max_frame_age_ms", "500").unwrap();
        assert!(a.new_content.starts_with(SAMPLE.trim_end_matches('\n')));
        assert_eq!(value_of(&a.new_content, "camera.max_frame_age_ms"), "500");
        let b = plan_set(SAMPLE, "calibration.default_ttl_secs", "1800").unwrap();
        assert!(b.new_content.contains("[calibration]"));
        assert!(b.new_content.contains("# five tries"));
        assert_eq!(
            check_str(&b.new_content)
                .unwrap()
                .config
                .calibration
                .default_ttl_secs,
            1800
        );
    }

    #[test]
    fn set_works_on_an_empty_file_and_for_strings_booleans_and_floats() {
        let p = plan_set("", "camera.fourcc", "MJPG").unwrap();
        assert_eq!(value_of(&p.new_content, "camera.fourcc"), "\"MJPG\"");
        let p = plan_set("", "liveness.shadow_pad_logging", "true").unwrap();
        assert_eq!(
            value_of(&p.new_content, "liveness.shadow_pad_logging"),
            "true"
        );
        // An integer for a float key becomes a float.
        let p = plan_set("", "authentication.similarity_threshold", "1").unwrap();
        assert_eq!(
            value_of(&p.new_content, "authentication.similarity_threshold"),
            "1.0"
        );
        let p = plan_set("", "authentication.similarity_threshold", "0.1").unwrap_err();
        assert!(p.to_string().contains("frame_identity_floor"), "{p}");
        let p = plan_set("", "authentication.similarity_threshold", "0.45").unwrap();
        assert_eq!(
            value_of(&p.new_content, "authentication.similarity_threshold"),
            "0.45"
        );
    }

    #[test]
    fn unknown_keys_are_refused_with_did_you_mean() {
        let e = plan_set(SAMPLE, "ipc.face_max_judged_failure", "3").unwrap_err();
        match &e {
            EditError::UnknownKey { suggestions, .. } => {
                assert_eq!(
                    suggestions.first().map(String::as_str),
                    Some("ipc.face_max_judged_failures"),
                    "{e}"
                );
            }
            other => panic!("{other}"),
        }
        assert!(e.to_string().contains("did you mean"));
        let e = plan_set(SAMPLE, "totally.bogus", "1").unwrap_err();
        assert!(matches!(e, EditError::UnknownKey { .. }));
        // Wrong section for a real key name.
        let e = plan_set(SAMPLE, "camera.face_max_judged_failures", "3").unwrap_err();
        assert!(
            e.to_string().contains("ipc.face_max_judged_failures"),
            "{e}"
        );
    }

    #[test]
    fn a_section_is_not_a_value() {
        let e = plan_set(SAMPLE, "ipc", "1").unwrap_err();
        assert!(matches!(e, EditError::Section { .. }));
        assert!(e.to_string().contains("ipc.face_cooldown_secs"));
    }

    #[test]
    fn wrong_type_or_range_is_an_explicit_error_and_writes_nothing() {
        for (key, value) in [
            ("ipc.face_max_judged_failures", "\"many\""),
            ("ipc.face_max_judged_failures", "101"),
            ("ipc.face_max_judged_failures", "-1"),
            ("ipc.face_cooldown_secs", "30"),
            ("liveness.shadow_pad_logging", "maybe"),
            ("calibration.max_ttl_secs", "999999"),
            ("calibration.default_ttl_secs", "1"),
        ] {
            let e = plan_set(SAMPLE, key, value).unwrap_err();
            assert!(
                matches!(e, EditError::Invalid(_) | EditError::BadValue { .. }),
                "{key}={value}: {e}"
            );
            assert!(!e.to_string().is_empty());
        }
    }

    #[test]
    fn unset_removes_the_key_and_restores_the_default() {
        let plan = plan_unset(SAMPLE, "ipc.face_cooldown_secs").unwrap();
        assert!(plan.changed);
        assert!(!plan.new_content.contains("face_cooldown_secs"));
        assert!(plan.new_content.contains("# why: lab machine"));
        let entry = get(&plan.new_content, "ipc.face_cooldown_secs")
            .unwrap()
            .remove(0);
        assert_eq!(entry.source, Source::Default);
        assert_eq!(plan.before.as_deref(), Some("[10, 30, 120]"));
        // Unsetting something that is not there changes nothing and creates nothing.
        let none = plan_unset(SAMPLE, "calibration.max_ttl_secs").unwrap();
        assert!(!none.changed);
        assert_eq!(none.new_content, SAMPLE);
    }

    #[test]
    fn get_reports_file_values_defaults_and_sections() {
        let one = get(SAMPLE, "ipc.face_cooldown_secs").unwrap();
        assert_eq!(
            one,
            vec![Entry {
                key: "ipc.face_cooldown_secs".into(),
                value: "[10, 30, 120]".into(),
                source: Source::File
            }]
        );
        let d = get(SAMPLE, "calibration.max_ttl_secs").unwrap().remove(0);
        assert_eq!((d.value.as_str(), d.source), ("7200", Source::Default));
        let section = get(SAMPLE, "calibration").unwrap();
        assert_eq!(section.len(), 3);
        assert!(get(SAMPLE, "nope.nope").is_err());
        let unset = get("", "ipc.stream_socket_group").unwrap().remove(0);
        assert_eq!(unset.value, "unset");
    }

    #[test]
    fn deprecated_keys_are_refused() {
        let e = plan_set(SAMPLE, "dataset.raw_face_frame_bundle_size", "5").unwrap_err();
        assert!(matches!(e, EditError::Deprecated { .. }), "{e}");
    }

    #[test]
    fn diff_shows_context_and_is_empty_when_equal() {
        assert_eq!(diff("a\n", "a\n"), "");
        let d = diff("1\n2\n3\n4\n5\n6\n", "1\n2\n3\nX\n5\n6\n");
        assert_eq!(d, "@@ line 2 @@\n 2\n 3\n-4\n+X\n 5\n 6\n");
        let added = diff("a\nb\n", "a\nb\nc\n");
        assert!(added.contains("+c"));
    }

    // --- writing ---

    struct FakeMeta(HashMap<PathBuf, FileFacts>);
    impl MetadataProvider for FakeMeta {
        fn canonicalize(&self, path: &Path) -> std::io::Result<PathBuf> {
            Ok(path.to_path_buf())
        }
        fn facts(&self, path: &Path) -> std::io::Result<FileFacts> {
            self.0
                .get(path)
                .copied()
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
        }
    }

    fn trusted(file: &Path) -> FakeMeta {
        let ok = FileFacts {
            uid: 0,
            mode: 0o644,
        };
        FakeMeta(HashMap::from([
            (file.to_path_buf(), ok),
            (
                file.parent().unwrap().to_path_buf(),
                FileFacts {
                    uid: 0,
                    mode: 0o755,
                },
            ),
        ]))
    }

    fn setup() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("doorman.toml");
        std::fs::write(&path, SAMPLE).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        (dir, path)
    }

    #[test]
    fn apply_writes_atomically_keeps_mode_makes_a_private_backup_and_leaves_no_temp_file() {
        let (dir, path) = setup();
        let plan = plan_set(SAMPLE, "ipc.face_max_judged_failures", "0").unwrap();
        let applied = apply(
            &path,
            SAMPLE,
            &plan.new_content,
            "20260101T000000Z",
            Some(&trusted(&path)),
        )
        .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), plan.new_content);
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o644
        );
        assert_eq!(std::fs::read_to_string(&applied.backup).unwrap(), SAMPLE);
        assert_eq!(
            std::fs::metadata(&applied.backup)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        assert!(applied
            .backup
            .ends_with("doorman.toml.bak.20260101T000000Z"));
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp."))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn apply_refuses_an_untrusted_file_and_writes_nothing() {
        let (dir, path) = setup();
        let plan = plan_set(SAMPLE, "ipc.face_max_judged_failures", "0").unwrap();
        for facts in [
            FileFacts {
                uid: 1000,
                mode: 0o644,
            },
            FileFacts {
                uid: 0,
                mode: 0o666,
            },
            FileFacts {
                uid: 0,
                mode: 0o664,
            },
        ] {
            let mut meta = trusted(&path);
            meta.0.insert(path.clone(), facts);
            let e = apply(&path, SAMPLE, &plan.new_content, "T", Some(&meta)).unwrap_err();
            assert!(matches!(e, EditError::Trust(_)), "{facts:?}: {e}");
        }
        // A group-writable directory is refused too.
        let mut meta = trusted(&path);
        meta.0.insert(
            dir.path().to_path_buf(),
            FileFacts {
                uid: 0,
                mode: 0o775,
            },
        );
        assert!(matches!(
            apply(&path, SAMPLE, &plan.new_content, "T", Some(&meta)).unwrap_err(),
            EditError::Trust(_)
        ));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), SAMPLE, "untouched");
        assert!(
            std::fs::read_dir(dir.path()).unwrap().count() == 1,
            "no backup, no temp"
        );
    }

    #[test]
    fn apply_refuses_when_the_file_changed_meanwhile_and_when_the_result_is_invalid() {
        let (_dir, path) = setup();
        let plan = plan_set(SAMPLE, "ipc.face_max_judged_failures", "0").unwrap();
        std::fs::write(&path, "# someone else edited\n").unwrap();
        let e = apply(&path, SAMPLE, &plan.new_content, "T", None).unwrap_err();
        assert!(e.to_string().contains("changed while"), "{e}");
        let e = apply(
            &path,
            "# someone else edited\n",
            "[ipc]\nface_max_judged_failures = 999\n",
            "T",
            None,
        )
        .unwrap_err();
        assert!(matches!(e, EditError::Invalid(_)));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "# someone else edited\n"
        );
    }

    #[test]
    fn only_the_newest_backups_are_kept() {
        let (dir, path) = setup();
        let mut content = SAMPLE.to_string();
        for i in 0..(MAX_BACKUPS + 4) {
            let plan = plan_set(&content, "camera.width", &(320 + i).to_string()).unwrap();
            apply(
                &path,
                &content,
                &plan.new_content,
                &format!("20260101T0000{i:02}Z"),
                None,
            )
            .unwrap();
            content = plan.new_content;
        }
        let mut backups: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".bak."))
            .collect();
        backups.sort();
        assert_eq!(backups.len(), MAX_BACKUPS);
        assert!(
            backups.first().unwrap().ends_with("T000004Z"),
            "{backups:?}"
        );
        assert!(backups.last().unwrap().ends_with("T000013Z"));
    }

    #[test]
    fn the_edited_file_still_loads_with_the_daemon_loader() {
        let (_dir, path) = setup();
        let plan = plan_set(SAMPLE, "calibration.default_ttl_secs", "1500").unwrap();
        apply(&path, SAMPLE, &plan.new_content, "T", None).unwrap();
        let cfg = doorman_shared::Config::load_from(&path).unwrap();
        assert_eq!(cfg.calibration.default_ttl_secs, 1500);
        assert_eq!(cfg.ipc.face_cooldown_secs, vec![10, 30, 120]);
    }
}
