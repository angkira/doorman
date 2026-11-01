//! Dataset layout, image discovery and the path rules used by `record`.
//!
//! ```text
//! <root>/
//!   <user>/<condition>/NNNN.jpg      genuine captures (also used for enrolment)
//!   impostor/<identity>/NNNN.jpg     other people
//!   attacks/<type>/NNNN.jpg          presentation attacks (phone, screen, print)
//! ```
//!
//! For the directory mode of `accuracy`, `--enroll`, `--genuine` and
//! `--impostor` each point at a directory whose images are either directly in
//! it or one level down in `<condition>/` subdirectories (the condition is
//! the subdirectory name).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Image extensions the loader accepts (lowercase).
pub const IMAGE_EXTENSIONS: [&str; 3] = ["jpg", "jpeg", "png"];
/// Reserved top-level directory names under a dataset root.
pub const IMPOSTOR_DIR: &str = "impostor";
pub const ATTACKS_DIR: &str = "attacks";
/// Condition name for images found directly in a directory.
pub const NO_CONDITION: &str = "(none)";
/// Zero-padding width of recorded file names.
pub const FILE_INDEX_WIDTH: usize = 4;

#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("cannot read directory {path}: {source}")]
    ReadDir {
        path: String,
        source: std::io::Error,
    },
    #[error("no images (.jpg/.jpeg/.png) found under {0}")]
    NoImages(String),
    #[error("invalid name `{0}`: use only letters, digits, `_` and `-`")]
    BadName(String),
    #[error("`{0}` is reserved as a dataset directory name and cannot be used as a user")]
    ReservedName(String),
    #[error(
        "refusing to write inside a git repository ({repo}): recorded face data must stay out of the repo. Choose an output directory elsewhere"
    )]
    InsideGitRepo { repo: String },
    #[error("cannot resolve output directory {path}: {source}")]
    Resolve {
        path: String,
        source: std::io::Error,
    },
}

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn read_dir_sorted(dir: &Path) -> Result<Vec<PathBuf>, DatasetError> {
    let rd = std::fs::read_dir(dir).map_err(|source| DatasetError::ReadDir {
        path: dir.display().to_string(),
        source,
    })?;
    let mut v: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    v.sort();
    Ok(v)
}

/// All images under `dir`, recursively, in sorted order.
pub fn list_images(dir: &Path) -> Result<Vec<PathBuf>, DatasetError> {
    let mut out = Vec::new();
    for p in read_dir_sorted(dir)? {
        if p.is_dir() {
            out.extend(list_images(&p)?);
        } else if is_image(&p) {
            out.push(p);
        }
    }
    Ok(out)
}

/// Images grouped by condition: files directly in `dir` -> [`NO_CONDITION`],
/// files below `dir/<c>/` -> `c`.
pub fn list_images_by_condition(
    dir: &Path,
) -> Result<BTreeMap<String, Vec<PathBuf>>, DatasetError> {
    let mut out: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for p in read_dir_sorted(dir)? {
        if p.is_dir() {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let imgs = list_images(&p)?;
            if !imgs.is_empty() {
                out.entry(name).or_default().extend(imgs);
            }
        } else if is_image(&p) {
            out.entry(NO_CONDITION.to_string()).or_default().push(p);
        }
    }
    if out.is_empty() {
        return Err(DatasetError::NoImages(dir.display().to_string()));
    }
    Ok(out)
}

fn valid_name(n: &str) -> Result<(), DatasetError> {
    if !n.is_empty()
        && n.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        Ok(())
    } else {
        Err(DatasetError::BadName(n.to_string()))
    }
}

/// What a recording session captures.
#[derive(Debug, Clone, PartialEq)]
pub enum RecordTarget {
    Genuine { user: String, condition: String },
    Impostor { identity: String },
    Attack { kind: String },
}

impl RecordTarget {
    /// Directory (under `root`) that receives the frames. Validates names.
    pub fn dir(&self, root: &Path) -> Result<PathBuf, DatasetError> {
        match self {
            RecordTarget::Genuine { user, condition } => {
                valid_name(user)?;
                valid_name(condition)?;
                if user == IMPOSTOR_DIR || user == ATTACKS_DIR {
                    return Err(DatasetError::ReservedName(user.clone()));
                }
                Ok(root.join(user).join(condition))
            }
            RecordTarget::Impostor { identity } => {
                valid_name(identity)?;
                Ok(root.join(IMPOSTOR_DIR).join(identity))
            }
            RecordTarget::Attack { kind } => {
                valid_name(kind)?;
                Ok(root.join(ATTACKS_DIR).join(kind))
            }
        }
    }
}

/// File name of the `index`-th frame (0-based index, 1-based file number).
pub fn frame_file_name(index: usize) -> String {
    format!("{:0width$}.jpg", index + 1, width = FILE_INDEX_WIDTH)
}

/// Next free file index in `dir` so that repeated sessions append instead of
/// overwriting. Returns 0 when the directory does not exist yet.
pub fn next_index(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| is_image(&e.path()))
                .count()
        })
        .unwrap_or(0)
}

/// Refuse output directories inside a git work tree.
///
/// The nearest existing ancestor of `out` is canonicalised and walked upwards;
/// any ancestor containing `.git` (directory or file, to cover worktrees)
/// rejects the path.
pub fn ensure_outside_git_repo(out: &Path) -> Result<(), DatasetError> {
    let mut probe = if out.is_absolute() {
        out.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|source| DatasetError::Resolve {
                path: out.display().to_string(),
                source,
            })?
            .join(out)
    };
    while !probe.exists() {
        if !probe.pop() {
            break;
        }
    }
    let canon = probe
        .canonicalize()
        .map_err(|source| DatasetError::Resolve {
            path: probe.display().to_string(),
            source,
        })?;
    for anc in canon.ancestors() {
        if anc.join(".git").exists() {
            return Err(DatasetError::InsideGitRepo {
                repo: anc.display().to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_target_layout() {
        let root = Path::new("/data");
        let g = RecordTarget::Genuine {
            user: "alice".into(),
            condition: "evening".into(),
        };
        assert_eq!(g.dir(root).unwrap(), Path::new("/data/alice/evening"));
        let i = RecordTarget::Impostor {
            identity: "p01".into(),
        };
        assert_eq!(i.dir(root).unwrap(), Path::new("/data/impostor/p01"));
        let a = RecordTarget::Attack {
            kind: "phone_replay".into(),
        };
        assert_eq!(
            a.dir(root).unwrap(),
            Path::new("/data/attacks/phone_replay")
        );
    }

    #[test]
    fn rejects_unsafe_and_reserved_names() {
        let root = Path::new("/data");
        for bad in ["", "..", "a/b", "a b", "../x"] {
            let t = RecordTarget::Attack { kind: bad.into() };
            assert!(
                matches!(t.dir(root), Err(DatasetError::BadName(_))),
                "{bad}"
            );
        }
        let t = RecordTarget::Genuine {
            user: "attacks".into(),
            condition: "day".into(),
        };
        assert!(matches!(t.dir(root), Err(DatasetError::ReservedName(_))));
    }

    #[test]
    fn frame_names_are_padded_and_one_based() {
        assert_eq!(frame_file_name(0), "0001.jpg");
        assert_eq!(frame_file_name(41), "0042.jpg");
    }

    #[test]
    fn refuses_git_repo_but_allows_plain_dir() {
        let plain = tempfile::tempdir().unwrap();
        ensure_outside_git_repo(&plain.path().join("new/sub")).unwrap();

        let repo = tempfile::tempdir().unwrap();
        std::fs::create_dir(repo.path().join(".git")).unwrap();
        let err = ensure_outside_git_repo(&repo.path().join("data/run1")).unwrap_err();
        assert!(matches!(err, DatasetError::InsideGitRepo { .. }));
        // A `.git` *file* (git worktree) counts too.
        let wt = tempfile::tempdir().unwrap();
        std::fs::write(wt.path().join(".git"), "gitdir: elsewhere").unwrap();
        assert!(ensure_outside_git_repo(wt.path()).is_err());
    }

    #[test]
    fn lists_images_by_condition() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir(d.path().join("day")).unwrap();
        std::fs::write(d.path().join("day/0001.jpg"), b"x").unwrap();
        std::fs::write(d.path().join("top.PNG"), b"x").unwrap();
        std::fs::write(d.path().join("notes.txt"), b"x").unwrap();
        let m = list_images_by_condition(d.path()).unwrap();
        assert_eq!(m.keys().cloned().collect::<Vec<_>>(), vec!["(none)", "day"]);
        let empty = tempfile::tempdir().unwrap();
        assert!(matches!(
            list_images_by_condition(empty.path()),
            Err(DatasetError::NoImages(_))
        ));
        assert_eq!(next_index(&d.path().join("day")), 1);
        assert_eq!(next_index(&d.path().join("missing")), 0);
    }
}
