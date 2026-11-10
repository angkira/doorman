//! Trust check for configuration files read by a privileged process.
//!
//! A process running as root must not take security-relevant settings (for
//! example `daemon.data_dir`, which `doorman dataset purge` deletes under)
//! from a file an unprivileged user can create or edit. [`check_trusted`]
//! decides this from file metadata only; the metadata source is injectable so
//! the decision can be tested without owning files as root.

use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use crate::config::ConfigError;

/// The effective uid that is treated as the administrator.
pub const ROOT_UID: u32 = 0;

/// Mode bits granting write access to the group or to others.
const GROUP_OTHER_WRITE: u32 = 0o022;

/// Mask selecting the permission bits (including setuid/setgid/sticky).
const PERMISSION_BITS: u32 = 0o7777;

/// The only config path a root process picks without being told.
pub const SYSTEM_CONFIG_PATH: &str = "/etc/doorman/doorman.toml";

/// The metadata facts the trust decision needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileFacts {
    pub uid: u32,
    /// `st_mode`; only the permission bits are inspected.
    pub mode: u32,
}

/// Source of file metadata; the real one reads the filesystem.
pub trait MetadataProvider {
    /// Resolve symlinks and `..`.
    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf>;
    /// Metadata of an already canonical path.
    fn facts(&self, path: &Path) -> io::Result<FileFacts>;
}

/// [`MetadataProvider`] backed by the real filesystem.
pub struct FsMetadata;

impl MetadataProvider for FsMetadata {
    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        std::fs::canonicalize(path)
    }

    fn facts(&self, path: &Path) -> io::Result<FileFacts> {
        let m = std::fs::metadata(path)?;
        Ok(FileFacts {
            uid: m.uid(),
            mode: m.mode(),
        })
    }
}

/// The effective uid of this process.
pub fn effective_uid() -> u32 {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() }
}

fn untrusted(path: &Path, why: &str) -> ConfigError {
    ConfigError::Invalid(format!(
        "refusing config {} for a root process: {why}; use /etc/doorman/doorman.toml \
         or fix its ownership (root) and permissions (no group/other write, also on its directory)",
        path.display()
    ))
}

fn check_one(
    config: &Path,
    path: &Path,
    what: &str,
    meta: &dyn MetadataProvider,
) -> Result<(), ConfigError> {
    let f = meta.facts(path).map_err(|e| {
        untrusted(
            config,
            &format!("cannot stat {what} {}: {e}", path.display()),
        )
    })?;
    if f.uid != ROOT_UID {
        return Err(untrusted(
            config,
            &format!(
                "{what} {} is owned by uid {}, not root",
                path.display(),
                f.uid
            ),
        ));
    }
    if f.mode & GROUP_OTHER_WRITE != 0 {
        return Err(untrusted(
            config,
            &format!(
                "{what} {} is group- or world-writable (mode {:o})",
                path.display(),
                f.mode & PERMISSION_BITS
            ),
        ));
    }
    Ok(())
}

/// Accept `path` only if the file it resolves to, and that file's directory,
/// are owned by root and not writable by group or others.
pub fn check_trusted(path: &Path, meta: &dyn MetadataProvider) -> Result<(), ConfigError> {
    let real = meta
        .canonicalize(path)
        .map_err(|e| untrusted(path, &format!("cannot resolve it: {e}")))?;
    check_one(path, &real, "file", meta)?;
    let parent = real
        .parent()
        .ok_or_else(|| untrusted(path, "it has no parent directory"))?;
    check_one(path, parent, "directory", meta)
}

/// Which file a process should load.
///
/// Non-root: `explicit` > `env` > `standard` (the caller passes the first
/// existing standard location). Root: `explicit` > `env` >
/// [`SYSTEM_CONFIG_PATH`] if it exists; the working directory and the user
/// config directory are never consulted, and every chosen path must pass
/// [`check_trusted`]. `Ok(None)` means "use built-in defaults".
pub fn resolve(
    euid: u32,
    explicit: Option<&Path>,
    env: Option<&str>,
    standard: Option<PathBuf>,
    meta: &dyn MetadataProvider,
) -> Result<Option<PathBuf>, ConfigError> {
    let chosen = explicit
        .map(Path::to_path_buf)
        .or_else(|| env.filter(|v| !v.is_empty()).map(PathBuf::from));
    if euid != ROOT_UID {
        return Ok(chosen.or(standard));
    }
    let path = match chosen {
        Some(p) => p,
        None => {
            let system = PathBuf::from(SYSTEM_CONFIG_PATH);
            if meta.canonicalize(&system).is_err() {
                return Ok(None);
            }
            system
        }
    };
    check_trusted(&path, meta)?;
    Ok(Some(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct Fake {
        facts: HashMap<PathBuf, FileFacts>,
        links: HashMap<PathBuf, PathBuf>,
    }

    impl Fake {
        fn with(mut self, p: &str, uid: u32, mode: u32) -> Self {
            self.facts.insert(PathBuf::from(p), FileFacts { uid, mode });
            self
        }
    }

    impl MetadataProvider for Fake {
        fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
            if let Some(t) = self.links.get(path) {
                return Ok(t.clone());
            }
            if self.facts.contains_key(path) {
                Ok(path.to_path_buf())
            } else {
                Err(io::Error::from(io::ErrorKind::NotFound))
            }
        }
        fn facts(&self, path: &Path) -> io::Result<FileFacts> {
            self.facts
                .get(path)
                .copied()
                .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))
        }
    }

    const USER_UID: u32 = 1000;
    const FILE_644: u32 = 0o100644;
    const DIR_755: u32 = 0o040755;

    fn good() -> Fake {
        Fake::default()
            .with(SYSTEM_CONFIG_PATH, ROOT_UID, FILE_644)
            .with("/etc/doorman", ROOT_UID, DIR_755)
    }

    #[test]
    fn root_owned_unwritable_file_and_dir_is_trusted() {
        check_trusted(Path::new(SYSTEM_CONFIG_PATH), &good()).unwrap();
    }

    #[test]
    fn rejects_user_owned_file() {
        let m = Fake::default()
            .with(SYSTEM_CONFIG_PATH, USER_UID, FILE_644)
            .with("/etc/doorman", ROOT_UID, DIR_755);
        assert!(check_trusted(Path::new(SYSTEM_CONFIG_PATH), &m).is_err());
    }

    #[test]
    fn rejects_group_or_world_writable_file() {
        for mode in [0o100664, 0o100646, 0o100666] {
            let m = Fake::default()
                .with(SYSTEM_CONFIG_PATH, ROOT_UID, mode)
                .with("/etc/doorman", ROOT_UID, DIR_755);
            assert!(check_trusted(Path::new(SYSTEM_CONFIG_PATH), &m).is_err());
        }
    }

    #[test]
    fn rejects_writable_or_user_owned_parent() {
        for (uid, mode) in [
            (ROOT_UID, 0o040775),
            (ROOT_UID, 0o040757),
            (USER_UID, DIR_755),
        ] {
            let m = Fake::default()
                .with(SYSTEM_CONFIG_PATH, ROOT_UID, FILE_644)
                .with("/etc/doorman", uid, mode);
            assert!(check_trusted(Path::new(SYSTEM_CONFIG_PATH), &m).is_err());
        }
    }

    #[test]
    fn symlink_is_judged_by_its_target() {
        let mut m = good().with("/home/u/evil", USER_UID, FILE_644);
        m.links.insert(
            PathBuf::from("/etc/doorman/link.toml"),
            PathBuf::from("/home/u/evil"),
        );
        assert!(check_trusted(Path::new("/etc/doorman/link.toml"), &m).is_err());
    }

    #[test]
    fn missing_file_is_refused() {
        assert!(check_trusted(Path::new("/nope/x.toml"), &good()).is_err());
    }

    #[test]
    fn root_ignores_standard_search_result_and_uses_system_file() {
        let got = resolve(
            ROOT_UID,
            None,
            None,
            Some(PathBuf::from("doorman.toml")),
            &good(),
        )
        .unwrap();
        assert_eq!(got, Some(PathBuf::from(SYSTEM_CONFIG_PATH)));
    }

    #[test]
    fn root_without_system_file_uses_defaults() {
        let got = resolve(
            ROOT_UID,
            None,
            None,
            Some(PathBuf::from("doorman.toml")),
            &Fake::default(),
        )
        .unwrap();
        assert_eq!(got, None);
    }

    #[test]
    fn root_explicit_and_env_paths_must_be_trusted() {
        let planted = "/home/u/doorman.toml";
        let m = good().with(planted, USER_UID, FILE_644);
        assert!(resolve(ROOT_UID, Some(Path::new(planted)), None, None, &m).is_err());
        assert!(resolve(ROOT_UID, None, Some(planted), None, &m).is_err());
        let ok = resolve(
            ROOT_UID,
            Some(Path::new(SYSTEM_CONFIG_PATH)),
            Some(planted),
            None,
            &m,
        )
        .unwrap();
        assert_eq!(ok, Some(PathBuf::from(SYSTEM_CONFIG_PATH)));
    }

    #[test]
    fn non_root_keeps_precedence_without_trust_checks() {
        let m = Fake::default();
        let std_path = Some(PathBuf::from("doorman.toml"));
        let pick = |x: Option<&Path>, v: Option<&str>| {
            resolve(USER_UID, x, v, std_path.clone(), &m).unwrap()
        };
        assert_eq!(pick(Some(Path::new("a")), Some("b")), Some("a".into()));
        assert_eq!(pick(None, Some("b")), Some("b".into()));
        assert_eq!(pick(None, None), Some("doorman.toml".into()));
    }

    #[test]
    fn load_resolved_as_root_refuses_untrusted_file_and_reads_trusted_one() {
        let dir = std::env::temp_dir().join(format!("doorman-trust-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("doorman.toml");
        std::fs::write(&file, "[daemon]\ndata_dir = \"/srv/planted\"\n").unwrap();
        let canon = std::fs::canonicalize(&file).unwrap();
        let parent = canon.parent().unwrap().to_str().unwrap().to_owned();
        let untrusted = Fake::default()
            .with(canon.to_str().unwrap(), USER_UID, FILE_644)
            .with(&parent, ROOT_UID, DIR_755);
        let trusted = Fake::default()
            .with(canon.to_str().unwrap(), ROOT_UID, FILE_644)
            .with(&parent, ROOT_UID, DIR_755);
        let err = crate::Config::load_resolved(ROOT_UID, Some(&file), None, &untrusted);
        assert!(err.is_err());
        let cfg = crate::Config::load_resolved(ROOT_UID, Some(&file), None, &trusted).unwrap();
        assert_eq!(cfg.daemon.data_dir, "/srv/planted");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
