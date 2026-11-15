//! SHA-256 integrity check of the model files the daemon loads.
//!
//! A swapped detector, recognizer or liveness model is a silent bypass (or a
//! parser-exploit vector), so the files in `[ml] models_dir` are hashed at
//! start and compared with a manifest in `sha256sum` format. The manifest
//! shipped in `packaging/models.sha256` is compiled into the binary; an
//! optional `[models] manifest_path` adds or overrides entries.
//!
//! Policy: a loaded file that is listed and does NOT match is a failure (fail
//! closed: the daemon refuses to start, unless `[models]
//! allow_unverified_models = true`). A loaded file that is not listed, or not
//! present, is reported but is not a mismatch. No model binary or upstream hash
//! is invented here: the manifest holds only the hashes pinned in
//! `scripts/fetch_models.sh`.

use crate::ml::{DepthPadConfig, DetectorConfig, LivenessConfig, RecognizerConfig};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

/// The manifest shipped with the package, compiled in.
pub const DEFAULT_MANIFEST: &str = include_str!("../../packaging/models.sha256");

const SHA256_HEX_LEN: usize = 64;
const READ_CHUNK: usize = 1 << 20;

/// Parsed manifest: file name -> lowercase hex SHA-256.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Manifest(BTreeMap<String, String>);

impl Manifest {
    /// Parse `sha256sum` format. `#` comments and blank lines are ignored.
    /// Anything else malformed is an explicit error naming the line.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut map = BTreeMap::new();
        for (n, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (hash, name) = line
                .split_once(char::is_whitespace)
                .ok_or_else(|| format!("manifest line {}: expected `<sha256>  <file>`", n + 1))?;
            // `sha256sum` marks binary mode with a leading `*` on the name.
            let name = name.trim_start().trim_start_matches('*');
            if hash.len() != SHA256_HEX_LEN || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!(
                    "manifest line {}: `{hash}` is not a SHA-256 (64 hex digits)",
                    n + 1
                ));
            }
            if name.is_empty() || name.contains('/') || name == ".." || name == "." {
                return Err(format!(
                    "manifest line {}: `{name}` must be a bare file name",
                    n + 1
                ));
            }
            if map
                .insert(name.to_string(), hash.to_ascii_lowercase())
                .is_some()
            {
                return Err(format!("manifest line {}: `{name}` is listed twice", n + 1));
            }
        }
        Ok(Self(map))
    }

    /// The compiled-in manifest.
    pub fn builtin() -> Self {
        Self::parse(DEFAULT_MANIFEST).expect("packaging/models.sha256 is valid (unit-tested)")
    }

    /// Entries of `other` win over this manifest's.
    pub fn overlaid_with(mut self, other: Manifest) -> Self {
        self.0.extend(other.0);
        self
    }

    pub fn get(&self, file: &str) -> Option<&str> {
        self.0.get(file).map(String::as_str)
    }
}

/// Result for one model file.
#[derive(Debug, Clone, PartialEq)]
pub enum FileStatus {
    /// Listed and the hash matches.
    Verified,
    /// Listed and the hash differs.
    Mismatch { expected: String, actual: String },
    /// Present but not in the manifest: cannot be verified.
    Unlisted { actual: String },
    /// Not in the models dir (the backend reports that on its own).
    Missing,
    /// Present but unreadable.
    Unreadable(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub files: Vec<(String, FileStatus)>,
}

impl Report {
    pub fn mismatches(&self) -> Vec<&str> {
        self.files
            .iter()
            .filter(|(_, s)| matches!(s, FileStatus::Mismatch { .. } | FileStatus::Unreadable(_)))
            .map(|(f, _)| f.as_str())
            .collect()
    }

    pub fn all_verified(&self) -> bool {
        self.files
            .iter()
            .all(|(_, s)| matches!(s, FileStatus::Verified))
    }
}

/// The model files the daemon loads: detector, liveness (MiniFASNet),
/// recognizer, and the depth model while the liveness gate is on. Legacy files
/// that may sit in the same directory are deliberately not listed.
pub fn loaded_model_files(liveness_enabled: bool) -> Vec<&'static str> {
    let mut files = vec![
        DetectorConfig::YUNET.model_file,
        LivenessConfig::MINIFASNET.model_file,
        RecognizerConfig::EDGEFACE.model_file,
    ];
    if liveness_enabled {
        files.push(DepthPadConfig::DEPTH_ANYTHING_V2.model_file);
    }
    files
}

/// Streaming SHA-256 of a file, lowercase hex.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; READ_CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Hash `files` in `models_dir` and compare with `manifest`.
pub fn verify(models_dir: &Path, files: &[&str], manifest: &Manifest) -> Report {
    let files = files
        .iter()
        .map(|name| {
            let path = models_dir.join(name);
            let status = if !path.is_file() {
                FileStatus::Missing
            } else {
                match sha256_file(&path) {
                    Err(e) => FileStatus::Unreadable(e.to_string()),
                    Ok(actual) => match manifest.get(name) {
                        None => FileStatus::Unlisted { actual },
                        Some(expected) if expected == actual => FileStatus::Verified,
                        Some(expected) => FileStatus::Mismatch {
                            expected: expected.to_string(),
                            actual,
                        },
                    },
                }
            };
            ((*name).to_string(), status)
        })
        .collect();
    Report { files }
}

/// Fail closed on a mismatch (or an unreadable file) unless
/// `allow_unverified` is set. The error names every offending file.
pub fn enforce(report: &Report, allow_unverified: bool) -> Result<(), String> {
    let bad = report.mismatches();
    if bad.is_empty() || allow_unverified {
        return Ok(());
    }
    Err(format!(
        "model integrity check failed for {}: the file hash does not match the manifest. \
         Refusing to start. Restore the files (scripts/fetch_models.sh, or reinstall the \
         package), update `[models] manifest_path` if you replaced a model on purpose, or set \
         `[models] allow_unverified_models = true` to accept unverified models (not recommended).",
        bad.join(", ")
    ))
}

/// Build the manifest for a config: the compiled-in one, overlaid with the
/// user's `manifest_path` when set (an unreadable or malformed user manifest
/// is an error, never silently ignored).
pub fn manifest_for(user_manifest: Option<&Path>) -> Result<Manifest, String> {
    let builtin = Manifest::builtin();
    match user_manifest {
        None => Ok(builtin),
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("cannot read models.manifest_path {}: {e}", path.display()))?;
            let user = Manifest::parse(&text)
                .map_err(|e| format!("models.manifest_path {}: {e}", path.display()))?;
            Ok(builtin.overlaid_with(user))
        }
    }
}

/// The start-up check: verify the loaded models of a config and enforce.
/// Returns the report for logging on success.
pub fn check_config(config: &doorman_shared::Config) -> Result<Report, (Report, String)> {
    let user_manifest = config
        .models
        .manifest_path
        .as_ref()
        .map(std::path::PathBuf::from);
    let empty = Report { files: Vec::new() };
    let manifest = manifest_for(user_manifest.as_deref()).map_err(|e| (empty.clone(), e))?;
    let files = loaded_model_files(config.authentication.liveness_enabled);
    let report = verify(Path::new(&config.ml.models_dir), &files, &manifest);
    match enforce(&report, config.models.allow_unverified_models) {
        Ok(()) => Ok(report),
        Err(e) => Err((report, e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ABC_SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn dir_with(files: &[(&str, &[u8])]) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        for (n, c) in files {
            std::fs::write(d.path().join(n), c).unwrap();
        }
        d
    }

    fn manifest(name: &str, hash: &str) -> Manifest {
        Manifest::parse(&format!("{hash}  {name}\n")).unwrap()
    }

    #[test]
    fn the_shipped_manifest_parses_and_lists_the_pinned_models() {
        let m = Manifest::builtin();
        for f in [
            "face_detection_yunet_2023mar.onnx",
            "edgeface_s.onnx",
            "minifasnet_v2se.onnx",
        ] {
            assert!(m.get(f).is_some(), "{f}");
        }
        assert!(loaded_model_files(true)
            .iter()
            .any(|f| f.contains("depth_anything")));
        assert!(!loaded_model_files(false)
            .iter()
            .any(|f| f.contains("depth_anything")));
    }

    #[test]
    fn the_shipped_manifest_matches_the_pins_of_fetch_models_sh() {
        let script = include_str!("../../scripts/fetch_models.sh");
        let m = Manifest::builtin();
        for (var, file) in [
            ("YUNET_SHA256", "face_detection_yunet_2023mar.onnx"),
            ("RECOGNIZER_SHA256", "edgeface_s.onnx"),
            ("MINIFASNET_SHA256", "minifasnet_v2se.onnx"),
            ("MINIFASNET_INT8_SHA256", "minifasnet_v2se_int8.onnx"),
        ] {
            let pin = script
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{var}=\"")))
                .and_then(|r| r.strip_suffix('"'))
                .unwrap_or_else(|| panic!("{var} not found"));
            assert_eq!(m.get(file), Some(pin), "{file}");
        }
    }

    #[test]
    fn parser_rejects_malformed_manifests() {
        for bad in [
            "nothex  a.onnx",
            ABC_SHA,
            &format!("{ABC_SHA}  ../a.onnx"),
            &format!("{ABC_SHA}  dir/a.onnx"),
            &format!("{}  a.onnx", &ABC_SHA[..63]),
            &format!("{ABC_SHA}  a.onnx\n{ABC_SHA}  a.onnx"),
        ] {
            assert!(Manifest::parse(bad).is_err(), "{bad}");
        }
        let ok = Manifest::parse(&format!("# c\n\n{ABC_SHA} *a.onnx\n")).unwrap();
        assert_eq!(ok.get("a.onnx"), Some(ABC_SHA));
    }

    #[test]
    fn sha256_matches_a_known_vector() {
        let d = dir_with(&[("a", b"abc")]);
        assert_eq!(sha256_file(&d.path().join("a")).unwrap(), ABC_SHA);
    }

    #[test]
    fn a_matching_file_is_verified() {
        let d = dir_with(&[("a.onnx", b"abc")]);
        let r = verify(d.path(), &["a.onnx"], &manifest("a.onnx", ABC_SHA));
        assert!(r.all_verified());
        assert!(enforce(&r, false).is_ok());
    }

    #[test]
    fn a_tampered_file_fails_closed_and_names_the_file() {
        let d = dir_with(&[("a.onnx", b"abd")]);
        let r = verify(d.path(), &["a.onnx"], &manifest("a.onnx", ABC_SHA));
        assert!(matches!(r.files[0].1, FileStatus::Mismatch { .. }));
        let err = enforce(&r, false).unwrap_err();
        assert!(
            err.contains("a.onnx") && err.contains("Refusing to start"),
            "{err}"
        );
        // The explicit opt-out accepts it.
        assert!(enforce(&r, true).is_ok());
    }

    #[test]
    fn unlisted_and_missing_files_are_not_mismatches() {
        let d = dir_with(&[("b.onnx", b"abc")]);
        let m = manifest("a.onnx", ABC_SHA);
        let r = verify(d.path(), &["a.onnx", "b.onnx"], &m);
        assert_eq!(r.files[0].1, FileStatus::Missing);
        assert!(matches!(r.files[1].1, FileStatus::Unlisted { .. }));
        assert!(enforce(&r, false).is_ok());
        assert!(!r.all_verified());
    }

    #[test]
    fn only_the_listed_loaded_files_are_checked_not_legacy_ones() {
        // A legacy file with a wrong hash in the same directory is ignored.
        let d = dir_with(&[("legacy.onnx", b"junk"), ("a.onnx", b"abc")]);
        let r = verify(d.path(), &["a.onnx"], &manifest("legacy.onnx", ABC_SHA));
        assert_eq!(r.files.len(), 1);
        assert!(enforce(&r, false).is_ok());
    }

    #[test]
    fn a_user_manifest_overrides_and_extends_the_builtin_one() {
        let user = tempfile::tempdir().unwrap();
        let p = user.path().join("m.sha256");
        std::fs::write(
            &p,
            format!("{ABC_SHA}  edgeface_s.onnx\n{ABC_SHA}  depth.onnx\n"),
        )
        .unwrap();
        let m = manifest_for(Some(&p)).unwrap();
        assert_eq!(m.get("edgeface_s.onnx"), Some(ABC_SHA));
        assert_eq!(m.get("depth.onnx"), Some(ABC_SHA));
        assert!(
            m.get("minifasnet_v2se.onnx").is_some(),
            "builtin entries stay"
        );
        assert!(manifest_for(Some(&user.path().join("nope"))).is_err());
        std::fs::write(&p, "garbage").unwrap();
        assert!(manifest_for(Some(&p)).is_err());
    }

    fn config_for(dir: &Path) -> doorman_shared::Config {
        let mut c = doorman_shared::Config::default();
        c.ml.models_dir = dir.to_string_lossy().into_owned();
        c
    }

    /// The daemon start-up check: a tampered listed model refuses, the opt-out
    /// accepts, an untouched or absent set passes.
    #[test]
    fn startup_check_refuses_a_tampered_model_unless_opted_out() {
        let d = dir_with(&[("edgeface_s.onnx", b"not the pinned model")]);
        let mut cfg = config_for(d.path());
        let (report, err) = check_config(&cfg).unwrap_err();
        assert_eq!(report.mismatches(), vec!["edgeface_s.onnx"]);
        assert!(err.contains("edgeface_s.onnx"), "{err}");
        cfg.models.allow_unverified_models = true;
        assert!(check_config(&cfg).is_ok());
        // Nothing loaded yet (fresh install): not a mismatch.
        let empty = tempfile::tempdir().unwrap();
        assert!(check_config(&config_for(empty.path())).is_ok());
    }

    #[test]
    fn startup_check_fails_closed_on_a_broken_user_manifest() {
        let d = tempfile::tempdir().unwrap();
        let mut cfg = config_for(d.path());
        cfg.models.manifest_path = Some(
            d.path()
                .join("missing.sha256")
                .to_string_lossy()
                .into_owned(),
        );
        assert!(check_config(&cfg).is_err());
    }
}
