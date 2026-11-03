use anyhow::{Context, Result};
use doorman_shared::DATA_DIR;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// First bytes of a versioned `embeddings.bin`. The legacy (version 1) file has
/// no header: it starts with the bincode `u64` element count, which can never
/// equal these bytes, so the two layouts are told apart unambiguously.
const FILE_MAGIC: &[u8; 8] = b"DRMNEMB\0";
/// Layout version written by this build: magic, `u32` little-endian version,
/// then the bincode-encoded `Vec<UserEmbedding>`.
const FORMAT_VERSION: u32 = 2;
const HEADER_LEN: usize = FILE_MAGIC.len() + std::mem::size_of::<u32>();
/// Name (next to `embeddings.bin`) of the one-time copy of a legacy file made
/// before its first version-2 write. Never overwritten; see docs/storage.md.
const LEGACY_BACKUP_NAME: &str = "embeddings.bin.v1.bak";
/// Mode of every file holding biometric data.
const PRIVATE_FILE_MODE: u32 = 0o600;
/// Mode of the data directory.
const DATA_DIR_MODE: u32 = 0o700;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEmbedding {
    pub username: String,
    /// Multiple embeddings capturing different angles/variations of the face
    pub embeddings: Vec<Vec<f32>>,
    pub enrolled_at: String,
}

pub struct Storage {
    embeddings: HashMap<String, UserEmbedding>,
    file_path: PathBuf,
}

impl Storage {
    /// Initialize storage with custom data directory
    pub async fn new_with_dir(data_dir: impl Into<PathBuf>) -> Result<Self> {
        let data_dir: PathBuf = data_dir.into();

        // Ensure data directory exists
        fs::create_dir_all(&data_dir).context("Failed to create data directory")?;

        // The directory must be private. Root always enforces it; the unprivileged
        // service enforces it on a directory it owns (a root-only check never ran
        // for it; the unit's StateDirectoryMode=0700 is the primary guard).
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            let owned = fs::metadata(&data_dir)
                .context("Failed to stat data directory")?
                .uid()
                == nix::unistd::geteuid().as_raw();
            if nix::unistd::geteuid().is_root() || owned {
                let perms = fs::Permissions::from_mode(DATA_DIR_MODE);
                fs::set_permissions(&data_dir, perms)
                    .context("Failed to set data directory permissions")?;
            }
        }

        let file_path = data_dir.join("embeddings.bin");

        // Load existing embeddings if file exists
        let embeddings = if file_path.exists() {
            debug!("Loading existing embeddings from {:?}", file_path);
            Self::load_embeddings(&file_path)?
        } else {
            info!("No existing embeddings found, starting fresh");
            HashMap::new()
        };

        info!("Storage initialized with {} users", embeddings.len());

        Ok(Self {
            embeddings,
            file_path,
        })
    }

    /// Initialize storage with default system directory (legacy)
    pub async fn new() -> Result<Self> {
        Self::new_with_dir(DATA_DIR).await
    }

    /// Decode an `embeddings.bin`: the versioned layout, or the legacy
    /// headerless one (read as version 1; the next save upgrades it).
    pub fn decode_embeddings(data: &[u8]) -> Result<Vec<UserEmbedding>> {
        match data.strip_prefix(FILE_MAGIC.as_slice()) {
            Some(rest) => {
                let version_bytes: [u8; 4] = rest
                    .get(..std::mem::size_of::<u32>())
                    .and_then(|b| b.try_into().ok())
                    .context("Embeddings file header is truncated")?;
                let version = u32::from_le_bytes(version_bytes);
                if version != FORMAT_VERSION {
                    anyhow::bail!(
                        "Unsupported embeddings file version {version} (this build reads 1 and {FORMAT_VERSION})"
                    );
                }
                bincode::deserialize(&data[HEADER_LEN..])
                    .context("Failed to deserialize embeddings")
            }
            None => bincode::deserialize(data)
                .context("Failed to deserialize embeddings (legacy format)"),
        }
    }

    pub fn encode_embeddings(users: &[UserEmbedding]) -> Result<Vec<u8>> {
        let mut data = Vec::with_capacity(HEADER_LEN);
        data.extend_from_slice(FILE_MAGIC);
        data.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bincode::serialize_into(&mut data, users).context("Failed to serialize embeddings")?;
        Ok(data)
    }

    fn load_embeddings(path: &PathBuf) -> Result<HashMap<String, UserEmbedding>> {
        let data = fs::read(path).context("Failed to read embeddings file")?;
        let map = Self::decode_embeddings(&data)?
            .into_iter()
            .map(|e| (e.username.clone(), e))
            .collect();
        Ok(map)
    }

    /// Copy a legacy (headerless, version 1) `embeddings.bin` to
    /// `embeddings.bin.v1.bak` before it is replaced by a version-2 file, so an
    /// older daemon can be restored. Created with mode 0600 and `create_new`:
    /// an existing backup is never overwritten.
    fn backup_legacy_file(&self) -> Result<()> {
        let data = match fs::read(&self.file_path) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e).context("Failed to read embeddings file for backup"),
        };
        if data.starts_with(FILE_MAGIC) {
            return Ok(());
        }
        let backup = self.file_path.with_file_name(LEGACY_BACKUP_NAME);
        match write_private_file(&backup, &data, false) {
            Ok(()) => {
                info!("Backed up the version 1 embeddings file to {:?}", backup);
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            Err(e) => Err(e).with_context(|| format!("Failed to write backup {backup:?}")),
        }
    }

    fn save_embeddings(&self) -> Result<()> {
        let embeddings_vec: Vec<UserEmbedding> = self.embeddings.values().cloned().collect();
        let data = Self::encode_embeddings(&embeddings_vec)?;

        self.backup_legacy_file()?;

        // Write a sibling file (0600 from creation, fsynced) and rename it over
        // the target, then fsync the directory so the rename is durable: a crash
        // never leaves a truncated, empty or world-readable embeddings.bin.
        let tmp_path = self.file_path.with_extension("bin.tmp");
        write_private_file(&tmp_path, &data, true).context("Failed to write embeddings file")?;
        fs::rename(&tmp_path, &self.file_path).context("Failed to replace embeddings file")?;
        let dir = self
            .file_path
            .parent()
            .context("Embeddings file has no parent directory")?;
        fs::File::open(dir)
            .and_then(|d| d.sync_all())
            .context("Failed to fsync data directory")?;

        debug!("Embeddings saved to {:?}", self.file_path);
        Ok(())
    }

    /// Store a user's embeddings (multiple variations)
    pub async fn store_embeddings(
        &mut self,
        username: String,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<()> {
        let enrolled_at = chrono::Local::now().to_rfc3339();

        let user_embedding = UserEmbedding {
            username: username.clone(),
            embeddings,
            enrolled_at,
        };

        let num_embeddings = user_embedding.embeddings.len();
        self.embeddings.insert(username.clone(), user_embedding);
        self.save_embeddings()?;

        info!(
            "Stored {} embeddings for user: {}",
            num_embeddings, username
        );
        Ok(())
    }

    /// Get a user's embeddings
    pub fn get_embeddings(&self, username: &str) -> Option<&Vec<Vec<f32>>> {
        self.embeddings.get(username).map(|e| &e.embeddings)
    }

    /// Remove a user's embedding
    pub async fn remove_embedding(&mut self, username: &str) -> Result<bool> {
        if self.embeddings.remove(username).is_some() {
            self.save_embeddings()?;
            info!("Removed embedding for user: {}", username);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// List all enrolled users
    pub fn list_users(&self) -> Vec<doorman_shared::UserInfo> {
        self.embeddings
            .values()
            .map(|e| doorman_shared::UserInfo {
                username: e.username.clone(),
                enrolled_at: e.enrolled_at.clone(),
                // Report the real number of stored per-angle embeddings (enrollment
                // selects up to TARGET_EMBEDDINGS diverse vectors), not a hardcoded 1.
                num_embeddings: e.embeddings.len(),
            })
            .collect()
    }

    /// Get number of enrolled users
    pub fn count(&self) -> usize {
        self.embeddings.len()
    }
}

/// Create `path` with mode 0600 (applied at creation, so the data is never
/// readable by others), write `data` and fsync it. With `replace`, a stale file
/// at `path` is removed first so its (possibly looser) mode cannot survive;
/// without it an existing file is an `AlreadyExists` error.
fn write_private_file(path: &Path, data: &[u8], replace: bool) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    if replace {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(PRIVATE_FILE_MODE)
        .open(path)?;
    file.write_all(data)?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<UserEmbedding> {
        vec![UserEmbedding {
            username: "alice".into(),
            embeddings: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            enrolled_at: "2026-01-01T00:00:00+00:00".into(),
        }]
    }

    /// Byte-exact fixture of the pre-versioning file: bincode of the Vec alone.
    fn legacy_bytes() -> Vec<u8> {
        bincode::serialize(&sample()).unwrap()
    }

    #[test]
    fn legacy_headerless_file_is_read() {
        let users = Storage::decode_embeddings(&legacy_bytes()).unwrap();
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].username, "alice");
        assert_eq!(users[0].embeddings, sample()[0].embeddings);
    }

    #[test]
    fn versioned_round_trip_keeps_everything() {
        let bytes = Storage::encode_embeddings(&sample()).unwrap();
        assert_eq!(&bytes[..FILE_MAGIC.len()], FILE_MAGIC);
        let users = Storage::decode_embeddings(&bytes).unwrap();
        assert_eq!(users[0].embeddings, sample()[0].embeddings);
        assert_eq!(users[0].enrolled_at, sample()[0].enrolled_at);
    }

    #[test]
    fn unknown_version_and_truncated_header_are_explicit_errors() {
        let mut bytes = Storage::encode_embeddings(&sample()).unwrap();
        bytes[FILE_MAGIC.len()..HEADER_LEN].copy_from_slice(&99u32.to_le_bytes());
        let err = Storage::decode_embeddings(&bytes).unwrap_err().to_string();
        assert!(err.contains("version 99"), "{err}");
        let err = Storage::decode_embeddings(FILE_MAGIC)
            .unwrap_err()
            .to_string();
        assert!(err.contains("truncated"), "{err}");
    }

    #[tokio::test]
    async fn legacy_file_upgrades_on_first_save_and_reloads() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("embeddings.bin"), legacy_bytes()).unwrap();

        let mut storage = Storage::new_with_dir(dir.path()).await.unwrap();
        assert_eq!(storage.get_embeddings("alice").unwrap().len(), 2);
        // Reading alone does not rewrite the file.
        assert_eq!(
            fs::read(dir.path().join("embeddings.bin")).unwrap(),
            legacy_bytes()
        );

        storage
            .store_embeddings("alice".into(), vec![vec![0.0, 0.0, 1.0]])
            .await
            .unwrap();
        let on_disk = fs::read(dir.path().join("embeddings.bin")).unwrap();
        assert!(on_disk.starts_with(FILE_MAGIC));

        let reloaded = Storage::new_with_dir(dir.path()).await.unwrap();
        assert_eq!(
            reloaded.get_embeddings("alice").unwrap(),
            &vec![vec![0.0, 0.0, 1.0]]
        );
    }

    fn mode_of(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path).unwrap().permissions().mode() & 0o7777
    }

    #[test]
    fn private_file_is_0600_at_creation_and_replaces_a_loose_stale_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("embeddings.bin.tmp");
        // A stale temp file with a loose mode must not leak its mode.
        fs::write(&path, b"stale").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        write_private_file(&path, b"new", true).unwrap();
        assert_eq!(mode_of(&path), 0o600);
        assert_eq!(fs::read(&path).unwrap(), b"new");
        // Without `replace` an existing file is an error, not an overwrite.
        let err = write_private_file(&path, b"other", false).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read(&path).unwrap(), b"new");
    }

    #[tokio::test]
    async fn saved_file_is_0600_and_no_temp_file_is_left() {
        let dir = tempfile::tempdir().unwrap();
        let mut storage = Storage::new_with_dir(dir.path()).await.unwrap();
        storage
            .store_embeddings("alice".into(), vec![vec![1.0]])
            .await
            .unwrap();
        assert_eq!(mode_of(&dir.path().join("embeddings.bin")), 0o600);
        assert!(!dir.path().join("embeddings.bin.tmp").exists());
        // A fresh install has nothing to back up.
        assert!(!dir.path().join(LEGACY_BACKUP_NAME).exists());
    }

    #[tokio::test]
    async fn data_directory_is_made_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
        Storage::new_with_dir(dir.path()).await.unwrap();
        assert_eq!(mode_of(dir.path()), 0o700);
    }

    #[tokio::test]
    async fn legacy_backup_is_created_once_and_never_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let data_file = dir.path().join("embeddings.bin");
        let backup = dir.path().join(LEGACY_BACKUP_NAME);
        fs::write(&data_file, legacy_bytes()).unwrap();

        let mut storage = Storage::new_with_dir(dir.path()).await.unwrap();
        storage
            .store_embeddings("alice".into(), vec![vec![0.0, 0.0, 1.0]])
            .await
            .unwrap();
        // First v2 write: the backup is the untouched v1 file, mode 0600.
        assert_eq!(fs::read(&backup).unwrap(), legacy_bytes());
        assert_eq!(mode_of(&backup), 0o600);
        assert!(fs::read(&data_file).unwrap().starts_with(FILE_MAGIC));

        // A second write (file is v2 now) leaves the backup alone.
        storage
            .store_embeddings("alice".into(), vec![vec![9.0]])
            .await
            .unwrap();
        assert_eq!(fs::read(&backup).unwrap(), legacy_bytes());

        // Even if a v1 file reappears (rollback), an existing backup is kept.
        let other_v1 = bincode::serialize(&Vec::<UserEmbedding>::new()).unwrap();
        fs::write(&data_file, other_v1).unwrap();
        storage
            .store_embeddings("alice".into(), vec![vec![5.0]])
            .await
            .unwrap();
        assert_eq!(fs::read(&backup).unwrap(), legacy_bytes());
    }
}
