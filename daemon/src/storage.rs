use crate::storage_key::{self, DataKey, DATA_KEY_LEN};
use anyhow::{Context, Result};
use doorman_shared::DATA_DIR;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// First bytes of a versioned `embeddings.bin`. The legacy (version 1) file has
/// no header: it starts with the bincode `u64` element count, which can never
/// equal these bytes, so the two layouts are told apart unambiguously.
const FILE_MAGIC: &[u8; 8] = b"DRMNEMB\0";
/// Layout version written by this build: magic, `u32` little-endian version,
/// then the bincode-encoded `Vec<UserEmbedding>`. Version 2 had no per-template
/// metadata (its records are [`UserEmbeddingV2`]); version 3 adds
/// [`UserEmbedding::templates_meta`]. Versions 1 (headerless) and 2 are still read.
const FORMAT_VERSION: u32 = 3;
/// Sealed (encrypted) envelope version: magic, `u32` version 4, 12-byte nonce,
/// then the ChaCha20-Poly1305 ciphertext of a whole version-3 file image.
/// Written only when a data key is loaded (see `storage_key`); read only with
/// the key. There is no plaintext downgrade path.
pub const SEALED_VERSION: u32 = 4;
/// Sealed header length: magic + version + nonce.
pub const SEALED_HEADER_LEN: usize =
    FILE_MAGIC.len() + std::mem::size_of::<u32>() + storage_key::NONCE_LEN;
/// Last layout without per-template metadata.
const LEGACY_META_FREE_VERSION: u32 = 2;
const HEADER_LEN: usize = FILE_MAGIC.len() + std::mem::size_of::<u32>();
/// Name (next to `embeddings.bin`) of the one-time copy of a headerless
/// (version 1) file made before its first versioned write. Never overwritten.
const LEGACY_BACKUP_NAME: &str = "embeddings.bin.v1.bak";
/// Same for a version 2 file, made before its first version 3 write (an older
/// daemon cannot read version 3; restore this copy to roll back).
const V2_BACKUP_NAME: &str = "embeddings.bin.v2.bak";
/// Same for a version 3 file, made before its first sealed (v4) write (only a
/// key-holding daemon reads version 4; restore this copy to roll back or to
/// abandon encryption).
const V3_BACKUP_NAME: &str = "embeddings.bin.v3.bak";
/// Mode of every file holding biometric data.
const PRIVATE_FILE_MODE: u32 = 0o600;
/// Mode of the data directory.
const DATA_DIR_MODE: u32 = 0o700;

/// What was measured when one template was captured. Never contains pixels.
/// Used for enrollment-session accounting and later calibration; it plays no
/// part in authentication decisions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateMeta {
    /// Capture session: Unix milliseconds at which the session began.
    pub session_id: u64,
    /// RFC 3339 time this frame was captured.
    pub captured_at: String,
    /// Monocular-depth relief of the frame (`None` when liveness was disabled).
    pub relief: Option<f32>,
    /// Mean luma (0..=255) inside the face box.
    pub brightness: f32,
    /// Face box area as a fraction of the frame.
    pub face_area_frac: f32,
    /// Horizontal head-turn proxy from the 5 landmarks (0 = frontal), if known.
    pub yaw_ratio: Option<f32>,
    /// Vertical head-tilt proxy from the 5 landmarks (about 0.5 = frontal), if known.
    pub pitch_ratio: Option<f32>,
    /// Daemon version that captured it.
    pub build: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEmbedding {
    pub username: String,
    /// Multiple embeddings capturing different angles/variations of the face
    pub embeddings: Vec<Vec<f32>>,
    pub enrolled_at: String,
    /// Metadata parallel to `embeddings` (same length, `None` for templates
    /// stored before metadata existed or added without it).
    pub templates_meta: Vec<Option<TemplateMeta>>,
}

impl UserEmbedding {
    /// A record without metadata (every template unannotated).
    pub fn without_meta(username: String, embeddings: Vec<Vec<f32>>, enrolled_at: String) -> Self {
        let templates_meta = vec![None; embeddings.len()];
        Self {
            username,
            embeddings,
            enrolled_at,
            templates_meta,
        }
    }
}

/// On-disk record of layouts 1 and 2 (no metadata).
#[derive(Deserialize)]
struct UserEmbeddingV2 {
    username: String,
    embeddings: Vec<Vec<f32>>,
    enrolled_at: String,
}

impl From<UserEmbeddingV2> for UserEmbedding {
    fn from(v: UserEmbeddingV2) -> Self {
        Self::without_meta(v.username, v.embeddings, v.enrolled_at)
    }
}

/// Marker for "the file is sealed but no data key is loaded". Carried inside
/// the `anyhow` chain (with context around it) so IPC handlers can fail closed
/// with the stable [`doorman_shared::STORAGE_LOCKED_REASON`] reason instead of
/// an opaque I/O error. Never contains key material.
#[derive(Debug)]
pub struct StorageLocked;

impl std::fmt::Display for StorageLocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", doorman_shared::STORAGE_LOCKED_REASON)
    }
}

impl std::error::Error for StorageLocked {}

/// True when `err` (possibly with context attached) is a locked-store refusal.
/// Used by IPC handlers and tests; kept even where a build context does not
/// call it (this module compiles both as lib and inside the daemon binary).
#[cfg_attr(not(test), allow(dead_code))]
pub fn is_locked_error(err: &anyhow::Error) -> bool {
    err.downcast_ref::<StorageLocked>().is_some()
}

pub struct Storage {
    embeddings: HashMap<String, UserEmbedding>,
    file_path: PathBuf,
    /// At-rest encryption key. `Some` = every save writes the sealed (v4)
    /// envelope; `None` = current v1/v2/v3 behavior is kept byte-for-byte.
    data_key: Option<DataKey>,
}

impl Storage {
    /// Initialize storage with custom data directory
    pub async fn new_with_dir(data_dir: impl Into<PathBuf>) -> Result<Self> {
        Self::new_with_dir_and_key(data_dir, None).await
    }

    /// Initialize storage with custom data directory and an optional at-rest
    /// data key (from the systemd credential or a prior unlock). A sealed (v4)
    /// file without a key aborts here: the daemon fails closed at startup
    /// instead of running with an unreadable store.
    pub async fn new_with_dir_and_key(
        data_dir: impl Into<PathBuf>,
        data_key: Option<DataKey>,
    ) -> Result<Self> {
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
        let key_bytes: Option<[u8; DATA_KEY_LEN]> = data_key.as_ref().map(|k| *k.bytes());
        let embeddings = if file_path.exists() {
            debug!("Loading existing embeddings from {:?}", file_path);
            Self::load_embeddings_with_key(&file_path, key_bytes.as_ref())?
        } else {
            info!("No existing embeddings found, starting fresh");
            HashMap::new()
        };

        info!("Storage initialized with {} users", embeddings.len());

        Ok(Self {
            embeddings,
            file_path,
            data_key,
        })
    }

    /// Load the at-rest data key into a running store and re-read the file, so
    /// a `UnlockStorage` IPC can open a sealed store without a restart.
    /// Fails closed (in-memory state untouched) when the file is sealed and
    /// the key does not open it. A missing file is an empty store, not an
    /// error (fresh PIN-mode daemon before the first enroll).
    pub fn set_data_key(&mut self, key: DataKey) -> Result<()> {
        let key_bytes = *key.bytes();
        let map = match Self::load_embeddings_with_key(&self.file_path, Some(&key_bytes)) {
            Ok(m) => m,
            Err(_) if !self.file_path.exists() => {
                debug!("no embeddings file yet; key loaded for future sealed writes");
                HashMap::new()
            }
            Err(e) => return Err(e),
        };
        self.embeddings = map;
        self.data_key = Some(key);
        info!("storage data key loaded; sealed store re-read");
        Ok(())
    }

    /// Swap the loaded data key and re-seal the store with it (rotation).
    /// The in-memory user set is authoritative: the file is rewritten sealed
    /// with the new key. Fails closed: on any error the old key is restored
    /// and the file on disk is untouched (crash-safe write semantics).
    pub fn rekey(&mut self, key: DataKey) -> Result<()> {
        let old = self.data_key.replace(key);
        if let Err(e) = self.save_embeddings() {
            self.data_key = old;
            return Err(e);
        }
        info!("storage data key rotated; store re-sealed");
        Ok(())
    }

    /// True when a data key is loaded (sealed writes/reads possible).
    pub fn key_loaded(&self) -> bool {
        self.data_key.is_some()
    }

    /// True when the file on disk is a sealed (v4) envelope. Reads only the
    /// 12-byte magic + 4-byte version, never the payload.
    pub fn file_is_sealed(path: &Path) -> bool {
        let header = fs::read(path).ok().filter(|d| d.len() >= HEADER_LEN);
        match header {
            Some(d) => {
                d.starts_with(FILE_MAGIC)
                    && u32::from_le_bytes(d[8..12].try_into().unwrap_or([0; 4])) == SEALED_VERSION
            }
            None => false,
        }
    }

    /// Initialize storage with default system directory (legacy)
    pub async fn new() -> Result<Self> {
        Self::new_with_dir(DATA_DIR).await
    }

    /// Decode an `embeddings.bin` without a data key. A sealed (v4) file
    /// fails with a [`StorageLocked`] refusal; every older layout decodes.
    pub fn decode_embeddings(data: &[u8]) -> Result<Vec<UserEmbedding>> {
        Self::decode_embeddings_with_key(data, None)
    }

    /// Decode an `embeddings.bin`: the current layout, a sealed (v4) envelope
    /// (opened only with `key`), or an older one (the headerless version 1
    /// and version 2 carry no metadata; the next save upgrades them).
    pub fn decode_embeddings_with_key(
        data: &[u8],
        key: Option<&[u8; DATA_KEY_LEN]>,
    ) -> Result<Vec<UserEmbedding>> {
        match data.strip_prefix(FILE_MAGIC.as_slice()) {
            Some(rest) => {
                let version_bytes: [u8; 4] = rest
                    .get(..std::mem::size_of::<u32>())
                    .and_then(|b| b.try_into().ok())
                    .context("Embeddings file header is truncated")?;
                let version = u32::from_le_bytes(version_bytes);
                let body = &data[HEADER_LEN..];
                let users: Vec<UserEmbedding> = match version {
                    FORMAT_VERSION => bincode::deserialize(body)
                        .context("Failed to deserialize embeddings")?,
                    LEGACY_META_FREE_VERSION => {
                        let old: Vec<UserEmbeddingV2> = bincode::deserialize(body)
                            .context("Failed to deserialize embeddings (version 2)")?;
                        old.into_iter().map(Into::into).collect()
                    }
                    SEALED_VERSION => {
                        let key = key.ok_or_else(|| {
                            anyhow::Error::new(StorageLocked).context(
                                "embeddings file is sealed (v4) and no data key is loaded",
                            )
                        })?;
                        let opened = Self::open_sealed(&data[HEADER_LEN..], key)?;
                        // The plaintext is a whole v3 file image: recurse
                        // without the key (a sealed-in-sealed file is corrupt,
                        // not a reason to touch the key again).
                        return Self::decode_embeddings(&opened);
                    }
                    _ => anyhow::bail!(
                        "Unsupported embeddings file version {version} (this build reads 1, {LEGACY_META_FREE_VERSION}, {FORMAT_VERSION} and sealed {SEALED_VERSION})"
                    ),
                };
                Self::check_meta_lengths(users)
            }
            None => {
                let old: Vec<UserEmbeddingV2> = bincode::deserialize(data)
                    .context("Failed to deserialize embeddings (legacy format)")?;
                Ok(old.into_iter().map(Into::into).collect())
            }
        }
    }

    /// A record whose metadata is not parallel to its embeddings is corrupt.
    fn check_meta_lengths(users: Vec<UserEmbedding>) -> Result<Vec<UserEmbedding>> {
        if let Some(bad) = users
            .iter()
            .find(|u| u.templates_meta.len() != u.embeddings.len())
        {
            anyhow::bail!(
                "Embeddings file is inconsistent: {} templates but {} metadata entries",
                bad.embeddings.len(),
                bad.templates_meta.len()
            );
        }
        Ok(users)
    }

    pub fn encode_embeddings(users: &[UserEmbedding]) -> Result<Vec<u8>> {
        let mut data = Vec::with_capacity(HEADER_LEN);
        data.extend_from_slice(FILE_MAGIC);
        data.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bincode::serialize_into(&mut data, users).context("Failed to serialize embeddings")?;
        Ok(data)
    }

    /// Seal a user list into the v4 envelope: `magic || u32(4) || blob` where
    /// the blob is `nonce || AEAD(v3 file image)`. The AAD binds this exact
    /// envelope, so a version byte flipped to 3 (or a spliced nonce) fails
    /// authentication on open instead of decoding as plaintext.
    pub fn encode_sealed(users: &[UserEmbedding], key: &[u8; DATA_KEY_LEN]) -> Result<Vec<u8>> {
        let plain = Self::encode_embeddings(users)?;
        let blob = storage_key::seal_blob(&plain, key)?;
        // Header (magic + version + nonce) is fixed-size; the tag adds 16.
        let mut data = Vec::with_capacity(SEALED_HEADER_LEN + plain.len() + 16);
        data.extend_from_slice(FILE_MAGIC);
        data.extend_from_slice(&SEALED_VERSION.to_le_bytes());
        data.extend_from_slice(&blob);
        Ok(data)
    }

    fn open_sealed(blob: &[u8], key: &[u8; DATA_KEY_LEN]) -> Result<Vec<u8>> {
        storage_key::open_blob(blob, key).context("sealed embeddings failed authentication")
    }

    fn load_embeddings_with_key(
        path: &PathBuf,
        key: Option<&[u8; DATA_KEY_LEN]>,
    ) -> Result<HashMap<String, UserEmbedding>> {
        let data = fs::read(path).context("Failed to read embeddings file")?;
        let map = Self::decode_embeddings_with_key(&data, key)?
            .into_iter()
            .map(|e| (e.username.clone(), e))
            .collect();
        Ok(map)
    }

    /// Copy an older-layout `embeddings.bin` (headerless version 1, version
    /// 2, or version 3 when the pending write seals it) to its `.bak` before
    /// it is replaced by the current layout, so an older daemon can be
    /// restored. Created with mode 0600 and `create_new`: an existing backup
    /// is never overwritten. A file already in the layout being written needs
    /// no backup. `sealing` = the pending write is the v4 envelope.
    fn backup_old_layout(&self, sealing: bool) -> Result<()> {
        let data = match fs::read(&self.file_path) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e).context("Failed to read embeddings file for backup"),
        };
        let name = match data.strip_prefix(FILE_MAGIC.as_slice()) {
            None => LEGACY_BACKUP_NAME,
            Some(rest) => match rest.get(..4).and_then(|b| <[u8; 4]>::try_from(b).ok()) {
                Some(v) if u32::from_le_bytes(v) == LEGACY_META_FREE_VERSION => V2_BACKUP_NAME,
                Some(v) if sealing && u32::from_le_bytes(v) == FORMAT_VERSION => V3_BACKUP_NAME,
                _ => {
                    self.warn_if_backup_damaged(V3_BACKUP_NAME);
                    return Ok(());
                }
            },
        };
        let backup = self.file_path.with_file_name(name);
        // An existing backup is kept unless it is provably a partial write of
        // this very file (a strict prefix: a crash or full disk mid-copy).
        let replace = match fs::read(&backup) {
            Ok(existing) if existing == data => return Ok(()),
            Ok(existing) if existing.len() < data.len() && data.starts_with(&existing) => {
                warn!(
                    "Embeddings backup {:?} is truncated ({} of {} bytes); rewriting it",
                    backup,
                    existing.len(),
                    data.len()
                );
                true
            }
            Ok(_) => return Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            Err(e) => return Err(e).with_context(|| format!("Failed to read backup {backup:?}")),
        };
        self.install_backup(&backup, &data, replace)
            .with_context(|| format!("Failed to write backup {backup:?}"))?;
        info!("Backed up the previous embeddings file to {:?}", backup);
        Ok(())
    }

    /// Crash-safe backup write: a 0600 temp file (fsynced) is moved into place
    /// atomically, so `backup` is either absent/old or complete, never partial.
    /// Without `replace` an existing backup is left alone (hard link refuses).
    fn install_backup(&self, backup: &Path, data: &[u8], replace: bool) -> std::io::Result<()> {
        let tmp = backup.with_extension("bak.tmp");
        write_private_file(&tmp, data, true)?;
        let placed = if replace {
            fs::rename(&tmp, backup)
        } else {
            let linked = fs::hard_link(&tmp, backup);
            let _ = fs::remove_file(&tmp);
            match linked {
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
                other => other,
            }
        };
        if placed.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        placed?;
        match backup.parent() {
            Some(dir) => fs::File::open(dir).and_then(|d| d.sync_all()),
            None => Ok(()),
        }
    }

    /// The main file is already version 3: a version 2 backup that no longer
    /// decodes cannot be repaired (its source is gone), so only warn.
    fn warn_if_backup_damaged(&self, name: &str) {
        let backup = self.file_path.with_file_name(name);
        if let Ok(bytes) = fs::read(&backup) {
            if Self::decode_embeddings(&bytes).is_err() {
                warn!(
                    "Embeddings backup {:?} is damaged and the main file is already upgraded; it cannot be rebuilt",
                    backup
                );
            }
        }
    }

    fn save_embeddings(&self) -> Result<()> {
        // Never silently downgrade a sealed file: without the key a save
        // would replace ciphertext with plaintext. Fail closed instead.
        if self.data_key.is_none() && Self::file_is_sealed(&self.file_path) {
            return Err(anyhow::Error::new(StorageLocked).context(
                "embeddings file is sealed (v4) and no data key is loaded; refusing to overwrite it with plaintext",
            ));
        }
        let embeddings_vec: Vec<UserEmbedding> = self.embeddings.values().cloned().collect();
        let sealing = self.data_key.is_some();
        let data = match &self.data_key {
            Some(k) => Self::encode_sealed(&embeddings_vec, k.bytes())?,
            None => Self::encode_embeddings(&embeddings_vec)?,
        };

        self.backup_old_layout(sealing)?;

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

    /// Store a user's embeddings (multiple variations) without metadata. A
    /// template that is already stored keeps its metadata (matched by exact
    /// vector equality); new vectors get none.
    pub async fn store_embeddings(
        &mut self,
        username: String,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<()> {
        let previous = self.embeddings.get(&username);
        let templates_meta = embeddings
            .iter()
            .map(|e| {
                previous.and_then(|p| {
                    p.embeddings
                        .iter()
                        .position(|old| old == e)
                        .and_then(|i| p.templates_meta.get(i).cloned().flatten())
                })
            })
            .collect();
        self.store_record(username, embeddings, templates_meta)
    }

    /// Store a user's templates together with their capture metadata
    /// (`templates_meta` is parallel to `embeddings`).
    pub async fn store_templates(
        &mut self,
        username: String,
        embeddings: Vec<Vec<f32>>,
        templates_meta: Vec<Option<TemplateMeta>>,
    ) -> Result<()> {
        anyhow::ensure!(
            embeddings.len() == templates_meta.len(),
            "templates and metadata differ in length ({} vs {})",
            embeddings.len(),
            templates_meta.len()
        );
        self.store_record(username, embeddings, templates_meta)
    }

    fn store_record(
        &mut self,
        username: String,
        embeddings: Vec<Vec<f32>>,
        templates_meta: Vec<Option<TemplateMeta>>,
    ) -> Result<()> {
        let user_embedding = UserEmbedding {
            username: username.clone(),
            embeddings,
            enrolled_at: chrono::Local::now().to_rfc3339(),
            templates_meta,
        };

        let num_embeddings = user_embedding.embeddings.len();
        let previous = self.embeddings.insert(username.clone(), user_embedding);
        if let Err(e) = self.save_embeddings() {
            // Keep memory consistent with disk: a failed write changes nothing.
            match previous {
                Some(p) => self.embeddings.insert(username, p),
                None => self.embeddings.remove(&username),
            };
            return Err(e);
        }

        info!(
            "Stored {} embeddings for user: {}",
            num_embeddings, username
        );
        Ok(())
    }

    /// A user's full record (templates with their metadata).
    pub fn get_user(&self, username: &str) -> Option<&UserEmbedding> {
        self.embeddings.get(username)
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
        vec![UserEmbedding::without_meta(
            "alice".into(),
            vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            "2026-01-01T00:00:00+00:00".into(),
        )]
    }

    /// Frozen copy of the pre-metadata record, so the legacy fixtures cannot
    /// drift with the current struct.
    #[derive(Serialize)]
    struct OldRecord {
        username: String,
        embeddings: Vec<Vec<f32>>,
        enrolled_at: String,
    }

    fn old_records() -> Vec<OldRecord> {
        sample()
            .into_iter()
            .map(|u| OldRecord {
                username: u.username,
                embeddings: u.embeddings,
                enrolled_at: u.enrolled_at,
            })
            .collect()
    }

    /// Byte-exact fixture of the pre-versioning file: bincode of the Vec alone.
    fn legacy_bytes() -> Vec<u8> {
        bincode::serialize(&old_records()).unwrap()
    }

    /// Byte-exact fixture of a version 2 file (header + metadata-free records).
    fn v2_bytes() -> Vec<u8> {
        let mut data = FILE_MAGIC.to_vec();
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend(bincode::serialize(&old_records()).unwrap());
        data
    }

    fn meta(session: u64, relief: f32) -> TemplateMeta {
        TemplateMeta {
            session_id: session,
            captured_at: "2026-11-11T08:00:00+00:00".into(),
            relief: Some(relief),
            brightness: 120.0,
            face_area_frac: 0.12,
            yaw_ratio: Some(0.05),
            pitch_ratio: None,
            build: "test".into(),
        }
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
        let other_v1 = bincode::serialize(&Vec::<OldRecord>::new()).unwrap();
        fs::write(&data_file, other_v1).unwrap();
        storage
            .store_embeddings("alice".into(), vec![vec![5.0]])
            .await
            .unwrap();
        assert_eq!(fs::read(&backup).unwrap(), legacy_bytes());
    }

    #[test]
    fn version_2_file_is_read_without_metadata() {
        let users = Storage::decode_embeddings(&v2_bytes()).unwrap();
        assert_eq!(users[0].embeddings, sample()[0].embeddings);
        assert_eq!(users[0].templates_meta, vec![None, None]);
    }

    #[test]
    fn metadata_round_trips_and_mismatched_lengths_are_rejected() {
        let mut u = sample();
        u[0].templates_meta = vec![Some(meta(7, 0.12)), None];
        let bytes = Storage::encode_embeddings(&u).unwrap();
        let back = Storage::decode_embeddings(&bytes).unwrap();
        assert_eq!(back[0].templates_meta, u[0].templates_meta);

        u[0].templates_meta.pop();
        let bad = Storage::encode_embeddings(&u).unwrap();
        let err = Storage::decode_embeddings(&bad).unwrap_err().to_string();
        assert!(err.contains("inconsistent"), "{err}");
    }

    #[tokio::test]
    async fn v2_file_is_backed_up_once_then_upgraded_with_metadata_kept() {
        let dir = tempfile::tempdir().unwrap();
        let data_file = dir.path().join("embeddings.bin");
        let backup = dir.path().join(V2_BACKUP_NAME);
        fs::write(&data_file, v2_bytes()).unwrap();

        let mut storage = Storage::new_with_dir(dir.path()).await.unwrap();
        let old = storage.get_user("alice").unwrap().embeddings.clone();
        // No data loss: the old templates are re-stored together with a new one.
        let mut all = old.clone();
        all.push(vec![0.0, 0.0, 1.0]);
        storage
            .store_templates(
                "alice".into(),
                all.clone(),
                vec![None, None, Some(meta(9, 0.2))],
            )
            .await
            .unwrap();
        assert_eq!(fs::read(&backup).unwrap(), v2_bytes());
        assert_eq!(mode_of(&backup), 0o600);
        assert_eq!(mode_of(&data_file), 0o600);
        assert!(!dir.path().join(LEGACY_BACKUP_NAME).exists());

        let reloaded = Storage::new_with_dir(dir.path()).await.unwrap();
        let u = reloaded.get_user("alice").unwrap();
        assert_eq!(u.embeddings, all);
        assert_eq!(u.templates_meta[2], Some(meta(9, 0.2)));
        assert_eq!(&u.embeddings[..2], &old[..]);

        // A third write (file is v3) leaves the backup untouched.
        let mut storage = reloaded;
        storage.store_embeddings("alice".into(), all).await.unwrap();
        assert_eq!(fs::read(&backup).unwrap(), v2_bytes());
    }

    #[tokio::test]
    async fn truncated_v2_backup_is_repaired_while_the_main_file_is_still_v2() {
        let dir = tempfile::tempdir().unwrap();
        let data_file = dir.path().join("embeddings.bin");
        let backup = dir.path().join(V2_BACKUP_NAME);
        fs::write(&data_file, v2_bytes()).unwrap();
        // A crash left only the first 10 bytes of the copy.
        fs::write(&backup, &v2_bytes()[..10]).unwrap();

        let mut storage = Storage::new_with_dir(dir.path()).await.unwrap();
        storage
            .store_embeddings("alice".into(), vec![vec![0.0, 0.0, 1.0]])
            .await
            .unwrap();
        assert_eq!(fs::read(&backup).unwrap(), v2_bytes(), "repaired");
        assert_eq!(mode_of(&backup), 0o600);
        // Neither temp file is left behind.
        assert!(!dir.path().join("embeddings.bin.v2.bak.tmp").exists());
        assert!(!dir.path().join("embeddings.bin.tmp").exists());
    }

    #[tokio::test]
    async fn a_complete_but_different_backup_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join(V2_BACKUP_NAME);
        fs::write(dir.path().join("embeddings.bin"), v2_bytes()).unwrap();
        let mut other = v2_bytes();
        other.push(0xAA);
        fs::write(&backup, &other).unwrap();
        let mut storage = Storage::new_with_dir(dir.path()).await.unwrap();
        storage
            .store_embeddings("alice".into(), vec![vec![1.0]])
            .await
            .unwrap();
        assert_eq!(fs::read(&backup).unwrap(), other);
    }

    #[tokio::test]
    async fn damaged_backup_is_left_alone_once_the_main_file_is_v3() {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join(V2_BACKUP_NAME);
        let mut storage = Storage::new_with_dir(dir.path()).await.unwrap();
        storage
            .store_embeddings("alice".into(), vec![vec![1.0]])
            .await
            .unwrap();
        fs::write(&backup, b"DMN").unwrap();
        storage
            .store_embeddings("alice".into(), vec![vec![2.0]])
            .await
            .unwrap();
        assert_eq!(fs::read(&backup).unwrap(), b"DMN", "kept as is (warned)");
    }

    #[tokio::test]
    async fn failed_backup_write_leaves_no_partial_backup_and_the_main_file_intact() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let data_file = dir.path().join("embeddings.bin");
        fs::write(&data_file, v2_bytes()).unwrap();
        let mut storage = Storage::new_with_dir(dir.path()).await.unwrap();
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o500)).unwrap();
        let res = storage
            .store_embeddings("alice".into(), vec![vec![1.0]])
            .await;
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700)).unwrap();
        if res.is_ok() {
            return; // running as root: permissions are not enforced
        }
        assert_eq!(fs::read(&data_file).unwrap(), v2_bytes());
        assert!(!dir.path().join(V2_BACKUP_NAME).exists());
    }

    #[tokio::test]
    async fn store_embeddings_keeps_metadata_of_templates_that_stay() {
        let dir = tempfile::tempdir().unwrap();
        let mut storage = Storage::new_with_dir(dir.path()).await.unwrap();
        storage
            .store_templates(
                "alice".into(),
                vec![vec![1.0], vec![2.0]],
                vec![Some(meta(1, 0.1)), Some(meta(2, 0.2))],
            )
            .await
            .unwrap();
        // Adaptive append path: one old template kept, one new without metadata.
        storage
            .store_embeddings("alice".into(), vec![vec![2.0], vec![3.0]])
            .await
            .unwrap();
        let u = storage.get_user("alice").unwrap();
        assert_eq!(u.templates_meta, vec![Some(meta(2, 0.2)), None]);
        let err = storage
            .store_templates("alice".into(), vec![vec![1.0]], vec![])
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("differ in length"), "{err}");
    }

    // --- sealed (v4) envelope -------------------------------------------------

    fn sealed_key() -> [u8; DATA_KEY_LEN] {
        [9u8; DATA_KEY_LEN]
    }

    #[test]
    fn sealed_envelope_round_trips_and_needs_the_key() {
        let users = sample();
        let sealed = Storage::encode_sealed(&users, &sealed_key()).unwrap();
        assert!(sealed.starts_with(FILE_MAGIC));
        assert_eq!(
            u32::from_le_bytes(sealed[8..12].try_into().unwrap()),
            SEALED_VERSION
        );
        // Two seals of the same input differ (random nonce).
        let again = Storage::encode_sealed(&users, &sealed_key()).unwrap();
        assert_ne!(sealed, again);

        let back = Storage::decode_embeddings_with_key(&sealed, Some(&sealed_key())).unwrap();
        assert_eq!(back[0].embeddings, users[0].embeddings);

        // Without the key the refusal is typed, not an opaque parse error.
        let err = Storage::decode_embeddings(&sealed).unwrap_err();
        assert!(is_locked_error(&err), "{err:#}");
        assert!(format!("{err:#}").contains("storage_locked"));
    }

    #[test]
    fn sealed_envelope_rejects_wrong_key_and_tamper() {
        let sealed = Storage::encode_sealed(&sample(), &sealed_key()).unwrap();
        let mut other = sealed_key();
        other[0] ^= 1;
        let err = Storage::decode_embeddings_with_key(&sealed, Some(&other)).unwrap_err();
        assert!(
            !is_locked_error(&err),
            "wrong key is corruption, not {err:#}"
        );
        let mut cut = sealed.clone();
        let n = cut.len();
        cut[n - 1] ^= 1;
        assert!(Storage::decode_embeddings_with_key(&cut, Some(&sealed_key())).is_err());
        // Flipping the version byte to 3 must not decode as plaintext.
        let mut downgraded = sealed.clone();
        downgraded[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        assert!(Storage::decode_embeddings(&downgraded).is_err());
    }

    #[tokio::test]
    async fn first_sealed_write_backs_up_v3_once() {
        let dir = tempfile::tempdir().unwrap();
        let data_file = dir.path().join("embeddings.bin");
        let backup = dir.path().join(V3_BACKUP_NAME);
        let v3 = Storage::encode_embeddings(&sample()).unwrap();
        fs::write(&data_file, &v3).unwrap();

        let mut storage = Storage::new_with_dir_and_key(
            dir.path(),
            Some(DataKey::from_slice(&sealed_key()).unwrap()),
        )
        .await
        .unwrap();
        // The save seals + backs up the v3 original (old daemons read v3).
        storage
            .store_embeddings("alice".into(), vec![vec![0.0, 0.0, 1.0]])
            .await
            .unwrap();
        let on_disk = fs::read(&data_file).unwrap();
        assert!(on_disk.starts_with(FILE_MAGIC));
        assert_eq!(
            u32::from_le_bytes(on_disk[8..12].try_into().unwrap()),
            SEALED_VERSION
        );
        assert_eq!(mode_of(&data_file), 0o600);
        assert_eq!(fs::read(&backup).unwrap(), v3);
        assert_eq!(mode_of(&backup), 0o600);

        // A second sealed write leaves the backup alone; no temp files remain.
        storage
            .store_embeddings("alice".into(), vec![vec![5.0]])
            .await
            .unwrap();
        assert_eq!(fs::read(&backup).unwrap(), v3);
        assert!(!dir.path().join("embeddings.bin.tmp").exists());

        // The sealed store reloads with the key and serves the data.
        let reloaded = Storage::new_with_dir_and_key(
            dir.path(),
            Some(DataKey::from_slice(&sealed_key()).unwrap()),
        )
        .await
        .unwrap();
        assert!(reloaded.key_loaded());
        assert_eq!(reloaded.get_embeddings("alice").unwrap(), &vec![vec![5.0]]);
    }

    #[tokio::test]
    async fn sealed_file_without_key_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let data_file = dir.path().join("embeddings.bin");
        fs::write(
            &data_file,
            Storage::encode_sealed(&sample(), &sealed_key()).unwrap(),
        )
        .unwrap();
        assert!(Storage::file_is_sealed(&data_file));

        // Startup refuses: no silent empty store.
        let err = match Storage::new_with_dir(dir.path()).await {
            Ok(_) => panic!("sealed store opened without a key"),
            Err(e) => e,
        };
        assert!(is_locked_error(&err), "{err:#}");

        // And a keyless save never downgrades ciphertext to plaintext.
        let mut storage = Storage::new_with_dir_and_key(
            dir.path(),
            Some(DataKey::from_slice(&sealed_key()).unwrap()),
        )
        .await
        .unwrap();
        storage.data_key = None;
        let before = fs::read(&data_file).unwrap();
        let err = storage
            .store_embeddings("alice".into(), vec![vec![1.0]])
            .await
            .unwrap_err();
        assert!(is_locked_error(&err), "{err:#}");
        assert_eq!(fs::read(&data_file).unwrap(), before);
    }

    #[test]
    fn file_is_sealed_only_for_v4() {
        let dir = tempfile::tempdir().unwrap();
        let v3 = dir.path().join("v3.bin");
        fs::write(&v3, Storage::encode_embeddings(&sample()).unwrap()).unwrap();
        assert!(!Storage::file_is_sealed(&v3));
        let v4 = dir.path().join("v4.bin");
        fs::write(
            &v4,
            Storage::encode_sealed(&sample(), &sealed_key()).unwrap(),
        )
        .unwrap();
        assert!(Storage::file_is_sealed(&v4));
        assert!(!Storage::file_is_sealed(dir.path()));
    }

    #[tokio::test]
    async fn set_key_on_missing_file_starts_empty_and_seals_first_write() {
        let dir = tempfile::tempdir().unwrap();
        let data_file = dir.path().join("embeddings.bin");
        let mut storage = Storage::new_with_dir(dir.path()).await.unwrap();
        storage
            .set_data_key(DataKey::from_slice(&sealed_key()).unwrap())
            .unwrap();
        assert!(storage.key_loaded());
        storage
            .store_embeddings("alice".into(), vec![vec![1.0]])
            .await
            .unwrap();
        assert!(Storage::file_is_sealed(&data_file));
        let reloaded = Storage::new_with_dir_and_key(
            dir.path(),
            Some(DataKey::from_slice(&sealed_key()).unwrap()),
        )
        .await
        .unwrap();
        assert_eq!(reloaded.get_embeddings("alice").unwrap(), &vec![vec![1.0]]);
    }
}
