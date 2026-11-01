use anyhow::{Context, Result};
use doorman_shared::{DATA_DIR};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info};

/// First bytes of a versioned `embeddings.bin`. The legacy (version 1) file has
/// no header: it starts with the bincode `u64` element count, which can never
/// equal these bytes, so the two layouts are told apart unambiguously.
const FILE_MAGIC: &[u8; 8] = b"DRMNEMB\0";
/// Layout version written by this build: magic, `u32` little-endian version,
/// then the bincode-encoded `Vec<UserEmbedding>`.
const FORMAT_VERSION: u32 = 2;
const HEADER_LEN: usize = FILE_MAGIC.len() + std::mem::size_of::<u32>();

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
        fs::create_dir_all(&data_dir)
            .context("Failed to create data directory")?;

        // Set proper permissions (0700 for user, or current perms)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // Don't change permissions in user mode - just use defaults
            // Only set 0700 if running as root
            if nix::unistd::getuid().is_root() {
                let perms = fs::Permissions::from_mode(0o700);
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
    fn decode_embeddings(data: &[u8]) -> Result<Vec<UserEmbedding>> {
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

    fn encode_embeddings(users: &[UserEmbedding]) -> Result<Vec<u8>> {
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

    fn save_embeddings(&self) -> Result<()> {
        let embeddings_vec: Vec<UserEmbedding> = self.embeddings.values().cloned().collect();
        let data = Self::encode_embeddings(&embeddings_vec)?;

        // Write a sibling file and rename it over the target, so a crash while
        // upgrading a legacy file never leaves a truncated embeddings.bin.
        let tmp_path = self.file_path.with_extension("bin.tmp");
        fs::write(&tmp_path, data).context("Failed to write embeddings file")?;
        fs::rename(&tmp_path, &self.file_path).context("Failed to replace embeddings file")?;

        // Set restrictive permissions (0600 - only root read/write)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = fs::Permissions::from_mode(0o600);
            fs::set_permissions(&self.file_path, perms)
                .context("Failed to set embeddings file permissions")?;
        }

        debug!("Embeddings saved to {:?}", self.file_path);
        Ok(())
    }

    /// Store a user's embeddings (multiple variations)
    pub async fn store_embeddings(&mut self, username: String, embeddings: Vec<Vec<f32>>) -> Result<()> {
        let enrolled_at = chrono::Local::now().to_rfc3339();
        
        let user_embedding = UserEmbedding {
            username: username.clone(),
            embeddings,
            enrolled_at,
        };

        let num_embeddings = user_embedding.embeddings.len();
        self.embeddings.insert(username.clone(), user_embedding);
        self.save_embeddings()?;

        info!("Stored {} embeddings for user: {}", num_embeddings, username);
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
        let err = Storage::decode_embeddings(FILE_MAGIC).unwrap_err().to_string();
        assert!(err.contains("truncated"), "{err}");
    }

    #[tokio::test]
    async fn legacy_file_upgrades_on_first_save_and_reloads() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("embeddings.bin"), legacy_bytes()).unwrap();

        let mut storage = Storage::new_with_dir(dir.path()).await.unwrap();
        assert_eq!(storage.get_embeddings("alice").unwrap().len(), 2);
        // Reading alone does not rewrite the file.
        assert_eq!(fs::read(dir.path().join("embeddings.bin")).unwrap(), legacy_bytes());

        storage
            .store_embeddings("alice".into(), vec![vec![0.0, 0.0, 1.0]])
            .await
            .unwrap();
        let on_disk = fs::read(dir.path().join("embeddings.bin")).unwrap();
        assert!(on_disk.starts_with(FILE_MAGIC));

        let reloaded = Storage::new_with_dir(dir.path()).await.unwrap();
        assert_eq!(reloaded.get_embeddings("alice").unwrap(), &vec![vec![0.0, 0.0, 1.0]]);
    }
}
