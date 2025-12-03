use anyhow::{Context, Result};
use doorman_shared::{StreamMessage, embeddings_path, DATA_DIR};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info};

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
    data_dir: PathBuf,
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
            data_dir,
        })
    }

    /// Initialize storage with default system directory (legacy)
    pub async fn new() -> Result<Self> {
        Self::new_with_dir(DATA_DIR).await
    }

    fn load_embeddings(path: &PathBuf) -> Result<HashMap<String, UserEmbedding>> {
        let data = fs::read(path).context("Failed to read embeddings file")?;
        
        let embeddings: Vec<UserEmbedding> = bincode::deserialize(&data)
            .context("Failed to deserialize embeddings")?;

        let map = embeddings
            .into_iter()
            .map(|e| (e.username.clone(), e))
            .collect();

        Ok(map)
    }

    fn save_embeddings(&self) -> Result<()> {
        let embeddings_vec: Vec<UserEmbedding> = self.embeddings.values().cloned().collect();
        let data = bincode::serialize(&embeddings_vec)
            .context("Failed to serialize embeddings")?;

        fs::write(&self.file_path, data)
            .context("Failed to write embeddings file")?;

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
                num_embeddings: 1, // We store one averaged embedding per user
            })
            .collect()
    }

    /// Get number of enrolled users
    pub fn count(&self) -> usize {
        self.embeddings.len()
    }
}

