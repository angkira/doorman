use anyhow::{Context, Result};
use doorman_shared::{embeddings_path, DATA_DIR};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEmbedding {
    pub username: String,
    pub embedding: Vec<f32>,
    pub enrolled_at: String,
}

pub struct Storage {
    embeddings: HashMap<String, UserEmbedding>,
    file_path: PathBuf,
}

impl Storage {
    /// Initialize storage
    pub async fn new() -> Result<Self> {
        // Ensure data directory exists
        fs::create_dir_all(DATA_DIR)
            .context("Failed to create data directory")?;

        // Set proper permissions (0700 - only root)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = fs::Permissions::from_mode(0o700);
            fs::set_permissions(DATA_DIR, perms)
                .context("Failed to set data directory permissions")?;
        }

        let file_path = embeddings_path();
        
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

    /// Store a user's embedding
    pub async fn store_embedding(&mut self, username: String, embedding: Vec<f32>) -> Result<()> {
        let enrolled_at = chrono::Local::now().to_rfc3339();
        
        let user_embedding = UserEmbedding {
            username: username.clone(),
            embedding,
            enrolled_at,
        };

        self.embeddings.insert(username.clone(), user_embedding);
        self.save_embeddings()?;

        info!("Stored embedding for user: {}", username);
        Ok(())
    }

    /// Get a user's embedding
    pub fn get_embedding(&self, username: &str) -> Option<&Vec<f32>> {
        self.embeddings.get(username).map(|e| &e.embedding)
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

