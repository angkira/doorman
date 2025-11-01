use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod config;
pub use config::Config;

/// IPC protocol messages between PAM module/CLI and daemon
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Authenticate a user with face recognition
    Authenticate { username: String },
    /// Enroll a new user
    Enroll { username: String },
    /// List enrolled users
    ListUsers,
    /// Remove a user's enrollment
    RemoveUser { username: String },
    /// Get daemon status
    Status,
    /// Shutdown daemon (for debugging)
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Response {
    Success {
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<ResponseData>,
    },
    Failure {
        reason: String,
    },
    Progress {
        message: String,
        current: u32,
        total: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseData {
    UserList { users: Vec<UserInfo> },
    DaemonStatus { info: DaemonInfo },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub username: String,
    pub enrolled_at: String,
    pub num_embeddings: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonInfo {
    pub version: String,
    pub uptime_secs: u64,
    pub camera_available: bool,
    pub models_loaded: bool,
    pub enrolled_users: usize,
}

/// Configuration paths
pub const SOCKET_PATH: &str = "/run/doorman.sock";
pub const DATA_DIR: &str = "/var/lib/doorman";
pub const EMBEDDINGS_FILE: &str = "embeddings.bin";

/// Get the full path to the embeddings file
pub fn embeddings_path() -> PathBuf {
    PathBuf::from(DATA_DIR).join(EMBEDDINGS_FILE)
}

/// Authentication timeout for PAM module (seconds)
pub const AUTH_TIMEOUT_SECS: u64 = 3;

/// Number of frames to capture during authentication
pub const AUTH_FRAMES: usize = 10;

/// Number of frames to capture during enrollment
pub const ENROLL_FRAMES: usize = 20;

/// Cosine similarity threshold for face matching (0.0-1.0)
pub const SIMILARITY_THRESHOLD: f32 = 0.65;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Protocol error: {0}")]
    Protocol(String),
}

