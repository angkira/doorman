use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod config;
pub use config::{Config, RecognitionConfig};

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
    /// Detect and recognize faces in current frame (for preview)
    DetectAndRecognize,
    /// Get latest cached detection result (fast, no processing)
    GetLatestDetection,
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
    DetectionResult { result: DetectionInfo },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionInfo {
    /// Bounding box if face detected: (x, y, width, height) in pixels
    pub bbox: Option<(u32, u32, u32, u32)>,
    /// Frame dimensions: (width, height) - for proper bbox scaling
    pub frame_size: Option<(u32, u32)>,
    /// Detection confidence score (0.0-1.0)
    pub confidence: Option<f32>,
    /// Recognition result if face was recognized
    pub recognized_user: Option<String>,
    /// Similarity score if recognized (0.0-1.0)
    pub similarity: Option<f32>,
    /// Frame image (JPEG encoded as base64) - only for preview/debug
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_jpeg_base64: Option<String>,
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
///
/// SYSTEM-mode command socket. It lives UNDER `/run/doorman/` (the systemd
/// `RuntimeDirectory=doorman`, owned by the sandboxed `doorman` user) — NOT at
/// `/run/doorman.sock` in root-owned `/run`, which the daemon user cannot
/// create under `ProtectSystem=strict`. This is the single source of truth
/// shared by the daemon (system mode), the PAM module, and the `doorman` CLI.
/// `--user` (dev) mode overrides this with `$XDG_RUNTIME_DIR/doorman.sock`.
pub const SOCKET_PATH: &str = "/run/doorman/doorman.sock";
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

/// Duration to record video during enrollment (in seconds)
pub const ENROLL_DURATION_SECS: u64 = 10;

/// Cosine similarity threshold for face matching (0.0-1.0).
/// Higher value = stricter matching, fewer false positives.
///
/// Tuned for the **EdgeFace-S** recognizer (`edgeface_s.onnx`, CC-BY-NC-SA 4.0,
/// non-commercial) with
/// landmark alignment. Measured on this repo's LFW fixtures (3xA enroll/probe,
/// B & C impostors): genuine A↔A = 0.79, impostor B↔A = -0.06, impostor C↔A =
/// 0.05 — a 0.74 separation gap. 0.4 sits centered in that gap (~0.35 above the
/// top impostor, ~0.39 below the genuine), giving margin for harder genuine
/// pairs (pose/lighting) while still rejecting impostors decisively.
/// Configurable via `authentication.similarity_threshold`.
pub const SIMILARITY_THRESHOLD: f32 = 0.4;

/// Stream message types sent to preview/debug clients
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamMessage {
    /// Detection/recognition result
    Detection {
        timestamp_ms: u64,
        detection: DetectionInfo,
        system_locked: bool,
        processing_time_ms: u32,
    },
    /// Enrollment progress update
    Enrollment {
        timestamp_ms: u64,
        phase: EnrollmentPhase,
        current: usize,
        total: usize,
        username: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnrollmentPhase {
    Recording,
    Processing,
    Selecting,
    Complete,
}

/// Legacy type alias for backwards compatibility during refactoring
pub type DebugStreamMessage = StreamMessage;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Protocol error: {0}")]
    Protocol(String),
}
