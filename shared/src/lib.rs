use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod config;
pub mod config_check;
pub use config::{Config, DatasetConfig, IpcConfig, RecognitionConfig};

/// Current IPC protocol major version. Requests without a `protocol_version`
/// field (old PAM module / CLI) are treated as this version.
pub const PROTOCOL_VERSION: u32 = 1;

/// IPC protocol messages between PAM module/CLI and daemon
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Authenticate a user with face recognition. `service` and `tty` are the
    /// PAM_SERVICE / PAM_TTY hints of the PAM module; absent for old clients
    /// and the CLI (additive: protocol_version stays 1).
    Authenticate {
        username: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        service: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tty: Option<String>,
    },
    /// Enroll a new user. With `add`, the captured templates are appended to the
    /// user's existing ones without evicting any (absent = false: old clients).
    Enroll {
        username: String,
        #[serde(default)]
        add: bool,
    },
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
    /// Hold the connection until `username` is recognized (same decision as
    /// `Authenticate`), the (capped) timeout expires, or the session is not
    /// armed. Answered with exactly one terminal `Matched` or `NoMatch`.
    WaitForMatch { username: String, timeout_ms: u64 },
}

/// Wire envelope for a request: the `Request` fields stay at the top level of
/// the JSON object (flattened), with an optional `protocol_version` beside them,
/// so the old wire format `{"type":"authenticate","username":"x"}` still parses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestEnvelope {
    /// Absent means [`PROTOCOL_VERSION`] 1 (clients that predate the field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_version: Option<u32>,
    #[serde(flatten)]
    pub request: Request,
}

impl RequestEnvelope {
    /// Wrap a request, stamping the current protocol version.
    pub fn new(request: Request) -> Self {
        Self {
            protocol_version: Some(PROTOCOL_VERSION),
            request,
        }
    }
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
    /// The peer is not authorized for this request (peer-credential policy).
    Denied {
        reason: String,
    },
    /// Too many authenticate attempts from this peer uid.
    RateLimited {
        retry_after_s: u64,
    },
    /// The request's `protocol_version` major is not supported by the daemon.
    UnsupportedProtocol {
        reason: String,
        supported_version: u32,
    },
    /// Terminal answer to `WaitForMatch`: the user passed the full
    /// authentication decision (quality gate, liveness, threshold).
    Matched {
        username: String,
        similarity: f32,
        /// Unix time of the decision, milliseconds.
        ts_unix_ms: u64,
    },
    /// Terminal answer to `WaitForMatch`: no match was made.
    NoMatch {
        reason: NoMatchReason,
    },
}

/// Why a `WaitForMatch` ended without a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoMatchReason {
    /// The wait expired while the session was armed (or not lock-gated).
    Timeout,
    /// The session was not locked and awake for the whole wait.
    NotArmed,
    /// The user has no enrollment; waiting longer cannot help.
    NotEnrolled,
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
    /// Loaded ONNX Runtime linkage and version; absent from old daemons and
    /// from builds without an ORT backend. Additive (protocol_version stays 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ort_runtime: Option<String>,
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

/// Duration to record video during enrollment (in seconds). Longer = more frames
/// across which to vary pose/lighting, so the diverse-embedding selection has a
/// richer pool to draw a capacious template from.
pub const ENROLL_DURATION_SECS: u64 = 15;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_daemon_info_without_ort_runtime_parses() {
        let info: DaemonInfo = serde_json::from_str(
            r#"{"version":"0.1.0","uptime_secs":1,"camera_available":true,"models_loaded":true,"enrolled_users":2}"#,
        )
        .unwrap();
        assert_eq!(info.ort_runtime, None);
    }

    #[test]
    fn legacy_request_without_version_parses() {
        let env: RequestEnvelope =
            serde_json::from_str(r#"{"type":"authenticate","username":"bob"}"#).unwrap();
        assert_eq!(env.protocol_version, None);
        assert!(
            matches!(env.request, Request::Authenticate { ref username, service: None, tty: None } if username == "bob")
        );
    }

    #[test]
    fn enroll_add_defaults_to_false_for_old_clients() {
        let old: Request = serde_json::from_str(r#"{"type":"enroll","username":"bob"}"#).unwrap();
        assert!(matches!(old, Request::Enroll { add: false, .. }));
        let new: Request =
            serde_json::from_str(r#"{"type":"enroll","username":"bob","add":true}"#).unwrap();
        assert!(matches!(new, Request::Enroll { add: true, .. }));
    }

    #[test]
    fn wait_for_match_wire_format() {
        let env: RequestEnvelope = serde_json::from_str(
            r#"{"type":"wait_for_match","username":"bob","timeout_ms":5000,"protocol_version":1}"#,
        )
        .unwrap();
        assert!(matches!(
            env.request,
            Request::WaitForMatch { ref username, timeout_ms: 5000 } if username == "bob"
        ));

        let matched = Response::Matched {
            username: "bob".into(),
            similarity: 0.5,
            ts_unix_ms: 7,
        };
        assert_eq!(
            serde_json::to_string(&matched).unwrap(),
            r#"{"status":"matched","username":"bob","similarity":0.5,"ts_unix_ms":7}"#
        );
        let no_match = Response::NoMatch {
            reason: NoMatchReason::NotArmed,
        };
        assert_eq!(
            serde_json::to_string(&no_match).unwrap(),
            r#"{"status":"no_match","reason":"not_armed"}"#
        );
        let back: Response =
            serde_json::from_str(r#"{"status":"no_match","reason":"timeout"}"#).unwrap();
        assert!(matches!(
            back,
            Response::NoMatch {
                reason: NoMatchReason::Timeout
            }
        ));
    }

    #[test]
    fn unit_variant_with_version_parses() {
        let env: RequestEnvelope =
            serde_json::from_str(r#"{"type":"status","protocol_version":2}"#).unwrap();
        assert_eq!(env.protocol_version, Some(2));
        assert!(matches!(env.request, Request::Status));
    }

    #[test]
    fn new_envelope_stamps_version_and_keeps_flat_layout() {
        let json = serde_json::to_string(&RequestEnvelope::new(Request::Authenticate {
            username: "bob".into(),
            service: None,
            tty: None,
        }))
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "authenticate");
        assert_eq!(v["username"], "bob");
        assert_eq!(v["protocol_version"], PROTOCOL_VERSION);
    }

    #[test]
    fn authenticate_service_and_tty_are_additive() {
        let env: RequestEnvelope = serde_json::from_str(
            r#"{"type":"authenticate","username":"bob","service":"sudo","tty":"/dev/pts/3","protocol_version":1}"#,
        )
        .unwrap();
        assert!(matches!(
            env.request,
            Request::Authenticate { ref service, ref tty, .. }
                if service.as_deref() == Some("sudo") && tty.as_deref() == Some("/dev/pts/3")
        ));
        // Without hints the serialized form is the old wire format.
        let json = serde_json::to_string(&Request::Authenticate {
            username: "bob".into(),
            service: None,
            tty: None,
        })
        .unwrap();
        assert_eq!(json, r#"{"type":"authenticate","username":"bob"}"#);
    }

    #[test]
    fn new_response_variants_roundtrip() {
        for r in [
            Response::Denied { reason: "x".into() },
            Response::RateLimited { retry_after_s: 7 },
            Response::UnsupportedProtocol {
                reason: "x".into(),
                supported_version: PROTOCOL_VERSION,
            },
        ] {
            let back: Response = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
            assert_eq!(format!("{r:?}"), format!("{back:?}"));
        }
    }
}
