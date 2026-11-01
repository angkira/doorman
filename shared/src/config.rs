use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(#[from] toml::de::Error),

    #[error("Config not found at any standard location")]
    NotFound,

    #[error("Invalid config: {0}")]
    Invalid(String),
}

/// Upper bound (inclusive) for `[camera] warmup_frames`.
pub const MAX_WARMUP_FRAMES: u32 = 10;
/// Upper bound (inclusive, ms) for `[camera] prearm_window_ms`.
pub const MAX_PREARM_WINDOW_MS: u32 = 30_000;
/// Default `[decision] accept_n`: frames that must clear the threshold.
pub const DEFAULT_ACCEPT_N: usize = 3;
/// Default `[decision] window_m`: size of the sliding window of recent frames.
pub const DEFAULT_WINDOW_M: usize = 5;

fn default_prearm_window_ms() -> u32 {
    5000
}

/// Accepted `[camera] fourcc` spellings (case-insensitive) for MJPEG.
const FOURCC_MJPEG_NAMES: [&str; 2] = ["MJPG", "MJPEG"];
/// Accepted `[camera] fourcc` spellings (case-insensitive) for YUYV.
const FOURCC_YUYV_NAMES: [&str; 2] = ["YUYV", "YUY2"];

/// Pixel format requested from the camera via `[camera] fourcc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraFourcc {
    Mjpeg,
    Yuyv,
}

impl CameraFourcc {
    /// Parse a user-supplied fourcc string.
    pub fn parse(s: &str) -> Result<Self, ConfigError> {
        if FOURCC_MJPEG_NAMES.iter().any(|n| n.eq_ignore_ascii_case(s)) {
            Ok(Self::Mjpeg)
        } else if FOURCC_YUYV_NAMES.iter().any(|n| n.eq_ignore_ascii_case(s)) {
            Ok(Self::Yuyv)
        } else {
            Err(ConfigError::Invalid(format!(
                "camera.fourcc = {s:?} is not supported (use \"MJPG\" or \"YUYV\")"
            )))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub daemon: DaemonConfig,

    #[serde(default)]
    pub camera: CameraConfig,

    #[serde(default)]
    pub ml: MLConfig,

    #[serde(default)]
    pub authentication: AuthConfig,

    #[serde(default)]
    pub enrollment: EnrollmentConfig,

    #[serde(default)]
    pub preprocessing: PreprocessingConfig,

    #[serde(default)]
    pub models: ModelsConfig,

    #[serde(default)]
    pub recognition: RecognitionConfig,

    #[serde(default)]
    pub dataset: DatasetConfig,

    #[serde(default)]
    pub ipc: IpcConfig,

    #[serde(default)]
    pub telemetry: TelemetryConfig,

    #[serde(default)]
    pub decision: DecisionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    #[serde(default = "default_socket_path")]
    pub socket_path: String,

    #[serde(default = "default_data_dir")]
    pub data_dir: String,

    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Debug stream socket for preview (sends detection data)
    #[serde(default = "default_debug_socket")]
    pub debug_socket: String,

    /// Frame stream socket for preview (sends raw JPEG frames)
    #[serde(default = "default_frame_socket")]
    pub frame_socket: String,

    /// Processing FPS for continuous face detection (independent of camera FPS)
    #[serde(default = "default_processing_fps")]
    pub processing_fps: u32,

    /// Run as user service (not root) - enables PipeWire/GStreamer camera access
    #[serde(default)]
    pub user_mode: bool,

    /// Debug mode: process frames even when system is unlocked (for preview/testing)
    #[serde(default)]
    pub debug_mode: bool,

    /// Preview mode: stream frames to preview clients (enables frame_socket)
    #[serde(default)]
    pub preview_mode: bool,

    /// Initial system lock state at daemon startup.
    ///
    /// In a real deployment the daemon runs as an always-on system service and
    /// the screen is locked until a face is recognized, so this defaults to
    /// `true`. The background recognition pipeline only processes frames while
    /// locked (or in `debug_mode`), and clears the lock on a match.
    ///
    /// PAM authentication does NOT depend on this flag: an `Authenticate`
    /// request always triggers a fresh on-demand capture + match regardless of
    /// lock state. The flag only gates the autonomous background unlock loop.
    ///
    /// Set `false` (or pass `--start-unlocked`) for dev/preview so the pipeline
    /// keeps running without auto-locking the session.
    #[serde(default = "default_start_locked")]
    pub start_locked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    #[serde(default)]
    pub device_index: u32,

    pub video_file: Option<String>,

    #[serde(default = "default_camera_width")]
    pub width: u32,

    #[serde(default = "default_camera_height")]
    pub height: u32,

    #[serde(default = "default_fps")]
    pub fps: u32,

    /// Super-security: keep the camera capturing ALWAYS, not only while the screen
    /// is locked. Default `false` — the camera (and its idle CPU) is gated to the
    /// armed state (screen locked) / an attached preview client, so an unlocked
    /// desktop pays ~0 for it. Set `true` for continuous presence tracking.
    #[serde(default)]
    pub always_track: bool,

    /// FUTURE (super-security, requires `always_track = true`): auto-lock the
    /// session after this many seconds with NO recognized face present, so the box
    /// locks itself when you walk away. `0` = disabled. The monitoring loop is not
    /// wired yet — this is the config hook the feature will read.
    #[serde(default)]
    pub presence_autolock_secs: u64,

    /// Pixel format to request from the camera: "MJPG" or "YUYV". Unset keeps
    /// the historical behaviour (YUYV at the configured size; with the native
    /// backend a size the device only offers as MJPEG falls back to MJPEG).
    /// For the ffmpeg backend "MJPG" adds `-input_format mjpeg`.
    #[serde(default)]
    pub fourcc: Option<String>,

    /// Frames to discard after a cold camera open, while auto-exposure settles.
    /// `0` (default) keeps every frame. Valid range 0..=10; 3 is recommended
    /// once measured on the host (see docs/camera.md).
    #[serde(default)]
    pub warmup_frames: u32,

    /// Start opening the camera as soon as the session reports `LockedHint`,
    /// without waiting for the display to be non-idle. The camera is still
    /// closed whenever the session is unlocked. Default `false`; `true` is
    /// recommended (see docs/camera.md).
    #[serde(default)]
    pub prearm_on_lock: bool,

    /// With `prearm_on_lock`, how long after the lock event (LockedHint
    /// false->true) the camera may stay open while the display is still idle.
    /// After the window the normal `locked && !idle` rule applies. 0..=30000.
    #[serde(default = "default_prearm_window_ms")]
    pub prearm_window_ms: u32,
}

impl CameraConfig {
    /// Reject out-of-range or unsupported `[camera]` values with an explicit error.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.warmup_frames > MAX_WARMUP_FRAMES {
            return Err(ConfigError::Invalid(format!(
                "camera.warmup_frames = {} is out of range (0..={MAX_WARMUP_FRAMES})",
                self.warmup_frames
            )));
        }
        if self.prearm_window_ms > MAX_PREARM_WINDOW_MS {
            return Err(ConfigError::Invalid(format!(
                "camera.prearm_window_ms = {} is out of range (0..={MAX_PREARM_WINDOW_MS})",
                self.prearm_window_ms
            )));
        }
        self.fourcc()?;
        Ok(())
    }

    /// The parsed `fourcc` request, `None` when unset.
    pub fn fourcc(&self) -> Result<Option<CameraFourcc>, ConfigError> {
        self.fourcc.as_deref().map(CameraFourcc::parse).transpose()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    #[serde(default = "default_models_dir")]
    pub models_dir: String,

    #[serde(default = "default_backend")]
    pub backend: String,

    #[serde(default = "default_device")]
    pub device: String,

    #[serde(default)]
    pub cpu_threads: i32,

    #[serde(default)]
    pub gpu_device_id: i32,

    /// Docker inference service endpoint (for backend = "docker")
    #[serde(default)]
    pub docker_endpoint: Option<String>,

    /// Unix socket path for ML inference (for backend = "socket")
    #[serde(default)]
    pub socket_path: Option<String>,

    /// ROCm/HIP GPU to bind to, selected by its STABLE UUID (e.g.
    /// "GPU-31119c444c4c0950" as printed by `rocminfo`/`rocm-smi`). When set, the
    /// daemon exports `ROCR_VISIBLE_DEVICES=<uuid>` before any HIP init so ONLY
    /// that GPU is ever visible — index-independent, so it can never accidentally
    /// select a different card than you intend (e.g. a dGPU you keep off-limits).
    /// Leave empty to use whatever `ROCR_VISIBLE_DEVICES`/default ordering the
    /// environment provides. Only meaningful when `device` is a GPU backend.
    #[serde(default)]
    pub gpu_uuid: Option<String>,

    /// Optional `HSA_OVERRIDE_GFX_VERSION` value (e.g. "11.0.3" for the Radeon
    /// 780M / gfx1103). When set, the daemon exports it before HIP init so an
    /// unsupported gfx target reports as a supported one. Leave empty to use the
    /// environment value (set by systemd) or the built-in best-effort default.
    #[serde(default)]
    pub hsa_gfx_version: Option<String>,

    /// ncnn/Vulkan device name substring (e.g. "RADV PHOENIX") to bind the iGPU
    /// depth backend to; empty = first integrated GPU (never the discrete dGPU).
    #[serde(default)]
    pub vulkan_device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,

    #[serde(default = "default_auth_frames")]
    pub auth_frames: usize,

    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// Run the depth-relief liveness (anti-spoofing) check during recognition.
    /// Default: `true`. When enabled the check is **FATAL**: a face whose
    /// monocular-depth relief falls below `liveness_depth_threshold` is treated
    /// as a spoof and authentication FAILS (the frame is rejected as if no valid
    /// face were present). Set `false` to skip liveness inference entirely.
    #[serde(default = "default_liveness_enabled")]
    pub liveness_enabled: bool,

    /// Minimum monocular-depth face relief required to accept a face as a live
    /// 3D face. The relief score is `std(face_depth) / (depth_range + eps)`
    /// computed from Depth-Anything-V2 over the detected face bbox: a real 3D
    /// face has high relief, a flat phone/print replay has near-zero relief.
    /// Calibrated through the daemon at its native capture resolution
    /// (1024x720) on in-situ genuine vs screen-replay captures to give APCER=0
    /// (no spoof passes) with margin. Only used when `liveness_enabled = true`.
    #[serde(default = "default_liveness_depth_threshold")]
    pub liveness_depth_threshold: f32,

    /// Proactive auto-unlock: when `true`, the lock-gated recognition pipeline
    /// (which runs only while the screen is locked and the display is on) will
    /// call logind `Session.Unlock()` itself once a face is RECOGNIZED at or
    /// above `similarity_threshold` AND passes the FATAL depth-liveness gate —
    /// no button press required. After a successful unlock it disarms until the
    /// next `Lock` signal (one-shot, no repeat-unlock spam).
    ///
    /// Default: `false` (safe for the public repo). When `false`, the lock
    /// supervisor still GATES the pipeline on lock/idle but never unlocks; the
    /// password / PAM path remains the only unlock trigger. Enabling it also
    /// requires a polkit rule granting the daemon user the
    /// `org.freedesktop.login1.unlock-sessions` action (installed separately).
    #[serde(default = "default_proactive_unlock")]
    pub proactive_unlock: bool,

    /// Continuous improvement (OPT-IN): after a face is recognized AND passes the
    /// FATAL depth-liveness gate, fold that embedding into the user's template so
    /// recognition adapts to changing lighting/angle over time.
    ///
    /// Default `true` — the AUTHENTICATION-DRIFT risk (repeatedly adapting on
    /// borderline matches walking the stored identity toward a persistent look-alike)
    /// is BOUNDED by the hardening below; set `false` to disable adaptation entirely.
    /// Hardening — adaptation ONLY happens on a HIGH-confidence match
    /// (cosine >= max(0.85, similarity_threshold + 0.05), well above the auth
    /// threshold), the original enrolled embeddings are APPEND-only-anchored (never
    /// evicted by adaptation), the gallery is hard-capped, near-duplicates are
    /// skipped, and a spoof/wrong face can't enter (liveness + recognition both gate
    /// it). Enable only if you understand and accept the drift trade-off.
    #[serde(default = "default_adaptive_enroll")]
    pub adaptive_enroll: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrollmentConfig {
    #[serde(default = "default_enroll_frames")]
    pub enroll_frames: usize,

    #[serde(default = "default_min_valid_frames")]
    pub min_valid_frames: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    #[serde(default = "default_image_size")]
    pub image_width: u32,

    #[serde(default = "default_image_size")]
    pub image_height: u32,

    #[serde(default = "default_filter_type")]
    pub filter_type: String,
}

/// Phase 1 recognition-stability tuning.
///
/// All fields are config-driven with sane defaults so existing `doorman.toml`
/// files keep working unchanged. These control multi-frame embedding
/// aggregation, preview box-color hysteresis and the pre-embedding frame-quality
/// gate. None of them touch the recognizer/detector models or the
/// `authentication.similarity_threshold` value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionConfig {
    /// Number of recent quality-passed per-frame embeddings to average (rolling
    /// window) before scoring against the stored template. Averaging
    /// L2-normalized embeddings suppresses per-frame noise and stabilizes the
    /// genuine cosine. `1` reproduces the old per-frame behavior.
    #[serde(default = "default_aggregation_window")]
    pub aggregation_window: usize,

    /// Preview box-color hysteresis: require this many consecutive aggregated
    /// results on the opposite side of the threshold before flipping the
    /// RECOGNIZED <-> unknown decision. Prevents the preview box from flickering.
    #[serde(default = "default_hysteresis_frames")]
    pub hysteresis_frames: u32,

    /// Frame-quality gate: minimum sharpness (variance of the Laplacian computed
    /// on the aligned/crop 112x112 grayscale). Frames below this are skipped
    /// (not added to the aggregation window, not enrolled). `0.0` disables the
    /// sharpness gate.
    #[serde(default = "default_min_sharpness")]
    pub min_sharpness: f32,

    /// Frame-quality gate: minimum face bbox area as a fraction of the full
    /// frame area. Rejects tiny / far-away faces. `0.0` disables the size gate.
    #[serde(default = "default_min_face_area_frac")]
    pub min_face_area_frac: f32,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            aggregation_window: default_aggregation_window(),
            hysteresis_frames: default_hysteresis_frames(),
            min_sharpness: default_min_sharpness(),
            min_face_area_frac: default_min_face_area_frac(),
        }
    }
}

/// How the per-frame evidence becomes an accept/reject decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionMode {
    /// Collect `auth_frames` embeddings, average them, score once (original behaviour).
    Mean,
    /// Score every frame; accept as soon as `accept_n` of the last `window_m`
    /// frames reach the threshold (and liveness passed).
    NOfM,
}

/// How one embedding is scored against a user's stored templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemplateScore {
    /// Highest cosine over the stored templates (original behaviour).
    Max,
    /// Mean of the three highest template cosines.
    Top3Mean,
    /// Cosine against the renormalised mean of all templates.
    Centroid,
}

/// `[decision]`: decision rule and template scoring. Defaults reproduce the
/// original behaviour; see docs/decision-and-liveness.md.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionConfig {
    #[serde(default = "default_decision_mode")]
    pub mode: DecisionMode,

    /// `n_of_m` only: frames that must be at or above the threshold.
    #[serde(default = "default_accept_n")]
    pub accept_n: usize,

    /// `n_of_m` only: size of the sliding window of most recent quality frames.
    #[serde(default = "default_window_m")]
    pub window_m: usize,

    #[serde(default = "default_template_score")]
    pub template_score: TemplateScore,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            mode: default_decision_mode(),
            accept_n: default_accept_n(),
            window_m: default_window_m(),
            template_score: default_template_score(),
        }
    }
}

impl DecisionConfig {
    /// Reject `accept_n`/`window_m` outside `1 <= N <= M` with an explicit error.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.accept_n < 1 || self.accept_n > self.window_m {
            return Err(ConfigError::Invalid(format!(
                "decision.accept_n = {} and decision.window_m = {} must satisfy 1 <= accept_n <= window_m",
                self.accept_n, self.window_m
            )));
        }
        Ok(())
    }
}

/// Default per-uid Authenticate budget per sliding minute.
pub const DEFAULT_MAX_AUTH_ATTEMPTS_PER_MINUTE: u32 = 20;
/// Default cap for a single `WaitForMatch` (milliseconds).
pub const DEFAULT_WAIT_FOR_MATCH_MAX_MS: u64 = 60_000;

/// IPC peer-credential policy (`[ipc]`). See docs/ipc-security.md.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcConfig {
    /// Extra uids allowed to Authenticate for ANY user (besides root and the
    /// target user themself). Default empty.
    #[serde(default)]
    pub trusted_uids: Vec<u32>,

    /// Extra uids allowed to run admin operations (Enroll, RemoveUser, ListUsers,
    /// DetectAndRecognize, Shutdown, ...) besides root. Default empty.
    #[serde(default)]
    pub admin_uids: Vec<u32>,

    /// Per-peer-uid Authenticate attempts allowed per sliding 60 s window.
    /// `0` disables rate limiting.
    #[serde(default = "default_max_auth_attempts_per_minute")]
    pub max_auth_attempts_per_minute: u32,

    /// If set, the frame/debug stream sockets are chowned to this group and made
    /// 0660; otherwise they are 0600 (owner/root only).
    #[serde(default)]
    pub stream_socket_group: Option<String>,

    /// Upper bound for the `timeout_ms` of one `WaitForMatch`; larger requests
    /// are clamped to it.
    #[serde(default = "default_wait_for_match_max_ms")]
    pub wait_for_match_max_ms: u64,
}

fn default_wait_for_match_max_ms() -> u64 {
    DEFAULT_WAIT_FOR_MATCH_MAX_MS
}

fn default_max_auth_attempts_per_minute() -> u32 {
    DEFAULT_MAX_AUTH_ATTEMPTS_PER_MINUTE
}

impl Default for IpcConfig {
    fn default() -> Self {
        Self {
            trusted_uids: Vec::new(),
            admin_uids: Vec::new(),
            max_auth_attempts_per_minute: DEFAULT_MAX_AUTH_ATTEMPTS_PER_MINUTE,
            stream_socket_group: None,
            wait_for_match_max_ms: DEFAULT_WAIT_FOR_MATCH_MAX_MS,
        }
    }
}

/// Attempt-log dataset (R&D milestone M1). When enabled, the daemon appends one
/// embeddings-only record per scored unlock attempt under `<data_dir>/dataset/`, to
/// seed the future learnable depth projector / fusion / self-training work. SEPARATE
/// from the live gallery; never read on the auth hot path. Default OFF — opt-in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Master switch. When `false` (default) NOTHING is written and the auth path
    /// is byte-for-byte unaffected.
    #[serde(default)]
    pub logging_enabled: bool,

    /// FUTURE hook (not wired yet): also persist the aligned raw frame alongside the
    /// embedding for offline depth-projector training. Default `false` and intended
    /// to stay off on the deployed box — embeddings-only is the privacy-preserving
    /// path; raw frames must NEVER reach the public repo.
    #[serde(default)]
    pub keep_raw_frames: bool,

    /// Legacy cap for generic raw capture bundles.
    /// Only used when `keep_raw_frames=true`.
    #[serde(default = "default_raw_frame_bundle_size")]
    pub raw_frame_bundle_size: usize,

    /// Maximum raw capture bundles written per day. `0` disables bundle writes
    /// even when `keep_raw_frames=true`.
    #[serde(default = "default_max_capture_packs_per_day")]
    pub max_capture_packs_per_day: usize,

    /// JPEG quality for opt-in raw capture bundles.
    #[serde(default = "default_raw_frame_jpeg_quality")]
    pub raw_frame_jpeg_quality: u8,

    /// Maximum attempt records written to one daily shard. `0` disables attempt
    /// writes even when logging is enabled. Labels are separate and are not capped
    /// here because they are low-volume ground-truth events.
    #[serde(default = "default_dataset_max_attempt_records_per_day")]
    pub max_attempt_records_per_day: usize,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            logging_enabled: false,
            keep_raw_frames: false,
            raw_frame_bundle_size: default_raw_frame_bundle_size(),
            max_capture_packs_per_day: default_max_capture_packs_per_day(),
            raw_frame_jpeg_quality: default_raw_frame_jpeg_quality(),
            max_attempt_records_per_day: default_dataset_max_attempt_records_per_day(),
        }
    }
}

/// Per-attempt timing telemetry (workstream WS6). Default OFF.
///
/// When `timing_log` is enabled the daemon appends ONE JSON line per
/// authentication attempt to `timing_log_path`. A line holds stage durations,
/// frame counters, the outcome, and the two calibration values (final cosine
/// similarity and depth-liveness relief). It never holds usernames, uids,
/// frames or embeddings. The file is rotated to `<path>.1` (one generation
/// kept) once it exceeds `timing_log_max_bytes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub timing_log: bool,

    #[serde(default = "default_timing_log_path")]
    pub timing_log_path: String,

    #[serde(default = "default_timing_log_max_bytes")]
    pub timing_log_max_bytes: u64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            timing_log: false,
            timing_log_path: default_timing_log_path(),
            timing_log_max_bytes: default_timing_log_max_bytes(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    // Face detector (BlazeFace)
    #[serde(default = "default_detector_input_width")]
    pub detector_input_width: u32,

    #[serde(default = "default_detector_input_height")]
    pub detector_input_height: u32,

    #[serde(default = "default_detector_num_anchors")]
    pub detector_num_anchors: usize,

    // Liveness detector
    #[serde(default = "default_liveness_input_width")]
    pub liveness_input_width: u32,

    #[serde(default = "default_liveness_input_height")]
    pub liveness_input_height: u32,

    // Face recognizer (MobileFaceNet)
    #[serde(default = "default_recognizer_input_width")]
    pub recognizer_input_width: u32,

    #[serde(default = "default_recognizer_input_height")]
    pub recognizer_input_height: u32,

    #[serde(default = "default_recognizer_embedding_size")]
    pub recognizer_embedding_size: usize,
}

// Default values
// SYSTEM-mode socket paths live under /run/doorman/ — the systemd
// RuntimeDirectory=doorman, owned by the sandboxed `doorman` user. Placing them
// in root-owned /run (e.g. /run/doorman.sock) makes the daemon fail to bind
// under ProtectSystem=strict. `--user` (dev) mode overrides these with XDG paths.
fn default_socket_path() -> String {
    "/run/doorman/doorman.sock".to_string()
}
fn default_debug_socket() -> String {
    "/run/doorman/doorman-debug.sock".to_string()
}
fn default_frame_socket() -> String {
    "/run/doorman/doorman-frames.sock".to_string()
}
fn default_timing_log_path() -> String {
    "/run/doorman/timing.jsonl".to_string()
}
fn default_timing_log_max_bytes() -> u64 {
    1024 * 1024
}
fn default_processing_fps() -> u32 {
    10
}
fn default_start_locked() -> bool {
    true
}
fn default_data_dir() -> String {
    "/var/lib/doorman".to_string()
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_models_dir() -> String {
    "/var/lib/doorman/models".to_string()
}
fn default_backend() -> String {
    "ort".to_string()
}
fn default_device() -> String {
    "cpu".to_string()
}
fn default_similarity_threshold() -> f32 {
    0.4
}
fn default_auth_frames() -> usize {
    10
}
fn default_liveness_enabled() -> bool {
    true
}
fn default_proactive_unlock() -> bool {
    false
}
fn default_adaptive_enroll() -> bool {
    true
}
// Calibrated through the daemon at 1024x720 on in-situ genuine vs screen-replay
// captures (see .session/2026-06-08_liveness-wired.md). On the working
// (CPU-EP) depth path the relief separates cleanly: genuine 0.068..0.245
// (n=139), screen-replay spoof 0.002..0.019 (n=86). Any threshold in
// [0.019, 0.068] gives APCER=0 (no spoof passes) AND BPCER=0 (no genuine
// rejected). 0.05 sits with margin on both sides (2.7x over spoof max, 1.36x
// under genuine min), leaning safe for MAX-security.
fn default_liveness_depth_threshold() -> f32 {
    0.05
}
fn default_timeout_secs() -> u64 {
    3
}
fn default_decision_mode() -> DecisionMode {
    DecisionMode::Mean
}
fn default_accept_n() -> usize {
    DEFAULT_ACCEPT_N
}
fn default_window_m() -> usize {
    DEFAULT_WINDOW_M
}
fn default_template_score() -> TemplateScore {
    TemplateScore::Max
}
fn default_enroll_frames() -> usize {
    20
}
fn default_min_valid_frames() -> usize {
    5
}
fn default_aggregation_window() -> usize {
    5
}
fn default_hysteresis_frames() -> u32 {
    3
}
// Sharpness = variance of Laplacian on 112x112 grayscale. Sharp aligned faces
// measure well above this; obvious motion/Gaussian blur falls below it. Kept
// conservative so synthesized/real faces pass while clearly blurred frames fail.
fn default_min_sharpness() -> f32 {
    8.0
}
fn default_min_face_area_frac() -> f32 {
    0.02
}
fn default_dataset_max_attempt_records_per_day() -> usize {
    50_000
}
fn default_raw_frame_bundle_size() -> usize {
    10
}
fn default_max_capture_packs_per_day() -> usize {
    500
}
fn default_raw_frame_jpeg_quality() -> u8 {
    92
}
fn default_image_size() -> u32 {
    256
}
fn default_filter_type() -> String {
    "lanczos3".to_string()
}
fn default_camera_width() -> u32 {
    1280
}
fn default_camera_height() -> u32 {
    720
}
fn default_fps() -> u32 {
    30
}

// Model default values
fn default_detector_input_width() -> u32 {
    128
}
fn default_detector_input_height() -> u32 {
    128
}
fn default_detector_num_anchors() -> usize {
    896
}
fn default_liveness_input_width() -> u32 {
    80
}
fn default_liveness_input_height() -> u32 {
    80
}
fn default_recognizer_input_width() -> u32 {
    112
}
fn default_recognizer_input_height() -> u32 {
    112
}
fn default_recognizer_embedding_size() -> usize {
    128
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            socket_path: default_socket_path(),
            data_dir: default_data_dir(),
            log_level: default_log_level(),
            debug_socket: default_debug_socket(),
            frame_socket: default_frame_socket(),
            processing_fps: default_processing_fps(),
            user_mode: false,
            debug_mode: false,
            preview_mode: false,
            start_locked: default_start_locked(),
        }
    }
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            video_file: None,
            width: default_camera_width(),
            height: default_camera_height(),
            fps: default_fps(),
            always_track: false,
            presence_autolock_secs: 0,
            fourcc: None,
            warmup_frames: 0,
            prearm_on_lock: false,
            prearm_window_ms: default_prearm_window_ms(),
        }
    }
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            backend: default_backend(),
            device: default_device(),
            cpu_threads: 0,
            gpu_device_id: 0,
            docker_endpoint: None,
            socket_path: None,
            gpu_uuid: None,
            hsa_gfx_version: None,
            vulkan_device: None,
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: default_similarity_threshold(),
            auth_frames: default_auth_frames(),
            timeout_secs: default_timeout_secs(),
            liveness_enabled: default_liveness_enabled(),
            liveness_depth_threshold: default_liveness_depth_threshold(),
            proactive_unlock: default_proactive_unlock(),
            adaptive_enroll: default_adaptive_enroll(),
        }
    }
}

impl Default for EnrollmentConfig {
    fn default() -> Self {
        Self {
            enroll_frames: default_enroll_frames(),
            min_valid_frames: default_min_valid_frames(),
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            image_width: default_image_size(),
            image_height: default_image_size(),
            filter_type: default_filter_type(),
        }
    }
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            detector_input_width: default_detector_input_width(),
            detector_input_height: default_detector_input_height(),
            detector_num_anchors: default_detector_num_anchors(),
            liveness_input_width: default_liveness_input_width(),
            liveness_input_height: default_liveness_input_height(),
            recognizer_input_width: default_recognizer_input_width(),
            recognizer_input_height: default_recognizer_input_height(),
            recognizer_embedding_size: default_recognizer_embedding_size(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            daemon: DaemonConfig::default(),
            camera: CameraConfig::default(),
            ml: MLConfig::default(),
            authentication: AuthConfig::default(),
            enrollment: EnrollmentConfig::default(),
            preprocessing: PreprocessingConfig::default(),
            models: ModelsConfig::default(),
            recognition: RecognitionConfig::default(),
            dataset: DatasetConfig::default(),
            ipc: IpcConfig::default(),
            telemetry: TelemetryConfig::default(),
            decision: DecisionConfig::default(),
        }
    }
}

impl Config {
    /// Validate every section that has range or consistency rules.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.camera.validate()?;
        self.decision.validate()?;
        Ok(())
    }

    /// Load config from standard locations
    /// Priority: ./doorman.toml > ~/.config/doorman/doorman.toml > /etc/doorman/doorman.toml
    pub fn load() -> Result<Self, ConfigError> {
        let locations = vec![
            PathBuf::from("doorman.toml"),
            dirs::config_dir()
                .map(|p| p.join("doorman/doorman.toml"))
                .unwrap_or_default(),
            PathBuf::from("/etc/doorman/doorman.toml"),
        ];

        for path in locations {
            if path.exists() {
                return Self::load_from(&path);
            }
        }

        // If no config found, use defaults
        Ok(Self::default())
    }

    /// Load config from specific path
    pub fn load_from(path: &Path) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Save config to path
    pub fn save_to(&self, path: &Path) -> Result<(), ConfigError> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn ipc_section_defaults_when_absent_and_parses_when_present() {
        let cfg: super::Config = toml::from_str("").unwrap();
        assert!(cfg.ipc.trusted_uids.is_empty());
        assert!(cfg.ipc.admin_uids.is_empty());
        assert_eq!(cfg.ipc.max_auth_attempts_per_minute, 20);
        assert_eq!(cfg.ipc.stream_socket_group, None);
        assert_eq!(cfg.ipc.wait_for_match_max_ms, 60_000);

        let cfg: super::Config = toml::from_str(
            "[ipc]\ntrusted_uids=[1000]\nadmin_uids=[1001]\nmax_auth_attempts_per_minute=5\nstream_socket_group=\"doorman\"\nwait_for_match_max_ms=1500\n",
        )
        .unwrap();
        assert_eq!(cfg.ipc.trusted_uids, vec![1000]);
        assert_eq!(cfg.ipc.admin_uids, vec![1001]);
        assert_eq!(cfg.ipc.max_auth_attempts_per_minute, 5);
        assert_eq!(cfg.ipc.stream_socket_group.as_deref(), Some("doorman"));
        assert_eq!(cfg.ipc.wait_for_match_max_ms, 1500);
    }

    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.daemon.socket_path, "/run/doorman/doorman.sock");
        assert_eq!(config.ml.device, "cpu");
        assert_eq!(config.authentication.similarity_threshold, 0.4);
        assert!(config.authentication.liveness_enabled);
        // Real deployment default: daemon boots locked.
        assert!(config.daemon.start_locked);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.daemon.socket_path, parsed.daemon.socket_path);
    }

    #[test]
    fn test_config_with_gpu() {
        let toml_str = r#"
            [ml]
            device = "rocm"
            gpu_device_id = 1
        "#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.ml.device, "rocm");
        assert_eq!(config.ml.gpu_device_id, 1);
    }

    #[test]
    fn test_telemetry_defaults_and_old_config() {
        // A config without a [telemetry] table must parse and keep logging off.
        let config: Config = toml::from_str("[ml]\ndevice = \"cpu\"\n").unwrap();
        assert!(!config.telemetry.timing_log);
        assert_eq!(
            config.telemetry.timing_log_path,
            "/run/doorman/timing.jsonl"
        );
        assert_eq!(config.telemetry.timing_log_max_bytes, 1024 * 1024);
        let on: Config = toml::from_str("[telemetry]\ntiming_log = true\n").unwrap();
        assert!(on.telemetry.timing_log);
        assert_eq!(on.telemetry.timing_log_max_bytes, 1024 * 1024);
    }

    #[test]
    fn decision_defaults_keep_current_behaviour_and_old_configs_parse() {
        let old: Config = toml::from_str("[ml]\ndevice = \"cpu\"\n").unwrap();
        assert_eq!(old.decision.mode, DecisionMode::Mean);
        assert_eq!(old.decision.template_score, TemplateScore::Max);
        assert_eq!((old.decision.accept_n, old.decision.window_m), (3, 5));
        assert!(old.validate().is_ok());

        let new: Config = toml::from_str(
            "[decision]\nmode = \"n_of_m\"\ntemplate_score = \"top3_mean\"\naccept_n = 2\nwindow_m = 4\n",
        )
        .unwrap();
        assert_eq!(new.decision.mode, DecisionMode::NOfM);
        assert_eq!(new.decision.template_score, TemplateScore::Top3Mean);
        assert!(new.validate().is_ok());
    }

    #[test]
    fn decision_validation_rejects_bad_n_and_m_and_unknown_modes() {
        let bad = |n, m| DecisionConfig {
            accept_n: n,
            window_m: m,
            ..DecisionConfig::default()
        };
        assert!(bad(1, 1).validate().is_ok());
        for (n, m) in [(0, 5), (6, 5)] {
            let err = bad(n, m).validate().unwrap_err().to_string();
            assert!(err.contains("accept_n") && err.contains("window_m"), "{err}");
        }
        assert!(toml::from_str::<Config>("[decision]\nmode = \"sum\"\n").is_err());
        assert!(toml::from_str::<Config>("[decision]\ntemplate_score = \"min\"\n").is_err());
    }

    #[test]
    fn test_camera_latency_options_default_and_validate() {
        // Old configs (no new keys) keep the current behaviour.
        let old: Config = toml::from_str("[camera]\nwidth = 1024\n").unwrap();
        assert_eq!(old.camera.fourcc, None);
        assert_eq!(old.camera.warmup_frames, 0);
        assert!(!old.camera.prearm_on_lock);
        assert_eq!(old.camera.prearm_window_ms, 5000);
        assert!(old.camera.validate().is_ok());

        let new: Config = toml::from_str(
            "[camera]\nfourcc = \"mjpg\"\nwarmup_frames = 3\nprearm_on_lock = true\n",
        )
        .unwrap();
        assert_eq!(new.camera.fourcc().unwrap(), Some(CameraFourcc::Mjpeg));
        assert_eq!(new.camera.warmup_frames, 3);
        assert!(new.camera.prearm_on_lock);

        let with_warmup = |warmup_frames| CameraConfig {
            warmup_frames,
            ..CameraConfig::default()
        };
        assert!(with_warmup(MAX_WARMUP_FRAMES).validate().is_ok());
        let err = with_warmup(MAX_WARMUP_FRAMES + 1)
            .validate()
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("warmup_frames") && err.contains("0..=10"),
            "{err}"
        );

        let window = |prearm_window_ms| CameraConfig {
            prearm_window_ms,
            ..CameraConfig::default()
        };
        assert!(window(MAX_PREARM_WINDOW_MS).validate().is_ok());
        let err = window(MAX_PREARM_WINDOW_MS + 1)
            .validate()
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("prearm_window_ms") && err.contains("0..=30000"),
            "{err}"
        );

        let bad_fourcc = CameraConfig {
            fourcc: Some("H264".into()),
            ..CameraConfig::default()
        };
        assert!(bad_fourcc
            .validate()
            .unwrap_err()
            .to_string()
            .contains("fourcc"));
    }
}
