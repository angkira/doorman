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
    pub telemetry: TelemetryConfig,
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
            telemetry: TelemetryConfig::default(),
        }
    }
}

impl Config {
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
}
