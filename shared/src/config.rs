use crate::config_trust;
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
/// Bounds (inclusive, ms) for `[camera] native_first_frame_timeout_ms`.
pub const MIN_NATIVE_FIRST_FRAME_TIMEOUT_MS: u32 = 100;
pub const MAX_NATIVE_FIRST_FRAME_TIMEOUT_MS: u32 = 10_000;
/// Upper bound (inclusive) for `[camera] native_warmup_frames`.
pub const MAX_NATIVE_WARMUP_FRAMES: u32 = 10;
/// Default `[camera] native_warmup_frames`: UVC cameras often deliver junk or
/// partial buffers right after STREAMON.
pub const DEFAULT_NATIVE_WARMUP_FRAMES: u32 = 3;
/// Upper bound (inclusive, seconds) for `[camera] native_failure_cooldown_secs` (one day).
pub const MAX_NATIVE_FAILURE_COOLDOWN_SECS: u32 = 86_400;
/// Default `[camera] native_failure_cooldown_secs` (ten minutes).
pub const DEFAULT_NATIVE_FAILURE_COOLDOWN_SECS: u32 = 600;
/// Largest 8-bit luma value; upper bound of `[recognition] min_brightness`.
pub const MAX_LUMA: f32 = 255.0;
/// Largest `[liveness] frames`.
pub const MAX_LIVENESS_FRAMES: usize = 9;
/// Default `[liveness] depth_input_size` (Depth-Anything-V2 export, 37 * 14).
pub const DEFAULT_DEPTH_INPUT_SIZE: u32 = 518;
/// Depth-Anything-V2 patch size: `depth_input_size` must be a multiple of it.
pub const DEPTH_INPUT_MULTIPLE: u32 = 14;
/// Default `[decision] accept_n`: frames that must clear the threshold.
pub const DEFAULT_ACCEPT_N: usize = 3;
/// Default `[decision] window_m`: size of the sliding window of recent frames.
pub const DEFAULT_WINDOW_M: usize = 5;

/// Current `schema_version` of the config file format.
pub const CONFIG_SCHEMA_VERSION: u32 = 1;

fn default_schema_version() -> u32 {
    CONFIG_SCHEMA_VERSION
}

fn default_prearm_window_ms() -> u32 {
    5000
}

fn default_native_first_frame_timeout_ms() -> u32 {
    1500
}

fn default_native_warmup_frames() -> u32 {
    DEFAULT_NATIVE_WARMUP_FRAMES
}

fn default_native_failure_cooldown_secs() -> u32 {
    DEFAULT_NATIVE_FAILURE_COOLDOWN_SECS
}

/// Which live-camera backend `[camera] backend` selects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CameraBackendChoice {
    /// Native (V4L2) backend, accepted only after it delivered a real first
    /// frame; otherwise the ffmpeg backend.
    #[default]
    Auto,
    /// Native backend only (no ffmpeg fallback).
    Native,
    /// ffmpeg backend only (the native backend is not tried).
    Ffmpeg,
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
    /// Config schema version (see docs/CONFIG.md). Files that omit it are read
    /// as the current version; a version newer than this build is rejected.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,

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

    #[serde(default)]
    pub liveness: LivenessConfig,

    #[serde(default)]
    pub pam: PamConfig,
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

    /// Live-camera backend: "auto" (native V4L2 with a validated first frame,
    /// then ffmpeg), "native" (no ffmpeg fallback) or "ffmpeg" (skip native).
    #[serde(default)]
    pub backend: CameraBackendChoice,

    /// With the native backend: how long (ms) the first decoded frame may take
    /// before the backend is rejected (then ffmpeg under "auto"). 100..=10000.
    #[serde(default = "default_native_first_frame_timeout_ms")]
    pub native_first_frame_timeout_ms: u32,

    /// With the native backend: buffers read and thrown away right after
    /// STREAMON before one may count as the first frame (UVC cameras can
    /// deliver junk or partial buffers first). Candidates after the warm-up are
    /// validated and retried within `native_first_frame_timeout_ms`.
    /// 0..=10, default 3. Not the same as `warmup_frames` (cold-open discard
    /// by the producer).
    #[serde(default = "default_native_warmup_frames")]
    pub native_warmup_frames: u32,

    /// With `backend = "auto"`: after the native backend failed in this daemon
    /// process, go straight to ffmpeg for this many seconds (per device and
    /// requested mode), then probe native again. 0 disables the memory.
    /// 0..=86400, default 600. Ignored when `backend = "native"`.
    #[serde(default = "default_native_failure_cooldown_secs")]
    pub native_failure_cooldown_secs: u32,

    /// Pin the camera by a stable device node instead of `device_index`, for
    /// example `/dev/v4l/by-id/usb-Vendor_Camera-video-index0`. The path is
    /// resolved on every open and must end at a `/dev/videoN` character
    /// device. Unset: use `device_index`.
    #[serde(default)]
    pub device_path: Option<String>,

    /// Accept virtual capture devices (v4l2loopback, vivid, ...). Default
    /// `false`: members of the `video` group can feed arbitrary frames into
    /// such a device, so the daemon refuses them. Enable only for testing.
    #[serde(default)]
    pub allow_virtual_devices: bool,

    /// Longest a frame may sit between capture and being taken up by an
    /// authentication attempt; older frames are skipped (a frozen or stalled
    /// feed can never decide). Milliseconds, `0` disables the guard, otherwise
    /// 100..=10000. Complements the request-time freshness floor of
    /// `WaitForMatch`, which drops frames captured before the request.
    #[serde(default = "default_max_frame_age_ms")]
    pub max_frame_age_ms: u64,
}

fn default_max_frame_age_ms() -> u64 {
    DEFAULT_MAX_FRAME_AGE_MS
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
        if !(MIN_NATIVE_FIRST_FRAME_TIMEOUT_MS..=MAX_NATIVE_FIRST_FRAME_TIMEOUT_MS)
            .contains(&self.native_first_frame_timeout_ms)
        {
            return Err(ConfigError::Invalid(format!(
                "camera.native_first_frame_timeout_ms = {} is out of range \
                 ({MIN_NATIVE_FIRST_FRAME_TIMEOUT_MS}..={MAX_NATIVE_FIRST_FRAME_TIMEOUT_MS})",
                self.native_first_frame_timeout_ms
            )));
        }
        if self.native_warmup_frames > MAX_NATIVE_WARMUP_FRAMES {
            return Err(ConfigError::Invalid(format!(
                "camera.native_warmup_frames = {} is out of range (0..={MAX_NATIVE_WARMUP_FRAMES})",
                self.native_warmup_frames
            )));
        }
        if self.native_failure_cooldown_secs > MAX_NATIVE_FAILURE_COOLDOWN_SECS {
            return Err(ConfigError::Invalid(format!(
                "camera.native_failure_cooldown_secs = {} is out of range \
                 (0..={MAX_NATIVE_FAILURE_COOLDOWN_SECS})",
                self.native_failure_cooldown_secs
            )));
        }
        if self.max_frame_age_ms != 0
            && !(MIN_MAX_FRAME_AGE_MS..=MAX_MAX_FRAME_AGE_MS).contains(&self.max_frame_age_ms)
        {
            return Err(ConfigError::Invalid(format!(
                "camera.max_frame_age_ms = {} is out of range (0 to disable, else {MIN_MAX_FRAME_AGE_MS}..={MAX_MAX_FRAME_AGE_MS})",
                self.max_frame_age_ms
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

    /// `mean` decision mode: the least number of judged-good frames an attempt
    /// must have collected, otherwise it is rejected (`insufficient_frames`)
    /// instead of deciding on whatever the deadline left, e.g. one frame on a
    /// loaded CPU. `0` (default) derives it: `auth_frames` minus 2 tolerated
    /// drops (8 of 10), at least 1. Otherwise 1..=`auth_frames`. Not used by
    /// `n_of_m`, which has its own N-of-M requirement.
    #[serde(default)]
    pub min_decision_frames: usize,
}

impl AuthConfig {
    /// The enforced minimum of `mean` decision frames (see `min_decision_frames`).
    pub fn effective_min_decision_frames(&self) -> usize {
        if self.min_decision_frames == 0 {
            self.auth_frames
                .saturating_sub(MIN_DECISION_TOLERATED_DROPS)
                .max(1)
        } else {
            self.min_decision_frames
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrollmentConfig {
    #[serde(default = "default_enroll_frames")]
    pub enroll_frames: usize,

    #[serde(default = "default_min_valid_frames")]
    pub min_valid_frames: usize,

    /// Enrolment drops a captured template whose cosine to an already kept one
    /// is at or above this value (near-duplicate pruning). Range (0, 1].
    #[serde(default = "default_prune_cosine")]
    pub prune_cosine: f32,
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

    /// Frame-quality gate: minimum mean luma (0..=255) inside the face box.
    /// Rejects frames that are too dark to embed reliably. `0.0` disables it.
    #[serde(default)]
    pub min_brightness: f32,

    /// What to do when a frame shows more than one face (a second face whose
    /// box is at least `multi_face_min_area_ratio` of the main face's area and
    /// whose detector score is at least `multi_face_min_score`). `reject`
    /// (default): the frame is not accepted, so a photo held next to a live
    /// face cannot alternate between the two; an attempt left with too few
    /// frames is rejected as `ambiguous_faces`. `largest`: use the largest
    /// face instead of the highest-scoring one. `off`: the original behaviour
    /// (highest-scoring face, others ignored).
    #[serde(default)]
    pub multi_face_policy: MultiFacePolicy,

    /// Area of a second face relative to the main face to count. Range (0, 1].
    #[serde(default = "default_multi_face_min_area_ratio")]
    pub multi_face_min_area_ratio: f32,

    /// Detector score a second face needs to count. Range [0, 1].
    #[serde(default = "default_multi_face_min_score")]
    pub multi_face_min_score: f32,
}

fn default_multi_face_min_area_ratio() -> f32 {
    DEFAULT_MULTI_FACE_MIN_AREA_RATIO
}

fn default_multi_face_min_score() -> f32 {
    DEFAULT_MULTI_FACE_MIN_SCORE
}

/// `[recognition] multi_face_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiFacePolicy {
    /// A frame with a second qualifying face is not accepted.
    #[default]
    Reject,
    /// Use the largest face instead of the highest-scoring one.
    Largest,
    /// Original behaviour: the highest-scoring face, others ignored.
    Off,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            aggregation_window: default_aggregation_window(),
            hysteresis_frames: default_hysteresis_frames(),
            min_sharpness: default_min_sharpness(),
            min_face_area_frac: default_min_face_area_frac(),
            min_brightness: 0.0,
            multi_face_policy: MultiFacePolicy::default(),
            multi_face_min_area_ratio: DEFAULT_MULTI_FACE_MIN_AREA_RATIO,
            multi_face_min_score: DEFAULT_MULTI_FACE_MIN_SCORE,
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

    /// Flip test-time augmentation: embed the aligned crop and its mirror,
    /// average and renormalise. Costs one extra recognizer pass per frame.
    #[serde(default)]
    pub flip_tta: bool,

    /// `mean` mode only: every frame that feeds the aggregate must itself
    /// score at least this cosine against the enrolled templates, otherwise the
    /// attempt is rejected (`identity_floor`, fail closed). It binds liveness to
    /// identity: a live face of someone else on the checked frames cannot be
    /// averaged together with photos of the owner in between. Range
    /// 0..=`similarity_threshold`; `0` disables. Default 0.30.
    #[serde(default = "default_frame_identity_floor")]
    pub frame_identity_floor: f32,
}

fn default_frame_identity_floor() -> f32 {
    DEFAULT_FRAME_IDENTITY_FLOOR
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            mode: default_decision_mode(),
            accept_n: default_accept_n(),
            window_m: default_window_m(),
            template_score: default_template_score(),
            flip_tta: false,
            frame_identity_floor: DEFAULT_FRAME_IDENTITY_FLOOR,
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
        if !(0.0..=1.0).contains(&self.frame_identity_floor) {
            return Err(ConfigError::Invalid(format!(
                "decision.frame_identity_floor = {} is out of range (0..=1)",
                self.frame_identity_floor
            )));
        }
        Ok(())
    }
}

/// How per-frame liveness results become one verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LivenessAggregate {
    /// Median relief over the evaluated frames (majority of frames on one
    /// side of the threshold, which is equivalent for an odd frame count).
    Median,
}

/// Statistic that turns a depth patch into the relief value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReliefStat {
    /// `std(face depth) / (global max - global min)` (original behaviour).
    #[serde(rename = "minmax")]
    MinMax,
    /// `(p90 - p10)` of the face depth over the same range of the whole map,
    /// robust to outlier pixels. Thresholds are NOT comparable with `minmax`.
    #[serde(rename = "p10_p90")]
    P10P90,
}

/// `[liveness]`: how the depth-relief liveness gate is evaluated. The relief
/// threshold itself stays `authentication.liveness_depth_threshold`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessConfig {
    /// Frames to evaluate (odd, 1..=9). `1` is the original single-frame check.
    #[serde(default = "default_liveness_frames")]
    pub frames: usize,

    #[serde(default = "default_liveness_aggregate")]
    pub aggregate: LivenessAggregate,

    #[serde(default = "default_relief_stat")]
    pub relief_stat: ReliefStat,

    /// Square depth network input size; a multiple of 14. Sizes other than the
    /// default need a model exported for that size (or with dynamic axes).
    #[serde(default = "default_depth_input_size")]
    pub depth_input_size: u32,

    /// `mean` decision mode only: depth-check this many frames, evenly sampled
    /// across the aggregated sequence (besides the deciding frame, which is
    /// always checked). `0` (default) keeps the original behaviour: only the
    /// first quality frames and the deciding frame are depth-checked. Any
    /// failing sample rejects the attempt. Range `0..=authentication.auth_frames`.
    #[serde(default)]
    pub mean_mode_depth_samples: u32,

    /// Extra liveness signals (screen flash, head pose). None are implemented:
    /// any entry is rejected so a security setting is never silently ignored.
    #[serde(default)]
    pub signals: Vec<String>,
}

impl Default for LivenessConfig {
    fn default() -> Self {
        Self {
            frames: default_liveness_frames(),
            aggregate: default_liveness_aggregate(),
            relief_stat: default_relief_stat(),
            depth_input_size: default_depth_input_size(),
            mean_mode_depth_samples: 0,
            signals: Vec::new(),
        }
    }
}

impl LivenessConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.frames.is_multiple_of(2) || self.frames > MAX_LIVENESS_FRAMES {
            return Err(ConfigError::Invalid(format!(
                "liveness.frames = {} must be odd and at most {MAX_LIVENESS_FRAMES}",
                self.frames
            )));
        }
        if self.depth_input_size == 0 || !self.depth_input_size.is_multiple_of(DEPTH_INPUT_MULTIPLE)
        {
            return Err(ConfigError::Invalid(format!(
                "liveness.depth_input_size = {} must be a positive multiple of {DEPTH_INPUT_MULTIPLE}",
                self.depth_input_size
            )));
        }
        if let Some(name) = self.signals.first() {
            return Err(ConfigError::Invalid(format!(
                "liveness.signals contains {name:?}, but no extra liveness signal is implemented yet"
            )));
        }
        Ok(())
    }
}

/// Default `[pam.sudo] service_names`: the PAM service name sudo and sudo-rs
/// use (`sudo-i`, used for `-i` where the distribution configures it, is a
/// separate opt-in).
pub const DEFAULT_SUDO_SERVICE: &str = "sudo";
/// Default `[pam.sudo] accept_n`: stricter than the lock screen (3 of 5).
pub const DEFAULT_SUDO_ACCEPT_N: usize = 4;
/// Default `[pam.sudo] window_m`.
pub const DEFAULT_SUDO_WINDOW_M: usize = 5;
/// Default and minimum `[pam.sudo] liveness_frames`: liveness is never a
/// single-frame check for sudo.
pub const MIN_SUDO_LIVENESS_FRAMES: usize = 3;
/// Default `[pam.sudo] timeout_ms`.
pub const DEFAULT_SUDO_TIMEOUT_MS: u64 = 4000;
/// Upper bound (inclusive) for `[pam.sudo] timeout_ms`: past it the password
/// prompt would be delayed too long.
pub const MAX_SUDO_TIMEOUT_MS: u64 = 4000;
/// Default `[pam.sudo] max_failures`.
pub const DEFAULT_SUDO_MAX_FAILURES: u32 = 3;
/// Default `[pam.sudo] suspend_secs` (10 minutes).
pub const DEFAULT_SUDO_SUSPEND_SECS: u64 = 600;

/// `[pam]`: per-PAM-service policies.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PamConfig {
    #[serde(default)]
    pub sudo: SudoConfig,
}

/// `[pam.sudo]`: opt-in strict face authentication for sudo. Disabled by
/// default; see docs/SUDO.md.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SudoConfig {
    /// Master switch. While `false` an Authenticate that names a service in
    /// `service_names` is denied (the PAM module falls through to the password).
    #[serde(default)]
    pub enabled: bool,

    /// PAM service names (`PAM_SERVICE`) that get the strict profile.
    #[serde(default = "default_sudo_service_names")]
    pub service_names: Vec<String>,

    /// Cosine threshold for sudo. `None` = `authentication.similarity_threshold`;
    /// when set it must be at least that value.
    #[serde(default)]
    pub similarity_threshold: Option<f32>,

    /// Frames that must reach the threshold (decision is always `n_of_m`).
    #[serde(default = "default_sudo_accept_n")]
    pub accept_n: usize,

    /// Sliding window of recent quality frames.
    #[serde(default = "default_sudo_window_m")]
    pub window_m: usize,

    /// Liveness frames (odd, `MIN_SUDO_LIVENESS_FRAMES..=MAX_LIVENESS_FRAMES`).
    /// Liveness itself is not optional for sudo.
    #[serde(default = "default_sudo_liveness_frames")]
    pub liveness_frames: usize,

    /// Total time for one sudo attempt, 1..=`MAX_SUDO_TIMEOUT_MS`.
    #[serde(default = "default_sudo_timeout_ms")]
    pub timeout_ms: u64,

    /// Consecutive failed sudo attempts of one user that suspend face auth.
    #[serde(default = "default_sudo_max_failures")]
    pub max_failures: u32,

    /// Suspension length, and the window in which failures are counted.
    #[serde(default = "default_sudo_suspend_secs")]
    pub suspend_secs: u64,
}

fn default_sudo_service_names() -> Vec<String> {
    vec![DEFAULT_SUDO_SERVICE.to_string()]
}
fn default_sudo_accept_n() -> usize {
    DEFAULT_SUDO_ACCEPT_N
}
fn default_sudo_window_m() -> usize {
    DEFAULT_SUDO_WINDOW_M
}
fn default_sudo_liveness_frames() -> usize {
    MIN_SUDO_LIVENESS_FRAMES
}
fn default_sudo_timeout_ms() -> u64 {
    DEFAULT_SUDO_TIMEOUT_MS
}
fn default_sudo_max_failures() -> u32 {
    DEFAULT_SUDO_MAX_FAILURES
}
fn default_sudo_suspend_secs() -> u64 {
    DEFAULT_SUDO_SUSPEND_SECS
}

impl Default for SudoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            service_names: default_sudo_service_names(),
            similarity_threshold: None,
            accept_n: DEFAULT_SUDO_ACCEPT_N,
            window_m: DEFAULT_SUDO_WINDOW_M,
            liveness_frames: MIN_SUDO_LIVENESS_FRAMES,
            timeout_ms: DEFAULT_SUDO_TIMEOUT_MS,
            max_failures: DEFAULT_SUDO_MAX_FAILURES,
            suspend_secs: DEFAULT_SUDO_SUSPEND_SECS,
        }
    }
}

impl SudoConfig {
    /// The cosine threshold of the strict profile.
    pub fn effective_threshold(&self, auth: &AuthConfig) -> f32 {
        self.similarity_threshold
            .unwrap_or(auth.similarity_threshold)
    }

    /// Range checks always; cross-section checks (liveness on, frame budget)
    /// only when `enabled`, so an unused section never blocks startup.
    pub fn validate(&self, auth: &AuthConfig) -> Result<(), ConfigError> {
        let invalid = |msg: String| Err(ConfigError::Invalid(msg));
        if let Some(t) = self.similarity_threshold {
            if !(t.is_finite() && t <= 1.0 && t >= auth.similarity_threshold) {
                return invalid(format!(
                    "pam.sudo.similarity_threshold = {t} must be at most 1.0 and at least authentication.similarity_threshold = {}",
                    auth.similarity_threshold
                ));
            }
        }
        if self.accept_n < 1 || self.accept_n > self.window_m {
            return invalid(format!(
                "pam.sudo.accept_n = {} and pam.sudo.window_m = {} must satisfy 1 <= accept_n <= window_m",
                self.accept_n, self.window_m
            ));
        }
        if self.liveness_frames.is_multiple_of(2)
            || !(MIN_SUDO_LIVENESS_FRAMES..=MAX_LIVENESS_FRAMES).contains(&self.liveness_frames)
        {
            return invalid(format!(
                "pam.sudo.liveness_frames = {} must be odd and within {MIN_SUDO_LIVENESS_FRAMES}..={MAX_LIVENESS_FRAMES}",
                self.liveness_frames
            ));
        }
        if self.timeout_ms == 0 || self.timeout_ms > MAX_SUDO_TIMEOUT_MS {
            return invalid(format!(
                "pam.sudo.timeout_ms = {} must be within 1..={MAX_SUDO_TIMEOUT_MS}",
                self.timeout_ms
            ));
        }
        if self.max_failures == 0 || self.suspend_secs == 0 {
            return invalid(format!(
                "pam.sudo.max_failures = {} and pam.sudo.suspend_secs = {} must both be at least 1",
                self.max_failures, self.suspend_secs
            ));
        }
        if let Some(name) = self
            .service_names
            .iter()
            .find(|n| n.is_empty() || n.chars().any(char::is_whitespace))
        {
            return invalid(format!(
                "pam.sudo.service_names contains {name:?}: names must be non-empty without whitespace"
            ));
        }
        if self.enabled {
            if self.service_names.is_empty() {
                return invalid(
                    "pam.sudo.enabled = true needs at least one pam.sudo.service_names entry"
                        .to_string(),
                );
            }
            if !auth.liveness_enabled {
                return invalid("pam.sudo.enabled = true requires authentication.liveness_enabled = true (liveness is never advisory for sudo)".to_string());
            }
            if auth.auth_frames < self.accept_n {
                return invalid(format!(
                    "authentication.auth_frames = {} is the frame budget of the sudo profile and must be >= pam.sudo.accept_n = {}",
                    auth.auth_frames, self.accept_n
                ));
            }
        }
        Ok(())
    }
}

/// Default per-uid Authenticate budget per sliding minute.
pub const DEFAULT_MAX_AUTH_ATTEMPTS_PER_MINUTE: u32 = 20;
/// Default cap for a single `WaitForMatch` (milliseconds).
pub const DEFAULT_WAIT_FOR_MATCH_MAX_MS: u64 = 60_000;
/// Default for `[ipc] prearm_camera`: start recognition when `WaitForMatch`
/// is requested instead of waiting for the lock state.
pub const DEFAULT_PREARM_CAMERA: bool = true;
/// Default for `[ipc] prearm_max_hold_secs`.
pub const DEFAULT_PREARM_MAX_HOLD_SECS: u64 = 5;
/// Inclusive bounds for `[ipc] prearm_max_hold_secs`.
pub const MIN_PREARM_MAX_HOLD_SECS: u64 = 1;
pub const MAX_PREARM_MAX_HOLD_SECS: u64 = 30;
/// Default for `[decision] frame_identity_floor`: every frame of a `mean`
/// decision must individually score at least this cosine against the enrolled
/// templates. Well below the 0.4 match threshold (genuine frames observed at
/// 0.74-0.82 on the reference host) and well above where impostor frames sit
/// (near 0), so it never fires on a real face and always on a different one.
pub const DEFAULT_FRAME_IDENTITY_FLOOR: f32 = 0.30;
/// Default for `[authentication] min_decision_frames`: `0` derives the minimum
/// from `auth_frames` minus this many tolerated dropped frames (8 of 10).
pub const MIN_DECISION_TOLERATED_DROPS: usize = 2;
/// Default for `[camera] max_frame_age_ms`: a frame is used only if it was
/// captured at most this long before it is taken up. Above the normal
/// capture-to-consumption latency (one frame period, 33-200 ms), far below a
/// frozen feed.
pub const DEFAULT_MAX_FRAME_AGE_MS: u64 = 1_000;
/// Inclusive bounds for a non-zero `[camera] max_frame_age_ms` (`0` disables).
pub const MIN_MAX_FRAME_AGE_MS: u64 = 100;
pub const MAX_MAX_FRAME_AGE_MS: u64 = 10_000;
/// Default for `[recognition] multi_face_min_area_ratio`: a second face counts
/// when its box area is at least this fraction of the main face's area.
pub const DEFAULT_MULTI_FACE_MIN_AREA_RATIO: f32 = 0.25;
/// Default for `[recognition] multi_face_min_score`: detector confidence a
/// second face needs to count (the detector itself already drops below 0.6).
pub const DEFAULT_MULTI_FACE_MIN_SCORE: f32 = 0.75;
/// Default cap on simultaneously open command connections per peer uid.
pub const DEFAULT_MAX_CONNECTIONS_PER_UID: u32 = 8;
/// Default for `[ipc] face_max_judged_failures`: judged failures that suspend
/// face auth for a user (`0` disables the budget).
pub const DEFAULT_FACE_MAX_JUDGED_FAILURES: u32 = 5;
/// Inclusive upper bound for `[ipc] face_max_judged_failures`.
pub const MAX_FACE_MAX_JUDGED_FAILURES: u32 = 100;
/// Default for `[ipc] face_failure_window_secs`: judged failures older than this
/// no longer count towards the budget.
pub const DEFAULT_FACE_FAILURE_WINDOW_SECS: u64 = 300;
/// Inclusive bounds for `[ipc] face_failure_window_secs`.
pub const MIN_FACE_FAILURE_WINDOW_SECS: u64 = 10;
pub const MAX_FACE_FAILURE_WINDOW_SECS: u64 = 86_400;
/// Default for `[ipc] face_cooldown_secs`: escalating suspension lengths
/// (30 s, 2 min, 10 min). The last entry repeats once the list is used up.
pub const DEFAULT_FACE_COOLDOWN_SECS: [u64; 3] = [30, 120, 600];
/// Bounds for `[ipc] face_cooldown_secs`: entries and each entry in seconds.
pub const MAX_FACE_COOLDOWN_STEPS: usize = 8;
pub const MIN_FACE_COOLDOWN_SECS: u64 = 1;
pub const MAX_FACE_COOLDOWN_SECS: u64 = 86_400;
/// Default for `[ipc] face_budget_quiet_reset_secs`: with no judged failure for
/// this long, the escalation starts again at the first cooldown.
pub const DEFAULT_FACE_BUDGET_QUIET_RESET_SECS: u64 = 1_800;
/// Inclusive bounds for `[ipc] face_budget_quiet_reset_secs`.
pub const MIN_FACE_BUDGET_QUIET_RESET_SECS: u64 = 60;
pub const MAX_FACE_BUDGET_QUIET_RESET_SECS: u64 = 604_800;

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

    /// Pre-arm: when a `WaitForMatch` arrives while the session is not yet
    /// reported locked, open the camera and run the full decision right away,
    /// pipelined with the wait for `LockedHint`. `Matched` is still delivered
    /// only once the lock is confirmed, and only from frames captured after the
    /// request. Changes timing only, never the decision rule.
    #[serde(default = "default_prearm_camera")]
    pub prearm_camera: bool,

    /// Longest a pre-armed camera/decision is held without the lock being
    /// confirmed, counted from the request. After it the camera is released and
    /// the result (if any) discarded; the call then waits for the lock as
    /// without pre-arm. Range 1..=30.
    #[serde(default = "default_prearm_max_hold_secs")]
    pub prearm_max_hold_secs: u64,

    /// Simultaneously open command connections allowed per peer uid (the global
    /// cap is 64). Extra connections are dropped. `0` disables the per-uid cap.
    #[serde(default = "default_max_connections_per_uid")]
    pub max_connections_per_uid: u32,

    /// Failure budget of face auth, for EVERY entry point (`Authenticate`, the
    /// PAM module, `doorman test`, and each `WaitForMatch` retry on the lock
    /// screen). After this many *judged* failures inside
    /// `face_failure_window_secs` face auth for that user is suspended for the
    /// next step of `face_cooldown_secs`. A judged failure is a completed
    /// decision that rejected (below threshold, liveness rejected, identity
    /// floor, ambiguous faces); camera problems, timeouts, no face in view and
    /// an unavailable liveness model are NOT counted. A successful match resets
    /// the budget. The password path is never affected. Side effect: someone
    /// who fails on purpose can switch the owner's face unlock off for the
    /// cooldown (then the password works, and a match resets it). `0` disables
    /// the budget. Range 0..=100.
    #[serde(default = "default_face_max_judged_failures")]
    pub face_max_judged_failures: u32,

    /// Window in which judged failures are counted. Range 10..=86400.
    #[serde(default = "default_face_failure_window_secs")]
    pub face_failure_window_secs: u64,

    /// Escalating suspension lengths in seconds, one per consecutive
    /// suspension; the last entry repeats. Non-decreasing, 1..=8 entries, each
    /// 1..=86400.
    #[serde(default = "default_face_cooldown_secs")]
    pub face_cooldown_secs: Vec<u64>,

    /// With no judged failure for this long the escalation restarts at the
    /// first cooldown. Range 60..=604800.
    #[serde(default = "default_face_budget_quiet_reset_secs")]
    pub face_budget_quiet_reset_secs: u64,
}

fn default_face_max_judged_failures() -> u32 {
    DEFAULT_FACE_MAX_JUDGED_FAILURES
}

fn default_face_failure_window_secs() -> u64 {
    DEFAULT_FACE_FAILURE_WINDOW_SECS
}

fn default_face_cooldown_secs() -> Vec<u64> {
    DEFAULT_FACE_COOLDOWN_SECS.to_vec()
}

fn default_face_budget_quiet_reset_secs() -> u64 {
    DEFAULT_FACE_BUDGET_QUIET_RESET_SECS
}

fn default_prearm_camera() -> bool {
    DEFAULT_PREARM_CAMERA
}

fn default_prearm_max_hold_secs() -> u64 {
    DEFAULT_PREARM_MAX_HOLD_SECS
}

impl IpcConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(MIN_PREARM_MAX_HOLD_SECS..=MAX_PREARM_MAX_HOLD_SECS)
            .contains(&self.prearm_max_hold_secs)
        {
            return Err(ConfigError::Invalid(format!(
                "ipc.prearm_max_hold_secs = {} is out of range ({MIN_PREARM_MAX_HOLD_SECS}..={MAX_PREARM_MAX_HOLD_SECS})",
                self.prearm_max_hold_secs
            )));
        }
        self.validate_face_budget()
    }

    fn validate_face_budget(&self) -> Result<(), ConfigError> {
        if self.face_max_judged_failures > MAX_FACE_MAX_JUDGED_FAILURES {
            return Err(ConfigError::Invalid(format!(
                "ipc.face_max_judged_failures = {} is out of range (0..={MAX_FACE_MAX_JUDGED_FAILURES})",
                self.face_max_judged_failures
            )));
        }
        if !(MIN_FACE_FAILURE_WINDOW_SECS..=MAX_FACE_FAILURE_WINDOW_SECS)
            .contains(&self.face_failure_window_secs)
        {
            return Err(ConfigError::Invalid(format!(
                "ipc.face_failure_window_secs = {} is out of range ({MIN_FACE_FAILURE_WINDOW_SECS}..={MAX_FACE_FAILURE_WINDOW_SECS})",
                self.face_failure_window_secs
            )));
        }
        if !(MIN_FACE_BUDGET_QUIET_RESET_SECS..=MAX_FACE_BUDGET_QUIET_RESET_SECS)
            .contains(&self.face_budget_quiet_reset_secs)
        {
            return Err(ConfigError::Invalid(format!(
                "ipc.face_budget_quiet_reset_secs = {} is out of range ({MIN_FACE_BUDGET_QUIET_RESET_SECS}..={MAX_FACE_BUDGET_QUIET_RESET_SECS})",
                self.face_budget_quiet_reset_secs
            )));
        }
        let steps = &self.face_cooldown_secs;
        if steps.is_empty() || steps.len() > MAX_FACE_COOLDOWN_STEPS {
            return Err(ConfigError::Invalid(format!(
                "ipc.face_cooldown_secs has {} entries (1..={MAX_FACE_COOLDOWN_STEPS} required)",
                steps.len()
            )));
        }
        if steps
            .iter()
            .any(|s| !(MIN_FACE_COOLDOWN_SECS..=MAX_FACE_COOLDOWN_SECS).contains(s))
        {
            return Err(ConfigError::Invalid(format!(
                "ipc.face_cooldown_secs = {steps:?}: every entry must be in {MIN_FACE_COOLDOWN_SECS}..={MAX_FACE_COOLDOWN_SECS}"
            )));
        }
        if steps.windows(2).any(|w| w[0] > w[1]) {
            return Err(ConfigError::Invalid(format!(
                "ipc.face_cooldown_secs = {steps:?} must not decrease (escalating cooldowns)"
            )));
        }
        Ok(())
    }
}

fn default_max_connections_per_uid() -> u32 {
    DEFAULT_MAX_CONNECTIONS_PER_UID
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
            prearm_camera: DEFAULT_PREARM_CAMERA,
            prearm_max_hold_secs: DEFAULT_PREARM_MAX_HOLD_SECS,
            max_connections_per_uid: DEFAULT_MAX_CONNECTIONS_PER_UID,
            face_max_judged_failures: DEFAULT_FACE_MAX_JUDGED_FAILURES,
            face_failure_window_secs: DEFAULT_FACE_FAILURE_WINDOW_SECS,
            face_cooldown_secs: default_face_cooldown_secs(),
            face_budget_quiet_reset_secs: DEFAULT_FACE_BUDGET_QUIET_RESET_SECS,
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

    /// Age in days used by `doorman dataset purge` when `--older-than` is
    /// omitted. Advisory only: the daemon never deletes dataset files itself.
    #[serde(default = "default_dataset_retention_days")]
    pub retention_days: u32,
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
            retention_days: default_dataset_retention_days(),
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

    /// Accept model files whose SHA-256 does not match the manifest. Default
    /// `false`: a loaded model (detector, recognizer, liveness, depth) that is
    /// listed in the manifest and differs makes the daemon refuse to start.
    /// Set `true` only after replacing a model on purpose and before adding its
    /// hash to `manifest_path`. Models not listed at all are never refused.
    #[serde(default)]
    pub allow_unverified_models: bool,

    /// Extra manifest (`sha256sum` format, bare file names) merged over the
    /// one compiled into the daemon: add the depth model or a replaced model.
    /// Must be root-owned and not writable by others. Unset: built-in only.
    /// Generate with `doorman models verify --print-manifest`.
    #[serde(default)]
    pub manifest_path: Option<String>,
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
fn default_liveness_frames() -> usize {
    1
}
fn default_liveness_aggregate() -> LivenessAggregate {
    LivenessAggregate::Median
}
fn default_relief_stat() -> ReliefStat {
    ReliefStat::MinMax
}
fn default_depth_input_size() -> u32 {
    DEFAULT_DEPTH_INPUT_SIZE
}
fn default_enroll_frames() -> usize {
    20
}
fn default_min_valid_frames() -> usize {
    5
}
fn default_prune_cosine() -> f32 {
    0.95
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
/// Frames per raw capture bundle (only used when `keep_raw_frames = true`).
fn default_raw_frame_bundle_size() -> usize {
    5
}
/// Raw capture bundles per day (only used when `keep_raw_frames = true`).
fn default_max_capture_packs_per_day() -> usize {
    20
}
/// Advisory age in days that `doorman dataset purge` uses when `--older-than`
/// is omitted. Nothing is ever deleted automatically.
fn default_dataset_retention_days() -> u32 {
    14
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
            backend: CameraBackendChoice::default(),
            native_first_frame_timeout_ms: default_native_first_frame_timeout_ms(),
            native_warmup_frames: default_native_warmup_frames(),
            native_failure_cooldown_secs: default_native_failure_cooldown_secs(),
            device_path: None,
            allow_virtual_devices: false,
            max_frame_age_ms: DEFAULT_MAX_FRAME_AGE_MS,
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
            min_decision_frames: 0,
        }
    }
}

impl Default for EnrollmentConfig {
    fn default() -> Self {
        Self {
            enroll_frames: default_enroll_frames(),
            min_valid_frames: default_min_valid_frames(),
            prune_cosine: default_prune_cosine(),
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
            allow_unverified_models: false,
            manifest_path: None,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            schema_version: CONFIG_SCHEMA_VERSION,
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
            liveness: LivenessConfig::default(),
            pam: PamConfig::default(),
        }
    }
}

impl Config {
    /// Validate every section that has range or consistency rules.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.schema_version > CONFIG_SCHEMA_VERSION {
            return Err(ConfigError::Invalid(format!(
                "schema_version = {} is newer than this build supports ({CONFIG_SCHEMA_VERSION}); upgrade doorman",
                self.schema_version
            )));
        }
        self.camera.validate()?;
        self.ipc.validate()?;
        self.decision.validate()?;
        self.liveness.validate()?;
        self.pam.sudo.validate(&self.authentication)?;
        let brightness = self.recognition.min_brightness;
        if !(0.0..=MAX_LUMA).contains(&brightness) {
            return Err(ConfigError::Invalid(format!(
                "recognition.min_brightness = {brightness} is out of range (0..={MAX_LUMA})"
            )));
        }
        let floor = self.decision.frame_identity_floor;
        if floor > self.authentication.similarity_threshold {
            return Err(ConfigError::Invalid(format!(
                "decision.frame_identity_floor = {floor} must not exceed authentication.similarity_threshold = {}",
                self.authentication.similarity_threshold
            )));
        }
        let min_frames = self.authentication.min_decision_frames;
        if min_frames > self.authentication.auth_frames {
            return Err(ConfigError::Invalid(format!(
                "authentication.min_decision_frames = {min_frames} is out of range (0 = derived, else 1..=authentication.auth_frames = {})",
                self.authentication.auth_frames
            )));
        }
        let ratio = self.recognition.multi_face_min_area_ratio;
        if !(ratio > 0.0 && ratio <= 1.0) {
            return Err(ConfigError::Invalid(format!(
                "recognition.multi_face_min_area_ratio = {ratio} is out of range (0, 1]"
            )));
        }
        let score = self.recognition.multi_face_min_score;
        if !(0.0..=1.0).contains(&score) {
            return Err(ConfigError::Invalid(format!(
                "recognition.multi_face_min_score = {score} is out of range (0..=1)"
            )));
        }
        let prune = self.enrollment.prune_cosine;
        if !(prune > 0.0 && prune <= 1.0) {
            return Err(ConfigError::Invalid(format!(
                "enrollment.prune_cosine = {prune} is out of range (0, 1]"
            )));
        }
        let samples = self.liveness.mean_mode_depth_samples;
        if usize::try_from(samples).map_or(true, |n| n > self.authentication.auth_frames) {
            return Err(ConfigError::Invalid(format!(
                "liveness.mean_mode_depth_samples = {samples} is out of range (0..=authentication.auth_frames = {})",
                self.authentication.auth_frames
            )));
        }
        if self.decision.mode == DecisionMode::NOfM
            && self.authentication.auth_frames < self.decision.accept_n
        {
            return Err(ConfigError::Invalid(format!(
                "authentication.auth_frames = {} is the frame budget of decision.mode = \"n_of_m\" and must be >= decision.accept_n = {}",
                self.authentication.auth_frames, self.decision.accept_n
            )));
        }
        Ok(())
    }

    /// Load config from the standard locations (see [`Config::load_with`]).
    pub fn load() -> Result<Self, ConfigError> {
        Self::load_with(None)
    }

    /// Load config for this process.
    ///
    /// Non-root: `explicit` > `DOORMAN_CONFIG` > ./doorman.toml >
    /// ~/.config/doorman/doorman.toml > /etc/doorman/doorman.toml > defaults.
    /// Root: `explicit` > `DOORMAN_CONFIG` > /etc/doorman/doorman.toml >
    /// defaults, and the chosen file must be root-owned and not writable by
    /// others (see [`crate::config_trust`]); otherwise this is an error.
    pub fn load_with(explicit: Option<&Path>) -> Result<Self, ConfigError> {
        let env = std::env::var("DOORMAN_CONFIG").ok();
        Self::load_resolved(
            config_trust::effective_uid(),
            explicit,
            env.as_deref(),
            &config_trust::FsMetadata,
        )
    }

    /// [`Config::load_with`] with the uid, environment value and metadata
    /// source injected.
    pub fn load_resolved(
        euid: u32,
        explicit: Option<&Path>,
        env: Option<&str>,
        meta: &dyn config_trust::MetadataProvider,
    ) -> Result<Self, ConfigError> {
        let standard = if euid == config_trust::ROOT_UID {
            None
        } else {
            Self::find_config_path()
        };
        match config_trust::resolve(euid, explicit, env, standard, meta)? {
            Some(path) => Self::load_from(&path),
            // If no config found, use defaults
            None => Ok(Self::default()),
        }
    }

    /// The config file `load` would use: the first existing standard location.
    pub fn find_config_path() -> Option<PathBuf> {
        let locations = vec![
            PathBuf::from("doorman.toml"),
            dirs::config_dir()
                .map(|p| p.join("doorman/doorman.toml"))
                .unwrap_or_default(),
            PathBuf::from("/etc/doorman/doorman.toml"),
        ];
        locations.into_iter().find(|path| path.exists())
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
        let content = toml::to_string_pretty(self).map_err(std::io::Error::other)?;
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
        assert!(!old.decision.flip_tta);
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
    fn liveness_defaults_keep_current_behaviour_and_are_validated() {
        let old: Config = toml::from_str("[ml]\ndevice = \"cpu\"\n").unwrap();
        assert_eq!(old.liveness.frames, 1);
        assert_eq!(old.liveness.aggregate, LivenessAggregate::Median);
        assert_eq!(old.liveness.relief_stat, ReliefStat::MinMax);
        assert_eq!(old.liveness.depth_input_size, 518);
        assert!(old.liveness.signals.is_empty());
        assert_eq!(old.liveness.mean_mode_depth_samples, 0);
        assert!(old.validate().is_ok());

        let new: Config = toml::from_str(
            "[liveness]\nframes = 3\nrelief_stat = \"p10_p90\"\ndepth_input_size = 308\n",
        )
        .unwrap();
        assert_eq!(new.liveness.relief_stat, ReliefStat::P10P90);
        assert!(new.validate().is_ok());

        let bad = |toml: &str| {
            let cfg: Config = toml::from_str(toml).unwrap();
            cfg.validate().unwrap_err().to_string()
        };
        assert!(bad("[liveness]\nframes = 2\n").contains("liveness.frames"));
        assert!(bad("[liveness]\nframes = 11\n").contains("liveness.frames"));
        assert!(bad("[liveness]\ndepth_input_size = 500\n").contains("multiple of 14"));
        assert!(bad("[liveness]\ndepth_input_size = 0\n").contains("depth_input_size"));
        assert!(bad("[liveness]\nsignals = [\"screen_flash\"]\n").contains("screen_flash"));
        assert!(bad(
            "[authentication]\nauth_frames = 5\n[liveness]\nmean_mode_depth_samples = 6\n"
        )
        .contains("mean_mode_depth_samples"));
        let ok: Config = toml::from_str(
            "[authentication]\nauth_frames = 5\n[liveness]\nmean_mode_depth_samples = 5\n",
        )
        .unwrap();
        assert!(ok.validate().is_ok());
        assert!(toml::from_str::<Config>("[liveness]\nrelief_stat = \"mean\"\n").is_err());
    }

    #[test]
    fn min_brightness_is_off_by_default_and_range_checked() {
        let old: Config = toml::from_str("[recognition]\nmin_sharpness = 8.0\n").unwrap();
        assert_eq!(old.recognition.min_brightness, 0.0);
        assert!(old.validate().is_ok());
        for bad in ["-1.0", "256.0"] {
            let cfg: Config =
                toml::from_str(&format!("[recognition]\nmin_brightness = {bad}\n")).unwrap();
            assert!(cfg
                .validate()
                .unwrap_err()
                .to_string()
                .contains("min_brightness"));
        }
    }

    #[test]
    fn enrollment_prune_cosine_defaults_and_is_validated() {
        let old: Config = toml::from_str("[enrollment]\nenroll_frames = 20\n").unwrap();
        assert_eq!(old.enrollment.prune_cosine, 0.95);
        for bad in ["0.0", "1.5", "-0.1"] {
            let cfg: Config =
                toml::from_str(&format!("[enrollment]\nprune_cosine = {bad}\n")).unwrap();
            let err = cfg.validate().unwrap_err().to_string();
            assert!(err.contains("prune_cosine"), "{err}");
        }
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
            assert!(
                err.contains("accept_n") && err.contains("window_m"),
                "{err}"
            );
        }
        let short_budget: Config =
            toml::from_str("[authentication]\nauth_frames = 2\n[decision]\nmode = \"n_of_m\"\n")
                .unwrap();
        let err = short_budget.validate().unwrap_err().to_string();
        assert!(
            err.contains("auth_frames") && err.contains("accept_n"),
            "{err}"
        );
        assert!(toml::from_str::<Config>("[decision]\nmode = \"sum\"\n").is_err());
        assert!(toml::from_str::<Config>("[decision]\ntemplate_score = \"min\"\n").is_err());
    }

    #[test]
    fn test_camera_backend_choice_and_first_frame_timeout() {
        let old: Config = toml::from_str("[camera]\nwidth = 1024\n").unwrap();
        assert_eq!(old.camera.backend, CameraBackendChoice::Auto);
        assert_eq!(old.camera.native_first_frame_timeout_ms, 1500);

        for (name, want) in [
            ("auto", CameraBackendChoice::Auto),
            ("native", CameraBackendChoice::Native),
            ("ffmpeg", CameraBackendChoice::Ffmpeg),
        ] {
            let cfg: Config = toml::from_str(&format!("[camera]\nbackend = \"{name}\"\n")).unwrap();
            assert_eq!(cfg.camera.backend, want);
        }
        assert!(toml::from_str::<Config>("[camera]\nbackend = \"gstreamer\"\n").is_err());

        let timeout = |native_first_frame_timeout_ms| CameraConfig {
            native_first_frame_timeout_ms,
            ..CameraConfig::default()
        };
        assert!(timeout(MIN_NATIVE_FIRST_FRAME_TIMEOUT_MS)
            .validate()
            .is_ok());
        assert!(timeout(MAX_NATIVE_FIRST_FRAME_TIMEOUT_MS)
            .validate()
            .is_ok());
        for bad in [
            MIN_NATIVE_FIRST_FRAME_TIMEOUT_MS - 1,
            MAX_NATIVE_FIRST_FRAME_TIMEOUT_MS + 1,
        ] {
            let err = timeout(bad).validate().unwrap_err().to_string();
            assert!(
                err.contains("native_first_frame_timeout_ms") && err.contains("100..=10000"),
                "{err}"
            );
        }
    }

    #[test]
    fn test_native_warmup_and_cooldown_default_and_validate() {
        let old: Config = toml::from_str("[camera]\nwidth = 1024\n").unwrap();
        assert_eq!(old.camera.native_warmup_frames, 3);
        assert_eq!(old.camera.native_failure_cooldown_secs, 600);

        let ok: Config = toml::from_str(
            "[camera]\nnative_warmup_frames = 0\nnative_failure_cooldown_secs = 0\n",
        )
        .unwrap();
        assert!(ok.camera.validate().is_ok());

        let bad_warmup = CameraConfig {
            native_warmup_frames: MAX_NATIVE_WARMUP_FRAMES + 1,
            ..CameraConfig::default()
        };
        let err = bad_warmup.validate().unwrap_err().to_string();
        assert!(
            err.contains("native_warmup_frames") && err.contains("0..=10"),
            "{err}"
        );
        let bad_cooldown = CameraConfig {
            native_failure_cooldown_secs: MAX_NATIVE_FAILURE_COOLDOWN_SECS + 1,
            ..CameraConfig::default()
        };
        let err = bad_cooldown.validate().unwrap_err().to_string();
        assert!(
            err.contains("native_failure_cooldown_secs") && err.contains("0..=86400"),
            "{err}"
        );
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

    #[test]
    fn sudo_defaults_are_disabled_and_old_configs_parse() {
        let old: Config = toml::from_str("[ml]\ndevice = \"cpu\"\n").unwrap();
        let sudo = &old.pam.sudo;
        assert!(!sudo.enabled);
        assert_eq!(sudo.service_names, vec!["sudo".to_string()]);
        assert_eq!((sudo.accept_n, sudo.window_m), (4, 5));
        assert_eq!(sudo.liveness_frames, 3);
        assert_eq!(
            (sudo.timeout_ms, sudo.max_failures, sudo.suspend_secs),
            (4000, 3, 600)
        );
        assert_eq!(sudo.effective_threshold(&old.authentication), 0.4);
        assert!(old.validate().is_ok());
    }

    #[test]
    fn sudo_validation_rejects_weaker_or_inconsistent_settings() {
        let bad = |toml: &str| {
            let cfg: Config = toml::from_str(toml).unwrap();
            cfg.validate().unwrap_err().to_string()
        };
        assert!(bad("[pam.sudo]\nsimilarity_threshold = 0.3").contains("similarity_threshold"));
        assert!(bad("[pam.sudo]\naccept_n = 6").contains("accept_n"));
        assert!(bad("[pam.sudo]\nliveness_frames = 1").contains("liveness_frames"));
        assert!(bad("[pam.sudo]\nliveness_frames = 4").contains("liveness_frames"));
        assert!(bad("[pam.sudo]\ntimeout_ms = 4001").contains("timeout_ms"));
        assert!(bad("[pam.sudo]\nmax_failures = 0").contains("max_failures"));
        assert!(bad("[pam.sudo]\nservice_names = [\"su do\"]").contains("service_names"));
        assert!(bad("[pam.sudo]\nenabled = true\nservice_names = []").contains("service_names"));
        assert!(
            bad("[authentication]\nliveness_enabled = false\n[pam.sudo]\nenabled = true")
                .contains("liveness_enabled")
        );
        assert!(
            bad("[authentication]\nauth_frames = 3\n[pam.sudo]\nenabled = true")
                .contains("auth_frames")
        );
        let ok: Config =
            toml::from_str("[pam.sudo]\nenabled = true\nsimilarity_threshold = 0.5").unwrap();
        assert!(ok.validate().is_ok());
        assert_eq!(ok.pam.sudo.effective_threshold(&ok.authentication), 0.5);
    }
}

#[cfg(test)]
mod camera_guard_tests {
    use super::Config;

    #[test]
    fn virtual_devices_are_refused_by_default_and_device_path_is_unset() {
        let cfg = Config::default();
        assert!(!cfg.camera.allow_virtual_devices);
        assert_eq!(cfg.camera.device_path, None);
        let cfg: Config = toml::from_str("").unwrap();
        assert!(!cfg.camera.allow_virtual_devices);
    }

    #[test]
    fn camera_guard_keys_parse() {
        let cfg: Config = toml::from_str(
            "[camera]\nallow_virtual_devices = true\ndevice_path = \"/dev/v4l/by-id/usb-X-video-index0\"\n",
        )
        .unwrap();
        assert!(cfg.camera.allow_virtual_devices);
        assert_eq!(
            cfg.camera.device_path.as_deref(),
            Some("/dev/v4l/by-id/usb-X-video-index0")
        );
    }

    #[test]
    fn prearm_defaults_and_bounds() {
        let cfg = Config::default();
        assert!(cfg.ipc.prearm_camera);
        assert_eq!(cfg.ipc.prearm_max_hold_secs, 5);
        let old: Config = toml::from_str("[ipc]\nwait_for_match_max_ms=1500\n").unwrap();
        assert!(old.ipc.prearm_camera);
        assert_eq!(old.ipc.prearm_max_hold_secs, 5);
        for bad in [0u64, 31] {
            let cfg: Config =
                toml::from_str(&format!("[ipc]\nprearm_max_hold_secs={bad}\n")).unwrap();
            let err = cfg.validate().unwrap_err().to_string();
            assert!(
                err.contains("prearm_max_hold_secs") && err.contains("1..=30"),
                "{err}"
            );
        }
        for ok in [1u64, 30] {
            let cfg: Config = toml::from_str(&format!(
                "[ipc]\nprearm_camera=false\nprearm_max_hold_secs={ok}\n"
            ))
            .unwrap();
            assert!(cfg.validate().is_ok());
            assert!(!cfg.ipc.prearm_camera);
        }
    }
    #[test]
    fn face_budget_defaults_and_bounds() {
        let cfg = Config::default();
        assert_eq!(cfg.ipc.face_max_judged_failures, 5);
        assert_eq!(cfg.ipc.face_failure_window_secs, 300);
        assert_eq!(cfg.ipc.face_cooldown_secs, vec![30, 120, 600]);
        assert_eq!(cfg.ipc.face_budget_quiet_reset_secs, 1_800);
        let old: Config = toml::from_str("[ipc]\nwait_for_match_max_ms=1500\n").unwrap();
        assert_eq!(old.ipc.face_max_judged_failures, 5);
        assert!(old.validate().is_ok());
        for bad in [
            "face_max_judged_failures=101",
            "face_failure_window_secs=9",
            "face_failure_window_secs=86401",
            "face_budget_quiet_reset_secs=59",
            "face_cooldown_secs=[]",
            "face_cooldown_secs=[0]",
            "face_cooldown_secs=[600,30]",
            "face_cooldown_secs=[1,2,3,4,5,6,7,8,9]",
            "face_cooldown_secs=[86401]",
        ] {
            let cfg: Config = toml::from_str(&format!("[ipc]\n{bad}\n")).unwrap();
            let err = cfg.validate().unwrap_err().to_string();
            assert!(err.contains("ipc.face_"), "{bad}: {err}");
        }
        for ok in [
            "face_max_judged_failures=0",
            "face_max_judged_failures=100",
            "face_cooldown_secs=[5]",
            "face_cooldown_secs=[30,30,120]",
        ] {
            let cfg: Config = toml::from_str(&format!("[ipc]\n{ok}\n")).unwrap();
            assert!(cfg.validate().is_ok(), "{ok}");
        }
    }
    #[test]
    fn mean_mode_hardening_defaults_and_bounds() {
        let cfg = Config::default();
        assert_eq!(cfg.decision.frame_identity_floor, 0.30);
        assert_eq!(cfg.authentication.min_decision_frames, 0);
        assert_eq!(cfg.authentication.effective_min_decision_frames(), 8);
        assert_eq!(cfg.camera.max_frame_age_ms, 1_000);
        assert_eq!(
            cfg.recognition.multi_face_policy,
            crate::config::MultiFacePolicy::Reject
        );
        assert_eq!(cfg.recognition.multi_face_min_area_ratio, 0.25);
        assert!(!cfg.models.allow_unverified_models);
        assert!(cfg.models.manifest_path.is_none());
        assert!(cfg.validate().is_ok());
        let one: Config = toml::from_str("[authentication]\nauth_frames=1\n").unwrap();
        assert_eq!(one.authentication.effective_min_decision_frames(), 1);
        let explicit: Config =
            toml::from_str("[authentication]\nauth_frames=10\nmin_decision_frames=10\n").unwrap();
        assert_eq!(explicit.authentication.effective_min_decision_frames(), 10);
        for bad in [
            "[decision]\nframe_identity_floor=-0.1",
            "[decision]\nframe_identity_floor=1.5",
            "[decision]\nframe_identity_floor=0.5",
            "[authentication]\nmin_decision_frames=11",
            "[camera]\nmax_frame_age_ms=99",
            "[camera]\nmax_frame_age_ms=10001",
            "[recognition]\nmulti_face_min_area_ratio=0.0",
            "[recognition]\nmulti_face_min_area_ratio=1.5",
            "[recognition]\nmulti_face_min_score=1.5",
        ] {
            let cfg: Config = toml::from_str(bad).unwrap();
            assert!(cfg.validate().is_err(), "{bad}");
        }
        for ok in [
            "[decision]\nframe_identity_floor=0.0",
            "[camera]\nmax_frame_age_ms=0",
            "[camera]\nmax_frame_age_ms=100",
            "[recognition]\nmulti_face_policy=\"largest\"",
            "[recognition]\nmulti_face_policy=\"off\"",
        ] {
            let cfg: Config = toml::from_str(ok).unwrap();
            assert!(cfg.validate().is_ok(), "{ok}");
        }
        assert!(toml::from_str::<Config>("[recognition]\nmulti_face_policy=\"x\"").is_err());
    }
}
