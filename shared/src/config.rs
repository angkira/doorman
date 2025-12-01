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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
    
    #[serde(default = "default_auth_frames")]
    pub auth_frames: usize,
    
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
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

// Default values
fn default_socket_path() -> String { "/run/doorman.sock".to_string() }
fn default_debug_socket() -> String { "/run/doorman-debug.sock".to_string() }
fn default_frame_socket() -> String { "/run/doorman-frames.sock".to_string() }
fn default_processing_fps() -> u32 { 10 }
fn default_data_dir() -> String { "/var/lib/doorman".to_string() }
fn default_log_level() -> String { "info".to_string() }
fn default_models_dir() -> String { "/var/lib/doorman/models".to_string() }
fn default_backend() -> String { "tract".to_string() }
fn default_device() -> String { "cpu".to_string() }
fn default_similarity_threshold() -> f32 { 0.65 }
fn default_auth_frames() -> usize { 10 }
fn default_timeout_secs() -> u64 { 3 }
fn default_enroll_frames() -> usize { 20 }
fn default_min_valid_frames() -> usize { 5 }
fn default_image_size() -> u32 { 256 }
fn default_filter_type() -> String { "lanczos3".to_string() }
fn default_camera_width() -> u32 { 1280 }
fn default_camera_height() -> u32 { 720 }
fn default_fps() -> u32 { 30 }

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
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: default_similarity_threshold(),
            auth_frames: default_auth_frames(),
            timeout_secs: default_timeout_secs(),
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

impl Default for Config {
    fn default() -> Self {
        Self {
            daemon: DaemonConfig::default(),
            camera: CameraConfig::default(),
            ml: MLConfig::default(),
            authentication: AuthConfig::default(),
            enrollment: EnrollmentConfig::default(),
            preprocessing: PreprocessingConfig::default(),
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
        assert_eq!(config.daemon.socket_path, "/run/doorman.sock");
        assert_eq!(config.ml.device, "cpu");
        assert_eq!(config.authentication.similarity_threshold, 0.65);
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
}

