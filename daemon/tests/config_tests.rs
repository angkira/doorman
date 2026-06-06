use doorman_shared::Config;
use tempfile::TempDir;
use std::fs;

#[test]
fn test_default_config() {
    let config = Config::default();
    
    assert_eq!(config.daemon.socket_path, "/run/doorman.sock");
    assert_eq!(config.daemon.data_dir, "/var/lib/doorman");
    assert_eq!(config.daemon.log_level, "info");
    
    assert_eq!(config.camera.device_index, 0);
    assert_eq!(config.camera.video_file, None);
    
    assert_eq!(config.ml.device, "cpu");
    assert_eq!(config.ml.cpu_threads, 0);
    assert_eq!(config.ml.gpu_device_id, 0);
    
    assert_eq!(config.authentication.similarity_threshold, 0.4);
    assert_eq!(config.authentication.auth_frames, 10);
    assert_eq!(config.authentication.timeout_secs, 3);
    
    assert_eq!(config.enrollment.enroll_frames, 20);
    assert_eq!(config.enrollment.min_valid_frames, 5);
    
    assert_eq!(config.preprocessing.image_width, 256);
    assert_eq!(config.preprocessing.image_height, 256);
}

#[test]
fn test_config_serialization() {
    let config = Config::default();
    let toml_str = toml::to_string(&config).unwrap();
    let parsed: Config = toml::from_str(&toml_str).unwrap();
    
    assert_eq!(config.daemon.socket_path, parsed.daemon.socket_path);
    assert_eq!(config.ml.device, parsed.ml.device);
    assert_eq!(config.authentication.similarity_threshold, parsed.authentication.similarity_threshold);
}

#[test]
fn test_config_gpu_rocm() {
    let toml_str = r#"
        [ml]
        device = "rocm"
        gpu_device_id = 0
    "#;
    
    let config: Config = toml::from_str(toml_str).unwrap();
    assert_eq!(config.ml.device, "rocm");
    assert_eq!(config.ml.gpu_device_id, 0);
}

#[test]
fn test_config_gpu_cuda() {
    let toml_str = r#"
        [ml]
        device = "cuda"
        gpu_device_id = 1
    "#;
    
    let config: Config = toml::from_str(toml_str).unwrap();
    assert_eq!(config.ml.device, "cuda");
    assert_eq!(config.ml.gpu_device_id, 1);
}

#[test]
fn test_config_custom_threshold() {
    let toml_str = r#"
        [authentication]
        similarity_threshold = 0.8
        auth_frames = 15
    "#;
    
    let config: Config = toml::from_str(toml_str).unwrap();
    assert_eq!(config.authentication.similarity_threshold, 0.8);
    assert_eq!(config.authentication.auth_frames, 15);
}

#[test]
fn test_config_video_file() {
    let toml_str = r#"
        [camera]
        video_file = "/path/to/test.mp4"
    "#;
    
    let config: Config = toml::from_str(toml_str).unwrap();
    assert_eq!(config.camera.video_file, Some("/path/to/test.mp4".to_string()));
}

#[test]
fn test_config_save_and_load() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("doorman.toml");
    
    // Create and save config
    let mut config = Config::default();
    config.ml.device = "rocm".to_string();
    config.authentication.similarity_threshold = 0.75;
    
    config.save_to(&config_path).unwrap();
    
    // Load and verify
    let loaded = Config::load_from(&config_path).unwrap();
    assert_eq!(loaded.ml.device, "rocm");
    assert_eq!(loaded.authentication.similarity_threshold, 0.75);
}

#[test]
fn test_config_preprocessing() {
    let toml_str = r#"
        [preprocessing]
        image_width = 512
        image_height = 512
        filter_type = "gaussian"
    "#;
    
    let config: Config = toml::from_str(toml_str).unwrap();
    assert_eq!(config.preprocessing.image_width, 512);
    assert_eq!(config.preprocessing.image_height, 512);
    assert_eq!(config.preprocessing.filter_type, "gaussian");
}

#[test]
fn test_config_partial() {
    // Test that missing sections use defaults
    let toml_str = r#"
        [ml]
        device = "cuda"
    "#;
    
    let config: Config = toml::from_str(toml_str).unwrap();
    assert_eq!(config.ml.device, "cuda");
    assert_eq!(config.daemon.socket_path, "/run/doorman.sock"); // Should use default
    assert_eq!(config.authentication.auth_frames, 10); // Should use default
}

