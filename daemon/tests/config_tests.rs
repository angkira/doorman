use doorman_shared::Config;
use tempfile::TempDir;

#[test]
fn test_default_config() {
    let config = Config::default();

    assert_eq!(config.daemon.socket_path, "/run/doorman/doorman.sock");
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

    assert!(!config.dataset.logging_enabled);
    assert!(!config.dataset.keep_raw_frames);
    assert_eq!(config.dataset.raw_frame_bundle_size, 10);
    assert_eq!(config.dataset.max_capture_packs_per_day, 500);
    assert_eq!(config.dataset.raw_frame_jpeg_quality, 92);
    assert_eq!(config.dataset.max_attempt_records_per_day, 50_000);
}

#[test]
fn test_config_serialization() {
    let config = Config::default();
    let toml_str = toml::to_string(&config).unwrap();
    let parsed: Config = toml::from_str(&toml_str).unwrap();

    assert_eq!(config.daemon.socket_path, parsed.daemon.socket_path);
    assert_eq!(config.ml.device, parsed.ml.device);
    assert_eq!(
        config.authentication.similarity_threshold,
        parsed.authentication.similarity_threshold
    );
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
    assert_eq!(
        config.camera.video_file,
        Some("/path/to/test.mp4".to_string())
    );
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
    assert_eq!(config.daemon.socket_path, "/run/doorman/doorman.sock"); // Should use default
    assert_eq!(config.authentication.auth_frames, 10); // Should use default
}

#[test]
fn test_proactive_unlock_defaults_to_false() {
    // Safety requirement: proactive_unlock MUST default to false so a fresh
    // install never auto-unlocks without explicit opt-in (--auto-unlock).
    let config = Config::default();
    assert!(
        !config.authentication.proactive_unlock,
        "proactive_unlock must default to false (safe default for public repo)"
    );
}

#[test]
fn test_proactive_unlock_can_be_set_true() {
    // Verify the config key round-trips when the user enables it.
    let toml_str = r#"
        [authentication]
        proactive_unlock = true
    "#;
    let config: Config = toml::from_str(toml_str).unwrap();
    assert!(config.authentication.proactive_unlock);
    // Remaining auth fields should keep their defaults.
    assert_eq!(config.authentication.similarity_threshold, 0.4);
    assert!(config.authentication.liveness_enabled);
}

#[test]
fn test_dataset_logging_defaults_off_and_can_be_enabled() {
    let default_config = Config::default();
    assert!(!default_config.dataset.logging_enabled);
    assert!(!default_config.dataset.keep_raw_frames);

    let toml_str = r#"
        [dataset]
        logging_enabled = true
        keep_raw_frames = true
        raw_frame_bundle_size = 4
        max_capture_packs_per_day = 9
        raw_frame_jpeg_quality = 87
        max_attempt_records_per_day = 12
    "#;
    let config: Config = toml::from_str(toml_str).unwrap();
    assert!(config.dataset.logging_enabled);
    assert!(config.dataset.keep_raw_frames);
    assert_eq!(config.dataset.raw_frame_bundle_size, 4);
    assert_eq!(config.dataset.max_capture_packs_per_day, 9);
    assert_eq!(config.dataset.raw_frame_jpeg_quality, 87);
    assert_eq!(config.dataset.max_attempt_records_per_day, 12);
}
