/// Tests for video file input support
#[cfg(feature = "video")]
mod video_file_tests {
    use std::path::Path;

    #[test]
    fn test_data_directory_exists() {
        // Check if data directory exists (created by user)
        let data_dir = Path::new("data");
        
        if data_dir.exists() {
            assert!(data_dir.is_dir(), "data should be a directory");
            println!("✓ data/ directory found");
        } else {
            println!("⚠ data/ directory not found (create it to add test videos)");
        }
    }

    #[test]
    fn test_find_mp4_files() {
        let data_dir = Path::new("data");
        
        if !data_dir.exists() {
            println!("Skipping: data/ directory not found");
            return;
        }

        let entries = std::fs::read_dir(data_dir).unwrap();
        let mut video_count = 0;

        for entry in entries {
            let entry = entry.unwrap();
            let path = entry.path();
            
            if let Some(ext) = path.extension() {
                if ext == "mp4" {
                    video_count += 1;
                    println!("Found video: {:?}", path.file_name().unwrap());
                }
            }
        }

        println!("Total MP4 files found: {}", video_count);
        
        if video_count == 0 {
            println!("⚠ No MP4 files found in data/");
            println!("  Add test videos to data/ directory to run video tests");
        }
    }

    #[test]
    fn test_video_file_metadata() {
        let data_dir = Path::new("data");
        
        if !data_dir.exists() {
            println!("Skipping: data/ directory not found");
            return;
        }

        for entry in std::fs::read_dir(data_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            
            if path.extension().and_then(|e| e.to_str()) == Some("mp4") {
                let metadata = std::fs::metadata(&path).unwrap();
                let size_mb = metadata.len() as f64 / 1_000_000.0;
                
                println!("Video: {:?}", path.file_name().unwrap());
                println!("  Size: {:.2} MB", size_mb);
                
                // Basic sanity checks
                assert!(size_mb > 0.0, "Video file should not be empty");
                assert!(size_mb < 1000.0, "Video file seems unusually large");
            }
        }
    }
}

#[cfg(not(feature = "video"))]
mod video_disabled_tests {
    #[test]
    fn test_video_feature_not_enabled() {
        println!("Video support not compiled.");
        println!("To enable video support, rebuild with:");
        println!("  cargo build --features video");
    }
}

// Test video configuration in TOML
#[test]
fn test_video_config_parsing() {
    let config_with_video = r#"
        [camera]
        video_file = "data/test_video.mp4"
    "#;
    
    let config: doorman_shared::Config = toml::from_str(config_with_video).unwrap();
    assert_eq!(config.camera.video_file, Some("data/test_video.mp4".to_string()));
    println!("✓ Video file configuration parses correctly");
}

// Simulate video-based authentication
#[test]
fn test_video_auth_workflow() {
    use doorman_shared::Config;
    
    // Create config with video file
    let mut config = Config::default();
    config.camera.video_file = Some("data/test_face.mp4".to_string());
    
    // Verify config
    assert!(config.camera.video_file.is_some());
    
    let video_path = config.camera.video_file.unwrap();
    println!("Would use video file: {}", video_path);
    
    // In real use, the daemon would:
    // 1. Check if video_file is set
    // 2. Use VideoReader instead of Camera
    // 3. Read frames from video
    // 4. Process frames through ML pipeline
    // 5. Return auth result
}

// Test that we can enumerate all video files in data/
#[test]
fn test_enumerate_test_videos() {
    let data_dir = std::path::Path::new("data");
    
    if !data_dir.exists() {
        println!("✓ data/ directory will be created when you add test videos");
        return;
    }

    let video_extensions = ["mp4", "avi", "mov", "mkv"];
    let mut videos = Vec::new();

    if let Ok(entries) = std::fs::read_dir(data_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if video_extensions.contains(&ext) {
                    videos.push(path);
                }
            }
        }
    }

    if videos.is_empty() {
        println!("No test videos found in data/");
        println!("Add MP4 files to data/ to enable video-based testing:");
        println!("  mkdir -p data");
        println!("  cp /path/to/test_video.mp4 data/");
    } else {
        println!("Found {} test video(s):", videos.len());
        for video in &videos {
            println!("  - {:?}", video.file_name().unwrap());
        }
    }
}

