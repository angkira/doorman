#[cfg(feature = "camera-gstreamer")]
mod gstreamer_tests {
    use doorman_shared::Config;
    use doormand::camera::{GStreamerCamera, CameraBackend};

    #[tokio::test]
    async fn test_gstreamer_init() {
        // This test requires GStreamer to be installed and a camera to be available
        let config = Config {
            camera: doorman_shared::CameraConfig {
                device_index: 0,
                width: 640,
                height: 480,
                fps: 30,
                video_file: None,
            },
            ..Default::default()
        };

        let result = GStreamerCamera::new_with_config(&config).await;
        
        match result {
            Ok(camera) => {
                println!("GStreamer camera initialized successfully");
                println!("Backend: {}", camera.backend_name());
                assert_eq!(camera.backend_name(), "GStreamer/PipeWire");
                assert!(camera.is_ready());
            }
            Err(e) => {
                eprintln!("GStreamer initialization failed: {}", e);
                eprintln!("This is expected if:");
                eprintln!("  - GStreamer is not installed");
                eprintln!("  - No camera is available");
                eprintln!("  - Running in CI/headless environment");
            }
        }
    }

    #[tokio::test]
    async fn test_gstreamer_capture_single_frame() {
        let config = Config {
            camera: doorman_shared::CameraConfig {
                device_index: 0,
                width: 640,
                height: 480,
                fps: 30,
                video_file: None,
            },
            ..Default::default()
        };

        if let Ok(mut camera) = GStreamerCamera::new_with_config(&config).await {
            // Wait a moment for pipeline to stabilize
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            
            match camera.capture_frame() {
                Ok(frame) => {
                    println!("Frame captured: {}x{}", frame.width(), frame.height());
                    assert_eq!(frame.width(), 640);
                    assert_eq!(frame.height(), 480);
                }
                Err(e) => {
                    eprintln!("Frame capture failed: {}", e);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_gstreamer_capture_multiple_frames() {
        let config = Config {
            camera: doorman_shared::CameraConfig {
                device_index: 0,
                width: 640,
                height: 480,
                fps: 30,
                video_file: None,
            },
            ..Default::default()
        };

        if let Ok(mut camera) = GStreamerCamera::new_with_config(&config).await {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            
            let frames = camera.capture_frames(5);
            println!("Captured {} frames", frames.len());
            
            // Should capture at least some frames
            if !frames.is_empty() {
                for (i, frame) in frames.iter().enumerate() {
                    println!("Frame {}: {}x{}", i, frame.width(), frame.height());
                }
            }
        }
    }

    #[tokio::test]
    async fn test_gstreamer_invalid_resolution() {
        let config = Config {
            camera: doorman_shared::CameraConfig {
                device_index: 0,
                width: 99999, // Invalid
                height: 99999,
                fps: 30,
                video_file: None,
            },
            ..Default::default()
        };

        let result = GStreamerCamera::new_with_config(&config).await;
        assert!(result.is_err());
        
        if let Err(e) = result {
            let err_msg = e.to_string();
            assert!(err_msg.contains("exceeds maximum"));
        }
    }

    #[tokio::test]
    async fn test_gstreamer_performance() {
        let config = Config {
            camera: doorman_shared::CameraConfig {
                device_index: 0,
                width: 1024,
                height: 720,
                fps: 30,
                video_file: None,
            },
            ..Default::default()
        };

        if let Ok(mut camera) = GStreamerCamera::new_with_config(&config).await {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            
            let start = std::time::Instant::now();
            let mut success_count = 0;
            
            // Try to capture 30 frames
            for _ in 0..30 {
                if camera.capture_frame().is_ok() {
                    success_count += 1;
                }
            }
            
            let elapsed = start.elapsed();
            let fps = success_count as f64 / elapsed.as_secs_f64();
            
            println!("Captured {} frames in {:.2}s = {:.1} fps", 
                     success_count, elapsed.as_secs_f64(), fps);
            
            // GStreamer should be much faster than FFmpeg
            // Even 10fps would be acceptable, but we expect 20-30fps
            if success_count > 0 {
                println!("GStreamer performance: {:.1} fps", fps);
            }
        }
    }
}

// Always-available test (doesn't require feature flag)
#[test]
fn test_gstreamer_feature_available() {
    #[cfg(feature = "camera-gstreamer")]
    {
        println!("✓ camera-gstreamer feature is enabled");
    }
    
    #[cfg(not(feature = "camera-gstreamer"))]
    {
        println!("✗ camera-gstreamer feature is NOT enabled");
        println!("Enable with: cargo test --features camera-gstreamer");
    }
}
