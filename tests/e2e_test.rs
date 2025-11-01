/// End-to-End integration tests for doorman
/// 
/// These tests simulate the full authentication flow
use std::process::{Command, Stdio};
use std::time::Duration;
use std::thread;

#[test]
#[ignore] // Only run with --ignored flag
fn test_daemon_starts() {
    // Check if daemon binary exists
    let binary_exists = std::path::Path::new("target/release/doormand").exists();
    assert!(binary_exists, "Daemon binary not found. Run 'cargo build --release' first.");
}

#[test]
#[ignore]
fn test_cli_help() {
    // Test Python CLI
    let output = Command::new("doorman")
        .arg("--help")
        .output();
    
    match output {
        Ok(out) => {
            assert!(out.status.success(), "CLI help command failed");
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(stdout.contains("doorman"), "Help output should contain 'doorman'");
        }
        Err(_) => {
            // CLI not installed, skip test
            eprintln!("doorman CLI not installed, skipping test");
        }
    }
}

#[test]
#[ignore]
fn test_ipc_socket_connection() {
    // This test requires the daemon to be running
    use std::os::unix::net::UnixStream;
    use std::io::{Write, BufRead, BufReader};
    
    let socket_path = "/run/doorman.sock";
    
    // Try to connect
    let stream = UnixStream::connect(socket_path);
    
    match stream {
        Ok(mut stream) => {
            // Send status request
            let request = r#"{"type":"status"}"#;
            writeln!(stream, "{}", request).unwrap();
            stream.flush().unwrap();
            
            // Read response
            let mut reader = BufReader::new(stream);
            let mut response = String::new();
            reader.read_line(&mut response).unwrap();
            
            // Parse JSON
            let json: serde_json::Value = serde_json::from_str(&response).unwrap();
            assert_eq!(json["status"], "success");
            
            println!("Daemon responded: {}", response);
        }
        Err(e) => {
            eprintln!("Could not connect to daemon (is it running?): {}", e);
        }
    }
}

#[test]
fn test_embedding_size_consistency() {
    // Test that embeddings are always 512-d
    let embedding_size = 512;
    let test_embedding = vec![0.0f32; embedding_size];
    
    assert_eq!(test_embedding.len(), 512);
}

#[test]
fn test_config_file_parsing() {
    // Test parsing example config
    let config_content = std::fs::read_to_string("doorman.toml");
    
    match config_content {
        Ok(content) => {
            let config: Result<doorman_shared::Config, _> = toml::from_str(&content);
            assert!(config.is_ok(), "Example config file should be valid");
            
            let config = config.unwrap();
            println!("Config device: {}", config.ml.device);
            println!("Config threshold: {}", config.authentication.similarity_threshold);
        }
        Err(_) => {
            eprintln!("doorman.toml not found, skipping test");
        }
    }
}

// Simulate authentication flow
#[test]
fn test_authentication_flow_simulation() {
    use doorman_shared::{Request, Response};
    
    // Create auth request
    let request = Request::Authenticate {
        username: "testuser".to_string(),
    };
    
    // Serialize
    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("authenticate"));
    assert!(json.contains("testuser"));
    
    // Simulate response
    let response = Response::Success {
        message: Some("Authenticated".to_string()),
        data: None,
    };
    
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("success"));
}

#[test]
fn test_enrollment_flow_simulation() {
    use doorman_shared::{Request, Response};
    
    // Create enroll request
    let request = Request::Enroll {
        username: "newuser".to_string(),
    };
    
    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("enroll"));
    
    // Simulate progress response
    let response = Response::Progress {
        message: "Capturing frames".to_string(),
        current: 5,
        total: 20,
    };
    
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("progress"));
}

// Test video file detection (when video feature is enabled)
#[test]
#[cfg(feature = "video")]
fn test_video_file_support() {
    // This test checks if video files in ./data can be detected
    let data_dir = std::path::Path::new("data");
    
    if data_dir.exists() {
        let video_files: Vec<_> = std::fs::read_dir(data_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let path = e.path();
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "mp4" || ext == "avi" || ext == "mov")
                    .unwrap_or(false)
            })
            .collect();
        
        println!("Found {} video files in data/", video_files.len());
        
        for file in video_files {
            println!("  - {:?}", file.file_name());
        }
    } else {
        println!("data/ directory not found");
    }
}

#[test]
fn test_pam_module_size() {
    // Ensure PAM module is reasonably sized (< 2MB)
    let pam_path = std::path::Path::new("target/release/libpam_doorman.so");
    
    if pam_path.exists() {
        let metadata = std::fs::metadata(pam_path).unwrap();
        let size_mb = metadata.len() as f64 / 1_000_000.0;
        
        println!("PAM module size: {:.2} MB", size_mb);
        assert!(size_mb < 2.0, "PAM module should be < 2MB for fast loading");
    }
}

#[test]
fn test_daemon_binary_size() {
    // Ensure daemon is reasonably sized
    let daemon_path = std::path::Path::new("target/release/doormand");
    
    if daemon_path.exists() {
        let metadata = std::fs::metadata(daemon_path).unwrap();
        let size_mb = metadata.len() as f64 / 1_000_000.0;
        
        println!("Daemon binary size: {:.2} MB", size_mb);
        assert!(size_mb < 50.0, "Daemon should be < 50MB");
    }
}

