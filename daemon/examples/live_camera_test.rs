use anyhow::Result;
use doorman_shared::Config;
use image::GenericImageView;

// Import the camera module
use doormand::camera::Camera;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🎥 Live Camera Test\n");
    
    // Load config (or use defaults)
    let config = Config::load().unwrap_or_else(|_| {
        println!("Using default configuration");
        Config::default()
    });
    
    println!("Camera config:");
    println!("  Device: {}", config.camera.device_index);
    println!("  Resolution: {}x{}", config.camera.width, config.camera.height);
    println!("  FPS: {}\n", config.camera.fps);
    
    // Initialize camera
    println!("Initializing camera...");
    let mut camera = Camera::new_with_config(&config).await?;
    println!("✅ Camera initialized!\n");
    
    // Capture multiple frames
    println!("Capturing 5 test frames...");
    for i in 1..=5 {
        match camera.capture_frame() {
            Ok(frame) => {
                let (w, h) = frame.dimensions();
                println!("  Frame {}: {}x{} captured", i, w, h);
            }
            Err(e) => {
                eprintln!("  Frame {}: Error - {}", i, e);
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }
    
    println!("\n🎉 Live camera test completed successfully!");
    println!("\nNext steps:");
    println!("  1. Download ONNX models (see MODELS.md)");
    println!("  2. Run daemon: sudo target/release/doormand");
    println!("  3. Enroll face: uv run doorman enroll <username>");
    
    Ok(())
}

