/// Comprehensive pipeline test - trace all data transformations
use anyhow::Result;
use doorman_shared::Config;
use doormand::ml::MLPipeline;
use image::{DynamicImage, GenericImageView};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    println!("\n=== DOORMAN PIPELINE DEBUG ===\n");

    // Load config
    let home = env::var("HOME").unwrap();
    let mut config = Config::default();
    config.ml.models_dir = format!("{}/Home/doorman/data/models", home);

    println!("Configuration:");
    println!("  Models dir: {}", config.ml.models_dir);
    println!("  Backend: {}", config.ml.backend);
    println!("  Device: {}", config.ml.device);

    // Load ML pipeline
    println!("\nInitializing ML pipeline...");
    let pipeline = MLPipeline::new(&config).await?;
    println!("✓ ML pipeline initialized\n");

    // Test with video frames
    for frame_num in 1..=3 {
        println!("========================================");
        println!("FRAME {}", frame_num);
        println!("========================================");

        let frame_path = format!("/tmp/test_frame_{}.png", frame_num);

        match image::open(&frame_path) {
            Ok(img) => {
                println!("\n1. INPUT IMAGE");
                println!("   Path: {}", frame_path);
                println!("   Dimensions: {}x{}", img.width(), img.height());
                println!("   Format: {:?}", img.color());
                println!("   Size: {} bytes", img.as_bytes().len());

                // Get first few pixels to verify data
                let rgb_img = img.to_rgb8();
                print!("   First 3 pixels (RGB): ");
                for i in 0..3 {
                    let pixel = rgb_img.get_pixel(i, 0);
                    print!("[{},{},{}] ", pixel[0], pixel[1], pixel[2]);
                }
                println!();

                println!("\n2. PROCESSING");
                println!("   Running through ML pipeline...");

                match pipeline.process_frame(&img).await {
                    Ok(Some((face, embedding))) => {
                        println!("   ✓ DETECTION SUCCESS");
                        println!("   Face bbox: {:?}", face.bbox);
                        println!("   Face confidence: {:.3}", face.confidence);
                        println!("   Embedding dimensions: {}", embedding.len());
                        println!("   Embedding sample: {:?}", &embedding[0..5.min(embedding.len())]);
                    }
                    Ok(None) => {
                        println!("   ✗ NO DETECTION");
                        println!("   (Either no face detected or failed liveness)");
                    }
                    Err(e) => {
                        println!("   ✗ ERROR: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("Frame {} not found: {}", frame_num, e);
            }
        }
        println!();
    }

    println!("\n=== TEST COMPLETE ===\n");
    Ok(())
}
