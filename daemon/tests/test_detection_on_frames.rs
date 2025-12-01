use doormand::ml::MLPipeline;
use doorman_shared::Config;
use image;
use std::env;

#[tokio::test]
async fn test_video_frame_detection() {
    let home = env::var("HOME").unwrap();
    let mut config = Config::default();
    config.ml.models_dir = format!("{}/.local/share/doorman/models", home);

    let pipeline = MLPipeline::new(&config).await.expect("Failed to create ML pipeline");

    for i in 1..=5 {
        let frame_path = format!("/tmp/test_frame_{}.png", i);

        match image::open(&frame_path) {
            Ok(img) => {
                println!("\n=== Frame {} ===", i);
                match pipeline.process_frame(&img).await {
                    Ok(Some(embedding)) => {
                        println!("✓ Face detected and recognized!");
                        println!("  Embedding size: {}", embedding.len());
                    }
                    Ok(None) => {
                        println!("✗ No face detected or failed liveness");
                    }
                    Err(e) => {
                        println!("✗ Error: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("Frame {} not found: {}", i, e);
            }
        }
    }
}
