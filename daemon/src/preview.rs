#!/usr/bin/env rust
//! Live camera preview with face detection visualization
//!
//! Shows:
//! - Real-time camera feed
//! - Face detection bounding boxes (green = detected, red = not detected)
//! - Liveness status
//! - Frame processing stats
//!
//! Usage: doorman-preview
//! Press 'q' to quit

use anyhow::{Context, Result};
use doorman_shared::Config;
use doormand::{camera::Camera, ml::MLPipeline};
use image::{DynamicImage, ImageBuffer, Rgb};
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size, Vector},
    highgui,
    imgproc,
    prelude::*,
};
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const WINDOW_NAME: &str = "Doorman - Camera Preview";
const GREEN: Scalar = Scalar::new(0.0, 255.0, 0.0, 0.0); // BGR format
const RED: Scalar = Scalar::new(0.0, 0.0, 255.0, 0.0);
const YELLOW: Scalar = Scalar::new(0.0, 255.0, 255.0, 0.0);
const WHITE: Scalar = Scalar::new(255.0, 255.0, 255.0, 0.0);

fn dynamic_image_to_mat(img: &DynamicImage) -> Result<Mat> {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    
    // Convert RGB to BGR for OpenCV
    let mut bgr_data = Vec::with_capacity((width * height * 3) as usize);
    for pixel in rgb_img.pixels() {
        bgr_data.push(pixel[2]); // B
        bgr_data.push(pixel[1]); // G
        bgr_data.push(pixel[0]); // R
    }
    
    let mat = Mat::from_slice(&bgr_data)?;
    let mat = mat.reshape(3, height as i32)?;
    let size = Size::new(width as i32, height as i32);
    let mut resized = Mat::default();
    opencv::imgproc::resize(&mat, &mut resized, size, 0.0, 0.0, opencv::imgproc::INTER_LINEAR)?;
    
    Ok(resized)
}

fn draw_text(mat: &mut Mat, text: &str, position: Point, color: Scalar) -> Result<()> {
    imgproc::put_text(
        mat,
        text,
        position,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "doorman_preview=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting Doorman camera preview...");

    // Load configuration
    let config = Config::load().unwrap_or_default();
    
    // Initialize ML pipeline
    info!("Initializing ML pipeline...");
    let ml_pipeline = match MLPipeline::new(&config).await {
        Ok(pipeline) => pipeline,
        Err(e) => {
            error!("Failed to initialize ML pipeline: {}", e);
            warn!("Preview will show camera feed without face detection");
            MLPipeline::dummy(&config)
        }
    };
    
    // Initialize camera
    info!("Initializing camera...");
    let mut camera = Camera::new_with_config(&config)
        .await
        .context("Failed to initialize camera")?;
    
    info!(
        "Camera initialized: {}x{} @ {}fps",
        config.camera.width, config.camera.height, config.camera.fps
    );

    // Create OpenCV window
    highgui::named_window(WINDOW_NAME, highgui::WINDOW_AUTOSIZE)?;
    
    info!("Preview started. Press 'q' to quit.");
    
    let mut frame_count = 0u64;
    let start_time = std::time::Instant::now();
    
    loop {
        // Capture frame
        let frame = match camera.capture_frame() {
            Ok(f) => f,
            Err(e) => {
                warn!("Failed to capture frame: {}", e);
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
        };
        
        frame_count += 1;
        
        // Convert to OpenCV Mat
        let mut mat = match dynamic_image_to_mat(&frame) {
            Ok(m) => m,
            Err(e) => {
                error!("Failed to convert image: {}", e);
                continue;
            }
        };
        
        // Process frame through ML pipeline
        let process_start = std::time::Instant::now();
        let result = ml_pipeline.process_frame(&frame).await;
        let process_time = process_start.elapsed();
        
        // Draw results
        match result {
            Ok(Some(_embedding)) => {
                // Face detected and passed liveness
                let status_text = format!("✓ Face Detected & Live ({:.0}ms)", process_time.as_millis());
                draw_text(&mut mat, &status_text, Point::new(10, 30), GREEN)?;
                
                // Draw green bounding box (approximate center area)
                let (width, height) = (mat.cols(), mat.rows());
                let box_w = width / 2;
                let box_h = height / 2;
                let box_x = (width - box_w) / 2;
                let box_y = (height - box_h) / 2;
                
                imgproc::rectangle(
                    &mut mat,
                    Rect::new(box_x, box_y, box_w, box_h),
                    GREEN,
                    3,
                    imgproc::LINE_8,
                    0,
                )?;
            }
            Ok(None) => {
                // No face detected or failed liveness
                let status_text = format!("✗ No Face / Not Live ({:.0}ms)", process_time.as_millis());
                draw_text(&mut mat, &status_text, Point::new(10, 30), RED)?;
            }
            Err(e) => {
                let status_text = format!("⚠ Error: {} ({:.0}ms)", e, process_time.as_millis());
                draw_text(&mut mat, &status_text, Point::new(10, 30), YELLOW)?;
            }
        }
        
        // Draw FPS counter
        let elapsed = start_time.elapsed().as_secs_f64();
        let fps = if elapsed > 0.0 {
            frame_count as f64 / elapsed
        } else {
            0.0
        };
        let fps_text = format!("FPS: {:.1}", fps);
        draw_text(&mut mat, &fps_text, Point::new(10, 60), WHITE)?;
        
        // Draw instructions
        draw_text(&mut mat, "Press 'q' to quit", Point::new(10, mat.rows() - 20), WHITE)?;
        
        // Display frame
        highgui::imshow(WINDOW_NAME, &mat)?;
        
        // Check for quit key
        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            // 'q' or ESC
            info!("Quit requested");
            break;
        }
    }
    
    // Cleanup
    highgui::destroy_all_windows()?;
    info!("Preview stopped");
    
    Ok(())
}

