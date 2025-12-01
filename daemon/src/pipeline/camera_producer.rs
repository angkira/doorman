use super::types::RawFrame;
use crate::camera::Camera;
use anyhow::Result;
use doorman_shared::Config;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};

/// Camera producer task
/// Owns camera exclusively and captures frames at camera native FPS
/// Never blocks on downstream consumers
pub async fn run_camera_producer(
    camera: Arc<RwLock<Option<Camera>>>,
    frame_tx: mpsc::Sender<RawFrame>,
    config: Arc<Config>,
) {
    let fps = config.camera.fps;
    let interval_ms = 1000 / fps as u64;
    let mut ticker = interval(Duration::from_millis(interval_ms));
    let mut sequence = 0u64;

    info!("Camera producer started (target {}fps)", fps);

    loop {
        ticker.tick().await;

        // Get camera (might not be available at startup)
        let mut camera_guard = camera.write().await;
        let cam = match camera_guard.as_mut() {
            Some(c) => c,
            None => {
                // Try to initialize camera
                drop(camera_guard);
                match Camera::new_with_config(&config).await {
                    Ok(new_cam) => {
                        info!("Camera initialized on-demand");
                        let mut guard = camera.write().await;
                        *guard = Some(new_cam);
                        continue;
                    }
                    Err(e) => {
                        debug!("Camera still not available: {}", e);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                }
            }
        };

        // Capture frame (FFmpeg I/O - this blocks but unavoidable with current backend)
        let frame_result = match cam.capture_frame() {
            Ok(f) => Ok(f),
            Err(e) => Err(e),
        };
        drop(camera_guard); // Release lock immediately after capture

        let frame = match frame_result {
            Ok(f) => f,
            Err(e) => {
                warn!("Capture error: {}", e);
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };

        sequence += 1;
        let raw_frame = RawFrame {
            image: Arc::new(frame),
            timestamp: std::time::Instant::now(),
            sequence,
        };

        // Non-blocking send (drop frame if channel full)
        if let Err(e) = frame_tx.try_send(raw_frame) {
            debug!("Dropped frame {} (channel full)", sequence);
        }
    }
}
