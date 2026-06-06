use super::types::RawFrame;
use crate::camera::Camera;
use anyhow::Result;
use doorman_shared::Config;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};

/// Camera producer task
/// Runs camera capture on a dedicated thread to avoid blocking tokio runtime
/// Never blocks on downstream consumers
pub async fn run_camera_producer(
    camera: Arc<RwLock<Option<Camera>>>,
    frame_tx: mpsc::Sender<RawFrame>,
    config: Arc<Config>,
) {
    let fps = config.camera.fps;
    info!("Camera producer started (target {}fps) - using dedicated thread", fps);

    // Run camera capture on a dedicated OS thread (not tokio blocking pool)
    // This ensures camera I/O never blocks async runtime
    let config_clone = config.clone();
    let camera_clone = camera.clone();

    std::thread::spawn(move || {
        let interval_ms = 1000 / fps as u64;
        let mut sequence = 0u64;
        let mut last_capture = std::time::Instant::now();

        loop {
            // Rate limit captures
            let elapsed = last_capture.elapsed().as_millis() as u64;
            if elapsed < interval_ms {
                std::thread::sleep(Duration::from_millis(interval_ms - elapsed));
            }
            last_capture = std::time::Instant::now();

            // Try to get camera - blocking is OK here (dedicated thread)
            let mut camera_guard = match camera_clone.try_write() {
                Ok(guard) => guard,
                Err(_) => {
                    // Lock contention - skip this frame
                    debug!("Camera lock contention, skipping frame");
                    continue;
                }
            };

            let cam = match camera_guard.as_mut() {
                Some(c) => c,
                None => {
                    drop(camera_guard);
                    debug!("Camera not initialized");
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
            };

            // Capture frame - blocking is OK here (dedicated thread)
            let frame = match cam.capture_frame() {
                Ok(f) => f,
                Err(e) => {
                    warn!("Capture error: {}", e);
                    drop(camera_guard);
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
            };
            drop(camera_guard);

            sequence += 1;
            let raw_frame = RawFrame {
                image: Arc::new(frame),
                timestamp: std::time::Instant::now(),
                sequence,
            };

            // Non-blocking send (drop frame if channel full)
            if let Err(_) = frame_tx.try_send(raw_frame) {
                debug!("Dropped frame {} (channel full)", sequence);
            }
        }
    });

    // Keep the async task alive
    loop {
        tokio::time::sleep(Duration::from_secs(3600)).await;
    }
}
