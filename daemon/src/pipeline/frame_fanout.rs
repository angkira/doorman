use super::types::RawFrame;
use crate::frame_stream::FrameStreamBroadcaster;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, info};

/// Frame fanout task
/// Receives frames from camera and distributes to:
/// 1. Preview clients (raw frames immediately for smooth preview)
/// 2. Detection pipeline (at target FPS)
pub async fn run_frame_fanout(
    mut frame_rx: mpsc::Receiver<RawFrame>,
    frame_broadcaster: Option<Arc<FrameStreamBroadcaster>>,
    detection_tx: mpsc::Sender<RawFrame>,
    target_detection_fps: u32,
) {
    let detection_interval_ms = 1000 / target_detection_fps as u64;
    let mut last_detection = Instant::now();
    
    // Rate limit preview to 15fps to avoid overwhelming blocking thread pool
    let preview_interval_ms = 66; // ~15fps
    let mut last_preview = Instant::now();
    
    let mut frame_count = 0u64;
    let mut last_log = Instant::now();

    info!("Frame fanout started (detection @ {}fps, preview @ ~15fps)", target_detection_fps);

    while let Some(raw_frame) = frame_rx.recv().await {
        frame_count += 1;

        // Log stats every 5 seconds
        if last_log.elapsed().as_secs() >= 5 {
            let elapsed = last_log.elapsed().as_secs_f64();
            let fps = frame_count as f64 / elapsed;
            info!("Camera capture: {:.1} fps", fps);
            frame_count = 0;
            last_log = Instant::now();
        }

        // Broadcast to preview clients (if enabled) - rate limited to 15fps
        if let Some(ref broadcaster) = frame_broadcaster {
            if last_preview.elapsed().as_millis() >= preview_interval_ms as u128 {
                // Clone Arc (cheap - just pointer copy)
                let broadcaster = broadcaster.clone();
                let image = raw_frame.image.clone();  // Arc clone - just increments refcount
                // Spawn blocking task for CPU-intensive JPEG encoding
                tokio::task::spawn_blocking(move || {
                    if let Err(e) = broadcaster.broadcast_frame(&*image) {
                        debug!("Failed to broadcast preview frame: {}", e);
                    }
                });
                last_preview = Instant::now();
            }
        }

        // Send to detection at target FPS
        if last_detection.elapsed().as_millis() >= detection_interval_ms as u128 {
            if let Err(_) = detection_tx.try_send(raw_frame) {
                debug!("Detection channel full, skipping frame");
            }
            last_detection = Instant::now();
        }
    }

    info!("Frame fanout stopped");
}
