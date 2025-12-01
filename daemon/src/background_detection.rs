use crate::DaemonState;
use doorman_shared::{DebugStreamMessage, DetectionInfo, SIMILARITY_THRESHOLD};
use tokio::time::{interval, Duration};
use tracing::{debug, info};

/// Background task that continuously processes frames for face authentication
/// Runs at configurable FPS, only when system is locked
/// Broadcasts detection results to debug stream for preview
/// Logs only important events (successful auth)
pub async fn run_background_detection(state: DaemonState) {
    let processing_fps = state.config.daemon.processing_fps;
    let interval_ms = 1000 / processing_fps as u64;

    info!("Starting continuous face recognition ({}Hz)", processing_fps);

    // Process at configured FPS
    let mut ticker = interval(Duration::from_millis(interval_ms));

    loop {
        ticker.tick().await;
        let frame_start = std::time::Instant::now();

        // Check if system is locked (only process when locked, unless debug mode)
        let is_locked = *state.system_locked.read().await;
        let debug_mode = state.config.daemon.debug_mode;

        if !is_locked && !debug_mode {
            // System unlocked and not in debug mode, skip processing
            let message = DebugStreamMessage {
                timestamp_ms: state.start_time.elapsed().as_millis() as u64,
                detection: DetectionInfo {
                    bbox: None,
                    frame_size: None,
                    confidence: None,
                    recognized_user: None,
                    similarity: None,
                frame_jpeg_base64: None,
                },
                system_locked: false,
                processing_time_ms: 0,
            };
            state.debug_broadcaster.broadcast(message);
            continue;
        }

        // Check if camera is available and capture frame
        let frame = {
            let mut camera_guard = state.camera.write().await;
            let camera = match camera_guard.as_mut() {
                Some(cam) => cam,
                None => {
                    // Camera not available, wait and retry
                    drop(camera_guard);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            };

            // Capture frame while holding the lock
            match camera.capture_frame() {
                Ok(f) => f,
                Err(e) => {
                    debug!("Failed to capture frame: {}", e);
                    drop(camera_guard);
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                }
            }
            // Lock automatically released at end of block
        };

        // Broadcast frame to preview clients (if preview mode enabled)
        if let Some(ref frame_broadcaster) = state.frame_broadcaster {
            if let Err(e) = frame_broadcaster.broadcast_frame(&frame) {
                debug!("Failed to broadcast frame: {}", e);
            }
        }

        // Detect face
        let face = match state.ml_pipeline.detect_face(&frame).await {
            Ok(Some(f)) => f,
            Ok(None) => {
                // No face detected - broadcast to debug stream
                let message = DebugStreamMessage {
                    timestamp_ms: state.start_time.elapsed().as_millis() as u64,
                    detection: DetectionInfo {
                        bbox: None,
                    frame_size: None,
                        confidence: None,
                        recognized_user: None,
                        similarity: None,
                frame_jpeg_base64: None,
                    },
                    system_locked: true,
                    processing_time_ms: frame_start.elapsed().as_millis() as u32,
                };
                state.debug_broadcaster.broadcast(message);
                continue;
            }
            Err(e) => {
                tracing::warn!("Face detection failed: {}", e);
                continue;
            }
        };

        // Convert bbox
        let (x, y, w, h) = face.bbox;
        let bbox = (x as u32, y as u32, w as u32, h as u32);

        // Check if any users enrolled
        let storage = state.storage.read().await;
        if storage.count() == 0 {
            // No users enrolled, just broadcast detection
            let message = DebugStreamMessage {
                timestamp_ms: state.start_time.elapsed().as_millis() as u64,
                detection: DetectionInfo {
                    bbox: Some(bbox),
                    frame_size: None,
                    confidence: Some(face.confidence),
                    recognized_user: None,
                    similarity: None,
                frame_jpeg_base64: None,
                },
                system_locked: true,
                processing_time_ms: frame_start.elapsed().as_millis() as u32,
            };
            state.debug_broadcaster.broadcast(message);
            drop(storage);
            continue;
        }

        // Extract embedding and try to recognize
        let embedding = match state.ml_pipeline.extract_embedding(&frame, &face).await {
            Ok(emb) => emb,
            Err(e) => {
                tracing::warn!("Embedding extraction failed: {}", e);
                let message = DebugStreamMessage {
                    timestamp_ms: state.start_time.elapsed().as_millis() as u64,
                    detection: DetectionInfo {
                        bbox: Some(bbox),
                    frame_size: None,
                        confidence: Some(face.confidence),
                        recognized_user: None,
                        similarity: None,
                frame_jpeg_base64: None,
                    },
                    system_locked: true,
                    processing_time_ms: frame_start.elapsed().as_millis() as u32,
                };
                state.debug_broadcaster.broadcast(message);
                drop(storage);
                continue;
            }
        };

        // Try to match against enrolled users
        let mut best_match: Option<(String, f32)> = None;
        let mut best_similarity = 0.0f32;

        let users = storage.list_users();
        for user_info in users {
            if let Some(stored_embedding) = storage.get_embedding(&user_info.username) {
                let similarity = crate::ml::cosine_similarity(&embedding, stored_embedding);
                if similarity > best_similarity {
                    best_similarity = similarity;
                    if similarity >= SIMILARITY_THRESHOLD {
                        best_match = Some((user_info.username.clone(), similarity));
                    }
                }
            }
        }
        drop(storage);

        let processing_time_ms = frame_start.elapsed().as_millis() as u32;

        let (recognized_user, similarity) = match best_match {
            Some((username, score)) => {
                // Successful recognition - log it and unlock system
                info!("✓ User recognized: {} (similarity: {:.2})", username, score);

                // TODO: Unlock system via D-Bus
                // For now, just keep system_locked = true for debugging

                (Some(username), Some(score))
            }
            None => (None, None),
        };

        // Broadcast detection result to debug stream
        let message = DebugStreamMessage {
            timestamp_ms: state.start_time.elapsed().as_millis() as u64,
            detection: DetectionInfo {
                bbox: Some(bbox),
                    frame_size: None,
                confidence: Some(face.confidence),
                recognized_user,
                similarity,
                frame_jpeg_base64: None,
            },
            system_locked: true,
            processing_time_ms,
        };
        state.debug_broadcaster.broadcast(message);
    }
}
