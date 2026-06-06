use super::types::DetectionResult;
use crate::debug_stream::DebugStreamBroadcaster;
use crate::storage::Storage;
use doorman_shared::{StreamMessage, Config, DetectionInfo};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Recognition pipeline task
/// Receives detection results
/// Matches embeddings against storage
/// Updates system lock state
/// Broadcasts debug information
pub async fn run_recognition_pipeline(
    mut result_rx: tokio::sync::mpsc::Receiver<DetectionResult>,
    storage: Arc<RwLock<Storage>>,
    debug_broadcaster: Arc<DebugStreamBroadcaster>,
    system_locked: Arc<RwLock<bool>>,
    config: Arc<Config>,
    start_time: Instant,
) {
    info!("Recognition pipeline started");

    while let Some(result) = result_rx.recv().await {
        let is_locked = *system_locked.read().await;
        let debug_mode = config.daemon.debug_mode;

        // Check if we should process (locked or debug mode)
        if !is_locked && !debug_mode {
            continue;
        }

        let processing_time_ms = result.processing_time.as_millis() as u32;
        let timestamp_ms = start_time.elapsed().as_millis() as u64;

        // No face detected
        let (face, embedding) = match (result.face, result.embedding) {
            (Some(f), Some(e)) => (f, e),
            _ => {
                // Broadcast "no face" to debug stream
                let message = StreamMessage::Detection {
                    timestamp_ms,
                    detection: DetectionInfo {
                        bbox: None,
                        frame_size: Some((result.frame_width, result.frame_height)),
                        confidence: None,
                        recognized_user: None,
                        similarity: None,
                        frame_jpeg_base64: None,
                    },
                    system_locked: is_locked,
                    processing_time_ms,
                };
                debug_broadcaster.broadcast(message);
                continue;
            }
        };

        // Match against enrolled users
        let storage_guard = storage.read().await;
        let mut best_match: Option<(String, f32)> = None;
        let mut best_similarity = 0.0f32;

        for user_info in storage_guard.list_users() {
            if let Some(stored_embeddings) = storage_guard.get_embeddings(&user_info.username) {
                // Compare with all stored embeddings and take best match
                let max_similarity = stored_embeddings
                    .iter()
                    .map(|stored_emb| cosine_similarity(&embedding, stored_emb))
                    .fold(0.0f32, f32::max);
                    
                if max_similarity > best_similarity {
                    best_similarity = max_similarity;
                    if max_similarity >= config.authentication.similarity_threshold {
                        best_match = Some((user_info.username.clone(), max_similarity));
                    }
                }
            }
        }
        drop(storage_guard);

        // Handle recognition result
        if let Some((username, similarity)) = best_match {
            info!(
                "✓ User recognized: {} (similarity: {:.2}, frame: {})",
                username, similarity, result.sequence
            );

            // Unlock system if locked
            if is_locked {
                let mut locked = system_locked.write().await;
                *locked = false;
                info!("System unlocked for user: {}", username);
            }

            // Broadcast success to debug stream
            // Convert normalized bbox [0,1] to pixel coords
            let bbox_px = (
                (face.bbox.0 * result.frame_width as f32) as u32,
                (face.bbox.1 * result.frame_height as f32) as u32,
                (face.bbox.2 * result.frame_width as f32) as u32,
                (face.bbox.3 * result.frame_height as f32) as u32,
            );
            let message = StreamMessage::Detection {
                timestamp_ms,
                detection: DetectionInfo {
                    bbox: Some(bbox_px),
                    frame_size: Some((result.frame_width, result.frame_height)),
                    confidence: Some(face.confidence),
                    recognized_user: Some(username),
                    similarity: Some(similarity),
                    frame_jpeg_base64: None,
                },
                system_locked: false, // System is now unlocked
                processing_time_ms,
            };
            debug_broadcaster.broadcast(message);
        } else {
            // Unknown face
            debug!(
                "Unknown face detected (best similarity: {:.2}, frame: {})",
                best_similarity, result.sequence
            );

            // Convert normalized bbox [0,1] to pixel coords
            let bbox_px = (
                (face.bbox.0 * result.frame_width as f32) as u32,
                (face.bbox.1 * result.frame_height as f32) as u32,
                (face.bbox.2 * result.frame_width as f32) as u32,
                (face.bbox.3 * result.frame_height as f32) as u32,
            );
            let message = StreamMessage::Detection {
                timestamp_ms,
                detection: DetectionInfo {
                    bbox: Some(bbox_px),
                    frame_size: Some((result.frame_width, result.frame_height)),
                    confidence: Some(face.confidence),
                    recognized_user: None,
                    similarity: Some(best_similarity),
                    frame_jpeg_base64: None,
                },
                system_locked: is_locked,
                processing_time_ms,
            };
            debug_broadcaster.broadcast(message);
        }
    }

    info!("Recognition pipeline stopped");
}

/// Calculate cosine similarity between two embeddings
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}
