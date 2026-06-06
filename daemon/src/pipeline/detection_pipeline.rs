use super::types::{DetectionResult, Face, RawFrame};
use crate::{ml::MLPipeline, debug_stream::DebugStreamBroadcaster};
use doorman_shared::StreamMessage;
use image::GenericImageView;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Detection pipeline task
/// Receives frames at target detection rate
/// Processes frames sequentially in async context
/// Sends detection results to debug preview
pub async fn run_detection_pipeline(
    mut frame_rx: mpsc::Receiver<RawFrame>,
    result_tx: mpsc::Sender<DetectionResult>,
    ml_pipeline: Arc<MLPipeline>,
    debug_broadcaster: Option<Arc<DebugStreamBroadcaster>>,
) {
    info!("Detection pipeline started");

    let mut detection_count = 0u64;
    let mut last_log = Instant::now();
    let mut frame_in_cycle = 0u32;
    const EMBEDDING_INTERVAL: u32 = 3; // Extract embedding every 3rd frame

    // Process frames sequentially - ML pipeline handles blocking internally
    while let Some(raw_frame) = frame_rx.recv().await {
        detection_count += 1;
        frame_in_cycle = (frame_in_cycle + 1) % EMBEDDING_INTERVAL;

        // Log detection stats every 10 seconds
        if last_log.elapsed().as_secs() >= 10 {
            let elapsed = last_log.elapsed().as_secs_f64();
            let fps = detection_count as f64 / elapsed;
            info!("Detection processing: {:.1} fps", fps);
            detection_count = 0;
            last_log = Instant::now();
        }

        let frame = raw_frame.image.clone();
        let camera_dimensions = frame.dimensions();
        let timestamp = raw_frame.timestamp;
        let sequence = raw_frame.sequence;
        let start = Instant::now();

        // Hybrid approach: detect_only most frames, full processing every Nth frame
        // This gives ~30fps detection for preview, ~10fps recognition for matching
        let need_embedding = frame_in_cycle == 0;

        let ml = ml_pipeline.clone();
        let result = tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(async {
                if need_embedding {
                    // Full processing with embedding
                    ml.process_frame(&*frame).await
                } else {
                    // Fast detection only (no embedding)
                    match ml.detect_only(&*frame).await {
                        Ok(Some(face)) => Ok(Some((face, Vec::new()))), // Empty embedding
                        Ok(None) => Ok(None),
                        Err(e) => Err(e),
                    }
                }
            })
        }).await;

        // Unwrap the nested Result from spawn_blocking
        let result = match result {
            Ok(inner) => inner,
            Err(e) => {
                warn!("spawn_blocking join error on frame {}: {}", sequence, e);
                let (camera_width, camera_height) = camera_dimensions;
                let _ = result_tx.try_send(DetectionResult {
                    sequence, timestamp, face: None, embedding: None,
                    processing_time: start.elapsed(), frame_width: camera_width, frame_height: camera_height,
                });
                continue;
            }
        };

        let (face, embedding) = match result {
            Ok(Some(r)) => r,
            Ok(None) => {
                // No face detected or liveness check failed
                if let Some(ref bc) = debug_broadcaster {
                    bc.broadcast(StreamMessage::Detection {
                        timestamp_ms: timestamp.elapsed().as_millis() as u64,
                        detection: doorman_shared::DetectionInfo {
                            bbox: None,
                            frame_size: Some(camera_dimensions),
                            confidence: None,
                            recognized_user: None,
                            similarity: None,
                            frame_jpeg_base64: None,
                        },
                        system_locked: false,
                        processing_time_ms: start.elapsed().as_millis() as u32,
                    });
                }

                let (camera_width, camera_height) = camera_dimensions;
                let _ = result_tx.try_send(DetectionResult {
                    sequence,
                    timestamp,
                    face: None,
                    embedding: None,
                    processing_time: start.elapsed(),
                    frame_width: camera_width,
                    frame_height: camera_height,
                });
                continue;
            }
            Err(ref e) => {
                warn!("ML processing error on frame {}: {}", sequence, e);
                let (camera_width, camera_height) = camera_dimensions;
                let _ = result_tx.try_send(DetectionResult {
                    sequence,
                    timestamp,
                    face: None,
                    embedding: None,
                    processing_time: start.elapsed(),
                    frame_width: camera_width,
                    frame_height: camera_height,
                });
                continue;
            }
        };

        let processing_time = start.elapsed();
        debug!(
            "Frame {} processed in {}ms",
            sequence,
            processing_time.as_millis()
        );

        // Send detection result to debug preview
        if let Some(ref bc) = debug_broadcaster {
            let (camera_width, camera_height) = camera_dimensions;
            // face.bbox contains NORMALIZED coordinates [0,1], convert to pixels
            let (x_norm, y_norm, w_norm, h_norm) = face.bbox;

            // Convert normalized coords to pixel coordinates
            let x_px = x_norm * camera_width as f32;
            let y_px = y_norm * camera_height as f32;
            let w_px = w_norm * camera_width as f32;
            let h_px = h_norm * camera_height as f32;

            let bbox_pixels = (
                x_px as u32,
                y_px as u32,
                w_px as u32,
                h_px as u32,
            );
            debug!("Face bbox normalized: x={:.3}, y={:.3}, w={:.3}, h={:.3}", x_norm, y_norm, w_norm, h_norm);
            debug!("Converted to pixels ({}x{}): x={:.1}, y={:.1}, w={:.1}, h={:.1}", camera_width, camera_height, x_px, y_px, w_px, h_px);
            debug!("  Top-left corner: ({:.1}, {:.1})", x_px, y_px);
            debug!("  Bottom-right corner: ({:.1}, {:.1})", x_px + w_px, y_px + h_px);
            info!("Broadcasting detection: bbox=({}, {}, {}, {}) = top_left + size, confidence={:.3}",
                bbox_pixels.0, bbox_pixels.1, bbox_pixels.2, bbox_pixels.3, face.confidence);
            bc.broadcast(StreamMessage::Detection {
                timestamp_ms: timestamp.elapsed().as_millis() as u64,
                detection: doorman_shared::DetectionInfo {
                    bbox: Some(bbox_pixels),
                    frame_size: Some(camera_dimensions),
                    confidence: Some(face.confidence),
                    recognized_user: None,  // Will be set by recognition pipeline
                    similarity: None,
                    frame_jpeg_base64: None,
                },
                system_locked: false,  // Will be updated by recognition pipeline
                processing_time_ms: processing_time.as_millis() as u32,
            });
        }

        // We have an embedding and face bbox from detection
        let (camera_width, camera_height) = camera_dimensions;
        let _ = result_tx.try_send(DetectionResult {
            sequence,
            timestamp,
            face: Some(Face {
                bbox: face.bbox,
                confidence: face.confidence,
            }),
            embedding: Some(embedding),
            processing_time,
            frame_width: camera_width,
            frame_height: camera_height,
        });
    }

    info!("Detection pipeline stopped");
}
