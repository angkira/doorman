use crate::DaemonState;
use anyhow::{Context, Result};
use doorman_shared::{
    Request, Response, ResponseData, DaemonInfo, SOCKET_PATH, StreamMessage, EnrollmentPhase,
    AUTH_FRAMES, ENROLL_DURATION_SECS, SIMILARITY_THRESHOLD,
};
use image::GenericImageView;
use std::fs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tracing::{debug, error, info, warn};

/// Select diverse embeddings from a larger set using maximal distance approach
/// This ensures we capture different angles/variations of the face
fn select_diverse_embeddings(embeddings: Vec<Vec<f32>>, count: usize) -> Vec<Vec<f32>> {
    if embeddings.len() <= count {
        return embeddings;
    }

    let mut selected = Vec::with_capacity(count);
    let mut remaining: Vec<_> = embeddings.into_iter().enumerate().collect();

    // Start with first embedding
    selected.push(remaining[0].1.clone());
    remaining.remove(0);

    // Greedily select embeddings that are furthest from already selected ones
    while selected.len() < count && !remaining.is_empty() {
        let mut max_min_dist = 0.0f32;
        let mut best_idx = 0;

        for (idx, (_orig_idx, candidate)) in remaining.iter().enumerate() {
            // Find minimum distance to any selected embedding
            let min_dist = selected
                .iter()
                .map(|selected_emb| {
                    // Cosine distance = 1 - cosine_similarity
                    let dot: f32 = candidate
                        .iter()
                        .zip(selected_emb.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    1.0 - dot // Already normalized embeddings
                })
                .fold(f32::INFINITY, f32::min);

            if min_dist > max_min_dist {
                max_min_dist = min_dist;
                best_idx = idx;
            }
        }

        selected.push(remaining[best_idx].1.clone());
        remaining.remove(best_idx);
    }

    selected
}

pub async fn run_server(state: DaemonState) -> Result<()> {
    let socket_path = &state.config.daemon.socket_path;

    // Remove old socket if it exists
    let _ = fs::remove_file(socket_path);

    let listener = UnixListener::bind(socket_path)
        .context("Failed to bind UNIX socket")?;

    // Set socket permissions to allow all users (PAM runs in different contexts)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o666);
        fs::set_permissions(socket_path, perms)
            .context("Failed to set socket permissions")?;
    }

    info!("IPC server listening on {}", socket_path);

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                if let Err(e) = handle_connection(stream, state.clone()).await {
                    error!("Connection error: {}", e);
                }
            }
            Err(e) => {
                error!("Accept error: {}", e);
            }
        }
    }
}

async fn handle_connection(stream: UnixStream, state: DaemonState) -> Result<()> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();

    reader.read_line(&mut line).await?;
    
    let request: Request = serde_json::from_str(line.trim())
        .context("Failed to parse request")?;

    debug!("Received request: {:?}", request);

    let response = match request {
        Request::Authenticate { username } => handle_authenticate(&state, &username).await,
        Request::Enroll { username } => handle_enroll(&state, &username, &mut reader).await,
        Request::ListUsers => handle_list_users(&state).await,
        Request::RemoveUser { username } => handle_remove_user(&state, &username).await,
        Request::Status => handle_status(&state).await,
        Request::DetectAndRecognize => handle_detect_and_recognize(&state).await,
        Request::GetLatestDetection => {
            // Not implemented yet - would return cached detection result
            Response::Failure {
                reason: "GetLatestDetection not implemented".to_string(),
            }
        }
        Request::Shutdown => {
            info!("Shutdown requested");
            Response::Success {
                message: Some("Daemon shutting down".to_string()),
                data: None,
            }
        }
    };

    // Send response
    let response_json = serde_json::to_string(&response)?;
    let stream = reader.into_inner();
    let mut stream = stream;
    stream.write_all(response_json.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    stream.flush().await?;

    Ok(())
}

/// Wait until the single-owner camera producer thread has published at least
/// one frame, then return the most recent frame. The producer owns the camera
/// exclusively; consumers never touch the camera directly.
async fn wait_for_frame(state: &DaemonState) -> Result<std::sync::Arc<image::DynamicImage>, String> {
    const TIMEOUT_MS: u64 = 3000;
    let mut rx = state.latest_frame.clone();

    // Fast path: a frame is already available.
    if let Some(frame) = rx.borrow().clone() {
        return Ok(frame);
    }

    // Otherwise wait for the producer to publish a frame.
    let wait = async {
        loop {
            if rx.changed().await.is_err() {
                return Err("Camera producer stopped".to_string());
            }
            if let Some(frame) = rx.borrow().clone() {
                return Ok(frame);
            }
        }
    };

    match tokio::time::timeout(tokio::time::Duration::from_millis(TIMEOUT_MS), wait).await {
        Ok(res) => res,
        Err(_) => Err("Camera not available (no frames). Please connect camera and try again.".to_string()),
    }
}

/// Collect up to `count` distinct successive frames from the camera producer.
/// Frames are sampled from the latest-frame watch channel as the producer
/// publishes them.
async fn collect_frames(state: &DaemonState, count: usize) -> Vec<image::DynamicImage> {
    let mut rx = state.latest_frame.clone();
    let mut frames = Vec::with_capacity(count);

    // Include whatever is currently available.
    if let Some(frame) = rx.borrow().clone() {
        frames.push((*frame).clone());
    }

    while frames.len() < count {
        match tokio::time::timeout(tokio::time::Duration::from_millis(2000), rx.changed()).await {
            Ok(Ok(())) => {
                if let Some(frame) = rx.borrow().clone() {
                    frames.push((*frame).clone());
                }
            }
            _ => break, // producer stopped or timed out
        }
    }

    frames
}

/// Collect frames from the camera producer for a given duration.
async fn collect_frames_for_duration(state: &DaemonState, duration_secs: u64) -> Vec<image::DynamicImage> {
    let mut rx = state.latest_frame.clone();
    let mut frames = Vec::new();
    let start = std::time::Instant::now();
    let duration = std::time::Duration::from_secs(duration_secs);

    if let Some(frame) = rx.borrow().clone() {
        frames.push((*frame).clone());
    }

    while start.elapsed() < duration {
        let remaining = duration.saturating_sub(start.elapsed());
        match tokio::time::timeout(remaining.min(std::time::Duration::from_millis(2000)), rx.changed()).await {
            Ok(Ok(())) => {
                if let Some(frame) = rx.borrow().clone() {
                    frames.push((*frame).clone());
                }
            }
            Ok(Err(_)) => break, // producer stopped
            Err(_) => {} // timeout slice; loop re-checks duration
        }
    }

    frames
}

async fn handle_authenticate(state: &DaemonState, username: &str) -> Response {
    info!("Authentication request for user: {}", username);

    // Check if user is enrolled
    let storage = state.storage.read().await;
    let stored_embeddings = match storage.get_embeddings(username) {
        Some(embs) => embs.clone(),
        None => {
            warn!("User not enrolled: {}", username);
            return Response::Failure {
                reason: "User not enrolled".to_string(),
            };
        }
    };
    drop(storage);

    // Ensure the camera producer is publishing frames.
    if let Err(err_msg) = wait_for_frame(state).await {
        error!("{}", err_msg);
        return Response::Failure {
            reason: err_msg,
        };
    }

    // Collect frames from the camera producer (single source of truth).
    let frames = collect_frames(state, AUTH_FRAMES).await;

    if frames.is_empty() {
        error!("Failed to capture any frames");
        return Response::Failure {
            reason: "Failed to capture frames".to_string(),
        };
    }

    info!("Captured {} frames for authentication", frames.len());

    // Phase 1: collect quality-passed per-frame embeddings, AGGREGATE them
    // (renormalized mean), and score the aggregate ONCE against the stored
    // template. Averaging suppresses per-frame noise and stabilizes the genuine
    // cosine versus the old early-exit-on-first-frame-pass behavior.
    let rec = &state.config.recognition;
    let threshold = state.config.authentication.similarity_threshold;
    let mut passed_embeddings: Vec<Vec<f32>> = Vec::with_capacity(frames.len());
    for (i, frame) in frames.iter().enumerate() {
        match state.ml_pipeline.process_frame(frame).await {
            Ok(Some((face, embedding))) => {
                if !frame_passes_quality_gate(frame, &face, rec, i, "auth") {
                    continue;
                }
                passed_embeddings.push(embedding);
            }
            Ok(None) => {
                debug!("Frame {}: No valid face detected", i);
            }
            Err(e) => {
                warn!("Frame {}: Processing error: {}", i, e);
            }
        }
    }

    if passed_embeddings.is_empty() {
        warn!("Authentication failed for {}: no quality-passed faces", username);
        return Response::Failure {
            reason: "No clear face detected. Please ensure good lighting and look at the camera.".to_string(),
        };
    }

    let aggregated = crate::pipeline::aggregate_embeddings(&passed_embeddings)
        .unwrap_or_else(|| passed_embeddings[0].clone());

    let best_similarity = stored_embeddings
        .iter()
        .map(|stored_emb| crate::ml::cosine_similarity(&aggregated, stored_emb))
        .fold(0.0f32, f32::max);

    info!(
        "Authentication aggregate for {}: {} quality frames, cosine={:.4} (threshold {:.2})",
        username, passed_embeddings.len(), best_similarity, threshold
    );

    let _ = SIMILARITY_THRESHOLD; // kept for back-compat; threshold comes from config now
    if best_similarity >= threshold {
        info!("Authentication successful for {} (similarity: {:.4})", username, best_similarity);
        Response::Success {
            message: Some(format!("Authenticated (confidence: {:.2}%)", best_similarity * 100.0)),
            data: None,
        }
    } else {
        warn!("Authentication failed for {} (aggregated similarity: {:.4})", username, best_similarity);
        Response::Failure {
            reason: format!("Face not recognized (confidence: {:.2}%)", best_similarity * 100.0),
        }
    }
}

/// Phase 1 frame-quality gate shared by auth and enrollment: rejects tiny faces
/// (bbox area too small) and blurry frames (low variance-of-Laplacian sharpness)
/// before their embedding is used. Returns `true` if the frame passes.
fn frame_passes_quality_gate(
    frame: &image::DynamicImage,
    face: &crate::ml::Face,
    rec: &doorman_shared::RecognitionConfig,
    idx: usize,
    ctx: &str,
) -> bool {
    let (_, _, w_norm, h_norm) = face.bbox;
    let area_frac = (w_norm * h_norm).abs();
    let area_ok = rec.min_face_area_frac <= 0.0 || area_frac >= rec.min_face_area_frac;

    let sharpness = if rec.min_sharpness <= 0.0 {
        f32::INFINITY
    } else {
        crate::pipeline::sharpness_score(frame)
    };
    let sharp_ok = rec.min_sharpness <= 0.0 || sharpness >= rec.min_sharpness;

    let pass = area_ok && sharp_ok;
    debug!(
        "{} quality gate frame {}: sharpness={:.2} (min {:.2}) area_frac={:.4} (min {:.4}) -> {}",
        ctx, idx, sharpness, rec.min_sharpness, area_frac, rec.min_face_area_frac,
        if pass { "PASS" } else { "REJECT" }
    );
    pass
}

async fn handle_enroll(
    state: &DaemonState,
    username: &str,
    _reader: &mut BufReader<UnixStream>,
) -> Response {
    info!("Enrollment request for user: {}", username);

    // Ensure the camera producer is publishing frames.
    if let Err(err_msg) = wait_for_frame(state).await {
        error!("{}", err_msg);
        return Response::Failure {
            reason: err_msg,
        };
    }

    // Get frames from the camera producer (single source of truth).
    info!("Recording video for {} seconds for enrollment...", ENROLL_DURATION_SECS);
    
    // Broadcast enrollment start
    state.debug_broadcaster.broadcast(StreamMessage::Enrollment {
        timestamp_ms: state.start_time.elapsed().as_millis() as u64,
        phase: EnrollmentPhase::Recording,
        current: 0,
        total: (ENROLL_DURATION_SECS * 30) as usize, // Estimate ~30fps
        username: username.to_string(),
    });
    
    let frames = collect_frames_for_duration(state, ENROLL_DURATION_SECS).await;

    if frames.is_empty() {
        error!("Failed to capture any frames");
        return Response::Failure {
            reason: "Failed to capture frames".to_string(),
        };
    }

    // Subsample the captured frames before running (serial, CPU-bound) inference.
    //
    // Recording ~10s at ~30fps yields ~300 frames. Embedding every one of them
    // (detect → align → embed) ran serially for ~57s and, because each
    // `process_frame` borrows a recognizer session from the shared pool, it also
    // starved the background detection pipeline. Consecutive frames are nearly
    // identical, so we evenly sample at most ENROLL_MAX_PROCESS_FRAMES of them —
    // this is enough variation for the diverse-embedding selection below while
    // cutting enrollment processing time by ~10x and freeing recognizer sessions
    // for live detection sooner.
    const ENROLL_MAX_PROCESS_FRAMES: usize = 30;
    let frames: Vec<image::DynamicImage> = if frames.len() > ENROLL_MAX_PROCESS_FRAMES {
        let total = frames.len();
        // Evenly spaced indices across the whole recording.
        let step = total as f64 / ENROLL_MAX_PROCESS_FRAMES as f64;
        (0..ENROLL_MAX_PROCESS_FRAMES)
            .map(|i| frames[((i as f64 * step) as usize).min(total - 1)].clone())
            .collect()
    } else {
        frames
    };

    info!("Captured frames, processing {} sampled frames...", frames.len());

    // Broadcast processing start
    state.debug_broadcaster.broadcast(StreamMessage::Enrollment {
        timestamp_ms: state.start_time.elapsed().as_millis() as u64,
        phase: EnrollmentPhase::Processing,
        current: 0,
        total: frames.len(),
        username: username.to_string(),
    });

    // Process frames and collect valid embeddings
    let mut valid_embeddings = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        // Broadcast progress every 5 frames
        if i % 5 == 0 {
            state.debug_broadcaster.broadcast(StreamMessage::Enrollment {
                timestamp_ms: state.start_time.elapsed().as_millis() as u64,
                phase: EnrollmentPhase::Processing,
                current: i,
                total: frames.len(),
                username: username.to_string(),
            });
        }
        
        match state.ml_pipeline.process_frame(frame).await {
            Ok(Some((face, embedding))) => {
                // Phase 1: only enroll quality-passed frames so the stored
                // template is built from sharp, adequately-sized faces.
                if !frame_passes_quality_gate(frame, &face, &state.config.recognition, i, "enroll") {
                    continue;
                }
                debug!("Frame {}: Valid embedding extracted", i);
                valid_embeddings.push(embedding);
            }
            Ok(None) => {
                debug!("Frame {}: No valid face detected", i);
            }
            Err(e) => {
                warn!("Frame {}: Processing error: {}", i, e);
            }
        }
    }
    
    // Broadcast completion
    state.debug_broadcaster.broadcast(StreamMessage::Enrollment {
        timestamp_ms: state.start_time.elapsed().as_millis() as u64,
        phase: EnrollmentPhase::Complete,
        current: valid_embeddings.len(),
        total: frames.len(),
        username: username.to_string(),
    });

    if valid_embeddings.is_empty() {
        error!("No valid face embeddings extracted");
        return Response::Failure {
            reason: "No valid faces detected. Please ensure good lighting and look at camera.".to_string(),
        };
    }

    let num_valid = valid_embeddings.len();
    info!("Extracted {} valid embeddings", num_valid);

    // Select diverse embeddings that cover different angles/variations
    // Use k-means-like approach: pick embeddings that are furthest apart
    const TARGET_EMBEDDINGS: usize = 10;
    let selected_embeddings = if valid_embeddings.len() <= TARGET_EMBEDDINGS {
        valid_embeddings
    } else {
        select_diverse_embeddings(valid_embeddings, TARGET_EMBEDDINGS)
    };

    info!("Selected {} diverse embeddings for storage", selected_embeddings.len());

    // Store the embeddings
    let mut storage = state.storage.write().await;
    match storage.store_embeddings(username.to_string(), selected_embeddings.clone()).await {
        Ok(()) => {
            info!("Successfully enrolled user: {}", username);
            Response::Success {
                message: Some(format!(
                    "Enrollment successful! Processed {}/{} frames, selected {} high-quality embeddings.",
                    num_valid,
                    frames.len(),
                    selected_embeddings.len()
                )),
                data: None,
            }
        }
        Err(e) => {
            error!("Failed to store embedding: {}", e);
            Response::Failure {
                reason: format!("Failed to store enrollment: {}", e),
            }
        }
    }
}

async fn handle_list_users(state: &DaemonState) -> Response {
    let storage = state.storage.read().await;
    let users = storage.list_users();
    
    Response::Success {
        message: None,
        data: Some(ResponseData::UserList { users }),
    }
}

async fn handle_remove_user(state: &DaemonState, username: &str) -> Response {
    let mut storage = state.storage.write().await;
    match storage.remove_embedding(username).await {
        Ok(true) => Response::Success {
            message: Some(format!("Removed user: {}", username)),
            data: None,
        },
        Ok(false) => Response::Failure {
            reason: format!("User not found: {}", username),
        },
        Err(e) => Response::Failure {
            reason: format!("Failed to remove user: {}", e),
        },
    }
}

async fn handle_status(state: &DaemonState) -> Response {
    let storage = state.storage.read().await;
    let camera_available = state.latest_frame.borrow().is_some();

    let info = DaemonInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: state.start_time.elapsed().as_secs(),
        camera_available,
        models_loaded: state.ml_pipeline.models_loaded(),
        enrolled_users: storage.count(),
    };

    Response::Success {
        message: None,
        data: Some(ResponseData::DaemonStatus { info }),
    }
}

async fn handle_detect_and_recognize(state: &DaemonState) -> Response {
    use doorman_shared::DetectionInfo;

    // Get the latest frame from the camera producer (single source of truth).
    let frame = match wait_for_frame(state).await {
        Ok(f) => (*f).clone(),
        Err(e) => {
            return Response::Failure {
                reason: e,
            }
        }
    };

    // Detect face
    let face = match state.ml_pipeline.detect_face(&frame).await {
        Ok(Some(f)) => f,
        Ok(None) => {
            // No face detected
            let (width, height) = frame.dimensions();
            let info = DetectionInfo {
                bbox: None,
                frame_size: Some((width, height)),
                confidence: None,
                recognized_user: None,
                similarity: None,
                frame_jpeg_base64: None,
            };
            return Response::Success {
                message: None,
                data: Some(ResponseData::DetectionResult { result: info }),
            };
        }
        Err(e) => {
            return Response::Failure {
                reason: format!("Face detection failed: {}", e),
            };
        }
    };

    // Extract embedding and try to recognize
    let embedding = match state.ml_pipeline.extract_embedding(&frame, &face).await {
        Ok(emb) => emb,
        Err(e) => {
            // Return detection without recognition
            let (width, height) = frame.dimensions();
            let bbox_px = (
                (face.bbox.0 * width as f32) as u32,
                (face.bbox.1 * height as f32) as u32,
                (face.bbox.2 * width as f32) as u32,
                (face.bbox.3 * height as f32) as u32,
            );
            let info = DetectionInfo {
                bbox: Some(bbox_px),
                frame_size: Some((width, height)),
                confidence: Some(face.confidence),
                recognized_user: None,
                similarity: None,
                frame_jpeg_base64: None,
            };
            debug!("Failed to extract embedding: {}", e);
            return Response::Success {
                message: None,
                data: Some(ResponseData::DetectionResult { result: info }),
            };
        }
    };

    // Try to recognize against enrolled users
    let storage = state.storage.read().await;
    let (recognized_user, similarity) = if storage.count() > 0 {
        let mut best_match = None;
        let mut best_similarity = 0.0f32;

        for username in storage.list_users() {
            if let Some(stored_embeddings) = storage.get_embeddings(&username.username) {
                // Compare with all stored embeddings and take best match
                let max_similarity = stored_embeddings
                    .iter()
                    .map(|stored_emb| crate::ml::cosine_similarity(&embedding, stored_emb))
                    .fold(0.0f32, f32::max);
                    
                if max_similarity > best_similarity {
                    best_similarity = max_similarity;
                    best_match = Some(username.username.clone());
                }
            }
        }

        if best_similarity >= SIMILARITY_THRESHOLD {
            (best_match, Some(best_similarity))
        } else {
            (None, Some(best_similarity))
        }
    } else {
        (None, None)
    };
    drop(storage);

    let (width, height) = frame.dimensions();
    let bbox_px = (
        (face.bbox.0 * width as f32) as u32,
        (face.bbox.1 * height as f32) as u32,
        (face.bbox.2 * width as f32) as u32,
        (face.bbox.3 * height as f32) as u32,
    );
    let info = DetectionInfo {
        bbox: Some(bbox_px),
        frame_size: Some((width, height)),
        confidence: Some(face.confidence),
        recognized_user,
        similarity,
        frame_jpeg_base64: None,
    };

    Response::Success {
        message: None,
        data: Some(ResponseData::DetectionResult { result: info }),
    }
}

