use crate::DaemonState;
use crate::camera::Camera;
use anyhow::{Context, Result};
use doorman_shared::{
    Request, Response, ResponseData, DaemonInfo, SOCKET_PATH, 
    AUTH_FRAMES, ENROLL_FRAMES, SIMILARITY_THRESHOLD,
};
use image::GenericImageView;
use std::fs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tracing::{debug, error, info, warn};

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

/// Attempt to ensure camera is available, reinitializing if needed
async fn ensure_camera_available(state: &DaemonState) -> Result<(), String> {
    const MAX_RETRIES: usize = 3;
    const RETRY_DELAY_MS: u64 = 500;
    
    for attempt in 1..=MAX_RETRIES {
        let mut camera_lock = state.camera.write().await;
        
        // Check if camera is already available
        if camera_lock.is_some() {
            return Ok(());
        }
        
        // Try to initialize camera
        info!("Camera not available, attempting to initialize (attempt {}/{})", attempt, MAX_RETRIES);
        match Camera::new_with_config(&state.config).await {
            Ok(camera) => {
                info!("Successfully initialized camera on attempt {}", attempt);
                *camera_lock = Some(camera);
                return Ok(());
            }
            Err(e) => {
                warn!("Failed to initialize camera (attempt {}): {}", attempt, e);
                drop(camera_lock); // Release lock before sleeping
                
                if attempt < MAX_RETRIES {
                    tokio::time::sleep(tokio::time::Duration::from_millis(RETRY_DELAY_MS)).await;
                }
            }
        }
    }
    
    Err(format!("Camera not available after {} attempts. Please connect camera and try again.", MAX_RETRIES))
}

async fn handle_authenticate(state: &DaemonState, username: &str) -> Response {
    info!("Authentication request for user: {}", username);

    // Check if user is enrolled
    let storage = state.storage.read().await;
    let stored_embedding = match storage.get_embedding(username) {
        Some(emb) => emb.clone(),
        None => {
            warn!("User not enrolled: {}", username);
            return Response::Failure {
                reason: "User not enrolled".to_string(),
            };
        }
    };
    drop(storage);

    // Ensure camera is available (with retry logic)
    if let Err(err_msg) = ensure_camera_available(state).await {
        error!("{}", err_msg);
        return Response::Failure {
            reason: err_msg,
        };
    }

    // Get camera and capture frames
    let frames = {
        let mut camera_lock = state.camera.write().await;
        let camera = camera_lock.as_mut().expect("Camera should be available after ensure_camera_available");

        // Capture frames before dropping the lock
        camera.capture_frames(AUTH_FRAMES)
    }; // Lock is dropped here

    if frames.is_empty() {
        error!("Failed to capture any frames");
        return Response::Failure {
            reason: "Failed to capture frames".to_string(),
        };
    }

    info!("Captured {} frames for authentication", frames.len());

    // Process frames
    let mut best_similarity = 0.0f32;
    for (i, frame) in frames.iter().enumerate() {
        match state.ml_pipeline.process_frame(frame).await {
            Ok(Some((_face, embedding))) => {
                let similarity = crate::ml::cosine_similarity(&embedding, &stored_embedding);
                debug!("Frame {}: similarity = {:.4}", i, similarity);
                
                if similarity > best_similarity {
                    best_similarity = similarity;
                }

                if similarity >= SIMILARITY_THRESHOLD {
                    info!("Authentication successful for {} (similarity: {:.4})", username, similarity);
                    return Response::Success {
                        message: Some(format!("Authenticated (confidence: {:.2}%)", similarity * 100.0)),
                        data: None,
                    };
                }
            }
            Ok(None) => {
                debug!("Frame {}: No valid face detected", i);
            }
            Err(e) => {
                warn!("Frame {}: Processing error: {}", i, e);
            }
        }
    }

    warn!("Authentication failed for {} (best similarity: {:.4})", username, best_similarity);
    Response::Failure {
        reason: format!("Face not recognized (confidence: {:.2}%)", best_similarity * 100.0),
    }
}

async fn handle_enroll(
    state: &DaemonState,
    username: &str,
    _reader: &mut BufReader<UnixStream>,
) -> Response {
    info!("Enrollment request for user: {}", username);

    // Ensure camera is available (with retry logic)
    if let Err(err_msg) = ensure_camera_available(state).await {
        error!("{}", err_msg);
        return Response::Failure {
            reason: err_msg,
        };
    }

    // Get camera and capture frames
    info!("Capturing {} frames for enrollment...", ENROLL_FRAMES);
    let frames = {
        let mut camera_lock = state.camera.write().await;
        let camera = camera_lock.as_mut().expect("Camera should be available after ensure_camera_available");

        // Capture frames before dropping the lock
        camera.capture_frames(ENROLL_FRAMES)
    }; // Lock is dropped here

    if frames.is_empty() {
        error!("Failed to capture any frames");
        return Response::Failure {
            reason: "Failed to capture frames".to_string(),
        };
    }

    info!("Captured {} frames, processing...", frames.len());

    // Process frames and collect valid embeddings
    let mut valid_embeddings = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        match state.ml_pipeline.process_frame(frame).await {
            Ok(Some((_face, embedding))) => {
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

    if valid_embeddings.is_empty() {
        error!("No valid face embeddings extracted");
        return Response::Failure {
            reason: "No valid faces detected. Please ensure good lighting and look at camera.".to_string(),
        };
    }

    info!("Extracted {} valid embeddings", valid_embeddings.len());

    // Average the embeddings to create master embedding
    let embedding_dim = valid_embeddings[0].len();
    let mut master_embedding = vec![0.0f32; embedding_dim];

    for embedding in &valid_embeddings {
        for (i, val) in embedding.iter().enumerate() {
            master_embedding[i] += val;
        }
    }

    for val in &mut master_embedding {
        *val /= valid_embeddings.len() as f32;
    }

    // Normalize the embedding
    let norm: f32 = master_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut master_embedding {
            *val /= norm;
        }
    }

    // Store the embedding
    let mut storage = state.storage.write().await;
    match storage.store_embedding(username.to_string(), master_embedding).await {
        Ok(()) => {
            info!("Successfully enrolled user: {}", username);
            Response::Success {
                message: Some(format!(
                    "Enrolled successfully ({}/{} frames used)",
                    valid_embeddings.len(),
                    frames.len()
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
    let camera_lock = state.camera.read().await;

    let info = DaemonInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: state.start_time.elapsed().as_secs(),
        camera_available: camera_lock.is_some(),
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

    // Capture a single frame
    let mut camera = state.camera.write().await;
    let camera = match camera.as_mut() {
        Some(cam) => cam,
        None => {
            return Response::Failure {
                reason: "Camera not available".to_string(),
            }
        }
    };

    let frame = match camera.capture_frame() {
        Ok(f) => f,
        Err(e) => {
            return Response::Failure {
                reason: format!("Failed to capture frame: {}", e),
            }
        }
    };
    drop(camera); // Release camera lock

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
            if let Some(stored_embedding) = storage.get_embedding(&username.username) {
                let similarity = crate::ml::cosine_similarity(&embedding, stored_embedding);
                if similarity > best_similarity {
                    best_similarity = similarity;
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

