use crate::dataset::{self, AttemptConditions, GtSource, LabelEvent, Outcome};
use crate::DaemonState;
use anyhow::Result;
use chrono::Timelike;
use doorman_shared::{
    DaemonInfo, EnrollmentPhase, Request, Response, ResponseData, StreamMessage,
    ENROLL_DURATION_SECS, SIMILARITY_THRESHOLD,
};
use doormand::ipc_policy::{self, Gated, Gatekeeper};
use image::GenericImageView;
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tracing::{debug, error, info, warn};

/// Select diverse embeddings from a larger set using maximal distance approach
/// This ensures we capture different angles/variations of the face
pub(crate) fn select_diverse_embeddings(embeddings: Vec<Vec<f32>>, count: usize) -> Vec<Vec<f32>> {
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

    // World-connectable: PAM (root) and the CLI/lock watcher (user) differ in uid.
    // Authorization is entirely SO_PEERCRED policy, see ipc_policy / docs/ipc-security.md.
    let listener = ipc_policy::bind_socket(socket_path, ipc_policy::COMMAND_SOCKET_MODE, None)?;
    let gate = Gatekeeper::new(state.config.ipc.clone());

    info!("IPC server listening on {}", socket_path);

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                if let Err(e) = handle_connection(stream, state.clone(), &gate).await {
                    error!("Connection error: {}", e);
                }
            }
            Err(e) => {
                error!("Accept error: {}", e);
            }
        }
    }
}

async fn handle_connection(
    stream: UnixStream,
    state: DaemonState,
    gate: &Gatekeeper,
) -> Result<()> {
    // Fail closed: without peer credentials no request can be authorized.
    let peer = match ipc_policy::peer_from_stream(&stream) {
        Ok(p) => p,
        Err(e) => {
            warn!("Rejecting connection: {e:#}");
            return Ok(());
        }
    };
    let mut reader = BufReader::new(stream);

    let request = match ipc_policy::read_gated_request(&mut reader, &peer, gate).await? {
        Gated::Allowed(request) => request,
        Gated::Rejected(response) => {
            return write_response(reader.into_inner(), &response).await;
        }
    };

    debug!("Received request: {}", ipc_policy::request_name(&request));
    let is_authenticate = matches!(request, Request::Authenticate { .. });

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

    if is_authenticate {
        let outcome = match response {
            Response::Success { .. } => "success",
            _ => "failure",
        };
        info!(
            peer_uid = peer.uid,
            peer_pid = ?peer.pid,
            outcome,
            "Authenticate decision"
        );
    }

    write_response(reader.into_inner(), &response).await
}

async fn write_response(mut stream: UnixStream, response: &Response) -> Result<()> {
    let response_json = serde_json::to_string(response)?;
    stream.write_all(response_json.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    stream.flush().await?;

    Ok(())
}

/// Wait until the single-owner camera producer thread has published at least
/// one frame, then return the most recent frame. The producer owns the camera
/// exclusively; consumers never touch the camera directly.
async fn wait_for_frame(
    state: &DaemonState,
) -> Result<std::sync::Arc<image::DynamicImage>, String> {
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
        Err(_) => Err(
            "Camera not available (no frames). Please connect camera and try again.".to_string(),
        ),
    }
}

async fn wait_for_auth_frame_until(
    rx: &mut tokio::sync::watch::Receiver<Option<std::sync::Arc<image::DynamicImage>>>,
    deadline: std::time::Instant,
) -> Option<std::sync::Arc<image::DynamicImage>> {
    loop {
        let now = std::time::Instant::now();
        if now >= deadline {
            return None;
        }
        let remaining = deadline.saturating_duration_since(now);
        match tokio::time::timeout(
            remaining.min(std::time::Duration::from_millis(250)),
            rx.changed(),
        )
        .await
        {
            Ok(Ok(())) => {
                if let Some(frame) = rx.borrow_and_update().clone() {
                    return Some(frame);
                }
            }
            Ok(Err(_)) => return None,
            Err(_) => continue,
        }
    }
}

/// Collect frames from the camera producer for a given duration.
async fn collect_frames_for_duration(
    state: &DaemonState,
    duration_secs: u64,
) -> Vec<image::DynamicImage> {
    let mut rx = state.latest_frame.clone();
    let mut frames = Vec::new();
    let start = std::time::Instant::now();
    let duration = std::time::Duration::from_secs(duration_secs);

    if let Some(frame) = rx.borrow().clone() {
        frames.push((*frame).clone());
    }

    while start.elapsed() < duration {
        let remaining = duration.saturating_sub(start.elapsed());
        match tokio::time::timeout(
            remaining.min(std::time::Duration::from_millis(2000)),
            rx.changed(),
        )
        .await
        {
            Ok(Ok(())) => {
                if let Some(frame) = rx.borrow().clone() {
                    frames.push((*frame).clone());
                }
            }
            Ok(Err(_)) => break, // producer stopped
            Err(_) => {}         // timeout slice; loop re-checks duration
        }
    }

    frames
}

async fn handle_authenticate(state: &DaemonState, username: &str) -> Response {
    info!("Authentication request for user: {}", username);
    // Emits one timing record (debug log + optional JSONL file) when dropped.
    let mut timing = crate::timing::AttemptTiming::start(&state.config.telemetry);

    // Check if user is enrolled
    let storage = state.storage.read().await;
    let stored_embeddings = match storage.get_embeddings(username) {
        Some(embs) => embs.clone(),
        None => {
            warn!("User not enrolled: {}", username);
            timing.outcome = crate::timing::TimingOutcome::NotEnrolled;
            return Response::Failure {
                reason: "User not enrolled".to_string(),
            };
        }
    };
    drop(storage);

    // Turn the camera ON on-demand (works even when the screen is unlocked and the
    // camera is otherwise gated off) and wait for frames. The lease is held for the
    // rest of this call so the camera stays up through the capture.
    let _camera_lease = match state
        .acquire_camera(std::time::Duration::from_secs(8))
        .await
    {
        Ok(lease) => lease,
        Err(err_msg) => {
            error!("{}", err_msg);
            timing.outcome = crate::timing::TimingOutcome::CameraUnavailable;
            return Response::Failure { reason: err_msg };
        }
    };
    timing.camera_open = Some(timing.since_start());

    let attempt_id = format!(
        "auth-{}-{}",
        username,
        chrono::Utc::now().timestamp_micros()
    );

    // Phase 1: collect quality-passed per-frame embeddings, AGGREGATE them
    // (renormalized mean), and score the aggregate ONCE against the stored
    // template. Raw dataset capture is intentionally independent from the
    // recognition target below: every detector-positive frame observed during
    // this auth attempt is persisted for offline R&D, while `auth_frames` only
    // controls how many quality embeddings are needed for the auth decision.
    let auth_frames = state.config.authentication.auth_frames.max(1);
    let auth_timeout =
        std::time::Duration::from_secs(state.config.authentication.timeout_secs.max(1));
    let deadline = std::time::Instant::now() + auth_timeout;
    let rec = &state.config.recognition;
    let threshold = state.config.authentication.similarity_threshold;
    let mut passed_embeddings: Vec<Vec<f32>> = Vec::with_capacity(auth_frames);
    let mut raw_face_frames: Vec<dataset::CaptureFrameImage> = Vec::new();
    let mut first_quality_face_area_frac = None;
    let mut liveness_checked = false;
    let mut frames_seen = 0usize;
    let mut rx = state.latest_frame.clone();
    let mut maybe_frame = rx.borrow_and_update().clone();

    while passed_embeddings.len() < auth_frames {
        let frame = match maybe_frame.take() {
            Some(frame) => frame,
            None => match wait_for_auth_frame_until(&mut rx, deadline).await {
                Some(frame) => frame,
                None => break,
            },
        };
        let frame = (*frame).clone();
        let i = frames_seen;
        frames_seen += 1;
        timing.frames_seen = frames_seen;
        timing.mark_first_frame();

        let detect_started = std::time::Instant::now();
        let detect_result = state.ml_pipeline.detect_face(&frame).await;
        timing.detect.record(detect_started.elapsed());
        let face = match detect_result {
            Ok(Some(f)) => f,
            Ok(None) => {
                debug!("Frame {}: No valid face detected", i);
                maybe_frame = wait_for_auth_frame_until(&mut rx, deadline).await;
                continue;
            }
            Err(e) => {
                warn!("Frame {}: Detection error: {}", i, e);
                maybe_frame = wait_for_auth_frame_until(&mut rx, deadline).await;
                continue;
            }
        };
        raw_face_frames.push(auth_face_capture_frame(i, &frame, &face));

        if !frame_passes_quality_gate(&frame, &face, rec, i, "auth") {
            maybe_frame = wait_for_auth_frame_until(&mut rx, deadline).await;
            continue;
        }
        timing.mark_first_valid_frame();
        if first_quality_face_area_frac.is_none() {
            first_quality_face_area_frac = face_area_frac(&face);
        }
        // First quality face: run the anti-spoofing depth gate ONCE.
        if !liveness_checked {
            liveness_checked = true;
            let liveness_started = std::time::Instant::now();
            let liveness_result = state.ml_pipeline.liveness_gate(&frame, &face).await;
            timing.liveness = Some(liveness_started.elapsed());
            timing.liveness_passed = liveness_result.as_ref().ok().copied();
            match liveness_result {
                Ok(true) => {}
                Ok(false) => {
                    timing.outcome = crate::timing::TimingOutcome::LivenessRejected;
                    warn!(
                        "Authentication failed for {}: liveness/anti-spoof gate rejected",
                        username
                    );
                    log_auth_failure_attempt(
                        state,
                        &attempt_id,
                        username,
                        Outcome::SuspectedSpoof,
                        Some(false),
                        "liveness_rejected",
                        AttemptConditions {
                            face_area_frac: first_quality_face_area_frac,
                            ..AttemptConditions::default()
                        },
                    );
                    log_auth_capture_pack(
                        state,
                        &attempt_id,
                        Some(username),
                        Outcome::SuspectedSpoof,
                        GtSource::PamFaceAuth,
                        raw_face_frames,
                    );
                    return Response::Failure {
                        reason: "Liveness check failed (not a live face).".to_string(),
                    };
                }
                Err(e) => {
                    timing.outcome = crate::timing::TimingOutcome::LivenessError;
                    warn!(
                        "Authentication failed for {}: liveness error: {}",
                        username, e
                    );
                    log_auth_failure_attempt(
                        state,
                        &attempt_id,
                        username,
                        Outcome::SuspectedSpoof,
                        Some(false),
                        "liveness_error",
                        AttemptConditions {
                            face_area_frac: first_quality_face_area_frac,
                            ..AttemptConditions::default()
                        },
                    );
                    log_auth_capture_pack(
                        state,
                        &attempt_id,
                        Some(username),
                        Outcome::SuspectedSpoof,
                        GtSource::PamFaceAuth,
                        raw_face_frames,
                    );
                    return Response::Failure {
                        reason: "Liveness check error.".to_string(),
                    };
                }
            }
        }
        let embed_started = std::time::Instant::now();
        let embed_result = state.ml_pipeline.extract_embedding(&frame, &face).await;
        timing.embed.record(embed_started.elapsed());
        match embed_result {
            Ok(embedding) => passed_embeddings.push(embedding),
            Err(e) => warn!("Frame {}: Embedding error: {}", i, e),
        }
        if passed_embeddings.len() >= auth_frames {
            break;
        }
        maybe_frame = wait_for_auth_frame_until(&mut rx, deadline).await;
    }

    timing.frames_used = passed_embeddings.len();
    if frames_seen == 0 {
        timing.outcome = crate::timing::TimingOutcome::NoFrames;
        error!("Failed to capture any frames");
        log_auth_failure_attempt(
            state,
            &attempt_id,
            username,
            Outcome::Rejected,
            None,
            "no_frames",
            AttemptConditions::default(),
        );
        return Response::Failure {
            reason: "Failed to capture frames".to_string(),
        };
    }

    info!(
        "Captured {} auth frames; {} had detected faces; {} quality embeddings collected (target {})",
        frames_seen,
        raw_face_frames.len(),
        passed_embeddings.len(),
        auth_frames
    );

    if passed_embeddings.is_empty() {
        timing.outcome = crate::timing::TimingOutcome::NoClearFace;
        warn!(
            "Authentication failed for {}: no quality-passed faces",
            username
        );
        log_auth_failure_attempt(
            state,
            &attempt_id,
            username,
            Outcome::FalseNegative,
            None,
            "no_clear_face",
            AttemptConditions::default(),
        );
        dataset::log_label_event(
            &state.config.daemon.data_dir,
            state.config.dataset.logging_enabled,
            &LabelEvent {
                schema_version: 1,
                attempt_id: attempt_id.clone(),
                timestamp_utc: chrono::Utc::now().to_rfc3339(),
                outcome: Outcome::FalseNegative,
                gt_source: GtSource::PamFaceAuth,
            },
        );
        log_auth_capture_pack(
            state,
            &attempt_id,
            Some(username),
            Outcome::FalseNegative,
            GtSource::PamFaceAuth,
            raw_face_frames,
        );
        return Response::Failure {
            reason: "No clear face detected. Please ensure good lighting and look at the camera."
                .to_string(),
        };
    }

    let decision_started = std::time::Instant::now();
    let aggregated = crate::pipeline::aggregate_embeddings(&passed_embeddings)
        .unwrap_or_else(|| passed_embeddings[0].clone());

    let gallery_similarities: Vec<f32> = stored_embeddings
        .iter()
        .map(|stored_emb| crate::ml::cosine_similarity(&aggregated, stored_emb))
        .collect();
    let best_similarity = gallery_similarities.iter().copied().fold(0.0f32, f32::max);
    let matched = best_similarity >= threshold;
    timing.decision = Some(decision_started.elapsed());
    timing.similarity = Some(best_similarity);
    timing.outcome = if matched {
        crate::timing::TimingOutcome::Matched
    } else {
        crate::timing::TimingOutcome::NotMatched
    };
    dataset::log_attempt_capped(
        &state.config.daemon.data_dir,
        state.config.dataset.logging_enabled,
        state.config.dataset.max_attempt_records_per_day,
        &dataset::new_attempt_record(
            attempt_id.clone(),
            0,
            if matched {
                Some(username.to_string())
            } else {
                None
            },
            best_similarity,
            None,
            Some(true),
            if matched {
                Outcome::Unknown
            } else {
                Outcome::FalseNegative
            },
            if matched {
                GtSource::None
            } else {
                GtSource::PamFaceAuth
            },
            &aggregated,
            None,
            dataset::gallery_spread_from_similarities(&gallery_similarities),
            AttemptConditions::default(),
        ),
    );

    info!(
        "Authentication aggregate for {}: {} quality frames, cosine={:.4} (threshold {:.2})",
        username,
        passed_embeddings.len(),
        best_similarity,
        threshold
    );

    let _ = SIMILARITY_THRESHOLD; // kept for back-compat; threshold comes from config now
    if matched {
        dataset::log_label_event(
            &state.config.daemon.data_dir,
            state.config.dataset.logging_enabled,
            &LabelEvent {
                schema_version: 1,
                attempt_id: attempt_id.clone(),
                timestamp_utc: chrono::Utc::now().to_rfc3339(),
                outcome: Outcome::ConfirmedGenuine,
                gt_source: GtSource::PamFaceAuth,
            },
        );
        log_auth_capture_pack(
            state,
            &attempt_id,
            Some(username),
            Outcome::ConfirmedGenuine,
            GtSource::PamFaceAuth,
            raw_face_frames,
        );
        info!(
            "Authentication successful for {} (similarity: {:.4})",
            username, best_similarity
        );
        Response::Success {
            message: Some(format!(
                "Authenticated (confidence: {:.2}%)",
                best_similarity * 100.0
            )),
            data: None,
        }
    } else {
        warn!(
            "Authentication failed for {} (aggregated similarity: {:.4})",
            username, best_similarity
        );
        dataset::log_label_event(
            &state.config.daemon.data_dir,
            state.config.dataset.logging_enabled,
            &LabelEvent {
                schema_version: 1,
                attempt_id: attempt_id.clone(),
                timestamp_utc: chrono::Utc::now().to_rfc3339(),
                outcome: Outcome::FalseNegative,
                gt_source: GtSource::PamFaceAuth,
            },
        );
        log_auth_capture_pack(
            state,
            &attempt_id,
            Some(username),
            Outcome::FalseNegative,
            GtSource::PamFaceAuth,
            raw_face_frames,
        );
        Response::Failure {
            reason: format!(
                "Face not recognized (confidence: {:.2}%)",
                best_similarity * 100.0
            ),
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
        ctx,
        idx,
        sharpness,
        rec.min_sharpness,
        area_frac,
        rec.min_face_area_frac,
        if pass { "PASS" } else { "REJECT" }
    );
    pass
}

fn face_area_frac(face: &crate::ml::Face) -> Option<f32> {
    let (_, _, w, h) = face.bbox;
    Some((w * h).abs())
}

fn face_bbox_norm(face: &crate::ml::Face) -> [f32; 4] {
    let (x, y, w, h) = face.bbox;
    [x, y, w, h]
}

fn capture_conditions(frame: &image::DynamicImage, face: &crate::ml::Face) -> AttemptConditions {
    AttemptConditions {
        brightness: None,
        face_area_frac: face_area_frac(face),
        sharpness: Some(crate::pipeline::sharpness_score(frame)),
        hour_utc: Some(chrono::Utc::now().hour()),
    }
}

fn auth_face_capture_frame(
    source_index: usize,
    frame: &image::DynamicImage,
    face: &crate::ml::Face,
) -> dataset::CaptureFrameImage {
    dataset::CaptureFrameImage {
        source_index,
        image: frame.clone(),
        bbox_norm: Some(face_bbox_norm(face)),
        conditions: capture_conditions(frame, face),
    }
}

#[cfg(test)]
fn build_auth_face_capture_frames(
    frames: &[image::DynamicImage],
    detected_faces: &[Option<crate::ml::Face>],
) -> Vec<dataset::CaptureFrameImage> {
    frames
        .iter()
        .enumerate()
        .filter_map(|(i, frame)| {
            let face = detected_faces.get(i).and_then(|face| face.as_ref())?;
            Some(auth_face_capture_frame(i, frame, face))
        })
        .collect()
}

fn log_auth_failure_attempt(
    state: &DaemonState,
    attempt_id: &str,
    username: &str,
    outcome: Outcome,
    liveness_pass: Option<bool>,
    failure_reason: &str,
    conditions: AttemptConditions,
) {
    dataset::log_attempt_capped(
        &state.config.daemon.data_dir,
        state.config.dataset.logging_enabled,
        state.config.dataset.max_attempt_records_per_day,
        &dataset::new_failure_attempt_record(
            attempt_id.to_string(),
            0,
            Some(username.to_string()),
            outcome,
            liveness_pass,
            failure_reason.to_string(),
            conditions,
        ),
    );
}

fn log_auth_capture_pack(
    state: &DaemonState,
    attempt_id: &str,
    matched_user: Option<&str>,
    outcome: Outcome,
    gt_source: GtSource,
    frames: Vec<dataset::CaptureFrameImage>,
) {
    if !state.config.dataset.logging_enabled
        || !state.config.dataset.keep_raw_frames
        || frames.is_empty()
    {
        return;
    }
    let data_dir = state.config.daemon.data_dir.clone();
    let max_packs_per_day = state.config.dataset.max_capture_packs_per_day;
    let jpeg_quality = state.config.dataset.raw_frame_jpeg_quality;
    let attempt_id = attempt_id.to_string();
    let matched_user = matched_user.map(str::to_string);
    tokio::task::spawn_blocking(move || {
        dataset::write_capture_pack(
            &data_dir,
            true,
            &attempt_id,
            matched_user,
            outcome,
            gt_source,
            max_packs_per_day,
            jpeg_quality,
            &frames,
        );
    });
}

// Production handlers follow this module; relocating the tests would be a large move.
#[allow(clippy::items_after_test_module)]
#[cfg(test)]
mod tests {
    use super::*;

    fn test_frame(value: u8) -> image::DynamicImage {
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            4,
            3,
            image::Rgb([value, value, value]),
        ))
    }

    fn test_face() -> crate::ml::Face {
        crate::ml::Face {
            bbox: (0.1, 0.2, 0.3, 0.4),
            confidence: 0.95,
            frame_dimensions: (4, 3),
            landmarks: None,
        }
    }

    #[test]
    fn auth_face_capture_pack_keeps_detected_face_frames_before_recognition() {
        let frames = vec![test_frame(1), test_frame(2), test_frame(3), test_frame(4)];
        let faces = vec![None, Some(test_face()), Some(test_face()), None];

        let captured = build_auth_face_capture_frames(&frames, &faces);

        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].source_index, 1);
        assert_eq!(captured[1].source_index, 2);
        assert_eq!(captured[0].bbox_norm, Some([0.1, 0.2, 0.3, 0.4]));
        let area = captured[1].conditions.face_area_frac.unwrap();
        assert!((area - 0.12).abs() < 0.000001);
    }

    #[test]
    fn auth_face_capture_pack_has_no_dataset_specific_frame_limit() {
        let frames = vec![test_frame(1), test_frame(2), test_frame(3), test_frame(4)];
        let faces = vec![
            None,
            Some(test_face()),
            Some(test_face()),
            Some(test_face()),
        ];

        let captured = build_auth_face_capture_frames(&frames, &faces);

        assert_eq!(captured.len(), 3);
        assert_eq!(captured[0].source_index, 1);
        assert_eq!(captured[1].source_index, 2);
        assert_eq!(captured[2].source_index, 3);
    }
}

async fn handle_enroll(
    state: &DaemonState,
    username: &str,
    _reader: &mut BufReader<UnixStream>,
) -> Response {
    info!("Enrollment request for user: {}", username);

    // Turn the camera ON on-demand (works even when the screen is unlocked and the
    // camera is otherwise gated off) and wait for frames. The lease is held for the
    // rest of this call so the camera stays up through the capture.
    let _camera_lease = match state
        .acquire_camera(std::time::Duration::from_secs(8))
        .await
    {
        Ok(lease) => lease,
        Err(err_msg) => {
            error!("{}", err_msg);
            return Response::Failure { reason: err_msg };
        }
    };

    // Get frames from the camera producer (single source of truth).
    info!(
        "Recording video for {} seconds for enrollment...",
        ENROLL_DURATION_SECS
    );

    // Broadcast enrollment start
    state
        .debug_broadcaster
        .broadcast(StreamMessage::Enrollment {
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
    const ENROLL_MAX_PROCESS_FRAMES: usize = 40;
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

    info!(
        "Captured frames, processing {} sampled frames...",
        frames.len()
    );

    // Broadcast processing start
    state
        .debug_broadcaster
        .broadcast(StreamMessage::Enrollment {
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
            state
                .debug_broadcaster
                .broadcast(StreamMessage::Enrollment {
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
                if !frame_passes_quality_gate(frame, &face, &state.config.recognition, i, "enroll")
                {
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
    state
        .debug_broadcaster
        .broadcast(StreamMessage::Enrollment {
            timestamp_ms: state.start_time.elapsed().as_millis() as u64,
            phase: EnrollmentPhase::Complete,
            current: valid_embeddings.len(),
            total: frames.len(),
            username: username.to_string(),
        });

    if valid_embeddings.is_empty() {
        error!("No valid face embeddings extracted");
        return Response::Failure {
            reason: "No valid faces detected. Please ensure good lighting and look at camera."
                .to_string(),
        };
    }

    let num_valid = valid_embeddings.len();
    info!("Extracted {} valid embeddings", num_valid);

    // Capacious enrollment. The stored template must cover the conditions you'll
    // actually unlock under (lighting, angle, distance, glasses). Two parts:
    //   1. UNION the freshly-captured embeddings with any ALREADY stored for this
    //      user, so re-running `enroll` under a NEW condition ADDS coverage instead
    //      of overwriting it (re-enroll a few times across conditions to build a
    //      robust set; `doorman remove <user>` resets).
    //   2. Keep the most DIVERSE TARGET_EMBEDDINGS from the combined pool via
    //      farthest-point selection, so the stored set is spread across conditions
    //      rather than clustered on one.
    const TARGET_EMBEDDINGS: usize = 30;
    let mut pool = valid_embeddings;
    {
        let storage = state.storage.read().await;
        if let Some(existing) = storage.get_embeddings(username) {
            info!(
                "Capacious enroll: merging {} new + {} existing embeddings for {}",
                pool.len(),
                existing.len(),
                username
            );
            pool.extend(existing.iter().cloned());
        }
    }
    let selected_embeddings = if pool.len() <= TARGET_EMBEDDINGS {
        pool
    } else {
        select_diverse_embeddings(pool, TARGET_EMBEDDINGS)
    };

    info!(
        "Selected {} diverse embeddings for storage",
        selected_embeddings.len()
    );

    // Store the embeddings
    let mut storage = state.storage.write().await;
    match storage
        .store_embeddings(username.to_string(), selected_embeddings.clone())
        .await
    {
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

    // Turn the camera on on-demand (it's gated off while the screen is unlocked).
    let _camera_lease = match state
        .acquire_camera(std::time::Duration::from_secs(5))
        .await
    {
        Ok(l) => l,
        Err(e) => return Response::Failure { reason: e },
    };
    // Get the latest frame from the camera producer (single source of truth).
    let frame = match wait_for_frame(state).await {
        Ok(f) => (*f).clone(),
        Err(e) => return Response::Failure { reason: e },
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
