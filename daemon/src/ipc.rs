use crate::dataset::{self, AttemptConditions, GtSource, LabelEvent, Outcome};
use crate::pipeline::{LivenessState, SequentialDecider, Step};
use crate::DaemonState;
use anyhow::Result;
use async_trait::async_trait;
use chrono::Timelike;
use doorman_shared::config::DecisionMode;
use doorman_shared::{
    DaemonInfo, EnrollmentPhase, Request, Response, ResponseData, StreamMessage,
    ENROLL_DURATION_SECS, SIMILARITY_THRESHOLD,
};
use doormand::decision::{AuthDecision, AuthOutcome};
use doormand::ipc_policy::{self, Gatekeeper, PeerCred};
use doormand::liveness::{LivenessAggregator, LivenessOutcome, LivenessProgress};
use doormand::sudo_policy::{self, Profile, StrictProfile};
use doormand::unlock_wait::{self, AttemptOutcome, DecisionSource};
use image::GenericImageView;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Mean-mode swap tests through the real `run_authenticate` (test-only seam).
#[cfg(test)]
mod real_path_tests;

/// Longest an authentication waits for the camera to deliver its first frame
/// (the strict sudo profile caps this further by its own deadline).
const AUTH_CAMERA_ACQUIRE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(8);

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
    let gate = Arc::new(Gatekeeper::new(state.config.ipc.clone()));

    info!("IPC server listening on {}", socket_path);

    // Connections are served concurrently (one task each); the camera producer
    // is the single frame source and `DaemonState` is cheap to clone.
    let handler = Arc::new(move |peer: PeerCred, request: Request| {
        let state = state.clone();
        async move { dispatch(&state, peer, request).await }
    });
    ipc_policy::serve(listener, gate, handler).await;
    Ok(())
}

async fn dispatch(state: &DaemonState, peer: PeerCred, request: Request) -> Response {
    debug!("Received request: {}", ipc_policy::request_name(&request));
    let is_authenticate = matches!(request, Request::Authenticate { .. });

    let response = match request {
        Request::Authenticate {
            username, service, ..
        } => handle_authenticate(state, &peer, &username, service.as_deref()).await,
        Request::Enroll { username, add } => handle_enroll(state, &username, add).await,
        Request::ListUsers => handle_list_users(state).await,
        Request::RemoveUser { username } => handle_remove_user(state, &username).await,
        Request::Status => handle_status(state).await,
        Request::DetectAndRecognize => handle_detect_and_recognize(state).await,
        Request::GetLatestDetection => {
            // Not implemented yet - would return cached detection result
            Response::Failure {
                reason: "GetLatestDetection not implemented".to_string(),
            }
        }
        Request::WaitForMatch {
            username,
            timeout_ms,
        } => handle_wait_for_match(state, &username, timeout_ms).await,
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

    response
}

/// `WaitForMatch` decision source: every attempt is a complete `Authenticate`
/// (quality gate, liveness, aggregated threshold), so a `Matched` is never
/// weaker than an `Authenticate` success.
struct AuthenticateSource<'a> {
    state: &'a DaemonState,
    timing: &'a unlock_wait::WaitTiming,
}

#[async_trait]
impl DecisionSource for AuthenticateSource<'_> {
    async fn attempt(&self, username: &str) -> AttemptOutcome {
        if self
            .state
            .storage
            .read()
            .await
            .get_embeddings(username)
            .is_none()
        {
            return AttemptOutcome::NotEnrolled;
        }
        run_authenticate(self.state, username, None, Some(self.timing))
            .await
            .attempt_outcome()
    }
}

async fn handle_wait_for_match(state: &DaemonState, username: &str, timeout_ms: u64) -> Response {
    let timeout =
        unlock_wait::effective_timeout(timeout_ms, state.config.ipc.wait_for_match_max_ms);
    info!(
        timeout_ms = timeout.as_millis() as u64,
        "WaitForMatch request"
    );
    let ipc = &state.config.ipc;
    let prearm_hold = ipc
        .prearm_camera
        .then(|| std::time::Duration::from_secs(ipc.prearm_max_hold_secs));
    let timing = unlock_wait::WaitTiming::new();
    let response = unlock_wait::wait_for_match(
        &AuthenticateSource {
            state,
            timing: &timing,
        },
        state.armed.clone(),
        username,
        timeout,
        prearm_hold,
        &timing,
    )
    .await;
    // Request-relative stages for the host re-measure (no identity, no frames).
    // The per-attempt record (`[telemetry] timing_log`) is written when the
    // attempt ends, which can be before the lock is confirmed, so the lock
    // stage is reported here.
    info!(
        prearm = prearm_hold.is_some(),
        hint_wait_ms = timing.hint_wait().map(|d| d.as_millis() as u64),
        total_ms = timing.request_start.elapsed().as_millis() as u64,
        "WaitForMatch timing"
    );
    match &response {
        Response::Matched { similarity, .. } => {
            info!(similarity, "WaitForMatch decision: matched")
        }
        other => info!("WaitForMatch decision: {other:?}"),
    }
    response
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
        return Ok(frame.image);
    }

    // Otherwise wait for the producer to publish a frame.
    let wait = async {
        loop {
            if rx.changed().await.is_err() {
                return Err("Camera producer stopped".to_string());
            }
            if let Some(frame) = rx.borrow().clone() {
                return Ok(frame.image);
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
    rx: &mut tokio::sync::watch::Receiver<Option<crate::pipeline::RawFrame>>,
    deadline: std::time::Instant,
) -> Option<crate::pipeline::RawFrame> {
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
        frames.push((*frame.image).clone());
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
                    frames.push((*frame.image).clone());
                }
            }
            Ok(Err(_)) => break, // producer stopped
            Err(_) => {}         // timeout slice; loop re-checks duration
        }
    }

    frames
}

/// `Authenticate`: select the policy profile from the PAM service hint. Every
/// service outside `[pam.sudo] service_names` runs the unchanged default path.
async fn handle_authenticate(
    state: &DaemonState,
    peer: &PeerCred,
    username: &str,
    service: Option<&str>,
) -> Response {
    let request_start = std::time::Instant::now();
    let strict = match sudo_policy::select_profile(service, &state.config) {
        Profile::Default => {
            return run_authenticate(state, username, None, None)
                .await
                .into_response()
        }
        Profile::Denied(reason) => {
            sudo_policy::audit(
                service.unwrap_or_default(),
                peer.uid,
                username,
                false,
                sudo_policy::code::DISABLED,
                None,
            );
            return Response::Denied {
                reason: reason.to_string(),
            };
        }
        Profile::Strict(profile) => profile,
    };
    let service = service.unwrap_or_default();
    let refuse = |code: &str, reason: &str| {
        sudo_policy::audit(service, peer.uid, username, false, code, None);
        Response::Denied {
            reason: reason.to_string(),
        }
    };
    if let Err((code, reason)) = sudo_policy::check_presence(state.presence()) {
        return refuse(code, reason);
    }
    if state.sudo_guard.check(username, request_start).is_err() {
        return refuse(
            sudo_policy::code::SUSPENDED,
            sudo_policy::DENIED_SUDO_SUSPENDED,
        );
    }

    let decision = run_authenticate(state, username, Some((&strict, request_start)), None).await;
    let matched = decision.outcome == AuthOutcome::Matched;
    if matched {
        state.sudo_guard.record_success(username);
    } else if sudo_policy::counts_as_failure(&decision.outcome) {
        state
            .sudo_guard
            .record_failure(username, std::time::Instant::now());
    }
    sudo_policy::audit(
        service,
        peer.uid,
        username,
        matched,
        sudo_policy::outcome_code(&decision.outcome),
        decision.similarity,
    );
    decision.into_response()
}

/// Decision core of `Authenticate` and `WaitForMatch` behind the face-auth
/// failure budget (`[ipc] face_*`): while the user is suspended nothing runs
/// (no camera, no models) and the answer is [`AuthOutcome::Suspended`]; a
/// completed decision is accounted afterwards (match resets, judged failure
/// counts). Every entry point goes through here, including each retry of a
/// `WaitForMatch`.
async fn run_authenticate(
    state: &DaemonState,
    username: &str,
    strict: Option<(&StrictProfile, std::time::Instant)>,
    wait: Option<&unlock_wait::WaitTiming>,
) -> AuthDecision {
    if let Err(remaining) = state.face_budget.check(username, std::time::Instant::now()) {
        let retry_after_s = doormand::face_budget::retry_after_secs(remaining);
        info!(
            target: doormand::face_budget::AUDIT_TARGET,
            retry_after_s,
            reason = doormand::face_budget::FACE_SUSPENDED_CODE,
            "face auth refused: suspended after repeated failed attempts"
        );
        return AuthDecision::bare(AuthOutcome::Suspended { retry_after_s });
    }
    let decision = decide(state, username, strict, wait).await;
    if let Some(cooldown) =
        state
            .face_budget
            .record(username, &decision.outcome, std::time::Instant::now())
    {
        warn!(
            target: doormand::face_budget::AUDIT_TARGET,
            cooldown_s = cooldown.as_secs(),
            reason = doormand::face_budget::FACE_SUSPENDED_CODE,
            "face auth suspended: failure budget used up (password unaffected)"
        );
    }
    decision
}

/// The decision itself; callers build the wire
/// response (or the wait outcome) from the returned [`AuthDecision`].
///
/// `strict` is the sudo profile together with the request arrival time: it
/// forces the `n_of_m` decision with the profile's N/M and threshold, its
/// multi-frame liveness and its timeout (counted from the arrival), and uses
/// only frames captured after the arrival.
///
/// `wait` is set by `WaitForMatch`: only frames captured at or after its
/// request time are used (the same freshness rule as the strict profile), and
/// the timing record gets the lock-relative stages.
async fn decide(
    state: &DaemonState,
    username: &str,
    strict: Option<(&StrictProfile, std::time::Instant)>,
    wait: Option<&unlock_wait::WaitTiming>,
) -> AuthDecision {
    info!("Authentication request for user: {}", username);
    // Emits one timing record (debug log + optional JSONL file) when dropped.
    let mut timing = crate::timing::AttemptTiming::start(&state.config.telemetry);
    // Frames older than this are never used (strict arrival or WaitForMatch request).
    let fresh_floor = strict
        .map(|(_, arrival)| arrival)
        .or_else(|| wait.map(|w| w.request_start));
    if let Some(w) = wait {
        timing.set_wait(w.request_start, w.armed_cell());
    }

    // Check if user is enrolled
    let storage = state.storage.read().await;
    let stored_embeddings = match storage.get_embeddings(username) {
        Some(embs) => embs.clone(),
        None => {
            warn!("User not enrolled: {}", username);
            timing.outcome = crate::timing::TimingOutcome::NotEnrolled;
            return AuthDecision::bare(AuthOutcome::NotEnrolled);
        }
    };
    drop(storage);

    // Turn the camera ON on-demand (works even when the screen is unlocked and the
    // camera is otherwise gated off) and wait for frames. The lease is held for the
    // rest of this call so the camera stays up through the capture.
    // The strict sudo profile bounds the wait by its remaining deadline, so a slow
    // camera cannot outlast the strict timeout or the PAM module's give-up time.
    let acquire_wait = sudo_policy::camera_acquire_wait(
        AUTH_CAMERA_ACQUIRE_TIMEOUT,
        strict.map(|(profile, request_start)| request_start + profile.timeout),
        std::time::Instant::now(),
    );
    let acquired = match acquire_wait {
        Some(wait) => state.acquire_camera(wait).await,
        None => Err("Camera wait skipped: strict sudo deadline already passed.".to_string()),
    };
    let _camera_lease = match acquired {
        Ok(lease) => lease,
        Err(err_msg) => {
            error!("{}", err_msg);
            timing.outcome = crate::timing::TimingOutcome::CameraUnavailable;
            return AuthDecision::bare(AuthOutcome::CameraUnavailable { message: err_msg });
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
    let deadline = match strict {
        Some((profile, request_start)) => request_start + profile.timeout,
        None => std::time::Instant::now() + auth_timeout,
    };
    let rec = &state.config.recognition;
    let threshold = strict.map_or(
        state.config.authentication.similarity_threshold,
        |(p, _)| p.threshold,
    );
    let mut passed_embeddings: Vec<Vec<f32>> = Vec::with_capacity(auth_frames);
    // `n_of_m` mode: score every frame as it arrives and stop at the first
    // terminal step. `mean` mode leaves the decider unset.
    let dec = &state.config.decision;
    let n_of_m = match strict {
        Some((p, _)) => Some((p.accept_n, p.window_m)),
        None => (dec.mode == DecisionMode::NOfM).then_some((dec.accept_n, dec.window_m)),
    };
    let mut sequential = n_of_m.map(|(n, m)| SequentialDecider::new(threshold, n, m, auth_frames));
    let mut sequential_step = Step::Continue;
    let mut raw_face_frames: Vec<dataset::CaptureFrameImage> = Vec::new();
    let mut first_quality_face_area_frac = None;
    // Liveness: the aggregator exists while the check is enabled; `liveness`
    // holds its final verdict once decided (None = disabled or still pending).
    let liveness_frames = strict.map_or(state.config.liveness.frames, |(p, _)| p.liveness_frames);
    let mut liveness_agg = state
        .config
        .authentication
        .liveness_enabled
        .then(|| LivenessAggregator::new(liveness_frames));
    let mut liveness: Option<LivenessOutcome> = None;
    // The last quality frame whose embedding was scored (it completes the
    // decision), and whether liveness was already evaluated on that very frame.
    let mut deciding_frame: Option<(image::DynamicImage, crate::ml::Face, bool)> = None;
    // `mean` mode with `[liveness] mean_mode_depth_samples > 0`: every scored
    // frame before the deciding one is kept (at most `auth_frames`) so a sample
    // of them can be depth-checked once the attempt would be accepted. The flag
    // says whether that frame already had a depth check.
    let mean_depth_samples = match (&sequential, liveness_agg.is_some()) {
        (None, true) => state.config.liveness.mean_mode_depth_samples as usize,
        _ => 0,
    };
    let mut scored_frames: Vec<(image::DynamicImage, crate::ml::Face)> = Vec::new();
    let mut scored_checked: Vec<bool> = Vec::new();
    let mut frames_seen = 0usize;
    // Frames not accepted by the multi-face gate / the frame-age guard.
    let mut multi_face_dropped = 0usize;
    let mut stale_dropped = 0usize;
    let max_frame_age = std::time::Duration::from_millis(state.config.camera.max_frame_age_ms);
    // `mean` mode: every frame must reach the identity floor, and at least this
    // many judged-good frames must be collected (`n_of_m` has its own N-of-M).
    let min_decision_frames = state.config.authentication.effective_min_decision_frames();
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
        if let Some(floor) = fresh_floor {
            if !sudo_policy::is_fresh(frame.timestamp, floor) {
                debug!("Skipping a frame captured before the request arrived");
                continue; // `maybe_frame` is empty: waits for the next frame
            }
        }
        if !doormand::frame_policy::is_recent(
            frame.timestamp,
            std::time::Instant::now(),
            max_frame_age,
        ) {
            debug!("Skipping a frame older than camera.max_frame_age_ms");
            stale_dropped += 1;
            continue;
        }
        let frame = (*frame.image).clone();
        let i = frames_seen;
        frames_seen += 1;
        timing.frames_seen = frames_seen;
        timing.mark_first_frame();

        let detect_started = std::time::Instant::now();
        let detect_result = state.ml_pipeline.detect_face(&frame).await;
        timing.detect.record(detect_started.elapsed());
        let mut face = match detect_result {
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

        // Multi-face gate: a second face in view means this frame cannot be
        // attributed to one person (a photo held next to a live face).
        let others: Vec<doormand::frame_policy::Sighting> = face
            .other_faces
            .iter()
            .map(|o| doormand::frame_policy::Sighting {
                area: o.area_frac(),
                score: o.confidence,
            })
            .collect();
        match doormand::frame_policy::choose_face(
            rec.multi_face_policy,
            rec.multi_face_min_area_ratio,
            rec.multi_face_min_score,
            doormand::frame_policy::Sighting {
                area: face.area_frac(),
                score: face.confidence,
            },
            &others,
        ) {
            doormand::frame_policy::FaceChoice::Main => {}
            doormand::frame_policy::FaceChoice::Other(idx) => {
                face = face.other_faces.swap_remove(idx);
            }
            doormand::frame_policy::FaceChoice::Ambiguous => {
                debug!(
                    "Frame {}: more than one face in view; frame not accepted",
                    i
                );
                multi_face_dropped += 1;
                maybe_frame = wait_for_auth_frame_until(&mut rx, deadline).await;
                continue;
            }
        }

        if !frame_passes_quality_gate(&frame, &face, rec, i, "auth") {
            maybe_frame = wait_for_auth_frame_until(&mut rx, deadline).await;
            continue;
        }
        timing.mark_first_valid_frame();
        if first_quality_face_area_frac.is_none() {
            first_quality_face_area_frac = face_area_frac(&face);
        }
        // Anti-spoofing depth gate: evaluated on the first quality frames until the
        // aggregate (`[liveness] frames`, median) is decided; 1 frame by default.
        // The deciding frame is confirmed separately before accepting (below).
        let mut checked_this_frame = false;
        if let (Some(agg), None) = (liveness_agg.as_mut(), liveness.as_ref()) {
            checked_this_frame = true;
            let liveness_started = std::time::Instant::now();
            let frame_result = state.ml_pipeline.liveness_check(&frame, &face).await;
            let depth_elapsed = liveness_started.elapsed();
            timing.liveness = Some(timing.liveness.unwrap_or_default() + depth_elapsed);
            timing.liveness_frames += 1;
            if tracing::enabled!(tracing::Level::DEBUG) {
                let (_, _, box_w, box_h) = face.bbox;
                debug!(
                    "Liveness frame {}: relief={:?} face_box={:.0}x{:.0}px sharpness={:.2} brightness={:.1} depth_ms={:.1}",
                    i,
                    frame_result.as_ref().and_then(LivenessOutcome::relief),
                    box_w * frame.width() as f32,
                    box_h * frame.height() as f32,
                    crate::pipeline::sharpness_score(&frame),
                    crate::pipeline::face_brightness(&frame, face.bbox),
                    depth_elapsed.as_secs_f64() * 1000.0,
                );
            }
            if let Some(outcome) = frame_result {
                if let LivenessProgress::Decided(verdict) = agg.push(outcome) {
                    timing.liveness_passed = Some(verdict.is_pass());
                    timing.liveness_relief = verdict.relief();
                    liveness = Some(verdict);
                }
            }
            if matches!(
                liveness,
                Some(LivenessOutcome::Fail { .. } | LivenessOutcome::Unavailable { .. })
            ) {
                return liveness_failure(
                    state,
                    &mut timing,
                    &attempt_id,
                    username,
                    first_quality_face_area_frac,
                    raw_face_frames,
                    frames_seen,
                    liveness,
                );
            }
        }
        let embed_started = std::time::Instant::now();
        let embed_result = state.ml_pipeline.extract_embedding(&frame, &face).await;
        timing.embed.record(embed_started.elapsed());
        match embed_result {
            Ok(embedding) => {
                if sequential.is_none() {
                    let frame_score = crate::pipeline::template_score(
                        dec.template_score,
                        &embedding,
                        &stored_embeddings,
                    );
                    if doormand::frame_policy::below_identity_floor(
                        frame_score,
                        dec.frame_identity_floor,
                    ) {
                        return rejected_frame(
                            state,
                            &mut timing,
                            RejectedFrame {
                                outcome: AuthOutcome::IdentityFloor,
                                timing_outcome: crate::timing::TimingOutcome::IdentityFloor,
                                code: doormand::decision::IDENTITY_FLOOR_CODE,
                            },
                            &attempt_id,
                            username,
                            raw_face_frames,
                            frames_seen,
                            liveness,
                        );
                    }
                }
                if let Some(decider) = sequential.as_mut() {
                    let score = crate::pipeline::template_score(
                        dec.template_score,
                        &embedding,
                        &stored_embeddings,
                    );
                    debug!("Frame {}: template score {:.4}", i, score);
                    sequential_step =
                        decider.push(score, liveness_state(liveness_agg.is_some(), &liveness));
                }
                passed_embeddings.push(embedding);
                if let Some((prev, prev_face, prev_checked)) =
                    deciding_frame.take().filter(|_| mean_depth_samples > 0)
                {
                    scored_frames.push((prev, prev_face));
                    scored_checked.push(prev_checked);
                }
                deciding_frame = Some((frame, face, checked_this_frame));
            }
            Err(e) => warn!("Frame {}: Embedding error: {}", i, e),
        }
        if passed_embeddings.len() >= auth_frames || sequential_step != Step::Continue {
            break;
        }
        maybe_frame = wait_for_auth_frame_until(&mut rx, deadline).await;
    }

    timing.frames_used = passed_embeddings.len();
    timing.frames_collect = timing
        .camera_open
        .map(|ready| timing.since_start().saturating_sub(ready));
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
        return AuthDecision::bare(AuthOutcome::NoFrames);
    }

    info!(
        "Captured {} auth frames; {} had detected faces; {} quality embeddings collected (target {})",
        frames_seen,
        raw_face_frames.len(),
        passed_embeddings.len(),
        auth_frames
    );

    if passed_embeddings.is_empty() && multi_face_dropped > 0 {
        return rejected_frame(
            state,
            &mut timing,
            RejectedFrame {
                outcome: AuthOutcome::AmbiguousFaces,
                timing_outcome: crate::timing::TimingOutcome::AmbiguousFaces,
                code: doormand::decision::AMBIGUOUS_FACES_CODE,
            },
            &attempt_id,
            username,
            raw_face_frames,
            frames_seen,
            liveness,
        );
    }

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
        return AuthDecision {
            frames_seen,
            liveness,
            ..AuthDecision::bare(AuthOutcome::NoClearFace)
        };
    }

    // `mean` mode: do not decide on whatever the deadline left (one frame on a
    // loaded CPU). Too few frames is a clear, unjudged rejection, or an
    // ambiguity rejection when the multi-face gate was what removed them.
    if sequential.is_none()
        && !doormand::frame_policy::enough_frames(passed_embeddings.len(), min_decision_frames)
    {
        warn!(
            "Authentication for {}: {} judged-good frames, {} required; {} dropped by the multi-face gate, {} by the frame-age guard",
            username,
            passed_embeddings.len(),
            min_decision_frames,
            multi_face_dropped,
            stale_dropped
        );
        let (outcome, timing_outcome, code) = if multi_face_dropped > 0 {
            (
                AuthOutcome::AmbiguousFaces,
                crate::timing::TimingOutcome::AmbiguousFaces,
                doormand::decision::AMBIGUOUS_FACES_CODE,
            )
        } else {
            (
                AuthOutcome::InsufficientFrames,
                crate::timing::TimingOutcome::InsufficientFrames,
                doormand::decision::INSUFFICIENT_FRAMES_CODE,
            )
        };
        return rejected_frame(
            state,
            &mut timing,
            RejectedFrame {
                outcome,
                timing_outcome,
                code,
            },
            &attempt_id,
            username,
            raw_face_frames,
            frames_seen,
            liveness,
        );
    }

    // The authentication timeout ended the run before the liveness aggregate
    // was decided: fail closed, as unavailable.
    if liveness_agg.is_some() && liveness.is_none() {
        let undecided = Some(LivenessOutcome::Unavailable {
            reason: "liveness undecided before the authentication timeout".to_string(),
        });
        return liveness_failure(
            state,
            &mut timing,
            &attempt_id,
            username,
            first_quality_face_area_frac,
            raw_face_frames,
            frames_seen,
            undecided,
        );
    }

    let decision_started = std::time::Instant::now();
    let aggregated = crate::pipeline::aggregate_embeddings(&passed_embeddings)
        .unwrap_or_else(|| passed_embeddings[0].clone());

    let gallery_similarities: Vec<f32> = stored_embeddings
        .iter()
        .map(|stored_emb| crate::ml::cosine_similarity(&aggregated, stored_emb))
        .collect();
    let (matched, best_similarity) = match &sequential {
        Some(decider) => match sequential_step {
            Step::Accept { similarity } => (true, similarity),
            // Rejected or ran out of frames/time: report the best frame score.
            Step::Reject(_) | Step::Continue => (false, decider.best_in_window()),
        },
        None => {
            let score = crate::pipeline::template_score(
                dec.template_score,
                &aggregated,
                &stored_embeddings,
            );
            (score >= threshold, score)
        }
    };
    // Bind liveness to the frames that produced the match: the first quality
    // frames were checked, but a photo shown afterwards would still score as the
    // owner. The deciding (last scored) frame must itself be live. Costs one more
    // depth inference unless that frame was the one already checked.
    if matched && liveness_agg.is_some() {
        if let Some((frame, face, checked)) = deciding_frame.take() {
            let recheck_started = std::time::Instant::now();
            let confirmed = doormand::liveness::confirm_on_deciding_frame(checked, || {
                state.ml_pipeline.liveness_check(&frame, &face)
            })
            .await;
            if !checked {
                timing.liveness =
                    Some(timing.liveness.unwrap_or_default() + recheck_started.elapsed());
                timing.liveness_frames += 1;
            }
            if let Err(not_live) = confirmed {
                timing.liveness_passed = Some(false);
                timing.liveness_relief = not_live.relief();
                return liveness_failure(
                    state,
                    &mut timing,
                    &attempt_id,
                    username,
                    first_quality_face_area_frac,
                    raw_face_frames,
                    frames_seen,
                    Some(not_live),
                );
            }
        }
    }
    // `mean` mode, opt-in: also depth-check frames sampled across the sequence,
    // so a photo shown between the checked first frames and the deciding frame
    // cannot be averaged in unseen.
    if matched && mean_depth_samples > 0 {
        let sampled_started = std::time::Instant::now();
        let sampled = doormand::liveness::confirm_sampled_frames(
            &scored_checked,
            mean_depth_samples,
            |index| {
                let (frame, face) = &scored_frames[index];
                state.ml_pipeline.liveness_check(frame, face)
            },
        )
        .await;
        let performed = match &sampled {
            Ok(n) => *n,
            // The failing check ran too; the exact count is not needed for timing.
            Err(_) => 1,
        };
        timing.liveness = Some(timing.liveness.unwrap_or_default() + sampled_started.elapsed());
        timing.liveness_frames += performed as u32;
        if let Err(not_live) = sampled {
            timing.liveness_passed = Some(false);
            timing.liveness_relief = not_live.relief();
            return liveness_failure(
                state,
                &mut timing,
                &attempt_id,
                username,
                first_quality_face_area_frac,
                raw_face_frames,
                frames_seen,
                Some(not_live),
            );
        }
    }
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
    }
    AuthDecision {
        outcome: if matched {
            AuthOutcome::Matched
        } else {
            AuthOutcome::NotMatched
        },
        similarity: Some(best_similarity),
        frames_used: passed_embeddings.len(),
        frames_seen,
        liveness,
    }
}

/// How a frame-level rule ended an attempt.
struct RejectedFrame {
    outcome: AuthOutcome,
    timing_outcome: crate::timing::TimingOutcome,
    code: &'static str,
}

/// Terminal rejection by a frame-level rule (identity floor, multi-face gate,
/// too few frames): record the attempt and build the decision. Same
/// bookkeeping as [`liveness_failure`], without the spoof label.
#[allow(clippy::too_many_arguments)]
fn rejected_frame(
    state: &DaemonState,
    timing: &mut crate::timing::AttemptTiming,
    why: RejectedFrame,
    attempt_id: &str,
    username: &str,
    raw_face_frames: Vec<dataset::CaptureFrameImage>,
    frames_seen: usize,
    liveness: Option<LivenessOutcome>,
) -> AuthDecision {
    timing.outcome = why.timing_outcome;
    warn!(
        "Authentication failed for {}: frame rule [{}]",
        username, why.code
    );
    log_auth_failure_attempt(
        state,
        attempt_id,
        username,
        Outcome::Rejected,
        None,
        why.code,
        AttemptConditions::default(),
    );
    log_auth_capture_pack(
        state,
        attempt_id,
        Some(username),
        Outcome::Rejected,
        GtSource::PamFaceAuth,
        raw_face_frames,
    );
    AuthDecision {
        frames_seen,
        liveness,
        ..AuthDecision::bare(why.outcome)
    }
}

/// Liveness as the sequential decider sees it. A rejected or unavailable check
/// ends the run before any frame is scored, so only pass or disabled reach it.
fn liveness_state(enabled: bool, verdict: &Option<LivenessOutcome>) -> LivenessState {
    match (enabled, verdict) {
        (false, _) | (true, Some(LivenessOutcome::Pass { .. })) => LivenessState::Passed,
        (true, None) => LivenessState::Pending,
        (true, Some(LivenessOutcome::Fail { .. } | LivenessOutcome::Unavailable { .. })) => {
            LivenessState::Failed
        }
    }
}

/// Terminal liveness failure (rejected, unavailable or undecided): record the
/// attempt and build the decision. `liveness` is the final `Fail` or
/// `Unavailable` verdict.
#[allow(clippy::too_many_arguments)]
fn liveness_failure(
    state: &DaemonState,
    timing: &mut crate::timing::AttemptTiming,
    attempt_id: &str,
    username: &str,
    face_area_frac: Option<f32>,
    raw_face_frames: Vec<dataset::CaptureFrameImage>,
    frames_seen: usize,
    liveness: Option<LivenessOutcome>,
) -> AuthDecision {
    let (auth_outcome, timing_outcome, code) =
        if matches!(liveness, Some(LivenessOutcome::Unavailable { .. })) {
            (
                AuthOutcome::LivenessError,
                crate::timing::TimingOutcome::LivenessError,
                doormand::liveness::LIVENESS_UNAVAILABLE_CODE,
            )
        } else {
            (
                AuthOutcome::LivenessRejected,
                crate::timing::TimingOutcome::LivenessRejected,
                doormand::liveness::LIVENESS_REJECTED_CODE,
            )
        };
    timing.outcome = timing_outcome;
    warn!(
        "Authentication failed for {}: liveness gate [{}]: {:?}",
        username, code, liveness
    );
    log_auth_failure_attempt(
        state,
        attempt_id,
        username,
        Outcome::SuspectedSpoof,
        Some(false),
        code,
        AttemptConditions {
            face_area_frac,
            ..AttemptConditions::default()
        },
    );
    log_auth_capture_pack(
        state,
        attempt_id,
        Some(username),
        Outcome::SuspectedSpoof,
        GtSource::PamFaceAuth,
        raw_face_frames,
    );
    AuthDecision {
        frames_seen,
        liveness,
        ..AuthDecision::bare(auth_outcome)
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

    let brightness = crate::pipeline::face_brightness(frame, face.bbox);
    let bright_ok = rec.min_brightness <= 0.0 || brightness >= rec.min_brightness;

    let pass = area_ok && sharp_ok && bright_ok;
    debug!(
        "{} quality gate frame {}: sharpness={:.2} (min {:.2}) area_frac={:.4} (min {:.4}) brightness={:.1} (min {:.1}) -> {}",
        ctx,
        idx,
        sharpness,
        rec.min_sharpness,
        area_frac,
        rec.min_face_area_frac,
        brightness,
        rec.min_brightness,
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
    if !dataset::capture_pack_allowed(&state.config.dataset, outcome) || frames.is_empty() {
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
            other_faces: Vec::new(),
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

    /// Synthetic frame: its similarity to the owner's template and the depth
    /// relief it would show (a live face has relief, a photo is flat).
    #[derive(Clone, Copy)]
    struct SynthFrame {
        score: f32,
        relief: f32,
    }

    const LIVE: SynthFrame = SynthFrame {
        score: 0.8,
        relief: 0.2,
    };
    /// A photo of the owner: matches just as well, but flat.
    const PHOTO: SynthFrame = SynthFrame {
        score: 0.8,
        relief: 0.01,
    };
    const THRESHOLD: f32 = 0.4;
    const RELIEF_THRESHOLD: f32 = 0.05;

    fn depth_check(frame: SynthFrame) -> LivenessOutcome {
        if frame.relief >= RELIEF_THRESHOLD {
            LivenessOutcome::Pass {
                relief: frame.relief,
            }
        } else {
            LivenessOutcome::Fail {
                relief: frame.relief,
            }
        }
    }

    /// Mirrors the order of `run_authenticate`: liveness on the first quality
    /// frames until decided, every frame scored, accept via the sequential
    /// decider, then the deciding-frame confirmation. `Ok(similarity)` = accepted.
    async fn run(frames: &[SynthFrame], liveness_frames: usize, confirm: bool) -> bool {
        use crate::pipeline::{LivenessState, SequentialDecider, Step};
        use doormand::liveness::{LivenessAggregator, LivenessProgress};
        let mut agg = LivenessAggregator::new(liveness_frames);
        let mut verdict: Option<LivenessOutcome> = None;
        let mut decider = SequentialDecider::new(THRESHOLD, 3, 5, frames.len());
        let mut deciding: Option<(SynthFrame, bool)> = None;
        let mut step = Step::Continue;
        for frame in frames {
            let mut checked = false;
            if verdict.is_none() {
                checked = true;
                if let LivenessProgress::Decided(v) = agg.push(depth_check(*frame)) {
                    verdict = Some(v);
                }
            }
            if matches!(verdict, Some(ref v) if !v.is_pass()) {
                return false;
            }
            let state = match verdict {
                Some(_) => LivenessState::Passed,
                None => LivenessState::Pending,
            };
            step = decider.push(frame.score, state);
            deciding = Some((*frame, checked));
            if step != Step::Continue {
                break;
            }
        }
        if !matches!(step, Step::Accept { .. }) {
            return false;
        }
        if !confirm {
            return true;
        }
        let (frame, checked) = deciding.expect("accepted after at least one frame");
        doormand::liveness::confirm_on_deciding_frame(checked, || async move {
            Some(depth_check(frame))
        })
        .await
        .is_ok()
    }

    #[tokio::test]
    async fn live_then_photo_frames_are_accepted_without_the_binding() {
        // The review's finding: this is the hole the binding closes.
        let swap = [LIVE, PHOTO, PHOTO, PHOTO];
        assert!(run(&swap, 1, false).await);
    }

    #[tokio::test]
    async fn live_then_photo_swap_is_rejected_by_the_deciding_frame_check() {
        for swap in [
            vec![LIVE, PHOTO, PHOTO, PHOTO],
            vec![LIVE, LIVE, PHOTO, PHOTO, PHOTO],
        ] {
            assert!(!run(&swap, 1, true).await, "swap must not accept");
        }
        // Multi-frame liveness (3 frames) decided by two live frames (majority),
        // then photos complete the three accepting hits.
        let swap = [LIVE, LIVE, PHOTO, PHOTO, PHOTO];
        assert!(!run(&swap, 3, true).await);
    }

    /// Mirrors `mean` mode: liveness on the first quality frames until decided,
    /// the mean score decides, then the deciding-frame confirmation and the
    /// optional sampled depth checks. Returns (accepted, depth checks run).
    async fn run_mean(
        frames: &[SynthFrame],
        liveness_frames: usize,
        samples: usize,
    ) -> (bool, usize) {
        use doormand::liveness::{LivenessAggregator, LivenessProgress};
        let mut agg = LivenessAggregator::new(liveness_frames);
        let mut verdict: Option<LivenessOutcome> = None;
        let mut checks = 0;
        let mut checked_flags = Vec::new();
        for frame in frames {
            let checked = verdict.is_none();
            if checked {
                checks += 1;
                if let LivenessProgress::Decided(v) = agg.push(depth_check(*frame)) {
                    verdict = Some(v);
                }
            }
            if matches!(verdict, Some(ref v) if !v.is_pass()) {
                return (false, checks);
            }
            checked_flags.push(checked);
        }
        let mean = frames.iter().map(|f| f.score).sum::<f32>() / frames.len() as f32;
        if mean < THRESHOLD {
            return (false, checks);
        }
        let (last, rest) = frames.split_last().expect("at least one frame");
        let last_checked = checked_flags.pop().expect("one flag per frame");
        let deciding = doormand::liveness::confirm_on_deciding_frame(last_checked, || async {
            checks += 1;
            Some(depth_check(*last))
        })
        .await;
        if deciding.is_err() {
            return (false, checks);
        }
        let sampled = doormand::liveness::confirm_sampled_frames(
            &checked_flags,
            samples,
            |index| async move { Some(depth_check(rest[index])) },
        )
        .await;
        match sampled {
            Ok(n) => (true, checks + n),
            Err(_) => (false, checks + 1),
        }
    }

    #[tokio::test]
    async fn mean_mode_default_checks_only_first_and_deciding_frames() {
        // Documents the default (`mean_mode_depth_samples = 0`): photos between
        // the checked first frame and the live deciding frame are averaged in
        // unseen. Enable sampling to close this; see docs/decision-and-liveness.md.
        let swap = [LIVE, PHOTO, PHOTO, PHOTO, LIVE];
        assert_eq!(run_mean(&swap, 1, 0).await, (true, 2));
    }

    #[tokio::test]
    async fn mean_mode_samples_catch_photo_frames_mid_sequence() {
        let swap = [LIVE, PHOTO, PHOTO, PHOTO, LIVE];
        assert!(!run_mean(&swap, 1, 1).await.0, "middle sample is a photo");
        assert!(!run_mean(&swap, 1, 3).await.0);
        // A single photo that no sample lands on stays unseen: sampling is
        // statistical, not exhaustive (documented).
        let one_photo = [LIVE, LIVE, PHOTO, LIVE, LIVE, LIVE, LIVE, LIVE, LIVE];
        assert!(run_mean(&one_photo, 1, 1).await.0);
        assert!(!run_mean(&one_photo, 1, 7).await.0);
    }

    #[tokio::test]
    async fn mean_mode_samples_accept_genuine_live_sequences_and_cost_one_check_each() {
        let live = [LIVE; 8];
        assert_eq!(run_mean(&live, 1, 0).await, (true, 2));
        assert_eq!(run_mean(&live, 1, 3).await, (true, 5));
        assert!(!run_mean(&[PHOTO; 8], 1, 3).await.0);
    }

    #[tokio::test]
    async fn genuine_live_frames_are_still_accepted_and_photos_alone_never() {
        assert!(run(&[LIVE, LIVE, LIVE, LIVE], 1, true).await);
        assert!(!run(&[PHOTO, PHOTO, PHOTO, PHOTO], 1, true).await);
    }
}

async fn handle_enroll(state: &DaemonState, username: &str, add: bool) -> Response {
    info!("Enrollment request for user: {} (add: {})", username, add);

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
    let existing_templates: Vec<Vec<f32>> = {
        let storage = state.storage.read().await;
        storage
            .get_embeddings(username)
            .cloned()
            .unwrap_or_default()
    };
    if !existing_templates.is_empty() {
        info!(
            "Capacious enroll: merging {} new + {} existing embeddings for {}",
            valid_embeddings.len(),
            existing_templates.len(),
            username
        );
    }
    let selected_embeddings = match crate::pipeline::plan_templates(
        valid_embeddings,
        &existing_templates,
        if add {
            crate::pipeline::EnrollMode::Append
        } else {
            crate::pipeline::EnrollMode::Merge
        },
        state.config.enrollment.prune_cosine,
        TARGET_EMBEDDINGS,
    ) {
        Ok(templates) => templates,
        Err(reason) => return Response::Failure { reason },
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

/// Loaded ONNX Runtime linkage/version, `None` when built without an ORT backend.
#[cfg(feature = "_ort")]
fn ort_runtime_info() -> Option<String> {
    Some(crate::ml::ort_backend::runtime_info())
}

#[cfg(not(feature = "_ort"))]
fn ort_runtime_info() -> Option<String> {
    None
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
        ort_runtime: ort_runtime_info(),
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
