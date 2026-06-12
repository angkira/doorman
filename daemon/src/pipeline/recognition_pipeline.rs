use super::aggregate::aggregate_embeddings;
use super::types::DetectionResult;
use super::UnlockContext;
use crate::debug_stream::DebugStreamBroadcaster;
use crate::ml::MLPipeline;
use crate::storage::Storage;
use doorman_shared::{StreamMessage, Config, DetectionInfo};
use std::collections::VecDeque;
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
    // Used for the Stage-B proactive-unlock liveness gate (depth-relief). Same
    // model session the PAM auth path uses; `liveness_gate` runs ONCE per unlock
    // attempt, never per frame.
    ml_pipeline: Arc<MLPipeline>,
    // `Some` only in system mode with a resolved logind session: carries the
    // `Unlocker` + armed/lock-generation watches. `None` in --user/--preview/dev.
    unlock_ctx: Option<UnlockContext>,
) {
    info!("Recognition pipeline started");

    // Stage-B proactive-unlock one-shot guard. `armed_for_gen` is the lock
    // generation we last consumed an unlock for; while it equals the current
    // `lock_gen` we are "spent" and won't try to unlock again until the next
    // `Lock` signal bumps the generation (re-arming us). Starts unset so the
    // very first locked session is eligible.
    let proactive_enabled = config.authentication.proactive_unlock && unlock_ctx.is_some();
    let mut spent_for_gen: Option<u64> = None;

    // Phase 1: rolling window of recent quality-passed embeddings for the
    // tracked face. Single-user/single-face preview, so one window is enough.
    // We score the AGGREGATED embedding (renormalized mean) against the stored
    // template instead of the noisy per-frame embedding.
    let window_size = config.recognition.aggregation_window.max(1);
    let mut embedding_window: VecDeque<Vec<f32>> = VecDeque::with_capacity(window_size);

    // Box-color hysteresis: require K consecutive aggregated results on the
    // opposite side of the threshold before flipping the displayed decision.
    let hysteresis_frames = config.recognition.hysteresis_frames.max(1);
    let mut displayed_recognized = false; // current debounced preview decision
    let mut pending_recognized: Option<bool> = None; // candidate state being confirmed
    let mut pending_count: u32 = 0;

    while let Some(mut result) = result_rx.recv().await {
        let is_locked = *system_locked.read().await;

        // Pull the (optional) liveness inputs off the result up front, before the
        // `result.face` / `result.embedding` matches move pieces of it. These are
        // `Some` only on quality-passed embedding frames (set by the detection
        // stage); detect-only / no-face frames carry neither and can never drive
        // a proactive unlock.
        let liveness_frame = result.frame.take();
        let liveness_face = result.ml_face.take();

        // NOTE: we no longer `continue` when the session is unlocked. The
        // detection/recognition pipeline now runs subscriber-gated (only while a
        // preview client is connected — see the supervisor in `main.rs`), so when
        // it IS running we always want to emit per-frame metadata for the preview
        // overlay regardless of lock state. The *unlock* side-effect below stays
        // gated on `is_locked`, so PAM/auto-unlock semantics are unchanged: an
        // already-unlocked session is never re-locked or otherwise mutated here.
        let processing_time_ms = result.processing_time.as_millis() as u32;
        let timestamp_ms = start_time.elapsed().as_millis() as u64;

        // No face detected at all -> broadcast "no face" and reset hysteresis.
        let face = match result.face {
            Some(f) => f,
            None => {
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

        // Face present but this frame carries no (quality-passed) embedding —
        // detect-only frame or one rejected by the quality gate. Keep tracking
        // the box but do not score or disturb the aggregation window.
        let embedding = match result.embedding {
            Some(e) if !e.is_empty() => e,
            _ => {
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
                        // Preserve the current debounced decision so the box
                        // color doesn't drop to "unknown" on detect-only frames.
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

        // Push this quality-passed embedding into the rolling window and score
        // the AGGREGATED (renormalized-mean) embedding rather than the raw frame.
        embedding_window.push_back(embedding);
        while embedding_window.len() > window_size {
            embedding_window.pop_front();
        }
        let window: Vec<Vec<f32>> = embedding_window.iter().cloned().collect();
        let aggregated = aggregate_embeddings(&window).unwrap_or_else(|| window[0].clone());

        // Match aggregated embedding against enrolled users.
        let storage_guard = storage.read().await;
        let mut best_match: Option<(String, f32)> = None;
        let mut best_similarity = 0.0f32;

        for user_info in storage_guard.list_users() {
            if let Some(stored_embeddings) = storage_guard.get_embeddings(&user_info.username) {
                // Compare with all stored embeddings and take best match
                let max_similarity = stored_embeddings
                    .iter()
                    .map(|stored_emb| cosine_similarity(&aggregated, stored_emb))
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

        debug!(
            "Aggregated recognition: window={} cosine={:.4} threshold={:.2} (frame {})",
            window.len(), best_similarity, config.authentication.similarity_threshold, result.sequence
        );

        // Box-color hysteresis: debounce RECOGNIZED <-> unknown flips.
        let raw_recognized = best_match.is_some();
        if raw_recognized == displayed_recognized {
            pending_recognized = None;
            pending_count = 0;
        } else {
            match pending_recognized {
                Some(p) if p == raw_recognized => pending_count += 1,
                _ => {
                    pending_recognized = Some(raw_recognized);
                    pending_count = 1;
                }
            }
            if pending_count >= hysteresis_frames {
                displayed_recognized = raw_recognized;
                pending_recognized = None;
                pending_count = 0;
            }
        }

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

            // ---- Stage B: proactive logind unlock -------------------------------
            // Decision path: gate on (a) proactive_unlock enabled + a resolved
            // session, (b) the lock supervisor reports `armed` (LockedHint &&
            // !IdleHint), (c) this lock generation hasn't already been spent by a
            // prior unlock (one-shot), (d) the FATAL depth-liveness gate PASSES.
            // Only then do we call `Session.Unlock()`. Liveness missing/errored =>
            // NO unlock (fail-safe reject), identical semantics to the PAM path.
            if proactive_enabled {
                let ctx = unlock_ctx.as_ref().expect("proactive_enabled implies unlock_ctx");
                let armed = *ctx.armed_rx.borrow();
                let cur_gen = *ctx.lock_gen_rx.borrow();
                let already_spent = spent_for_gen == Some(cur_gen);

                if should_attempt_unlock(proactive_enabled, armed, already_spent) {
                    // similarity is already >= similarity_threshold (best_match was
                    // only set on that condition), so we only need the liveness gate.
                    match (liveness_frame.as_ref(), liveness_face.as_ref()) {
                        (Some(frame), Some(face)) => {
                            match ml_pipeline.liveness_gate(frame, face).await {
                                Ok(true) => {
                                    // One-shot: mark this generation spent BEFORE the
                                    // call so a slow/failed Unlock can't be retried in a
                                    // tight loop within the same lock.
                                    spent_for_gen = Some(cur_gen);
                                    info!(
                                        "Proactive unlock: {} recognized (cosine {:.4} >= {:.2}) + liveness PASS, armed (gen={}) -> calling Session.Unlock()",
                                        username, similarity, config.authentication.similarity_threshold, cur_gen
                                    );
                                    match ctx.unlocker.unlock().await {
                                        Ok(()) => info!("Proactive unlock: Session.Unlock() succeeded"),
                                        Err(e) => warn!(
                                            "Proactive unlock: Session.Unlock() failed ({e}). \
                                             If 'not authorized', the polkit rule granting \
                                             'org.freedesktop.login1.unlock-sessions' to the daemon \
                                             user is not installed yet (installer stage)."
                                        ),
                                    }
                                }
                                Ok(false) => warn!(
                                    "Proactive unlock REJECTED: liveness/anti-spoof gate failed for {} — NOT unlocking",
                                    username
                                ),
                                Err(e) => warn!(
                                    "Proactive unlock REJECTED: liveness error for {} ({e}) — NOT unlocking (fail-safe)",
                                    username
                                ),
                            }
                        }
                        _ => {
                            // Recognized on a frame that carried no liveness inputs
                            // (shouldn't happen for quality-passed embeddings). Fail
                            // safe: no inputs -> no unlock.
                            debug!("Proactive unlock: no liveness inputs on this frame — skipping unlock");
                        }
                    }
                } else if already_spent {
                    debug!("Proactive unlock: lock generation {} already spent — skipping (one-shot)", cur_gen);
                }
            }
            // ---------------------------------------------------------------------

            // Broadcast success to debug stream
            // Convert normalized bbox [0,1] to pixel coords
            let bbox_px = (
                (face.bbox.0 * result.frame_width as f32) as u32,
                (face.bbox.1 * result.frame_height as f32) as u32,
                (face.bbox.2 * result.frame_width as f32) as u32,
                (face.bbox.3 * result.frame_height as f32) as u32,
            );
            // Preview color is debounced by hysteresis: only show RECOGNIZED
            // once the decision has been confirmed for `hysteresis_frames`.
            let message = StreamMessage::Detection {
                timestamp_ms,
                detection: DetectionInfo {
                    bbox: Some(bbox_px),
                    frame_size: Some((result.frame_width, result.frame_height)),
                    confidence: Some(face.confidence),
                    recognized_user: if displayed_recognized { Some(username) } else { None },
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

/// Pure Stage-B gate: should the recognition path even ATTEMPT a proactive
/// unlock for this (already recognized + above-threshold) frame? This is the
/// gate that runs BEFORE the FATAL liveness check; liveness PASS is the final
/// condition the caller enforces separately. Decision = proactive_unlock
/// enabled AND a resolved session present (`enabled`) AND the lock supervisor
/// reports the session armed (locked && !idle) AND the current lock generation
/// has not already been spent by a prior unlock (one-shot re-arm on next Lock).
fn should_attempt_unlock(enabled: bool, armed: bool, already_spent: bool) -> bool {
    enabled && armed && !already_spent
}

#[cfg(test)]
mod proactive_unlock_tests {
    use super::should_attempt_unlock;

    #[test]
    fn unlocks_only_when_enabled_armed_and_fresh() {
        // recognized + live is implied (caller only reaches this after a match
        // and runs liveness AFTER this gate). Here we assert the pre-liveness gate.
        assert!(should_attempt_unlock(true, true, false), "enabled + armed + fresh -> attempt");
    }

    #[test]
    fn disabled_never_unlocks() {
        // proactive_unlock=false (or no session) -> pipeline still gates/recognizes
        // but PAM stays the unlock path.
        assert!(!should_attempt_unlock(false, true, false));
    }

    #[test]
    fn disarmed_never_unlocks() {
        // not locked, or display idle -> not armed -> no unlock.
        assert!(!should_attempt_unlock(true, false, false));
    }

    #[test]
    fn one_shot_spent_generation_blocks_repeat() {
        // after a successful unlock the generation is spent until the next Lock
        // signal bumps it -> no repeat-unlock spam.
        assert!(!should_attempt_unlock(true, true, true));
    }
}
