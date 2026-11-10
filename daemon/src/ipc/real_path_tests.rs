//! Mean-mode swap tests through the REAL `run_authenticate` path.
//!
//! The seam is test-only and touches no production code: a [`DaemonState`] is
//! assembled in the test with
//! * a `watch` channel as the camera frame source, fed in lock step by the mock
//!   backend (it publishes frame `i + 1` while it detects frame `i`, so the
//!   consumer never misses a frame), and
//! * an [`MLPipeline`] built with `MLPipeline::with_backend` around a scripted
//!   backend that decodes the frame index from pixel (0, 0) and answers the
//!   detection, the embedding and the depth relief from the script.
//!
//! Everything else (quality gate, liveness aggregation, mean decision, deciding
//! frame check, sampled depth checks, timing, dataset hooks) is production code.

use super::run_authenticate;
use crate::ml::{Face, MLBackend, MLPipeline};
use crate::pipeline::RawFrame;
use crate::storage::Storage;
use crate::DaemonState;
use doorman_shared::config::DecisionMode;
use doorman_shared::Config;
use doormand::decision::{AuthDecision, AuthOutcome};
use doormand::liveness::LivenessOutcome;
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{watch, RwLock};

const USER: &str = "owner";
const SIMILARITY_THRESHOLD: f32 = 0.4;
const RELIEF_THRESHOLD: f32 = 0.05;
const FRAME_SIDE: u32 = 8;
/// Relief of a live face and of a flat photo (threshold is `RELIEF_THRESHOLD`).
const LIVE_RELIEF: f32 = 0.2;
const FLAT_RELIEF: f32 = 0.01;
/// Embeddings are 3-d unit vectors: cosine to the owner template is 1.0 for the
/// owner and 0.0 for the impostor, around `SIMILARITY_THRESHOLD`.
const OWNER_EMBEDDING: [f32; 3] = [1.0, 0.0, 0.0];
const IMPOSTOR_EMBEDDING: [f32; 3] = [0.0, 1.0, 0.0];

/// What the camera shows in one frame.
#[derive(Clone, Copy, Debug)]
enum Scene {
    /// The enrolled owner, in person: owner embedding, 3-d relief.
    LiveOwner,
    /// A live person who is not the owner: other embedding, 3-d relief.
    LiveImpostor,
    /// A photo of the owner: owner embedding, flat.
    Photo,
}

impl Scene {
    fn embedding(self) -> Vec<f32> {
        match self {
            Scene::LiveOwner | Scene::Photo => OWNER_EMBEDDING.to_vec(),
            Scene::LiveImpostor => IMPOSTOR_EMBEDDING.to_vec(),
        }
    }

    fn relief(self) -> f32 {
        match self {
            Scene::LiveOwner | Scene::LiveImpostor => LIVE_RELIEF,
            Scene::Photo => FLAT_RELIEF,
        }
    }
}

use Scene::{LiveImpostor, LiveOwner, Photo};

fn frame(index: usize) -> RawFrame {
    let mut image = RgbImage::new(FRAME_SIDE, FRAME_SIDE);
    image.put_pixel(
        0,
        0,
        Rgb([u8::try_from(index).expect("small script"), 0, 0]),
    );
    RawFrame {
        image: Arc::new(DynamicImage::ImageRgb8(image)),
        timestamp: std::time::Instant::now(),
        sequence: index as u64,
    }
}

fn frame_index(image: &DynamicImage) -> usize {
    usize::from(image.get_pixel(0, 0).0[0])
}

struct ScriptedBackend {
    script: Vec<Scene>,
    camera: watch::Sender<Option<RawFrame>>,
    depth_calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl MLBackend for ScriptedBackend {
    async fn detect_face(&self, image: &DynamicImage) -> anyhow::Result<Option<Face>> {
        // Lock step: the next frame becomes visible once this one is being handled.
        let next = frame_index(image) + 1;
        if next < self.script.len() {
            self.camera.send_replace(Some(frame(next)));
        }
        Ok(Some(Face {
            bbox: (0.25, 0.25, 0.5, 0.5),
            confidence: 0.99,
            frame_dimensions: (FRAME_SIDE, FRAME_SIDE),
            landmarks: None,
        }))
    }

    async fn check_liveness(&self, _image: &DynamicImage, _face: &Face) -> anyhow::Result<bool> {
        Ok(true)
    }

    async fn depth_relief(
        &self,
        image: &DynamicImage,
        _face: &Face,
    ) -> anyhow::Result<Option<f32>> {
        self.depth_calls.fetch_add(1, Ordering::SeqCst);
        Ok(Some(self.script[frame_index(image)].relief()))
    }

    async fn extract_embedding(
        &self,
        image: &DynamicImage,
        _face: &Face,
    ) -> anyhow::Result<Vec<f32>> {
        Ok(self.script[frame_index(image)].embedding())
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "scripted-mock"
    }
}

/// Outcome of one real `run_authenticate` call and the depth inferences it ran.
struct Run {
    decision: AuthDecision,
    depth_calls: usize,
}

async fn authenticate(script: &[Scene], mean_mode_depth_samples: u32) -> Run {
    let (state, depth_calls, _camera, _dir) =
        build_state(script, script.len(), mean_mode_depth_samples).await;
    let decision = run_authenticate(&state, USER, None, None).await;
    Run {
        decision,
        depth_calls: depth_calls.load(Ordering::SeqCst),
    }
}

async fn build_state(
    script: &[Scene],
    auth_frames: usize,
    mean_mode_depth_samples: u32,
) -> (
    DaemonState,
    Arc<AtomicUsize>,
    watch::Sender<Option<RawFrame>>,
    tempfile::TempDir,
) {
    let data_dir = tempfile::tempdir().expect("temp data dir");
    let mut config = Config::default();
    config.daemon.data_dir = data_dir.path().to_string_lossy().into_owned();
    config.decision.mode = DecisionMode::Mean;
    config.authentication.auth_frames = auth_frames;
    config.authentication.similarity_threshold = SIMILARITY_THRESHOLD;
    config.authentication.liveness_enabled = true;
    config.authentication.liveness_depth_threshold = RELIEF_THRESHOLD;
    config.liveness.frames = 1;
    config.liveness.mean_mode_depth_samples = mean_mode_depth_samples;
    // The synthetic 8x8 frames have no texture: switch the quality gate off.
    config.recognition.min_sharpness = 0.0;
    config.recognition.min_face_area_frac = 0.0;
    config.recognition.min_brightness = 0.0;
    config.dataset.logging_enabled = false;

    let (camera_tx, camera_rx) = watch::channel(Some(frame(0)));
    let depth_calls = Arc::new(AtomicUsize::new(0));
    let backend = Arc::new(ScriptedBackend {
        script: script.to_vec(),
        camera: camera_tx.clone(),
        depth_calls: depth_calls.clone(),
    });
    let mut storage = Storage::new_with_dir(data_dir.path())
        .await
        .expect("temp storage");
    storage
        .store_embeddings(USER.to_string(), vec![OWNER_EMBEDDING.to_vec()])
        .await
        .expect("store template");

    let state = DaemonState {
        latest_frame: camera_rx,
        ml_pipeline: Arc::new(MLPipeline::with_backend(backend, &config)),
        storage: Arc::new(RwLock::new(storage)),
        start_time: std::time::Instant::now(),
        config: Arc::new(config.clone()),
        latest_detection: Arc::new(RwLock::new(None)),
        debug_broadcaster: Arc::new(crate::debug_stream::DebugStreamBroadcaster::new(1)),
        frame_broadcaster: None,
        system_locked: Arc::new(RwLock::new(true)),
        armed: None,
        locked: None,
        sudo_guard: Arc::new(doormand::sudo_policy::SudoGuard::new(
            config.pam.sudo.max_failures,
            std::time::Duration::from_secs(config.pam.sudo.suspend_secs),
        )),
        capture_requests: Arc::new(AtomicUsize::new(0)),
    };

    (state, depth_calls, camera_tx, data_dir)
}

fn assert_spoof_rejected(run: &Run) {
    assert_eq!(run.decision.outcome, AuthOutcome::LivenessRejected);
    assert!(matches!(
        run.decision.liveness,
        Some(LivenessOutcome::Fail { .. })
    ));
}

#[tokio::test]
async fn mean_mode_without_samples_rejects_when_the_deciding_frame_is_a_photo() {
    // Live impostor first (passes the first-frame check), then photos of the
    // owner lift the mean above the threshold; the deciding frame is a photo.
    let run = authenticate(&[LiveImpostor, LiveImpostor, Photo, Photo, Photo], 0).await;
    assert_spoof_rejected(&run);
    // First frame + the deciding-frame recheck.
    assert_eq!(run.depth_calls, 2);
}

#[tokio::test]
async fn mean_mode_without_samples_still_averages_in_unseen_middle_photos() {
    // Documented residual risk of the default (0 samples): photos between the
    // checked first frame and a live deciding frame are not depth-checked.
    let run = authenticate(&[LiveImpostor, Photo, Photo, Photo, LiveImpostor], 0).await;
    assert_eq!(run.decision.outcome, AuthOutcome::Matched);
    assert_eq!(run.depth_calls, 2);
}

#[tokio::test]
async fn mean_mode_samples_reject_photo_frames_mid_sequence_with_a_live_deciding_frame() {
    let run = authenticate(&[LiveImpostor, Photo, Photo, Photo, LiveImpostor], 1).await;
    assert_spoof_rejected(&run);
    let run = authenticate(&[LiveImpostor, Photo, Photo, Photo, LiveImpostor], 3).await;
    assert_spoof_rejected(&run);
}

#[tokio::test]
async fn mean_mode_accepts_a_genuine_live_sequence_and_each_sample_costs_one_check() {
    let live = [LiveOwner; 5];
    let base = authenticate(&live, 0).await;
    assert_eq!(base.decision.outcome, AuthOutcome::Matched);
    assert_eq!(base.depth_calls, 2);

    let sampled = authenticate(&live, 3).await;
    assert_eq!(sampled.decision.outcome, AuthOutcome::Matched);
    assert_eq!(sampled.depth_calls, base.depth_calls + 3);
}

#[tokio::test]
async fn mean_mode_rejects_photos_alone() {
    let run = authenticate(&[Photo; 5], 3).await;
    assert_eq!(run.decision.outcome, AuthOutcome::LivenessRejected);
}

/// A frame captured before the `WaitForMatch` request must never be scored:
/// here the camera still holds an old photo frame when the request arrives.
#[tokio::test]
async fn wait_for_match_ignores_frames_captured_before_the_request() {
    // Index 0..=2 live owner (fresh); index 3 is a stale photo that sits in the
    // channel from before the request.
    let script = [LiveOwner, LiveOwner, LiveOwner, Photo];
    let (state, depth_calls, camera, _dir) = build_state(&script, 3, 0).await;
    let stale = frame(3);
    camera.send_replace(Some(stale));
    std::thread::sleep(std::time::Duration::from_millis(5));
    let wait = doormand::unlock_wait::WaitTiming::new();
    let feed = async {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        camera.send_replace(Some(frame(0)));
    };
    let (decision, ()) = tokio::join!(run_authenticate(&state, USER, None, Some(&wait)), feed);
    assert_eq!(decision.outcome, AuthOutcome::Matched);
    assert_eq!(decision.frames_seen, 3, "the stale photo frame was skipped");
    assert_eq!(depth_calls.load(Ordering::SeqCst), 2);
}

/// Control: without a request time the same stale frame IS scored first.
#[tokio::test]
async fn authenticate_without_a_request_time_still_scores_the_buffered_frame() {
    let script = [LiveOwner, LiveOwner, LiveOwner, Photo];
    let (state, _depth, camera, _dir) = build_state(&script, 3, 0).await;
    camera.send_replace(Some(frame(3)));
    let decision = run_authenticate(&state, USER, None, None).await;
    assert_eq!(decision.outcome, AuthOutcome::LivenessRejected);
}
