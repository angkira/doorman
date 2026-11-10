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
    /// The live owner with a poorer frame: cosine 0.35 to the template (above
    /// the 0.30 identity floor, below the 0.4 threshold).
    WeakOwner,
    /// A frame whose cosine to the template is 0.25 (below the floor).
    TooWeak,
    /// The live owner with a phone showing a face next to them: a second face
    /// of 36% of the main face's area is detected.
    OwnerWithPhoneFace,
    /// The live owner with a small face far in the background (4% of the area).
    OwnerWithBackgroundFace,
}

impl Scene {
    fn embedding(self) -> Vec<f32> {
        match self {
            Scene::LiveOwner
            | Scene::Photo
            | Scene::OwnerWithPhoneFace
            | Scene::OwnerWithBackgroundFace => OWNER_EMBEDDING.to_vec(),
            Scene::LiveImpostor => IMPOSTOR_EMBEDDING.to_vec(),
            Scene::WeakOwner => vec![0.35, (1.0f32 - 0.35 * 0.35).sqrt(), 0.0],
            Scene::TooWeak => vec![0.25, (1.0f32 - 0.25 * 0.25).sqrt(), 0.0],
        }
    }

    fn relief(self) -> f32 {
        match self {
            Scene::Photo => FLAT_RELIEF,
            _ => LIVE_RELIEF,
        }
    }
}

use Scene::{
    LiveImpostor, LiveOwner, OwnerWithBackgroundFace, OwnerWithPhoneFace, Photo, TooWeak, WeakOwner,
};

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

/// State of the (shadow-only) MiniFASNet model in the scripted backend.
#[derive(Clone, Copy, Debug, PartialEq)]
enum MiniMode {
    /// Loaded: real (+8) for live scenes, spoof (-8) for photos.
    Loaded,
    /// Not loaded: the backend answers `Ok(None)`.
    Missing,
    /// Loaded but the inference errors.
    Failing,
}

struct ScriptedBackend {
    mini: MiniMode,
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
        let second_face_side = match self.script[frame_index(image)] {
            Scene::OwnerWithPhoneFace => Some(0.3),
            Scene::OwnerWithBackgroundFace => Some(0.1),
            _ => None,
        };
        Ok(Some(Face {
            bbox: (0.25, 0.25, 0.5, 0.5),
            confidence: 0.99,
            frame_dimensions: (FRAME_SIDE, FRAME_SIDE),
            landmarks: None,
            other_faces: second_face_side
                .map(|side| Face {
                    bbox: (0.0, 0.0, side, side),
                    confidence: 0.9,
                    frame_dimensions: (FRAME_SIDE, FRAME_SIDE),
                    landmarks: None,
                    other_faces: Vec::new(),
                })
                .into_iter()
                .collect(),
        }))
    }

    async fn check_liveness(&self, _image: &DynamicImage, _face: &Face) -> anyhow::Result<bool> {
        Ok(true)
    }

    async fn minifasnet_logit_diff(
        &self,
        image: &DynamicImage,
        _face: &Face,
    ) -> anyhow::Result<Option<f32>> {
        match self.mini {
            MiniMode::Missing => Ok(None),
            MiniMode::Failing => Err(anyhow::anyhow!("scripted MiniFASNet failure")),
            MiniMode::Loaded => Ok(Some(match self.script[frame_index(image)] {
                Scene::Photo => -8.0,
                _ => 8.0,
            })),
        }
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
    // The liveness-binding tests below predate the per-frame identity floor
    // and use an impostor face on purpose: they run with the floor off.
    build_state_cfg(script, auth_frames, mean_mode_depth_samples, |c| {
        c.decision.frame_identity_floor = 0.0;
    })
    .await
}

async fn build_state_cfg(
    script: &[Scene],
    auth_frames: usize,
    mean_mode_depth_samples: u32,
    tweak: impl FnOnce(&mut Config),
) -> (
    DaemonState,
    Arc<AtomicUsize>,
    watch::Sender<Option<RawFrame>>,
    tempfile::TempDir,
) {
    build_state_mini(
        script,
        auth_frames,
        mean_mode_depth_samples,
        tweak,
        MiniMode::Loaded,
    )
    .await
}

async fn build_state_mini(
    script: &[Scene],
    auth_frames: usize,
    mean_mode_depth_samples: u32,
    tweak: impl FnOnce(&mut Config),
    mini: MiniMode,
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
    // Real time: keep attempts that wait for more frames short.
    config.authentication.timeout_secs = 1;
    tweak(&mut config);

    let (camera_tx, camera_rx) = watch::channel(Some(frame(0)));
    let depth_calls = Arc::new(AtomicUsize::new(0));
    let backend = Arc::new(ScriptedBackend {
        mini,
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
        face_budget: Arc::new(doormand::face_budget::FaceBudget::new(
            doormand::face_budget::FaceBudgetSettings::from_config(&config.ipc),
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

// ---------------------------------------------------------------------------
// Failure budget (`[ipc] face_*`) through the real decision path
// ---------------------------------------------------------------------------

use doormand::face_budget::{FaceBudget, FaceBudgetSettings};

fn shared_budget() -> Arc<FaceBudget> {
    Arc::new(FaceBudget::new(FaceBudgetSettings::from_config(
        &doorman_shared::IpcConfig::default(),
    )))
}

/// One real attempt against `script`, accounted in the shared `budget`.
async fn attempt_with(budget: &Arc<FaceBudget>, script: &[Scene]) -> (AuthDecision, usize, usize) {
    let (mut state, depth_calls, _camera, _dir) = build_state(script, script.len(), 0).await;
    state.face_budget = budget.clone();
    let decision = run_authenticate(&state, USER, None, None).await;
    (
        decision,
        depth_calls.load(Ordering::SeqCst),
        state.capture_requests.load(Ordering::SeqCst),
    )
}

#[tokio::test]
async fn five_judged_failures_suspend_face_auth_before_any_camera_or_model_work() {
    let budget = shared_budget();
    for n in 1..=5 {
        let (d, _, _) = attempt_with(&budget, &[LiveImpostor; 5]).await;
        assert!(
            doormand::face_budget::is_judged_failure(&d.outcome),
            "attempt {n}: {:?}",
            d.outcome
        );
    }
    // Even the owner's genuine face is refused while suspended: no frames are
    // read and no depth inference runs.
    let (d, depth, leases) = attempt_with(&budget, &[LiveOwner; 5]).await;
    assert!(
        matches!(d.outcome, AuthOutcome::Suspended { retry_after_s } if (1..=30).contains(&retry_after_s)),
        "{:?}",
        d.outcome
    );
    assert_eq!((d.frames_seen, depth, leases), (0, 0, 0));
    assert!(matches!(
        d.into_response(),
        doorman_shared::Response::Denied { reason } if reason.contains("face_auth_suspended")
    ));
}

#[tokio::test]
async fn four_failures_then_a_match_reset_the_budget() {
    let budget = shared_budget();
    for _ in 0..4 {
        attempt_with(&budget, &[LiveImpostor; 5]).await;
    }
    let (d, _, _) = attempt_with(&budget, &[LiveOwner; 5]).await;
    assert_eq!(d.outcome, AuthOutcome::Matched);
    // Four more failures do not suspend: the match wiped the earlier four.
    for _ in 0..4 {
        attempt_with(&budget, &[LiveImpostor; 5]).await;
    }
    let (d, _, _) = attempt_with(&budget, &[LiveOwner; 5]).await;
    assert_eq!(d.outcome, AuthOutcome::Matched);
}

#[tokio::test]
async fn a_disabled_budget_never_suspends() {
    let budget = Arc::new(FaceBudget::new(FaceBudgetSettings::from_config(
        &doorman_shared::IpcConfig {
            face_max_judged_failures: 0,
            ..Default::default()
        },
    )));
    for _ in 0..8 {
        attempt_with(&budget, &[LiveImpostor; 5]).await;
    }
    let (d, _, _) = attempt_with(&budget, &[LiveOwner; 5]).await;
    assert_eq!(d.outcome, AuthOutcome::Matched);
}

// ---------------------------------------------------------------------------
// Round 9a B-D: identity floor, multi-face gate, minimum frames, frame age
// ---------------------------------------------------------------------------

async fn run_cfg(
    script: &[Scene],
    auth_frames: usize,
    tweak: impl FnOnce(&mut Config),
) -> AuthDecision {
    let (state, _depth, _camera, _dir) = build_state_cfg(script, auth_frames, 0, tweak).await;
    run_authenticate(&state, USER, None, None).await
}

/// The mix the floor closes: a live other face on the frames that carry the
/// liveness checks, owner photos in between. Without the floor it is ACCEPTED
/// (pinned below with the floor off), with the default floor it is rejected.
#[tokio::test]
async fn identity_floor_rejects_a_live_attacker_around_owner_photos() {
    let mix = [LiveImpostor, Photo, Photo, Photo, LiveImpostor];
    let before = run_cfg(&mix, 5, |c| c.decision.frame_identity_floor = 0.0).await;
    assert_eq!(
        before.outcome,
        AuthOutcome::Matched,
        "accepted without the floor"
    );
    let after = run_cfg(&mix, 5, |_| {}).await;
    assert_eq!(after.outcome, AuthOutcome::IdentityFloor);
    assert!(doormand::face_budget::is_judged_failure(&after.outcome));
}

#[tokio::test]
async fn identity_floor_leaves_genuine_sequences_unchanged() {
    let genuine = run_cfg(&[LiveOwner; 5], 5, |_| {}).await;
    assert_eq!(genuine.outcome, AuthOutcome::Matched);
    // Poorer genuine frames (cosine 0.35, above the floor) still average above
    // the threshold exactly as before.
    let seq = [LiveOwner, WeakOwner, LiveOwner, WeakOwner, LiveOwner];
    let with_floor = run_cfg(&seq, 5, |_| {}).await;
    let without = run_cfg(&seq, 5, |c| c.decision.frame_identity_floor = 0.0).await;
    assert_eq!(with_floor.outcome, AuthOutcome::Matched);
    assert_eq!(with_floor.outcome, without.outcome);
    assert_eq!(with_floor.similarity, without.similarity);
}

#[tokio::test]
async fn identity_floor_boundary_is_between_035_and_025() {
    let ok = run_cfg(
        &[LiveOwner, LiveOwner, LiveOwner, LiveOwner, WeakOwner],
        5,
        |_| {},
    )
    .await;
    assert_eq!(ok.outcome, AuthOutcome::Matched);
    let bad = run_cfg(
        &[LiveOwner, LiveOwner, LiveOwner, LiveOwner, TooWeak],
        5,
        |_| {},
    )
    .await;
    assert_eq!(bad.outcome, AuthOutcome::IdentityFloor);
}

#[tokio::test]
async fn multi_face_reject_drops_frames_with_a_second_qualifying_face() {
    let seq = [OwnerWithPhoneFace; 5];
    let d = run_cfg(&seq, 5, |_| {}).await;
    assert_eq!(d.outcome, AuthOutcome::AmbiguousFaces);
    assert!(doormand::face_budget::is_judged_failure(&d.outcome));
    // Policy off restores the original behaviour (the owner's frames win).
    let off = run_cfg(&seq, 5, |c| {
        c.recognition.multi_face_policy = doorman_shared::config::MultiFacePolicy::Off
    })
    .await;
    assert_eq!(off.outcome, AuthOutcome::Matched);
    // The main face is the largest here, so `largest` also accepts.
    let largest = run_cfg(&seq, 5, |c| {
        c.recognition.multi_face_policy = doorman_shared::config::MultiFacePolicy::Largest
    })
    .await;
    assert_eq!(largest.outcome, AuthOutcome::Matched);
}

#[tokio::test]
async fn a_small_background_face_does_not_trip_the_multi_face_gate() {
    let d = run_cfg(&[OwnerWithBackgroundFace; 5], 5, |_| {}).await;
    assert_eq!(d.outcome, AuthOutcome::Matched);
}

#[tokio::test]
async fn a_phone_face_on_some_frames_leaves_too_few_frames() {
    // Three clean frames then two with a phone face, 5 wanted, 3 needed ->
    // accepted; with 4 needed the two dropped frames make it ambiguous.
    let seq = [
        LiveOwner,
        LiveOwner,
        LiveOwner,
        OwnerWithPhoneFace,
        OwnerWithPhoneFace,
    ];
    let ok = run_cfg(&seq, 5, |c| c.authentication.min_decision_frames = 3).await;
    assert_eq!(ok.outcome, AuthOutcome::Matched);
    let bad = run_cfg(&seq, 5, |c| c.authentication.min_decision_frames = 4).await;
    assert_eq!(bad.outcome, AuthOutcome::AmbiguousFaces);
}

#[tokio::test]
async fn too_few_judged_good_frames_are_rejected_not_decided() {
    // Default minimum for 10 frames is 8. Seven genuine frames then silence.
    let seven = run_cfg(&[LiveOwner; 7], 10, |_| {}).await;
    assert_eq!(seven.outcome, AuthOutcome::InsufficientFrames);
    assert!(!doormand::face_budget::is_judged_failure(&seven.outcome));
    assert!(matches!(
        seven.into_response(),
        doorman_shared::Response::Failure { reason } if reason.contains("insufficient_frames")
    ));
    // A single frame must not decide, even though it is the owner.
    let one = run_cfg(&[LiveOwner], 10, |_| {}).await;
    assert_eq!(one.outcome, AuthOutcome::InsufficientFrames);
    // Eight are enough.
    let eight = run_cfg(&[LiveOwner; 8], 10, |_| {}).await;
    assert_eq!(eight.outcome, AuthOutcome::Matched);
    // The old behaviour is reachable with an explicit minimum of 1.
    let legacy = run_cfg(&[LiveOwner], 10, |c| {
        c.authentication.min_decision_frames = 1
    })
    .await;
    assert_eq!(legacy.outcome, AuthOutcome::Matched);
}

/// A frame captured long before it is taken up (a stalled feed) is not used.
#[tokio::test]
async fn a_stale_frame_is_skipped_by_the_frame_age_guard() {
    let script = [LiveOwner, LiveOwner, LiveOwner];
    let stale_by = std::time::Duration::from_secs(5);
    let old = std::time::Instant::now()
        .checked_sub(stale_by)
        .expect("monotonic clock far enough from its origin");

    // Guard on (default 1000 ms): the only frame is 5 s old -> nothing usable.
    let (state, _d, camera, _dir) = build_state_cfg(&script, 3, 0, |_| {}).await;
    let mut stale = frame(0);
    stale.timestamp = old;
    camera.send_replace(Some(stale));
    let d = run_authenticate(&state, USER, None, None).await;
    assert_eq!(d.outcome, AuthOutcome::NoFrames);

    // Control: guard off -> the same stale frame is scored and the owner matches.
    let (state, _d, camera, _dir) =
        build_state_cfg(&script, 3, 0, |c| c.camera.max_frame_age_ms = 0).await;
    let mut stale = frame(0);
    stale.timestamp = old;
    camera.send_replace(Some(stale));
    let d = run_authenticate(&state, USER, None, None).await;
    assert_eq!(d.outcome, AuthOutcome::Matched);
}

/// A stale frame followed by fresh ones: the stale one is skipped, the fresh
/// ones decide.
#[tokio::test]
async fn fresh_frames_after_a_stale_one_still_decide() {
    let script = [LiveOwner, LiveOwner, LiveOwner];
    let (state, _d, camera, _dir) = build_state_cfg(&script, 3, 0, |_| {}).await;
    let mut stale = frame(2);
    stale.timestamp = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_secs(5))
        .unwrap();
    camera.send_replace(Some(stale));
    let feed = async {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        camera.send_replace(Some(frame(0)));
    };
    let (d, ()) = tokio::join!(run_authenticate(&state, USER, None, None), feed);
    assert_eq!(d.outcome, AuthOutcome::Matched);
    assert_eq!(d.frames_seen, 3, "the stale frame was not counted");
}

// ---------------------------------------------------------------------------
// Round 9a F: raw frames of successful unlocks are not stored by default
// ---------------------------------------------------------------------------

async fn captures_after(script: &[Scene], capture_successful: bool) -> usize {
    let (state, _d, _c, dir) = build_state_cfg(script, script.len(), 0, |c| {
        c.dataset.logging_enabled = true;
        c.dataset.keep_raw_frames = true;
        c.dataset.capture_successful_unlocks = capture_successful;
    })
    .await;
    let outcome = run_authenticate(&state, USER, None, None).await.outcome;
    // The pack is written on a blocking task: give it time to appear (or not).
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let captures = dir.path().join("dataset").join("captures");
    let packs = std::fs::read_dir(&captures)
        .map(|days| {
            days.flatten()
                .map(|d| std::fs::read_dir(d.path()).map_or(0, |p| p.count()))
                .sum()
        })
        .unwrap_or(0);
    assert!(matches!(
        outcome,
        AuthOutcome::Matched | AuthOutcome::NotMatched | AuthOutcome::IdentityFloor
    ));
    packs
}

#[tokio::test]
async fn a_successful_unlock_stores_no_raw_frames_by_default() {
    assert_eq!(captures_after(&[LiveOwner; 5], false).await, 0);
}

#[tokio::test]
async fn a_successful_unlock_stores_raw_frames_only_when_enabled() {
    assert_eq!(captures_after(&[LiveOwner; 5], true).await, 1);
}

#[tokio::test]
async fn a_rejected_attempt_is_still_captured_by_default() {
    assert_eq!(captures_after(&[LiveImpostor; 5], false).await, 1);
}

// ---------------------------------------------------------------------------
// Round 9b-1: shadow PAD logging observes the attempt and never steers it
// ---------------------------------------------------------------------------

use crate::shadow_pad::{Fault, FAULT};

/// Clears the injected fault when the test ends, pass or fail.
struct FaultGuard;
impl FaultGuard {
    fn set(fault: Fault) -> Self {
        FAULT.with(|f| f.set(Some(fault)));
        FaultGuard
    }
}
impl Drop for FaultGuard {
    fn drop(&mut self) {
        FAULT.with(|f| f.set(None));
    }
}

/// What the decision exposes: outcome, similarity, frames, liveness verdict,
/// plus the depth inferences the attempt ran.
type Observed = (AuthDecision, usize);

/// One attempt with the shadow flag as given; returns the decision and the
/// shadow log lines written (after letting the detached task finish).
async fn shadow_attempt(
    script: &[Scene],
    samples: u32,
    n_of_m: bool,
    shadow: bool,
    mini: MiniMode,
) -> (Observed, Vec<serde_json::Value>) {
    let log_dir = tempfile::tempdir().expect("log dir");
    let log = log_dir.path().join("shadow.jsonl");
    let log_str = log.to_string_lossy().into_owned();
    let (state, depth_calls, _camera, _dir) = build_state_mini(
        script,
        script.len(),
        samples,
        |c| {
            c.decision.frame_identity_floor = 0.0;
            if n_of_m {
                c.decision.mode = DecisionMode::NOfM;
                c.decision.accept_n = 3;
                c.decision.window_m = 5;
            }
            c.liveness.shadow_pad_logging = shadow;
            c.telemetry.shadow_pad_log_path = log_str.clone();
        },
        mini,
    )
    .await;
    let decision = run_authenticate(&state, USER, None, None).await;
    // The record is written by a detached task: give it time, bounded.
    for _ in 0..100 {
        if log.exists()
            && std::fs::metadata(&log)
                .map(|m| m.len() > 0)
                .unwrap_or(false)
        {
            break;
        }
        if !shadow {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let lines = std::fs::read_to_string(&log)
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).expect("valid JSON line"))
        .collect();
    ((decision, depth_calls.load(Ordering::SeqCst)), lines)
}

fn sequences() -> Vec<(&'static str, Vec<Scene>)> {
    vec![
        ("genuine", vec![LiveOwner; 5]),
        ("photo", vec![Photo; 5]),
        ("impostor", vec![LiveImpostor; 5]),
        (
            "live-photo-live",
            vec![LiveImpostor, Photo, Photo, Photo, LiveImpostor],
        ),
        ("weak-owner", vec![WeakOwner; 5]),
        ("too-weak", vec![TooWeak; 5]),
        ("phone-next-to-face", vec![OwnerWithPhoneFace; 5]),
        ("background-face", vec![OwnerWithBackgroundFace; 5]),
        (
            "mixed",
            vec![LiveOwner, LiveOwner, WeakOwner, LiveOwner, LiveOwner],
        ),
    ]
}

#[tokio::test]
async fn shadow_logging_leaves_every_decision_unchanged() {
    for (name, script) in sequences() {
        for (samples, n_of_m) in [(0, false), (3, false), (0, true)] {
            let (off, off_lines) =
                shadow_attempt(&script, samples, n_of_m, false, MiniMode::Loaded).await;
            assert!(off_lines.is_empty(), "{name}: flag off writes nothing");
            let (on, on_lines) =
                shadow_attempt(&script, samples, n_of_m, true, MiniMode::Loaded).await;
            assert_eq!(on, off, "{name} samples={samples} n_of_m={n_of_m}");
            assert_eq!(on_lines.len(), 1, "{name}: exactly one record per attempt");
            let rec = &on_lines[0];
            assert_eq!(rec["schema_version"], 1);
            assert_eq!(rec["frames_used"], on.0.frames_used as u64);
            assert_eq!(rec["frames"]["seen"], on.0.frames_seen as u64);
        }
    }
}

#[tokio::test]
async fn decisions_survive_shadow_faults_and_a_broken_minifasnet() {
    // Accept, spoof-reject, swap-reject and identity-reject paths; the full
    // sequence list is covered by the on/off comparison above.
    let names = [
        "genuine",
        "photo",
        "live-photo-live",
        "impostor",
        "weak-owner",
    ];
    let combos = [
        (MiniMode::Loaded, None),
        (MiniMode::Loaded, Some(Fault::Hook)),
        (MiniMode::Loaded, Some(Fault::Finish)),
        (MiniMode::Loaded, Some(Fault::Emit)),
        (MiniMode::Missing, None),
        (MiniMode::Failing, None),
    ];
    for (name, script) in sequences().into_iter().filter(|(n, _)| names.contains(n)) {
        let (baseline, _) = shadow_attempt(&script, 3, false, false, MiniMode::Loaded).await;
        for (mini, fault) in combos {
            let _guard = fault.map(FaultGuard::set);
            let (observed, _lines) = shadow_attempt(&script, 3, false, true, mini).await;
            assert_eq!(
                observed, baseline,
                "{name}: mini={mini:?} fault={fault:?} changed the decision"
            );
        }
    }
}

#[tokio::test]
async fn a_shadow_fault_leaves_no_record_but_the_decision_stands() {
    for fault in [Fault::Hook, Fault::Finish, Fault::Emit] {
        let _guard = FaultGuard::set(fault);
        let ((decision, _), lines) =
            shadow_attempt(&[LiveOwner; 5], 0, false, true, MiniMode::Loaded).await;
        assert_eq!(decision.outcome, AuthOutcome::Matched, "{fault:?}");
        assert!(lines.is_empty(), "{fault:?}: no partial record");
    }
}

#[tokio::test]
async fn the_record_of_a_genuine_attempt_carries_the_calibration_aggregates() {
    let ((decision, _), lines) =
        shadow_attempt(&[LiveOwner; 5], 0, false, true, MiniMode::Loaded).await;
    assert_eq!(decision.outcome, AuthOutcome::Matched);
    let rec = &lines[0];
    assert_eq!(rec["decision"], "recognized");
    assert_eq!(rec["reason_code"], "matched");
    assert_eq!(rec["decision_mode"], "mean");
    assert_eq!(rec["frames"]["judged"], 5);
    assert_eq!(rec["frames"]["quality_rejected"], 0);
    let sim = &rec["similarity"]["per_frame"];
    assert_eq!(sim["n"], 5);
    assert_eq!(sim["min"].as_f64().unwrap(), 1.0);
    assert!((rec["similarity"]["margin"].as_f64().unwrap() - 0.6).abs() < 1e-5);
    // First frame + deciding frame were depth-checked, both at LIVE_RELIEF.
    let depth = &rec["depth"];
    assert_eq!(depth["relief"]["n"], 2);
    let margin = depth["margin_min"].as_f64().unwrap();
    assert!((margin - f64::from(LIVE_RELIEF - RELIEF_THRESHOLD)).abs() < 1e-5);
    assert_eq!(rec["minifasnet"]["state"], "scored");
    assert_eq!(rec["minifasnet"]["logit_diff"].as_f64().unwrap(), 8.0);
    let q = &rec["quality"];
    assert!((q["face_area_frac"].as_f64().unwrap() - 0.25).abs() < 1e-6);
    assert!(q["yaw_proxy"].is_null(), "the mock has no landmarks");
    assert!(q["sharpness"].is_number() && q["brightness"].is_number());
}

#[tokio::test]
async fn a_photo_is_recorded_with_its_flat_relief_and_spoof_score() {
    let ((decision, _), lines) =
        shadow_attempt(&[Photo; 5], 0, false, true, MiniMode::Loaded).await;
    assert_eq!(decision.outcome, AuthOutcome::LivenessRejected);
    let rec = &lines[0];
    assert_eq!(rec["reason_code"], "liveness_rejected");
    let margin = rec["depth"]["margin_min"].as_f64().unwrap();
    assert!(margin < 0.0, "below the threshold: {margin}");
    assert_eq!(rec["minifasnet"]["logit_diff"].as_f64().unwrap(), -8.0);
}

#[tokio::test]
async fn missing_and_failing_minifasnet_are_recorded_as_such() {
    for (mini, state) in [
        (MiniMode::Missing, "model_not_loaded"),
        (MiniMode::Failing, "error"),
    ] {
        let ((decision, _), lines) = shadow_attempt(&[LiveOwner; 5], 0, false, true, mini).await;
        assert_eq!(decision.outcome, AuthOutcome::Matched, "{mini:?}");
        assert_eq!(lines[0]["minifasnet"]["state"], state);
        assert!(lines[0]["minifasnet"]["logit_diff"].is_null());
    }
}

#[tokio::test]
async fn the_record_has_no_names_frames_or_embeddings() {
    fn keys(value: &serde_json::Value, out: &mut Vec<String>) {
        if let Some(map) = value.as_object() {
            for (k, v) in map {
                out.push(k.clone());
                keys(v, out);
            }
        }
    }
    let (_, lines) = shadow_attempt(&[LiveOwner; 5], 0, false, true, MiniMode::Loaded).await;
    let mut names = Vec::new();
    keys(&lines[0], &mut names);
    for key in &names {
        // `embedding_error` is a frame counter, not data.
        let k = key.replace("embedding_error", "");
        for forbidden in [
            "embedding",
            "image",
            "pixel",
            "user",
            "uid",
            "name",
            "path",
            "frame_data",
        ] {
            assert!(!k.contains(forbidden), "key {key} looks like {forbidden}");
        }
    }
    let values = serde_json::to_string(&lines[0]).unwrap();
    assert!(
        !values.contains(USER),
        "the user name is not in the record: {values}"
    );
}

#[tokio::test]
async fn ambiguous_frames_and_their_reason_are_counted() {
    let ((decision, _), lines) =
        shadow_attempt(&[OwnerWithPhoneFace; 5], 0, false, true, MiniMode::Loaded).await;
    assert_eq!(decision.outcome, AuthOutcome::AmbiguousFaces);
    assert_eq!(lines[0]["reason_code"], "ambiguous_faces");
    assert_eq!(lines[0]["frames"]["ambiguous_faces"], 5);
    assert_eq!(lines[0]["frames"]["judged"], 0);
}

#[cfg(unix)]
#[tokio::test]
async fn the_shadow_log_is_not_world_readable() {
    use std::os::unix::fs::PermissionsExt;
    let log_dir = tempfile::tempdir().unwrap();
    let log = log_dir.path().join("shadow.jsonl");
    let log_str = log.to_string_lossy().into_owned();
    let (state, _d, _c, _dir) = build_state_mini(
        &[LiveOwner; 5],
        5,
        0,
        |c| {
            c.liveness.shadow_pad_logging = true;
            c.telemetry.shadow_pad_log_path = log_str.clone();
        },
        MiniMode::Loaded,
    )
    .await;
    run_authenticate(&state, USER, None, None).await;
    for _ in 0..100 {
        if log.exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let mode = std::fs::metadata(&log).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode & 0o007, 0, "no access for others: {mode:o}");
}
