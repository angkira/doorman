// Use library modules (camera backend selected at compile time)
// GStreamer backend is default (PipeWire-integrated)
// Falls back to V4L2 if GStreamer not available
use doormand::camera;
use doormand::liveness;
mod dataset;
mod debug_stream;
mod frame_stream;
mod ipc;
mod lock_supervisor;
mod ml;
mod pipeline;
mod storage;
mod timing;

use anyhow::Result;
use doorman_shared::Config;
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
pub struct DaemonState {
    /// Latest captured frame, published by the single-owner camera producer
    /// thread (see `pipeline::camera_producer`). The `Camera` itself is owned
    /// solely by that thread and is never shared across threads; consumers
    /// (IPC enroll/auth/status/detect) read frames from this watch channel
    /// instead of touching the camera directly.
    pub latest_frame: tokio::sync::watch::Receiver<Option<pipeline::RawFrame>>,
    pub ml_pipeline: Arc<ml::MLPipeline>,
    pub storage: Arc<RwLock<storage::Storage>>,
    pub start_time: std::time::Instant,
    pub config: Arc<Config>,
    pub latest_detection: Arc<RwLock<Option<doorman_shared::DetectionInfo>>>,
    pub debug_broadcaster: Arc<debug_stream::DebugStreamBroadcaster>,
    pub frame_broadcaster: Option<Arc<frame_stream::FrameStreamBroadcaster>>,
    pub system_locked: Arc<RwLock<bool>>,
    /// Lock supervisor gate `locked && !idle`. `None` when the daemon is not
    /// lock-gated (`--user`/preview/dev mode). `WaitForMatch` only attempts
    /// recognition while this is `true`.
    pub armed: Option<tokio::sync::watch::Receiver<bool>>,
    /// Raw `LockedHint` of the tracked session. `None` when there is no lock
    /// supervisor or it found no session; the sudo profile then fails closed.
    pub locked: Option<tokio::sync::watch::Receiver<bool>>,
    /// Per-user suspension of the sudo profile (`[pam.sudo]`).
    pub sudo_guard: Arc<doormand::sudo_policy::SudoGuard>,
    /// On-demand camera capture leases. IPC operations that need live frames
    /// (enroll, `test`/PAM auth) increment this; the camera gate keeps the camera ON
    /// while it is > 0, so capture works even when the screen is unlocked and the
    /// pipeline is otherwise gated off. Released via `CaptureLease` (RAII).
    pub capture_requests: Arc<std::sync::atomic::AtomicUsize>,
}

// Safety: All fields are wrapped in Arc/RwLock making them thread-safe
unsafe impl Send for DaemonState {}
unsafe impl Sync for DaemonState {}

/// RAII lease that keeps the camera ON for an on-demand IPC capture (enroll/auth).
/// Increments `DaemonState::capture_requests` on creation; releases it on drop.
pub struct CaptureLease {
    counter: Arc<std::sync::atomic::AtomicUsize>,
}
impl CaptureLease {
    fn new(counter: Arc<std::sync::atomic::AtomicUsize>) -> Self {
        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Self { counter }
    }
}
impl Drop for CaptureLease {
    fn drop(&mut self) {
        self.counter
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl DaemonState {
    /// What the lock supervisor knows about the tracked session.
    pub fn presence(&self) -> doormand::sudo_policy::Presence {
        use doormand::sudo_policy::Presence;
        match &self.locked {
            None => Presence::Unavailable,
            Some(rx) if *rx.borrow() => Presence::Locked,
            Some(_) => Presence::Unlocked,
        }
    }

    /// Acquire an on-demand capture lease and wait (up to `timeout`) for the camera
    /// to produce a live frame. This turns the camera ON via the gate even when the
    /// screen is unlocked (enroll/preview/test). Returns a guard that releases the
    /// lease on drop, or an error if no frame arrives within the timeout (no camera).
    pub async fn acquire_camera(
        &self,
        timeout: std::time::Duration,
    ) -> Result<CaptureLease, String> {
        let lease = CaptureLease::new(self.capture_requests.clone());
        let mut rx = self.latest_frame.clone();
        let start = std::time::Instant::now();
        // The producer publishes `None` while the live camera is released and `Some`
        // once it has (re)opened and captured. Wait for a fresh `Some`.
        loop {
            if rx.borrow_and_update().is_some() {
                return Ok(lease);
            }
            if start.elapsed() >= timeout {
                return Err(
                    "Camera not available (no frames). Please connect a camera and try again."
                        .to_string(),
                );
            }
            let _ = tokio::time::timeout(std::time::Duration::from_millis(250), rx.changed()).await;
        }
    }
}

/// Subscriber-gated preview supervisor (production / system-daemon path).
///
/// Owns the camera frame `Receiver` for the lifetime of the daemon and performs
/// the frame fanout itself. It watches the frame + debug broadcasters' live
/// subscriber counts and runs the expensive detection + recognition ML pipeline
/// ONLY while at least one preview client (doorman-preview) is connected:
///   - first client connects  -> spin up detection+recognition, start
///     broadcasting JPEG frames + per-frame recognition metadata.
///   - last client disconnects -> abort the ML tasks, stop encoding/broadcasting,
///     drop the daemon back to ~0% CPU. The camera producer keeps running so PAM
///     auth always has fresh frames.
///
/// PAM auth (`ipc::handle_authenticate`) is entirely independent of this task —
/// it runs its own on-demand detection against `state.latest_frame` — so preview
/// being on or off never affects authentication.
async fn run_preview_supervisor(
    state: DaemonState,
    frame_broadcaster: Option<Arc<frame_stream::FrameStreamBroadcaster>>,
    debug_broadcaster: Arc<debug_stream::DebugStreamBroadcaster>,
    camera_rx: Arc<tokio::sync::Mutex<Option<tokio::sync::mpsc::Receiver<pipeline::RawFrame>>>>,
    // Lock-supervisor gate: `armed = LockedHint && !IdleHint`. The pipeline runs
    // when a preview client is connected OR the session is armed (screen locked,
    // display on). `None` in --user/dev mode (no lock-gating there).
    armed_rx: Option<tokio::sync::watch::Receiver<bool>>,
    // Stage B proactive-unlock plumbing, forwarded to the recognition pipeline.
    unlock_ctx: Option<pipeline::UnlockContext>,
) {
    use std::time::Instant;

    let mut camera_rx = camera_rx
        .lock()
        .await
        .take()
        .expect("camera_rx consumed exactly once by the preview supervisor");

    let processing_fps = state.config.daemon.processing_fps.max(1);
    let detection_interval_ms = (1000 / processing_fps) as u128;
    let preview_interval_ms: u128 = 66; // ~15fps JPEG preview

    // Active-pipeline state. `None` => idle (no ML running).
    let mut active: Option<(
        tokio::sync::mpsc::Sender<pipeline::RawFrame>,
        Vec<tokio::task::JoinHandle<()>>,
    )> = None;

    let mut last_detection = Instant::now();
    let mut last_preview = Instant::now();

    info!("Preview supervisor started (idle; pipeline starts on preview connect or screen lock)");

    while let Some(raw_frame) = camera_rx.recv().await {
        // Combined live subscriber count across both preview sockets. A preview
        // client connects to BOTH (frames + metadata), but counting either keeps
        // us robust if only one is attached.
        let subscribers = frame_broadcaster
            .as_ref()
            .map(|b| b.receiver_count())
            .unwrap_or(0)
            + debug_broadcaster.receiver_count();

        // Lock-supervisor gate (system mode only). The pipeline must run if a
        // preview client is connected OR the session is armed (locked && !idle),
        // and stop only when BOTH are false. The preview gate is unchanged; this
        // ORs the lock gate on top of it.
        let armed = armed_rx.as_ref().map(|rx| *rx.borrow()).unwrap_or(false);
        let want_pipeline = subscribers > 0 || armed;

        match (want_pipeline, active.is_some()) {
            // Either gate just opened -> start the ML pipeline.
            (true, false) => {
                info!(
                    "Pipeline gate opened (preview subscribers={}, armed={}) -> starting detection/recognition pipeline",
                    subscribers, armed
                );
                let (detection_tx, detection_rx) =
                    tokio::sync::mpsc::channel::<pipeline::RawFrame>(10);
                let handles =
                    pipeline::start_detection_recognition(&state, detection_rx, unlock_ctx.clone());
                active = Some((detection_tx, handles));
                last_detection =
                    Instant::now() - std::time::Duration::from_millis(detection_interval_ms as u64);
                last_preview =
                    Instant::now() - std::time::Duration::from_millis(preview_interval_ms as u64);
            }
            // Both gates closed -> tear the ML pipeline down (back to ~0% CPU).
            (false, true) => {
                info!("Pipeline gate closed (no preview clients, disarmed) -> stopping detection/recognition pipeline (idle)");
                if let Some((detection_tx, handles)) = active.take() {
                    drop(detection_tx); // closes detection channel
                    for h in handles {
                        h.abort();
                    }
                }
            }
            _ => {}
        }

        // While active, fan the frame out: broadcast JPEG previews (~15fps) and
        // feed the detection stage at processing_fps. While idle we simply drop
        // the frame here, so the camera channel never backs up.
        if let Some((detection_tx, _)) = active.as_ref() {
            if let Some(ref broadcaster) = frame_broadcaster {
                if last_preview.elapsed().as_millis() >= preview_interval_ms {
                    let broadcaster = broadcaster.clone();
                    let image = raw_frame.image.clone();
                    tokio::task::spawn_blocking(move || {
                        if let Err(e) = broadcaster.broadcast_frame(&image) {
                            tracing::debug!("Failed to broadcast preview frame: {}", e);
                        }
                    });
                    last_preview = Instant::now();
                }
            }

            if last_detection.elapsed().as_millis() >= detection_interval_ms {
                let _ = detection_tx.try_send(raw_frame);
                last_detection = Instant::now();
            }
        }
    }

    info!("Preview supervisor stopped (camera channel closed)");
}

#[tokio::main]
async fn main() -> Result<()> {
    // Check for --user, --preview, --start-unlocked, --config and --video-file flags
    let user_mode = std::env::args().any(|arg| arg == "--user");
    let preview_mode = std::env::args().any(|arg| arg == "--preview");
    // Testing override: keep the session unlocked at startup (no auto-lock loop).
    let start_unlocked = std::env::args().any(|arg| arg == "--start-unlocked");
    let config_file = std::env::args()
        .skip_while(|arg| arg != "--config")
        .nth(1)
        .map(std::path::PathBuf::from);
    let video_file = std::env::args()
        .skip_while(|arg| arg != "--video-file")
        .nth(1)
        .map(std::path::PathBuf::from);
    // Inference device override: `--device <cpu|coreml|ane|gpu|auto>`. Takes
    // precedence over the config's `ml.device`. On Apple Silicon with the
    // `backend-ort-coreml` feature compiled, `auto`/`coreml` selects the CoreML
    // EP (Neural Engine + GPU + CPU fallback); otherwise the daemon stays on CPU.
    let device_override = std::env::args().skip_while(|arg| arg != "--device").nth(1);

    // Load configuration from --config, DOORMAN_CONFIG env var, or standard locations
    let mut config = if let Some(config_path) = config_file {
        info!("Loading config from --config: {:?}", config_path);
        Config::load_from(&config_path).unwrap_or_else(|e| {
            warn!("Failed to load config from --config: {}", e);
            Config::default()
        })
    } else if let Ok(config_path) = std::env::var("DOORMAN_CONFIG") {
        info!("Loading config from DOORMAN_CONFIG: {}", config_path);
        Config::load_from(&std::path::PathBuf::from(config_path)).unwrap_or_else(|e| {
            warn!("Failed to load config from DOORMAN_CONFIG: {}", e);
            Config::default()
        })
    } else {
        Config::load().unwrap_or_default()
    };
    if preview_mode {
        config.daemon.preview_mode = true;
        config.daemon.debug_mode = true; // Preview mode implies debug mode
    }

    // Resolve the inference device.
    //
    // Precedence: `--device` flag > `ml.device` config > platform default.
    // Platform default: on macOS built WITH the CoreML feature, an unset device
    // (still the config default "cpu") is promoted to "coreml" so Apple Silicon
    // uses the Neural Engine/GPU out of the box; everywhere else it stays "cpu".
    if let Some(dev) = device_override {
        info!("Device override via --device: {}", dev);
        config.ml.device = dev;
    } else {
        #[cfg(all(target_os = "macos", feature = "backend-ort-coreml"))]
        {
            if config.ml.device == "cpu" {
                info!("Apple Silicon + CoreML feature: defaulting ml.device to 'coreml'");
                config.ml.device = "coreml".to_string();
            }
        }
    }

    // Resolve the initial system lock state.
    //
    // Real deployment: daemon is an always-on system service that should boot
    // *locked* and only clear the lock when a face is recognized
    // (config.daemon.start_locked defaults to true).
    //
    // Dev/preview: --start-unlocked, or debug/preview mode, keeps the session
    // unlocked so the pipeline streams frames without auto-locking the desktop.
    let initial_locked = config.daemon.start_locked && !start_unlocked && !config.daemon.debug_mode;

    if user_mode {
        config.daemon.user_mode = true;

        // Adjust paths for user mode (use XDG directories)
        if let Some(runtime_dir) = std::env::var_os("XDG_RUNTIME_DIR") {
            let runtime_path = std::path::PathBuf::from(runtime_dir);
            config.daemon.socket_path = runtime_path
                .join("doorman.sock")
                .to_string_lossy()
                .to_string();
            config.daemon.debug_socket = runtime_path
                .join("doorman-debug.sock")
                .to_string_lossy()
                .to_string();
            config.daemon.frame_socket = runtime_path
                .join("doorman-frames.sock")
                .to_string_lossy()
                .to_string();
        }

        if let Some(data_dir) = std::env::var_os("XDG_DATA_HOME") {
            config.daemon.data_dir = std::path::PathBuf::from(data_dir)
                .join("doorman")
                .to_string_lossy()
                .to_string();
        } else if let Some(home) = std::env::var_os("HOME") {
            config.daemon.data_dir = std::path::PathBuf::from(home)
                .join(".local/share/doorman")
                .to_string_lossy()
                .to_string();
        }

        // Also update models directory for user mode
        config.ml.models_dir = std::path::PathBuf::from(&config.daemon.data_dir)
            .join("models")
            .to_string_lossy()
            .to_string();
    }

    // Initialize logging.
    //
    // Include the `ort` target at info by default so ort's own EP registration
    // logs ("Successfully registered ROCMExecutionProvider" / CPU fallback) are
    // visible — otherwise an operator can't tell a GPU run from a silent CPU
    // fallback. Still overridable via RUST_LOG.
    let log_level = std::env::var("RUST_LOG").unwrap_or_else(|_| {
        format!(
            "doormand={},ort=info,{}=info",
            config.daemon.log_level,
            doormand::sudo_policy::AUDIT_TARGET
        )
    });

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_env("RUST_LOG")
                .unwrap_or_else(|_| log_level.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("doorman daemon starting...");
    info!(
        "Mode: {}",
        if config.daemon.user_mode {
            "user service"
        } else {
            "system service (root)"
        }
    );
    info!("Socket: {}", config.daemon.socket_path);
    info!("Data dir: {}", config.daemon.data_dir);
    info!(
        "Dataset capture (effective): logging_enabled={} keep_raw_frames={} raw_frame_bundle_size={} max_capture_packs_per_day={} max_attempt_records_per_day={} retention_days={} (retention is advisory: purge manually with `doorman dataset purge`)",
        config.dataset.logging_enabled,
        config.dataset.keep_raw_frames,
        config.dataset.raw_frame_bundle_size,
        config.dataset.max_capture_packs_per_day,
        config.dataset.max_attempt_records_per_day,
        config.dataset.retention_days,
    );
    info!("ML backend: {}", config.ml.backend);
    info!("Device: {}", config.ml.device);
    if config.daemon.debug_mode {
        warn!("DEBUG MODE ENABLED - processing frames even when unlocked");
    }
    info!(
        "Initial lock state: {} (PAM authenticate always runs an on-demand face check regardless)",
        if initial_locked { "LOCKED" } else { "unlocked" }
    );

    // Initialize components
    // Ensure data directory exists
    std::fs::create_dir_all(&config.daemon.data_dir)?;

    info!("Initializing ML pipeline...");
    let ml_pipeline = match ml::MLPipeline::new(&config).await {
        Ok(pipeline) => {
            info!("ML pipeline initialized, warming up models...");
            // Warmup models by running dummy inference
            // This ensures models are compiled/loaded before camera starts
            if let Err(e) = pipeline.warmup().await {
                warn!("Model warmup failed: {}. First inference may be slow.", e);
            } else {
                info!("✓ Models warmed up and ready");
            }
            Arc::new(pipeline)
        }
        Err(e) => {
            error!("Failed to initialize ML pipeline: {}", e);
            warn!(
                "Daemon will start but face recognition will not work until models are available"
            );
            Arc::new(ml::MLPipeline::dummy(&config))
        }
    };

    info!("Initializing storage...");
    let storage = Arc::new(RwLock::new(
        storage::Storage::new_with_dir(&config.daemon.data_dir).await?,
    ));

    info!("Initializing camera...");
    // The camera is OPENED and OWNED entirely by the producer thread (see
    // `pipeline::camera_producer`). It must be opened on the same thread that
    // captures from it — the V4L2 stream's mmap buffers are bound to the opening
    // thread's fd context, and capturing (VIDIOC_DQBUF) from a different thread
    // is undefined behavior. We therefore only describe the source here; the
    // producer thread does the actual `open_stream()` + capture loop.
    let camera_source = if let Some(ref video_path) = video_file {
        info!("Using video file as camera source: {:?}", video_path);
        pipeline::CameraSource::VideoFile {
            path: video_path.clone(),
            width: config.camera.width,
            height: config.camera.height,
            fps: config.camera.fps,
            loop_playback: true,
        }
    } else {
        pipeline::CameraSource::Config
    };

    // Spawn the single-owner camera producer thread. It publishes the latest
    // frame to `latest_frame_rx` (for IPC consumers) and forwards `RawFrame`s to
    // the pipeline via `camera_rx`.
    let (latest_frame_tx, latest_frame_rx) =
        tokio::sync::watch::channel::<Option<pipeline::RawFrame>>(None);
    let (camera_tx, camera_rx) = tokio::sync::mpsc::channel::<pipeline::RawFrame>(30);
    // Camera capture gate. The producer opens/holds the live camera only while this
    // is `true`. Initial value = `always_track` (super-security: camera on from boot);
    // otherwise the gate task / mode setup below drives it from armed + preview.
    let (capture_active_tx, capture_active_rx) =
        tokio::sync::watch::channel::<bool>(config.camera.always_track);
    // On-demand capture leases: IPC ops (enroll / `test` / PAM auth) bump this so the
    // gate keeps the camera ON even while the screen is unlocked.
    let capture_requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    pipeline::spawn_camera_producer(
        camera_source,
        Arc::new(config.clone()),
        latest_frame_tx,
        camera_tx,
        capture_active_rx,
    );
    let camera_rx = Arc::new(tokio::sync::Mutex::new(Some(camera_rx)));

    info!("Initializing debug stream broadcaster...");
    let debug_broadcaster = Arc::new(debug_stream::DebugStreamBroadcaster::new(100));

    // Frame broadcaster is ALWAYS created now (system + preview modes alike).
    //
    // Binding the frame/debug sockets is cheap (a listener + an idle broadcast
    // channel with zero subscribers is ~free), so the production system daemon
    // exposes them too. The EXPENSIVE work — detection + recognition ML
    // inference — is started on-demand by the subscriber-gated supervisor below,
    // only while at least one preview client is connected. With nobody watching,
    // these broadcasters sit idle and the daemon stays at ~0% CPU.
    info!("Initializing frame stream broadcaster...");
    let frame_broadcaster = Some(Arc::new(frame_stream::FrameStreamBroadcaster::new(30))); // Buffer 30 frames

    // Lock supervisor (SYSTEM-daemon mode only). Connects to logind on the
    // system bus, resolves the active graphical seat0 session, and exposes
    // `armed = LockedHint && !IdleHint` plus a lock-generation counter and an
    // `Unlocker` handle. Robust by construction: any failure (no bus, no
    // session) yields `armed=false` and `unlocker=None` without crashing.
    //
    // We skip it in `--user`/dev mode: that path targets a developer's own
    // session and has no need to lock-gate or proactively unlock. Spawned before
    // the state so `WaitForMatch` can share the `armed` gate.
    let lock_sup = if config.daemon.preview_mode || config.daemon.user_mode {
        None
    } else {
        Some(lock_supervisor::spawn().await)
    };

    let state = DaemonState {
        latest_frame: latest_frame_rx,
        ml_pipeline,
        storage,
        start_time: std::time::Instant::now(),
        config: Arc::new(config.clone()),
        latest_detection: Arc::new(RwLock::new(None)),
        debug_broadcaster: debug_broadcaster.clone(),
        frame_broadcaster: frame_broadcaster.clone(),
        system_locked: Arc::new(RwLock::new(initial_locked)),
        armed: lock_sup.as_ref().map(|sup| sup.armed_rx.clone()),
        locked: lock_sup
            .as_ref()
            .filter(|sup| sup.unlocker.is_some())
            .map(|sup| sup.locked_rx.clone()),
        sudo_guard: Arc::new(doormand::sudo_policy::SudoGuard::new(
            config.pam.sudo.max_failures,
            std::time::Duration::from_secs(config.pam.sudo.suspend_secs),
        )),
        capture_requests: capture_requests.clone(),
    };

    // Setup signal handlers for graceful shutdown
    let mut signals = signal_hook_tokio::Signals::new(&[
        signal_hook::consts::SIGTERM,
        signal_hook::consts::SIGINT,
    ])?;

    let signal_handle = signals.handle();
    let signal_task = tokio::spawn(async move {
        use tokio_util::sync::CancellationToken;
        let token = CancellationToken::new();

        while let Some(signal) = signals.next().await {
            match signal {
                signal_hook::consts::SIGTERM | signal_hook::consts::SIGINT => {
                    info!("Received shutdown signal");
                    token.cancel();
                    break;
                }
                _ => {}
            }
        }
    });

    // Start the debug stream server. The production SYSTEM daemon now binds it
    // too: its socket lives under the systemd `RuntimeDirectory=doorman`
    // (`/run/doorman/`, writable via `ReadWritePaths=/run`), so the old
    // "Read-only file system" failure (which came from a `/run/...`-root path
    // outside the RuntimeDirectory) does not apply. Binding a listener + an idle
    // broadcast channel is ~free; the costly ML pipeline is gated separately by
    // the subscriber supervisor below. `--user` mode paths the socket under XDG.
    let debug_server_task = {
        let debug_socket_path = config.daemon.debug_socket.clone();
        let debug_bcast = debug_broadcaster.clone();
        let debug_ipc = config.ipc.clone();
        info!("Starting debug stream server on {}", debug_socket_path);
        Some(tokio::spawn(async move {
            if let Err(e) =
                debug_stream::run_debug_server(debug_socket_path, debug_bcast, debug_ipc).await
            {
                error!("Debug stream server error: {}", e);
            }
        }))
    };

    // Start the frame stream server (always bound — see frame_broadcaster note).
    let frame_server_task = if let Some(ref frame_bcast) = frame_broadcaster {
        let frame_socket_path = config.daemon.frame_socket.clone();
        let frame_bcast_clone = frame_bcast.clone();
        let frame_ipc = config.ipc.clone();
        info!("Starting frame stream server on {}", frame_socket_path);
        Some(tokio::spawn(async move {
            if let Err(e) =
                frame_stream::run_frame_server(frame_socket_path, frame_bcast_clone, frame_ipc)
                    .await
            {
                error!("Frame stream server error: {}", e);
            }
        }))
    } else {
        None
    };

    // SUBSCRIBER-GATED detection/recognition pipeline.
    //
    // The detection_pipeline (YuNet) + recognition_pipeline (embedding) run ML
    // inference on EVERY frame and exist solely to feed the preview overlay
    // (doorman-preview: bbox + recognized user + green/red box + confidence) via
    // the frame & debug streams. They are NOT needed for PAM auth:
    // `ipc::handle_authenticate` reads fresh frames from `state.latest_frame`
    // (published by the always-running camera producer) and runs its OWN
    // on-demand detect/liveness/embed. Running these pipelines 24/7 burned ~40%
    // CPU continuously while idle.
    //
    // So instead of running them unconditionally, a lightweight supervisor task
    // watches the frame + debug broadcasters' subscriber counts and:
    //   - starts the pipeline when the FIRST preview client connects, and
    //   - stops it (dropping back to ~0% CPU) when the LAST one disconnects.
    // The camera producer keeps running either way (it is cheap and gives PAM
    // auth fresh frames); only the ML work is gated.
    //
    // `--preview` keeps its eager always-on behavior: it sets preview_mode, which
    // here means "start the pipeline immediately and never tear it down", so the
    // developer flow is unchanged.
    let pipeline_handles = if config.daemon.preview_mode {
        info!("Starting pipeline (preview mode - detection+recognition overlay, always-on)");
        let _ = capture_active_tx.send(true); // preview => camera always capturing
        let camera_rx = camera_rx
            .lock()
            .await
            .take()
            .expect("camera_rx consumed exactly once by the pipeline");
        let handles = pipeline::start_pipeline(state.clone(), camera_rx).await;
        Some(handles)
    } else {
        info!("Detection/recognition pipeline idle (production mode - starts on preview connect, auth runs on-demand detection)");

        let (armed_rx, locked_rx, unlock_ctx) = if let Some(sup) = lock_sup.as_ref() {
            // Build the Stage-B unlock context only when proactive unlock is
            // enabled AND a session was actually resolved (so we have an Unlocker).
            // When disabled, the pipeline is still lock-GATED via `armed_rx`; it
            // just never calls Unlock (PAM remains the unlock path).
            let unlock_ctx = if config.authentication.proactive_unlock {
                sup.unlocker
                    .clone()
                    .map(|unlocker| pipeline::UnlockContext {
                        unlocker,
                        armed_rx: sup.armed_rx.clone(),
                        lock_gen_rx: sup.lock_gen_rx.clone(),
                    })
            } else {
                None
            };
            (
                Some(sup.armed_rx.clone()),
                Some(sup.locked_rx.clone()),
                unlock_ctx,
            )
        } else {
            (None, None, None)
        };

        // Camera capture gate (system mode): drive `capture_active` from armed
        // (screen locked) OR a connected preview client OR `always_track`. Re-evaluated
        // on every armed change and every 300ms — cheap, and crucially runs even when
        // the camera is OFF (unlike the frame-driven pipeline supervisor), so it can
        // turn the camera back ON when you lock the screen. In --user/dev mode
        // (armed_rx=None) there is no lock gate, so capture stays on.
        if let (Some(gate_armed), Some(gate_locked)) = (armed_rx.clone(), locked_rx) {
            let cap_tx = capture_active_tx;
            let fb = frame_broadcaster.clone();
            let db = debug_broadcaster.clone();
            let always = config.camera.always_track;
            let prearm = config.camera.prearm_on_lock;
            let prearm_window =
                std::time::Duration::from_millis(u64::from(config.camera.prearm_window_ms));
            let reqs = capture_requests.clone();
            tokio::spawn(async move {
                let mut gate_armed = gate_armed;
                let mut gate_locked = gate_locked;
                let mut tick = tokio::time::interval(std::time::Duration::from_millis(200));
                let clock = tokio::time::Instant::now();
                let mut edge = camera::gate::LockEdge::default();
                loop {
                    let subs =
                        fb.as_ref().map(|b| b.receiver_count()).unwrap_or(0) + db.receiver_count();
                    // On-demand IPC leases (enroll/auth) force the camera on too.
                    let on_demand = reqs.load(std::sync::atomic::Ordering::SeqCst) > 0;
                    // armed = locked && !idle, so idle follows from the two hints.
                    let locked = *gate_locked.borrow();
                    let since_lock_edge = edge.update(locked, clock.elapsed());
                    let want = camera::gate::camera_wanted(camera::gate::GateInputs {
                        locked,
                        idle: locked && !*gate_armed.borrow(),
                        prearm_on_lock: prearm,
                        prearm_window,
                        since_lock_edge,
                        always_track: always,
                        stream_subscribers: subs,
                        capture_leases: usize::from(on_demand),
                    });
                    if cap_tx.send(want).is_err() {
                        break; // producer gone
                    }
                    tokio::select! {
                        r = gate_armed.changed() => { if r.is_err() { break; } }
                        r = gate_locked.changed() => { if r.is_err() { break; } }
                        _ = tick.tick() => {}
                    }
                }
            });
        } else {
            let _ = capture_active_tx.send(true);
        }

        let supervisor_state = state.clone();
        let frame_bcast_for_sup = frame_broadcaster.clone();
        let debug_bcast_for_sup = debug_broadcaster.clone();
        let camera_rx_for_sup = camera_rx.clone();
        tokio::spawn(async move {
            run_preview_supervisor(
                supervisor_state,
                frame_bcast_for_sup,
                debug_bcast_for_sup,
                camera_rx_for_sup,
                armed_rx,
                unlock_ctx,
            )
            .await;
        });
        None
    };

    // Start IPC server
    info!("Starting IPC server...");
    info!("doorman daemon running");

    // Run server (handles shutdown gracefully on error)
    tokio::select! {
        result = ipc::run_server(state.clone()) => {
            match result {
                Ok(()) => info!("IPC server shut down normally"),
                Err(e) => error!("IPC server error: {}", e),
            }
        }
        _ = signal_task => {
            info!("Shutdown signal received");
        }
        _ = async {
            if let Some(handles) = pipeline_handles {
                // Wait for any pipeline task to complete
                let _ = futures::future::select_all(handles).await;
            } else {
                std::future::pending::<()>().await;
            }
        } => {
            info!("Pipeline stopped");
        }
        _ = async {
            if let Some(task) = debug_server_task {
                let _ = task.await;
            } else {
                std::future::pending::<()>().await;
            }
        } => {
            info!("Debug stream server stopped");
        }
        _ = async {
            if let Some(task) = frame_server_task {
                task.await
            } else {
                std::future::pending::<()>().await;
                Ok(())
            }
        } => {
            info!("Frame server stopped");
        }
    }

    signal_handle.close();
    info!("doorman daemon stopped");

    // All application state has been dropped cleanly by the shutdown path above.
    // On the ROCm path, letting main() return lets the C++ static-destructor sweep
    // in _dl_fini run onnxruntime::Environment::~Environment() AFTER librocroller
    // (ROCm) has released its HIP handles, causing a use-after-free SIGSEGV at exit.
    // Exit directly to skip those cross-library global dtors — the kernel reclaims
    // all fds/sockets/memory anyway. (CPU ORT doesn't hit this, but the call is safe
    // unconditionally.)
    std::process::exit(0);
}
