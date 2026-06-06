// Use library modules (camera backend selected at compile time)
// GStreamer backend is default (PipeWire-integrated)
// Falls back to V4L2 if GStreamer not available
use doormand::camera;
mod debug_stream;
mod frame_stream;
mod ipc;
mod ml;
mod pipeline;
mod storage;

use anyhow::Result;
use doorman_shared::Config;
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
pub struct DaemonState {
    pub camera: Arc<RwLock<Option<camera::Camera>>>,
    pub ml_pipeline: Arc<ml::MLPipeline>,
    pub storage: Arc<RwLock<storage::Storage>>,
    pub start_time: std::time::Instant,
    pub config: Arc<Config>,
    pub latest_detection: Arc<RwLock<Option<doorman_shared::DetectionInfo>>>,
    pub debug_broadcaster: Arc<debug_stream::DebugStreamBroadcaster>,
    pub frame_broadcaster: Option<Arc<frame_stream::FrameStreamBroadcaster>>,
    pub system_locked: Arc<RwLock<bool>>,
}

// Safety: All fields are wrapped in Arc/RwLock making them thread-safe
unsafe impl Send for DaemonState {}
unsafe impl Sync for DaemonState {}

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

    // Load configuration from --config, DOORMAN_CONFIG env var, or standard locations
    let mut config = if let Some(config_path) = config_file {
        info!("Loading config from --config: {:?}", config_path);
        Config::load_from(&config_path)
            .unwrap_or_else(|e| {
                warn!("Failed to load config from --config: {}", e);
                Config::default()
            })
    } else if let Ok(config_path) = std::env::var("DOORMAN_CONFIG") {
        info!("Loading config from DOORMAN_CONFIG: {}", config_path);
        Config::load_from(&std::path::PathBuf::from(config_path))
            .unwrap_or_else(|e| {
                warn!("Failed to load config from DOORMAN_CONFIG: {}", e);
                Config::default()
            })
    } else {
        Config::load().unwrap_or_default()
    };
    if preview_mode {
        config.daemon.preview_mode = true;
        config.daemon.debug_mode = true;  // Preview mode implies debug mode
    }

    // Resolve the initial system lock state.
    //
    // Real deployment: daemon is an always-on system service that should boot
    // *locked* and only clear the lock when a face is recognized
    // (config.daemon.start_locked defaults to true).
    //
    // Dev/preview: --start-unlocked, or debug/preview mode, keeps the session
    // unlocked so the pipeline streams frames without auto-locking the desktop.
    let initial_locked =
        config.daemon.start_locked && !start_unlocked && !config.daemon.debug_mode;

    if user_mode {
        config.daemon.user_mode = true;

        // Adjust paths for user mode (use XDG directories)
        if let Some(runtime_dir) = std::env::var_os("XDG_RUNTIME_DIR") {
            let runtime_path = std::path::PathBuf::from(runtime_dir);
            config.daemon.socket_path = runtime_path.join("doorman.sock").to_string_lossy().to_string();
            config.daemon.debug_socket = runtime_path.join("doorman-debug.sock").to_string_lossy().to_string();
            config.daemon.frame_socket = runtime_path.join("doorman-frames.sock").to_string_lossy().to_string();
        }

        if let Some(data_dir) = std::env::var_os("XDG_DATA_HOME") {
            config.daemon.data_dir = std::path::PathBuf::from(data_dir).join("doorman").to_string_lossy().to_string();
        } else if let Some(home) = std::env::var_os("HOME") {
            config.daemon.data_dir = std::path::PathBuf::from(home).join(".local/share/doorman").to_string_lossy().to_string();
        }

        // Also update models directory for user mode
        config.ml.models_dir = std::path::PathBuf::from(&config.daemon.data_dir).join("models").to_string_lossy().to_string();
    }

    // Initialize logging
    let log_level = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| format!("doormand={}", config.daemon.log_level));

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_env("RUST_LOG")
                .unwrap_or_else(|_| log_level.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("doorman daemon starting...");
    info!("Mode: {}", if config.daemon.user_mode { "user service" } else { "system service (root)" });
    info!("Socket: {}", config.daemon.socket_path);
    info!("Data dir: {}", config.daemon.data_dir);
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
        },
        Err(e) => {
            error!("Failed to initialize ML pipeline: {}", e);
            warn!(
                "Daemon will start but face recognition will not work until models are available"
            );
            Arc::new(ml::MLPipeline::dummy(&config))
        }
    };

    info!("Initializing storage...");
    let storage = Arc::new(RwLock::new(storage::Storage::new_with_dir(&config.daemon.data_dir).await?));

    info!("Initializing camera...");
    let camera = if let Some(ref video_path) = video_file {
        info!("Using video file as camera source: {:?}", video_path);
        let width = config.camera.width;
        let height = config.camera.height;
        let fps = config.camera.fps;
        let cam = camera::Camera::from_video_file(video_path.clone(), width, height, fps, true);
        Arc::new(RwLock::new(Some(cam)))
    } else {
        match camera::Camera::new_with_config(&config).await {
            Ok(cam) => {
                info!("Camera initialized successfully");
                Arc::new(RwLock::new(Some(cam)))
            }
            Err(e) => {
                warn!("Camera not available at startup: {}", e);
                warn!("Camera will be initialized on-demand when needed");
                Arc::new(RwLock::new(None))
            }
        }
    };

    info!("Initializing debug stream broadcaster...");
    let debug_broadcaster = Arc::new(debug_stream::DebugStreamBroadcaster::new(100));

    // Initialize frame broadcaster only in preview mode
    let frame_broadcaster = if config.daemon.preview_mode {
        info!("Initializing frame stream broadcaster (preview mode enabled)...");
        Some(Arc::new(frame_stream::FrameStreamBroadcaster::new(30)))  // Buffer 30 frames
    } else {
        None
    };

    let state = DaemonState {
        camera,
        ml_pipeline,
        storage,
        start_time: std::time::Instant::now(),
        config: Arc::new(config.clone()),
        latest_detection: Arc::new(RwLock::new(None)),
        debug_broadcaster: debug_broadcaster.clone(),
        frame_broadcaster: frame_broadcaster.clone(),
        system_locked: Arc::new(RwLock::new(initial_locked)),
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

    // Start debug stream server (for preview clients)
    let debug_socket_path = config.daemon.debug_socket.clone();
    let debug_server_task = tokio::spawn(async move {
        if let Err(e) = debug_stream::run_debug_server(debug_socket_path, debug_broadcaster).await {
            error!("Debug stream server error: {}", e);
        }
    });

    // Start frame stream server (if preview mode enabled)
    let frame_server_task = if let Some(ref frame_bcast) = frame_broadcaster {
        let frame_socket_path = config.daemon.frame_socket.clone();
        let frame_bcast_clone = frame_bcast.clone();
        Some(tokio::spawn(async move {
            if let Err(e) = frame_stream::run_frame_server(frame_socket_path, frame_bcast_clone).await {
                error!("Frame stream server error: {}", e);
            }
        }))
    } else {
        None
    };

    // Start pipeline in production mode OR preview mode
    // Only disabled in debug mode when NOT in preview mode (for testing without camera)
    let pipeline_handles = if !config.daemon.debug_mode || config.daemon.preview_mode {
        if config.daemon.preview_mode {
            info!("Starting pipeline (preview mode - streaming frames)");
        } else {
            info!("Starting pipeline (production mode)");
        }
        let handles = pipeline::start_pipeline(state.clone()).await;
        Some(handles)
    } else {
        info!("Pipeline disabled (debug mode without preview)");
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
        _ = debug_server_task => {
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
    Ok(())
}
