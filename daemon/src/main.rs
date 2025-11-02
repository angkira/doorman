mod camera;
mod ipc;
mod ml;
mod storage;

use anyhow::Result;
use doorman_shared::Config;
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, error, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
pub struct DaemonState {
    pub camera: Arc<RwLock<Option<camera::Camera>>>,
    pub ml_pipeline: Arc<ml::MLPipeline>,
    pub storage: Arc<RwLock<storage::Storage>>,
    pub start_time: std::time::Instant,
    pub config: Arc<Config>,
}

// Safety: All fields are wrapped in Arc/RwLock making them thread-safe
unsafe impl Send for DaemonState {}
unsafe impl Sync for DaemonState {}

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let config = Config::load().unwrap_or_default();
    
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
    info!("ML backend: {}", config.ml.backend);
    info!("Device: {}", config.ml.device);

    // Initialize components
    info!("Initializing ML pipeline...");
    let ml_pipeline = match ml::MLPipeline::new(&config).await {
        Ok(pipeline) => Arc::new(pipeline),
        Err(e) => {
            error!("Failed to initialize ML pipeline: {}", e);
            warn!("Daemon will start but face recognition will not work until models are available");
            Arc::new(ml::MLPipeline::dummy(&config))
        }
    };

    info!("Initializing storage...");
    let storage = Arc::new(RwLock::new(storage::Storage::new().await?));

    info!("Initializing camera...");
    let camera = match camera::Camera::new_with_config(&config).await {
        Ok(cam) => {
            info!("Camera initialized successfully");
            Arc::new(RwLock::new(Some(cam)))
        }
        Err(e) => {
            warn!("Camera not available at startup: {}", e);
            warn!("Camera will be initialized on-demand when needed");
            Arc::new(RwLock::new(None))
        }
    };

    let state = DaemonState {
        camera,
        ml_pipeline,
        storage,
        start_time: std::time::Instant::now(),
        config: Arc::new(config.clone()),
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
    }

    signal_handle.close();
    info!("doorman daemon stopped");
    Ok(())
}

