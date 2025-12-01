/// Non-blocking producer-consumer pipeline for face recognition
/// 
/// Architecture:
/// Camera Producer → Frame Fanout → Detection Pipeline → Recognition Pipeline
///                        ↓
///                   Preview Clients

mod camera_producer;
mod frame_fanout;
mod detection_pipeline;
mod recognition_pipeline;
mod types;

pub use camera_producer::run_camera_producer;
pub use frame_fanout::run_frame_fanout;
pub use detection_pipeline::run_detection_pipeline;
pub use recognition_pipeline::run_recognition_pipeline;
pub use types::{RawFrame, DetectionResult};

use crate::DaemonState;
use tokio::sync::mpsc;
use tracing::info;

/// Start all pipeline stages
/// Returns handles to all spawned tasks for graceful shutdown
pub async fn start_pipeline(state: DaemonState) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::new();

    // Channel configuration from architecture spec
    let (camera_tx, camera_rx) = mpsc::channel::<RawFrame>(5);
    let (detection_tx, detection_rx) = mpsc::channel::<RawFrame>(2);
    let (result_tx, result_rx) = mpsc::channel::<DetectionResult>(10);

    info!("Starting pipeline stages...");

    // Stage 1: Camera Producer (owns camera exclusively)
    let camera_handle = tokio::spawn(run_camera_producer(
        state.camera.clone(),
        camera_tx,
        state.config.clone(),
    ));
    handles.push(camera_handle);

    // Stage 2: Frame Fanout (distributes to preview and detection)
    let fanout_handle = tokio::spawn(run_frame_fanout(
        camera_rx,
        state.frame_broadcaster.clone(),
        detection_tx,
        state.config.daemon.processing_fps,
    ));
    handles.push(fanout_handle);

    // Stage 3: Detection Pipeline (ML inference in blocking threads, sends results to debug preview)
    let detection_handle = tokio::spawn(run_detection_pipeline(
        detection_rx,
        result_tx,
        state.ml_pipeline.clone(),
        Some(state.debug_broadcaster.clone()),
    ));
    handles.push(detection_handle);

    // Stage 4: Recognition Pipeline (matching and unlock)
    let recognition_handle = tokio::spawn(run_recognition_pipeline(
        result_rx,
        state.storage.clone(),
        state.debug_broadcaster.clone(),
        state.system_locked.clone(),
        state.config.clone(),
        state.start_time,
    ));
    handles.push(recognition_handle);

    info!("All pipeline stages started");
    handles
}
