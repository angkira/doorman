use super::types::RawFrame;
use crate::camera::Camera;
use doorman_shared::Config;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, warn};

/// Describes how the producer thread should open the camera.
///
/// The camera MUST be opened on the same thread that captures from it: the V4L2
/// backend's `open_stream()` sets up mmap buffers bound to the calling thread's
/// fd context, and calling `frame()` (VIDIOC_DQBUF) from a different thread is
/// undefined behavior. Opening on the main tokio thread and capturing on the
/// producer thread corrupted the runtime (the previous bug). So the producer
/// thread opens the camera itself from this description.
pub enum CameraSource {
    /// Open the live camera using the daemon config.
    Config,
    /// Replay a video file (used for tests / `--video-file`).
    VideoFile {
        path: std::path::PathBuf,
        width: u32,
        height: u32,
        fps: u32,
        loop_playback: bool,
    },
}

/// Spawn the single-owner camera producer.
///
/// The producer runs on a dedicated OS thread that takes ownership of the
/// `Camera` by value (moved in exactly once) and is the ONLY thread that ever
/// touches it. This is the root-cause fix for the previous design, which shared
/// the `Camera` across tokio worker threads via `Arc<RwLock<Option<Camera>>>`.
/// The `Camera` contains a non-`Send` V4L2 `MmapStream`; sharing it across
/// threads was undefined behavior and caused the stream to be torn down on a
/// foreign thread (VIDIOC_STREAMOFF), closing the frame channel and killing the
/// daemon ~0.5s after launch.
///
/// The thread captures frames in a plain blocking loop and:
///  - publishes the latest frame to `latest_frame_tx` (read by IPC consumers:
///    enroll/auth/status/detect — they no longer touch the camera directly), and
///  - forwards each frame as a `RawFrame` to the pipeline via `frame_tx`.
///
/// It exits only on a fatal condition (no camera and none could be opened) or
/// when all receivers are gone.
pub fn spawn_camera_producer(
    source: CameraSource,
    config: Arc<Config>,
    latest_frame_tx: watch::Sender<Option<Arc<image::DynamicImage>>>,
    frame_tx: mpsc::Sender<RawFrame>,
) {
    let fps = config.camera.fps;
    info!("Camera producer started (target {}fps) - single-owner dedicated thread", fps);

    std::thread::spawn(move || {
        let interval_ms = 1000 / fps.max(1) as u64;
        let mut sequence = 0u64;
        let mut last_capture = std::time::Instant::now();

        // Open the camera ON THIS THREAD so the V4L2 stream's mmap buffers are
        // bound to this thread's fd context. This thread is the SOLE owner of
        // the camera and the only thread that ever touches it.
        let mut camera = match &source {
            CameraSource::VideoFile { path, width, height, fps, loop_playback } => {
                Camera::from_video_file(path.clone(), *width, *height, *fps, *loop_playback)
            }
            CameraSource::Config => {
                match futures::executor::block_on(Camera::new_with_config(&config)) {
                    Ok(cam) => {
                        info!("Camera opened in producer thread");
                        cam
                    }
                    Err(e) => {
                        error!("Camera not available and could not be opened: {}", e);
                        return;
                    }
                }
            }
        };

        loop {
            // Stop if every receiver has gone away (graceful shutdown).
            if frame_tx.is_closed() && latest_frame_tx.is_closed() {
                debug!("All frame receivers gone, camera producer exiting");
                return;
            }

            // Rate limit captures.
            let elapsed = last_capture.elapsed().as_millis() as u64;
            if elapsed < interval_ms {
                std::thread::sleep(std::time::Duration::from_millis(interval_ms - elapsed));
            }
            last_capture = std::time::Instant::now();

            // Blocking capture is fine: this is a dedicated OS thread and the
            // camera is never accessed from anywhere else.
            let frame = match camera.capture_frame() {
                Ok(f) => f,
                Err(e) => {
                    warn!("Capture error: {}", e);
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
            };

            sequence += 1;
            let image = Arc::new(frame);

            // Publish latest frame for IPC consumers (ignore if no receivers).
            let _ = latest_frame_tx.send(Some(image.clone()));

            let raw_frame = RawFrame {
                image,
                timestamp: std::time::Instant::now(),
                sequence,
            };

            // Non-blocking send to the pipeline (drop frame if channel full).
            if let Err(mpsc::error::TrySendError::Closed(_)) = frame_tx.try_send(raw_frame) {
                // Pipeline gone but IPC may still want frames; keep going unless
                // both are closed (checked at the top of the loop).
                debug!("Pipeline frame channel closed");
            }
        }
    });
}
