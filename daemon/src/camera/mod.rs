//! Camera abstraction layer for doorman.
//!
//! Backends (selected at compile time via Cargo features):
//! - `camera-mock`      : cross-platform synthetic / video-file frames (default, dev/CI)
//! - `camera-ffmpeg`    : live webcam via the `ffmpeg` CLI (macOS AVFoundation / Linux V4L2)
//! - `camera-gstreamer` : GStreamer + PipeWire (Linux)
//! - `camera-v4l2`      : V4L2 via nokhwa (Linux)
//! - `camera-nokhwa`    : native webcam (AVFoundation on macOS, MSMF on Windows, V4L2 on Linux)
//!
//! `Camera::new_with_config` tries the real backends first and always falls
//! back to the mock backend so the pipeline can run anywhere (e.g. macOS dev,
//! headless CI) without a physical camera.

use anyhow::Result;
use doorman_shared::Config;
use image::DynamicImage;

/// Camera backend trait - defines interface for all camera implementations.
pub trait CameraBackend: Send {
    /// Initialize camera with configuration.
    fn new_with_config(config: &Config) -> impl std::future::Future<Output = Result<Self>> + Send
    where
        Self: Sized;

    /// Initialize with default configuration.
    #[allow(dead_code)]
    fn new() -> impl std::future::Future<Output = Result<Self>> + Send
    where
        Self: Sized,
    {
        async { Self::new_with_config(&Config::default()).await }
    }

    /// Capture a single frame.
    fn capture_frame(&mut self) -> Result<DynamicImage>;

    /// Capture multiple frames.
    fn capture_frames(&mut self, count: usize) -> Vec<DynamicImage>;

    /// Check if camera is ready for capture.
    fn is_ready(&self) -> bool;

    /// Get backend name for logging/debugging.
    fn backend_name(&self) -> &'static str;
}

// --- Backend modules (feature-gated) ---

mod mock_backend;
pub use mock_backend::MockCamera;

#[cfg(feature = "camera-ffmpeg")]
mod ffmpeg_backend;
#[cfg(feature = "camera-ffmpeg")]
pub use ffmpeg_backend::FfmpegCamera;

#[cfg(feature = "camera-gstreamer")]
mod gstreamer_backend;
#[cfg(feature = "camera-gstreamer")]
pub use gstreamer_backend::GStreamerCamera;

#[cfg(any(feature = "camera-v4l2", feature = "camera-nokhwa"))]
mod v4l2_backend;
#[cfg(any(feature = "camera-v4l2", feature = "camera-nokhwa"))]
pub use v4l2_backend::V4L2Camera;

/// Backend enum for the unified camera.
enum CameraBackendInner {
    Mock(MockCamera),
    #[cfg(feature = "camera-ffmpeg")]
    Ffmpeg(FfmpegCamera),
    #[cfg(feature = "camera-gstreamer")]
    GStreamer(GStreamerCamera),
    #[cfg(any(feature = "camera-v4l2", feature = "camera-nokhwa"))]
    V4L2(V4L2Camera),
}

/// Public Camera type with automatic fallback to the mock backend.
pub struct Camera {
    inner: CameraBackendInner,
}

impl Camera {
    /// Create a camera that sources frames from a video file (dev/testing).
    pub fn from_video_file(
        path: std::path::PathBuf,
        width: u32,
        height: u32,
        fps: u32,
        loop_playback: bool,
    ) -> Self {
        Self {
            inner: CameraBackendInner::Mock(MockCamera::from_video_file(
                path,
                width,
                height,
                fps,
                loop_playback,
            )),
        }
    }

    pub async fn new_with_config(config: &Config) -> Result<Self> {
        // Try GStreamer first (PipeWire-integrated, Linux).
        #[cfg(feature = "camera-gstreamer")]
        {
            match GStreamerCamera::new_with_config(config).await {
                Ok(cam) => {
                    tracing::info!("Using GStreamer camera backend");
                    return Ok(Self {
                        inner: CameraBackendInner::GStreamer(cam),
                    });
                }
                Err(e) => tracing::warn!("GStreamer camera failed: {}, trying next backend", e),
            }
        }

        // Then V4L2 / native webcam (nokhwa).
        #[cfg(any(feature = "camera-v4l2", feature = "camera-nokhwa"))]
        {
            match V4L2Camera::new_with_config(config).await {
                Ok(cam) => {
                    tracing::info!("Using V4L2/nokhwa camera backend");
                    return Ok(Self {
                        inner: CameraBackendInner::V4L2(cam),
                    });
                }
                Err(e) => tracing::warn!("V4L2/nokhwa camera failed: {}, trying next backend", e),
            }
        }

        // Live webcam via the ffmpeg CLI (reliable real camera on macOS, where
        // nokhwa's AVFoundation path hangs). Degrades to mock if ffmpeg is
        // missing or the OS denies camera access.
        #[cfg(feature = "camera-ffmpeg")]
        {
            match FfmpegCamera::new_with_config(config).await {
                Ok(cam) => {
                    tracing::info!("Using ffmpeg live webcam backend");
                    return Ok(Self {
                        inner: CameraBackendInner::Ffmpeg(cam),
                    });
                }
                Err(e) => tracing::warn!("ffmpeg webcam failed: {}, falling back to mock", e),
            }
        }

        // Always-available cross-platform fallback.
        tracing::info!("Using mock camera backend (synthetic frames)");
        Ok(Self {
            inner: CameraBackendInner::Mock(MockCamera::new_with_config(config).await?),
        })
    }

    #[allow(dead_code)]
    pub async fn new() -> Result<Self> {
        Self::new_with_config(&Config::default()).await
    }

    pub fn capture_frame(&mut self) -> Result<DynamicImage> {
        match &mut self.inner {
            CameraBackendInner::Mock(cam) => cam.capture_frame(),
            #[cfg(feature = "camera-ffmpeg")]
            CameraBackendInner::Ffmpeg(cam) => cam.capture_frame(),
            #[cfg(feature = "camera-gstreamer")]
            CameraBackendInner::GStreamer(cam) => cam.capture_frame(),
            #[cfg(any(feature = "camera-v4l2", feature = "camera-nokhwa"))]
            CameraBackendInner::V4L2(cam) => cam.capture_frame(),
        }
    }

    pub fn capture_frames(&mut self, count: usize) -> Vec<DynamicImage> {
        match &mut self.inner {
            CameraBackendInner::Mock(cam) => cam.capture_frames(count),
            #[cfg(feature = "camera-ffmpeg")]
            CameraBackendInner::Ffmpeg(cam) => cam.capture_frames(count),
            #[cfg(feature = "camera-gstreamer")]
            CameraBackendInner::GStreamer(cam) => cam.capture_frames(count),
            #[cfg(any(feature = "camera-v4l2", feature = "camera-nokhwa"))]
            CameraBackendInner::V4L2(cam) => cam.capture_frames(count),
        }
    }

    /// Capture frames for a specified duration (for enrollment).
    pub fn capture_frames_for_duration(&mut self, duration_secs: u64) -> Vec<DynamicImage> {
        use std::time::{Duration, Instant};

        let mut frames = Vec::new();
        let duration = Duration::from_secs(duration_secs);
        let start = Instant::now();

        while start.elapsed() < duration {
            if let Ok(frame) = self.capture_frame() {
                frames.push(frame);
            }
            std::thread::sleep(Duration::from_millis(33)); // ~30fps max
        }

        frames
    }

    #[allow(dead_code)]
    pub fn is_open(&self) -> bool {
        match &self.inner {
            CameraBackendInner::Mock(cam) => cam.is_open(),
            #[cfg(feature = "camera-ffmpeg")]
            CameraBackendInner::Ffmpeg(_) => true,
            #[cfg(feature = "camera-gstreamer")]
            CameraBackendInner::GStreamer(_) => true,
            #[cfg(any(feature = "camera-v4l2", feature = "camera-nokhwa"))]
            CameraBackendInner::V4L2(_) => true,
        }
    }
}
