/// Camera abstraction layer for doorman
///
/// Provides multiple backends for camera access with proper Linux integration:
/// - GStreamer: PipeWire-aware, desktop-integrated (default)
/// - V4L2: Direct hardware access, fallback for compatibility
///
/// Security considerations:
/// - GStreamer backend uses PipeWire for permission management
/// - Proper error handling with context
/// - Resource cleanup via RAII (Drop trait)
/// - Frame validation and sanitization
use anyhow::Result;
use doorman_shared::Config;
use image::DynamicImage;

/// Camera backend trait - defines interface for all camera implementations
pub trait CameraBackend: Send {
    /// Initialize camera with configuration
    fn new_with_config(config: &Config) -> impl std::future::Future<Output = Result<Self>> + Send
    where
        Self: Sized;

    /// Initialize with default configuration
    fn new() -> impl std::future::Future<Output = Result<Self>> + Send
    where
        Self: Sized,
    {
        async { Self::new_with_config(&Config::default()).await }
    }

    /// Capture a single frame
    ///
    /// Returns DynamicImage which is validated and safe to process
    fn capture_frame(&mut self) -> Result<DynamicImage>;

    /// Capture multiple frames with timing control
    ///
    /// Arguments:
    /// - count: Number of frames to capture
    ///
    /// Returns vector of successfully captured frames (may be less than count)
    fn capture_frames(&mut self, count: usize) -> Vec<DynamicImage>;

    /// Check if camera is ready for capture
    fn is_ready(&self) -> bool;

    /// Get backend name for logging/debugging
    fn backend_name(&self) -> &'static str;
}

// Select camera backend based on features
#[cfg(feature = "camera-gstreamer")]
mod gstreamer_backend;

#[cfg(feature = "camera-gstreamer")]
pub use gstreamer_backend::GStreamerCamera;

#[cfg(feature = "camera-v4l2")]
mod v4l2_backend;

#[cfg(feature = "camera-v4l2")]
pub use v4l2_backend::V4L2Camera;

// OpenCV backend (most stable)
#[cfg(feature = "video")]
mod opencv_backend;

#[cfg(feature = "video")]
pub use opencv_backend::OpenCVCamera;

// rscam V4L2 backend (simple and stable)
#[cfg(feature = "camera-rscam")]
mod rscam_backend;

#[cfg(feature = "camera-rscam")]
pub use rscam_backend::RscamV4L2Camera;

// PipeWire backend (native pw-stream API)
#[cfg(feature = "camera-pipewire")]
mod pipewire_backend;

#[cfg(feature = "camera-pipewire")]
pub use pipewire_backend::PipeWireCamera;

// FFmpeg backend (most reliable, uses CLI)
mod ffmpeg_backend;
pub use ffmpeg_backend::FFmpegCamera;

// Video file backend (for testing with video files)
mod video_file_backend;
pub use video_file_backend::VideoFileBackend;

/// Backend enum for hybrid camera support
enum CameraBackendInner {
    FFmpeg(FFmpegCamera),
    VideoFile(VideoFileBackend),
    #[cfg(feature = "video")]
    OpenCV(OpenCVCamera),
    #[cfg(feature = "camera-gstreamer")]
    GStreamer(GStreamerCamera),
    #[cfg(feature = "camera-pipewire")]
    PipeWire(PipeWireCamera),
    #[cfg(feature = "camera-v4l2")]
    V4L2(V4L2Camera),
    #[cfg(feature = "camera-rscam")]
    RscamV4L2(RscamV4L2Camera),
}

/// Public Camera type with automatic fallback
/// Tries OpenCV first (most stable), then GStreamer (PipeWire), then V4L2
pub struct Camera {
    inner: CameraBackendInner,
}

impl Camera {
    /// Create camera from video file (for testing)
    pub fn from_video_file(path: std::path::PathBuf, width: u32, height: u32, fps: u32, loop_playback: bool) -> Self {
        Self {
            inner: CameraBackendInner::VideoFile(VideoFileBackend::new(path, width, height, fps, loop_playback)),
        }
    }

    pub async fn new_with_config(config: &Config) -> Result<Self> {
        // Try PipeWire first (native pw-stream API, fastest and most compatible)
        #[cfg(feature = "camera-pipewire")]
        {
            match PipeWireCamera::new_with_config(config).await {
                Ok(cam) => {
                    tracing::info!("Using PipeWire camera backend (native, fast, desktop-integrated)");
                    return Ok(Self {
                        inner: CameraBackendInner::PipeWire(cam),
                    });
                }
                Err(e) => {
                    tracing::warn!("PipeWire camera failed: {}, trying next backend", e);
                }
            }
        }

        // Try GStreamer second (PipeWire-integrated, fast streaming)
        #[cfg(feature = "camera-gstreamer")]
        {
            match GStreamerCamera::new_with_config(config).await {
                Ok(cam) => {
                    tracing::info!("Using GStreamer camera backend (PipeWire-integrated, fast)");
                    return Ok(Self {
                        inner: CameraBackendInner::GStreamer(cam),
                    });
                }
                Err(e) => {
                    tracing::warn!("GStreamer camera failed: {}, trying next backend", e);
                }
            }
        }

        // Try OpenCV third (if available)
        #[cfg(feature = "video")]
        {
            match OpenCVCamera::new_with_config(config).await {
                Ok(cam) => {
                    tracing::info!("Using OpenCV camera backend (stable, cross-platform)");
                    return Ok(Self {
                        inner: CameraBackendInner::OpenCV(cam),
                    });
                }
                Err(e) => {
                    tracing::warn!("OpenCV camera failed: {}, trying next backend", e);
                }
            }
        }

        // Try FFmpeg last (slow but reliable fallback, uses CLI)
        match FFmpegCamera::new_with_config(config).await {
            Ok(cam) => {
                tracing::info!("Using FFmpeg camera backend (slow fallback, CLI-based)");
                return Ok(Self {
                    inner: CameraBackendInner::FFmpeg(cam),
                });
            }
            Err(e) => {
                tracing::warn!("FFmpeg camera failed: {}, trying next backend", e);
            }
        }

        // Try rscam V4L2 (simple and stable)
        #[cfg(feature = "camera-rscam")]
        {
            match RscamV4L2Camera::new_with_config(config).await {
                Ok(cam) => {
                    tracing::info!("Using rscam V4L2 camera backend (simple and stable)");
                    return Ok(Self {
                        inner: CameraBackendInner::RscamV4L2(cam),
                    });
                }
                Err(e) => {
                    tracing::warn!("rscam V4L2 camera failed: {}, trying next backend", e);
                }
            }
        }

        // Try nokhwa V4L2 last (may be unstable)
        #[cfg(feature = "camera-v4l2")]
        {
            let cam = V4L2Camera::new_with_config(config).await?;
            tracing::info!("Using nokhwa V4L2 camera backend (direct hardware access)");
            return Ok(Self {
                inner: CameraBackendInner::V4L2(cam),
            });
        }

        // If we get here, all backends failed (including FFmpeg which is always available)
        Err(anyhow::anyhow!("All camera backends failed. Is ffmpeg installed? (sudo apt-get install ffmpeg)"))
    }

    pub async fn new() -> Result<Self> {
        Self::new_with_config(&Config::default()).await
    }

    pub fn capture_frame(&mut self) -> Result<DynamicImage> {
        match &mut self.inner {
            CameraBackendInner::FFmpeg(cam) => cam.capture_frame(),
            CameraBackendInner::VideoFile(cam) => cam.capture_frame(),
            #[cfg(feature = "video")]
            CameraBackendInner::OpenCV(cam) => cam.capture_frame(),
            #[cfg(feature = "camera-gstreamer")]
            CameraBackendInner::GStreamer(cam) => cam.capture_frame(),
            #[cfg(feature = "camera-pipewire")]
            CameraBackendInner::PipeWire(cam) => cam.capture_frame(),
            #[cfg(feature = "camera-v4l2")]
            CameraBackendInner::V4L2(cam) => cam.capture_frame(),
            #[cfg(feature = "camera-rscam")]
            CameraBackendInner::RscamV4L2(cam) => cam.capture_frame(),
        }
    }

    pub fn capture_frames(&mut self, count: usize) -> Vec<DynamicImage> {
        match &mut self.inner {
            CameraBackendInner::FFmpeg(cam) => cam.capture_frames(count),
            CameraBackendInner::VideoFile(cam) => {
                // VideoFile doesn't implement capture_frames, do it manually
                (0..count)
                    .filter_map(|_| cam.capture_frame().ok())
                    .collect()
            },
            #[cfg(feature = "video")]
            CameraBackendInner::OpenCV(cam) => cam.capture_frames(count),
            #[cfg(feature = "camera-gstreamer")]
            CameraBackendInner::GStreamer(cam) => cam.capture_frames(count),
            #[cfg(feature = "camera-pipewire")]
            CameraBackendInner::PipeWire(cam) => cam.capture_frames(count),
            #[cfg(feature = "camera-v4l2")]
            CameraBackendInner::V4L2(cam) => cam.capture_frames(count),
            #[cfg(feature = "camera-rscam")]
            CameraBackendInner::RscamV4L2(cam) => cam.capture_frames(count),
        }
    }

    pub fn is_open(&self) -> bool {
        match &self.inner {
            CameraBackendInner::FFmpeg(cam) => cam.is_open(),
            CameraBackendInner::VideoFile(cam) => cam.is_open(),
            #[cfg(feature = "video")]
            CameraBackendInner::OpenCV(_) => true,
            #[cfg(feature = "camera-gstreamer")]
            CameraBackendInner::GStreamer(_) => true,
            #[cfg(feature = "camera-pipewire")]
            CameraBackendInner::PipeWire(_) => true,
            #[cfg(feature = "camera-v4l2")]
            CameraBackendInner::V4L2(_) => true,
            #[cfg(feature = "camera-rscam")]
            CameraBackendInner::RscamV4L2(_) => true,
        }
    }
}

// Old backend-specific implementations removed - now using unified hybrid backend
