/// OpenCV camera backend - Stable cross-platform access
///
/// This implementation uses OpenCV's VideoCapture which is widely supported.
/// - Stable and mature
/// - Works across Linux, Windows, macOS
/// - Better driver compatibility than raw V4L2
/// - Simpler than GStreamer
use super::CameraBackend;
use anyhow::{anyhow, Context, Result};
use doorman_shared::Config;
use image::{DynamicImage, ImageBuffer, Rgb};
use opencv::{
    prelude::*,
    videoio,
};
use tracing::{debug, info, warn};

pub struct OpenCVCamera {
    camera: videoio::VideoCapture,
    width: u32,
    height: u32,
}

// Safety: opencv::VideoCapture doesn't implement Send/Sync,
// but we only use it from a single thread at a time (protected by RwLock in daemon).
unsafe impl Send for OpenCVCamera {}
unsafe impl Sync for OpenCVCamera {}

impl CameraBackend for OpenCVCamera {
    async fn new_with_config(config: &Config) -> Result<Self> {
        info!("Initializing OpenCV camera backend");
        info!("Opening camera {}...", config.camera.device_index);

        // Open camera
        let mut camera = videoio::VideoCapture::new(config.camera.device_index as i32, videoio::CAP_ANY)
            .context("Failed to create VideoCapture")?;

        if !camera.is_opened().context("Failed to check if camera is opened")? {
            return Err(anyhow!("Failed to open camera {}", config.camera.device_index));
        }

        // Set resolution
        camera.set(videoio::CAP_PROP_FRAME_WIDTH, config.camera.width as f64)
            .context("Failed to set frame width")?;
        camera.set(videoio::CAP_PROP_FRAME_HEIGHT, config.camera.height as f64)
            .context("Failed to set frame height")?;
        camera.set(videoio::CAP_PROP_FPS, config.camera.fps as f64)
            .context("Failed to set FPS")?;

        // Get actual resolution (camera may not support requested resolution)
        let actual_width = camera.get(videoio::CAP_PROP_FRAME_WIDTH)
            .context("Failed to get frame width")? as u32;
        let actual_height = camera.get(videoio::CAP_PROP_FRAME_HEIGHT)
            .context("Failed to get frame height")? as u32;
        let actual_fps = camera.get(videoio::CAP_PROP_FPS)
            .context("Failed to get FPS")? as u32;

        info!(
            "OpenCV camera opened: {}x{} @ {}fps",
            actual_width, actual_height, actual_fps
        );

        if actual_width != config.camera.width || actual_height != config.camera.height {
            warn!(
                "Camera resolution {}x{} differs from requested {}x{}",
                actual_width, actual_height, config.camera.width, config.camera.height
            );
        }

        Ok(Self {
            camera,
            width: actual_width,
            height: actual_height,
        })
    }

    fn capture_frame(&mut self) -> Result<DynamicImage> {
        let mut frame = Mat::default();

        // Read frame
        self.camera.read(&mut frame)
            .context("Failed to read frame from camera")?;

        if frame.empty() {
            return Err(anyhow!("Captured frame is empty"));
        }

        // Convert from BGR (OpenCV default) to RGB
        let mut rgb_frame = Mat::default();
        opencv::imgproc::cvt_color(&frame, &mut rgb_frame, opencv::imgproc::COLOR_BGR2RGB, 0)
            .context("Failed to convert BGR to RGB")?;

        // Extract data
        let data = rgb_frame.data_bytes()
            .context("Failed to get frame data")?
            .to_vec();

        let width = rgb_frame.cols() as u32;
        let height = rgb_frame.rows() as u32;

        // Create image buffer
        let img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(width, height, data)
            .ok_or_else(|| anyhow!("Failed to create image buffer from frame data"))?;

        Ok(DynamicImage::ImageRgb8(img_buffer))
    }

    fn capture_frames(&mut self, count: usize) -> Vec<DynamicImage> {
        let mut frames = Vec::new();

        for i in 0..count {
            match self.capture_frame() {
                Ok(frame) => {
                    debug!("Captured frame {}/{}", i + 1, count);
                    frames.push(frame);
                }
                Err(e) => {
                    warn!("Failed to capture frame {}/{}: {}", i + 1, count, e);
                }
            }

            // Small delay between captures
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        frames
    }

    fn is_ready(&self) -> bool {
        self.camera.is_opened().unwrap_or(false)
    }

    fn backend_name(&self) -> &'static str {
        "OpenCV"
    }
}

impl Drop for OpenCVCamera {
    fn drop(&mut self) {
        debug!("OpenCV camera dropped");
        let _ = self.camera.release();
    }
}

// Re-export Mat for use in other modules
pub use opencv::core::Mat;
