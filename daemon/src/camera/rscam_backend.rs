/// rscam V4L2 camera backend - Simple and stable
///
/// This implementation uses rscam for direct V4L2 access.
/// - Simple and reliable
/// - Pure Rust wrapper over V4L2
/// - No complex C bindings
/// - Better stability than nokhwa
use super::CameraBackend;
use anyhow::{anyhow, Context, Result};
use doorman_shared::Config;
use image::{DynamicImage, ImageBuffer, Rgb};
use rscam::{Camera as RscamCamera, Config as CameraConfig};
use tracing::{debug, info, warn};

pub struct RscamV4L2Camera {
    camera: RscamCamera,
    width: u32,
    height: u32,
}

// Safety: rscam's Camera doesn't implement Send/Sync,
// but we only use it from a single thread at a time (protected by RwLock in daemon).
unsafe impl Send for RscamV4L2Camera {}
unsafe impl Sync for RscamV4L2Camera {}

impl CameraBackend for RscamV4L2Camera {
    async fn new_with_config(config: &Config) -> Result<Self> {
        info!("Initializing rscam V4L2 camera backend");
        let device_path = format!("/dev/video{}", config.camera.device_index);
        info!("Opening camera at {}...", device_path);

        // Open camera
        let mut camera = RscamCamera::new(&device_path)
            .context(format!("Failed to open camera at {}", device_path))?;

        // Start camera with RGB3 format (24-bit RGB)
        camera.start(&CameraConfig {
            interval: (1, config.camera.fps),  // (numerator, denominator)
            resolution: (config.camera.width, config.camera.height),
            format: b"RGB3",  // 24-bit RGB
            ..Default::default()
        }).context("Failed to start camera with RGB3 format")?;

        info!(
            "rscam camera opened: {}x{} @ {}fps",
            config.camera.width, config.camera.height, config.camera.fps
        );

        Ok(Self {
            camera,
            width: config.camera.width,
            height: config.camera.height,
        })
    }

    fn capture_frame(&mut self) -> Result<DynamicImage> {
        // Capture frame
        let frame = self.camera.capture()
            .context("Failed to capture frame")?;

        // rscam returns RGB24 data directly
        let data = frame.to_vec();

        // Create image buffer
        let img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(self.width, self.height, data)
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
        true // rscam camera is ready once initialized
    }

    fn backend_name(&self) -> &'static str {
        "rscam V4L2"
    }
}

impl Drop for RscamV4L2Camera {
    fn drop(&mut self) {
        debug!("rscam camera dropped");
    }
}
