use anyhow::{anyhow, Context, Result};
use doorman_shared::Config;
use image::{DynamicImage, ImageBuffer, Rgb};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::Camera as NokhwaCamera;
use tracing::{debug, info, warn};

pub struct Camera {
    camera: NokhwaCamera,
}

impl Camera {
    /// Initialize camera with config
    pub async fn new_with_config(config: &Config) -> Result<Self> {
        info!("Opening camera {}...", config.camera.device_index);
        
        let index = CameraIndex::Index(config.camera.device_index);
        
        // Request specific resolution from config
        let requested = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::AbsoluteHighestResolution
        );

        let mut camera = NokhwaCamera::new(index, requested)
            .context("Failed to open camera")?;
            
        // Try to set the requested resolution
        let resolution = Resolution::new(config.camera.width, config.camera.height);
        let _ = camera.set_resolution(resolution);

        // Start the camera stream
        camera.open_stream()
            .context("Failed to start camera stream")?;

        info!(
            "Camera opened: {}x{} @ {}fps",
            config.camera.width, config.camera.height, config.camera.fps
        );
        
        Ok(Self { camera })
    }
    
    /// Initialize with defaults
    pub async fn new() -> Result<Self> {
        Self::new_with_config(&Config::default()).await
    }

    /// Capture a frame from the camera
    pub fn capture_frame(&mut self) -> Result<DynamicImage> {
        let frame = self.camera.frame()
            .context("Failed to capture frame")?;
        
        let decoded = frame.decode_image::<RgbFormat>()
            .context("Failed to decode frame")?;
        
        let width = decoded.width();
        let height = decoded.height();
        let data = decoded.into_raw();
        
        let img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(width, height, data)
            .ok_or_else(|| anyhow!("Failed to create image buffer"))?;
        
        Ok(DynamicImage::ImageRgb8(img_buffer))
    }

    /// Capture multiple frames
    pub fn capture_frames(&mut self, count: usize) -> Vec<DynamicImage> {
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
}

impl Drop for Camera {
    fn drop(&mut self) {
        debug!("Camera dropped");
    }
}

