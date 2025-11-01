use anyhow::{anyhow, Context, Result};
use image::{DynamicImage, ImageBuffer, Rgb};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::Camera as NokhwaCamera;
use tracing::{debug, info, warn};

pub struct Camera {
    camera: NokhwaCamera,
}

impl Camera {
    /// Initialize the camera
    pub async fn new() -> Result<Self> {
        info!("Opening camera...");
        
        // Try to find the first available camera
        let index = CameraIndex::Index(0);
        
        // Request a reasonable resolution (will downscale anyway)
        let requested = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::AbsoluteHighestResolution
        );

        let camera = NokhwaCamera::new(index, requested)
            .context("Failed to open camera")?;

        info!("Camera opened successfully");
        Ok(Self { camera })
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

