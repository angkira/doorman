use anyhow::{anyhow, Context, Result};
use image::{DynamicImage, ImageBuffer, Rgb};
use std::path::Path;
use tracing::{debug, info, warn};

#[cfg(feature = "video")]
use opencv::{prelude::*, videoio, core};

/// Video file reader for testing
pub struct VideoReader {
    #[cfg(feature = "video")]
    capture: videoio::VideoCapture,
    
    #[cfg(not(feature = "video"))]
    _phantom: std::marker::PhantomData<()>,
}

impl VideoReader {
    /// Open a video file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        #[cfg(feature = "video")]
        {
            let path_str = path.as_ref().to_str()
                .ok_or_else(|| anyhow!("Invalid path"))?;
            
            info!("Opening video file: {}", path_str);
            
            let capture = videoio::VideoCapture::from_file(path_str, videoio::CAP_ANY)
                .context("Failed to open video file")?;
            
            if !capture.is_opened()? {
                return Err(anyhow!("Video file could not be opened"));
            }
            
            Ok(Self { capture })
        }
        
        #[cfg(not(feature = "video"))]
        {
            let _ = path;
            Err(anyhow!("Video support not compiled. Enable 'video' feature."))
        }
    }

    /// Read the next frame from the video
    pub fn read_frame(&mut self) -> Result<Option<DynamicImage>> {
        #[cfg(feature = "video")]
        {
            let mut frame = core::Mat::default();
            
            if !self.capture.read(&mut frame)? {
                return Ok(None);
            }
            
            if frame.empty() {
                return Ok(None);
            }
            
            // Convert BGR to RGB
            let mut rgb_frame = core::Mat::default();
            opencv::imgproc::cvt_color(&frame, &mut rgb_frame, opencv::imgproc::COLOR_BGR2RGB, 0)?;
            
            let width = rgb_frame.cols() as u32;
            let height = rgb_frame.rows() as u32;
            
            // Convert to image buffer
            let data = rgb_frame.data_bytes()?.to_vec();
            
            let img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(width, height, data)
                .ok_or_else(|| anyhow!("Failed to create image buffer"))?;
            
            Ok(Some(DynamicImage::ImageRgb8(img_buffer)))
        }
        
        #[cfg(not(feature = "video"))]
        {
            Err(anyhow!("Video support not compiled"))
        }
    }

    /// Read all frames from the video
    pub fn read_all_frames(&mut self) -> Vec<DynamicImage> {
        let mut frames = Vec::new();
        
        loop {
            match self.read_frame() {
                Ok(Some(frame)) => {
                    debug!("Read frame {}", frames.len());
                    frames.push(frame);
                }
                Ok(None) => break,
                Err(e) => {
                    warn!("Error reading frame: {}", e);
                    break;
                }
            }
        }
        
        info!("Read {} frames from video", frames.len());
        frames
    }

    /// Get video properties
    #[cfg(feature = "video")]
    pub fn get_fps(&self) -> Result<f64> {
        Ok(self.capture.get(videoio::CAP_PROP_FPS)?)
    }

    #[cfg(feature = "video")]
    pub fn get_frame_count(&self) -> Result<i32> {
        Ok(self.capture.get(videoio::CAP_PROP_FRAME_COUNT)? as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "video")]
    fn test_video_reader_creation() {
        // This will fail without a real video file, but tests the API
        let result = VideoReader::new("nonexistent.mp4");
        assert!(result.is_err());
    }
}

