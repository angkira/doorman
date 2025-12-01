/// Video file camera backend for testing
use anyhow::{anyhow, Result};
use image::{DynamicImage, ImageBuffer, Rgb};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
use std::thread;
use tracing::{debug, error, info, warn};

use super::CameraBackend;

pub struct VideoBackend {
    video_path: PathBuf,
    width: u32,
    height: u32,
    fps: u32,
    current_frame: Arc<Mutex<Option<DynamicImage>>>,
    _ffmpeg_process: Option<Child>,
}

impl VideoBackend {
    pub fn new(video_path: PathBuf, width: u32, height: u32) -> Result<Self> {
        if !video_path.exists() {
            return Err(anyhow!("Video file not found: {:?}", video_path));
        }

        info!("Initializing Video backend from file: {:?}", video_path);
        info!("Target resolution: {}x{}", width, height);

        Ok(Self {
            video_path,
            width,
            height,
            fps: 30, // Default FPS
            current_frame: Arc::new(Mutex::new(None)),
            _ffmpeg_process: None,
        })
    }

    fn start_ffmpeg_stream(&mut self) -> Result<()> {
        info!("Starting FFmpeg video stream from {:?}", self.video_path);

        // FFmpeg command to read video and output raw RGB24 frames
        let mut child = Command::new("ffmpeg")
            .args(&[
                "-re", // Read input at native frame rate
                "-i", self.video_path.to_str().unwrap(),
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", &format!("{}x{}", self.width, self.height),
                "-r", &format!("{}", self.fps),
                "pipe:1",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;

        let stdout = child.stdout.take().ok_or_else(|| anyhow!("Failed to capture stdout"))?;
        let current_frame = Arc::clone(&self.current_frame);
        let width = self.width;
        let height = self.height;
        let frame_size = (width * height * 3) as usize; // RGB24 = 3 bytes per pixel

        // Spawn thread to read frames
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut buffer = vec![0u8; frame_size];

            loop {
                match reader.read_exact(&mut buffer) {
                    Ok(_) => {
                        // Convert RGB24 buffer to DynamicImage
                        if let Some(img_buf) = ImageBuffer::<Rgb<u8>, _>::from_raw(
                            width,
                            height,
                            buffer.clone(),
                        ) {
                            let img = DynamicImage::ImageRgb8(img_buf);
                            if let Ok(mut frame) = current_frame.lock() {
                                *frame = Some(img);
                            }
                        }
                    }
                    Err(e) => {
                        if e.kind() != std::io::ErrorKind::UnexpectedEof {
                            error!("Error reading video frame: {}", e);
                        }
                        break;
                    }
                }
            }
            
            info!("Video stream ended");
        });

        info!("Video stream started");
        Ok(())
    }
}

impl CameraBackend for VideoBackend {
    fn capture(&mut self) -> Result<DynamicImage> {
        // Start streaming if not started
        if self._ffmpeg_process.is_none() {
            self.start_ffmpeg_stream()?;
            
            // Wait a bit for first frame
            std::thread::sleep(std::time::Duration::from_millis(500));
        }

        // Get current frame
        let frame = self.current_frame.lock()
            .map_err(|e| anyhow!("Failed to lock frame: {}", e))?
            .clone();

        frame.ok_or_else(|| anyhow!("No frame available from video"))
    }

    fn stop(&mut self) {
        info!("Stopping video backend");
        // FFmpeg process will be killed when dropped
    }

    fn name(&self) -> &'static str {
        "Video File"
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl Drop for VideoBackend {
    fn drop(&mut self) {
        if let Some(mut child) = self._ffmpeg_process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
