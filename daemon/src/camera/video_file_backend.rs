use anyhow::{anyhow, Result};
use image::{DynamicImage, ImageBuffer, Rgb};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use super::CameraBackend;

pub struct VideoFileBackend {
    video_path: PathBuf,
    width: u32,
    height: u32,
    fps: u32,
    process: Arc<Mutex<Option<Child>>>,
    frame_interval: Duration,
    last_frame_time: Arc<Mutex<Instant>>,
    loop_playback: bool,
}

impl VideoFileBackend {
    pub fn new(video_path: PathBuf, width: u32, height: u32, fps: u32, loop_playback: bool) -> Self {
        info!("Initializing Video File camera backend");
        info!("Video file: {:?}", video_path);
        info!("Output resolution: {}x{} @ {}fps", width, height, fps);
        info!("Loop playback: {}", loop_playback);

        Self {
            video_path,
            width,
            height,
            fps,
            process: Arc::new(Mutex::new(None)),
            frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
            last_frame_time: Arc::new(Mutex::new(Instant::now())),
            loop_playback,
        }
    }

    fn start_ffmpeg_process(&self) -> Result<Child> {
        let mut cmd = Command::new("ffmpeg");
        
        // Input options
        if self.loop_playback {
            cmd.arg("-stream_loop").arg("-1"); // Loop indefinitely
        }
        cmd.arg("-i").arg(&self.video_path);
        
        // Output options: scale and convert to RGB24 raw frames
        cmd.arg("-vf")
           .arg(format!("scale={}:{}", self.width, self.height))
           .arg("-f").arg("rawvideo")
           .arg("-pix_fmt").arg("rgb24")
           .arg("-r").arg(self.fps.to_string())
           .arg("-");
        
        cmd.stdout(Stdio::piped())
           .stderr(Stdio::null());
        
        debug!("Starting FFmpeg: {:?}", cmd);
        
        let child = cmd.spawn()
            .map_err(|e| anyhow!("Failed to spawn FFmpeg process: {}", e))?;
        
        info!("FFmpeg process started for video file");
        Ok(child)
    }

    fn ensure_process_running(&self) -> Result<()> {
        let mut process_guard = self.process.lock().unwrap();
        
        if let Some(ref mut child) = *process_guard {
            // Check if process is still running
            match child.try_wait() {
                Ok(None) => return Ok(()), // Still running
                Ok(Some(status)) => {
                    if self.loop_playback {
                        warn!("FFmpeg process exited unexpectedly with status: {}, restarting...", status);
                    } else {
                        return Err(anyhow!("Video playback finished"));
                    }
                }
                Err(e) => {
                    warn!("Failed to check FFmpeg process status: {}, restarting...", e);
                }
            }
        }
        
        // Start or restart process
        let child = self.start_ffmpeg_process()?;
        *process_guard = Some(child);
        Ok(())
    }
}

impl CameraBackend for VideoFileBackend {
    async fn new_with_config(_config: &doorman_shared::Config) -> Result<Self> {
        Err(anyhow!("VideoFileBackend requires explicit construction with from_video_file()"))
    }

    fn capture_frame(&mut self) -> Result<DynamicImage> {
        // Rate limiting to match target FPS
        let now = Instant::now();
        let mut last_time = self.last_frame_time.lock().unwrap();
        let elapsed = now.duration_since(*last_time);
        if elapsed < self.frame_interval {
            std::thread::sleep(self.frame_interval - elapsed);
        }
        *last_time = Instant::now();
        
        // Ensure FFmpeg process is running
        self.ensure_process_running()?;
        
        // Read one frame from stdout
        let frame_size = (self.width * self.height * 3) as usize; // RGB24
        let mut buffer = vec![0u8; frame_size];
        
        let mut process_guard = self.process.lock().unwrap();
        let child = process_guard.as_mut()
            .ok_or_else(|| anyhow!("No FFmpeg process"))?;
        
        let stdout = child.stdout.as_mut()
            .ok_or_else(|| anyhow!("No stdout on FFmpeg process"))?;
        
        // Read exact frame size
        stdout.read_exact(&mut buffer)
            .map_err(|e| anyhow!("Failed to read frame from FFmpeg: {}", e))?;
        
        // Convert RGB buffer to DynamicImage
        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(self.width, self.height, buffer)
            .ok_or_else(|| anyhow!("Failed to create image from buffer"))?;
        
        Ok(DynamicImage::ImageRgb8(img))
    }

    fn capture_frames(&mut self, count: usize) -> Vec<DynamicImage> {
        (0..count)
            .filter_map(|_| self.capture_frame().ok())
            .collect()
    }

    fn is_ready(&self) -> bool {
        let mut process_guard = self.process.lock().unwrap();
        if let Some(ref mut child) = *process_guard {
            matches!(child.try_wait(), Ok(None))
        } else {
            false
        }
    }

    fn backend_name(&self) -> &'static str {
        "VideoFile"
    }
}

impl VideoFileBackend {
    pub fn is_open(&self) -> bool {
        let mut process_guard = self.process.lock().unwrap();
        if let Some(ref mut child) = *process_guard {
            matches!(child.try_wait(), Ok(None))
        } else {
            false
        }
    }
}

impl Drop for VideoFileBackend {
    fn drop(&mut self) {
        if let Some(mut child) = self.process.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
            info!("FFmpeg process terminated");
        }
    }
}
