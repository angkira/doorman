/// FFmpeg camera backend - Continuous streaming implementation
///
/// This implementation uses a persistent FFmpeg process for continuous frame capture.
/// - One ffmpeg process for the entire session (not per frame!)
/// - Streams raw RGB frames continuously via stdout
/// - Extremely stable and battle-tested
/// - Works with any camera supported by FFmpeg
use super::CameraBackend;
use anyhow::{anyhow, Context, Result};
use doorman_shared::Config;
use image::{DynamicImage, ImageBuffer, Rgb};
use std::io::Read;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

pub struct FFmpegCamera {
    source: CameraSource,
    width: u32,
    height: u32,
    fps: u32,
    process: Arc<Mutex<Option<Child>>>,
    stdout: Arc<Mutex<Option<ChildStdout>>>,
    frame_size: usize,
}

enum CameraSource {
    Device(String),  // /dev/videoN
    VideoFile(String),  // path to video file
}

// FFmpegCamera is thread-safe via Arc<Mutex>
unsafe impl Send for FFmpegCamera {}
unsafe impl Sync for FFmpegCamera {}

impl CameraBackend for FFmpegCamera {
    async fn new_with_config(config: &Config) -> Result<Self> {
        info!("Initializing FFmpeg camera backend (continuous streaming)");
        
        // Determine source: video file or camera device
        let source = if let Some(ref video_file) = config.camera.video_file {
            info!("Using video file: {}", video_file);
            CameraSource::VideoFile(video_file.clone())
        } else {
            let device_path = format!("/dev/video{}", config.camera.device_index);
            info!("Using camera device: {}", device_path);
            CameraSource::Device(device_path)
        };

        // Test that ffmpeg is available
        let test = Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        if test.is_err() || !test.unwrap().success() {
            return Err(anyhow!("ffmpeg not found. Install with: sudo apt-get install ffmpeg"));
        }

        // For video files, skip camera test and use configured resolution
        let (actual_width, actual_height) = match &source {
            CameraSource::VideoFile(_) => {
                info!("Video file source, using configured resolution: {}x{}", 
                      config.camera.width, config.camera.height);
                (config.camera.width, config.camera.height)
            }
            CameraSource::Device(device_path) => {
                // Test camera access and detect actual resolution
                info!("Testing camera access and detecting actual resolution...");
                let test_capture = Command::new("ffmpeg")
                    .args(&[
                        "-f", "v4l2",
                        "-i", device_path,
                        "-frames:v", "1",
                        "-f", "null",
                        "-"
                    ])
                    .stdout(Stdio::null())
                    .stderr(Stdio::piped())
                    .output();

                match test_capture {
                    Ok(output) => {
                        let stderr = String::from_utf8_lossy(&output.stderr);

                        // Check if camera is busy - if so, just use config resolution
                        if stderr.contains("Device or resource busy") {
                            warn!("Camera is busy during test, using configured resolution: {}x{}",
                                  config.camera.width, config.camera.height);
                            (config.camera.width, config.camera.height)
                        } else {
                            if output.status.success() {
                                info!("FFmpeg camera test successful, using configured resolution: {}x{}", 
                                      config.camera.width, config.camera.height);
                            } else {
                                warn!("Camera test had issues: {}", stderr.lines().take(3).collect::<Vec<_>>().join(" | "));
                                warn!("Continuing with configured resolution: {}x{}", 
                                      config.camera.width, config.camera.height);
                            }

                            (config.camera.width, config.camera.height)
                        }
                    }
                    Err(e) => {
                        return Err(anyhow!("Failed to test camera: {}", e));
                    }
                }
            }
        };

        let frame_size = (actual_width * actual_height * 3) as usize;

        info!(
            "FFmpeg camera ready: {}x{} @ {}fps (continuous streaming)",
            actual_width, actual_height, config.camera.fps
        );

        Ok(Self {
            source,
            width: actual_width,
            height: actual_height,
            fps: config.camera.fps,
            process: Arc::new(Mutex::new(None)),
            stdout: Arc::new(Mutex::new(None)),
            frame_size,
        })
    }

    fn capture_frame(&mut self) -> Result<DynamicImage> {
        // Start the streaming process if not already running
        {
            let mut process_lock = self.process.lock().unwrap();
            if process_lock.is_none() {
                debug!("Starting FFmpeg streaming process");
                
                // Build ffmpeg command based on source type
                let mut cmd = Command::new("ffmpeg");
                cmd.arg("-loglevel").arg("panic"); // Suppress warnings
                
                match &self.source {
                    CameraSource::Device(device_path) => {
                        cmd.arg("-f").arg("v4l2")
                           .arg("-framerate").arg(self.fps.to_string())
                           .arg("-video_size").arg(format!("{}x{}", self.width, self.height))
                           .arg("-i").arg(device_path);
                    }
                    CameraSource::VideoFile(video_path) => {
                        cmd.arg("-stream_loop").arg("-1") // Loop video indefinitely
                           .arg("-re") // Read at native framerate
                           .arg("-i").arg(video_path)
                           .arg("-vf").arg(format!("scale={}:{}", self.width, self.height));
                    }
                }
                
                cmd.arg("-f").arg("rawvideo")
                   .arg("-pix_fmt").arg("rgb24")
                   .arg("-");
                
                let mut child = cmd
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
                    .context("Failed to spawn ffmpeg process")?;

                let stdout = child.stdout.take()
                    .ok_or_else(|| anyhow!("Failed to get ffmpeg stdout"))?;

                *self.stdout.lock().unwrap() = Some(stdout);
                *process_lock = Some(child);
                
                match &self.source {
                    CameraSource::Device(_) => info!("FFmpeg continuous streaming started from camera"),
                    CameraSource::VideoFile(path) => info!("FFmpeg continuous streaming started from video file: {}", path),
                }
            }
        }

        // Read one frame from the continuous stream
        let mut stdout_lock = self.stdout.lock().unwrap();
        let stdout = stdout_lock.as_mut()
            .ok_or_else(|| anyhow!("FFmpeg stdout not available"))?;

        let mut buffer = vec![0u8; self.frame_size];

        // Read exactly one frame worth of data
        stdout.read_exact(&mut buffer)
            .context("Failed to read frame from ffmpeg stream")?;

        // Create image from raw RGB data
        let img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(self.width, self.height, buffer)
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
                    // Try to restart the stream on error
                    *self.process.lock().unwrap() = None;
                    *self.stdout.lock().unwrap() = None;
                }
            }
        }

        frames
    }

    fn is_ready(&self) -> bool {
        // Always ready - we test on creation
        true
    }

    fn backend_name(&self) -> &'static str {
        "FFmpeg-Stream"
    }
}

impl FFmpegCamera {
    pub fn is_open(&self) -> bool {
        let process_guard = self.process.lock().unwrap();
        process_guard.is_some()
    }
}

impl Drop for FFmpegCamera {
    fn drop(&mut self) {
        debug!("Stopping FFmpeg camera streaming");
        if let Some(mut child) = self.process.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        debug!("FFmpeg camera dropped");
    }
}
