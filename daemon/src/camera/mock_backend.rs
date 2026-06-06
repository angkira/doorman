//! Cross-platform mock camera backend.
//!
//! This is the default camera on macOS and inside CI containers. It needs no
//! system camera and no native libraries:
//!
//! - If constructed with a video file (`MockCamera::from_video_file`) and the
//!   `ffmpeg` CLI is available, it decodes/loops that file into RGB frames.
//! - Otherwise (no video file, or ffmpeg missing/failed) it emits a synthetic
//!   animated test image so the full pipeline + preview still run.
//!
//! Frame production is rate-limited to the configured FPS.

use anyhow::{anyhow, Result};
use image::{DynamicImage, ImageBuffer, Rgb};
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use super::CameraBackend;

pub struct MockCamera {
    width: u32,
    height: u32,
    fps: u32,
    frame_interval: Duration,
    last_frame_time: Mutex<Instant>,
    frame_counter: Mutex<u64>,
    /// Optional video-file source. When set we try to decode it via ffmpeg.
    video_path: Option<PathBuf>,
    loop_playback: bool,
    process: Mutex<Option<Child>>,
    /// Set once ffmpeg proves unavailable so we stop retrying and use synthetic frames.
    ffmpeg_failed: Mutex<bool>,
}

impl MockCamera {
    /// Synthetic-only mock camera (no video source).
    pub fn synthetic(width: u32, height: u32, fps: u32) -> Self {
        info!("Initializing mock camera backend (synthetic frames {}x{} @ {}fps)", width, height, fps);
        Self::build(width, height, fps, None, false)
    }

    /// Mock camera that loops a video file through the `ffmpeg` CLI when available,
    /// falling back to synthetic frames otherwise.
    pub fn from_video_file(video_path: PathBuf, width: u32, height: u32, fps: u32, loop_playback: bool) -> Self {
        info!("Initializing mock camera backend (video file {:?}, {}x{} @ {}fps, loop={})",
            video_path, width, height, fps, loop_playback);
        Self::build(width, height, fps, Some(video_path), loop_playback)
    }

    fn build(width: u32, height: u32, fps: u32, video_path: Option<PathBuf>, loop_playback: bool) -> Self {
        let fps = fps.max(1);
        Self {
            width,
            height,
            fps,
            frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
            last_frame_time: Mutex::new(Instant::now()),
            frame_counter: Mutex::new(0),
            video_path,
            loop_playback,
            process: Mutex::new(None),
            ffmpeg_failed: Mutex::new(false),
        }
    }

    fn rate_limit(&self) {
        let now = Instant::now();
        let mut last_time = self.last_frame_time.lock().unwrap();
        let elapsed = now.duration_since(*last_time);
        if elapsed < self.frame_interval {
            std::thread::sleep(self.frame_interval - elapsed);
        }
        *last_time = Instant::now();
    }

    /// Generate a deterministic synthetic frame: a moving gradient background
    /// with a centered "face-like" lighter rectangle so detectors have something
    /// plausible to chew on during development.
    fn synthetic_frame(&self) -> DynamicImage {
        let mut counter = self.frame_counter.lock().unwrap();
        let t = *counter;
        *counter = counter.wrapping_add(1);
        drop(counter);

        let (w, h) = (self.width.max(1), self.height.max(1));
        let phase = (t % 256) as u8;

        let face_w = w / 3;
        let face_h = h / 2;
        let face_x0 = (w / 2).saturating_sub(face_w / 2);
        let face_y0 = (h / 2).saturating_sub(face_h / 2);
        let face_x1 = face_x0 + face_w;
        let face_y1 = face_y0 + face_h;

        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_fn(w, h, |x, y| {
            if x >= face_x0 && x < face_x1 && y >= face_y0 && y < face_y1 {
                // Lighter "face" region
                Rgb([200u8, 180, 160])
            } else {
                let r = ((x.wrapping_add(t as u32)) % 256) as u8;
                let g = ((y.wrapping_add(t as u32)) % 256) as u8;
                Rgb([r, g, phase])
            }
        });

        DynamicImage::ImageRgb8(img)
    }

    fn start_ffmpeg(&self, video_path: &PathBuf) -> Result<Child> {
        let mut cmd = Command::new("ffmpeg");
        if self.loop_playback {
            cmd.arg("-stream_loop").arg("-1");
        }
        cmd.arg("-i").arg(video_path)
            .arg("-vf").arg(format!("scale={}:{}", self.width, self.height))
            .arg("-f").arg("rawvideo")
            .arg("-pix_fmt").arg("rgb24")
            .arg("-r").arg(self.fps.to_string())
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        debug!("Starting ffmpeg for mock camera: {:?}", cmd);
        cmd.spawn().map_err(|e| anyhow!("Failed to spawn ffmpeg: {}", e))
    }

    /// Try to read one frame from the ffmpeg pipe. Returns Ok(None) if ffmpeg is
    /// unavailable so the caller can fall back to synthetic frames.
    fn try_video_frame(&self, video_path: &PathBuf) -> Result<Option<DynamicImage>> {
        if *self.ffmpeg_failed.lock().unwrap() {
            return Ok(None);
        }

        // Ensure a running process.
        {
            let mut guard = self.process.lock().unwrap();
            let needs_start = match guard.as_mut() {
                Some(child) => match child.try_wait() {
                    Ok(None) => false,
                    Ok(Some(status)) => {
                        if self.loop_playback {
                            warn!("ffmpeg exited ({}), restarting", status);
                            true
                        } else {
                            return Err(anyhow!("Video playback finished"));
                        }
                    }
                    Err(e) => {
                        warn!("ffmpeg status check failed ({}), restarting", e);
                        true
                    }
                },
                None => true,
            };

            if needs_start {
                match self.start_ffmpeg(video_path) {
                    Ok(child) => *guard = Some(child),
                    Err(e) => {
                        warn!("ffmpeg unavailable ({}); using synthetic frames", e);
                        *self.ffmpeg_failed.lock().unwrap() = true;
                        return Ok(None);
                    }
                }
            }
        }

        let frame_size = (self.width * self.height * 3) as usize;
        let mut buffer = vec![0u8; frame_size];

        let mut guard = self.process.lock().unwrap();
        let child = guard.as_mut().ok_or_else(|| anyhow!("No ffmpeg process"))?;
        let stdout = child.stdout.as_mut().ok_or_else(|| anyhow!("No ffmpeg stdout"))?;
        stdout
            .read_exact(&mut buffer)
            .map_err(|e| anyhow!("Failed to read frame from ffmpeg: {}", e))?;

        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(self.width, self.height, buffer)
            .ok_or_else(|| anyhow!("Failed to build image from ffmpeg buffer"))?;
        Ok(Some(DynamicImage::ImageRgb8(img)))
    }

    pub fn is_open(&self) -> bool {
        true
    }
}

impl CameraBackend for MockCamera {
    async fn new_with_config(config: &doorman_shared::Config) -> Result<Self> {
        Ok(Self::synthetic(config.camera.width, config.camera.height, config.camera.fps))
    }

    fn capture_frame(&mut self) -> Result<DynamicImage> {
        self.rate_limit();

        if let Some(path) = self.video_path.clone() {
            if let Some(frame) = self.try_video_frame(&path)? {
                return Ok(frame);
            }
        }

        Ok(self.synthetic_frame())
    }

    fn capture_frames(&mut self, count: usize) -> Vec<DynamicImage> {
        (0..count).filter_map(|_| self.capture_frame().ok()).collect()
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn backend_name(&self) -> &'static str {
        "Mock"
    }
}

impl Drop for MockCamera {
    fn drop(&mut self) {
        if let Some(mut child) = self.process.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
