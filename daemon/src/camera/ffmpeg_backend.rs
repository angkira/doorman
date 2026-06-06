//! ffmpeg-based live webcam capture backend.
//!
//! This is the reliable real-camera path for macOS development (and a portable
//! fallback on Linux). `nokhwa` 0.10's AVFoundation path is known to hang
//! (`frame()` never returns), so instead we spawn the `ffmpeg` CLI and read raw
//! RGB24 frames from its stdout pipe.
//!
//! Input device is gated by target OS:
//! - macOS:   `-f avfoundation -i "<index>"`        (capture device by index)
//! - Linux:   `-f v4l2 -i /dev/video<index>`        (V4L2 device)
//!
//! If `ffmpeg` is not installed the constructor fails so the caller falls back
//! to the mock backend. On macOS the user must also grant the terminal camera
//! permission, otherwise ffmpeg will error out when opening the device.

use anyhow::{anyhow, Result};
use doorman_shared::Config;
use image::{DynamicImage, ImageBuffer, Rgb};
use std::io::Read;
use std::process::{Child, Command, Stdio};
use tracing::{debug, info, warn};

use super::CameraBackend;

pub struct FfmpegCamera {
    width: u32,
    height: u32,
    process: Child,
}

// Safety: the `Child` is only ever touched from a single thread at a time;
// the daemon serializes camera access behind an RwLock.
unsafe impl Send for FfmpegCamera {}
unsafe impl Sync for FfmpegCamera {}

impl FfmpegCamera {
    /// Returns true if the `ffmpeg` CLI is available on PATH.
    pub fn ffmpeg_available() -> bool {
        which_ffmpeg().is_some()
    }

    fn spawn_capture(width: u32, height: u32, fps: u32, device_index: u32) -> Result<Child> {
        let ffmpeg = which_ffmpeg().ok_or_else(|| {
            anyhow!("ffmpeg CLI not found on PATH (macOS: `brew install ffmpeg`)")
        })?;

        let mut cmd = Command::new(ffmpeg);

        // Platform-gated input device.
        #[cfg(target_os = "macos")]
        {
            // AVFoundation: "<video>:<audio>"; we only want video, by index.
            cmd.arg("-f")
                .arg("avfoundation")
                .arg("-framerate")
                .arg(fps.to_string())
                .arg("-video_size")
                .arg(format!("{}x{}", width, height))
                .arg("-i")
                .arg(format!("{}", device_index));
        }

        #[cfg(target_os = "linux")]
        {
            cmd.arg("-f")
                .arg("v4l2")
                .arg("-framerate")
                .arg(fps.to_string())
                .arg("-video_size")
                .arg(format!("{}x{}", width, height))
                .arg("-i")
                .arg(format!("/dev/video{}", device_index));
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = (fps, device_index);
            return Err(anyhow!(
                "ffmpeg live capture is only wired for macOS and Linux"
            ));
        }

        // Common output: scaled raw RGB24 to stdout pipe.
        cmd.arg("-vf")
            .arg(format!("scale={}:{}", width, height))
            .arg("-pix_fmt")
            .arg("rgb24")
            .arg("-f")
            .arg("rawvideo")
            .arg("pipe:1")
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        debug!("Spawning ffmpeg live capture: {:?}", cmd);
        cmd.spawn()
            .map_err(|e| anyhow!("Failed to spawn ffmpeg: {}", e))
    }
}

impl CameraBackend for FfmpegCamera {
    async fn new_with_config(config: &Config) -> Result<Self> {
        if !Self::ffmpeg_available() {
            return Err(anyhow!(
                "ffmpeg not installed; cannot use live webcam capture (macOS: `brew install ffmpeg`)"
            ));
        }

        let width = config.camera.width;
        let height = config.camera.height;
        let fps = config.camera.fps;
        let device_index = config.camera.device_index;

        info!(
            "Initializing ffmpeg webcam backend (device {}, {}x{} @ {}fps)",
            device_index, width, height, fps
        );

        let process = Self::spawn_capture(width, height, fps, device_index)?;

        Ok(Self {
            width,
            height,
            process,
        })
    }

    fn capture_frame(&mut self) -> Result<DynamicImage> {
        let frame_size = (self.width * self.height * 3) as usize;
        let mut buffer = vec![0u8; frame_size];

        // ffmpeg may still be opening the device; if it has exited, surface that.
        if let Ok(Some(status)) = self.process.try_wait() {
            return Err(anyhow!(
                "ffmpeg capture process exited ({}); check camera permissions",
                status
            ));
        }

        let stdout = self
            .process
            .stdout
            .as_mut()
            .ok_or_else(|| anyhow!("ffmpeg has no stdout pipe"))?;

        stdout
            .read_exact(&mut buffer)
            .map_err(|e| anyhow!("Failed to read RGB frame from ffmpeg: {}", e))?;

        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(self.width, self.height, buffer)
            .ok_or_else(|| anyhow!("Failed to build image from ffmpeg buffer"))?;

        Ok(DynamicImage::ImageRgb8(img))
    }

    fn capture_frames(&mut self, count: usize) -> Vec<DynamicImage> {
        (0..count)
            .filter_map(|i| match self.capture_frame() {
                Ok(f) => Some(f),
                Err(e) => {
                    warn!("ffmpeg capture_frame {}/{} failed: {}", i + 1, count, e);
                    None
                }
            })
            .collect()
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn backend_name(&self) -> &'static str {
        "ffmpeg"
    }
}

impl Drop for FfmpegCamera {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

/// Locate the `ffmpeg` executable. Honors an explicit PATH lookup plus the
/// common Homebrew location on macOS where login-shell PATH may not apply.
fn which_ffmpeg() -> Option<String> {
    // 1. Anything on PATH.
    if let Ok(output) = Command::new("which").arg("ffmpeg").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
    }

    // 2. Common fixed locations.
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"] {
        if std::path::Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }

    None
}
