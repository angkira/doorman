/// V4L2 camera backend - Direct hardware access (fallback)
///
/// This implementation uses direct V4L2 access via nokhwa.
/// - Simple and reliable
/// - Works without PipeWire
/// - Exclusive camera access (locks device)
/// - Fallback for systems without GStreamer/PipeWire
///
/// Note: Prefer GStreamer backend for desktop integration
use super::CameraBackend;
use anyhow::{anyhow, Context, Result};
use doorman_shared::Config;
use image::{DynamicImage, ImageBuffer, Rgb};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
};
use nokhwa::Camera as NokhwaCamera;
use tracing::{debug, info, warn};

pub struct V4L2Camera {
    camera: NokhwaCamera,
}

// Safety: nokhwa::Camera's internal state doesn't implement Send/Sync,
// but we only use it from a single thread at a time (protected by RwLock in daemon).
// The RwLock ensures exclusive access, making this safe.
unsafe impl Send for V4L2Camera {}
unsafe impl Sync for V4L2Camera {}

impl CameraBackend for V4L2Camera {
    async fn new_with_config(config: &Config) -> Result<Self> {
        info!("Initializing V4L2 camera backend (direct hardware access)");
        info!("Opening camera {}...", config.camera.device_index);

        let index = CameraIndex::Index(config.camera.device_index);

        // Request the EXACT mode the camera actually streams at: YUYV at the
        // configured resolution/fps. This webcam exposes some resolutions (e.g.
        // 1024x720) as YUYV only — they have no MJPEG variant. The previous
        // request (RgbFormat + AbsoluteHighestResolution, then a post-open
        // set_resolution) let nokhwa negotiate a different/incompatible mode, so
        // the very first VIDIOC_DQBUF never completed and nokhwa's frame() —
        // which polls the v4l fd with an INFINITE timeout — blocked the producer
        // thread forever, producing zero frames. Requesting the native YUYV
        // CameraFormat up front makes the first buffer arrive. Decoding to RGB is
        // unaffected: decode_image::<RgbFormat>() converts from the buffer's real
        // FrameFormat (YUYV) regardless of what we requested here.
        let resolution = Resolution::new(config.camera.width, config.camera.height);
        let camera_format =
            CameraFormat::new(resolution, FrameFormat::YUYV, config.camera.fps);
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(camera_format));

        let mut camera = NokhwaCamera::new(index, requested).context("Failed to open camera")?;

        // Start the camera stream
        camera
            .open_stream()
            .context("Failed to start camera stream")?;

        info!(
            "V4L2 camera opened: {}x{} @ {}fps",
            config.camera.width, config.camera.height, config.camera.fps
        );

        Ok(Self { camera })
    }

    fn capture_frame(&mut self) -> Result<DynamicImage> {
        let frame = self.camera.frame().context("Failed to capture frame")?;

        let decoded = frame
            .decode_image::<RgbFormat>()
            .context("Failed to decode frame")?;

        let width = decoded.width();
        let height = decoded.height();
        let data = decoded.into_raw();

        let img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(width, height, data)
            .ok_or_else(|| anyhow!("Failed to create image buffer"))?;

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
        true // V4L2 camera is always ready once initialized
    }

    fn backend_name(&self) -> &'static str {
        "V4L2 (Direct)"
    }
}

impl Drop for V4L2Camera {
    fn drop(&mut self) {
        debug!("V4L2 camera dropped");
    }
}
