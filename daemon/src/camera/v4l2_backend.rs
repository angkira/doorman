/// V4L2 camera backend - Direct hardware access (fallback)
///
/// This implementation uses direct V4L2 access via nokhwa.
/// - Simple and reliable
/// - Works without PipeWire
/// - Exclusive camera access (locks device)
/// - Fallback for systems without GStreamer/PipeWire
///
/// Note: Prefer GStreamer backend for desktop integration
use super::format_select::{select_mode, DeviceMode, WantedMode};
use super::CameraBackend;
use anyhow::{anyhow, Context, Result};
use doorman_shared::config::CameraFourcc;
use doorman_shared::Config;
use image::{DynamicImage, ImageBuffer, Rgb};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    Resolution,
};
use nokhwa::Camera as NokhwaCamera;
use std::time::Instant;
use tracing::{debug, info, warn};

pub struct V4L2Camera {
    camera: NokhwaCamera,
}

// Safety: nokhwa::Camera's internal state doesn't implement Send/Sync,
// but we only use it from a single thread at a time (protected by RwLock in daemon).
// The RwLock ensures exclusive access, making this safe.
unsafe impl Send for V4L2Camera {}
unsafe impl Sync for V4L2Camera {}

/// nokhwa mode -> negotiable mode; formats other than MJPEG/YUYV are not offered
/// to the selector.
fn to_device_mode(f: &CameraFormat) -> Option<DeviceMode> {
    let format = match f.format() {
        FrameFormat::MJPEG => CameraFourcc::Mjpeg,
        FrameFormat::YUYV => CameraFourcc::Yuyv,
        _ => return None,
    };
    Some(DeviceMode {
        format,
        width: f.width(),
        height: f.height(),
        fps: f.frame_rate(),
    })
}

fn to_frame_format(f: CameraFourcc) -> FrameFormat {
    match f {
        CameraFourcc::Mjpeg => FrameFormat::MJPEG,
        CameraFourcc::Yuyv => FrameFormat::YUYV,
    }
}

/// Human-readable list of the devices nokhwa can see, appended to open errors.
fn known_devices() -> String {
    match nokhwa::query(ApiBackend::Auto) {
        Ok(list) if list.is_empty() => "; no capture devices found".to_string(),
        Ok(list) => format!(
            "; devices seen: {}",
            list.iter()
                .map(|d| format!("[{}] {} ({})", d.index(), d.human_name(), d.description()))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Err(e) => format!("; device query failed: {e}"),
    }
}

impl CameraBackend for V4L2Camera {
    async fn new_with_config(config: &Config) -> Result<Self> {
        info!("Initializing V4L2 camera backend (direct hardware access)");
        info!("Opening camera {}...", config.camera.device_index);

        let index = CameraIndex::Index(config.camera.device_index);
        let wanted = WantedMode {
            width: config.camera.width,
            height: config.camera.height,
            fps: config.camera.fps,
            fourcc: config.camera.fourcc()?,
        };

        // Open the device without committing to a mode, ask it what it offers
        // and request one of those. The previous `Closest(YUYV, WxH)` request made
        // nokhwa fail with "Failed to Fufill" whenever the exact size was not
        // offered as YUYV (see docs/camera.md). Every failure carries the full
        // nokhwa error and the device list so the log says what went wrong.
        let open_started = Instant::now();
        let mut camera = NokhwaCamera::new(
            index.clone(),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
        )
        .map_err(|e| {
            anyhow!(
                "cannot open camera index {}: {e}{}",
                config.camera.device_index,
                known_devices()
            )
        })?;

        let offered = camera.compatible_camera_formats().map_err(|e| {
            anyhow!(
                "cannot list formats of camera index {}: {e}",
                config.camera.device_index
            )
        })?;
        let modes: Vec<DeviceMode> = offered.iter().filter_map(to_device_mode).collect();
        let chosen = select_mode(&modes, &wanted).map_err(|e| {
            anyhow!(
                "camera index {} ({}): {e}{}",
                config.camera.device_index,
                camera.info().human_name(),
                known_devices()
            )
        })?;
        if (chosen.width, chosen.height) != (wanted.width, wanted.height) {
            warn!(
                "Requested {}x{} is not offered; using {}x{}",
                wanted.width, wanted.height, chosen.width, chosen.height
            );
        }
        camera
            .set_camera_requset(RequestedFormat::new::<RgbFormat>(
                RequestedFormatType::Exact(CameraFormat::new(
                    Resolution::new(chosen.width, chosen.height),
                    to_frame_format(chosen.format),
                    chosen.fps,
                )),
            ))
            .map_err(|e| anyhow!("camera rejected mode {chosen:?}: {e}"))?;

        // Start the camera stream
        camera
            .open_stream()
            .map_err(|e| anyhow!("cannot start camera stream at {chosen:?}: {e}"))?;

        info!(
            "V4L2 camera opened: {} {}x{} @ {}fps {:?} in {:.1} ms",
            camera.info().human_name(),
            chosen.width,
            chosen.height,
            chosen.fps,
            chosen.format,
            open_started.elapsed().as_secs_f64() * 1000.0
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Frame size of the synthetic JPEG decode measurement.
    const BENCH_WIDTH: u32 = 640;
    const BENCH_HEIGHT: u32 = 480;
    /// Decodes averaged by the measurement.
    const BENCH_ITERATIONS: u32 = 50;
    /// JPEG quality of the synthetic frame (typical UVC output).
    const BENCH_JPEG_QUALITY: u8 = 85;

    /// Measures the MJPEG -> RGB decode nokhwa performs for every native frame
    /// (mozjpeg). Prints the number (run with `--nocapture`); no timing assert.
    #[test]
    fn mjpeg_decode_time_on_synthetic_frame() {
        let img = image::RgbImage::from_fn(BENCH_WIDTH, BENCH_HEIGHT, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, BENCH_JPEG_QUALITY)
            .encode_image(&img)
            .unwrap();

        let start = Instant::now();
        for _ in 0..BENCH_ITERATIONS {
            let rgb = nokhwa::utils::mjpeg_to_rgb(&jpeg, false).unwrap();
            assert_eq!(rgb.len(), (BENCH_WIDTH * BENCH_HEIGHT * 3) as usize);
        }
        println!(
            "mjpeg_to_rgb {}x{}: {:.2} ms/frame ({} bytes JPEG)",
            BENCH_WIDTH,
            BENCH_HEIGHT,
            start.elapsed().as_secs_f64() * 1000.0 / f64::from(BENCH_ITERATIONS),
            jpeg.len()
        );
    }
}
