/// V4L2 camera backend - Direct hardware access (fallback)
///
/// This implementation uses direct V4L2 access via nokhwa.
/// - Simple and reliable
/// - Works without PipeWire
/// - Exclusive camera access (locks device)
/// - Fallback for systems without GStreamer/PipeWire
///
/// Note: Prefer GStreamer backend for desktop integration
use super::device_guard;
use super::format_select::{check_negotiated, select_mode, DeviceMode, WantedMode};
use super::native_frames::{
    decode_guarded, first_usable_frame, log_rejection, loop_budget, validate_frame, FrameContext,
    FrameSource, RawFrame, MAX_DETAILED_REJECTION_LOGS,
};
use super::probe::{drop_contained, validate_first_frame};
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
use nokhwa::Buffer as NokhwaBuffer;
use nokhwa::Camera as NokhwaCamera;
use std::borrow::Cow;
use std::path::Path;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

pub struct V4L2Camera {
    camera: NokhwaCamera,
    /// Format the driver reported after the stream was opened (not the request).
    ctx: FrameContext,
    /// Buffers read from the driver so far (frame index in log lines).
    frames_read: u64,
    /// Rejections logged with full detail so far (see `MAX_DETAILED_REJECTION_LOGS`).
    detailed_logs: u32,
}

impl FrameSource for V4L2Camera {
    fn next_frame(&mut self) -> Result<RawFrame<'_>> {
        self.frames_read += 1;
        // nokhwa hands out the whole mmap'd buffer and does not expose
        // `bytesused`; `validate_frame` finds the end of the JPEG itself.
        match self.camera.frame_raw().context("Failed to capture frame")? {
            Cow::Borrowed(bytes) => Ok(RawFrame {
                bytes,
                bytesused: None,
            }),
            // The V4L2 backend lends the mmap'd buffer; an owned copy would
            // mean a different nokhwa backend than the one this file targets.
            Cow::Owned(_) => Err(anyhow!("nokhwa returned an owned frame buffer")),
        }
    }
}

/// Decodes validated frame bytes to RGB with nokhwa (mozjpeg for MJPEG).
fn decode_rgb(ctx: &FrameContext, span: &[u8]) -> Result<DynamicImage> {
    let buffer = NokhwaBuffer::new(
        Resolution::new(ctx.width, ctx.height),
        span,
        to_frame_format(ctx.fourcc),
    );
    let decoded = buffer
        .decode_image::<RgbFormat>()
        .context("Failed to decode frame")?;
    let (width, height) = (decoded.width(), decoded.height());
    ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(width, height, decoded.into_raw())
        .map(DynamicImage::ImageRgb8)
        .ok_or_else(|| anyhow!("Failed to create image buffer"))
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
        let node = device_guard::resolve_node(
            config.camera.device_path.as_deref(),
            config.camera.device_index,
            Path::new(device_guard::DEV_ROOT),
            &device_guard::RealNodeFs,
        )?;
        info!("Opening camera {} ({})...", node.index, node.path.display());

        // Cheap pre-open check from sysfs; an absent sysfs entry is left to the
        // authoritative VIDIOC_QUERYCAP check below.
        let sysfs =
            device_guard::read_sysfs_facts(Path::new(device_guard::SYSFS_VIDEO4LINUX), node.index);
        if sysfs.is_some() {
            device_guard::enforce(
                &device_guard::DeviceFacts {
                    sysfs: sysfs.clone(),
                    ..Default::default()
                },
                &node,
                config.camera.allow_virtual_devices,
            )?;
        }

        let index = CameraIndex::Index(node.index);
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

        // VIDIOC_QUERYCAP as reported through nokhwa: driver, card, bus info.
        device_guard::enforce(
            &device_guard::DeviceFacts {
                driver: Some(camera.info().description().to_string()),
                card: Some(camera.info().human_name()),
                bus_info: Some(camera.info().misc()),
                sysfs,
            },
            &node,
            config.camera.allow_virtual_devices,
        )?;

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

        // The driver, not the request, decides what the buffers contain: read
        // back what it settled on and refuse to decode against another format.
        let negotiated = match camera.refresh_camera_format() {
            Ok(f) => f,
            Err(e) => {
                let _ = drop_contained(camera);
                return Err(anyhow!("cannot read back the negotiated format: {e}"));
            }
        };
        let actual = to_device_mode(&negotiated);
        if let Err(msg) = check_negotiated(&chosen, actual, &negotiated.to_string()) {
            let _ = drop_contained(camera);
            return Err(anyhow!(
                "camera index {}: {msg}",
                config.camera.device_index
            ));
        }
        let actual = actual.unwrap_or(chosen);
        let ctx = FrameContext {
            fourcc: actual.format,
            width: actual.width,
            height: actual.height,
        };

        info!(
            "V4L2 camera opened: {} {}x{} @ {}fps {:?} in {:.1} ms",
            camera.info().human_name(),
            chosen.width,
            chosen.height,
            chosen.fps,
            chosen.format,
            open_started.elapsed().as_secs_f64() * 1000.0
        );

        // An open stream proves nothing about frames (see docs/camera.md): read
        // and decode one real frame before declaring the backend usable, so a
        // device that cannot deliver falls through to the next backend.
        let timeout = Duration::from_millis(u64::from(config.camera.native_first_frame_timeout_ms));
        let first_frame_started = Instant::now();
        let warmup = config.camera.native_warmup_frames;
        let budget = loop_budget(timeout);
        let camera = Self {
            camera,
            ctx,
            frames_read: 0,
            detailed_logs: 0,
        };
        let camera = validate_first_frame(camera, timeout, move |cam| {
            let found = first_usable_frame(cam, &ctx, warmup, budget, |span| {
                decode_rgb(&ctx, span).map(|_| ())
            })?;
            info!(
                "V4L2 first usable frame: {} buffers read ({} warm-up dropped, {} rejected)",
                found.frames_read, warmup, found.rejected
            );
            cam.detailed_logs = found.rejected.min(MAX_DETAILED_REJECTION_LOGS);
            Ok(())
        })
        .with_context(|| {
            format!(
                "native camera opened at {chosen:?} but delivered no usable first frame \
                 (native_first_frame_timeout_ms = {})",
                config.camera.native_first_frame_timeout_ms
            )
        })?;
        info!(
            "V4L2 first frame validated in {:.1} ms",
            first_frame_started.elapsed().as_secs_f64() * 1000.0
        );
        Ok(camera)
    }

    fn capture_frame(&mut self) -> Result<DynamicImage> {
        let ctx = self.ctx;
        let index = self.frames_read;
        let detailed = self.detailed_logs < MAX_DETAILED_REJECTION_LOGS;
        let outcome = {
            let frame = self.next_frame()?;
            match validate_frame(&ctx, &frame)
                .and_then(|span| decode_guarded(span, &mut |s| decode_rgb(&ctx, s)))
            {
                Ok(img) => Ok(img),
                Err(why) => {
                    log_rejection(&ctx, index, &frame, &why, detailed);
                    Err(why)
                }
            }
        };
        outcome.map_err(|why| {
            self.detailed_logs = self.detailed_logs.saturating_add(1);
            anyhow!("frame {index} rejected: {why}")
        })
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
