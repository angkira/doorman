use super::types::RawFrame;
use crate::camera::gate::WarmupGate;
use crate::camera::Camera;
use doorman_shared::Config;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch};
use tracing::{debug, info, warn};

/// How often the idle (gated-off) producer re-checks the capture gate. This
/// bounds the delay between the gate turning on and the camera open starting.
const GATE_IDLE_POLL: Duration = Duration::from_millis(20);

/// Timestamps of one cold open, from the start of the open call.
struct ColdOpen {
    started: Instant,
    open: Duration,
    first_frame: Option<Duration>,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// The one info line per open that feeds the latency analysis (WS6).
fn open_summary(
    backend: &str,
    resolution: (u32, u32),
    fourcc: &str,
    t: (Duration, Duration, Duration),
) -> String {
    format!(
        "Camera ready: backend={backend} resolution={}x{} fourcc={fourcc} \
         open_ms={:.1} first_frame_ms={:.1} first_valid_ms={:.1}",
        resolution.0,
        resolution.1,
        ms(t.0),
        ms(t.1),
        ms(t.2)
    )
}

/// Describes how the producer thread should open the camera.
///
/// The camera MUST be opened on the same thread that captures from it: the V4L2
/// backend's `open_stream()` sets up mmap buffers bound to the calling thread's
/// fd context, and calling `frame()` (VIDIOC_DQBUF) from a different thread is
/// undefined behavior. Opening on the main tokio thread and capturing on the
/// producer thread corrupted the runtime (the previous bug). So the producer
/// thread opens the camera itself from this description.
pub enum CameraSource {
    /// Open the live camera using the daemon config.
    Config,
    /// Replay a video file (used for tests / `--video-file`).
    VideoFile {
        path: std::path::PathBuf,
        width: u32,
        height: u32,
        fps: u32,
        loop_playback: bool,
    },
}

/// Spawn the single-owner camera producer.
///
/// The producer runs on a dedicated OS thread that takes ownership of the
/// `Camera` by value (moved in exactly once) and is the ONLY thread that ever
/// touches it. This is the root-cause fix for the previous design, which shared
/// the `Camera` across tokio worker threads via `Arc<RwLock<Option<Camera>>>`.
/// The `Camera` contains a non-`Send` V4L2 `MmapStream`; sharing it across
/// threads was undefined behavior and caused the stream to be torn down on a
/// foreign thread (VIDIOC_STREAMOFF), closing the frame channel and killing the
/// daemon ~0.5s after launch.
///
/// The thread captures frames in a plain blocking loop and:
///  - publishes the latest frame to `latest_frame_tx` (read by IPC consumers:
///    enroll/auth/status/detect — they no longer touch the camera directly), and
///  - forwards each frame as a `RawFrame` to the pipeline via `frame_tx`.
///
/// It exits only on a fatal condition (no camera and none could be opened) or
/// when all receivers are gone.
pub fn spawn_camera_producer(
    source: CameraSource,
    config: Arc<Config>,
    latest_frame_tx: watch::Sender<Option<Arc<image::DynamicImage>>>,
    frame_tx: mpsc::Sender<RawFrame>,
    // Capture gate: when `false` the LIVE camera is RELEASED (V4L2 device closed,
    // LED off) and the thread idles at ~0 CPU; when `true` it (re)opens and
    // captures at the configured fps. Driven from `armed || preview || always_track`
    // so an unlocked desktop pays nothing for the camera. (Video-file sources ignore
    // the gate and always replay.)
    capture_active: watch::Receiver<bool>,
) {
    let open_config = config.clone();
    spawn_camera_producer_with(
        source,
        config,
        latest_frame_tx,
        frame_tx,
        capture_active,
        move || futures::executor::block_on(Camera::new_with_config(&open_config)),
    );
}

/// [`spawn_camera_producer`] with an injectable camera opener (tests use a
/// synthetic camera so no device or ffmpeg is needed).
fn spawn_camera_producer_with(
    source: CameraSource,
    config: Arc<Config>,
    latest_frame_tx: watch::Sender<Option<Arc<image::DynamicImage>>>,
    frame_tx: mpsc::Sender<RawFrame>,
    capture_active: watch::Receiver<bool>,
    mut open_camera: impl FnMut() -> anyhow::Result<Camera> + Send + 'static,
) {
    let fps = config.camera.fps;
    info!("Camera producer thread started (target {}fps; opens the camera only while capture is gated ON)", fps);

    std::thread::spawn(move || {
        let interval_ms = 1000 / fps.max(1) as u64;
        let mut sequence = 0u64;
        let mut last_capture = Instant::now();
        // Set on every cold open; `None` while the stream is warm (video file, or
        // once this open's first valid frame was published).
        let mut cold: Option<ColdOpen> = None;
        let mut warmup = WarmupGate::new(0);
        let fourcc_label = config
            .camera
            .fourcc
            .clone()
            .unwrap_or_else(|| "auto".to_string());
        const REPROBE_SECS: u64 = 3;

        // VideoFile (test/replay) is opened once and is NOT gated. The live Config
        // camera is opened lazily when capture is active and released when gated off.
        let is_video = matches!(source, CameraSource::VideoFile { .. });
        let mut camera: Option<Camera> = match &source {
            CameraSource::VideoFile { path, width, height, fps, loop_playback } => {
                Some(Camera::from_video_file(path.clone(), *width, *height, *fps, *loop_playback))
            }
            CameraSource::Config => None,
        };

        loop {
            // Graceful shutdown: every receiver gone.
            if frame_tx.is_closed() && latest_frame_tx.is_closed() {
                debug!("All frame receivers gone, camera producer exiting");
                return;
            }

            // Capture gate. Video always captures; the live camera follows the gate.
            let active = is_video || *capture_active.borrow();
            if !active {
                // Release the live camera (closes the device / LED off) and idle.
                if camera.is_some() {
                    info!("Camera gated OFF (screen unlocked / no preview) — releasing device");
                    camera = None;
                    let _ = latest_frame_tx.send(None);
                }
                std::thread::sleep(GATE_IDLE_POLL);
                continue;
            }

            // Active: ensure the live camera is open; retry with backoff (a missing
            // or just-reconnected camera must not kill the thread).
            if camera.is_none() {
                let open_started = Instant::now();
                match open_camera() {
                    Ok(cam) => {
                        info!("Camera gated ON — opened in producer thread");
                        camera = Some(cam);
                        last_capture = Instant::now();
                        // Cold open: restart the warm-up discard and the timing.
                        warmup = WarmupGate::new(config.camera.warmup_frames);
                        cold = Some(ColdOpen {
                            started: open_started,
                            open: open_started.elapsed(),
                            first_frame: None,
                        });
                    }
                    Err(e) => {
                        warn!("Camera not available ({}); will re-probe in {}s", e, REPROBE_SECS);
                        std::thread::sleep(std::time::Duration::from_secs(REPROBE_SECS));
                        continue;
                    }
                }
            }

            // Rate limit captures.
            let elapsed = last_capture.elapsed().as_millis() as u64;
            if elapsed < interval_ms {
                std::thread::sleep(std::time::Duration::from_millis(interval_ms - elapsed));
            }
            last_capture = std::time::Instant::now();

            // Blocking capture is fine: this dedicated OS thread is the sole owner.
            let frame = match camera.as_mut().unwrap().capture_frame() {
                Ok(f) => f,
                Err(e) => {
                    warn!("Capture error: {} — releasing camera to re-open", e);
                    camera = None; // drop + re-open next active iteration (handles disconnect)
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
            };

            if let Some(c) = cold.as_mut() {
                if c.first_frame.is_none() {
                    c.first_frame = Some(c.started.elapsed());
                }
            }

            // Drop the first frames after a cold open (auto-exposure settling).
            if warmup.discard_next() {
                debug!("Discarding warm-up frame");
                continue;
            }

            if let Some(c) = cold.take() {
                info!(
                    "{}",
                    open_summary(
                        camera.as_ref().map_or("none", |cam| cam.backend_name()),
                        (frame.width(), frame.height()),
                        &fourcc_label,
                        (
                            c.open,
                            c.first_frame.unwrap_or_default(),
                            c.started.elapsed()
                        ),
                    )
                );
            }

            sequence += 1;
            let image = Arc::new(frame);

            // Publish latest frame for IPC consumers (ignore if no receivers).
            let _ = latest_frame_tx.send(Some(image.clone()));

            let raw_frame = RawFrame {
                image,
                timestamp: std::time::Instant::now(),
                sequence,
            };

            // Non-blocking send to the pipeline (drop frame if channel full).
            if let Err(mpsc::error::TrySendError::Closed(_)) = frame_tx.try_send(raw_frame) {
                debug!("Pipeline frame channel closed");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    const TEST_FPS: u32 = 200;
    const TEST_SIZE: (u32, u32) = (64, 48);
    /// Upper bound for any wait in these tests.
    const TEST_TIMEOUT: Duration = Duration::from_secs(5);
    /// Poll interval while waiting in these tests.
    const TEST_POLL: Duration = Duration::from_millis(2);

    struct Harness {
        rx: mpsc::Receiver<RawFrame>,
        latest: watch::Receiver<Option<Arc<image::DynamicImage>>>,
        gate: watch::Sender<bool>,
        opens: Arc<AtomicU32>,
    }

    fn start(warmup_frames: u32) -> Harness {
        let mut config = Config::default();
        config.camera.fps = TEST_FPS;
        config.camera.warmup_frames = warmup_frames;
        let (latest_tx, latest) = watch::channel(None);
        let (tx, rx) = mpsc::channel(64);
        let (gate, gate_rx) = watch::channel(true);
        let opens = Arc::new(AtomicU32::new(0));
        let opens_in = opens.clone();
        spawn_camera_producer_with(
            CameraSource::Config,
            Arc::new(config),
            latest_tx,
            tx,
            gate_rx,
            move || {
                opens_in.fetch_add(1, Ordering::SeqCst);
                Ok(Camera::synthetic(TEST_SIZE.0, TEST_SIZE.1, TEST_FPS))
            },
        );
        Harness {
            rx,
            latest,
            gate,
            opens,
        }
    }

    /// The synthetic camera encodes its frame counter in the red channel of
    /// pixel (0, 0), so this is the index of the frame inside one open.
    fn frame_index(f: &RawFrame) -> u8 {
        f.image.to_rgb8().get_pixel(0, 0).0[0]
    }

    fn recv(h: &mut Harness) -> RawFrame {
        let deadline = Instant::now() + TEST_TIMEOUT;
        loop {
            match h.rx.try_recv() {
                Ok(f) => return f,
                Err(e) if Instant::now() >= deadline => {
                    panic!("no frame within {TEST_TIMEOUT:?}: {e}")
                }
                Err(_) => std::thread::sleep(TEST_POLL),
            }
        }
    }

    #[test]
    fn cold_open_drops_warmup_frames_and_warm_stream_drops_none() {
        let mut h = start(3);
        let got: Vec<(u64, u8)> = (0..4)
            .map(|_| {
                let f = recv(&mut h);
                (f.sequence, frame_index(&f))
            })
            .collect();
        // Frames 0..3 were discarded; after that every frame is delivered.
        assert_eq!(got, [(1, 3), (2, 4), (3, 5), (4, 6)]);
        assert_eq!(h.opens.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn zero_warmup_delivers_the_first_frame() {
        let mut h = start(0);
        assert_eq!(frame_index(&recv(&mut h)), 0);
    }

    #[test]
    fn gate_off_releases_camera_and_next_cold_open_discards_again() {
        let mut h = start(2);
        assert_eq!(frame_index(&recv(&mut h)), 2);

        h.gate.send(false).unwrap();
        let deadline = Instant::now() + TEST_TIMEOUT;
        while h.latest.borrow().is_some() {
            assert!(Instant::now() < deadline, "camera was not released");
            std::thread::sleep(TEST_POLL);
        }
        while h.rx.try_recv().is_ok() {}

        h.gate.send(true).unwrap();
        assert_eq!(
            frame_index(&recv(&mut h)),
            2,
            "second cold open warms up again"
        );
        assert_eq!(h.opens.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn open_summary_has_all_fields() {
        let line = open_summary(
            "ffmpeg",
            (640, 480),
            "MJPG",
            (
                Duration::from_millis(12),
                Duration::from_millis(340),
                Duration::from_millis(410),
            ),
        );
        assert_eq!(
            line,
            "Camera ready: backend=ffmpeg resolution=640x480 fourcc=MJPG \
             open_ms=12.0 first_frame_ms=340.0 first_valid_ms=410.0"
        );
    }
}
