/// PipeWire camera backend with separate thread
use super::CameraBackend;
use anyhow::{anyhow, Result};
use crossbeam::channel::{bounded, Receiver, Sender};
use doorman_shared::Config;
use image::{DynamicImage, ImageBuffer, Rgb};
use pipewire as pw;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use tracing::{debug, info, warn};

pub struct PipeWireCamera {
    width: u32,
    height: u32,
    frame_rx: Receiver<Vec<u8>>,
    running: Arc<AtomicBool>,
    _thread: Option<JoinHandle<()>>,
}

impl CameraBackend for PipeWireCamera {
    async fn new_with_config(config: &Config) -> Result<Self> {
        info!("Initializing PipeWire camera backend");

        let (frame_tx, frame_rx) = bounded::<Vec<u8>>(2);
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        let width = config.camera.width;
        let height = config.camera.height;
        let fps = config.camera.fps;

        // Spawn PipeWire thread
        let thread = std::thread::spawn(move || {
            if let Err(e) = run_pipewire_loop(width, height, fps, frame_tx, running_clone) {
                warn!("PipeWire thread error: {}", e);
            }
        });

        // Wait a bit for initialization
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        info!("PipeWire camera backend ready: {}x{} @ {}fps", width, height, fps);

        Ok(Self {
            width,
            height,
            frame_rx,
            running,
            _thread: Some(thread),
        })
    }

    fn capture_frame(&mut self) -> Result<DynamicImage> {
        let frame_data = self
            .frame_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .map_err(|_| anyhow!("No frame available"))?;

        let expected = (self.width * self.height * 3) as usize;
        if frame_data.len() != expected {
            return Err(anyhow!(
                "Frame size mismatch: got {}, expected {}",
                frame_data.len(),
                expected
            ));
        }

        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(self.width, self.height, frame_data)
            .ok_or_else(|| anyhow!("Failed to create image"))?;

        Ok(DynamicImage::ImageRgb8(img))
    }

    fn capture_frames(&mut self, count: usize) -> Vec<DynamicImage> {
        let mut frames = Vec::new();
        for _ in 0..count {
            if let Ok(frame) = self.capture_frame() {
                frames.push(frame);
            }
        }
        frames
    }

    fn is_ready(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    fn backend_name(&self) -> &'static str {
        "PipeWire"
    }
}

impl Drop for PipeWireCamera {
    fn drop(&mut self) {
        debug!("Shutting down PipeWire camera");
        self.running.store(false, Ordering::Relaxed);
    }
}

fn run_pipewire_loop(
    width: u32,
    height: u32,
    fps: u32,
    frame_tx: Sender<Vec<u8>>,
    running: Arc<AtomicBool>,
) -> Result<()> {
    use pw::spa;

    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;

    // Target camera device by ID (found via pw-dump)
    // Device 35 = UGREEN Camera 4K (/dev/video0)
    let device_id_str = "35";
    info!("Connecting to PipeWire device ID: {}", device_id_str);

    // Create stream as INPUT (consume from camera device)
    let stream = pw::stream::StreamBox::new(
        &core,
        "doorman-camera",
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Camera",
            "target.object" => device_id_str,
        },
    )?;

    // User data to track format
    struct UserData {
        format: spa::param::video::VideoInfoRaw,
        tx: Sender<Vec<u8>>,
        width: u32,
        height: u32,
    }

    let user_data = UserData {
        format: Default::default(),
        tx: frame_tx.clone(),
        width,
        height,
    };

    let _listener = stream
        .add_local_listener_with_user_data(user_data)
        .state_changed(|_, _, old, new| {
            info!("PipeWire stream state: {:?} -> {:?}", old, new);
        })
        .param_changed(|_, user_data, id, param| {
            let Some(param) = param else {
                debug!("param_changed: no param");
                return;
            };
            
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }

            let (media_type, media_subtype) = match spa::param::format_utils::parse_format(param) {
                Ok(v) => v,
                Err(e) => {
                    warn!("Failed to parse format: {:?}", e);
                    return;
                }
            };

            if media_type != spa::param::format::MediaType::Video
                || media_subtype != spa::param::format::MediaSubtype::Raw
            {
                warn!("Unexpected format: {:?}/{:?}", media_type, media_subtype);
                return;
            }

            if let Err(e) = user_data.format.parse(param) {
                warn!("Failed to parse VideoInfoRaw: {:?}", e);
                return;
            }

            info!("✓ Video format negotiated:");
            info!("  Format: {:?}", user_data.format.format());
            info!("  Size: {}x{}", user_data.format.size().width, user_data.format.size().height);
            info!("  FPS: {}/{}", user_data.format.framerate().num, user_data.format.framerate().denom);
        })
        .process(move |stream, user_data| {
            if let Some(mut buffer) = stream.dequeue_buffer() {
                let datas = buffer.datas_mut();
                if let Some(data) = datas.first_mut() {
                    let size = data.chunk().size() as usize;
                    let expected = (user_data.width * user_data.height * 3) as usize;
                    let copy_size = size.min(expected);

                    if copy_size > 0 {
                        if let Some(slice) = data.data() {
                            if slice.len() >= copy_size {
                                let _ = user_data.tx.try_send(slice[..copy_size].to_vec());
                            }
                        }
                    }
                }
            }
        })
        .register()?;

    // Build format params
    let obj = pw::spa::pod::object!(
        pw::spa::utils::SpaTypes::ObjectParamFormat,
        pw::spa::param::ParamType::EnumFormat,
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::MediaType,
            Id,
            pw::spa::param::format::MediaType::Video
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::MediaSubtype,
            Id,
            pw::spa::param::format::MediaSubtype::Raw
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoFormat,
            Choice,
            Enum,
            Id,
            pw::spa::param::video::VideoFormat::YUY2,
            pw::spa::param::video::VideoFormat::YUY2,
            pw::spa::param::video::VideoFormat::UYVY,
            pw::spa::param::video::VideoFormat::I420,
            pw::spa::param::video::VideoFormat::RGB,
            pw::spa::param::video::VideoFormat::BGR,
            pw::spa::param::video::VideoFormat::RGBA,
            pw::spa::param::video::VideoFormat::BGRA,
            pw::spa::param::video::VideoFormat::RGBx,
            pw::spa::param::video::VideoFormat::BGRx
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoSize,
            Choice,
            Range,
            Rectangle,
            pw::spa::utils::Rectangle { width, height },
            pw::spa::utils::Rectangle { width: 1, height: 1 },
            pw::spa::utils::Rectangle { width: 4096, height: 4096 }
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoFramerate,
            Choice,
            Range,
            Fraction,
            pw::spa::utils::Fraction { num: fps, denom: 1 },
            pw::spa::utils::Fraction { num: 1, denom: 1 },
            pw::spa::utils::Fraction { num: 60, denom: 1 }
        ),
    );

    let values = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .map_err(|e| anyhow!("Serialize failed: {:?}", e))?
    .0
    .into_inner();

    let mut params = [spa::pod::Pod::from_bytes(&values).ok_or_else(|| anyhow!("Pod parse failed"))?];

    stream.connect(
        spa::utils::Direction::Input,
        None,
        pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
        &mut params,
    )?;

    info!("PipeWire stream connected");

    // Run mainloop (blocks)
    mainloop.run();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipewire_init() {
        let config = Config::default();
        let result = PipeWireCamera::new_with_config(&config).await;

        match result {
            Ok(camera) => {
                println!("PipeWire camera initialized");
                println!("Backend: {}", camera.backend_name());
                assert_eq!(camera.backend_name(), "PipeWire");
            }
            Err(e) => {
                eprintln!("PipeWire initialization failed: {}", e);
                eprintln!("This is expected if PipeWire is not running");
            }
        }
    }
}
