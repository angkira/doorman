//! doorman-preview — live preview window for the doorman daemon.
//!
//! Connects to the daemon's two Unix sockets and renders the camera feed with a
//! bounding-box overlay:
//!   - frame socket: `[4-byte BE u32 length][JPEG payload]` repeated (RGB8 JPEG).
//!   - debug socket: newline-delimited JSON `StreamMessage` (detection info).
//!
//! Box color: GREEN when a face is recognized (recognized_user is Some AND
//! similarity >= SIMILARITY_THRESHOLD); RED otherwise (unknown face, or locked).
//! No box is drawn when bbox is null.
//!
//! Socket paths are resolved the same way the daemon resolves them:
//!   <runtime-dir>/doorman-frames.sock and <runtime-dir>/doorman-debug.sock
//! where runtime-dir = --runtime-dir CLI arg, else $XDG_RUNTIME_DIR, else the
//! /tmp/doorman-run default we run the daemon with.
//!
//! Both socket readers run on background threads, reconnect on failure, and
//! never crash the GUI if the daemon starts later.

use std::io::Read;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use doorman_shared::{DetectionInfo, StreamMessage, SIMILARITY_THRESHOLD};

/// Default runtime dir matching how we launch the daemon on this Mac.
const DEFAULT_RUNTIME_DIR: &str = "/tmp/doorman-run";

/// Latest frame handed from the frame-reader thread to the UI.
#[derive(Default)]
struct SharedState {
    /// Most recent decoded frame as (width, height, rgba bytes).
    frame: Option<(u32, u32, Vec<u8>)>,
    /// Monotonic counter so the UI knows when a new frame arrived.
    frame_seq: u64,
    /// Latest detection info from the debug socket.
    detection: Option<DetectionInfo>,
    /// Latest system_locked flag from the debug socket.
    system_locked: bool,
    /// Connection status flags for the HUD.
    frame_connected: bool,
    debug_connected: bool,
}

fn resolve_runtime_dir() -> PathBuf {
    // 1. --runtime-dir <path>
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--runtime-dir" {
            if let Some(dir) = args.next() {
                return PathBuf::from(dir);
            }
        } else if let Some(rest) = arg.strip_prefix("--runtime-dir=") {
            return PathBuf::from(rest);
        }
    }

    // 2. $XDG_RUNTIME_DIR (same env the daemon honors in --user mode).
    if let Some(dir) = std::env::var_os("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(dir);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }

    // 3. Default we run the daemon with.
    PathBuf::from(DEFAULT_RUNTIME_DIR)
}

fn read_exact_blocking(stream: &mut UnixStream, buf: &mut [u8]) -> std::io::Result<()> {
    stream.read_exact(buf)
}

/// Background thread: connect to the frame socket, read framed JPEGs, decode,
/// and publish the latest frame. Reconnects forever.
fn spawn_frame_reader(path: PathBuf, state: Arc<Mutex<SharedState>>) {
    std::thread::spawn(move || loop {
        let mut stream = match UnixStream::connect(&path) {
            Ok(s) => {
                let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
                eprintln!("[preview] frame socket connected: {}", path.display());
                if let Ok(mut st) = state.lock() {
                    st.frame_connected = true;
                }
                s
            }
            Err(_) => {
                if let Ok(mut st) = state.lock() {
                    st.frame_connected = false;
                }
                std::thread::sleep(Duration::from_millis(500));
                continue;
            }
        };

        // Read frames until the connection breaks, then reconnect.
        loop {
            let mut len_bytes = [0u8; 4];
            if read_exact_blocking(&mut stream, &mut len_bytes).is_err() {
                break;
            }
            let len = u32::from_be_bytes(len_bytes) as usize;
            if len == 0 || len > 64 * 1024 * 1024 {
                eprintln!("[preview] frame socket: implausible length {len}, reconnecting");
                break;
            }
            let mut payload = vec![0u8; len];
            if read_exact_blocking(&mut stream, &mut payload).is_err() {
                break;
            }

            match image::load_from_memory(&payload) {
                Ok(img) => {
                    let rgba = img.to_rgba8();
                    let (w, h) = rgba.dimensions();
                    if let Ok(mut st) = state.lock() {
                        st.frame = Some((w, h, rgba.into_raw()));
                        st.frame_seq = st.frame_seq.wrapping_add(1);
                    }
                }
                Err(e) => eprintln!("[preview] JPEG decode failed: {e}"),
            }
        }

        if let Ok(mut st) = state.lock() {
            st.frame_connected = false;
        }
        eprintln!("[preview] frame socket disconnected, retrying...");
        std::thread::sleep(Duration::from_millis(500));
    });
}

/// Background thread: connect to the debug socket, parse newline-delimited
/// StreamMessage JSON, keep the latest detection + system_locked. Reconnects.
fn spawn_debug_reader(path: PathBuf, state: Arc<Mutex<SharedState>>) {
    std::thread::spawn(move || loop {
        let stream = match UnixStream::connect(&path) {
            Ok(s) => {
                eprintln!("[preview] debug socket connected: {}", path.display());
                if let Ok(mut st) = state.lock() {
                    st.debug_connected = true;
                }
                s
            }
            Err(_) => {
                if let Ok(mut st) = state.lock() {
                    st.debug_connected = false;
                }
                std::thread::sleep(Duration::from_millis(500));
                continue;
            }
        };

        let mut reader = std::io::BufReader::new(stream);
        let mut line = String::new();
        loop {
            line.clear();
            use std::io::BufRead;
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<StreamMessage>(trimmed) {
                        Ok(StreamMessage::Detection {
                            detection,
                            system_locked,
                            ..
                        }) => {
                            if let Ok(mut st) = state.lock() {
                                st.detection = Some(detection);
                                st.system_locked = system_locked;
                            }
                        }
                        Ok(StreamMessage::Enrollment { .. }) => { /* ignored in preview */ }
                        Err(e) => eprintln!("[preview] debug JSON parse error: {e}"),
                    }
                }
                Err(_) => break,
            }
        }

        if let Ok(mut st) = state.lock() {
            st.debug_connected = false;
        }
        eprintln!("[preview] debug socket disconnected, retrying...");
        std::thread::sleep(Duration::from_millis(500));
    });
}

struct PreviewApp {
    state: Arc<Mutex<SharedState>>,
    texture: Option<egui::TextureHandle>,
    last_frame_seq: u64,
    // FPS tracking for the displayed video.
    last_fps_instant: Instant,
    frames_since: u32,
    fps: f32,
}

impl PreviewApp {
    fn new(state: Arc<Mutex<SharedState>>) -> Self {
        Self {
            state,
            texture: None,
            last_frame_seq: 0,
            last_fps_instant: Instant::now(),
            frames_since: 0,
            fps: 0.0,
        }
    }
}

impl eframe::App for PreviewApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Repaint continuously so new frames show up promptly.
        ctx.request_repaint_after(Duration::from_millis(16));

        // Snapshot shared state quickly to minimize lock hold time.
        let (frame_data, frame_seq, detection, system_locked, frame_conn, debug_conn) = {
            let st = self.state.lock().unwrap();
            (
                st.frame.clone(),
                st.frame_seq,
                st.detection.clone(),
                st.system_locked,
                st.frame_connected,
                st.debug_connected,
            )
        };

        // Upload a new texture only when the frame changed.
        if frame_seq != self.last_frame_seq {
            if let Some((w, h, rgba)) = &frame_data {
                let image = egui::ColorImage::from_rgba_unmultiplied([*w as usize, *h as usize], rgba);
                match &mut self.texture {
                    Some(tex) => tex.set(image, egui::TextureOptions::LINEAR),
                    None => {
                        self.texture =
                            Some(ctx.load_texture("camera-frame", image, egui::TextureOptions::LINEAR));
                    }
                }
            }
            self.last_frame_seq = frame_seq;

            // FPS update.
            self.frames_since += 1;
            let elapsed = self.last_fps_instant.elapsed().as_secs_f32();
            if elapsed >= 0.5 {
                self.fps = self.frames_since as f32 / elapsed;
                self.frames_since = 0;
                self.last_fps_instant = Instant::now();
            }
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::BLACK))
            .show(ctx, |ui| {
                let avail = ui.available_rect_before_wrap();

                // Draw the video frame scaled to fit (letterboxed) inside the window.
                let image_rect = if let Some(tex) = &self.texture {
                    let tex_size = tex.size_vec2();
                    let scale = (avail.width() / tex_size.x).min(avail.height() / tex_size.y);
                    let draw_size = tex_size * scale;
                    let rect = egui::Rect::from_center_size(avail.center(), draw_size);
                    ui.painter().image(
                        tex.id(),
                        rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                    Some(rect)
                } else {
                    ui.painter().text(
                        avail.center(),
                        egui::Align2::CENTER_CENTER,
                        "Waiting for frames…",
                        egui::FontId::proportional(24.0),
                        egui::Color32::GRAY,
                    );
                    None
                };

                // Overlay the bbox if present.
                if let (Some(rect), Some(det)) = (image_rect, &detection) {
                    if let (Some((bx, by, bw, bh)), Some((fw, fh))) = (det.bbox, det.frame_size) {
                        if fw > 0 && fh > 0 {
                            let sx = rect.width() / fw as f32;
                            let sy = rect.height() / fh as f32;
                            let box_rect = egui::Rect::from_min_size(
                                egui::pos2(rect.min.x + bx as f32 * sx, rect.min.y + by as f32 * sy),
                                egui::vec2(bw as f32 * sx, bh as f32 * sy),
                            );

                            // GREEN iff recognized AND similarity >= threshold AND not locked.
                            let recognized = det.recognized_user.is_some()
                                && det.similarity.unwrap_or(0.0) >= SIMILARITY_THRESHOLD
                                && !system_locked;
                            let color = if recognized {
                                egui::Color32::from_rgb(0, 220, 0)
                            } else {
                                egui::Color32::from_rgb(230, 30, 30)
                            };
                            ui.painter().rect_stroke(
                                box_rect,
                                2.0,
                                egui::Stroke::new(3.0, color),
                            );

                            // Label above the box.
                            let name = det.recognized_user.clone().unwrap_or_else(|| "unknown".into());
                            let sim = det.similarity.unwrap_or(0.0) * 100.0;
                            ui.painter().text(
                                egui::pos2(box_rect.min.x, (box_rect.min.y - 18.0).max(rect.min.y)),
                                egui::Align2::LEFT_TOP,
                                format!("{name} {sim:.0}%"),
                                egui::FontId::monospace(16.0),
                                color,
                            );
                        }
                    }
                }

                // HUD text in the top-left.
                let det_ref = detection.as_ref();
                let user = det_ref
                    .and_then(|d| d.recognized_user.clone())
                    .unwrap_or_else(|| "unknown".into());
                let sim = det_ref.and_then(|d| d.similarity).unwrap_or(0.0) * 100.0;
                let conf = det_ref.and_then(|d| d.confidence).unwrap_or(0.0) * 100.0;
                let hud = format!(
                    "user: {user}   sim: {sim:.1}%   conf: {conf:.1}%   locked: {system_locked}   fps: {:.1}\nframe socket: {}   debug socket: {}",
                    self.fps,
                    if frame_conn { "connected" } else { "waiting" },
                    if debug_conn { "connected" } else { "waiting" },
                );
                let hud_pos = avail.min + egui::vec2(8.0, 8.0);
                // Shadow for readability over bright video.
                ui.painter().text(
                    hud_pos + egui::vec2(1.0, 1.0),
                    egui::Align2::LEFT_TOP,
                    &hud,
                    egui::FontId::monospace(15.0),
                    egui::Color32::BLACK,
                );
                ui.painter().text(
                    hud_pos,
                    egui::Align2::LEFT_TOP,
                    &hud,
                    egui::FontId::monospace(15.0),
                    egui::Color32::from_rgb(0, 255, 120),
                );
            });
    }
}

fn main() -> eframe::Result<()> {
    let runtime_dir = resolve_runtime_dir();
    let frame_path = runtime_dir.join("doorman-frames.sock");
    let debug_path = runtime_dir.join("doorman-debug.sock");

    eprintln!("[preview] runtime dir: {}", runtime_dir.display());
    eprintln!("[preview] frame socket: {}", frame_path.display());
    eprintln!("[preview] debug socket: {}", debug_path.display());

    let state = Arc::new(Mutex::new(SharedState::default()));
    spawn_frame_reader(frame_path, state.clone());
    spawn_debug_reader(debug_path, state.clone());

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("doorman preview")
            .with_inner_size([1024.0, 640.0]),
        ..Default::default()
    };

    eframe::run_native(
        "doorman-preview",
        options,
        Box::new(move |_cc| Ok(Box::new(PreviewApp::new(state)))),
    )
}
