//! doorman-depthd — depth-liveness SIDECAR.
//!
//! WHY THIS EXISTS: loading the Depth-Anything-V2 model on the ONNX Runtime CPU
//! EP inside the SAME process as the daemon's ROCm/CUDA-EP sessions SEGVs (ORT's
//! global `OrtEnv` cannot mix a CPU session with GPU sessions). So when the
//! `doormand` daemon runs on a GPU EP it does NOT load the depth model; instead
//! it forwards each frame+bbox to THIS process over a Unix socket. This binary
//! loads ONLY the Depth-Anything-V2 model on the CPU EP, computes the SAME
//! depth-relief PAD score the in-process gate uses (via the shared
//! `compute_depth_relief`), and replies with the relief.
//!
//! EXECUTION PROVIDER (depth EP):
//!   The depth session runs on the CPU EP by DEFAULT (portable, zero system deps).
//!   On a ROCm install it can instead run on the iGPU (ROCm EP) so the CPU stays
//!   light during auth. The EP is chosen at RUNTIME by the `DOORMAN_DEPTH_EP` env
//!   var (or `--depth-ep <ep>` CLI flag): `cpu` (default) | `rocm`.
//!
//!   A separate ROCm process for depth is SAFE: the SEGV the sidecar exists to
//!   avoid was a CPU depth session co-loaded with the daemon's ROCm sessions in
//!   ONE `OrtEnv`. This sidecar is a DISTINCT process loading ONLY ROCm sessions;
//!   the daemon is a distinct process loading ONLY ROCm sessions — two pure-ROCm
//!   processes share the iGPU at the HIP/driver level. ROCm depth requires a build
//!   WITH the ROCm EP compiled in:
//!     cargo build --release --no-default-features \
//!       --features backend-ort-rocm,camera-v4l2,camera-ffmpeg --bin doorman-depthd
//!   Default CPU build (any platform):
//!     cargo build --release --no-default-features \
//!       --features backend-ort,camera-v4l2,camera-ffmpeg --bin doorman-depthd
//!   If `DOORMAN_DEPTH_EP=rocm` but the binary was built without `backend-ort-rocm`,
//!   the ROCm EP is never registered and the session silently falls back to CPU —
//!   so set the EP env AND build with the matching feature.
//!
//! USAGE:
//!   doorman-depthd [--socket <path>] [--models-dir <dir>] [--user] [--config <f>]
//!
//! Socket path resolution (first wins):
//!   --socket arg  >  DOORMAN_DEPTH_SOCKET env  >  runtime-dir default
//!   (`<dir of daemon command socket>/doorman-depth.sock`, i.e.
//!    /run/doorman/doorman-depth.sock in system mode, or XDG_RUNTIME_DIR in
//!    --user mode).

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;

use anyhow::{Context, Result};
use doorman_shared::Config;
use doormand::ml::depth_ipc;

#[cfg(feature = "_ort")]
use doormand::ml::ort_backend::{self, DepthSession};

fn arg_value(flag: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != flag).nth(1)
}

fn main() -> Result<()> {
    // Minimal logging (mirrors the daemon's default targets so EP registration
    // and load errors are visible).
    let log_level = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| "doorman_depthd=info,doormand=info,ort=info".to_string());
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("RUST_LOG")
                .unwrap_or_else(|_| log_level.into()),
        )
        .init();

    let user_mode = std::env::args().any(|a| a == "--user");
    let config_file = arg_value("--config").map(PathBuf::from);

    // Load config (same precedence and root trust rules as the daemon).
    let mut config = Config::load_with(config_file.as_deref())
        .map_err(|e| anyhow::anyhow!("refusing to start: invalid configuration: {e}"))?;

    // Mirror the daemon's --user path rewriting so the sidecar's socket + models
    // dir land in the SAME place the daemon expects.
    if user_mode {
        if let Some(rt) = std::env::var_os("XDG_RUNTIME_DIR") {
            config.daemon.socket_path = PathBuf::from(rt)
                .join("doorman.sock")
                .to_string_lossy()
                .to_string();
        }
        if let Some(data) = std::env::var_os("XDG_DATA_HOME") {
            config.daemon.data_dir = PathBuf::from(data)
                .join("doorman")
                .to_string_lossy()
                .to_string();
        } else if let Some(home) = std::env::var_os("HOME") {
            config.daemon.data_dir = PathBuf::from(home)
                .join(".local/share/doorman")
                .to_string_lossy()
                .to_string();
        }
        config.ml.models_dir = PathBuf::from(&config.daemon.data_dir)
            .join("models")
            .to_string_lossy()
            .to_string();
    }

    if let Some(dir) = arg_value("--models-dir") {
        config.ml.models_dir = dir;
    }

    // Resolve the socket path: --socket > DOORMAN_DEPTH_SOCKET > runtime-dir
    // default derived from the daemon command socket's directory.
    let default_sock = {
        let cmd = std::path::Path::new(&config.daemon.socket_path);
        match cmd.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => {
                dir.join("doorman-depth.sock").to_string_lossy().to_string()
            }
            _ => depth_ipc::DEFAULT_DEPTH_SOCKET.to_string(),
        }
    };
    let socket_path =
        arg_value("--socket").unwrap_or_else(|| depth_ipc::resolve_socket_path(&default_sock));

    // Depth EP: --depth-ep flag > DOORMAN_DEPTH_EP env > "cpu" default.
    let depth_ep = arg_value("--depth-ep")
        .or_else(|| std::env::var("DOORMAN_DEPTH_EP").ok())
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "cpu".to_string());

    run(&config, &socket_path, &depth_ep)
}

#[cfg(feature = "_ort")]
fn run(config: &Config, socket_path: &str, depth_ep: &str) -> Result<()> {
    let models_dir = PathBuf::from(&config.ml.models_dir);

    let want_rocm = matches!(depth_ep, "rocm" | "gpu");
    tracing::info!(
        "doorman-depthd starting (depth EP: {})",
        if want_rocm { "rocm/iGPU" } else { "cpu" }
    );
    tracing::info!("  models_dir: {}", models_dir.display());
    tracing::info!("  socket:     {}", socket_path);

    // Load ONLY the depth model on the selected EP. A missing model is NOT fatal:
    // the sidecar still serves and replies RESP_NONE so the daemon applies its
    // fail-safe reject (rather than the sidecar silently dying).
    let load_result = if want_rocm {
        ort_backend::load_depth_session_rocm(&models_dir, config)
    } else {
        ort_backend::load_depth_session_cpu(&models_dir, config)
    };
    let depth: Option<DepthSession> = match load_result {
        Ok(sess) => {
            tracing::info!(
                "✓ Loaded Depth-Anything-V2 ({} EP)",
                if want_rocm { "ROCm" } else { "CPU" }
            );
            Some(sess)
        }
        Err(e) => {
            tracing::error!("✗ Depth model not loaded: {} — will reply RESP_NONE", e);
            None
        }
    };

    // Bind the socket (remove a stale one first).
    let _ = std::fs::remove_file(socket_path);
    if let Some(parent) = std::path::Path::new(socket_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("failed to bind depth socket {}", socket_path))?;
    tracing::info!("doorman-depthd listening on {}", socket_path);

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                if let Err(e) = handle_conn(stream, depth.as_ref(), &config.liveness) {
                    tracing::warn!("connection error: {}", e);
                }
            }
            Err(e) => tracing::warn!("accept error: {}", e),
        }
    }
    Ok(())
}

#[cfg(feature = "_ort")]
fn handle_conn(
    mut stream: UnixStream,
    depth: Option<&DepthSession>,
    liveness: &doorman_shared::config::LivenessConfig,
) -> Result<()> {
    use doormand::ml::Face;

    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(10)));

    // Read the fixed header.
    let mut hdr = [0u8; depth_ipc::REQ_HEADER_LEN];
    stream.read_exact(&mut hdr).context("read header")?;
    let req = match depth_ipc::decode_req_header(&hdr) {
        Some(r) => r,
        None => {
            let _ = stream.write_all(&depth_ipc::encode_resp(depth_ipc::RESP_ERR, 0.0));
            anyhow::bail!("bad request magic");
        }
    };

    // Read the RGB8 body.
    let n_bytes = (req.width as usize) * (req.height as usize) * 3;
    let mut pixels = vec![0u8; n_bytes];
    stream.read_exact(&mut pixels).context("read pixels")?;

    // Reconstruct the image.
    let img = match image::RgbImage::from_raw(req.width, req.height, pixels) {
        Some(rgb) => image::DynamicImage::ImageRgb8(rgb),
        None => {
            let _ = stream.write_all(&depth_ipc::encode_resp(depth_ipc::RESP_ERR, 0.0));
            anyhow::bail!("pixel buffer size mismatch");
        }
    };

    let face = Face {
        bbox: req.bbox,
        confidence: 1.0,
        frame_dimensions: (req.width, req.height),
        landmarks: None,
    };

    let resp = match depth {
        None => depth_ipc::encode_resp(depth_ipc::RESP_NONE, 0.0),
        Some(session) => match ort_backend::compute_depth_relief(session, &img, &face, liveness) {
            Ok(relief) => {
                tracing::debug!("relief={:.4} for bbox={:?}", relief, req.bbox);
                depth_ipc::encode_resp(depth_ipc::RESP_OK, relief)
            }
            Err(e) => {
                tracing::warn!("depth inference error: {}", e);
                depth_ipc::encode_resp(depth_ipc::RESP_ERR, 0.0)
            }
        },
    };
    stream.write_all(&resp).context("write response")?;
    stream.flush().ok();
    Ok(())
}

#[cfg(not(feature = "_ort"))]
fn run(_config: &Config, _socket_path: &str, _depth_ep: &str) -> Result<()> {
    anyhow::bail!("doorman-depthd built without an ORT backend. Build with --features backend-ort")
}
