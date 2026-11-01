#[cfg(feature = "_ort")]
use super::backend::{Face, MLBackend};
#[cfg(feature = "_ort")]
use anyhow::{anyhow, Context, Result};
#[cfg(feature = "_ort")]
use async_trait::async_trait;
#[cfg(feature = "_ort")]
use doorman_shared::Config;
#[cfg(feature = "_ort")]
use image::DynamicImage;
#[cfg(feature = "_ort")]
use ort::session::{builder::GraphOptimizationLevel, Session};
#[cfg(feature = "_ort")]
use ort::value::Value;
#[cfg(feature = "_ort")]
use std::path::Path;
#[cfg(feature = "_ort")]
use std::sync::Mutex;
#[cfg(feature = "_ort")]
use tracing::{info, warn};
#[cfg(feature = "_ort")]
use image::GenericImageView;

#[cfg(feature = "_ort")]
macro_rules! ort_try {
    ($expr:expr) => {
        $expr.map_err(|e| anyhow!("ORT error: {}", e))?
    };
}

#[cfg(feature = "_ort")]
/// ONNX Runtime backend (supports GPU via ROCm/CUDA)
/// Uses session pooling for concurrent requests
pub struct OrtBackend {
    detector_pool: Vec<Mutex<Session>>,
    /// Single MiniFASNetV2-SE liveness model session pool (may be empty: liveness
    /// is NON-FATAL and short-circuits to "pass" when unavailable).
    liveness_pool: Vec<Mutex<Session>>,
    /// Where the FATAL depth-relief PAD gate gets its depth pass from. On a CPU
    /// device the model is loaded in-process (`InProcess` pool); on a GPU EP
    /// (rocm/cuda) it would SEGV co-loaded with the GPU sessions, so it is run by
    /// the out-of-process `doorman-depthd` sidecar over `Sidecar(socket_path)`.
    depth_source: DepthSource,
    recognizer_pool: Vec<Mutex<Session>>,
    pool_index: AtomicUsize,
}

/// Source of the depth-relief PAD pass. See `OrtBackend::depth_source`.
#[cfg(feature = "_ort")]
enum DepthSource {
    /// In-process CPU depth sessions (CPU-device daemon). May be empty if the
    /// model file is missing → `depth_relief` returns `Ok(None)`.
    InProcess(Vec<Mutex<Session>>),
    /// Query the out-of-process depth sidecar at this Unix socket path (GPU-device
    /// daemon). A connect/IO/`RESP_ERR` failure surfaces as `Err`; `RESP_NONE`
    /// (sidecar has no model) surfaces as `Ok(None)` — same fail-safe semantics.
    Sidecar(String),
}

#[cfg(feature = "_ort")]
use std::sync::atomic::{AtomicUsize, Ordering};

/// ROCm/HIP pre-flight: does the HIP runtime see at least one usable GPU under the
/// CURRENT device scoping (ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)?
///
/// WHY this exists: ORT's ROCm EP calls `hipGetDeviceProperties(.., 0)` during
/// session init; when no device is visible (e.g. the gfx1103 iGPU whose KFD node
/// has `local_mem_size=0`, or a ROCR_VISIBLE_DEVICES filter pointing at a
/// non-existent index) it THROWS a C++ exception that surfaces as
/// `std::terminate()` → SIGABRT and kills the whole daemon before any Rust-side
/// fallback can run. A C++ terminate cannot be caught from Rust, so the decision
/// must be made BEFORE the EP is registered. `hipGetDeviceCount` only ever returns
/// an error code (it never aborts) and honours the same env scoping ORT uses, so
/// dlopen'ing libamdhip64 and calling it is a safe, accurate probe. Result cached.
#[cfg(feature = "backend-ort-rocm")]
fn rocm_runtime_has_device() -> bool {
    use std::sync::OnceLock;
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        let candidates = [
            format!("{rocm_path}/lib/libamdhip64.so"),
            "libamdhip64.so".to_string(),
        ];
        for path in &candidates {
            unsafe {
                let lib = match libloading::Library::new(path) {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let hip_get_device_count: libloading::Symbol<
                    unsafe extern "C" fn(*mut i32) -> i32,
                > = match lib.get(b"hipGetDeviceCount\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let mut count: i32 = 0;
                let err = hip_get_device_count(&mut count as *mut i32);
                // Keep libamdhip64 resident for the process lifetime (ORT loads it
                // again anyway); avoids churning HIP runtime state on unload.
                std::mem::forget(lib);
                let ok = err == 0 && count > 0;
                if ok {
                    info!("ROCm pre-flight: HIP reports {} visible GPU device(s)", count);
                } else {
                    warn!(
                        "ROCm pre-flight: no usable GPU (hipGetDeviceCount err={}, count={}) — falling back to CPU instead of crashing the daemon",
                        err, count
                    );
                }
                return ok;
            }
        }
        warn!("ROCm pre-flight: could not load libamdhip64 from ROCM_PATH or the linker path — treating ROCm as unavailable (CPU fallback)");
        false
    })
}

/// Whether the ROCm execution provider should be used, decided from config (no
/// hardcoded device list). ROCm is selected ONLY when ALL hold:
///   1. `ml.device` is a GPU backend (`rocm`/`gpu`),
///   2. `ml.gpu_uuid` names a specific GPU — we never auto-pick by index, so the
///      daemon can't bind an off-limits card (e.g. a dGPU) it wasn't told to use,
///   3. the HIP runtime actually sees that device (pre-flight).
/// Otherwise the daemon runs on CPU (logging the reason once). The decision is
/// cached so the per-model calls don't re-log.
/// Export the config-driven GPU binding env vars BEFORE any HIP init: select the
/// ROCm GPU by its STABLE UUID (`ml.gpu_uuid` → `ROCR_VISIBLE_DEVICES`) and apply
/// an optional `ml.hsa_gfx_version` override. Index-independent and operator-owned
/// (purely doorman.toml) — the daemon never guesses or hardcodes a device, so it
/// cannot bind a card you keep off-limits. Called by both the daemon backend ctor
/// and the depth sidecar's session loader. Idempotent; no-op for non-GPU devices.
#[cfg(feature = "backend-ort-rocm")]
fn apply_gpu_env(config: &Config) {
    if !matches!(config.ml.device.as_str(), "rocm" | "gpu") {
        return;
    }
    if let Some(uuid) = config.ml.gpu_uuid.as_deref().filter(|s| !s.is_empty()) {
        std::env::set_var("ROCR_VISIBLE_DEVICES", uuid);
        info!("Config gpu_uuid → ROCR_VISIBLE_DEVICES={} (ROCm bound to this GPU only)", uuid);
    }
    if let Some(v) = config.ml.hsa_gfx_version.as_deref().filter(|s| !s.is_empty()) {
        std::env::set_var("HSA_OVERRIDE_GFX_VERSION", v);
        info!("Config hsa_gfx_version → HSA_OVERRIDE_GFX_VERSION={}", v);
    }
}

#[cfg(feature = "backend-ort-rocm")]
fn rocm_ep_selected(config: &Config) -> bool {
    use std::sync::OnceLock;
    static DECISION: OnceLock<bool> = OnceLock::new();
    *DECISION.get_or_init(|| {
        if !matches!(config.ml.device.as_str(), "rocm" | "gpu") {
            return false;
        }
        match config.ml.gpu_uuid.as_deref() {
            Some(u) if !u.is_empty() => {}
            _ => {
                warn!(
                    "ml.device='{}' but ml.gpu_uuid is empty — refusing to auto-pick a GPU (could bind a card you keep off-limits); using CPU. Set ml.gpu_uuid to a UUID from `rocminfo` to enable ROCm.",
                    config.ml.device
                );
                return false;
            }
        }
        rocm_runtime_has_device()
    })
}

/// Which ONNX Runtime is loaded, for startup logs and benchmark reports.
///
/// `backend-ort`/`-cuda` statically link the runtime that ort-sys downloads at
/// build time (bundled); `backend-ort-rocm` loads a system libonnxruntime.so
/// chosen via ORT_DYLIB_PATH. The build string carries the runtime's version.
#[cfg(feature = "_ort")]
pub fn runtime_info() -> String {
    let linkage = if cfg!(feature = "backend-ort-rocm") {
        "system libonnxruntime.so (ort load-dynamic)"
    } else {
        "bundled (ort download-binaries)"
    };
    format!("{linkage}; {}", ort::info())
}

#[cfg(feature = "_ort")]
impl OrtBackend {
        pub fn new(models_dir: &Path, config: &Config) -> Result<Self> {
        info!("Initializing ONNX Runtime backend with session pooling...");
        info!("ONNX Runtime: {}", runtime_info());
        info!("Device: {}", config.ml.device);

        // GPU-aware pool size: a single GPU session already opens its own
        // MIOpen/cuDNN context + memory arena, so 4 sessions per model on a
        // memory-constrained iGPU (e.g. Radeon 780M) is wasteful/OOM-prone. Use
        // a small pool on GPU; keep 4 on CPU where it aids concurrency.
        let pool_size = Self::pool_size_for_device(&config.ml.device);
        info!(
            "ML device requested: {} (device_id={}) — check ort logs for EP registration result",
            config.ml.device, config.ml.gpu_device_id
        );

        // Config-driven GPU binding — must run before the first HIP init (the
        // model loads / ROCm pre-flight below). See apply_gpu_env: binds ROCm to
        // exactly the configured GPU by stable UUID, no hardcoded device list.
        #[cfg(feature = "backend-ort-rocm")]
        apply_gpu_env(config);

        // Load detector pool (YuNet)
        let detector_path = models_dir.join(super::model_config::DetectorConfig::YUNET.model_file);
        let mut detector_pool = Vec::new();
        for i in 0..pool_size {
            match Self::load_model(&detector_path, config) {
                Ok(model) => {
                    detector_pool.push(Mutex::new(model));
                }
                Err(e) => {
                    warn!("✗ Failed to load detector session {}: {}", i, e);
                }
            }
        }
        if !detector_pool.is_empty() {
            info!("✓ Loaded {} face detector sessions", detector_pool.len());
        }

        // Load the single MiniFASNetV2-SE liveness model pool.
        // Liveness is NON-FATAL: a missing/failed model yields an empty pool and
        // check_liveness short-circuits to "pass" with a warn.
        let liveness_cfg = super::model_config::LivenessConfig::MINIFASNET;
        let liveness_path = models_dir.join(liveness_cfg.model_file);
        let mut liveness_pool = Vec::new();
        for i in 0..pool_size {
            match Self::load_model(&liveness_path, config) {
                Ok(model) => liveness_pool.push(Mutex::new(model)),
                Err(e) => warn!("✗ Failed to load liveness {} session {}: {}", liveness_cfg.model_file, i, e),
            }
        }
        if !liveness_pool.is_empty() {
            info!("✓ Loaded {} sessions for liveness model {}", liveness_pool.len(), liveness_cfg.model_file);
        } else {
            warn!("✗ Liveness model {} unavailable — liveness will be skipped", liveness_cfg.model_file);
        }

        // Resolve the FATAL depth-relief PAD source.
        //
        // The depth transformer is run on the CPU EP even when ml.device=rocm:
        // on the gfx1103 iGPU (ROCm EP, ORT 1.22.x, HSA gfx-version override) the
        // Depth-Anything-V2 graph produces a numerically COMPRESSED depth range
        // (~2.6 vs ~4.9 on CPU), which collapses the genuine-vs-spoof relief
        // separation. WORSE: co-loading a CPU depth session into the SAME process
        // as the ROCm-EP sessions SEGVs (ORT's global OrtEnv can't mix CPU + GPU
        // sessions). So:
        //   - GPU device (rocm/cuda/gpu): run depth OUT of process via the
        //     `doorman-depthd` sidecar over a Unix socket (`Sidecar`). The
        //     detector/recognizer keep running on the iGPU here.
        //   - CPU device: load depth in-process on the CPU EP as before
        //     (`InProcess`) — the working, validated path; no sidecar needed.
        let is_gpu = {
            let dev = config.ml.device.as_str();
            // `mut` is only used under backend-ort-rocm (the downgrade below); on
            // other builds the reassignment is cfg'd out.
            #[allow(unused_mut)]
            let mut gpu = matches!(dev, "rocm" | "cuda" | "gpu");
            // If ROCm was requested but its runtime sees no GPU, load_model falls
            // back to the CPU EP for detector/recognizer — so depth must ALSO run
            // in-process on CPU, NOT via the ROCm sidecar (which would be down for
            // the same reason). Without this the daemon would survive but every
            // auth's liveness gate would fail (sidecar unreachable).
            #[cfg(feature = "backend-ort-rocm")]
            if matches!(dev, "rocm" | "gpu") && !rocm_ep_selected(config) {
                gpu = false;
            }
            gpu
        };
        let depth_source = if is_gpu {
            let sock = super::depth_ipc::resolve_socket_path(
                &Self::default_depth_socket_for(config),
            );
            info!(
                "Depth-PAD: GPU device '{}' → using out-of-process depth sidecar at {} (in-process CPU depth co-load would SEGV under the GPU EP)",
                config.ml.device, sock
            );
            DepthSource::Sidecar(sock)
        } else {
            let depth_cfg = super::model_config::DepthPadConfig::DEPTH_ANYTHING_V2;
            let depth_path = models_dir.join(depth_cfg.model_file);
            let depth_cpu_config = {
                let mut c = config.clone();
                c.ml.device = "cpu".to_string();
                c
            };
            let mut depth_pool = Vec::new();
            for i in 0..pool_size {
                match Self::load_model(&depth_path, &depth_cpu_config) {
                    Ok(model) => depth_pool.push(Mutex::new(model)),
                    Err(e) => warn!("✗ Failed to load depth-PAD {} session {}: {}", depth_cfg.model_file, i, e),
                }
            }
            if !depth_pool.is_empty() {
                info!("✓ Loaded {} in-process CPU sessions for depth-PAD model {}", depth_pool.len(), depth_cfg.model_file);
            } else {
                warn!("✗ Depth-PAD model {} unavailable — depth liveness gate will be skipped", depth_cfg.model_file);
            }
            DepthSource::InProcess(depth_pool)
        };

        // Load recognizer pool
        let recognizer_path = models_dir.join(super::model_config::RecognizerConfig::EDGEFACE.model_file);
        let mut recognizer_pool = Vec::new();
        for i in 0..pool_size {
            match Self::load_model(&recognizer_path, config) {
                Ok(model) => {
                    recognizer_pool.push(Mutex::new(model));
                }
                Err(e) => {
                    warn!("✗ Failed to load recognizer session {}: {}", i, e);
                }
            }
        }
        if !recognizer_pool.is_empty() {
            info!("✓ Loaded {} face recognizer sessions", recognizer_pool.len());
        }

        info!(
            "ORT backend: loaded {}/{} detector, {}/{} liveness, {}/{} recognizer sessions",
            detector_pool.len(), pool_size,
            liveness_pool.len(), pool_size,
            recognizer_pool.len(), pool_size
        );

        Ok(Self {
            detector_pool,
            liveness_pool,
            depth_source,
            recognizer_pool,
            pool_index: AtomicUsize::new(0),
        })
    }

    /// Compiled-default depth sidecar socket path for this config, used when the
    /// `DOORMAN_DEPTH_SOCKET` env override is unset. In `--user` mode the daemon
    /// rewrites socket paths under `XDG_RUNTIME_DIR`; mirror that for the depth
    /// socket by deriving it from the command socket's directory so both live in
    /// the same runtime dir. Falls back to the system default otherwise.
    fn default_depth_socket_for(config: &Config) -> String {
        let cmd_sock = std::path::Path::new(&config.daemon.socket_path);
        if let Some(dir) = cmd_sock.parent() {
            if !dir.as_os_str().is_empty() {
                return dir
                    .join("doorman-depth.sock")
                    .to_string_lossy()
                    .to_string();
            }
        }
        super::depth_ipc::DEFAULT_DEPTH_SOCKET.to_string()
    }
    
    /// Query the depth sidecar for one frame+bbox over its Unix socket.
    ///
    /// Returns `Ok(Some(relief))` on `RESP_OK`, `Ok(None)` on `RESP_NONE` (sidecar
    /// has no model loaded → same as the in-process missing-model case), and
    /// `Err` on connect/IO/protocol/`RESP_ERR` failures. The caller
    /// (`MLPipeline::liveness_gate`) treats both `Ok(None)` and `Err` as a
    /// fail-safe REJECT when liveness is enabled, so an unreachable sidecar can
    /// never silently bypass the PAD gate.
    fn query_depth_sidecar(
        sock: &str,
        image: &DynamicImage,
        face: &Face,
    ) -> Result<Option<f32>> {
        use super::depth_ipc;
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;

        let (w, h) = image.dimensions();
        let rgb = image.to_rgb8();

        let mut stream = UnixStream::connect(sock)
            .with_context(|| format!("depth sidecar unreachable at {}", sock))?;
        // Bound the round-trip so a wedged sidecar can't hang an auth attempt.
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
        let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(5)));

        let header = depth_ipc::encode_req_header(w, h, face.bbox);
        stream
            .write_all(&header)
            .context("depth sidecar: write header failed")?;
        stream
            .write_all(rgb.as_raw())
            .context("depth sidecar: write pixels failed")?;
        stream.flush().ok();

        let mut resp = [0u8; depth_ipc::RESP_LEN];
        stream
            .read_exact(&mut resp)
            .context("depth sidecar: read response failed")?;
        let (status, relief) = depth_ipc::decode_resp(&resp);
        match status {
            depth_ipc::RESP_OK => Ok(Some(relief)),
            depth_ipc::RESP_NONE => Ok(None),
            _ => Err(anyhow!("depth sidecar reported an inference error (status={})", status)),
        }
    }

    fn get_next_session<'a>(&'a self, pool: &'a [Mutex<Session>]) -> Option<&'a Mutex<Session>> {
        if pool.is_empty() {
            return None;
        }
        let idx = self.pool_index.fetch_add(1, Ordering::Relaxed) % pool.len();
        Some(&pool[idx])
    }

    /// Number of sessions to pool per model, based on the selected device.
    ///
    /// A GPU device (rocm/gpu/cuda) gets a small pool of 1: each GPU session
    /// opens its own MIOpen/cuDNN context + memory arena, so multiple sessions
    /// on a memory-constrained iGPU (e.g. Radeon 780M) are wasteful and
    /// OOM-prone.
    ///
    /// CPU gets a pool of 2 (was 4). This is a system face-auth daemon: PAM auth
    /// is SEQUENTIAL (one unlock at a time), not a high-concurrency server, so a
    /// large pool only inflates resident memory (each session loads its own copy
    /// of every model — the dominant driver of the ~1.6 GB idle RSS) without
    /// improving auth latency. Two sessions keep enough headroom for the brief
    /// detect→liveness→embed overlap inside a single auth attempt.
    fn pool_size_for_device(device: &str) -> usize {
        match device {
            "rocm" | "gpu" | "cuda" => 1,
            _ => 2,
        }
    }


    fn load_model(path: &Path, config: &Config) -> Result<Session> {
        let threads = if config.ml.cpu_threads > 0 {
            config.ml.cpu_threads as usize
        } else {
            4
        };

        let builder = Session::builder()
            .map_err(|e| anyhow!("Failed to create session builder: {}", e))?;
        // ort 2.0.0-rc.12 targets ORT 1.24; the ROCm load-dynamic path links ORT 1.22.x
        // (the last ORT with the ROCm EP), which rejects rc.12's graph-optimization-level
        // call ("graph_optimization_level is not valid"). Apply it best-effort: on a 1.24
        // runtime (CPU build) it is honored as before; on a 1.22.x runtime fall back to
        // ORT's default optimization level rather than failing session creation.
        let builder = match builder.with_optimization_level(GraphOptimizationLevel::Level3) {
            Ok(b) => b,
            Err(e) => {
                warn!("Graph optimization level not applied (ORT runtime rejected it): {}", e);
                Session::builder().map_err(|e| anyhow!("Failed to create session builder: {}", e))?
            }
        };
        let builder = builder
            .with_intra_threads(threads)
            .map_err(|e| anyhow!("Failed to set threads: {}", e))?;

        // Configure execution provider based on device
        #[cfg(feature = "backend-ort-cuda")]
        let builder = if config.ml.device == "cuda" || config.ml.device == "gpu" {
            info!("Configuring CUDA execution provider for {:?}", path);
            builder.with_execution_providers([
                ort::execution_providers::CUDA::default()
                    .with_device_id(config.ml.gpu_device_id)
                    .build(),
            ])
            .map_err(|e| anyhow!("Failed to set CUDA EP: {}", e))?
        } else {
            builder
        };

        // ROCm EP is registered only when config selects it AND HIP sees the
        // configured GPU (rocm_ep_selected). This is the FATAL-crash guard: ORT's
        // ROCm init does an uncatchable C++ terminate() ("no ROCm-capable device")
        // if registered with no visible device, which SIGABRTs the whole daemon.
        // Falling through to the CPU EP keeps the daemon alive (password/PAM still
        // work; CPU inference still authenticates) instead of crash-looping.
        #[cfg(feature = "backend-ort-rocm")]
        let builder = if rocm_ep_selected(config) {
            // HSA override precedence: config `ml.hsa_gfx_version` (applied in
            // new()) > operator env (systemd) > this best-effort default for the
            // gfx1103 iGPU. Only set if still unset so we never clobber the above.
            if std::env::var("HSA_OVERRIDE_GFX_VERSION").is_err() {
                std::env::set_var("HSA_OVERRIDE_GFX_VERSION", "11.0.3");
                info!("Set HSA_OVERRIDE_GFX_VERSION=11.0.3 for gfx1103 support (override via ml.hsa_gfx_version or systemd)");
            }
            info!("Configuring ROCm execution provider for {:?}", path);
            builder.with_execution_providers([
                ort::execution_providers::ROCm::default()
                    .with_device_id(config.ml.gpu_device_id)
                    .build(),
            ])
            .map_err(|e| anyhow!("Failed to set ROCm EP: {}", e))?
        } else {
            // device=cpu, or a GPU device whose configured card isn't usable —
            // rocm_ep_selected() already logged the reason once. Run on CPU.
            builder
        };

        // CoreML execution provider (Apple Silicon: Neural Engine / GPU / CPU).
        //
        // Selected when `ml.device` is one of `coreml`/`ane`/`gpu`/`auto` and the
        // `backend-ort-coreml` feature was compiled. Registered via the
        // non-fatal `.build()` so that any node/op CoreML can't run falls back to
        // the ORT CPU EP automatically (CoreML partitions the graph). Compute
        // units = ALL lets the runtime place subgraphs on ANE + GPU + CPU;
        // MLProgram is the newer, more-op-complete format; the model cache lets
        // CoreML reuse the compiled artifact across session loads (warmup absorbs
        // the first compile).
        #[cfg(feature = "backend-ort-coreml")]
        let builder = {
            let dev = config.ml.device.as_str();
            if matches!(dev, "coreml" | "ane" | "gpu" | "auto") {
                // ort 2.0.0-rc.12 renamed the CoreML EP types:
                //   CoreMLExecutionProvider -> CoreML
                //   CoreMLComputeUnits       -> ComputeUnits
                //   CoreMLModelFormat        -> ModelFormat
                use ort::execution_providers::coreml::{
                    ComputeUnits, CoreML, ModelFormat,
                };
                info!("Configuring CoreML execution provider (device='{}') for {:?}", dev, path);
                let cache_dir = std::env::temp_dir().join("doorman_coreml_cache");
                let _ = std::fs::create_dir_all(&cache_dir);
                let coreml = CoreML::default()
                    .with_compute_units(ComputeUnits::All)
                    .with_model_format(ModelFormat::MLProgram)
                    .with_static_input_shapes(true)
                    .with_model_cache_dir(cache_dir.to_string_lossy().to_string());
                builder
                    .with_execution_providers([coreml.build()])
                    .map_err(|e| anyhow!("Failed to set CoreML EP: {}", e))?
            } else {
                builder
            }
        };

        // Load model file
        let model_bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read model file: {:?}", path))?;

        // ort 2.0.0-rc.12 changed `commit_from_memory` to take `&mut self`
        // (previously consumed `self`), so the builder must be mutable here.
        let mut builder = builder;
        let session = builder
            .commit_from_memory(&model_bytes)
            .map_err(|e| anyhow!("Failed to create session from model: {}", e))?;

        Ok(session)
    }
}

/// A loaded depth session, mutex-guarded for `&self` use. Exposed so the depth
/// sidecar binary can hold one without naming the `ort` crate directly.
/// (Consumed by the `doorman-depthd` bin, not the lib — hence `allow(dead_code)`.)
#[cfg(feature = "_ort")]
#[allow(dead_code)]
pub type DepthSession = Mutex<Session>;

/// Stretch-resize `image` to `size`×`size` RGB, then apply ImageNet normalization
/// `((x/255) - mean) / std` with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
/// and pack into a planar CHW f32 buffer (RGB order, length 3×size×size).
///
/// Matches `DepthPadConfig::DEPTH_ANYTHING_V2` preprocessing exactly.
#[cfg(feature = "_ort")]
pub fn depth_preprocess(image: &DynamicImage, size: u32) -> Vec<f32> {
    use super::model_config::DepthPadConfig;
    let cfg = DepthPadConfig::DEPTH_ANYTHING_V2;

    let resized = image.resize_exact(size, size, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();

    let n = (size * size) as usize;
    let mut input = vec![0.0f32; 3 * n];
    for (i, px) in rgb.pixels().enumerate() {
        input[i] = ((px[0] as f32 / 255.0) - cfg.norm_mean[0]) / cfg.norm_std[0];
        input[n + i] = ((px[1] as f32 / 255.0) - cfg.norm_mean[1]) / cfg.norm_std[1];
        input[2 * n + i] = ((px[2] as f32 / 255.0) - cfg.norm_mean[2]) / cfg.norm_std[2];
    }
    input
}

/// Compute the depth-relief PAD score from a depth map and a normalized face bbox.
///
/// `depth` is the per-pixel inverse-depth vector (length ≥ d_h × d_w) in row-major
/// order. Maps the face bbox into depth-map coordinates, computes
/// `relief = std(face_depth) / (depth_range + eps)` clamped to `[0, 1]`.
///
/// When the face-region crop is empty (degenerate bbox) the whole-map depth values
/// are used, producing a near-zero relief.
#[cfg(feature = "_ort")]
pub fn relief_from_depth_map(depth: &[f32], d_h: usize, d_w: usize, face: &Face) -> f32 {
    // Global depth range over the whole map.
    let mut gmin = f32::INFINITY;
    let mut gmax = f32::NEG_INFINITY;
    for &v in depth.iter() {
        if v < gmin {
            gmin = v;
        }
        if v > gmax {
            gmax = v;
        }
    }
    let global_range = (gmax - gmin) + 1e-8;

    // Map the normalized face bbox into depth-map pixel coordinates.
    let (nx, ny, nw, nh) = face.bbox;
    let x0 = ((nx.clamp(0.0, 1.0)) * d_w as f32) as usize;
    let y0 = ((ny.clamp(0.0, 1.0)) * d_h as f32) as usize;
    let x1 = (((nx + nw).clamp(0.0, 1.0)) * d_w as f32).ceil() as usize;
    let y1 = (((ny + nh).clamp(0.0, 1.0)) * d_h as f32).ceil() as usize;
    let x1 = x1.min(d_w);
    let y1 = y1.min(d_h);

    // Collect the face-region depth values; fall back to whole map if empty.
    let mut face_vals: Vec<f32> = Vec::new();
    if x1 > x0 && y1 > y0 {
        for y in y0..y1 {
            let row = y * d_w;
            for x in x0..x1 {
                face_vals.push(depth[row + x]);
            }
        }
    }
    let face_vals = if face_vals.is_empty() {
        depth.to_vec()
    } else {
        face_vals
    };

    // std(face_depth).
    let count = face_vals.len() as f32;
    let mean = face_vals.iter().sum::<f32>() / count;
    let var = face_vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / count;
    let std = var.sqrt();

    (std / global_range).clamp(0.0, 1.0)
}

/// Run the Depth-Anything-V2 depth pass on `session` for `image` and compute the
/// face-region depth-relief PAD score for `face`.
///
/// This is the SINGLE source of truth for the depth-relief computation, shared by
/// the in-process `OrtBackend::depth_relief` (CPU-device daemon) and the
/// out-of-process depth sidecar (`doorman-depthd`, used when the daemon runs on a
/// GPU EP so the CPU depth model is never co-loaded with ROCm/CUDA sessions in the
/// same `OrtEnv` — that mix SEGVs).
///
/// Preprocessing + relief math MUST match `DepthPadConfig::DEPTH_ANYTHING_V2`:
/// stretch-resize to `input_size`², ImageNet-normalize, NCHW `pixel_values`,
/// `relief = std(face_depth) / (depth_range + eps)` clamped to `[0,1]`.
#[cfg(feature = "_ort")]
pub fn compute_depth_relief(
    session: &Mutex<Session>,
    image: &DynamicImage,
    face: &Face,
) -> Result<f32> {
    use super::model_config::DepthPadConfig;

    let cfg = DepthPadConfig::DEPTH_ANYTHING_V2;
    let size = cfg.input_size;

    let input = depth_preprocess(image, size);

    let input_tensor = Value::from_array((
        [1usize, 3, size as usize, size as usize],
        input,
    ))
    .map_err(|e| anyhow!("ORT error: {}", e))?;

    let mut lock = session.lock().unwrap();
    // The model's input is named `pixel_values`.
    let outputs = lock
        .run(ort::inputs!["pixel_values" => input_tensor])
        .map_err(|e| anyhow!("ORT error: {}", e))?;
    let (shape, depth_view) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow!("ORT error: {}", e))?;
    // Copy out before releasing the session lock (the view borrows `outputs`).
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let depth: Vec<f32> = depth_view.to_vec();
    drop(outputs);
    drop(lock);

    let (d_h, d_w) = match dims.as_slice() {
        [_, h, w] => (*h, *w),
        [h, w] => (*h, *w),
        _ => return Err(anyhow!("Unexpected depth output shape {:?}", dims)),
    };
    if d_h == 0 || d_w == 0 || depth.len() < d_h * d_w {
        return Err(anyhow!("Depth output too small: shape {:?} len {}", dims, depth.len()));
    }

    let relief = relief_from_depth_map(&depth, d_h, d_w, face);
    tracing::debug!(
        "Depth-PAD relief={:.4} depth_grid={}x{}",
        relief, d_w, d_h
    );
    Ok(relief)
}

/// Build a single CPU-EP Depth-Anything-V2 session from `models_dir`.
///
/// Used by the depth sidecar (`doorman-depthd`): it loads ONLY the depth model on
/// the CPU EP, so it never co-loads a CPU session into the same process as the
/// daemon's ROCm/CUDA sessions. Returns the model path that was attempted in the
/// error so the operator can see what was missing.
#[cfg(feature = "_ort")]
#[allow(dead_code)] // used by the doorman-depthd bin, not the daemon lib path
pub fn load_depth_session_cpu(models_dir: &Path, config: &Config) -> Result<DepthSession> {
    load_depth_session_for_ep(models_dir, config, "cpu")
}

/// Build a single ROCm-EP Depth-Anything-V2 session from `models_dir`.
///
/// Used by the depth sidecar (`doorman-depthd`) on ROCm installs so the depth
/// pass runs on the iGPU instead of the CPU, keeping the CPU light during auth.
///
/// SAFETY (the whole reason a SIDECAR exists): this loads the depth model on the
/// ROCm EP in a SEPARATE PROCESS from the daemon. The known SEGV was mixing a CPU
/// depth session with ROCm sessions in ONE `OrtEnv`; a pure-ROCm process (this
/// one) is fine, as is the daemon's pure-ROCm process. Two distinct processes
/// each with their own ROCm sessions share the iGPU at the HIP/driver level.
///
/// Requires the `backend-ort-rocm` feature (else `load_model` never registers the
/// ROCm EP and ORT silently falls back to CPU — which would defeat the purpose).
#[cfg(feature = "_ort")]
#[allow(dead_code)] // used by the doorman-depthd bin, not the daemon lib path
pub fn load_depth_session_rocm(models_dir: &Path, config: &Config) -> Result<DepthSession> {
    load_depth_session_for_ep(models_dir, config, "rocm")
}

/// Shared depth-session loader: load ONLY the depth model with `config` rewritten
/// to target `device`. The EP is actually registered inside `load_model` (gated on
/// the compiled `backend-ort-*` feature), so passing `device="rocm"` only has an
/// effect when the binary was built with `backend-ort-rocm`.
#[cfg(feature = "_ort")]
#[allow(dead_code)]
fn load_depth_session_for_ep(models_dir: &Path, config: &Config, device: &str) -> Result<DepthSession> {
    let depth_cfg = super::model_config::DepthPadConfig::DEPTH_ANYTHING_V2;
    let depth_path = models_dir.join(depth_cfg.model_file);
    let mut ep_config = config.clone();
    ep_config.ml.device = device.to_string();
    // Bind ROCm to the configured GPU by UUID before HIP init (sidecar path).
    #[cfg(feature = "backend-ort-rocm")]
    apply_gpu_env(&ep_config);
    let session = OrtBackend::load_model(&depth_path, &ep_config)
        .with_context(|| format!("Failed to load depth model {:?} on {} EP", depth_path, device))?;
    Ok(Mutex::new(session))
}

#[cfg(feature = "_ort")]
impl OrtBackend {
    /// Preprocess an image into YuNet's input tensor.
    ///
    /// YuNet 2023mar has a FIXED `[1, 3, 640, 640]` input expecting **BGR**,
    /// raw float `0..255` (NO mean/std normalization), NCHW. The image is
    /// stretch-resized to the square input (aspect ratio NOT preserved); the
    /// decoder maps coordinates back to the original frame.
    fn yunet_preprocess(image: &DynamicImage, size: u32) -> Vec<f32> {
        // Stretch-resize to the square network input.
        let resized = image.resize_exact(size, size, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        let n = (size * size) as usize;
        // NCHW, channel order B, G, R (YuNet/OpenCV expects BGR).
        let mut data = vec![0.0f32; 3 * n];
        let (b_off, g_off, r_off) = (0, n, 2 * n);
        for (i, px) in rgb.pixels().enumerate() {
            // px is RGB; write into BGR planes, raw 0..255.
            data[b_off + i] = px[2] as f32;
            data[g_off + i] = px[1] as f32;
            data[r_off + i] = px[0] as f32;
        }
        data
    }

    /// Fallback face crop when landmarks are unavailable: crop the (normalized)
    /// bbox from the full frame and resize to `size`x`size`. This is the legacy
    /// path and is noticeably less accurate than landmark alignment; it is only
    /// hit by warmup / landmark-less callers.
    /// Replicate facenox `crop()`: a SQUARE crop of side
    /// `max(bbox_w, bbox_h) * factor` centered on the bbox center, with any
    /// out-of-frame region filled by **reflect-101** padding (mirror without
    /// repeating the edge pixel, i.e. `cv2.BORDER_REFLECT_101`), then resized to
    /// `size`x`size` RGB. Mirrors the repo's integer arithmetic.
    fn antispoof_square_crop(
        image: &DynamicImage,
        face: &Face,
        factor: f32,
        size: u32,
    ) -> image::RgbImage {
        let src = image.to_rgb8();
        let (img_w, img_h) = (src.width() as i64, src.height() as i64);

        // bbox in source pixels (normalized [0,1] -> px), matching the repo's
        // (x, y, x+w, y+h) -> (x, y, w, h) reconstruction.
        let (nx, ny, nw, nh) = face.bbox;
        let bx = nx * img_w as f32;
        let by = ny * img_h as f32;
        let bw = (nw * img_w as f32).max(1.0);
        let bh = (nh * img_h as f32).max(1.0);

        let max_dim = bw.max(bh);
        let center_x = bx + bw / 2.0;
        let center_y = by + bh / 2.0;
        let crop_size = (max_dim * factor) as i64; // int() truncation, as in repo
        let crop_size = crop_size.max(1);
        let x0 = (center_x - max_dim * factor / 2.0) as i64; // int() truncation
        let y0 = (center_y - max_dim * factor / 2.0) as i64;

        // Reflect-101 index mapping into [0, len): mirror about the edges
        // without repeating the border pixel (matches OpenCV BORDER_REFLECT_101).
        fn reflect101(mut i: i64, len: i64) -> i64 {
            if len == 1 {
                return 0;
            }
            let period = 2 * (len - 1);
            i = ((i % period) + period) % period;
            if i >= len {
                i = period - i;
            }
            i
        }

        // Build the crop_size x crop_size square, sampling source with reflect-101.
        let mut square = image::RgbImage::new(crop_size as u32, crop_size as u32);
        for cy in 0..crop_size {
            let sy = reflect101(y0 + cy, img_h);
            for cx in 0..crop_size {
                let sx = reflect101(x0 + cx, img_w);
                let px = *src.get_pixel(sx as u32, sy as u32);
                square.put_pixel(cx as u32, cy as u32, px);
            }
        }

        // Letterbox-resize to `size` (no-op aspect change for a square crop).
        image::imageops::resize(
            &square,
            size,
            size,
            image::imageops::FilterType::Lanczos3,
        )
    }

    fn bbox_crop_resize(image: &DynamicImage, face: &Face, size: u32) -> image::RgbImage {
        let (img_w, img_h) = image.dimensions();
        let (nx, ny, nw, nh) = face.bbox;
        let x = (nx.clamp(0.0, 1.0) * img_w as f32) as u32;
        let y = (ny.clamp(0.0, 1.0) * img_h as f32) as u32;
        let w = ((nw.clamp(0.0, 1.0) * img_w as f32) as u32).max(1).min(img_w.saturating_sub(x).max(1));
        let h = ((nh.clamp(0.0, 1.0) * img_h as f32) as u32).max(1).min(img_h.saturating_sub(y).max(1));
        image
            .crop_imm(x, y, w, h)
            .resize_exact(size, size, image::imageops::FilterType::Lanczos3)
            .to_rgb8()
    }
}

#[cfg(feature = "_ort")]
#[async_trait]
impl MLBackend for OrtBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        use super::model_config::DetectorConfig;
        use super::yunet_decoder::{self, StrideOutputs};

        let detector = self
            .get_next_session(&self.detector_pool)
            .ok_or_else(|| anyhow!("Detector not loaded"))?;

        let cfg = DetectorConfig::YUNET;
        let (orig_width, orig_height) = image.dimensions();
        let input_size = cfg.input_width; // YuNet is square (640x640)

        // Preprocess -> BGR, raw 0..255, NCHW [1,3,640,640].
        let input_data = Self::yunet_preprocess(image, input_size);
        let input_tensor = ort_try!(Value::from_array((
            [1usize, 3, input_size as usize, input_size as usize],
            input_data
        )));

        let mut detector_lock = detector.lock().unwrap();
        let outputs = ort_try!(detector_lock.run(ort::inputs![input_tensor]));

        // Collect the per-stride output slices by name. The slices borrow from
        // `outputs`, which outlives the decode call below.
        let mut stride_views: Vec<StrideOutputs> = Vec::with_capacity(DetectorConfig::YUNET_STRIDES.len());
        for &stride in DetectorConfig::YUNET_STRIDES.iter() {
            let (_, cls) = ort_try!(outputs[format!("cls_{}", stride).as_str()].try_extract_tensor::<f32>());
            let (_, obj) = ort_try!(outputs[format!("obj_{}", stride).as_str()].try_extract_tensor::<f32>());
            let (_, bbox) = ort_try!(outputs[format!("bbox_{}", stride).as_str()].try_extract_tensor::<f32>());
            let (_, kps) = ort_try!(outputs[format!("kps_{}", stride).as_str()].try_extract_tensor::<f32>());
            stride_views.push(StrideOutputs { stride, cls, obj, bbox, kps });
        }

        let dets = yunet_decoder::decode(
            &stride_views,
            input_size,
            orig_width,
            orig_height,
            cfg.confidence_threshold,
        );
        let dets = yunet_decoder::nms(dets, cfg.iou_threshold);

        // Single-user behavior: pick the highest-scoring face.
        let best = dets
            .into_iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some(d) => {
                // Clamp the normalized bbox to the frame.
                let (mut x, mut y, w, h) = d.bbox;
                x = x.clamp(0.0, 1.0);
                y = y.clamp(0.0, 1.0);
                let w = w.clamp(0.0, 1.0 - x);
                let h = h.clamp(0.0, 1.0 - y);

                tracing::debug!(
                    "YuNet detection: score={:.3} bbox_norm=({:.3},{:.3},{:.3},{:.3}) frame={}x{}",
                    d.score, x, y, w, h, orig_width, orig_height
                );

                Ok(Some(Face {
                    bbox: (x, y, w, h),
                    confidence: d.score,
                    frame_dimensions: (orig_width, orig_height),
                    landmarks: Some(d.landmarks),
                }))
            }
            None => {
                tracing::debug!("YuNet: no face above threshold {:.2}", cfg.confidence_threshold);
                Ok(None)
            }
        }
    }

    /// MiniFASNetV2-SE anti-spoofing (facenox/face-antispoof-onnx, 128x128).
    ///
    /// Replicates the repo pipeline EXACTLY:
    /// 1. Square crop of side `max(bbox_w, bbox_h) * bbox_expansion_factor`
    ///    (default 1.5), centered on the bbox center; out-of-frame regions are
    ///    **reflect-101** padded (`cv2.BORDER_REFLECT_101`).
    /// 2. Resize to 128x128, color order **RGB**, normalize **`/255` -> [0,1]**
    ///    (NO mean/std), NCHW float32 `[1, 3, 128, 128]`.
    /// 3. Output `[1, 2]` raw logits (index 0 = real, index 1 = spoof). Decide
    ///    `is_real = (real_logit - spoof_logit) >= ln(p/(1-p))` for the configured
    ///    real-probability `p` (default 0.5 -> plain argmax).
    ///
    /// NON-FATAL: with no loaded model, returns `Ok(true)` (skip) with a warn so
    /// liveness can never block recognition.
    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool> {
        use super::model_config::LivenessConfig;

        if self.liveness_pool.is_empty() {
            warn!("Liveness skipped: MiniFASNetV2-SE model not loaded");
            return Ok(true);
        }

        let cfg = LivenessConfig::MINIFASNET;
        let size = cfg.input_size; // 128
        let session = match self.get_next_session(&self.liveness_pool) {
            Some(s) => s,
            None => {
                warn!("Liveness skipped: no MiniFASNetV2-SE session available");
                return Ok(true);
            }
        };

        // Build the reflect-101-padded square crop resized to `size`x`size`, RGB.
        let crop = Self::antispoof_square_crop(image, face, cfg.bbox_expansion_factor, size);

        // RGB, /255 -> [0,1], NCHW planar (matches the repo's `preprocess`).
        let n = (size * size) as usize;
        let mut input = vec![0.0f32; 3 * n];
        let (r_off, g_off, b_off) = (0, n, 2 * n);
        for (i, px) in crop.pixels().enumerate() {
            input[r_off + i] = px[0] as f32 / 255.0;
            input[g_off + i] = px[1] as f32 / 255.0;
            input[b_off + i] = px[2] as f32 / 255.0;
        }

        let input_tensor = ort_try!(Value::from_array((
            [1usize, 3, size as usize, size as usize],
            input
        )));
        let mut lock = session.lock().unwrap();
        let outputs = ort_try!(lock.run(ort::inputs![input_tensor]));
        let (_, logits) = ort_try!(outputs[0].try_extract_tensor::<f32>());

        if logits.len() <= cfg.real_class_index || logits.len() <= cfg.spoof_class_index {
            warn!("Liveness skipped: unexpected MiniFASNetV2-SE output len {}", logits.len());
            return Ok(true);
        }
        let real_logit = logits[cfg.real_class_index];
        let spoof_logit = logits[cfg.spoof_class_index];
        let diff = real_logit - spoof_logit;

        // logit threshold = ln(p/(1-p)); p clamped away from 0/1.
        let p = cfg.real_prob_threshold.clamp(1e-6, 1.0 - 1e-6);
        let logit_threshold = (p / (1.0 - p)).ln();
        let is_real = diff >= logit_threshold;

        // Softmax real-probability for logging only.
        let p_real = 1.0 / (1.0 + (-diff).exp());
        tracing::debug!(
            "MiniFASNetV2-SE liveness: real_logit={:.3} spoof_logit={:.3} diff={:.3} p_real={:.4} threshold_logit={:.3} -> {}",
            real_logit, spoof_logit, diff, p_real, logit_threshold,
            if is_real { "REAL" } else { "SPOOF" }
        );
        Ok(is_real)
    }

    /// Monocular-depth face relief PAD score (Depth-Anything-V2).
    ///
    /// Preprocessing (must match `DepthPadConfig::DEPTH_ANYTHING_V2`):
    /// 1. Stretch-resize the full RGB frame to `input_size`x`input_size` (518).
    /// 2. ImageNet-normalize `((x/255) - mean) / std`, NCHW float32, input
    ///    tensor name `pixel_values`.
    /// 3. Output `predicted_depth` `[1, Hd, Wd]` inverse-depth map.
    /// 4. Map the (clamped) face bbox into depth-map coordinates, compute
    ///    `relief = std(face_depth) / (depth_max - depth_min + eps)`, clamp to
    ///    `[0, 1]`. Higher = more 3D structure = more likely live.
    ///
    /// Returns `Ok(None)` if no depth model is loaded.
    async fn depth_relief(&self, image: &DynamicImage, face: &Face) -> Result<Option<f32>> {
        match &self.depth_source {
            // CPU device: run the shared depth-relief computation in-process.
            DepthSource::InProcess(pool) => {
                let session = match self.get_next_session(pool) {
                    Some(s) => s,
                    None => return Ok(None),
                };
                let relief = compute_depth_relief(session, image, face)?;
                Ok(Some(relief))
            }
            // GPU device: delegate the depth pass to the out-of-process sidecar.
            DepthSource::Sidecar(sock) => Self::query_depth_sidecar(sock, image, face),
        }
    }

    /// Extract a 512-d face embedding using the EdgeFace-S recognizer.
    ///
    /// Preprocessing (must match `RecognizerConfig::EDGEFACE`):
    /// 1. **Align** the face to the canonical 112x112 5-point template via a
    ///    similarity transform from the detector's 5 landmarks. If the detector
    ///    provided no landmarks, fall back to a plain bbox crop+resize (degraded —
    ///    used only by warmup / landmark-less callers).
    /// 2. Color order **RGB**, normalization `(x - 127.5) / 127.5` -> [-1, 1]
    ///    (EdgeFace's `ToTensor()` + `Normalize(0.5, 0.5)`; identical to ArcFace).
    /// 3. NCHW float32 `[1, 3, 112, 112]`.
    /// 4. **L2-normalize** the 512-d output so cosine == dot product.
    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        use super::align;
        use super::model_config::RecognizerConfig;

        const SIZE: u32 = 112;
        let recognizer = self
            .get_next_session(&self.recognizer_pool)
            .ok_or_else(|| anyhow!("Recognizer not loaded"))?;

        let (img_w, img_h) = image.dimensions();

        // Build the aligned 112x112 RGB crop.
        let aligned: image::RgbImage = match face.landmarks {
            Some(landmarks_norm) => {
                // Landmarks are normalized [0,1]; convert to source pixels.
                let landmarks_px: [(f32, f32); 5] = std::array::from_fn(|i| {
                    (
                        landmarks_norm[i].0 * img_w as f32,
                        landmarks_norm[i].1 * img_h as f32,
                    )
                });
                match align::align_to_template(
                    image,
                    &landmarks_px,
                    &RecognizerConfig::RECOGNIZER_TEMPLATE_112,
                    SIZE,
                ) {
                    Some(a) => a,
                    None => Self::bbox_crop_resize(image, face, SIZE),
                }
            }
            None => Self::bbox_crop_resize(image, face, SIZE),
        };

        // Preprocess: RGB, (x - 127.5)/127.5, NCHW planar.
        let n = (SIZE * SIZE) as usize;
        let mut input = vec![0.0f32; 3 * n];
        let (r_off, g_off, b_off) = (0, n, 2 * n);
        for (i, px) in aligned.pixels().enumerate() {
            input[r_off + i] = (px[0] as f32 - 127.5) / 127.5;
            input[g_off + i] = (px[1] as f32 - 127.5) / 127.5;
            input[b_off + i] = (px[2] as f32 - 127.5) / 127.5;
        }

        let input_tensor = ort_try!(Value::from_array((
            [1usize, 3, SIZE as usize, SIZE as usize],
            input
        )));
        let mut recognizer_lock = recognizer.lock().unwrap();
        let outputs = ort_try!(recognizer_lock.run(ort::inputs![input_tensor]));

        let (_, embedding_data) = ort_try!(outputs[0].try_extract_tensor::<f32>());
        let embedding: Vec<f32> = embedding_data.iter().copied().collect();

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized = if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding
        };

        Ok(normalized)
    }

    fn is_ready(&self) -> bool {
        // Liveness is a non-fatal convenience check and does NOT gate readiness.
        !self.detector_pool.is_empty() && !self.recognizer_pool.is_empty()
    }

    fn name(&self) -> &'static str {
        "ONNX Runtime"
    }
}
