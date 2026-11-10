mod backend;
mod model_config;

/// Depth-sidecar wire protocol (pure std; shared by the daemon's GPU-EP depth
/// routing and the `doorman-depthd` sidecar binary). Public so the bin can use it.
pub mod depth_ipc;

/// Locate the ONNX Runtime shared library mapped into this process.
pub mod ort_probe;

#[cfg(feature = "_ort")]
mod align;

#[cfg(feature = "_ort")]
pub mod ort_backend;

#[cfg(feature = "_ort")]
mod yunet_decoder;

/// ncnn-Vulkan iGPU backend FFI wrapper (PHASE 1 foundation). Gated on
/// `backend-ncnn`; not wired into `MLBackend` yet (that is PHASE 2). Public so
/// the gated `ncnn_igpu` example/test can exercise the FFI directly.
#[cfg(feature = "backend-ncnn")]
pub mod ncnn_backend;

/// ncnn-Vulkan hybrid backend (PHASE 2): delegates detect/embed/liveness to the
/// CPU `OrtBackend` and overrides depth_relief to run on the iGPU via ncnn.
/// Requires both the `_ort` and `backend-ncnn` features.
#[cfg(all(feature = "_ort", feature = "backend-ncnn"))]
pub mod ncnn_hybrid;

mod tta;

#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use model_config::{
    DepthPadConfig, DetectorConfig, LivenessConfig, ModelSet, RecognizerConfig,
};

use crate::liveness::LivenessOutcome;
use anyhow::Result;
#[allow(dead_code)]
pub use backend::Face;
pub use backend::{BackendType, MLBackend};
use doorman_shared::Config;
use image::DynamicImage;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

/// ML Pipeline with pluggable backend
pub struct MLPipeline {
    backend: Arc<dyn MLBackend>,
    config: Config,
}

impl MLPipeline {
    // Without an ORT feature the only backend arm returns early with an error,
    // so the code after the match (and the match binding) is intentionally dead.
    #[cfg_attr(not(feature = "_ort"), allow(unreachable_code, unused_variables))]
    pub async fn new(config: &Config) -> Result<Self> {
        let models_dir = PathBuf::from(&config.ml.models_dir);

        // ncnn hybrid backend: must be checked before the generic BackendType
        // match because `config.ml.backend == "ncnn"` is not a BackendType variant.
        #[cfg(all(feature = "_ort", feature = "backend-ncnn"))]
        if config.ml.backend == "ncnn" {
            let backend: std::sync::Arc<dyn backend::MLBackend> =
                std::sync::Arc::new(ncnn_hybrid::NcnnHybridBackend::new(&models_dir, config)?);
            info!("Using ML backend: {}", backend.name());
            return Ok(Self {
                backend,
                config: config.clone(),
            });
        }

        // Select backend based on config
        let backend_type = BackendType::from_str(&config.ml.backend);

        info!("Initializing ML backend: {:?}", backend_type);

        let backend: Arc<dyn MLBackend> = match backend_type {
            BackendType::OnnxRuntime => {
                #[cfg(feature = "_ort")]
                {
                    Arc::new(ort_backend::OrtBackend::new(&models_dir, config)?)
                }
                #[cfg(not(feature = "_ort"))]
                {
                    let _ = &models_dir;
                    return Err(anyhow::anyhow!(
                        "ORT backend not compiled. Build with --features backend-ort"
                    ));
                }
            }
        };

        info!("Using ML backend: {}", backend.name());

        Ok(Self {
            backend,
            config: config.clone(),
        })
    }

    pub fn dummy(config: &Config) -> Self {
        // For testing / startup fallback - create a backend even if model files
        // are missing (sessions simply fail to load and is_ready() returns false).
        #[cfg(feature = "_ort")]
        {
            let models_dir = PathBuf::from(&config.ml.models_dir);
            let backend = ort_backend::OrtBackend::new(&models_dir, config)
                .unwrap_or_else(|e| panic!("Failed to create dummy ORT backend: {}", e));

            Self {
                backend: Arc::new(backend),
                config: config.clone(),
            }
        }
        #[cfg(not(feature = "_ort"))]
        {
            let _ = config;
            panic!("No backend available for dummy. Compile with --features backend-ort");
        }
    }

    /// Build a pipeline around an arbitrary backend (tests with a mock backend).
    #[cfg(test)]
    pub(crate) fn with_backend(backend: Arc<dyn MLBackend>, config: &Config) -> Self {
        Self {
            backend,
            config: config.clone(),
        }
    }

    /// Run one backend call on tokio's blocking pool instead of an async worker.
    ///
    /// ONNX inference is CPU-bound and the backend takes a std `Mutex` on its
    /// session pool, so calling it directly from an async task can park a
    /// runtime worker for the whole inference (and stall `Status` and every
    /// other connection when several attempts overlap). The backend futures
    /// never await real I/O, so `block_on` on the blocking thread is safe (same
    /// reasoning as `detect_face_sync`). The image is cloned: the blocking task
    /// must be `'static`.
    async fn run_blocking<T, F>(&self, image: &DynamicImage, call: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&dyn MLBackend, &DynamicImage) -> Result<T> + Send + 'static,
    {
        let backend = self.backend.clone();
        let image = image.clone();
        tokio::task::spawn_blocking(move || call(backend.as_ref(), &image))
            .await
            .map_err(|e| anyhow::anyhow!("inference task failed: {e}"))?
    }

    async fn detect_blocking(&self, image: &DynamicImage) -> Result<Option<backend::Face>> {
        self.run_blocking(image, |backend, image| {
            tokio::runtime::Handle::current().block_on(backend.detect_face(image))
        })
        .await
    }

    async fn depth_relief_blocking(
        &self,
        image: &DynamicImage,
        face: &backend::Face,
    ) -> Result<Option<f32>> {
        let face = face.clone();
        self.run_blocking(image, move |backend, image| {
            tokio::runtime::Handle::current().block_on(backend.depth_relief(image, &face))
        })
        .await
    }

    async fn embed_blocking(&self, image: &DynamicImage, face: &backend::Face) -> Result<Vec<f32>> {
        let face = face.clone();
        self.run_blocking(image, move |backend, image| {
            tokio::runtime::Handle::current().block_on(backend.extract_embedding(image, &face))
        })
        .await
    }

    async fn check_liveness_blocking(
        &self,
        image: &DynamicImage,
        face: &backend::Face,
    ) -> Result<bool> {
        let face = face.clone();
        self.run_blocking(image, move |backend, image| {
            tokio::runtime::Handle::current().block_on(backend.check_liveness(image, &face))
        })
        .await
    }

    /// Full processing: detection + liveness + embedding (for recognition)
    pub async fn process_frame(
        &self,
        image: &DynamicImage,
    ) -> Result<Option<(backend::Face, Vec<f32>)>> {
        // Stage 1: Detect (on full-size image, detector will resize internally)
        let face = match self.detect_blocking(image).await? {
            Some(f) => f,
            None => return Ok(None),
        };

        // Stage 2: Liveness (FATAL monocular-depth anti-spoofing).
        //
        // Controlled by `authentication.liveness_enabled` (default true). The
        // gate is FATAL: when a face's depth-relief score falls below
        // `authentication.liveness_depth_threshold` it is treated as a spoof and
        // we return `Ok(None)` (the same "no valid face" path), so the IPC
        // authenticate replies Failure and PAM falls through to password. A real
        // 3D face has high depth relief; a flat phone/print replay has ~0.
        //
        // The OLD MiniFASNet `check_liveness` is intentionally NOT used here: it
        // rejected genuine faces and was non-fatal. The depth-relief gate is the
        // real anti-spoofing check.
        if self.config.authentication.liveness_enabled {
            match self.depth_relief_blocking(image, &face).await {
                Ok(Some(relief)) => {
                    let threshold = self.config.authentication.liveness_depth_threshold;
                    if relief < threshold {
                        tracing::warn!(
                            "Liveness REJECTED (depth gate): relief={:.4} < threshold={:.4} — treating as spoof",
                            relief, threshold
                        );
                        return Ok(None);
                    }
                    tracing::debug!(
                        "Liveness passed (depth gate): relief={:.4} >= threshold={:.4}",
                        relief,
                        threshold
                    );
                }
                Ok(None) => {
                    // Depth model not loaded. Fail-safe for a MAX-security gate:
                    // if the operator enabled liveness but the depth model is
                    // missing, reject rather than silently letting everything
                    // through.
                    tracing::error!(
                        "Liveness ENABLED but depth-PAD model not loaded — rejecting frame (fail-safe)"
                    );
                    return Ok(None);
                }
                Err(e) => {
                    // An inference error must not become a silent bypass.
                    tracing::error!(
                        "Liveness depth check errored — rejecting frame (fail-safe): {}",
                        e
                    );
                    return Ok(None);
                }
            }
        } else {
            tracing::debug!("Liveness disabled via config; skipping depth gate");
        }

        // Stage 3: Embedding — aligns via landmarks, then ArcFace 512-d.
        let embedding = self.embed_blocking(image, &face).await?;

        Ok(Some((face, embedding)))
    }

    /// Run ONLY the FATAL depth-liveness gate on an already-detected face and
    /// return the typed outcome (with the relief value). `None` = liveness is
    /// disabled by config. A missing model or inference error is
    /// [`LivenessOutcome::Unavailable`] (fail closed, never a silent pass).
    /// The auth path calls this ONCE per attempt instead of per-frame: depth is
    /// the slow stage on CPU (~0.5s), and running it on every one of the ~10
    /// auth frames overran the PAM screen-unlock deadline.
    pub async fn liveness_check(
        &self,
        image: &DynamicImage,
        face: &backend::Face,
    ) -> Option<LivenessOutcome> {
        if !self.config.authentication.liveness_enabled {
            return None;
        }
        Some(match self.depth_relief_blocking(image, face).await {
            Ok(Some(relief)) => {
                let t = self.config.authentication.liveness_depth_threshold;
                if relief < t {
                    tracing::warn!(
                        "Liveness REJECTED (depth gate): relief={:.4} < threshold={:.4} — treating as spoof",
                        relief, t
                    );
                    LivenessOutcome::Fail { relief }
                } else {
                    tracing::debug!(
                        "Liveness passed (depth gate): relief={:.4} >= {:.4}",
                        relief,
                        t
                    );
                    LivenessOutcome::Pass { relief }
                }
            }
            Ok(None) => {
                tracing::error!(
                    "Liveness ENABLED but depth-PAD model not loaded — reject (fail-safe)"
                );
                LivenessOutcome::Unavailable {
                    reason: "depth model not loaded".to_string(),
                }
            }
            Err(e) => {
                tracing::error!("Liveness depth check errored — reject (fail-safe): {}", e);
                LivenessOutcome::Unavailable {
                    reason: format!("depth inference error: {e}"),
                }
            }
        })
    }

    /// Boolean view of [`Self::liveness_check`]: `Ok(true)` = live or liveness
    /// disabled, `Ok(false)` = spoof / rejected / model missing.
    pub async fn liveness_gate(&self, image: &DynamicImage, face: &backend::Face) -> Result<bool> {
        Ok(self
            .liveness_check(image, face)
            .await
            .is_none_or(|outcome| outcome.is_pass()))
    }

    /// Fast detection only - no embedding extraction (for real-time preview)
    /// Returns face bounding box without embedding (~2x faster)
    pub async fn detect_only(&self, image: &DynamicImage) -> Result<Option<backend::Face>> {
        self.detect_blocking(image).await
    }

    pub fn models_loaded(&self) -> bool {
        self.backend.is_ready()
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Detect face in image (for preview/debugging)
    pub async fn detect_face(&self, image: &DynamicImage) -> Result<Option<backend::Face>> {
        self.detect_blocking(image).await
    }

    /// Extract embedding from detected face (for preview/debugging)
    pub async fn extract_embedding(
        &self,
        image: &DynamicImage,
        face: &backend::Face,
    ) -> Result<Vec<f32>> {
        self.embed_blocking(image, face).await
    }

    /// Run the liveness check on a detected face (for preview/debugging/tests).
    /// Returns `true` if the face is considered real (or if liveness is skipped).
    pub async fn check_liveness(&self, image: &DynamicImage, face: &backend::Face) -> Result<bool> {
        self.check_liveness_blocking(image, face).await
    }

    /// Warmup models by running dummy inference
    /// This preloads and compiles models before processing real frames
    pub async fn warmup(&self) -> Result<()> {
        info!("Warming up models (preloading/compiling)...");

        // Create a dummy 640x480 black image (typical camera resolution)
        let dummy_img = DynamicImage::new_rgb8(640, 480);

        // Run detection warmup
        info!("  Warming up detector...");
        let _ = self.backend.detect_face(&dummy_img).await;

        // Create dummy face for subsequent stages
        let dummy_face = backend::Face {
            bbox: (0.2, 0.2, 0.4, 0.5), // Normalized coords (x, y, w, h)
            confidence: 0.9,
            frame_dimensions: (640, 480),
            landmarks: None,
            other_faces: Vec::new(),
        };

        // Run liveness warmup
        info!("  Warming up liveness detector...");
        let _ = self.backend.check_liveness(&dummy_img, &dummy_face).await;

        // Run depth-PAD warmup (JIT-compiles the depth model on the GPU EP).
        info!("  Warming up depth-PAD model...");
        let _ = self.backend.depth_relief(&dummy_img, &dummy_face).await;

        // Run embedding warmup
        info!("  Warming up face recognizer...");
        let _ = self
            .backend
            .extract_embedding(&dummy_img, &dummy_face)
            .await;

        info!("✓ Model warmup complete");
        Ok(())
    }

    /// Synchronous face detection for use in spawn_blocking
    pub fn detect_face_sync(&self, image: &DynamicImage) -> Result<Option<backend::Face>> {
        // Since backend methods are async, we need to block on them
        // This is safe because we're already in a spawn_blocking context
        tokio::runtime::Handle::current().block_on(self.backend.detect_face(image))
    }

    /// Synchronous embedding extraction for use in spawn_blocking.
    ///
    /// Takes the full detected `Face` (including its 5-point landmarks) so the
    /// recognizer uses the aligned-crop path. Previously this discarded the
    /// landmarks (set `landmarks: None`), forcing the degraded unaligned
    /// bbox-crop fallback in the backend; carrying them through fixes that.
    pub fn extract_embedding_sync(
        &self,
        image: &DynamicImage,
        face: &backend::Face,
    ) -> Result<Vec<f32>> {
        tokio::runtime::Handle::current().block_on(self.backend.extract_embedding(image, face))
    }
}

/// Calculate cosine similarity
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
