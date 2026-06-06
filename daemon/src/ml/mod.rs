mod backend;
mod model_config;

#[cfg(feature = "backend-ort")]
mod align;

#[cfg(feature = "backend-ort")]
mod ort_backend;

#[cfg(feature = "backend-ort")]
mod yunet_decoder;

#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use model_config::{DetectorConfig, LivenessConfig, ModelSet, RecognizerConfig};

use anyhow::Result;
#[allow(dead_code)]
pub use backend::Face;
pub use backend::{BackendType, MLBackend};
use doorman_shared::Config;
use image::{DynamicImage, GenericImageView};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

/// ML Pipeline with pluggable backend
pub struct MLPipeline {
    backend: Arc<dyn MLBackend>,
    config: Config,
}

impl MLPipeline {
    pub async fn new(config: &Config) -> Result<Self> {
        let models_dir = PathBuf::from(&config.ml.models_dir);

        // Select backend based on config
        let backend_type = BackendType::from_str(&config.ml.backend);

        info!("Initializing ML backend: {:?}", backend_type);

        let backend: Arc<dyn MLBackend> = match backend_type {
            BackendType::OnnxRuntime => {
                #[cfg(feature = "backend-ort")]
                {
                    Arc::new(ort_backend::OrtBackend::new(&models_dir, config)?)
                }
                #[cfg(not(feature = "backend-ort"))]
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
        #[cfg(feature = "backend-ort")]
        {
            let models_dir = PathBuf::from(&config.ml.models_dir);
            let backend = ort_backend::OrtBackend::new(&models_dir, config)
                .unwrap_or_else(|e| panic!("Failed to create dummy ORT backend: {}", e));

            Self {
                backend: Arc::new(backend),
                config: config.clone(),
            }
        }
        #[cfg(not(feature = "backend-ort"))]
        {
            let _ = config;
            panic!("No backend available for dummy. Compile with --features backend-ort");
        }
    }

    /// Full processing: detection + liveness + embedding (for recognition)
    pub async fn process_frame(&self, image: &DynamicImage) -> Result<Option<(backend::Face, Vec<f32>)>> {
        // Stage 1: Detect (on full-size image, detector will resize internally)
        let face = match self.backend.detect_face(image).await? {
            Some(f) => f,
            None => return Ok(None),
        };

        // Stage 2: Liveness (MiniFASNet anti-spoofing).
        //
        // Controlled by `authentication.liveness_enabled` (default true). The
        // check is NON-FATAL: a failure or error logs and is skipped so it can
        // never block recognition (the bundled MiniFASNet models are a
        // convenience deterrent, not high-security).
        if self.config.authentication.liveness_enabled {
            match self.backend.check_liveness(image, &face).await {
                Ok(true) => {}
                Ok(false) => {
                    tracing::warn!("Liveness check failed (non-fatal): proceeding with recognition");
                }
                Err(e) => {
                    tracing::warn!("Liveness check errored (non-fatal): {} — proceeding", e);
                }
            }
        } else {
            tracing::debug!("Liveness disabled via config; skipping");
        }

        // Stage 3: Embedding — aligns via landmarks, then ArcFace 512-d.
        let embedding = self.backend.extract_embedding(image, &face).await?;

        Ok(Some((face, embedding)))
    }

    /// Fast detection only - no embedding extraction (for real-time preview)
    /// Returns face bounding box without embedding (~2x faster)
    pub async fn detect_only(&self, image: &DynamicImage) -> Result<Option<backend::Face>> {
        self.backend.detect_face(image).await
    }

    pub fn models_loaded(&self) -> bool {
        self.backend.is_ready()
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Detect face in image (for preview/debugging)
    pub async fn detect_face(&self, image: &DynamicImage) -> Result<Option<backend::Face>> {
        self.backend.detect_face(image).await
    }

    /// Extract embedding from detected face (for preview/debugging)
    pub async fn extract_embedding(
        &self,
        image: &DynamicImage,
        face: &backend::Face,
    ) -> Result<Vec<f32>> {
        self.backend.extract_embedding(image, face).await
    }

    /// Run the liveness check on a detected face (for preview/debugging/tests).
    /// Returns `true` if the face is considered real (or if liveness is skipped).
    pub async fn check_liveness(
        &self,
        image: &DynamicImage,
        face: &backend::Face,
    ) -> Result<bool> {
        self.backend.check_liveness(image, face).await
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
        };
        
        // Run liveness warmup
        info!("  Warming up liveness detector...");
        let _ = self.backend.check_liveness(&dummy_img, &dummy_face).await;
        
        // Run embedding warmup
        info!("  Warming up face recognizer...");
        let _ = self.backend.extract_embedding(&dummy_img, &dummy_face).await;
        
        info!("✓ Model warmup complete");
        Ok(())
    }

    /// Synchronous face detection for use in spawn_blocking
    pub fn detect_face_sync(&self, image: &DynamicImage) -> Result<Option<backend::Face>> {
        // Since backend methods are async, we need to block on them
        // This is safe because we're already in a spawn_blocking context
        tokio::runtime::Handle::current().block_on(self.backend.detect_face(image))
    }

    /// Synchronous embedding extraction for use in spawn_blocking
    pub fn extract_embedding_sync(
        &self,
        image: &DynamicImage,
        bbox: &(f32, f32, f32, f32),
    ) -> Result<Vec<f32>> {
        let face = backend::Face {
            bbox: *bbox,
            confidence: 1.0, // Confidence not needed for embedding extraction
            frame_dimensions: image.dimensions(),
            landmarks: None,
        };
        tokio::runtime::Handle::current().block_on(self.backend.extract_embedding(image, &face))
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
