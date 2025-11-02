mod backend;
#[cfg(feature = "backend-tract")]
mod tract_backend;
#[cfg(feature = "backend-ort")]
mod ort_backend;

pub use backend::{MLBackend, BackendType};
#[allow(dead_code)]
pub use backend::Face;
use anyhow::Result;
use doorman_shared::Config;
use image::DynamicImage;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

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
            BackendType::Tract => {
                #[cfg(feature = "backend-tract")]
                {
                    Arc::new(tract_backend::TractBackend::new(&models_dir)?)
                }
                #[cfg(not(feature = "backend-tract"))]
                {
                    return Err(anyhow::anyhow!("Tract backend not compiled. Build with --features backend-tract"));
                }
            }
            BackendType::OnnxRuntime => {
                #[cfg(feature = "backend-ort")]
                {
                    Arc::new(ort_backend::OrtBackend::new(&models_dir, config)?)
                }
                #[cfg(not(feature = "backend-ort"))]
                {
                    return Err(anyhow::anyhow!("ORT backend not compiled. Build with --features backend-ort"));
                }
            }
            BackendType::Candle => {
                warn!("Candle backend not yet implemented");
                return Err(anyhow::anyhow!("Candle backend not yet implemented"));
            }
        };
        
        info!("Using ML backend: {}", backend.name());
        
        Ok(Self {
            backend,
            config: config.clone(),
        })
    }
    
    pub fn dummy(config: &Config) -> Self {
        // For testing - create dummy backend
        #[cfg(feature = "backend-tract")]
        {
            let models_dir = PathBuf::from("/nonexistent");
            let backend = tract_backend::TractBackend::new(&models_dir)
                .unwrap_or_else(|_| panic!("Failed to create dummy backend"));
            
            Self {
                backend: Arc::new(backend),
                config: config.clone(),
            }
        }
        #[cfg(not(feature = "backend-tract"))]
        {
            panic!("No backend available for dummy. Compile with --features backend-tract");
        }
    }
    
    pub async fn process_frame(&self, image: &DynamicImage) -> Result<Option<Vec<f32>>> {
        let filter = match self.config.preprocessing.filter_type.as_str() {
            "nearest" => image::imageops::FilterType::Nearest,
            "triangle" => image::imageops::FilterType::Triangle,
            "catmullrom" => image::imageops::FilterType::CatmullRom,
            "gaussian" => image::imageops::FilterType::Gaussian,
            _ => image::imageops::FilterType::Lanczos3,
        };
        
        let small_img = image.resize_exact(
            self.config.preprocessing.image_width,
            self.config.preprocessing.image_height,
            filter,
        );
        
        // Stage 1: Detect
        let face = match self.backend.detect_face(&small_img).await? {
            Some(f) => f,
            None => return Ok(None),
        };
        
        // Stage 2: Liveness
        if !self.backend.check_liveness(&small_img, &face).await? {
            return Ok(None);
        }
        
        // Stage 3: Embedding
        let embedding = self.backend.extract_embedding(&small_img, &face).await?;
        
        Ok(Some(embedding))
    }
    
    pub fn models_loaded(&self) -> bool {
        self.backend.is_ready()
    }
    
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
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

