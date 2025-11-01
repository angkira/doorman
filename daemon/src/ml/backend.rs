use anyhow::Result;
use image::DynamicImage;
use async_trait::async_trait;

/// Face detection result
#[derive(Debug, Clone)]
pub struct Face {
    pub bbox: (f32, f32, f32, f32), // x, y, w, h
    pub confidence: f32,
}

/// Abstract ML backend trait - implements driver pattern
#[async_trait]
pub trait MLBackend: Send + Sync {
    /// Detect faces in image
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>>;
    
    /// Check if face is real (anti-spoofing)
    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool>;
    
    /// Extract face embedding (512-d vector)
    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>>;
    
    /// Check if backend is ready
    fn is_ready(&self) -> bool;
    
    /// Get backend name
    fn name(&self) -> &'static str;
}

/// Backend selection configuration
#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    /// ONNX Runtime (ort crate)
    OnnxRuntime,
    /// Tract (pure Rust)
    Tract,
    /// Candle (Hugging Face)
    Candle,
}

impl BackendType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "tract" => Self::Tract,
            "candle" => Self::Candle,
            "onnx" | "onnxruntime" | "ort" => Self::OnnxRuntime,
            _ => Self::Tract, // Default to tract
        }
    }
}

